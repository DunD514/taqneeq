"""
Learning loop: store context → action → outcome tuples;
evaluate whether actions helped or hurt.
LLM used only for post-hoc learning summary, not for applying changes.
"""
import os
from typing import Any, Optional

from models import Action, OutcomeRecord, WindowMetrics


# Thresholds to decide "helped": success rate improved by this much
HELPED_SUCCESS_IMPROVEMENT = 0.03

# Or latency reduced by this fraction
HELPED_LATENCY_REDUCTION = 0.15

# NEW (Plan A): cost / retry guardrails
HARMFUL_COST_INCREASE = 0.04
HARMFUL_ATTEMPT_AMPLIFICATION = 1.6

# Max records to keep in memory
MAX_OUTCOME_RECORDS = 500


class Learner:
    """
    Outcome memory and learning. Records (context, action, outcome),
    evaluates helped/hurt, and can produce LLM-summarized learning (optional).
    """

    def __init__(self):
        self._outcomes: list[OutcomeRecord] = []
        self._context_before_action: Optional[dict[str, Any]] = None
        self._action_pending: Optional[Action] = None

        # NEW: internal action effectiveness tracking
        self._action_stats: dict[str, dict[str, int]] = {}

    def record_decision_context(self, metrics: WindowMetrics, action: Action) -> None:
        """Called when we are about to execute an action; store context for later outcome."""
        if action.action_type == "no_op":
            return
        self._context_before_action = _metrics_snapshot(metrics)
        self._action_pending = action

    def record_outcome(
        self,
        metrics_after: WindowMetrics,
        rollback_applied: bool = False,
    ) -> None:
        """
        Called after an action had time to take effect (e.g. next window).
        Evaluates helped/hurt and appends OutcomeRecord.
        """
        if self._context_before_action is None or self._action_pending is None:
            return

        outcome_metrics = _metrics_snapshot(metrics_after)

        helped, learning_score = _evaluate_helped(
            self._context_before_action,
            outcome_metrics,
            rollback_applied,
        )

        record = OutcomeRecord(
            context_snapshot=self._context_before_action,
            action=self._action_pending,
            outcome_metrics=outcome_metrics,
            helped=helped,
            rollback_applied=rollback_applied,
        )

        self._outcomes.append(record)
        if len(self._outcomes) > MAX_OUTCOME_RECORDS:
            self._outcomes = self._outcomes[-MAX_OUTCOME_RECORDS:]

        # ---------------- NEW: track action effectiveness ----------------
        a_type = self._action_pending.action_type
        if a_type not in self._action_stats:
            self._action_stats[a_type] = {"helped": 0, "hurt": 0, "neutral": 0}

        if learning_score > 0:
            self._action_stats[a_type]["helped"] += 1
        elif learning_score < 0:
            self._action_stats[a_type]["hurt"] += 1
        else:
            self._action_stats[a_type]["neutral"] += 1

        self._context_before_action = None
        self._action_pending = None

    def cancel_pending(self) -> None:
        """If we decided not to execute (e.g. human-in-the-loop), clear pending."""
        self._context_before_action = None
        self._action_pending = None

    def get_recent_outcomes(self, n: int = 20) -> list[OutcomeRecord]:
        return self._outcomes[-n:]

    def get_action_effectiveness(self) -> dict[str, dict[str, int]]:
        """
        NEW: Returns how often each action type helped, hurt, or was neutral.
        Useful for dashboards or judge demos.
        """
        return self._action_stats.copy()

    def get_learning_state(self) -> dict[str, Any]:
        """
        Real-time learning state for dashboard: helped/hurt/neutral counts and recent rollbacks.
        Stream-aware: reflects current outcome totals and last rollbacks.
        """
        recent = self.get_recent_outcomes(20)
        helped = sum(1 for r in recent if r.helped and not r.rollback_applied)
        hurt = sum(1 for r in recent if r.rollback_applied)
        neutral = len(recent) - helped - hurt
        rollbacks = [f"{r.action.action_type} target={r.action.target}" for r in recent if r.rollback_applied]
        return {
            "helped": helped,
            "hurt": hurt,
            "neutral": neutral,
            "recent_rollbacks": rollbacks[-5:],  # last 5 rollbacks
            "action_effectiveness": self._action_stats.copy(),
        }

    def summarize_learning_llm(self) -> Optional[str]:
        """
        Optional: use Gemini to summarize recent outcomes for human review.
        Does not apply any changes; returns a string summary or None.
        """
        recent = self.get_recent_outcomes(10)
        if not recent:
            return None

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")

            lines = []
            for r in recent:
                lines.append(
                    f"- Action: {r.action.action_type} target={r.action.target} "
                    f"helped={r.helped} rollback={r.rollback_applied}"
                )

            prompt = (
                "Summarize in 2-3 sentences what the payment operations agent "
                "learned from these outcome records (what helped, what hurt, "
                "and which actions appear risky). "
                "Do not suggest code or config changes.\n\n" + "\n".join(lines)
            )

            try:
                config = genai.types.GenerationConfig(
                    temperature=0.3, max_output_tokens=200
                )
            except (AttributeError, TypeError):
                config = {"temperature": 0.3, "max_output_tokens": 200}

            response = model.generate_content(prompt, generation_config=config)
            content = getattr(response, "text", None) or ""
            if not content and response.candidates:
                parts = response.candidates[0].content.parts
                content = (parts[0].text if parts else "") or ""
            return (content or "").strip() or None

        except Exception:
            return None

    def summarize_learning_heuristic(self) -> str:
        """Deterministic summary of recent outcomes (no LLM)."""
        recent = self.get_recent_outcomes(10)
        if not recent:
            return "No outcomes recorded yet."

        helped = sum(1 for r in recent if r.helped and not r.rollback_applied)
        hurt = sum(1 for r in recent if r.rollback_applied)
        neutral = len(recent) - helped - hurt

        return (
            f"Recent outcomes: {helped} helped, {hurt} rollbacks, {neutral} neutral. "
            "Use this to refine future decisions."
        )


def _metrics_snapshot(m: WindowMetrics) -> dict[str, Any]:
    """
    Snapshot metrics for learning.
    Safe: ignores missing attributes.
    """
    return {
        "success_rate": m.success_rate,
        "p95_latency_ms": m.p95_latency_ms,
        "retry_amplification": m.retry_amplification,
        "sample_count": m.sample_count,
        # Optional Plan A fields
        "average_estimated_cost": getattr(m, "average_estimated_cost", None),
        "attempt_amplification": getattr(m, "attempt_amplification", None),
    }


def _evaluate_helped(
    before: dict[str, Any],
    after: dict[str, Any],
    rollback_applied: bool,
) -> tuple[bool, float]:
    """
    Returns (helped, learning_score).
    learning_score: +1 helpful, 0 neutral, -1 harmful
    """
    if rollback_applied:
        return False, -1.0

    sr_before = before.get("success_rate", 0)
    sr_after = after.get("success_rate", 0)
    lat_before = before.get("p95_latency_ms") or 0
    lat_after = after.get("p95_latency_ms") or 0

    sr_improved = sr_after - sr_before >= HELPED_SUCCESS_IMPROVEMENT
    lat_reduced = (
        lat_before > 0
        and (lat_before - lat_after) / lat_before >= HELPED_LATENCY_REDUCTION
    )

    # NEW: cost & retry harm detection
    cost_after = after.get("average_estimated_cost")
    attempt_after = after.get("attempt_amplification")

    if cost_after is not None and cost_after >= HARMFUL_COST_INCREASE:
        return False, -1.0

    if attempt_after is not None and attempt_after >= HARMFUL_ATTEMPT_AMPLIFICATION:
        return False, -1.0


    if sr_improved or lat_reduced:
        return True, +1.0

    return False, 0.0

    def record_human_feedback(self, action: Action, approved: bool) -> None:
        """Record explicit human feedback signal."""
        a_type = action.action_type
        if a_type not in self._action_stats:
            self._action_stats[a_type] = {"helped": 0, "hurt": 0, "neutral": 0}
            
        if approved:
            # Approval is a positive signal (validated the agent's proposal)
            self._action_stats[a_type]["helped"] += 1
        else:
            # Rejection is a negative/neutral signal (agent was wrong to propose)
            self._action_stats[a_type]["neutral"] += 1

