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

    def record_decision_context(self, metrics: WindowMetrics, action: Action) -> None:
        """Called when we are about to execute an action; store context for later outcome."""
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
        helped = _evaluate_helped(
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
        self._context_before_action = None
        self._action_pending = None

    def cancel_pending(self) -> None:
        """If we decided not to execute (e.g. human-in-the-loop), clear pending."""
        self._context_before_action = None
        self._action_pending = None

    def get_recent_outcomes(self, n: int = 20) -> list[OutcomeRecord]:
        return self._outcomes[-n:]

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
                "learned from these outcome records (what helped, what hurt). "
                "Do not suggest code or config changes.\n\n" + "\n".join(lines)
            )
            try:
                config = genai.types.GenerationConfig(temperature=0.3, max_output_tokens=200)
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
    return {
        "success_rate": m.success_rate,
        "p95_latency_ms": m.p95_latency_ms,
        "retry_amplification": m.retry_amplification,
        "sample_count": m.sample_count,
    }


def _evaluate_helped(
    before: dict[str, Any],
    after: dict[str, Any],
    rollback_applied: bool,
) -> bool:
    """True if the action improved success rate or reduced latency (and no rollback)."""
    if rollback_applied:
        return False
    sr_improved = after.get("success_rate", 0) - before.get("success_rate", 0) >= HELPED_SUCCESS_IMPROVEMENT
    lat_before = before.get("p95_latency_ms") or 0
    lat_after = after.get("p95_latency_ms") or 0
    lat_reduced = (
        lat_before > 0
        and (lat_before - lat_after) / lat_before >= HELPED_LATENCY_REDUCTION
    )
    return sr_improved or lat_reduced
