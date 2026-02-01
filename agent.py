"""
Agent orchestration: Observe -> Reason -> Decide -> Act -> Learn loop.
Event-driven: evaluates metrics on every incoming event; reasoning+decision triggered by
uncertainty, confidence, or risk accumulation. Time-based debouncing prevents action spam.
"""
import time
from typing import Any, Callable, Iterator, Optional

from decision import decide
from executor import Executor
from learner import Learner
from models import ActionType, PaymentEvent, WindowMetrics
from observer import Observer
from reasoner import reason
from simulator import FailureMode, PaymentSimulator  # MULTI_MERCHANT_ESCALATION for demo human handover
from state_writer import (
    write_action,
    write_control_state,
    write_hypothesis,
    write_metrics,
    SYSTEM_MODE_COOLDOWN_ACTIVE,
    SYSTEM_MODE_DEGRADED,
    SYSTEM_MODE_HUMAN_APPROVAL_REQUIRED,
    SYSTEM_MODE_NORMAL,
    OUTCOME_BLOCKED,
    OUTCOME_EXECUTED,
    OUTCOME_ROLLED_BACK,
    OUTCOME_SKIPPED_COOLDOWN,
)

# --------------- Event-driven decision triggers (critical: when to reason -> decide -> act) ---------------
UNCERTAINTY_THRESHOLD = 0.5   # Trigger when uncertainty >= this (e.g. conflicting signals)
MIN_CONFIDENCE_TO_ACT = 0.6   # Trigger when confidence crosses this (possible action)
RISK_ACCUMULATION_SAME_CAUSE_COUNT = 2  # Same cause/target in last N recent_causes -> risk signal
DEBOUNCE_SEC = 3.0  # Prevent repeating the same decision within N seconds (store last decision timestamp)


class Agent:
    """
    Agentic payment operations manager. Runs the full loop:
    Observe (sliding window metrics) → Reason (hypothesis) → Decide (action) →
    Act (guarded execution / human-in-the-loop) → Learn (outcome memory).
    """

    def __init__(
        self,
        simulator: PaymentSimulator,
        observer: Observer,
        executor: Executor,
        learner: Learner,
        window_events_between_cycles: int = 50,
    ):
        self.simulator = simulator
        self.observer = observer
        self.executor = executor
        self.learner = learner
        self.window_events_between_cycles = window_events_between_cycles
        self._last_metrics: Optional[WindowMetrics] = None
        self._last_action_executed: bool = False
        self._pending_outcome: bool = False  # true after we executed; next window we learn
        self._cycle_count = 0

        # Real-time: decision context and debouncing
        self._recent_causes: list[dict[str, Any]] = []  # last N {cause, target} for risk accumulation
        self._max_recent_causes = 10
        self._last_executed_action_key: Optional[tuple[str, Optional[str]]] = None  # (action_type, target)
        self._last_executed_cycle: int = -1
        self._cooldown_until_cycle: Optional[int] = None
        self._cooldown_cycles = 2
        self._last_written_action_key: Optional[tuple[str, Optional[str], str]] = None  # (action_type, target, outcome)
        self._last_executed_trace: Any = None  # for writing rolled_back outcome when rollback occurs
        # Event-driven: time-based debounce (same decision within N seconds -> skip execution)
        self._last_decision_ts: float = 0.0
        self._last_decision_action_key: Optional[tuple[str, Optional[str]]] = None  # (action_type, target)
        # Explainability: last metrics/hypothesis for "what changed since last decision"
        self._last_decision_metrics_snapshot: Optional[dict[str, Any]] = None
        self._last_decision_hypothesis_cause: Optional[str] = None

        # Wire executor to simulator for reroute / retry / suppress
        def simulator_control(cmd: str, params: dict) -> None:
            if cmd == "reroute":
                # Simulate regression: brief latency bump so rollback can trigger (demo)
                self.simulator.trigger_post_action_latency_bump(num_events=80, multiplier=2.0)
            elif cmd == "retry_policy":
                pass  # Would apply to gateway
            elif cmd == "suppress":
                pass  # Would suppress heavy path
            elif cmd == "rollback":
                self.simulator.clear_failure_mode()

        self.executor.set_simulator_control(simulator_control)

    def observe(self, event: PaymentEvent) -> Optional[WindowMetrics]:
        """Ingest one event; return current window metrics if window is ready (batch mode)."""
        self.observer.ingest(event)
        if not self.observer.ready():
            return None
        metrics = self.observer.get_current_metrics()
        if metrics:
            self._last_metrics = metrics
        return metrics

    def get_partial_metrics(self) -> Optional[WindowMetrics]:
        """Partial metrics on current buffer (every event). Enables event-driven evaluation."""
        return self.observer.get_partial_metrics()

    def _risk_accumulation_signal(self) -> bool:
        """True when same cause/target repeated in recent_causes (risk accumulation -> trigger decide)."""
        if len(self._recent_causes) < RISK_ACCUMULATION_SAME_CAUSE_COUNT:
            return False
        recent = self._recent_causes[-5:]
        from collections import Counter
        keys = [(c.get("cause"), c.get("target")) for c in recent if isinstance(c, dict)]
        if not keys:
            return False
        counts = Counter(keys)
        return max(counts.values()) >= RISK_ACCUMULATION_SAME_CAUSE_COUNT

    def _should_trigger_reason_decide(self, hypothesis: Any) -> bool:
        """
        Decision trigger: reason -> decide -> act when uncertainty increases, confidence crosses threshold, or risk accumulation.
        Called on every event after reason(); gates whether we run decide and act.
        """
        if hypothesis.uncertainty >= UNCERTAINTY_THRESHOLD:
            return True  # Uncertainty increased -> evaluate (may still no-op)
        if hypothesis.confidence >= MIN_CONFIDENCE_TO_ACT:
            return True  # Confidence crossed threshold -> possible action
        if self._risk_accumulation_signal():
            return True  # Same cause repeating -> risk accumulation
        return False

    def run_cycle(self, metrics: WindowMetrics, hypothesis: Optional[Any] = None) -> None:
        """
        One full cycle: Reason (if hypothesis not provided) -> Decide -> Act -> (outcome later in Learn).
        Real-time: context for risk accumulation, time-based debounce, only new decisions appended to timeline.
        """
        self._cycle_count += 1
        self._last_action_executed = False
        ts = time.time()

        if hypothesis is None:
            hypothesis = reason(metrics)
        write_hypothesis(
            cause=hypothesis.cause,
            confidence=hypothesis.confidence,
            evidence=hypothesis.evidence,
            source=hypothesis.source,
            ts=ts,
            uncertainty=getattr(hypothesis, "uncertainty", None),
        )

        # Expire cooldown
        if self._cooldown_until_cycle is not None and self._cycle_count > self._cooldown_until_cycle:
            self._cooldown_until_cycle = None

        # Build decision context for risk accumulation and forced handover
        pending = self.executor.get_escalation_state()
        context = {
            "recent_causes": list(self._recent_causes),
            "rollback_count": self.executor.get_rollback_count(),
            "pending_approval": pending if (pending and pending.get("active")) else None,
        }

        # Check for approval on pending actions (polls every cycle)
        approved, approval_msg, approved_action = self.executor.check_and_apply_approval(metrics)
        if approved:
            print(f"[Human Approval] {approval_msg}")
            # If approved and executed, we treat it as an action taken this cycle
            if "APPROVED" in approval_msg and approved_action:
                self._last_action_executed = True
                self._pending_outcome = True
                self._last_decision_ts = ts
                # CRITICAL: Update decision key so cooldown logic sees this as 'just executed'
                self._last_executed_action_key = (approved_action.action_type, approved_action.target)
                self._last_executed_cycle = self._cycle_count 
                # We don't have the original trace easily here, but executor logged it.
                # For learning, we might miss the exact context match unless we stored it.
                # Simplification: we'll skip immediate learner record update for this async path
                # or rely on the fact that the next window will show improvement.
            else:
                # Rejected
                self.learner.cancel_pending()
            
            # Resume normal flow next cycle; for now, we just proceed
            self._write_control_state(metrics, ts, None, self._last_action_executed, False)
            return

        # Persist current metrics for dashboard (real-time KPIs and merchant health)

        # Persist current metrics for dashboard (real-time KPIs and merchant health)
        write_metrics(
            success_rate=metrics.success_rate,
            p95_latency_ms=metrics.p95_latency_ms,
            success_rate_by_issuer=metrics.success_rate_by_issuer,
            retry_amplification=metrics.retry_amplification,
            sample_count=metrics.sample_count,
            window_id=metrics.window_id,
            ts=ts,
            average_estimated_cost=getattr(metrics, "average_estimated_cost", None),
            attempt_amplification=getattr(metrics, "attempt_amplification", None),
            success_rate_by_merchant=getattr(metrics, "success_rate_by_merchant", None),
            avg_cost_by_merchant=getattr(metrics, "avg_cost_by_merchant", None),
            attempt_amplification_by_merchant=getattr(metrics, "attempt_amplification_by_merchant", None),
        )

        # ---------------- DEADLOCK FIX: Explicit WAITING_FOR_HUMAN state ----------------
        # If an escalation is pending, we FREEZE the decision loop.
        # We continued to observe (update metrics) above, but we do NOT call decide().
        if pending and pending.get("active"):
            if self._cycle_count % 5 == 0:
                print(f"[Agent] WAITING_FOR_HUMAN. Pending action: {pending.get('action_type')} target={pending.get('target')}")
            
            # Ensure we don't look like we're "running" normally in control state
            self._write_control_state(metrics, ts, None, False, False)
            return

        print(f"[Hypothesis] cause={hypothesis.cause} confidence={hypothesis.confidence:.2f} uncertainty={getattr(hypothesis, 'uncertainty', 0):.2f} source={hypothesis.source}")
        print(f"  evidence: {hypothesis.evidence}")

        # Decide: deterministic action (or NO_OP) with context
        trace = decide(metrics, hypothesis, context)
        print(f"[Decision] action={trace.action.action_type} target={trace.action.target} risk={trace.risk_score:.2f}")
        print(f"  reasoning: {trace.reasoning}")

        # Track recent causes for next cycle
        self._recent_causes.append({
            "cause": hypothesis.cause,
            "target": trace.action.target,
        })
        if len(self._recent_causes) > self._max_recent_causes:
            self._recent_causes = self._recent_causes[-self._max_recent_causes:]

        guardrails_triggered = []
        if trace.action.params.get("requires_human_approval"):
            guardrails_triggered.append("Human approval required")

        if trace.action.action_type == ActionType.NO_OP:
            outcome = "executed"  # no_op is always "taken"
            current_key = ("no_op", None, outcome)
            append = current_key != self._last_written_action_key
            write_action(
                action_type="no_op",
                target=None,
                risk_score=0.0,
                reason=trace.action.reason,
                reasoning=trace.reasoning,
                executed=False,
                message="No intervention needed",
                ts=ts,
                outcome=outcome,
                guardrails_triggered=guardrails_triggered,
                append_to_history=append,
            )
            if append:
                self._last_written_action_key = current_key
            self._write_control_state(metrics, ts, None, False, False)
            return

        # Debouncing: same action+target within cooldown (cycles) OR within N seconds (time-based)
        # FIX: Cooldown should only apply if we actually EXECUTED recently.
        # If the last decision was BLOCKED (pending), do NOT treat it as a cooldown trigger for the next identical proposal?
        # Actually, if blocked, we handled it in the thrashing check above.
        # But if we rejected it? Then maybe cooldown? Or immediate retry?
        # Let's rely on executed status.
        
        action_key = (trace.action.action_type, trace.action.target)
        now = time.time()
        
        # Only check cooldown if the last action was actually executed?
        # Standard logic: if we executed X recently, don't do X again.
        
        in_cycle_cooldown = (
            self._last_executed_action_key == action_key
            and (self._cycle_count - self._last_executed_cycle) <= self._cooldown_cycles
        )
        in_time_cooldown = (
            self._last_decision_action_key == action_key
            and (now - self._last_decision_ts) < DEBOUNCE_SEC
            and self._last_action_executed # Only debounce if actually executed?
        )
        # If we blocked previous one, last_action_executed is False for that decision.
        
        if in_cycle_cooldown or in_time_cooldown:
            self._cooldown_until_cycle = self._cycle_count + self._cooldown_cycles
            self.learner.cancel_pending()
            outcome = OUTCOME_SKIPPED_COOLDOWN
            msg = f"[COOLDOWN] Skipped: {trace.action.action_type} target={trace.action.target} (same action within cooldown)"
            print(msg)
            current_key = (trace.action.action_type, trace.action.target, outcome)
            append = current_key != self._last_written_action_key
            write_action(
                action_type=trace.action.action_type,
                target=trace.action.target,
                risk_score=trace.action.risk_score,
                reason=trace.action.reason,
                reasoning=trace.reasoning,
                executed=False,
                message=msg,
                ts=ts,
                outcome=outcome,
                guardrails_triggered=["Cooldown active"],
                append_to_history=append,
            )
            if append:
                self._last_written_action_key = current_key
            self._write_control_state(metrics, ts, None, False, True)
            return

        # Act: execute only if low risk; else human-in-the-loop
        executed, msg = self.executor.execute(trace, metrics)
        
        # ---------------- FIXED COOLDOWN LOGIC ----------------
        # 1. NO_OP never triggers cooldown.
        # 2. Blocked actions never trigger cooldown.
        # 3. Only EXECUTED actions trigger cooldown.
        
        outcome = "executed" if executed else "blocked"
        if trace.action.action_type == ActionType.NO_OP:
             outcome = "executed" # Logic consistency: NO_OP is "successfully done"

        # Update state for cooldowns
        if executed and trace.action.action_type != ActionType.NO_OP:
            self._last_action_executed = True
            self._last_executed_action_key = action_key
            self._last_executed_cycle = self._cycle_count
            self._last_decision_ts = now
            self._last_decision_action_key = action_key
            # For learner:
            self._pending_outcome = True
            self._baseline_metrics = metrics
        elif not executed:
            # Blocked or failed. Do NOT update _last_executed_action_key.
            self._last_action_executed = False
            # Pass on timestamp updates so we don't start cooldown
        else:
             # NO_OP executed
             self._last_action_executed = True

        # Explainability: what changed since last decision, why action now, why human approval
        what_changed = self._explain_what_changed(metrics)
        why_now = self._explain_why_action_now(hypothesis)
        why_human = "; ".join(guardrails_triggered) if guardrails_triggered else None
        self._last_decision_metrics_snapshot = {
            "success_rate": metrics.success_rate,
            "p95_latency_ms": metrics.p95_latency_ms,
            "sample_count": metrics.sample_count,
        }
        self._last_decision_hypothesis_cause = hypothesis.cause

        current_key = (trace.action.action_type, trace.action.target, outcome)
        append = current_key != self._last_written_action_key
        write_action(
            action_type=trace.action.action_type,
            target=trace.action.target,
            risk_score=trace.action.risk_score,
            reason=trace.action.reason,
            reasoning=trace.reasoning,
            executed=executed,
            message=msg,
            ts=ts,
            outcome=outcome,
            guardrails_triggered=guardrails_triggered,
            append_to_history=append,
            what_changed_since_last=what_changed,
            why_action_now=why_now,
            why_human_approval=why_human,
        )
        if append:
            self._last_written_action_key = current_key
        print(f"[Action] {msg}")

        if executed:
            self._last_action_executed = True
            self._last_executed_trace = trace
            self.learner.record_decision_context(metrics, trace.action)
            if trace.action.action_type == "reroute" and trace.action.target:
                self.simulator.clear_failure_mode()
        else:
            self.learner.cancel_pending()
            self._last_executed_trace = None

        self._write_control_state(metrics, ts, trace, executed, False)

    def _explain_what_changed(self, metrics: WindowMetrics) -> Optional[str]:
        """Real-time narration: what changed since last decision (for explainability)."""
        if self._last_decision_metrics_snapshot is None:
            return "First decision in this run."
        prev = self._last_decision_metrics_snapshot
        parts = []
        if abs(metrics.success_rate - prev.get("success_rate", 0)) >= 0.02:
            parts.append(f"success_rate {prev.get('success_rate', 0):.1%} -> {metrics.success_rate:.1%}")
        if abs(metrics.p95_latency_ms - prev.get("p95_latency_ms", 0)) >= 20:
            parts.append(f"p95_latency {prev.get('p95_latency_ms', 0):.0f} -> {metrics.p95_latency_ms:.0f} ms")
        if metrics.sample_count != prev.get("sample_count"):
            parts.append(f"sample_count {prev.get('sample_count', 0)} -> {metrics.sample_count}")
        return "; ".join(parts) if parts else "No significant metric change since last decision."

    def _explain_why_action_now(self, hypothesis: Any) -> str:
        """Real-time narration: why action is proposed now (trigger reason)."""
        if hypothesis.confidence >= MIN_CONFIDENCE_TO_ACT:
            return f"Confidence crossed threshold ({hypothesis.confidence:.2f} >= {MIN_CONFIDENCE_TO_ACT})."
        if hypothesis.uncertainty >= UNCERTAINTY_THRESHOLD:
            return f"Uncertainty high ({hypothesis.uncertainty:.2f}); evaluation triggered."
        if self._risk_accumulation_signal():
            return "Risk accumulation: same cause/target repeating."
        return "Trigger: confidence or uncertainty or risk signal."

    def _write_control_state(
        self,
        metrics: WindowMetrics,
        ts: float,
        trace: Any,
        executed: bool,
        cooldown_active: bool,
    ) -> None:
        """Write control_state.json for real-time dashboard (system mode, escalation, learning)."""
        escalation = self.executor.get_escalation_state()
        learning = self.learner.get_learning_state()

        if escalation and escalation.get("active"):
            system_mode = SYSTEM_MODE_HUMAN_APPROVAL_REQUIRED
        elif cooldown_active or (self._cooldown_until_cycle is not None and self._cycle_count <= self._cooldown_until_cycle):
            system_mode = SYSTEM_MODE_COOLDOWN_ACTIVE
        elif metrics.success_rate < 0.78:
            system_mode = SYSTEM_MODE_DEGRADED
        else:
            system_mode = SYSTEM_MODE_NORMAL

        cooldown_until_ts = None
        if self._cooldown_until_cycle is not None:
            cooldown_until_ts = ts + (self._cooldown_until_cycle - self._cycle_count) * 2.0  # approximate sec per cycle

        write_control_state(
            system_mode=system_mode,
            ts=ts,
            cooldown_until_ts=cooldown_until_ts,
            escalation=escalation,
            learning=learning,
        )

    def check_rollback_and_learn(self, metrics: WindowMetrics) -> None:
        """Check for rollback; record outcome for learning; append ROLLED_BACK to timeline."""
        rollbacks = self.executor.check_rollback(metrics)
        for rb in rollbacks:
            print(f"[Rollback] {rb}")

        if rollbacks and self._last_executed_trace is not None:
            trace = self._last_executed_trace
            write_action(
                action_type=trace.action.action_type,
                target=trace.action.target,
                risk_score=trace.action.risk_score,
                reason=trace.action.reason,
                reasoning="Rolled back due to metric regression",
                executed=False,
                message=f"Rollback: {trace.action.action_type} target={trace.action.target}",
                ts=time.time(),
                outcome=OUTCOME_ROLLED_BACK,
                guardrails_triggered=["Rollback applied"],
                append_to_history=True,
            )
            self._last_written_action_key = (trace.action.action_type, trace.action.target, OUTCOME_ROLLED_BACK)
            self._last_executed_trace = None

        if self._last_action_executed:
            self.learner.record_outcome(
                metrics,
                rollback_applied=len(rollbacks) > 0,
            )

    def run(
        self,
        event_queue: Any,  # queue.Queue
        # Removed legacy stream args since thread handles them
    ) -> None:
        """
        Event-driven loop: consume from queue. Non-blocking observation.
        """
        print("Agent started. Observe -> Reason -> Decide -> Act -> Learn (event-driven)")
        print("-" * 60)
        event_count = 0

        while True:
            try:
                # Poll queue with timeout to allow periodic tasks even if no events
                event = event_queue.get(timeout=0.2)
                if event is None:
                    break # Signal from thread
            except:
                # Empty queue, skip this iter
                # Check approval even if no events coming in?
                # Actually, metrics update on events. If no events, metrics stale.
                # However, human approval might come in.
                approved, approval_msg, approved_action = self.executor.check_and_apply_approval(self._last_metrics)
                if approved:
                    print(f"[Human Approval async] {approval_msg}")
                    if "APPROVED" in approval_msg and approved_action:
                        self._last_action_executed = True
                        self._pending_outcome = True
                        self._last_decision_ts = time.time()
                        # CRITICAL: Update decision key so cooldown logic sees this as 'just executed'
                        self._last_executed_action_key = (approved_action.action_type, approved_action.target)
                        self._last_executed_cycle = self._cycle_count # Approximate
                    else:
                        self.learner.cancel_pending()
                continue

            event_count += 1


            # Legacy injection logic moved to run.py thread
            # if event_count == failure_injection_after_events: ...


            # --------------- Event-driven: ingest and get partial metrics on every event ---------------
            self.observer.ingest(event)
            metrics = self.get_partial_metrics()
            if metrics is None:
                continue

            # Persist current metrics so dashboard updates live (rolling; append in state_writer)
            write_metrics(
                success_rate=metrics.success_rate,
                p95_latency_ms=metrics.p95_latency_ms,
                success_rate_by_issuer=metrics.success_rate_by_issuer,
                retry_amplification=metrics.retry_amplification,
                sample_count=metrics.sample_count,
                window_id=metrics.window_id,
                ts=time.time(),
                average_estimated_cost=getattr(metrics, "average_estimated_cost", None),
                attempt_amplification=getattr(metrics, "attempt_amplification", None),
                success_rate_by_merchant=getattr(metrics, "success_rate_by_merchant", None),
                avg_cost_by_merchant=getattr(metrics, "avg_cost_by_merchant", None),
                attempt_amplification_by_merchant=getattr(metrics, "attempt_amplification_by_merchant", None),
            )

            # Reason on every event (hypothesis + uncertainty for trigger and dashboard)
            hypothesis = reason(metrics)
            write_hypothesis(
                cause=hypothesis.cause,
                confidence=hypothesis.confidence,
                evidence=hypothesis.evidence,
                source=hypothesis.source,
                ts=time.time(),
                uncertainty=getattr(hypothesis, "uncertainty", None),
            )

            # On next window after an action, check rollback and record outcome
            if self._pending_outcome:
                self.check_rollback_and_learn(metrics)
                self._pending_outcome = False

            # --------------- Decision trigger: uncertainty, confidence, or risk accumulation ---------------
            if self._should_trigger_reason_decide(hypothesis):
                self.run_cycle(metrics, hypothesis)
                if self._last_action_executed:
                    self._pending_outcome = True

        # Final learning summary
        summary = self.learner.summarize_learning_heuristic()
        print("-" * 60)
        print("[Learning] " + summary)
        llm_summary = self.learner.summarize_learning_llm()
        if llm_summary:
            print("[Learning LLM] " + llm_summary)
        print("Agent run complete.")


def create_agent(
    window_size: int = 200,
    window_advance: int = 50,
    cycle_interval_events: int = 50,
) -> Agent:
    """Factory: simulator, observer, executor, learner, and agent."""
    sim = PaymentSimulator(seed=42, base_success_rate=0.92)
    obs = Observer(window_size=window_size, window_advance_events=window_advance)
    executor = Executor()
    learner = Learner()
    return Agent(
        simulator=sim,
        observer=obs,
        executor=executor,
        learner=learner,
        window_events_between_cycles=cycle_interval_events,
    )
