"""
Guarded action executor. Executes only low-risk actions autonomously;
high-risk actions escalate to human-in-the-loop (logged, not auto-applied).
Supports reroute, retry_policy, suppress. Automatic rollback if metrics degrade.
"""
import time
from typing import Any, Callable, Optional

from models import Action, ActionType, DecisionTrace, WindowMetrics


# Risk above this: do not auto-execute; log for human approval
AUTO_EXECUTE_RISK_THRESHOLD = 0.45
# After executing, we compare metrics in next window; rollback if success rate drops by this much
ROLLBACK_SUCCESS_RATE_DROP = 0.08
# Or if p95 latency increases by this fraction
ROLLBACK_LATENCY_INCREASE_FRACTION = 0.25


class Executor:
    """
    Executes actions from the decision engine with guardrails.
    Tracks active interventions for rollback when metrics regress.
    """

    def __init__(self):
        self._active_actions: list[tuple[DecisionTrace, dict[str, Any]]] = []
        self._baseline_metrics: Optional[WindowMetrics] = None
        self._rollback_log: list[str] = []
        self._simulator_control: Optional[Callable[[str, Any], None]] = None

    def set_simulator_control(self, fn: Callable[[str, Any], None]) -> None:
        """Inject callback to control simulator (e.g. clear failure mode, adjust params)."""
        self._simulator_control = fn

    def execute(
        self,
        trace: DecisionTrace,
        current_metrics: WindowMetrics,
    ) -> tuple[bool, str]:
        """
        Execute the action in trace if risk is below threshold; else log for human.
        Returns (executed: bool, message: str).
        """
        action = trace.action
        if action.action_type == ActionType.NO_OP:
            return False, "NO_OP"

        if action.risk_score >= AUTO_EXECUTE_RISK_THRESHOLD:
            msg = (
                f"[HUMAN-IN-THE-LOOP] High-risk action not auto-executed: "
                f"{action.action_type} target={action.target} risk={action.risk_score:.2f}"
            )
            return False, msg

        # Store baseline for rollback check
        if not self._active_actions:
            self._baseline_metrics = current_metrics

        # Apply action via simulator control (or in-memory state for demo)
        applied = self._apply_action(trace)
        if applied:
            self._active_actions.append((trace, {"applied_at": time.time()}))
        return True, f"Executed: {action.action_type} target={action.target}"

    def _apply_action(self, trace: DecisionTrace) -> bool:
        """Apply action to simulator or internal state."""
        action = trace.action
        if self._simulator_control:
            if action.action_type == ActionType.REROUTE and action.target:
                self._simulator_control("reroute", {"issuer": action.target, **action.params})
            elif action.action_type == ActionType.RETRY_POLICY:
                self._simulator_control("retry_policy", action.params)
            elif action.action_type == ActionType.SUPPRESS:
                self._simulator_control("suppress", action.params)
        return True

    def check_rollback(self, current_metrics: WindowMetrics) -> list[str]:
        """
        If we have active actions and baseline metrics, check whether
        performance regressed; if so, rollback and return list of rollback messages.
        """
        if not self._active_actions or self._baseline_metrics is None:
            return []

        base = self._baseline_metrics
        cur = current_metrics
        rollbacks: list[str] = []

        success_dropped = base.success_rate - cur.success_rate >= ROLLBACK_SUCCESS_RATE_DROP
        latency_increased = (
            base.p95_latency_ms > 0
            and (cur.p95_latency_ms - base.p95_latency_ms) / base.p95_latency_ms
            >= ROLLBACK_LATENCY_INCREASE_FRACTION
        )

        if success_dropped or latency_increased:
            for trace, _ in self._active_actions:
                self._rollback_action(trace)
                msg = (
                    f"Rollback: {trace.action.action_type} (success_drop={success_dropped}, "
                    f"latency_inc={latency_increased})"
                )
                rollbacks.append(msg)
                self._rollback_log.append(msg)
            self._active_actions.clear()
            self._baseline_metrics = None

        return rollbacks

    def _rollback_action(self, trace: DecisionTrace) -> None:
        """Revert a single action (e.g. clear reroute/suppress via simulator)."""
        if self._simulator_control:
            self._simulator_control("rollback", {"action_type": trace.action.action_type, "target": trace.action.target})

    def get_active_actions(self) -> list[DecisionTrace]:
        return [t for t, _ in self._active_actions]

    def get_rollback_log(self) -> list[str]:
        return self._rollback_log.copy()
