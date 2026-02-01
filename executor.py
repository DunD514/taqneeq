"""
Guarded action executor. Executes only low-risk actions autonomously;
high-risk actions escalate to human-in-the-loop (logged, not auto-applied).
Supports reroute, retry_policy, suppress. Automatic rollback if metrics degrade.
"""
import time
from typing import Any, Callable, Optional

from models import Action, ActionType, DecisionTrace, WindowMetrics, PendingApproval
from state_writer import write_pending_approval, read_pending_approval


# Risk above this: do not auto-execute; log for human approval
AUTO_EXECUTE_RISK_THRESHOLD = 0.45

# After executing, we compare metrics in next window; rollback if success rate drops by this much
ROLLBACK_SUCCESS_RATE_DROP = 0.08

# Or if p95 latency increases by this fraction
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
        self._execution_log: list[str] = []
        self._simulator_control: Optional[Callable[[str, Any], None]] = None
        self._pending_escalation: Optional[dict[str, Any]] = None
        # Initialize validation of any existing pending state
        self._sync_pending_state()

    def _sync_pending_state(self) -> None:
        """Sync internal state with persisted pending state on startup."""
        pending = read_pending_approval()
        if pending and pending.get("status") == "pending":
            self._pending_escalation = pending

    def set_simulator_control(self, fn: Callable[[str, Any], None]) -> None:
        """Inject callback to control simulator (e.g. clear failure mode, adjust params)."""
        self._simulator_control = fn

    def execute(
        self,
        trace: DecisionTrace,
        current_metrics: WindowMetrics,
    ) -> tuple[bool, str]:
        """
        Execute the action in trace if allowed.
        Returns (executed: bool, message: str).
        """
        action = trace.action

        if action.action_type == ActionType.NO_OP:
            return True, "NO_OP" # NO_OP is 'executed' successfully (by doing nothing)

        # ---------------- EXPLICIT human-approval boundary (Plan A) ----------------
        if action.params.get("requires_human_approval") is True:
            msg = (
                f"[HUMAN-APPROVAL-REQUIRED] Action requires human approval: "
                f"{action.action_type} target={action.target} "
                f"(risk={action.risk_score:.2f})"
            )
            self._execution_log.append(msg)
            self._set_pending_escalation(
                action,
                "Human approval required (cost/retry guardrail)",
            )
            return False, msg

        # ---------------- Existing risk-based guard ----------------
        if action.risk_score >= AUTO_EXECUTE_RISK_THRESHOLD:
            msg = (
                f"[HUMAN-IN-THE-LOOP] High-risk action not auto-executed: "
                f"{action.action_type} target={action.target} "
                f"risk={action.risk_score:.2f}"
            )
            self._execution_log.append(msg)
            self._set_pending_escalation(
                action,
                "High risk; human approval required",
            )
            return False, msg

        # ---------------- Store baseline for rollback checks ----------------
        if not self._active_actions:
            self._baseline_metrics = current_metrics

        applied = self._apply_action(trace)

        if applied:
            self._active_actions.append(
                (trace, {"applied_at": time.time(), "status": "executed"})
            )
            msg = f"Executed: {action.action_type} target={action.target}"
            self._execution_log.append(msg)
            return True, msg

        msg = f"Failed to execute: {action.action_type}"
        self._execution_log.append(msg)
        return False, msg

    def _apply_action(self, trace: DecisionTrace) -> bool:
        """Apply action to simulator or internal state."""
        action = trace.action

        if self._simulator_control:
            if action.action_type == ActionType.REROUTE and action.target:
                self._simulator_control(
                    "reroute",
                    {"issuer": action.target, **action.params},
                )
            elif action.action_type == ActionType.RETRY_POLICY:
                self._simulator_control(
                    "retry_policy",
                    action.params,
                )
            elif action.action_type == ActionType.SUPPRESS:
                self._simulator_control(
                    "suppress",
                    action.params,
                )
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

        success_dropped = (
            base.success_rate - cur.success_rate
            >= ROLLBACK_SUCCESS_RATE_DROP
        )

        latency_increased = (
            base.p95_latency_ms > 0
            and (cur.p95_latency_ms - base.p95_latency_ms)
            / base.p95_latency_ms
            >= ROLLBACK_LATENCY_INCREASE_FRACTION
        )

        if success_dropped or latency_increased:
            for trace, meta in self._active_actions:
                self._rollback_action(trace)
                msg = (
                    f"Rollback: {trace.action.action_type} "
                    f"(success_drop={success_dropped}, latency_inc={latency_increased})"
                )
                rollbacks.append(msg)
                self._rollback_log.append(msg)
                meta["status"] = "rolled_back"

            self._active_actions.clear()
            self._baseline_metrics = None

        return rollbacks

    def _rollback_action(self, trace: DecisionTrace) -> None:
        """Revert a single action (e.g. clear reroute/suppress via simulator)."""
        if self._simulator_control:
            self._simulator_control(
                "rollback",
                {
                    "action_type": trace.action.action_type,
                    "target": trace.action.target,
                },
            )

    def get_active_actions(self) -> list[DecisionTrace]:
        return [t for t, _ in self._active_actions]

    def get_rollback_log(self) -> list[str]:
        return self._rollback_log.copy()

    def get_rollback_count(self) -> int:
        """Total rollbacks so far (for decision context / forced human handover)."""
        return len(self._rollback_log)

    # ---------------- NEW (Plan A): explainability & observability ----------------
    def get_execution_log(self) -> list[str]:
        """
        Returns a chronological log of:
        - executed actions
        - blocked actions
        - human-approval escalations
        """
        return self._execution_log.copy()

    # ---------------- Real-time control plane: escalation state for dashboard ----------------
    def get_escalation_state(self) -> Optional[dict[str, Any]]:
        """
        Returns current human-in-the-loop escalation state for dashboard.
        None if no pending escalation; else dict with reason, action_type, target, risk_score.
        """
        if not self._pending_escalation:
            return None
        return self._pending_escalation.copy()

    def check_and_apply_approval(self, metrics: WindowMetrics) -> tuple[bool, Optional[str], Optional[Action]]:
        """
        Check if pending action has been approved or rejected.
        Returns: (processed_something: bool, message: Optional[str], approved_action: Optional[Action])
        """
        if not self._pending_escalation:
            # Also check disk in case restart lost memory but file is there
            disk_state = read_pending_approval()
            if disk_state and disk_state.get("status") == "pending":
                self._pending_escalation = disk_state
            else:
                return False, None, None

        # Re-read disk to see if status changed
        persisted = read_pending_approval()
        if not persisted:
            # Maybe cleared manually?
            self._pending_escalation = None
            return False, None, None

        status = persisted.get("status")
        if status == "approved":
            # Reconstruct action
            action_dict = persisted.get("action", {})
            action = Action(
                action_type=action_dict.get("action_type"),
                target=action_dict.get("target"),
                params=action_dict.get("params", {}),
                risk_score=action_dict.get("risk_score", 0.0),
                reason=action_dict.get("reason", ""),
            )
            # Trace wrapper for execution
            trace = DecisionTrace(
                hypothesis=None,  # Not needed for execution
                action=action,
                risk_score=action.risk_score,
                reasoning="Human approved via dashboard",
                timestamp=time.time(),
            )
            
            # Execute!
            self._baseline_metrics = metrics  # Set baseline at approval time
            success = self._apply_action(trace)
            
            msg = f"Human APPROVED: {action.action_type} target={action.target}"
            self._execution_log.append(msg)
            
            if success:
                self._active_actions.append(
                    (trace, {"applied_at": time.time(), "status": "executed"})
                )
            
            # Clear pending
            write_pending_approval(None)
            self._pending_escalation = None
            return True, msg, action

        elif status == "rejected":
            msg = "Human REJECTED the proposed action."
            self._execution_log.append(msg)
            write_pending_approval(None)
            self._pending_escalation = None
            return True, msg, None

        return False, None, None  # Still pending

    def _set_pending_escalation(
        self,
        action: Action,
        reason: str,
    ) -> None:
        """Store pending action and persist to disk."""
        # Convert Action to dict for JSON
        import dataclasses
        action_dict = dataclasses.asdict(action)
        
        self._pending_escalation = {
            "active": True,
            "action": action_dict,
            "action_type": action.action_type,  # flatten for easy dashboard read
            "target": action.target,
            "risk_score": action.risk_score,
            "reason": reason,
            "status": "pending",
            "timestamp": time.time(),
        }
        write_pending_approval(self._pending_escalation)

    def clear_escalation(self) -> None:
        """Clear escalation after human approval or cancel."""
        self._pending_escalation = None
        write_pending_approval(None)
