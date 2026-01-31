"""
Deterministic decision engine. Optimizes tradeoffs between
success rate, latency, cost, user friction, and risk.
Outputs an Action (or NO_OP); does not execute anything.
"""
import time
from typing import Optional

from models import Action, ActionType, DecisionTrace, Hypothesis, WindowMetrics


# Risk thresholds: only actions below HIGH_RISK_THRESHOLD are auto-executed
LOW_RISK_THRESHOLD = 0.35
HIGH_RISK_THRESHOLD = 0.65
# Confidence threshold below which we prefer NO_OP
MIN_CONFIDENCE_TO_ACT = 0.6
# Success rate below which we consider intervention
SUCCESS_RATE_INTERVENE = 0.78
# P95 latency (ms) above which latency-focused actions are considered
P95_LATENCY_INTERVENE_MS = 500.0


def _risk_score(
    hypothesis: Hypothesis,
    action_type: str,
    target: Optional[str],
) -> float:
    """
    Deterministic risk score in [0, 1].
    Reroute and suppress are higher risk; retry_policy lower.
    Higher hypothesis confidence lowers risk.
    """
    base = 0.5
    if action_type == "no_op":
        return 0.0
    if action_type == "retry_policy":
        base = 0.25
    elif action_type == "reroute":
        base = 0.55
    elif action_type == "suppress":
        base = 0.6
    # Confidence reduces effective risk
    risk = base * (1.0 - hypothesis.confidence * 0.4)
    return max(0.0, min(1.0, risk))


def decide(
    metrics: WindowMetrics,
    hypothesis: Hypothesis,
) -> DecisionTrace:
    """
    Deterministic policy: map hypothesis + metrics to a single action.
    Returns DecisionTrace (action + reasoning). No execution here.
    """
    action = Action(
        action_type=ActionType.NO_OP,
        target=None,
        params={},
        risk_score=0.0,
        reason="No intervention needed",
    )
    reasoning = f"Hypothesis: {hypothesis.cause} (confidence={hypothesis.confidence:.2f}, source={hypothesis.source}). "

    if hypothesis.confidence < MIN_CONFIDENCE_TO_ACT:
        reasoning += "Confidence below threshold; no action."
        return DecisionTrace(
            hypothesis=hypothesis,
            action=action,
            risk_score=0.0,
            reasoning=reasoning,
            timestamp=time.time(),
        )

    # Map hypothesis cause to candidate action
    cause = hypothesis.cause.upper()
    target_issuer = None
    if "ISSUER_" in cause and "_DEGRADATION" in cause:
        parts = cause.replace("ISSUER_", "").replace("_DEGRADATION", "").split("_")
        target_issuer = "_".join(parts) if parts else None
        if not target_issuer and metrics.success_rate_by_issuer:
            target_issuer = min(
                metrics.success_rate_by_issuer.items(),
                key=lambda x: x[1],
            )[0]

    if "ISSUER_" in cause and "_DEGRADATION" in cause and target_issuer:
        # Reroute traffic away from degraded issuer (moderate risk)
        action = Action(
            action_type=ActionType.REROUTE,
            target=target_issuer,
            params={"weight_reduce": 0.5},
            risk_score=_risk_score(hypothesis, ActionType.REROUTE, target_issuer),
            reason=f"Reroute traffic from degraded issuer {target_issuer}",
        )
        reasoning += f"Degraded issuer detected; proposing reroute from {target_issuer}."
    elif "RETRY_STORM" in cause:
        # Adjust retry policy (lower risk)
        action = Action(
            action_type=ActionType.RETRY_POLICY,
            target=None,
            params={"max_retries": 2, "backoff_scale": 1.5},
            risk_score=_risk_score(hypothesis, ActionType.RETRY_POLICY, None),
            reason="Reduce retries to dampen storm",
        )
        reasoning += "Retry storm detected; proposing retry policy adjustment."
    elif "LATENCY_SPIKE" in cause:
        # Temporary suppression of heavy path (higher risk)
        action = Action(
            action_type=ActionType.SUPPRESS,
            target="heavy_path",
            params={"duration_sec": 60},
            risk_score=_risk_score(hypothesis, ActionType.SUPPRESS, "heavy_path"),
            reason="Temporarily suppress heavy path during latency spike",
        )
        reasoning += "Latency spike; proposing temporary suppression."
    elif "GENERAL_DEGRADATION" in cause or "DEGRADATION" in cause:
        # Conservative: retry policy first
        action = Action(
            action_type=ActionType.RETRY_POLICY,
            target=None,
            params={"max_retries": 2},
            risk_score=_risk_score(hypothesis, ActionType.RETRY_POLICY, None),
            reason="General degradation; soften retry policy",
        )
        reasoning += "General degradation; proposing retry policy adjustment."

    action.risk_score = _risk_score(hypothesis, action.action_type, action.target)
    return DecisionTrace(
        hypothesis=hypothesis,
        action=action,
        risk_score=action.risk_score,
        reasoning=reasoning,
        timestamp=time.time(),
    )
