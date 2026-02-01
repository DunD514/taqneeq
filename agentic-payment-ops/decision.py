"""
Deterministic decision engine. Optimizes tradeoffs between
success rate, latency, cost, user friction, and risk.
Outputs an Action (or NO_OP); does not execute anything.
Learning signals (when available) influence confidence/caution, not hard rules.
"""
import time
from typing import Any, Optional

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
# Attempt amplification: above this, retry-policy actions are considered
ATTEMPT_AMPLIFICATION_THRESHOLD = 1.2
# Cost: if average cost is above this, cost-aware caution (optional flag)
AVERAGE_COST_CAUTION_THRESHOLD = 0.03
# Risk above this: set requires_human_approval
REQUIRES_HUMAN_APPROVAL_RISK = 0.5


def _risk_score(
    hypothesis: Hypothesis,
    action_type: str,
    target: Optional[str],
    metrics: Optional["WindowMetrics"] = None,
) -> float:
    """
    Deterministic risk score in [0, 1].
    Reroute and suppress are higher risk; retry_policy lower.
    Higher hypothesis confidence lowers risk. Cost and attempt amplification can increase risk.
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
    # Cost-aware: if average cost is high, slightly increase risk (caution)
    if metrics and getattr(metrics, "average_estimated_cost", None) is not None:
        if metrics.average_estimated_cost >= AVERAGE_COST_CAUTION_THRESHOLD:
            risk = min(1.0, risk + 0.05)
    return max(0.0, min(1.0, risk))


def decide(
    metrics: WindowMetrics,
    hypothesis: Hypothesis,
    learning_signal: Optional[dict[str, Any]] = None,
    persistence_multiplier: float = 1.0,
) -> DecisionTrace:
    """
    Deterministic policy: map hypothesis + metrics to a single action.
    Returns DecisionTrace (action + reasoning). No execution here.
    metrics: current window metrics.
    hypothesis: current reasoning.
    learning_signal: feedback from learner.
    persistence_multiplier: >1.0 if this hypothesis has persisted across multiple windows.
    """
    action = Action(
        action_type=ActionType.NO_OP,
        target=None,
        params={},
        risk_score=0.0,
        reason="No intervention needed",
    )
    reasoning = f"Hypothesis: {hypothesis.cause} (confidence={hypothesis.confidence:.2f}, source={hypothesis.source}). "

    if hypothesis.cause == "INSUFFICIENT_SIGNAL":
        return DecisionTrace(
            hypothesis=hypothesis,
            action=action,
            risk_score=0.0,
            reasoning="Insufficient signal; waiting for more data.",
            timestamp=time.time(),
        )

    effective_min_confidence = MIN_CONFIDENCE_TO_ACT
    if learning_signal and learning_signal.get("hurt", 0) > learning_signal.get("helped", 0):
        effective_min_confidence = min(0.75, MIN_CONFIDENCE_TO_ACT + 0.05)
        reasoning += "Recent outcomes favour caution; requiring higher confidence to act. "

    if hypothesis.confidence < effective_min_confidence:
        reasoning += "Confidence below threshold; no action."
        return DecisionTrace(
            hypothesis=hypothesis,
            action=action,
            risk_score=0.0,
            reasoning=reasoning,
            timestamp=time.time(),
        )

    # RISK ACCUMULATION: Increase risk if problem persists
    risk_boost = (persistence_multiplier - 1.0) * 0.1
    if risk_boost > 0:
        reasoning += f"Risk accumulated due to persistence (factor={persistence_multiplier:.1f}). "

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
        risk = _risk_score(hypothesis, ActionType.REROUTE, target_issuer, metrics) + risk_boost
        action = Action(
            action_type=ActionType.REROUTE,
            target=target_issuer,
            params={"weight_reduce": 0.5},
            risk_score=risk,
            reason=f"Reroute traffic from degraded issuer {target_issuer}",
            requires_human_approval=risk >= REQUIRES_HUMAN_APPROVAL_RISK,
        )
        reasoning += f"Degraded issuer detected; proposing reroute from {target_issuer}."
    elif "RETRY_STORM" in cause:
        # Adjust retry policy
        risk = _risk_score(hypothesis, ActionType.RETRY_POLICY, None, metrics) + risk_boost
        attempt_amp = getattr(metrics, "retry_amplification", 0.0) or 0.0
        if attempt_amp >= ATTEMPT_AMPLIFICATION_THRESHOLD:
            reasoning += f"Attempt amplification {attempt_amp:.2f} above threshold; "
            
        action = Action(
            action_type=ActionType.RETRY_POLICY,
            target=None,
            params={"max_retries": 2, "backoff_scale": 1.5},
            risk_score=risk,
            reason="Reduce retries to dampen storm",
            requires_human_approval=risk >= REQUIRES_HUMAN_APPROVAL_RISK,
        )
        reasoning += "Retry storm detected; proposing retry policy adjustment."
    elif "LATENCY_SPIKE" in cause:
        # Temporary suppression
        risk = _risk_score(hypothesis, ActionType.SUPPRESS, "heavy_path", metrics) + risk_boost
        action = Action(
            action_type=ActionType.SUPPRESS,
            target="heavy_path",
            params={"duration_sec": 60},
            risk_score=risk,
            reason="Temporarily suppress heavy path during latency spike",
            requires_human_approval=risk >= REQUIRES_HUMAN_APPROVAL_RISK,
        )
        reasoning += "Latency spike; proposing temporary suppression."
    elif "GENERAL_DEGRADATION" in cause or "DEGRADATION" in cause:
        risk = _risk_score(hypothesis, ActionType.RETRY_POLICY, None, metrics) + risk_boost
        action = Action(
            action_type=ActionType.RETRY_POLICY,
            target=None,
            params={"max_retries": 2},
            risk_score=risk,
            reason="General degradation; soften retry policy",
            requires_human_approval=risk >= REQUIRES_HUMAN_APPROVAL_RISK,
        )
        reasoning += "General degradation; proposing retry policy adjustment."

    # Final Risk Score Clamping
    action.risk_score = min(1.0, max(0.0, action.risk_score))

    # FORCED HUMAN HANDOVER TRIGGERS
    # 1. Severe Degradation: SR < 0.55 AND P95 > 800ms
    if metrics.success_rate < 0.55 and metrics.p95_latency_ms > 800:
        action.requires_human_approval = True
        reasoning += " [CRITICAL] Severe degradation detected; forcing human handover."
    
    # 2. Economic Risk: Attempt Amplification > 3.0 (Very High)
    attempt_amp = getattr(metrics, "retry_amplification", 0.0) or 0.0
    if attempt_amp > 3.0:
        action.requires_human_approval = True
        reasoning += " [CRITICAL] Retry amplification > 3x; forcing human handover."

    # 3. High Accumulated Risk
    if action.risk_score >= 0.8:
        action.requires_human_approval = True
        reasoning += " [CRITICAL] Accumulated risk score > 0.8; forcing human handover."

    return DecisionTrace(
        hypothesis=hypothesis,
        action=action,
        risk_score=action.risk_score,
        reasoning=reasoning,
        timestamp=time.time(),
    )
