"""
Deterministic decision engine. Optimizes tradeoffs between
success rate, latency, cost, user friction, and risk.
Outputs an Action (or NO_OP); does not execute anything.
Real-time: risk accumulation over time, forced human handover.
"""
import time
from typing import Any, Optional

from models import Action, ActionType, DecisionTrace, Hypothesis, WindowMetrics


# ---------------- Risk thresholds ----------------
LOW_RISK_THRESHOLD = 0.35
HIGH_RISK_THRESHOLD = 0.65

# Confidence threshold below which we prefer NO_OP
MIN_CONFIDENCE_TO_ACT = 0.6

# ---------------- Real-time: risk accumulation & human handover ----------------
# Same action+target repeated within this many recent causes -> boost risk
REPEATED_ISSUER_WINDOW = 5
RISK_BOOST_PER_REPEAT = 0.08
# Rollbacks: after this many, force human approval for all interventions (during runtime)
ROLLBACK_COUNT_FORCE_HUMAN = 2
# Severe degradation: success rate below this -> consider forced handover
SEVERE_DEGRADATION_SUCCESS_RATE = 0.65
# Uncertainty above this -> force human approval (unclear root cause; must not auto-act)
UNCERTAINTY_HANDOVER_THRESHOLD = 0.5
# Multiple merchants failing -> force human handover (multi-tenant impact; human must decide)
MULTI_MERCHANT_DEGRADED_THRESHOLD = 2  # number of merchants with success < this triggers handover
MERCHANT_DEGRADED_SUCCESS_RATE = 0.75

# Success rate below which we consider intervention
SUCCESS_RATE_INTERVENE = 0.78

# P95 latency (ms) above which latency-focused actions are considered
P95_LATENCY_INTERVENE_MS = 500.0

# ---------------- NEW (Plan A): cost & retry guardrails ----------------
HIGH_COST_ESCALATION_THRESHOLD = 0.05     # avg cost per txn
HIGH_ATTEMPT_AMPLIFICATION = 1.6          # avg attempts per txn


def _risk_score(
    hypothesis: Hypothesis,
    action_type: str,
    target: Optional[str],
    risk_accumulation: float = 0.0,
) -> float:
    """
    Deterministic risk score in [0, 1].
    Real-time: risk_accumulation adds when same issuer/degradation persists.
    """
    base = 0.5

    if action_type == ActionType.NO_OP:
        return 0.0
    if action_type == ActionType.RETRY_POLICY:
        base = 0.25
    elif action_type == ActionType.REROUTE:
        base = 0.55
    elif action_type == ActionType.SUPPRESS:
        base = 0.6

    risk = base * (1.0 - hypothesis.confidence * 0.4) + risk_accumulation
    return max(0.0, min(1.0, risk))


def _compute_risk_accumulation(
    hypothesis: Hypothesis,
    target_issuer: Optional[str],
    context: Optional[dict[str, Any]],
) -> float:
    """Cumulative risk when same issuer repeats or degradation persists across windows."""
    if not context or not target_issuer:
        return 0.0
    recent_causes = context.get("recent_causes") or []
    # Count how often same issuer appeared in recent windows
    count = 0
    for c in recent_causes[-REPEATED_ISSUER_WINDOW:]:
        if isinstance(c, dict):
            cause = c.get("cause") or ""
            target = c.get("target")
            if target == target_issuer or (target is None and target_issuer in cause):
                count += 1
        elif target_issuer in str(c):
            count += 1
    if count <= 1:
        return 0.0
    return min(0.25, (count - 1) * RISK_BOOST_PER_REPEAT)


def _should_force_human_handover(
    metrics: WindowMetrics,
    hypothesis: Hypothesis,
    context: Optional[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """
    Force escalation during runtime when:
    - Severe degradation persists
    - Economic risk compounds
    - Multiple rollbacks occurred
    - Multiple merchants affected (multi-tenant impact; human must decide scope)
    - Uncertainty remains high (unclear root cause; do not auto-execute)
    Returns (force, list of guardrail reasons) for explainability.
    """
    reasons = []
    if metrics.success_rate < SEVERE_DEGRADATION_SUCCESS_RATE:
        reasons.append("Severe degradation (success rate below threshold)")
    rollback_count = (context or {}).get("rollback_count") or 0
    if rollback_count >= ROLLBACK_COUNT_FORCE_HUMAN:
        reasons.append(f"Multiple rollbacks ({rollback_count}); human approval required")
    avg_cost = getattr(metrics, "average_estimated_cost", None)
    if avg_cost is not None and avg_cost >= HIGH_COST_ESCALATION_THRESHOLD:
        reasons.append("Economic risk (cost above escalation threshold)")
    # Forced human handover: multiple merchants failing (post-migration / unclear scope)
    by_merchant = getattr(metrics, "success_rate_by_merchant", None) or {}
    degraded_merchants = [m for m, r in by_merchant.items() if r < MERCHANT_DEGRADED_SUCCESS_RATE]
    if len(degraded_merchants) >= MULTI_MERCHANT_DEGRADED_THRESHOLD:
        reasons.append(
            f"Multiple merchants affected ({len(degraded_merchants)}); "
            "human approval required to confirm scope and action."
        )
    # Forced human handover: uncertainty remains high (conflicting signals / unclear root cause)
    uncertainty = getattr(hypothesis, "uncertainty", 0.0)
    if uncertainty >= UNCERTAINTY_HANDOVER_THRESHOLD:
        reasons.append(
            f"Uncertainty high ({uncertainty:.2f}); root cause unclear. "
            "Human approval required before acting."
        )
    return (len(reasons) > 0, reasons)


def decide(
    metrics: WindowMetrics,
    hypothesis: Hypothesis,
    context: Optional[dict[str, Any]] = None,
) -> DecisionTrace:
    """
    Deterministic policy: map hypothesis + metrics to a single action.
    Real-time: uses context for risk accumulation and forced human handover.
    """

    # ---------------- Guard: INSUFFICIENT_SIGNAL (explicit uncertainty) ----------------
    if hypothesis.cause and "INSUFFICIENT_SIGNAL" in hypothesis.cause.upper():
        reasoning = (
            f"Hypothesis: {hypothesis.cause} (confidence={hypothesis.confidence:.2f}). "
            f"{hypothesis.evidence} No action taken."
        )
        return DecisionTrace(
            hypothesis=hypothesis,
            action=Action(
                action_type=ActionType.NO_OP,
                target=None,
                params={},
                risk_score=0.0,
                reason="Insufficient signal",
            ),
            risk_score=0.0,
            reasoning=reasoning,
            timestamp=time.time(),
        )

    # ---------------- Default NO_OP ----------------
    action = Action(
        action_type=ActionType.NO_OP,
        target=None,
        params={},
        risk_score=0.0,
        reason="No intervention needed",
    )

    reasoning = (
        f"Hypothesis: {hypothesis.cause} "
        f"(confidence={hypothesis.confidence:.2f}, source={hypothesis.source}). "
    )

    # ---------------- Guard: low confidence ----------------
    if hypothesis.confidence < MIN_CONFIDENCE_TO_ACT:
        reasoning += "Confidence below threshold; no action taken."
        return DecisionTrace(
            hypothesis=hypothesis,
            action=action,
            risk_score=0.0,
            reasoning=reasoning,
            timestamp=time.time(),
        )

    # ---------------- Real-time: forced human handover (during runtime, not post-run) ----------------
    force_human, guardrail_reasons = _should_force_human_handover(metrics, hypothesis, context)

    # ---------------- Extract optional Plan A signals safely ----------------
    attempt_amp = getattr(metrics, "attempt_amplification", None)
    avg_cost = getattr(metrics, "average_estimated_cost", None)

    cause = hypothesis.cause.upper()
    target_issuer = None

    # ---------------- Issuer degradation ----------------
    if "ISSUER_" in cause and "_DEGRADATION" in cause:
        parts = cause.replace("ISSUER_", "").replace("_DEGRADATION", "").split("_")
        target_issuer = "_".join(parts) if parts else None

        if not target_issuer and metrics.success_rate_by_issuer:
            target_issuer = min(
                metrics.success_rate_by_issuer.items(),
                key=lambda x: x[1],
            )[0]

        if target_issuer:
            risk_accum = _compute_risk_accumulation(
                hypothesis, target_issuer, context
            )
            action = Action(
                action_type=ActionType.REROUTE,
                target=target_issuer,
                params={"weight_reduce": 0.5},
                risk_score=_risk_score(
                    hypothesis, ActionType.REROUTE, target_issuer, risk_accum
                ),
                reason=f"Reroute traffic from degraded issuer {target_issuer}",
            )
            if force_human:
                action.params["requires_human_approval"] = True
                reasoning += " Forced human approval: " + "; ".join(guardrail_reasons) + ". "
            reasoning += (
                f"Issuer-level degradation detected; proposing reroute from {target_issuer}."
            )

    # ---------------- Retry storm (cost-aware) ----------------
    elif "RETRY_STORM" in cause:
        risk_accum = _compute_risk_accumulation(hypothesis, None, context)
        action = Action(
            action_type=ActionType.RETRY_POLICY,
            target=None,
            params={
                "max_retries": 2,
                "backoff_scale": 1.5,
                "intent": "reduce_retry_amplification",
            },
            risk_score=_risk_score(hypothesis, ActionType.RETRY_POLICY, None, risk_accum),
            reason="Retry storm detected; reduce retries to limit cost and latency",
        )
        reasoning += "Retry storm detected; proposing retry policy tightening."

        # -------- Human approval boundary (explicit & explainable) --------
        if force_human:
            action.params["requires_human_approval"] = True
            reasoning += " Forced human approval: " + "; ".join(guardrail_reasons) + ". "
        elif (
            avg_cost is not None and avg_cost >= HIGH_COST_ESCALATION_THRESHOLD
        ) or (
            attempt_amp is not None and attempt_amp >= HIGH_ATTEMPT_AMPLIFICATION
        ):
            action.params["requires_human_approval"] = True
            reasoning += (
                " Elevated cost or retry amplification detected; "
                "human approval required before applying."
            )

    # ---------------- Latency spike ----------------
    elif "LATENCY_SPIKE" in cause:
        action = Action(
            action_type=ActionType.SUPPRESS,
            target="heavy_path",
            params={"duration_sec": 60},
            risk_score=_risk_score(hypothesis, ActionType.SUPPRESS, "heavy_path"),
            reason="Latency spike detected; temporarily suppress heavy path",
        )
        reasoning += "Latency spike detected; proposing temporary suppression."

    # ---------------- General degradation ----------------
    elif "GENERAL_DEGRADATION" in cause or "DEGRADATION" in cause:
        action = Action(
            action_type=ActionType.RETRY_POLICY,
            target=None,
            params={"max_retries": 2},
            risk_score=_risk_score(hypothesis, ActionType.RETRY_POLICY, None),
            reason="General degradation; conservative retry reduction",
        )
        reasoning += "General degradation detected; proposing conservative retry adjustment."

    # ---------------- Final risk assignment (with accumulation) ----------------
    risk_accum = _compute_risk_accumulation(
        hypothesis, action.target, context
    )
    action.risk_score = _risk_score(
        hypothesis,
        action.action_type,
        action.target,
        risk_accum,
    )
    if force_human and action.action_type != ActionType.NO_OP:
        action.params["requires_human_approval"] = True

    # ---------------- Check for PENDING state (Prevent Thrashing) ----------------
    pending = (context or {}).get("pending_approval")
    if pending and action.action_type != ActionType.NO_OP:
        p_type = pending.get("action_type")
        p_target = pending.get("target")
        if action.action_type == p_type and action.target == p_target:
            # Downgrade to NO_OP to wait for human
            reasoning += " [WAITING] Identical action pending human approval."
            return DecisionTrace(
                hypothesis=hypothesis,
                action=Action(
                     action_type=ActionType.NO_OP,
                     target=None,
                     risk_score=0.0,
                     reason="Waiting for human approval",
                ),
                risk_score=0.0,
                reasoning=reasoning,
                timestamp=time.time(),
            )

    return DecisionTrace(
        hypothesis=hypothesis,
        action=action,
        risk_score=action.risk_score,
        reasoning=reasoning,
        timestamp=time.time(),
    )
