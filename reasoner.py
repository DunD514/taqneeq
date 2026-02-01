"""
Reasoner: wraps LLM hypothesis generator with deterministic fallback.
If LLM fails or returns invalid JSON, we use heuristic pattern detection.
LLM is used only for interpreting metrics and producing hypotheses; it does not execute actions.
"""
from typing import Optional

from llm_reasoner import generate_hypothesis_llm
from models import Hypothesis, WindowMetrics


# Heuristic thresholds (documented: used when LLM unavailable or for sanity checks)
SUCCESS_RATE_DEGRADATION_THRESHOLD = 0.75
P95_LATENCY_SPIKE_MS = 600.0
RETRY_AMPLIFICATION_STORM = 1.2

# ---------------- NEW (Plan A): retry intelligence thresholds ----------------
ATTEMPT_AMPLIFICATION_STORM = 1.5     # avg attempts per txn
COST_ESCALATION_THRESHOLD = 0.03      # avg cost per txn (demo-safe)

# ---------------- Real-time: uncertainty-based NO-OP ----------------
MIN_SAMPLE_FOR_ACT = 30   # below this, emit INSUFFICIENT_SIGNAL
CAUSE_INSUFFICIENT_SIGNAL = "INSUFFICIENT_SIGNAL"

# Uncertainty: 0-1; higher when signals conflict or evidence weak (influences decision timing)
def _compute_uncertainty(
    cause: str,
    confidence: float,
    evidence_parts: list[str],
    metrics: WindowMetrics,
) -> float:
    """
    Increase uncertainty when: signals conflict, evidence weak/incomplete, multiple patterns.
    Used by agent to trigger reason/decide (e.g. when uncertainty >= threshold).
    """
    u = 0.0
    # Weak or no evidence -> high uncertainty
    if not evidence_parts or cause in ("Unknown", CAUSE_INSUFFICIENT_SIGNAL):
        u = 1.0 - confidence
        u = max(u, 0.5)  # at least 0.5 when unknown
        return min(1.0, u)
    # Low sample -> incomplete picture
    if metrics.sample_count < 50:
        u += 0.25
    if metrics.sample_count < 100:
        u += 0.15
    # Multiple issuers degraded? (conflicting / unclear root cause)
    by_issuer = metrics.success_rate_by_issuer or {}
    degraded_issuers = [i for i, r in by_issuer.items() if r < SUCCESS_RATE_DEGRADATION_THRESHOLD]
    if len(degraded_issuers) > 1:
        u += 0.3  # conflicting: multiple issuers, unclear which to act on
    # Mixed error distribution (conflicting error patterns)
    err_dist = metrics.error_distribution or {}
    if len(err_dist) > 3 and max(err_dist.values() or [0]) < 0.5:
        u += 0.2  # no dominant error, unclear root cause
    return min(1.0, u)


def _heuristic_hypothesis(metrics: WindowMetrics) -> Hypothesis:
    """
    Deterministic hypothesis from metrics. Used when LLM is unavailable
    or returns invalid output. Identifies issuer degradation, retry storm,
    latency spike, and retry amplification storms.
    """
    cause = "Unknown"
    confidence = 0.0
    evidence_parts = []

    # ---------------- Real-time: insufficient signal (blocks action, explains why) ----------------
    if metrics.sample_count < MIN_SAMPLE_FOR_ACT:
        return Hypothesis(
            cause=CAUSE_INSUFFICIENT_SIGNAL,
            confidence=0.0,
            evidence=f"Sample count too low ({metrics.sample_count}) for reliable hypothesis.",
            source="heuristic",
            uncertainty=1.0,  # high uncertainty -> influences decision timing
        )

    # ---------------- Existing logic (UNCHANGED) ----------------

    # Check for single-issuer degradation (worst success rate)
    if metrics.success_rate_by_issuer:
        worst_issuer = min(
            metrics.success_rate_by_issuer.items(),
            key=lambda x: x[1],
        )
        issuer_name, rate = worst_issuer
        if rate < SUCCESS_RATE_DEGRADATION_THRESHOLD:
            cause = f"Issuer_{issuer_name}_Degradation"
            confidence = 0.85
            evidence_parts.append(
                f"Success rate for {issuer_name} dropped to {rate:.1%}"
            )

    # Legacy retry storm (retry-based)
    if metrics.retry_amplification >= RETRY_AMPLIFICATION_STORM and metrics.success_rate < 0.7:
        if confidence < 0.5:
            cause = "Retry_Storm"
            confidence = 0.8
            evidence_parts.append(
                f"Retry amplification {metrics.retry_amplification:.2f} with success rate {metrics.success_rate:.1%}"
            )

    # Latency spike
    if metrics.p95_latency_ms >= P95_LATENCY_SPIKE_MS:
        if confidence < 0.5:
            cause = "Latency_Spike"
            confidence = 0.75
            evidence_parts.append(
                f"P95 latency {metrics.p95_latency_ms:.0f} ms"
            )
        else:
            evidence_parts.append(f"P95 latency {metrics.p95_latency_ms:.0f} ms")

    # Global success rate drop
    if metrics.success_rate < SUCCESS_RATE_DEGRADATION_THRESHOLD and not evidence_parts:
        cause = "General_Degradation"
        confidence = 0.7
        evidence_parts.append(f"Overall success rate {metrics.success_rate:.1%}")

    # ---------------- NEW (Plan A): attempt + cost aware retry storm ----------------

    attempt_amp = getattr(metrics, "attempt_amplification", None)
    avg_cost = getattr(metrics, "average_estimated_cost", None)

    if (
        attempt_amp is not None
        and attempt_amp >= ATTEMPT_AMPLIFICATION_STORM
        and metrics.success_rate < 0.8
    ):
        # Prefer this hypothesis over legacy retry storm
        cause = "Retry_Storm_Detected"
        confidence = 0.85
        evidence_parts = [
            f"Avg attempts per txn {attempt_amp:.2f}",
            f"Success rate {metrics.success_rate:.1%}",
        ]

        if avg_cost is not None and avg_cost >= COST_ESCALATION_THRESHOLD:
            confidence = min(0.95, confidence + 0.05)
            evidence_parts.append(f"Avg cost per txn {avg_cost:.3f}")

    evidence = " ".join(evidence_parts) if evidence_parts else "No strong pattern detected."

    # ---------------- Real-time: no clear dominant degradation -> INSUFFICIENT_SIGNAL ----------------
    if cause == "Unknown" and confidence == 0.0:
        cause = CAUSE_INSUFFICIENT_SIGNAL
        evidence = "No strong pattern detected; insufficient signal to act (metrics stable or no clear degradation)."

    uncertainty = _compute_uncertainty(cause, confidence, evidence_parts, metrics)
    return Hypothesis(
        cause=cause,
        confidence=confidence,
        evidence=evidence,
        source="heuristic",
        uncertainty=uncertainty,
    )


def reason(metrics: WindowMetrics) -> Hypothesis:
    """
    Produce a single hypothesis from current metrics, with confidence and uncertainty.
    Uncertainty is high when signals conflict or evidence weak; influences decision timing.
    Tries LLM first; on failure or missing API key, falls back to heuristic.
    """
    llm_h = generate_hypothesis_llm(metrics)
    if llm_h is not None and llm_h.confidence > 0:
        # Ensure LLM hypothesis has uncertainty (default from heuristic if missing)
        if getattr(llm_h, "uncertainty", None) is None:
            u = _compute_uncertainty(llm_h.cause, llm_h.confidence, [llm_h.evidence], metrics)
            llm_h = Hypothesis(
                cause=llm_h.cause,
                confidence=llm_h.confidence,
                evidence=llm_h.evidence,
                source=llm_h.source,
                uncertainty=u,
            )
        return llm_h
    return _heuristic_hypothesis(metrics)
