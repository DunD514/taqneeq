"""
Reasoner: wraps LLM hypothesis generator with deterministic fallback.
If LLM fails or returns invalid JSON, we use heuristic pattern detection.
LLM is used only for interpreting metrics and producing hypotheses; it does not execute actions.
"""
from typing import Optional

from llm_reasoner import generate_hypothesis_llm
from models import Hypothesis, WindowMetrics


# Heuristic thresholds (documented: used when LLM unavailable or for sanity checks)
# Success rate below this suggests degradation
SUCCESS_RATE_DEGRADATION_THRESHOLD = 0.75
# P95 latency above this (ms) suggests latency spike
P95_LATENCY_SPIKE_MS = 600.0
# Retry amplification above this suggests retry storm
RETRY_AMPLIFICATION_STORM = 1.2


def _heuristic_hypothesis(metrics: WindowMetrics) -> Hypothesis:
    """
    Deterministic hypothesis from metrics. Used when LLM is unavailable
    or returns invalid output. Identifies issuer degradation, retry storm, latency spike.
    """
    cause = "Unknown"
    confidence = 0.0
    evidence_parts = []

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

    # Retry storm: high retry amplification with low overall success
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

    # Global success rate drop without single-issuer signal
    if metrics.success_rate < SUCCESS_RATE_DEGRADATION_THRESHOLD and not evidence_parts:
        cause = "General_Degradation"
        confidence = 0.7
        evidence_parts.append(f"Overall success rate {metrics.success_rate:.1%}")

    evidence = " ".join(evidence_parts) if evidence_parts else "No strong pattern detected."
    return Hypothesis(
        cause=cause,
        confidence=confidence,
        evidence=evidence,
        source="heuristic",
    )


def reason(metrics: WindowMetrics) -> Hypothesis:
    """
    Produce a single hypothesis from current metrics.
    Tries LLM first; on failure or missing API key, falls back to heuristic.
    """
    llm_h = generate_hypothesis_llm(metrics)
    if llm_h is not None and llm_h.confidence > 0:
        return llm_h
    return _heuristic_hypothesis(metrics)
