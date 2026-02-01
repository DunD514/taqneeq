"""
Learning policy: define what "helped", "hurt", and "neutral" mean.
Includes cost and retry harm detection. No automatic policy mutation — learning is observational.
"""
from typing import Any


# Success rate improvement to count as "helped"
HELPED_SUCCESS_IMPROVEMENT = 0.03
# Latency reduction (fraction) to count as "helped"
HELPED_LATENCY_REDUCTION = 0.15
# Cost explosion: if cost increases by this fraction, count as hurt (even without rollback)
COST_HARM_INCREASE_FRACTION = 0.25
# Retry amplification increase: if retries per attempt increase by this much, count as hurt
RETRY_HARM_INCREASE = 0.3


def helped(
    before: dict[str, Any],
    after: dict[str, Any],
    rollback_applied: bool,
) -> bool:
    """True if the action improved success rate or reduced latency (and no rollback)."""
    if rollback_applied:
        return False
    sr_improved = (
        after.get("success_rate", 0) - before.get("success_rate", 0)
        >= HELPED_SUCCESS_IMPROVEMENT
    )
    lat_before = before.get("p95_latency_ms") or 0
    lat_after = after.get("p95_latency_ms") or 0
    lat_reduced = (
        lat_before > 0
        and (lat_before - lat_after) / lat_before >= HELPED_LATENCY_REDUCTION
    )
    return sr_improved or lat_reduced


def cost_harm(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """True if average cost increased significantly (cost explosion)."""
    b_cost = before.get("average_estimated_cost")
    a_cost = after.get("average_estimated_cost")
    if b_cost is None or a_cost is None or b_cost <= 0:
        return False
    return (a_cost - b_cost) / b_cost >= COST_HARM_INCREASE_FRACTION


def retry_harm(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """True if retry amplification increased significantly."""
    b_retry = before.get("retry_amplification", 0) or 0
    a_retry = after.get("retry_amplification", 0) or 0
    return (a_retry - b_retry) >= RETRY_HARM_INCREASE


def hurt(
    before: dict[str, Any],
    after: dict[str, Any],
    rollback_applied: bool,
) -> bool:
    """True if rollback, or cost explosion, or retry harm."""
    if rollback_applied:
        return True
    if cost_harm(before, after):
        return True
    if retry_harm(before, after):
        return True
    return False
