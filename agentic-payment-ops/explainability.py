"""
Human-readable explanations for observations, hypotheses, decisions,
trade-offs, guardrails, and rollback reasons. Explanation only; no execution.
"""
from typing import Any, Optional

from models import Action, DecisionTrace, Hypothesis, WindowMetrics


def explain_observation(metrics: WindowMetrics) -> str:
    """Plain-English summary of current window metrics."""
    parts = []
    parts.append(f"Success rate is {metrics.success_rate:.0%}.")
    parts.append(f"Typical response time is around {metrics.p95_latency_ms:.0f} ms.")
    if metrics.retry_amplification > 0.5:
        parts.append(f"Retries are elevated ({metrics.retry_amplification:.2f} retries per attempt).")
    if getattr(metrics, "average_estimated_cost", None) is not None:
        parts.append(f"Average estimated cost per transaction is {metrics.average_estimated_cost:.4f}.")
    return " ".join(parts)


def explain_hypothesis(hypothesis: Hypothesis) -> str:
    """Plain-English summary of the current hypothesis."""
    if hypothesis.confidence < 0.3:
        return "No strong pattern detected; the system is watching."
    cause = hypothesis.cause
    if cause == "Unknown" or not cause:
        return "No clear cause identified yet."
    evidence = hypothesis.evidence or ""
    return f"The system suspects: {cause}. Evidence: {evidence} (confidence {hypothesis.confidence:.0%})."


def explain_decision(trace: DecisionTrace) -> str:
    """Plain-English summary of the decision and reasoning."""
    action = trace.action
    if action.action_type == "no_op":
        return "No action taken. " + (trace.reasoning or "No intervention needed.")
    trade_off = "Trade-off: " + (action.reason or "attempting to improve metrics.")
    if action.requires_human_approval:
        trade_off += " This action requires human approval and was not auto-executed."
    return trade_off + " Reasoning: " + (trace.reasoning or "")


def explain_trade_offs(metrics: WindowMetrics, trace: DecisionTrace) -> str:
    """Explicit trade-offs: success vs latency vs cost."""
    action = trace.action
    if action.action_type == "no_op":
        return "No trade-off applied; system chose to wait."
    parts = ["Trade-offs considered:"]
    parts.append(f"Success rate (current: {metrics.success_rate:.0%}).")
    parts.append(f"Latency (current p95: {metrics.p95_latency_ms:.0f} ms).")
    if getattr(metrics, "average_estimated_cost", None) is not None:
        parts.append(f"Cost (current avg: {metrics.average_estimated_cost:.4f}).")
    parts.append(f"Risk score of proposed action: {action.risk_score:.2f}.")
    return " ".join(parts)


def explain_guardrail(executed: bool, message: str, requires_human_approval: bool = False) -> str:
    """Why an action was or was not auto-executed."""
    if executed:
        return "Action was within risk limits and was executed automatically."
    if requires_human_approval:
        return "Action was blocked: it requires human approval. " + (message or "")
    return "Action was blocked: risk too high for auto-execution. " + (message or "")


def explain_rollback(reason: str, success_drop: bool, latency_inc: bool) -> str:
    """Why a rollback happened (rollback is not failure)."""
    parts = ["Rollback applied: the previous action was reverted."]
    if success_drop:
        parts.append("Success rate had dropped after the action.")
    if latency_inc:
        parts.append("Latency had increased after the action.")
    parts.append("Rollback is a safety mechanism, not a failure of the system.")
    return " ".join(parts) + " " + (reason or "")
