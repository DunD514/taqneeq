"""
Shared data models for the agentic payment operations system.
All types used across observer, reasoner, decision, executor, and learner.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class PaymentOutcome(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    TIMEOUT = "timeout"


class ErrorCode(str, Enum):
    NONE = "none"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    ISSUER_UNAVAILABLE = "issuer_unavailable"
    NETWORK_TIMEOUT = "network_timeout"
    INVALID_CARD = "invalid_card"
    RATE_LIMITED = "rate_limited"
    UNKNOWN = "unknown"


@dataclass
class PaymentEvent:
    """Single payment transaction event from the simulator."""
    event_id: str
    issuer_bank: str
    payment_method: str
    latency_ms: float
    retries: int
    outcome: PaymentOutcome
    error_code: ErrorCode
    timestamp: float = 0.0  # Unix epoch; set by simulator if needed


@dataclass
class WindowMetrics:
    """Aggregated metrics over a sliding window."""
    window_id: str
    start_ts: float
    end_ts: float
    success_rate: float
    p95_latency_ms: float
    retry_amplification: float  # retries per attempt
    error_distribution: dict[str, float]  # error_code -> fraction
    success_rate_by_issuer: dict[str, float]
    sample_count: int


@dataclass
class Hypothesis:
    """Structured hypothesis from the reasoner (LLM or fallback)."""
    cause: str
    confidence: float
    evidence: str
    source: str = "heuristic"  # "llm" | "heuristic"


@dataclass
class ActionType:
    """Action types the executor can perform."""
    REROUTE = "reroute"
    RETRY_POLICY = "retry_policy"
    SUPPRESS = "suppress"
    NO_OP = "no_op"


@dataclass
class Action:
    """Decision engine output: what to do (or no-op)."""
    action_type: str  # ActionType.*
    target: Optional[str] = None  # e.g. issuer name, flow name
    params: dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    reason: str = ""


@dataclass
class DecisionTrace:
    """Explainable record of how a decision was made."""
    hypothesis: Optional[Hypothesis]
    action: Action
    risk_score: float
    reasoning: str
    timestamp: float = 0.0


@dataclass
class OutcomeRecord:
    """Context → action → outcome for learning."""
    context_snapshot: dict[str, Any]  # metrics summary at decision time
    action: Action
    outcome_metrics: dict[str, Any]  # metrics after action window
    helped: bool  # did the action improve things?
    rollback_applied: bool = False
