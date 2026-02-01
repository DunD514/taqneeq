"""
Writes agent state to state/*.json for the operator dashboard.
All writes are atomic (write to temp then rename) for safe concurrent read.
"""
import json
import os
from pathlib import Path
from typing import Any, Optional

# Directory for shared state (dashboard reads from here)
STATE_DIR = Path(__file__).resolve().parent / "state"
METRICS_PATH = STATE_DIR / "metrics.json"
HYPOTHESES_PATH = STATE_DIR / "hypotheses.json"
ACTIONS_PATH = STATE_DIR / "actions.json"
PENDING_APPROVAL_PATH = STATE_DIR / "pending_approval.json"

# Max history entries to keep in JSON (for latency trend and recent hypotheses/actions)
MAX_LATENCY_POINTS = 100
MAX_HYPOTHESES_HISTORY = 50
MAX_ACTIONS_HISTORY = 50

CONTROL_STATE_PATH = STATE_DIR / "control_state.json"

# System modes for real-time dashboard
SYSTEM_MODE_NORMAL = "NORMAL"
SYSTEM_MODE_DEGRADED = "DEGRADED"
SYSTEM_MODE_HUMAN_APPROVAL_REQUIRED = "HUMAN_APPROVAL_REQUIRED"
SYSTEM_MODE_COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"

# Action outcome for decision timeline
OUTCOME_EXECUTED = "executed"
OUTCOME_BLOCKED = "blocked"
OUTCOME_SKIPPED_COOLDOWN = "skipped_cooldown"
OUTCOME_ROLLED_BACK = "rolled_back"


def _ensure_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    _ensure_dir()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    try:
        tmp.replace(path)
    except PermissionError:
        # On Windows, replace can fail if path is open (e.g. dashboard reading). Write directly as fallback.
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def write_metrics(
    success_rate: float,
    p95_latency_ms: float,
    success_rate_by_issuer: dict[str, float],
    retry_amplification: float,
    sample_count: int,
    window_id: str,
    ts: float,
    *,
    average_estimated_cost: Optional[float] = None,
    attempt_amplification: Optional[float] = None,
    success_rate_by_merchant: Optional[dict[str, float]] = None,
    avg_cost_by_merchant: Optional[dict[str, float]] = None,
    attempt_amplification_by_merchant: Optional[dict[str, float]] = None,
) -> None:
    """Append current metrics and latency trend for dashboard (real-time KPIs and merchant health)."""
    current = {
        "success_rate": success_rate,
        "p95_latency_ms": p95_latency_ms,
        "success_rate_by_issuer": success_rate_by_issuer,
        "retry_amplification": retry_amplification,
        "sample_count": sample_count,
        "window_id": window_id,
        "ts": ts,
    }
    if average_estimated_cost is not None:
        current["average_estimated_cost"] = average_estimated_cost
    if attempt_amplification is not None:
        current["attempt_amplification"] = attempt_amplification
    if success_rate_by_merchant is not None:
        current["success_rate_by_merchant"] = success_rate_by_merchant
    if avg_cost_by_merchant is not None:
        current["avg_cost_by_merchant"] = avg_cost_by_merchant
    if attempt_amplification_by_merchant is not None:
        current["attempt_amplification_by_merchant"] = attempt_amplification_by_merchant
    existing = _read_json(METRICS_PATH)
    trend = existing.get("latency_trend", [])
    trend.append({"window_id": window_id, "p95_latency_ms": p95_latency_ms, "ts": ts})
    if len(trend) > MAX_LATENCY_POINTS:
        trend = trend[-MAX_LATENCY_POINTS:]
    _write_json(METRICS_PATH, {"current": current, "latency_trend": trend})


def write_hypothesis(
    cause: str,
    confidence: float,
    evidence: str,
    source: str,
    ts: float,
    *,
    uncertainty: Optional[float] = None,
) -> None:
    """Write latest hypothesis and append to history (uncertainty for real-time decision timing)."""
    latest = {"cause": cause, "confidence": confidence, "evidence": evidence, "source": source, "ts": ts}
    if uncertainty is not None:
        latest["uncertainty"] = uncertainty
    existing = _read_json(HYPOTHESES_PATH)
    history = existing.get("history", [])
    history.append(latest)
    if len(history) > MAX_HYPOTHESES_HISTORY:
        history = history[-MAX_HYPOTHESES_HISTORY:]
    _write_json(HYPOTHESES_PATH, {"latest": latest, "history": history})


def write_action(
    action_type: str,
    target: Optional[str],
    risk_score: float,
    reason: str,
    reasoning: str,
    executed: bool,
    message: str,
    ts: float,
    *,
    outcome: str = "executed",  # executed | blocked | skipped_cooldown | rolled_back
    guardrails_triggered: Optional[list[str]] = None,
    append_to_history: bool = True,
    what_changed_since_last: Optional[str] = None,
    why_action_now: Optional[str] = None,
    why_human_approval: Optional[str] = None,
) -> None:
    """Write latest decision/action for dashboard. Explainability: what changed, why now, why human approval."""
    latest = {
        "action_type": action_type,
        "target": target,
        "risk_score": risk_score,
        "reason": reason,
        "reasoning": reasoning,
        "executed": executed,
        "message": message,
        "ts": ts,
        "outcome": outcome,
        "guardrails_triggered": guardrails_triggered or [],
    }
    if what_changed_since_last is not None:
        latest["what_changed_since_last"] = what_changed_since_last
    if why_action_now is not None:
        latest["why_action_now"] = why_action_now
    if why_human_approval is not None:
        latest["why_human_approval"] = why_human_approval
    existing = _read_json(ACTIONS_PATH)
    history = existing.get("history", [])
    if append_to_history:
        history.append(latest)
        if len(history) > MAX_ACTIONS_HISTORY:
            history = history[-MAX_ACTIONS_HISTORY:]
    _write_json(ACTIONS_PATH, {"latest": latest, "history": history})


def write_control_state(
    system_mode: str,
    ts: float,
    *,
    cooldown_until_ts: Optional[float] = None,
    escalation: Optional[dict[str, Any]] = None,
    learning: Optional[dict[str, Any]] = None,
) -> None:
    """Write control-plane state for real-time dashboard (system mode, escalation, learning)."""
    data = {
        "system_mode": system_mode,
        "ts": ts,
    }
    if cooldown_until_ts is not None:
        data["cooldown_until_ts"] = cooldown_until_ts
    if escalation is not None:
        data["escalation"] = escalation
    if learning is not None:
        data["learning"] = learning
    _write_json(CONTROL_STATE_PATH, data)


def write_pending_approval(approval: Optional[dict[str, Any]]) -> None:
    """Write pending approval state (or clear it if None)."""
    if approval is None:
        if PENDING_APPROVAL_PATH.exists():
            try:
                PENDING_APPROVAL_PATH.unlink()
            except OSError:
                _write_json(PENDING_APPROVAL_PATH, {})  # Fallback
    else:
        _write_json(PENDING_APPROVAL_PATH, approval)


def read_pending_approval() -> Optional[dict[str, Any]]:
    """Read pending approval state."""
    data = _read_json(PENDING_APPROVAL_PATH)
    if not data or not data.get("active", False):  # We'll use "active" flag or check status
        # Support both formats for robustness, but let's stick to the dict structure
        return data if data else None
    return data
