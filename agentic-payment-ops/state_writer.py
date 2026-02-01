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

# Max history entries to keep in JSON (for latency trend and recent hypotheses/actions)
MAX_LATENCY_POINTS = 100
MAX_HYPOTHESES_HISTORY = 50
MAX_ACTIONS_HISTORY = 50


def _ensure_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    _ensure_dir()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(path)


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
) -> None:
    """Append current metrics and latency trend for dashboard."""
    current = {
        "success_rate": success_rate,
        "p95_latency_ms": p95_latency_ms,
        "success_rate_by_issuer": success_rate_by_issuer,
        "retry_amplification": retry_amplification,
        "sample_count": sample_count,
        "window_id": window_id,
        "ts": ts,
    }
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
) -> None:
    """Write latest hypothesis and append to history."""
    latest = {"cause": cause, "confidence": confidence, "evidence": evidence, "source": source, "ts": ts}
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
) -> None:
    """Write latest decision/action and explanation for dashboard."""
    latest = {
        "action_type": action_type,
        "target": target,
        "risk_score": risk_score,
        "reason": reason,
        "reasoning": reasoning,
        "executed": executed,
        "message": message,
        "ts": ts,
    }
    existing = _read_json(ACTIONS_PATH)
    history = existing.get("history", [])
    history.append(latest)
    if len(history) > MAX_ACTIONS_HISTORY:
        history = history[-MAX_ACTIONS_HISTORY:]
    _write_json(ACTIONS_PATH, {"latest": latest, "history": history})
