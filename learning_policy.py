"""
Learning Policy Layer (Plan A)

Purpose:
- Translate historical action outcomes into soft decision biases
- Influence future decisions WITHOUT overriding core logic
- No ML, no retraining, no black box

This makes learning operational, not cosmetic.
"""

from collections import defaultdict
from typing import Dict

from models import ActionType, OutcomeRecord


class LearningPolicy:
    """
    Maintains lightweight performance priors for actions.
    These priors can bias risk or confidence in future decisions.
    """

    def __init__(self):
        # action_type -> counters
        self._stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"helped": 0, "hurt": 0, "neutral": 0}
        )

    # ---------------- Record outcomes ----------------
    def ingest_outcome(self, record: OutcomeRecord) -> None:
        """
        Update action performance statistics.
        """
        action_type = record.action.action_type

        if record.rollback_applied:
            self._stats[action_type]["hurt"] += 1
        elif record.helped:
            self._stats[action_type]["helped"] += 1
        else:
            self._stats[action_type]["neutral"] += 1

    # ---------------- Bias computation ----------------
    def risk_bias(self, action_type: str) -> float:
        """
        Returns a bias in [-0.15, +0.15] to be applied to risk score.
        Negative = safer (historically helped)
        Positive = riskier (historically hurt)
        """
        s = self._stats.get(action_type)
        if not s:
            return 0.0

        total = s["helped"] + s["hurt"] + s["neutral"]
        if total < 3:
            return 0.0  # not enough signal yet

        score = (s["hurt"] - s["helped"]) / max(1, total)

        # Clamp to safe range
        return max(-0.15, min(0.15, score * 0.2))

    # ---------------- Explainability ----------------
    def explain(self) -> str:
        """
        Human-readable explanation of what the agent has learned.
        """
        if not self._stats:
            return "No learning signals accumulated yet."

        lines = ["Learning policy summary:"]
        for action, s in self._stats.items():
            total = s["helped"] + s["hurt"] + s["neutral"]
            if total == 0:
                continue
            lines.append(
                f"- {action}: helped={s['helped']}, "
                f"hurt={s['hurt']}, neutral={s['neutral']}"
            )
        return "\n".join(lines)
