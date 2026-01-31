"""
Observation layer: sliding window state and feature extraction.
Computes success rate by issuer, retry amplification, p95 latency, error distribution.
"""
import time
from collections import defaultdict
from typing import Optional

from models import ErrorCode, PaymentEvent, PaymentOutcome, WindowMetrics


class Observer:
    """
    Maintains a sliding window of payment events and computes
    aggregated metrics for the reasoner and decision engine.
    """

    def __init__(
        self,
        window_size: int = 200,
        window_advance_events: int = 50,
    ):
        self.window_size = window_size
        self.window_advance_events = window_advance_events
        self._buffer: list[PaymentEvent] = []
        self._window_counter = 0

    def ingest(self, event: PaymentEvent) -> None:
        """Append one event; keep buffer at most window_size (FIFO)."""
        self._buffer.append(event)
        if len(self._buffer) > self.window_size:
            self._buffer = self._buffer[-self.window_size :]

    def ready(self) -> bool:
        """True if we have enough events to emit a window."""
        return len(self._buffer) >= self.window_advance_events

    def _latency_p95(self, events: list[PaymentEvent]) -> float:
        if not events:
            return 0.0
        sorted_lat = sorted(e.latency_ms for e in events)
        idx = max(0, int(len(sorted_lat) * 0.95) - 1)
        return sorted_lat[idx]

    def _success_rate(self, events: list[PaymentEvent]) -> float:
        if not events:
            return 0.0
        ok = sum(1 for e in events if e.outcome == PaymentOutcome.SUCCESS)
        return ok / len(events)

    def _retry_amplification(self, events: list[PaymentEvent]) -> float:
        """Retries per attempt (total retries / count)."""
        if not events:
            return 0.0
        total_retries = sum(e.retries for e in events)
        return total_retries / len(events)

    def _error_distribution(self, events: list[PaymentEvent]) -> dict[str, float]:
        counts: dict[str, int] = defaultdict(int)
        for e in events:
            counts[e.error_code.value] = counts[e.error_code.value] + 1
        n = len(events)
        return {k: v / n for k, v in counts.items()} if n else {}

    def _success_rate_by_issuer(self, events: list[PaymentEvent]) -> dict[str, float]:
        by_issuer: dict[str, list[bool]] = defaultdict(list)
        for e in events:
            by_issuer[e.issuer_bank].append(e.outcome == PaymentOutcome.SUCCESS)
        return {
            issuer: sum(v) / len(v) if v else 0.0
            for issuer, v in by_issuer.items()
        }

    def get_current_metrics(self) -> Optional[WindowMetrics]:
        """
        Compute metrics over the current buffer (sliding window).
        Does not consume the buffer; same window can be read again.
        """
        if not self._buffer:
            return None
        events = self._buffer[-self.window_size :]
        self._window_counter += 1
        wid = f"w-{self._window_counter}"
        start_ts = min(e.timestamp for e in events)
        end_ts = max(e.timestamp for e in events)
        return WindowMetrics(
            window_id=wid,
            start_ts=start_ts,
            end_ts=end_ts,
            success_rate=self._success_rate(events),
            p95_latency_ms=self._latency_p95(events),
            retry_amplification=self._retry_amplification(events),
            error_distribution=self._error_distribution(events),
            success_rate_by_issuer=self._success_rate_by_issuer(events),
            sample_count=len(events),
        )

    def advance(self) -> None:
        """
        Advance window by dropping oldest events (optional; can also
        keep full buffer and always compute on last N).
        """
        if len(self._buffer) >= self.window_advance_events:
            self._buffer = self._buffer[self.window_advance_events :]
