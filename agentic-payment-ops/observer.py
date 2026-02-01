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

    def _success_rate_by_merchant(self, events: list[PaymentEvent]) -> dict[str, float]:
        """Derived: success rate per merchant (only events with merchant_id)."""
        by_merchant: dict[str, list[bool]] = defaultdict(list)
        for e in events:
            mid = getattr(e, "merchant_id", None)
            if mid:
                by_merchant[mid].append(e.outcome == PaymentOutcome.SUCCESS)
        return {m: sum(v) / len(v) if v else 0.0 for m, v in by_merchant.items()}

    def _attempt_amplification_by_merchant(self, events: list[PaymentEvent]) -> dict[str, float]:
        """Derived: retries per attempt by merchant (only events with merchant_id)."""
        by_merchant: dict[str, list[float]] = defaultdict(list)
        for e in events:
            mid = getattr(e, "merchant_id", None)
            if mid is not None and getattr(e, "retry_amplification_factor", None) is not None:
                by_merchant[mid].append(e.retry_amplification_factor)
        return {
            m: sum(v) / len(v) if v else 0.0
            for m, v in by_merchant.items()
        }

    def _average_cost_by_merchant(self, events: list[PaymentEvent]) -> dict[str, float]:
        """Derived: average estimated_cost per merchant (only events with estimated_cost)."""
        by_merchant: dict[str, list[float]] = defaultdict(list)
        for e in events:
            mid = getattr(e, "merchant_id", None)
            cost = getattr(e, "estimated_cost", None)
            if mid is not None and cost is not None:
                by_merchant[mid].append(cost)
        return {m: sum(v) / len(v) if v else 0.0 for m, v in by_merchant.items()}

    def _average_estimated_cost(self, events: list[PaymentEvent]) -> float | None:
        """Derived: window-level average estimated cost."""
        costs = [e.estimated_cost for e in events if getattr(e, "estimated_cost", None) is not None]
        if not costs:
            return None
        return sum(costs) / len(costs)

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
            success_rate_by_merchant=self._success_rate_by_merchant(events),
            attempt_amplification_by_merchant=self._attempt_amplification_by_merchant(events),
            average_cost_by_merchant=self._average_cost_by_merchant(events),
            average_estimated_cost=self._average_estimated_cost(events),
        )

    def advance(self) -> None:
        """
        Advance window by dropping oldest events (optional; can also
        keep full buffer and always compute on last N).
        """
        if len(self._buffer) >= self.window_advance_events:
            self._buffer = self._buffer[self.window_advance_events :]
