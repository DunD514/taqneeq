"""
Payment traffic simulator with highly varied, realistic conditions.
Time-based state machine per issuer; method-specific behavior; retry storms;
traffic shape variation; random noise. Output format unchanged for agent.
"""
import random
import time
from enum import Enum
from typing import Iterator

from models import (
    ErrorCode,
    PaymentEvent,
    PaymentOutcome,
)

# Default issuers and methods
DEFAULT_ISSUERS = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK"]
DEFAULT_METHODS = ["card", "upi", "netbanking", "wallet"]
ERROR_CODES = [e for e in ErrorCode if e != ErrorCode.NONE]

# ---------------- ADDITIVE (Phase 1): Merchant universe ----------------
DEFAULT_MERCHANTS = [
    "m_smb_001",
    "m_smb_002",
    "m_mid_001",
    "m_ent_001",
    "m_ent_002",
]

# ---------------- ADDITIVE (Phase 1): Cost model ----------------
COST_PER_ATTEMPT = {
    "card": 0.015,
    "upi": 0.002,
    "netbanking": 0.010,
    "wallet": 0.008,
}

# Issuer health state (time-based state machine)
class IssuerState(Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    SEVERELY_DEGRADED = "severely_degraded"
    RECOVERING = "recovering"


# Payment method latency/success modifiers (relative to base)
METHOD_LATENCY_MULT = {"card": 1.35, "upi": 0.75, "netbanking": 1.0, "wallet": 0.85}
METHOD_SUCCESS_BONUS = {"card": 0.02, "upi": 0.0, "netbanking": -0.01, "wallet": 0.03}
METHOD_RETRY_SENSITIVITY = {"card": 0.8, "upi": 1.5, "netbanking": 1.0, "wallet": 0.6}


class FailureMode:
    """Legacy hook: agent/executor can still inject or clear failure mode."""
    NONE = "none"
    ISSUER_DEGRADATION = "issuer_degradation"
    RETRY_STORM = "retry_storm"
    LATENCY_SPIKE = "latency_spike"
    # Synthetic escalation (demo-critical): multiple merchants failing, conflicting errors, unclear root cause
    MULTI_MERCHANT_ESCALATION = "multi_merchant_escalation"


class PaymentSimulator:
    """
    Simulates diverse, changing payment traffic: healthy periods, gradual issuer
    degradation, recovery, retry storms, bursty/quiet traffic, and small noise.
    Outputs PaymentEvent for the agent.
    """

    def __init__(
        self,
        issuers: list[str] | None = None,
        payment_methods: list[str] | None = None,
        merchants: list[str] | None = None,
        base_success_rate: float = 0.92,
        base_latency_p50: float = 120.0,
        base_latency_p95: float = 350.0,
        seed: int | None = None,
    ):
        self.issuers = issuers or DEFAULT_ISSUERS.copy()
        self.methods = payment_methods or DEFAULT_METHODS.copy()
        self.merchants = merchants or DEFAULT_MERCHANTS.copy()

        self.base_success_rate = base_success_rate
        self.base_latency_p50 = base_latency_p50
        self.base_latency_p95 = base_latency_p95

        self._rng = random.Random(seed)
        self._event_counter = 0
        self._start_time = time.time()

        # Per-issuer state machine
        self._issuer_state = {i: IssuerState.NORMAL for i in self.issuers}
        self._issuer_phase = {i: 0.0 for i in self.issuers}
        self._issuer_events_in_state = {i: 0 for i in self.issuers}
        self._issuer_min_duration = {i: 30 + self._rng.randint(0, 50) for i in self.issuers}
        self._issuer_max_duration = {i: 120 + self._rng.randint(0, 80) for i in self.issuers}

        # Retry storm
        self._retry_storm_phase = 0.0
        self._retry_storm_events_left = 0
        self._retry_storm_events_total = 0

        # Traffic shape
        self._traffic_regime = "normal"
        self._traffic_regime_events = 0
        self._traffic_regime_duration = 80 + self._rng.randint(0, 60)

        # Legacy failure hooks
        self._current_failure_mode = FailureMode.NONE
        self._failure_target_issuer = None
        self._failure_start_ts = 0.0
        self._post_action_latency_bump_remaining = 0
        self._latency_bump_multiplier = 2.0
        # Synthetic escalation: merchants to degrade (multiple failing post-migration)
        self._escalation_merchants: list[str] = []

        self._debug_log_state = False

    def set_failure_mode(self, mode: str, target_issuer: str | None = None) -> None:
        self._current_failure_mode = mode
        self._failure_target_issuer = target_issuer or (
            self._rng.choice(self.issuers) if mode != FailureMode.NONE else None
        )
        self._failure_start_ts = time.time()
        # Synthetic escalation: pick multiple merchants to fail (post-migration / unclear root cause)
        if mode == FailureMode.MULTI_MERCHANT_ESCALATION and self.merchants:
            n = min(3, len(self.merchants))
            self._escalation_merchants = self._rng.sample(self.merchants, n)
        else:
            self._escalation_merchants = []

    def clear_failure_mode(self) -> None:
        self._current_failure_mode = FailureMode.NONE
        self._failure_target_issuer = None

    def trigger_post_action_latency_bump(self, num_events: int = 80, multiplier: float = 2.0) -> None:
        self._post_action_latency_bump_remaining = num_events
        self._latency_bump_multiplier = multiplier

    def set_debug_log_state(self, on: bool = True) -> None:
        self._debug_log_state = on

    def _next_id(self) -> str:
        self._event_counter += 1
        return f"pay-{self._event_counter}-{self._rng.randint(10000, 99999)}"

    # ---------------- EXISTING LOGIC (UNCHANGED) ----------------
    # issuer state, retry storm, traffic, latency, retries, outcome
    # ALL METHODS BELOW ARE IDENTICAL TO YOUR PREVIOUS VERSION

    # ---------- Issuer state machine ----------
    def _issuer_success_modifier(self, issuer: str) -> float:
        s = self._issuer_state.get(issuer, IssuerState.NORMAL)
        p = self._issuer_phase.get(issuer, 0.0)
        if s == IssuerState.NORMAL:
            return 0.0
        if s == IssuerState.DEGRADED:
            return -0.15 - 0.25 * p
        if s == IssuerState.SEVERELY_DEGRADED:
            return -0.50 - 0.20 * p
        if s == IssuerState.RECOVERING:
            return -0.30 + 0.35 * p
        return 0.0

    def _issuer_latency_modifier(self, issuer: str) -> float:
        s = self._issuer_state.get(issuer, IssuerState.NORMAL)
        p = self._issuer_phase.get(issuer, 0.0)
        if s == IssuerState.NORMAL:
            return 1.0
        if s == IssuerState.DEGRADED:
            return 1.2 + 0.4 * p
        if s == IssuerState.SEVERELY_DEGRADED:
            return 1.8 + 0.5 * p
        if s == IssuerState.RECOVERING:
            return 1.4 - 0.4 * p
        return 1.0

    def _issuer_error_bias(self, issuer: str) -> list[ErrorCode]:
        s = self._issuer_state.get(issuer, IssuerState.NORMAL)
        if s in (IssuerState.DEGRADED, IssuerState.SEVERELY_DEGRADED):
            return [ErrorCode.ISSUER_UNAVAILABLE, ErrorCode.NETWORK_TIMEOUT, ErrorCode.RATE_LIMITED] + ERROR_CODES
        return ERROR_CODES

    def _advance_issuer_state(self, issuer: str) -> None:
        self._issuer_events_in_state[issuer] += 1
        n = self._issuer_events_in_state[issuer]
        min_d = self._issuer_min_duration[issuer]
        max_d = self._issuer_max_duration[issuer]
        if n < min_d:
            return
        self._issuer_phase[issuer] = min(1.0, (n - min_d) / max(1, max_d - min_d))
        if n < max_d and self._rng.random() > 0.08:
            return
        s = self._issuer_state[issuer]
        next_states = {
            IssuerState.NORMAL: [IssuerState.NORMAL, IssuerState.DEGRADED],
            IssuerState.DEGRADED: [IssuerState.DEGRADED, IssuerState.SEVERELY_DEGRADED, IssuerState.RECOVERING],
            IssuerState.SEVERELY_DEGRADED: [IssuerState.SEVERELY_DEGRADED, IssuerState.RECOVERING],
            IssuerState.RECOVERING: [IssuerState.RECOVERING, IssuerState.NORMAL],
        }
        weights = {
            IssuerState.NORMAL: (0.85, 0.15),
            IssuerState.DEGRADED: (0.55, 0.28, 0.17),
            IssuerState.SEVERELY_DEGRADED: (0.45, 0.55),
            IssuerState.RECOVERING: (0.65, 0.35),
        }
        self._issuer_state[issuer] = self._rng.choices(
            next_states[s], weights=weights[s], k=1
        )[0]
        self._issuer_phase[issuer] = 0.0
        self._issuer_events_in_state[issuer] = 0
        self._issuer_min_duration[issuer] = 25 + self._rng.randint(0, 60)
        self._issuer_max_duration[issuer] = 80 + self._rng.randint(0, 100)

    # ---------- Retry storm ----------
    def _advance_retry_storm(self) -> None:
        if self._retry_storm_events_left > 0:
            self._retry_storm_events_left -= 1
            return
        if self._retry_storm_phase == 0.0 and self._event_counter > 150:
            if self._rng.random() < 0.018:
                self._retry_storm_phase = 1.0
                self._retry_storm_events_total = 80 + self._rng.randint(0, 80)
                self._retry_storm_events_left = self._retry_storm_events_total
        elif self._retry_storm_phase >= 1.0 and self._retry_storm_events_left <= 0:
            self._retry_storm_phase = 0.0

    def _retry_storm_retries(self) -> int:
        if self._retry_storm_phase == 0.0:
            return 0
        progress = 1.0 - (self._retry_storm_events_left / max(1, self._retry_storm_events_total))
        if progress < 0.4:
            return self._rng.randint(1, 4)
        if progress < 0.75:
            return self._rng.randint(2, 6)
        return self._rng.randint(3, 8)

    def _retry_storm_success_inflation(self) -> float:
        if self._retry_storm_phase == 0.0:
            return 0.0
        progress = 1.0 - (self._retry_storm_events_left / max(1, self._retry_storm_events_total))
        if progress < 0.35:
            return 0.05
        if progress < 0.7:
            return -0.25
        return -0.40

    def _retry_storm_latency_mult(self) -> float:
        if self._retry_storm_phase == 0.0:
            return 1.0
        progress = 1.0 - (self._retry_storm_events_left / max(1, self._retry_storm_events_total))
        return 1.0 + 0.5 * progress

    # ---------- Traffic ----------
    def _advance_traffic_regime(self) -> None:
        self._traffic_regime_events += 1
        if self._traffic_regime_events >= self._traffic_regime_duration:
            self._traffic_regime_events = 0
            self._traffic_regime_duration = 50 + self._rng.randint(0, 100)
            self._traffic_regime = self._rng.choices(
                ["normal", "burst", "quiet", "spike"],
                weights=[0.5, 0.2, 0.2, 0.1],
                k=1,
            )[0]

    def _interval_sec(self, base: float = 0.03) -> float:
        if self._traffic_regime == "burst":
            return base * (0.3 + self._rng.uniform(0, 0.4))
        if self._traffic_regime == "quiet":
            return base * (2.0 + self._rng.uniform(0, 1.5))
        if self._traffic_regime == "spike":
            return base * (0.05 + self._rng.uniform(0, 0.1))
        return base * (0.8 + self._rng.uniform(0, 0.6))

    # ---------- Noise ----------
    def _noise_latency(self, raw_ms: float) -> float:
        return max(10.0, raw_ms * (1.0 + self._rng.uniform(-0.06, 0.06)))

    def _noise_success(self, prob: float) -> float:
        return max(0.0, min(1.0, prob + self._rng.uniform(-0.025, 0.025)))

    # ---------- Latency / retries / outcome ----------
    def _latency_ms(self, issuer: str, method: str) -> float:
        base_p50 = self.base_latency_p50
        base_p95 = self.base_latency_p95
        issuer_mult = self._issuer_latency_modifier(issuer)
        method_mult = METHOD_LATENCY_MULT.get(method, 1.0)
        storm_mult = self._retry_storm_latency_mult()
        if self._current_failure_mode == FailureMode.LATENCY_SPIKE:
            storm_mult *= 2.0
        if self._post_action_latency_bump_remaining > 0:
            self._post_action_latency_bump_remaining -= 1
            storm_mult *= self._latency_bump_multiplier
        p95 = base_p95 * issuer_mult * method_mult * storm_mult
        p50 = base_p50 * issuer_mult * method_mult * storm_mult
        u = self._rng.random()
        raw = (
            p50 + self._rng.uniform(0, (p95 - p50) * 0.5)
            if u < 0.92
            else p95 + self._rng.uniform(0, p95 * 0.25)
        )
        return self._noise_latency(raw)

    def _retries(self, issuer: str, method: str) -> int:
        base = self._rng.randint(1, 2) if self._rng.random() < 0.12 else 0
        storm_retries = self._retry_storm_retries()
        issuer_stress = self._rng.randint(0, 2) if self._issuer_state[issuer] != IssuerState.NORMAL else 0
        total = base + int(storm_retries * METHOD_RETRY_SENSITIVITY.get(method, 1.0)) + issuer_stress
        return max(0, min(8, total))

    def _outcome_and_error(self, issuer: str, method: str, merchant_id: str | None = None):
        prob = (
            self.base_success_rate
            + self._issuer_success_modifier(issuer)
            + METHOD_SUCCESS_BONUS.get(method, 0.0)
            + self._retry_storm_success_inflation()
        )
        if self._current_failure_mode == FailureMode.ISSUER_DEGRADATION and issuer == self._failure_target_issuer:
            prob -= 0.45
        if self._current_failure_mode == FailureMode.RETRY_STORM:
            prob -= 0.30
        # Synthetic escalation: multiple merchants failing (post-migration; conflicting error patterns)
        if self._current_failure_mode == FailureMode.MULTI_MERCHANT_ESCALATION and merchant_id and merchant_id in self._escalation_merchants:
            prob -= 0.40  # multiple merchants degraded -> forces human handover
        prob = self._noise_success(max(0.05, min(0.98, prob)))
        if self._rng.random() < prob:
            return PaymentOutcome.SUCCESS, ErrorCode.NONE
        # Conflicting error patterns: when in escalation, spread errors across codes (unclear root cause)
        if self._current_failure_mode == FailureMode.MULTI_MERCHANT_ESCALATION and merchant_id and merchant_id in self._escalation_merchants:
            return PaymentOutcome.FAILED, self._rng.choice(ERROR_CODES)
        return PaymentOutcome.FAILED, self._rng.choice(self._issuer_error_bias(issuer))

    # ---------- Event ----------
    def generate_one(self) -> PaymentEvent:
        issuer = self._rng.choice(self.issuers)
        self._advance_issuer_state(issuer)
        self._advance_retry_storm()
        self._advance_traffic_regime()

        method = self._rng.choice(self.methods)
        merchant_id = self._rng.choice(self.merchants)

        latency = self._latency_ms(issuer, method)
        retries = self._retries(issuer, method)
        outcome, error_code = self._outcome_and_error(issuer, method, merchant_id)
        ts = time.time()

        total_attempts = 1 + retries
        cost_per_attempt = COST_PER_ATTEMPT.get(method, 0.01)
        estimated_cost = total_attempts * cost_per_attempt

        if self._debug_log_state:
            states = {i: self._issuer_state[i].value for i in self.issuers}
            print(
                f"[SimState] issuers={states} "
                f"retry_storm_phase={self._retry_storm_phase:.2f} "
                f"traffic={self._traffic_regime}"
            )

        return PaymentEvent(
            event_id=self._next_id(),
            issuer_bank=issuer,
            payment_method=method,
            merchant_id=merchant_id,
            latency_ms=latency,
            retries=retries,
            outcome=outcome,
            error_code=error_code,
            timestamp=ts,
            total_attempts=total_attempts,
            retry_amplification_factor=total_attempts,
            estimated_cost=estimated_cost,
            cost_per_attempt=cost_per_attempt,
        )

    def stream(self, interval_sec: float = 0.03, max_events: int | None = None) -> Iterator[PaymentEvent]:
        emitted = 0
        while max_events is None or emitted < max_events:
            yield self.generate_one()
            emitted += 1
            time.sleep(max(0.001, self._interval_sec(base=interval_sec)))
