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
from merchant_profiles import DEFAULT_MERCHANT_PROFILES, get_merchant_ids

# Default issuers and methods
DEFAULT_ISSUERS = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK"]
DEFAULT_METHODS = ["card", "upi", "netbanking", "wallet"]
ERROR_CODES = [e for e in ErrorCode if e != ErrorCode.NONE]

# Cost per attempt by payment method (reality-grounded; used for estimated_cost)
COST_PER_ATTEMPT_BY_METHOD: dict[str, float] = {
    "card": 0.025,
    "upi": 0.008,
    "netbanking": 0.015,
    "wallet": 0.010,
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
# UPI: more retries when system is stressed; wallet: fewer retries unless issuer down
METHOD_RETRY_SENSITIVITY = {"card": 0.8, "upi": 1.5, "netbanking": 1.0, "wallet": 0.6}


class FailureMode:
    """Legacy hook: agent/executor can still inject or clear failure mode."""
    NONE = "none"
    ISSUER_DEGRADATION = "issuer_degradation"
    RETRY_STORM = "retry_storm"
    LATENCY_SPIKE = "latency_spike"


class PaymentSimulator:
    """
    Simulates diverse, changing payment traffic: healthy periods, gradual issuer
    degradation, recovery, retry storms, bursty/quiet traffic, and small noise.
    Outputs PaymentEvent (event_id, issuer_bank, payment_method, latency_ms,
    retries, outcome, error_code, timestamp) for the agent.
    """

    def __init__(
        self,
        issuers: list[str] | None = None,
        payment_methods: list[str] | None = None,
        base_success_rate: float = 0.92,
        base_latency_p50: float = 120.0,
        base_latency_p95: float = 350.0,
        seed: int | None = None,
    ):
        self.issuers = issuers or DEFAULT_ISSUERS.copy()
        self.methods = payment_methods or DEFAULT_METHODS.copy()
        self.base_success_rate = base_success_rate
        self.base_latency_p50 = base_latency_p50
        self.base_latency_p95 = base_latency_p95
        self._rng = random.Random(seed)
        self._event_counter = 0
        self._start_time = time.time()

        # Per-issuer state machine: state + phase (0..1) for gradual ramp
        self._issuer_state: dict[str, IssuerState] = {i: IssuerState.NORMAL for i in self.issuers}
        self._issuer_phase: dict[str, float] = {i: 0.0 for i in self.issuers}
        self._issuer_events_in_state: dict[str, int] = {i: 0 for i in self.issuers}
        self._issuer_min_duration: dict[str, int] = {i: 30 + self._rng.randint(0, 50) for i in self.issuers}
        self._issuer_max_duration: dict[str, int] = {i: 120 + self._rng.randint(0, 80) for i in self.issuers}

        # Global retry-storm: phases (0=off, 1=building, 2=inflated success + high retries, 3=collapse)
        self._retry_storm_phase: float = 0.0  # 0..1 for building, then 2=active, 3=collapse
        self._retry_storm_events_left: int = 0
        self._retry_storm_events_total: int = 0

        # Traffic shape: current regime (burst / normal / quiet / spike)
        self._traffic_regime = "normal"
        self._traffic_regime_events: int = 0
        self._traffic_regime_duration: int = 80 + self._rng.randint(0, 60)

        # Legacy: agent-injected failure and post-action latency bump (unchanged API)
        self._current_failure_mode = FailureMode.NONE
        self._failure_target_issuer: str | None = None
        self._failure_start_ts: float = 0.0
        self._post_action_latency_bump_remaining: int = 0
        self._latency_bump_multiplier: float = 2.0

        # Observability
        self._debug_log_state = False

        # Merchant dimension: weighted by volume_factor for selection
        self._merchant_profiles = DEFAULT_MERCHANT_PROFILES
        self._merchant_weights = [p.volume_factor for p in self._merchant_profiles]

    def set_failure_mode(self, mode: str, target_issuer: str | None = None) -> None:
        """Legacy: agent can force a failure mode (still honored)."""
        self._current_failure_mode = mode
        self._failure_target_issuer = target_issuer or (
            self._rng.choice(self.issuers) if mode != FailureMode.NONE else None
        )
        self._failure_start_ts = time.time()

    def clear_failure_mode(self) -> None:
        """Legacy: turn off agent-injected failure."""
        self._current_failure_mode = FailureMode.NONE
        self._failure_target_issuer = None

    def trigger_post_action_latency_bump(self, num_events: int = 80, multiplier: float = 2.0) -> None:
        """Legacy: executor triggers latency bump for rollback demo."""
        self._post_action_latency_bump_remaining = num_events
        self._latency_bump_multiplier = multiplier

    def set_debug_log_state(self, on: bool = True) -> None:
        """Optional: log current issuer states when generating events."""
        self._debug_log_state = on

    def _next_id(self) -> str:
        self._event_counter += 1
        return f"pay-{self._event_counter}-{self._rng.randint(10000, 99999)}"

    # ---------- Issuer state machine (probabilistic transitions) ----------
    def _issuer_success_modifier(self, issuer: str) -> float:
        """Success rate modifier from issuer state and phase (gradual ramp)."""
        s = self._issuer_state.get(issuer, IssuerState.NORMAL)
        p = self._issuer_phase.get(issuer, 0.0)
        if s == IssuerState.NORMAL:
            return 0.0
        if s == IssuerState.DEGRADED:
            return -0.15 - 0.25 * p  # ramp from -0.15 to -0.40
        if s == IssuerState.SEVERELY_DEGRADED:
            return -0.50 - 0.20 * p  # -0.50 to -0.70
        if s == IssuerState.RECOVERING:
            return -0.30 + 0.35 * p  # ramp from -0.30 toward +0.05
        return 0.0

    def _issuer_latency_modifier(self, issuer: str) -> float:
        """Latency multiplier from issuer state and phase."""
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
        """When failed, bias error codes by issuer state."""
        s = self._issuer_state.get(issuer, IssuerState.NORMAL)
        if s in (IssuerState.DEGRADED, IssuerState.SEVERELY_DEGRADED):
            return [ErrorCode.ISSUER_UNAVAILABLE, ErrorCode.NETWORK_TIMEOUT, ErrorCode.RATE_LIMITED] + ERROR_CODES
        return ERROR_CODES

    def _advance_issuer_state(self, issuer: str) -> None:
        """Probabilistic transition after min/max duration in state."""
        self._issuer_events_in_state[issuer] = self._issuer_events_in_state.get(issuer, 0) + 1
        n = self._issuer_events_in_state[issuer]
        min_d = self._issuer_min_duration.get(issuer, 40)
        max_d = self._issuer_max_duration.get(issuer, 150)
        if n < min_d:
            return
        # Phase advances within state (for gradual ramp)
        self._issuer_phase[issuer] = min(1.0, (n - min_d) / max(1, max_d - min_d))
        if n < max_d and self._rng.random() > 0.08:
            return
        # Transition
        s = self._issuer_state.get(issuer, IssuerState.NORMAL)
        next_states = {
            IssuerState.NORMAL: [IssuerState.NORMAL, IssuerState.DEGRADED],
            IssuerState.DEGRADED: [IssuerState.DEGRADED, IssuerState.SEVERELY_DEGRADED, IssuerState.RECOVERING],
            IssuerState.SEVERELY_DEGRADED: [IssuerState.SEVERELY_DEGRADED, IssuerState.RECOVERING],
            IssuerState.RECOVERING: [IssuerState.RECOVERING, IssuerState.NORMAL],
        }
        # Demo pacing: meaningful chance of degradation and recovery in 2–3 min run
        weights = {
            IssuerState.NORMAL: (0.85, 0.15),
            IssuerState.DEGRADED: (0.55, 0.28, 0.17),
            IssuerState.SEVERELY_DEGRADED: (0.45, 0.55),
            IssuerState.RECOVERING: (0.65, 0.35),
        }
        candidates = next_states[s]
        w = weights[s][: len(candidates)]
        self._issuer_state[issuer] = self._rng.choices(candidates, weights=w, k=1)[0]
        self._issuer_phase[issuer] = 0.0
        self._issuer_events_in_state[issuer] = 0
        self._issuer_min_duration[issuer] = 25 + self._rng.randint(0, 60)
        self._issuer_max_duration[issuer] = 80 + self._rng.randint(0, 100)

    # ---------- Retry storm (build -> inflated success + high retries -> collapse) ----------
    def _advance_retry_storm(self) -> None:
        """Drive retry storm from global event count (demo pacing)."""
        if self._retry_storm_events_left > 0:
            self._retry_storm_events_left -= 1
            return
        # Probabilistically start a retry storm (higher chance after many events)
        if self._retry_storm_phase == 0.0 and self._event_counter > 150:
            if self._rng.random() < 0.018:
                self._retry_storm_phase = 1.0
                self._retry_storm_events_total = 80 + self._rng.randint(0, 80)
                self._retry_storm_events_left = self._retry_storm_events_total
        elif self._retry_storm_phase >= 1.0 and self._retry_storm_events_left <= 0:
            self._retry_storm_phase = 0.0

    def _retry_storm_retries(self) -> int:
        """Extra retries during retry storm (phase 2: inflated; phase 3: collapse)."""
        if self._retry_storm_phase == 0.0:
            return 0
        progress = 1.0 - (self._retry_storm_events_left / max(1, self._retry_storm_events_total))
        if progress < 0.4:
            return self._rng.randint(1, 4)
        if progress < 0.75:
            return self._rng.randint(2, 6)
        return self._rng.randint(3, 8)

    def _retry_storm_success_inflation(self) -> float:
        """Temporary success inflation in mid phase (false recovery), then drop."""
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

    # ---------- Traffic shape (burst / normal / quiet / spike) ----------
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
        """Next event delay: burst=fast, quiet=slow, spike=very fast."""
        if self._traffic_regime == "burst":
            return base * (0.3 + self._rng.uniform(0, 0.4))
        if self._traffic_regime == "quiet":
            return base * (2.0 + self._rng.uniform(0, 1.5))
        if self._traffic_regime == "spike":
            return base * (0.05 + self._rng.uniform(0, 0.1))
        return base * (0.8 + self._rng.uniform(0, 0.6))

    # ---------- Payment method behavior ----------
    def _method_latency_mult(self, method: str) -> float:
        return METHOD_LATENCY_MULT.get(method, 1.0)

    def _method_success_bonus(self, method: str) -> float:
        return METHOD_SUCCESS_BONUS.get(method, 0.0)

    def _method_retry_sensitivity(self, method: str) -> float:
        return METHOD_RETRY_SENSITIVITY.get(method, 1.0)

    # ---------- Noise (small fluctuations) ----------
    def _noise_latency(self, raw_ms: float) -> float:
        jitter = 1.0 + self._rng.uniform(-0.06, 0.06)
        return max(10.0, raw_ms * jitter)

    def _noise_success(self, prob: float) -> float:
        return max(0.0, min(1.0, prob + self._rng.uniform(-0.025, 0.025)))

    # ---------- Event generation ----------
    def _latency_ms(self, issuer: str, method: str) -> float:
        base_p50 = self.base_latency_p50
        base_p95 = self.base_latency_p95
        issuer_mult = self._issuer_latency_modifier(issuer)
        method_mult = self._method_latency_mult(method)
        storm_mult = self._retry_storm_latency_mult()
        if self._current_failure_mode == FailureMode.LATENCY_SPIKE:
            storm_mult *= 2.0
        if self._post_action_latency_bump_remaining > 0:
            self._post_action_latency_bump_remaining -= 1
            storm_mult *= self._latency_bump_multiplier
        p95 = base_p95 * issuer_mult * method_mult * storm_mult
        p50 = base_p50 * issuer_mult * method_mult * storm_mult
        u = self._rng.random()
        if u < 0.92:
            raw = p50 + self._rng.uniform(0, (p95 - p50) * 0.5)
        else:
            raw = p95 + self._rng.uniform(0, p95 * 0.25)
        return self._noise_latency(raw)

    def _retries(self, issuer: str, method: str) -> int:
        base = 0
        if self._rng.random() < 0.12:
            base = self._rng.randint(1, 2)
        storm_retries = self._retry_storm_retries()
        sensitivity = self._method_retry_sensitivity(method)
        issuer_stress = 0
        s = self._issuer_state.get(issuer, IssuerState.NORMAL)
        if s in (IssuerState.DEGRADED, IssuerState.SEVERELY_DEGRADED):
            issuer_stress = self._rng.randint(0, 2)
        total = base + int(storm_retries * sensitivity) + issuer_stress
        return max(0, min(8, total))

    def _outcome_and_error(self, issuer: str, method: str) -> tuple[PaymentOutcome, ErrorCode]:
        prob = self.base_success_rate
        prob += self._issuer_success_modifier(issuer)
        prob += self._method_success_bonus(method)
        prob += self._retry_storm_success_inflation()
        if self._current_failure_mode == FailureMode.ISSUER_DEGRADATION and issuer == self._failure_target_issuer:
            prob -= 0.45
        if self._current_failure_mode == FailureMode.RETRY_STORM:
            prob -= 0.30
        prob = self._noise_success(prob)
        prob = max(0.05, min(0.98, prob))
        if self._rng.random() < prob:
            return PaymentOutcome.SUCCESS, ErrorCode.NONE
        bias = self._issuer_error_bias(issuer)
        return PaymentOutcome.FAILED, self._rng.choice(bias)

    def _pick_merchant(self) -> str:
        """Pick merchant weighted by volume_factor."""
        if not self._merchant_profiles:
            return "M-DEFAULT"
        chosen = self._rng.choices(
            self._merchant_profiles,
            weights=self._merchant_weights,
            k=1,
        )[0]
        return chosen.merchant_id

    def _estimated_cost(self, method: str, total_attempts: int) -> float:
        """Cost model: cost per attempt * total attempts."""
        base = COST_PER_ATTEMPT_BY_METHOD.get(method, 0.01)
        return base * total_attempts

    def generate_one(self) -> PaymentEvent:
        """Emit one payment event; advance state machine and traffic regime."""
        # Advance state (one issuer per event to keep it stochastic)
        issuer = self._rng.choice(self.issuers)
        self._advance_issuer_state(issuer)
        self._advance_retry_storm()
        self._advance_traffic_regime()

        method = self._rng.choice(self.methods)
        latency = self._latency_ms(issuer, method)
        retries = self._retries(issuer, method)
        outcome, error_code = self._outcome_and_error(issuer, method)
        ts = time.time()

        # Merchant dimension and cost (additive)
        merchant_id = self._pick_merchant()
        total_attempts = 1 + retries
        retry_amplification_factor = (retries / 1.0) if total_attempts >= 1 else 0.0
        cost_per_attempt = COST_PER_ATTEMPT_BY_METHOD.get(method, 0.01)
        estimated_cost = self._estimated_cost(method, total_attempts)

        if self._debug_log_state:
            states = {i: self._issuer_state[i].value for i in self.issuers}
            print(f"[SimState] issuers={states} retry_storm_phase={self._retry_storm_phase:.2f} traffic={self._traffic_regime}")

        return PaymentEvent(
            event_id=self._next_id(),
            issuer_bank=issuer,
            payment_method=method,
            latency_ms=latency,
            retries=retries,
            outcome=outcome,
            error_code=error_code,
            timestamp=ts,
            merchant_id=merchant_id,
            total_attempts=total_attempts,
            retry_amplification_factor=retry_amplification_factor,
            estimated_cost=estimated_cost,
            cost_per_attempt=cost_per_attempt,
        )

    def stream(
        self,
        interval_sec: float = 0.03,
        max_events: int | None = None,
    ) -> Iterator[PaymentEvent]:
        """
        Stream payment events. interval_sec is the base delay; actual spacing
        varies with traffic regime (burst/quiet/spike).
        """
        emitted = 0
        while max_events is None or emitted < max_events:
            yield self.generate_one()
            emitted += 1
            delay = self._interval_sec(base=interval_sec)
            time.sleep(max(0.001, delay))
