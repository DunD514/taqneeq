"""
Merchant universe for reality-grounded simulation.
Merchants differ in traffic shape (volume, burstiness) and sensitivity (latency, failure).
Used by the simulator to attach merchant_id and influence event generation.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class MerchantProfile:
    """Profile for a single merchant: id, traffic shape, sensitivity."""
    merchant_id: str
    # Traffic shape: relative volume (events per unit time)
    volume_factor: float  # 0.5 = quiet, 1.0 = normal, 2.0 = high
    # Burstiness: tendency to spike (0 = smooth, 1 = bursty)
    burstiness: float
    # Sensitivity: how much latency/failure affects this merchant (0 = low, 1 = high)
    latency_sensitivity: float
    failure_sensitivity: float


# Default merchant universe: mix of sizes and sensitivities
DEFAULT_MERCHANT_PROFILES: list[MerchantProfile] = [
    MerchantProfile("M-LARGE-001", volume_factor=1.8, burstiness=0.3, latency_sensitivity=0.4, failure_sensitivity=0.5),
    MerchantProfile("M-LARGE-002", volume_factor=1.5, burstiness=0.5, latency_sensitivity=0.5, failure_sensitivity=0.6),
    MerchantProfile("M-MID-001", volume_factor=1.0, burstiness=0.4, latency_sensitivity=0.6, failure_sensitivity=0.6),
    MerchantProfile("M-MID-002", volume_factor=1.0, burstiness=0.6, latency_sensitivity=0.5, failure_sensitivity=0.5),
    MerchantProfile("M-MID-003", volume_factor=0.9, burstiness=0.2, latency_sensitivity=0.7, failure_sensitivity=0.7),
    MerchantProfile("M-SMALL-001", volume_factor=0.6, burstiness=0.8, latency_sensitivity=0.8, failure_sensitivity=0.8),
    MerchantProfile("M-SMALL-002", volume_factor=0.5, burstiness=0.5, latency_sensitivity=0.7, failure_sensitivity=0.7),
    MerchantProfile("M-SMALL-003", volume_factor=0.5, burstiness=0.3, latency_sensitivity=0.6, failure_sensitivity=0.6),
]


def get_merchant_ids() -> list[str]:
    """Return list of merchant IDs for the simulator."""
    return [p.merchant_id for p in DEFAULT_MERCHANT_PROFILES]


def get_profile(merchant_id: str) -> Optional[MerchantProfile]:
    """Return profile for a merchant ID, or None."""
    for p in DEFAULT_MERCHANT_PROFILES:
        if p.merchant_id == merchant_id:
            return p
    return None
