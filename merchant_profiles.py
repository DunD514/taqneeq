"""
Merchant profiles for realism and grounding.

This module defines merchant-level characteristics that influence
payment behavior, retry tolerance, and cost sensitivity.

IMPORTANT:
- Additive only
- No decisions are made here
- Used by simulator and observer for context
"""

from dataclasses import dataclass
import random
from typing import Dict


@dataclass
class MerchantProfile:
    """
    Represents a merchant's operational characteristics.
    """
    merchant_id: str

    # Business characteristics
    tier: str                  # smb | mid | enterprise
    primary_methods: list[str] # preferred payment methods
    avg_ticket_size: float     # average order value

    # Operational sensitivities
    retry_tolerance: float     # higher = more retries acceptable
    cost_sensitivity: float   # higher = more sensitive to processing cost
    latency_sensitivity: float  # higher = more sensitive to slow checkout


class MerchantRegistry:
    """
    Registry of merchants participating in the system.
    Acts as a lightweight context provider for the agent.
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._merchants: Dict[str, MerchantProfile] = {}
        self._load_default_merchants()

    def _load_default_merchants(self) -> None:
        """
        Preload a diverse but small merchant set.
        This is enough for hackathon realism.
        """
        presets = [
            # SMB merchants
            MerchantProfile(
                merchant_id="m_smb_001",
                tier="smb",
                primary_methods=["upi", "wallet"],
                avg_ticket_size=450,
                retry_tolerance=1.8,
                cost_sensitivity=1.5,
                latency_sensitivity=1.3,
            ),
            MerchantProfile(
                merchant_id="m_smb_002",
                tier="smb",
                primary_methods=["upi"],
                avg_ticket_size=300,
                retry_tolerance=2.0,
                cost_sensitivity=1.7,
                latency_sensitivity=1.2,
            ),

            # Mid-market merchants
            MerchantProfile(
                merchant_id="m_mid_001",
                tier="mid",
                primary_methods=["card", "upi"],
                avg_ticket_size=1200,
                retry_tolerance=1.3,
                cost_sensitivity=1.0,
                latency_sensitivity=1.0,
            ),

            # Enterprise merchants
            MerchantProfile(
                merchant_id="m_ent_001",
                tier="enterprise",
                primary_methods=["card", "netbanking"],
                avg_ticket_size=3200,
                retry_tolerance=0.9,
                cost_sensitivity=0.7,
                latency_sensitivity=1.4,
            ),
            MerchantProfile(
                merchant_id="m_ent_002",
                tier="enterprise",
                primary_methods=["card"],
                avg_ticket_size=5000,
                retry_tolerance=0.8,
                cost_sensitivity=0.6,
                latency_sensitivity=1.5,
            ),
        ]

        for m in presets:
            self._merchants[m.merchant_id] = m

    def get(self, merchant_id: str) -> MerchantProfile | None:
        """Fetch a merchant profile safely."""
        return self._merchants.get(merchant_id)

    def random_merchant(self) -> MerchantProfile:
        """Return a random merchant (used by simulator)."""
        return self._rng.choice(list(self._merchants.values()))

    def all_merchants(self) -> Dict[str, MerchantProfile]:
        """Return all registered merchants."""
        return self._merchants.copy()
