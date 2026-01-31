#!/usr/bin/env python3
"""
Runnable entry point for the agentic payment operations system.

Run: python run.py
Dashboard: streamlit run dashboard.py  (in a separate terminal)

Set GEMINI_API_KEY for Gemini 2.5 Flash (optional; heuristics used if unset).
State is written to state/*.json for the operator dashboard.
"""
import sys
from pathlib import Path

# Ensure package root is on path when running as script
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from agent import create_agent
from simulator import FailureMode, PaymentSimulator


def main() -> None:
    print("Agentic Payment Operations Manager")
    print("==================================")
    agent = create_agent(
        window_size=200,
        window_advance=50,
        cycle_interval_events=50,
    )
    simulator = agent.simulator
    stream = simulator.stream(interval_sec=0.02, max_events=500)
    agent.run(
        event_stream=stream,
        max_events=500,
        failure_injection_after_events=120,
        failure_mode=FailureMode.ISSUER_DEGRADATION,
    )


if __name__ == "__main__":
    main()
