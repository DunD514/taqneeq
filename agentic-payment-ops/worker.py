#!/usr/bin/env python3
"""
Background worker: runs payment simulation and agent in an infinite loop.
Continuously generates payment events, feeds them to the agent, and updates
shared state (state/*.json) for the dashboard. Run separately from the dashboard:

  python worker.py
  streamlit run dashboard.py

No fixed event limit; failures are injected dynamically by the simulator.
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from agent import create_agent


def main() -> None:
    print("Payment Ops Worker (continuous)")
    print("===============================")
    print("Generating payments and running agent. State written to state/*.json")
    print("Start the dashboard in another terminal: streamlit run dashboard.py")
    print("Press Ctrl+C to stop.")
    print("-" * 60)

    agent = create_agent(
        window_size=200,
        window_advance=50,
        cycle_interval_events=50,
    )
    simulator = agent.simulator
    stream = simulator.stream(interval_sec=0.15, max_events=None)
    agent.run(
        event_stream=stream,
        max_events=None,
        failure_injection_after_events=None,
    )


if __name__ == "__main__":
    main()
