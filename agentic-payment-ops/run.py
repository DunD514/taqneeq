#!/usr/bin/env python3
"""
Runnable entry point for the agentic payment operations system.

Run: python run.py
Dashboard: streamlit run dashboard.py  (in a separate terminal)

Structured phases: warm-up, failure injection, agent loop, post-run analysis.
Prints learning summary, action effectiveness, human-approval decisions, rollbacks.
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from agent import create_agent
from simulator import FailureMode, PaymentSimulator


def main() -> None:
    print("Agentic Payment Operations Manager")
    print("==================================")

    # Phase: setup
    print("\n[Phase] Warm-up: creating agent and simulator.")
    agent = create_agent(
        window_size=200,
        window_advance=50,
        cycle_interval_events=50,
    )
    simulator = agent.simulator
    stream = simulator.stream(interval_sec=0.02, max_events=500)

    # Phase: failure injection (handled inside agent.run)
    print("[Phase] Failure injection: will inject issuer degradation at event 120.")
    print("[Phase] Agent loop: Observe -> Reason -> Decide -> Act -> Learn\n")
    print("-" * 60)

    agent.run(
        event_stream=stream,
        max_events=500,
        failure_injection_after_events=120,
        failure_mode=FailureMode.ISSUER_DEGRADATION,
    )

    # Phase: post-run analysis
    print("-" * 60)
    print("\n[Phase] Post-run analysis")
    print("------------------------")
    summary = agent.learner.summarize_learning_heuristic()
    print("Learning summary:", summary)
    llm_summary = agent.learner.summarize_learning_llm()
    if llm_summary:
        print("Learning (LLM):", llm_summary)
    recent = agent.learner.get_recent_outcomes(10)
    helped = sum(1 for r in recent if r.helped and not r.rollback_applied)
    hurt = sum(1 for r in recent if r.rollback_applied)
    neutral = len(recent) - helped - hurt
    print("Action effectiveness (recent):", helped, "helped,", hurt, "rollbacks,", neutral, "neutral.")
    rollback_log = agent.executor.get_rollback_log()
    if rollback_log:
        print("Rollbacks:", len(rollback_log))
        for rb in rollback_log[-5:]:
            print("  -", rb)
    else:
        print("Rollbacks: none.")
    print("\nRun complete.")


if __name__ == "__main__":
    main()
