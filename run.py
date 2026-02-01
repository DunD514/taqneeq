#!/usr/bin/env python3
"""
Runnable entry point for the agentic payment operations system.

Run: python run.py
Dashboard: streamlit run dashboard.py  (optional, separate terminal)

Set GEMINI_API_KEY for Gemini 2.5 Flash (optional; heuristics used if unset).
"""
import sys
import time
from pathlib import Path

# Ensure package root is on path when running as script
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from agent import create_agent
from simulator import FailureMode


def main() -> None:
    print("\nAgentic Payment Operations Manager")
    print("==================================")
    print("Observe -> Reason -> Decide -> Act -> Learn\n")

    # ---------------- Create agent ----------------
    agent = create_agent(
        window_size=200,
        window_advance=50,
        cycle_interval_events=50,
    )

    simulator = agent.simulator

    # ---------------- Configure stream ----------------
    stream = simulator.stream(
        interval_sec=0.02,
        max_events=500,
    )

    # ---------------- Phase 1: Warm-up ----------------
    print("[Phase 1] Warm-up: normal traffic")
    print("Goal: establish baseline success, latency, and cost metrics\n")

    # ---------------- Phase 2: Controlled failure ----------------
    print("[Phase 2] Injecting failure scenario")
    print("Failure mode: ISSUER_DEGRADATION (simulated real-world incident)\n")

    # ---------------- Threaded Simulator (Continuous Data) ----------------
    import threading
    import queue
    
    event_queue = queue.Queue(maxsize=1000)
    
    def run_simulator_loop(sim, q, max_ev, failure_event):
        """Generates events continuously in a background thread."""
        stream = sim.stream(interval_sec=0.02)
        count = 0
        try:
            for event in stream:
                count += 1
                if max_ev and count >= max_ev:
                    break
                
                # Injection hooks inside the thread
                if count == failure_event:
                    sim.set_failure_mode(FailureMode.ISSUER_DEGRADATION)
                    print(f"\n[Simulator] Failure injected: ISSUER_DEGRADATION (event #{count})\n")
                if count == failure_event + 130:
                    sim.set_failure_mode(FailureMode.MULTI_MERCHANT_ESCALATION)
                    print(f"\n[Simulator] ESCALATION injected: MULTI_MERCHANT_ESCALATION (event #{count})\n")

                if not q.full():
                    q.put(event)
                else:
                    # Drop event if agent too slow? Or wait? 
                    # For a demo, dropping is better than blocking simulator to keep "live" feel,
                    # but metrics might jump. Let's block briefly.
                    try:
                        q.put(event, timeout=0.1)
                    except queue.Full:
                        pass
        except Exception as e:
            print(f"Simulator thread error: {e}")
        finally:
             q.put(None) # Signal done

    sim_thread = threading.Thread(
        target=run_simulator_loop,
        args=(simulator, event_queue, None, 120),
        daemon=True
    )
    sim_thread.start()

    agent.run(
        event_queue=event_queue,  # Pass queue instead of stream
        # max_events handled by queue None signal
    )

    # ---------------- Post-run summaries ----------------
    print("\n==================================")
    print("Run complete. Post-run analysis")
    print("==================================\n")

    learner = agent.learner
    executor = agent.executor

    # ---------------- Learning summary ----------------
    print("[Learning Summary]")
    summary = learner.summarize_learning_heuristic()
    print(summary)

    # ---------------- Action effectiveness ----------------
    # (Added in Plan A: visible learning impact)
    if hasattr(learner, "get_action_effectiveness"):
        action_stats = learner.get_action_effectiveness()
        if action_stats:
            print("\n[Action Effectiveness]")
            for action, stats in action_stats.items():
                print(
                    f"- {action}: "
                    f"helped={stats['helped']} "
                    f"hurt={stats['hurt']} "
                    f"neutral={stats['neutral']}"
                )

    # ---------------- Execution decisions ----------------
    exec_log = executor.get_execution_log()
    if exec_log:
        print("\n[Execution Decisions]")
        print("Showing last 10 decisions (executed / blocked / escalated):")
        for line in exec_log[-10:]:
            print(line)

    # ---------------- Rollback visibility ----------------
    rollback_log = executor.get_rollback_log()
    if rollback_log:
        print("\n[Rollbacks]")
        for line in rollback_log:
            print(line)

    # ---------------- Final narrative ----------------
    print("\nDemo complete.")
    print("This agent demonstrated:")
    print("- Continuous observation under noisy, realistic traffic")
    print("- Hypothesis-driven reasoning (not rule-based automation)")
    print("- Cost & retry-aware decision-making")
    print("- Explicit human-approval safety boundaries")
    print("- Automatic rollback when actions backfire")
    print("- Learning from outcomes over time\n")


if __name__ == "__main__":
    main()
