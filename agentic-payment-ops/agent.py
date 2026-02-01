"""
Agent orchestration: Observe → Reason → Decide → Act → Learn loop.
Coordinates simulator, observer, reasoner, decision, executor, learner.
"""
import time
from typing import Callable, Iterator, Optional

from decision import decide
from executor import Executor
from learner import Learner
from models import ActionType, PaymentEvent, WindowMetrics
from observer import Observer
from reasoner import reason
from simulator import FailureMode, PaymentSimulator
from state_writer import write_action, write_hypothesis, write_metrics


class Agent:
    """
    Agentic payment operations manager. Runs the full loop:
    Observe (sliding window metrics) → Reason (hypothesis) → Decide (action) →
    Act (guarded execution / human-in-the-loop) → Learn (outcome memory).
    """

    def __init__(
        self,
        simulator: PaymentSimulator,
        observer: Observer,
        executor: Executor,
        learner: Learner,
        window_events_between_cycles: int = 50,
    ):
        self.simulator = simulator
        self.observer = observer
        self.executor = executor
        self.learner = learner
        self.window_events_between_cycles = window_events_between_cycles
        self._last_metrics: Optional[WindowMetrics] = None
        self._last_action_executed: bool = False
        self._pending_outcome: bool = False  # true after we executed; next window we learn
        self._cycle_count = 0
        
        # State for control plane logic
        self._last_hypothesis_cause: Optional[str] = None
        self._hypothesis_persistence_count: int = 0
        self._last_action_signature: Optional[str] = None
        self._last_action_ts: float = 0.0

        # Wire executor to simulator for reroute / retry / suppress
        def simulator_control(cmd: str, params: dict) -> None:
            if cmd == "reroute":
                # Simulate regression: brief latency bump so rollback can trigger (demo)
                self.simulator.trigger_post_action_latency_bump(num_events=80, multiplier=2.0)
            elif cmd == "retry_policy":
                pass  # Would apply to gateway
            elif cmd == "suppress":
                pass  # Would suppress heavy path
            elif cmd == "rollback":
                self.simulator.clear_failure_mode()

        self.executor.set_simulator_control(simulator_control)

    def observe(self, event: PaymentEvent) -> Optional[WindowMetrics]:
        """Ingest one event; return current window metrics if window is ready."""
        self.observer.ingest(event)
        if not self.observer.ready():
            return None
        metrics = self.observer.get_current_metrics()
        if metrics:
            self._last_metrics = metrics
        return metrics

    def run_cycle(self, metrics: WindowMetrics) -> None:
        """
        One full cycle: Reason -> Decide -> Act -> (outcome recorded later in Learn).
        Writes metrics, hypothesis, and action to state/ for the dashboard.
        """
        self._cycle_count += 1
        self._last_action_executed = False
        ts = time.time()

        # Persist current metrics for dashboard (success rate by issuer, latency trend)
        write_metrics(
            success_rate=metrics.success_rate,
            p95_latency_ms=metrics.p95_latency_ms,
            success_rate_by_issuer=metrics.success_rate_by_issuer,
            retry_amplification=metrics.retry_amplification,
            sample_count=metrics.sample_count,
            window_id=metrics.window_id,
            ts=ts,
        )

        # Reason: hypothesis from LLM or heuristic
        hypothesis = reason(metrics)
        write_hypothesis(
            cause=hypothesis.cause,
            confidence=hypothesis.confidence,
            evidence=hypothesis.evidence,
            source=hypothesis.source,
            ts=ts,
        )
        print(f"[Hypothesis] cause={hypothesis.cause} confidence={hypothesis.confidence:.2f} source={hypothesis.source}")
        print(f"  evidence: {hypothesis.evidence}")

        # Track persistence
        if hypothesis.cause == self._last_hypothesis_cause and hypothesis.cause not in ["Unknown", "INSUFFICIENT_SIGNAL", "UNCERTAIN"]:
            self._hypothesis_persistence_count += 1
        else:
            self._hypothesis_persistence_count = 1
        self._last_hypothesis_cause = hypothesis.cause
        
        persistence_multiplier = 1.0 + (max(0, self._hypothesis_persistence_count - 1) * 0.5)

        # Decide: deterministic action (or NO_OP); learning influences caution
        learning_signal = self.learner.get_action_effectiveness_stats(10)
        trace = decide(
            metrics, 
            hypothesis, 
            learning_signal=learning_signal,
            persistence_multiplier=persistence_multiplier
        )
        print(f"[Decision] action={trace.action.action_type} target={trace.action.target} risk={trace.risk_score:.2f}")
        print(f"  reasoning: {trace.reasoning}")

        if trace.action.action_type == ActionType.NO_OP:
            write_action(
                action_type="no_op",
                target=None,
                risk_score=0.0,
                reason=trace.action.reason,
                reasoning=trace.reasoning,
                executed=False,
                message="No intervention needed",
                ts=ts,
            )
            return

        # DEBOUNCING: Check cooldown
        action_sig = f"{trace.action.action_type}:{trace.action.target}"
        cooldown_period = 15.0 # seconds
        if action_sig == self._last_action_signature and (ts - self._last_action_ts) < cooldown_period:
            print(f"[Cooldown] Skipping identical action {action_sig} (last executed {ts - self._last_action_ts:.1f}s ago)")
            write_action(
                action_type=trace.action.action_type,
                target=trace.action.target,
                risk_score=trace.action.risk_score,
                reason=trace.action.reason,
                reasoning=trace.reasoning + " [SKIPPED: COOLDOWN]",
                executed=False,
                message="Cooldown active; skipping execution",
                ts=ts,
            )
            return

        # Act: execute only if low risk; else human-in-the-loop
        executed, msg = self.executor.execute(trace, metrics)
        
        # Update state if executed OR if blocked by human approval (we don't want to spam log the human request either)
        # Actually, if it requires human approval, executor returns False.
        # We should update last_action_ts even if blocked/failed, to debounce the human request log too?
        # Yes, prevent spamming "Human Approval Required".
        self._last_action_signature = action_sig
        self._last_action_ts = ts
        
        write_action(
            action_type=trace.action.action_type,
            target=trace.action.target,
            risk_score=trace.action.risk_score,
            reason=trace.action.reason,
            reasoning=trace.reasoning,
            executed=executed,
            message=msg,
            ts=ts,
        )
        print(f"[Action] {msg}")

        if executed:
            self._last_action_executed = True
            self.learner.record_decision_context(metrics, trace.action)
            # If we rerouted away from degraded issuer, clear that failure in simulator for demo
            if trace.action.action_type == "reroute" and trace.action.target:
                self.simulator.clear_failure_mode()
        else:
            self.learner.cancel_pending()

    def check_rollback_and_learn(self, metrics: WindowMetrics) -> None:
        """Check for rollback; record outcome for learning."""
        rollbacks = self.executor.check_rollback(metrics)
        for rb in rollbacks:
            print(f"[Rollback] {rb}")

        if self._last_action_executed:
            self.learner.record_outcome(
                metrics,
                rollback_applied=len(rollbacks) > 0,
            )

    def run(
        self,
        event_stream: Iterator[PaymentEvent],
        max_events: int | None = 800,
        failure_injection_after_events: int | None = 150,
        failure_mode: str = FailureMode.ISSUER_DEGRADATION,
    ) -> None:
        """
        Run the agent loop: consume event stream, run cycles when window ready.
        If max_events is None, run indefinitely (for worker.py). If
        failure_injection_after_events is None, do not inject legacy failure
        (simulator uses its own dynamic failure injection).
        """
        print("Agent started. Observe -> Reason -> Decide -> Act -> Learn")
        print("-" * 60)
        event_count = 0
        last_cycle_event_count = 0

        for event in event_stream:
            event_count += 1
            if max_events is not None and event_count >= max_events:
                break

            if failure_injection_after_events is not None and event_count == failure_injection_after_events:
                self.simulator.set_failure_mode(failure_mode)
                print(f"\n[Simulator] Failure injected: {failure_mode} (event #{event_count})\n")

            metrics = self.observe(event)
            if metrics is None:
                continue

            if self._pending_outcome:
                self.check_rollback_and_learn(metrics)
                self._pending_outcome = False

            if event_count - last_cycle_event_count >= self.window_events_between_cycles:
                self.run_cycle(metrics)
                last_cycle_event_count = event_count
                if self._last_action_executed:
                    self._pending_outcome = True

        if max_events is not None:
            summary = self.learner.summarize_learning_heuristic()
            print("-" * 60)
            print("[Learning] " + summary)
            llm_summary = self.learner.summarize_learning_llm()
            if llm_summary:
                print("[Learning LLM] " + llm_summary)
            print("Agent run complete.")


def create_agent(
    window_size: int = 200,
    window_advance: int = 50,
    cycle_interval_events: int = 50,
) -> Agent:
    """Factory: simulator, observer, executor, learner, and agent."""
    sim = PaymentSimulator(seed=42, base_success_rate=0.92)
    obs = Observer(window_size=window_size, window_advance_events=window_advance)
    executor = Executor()
    learner = Learner()
    return Agent(
        simulator=sim,
        observer=obs,
        executor=executor,
        learner=learner,
        window_events_between_cycles=cycle_interval_events,
    )
