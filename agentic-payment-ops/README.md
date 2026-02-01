# Agentic Payment Operations Manager

A production-style agentic AI system for real-time payment operations: **Observe -> Reason -> Decide -> Act -> Learn**. The system simulates payment traffic, detects failures, reasons using **Gemini 2.5 Flash** (with deterministic fallback), makes safe decisions via a deterministic policy engine, executes guarded actions with rollback, learns from outcomes, and exposes state to operators via a **Streamlit dashboard**.

## How to Run

**1. Install dependencies**

```bash
cd agentic-payment-ops
pip install -r requirements.txt
```

**2. Run the agent** (one-time simulation; fixed event count)

```bash
python run.py
```

**3. Run the operator dashboard** (in a separate terminal; run from `agentic-payment-ops`)

```bash
cd agentic-payment-ops
streamlit run dashboard.py
```

**Live mode (continuous)** — payment generation and agent processing run indefinitely; dashboard updates automatically every 2 seconds. Start the worker in one terminal, then the dashboard in another:

```bash
# Terminal 1: background worker (infinite loop)
cd agentic-payment-ops
python worker.py

# Terminal 2: dashboard (auto-refreshes)
cd agentic-payment-ops
streamlit run dashboard.py
```

- **Without `GEMINI_API_KEY`**: The system runs end-to-end using heuristic reasoning and learning. No API key required.
- **With `GEMINI_API_KEY`**: The reasoner uses Gemini 2.5 Flash for hypotheses and the learner can summarize outcomes via Gemini. Set the env var to your API key (e.g. `set GEMINI_API_KEY=your_key` on Windows or `export GEMINI_API_KEY=your_key` on Linux/macOS). Actions are still decided and executed only by the deterministic engine.

## Project Structure

```
agentic-payment-ops/
├── simulator.py          # Payment traffic simulator (issuer, merchant, cost, retry)
├── merchant_profiles.py  # Merchant universe (traffic shape, sensitivity)
├── observer.py            # Sliding window + merchant/cost/attempt metrics
├── llm_reasoner.py        # Gemini 2.5 Flash reasoning (JSON only)
├── reasoner.py            # Wraps LLM + heuristic fallback
├── decision.py            # Deterministic policy engine (cost, retry, human-approval)
├── executor.py            # Guarded executor + rollback + human-approval enforcement
├── learner.py             # Outcome memory, learning_policy, action effectiveness
├── learning_policy.py     # Helped/hurt/neutral, cost & retry harm (observational)
├── explainability.py      # Human-readable explanations (no execution)
├── agent.py               # Orchestrates observe->learn loop
├── run.py                 # One-time run (phases, post-run analysis)
├── worker.py              # Continuous run (infinite loop) for live dashboard
├── dashboard.py           # Streamlit dashboard (auto-refresh in live mode)
├── state_writer.py        # Writes agent state to state/*.json
├── state/                 # Shared state for dashboard
│   ├── metrics.json       # Current metrics + latency trend
│   ├── hypotheses.json   # Latest + history of hypotheses
│   └── actions.json      # Latest + history of decisions/actions
└── README.md
```

## Design Overview

### Architecture

| Module | Role |
|--------|------|
| **simulator.py** | Generates payment events (issuer, method, latency, retries, outcome, error codes). Injects failure modes: issuer degradation, retry storm, latency spike. |
| **observer.py** | Sliding-window state; computes success rate by issuer, retry amplification, p95 latency, error distribution. |
| **llm_reasoner.py** | Calls **Gemini 2.5 Flash** to interpret metrics and output a single JSON hypothesis (cause, confidence, evidence). No actions. Conservative prompt; strict JSON only. |
| **reasoner.py** | Wraps LLM + fallback: if Gemini fails or key missing, uses deterministic heuristics to produce hypotheses. |
| **decision.py** | Deterministic policy engine. Maps hypothesis + metrics to one Action (reroute, retry_policy, suppress) or NO_OP. Optimizes tradeoffs (success rate, latency, risk). Does not execute. |
| **executor.py** | Executes only low-risk actions; high-risk actions are logged for human-in-the-loop. Supports rollback when metrics regress. |
| **learner.py** | Stores context -> action -> outcome; evaluates helped/hurt; optional Gemini summary for learning (no auto-apply). |
| **agent.py** | Orchestrates the loop; writes metrics, hypotheses, and actions to `state/*.json` for the dashboard. |
| **dashboard.py** | Streamlit UI: success rate by issuer (bar chart), latency trend (line chart), detected hypotheses, current decision, and clear explanation of why the agent acted or didn't act. |

### What Makes This System Agentic

- **Observe → Reason → Decide → Act → Learn**: The agent continuously ingests signals, forms hypotheses (LLM or heuristics), makes deterministic decisions, executes guarded actions, and records outcomes. Learning influences future caution (e.g. higher confidence threshold when recent actions hurt), but does not auto-adjust policy.
- **Not a rules engine**: Hypotheses are generated from patterns (LLM or heuristics); the decision engine maps hypotheses to actions using risk, cost, and attempt-amplification thresholds. The agent sometimes chooses NO_OP, sometimes rolls back, and learning sometimes shows actions were ineffective — that is expected and desired.
- **Not a single LLM call**: The LLM only interprets metrics and produces hypotheses (JSON). It does not execute actions or choose what to do.

### What It Can and Cannot Do

- **Can**: Detect issuer degradation, retry storms, latency spikes; propose reroute, retry-policy, or suppress; auto-execute low-risk actions; roll back when metrics regress; learn from outcomes (helped/hurt/neutral) and expose summaries; require human approval for high-risk or explicitly flagged actions.
- **Cannot**: Automatically change its own policy from learning; execute actions without guardrails; bypass human-in-the-loop when risk or `requires_human_approval` is set.

### Why Rollback Is Not Failure

- Rollback is a **safety mechanism**: when an action causes success rate to drop or latency to increase, the system reverts the action and records the outcome. That protects payment health and provides a clear signal that the action was unhelpful. The learner records this so the system can be more cautious (e.g. require higher confidence) when recent outcomes were bad. In fintech, reverting a bad change quickly is success, not failure.

### Fintech Safety, Human Trust, Autonomy with Guardrails

- **Explicit human-approval boundaries**: Actions above a risk threshold or with `requires_human_approval` are not auto-executed; they are logged for human review. The executor enforces this.
- **Cost- and retry-aware decisions**: The decision engine considers cost and attempt amplification; learning penalizes cost explosions and retry harm. This keeps the system reality-grounded and merchant-aware.
- **Explainability**: The `explainability` module provides human-readable explanations for observations, hypotheses, decisions, trade-offs, guardrails, and rollback reasons. No execution — explanation only.
- **Autonomy with guardrails**: The agent can act autonomously when risk is low and no human-approval flag is set. When in doubt, it waits or escalates.

### Safety and Constraints

- **LLM usage**: Gemini is used only for (1) interpreting metrics and generating hypotheses (JSON), and (2) optional post-hoc learning summary. The LLM does **not** execute or choose actions.
- **Deterministic decisions**: All actions are chosen by the decision engine from structured hypotheses. Learning influences confidence/caution, not hard rules.
- **Guardrails**: Risk scoring; `requires_human_approval`; only actions below risk threshold and without human-approval flag are auto-executed.
- **Rollback**: If success rate drops or latency increases after an action, the executor rolls back and the learner records the outcome.
- **Dashboard**: Reads from `state/*.json`; operators see what the agent is doing and why in real time.

### New / Extended Modules (Additive)

| Module | Role |
|--------|------|
| **merchant_profiles.py** | Realistic merchant universe (traffic shape, sensitivity); used by simulator. |
| **explainability.py** | Human-readable explanations for observations, hypotheses, decisions, trade-offs, guardrails, rollback. No execution. |
| **learning_policy.py** | Defines helped/hurt/neutral; cost and retry harm detection. Observational only; no auto-apply. |
| **learner.py** (extended) | Uses learning_policy; tracks action effectiveness; penalizes cost/retry harm; exposes `get_action_effectiveness_stats`. |
| **decision.py** (extended) | Cost-aware and attempt-amplification thresholds; `requires_human_approval`; optional learning_signal for caution. |
| **executor.py** (extended) | Enforces `requires_human_approval`; does not auto-execute when flag is set. |

### Acceptance Criteria

- Fully runnable end-to-end via `python run.py` and `streamlit run dashboard.py`.
- Demonstrates autonomous intervention (e.g. reroute on issuer degradation).
- Demonstrates rollback behavior when metrics regress.
- Uses Gemini 2.5 Flash safely (hypotheses and explanations only; fallback heuristics when unavailable).
- Dashboard clearly explains agent behavior (hypotheses, decisions, reasoning).
- System is clearly not a rules-only engine: hypothesis-based reasoning with LLM or heuristics, then deterministic policy.
