# Agentic Payment Operations Manager

A production-style agentic AI system for real-time payment operations: **Observe -> Reason -> Decide -> Act -> Learn**. The system simulates payment traffic, detects failures, reasons using **Gemini 2.5 Flash** (with deterministic fallback), makes safe decisions via a deterministic policy engine, executes guarded actions with rollback, learns from outcomes, and exposes state to operators via a **Streamlit dashboard**.

## How to Run

**1. Install dependencies**

```bash
cd agentic-payment-ops
pip install -r requirements.txt
```

**2. Run the agent** (simulates traffic and runs the Observe -> Learn loop)

```bash
python run.py
```

**3. Run the operator dashboard** (in a separate terminal; run from `agentic-payment-ops`)

```bash
cd agentic-payment-ops
streamlit run dashboard.py
```

- **Without `GEMINI_API_KEY`**: The system runs end-to-end using heuristic reasoning and learning. No API key required.
- **With `GEMINI_API_KEY`**: The reasoner uses Gemini 2.5 Flash for hypotheses and the learner can summarize outcomes via Gemini. Set the env var to your API key (e.g. `set GEMINI_API_KEY=your_key` on Windows or `export GEMINI_API_KEY=your_key` on Linux/macOS). Actions are still decided and executed only by the deterministic engine.

## Project Structure

```
agentic-payment-ops/
├── simulator.py          # Payment traffic simulator with failure injection
├── observer.py            # Sliding window feature extraction
├── llm_reasoner.py        # Gemini 2.5 Flash reasoning (JSON only)
├── reasoner.py            # Wraps LLM + heuristic fallback
├── decision.py            # Deterministic policy engine
├── executor.py            # Guarded action executor + rollback
├── learner.py             # Outcome memory and learning
├── agent.py               # Orchestrates observe->learn loop
├── run.py                 # Runnable entry point
├── dashboard.py           # Streamlit dashboard for operators
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

### Safety and Constraints

- **LLM usage**: Gemini is used only for (1) interpreting metrics and generating hypotheses (JSON), and (2) optional post-hoc learning summary. The LLM does **not** execute or choose actions.
- **Deterministic decisions**: All actions are chosen by the decision engine from structured hypotheses.
- **Guardrails**: Risk scoring; only actions below a risk threshold are auto-executed; higher risk is human-in-the-loop (logged).
- **Rollback**: If success rate drops or latency increases after an action, the executor rolls back and the learner records the outcome.
- **Dashboard**: Reads from `state/*.json`; operators see what the agent is doing and why in real time.

### Acceptance Criteria

- Fully runnable end-to-end via `python run.py` and `streamlit run dashboard.py`.
- Demonstrates autonomous intervention (e.g. reroute on issuer degradation).
- Demonstrates rollback behavior when metrics regress.
- Uses Gemini 2.5 Flash safely (hypotheses and explanations only; fallback heuristics when unavailable).
- Dashboard clearly explains agent behavior (hypotheses, decisions, reasoning).
- System is clearly not a rules-only engine: hypothesis-based reasoning with LLM or heuristics, then deterministic policy.
