# Real-Time Dashboard Implementation Plan

## Task 1 — Analysis Summary

### Where refresh happens
- **Current**: Manual "Refresh now" button or checkbox "Auto-refresh every 5 seconds" (when checked, `time.sleep(5)` then `st.rerun()`). No automatic live refresh; entire script re-runs on rerun.
- **Data load**: Single read of `state/metrics.json`, `state/hypotheses.json`, `state/actions.json` at script start; no polling loop.

### Which components re-render
- **All**: On every `st.rerun()`, the whole page re-renders from fresh JSON. There is no partial update; no session state used to avoid re-rendering identical content.

### Which values are frozen or repeated
- **Metrics row**: Uses `current` from JSON; updates when file changes and user refreshes.
- **Bar chart (success by issuer)**: From `current.success_rate_by_issuer`; same snapshot.
- **Latency trend**: From `latency_trend` array; can grow unbounded in UI (backend caps at MAX_LATENCY_POINTS). No rolling window in UI.
- **Hypotheses**: `latest` + `history`; every cycle appends to history, so same hypothesis can appear many times if agent keeps emitting same cause.
- **Decision / actions**: `latest` + `history`; **every decision cycle writes a new action** even when it is identical (e.g. repeated "reroute AXIS"). History shows duplicates; no deduplication.

### Data sources

| Source | Event-driven | Rolling / live | Real-time decision signal |
|--------|--------------|----------------|---------------------------|
| `state/metrics.json` | Yes (written each cycle) | `latency_trend` is rolling (capped); `current` is latest window | `current` reflects latest window |
| `state/hypotheses.json` | Yes (each reasoner call) | History is append-only; no "current only" view | `latest` is current hypothesis |
| `state/actions.json` | Yes (each decision cycle) | History append-only; **same action repeated** each cycle | `latest` is last decision; no EXECUTED/BLOCKED/SKIPPED/ROLLED_BACK flag |

### Gaps identified
1. No **system mode** (NORMAL / DEGRADED / HUMAN_APPROVAL_REQUIRED / COOLDOWN_ACTIVE).
2. No **avg cost per txn** or **retry amplification** in top-level KPIs (data exists in metrics but not in state or UI).
3. No **merchant health** (success rate / cost / attempt amplification by merchant).
4. No **live risk flags** (retry/cost trend thresholds).
5. Decision timeline re-renders **all history** and shows **duplicate** identical decisions.
6. No **explainability** per decision (guardrails triggered, why skipped).
7. No **Human-in-the-loop** panel (escalation reason, proposed action, approve/cancel).
8. Learning panel missing; no **stream-aware** helped/hurt/rollback counts.
9. **Refresh interval** is fixed 5s when auto-refresh on; no configurable interval; no streamlit built-in auto-rerun for true "live" feel.

---

## Task 2 — Implementation Plan

### Refresh strategy
- **Interval**: Default **3 seconds** for full-page refresh when "Live" mode is on. Use `st.rerun()` after `time.sleep(3)` so dashboard stays current without overwhelming the backend. Option: query parameter or slider (3 / 5 / 10 s).
- **Rolling buffers**:
  - **Latency trend**: Use last N points from `latency_trend` (e.g. 50) for line chart so graph is rolling; no change to backend cap.
  - **Decision timeline**: Consume **deduplicated** list: only append when `(action_type, target, outcome, ts)` differs from previous entry so we do not show same "reroute AXIS" 10 times.
- **Panels that update only on state change**:
  - **System mode**: Recompute from current state (escalation active? cooldown? degradation?) each refresh.
  - **Human-in-the-loop panel**: Visible only when `system_mode == HUMAN_APPROVAL_REQUIRED`; content from `state/actions.json` or new `state/escalation.json` (latest blocked action + reason).
  - **Learning panel**: From new `state/learning.json` or extended actions + rollback log; update when file changes.
- **Duplicate decision filtering**:
  - When rendering **Decision timeline**, keep a **display list**: from `history` (newest first), skip an entry if it has the same `(action_type, target, executed, message_key)` as the previous one (same cycle outcome). Alternatively: backend only writes action when **outcome or action changes** (Phase 2 agent debouncing).

### Graph and panel responsibilities
- **KPI row**: Every refresh — from `metrics.current` + derived system_mode.
- **Success by issuer**: Every refresh — from `current.success_rate_by_issuer` (existing).
- **Latency trend**: Rolling window of last 50 points from `latency_trend`; every refresh.
- **Merchant health**: Every refresh — from `current.success_rate_by_merchant`, `avg_cost_by_merchant`, `attempt_amplification_by_merchant` (new in state).
- **Risk flags**: Every refresh — compare `current.retry_amplification` and `current.average_estimated_cost` to thresholds; set flags on/off.
- **Decision timeline**: Event-driven display list; only **new** distinct decisions appended; show EXECUTED / BLOCKED / SKIPPED (cooldown) / ROLLED_BACK.
- **Explainability**: Current cycle only — from `actions.latest` + hypotheses.latest; show hypothesis, risk, guardrails, why taken/skipped.
- **Human-in-the-loop**: Only when escalation active; show escalation reason, affected entity, proposed action; clear on approve/cancel (state update).
- **Learning**: From `state/learning.json` (or embedded in state); show helped/hurt/neutral counts and last rollbacks; update each refresh when learning state exists.

### Backend state extensions (for real-time)
- **metrics.json** `current`: Add `average_estimated_cost`, and ensure `retry_amplification` is present (already is). Add optional `success_rate_by_merchant`, `avg_cost_by_merchant`, `attempt_amplification_by_merchant` for merchant health.
- **actions.json** `latest`: Add `outcome` = `executed` | `blocked` | `skipped_cooldown` | `rolled_back`; add `guardrails_triggered` list for explainability.
- **New or extended**: `system_mode` in a single source (e.g. `state/control_state.json` with `system_mode`, `cooldown_until_ts`, `escalation`, `learning_summary`) so dashboard can show NORMAL / DEGRADED / HUMAN_APPROVAL_REQUIRED / COOLDOWN_ACTIVE and human panel content.

---

## Implementation order (no code in this doc)

1. Extend state_writer and models: system_mode, avg_cost, merchant metrics, action outcome, control_state (escalation, learning).
2. Dashboard: KPI row with system mode; merchant health; risk panel; decision timeline with dedup and outcome badges; explainability; human panel; learning panel; live refresh (3 s default).
3. Reasoner: INSUFFICIENT_SIGNAL when metrics fluctuate or sample_count low.
4. Decision: risk accumulation over time; forced human handover when severe degradation / multiple rollbacks.
5. Agent: debouncing (same action+target within cooldown → skip, write skipped_cooldown); only write action when decision is new or outcome different.
6. Executor/Learner: expose escalation and learning state for state_writer.
