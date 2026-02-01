"""
Operator dashboard for the agentic payment operations system (REAL-TIME).
Reads shared state (state/metrics.json, hypotheses.json, actions.json, control_state.json) and displays:
- Live KPI row (success rate, P95 latency, retry amplification, avg cost, SYSTEM MODE)
- Success rate by issuer (bar chart) + merchant health (dynamic)
- Latency trends (rolling window)
- Retry & cost risk panel (live flags)
- Decision timeline (event-driven; no duplicate entries; EXECUTED/BLOCKED/SKIPPED/ROLLED_BACK)
- Explainability (current hypothesis, risk, guardrails, why taken/skipped)
- Human-in-the-loop panel (when escalation active)
- Learning outcomes panel (stream-aware)
All graphs update dynamically; refresh interval configurable (default 3s).
"""
import json
import time
from pathlib import Path

import streamlit as st

STATE_DIR = Path(__file__).resolve().parent / "state"
METRICS_PATH = STATE_DIR / "metrics.json"
HYPOTHESES_PATH = STATE_DIR / "hypotheses.json"
ACTIONS_PATH = STATE_DIR / "actions.json"
CONTROL_STATE_PATH = STATE_DIR / "control_state.json"

# Rolling window sizes for live graphs
LATENCY_TREND_POINTS = 50
DECISION_TIMELINE_MAX = 30

# Risk thresholds (must match decision.py / reasoner.py for live flags)
HIGH_ATTEMPT_AMPLIFICATION = 1.6
HIGH_COST_ESCALATION = 0.05
RETRY_AMPLIFICATION_STORM = 1.2
COST_ESCALATION_THRESHOLD = 0.03


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def main() -> None:
    st.set_page_config(
        page_title="Payment Ops Agent Dashboard",
        page_icon="📊",
        layout="wide",
    )
    st.title("Payment Operations Agent Dashboard (Live)")
    st.caption("Observe -> Reason -> Decide -> Act -> Learn | Real-time state from state/*.json")

    # Main loop for live updates
    placeholder_banner = st.empty()
    placeholder_kpis = st.empty()
    placeholder_charts = st.empty()
    placeholder_hyp = st.empty()
    placeholder_timeline = st.empty()
    placeholder_explain = st.empty()
    placeholder_hitl = st.empty()
    placeholder_learn = st.empty()

    while True:
        # Load latest state
        metrics_data = load_json(METRICS_PATH)
        hypotheses_data = load_json(HYPOTHESES_PATH)
        actions_data = load_json(ACTIONS_PATH)
        control_data = load_json(CONTROL_STATE_PATH)

        current = metrics_data.get("current") or {}
        latency_trend = metrics_data.get("latency_trend") or []
        hyp_latest = hypotheses_data.get("latest") or {}
        hyp_history = hypotheses_data.get("history") or []
        act_latest = actions_data.get("latest") or {}
        act_history = actions_data.get("history") or []
        system_mode = control_data.get("system_mode", "NORMAL")
        escalation = control_data.get("escalation") or {}
        learning = control_data.get("learning") or {}

        # --- State Banner ---
        with placeholder_banner.container():
            status_color = "🟢"
            status_text = "RUNNING"
            status_msg = "Agent is optimizing normally."
            bg_color = "#1a3c30" # deep green
            border_color = "#3a6c50"

            if escalation.get("active") or system_mode == "HUMAN_APPROVAL_REQUIRED":
                status_color = "🔴"
                status_text = "WAITING FOR HUMAN"
                status_msg = "Decision loop paused. Authorization required."
                bg_color = "#3c1a1a" # deep red
                border_color = "#6c3a3a"
            elif system_mode == "DEGRADED":
                 status_color = "🟡"
                 status_text = "DEGRADED"
                 status_msg = "Performance degraded, agent attempting recovery."
                 bg_color = "#3c3c1a" # deep yellow
                 border_color = "#6c6c3a"
            
            st.markdown(f"""
            <div style="padding: 1rem; border-radius: 0.5rem; background-color: {bg_color}; border: 1px solid {border_color}; margin-bottom: 1rem;">
                <h2 style="margin:0; padding:0; color: white;">{status_color} {status_text}</h2>
                <p style="margin:0; opacity: 0.9; color: #ddd;">{status_msg}</p>
            </div>
            """, unsafe_allow_html=True)

        with placeholder_kpis.container():
            st.subheader("Live KPIs")
            sr = current.get("success_rate")
            p95 = current.get("p95_latency_ms")
            retry_amp = current.get("retry_amplification")
            avg_cost = current.get("average_estimated_cost")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Success rate", f"{sr:.1%}" if sr is not None else "—")
            with col2:
                st.metric("P95 latency", f"{p95:.0f}" if p95 is not None else "—")
            with col3:
                st.metric("Retry amp", f"{retry_amp:.2f}" if retry_amp is not None else "—")
            with col4:
                st.metric("Avg cost", f"{avg_cost:.3f}" if avg_cost is not None else "—")

        with placeholder_charts.container():
            # Risk flags
            attempt_amp = current.get("attempt_amplification")
            risk_retry = (attempt_amp is not None and attempt_amp >= HIGH_ATTEMPT_AMPLIFICATION) or (retry_amp is not None and retry_amp >= RETRY_AMPLIFICATION_STORM)
            risk_cost = avg_cost is not None and avg_cost >= COST_ESCALATION_THRESHOLD
            c1, c2 = st.columns(2)
            with c1:
                if risk_retry:
                    st.error("Retry / attempt amplification above threshold")
                else:
                    st.success("Retry amplification within range")
            with c2:
                if risk_cost:
                    st.error("Cost above escalation threshold")
                else:
                    st.success("Cost within range")

            st.subheader("Latency trend (P95 ms) — rolling")
            if latency_trend:
                import pandas as pd
                trend_slice = latency_trend[-LATENCY_TREND_POINTS:]
                df_lat = pd.DataFrame(trend_slice)
                if "p95_latency_ms" in df_lat.columns:
                    df_lat = df_lat[["p95_latency_ms"]].rename_axis("window")
                    st.line_chart(df_lat)
                else:
                    st.line_chart(df_lat)
            else:
                st.info("No latency history yet.")

        with placeholder_hyp.container():
            st.subheader("Detected hypotheses")
            if hyp_latest and hyp_latest.get("cause") is not None:
                cause = hyp_latest.get("cause", "—")
                confidence = hyp_latest.get("confidence", 0)
                uncertainty = hyp_latest.get("uncertainty", 0)
                evidence = hyp_latest.get("evidence", "")
                st.markdown(f"**Cause:** `{cause}` · **Confidence:** {confidence:.2f} · **Uncertainty:** {uncertainty:.2f}")
                st.markdown(f"**Evidence:** {evidence}")
            else:
                st.info("No hypothesis yet.")

        with placeholder_timeline.container():
            st.subheader("Decision timeline (event-driven)")
            display_actions = []
            prev_key = None
            for a in reversed(act_history[-DECISION_TIMELINE_MAX:]):
                outcome = a.get("outcome", "executed" if a.get("executed") else "blocked")
                key = (a.get("action_type"), a.get("target"), outcome)
                if key != prev_key:
                    display_actions.append(a)
                    prev_key = key
            display_actions.reverse()
            for a in display_actions[-10:]: # Reduce count for layout stability
                outcome = a.get("outcome", "executed" if a.get("executed") else "blocked")
                label = {"executed": "EXECUTED", "blocked": "BLOCKED", "skipped_cooldown": "SKIPPED", "rolled_back": "ROLLBACK"}.get(outcome, outcome.upper())
                msg_str = f"**{label}** — {a.get('action_type', '—')} target={a.get('target', '—')}"
                if outcome == "executed":
                    st.success(msg_str)
                elif outcome == "blocked":
                    st.warning(msg_str)
                else:
                    st.info(msg_str)

        with placeholder_explain.container():
            st.subheader("Current decision explainability")
            if act_latest and act_latest.get("action_type") is not None:
                action_type = act_latest.get("action_type", "—")
                target = act_latest.get("target") or "—"
                risk = act_latest.get("risk_score", 0)
                executed = act_latest.get("executed", False)
                outcome = act_latest.get("outcome", "executed" if executed else "blocked")
                st.markdown(f"**Action:** `{action_type}` · **Target:** `{target}` · **Risk:** {risk:.2f}")
                st.markdown(f"**Outcome:** {outcome.upper()}")
                st.markdown(f"> {act_latest.get('reasoning', '')}")
            else:
                st.info("No decision yet.")

        with placeholder_hitl.container():
            st.subheader("Human-in-the-loop")
            if escalation.get("active"):
                st.error("Escalation active — autonomous execution frozen")
                st.markdown(f"**Action:** {escalation.get('action_type', '—')} target={escalation.get('target', '—')}")
                st.markdown(f"**Reason:** {escalation.get('reason')}")
                
                st.markdown("### Authorization required")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ Approve Action", type="primary", use_container_width=True, key=f"btn_app_{time.time()}"):
                        escalation["status"] = "approved"
                        from state_writer import write_pending_approval
                        write_pending_approval(escalation)
                        st.success("Approved!")
                        time.sleep(0.5)
                with c2:
                    if st.button("🚫 Reject Action", use_container_width=True, key=f"btn_rej_{time.time()}"):
                        escalation["status"] = "rejected"
                        from state_writer import write_pending_approval
                        write_pending_approval(escalation)
                        st.warning("Rejected.")
                        time.sleep(0.5)
            else:
                st.success("No pending escalation")

        with placeholder_learn.container():
            st.subheader("Learning outcomes (stream-aware)")
            helped = learning.get("helped", 0)
            hurt = learning.get("hurt", 0)
            neutral = learning.get("neutral", 0)
            st.metric("Recent outcomes", f"Helped: {helped} | Hurt: {hurt} | Neutral: {neutral}")

        time.sleep(1.0)


if __name__ == "__main__":
    main()
