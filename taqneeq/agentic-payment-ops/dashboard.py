"""
Operator dashboard for the agentic payment operations system.
Reads shared state (state/metrics.json, hypotheses.json, actions.json) and displays:
- Success rate by issuer (bar chart)
- Latency trends (line chart)
- Detected hypotheses with confidence
- Current agent decision and clear explanation of why the agent acted or didn't act
"""
import json
import time
from pathlib import Path

import streamlit as st

STATE_DIR = Path(__file__).resolve().parent / "state"
METRICS_PATH = STATE_DIR / "metrics.json"
HYPOTHESES_PATH = STATE_DIR / "hypotheses.json"
ACTIONS_PATH = STATE_DIR / "actions.json"


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
    st.title("Payment Operations Agent Dashboard")
    st.caption("Observe → Reason → Decide → Act → Learn | State is read from state/*.json")

    metrics_data = load_json(METRICS_PATH)
    hypotheses_data = load_json(HYPOTHESES_PATH)
    actions_data = load_json(ACTIONS_PATH)

    current = metrics_data.get("current") or {}
    latency_trend = metrics_data.get("latency_trend") or []
    hyp_latest = hypotheses_data.get("latest") or {}
    hyp_history = hypotheses_data.get("history") or []
    act_latest = actions_data.get("latest") or {}
    act_history = actions_data.get("history") or []

    # --- Metrics row ---
    col1, col2, col3 = st.columns(3)
    with col1:
        sr = current.get("success_rate")
        if sr is not None:
            st.metric("Overall success rate", f"{sr:.1%}")
        else:
            st.metric("Overall success rate", "—")
    with col2:
        p95 = current.get("p95_latency_ms")
        if p95 is not None:
            st.metric("P95 latency (ms)", f"{p95:.0f}")
        else:
            st.metric("P95 latency (ms)", "—")
    with col3:
        samp = current.get("sample_count")
        if samp is not None:
            st.metric("Window sample count", str(samp))
        else:
            st.metric("Window sample count", "—")

    # --- Success rate by issuer (bar chart) ---
    st.subheader("Success rate by issuer")
    by_issuer = current.get("success_rate_by_issuer") or {}
    if by_issuer:
        import pandas as pd
        df_issuer = pd.DataFrame(
            [{"Issuer": k, "Success rate": v} for k, v in by_issuer.items()]
        )
        st.bar_chart(df_issuer.set_index("Issuer"))
    else:
        st.info("No issuer data yet. Run the agent (python run.py) to populate.")

    # --- Latency trends (line chart) ---
    st.subheader("Latency trend (P95 ms)")
    if latency_trend:
        import pandas as pd
        df_lat = pd.DataFrame(latency_trend)
        if "p95_latency_ms" in df_lat.columns:
            df_lat = df_lat[["p95_latency_ms"]].rename_axis("window")
            st.line_chart(df_lat)
        else:
            st.line_chart(df_lat)
    else:
        st.info("No latency history yet. Run the agent to populate.")

    # --- Hypotheses ---
    st.subheader("Detected hypotheses")
    if hyp_latest and hyp_latest.get("cause") is not None:
        cause = hyp_latest.get("cause", "—")
        confidence = hyp_latest.get("confidence", 0)
        evidence = hyp_latest.get("evidence", "")
        source = hyp_latest.get("source", "")
        st.markdown(f"**Cause:** `{cause}` · **Confidence:** {confidence:.2f} · **Source:** {source}")
        st.markdown(f"**Evidence:** {evidence}")
        if hyp_history:
            with st.expander("Hypothesis history"):
                for h in reversed(hyp_history[-10:]):
                    ev = (h.get("evidence") or "")[:80]
                    if len((h.get("evidence") or "")) > 80:
                        ev += "..."
                    st.markdown(f"- `{h.get('cause', '—')}` (conf={h.get('confidence', 0):.2f}) {ev}")
    else:
        st.info("No hypothesis yet. Run the agent to populate.")

    # --- Current agent decision and explanation ---
    st.subheader("Current agent decision")
    if act_latest and act_latest.get("action_type") is not None:
        action_type = act_latest.get("action_type", "—")
        target = act_latest.get("target") or "—"
        risk = act_latest.get("risk_score", 0)
        executed = act_latest.get("executed", False)
        message = act_latest.get("message", "")
        reason = act_latest.get("reason", "")
        reasoning = act_latest.get("reasoning", "")

        st.markdown(f"**Action:** `{action_type}` · **Target:** `{target}` · **Risk score:** {risk:.2f}")
        st.markdown(f"**Executed:** {'Yes' if executed else 'No (human-in-the-loop or NO_OP)'}")
        if message:
            st.markdown(f"**Message:** {message}")
        if reason:
            st.markdown(f"**Reason:** {reason}")
        st.markdown("**Why the agent acted or didn't act:**")
        st.markdown(f"> {reasoning}")
        if act_history:
            with st.expander("Decision history"):
                for a in reversed(act_history[-10:]):
                    st.markdown(f"- {a.get('action_type', '—')} target={a.get('target', '—')} executed={a.get('executed', False)}")
    else:
        st.info("No decision yet. Run the agent to populate.")

    st.divider()
    st.caption("Refresh the page or use auto-refresh below to see latest state.")
    if st.button("Refresh now"):
        st.rerun()
    auto = st.checkbox("Auto-refresh every 5 seconds", value=False)
    if auto:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
