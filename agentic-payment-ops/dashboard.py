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


# Auto-refresh interval (seconds)
REFRESH_INTERVAL = 2


def main() -> None:
    st.set_page_config(
        page_title="Payment Ops Agent Dashboard",
        page_icon="📊",
        layout="wide",
    )
    st.title("Payment Operations Agent Dashboard")
    st.caption("Observe → Reason → Decide → Act → Learn | State is read from state/*.json")
    st.markdown("**Live** — updating every " + str(REFRESH_INTERVAL) + " s")

    metrics_data = load_json(METRICS_PATH)
    hypotheses_data = load_json(HYPOTHESES_PATH)
    actions_data = load_json(ACTIONS_PATH)

    current = metrics_data.get("current") or {}
    latency_trend = metrics_data.get("latency_trend") or []
    hyp_latest = hypotheses_data.get("latest") or {}
    hyp_history = hypotheses_data.get("history") or []
    act_latest = actions_data.get("latest") or {}
    act_history = actions_data.get("history") or []

    # 1️⃣ Top-Level KPI Row
    st.divider()
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        sr = current.get("success_rate")
        st.metric("Overall Success Rate", f"{sr:.1%}" if sr is not None else "N/A")
    with kpi2:
        p95 = current.get("p95_latency_ms")
        st.metric("P95 Latency", f"{p95:.0f} ms" if p95 is not None else "N/A")
    with kpi3:
        avg_cost = current.get("avg_cost_per_txn")
        st.metric("Avg Est. Cost / Txn", f"${avg_cost:.4f}" if avg_cost is not None else "N/A")
    with kpi4:
        amp = current.get("retry_amplification")
        st.metric("Avg Attempt Amplification", f"{amp:.2f}x" if amp is not None else "N/A")

    # 2️⃣ Merchant Health View
    st.subheader("Merchant-Level Health")
    col_m1, col_m2 = st.columns(2)
    by_issuer = current.get("success_rate_by_issuer") or {}
    
    with col_m1:
        if by_issuer:
            import pandas as pd
            df_issuer = pd.DataFrame(
                [{"Issuer": k, "Success Rate": v} for k, v in by_issuer.items()]
            )
            st.bar_chart(df_issuer.set_index("Issuer"))
            
            # Highlight worst performing
            worst_issuer = min(by_issuer, key=by_issuer.get)
            worst_val = by_issuer[worst_issuer]
            st.caption(f"⚠️ Needs Attention: **{worst_issuer}** ({worst_val:.1%})")
        else:
            st.info("No issuer data available.")

    with col_m2:
        cost_by_issuer = current.get("avg_cost_by_issuer") or {}
        if cost_by_issuer:
            import pandas as pd
            df_cost = pd.DataFrame(
                [{"Issuer": k, "Avg Cost": v} for k, v in cost_by_issuer.items()]
            )
            st.bar_chart(df_cost.set_index("Issuer"))
        else:
            st.info("Cost per merchant data N/A")

    # 3️⃣ Retry & Cost Risk Panel
    st.subheader("Retry & Cost Risk Monitor")
    risk1, risk2, risk3 = st.columns(3)
    with risk1:
        if amp is not None and amp > 3.0:
            st.error(f"🚨 High Retry Amplification: {amp:.2f}x")
        elif amp is not None:
             st.success(f"✅ Retry Amplification Healthy: {amp:.2f}x")
        else:
            st.info("Retry Amplification N/A")
    
    with risk2:
       # Placeholder for cost trend if available in future
       st.caption("Cost Escalation Risk: Monitoring...")
    
    with risk3:
        # Latency Trend
        if latency_trend:
            import pandas as pd
            df_lat = pd.DataFrame(latency_trend)
            if "p95_latency_ms" in df_lat.columns:
                 st.line_chart(df_lat["p95_latency_ms"], height=100)
            else:
                 st.caption("No latency trend data")
        else:
            st.caption("No latency history")

    # 4️⃣ Decision Timeline & Explainability
    st.subheader("Decision Timeline")
    
    if act_history:
        # Show recent actions (reverse order)
        for i, a in enumerate(reversed(act_history[-10:])):
            ts = a.get("ts", 0)
            import datetime
            dt_str = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            
            action_type = a.get("action_type", "—")
            target = a.get("target") or "—"
            risk = a.get("risk_score", 0)
            executed = a.get("executed", False)
            outcome = a.get("outcome", "unknown")  # planted logic for future
            rolled_back = a.get("rolled_back", False) # status field
            
            # Status visualization
            if rolled_back or outcome == "hurt":
                status_color = "🔴 ROLLED BACK"
                border_color = "red"
            elif not executed:
                status_color = "🟡 BLOCKED (Approval Req)"
                border_color = "orange"
            else:
                status_color = "🟢 EXECUTED"
                border_color = "green"
                
            with st.expander(f"{status_color} {dt_str} | **{action_type}** on **{target}** (Risk: {risk:.2f})"):
                st.markdown(f"**Status:** {status_color}")
                st.markdown(f"**Reason:** {a.get('reason', '—')}")
                st.markdown(f"**Message:** {a.get('message', '—')}")
                
                # 5️⃣ Explainability
                st.divider()
                st.markdown("**🧠 Agent Reasoning:**")
                st.info(a.get('reasoning', 'No detailed reasoning provided.'))
                
                if rolled_back:
                    st.error(f"**Rollback Reason:** {a.get('rollback_reason', 'Not specified')}")
                
                st.markdown("**Context:**")
                st.json({
                     "risk_score": risk,
                     "hypothesis": a.get("hypothesis_id", "N/A"),
                     "outcome_feedback": outcome
                })

    else:
        st.info("No decisions recorded yet.")

    # 6️⃣ Human-in-the-Loop Visibility
    st.subheader("Human Approval Queue")
    blocked_actions = [a for a in act_history if not a.get("executed", True)] # assuming default True if missing to avoid noise, but logically False if blocked
    # Actually, looking at current actions.json, executed is explicit. If it's false, it's blocked/noop.
    # We should look for explicit blocked status or executed=False AND action!=NO_OP
    
    meaningful_blocks = [a for a in blocked_actions if a.get("action_type") != "no_op"]
    
    if meaningful_blocks:
        for a in reversed(meaningful_blocks[-5:]):
            st.warning(f"✋ Blocked: **{a.get('action_type')}** on {a.get('target')} (Risk: {a.get('risk_score', 0):.2f})")
            st.caption(f"Reason: {a.get('reason')}")
    else:
        st.success("No actions currently waiting for approval.")

    # 7️⃣ Learning Outcomes Panel
    st.subheader("Agent Learning Feedback")
    
    # Calculate stats
    helped = sum(1 for a in act_history if a.get("outcome") == "helped")
    hurt = sum(1 for a in act_history if a.get("outcome") == "hurt" or a.get("rolled_back"))
    neutral = sum(1 for a in act_history if a.get("outcome") == "neutral")
    unknown = len(act_history) - helped - hurt - neutral
    
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Helped", helped)
    l2.metric("Hurt/Rolled Back", hurt)
    l3.metric("Neutral", neutral)
    l4.metric("Pending/Unknown", unknown)

    time.sleep(REFRESH_INTERVAL)
    st.rerun()


if __name__ == "__main__":
    main()
