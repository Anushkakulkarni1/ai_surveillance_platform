
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import pandas as pd
import streamlit as st

try:
    from styles import load_css
    from metrics import load_all_data, calculate_metrics
    from components import (
        page_header,
        section_header,
        kpi_card,
        chart_card_header,
        evidence_card,
        ai_report,
        render_chat_panel_header,
        recent_incidents,
        divider,
    )
    from charts import (
        event_distribution,
        zone_distribution,
        occupancy_chart,
        dwell_chart,
        top_person_chart,
        incident_timeline_chart,
        vad_live_chart,
        latest_evidence,
    )
    from backend_client import fetch_recent_telemetry, fetch_backend_health
except ModuleNotFoundError:
    from dashboard.styles import load_css
    from dashboard.metrics import load_all_data, calculate_metrics
    from dashboard.components import (
        page_header,
        section_header,
        kpi_card,
        chart_card_header,
        evidence_card,
        ai_report,
        render_chat_panel_header,
        recent_incidents,
        divider,
    )
    from dashboard.charts import (
        event_distribution,
        zone_distribution,
        occupancy_chart,
        dwell_chart,
        top_person_chart,
        incident_timeline_chart,
        vad_live_chart,
        latest_evidence,
    )
    from dashboard.backend_client import fetch_recent_telemetry, fetch_backend_health

from ai.fallback_assistant import generate_fallback_answer


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Surveillance Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(load_css(), unsafe_allow_html=True)

if "vad_threshold" not in st.session_state:
    st.session_state.vad_threshold = 0.65


def _is_valid_chat_entry(entry) -> bool:
    if not isinstance(entry, (tuple, list)) or len(entry) != 3:
        return False
    role, blocks, used_fallback = entry
    if not isinstance(role, str) or not isinstance(blocks, list):
        return False
    return all(isinstance(b, dict) and "type" in b for b in blocks)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
elif not all(_is_valid_chat_entry(e) for e in st.session_state.chat_history):
    st.session_state.chat_history = []

data = load_all_data()

# Fetch live telemetry from backend; fall back to CSV if unreachable
telemetry_df, backend_reachable = fetch_recent_telemetry(BACKEND_URL, count=200)

if backend_reachable and not telemetry_df.empty:
    data["vad"] = telemetry_df
    vad_source_label = "Live Backend Telemetry"
else:
    vad_source_label = "Local CSV Log"

metrics = calculate_metrics(data, vad_alert_threshold=st.session_state.vad_threshold)

rag = None
rag_online = False

try:
    from ai.rag_pipeline import RAGPipeline

    rag = RAGPipeline()
    rag_online = True
except Exception:
    rag = None
    rag_online = False


def ask_assistant(question: str) -> tuple:
    if rag is None:
        return generate_fallback_answer(question, data), True

    try:
        if hasattr(rag, "query"):
            result = rag.query(question)
        else:
            result = rag.answer(question)

        if isinstance(result, tuple):
            result = result[0]

        return [{"type": "text", "content": result}], False

    except Exception:
        return generate_fallback_answer(question, data), True


with st.sidebar:

    st.markdown(
        """
<div class="sidebar-brand">🛡️ AI Surveillance</div>
<div class="sidebar-brand-sub">Enterprise Security Operations</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section-label">Platform Status</div>',
        unsafe_allow_html=True,
    )

    platform_status = [
        ("YOLOv8 Detection", "#2DD4A7"),
        ("Gemini AI", "#2DD4A7" if rag_online else "#FB3B5A"),
        ("FAISS Index", "#2DD4A7"),
        ("Knowledge Base", "#2DD4A7"),
        ("Evidence Storage", "#2DD4A7"),
        ("Telemetry Backend", "#2DD4A7" if backend_reachable else "#FB3B5A"),
        (
            "VAD Engine (Conv3D)",
            "#FB3B5A" if metrics["vad_status"] == "CRITICAL" else "#2DD4A7",
        ),
    ]

    for name, color in platform_status:
        st.markdown(
            f"""
<div class="sidebar-status-row">
<span class="sidebar-status-name"><span class="sidebar-dot" style="background:{color};color:{color};"></span>{name}</span>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-section-label">Live Statistics</div>',
        unsafe_allow_html=True,
    )

    sidebar_stats = [
        ("Critical Alerts", metrics["critical_alerts"]),
        ("Active Persons", metrics["active_persons"]),
        ("Current Occupancy", metrics["current_occupancy"]),
        ("Peak Occupancy", metrics["peak_occupancy"]),
        ("Evidence Images", metrics["evidence"]),
        ("Most Active Zone", metrics["most_active_zone"]),
        ("VAD Anomaly Score", f"{metrics['vad_latest_score']:.3f}"),
        ("VAD Threshold Breaches", metrics["vad_critical_count"]),
    ]

    for label, value in sidebar_stats:
        st.markdown(
            f"""<div class="sidebar-stat"><span>{label}</span><b>{value}</b></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-section-label">Live Refresh</div>', unsafe_allow_html=True
    )

    st.session_state.setdefault("live_refresh_enabled", False)
    st.session_state.live_refresh_enabled = st.toggle(
        "Auto-refresh VAD chart",
        value=st.session_state.live_refresh_enabled,
        help="Periodically re-fetches backend telemetry and redraws the "
        "'Live Reconstruction Error' chart on its own, without rerunning "
        "the rest of the page (so it never interrupts an in-progress "
        "chat message or reset a widget you're mid-adjustment on).",
    )

    if st.session_state.live_refresh_enabled:
        st.session_state.setdefault("live_refresh_interval", 10)
        st.session_state.live_refresh_interval = st.slider(
            "Refresh interval (seconds)",
            min_value=5,
            max_value=60,
            value=st.session_state.live_refresh_interval,
            step=5,
        )

    st.caption("Enterprise CCTV Analytics Platform")

page_header(
    metrics["latest_event"],
    metrics["risk"],
    "ACTIVE" if rag_online else "DEGRADED",
    vad_status=metrics["vad_status"],
)

divider()

section_header("Executive Overview", "Live AI-powered surveillance analytics.")

k1, k2, k3, k4, k5, k6, k7 = st.columns(7)

with k1:
    kpi_card(
        "Critical Alerts",
        metrics["critical_alerts"],
        f"{metrics['intrusions']} Intrusions",
        "#FB3B5A",
    )

with k2:
    kpi_card(
        "Active Persons",
        metrics["active_persons"],
        f"Top Person: {metrics['top_person']}",
        "#22D3EE",
    )

with k3:
    kpi_card(
        "Occupancy",
        metrics["current_occupancy"],
        f"Peak: {metrics['peak_occupancy']}",
        "#2DD4A7",
    )

with k4:
    kpi_card("Evidence", metrics["evidence"], "Captured Images", "#F5A623")

with k5:
    kpi_card(
        "Avg Dwell",
        f"{metrics['average_dwell']}s",
        f"Max: {metrics['highest_dwell']}s",
        "#8B7CF6",
    )

with k6:
    kpi_card(
        "VAD Reconstruction Error",
        f"{metrics['vad_latest_score']:.3f}",
        f"Avg: {metrics['vad_avg_score']:.3f}",
        "#FB3B5A" if metrics["vad_status"] == "CRITICAL" else "#22D3EE",
    )

with k7:
    kpi_card(
        "Behavioral Anomalies",
        metrics["vad_critical_count"],
        f"of {metrics['vad_events']} clips scored",
        "#FB3B5A" if metrics["vad_critical_count"] > 0 else "#2DD4A7",
    )

divider()

section_header(
    "Live Analytics", "Real-time security monitoring and incident visualization."
)

left, right = st.columns(2, gap="large")

with left:
    chart_card_header(
        "Incident Timeline", "Security incidents over time, grouped hourly"
    )
    try:
        fig = incident_timeline_chart(data)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except Exception:
        st.info("No incident timeline available.")

with right:
    chart_card_header(
        "Occupancy Analytics", "Live occupancy trend from the people-counting stream"
    )
    try:
        fig = occupancy_chart(data)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except Exception:
        st.info("No occupancy data available.")

divider()

section_header(
    "AI Vision Core — Behavioral Anomaly Engine",
    "Spatio-Temporal Conv3D Autoencoder · unsupervised reconstruction-error monitoring",
)

vad_pill_color = "#FB3B5A" if metrics["vad_status"] == "CRITICAL" else "#2DD4A7"

chart_card_header(
    "Live Reconstruction Error",
    f"Normalized anomaly score per 10-frame clip · latest zone: {metrics['vad_latest_zone']} "
    f"· source: {vad_source_label}",
    pill_text=metrics["vad_status"],
    pill_color=vad_pill_color,
)

ctrl_col, _ = st.columns([1, 2])

with ctrl_col:
    st.session_state.vad_threshold = st.slider(
        "Safety Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.vad_threshold,
        step=0.01,
        help="Clips scoring at or above this normalized reconstruction-error "
        "threshold are flagged as behavioral anomalies (e.g. fighting, "
        "running, panic movement) and logged to logs/abstract_anomalies.csv.",
    )

metrics = calculate_metrics(data, vad_alert_threshold=st.session_state.vad_threshold)

# Isolated fragment for live chart auto-refresh. 
# Fetches fresh telemetry per tick so the chart updates without triggering a full page rerun.
_refresh_interval = (
    st.session_state.live_refresh_interval
    if st.session_state.live_refresh_enabled
    else None
)


@st.fragment(run_every=f"{_refresh_interval}s" if _refresh_interval else None)
def _render_live_vad_chart() -> None:
    fragment_telemetry_df, fragment_backend_reachable = fetch_recent_telemetry(
        BACKEND_URL, count=200
    )

    if fragment_backend_reachable and not fragment_telemetry_df.empty:
        fragment_vad_df = fragment_telemetry_df
    else:
        fragment_vad_df = data.get("vad")

    fragment_metrics = calculate_metrics(
        {**data, "vad": fragment_vad_df},
        vad_alert_threshold=st.session_state.vad_threshold,
    )

    fig = vad_live_chart(fragment_vad_df, threshold=st.session_state.vad_threshold)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    if fragment_metrics["vad_status"] == "CRITICAL":
        st.markdown(
            f"""<div style="margin-top:-6px;color:#FB3B5A;font-size:12.5px;">⚠ {fragment_metrics['vad_latest_description']}</div>""",
            unsafe_allow_html=True,
        )

    if st.session_state.live_refresh_enabled:
        st.caption(f"🔄 Auto-refreshing every {_refresh_interval}s")


_render_live_vad_chart()

divider()

left, right = st.columns(2, gap="large")

with left:
    chart_card_header("Incident Distribution", "Distribution of detected events")
    try:
        fig = event_distribution(data)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except Exception:
        st.info("No event data available.")

with right:
    chart_card_header(
        "Zone Activity", "Distribution of events across surveillance zones"
    )
    try:
        fig = zone_distribution(data)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except Exception:
        st.info("No zone activity available.")

divider()

left, right = st.columns(2, gap="large")

with left:
    chart_card_header(
        "Dwell Time Analytics", "Average time spent inside monitored areas"
    )
    try:
        fig = dwell_chart(data)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except Exception:
        st.info("No dwell analytics available.")

with right:
    chart_card_header("Person Activity", "Most frequently detected individuals")
    try:
        fig = top_person_chart(data)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except Exception:
        st.info("No person activity available.")

divider()

section_header(
    "Security Intelligence", "Latest evidence captured by the surveillance system."
)

left, right = st.columns([1.2, 1], gap="large")

with left:
    chart_card_header(
        "Latest Evidence", "Most recently captured surveillance snapshots"
    )
    try:
        images = latest_evidence()
    except Exception:
        images = []

    if len(images) == 0:
        st.info("No evidence images available.")
    else:
        cols = st.columns(3)
        for i, image in enumerate(images[:6]):
            with cols[i % 3]:
                evidence_card(image, "Captured Event", "-", "-", "-")

with right:
    findings = [
        f"Detected {metrics['critical_alerts']} critical alerts.",
        f"Most active zone: {metrics['most_active_zone']}.",
        f"{metrics['active_persons']} unique people tracked.",
        f"VAD engine flagged {metrics['vad_critical_count']} of {metrics['vad_events']} clips as anomalous.",
    ]

    recommendations = [
        "Increase monitoring in the busiest zone.",
        "Review recent intrusion evidence.",
        "Investigate flagged behavioral anomaly clips.",
        "Continue AI-assisted surveillance.",
    ]

    ai_report(metrics["risk"], findings, recommendations)

divider()

section_header(
    "Operations Log & Assistant", "Recent activity and the AI surveillance assistant."
)

left, right = st.columns([1.3, 1], gap="large")

with left:
    chart_card_header(
        "Recent Incidents", "Latest security events recorded by the surveillance system"
    )

    incident_frames = []

    for key in ["intrusion", "loitering", "fall"]:
        df = data.get(key)
        if df is not None and not df.empty:
            incident_frames.append(df)

    if incident_frames:
        incidents = pd.concat(incident_frames, ignore_index=True)
        if "Timestamp" in incidents.columns:
            incidents = incidents.sort_values("Timestamp", ascending=False)
        recent_incidents(incidents.head(20))
    else:
        st.info("No incidents available.")

with right:
    render_chat_panel_header(engine_online=rag_online)

    query = st.chat_input("Ask about surveillance events, evidence, or anomalies...")

    if query:
        st.session_state.chat_history.append(
            ("user", [{"type": "text", "content": query}], False)
        )

        with st.spinner("Analyzing surveillance data..."):
            blocks, used_fallback = ask_assistant(query)

        st.session_state.chat_history.append(("assistant", blocks, used_fallback))

    chat_container = st.container(height=460, border=False)

    with chat_container:
        if not st.session_state.chat_history:
            st.caption(
                "Ask about intrusions, occupancy, dwell time, or "
                "behavioral anomalies detected by the VAD engine."
            )

        for entry in st.session_state.chat_history:
            if not _is_valid_chat_entry(entry):
                continue

            role, blocks, used_fallback = entry
            avatar = "🧑" if role == "user" else "🛰️"

            with st.chat_message(role, avatar=avatar):
                if used_fallback:
                    st.caption("Offline mode · keyword search · Gemini unavailable")

                for block in blocks:
                    block_type = block.get("type") if isinstance(block, dict) else None
                    if block_type == "text":
                        st.markdown(block.get("content", ""))
                    elif block_type == "table":
                        if block.get("title"):
                            st.markdown(f"**{block['title']}**")
                        st.dataframe(
                            block.get("data"), use_container_width=True, hide_index=True
                        )

divider()

st.markdown(
    """
<div style="text-align:center;padding:20px;color:#5C6579;font-size:12.5px;">
AI Surveillance Intelligence Platform
&nbsp;&nbsp;|&nbsp;&nbsp;
Enterprise CCTV Analytics
&nbsp;&nbsp;|&nbsp;&nbsp;
Powered by YOLOv8 • Conv3D VAD • Gemini • FAISS • RAG
</div>
""",
    unsafe_allow_html=True,
)
