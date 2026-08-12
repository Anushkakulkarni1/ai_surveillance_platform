

import html
import streamlit as st
from contextlib import contextmanager


# Page Header

def page_header(last_update: str, risk: str, ai_status: str, vad_status: str = "NOMINAL"):
    risk_color = {
        "LOW": "#2DD4A7",
        "MEDIUM": "#F5A623",
        "HIGH": "#FB3B5A",
    }.get(risk.upper(), "#22D3EE")

    ai_color = "#2DD4A7" if ai_status.upper() == "ACTIVE" else "#FB3B5A"
    vad_color = "#FB3B5A" if vad_status.upper() == "CRITICAL" else "#2DD4A7"

    
    html_content = f"""<div class="header">
<div>
<div class="dashboard-title">AI Surveillance Intelligence Platform</div>
<div class="dashboard-subtitle">Enterprise Security Operations Center</div>
</div>
<div class="header-badges">
<div class="badge">
<div class="badge-label">Last Update</div>
<div class="badge-value">{html.escape(str(last_update))}</div>
</div>
<div class="badge" style="border-left:3px solid {risk_color};">
<div class="badge-label">Risk Level</div>
<div class="badge-value">{html.escape(str(risk))}</div>
</div>
<div class="badge" style="border-left:3px solid {vad_color};">
<div class="badge-label">VAD Engine</div>
<div class="badge-value">{html.escape(str(vad_status))}</div>
</div>
<div class="badge" style="border-left:3px solid {ai_color};">
<div class="badge-label">AI Status</div>
<div class="badge-value">{html.escape(str(ai_status))}</div>
</div>
</div>
</div>""".strip()

    st.markdown(html_content, unsafe_allow_html=True)


# Section header

def section_header(title: str, subtitle: str = ""):
    html_content = f"""<div style="margin-top:20px;margin-bottom:16px;">
<div class="section-title">{html.escape(title)}</div>
<div class="section-subtitle">{html.escape(subtitle)}</div>
</div>""".strip()
    st.markdown(html_content, unsafe_allow_html=True)


# Kpi card

def kpi_card(title: str, value, subtitle: str, accent: str = "#22D3EE"):
    html_content = f"""<div class="kpi-card">
<div class="kpi-top">
<span class="kpi-accent" style="background:{accent};color:{accent};"></span>
<span class="kpi-label">{html.escape(title)}</span>
</div>
<div class="kpi-value">{value}</div>
<div class="kpi-footer">{html.escape(str(subtitle))}</div>
</div>""".strip()
    st.markdown(html_content, unsafe_allow_html=True)


# Chart container

@contextmanager
def chart_container(title: str, subtitle: str = ""):
    # Modernized approach avoids mixing unclosed HTML layers across blocks
    st.markdown(f"""<div class="chart-title">{html.escape(title)}</div>
<div class="chart-subtitle" style="margin-bottom: 10px;">{html.escape(subtitle)}</div>""", unsafe_allow_html=True)
    

    with st.container():
        yield

def chart_card_header(title: str, subtitle: str = "", pill_text: str = None, pill_color: str = "#2DD4A7"):
    pill_html = ""
    if pill_text:
        pill_html = f"""<span class="status-pill" style="background:{pill_color}22;color:{pill_color};border:1px solid {pill_color}55;">{html.escape(pill_text)}</span>"""

    html_content = f"""<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
<div>
<div class="chart-title">{html.escape(title)}</div>
<div class="chart-subtitle">{html.escape(subtitle)}</div>
</div>
{pill_html}
</div>""".strip()
    st.markdown(html_content, unsafe_allow_html=True)


# Status badge

def status_badge(text: str, color: str = "#2DD4A7"):
    st.markdown(f"""<span class="status-pill" style="background:{color}22;color:{color};border:1px solid {color}55;">{html.escape(text)}</span>""", unsafe_allow_html=True)


# Evidence card

def evidence_card(image_path: str, label: str, zone: str, person: str, timestamp: str):
    st.image(image_path, use_container_width=True)
    html_content = f"""<div class="evidence-caption">
<span>{html.escape(str(label))}</span>
<span>{html.escape(str(zone))}</span>
</div>""".strip()
    st.markdown(html_content, unsafe_allow_html=True)


# Ai report

def ai_report(risk: str, findings: list, recommendations: list):
    risk_color = {
        "LOW": "#2DD4A7",
        "MEDIUM": "#F5A623",
        "HIGH": "#FB3B5A",
    }.get(risk.upper(), "#22D3EE")

    findings_html = "".join(f"<li>{html.escape(f)}</li>" for f in findings)
    reco_html = "".join(f"<li>{html.escape(r)}</li>" for r in recommendations)

    html_content = f"""<div class="ai-report">
<div style="display:flex;align-items:center;justify-content:space-between;">
<div class="ai-report-title">AI Situation Report</div>
<span class="status-pill" style="background:{risk_color}22;color:{risk_color};border:1px solid {risk_color}55;">{html.escape(risk)} RISK</span>
</div>
<div style="font-size:11px;color:#5C6579;text-transform:uppercase;letter-spacing:1px;margin-top:14px;">Key Findings</div>
<ul class="ai-report-list">{findings_html}</ul>
<div style="font-size:11px;color:#5C6579;text-transform:uppercase;letter-spacing:1px;margin-top:10px;">Recommended Actions</div>
<ul class="ai-report-list">{reco_html}</ul>
</div>""".strip()
    st.markdown(html_content, unsafe_allow_html=True)


# Chat rag pipeline

def render_chat_panel_header(engine_online: bool):

    status_text = "Gemini RAG · Online" if engine_online else "Gemini RAG · Offline"
    status_color = "#2DD4A7" if engine_online else "#FB3B5A"

    st.markdown(
        f"""<div class="chat-panel-header-standalone">
<div class="chat-avatar">🛰️</div>
<div>
<div class="chat-panel-title">Surveillance Intelligence Assistant</div>
<div class="chat-panel-sub" style="color:{status_color};">
<span style="width:6px;height:6px;border-radius:50%;background:{status_color};display:inline-block;box-shadow:0 0 6px {status_color};"></span>
{status_text}
</div>
</div>
</div>""".strip(),
        unsafe_allow_html=True,
    )




def recent_incidents(df):
    if df is None or df.empty:
        st.info("No incidents available.")
        return

    cols = [c for c in ["Timestamp", "Event", "Person_ID", "Zone"] if c in df.columns]
    rows_html = ""

    for _, row in df[cols].iterrows():
        cells = "".join(f"<td>{html.escape(str(row[c]))}</td>" for c in cols)
        rows_html += f"<tr>{cells}</tr>"

    header_html = "".join(f"<th>{html.escape(c)}</th>" for c in cols)

    html_content = f"""<div class="incident-table-wrap">
<table class="incident-table">
<thead><tr>{header_html}</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>""".strip()
    st.markdown(html_content, unsafe_allow_html=True)


# Divider

def divider():
    st.markdown("<hr class='hr-soft'/>", unsafe_allow_html=True)