

def load_css() -> str:
    raw_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --canvas: #0E1117;
    --panel: #131720;
    --panel-raised: #171C27;
    --panel-hover: #1B212D;
    --border: #232A38;
    --border-soft: #1B212D;
    --cyan: #22D3EE;
    --cyan-dim: rgba(34,211,238,0.14);
    --cyan-glow: rgba(34,211,238,0.35);
    --crimson: #FB3B5A;
    --crimson-dim: rgba(251,59,90,0.14);
    --crimson-glow: rgba(251,59,90,0.4);
    --amber: #F5A623;
    --amber-dim: rgba(245,166,35,0.14);
    --emerald: #2DD4A7;
    --emerald-dim: rgba(45,212,167,0.14);
    --violet: #8B7CF6;
    --text-primary: #EDF1F7;
    --text-secondary: #9AA5B6;
    --text-muted: #5C6579;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Inter', sans-serif;
    --radius: 14px;
    --radius-sm: 10px;
}

html, body, [class*="css"], .stApp {
    font-family: var(--sans);
    background: var(--canvas) !important;
    color: var(--text-primary);
}

.stApp {
    background-image:
        radial-gradient(circle at 15% 0%, rgba(34,211,238,0.05) 0%, transparent 45%),
        radial-gradient(circle at 85% 20%, rgba(251,59,90,0.04) 0%, transparent 40%);
    padding-top: 6px;
}

.block-container {
    padding-top: 1.1rem;
    padding-left: 2.2rem;
    padding-right: 2.2rem;
    padding-bottom: 2.5rem;
    max-width: 1760px;
}

header, footer, #MainMenu { visibility: hidden; display: none !important; }
div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] { display: none !important; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F131B 0%, #0B0E14 100%);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * { color: var(--text-primary); }
section[data-testid="stSidebar"] .stMarkdown p { color: var(--text-secondary); font-size: 13px; }
section[data-testid="stSidebar"] hr { border-color: var(--border); margin: 14px 0; }

.sidebar-brand {
    font-family: var(--sans);
    font-weight: 800;
    font-size: 19px;
    letter-spacing: 0.2px;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 2px;
}

.sidebar-brand-sub {
    font-size: 11.5px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin-bottom: 16px;
}

.sidebar-section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 6px 0 10px 0;
}

.sidebar-status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 7px 10px;
    border-radius: 8px;
    margin-bottom: 4px;
    background: var(--panel);
    border: 1px solid var(--border-soft);
}

.sidebar-status-name { font-size: 12.5px; color: var(--text-secondary); }
.sidebar-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; margin-right: 7px; box-shadow: 0 0 8px currentColor; }

.sidebar-stat {
    display: flex;
    justify-content: space-between;
    font-size: 12.5px;
    color: var(--text-secondary);
    padding: 5px 2px;
    border-bottom: 1px dashed var(--border-soft);
}
.sidebar-stat b { font-family: var(--mono); color: var(--text-primary); font-weight: 600; }

.header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 24px;
    background: linear-gradient(135deg, var(--panel) 0%, var(--panel-raised) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}

.header::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--cyan), transparent 60%);
}

.dashboard-title { font-size: 22px; font-weight: 800; color: var(--text-primary); letter-spacing: 0.2px; }
.dashboard-subtitle { font-size: 12.5px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.3px; margin-top: 3px; }
.header-badges { display: flex; gap: 12px; }

.badge {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 9px 18px;
    min-width: 118px;
}
.badge-label { font-size: 10px; color: var(--text-muted); letter-spacing: 1.1px; text-transform: uppercase; margin-bottom: 3px; }
.badge-value { font-family: var(--mono); font-size: 15px; font-weight: 600; color: var(--text-primary); }

.section-title { font-size: 15px; font-weight: 700; color: var(--text-primary); display: flex; align-items: center; gap: 8px; }
.section-title::before {
    content: ""; width: 3px; height: 15px; background: var(--cyan); border-radius: 2px; display: inline-block; box-shadow: 0 0 8px var(--cyan-glow);
}
.section-subtitle { font-size: 12.5px; color: var(--text-muted); margin-top: 3px; margin-left: 11px; }

.kpi-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    position: relative;
    transition: border-color 0.15s ease, transform 0.15s ease;
    height: 100%;
}
.kpi-top { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
.kpi-accent { width: 8px; height: 8px; border-radius: 50%; box-shadow: 0 0 8px currentColor; flex-shrink: 0; }
.kpi-label { font-size: 11px; font-weight: 600; letter-spacing: 0.9px; text-transform: uppercase; color: var(--text-muted); }
.kpi-value { font-family: var(--mono); font-size: 28px; font-weight: 700; color: var(--text-primary); line-height: 1.1; margin-bottom: 6px; }
.kpi-footer { font-size: 12px; color: var(--text-secondary); }

.chart-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px 18px 20px;
    margin-bottom: 16px;
}
.chart-title { font-size: 14px; font-weight: 700; color: var(--text-primary); }
.chart-subtitle { font-size: 12px; color: var(--text-muted); margin-top: 2px; margin-bottom: 12px; }
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.6px;
    padding: 4px 10px;
    border-radius: 100px;
    text-transform: uppercase;
}

.evidence-card { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius-sm); overflow: hidden; margin-bottom: 14px; }
.evidence-caption { padding: 9px 12px; font-size: 11.5px; color: var(--text-secondary); display: flex; justify-content: space-between; border-top: 1px solid var(--border-soft); }

.ai-report { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); padding: 18px 20px; height: 100%; }
.ai-report-title { font-size: 14px; font-weight: 700; margin-bottom: 2px; }
.ai-report-list { margin: 10px 0 0 0; padding-left: 0; list-style: none; }
.ai-report-list li { font-size: 12.5px; color: var(--text-secondary); padding: 7px 0 7px 18px; border-bottom: 1px solid var(--border-soft); position: relative; }
.ai-report-list li::before { content: "›"; position: absolute; left: 0; color: var(--cyan); font-weight: 700; }

.chat-panel-header-standalone { display: flex; align-items: center; gap: 10px; padding: 14px 18px; border: 1px solid var(--border); border-bottom: none; border-top-left-radius: var(--radius); border-top-right-radius: var(--radius); background: linear-gradient(180deg, var(--panel-raised) 0%, var(--panel) 100%); }
.chat-avatar { width: 30px; height: 30px; border-radius: 9px; background: linear-gradient(135deg, var(--cyan), var(--violet)); display: flex; align-items: center; justify-content: center; font-size: 15px; box-shadow: 0 0 14px var(--cyan-glow); flex-shrink: 0; }
.chat-panel-title { font-size: 13.5px; font-weight: 700; color: var(--text-primary); }
.chat-panel-sub { font-size: 11px; color: var(--emerald); display: flex; align-items: center; gap: 5px; }

/* Native st.container(border=False) sitting right below the header --
   styled to look like the body of the same panel, so the header
   (custom HTML) and the message area (native Streamlit widgets) read as
   one continuous bordered box even though they're two separate elements. */
div[data-testid="stVerticalBlockBorderWrapper"] { border: 1px solid var(--border) !important; border-top: none !important; border-bottom-left-radius: var(--radius) !important; border-bottom-right-radius: var(--radius) !important; background: var(--panel) !important; }

/* Native chat message bubbles, restyled to match the dark theme instead
   of Streamlit's default appearance. */
[data-testid="stChatMessage"] { background: transparent !important; }
[data-testid="stChatMessageContent"] { background: var(--panel-hover) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; padding: 10px 14px !important; }
[data-testid="stChatMessage"] table { width: 100%; border-collapse: collapse; font-size: 12px; }
[data-testid="stChatMessage"] th { text-align: left; padding: 8px 10px; color: var(--text-muted); font-size: 10.5px; text-transform: uppercase; border-bottom: 1px solid var(--border); }
[data-testid="stChatMessage"] td { padding: 8px 10px; color: var(--text-secondary); border-bottom: 1px solid var(--border-soft); font-family: var(--mono); }

[data-testid="stChatInput"] { background: var(--panel-raised) !important; border-top: 1px solid var(--border) !important; border-radius: 0 !important; }
[data-testid="stChatInput"] textarea { background: var(--panel-hover) !important; color: var(--text-primary) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }

.incident-table-wrap { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
table.incident-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
table.incident-table th { text-align: left; padding: 10px 16px; color: var(--text-muted); font-size: 10.5px; letter-spacing: 0.8px; text-transform: uppercase; border-bottom: 1px solid var(--border); background: rgba(255,255,255,0.015); }
table.incident-table td { padding: 10px 16px; color: var(--text-secondary); border-bottom: 1px solid var(--border-soft); font-family: var(--mono); font-size: 12px; }
table.incident-table tr:last-child td { border-bottom: none; }

.stSlider label, .stTextInput label { color: var(--text-secondary) !important; font-size: 12.5px !important; }
.stSlider [data-baseweb="slider"] > div > div { background: var(--cyan) !important; }
.stSlider [role="slider"] { background: var(--cyan) !important; box-shadow: 0 0 10px var(--cyan-glow) !important; border: 2px solid var(--canvas) !important; }
div[data-testid="stTextInput"] input, textarea { background: var(--panel-hover) !important; color: var(--text-primary) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
button[kind="secondary"], button[kind="primary"], .stButton button { background: var(--panel-hover) !important; color: var(--text-primary) !important; border: 1px solid var(--border) !important; border-radius: 9px !important; font-size: 12.5px !important; }
button[kind="primary"] { background: linear-gradient(135deg, var(--cyan), #0EA5C4) !important; color: #04141A !important; border: none !important; font-weight: 600 !important; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #2E3747; }
.hr-soft { border: none; border-top: 1px solid var(--border); margin: 26px 0; }
</style>
"""
    # Remove all leading line indentation so markdown engine doesn't break
    cleaned_lines = [line.lstrip() for line in raw_css.splitlines()]
    return "\n".join(cleaned_lines).strip()