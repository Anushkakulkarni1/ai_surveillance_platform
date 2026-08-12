

import os
import pandas as pd
import plotly.graph_objects as go



PANEL = "#131720"
GRID = "#232A38"
TEXT_PRIMARY = "#EDF1F7"
TEXT_MUTED = "#6B7385"

CYAN = "#22D3EE"
CRIMSON = "#FB3B5A"
AMBER = "#F5A623"
EMERALD = "#2DD4A7"
VIOLET = "#8B7CF6"

EVENT_COLORS = {
    "Intrusion": CRIMSON,
    "Loitering": AMBER,
    "Fall": VIOLET,
}

EVIDENCE_DIR = "evidence"




def _apply_dark_layout(fig: go.Figure, height: int = 320, show_legend: bool = False) -> go.Figure:

    fig.update_layout(

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(family="Inter, sans-serif", color=TEXT_MUTED, size=12),

        margin=dict(l=10, r=10, t=10, b=10),

        height=height,

        showlegend=show_legend,

        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(color=TEXT_MUTED, size=11),
        ),

        hoverlabel=dict(
            bgcolor=PANEL,
            bordercolor=GRID,
            font=dict(color=TEXT_PRIMARY, family="JetBrains Mono, monospace", size=12),
        ),

    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        color=TEXT_MUTED,
        linecolor=GRID,
        tickfont=dict(size=11),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        zeroline=False,
        color=TEXT_MUTED,
        tickfont=dict(size=11),
    )

    return fig


def _empty_figure(message: str) -> go.Figure:

    fig = go.Figure()

    fig.add_annotation(
        text=message,
        showarrow=False,
        font=dict(color=TEXT_MUTED, size=13),
        xref="paper", yref="paper",
        x=0.5, y=0.5,
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return _apply_dark_layout(fig, height=280)


def incident_timeline_chart(data: dict) -> go.Figure:

    frames = []

    for key, label in [("intrusion", "Intrusion"), ("loitering", "Loitering"), ("fall", "Fall")]:

        df = data.get(key)

        if df is not None and not df.empty and "Timestamp" in df.columns:

            tmp = df.copy()
            tmp["Event"] = label
            frames.append(tmp[["Timestamp", "Event"]])

    if not frames:
        raise ValueError("No incident data available.")

    combined = pd.concat(frames, ignore_index=True)
    combined["Timestamp"] = pd.to_datetime(combined["Timestamp"], errors="coerce")
    combined = combined.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    combined["Bucket"] = combined["Timestamp"].dt.floor("h")

    grouped = (
        combined.groupby(["Bucket", "Event"]).size().reset_index(name="Count")
    )

    fig = go.Figure()

    for label in ["Intrusion", "Loitering", "Fall"]:

        subset = grouped[grouped["Event"] == label]

        if subset.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=subset["Bucket"],
                y=subset["Count"],
                mode="lines+markers",
                name=label,
                line=dict(color=EVENT_COLORS[label], width=2.4, shape="spline"),
                marker=dict(size=6, color=EVENT_COLORS[label]),
                fill="tozeroy",
                fillcolor=EVENT_COLORS[label].replace(")", ", 0.08)").replace("rgb", "rgba")
                if EVENT_COLORS[label].startswith("rgb") else _hex_to_rgba(EVENT_COLORS[label], 0.08),
            )
        )

    return _apply_dark_layout(fig, height=320, show_legend=True)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:

    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"



def event_distribution(data: dict) -> go.Figure:

    counts = {
        "Intrusion": len(data.get("intrusion", pd.DataFrame())),
        "Loitering": len(data.get("loitering", pd.DataFrame())),
        "Fall": len(data.get("fall", pd.DataFrame())),
    }

    counts = {k: v for k, v in counts.items() if v > 0}

    if not counts:
        raise ValueError("No event data available.")

    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(counts.keys()),
                values=list(counts.values()),
                hole=0.62,
                marker=dict(
                    colors=[EVENT_COLORS[k] for k in counts.keys()],
                    line=dict(color=PANEL, width=3),
                ),
                textfont=dict(color=TEXT_PRIMARY, size=12),
                textinfo="label+percent",
                hoverinfo="label+value",
            )
        ]
    )

    total = sum(counts.values())

    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:11px;color:{TEXT_MUTED}'>EVENTS</span>",
        showarrow=False,
        font=dict(color=TEXT_PRIMARY, size=20, family="JetBrains Mono, monospace"),
        x=0.5, y=0.5,
    )

    return _apply_dark_layout(fig, height=320, show_legend=True)



def zone_distribution(data: dict) -> go.Figure:

    frames = []

    for key in ["intrusion", "loitering"]:

        df = data.get(key)

        if df is not None and not df.empty and "Zone" in df.columns:
            frames.append(df["Zone"])

    if not frames:
        raise ValueError("No zone data available.")

    zones = pd.concat(frames, ignore_index=True).value_counts().sort_values(ascending=True)

    fig = go.Figure(
        go.Bar(
            x=zones.values,
            y=zones.index,
            orientation="h",
            marker=dict(
                color=zones.values,
                colorscale=[[0, "#0EA5C4"], [1, CYAN]],
                line=dict(width=0),
            ),
            text=zones.values,
            textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, size=11),
        )
    )

    return _apply_dark_layout(fig, height=320)




def occupancy_chart(data: dict) -> go.Figure:

    df = data.get("counting")

    if df is None or df.empty or "Current_Occupancy" not in df.columns:
        raise ValueError("No occupancy data available.")

    tmp = df.copy()
    tmp["Timestamp"] = pd.to_datetime(tmp["Timestamp"], errors="coerce")
    tmp = tmp.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=tmp["Timestamp"],
            y=tmp["Current_Occupancy"],
            mode="lines",
            line=dict(color=CYAN, width=2.4, shape="spline"),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(CYAN, 0.12),
            name="Occupancy",
        )
    )

    peak = tmp["Current_Occupancy"].max()

    fig.add_hline(
        y=peak,
        line=dict(color=AMBER, width=1, dash="dot"),
        annotation_text=f"Peak {peak}",
        annotation_font=dict(color=AMBER, size=10),
        annotation_position="top left",
    )

    return _apply_dark_layout(fig, height=320)




def dwell_chart(data: dict) -> go.Figure:

    df = data.get("behavior")

    if df is None or df.empty or "Dwell_Time_Seconds" not in df.columns:
        raise ValueError("No dwell analytics available.")

    tmp = df.copy()

    if "Zone" in tmp.columns:
        grouped = tmp.groupby("Zone")["Dwell_Time_Seconds"].mean().sort_values(ascending=False)
        x_labels = grouped.index
    else:
        grouped = tmp["Dwell_Time_Seconds"]
        x_labels = [f"#{i+1}" for i in range(len(grouped))]

    fig = go.Figure(
        go.Bar(
            x=list(x_labels),
            y=grouped.values,
            marker=dict(color=VIOLET, line=dict(width=0)),
            text=[f"{v:.0f}s" for v in grouped.values],
            textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, size=11),
        )
    )

    return _apply_dark_layout(fig, height=320)


# Top person activity

def top_person_chart(data: dict) -> go.Figure:

    frames = []

    for key in ["intrusion", "loitering", "fall"]:

        df = data.get(key)

        if df is not None and not df.empty and "Person_ID" in df.columns:
            frames.append(df["Person_ID"])

    if not frames:
        raise ValueError("No person activity available.")

    counts = pd.concat(frames, ignore_index=True).value_counts().head(8).sort_values()

    fig = go.Figure(
        go.Bar(
            x=counts.values,
            y=[f"Person {p}" for p in counts.index],
            orientation="h",
            marker=dict(color=EMERALD, line=dict(width=0)),
            text=counts.values,
            textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, size=11),
        )
    )

    return _apply_dark_layout(fig, height=320)



# Hourly heat bar


def hourly_activity_chart(data: dict) -> go.Figure:

    frames = []

    for key in ["intrusion", "loitering", "fall"]:

        df = data.get(key)

        if df is not None and not df.empty and "Timestamp" in df.columns:
            frames.append(df["Timestamp"])

    if not frames:
        raise ValueError("No activity data available.")

    ts = pd.to_datetime(pd.concat(frames, ignore_index=True), errors="coerce").dropna()
    hourly = ts.dt.hour.value_counts().sort_index()

    fig = go.Figure(
        go.Bar(
            x=[f"{h:02d}:00" for h in hourly.index],
            y=hourly.values,
            marker=dict(color=CYAN, line=dict(width=0)),
        )
    )

    return _apply_dark_layout(fig, height=300)



# LIVE VAD RECONSTRUCTION, ERROR TRACKING CHART


def vad_live_chart(vad_df: pd.DataFrame, threshold: float = 0.65) -> go.Figure:

    if vad_df is None or vad_df.empty or "Anomaly_Score" not in vad_df.columns:
        return _empty_figure("No VAD telemetry received yet.")

    tmp = vad_df.copy()

    if "Timestamp" in tmp.columns:
        tmp["Timestamp"] = pd.to_datetime(tmp["Timestamp"], errors="coerce")
        tmp = tmp.dropna(subset=["Timestamp"]).sort_values("Timestamp")
        x = tmp["Timestamp"]
    else:
        x = list(range(len(tmp)))

    y = pd.to_numeric(tmp["Anomaly_Score"], errors="coerce").fillna(0.0)

    fig = go.Figure()

    # Base trace: cyan "nominal" line + soft fill
    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(color=CYAN, width=2),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(CYAN, 0.08),
            name="Reconstruction Error",
            hovertemplate="%{y:.3f}<extra></extra>",
        )
    )

   
    breach_mask = y >= threshold

    if breach_mask.any():

        fig.add_trace(
            go.Scatter(
                x=x[breach_mask], y=y[breach_mask],
                mode="markers",
                marker=dict(
                    color=CRIMSON, size=8, symbol="circle",
                    line=dict(color="#04141A", width=1),
                ),
                name="Threshold Breach",
                hovertemplate="⚠ %{y:.3f}<extra></extra>",
            )
        )

    # Adjustable safety threshold line
    fig.add_hline(
        y=threshold,
        line=dict(color=CRIMSON, width=1.6, dash="dash"),
        annotation_text=f"SAFETY THRESHOLD  {threshold:.2f}",
        annotation_font=dict(color=CRIMSON, size=10, family="JetBrains Mono, monospace"),
        annotation_position="top left",
    )

    fig.update_yaxes(range=[0, 1.05], title=None)

    return _apply_dark_layout(fig, height=340, show_legend=True)



# Latest evidence


def latest_evidence(limit: int = 6):

    if not os.path.isdir(EVIDENCE_DIR):
        return []

    files = [
        os.path.join(EVIDENCE_DIR, f)
        for f in os.listdir(EVIDENCE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    files.sort(key=os.path.getmtime, reverse=True)

    return files[:limit]
