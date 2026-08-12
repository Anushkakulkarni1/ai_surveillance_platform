from __future__ import annotations

import re
from typing import List, Dict, Any, Optional

import pandas as pd


def _extract_zone(query: str) -> Optional[str]:
    match = re.search(r"zone[\s_-]*([a-zA-Z])", query, re.IGNORECASE)
    if match:
        return f"ZONE_{match.group(1).upper()}"
    return None


def _filter_zone(df: pd.DataFrame, zone: Optional[str]) -> pd.DataFrame:
    if zone is None or df.empty or "Zone" not in df.columns:
        return df
    return df[df["Zone"].astype(str).str.upper() == zone]


def _extract_person_id(query: str) -> Optional[int]:
    match = re.search(r"person[\s_#]*([0-9]+)", query, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _extract_count(query: str, default: int = 5) -> int:
    match = re.search(r"\b(?:last|recent|latest|top)\s+([0-9]+)\b", query, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return default


def _text(content: str) -> Dict[str, Any]:
    return {"type": "text", "content": content}


def _table(title: str, df: pd.DataFrame) -> Dict[str, Any]:
    return {"type": "table", "title": title, "data": df}


def _no_data(label: str) -> List[Dict[str, Any]]:
    return [_text(f"No {label} data available.")]


INTRUSION_WORDS = ("intrusion", "intruder", "break-in", "breakin", "unauthorized", "trespass")
LOITERING_WORDS = ("loiter", "linger", "hanging around", "staying too long")
FALL_WORDS = ("fall", "fallen", "collapse", "collapsed", "down on the ground")
COUNT_WORDS = ("occupancy", "how many people", "headcount", "crowd")
DWELL_WORDS = ("dwell", "stay", "stayed", "duration", "how long")
SUMMARY_WORDS = ("summary", "overview", "report", "situation", "status", "recap")
LATEST_WORDS = ("latest", "most recent", "last event", "newest", "recent", "last ")
ZONE_RANK_WORDS = ("busiest zone", "most active zone", "which zone", "zone with the most")
PERSON_WORDS = ("person",)
COMPARE_WORDS = ("compare", "vs", "versus", "difference between")
HELP_WORDS = ("help", "what can you", "what can i ask", "commands", "options")


def _matches_any(query: str, words: tuple) -> bool:
    return any(w in query for w in words)


def _is_count_question(query: str) -> bool:
    return bool(re.search(r"\bhow many\b|\bcount\b|\btotal\b|\bnumber of\b", query))


def _handle_event_type(query: str, df: pd.DataFrame, label: str) -> List[Dict[str, Any]]:
    zone = _extract_zone(query)
    filtered = _filter_zone(df, zone)

    zone_note = f" in {zone}" if zone else ""

    if _is_count_question(query):
        return [_text(f"**{len(filtered)}** {label} event(s){zone_note}.")]

    if filtered.empty:
        return [_text(f"No {label} events found{zone_note}.")]

    return [_table(f"{label.title()} Events{zone_note}", filtered)]


def _handle_dwell(query: str, behavior_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if behavior_df.empty or "Dwell_Time_Seconds" not in behavior_df.columns:
        return _no_data("dwell time")

    if "longest" in query or "maximum" in query or "max" in query:
        row = behavior_df.loc[behavior_df["Dwell_Time_Seconds"].idxmax()]
        return [_text(f"**Longest stay:** Person {row['Person_ID']} — {round(row['Dwell_Time_Seconds'], 2)} sec")]

    if "shortest" in query or "minimum" in query or "min" in query:
        row = behavior_df.loc[behavior_df["Dwell_Time_Seconds"].idxmin()]
        return [_text(f"**Shortest stay:** Person {row['Person_ID']} — {round(row['Dwell_Time_Seconds'], 2)} sec")]

    if "average" in query or "mean" in query:
        avg = round(behavior_df["Dwell_Time_Seconds"].mean(), 2)
        return [_text(f"**Average dwell time:** {avg} sec across {len(behavior_df)} tracked people.")]

    return [_table("Dwell Time by Person", behavior_df)]


def _handle_occupancy(query: str, counting_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if counting_df.empty or "Current_Occupancy" not in counting_df.columns:
        return _no_data("occupancy")

    if "maximum" in query or "max" in query or "peak" in query:
        return [_text(f"**Peak occupancy:** {counting_df['Current_Occupancy'].max()}")]

    if "minimum" in query or "min" in query or "lowest" in query:
        return [_text(f"**Lowest occupancy:** {counting_df['Current_Occupancy'].min()}")]

    if "average" in query or "mean" in query:
        return [_text(f"**Average occupancy:** {round(counting_df['Current_Occupancy'].mean(), 1)}")]

    return [_text(f"**Current occupancy:** {counting_df.iloc[-1]['Current_Occupancy']}")]


def _handle_person(query: str, data: dict) -> List[Dict[str, Any]]:
    person_id = _extract_person_id(query)
    if person_id is None:
        return [_text("Try a query like: 'show events for person 1' or 'what did person 3 do'.")]

    blocks: List[Dict[str, Any]] = []
    for label, key in [("Intrusion", "intrusion"), ("Loitering", "loitering"), ("Fall", "fall")]:
        df = data.get(key, pd.DataFrame())
        if not df.empty and "Person_ID" in df.columns:
            matches = df[df["Person_ID"] == person_id]
            if len(matches):
                blocks.append(_table(f"{label} Events - Person {person_id}", matches))

    behavior_df = data.get("behavior", pd.DataFrame())
    if not behavior_df.empty and "Person_ID" in behavior_df.columns:
        matches = behavior_df[behavior_df["Person_ID"] == person_id]
        if len(matches):
            blocks.append(_table(f"Dwell Time - Person {person_id}", matches))

    if not blocks:
        return [_text(f"No events found for person {person_id}.")]
    return blocks


def _handle_latest(query: str, data: dict) -> List[Dict[str, Any]]:
    frames = [
        data.get(k, pd.DataFrame()) for k in ("intrusion", "loitering", "fall")
    ]
    frames = [df for df in frames if not df.empty and "Timestamp" in df.columns]
    if not frames:
        return _no_data("event")

    combined = pd.concat(frames, ignore_index=True)
    combined["Timestamp"] = pd.to_datetime(combined["Timestamp"], errors="coerce")
    combined = combined.dropna(subset=["Timestamp"]).sort_values("Timestamp", ascending=False)
    if combined.empty:
        return _no_data("event")

    n = _extract_count(query, default=1)
    return [_table(f"{'Latest Event' if n == 1 else f'Latest {n} Events'}", combined.head(n))]


def _handle_zone_ranking(query: str, data: dict) -> List[Dict[str, Any]]:
    frames = []
    for key in ("intrusion", "loitering"):
        df = data.get(key, pd.DataFrame())
        if not df.empty and "Zone" in df.columns:
            frames.append(df["Zone"])
    if not frames:
        return _no_data("zone activity")

    counts = pd.concat(frames, ignore_index=True).value_counts()
    top_zone = counts.idxmax()
    lines = [f"**Busiest zone: {top_zone}** ({counts.max()} events)", ""]
    for zone, count in counts.items():
        lines.append(f"- {zone}: {count}")
    return [_text("\n".join(lines))]


def _handle_summary(data: dict) -> List[Dict[str, Any]]:
    intrusion_df = data.get("intrusion", pd.DataFrame())
    loitering_df = data.get("loitering", pd.DataFrame())
    fall_df = data.get("fall", pd.DataFrame())
    counting_df = data.get("counting", pd.DataFrame())
    behavior_df = data.get("behavior", pd.DataFrame())

    lines = [
        "**Security Summary**",
        "",
        f"- Intrusions: {len(intrusion_df)}",
        f"- Loitering: {len(loitering_df)}",
        f"- Falls: {len(fall_df)}",
    ]
    if not counting_df.empty and "Current_Occupancy" in counting_df.columns:
        lines.append(f"- Peak Occupancy: {counting_df['Current_Occupancy'].max()}")
    if not behavior_df.empty and "Dwell_Time_Seconds" in behavior_df.columns:
        lines.append(f"- Average Dwell Time: {round(behavior_df['Dwell_Time_Seconds'].mean(), 2)} sec")

    return [_text("\n".join(lines))]


def _handle_help() -> List[Dict[str, Any]]:
    return [_text(
        "**I can help with things like:**\n\n"
        "- 'show all intrusions' / 'show intrusions in zone A'\n"
        "- 'how many loitering events in zone B'\n"
        "- 'show all falls'\n"
        "- 'who stayed longest' / 'average dwell time'\n"
        "- 'peak occupancy' / 'current occupancy'\n"
        "- 'busiest zone'\n"
        "- 'latest event' / 'last 5 events'\n"
        "- 'show events for person 1'\n"
        "- 'security summary'"
    )]


def generate_fallback_answer(query: str, data: dict) -> List[Dict[str, Any]]:
    q = query.lower().strip()

    intrusion_df = data.get("intrusion", pd.DataFrame())
    loitering_df = data.get("loitering", pd.DataFrame())
    fall_df = data.get("fall", pd.DataFrame())
    counting_df = data.get("counting", pd.DataFrame())
    behavior_df = data.get("behavior", pd.DataFrame())

    if _matches_any(q, ZONE_RANK_WORDS):
        return _handle_zone_ranking(q, data)

    if "person" in q and _extract_person_id(q) is not None:
        return _handle_person(q, data)

    if _matches_any(q, INTRUSION_WORDS):
        return _handle_event_type(q, intrusion_df, "intrusion")

    if _matches_any(q, LOITERING_WORDS):
        return _handle_event_type(q, loitering_df, "loitering")

    if _matches_any(q, FALL_WORDS):
        return _handle_event_type(q, fall_df, "fall")

    if _matches_any(q, DWELL_WORDS):
        return _handle_dwell(q, behavior_df)

    if _matches_any(q, COUNT_WORDS) or "occupancy" in q:
        return _handle_occupancy(q, counting_df)

    if _matches_any(q, LATEST_WORDS):
        return _handle_latest(q, data)

    if _matches_any(q, HELP_WORDS):
        return _handle_help()

    if _matches_any(q, SUMMARY_WORDS):
        return _handle_summary(data)

    return [_text(
        "I couldn't match that to a known query type. Ask me for 'help' "
        "to see everything I can answer."
    )]
