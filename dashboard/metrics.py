import os
import pandas as pd

from config import LOGS_DIR

def load_csv(filename):

    path = os.path.join(LOGS_DIR, filename)

    if os.path.exists(path):

        try:
            df = pd.read_csv(path)
            return df

        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def load_all_data():

    return {

        "intrusion": load_csv("events.csv"),

        "loitering": load_csv("loitering_events.csv"),

        "fall": load_csv("fall_events.csv"),

        "counting": load_csv("counting_events.csv"),

        "behavior": load_csv("behavior_analytics.csv"),

        # Milestone 2 output — self-supervised VAD engine anomaly log
        "vad": load_csv("abstract_anomalies.csv"),

    }


# Dashboard metrics
def calculate_metrics(data, vad_alert_threshold: float = 0.65):

    intrusion = data["intrusion"]

    loitering = data["loitering"]

    fall = data["fall"]

    counting = data["counting"]

    behavior = data["behavior"]

    vad = data["vad"]

    
    # Total Events
    

    total_events = len(intrusion) + len(loitering) + len(fall)

    
    # Active Persons
    

    people = set()

    for df in [intrusion, loitering, fall, counting, behavior]:

        if not df.empty and "Person_ID" in df.columns:

            people.update(df["Person_ID"].dropna().tolist())

    active_persons = len(people)

    
    # Current Occupancy
    

    current_occ = 0

    peak_occ = 0

    if not counting.empty and "Current_Occupancy" in counting.columns:

        current_occ = int(counting["Current_Occupancy"].iloc[-1])

        peak_occ = int(counting["Current_Occupancy"].max())

    
    # Average Dwell
    

    avg_dwell = 0

    highest_dwell = 0

    if not behavior.empty and "Dwell_Time_Seconds" in behavior.columns:

        avg_dwell = round(behavior["Dwell_Time_Seconds"].mean(), 1)

        highest_dwell = int(behavior["Dwell_Time_Seconds"].max())

    
    # Evidence Count
    

    evidence_folder = "evidence"

    evidence = 0

    if os.path.exists(evidence_folder):

        evidence = len([
            f for f in os.listdir(evidence_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    
    # Most Active Zone
    

    zones = []

    for df in [intrusion, loitering]:

        if not df.empty and "Zone" in df.columns:

            zones.extend(df["Zone"].dropna().tolist())

    most_zone = pd.Series(zones).mode()[0] if len(zones) else "-"

    
    # Most Frequent Person
    

    persons = []

    for df in [intrusion, loitering, fall]:

        if not df.empty and "Person_ID" in df.columns:

            persons.extend(df["Person_ID"].tolist())

    top_person = pd.Series(persons).mode()[0] if len(persons) else "-"

    
    # Latest Timestamp (across rule-based logs)
    

    timestamps = []

    for df in [intrusion, loitering, fall, counting, behavior]:

        if not df.empty and "Timestamp" in df.columns:

            timestamps.extend(df["Timestamp"].tolist())

    latest = max(timestamps) if len(timestamps) else "-"

    
    # VAD (Video Anomaly Detection) Engine Metrics
   

    vad_events = 0
    vad_latest_score = 0.0
    vad_avg_score = 0.0
    vad_peak_score = 0.0
    vad_critical_count = 0
    vad_status = "OFFLINE"
    vad_latest_zone = "-"
    vad_latest_description = "No behavioral telemetry received yet."

    if not vad.empty and "Anomaly_Score" in vad.columns:

        vad_sorted = vad.copy()

        if "Timestamp" in vad_sorted.columns:
            vad_sorted = vad_sorted.sort_values("Timestamp")

        scores = pd.to_numeric(vad_sorted["Anomaly_Score"], errors="coerce").dropna()

        vad_events = len(vad_sorted)
        vad_avg_score = round(float(scores.mean()), 3) if len(scores) else 0.0
        vad_peak_score = round(float(scores.max()), 3) if len(scores) else 0.0
        vad_critical_count = int((scores >= vad_alert_threshold).sum())

        if len(vad_sorted):

            last_row = vad_sorted.iloc[-1]

            vad_latest_score = round(
                float(pd.to_numeric(last_row.get("Anomaly_Score", 0.0), errors="coerce") or 0.0), 3
            )
            vad_latest_zone = last_row.get("Zone", "-")
            vad_latest_description = last_row.get("Description", "-")

            vad_status = "CRITICAL" if vad_latest_score >= vad_alert_threshold else "NOMINAL"

    
    # Risk Level  (rule-based events + VAD critical crossings)
    

    risk_weight = total_events + (vad_critical_count * 2)

    if risk_weight >= 25:
        risk = "HIGH"
    elif risk_weight >= 10:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {

        "critical_alerts": len(intrusion) + len(fall),

        "intrusions": len(intrusion),

        "loitering": len(loitering),

        "falls": len(fall),

        "total_events": total_events,

        "active_persons": active_persons,

        "current_occupancy": current_occ,

        "peak_occupancy": peak_occ,

        "average_dwell": avg_dwell,

        "highest_dwell": highest_dwell,

        "evidence": evidence,

        "most_active_zone": most_zone,

        "top_person": top_person,

        "latest_event": latest,

        "risk": risk,

        # VAD engine block
        "vad_events": vad_events,

        "vad_latest_score": vad_latest_score,

        "vad_avg_score": vad_avg_score,

        "vad_peak_score": vad_peak_score,

        "vad_critical_count": vad_critical_count,

        "vad_status": vad_status,

        "vad_latest_zone": vad_latest_zone,

        "vad_latest_description": vad_latest_description,

    }
