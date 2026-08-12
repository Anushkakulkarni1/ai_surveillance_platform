import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import (
    LOGS_DIR,
    KNOWLEDGE_DIR,
    KNOWLEDGE_BASE,
    EVIDENCE_DIR
)

LOG_FOLDER = LOGS_DIR
OUTPUT_FOLDER = KNOWLEDGE_DIR

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def load_csv(filename):

    path = os.path.join(LOG_FOLDER, filename)

    if os.path.exists(path):

        try:
            return pd.read_csv(path)

        except:
            return pd.DataFrame()

    return pd.DataFrame()


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)  # ai/ -> project root


def find_evidence(prefix, person_id, timestamp):
   

    if not os.path.exists(EVIDENCE_DIR):
        return ""

    try:
       
        # f"evidence/{prefix}_{person_id}_{timestamp:%Y%m%d_%H%M%S}.jpg"
        parsed = pd.to_datetime(timestamp)
        expected_filename = f"{prefix}_{person_id}_{parsed.strftime('%Y%m%d_%H%M%S')}.jpg"
    except (ValueError, TypeError):
        return ""

    expected_path = os.path.join(EVIDENCE_DIR, expected_filename)

    if os.path.exists(expected_path):
        return os.path.relpath(expected_path, PROJECT_ROOT).replace("\\", "/")

    return ""


intrusion_df = load_csv("events.csv")
loitering_df = load_csv("loitering_events.csv")
fall_df = load_csv("fall_events.csv")
counting_df = load_csv("counting_events.csv")
behavior_df = load_csv("behavior_analytics.csv")

knowledge_rows = []



if not intrusion_df.empty:

    for _, row in intrusion_df.iterrows():

        timestamp = row.get("Timestamp", "")
        person = row.get("Person_ID", "")
        zone = row.get("Zone", "")

        evidence = find_evidence("intrusion", person, timestamp)

        description = (
            f"At {timestamp}, Person {person} "
            f"performed an intrusion inside {zone}. "
            f"This event indicates unauthorized access."
        )

        if evidence != "":

            description += (
                f" Evidence image: {os.path.basename(evidence)}."
            )

        knowledge_rows.append({

            "Timestamp": timestamp,
            "Event": "Intrusion",
            "Person_ID": person,
            "Zone": zone,
            "Dwell_Time": "",
            "Occupancy": "",
            "Evidence": evidence,
            "Description": description

        })

if not loitering_df.empty:

    for _, row in loitering_df.iterrows():

        timestamp = row.get("Timestamp", "")
        person = row.get("Person_ID", "")
        zone = row.get("Zone", "")

        evidence = find_evidence("loitering", person, timestamp)

        description = (
            f"At {timestamp}, Person {person} "
            f"was detected loitering in {zone}. "
            f"The person remained in the monitored area "
            f"longer than the allowed threshold."
        )

        if evidence != "":

            description += (
                f" Evidence image: {os.path.basename(evidence)}."
            )

        knowledge_rows.append({

            "Timestamp": timestamp,
            "Event": "Loitering",
            "Person_ID": person,
            "Zone": zone,
            "Dwell_Time": "",
            "Occupancy": "",
            "Evidence": evidence,
            "Description": description

        })

if not fall_df.empty:

    for _, row in fall_df.iterrows():

        timestamp = row.get("Timestamp", "")
        person = row.get("Person_ID", "")

        evidence = find_evidence("fall", person, timestamp)

        description = (
            f"At {timestamp}, Person {person} "
            f"experienced a fall event detected by the "
            f"surveillance system."
        )

        if evidence != "":

            description += (
                f" Evidence image: {os.path.basename(evidence)}."
            )

        knowledge_rows.append({

            "Timestamp": timestamp,
            "Event": "Fall",
            "Person_ID": person,
            "Zone": "",
            "Dwell_Time": "",
            "Occupancy": "",
            "Evidence": evidence,
            "Description": description

        })


if not counting_df.empty:

    for _, row in counting_df.iterrows():

        timestamp = row.get("Timestamp", "")
        person = row.get("Person_ID", "")
        occ = row.get("Current_Occupancy", "")

        description = (
            f"At {timestamp}, occupancy monitoring "
            f"recorded Person {person}. "
            f"Current occupancy was {occ} people."
        )

        knowledge_rows.append({

            "Timestamp": timestamp,
            "Event": row.get("Event", "Occupancy"),
            "Person_ID": person,
            "Zone": "",
            "Dwell_Time": "",
            "Occupancy": occ,
            "Evidence": "",
            "Description": description

        })


if not behavior_df.empty:

    for _, row in behavior_df.iterrows():

        timestamp = row.get("Timestamp", "")
        person = row.get("Person_ID", "")
        dwell = row.get("Dwell_Time_Seconds", "")

        description = (
            f"Behavior analytics measured that "
            f"Person {person} remained visible for "
            f"{dwell} seconds at {timestamp}."
        )

        knowledge_rows.append({

            "Timestamp": timestamp,
            "Event": "Behavior",
            "Person_ID": person,
            "Zone": "",
            "Dwell_Time": dwell,
            "Occupancy": "",
            "Evidence": "",
            "Description": description

        })

knowledge_df = pd.DataFrame(knowledge_rows)

knowledge_df.to_csv(
    KNOWLEDGE_BASE,
    index=False
)

print("\nKnowledge Base Updated Successfully")

print(knowledge_df.head())