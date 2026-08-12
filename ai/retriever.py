from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import re

from config import KNOWLEDGE_BASE, FAISS_INDEX


class SurveillanceRetriever:

    def __init__(self):

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.index = faiss.read_index(
            FAISS_INDEX
        )

        self.knowledge = pd.read_csv(
            KNOWLEDGE_BASE
        )

 

    def detect_intent(self, query):

        query = query.lower()

        intents = {

            "Intrusion": [
                "intrusion",
                "restricted",
                "unauthorized",
                "entered",
                "entry",
                "trespass",
                "crossed boundary",
                "security breach",
                "illegal entry"
            ],

            "Loitering": [
                "loiter",
                "lingering",
                "wander",
                "wandering",
                "stayed",
                "too long",
                "idle"
            ],

            "Fall": [
                "fall",
                "fallen",
                "collapse",
                "collapsed",
                "slipped"
            ],

            "Occupancy": [
                "occupancy",
                "count",
                "crowd",
                "people count",
                "how many people"
            ],

            "Behavior": [
                "behavior",
                "behaviour",
                "tracking",
                "movement",
                "dwell",
                "visible"
            ]
        }

        for event, keywords in intents.items():

            for keyword in keywords:

                if keyword in query:
                    return event

        return None


    def detect_zone(self, query):

        query = query.upper()

        match = re.search(r"ZONE[_ ]?([A-Z])", query)

        if match:
            return f"ZONE_{match.group(1)}"

        return None


    def search(self, query, top_k=10):

        query_embedding = self.model.encode(
            [query]
        ).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        detected_event = self.detect_intent(query)
        detected_zone = self.detect_zone(query)

        results = []

        for rank, idx in enumerate(indices[0]):

            if idx == -1:
                continue

            row = self.knowledge.iloc[idx]

            # Event filter

            if detected_event is not None:

                if str(row["Event"]) != detected_event:
                    continue

           
            # Zone Filter
           

            if detected_zone is not None:

                if str(row["Zone"]).upper() != detected_zone:
                    continue

            results.append({

                "rank": len(results) + 1,

                "score": round(float(distances[0][rank]), 3),

                "timestamp": row["Timestamp"],

                "event": row["Event"],

                "person_id": row["Person_ID"],

                "zone": row["Zone"],

                "dwell_time": row["Dwell_Time"],

                "occupancy": row["Occupancy"],

                "evidence": row["Evidence"],

                "description": row["Description"]

            })

        return results