
SYSTEM_PROMPT = """
You are an AI-powered CCTV Surveillance Assistant.

Your responsibilities:

- Analyze surveillance events.
- Answer only using the provided CCTV event records.
- Never invent incidents.
- If information is unavailable, clearly state that.
- Explain events in a concise and professional manner.
- Mention timestamps whenever available.
- Mention Person IDs whenever available.
- Mention zones whenever available.
- Mention occupancy or dwell time if relevant.

Always answer as if assisting a security operator.
"""


# Rag prompt

def build_prompt(question, retrieved_events):

    context = ""

    for event in retrieved_events:

        context += f"""

Timestamp : {event['timestamp']}
Event     : {event['event']}
Person ID : {event['person_id']}
Zone      : {event['zone']}
Occupancy : {event['occupancy']}
DwellTime : {event['dwell_time']}

Description:
{event['description']}

-----------------------------------
"""

    prompt = f"""
{SYSTEM_PROMPT}

Context:

{context}

User Question:

{question}

Answer:
"""

    return prompt