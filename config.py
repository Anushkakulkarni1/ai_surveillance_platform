import os

# ==========================================
# PROJECT ROOT
# ==========================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# FOLDERS
# ==========================================

MODELS_DIR = os.path.join(ROOT_DIR, "models")
VIDEOS_DIR = os.path.join(ROOT_DIR, "videos")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
KNOWLEDGE_DIR = os.path.join(ROOT_DIR, "knowledge")
EVIDENCE_DIR = os.path.join(ROOT_DIR, "evidence")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

# ==========================================
# YOLO MODELS
# ==========================================

YOLO_N = os.path.join(MODELS_DIR, "yolov8n.pt")
YOLO_M = os.path.join(MODELS_DIR, "yolov8m.pt")
YOLO_POSE = os.path.join(MODELS_DIR, "yolov8n-pose.pt")

# ==========================================
# LOG FILES
# ==========================================

EVENTS_LOG = os.path.join(LOGS_DIR, "events.csv")
LOITERING_LOG = os.path.join(LOGS_DIR, "loitering_events.csv")
COUNTING_LOG = os.path.join(LOGS_DIR, "counting_events.csv")
FALL_LOG = os.path.join(LOGS_DIR, "fall_events.csv")
BEHAVIOR_LOG = os.path.join(LOGS_DIR, "behavior_analytics.csv")

# ==========================================
# KNOWLEDGE FILES
# ==========================================

KNOWLEDGE_BASE = os.path.join(
    KNOWLEDGE_DIR,
    "knowledge_base.csv"
)

EMBEDDINGS = os.path.join(
    KNOWLEDGE_DIR,
    "event_embeddings.npy"
)

FAISS_INDEX = os.path.join(
    KNOWLEDGE_DIR,
    "faiss.index"
)