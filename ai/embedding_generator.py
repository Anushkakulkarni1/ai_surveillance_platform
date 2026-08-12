import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from config import KNOWLEDGE_BASE, EMBEDDINGS
import pandas as pd
import numpy as np





print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Model Loaded Successfully.")



df = pd.read_csv(KNOWLEDGE_BASE)





texts = df["Description"].fillna("").tolist()





print("Generating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True
)





os.makedirs(
    os.path.dirname(EMBEDDINGS),
    exist_ok=True
)

np.save(
    EMBEDDINGS,
    embeddings
)

print("\nEmbeddings Saved Successfully!")

print(f"Total Events : {len(texts)}")

print(f"Embedding Shape : {embeddings.shape}")