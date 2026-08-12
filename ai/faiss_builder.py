import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EMBEDDINGS, FAISS_INDEX
import numpy as np
import faiss

embeddings = np.load(EMBEDDINGS)

print(f"Loaded {len(embeddings)} embeddings.")
embeddings = embeddings.astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print(f"Indexed {index.ntotal} vectors.")





os.makedirs(
    os.path.dirname(FAISS_INDEX),
    exist_ok=True
)

faiss.write_index(
    index,
    FAISS_INDEX
)

print("\nFAISS Index Saved Successfully!")

print(f"Vector Dimension : {dimension}")
print(f"Total Vectors    : {index.ntotal}")