from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss



print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Model Loaded.")



knowledge_df = pd.read_csv(
    "knowledge/knowledge_base.csv"
)



index = faiss.read_index(
    "knowledge/faiss.index"
)

print(f"Loaded {index.ntotal} vectors.")


print("\n-----------------------------")
print("SEMANTIC CCTV SEARCH")
print("-------------------------------")

print("\nType 'exit' to quit.\n")

while True:

    query = input("Ask: ").strip()

    if query.lower() == "exit":
        break

  

    query_embedding = model.encode(
        [query]
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        5
    )

    print("\nTop Matching Events\n")

    for rank, idx in enumerate(indices[0]):

        if idx == -1:
            continue

        row = knowledge_df.iloc[idx]

        print(f"Result {rank+1}")

        print("----------------------------")

        print(f"Timestamp : {row['Timestamp']}")
        print(f"Event     : {row['Event']}")
        print(f"Person ID : {row['Person_ID']}")
        print(f"Zone      : {row['Zone']}")
        print(f"Dwell     : {row['Dwell_Time']}")
        print(f"Occupancy : {row['Occupancy']}")
        print(f"Evidence  : {row['Evidence']}")

        print(f"Distance  : {distances[0][rank]:.2f}")

        print()