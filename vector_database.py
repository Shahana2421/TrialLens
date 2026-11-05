
import numpy as np
import faiss
import pandas as pd

print("=" * 45)
print("       TRIALLENS VECTOR DATABASE")
print("=" * 45)

# 1. Load generated embeddings
embeddings = np.load("trial_embeddings.npy")

print("\nEmbeddings loaded:", embeddings.shape)

# 2. Load chunk information
df = pd.read_csv("clinical_trial_chunks.csv")

# 3. Validate the data
if len(embeddings) != len(df):
    raise ValueError(
        "Mismatch: embedding count and chunk count are different."
    )

# 4. Convert embeddings to float32 for FAISS
embeddings = embeddings.astype("float32")

# 5. Normalize vectors for cosine similarity
faiss.normalize_L2(embeddings)

# 6. Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

# 7. Add embeddings to the index
index.add(embeddings)

# 8. Save the FAISS index
faiss.write_index(index, "clinical_trials.index")

# 9. Display database information
print("\nVECTOR DATABASE CREATED SUCCESSFULLY")
print("-" * 45)

print("Total vectors:", index.ntotal)
print("Vector dimensions:", index.d)
print("Index type: IndexFlatIP")
print("Similarity method: Cosine similarity")
print("Saved file: clinical_trials.index")

print("\n" + "=" * 45)
