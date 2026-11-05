
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("=" * 45)
print("      TRIALLENS EMBEDDING GENERATION")
print("=" * 45)

# 1. Load clinical trial chunks
df = pd.read_csv("clinical_trial_chunks.csv")

print("\nTotal chunks loaded:", len(df))

# 2. Extract text for embedding
texts = df["Chunk"].fillna("").astype(str).tolist()

# 3. Load Sentence Transformer model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("\nLoading embedding model:", MODEL_NAME)

model = SentenceTransformer(MODEL_NAME)

# 4. Generate embeddings
print("\nGenerating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    batch_size=32,
    normalize_embeddings=True
)

# 5. Save embeddings
np.save("trial_embeddings.npy", embeddings)

# 6. Display results
print("\nEMBEDDING GENERATION COMPLETED")
print("-" * 45)

print("Total text chunks:", len(texts))
print("Embedding shape:", embeddings.shape)
print("Embedding dimensions:", embeddings.shape[1])
print("Embeddings saved as: trial_embeddings.npy")

print("\n" + "=" * 45)
