
import pandas as pd

from rag_pipeline import (
    embedding_model,
    index,
    chunks_df,
    condition_matches
)

QUERY = "Type 2 diabetes insulin therapy and blood glucose control"

CONDITION = "Type 2 Diabetes"
STATUS = "RECRUITING"

ALLOWED_SECTIONS = ["Title", "Summary"]

print("\nTRIALLENS RETRIEVAL DIAGNOSTICS")
print("=" * 65)

# Generate query embedding
query_embedding = embedding_model.encode(
    [QUERY],
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

# Search complete FAISS index
scores, indices = index.search(
    query_embedding,
    index.ntotal
)

candidates = []

for score, idx in zip(scores[0], indices[0]):

    if idx == -1:
        continue

    row = chunks_df.iloc[idx]

    # Actual condition validation
    if not condition_matches(row["Conditions"], CONDITION):
        continue

    # Exact recruitment status
    if str(row["Status"]).strip().upper() != STATUS:
        continue

    # Only Title and Summary sections
    if row["Section"] not in ALLOWED_SECTIONS:
        continue

    candidates.append({
        "NCTId": row["NCTId"],
        "Title": row["Title"],
        "Conditions": row["Conditions"],
        "Status": row["Status"],
        "Section": row["Section"],
        "Score": float(score),
        "Chunk": row["Chunk"]
    })

if not candidates:

    print("\nNo candidates passed the filters.")

else:

    results = pd.DataFrame(candidates)

    # Keep the highest-scoring chunk per trial
    results = results.sort_values(
        by="Score",
        ascending=False
    ).drop_duplicates(
        subset="NCTId",
        keep="first"
    )

    print("\nTrials passing all filters:", len(results))

    print("\nTOP 20 CANDIDATES (including scores below threshold)")
    print("=" * 65)

    for rank, (_, row) in enumerate(
        results.head(20).iterrows(),
        start=1
    ):

        print(f"\nRank: {rank}")
        print("NCT ID:", row["NCTId"])
        print("Title:", row["Title"])
        print("Section:", row["Section"])
        print("Similarity:", round(row["Score"], 4))
        print("Conditions:", row["Conditions"])
        print("Text:", row["Chunk"][:300])
        print("-" * 65)

    results.to_csv(
        "retrieval_debug_results.csv",
        index=False
    )

    print("\nSaved: retrieval_debug_results.csv")

print("\n" + "=" * 65)
