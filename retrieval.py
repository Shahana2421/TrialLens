
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


print("=" * 50)
print("       TRIALLENS IMPROVED RETRIEVAL")
print("=" * 50)


# 1. Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MIN_SIMILARITY_SCORE = 0.30


# 2. Load embedding model
print("\nLoading embedding model...")
model = SentenceTransformer(MODEL_NAME)


# 3. Load FAISS index
print("Loading FAISS index...")
index = faiss.read_index("clinical_trials.index")

print("FAISS vectors:", index.ntotal)


# 4. Load chunk metadata
chunks_df = pd.read_csv("clinical_trial_chunks.csv")

if len(chunks_df) != index.ntotal:
    raise ValueError("FAISS index and chunk records do not match.")

required_columns = [
    "NCTId",
    "Chunk",
    "ConditionSearched",
    "Status"
]

for column in required_columns:
    if column not in chunks_df.columns:
        raise ValueError(f"Missing required column: {column}")

print("Chunk records:", len(chunks_df))


# 5. Retrieve relevant clinical trials
def retrieve_trials(
    query,
    top_k=5,
    condition=None,
    status=None,
    min_score=MIN_SIMILARITY_SCORE
):

    # Validate query
    if not isinstance(query, str) or not query.strip():
        print("\nPlease enter a valid search query.")
        return pd.DataFrame()

    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    # Validate condition
    if condition and condition != "All":

        available_conditions = (
            chunks_df["ConditionSearched"]
            .dropna()
            .astype(str)
            .str.lower()
            .unique()
        )

        if condition.lower() not in available_conditions:
            print("\nCondition not found.")
            return pd.DataFrame()

    # Validate exact recruitment status
    if status and status != "All":

        available_statuses = (
            chunks_df["Status"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
        )

        status = status.strip().upper().replace(" ", "_")

        if status not in available_statuses:
            print("\nRecruitment status not found in dataset.")
            return pd.DataFrame()

    # Convert query into embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # Search complete FAISS index
    scores, indices = index.search(
        query_embedding,
        index.ntotal
    )

    results = []

    # 6. Collect matching chunks
    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        row = chunks_df.iloc[idx]

        # Condition filtering
        if condition and condition != "All":

            if str(row["ConditionSearched"]).lower() != condition.lower():
                continue

        # Exact status filtering
        if status and status != "All":

            if str(row["Status"]).upper() != status:
                continue

        # Similarity threshold
        if min_score is not None and score < min_score:
            continue

        result = row.to_dict()
        result["SimilarityScore"] = float(score)

        results.append(result)

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # 7. Rank trials using best matching chunk
    trial_results = (
        results_df
        .sort_values(
            by="SimilarityScore",
            ascending=False
        )
        .groupby("NCTId", as_index=False, sort=False)
        .first()
    )

    # 8. Count matching chunks
    match_counts = (
        results_df
        .groupby("NCTId")
        .size()
        .rename("MatchedChunkCount")
    )

    trial_results = trial_results.merge(
        match_counts,
        on="NCTId",
        how="left"
    )

    # 9. Sort and return final results
    trial_results = trial_results.sort_values(
        by="SimilarityScore",
        ascending=False
    )

    return trial_results.head(top_k).reset_index(drop=True)


# 10. Display results
def display_results(results):

    if results.empty:
        print("\nNo trials matched your search criteria.")
        return

    print("\nRETRIEVED CLINICAL TRIALS")
    print("-" * 50)

    for i, row in results.iterrows():

        print(f"\nResult {i + 1}")
        print("-" * 35)

        print("NCT ID:", row["NCTId"])
        print("Title:", row["Title"])
        print("Condition:", row["Conditions"])
        print("Searched Condition:", row["ConditionSearched"])
        print("Status:", row["Status"])
        print("Study Type:", row["StudyType"])
        print("Phase:", row["Phases"])

        print(
            "Age Range:",
            row["MinimumAge"],
            "-",
            row["MaximumAge"]
        )

        print("Sex:", row["Sex"])
        print("Best Matching Section:", row["Section"])

        print(
            "Similarity Score:",
            round(row["SimilarityScore"], 4)
        )

        print("Matching Chunks:", row["MatchedChunkCount"])
        print("Matching Text:", row["Chunk"][:500])
        print("Source:", row["Source"])


# 11. Interactive testing
if __name__ == "__main__":

    condition_options = {
        "1": "Type 2 Diabetes",
        "2": "Breast Cancer",
        "3": "Cardiovascular Diseases",
        "4": "Asthma",
        "5": "Alzheimer Disease",
        "6": "Chronic Kidney Disease",
        "7": "All"
    }

    print("\nAvailable conditions:")

    for key, value in condition_options.items():
        print(f"{key}. {value}")

    choice = input(
        "\nSelect a condition (1-7): "
    ).strip()

    selected_condition = condition_options.get(
        choice,
        "All"
    )

    # Exact status choices
    status_options = {
        "1": "All",
        "2": "RECRUITING",
        "3": "NOT_YET_RECRUITING",
        "4": "ACTIVE_NOT_RECRUITING",
        "5": "COMPLETED",
        "6": "TERMINATED",
        "7": "WITHDRAWN",
        "8": "SUSPENDED",
        "9": "ENROLLING_BY_INVITATION",
        "10": "UNKNOWN"
    }

    print("\nAvailable recruitment statuses:")

    for key, value in status_options.items():
        print(f"{key}. {value}")

    status_choice = input(
        "\nSelect recruitment status (1-10): "
    ).strip()

    selected_status = status_options.get(
        status_choice,
        "All"
    )

    query = input(
        "\nEnter your clinical trial search query: "
    ).strip()

    print("\nSearching for relevant trials...")

    results = retrieve_trials(
        query=query,
        top_k=5,
        condition=selected_condition,
        status=selected_status
    )

    display_results(results)

    print("\n" + "=" * 50)
