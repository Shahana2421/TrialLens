
import pandas as pd

# Load the manually labelled evaluation dataset
INPUT_FILE = "retrieval_evaluation_dataset_v2_labeled.csv"

df = pd.read_csv(INPUT_FILE)

# Check required columns
required_columns = ["Query_ID", "Relevant"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

# Check that all candidates have been labelled
if df["Relevant"].isna().any():
    raise ValueError(
        "Some trials are not labelled. Fill all Relevant cells with 0 or 1."
    )

# Convert labels into integers
df["Relevant"] = pd.to_numeric(df["Relevant"], errors="raise")

if not df["Relevant"].isin([0, 1]).all():
    raise ValueError("Relevant column must contain only 0 or 1.")

# Calculate metrics for each query
results = []

for query_id, group in df.groupby("Query_ID", sort=True):

    # Preserve retrieval ranking from the original CSV
    group = group.reset_index(drop=True)

    top_5 = group.head(5)

    # Precision@5
    relevant_in_top5 = int(top_5["Relevant"].sum())
    precision_at_5 = relevant_in_top5 / 5

    # Recall@5: relevant trials in top 5 divided by
    # all relevant trials in the top-10 candidate set
    total_relevant = int(group["Relevant"].sum())

    if total_relevant > 0:
        recall_at_5 = relevant_in_top5 / total_relevant
    else:
        recall_at_5 = 0.0

    # MRR@5: reciprocal rank of the first relevant
    # trial within the top 5
    relevant_ranks = [
        rank
        for rank, label in enumerate(top_5["Relevant"], start=1)
        if label == 1
    ]

    if relevant_ranks:
        mrr_at_5 = 1 / relevant_ranks[0]
    else:
        mrr_at_5 = 0.0

    results.append({
        "Query_ID": query_id,
        "Total_Candidates": len(group),
        "Total_Relevant": total_relevant,
        "Relevant_in_Top5": relevant_in_top5,
        "Precision@5": precision_at_5,
        "Recall@5": recall_at_5,
        "MRR@5": mrr_at_5
    })

# Create results table
metrics_df = pd.DataFrame(results)

# Calculate macro averages across queries
average_row = {
    "Query_ID": "OVERALL AVERAGE",
    "Total_Candidates": metrics_df["Total_Candidates"].sum(),
    "Total_Relevant": metrics_df["Total_Relevant"].sum(),
    "Relevant_in_Top5": metrics_df["Relevant_in_Top5"].sum(),
    "Precision@5": metrics_df["Precision@5"].mean(),
    "Recall@5": metrics_df["Recall@5"].mean(),
    "MRR@5": metrics_df["MRR@5"].mean()
}

metrics_df = pd.concat(
    [metrics_df, pd.DataFrame([average_row])],
    ignore_index=True
)

# Display results
print("\n" + "=" * 65)
print("          TRIALLENS RETRIEVAL EVALUATION")
print("=" * 65)

print(metrics_df.to_string(
    index=False,
    formatters={
        "Precision@5": "{:.4f}".format,
        "Recall@5": "{:.4f}".format,
        "MRR@5": "{:.4f}".format
    }
))

# Save evaluation results
OUTPUT_FILE = "retrieval_metrics_results.csv"
metrics_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 65)
print("Evaluation completed!")
print(f"Results saved to: {OUTPUT_FILE}")
print("=" * 65)
