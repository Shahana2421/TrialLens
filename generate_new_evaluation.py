
import pandas as pd
from rag_pipeline import retrieve_trials

# Six test queries covering all six conditions
TEST_QUERIES = [
    {
        "query_id": "Q1",
        "condition": "Type 2 Diabetes",
        "query": "Type 2 diabetes insulin therapy and blood glucose control"
    },
    {
        "query_id": "Q2",
        "condition": "Breast Cancer",
        "query": "Breast cancer chemotherapy and treatment"
    },
    {
        "query_id": "Q3",
        "condition": "Cardiovascular Diseases",
        "query": "Heart failure cardiovascular treatment"
    },
    {
        "query_id": "Q4",
        "condition": "Asthma",
        "query": "Asthma inhaled corticosteroid treatment"
    },
    {
        "query_id": "Q5",
        "condition": "Alzheimer Disease",
        "query": "Alzheimer disease cognitive decline treatment"
    },
    {
        "query_id": "Q6",
        "condition": "Chronic Kidney Disease",
        "query": "Chronic kidney disease treatment and progression"
    }
]

TOP_K = 10
results = []

print("\nGenerating new retrieval evaluation candidates...\n")

for test in TEST_QUERIES:

    print(f"\nProcessing {test['query_id']}: {test['condition']}")

    trials = retrieve_trials(
        query=test["query"],
        top_k=TOP_K,
        condition=test["condition"],
        status="All"
    )

    if trials.empty:
        print("  No matching trials found.")
        continue

    # DataFrame must be iterated row by row
    for _, trial in trials.iterrows():

        results.append({
            "Query_ID": test["query_id"],
            "Condition": test["condition"],
            "Query": test["query"],
            "NCTId": trial.get("NCTId", ""),
            "Title": trial.get("Title", ""),
            "Conditions": trial.get("Conditions", ""),
            "Status": trial.get("Status", ""),
            "Section": trial.get("Section", ""),
            "SimilarityScore": trial.get("SimilarityScore", ""),
            "Relevant": "",
            "JudgementNotes": ""
        })

        print(
            f"  {trial.get('NCTId', '')} | "
            f"{str(trial.get('Title', ''))[:65]} | "
            f"Score: {trial.get('SimilarityScore', 0):.4f}"
        )

# Save results
df = pd.DataFrame(results)

output_file = "retrieval_evaluation_dataset_v2.csv"

df.to_csv(output_file, index=False)

print("\n" + "=" * 50)
print("RETRIEVAL EVALUATION CANDIDATES GENERATED")
print("=" * 50)

print(f"Total candidates: {len(df)}")
print(f"Saved file: {output_file}")

if not df.empty:
    print("\nCandidates per query:")
    print(df.groupby("Query_ID").size())

print("\nNext step: Manually label the Relevant column.")
