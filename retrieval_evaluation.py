
import pandas as pd
from rag_pipeline import retrieve_trials

# Search queries for evaluating all six conditions
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

review_rows = []

print("\nGenerating retrieval evaluation candidates...\n")

for test in TEST_QUERIES:

    print(f"Processing: {test['condition']}")

    results = retrieve_trials(
        query=test["query"],
        top_k=10,
        condition=test["condition"],
        status=None
    )

    if results.empty:
        print("No matching trials found.")
        continue

    for _, trial in results.iterrows():

        review_rows.append({
            "query_id": test["query_id"],
            "condition": test["condition"],
            "query": test["query"],
            "NCTId": trial["NCTId"],
            "Title": trial["Title"],
            "Conditions": trial["Conditions"],
            "Status": trial["Status"],
            "SimilarityScore": trial["SimilarityScore"],
            "Relevant": ""
        })

# Create review dataset
review_df = pd.DataFrame(review_rows)

if not review_df.empty:

    review_df.to_csv(
        "retrieval_evaluation_dataset.csv",
        index=False
    )

    print("\nEvaluation candidate sheet created successfully!")
    print("File: retrieval_evaluation_dataset.csv")
    print("Total candidates:", len(review_df))

    print("\nCandidates per condition:")
    print(review_df.groupby("condition").size())

else:
    print("\nNo candidates generated. Check your retrieval pipeline.")
