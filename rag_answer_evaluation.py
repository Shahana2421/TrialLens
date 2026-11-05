
from rag_pipeline import generate_answer, retrieve_trials
import pandas as pd

# Six evaluation test cases
TEST_CASES = [
    {
        "Query_ID": "RAG_Q1",
        "Condition": "Type 2 Diabetes",
        "Question": "Find clinical trials related to insulin therapy and blood glucose control in Type 2 Diabetes."
    },
    {
        "Query_ID": "RAG_Q2",
        "Condition": "Breast Cancer",
        "Question": "Find breast cancer clinical trials related to chemotherapy and treatment."
    },
    {
        "Query_ID": "RAG_Q3",
        "Condition": "Cardiovascular Diseases",
        "Question": "Find clinical trials related to heart failure and cardiovascular treatment."
    },
    {
        "Query_ID": "RAG_Q4",
        "Condition": "Asthma",
        "Question": "Find asthma clinical trials related to inhaled corticosteroid treatment."
    },
    {
        "Query_ID": "RAG_Q5",
        "Condition": "Alzheimer Disease",
        "Question": "Find Alzheimer's disease clinical trials related to cognitive decline and treatment."
    },
    {
        "Query_ID": "RAG_Q6",
        "Condition": "Chronic Kidney Disease",
        "Question": "Find clinical trials related to chronic kidney disease treatment and progression."
    }
]

TOP_K = 5
STATUS = "All"

all_results = []

print("\n" + "=" * 60)
print("TRIALLENS EVIDENCE-BASED RAG EVALUATION")
print("=" * 60)

for test in TEST_CASES:

    query_id = test["Query_ID"]
    condition = test["Condition"]
    question = test["Question"]

    print(f"\nProcessing {query_id}: {condition}")

    # Retrieve trials and their actual text chunks
    retrieved_trials = retrieve_trials(
        query=question,
        top_k=TOP_K,
        condition=condition,
        status=STATUS
    )

    if retrieved_trials.empty:
        print("No trials retrieved. Skipping.")
        continue

    # Generate the answer using the existing RAG pipeline
    answer = generate_answer(
        query=question,
        top_k=TOP_K,
        condition=condition,
        status=STATUS
    )

    # Save each retrieved trial as a separate row
    for _, trial in retrieved_trials.iterrows():

        all_results.append({
            "Query_ID": query_id,
            "Condition": condition,
            "Question": question,
            "NCTId": trial.get("NCTId", ""),
            "Title": trial.get("Title", ""),
            "Actual_Conditions": trial.get("Conditions", ""),
            "Status": trial.get("Status", ""),
            "Section": trial.get("Section", ""),
            "Retrieved_Evidence": trial.get("Chunk", ""),
            "SimilarityScore": trial.get("SimilarityScore", ""),
            "Source": trial.get("Source", ""),
            "Generated_Answer": answer
        })

    print("Retrieved trials:", len(retrieved_trials))
    print("Evidence collected successfully.")

# Create evaluation dataset
df = pd.DataFrame(all_results)

output_file = "rag_evidence_evaluation_dataset.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("EVIDENCE-BASED DATASET GENERATED")
print("=" * 60)

print("Total evidence records:", len(df))
print("Output file:", output_file)

if not df.empty:
    print("\nRecords per query:")
    print(df.groupby("Query_ID").size())

print("\nEvaluation dataset generation completed!")
