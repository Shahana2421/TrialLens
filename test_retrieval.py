
from rag_pipeline import retrieve_trials

query = "Type 2 diabetes insulin therapy and blood glucose control"

results = retrieve_trials(
    query=query,
    top_k=10,
    condition="Type 2 Diabetes",
    status="All"
)

print("\nTOP 10 INSULIN-RELATED TRIAL SEARCH RESULTS")
print("=" * 70)

if results.empty:

    print("No trials found.")

else:

    for i, row in results.iterrows():

        print(f"\nRank: {i + 1}")
        print("NCT ID:", row["NCTId"])
        print("Title:", row["Title"])
        print("Conditions:", row["Conditions"])
        print("Status:", row["Status"])
        print("Study Type:", row["StudyType"])
        print("Section:", row["Section"])

        print(
            "Similarity:",
            round(row["SimilarityScore"], 4)
        )

        print("Matching Text:", row["Chunk"][:350])

        print("-" * 70)
