
import pandas as pd
import json
import re

print("=" * 45)
print("       TRIALLENS DOCUMENT CHUNKING")
print("=" * 45)

# 1. Load preprocessed clinical trial data
df = pd.read_csv("cleaned_clinical_trials.csv")

# 2. Chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# 3. Split text into chunks while preserving word boundaries
def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    if not isinstance(text, str) or not text.strip():
        return []

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):

        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


# 4. Define sections to be indexed separately
sections = {
    "Title": "Title",
    "Summary": "Summary",
    "Conditions": "Conditions",
    "Status": "Status",
    "Eligibility": "EligibilityCriteria"
}

# 5. Metadata to preserve for every chunk
metadata_columns = [
    "NCTId",
    "Title",
    "ConditionSearched",
    "Conditions",
    "Status",
    "StudyType",
    "Phases",
    "MinimumAge",
    "MaximumAge",
    "Sex",
    "StudyFirstPostDate",
    "LastUpdatePostDate",
    "DateCollected",
    "Source"
]

# 6. Generate chunks
all_chunks = []
chunk_id = 1

for _, row in df.iterrows():

    for section_name, column_name in sections.items():

        if column_name not in df.columns:
            continue

        text = row[column_name]

        if pd.isna(text) or str(text).strip() in ["", "Not available"]:
            continue

        text_chunks = split_into_chunks(str(text))

        for chunk_text in text_chunks:

            chunk_data = {
                "chunk_id": chunk_id,
                "Section": section_name,
                "Chunk": chunk_text
            }

            # Preserve trial metadata
            for column in metadata_columns:
                if column in df.columns:
                    chunk_data[column] = row[column]

            all_chunks.append(chunk_data)
            chunk_id += 1


# 7. Convert chunks into a DataFrame
chunks_df = pd.DataFrame(all_chunks)

# 8. Save CSV
chunks_df.to_csv("clinical_trial_chunks.csv", index=False)

# 9. Save JSON
chunks_df.to_json(
    "clinical_trial_chunks.json",
    orient="records",
    indent=4
)

# 10. Display results
print("\nCHUNKING COMPLETED")
print("-" * 45)

print("Total clinical trials:", len(df))
print("Total chunks generated:", len(chunks_df))
print("Chunk size:", CHUNK_SIZE, "words")
print("Chunk overlap:", CHUNK_OVERLAP, "words")

print("\nChunks by section:")
print(chunks_df["Section"].value_counts())

print("\nChunks by searched condition:")
print(chunks_df.groupby("ConditionSearched")["chunk_id"].count())

print("\nSaved files:")
print("1. clinical_trial_chunks.csv")
print("2. clinical_trial_chunks.json")

print("\nSample chunk:")
if not chunks_df.empty:
    print(chunks_df.iloc[0]["Chunk"])

print("\n" + "=" * 45)
