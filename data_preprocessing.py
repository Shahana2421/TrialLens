
import pandas as pd
import re

print("=" * 45)
print("       TRIALLENS DATA PREPROCESSING")
print("=" * 45)

# 1. Load the collected clinical trial dataset
df = pd.read_csv("clinical_trials.csv")

print("\nOriginal dataset shape:", df.shape)

# 2. Remove duplicate trials
df = df.drop_duplicates(subset=["NCTId"], keep="first")

# 3. Fill missing values without changing the original meaning
text_columns = [
    "Title",
    "Summary",
    "Conditions",
    "ConditionSearched",
    "Status",
    "StudyType",
    "Phases",
    "MinimumAge",
    "MaximumAge",
    "Sex",
    "StudyFirstPostDate",
    "LastUpdatePostDate",
    "EligibilityCriteria",
    "DateCollected",
    "Source"
]

for column in text_columns:
    if column in df.columns:
        df[column] = df[column].fillna("Not available").astype(str)

# 4. Clean text carefully
def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Preserve punctuation and symbols important to eligibility criteria
    text = re.sub(r"[^\w\s.,:;()/%+\-<>='\"]", " ", text)

    # Remove unnecessary extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


columns_to_clean = [
    "Title",
    "Summary",
    "Conditions",
    "ConditionSearched",
    "EligibilityCriteria"
]

for column in columns_to_clean:
    if column in df.columns:
        df[column] = df[column].apply(clean_text)

# 5. Create a combined document for each clinical trial
def create_document(row):
    return f"""
NCT ID: {row['NCTId']}

Title: {row['Title']}

Summary: {row['Summary']}

Conditions: {row['Conditions']}

Searched Condition: {row['ConditionSearched']}

Recruitment Status: {row['Status']}

Study Type: {row['StudyType']}

Study Phase: {row['Phases']}

Minimum Age: {row['MinimumAge']}

Maximum Age: {row['MaximumAge']}

Sex: {row['Sex']}

Eligibility Criteria:
{row['EligibilityCriteria']}
""".strip()


df["Document"] = df.apply(create_document, axis=1)

# 6. Save the cleaned dataset
df.to_csv("cleaned_clinical_trials.csv", index=False)

print("\nPREPROCESSING COMPLETED")
print("-" * 45)
print("Total unique trials:", len(df))
print("Total columns:", len(df.columns))
print("Cleaned dataset saved as: cleaned_clinical_trials.csv")

print("\nTrials by searched condition:")
print(df["ConditionSearched"].value_counts())

print("\nSample document:")
print(df["Document"].iloc[0][:1000])

print("\n" + "=" * 45)
