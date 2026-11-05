
import pandas as pd

df = pd.read_csv("cleaned_clinical_trials.csv")

keywords = [
    "insulin",
    "glucose control",
    "glycemic control",
    "blood glucose",
    "hba1c",
    "metformin"
]

mask = df["Document"].str.contains(
    "|".join(keywords),
    case=False,
    na=False
)

results = df[mask].copy()

print("\nINSULIN AND GLUCOSE-RELATED TRIALS")
print("=" * 65)

print("Total matching trials:", len(results))

for _, row in results.iterrows():

    print("\nNCT ID:", row["NCTId"])
    print("Title:", row["Title"])
    print("Conditions:", row["Conditions"])
    print("Status:", row["Status"])
    print("Study Type:", row["StudyType"])

    print("-" * 65)

results.to_csv(
    "insulin_related_trials.csv",
    index=False
)

print("\nSaved: insulin_related_trials.csv")
