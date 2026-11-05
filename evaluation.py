
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from eligibility_retrieval import compare_patient_to_criteria


# ==========================================
# 1. LOAD EVALUATION DATASET
# ==========================================

df = pd.read_csv("evaluation_dataset.csv")

results = []


# ==========================================
# 2. DEFINE EXPECTED RESULTS
# ==========================================

def get_expected_result(row):

    hba1c = pd.to_numeric(
        row["hba1c"], errors="coerce"
    )

    # Check known conflicts
    conflict = (
        row["diabetes_type"] == "Type 1"
        or row["two_daily_insulin"] == "No"
        or row["consent_signed"] == "No"
        or row["glibenclamide"] == "Yes"
        or row["kidney_liver"] == "Yes"
        or row["sepsis"] == "Yes"
        or row["pregnancy"] == "Yes"
        or (pd.notna(hba1c) and hba1c > 10)
    )

    if conflict:
        return "potential_conflict"

    # Check missing information
    required_fields = [
        "diabetes_type",
        "two_daily_insulin",
        "consent_signed",
        "glibenclamide",
        "kidney_liver",
        "sepsis",
        "pregnancy",
        "hba1c"
    ]

    if any(pd.isna(row[field]) for field in required_fields):
        return "cannot_assess"

    return "no_apparent_conflict"


# ==========================================
# 3. CONVERT AI OUTPUT TO A CLASS
# ==========================================

def get_predicted_result(output):

    if output is None:
        return "cannot_assess"

    rows = output.get("rows", [])

    predictions = [
        item["result"] for item in rows
    ]

    if "potential_conflict" in predictions:
        return "potential_conflict"

    if (
        "cannot_assess" in predictions
        or not predictions
    ):
        return "cannot_assess"

    return "no_apparent_conflict"


# ==========================================
# 4. TEST EACH PATIENT
# ==========================================

for _, row in df.iterrows():

    print("\nProcessing Case:", row["case_id"])

    patient = {
        "Diabetes type": row["diabetes_type"],
        "Two daily insulin doses": row["two_daily_insulin"],
        "Signed informed consent": row["consent_signed"],
        "Glibenclamide treatment": row["glibenclamide"],
        "Kidney or liver condition": row["kidney_liver"],
        "Sepsis": row["sepsis"],
        "Pregnancy": row["pregnancy"],
        "HbA1c": (
            row["hba1c"]
            if pd.notna(row["hba1c"])
            else "Not provided"
        )
    }

    expected = get_expected_result(row)

    output = compare_patient_to_criteria(
        "NCT00994266",
        patient
    )

    predicted = get_predicted_result(output)

    results.append({
        "case_id": row["case_id"],
        "expected": expected,
        "predicted": predicted
    })

    print("Expected:", expected)
    print("Predicted:", predicted)


# ==========================================
# 5. EVALUATION METRICS
# ==========================================

results_df = pd.DataFrame(results)

y_true = results_df["expected"]
y_pred = results_df["predicted"]

print("\n================================")
print("TRIALLENS EVALUATION REPORT")
print("================================")

print("\nAccuracy:",
      round(accuracy_score(y_true, y_pred), 3))

print("\nPrecision:",
      round(
          precision_score(
              y_true,
              y_pred,
              average="macro",
              zero_division=0
          ), 3
      ))

print("\nRecall:",
      round(
          recall_score(
              y_true,
              y_pred,
              average="macro",
              zero_division=0
          ), 3
      ))

print("\nF1-score:",
      round(
          f1_score(
              y_true,
              y_pred,
              average="macro",
              zero_division=0
          ), 3
      ))

print("\nDetailed Classification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# 6. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "evaluation_results.csv",
    index=False
)

print("\nEvaluation results saved successfully!")
