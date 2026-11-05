
import requests
import pandas as pd

from datetime import datetime, timezone


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

CONDITIONS = [
    "Type 2 Diabetes",
    "Breast Cancer",
    "Cardiovascular Diseases",
    "Asthma",
    "Alzheimer Disease",
    "Chronic Kidney Disease"
]

MAX_STUDIES_PER_CONDITION = 50

all_trials = []

collection_date = datetime.now(
    timezone.utc
).strftime("%Y-%m-%d %H:%M:%S UTC")


# ==========================================
# 2. EXTRACT TRIAL INFORMATION
# ==========================================

def extract_trial(study, searched_condition):

    protocol = study.get("protocolSection", {})

    identification = protocol.get(
        "identificationModule", {}
    )

    status_module = protocol.get(
        "statusModule", {}
    )

    description = protocol.get(
        "descriptionModule", {}
    )

    conditions_module = protocol.get(
        "conditionsModule", {}
    )

    eligibility = protocol.get(
        "eligibilityModule", {}
    )

    design = protocol.get(
        "designModule", {}
    )

    # Trial identification
    nct_id = identification.get("nctId", "")

    title = (
        identification.get("officialTitle")
        or identification.get("briefTitle", "")
    )

    # Study status and dates
    status = status_module.get(
        "overallStatus", "Unknown"
    )

    first_posted = status_module.get(
        "studyFirstPostDateStruct", {}
    ).get("date", "")

    last_updated = status_module.get(
        "studyLastUpdatePostDateStruct", {}
    ).get("date", "")

    # Clinical conditions
    conditions = conditions_module.get(
        "conditions", []
    )

    # Eligibility information
    eligibility_criteria = eligibility.get(
        "eligibilityCriteria", ""
    )

    min_age = eligibility.get("minimumAge", "")
    max_age = eligibility.get("maximumAge", "")

    sex = eligibility.get("sex", "")

    # Study design
    study_type = design.get("studyType", "")

    phases = design.get("phases", [])

    # Summary
    summary = description.get(
        "briefSummary", ""
    )

    return {

        "NCTId": nct_id,

        "Title": title,

        "Summary": summary,

        "Conditions": "; ".join(conditions),

        "ConditionSearched": searched_condition,

        "Status": status,

        "StudyType": study_type,

        "Phases": "; ".join(phases),

        "MinimumAge": min_age,

        "MaximumAge": max_age,

        "Sex": sex,

        "StudyFirstPostDate": first_posted,

        "LastUpdatePostDate": last_updated,

        "EligibilityCriteria": eligibility_criteria,

        "DateCollected": collection_date,

        "Source": (
            f"https://clinicaltrials.gov/study/{nct_id}"
            if nct_id else ""
        )
    }


# ==========================================
# 3. COLLECT TRIALS FOR EACH CONDITION
# ==========================================

def collect_trials(condition):

    print(f"\nCollecting trials for: {condition}")

    params = {
        "query.cond": condition,
        "pageSize": MAX_STUDIES_PER_CONDITION,
        "format": "json"
    }

    try:

        response = requests.get(
            BASE_URL,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        studies = data.get("studies", [])

        condition_trials = []

        for study in studies:

            trial = extract_trial(
                study,
                condition
            )

            if trial["NCTId"]:
                condition_trials.append(trial)

        print(
            f"Successfully collected "
            f"{len(condition_trials)} trials for {condition}"
        )

        return condition_trials

    except requests.exceptions.RequestException as e:

        print(f"API request failed for {condition}: {e}")

    except (ValueError, KeyError, TypeError) as e:

        print(f"Data processing failed for {condition}: {e}")

    return []


# ==========================================
# 4. COLLECT ALL CONDITIONS
# ==========================================

if __name__ == "__main__":

    print("===================================")
    print("       TRIALLENS DATA COLLECTION")
    print("===================================")

    for condition in CONDITIONS:

        trials = collect_trials(condition)

        all_trials.extend(trials)

    # Remove duplicate trial records
    df = pd.DataFrame(all_trials)

    if not df.empty:

        df.drop_duplicates(
            subset="NCTId",
            keep="first",
            inplace=True
        )

        df.reset_index(drop=True, inplace=True)

        # Save collected data
        df.to_csv(
            "clinical_trials.csv",
            index=False
        )

        print("\n===================================")
        print("DATA COLLECTION COMPLETED")
        print("===================================")

        print("Total unique trials:", len(df))

        print("\nTrials by searched condition:")

        print(
            df["ConditionSearched"].value_counts()
        )

        print("\nTrials by recruitment status:")

        print(
            df["Status"].value_counts()
        )

        print("\nDataset shape:", df.shape)

        print(
            "\nDataset saved as clinical_trials.csv"
        )

        print(
            "\nDataset collection date:",
            collection_date
        )

    else:

        print(
            "\nNo trial records were collected. "
            "Please check your internet connection or API response."
        )
