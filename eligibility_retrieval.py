
import os
import json
import pandas as pd

from dotenv import load_dotenv
from groq import Groq


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "Groq API key not found in .env file"
    )

client = Groq(api_key=api_key)

CHUNKS_FILE = "clinical_trial_chunks.csv"

if not os.path.exists(CHUNKS_FILE):
    raise FileNotFoundError(
        f"{CHUNKS_FILE} not found. "
        "Please run document_chunking.py first."
    )

df = pd.read_csv(CHUNKS_FILE)

required_columns = {"NCTId", "Section", "Chunk"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        "Clinical trial chunks file is missing "
        "required columns."
    )


# ==========================================
# 2. RETRIEVE ELIGIBILITY CRITERIA
# ==========================================

def get_eligibility_criteria(nct_id):

    nct_id = str(nct_id).strip().upper()

    if not nct_id.startswith("NCT"):
        return None

    trial_chunks = df[
        (df["NCTId"].astype(str).str.upper() == nct_id) &
        (df["Section"].astype(str).str.strip().str.lower()
         == "eligibility")
    ]

    if trial_chunks.empty:
        return None

    criteria = "\n".join(
        trial_chunks["Chunk"]
        .dropna()
        .astype(str)
        .tolist()
    )

    if not criteria.strip():
        return None

    return criteria


# ==========================================
# 3. ELIGIBILITY QUESTION ANSWERING
# ==========================================

def generate_eligibility_answer(nct_id, question):

    nct_id = str(nct_id).strip().upper()

    criteria = get_eligibility_criteria(nct_id)

    if criteria is None:
        return (
            f"No eligibility information found for {nct_id} "
            "in the collected dataset."
        )

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0.2,
            max_completion_tokens=2500,
            messages=[
                {
                    "role": "system",
                    "content": """
You are TrialLens, an educational clinical trial
information assistant.

Answer using ONLY the supplied eligibility criteria.

Rules:

1. Separate inclusion and exclusion criteria.
2. Do not invent missing information.
3. If information is unavailable, say so.
4. Use simple and clear language.
5. Include the NCT ID.
6. Do not make medical eligibility decisions.
7. Distinguish published criteria from interpretation.
8. Do not claim that a trial is currently recruiting
   based only on old collected data.
"""
                },
                {
                    "role": "user",
                    "content": f"""
TRIAL ID:
{nct_id}

PUBLISHED ELIGIBILITY CRITERIA:
{criteria}

QUESTION:
{question}

Answer using only the supplied criteria.
"""
                }
            ]
        )

        answer = response.choices[0].message.content

        if not answer:
            return "The language model returned an empty answer."

        return answer

    except Exception as e:

        return (
            "Unable to generate the eligibility answer. "
            f"Error: {str(e)}"
        )


# ==========================================
# 4. PATIENT PROFILE COMPARISON
# ==========================================

def compare_patient_to_criteria(nct_id, patient):

    nct_id = str(nct_id).strip().upper()

    criteria = get_eligibility_criteria(nct_id)

    if criteria is None:
        return None

    patient_text = "\n".join(
        f"- {key}: "
        f"{value if value not in ('', None) else 'Not provided'}"
        for key, value in patient.items()
    )

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0,
            max_completion_tokens=3000,
            messages=[
                {
                    "role": "system",
                    "content": """
You are TrialLens, an educational clinical trial
information assistant.

Compare the supplied synthetic patient profile
with the published eligibility criteria.

IMPORTANT RULES:

1. Use only the supplied trial criteria and
   patient information.
2. Evaluate each criterion separately.
3. Never invent criteria or patient information.
4. Never assume missing patient information.
5. Do not make a clinical eligibility decision.
6. Do not infer diagnoses from measurements.
7. Missing information must be marked cannot_assess.
8. Do not treat missing values as negative answers.
9. Include source criteria only.
10. Evaluate both inclusion and exclusion criteria
    when available.

Return ONLY a valid JSON object.

Required structure:

{
  "criteria": [
    {
      "type": "inclusion",
      "criterion": "Short criterion from source",
      "patient_info": "Relevant patient information",
      "result": "no_apparent_conflict",
      "reason": "Short explanation"
    }
  ]
}

Allowed type values:
- inclusion
- exclusion

Allowed result values:
- no_apparent_conflict
- potential_conflict
- cannot_assess

For inclusion criteria:

- no_apparent_conflict:
  Available information supports the criterion.

- potential_conflict:
  Available information suggests the criterion
  may not be met.

- cannot_assess:
  Required patient information is missing or unclear.

For exclusion criteria:

- no_apparent_conflict:
  Available information does not indicate
  that the exclusion applies.

- potential_conflict:
  Available information suggests the exclusion
  may apply.

- cannot_assess:
  Required patient information is missing or unclear.

Do not include markdown code fences.
Do not include text outside the JSON object.
"""
                },
                {
                    "role": "user",
                    "content": f"""
TRIAL ID:
{nct_id}

PUBLISHED ELIGIBILITY CRITERIA:
{criteria}

PATIENT PROFILE:
{patient_text}

Compare the profile with the published criteria.
Return the required JSON object.
"""
                }
            ]
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError(
                "Groq returned an empty response."
            )

        content = content.strip()

        # Remove markdown fences if present.

        if content.startswith("```"):

            content = content.replace(
                "```json", "", 1
            )

            content = content.replace(
                "```", ""
            ).strip()

        data = json.loads(content)

        if not isinstance(data, dict):
            raise ValueError(
                "The response is not a valid JSON object."
            )

        rows = data.get("criteria", [])

        if not isinstance(rows, list):
            raise ValueError(
                "The criteria field must be a list."
            )

        valid_types = {
            "inclusion",
            "exclusion"
        }

        valid_results = {
            "no_apparent_conflict",
            "potential_conflict",
            "cannot_assess"
        }

        normalized_source = " ".join(
            criteria.split()
        ).casefold()

        verified_rows = []

        for row in rows:

            if not isinstance(row, dict):
                continue

            criterion_type = row.get("type")

            result = row.get("result")

            criterion = str(
                row.get("criterion", "")
            ).strip()

            if (
                criterion_type not in valid_types
                or result not in valid_results
                or not criterion
            ):
                continue

            normalized_criterion = " ".join(
                criterion.split()
            ).casefold()

            reason = str(
                row.get(
                    "reason",
                    "No explanation provided."
                )
            )

            # Conservative source verification.
            # Paraphrased criteria may not pass this check.

            if normalized_criterion not in normalized_source:

                result = "cannot_assess"

                reason = (
                    "The criterion could not be verified "
                    "as an exact phrase in the retrieved "
                    "source text."
                )

            verified_rows.append({

                "type": criterion_type,

                "criterion": criterion,

                "patient_info": str(
                    row.get(
                        "patient_info",
                        "Not provided"
                    )
                ),

                "result": result,

                "reason": reason

            })

        # ==================================
        # 5. CALCULATE SUMMARY
        # ==================================

        results = [
            row["result"]
            for row in verified_rows
        ]

        if not results:

            overall = (
                "No criteria could be compared. "
                "Please review the source criteria."
            )

        elif "potential_conflict" in results:

            overall = (
                "Potential conflicts were identified "
                "in the supplied information. "
                "Consult the clinical trial team."
            )

        elif "cannot_assess" in results:

            overall = (
                "Some criteria could not be assessed "
                "because information is missing or unclear."
            )

        else:

            overall = (
                "No apparent conflicts were identified "
                "in the information provided. "
                "This does not confirm clinical eligibility."
            )

        return {

            "rows": verified_rows,

            "overall": overall

        }

    except Exception as e:

        return {

            "rows": [],

            "overall": (
                f"Comparison failed: {str(e)}"
            )

        }


# ==========================================
# 6. INDEPENDENT TESTING
# ==========================================

if __name__ == "__main__":

    print("Welcome to TrialLens!")

    while True:

        nct_id = input(
            "\nEnter Trial ID (or type exit): "
        ).strip()

        if nct_id.lower() == "exit":

            print(
                "Thank you for using TrialLens!"
            )

            break

        question = input(
            "Ask about eligibility criteria: "
        ).strip()

        if not question:

            print(
                "Please enter a question."
            )

            continue

        answer = generate_eligibility_answer(
            nct_id,
            question
        )

        print("\nTrialLens Answer:\n")

        print(answer)
