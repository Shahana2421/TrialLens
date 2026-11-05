
import os
import re
import time
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# TRIALLENS - RESUMABLE CLAIM-LEVEL FAITHFULNESS EVALUATION
# ============================================================

print("=" * 65)
print("       TRIALLENS CLAIM-LEVEL EVALUATION")
print("=" * 65)


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

INPUT_FILE = "rag_evidence_evaluation_dataset.csv"

CLAIM_OUTPUT_FILE = "claim_verification_results.csv"
SUMMARY_OUTPUT_FILE = "claim_verification_summary.csv"
RAW_OUTPUT_FILE = "claim_verification_raw_responses.txt"

GROQ_MODEL = "openai/gpt-oss-20b"

MAX_RETRIES = 1
MAX_COMPLETION_TOKENS = 2500

SUMMARY_COLUMNS = [
    "Query_ID",
    "Total_Claims",
    "Supported",
    "Contradicted",
    "Not_Established",
    "Supported_Percentage",
    "Evaluation_Status",
    "NCT_Check",
    "Evaluation_Error"
]

CLAIM_COLUMNS = [
    "Query_ID",
    "Question",
    "Claim_Number",
    "Claim",
    "Classification",
    "Reason",
    "NCT_Check"
]


# ------------------------------------------------------------
# 2. LOAD GROQ API
# ------------------------------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )

client = Groq(api_key=api_key)


# ------------------------------------------------------------
# 3. LOAD INPUT DATASET
# ------------------------------------------------------------

if not Path(INPUT_FILE).exists():
    raise FileNotFoundError(
        f"{INPUT_FILE} not found. Check your project folder."
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "Query_ID",
    "Question",
    "NCTId",
    "Title",
    "Actual_Conditions",
    "Status",
    "Section",
    "Retrieved_Evidence",
    "Generated_Answer"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nDataset loaded successfully.")
print("Total evidence records:", len(df))


# ------------------------------------------------------------
# 4. LOAD PREVIOUS RESULTS
# ------------------------------------------------------------

# Preserve previously completed claims.

if Path(CLAIM_OUTPUT_FILE).exists():
    claims_df = pd.read_csv(CLAIM_OUTPUT_FILE)

    if "Query_ID" not in claims_df.columns:
        claims_df = pd.DataFrame(columns=CLAIM_COLUMNS)

    print(
        "Previous claim results loaded:",
        len(claims_df)
    )

else:
    claims_df = pd.DataFrame(columns=CLAIM_COLUMNS)


# Preserve previous query summaries.

if Path(SUMMARY_OUTPUT_FILE).exists():
    summary_df = pd.read_csv(SUMMARY_OUTPUT_FILE)

    if "Query_ID" not in summary_df.columns:
        summary_df = pd.DataFrame(columns=SUMMARY_COLUMNS)

    print(
        "Previous query summaries loaded:",
        len(summary_df)
    )

else:
    summary_df = pd.DataFrame(columns=SUMMARY_COLUMNS)


# Ensure all required columns exist.

for column in CLAIM_COLUMNS:
    if column not in claims_df.columns:
        claims_df[column] = ""

for column in SUMMARY_COLUMNS:
    if column not in summary_df.columns:
        summary_df[column] = ""

claims_df = claims_df[CLAIM_COLUMNS]
summary_df = summary_df[SUMMARY_COLUMNS]


# ------------------------------------------------------------
# 5. TEXT NORMALIZATION
# ------------------------------------------------------------

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ------------------------------------------------------------
# 6. NCT ID VALIDATION
# ------------------------------------------------------------

def extract_nct_ids(text):

    if pd.isna(text):
        return []

    return sorted(
        set(
            re.findall(
                r"\bNCT\d{8}\b",
                str(text),
                flags=re.IGNORECASE
            )
        )
    )


def validate_nct_ids(answer, evidence_ids):

    generated_ids = {
        value.upper()
        for value in extract_nct_ids(answer)
    }

    available_ids = {
        str(value).upper()
        for value in evidence_ids
        if pd.notna(value)
    }

    unsupported_ids = generated_ids - available_ids
    missing_ids = available_ids - generated_ids

    return {
        "Generated_NCT_IDs": ", ".join(
            sorted(generated_ids)
        ),
        "Unsupported_NCT_IDs": ", ".join(
            sorted(unsupported_ids)
        ),
        "Missing_NCT_IDs": ", ".join(
            sorted(missing_ids)
        ),
        "NCT_Check": (
            "PASS" if not unsupported_ids else "FAIL"
        )
    }


# ------------------------------------------------------------
# 7. PREPARE EVIDENCE CONTEXT
# ------------------------------------------------------------

def prepare_evidence_context(query_rows):

    evidence_parts = []

    unique_trials = query_rows.drop_duplicates(
        subset="NCTId"
    )

    for _, row in unique_trials.iterrows():

        evidence = f"""
NCT ID: {row['NCTId']}
Title: {row['Title']}
Actual Conditions: {row['Actual_Conditions']}
Recruitment Status: {row['Status']}
Relevant Section: {row['Section']}

Retrieved Evidence:
{row['Retrieved_Evidence']}
"""

        evidence_parts.append(evidence)

    return "\n".join(evidence_parts)


# ------------------------------------------------------------
# 8. GROQ REQUEST
# ------------------------------------------------------------

def call_groq(prompt):

    try:

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            reasoning_effort="low",
            include_reasoning=False,
            stream=False
        )

        message = response.choices[0].message

        content = (
            message.content or ""
        ).strip()

        if not content:
            finish_reason = (
                response.choices[0].finish_reason
            )

            raise ValueError(
                "Groq returned empty content. "
                f"Finish reason: {finish_reason}"
            )

        return content

    except Exception as error:

        error_text = str(error)

        # Do not waste retries on quota errors.

        if "429" in error_text or "rate_limit" in error_text.lower():
            raise RuntimeError(
                "Groq quota/rate limit reached. "
                "Wait until the quota resets before continuing. "
                f"Details: {error_text}"
            )

        raise RuntimeError(
            f"Groq request failed: {error_text}"
        )


# ------------------------------------------------------------
# 9. BUILD EVALUATION PROMPT
# ------------------------------------------------------------

def build_evaluation_prompt(
    question,
    answer,
    evidence
):

    return f"""
You are a strict evidence verification assistant
for the TrialLens clinical trial discovery project.

Verify factual claims in the generated answer using
ONLY the supplied clinical trial evidence.

QUESTION:
{question}

GENERATED ANSWER:
{answer}

RETRIEVED EVIDENCE:
{evidence}

INSTRUCTIONS:

1. Extract individual factual claims from the answer.
2. Include claims about trial purpose, conditions,
   treatments, recruitment status, study type, phase,
   age, sex, sample size, duration and outcomes.
3. Exclude headings and general disclaimers.
4. Verify each claim independently using only the evidence.
5. Do not use outside medical knowledge.
6. Do not infer outcomes from a title alone.
7. Missing information means NOT ESTABLISHED,
   not CONTRADICTED.
8. If evidence conflicts, use NOT ESTABLISHED.
9. Do not invent claims or evidence.
10. Keep claims and reasons concise.

Use exactly these verdicts:

SUPPORTED
The evidence directly supports the claim.

CONTRADICTED
The evidence directly conflicts with the claim.

NOT ESTABLISHED
The evidence does not provide enough information.

OUTPUT FORMAT:

CLAIM: The study investigated asthma treatment.
VERDICT: SUPPORTED
REASON: The evidence describes the asthma study.

CLAIM: The treatment reduced symptoms by 50 percent.
VERDICT: NOT ESTABLISHED
REASON: The evidence does not report this result.

Return one block per claim.
Use exactly CLAIM, VERDICT and REASON labels.
Do not use JSON or Markdown tables.
Do not add an introduction or conclusion.
"""


# ------------------------------------------------------------
# 10. PARSE CLAIM RESULTS
# ------------------------------------------------------------

def parse_claim_results(response_text):

    results = []

    blocks = re.split(
        r"(?im)(?=^\s*(?:\d+[.)]\s*)?CLAIM\s*:)",
        response_text
    )

    for block in blocks:

        block = block.strip()

        if not re.search(
            r"(?im)^\s*(?:\d+[.)]\s*)?CLAIM\s*:",
            block
        ):
            continue

        claim_match = re.search(
            r"(?im)^\s*(?:\d+[.)]\s*)?CLAIM\s*:\s*(.+)$",
            block
        )

        verdict_match = re.search(
            r"(?im)^\s*VERDICT\s*:\s*"
            r"(SUPPORTED|CONTRADICTED|NOT ESTABLISHED)\s*$",
            block
        )

        reason_match = re.search(
            r"(?ims)^\s*REASON\s*:\s*(.+?)(?=^\s*(?:CLAIM|VERDICT)\s*:|\Z)",
            block
        )

        if not claim_match or not verdict_match:
            continue

        claim = claim_match.group(1).strip()
        verdict = verdict_match.group(1).strip().upper()

        reason = (
            reason_match.group(1).strip()
            if reason_match
            else "No explanation provided."
        )

        reason = re.sub(r"\s+", " ", reason)

        results.append({
            "Claim": claim,
            "Classification": verdict,
            "Reason": reason
        })

    if not results:
        raise ValueError(
            "No valid CLAIM / VERDICT / REASON blocks found."
        )

    return results


# ------------------------------------------------------------
# 11. SAVE PROGRESS
# ------------------------------------------------------------

def save_progress():

    claims_df.to_csv(
        CLAIM_OUTPUT_FILE,
        index=False
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT_FILE,
        index=False
    )


def save_raw_response(query_id, response_text):

    with open(
        RAW_OUTPUT_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            f"\n{'=' * 65}\n"
            f"QUERY: {query_id}\n"
            f"{'=' * 65}\n"
            f"{response_text}\n"
        )


# ------------------------------------------------------------
# 12. RESUMABLE QUERY EVALUATION
# ------------------------------------------------------------

unique_queries = df["Query_ID"].dropna().unique()

print("\nStarting claim-level evaluation...")
print("Resume mode enabled.")


for query_id in unique_queries:

    print("\n" + "-" * 65)
    print("Processing:", query_id)

    # Check whether this query was already completed.

    previous = summary_df[
        summary_df["Query_ID"] == query_id
    ]

    if (
        not previous.empty
        and previous.iloc[-1]["Evaluation_Status"] == "COMPLETED"
    ):

        print(
            "SKIPPED: This query was already completed."
        )

        continue

    query_rows = df[
        df["Query_ID"] == query_id
    ].copy()

    first_row = query_rows.iloc[0]

    question = str(first_row["Question"])
    answer = str(first_row["Generated_Answer"])

    evidence_context = prepare_evidence_context(
        query_rows
    )

    evidence_ids = (
        query_rows["NCTId"]
        .dropna()
        .unique()
    )

    # NCT validation.

    nct_result = validate_nct_ids(
        answer,
        evidence_ids
    )

    print(
        "NCT consistency:",
        nct_result["NCT_Check"]
    )

    prompt = build_evaluation_prompt(
        question,
        answer,
        evidence_context
    )

    try:

        response_text = call_groq(prompt)

        claims = parse_claim_results(
            response_text
        )

        # Save raw response immediately.

        save_raw_response(
            query_id,
            response_text
        )

    except Exception as error:

        print("EVALUATION FAILED:", str(error))

        failed_summary = {
            "Query_ID": query_id,
            "Total_Claims": 0,
            "Supported": 0,
            "Contradicted": 0,
            "Not_Established": 0,
            "Supported_Percentage": 0,
            "Evaluation_Status": "FAILED",
            "NCT_Check": nct_result["NCT_Check"],
            "Evaluation_Error": str(error)
        }

        # Replace previous failed summary for this query.

        summary_df = summary_df[
            summary_df["Query_ID"] != query_id
        ]

        summary_df = pd.concat(
            [
                summary_df,
                pd.DataFrame([failed_summary])
            ],
            ignore_index=True
        )

        save_progress()

        print(
            "Progress saved. The next run will retry this query."
        )

        # Stop here if quota is exhausted.
        # Do not waste further requests.

        if (
            "429" in str(error)
            or "quota" in str(error).lower()
            or "rate limit" in str(error).lower()
        ):

            print(
                "\nStopping evaluation because the Groq quota "
                "has been reached."
            )

            break

        continue

    # --------------------------------------------------------
    # COUNT CLAIM VERDICTS
    # --------------------------------------------------------

    supported = 0
    contradicted = 0
    not_established = 0

    new_claim_rows = []

    for claim_number, item in enumerate(
        claims,
        start=1
    ):

        classification = item["Classification"]

        if classification == "SUPPORTED":
            supported += 1

        elif classification == "CONTRADICTED":
            contradicted += 1

        elif classification == "NOT ESTABLISHED":
            not_established += 1

        new_claim_rows.append({
            "Query_ID": query_id,
            "Question": question,
            "Claim_Number": claim_number,
            "Claim": item["Claim"],
            "Classification": classification,
            "Reason": item["Reason"],
            "NCT_Check": nct_result["NCT_Check"]
        })

    total_claims = len(claims)

    supported_rate = (
        supported / total_claims * 100
        if total_claims > 0
        else 0
    )

    # Remove old claim rows for this query before replacing.

    claims_df = claims_df[
        claims_df["Query_ID"] != query_id
    ]

    claims_df = pd.concat(
        [
            claims_df,
            pd.DataFrame(new_claim_rows)
        ],
        ignore_index=True
    )

    completed_summary = {
        "Query_ID": query_id,
        "Total_Claims": total_claims,
        "Supported": supported,
        "Contradicted": contradicted,
        "Not_Established": not_established,
        "Supported_Percentage": round(
            supported_rate,
            2
        ),
        "Evaluation_Status": "COMPLETED",
        "NCT_Check": nct_result["NCT_Check"],
        "Evaluation_Error": ""
    }

    # Replace previous summary for this query.

    summary_df = summary_df[
        summary_df["Query_ID"] != query_id
    ]

    summary_df = pd.concat(
        [
            summary_df,
            pd.DataFrame([completed_summary])
        ],
        ignore_index=True
    )

    # Save immediately after each successful query.

    save_progress()

    print("Claims extracted and verified:", total_claims)

    print(
        f"Supported: {supported} | "
        f"Contradicted: {contradicted} | "
        f"Not established: {not_established}"
    )

    print("Progress saved successfully.")


# ------------------------------------------------------------
# 13. FINAL SUMMARY
# ------------------------------------------------------------

# Keep summaries in the same order as the input queries.

summary_df["_order"] = summary_df["Query_ID"].map(
    {
        query_id: position
        for position, query_id in enumerate(unique_queries)
    }
)

summary_df = (
    summary_df
    .sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)

save_progress()

total_claims = len(claims_df)

if total_claims > 0:

    supported = (
        claims_df["Classification"] == "SUPPORTED"
    ).sum()

    contradicted = (
        claims_df["Classification"] == "CONTRADICTED"
    ).sum()

    not_established = (
        claims_df["Classification"] == "NOT ESTABLISHED"
    ).sum()

    supported_rate = (
        supported / total_claims
    ) * 100

else:

    supported = 0
    contradicted = 0
    not_established = 0
    supported_rate = 0


completed_queries = (
    summary_df["Evaluation_Status"] == "COMPLETED"
).sum()

failed_queries = (
    summary_df["Evaluation_Status"] == "FAILED"
).sum()

print("\n" + "=" * 65)
print("TRIALLENS CLAIM-LEVEL EVALUATION SUMMARY")
print("=" * 65)

print("Total queries:", len(unique_queries))
print("Completed queries:", completed_queries)
print("Failed queries:", failed_queries)

print("Total claims evaluated:", total_claims)
print("Supported:", supported)
print("Contradicted:", contradicted)
print("Not established:", not_established)

print(
    "Supported claim rate:",
    round(supported_rate, 2),
    "%"
)

print("\nQuery-wise results:")
print(summary_df.to_string(index=False))

print("\nFiles saved:")
print(CLAIM_OUTPUT_FILE)
print(SUMMARY_OUTPUT_FILE)
print(RAW_OUTPUT_FILE)

print("\n" + "=" * 65)
print("EVALUATION FINISHED")
print("=" * 65)
