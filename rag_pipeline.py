
import os
import re
import pandas as pd
import faiss

from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# TRIALLENS - RETRIEVAL AUGMENTED GENERATION PIPELINE
# ============================================================

print("=" * 65)
print("              TRIALLENS RAG PIPELINE")
print("=" * 65)


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_MODEL = "openai/gpt-oss-20b"

INDEX_FILE = "clinical_trials.index"

CHUNKS_FILE = "clinical_trial_chunks.csv"

MIN_SIMILARITY_SCORE = 0.30
MAX_CHUNKS_PER_TRIAL = 3


# ------------------------------------------------------------
# 2. LOAD ENVIRONMENT VARIABLES
# ------------------------------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )

client = Groq(api_key=api_key)


# ------------------------------------------------------------
# 3. LOAD EMBEDDING MODEL
# ------------------------------------------------------------

print("\nLoading Sentence Transformer model...")

embedding_model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded successfully.")


# ------------------------------------------------------------
# 4. LOAD FAISS VECTOR DATABASE
# ------------------------------------------------------------

if not Path(INDEX_FILE).exists():
    raise FileNotFoundError(
        f"{INDEX_FILE} not found. Generate the vector database first."
    )

index = faiss.read_index(INDEX_FILE)

print("\nFAISS index loaded successfully.")

print("Total vectors:", index.ntotal)

print("Vector dimensions:", index.d)


# ------------------------------------------------------------
# 5. LOAD CHUNK DATASET
# ------------------------------------------------------------

if not Path(CHUNKS_FILE).exists():
    raise FileNotFoundError(
        f"{CHUNKS_FILE} not found."
    )

chunks_df = pd.read_csv(CHUNKS_FILE)

required_columns = [
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
    "Section",
    "Chunk",
    "Source"
]

missing_columns = [
    column for column in required_columns
    if column not in chunks_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in chunk dataset: {missing_columns}"
    )

if index.ntotal != len(chunks_df):

    raise ValueError(
        "FAISS index and chunk dataset have different sizes. "
        "Regenerate the vector database."
    )

print("Chunk dataset loaded successfully.")

print("Total chunks:", len(chunks_df))


# ------------------------------------------------------------
# 6. CONDITION KEYWORD MAPPING
# ------------------------------------------------------------

CONDITION_KEYWORDS = {

    "Type 2 Diabetes": [
        "type 2 diabetes",
        "type 2 diabetes mellitus",
        "diabetes mellitus type 2",
        "diabetes mellitus, type 2"
    ],

    "Breast Cancer": [
        "breast cancer",
        "breast carcinoma",
        "carcinoma of breast"
    ],

    "Cardiovascular Diseases": [
        "cardiovascular disease",
        "cardiovascular diseases",
        "heart failure",
        "coronary artery disease",
        "coronary heart disease",
        "myocardial infarction",
        "ischemic heart disease",
        "ischaemic heart disease",
        "cardiac disease",
        "cardiomyopathy",
        "hypertension"
    ],

    "Asthma": [
        "asthma"
    ],

    "Alzheimer Disease": [
        "alzheimer disease",
        "alzheimer's disease",
        "alzheimer dementia"
    ],

    "Chronic Kidney Disease": [
        "chronic kidney disease",
        "chronic kidney diseases",
        "chronic renal disease",
        "chronic renal failure"
    ]
}


# ------------------------------------------------------------
# 7. TEXT NORMALIZATION
# ------------------------------------------------------------

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ------------------------------------------------------------
# 8. CONDITION MATCHING
# ------------------------------------------------------------

def condition_matches(actual_conditions, selected_condition):

    if not selected_condition:
        return True

    actual_text = normalize_text(actual_conditions)

    keywords = CONDITION_KEYWORDS.get(
        selected_condition,
        [selected_condition.lower()]
    )

    for keyword in keywords:

        keyword = normalize_text(keyword)

        if selected_condition == "Type 2 Diabetes":

            if re.search(
                r"\b" + re.escape(keyword) + r"\b",
                actual_text
            ):
                return True

        else:

            if keyword in actual_text:
                return True

    return False


# ------------------------------------------------------------
# 9. RETRIEVE CLINICAL TRIALS
# ------------------------------------------------------------

def retrieve_trials(
    query,
    top_k=5,
    condition=None,
    status=None
):

    print("\nSearching clinical trial database...")

    # Generate query embedding.
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    # Search more chunks than the requested number of trials because
    # each trial may contribute multiple useful evidence sections.
    search_k = min(
        len(chunks_df),
        max(top_k * 30, 100)
    )

    similarities, indices = index.search(query_embedding, search_k)

    results = []

    eligibility_terms = [
        "eligibility", "eligible", "inclusion criteria",
        "exclusion criteria", "age requirement", "participant criteria"
    ]

    query_text = normalize_text(query)
    include_eligibility = any(
        term in query_text for term in eligibility_terms
    )

    for similarity, idx in zip(similarities[0], indices[0]):
        if idx < 0:
            continue

        row = chunks_df.iloc[int(idx)].copy()
        score = float(similarity)

        if score < MIN_SIMILARITY_SCORE:
            continue

        # Validate against the trial's actual conditions.
        if not condition_matches(row["Conditions"], condition):
            continue

        # Apply recruitment-status filter.
        if status and normalize_text(status) != "all":
            if normalize_text(row["Status"]) != normalize_text(status):
                continue

        section = str(row["Section"]).lower()

        # For general discovery queries, prioritize title and summary.
        # Eligibility chunks are included when the query asks about criteria.
        if not include_eligibility and section not in ["title", "summary"]:
            continue

        results.append({
            "NCTId": row["NCTId"],
            "Title": row["Title"],
            "ConditionSearched": row["ConditionSearched"],
            "Conditions": row["Conditions"],
            "Status": row["Status"],
            "StudyType": row["StudyType"],
            "Phases": row["Phases"],
            "MinimumAge": row["MinimumAge"],
            "MaximumAge": row["MaximumAge"],
            "Sex": row["Sex"],
            "Section": row["Section"],
            "Chunk": row["Chunk"],
            "SimilarityScore": score,
            "Source": row["Source"]
        })

    if not results:
        print("Clinical trials retrieved: 0")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    section_priority = {
        "title": 0,
        "summary": 1,
        "eligibility": 2,
        "conditions": 3,
        "status": 4
    }

    results_df["SectionPriority"] = (
        results_df["Section"].astype(str).str.lower()
        .map(section_priority).fillna(5)
    )

    # Rank evidence chunks by semantic similarity. Section priority is
    # used only to break ties, not to override a stronger similarity score.
    results_df = results_df.sort_values(
        by=["SimilarityScore", "SectionPriority", "NCTId"],
        ascending=[False, True, True],
        kind="mergesort"
    )

    # Select the top_k trials by their best matching chunk.
    trial_ranking = (
        results_df.groupby("NCTId", sort=False)["SimilarityScore"]
        .max().sort_values(ascending=False)
    )
    selected_nct_ids = trial_ranking.head(top_k).index.tolist()

    selected_chunks = results_df[
        results_df["NCTId"].isin(selected_nct_ids)
    ].copy()

    # Keep multiple chunks for each selected trial instead of discarding
    # all but one chunk. The maximum is controlled by the config above.
    selected_chunks["TrialRank"] = selected_chunks["NCTId"].map(
        {nct_id: rank for rank, nct_id in enumerate(selected_nct_ids)}
    )
    selected_chunks = selected_chunks.sort_values(
        by=["TrialRank", "SimilarityScore", "SectionPriority"],
        ascending=[True, False, True],
        kind="mergesort"
    )

    selected_chunks = (
        selected_chunks.groupby("NCTId", sort=False, group_keys=False)
        .head(MAX_CHUNKS_PER_TRIAL)
        .drop(columns=["SectionPriority", "TrialRank"])
        .reset_index(drop=True)
    )

    print("Clinical trials retrieved:", selected_chunks["NCTId"].nunique())
    print("Evidence chunks retrieved:", len(selected_chunks))

    return selected_chunks

# ------------------------------------------------------------
# 10. BUILD TRIAL-SPECIFIC EVIDENCE CONTEXT
# ------------------------------------------------------------

def prepare_trial_context(retrieved_trials):

    context_parts = []

    # retrieved_trials now contains multiple evidence chunks per NCT ID.
    for nct_id, trial_chunks in retrieved_trials.groupby("NCTId", sort=False):
        first = trial_chunks.iloc[0]

        evidence_parts = []
        for chunk_number, (_, chunk_row) in enumerate(
            trial_chunks.iterrows(), start=1
        ):
            evidence_parts.append(
                f"Evidence chunk {chunk_number} "
                f"(Section: {chunk_row['Section']}; "
                f"Similarity: {chunk_row['SimilarityScore']:.3f}):\n"
                f"{chunk_row['Chunk']}"
            )

        trial_context = f"""
NCT ID: {nct_id}

Title: {first['Title']}

Actual Conditions: {first['Conditions']}

Searched Condition: {first['ConditionSearched']}

Recruitment Status: {first['Status']}

Study Type: {first['StudyType']}

Study Phase: {first['Phases']}

Minimum Age: {first['MinimumAge']}

Maximum Age: {first['MaximumAge']}

Sex: {first['Sex']}

Retrieved Evidence:
{chr(10).join(evidence_parts)}

Source: {first['Source']}
"""
        context_parts.append(trial_context)

    return "\n".join(context_parts)

# ------------------------------------------------------------
# 11. BUILD IMPROVED ANSWER-GENERATION PROMPT
# ------------------------------------------------------------

def build_answer_prompt(
    query,
    condition,
    status,
    context
):

    prompt = f"""
You are TrialLens, an educational clinical-trial discovery assistant.
Your task is to summarize the supplied trial records accurately.

USER QUESTION:
{query}

SELECTED CONDITION:
{condition or "All"}

SELECTED RECRUITMENT STATUS:
{status or "All"}

RETRIEVED CLINICAL TRIAL EVIDENCE:
{context}

STRICT EVIDENCE AND METADATA RULES:

1. Use only the supplied context. Do not use outside medical knowledge,
   assumptions, or information remembered from other trials.

2. Treat every NCT ID as a separate study. Never transfer a treatment,
   outcome, eligibility criterion, phase, age, sex, or status from one
   NCT ID to another.

3. For structured metadata fields (Actual Conditions, Recruitment Status,
   Study Type, Study Phase, Minimum Age, Maximum Age, and Sex), use the
   corresponding labeled metadata values supplied for that NCT ID.
   These fields are authoritative for reporting those metadata values.
   Do not say a metadata field is missing if its labeled value is present.

4. For study objectives, interventions, treatment schedules, eligibility
   details, and outcomes, use only the evidence chunks supplied under the
   same NCT ID. Do not infer results from a title, objective, intervention,
   or planned outcome measure.

5. Clearly distinguish a treatment being studied from supportive care
   given during another treatment. For example, a trial comparing a
   supportive agent used during chemotherapy is not necessarily comparing
   two breast-cancer chemotherapy regimens.

6. Do not describe all retrieved studies as investigating the same type
   of treatment unless that is explicitly true for every study. Use a
   cautious overview that reflects differences between studies.

7. Report study phase, age, and sex separately for each trial, using the
   supplied metadata. If a value is explicitly missing or marked
   "Not available", report "Not available in the collected data".
   Never replace a supplied value with a claim that it is missing.

8. Do not invent sample sizes, study durations, treatment effects,
   percentages, clinical results, statistical findings, or safety results.

9. Distinguish study objectives and outcome measures from actual results.
   A planned assessment of survival or response does not mean the trial
   demonstrated a survival benefit or response.

10. Report recruitment status exactly as supplied. Explain that it is a
    collected snapshot and may have changed.

11. Distinguish Actual Conditions from Searched Condition. Do not assume
    that the searched condition is the only condition in the study.

12. Never confuse Type 1 Diabetes with Type 2 Diabetes.

13. If information is not present in the supplied evidence for a specific
    trial, say "Not found in the retrieved evidence for this trial".
    Do not say it is absent from the complete ClinicalTrials.gov record.

14. In the Information Gaps section, list gaps only when the information
    is genuinely absent from the supplied metadata and evidence chunks.
    Identify the relevant NCT ID for each gap. Do not make blanket claims
    such as "no trial provides dosing details" if even one supplied
    evidence chunk contains dosing information.

15. Do not describe an entire category of information as unavailable
    based on one trial. Check all supplied chunks and metadata across the
    relevant NCT IDs before making a general statement.

16. Do not make diagnoses, prescribe treatments, determine individual
    patient eligibility, or claim a trial is suitable for a patient.

17. Do not describe a trial as successful, effective, safe, or beneficial
    unless actual findings supporting that statement are explicitly
    included in the supplied evidence.

18. Before finalizing, check every factual statement against the evidence
    and metadata for its corresponding NCT ID. If uncertain, omit the
    statement or clearly identify the information as unavailable.

OUTPUT FORMAT:

### Search Overview
Briefly summarize the number and broad types of retrieved studies.
Mention the selected condition and recruitment status. Avoid saying all
trials investigate the same treatment when their purposes differ.

### Retrieved Trials
Create a separate subsection for every NCT ID. For each trial provide:
- NCT ID and title
- Actual conditions
- Recruitment status
- Study type and phase
- Minimum age, maximum age, and sex, using supplied metadata
- Relevant objectives, interventions, eligibility details, or outcome
  measures that are explicitly supported by its evidence chunks
Clearly label objectives or planned outcome measures; do not present them
as actual study results.

### Information Gaps
List only information missing from the supplied context. Specify the
NCT ID(s) to which each gap applies. Do not repeat a gap if the information
is available elsewhere in that trial's supplied evidence or metadata.

### Important Note
State that TrialLens is an educational research prototype, not a
clinically validated decision system. Recruitment status reflects the
collected snapshot and may have changed.

Keep the response factual, concise, and organized.
"""

    return prompt


# ------------------------------------------------------------
# 12. GENERATE ANSWER USING GROQ
# ------------------------------------------------------------

def generate_answer(
    query,
    top_k=5,
    condition=None,
    status=None
):

    # Retrieve clinical trials.

    retrieved_trials = retrieve_trials(
        query=query,
        top_k=top_k,
        condition=condition,
        status=status
    )

    if retrieved_trials.empty:

        return (
            "No clinical trials matched the search criteria "
            "in the collected dataset. Try a broader query "
            "or a different condition or recruitment status."
        )

    # Prepare evidence context.

    context = prepare_trial_context(
        retrieved_trials
    )

    # Build prompt.

    prompt = build_answer_prompt(
        query=query,
        condition=condition,
        status=status,
        context=context
    )

    print(
        "\nGenerating evidence-based answer using Groq..."
    )

    generated_answer = ""

    try:

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are TrialLens, a strict "
                        "evidence-grounded clinical trial "
                        "discovery assistant. Treat each "
                        "clinical trial independently. "
                        "Never invent or generalize trial "
                        "information. Use only supplied evidence."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=2500
        )

        generated_answer = (
            response.choices[0].message.content or ""
        ).strip()

    except Exception as error:

        print(
            "\nGroq generation error:",
            str(error)
        )

        generated_answer = ""

    # Handle empty responses.

    if not generated_answer:

        print(
            "\nWarning: Groq returned an empty answer. "
            "Using fallback message."
        )

        generated_answer = (
            "The language model did not generate an overview. "
            "The collected trial details are provided below."
        )

    # --------------------------------------------------------
    # BUILD FACTUAL TRIAL DETAILS FROM DATASET
    # --------------------------------------------------------

    factual_details = []

    unique_trials = retrieved_trials.drop_duplicates(subset=["NCTId"])

    for _, row in unique_trials.iterrows():

        factual_details.append(f"""
NCT ID: {row['NCTId']}

Title: {row['Title']}

Actual Conditions: {row['Conditions']}

Recruitment Status: {row['Status']}

Study Type: {row['StudyType']}

Phase: {row['Phases']}

Minimum Age: {row['MinimumAge']}

Maximum Age: {row['MaximumAge']}

Sex: {row['Sex']}

Source: {row['Source']}
""")

    verified_records = "\n".join(
        factual_details
    )

    # --------------------------------------------------------
    # BUILD FINAL RESPONSE
    # --------------------------------------------------------

    final_answer = f"""
==================================================
TRIALLENS SEARCH RESULTS
==================================================

Selected Condition: {condition or "All"}

Selected Recruitment Status: {status or "All"}


LLM-GENERATED OVERVIEW
--------------------------------------------------

{generated_answer}


TRIAL DETAILS FROM COLLECTED DATA
--------------------------------------------------

{verified_records}


IMPORTANT INFORMATION
--------------------------------------------------

The trial details above are taken from the
collected dataset.

Recruitment status reflects the collected snapshot
and may have changed. Verify current information
on ClinicalTrials.gov before making decisions.

TrialLens is an educational discovery prototype.
It does not diagnose, prescribe, or determine
clinical eligibility.
"""

    return final_answer


# ------------------------------------------------------------
# 13. INTERACTIVE COMMAND-LINE INTERFACE
# ------------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 65)
    print("             TRIALLENS CLINICAL TRIAL SEARCH")
    print("=" * 65)

    print("\nAvailable Conditions:")

    conditions = [
        "Type 2 Diabetes",
        "Breast Cancer",
        "Cardiovascular Diseases",
        "Asthma",
        "Alzheimer Disease",
        "Chronic Kidney Disease"
    ]

    for number, item in enumerate(
        conditions,
        start=1
    ):

        print(f"{number}. {item}")

    print("7. All Conditions")

    try:

        condition_choice = int(
            input("\nSelect condition (1-7): ")
        )

        if condition_choice == 7:

            selected_condition = None

        elif 1 <= condition_choice <= 6:

            selected_condition = conditions[
                condition_choice - 1
            ]

        else:

            raise ValueError(
                "Invalid condition selection."
            )

    except ValueError as error:

        print("Invalid input:", error)

        raise SystemExit

    print("\nAvailable Recruitment Statuses:")

    statuses = [
        "RECRUITING",
        "COMPLETED",
        "TERMINATED",
        "NOT_YET_RECRUITING",
        "ACTIVE_NOT_RECRUITING",
        "WITHDRAWN",
        "ENROLLING_BY_INVITATION",
        "UNKNOWN"
    ]

    for number, item in enumerate(
        statuses,
        start=1
    ):

        print(f"{number}. {item}")

    print("9. All Statuses")

    try:

        status_choice = int(
            input("\nSelect status (1-9): ")
        )

        if status_choice == 9:

            selected_status = "All"

        elif 1 <= status_choice <= 8:

            selected_status = statuses[
                status_choice - 1
            ]

        else:

            raise ValueError(
                "Invalid recruitment status selection."
            )

    except ValueError as error:

        print("Invalid input:", error)

        raise SystemExit

    query = input(
        "\nEnter your clinical trial search query: "
    ).strip()

    if not query:

        print("Search query cannot be empty.")

        raise SystemExit

    answer = generate_answer(
        query=query,
        top_k=5,
        condition=selected_condition,
        status=selected_status
    )

    print("\n" + "=" * 65)

    print("             TRIALLENS SEARCH ANSWER")

    print("=" * 65)

    print(answer)

    print("\n" + "=" * 65)

    print("Search completed.")
