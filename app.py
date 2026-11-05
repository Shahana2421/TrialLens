
import streamlit as st
import pandas as pd

from rag_pipeline import generate_answer

from eligibility_retrieval import (
    generate_eligibility_answer,
    compare_patient_to_criteria
)


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="TrialLens",
    page_icon="🧬",
    layout="wide"
)


# ==========================================
# 2. APPLICATION HEADER
# ==========================================

st.title("🧬 TrialLens")

st.subheader(
    "Clinical Trial Discovery and Eligibility Assistant"
)

st.write(
    "Discover clinical trials and understand their "
    "published eligibility criteria using AI."
)

st.info(
    "TrialLens is an educational information tool. "
    "It does not provide medical advice or determine "
    "actual clinical eligibility."
)


# ==========================================
# 3. CREATE THREE TABS
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "🔍 Clinical Trial Search",
    "📋 Eligibility Q&A",
    "👤 Patient Profile Comparison"
])


# ==========================================
# TAB 1: CLINICAL TRIAL SEARCH
# ==========================================

with tab1:

    st.header("Clinical Trial Discovery")

    st.write(
        "Ask questions about clinical trials "
        "in the collected dataset."
    )

    query = st.text_input(
        "Enter your clinical trial question:",
        placeholder="Example: Find trials related to Type 2 diabetes"
    )

    if st.button("Search Trials"):

        if query.strip():

            with st.spinner("Searching clinical trials..."):

                answer = generate_answer(query)

            st.subheader("TrialLens Answer")

            st.markdown(answer)

        else:

            st.warning("Please enter a question.")


# ==========================================
# TAB 2: ELIGIBILITY QUESTION ANSWERING
# ==========================================

with tab2:

    st.header("Eligibility Criteria Assistant")

    st.write(
        "Ask questions about the inclusion and exclusion "
        "criteria of a clinical trial."
    )

    nct_id = st.text_input(
        "Enter Clinical Trial ID:",
        placeholder="Example: NCT00994266",
        key="qa_nct_id"
    )

    question = st.text_input(
        "Ask your question:",
        placeholder="Example: What are the exclusion criteria?",
        key="qa_question"
    )

    if st.button("Ask Question"):

        if nct_id.strip() and question.strip():

            with st.spinner("Analyzing eligibility criteria..."):

                answer = generate_eligibility_answer(
                    nct_id.strip(),
                    question.strip()
                )

            st.subheader("TrialLens Answer")

            st.markdown(answer)

            st.markdown(
                f"[View official trial record]"
                f"(https://clinicaltrials.gov/study/{nct_id.strip()})"
            )

        else:

            st.warning(
                "Please enter both the Trial ID and your question."
            )


# ==========================================
# TAB 3: GENERAL PATIENT PROFILE COMPARISON
# ==========================================

with tab3:

    st.header("Patient Profile Comparison")

    st.write(
        "Compare general patient information with the "
        "published eligibility criteria of a clinical trial."
    )

    patient_nct_id = st.text_input(
        "Enter Clinical Trial ID:",
        placeholder="Example: NCT00994266",
        key="patient_nct_id"
    )

    # --------------------------------------
    # BASIC INFORMATION
    # --------------------------------------

    st.subheader("Basic Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age (years)",
            min_value=0,
            max_value=120,
            value=0
        )

    with col2:

        sex = st.selectbox(
            "Sex",
            [
                "Not provided",
                "Female",
                "Male",
                "Other",
                "Prefer not to say"
            ]
        )

    # --------------------------------------
    # MEDICAL INFORMATION
    # --------------------------------------

    st.subheader("Medical Information")

    medical_conditions = st.text_area(
        "Medical conditions / Diagnoses",
        placeholder=(
            "Example: Type 2 diabetes, hypertension, "
            "asthma"
        )
    )

    current_medications = st.text_area(
        "Current medications",
        placeholder="Example: Insulin, Metformin"
    )

    previous_treatments = st.text_area(
        "Previous treatments",
        placeholder="Example: Chemotherapy, surgery, physiotherapy"
    )

    medical_history = st.text_area(
        "Medical history",
        placeholder=(
            "Example: Previous surgeries, "
            "hospitalization, chronic conditions"
        )
    )

    # --------------------------------------
    # MEASUREMENTS
    # --------------------------------------

    st.subheader("Clinical Measurements")

    col3, col4 = st.columns(2)

    with col3:

        height = st.text_input(
            "Height",
            placeholder="Example: 165 cm"
        )

        weight = st.text_input(
            "Weight",
            placeholder="Example: 65 kg"
        )

        bmi = st.text_input(
            "BMI",
            placeholder="Example: 23.9"
        )

    with col4:

        blood_pressure = st.text_input(
            "Blood pressure",
            placeholder="Example: 120/80 mmHg"
        )

        hba1c = st.text_input(
            "HbA1c (%)",
            placeholder="Example: 8.5"
        )

        other_measurements = st.text_area(
            "Other relevant measurements",
            placeholder="Example: Blood glucose, cholesterol"
        )

    # --------------------------------------
    # OTHER INFORMATION
    # --------------------------------------

    st.subheader("Additional Information")

    allergies = st.text_area(
        "Allergies",
        placeholder="Enter known allergies or leave blank"
    )

    pregnancy = st.selectbox(
        "Pregnancy status",
        [
            "Not provided",
            "Yes",
            "No",
            "Not applicable"
        ]
    )

    other_information = st.text_area(
        "Other relevant information",
        placeholder="Enter any other relevant details"
    )

    st.caption(
        "Use synthetic information for testing. "
        "Do not enter names, contact details or "
        "identifiable patient information."
    )

    # --------------------------------------
    # COMPARE PATIENT PROFILE
    # --------------------------------------

    if st.button("Compare Patient Profile"):

        if not patient_nct_id.strip():

            st.warning("Please enter a Clinical Trial ID.")

        else:

            patient = {

                "Age": (
                    age if age > 0 else "Not provided"
                ),

                "Sex": sex,

                "Medical conditions": medical_conditions,

                "Current medications": current_medications,

                "Previous treatments": previous_treatments,

                "Medical history": medical_history,

                "Height": height,

                "Weight": weight,

                "BMI": bmi,

                "Blood pressure": blood_pressure,

                "HbA1c": hba1c,

                "Other measurements": other_measurements,

                "Allergies": allergies,

                "Pregnancy status": pregnancy,

                "Other information": other_information

            }

            with st.spinner(
                "Comparing patient information with trial criteria..."
            ):

                output = compare_patient_to_criteria(
                    patient_nct_id.strip(),
                    patient
                )

            if output is None:

                st.warning(
                    "No eligibility information found for this trial."
                )

            else:

                st.subheader("Comparison Results")

                if output["rows"]:

                    results_df = pd.DataFrame(
                        output["rows"]
                    )

                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.warning(
                        "No criteria were successfully compared."
                    )

                st.info(output["overall"])

                st.markdown(
                    f"[View official trial record]"
                    f"(https://clinicaltrials.gov/study/{patient_nct_id.strip()})"
                )


# ==========================================
# 4. FOOTER
# ==========================================

st.divider()

st.caption(
    "TrialLens | Powered by Python, FAISS, "
    "Sentence Transformers, Groq and Streamlit"
)

st.warning(
    "Important: Patient profile comparisons are experimental "
    "and may be incomplete or incorrect. They do not establish "
    "clinical eligibility. Always consult the clinical trial "
    "team or a qualified healthcare professional."
)
