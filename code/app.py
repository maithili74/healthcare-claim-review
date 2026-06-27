import pandas as pd
import streamlit as st

from tools.clinical_tool import run_clinical_tool
from tools.financial_tool import run_financial_tool
from tools.provider_tool import run_provider_tool
from tools.fraud_tool import run_fraud_tool
from llm.decision_agent import run_decision_agent


st.set_page_config(
    page_title="Agentic Claims Review Assistant",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv("data/processed/claim_review_dataset.csv")


def assessment_card(title, assessment, reasoning, confidence=None, extra=None):
    with st.container(border=True):
        st.subheader(title)
        st.write(f"**Assessment:** {assessment}")

        if confidence is not None:
            st.write(f"**Confidence:** {confidence}")

        if extra:
            for label, value in extra.items():
                st.write(f"**{label}:** {value}")

        st.text(reasoning)


df = load_data()

st.title("Agentic Healthcare Payment Integrity Review Assistant")

st.write(
    "This reviews healthcare claims using clinical RAG, financial prediction, "
    "provider behavior analysis, fraud anomaly detection, and an LLM decision agent."
)

st.sidebar.header("Claim Selection")

claim_id = st.sidebar.selectbox(
    "Select Claim ID",
    df["claim_id"].astype(str).unique()
)

claim = df[df["claim_id"].astype(str) == claim_id].iloc[0].to_dict()

st.subheader("Selected Claim")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Patient Age", claim.get("age"))
    st.write("**Gender:**", claim.get("gender"))
    st.write("**Condition:**", claim.get("condition_description"))

with col2:
    st.write("**Procedure:**", claim.get("procedure_description"))
    st.write("**Reason:**", claim.get("reason_description"))
    st.metric("Claim Amount", f"${float(claim.get('claim_amount', 0)):,.2f}")

with col3:
    st.write("**Provider:**", claim.get("provider_name"))
    st.write("**Specialty:**", claim.get("provider_speciality"))
    st.write("**Service Date:**", claim.get("service_date"))


if st.button("Run AI Claim Review", type="primary"):

    with st.spinner("Running Clinical RAG Tool..."):
        clinical = run_clinical_tool(claim)

    with st.spinner("Running Financial Prediction Tool..."):
        financial = run_financial_tool(claim, df)

    with st.spinner("Running Provider Behavior Tool..."):
        provider = run_provider_tool(claim)

    with st.spinner("Running Fraud Anomaly Tool..."):
        fraud = run_fraud_tool(claim, df)

    with st.spinner("Running Decision Agent..."):
        decision = run_decision_agent(
            claim,
            clinical,
            financial,
            provider,
            fraud
        )

    st.divider()
    st.header("Final Claim Review Decision")

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.metric("Claim Decision", decision.get("claim_decision"))

    with d2:
        st.metric("Potential Fraud Flag", decision.get("potential_fraud_flag"))

    with d3:
        st.metric("Fraud Risk", decision.get("fraud_risk"))

    with d4:
        st.metric("Risk Score", decision.get("overall_risk_score"))

    st.subheader("Final Explanation")
    st.write(decision.get("final_explanation"))

    st.subheader("Recommended Action")
    st.info(decision.get("recommended_action"))

    st.subheader("Top Reasons")
    for reason in decision.get("top_reasons", []):
        st.write(f"- {reason}")

    st.divider()
    st.header("Specialist Tool Summary")

    col_left, col_right = st.columns(2)

    with col_left:
        assessment_card(
            title="Clinical RAG Tool",
            assessment=clinical.get("clinical_assessment", clinical.get("clinical_risk_level")),
            reasoning=clinical.get("reasoning", ""),
            confidence=clinical.get("confidence"),
            extra={
                "Risk Level": clinical.get("clinical_risk_level"),
                "Method": clinical.get("method")
            }
        )

        assessment_card(
            title="Provider Behavior Tool",
            assessment=provider.get("provider_assessment"),
            reasoning=provider.get("reasoning", ""),
            confidence=provider.get("confidence")
        )

    with col_right:
        assessment_card(
            title="Financial Prediction Tool",
            assessment=financial.get("financial_assessment"),
            reasoning=financial.get("reasoning", ""),
            confidence=financial.get("confidence"),
            extra={
                "Method": financial.get("method"),
                "Actual Amount": f"${financial.get('actual_claim_amount', 0):,.2f}",
                "Expected Amount": f"${financial.get('expected_claim_amount', 0):,.2f}",
                "Difference": f"{financial.get('percent_difference', 0)}%"
            }
        )

        assessment_card(
            title="Fraud Anomaly Tool",
            assessment=f"{fraud.get('fraud_risk')} Risk",
            reasoning=fraud.get("reasoning", ""),
            extra={
                "Method": fraud.get("method"),
                "Anomaly Prediction": fraud.get("anomaly_prediction"),
                "ML Fraud Score": fraud.get("ml_fraud_score"),
                "Fraud Flag": fraud.get("potential_fraud_flag")
            }
        )

    st.divider()
    st.subheader("Engineered Claim Signals")

    signal_cols = [
        "claim_amount",
        "base_cost",
        "outstanding_amount",
        "coverage_ratio",
        "provider_avg_claim_amount",
        "specialty_avg_claim_amount",
        "amount_vs_provider_avg",
        "amount_vs_specialty_avg",
        "procedure_amount_zscore",
        "duplicate_flag",
        "repeat_procedure_30d",
        "provider_claim_count",
        "provider_monthly_claim_count"
    ]

    signals = {
        col: claim.get(col)
        for col in signal_cols
        if col in claim
    }

    st.dataframe(
        pd.DataFrame([signals]).T.rename(columns={0: "Value"}),
        use_container_width=True
    )

    with st.expander("View full technical tool outputs"):
        tab1, tab2, tab3, tab4 = st.tabs([
            "Clinical JSON",
            "Financial JSON",
            "Provider JSON",
            "Fraud JSON"
        ])

        with tab1:
            st.json(clinical)

        with tab2:
            st.json(financial)

        with tab3:
            st.json(provider)

        with tab4:
            st.json(fraud)

else:
    st.info("Select a claim and click 'Run AI Claim Review'.")