# Healthcare Claim Review

Agentic healthcare payment integrity review assistant for claim evaluation, fraud detection, provider behavior analysis, and clinical appropriateness assessment.

## Overview

This project combines data-driven tools and a lightweight LLM-enabled review agent to assist with healthcare claims screening.

Key capabilities:
- Streamlit dashboard for interactive claim review
- Clinical RAG assessment using a guideline knowledge base
- Financial anomaly detection with a trained regression model
- Provider behavior risk scoring
- Fraud anomaly detection with Isolation Forest
- Final decision explanation via an LLM decision agent
- Synthetic test dataset generation and evaluation automation

## Repository Structure

- `code/app.py` - Streamlit application entry point
- `code/llm/groq_client.py` - Groq LLM client wrapper, depends on `GROQ_API_KEY`
- `code/llm/decision_agent.py` - Final decision agent that synthesizes tool outputs and generates explanations
- `code/tools/clinical_tool.py` - Clinical appropriateness tool using retrieved guidance
- `code/tools/financial_tool.py` - Expected claim amount predictor and deviation explainer
- `code/tools/provider_tool.py` - Provider behavior assessment tool
- `code/tools/fraud_tool.py` - Fraud anomaly detection tool using Isolation Forest
- `code/utils/data_prep.py` - Prepares raw dataset into processed claim review dataset
- `code/utils/create_synthetic_test_dataset.py` - Creates a synthetic test set for evaluation
- `code/evaluation/evaluate_synthetic_tests.py` - Runs the claim review pipeline on synthetic cases
- `code/data/raw/` - Raw source data CSV files
- `code/data/processed/` - Processed dataset outputs
- `code/knowledge_base/clinical_guidelines.txt` - Guideline text used by the clinical tool

## Dependencies

Dependencies are listed in the repository root `requirements.txt`:

- pandas
- numpy
- scikit-learn
- streamlit
- plotly
- python-dotenv
- groq
- langgraph
- sentence-transformers

## Setup

1. Create a Python environment and install dependencies.

```bash
cd ~/healthcare-claim-review/code
pip install -r requirements.txt
```

2. Create a `.env` file in `healthcare-claim-review/code` with:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

3. Prepare the processed dataset if it is not already available:

```bash
python utils/data_prep.py
```

4. Optionally generate the synthetic test dataset:

```bash
python utils/create_synthetic_test_dataset.py
```

## Running the App

From `healthcare-claim-review/code`:

```bash
streamlit run app.py
```

The Streamlit app loads `data/processed/claim_review_dataset.csv` and lets you select a claim to run the full review pipeline.

## Evaluation

Run synthetic test evaluation from the same folder:

```bash
python evaluation/evaluate_synthetic_tests.py
```

This script compares pipeline outputs against `data/processed/synthetic_test_claims_20.csv` and writes results to `evaluation/synthetic_test_results.csv`.

## Notes

- The project uses a Groq LLM endpoint via `code/llm/groq_client.py`.
- The clinical tool uses `sentence-transformers` to retrieve relevant guideline snippets.
- The fraud tool uses engineered features and an Isolation Forest to flag anomalous claims.
- The decision agent uses deterministic scoring logic plus LLM-generated explanations.

## Recommended Workflow

1. Prepare or verify processed claim data.
2. Start the Streamlit app and select a claim.
3. Run the review pipeline to see tool outputs, final decision, and explanations.
4. Use synthetic evaluation to validate behavior against known test scenarios.
