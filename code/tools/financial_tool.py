import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from llm.groq_client import call_llm_json

# numerical features from the dataset
NUMERIC_FEATURES = [
    "age",
    "base_cost",
    "outstanding_amount",
    "coverage_ratio",
    "provider_avg_claim_amount",
    "specialty_avg_claim_amount",
    "provider_claim_count",
    "provider_monthly_claim_count",
]

CATEGORICAL_FEATURES = [
    "gender",
    "condition_description",
    "procedure_description",
    "provider_speciality",
]


def _prepare_training_data(df):
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    train_df = df[features + ["claim_amount"]].copy()
    train_df = train_df.replace([np.inf, -np.inf], np.nan)

    for col in NUMERIC_FEATURES:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0)

    for col in CATEGORICAL_FEATURES:
        train_df[col] = train_df[col].fillna("Unknown").astype(str)

    train_df["claim_amount"] = pd.to_numeric(
        train_df["claim_amount"], errors="coerce"
    ).fillna(0)

    return train_df


def train_financial_model(df):
    train_df = _prepare_training_data(df)

    X = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = train_df["claim_amount"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X, y)

    return pipeline


def run_financial_tool(claim, df):
    """
    Financial Tool

    Random Forest predicts expected claim amount.
    LLM explains actual vs expected deviation.
    """

    model = train_financial_model(df)

    claim_df = pd.DataFrame([claim])
    claim_df = claim_df.replace([np.inf, -np.inf], np.nan)

    for col in NUMERIC_FEATURES:
        claim_df[col] = pd.to_numeric(claim_df[col], errors="coerce").fillna(0)

    for col in CATEGORICAL_FEATURES:
        claim_df[col] = claim_df[col].fillna("Unknown").astype(str)

    expected_amount = float(
        model.predict(claim_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[0]
    )

    actual_amount = float(claim.get("claim_amount", 0))

    difference = actual_amount - expected_amount

    percent_difference = (
        difference / expected_amount * 100
        if expected_amount > 0
        else 0
    )

    abs_percent_difference = abs(percent_difference)

    # Python decides the financial assessment
    if percent_difference >= 100:
        financial_assessment = "High Concern"
    elif percent_difference >= 40:
        financial_assessment = "Moderate Concern"
    else:
        financial_assessment = "Low Concern"

    system_prompt = """
You are a healthcare payment integrity financial analyst.

A machine learning model has already predicted the expected claim amount.
Your job is ONLY to explain whether the actual claim amount is reasonable compared to the predicted amount.

Do NOT change the computed financial assessment.

Confidence should be a number between 0 and 1:
0.90-1.00 = very confident
0.70-0.89 = confident
0.40-0.69 = moderate confidence
below 0.40 = low confidence

Deviation guide:
- Less than 40% difference is Low Concern and should be described as a small or acceptable deviation.
- 40% to 100% higher than expected is Moderate Concern.
- More than 100% higher than expected is High Concern.

Return ONLY valid JSON:

{
  "tool": "Financial Tool",
  "method": "Random Forest Regressor",
  "financial_assessment": "...",
  "actual_claim_amount": 0.0,
  "expected_claim_amount": 0.0,
  "percent_difference": 0.0,
  "reasoning": "...",
  "confidence": 0.0
}
"""

    user_prompt = f"""
Claim ID:
{claim.get("claim_id")}

Procedure:
{claim.get("procedure_description")}

Condition:
{claim.get("condition_description")}

Provider Specialty:
{claim.get("provider_speciality")}

Actual Claim Amount:
{actual_amount:.2f}

ML Predicted Expected Claim Amount:
{expected_amount:.2f}

Difference:
{difference:.2f}

Percent Difference:
{percent_difference:.2f}%

Amount vs Provider Average:
{claim.get("amount_vs_provider_avg")}

Amount vs Specialty Average:
{claim.get("amount_vs_specialty_avg")}

Procedure Amount Z-score:
{claim.get("procedure_amount_zscore")}

Computed Financial Assessment:
{financial_assessment}

IMPORTANT:
Keep financial_assessment exactly as:
{financial_assessment}

Only explain why the actual amount is or is not unusual compared to the ML-predicted expected amount.
"""

    result = call_llm_json(system_prompt, user_prompt)

    if "raw_response" in result:
        result = {
            "tool": "Financial Tool",
            "method": "Random Forest Regressor",
            "financial_assessment": financial_assessment,
            "actual_claim_amount": round(actual_amount, 2),
            "expected_claim_amount": round(expected_amount, 2),
            "percent_difference": round(percent_difference, 2),
            "reasoning": result["raw_response"],
            "confidence": 0.75,
        }

    result["financial_assessment"] = financial_assessment
    result["actual_claim_amount"] = round(actual_amount, 2)
    result["expected_claim_amount"] = round(expected_amount, 2)
    result["percent_difference"] = round(percent_difference, 2)

    return result