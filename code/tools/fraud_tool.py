import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# financial columns in the dataset
FEATURES = [
    "claim_amount",
    "base_cost",
    "outstanding_amount",
    "coverage_ratio",
    "amount_vs_provider_avg",
    "amount_vs_specialty_avg",
    "procedure_amount_zscore",
    "provider_claim_count",
    "provider_monthly_claim_count",
    "duplicate_flag",
    "repeat_procedure_30d",
]


def run_fraud_tool(claim, df):
    """
    Fraud Tool using Isolation Forest.

    This does not confirm fraud.
    It detects claims that look unusual compared to other claims.
    """

    model_df = df[FEATURES].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(model_df)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )

    model.fit(X)

    claim_features = pd.DataFrame([claim])[FEATURES]
    claim_features = claim_features.replace([np.inf, -np.inf], np.nan).fillna(0)

    claim_scaled = scaler.transform(claim_features)

    prediction = model.predict(claim_scaled)[0]
    anomaly_score_raw = model.decision_function(claim_scaled)[0]

    # Lower decision_function score = more anomalous
    ml_fraud_score = round(float(1 / (1 + np.exp(5 * anomaly_score_raw))), 3)

    suspicious_signals = []

    if int(claim.get("duplicate_flag", 0)) == 1:
        suspicious_signals.append("Possible duplicate claim")

    if int(claim.get("repeat_procedure_30d", 0)) == 1:
        suspicious_signals.append("Repeat procedure within 30 days")

    if float(claim.get("procedure_amount_zscore", 0)) >= 2.5:
        suspicious_signals.append("Claim amount is a statistical outlier")

    if float(claim.get("amount_vs_provider_avg", 0)) >= 2:
        suspicious_signals.append("Claim amount is much higher than provider average")

    if float(claim.get("amount_vs_specialty_avg", 0)) >= 2:
        suspicious_signals.append("Claim amount is much higher than specialty average")

    if prediction == -1:
        suspicious_signals.append("Isolation Forest flagged this claim as anomalous")

    if prediction == -1 and len(suspicious_signals) >= 2:
        potential_fraud_flag = "YES"
        fraud_risk = "HIGH"
    elif prediction == -1 or len(suspicious_signals) >= 2:
        potential_fraud_flag = "YES"
        fraud_risk = "MEDIUM"
    else:
        potential_fraud_flag = "NO"
        fraud_risk = "LOW"

    return {
        "tool": "Fraud Tool",
        "method": "Isolation Forest",
        "anomaly_prediction": "Anomalous" if prediction == -1 else "Normal",
        "ml_fraud_score": ml_fraud_score,
        "potential_fraud_flag": potential_fraud_flag,
        "fraud_risk": fraud_risk,
        "suspicious_signals": suspicious_signals,
        "reasoning": (
            "Isolation Forest compared this claim against engineered claim, provider, "
            "billing, and repeat-procedure features. The fraud flag indicates a potential "
            "suspicious pattern, not confirmed fraud."
        )
    }