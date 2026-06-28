import time
import os
import pandas as pd

from tools.clinical_tool import run_clinical_tool
from tools.financial_tool import run_financial_tool
from tools.provider_tool import run_provider_tool
from tools.fraud_tool import run_fraud_tool
from llm.decision_agent import run_decision_agent


TRAIN_PATH = "data/processed/claim_review_dataset.csv"
TEST_PATH = "data/processed/synthetic_test_claims_20.csv"
OUT_PATH = "evaluation/synthetic_test_results.csv"

os.makedirs("evaluation", exist_ok=True)


def run_pipeline(claim, train_df):
    times = {}

    start = time.time()
    clinical = run_clinical_tool(claim)
    times["clinical_latency"] = round(time.time() - start, 3)

    start = time.time()
    financial = run_financial_tool(claim, train_df)
    times["financial_latency"] = round(time.time() - start, 3)

    start = time.time()
    provider = run_provider_tool(claim)
    times["provider_latency"] = round(time.time() - start, 3)

    start = time.time()
    fraud = run_fraud_tool(claim, train_df)
    times["fraud_latency"] = round(time.time() - start, 3)

    start = time.time()
    decision = run_decision_agent(claim, clinical, financial, provider, fraud)
    times["decision_latency"] = round(time.time() - start, 3)

    times["total_latency"] = round(sum(times.values()), 3)

    return clinical, financial, provider, fraud, decision, times


if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    rows = []

    for _, claim_row in test_df.iterrows():
        claim = claim_row.to_dict()

        print(f"\nRunning: {claim['claim_id']} | {claim['test_scenario']}")

        clinical, financial, provider, fraud, decision, times = run_pipeline(
            claim,
            train_df
        )

        row = {
            "claim_id": claim["claim_id"],
            "test_scenario": claim["test_scenario"],
            "expected_decision": claim["expected_decision"],
            "actual_decision": decision.get("claim_decision"),
            "decision_correct": claim["expected_decision"] == decision.get("claim_decision"),
            "expected_fraud_flag": claim["expected_fraud_flag"],
            "actual_fraud_flag": decision.get("potential_fraud_flag"),
            "fraud_flag_correct": claim["expected_fraud_flag"] == decision.get("potential_fraud_flag"),
            "fraud_risk": decision.get("fraud_risk"),
            "risk_score": decision.get("overall_risk_score"),
            "clinical_assessment": clinical.get("clinical_assessment"),
            "clinical_risk_level": clinical.get("clinical_risk_level"),
            "financial_assessment": financial.get("financial_assessment"),
            "provider_assessment": provider.get("provider_assessment"),
            "fraud_anomaly_prediction": fraud.get("anomaly_prediction"),
            "ml_fraud_score": fraud.get("ml_fraud_score"),
            "top_reasons": " | ".join(decision.get("top_reasons", [])),
        }

        row.update(times)
        rows.append(row)

        print("Expected Decision:", row["expected_decision"])
        print("Actual Decision:", row["actual_decision"])
        print("Expected Fraud:", row["expected_fraud_flag"])
        print("Actual Fraud:", row["actual_fraud_flag"])
        print("Total Latency:", row["total_latency"])

    results = pd.DataFrame(rows)
    results.to_csv(OUT_PATH, index=False)

    decision_accuracy = results["decision_correct"].mean()
    fraud_flag_accuracy = results["fraud_flag_correct"].mean()
    avg_latency = results["total_latency"].mean()

    print("\n==============================")
    print("EVALUATION SUMMARY")
    print("==============================")
    print(f"Total test cases: {len(results)}")
    print(f"Decision accuracy: {decision_accuracy:.2%}")
    print(f"Fraud flag accuracy: {fraud_flag_accuracy:.2%}")
    print(f"Average total latency: {avg_latency:.2f} sec")
    print(f"Saved results to: {OUT_PATH}")