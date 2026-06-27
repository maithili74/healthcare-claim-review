import json
from llm.groq_client import call_llm_json


def _is_high(value):
    return "high" in str(value).lower()


def _is_moderate(value):
    value = str(value).lower()
    return "moderate" in value or "medium" in value


def run_decision_agent(claim, clinical, financial, provider, fraud):
    """
    Final Decision Agent.

    Python computes decision and risk score.
    LLM explains the decision.
    """

    clinical_assessment = clinical.get("clinical_risk_level", "")
    financial_assessment = financial.get("financial_assessment", "")
    provider_assessment = provider.get("provider_assessment", "")

    fraud_flag = fraud.get("potential_fraud_flag", "NO")
    fraud_risk = fraud.get("fraud_risk", "Low")
    suspicious_signals = fraud.get("suspicious_signals", [])
    
    repeat_30d = int(claim.get("repeat_procedure_30d", 0))
    duplicate_flag = int(claim.get("duplicate_flag", 0))

    high_count = sum([
        _is_high(clinical_assessment),
        _is_high(financial_assessment),
        _is_high(provider_assessment),
        _is_high(fraud_risk),
    ])

    moderate_count = sum([
        _is_moderate(clinical_assessment),
        _is_moderate(financial_assessment),
        _is_moderate(provider_assessment),
        _is_moderate(fraud_risk),
    ])

    overall_risk_score = 20 + (high_count * 25) + (moderate_count * 10)

    if fraud_flag == "YES":
        overall_risk_score += 15

    overall_risk_score = min(100, int(overall_risk_score))

    # Final decision logic
    # Final decision logic
    if duplicate_flag == 1:
        claim_decision = "DENY"
    elif repeat_30d == 1:
        claim_decision = "MANUAL REVIEW"
    elif fraud_flag == "YES":
        claim_decision = "MANUAL REVIEW"
    elif _is_high(clinical_assessment):
        claim_decision = "MANUAL REVIEW"
    elif high_count >= 2:
        claim_decision = "MANUAL REVIEW"
    elif moderate_count >= 2:
        claim_decision = "MANUAL REVIEW"
    elif overall_risk_score < 35:
        claim_decision = "APPROVE"
    else:
        claim_decision = "MANUAL REVIEW"

    system_prompt = """
You are a senior healthcare payment integrity reviewer.

Python has already computed:
- claim_decision
- fraud_flag
- fraud_risk
- overall_risk_score

Your job is ONLY to explain these computed values clearly.

Do NOT change the computed values.
Do NOT say fraud is confirmed.
Only discuss potential fraud indicators.

If claim_decision is APPROVE:
- final_explanation must support approval.
- recommended_action must say to approve/process the claim normally.
- Do NOT recommend manual review.

If claim_decision is MANUAL REVIEW:
- recommended_action should say manual review is needed.

If claim_decision is DENY:
- recommended_action should say deny the claim due to the identified issue.

Return ONLY valid JSON in this exact format:

{
  "agent": "Decision Agent",
  "claim_decision": "APPROVE | DENY | MANUAL REVIEW",
  "potential_fraud_flag": "YES | NO",
  "fraud_risk": "LOW | MEDIUM | HIGH",
  "overall_risk_score": 0,
  "top_reasons": ["reason 1", "reason 2", "reason 3"],
  "final_explanation": "short paragraph",
  "recommended_action": "short action"
}
"""

    user_prompt = f"""
Computed Claim Decision:
{claim_decision}

Computed Potential Fraud Flag:
{fraud_flag}

Computed Fraud Risk:
{fraud_risk.upper()}

Computed Overall Risk Score:
{overall_risk_score}

Suspicious Signals:
{suspicious_signals}

CLAIM DETAILS:
Claim ID: {claim.get("claim_id")}
Patient Age: {claim.get("age")}
Gender: {claim.get("gender")}
Condition: {claim.get("condition_description")}
Procedure: {claim.get("procedure_description")}
Reason: {claim.get("reason_description")}
Provider: {claim.get("provider_name")}
Provider Specialty: {claim.get("provider_speciality")}
Claim Amount: {claim.get("claim_amount")}
Claim Status: {claim.get("claim_status")}
Outstanding Amount: {claim.get("outstanding_amount")}

ENGINEERED SIGNALS:
Duplicate Flag: {claim.get("duplicate_flag")}
Repeat Procedure Within 30 Days: {claim.get("repeat_procedure_30d")}
Amount vs Provider Average: {claim.get("amount_vs_provider_avg")}
Amount vs Specialty Average: {claim.get("amount_vs_specialty_avg")}
Procedure Amount Z-score: {claim.get("procedure_amount_zscore")}
Provider Claim Count: {claim.get("provider_claim_count")}

SPECIALIST TOOL OUTPUTS:

Clinical Tool:
{json.dumps(clinical, indent=2)}

Financial Tool:
{json.dumps(financial, indent=2)}

Provider Tool:
{json.dumps(provider, indent=2)}

Fraud Tool:
{json.dumps(fraud, indent=2)}

Important:
Use the computed values exactly.
Do not change claim_decision, fraud flag, fraud risk, or risk score.
Explain the decision in clear language for a payment integrity reviewer.
"""

    result = call_llm_json(system_prompt, user_prompt)

    if "raw_response" in result:
        result = {
            "agent": "Decision Agent",
            "claim_decision": claim_decision,
            "potential_fraud_flag": fraud_flag,
            "fraud_risk": fraud_risk.upper(),
            "overall_risk_score": overall_risk_score,
            "top_reasons": suspicious_signals if suspicious_signals else [
                "Clinical, financial, provider, and fraud tools did not identify strong concerns."
            ],
            "final_explanation": result["raw_response"],
            "recommended_action": "Proceed according to the computed recommendation."
        }

    result["claim_decision"] = claim_decision
    result["potential_fraud_flag"] = fraud_flag
    result["fraud_risk"] = fraud_risk.upper()
    result["overall_risk_score"] = overall_risk_score

    if claim_decision == "APPROVE":
        result["recommended_action"] = "Approve/process the claim normally."
    elif claim_decision == "MANUAL REVIEW":
        result["recommended_action"] = "Route the claim to a payment integrity reviewer for manual review."
    elif claim_decision == "DENY":
        result["recommended_action"] = "Deny the claim based on the identified issue."

    return result