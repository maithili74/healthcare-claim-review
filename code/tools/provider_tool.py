from llm.groq_client import call_llm_json

# python computes provider concern level and LLM explains why
def run_provider_tool(claim):
    

    amount_vs_provider = float(claim.get("amount_vs_provider_avg", 0))
    amount_vs_specialty = float(claim.get("amount_vs_specialty_avg", 0))
    provider_claim_count = int(claim.get("provider_claim_count", 0))
    provider_utilization = float(claim.get("provider_utilization", 0))

    if amount_vs_provider >= 2.0 or amount_vs_specialty >= 2.0:
        provider_assessment = "High Concern"
    elif amount_vs_provider >= 1.5 or amount_vs_specialty >= 1.5:
        provider_assessment = "Moderate Concern"
    else:
        provider_assessment = "Low Concern"
        
    if provider_assessment == "Low Concern":
        confidence = 0.90
    elif provider_assessment == "Moderate Concern":
        confidence = 0.70
    else:  # High Concern
        confidence = 0.90

    system_prompt = """
You are a healthcare payment integrity provider behavior analyst.

Python has already computed the provider concern level.
Your job is ONLY to explain why the assessment makes sense.

Do NOT change the assessment.

Confidence should be a number between 0 and 1:
0.90-1.00 = very confident
0.70-0.89 = confident
0.40-0.69 = moderate confidence
below 0.40 = low confidence

Return ONLY valid JSON:

{
  "tool": "Provider Tool",
  "provider_assessment": "...",
  "reasoning": "...",
  "confidence": 0.0
}
"""

    user_prompt = f"""
Provider ID:
{claim.get("provider_id")}

Provider Name:
{claim.get("provider_name")}

Provider Specialty:
{claim.get("provider_speciality")}

Provider Utilization:
{provider_utilization}

Provider Claim Count:
{provider_claim_count}

Current Claim Amount:
{claim.get("claim_amount")}

Provider Average Claim Amount:
{claim.get("provider_avg_claim_amount")}

Specialty Average Claim Amount:
{claim.get("specialty_avg_claim_amount")}

Amount vs Provider Average:
{amount_vs_provider}

Amount vs Specialty Average:
{amount_vs_specialty}

Computed Provider Assessment:
{provider_assessment}

IMPORTANT:
Keep provider_assessment exactly as:
{provider_assessment}

Only explain why this assessment was assigned.
"""

    result = call_llm_json(system_prompt, user_prompt)

    if "raw_response" in result:
        result = {
            "tool": "Provider Tool",
            "provider_assessment": provider_assessment,
            "reasoning": result["raw_response"],
            "confidence": 0.75,
        }

    result["provider_assessment"] = provider_assessment
    result["confidence"] = confidence
    return result