import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from llm.groq_client import call_llm_json


GUIDELINE_PATH = Path("knowledge_base/clinical_guidelines.txt")  

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# loading the guidelines from the knowledge base
def load_guidelines():
    text = GUIDELINE_PATH.read_text()
    chunks = [
        chunk.strip()
        for chunk in text.split("------------------------------------------------------------")
        if chunk.strip()
    ]
    return chunks


def retrieve_guidelines(query, top_k=3):
    chunks = load_guidelines()

    chunk_embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)[0]

    similarities = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    top_indices = similarities.argsort()[-top_k:][::-1]

    return [
        {
            "guideline": chunks[i],
            "similarity": round(float(similarities[i]), 3)
        }
        for i in top_indices
    ]

    # Using clinical Tool using lightweight RAG.

    # Retrieves relevant clinical guideline snippets and asks the LLM to explain clinical appropriateness.
    
def run_clinical_tool(claim):


    query = f"""
Condition: {claim.get("condition_description")}
Procedure: {claim.get("procedure_description")}
Reason: {claim.get("reason_description")}
Age: {claim.get("age")}
Gender: {claim.get("gender")}
"""

    retrieved_guidelines = retrieve_guidelines(query, top_k=3)

    system_prompt = """
You are a healthcare clinical review assistant for a payment integrity team.

You will receive claim information and retrieved clinical guideline evidence.
Use the retrieved guidelines to assess clinical appropriateness.

Return ONLY valid JSON:

{
  "tool": "Clinical Tool",
  "method": "Lightweight RAG over clinical guideline knowledge base",
  "clinical_assessment": "Clinically Appropriate | Possibly Appropriate | Clinically Inconsistent",
  "clinical_risk_level": "Low Concern | Moderate Concern | High Concern",
  "retrieved_guidelines_used": ["guideline 1", "guideline 2"],
  "reasoning": "short explanation",
  "confidence": 0.0
}

Guidance:
- Use Clinically Appropriate / Low Concern when the procedure is clearly supported.
- Use Possibly Appropriate / Moderate Concern when the procedure may be reasonable but documentation is incomplete or nonspecific.
- Use Clinically Inconsistent / High Concern only when the procedure clearly conflicts with the condition, demographics, or retrieved evidence.
- Do not deny claims. Only assess clinical appropriateness.
"""

    user_prompt = f"""
Claim Information:

Age:
{claim.get("age")}

Gender:
{claim.get("gender")}

Condition:
{claim.get("condition_description")}

Procedure:
{claim.get("procedure_description")}

Reason:
{claim.get("reason_description")}

Retrieved Clinical Guidelines:
{retrieved_guidelines}

Question:
Based on the claim and retrieved guideline evidence, assess clinical appropriateness.
"""

    result = call_llm_json(system_prompt, user_prompt)

    if "raw_response" in result:
        result = {
            "tool": "Clinical Tool",
            "method": "Lightweight RAG over clinical guideline knowledge base",
            "clinical_assessment": "Possibly Appropriate",
            "clinical_risk_level": "Moderate Concern",
            "retrieved_guidelines_used": [
                item["guideline"] for item in retrieved_guidelines
            ],
            "reasoning": result["raw_response"],
            "confidence": 0.6
        }

    return result