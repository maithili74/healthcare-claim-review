import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_NAME = "llama-3.1-8b-instant"


def call_llm(system_prompt, user_prompt):
    """
    Calls Groq LLM and returns text response.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=700
    )

    return response.choices[0].message.content


def call_llm_json(system_prompt, user_prompt):
    """
    Calls Groq LLM and tries to return JSON.
    If JSON parsing fails, returns raw response.
    """

    response_text = call_llm(system_prompt, user_prompt)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {
            "raw_response": response_text
        }