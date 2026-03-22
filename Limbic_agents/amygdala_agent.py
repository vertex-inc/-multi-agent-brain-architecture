"""
Amygdala Agent — Threat detection + emotional salience scoring.

Role: Evaluates the perception for potential threats, urgency,
and emotional significance. Its output drives the limbic dominance
weight in the Integration Layer.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

SYSTEM_PROMPT = """You are the Amygdala — the brain's threat-detection center.

Given a perception object (entities, intent, emotional_load, urgency, topic),
evaluate the input for:
1. Whether a threat is detected (physical, social, financial, existential).
2. An urgency score (0-10) reflecting how immediately the situation demands action.
3. An emotional salience score (0-10) reflecting how emotionally significant the input is.
4. A flag: one of "fear", "caution", or "safe".

Return ONLY valid JSON with these exact keys:
{
  "threat_detected": true/false,
  "urgency_score": 0-10,
  "emotional_salience": 0-10,
  "flag": "fear" | "caution" | "safe"
}

No markdown, no explanation — just the JSON object."""


async def run_amygdala(perception: dict) -> dict:
    """Run the Amygdala agent on the given perception."""
    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(perception)),
    ])

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {
            "threat_detected": False,
            "urgency_score": perception.get("urgency", 5),
            "emotional_salience": perception.get("emotional_load", 5),
            "flag": "caution",
        }

    # Ensure required keys
    result.setdefault("threat_detected", False)
    result.setdefault("urgency_score", 5)
    result.setdefault("emotional_salience", 5)
    result.setdefault("flag", "caution")

    return result
