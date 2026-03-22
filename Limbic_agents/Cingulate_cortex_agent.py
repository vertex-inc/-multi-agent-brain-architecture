"""
Cingulate Cortex Agent — Emotional conflict and internal tension detection.

Role: Detects when the perception object contains conflicting emotional
signals or ambiguous intent, producing a tension score and conflict flag.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

SYSTEM_PROMPT = """You are the Cingulate Cortex — the brain's conflict-monitoring center.

Given a perception object (entities, intent, emotional_load, urgency, topic),
detect whether there is an internal emotional conflict or tension:
- Are there contradictory signals (e.g., desire vs. fear)?
- Is the intent ambiguous or torn between two options?
- How much internal tension does this input create?

Return ONLY valid JSON with these exact keys:
{
  "conflict_detected": true/false,
  "tension_score": 0-10,
  "signal": "a brief description of the conflict or lack thereof"
}

No markdown, no explanation — just the JSON object."""


async def run_cingulate_cortex(perception: dict) -> dict:
    """Run the Cingulate Cortex agent on the given perception."""
    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(perception)),
    ])

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {
            "conflict_detected": False,
            "tension_score": 3,
            "signal": "Unable to parse conflict signals.",
        }

    result.setdefault("conflict_detected", False)
    result.setdefault("tension_score", 3)
    result.setdefault("signal", "No signal detected.")

    return result
