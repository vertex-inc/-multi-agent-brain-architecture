"""
Orbitofrontal Cortex Agent — Consequence evaluation adjusted by threat level.

Role: Uses the perception and the Amygdala's urgency score to evaluate
potential consequences of different actions, adjusting recommendations
based on the current threat level.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

SYSTEM_PROMPT = """You are the Orbitofrontal Cortex — the brain's consequence-evaluation center.

You receive:
1. A perception object (entities, intent, emotional_load, urgency, topic).
2. An urgency_score (0-10) from the Amygdala reflecting the threat level.

Your job:
- Enumerate the likely consequences of acting on this perception.
- Adjust your recommendation based on the threat/urgency level:
    - High urgency → prioritize safety and caution
    - Low urgency  → prioritize optimal outcomes

Return ONLY valid JSON with these exact keys:
{
  "consequences": ["consequence 1", "consequence 2", ...],
  "adjusted_recommendation": "your recommended action given the threat level"
}

No markdown, no explanation — just the JSON object."""


async def run_orbitofrontal(perception: dict, amygdala_urgency: int) -> dict:
    """Run the Orbitofrontal agent with perception and amygdala urgency score."""
    llm_input = {
        "perception": perception,
        "amygdala_urgency_score": amygdala_urgency,
    }

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(llm_input)),
    ])

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {
            "consequences": ["Unable to evaluate consequences."],
            "adjusted_recommendation": "Proceed with caution.",
        }

    result.setdefault("consequences", [])
    result.setdefault("adjusted_recommendation", "No recommendation available.")

    return result
