"""
Parietal Cortex Agent — Numerical and logical estimation.

Role: Performs quantitative analysis, numerical estimates, and
logical comparisons relevant to the perception input.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

SYSTEM_PROMPT = """You are the Parietal Cortex — the brain's numerical and logical estimation center.

Given a perception object (entities, intent, emotional_load, urgency, topic), perform:
1. Numerical estimates — any relevant quantities, probabilities, or magnitudes.
2. Logical comparisons — weigh pros vs. cons, compare alternatives logically.

Return ONLY valid JSON with these exact keys:
{
  "estimates": ["estimate 1", "estimate 2", ...],
  "logical_comparison": "a structured comparison or logical analysis"
}

No markdown, no explanation — just the JSON object."""


async def run_parietal(perception: dict) -> dict:
    """Run the Parietal Cortex agent on the given perception."""
    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(perception)),
    ])

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {
            "estimates": ["Unable to generate numerical estimates."],
            "logical_comparison": "Unable to parse logical analysis.",
        }

    result.setdefault("estimates", [])
    result.setdefault("logical_comparison", "No comparison available.")

    return result
