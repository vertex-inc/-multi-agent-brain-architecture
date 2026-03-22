"""
Ventromedial Agent — Value comparison between options.

Role: Uses the perception and the Nucleus Accumbens reward score
to compare options and make value-based judgments about the best
course of action.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

SYSTEM_PROMPT = """You are the Ventromedial Prefrontal Cortex — the brain's value-comparison center.

You receive:
1. A perception object (entities, intent, emotional_load, urgency, topic).
2. A reward_score (0-10) from the Nucleus Accumbens indicating how rewarding engagement is.

Your job:
- Identify possible options or courses of action.
- Compare their subjective values considering the reward signal.
- Recommend the preferred option with a value judgment.

Return ONLY valid JSON with these exact keys:
{
  "preferred_option": "description of the best course of action",
  "value_judgment": "reasoning about why this option has highest value"
}

No markdown, no explanation — just the JSON object."""


async def run_ventromedial(perception: dict, reward_score: int) -> dict:
    """Run the Ventromedial agent with perception and reward_score."""
    llm_input = {
        "perception": perception,
        "reward_score": reward_score,
    }

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(llm_input)),
    ])

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {
            "preferred_option": "Insufficient data to compare options.",
            "value_judgment": "Unable to parse value comparison output.",
        }

    result.setdefault("preferred_option", "No preference determined.")
    result.setdefault("value_judgment", "No value judgment available.")

    return result
