"""
Nucleus Accumbens Agent — Reward potential evaluation.

Role: Evaluates how rewarding it would be to engage with or respond
to the current perception. Produces a reward score and a motivation
direction (approach / avoid / neutral).
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

SYSTEM_PROMPT = """You are the Nucleus Accumbens — the brain's reward-evaluation center.

Given a perception object (entities, intent, emotional_load, urgency, topic),
evaluate the reward potential of engaging with this input:
- How rewarding or satisfying would it be to respond? (reward_score 0-10)
- What is the motivational direction? One of:
    "approach" — high reward, worth engaging
    "avoid"    — low or negative reward, better to disengage
    "neutral"  — ambiguous reward signal

Return ONLY valid JSON with these exact keys:
{
  "reward_score": 0-10,
  "motivation": "approach" | "avoid" | "neutral"
}

No markdown, no explanation — just the JSON object."""


async def run_nucleus_accumbens(perception: dict) -> dict:
    """Run the Nucleus Accumbens agent on the given perception."""
    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(perception)),
    ])

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {
            "reward_score": 5,
            "motivation": "neutral",
        }

    result.setdefault("reward_score", 5)
    result.setdefault("motivation", "neutral")

    return result
