"""
Hypothalamus Agent — Internal state and resource pressure assessment.

Role: Evaluates the perception in the context of the system's internal
fuel state (compute budget, context usage). Produces an urgency drive,
discomfort signal, and fuel penalty that modulates integration weights.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

SYSTEM_PROMPT = """You are the Hypothalamus — the brain's internal-state regulator.

You receive:
1. A perception object (entities, intent, emotional_load, urgency, topic).
2. A fuel_state dict with "compute_budget" (normal/low) and "context_used_pct" (0-100).

Assess:
- How urgently the system's internal state demands a response (urgency_drive 0-10).
- Describe any discomfort or pressure (discomfort_signal: short sentence).
- Compute fuel_penalty: 0.0 if compute_budget is "normal", 0.4 if anything else.

Return ONLY valid JSON with these exact keys:
{
  "urgency_drive": 0-10,
  "discomfort_signal": "...",
  "fuel_penalty": 0.0 or 0.4
}

No markdown, no explanation — just the JSON object."""


async def run_hypothalamus(perception: dict, fuel_state: dict) -> dict:
    """Run the Hypothalamus agent with perception and fuel_state."""
    llm_input = {
        "perception": perception,
        "fuel_state": fuel_state,
    }

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(llm_input)),
    ])

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        # Deterministic fallback based on fuel_state
        penalty = 0.0 if fuel_state.get("compute_budget") == "normal" else 0.4
        result = {
            "urgency_drive": perception.get("urgency", 5),
            "discomfort_signal": "Fallback: unable to parse hypothalamus output.",
            "fuel_penalty": penalty,
        }

    result.setdefault("urgency_drive", 5)
    result.setdefault("discomfort_signal", "No discomfort detected.")

    # Enforce deterministic fuel_penalty rule
    if fuel_state.get("compute_budget") == "normal":
        result["fuel_penalty"] = 0.0
    else:
        result["fuel_penalty"] = 0.4

    return result
