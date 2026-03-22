"""
Prefrontal Cortex Agent — Goal-directed planning.

Role: Synthesizes ALL cortex phase-1 outputs plus hippocampus memories
into a coherent plan. This agent runs AFTER Dorsolateral, Ventromedial,
Orbitofrontal, and Parietal complete.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

SYSTEM_PROMPT = """You are the Prefrontal Cortex — the brain's executive planner.

You receive:
1. A perception object (entities, intent, emotional_load, urgency, topic).
2. Outputs from all phase-1 cortex agents:
   - dorsolateral: reasoning_steps, working_memory_summary
   - ventromedial: preferred_option, value_judgment
   - orbitofrontal: consequences, adjusted_recommendation
   - parietal: estimates, logical_comparison
3. Relevant memories from the Hippocampus.
4. Optionally, conflict context from a previous ACC reloop.

Your job:
- Synthesize all inputs into a coherent, goal-directed plan.
- State the primary goal clearly.
- Assign a confidence level (0.0 to 1.0) to the plan.

Return ONLY valid JSON with these exact keys:
{
  "plan": ["step 1", "step 2", ...],
  "goal": "the primary objective",
  "confidence": 0.0-1.0
}

No markdown, no explanation — just the JSON object."""


async def run_prefrontal(
    perception: dict,
    cortex_phase1_outputs: dict,
    hippocampus_memories: list,
    conflict_context: str = None,
) -> dict:
    """Run the Prefrontal Cortex agent after phase-1 cortex agents."""
    llm_input = {
        "perception": perception,
        "cortex_outputs": cortex_phase1_outputs,
        "hippocampus_memories": hippocampus_memories,
    }

    if conflict_context:
        llm_input["conflict_context"] = conflict_context

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(llm_input)),
    ])

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {
            "plan": ["Unable to generate plan — defaulting to cautious approach."],
            "goal": perception.get("intent", "unknown"),
            "confidence": 0.5,
        }

    result.setdefault("plan", [])
    result.setdefault("goal", "unknown")
    result.setdefault("confidence", 0.5)

    # Clamp confidence to [0.0, 1.0]
    result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))

    return result
