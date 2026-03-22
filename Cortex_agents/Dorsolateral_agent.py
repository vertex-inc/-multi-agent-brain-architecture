"""
Dorsolateral Agent — Working memory and step-by-step reasoning.

Role: Takes the perception and relevant hippocampal memories,
performs structured step-by-step reasoning, and produces a
working memory summary.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

SYSTEM_PROMPT = """You are the Dorsolateral Prefrontal Cortex — the brain's working-memory and reasoning engine.

You receive:
1. A perception object (entities, intent, emotional_load, urgency, topic).
2. A list of relevant past memories retrieved by the Hippocampus.

Your job:
- Break down the problem into clear, sequential reasoning steps.
- Summarize what is currently being held in working memory.

Return ONLY valid JSON with these exact keys:
{
  "reasoning_steps": ["step 1 text", "step 2 text", ...],
  "working_memory_summary": "a concise summary of key facts being tracked"
}

No markdown, no explanation — just the JSON object."""


async def run_dorsolateral(perception: dict, hippocampus_memories: list) -> dict:
    """Run the Dorsolateral agent with perception and hippocampus memories."""
    llm_input = {
        "perception": perception,
        "hippocampus_memories": hippocampus_memories,
    }

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(llm_input)),
    ])

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {
            "reasoning_steps": [f"Analyzing input about: {perception.get('topic', 'unknown')}"],
            "working_memory_summary": "Unable to parse reasoning output.",
        }

    result.setdefault("reasoning_steps", [])
    result.setdefault("working_memory_summary", "")

    return result
