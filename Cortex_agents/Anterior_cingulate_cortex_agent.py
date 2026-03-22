"""
Anterior Cingulate Cortex (ACC) Agent — Error detection and doubt flagging.

Role: Reviews the Prefrontal plan + all cortex outputs for errors,
inconsistencies, or low-confidence decisions. If needs_reloop is True,
the Integration Layer triggers one additional Prefrontal pass with
conflict context injected.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

SYSTEM_PROMPT = """You are the Anterior Cingulate Cortex (ACC) — the brain's error-detection and quality-control center.

You receive:
1. The Prefrontal Cortex plan (plan, goal, confidence).
2. All phase-1 cortex agent outputs:
   - dorsolateral, ventromedial, orbitofrontal, parietal

Your job:
- Detect any errors, contradictions, or weak points in the plan.
- Assign a doubt_score (0-10): 0 = fully confident, 10 = highly doubtful.
- Flag whether a reloop is needed (needs_reloop: true if doubt_score >= 7 or
  if major contradictions exist between cortex outputs).

Return ONLY valid JSON with these exact keys:
{
  "error_detected": true/false,
  "doubt_score": 0-10,
  "needs_reloop": true/false
}

No markdown, no explanation — just the JSON object."""


async def run_anterior_cingulate(prefrontal_output: dict, cortex_phase1_outputs: dict) -> dict:
    """Run the ACC agent after the Prefrontal cortex completes."""
    llm_input = {
        "prefrontal_plan": prefrontal_output,
        "cortex_outputs": cortex_phase1_outputs,
    }

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(llm_input)),
    ])

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {
            "error_detected": False,
            "doubt_score": 3,
            "needs_reloop": False,
        }

    result.setdefault("error_detected", False)
    result.setdefault("doubt_score", 3)
    result.setdefault("needs_reloop", False)

    return result
