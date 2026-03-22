"""
Integration Layer — Receives all limbic + cortex outputs, computes dominance,
handles ACC reloop logic, and produces the final structured decision object.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)


async def _generate_response_text(perception: dict, decision_context: dict) -> str:
    """Use LLM to generate a natural-language response based on the final decision."""
    system_prompt = (
        "You are the brain's final output synthesizer. Given the perception "
        "and the internal decision context (dominant system, reasoning chain, "
        "emotional tone, urgency), produce a clear, helpful, natural-language "
        "response to the user. Be concise and direct. "
        "Return ONLY the response text, no JSON."
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps({
            "perception": perception,
            "decision_context": decision_context,
        })),
    ])

    return response.content.strip()


async def integrate(
    perception: dict,
    limbic_outputs: dict,
    cortex_outputs: dict,
    run_prefrontal_fn=None,
    hippocampus_memories: list = None,
) -> dict:
    """
    Main integration function.

    Computes limbic vs. cortical dominance, handles ACC reloop if needed,
    and produces the final decision object.
    """

    # ── Extract key values from agent outputs ─────────────────────────
    amygdala = limbic_outputs.get("amygdala", {})
    cingulate = limbic_outputs.get("cingulate", {})
    hippocampus = limbic_outputs.get("hippocampus", {})
    hypothalamus = limbic_outputs.get("hypothalamus", {})
    nucleus_accumbens = limbic_outputs.get("nucleus_accumbens", {})

    prefrontal = cortex_outputs.get("prefrontal", {})
    acc = cortex_outputs.get("acc", {})
    dorsolateral = cortex_outputs.get("dorsolateral", {})

    # ── ACC reloop logic ──────────────────────────────────────────────
    if (
        cingulate.get("conflict_detected", False)
        and acc.get("needs_reloop", False)
        and run_prefrontal_fn is not None
    ):
        print("🔄 ACC triggered reloop — running Prefrontal again with conflict context")

        conflict_context = (
            f"Conflict detected by Cingulate Cortex: {cingulate.get('signal', 'unknown')}. "
            f"ACC doubt score: {acc.get('doubt_score', 0)}. "
            f"Please revise the plan accounting for this internal conflict."
        )

        # Extract cortex phase-1 outputs (exclude prefrontal and acc)
        cortex_phase1 = {
            k: v for k, v in cortex_outputs.items()
            if k not in ("prefrontal", "acc")
        }

        prefrontal = await run_prefrontal_fn(
            perception,
            cortex_phase1,
            hippocampus_memories or [],
            conflict_context=conflict_context,
        )

        # Update the cortex outputs with the new prefrontal result
        cortex_outputs["prefrontal"] = prefrontal
        print("✅ Prefrontal reloop completed")

    # ── Compute dominance weights ─────────────────────────────────────
    urgency_score = float(amygdala.get("urgency_score", 5))
    emotional_salience = float(amygdala.get("emotional_salience", 5))
    fuel_penalty = float(hypothalamus.get("fuel_penalty", 0.0))
    confidence = float(prefrontal.get("confidence", 0.5))
    error_detected = acc.get("error_detected", False)

    # limbic_weight = amygdala.urgency_score * hypothalamus.fuel_penalty * amygdala.emotional_salience
    limbic_weight = urgency_score * fuel_penalty * emotional_salience

    # cortical_weight = prefrontal.confidence * (1 - fuel_penalty) * (1 if not acc.error_detected else 0.5)
    error_factor = 0.5 if error_detected else 1.0
    cortical_weight = confidence * (1.0 - fuel_penalty) * error_factor

    # ── Determine dominant system ─────────────────────────────────────
    if limbic_weight > cortical_weight * 1.5:
        dominant = "limbic"
    else:
        dominant = "cortical"

    # ── Map amygdala flag to emotional tone ────────────────────────────
    flag = amygdala.get("flag", "safe")
    emotional_tone_map = {
        "fear": "fearful / alarmed",
        "caution": "cautious / watchful",
        "safe": "calm / confident",
    }
    emotional_tone = emotional_tone_map.get(flag, "neutral")

    # ── Collect flags ─────────────────────────────────────────────────
    flags = []
    if acc.get("error_detected", False):
        flags.append(f"ACC: error detected (doubt={acc.get('doubt_score', '?')})")
    if acc.get("needs_reloop", False):
        flags.append("ACC: reloop was triggered")
    if cingulate.get("conflict_detected", False):
        flags.append(f"Cingulate: conflict ({cingulate.get('signal', '?')})")
    if amygdala.get("threat_detected", False):
        flags.append(f"Amygdala: threat detected (flag={flag})")

    # ── Build decision context for response generation ────────────────
    decision_context = {
        "dominant_system": dominant,
        "confidence": round(confidence, 2),
        "emotional_tone": emotional_tone,
        "urgency_level": int(urgency_score),
        "reasoning_chain": dorsolateral.get("reasoning_steps", []),
        "plan": prefrontal.get("plan", []),
        "goal": prefrontal.get("goal", ""),
        "memory_references": hippocampus.get("episode_ids", []),
        "flags": flags,
    }

    # ── Generate final response text ──────────────────────────────────
    response_text = await _generate_response_text(perception, decision_context)

    # ── Final decision object ─────────────────────────────────────────
    decision = {
        "dominant_system": dominant,
        "confidence": round(confidence, 2),
        "emotional_tone": emotional_tone,
        "urgency_level": int(urgency_score),
        "reasoning_chain": dorsolateral.get("reasoning_steps", []),
        "memory_references": hippocampus.get("episode_ids", []),
        "response_text": response_text,
        "flags": flags,
        # Internal weights for debug panel
        "_limbic_weight": round(limbic_weight, 3),
        "_cortical_weight": round(cortical_weight, 3),
    }

    return decision
