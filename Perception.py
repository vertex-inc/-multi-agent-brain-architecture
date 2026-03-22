"""
Perception.py — Entry point for the Brain Simulation System.

Accepts raw user text input, runs a lightweight LLM call to extract
structured perception data, then broadcasts perception + fuel_state
to all agents in parallel via asyncio.
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Add project root to path so agent packages are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Limbic_agents.amygdala_agent import run_amygdala
from Limbic_agents.Cingulate_cortex_agent import run_cingulate_cortex
from Limbic_agents.Hippocampus_agent import run_hippocampus
from Limbic_agents.Hypothalamus_agent import run_hypothalamus
from Limbic_agents.Nucleus_accumbens_agent import run_nucleus_accumbens

from Cortex_agents.Dorsolateral_agent import run_dorsolateral
from Cortex_agents.Ventromedial_agent import run_ventromedial
from Cortex_agents.Orbitofrontal_cortex_agent import run_orbitofrontal
from Cortex_agents.Parietal_cortex_agent import run_parietal
from Cortex_agents.Prefrontal_cortex_agent import run_prefrontal
from Cortex_agents.Anterior_cingulate_cortex_agent import run_anterior_cingulate

from Integration_layer import integrate
from Decision_output import display_decision

load_dotenv()

# ── Shared LLM instance ──────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# ── Simulated fuel state ─────────────────────────────────────────────
FUEL_STATE = {
    "compute_budget": "normal",
    "context_used_pct": 45,
}


async def extract_perception(user_input: str) -> dict:
    """Use a lightweight LLM call to extract structured perception data."""

    system_prompt = (
        "You are a perception-extraction module. Given raw user text, "
        "return ONLY valid JSON with these keys:\n"
        '  "entities": list of key nouns/objects mentioned,\n'
        '  "intent": a short phrase describing the user\'s goal,\n'
        '  "emotional_load": integer 0-10 (how emotionally charged the input is),\n'
        '  "urgency": integer 0-10 (how time-sensitive the input is),\n'
        '  "topic": a single-word or short-phrase topic label.\n'
        "Return ONLY the JSON object, no markdown, no explanation."
    )

    response = await llm.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]
    )

    try:
        perception = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback: build a default perception if LLM output isn't valid JSON
        perception = {
            "entities": [],
            "intent": "unknown",
            "emotional_load": 5,
            "urgency": 5,
            "topic": "general",
        }

    # Ensure all required keys exist
    perception.setdefault("entities", [])
    perception.setdefault("intent", "unknown")
    perception.setdefault("emotional_load", 5)
    perception.setdefault("urgency", 5)
    perception.setdefault("topic", "general")

    return perception


async def run_brain(user_input: str) -> dict:
    """
    Full brain pipeline:
    1. Extract perception
    2. Run limbic agents in parallel
    3. Run cortex agents (phase 1 parallel, phase 2 prefrontal, phase 3 ACC)
    4. Integrate all outputs
    5. Display decision
    """

    # ── Step 1: Perception extraction ─────────────────────────────────
    perception = await extract_perception(user_input)
    print(f"\n🧠 Perception extracted: {json.dumps(perception, indent=2)}")

    fuel_state = FUEL_STATE.copy()

    # ── Step 2: Run ALL limbic agents in parallel ─────────────────────
    (
        amygdala_out,
        cingulate_out,
        hippocampus_out,
        hypothalamus_out,
        nucleus_accumbens_out,
    ) = await asyncio.gather(
        run_amygdala(perception),
        run_cingulate_cortex(perception),
        run_hippocampus(perception),
        run_hypothalamus(perception, fuel_state),
        run_nucleus_accumbens(perception),
    )

    print("✅ Limbic agents completed")

    # ── Step 3a: Run cortex phase-1 agents in parallel ────────────────
    # These receive cross-group data from limbic outputs
    (
        dorsolateral_out,
        ventromedial_out,
        orbitofrontal_out,
        parietal_out,
    ) = await asyncio.gather(
        run_dorsolateral(perception, hippocampus_out.get("memories", [])),
        run_ventromedial(perception, nucleus_accumbens_out.get("reward_score", 5)),
        run_orbitofrontal(perception, amygdala_out.get("urgency_score", 5)),
        run_parietal(perception),
    )

    print("✅ Cortex phase-1 agents completed")

    # ── Step 3b: Prefrontal runs AFTER phase-1 ────────────────────────
    all_cortex_phase1 = {
        "dorsolateral": dorsolateral_out,
        "ventromedial": ventromedial_out,
        "orbitofrontal": orbitofrontal_out,
        "parietal": parietal_out,
    }

    prefrontal_out = await run_prefrontal(
        perception,
        all_cortex_phase1,
        hippocampus_out.get("memories", []),
    )

    print("✅ Prefrontal cortex completed")

    # ── Step 3c: ACC runs AFTER Prefrontal ────────────────────────────
    acc_out = await run_anterior_cingulate(prefrontal_out, all_cortex_phase1)

    print("✅ Anterior Cingulate Cortex completed")

    # ── Step 4: Integration ───────────────────────────────────────────
    all_limbic = {
        "amygdala": amygdala_out,
        "cingulate": cingulate_out,
        "hippocampus": hippocampus_out,
        "hypothalamus": hypothalamus_out,
        "nucleus_accumbens": nucleus_accumbens_out,
    }

    all_cortex = {
        **all_cortex_phase1,
        "prefrontal": prefrontal_out,
        "acc": acc_out,
    }

    decision = await integrate(
        perception=perception,
        limbic_outputs=all_limbic,
        cortex_outputs=all_cortex,
        run_prefrontal_fn=run_prefrontal,
        hippocampus_memories=hippocampus_out.get("memories", []),
    )

    # ── Step 5: Display ───────────────────────────────────────────────
    display_decision(decision, all_limbic, all_cortex)

    return decision


def main():
    """Interactive input loop."""
    print("=" * 60)
    print("  🧬  MULTI-AGENT BRAIN SIMULATION SYSTEM  🧬")
    print("=" * 60)
    print("Type your thoughts / questions below. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("🗣️  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        asyncio.run(run_brain(user_input))
        print("\n" + "─" * 60 + "\n")


if __name__ == "__main__":
    main()
