"""
Decision Output — Formatted terminal display of the brain's decision.

Prints the final response text along with a structured debug panel
showing agent scores, dominant system, and flags.
"""


def display_decision(decision: dict, limbic_outputs: dict, cortex_outputs: dict):
    """Print the decision and debug panel to the terminal."""

    # ══════════════════════════════════════════════════════════════════
    # RESPONSE
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  🧬  BRAIN RESPONSE")
    print("═" * 60)
    print()
    print(decision.get("response_text", "(no response generated)"))
    print()

    # ══════════════════════════════════════════════════════════════════
    # DEBUG PANEL
    # ══════════════════════════════════════════════════════════════════
    print("┌" + "─" * 58 + "┐")
    print("│  📊  DEBUG PANEL" + " " * 40 + "│")
    print("├" + "─" * 58 + "┤")

    dominant = decision.get("dominant_system", "unknown")
    confidence = decision.get("confidence", 0.0)
    urgency = decision.get("urgency_level", 0)
    emotional_tone = decision.get("emotional_tone", "unknown")

    _print_row("Dominant System", dominant.upper())
    _print_row("Confidence", f"{confidence:.2f}")
    _print_row("Urgency Level", f"{urgency}/10")
    _print_row("Emotional Tone", emotional_tone)
    _print_row("Limbic Weight", f"{decision.get('_limbic_weight', 0):.3f}")
    _print_row("Cortical Weight", f"{decision.get('_cortical_weight', 0):.3f}")

    # ── Flags ─────────────────────────────────────────────────────────
    flags = decision.get("flags", [])
    if flags:
        print("├" + "─" * 58 + "┤")
        _print_row("Flags", "")
        for flag in flags:
            print(f"│    ⚠️  {flag:<49}│")
    else:
        print("├" + "─" * 58 + "┤")
        _print_row("Flags", "None")

    # ── Agent Scores ──────────────────────────────────────────────────
    print("├" + "─" * 58 + "┤")
    print("│  🔬  AGENT SCORES" + " " * 39 + "│")
    print("├" + "─" * 58 + "┤")

    # Limbic agents
    amygdala = limbic_outputs.get("amygdala", {})
    cingulate = limbic_outputs.get("cingulate", {})
    hippocampus = limbic_outputs.get("hippocampus", {})
    hypothalamus = limbic_outputs.get("hypothalamus", {})
    nucleus = limbic_outputs.get("nucleus_accumbens", {})

    _print_section("LIMBIC SYSTEM")
    _print_row("  Amygdala — Urgency", f"{amygdala.get('urgency_score', '?')}/10")
    _print_row("  Amygdala — Salience", f"{amygdala.get('emotional_salience', '?')}/10")
    _print_row("  Amygdala — Flag", f"{amygdala.get('flag', '?')}")
    _print_row("  Cingulate — Tension", f"{cingulate.get('tension_score', '?')}/10")
    _print_row("  Cingulate — Conflict", f"{cingulate.get('conflict_detected', '?')}")
    _print_row("  Hippocampus — Familiarity", f"{hippocampus.get('familiarity_score', '?')}/10")
    _print_row("  Hypothalamus — Drive", f"{hypothalamus.get('urgency_drive', '?')}/10")
    _print_row("  Hypothalamus — Fuel Penalty", f"{hypothalamus.get('fuel_penalty', '?')}")
    _print_row("  Nucleus Acc — Reward", f"{nucleus.get('reward_score', '?')}/10")
    _print_row("  Nucleus Acc — Motivation", f"{nucleus.get('motivation', '?')}")

    # Cortex agents
    dorsolateral = cortex_outputs.get("dorsolateral", {})
    ventromedial = cortex_outputs.get("ventromedial", {})
    orbitofrontal = cortex_outputs.get("orbitofrontal", {})
    parietal = cortex_outputs.get("parietal", {})
    prefrontal = cortex_outputs.get("prefrontal", {})
    acc = cortex_outputs.get("acc", {})

    _print_section("CORTEX SYSTEM")
    _print_row("  Dorsolateral — Steps", f"{len(dorsolateral.get('reasoning_steps', []))} steps")
    _print_row("  Ventromedial — Option", _truncate(ventromedial.get('preferred_option', '?'), 30))
    _print_row("  Orbitofrontal — Conseq.", f"{len(orbitofrontal.get('consequences', []))} items")
    _print_row("  Parietal — Estimates", f"{len(parietal.get('estimates', []))} items")
    _print_row("  Prefrontal — Confidence", f"{prefrontal.get('confidence', '?')}")
    _print_row("  Prefrontal — Goal", _truncate(prefrontal.get('goal', '?'), 30))
    _print_row("  ACC — Doubt Score", f"{acc.get('doubt_score', '?')}/10")
    _print_row("  ACC — Error Detected", f"{acc.get('error_detected', '?')}")
    _print_row("  ACC — Reloop Needed", f"{acc.get('needs_reloop', '?')}")

    # ── Memory References ─────────────────────────────────────────────
    memory_refs = decision.get("memory_references", [])
    if memory_refs:
        print("├" + "─" * 58 + "┤")
        _print_row("Memory References", f"{len(memory_refs)} episodes")

    # ── Reasoning Chain ───────────────────────────────────────────────
    reasoning = decision.get("reasoning_chain", [])
    if reasoning:
        print("├" + "─" * 58 + "┤")
        _print_section("REASONING CHAIN")
        for i, step in enumerate(reasoning, 1):
            text = _truncate(step, 48)
            print(f"│    {i}. {text:<52}│")

    print("└" + "─" * 58 + "┘")
    print()


def _print_row(label: str, value: str):
    """Print a formatted row in the debug panel."""
    content = f"  {label}: {value}"
    print(f"│{content:<58}│")


def _print_section(title: str):
    """Print a section header."""
    print(f"│  ── {title} {'─' * (50 - len(title))}│")


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len with ellipsis."""
    text = str(text)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text
