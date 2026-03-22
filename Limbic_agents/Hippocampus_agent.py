"""
Hippocampus Agent — Memory retrieval and storage using Chroma vector store.

Role: Embeds the current perception, retrieves the top-3 most similar
past episodes from an in-memory Chroma vector store, and after each
session writes a summary back to the store.
"""

import json
import uuid
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
embeddings = OpenAIEmbeddings()

# ── In-memory Chroma vector store (persistent across the session) ────
vector_store = Chroma(
    collection_name="hippocampus_episodes",
    embedding_function=embeddings,
)

SYSTEM_PROMPT = """You are the Hippocampus — the brain's memory center.

You will receive:
1. A perception object (entities, intent, emotional_load, urgency, topic).
2. A list of retrieved past memories (may be empty on first run).

Your job:
- Evaluate how familiar the current perception is compared to past memories.
- Provide a familiarity_score (0-10): 0 = completely novel, 10 = exact match.
- Generate a one-sentence summary of the current episode for future storage.

Return ONLY valid JSON with these exact keys:
{
  "memories": ["list of relevant past memory texts"],
  "familiarity_score": 0-10,
  "episode_ids": ["list of episode IDs from retrieved memories"],
  "episode_summary": "one-sentence summary of the current episode"
}

No markdown, no explanation — just the JSON object."""


async def run_hippocampus(perception: dict) -> dict:
    """Run the Hippocampus agent: retrieve memories, evaluate, then store."""
    # ── Retrieve top-3 similar past episodes ──────────────────────────
    query_text = json.dumps(perception)
    retrieved_memories = []
    episode_ids = []

    try:
        results = vector_store.similarity_search(query_text, k=3)
        for doc in results:
            retrieved_memories.append(doc.page_content)
            episode_ids.append(doc.metadata.get("episode_id", "unknown"))
    except Exception:
        # Store might be empty on first run
        pass

    # ── Build input for the LLM ───────────────────────────────────────
    llm_input = {
        "perception": perception,
        "retrieved_memories": retrieved_memories,
    }

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(llm_input)),
    ])

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {
            "memories": retrieved_memories,
            "familiarity_score": 0,
            "episode_ids": episode_ids,
            "episode_summary": f"Perception about {perception.get('topic', 'unknown')}",
        }

    result.setdefault("memories", retrieved_memories)
    result.setdefault("familiarity_score", 0)
    result.setdefault("episode_ids", episode_ids)
    result.setdefault("episode_summary", "")

    # ── Store episode summary back into vector store ──────────────────
    episode_summary = result.get("episode_summary", "")
    if episode_summary:
        new_id = str(uuid.uuid4())
        vector_store.add_texts(
            texts=[episode_summary],
            metadatas=[{"episode_id": new_id, "topic": perception.get("topic", "")}],
        )
        # Add the new episode ID to the output
        result["episode_ids"].append(new_id)

    # Remove the intermediate field from the output
    result.pop("episode_summary", None)

    return result
