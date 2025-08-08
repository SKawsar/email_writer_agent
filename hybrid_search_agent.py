"""
Hybrid Search Agent using OpenAI Agent SDK, ChromaDB, and OpenAI built‑in web_search tool.

Requirements:
    pip install openai chromadb python-dotenv

Environment variables required:
    OPENAI_API_KEY – your OpenAI key

Internal index location:
    A persistent ChromaDB database will be created at ./internal_index

Run:
    python hybrid_search_agent.py

The script exposes one helper function `chat(query)` that automatically decides
when to call `internal_search` (Chroma) or `web_search` (OpenAI built‑in), or both,
 to answer the user’s question. Replace the Chroma ingestion example at the bottom
with your own documents.
"""

import os
import json
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment & keys
# ---------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

client = OpenAI()

# ---------------------------------------------------------------------------
# ChromaDB setup
# ---------------------------------------------------------------------------
chroma_client = chromadb.PersistentClient(
    path="internal_index",
    settings=Settings(allow_reset=True),
)
collection = chroma_client.get_or_create_collection("internal_docs")


def ingest_documents(docs: List[str]) -> None:
    """Add raw text documents to the Chroma collection."""
    import uuid
    collection.add(
        documents=docs,
        ids=[str(uuid.uuid4()) for _ in docs],
        metadatas=[{} for _ in docs],
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def internal_search_tool(query: str, top_k: int = 5) -> List[Dict]:
    """Search internal ChromaDB for relevant documents."""
    res = collection.query(query_texts=[query], n_results=top_k)
    hits = []
    for rank, (doc_id, doc_text, score) in enumerate(
        zip(res["ids"][0], res["documents"][0], res["distances"][0]), start=1
    ):
        hits.append({"rank": rank, "id": doc_id, "text": doc_text, "score": score})
    return hits


# ---------------------------------------------------------------------------
# OpenAI function‑tool specifications
# ---------------------------------------------------------------------------
internal_search_spec = {
    "name": "internal_search",
    "description": "Searches the internal ChromaDB vector store for relevant documents.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The user search query."},
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

# Built‑in web_search tool – no schema needed beyond type declaration
web_search_spec = {"type": "web_search"}


def dispatch_tool_call(call) -> List[Dict]:
    """Executes Python-side tools. Built‑in tools are handled by OpenAI."""
    if call["name"] == "internal_search":
        return internal_search_tool(**call.get("arguments", {}))
    # Any other tool (e.g., web_search) is handled by the platform
    return []


# ---------------------------------------------------------------------------
# Chat loop with automatic tool use
# ---------------------------------------------------------------------------

def chat(query: str, history: List[Dict] | None = None, model: str = "gpt-4o-mini") -> str:
    """Chat with the assistant. It will decide when to call tools automatically."""
    history = history or []
    history.append({"role": "user", "content": query})

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=history,
            tools=[
                {"type": "function", "function": internal_search_spec},
                web_search_spec,
            ],
            tool_choice="auto",
            temperature=0.3,
        )

        msg = response.choices[0].message

        # Assistant wants to call a tool
        if msg.tool_calls:
            history.append(msg)  # keep the assistant's tool_call message
            for call in msg.tool_calls:
                # only dispatch internal_search locally
                if call.name == "internal_search":
                    result = dispatch_tool_call(call)
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
            # Loop again so the model can incorporate tool results
            continue

        # Tool message returned from OpenAI (e.g., for web_search)
        if msg.role == "tool":
            history.append(msg)
            continue

        # Final assistant answer
        history.append({"role": "assistant", "content": msg.content})
        return msg.content


# ---------------------------------------------------------------------------
# Example ingestion & REPL
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Index sample docs if collection is empty
    if collection.count() == 0:
        sample_docs = [
            "OpenAI released GPT-4o in 2025, offering multimodal capabilities.",
            "IDARE AI's AutoML platform simplifies model training on AWS S3 data.",
        ]
        ingest_documents(sample_docs)
        print("[+] Sample documents ingested into ChromaDB for demo…")

    print("\nHybrid Search Agent – type your question (Ctrl-C to exit):\n")
    try:
        while True:
            user_q = input("> ").strip()
            if not user_q:
                continue
            answer = chat(user_q)
            print("\nAssistant:\n", answer, "\n")
    except KeyboardInterrupt:
        print("\nGoodbye!")
