"""
Embedder — wraps OpenAI text-embedding-3-small.
Caches embeddings to disk so re-runs don't rebill.
Also expands each research interest into synonyms via a single LLM call.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

from src.profile_parser import StudentProfile
from src.llm_client import LLMClient

log = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/embeddings")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


async def expand_synonyms(interests: list[str], client: AsyncOpenAI | LLMClient) -> dict[str, list[str]]:
    """
    Single LLM call to expand each research interest into 3 alternative phrasings
    used as additional API query terms.
    """
    if not isinstance(client, LLMClient):
        client = LLMClient(client)
    prompt = (
        "For each research area below, give exactly 3 alternative academic phrasings or "
        "closely related sub-fields that would appear in paper/grant titles. "
        "Return ONLY a JSON object mapping each original term to a list of 3 strings.\n\n"
        f"Research areas: {json.dumps(interests)}"
    )
    try:
        return await client.chat_complete_json(prompt)
    except Exception as e:
        log.warning(f"Synonym expansion failed: {e} — using empty synonyms")
        return {i: [] for i in interests}


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


async def embed_texts(texts: list[str], client: AsyncOpenAI | LLMClient) -> list[list[float]]:
    """Embed a list of strings, using disk cache for each."""
    if not isinstance(client, LLMClient):
        client = LLMClient(client)
    results: list[list[float] | None] = [None] * len(texts)
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for i, text in enumerate(texts):
        cache_file = CACHE_DIR / f"{_cache_key(text)}.json"
        if cache_file.exists():
            results[i] = json.loads(cache_file.read_text())
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        try:
            vecs = await client.get_embedding(uncached_texts)
            for idx, vec in zip(uncached_indices, vecs):
                results[idx] = vec
                cache_file = CACHE_DIR / f"{_cache_key(texts[idx])}.json"
                cache_file.write_text(json.dumps(vec))
        except Exception as e:
            log.warning(f"Embedding failed: {e}. Fallback to mock.")
            fallback_client = LLMClient(None)
            vecs = await fallback_client.get_embedding(uncached_texts)
            for idx, vec in zip(uncached_indices, vecs):
                results[idx] = vec
                cache_file = CACHE_DIR / f"{_cache_key(texts[idx])}.json"
                cache_file.write_text(json.dumps(vec))

    return results  # type: ignore[return-value]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


async def embed_profile(profile: StudentProfile, client: AsyncOpenAI | LLMClient) -> StudentProfile:
    """
    Mutates profile in-place:
    - Expands interests to synonyms
    - Embeds each research interest
    Returns the updated profile.
    """
    log.info("Expanding research interest synonyms…")
    profile.interest_synonyms = await expand_synonyms(profile.research_interests, client)

    log.info("Embedding research interests…")
    vecs = await embed_texts(profile.research_interests, client)
    profile.interest_embeddings = {
        interest: vec
        for interest, vec in zip(profile.research_interests, vecs)
    }
    return profile


def best_interest_similarity(
    candidate_topics: list[str],
    profile: StudentProfile,
    topic_embeddings: dict[str, list[float]],
) -> float:
    """
    Compute the maximum cosine similarity between any candidate topic embedding
    and any student interest embedding.
    Used in domain guard and scoring.
    """
    if not candidate_topics or not profile.interest_embeddings:
        return 0.0

    best = 0.0
    for topic in candidate_topics:
        if topic not in topic_embeddings:
            continue
        t_vec = topic_embeddings[topic]
        for i_vec in profile.interest_embeddings.values():
            sim = cosine_similarity(t_vec, i_vec)
            best = max(best, sim)
    return best

