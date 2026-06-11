"""
Domain leakage guard — two-gate approach.

Gate A (fast, no LLM): anchor concept check.
  Each research area has a set of required anchor terms.
  If a candidate's topics contain zero anchor terms → discard.
  This is O(n) with no API calls.

Gate B (LLM): discipline classification.
  For candidates passing Gate A, send their best evidence (abstract/topics)
  to GPT-4o-mini for a binary domain check.
  This catches subtle cross-domain leakage like:
    - "DNA barcoding" → chromatin biology (plant bio student)
    - "trauma-informed" → Roman antiquity (clinical psych student)
    - "high-elevation systems" → Pacific Northwest fire archaeology

Design trade-off:
  Gate A alone has too many false negatives (misses subtle leakage).
  Gate B alone is too slow for 500+ candidates.
  Two-gate approach: Gate A reduces the pool; Gate B handles the hard cases.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from openai import AsyncOpenAI

from src.candidate import Candidate
from src.profile_parser import StudentProfile
from src.llm_client import LLMClient

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gate A: anchor concept sets per broad domain
# These are OpenAlex topic display-name fragments (lowercase, partial match OK)
# ---------------------------------------------------------------------------
DOMAIN_ANCHORS: dict[str, list[str]] = {
    # Neuroscience / brain
    "computational neuroscience": ["neuroscience", "neural", "cortex", "synapse", "neuron", "brain", "cognit", "fmri", "eeg", "spike", "plasticity", "hippocampus"],
    "neural coding": ["neural coding", "neural representation", "population code", "place cell", "grid cell", "neural activity", "sensory coding"],
    "brain-computer interface": ["brain-computer", "bci", "neural interface", "motor imagery", "eeg", "neuroprosthetic", "closed-loop"],
    "systems neuroscience": ["systems neuroscience", "neural circuit", "in vivo", "electrophysiology", "calcium imaging", "optogenetics"],
    "neuroimaging": ["fmri", "neuroimaging", "brain imaging", "bold signal", "connectome", "diffusion tensor", "pet scan"],
    # AI / ML core — covers student_001's research interests
    "machine learning": ["machine learning", "deep learning", "neural network", "reinforcement learning", "transformer", "llm", "generative model", "gradient descent", "supervised learning", "unsupervised"],
    "artificial intelligence": ["artificial intelligence", "machine learning", "deep learning", "neural network", "ai system", "intelligent system", "autonomous", "reasoning"],
    "natural language processing": ["nlp", "natural language", "language model", "text classification", "named entity", "question answering", "sentiment", "bert", "gpt"],
    "large language models": ["large language model", "llm", "gpt", "language model", "foundation model", "instruction tuning", "rlhf", "pretrained", "transformer", "prompt"],
    "generative ai": ["generative", "diffusion model", "gan", "vae", "image generation", "text generation", "stable diffusion", "gpt", "llm", "foundation model"],
    "agentic ai systems": ["agent", "agentic", "multi-agent", "autonomous agent", "tool use", "planning", "reasoning", "llm agent", "workflow", "orchestration"],
    "multimodal ai": ["multimodal", "vision-language", "image-text", "cross-modal", "clip", "vqa", "visual question", "captioning", "audio-visual"],
    "computer vision": ["computer vision", "image recognition", "object detection", "convolutional", "semantic segmentation", "visual", "yolo", "resnet", "vit", "feature extraction"],
    "industrial ai": ["industrial", "manufacturing", "defect detection", "quality control", "robotics", "automation", "iot", "edge computing", "predictive maintenance"],
    "human-ai interaction": ["human-ai", "human-computer", "hci", "explainability", "interpretability", "fairness", "user study", "interface", "trust", "transparency"],
    # Life sciences
    "biomaterials": ["biomaterial", "scaffold", "tissue engineering", "biocompatibility", "hydrogel", "implant", "biodegradable"],
    "climate science": ["climate change", "carbon", "greenhouse", "atmospheric", "sea level", "glacial", "ocean circulation"],
    "ecology": ["ecosystem", "biodiversity", "species", "habitat", "population dynamics", "food web", "conservation"],
    "genetics": ["genome", "dna sequencing", "gene expression", "mutation", "crispr", "epigenetics", "rna"],
    "chemistry": ["synthesis", "catalysis", "organic chemistry", "polymer", "spectroscopy", "reaction mechanism"],
    "physics": ["quantum", "particle physics", "condensed matter", "thermodynamics", "optics", "photonics"],
    "economics": ["economics", "market", "welfare", "monetary policy", "gdp", "inequality", "labor market"],
    "psychology": ["psychology", "cognition", "behavior", "mental health", "anxiety", "depression", "therapy"],
    "history": ["history", "historical", "archival", "medieval", "ancient", "colonial", "manuscript"],
    "literature": ["literary", "literature", "narrative", "poetry", "fiction", "rhetoric", "textual analysis"],
}


def _get_anchors_for_interest(interest: str) -> list[str]:
    """
    Find the best matching anchor set for a research interest string.
    Falls back to splitting the interest into keywords.
    """
    interest_lower = interest.lower()

    # Direct lookup
    if interest_lower in DOMAIN_ANCHORS:
        return DOMAIN_ANCHORS[interest_lower]

    # Partial match
    for domain, anchors in DOMAIN_ANCHORS.items():
        if any(word in interest_lower for word in domain.split()):
            return anchors

    # No match — use the interest words themselves as anchors
    return [w for w in interest_lower.split() if len(w) > 4]


def gate_a_check(candidate: Candidate, profile: StudentProfile) -> bool:
    """
    Returns True if the candidate passes Gate A (should proceed to Gate B or acceptance).
    Returns False if the candidate is clearly out of domain.
    """
    candidate_topics_lower = " ".join(candidate.topics).lower()

    # When no OpenAlex topics, fall back to grant titles as signal text
    if not candidate_topics_lower:
        grant_text = " ".join(
            g.title for g in candidate.grants if g.title
        ).lower()
        if not grant_text:
            return True  # truly no evidence — give benefit of doubt to Gate B

        # Check grant titles against anchors
        for interest in profile.research_interests:
            anchors = _get_anchors_for_interest(interest)
            if any(anchor in grant_text for anchor in anchors):
                return True
        # Grant titles exist but none matched any anchor → reject at Gate A
        log.debug(
            f"Gate A (grant-title check) REJECTED: {candidate.name} — "
            f"grants: {grant_text[:80]}"
        )
        return False

    for interest in profile.research_interests:
        anchors = _get_anchors_for_interest(interest)
        if any(anchor in candidate_topics_lower for anchor in anchors):
            return True  # at least one interest matches at least one anchor

    return False


async def gate_b_check(
    candidate: Candidate,
    profile: StudentProfile,
    client: AsyncOpenAI | LLMClient,
) -> tuple[bool, float]:
    """
    LLM domain check. Returns (passes: bool, confidence: float).
    Sends candidate topics/abstract to GPT-4o-mini.
    """
    if not isinstance(client, LLMClient):
        client = LLMClient(client)

    topic_text = ", ".join(candidate.topics[:8]) if candidate.topics else ""
    paper_titles = "; ".join(p.title for p in candidate.papers[:2]) if candidate.papers else ""
    grant_titles = "; ".join(g.title for g in candidate.grants[:3] if g.title) if candidate.grants else ""
    evidence = topic_text or paper_titles or grant_titles or "No topic information available"

    prompt = (
        "You are classifying whether a researcher's work is in the same scientific domain "
        "as a PhD student's research interests.\n\n"
        f"Student research interests: {json.dumps(profile.research_interests)}\n"
        f"Researcher's topics/papers: {evidence}\n\n"
        "Task: Is this researcher working in the same or closely adjacent domain as the student? "
        "A 'yes' means their work would be relevant to the student. "
        "A 'no' means it's a different discipline (even if some keywords overlap).\n\n"
        "Respond ONLY with JSON: "
        '{"match": true or false, "discipline": "one word domain name", "reason": "1 sentence"}'
    )

    try:
        result = await client.chat_complete_json(prompt)
        return bool(result.get("match", True)), 1.0
    except Exception as exc:
        log.warning(f"Gate B LLM call failed for {candidate.name}: {exc}")
        return True, 0.5  # default: pass (prefer false negatives over false positives here)


async def apply_domain_guard(
    candidates: list[Candidate],
    profile: StudentProfile,
    client: AsyncOpenAI | LLMClient,
) -> list[Candidate]:
    """
    Apply Gate A then Gate B in sequence.
    Returns filtered list. Logs rejection counts per gate.
    """
    # Gate A (synchronous)
    gate_a_passed = []
    gate_a_rejected = 0
    for c in candidates:
        if gate_a_check(c, profile):
            gate_a_passed.append(c)
        else:
            log.debug(f"Domain Gate A REJECTED: {c.name} — topics: {c.topics[:3]}")
            gate_a_rejected += 1

    log.info(f"Domain Gate A: {gate_a_rejected} rejected, {len(gate_a_passed)} passed")

    # Gate B (async LLM — batch all at once)
    async def _check(c: Candidate) -> tuple[Candidate, bool]:
        passed, _ = await gate_b_check(c, profile, client)
        return c, passed

    tasks = [_check(c) for c in gate_a_passed]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    gate_b_passed = []
    gate_b_rejected = 0
    for result in results:
        if isinstance(result, Exception):
            continue
        c, passed = result
        if passed:
            gate_b_passed.append(c)
        else:
            log.debug(f"Domain Gate B REJECTED: {c.name} — topics: {c.topics[:3]}")
            gate_b_rejected += 1

    log.info(f"Domain Gate B: {gate_b_rejected} rejected, {len(gate_b_passed)} passed")
    return gate_b_passed
