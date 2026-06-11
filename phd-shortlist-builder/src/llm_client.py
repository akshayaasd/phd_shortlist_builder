"""
Unified LLM and Embedding Client supporting OpenAI, Gemini, and keyless Offline fallback.
"""
from __future__ import annotations

import os
import json
import logging
import httpx
from typing import Any
from openai import AsyncOpenAI

log = logging.getLogger(__name__)

# Constants
OPENAI_PLACEHOLDER = "sk-mnopqrstijkl5678mnopqrstijkl5678mnopqrst"

def is_valid_openai_key(key: str | None) -> bool:
    if not key:
        return False
    key = key.strip()
    if not key or key == OPENAI_PLACEHOLDER or key.startswith("your_") or "placeholder" in key.lower():
        return False
    return True

class LLMClient:
    def __init__(self, openai_client: AsyncOpenAI | None = None):
        self.openai_client = openai_client
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")

        # Set flags based on key validity
        self.use_openai = is_valid_openai_key(self.openai_key)
        self.use_gemini = bool(self.gemini_key and self.gemini_key.strip())

        if not self.use_openai and not self.use_gemini:
            log.warning("No valid OPENAI_API_KEY or GEMINI_API_KEY found. Running in Offline Mock mode (completely free!).")

    async def get_embedding(self, texts: list[str]) -> list[list[float]]:
        """Gets embeddings for the texts, falling back from OpenAI -> Gemini -> Mock."""
        if self.use_openai and self.openai_client:
            try:
                log.info(f"Requesting OpenAI embeddings for {len(texts)} texts...")
                response = await self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                log.warning(f"OpenAI embedding request failed: {e}. Trying Gemini/Mock fallback...")
                self.use_openai = False  # disable for this run

        if self.use_gemini or os.getenv("GEMINI_API_KEY"):
            key = self.gemini_key or os.getenv("GEMINI_API_KEY")
            try:
                log.info(f"Requesting Gemini embeddings for {len(texts)} texts...")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={key}"
                headers = {"Content-Type": "application/json"}
                results = []
                async with httpx.AsyncClient() as http:
                    for text in texts:
                        payload = {"content": {"parts": [{"text": text}]}}
                        res = await http.post(url, json=payload, headers=headers, timeout=20.0)
                        res.raise_for_status()
                        data = res.json()
                        results.append(data["embedding"]["values"])
                return results
            except Exception as e:
                log.warning(f"Gemini embedding request failed: {e}. Using Offline Mock embeddings...")

        # Offline Mock embedding fallback (random but deterministic-by-hash normalized vectors)
        log.info("Generating Offline Mock embeddings...")
        import hashlib
        results = []
        for text in texts:
            # Generate a deterministic vector based on MD5 hash of text
            h = hashlib.md5(text.encode('utf-8')).digest()
            vec = []
            for i in range(1536):
                # Pseudo-random float from hash bytes
                val = ((h[i % len(h)] + i) % 256) / 255.0 - 0.5
                vec.append(val)
            # Normalize vector
            norm = sum(x*x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            results.append(vec)
        return results

    async def chat_complete_json(self, prompt: str, system_instruction: str = "") -> dict[str, Any]:
        """Gets a JSON response from OpenAI -> Gemini -> Mock."""
        if self.use_openai and self.openai_client:
            try:
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})

                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                raw = response.choices[0].message.content or "{}"
                return json.loads(raw)
            except Exception as e:
                log.warning(f"OpenAI chat completion failed: {e}. Trying Gemini/Mock fallback...")
                self.use_openai = False

        if self.use_gemini or os.getenv("GEMINI_API_KEY"):
            key = self.gemini_key or os.getenv("GEMINI_API_KEY")
            try:
                log.info("Requesting Gemini chat completion...")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"
                headers = {"Content-Type": "application/json"}
                
                # Combine instructions and prompt for Gemini
                contents = []
                if system_instruction:
                    contents.append({"role": "user", "parts": [{"text": f"System Instruction: {system_instruction}\n\nUser Request: {prompt}"}]})
                else:
                    contents.append({"role": "user", "parts": [{"text": prompt}]})

                payload = {
                    "contents": contents,
                    "generationConfig": {
                        "responseMimeType": "application/json"
                    }
                }
                async with httpx.AsyncClient() as http:
                    res = await http.post(url, json=payload, headers=headers, timeout=30.0)
                    res.raise_for_status()
                    data = res.json()
                    raw = data["candidates"][0]["content"]["parts"][0]["text"]
                    return json.loads(raw)
            except Exception as e:
                log.warning(f"Gemini chat completion failed: {e}. Using Offline Mock response...")

        # Offline Mock JSON responses based on keywords in prompt
        log.info("Generating Offline Mock chat completion...")
        prompt_lower = prompt.lower()
        if "synonym" in prompt_lower or "alternative academic phrasing" in prompt_lower:
            # We are expanding synonyms
            # Extract areas from prompt if possible
            try:
                # Prompt lists areas as json: f"Research areas: {json.dumps(interests)}"
                if "research areas:" in prompt_lower:
                    idx = prompt_lower.index("research areas:")
                    raw_list = prompt[idx + 15:].strip()
                    interests = json.loads(raw_list)
                else:
                    interests = []
            except Exception:
                interests = []

            mock_syns = {}
            for item in interests:
                # Provide reasonable dummy academic synonyms
                mock_syns[item] = [
                    f"advanced {item}",
                    f"computational {item}",
                    f"applied {item} research"
                ]
            return mock_syns

        elif "match rationale" in prompt_lower or "why_match" in prompt_lower:
            # We are generating why_match — extract candidate details from the prompt to avoid generic text
            why = ""
            conf = 0.75
            # Try to extract paper title from prompt
            import re as _re
            paper_match = _re.search(r"- (.+?) \((\d{4}), cited (\d+)x\)", prompt)
            grant_match = _re.search(r"- (.+?) \[(NIH|UKRI|ARC)\]", prompt)
            prof_match = _re.search(r"Professor: (.+?) \((.+?)\)", prompt)
            interest_match = _re.search(r'Student research areas: (\[.+?\])', prompt)
            prof_name = prof_match.group(1).split()[0] if prof_match else "This professor"
            interests = []
            if interest_match:
                try:
                    interests = json.loads(interest_match.group(1))
                except Exception:
                    pass
            top_interest = interests[0] if interests else "AI research"
            if paper_match:
                paper_title = paper_match.group(1)
                year = paper_match.group(2)
                why = (
                    f"Prof. {prof_name}'s {year} work on \"{paper_title}\" directly addresses "
                    f"{top_interest}, which is central to my research goals. "
                    f"The methodologies in that paper align well with my background in "
                    f"{', '.join(interests[1:3]) if len(interests) > 1 else 'machine learning'}."
                )
                conf = 0.80
            elif grant_match:
                grant_title = grant_match.group(1)
                funder = grant_match.group(2)
                why = (
                    f"Prof. {prof_name}'s active {funder} grant on \"{grant_title}\" "
                    f"signals ongoing funded research in {top_interest}. "
                    f"This matches my interest in {', '.join(interests[:2]) if interests else 'applied AI'}."
                )
                conf = 0.72
            else:
                why = (
                    f"Prof. {prof_name}'s research in {top_interest} aligns with my "
                    f"background in {', '.join(interests[1:3]) if len(interests) > 1 else 'machine learning and computer vision'}. "
                    f"The group's focus presents a strong match for my PhD goals."
                )
                conf = 0.65
            return {"why_match": why, "fit_confidence": conf}

        elif "same scientific domain" in prompt_lower or "match" in prompt_lower:
            # We are doing domain check
            return {
                "match": True,
                "discipline": "STEM",
                "reason": "Offline check: broad keyword overlap detected in academic profile."
            }

        return {}
