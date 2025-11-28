"""AI-assisted filtering logic shared by phase 3."""

from __future__ import annotations

import json

from tqdm import tqdm

from .api import chat_completion, get_message_content
from .config import BATCH_P1, MODEL_P1_CTX
from .text_processing import extract_json_array_of_objects


def get_related_terms(term: str, existing_terms: set[str]) -> list[str]:
    term_tokens = set(term.lower().split())
    related = []
    for t in existing_terms:
        t_tokens = set(t.lower().split())
        if term_tokens & t_tokens:
            related.append(t)
    return related


def ai_classify_with_context(all_terms, contexts_cache, existing_terms):
    tag_map = {}
    if not all_terms:
        return tag_map

    for i in tqdm(range(0, len(all_terms), BATCH_P1), desc="AI Step 1: With Context"):
        batch = all_terms[i:i + BATCH_P1]

        objs = []
        for t in batch:
            ctx = contexts_cache.get(t, [])
            related_existing = get_related_terms(t, existing_terms)
            objs.append(
                {
                    "term": t,
                    "contexts": ctx[:30],
                    "existing_terms": sorted(related_existing),
                }
            )

        prompt = _build_context_prompt(objs)
        data = chat_completion(
            [{"role": "system", "content": "Output JSON array only."}, {"role": "user", "content": prompt}],
            model=MODEL_P1_CTX,
        )
        content = get_message_content(data) if data else ""

        try:
            rows = extract_json_array_of_objects(content)
        except Exception:
            for t in batch:
                tag_map[t] = "Need Recheck"
            continue

        for obj in rows:
            try:
                t = str(obj.get("term", "")).strip()
                tag = str(obj.get("tag", "")).strip()
                if t in batch and tag in {"Keep", "Remove", "Need Recheck"}:
                    tag_map[t] = tag
                    if tag == "Keep":
                        existing_terms.add(t)
            except Exception:
                continue

        for t in batch:
            if t not in tag_map:
                tag_map[t] = "Need Recheck"

    return tag_map


def ai_prune_redundant_terms(keep_terms, existing_terms):
    tag_map = {}
    if not keep_terms:
        return tag_map

    for i in tqdm(range(0, len(keep_terms), BATCH_P1), desc="AI Step 2: Prune Redundant"):
        batch = keep_terms[i:i + BATCH_P1]
        objs = []
        for t in batch:
            related_existing = get_related_terms(t, existing_terms)
            objs.append({"term": t, "existing_terms": sorted(related_existing)})

        prompt = _build_redundancy_prompt(objs)
        data = chat_completion(
            [{"role": "system", "content": "Output JSON array only."}, {"role": "user", "content": prompt}],
            model=MODEL_P1_CTX,
        )
        content = get_message_content(data) if data else ""

        try:
            rows = extract_json_array_of_objects(content)
        except Exception:
            for t in batch:
                tag_map[t] = "Keep"
            continue

        for obj in rows:
            try:
                t = str(obj.get("term", "")).strip()
                tag = str(obj.get("tag", "")).strip()
                if t in batch and tag in {"Keep", "Remove"}:
                    tag_map[t] = tag
                    if tag == "Keep":
                        existing_terms.add(t)
                    elif tag == "Remove" and t in existing_terms:
                        existing_terms.remove(t)
            except Exception:
                continue

        for t in batch:
            if t not in tag_map:
                tag_map[t] = "Keep"

    return tag_map


def _build_context_prompt(objs):
    return f"""
You are reviewing English localization terms for a video game.

Each term must be evaluated **as-is**, based only on:
- Its own content
- Context lines provided
- Related existing terms (existing_terms)

Do NOT infer meanings, synonyms, or definitions. 
Only compare tokens exactly; do not match substrings within other words.

Assign each term exactly one tag:
- Keep: term is complete, meaningful, necessary, and not redundant
- Remove: term is incomplete, redundant, or already covered by existing_terms
- Need Recheck: ambiguous, unsure, or insufficient context
Return JSON array ONLY, one object per term:
[
  {{"term":"...","tag":"Keep"}},
  {{"term":"...","tag":"Remove"}},
  {{"term":"...","tag":"Need Recheck"}}
]

Input terms with contexts and related existing terms:
{json.dumps(objs, ensure_ascii=False, indent=2)}
""".strip()


def _build_redundancy_prompt(objs):
    return f"""
You are reviewing English localization terms for a video game. 
Each term must be evaluated strictly as a stand-alone term for translation. 
Do NOT provide definitions, explanations, or suggestions.

Input for each term:
- "term": the term to evaluate (single word or multi-word)
- "existing_terms": list of existing terms already approved

Your task:
For each term, decide whether it should be:
- Keep: the term is complete, meaningful, and not redundant
- Remove: the term is incomplete, redundant, or its meaning is already covered by existing_terms

Important:
- Only use existing_terms for redundancy checks; do not invent synonyms or definitions

Return JSON array ONLY, one object per term:
[
  {{"term":"...","tag":"Keep"}},
  {{"term":"...","tag":"Remove"}}
]

Input terms with related existing terms:
{json.dumps(objs, ensure_ascii=False, indent=2)}
""".strip()

