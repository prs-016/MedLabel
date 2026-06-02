"""
src/knowledge/generate_interaction_data.py
============================================
Generate training data for the *interaction_check* cross-encoder reranker
(models/medlabel_reranker/).

interaction_check answers "is it safe to combine drug X with Y / alcohol /
food / a condition?" so — unlike the legacy generic generator — we:
  * restrict sampling to interaction-relevant label sections, and
  * ask Grok for interaction-shaped queries (combining drugs, alcohol,
    contraindications, "who should not take", drug-disease conflicts).

For each sampled chunk:
    good_query : an interaction/contraindication question the chunk answers (1.0)
    bad_query  : a hard negative that reuses similar terms but is NOT
                 answered by the chunk                                    (0.0)
plus a cross-chunk hard negative (good_query vs an unrelated chunk).

Output: data/interaction_pairs.json   (query / chunk / label)
"""

from __future__ import annotations

import json
import os
import random
import time

from knowledge.generate_vector_search_data import (
    _load_chunks,
    _stratified_sample,
    _truncate,
    client,
    GROK_MODEL,
    ROOT_DIR,
)

OUTPUT_PATH = os.path.join(ROOT_DIR, "data", "interaction_pairs.json")

# Sections interaction_check actually searches over (see _INTERACTION_SECTIONS
# in query_functions.py). Restricting here keeps the reranker focused on
# combination/contraindication content rather than general label text.
INTERACTION_SECTIONS = {
    "drug_drug_interaction",
    "drug_interactions",
    "interaction_summary",
    "drug_disease_interaction",
    "contraindications",
    "overdosage",
}

MAX_CHUNKS = int(os.getenv("IC_MAX_CHUNKS", "280"))


def _grok_interaction_queries(chunk_text: str) -> dict | None:
    """Ask Grok for one good and one hard-negative interaction query."""
    system_prompt = (
        "You are an expert clinical-pharmacology data annotator generating "
        "training data for a Cross-Encoder reranker that retrieves DRUG "
        "INTERACTION and CONTRAINDICATION information (drug-drug, drug-alcohol, "
        "drug-food, drug-disease conflicts, and who must not take a medicine). "
        "Respond ONLY with a raw JSON object with exactly two keys: "
        "'good_query' and 'bad_query'. No markdown, no code fences."
    )
    user_prompt = (
        f'Read this drug-label interaction/contraindication text chunk:\n'
        f'"""{chunk_text}"""\n\n'
        "1. good_query: a natural question a patient or clinician would ask "
        "about COMBINING this medicine with another drug, alcohol, or food, or "
        "about who should not take it / what conditions make it unsafe — phrased "
        "the way people actually ask (e.g. 'Can I take this with ibuprofen?', "
        "'Is it safe to drink alcohol on this?', 'Who should not take this?'). "
        "The question must be one THIS chunk directly answers.\n"
        "2. bad_query: a tricky interaction- or safety-sounding question that "
        "reuses similar drug/condition terms but that this chunk does NOT "
        "actually answer."
    )
    # Per-request timeout + a couple of retries so one slow/hung Grok call
    # cannot block the whole run forever (the shared client has no timeout).
    safe_client = client.with_options(timeout=40.0, max_retries=2)
    try:
        resp = safe_client.chat.completions.create(
            model=GROK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw[raw.find("{"):]
        out = json.loads(raw)
        if "good_query" in out and "bad_query" in out:
            return out
    except Exception as exc:  # noqa: BLE001
        print(f"   ! Grok error: {str(exc)[:100]}")
    return None


def _save(pairs: list[dict]) -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)


def main() -> None:
    print("Loading chunks ...")
    all_chunks = _load_chunks()
    interaction_chunks = [
        c for c in all_chunks
        if c["metadata"].get("section_type", "") in INTERACTION_SECTIONS
    ]
    print(f"  {len(all_chunks)} total chunks, "
          f"{len(interaction_chunks)} in interaction sections.")
    if not interaction_chunks:
        raise SystemExit(
            "No interaction-section chunks found -- seed the DB first."
        )

    sample = _stratified_sample(interaction_chunks, MAX_CHUNKS)
    print(f"Sampling {len(sample)} diverse interaction chunks "
          f"(model={GROK_MODEL}).\n")

    pairs: list[dict] = []
    for i, chunk in enumerate(sample, 1):
        drug = chunk["metadata"].get("drug_name", "?")
        sec  = chunk["metadata"].get("section_type", "?")
        print(f"[{i}/{len(sample)}] {drug} / {sec}")

        qs = _grok_interaction_queries(_truncate(chunk["text"]))
        if not qs:
            continue

        passage = _truncate(chunk["text"])
        pairs.append({"query": qs["good_query"], "chunk": passage, "label": 1.0})
        pairs.append({"query": qs["bad_query"],  "chunk": passage, "label": 0.0})
        other = random.choice(sample)
        if other is not chunk:
            pairs.append({
                "query": qs["good_query"],
                "chunk": _truncate(other["text"]),
                "label": 0.0,
            })

        # Checkpoint periodically so a crash/hang keeps most progress.
        if i % 20 == 0:
            _save(pairs)
            print(f"   .. checkpoint saved ({len(pairs)} pairs)")

        time.sleep(0.3)

    _save(pairs)

    n_pos = sum(1 for p in pairs if p["label"] == 1.0)
    n_neg = sum(1 for p in pairs if p["label"] == 0.0)
    print(f"\nDone. {len(pairs)} pairs  ({n_pos} positive / {n_neg} negative)")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
