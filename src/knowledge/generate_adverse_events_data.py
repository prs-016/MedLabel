"""
src/knowledge/generate_adverse_events_data.py
==============================================
Generate training data for the *adverse_events* cross-encoder reranker.

Unlike vector_search (general search), adverse_events is domain-specific:
it answers "what bad things can this drug cause?" So we restrict the
sampled chunks to the safety-related label sections and ask Grok for
queries phrased the way patients/clinicians ask about side effects,
warnings, and contraindications.

For each sampled chunk:
    good_query : a side-effect / warning question the chunk answers  (1.0)
    bad_query  : a hard negative with similar safety vocabulary       (0.0)
plus a cross-chunk hard negative (good_query vs an unrelated chunk).

Output: data/adverse_events_pairs.json   (query / chunk / label)
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

OUTPUT_PATH = os.path.join(ROOT_DIR, "data", "adverse_events_pairs.json")

# Safety-related sections that adverse_events actually searches over.
ADVERSE_SECTIONS = {
    "adverse_reactions",
    "warnings_and_cautions",
    "boxed_warning",
    "contraindications",
    "overdosage",
    "warnings",
}

MAX_CHUNKS = int(os.getenv("AE_MAX_CHUNKS", "120"))


def _grok_adverse_queries(chunk_text: str) -> dict | None:
    """Ask Grok for one good and one hard-negative adverse-event query."""
    system_prompt = (
        "You are an expert pharmacovigilance data annotator generating training "
        "data for a Cross-Encoder reranker that retrieves drug SAFETY information "
        "(adverse reactions, side effects, warnings, contraindications). "
        "Respond ONLY with a raw JSON object with exactly two keys: "
        "'good_query' and 'bad_query'. No markdown, no code fences."
    )
    user_prompt = (
        f'Read this drug-label safety text chunk:\n"""{chunk_text}"""\n\n'
        "1. good_query: a natural question a patient or clinician would ask about "
        "side effects, adverse reactions, warnings, or contraindications that THIS "
        "chunk directly answers.\n"
        "2. bad_query: a tricky safety-sounding question that reuses similar terms "
        "but that this chunk does NOT actually answer."
    )
    try:
        resp = client.chat.completions.create(
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


def main() -> None:
    print("Loading chunks ...")
    all_chunks = _load_chunks()
    safety_chunks = [
        c for c in all_chunks
        if c["metadata"].get("section_type", "") in ADVERSE_SECTIONS
    ]
    print(f"  {len(all_chunks)} total chunks, "
          f"{len(safety_chunks)} in safety sections.")
    if not safety_chunks:
        raise SystemExit("No safety-section chunks found -- seed the DB first.")

    sample = _stratified_sample(safety_chunks, MAX_CHUNKS)
    print(f"Sampling {len(sample)} diverse safety chunks "
          f"(model={GROK_MODEL}).\n")

    pairs: list[dict] = []
    for i, chunk in enumerate(sample, 1):
        drug = chunk["metadata"].get("drug_name", "?")
        sec  = chunk["metadata"].get("section_type", "?")
        print(f"[{i}/{len(sample)}] {drug} / {sec}")

        qs = _grok_adverse_queries(_truncate(chunk["text"]))
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

        time.sleep(0.3)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)

    n_pos = sum(1 for p in pairs if p["label"] == 1.0)
    n_neg = sum(1 for p in pairs if p["label"] == 0.0)
    print(f"\nDone. {len(pairs)} pairs  ({n_pos} positive / {n_neg} negative)")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
