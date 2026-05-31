"""
test_pipeline.py
=================
Unified end-to-end test of the MedLabel pipeline.

Flow being tested:
    User question
        → Grok router (classifies intent)
        → ChromaDB retrieval (BGE-M3 embeddings, 2 datasets)
              fda_label    — dosage, warnings, adverse reactions, etc.
              ddinter2_web — drug-drug interactions
        → Two-stage reranker (BGE → cross-encoder)
        → Grok synthesizes final answer

Usage:
    PYTHONPATH=src python test_pipeline.py
"""

from __future__ import annotations
import os, sys, textwrap, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv()

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN  = "\033[96m"; BOLD = "\033[1m"; RESET  = "\033[0m"

passed = []; failed = []

def ok(label, msg=""):
    passed.append(label)
    print(f"{GREEN}  ✅ {label}{RESET}" + (f"  {msg}" if msg else ""))

def fail(label, msg=""):
    failed.append(label)
    print(f"{RED}  ❌ {label}{RESET}" + (f"  {msg}" if msg else ""))

def hdr(title):
    print(f"\n{BOLD}{CYAN}{'═'*64}\n  {title}\n{'═'*64}{RESET}")

def subhdr(title):
    print(f"\n{BOLD}{YELLOW}  ── {title} ──{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. ChromaDB — verify both datasets are present
# ══════════════════════════════════════════════════════════════════════════════
hdr("1 · ChromaDB  (fda_label + ddinter2_web)")
try:
    from knowledge.embedder import DrugEmbedder
    db      = DrugEmbedder()
    count   = db.collection.count()
    results = db.query("dosage", n_results=500)
    sources  = set(r["metadata"].get("source", "?")  for r in results)
    sections = set(r["metadata"].get("section_type","?") for r in results)
    drugs    = set(r["metadata"].get("drug_name","?")    for r in results)

    if count > 0:
        ok("Connected", f"{count} total chunks | {len(drugs)} drugs")
    else:
        fail("ChromaDB EMPTY — run ingest_data.py first")
        sys.exit(1)

    for expected in ("fda_label", "ddinter2_web"):
        if expected in sources:
            ok(f"Dataset present: {expected}")
        else:
            fail(f"Dataset missing: {expected}", "re-run ingest_data.py")

    ok("Section types found", ", ".join(sorted(sections)[:5]) + "…")

except Exception as e:
    fail("ChromaDB", str(e))
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 2. vector_search — fda_label chunks, varied drugs
# ══════════════════════════════════════════════════════════════════════════════
hdr("2 · vector_search  (fda_label → BGE-M3 → reranker)")
try:
    from knowledge.query_functions import vector_search

    cases = [
        # (query, drug, expected_section_hint)
        ("What is the adult dosage of metformin?",             "metformin",     "dosage"),
        ("What are the contraindications for ibuprofen?",      "ibuprofen",     "warnings"),
        ("How should levothyroxine be taken?",                 "levothyroxine", "dosage"),
        ("Is gabapentin safe during pregnancy?",               "gabapentin",    "pregnancy"),
        ("What are the warnings for alprazolam?",              "alprazolam",    "warnings"),
        ("What is the dosage of amoxicillin for children?",    "amoxicillin",   "dosage"),
        ("What is the maximum dose of quetiapine?",            "quetiapine",    "dosage"),
        ("What are the overdose symptoms of zolpidem?",        "zolpidem",      "overdosage"),
        ("How does atorvastatin work?",                        "atorvastatin",  "clinical_pharmacology"),
        ("What are the indications for prednisone?",           "prednisone",    "indications"),
    ]

    for query, drug, hint in cases:
        results = vector_search(query, drug_name=drug, top_n=3)
        if results:
            top     = results[0]
            name    = top["metadata"].get("drug_name", "?")
            section = top["metadata"].get("section_type", "?")
            snip    = top["text"][:60].replace("\n", " ")
            ok(f"[{drug}]", f"section={section} | {snip}…")
        else:
            fail(f"[{drug}]", f"0 results")

except Exception as e:
    fail("vector_search", str(e))
    import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# 3. interaction_check — ddinter2_web + fda_label chunks
# ══════════════════════════════════════════════════════════════════════════════
hdr("3 · interaction_check  (ddinter2_web + fda_label → reranker)")
try:
    from knowledge.query_functions import interaction_check

    cases = [
        ("warfarin",     "aspirin"),
        ("fluoxetine",   "tramadol"),
        ("simvastatin",  "cyclosporine"),
        ("lisinopril",   "spironolactone"),
        ("phenytoin",    "carbamazepine"),
        ("metoprolol",   "carvedilol"),
        ("digoxin",      None),
        ("methotrexate", None),
    ]

    for drug_a, drug_b in cases:
        label  = f"{drug_a} + {drug_b}" if drug_b else drug_a
        kwargs = {"drug_a": drug_a, "top_n": 3}
        if drug_b:
            kwargs["drug_b"] = drug_b
        results = interaction_check(**kwargs)
        if results:
            top     = results[0]
            name    = top["metadata"].get("drug_name", "?")
            source  = top["metadata"].get("source", "?")
            snip    = top["text"][:60].replace("\n", " ")
            ok(f"[{label}]", f"source={source} | {snip}…")
        else:
            fail(f"[{label}]", "0 results")

except Exception as e:
    fail("interaction_check", str(e))
    import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# 4. adverse_events — fda_label adverse_reactions sections
# ══════════════════════════════════════════════════════════════════════════════
hdr("4 · adverse_events  (fda_label adverse sections → reranker)")
try:
    from knowledge.query_functions import adverse_events

    cases = [
        ("metformin",    None),
        ("ibuprofen",    "stomach bleeding"),
        ("fluoxetine",   "serotonin syndrome"),
        ("atorvastatin", "muscle pain"),
        ("lisinopril",   "cough"),
        ("gabapentin",   None),
        ("prednisone",   None),
        ("sertraline",   None),
    ]

    for drug, hint in cases:
        kwargs = {"drug_name": drug, "top_n": 3}
        if hint:
            kwargs["reaction_hint"] = hint
        results = adverse_events(**kwargs)
        label   = f"{drug} [{hint}]" if hint else drug
        if results:
            top     = results[0]
            name    = top["metadata"].get("drug_name", "?")
            section = top["metadata"].get("section_type", "?")
            snip    = top["text"][:60].replace("\n", " ")
            ok(f"[{label}]", f"section={section} | {snip}…")
        else:
            fail(f"[{label}]", "0 results")

except Exception as e:
    fail("adverse_events", str(e))
    import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Grok router accuracy
# ══════════════════════════════════════════════════════════════════════════════
hdr("5 · Grok intent router")
xai_key = os.getenv("XAI_API_KEY", "")
if not xai_key:
    print(f"{YELLOW}  ⚠️  XAI_API_KEY not set — skipping sections 5 & 6{RESET}")
else:
    try:
        from agent.brain import _route_query

        routing_cases = [
            ("What is the dosage of metformin for adults?",           "vector_search"),
            ("What are the active ingredients in gabapentin?",        "vector_search"),
            ("Can I take fluoxetine with tramadol?",                  "interaction_check"),
            ("Is it safe to combine lisinopril and spironolactone?",  "interaction_check"),
            ("What are the side effects of prednisone?",              "adverse_events"),
            ("What reactions have been reported with atorvastatin?",  "adverse_events"),
            ("How does levothyroxine interact with calcium?",         "interaction_check"),
            ("What is the max dose of quetiapine?",                   "vector_search"),
        ]

        for question, expected in routing_cases:
            got = _route_query(question)
            if got == expected:
                ok(f"Routed correctly → {got}", textwrap.shorten(question, 55))
            else:
                fail(f"Wrong route", f"'{textwrap.shorten(question,45)}' → got {got}, expected {expected}")
            time.sleep(0.3)

    except Exception as e:
        fail("Grok router", str(e))
        import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# 6. Full unified pipeline — Grok → ChromaDB → Grok answer
# ══════════════════════════════════════════════════════════════════════════════
hdr("6 · Full pipeline  (Grok router → ChromaDB → Grok answer)")
if not xai_key:
    print(f"{YELLOW}  ⚠️  Skipped (no XAI_API_KEY){RESET}")
else:
    try:
        from agent.brain import MedBrain
        brain = MedBrain()

        pipeline_cases = [
            # vector_search questions
            ("What is the adult dosage of metformin?",               "metformin",     "vector_search"),
            ("What are the warnings for ibuprofen?",                 "ibuprofen",     "vector_search"),
            ("How should levothyroxine be taken?",                   "levothyroxine", "vector_search"),
            ("Is gabapentin safe during pregnancy?",                 "gabapentin",    "vector_search"),
            # interaction_check questions
            ("Can I take fluoxetine with tramadol?",                 "fluoxetine",    "interaction_check"),
            ("Is it safe to combine lisinopril and spironolactone?", "lisinopril",    "interaction_check"),
            ("What happens if I take simvastatin with cyclosporine?","simvastatin",   "interaction_check"),
            # adverse_events questions
            ("What are the side effects of prednisone?",             "prednisone",    "adverse_events"),
            ("What reactions have been reported with atorvastatin?", "atorvastatin",  "adverse_events"),
            ("Can sertraline cause serotonin syndrome?",             "sertraline",    "adverse_events"),
        ]

        for question, drug, expected_fn in pipeline_cases:
            result  = brain.ask(question, drug_name=drug)
            fn      = result.function_used
            hits    = result.hit_count
            answer  = result.summary.strip()
            fn_tag  = f"fn={fn}" + ("✓" if fn == expected_fn else f" ⚠expected {expected_fn}")

            if answer and len(answer) > 20:
                first = textwrap.shorten(answer.split("\n")[0], width=60)
                ok(f"[{drug}]", f"{fn_tag} hits={hits} | {first}")
            else:
                fail(f"[{drug}]", f"empty answer ({fn_tag})")
            time.sleep(0.5)

    except Exception as e:
        fail("Full pipeline", str(e))
        import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# 7. Training pairs check
# ══════════════════════════════════════════════════════════════════════════════
hdr("7 · Training pairs  (data/training_pairs.json)")
pairs_path = "data/training_pairs.json"
if not os.path.exists(pairs_path):
    print(f"{YELLOW}  ⚠️  Not found — run generate_training_pairs.py first{RESET}")
else:
    try:
        with open(pairs_path) as f:
            pairs = json.load(f)

        total    = len(pairs)
        positive = sum(1 for p in pairs if p.get("label") == 1)
        negative = sum(1 for p in pairs if p.get("label") == 0)
        chunk_key = "chunk_text" if "chunk_text" in pairs[0] else "chunk"
        has_text  = sum(1 for p in pairs if len(p.get(chunk_key, "")) > 20)
        has_q     = sum(1 for p in pairs if len(p.get("query", "")) > 5)
        balance   = min(positive, negative) / max(positive, negative, 1)
        drug_set  = set(
            p.get("drug_name", p.get("metadata", {}).get("drug_name", "?"))
            for p in pairs
        )

        ok("Pairs loaded", f"{total} total | {positive} pos | {negative} neg | {len(drug_set)} drugs")
        ok("All have queries")    if has_q    == total else fail("Missing queries",    f"{total-has_q} empty")
        ok("All have chunk text") if has_text == total else fail("Missing chunk text", f"{total-has_text} empty")
        ok("Balanced",f"{balance:.0%}") if balance >= 0.7 else fail("Imbalanced", f"only {balance:.0%} — regenerate")

        print(f"\n{YELLOW}  Sample pairs:{RESET}")
        for p in pairs[:4]:
            tag  = "✅ GOOD" if p.get("label") == 1 else "❌ BAD "
            drug = p.get("drug_name", "?")
            q    = textwrap.shorten(p.get("query", ""), width=65)
            print(f"    {tag} | {drug:20s} | {q}")

    except Exception as e:
        fail("Training pairs", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
total_p = len(passed); total_f = len(failed); total = total_p + total_f
print(f"\n{BOLD}{'═'*64}")
print(f"  Results: {GREEN}{total_p} passed{RESET}{BOLD}  {RED}{total_f} failed{RESET}{BOLD}  ({total} total)")
if total_f == 0:
    print(f"  {GREEN}All tests passed — ready to push! 🚀{RESET}")
else:
    print(f"  {RED}Fix the ❌ above before pushing.{RESET}")
    for f_name in failed:
        print(f"    • {f_name}")
print(f"{BOLD}{'═'*64}{RESET}\n")
