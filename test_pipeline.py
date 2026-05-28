import sys, logging
sys.path.insert(0, "src")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

# ── Test 1: RxNorm ─────────────────────────────────────────────────────────────
print("\n=== RxNorm ===")
from api.rxnorm import RxNormClient
rx = RxNormClient()
result = rx.resolve_ocr_string("Tylenol")
print(f"  input:   Tylenol")
print(f"  rxcui:   {result.rxcui}")
print(f"  generic: {result.generic_name}")
print(f"  found:   {result.found}")

# ── Test 2: openFDA ────────────────────────────────────────────────────────────
print("\n=== openFDA ===")
from api.openfda import OpenFDAClient
fda = OpenFDAClient()
chunks = fda.get_label_chunks("warfarin")
print(f"  chunks returned: {len(chunks)}")
if chunks:
    print(f"  first chunk section: {chunks[0]['metadata']['section_type']}")
    print(f"  first chunk preview: {chunks[0]['text'][:100]}")

# ── Test 3: DDInter ────────────────────────────────────────────────────────────
# print("\n=== DDInter ===")
# from api.ddInter import DDInterClient
# dd = DDInterClient("data/ddinter_downloads_code_A.csv")
# interactions = dd.get_interactions("warfarin")
# print(f"  interactions found: {len(interactions)}")
# if interactions:
#     print(f"  first: {interactions[0].drug_a} + {interactions[0].drug_b}")
#     print(f"  severity: {interactions[0].severity_label} ({interactions[0].severity_score})")
#     print(f"  max severity score: {dd.get_max_severity('warfarin')}")

# ── Test 4: ChromaDB embedder ──────────────────────────────────────────────────
# print("\n=== Embedder ===")
# from knowledge.embedder import DrugEmbedder
# embedder = DrugEmbedder(db_path="drug_db")
# test_chunks = dd.get_chunks("warfarin")[:3]
# n = embedder.upsert_chunks(test_chunks)
# print(f"  upserted: {n} chunks")
# results = embedder.query("warfarin bleeding risk")
# print(f"  query results: {len(results)}")
# if results:
#     print(f"  top result: {results[0]['text'][:100]}")

# ── Test 4: ChromaDB embedder ──────────────────────────────────────────────────
print("\n=== Embedder ===")
from knowledge.embedder import DrugEmbedder
embedder = DrugEmbedder(db_path="drug_db")
test_chunks = fda.get_label_chunks("warfarin")[:3]  # ← use fda instead of dd
n = embedder.upsert_chunks(test_chunks)
print(f"  upserted: {n} chunks")
results = embedder.query("warfarin bleeding risk")
print(f"  query results: {len(results)}")
if results:
    print(f"  top result: {results[0]['text'][:100]}")

# ── Test 5: OCR (uses a real image) ───────────────────────────────────────────
print("\n=== OCR ===")
from vision.ocr import run_ocr
try:
    ocr_result = run_ocr("bottle.HEIC", packaging_type="cylindrical")
    print(f"  drug_name:  {ocr_result.drug_name}")
    print(f"  dosage:     {ocr_result.dosage}")
    print(f"  confidence: {ocr_result.confidence}")
    print(f"  flags:      {ocr_result.hallucination_flags}")
except Exception as e:
    print(f"  OCR error: {e}")

print("\n=== All tests done ===")