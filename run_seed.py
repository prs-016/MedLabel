import sys, logging
sys.path.insert(0, "src")
from pipeline import MedLabelPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

pipeline = MedLabelPipeline(db_path="drug_db")
pipeline.seed_database()