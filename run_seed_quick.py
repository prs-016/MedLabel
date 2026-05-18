import sys, logging
sys.path.insert(0, "src")
import unittest.mock as mock
sys.modules['vision'] = mock.MagicMock()
sys.modules['vision.ocr'] = mock.MagicMock()
from pipeline import MedLabelPipeline
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
pipeline = MedLabelPipeline(db_path="drug_db")
pipeline.seed_database()
