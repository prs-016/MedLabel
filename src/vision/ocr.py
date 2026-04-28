from paddleocr import PaddleOCR
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class HybridOCR:
    def __init__(self):
        # Path A: PaddleOCR (Standard for flat labels)
        self.paddle_ocr = PaddleOCR(lang='en', use_angle_cls=True)
        
        # Path B: Gemini Vision (Best for curved/cylindrical bottles)
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.genai_model = genai.GenerativeModel('gemini-1.5-flash')

    def extract_from_flat(self, image_path):
        """
        Uses PaddleOCR to extract text from flat surfaces.
        """
        # Week 2 Task: Implement PaddleOCR processing and line merging
        result = self.paddle_ocr.ocr(image_path, cls=True)
        raw_text = ""
        # logic to merge lines...
        return "Acetaminophen 500mg - Extracted via PaddleOCR"

    def extract_from_cylindrical(self, image):
        """
        Uses Gemini Vision to read curved text from pill bottles.
        """
        # Path B avoids the cylindrical unwarp problem by using VLM
        prompt = "Read the medicine label on this bottle and return the text as JSON."
        # response = self.genai_model.generate_content([prompt, image])
        # return response.text
        return "Ibuprofen 200mg - Simulated Gemini Vision Extract"

if __name__ == "__main__":
    ocr = HybridOCR()
    print("Hybrid OCR Engine initialized.")
