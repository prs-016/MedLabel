import requests

class RxNormClient:
    def __init__(self):
        self.base_url = "https://rxnav.nlm.nih.gov/REST"

    def normalize_drug_name(self, raw_text):
        """
        Takes potentially messy OCR text and finds the standardized RxNorm name and RxCUI.
        """
        # Week 3 Task: Implement fuzzy name matching strategy
        # Documentation: https://rxnav.nlm.nih.gov/REST/rxcui.json?name=...
        params = {"name": raw_text}
        response = requests.get(f"{self.base_url}/rxcui.json", params=params)
        
        if response.status_code == 200:
            data = response.json()
            # extract RxCUI and canonical name
            return data
        return None

if __name__ == "__main__":
    client = RxNormClient()
    print("Normalizing 'Warfarin'...")
    print(client.normalize_drug_name("warfarin"))
