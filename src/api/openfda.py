import requests
import os
from dotenv import load_dotenv

load_dotenv()

class OpenFDAClient:
    def __init__(self):
        self.api_key = os.getenv("OPENFDA_API_KEY")
        self.base_url = "https://api.fda.gov/drug"

    def fetch_drug_label(self, drug_name):
        """
        Queries openFDA /drug/label for official documentation.
        """
        # Week 3 Task: Implement robust query and JSON parsing
        params = {
            "search": f'openfda.brand_name:"{drug_name}"',
            "limit": 1
        }
        if self.api_key:
            params["api_key"] = self.api_key
            
        response = requests.get(f"{self.base_url}/label.json", params=params)
        return response.json() if response.status_code == 200 else None

    def fetch_adverse_events(self, drug_name):
        """
        Queries openFDA /drug/event for reported side effects.
        """
        # Week 4 Task: Implement tool logic for Agent
        params = {
            "search": f'patient.drug.medicinalproduct:"{drug_name}"',
            "limit": 5
        }
        if self.api_key:
            params["api_key"] = self.api_key

        response = requests.get(f"{self.base_url}/event.json", params=params)
        return response.json() if response.status_code == 200 else None

if __name__ == "__main__":
    client = OpenFDAClient()
    print("Testing openFDA API Key for Tylenol...")
    res = client.fetch_drug_label("Tylenol")
    if res:
        print("Success! FDA Label found.")
    else:
        print("Failed to fetch label (check API key or name).")
