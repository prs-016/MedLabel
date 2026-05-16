# 1. Import the library
import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()
# 2. Connect to your workflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

# 3. Run your workflow on an image
result = client.run_workflow(
    workspace_name="akshays-workspace-wnzzr",
    workflow_id="router",
    images={
        "image": "YOUR_IMAGE.jpg" # Path to your image file
    },
    use_cache=True # Speeds up repeated requests
)

# 4. Get your results
print("Shape: ", result[0]["model_output"]["top"])
print("Confidence: ", result[0]["model_output"]["confidence"])