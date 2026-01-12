# test_config.py
import os
from dotenv import load_dotenv

load_dotenv()

print("Azure OpenAI Configuration:")
print(f"Key: {os.getenv('AZURE_OPENAI_KEY')[:10]}..." if os.getenv('AZURE_OPENAI_KEY') else "NOT SET")
print(f"Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
print(f"Deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT')}")