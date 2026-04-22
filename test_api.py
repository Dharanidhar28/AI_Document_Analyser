import os
from dotenv import load_dotenv
load_dotenv()
import requests

url = 'https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2'
headers = {'Authorization': f'Bearer {os.getenv("HF_TOKEN")}'}
r = requests.post(url, headers=headers, json={'inputs': ['test']})
print("Status Code:", r.status_code)
print("Response:", r.text[:500])
