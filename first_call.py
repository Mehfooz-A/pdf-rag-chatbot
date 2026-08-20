import os
from dotenv import load_dotenv
from google import genai

# Load the .env file so GEMINI_API_KEY becomes available
load_dotenv()

# Create a client using the key from the environment — never hardcoded
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Send a simple prompt and get a response back
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="In one sentence, explain what RAG (Retrieval-Augmented Generation) is."
)

print(response.text)