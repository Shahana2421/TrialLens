
import os
from dotenv import load_dotenv
from groq import Groq

# Step 1: Load environment variables
load_dotenv()

# Step 2: Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Groq API key not found. Check your .env file.")

# Step 3: Create Groq client
client = Groq(api_key=api_key)

# Step 4: Send a test question
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "Explain Type 2 diabetes in simple words."
        }
    ],
    temperature=0.3
)

# Step 5: Display response
answer = response.choices[0].message.content

print("Groq API connection successful!")
print("\nAI Response:")
print(answer)
