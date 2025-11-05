import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env in the repo root (if present)
load_dotenv()

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("DOWNSTREAM_URL", "https://ellm.nrp-nautilus.io/v1"),
)

completion = client.chat.completions.create(
    model="gemma3",
    messages=[
        {"role": "system", "content": "Talk like a pirate."},
        {
            "role": "user",
            "content": "How do I check if a Python object is an instance of a class?",
        },
    ],
)

print(completion.choices[0].message.content)