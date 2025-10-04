from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def translate_pet_sound(analysis):
    """
    Takes the pet_analyzer output and returns a fun, AI-generated translation.
    """
    prompt = (f"You are a funny pet translator."
              f"A {analysis['mood']} {analysis['type']} made a sound lasting "
              f"Translate what this {analysis['type']} might be saying, in one short and humorous human sentence."
              )
    

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[
            {"role": "system", "content": "You are a funny pet translator."},
            {"role": "user", "content": prompt},
        ]
    )

    return response.choices[0].message.content.strip()
