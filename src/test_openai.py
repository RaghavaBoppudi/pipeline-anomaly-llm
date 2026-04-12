from openai import OpenAI
from dotenv import load_dotenv

# Loads the API key from .env file
# Key never appears in code — security best practice
load_dotenv()

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",       # Cheaper than gpt-4o, sufficient for structured tasks
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Say hello in exactly 5 words."
        }
    ],
    temperature=0.7            # We'll lower this later for consistent explanations
)

content = response.choices[0].message.content
finish_reason = response.choices[0].finish_reason
tokens_used = response.usage.total_tokens

print(f"Response: {content}")
print(f"Finish reason: {finish_reason}")
print(f"Tokens used: {tokens_used}")

