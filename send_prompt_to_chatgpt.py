import os
import openai

# Option 1: Set your API key directly (not recommended for production)
# openai.api_key = "YOUR_API_KEY_HERE"

# Option 2 (recommended): Store your API key as an environment variable
# Make sure you have set the environment variable OPENAI_API_KEY beforehand
openai.api_key = os.environ.get("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("Please set your OpenAI API key in the OPENAI_API_KEY environment variable.")

# Define the prompt you want to send to ChatGPT
prompt = "Tell me a joke about technology."

# Create a chat completion using the ChatGPT (gpt-3.5-turbo) model
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        # Optionally, you can add a system message to set the behavior of the assistant
        {"role": "system", "content": "You are a friendly assistant."},
        {"role": "user", "content": prompt}
    ]
)

# Extract and print the assistant's reply
reply = response['choices'][0]['message']['content']
print("ChatGPT says:")
print(reply)
