import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = "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"#os.getenv("OPENAI_API_KEY")

def get_gpt_response(memory_variables, prompt):
    # Extract the history (messages) from the memory variables (dict)
    history = memory_variables.get("chat_history", [])

    # Send the chat history to OpenAI's API and stream the response
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  
        messages=history,
        stream=True
    )

    full_response = ""
    for chunk in response:
        chunk_message = chunk['choices'][0]['delta'].get('content', '')
        full_response += chunk_message
        yield chunk_message
    return history
