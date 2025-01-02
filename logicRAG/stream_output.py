import openai
from dotenv import load_dotenv
import os
from together import Together

client = Together(api_key="LA-4069dea8c269422ab75127cb874bf69fa4824d2ec85d4505928973f55a2c321c")
#load_dotenv()
#openai.api_key = os.getenv("OPENAI_API_KEY")

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

def get_llama_response(memory_variables, prompt):
    history = memory_variables.get("chat_history", [])
    
    print(history.shape)
    # response = client.chat.completions.create(
    #     model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    #     messages= [
    #         {
    #     "role": "user",
    #     "content": [
    #         {"type": "text", "text": query}, #query
    #         {
    #         "type": "image_url",
    #         "image_url": {
    #             "url": "",#"data:image/jpeg;base64,{returned_page}", #retrieved page image
    #         },
    #         }, ]
    #         }  
    #     ]
    # )

    # # full_response = ""
    # # for chunk in response:
    # #     chunk_message = chunk['choices'][0]['delta'].get('content', '')
    # #     full_response += chunk_message
    # #     yield chunk_message
    # return history