import openai
from dotenv import load_dotenv
import os
from together import Together
from huggingface_hub import InferenceClient
import copy

client = InferenceClient(api_key="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")


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

def intergrate_context(intergrate_list):
    all_content = ""
    for content in intergrate_list:
        all_content += f'One Source is : {content} and'

    prompt = [
        {
            "role": "user",
            "content": all_content + "You need to read current source text and summary of previous source text, and generate a summary to include them both, cover all important infomation",
        }
    ]
    
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct",
        messages=prompt,
        max_tokens=500,
        stream=False
    )
    return response['choices'][0]['message']['content'] 

def get_llama_response(memory_variables, current_retrived_docs, prompt):
    history = memory_variables.get("chat_history", [])
    temp_history = copy.deepcopy(history)
    temp_history.append(current_retrived_docs)
    
    #print("ORIGIN HISTORY IS : ",history)
    #print("TEMP HISTORY IS : ",temp_history)
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct",
        messages=temp_history,
        max_tokens=500,
        stream=True
    )
    #print("RESPONSE LÀ : ", response)
    full_response = ""
    stepi = 0
    for chunk in response:
        chunk_message = chunk['choices'][0]['delta'].get('content', '')
        #stepi += 1
        #print("AT STEP : ",stepi, "CHUNK MESSAGE: ", chunk_message)
        full_response += chunk_message
        yield chunk_message
    
    #print("FULL RESPONSE: ", full_response)
    return history