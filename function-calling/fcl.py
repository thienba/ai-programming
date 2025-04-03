import os
from openai import OpenAI
import json

def get_current_weather(location: str, unit: str):
    """Get the current weather in a given location"""
    weather = {
        "location": location,
        "temperature": 75,
        "unit": unit,
    }
    return weather

def get_stock_price(symbol: str):
    pass

def view_website(url: str):
    pass

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The location to get the weather for"},
                    "unit": {"type": "string", "description": "The unit to get the weather in"}
                },
                "required": ["location", "unit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "The symbol to get the stock price for"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "view_website",
            "description": "View a website",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to view"}
                },
                "required": ["url"]
            }
        }
    }
]

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))

messages = [
    {"role": "system", "content": "You are a helpful assistant that can answer questions and perform tasks. When checking weather, always specify both the location and unit (use 'celsius' or 'fahrenheit')."},
    {"role": "user", "content": "What is the weather in Tokyo? With fahrenheit."}
]

MODEL_USAGE = "llama3-8b-8192"

response = client.chat.completions.create(
    model=MODEL_USAGE,
    messages=messages,
    tools=tools
)

if response.choices[0].message.tool_calls[0].function.name == "get_current_weather":
    args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    print("args:  ", args)
    weather_response = get_current_weather(**args)
    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "tool_call_id": response.choices[0].message.tool_calls[0].id,
        "name": "get_current_weather",
        "content": json.dumps(weather_response)
    })
    final_response = client.chat.completions.create(
        model=MODEL_USAGE,
        messages=messages
    )
    print(final_response.choices[0].message.content)
    exit()
elif response.choices[0].message.tool_calls[0].function.name == "get_stock_price":
    args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    stock_price_response = get_stock_price(**args)
    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "tool_call_id": response.choices[0].message.tool_calls[0].id,
        "name": "get_stock_price",
        "content": json.dumps(stock_price_response)
    })
elif response.choices[0].message.tool_calls[0].function.name == "view_website":
    args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    view_website_response = view_website(**args)
    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "tool_call_id": response.choices[0].message.tool_calls[0].id,
        "name": "view_website",
        "content": json.dumps(view_website_response)
    })

print(messages)
