import inspect
import json
import os
from dotenv import load_dotenv
import openai
import requests
import yfinance as yf
from pydantic import TypeAdapter
import signal
import sys

load_dotenv()

def get_symbol(company_name: str) -> str:
    """
    Retrieve the stock symbol for a specified company using the Yahoo Finance API.
    :param company: The name of the company for which to retrieve the stock symbol, e.g., 'Nvidia'.
    :output: The stock symbol for the specified company.
    """
    try:
        print(f"Looking up symbol for {company_name}...")  # Loading indicator
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {
            "q": company_name,
            "country": "United States",
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"
        }
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        if not data.get('quotes'):
            return f"No symbol found for company: {company_name}"
            
        return data['quotes'][0]['symbol']
    except Exception as e:
        return f"Error finding symbol: {str(e)}"

def get_stock_price(symbol: str) -> float:
    """
    Retrieve the current stock price for a specified company using the Yahoo Finance API.
    :param symbol: The stock symbol for the company, e.g., 'NVDA'.
    :output: The current stock price for the specified company.
    """
    try:
        print(f"Fetching stock price for {symbol}...")  # Loading indicator
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1d", interval="1m")
        
        if hist.empty:
            return {
                "error": f"No data found for symbol {symbol}. The stock might be delisted or invalid."
            }
            
        latest = hist.iloc[-1]
        return {
            "timestamp": str(latest.name),
            "open": float(latest["Open"]),
            "high": float(latest["High"]),
            "low": float(latest["Low"]),
            "close": float(latest["Close"]),
            "volume": int(latest["Volume"])
        }
    except Exception as e:
        return {
            "error": f"Error fetching stock price: {str(e)}"
        }

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": inspect.getdoc(get_stock_price),
            "parameters": TypeAdapter(get_stock_price).json_schema(),
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_symbol",
            "description": inspect.getdoc(get_symbol),
            "parameters": TypeAdapter(get_symbol).json_schema(),
        }
    }
]

FUNCTION_MAP = {
    "get_stock_price": get_stock_price,
    "get_symbol": get_symbol
}

def chat_completion(message: str, messages: list[dict]) -> str:
    client = openai.OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url=os.getenv("GROQ_BASE_URL")
    )
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        tools=tools
    )
    return response

messages = [
    {"role": "system", "content": """You are a professional stock market assistant. Follow these rules:
    1. Always format prices with currency symbols (e.g., $150.50, ¥500)
    2. Keep responses brief and focused on the financial data
    3. When reporting stock prices:
    - Highlight significant price changes
    - Include the currency and market status
    - Format large numbers with appropriate separators (e.g., 1,234,567)
    4. If there's an error, explain it simply and suggest alternatives
    5. Don't provide investment advice or predictions

    Example response formats:
    - "AAPL is currently trading at $175.23 (▲2.1%) | Market: Open"
    - "TSLA: $242.50 | Volume: 15.2M | Market: Closed"
    - "Error: Symbol not found. Did you mean to search for 'GOOGL' instead of 'GOOG'?"
    """}
]

def chat_logic(message: str) -> str:
    messages.append({"role": "user", "content": message})
    response = chat_completion(message, messages)
    first_choice = response.choices[0]
    finish_reason = first_choice.finish_reason
    while finish_reason != "stop":
        tool_call = first_choice.message.tool_calls[0]
        function_name = tool_call.function.name
        function = FUNCTION_MAP[function_name]
        function_args = json.loads(tool_call.function.arguments)
        result = function(**function_args)
        messages.append(first_choice.message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": json.dumps(result)
        })
        response = chat_completion(message, messages)
        first_choice = response.choices[0]
        finish_reason = first_choice.finish_reason

    return first_choice.message.content

def signal_handler(sig, frame):
    """Handle keyboard interrupt (Ctrl+C) gracefully"""
    print("\nGoodbye! Exiting program...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while True:
            print("Enter 'exit' to quit (or Ctrl+C)")
            message = input("Enter a message to get stock price: ")
            if message.lower() == 'exit':
                print("Goodbye! Exiting program...")
                break
                
            print("--------------------------------")
            print("--------------------------------")
            print(chat_logic(message))
            print("--------------------------------")
            print("--------------------------------")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Exiting program...")
    finally:
        sys.exit(0)
