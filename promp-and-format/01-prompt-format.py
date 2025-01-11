from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

messages = [
    {
        "role": "user",
        "content": "What is France's capital?"
    },
    {
        "role": "assistant",
        "content": "France's capital is Paris."
    }]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False)
print(prompt)