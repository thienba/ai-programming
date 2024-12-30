from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

model_inputs = tokenizer('I really like programming', return_tensors='pt')

greedy_output = model.generate(**model_inputs, max_new_tokens=40)

print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))