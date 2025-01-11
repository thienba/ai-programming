import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4o-mini")

# Dùng tiktoken để encode văn bản thành 1 chuối token
tokens = encoding.encode("I enjoy watching Japanese Action Movies.")
print(tokens)
print(len(tokens))