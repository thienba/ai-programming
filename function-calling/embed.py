from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L12-v2')

embeddings = model.encode('Hello, world!')

print(embeddings)