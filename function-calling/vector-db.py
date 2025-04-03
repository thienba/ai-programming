import numpy as np
from sentence_transformers import SentenceTransformer
class SimpleVectorDB:
  def __init__(self):
    self.items = []
    self.embeddings = []

  def add_item(self, item, embedding):
    self.items.append(item)
    self.embeddings.append(embedding)

  def find_similar(self, query_embedding, top_k: int = 3):
      if not self.embeddings:
          return []

      similarities = np.dot(self.embeddings, query_embedding) / (
          np.linalg.norm(self.embeddings, axis=1) *
          np.linalg.norm(query_embedding)
      ) # cosine similarity

      top_indices = np.argsort(similarities)[::-1][:top_k]
      return [(self.items[i], similarities[i]) for i in top_indices]
  
model = SentenceTransformer('all-MiniLM-L12-v2')

db = SimpleVectorDB()

dictionary = ["Cat", "Dog", "Fish", "Bird",
         "Elephant", "King", "Knight", "Man", "Woman"]

for item in dictionary:
    embedding = model.encode(item)
    db.add_item(item, embedding)

def search(query: str, model, db):
    # Perform a query
    query_embedding = model.encode(query)

    # Find similar items
    results = db.find_similar(query_embedding)

    # Print results
    print(f"Query: {query}")
    print("Similar items:")
    for item, similarity in results:
        print(f"  {item}: {similarity:.4f}")

search("Queen", model, db)
search("Human", model, db)

search("Kitten", model, db)
search("Parrot", model, db)
search("Doggy", model, db)