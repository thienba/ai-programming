import chromadb
from chromadb.utils import embedding_functions
from wikipediaapi import Wikipedia
import os
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv()

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./data/chroma_db")
client.heartbeat()

# Initialize embedding function
embedding_function = embedding_functions.DefaultEmbeddingFunction()

def extract_topic_from_query(query):
    """Extract potential topic from user query"""
    # Initialize OpenAI client
    openai_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))
    
    # Create a prompt to extract the topic
    extraction_prompt = f"""
    Extract the main topic or person that needs to be researched from this query.
    Return ONLY the topic name that should be searched on Wikipedia, nothing else.
    
    Query: {query}
    """
    
    response = openai_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": extraction_prompt}],
    )
    
    topic = response.choices[0].message.content.strip()
    # Replace spaces with underscores
    topic = re.sub(r'\s+', '_', topic)
    
    return topic

def get_wikipedia_content(topic):
    """Get content from Wikipedia for a given topic"""
    wiki = Wikipedia('ba-huynh', 'en')
    page = wiki.page(topic)
    
    if page.exists():
        return page.text
    else:
        return None

def create_or_get_collection(collection_name):
    """Create a new collection or get existing one"""
    try:
        # Try to get existing collection
        collection = client.get_collection(name=collection_name, embedding_function=embedding_function)
        return collection
    except:
        # Create new collection if it doesn't exist
        return client.create_collection(name=collection_name, embedding_function=embedding_function)

def process_query(user_query):
    # Extract topic from query
    topic = extract_topic_from_query(user_query)
    
    # Create a collection name based on the topic
    collection_name = re.sub(r'[^a-zA-Z0-9]', '-', topic.lower())
    
    # Get Wikipedia content
    doc = get_wikipedia_content(topic)
    
    if not doc:
        return f"Could not find information about {topic} on Wikipedia."
    
    # Create or get collection
    collection = create_or_get_collection(collection_name)
    
    # Check if collection is empty and needs to be populated
    if collection.count() == 0:
        # Split document into paragraphs
        paragraphs = doc.split('\n\n')
        
        # Add paragraphs to collection
        for index, paragraph in enumerate(paragraphs):
            if paragraph.strip():  # Only add non-empty paragraphs
                collection.add(
                    ids=[str(index)],
                    documents=[paragraph],
                )
    
    # Query the collection
    results = collection.query(query_texts=[user_query], n_results=3)
    context = results["documents"][0]

    # Build prompt with retrieved context
    prompt = f"""
    Use the following CONTEXT to answer the QUESTION at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use an unbiased and journalistic tone.

    CONTEXT: {context}

    QUESTION: {user_query}
    """
    
    # Generate response
    openai_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))
    response = openai_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
    )
    
    return response.choices[0].message.content

query = input("Enter your question: ")
answer = process_query(query)
print("\nAnswer: ", answer)