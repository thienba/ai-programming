import gradio as gr
import weaviate
from weaviate.embedded import EmbeddedOptions

embedded_options = EmbeddedOptions(
    additional_env_vars={
        "ENABLE_MODULES": "backup-filesystem,text2vec-transformers",
        "BACKUP_FILESYSTEM_PATH": "/tmp/backups",
        "LOG_LEVEL": "panic",
        "TRANSFORMERS_INFERENCE_API": "http://localhost:8000"
    },
    persistence_data_path="data",
)

vector_db_client = weaviate.WeaviateClient(
    embedded_options=embedded_options
)
vector_db_client.connect()

# Cấu hình tên collection
COLLECTION_NAME = "MovieCollection"

def search_movie(query):
    movie_collection = vector_db_client.collections.get(COLLECTION_NAME)
    response = movie_collection.query.near_text(query=query, limit=10)
    
    # Trả về thumbnail và title của các phim liên quan
    results = []
    for movie in response.objects:
        # Tuple format: (thumbnail, title) for gallery
        movie_tuple = (movie.properties['thumbnail'], movie.properties['title'])
        results.append(movie_tuple)
    return results

with gr.Blocks(title="Weaviate Search") as interface:
    gr.Markdown("Enter a search query:")
    query = gr.Textbox(label="Search Query")
    search = gr.Button("Search")
    gallery = gr.Gallery(label="Movies", show_label=False, columns=[5], rows=[3], object_fit="contain", height="auto")
    search.click(fn=search_movie, inputs=query, outputs=gallery)

interface.queue().launch()

vector_db_client.close()