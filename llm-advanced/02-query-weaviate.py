# Import các thư viện cần thiết
import pandas as pd
import weaviate
from weaviate.embedded import EmbeddedOptions
from weaviate.classes.config import Configure, Property, DataType, Tokenization


# Cần chạy docker container cho model embedding:
# docker run -itp "8000:8080" semitechnologies/transformers-inference:sentence-transformers-multi-qa-MiniLM-L6-cos-v1
embedded_options = EmbeddedOptions(
    additional_env_vars={
        # Kích hoạt các module cần thiết: text2vec-transformers
        "ENABLE_MODULES": "backup-filesystem,text2vec-transformers",
        "BACKUP_FILESYSTEM_PATH": "/tmp/backups",  # Chỉ định thư mục backup
        "LOG_LEVEL": "panic",  # Chỉ định level log, chỉ log khi có lỗi
        "TRANSFORMERS_INFERENCE_API": "http://localhost:8000"  # API của model embedding
    },
    persistence_data_path="data",  # Thư mục lưu dữ liệu
)

# Khởi tạo Weaviate và kết nối
vector_db_client = weaviate.WeaviateClient(
    embedded_options=embedded_options
)
vector_db_client.connect()

COLLECTION_NAME = "MovieCollection"

movie_collection = vector_db_client.collections.get(COLLECTION_NAME)

# 1. Tìm kiếm theo ngữ nghĩa với near_text
response = movie_collection.query.near_text(
    query="funny children story", limit=5
)

# 2. Tìm kiếm theo vector với near_vector
# response = movie_collection.query.hybrid(
#     query="thriller",
#     alpha=0.5,
#     limit=5
# )

# 3. Tìm kiếm theo vector với near_vector
# object_id = "ff59a0d2-134a-4559-bfcf-8ebaa89416d7"# ID cho "Hard Kill"
# response = movie_collection.query.near_object(near_object=object_id, limit=5)

# In kết quả
for result in response.objects:
    movie = result.properties
    print("Title: {}, Year: {}, Genre: {}".format(movie['title'], movie['thumbnail'], movie['genres']))

vector_db_client.close()