import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Embedding Model (Local)
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Model nhúng miễn phí
    CHUNK_SIZE = 300                      # Kích thước chunk
    
    # GPT Model từ g4f (miễn phí)
    GPT_MODEL = "gpt-4-mini"              # Model không cần API key
    MAX_TOKENS = 500                      # Giới hạn token response
    
    # Data Paths
    DATA_DIR = "data"
    FAISS_INDEX_PATH = "faiss_index.bin"   # File lưu FAISS index
    METADATA_PATH = "metadata.pkl"         # File lưu metadata
    
    # Language
    DEFAULT_LANG = "vi"                    # Ngôn ngữ mặc định