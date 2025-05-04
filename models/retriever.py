import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from config import Config
from typing import List, Dict
import os

class FAISSRetriever:
    def __init__(self):
        self.encoder = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.index = faiss.IndexFlatL2(self.encoder.get_sentence_embedding_dimension())
        self.metadata = []
        
        if os.path.exists(Config.FAISS_INDEX_PATH):
            self._load_index()

    def add_documents(self, documents: List[Dict]):
        """Thêm dữ liệu vào FAISS index"""
        contents = [doc["content"] for doc in documents]
        embeddings = self.encoder.encode(contents, show_progress_bar=True)
        
        self.index.add(np.array(embeddings).astype('float32'))
        self.metadata.extend(documents)
        self._save_index()

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Tìm kiếm semantic"""
        query_embed = self.encoder.encode([query])
        distances, indices = self.index.search(np.array(query_embed).astype('float32'), top_k)
        
        return [
            {
                **self.metadata[i],
                "score": float(1/(1 + d))  # Convert distance to similarity
            }
            for d, i in zip(distances[0], indices[0]) if i != -1
        ]

    def _save_index(self):
        """Lưu index ra file"""
        faiss.write_index(self.index, Config.FAISS_INDEX_PATH)
        with open(Config.METADATA_PATH, 'wb') as f:
            pickle.dump(self.metadata, f)

    def _load_index(self):
        """Đọc index từ file"""
        self.index = faiss.read_index(Config.FAISS_INDEX_PATH)
        with open(Config.METADATA_PATH, 'rb') as f:
            self.metadata = pickle.load(f)