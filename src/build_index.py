from __future__ import annotations
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from pydantic import BaseModel
import pandas as pd
import os

from sentence_transformers import SentenceTransformer

class TextChunk(BaseModel):
    page_num: int
    text: str

@dataclass
class TextIndex:
    model_name: str
    embeddings: np.ndarray  # (N, d)
    items: List[TextChunk]

    def query(self, q: str, top_k: int = 5) -> List[Tuple[TextChunk, float]]:
        # cosine similarity search with sklearn or faiss
        qv = _encode(self.model_name, [q])[0]
        sims = cosine_sim_matrix(qv.reshape(1,-1), self.embeddings)[0]
        idxs = sims.argsort()[::-1][:top_k]
        return [(self.items[i], float(sims[i])) for i in idxs]

def _encode(model_name: str, texts: List[str]) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embs, dtype="float32")

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (A @ B.T)

def build_text_index(chunks: List[TextChunk], model_name: str) -> TextIndex:
    texts = [c.text for c in chunks]
    embs = _encode(model_name, texts)
    return TextIndex(model_name=model_name, embeddings=embs, items=chunks)

def tables_summary(tables: List[pd.DataFrame]) -> List[str]:
    out = []
    for i, df in enumerate(tables, start=1):
        cols = ", ".join(map(str, df.columns.tolist()))
        out.append(f"Table {i} with {len(df)} rows, columns: {cols}")
    return out
