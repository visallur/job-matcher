import faiss
import numpy as np
import pandas as pd
from embeddings import get_embedding

df = pd.read_csv("resume_lookup.csv")
index = faiss.read_index("resume_index.faiss")

def match_candidates(job_description, top_k=5):
    job_vec = np.array([get_embedding(job_description)])
    distances, indices = index.search(job_vec, top_k)
    results = df.iloc[indices[0]].copy()
    results["score"] = distances[0]
    return results