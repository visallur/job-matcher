import faiss
import numpy as np
import pandas as pd
from embeddings import get_embedding

def build_index(resume_path="data/resumes.csv"):
    df = pd.read_csv(resume_path)
    texts = df["resume_text"].tolist()

    embeddings = np.array([get_embedding(t) for t in texts])

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, "resume_index.faiss")
    df.to_csv("resume_lookup.csv", index=False)

    print("FAISS index built.")