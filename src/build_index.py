import os
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from src.config import PROCESSED_DATA_DIR, INDEX_DIR, MODEL_NAME

def build_indices():
    # 1. Setup
    print(f"Loading AI Model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    os.makedirs(INDEX_DIR, exist_ok=True)

    # 2. Build Job Index
    jobs_path = os.path.join(PROCESSED_DATA_DIR, 'jobs_processed.csv')
    if os.path.exists(jobs_path):
        print("\n--- Processing Jobs ---")
        df_jobs = pd.read_csv(jobs_path)
        
        # Safety: ensure text column is string
        df_jobs['text'] = df_jobs['text'].fillna("").astype(str)
        
        print(f"Generating embeddings for {len(df_jobs)} jobs... (Sit tight, this takes time)")
        job_embeddings = model.encode(df_jobs['text'].tolist(), show_progress_bar=True)
        
        # Create FAISS Index
        d = job_embeddings.shape[1]
        job_index = faiss.IndexFlatL2(d)
        job_index.add(job_embeddings)
        
        # Save Index and Metadata
        faiss.write_index(job_index, os.path.join(INDEX_DIR, 'jobs.index'))
        df_jobs.to_pickle(os.path.join(INDEX_DIR, 'jobs_metadata.pkl'))
        print("✔ Jobs index saved.")
    
    # 3. Build Resume Index
    resumes_path = os.path.join(PROCESSED_DATA_DIR, 'resumes_processed.csv')
    if os.path.exists(resumes_path):
        print("\n--- Processing Resumes ---")
        df_resumes = pd.read_csv(resumes_path)
        
        df_resumes['text'] = df_resumes['text'].fillna("").astype(str)
        
        print(f"Generating embeddings for {len(df_resumes)} resumes...")
        resume_embeddings = model.encode(df_resumes['text'].tolist(), show_progress_bar=True)
        
        d = resume_embeddings.shape[1]
        resume_index = faiss.IndexFlatL2(d)
        resume_index.add(resume_embeddings)
        
        faiss.write_index(resume_index, os.path.join(INDEX_DIR, 'resumes.index'))
        df_resumes.to_pickle(os.path.join(INDEX_DIR, 'resumes_metadata.pkl'))
        print("✔ Resumes index saved.")

if __name__ == "__main__":
    build_indices()