import os
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import INDEX_DIR, MODEL_NAME

class JobMatcher:
    def __init__(self):
        print("Loading model...")
        self.model = SentenceTransformer(MODEL_NAME)
        
        print("Loading indices...")
        self.job_index = faiss.read_index(os.path.join(INDEX_DIR, 'jobs.index'))
        self.resume_index = faiss.read_index(os.path.join(INDEX_DIR, 'resumes.index'))
        

        self.jobs_df = pd.read_pickle(os.path.join(INDEX_DIR, 'jobs_metadata.pkl'))
        self.resumes_df = pd.read_pickle(os.path.join(INDEX_DIR, 'resumes_metadata.pkl'))
        print("✔ System Ready!")

    def search_jobs_for_resume(self, resume_id, k=5):
        """
        Given a resume ID, find the top K matching jobs.
        """
        resume_row = self.resumes_df[self.resumes_df['id'] == resume_id]
        if resume_row.empty:
            return f"Error: Resume ID {resume_id} not found."
        
        resume_text = resume_row.iloc[0]['text']
        print(f"\nSearching for Resume ID: {resume_id}")
        
        query_vector = self.model.encode([resume_text])
        
        # Search FAISS
        distances, indices = self.job_index.search(query_vector, k)
        
        # Format Results
        results = []
        for i, idx in enumerate(indices[0]):
            match = self.jobs_df.iloc[idx]
            results.append({
                "rank": i+1,
                "score": f"{distances[0][i]:.4f}", 
                "title": match['title'],
                "text": match['text'][:200] + "..." 
            })
        return results

    def search_resumes_for_query(self, query_text, k=5):
        """
        Free text search: 'Find me a Python developer'
        """
        print(f"\nSearching for: '{query_text}'")
        query_vector = self.model.encode([query_text])
        
        distances, indices = self.resume_index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            match = self.resumes_df.iloc[idx]
            results.append({
                "rank": i+1,
                "score": f"{distances[0][i]:.4f}",
                "category": match['Category'],
                "text": match['text'][:200] + "..."
            })
        return results

# --- Testing Section ---
if __name__ == "__main__":
    matcher = JobMatcher()
    
    print("\n--- TEST: Finding candidates for 'Data Scientist with Python' ---")
    candidates = matcher.search_resumes_for_query("Data Scientist machine learning Python", k=3)
    for c in candidates:
        print(c)

    first_resume_id = matcher.resumes_df.iloc[0]['id']
    print(f"\n--- TEST: Matching Jobs for Resume #{first_resume_id} ---")
    jobs = matcher.search_jobs_for_resume(first_resume_id, k=3)
    for j in jobs:
        print(j)