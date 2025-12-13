import pandas as pd
import re
import os

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def clean_text(text):
    """
    Removes HTML tags, URLs, and extra whitespace.
    """
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)       # Remove HTML tags
    text = re.sub(r'http\S+', '', text)       # Remove URLs
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def ingest_jobs():
    """
    Reads jobs.csv, cleans text, creates metadata, saves to processed folder.
    """
    print("Loading Jobs data...")
    jobs_path = os.path.join(RAW_DATA_DIR, 'jobs.csv')
    
    if not os.path.exists(jobs_path):
        print(f"Error: {jobs_path} not found.")
        return

    df = pd.read_csv(jobs_path)
    
    print("Cleaning Jobs text...")
    # Combine title and description
    df['text'] = df.apply(
        lambda x: clean_text(f"{x.get('title', '')}. {x.get('description', '')}"), 
        axis=1
    )
    
    if 'job_id' in df.columns:
        df = df.rename(columns={'job_id': 'id'})
    else:
        df['id'] = df.index

    df_processed = df[['id', 'title', 'text']].copy()
    
    output_path = os.path.join(PROCESSED_DATA_DIR, 'jobs_processed.csv')
    df_processed.to_csv(output_path, index=False)
    print(f"Saved {len(df_processed)} jobs to {output_path}")

def ingest_resumes():
    """
    Reads resumes.csv, specifically adapted for the user's detailed dataset.
    """
    print("Loading Resumes data...")
    resumes_path = os.path.join(RAW_DATA_DIR, 'resumes.csv')

    if not os.path.exists(resumes_path):
        print(f"Error: {resumes_path} not found.")
        return

    df = pd.read_csv(resumes_path)
    
    df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True).str.strip()

    print("Cleaning Resumes text...")
    
    def build_resume_text(row):
        parts = []
        if pd.notna(row.get('job_position_name')):
            parts.append(f"Role: {row['job_position_name']}")
        if pd.notna(row.get('career_objective')):
            parts.append(f"Objective: {row['career_objective']}")
        if pd.notna(row.get('skills')):
            parts.append(f"Skills: {row['skills']}")
        if pd.notna(row.get('responsibilities')):
            parts.append(f"Experience: {row['responsibilities']}")
            
        return clean_text(" ".join(parts))

    df['text'] = df.apply(build_resume_text, axis=1)
    
    df['id'] = df.index

    df['Category'] = df.get('job_position_name', 'Unknown')
    
    df_processed = df[['id', 'Category', 'text']]
    
    output_path = os.path.join(PROCESSED_DATA_DIR, 'resumes_processed.csv')
    df_processed.to_csv(output_path, index=False)
    print(f"Saved {len(df_processed)} resumes to {output_path}")

if __name__ == "__main__":
    ingest_jobs()
    ingest_resumes()