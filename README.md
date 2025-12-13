# JobMatcher

Simple project to match job descriptions to resumes using sentence-transformer embeddings and FAISS.

Project layout
- data/
  - resumes.csv (your resume dataset)
  - jobs.csv (your job dataset)
- src/
  - api.py — FastAPI application
  - search.py — loads FAISS index and resumes lookup
  - index_faiss.py — index builder
  - embeddings.py — sentence-transformers wrapper
  - ui.py — Streamlit UI to query the API
- resume_index.faiss — FAISS index file (generated)
- resume_lookup.csv — CSV with resumes saved when index built

Quick setup
1. Create a venv and install:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Place your CSVs:
   - Put `resumes.csv` into `data/resumes.csv`. The resumes CSV must contain a column named `resume_text`.
   - Put `jobs.csv` into `data/jobs.csv` if you plan to use it.

3. Build the FAISS index:
   python build_index.py
   This will create `resume_index.faiss` and `resume_lookup.csv` in the repo root.

4. Run the API:
   uvicorn src.api:app --reload --port 8000

   Note: because src is a package, running `uvicorn src.api:app` from the repo root works.

5. Run the UI (in another terminal):
   streamlit run src/ui.py
