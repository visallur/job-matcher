from fastapi import FastAPI
from search import match_candidates

app = FastAPI()

@app.post("/match")
async def get_matches(job_description: str):
    results = match_candidates(job_description)
    return results.to_dict(orient="records")