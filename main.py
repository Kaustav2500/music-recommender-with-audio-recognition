from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os
import requests
from database.db_connect import load_all_songs

app = FastAPI()

# enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# global variables
song_names = []
latent_matrix = None


class SongRequest(BaseModel):
    song_name: str


class RecommendationResponse(BaseModel):
    query_song: str
    recommendations: list


@app.on_event("startup")
async def load_data():
    """Load the dataset from MySQL on startup"""
    global song_names, latent_matrix
    try:
        print("Connecting to MySQL...")
        song_names, latent_matrix = load_all_songs()

        if len(song_names) > 0:
            print(f"Loaded {len(song_names)} songs from database")
        else:
            print("Warning: Database is empty.")

    except Exception as e:
        print(f"Error loading from database: {e}")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('index.html')


@app.get("/songs")
async def get_all_songs():
    """Get list of all available songs"""
    if not song_names:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    return {
        "songs": song_names,
        "count": len(song_names)
    }


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_songs(request: SongRequest):
    """Get song recommendations based on input song name"""
    # check if database data is loaded
    if latent_matrix is None or not song_names:
        raise HTTPException(status_code=500, detail="Database data not loaded")

    query_name = request.song_name.strip()

    # find matching songs
    try:
        query_index = next(i for i, name in enumerate(song_names) if query_name.lower() in name.lower())
        matched_full_name = song_names[query_index]
    except StopIteration:
        raise HTTPException(
            status_code=404,
            detail=f"No song found containing '{query_name}'"
        )

    # extract the latent vector
    query_vector = latent_matrix[query_index].reshape(1, -1)

    # compute cosine similarity
    similarities = cosine_similarity(query_vector, latent_matrix)

    # create list of (index, similarity_score) tuples
    sim_scores = list(enumerate(similarities.flatten()))

    # sort by similarity (descending)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    # format recommendations
    recommendations = []
    for idx, score in sim_scores:
        recommendations.append({
            "name": song_names[idx],
            "similarity": f"{score:.4f}"
        })

    return {
        "query_song": matched_full_name,
        "recommendations": recommendations
    }


@app.get("/search")
def search(q: str):
    """Search iTunes API for music tracks"""
    url = "https://itunes.apple.com/search"
    params = {"term": q, "entity": "musicTrack", "limit": 10}
    r = requests.get(url, params=params).json()

    return r["results"]


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

