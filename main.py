from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import os
import requests
from database.db_connect import load_all_songs

@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the dataset from mysql
    global song_data, latent_matrix
    try:
        print("Connecting to mysql...")
        song_data, latent_matrix = load_all_songs()

        if len(song_data) > 0:
            print(f"Loaded {len(song_data)} songs from database")
        else:
            print("Warning: Database is empty.")

    except Exception as e:
        print(f"Error loading from database: {e}")

    yield

app = FastAPI(lifespan=lifespan)

# enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# global variables
song_data = []          # list of dicts with metadata
latent_matrix = None    # numpy array of all 256-dim vectors

class SongRequest(BaseModel):
    song_name: str


class RecommendationResponse(BaseModel):
    query_song: str
    recommendations: list

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('index.html')


@app.get("/songs")
async def get_all_songs():
    """Get list of all available songs with metadata"""
    if not song_data:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    return {
        "songs": song_data,
        "count": len(song_data)
    }


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_songs(request: SongRequest):
    """
    Get song recommendations using cosine similarity on latent vectors
    """
    # check if database data is loaded
    if latent_matrix is None or not song_data:
        raise HTTPException(status_code=500, detail="Database data not loaded")

    query_name = request.song_name.strip()

    # find matching songs]
    try:
        query_index = next(i for i, song in enumerate(song_data) if query_name.lower() in song['name'].lower())
        matched_song = song_data[query_index]
    except StopIteration:
        raise HTTPException(
            status_code=404,
            detail=f"No song found containing '{query_name}'"
        )

    # extract the 256 dim latent vector for the query song
    query_vector = latent_matrix[query_index].reshape(1, -1)

    # compute cosine similarity against all songs
    similarities = cosine_similarity(query_vector, latent_matrix)

    # create list of (index, similarity_score) tuples
    sim_scores = list(enumerate(similarities.flatten()))

    # sort by similarity (descending)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    # format recommendations with metadata from database
    recommendations = []
    for idx, score in sim_scores:
        song = song_data[idx]
        rec = {
            "name": song['name'],
            "artist": song.get('artist', 'Unknown'),
            "similarity": f"{score:.4f}"
        }
        # add album if available
        if song.get('album'):
            rec['album'] = song['album']
        recommendations.append(rec)

    return {
        "query_song": f"{matched_song['name']} - {matched_song.get('artist', 'Unknown')}",
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


