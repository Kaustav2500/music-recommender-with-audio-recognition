import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI()

# enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load the dataset with latent vectors
DATA_PATH = "songs_with_features.pkl"

# global variable to store the dataframe
global df


class SongRequest(BaseModel):
    song_name: str


class RecommendationResponse(BaseModel):
    query_song: str
    recommendations: list


@app.on_event("startup")
async def load_data():
    """Load the dataset on startup"""
    global df
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_pickle(DATA_PATH)
            print(f"Loaded dataset with {len(df)} songs")
        else:
            print(f"Warning: Dataset not found at {DATA_PATH}")
            print("Creating sample dataset for demonstration...")
            # create sample data for demonstration
            df = pd.DataFrame({
                'file_name': ['Song A', 'Song B', 'Song C', 'Song D', 'Song E'],
                'latent_vector': [np.random.rand(256) for _ in range(5)]
            })
    except Exception as e:
        print(f"Error loading dataset: {e}")
        df = None

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('index.html')


@app.get("/songs")
async def get_all_songs():
    """Get list of all available songs"""
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    return {
        "songs": df['file_name'].tolist(),
        "count": len(df)
    }


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_songs(request: SongRequest):
    """Get song recommendations based on input song name"""
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    query_name = request.song_name.strip()

    # find matching songs
    matches = df[df['file_name'].str.contains(query_name, case=False, na=False)]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No song found containing '{query_name}'"
        )

    # get the first match
    query_index = matches.index[0]
    matched_full_name = df.loc[query_index, 'file_name']

    # extract the latent vector for the query song
    query_vector = df['latent_vector'].iloc[query_index].reshape(1, -1)

    # stack all vectors into a matrix
    all_vectors = np.stack(df['latent_vector'].values)

    # compute cosine similarity
    similarities = cosine_similarity(query_vector, all_vectors)

    # create list of (index, similarity_score) tuples
    sim_scores = list(enumerate(similarities.flatten()))

    # sort by similarity (descending) and exclude the query song itself
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    # format recommendations
    recommendations = []
    for idx, score in sim_scores:
        recommendations.append({
            "name": df['file_name'].iloc[idx],
            "similarity": f"{score:.4f}"
        })

    return {
        "query_song": matched_full_name,
        "recommendations": recommendations
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

