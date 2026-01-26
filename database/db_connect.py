import mysql.connector
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()


def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "music_recommender")
    )


def save_song(song_name, vector):
    conn = get_db_connection()
    cursor = conn.cursor()
    # convert numpy array to bytes
    vector_blob = vector.tobytes()

    query = "INSERT INTO songs (song_name, latent_vector) VALUES (%s, %s)"
    cursor.execute(query, (song_name, vector_blob))
    conn.commit()
    cursor.close()
    conn.close()


def load_all_songs():
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT song_name, latent_vector FROM songs"
    cursor.execute(query)
    results = cursor.fetchall()

    songs = []
    vectors = []

    for name, blob in results:
        songs.append(name)
        # reconstruct numpy array from bytes 
        vectors.append(np.frombuffer(blob, dtype=np.float32))

    cursor.close()
    conn.close()

    return songs, np.array(vectors)