import mysql.connector
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()


def get_db_connection():
    """
    Create MySQL connection with auth plugin specified
    """
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "music_recommender"),
        auth_plugin='mysql_native_password'
    )


def save_song(song_name, artist, vector, album=None, duration=None):
    """
    Save song with metadata and latent vector to database
    vector: 256-dimensional numpy array from autoencoder
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # convert numpy array to bytes for storage
        vector_blob = vector.tobytes()

        query = "INSERT INTO songs (song_name, artist, album, duration, latent_vector) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(query, (song_name, artist, album, duration, vector_blob))
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f"Database error for {song_name}: {err}")
        raise


def load_all_songs():
    """
    Load all songs with metadata and vectors from database
    Returns: song_data (list of dicts), latent_matrix (numpy array)
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT song_name, artist, album, duration, latent_vector FROM songs"
    cursor.execute(query)
    results = cursor.fetchall()

    song_data = []
    vectors = []

    for name, artist, album, duration, blob in results:
        song_data.append({
            'name': name,
            'artist': artist if artist else 'Unknown',
            'album': album,
            'duration': duration
        })
        # reconstruct 256-dimensional numpy array from bytes
        vectors.append(np.frombuffer(blob, dtype=np.float32))

    cursor.close()
    conn.close()

    return song_data, np.array(vectors)


def search_song_by_name(partial_name):
    """
    Search for songs by partial name match
    Returns: list of matching songs with metadata
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT id, song_name, artist, album, duration FROM songs WHERE song_name LIKE %s"
    cursor.execute(query, (f"%{partial_name}%",))
    results = cursor.fetchall()

    songs = []
    for song_id, name, artist, album, duration in results:
        songs.append({
            'id': song_id,
            'name': name,
            'artist': artist if artist else 'Unknown',
            'album': album,
            'duration': duration
        })

    cursor.close()
    conn.close()
    return songs