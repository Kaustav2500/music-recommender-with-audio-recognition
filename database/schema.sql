CREATE DATABASE IF NOT EXISTS music_recommender;
USE music_recommender;

CREATE TABLE IF NOT EXISTS songs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    song_name VARCHAR(255) NOT NULL,
    spotify_id VARCHAR(255),
    artist VARCHAR(255),
    latent_vector LONGBLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);