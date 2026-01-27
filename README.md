# Music Recommendation System

This branch transitions the project from a local file-based architecture to a scalable, database-driven system. It integrates the iTunes Search API for automated data acquisition and uses a MySQL backend for persistent storage of song metadata and latent vectors.

---

## Project Structure
```
project/
├── .env                      # Database credentials 
├── .gitignore                # Version control exclusions
├── README.md                 # Project documentation
├── main.py                   # FastAPI server entry point 
├── requirements.txt          # Project dependencies 
├── database/
│   ├── db_connect.py         # MySQL connection logic
│   └── schema.sql            # Database table definitions
├── static/                   # Frontend web files
│   ├── index.html
│   ├── style.css
│   └── script.js
├── src/                      # Backend processing scripts
│   ├── preprocessing.py      # iTunes download & audio processing
│   ├── audiopipeline.py      # PyTorch dataset & loaders
│   ├── autoencoder.py        # CNN model definition & training
│   ├── extract_features.py   # Vector generation & DB sync
│   ├── recommender.py        # CLI recommendation tool
│   └── check_db.py           # Database status utility
├── data/                     # Audio previews and cache
└── models/
    └── audio_autoencoder.pth # Trained model weights
```

---

## Features

- **Database-Driven Architecture**: Persistent storage of song metadata and feature vectors in MySQL
- **Automated Data Collection**: iTunes Search API integration for seamless music data acquisition
- **Deep Learning Model**: CNN-based autoencoder for extracting meaningful audio features
- **Cosine Similarity Matching**: Fast recommendation engine using 256-dimensional latent vectors
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **Modern Web Interface**: Clean, responsive frontend for user interaction

---

## Setup Instructions

### 1. Database Configuration

- Ensure MySQL is running on your system
- Execute the SQL commands in `database/schema.sql` to create the `music_recommender` database and `songs` table
- Create a `.env` file in the project root with your credentials:
```env
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=localhost
DB_NAME=music_recommender
```

### 2. Backend Setup
```bash
# Install required libraries
pip install -r requirements.txt

# Start the FastAPI server
python main.py
```

The backend will be accessible at `http://localhost:8000`

### 3. Data Pipeline Execution

Run these scripts in order to populate the system:

1. **`src/preprocessing.py`**: Searches iTunes for keywords (e.g., "Weeknd", "KK"), downloads 30s previews, and generates spectrograms
2. **`src/autoencoder.py`**: Trains the CNN model using the downloaded audio data
3. **`src/extract_features.py`**: Generates 256-dimensional latent vectors and synchronizes them with MySQL metadata
4. **`src/check_db.py`**: Verify the song count in your database

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the primary web interface |
| `GET` | `/songs` | Retrieves a list of all songs stored in the database |
| `POST` | `/recommend` | Returns the top 5 similar songs based on cosine similarity |

### Example API Request

**Endpoint**: `POST /recommend`
```json
{
    "song_name": "Starboy"
}
```

**Response**:
```json
{
    "recommendations": [
        {
            "song_name": "I Feel It Coming",
            "artist": "The Weeknd",
            "similarity_score": 0.94
        },
        ...
    ]
}
```

---

## Technologies Used

- **Backend**: FastAPI, MySQL, Python 
- **Machine Learning**: PyTorch, scikit-learn
- **Audio Processing**: Torchaudio, FFmpeg
- **Data Source**: iTunes Search API
- **Frontend**: HTML5, CSS3, JavaScript 

---

## Future Enhancements

- Add user authentication and personalized playlists
- Implement collaborative filtering alongside content-based filtering
- Support for multiple audio sources (Spotify, SoundCloud)
- Real-time model retraining pipeline
- Advanced visualization of audio features and similarity clusters
- Export recommendations to Spotify/Apple Music playlists

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
