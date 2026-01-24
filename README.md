# Music Recommendation System

This repository contains a music recommendation system that leverages audio recognition technology to suggest songs based on user preferences. The system analyzes audio features and patterns to provide personalized music recommendations.

## Project Structure

```
project/
├── .env
├── .gitignore
├── README.md
├── main.py
├── requirements.txt
├── songs_with_features.pkl
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── src/
│   ├── preprocessing.py
│   ├── audiopipeline.py
│   ├── autoencoder.py
│   ├── extract_features.py
│   └── recommender.py
├── data/
│   └── songs_with_features.pkl
└── models/
    └── audio_autoencoder.pth
```

## Setup Instructions

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
python backend.py
```

The backend will start on `http://localhost:8000`

### 2. Frontend Setup

Simply open `index.html` in your web browser, or use a local server:

```bash
# Using Python's built-in server
cd static
python -m http.server 8080
```

Then visit `http://localhost:8080` in your browser.

### 3. Data Preparation

Before running the backend, make sure you have:

1. Processed your audio files using `preprocessing.py`
2. Trained the autoencoder using `autoencoder.py` through `audiopipeline.py`
3. Extracted features using `extract_features.py`
4. The resulting `songs_with_features.pkl` file in the `data/` directory

## API Endpoints

- `GET /` - Health check
- `GET /songs` - List all available songs
- `POST /recommend` - Get recommendations for a song


### Example API Request

```json
// POST /recommend
{
    "song_name": "Out of Time"
}
```

### Example API Response

```json
{
    "query_song": "Out of Time - The Weeknd",
    "recommendations": [
        {
            "name": "Starboy",
            "similarity": "0.9234"
        },
        ...
    ]
}
```

## Usage

1. Start the backend server
2. Open the frontend in your browser
3. Click "Get Started"
4. Enter a song name (or part of it)
5. View your personalized recommendations!

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: FastAPI, Python
- **ML**: PyTorch, scikit-learn
- **Data Processing**: pandas, numpy
- **Data Visualization**: matplotlib, seaborn
