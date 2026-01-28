# Music Recommendation System

A content-based music recommendation system that suggests songs based on their audio features (timbre and dynamics) and metadata. It uses a **Statistical Feature Extractor** to analyze spectrograms and a **MySQL** database for scalable storage.

Note: This system is currently a prototype processed using a local dataset. For commercial applications or production environments, the model requires training on a significantly larger and more diverse high-volume dataset to ensure robust recommendation accuracy and coverage.

---

## Key Features

* **Automated Data Pipeline**: Searches the iTunes API to download high-quality audio previews (30s) and metadata automatically.
* **Statistical Feature Extraction**: Uses mathematical analysis (Mean & Standard Deviation of frequency bands) to create a unique 256-dimensional "fingerprint" for each song, ensuring distinct recommendations even with small datasets.
* **Database Integration**: Stores song metadata and latent feature vectors in a persistent MySQL database.
* **Visual Analysis**: Includes a Jupyter Notebook (`eda.ipynb`) to visualize spectrograms and feature distributions.

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
│   ├── preprocessing.py      # Downloads audio from iTunes & generates spectrograms
│   ├── extract_features.py   # Calculates statistical features & saves to DB
│   ├── recommender.py        # CLI recommendation tool
│   └── check_db.py           # Database status utility
├── data/                     # Audio previews and .pkl cache
└── eda.ipynb                 # Exploratory Data Analysis notebook
```

---

## Setup Instructions

### 1. Database Configuration

* Ensure **MySQL** is installed and running.
* Execute the SQL commands in `database/schema.sql` to create the `music_recommender` database and `songs` table.
* Update your `.env` file with your `DB_PASSWORD` and `DB_USER`.

### 2. Python Environment
```bash
# Install dependencies
pip install -r requirements.txt
```

### 3. Data Pipeline Execution

Run these scripts in order to populate the system:

1. **`src/preprocessing.py`**:
   * Searches iTunes for specified artists (e.g., "Weeknd", "KK")
   * Downloads 30s audio previews
   * Converts audio to Mel-Spectrograms and saves to `data/processed_audio_df.pkl`

2. **`src/extract_features.py`**:
   * Loads the processed spectrograms
   * Calculates a 256-dim vector using Mean (Tonal Balance) and Standard Deviation (Dynamics) across frequency bands
   * Saves the vector + metadata into the MySQL database

3. **`src/check_db.py`**:
   * Verifies that songs have been successfully stored in the database

---

## Running the Application

### Backend Server
```bash
python main.py
```

* The API will start at `http://localhost:8000`
* **Endpoints**:
  * `GET /`: Frontend UI
  * `POST /recommend`: Returns top 5 similar songs based on Cosine Similarity

### CLI Tool 
```bash
python src/recommender.py
```

---

## Technologies Used

* **Backend**: FastAPI, Python
* **Database**: MySQL (mysql-connector-python)
* **Audio Processing**: Torchaudio, FFmpeg, NumPy
* **Data Source**: iTunes Search API

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
