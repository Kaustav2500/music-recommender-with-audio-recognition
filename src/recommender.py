import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

# add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from database.db_connect import load_all_songs

# load data from MySQL
print("Loading songs from database...")
try:
    song_data_list, latent_matrix = load_all_songs()
except Exception as e:
    print(f"Error connecting to database: {e}")
    exit()

if not song_data_list:
    print("Database is empty! Please run extract_features.py first.")
    exit()

df = pd.DataFrame(song_data_list)
print(f"Loaded {len(df)} songs.")

# pick a query song
query_name = input("Enter a song name to get recommendations: ").strip()

# check if the name is a substring in the dataframe
matches = df[df['name'].str.contains(query_name, case=False, na=False)]

if not matches.empty:
    # Pick the first match found
    query_index = matches.index[0]
    matched_name = df.loc[query_index, 'name']
    matched_artist = df.loc[query_index, 'artist']
    print(f"Match found: {matched_name} - {matched_artist}")

    # extract the vector for this specific song
    query_vector = latent_matrix[query_index].reshape(1, -1)

    # calculate similarity
    similarities = cosine_similarity(query_vector, latent_matrix)

    # sort results
    sim_scores = list(enumerate(similarities.flatten()))

    # sort by score (descending)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    # print recommendations
    count = 0
    print("\nTop 5 Recommendations:")
    for i, score in sim_scores:
        # skip the query song itself
        if i == query_index:
            continue

        rec_name = df.iloc[i]['name']
        rec_artist = df.iloc[i]['artist']

        # skip exact name duplicates
        if rec_name.lower().strip() == matched_name.lower().strip():
            continue

        print(f"{rec_name} - {rec_artist} (Similarity: {score:.4f})")

        # stop after 5 valid recommendations
        count += 1
        if count >= 5:
            break
else:
    print(f"No song found containing the name: {query_name}")

