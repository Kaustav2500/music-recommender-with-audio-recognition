import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# load dataframe with latent vectors
df = pd.read_pickle("../data/songs_with_features.pkl")

# pick a query song
print('Enter a song name to get recommendations: ')
# get user input
query_name = input("Enter the song name (or part of it): ").strip()

# check if the name is a substring in the dataframe
matches = df[df['file_name'].str.contains(query_name, case=False, na=False)]

if not matches.empty:
    # pick the first match found
    query_index = matches.index[0]
    matched_full_name = df.loc[query_index, 'file_name']
    print(f"Match found: {matched_full_name}")

    # extract the vector for this specific song
    query_vector = df['latent_vector'].iloc[query_index].reshape(1, -1)
    print(f"Finding recommendations for: {query_name}")

    # calculate similarity against all other songs and stack all vectors into a matrix
    all_vectors = np.stack(df['latent_vector'].values)

    # compute cosine similarity (1.0 = identical, -1.0 = opposite)
    similarities = cosine_similarity(query_vector, all_vectors)

    # sort results
    sim_scores = list(enumerate(similarities.flatten()))

    # sort by score, skip the first one
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    # print recommendations
    print("\nTop 5 Recommendations:")
    for i, score in sim_scores:
        print(f"{df['file_name'].iloc[i]} (Similarity: {score:.4f})")
else:
    print(f"No song found containing the name: {query_name}")

