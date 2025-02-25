import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

def main():
    artists_df = pd.read_csv('artists.csv', header=None, names=['artist_name'])
    artists_df['artist_offset'] = artists_df.index

    listening_data = pd.read_csv('scrobbler-small-sample.csv')

    listening_data['artist_offset'] = listening_data['artist_offset'].map(
        dict(zip(artists_df['artist_offset'], artists_df['artist_name']))
    )

    artist_user_matrix = listening_data.pivot_table(index='artist_offset', columns='user_offset', values='playcount', fill_value=0)

    pipeline = Pipeline([
        ('scaler', MaxAbsScaler()),
        ('nmf', NMF(n_components=20, random_state=42))
    ])

    nmf_features = pipeline.fit_transform(artist_user_matrix)

    target_idx = artist_user_matrix.index.get_loc("Bruce Springsteen")
    similarities = cosine_similarity(nmf_features[target_idx].reshape(1, -1), nmf_features).flatten()
    recommendations = pd.Series(similarities, index=artist_user_matrix.index).sort_values(ascending=False)
    print(recommendations.head(5))


if __name__ == '__main__':
    main()