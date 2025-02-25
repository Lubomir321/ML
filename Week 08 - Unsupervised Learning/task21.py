import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

def main():
    data = pd.read_csv('wikipedia-vectors.csv', index_col=0)

    titles = data.columns
    vectors = data.T.values
    #transpose here is needed because of the array titles

    nmf = NMF(n_components=10)
    nmf_features = nmf.fit_transform(vectors)

    nmf_features = normalize(nmf_features)

    target_article = "Cristiano Ronaldo"
    target_idx = list(titles).index(target_article)

    similarities = cosine_similarity(nmf_features[target_idx].reshape(1, -1), nmf_features).flatten()
    #reshape is need to match the requiered cosine dimension of the passed value

    similarity_series = pd.Series(similarities, index=titles)

    similarity_series = similarity_series.sort_values(ascending=False)

    print(similarity_series.head())


if __name__ == '__main__':
    main()