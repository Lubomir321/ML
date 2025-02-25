import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

def main():
    data = pd.read_csv('wikipedia-vectors.csv', index_col=0)

    titles = data.columns
    vectors = data.T.values

    print(titles)
    print(vectors)

    #Experimenting with n_iter more accurate clustering
    pipeline = Pipeline([
        ('svd', TruncatedSVD(n_components=50, n_iter=10, random_state=42)),
        ('kmeans', KMeans(n_clusters=6, random_state=42))
    ])
    #i saw that for some task not using random_state gave more accurate answer inspite it not being rquiered
    #should i use it on every task
    pipeline.fit(vectors)

    labels = pipeline.named_steps['kmeans'].labels_

    clustered_data = pd.DataFrame({
        'label': labels,
        'article': titles
    })
    
    sorted_data = clustered_data.sort_values('label')
    print(sorted_data)


if __name__ == '__main__':
    main()