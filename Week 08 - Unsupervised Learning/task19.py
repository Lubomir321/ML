import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

def main():
    data = pd.read_csv('wikipedia-vectors.csv', index_col=0)
    titles = data.columns
    vectors = data.T.values

    nmf = NMF(n_components=6, random_state=42)
    nmf_features = nmf.fit_transform(vectors)

    print(np.round(nmf_features[:6], 2))

    nmf_df = pd.DataFrame(nmf_features, index=titles)

    print(nmf_df.loc[['Anne Hathaway', 'Denzel Washington']])

    highest_feature = nmf_df.loc[['Anne Hathaway', 'Denzel Washington']].mean(axis=0).idxmax()
    #print(highest_feature)
    #The highest feature value is associated with feature 3. 
    #This indicates that the topic represented by this feature is most relevant to the articles about the actors.

    with open('wikipedia-vocabulary-utf8.txt', 'r') as f:
        vocabulary = f.read().splitlines()

    feature_weights = pd.Series(nmf.components_[highest_feature], index=vocabulary)
    print(f'The topic, that the articles about Anne Hathaway and Denzel Washington have in common, has the words:\n{feature_weights.sort_values(ascending=False).head()}')

    #The topic that the articles about Anne Hathaway and Denzel Washington have in
    #common includes words such as film, award, starred, role, and actress.

if __name__ == '__main__':
    main()