import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def main():
    file_path = 'fake_or_real_news.csv'
    data = pd.read_csv(file_path)

    print(data.head(7))

    label_distribution = data['label'].value_counts()
    label_percentage = (label_distribution / len(data)) * 100
    distribution_df = pd.DataFrame({"count": label_distribution, "proportion": label_percentage})
    print("\nDistribution of labels")
    print(distribution_df)

    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=0.33, random_state=52
    )

    count_vectorizer = CountVectorizer(stop_words='english')
    X_train_count = count_vectorizer.fit_transform(X_train)

    count_features = count_vectorizer.get_feature_names_out()
    print("\nFirst 10 tokens:", count_features[:10])
    print("Size of vocabulary:", len(count_features))

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    tfidf_features = tfidf_vectorizer.get_feature_names_out()
    print("\nFirst 10 tokens from TfidfVectorizer:", tfidf_features[:10])

    print("\nFirst 5 vectors (Tfidf):")
    print(X_train_tfidf[:5].toarray())

    df_count = pd.DataFrame(X_train_count.toarray(), columns=count_features)
    df_tfidf = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_features)

    print("\nDataFrame obtained by CountVectorizer - df_count:")
    print(df_count.head())
    print("\nDataFrame obtained by TfidfVectorizer - df_tfidf:")
    print(df_tfidf.head())

    count_only_tokens = set(count_features) - set(tfidf_features)
    print("\nTokens that are in df_count, but are not in df_tfidf:", count_only_tokens)

if __name__ == "__main__":
    main()