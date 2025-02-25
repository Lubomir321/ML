import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv("fake_or_real_news.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=0.33, random_state=43
    )

    count_vectorizer = CountVectorizer(stop_words='english')
    X_train_count = count_vectorizer.fit_transform(X_train)
    X_test_count = count_vectorizer.transform(X_test)

    nb_bow = MultinomialNB()
    nb_bow.fit(X_train_count, y_train)
    y_pred_bow = nb_bow.predict(X_test_count)

    accuracy_bow = accuracy_score(y_test, y_pred_bow)
    print(f"Accuracy when using BoW: {accuracy_bow}")

    conf_matrix_bow = confusion_matrix(y_test, y_pred_bow, labels=['FAKE', 'REAL'])

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    nb_tfidf = MultinomialNB()
    nb_tfidf.fit(X_train_tfidf, y_train)
    y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)

    accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
    print(f"Accuracy when using TF-IDF: {accuracy_tfidf}")

    conf_matrix_tfidf = confusion_matrix(y_test, y_pred_tfidf, labels=['FAKE', 'REAL'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))    
    disp_bow = ConfusionMatrixDisplay(conf_matrix_bow, display_labels=['FAKE', 'REAL'])
    disp_bow.plot(ax=axes[0], colorbar=True)

    disp_tfidf = ConfusionMatrixDisplay(conf_matrix_tfidf, display_labels=['FAKE', 'REAL'])
    disp_tfidf.plot(ax=axes[1], colorbar=True)
    
    plt.suptitle("Confusion matrix: BoW (left) vs TF-IDF(right)")
    plt.tight_layout()
    plt.show()

    alphas = np.arange(0, 1.1, 0.1)
    accuracy_scores = []

    for alpha in alphas:
        nb = MultinomialNB(alpha=alpha)
        nb.fit(X_train_tfidf, y_train)
        y_pred = nb.predict(X_test_tfidf)
        accuracy_scores.append(accuracy_score(y_test, y_pred))

    plt.figure(figsize=(10, 6))
    plt.plot(alphas, accuracy_scores)
    plt.title("Accuracy vs Alpha (TF-IDF)")
    plt.xlabel("Alpha")
    plt.ylabel("Accuracy")
    plt.grid() #when is grid needed?
    plt.show()

    best_alpha = alphas[np.argmax(accuracy_scores)]
    print(f"Best alpha value: {best_alpha}")

    nb_best = MultinomialNB(alpha=best_alpha)
    nb_best.fit(X_train_tfidf, y_train)

    feature_names = tfidf_vectorizer.get_feature_names_out()

    log_prob = nb_best.feature_log_prob_

    fake_top_indices = np.argsort(log_prob[0])[-20:]
    fake_top_words = feature_names[fake_top_indices]

    real_bottom_indices = np.argsort(log_prob[1])[:20]
    real_bottom_words = feature_names[real_bottom_indices]

    print("FAKE", fake_top_words)
    print("REAL", real_bottom_words)
if __name__ == "__main__":
    main()