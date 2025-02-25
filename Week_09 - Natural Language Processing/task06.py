import ast
from gensim import corpora, models
from collections import Counter

def main():
    file_path = "messy_articles.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        documents = ast.literal_eval(file.read())

    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    computer_id = dictionary.token2id.get("computer", "Not found")
    print(f'Id of "computer": {computer_id}')

    fifth_doc = corpus[4]

    first_10_word_ids = fifth_doc[:10]
    print(f"First 10 ids and frequency counts from the 5th document: {first_10_word_ids}")

    word_frequencies = {dictionary[word_id]: freq for word_id, freq in fifth_doc}
    top_5_words_fifth_doc = Counter(word_frequencies).most_common(5)
    print(f"Top 5 words in the 5th document: {list(dict(top_5_words_fifth_doc).keys())}")
    
    all_words = Counter()
    for doc in corpus:
        for word_id, freq in doc:
            all_words[word_id] += freq

    top_5_words_all_docs = all_words.most_common(5)
    top_5_words_with_counts = [(dictionary[word_id], count) for word_id, count in top_5_words_all_docs]
    print(f"Top 5 words across all documents: {top_5_words_with_counts}")

    tfidf_model = models.TfidfModel(corpus)
    tfidf_corpus = tfidf_model[corpus]

    tfidf_fifth_doc = tfidf_corpus[4]

    first_5_tfidf_terms = tfidf_fifth_doc[:5]
    print(f"First 5 term ids with their weights: {first_5_tfidf_terms}")

    tfidf_frequencies = {dictionary[word_id]: weight for word_id, weight in tfidf_fifth_doc}
    top_5_tfidf_words = Counter(tfidf_frequencies).most_common(5)
    print(f"Top 5 words in the 5th document when using tf-idf: {list(dict(top_5_tfidf_words).keys())}")

if __name__ == '__main__':
    main()