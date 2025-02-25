from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def load_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

def build_bag_of_words(text):
    tokens = word_tokenize(text.lower())
    return Counter(tokens)

def preprocess_and_build_bow(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(text.lower())
    preprocessed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens 
        if word.isalpha() and word not in stop_words
    ]
    return Counter(preprocessed_tokens)

if __name__ == "__main__":
    filepath = "article.txt"
    text = load_file(filepath)

    bow = build_bag_of_words(text)
    print("Top 10 most common tokens:", bow.most_common(10))

    preprocessed_bow = preprocess_and_build_bow(text)
    print("Top 10 most common tokens after preprocessing:", preprocessed_bow.most_common(10))

    #Based on the preprocessed tokens, we can infer the topic of the article.
    #The most common words suggest that the article is about debugging.
