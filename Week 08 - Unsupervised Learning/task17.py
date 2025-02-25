import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    documents = ['cats say meow', 'dogs say woof', 'dogs chase cats']

    vectorizer = TfidfVectorizer()

    csr_matrix = vectorizer.fit_transform(documents)

    words = vectorizer.get_feature_names_out()

    print(csr_matrix.toarray())
    print(words)


if __name__ == '__main__':
    main()