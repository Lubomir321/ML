import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
import matplotlib.pyplot as plt
from collections import Counter


def extract_entity_labels(sentences):
    entity_labels = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        pos_tags = pos_tag(tokens)
        
        tree = ne_chunk(pos_tags, binary=False)
        
        for subtree in tree:
            if isinstance(subtree, Tree): #this could be checked with type
                entity_labels.append(subtree.label())
    return entity_labels

def main():
    data_file = 'news_articles.txt'

    with open(data_file, 'r', encoding='utf-8') as f:
        articles = f.read()

    sentences = nltk.sent_tokenize(articles)

    entity_labels = extract_entity_labels(sentences)

    label_counts = Counter(entity_labels)

    plt.figure(figsize=(8, 8))
    plt.pie(
        label_counts.values(),
        labels=label_counts.keys(),
        autopct='%1.1f%%',
        startangle=140,
    )
    plt.title("Distribution of NER categories")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()