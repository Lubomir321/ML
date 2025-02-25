import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
import spacy

def nltk_ner(text):
        nltk_entities = []
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            pos_tags = pos_tag(tokens)
            tree = ne_chunk(pos_tags, binary=False)
            for subtree in tree:
                if isinstance(subtree, Tree):
                    entity_label = subtree.label()
                    entity_name = " ".join(token for token, _ in subtree.leaves())
                    nltk_entities.append((entity_name, entity_label))
        return nltk_entities

nlp = spacy.load("en_core_web_sm")
def spacy_ner(text):
    spacy_entities = []
    doc = nlp(text)
    for ent in doc.ents:
        spacy_entities.append((ent.text, ent.label_))
    return spacy_entities

def main():
    data_file = 'article_uber.txt'

    with open(data_file, 'r') as f:
        article = f.read()

    nltk_entities = nltk_ner(article)
    print("NLTK Named Entities:")
    for entity, label in nltk_entities:
        print(f"{label}: {entity}")

    spacy_entities = spacy_ner(article)
    print("\nspaCy Named Entities:")
    for entity, label in spacy_entities:
        print(f"{label}: {entity}")

    #The extra categories that spaCy uses compared to NLTK are
    #C. NORP, CARDINAL, MONEY, WORKOFART, LANGUAGE, EVENT

if __name__ == "__main__":
    main()