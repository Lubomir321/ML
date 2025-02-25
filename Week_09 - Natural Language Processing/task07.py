import nltk
from nltk import word_tokenize, pos_tag, ne_chunk_sents
from nltk.tree import Tree

def extract_named_entities(sentences):
        named_entities = []
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk_sents([pos_tags], binary=True)
            for tree in chunks:
                for subtree in tree:
                    if isinstance(subtree, Tree) and subtree.label() == 'NE':
                        named_entities.append(' '.join([token for token, pos in subtree.leaves()]))
        return named_entities


def main():
    data_file = 'article_uber.txt'

    with open(data_file, 'r', encoding='utf-8') as f:
        article = f.read()

    sentences = nltk.sent_tokenize(article)

    last_sentence_tokens = word_tokenize(sentences[-1])
    last_sentence_pos = pos_tag(last_sentence_tokens)
    print("Last sentence POS:", last_sentence_pos)

    first_sentence_tokens = word_tokenize(sentences[0])
    first_sentence_pos = pos_tag(first_sentence_tokens)

    ner_chunks = ne_chunk_sents([first_sentence_pos], binary=True)
    print("First sentence with NER applied:")
    for tree in ner_chunks:
        print(tree)

    all_named_entities = extract_named_entities(sentences)
    print("All chunks with label NE:")
    for entity in all_named_entities:
        print(f"\t(NE {entity})")

    # nltk.download('maxent_ne_chunker_tab')
    # nltk.download('words')
    # sentence = '''In New York, I like to ride the Metro to
    #               visit MOMA and some restaurants rated
    #               well by Ruth Reichl.'''
    # tokenized_sent = nltk.word_tokenize(sentence)
    # tagged_sent = nltk.pos_tag(tokenized_sent)
    # print(nltk.ne_chunk(tagged_sent))
if __name__ == '__main__':
    main()