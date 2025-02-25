from nltk.tokenize import regexp_tokenize, TweetTokenizer

def main():
    tweets = ['This is the best #nlp exercise ive found online! #python','#NLP is super fun! <3 #learning','Thanks @datacamp :) #nlp #python']

    hashtags_first_tweet = regexp_tokenize(tweets[0], r'#\w+')

    mentions_and_hashtags_last_tweet = regexp_tokenize(tweets[-1], r'[@#]\w+')

    tweet_tokenizer = TweetTokenizer()
    all_tokens = [tweet_tokenizer.tokenize(tweet) for tweet in tweets]

    print(f"All hashtags in first tweet: {hashtags_first_tweet}")
    print(f"All mentions and hashtags in last tweet: {mentions_and_hashtags_last_tweet}")
    print(f"All tokens: {all_tokens}")
if __name__ == "__main__":
    main()