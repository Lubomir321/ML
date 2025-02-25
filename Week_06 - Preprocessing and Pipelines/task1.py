import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_json("music_dirty.txt")

    print("Shape before one-hot-encoding:", df.shape)

    music_dummies = pd.get_dummies(df['genre'], drop_first=True, dtype=int)

    genres = list(music_dummies.head(0))
    genres.insert(0, 'Alternative')

    music_dummies = pd.concat([df, music_dummies], axis=1)

    music_dummies = music_dummies.drop(columns=['genre'])

    print("Shape after one-hot-encoding:", music_dummies.shape)

    popularity_data = [df[df['genre'] == genre]['popularity'] for genre in genres]

    plt.boxplot(popularity_data, labels=genres, patch_artist=True)
    plt.title("Boxplot grouped by genre popularity")
    plt.xlabel("genre")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()