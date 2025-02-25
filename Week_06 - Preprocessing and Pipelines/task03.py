import pandas as pd

def main():
    df = pd.read_json("music_dirty_missing_vals.txt")

    print("Shape of input dataframe:", df.shape)

    missing_percentages = df.isnull().mean().sort_values(ascending=False)

    print("Percentage of missing values:\n", missing_percentages)

    cols_to_drop_rows = missing_percentages[missing_percentages < 0.05].keys().tolist()
    print("Columns/Variables with missing values less than 5% of the dataset:", cols_to_drop_rows)

    df = df.dropna(subset=cols_to_drop_rows)

    df['genre'] = df['genre'].apply(lambda x: 1 if x == "Rock" else 0)

    print("First five entries in `genre` column:\n", df['genre'].head())

    print("Shape of preprocessed dataframe:", df.shape)

if __name__ == '__main__':
    main()