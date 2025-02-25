import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def preprocess(filepath: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the Indian Liver Patient Dataset.

    Parameters
    ----------
    filepath : str
        Path to the dataset CSV file.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Preprocessed features (X) and target variable (y).
    """
    columns = [
        "age", "gender", "tb", "db", "alkphos", "sgpt", "sgot", "tp",
        "alb", "a/g_ratio", "has_liver_disease"
    ]
    df = pd.read_csv(filepath, header=None, names=columns)

    df.fillna(df.mode().iloc[0], inplace=True)

    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})

    df["has_liver_disease"] = df["has_liver_disease"].map({2: 0, 1: 1})

    X = df.drop(columns=["has_liver_disease"])
    y = df["has_liver_disease"]

    return X, y

def main():
    filepath = "indian_liver_patient_dataset.csv"
    X, y = preprocess(filepath)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)

    bagging_clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(
            random_state=1, min_samples_leaf=8
        ),
        n_estimators=50,
        oob_score=True,
        random_state=1
    )

    bagging_clf.fit(X_train, y_train)

    oob_accuracy = bagging_clf.oob_score_
    test_accuracy = accuracy_score(y_test, bagging_clf.predict(X_test))

    print(f"Mean accuracy of aggregator on OOB instances: {oob_accuracy:.2f}")
    print(f"Test set accuracy: {test_accuracy:.2f}")
if __name__ == '__main__':
    main()