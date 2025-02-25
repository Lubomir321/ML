from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main():
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) #Question why we dont need stratify here?

    ovr_model = LogisticRegression(multi_class='ovr', random_state=42)
    ovr_model.fit(X_train, y_train)
    ovr_train_accuracy = ovr_model.score(X_train, y_train)
    ovr_test_accuracy = ovr_model.score(X_test, y_test)

    print(f"OVR training accuracy: {ovr_train_accuracy}")
    print(f"OVR test accuracy    : {ovr_test_accuracy}")

    softmax_model = LogisticRegression(multi_class='multinomial', random_state=42)
    softmax_model.fit(X_train, y_train)
    softmax_train_accuracy = softmax_model.score(X_train, y_train)  
    softmax_test_accuracy = softmax_model.score(X_test, y_test)

    print(f"Softmax training accuracy: {softmax_train_accuracy}")
    print(f"Softmax test accuracy    : {softmax_test_accuracy}")

if __name__ == '__main__':
    main()