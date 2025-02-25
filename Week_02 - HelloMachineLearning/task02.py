
def main():
    from sklearn.module import Model
    model = Model()
    model.fit(X, y)

    predictions = model.predict(X_new)

    print(predictions)
    
if __name__ == '__main__':
    main()