import numpy as np


def initialize_weights(x, y):
    return np.random.uniform(x, y)

def calculate_loss(weights, dataset):
    mse = 0
    for x1, x2, y in dataset:
        prediction = weights[0] * x1 + weights[1] * x2
        mse += (prediction - y) ** 2
    return mse / len(dataset)

# Model: y = w1 * x1 + w2 * x2 + b

def finite_difference_update(weights, dataset, learning_rate):
    epsilon = 1e-5
    updated_weights = weights.copy()
    for i in range(len(weights)):
        original_weight = weights[i]

        weights[i] = original_weight + epsilon
        loss_plus = calculate_loss(weights, dataset)
        
        weights[i] = original_weight - epsilon #also added subtraction and division by 2epsilon
        loss_minus = calculate_loss(weights, dataset)

        derivative = (loss_plus - loss_minus) / (2 * epsilon)
        updated_weights[i] = original_weight - learning_rate * derivative

        weights[i] = original_weight

    return updated_weights

def main():
    and_dataset = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    or_dataset = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]

    and_weights = [initialize_weights(0, 1), initialize_weights(0, 1)]
    or_weights = [initialize_weights(0, 1), initialize_weights(0, 1)]

    learning_rate = 0.001

    for epoch in range(100000):
        if epoch % 10000 == 0:
            print(f"Epoch {epoch}")
            print(f"AND weights: {and_weights}, Loss: {calculate_loss(and_weights, and_dataset)}")
            print(f"OR weights: {or_weights}, Loss: {calculate_loss(or_weights, or_dataset)}")

        and_weights = finite_difference_update(and_weights, and_dataset, learning_rate)
        or_weights = finite_difference_update(or_weights, or_dataset, learning_rate)

    print("\nFinal predictions:")
    print("AND gate:")
    for x1, x2, y in and_dataset:
        prediction = and_weights[0] * x1 + and_weights[1] * x2
        print(f"Input: ({x1}, {x2}), Prediction: {prediction}, Target: {y}")

    print("\nOR gate:")
    for x1, x2, y in or_dataset:
        prediction = or_weights[0] * x1 + or_weights[1] * x2
        print(f"Input: ({x1}, {x2}), Prediction: {prediction}, Target: {y}")

        #After training for 100,000 epochs, the model outputs values close to the expected targets.
        #The confidence is reflected in the predicted values being closer to the exact targets (0 or 1)

if __name__ == "__main__":
    main()