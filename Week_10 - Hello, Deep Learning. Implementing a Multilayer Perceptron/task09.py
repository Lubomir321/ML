import numpy as np
import matplotlib.pyplot as plt

def initialize_weights(x, y):
    return np.random.uniform(x, y)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_loss(weights, dataset):
    mse = 0
    for x1, x2, y in dataset:
        prediction = sigmoid(weights[0] * x1 + weights[1] * x2 + weights[2])
        mse += (prediction - y) ** 2
    return mse / len(dataset)

def finite_difference_update(weights, dataset, learning_rate):

    epsilon = 1e-5
    updated_weights = weights.copy()
    for i in range(len(weights)):
        original_weight = weights[i]

        weights[i] = original_weight + epsilon
        loss_plus = calculate_loss(weights, dataset)
        
        weights[i] = original_weight - epsilon
        loss_minus = calculate_loss(weights, dataset)

        derivative = (loss_plus - loss_minus) / (2 * epsilon)
        updated_weights[i] = original_weight - learning_rate * derivative

        weights[i] = original_weight

    return updated_weights

def main():
    learning_rate = 0.001

    nand_dataset = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    nand_weights = [initialize_weights(0, 1), initialize_weights(0, 1), initialize_weights(0, 1)]

    for epoch in range(100000):
        nand_weights = finite_difference_update(nand_weights, nand_dataset, learning_rate)

    print("\nNAND gate:")
    for x1, x2, y in nand_dataset:
        prediction = sigmoid(nand_weights[0] * x1 + nand_weights[1] * x2 + nand_weights[2])
        print(f"Input: ({x1}, {x2}), Prediction: {prediction}, Target: {y}")

if __name__ == "__main__":
    main()