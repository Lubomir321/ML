import numpy as np
import matplotlib.pyplot as plt

def calculate_loss(weights, dataset):
    mse = 0
    for x1, x2, y in dataset:
        prediction = weights[0] * x1 + weights[1] * x2 + weights[2]  # Adding the bias term
        mse += (prediction - y) ** 2
    return mse / len(dataset)


def calculate_loss_with_sigmoid(weights, dataset):
    mse = 0
    for x1, x2, y in dataset:
        prediction = sigmoid(weights[0] * x1 + weights[1] * x2 + weights[2])  # Adding the bias term
        mse += (prediction - y) ** 2
    return mse / len(dataset)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def initialize_weights(x, y):
    return np.random.uniform(x, y)

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
    and_data = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    or_data = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]

    and_weights = [initialize_weights(0, 1), initialize_weights(0, 1), initialize_weights(0, 1)]  # Including bias
    or_weights = [initialize_weights(0, 1), initialize_weights(0, 1), initialize_weights(0, 1)]  # Including bias

    learning_rate = 0.001

    and_loss_history_with_sigmoid = []
    or_loss_history_with_sigmoid = []

    and_loss_history_without_sigmoid = []
    or_loss_history_without_sigmoid = []

    for epoch in range(100000):
    
        and_weights = finite_difference_update(and_weights, and_data, learning_rate)
        or_weights = finite_difference_update(or_weights, or_data, learning_rate)

        and_loss_with_sigmoid = calculate_loss_with_sigmoid(and_weights, and_data)
        or_loss_with_sigmoid = calculate_loss_with_sigmoid(or_weights, or_data)
        and_loss_history_with_sigmoid.append(and_loss_with_sigmoid)
        or_loss_history_with_sigmoid.append(or_loss_with_sigmoid)

        and_loss_without_sigmoid = calculate_loss(and_weights, and_data)
        or_loss_without_sigmoid = calculate_loss(or_weights, or_data)
        and_loss_history_without_sigmoid.append(and_loss_without_sigmoid)
        or_loss_history_without_sigmoid.append(or_loss_without_sigmoid)


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(100000), and_loss_history_without_sigmoid, label="AND")
    plt.plot(range(100000), or_loss_history_without_sigmoid, label="OR")
    plt.title("Without sigmoid")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(range(100000), and_loss_history_with_sigmoid, label="AND Loss")
    plt.plot(range(100000), or_loss_history_with_sigmoid, label="OR Loss")
    plt.title("With sigmoid")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    #With sigmoid, the outputs are constrained between 0 and 1.

if __name__ == "__main__":
    main()