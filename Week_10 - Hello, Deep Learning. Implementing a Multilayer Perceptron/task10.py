import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def initialize_weights(x, y):
    return np.random.uniform(x, y)

def calculate_loss_with_sigmoid(weights, dataset):
    mse = 0
    for x1, x2, y in dataset:
        prediction = sigmoid(weights[0] * x1 + weights[1] * x2 + weights[2])  # Adding the bias term
        mse += (prediction - y) ** 2
    return mse / len(dataset)


def finite_difference_update(weights, dataset, learning_rate):
    epsilon = 1e-5
    updated_weights = weights.copy()
    for i in range(len(weights)):
        original_weight = weights[i]

        weights[i] = original_weight + epsilon
        loss_plus = calculate_loss_with_sigmoid(weights, dataset)
        
        weights[i] = original_weight - epsilon
        loss_minus = calculate_loss_with_sigmoid(weights, dataset)

        derivative = (loss_plus - loss_minus) / (2 * epsilon)
        updated_weights[i] = original_weight - learning_rate * derivative

        weights[i] = original_weight

    return updated_weights

def train_gate(data, weights, lr=0.1, epochs=10000):
    for _ in range(epochs):
        for x1, x2, y in data:
            weights = finite_difference_update(weights, data, lr)
    return weights

class Xor:
    def __init__(self, nand_weights, or_weights, and_weights,):
        self.nand_weights = nand_weights
        self.or_weights = or_weights
        self.and_weights = and_weights

    def forward(self, x1, x2):
        x = np.array([x1, x2])
        nand_output = sigmoid(np.dot(self.nand_weights[0:2], x) + self.nand_weights[2])
        or_output = sigmoid(np.dot(self.or_weights[0:2], x) + self.or_weights[2])
        xor_input = np.array([nand_output, or_output])
        xor_output = sigmoid(np.dot(self.and_weights[0:2], xor_input) + self.and_weights[2])
        return xor_output


def main():
    and_data = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    or_data = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
    nand_data = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    xor_data = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

    and_weights = [initialize_weights(0, 1), initialize_weights(0, 1), initialize_weights(0, 1)]
    or_weights = [initialize_weights(0, 1), initialize_weights(0, 1), initialize_weights(0, 1)]
    nand_weights = [initialize_weights(0, 1), initialize_weights(0, 1), initialize_weights(0, 1)]
    
    nand_weights_trained = train_gate(nand_data, nand_weights)
    or_weights_train = train_gate(or_data, or_weights)
    and_weights_train = train_gate(and_data, and_weights)

    xor_model = Xor(nand_weights_trained, or_weights_train, and_weights_train)

    print("\nXOR Gate Predictions:")
    for x1, x2, target in xor_data:
        pred = xor_model.forward(x1, x2)
        print(f"Input: ({x1}, {x2}) -> Predicted: {pred:.6f}, Target: {target}")


if __name__ == "__main__":
    main()
