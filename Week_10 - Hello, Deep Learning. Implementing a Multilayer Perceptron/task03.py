from task01 import create_dataset, initialize_weights
from task02 import calculate_loss
import numpy as np

def finite_difference_update(w, dataset, learning_rate):

    epsilon = 1e-5  # Small value to compute finite difference
    loss_before = calculate_loss(w, dataset)
    derivative = (calculate_loss(w + epsilon, dataset) - calculate_loss(w, dataset)) / (epsilon)
    w_updated = w - learning_rate * derivative
    loss_after = calculate_loss(w_updated, dataset)

    print(f"Loss before update: {loss_before}")
    print(f"Loss after update: {loss_after}")

    return w_updated

def main():
    dataset = create_dataset(6)

    np.random.seed(42)
    w = initialize_weights(0, 10)

    learning_rate = 0.001

    for epoch in range(10):
        print(f"Epoch {epoch + 1}")
        w = finite_difference_update(w, dataset, learning_rate)

if __name__ == '__main__':
    main()