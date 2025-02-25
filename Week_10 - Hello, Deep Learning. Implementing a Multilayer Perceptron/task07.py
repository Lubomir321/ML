import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    x_values = np.linspace(-10, 10, 200)
    y_values = sigmoid(x_values)
    plt.plot(x_values, y_values)
    plt.title("Sigmoid Function")
    plt.xlabel("Input")
    plt.ylabel("Sigmoid Output")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()