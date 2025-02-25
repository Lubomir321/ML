from task01 import create_dataset, initialize_weights
from task03 import finite_difference_update
import numpy as np

def main():
    dataset = create_dataset(6)
    np.random.seed(42)
    w = initialize_weights(0, 10)
    learning_rate = 0.001
    for epoch in range(500):
        print(f"Epoch {epoch + 1}")
        w = finite_difference_update(w, dataset, learning_rate)
    print(w)
if __name__ == '__main__':
    main()