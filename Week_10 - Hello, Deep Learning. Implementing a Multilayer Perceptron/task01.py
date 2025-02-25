import numpy as np


def create_dataset(n):
    return [(i, i*2) for i in range(0,n)]

def initialize_weights(x, y):
    return np.random.uniform(x,y)

def main():
    print(create_dataset(4))
    print(initialize_weights(0, 100))
    print(initialize_weights(0, 10))

if __name__ == '__main__':
    main()