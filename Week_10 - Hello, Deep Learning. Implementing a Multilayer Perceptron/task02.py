from task01 import create_dataset, initialize_weights
import numpy as np

def calculate_loss(w, dataset):
    mse = 0
    for x, y in dataset:
        prediction = w * x
        mse += (prediction - y) ** 2
    return mse / len(dataset)

def main():
    dataset = create_dataset(6)

    np.random.seed(42)
    w = initialize_weights(0, 10)

    loss = calculate_loss(w, dataset)
    print(f'MSE: {loss}')

    # loss_w_plus_2 = calculate_loss(w + 0.001 * 2, dataset)
    # loss_w_plus_1 = calculate_loss(w + 0.001, dataset)
    # loss_w_minus_1 = calculate_loss(w - 0.001, dataset)
    # loss_w_minus_2 = calculate_loss(w - 0.001 * 2, dataset)

    # print(f'MSE with w + 0.001 * 2: {loss_w_plus_2}')
    # print(f'MSE with w + 0.001: {loss_w_plus_1}')
    # print(f'MSE with w - 0.001: {loss_w_minus_1}')
    # print(f'MSE with w - 0.001 * 2: {loss_w_minus_2}')

    #When w changes slightly by adding or subtracting small values (0.001 or 0.001 * 2),
    #the MSE also changes accordingly. The closer the adjusted w is to the true weight (2),
    #the lower the MSE. Conversely, as w deviates further from the true weight, the MSE increases.
if __name__ == '__main__':
    main()