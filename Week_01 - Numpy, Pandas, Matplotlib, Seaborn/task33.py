import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    steps = [0]
    np.random.seed(123)
    sum = 0
    for i in range(100):
        dice = np.random.randint(1,7)
        if dice < 3: 
            sum -= 1
        elif dice >= 3 and dice < 6: 
            sum += 1
        elif dice == 6: 
            dice = np.random.randint(1, 7) 
            sum += dice
        if sum < 0:
            sum = 0
        steps.append(sum)
        

    plt.plot(steps) 
    plt.title("Random walk")
    plt.xlabel("Throw")
    plt.show()


if __name__ == '__main__':
    main()