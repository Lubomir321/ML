import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    np.random.seed(123)
    sum = 0
    array_steps = []
    for i in range(100):
        dice = np.random.random_integers(1,7)
        if dice < 3: 
            sum -= 1
        elif dice >= 3 and dice < 6: 
            sum += 1
        elif dice == 6: 
            dice = np.random.randint(1, 7) 
            sum += dice
        if sum < 0:
            sum = 0
        array_steps.append(sum)
    print(array_steps)

    
if __name__ == '__main__':
    main()