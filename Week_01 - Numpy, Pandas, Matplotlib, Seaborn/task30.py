import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    print(f"Random float: {np.random.rand()}")

    print(f"Random integer 1: {np.random.randint(1,7)}")

    print(f"Random integer 2: {np.random.randint(1,7)}")

    dice = np.random.randint(1, 7)
    steps = 50
    print(f"Befor throw step = {steps}")
    if dice < 3: 
        steps -= 1
    elif dice >= 3 and dice < 6: 
        steps += 1
    elif dice == 6: 
        dice = np.random.randint(1, 7) 
        steps += dice

    print(f"After throw dice = {dice}")
    print(f"After throw step = {steps}")

    np.random.seed(123)


if __name__ == '__main__':
    main()