import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    all_walks = []

    np.random.seed(123)

    for j in range(5):
        steps = [0]
        sum = 0
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
            steps.append(sum)
            #print(sum)
        all_walks.append(steps)
        

    for i in range(5):
        walk = np.array(all_walks[i])
        plt.plot(walk)
    plt.title("Random walks")
    plt.xlabel("Throw")
    plt.show()



if __name__ == '__main__':
    main()