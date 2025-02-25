import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    all_walks = []

    np.random.seed(123)

    for j in range(500):
        steps = [0]
        sum = 0
        for i in range(100):
            dice = np.random.randint(1,7)
            if dice < 3: 
                sum -= 1
            elif dice >= 3 and dice < 6: 
                sum += 1
            elif dice == 6: 
                dice = np.random.randint(1,7) 
                sum += dice
            if sum < 0:
                sum = 0
            falling = np.random.rand()
            if falling <= 0.005:
                sum = 0
            steps.append(sum)
            #print(sum)
        all_walks.append(steps)

    all_walks_np = np.array(all_walks)

    last_col = all_walks_np[:, -1]  

    plt.hist(last_col)
    plt.title("Random walks")
    plt.xlabel("End step")
    plt.show()
    #print(len(last_col[last_col >= 60]))

#304/500? = 0.608



if __name__ == '__main__':
    main()