import numpy as np
def main():
    #2
    print(np.zeros(10))
    #3
    print(np.zeros(10) + 1)
    #4
    print(np.zeros(10) + 5)
    #5
    arr = np.zeros(10)
    arr[4] = 1
    print(arr)
    #6
    print(np.arange(10,51))

    #7
    arr = np.array([1, 2, 3])

    print(arr[::-1])
    print(np.flip(arr))
    #8
    np_mat = np.array([[-1,0,0,0,-1],
                    [0,0,0,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0],
                    [-1,0,0,0,-1]])

    np_mat[0,:] += 1
    np_mat[:, 0] += 1
    np_mat[4, :] += 1
    np_mat[:, 4] += 1

    print(np_mat)

    #9
    arr = np.array([1, 2, 3])
    arr = np.append(arr, 0)
    arr = np.concatenate((np.array([0]), arr))
    print(arr)
    #10
    print(np.arange(10, 51, 2))

    #11
    matrix = np.arange(9).reshape(3, 3)
    print(matrix)
    #12
    identity_matrix = np.identity(3)
    print(identity_matrix)
    #13
    matrix = np.diag([1, 2, 3, 4], k=-1)
    print(matrix)

    #14
    print(np.random.rand())

    #15
    print(np.random.standard_normal(25))

    #16
    print(np.random.uniform(size = 27).reshape(3,3,3))

    #17
    rand_mat = np.random.uniform(size = (10, 10))
    print(rand_mat.max())
    print(rand_mat.min())

    #18
    uni_Dist = np.random.uniform(size = 30)
    uni_Dist.mean()

    #19
    np_mat = np.arange(0.01, 1.01, 0.01).reshape(10, 10)
    print(np_mat)
    #20
    print(np.linspace(0,1,20))

    mat = np.arange(1,26).reshape(5,5)
    print(mat)

    #21
    print(mat[2:, 1:])
    #22
    print(mat[3,4])
    #23
    print(mat[0:3, 1].reshape(3,1))
    #24
    print(mat[4])
    #25
    print(mat[3:])
    #26
    arr = np.array([1,2,0,0,4,0])
    print(np.arange(0, arr.size)[arr > 0])
    #27
    print(mat.sum())
    #28
    print(mat.std())
    #29
    print(mat.sum(axis = 0))
    #30

    mat1 = np.random.random_integers(0,10,15).reshape(5, 3)
    mat2 = np.random.random_integers(0,10,6).reshape(3,2)
    print(np.dot(mat1,mat2))

    #31

    arr = np.random.random_integers(0, 10, 10)
    print(arr[(arr < 3) | (arr > 8)])
    #array([ 4,  2,  5, -7, -2,  0, -3, -6, -8,  4])
    #if we generate numbers between 0-9 
    # why do we have negative numbers

    #32

    arr = np.random.binomial(2, 0.5, 10)
    arr.sort()
    print(arr)

    #33

    arr = np.random.uniform(size = 10)
    print(arr)
    to_replace = arr.max()
    arr[arr == to_replace] = 0
    print(arr)

    #34 CHATGPT HELPED
    # Set the seed for reproducibility
    np.random.seed(42)

    # Generate some random numbers
    arr = np.random.randint(0, 10, 5)
    print(arr)


if __name__ == '__main__':
    main()