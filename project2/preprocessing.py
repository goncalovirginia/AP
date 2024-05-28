import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def rgb2Ints(rgb):
    intBoard = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)

    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            if (rgb[i][j] == [0.5, 0.5, 0.5]).all(): # border
                intBoard[i][j] = 1
            elif 0.01 <= rgb[i][j][1] <= 0.4: # grass
                intBoard[i][j] = 2
            elif (rgb[i][j] == [0.0, 1.0, 0.0]).all(): # apple
                intBoard[i][j] = 3
            elif (rgb[i][j] == [1.0, 1.0, 1.0]).all(): # head
                intBoard[i][j] = 4
            elif rgb[i][j][0] == 1.0: # body 
                intBoard[i][j] = 5
            
    return intBoard