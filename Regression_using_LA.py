import numpy as np   
import matplotlib.pyplot as plt
import numpy.linalg as la


def Regression(x, y) :

    xm = np.matrix(x)   #data type transform
    ym = np.matrix(y)   #to suit the LA algo.

    DataShape = xm.shape  #shape loading
    A = np.matrix(np.zeros((DataShape[1], 2)))
    A[:, 0] = np.full((DataShape[1], 1), 1)

    for i in range(DataShape[1]) :
        A[i, 1] = xm.T[i]

    V = A.T*A
    W = A.T*ym.T  

    S = la.solve(V, W)

    return S


def graph(x, y, S) :

    plt.scatter(x, y, color = "m", marker = "o", s = 30) #scatter drawing

    y_pred = S[0] + S[1]*x
    y_pred_p = np.array(y_pred)
    
    plt.plot(x, y_pred) 

	# putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 
    # function to show plot 
    plt.show() 

def main() :

    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12]) 

    S = Regression(x, y)

    Sp = np.array(S)

    graph(x, y, Sp)

    print("b = ", S[0])
    print("a = ", S[1])

main()