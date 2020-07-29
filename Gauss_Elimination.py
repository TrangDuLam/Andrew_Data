import numpy as np

def Gauss_elim(A, b) :
    m, n = A.shape 
    # R is the reduced row echelon form
    R = np.matrix(np.zeros([m, n+1]))
    # variable assignment
    R[:, :n] = A
    R[:, n] = b

    #operation start
    for i in range(m) :
        # find the maximal element in the column i
        max_Ec = abs(R[i, i])  #default value on the diagonal 
        max_row = i
        for k in range(i+1, m) :    #checking the maximun value in other oposition
            if abs(R[k, i]) > max_Ec :
                max_Ec = R[k, i]  #if greater, change
                max_row = k
        #swapping maximun row with current row
        R[[i, max_row], i:] = R[[max_row, i], i:]
        #make all row below this one to 0
        for k in range(i+1, m) :  #operation
            c = -float(R[k, i])/R[i, i]
            R[k, i:] = R[k, i:] + c*R[i, i:]

    # Solving
    x = np.matrix(np.zeros([m, 1]))
    for i in range(m-1, -1, -1) :
        x[i] = float(R[i, -1])/R[i, i]
        for k in range(i-1, -1, -1) :
            R[k, -1] -= R[k, i]*x[i]

    return x


#From Ax = b

A = np.matrix([[8., 4., 5., 7.],     
              [4., 3., 4., 1.],
              [10., 9., 4., 6.],
              [10., 1., 8., 4.]])


b = np.matrix([[8.5],
               [1.23],
               [9.36],
               [4]])

x = Gauss_elim(A, b)

print(x)