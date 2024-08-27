import numpy as np

def resolver_sistema_triangular_superior(A, b):
    n = A.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        suma = 0
        for j in range(i,n): suma += A[i][j] * x[j]
        x[i] = (b[i] - suma) / A[i,i]  

    return x    

def eliminacion_gausseana_naive(A, b): 
    m = A.copy()
    n = A.shape[0]
    
    for i in range(n-1):    
        if (A[i][i] == 0): print("EXPLOTA")
        for j in range(i+1, n):
            m[j][i] = A[j][i]/A[i][i]
            b[j] = b[j] - (m[j][i] * b[i])
            for k in range(i, n):
                A[j][k] = A[j][k] - (m[j][i]*A[i][k])

    x = resolver_sistema_triangular_superior(A,b)

    return x


A = np.array([  [2,1,-1,3],
                [-2,0,0,0],
                [4,1,-2,4],
                [-6,-1,2,-3]])

b = np.array([13,-2,24,-10])


A_zero = np.array([ [0,1,-1,3],
                    [-2,0,0,0],
                    [4,1,-2,4],
                    [-6,-1,2,-3]])

naive_x = eliminacion_gausseana_naive(A,b)
zerp_x = eliminacion_gausseana_naive(A_zero,b)

