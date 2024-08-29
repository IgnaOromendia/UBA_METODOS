import unittest
import numpy as np

def resolver_sistema_triangular_superior(A, b):
    n = A.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        suma = 0
        for j in range(i,n): suma += A[i][j] * x[j]
        x[i] = (b[i] - suma) / A[i,i]  

    return x    

def eliminacion_gausseana_naive(T, b): 
    m = 0
    n = T.shape[0]
    A = T.copy()
    
    for i in range(0,n-1):
        for j in range(i+1, n):
            m = A[j][i]/A[i][i]       # coeficiente
            b[j] = b[j] - (m * b[i])            # aplicar a b
            for k in range(i, n):
                resta = A[j][k] - (m * A[i][k])
                A[j][k] = resta
            
    print(A)
    
    x = resolver_sistema_triangular_superior(A,b)

    return x

def eliminacion_gausseana_tridiagonal(T,b):
    n = T.shape[0]
    
    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    x = np.zeros(n)
    
    # ai xi−1 + bi xi + ci xi+1 = di
    
    # Armamos los vectores A B C 
    for i in range(n):
        B[i] = T[i][i]
        A[i] = 0 if i == 0 else T[i][i-1]
        C[i] = 0 if i == n-1 else T[i][i+1]
        
    # Resolvemos
    for i in range(1, n):
        coeficiente = A[i] / B[i-1]
        A[i] = A[i] - coeficiente * B[i-1]     # A_i - A_i / B_i-1 * B_i-1    
        B[i] = B[i] - coeficiente * C[i-1]  # B_i - A_i / B_i-1 * C_i-1
    
    T2 = T.copy()

    for i in range(n):
        T[i][i] = B[i]
        if (i >= 1): T[i][i-1] = A[i] 
        if (i < n-1): T[i][i+1] = C[i]
    
    x = resolver_sistema_triangular_superior(T,b)

    print("X EG naive: \n" + str(eliminacion_gausseana_naive(T2, b)))
    # print("T original: \n" + str(T2))
    # print("T EG: \n" + str(T))
    # print("X nuestro: " + str(x))
    print("X numpy: " + str([197/21, 121/21, 208/21, -1/21]))

    assert(np.allclose(np.dot(T2,x), b))
    
    
class TestEliminacionGausseana(unittest.TestCase):
    def test_resolver_sistema(self):
        
        A = np.array([  
                [2,1,-1,3],
                [-2,0,0,0],
                [4,1,-2,4],
                [-6,-1,2,-3]])

        b = np.array([13, -2, 24, -10])
        esperado_x = np.array([1, -30, 7, 16])

        resultado_x = eliminacion_gausseana_naive(A, b)

        np.testing.assert_array_almost_equal(resultado_x, esperado_x, decimal=5)


if __name__ == "__main__":
    A = np.array([[2,1,-1,3],
                [-2,0,0,0],
                [4,1,-2,4],
                [-6,-1,2,-3]], dtype=np.float64)



    A_zero = np.array([ [0,1,-1,3],
                        [-2,0,0,0],
                        [4,1,-2,4],
                        [-6,-1,2,-3]])
    b = np.array([13,-2,24,-10])
    
    # naive_x = eliminacion_gausseana_naive(A,b)
    # zerp_x = eliminacion_gausseana_naive(A_zero,b)
    #unittest.main() 
    # assert(np.allclose(np.dot(A, naive_x),b))

    m = np.array([  [ 2, -1,  0,  0],
                    [-1,  3, -1,  0],
                    [ 0, -1,  3, -1],
                    [ 0,  0, -1,  2]], dtype=np.float64)
    
    eliminacion_gausseana_tridiagonal(m, b)

