import unittest
import numpy as np
from copy import deepcopy as dc
import copy

def resolver_sistema_triangular_superior(A, b):
    n = len(A[0])
    x = [0] * n

    for i in range(n - 1, -1, -1):
        suma = 0
        for j in range(i,n): suma += A[i][j] * x[j]
        x[i] = (b[i] - suma) / A[i][i]  

    return x     

def eliminacion_gausseana_naive(T, x): 
    m = 0
    n = len(T[0])
    A0 = dc(T)
    A1 = dc(T)
    b = dc(x)
    
    for i in range(0,n):
        for j in range(i+1, n):
            m = A0[j][i]/A0[i][i]                 # coeficiente
            b[j] = b[j] - (m * b[i])            # aplicar a b
            for k in range(i, n):
                A1[j][k] = A0[j][k] - (m * A0[i][k])
        A0 = dc(A1)
            
    return resolver_sistema_triangular_superior(A1,b)

def permutar_filas(A, i, j):
    copia = dc(A[i])
    A[i] = A[j]
    A[j] = copia
    
def encontrar_pivote(A, i):
    max = abs(A[i][i])
    fila = i
    # Buscamos la fila con el numero mas grande dentro de la columna i
    for k in range(i, len(A)):
        # Nos quedamos con el numero de la fila para luego permutar
        # Updateamos el nuevo maximo
        if abs(A[k][i]) > max:
            fila = k
            max = A[k][i]
    # Devolvemos la fila
    return fila
            
def eliminacion_gausseana_pivoteo(T, x): 
    m = 0
    n = len(T[0])
    A0 = dc(T)
    A1 = dc(T)
    b = dc(x)

    for i in range(0,n):

        fila = encontrar_pivote(A0,i)  
        permutar_filas(A0, i, fila)
        permutar_filas(A1, i, fila)
        permutar_filas(b, i, fila)
        for j in range(i+1, n):
            # Aca va el pivoteo
            m = A0[j][i]/A0[i][i]                 # coeficiente
            b[j] = b[j] - (m * b[i])            # aplicar a b
            for k in range(i, n):
                A1[j][k] = A0[j][k] - (m * A0[i][k])
        A0 = dc(A1)

    return resolver_sistema_triangular_superior(A1,b)

def eliminacion_gausseana_tridiagonal(T,b):
    n = len(T[0])
    
    A = [0] * n
    B = [0] * n
    C = [0] * n

    # TODO: PREGUNTAR DIFERENCIA ENTRE 3.2 Y 3.3 - PREGUNTAR INFORME DE 1 - EXPERIMENTACION
    
    # ai xi−1 + bi xi + ci xi+1 = di
    
    # Armamos los vectores A B C 
    for i in range(n):
        B[i] = T[i][i]
        A[i] = 0 if i == 0 else T[i][i-1]
        C[i] = 0 if i == n-1 else T[i][i+1]
        
    # Resolvemos
    for i in range(1, n):
        coeficiente = A[i] / B[i-1]
        A[i] = A[i] - coeficiente * B[i-1]  # A_i - A_i / B_i-1 * B_i-1    
        B[i] = B[i] - coeficiente * C[i-1]  # B_i - A_i / B_i-1 * C_i-1

    for i in range(n):
        T[i][i] = B[i]
        if (i >= 1): T[i][i-1] = A[i] 
        if (i < n-1): T[i][i+1] = C[i]

    return resolver_sistema_triangular_superior(T,b)
    
class TestEliminacionGausseana(unittest.TestCase):
    def test_01_resolver_sistema_clase(self):
        A = [
            [2,1,-1,3],
            [-2,0,0,0],
            [4,1,-2,4],
            [-6,-1,2,-3]
            ]

        b = [13, -2, 24, -10]
        esperado_x = [1, -30, 7, 16]

        resultado_x = eliminacion_gausseana_naive(A, b)

        np.testing.assert_array_almost_equal(resultado_x, esperado_x, decimal=10)

    def test_02_resolver_sistema_internet(self):
        A = [
            [1,2,1],
            [1,0,1],
            [0,1,2]
            ]


        b = [0,2,1]
        
        x = eliminacion_gausseana_naive(dc(A),b)
        


        np.testing.assert_array_almost_equal(np.dot(np.array(A),np.array(x)), np.array(b), decimal=5)

    def test_03_resolver_tridiagonal(self):
        A = [
            [ 2, -1,  0,  0],
            [-1,  3, -1,  0],
            [ 0, -1,  3, -1],
            [ 0,  0, -1,  2]
            ]
    
        b = [1, 4, 7, 6]
        
        x = eliminacion_gausseana_tridiagonal(A, b)
        
        np.testing.assert_array_almost_equal(np.dot(A,x), b, decimal=5)

    def test_04_resolver_sistema_con_pivoteo(self):
        A = [
            [2, 1, 1],
            [4, 3, 1],
            [2, 3, 4]
        ]

        b = [3, 7, 10]
        x =  eliminacion_gausseana_pivoteo(A, b)
        
        np.testing.assert_array_almost_equal(np.dot(A,x), b, decimal=5)


if __name__ == "__main__":
    unittest.main()