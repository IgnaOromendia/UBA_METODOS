import unittest
import numpy as np
import matplotlib.pyplot as plt # type: ignore # type: ignore
from numba import jit

def multiplicar_matriz_vector(A,x):
    y = np.zeros(x.shape, dtype=np.float64)
    for i in range(A.shape[1]):
        y[i] = sum([A[i, j]*x[j] for j in range(A.shape[1])])
    return y
# Substitutions

@jit
def eliminacion_gausseana_naive(A, b): 
    m = 0
    n = A.shape[0]
    A0 = A.copy()
    b0 = b.copy()
    
    for i in range(0,n):
        for j in range(i+1, n):
            m = A0[j,i]/A0[i,i]                 # coeficiente
            b0[j] = b0[j] - (m * b0[i])            # aplicar a b
            for k in range(i, n):
                A0[j,k] = A0[j,k] - (m * A0[i,k])

            
    return backward_substitution(A0,b0)


@jit
def backward_substitution(A, b):
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)

    for i in range(n - 1, -1, -1):
        suma = 0
        for j in range(i,n): suma += A[i,j] * x[j]
        x[i] = (b[i] - suma) / A[i,i]  


    return x     

def foward_substitution(A, b):
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)

    for i in range(n):
        suma = 0
        for j in range(0,i): suma += A[i,j] * x[j]
        x[i] = (b[i] - suma) / A[i,i]  

    return x  

# Eliminacion Gaussiana

@jit
def foward_substitution(A, b):
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)

    for i in range(n):
        suma = 0
        for j in range(0,i): suma += A[i,j] * x[j]
        x[i] = (b[i] - suma) / A[i,i]  

    return x  

@jit
def eliminacion_gausseana_pivoteo(A, b): 
    m = 0
    n = A.shape[0]
    A0 = A.copy()
    b0 = b.copy()

    for i in range(0,n):
        fila = encontrar_pivote(A0,i)  
        permutar_filas(A0, i, fila)
        permutar_elementos_vector(b0, i, fila)
        for j in range(i+1, n):
            m = A0[j][i]/A0[i][i]                 # coeficiente
            b0[j] = b0[j] - (m * b0[i])            # aplicar a b
            for k in range(i, n):
                A0[j][k] = A0[j][k] - (m * A0[i][k])

    return backward_substitution(A0,b0)

@jit
def eliminacion_gaussiana(A, b): 
    m = 0
    n = A.shape[0]
    A0 = A.copy()
    b0 = b.copy()
    
    for i in range(0,n):
        for j in range(i+1, n):
            m = A0[j,i]/A0[i,i]                 # coeficiente
            b0[j] = b0[j] - (m * b0[i])            # aplicar a b
            for k in range(i, n):
                A0[j,k] = A0[j,k] - (m * A0[i,k])

            
    return backward_substitution(A0,b0)

@jit
def permutar_filas(A, i, j):
    copia = A[i].copy()
    A[i] = A[j]
    A[j] = copia

@jit
def permutar_elementos_vector(A, i, j):
    copia = A[i]
    A[i] = A[j]
    A[j] = copia

@jit
def encontrar_pivote(A, i):
    max = abs(A[i,i])
    fila = i
    # Buscamos la fila con el numero mas grande dentro de la columna i
    for k in range(i, len(A)):
        # Nos quedamos con el numero de la fila para luego permutar
        # Updateamos el nuevo maximo
        if abs(A[k,i]) > max:
            fila = k
            max = abs(A[k,i])
    # Devolvemos la fila
    return fila

@jit
def factorizar_LU(T):
    m = 0
    n = T.shape[0]
    A0 = T.copy()
    L = np.eye(n, dtype=np.float64)
    
    for i in range(0,n):
        for j in range(i+1, n):
            m = A0[j,i]/A0[i,i]
            L[j,i] = m
            for k in range(i, n):
                A0[j,k] = A0[j,k] - (m * A0[i,k])
    
    return L, A0    

@jit
def eliminacion_gausseana_tridiagonal(T,b):
    n = T.shape[0]
    L = np.eye(n, dtype=np.float64)
    A = np.zeros(n, dtype=np.float64)
    B = np.zeros(n, dtype=np.float64)
    C = np.zeros(n, dtype=np.float64)
    
    # ai xi−1 + bi xi + ci xi+1 = di
    
    # Armamos los vectores A B C 
    for i in range(n):
        B[i] = T[i][i]
        A[i] = 0 if i == 0 else T[i][i-1]
        C[i] = 0 if i == n-1 else T[i][i+1]
        
    # Resolvemos
    for i in range(1, n):
        m = A[i] / B[i-1]
        L[i,i-1] = m
        A[i] = A[i] - m * B[i-1]  # A_i - A_i / B_i-1 * B_i-1    
        B[i] = B[i] - m * C[i-1]  # B_i - A_i / B_i-1 * C_i-1
    
    for i in range(n):
        T[i][i] = B[i]
        if (i >= 1): T[i][i-1] = A[i] 
        if (i < n-1): T[i][i+1] = C[i]

    return backward_substitution(T,b)
    
@jit#
def factorizar_LU_tri(T):
    n = T.shape[0]
    L = np.eye(n, dtype=np.float64)
    A = np.zeros(n, dtype=np.float64)
    B = np.zeros(n, dtype=np.float64)
    C = np.zeros(n, dtype=np.float64)

    # ai xi−1 + bi xi + ci xi+1 = di
    
    # Armamos los vectores A B C 
    for i in range(n):
        B[i] = T[i][i]
        A[i] = 0 if i == 0 else T[i][i-1]
        C[i] = 0 if i == n-1 else T[i][i+1]
        
    # Resolvemos
    for i in range(1, n):
        coeficiente = A[i] / B[i-1]
        L[i,i-1] = coeficiente
        A[i] = A[i] - coeficiente * B[i-1]  # A_i - A_i / B_i-1 * B_i-1    
        B[i] = B[i] - coeficiente * C[i-1]  # B_i - A_i / B_i-1 * C_i-1
    
    for i in range(n):
        T[i][i] = B[i]
        if (i >= 1): T[i][i-1] = A[i] 
        if (i < n-1): T[i][i+1] = C[i]

    return L, T

def generar_laplaciano(n):
    A = np.zeros((n,n), dtype= np.float64)

    for i in range(n):
        A[i,i] = -2
        if i < n-1: A[i,i+1] = 1
        if i > 0:   A[i,i-1] = 1

    return A

def generar_u_0(n,r,m):
    u = np.zeros(n, dtype=np.float64)

    lower = n // 2 - r
    upper = n // 2 + r

    for i in range(lower,upper):
        u[i] = 1.0

    return u

def calcular_difusion(A,k,r,m):
    n = A.shape[0]
    u_0 = generar_u_0(n,r,m)
    u_1 = np.zeros(n, dtype=np.float64)
    difusion = [u_0]

    L,U = factorizar_LU_tri(A.copy())

    for _ in range(k):
        y   = backward_substitution(U,u_0)
        u_1 = backward_substitution(L, y)
        difusion.append(u_1)

    return difusion




# Laplaciano

def generar_laplaciano(n):
    A = np.zeros((n,n), dtype= np.float64)

    for i in range(n):
        A[i,i] = -2
        if i < n-1: A[i,i+1] = 1
        if i > 0:   A[i,i-1] = 1

    return A

def generar_d1(n):
    d = np.zeros(n, dtype=np.float64)
    d[n // 2 + 1] = 4 / n
    return d

def generar_d2(n):
    return np.full(n, 4 / n**2)

def generar_d3(n):
    d = np.zeros(n, dtype=np.float64)
    for i in range(n): d[i] = (-1 + ((2*i) / (n-1))) * (12/(n**2))
    return d

def verificar_implementacion_tri(n):
    d1 = generar_d1(n)
    d2 = generar_d2(n)
    d3 = generar_d3(n)

    # Matriz laplaciana
    A = generar_laplaciano(n)

    L,U = factorizar_LU_tri(A.copy())
    
    y   = foward_substitution(L,d1)
    u1  = backward_substitution(U,y)

    y   = foward_substitution(L,d2)
    u2  = backward_substitution(U,y)

    y   = foward_substitution(L,d3)
    u3  = backward_substitution(U,y)

    tams = [i for i in range(n)]

    plt.plot(tams,u1, label="(a)", color="steelblue")
    plt.plot(tams,u2, label="(b)", color="peru")
    plt.plot(tams,u3, label="(c)", color="forestgreen")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.show()

# Difusión

def generar_u0(n,r):
    u = np.zeros(n, dtype=np.float64)

    lower = (n // 2) - r
    upper = (n // 2) + r

    for i in range(lower+1,upper): u[i] = 1.0

    return u

def calcular_difusion(A,r,m):
    n = A.shape[0]
    u = [generar_u0(n,r)]

    L,U = factorizar_LU_tri(A.copy())

    for k in range(1,m):
        y  = foward_substitution(L,u[k-1])
        uk = backward_substitution(U, y)
        u.append(uk)

    return np.array(u)

def simular_difusion(alfa, n, r, m):
    # alfa  = multiplicador del laplaciano
    # m     = cantidad de pasos a simular
    # n     = tamaño del vector
    # r     = condensación

    # Matriz laplaciana
    T = generar_laplaciano(n)

    # Generamos A
    A = np.eye(n, dtype=np.float64) - alfa * T

    # Caluclamos 
    difusiones = calcular_difusion(A,r,m)

    # Plot
    X = np.cumsum(difusiones,axis=0)
    plt.plot(X,'k',alpha=0.3);
    plt.xlabel("k")
    plt.ylabel("x")
    plt.legend()
    plt.show()

#

class TestEliminaciongaussiana(unittest.TestCase):
    def test_01_resolver_sistema_clase(self):
        A = np.array([
            [2,1,-1,3],
            [-2,0,0,0],
            [4,1,-2,4],
            [-6,-1,2,-3]
            ], dtype=np.float64)

        b = np.array([13, -2, 24, -10], dtype=np.float64)
        esperado_x = np.array([1, -30, 7, 16], dtype=np.float64)

        resultado_x = eliminacion_gaussiana(A, b)

        np.testing.assert_array_almost_equal(resultado_x, esperado_x, decimal=10)

    def test_02_resolver_sistema_internet(self):
        A = np.array([
            [1,2,1],
            [1,0,1],
            [0,1,2]
            ], dtype=np.float64)


        b = np.array([0,2,1], dtype=np.float64)
        
        x = eliminacion_gaussiana(A.copy(),b)
        


        np.testing.assert_array_almost_equal(np.dot(np.array(A),np.array(x)), np.array(b), decimal=5)

    def test_03_resolver_tridiagonal(self):
        A = np.array([
            [ 2, -1,  0,  0],
            [-1,  3, -1,  0],
            [ 0, -1,  3, -1],
            [ 0,  0, -1,  2]
            ], dtype=np.float64)
    
        b = np.array([1, 4, 7, 6], dtype=np.float64)
        
        x = eliminacion_gausseana_tridiagonal(A,b)
        
        np.testing.assert_array_almost_equal(np.dot(A,x), b, decimal=5)

    def test_04_resolver_sistema_con_pivoteo(self):
        A = np.array([
            [2, 1, 1],
            [4, 3, 1],
            [2, 3, 4]
        ], dtype=np.float64)

        b = np.array([3, 7, 10], dtype=np.float64)
        x =  eliminacion_gausseana_pivoteo(A, b)
        
        
        np.testing.assert_array_almost_equal(np.dot(A,x), b, decimal=5)

    def test_05_factorizar_LU(self):
        A = np.array([
            [2,1,-1,3],
            [-2,0,0,0],
            [4,1,-2,4],
            [-6,-1,2,-3]
            ], dtype=np.float64)
    
        b = np.array([1, 4, 7, 6], dtype=np.float64)
        
        L, U = factorizar_LU(A.copy())
        L, U = factorizar_LU(A.copy())

        np.testing.assert_array_almost_equal(np.dot(L,U), A, decimal=5)

    def test_06_eliminacion_gaussiana_LU(self):
        A = np.array([
            [2,1,-1,3],
            [-2,0,0,0],
            [4,1,-2,4],
            [-6,-1,2,-3]
            ], dtype=np.float64)
    
        b = np.array([1, 4, 7, 6], dtype=np.float64)
        
        L, U = factorizar_LU(A.copy())

        np.testing.assert_array_almost_equal(np.dot(L,U), A, decimal=5)

        y = foward_substitution(L,b)
        x = backward_substitution(U,y)

        np.testing.assert_array_almost_equal(np.dot(A,x), b, decimal=5) 


    def test_07_factorizar_LU_tri(self):
        A = np.array([
            [ 2, -1,  0,  0],
            [-1,  3, -1,  0],
            [ 0, -1,  3, -1],
            [ 0,  0, -1,  2]
            ], dtype=np.float64)
    
        b = np.array([1, 4, 7, 6], dtype=np.float64)
        
        L,U = factorizar_LU_tri(A.copy())

        np.testing.assert_array_almost_equal(np.dot(L,U), A, decimal=5)

        y = foward_substitution(L,b)
        x = backward_substitution(U,y)
        
        np.testing.assert_array_almost_equal(np.dot(A,x), b, decimal=5)

if __name__ == "__main__":
    simular_difusion(1,101,10,1000)

    # verificar_implementacion_tri(101)

    # lista_size = [20,40,60,80,100,120,140,160,180,200]
    # t_n, t_p = calcular_tiempos_naive_vs_pivoteo(lista_size)
    # plt.plot(lista_size, t_n, color = 'blue',label='Eliminacion naive')
    # plt.plot(lista_size, t_p, color = 'red',label='Eliminacion pivoteo')
    # plt.ylabel('Tiempo en segundos')
    # plt.xlabel('Tamaño de la matriz')

    # #plt.xscale("log")
    # #plt.yscale("log")
    # plt.legend()
    # plt.show()
    
    unittest.main()