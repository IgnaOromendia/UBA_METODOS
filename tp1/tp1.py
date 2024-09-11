import unittest
import numpy as np
import matplotlib.pyplot as plt
import time

def multiplicar_matriz_vector(A,x):
    return np.dot(A,x)

def backward_substitution(A, b):
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)

    for i in range(n - 1, -1, -1):
        suma = 0
        for j in range(i,n): suma += A[i,j] * x[j]
        x[i] = (b[i] - suma) / A[i,i]  

    return x     

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

def permutar_filas(A, i, j):
    copia = A[i].copy()
    A[i] = A[j]
    A[j] = copia
    
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
            
def eliminacion_gausseana_pivoteo(A, b): 
    m = 0
    n = A.shape[0]
    A0 = A.copy()
    b0 = b.copy()

    for i in range(0,n):
        fila = encontrar_pivote(A0,i)  
        permutar_filas(A0, i, fila)
        permutar_filas(b0, i, fila)
        for j in range(i+1, n):
            m = A0[j][i]/A0[i][i]                 # coeficiente
            b0[j] = b0[j] - (m * b0[i])            # aplicar a b
            for k in range(i, n):
                A0[j][k] = A0[j][k] - (m * A0[i][k])

        

    return backward_substitution(A0,b0)

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

def eliminacion_gausseana_tridiagonal(T,b):
    n = T.shape[0]
    
    A = np.zeros(n, dtype=np.float64)
    B = np.zeros(n, dtype=np.float64)
    C = np.zeros(n, dtype=np.float64)

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

    return backward_substitution(T,b)
    
class TestEliminacionGausseana(unittest.TestCase):
    def test_01_resolver_sistema_clase(self):
        A = np.array([
            [2,1,-1,3],
            [-2,0,0,0],
            [4,1,-2,4],
            [-6,-1,2,-3]
            ], dtype=np.float64)

        b = np.array([13, -2, 24, -10], dtype=np.float64)
        esperado_x = np.array([1, -30, 7, 16], dtype=np.float64)

        resultado_x = eliminacion_gausseana_naive(A, b)

        np.testing.assert_array_almost_equal(resultado_x, esperado_x, decimal=10)

    def test_02_resolver_sistema_internet(self):
        A = np.array([
            [1,2,1],
            [1,0,1],
            [0,1,2]
            ], dtype=np.float64)


        b = np.array([0,2,1], dtype=np.float64)
        
        x = eliminacion_gausseana_naive(A.copy(),b)
        


        np.testing.assert_array_almost_equal(np.dot(np.array(A),np.array(x)), np.array(b), decimal=5)

    def test_03_resolver_tridiagonal(self):
        A = np.array([
            [ 2, -1,  0,  0],
            [-1,  3, -1,  0],
            [ 0, -1,  3, -1],
            [ 0,  0, -1,  2]
            ], dtype=np.float64)
    
        b = np.array([1, 4, 7, 6], dtype=np.float64)
        
        x = eliminacion_gausseana_tridiagonal(A, b)
        
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
        
        L, U = factorizar_LU(A)

        np.testing.assert_array_almost_equal(np.dot(L,U), A, decimal=5)

    def test_06_eliminacion_gausseana_LU(self):
        A = np.array([
            [2,1,-1,3],
            [-2,0,0,0],
            [4,1,-2,4],
            [-6,-1,2,-3]
            ], dtype=np.float64)
    
        b = np.array([1, 4, 7, 6], dtype=np.float64)
        
        L, U = factorizar_LU(A)

        # COMPLETAR

def calcular_tiempo_avr(f, l, h, s, cant):
    time_avg = 0
    for i in range(cant):
        start_time = time.time()
        calcular_sistema_de_n_elementos(l,h,s, f)
        end_time = time.time() 
        time_avg = time_avg + (end_time-start_time)
    
    return (time_avg/cant)

def calcular_sistema_de_n_elementos(l, h, s, f):
  np.random.seed(9)
  A = np.random.uniform(l, h, (s,s))
  b = np.random.uniform(l, h, s)
  return f(A, b)

def calcular_tiempos_naive_vs_pivoteo(l):
    tiempos_pivoteo = []
    tiempos_naive = []
    i = 0
    for s in l:
        t1 =  calcular_tiempo_avr(eliminacion_gausseana_naive, 10**-3,10**3, s, 10)
        t2 =  calcular_tiempo_avr(eliminacion_gausseana_pivoteo, 10**-3,10**3, s, 10)
        tiempos_naive.append(t1)
        tiempos_pivoteo.append(t2)
        print("Tiempo nainve: ", t1, " Tiempo pivoteo: ", t2, "En el paso: ", i)
        i = i+1 
    
    return tiempos_naive, tiempos_pivoteo

if __name__ == "__main__":
    lista_size = [20,40,60,80,100,120,140,160,180,200]
    t_n, t_p = calcular_tiempos_naive_vs_pivoteo(lista_size)
    plt.plot(lista_size, t_n, color = 'blue',label='Eliminacion naive')
    plt.plot(lista_size, t_p, color = 'red',label='Eliminacion pivoteo')
    plt.ylabel('Tiempo en segundos')
    plt.xlabel('Tamaño de la matriz')

    #plt.xscale("log")
    #plt.yscale("log")
    plt.legend()
    plt.show()
    #unittest.main()
    unittest.main()