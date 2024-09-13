import numpy as np

# Substitutions

@jit
def backward_substitution(A, b):
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)

    for i in range(n - 1, -1, -1):
        suma = 0
        for j in range(i, n): suma += A[i, j] * x[j]
        x[i] = (b[i] - suma) / A[i, i]

    return x

 
@jit
def forward_substitution(A, b):
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)

    for i in range(n):
        suma = 0
        for j in range(0, i): suma += A[i, j] * x[j]
    return x


# Eliminacion Gaussiana

@jit
def eliminacion_gaussiana(A, b): 
    m = 0
    n = A.shape[0]
    A0 = A.copy()
    b0 = b.copy()
        for j in range(i+1, n):
            m = A0[j,i]/A0[i,i]                 # coeficiente

    for i in range(0, n):
        for j in range(i + 1, n):
            m = A0[j, i] / A0[i, i]  # coeficiente
            b0[j] = b0[j] - (m * b0[i])  # aplicar a b
            for k in range(i, n):
                A0[j,k] = A0[j,k] - (m * A0[i,k])
                A0[j, k] = A0[j, k] - (m * A0[i, k])

    return backward_substitution(A0, b0)

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
    fila = i
    # Buscamos la fila con el numero mas grande dentro de la columna i
    for k in range(i, len(A)):
        # Nos quedamos con el numero de la fila para luego permutar
        # Updateamos el nuevo maximo
        if abs(A[k, i]) > max:
            fila = k
            max = abs(A[k, i])
    # Devolvemos la fila
    return fila

@jit
def eliminacion_gaussiana_pivoteo(A, b):
    m = 0
    n = A.shape[0]
    A0 = A.copy()
    b0 = b.copy()

    for i in range(0,n):
        fila = encontrar_pivote(A0,i)  
    for i in range(0, n):
        fila = encontrar_pivote(A0, i)
        permutar_filas(A0, i, fila)
        permutar_elementos_vector(b0, i, fila)
            for k in range(i, n):
                A0[j][k] = A0[j][k] - (m * A0[i][k])

    return backward_substitution(A0, b0)

@jit
    n = T.shape[0]
    A = np.zeros(n, dtype=np.float64)
    B = np.zeros(n, dtype=np.float64)
    C = np.zeros(n, dtype=np.float64)
    
    # ai xi−1 + bi xi + ci xi+1 = di
    

    for i in range(n):
        B[i] = T[i][i]
        A[i] = 0 if i == 0 else T[i][i - 1]
        C[i] = 0 if i == n - 1 else T[i][i + 1]

    # Resolvemos
    for i in range(1, n):
        m = A[i] / B[i-1]
        A[i] = A[i] - m * B[i-1]  # A_i - A_i / B_i-1 * B_i-1    
        A[i] = A[i] - m * B[i - 1]  # A_i - A_i / B_i-1 * B_i-1
        B[i] = B[i] - m * C[i - 1]  # B_i - A_i / B_i-1 * C_i-1
    for i in range(n):
        T[i][i] = B[i]
        if (i >= 1): T[i][i - 1] = A[i]
        if (i < n - 1): T[i][i + 1] = C[i]

    return backward_substitution(T, b)

    return backward_substitution(T,b)

# Factorización LU

@jit
def factorizar_LU(T):
    m = 0
    n = T.shape[0]
    A0 = T.copy()
    L = np.eye(n, dtype=np.float64)

    for i in range(0, n):
            L[j, i] = m
            for k in range(i, n):
                A0[j, k] = A0[j, k] - (m * A0[i, k])


@jit
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
        m = A[i] / B[i-1]
        L[i,i-1] = m
        A[i] = A[i] - m * B[i-1]  # A_i - A_i / B_i-1 * B_i-1    
        B[i] = B[i] - m * C[i-1]  # B_i - A_i / B_i-1 * C_i-1
    
    for i in range(n):
        T[i][i] = B[i]
        if (i >= 1): T[i][i-1] = A[i] 
        if (i < n-1): T[i][i+1] = C[i]

    return L, T

# Laplaciano

def generar_laplaciano(n):
    A = np.zeros((n,n), dtype= np.float64)

    for i in range(n):
        A[i,i] = -2
        if i < n-1: A[i,i+1] = 1
        if i > 0:   A[i,i-1] = 1
        if i < n - 1: A[i, i + 1] = 1

    return A


def generar_d1(n):
    d = np.zeros(n, dtype=np.float64)
    d[n // 2 + 1] = 4 / n
    return d


def generar_d2(n):
    return np.full(n, 4 / n ** 2)


def generar_d3(n):
    d = np.zeros(n, dtype=np.float64)
    for i in range(n): d[i] = (-1 + ((2 * i) / (n - 1))) * (12 / (n ** 2))
    return d


def verificar_implementacion_tri(n):
    d1 = generar_d1(n)
    d2 = generar_d2(n)
    d3 = generar_d3(n)

    # Matriz laplaciana
    A = generar_laplaciano(n)

    L, U = factorizar_LU_tri(A.copy())

    y = forward_substitution(L, d1)
    u1 = backward_substitution(U, y)

    y = forward_substitution(L, d2)
    u2 = backward_substitution(U, y)

    y = forward_substitution(L, d3)
    u3 = backward_substitution(U, y)

    tams = [i for i in range(n)]

    plt.plot(tams, u1, label="(a)", color="steelblue")
    plt.plot(tams, u2, label="(b)", color="peru")
    plt.plot(tams, u3, label="(c)", color="forestgreen")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.show()


# Difusión

def generar_u0(n, r):
    u = np.zeros(n, dtype=np.float64)

    lower = (n // 2) - r
    upper = (n // 2) + r

    for i in range(lower + 1, upper): u[i] = 1.0

    return u


def calcular_difusion(A, r, m):
    n = A.shape[0]
    u = [generar_u0(n, r)]

    L, U = factorizar_LU_tri(A.copy())

    for k in range(1, m):
        y = forward_substitution(L, u[k - 1])
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
    difusiones = calcular_difusion(A, r, m)

    # Plot
    X = np.cumsum(difusiones, axis=0)
    plt.plot(X, 'k', alpha=0.3);
    plt.xlabel("k")
    plt.ylabel("x")
    plt.show()


def generar_u0_2d(n, r):
    u0 = np.zeros((n, n))
    centro = n // 2
    for i in range(n):
        for j in range(n):
            if np.sqrt((i - centro) ** 2 + (j - centro) ** 2) < r:
                u0[i, j] = 1
    return u0


def calcular_difusion_2d(A, r, m):
    n = A.shape[0]
    u = [generar_u0_2d(n, r)]
    L, U = factorizar_LU_tri(A.copy())

    for k in range(1, m):
        u_k_prev = u[-1].flatten()
        y = forward_substitution(L, u_k_prev)
        u_k = backward_substitution(U, y)
        u.append(u_k.reshape((n, n)))

    return np.array(u)


def probar_difusion_2d():
    # Example usage
    n = 50  # Size of the grid (n x n)
    r = 10  # Initial condition parameter
    m = 100  # Number of time steps

    # Define a 2D diffusion matrix (e.g., Laplacian)
    A = np.zeros((n * n, n * n))
    for i in range(n):
        for j in range(n):
            index = i * n + j
            if i > 0:
                A[index, (i - 1) * n + j] = 1
            if i < n - 1:
                A[index, (i + 1) * n + j] = 1
            if j > 0:
                A[index, i * n + (j - 1)] = 1
            if j < n - 1:
                A[index, i * n + (j + 1)] = 1
            A[index, index] = -4  # Central element

    # Run the diffusion simulation
    results = calcular_difusion_2d(A, r, m)

    # Example: plot the results for the final time step
    plt.imshow(results[-1], cmap='hot', interpolation='none')
    plt.colorbar()
    plt.title('2D Diffusion at Final Time Step')
    plt.show()


if __name__ == "__main__":
    #simular_difusion(1, 101, 10, 1000)
    # verificar_implementacion_tri(101)
    probar_difusion_2d()
