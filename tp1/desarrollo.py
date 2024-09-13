import sys
import numpy as np
import matplotlib.pyplot as plt  # type: ignore para que no se queje vs code
from numba import jit  # type: ignore


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
        x[i] = (b[i] - suma) / A[i, i]

    return x


# Eliminacion Gaussiana

@jit
def eliminacion_gaussiana(A, b):
    m = 0
    n = A.shape[0]
    A0 = A.copy()
    b0 = b.copy()

    for i in range(0, n):
        for j in range(i + 1, n):
            m = A0[j, i] / A0[i, i]  # coeficiente
            b0[j] = b0[j] - (m * b0[i])  # aplicar a b
            for k in range(i, n):
                A0[j, k] = A0[j, k] - (m * A0[i, k])

    return backward_substitution(A0, b0)


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
    max = abs(A[i, i])
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

    for i in range(0, n):
        fila = encontrar_pivote(A0, i)
        permutar_filas(A0, i, fila)
        permutar_elementos_vector(b0, i, fila)
        for j in range(i + 1, n):
            m = A0[j][i] / A0[i][i]  # coeficiente
            b0[j] = b0[j] - (m * b0[i])  # aplicar a b
            for k in range(i, n):
                A0[j][k] = A0[j][k] - (m * A0[i][k])

    return backward_substitution(A0, b0)


@jit
def eliminacion_gaussiana_tridiagonal(T, b):
    n = T.shape[0]
    L = np.eye(n, dtype=np.float64)
    A = np.zeros(n, dtype=np.float64)
    B = np.zeros(n, dtype=np.float64)
    C = np.zeros(n, dtype=np.float64)

    # ai xi−1 + bi xi + ci xi+1 = di

    # Armamos los vectores A B C
    for i in range(n):
        B[i] = T[i][i]
        A[i] = 0 if i == 0 else T[i][i - 1]
        C[i] = 0 if i == n - 1 else T[i][i + 1]

    # Resolvemos
    for i in range(1, n):
        m = A[i] / B[i - 1]
        L[i, i - 1] = m
        A[i] = A[i] - m * B[i - 1]  # A_i - A_i / B_i-1 * B_i-1
        B[i] = B[i] - m * C[i - 1]  # B_i - A_i / B_i-1 * C_i-1

    for i in range(n):
        T[i][i] = B[i]
        if (i >= 1): T[i][i - 1] = A[i]
        if (i < n - 1): T[i][i + 1] = C[i]

    return backward_substitution(T, b)


# Factorización LU

@jit
def factorizar_LU(T):
    m = 0
    n = T.shape[0]
    A0 = T.copy()
    L = np.eye(n, dtype=np.float64)

    for i in range(0, n):
        for j in range(i + 1, n):
            m = A0[j, i] / A0[i, i]
            L[j, i] = m
            for k in range(i, n):
                A0[j, k] = A0[j, k] - (m * A0[i, k])

    return L, A0


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
        A[i] = 0 if i == 0 else T[i][i - 1]
        C[i] = 0 if i == n - 1 else T[i][i + 1]

    # Resolvemos
    for i in range(1, n):
        coeficiente = A[i] / B[i - 1]
        L[i, i - 1] = coeficiente
        A[i] = A[i] - coeficiente * B[i - 1]  # A_i - A_i / B_i-1 * B_i-1
        B[i] = B[i] - coeficiente * C[i - 1]  # B_i - A_i / B_i-1 * C_i-1

    for i in range(n):
        T[i][i] = B[i]
        if (i >= 1): T[i][i - 1] = A[i]
        if (i < n - 1): T[i][i + 1] = C[i]

    return L, T


# Laplaciano

def generar_laplaciano(n):
    T = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(T, -2)
    np.fill_diagonal(T[1:], 1)
    np.fill_diagonal(T[:, 1:], 1)
    return T


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

    # Calculamos
    difusiones = calcular_difusion(A, r, m)

    return difusiones

def plot_diffusion_evolution(alphas, n, r, m):
    """Plot the diffusion evolution for different alpha values."""
    fig, axes = plt.subplots(len(alphas), 1, figsize=(10, 5 * len(alphas)))

    if len(alphas) == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one subplot

    for idx, alfa in enumerate(alphas):
        # Compute diffusion for the given alpha
        difusion_result = simular_difusion(alfa, n, r, m)

        # Plot the evolution
        X = np.arange(difusion_result.shape[0])
        center_index = n // 2

        axes[idx].plot(X, difusion_result[:, center_index], label=f'α = {alfa}')
        axes[idx].set_title(f'Diffusion Evolution for α = {alfa}')
        axes[idx].set_xlabel('Time step')
        axes[idx].set_ylabel('Value at center')
        axes[idx].legend()

    plt.tight_layout()
    plt.show()


    # Parameters
    n = 101  # Grid size
    r = 10  # Condensation size
    m = 1000  # Number of time steps
    alphas = [0.5, 1.0, 2.0]  # Different values of alpha to compare



if __name__ == "__main__":
    #simular_difusion(0.1, 101, 10, 1000)
    # verificar_implementacion_tri(101)
    #difusion_result = simular_difusion_2d(0.1, 15, 100, 100)
    #plot_heatmaps(difusion_result)
    # Plot the diffusion evolution for different alpha values
    # Parameters
    n = 101  # Grid size
    r = 10  # Condensation size
    m = 1000  # Number of time steps
    alphas = [1.0, 2.0]  # Different values of alpha to compare
    plot_diffusion_evolution(alphas, n, r, m)


