import numpy as np
import matplotlib.pyplot as plt


# Substitutions

def backward_substitution(A, b):
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)

    for i in range(n - 1, -1, -1):
        suma = 0
        for j in range(i, n):
            suma += A[i, j] * x[j]
        x[i] = (b[i] - suma) / A[i, i]

    return x


def forward_substitution(A, b):
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)

    for i in range(n):
        suma = 0
        for j in range(0, i):
            suma += A[i, j] * x[j]
        x[i] = (b[i] - suma) / A[i, i]

    return x


def backward_substitution_tri(A, b):
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)

    for i in range(n - 1, -1, -1):
        suma = 0
        if i < n - 1:
            suma = A[i, i + 1] * x[i + 1]
        x[i] = (b[i] - suma) / A[i, i]

    return x


def forward_substitution_tri(A, b):
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)

    for i in range(n):
        suma = 0
        if i > 0:
            suma = A[i, i - 1] * x[i - 1]
        x[i] = (b[i] - suma) / A[i, i]

    return x


# Eliminacion Gaussiana

def eliminacion_gaussiana(A, b):
    n = A.shape[0]
    A0 = A.copy()
    b0 = b.copy()

    for i in range(0, n):
        if A[i, i] == 0:
            print("AAAA!")
        for j in range(i + 1, n):
            m = A0[j, i] / A0[i, i]
            b0[j] = b0[j] - (m * b0[i])  # aplicar a b
            for k in range(i, n):
                A0[j, k] = A0[j, k] - (m * A0[i, k])

    return backward_substitution(A0, b0)


def permutar_filas(A, i, j):
    copia = A[i].copy()
    A[i] = A[j]
    A[j] = copia


def permutar_elementos_vector(A, i, j):
    copia = A[i]
    A[i] = A[j]
    A[j] = copia


def encontrar_pivote(A, i):
    max_local = abs(A[i, i])
    fila = i
    # Buscamos la fila con el numero mas grande dentro de la columna i
    for k in range(i, len(A)):
        # Nos quedamos con el numero de la fila para luego permutar
        # Updateamos el nuevo maximo
        if abs(A[k, i]) > max_local:
            fila = k
            max_local = abs(A[k, i])
    # Devolvemos la fila
    return fila


def eliminacion_gaussiana_pivoteo(A, b):
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


def eliminacion_gaussiana_tridiagonal(a, b, c, d):
    n = a.shape[0]
    A = np.zeros((n, n), dtype=np.float64)

    a0 = a.copy()
    b0 = b.copy()
    c0 = c.copy()
    d0 = d.copy()

    # Resolvemos
    for i in range(1, n):
        m = a0[i] / b0[i - 1]
        a0[i] = a0[i] - m * b0[i - 1]
        b0[i] = b0[i] - m * c0[i - 1]
        d0[i] = d0[i] - m * d0[i - 1]

    for i in range(n):
        A[i][i] = b0[i]
        if i >= 1:
            A[i][i - 1] = a0[i]
        if i < n - 1:
            A[i][i + 1] = c0[i]

    return backward_substitution_tri(A, d0)


# Factorización LU

def factorizar_LU(T):
    n = T.shape[0]
    A0 = T.copy()
    L = np.eye(n, dtype=np.float64)

    for i in range(0, n):
        for j in range(i + 1, n):
            m = A0[j, i] / A0[i, i]
            L[j, i] = m
            for k in range(i, n):
                A0[j, k] = A0[j, k] - (m * A0[i, k])

    for i in range(0, n):
        for j in range(0, i):
            A0[i][j] = 0

    return L, A0


def factorizar_LU_tri(a, b, c):
    n = a.shape[0]
    L = np.eye(n, dtype=np.float64)
    U = np.zeros((n, n), dtype=np.float64)

    a0 = a.copy()
    b0 = b.copy()
    c0 = c.copy()

    # Resolvemos
    for i in range(1, n):
        m = a0[i] / b0[i - 1]
        L[i, i - 1] = m
        a0[i] = a0[i] - m * b0[i - 1]
        b0[i] = b0[i] - m * c0[i - 1]

    for i in range(n):
        U[i, i] = b0[i]
        if i >= 1:
            U[i, i - 1] = a0[i]
        if i < n - 1:
            U[i, i + 1] = c0[i]

    return L, U


def resolver_LU(L, U, b):
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x


def resolver_LU_tri(L, U, b):
    y = forward_substitution_tri(L, b)
    x = backward_substitution_tri(U, y)
    return x


def diagonales(A):
    a = np.insert(np.diag(A, k=-1), 0, 0)
    b = np.diag(A, k=0)
    c = np.append(np.diag(A, k=1), 0)
    return a, b, c


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
    for i in range(n):
        d[i] = (-1 + ((2 * i) / (n - 1))) * (12 / (n ** 2))
    return d


def verificar_implementacion_tri(n):
    d1 = generar_d1(n)
    d2 = generar_d2(n)
    d3 = generar_d3(n)

    # Matriz trifiagonal laplaciana
    a, b, c = diagonales(generar_laplaciano(n))

    L, U = factorizar_LU_tri(a, b, c)

    u1 = resolver_LU_tri(L, U, d1)
    u2 = resolver_LU_tri(L, U, d2)
    u3 = resolver_LU_tri(L, U, d3)

    tams = [i for i in range(n)]

    plt.plot(tams, u1, label="(a)", color="steelblue")
    plt.plot(tams, u2, label="(b)", color="peru")
    plt.plot(tams, u3, label="(c)", color="forestgreen")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.savefig("graficos/verificacion_triangular.png", format="PNG", bbox_inches='tight')


# Difusión

def generar_u0(n, r):
    u = np.zeros(n, dtype=np.float64)

    lower = (n // 2) - r
    upper = (n // 2) + r

    for i in range(lower + 1, upper):
        u[i] = 1.0

    return u

def simular_difusion(alfa, n, r, m):
    A = np.eye(n, dtype=np.float64) - alfa * generar_laplaciano(n)
    u = [generar_u0(n, r)]

    a, b, c = diagonales(A)
    L, U = factorizar_LU_tri(a, b, c)

    for k in range(1, m):
        uk = resolver_LU_tri(L, U, u[k - 1])
        u.append(uk)

    return np.array(u)


def plot_diffusion_evolution(alfa=1, n=101, r=10, m=1000):
    difusiones = simular_difusion(alfa, n, r, m)
    plt.pcolor(difusiones.T, cmap='hot')
    plt.colorbar(label='u')
    plt.title(f'Mapa de calor')
    plt.xlabel('k')
    plt.ylabel('x')
    plt.savefig("graficos/mapa_de_calor.png", format="PNG", bbox_inches='tight')


# Difusión 2D
def generar_laplaciano_2d(n, u_n):
    T = np.zeros((n, n), dtype=np.float64)
    u_n = int(np.sqrt(n))
    for i in range(n):
        T[i, i] = -4
        if i != n - 1 and (i + 1) % u_n != 0:
            T[i, i + 1] = 1
        if i != 0 and i % u_n != 0:
            T[i, i - 1] = 1
        if i <= n - u_n - 1:
            T[i, i + u_n] = 1
        if i >= u_n:
            T[i, i - u_n] = 1

    return T

def generar_u0_2d(n):
    u = np.zeros((n, n), dtype=np.float64)
    u[(n // 2), (n // 2)] = 100
    return u

def mantener_constantes(uk, n):
    uk[n // 2, n // 2] = 100
    uk[:, 0] = 0
    uk[:, n - 1] = 0
    uk[0, :] = 0
    uk[n - 1, :] = 0
    return uk

def simular_difusion_2d(alfa, n, m):
    A = np.eye(n * n, dtype=np.float64) - alfa * generar_laplaciano_2d(n * n, n)

    u = [generar_u0_2d(n)]

    L, U = factorizar_LU(A.copy())

    for k in range(1, m):
        uk = resolver_LU(L, U, u[k - 1].flatten())
        uk = uk.reshape(n, n)
        u.append(mantener_constantes(uk,n))

    return u


def plot_diffusion_evolution_2D(alfa=0.1, n=15, m=100, t=50):
    difusiones = simular_difusion_2d(alfa, n, m)
    plt.pcolor(difusiones[t], cmap='hot')
    plt.colorbar(label='u')
    plt.title(f'Mapa de calor')
    plt.xlabel('k')
    plt.ylabel('x')
    plt.show()


if __name__ == "__main__":
    # plot_diffusion_evolution()
    plot_diffusion_evolution_2D()
