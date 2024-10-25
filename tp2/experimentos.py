import matplotlib.pyplot as plt
import subprocess as sp
import numpy as np

def escribir_matriz(matrices, archivo):
    with open(archivo, 'w') as f:
        f.write(str(len(matrices)) + "\n")
        for A in matrices:
            f.write(str(len(A)) + " " + str(len(A[0])) + "\n")
            for i in range(len(A)):
                for j in range(len(A[0])):
                    f.write(str(A[i][j]) + " ")
                f.write("\n")

def leer_resultados_mp():
    with open("output_mp.out", 'r') as f:
        lines = f.readlines()
        
        cant_mat = int(lines[0].split()[0])
        cant_autoval = int(lines[1].split()[0]) 
        l = int(lines[2].split()[0]) # Cantidad de lineas a leer

        datos_matrices = []

        for k in range(1, cant_mat+1):
            line_idx = l * (k-1) + 3;

            autovalores = []
            autovectores = []
            iteraciones = []
            error = []

            i = line_idx 

            while i < l + line_idx:
                # print(i)
                data = lines[i].split()
                # print(data)
                autovalores.append(float(data[0]))
                iteraciones.append(int(data[1]))
                error.append(float(data[2]))
                v = []
                for j in range(1,cant_autoval):
                    v.append(float(lines[i + j].split()[0]))
                autovectores.append(v)
                i += cant_autoval + 1

            datos_matrices.append((autovalores, autovectores, iteraciones, error))

    return datos_matrices

def calular_autovalores(archivo, num):
    resultado = sp.run(['./ejecMP', archivo, str(num)], text=True)
    if resultado.returncode == 0:
        print("Datos en output_mp.out")
    else:
        print("Error:", resultado.stderr)

def plot_convergencia(resultados, epsilons):
    cant_lambdas = len(resultados[0][0])

    for i in range(cant_lambdas):
        plt.plot(epsilons, resultados[i][2], label=f'$\lambda_{{{i+1}}}$')

    plt.ylabel('Iteraciones')
    plt.xlabel('$\epsilon$')
    plt.xscale("log")
    plt.legend()
    # plt.savefig('graficos/grafico_convergencia.png')
    plt.show()

# plt.errorbar para error de aprox
def error_metodo(lambdas, errores, epsilons):
    for i in range(len(lambdas)):
        plt.plot(errores[i], epsilons[i], label='$\lambda_{i}$')

    plt.ylabel('Error')
    plt.xlabel('$\epsilon$')
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.legend()
    # plt.savefig('graficos/grafico_convergencia.png')
    plt.show()

def matriz_householder(w):
    D = np.diag(w)

    # Preguntar rango del random
    v = np.random.uniform(-1, 1, size=(D.shape[0],1))
    v = v / np.linalg.norm(v)

    v = v / np.linalg.norm(v)

    H = np.eye(D.shape[0]) - 2 * (v @ v.T)

    return H @ D @ H.T


if __name__ == "__main__":
    archivo = 'input_mp.dat'

    epsilons = np.logspace(-4, 0,num=5)
    matrices = []

    for eps in epsilons:
        v = [10, 10 - eps, 5, 2, 1]
        matrices.append(matriz_householder(v))

    escribir_matriz(matrices, archivo)
    
    calular_autovalores(archivo,len(v))

    resultados_mp = leer_resultados_mp()
    
    plot_convergencia(resultados_mp, epsilons)