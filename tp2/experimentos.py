import matplotlib.pyplot as plt
import numpy as np
import desarrollo as ds

def plot_convergencia(resultados, epsilons):
    cant_lambdas = len(resultados[0][0])

    for i in range(cant_lambdas):
        plt.errorbar(epsilons, resultados[i][2], yerr=resultados[i][3], label=f'$\lambda_{{{i+1}}}$')
        plt.scatter(epsilons, resultados[i][2])

    plt.ylabel('Iteraciones')
    plt.xlabel('$\epsilon$')
    plt.xscale("log")
    # plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.title("Iteraciones de cada $\lambda$ en el método de la potencia")
    plt.savefig('graficos/grafico_convergencia.png')
    plt.show()

def plot_error_metodo_potencia(resultados, epsilons):
    cant_lambdas = len(resultados[0][0])

    for i in range(cant_lambdas):
        plt.errorbar(epsilons, resultados[i][4], yerr=resultados[i][5], label=f'$\lambda_{{{i+1}}}$')
        plt.scatter(epsilons, resultados[i][4])

    plt.ylabel('Error')
    plt.xlabel('$\epsilon$')
    plt.xscale("log")
    plt.legend()
    plt.grid(True)
    plt.title("Error de cada $\lambda$ en el método de la potencia")
    plt.savefig('graficos/grafico_error_metodo_potencia.png')
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
    archivoEntrada = 'exper_input_mp.dat'
    archivoSalida  = 'exper_output_mp.csv' 

    epsilons = np.logspace(-4, 0,num=5)
    matrices_a_escribir = []

    matrices_por_eps = 5

    for eps in epsilons:
        v = [10, 10 - eps, 5, 2, 1]
        matrices_para_eps = []
        for _ in range(matrices_por_eps):
            matrices_para_eps.append(matriz_householder(v))
        matrices_a_escribir.append(matrices_para_eps)

    ds.escribir_input_experimento_mp(matrices_a_escribir, archivoEntrada)
    
    ds.calular_autovalores(archivoEntrada, archivoSalida, '1')

    resultados_mp = ds.leer_output_experimento_mp(archivoSalida)
    
    plot_convergencia(resultados_mp, epsilons)
    plot_error_metodo_potencia(resultados_mp, epsilons)