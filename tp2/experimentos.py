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

def plot_performance_Q_k():
    df = ds.leer_data_frame()

    Q_sample = [500,1000,5000]
    performance_Q = []

    for Q in Q_sample:
        X = ds.matriz_tokens(Q, df)

        train_set = df[df["split"] == "train"]
        test_set  = df[df["split"] == "test"]

        k_sample = [i for i in range(1,len(train_set))]

        print(Q)
        performance = []
        for k in k_sample:
            print(k)
            performance.append(ds.clasificador_de_genero(k, X, train_set, test_set))
        performance_Q.append(performance)

    for q in range(len(Q_sample)):
        plt.plot(k_sample, performance_Q[q], label=f'$Q = {{{Q_sample[q]}}}$')

    plt.ylabel('Performance')
    plt.xlabel('$k$')
    # plt.xscale("log")
    plt.legend()
    plt.grid(True)
    plt.title("Performance para distntos Q")
    plt.savefig('graficos/grafico_performance_Q.png')
    plt.show()
    
def plot_varianza_p():
    df = ds.leer_data_frame()

    Q_sample  = [500,1000,5000]
    p_sample  = [i for i in range(0,901,30)]
    varianzas = []

    for Q in Q_sample:
        X = ds.matriz_tokens(Q, df)

        train_set = df[df["split"] == "train"]
        X_train = X[train_set.index]  
        
        print(Q)
        varianzas_p = []
        for p in p_sample:
            print(p)
            var, V = ds.pca(X_train,'T')
            varianzas_p.append(var[p])

        varianzas.append(varianzas_p)

    for q in range(len(Q_sample)):
        plt.plot(p_sample, varianzas[q], label=f'$Q = {{{Q_sample[q]}}}$')

    plt.ylabel('Varianza')
    plt.xlabel('$p$')
    # plt.xscale("log")
    plt.legend()
    plt.grid(True)
    plt.title("Varianza PCA para distntos Q")
    plt.savefig('graficos/grafico_varianzas_Q.png')
    plt.show()
    

def matriz_householder(w):
    D = np.diag(w)

    # Preguntar rango del random
    v = np.random.uniform(-1, 1, size=(D.shape[0],1))
    v = v / np.linalg.norm(v)

    v = v / np.linalg.norm(v)

    H = np.eye(D.shape[0]) - 2 * (v @ v.T)

    return H @ D @ H.T

def experimentar_epsilon():
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


if __name__ == "__main__":
#    experimentar_epsilon()
    plot_performance_Q_k()
    plot_varianza_p()