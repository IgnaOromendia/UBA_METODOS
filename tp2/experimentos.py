import matplotlib.pyplot as plt
import subprocess as sp
import numpy as np
import csv

def escribir_matriz(mat_a_escribir, archivo):
    with open(archivo, 'w') as f:
        f.write(str(len(mat_a_escribir)) + "\n")
        f.write(str(len(mat_a_escribir[0])) + "\n")

        for matrices in mat_a_escribir:
            for A in matrices:
                f.write(str(len(A)) + " " + str(len(A[0])) + "\n")
                for i in range(len(A)):
                    for j in range(len(A[0])):
                        f.write(str(A[i][j]) + " ")
                    f.write("\n")

def leer_resultados_mp():
    reps = 5
    with open("output_mp.csv", 'r') as f:
        reader = csv.reader(f)
        
        next(reader) # Salteamos los headings

        mat_id = 1
        autovals = []
        autovecs = []
        it_prom = []
        it_des = []
        err_prom = []
        err_des = []
        datos_matrices = []

        for linea in reader:
            new_mat_id = int(linea[0])

            if mat_id != new_mat_id:
                datos_matrices.append((autovals.copy(), autovecs.copy(), it_prom.copy(), it_des.copy(), err_prom.copy(), err_des.copy()))
                mat_id = new_mat_id
                autovecs.clear()
                autovals.clear()
                it_prom.clear()
                it_des.clear()
                err_prom.clear()
                err_des.clear()
                
            autovals.append(float(linea[1]))
            autovecs.append(np.array([float(i) for i in linea[2:-4]]))
            it_prom.append(float(linea[-4]))
            it_des.append(float(linea[-3]))
            err_prom.append(float(linea[-2]))
            err_des.append(float(linea[-1]))

        datos_matrices.append((autovals.copy(), autovecs.copy(), it_prom.copy(), it_des.copy(), err_prom.copy(), err_des.copy())) # Agregamos los datos de la última matriz

    return datos_matrices

def calular_autovalores(archivo):
    resultado = sp.run(['./ejecMP', archivo], text=True)
    if resultado.returncode == 0:
        print("Datos en output_mp.csv")
    else:
        print("Error:", resultado.stderr)

def plot_convergencia(resultados, epsilons):
    cant_lambdas = len(resultados[0][0])

    for i in range(cant_lambdas):
        plt.errorbar(epsilons, resultados[i][2], yerr=resultados[i][3], label=f'$\lambda_{{{i+1}}}$')
        plt.scatter(epsilons, resultados[i][2])

    plt.ylabel('Iteraciones')
    plt.xlabel('$\epsilon$')
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
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
    archivo = 'input_mp.dat'

    epsilons = np.logspace(-4, 0,num=5)
    matrices_a_escribir = []

    matrices_por_eps = 5

    for eps in epsilons:
        v = [10, 10 - eps, 5, 2, 1]
        matrices_para_eps = []
        for _ in range(matrices_por_eps):
            matrices_para_eps.append(matriz_householder(v))
        matrices_a_escribir.append(matrices_para_eps)

    escribir_matriz(matrices_a_escribir, archivo)
    
    calular_autovalores(archivo)

    resultados_mp = leer_resultados_mp()
    
    # plot_convergencia(resultados_mp, epsilons)
    plot_error_metodo_potencia(resultados_mp, epsilons)