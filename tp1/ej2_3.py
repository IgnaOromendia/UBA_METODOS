from desarrollo import *

def explorar_error_numerico(lista_epsilons, fl):
    lista_resultados = []
    
    for i in range(len(lista_epsilons)):
        eps = lista_epsilons[i]
        A = np.array([[1, 2+eps, 3-eps],
                [1-eps, 2, 3+eps],
                [1+eps, 2-eps, 3]], dtype=fl)
        x = np.array([1,1,1], dtype=fl)
        b = np.array([6,6,6], dtype=fl)
        v = eliminacion_gaussiana_pivoteo(A, b)
        lista_resultados.append(np.max(abs(x-v)))

    return lista_resultados


if __name__ == "__main__":
    lista_epsilons = np.logspace(np.log10(10**-6), np.log10(1), num=100)
    
    a = explorar_error_numerico(lista_epsilons, np.float64)
    b = explorar_error_numerico(lista_epsilons, np.float32)
    
    plt.plot(lista_epsilons, a, 'o',color = 'blue', label="64bits")
    plt.plot(lista_epsilons, b,'o',color = 'red', label="32bits")
    
    plt.ylabel('Error numerico')
    plt.xlabel('Epsilon')
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Error numerico en funcion del epsilon")

    plt.savefig('graficos/grafico_error_numerico.png')
    plt.legend()
    plt.show()
