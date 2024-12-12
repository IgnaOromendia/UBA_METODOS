import matplotlib.pyplot as plt
import multiprocessing as mp
import desarrollo as ds
import numpy as np

def plot_error_sujeto_1():
    grados, err_ajuste, err_val = ds.predecir_sin_reg(sujeto=1)

    plt.plot(grados, err_ajuste,'.-', label='Ajuste')
    plt.plot(grados, err_val,'.-', label='Val')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.xlabel('Grado')
    plt.ylabel('Error')
    plt.title('ECM de ajuste y validación')
    plt.savefig('graficos/grafico_error_sujeto1.png')
    plt.show()

def heatmap_sujetos(error, sujeto, ax, cant_lambdas=100):
    cax = ax.pcolor(error, cmap='hot', shading='auto') 
    plt.colorbar(cax, ax=ax, label='Error')

    filas, cols = error.shape

    y_ticks = [0, 0.25, 0.5, 0.75, 1]
    y_positions = np.array(y_ticks) * (cant_lambdas - 1) 

    ax.set_yticks(y_positions + 0.5) 
    ax.set_yticklabels(y_ticks)

    x_positions = np.arange(0, cols, 20)
    x_labels = range(5, 5 + cols, 20) 
    ax.set_xticks(x_positions + 0.5)  
    ax.set_xticklabels(x_labels)

    ax.set_title(f'Mapa de Calor del error del sujeto nº {sujeto}')
    ax.set_xlabel('Grado')

def explorar_en_sujeto(sujeto):
    x_aju, y_aju = ds.leer_datos('./datos/ajuste.txt', sujeto)
    x_val, y_val = ds.leer_datos('./datos/validacion.txt', sujeto)

    max_g = 2*x_aju.shape[0] + 6
    cant_l = 100

    lambdas = np.logspace(1e-8, 1, cant_l)
    grados  = [i for i in range(5, max_g)]

    g_opt = -1
    l_opt = -1
    error = np.array([[0] * max_g for _ in range(cant_l)])
    min_err = 1e8

    for g in grados:
        for i,l in enumerate(lambdas):
            error[i,g] = ds.predecir_con_reg(x_aju, y_aju, x_val, y_val, g, l)
            if error[i,g] < min_err:
                min_err = error[i,g]
                g_opt = g
                l_opt = l

    print("Sujeto nº" + str(sujeto) + " terminó con g: " + str(g_opt) + " y l:" + str(l_opt))

    return sujeto, error, (g_opt, l_opt)

def explorar_hiperparametros():
    sujetos = [[s] for s in range(1,6)]

    with mp.Pool(processes=5) as pool:
        resultados = pool.starmap(explorar_en_sujeto, sujetos)

    optimos = [None] * 5
    errores = [None] * 5
    
    for sujeto, error, opt in resultados:
        optimos[sujeto-1] = opt
        errores[sujeto-1] = error

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    axes = axes.flatten()

    axes[-1].axis('off')

    for sujeto, error, _ in resultados:
        error_plt = error[:,5:]
        heatmap_sujetos(error_plt, sujeto, axes[sujeto-1])

    plt.tight_layout()
    plt.savefig("graficos/heatmap.png")
    plt.show()


if __name__ == "__main__":
    print("Experimentando...")
    # plot_error_sujeto_1()
    explorar_hiperparametros()