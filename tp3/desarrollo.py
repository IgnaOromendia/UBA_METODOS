import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

def leer_datos(nombreArchivo, sujeto):
    valores_x = []
    valores_y = []
    with open(nombreArchivo, 'r') as file:
        for line in file.readlines():
            data = line.split()
            valores_x.append(float(data[0]))
            valores_y.append(float(data[sujeto]))

    x = np.array(valores_x)
    y = np.array(valores_y)

    x.reshape(1,x.shape[0])
    y.reshape(1,y.shape[0])

    return (x, y)

def SVD(A):
    return np.linalg.svd(A)

def cuadrados_minimos(X, y):
    U, S, V = SVD(X)

    S_inv = np.diag(1 / np.diag(S))

    return V @ S_inv @ U.T @ y

def cuadrados_minimos_reg(X,y,l):
    U, S, V = SVD(X)

    n = V.shape[0]

    S = np.diag(S)

    inv = np.linalg.inv(S@S + l * np.identity(n))

    print(U.shape)
    print(y.shape)

    return V @ S @ inv @ U.T @ y

def predecir_legrande(sujeto=1):
    x_aju, y_aju = leer_datos('./datos/ajuste.txt', sujeto)
    x_val, y_val = leer_datos('./datos/validacion.txt', sujeto)

    grados = []
    err_ajuste = []
    err_val = []

    for i in range(1, 2*x_aju.shape[0]):
        X_aju = np.polynomial.legendre.legvander(x_aju, i)
        X_val = np.polynomial.legendre.legvander(x_val, i)

        beta_pred = sc.linalg.lstsq(X_aju,y_aju)[0]

        y_aju_pred = (beta_pred.T@X_aju.T).T
        y_val_pred = (beta_pred.T@X_val.T).T

        ecm_ajuste = np.sum((y_aju_pred-y_aju)**2)
        ecm_val = np.sum((y_val_pred-y_val)**2)

        grados.append(i)
        err_ajuste.append(ecm_ajuste)
        err_val.append(ecm_val)

    plt.figure()
    plt.plot(err_ajuste,'.-', label='Ajuste')
    plt.plot(err_val,'.-', label='Val')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Grado')
    plt.ylabel('Error')
    plt.title('ECM')
    plt.show()

def predecir_reg(sujeto):
    x_aju, y_aju = leer_datos('./datos/ajuste.txt', sujeto)
    x_val, y_val = leer_datos('./datos/validacion.txt', sujeto)

    grados  = [i for i in range(1, 2*x_aju.shape[0])]
    lambdas = [i for i in range(1, 50)]

    g_opt = -1
    l_opt = 1
    err_val_por_l = []
    min_err = 1e8

    for g in grados:
        err_val = []

        for l in lambdas:
            beta_pred = cuadrados_minimos_reg(x_aju, y_aju, l)
            y_val_pred = (beta_pred.T@x_val.T).T

            ecm_val = np.sum((y_val_pred-y_val)**2)
            err_val.append(ecm_val)

            if min_err > ecm_val:
                g_opt = g
                l_opt = l
                min_err = ecm_val

        err_val_por_l.append(err_val)
    
    return g_opt, l_opt, err_val_por_l, grados, lambdas

def explorar_hiperparams():
    fig, axs = plt.subplots(2, 3)
    axs = axs.flatten()

    for sujeto in range(1,5):
        g_opt, l_opt, errs, grados, lambdas = predecir_reg(sujeto)

        color = axs[sujeto].pcolor(errs, cmap='hot')
        plt.colorbar(color, ax=axs[sujeto], label='u')

        axs[sujeto].set_title(f'Error para sujeto nº' + str(sujeto))
        axs[sujeto].set_xlabel('$\lambda$')
        axs[sujeto].set_ylabel('Grado')

    plt.tight_layout()
    plt.show()

explorar_hiperparams()





