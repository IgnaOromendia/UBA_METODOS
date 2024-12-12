import numpy as np
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

    # x.reshape(1,x.shape[0])
    # y.reshape(1,y.shape[0])

    return (x, y)

def cuadrados_minimos(X, y):
    U, S, V = np.linalg.svd(X)

    S_inv = np.diag(1 / np.diag(S))

    return V @ S_inv @ U.T @ y

def cuadrados_minimos_reg(X,y,l):
    U, s, V = np.linalg.svd(X)

    m = U.shape[0]
    n = V.shape[0]

    S = np.zeros((m,n))
    for i in range(len(s)): S[i,i] = s[i]

    A = np.zeros((n,min(n,m)))

    for i in range(min(n,m)): A[i,i] = S[i,i] / (S[i,i]**2 + l)
    
    U = U[:,:n]
    
    return V @ A @ U.T @ y

def predecir_sin_reg(sujeto):
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

    return grados, err_ajuste, err_val

def predecir_con_reg(x_aju, y_aju, x_val, y_val, g, l):
    X_aju = np.polynomial.legendre.legvander(x_aju, g)
    X_val = np.polynomial.legendre.legvander(x_val, g)

    beta_pred = cuadrados_minimos_reg(X_aju, y_aju, l)

    y_val_pred = (beta_pred.T@X_val.T).T

    return np.sum((y_val_pred-y_val)**2)







