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
    U, S, Vt = np.linalg.svd(X)

    n = Vt.shape[0]
    m = U.shape[0]

    S_inv = np.zeros((n,min(n,m)))

    U = U[:,:n]

    for i in range(min(n,m)): S_inv[i,i] = 1 / S[i]

    return Vt.T @ S_inv @ U.T @ y

def cuadrados_minimos_reg(X,y,l):
    U, s, Vt = np.linalg.svd(X)

    m = U.shape[0]
    n = Vt.shape[0]

    S = np.zeros((m,n))
    for i in range(len(s)): S[i,i] = s[i]

    A = np.zeros((n,min(n,m)))

    for i in range(min(n,m)): A[i,i] = S[i,i] / (S[i,i]**2 + l)
    
    U = U[:,:n]
    
    return Vt.T @ A @ U.T @ y

def ecm_sin_regularizacion(x_aju, y_aju, x_val, y_val, grado):
    X_aju = np.polynomial.legendre.legvander(x_aju, grado)
    X_val = np.polynomial.legendre.legvander(x_val, grado)

    beta_pred = cuadrados_minimos(X_aju, y_aju)

    y_aju_pred = (beta_pred.T@X_aju.T).T
    y_val_pred = (beta_pred.T@X_val.T).T

    ecm_ajuste = np.sum((y_aju_pred-y_aju)**2)
    ecm_val = np.sum((y_val_pred-y_val)**2)

    return ecm_ajuste, ecm_val

def ecm_con_regularizacion(x_aju, y_aju, x_val, y_val, grado, l):
    X_aju = np.polynomial.legendre.legvander(x_aju, grado)
    X_val = np.polynomial.legendre.legvander(x_val, grado)

    beta_pred = cuadrados_minimos_reg(X_aju, y_aju, l)

    y_val_pred = (beta_pred.T@X_val.T).T

    ecm_val = np.sum((y_val_pred-y_val)**2)

    return ecm_val







