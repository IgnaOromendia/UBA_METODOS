import numpy as np
import pandas as pd
from scipy import stats #type: ignore

mapMovieIndex = {"science fiction": 0, "romance":1, "crime":2, "western":3}
mapIndexMovie = {0:"science fiction", 1:"romance", 2:"crime", 3:"western"}

def cov(x,y):
    n = len(x)
    media_x = sum(x) / n
    media_y = sum(y) / n

    prod = 0
    for i in range(n):
        prod += (x[i] - media_x) * (y[i] - media_y)

    return prod / (n - 1)

def corr(x,y):
    n = len(x)
    media_x = sum(x) / n
    media_y = sum(y) / n

    numerador = 0
    divisor = 0

    for i in range(n):
        numerador += (x[i] - media_x) * (y[i] - media_y)
        divisor += (x[i] - media_x)**2 * (y[i] - media_y)**2

    divisor = np.sqrt(divisor)

    return numerador / divisor

def dist_cos(x,y):
    return 1 - corr(x,y)

# Leer datos:                   
df = pd.read_csv("datos.csv")

# Matriz de tokens
def matriz_tokens():
    tokens = np.hstack(df["tokens"].apply(lambda x: x.split()).values)

    unique_tokens = pd.Series(tokens).value_counts().index[:1000].values
    unique_tokens_dict = dict(zip(unique_tokens, range(len(unique_tokens))))
    X = np.zeros((len(df), len(unique_tokens)), dtype=int)
    for i, row in df.iterrows():
        for token in row["tokens"].split():
            if unique_tokens_dict.get(token,False)!=False:
                X[i, unique_tokens_dict[token]] += 1
    
    return X

def knn(i, k, X):
    n = X.shape[0]
    C = (X @ X.T) / (n-1) # Lo hacemos al reves

    # print(C.shape[0])
    # print(df.shape[0])

    cercanos = np.argsort(C[i])[::-1]

    df["GenreID"] = df["Genre"].apply(lambda x: mapMovieIndex[x])

    ids = np.array(df["GenreID"].values)

    return stats.mode(ids).mode

print(mapIndexMovie[knn(15, 20, matriz_tokens())])

