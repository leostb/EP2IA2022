import math
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import shape

L = 10  # Número de entradas usadas para série temporal
ns = 1  # Número de saídas
h = 10  # Número de perceptrons na camada escondida


def prepara_dados(arquivo, lag):
    ''' A ideia seria montar as tuplas de entrada e label, mas não sei se realmente vai precisar'''
    series = np.array(open(arquivo).read().splitlines(), dtype=float)
    #cria_grafico(series, "Serie Original")
    series = normalizar(series)
    #cria_grafico(series, "Serie normalizada")
    X, y = [], []
    for i in range(len(series)):
        end_ix = i + lag
        if end_ix > len(series) - 1:
            break
        seq_x, seq_y = series[i:end_ix], series[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X, dtype=float), np.array(y, dtype=float)


def cria_grafico(series, title=""):
    i = 1
    line = []
    for value in series:
        line.append(value)
    plt.plot(line)
    plt.title(label=title)
    plt.show()


def divide_treinamento_validação(dados):
    """ (list) -> (list), (list)
        Recebe uma lista e retorna os dados divididos em treinamento e teste
    """

    tamanho_orig = tamanho = len(dados)
    treinamento = []
    while tamanho > 2 * tamanho_orig / 3:  # Condição de parada, 1/3 para treinamento
        indice = random.randint(0, tamanho - 1)  # Sorteia um índice para compor o treinamento, sem reposição
        treinamento.append(dados.pop(indice))  # Retira da lista original
        tamanho = len(dados)
    return treinamento, dados


@np.vectorize
def relu(x):
    if x <= 0:
        return 0
    else:
        return x


def calcular_saida(A, B, X, N):
    Zin = np.matmul(X, A.T)
    Z = sigmoid(Zin)
    Z = np.concatenate((Z, np.ones((Z.shape[0], 1))), axis=1)
    Yin = np.matmul(Z, B.T)
    Y = sigmoid(Yin)
    return Y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def calc_grad(X, Yd, A, B, N, ns):
    Zin = np.matmul(X, A.T)
    Z = sigmoid(Zin)
    Z = np.concatenate((Z, np.ones((Z.shape[0], 1))), axis=1)

    Yin = np.matmul(Z, B.T)
    Y = sigmoid(Yin)
    erro = Y - Yd

    gl = np.multiply(1 - Y, Y)

    Znovo = Z[:, 0:Z.shape[1] - 1]

    fl = np.multiply(1 - Znovo, Znovo)

    dJdB = (1 / N) * np.matmul(np.multiply(erro, gl).T, Z)

    dJdZ = np.matmul(np.multiply(erro, gl), B[:, 0:B.shape[1] - 1])
    dJdA = (1 / N) * np.matmul(np.multiply(dJdZ, fl).T, X)
    return dJdA, dJdB


def rna():
    X, Yd = prepara_dados("serie1_trein.txt", L)
    # np.savetxt("x.csv", np.around(X,4), delimiter=";", fmt='%f')
    # np.savetxt("yd.csv", Yd,delimiter=";")
    Yd = Yd.reshape((Yd.shape[0], 1))
    N, ne = shape(X)
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    A = np.random.rand(h, ne + 1)
    B = np.random.rand(ns, h + 1)
    Y = calcular_saida(A, B, X, N)

    erro = Y - Yd
    EQM = (erro ** 2).mean()

    nep = 0
    nepocasmax = 10000
    alfa = 0.95
    vet = []
    vet.append(EQM)

    while EQM > 1e-3 and nep < nepocasmax:
        nep = nep + 1
        dJdA, dJdB = calc_grad(X, Yd, A, B, N, ns)

        alfa = calc_alfa(dJdA, dJdB, A, B, X, Yd, N, ne, ns)
        A = A - alfa * dJdA
        B = B - alfa * dJdB

        Y = calcular_saida(A, B, X, N)

        erro = Y - Yd
        EQM = (erro ** 2).mean()
        vet.append(EQM)
        print('\nNumero Epocas = %d alfa = %2.4f EQM = %2.4f' % (nep, alfa, EQM))
    plt.plot(vet)
    plt.show()

    return A, B


def normalizar(M):
    dif = np.max(M) - np.min(M)
    z = (M - np.min(M)) / dif
    return z


def calc_alfa(dJdA, dJdB, A, B, X, Yd, N, ne, ns):
    dv = -np.concatenate([dJdA.flatten(), dJdB.flatten()])

    alfa_u = random.random()

    An = A - alfa_u * dJdA
    Bn = B - alfa_u * dJdB

    dJdAn, dJdBn = calc_grad(X, Yd, An, Bn, N, ns)

    g = np.concatenate([dJdAn.flatten(), dJdBn.flatten()])

    hl = np.matmul(g.T, dv)

    alfa_l = 0
    while hl < 0:
        alfa_l = alfa_u
        alfa_u = 2 * alfa_u
        An = A - alfa_u * dJdA
        Bn = B - alfa_u * dJdB
        dJdAn, dJdBn = calc_grad(X, Yd, An, Bn, N, ns)
        g = np.concatenate([dJdAn.flatten(), dJdBn.flatten()])
        hl = np.matmul(g.T, dv)

    epsilon = 1e-5
    kmax = math.ceil(math.log2((alfa_u - alfa_l) / epsilon))
    it = 0
    itmax = 20
    alfa_m = (alfa_l + alfa_u) / 2

    while it < kmax & it < itmax & abs(hl) > 1e-5:
        it = it + 1
        An = A - alfa_m * dJdA
        Bn = B - alfa_m * dJdB
        dJdAn, dJdBn = calc_grad(X, Yd, An, Bn, N, ns)
        g = np.concatenate([dJdAn.flatten(), dJdBn.flatten()])
        hl = np.matmul(g.T, dv)
        if hl > 0:
            alfa_u = alfa_m
        elif hl < 0:
            alfa_l = alfa_m
        else:
            break

        alfa_m = (alfa_l + alfa_u) / 2
    return alfa_m


rna()
