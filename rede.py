import math
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import shape

L = 10  # Número de entradas usadas para série temporal
ns = 1  # Número de saídas
h = 5  # Número de perceptrons na camada escondida
nepocasmax = 100
arquivo_entrada = "serie4_trein.txt"


def prepara_dados(arquivo, lag, scaler=None):
    series = np.array(open(arquivo).read().splitlines(), dtype=float)
    cria_grafico(series, "Serie Original")

    if scaler is None:
        scaler = Scaler(series)

    series = scaler.normalizar(series)
    cria_grafico(series, "Serie normalizada")
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
    line = []
    for value in series:
        line.append(value)
    plt.plot(line)
    plt.title(label=title)
    plt.show()


def divide_treinamento_teste(dados, labels, partition, shuffle=True):
    """ (list) -> (list), (list)
        Recebe uma ndarray e retorna os dados divididos em treinamento e teste
    """
    tamanho_orig = len(dados)
    permuta_x = dados
    permuta_y = labels
    if shuffle:
        permuta_x, permuta_y = unison_shuffled_copies(dados, labels)

    corte = math.floor(tamanho_orig * partition)
    treinamento_x, teste_x = permuta_x[0:corte], permuta_x[corte:]
    treinamento_y, teste_y = permuta_y[0:corte], permuta_y[corte:]

    return treinamento_x, teste_x, treinamento_y, teste_y


def calcular_saida(A, B, X):
    Zin = np.matmul(X, A.T)
    Z = sigmoid(Zin)
    Z = np.concatenate((Z, np.ones((Z.shape[0], 1))), axis=1)
    Yin = np.matmul(Z, B.T)
    Y = sigmoid(Yin)
    return Y


def calc_grad(X, Yd, A, B, N, ns):
    Zin = np.matmul(X, A.T)
    Z = sigmoid(Zin)
    Z = np.concatenate((Z, np.ones((Z.shape[0], 1))), axis=1)

    Yin = np.matmul(Z, B.T)
    Y = sigmoid(Yin)

    erro = calcular_erro(Y, Yd)

    gl = np.multiply(1 - Y, Y)

    Znovo = Z[:, 0:Z.shape[1] - 1]

    fl = np.multiply(1 - Znovo, Znovo)

    dJdB = (1 / N) * np.matmul(np.multiply(erro, gl).T, Z)

    dJdZ = np.matmul(np.multiply(erro, gl), B[:, 0:B.shape[1] - 1])
    dJdA = (1 / N) * np.matmul(np.multiply(dJdZ, fl).T, X)
    return dJdA, dJdB


def rna(X, Yd):
    Yd = Yd.reshape((Yd.shape[0], 1))
    N, ne = shape(X)
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    A = np.random.rand(h, ne + 1)
    B = np.random.rand(ns, h + 1)
    Y = calcular_saida(A, B, X)

    erro = calcular_erro(Y, Yd)
    EQM = (erro ** 2).mean()

    nep = 0
    alfa = 0.95
    vet = [EQM]

    while EQM > 1e-3 and nep < nepocasmax:
        nep = nep + 1
        dJdA, dJdB = calc_grad(X, Yd, A, B, N, ns)

        alfa = calc_alfa(dJdA, dJdB, A, B, X, Yd, N, ne, ns)
        A = A - alfa * dJdA
        B = B - alfa * dJdB

        Y = calcular_saida(A, B, X)

        erro = calcular_erro(Y, Yd)
        EQM = (erro ** 2).mean()
        vet.append(EQM)
        print('\nNumero Epocas = %d alfa = %2.4f EQM = %2.8f' % (nep, alfa, EQM))
    cria_grafico(vet, "Erro quadrático Médio ao longo das épocas")

    return A, B


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


def avaliar(A, B, x_teste, y_teste):
    x_teste = np.concatenate((x_teste, np.ones((x_teste.shape[0], 1))), axis=1)
    Y = calcular_saida(A, B, x_teste)

    erro = calcular_erro(Y, y_teste)
    EQM = (erro ** 2).mean()
    print("Erro no conjunto de teste %.6f" % (EQM))
    return EQM


def simples():
    X, Yd = prepara_dados(arquivo_entrada, L)
    X, x_teste, Yd, y_teste = divide_treinamento_teste(X, Yd, 2 / 3)
    A, B = rna(X, Yd)
    avaliar(A, B, x_teste, y_teste)


def k_fold(k):
    X_inteiro, Y_inteiro = prepara_dados(arquivo_entrada, L)
    permuta_x, permuta_y = unison_shuffled_copies(X_inteiro, Y_inteiro)

    vet_eqm = []

    for i in range(len(permuta_x)):
        A, B = rna(permuta_x[i], permuta_y[i])
        EQM = avaliar(A, B, complementar(permuta_x, permuta_x[i]), complementar(permuta_x, permuta_x[i]))
        vet_eqm.append(EQM)


# Funções auxiliares


def particionar(dados, labels, folds):
    tamanho_orig = len(dados)
    permuta_x, permuta_y = unison_shuffled_copies(dados, labels)
    return [np.array_split(permuta_x, folds)], [np.array_split(permuta_y, folds)]


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def calcular_erro(x, y):
    if x.shape[0] == y.shape[0] and x.shape[1] == 1:
        y = y.reshape((y.shape[0], 1))

    assert (x.shape == y.shape)
    return x - y


class Scaler:

    def __init__(self, M):
        self.min = np.min(M)
        self.max = np.max(M)
        self.dif = self.max - self.min

    def normalizar(self, M):
        z = (M - self.min) / self.dif
        return z

    def desnormalizar(self, M):
        return M * self.dif + self.min


if __name__ == "__main__":
    simples()
