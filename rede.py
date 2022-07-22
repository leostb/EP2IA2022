import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import shape

L = 10  # Número de entradas usadas para série temporal
ns = 1  # Número de saídas

h = 10  # Número de perceptrons na camada escondida


# funcao_de_ativação = f

# inicializar hidden layer


# normalizar

def prepara_dados(arquivo, lag):
    ''' A ideia seria montar as tuplas de entrada e label, mas não sei se realmente vai precisar'''
    series = open(arquivo).read().splitlines()
    X, y = [], []
    for i in range(len(series)):
        end_ix = i + lag
        if end_ix > len(series) - 1:
            break
        seq_x, seq_y = series[i:end_ix], series[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X, dtype=float), np.array(y, dtype=float)

def cria_grafico(series):
    i = 1
    line = []
    for value in series:
        line.append(value)  # Remove o \n que está no fim da linha do txt
    plt.plot(line)
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


def calcular_saida(A, B, X, N):
    Zin = np.matmul(X, A.T)
    Z = sigmoid(Zin)
    Z = np.concatenate((Z, np.ones((Z.shape[0], 1))), axis=1)
    Yin = np.matmul(Z, B.T)
    Y = sigmoid(Yin)
    return Y

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def calc_grad(X, Yd, A, B, N, ns):
    Zin = np.matmul(X, A.T)
    Z = sigmoid(-Zin)
    Z = np.concatenate((Z, np.ones((Z.shape[0], 1))), axis=1)

    Yin = np.matmul(Z, B.T)
    Y = sigmoid(-Yin)
    erro = Y - Yd

    gl = np.multiply(np.ones(Y.shape) - Y, Y)

    Znovo = Z[:, 0:Z.shape[1] - 1]
    ones = np.ones(Znovo.shape)
    fl = np.multiply(ones - Znovo, Znovo)

    dJdB = (1/N) * np.matmul(np.multiply(erro, gl).T, Z)

    dJdZ = np.matmul(np.multiply(erro, gl),  B[:, 0:B.shape[1] - 1])
    dJdA = (1/N) * np.matmul(np.multiply(dJdZ, fl).T, X)
    return dJdA, dJdB


def rna():
    X, Yd = prepara_dados("serie1_trein.txt", L)
    Yd = Yd.reshape((Yd.shape[0], 1))
    N, ne = shape(X)
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    A = np.random.rand(h, ne + 1)
    B = np.random.rand(ns, h + 1)
    Y = calcular_saida(A, B, X, N)

    erro = Y - Yd
    EQM = (erro**2).mean()

    nep = 0
    nepocasmax = 1000
    alfa = 0.95
    vet = []
    vet.append(EQM)

    while EQM > 1e-3 and nep < nepocasmax:
        nep = nep + 1
        dJdA, dJdB= calc_grad(X, Yd, A, B, N, ns)
       
        # alfa = calc_alfa(dJdA, dJdB, A, B, X, Yd, N, ne, ns)
        A = A - alfa * dJdA;
        B = B - alfa * dJdB;
    
        Y = calcular_saida(A, B, X, N);

        erro = Y - Yd
        EQM = (erro ** 2).mean()
        vet.append(EQM)
        print('\nNumero Epocas = %d alfa = %2.4f EQM = %2.4f' %(nep, alfa, EQM))
    plt.plot(vet)
    plt.show()

    return A, B


rna()
