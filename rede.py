import random
import matplotlib.pyplot as plt


L = 10  # Número de entradas usadas para série temporal
ne = 5  # N
w_entrada = []
ns = 1  # Número de saídas

# inicializar pesos randomicamente

k = 10  # Número de perceptrons na camada escondida


# funcao_de_ativação = f

# inicializar hidden layer


# normalizar


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


def prepara_dados(arquivo, lag):
    ''' A ideia seria montar as tuplas de entrada e label, mas não sei se realmente vai precisar'''

    series = open(arquivo).read().splitlines()
    cria_grafico(series)
    X, y = [], []
    for i in range(len(series)):
        end_ix = i + lag
        if end_ix > len(series) - 1:
            break
        seq_x, seq_y = series[i:end_ix], series[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return X, y





X, y = prepara_dados("serie1_trein.txt", L)
train, validacao = divide_treinamento_validação(X)
