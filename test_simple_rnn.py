import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.metrics import mean_squared_error

from rede import prepara_dados, divide_treinamento_teste, Scaler
from simple_rnn import train, hidden_dim, seq_len, sigmoid, output_dim

arquivo_entrada = "serie4_trein.txt"
arquivo_inteiro = np.array(open(arquivo_entrada).read().splitlines(), dtype=float)
scaler = Scaler(arquivo_inteiro)

Xtodo, Ytodo = prepara_dados(arquivo_entrada, seq_len, scaler)
X, Xvalida, Y, Yvalida = divide_treinamento_teste(Xtodo, Ytodo, 2 / 3, shuffle=False)

X = np.expand_dims(np.array(X), axis=2) # 100 x 50 x 1
Y = np.expand_dims(np.array(Y), axis=1) # 100 x 1

Xvalida = np.expand_dims(np.array(Xvalida), axis=2)
Yvalida = np.expand_dims(np.array(Yvalida), axis=1)


np.random.seed(12161)
U = np.random.uniform(0, 1, (hidden_dim, seq_len)) # weights from input to hidden layer
V = np.random.uniform(0, 1, (output_dim, hidden_dim)) # weights from hidden to output layer
W = np.random.uniform(0, 1, (hidden_dim, hidden_dim)) # recurrent weights for layer (RNN weigts)

U, V, W = train(U, V, W, X, Y, Xvalida, Yvalida)


predictions = []
for i in range(Y.shape[0]):
    x, y = X[i], Y[i]
    prev_activation = np.zeros((hidden_dim,1))
    # forward pass
    for timestep in range(seq_len):
        mulu = np.dot(U, x)
        mulw = np.dot(W, prev_activation)
        _sum = mulu + mulw
        activation = sigmoid(_sum)
        mulv = np.dot(V, activation)
        prev_activation = activation
    predictions.append(mulv)

predictions = scaler.desnormalizar(np.array(predictions))

plt.plot(predictions[:, 0,0], 'g')
plt.plot(Y[:, 0], 'r')
plt.title("Predições dos dados de Treinamento em Verde, Dados reais em vermelho")
plt.show()

val_predictions = []
for i in range(Yvalida.shape[0]):
    x, y = X[i], Y[i]
    prev_activation = np.zeros((hidden_dim,1))
    for timestep in range(seq_len):
        mulu = np.dot(U, x)
        mulw = np.dot(W, prev_activation)
        _sum = mulu + mulw
        activation = sigmoid(_sum)
        mulv = np.dot(V, activation)
        prev_activation = activation
    val_predictions.append(mulv)

val_predictions = scaler.desnormalizar(np.array(val_predictions))

plt.plot(val_predictions[:, 0,0], 'g')
plt.plot(Yvalida[:, 0], 'r')
plt.title("Predições dos dados de Teste em Verde, Dados reais em vermelho")
plt.show()

rmse = math.sqrt(mean_squared_error(Yvalida[:,0], val_predictions[:, 0, 0]))
print("EQM " + str(rmse))