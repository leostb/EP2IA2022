import numpy as np

from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error

from rede import prepara_dados, divide_treinamento_teste, Scaler


arquivo_entrada = "serie1_trein.txt"
L = 10

arquivo_inteiro = np.array(open(arquivo_entrada).read().splitlines(), dtype=float)
scaler = Scaler(arquivo_inteiro)
X, Yd = prepara_dados(arquivo_entrada, L, scaler)
x_train, x_teste, y_train, y_teste = divide_treinamento_teste(X, Yd, 2 / 3, shuffle=False)


model=Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, batch_size=100,epochs=10, validation_data=(x_teste, y_teste))
model.summary()

predictions = model.predict(x_teste)
#invertendo a escala
predictions = scaler.desnormalizar(predictions)
y_teste = scaler.desnormalizar(y_teste)
print('Erro médio absoluto :', mean_absolute_error(predictions,y_teste))
# print('Erro médio quadrado logarítmico :', mean_squared_log_error(predictions,y_teste))
print('Erro medio quadrado :', mean_squared_error(predictions,y_teste))
print('Coeficiente de determinação (R²):', r2_score(predictions,y_teste))

#plotando os dados
training_data_len= len(X)*2//3
train = X[:training_data_len]
valid = Yd[training_data_len:]

plt.figure(figsize=(20,5))
plt.plot(train)
plt.plot(valid)
plt.plot(predictions)
plt.show()

x_train=[]
y_train=[]

