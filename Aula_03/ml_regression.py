import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

#PASSO1 CARREGAR OS DADOS
df = pd.read_csv("FuelConsumptionCo2.csv")


#EXIBE A ESTRUTURA DO DATAFRAME
print(df.head())

#EXIBE O RESUMO DO DATAFRAME
print(df.describe())

#SELECIONAR SOMENTE ALGUMAS FEATURES
cdf = df[['ENGINESIZE','CO2EMISSIONS']]
print(cdf.head(9))

#EXIBE GRAFICO DE HISTOGRAMA DATAFRAME
cdf.hist()
plt.show()


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

#PASSO2 DIVIDIR OS DADOS EM DADOS DE TREINAMENTO E DADOS DE TESTE (TRAIND AND TEST DATASET)
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

#PASSO3 EXIBE A CORRELACAO ENTRE AS FEATURES DO DATASET DE TREINAMENTO
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

#PASSO4 TREINAR O MODELO
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

#EXIBIR OS COEFFICIENTS
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)
####LEMBRA A FORMULA: Y = INTERCEPT + COEFICIENT.X

#EXIBIR A FIT LINE 
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.ylabel("Emission")
plt.xlabel("Engine size")
plt.show()

#AVALIAÇÃO DO MODELO (ERRO QUADRATICO MEDIO / MSE)
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_pred = regr.predict(test_x)

print(regr.predict(3.7))

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_pred - test_y)))
print("Mean squared error: %.2f" % mean_squared_error(test_y, test_y_pred))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_pred - test_y) ** 2))
print ("RMSE: %.2f " % sqrt(mean_squared_error(test_y, test_y_pred)))
print("R2-score: %.2f" % r2_score(test_y_pred , test_y) )




