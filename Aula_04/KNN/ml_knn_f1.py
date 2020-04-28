import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report

#ABRE E CARREGA O ARQUIVO
df = pd.read_csv('teleCust1000t.csv')
print (df.head())

#MOSTRA QUANTOS ITENS EXISTEM POR CLASSES, NO CASO NO CAMPO CUSTCAT
print(df['custcat'].value_counts())

#MOSTRA O HISTOGRAMA DE UMA DAS COLUNAS
df.hist(column='age', bins=50)
plt.show()

#MOSTRA TODAS AS COLUNAS (CAMPOS) DO DATAFRAME
print(df.columns)

#CONVERTER DE DATAFRAME PARA UM ARRAY NUMPY PARA USAR NO SCIKITLEARN
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values
Y = df['custcat'].values

#PADRONIZA OS DADOS COM DIFERENTES MÉTRICAS/UNIDADES 
#(it will transform your data such that its distribution will have a mean value 0 and standard deviation of 1)
print (X[0:5])
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print (X[0:5])

#DIVIDINDO O DATASET NA ABORDAGEM TRAIN/TEST SPLIT
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_test.shape,  Y_test.shape)

#TREINAR O MODELO COM K=4
K = 4
neigh = KNeighborsClassifier(n_neighbors = K).fit(X_train, Y_train)
#MOSTRA INFORMACOES SOBRE O CLASSIFICADOR
print(neigh)

yhat = neigh.predict(X_test)
print(yhat[0:5])

#AVALIAÇÃO
targets_names = ['Basic', 'E-service', 'Plus Service', 'Total Service']
print (classification_report(Y_test, yhat, target_names=targets_names))
#print("Test set Accuracy: ", metrics.f1_score(Y_test, yhat, average=None))



#TESTAR DIFERENTES VALORES PARA K
Ks = 50
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,Y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(Y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==Y_test)/np.sqrt(yhat.shape[0])

mean_acc

#GRAFICO PARA OS DIFERENTES VALORES DE K
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 






