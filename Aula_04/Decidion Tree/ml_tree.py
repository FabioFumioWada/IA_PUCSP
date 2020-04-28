import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree


#CARREGAR OS DADOS PARA O DATAFRAME
my_data = pd.read_csv("drug200.csv", delimiter=",")
print(my_data[0:5])

#CONVERTE DE DATAFRAME PARA ARRAY

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])


#ARVORE DE DECISÃO NAO ACEITA VLAORES CATEGÓRICOS. SÓ NUMERICO
#ESSE TRECHO DE ODICO CONVERTE DADOS COMO O SEXO E BL PARA VALORES NUMERICOS

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 
print(X[0:5])

#CRIA O ARRAY Y PARA COLOCAR OS LABELS 
y = my_data["Drug"]
print(y[0:5])

#DIVIDIR O DATASET NOS DATASET DE TREINO E TESTE
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)


#TREINANDO A ARVORE
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
print(drugTree)
drugTree.fit(X_trainset, y_trainset)

#MOSTRAR AS FEATURES COM MAIS IMPORTANCIA
print(drugTree.feature_importances_)

#FAZER PREDIÇÕES COM A ARVORE
predTree = drugTree.predict(X_testset)

print(predTree[0:5])
print(y_testset[0:5])

#AVALIAÇÃO DO MODELO
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

#VISUALIZAR A ARVORE
dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


