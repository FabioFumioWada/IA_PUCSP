import pandas as pd
import numpy as np
from sklearn import tree, preprocessing

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

print (train.describe())
print (train["Pclass"].value_counts())
print (train["Survived"].value_counts())
print (train["Survived"][train["Sex"]== 'male'].value_counts())
print (train["Survived"][train["Sex"]== 'female'].value_counts())
train["Age"] = train["Age"].fillna(train["Age"].median())
print (train)
train["Age"] = train["Age"].fillna(train["Age"].median())

target = train["Survived"].values

encoded_sex = preprocessing.LabelEncoder()

train.Sex  = encoded_sex.fit_transform(train.Sex)

features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

my_tree_one = tree.DecisionTreeClassifier()
print (my_tree_one)
my_tree_one = my_tree_one.fit(features_one, target)
print (my_tree_one.score(features_one, target))



print(test)
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
test_econded_sex = preprocessing.LabelEncoder()
test.Sex = test_econded_sex.fit_transform(test.Sex)

test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

my_prediction = my_tree_one.predict(test_features)

#print (my_prediction)
#print (test)
passagers = np.array(test["PassengerId"]).astype(int)


my_solution = pd.DataFrame(my_prediction, passagers, columns = ["Survived"])
#print (my_solution)

tree.export_graphviz(my_tree_one, out_file="tree.dot")


