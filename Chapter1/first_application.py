from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import mglearn
import pandas as pd
iris_dataset = load_iris()

# print load iris
# print(iris_dataset)
# key of iris dataset
print("keys of iris dataset", iris_dataset.keys())
# short discription of datasets
print(iris_dataset["DESCR"][:192] + "\n...")

print("Value of target name:", (iris_dataset["target_names"]))
print("Feature name is ", (iris_dataset["feature_names"]))
print("types of data ", type(iris_dataset["data"]))
print("Shape of data ", (iris_dataset["data"].shape))
print("First five column of data ", (iris_dataset["data"][:5]))
print("type of target", (type(iris_dataset["target"])))
print("shape of target", (iris_dataset["target"].shape))
print("target", (iris_dataset["target"]))

x_train, x_test, y_train, y_test = train_test_split(
    iris_dataset["data"], iris_dataset["target"], random_state=0)
print("This is x_train shape", (x_train.shape))
print("This is y_train shape", (y_train.shape))
print("This is x_test shape", (x_test.shape))
print("This is y_test shape", (y_test.shape))
# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(x_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(
    15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
