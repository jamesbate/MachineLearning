import sklearn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import neighbors, datasets
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = datasets.load_iris(return_X_y=True)

plt.scatter(X[:50,0],X[:50,1],color='blue',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],color='red',label='versicolor')
plt.scatter(X[100:150,0],X[100:150,1],color='black',label='virginica')
plt.legend()
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

plt.scatter(X[:50,2],X[:50,3],color='blue',label='setosa')
plt.scatter(X[50:100,2],X[50:100,3],color='red',label='versicolor')
plt.scatter(X[100:150,2],X[100:150,3],color='black',label='virginica')
plt.legend()
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state = 42)

scaler = StandardScaler()
scaler.fit(X_train) # fit only the training data
#calculutes mean and std of data

X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

k = 3
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train_std, y_train)

pred = knn.predict(X_test_std)
print(accuracy_score(y_test,pred))

k_scores = []
for k in range(1,51):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv = 10, scoring='accuracy')
    k_scores.append(scores.mean())
plt.plot(np.arange(1,51), k_scores)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()
