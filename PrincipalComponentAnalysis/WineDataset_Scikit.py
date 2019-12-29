import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt 
#preamble
#############################################

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

#Train/Test split
X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,stratify=y)

#standardisation 

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

#The next step in PCA is creating the covariance matrix 
cov = np.cov(X_train_std.T)
eigenvals,eigenvecs = np.linalg.eig(cov)

#The end goal is to project onto a reduced subspace of the eigenvectors with 
#the highest eigenvalues 

#sort eigenvalues 

eigenpairs = [(eigenvals[i],eigenvecs[i]) for i in range(len(eigenvals))]
eigenpairs.sort(key=lambda k: k[0], reverse=True)

w = np.hstack((eigenpairs[0][1][:, np.newaxis], eigenpairs[1][1][:, np.newaxis]))

#w is now our transformation matrix. 

X_train_pca = X_train_std.dot(w)

#now lets visualise the projected data 

colours = ['r','g','b']
markers = ['s','x','o']

for l,c,m in zip(np.unique(y_train),colours,markers):
	plt.scatter(X_train_pca[y_train==l, 0], 
                X_train_pca[y_train==l, 1], 
                c=c, label=l, marker=m)	

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

#Note, in practise this can be very simply achieved by the sklearn pca package
#from sklearn.decomposition import PCA