# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.linalg as la
from sklearn.decomposition import PCA

'''
Data
'''
data = np.matrix([[1,2,4],
               [4,1,2],
               [5,4,8]])

df = pd.DataFrame(data)

'''
Manual calculation
'''

# standardize data 
standardized_data  = (df - df.mean()) / (df.std())

# Finding covariance
covarance = np.cov(standardized_data.T, bias = 1)

# find eigen value& eigen vector
eigenvalue, eigenvectors = np.linalg.eig(covarance)

# Find PCA
n_components = 3

pca_manual = np.matmul(np.array(standardized_data),eigenvectors)

pca_manual  = pca_manual[:,:n_components]

'''
calculate using SKlearn
'''

# PCA
pca_sklearn = (PCA(n_components).fit_transform(standardized_data))

print('Standardized data')
print(standardized_data.round(2))
print('')

print('Covariance')
print(covarance.round(2))
print('')

print('eigen_value')
print(eigenvalue.round(4))
print('')


print('eigen_vector')
print(eigenvectors.round(4))
print('')

print('PCA manually calculated')
print(pca_manual.round(2))
print('')

print('PCA - sklearn')
print(pca_sklearn.round(2))