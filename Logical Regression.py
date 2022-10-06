# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 22:08:34 2022

@author: HUAWEI
"""
#%%
#CODE
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
path = 'C://Users//HUAWEI//Desktop//predict.txt'
data = pd.read_csv(path,header=None,names=['exam1','exam2','ill'])
Positi = data[data['ill'].isin([1])]
Negati = data[data['ill'].isin([0])]
def sigmoid(z):  # sigmoid function
    return 1/(1+np.exp(-z))

def Cost(theta,X,y):    #cost function
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

data.insert(0, 'Ones', 1)# h (x)= θ0+ θ1𝓧1+ θ2𝓧2 的一个常数项可以看成是θ0和1的乘积，其他项为θi和𝓧i的乘积

#initialization，x,y,theta
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)
#gradient function
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)
    return grad

import scipy.optimize as opt
result = opt.fmin_tnc(func=Cost, x0=theta, fprime=gradient,args=(X,y))
#结果可视化
plotting_x1 = np.linspace(30, 100, 100)
plotting_h1 = ( - result[0][0] - result[0][1] * plotting_x1) / result[0][2]

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(plotting_x1, plotting_h1, 'p', label='predict ')
ax.scatter(Positi['exam1'], Positi['exam2'], s=50, c='m', marker='o', label='ill')
ax.scatter(Negati['exam1'], Negati['exam2'], s=50, c='g', marker='s', label='health')
ax.legend()
ax.set_xlabel('examdata1')
ax.set_ylabel('examdata2')
plt.show()













