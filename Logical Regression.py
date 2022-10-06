# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 22:08:34 2022

@author: HUAWEI
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']#中文
path='C://Users//HUAWEI//Desktop//predict.txt'
data=pd.read_csv(path,header=None,names=['Check1','Check2','ill'])

positive=data[data['ill'].isin([1])]
negative=data[data['ill'].isin([0])]
fig,ax=plt.subplots(figsize=(12,8))
ax.scatter(positive['Check1'],positive['Check2'],s=50,c='r',maker='o',label='ill')
ax.scatter(negative['Check1'],negative['Check2'],s=50,c='g',maker='s',label='health')
ax.legend()
ax.set_xlabel('check1_data')
ax.set_ylabel('check2_data')
plt.show()



#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#plt.rcParams['font.sans-serif'] = ['SimHei']#显示中文

path = 'C://Users//HUAWEI//Desktop//predict.txt'
data = pd.read_csv(path,header=None,names=['exam1','exam2','ill'])

Positi = data[data['ill'].isin([1])]
Negati = data[data['ill'].isin([0])]
'''
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(阳性['体检1'], 阳性['体检2'], s=50, c='r', marker='o', label='患病')
ax.scatter(阴性['体检1'], 阴性['体检2'], s=50, c='g', marker='s', label='不患病')
ax.legend()
ax.set_xlabel('体检 1 数据')
ax.set_ylabel('体检 2 数据')
plt.show()
'''
# 实现sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))


#实现代价函数
def Cost(theta,X,y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

# 加一列常数列
data.insert(0, 'Ones', 1)

# 初始化X，y，θ
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)
#实现梯度函数
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


# 用θ的计算结果代回代价函数计算
print(result[0])
print(Cost( result[0],X, y))


plotting_x1 = np.linspace(30, 100, 100)
plotting_h1 = ( - result[0][0] - result[0][1] * plotting_x1) / result[0][2]

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(plotting_x1, plotting_h1, 'p', label='predict ')


ax.scatter(Positi['exam1'], Positi['exam2'], s=50, c='m', marker='o', label='ill')
ax.scatter(Negati['exam1'], Negati['exam2'], s=50, c='b', marker='s', label='health')


ax.legend()
ax.set_xlabel('examdata1')
ax.set_ylabel('examdata2')
plt.show()

#%%
def hfunc1(theta, X):
    return sigmoid(np.dot(theta.T, X))
def predict(theta, X):
    probability = sigmoid(np.dot(theta.T, X))
    return [1 if probability >= 0.5 else 0]
print('得病率为：',hfunc1(result[0],[1,60,70]))
print('预测得不得病：',predict(result[0],[1,60,70]))
#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['simHei']#Chinese
path='C://Users//HUAWEI//Desktop//predict.txt'
data=pd.read_csv(path,header=None,names=['体检1','体检2','患病'])
阳性=data[data['患病'].isin([1])]
阴性=data[data['患病'].isin([0])]

#sigmoid function
def sigmoid(a):
    return 1/(1+np.exp(-a))
#cost function
def Cost(theta,x,y):
    theta=np.matrix(theta)
    x=np.matrix(x)
    y=np.matrix(y)
    first=np.multiply(-y,np.log(sigmoid(x*theta.T)))
    second=np.multiply((1-y),np.log(1-sigmoid(x*theta.T)))#theta.T:转置
    return np.sum(first-second)/(len(x))
data.insert(0,'Ones',1)# h (x)= θ0+ θ1𝓧1+ θ2𝓧2 的一个常数项可以看成是θ0和1的乘积，其他项为θi和𝓧i的乘积

cols=data.shape[1]#initialization，x,y,theta
x=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]

x=np.array(x.values)
y=np.array(y.values)
theta=np.zeros(3)
#GRADIENT FUNCTION
def gradient(theta,x,y):
    theta=np.matrix(theta)
    x=np.matrix(x)
    y=np.matrix(y)
    parameters=int(theta.ravel().shape[1])
    grad=np.zeros(parameters)
    error=sigmoid(x*theta.T)-y
    
    for i in range(parameters):
        term=np.multiply(error,x[:,i])
        grad[i]=np.sum(term)/len(x)
    return grad

import scipy.optimize as opt
result=opt.fmin_tnc(func=Cost,x0=theta,fprime=gradient,args=(x,y))

plotting_x1=np.linspace(30,100,100)
plotting_h1=(- result[0][0]-result[0][1]*plotting_x1)/result[0][2]

fig,ax=plt.subplots(figsize=(12,8))
ax.plot(plotting_x1,plotting_h1,'y',label='预测')

ax.scatter(阳性['体检1'],阳性['体检2'],s=50,c='y',maker='o',label='患病')
ax.scatter(阴性['体检1'],阴性['体检2'],s=50,c='b',maker='s',label='不患病')
      
ax.legend()
ax.set_xlabel('体检 1 数据')
ax.set_ylabel('体检 2 数据')
plt.show()   
#%%
#jiaozuoyede CODE
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













