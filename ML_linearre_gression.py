# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 20:17:57 2022

@author: HUAWEI
"""
#load Dependencies
import matplotlib.pyplot as plt
import math
import random
#load dataset

x1=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(0,18):
    x1[i]=random.uniform(15,20)
#    print(x1[i])
x2=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(0,18):
    x2[i]=random.randint(1,3)
#    print(x2[i])
y=[14.8,17.5,16.9,14.5,14.6,14.4,15.0,20.1,18.8,15.7,14.9,13.5,13.6,15.4,14.0,11.8,13.0,12.7]
#training set
x1_train=x1[0:9]
x2_train=x2[0:9]
y_train=y[0:9]
n_train=len(x1_train)
#testing set
x1_test=x1[9:]
x2_test=x2[9:]
y_test=y[9:]
n_test=len(x1_test)
#set parameter
w1=-0.1
w2=0.3
b=3
#learning rate
lr=0.00001
#training cycle
epoches=1000
#y=
for i in range(epoches):
    sum_w1=0.0
    sum_w2=0.0
    sum_b=0.0
    for i in range(n_train):
        y_hat=w1*x1_train[i]+w2*x2_train[i]+b
        sum_w1+=(y_train[i]-y_hat)*(-x1_train[i])
        sum_w2+=(y_train[i]-y_hat)*(-x2_train[i])
        sum_b+=(y_train[i]-y_hat)*(-1)
    decent_w1=2*sum_w1#梯度
    decent_w2=2*sum_w2
    det_b=2*sum_b
    w1=w1-lr*decent_w1
    w2=w2-lr*decent_w2
    b=b-lr*det_b
#show image
fig,ax=plt.subplots()
ax.scatter(x1_train,y_train)
ax.plot([i for i in range(10,25)],[w1*i+w2*i+b for i in range(10,25)])
plt.title('y=w1*x+w2*x+b')
plt.show()
#training loss
total_train_loss=0
for i in range(n_train):
    y_hat=w1*x1_train[i]+w2*x2_train[i]+b
    total_train_loss+=(y_train[i]-y_hat)**2
#test loss
total_test_loss=0
for i in range(n_test):
    y_hat=w1*x1_test[i]+w2*x2_test[i]+b
    total_test_loss+=(y_test[i]-y_hat)**2
print("total_train_loss:")
print(total_train_loss)
print("total_test_loss:")
print(total_test_loss)

#%%


