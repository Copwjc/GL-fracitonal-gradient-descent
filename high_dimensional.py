import numpy as np
from math import *
import matplotlib.pyplot as plt
import random
random.seed(1)



dim = 10
ordered = 0.9
k = 10
cons = (-1)**ceil(ordered)

def coefficient(k, ordered):
    t = 1
    for i in range(0, k):
        t *= (ordered - i)
    t *= (-1)**k
    t = t / gamma(k + 1)
    return t

coef = []

for i in range(0, k):
    coef.append(coefficient(i, ordered))

def ncasual(coef, x, f):
    global k,dim
    out = []
    for i in range(0,dim):
        d = np.zeros(dim)
        print(d)
        p1, p2 = [], []
        for j in range(0, k):
            d[i] = j
            temp1 = f(x - d)
            temp2 = f(x + d)
            p1.append(temp1)
            p2.append(temp2)
        tempg = sum(np.array(coef) * (np.array(p1) - np.array(p2)))
        out.append(tempg)
    return out


# def ncgd(x0, y0, f, learning_rate, num_iters):
#     global k, coef
#     x = x0
#     y = y0
#     nc = [(x0, y0)]
#     for i in range(0, num_iters):
#         grad = ncasual(coef, x, f)
#         x -= learning_rate * grad
#         nc.append(f(x))
#     return nc

def f(x):
    return sum((x - 5)**2)  # 10维的unimodal函数


x0 = [10*np.random.rand(dim)]
f0 = [f(x0[0])]
num_iters = 100
lr = 0.01

for i in range(num_iters):
    print(ncasual(coef, x0[i], f))
    tempx = x0[i] - lr * np.array(ncasual(coef, x0[i], f))
    x0.append(tempx)
    f0.append(f(tempx))


plt.plot(f0)
plt.show()
