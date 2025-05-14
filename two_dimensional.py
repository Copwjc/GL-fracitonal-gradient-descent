import numpy as np
from math import *
import matplotlib.pyplot as plt
import random
random.seed(2)
ordered = 0.8
k = 10
cons = (-1)**ceil(ordered)

def coefficient(k,ordered):
  t = 1
  for i in range(0,k):
    t *= (ordered-i)
  t *= (-1)**k
  t = t/gamma(k+1)
  return t
coef = []

for i in range(0,k):
  coef.append(coefficient(i,ordered))

def ncasual(coef,x,f):
    global k
    p4 = []
    p5 = []
    for i in range(0,k):
        i = i
        temp1 = f(x+i)
        temp2 = f(x-i)
        p4.append(temp1)
        p5.append(temp2)
    out = (sum(np.array(coef)*(np.array(p5)-np.array(p4))))
    return out

def ncgd(x0,f,learning_rate,num_iters):
    global k,coef
    x = x0
    nc = [x0]
    for i in range(0,num_iters):
        alpha = ncasual(coef,x,f)
        x -= learning_rate * alpha
        nc.append(x)
    return x,nc


def gradient_descent(f, df, x0, learning_rate, num_iters):
    x = x0
    gd = [x0]
    for i in range(num_iters):
        x -= learning_rate * df(x)
        gd.append(x)
    return x,gd

def nadam(f, df, x0, learning_rate, num_iters, beta1=0.9, beta2=0.999, epsilon=1e-8):
    x = x0
    m = 0 
    v = 0 
    nada = [x0]
    for i in range(1, num_iters + 1):
        g = df(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**i)
        v_hat = v / (1 - beta2**i)
        x -= learning_rate * (beta1 * m_hat + (1 - beta1) * g / (1 - beta1**i)) / (np.sqrt(v_hat) + epsilon)
        nada.append(x)
    return x,nada


def stochastic_gradient_descent(f, df, x0, learning_rate, num_iters):
    x = x0
    sgd = [x0]
    for i in range(num_iters):
        i = np.random.randint(0, len(x))
        x[i] -= learning_rate * df(x[i])
        sgd.append(x)
    return x,sgd

def minibatch_gradient_descent(f, df, x0, learning_rate, num_iters, batch_size):
    x = x0
    mgd = [x0]
    for i in range(num_iters):
        i = np.random.randint(0, len(x) - batch_size + 1) 
        x[i:i+batch_size] -= learning_rate * df(x[i:i+batch_size])
        mgd.append(x)
    return x,mgd

def nag(f, df, x0, learning_rate, num_iters, momentum=0.9):
    x = x0
    v = 0
    nag = [x0]
    for i in range(num_iters):
        g = df(x - momentum * v)
        v = momentum * v + learning_rate * g
        x -= v
        nag.append(x)
    return x,nag

def adagrad(df, x0, learning_rate, num_iters, epsilon=1e-8):
    x = x0
    grad_squared = 0
    ada = [x0]
    for i in range(num_iters):
        grad = df(x)
        grad_squared += grad ** 2
        x -= learning_rate * grad / (np.sqrt(grad_squared) + epsilon)
        ada.append(x)
    return x,ada

def rmsprop(df, x0, learning_rate, num_iters, beta=0.9, epsilon=1e-8):
    x = x0
    grad_squared = 0
    rms = [x0]
    for i in range(num_iters):
        grad = df(x)
        grad_squared = beta * grad_squared + (1 - beta) * grad ** 2
        x -= learning_rate * grad / (np.sqrt(grad_squared) + epsilon)
        rms.append(x)
    return x,rms

def adamax(df, x0, learning_rate, num_iters, beta1=0.9, beta2=0.999, epsilon=1e-8):
    x = x0
    m = 0
    v = 0
    adamax = [x0]
    for i in range(1, num_iters + 1):
        grad = df(x)
        m = beta1 * m + (1 - beta1) * grad
        v = max(beta2 * v, abs(grad))
        m_hat = m / (1 - beta1 ** i)
        x -= learning_rate * m_hat / (v + epsilon)
        adamax.append(x)
    return x,adamax

def adadelta(df, x0, num_iters, gamma=0.9, epsilon=1e-5):
    x = x0
    acc_grad = 0
    acc_update = 0
    adad = [x0]
    for i in range(num_iters):
        grad = df(x)
        acc_grad = gamma * acc_grad + (1 - gamma) * grad ** 2
        update = np.sqrt(acc_update + epsilon) / np.sqrt(acc_grad + epsilon) * grad
        acc_update = gamma * acc_update + (1 - gamma) * update ** 2
        x -= update
        adad.append(x)
    return x,adad

def adamw(df, x0, learning_rate, num_iters, weight_decay=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    x = x0
    m = 0
    v = 0
    adaw = [x0]
    for i in range(1, num_iters + 1):
        grad = df(x)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** i)
        v_hat = v / (1 - beta2 ** i)
        x -= learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * x)
        adaw.append(x)
    return x,adaw

f = lambda x: 2*x**2-1.05*x**4+1/6*x**6
df = lambda x: 4*x - 4.2*x**3 + 1.2*x**5

x0 = 5
learning_rate = 0.001
num_iters = 1000


marker = ['o','v','^','<','>','1','2','3','4','8','s','p','*','h','H','+','x','D','d','|','_']
ls = ['-','--','-.',':']
plt.figure(figsize=(12, 8))

nc, ncgd1 = ncgd(x0, f, learning_rate, num_iters)
gd, gdgd = gradient_descent(f, df, x0, learning_rate, num_iters)
nd, ndgd = nadam(f, df, x0, learning_rate, num_iters)
# sg, sggd = stochastic_gradient_descent(f, df, x0, learning_rate, num_iters)
# mg, mggd = minibatch_gradient_descent(f, df, x0, learning_rate, num_iters, batch_size=10)
ng, nggd = nag(f, df, x0, learning_rate, num_iters)
ag, aggd = adagrad(df, x0, learning_rate, num_iters)
rg, rms = rmsprop(df, x0, learning_rate, num_iters)
am, adamax1 = adamax(df, x0, learning_rate, num_iters)
ad, adad = adadelta(df, x0, num_iters)
aw, adamw1 = adamw(df, x0, learning_rate, num_iters)

plt.plot(gdgd, label='gd', marker = random.choice(marker), linestyle = random.choice(ls),markevery = 100)
plt.plot(ndgd, label='nadam', marker = random.choice(marker), linestyle = random.choice(ls),markevery = 100)
plt.plot(nggd, label='nag', marker = random.choice(marker), linestyle = random.choice(ls),markevery = 100)
plt.plot(aggd, label='adagrad', marker = random.choice(marker), linestyle = random.choice(ls),markevery = 100)
plt.plot(rms, label='rmsprop', marker = random.choice(marker), linestyle = random.choice(ls),markevery = 100)
plt.plot(adamax1, label='adamax', marker = random.choice(marker), linestyle = random.choice(ls),markevery = 100)
plt.plot(adad, label='adadelta', marker = random.choice(marker), linestyle = random.choice(ls),markevery = 100)
plt.plot(adamw1, label='adamw', marker = random.choice(marker), linestyle = random.choice(ls),markevery = 100)
plt.plot(ncgd1, label='ncgd', marker = random.choice(marker), linestyle = random.choice(ls),markevery = 100)
plt.legend(loc = 'lower right')
plt.show()

