import numpy as np
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

ordered = 0.5
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

def ncasual(coef,x,y,f):
    global k
    p4 = []
    p5 = []
    p6 = []
    p7 = []
    for i in range(0,k):
        i = i
        temp1 = f(x+i,y)
        temp2 = f(x-i,y)
        p4.append(temp1)
        p5.append(temp2)
        temp3 = f(x,y+i)
        temp4 = f(x,y-i)
        p6.append(temp3)
        p7.append(temp4)
    outx = (sum(np.array(coef)*(np.array(p5)-np.array(p4))))
    outy = (sum(np.array(coef)*(np.array(p7)-np.array(p6))))
    return outx,outy

def ncgd(x0, y0, f, learning_rate, num_iters):
    global k, coef
    x = x0
    y = y0
    nc = [(x0, y0)]
    a = None
    for i in range(0, num_iters):
        grad_x, grad_y = ncasual(coef, x, y, f)
        x -= learning_rate * grad_x
        y -= learning_rate * grad_y
        nc.append((x, y))
        if a is None and i > 10 and abs(f(x, y) - f(nc[-2][0],nc[-2][1])) < 1e-5:
            a = 0
            print('NC收敛',i) 
    print(f'NCGD收敛误差:{abs(f(nc[-1][0],nc[-1][1]) - 0)}')
    return (x, y), nc

def gradient_descent(f, df, x0, y0, learning_rate, num_iters):
    x = x0
    y = y0
    gd = [(x0, y0)]
    a = None
    for i in range(num_iters):
        dx, dy = df(x, y)
        x -= learning_rate * dx
        y -= learning_rate * dy
        gd.append((x, y))
        if a is None and i > 10 and abs(f(x, y) - f(gd[-2][0],gd[-2][1])) < 1e-5:
            a = 0
            print('GD收敛',i) 
    print(f'GD收敛误差:{abs(f(gd[-1][0],gd[-1][1]) - 0)}')
    return (x, y), gd

def nadam(f, df, x0, y0, learning_rate, num_iters, beta1=0.9, beta2=0.999, epsilon=1e-8):
    x = x0
    y = y0
    m_x = m_y = v_x = v_y = 0
    nada = [(x0, y0)]
    a = None
    for i in range(1, num_iters + 1):
        grad_x, grad_y = df(x, y)
        m_x = beta1 * m_x + (1 - beta1) * grad_x
        m_y = beta1 * m_y + (1 - beta1) * grad_y
        v_x = beta2 * v_x + (1 - beta2) * grad_x**2
        v_y = beta2 * v_y + (1 - beta2) * grad_y**2
        m_hat_x = m_x / (1 - beta1**i)
        m_hat_y = m_y / (1 - beta1**i)
        v_hat_x = v_x / (1 - beta2**i)
        v_hat_y = v_y / (1 - beta2**i)
        x -= learning_rate * (beta1 * m_hat_x + (1 - beta1) * grad_x / (1 - beta1**i)) / (np.sqrt(v_hat_x) + epsilon)
        y -= learning_rate * (beta1 * m_hat_y + (1 - beta1) * grad_y / (1 - beta1**i)) / (np.sqrt(v_hat_y) + epsilon)
        nada.append((x, y))
        if a is None and i > 10 and abs(f(x, y) - f(nada[-2][0],nada[-2][1])) < 1e-5:
            a = 0
            print('NAdam收敛',i) 
    print(f'NAdam收敛误差:{abs(f(nada[-1][0],nada[-1][1]) - 0)}')
    return (x, y), nada

def stochastic_gradient_descent(f, df, x0, y0, learning_rate, num_iters):
    x = x0
    y = y0
    sgd = [(x0, y0)]
    a = None
    for i in range(num_iters):
        i = random.random()
        if i <= 0.5:
            x -= learning_rate * df(x, y)[0]
        else:
            y -= learning_rate * df(x, y)[1]
        sgd.append((x, y))
        if a is None and i > 10 and abs(f(x, y) - f(sgd[-2][0],sgd[-2][1])) < 1e-5:
            a = 0
            print('SGD收敛',i)
    print(f'SGD收敛误差:{abs(f(sgd[-1][0],sgd[-1][1]) - 0)}')
    return (x, y), sgd

def nag(f, df, x0, y0, learning_rate, num_iters, momentum=0.9):
    x = x0
    y = y0
    v_x = v_y = 0
    nag = [(x0, y0)]
    a = None
    for i in range(num_iters):
        grad_x, grad_y = df(x - momentum * v_x, y - momentum * v_y)
        v_x = momentum * v_x + learning_rate * grad_x
        v_y = momentum * v_y + learning_rate * grad_y
        x -= v_x
        y -= v_y
        nag.append((x, y))
        if a is None and i > 10 and abs(f(x, y) - f(nag[-2][0],nag[-2][1])) < 1e-5:
            a = 0
            print('NAG收敛',i)
    print(f'NAG收敛误差:{abs(f(nag[-1][0],nag[-1][1]) - 0)}')
    return (x, y), nag

def rmsprop(df, x0, y0, learning_rate, num_iters, beta=0.9, epsilon=1e-8):
    x = x0
    y = y0
    grad_squared_x = grad_squared_y = 0
    rms = [(x0, y0)]
    a = None
    for i in range(num_iters):
        grad_x, grad_y = df(x, y)
        grad_squared_x = beta * grad_squared_x + (1 - beta) * grad_x ** 2
        grad_squared_y = beta * grad_squared_y + (1 - beta) * grad_y ** 2
        x -= learning_rate * grad_x / (np.sqrt(grad_squared_x) + epsilon)
        y -= learning_rate * grad_y / (np.sqrt(grad_squared_y) + epsilon)
        rms.append((x, y))
        if a is None and i > 10 and abs(f(x, y) - f(rms[-2][0],rms[-2][1])) < 1e-5:
            a = 0
            print('rms收敛',i)
    print(f'RMS收敛误差:{abs(f(rms[-1][0],rms[-1][1]) - 0)}')
    return (x, y), rms

def adamax(df, x0, y0, learning_rate, num_iters, beta1=0.9, beta2=0.999, epsilon=1e-8):
    x = x0
    y = y0
    m_x = m_y = v_x = v_y = 0
    adamax = [(x0, y0)]
    a = None
    for i in range(1, num_iters + 1):
        grad_x, grad_y = df(x, y)
        m_x = beta1 * m_x + (1 - beta1) * grad_x
        m_y = beta1 * m_y + (1 - beta1) * grad_y
        v_x = max(beta2 * v_x, abs(grad_x))
        v_y = max(beta2 * v_y, abs(grad_y))
        m_hat_x = m_x / (1 - beta1 ** i)
        m_hat_y = m_y / (1 - beta1 ** i)
        x -= learning_rate * m_hat_x / (v_x + epsilon)
        y -= learning_rate * m_hat_y / (v_y + epsilon)
        adamax.append((x, y))
        if a is None and i > 10 and abs(f(x, y) - f(adamax[-2][0],adamax[-2][1])) < 1e-5:
            a = 0
            print('ADAMAX收敛',i)
    print(f'ADAMAX收敛误差:{abs(f(adamax[-1][0],adamax[-1][1]) - 0)}')
    return (x, y), adamax

def adamw(df, x0, y0, learning_rate, num_iters, weight_decay=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    x = x0
    y = y0
    m_x = m_y = v_x = v_y = 0
    adaw = [(x0, y0)]
    a = None
    for i in range(1, num_iters + 1):
        grad_x, grad_y = df(x, y)
        m_x = beta1 * m_x + (1 - beta1) * grad_x
        m_y = beta1 * m_y + (1 - beta1) * grad_y
        v_x = beta2 * v_x + (1 - beta2) * grad_x ** 2
        v_y = beta2 * v_y + (1 - beta2) * grad_y ** 2
        m_hat_x = m_x / (1 - beta1 ** i)
        m_hat_y = m_y / (1 - beta1 ** i)
        v_hat_x = v_x / (1 - beta2 ** i)
        v_hat_y = v_y / (1 - beta2 ** i)
        x -= learning_rate * (m_hat_x / (np.sqrt(v_hat_x) + epsilon) + weight_decay * x)
        y -= learning_rate * (m_hat_y / (np.sqrt(v_hat_y) + epsilon) + weight_decay * y)
        adaw.append((x, y))
        if a is None and i > 10 and abs(f(x, y) - f(adaw[-2][0],adaw[-2][1])) < 1e-5:
            a = 0
            print('ADAW收敛',i)
    print(f'ADAW收敛误差:{abs(f(adaw[-1][0],adaw[-1][1]) - 0)}') 
    return (x, y), adaw

learning_rate = 0.01
num_iters = 200

# 测试函数1 
# def f(x,y):
#     out = 2*x**2-1.05*x**4+1/6*x**6 + x*y + y**2
#     return out

# def df(x,y):
#     gx = 4*x - 4.2*x**3 + 1.2*x**5 + y
#     gy = 2*y + x
#     return gx,gy

# # random.seed(2024)
# x0 = -2.7329
# y0 = 4.6629
# X = np.linspace(-3, 3, 200)
# Y = np.linspace(-5, 5, 200)


#测试函数2
def f(x,y):
	out = 2*(x-5)**2 + 3*(y-6)**2 + 10
	return out

def df(x,y):
    gx = 4*(x-5)
    gy = 6*(y-6)
    return gx,gy

random.seed(4)
x0 = 2.3604 
y0 = 2.0316
X = np.linspace(0, 10, 200)
Y = np.linspace(1, 11, 200)

#测试函数3
# def f(x,y):
#     out = x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y) + 20
#     return out

# def df(x,y):
#     gx = 2*x + 20*np.pi*np.sin(2*np.pi*x)
#     gy = 2*y + 20*np.pi*np.sin(2*np.pi*y)
#     return gx,gy

# # random.seed(4)
# x0 = 0.7311
# y0 = -0.7281
# # print(x0,y0)
# X = np.linspace(-5.12, 5.12, 200)
# Y = np.linspace(-5.12, 5.12, 200)

#测试函数4
# def f(x,y):
# 	out = (1-x)**2 + 100*(y-x**2)**2
# 	return out

# def df(x,y):
#     gx = 2*(200*x**3 - 200*x*y + x - 1)
#     gy = 200*(y - x**2)
#     return gx,gy

# random.seed(5)
# x0 = random.uniform(-15, 15)
# y0 = random.uniform(-30, 30)
# X = np.linspace(-30, 30, 200)
# Y = np.linspace(-30, 30, 200)

#测试函数5
# def f(x,y):
#     out = -(1 + np.cos(12 * np.sqrt(x**2 + y**2))) / (0.5*(x**2 + y**2) + 2)
#     return out

# def df(x,y):
#     gx = 12*x*np.sin(12* np.sqrt(x**2 + y**2))/(x**2 + y**2)**(3/2) + x/(x**2 + y**2 + 4)
#     gy = 12*y*np.sin(12* np.sqrt(x**2 + y**2))/(x**2 + y**2)**(3/2) + y/(x**2 + y**2 + 4)
#     return gx,gy
# random.seed(6)
# x0 = random.uniform(-5.12, 5.12)
# y0 = random.uniform(-5.12, 5.12)
# X = np.linspace(-5.12, 5.12, 200)
# Y = np.linspace(-5.12, 5.12, 200)

# 测试函数6
# def f(x,y):
#     out = (x - 1)**2 +  + 2 * (2 * y ** 2- x)**2
#     return out

# def df(x,y):
#     gx = 2*(2*x - 4*y**2 + 1)
#     gy = 8*y*(2*y**2 - x)
#     return gx,gy

# random.seed(2)
# x0 = random.uniform(-10, 10)
# y0 = random.uniform(-10, 10)
# X = np.linspace(-10, 10, 200)
# Y = np.linspace(-10, 10, 200)

marker = ['o','v','^','<','>','1','2','3','4','8','s','p','*','h','H','+','x','D','d','|','_']
ls = ['-','--','-.',':']


nc, ncgd1 = ncgd(x0, y0, f, learning_rate, num_iters)
gd, gdgd = gradient_descent(f, df, x0, y0, learning_rate, num_iters)
nd, ndgd = nadam(f, df, x0, y0, learning_rate, num_iters)
sg, sggd = stochastic_gradient_descent(f, df, x0, y0, learning_rate, num_iters)
ng, nggd = nag(f, df, x0, y0, learning_rate, num_iters)
rg, rms = rmsprop(df, x0, y0, learning_rate, num_iters)
am, adamax1 = adamax(df, x0, y0, learning_rate, num_iters)
aw, adamw1 = adamw(df, x0, y0, learning_rate, num_iters)
paths = [gdgd, ndgd, sggd, nggd, rms, adamax1, adamw1, ncgd1]
labels = ['GD', 'NAdam', 'SGD', 'NAG', 'RMSProp', 'Adamax', 'Adamw', 'NCGD']

random.seed(2)
plt.figure(figsize=(12, 8))

for path, label in zip(paths, labels):
    plt.plot([i[0] for i in path], label=label, marker=random.choice(marker), linestyle=random.choice(ls), markevery = int(num_iters/50))
plt.ylabel('X值',fontproperties = font, fontsize = 16)
plt.xlabel('迭代次数',fontproperties = font, fontsize = 16)
plt.legend()
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
# plt.savefig('fig/singlex.pdf',dpi = 500)

plt.figure(figsize=(12, 8))
random.seed(2)
for path, label in zip(paths, labels):
    plt.plot([i[1] for i in path], label=label, marker=random.choice(marker), linestyle=random.choice(ls), markevery = int(num_iters/50))
plt.ylabel('Y值',fontproperties = font, fontsize = 16)
plt.xlabel('迭代次数',fontproperties = font, fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend()
# plt.savefig('fig/singley.pdf',dpi = 500)

plt.figure(figsize=(12, 8))
random.seed(2)
for path, label in zip(paths, labels):
    plt.plot([f(i[0],i[1]) for i in path], label=label, marker=random.choice(marker), linestyle=random.choice(ls), markevery = int(num_iters/50))
plt.ylabel('函数值',fontproperties = font,fontsize = 16)
plt.xlabel('迭代次数',fontproperties = font,fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(loc = 'upper right')
# plt.savefig('fig/singlef.pdf',dpi = 500)


A, B = np.meshgrid(X, Y)
z = f(A, B)

# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(A, B, z, cmap = 'Greys', alpha = 0.5)

random.seed(2)
fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(111, projection='3d')
ax1.plot_surface(A, B, z, cmap = 'viridis', alpha = 0.5)

for path, label in zip(paths, labels):
    path = [[i[0], i[1], f(i[0], i[1])] for i in path]

    x_path = [i[0] for i in path]
    y_path = [i[1] for i in path]
    z_path = [i[2] for i in path]

    ax1.plot(x_path, y_path, z_path, label=label, marker = random.choice(marker), linestyle = random.choice(ls), markevery = int(num_iters/50))
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('f(x,y)')
ax1.tick_params(axis='both', which='major', labelsize=16)
# ax1.set_xticks(fontsize = 16)
# ax1.set_yticks(fontsize = 16)
ax1.legend()
# plt.savefig('single3D.pdf',dpi = 500)

fig = plt.figure(figsize=(12, 8))

ax2 = fig.add_subplot(111)

contour = ax2.contour(A, B, z, cmap='viridis', levels = 150, alpha = 0.5)

for path, label in zip(paths, labels):
    path = [[i[0], i[1], f(i[0], i[1])] for i in path]

    x_path = [i[0] for i in path]
    y_path = [i[1] for i in path]

    ax2.plot(x_path, y_path, label=label, marker = random.choice(marker), linestyle = random.choice(ls), markevery = int(num_iters/50))
ax2.set_xlabel('X',fontsize = 16)
ax2.set_ylabel('Y',fontsize = 16)
# ax2.set_xticks(fontsize = 16)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax2.legend()
# plt.savefig('fig/singlep.pdf',dpi = 500)
plt.show()
