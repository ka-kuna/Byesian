import numpy as np
import matplotlib.pyplot as plt

class RBFkernel:
    def __init__(self, *param, ):
        self.param = list(param)

    def __call__(self, x1, x2):
        a,s,w = self.param
        return a**2*np.exp(-((x1-x2)/s)**2) + w*(x1==x2)

def y(x):
    return 0.1*x**3-x**2+2*x+5

x0 = np.random.uniform(0,10,30)
y0 = y(x0) + np.random.normal(0,2,30)
x1 = np.linspace(-1,11,101)

kernel = RBFkernel(8,0.5,3.5)

k00 = kernel(*np.meshgrid(x0,x0))
k00_1 = np.linalg.inv(k00)
k01 = kernel(*np.meshgrid(x0,x1,indexing='ij'))
k10 = k01.T
k11 = kernel(*np.meshgrid(x1,x1))

mu = k10.dot(k00_1.dot(y0))
sigma = k11 - k10.dot(k00_1.dot(k01))


plt.scatter(x0,y0,c='#ff77aa')
plt.plot(x1,mu,'g') # 推測された平均
plt.plot(x1,y(x1),'--r') # 本物の関数
std = np.sqrt(sigma.diagonal()) # 各点の標準偏差は共分散行列の対角成分
plt.fill_between(x1,mu-std,mu+std,alpha=0.2,color='g') # 推測された標準偏差の中の領域
plt.show()