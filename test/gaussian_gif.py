import imageio
import numpy as np
import matplotlib.pyplot as plt

class RBFkernel:
    def __init__(self, *param, ):
        self.param = list(param)

    def __call__(self, x1, x2):
        a,s,w = self.param
        return a**2*np.exp(-((x1-x2)/s)**2) + w*(x1==x2)

def y(x):
    return 10*np.sin(np.pi*x/2)

n = 30
x0 = np.random.permutation(np.linspace(0.1,9.9,n))
y0 = y(x0) + np.random.normal(0,0.1,n)
gif = []
for i in range(n):
    x1 = np.linspace(0,10,1000)
    kernel = RBFkernel(8,0.5,0.1)

    k00 = kernel(*np.meshgrid(x0[:i],x0[:i]))
    k00_1 = np.linalg.inv(k00)
    k01 = kernel(*np.meshgrid(x0[:i],x1,indexing='ij'))
    k10 = k01.T
    k11 = kernel(*np.meshgrid(x1,x1))

    mu = k10.dot(k00_1.dot(y0[:i]))
    sigma = k11 - k10.dot(k00_1.dot(k01))
    std = np.sqrt(sigma.diagonal())

    fig = plt.figure()
    plt.scatter(x0[:i],y0[:i],color='w',edgecolor='b')
    plt.plot(x1,mu,'b')
    plt.plot(x1,y(x1),'r--')
    plt.fill_between(x1,mu-std,mu+std,alpha=0.1,color='b')
    plt.tight_layout()
    fig.canvas.draw()
    fig.show()
    gif.append(np.array(fig.canvas.renderer._renderer))
    plt.close()
imageio.mimsave('gp.gif',gif,fps=2.5)