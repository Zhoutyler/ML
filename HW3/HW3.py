#!/usr/bin/env python
__author__ = "Yanzhao Zhou yz3395"
import numpy as np
from numpy.linalg import inv
from scipy.special import digamma, gamma
from numpy.linalg import det
import matplotlib
import matplotlib.pyplot as plt
from numpy.linalg import slogdet

class LinearRegression:
    def __init__(self, x, y, a, b, e, f):
        
        # Load data
        self.y = LoadCsv(y)
        self.x = LoadCsv(x)
        print (self.x.shape)
        print (self.y.shape)
        self.d = self.x.shape[1]

        # Initialize parameters
        self.miu = np.zeros((self.d, 1))
        self.a0 = a
        self.b0 = b
        self.e0 = e
        self.f0 = f
        self.a = np.full((self.d,), a)
        self.b = np.full((self.d,), b)
        self.e = e
        self.f = f
        self.result = []

        # Initialize alpha1 to alphaD, used to initialize w1
        self.alpha = np.random.gamma(1, 1, self.d)  # shape k = a, scale theta = 1/beta
        print (self.alpha)
        self.sigma = np.diag(self.alpha)
        self.sigma = inv(self.sigma)
    
    def updateMiuSigma(self):
        tmp = self.a / self.b
        tmp1 = np.zeros((self.d, self.d))
        for i in range(self.x.shape[0]):
            tmp1 += np.dot(self.x[i][np.newaxis].T, self.x[i][np.newaxis])    
                    
        self.sigma = (np.diag(tmp) + self.e / self.f * tmp1)
        self.sigma = inv(self.sigma)
        
        tmp2 = np.zeros((self.d, 1))
        for i in range(self.x.shape[0]):
            tmp2 += self.y[i] * (self.x[i][np.newaxis].T)
        self.miu = np.dot(self.sigma, self.e / self.f * tmp2)

    def updateAiBi(self, i):
        self.a[i] = self.a0 + 0.5
        self.b[i] = self.b0 + 0.5 * (self.sigma[i][i] + self.miu[i][0] * self.miu[i][0])

    def updateEF(self):
        self.e = self.e0 + 0.5 * self.x.shape[0]
        tmp = 0
        for i in range(self.x.shape[0]):
            tmp += (self.y[i] - np.dot(self.x[i], self.miu)[0]) ** 2
            tmp += np.dot(np.dot(self.x[i], self.sigma), self.x[i][np.newaxis].T)[0]

        self.f = self.f0 + 0.5 * tmp

    def VI(self, iteration):
        for i in range(iteration):
            for j in range(self.d):
                self.updateAiBi(j)
            self.updateEF()
            self.updateMiuSigma()
            L = self.computeL()
            print(L)
            self.result.append(L)

    def computeL(self):
        L = 0
        L += -0.5 * (np.trace(np.dot(np.diag(self.a / self.b), self.sigma)) + np.dot(np.dot(self.miu.T, np.diag(self.a / self.b)), self.miu)[0][0])
        for i in range(self.d):
            L += 0.5 * (digamma(self.a[i]) - np.log(self.b[i]))

        tmp = 0
        for i in range(self.d):
            tmp += (digamma(self.a[i]) - np.log(self.b[i]))
        tmp *= (self.a0 - 1)
        for i in range(self.d):
            tmp -= self.b0 * self.a[i] / self.b[i]
        L += tmp
        
        tmp = 0
        tmp += (self.e0 - 1) * (digamma(self.e) - np.log(self.f)) - self.f0 * self.e / self.f
        L += tmp
        
        tmp = 0
        tmp += 0.5 * self.x.shape[0] * (digamma(self.e) - np.log(self.f))
        for i in range(self.x.shape[0]):
            tmp -= 0.5 * self.e / self.f * ((self.y[i] - np.dot(self.x[i], self.miu)[0]) ** 2 + np.dot(np.dot(self.x[i], self.sigma), self.x[i][np.newaxis].T)[0])
        L += tmp
        
        tmp = 0
        sign, logdet = slogdet(self.sigma)
        tmp = 0.5 * sign * logdet
        L += tmp

        tmp = 0
        for i in range(self.d):
            tmp += (1 - self.a[i]) * digamma(self.a[i]) + np.log(gamma(self.a[i])) - np.log(self.b[i]) + self.a[i]
        L += tmp
        
        tmp = 0
        tmp += (1 - self.e) * digamma(self.e) - np.log(self.f) + self.e
        L += tmp

        return L
        
def LoadCsv(filename):
    try:
        return np.loadtxt(open(filename, 'rb'), delimiter=',')
    except Exception as e:
        print ("Error: %e\n" % e) 

def plot(result,labelx, labely, title, x = None):
    if x is not None:
        plt.plot(x, result)
    else:
        plt.plot(result)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(title)
    plt.show()

def stem(result,labelx, labely, title, x = None):
    if x is not None:
        plt.stem(x, result)
    else:
        plt.stem(result)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(title)
    plt.show()

        

if __name__ == "__main__":
    plt.rcParams['axes.unicode_minus'] = False

    #-------------------------------3 datasets-----------------------------------#
    x_path = ["data_csv/X_set1.csv", "data_csv/X_set2.csv", "data_csv/X_set3.csv"]
    y_path = ["data_csv/y_set1.csv", "data_csv/y_set2.csv", "data_csv/y_set3.csv"]
    z_path = ["data_csv/z_set1.csv", "data_csv/z_set2.csv", "data_csv/z_set3.csv"]

    for i in range(len(x_path)):
        # Do variational inferance for 3 datasets
        lr = LinearRegression(x_path[i], y_path[i], 1e-16, 1e-16, 1, 1)
        lr.VI(500)

        # plot objective function #
        plot(lr.result, "t", r'$L(\mu^{\prime},\Sigma^{\prime},a^{\prime},b^{\prime},e^{\prime},f^{\prime})$', "Variational objective function over iterations")
        
        # plot 1 / expAlpha in the final iteration #
        expAlpha = lr.a / lr.b
        stem(1/expAlpha, "k", r'$E_q[\alpha_{k}]$', r'$E_q[\alpha_{k}]$')
        
        # plot 1/ expLambda in the final iteration #
        expLambda = lr.e / lr.f
        print ("final 1/Eq[lambda]: %f" % (1/expLambda))

        # Comparison btw regression, data points and ground truth #
        y_pred = np.dot(lr.x, lr.miu)
        z = LoadCsv(z_path[i])
        
        plt.xlabel(r'$z_i$')
        plt.plot(z, y_pred, 'r--', label = r'$\widehat{y_i}$' + " (regression)")
        plt.plot(z, lr.y, 'bo', label = r'$y_i$'+" (data points)")
        sincFunction = 10 * np.sinc(z)
        plt.plot(z, sincFunction, 'k-', label = r'$10sinc(z_i)$'+" (ground truth)")
        plt.legend()
        plt.title("Regresssion result v.s. Data points v.s. Ground truth")
        plt.show()


        
    
    
