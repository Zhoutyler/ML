#!/usr/bin/env python
__author__ = "Yanzhao Zhou yz3395"
import numpy as np
from numpy.linalg import inv
from scipy.special import digamma, gamma
from numpy.linalg import det
import matplotlib
import matplotlib.pyplot as plt
from numpy.linalg import slogdet
from scipy.special import comb
from scipy.special import gammaln

class BinomialMixtureModel:
    def __init__(self, x, k, alpha, a, b):
        self.x = LoadCsv(x)
        self.k = k
        self.n = self.x.shape[0]
        # maintain initializing value
        self.alpha0 = np.random.normal(alpha, 0.01, self.k)
        self.a0 = a
        self.b0 = b
        # used to update
        self.alpha = self.alpha0
        self.a = np.full(self.k, a)
        self.b = np.full(self.k, b)
        self.c = np.ones((self.n, self.k))
        self.c = self.c / self.k
        print (self.alpha)
        self.logcombx = np.log(comb(20, self.x))

    def updateC(self):
        dg_a = digamma(self.a)
        dg_b = digamma(self.b)
        dg_ab = digamma(self.a + self.b)
        dg_alpha = digamma(self.alpha)
        dg_nalpha = digamma(np.sum(self.alpha))
        for i in range(self.n):
            tmp = self.x[i] * (dg_a - dg_ab) + (20 - self.x[i]) * (dg_b - dg_ab) + dg_alpha - dg_nalpha
            self.c[i] = np.exp(tmp)
            self.c[i] *= comb(20, self.x[i])
            self.c[i] = self.c[i] / np.sum(self.c[i])
        print (self.c)
    
    def updateAlpha(self):
        self.alpha = np.sum(self.c, axis=0) + self.alpha0
        print(self.alpha)

    def updateAB(self):
        self.a = self.a0 + np.dot(self.c.T, self.x).ravel()
        self.b = self.b0 + np.dot(self.c.T, 20 - self.x).ravel()
        print(self.a)
        print(self.b)
    
    def VI(self, iteration):
        self.result = []
        for t in range(iteration):
            print("----iteration %d----" % t)
            self.updateC()
            self.updateAlpha()
            self.updateAB()
            L = self.computeL()
            self.result.append(L)

    def computeL(self):
        dg_a = digamma(self.a)
        dg_b = digamma(self.b)
        dg_ab = digamma(self.a + self.b)
        dg_alpha = digamma(self.alpha)
        dg_nalpha = digamma(np.sum(self.alpha))

        L = 0
        L += np.dot(np.dot(dg_a - dg_ab, self.c.T), self.x) + np.dot(np.dot(dg_b - dg_ab, self.c.T), 20-self.x)
        L += np.sum(np.dot(self.logcombx, self.c))
        L += np.sum(np.dot(dg_alpha - dg_nalpha, self.c.T))
        L += np.dot(self.alpha0 - 1, (dg_alpha - dg_nalpha)[np.newaxis].T)
        L += np.sum((self.a0 - 1) * (dg_a - dg_ab) + (self.b0 - 1) * (dg_b - dg_ab))
        L -= np.sum(np.multiply(self.c, np.log(self.c)))
        L -= np.dot(self.a - 1, (dg_a - dg_ab)[np.newaxis].T) + np.dot(self.b - 1, (dg_b - dg_ab)[np.newaxis].T) - np.sum(gammaln(self.a)+gammaln(self.b)-gammaln(self.a + self.b))
        L -= np.dot(self.alpha - 1, (dg_alpha - dg_nalpha)[np.newaxis].T) - np.sum(gammaln(self.alpha)) + gammaln(np.sum(self.alpha))
        print ("final L: %f" % L)
        return L

    def findCluster(self):
        self.cluster = []
        for n in range(20 + 1):
            idx = np.where(self.x == n)
            self.cluster.append(np.argmax(self.c[idx,:][0]))

def LoadCsv(filename):
    try:
        return np.loadtxt(open(filename, 'rb'), delimiter=',')
    except Exception as e:
        print ("Error: %e\n" % e) 


if __name__ == "__main__":
    plt.rcParams['axes.unicode_minus'] = False
    alpha = 0.1
    a = 0.5
    b = 0.5
    t = 1000
    k = [3, 15, 50]
    result1 = []
    result2 = []

    # plot VI objective function for cluster number=3, 15, 50 #
    for i in range(len(k)):
        Bmm = BinomialMixtureModel("./x.csv", k[i], alpha, a, b)
        Bmm.VI(t)
        Bmm.findCluster()
        result1.append(Bmm.result)
        result2.append(Bmm.cluster)
    for i in range(len(k)):
        plt.plot(np.arange(1, t+1), result1[i])
    plt.title(r"VI Objective Function $L$")
    plt.xticks(np.arange(1, t+1, t/10))
    plt.xlabel("Iteration")
    plt.show()
    
    # find most propable cluster for the integer 0-20 #
    for i in range(len(k)):
        plt.stem(np.arange(0, 21), result2[i])
        plt.xticks(np.arange(0, 21, 1))
        plt.yticks(np.arange(0, k[i], 1))
        plt.xlabel("Integer")
        plt.ylabel("Cluster index")
        plt.title("Most probable cluster for integer 0 - 20, k = %d" % k[i])
        plt.show()
        
    
    
