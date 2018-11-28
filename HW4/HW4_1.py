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

class BinomialMixtureModel:
    def __init__(self, x, k):
        self.x = LoadCsv(x)
        self.k = k
        self.n = self.x.shape[0]
        # Draw pi from dirichlet distribution
        # (Draw from uniform distribution will cause bug sometimes(when k = 3, n = 50))
        self.pi = np.random.dirichlet(np.full(self.k, 0.75), 1)[0]  
        self.c = np.zeros((self.n, self.k))
        self.theta = np.ones(self.k) / 3
        self.result = []

    def E_step(self):
        # compute c_i(j)
        for i in range(self.n):
            for j in range(self.k):
               self.c[i][j] = self.pi[j] * comb(20, self.x[i]) * (self.theta[j] ** self.x[i]) * ((1-self.theta[j]) ** (20-self.x[i]))
            self.c[i] = self.c[i] / np.sum(self.c[i])
    
    def M_step(self):
        # update theta #
        n_j = np.sum(self.c, axis=0)
        self.theta = np.dot(self.c.T, self.x[np.newaxis].T) / (20 * n_j[np.newaxis].T)
        # update pi #
        self.pi = n_j / self.n

    def EM(self, iteration):
        for t in range(iteration):
            print ("----iteration %d-----" % t)
            self.E_step()
            self.M_step()
            L = self.Likelihood()
            print (L)
            self.result.append(L)

    def Likelihood(self):
        L = 0
        for i in range(self.n):
            tmp = 0
            for j in range(self.k):
                tmp += comb(20, self.x[i]) * (self.theta[j] ** self.x[i]) * ((1-self.theta[j]) ** (20-self.x[i])) * self.pi[j]
            L += np.log(tmp)
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
    k = [3, 9, 15]
    t = 50
    result1 = []
    result2 = []

    # run GMM ML_EM for cluster number = 3, 9, 15 #
    for i in range(len(k)):
        Bmm = BinomialMixtureModel("./x.csv", k[i])
        Bmm.EM(t)
        Bmm.findCluster()
        result1.append(Bmm.result)
        result2.append(Bmm.cluster)

    # plot marginal likelihood over iterations #
    for i in range(len(k)):
        plt.plot(np.arange(2, t+1), result1[i][1:])
    plt.xticks(np.arange(2, t+1))
    plt.title("Marginal Likelihood")
    plt.xlabel("Iteration")
    plt.show()
    
    # plot most propable cluster for integer 0 - 20 #
    for i in range(len(k)):
        plt.stem(np.arange(0, 21), result2[i])
        plt.xticks(np.arange(0, 21, 1))
        plt.yticks(np.arange(0, k[i], 1))
        plt.ylabel("Cluster index")
        plt.xlabel("Iteration")
        plt.title("Most probable cluster for integer 0 - 20, k = %d" % k[i])
        plt.show()    
    
    
