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
from scipy.stats import rankdata
class BMMGibbsSampler:
    def __init__(self, x, alpha, a, b, k):
        self.x = LoadCsv(x);
        self.n = self.x.shape[0]
        self.k = k     # j = 0, ... , k-1
        self.c = np.random.choice(range(k), self.n)
        print (self.c)
        self.theta = np.random.beta(a, b, self.k)
        self.alpha = alpha
        self.a = a
        self.b = b

    def sampleC(self):
        # calculate n_j(-i) i = 0
        self.calN_i(self.k, 0)
        for i in range(self.n):
            # generate new fi_i
            fi = np.zeros(self.k + 1)
            # update fi_i value for all occupied cluster
            idx = np.where(self.n_i > 0)
            fi[idx] = np.power(self.theta[idx], self.x[i]) * np.power(1 - self.theta[idx], 20 - self.x[i]) * self.n_i[idx] / (self.alpha + self.n - 1)
            # update fi_i value for new cluster
            fi[-1] = self.alpha / (self.alpha + self.n - 1) * gamma(self.a + self.b) * gamma(self.x[i] + self.a) * \
                    gamma(20 - self.x[i] + self.b) / (gamma(self.a) * gamma(self.b) * gamma(self.a + self.b + 20))
            fi = fi / np.sum(fi)
            # sample c_i from categorical distribution over fi
            self.c[i] = np.random.choice(self.k + 1, 1, p=fi) 
            # update n_j(-i) to n_j(i+1)
            if i != self.n - 1:
                self.updateN_i(self.k, i)
            # generate new cluster? update k and theta : do nothing
            if self.c[i] == self.k:
                self.k += 1
                self.theta = np.append(self.theta, np.random.beta(self.x[i] + self.a, 20 - self.x[i] + self.b, 1))
            
    def reindex(self):
        self.c = rankdata(self.c,  method='dense').astype(int) - 1
        print (self.c)
        self.k = np.amax(self.c) + 1
        print (self.k)
    
    def calNi(self, k):   # calculate occupied cluster, minimum length k
        self.ni = np.bincount(self.c, minlength=k)
    
    def calN_i(self, k, i):  # calculate occupied cluster except for each i-th data point
        self.calNi(k)
        self.n_i = self.ni
        self.n_i[self.c[i]] -= 1
    
    def updateN_i(self, k, i):  # update n_j(i) to n_j(i+1)
        if self.c[i] == k:  # if c[i] was assigned to the new cluster, add one column to n_j(i+1)
            self.n_i = np.append(self.n_i, 1)
        else:
            self.n_i[self.c[i]] += 1
        self.n_i[self.c[i+1]] -= 1

    def sampleTheta(self):
        tmp1 = np.zeros(self.k)
        tmp2 = np.zeros(self.k)
        for i in range(self.n):
            tmp1[self.c[i]] += self.x[i]
            tmp2[self.c[i]] += 20 - self.x[i]
        self.theta = np.random.beta(tmp1 + self.a, tmp2 + self.b, self.k)

    def sampling(self, iteration, n):   # n: used for plot
        self.result1 = []
        self.result2 = []
        for t in range(iteration):
            print ("----iteration %d----" % t)
            self.sampleC()
            self.reindex()
            self.sampleTheta()
            # get data points numbers for n most probable clusters
            self.result1.append(self.biggestCluster(n))
            self.result2.append(self.k)
    def biggestCluster(self, n):
        self.calNi(self.k)
        idx = np.argsort(self.ni)
        idx = idx[: : -1]
        # if less than n cluster, pad zeros
        length = self.ni.shape[0]
        if n > length:
            tmp = np.zeros(n)
            tmp[:length] = self.ni[idx]
            return tmp
        return self.ni[idx][:n]
def LoadCsv(filename):
    try:
        return np.loadtxt(open(filename, 'rb'), delimiter=',')
    except Exception as e:
        print ("Error: %e\n" % e) 

if __name__ == "__main__":
    datapath = "./x.csv"
    alpha = 0.75
    a = 0.5
    b = 0.5
    k = 30   # initial number of clusters
    t = 1000
    n = 6  # used for plot
    
    # Gibbs sampling #
    Bmm = BMMGibbsSampler(datapath, alpha, a, b, k)
    Bmm.sampling(t, n)
    np.savetxt("./result.txt", Bmm.c, fmt='%i')
    
    # plot number of data points in most probable clusters over iterations #
    result1 = np.asarray(Bmm.result1).T
    for i in range(result1.shape[0]):
        plt.plot(np.arange(1, t+1), result1[i])
    plt.xticks(np.arange(1, t+1, t/10))
    plt.xlabel("Iteration")
    plt.title("Numbers of data points in the %d most probable clusters" % n)
    plt.show()

    # plot number of clusters over iterations #
    plt.plot(np.arange(1, t+1), Bmm.result2)
    plt.xticks(np.arange(1, t+1, t/10))
    plt.xlabel("Iterations")
    plt.title("Number of clusters")
    plt.show()
    
