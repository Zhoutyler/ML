#!/usr/bin/env python
__author__ = "Yanzhao Zhou yz3395"

import numpy as np
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
import pickle

class Recomendation:
    def __init__(self, r, d):
        self.data = self.LoadCsv(r)
        user = self.data[:,0]
        movie = self.data[:,1]
        rating = self.data[:,2]
        max1 = int(np.max(user) + 1)
        max2 = int(np.max(movie) + 1)
        #print (np.max(user))
        #print (np.max(movie))
        self.Eq = np.zeros((max1, max2))
        #print(self.Eq.shape)
        self.d = d
        
        self.u = np.zeros((d, max1))
        self.v = np.zeros((d, max2))
        #x = np.random.binomial(size=1, n=1, p = 0.1)
        #print (x[0])

    def Initialize(self, miu):
        miu_v = np.zeros(self.d) + miu
        for i in range (self.u.shape[1]):
            self.u[:,i] = np.random.multivariate_normal(miu_v, 0.1 * np.identity(self.d))
        for j in range (self.v.shape[1]):
            self.v[:,j] = np.random.multivariate_normal(miu_v, 0.1 * np.identity(self.d))
        print (self.u)
        print (self.v)

    def Confusion_matrix(self, test_path):
        truth = self.LoadCsv(test_path)
        truth_matrix = np.zeros((self.u.shape[1], self.v.shape[1]))
        pred = np.zeros((self.u.shape[1], self.v.shape[1]))
        for k in range(truth.shape[0]):
            i = int(truth[k,0])
            j = int(truth[k,1])
            rating = int(truth[k,2])
            truth_matrix[i-1,j-1] = rating
            p = norm.cdf(np.dot(self.u[:,i], self.v[:,j]))
            #print (p)
            pred[i-1,j-1] = np.random.binomial(size=1, n=1, p=p)[0]
            #print (self.pred[i-1,j-1])
        diff = truth_matrix != pred
        zero2zero = len(np.where((pred == 0) & (truth_matrix == -1))[0])
        one2one = len(np.where((pred == 1) & (truth_matrix == 1))[0])
        one2zero = len(np.where((pred == 1) & (truth_matrix == -1))[0])
        zero2one = len(np.where((pred == 0) & (truth_matrix == 1))[0])
        return np.array([[zero2zero, zero2one], [one2zero, one2one]])

    def E_step(self):
        for n in range(len(self.data)):
            i = int(self.data[n,:][0])
            #print i 
            j = int(self.data[n,:][1])
            rating = int(self.data[n,:][2])
            if rating == 1:
                self.Eq[i,j] = np.dot(self.u[:,i], self.v[:,j]) + (norm.pdf(-1 * np.dot(self.u[:,i], self.v[:,j])) / float(1 - norm.cdf(-1 * np.dot(self.u[:,i], self.v[:,j]))))
            else: 
                self.Eq[i,j] = np.dot(self.u[:,i], self.v[:,j]) + (-1 * norm.pdf(-1 * np.dot(self.u[:,i], self.v[:,j])) / float(norm.cdf(-1 * np.dot(self.u[:,i], self.v[:,j]))))
        pass

    def M_step(self):
        for i in range(self.u.shape[1]):
            # found those j in omega(i,j)
            r = self.Eq[i,:]
            l = np.where(r != 0)[0]
            ans = np.zeros((self.d, self.d))
            ans2 = np.zeros((self.d, 1))
            for j in l: 
                #print (j)
                #print ( np.dot(self.v[:,j].reshape(self.d, 1), self.v[:,j].reshape(1, self.d)).shape)
                ans += np.dot(self.v[:,j].reshape(self.d, 1), self.v[:,j].reshape(1, self.d))
            ans = ans + np.identity(self.d)
            ans = inv(ans)
            for j in l:
                if self.Eq[i,j] == 0:
                    raise ValueError
                #print (self.Eq[i,j])
                #print (self.v[:, j].shape)
                #print (ans2.shape)
                ans2 += (self.Eq[i,j] * self.v[:,j].reshape(self.d,1))
            self.u[:,i] = np.dot(ans, ans2).ravel()
        #self.v =
        for j in range(self.v.shape[1]):
            # found those j in omega(i,j)
            r = self.Eq[:,j]
            l = np.where(r != 0)[0]
            ans = np.zeros((self.d, self.d))
            ans2 = np.zeros((self.d, 1))
            for i in l: 
                #print (j)
                #print ( np.dot(self.v[:,j].reshape(self.d, 1), self.v[:,j].reshape(1, self.d)).shape)
                ans += np.dot(self.u[:,i].reshape(self.d, 1), self.u[:,i].reshape(1, self.d))
            ans = ans + np.identity(self.d)
            ans = inv(ans)
            for i in l:
                if self.Eq[i,j] == 0:
                    raise ValueError
                #print (self.Eq[i,j])
                #print (self.v[:, j].shape)
                #print (ans2.shape)
                ans2 += (self.Eq[i,j] * self.u[:,i].reshape(self.d,1))
            self.v[:,j] = np.dot(ans, ans2).ravel()

        pass
    def Posterior(self):
        self.Post = self.u.shape[1] * self.v.shape[1] * np.log(2 * math.pi) * (-1)
        for i in range(self.u.shape[1]):

            self.Post += (-0.5) * (np.dot(self.u[:,i], self.u[:,i]))
        for j in range(self.v.shape[1]):
            self.Post += (-0.5) * (np.dot(self.v[:,j], self.v[:,j]))
        for k in range(self.data.shape[0]):
            i = int(self.data[k,0])
            j = int(self.data[k,1])
            self.Post += self.data[k,2] * np.log(norm.cdf(np.dot(self.u[:,i], self.v[:,j])))
            self.Post += (1 - self.data[k,2]) * np.log(1 - norm.cdf(np.dot(self.u[:,i], self.v[:,j])))
    def LoadCsv(self, filename):
        try:
            return np.loadtxt(open(filename, 'rb'), delimiter=',')
        except Exception as e:
            printf("Error: %e\n", e)

if __name__ == "__main__":
    matplotlib.rcParams['axes.unicode_minus']=False
    Movie = Recomendation("./ratings.csv", 5)
    Movie.Initialize(0)
    ans = []
    for i in range(100):
        Movie.E_step()
        #print (np.where(Movie.Eq != 0))
        Movie.M_step()
        #print (Movie.u)
        #print (Movie.v)
        print ("Iteration %s\n", i)
        Movie.Posterior()
        print ("\n")
        print (Movie.Post)
        if (i != 0):
            ans.append(Movie.Post)
        print (Movie.Confusion_matrix("ratings_test.csv"))
    plt.plot(ans)
    plt.xlabel("iterations")
    plt.ylabel("Log joint likelihood")
    plt.show()
    plt.savefig("./result1.png")
    print (Movie.Confusion_matrix("ratings_test.csv"))
    pickle.dump(Movie.u, open("u.p", "wb"))
    pickle.dump(Movie.v, open("v.p", "wb"))

