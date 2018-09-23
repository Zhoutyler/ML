#!/usr/bin/env python
import pandas as pd
import numpy as np
from scipy.stats import nbinom

class NaiveBayesian:
    def __init__(self, x_train_path, y_train_path, x_test_path):
        self.X_train = self.LoadCsv(x_train_path);
        self.y_train = self.LoadCsv(y_train_path);
        self.x_test = self.LoadCsv(x_test_path);
        self.dim = self.X_train.shape[1]
        self.trainnum = self.X_train.shape[0]
        self.testnum = self.x_test.shape[0]
        print(self.X_train.shape)
        print(self.y_train.shape)
        print(self.x_test.shape)

    def LoadCsv(self, filename):
        try:
            return np.loadtxt(open(filename, 'rb'), delimiter=',')
        except Exception as e:
            printf("Error: %e\n", e)

    # Input: label y*=y, d for dth column xd #
    # Compute p(x*|y*=y, {xi:yi=y})     #
    def Likelihood(self, y, d):
        idx = np.where(self.y_train == y)
        N = len(idx[0])
        #print(N)
        #print(self.X_train[idx,d].shape)
        r = np.sum(self.X_train[idx,d]) + 1 # sigma xdi + alpha, where yi = y
        p = (N + 1) / (N + 2.0)      # (N+beta)/(N+beta+1) here beta=1
        # print r, p
        return nbinom.pmf(self.x_test[:,d], r, p)
        
    def Prior(self, y):
        sigma = len(np.where(self.y_train == y)[0])
        N = self.trainnum
        print "Prior of y*:", (1 + sigma) / (2.0 + N)
        return (1 + sigma) / (2.0 + N)   # pi~Beta(e,f) p(y*=1|y) = (e + sigma 1(yi = 1))/(N+e+f) p(y*=0|y) = (f+sigma 1(yi=0))/(N+e+f) here e=f=1

    def Posterior(self, y):
        res = np.ones(self.testnum) 
        for d in range(0, self.dim):
            res = np.multiply(self.Likelihood(y, d), res)  # Naive Bayesian classifier treate each column xd independent with one another
        # for d = 1:54 Likelihood(y,d)
        return res * self.Prior(y)
    
    def Gen_label(self):
        # Posterior(y)
        p1 = self.Posterior(1)
        p0 = self.Posterior(0)
        #print p1.shape
        #print p0.shape
        self.y_pred = np.where(p1>p0, np.ones(self.testnum), np.zeros(self.testnum))
        return self.y_pred

    def Confusion_matrix(self, label_test_path):
        y_truth = self.LoadCsv(label_test_path)
        #print self.y_truth.shape
        diff = y_truth != self.y_pred
        zero2zero = len(np.where((np.logical_not(diff)) & (y_truth == 0))[0])
        one2one = len(np.where((np.logical_not(diff)) & (y_truth == 1))[0])
        one2zero = len(np.where((diff) & (y_truth == 0))[0])
        zero2one = len(np.where((diff) & (y_truth == 1))[0])
        return np.array([[zero2zero, zero2one], [one2zero, one2one]])

if __name__ == '__main__':
    classifier = NaiveBayesian("./X_train.csv", "label_train.csv", "X_test.csv")
    print classifier.Likelihood(1, 0)
    #print classifier.Likelihood(0, 0).shape
    #print classifier.Posterior(1).shape
    print classifier.Posterior(1)
    print classifier.Posterior(0)
    print classifier.Gen_label()
    print classifier.Confusion_matrix("./label_test.csv")

