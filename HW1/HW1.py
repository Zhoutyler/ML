#!/usr/bin/env python
import pandas as pd
import numpy as np
from scipy.stats import nbinom
import matplotlib.pyplot as plt

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
        r = np.sum(self.X_train[idx,d]) + 1 # sigma xdi + alpha, where yi = y
        p = (N + 1) / (N + 2.0)      # (N+beta)/(N+beta+1) here beta=1
        return nbinom.pmf(self.x_test[:,d], r, p)
        
    def Prior(self, y):
        sigma = len(np.where(self.y_train == y)[0])
        N = self.trainnum
        # print "Prior of y*:", (1 + sigma) / (2.0 + N)
        return (1 + sigma) / (2.0 + N)   # pi~Beta(e,f) p(y*=1|y) = (e + sigma 1(yi = 1))/(N+e+f) p(y*=0|y) = (f+sigma 1(yi=0))/(N+e+f) here e=f=1

    def Posterior(self, y):
        res_y = np.ones(self.testnum)  # nominator for specified y I queried
        for d in range(0, self.dim):
            res_y = np.multiply(self.Likelihood(y, d), res_y)  # Naive Bayesian classifier treate each column xd independent with one another
       
        res_other = np.ones(self.testnum) # another y label, used on denominator
        for d in range(0, self.dim):
            res_other = np.multiply(self.Likelihood(1-y, d), res_other)  # Naive Bayesian classifier treate each column xd independent with one another

        return res_y * self.Prior(y)/(res_y * self.Prior(y) + res_other * self.Prior(1-y))
    
    def Gen_label(self):
        # Posterior(y)
        p1 = self.Posterior(1)
        p0 = self.Posterior(0)
        self.y_pred = np.where(p1>p0, np.ones(self.testnum), np.zeros(self.testnum))
        return self.y_pred

    def Confusion_matrix(self, label_test_path):
        self.y_truth = self.LoadCsv(label_test_path)
        diff = self.y_truth != self.y_pred
        zero2zero = len(np.where((np.logical_not(diff)) & (self.y_truth == 0))[0])
        one2one = len(np.where((np.logical_not(diff)) & (self.y_truth == 1))[0])
        one2zero = len(np.where((diff) & (self.y_truth == 0))[0])
        zero2one = len(np.where((diff) & (self.y_truth == 1))[0])
        return np.array([[zero2zero, zero2one], [one2zero, one2one]])

    def Lambda_mean(self, y):  # The posterior distribution of lambda given past data X
        lambda_m = []
        for d in range(0, self.dim):
            idx = np.where(self.y_train == y)
            sigma = np.sum(self.X_train[idx, d])
            N = len(idx[0])
            # Mean of Gamma(alpha,beta) is alpha / beta
            # here alpha = sigma + alpha, beta = N + beta, alpha = beta = 1
            lambda_m.append((sigma + 1)/(N + 1))
        return lambda_m

    def Plot_with_lambda(self, idx):
        for nplot, index in enumerate(idx):

            x = self.x_test[index,:]
            plt.plot(np.arange(0, self.dim), x, 'ro', label='x_test')
            plt.xticks(np.arange(0, self.dim), x_labels)
            plt.xticks(rotation=90)
            if max(x) == min(x):
                maximum = max(x) + 10
            else: 
                maximum = max(x) + 1
            plt.ylim(min(x), maximum)
            p1 = self.Posterior(1)[index]
            p0 = self.Posterior(0)[index]
            plt.title(r"$P(y^*=1|x^*,X,\vec y)=%.2f, P(y^*=0|x^*,X,\vec y)=%.2f$" % (p1, p0))
            plt.plot(np.arange(0, self.dim), self.Lambda_mean(1), 'bx', label=r'$E[\lambda_{1}]$')
            plt.plot(np.arange(0, self.dim), self.Lambda_mean(0), 'bo', label=r'$E[\lambda_{0}]$')
            plt.legend()
            #plt.subplot(1, 3, nplot+1)
        
            plt.show()

if __name__ == '__main__':
    classifier = NaiveBayesian("./X_train.csv", "label_train.csv", "X_test.csv")
    print classifier.Posterior(1)
    print classifier.Posterior(0)
    print classifier.Gen_label()
    print classifier.Confusion_matrix("./label_test.csv")
    
    # Read labels from README
    with open('./README') as f:
        f.readline()
        f.readline()
        f.readline()
        lines = f.readlines()
    x_labels = [line.strip() for line in lines]
    
    # Take first 3 misclassified elements
    diff = classifier.y_truth != classifier.y_pred
    i = 0
    idx = []
    for j in range(classifier.testnum):
        if diff[j]:
            idx.append(j)
            i = i + 1
        if (i == 3):
            break;
    classifier.Plot_with_lambda(idx)
    
    # Select 3 test examples with most ambiguous predictions
    p1 = classifier.Posterior(1)
    p2 = classifier.Posterior(0)
    idx = np.argsort(abs(p2 - p1))[:3]
    classifier.Plot_with_lambda(idx)
    

