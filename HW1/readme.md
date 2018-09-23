Using naive bayesian classifier to recognize spam mails
All of the vectors in both training set and testing set have 54 dimensions.
Each element of such a vector representing the frequency of a certain character appeared in the mail
For the generative model of $X$
Suppose $x_{ij} ~ Poisson(\lambda_j)$
And $\vec \lambda ~ Gamma(\alpha, \beta)$
For the label $\vec y$, suppose $y ~ Bern(\pi)$
and $\pi ~ beta(e,f)$
Here we assume $\alpha, \beta, e, f = 1$
