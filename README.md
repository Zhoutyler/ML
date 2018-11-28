## /HW1 Using naive bayesian classifier to recognize spam mails

All of the vectors in both training set and testing set have 54 dimensions.

Each element of such a vector representing the frequency of a certain character appeared in the mail

For the generative model of $ X $

Suppose $ x_{ij} ~ Poisson(\lambda_j) $

And $ \vec \lambda ~ Gamma(\alpha, \beta) $

For the label $ \vec y$, suppose $y ~ Bern(\pi) $

and $ \pi ~ beta(e,f) $

Here we assume $ \alpha, \beta, e, f = 1 $

## /HW4 ML-EM, VI, Marginalized Gibbs Sampler for Binomial Mixture Model

2000-dimensional data in range [0, 20]

Model as generating from binomial mixture model

![alt text](https://github.com/Zhoutyler/ML/blob/master/HW4/description_for_readme.png)

Using Maximum likelihood-EM to get MAP estimate for model variables pi, theta.

Using Variational inference to get parameters of full posterior of model variables.

Using Marginalized gibbs sampling to get cluster distribution for each data points in infinite cluster number context.
