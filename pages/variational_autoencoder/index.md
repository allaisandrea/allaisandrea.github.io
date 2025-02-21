---
layout: default
title: Auto-encoding variational Bayes
---

# Auto-encoding variational Bayes

## Model

We describe a technique for carrying out inference on directed probabilistic
models with continuous latent variables, as introduced in [[1]][ref_1]. That
is, the model is defined by a joint probability distribution $$p$$,
parametrized by $$\theta$$, over observed variables $$x$$ and unobserved,
continuous variables $$z$$. The model is directed, meaning that both the latent
prior $$p(z|\theta)$$ and the latent-conditional likelihood $$p(x | z,
\theta)$$ can be efficiently evaluated, as well as sampled from.

For concreteness, [[1]][ref_1] presents an example where the observed variables
$$x \in \{0, 1\}^M$$ are the pixel values of a MNIST image. $$M = 28 \times
28$$ is the number of pixels in the image. The latent variables $$z \in
\mathbb{R}^D$$ encode, among other things, the corresponding digit. The latent
prior is the standard normal: $$z \sim \normal{0}{I}$$, and the
latent-conditional likelihood is a product of Bernoulli distributions:

\begin{equation}
p(x|z) = \prod_{a = 1}^{M} \left[\pi_a(z, \theta)\right]^{x_a}
\left[1 - \pi_a(z, \theta)\right]^{1 - x_a}\,.
\end{equation}

The vector of Bernoulli parameters $$\pi(z, \theta) \in [0, 1]^M$$ is specified
by a multi-layer perceptron with inputs $$z$$ and parameters $$\theta$$.

## Motivation

Given an i.i.d. sample $$\{x_i\}_{i = 1}^{N}$$ of the observed variable, the
technique yields an approximation to the maximum likelihood parameters
$$\theta_{\star}$$, as well as an approximation to the latent posterior $$p(z |
x, \theta_\star)$$ that can be efficiently evaluated and sampled from.

There are two main motivations to consider this type of model and inference:

1. Generative modeling. Given an approximation to $$\theta_\star$$, more
samples of $$(x, z)$$ can be generated efficiently according to the model.
These samples will be similar to those drawn from the empirical distribution of
$$x$$, to the extent that it can be correctly captured by the model.

2. Representation learning. The latent posterior $$p(z | x, \theta_\star)$$
associates a distribution in latent space to each value of $$x$$. This
distribution, and its mean in particular, retains information about $$x$$ in a
compressed form that is tuned to the empirical distribution of $$x$$. From this
point of view, the approximate latent posterior can be seen as an *encoder*,
and the latent-conditional likelihood as a *decoder*.

## Variational bayes

Naive maximum likelihood inference on this type of models is difficult, because
the latent-marginal likelihood

\begin{equation}
p(x|\theta) = \int \dd z\ p(x, z|\theta)
\end{equation}

is not tractable. Given any set of parameters $$\theta$$, it is possible to
efficiently *sample* $$x$$ according to the model, but it is not possible to
efficiently _evaluate_ $$p(x|\theta)$$ at a given value $$x = x_i$$ drawn from
the empirical distribution. The integral domain is high-dimensional, and the
integrand is a complicated, strongly-peaked function.


## References

[ref_1]: https://arxiv.org/abs/1312.6114 "Auto-encoding variational Bayes"
1. Kingma, Welling. [Auto-encoding variational Bayes][ref_1]. (2013)
