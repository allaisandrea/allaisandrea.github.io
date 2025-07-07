---
layout: default
title: Consistency models
references:
 - tag: ref_1
   authors: Song <i>et al.</i>
   title: Consistency models
   year: 2023
   url: https://arxiv.org/abs/2303.01469
 - tag: ref_2
   authors: Albergo <i>et al.</i>
   title: "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions"
   year: 2023
   url: https://arxiv.org/abs/2303.08797
---

# Consistency models

## Goal
Consistency models [[1]][ref_1] is a technique for training few-step generative
models without distillation.

The goal is to regress a deterministic function $$f_\theta$$ that maps a
normally distributed random variable $$x_1$$ to a variable $$x_0$$ distributed
like the data:
\begin{equation}
x_1 \sim \normal{0}{I}\,,\quad  x_0 = f_\theta(x_1) \sim \mathrm{Data}\,.
\end{equation}
The PDF of the target distribution is not required, only the ability to sample
from it. For example, the target distribution may be that of images from ImageNet.

## Stochastic interpolant

The technique leverages a stochastic interpolant [[2]][ref_2]  $$x_t$$ that
bridges between the target distribution and the noise distribution:
\begin{equation}
x_t|x_0, x_1 = \alpha_t x_0 + \sigma_t x_1\,,\quad t \in [0, 1]\,,
\end{equation}
where $$\alpha$$ and $$\sigma$$ are interpolating functions such that $$\alpha_0
= \sigma_1 = 1$$, $$\alpha_1 = \sigma_0 = 0$$. For simplicity we will assume a
linear form: $$\alpha_t = 1 - t$$, $$\sigma_t = t$$, although all the arguments
can be generalized to the non-linear case.

The marginal distribution of $$x_t$$ interpolates between the target
distribution and the noise distribution. This is illustrated in the figure below
for a simple one-dimensional random variable $$x_t$$.
<figure>
<img src="probability_flow.png"
     alt="Example of proability flow reverse process"
     style="max-width:7in"/>
</figure>

Incidentally, the marginal distribution of $$x_t$$ matches the marginal
distribution of a certain diffusion process associated with $$\alpha$$ and
$$\sigma$$, but this fact is not relevant for consistency models. For more
information see the [diffusion page](/pages/diffusion#section-forward-process).

## Probability flow

It is possible to construct a deterministic function $$h: (t, x_0) \mapsto
h_t(x_0)$$ that maps the data random variable $$x_0$$ to the interpolant random
variable
$$x_t$$:
\begin{equation}
x_0 \sim \mathrm{Data}\quad \Leftrightarrow\quad h_t(x_0) \sim x_t.
\end{equation}
For any fixed $$t$$, the map is invertible, and its inverse maps samples from
$$x_t$$ to samples from the data distribution. In particular, the inverse of
$$h_1$$ is a generative function that maps pure noise to the data distribution.

The diagram above displays in white the trajectories of a few points under the map $$h$$.

The function $$h$$ satisfies the <i>probability flow</i> ODE. This fact is
illustrated in greater detail on the [diffusion
page](/pages/diffusion#section-probability-flow). When $$\alpha$$ and $$\sigma$$
are linear, it has a particularly simple form:
\begin{equation}
    \frac{\dd h_t}{\dd t} = \expectation{x_0, x_1 | x_t = h_t}{x_1 - x_0}\,,
\end{equation}
where the distribution $$x_0, x_1 | x_t$$ is obtained applying Bayes' rule to
the definition of the interpolant $$x_t$$.

Geometrically, the tangent to a flow trajectory at a point $$x_t$$ is the
average slope of all the lines connecting $$x_0\sim\mathrm{Data}$$ and
$$x_1\sim\normal{0}{I}$$ that pass through $$x_t$$. This is illustrated in the
figure below.
<figure>
<img src="/pages/diffusion/slope_distribution_1.svg"
     alt="Illustrate the relationship between the probability flow and the slope distribution"
     style="max-width:7in" id="img-slope-distribution"/>
</figure>

## Inductive learning

The training of consistency models uses a loose form of mathematical induction:
assume that a solution exists for interpolation time $$t$$, and use it to
construct a solution for time $$t + \Delta t$$.

<figure>
<img src="consistency_models_integration.svg"
     alt="One step of the inductive process"
     style="max-width:7in"/>
</figure>

With reference to the diagram above, let $$x_0 \sim \mathrm{Data}$$, $$x_1 \sim
\normal{0}{I}$$. When $$\alpha$$ and $$\sigma$$ are linear, the interpolant
$$x_t|x_0, x_1$$ lies on the line from $$x_0$$ to $$x_1$$.

Assume that a <i>target network</i> $$F$$ has been obtained that can predict the
starting point of any flow trajectory given its value $$x_t$$ at some fixed time
$$t$$. If we can construct a point <i>on the same trajectory</i> at later time
$$t + \Delta t$$, we can use the output of the target network as a regression
target for an <i>online network</i>  $$f_\theta$$ that predicts the the starting
point given $$x_{t + \Delta t}$$.

The two points $$x_t$$ and $$x_{t + \Delta t}$$ can be constructed to lie on the
same trajectory in two ways:

1. If the two points $$x_t$$ and $$x_{t + \Delta t}$$ are chosen to lie on the same
line from $$x_0$$ to $$x_1$$, they are not on the same trajectory, but they are
<i>almost</i> on the same trajectory. More precisely, in expectation, the
difference between the starting points $$h^{-1}(x_t)$$ and $$h^{-1}(x_{t +
\Delta t})$$ is of order $$\Delta t^2$$. This approach is dubbed training <i>in isolation</i>.

2. Alternatively, a separate model can be trained to predict the tangent to the
trajectory. The training objective for this model is essentially [denoising
score matching](/pages/diffusion#section-denoising-score-matching), and
consistency models paired with this approach can be seen as distilling a
standard diffusion model into a more efficent few-step sampler. The error
involved in this approach is also of order $$\Delta t^2$$.

In both cases, since the solution at $$t = 1$$ is obtained by repeating
induction $$1 / \Delta t$$ times, the final error is of order $$\Delta t$$, and
can be made arbitrarily small.


In practice, all the induction steps are trained simultaneously, with a single
model $$f_\theta(t, x_t)$$ serving as online network for all times, and an
exponential moving average $$f_{\theta^{-}}$$ of the same serving as target
network. Explicitly, when trainining in isolation, the loss is:
\begin{equation}
L(\theta) = \expectation{t, x_0, x_1}{\norm{f_{\theta}(t + \Delta t, x_t + (x_1 - x_0) \Delta t) - f_{\theta^{-}}(t, x_t)}^2}\,.
\end{equation}


## Proofs

<p>
\begin{equation}
L(\theta) = \expectation{x_t}{\norm{f_{\theta}(x_{t + \Delta t}) - F(x_t)}^2}\,.
\end{equation}
</p>

In principle, the point $$x_{t + \Delta t}$$ on the trajectory through $$x_t$$
can be obtained by Euler integration of the flow matching ODE:
\begin{equation}
     x_{t + \Delta t} = x_t + \expectation{x_0, x_1|x_t}{x_1 - x_0}\ \Delta t + O(\Delta t^2)\,.
\end{equation}
However, a naive application of this idea leads to intractable nested
expectation values in the loss. This difficulty can be circumvented in two ways.


A second way is to move the inner expectation outwards. Assuming the model is
twice differentiable, the error incurred in doing so is of order $$\Delta t^2$$,
the same as the discretization error in Euler integration:
\begin{equation}
f_{\theta}(x_t + \expectation{}{x_1 - x_0}\ \Delta t) =
\expectation{}{f_{\theta}(x_t + (x_1 - x_0)\ \Delta t)} + O(\Delta t^2)\,.
\end{equation}

With this transformation, the two expectations can be combined, yielding a
tractable loss:
<p>
\begin{equation}
L(\theta) = \expectation{x_0, x_1}{\norm{f_{\theta}(x_t + (x_1 - x_0)\Delta t) - F(x_t)}^2}\,.
\end{equation}
</p>



{% include references.md %}
