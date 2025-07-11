---
layout: default
title: Diffusion
references:
  - tag: anderson_1982
    authors: Anderson
    title: Reverse-time diffusion equation models
    year: 1982
    notes: >
      First derivation of the stochastic differential equation that describes
      the reverse diffusion process.
  - tag: vincent_2011
    authors: Vincent
    title: A Connection Between Score Matching and Denoising Autoencoders.
    year: 2011
  - tag: sohl_dickstein_2015
    authors: Sohl-Dickstein <i>et al.</i>
    title: Deep Unsupervised Learning using Nonequilibrium Thermodynamics
    year: 2015
    notes: >
      First time learning to reverse the diffusion process. It does not use the
      language of SDEs and is not aware of the connection to score matching,
      but the substance is the same as what is described here.
  - tag: song_2020
    authors: Song, Ermon
    title: Generative Modeling by Estimating Gradients of the Data Distribution
    year: 2020
    notes: >
      First time the lanuage of SDEs is applied to the problem, and the
      connection to score matching is exploited. Interesting discussion on how
      the noise in score matching improves learning stability and the coverage
      of multiple distribution modes.
  - tag: ho_2020
    authors: Ho <i>et al.</i>
    title: Denoising Diffusion Probabilistic Models
    year: 2020
    notes: >
      Contemporanous to the previous paper, and makes roughly the
      same points.
  - tag: song_2021
    authors: Song <i>et al.</i>
    title: Score-Based Generative Modeling through Stochastic Differential Equations
    year: 2021
    notes: >
      First time the probability flow ODE is introduced.
  - tag: lipmap_2023
    authors: Lipman <i>et al.</i>
    title: Flow Matching for Generative Modeling
    year: 2023
    notes: >
      This paper presents a fairly original derivation of the probability flow
      ODE and the score-matching objective. The paper is a bit ambiguous on the
      extent to which their method differs from previous literature. The method
      is in fact entirely equivalent to chosing an affine form for \(\alpha_t\)
      and \(\sigma_t\). This choice differs from most of the literature, and they
      claim it yields better FID on ImageNet. The training objective is also
      essentially equivalent to denoising score matching, except that it includes
      the entire RHS of the flow ODE, instead of just the score. This choice of
      schedule and objective supposedly makes for an easier optimization.
open_graph:
  image: pages/diffusion/diffusion.png
  description: >
    A primer on generative modeling using diffusion.
script: script.js
---

# Diffusion

## Forward process {#section-forward-process}

The goal of generative modeling is to map a random variable with a known and
simple distribution (<i>e.g.</i> the normal distribution), to a random
variable distributed like the data:
\begin{equation}
    x \sim \normal{0}{I}\,; \quad y = f(x) \sim \mathrm{Data}\,.
\end{equation}

The function $$f$$ is to be learned from a sample from the data distribution.  At
this level it is not at all clear how a learning objective may be specifed for
$$f$$. One approach is learning to reverse a diffusion process in which small
amounts of noise are progressively added to the data variable $$y$$ until it
becomes essentially pure noise.

The diffusion process is described at a high level by the graphical model:

<figure>
<img src="forward_process.png" alt="Forward diffusion graphical model" style="max-width:4.58in"/>
</figure>

and in detail by the recurrence relation:
\begin{equation}
    x_{t + \Delta t} = \left(1 - \lambda_t \Delta t\right) x_t + g_t \sqrt{\Delta t}\, z_{t}\,,
\end{equation}
where:

1.  $$\{x_t\}$$ is a set of random variables, one for each value of $$t$$. The
    first one, $$x_0$$, is distributed like the data, and the following are
    progressively noisier.

1.  $$z_t \sim \normal{0}{I}$$ is the noise variable that is added at each step.
    $$z_t$$ is independent from $$z_s$$ for all $$t \neq s$$, and is also independent
    from $$x_s$$ for $$s \leq t$$.

1.  $$\lambda$$ and $$g$$ are function parameters that control the mixing of
    signal and noise at each step of the diffusion process.

1.  $$\Delta t$$ is a discretization step. In the limit $$\Delta t \to 0$$ the
    recurrence relation becomes a stochastic differential equation. Results
    below are given in this limit.

The recurrence relation is linear in the random variables $$x_{t}$$ and
$$z_{t}$$. Consequently, the joint distribution of any set of variables
$$(x_{t_1}, x_{t_2}, \ldots)$$ conditional on  $$x_0$$ is normal, and, in
particular:
\begin{equation}
     x_t|x_0 \sim \normal{\alpha_t x_0}{\sigma_t^2 I}\,,
\end{equation}
where the functions $$\alpha$$ and $$\sigma$$ uniquely determine the recurrence
relation parameters $$\lambda$$ and $$g$$:

$$
\begin{align}
&\lambda_t = -\frac{\dd}{\dd t}\log \alpha_t\,,\\
&g^2_t = \sigma^2_t \frac{\dd}{\dd t} \log \frac{\sigma^2_t}{\alpha^2_t}\,.
\end{align}
$$

Importantly, the recurrence relation can achieve *any* profile of $$\alpha$$
and $$\sigma$$, as long as as the signal-to-noise ratio $$\alpha / \sigma$$ is
monotonically decreasing (so that $$g^2$$ above is a positive number).


The explicit expression for the distribution of $$x_t|x_0$$ bypasses the
recurrence relation, and enables efficient sampling of $$x_t$$, both
conditionally on $$x_0$$ and unconditionally.

If, at some time $$T > 0$$, $$\alpha_T = 0$$ and $$\sigma_T=1$$, then $$x_T$$ follows
a standard normal distribution, and is the noise variable of known distribution
from which we want to learn to generate new samples of $$x_0$$.

Below is an illustrative one-dimensional example of the diffusion process. On
the left is the <q>data</q> distribution, which we take to be a mixture of two
narrow normal distributions. On the right is the final distribution
$$\normal{0}{I}$$. In the middle is a contour plot of the PDF of $$x_t$$, and
overlayed as white lines are 16 samples from the process.

<figure>
<img src="diffusion.png" alt="One-dimensional example of diffusion process" style="max-width:7in"/>
</figure>

## Reverse process

Remarkably, the joint distribution of $$x_t$$ specified by the diffusion process
can also be obtained in reverse, <i>i.e.</i> according to the graphical
model:
<figure>
<img src="backward_process.png" alt="Backward diffusion graphical model" style="max-width:4.58in"/>
</figure>
through the recurrence relation:
\begin{equation}
    x_{t - \Delta t} = \left(1 + \lambda_t \Delta t \right) x_t + g^2_t \Delta t \nabla \log p(x_t, t) + g_t \sqrt{\Delta t}\, \bar{z}_t
\end{equation}
where:

1.  \$$x_T \sim \normal{0}{I}$$

1.  $$\bar{z}_t \sim \normal{0}{I}$$. $$\bar{z}_t$$ is independent from
    $$\bar{z}_s$$ for all $$t \neq s$$, and also from $$x_s$$ for $$s \geq t$$.

1.  $$p(x,t)$$ is the marginal PDF of $$x_t$$, <i>i.e.</i>
    $$\mathrm{Pr}[x_t\in D] = \int_D p(x, t) \dd x$$ for any domain
    $$D$$.

The joint distribution of $$x_t$$ generated by this recurrence relation is the
same as the one generated by the forward diffusion process. In particular,
$$x_0 \sim \mathrm{Data}$$, and the figure in the previous section is also an
entirely valid illustration of this reverse process.

The recurrence relation is not linear. This is necessary in order to generate a
non-normal distribution for $$x_0$$ from normally distributed $$x_T$$ and
$$\{z_t\}$$. Therefore, unlike the forward process, there is no way to sample
the process more efficiently than by using the recurrence relation.

<figure>
<img src="score_function.png" alt="Score function illustration" style="max-width:5in"/>
</figure>

The relation depends on the so-called <q>score function</q> $$\nabla \log p(x, t)$$.
This is a vector field that points everywhere towards regions of high
probability, and drives the reverse diffusion process there.  The practical use
of reverse diffusion to sample from the data distribution depends on the
ability to evaluate the score function efficiently.

## Denoising score matching {#section-denoising-score-matching}

The score function can be estimated efficiently thanks to the identity
\begin{equation}
\nabla \log p(x_t, t) = \expectation{x_0|x_t}{\nabla \log p(x_t|x_0, t)}\,.
\end{equation}
Unlike the marginal PDF $$p(x_t, t)$$, the conditional
PDF $$p(x_t|x_0, t)$$ is known in closed form, because $$x_t|x_0 \sim
\normal{\alpha_t x_0}{\sigma_t^2 I}$$:
\begin{equation}
\nabla \log p(x_t | x_0, t) = \frac{\alpha_t x_0 - x_t}{\sigma^2_t}\,.
\end{equation}

This identity can be leveraged to construct an objective for a
parametric approximation $$s_\theta(x, t)$$ to the score function $$\nabla \log
p(x, t)$$:
\begin{equation}
    L(\theta, t) = \expectation{x_0, x_t}{\left\Vert s_{\theta}(x_t, t) - \nabla \log p(x_t | x_0, t)\right\Vert^2}\,.
\end{equation}
The unconstrained optimum of this objective is the score function $$\nabla
p(x_t, t)$$.

Explicitly, the objective reads:
\begin{equation}
    L(\theta, t) = \expectation{x_0, x_t}{\left\Vert s_{\theta}(x_t, t) -
    \frac{\alpha_tx_0 - x_t}{\sigma_t^2}\right\Vert^2}\,,
\end{equation}
and an be interpreted as the objective for a denoising
autoencoder:

1.  Sample $$x_0$$ from the data distribution.

1.  Mix it with Gaussian noise $$z$$ to obtain $$x_t = \alpha_t\, x_0 +
    \sigma_t z$$.

1.  Regress the added noise $$z / \sigma_t$$ from the noisy
    variable $$x_t$$ to obtain $$s_\theta(x, t)$$.

## Probability flow {#section-probability-flow}

It is also possible to construct an ordinary differential equation that
reproduces the marginal distribution of $$x_t$$ according to the diffusion
process, but not the joint distribution of multiple variables $$(x_{t_0},
x_{t_1}, ...)$$:
\begin{equation}
    \frac{\dd x_t}{\dd t} = - \lambda_t x_t - \frac{1}{2}g^2_t \nabla \log p(x_t, t)\,.
\end{equation}

This ODE is known as the <q>probability flow</q> ODE, and it yields a
deterministic, invertible map between $$x_T \sim \normal{0}{I}$$ and $$x_0 \sim
\mathrm{Data}$$.  The trajectories generated by this ODE are illustrated in the
figure below:

<figure>
<img src="probability_flow.png" alt="Example of proability flow reverse process" style="max-width:7in"/>
</figure>

The probability flow ODE takes a particularly simple form when the schedules are
linear: $$\alpha_t = 1 - t$$, $$\sigma_t = t$$. In this case, substituting the
score matching identity yields:
\begin{equation}
    \frac{\dd x_t}{\dd t} = \expectation{x_0, x_1 | x_t}{x_1 - x_0}\,,
\end{equation}
where the expectation is with respect to $$x_0, x_1 | x_t$$, as can be obtained with Bayes' rule from:
\begin{equation}
x_0 \sim \mathrm{Data}\,,\quad x_1 \sim \normal{0}{I}\,,\quad x_t|x_0, x_1 = (1 - t)\, x_0 + t\, x_1\,.
\end{equation}

Geometrically, the RHS of the ODE is the average slope of all the lines
connecting $$x_0\sim\mathrm{Data}$$ and $$x_1\sim\normal{0}{I}$$ that pass
through $$x_t$$, as illustrated in the figure below.
<figure>
<img src="slope_distribution_0.svg" alt="Illustrate the relationship between the probability flow and the slope distribution" style="max-width:7in" id="img-slope-distribution"/>
</figure>


Using the probability flow ODE to sample from the reverse process is more
efficient than the SDE, and is amenable to distillation. However, it usually
yields samples of lower quality as measured by FID.

## Proofs

### Score function expectation identity
<p>
\begin{equation}
\begin{split}
\nabla \log p(x_t)
&= \frac{1}{p(x_t)} \nabla p(x_t) \\
&= \frac{1}{p(x_t)} \int p(x_0) \nabla p(x_t | x_0)\ \dd x_0 \\
&= \frac{1}{p(x_t)} \int \frac{p(x_t) p(x_0 | x_t)}{p(x_t | x_0)} \nabla p(x_t | x_0)\ \dd x_0 \\
&= \int p(x_0 | x_t)\nabla \log p(x_t | x_0)\ \dd x_0 \\
&= \expectation{x_0|x_t}{\nabla \log p(x_t | x_0)}
\end{split}
\end{equation}
</p>

### Score matching loss optimum {#section-proof-unconstrained-optimum}


The proposition
\begin{equation}
s_{\mathrm{opt}}(x_t, t) = \expectation{x_0|x_t}{\nabla \log p(x_t|x_0, t)}
\end{equation}
is a special case of the following proposition.

The optimum of the functional:
\begin{equation}
L[f] = \expectation{x, y}{\left \Vert f(x) - g(x, y)\right\Vert^2}
\end{equation}
is:
\begin{equation}
f_{\mathrm{opt}}(x) =  \expectation{y|x}{g(x, y)}
\end{equation}

Proof:
<p>
\begin{equation}
\begin{split}
\delta L[f]
&= 2 \expectation{x, y}{\delta f(x) \cdot \left(f(x) - g(x, y)\right) } \\
&= 2 \expectation{x}{\expectation{y|x}{\delta f(x)\cdot\left(f(x) - g(x, y)\right)}} \\
&= 2 \expectation{x}{\delta f(x) \cdot \expectation{y|x}{\left(f(x) - g(x, y)\right)}} \\
&= 2 \expectation{x}{\delta f(x) \cdot \left(f(x) - \expectation{y|x}{g(x, y)}\right)}\,.
\end{split}
\end{equation}
</p>

The optimum has zero variation for any perturbation $$\delta f$$, which yields
the theorem.



{% include references.md %}
