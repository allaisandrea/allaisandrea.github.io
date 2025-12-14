---
layout: default
title: Reinforcement learning
---

# Reinforcement learning

## Problem statement

Reinforcement learning refers to a collection of thechniques for training
synthetic agents in situations where success can be defined easily, but it's
difficult to determine the optimal sequence of actions leading to it.

Examples of such problems are:

* Games like chess, go, Pac-Man etc...: it's easy to determine the winner,
  but difficult to prescribe the winning strategy.

* Legged robot locomotion: it's easy to determine if the robot is making
  progress, but difficult to prescribe the corresponding motion.

The interaction of the agent and its environment is modeled as a Markov decision
process. This is a probabilistic model described by the graphical model below,

<figure>
<img src="graphical_model.svg" alt="Graphical model" style="max-width:5in;border:none"/>
</figure>

where:

* $$\mathbf{s}_t$$ is a random variable representing the state of the
  environment at time $$t$$.

* $$\mathbf{a}_t$$ is a random variable representing the action performed by the
  agent at time $$t$$.

* $$\mathbf{r}_t$$ is a random variable representing a reward that the agent
   receives upon taking an action.

* The agent action $$\textbf{a}_t$$ at time $$t$$ depends on the state of the
  world $$\mathbf{s}_t$$.

* The state of the world $$\mathbf{s}_{t + 1}$$ depends on the previous state
  $$\mathbf{s}_t$$ and on the previous agent action $$\mathbf{a}_t$$.

* The reward $$\mathbf{r}_t$$ depends on the state $$\mathbf{s}_t$$ and the action $$\mathbf{a}_t$$. It may be stochastic, but most
   commonly it is a deterministic function of the state alone.

* The conditional distributions $$(\mathbf{a}_t | \mathbf{s}_t)$$,
  $$(\mathbf{r}_t | \mathbf{s}_t, \mathbf{a}_t)$$, and
  $$(\mathbf{s}_{t + 1} | \mathbf{s}_t, \mathbf{a}_t)$$
  do not depend on time.

The conditional distribution of the agent actions is called _policy_, and we
denote its PDF/PMF by:

<p>\begin{equation}
\pi(a | s) = \mathrm{Prob}\left[\mathbf{a}_t = a | \mathbf{s}_t = s \right]\,.
\end{equation}</p>

The goal of reinforcement learning is to obtain a policy that maximizes the
expected total discounted reward:

<p>\begin{equation}
\eta(\pi) = \expectation{\pi}{\sum_{t = 0}^{\infty} \gamma^t \mathbf{r}_t}\,,
\end{equation}</p>

where $$\gamma \in [0, 1)$$ is a discount factor that reduces the present value
of future rewards.

## Dynamic programming algorithms

If states and actions belong to a finite set, what is called a _finite_ Markov
decision process, a suite of efficient policy-optimization algorithms exists,
belonging to the general class of _dynamic programming_ algorithms.

### Value functions

Dynamic programming algorithms for reinforcement learning are based on _value
functions_, which summarize all the necessary information about future rewards
in a single value. These are the _state-value function_:

<p>\begin{equation}
V_\pi(s) = \expectation{\pi}{\sum_{t = 0}^{\infty} \gamma^t
\mathbf{r}_t \Big| \mathbf{s}_0 = s}\,,
\end{equation}</p>

and the _action-value function_:

<p>\begin{equation}
Q_\pi(s, a) = \expectation{\pi}{\sum_{t = 0}^{\infty} \gamma^t
\mathbf{r}_t \Big| \mathbf{s}_0 = s,\, \mathbf{a}_0 = a}\,.
\end{equation}</p>

The state-value function assigns to each state $$s$$ the expected total
discounted reward that can be obtained by starting from that state and following
policy $$\pi$$. The action-value assign to each pair $$(s, a)$$ the reward
that can be obtained by starting from $$s$$, taking action $$a$$ and then
following the policy $$\pi$$.

Dynamic programming algorithms require a complete tabulation of the state value
function, and are therefore limited in practice to problems with fewer than a
few billion states.

### Policy iteration

Given values for a policy $$\pi_0$$, an improved, deterministic policy $$\pi_1$$
can be constructed by choosing in each state $$s$$ the action $$a$$ that
maximizes the action value of policy $$\pi_0$$:

<p>\begin{equation}
\pi_1(a|s) = \delta_{a, \pi_1(s)}\quad \text{such that}\quad
Q_{\pi_0}(s, \pi_1(s)) = \mathrm{max}_{a}\left[Q_{\pi_0}(s, a)\right]\,.
\end{equation}</p>

The policy $$\pi_1$$ constructed in this fashion constitutes an improvement over
$$\pi_0$$ in that it has value not less than $$\pi_0$$ in every state:

<p>\begin{equation}
V_{\pi_1}(s) \geq V_{\pi_0}(s)\quad \forall\, s\,,
\end{equation}</p>

and consequently the expected total reward $$\eta(\pi_1)$$ is also not less than
$$\eta(\pi_0)$$.

Moreover, the inequality is strict for state $$s$$ if $$\pi_1$$ improves
the action-value of $$\pi_0$$ for that state, that is if:

<p>\begin{equation}
\mathrm{max}_a Q_{\pi_0}(s, a) > Q_{\pi_0}(s, \pi_0(s))\,,
\end{equation}</p>

where we have assumed that $$\pi_0$$ is also deterministic. If $$\pi_0$$ is not
deterministic, the the condition for strict improvement is:
<p>\begin{equation}
\mathrm{max}_a Q_{\pi_0}(s, a) > \expectation{\pi_0}{Q_{\pi_0}(\mathbf{s}_0, \mathbf{a}_0)
\big| \mathbf{s}_0 = s}\,,
\end{equation}</p>
and it is always satisfied unless $$\pi_0$$ already assigns weight zero to all
sub-optimal actions.

Having obtained an improved policy $$\pi_1$$, the process can be repeated, using
the value functions of $$\pi_1$$ to obtain a further improved policy $$\pi_2$$.
The process terminates at policy $$\pi_\star$$ when no further improvement is
possible at any state, because
<p>\begin{equation}
\mathrm{max}_a Q_{\pi_\star}(s, a) = Q_{\pi_\star}(s, \pi_\star(s))\,\quad \forall\, s\,.
\end{equation}</p>

It is often claimed that the terminal policy is the optimal policy, i.e. that it
simultaneously maximizes the value at every state:

<p>\begin{equation}
V_{\pi_\star}(s) = \mathrm{max}_{\pi} V_{\pi}(s)\,,
\end{equation}</p>

however, I have not found a convincing proof yet.

### Policy evaluation {#section-policy-evaluation}

The state-value function necessary for policy iteration can be computed
efficiently as the fixed point of the following recursion relation:

<p>\begin{equation}
V_{\pi,\, k + 1}(s) = \expectation{\pi}{\mathbf{r}_0 +
\gamma V_{\pi,\, k}(\mathbf{s}_1)|\mathbf{s}_0 = s}\,,
\end{equation}</p>

which can be shown to exist and be unique.  Then the action values can be
computed as:

<p>\begin{equation}
Q_\pi(s, a) =  \expectation{}
{\mathbf{r}_0 + \gamma V_\pi(\mathbf{s}_1)\big|\mathbf{s}_0 = s,\, \mathbf{a}_0 = a}\,,
\end{equation}</p>

where the expectation does not depend on the policy because of the conditioning.

### Value iteration

It is not necessary to carry out the policy iteration algorithm explicitly.
Instead, the state value function of the terminal policy $$\pi_\star$$ can be
obtained as the fixed point of the recursion relation:

<p>\begin{equation}
V_{k + 1}(s) = \mathrm{max}_a\, \expectation{}
{\mathbf{r}_0 + \gamma\,
V_{k}(\mathbf{s}_1)\big|\mathbf{s}_0 = s,\, \mathbf{a}_0 = a}\,,
\end{equation}</p>

where the expectation does not depend on the policy because of the conditioning.
The fixed point can be shown to exist and be unique. The action-value function
can be obtained from the state value as before, and the terminal policy
$$\pi_\star$$ is the one that choses the action with maximum value at each
state.

Alternatively, at the cost of tabulating one value for each state-action pair,
instead of just for each state, the action-value function of the terminal policy
can be obtained as the fixed point of the following recursion relation:

<p>\begin{equation}
Q_{k + 1}(s, a) = \expectation{}
{\mathbf{r}_0 + \gamma\, \mathrm{max}_{a'}\, Q_{k}(\mathbf{s}_1, a')
\big|\mathbf{s}_0 = s,\, \mathbf{a}_0 = a}\,,
\end{equation}</p>

where the expectation does not depend on the policy because of the conditioning.



## Proofs

### Bellman equation and policy evaluation

First we prove that the state value function satisfies the following functional
equation, a type of Bellman equation:
<p>\begin{equation}
V_\pi(s) = \expectation{\pi}{\mathrm{r}_0 + \gamma V_\pi(\mathbf{s}_1)\big|\mathbf{s}_0 = s}\,.
\end{equation}</p>

This is because the value functions satisfy the following functional equations,
a type of Bellman equation:
<p>\begin{equation}
Q_\pi(s, a) = r(s, a) + \gamma\, \expectation{\pi}
{Q_\pi(\mathbf{s}_1, \mathrm{a}_1)\big|\mathbf{s}_0 = s,\, \mathbf{a}_0 = a}\,,
\end{equation}</p>
<p>\begin{equation}
V_\pi(s) = \expectation{\pi}{r(\mathrm{s}_0, \mathrm{a}_0) + \gamma V_\pi(\mathbf{s}_1, \mathbf{a}_1)\big|\mathbf{s}_0 = s}\,.
\end{equation}</p>
which are also the fixed point equations of the recursion relation above.
