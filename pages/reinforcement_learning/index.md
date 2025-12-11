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
<img src="graphical_model.svg" alt="Graphical model" style="max-width:5in"/>
</figure>

where:

* $$\mathbf{s}_t$$ is a random variable representing the state of the
  environment at time $$t$$,

* $$\mathbf{a}_t$$ is a random variable representing the action performed by the
  agent at time $$t$$,

* the agent action $$\textbf{a}_t$$ at time $$t$$ depends on the state of the
  world $$\mathbf{s}_t$$,

* the state of the world $$\mathbf{s}_{t + 1}$$ depends on the previous state
  $$\mathbf{s}_t$$ and on the previous agent action $$\mathbf{a}_t$$.

The conditional distribution of the agent actions is called _policy_, and we
denote its PDF/PMF by:

<p>\begin{equation}
\pi(a | s) = \mathrm{Prob}\left[\mathbf{a}_t = a | \mathbf{s}_t = s \right]\,.
\end{equation}</p>

We denote the PDF/PMF of the conditional distribution of the environment state by:

<p>\begin{equation}
p(s' | s, a) = \mathrm{Prob}\left[\mathbf{s}_{t + 1} = s' | \mathbf{s}_t = s, \mathbf{a}_t = a \right]\,.
\end{equation}</p>

It is assumed that neither depends on the time $$t$$.

Upon taking action $$a$$ from state $$s$$, the agent receives a reward
$$r(s, a)$$. The reward may be stochastic, in which case $$r(s, a)$$ stands for
its conditional expected value. Most commonly, the reward is a deterministic
function of the state.

The goal of reinforcement learning is to obtain a policy that maximizes the
expected total discounted reward:

<p>\begin{equation}
\eta(\pi) = \expectation{\pi}{\sum_{t = 0}^{\infty} \gamma^t
r(\mathbf{s}_t, \mathbf{a}_t)}\,,
\end{equation}</p>

where $$\gamma \in [0, 1)$$ is a discount factor that reduces the present value
of rewards in the far future.

## Policy iteration algorithm

If states and actions belong to a finite set, what is called a _finite_ Markov
decision process, an efficient algorithm for optimizing the policy is _policy
iteration_. The algorithm is based on the _state-value function_:

<p>\begin{equation}
V_\pi(s) = \expectation{\pi}{\sum_{t = 0}^{\infty} \gamma^t
r(\mathbf{s}_t, \mathbf{a}_t) \Big| \mathbf{s}_0 = s}\,,
\end{equation}</p>

and the _action-value function_:

<p>\begin{equation}
Q_\pi(s, a) = \expectation{\pi}{\sum_{t = 0}^{\infty} \gamma^t
r(\mathbf{s}_t, \mathbf{a}_t) \Big|
\mathbf{s}_0 = s,\, \mathbf{a}_0 = a}\,.
\end{equation}</p>

The state-value function assigns to each state $$s$$ the expected total
discounted reward that can be obtained by starting from that state and following
policy $$\pi$$. The action-value assign to each pair $$(s, a)$$ the reward
that can be obtained by starting from $$s$$, taking action $$a$$ and then
following the policy $$\pi$$.

The two are related by:
<p>\begin{equation}
Q_\pi(s, a) = r(s, a) + \gamma\, \expectation{}
{V_\pi(\mathbf{s}_1)\big|\mathbf{s}_0 = s,\, \mathbf{a}_0 = a}\,,\\
\end{equation}</p>
<p>\begin{equation}
V_\pi(s) = \expectation{\pi}{Q_\pi(\mathbf{s}_0, \mathbf{a}_0)\big|\mathbf{s}_0 = s}\,,
\end{equation}</p>
where the first expectation does not involve the policy because of the conditioning.

The optimization algorithm consists of the repeated application of two stages.
The first stage, called _policy evaluation_ estimates the value $$V_\pi(s)$$ of
each state, and the second stage, called _policy improvement_ uses the estimated
values to generate a new policy that is better than the previous one.

### Policy evaluation

The state-value function satisfies the following functional equation, a type of
Bellman equation:
<p>\begin{equation}
V_\pi(s) = \expectation{\pi}{r(\mathbf{s}_0, \mathbf{a}_0) +
\gamma V_{\pi}(\mathbf{s}_1)|\mathbf{s}_0 = s}\,.
\end{equation}</p>

The Bellman equation is the fixed point equation for the following recursion
relation:
<p>\begin{equation}
V_{\pi,\, k + 1}(s) = \expectation{\pi}{r(\mathbf{s}_0, \mathbf{a}_0) +
\gamma V_{\pi,\, k}(\mathbf{s}_1)|\mathbf{s}_0 = s}\,.
\end{equation}</p>

For a finite Markov decision process with $$\gamma \in [0, 1)$$, the Banach
fixed point theorem applies, and the recursion relation has a unique fixed
point. Thus the value function can be computed by iterating the recursion
relation until convergence. Of course this requires to tabulate the value for
each possible state, which for moderately complex problems may be infeasible.

The action-value function can be computed from the value function, or,
alternatively, by recursion from the Bellman equation:
<p>\begin{equation}
Q_\pi(s, a) = r(s, a) + \gamma\, \expectation{\pi}{
Q_{\pi}(\mathbf{s}_1, \mathbf{a}_1)|\mathbf{s}_0 = s, \mathbf{a}_0 = a}\,.
\end{equation}</p>


### Policy improvement

The policy improvement stage operates on deterministic policies $$\pi(a|s) =
\delta_{a, \pi(s)}$$.

Given policy $$\pi_{k}$$, the improved policy $$\pi_{k + 1}$$ is defined as the
policy that choses in each state $$s$$ the action $$a$$ that maximizes the
action value of policy $$\pi_{k}$$:

<p>\begin{equation}
\pi_{k + 1}(s) = \mathrm{argmax}_{a}\left[Q_{\pi_k(s, a)}\right]\,.
\end{equation}</p>

The policy constructed in this fashion is guaranteed to have value not less than
$$\pi_k$$ in every state:

<p>\begin{equation}
V_{\pi_{k + 1}}(s) \geq V_{\pi_k}(s)\quad \forall\, s\,.
\end{equation}</p>

Moreover, the inequality is strict for state $$s$$ if $$\pi_{k + 1}$$ improves
the action-value function for that state, that is if:

<p>\begin{equation}
Q_{\pi_k}(s, \pi_{k + 1}(s)) = \mathrm{max}_a Q_{\pi_k}(s, a) > Q_{\pi_k}(s, \pi_k(s))\,.
\end{equation}</p>

If policy $$\pi_{k + 1}$$ has the same action-value as $$\pi_k$$ for all states,
the policy improvement algorithm has converged.

### Optimality equation


<p>\begin{equation}
V_{\pi_\star}(s) = \mathrm{max}_a Q_{\pi_\star}(s, a)
\end{equation}</p>

