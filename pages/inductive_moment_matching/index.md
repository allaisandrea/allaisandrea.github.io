---
layout: default
title: Inductinve moment matching
---
# Inductive moment matching

For few step sampling you step from t to s < t. In practice, they go from t to
0, then interpolate back to s. Could you avoid going to zero? I don't know, it
seems necessary to avoiding collapse.

IMM uses the DDIM interpolant, but it's nothing more than the usual formula
x_s = alpha_s x_0 + sigma_s x_1
but reworked to use
x_t = alpha_t x_0 + sigma_t x_1
instead of x_1

For any given sample x_0, x_1, x_0 is not the right regression target for the
model at x_t. You want the starting point of the flow trajectory, which depends
on the whole distribution. This is particularly obvious at t=1, where x_0, x_1
are independent.

You can't use MMD to match the distributions of true and predicted x_0, because
they are too different when the model is learning. You'd need a lot of
particles.

IMM does inductive matching on the distribution at s, rather than point-wise at
t=0. Why is that a better idea?

The inductive matching is between a target network that does t -> 0 -> s and an
online network that does t + dt -> 0 -> s.

Why does the IMM model depend on both t and s? It seems that adding s is
redundant. Check this empirically.
