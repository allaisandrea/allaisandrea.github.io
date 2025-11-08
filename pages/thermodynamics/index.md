---
layout: default
title: Thermodynamics
---

# Microcanonical ensemble

Argument:
1. The dynamics of the system induce a distribution on the phase space, by
   visitation frequency.
2. This distribution is stationary
3. By Liouville's theorem, the distribution is constant along the trajectories
   of the system
4. You get a constant PDF on every reachable point.

A subtlety to iron out: The delta functions induced by conserved quantities.

Relatedly, integrable systems divide into multiple mutually unreachable parts,
because of the many conserved quantities.

By definition, time averages are equal to statistical averages.

Make subgroups, for example by number of particles on each side of a box. The
entropy of a subgroup is the log of its size. This is a special case of
Shannon's entropy. It's logarithmic so it's extensive.

The size of a subgroup is the phase space volume.

The largest subgroups dominate the probability distribution.

To first order, entropy is maximized, Then there are fluctations.

Fluctuations are small because the distribution of macroscopic quantities is
narrow, so a few macrostates take up most of the probabilty. This is because
macroscopic quantities are extensive, locally independent, and so the central
limit applies. If you consider a volume of phase space that contains only one
particle on average, and group by the number of particles in the volume, the
fluctuations will be big.

Because of time-reversal symmetry, entropy most likely decreases both forward
and backward in time. Which means that most points of lower entropy are minima,
instead of on the path towards higher or lower entropy. This means that the
time series of entropy is fractal.

The universe is not at maximum entropy because it's not done yet.

# Temperature

Allow two system to exchange energy -> same dS/dE. Then dS/dE is the
temperature.

Energy flows from high to low temperature.

For classical systems temperature must be positive or they will blow up to
maximize entropy.

# Pressure

In a piston scenario, the mechanical work done on the system is $$-p \dd V$$.
Assuming an adiabatic process means that this is equal to the change in
internal energy: there can be no heat transfer.

The maximum entropy principle then is
\begin{equation}
0 = dS = \frac{\partial S}{\partial E} (-p \dd V) + \frac{\partial{S}}{\partial V} \dd V
\end{equation}
which implies
\begin{equation}
p = T \left.\frac{\partial{S}}{\partial V}\right|_E
\end{equation}

From this you can also obtain
\begin{equation}
p = \left.\frac{\partial E}{\partial V}\right|_S
\end{equation}


# Canonical ensemble

The canonical ensemble is a different distribution over states.

If you assume the microcanonical distribution for system + thermostat, then the
system is distributed canonically.

# More topics

* The arrow of time
* Relationship to Clausius theorem
* How do you actually measure things experimentally
* Systems with negative absolute temperature
* Legendre transform
* Chemical potential, pressure, magnetization
* Jarzinsky equality
* Fluctuation dissipation
* Semiclassical quantum statistics

# Models

* Perfect gas
* Polymer chain
* Spring lattice
* Ising model
* Electromagnetic field




