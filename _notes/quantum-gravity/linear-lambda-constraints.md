---
title: "Linear global constraints on the cosmological constant: sign and drift"
date: 2026-10-29
category: dark-sector
order: 1
description: In a flat universe of dust and radiation, a global constraint that is linear in the stress tensor gives a non-positive vacuum energy when its weight is built from causal vectors or from sequestering multipliers, and weights whose energy component and trace have the same sign give a non-negative value that does not increase along an observer's worldline.
status: Sign theorem proved for spatially flat FRW matter with pressure between zero and one third of the density, and for causal-vector weights in any spacetime obeying the energy conditions; drift theorem proved for a single weight with constant ratio of energy component to trace, a separable measure on the past light cone and a log-concave kernel; one explicit non-linear counterexample; numbers are calculations on a LambdaCDM background; the Lagrangian-linear section is a derivation under stated model assumptions with a heuristic stability argument.
excludes: A positive cosmological constant from a linear global constraint whose weight is built from causal vectors or from sequestering multipliers, for matter with pressure between zero and one third of the density over the whole region. Weights with a positive energy component relative to their trace, matter outside that range, and constraints that are not linear in the stress tensor are not excluded.
---

A family of proposals removes the vacuum energy from the gravitational field equations by a global condition: a weighted spacetime average of the stress tensor, with the vacuum included, must vanish. The weight is a symmetric tensor field $$W^{ab}$$. The sequestering constraint of Kaloper and Padilla uses the metric itself as the weight and gives $$\rho_\Lambda=\tfrac14\langle3p-\rho\rangle$$, which is $$-\rho/4$$ for dust and $$0$$ for radiation. Which values of the net vacuum energy density $$\rho_\Lambda$$ can a weight give if it is an arbitrary tensor field, linear in the stress tensor? In a spatially flat Friedmann–Robertson–Walker (FRW) universe with pressure $$0\le p\le\rho/3$$, a weight acts only through its energy component $$A$$ in the matter rest frame and its trace $$B$$, and $$\rho_\Lambda\le0$$ whenever $$A/B\le-\tfrac14$$ (Theorem 1). Weights built from causal vectors satisfy this (Theorem 2), and so do sequestering multipliers for any number of matter sectors (Proposition 3).

A positive $$\rho_\Lambda$$ needs $$A/B>-\tfrac14$$ on a set of positive measure; the examples are weights defined by a frame, such as the spatial projector of the matter rest frame. When the weight is tied to the past light cone of an observer and $$A/B\ge0$$ is constant, the value assigned to the observer is non-negative and does not increase along the worldline, so that read as a dark-energy density it has $$w_{\rm eff}\ge-1$$ and cannot cross $$-1$$; for constant $$A/B\le-\tfrac14$$ the value is negative and rises towards zero (Theorem 3). In sequestering the average runs over the whole spacetime, including any final collapse. Theorems 1 and 2 hold for any region, and Theorem 3 concerns the past light cone.

The trace weight is the sequestering constraint of Kaloper and Padilla [[KP 2013]](#ref-kp1), [[KP 2014]](#ref-kp2), who observed that the sign of the residual vacuum energy follows the sign of $$\langle3p-\rho\rangle$$ and is positive only if the dominant fluid near the turning point of a recollapsing universe has $$w>1/3$$. Carroll and Remmen [[CR]](#ref-cr) average the full Lagrangian density instead; the net vacuum energy of their equations is $$\tfrac12\langle T\rangle-\langle L_m\rangle$$ (section 7), and $$\langle T\rangle\le0$$ for $$w\le1/3$$. Shaw and Barrow [[SB]](#ref-sb) obtain $$\Lambda$$ from a causally restricted variational principle, a construction that is not put in the form (3) below. Oda [[Oda 1]](#ref-oda1), [[Oda 2]](#ref-oda2) studied the radiative stability of the nonlocal constraints. What is added here is the reduction of an arbitrary linear weight to the two functions $$A$$ and $$B$$, with a sign condition and its converse for constant weights, a version for arbitrary causal vectors, the sector-wise statement for sequestering, and the behaviour along an observer's worldline. On the page on [the cosmological constant and the dark-matter mass as free parameters](/quantum-gravity/dark-sector-moduli/), the vacuum energy of a field theory on a fixed background is a coordinate on the space of counterterms; a global condition of the kind studied here is a separate relation that assigns it a value. The place of these constraints among other approaches is on the [landscape page](/quantum-gravity/landscape/).

## 1. Background and conventions

We use the signature $$(-,+,+,+)$$ and units with $$c=1$$. A spatially flat FRW universe has the metric

$$
ds^2=a(\eta)^2\left(-d\eta^2+d\mathbf x^2\right), \tag{1}
$$

where $$a$$ is the scale factor, with $$a\to0$$ at the big bang $$\eta=0$$, and $$\eta$$ is conformal time, related to the proper time $$t$$ of comoving observers by $$dt=a\,d\eta$$. The Hubble rate is $$H=\dot a/a$$, with a dot for $$d/dt$$, and comoving observers have four-velocity $$u^a$$. Light cones are $$\lvert\Delta\mathbf x\rvert=\lvert\Delta\eta\rvert$$, so the causal past of an event $$P$$ at conformal time $$\eta_P$$ and scale factor $$a_P$$ is the set $$\{(\eta,\mathbf x):\,\lvert\mathbf x-\mathbf x_P\rvert\le\eta_P-\eta\}$$, which we write $$J^-(P)$$.

Matter is described by a perfect fluid with energy density $$\rho$$ and pressure $$p$$, and stress tensor

$$
T_{ab}=(\rho+p)\,u_au_b+p\,g_{ab}, \tag{2}
$$

with trace $$T=3p-\rho$$. Conservation of energy reads $$\dot\rho+3H(\rho+p)=0$$, so in an expanding universe the density of a fluid with $$\rho+p>0$$ falls. The equation of state is $$w=p/\rho$$: dust has $$w=0$$, radiation $$w=1/3$$. The null energy condition (NEC) is $$T_{ab}k^ak^b\ge0$$ for null $$k^a$$, which for (2) means $$\rho+p\ge0$$. The weak energy condition (WEC) is $$T_{ab}v^av^b\ge0$$ for causal $$v^a$$, which means $$\rho\ge0$$ and $$\rho+p\ge0$$. The dominant energy condition (DEC) additionally requires that $$-T^a{}_bN^b$$ be future-causal for future-causal $$N^a$$, which for (2) means $$\rho\ge\lvert p\rvert$$. We call the set of fluids with $$0\le p\le\rho/3$$ the fluid wedge $$\mathcal F$$. It contains dust, radiation and every mixture of the two, and it satisfies the DEC.

A vacuum energy density $$\rho_\Lambda$$, meaning the bare term plus all loop contributions, has the stress tensor $$T^{\Lambda}_{ab}=-\rho_\Lambda g_{ab}$$, that is $$p=-\rho$$. A component with a varying density $$\rho_X(a)$$ of either sign has an effective equation of state $$w_{\rm eff}$$ defined by $$d\ln\lvert\rho_X\rvert/d\ln a=-3(1+w_{\rm eff})$$. Then $$\lvert\rho_X\rvert$$ does not increase with time exactly when $$w_{\rm eff}\ge-1$$.

## 2. A linear constraint reduces to two functions

Let $$\mathcal D$$ be a region of spacetime, $$\mu$$ a positive measure on it, and $$W^{ab}(x)$$ a symmetric tensor field, the weight. A constraint linear in the stress tensor says that the weighted integral of the total stress tensor vanishes. Different matter sectors $$s$$ may carry different weights $$W_s^{ab}$$, and the vacuum a weight $$W_\Lambda^{ab}$$. The constraint reads

$$
\sum_s\int_{\mathcal D} d\mu\;W_s^{ab}\,T^{(s)}_{ab}\;=\;\rho_\Lambda\int_{\mathcal D} d\mu\;W_\Lambda^{ab}\,g_{ab}. \tag{3}
$$

The right-hand side is $$-\int d\mu\,W_\Lambda^{ab}T^{\Lambda}_{ab}$$, moved to the other side. Since $$\rho_\Lambda$$ is the net coefficient of $$g_{ab}$$ in the stress tensor, the constraint determines it independently of how that coefficient is built up from bare and quantum contributions. A single weight for everything, $$W_s=W_\Lambda=W$$, is the case of one matter sector. The measure $$\mu$$ may be supported on a hypersurface, which covers boundary-flux terms, and it absorbs any positive scalar factor of the weight.

Two weights show the pattern. For the trace weight $$W^{ab}=g^{ab}$$, the matter integrand is $$T=3p-\rho$$ and the right-hand side is $$4\rho_\Lambda$$ times the volume, so $$\rho_\Lambda=\tfrac14\langle3p-\rho\rangle$$. For the energy weight $$W^{ab}=u^au^b$$ the integrand is $$\rho$$ and $$g_{ab}u^au^b=-1$$, so $$\rho_\Lambda=-\langle\rho\rangle$$. In both, the weight enters only through two numbers, which we define at each point as the energy component $$A_s=W_s^{ab}u_au_b$$ and the trace $$B_s=g_{ab}W_s^{ab}$$ of the weight of sector $$s$$, and $$B_\Lambda=g_{ab}W_\Lambda^{ab}$$ for the vacuum weight. The next statement shows that this holds for every weight.

<div class="proposition" markdown="1">
**Proposition 1** (two functions). Let the matter of sector $$s$$ be a perfect fluid (2) with $$(\rho_s,p_s)$$ and let $$\int_{\mathcal D} B_\Lambda\,d\mu\ne0$$. Then (3) is equivalent to

$$
\rho_\Lambda=\frac{\displaystyle\sum_s\int_{\mathcal D} d\mu\,\big[(\rho_s+p_s)A_s+p_sB_s\big]}{\displaystyle\int_{\mathcal D} d\mu\,B_\Lambda}. \tag{4}
$$

For a single weight the numerator can be written $$\int d\mu\,\big[\tfrac14B\,(3p-\rho)+(\rho+p)\,(A+\tfrac14B)\big]$$, where $$A+\tfrac14B$$ is the energy component of the traceless part of the weight.
</div>

<div class="proof" markdown="1">
*Proof.* Contracting (2) with $$W_s^{ab}$$ gives $$W_s^{ab}T^{(s)}_{ab}=(\rho_s+p_s)A_s+p_sB_s$$, and $$W_\Lambda^{ab}g_{ab}=B_\Lambda$$. Insert in (3). For a single weight, write $$W^{ab}=\tfrac14Bg^{ab}+\mathring W^{ab}$$ with $$g_{ab}\mathring W^{ab}=0$$. Then $$\mathring W^{ab}u_au_b=A+\tfrac14B$$, and $$\mathring W^{ab}T_{ab}=(\rho+p)(A+\tfrac14B)$$ because the trace part of $$T_{ab}$$ drops out, while $$\tfrac14Bg^{ab}T_{ab}=\tfrac14B(3p-\rho)$$. $$\square$$
</div>

Anisotropic weights, weights tilted relative to the fluid, weights tied to an observer's worldline, weights that depend on the shape of the region, and boundary fluxes all enter only through $$A$$ and $$B$$. A weight $$W=v\otimes v$$ built from a unit timelike vector $$v$$ moving with rapidity $$\zeta$$ relative to the fluid has $$A=\cosh^2\zeta$$ and $$B=-1$$. A weight $$e\otimes e$$ built from a unit spacelike vector $$e$$ at rapidity $$\zeta$$ to the fluid has $$A=\sinh^2\zeta$$ and $$B=1$$. Multiplying $$W$$ by a positive function changes $$\mu$$ and leaves $$A/B$$ alone.

<div class="example" markdown="1">
**Example 1** (four weights on a $$\Lambda$$CDM past light cone). Take $$\mu$$ as the four-volume of a region and consider four weights in a universe of dust and radiation. The trace weight $$W=g$$ has $$(A,B)=(-1,4)$$ and gives $$\rho_\Lambda=\tfrac14\langle3p-\rho\rangle$$, which vanishes for radiation and equals $$-\tfrac14\rho_m$$ for dust. The pressure weight $$W=g+uu$$, the spatial projector, has $$(0,3)$$ and gives $$\rho_\Lambda=\langle p\rangle$$, zero for dust and $$\rho_r/3$$ for radiation. The energy weight $$W=uu$$ has $$(1,-1)$$ and gives $$\rho_\Lambda=-\langle\rho\rangle$$. The null weight $$W=k\otimes k$$ with $$k$$ null has $$B=0$$: the vacuum does not appear in (3) because $$g_{ab}k^ak^b=0$$, and the constraint $$\int T_{ab}k^ak^b\,d\mu=0$$ cannot be satisfied by matter with $$\rho+p>0$$.

To put numbers on these we use a flat $$\Lambda$$CDM background with the Planck 2018 values (temperature, polarisation and lensing) $$H_0=67.36\pm0.54$$ km s$$^{-1}$$ Mpc$$^{-1}$$, $$\Omega_m=0.3153\pm0.0073$$, $$\Omega_bh^2=0.02237\pm0.00015$$ and $$\Omega_ch^2=0.1200\pm0.0012$$ [[Planck VI]](#ref-planck6), Table 2. The value of $$\Omega_m$$ is taken as quoted, with its small contribution from massive neutrinos. The photon temperature $$T_0=2.72548\pm0.00057$$ K [[Fixsen]](#ref-fixsen), together with three massless neutrino species ($$N_{\rm eff}=3.046$$), gives the radiation density parameter $$\Omega_r=9.2\times10^{-5}$$, and a flat universe has $$\Omega_\Lambda=1-\Omega_m-\Omega_r=0.685$$. The observed vacuum energy is $$\rho_\Lambda^{\rm obs}=\Omega_\Lambda\rho_{\rm crit}$$, with $$\rho_{\rm crit}$$ today's critical density. All numbers on this page that carry a $$\Lambda$$CDM value are computed on this background.

For the region $$J^-(P)$$ of an observer today, with $$\mu$$ the four-volume, the averages of the matter densities are $$\langle\rho_m\rangle=23.9\,\rho_\Lambda^{\rm obs}$$ and $$\langle\rho_r\rangle=0.117\,\rho_\Lambda^{\rm obs}$$. The four weights give $$\rho_\Lambda/\rho_\Lambda^{\rm obs}=-5.96$$ (trace), $$+0.0391$$ (pressure), $$-24.0$$ (energy), and no information (null). The region contains a large amount of dust, which is why the dust-sensitive weights have large magnitude, and little radiation pressure, which is why the dust-blind one is small.
</div>

## 3. The sign condition

The matter allowed by $$0\le p\le\rho/3$$ is the convex cone in the $$(\rho,p)$$ plane generated by dust, $$(\rho,p)=(1,0)$$, and radiation, $$(1,\tfrac13)$$. The numerator of (4) is linear in $$(\rho_s,p_s)$$, so its sign on the whole wedge follows from its sign on the two generators. For radiation the contribution of sector $$s$$ is $$\tfrac13\rho_s(4A_s+B_s)$$, for dust $$\rho_sA_s$$.

<div class="theorem" markdown="1">
**Theorem 1** (sign condition). Assume the following.

(H1) The geometry is spatially flat FRW (1) and each matter sector is a perfect fluid with $$0\le p_s\le\rho_s/3$$ at every point of $$\mathcal D$$, that is in the wedge $$\mathcal F$$.

(H2) The measure $$\mu$$ is positive, and all integrals in (4) are finite.

(H3) The vacuum weight has $$B_\Lambda$$ of one sign $$\epsilon\in\{+1,-1\}$$ on $$\mathcal D$$ ($$\epsilon B_\Lambda\ge0$$ $$\mu$$-almost everywhere) and $$\int_{\mathcal D}B_\Lambda\,d\mu\ne0$$.

(a) If $$\epsilon A_s\le0$$ and $$\epsilon(4A_s+B_s)\le0$$ $$\mu$$-almost everywhere for every sector $$s$$, then $$\rho_\Lambda\le0$$ for all matter in $$\mathcal F$$. For a single weight $$W$$ with $$\epsilon B\ge0$$ the two conditions reduce to $$\epsilon(A+\tfrac14B)\le0$$, that is $$A/B\le-\tfrac14$$ where $$B\ne0$$.

(b) Conversely, suppose the functions $$A_s,B_s,B_\Lambda$$ are constants. If $$A_s>0$$ for some sector and $$B_\Lambda>0$$, then dust in that sector alone gives $$\rho_\Lambda>0$$. If $$4A_s+B_s>0$$ for some sector and $$B_\Lambda>0$$, then radiation in that sector alone gives $$\rho_\Lambda>0$$. The case $$B_\Lambda<0$$ follows by replacing every weight $$W$$ by $$-W$$, which leaves (3) unchanged and flips the signs of $$A_s$$, $$B_s$$, $$B_\Lambda$$. So for constant weights the conditions in (a) are necessary as well as sufficient for $$\rho_\Lambda\le0$$ on all of $$\mathcal F$$.
</div>

<div class="proof" markdown="1">
*Proof.* Write $$p_s=t_s\rho_s$$ with $$t_s\in[0,\tfrac13]$$. The sector contribution to the numerator of (4) is
$$\rho_s\big[(1+t_s)A_s+t_sB_s\big]=\rho_s\big[(1-3t_s)A_s+t_s(4A_s+B_s)\big]$$,
and $$1-3t_s\ge0$$, $$t_s\ge0$$, $$\rho_s\ge0$$. Under the hypotheses of (a), $$\epsilon$$ times each bracket is non-positive, so $$\epsilon$$ times the numerator is non-positive, while $$\epsilon$$ times the denominator is positive. Hence $$\rho_\Lambda\le0$$. For a single weight, $$\epsilon B\ge0$$ and $$\epsilon(4A+B)\le0$$ give $$\epsilon A\le-\tfrac14\epsilon B\le0$$, so the first condition follows from the second. For (b), with constant functions and $$B_\Lambda>0$$ the numerator of a one-sector universe is $$\rho A_s$$ for dust and $$\tfrac13\rho(4A_s+B_s)$$ for radiation, and the denominator is positive. $$\square$$
</div>

The proof is a check on the two generating rays of the wedge. By Proposition 1, the condition $$\epsilon(A+B/4)\le0$$ says that the traceless part of the weight, which is what distinguishes it from the trace weight, has an energy component of the sign opposite to that of the trace. The trace weight $$W=g$$ has $$A/B=-\tfrac14$$, the boundary of the condition, and radiation, whose trace $$T=3p-\rho$$ vanishes, contributes exactly zero to it. For the trace weight the sign can flip only when matter leaves $$\mathcal F$$ on the stiff side, $$p>\rho/3$$. The kinetic energy of a scalar field ($$w=1$$) contributes $$+\tfrac12\rho$$ to $$\rho_\Lambda=\tfrac14\langle3p-\rho\rangle$$, which is the stiff-matter case noted in [[KP 2014]](#ref-kp2).

On the past light cone of an observer today the stiff-matter case changes little, because the volume near the big bang is small. Take an epoch of kination ($$w=1$$) with density $$\rho_k=\rho_{r,0}\,a_{\rm kin}^2a^{-6}$$, which equals the radiation density at $$a_{\rm kin}$$ and rises as $$a^{-6}$$ into the past, and cut the region off below a scale factor $$a_c$$, since the kination term contributes logarithmically at small $$a$$. For $$a_{\rm kin}=10^{-9}$$ the trace value on $$J^-(P)$$ is unchanged from $$-5.96\,\rho_\Lambda^{\rm obs}$$ at the precision quoted. For $$a_{\rm kin}=10^{-4}$$, a value that nucleosynthesis excludes, it lies between $$-5.946$$ and $$-5.956$$ for $$10^{-14}\le a_c\le10^{-8}$$.

A positive $$\rho_\Lambda$$ in the wedge $$\mathcal F$$ requires $$\epsilon(A_s+B_s/4)>0$$ for some sector on a set of positive measure. For a constant ratio $$A/B=c$$ with $$B>0$$, pure radiation gives $$\rho_\Lambda=\tfrac13(4c+1)\langle\rho_r\rangle$$ and pure dust gives $$c\langle\rho_m\rangle$$. A positive value from radiation requires $$c>-\tfrac14$$, and from dust $$c>0$$. A weight with $$A=0$$ and $$B>0$$ gives $$\rho_\Lambda=\langle p\rangle$$ and is controlled by the radiation pressure alone.

<div class="remark" markdown="1">
**Remark 1** (sector-dependent weights). If cold dark matter shares the trace weight of the vacuum, $$A_c=-B_\Lambda/4$$, it contributes $$-\tfrac14B_\Lambda\langle\rho_c\rangle$$ to the numerator. A positive $$\rho_\Lambda$$ then requires baryons with a weight $$A_b$$ satisfying $$A_b\langle\rho_b\rangle>\tfrac14B_\Lambda\langle\rho_c\rangle$$, that is $$A_b/B_\Lambda>\rho_c/(4\rho_b)=\Omega_ch^2/(4\,\Omega_bh^2)=1.34$$ with the Planck values of Example 1, where $$B_\Lambda$$ is taken constant and positive and dust is the only matter. The wedge alone does not forbid this; the weight of baryons would be a number to be supplied, at more than 1.3 times the vacuum's trace weight.
</div>

## 4. Weights built from the causal structure

The wedge is defined in FRW. A weight constructed from the causal structure can be treated without any assumption on the background, using only energy conditions.

<div class="theorem" markdown="1">
**Theorem 2** (causal vectors). Let $$(M,g)$$ be any spacetime and $$T_{ab}$$ the stress tensor of matter, and let the vacuum have $$T^\Lambda_{ab}=-\rho_\Lambda g_{ab}$$ with $$\rho_\Lambda$$ constant. Suppose the constraint (3) is formed with a single weight $$W^{ab}=\int d\nu_x(v)\,v^av^b$$, where at each $$x\in\mathcal D$$ the measure $$\nu_x$$ is positive and supported on causal vectors $$v$$ ($$g(v,v)\le0$$), and the matter obeys the WEC. If $$\int d\mu\,d\nu\,g(v,v)<0$$, then $$\rho_\Lambda\le0$$. If $$g(v,v)=0$$ for $$\mu\otimes\nu$$-almost every $$v$$, the constraint contains no $$\rho_\Lambda$$ and is solvable only if $$T_{ab}v^av^b=0$$ almost everywhere.

The same conclusions hold for $$W^{ab}=\int d\nu\,\tfrac12(\xi^aN^b+\xi^bN^a)$$ with $$\xi,N$$ future-directed causal vectors, if the matter obeys the DEC.
</div>

<div class="proof" markdown="1">
*Proof.* The constraint reads $$\int d\mu\,d\nu\,[T(v,v)-\rho_\Lambda g(v,v)]=0$$. By the WEC, $$T(v,v)\ge0$$, and $$g(v,v)\le0$$ by assumption, so $$\rho_\Lambda\int g(v,v)=\int T(v,v)\ge0$$. When $$\int g(v,v)<0$$ this gives $$\rho_\Lambda\le0$$. When $$g(v,v)=0$$ almost everywhere, the constraint is $$\int T(v,v)=0$$. For the mixed weight, $$T(\xi,N)\ge0$$: under the DEC the vector $$j^a=-T^a{}_bN^b$$ is future-directed causal, and two future-directed causal vectors have $$g(\xi,j)\le0$$, which is $$-T(\xi,N)\le0$$; also $$g(\xi,N)\le0$$. The same argument applies. $$\square$$
</div>

The weights covered include the tangent to an observer's worldline, the generators of a light cone, the unit normal to a family of hypersurfaces, the four-velocity of the matter, and the flux of energy through a causal boundary measured by a future-directed timelike Killing field. In FRW with matter in $$\mathcal F$$, Theorem 2 is a special case of Theorem 1(a): a causal $$v$$ has $$A=(v\cdot u)^2\ge-g(v,v)=-B\ge0$$, so $$A/B\le-1\le-\tfrac14$$ where $$B<0$$. Theorem 2 needs neither FRW nor $$p\ge0$$; the energy conditions suffice, and causality fixes the sign of the denominator where Theorem 1 assumes it.

## 5. Sequestering with several matter sectors

The trace weight arises from an action principle. In the construction of [[KP 2013]](#ref-kp1), [[KP 2014]](#ref-kp2) the matter fields see the metric $$\lambda^2g_{ab}$$, where $$\lambda$$ is a spacetime constant that is varied, and a term $$\Phi(\Lambda/\lambda^4m_*^4)$$, with $$m_*$$ a fixed mass scale, makes $$\Lambda$$ a global variable as well. To see whether sector-dependent rescalings change the sign statement, we generalise this construction to several sectors.

Let $$S=\int d^4x\sqrt{-g}\,\big(\tfrac12M_P^2R-\Lambda\big)+\sum_sS_s[\lambda_s^2g,\psi_s]+\Phi\big(\Lambda/(m_*^4\prod_s\lambda_s^{4n_s})\big)$$, where $$M_P$$ is the reduced Planck mass, $$S_s$$ is the action of the fields $$\psi_s$$ of sector $$s$$, which couple to the metric $$\lambda_s^2g$$, the numbers $$n_s\ge0$$ are fixed weights, $$\Lambda$$ and the $$\lambda_s>0$$ are spacetime constants, and $$\Phi$$ is a function of one variable. We write $$V_4=\int\sqrt{-g}\,d^4x$$, which we take finite, and $$\langle X\rangle=V_4^{-1}\int\sqrt{-g}\,X\,d^4x$$. For the stress tensor $$T_s^{ab}=(2/\sqrt{-g})\,\delta S_s/\delta g_{ab}$$ sourcing $$g_{ab}$$, we write $$T_s=g_{ab}T_s^{ab}$$.

<div class="proposition" markdown="1">
**Proposition 3** (sector-wise sequestering; derivation within the model). On solutions of the field equations of this action, $$\langle T_s\rangle=4n_s\Lambda$$ for every sector $$s$$. Let $$V$$ be the sum of the vacuum energy densities of all sectors, so that $$T_s$$ contains $$-4V_s$$, and let $$T^{\rm nv}_s=T_s+4V_s$$ be the trace without the vacuum part. The net vacuum energy density $$\rho_\Lambda=\Lambda+V$$ is

$$
\rho_\Lambda=\tfrac14\sum_s\langle T_s^{\rm nv}\rangle+\Big(1-\sum_sn_s\Big)\Lambda. \tag{5}
$$

If $$\sum_sn_s=1$$, then $$\rho_\Lambda=\tfrac14\langle T^{\rm nv}\rangle$$ is independent of $$V$$, it is the trace-weight case of (3), and $$\rho_\Lambda\le0$$ whenever the matter lies in $$\mathcal F$$ over the whole four-volume. If $$\sum_sn_s\ne1$$, then $$\rho_\Lambda$$ depends on $$V$$ with the coefficient $$1-1/\sum_sn_s$$.
</div>

<div class="proof" markdown="1">
*Proof.* Varying $$\Lambda$$ gives $$-V_4+\Phi'(x)/(m_*^4\prod\lambda_s^{4n_s})=0$$, where $$x$$ is the argument of $$\Phi$$; hence $$x\,\Phi'(x)=\Lambda V_4$$. For the variation with respect to $$\lambda_s$$ write $$\tilde g=\lambda_s^2g$$, so that $$\delta\tilde g_{ab}=2\lambda_sg_{ab}\,\delta\lambda_s$$ and $$\partial S_s/\partial\lambda_s=\lambda_s^{-1}\int\sqrt{-\tilde g}\,\tilde T_s$$, where $$\tilde T_s$$ is the trace of the stress tensor of sector $$s$$ taken with $$\tilde g$$, and $$\sqrt{-\tilde g}=\lambda_s^4\sqrt{-g}$$. The derivative of the $$\Phi$$ term is $$-4n_s\,x\Phi'(x)/\lambda_s$$. Equating the two, $$\lambda_s^4\int\sqrt{-g}\,\tilde T_s=4n_s\Lambda V_4$$. The source of $$g_{ab}$$ is the stress tensor $$T_s^{ab}=(2/\sqrt{-g})\,\delta S_s/\delta g_{ab}$$, and since $$g_{ab}\,\delta/\delta g_{ab}=\tilde g_{ab}\,\delta/\delta\tilde g_{ab}$$ for a constant rescaling, its trace is $$T_s=\lambda_s^4\tilde T_s$$ (the factor $$\lambda_s^4$$ is the ratio of the volume elements). Hence $$\langle T_s\rangle=4n_s\Lambda$$. Summing over $$s$$ and using $$T_s=T_s^{\rm nv}-4V_s$$ gives $$\sum_s\langle T_s^{\rm nv}\rangle-4V=4\Lambda\sum_sn_s$$, which rearranges to (5) with $$\rho_\Lambda=\Lambda+V$$. The remaining statements follow from (5) and Theorem 1(a) with the trace weight; the coefficient of $$V$$ follows from $$\Lambda=(\tfrac14\sum\langle T^{\rm nv}_s\rangle-V)/\sum n_s$$. $$\square$$
</div>

The $$N$$ equations $$\langle T_s\rangle=4n_s\Lambda$$ for $$N$$ sectors determine the constants $$\lambda_s$$ and $$\Lambda$$ jointly with the $$\Lambda$$-equation, and (5) uses only their sum. The positivity statement uses the matter content over the whole four-volume, which in this construction includes the final collapse. Kinetic-dominated matter near the crunch is therefore inside the region, and its effect depends on the model of the final stage.

<div class="example" markdown="1">
**Example 2** (labelled sectors). Let cold dark matter and baryons carry different weights $$n_c\ne n_b$$ with $$n_c+n_b=1$$. Equation (5) gives $$\rho_\Lambda=\tfrac14\langle T^{\rm nv}_c+T^{\rm nv}_b\rangle=-\tfrac14\langle\rho_c+\rho_b\rangle$$ for dust. The split between $$n_c$$ and $$n_b$$ does not appear in the net vacuum energy, because the sectors enter through their traces and the trace of dust is $$-\rho$$ whatever its composition.
</div>

## 6. Drift along the worldline

Theorems 1 and 2 are statements at a point, or about an average with any positive measure. For the past light cone of an observer there is a second statement, about how the answer changes with the observer's epoch.

Let $$P$$ be an event at conformal time $$\eta_P$$, and $$\mathcal D=J^-(P)$$. Write $$\tau=\eta_P-\eta$$ for the radial distance in conformal units from the apex, $$\chi=\lvert\mathbf x-\mathbf x_P\rvert$$, and suppose the weighted measure of the constraint factorises as $$B\,d\mu=b(\eta)\,\kappa(\chi)\,d\eta\,d^3x$$ with $$b\ge0$$ depending on time only and $$\kappa\ge0$$ on the distance from the apex only. The four-volume, for instance, has $$b=a^4$$, $$\kappa=1$$. Integrating over the sphere at distance $$\chi\le\tau$$ gives the kernel $$K(\tau)=\int_0^\tau4\pi\chi^2\kappa(\chi)\,d\chi$$, with $$K(0)=0$$ (for the four-volume, $$K=\tfrac43\pi\tau^3$$). For a single weight with $$B>0$$ and constant $$A/B=c$$, the prediction (4) is

$$
\rho_\Lambda(\eta_P)=\frac{\int_0^{\eta_P}b(\eta)\,f(\eta)\,K(\eta_P-\eta)\,d\eta}{\int_0^{\eta_P}b(\eta)\,K(\eta_P-\eta)\,d\eta},\qquad f=c\,(\rho+p)+p. \tag{6}
$$

The observer's epoch enters through the upper limit of the integrals and through $$K$$. Raising $$\eta_P$$ moves the weight of the average towards early times, and the theorem below shows what that does to the value.

<div class="theorem" markdown="1">
**Theorem 3** (drift). Assume (H1) and (H2) of Theorem 1, a single weight with $$B>0$$ and constant $$A/B=c$$, and the following.

(H4) $$\mathcal D=J^-(P)$$, the measure has the factorised form above with $$b\ge0$$, and $$\int_0^{\eta_P}b\,K(\eta_P-\eta)\,d\eta>0$$ for every $$\eta_P>0$$.

(H5) $$K$$ is differentiable, $$K>0$$ on $$(0,\infty)$$, $$K(0)=0$$, and $$\ln K$$ is concave.

(H6) Every fluid component has constant $$w_s\in[0,\tfrac13]$$ and a density that does not increase with $$\eta$$, and either $$c\ge0$$ or $$c\le-\tfrac14$$.

(H7) The integrals in (6), and the integrals of $$b\,\lvert f\rvert\,K'$$ and $$b\,K'$$, converge locally uniformly in $$\eta_P$$.

If $$c\ge0$$, then $$\rho_\Lambda(\eta_P)\ge0$$ and it is non-increasing in $$\eta_P$$. If $$c\le-\tfrac14$$, then $$\rho_\Lambda(\eta_P)\le0$$ and it is non-decreasing in $$\eta_P$$. In both cases $$\lvert\rho_\Lambda\rvert$$ is non-increasing along every comoving worldline, which is $$w_{\rm eff}\ge-1$$. The monotonicity is strict if $$\ln K$$ is strictly concave and $$f$$ is not constant on the support of $$b$$.
</div>

<div class="proof" markdown="1">
*Proof.* The argument is Chebyshev's association inequality applied to the derivative of (6). Write $$N(\eta_P)$$ and $$D(\eta_P)$$ for the numerator and denominator of (6). Because $$K(0)=0$$, the boundary term in the derivative with respect to the upper limit vanishes, and by (H7), $$N'=\int_0^{\eta_P}b\,f\,K'(\eta_P-\eta)\,d\eta$$ and $$D'=\int_0^{\eta_P}b\,K'(\eta_P-\eta)\,d\eta$$. Let $$\langle\cdot\rangle_K$$ be the average over $$\eta\in(0,\eta_P)$$ with the probability density proportional to $$b(\eta)K(\eta_P-\eta)$$, and put $$\gamma(\eta)=K'/K$$ evaluated at $$\eta_P-\eta$$. Then $$\rho_\Lambda=\langle f\rangle_K$$, and

$$
\frac{d\rho_\Lambda}{d\eta_P}=\frac{N'-\rho_\Lambda D'}{D}=\big\langle(f-\langle f\rangle_K)\,\gamma\big\rangle_K=\mathrm{Cov}_K(f,\gamma).
$$

Here $$\gamma=(\ln K)'$$ is non-increasing in $$\tau$$ by (H5), so it is non-decreasing in $$\eta$$. For the sign of $$f$$ we use (H6): a component with $$f_s=\rho_s\,[\,c+(1+c)w_s\,]$$ has a bracket that is linear in $$w_s$$, equal to $$c$$ at $$w_s=0$$ and to $$(4c+1)/3$$ at $$w_s=\tfrac13$$. For $$c\ge0$$ both are non-negative and $$f$$ is non-increasing in $$\eta$$; for $$c\le-\tfrac14$$ both are non-positive and $$f$$ is non-decreasing in $$\eta$$. Two functions that are monotone in the same direction have a non-negative covariance under any probability measure, and in opposite directions a non-positive one: for independent copies $$\eta,\eta'$$ from the measure, $$\mathrm{Cov}(f,\gamma)=\tfrac12E[(f(\eta)-f(\eta'))(\gamma(\eta)-\gamma(\eta'))]$$, and the two factors have a definite common sign. Hence $$d\rho_\Lambda/d\eta_P\le0$$ for $$c\ge0$$ and $$\ge0$$ for $$c\le-\tfrac14$$. The sign of $$\rho_\Lambda=\langle f\rangle_K$$ is that of $$f$$, so $$d\lvert\rho_\Lambda\rvert/d\eta_P\le0$$ in both cases, and since $$\eta_P$$ increases with the proper time of a comoving observer, $$d\lvert\rho_\Lambda\rvert/dt\le0$$, which is $$w_{\rm eff}\ge-1$$. Equality needs the product $$(f(\eta)-f(\eta'))(\gamma(\eta)-\gamma(\eta'))$$ to vanish almost surely, which for strictly monotone $$\gamma$$ means $$f$$ is constant. $$\square$$
</div>

The wedge enters through the monotonicity of $$f$$. For $$-\tfrac14<c<0$$ the dust contribution to $$f$$ is negative and rising, and the radiation contribution is positive and falling, so $$f$$ need not be monotone and the argument does not apply; the radiation channel is the positive part and the dust channel the negative part. For $$c=-0.1$$ with the four-volume measure on the $$\Lambda$$CDM background the value does rise between $$a_P=0.4$$ and $$5$$, but this is a numerical observation that the theorem does not cover.

Within the class of Theorem 3, the value assigned to an observer depends on the epoch of the observer, and its magnitude is larger for earlier observers. A constraint of this form assigns one constant to all observers only if $$f$$ is constant on the support of the weight. If the prediction is read as the vacuum energy density at the epoch of the observer, which adds that identification as an assumption, then for $$c\ge0$$ the relation $$\dot\rho_\Lambda\le0$$ means energy is passing from the vacuum to matter, and $$w_{\rm eff}\ge-1$$ at all times. The predicted dark energy then cannot show the crossing of $$-1$$ in the Chevallier–Polarski–Linder form $$w(a)=w_0+w_a(1-a)$$. The Dark Energy Spectroscopic Instrument (DESI) second data release, combined with the cosmic microwave background, favours $$w_0>-1$$ and $$w_a<0$$ at $$3.1\sigma$$ over a constant $$\Lambda$$, and at $$2.8$$–$$4.2\sigma$$ when supernovae are added, depending on the supernova sample [[DESI]](#ref-desi). In this parametrisation $$w$$ is above $$-1$$ today and decreases into the past, and it reaches $$-1$$ at a finite redshift if $$w_0+w_a<-1$$. Observational reconstructions of $$w$$ assume that matter is conserved and fix its density at an early epoch, and a vacuum energy that decreases while matter increases changes the reconstructed history: with a late-time normalisation of the matter density, an interacting vacuum that feeds matter can be reconstructed as phantom [[DCK]](#ref-dck). The theorem constrains the vacuum energy density itself; the page on [the sign of energy exchange between vacuum and matter](/quantum-gravity/vacuum-matter-sign/) treats the reconstruction.

<div class="example" markdown="1">
**Example 3** (the pressure and trace weights on $$\Lambda$$CDM). Take the pressure weight $$c=0$$, $$B=1$$, with $$\mu$$ the four-volume, so that $$b=a^4$$, $$K=\tfrac43\pi\tau^3$$, $$f=p$$. Then $$\rho_\Lambda=\tfrac13\rho_{r,0}\langle a^{-4}\rangle$$, where $$\rho_{r,0}$$ is today's radiation density and $$\langle a^{-4}\rangle$$ is the average of $$a^{-4}$$ over $$J^-(P)$$ with the four-volume weight. For the $$\Lambda$$CDM background of Example 1, $$H_0\eta_0=3.179$$ and $$\langle a^{-4}\rangle=870$$ for an observer today, hence $$\rho_\Lambda=0.039\,\rho_\Lambda^{\rm obs}$$. To match $$\rho_\Lambda^{\rm obs}$$ the weighted average of $$a^{-4}$$ would have to equal $$3\Omega_\Lambda/\Omega_r=2.2\times10^{4}$$, which a delta-function weight achieves at $$a_*=0.082$$, $$z_*=11.2$$.

For an observer with scale factor $$a_P$$ the same weight gives the first row of the table below, relative to its value at $$a_P=1$$. The trace weight ($$c=-\tfrac14$$) with the same measure gives the second row, and the third row is its value in units of $$\rho_\Lambda^{\rm obs}$$.

| $$a_P$$ | 0.4 | 0.6 | 0.8 | 1 | 1.25 | 2 | 5 |
|---|---|---|---|---|---|---|---|
| pressure weight, $$\rho_\Lambda(a_P)/\rho_\Lambda(1)$$ | 21.3 | 4.87 | 1.90 | 1 | 0.573 | 0.228 | 0.073 |
| trace weight, $$\rho_\Lambda(a_P)/\rho_\Lambda(1)$$ | 10.4 | 3.37 | 1.64 | 1 | 0.648 | 0.312 | 0.121 |
| trace weight, $$\rho_\Lambda/\rho_\Lambda^{\rm obs}$$ | $$-61.8$$ | $$-20.1$$ | $$-9.79$$ | $$-5.96$$ | $$-3.86$$ | $$-1.86$$ | $$-0.72$$ |

At $$a_P=1$$ the logarithmic slope is $$d\ln\rho_\Lambda/d\ln a_P=-2.69$$ for the pressure weight, an effective equation of state $$w_{\rm eff}=-0.10$$, and $$-2.08$$ for the trace weight, $$w_{\rm eff}=-0.31$$. Both are far from $$-1$$. The pressure weight falls from a positive value and the trace weight rises from a negative one towards zero, as in Theorem 3.
</div>

## 7. Constraints linear in the matter Lagrangian

### Which constraint

A linear constraint need not be built from the stress tensor. A condition on the average of the Lagrangian density is linear in the Lagrangian. With the vacuum contributing $$L=-\rho_\Lambda$$, the simplest such condition is

$$
\rho_\Lambda=\langle L_m\rangle, \tag{7}
$$

with $$L_m$$ the on-shell matter Lagrangian density. We treat (7) as a stand-alone constraint. It is not the constraint of [[CR]](#ref-cr), which averages the Lagrangian density of the full action, gravity included. Averaging the trace of their field equation (their eq. (16)), $$G_{ab}+\tfrac12\langle R\rangle g_{ab}=8\pi G\,(T_{ab}-\langle L_m\rangle g_{ab})$$, gives $$\langle R\rangle=8\pi G(\langle T\rangle-4\langle L_m\rangle)$$, so that their equation takes the form $$G_{ab}+8\pi G\rho_{\rm eff}\,g_{ab}=8\pi G\,T_{ab}$$ with the net vacuum energy density

$$
\rho_{\rm eff}=\tfrac12\langle T\rangle-\langle L_m\rangle. \tag{8}
$$

Here $$\langle T\rangle$$ enters with weight $$\tfrac12$$ instead of $$\tfrac14$$, and $$\langle L_m\rangle$$ enters with the opposite sign from (7).

Both (7) and (8) are outside Proposition 1, because the on-shell value of $$L_m$$ is not a function of $$T_{ab}$$. For a perfect fluid, two standard choices of fluid Lagrangian give $$L_m=p$$ and $$L_m=-\rho$$. They give $$\rho_\Lambda=\langle p\rangle$$ and $$\rho_\Lambda=-\langle\rho\rangle$$ in (7), which for the dust-and-radiation mixture of Example 1 are $$+0.039$$ and $$-24.0$$ in units of $$\rho_\Lambda^{\rm obs}$$ on $$J^-(P)$$. For a collection of particles with fixed classical structure, $$\langle L_m\rangle=\langle T\rangle$$ [[AS]](#ref-as), which for dust is $$-\rho$$. The fluid description therefore does not determine the answer, and a quantum field theory has to supply the Lagrangian value of a state.

### Feynman–Hellmann identity

<div class="proposition" markdown="1">
**Proposition 4** (Feynman–Hellmann for the Lagrangian). Let the action be $$S=\lambda S_1+S_2$$ with $$S_2$$ independent of $$\lambda$$, and let $$\lvert\Psi\rangle$$ be a normalised energy eigenstate with energy $$E(\lambda)$$, differentiable in $$\lambda$$, with all other parameters and conserved charges fixed. Assume that $$H$$ is the Legendre transform of $$L$$, and that $$E$$ and $$\langle L\rangle$$ are defined with the same $$\lambda$$-independent regularisation and counterterms. Then the spatial integral of the expectation value of the Lagrangian density is $$\int d^3x\,\langle L\rangle=-\lambda\,\partial E/\partial\lambda+\int d^3x\,\langle L_2\rangle$$.
</div>

<div class="proof" markdown="1">
*Proof.* For a Hamiltonian $$H(q,p;\lambda)$$ obtained by Legendre transform of a Lagrangian $$L(q,\dot q;\lambda)$$, $$\partial H/\partial\lambda\vert_{q,p}=-\partial L/\partial\lambda\vert_{q,\dot q}$$. The Feynman–Hellmann theorem gives $$\partial E/\partial\lambda=\langle\partial H/\partial\lambda\rangle$$, and here $$\partial L/\partial\lambda=L_1$$ with $$L=\lambda L_1+L_2$$. So $$\langle\lambda L_1\rangle=-\lambda\,\partial E/\partial\lambda$$, and adding $$\langle L_2\rangle$$ gives the claim. $$\square$$
</div>

For a free field, $$\lambda$$ multiplies the whole action, $$L_2=0$$, and $$E$$ does not depend on $$\lambda$$ because rescaling the action does not change the frequencies of the modes; the single-particle states and the vacuum then have $$\langle L\rangle=0$$. For a mode of a free scalar with $$\omega^2=k^2+m^2$$, the Lagrangian $$\tfrac12(\dot\phi^2-k^2\phi^2-m^2\phi^2)$$ has vacuum expectation value $$\tfrac12(\omega^2-k^2-m^2)/(2\omega)=0$$, while the energy is $$\tfrac12\omega$$ per mode. The Lagrangian does not see the zero-point energy of free fields. For a classical static scalar soliton the kinetic terms vanish, the Lagrangian is the negative of the energy, and the value is $$-1$$ in units of $$E$$, consistent with $$E\propto\lambda$$.

For a gauge theory, take $$\lambda=1/g^2(M)$$ multiplying the gauge-field action at an ultraviolet scale $$M$$, and assume that the matter Lagrangian vanishes on shell (matrix elements of equation-of-motion operators between physical states vanish) and that a hadron mass in the chiral limit is proportional to the dynamically generated scale $$\Lambda_{\rm QCD}$$. At one loop, $$\Lambda_{\rm QCD}=M\exp[-2\pi/(b_3\alpha_s(M))]$$ with $$b_3=11-\tfrac23n_f=9$$ the coefficient for the three light flavours; at one loop the derivative with respect to $$1/\alpha_s(M)$$ does not depend on the thresholds between $$M$$ and the hadronic scale. Rescaling $$1/\alpha_s(M)\to\lambda/\alpha_s(M)$$ gives $$-\lambda\,\partial E/\partial\lambda=2\pi E/(b_3\alpha_s)$$, and Proposition 4 gives, under these assumptions, the ratio $$\ell=\int\langle L\rangle/E$$ for a state,

$$
\ell=\frac{2\pi}{9\,\alpha_s(M)}\qquad(\text{chiral limit, one loop}). \tag{9}
$$

For the nucleon this is positive, and it grows with the scale $$M$$: with two-loop running from $$\alpha_s(M_Z)=0.118$$ as input, it is $$5.9$$ at $$M=M_Z$$, $$6.5$$ at $$172.5$$ GeV, $$31$$ at $$10^{16}$$ GeV and $$36$$ at the reduced Planck mass. The ratio is not an observable of the low-energy theory, since it depends on where the multiplier acts, and the quark-mass part of the nucleon mass, of order ten per cent, adds a correction.

### Consequences for the two constraints

Combining the cases, baryons carry $$\ell_b\approx5$$ to $$36$$ in (9), the value depending on a scale $$M$$ that has to be chosen, and cold dark matter carries $$\ell=0$$ if it is an elementary particle and $$\ell=-1$$ if it is a static soliton. In the stand-alone constraint (7) on $$J^-(P)$$, with $$\langle\rho_b\rangle=3.7\,\rho_\Lambda^{\rm obs}$$ and elementary dark matter, the value is $$\ell_b\times3.7\,\rho_\Lambda^{\rm obs}$$, that is $$20$$ to $$130\,\rho_\Lambda^{\rm obs}$$, with the number set by $$M$$. The sign can be positive, and the magnitude contains one free number.

In the equations of [[CR]](#ref-cr), (8) applies instead. For dust $$\langle T\rangle=-\langle\rho\rangle$$, so $$\rho_{\rm eff}=-\tfrac12\langle\rho_b+\rho_c\rangle-\ell_b\langle\rho_b\rangle-\ell_c\langle\rho_c\rangle$$ with $$\ell_c$$ the value for cold dark matter. For elementary dark matter this is negative. For soliton dark matter, $$\ell_c=-1$$, it is positive only if $$\rho_c/\rho_b>1+2\ell_b$$, while Planck gives $$\rho_c/\rho_b=5.4$$ and (9) gives $$1+2\ell_b\ge12.8$$. Carroll and Remmen argue that $$L_m$$ vanishes on shell for radiation and for fermionic matter [[CR]](#ref-cr), which agrees with the free-field result above; the gluon action in (9) is the term they set aside.

### Radiative stability

The multiplier of (7) has a further difficulty: it is not radiatively stable, by the following argument, which is a heuristic. A multiplier that acts as an overall factor on the matter action leaves free-field vacuum energy untouched, as the mode sum above shows, so the zero-point energy of free fields shifts the net vacuum energy without shifting the right side of the constraint. Oda [[Oda 1]](#ref-oda1) found the corresponding instability in the nonlocal constraint of [[CR]](#ref-cr) by a one-loop calculation. In the metric-rescaling construction of section 5 the multiplier couples to the trace of the stress tensor instead, the zero-point energy contributes $$-4V$$ to the trace, and (5) is independent of $$V$$ when $$\sum n_s=1$$.

## 8. What the theorems do not exclude

### Weights with a positive energy component relative to the trace

If $$\epsilon(A+B/4)>0$$ on a set of positive measure, Theorem 1 gives no sign. With the weight normalised so that its trace is positive, such a weight has a traceless part with positive energy component in the fluid frame, as the spatial projector and a boosted spatial stress do. The examples considered here are defined by a preferred frame $$u$$ or direction; no argument is given that a frame is necessary. Two classes are quantified. For the weight with $$A=0$$ the value is $$\langle p\rangle$$, and matching $$\rho_\Lambda^{\rm obs}$$ needs a weighted average of $$a^{-4}$$ equal to $$3\Omega_\Lambda/\Omega_r$$, which a sharply peaked weight at redshift $$z_*=11.2$$ delivers (Example 3). For a weight tilted by rapidity $$\zeta$$ to the fluid ($$A=\sinh^2\zeta$$, $$B=1$$) the value is $$\approx\sinh^2\zeta\,\langle\rho_m\rangle$$, so matching needs $$\sinh^2\zeta\approx\rho_\Lambda^{\rm obs}/\langle\rho_m\rangle=0.042$$ on the past cone of Example 1. If the tilt is the motion of the observer relative to the CMB frame, the dipole of amplitude $$3362\,\mu$$K [[Planck I]](#ref-planck1) corresponds to $$\beta\approx1.2\times10^{-3}$$ and $$\sinh^2\zeta\approx1.5\times10^{-6}$$, which gives a value smaller than $$\rho_\Lambda^{\rm obs}$$ by a factor of about $$3\times10^{4}$$.

### Varying weights

Theorem 1 holds pointwise, so a ratio $$A/B$$ that varies in spacetime does not enlarge the class of sign-definite weights. It does permit a family of positive values with a free measure. For a weight with $$A=0$$ and $$B\propto a^{-n}$$ on the four-volume of $$J^-(P)$$, the average of $$p$$ converges at the big bang only for $$n<1$$. With $$n=1$$ the integral grows logarithmically with the small-scale cutoff, and the value is $$55\,\rho_\Lambda^{\rm obs}$$ for a cutoff at the scale factor of a TeV-temperature universe ($$a_c=2.3\times10^{-16}$$) and $$125\,\rho_\Lambda^{\rm obs}$$ for a cutoff at the Planck density ($$a_c=2\times10^{-32}$$). An exponent $$n\approx0.67$$ gives $$\rho_\Lambda^{\rm obs}$$; it is a fitted value. This weight has the factorised form of Theorem 3 with $$b=a^{4-n}$$, and the prediction has $$w_{\rm eff}=-0.18$$ at the present epoch and is $$1.8$$ times larger for an observer at $$a_P=0.8$$.

### Stiff matter and negative pressure

Outside $$0\le p\le\rho/3$$ the wedge argument fails. The trace weight over a collapsing universe is positive when the kinetic energy of a scalar field dominates near the crunch [[KP 2014]](#ref-kp2), and for a region that extends into such a future the sign is not fixed by Theorem 1. On the past cone of an observer today, an early kinetic epoch does not change the sign (section 3). Matter with $$p<0$$, such as a scalar field on a potential plateau, enters the trace weight as $$\tfrac14(3p-\rho)<0$$, which pushes the value further negative.

### Regions that are not past-causal

A constraint averaged over the collapsing patches of a halo model [[Lombriser]](#ref-lombriser) has a measure that is not of the form considered in Theorem 3, and that construction is not worked out here. Theorem 1 applies to any region on which its hypotheses hold.

### Non-linear constraints

Theorem 1 uses linearity in the stress tensor. A constraint quadratic in the total stress tensor, such as $$\langle T_{\rm tot}^2-2\,T^{\rm tot}_{ab}T_{\rm tot}^{ab}\rangle_\mu=0$$ with $$T^{\rm tot}_{ab}=T_{ab}-\rho_\Lambda g_{ab}$$ and $$T_{\rm tot}$$ its trace, is a scalar condition with no preferred frame and no action behind it. Expanding gives the quadratic equation $$8\rho_\Lambda^2-4\langle T\rangle\rho_\Lambda+\langle T^2-2T_{ab}T^{ab}\rangle=0$$, with $$T=3p-\rho$$ and $$T_{ab}T^{ab}=\rho^2+3p^2$$. For matter in $$\mathcal F$$ and $$\rho>0$$, $$T^2-2T_{ab}T^{ab}=3p^2-6p\rho-\rho^2<0$$, so with finite averages the product of the roots is negative and there is exactly one positive root, for every positive measure and every region. For dust the root is $$\rho_\Lambda=\tfrac14\big(-\langle\rho\rangle+\sqrt{\langle\rho\rangle^2+2\langle\rho^2\rangle}\big)$$, which is $$\tfrac14(\sqrt3-1)\rho$$ for constant $$\rho$$. The root is positive whatever the measure, so the sign statement of Theorem 1 depends on linearity. No radiative stability has been shown for this constraint. On $$J^-(P)$$ with the four-volume measure $$\langle\rho^2\rangle$$ diverges at the big bang, so the averages need a cutoff there. The constructions of [[SB]](#ref-sb) and [[Oda 2]](#ref-oda2) are not covered by the theorems above; whether they can be put in the form (3) is not examined.

### Lagrangian-linear constraints

The stand-alone constraint (7) gives a positive value, with the free number $$M$$ and without radiative stability (section 7). The constraint (8) of [[CR]](#ref-cr) gives a negative value for the matter content considered there.

### Hypotheses dropped by these examples

The weights with $$A/B>-\tfrac14$$ drop the sign condition of Theorem 1(a), the stiff-matter case drops (H1), and the non-linear and Lagrangian-linear constraints drop linearity in $$T_{ab}$$. Each gives a positive $$\rho_\Lambda$$, so the exclusion is not a statement about all global constraints. Hypothesis (H3) fixes the sign of the denominator of (4); if $$B_\Lambda$$ changes sign on $$\mathcal D$$ the denominator can be small or of either sign, and no example of that kind is worked out here.

## 9. Open points

The reduction of a weight to $$A$$ and $$B$$ is for a perfect fluid on an FRW background. Inhomogeneities are not included, so a weight that responds to the anisotropic stress or to the velocity field of structure is not covered; it would add terms of the order of the density contrast to the numerator of (4). Theorem 3 assumes a separable measure, a constant ratio $$A/B$$ and, for the sign of the drift, either $$A/B\ge0$$ or $$A/B\le-\tfrac14$$; a measure that couples $$\eta$$ and $$\chi$$ and the window $$-\tfrac14<A/B<0$$ are not treated, and the monotonicity of $$f$$ fails there. Whether a sequestering mechanism with a single global variable can be made radiatively stable and still distinguish sectors by anything other than the trace is not settled by the identity (5). The identification of the value assigned to an observer with a running vacuum energy density, and the reading of its decrease as an energy flow into matter, add an assumption to Theorem 3 that the constraint itself does not supply.

## References

1. <a id="ref-kp1"></a>Kaloper, Padilla, “Sequestering the Standard Model Vacuum Energy,” Phys. Rev. Lett. 112, 091304 (2014), [arXiv:1309.6562](https://arxiv.org/abs/1309.6562).
2. <a id="ref-kp2"></a>Kaloper, Padilla, “Vacuum Energy Sequestering: The Framework and Its Cosmological Consequences,” Phys. Rev. D 90, 084023 (2014), [arXiv:1406.0711](https://arxiv.org/abs/1406.0711).
3. <a id="ref-cr"></a>Carroll, Remmen, “A Nonlocal Approach to the Cosmological Constant Problem,” Phys. Rev. D 95, 123504 (2017), [arXiv:1703.09715](https://arxiv.org/abs/1703.09715).
4. <a id="ref-sb"></a>Shaw, Barrow, “A Testable Solution of the Cosmological Constant and Coincidence Problems,” Phys. Rev. D 83, 043518 (2011), [arXiv:1010.4262](https://arxiv.org/abs/1010.4262).
5. <a id="ref-oda1"></a>Oda, “Quantum Aspects of Nonlocal Approach to the Cosmological Constant Problem,” Phys. Rev. D 96, 024027 (2017), [arXiv:1706.01683](https://arxiv.org/abs/1706.01683).
6. <a id="ref-oda2"></a>Oda, “Cosmic Acceleration in the Nonlocal Approach to the Cosmological Constant Problem,” [arXiv:1711.10476](https://arxiv.org/abs/1711.10476).
7. <a id="ref-lombriser"></a>Lombriser, “Late-time acceleration by a residual cosmological constant from sequestering vacuum energy in ultimate collapsed structures,” JCAP 09 (2019) 065, [arXiv:1805.05918](https://arxiv.org/abs/1805.05918).
8. <a id="ref-as"></a>Avelino, Sousa, “Matter Lagrangian of particles and fluids,” Phys. Rev. D 97, 064019 (2018), [arXiv:1802.03961](https://arxiv.org/abs/1802.03961).
9. <a id="ref-dck"></a>Das, Corasaniti, Khoury, “Super-acceleration as Signature of Dark Sector Interaction,” Phys. Rev. D 73, 083509 (2006), [arXiv:astro-ph/0510628](https://arxiv.org/abs/astro-ph/0510628).
10. <a id="ref-planck6"></a>Planck Collaboration, “Planck 2018 results. VI. Cosmological parameters,” Astron. Astrophys. 641, A6 (2020), [arXiv:1807.06209](https://arxiv.org/abs/1807.06209).
11. <a id="ref-planck1"></a>Planck Collaboration, “Planck 2018 results. I. Overview and the cosmological legacy of Planck,” Astron. Astrophys. 641, A1 (2020), [arXiv:1807.06205](https://arxiv.org/abs/1807.06205).
12. <a id="ref-fixsen"></a>Fixsen, “The Temperature of the Cosmic Microwave Background,” Astrophys. J. 707, 916 (2009), [arXiv:0911.1955](https://arxiv.org/abs/0911.1955).
13. <a id="ref-desi"></a>DESI Collaboration, “DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints,” Phys. Rev. D 112, 083515 (2025), [arXiv:2503.14738](https://arxiv.org/abs/2503.14738).
