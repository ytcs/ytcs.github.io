---
title: "Linear global constraints on the cosmological constant: sign and drift"
date: 2026-10-29
category: dark-sector
order: 1
description: In a flat expanding universe filled with dust and radiation, a global constraint that is linear in the stress tensor gives a non-positive vacuum energy unless its weight carries a positive energy component in the traceless part, which needs a preferred frame; weights tied to an observer's past light cone give a value that never increases along the worldline.
status: Theorems proved in full for spatially flat FRW matter with 0 <= p <= rho/3; one explicit non-linear counterexample; numerical checks on a LambdaCDM background only; the Lagrangian-linear section is a derivation under stated model assumptions.
excludes: Vacuum-energy sequestering and related linear averaging constraints as a source of a positive cosmological constant, when the weight is built from the causal structure or from a metric-rescaling multiplier and the matter has 0 <= p <= rho/3.
---

A family of proposals removes the vacuum energy from the gravitational field equations by adding a global condition: a weighted spacetime average of the stress tensor, with the vacuum included, must vanish. The condition then fixes the net vacuum energy density $$\rho_\Lambda$$ in terms of the matter content, whatever the quantum corrections to the vacuum energy are. We ask which values of $$\rho_\Lambda$$ a condition of this kind can give when it is linear in the stress tensor. In a spatially flat Friedmann–Robertson–Walker (FRW) universe whose matter has pressure $$0\le p\le\rho/3$$, every such condition depends on its weight only through two numbers at each point, the energy component $$A$$ and the trace $$B$$ of the weight in the matter rest frame. Then $$\rho_\Lambda\le0$$ whenever the energy component $$A+B/4$$ of the traceless part of the weight has the sign opposite to $$B$$ (Theorem 1). Weights built from the causal structure of spacetime satisfy this (Theorem 2), and so do the metric-rescaling multipliers of vacuum energy sequestering with any number of matter sectors whose exponents sum to one (Proposition 3). A positive $$\rho_\Lambda$$ needs a weight with a positive energy component in its traceless part, which in practice means a preferred frame. For a weight that depends on the observer only through the past light cone, the value assigned to an observer never increases along the observer's worldline (Theorem 3). Read as a dark-energy density, it then has an equation of state $$w\ge-1$$ and cannot cross $$-1$$.

The trace weight is the sequestering constraint of Kaloper and Padilla [[KP 2013]](#ref-kp1), [[KP 2014]](#ref-kp2). These authors observed that the sign of the residual vacuum energy follows the sign of $$\langle3p-\rho\rangle$$, positive only if the dominant fluid near the turning point of a recollapsing universe has $$w>1/3$$. Carroll and Remmen [[CR]](#ref-cr) require instead that the spacetime average of the Lagrangian density vanish, and note that the average of the trace $$\langle T\rangle$$ is non-positive for $$w\le1/3$$. Shaw and Barrow [[SB]](#ref-sb) fix $$\Lambda$$ by a constraint that follows from a causally restricted variational principle; it is a different construction from the linear weights considered here. Oda [[Oda 1]](#ref-oda1), [[Oda 2]](#ref-oda2) studied the radiative stability of the nonlocal constraints. What this page adds is the reduction of an arbitrary linear weight to two kernels, the sign cone with its converse for constant weights, the causal-vector theorem, the sector-wise statement, the monotonicity along worldlines, and a list of the places where each hypothesis is used and what lies outside it.

## 1. Background and conventions

We use the signature $$(-,+,+,+)$$ and units with $$c=1$$. A spatially flat FRW universe has the metric

$$
ds^2=a(\eta)^2\left(-d\eta^2+d\mathbf x^2\right), \tag{1}
$$

where $$a$$ is the scale factor and $$\eta$$ is conformal time, related to proper time $$t$$ of comoving observers by $$dt=a\,d\eta$$. The Hubble rate is $$H=\dot a/a$$, with a dot for $$d/dt$$. Comoving observers have four-velocity $$u^a$$, and $$a=0$$ at the big bang $$\eta=0$$. Light cones are $$\lvert\Delta\mathbf x\rvert=\lvert\Delta\eta\rvert$$, so the causal past of an event $$p$$ at conformal time $$\eta_p$$ is the set $$\{(\eta,\mathbf x):\,\lvert\mathbf x-\mathbf x_p\rvert\le\eta_p-\eta\}$$, which we write $$J^-(p)$$.

Matter is described by a perfect fluid with energy density $$\rho$$ and pressure $$p$$, and stress tensor

$$
T_{ab}=(\rho+p)\,u_au_b+p\,g_{ab}. \tag{2}
$$

Conservation of energy reads $$\dot\rho+3H(\rho+p)=0$$, so in an expanding universe the density of a fluid with $$\rho+p>0$$ falls. The equation of state is $$w=p/\rho$$: dust has $$w=0$$, radiation $$w=1/3$$. The null energy condition (NEC) is $$T_{ab}k^ak^b\ge0$$ for null $$k^a$$, which for (2) means $$\rho+p\ge0$$. The weak energy condition (WEC) is $$T_{ab}v^av^b\ge0$$ for causal $$v^a$$, which means $$\rho\ge0$$ and $$\rho+p\ge0$$. The dominant energy condition (DEC) additionally requires that $$-T^a{}_bN^b$$ be future-causal for future-causal $$N^a$$, which for (2) means $$\rho\ge\lvert p\rvert$$. We call the class of fluids with $$0\le p\le\rho/3$$ the cone $$\mathcal T$$; it contains dust, radiation, and every mixture of the two, and it satisfies the DEC.

A vacuum energy density $$\rho_\Lambda$$, meaning the bare term plus all loop contributions, has the stress tensor $$T^{\Lambda}_{ab}=-\rho_\Lambda g_{ab}$$, that is $$p=-\rho$$. A component whose energy density $$\rho_X(a)$$ varies is described by $$d\ln\rho_X/d\ln a=-3(1+w_{\rm eff})$$, which defines $$w_{\rm eff}$$. If $$\rho_X$$ never increases with time, then $$w_{\rm eff}\ge-1$$.

Observational numbers enter only in the examples. The Planck 2018 analysis of the cosmic microwave background (CMB) temperature, polarisation and lensing gives $$H_0=67.36\pm0.54$$ km s$$^{-1}$$ Mpc$$^{-1}$$, matter density parameter $$\Omega_m=0.3153\pm0.0073$$, $$\Omega_bh^2=0.02237\pm0.00015$$ and $$\Omega_ch^2=0.1200\pm0.0012$$ [[Planck VI]](#ref-planck6), Table 2. The photon temperature is $$T_0=2.72548\pm0.00057$$ K [[Fixsen]](#ref-fixsen), which with three massless neutrino species ($$N_{\rm eff}=3.046$$) gives a radiation density parameter $$\Omega_r=9.2\times10^{-5}$$ (our computation), and $$\Omega_\Lambda=1-\Omega_m-\Omega_r=0.685$$ in a flat universe. The observed vacuum energy is then $$\rho_\Lambda^{\rm obs}=\Omega_\Lambda\rho_{c}$$, with $$\rho_c$$ today's critical density. The Dark Energy Spectroscopic Instrument (DESI) second data release, combined with the CMB, favours a dark-energy equation of state $$w(a)=w_0+w_a(1-a)$$ with $$w_0>-1$$ and $$w_a<0$$, at $$3.1\sigma$$ over a constant $$\Lambda$$, and at $$2.8$$–$$4.2\sigma$$ when supernovae are added, depending on the supernova sample [[DESI]](#ref-desi). In this parametrisation $$w$$ is above $$-1$$ today and decreases into the past, and it reaches $$-1$$ at a finite redshift if $$w_0+w_a<-1$$. Whether a crossing of $$-1$$ is present in the data is a question about fits; we note only that Theorem 3 below excludes it for the class of constraints considered.

## 2. A linear constraint reduces to two kernels

Let $$R$$ be a region of spacetime, $$\mu$$ a positive measure on it, and $$w^{ab}(x)$$ a symmetric tensor field, the weight. A constraint linear in the stress tensor says that the weighted integral of the total stress tensor vanishes. If different matter sectors $$s$$ carry different weights $$w_s^{ab}$$ and the vacuum carries $$w_0^{ab}$$, the constraint reads

$$
\sum_s\int_R d\mu\;w_s^{ab}\,T^{(s)}_{ab}\;=\;\rho_\Lambda\int_R d\mu\;w_0^{ab}\,g_{ab}. \tag{3}
$$

The right-hand side is $$-\int d\mu\,w_0^{ab}T^{\Lambda}_{ab}$$, moved to the other side. Since $$\rho_\Lambda$$ is the net coefficient of $$g_{ab}$$ in the stress tensor, the constraint determines it independently of how that coefficient is built up from bare and quantum contributions. A single weight for everything, $$w_s=w_0=w$$, is the case of one matter sector. The measure $$\mu$$ may be supported on a hypersurface, which covers boundary-flux terms, and it absorbs any positive scalar factor of the weight.

We define, at each point, the energy component $$A_s=w_s^{ab}u_au_b$$ and the trace $$B_s=g_{ab}w_s^{ab}$$ of the weight of sector $$s$$, and $$B_0=g_{ab}w_0^{ab}$$ for the vacuum weight.

<div class="proposition" markdown="1">
**Proposition 1** (two kernels). Let the matter of sector $$s$$ be a perfect fluid (2) with $$(\rho_s,p_s)$$ and let $$\int_R B_0\,d\mu\ne0$$. Then (3) is equivalent to

$$
\rho_\Lambda=\frac{\displaystyle\sum_s\int_R d\mu\,\big[(\rho_s+p_s)A_s+p_sB_s\big]}{\displaystyle\int_R d\mu\,B_0}. \tag{4}
$$

For a single weight the numerator can be written $$\int d\mu\,\big[\tfrac14B\,(3p-\rho)+(\rho+p)\,(A+\tfrac14B)\big]$$, where $$A+\tfrac14B$$ is the energy component of the traceless part of the weight.
</div>

<div class="proof" markdown="1">
*Proof.* Contracting (2) with $$w_s^{ab}$$ gives $$w_s^{ab}T^{(s)}_{ab}=(\rho_s+p_s)A_s+p_sB_s$$, and $$w_0^{ab}g_{ab}=B_0$$. Insert in (3). For a single weight, write $$w^{ab}=\tfrac14Bg^{ab}+\mathring w^{ab}$$ with $$g_{ab}\mathring w^{ab}=0$$. Then $$\mathring w^{ab}u_au_b=A+\tfrac14B$$, and $$\mathring w^{ab}T_{ab}=(\rho+p)(A+\tfrac14B)$$ because the trace part of $$T_{ab}$$ drops out, while $$\tfrac14Bg^{ab}T_{ab}=\tfrac14B(3p-\rho)$$. $$\square$$
</div>

This step is routine, and it carries the generality of the result: anisotropic weights, weights tilted relative to the fluid, weights tied to an observer's worldline, weights that depend on the shape of the region, and boundary fluxes all enter only through $$A$$ and $$B$$. A weight $$w=v\otimes v$$ built from a unit timelike vector $$v$$ moving with velocity $$\tanh\zeta$$ relative to the fluid has $$A=\cosh^2\zeta$$ and $$B=-1$$. A weight $$e\otimes e$$ built from a unit spacelike vector $$e$$ at rapidity $$\zeta$$ to the fluid has $$A=\sinh^2\zeta$$ and $$B=1$$. Multiplying $$w$$ by a positive function changes $$\mu$$ and leaves $$A/B$$ alone.

<div class="example" markdown="1">
**Example 1** (four weights). In a universe of dust and radiation, take $$\mu$$ as the four-volume of a region and consider four weights. The trace weight $$w=g$$ has $$(A,B)=(-1,4)$$ and gives $$\rho_\Lambda=\tfrac14\langle3p-\rho\rangle$$, which vanishes for radiation and equals $$-\tfrac14\rho_m$$ for dust. The pressure weight $$w=g+uu$$, the spatial projector, has $$(0,3)$$ and gives $$\rho_\Lambda=\langle p\rangle$$, zero for dust and $$\rho_r/3$$ for radiation. The energy weight $$w=uu$$ has $$(1,-1)$$ and gives $$\rho_\Lambda=-\langle\rho\rangle$$. The null weight $$w=k\otimes k$$ with $$k$$ null has $$B=0$$: the vacuum does not appear in (3), because $$g_{ab}k^ak^b=0$$, and the constraint $$\int T_{ab}k^ak^b\,d\mu=0$$ cannot be satisfied by matter with $$\rho+p>0$$.

For the region $$J^-(p)$$ of an observer today in the $$\Lambda$$CDM universe of section 1, with $$\mu$$ the four-volume, the numbers are as follows (our computation; photons plus three massless neutrino species, no neutrino masses). The averages of the matter densities over the region are $$\langle\rho_m\rangle=23.9\,\rho_\Lambda^{\rm obs}$$ and $$\langle\rho_r\rangle=0.117\,\rho_\Lambda^{\rm obs}$$. The four weights give $$\rho_\Lambda/\rho_\Lambda^{\rm obs}=-5.96$$ (trace), $$+0.0391$$ (pressure), $$-24.0$$ (energy), and no information (null). The region contains a large amount of dust, which is why the dust-sensitive weights have large magnitude, and little radiation pressure, which is why the dust-blind one is small.
</div>

## 3. The sign cone

The matter allowed by $$0\le p\le\rho/3$$ is the convex cone generated by dust, $$(\rho,p)=(1,0)$$, and radiation, $$(1,\tfrac13)$$. The numerator of (4) is linear in $$(\rho_s,p_s)$$, so its sign on the whole cone follows from its sign on the two generators. For radiation the contribution of sector $$s$$ is $$\tfrac13\rho_s(4A_s+B_s)$$, for dust $$\rho_sA_s$$.

<div class="theorem" markdown="1">
**Theorem 1** (sign cone). Assume the following.

(H1) The geometry is spatially flat FRW (1) and each matter sector is a perfect fluid with $$0\le p_s\le\rho_s/3$$ at every point of $$R$$ (the cone $$\mathcal T$$).

(H2) The measure $$\mu$$ is positive, and all integrals in (4) are finite.

(H3) The vacuum weight has $$B_0$$ of one sign $$\sigma\in\{+1,-1\}$$ on $$R$$ ($$\sigma B_0\ge0$$ $$\mu$$-almost everywhere) and $$\int_RB_0\,d\mu\ne0$$.

(a) If $$\sigma A_s\le0$$ and $$\sigma(4A_s+B_s)\le0$$ $$\mu$$-almost everywhere for every sector $$s$$, then $$\rho_\Lambda\le0$$ for all matter in $$\mathcal T$$. For a single weight $$w$$ with $$\sigma B\ge0$$ the two conditions reduce to $$\sigma(A+\tfrac14B)\le0$$, that is $$A/B\le-\tfrac14$$ where $$B\ne0$$.

(b) Conversely, suppose the kernels $$A_s,B_s,B_0$$ are constants and $$\sigma=+1$$. If $$A_s>0$$ for some sector, then dust in that sector alone gives $$\rho_\Lambda>0$$ when $$B_0>0$$. If $$4A_s+B_s>0$$ for some sector, then radiation in that sector alone gives $$\rho_\Lambda>0$$. So for constant weights the conditions in (a) are necessary as well as sufficient for $$\rho_\Lambda\le0$$ on all of $$\mathcal T$$.
</div>

<div class="proof" markdown="1">
*Proof.* Write $$p_s=t_s\rho_s$$ with $$t_s\in[0,\tfrac13]$$. The sector contribution to the numerator of (4) is
$$\rho_s\big[(1+t_s)A_s+t_sB_s\big]=\rho_s\big[(1-3t_s)A_s+t_s(4A_s+B_s)\big]$$,
and $$1-3t_s\ge0$$, $$t_s\ge0$$, $$\rho_s\ge0$$. Under the hypotheses of (a), $$\sigma$$ times each bracket is non-positive, so $$\sigma$$ times the numerator is non-positive, while $$\sigma$$ times the denominator is positive. Hence $$\rho_\Lambda\le0$$. For a single weight, $$\sigma B\ge0$$ and $$\sigma(4A+B)\le0$$ give $$\sigma A\le-\tfrac14\sigma B\le0$$, so the first condition follows from the second. For (b), with constant kernels and $$B_0>0$$ the numerator of a one-sector universe is $$\rho A_s$$ for dust and $$\tfrac13\rho(4A_s+B_s)$$ for radiation, and the denominator is positive. $$\square$$
</div>

The idea is that $$\mathcal T$$ is a two-ray cone and the constraint is a linear functional on it. There is no more to the proof; the content is in what the condition means. By Proposition 1, the condition $$\sigma(A+B/4)\le0$$ says that the traceless part of the weight, which is the only part that distinguishes the weight from the trace, has an energy component of the sign opposite to that of the trace. The trace weight $$w=g$$ lies on the edge of the cone, $$A/B=-\tfrac14$$, and it is an extreme ray in the sense that radiation, whose trace $$T=3p-\rho$$ vanishes, contributes exactly zero. For the trace weight the sign flips only when matter leaves $$\mathcal T$$ on the stiff side, $$p>\rho/3$$, which gives $$\rho_\Lambda=\tfrac14\langle3p-\rho\rangle>0$$ for the kinetic energy of a scalar field ($$w=1$$ contributes $$+\tfrac12\rho$$). This is the loophole of [[KP 2014]](#ref-kp2). It is closed on the past light cone of an observer today by the smallness of the volume near the big bang: with an epoch of kination ($$w=1$$, density equal to that of radiation at $$a_{\rm kin}$$ and rising as $$a^{-6}$$ into the past) below $$a_{\rm kin}=10^{-9}$$ the trace member on $$J^-(p)$$ is unchanged at four decimal places, $$-5.96\,\rho_\Lambda^{\rm obs}$$, and for $$a_{\rm kin}=10^{-4}$$, a value that nucleosynthesis excludes, it is $$-5.89\,\rho_\Lambda^{\rm obs}$$ (our computation, with the expansion history modified accordingly).

A positive $$\rho_\Lambda$$ in the cone $$\mathcal T$$ requires $$\sigma(A_s+B_s/4)>0$$ for some sector on a set of positive measure. Constant weights make this a statement about one ratio. With $$A/B=c$$ and $$B>0$$, pure radiation gives $$\rho_\Lambda=\tfrac13(4c+1)\langle\rho_r\rangle$$ and pure dust gives $$c\langle\rho_m\rangle$$. A positive value from radiation requires $$c>-\tfrac14$$, and from dust $$c>0$$. The weights of the dust-blind kind, $$A=0$$, $$B>0$$, give $$\rho_\Lambda=\langle p\rangle$$ and so are controlled by the radiation pressure alone.

<div class="remark" markdown="1">
**Remark 1** (sector-dependent weights). If cold dark matter shares the trace weight of the vacuum, $$A_c=-B_0/4$$, it contributes $$-\tfrac14\langle\rho_c\rangle$$ to the numerator. A positive $$\rho_\Lambda$$ then requires baryons with a weight $$A_b$$ satisfying $$A_b\langle\rho_b\rangle>\tfrac14B_0\langle\rho_c\rangle$$, that is $$A_b/B_0>\rho_c/(4\rho_b)=\Omega_ch^2/(4\,\Omega_bh^2)=1.34$$ with the Planck values of section 1, where $$B_0$$ is taken constant and dust is the only matter. The cone alone does not forbid this; the weight of baryons would be a number to be supplied, at more than 1.3 times the vacuum's trace weight.
</div>

## 4. Weights built from the causal structure

The cone is defined in FRW. A weight constructed from the causal structure can be treated without any assumption on the background, using only energy conditions.

<div class="theorem" markdown="1">
**Theorem 2** (causal vectors). Let $$(M,g)$$ be any spacetime and $$T_{ab}$$ the stress tensor of matter, and let the vacuum have $$T^\Lambda_{ab}=-\rho_\Lambda g_{ab}$$ with $$\rho_\Lambda$$ constant. Suppose the constraint (3) is formed with a single weight $$w^{ab}=\int d\nu_x(v)\,v^av^b$$, where at each $$x\in R$$ the measure $$\nu_x$$ is positive and supported on causal vectors $$v$$ ($$g(v,v)\le0$$), and the matter obeys the WEC. If $$\int d\mu\,d\nu\,g(v,v)<0$$, then $$\rho_\Lambda\le0$$. If $$g(v,v)=0$$ for $$\mu\otimes\nu$$-almost every $$v$$, the constraint contains no $$\rho_\Lambda$$ and is solvable only if $$T_{ab}v^av^b=0$$ almost everywhere.

The same conclusions hold for $$w^{ab}=\int d\nu\,\tfrac12(\xi^aN^b+\xi^bN^a)$$ with $$\xi,N$$ future-directed causal vectors, if the matter obeys the DEC.
</div>

<div class="proof" markdown="1">
*Proof.* The constraint reads $$\int d\mu\,d\nu\,[T(v,v)-\rho_\Lambda g(v,v)]=0$$. By the WEC, $$T(v,v)\ge0$$, and $$g(v,v)\le0$$ by assumption, so $$\rho_\Lambda\int g(v,v)=\int T(v,v)\ge0$$. When $$\int g(v,v)<0$$ this gives $$\rho_\Lambda\le0$$. When $$g(v,v)=0$$ almost everywhere, the constraint is $$\int T(v,v)=0$$. For the mixed weight, $$T(\xi,N)\ge0$$: under the DEC the vector $$j^a=-T^a{}_bN^b$$ is future-directed causal, and two future-directed causal vectors have $$g(\xi,j)\le0$$, which is $$-T(\xi,N)\le0$$; also $$g(\xi,N)\le0$$. The same argument applies. $$\square$$
</div>

The weights covered include the tangent to an observer's worldline, the generators of a light cone, the unit normal to a family of hypersurfaces, the four-velocity of the matter, and the flux of energy through a causal boundary measured by a future-directed timelike Killing field. In FRW with matter in $$\mathcal T$$, Theorem 2 is a special case of Theorem 1(a): a causal $$v$$ has $$A=(v\cdot u)^2\ge-g(v,v)=-B\ge0$$, so $$A/B\le-1\le-\tfrac14$$. Theorem 2 is stronger in that it needs neither FRW nor $$p\ge0$$: the energy conditions suffice, with the sign of the denominator fixed by causality instead of by an assumption.

## 5. Sequestering with several matter sectors

The trace weight arises from an action principle. In the construction of [[KP 2013]](#ref-kp1), [[KP 2014]](#ref-kp2) the matter fields see the metric $$\lambda^2g_{ab}$$, where $$\lambda$$ is a spacetime constant that is varied, and a term $$\sigma(\Lambda/\lambda^4\mu^4)$$, with $$\mu$$ a fixed mass scale, makes $$\Lambda$$ a global variable as well. To see whether sector-dependent rescalings change the sign statement, we generalise this construction to several sectors.

Let $$S=\int d^4x\sqrt{-g}\,\big(\tfrac12M_P^2R-\Lambda\big)+\sum_sS_s[\lambda_s^2g,\psi_s]+\sigma\big(\Lambda/(\mu^4\prod_s\lambda_s^{4n_s})\big)$$, where $$M_P$$ is the reduced Planck mass, $$S_s$$ is the action of the fields $$\psi_s$$ of sector $$s$$, which couple to the metric $$\lambda_s^2g$$, the numbers $$n_s\ge0$$ are fixed weights, $$\Lambda$$ and the $$\lambda_s>0$$ are spacetime constants, and $$\sigma$$ is a function of one variable. We write $$V_4=\int\sqrt{-g}\,d^4x$$, which we take finite, and $$\langle X\rangle=V_4^{-1}\int\sqrt{-g}\,X\,d^4x$$. For the stress tensor $$T_s^{ab}=(2/\sqrt{-g})\,\delta S_s/\delta g_{ab}$$ sourcing $$g_{ab}$$, we write $$T_s=g_{ab}T_s^{ab}$$.

<div class="proposition" markdown="1">
**Proposition 3** (sector-wise sequestering; derivation within the model). On solutions of the field equations of this action, $$\langle T_s\rangle=4n_s\Lambda$$ for every sector $$s$$. Let $$V$$ be the sum of the vacuum energy densities of all sectors, so that $$T_s$$ contains $$-4V_s$$, and let $$T^{\rm nv}_s=T_s+4V_s$$ be the trace without the vacuum part. The net vacuum energy density $$\rho_\Lambda=\Lambda+V$$ is

$$
\rho_\Lambda=\tfrac14\sum_s\langle T_s^{\rm nv}\rangle+\Big(1-\sum_sn_s\Big)\Lambda. \tag{5}
$$

If $$\sum_sn_s=1$$, then $$\rho_\Lambda=\tfrac14\langle T^{\rm nv}\rangle$$ is independent of $$V$$, it is the trace-weight case of (3), and $$\rho_\Lambda\le0$$ whenever the matter lies in $$\mathcal T$$ over the whole four-volume. If $$\sum_sn_s\ne1$$, then $$\rho_\Lambda$$ depends on $$V$$ with the coefficient $$1-1/\sum_sn_s$$.
</div>

<div class="proof" markdown="1">
*Proof.* Varying $$\Lambda$$ gives $$-V_4+\sigma'(x)/(\mu^4\prod\lambda_s^{4n_s})=0$$, where $$x$$ is the argument of $$\sigma$$; hence $$x\,\sigma'(x)=\Lambda V_4$$. For the variation with respect to $$\lambda_s$$ write $$\tilde g=\lambda_s^2g$$, so that $$\delta\tilde g_{ab}=2\lambda_sg_{ab}\,\delta\lambda_s$$ and $$\partial S_s/\partial\lambda_s=\lambda_s^{-1}\int\sqrt{-\tilde g}\,\tilde T_s$$, where $$\tilde T_s$$ is the trace of the stress tensor of sector $$s$$ taken with $$\tilde g$$, and $$\sqrt{-\tilde g}=\lambda_s^4\sqrt{-g}$$. The derivative of the $$\sigma$$ term is $$-4n_s\,x\sigma'(x)/\lambda_s$$. Equating the two, $$\lambda_s^4\int\sqrt{-g}\,\tilde T_s=4n_s\Lambda V_4$$. The source of $$g_{ab}$$ is the stress tensor $$T_s^{ab}=(2/\sqrt{-g})\,\delta S_s/\delta g_{ab}$$, and since $$g_{ab}\,\delta/\delta g_{ab}=\tilde g_{ab}\,\delta/\delta\tilde g_{ab}$$ for a constant rescaling, its trace is $$T_s=\lambda_s^4\tilde T_s$$ (the factor $$\lambda_s^4$$ is the ratio of the volume elements). Hence $$\langle T_s\rangle=4n_s\Lambda$$. Summing over $$s$$ and using $$T_s=T_s^{\rm nv}-4V_s$$ gives $$\sum_s\langle T_s^{\rm nv}\rangle-4V=4\Lambda\sum_sn_s$$, which rearranges to (5) with $$\rho_\Lambda=\Lambda+V$$. The remaining statements follow from (5) and Theorem 1(a) with the trace weight; the coefficient of $$V$$ follows from $$\Lambda=(\tfrac14\sum\langle T^{\rm nv}_s\rangle-V)/\sum n_s$$. $$\square$$
</div>

Two points about the scope. The individual equations $$\langle T_s\rangle=4n_s\Lambda$$ are $$N$$ conditions for $$N$$ sectors; they determine the constants $$\lambda_s$$ and $$\Lambda$$ jointly, and (5) uses only their sum. The positivity theorem uses the matter content over the whole four-volume, which in this construction includes the final collapse. The loophole of kinetic-dominated matter near the crunch is therefore inside the region, not outside it, and its effect depends on the model of the final stage.

<div class="example" markdown="1">
**Example 2** (labelled sectors). Let cold dark matter and baryons carry different weights $$n_c\ne n_b$$ with $$n_c+n_b=1$$. Equation (5) gives $$\rho_\Lambda=\tfrac14\langle T^{\rm nv}_c+T^{\rm nv}_b\rangle=-\tfrac14\langle\rho_c+\rho_b\rangle$$ for dust. The split between $$n_c$$ and $$n_b$$ does not appear in the net vacuum energy, because the sectors enter through their traces and the trace of dust is $$-\rho$$ whatever its composition.
</div>

## 6. Drift along the worldline

Theorems 1 and 2 are statements at a point or about an average with any positive measure. For the past light cone of an observer there is a second statement, about how the answer changes with the observer's epoch.

Let $$p$$ be an event at conformal time $$\eta_p$$, and $$R=J^-(p)$$. Let $$s=\eta_p-\eta$$ be the radial distance in conformal units from the apex, and write the weighted measure of the constraint as $$B\,d\mu=b(\eta)\,\kappa(\chi)\,d\eta\,d^3x$$, where $$\chi=\lvert\mathbf x-\mathbf x_p\rvert$$, $$b\ge0$$ depends on time only, and $$\kappa\ge0$$ depends on the distance from the apex only. The four-volume, for instance, has $$b=a^4$$, $$\kappa=1$$. Integrating over the sphere at distance $$\chi\le s$$ gives the kernel $$K(s)=\int_0^s4\pi\chi^2\kappa(\chi)\,d\chi$$, with $$K(0)=0$$. For a constant ratio $$A/B=c$$ the prediction (4) is

$$
\rho_\Lambda(\eta_p)=\frac{\int_0^{\eta_p}b(\eta)\,\sigma(\eta)\,K(\eta_p-\eta)\,d\eta}{\int_0^{\eta_p}b(\eta)\,K(\eta_p-\eta)\,d\eta},\qquad \sigma=c\,(\rho+p)+p. \tag{6}
$$

The observer's epoch enters through the upper limit of the integrals and through $$K$$.

<div class="theorem" markdown="1">
**Theorem 3** (drift). Assume the setting of Theorem 1 with a single weight, $$B>0$$ and constant $$A/B=c\ge0$$. Assume further the following.

(H4) $$R=J^-(p)$$, and the measure has the factorised form above with $$b\ge0$$.

(H5) $$K$$ is continuous, $$K>0$$ on $$(0,\infty)$$, $$K(0)=0$$, $$K$$ is differentiable and $$\ln K$$ is concave.

(H6) $$\rho$$ and $$p$$ are non-increasing in $$\eta$$ (for example, all components have constant $$w_s\in[0,\tfrac13]$$ and the universe expands), so that $$\sigma$$ is non-increasing in $$\eta$$.

(H7) The integrals in (6) and the integral of $$b\,\sigma\,\lvert K'\rvert$$ converge, locally uniformly in $$\eta_p$$.

Then $$\rho_\Lambda(\eta_p)$$ is non-increasing in $$\eta_p$$, and strictly decreasing if $$\sigma$$ is not constant on the support of the weight and $$\ln K$$ is strictly concave. Along each comoving worldline the prediction, read as a vacuum energy density, has $$w_{\rm eff}\ge-1$$.
</div>

<div class="proof" markdown="1">
*Proof.* The idea is that raising $$\eta_p$$ moves the weight of the average towards early times, where $$\sigma$$ is large. Write $$N(\eta_p)$$ and $$D(\eta_p)$$ for the numerator and denominator of (6). Because $$K(0)=0$$, the boundary term in the derivative with respect to the upper limit vanishes, and by (H7), $$N'=\int_0^{\eta_p}b\sigma K'(\eta_p-\eta)\,d\eta$$, and $$D'=\int_0^{\eta_p}bK'(\eta_p-\eta)\,d\eta$$. Let $$\langle\cdot\rangle_K$$ be the average over $$\eta\in(0,\eta_p)$$ with the probability density proportional to $$b(\eta)K(\eta_p-\eta)$$, and put $$\ell(\eta)=K'/K$$ evaluated at $$\eta_p-\eta$$. Then $$\frac{d}{d\eta_p}\ln\rho_\Lambda=\frac{\langle\sigma\ell\rangle_K}{\langle\sigma\rangle_K}-\langle\ell\rangle_K=\frac{\mathrm{Cov}_K(\sigma,\ell)}{\langle\sigma\rangle_K}$$. Here $$\ell=(\ln K)'$$ is non-increasing in $$s$$ by (H5), so it is non-decreasing in $$\eta$$, while $$\sigma$$ is non-increasing in $$\eta$$ by (H6). Two functions that are monotone in opposite directions have a non-positive covariance under any probability measure: for independent copies $$\eta,\eta'$$ from the measure, $$\mathrm{Cov}(\sigma,\ell)=\tfrac12E[(\sigma(\eta)-\sigma(\eta'))(\ell(\eta)-\ell(\eta'))]\le0$$ since the two factors have opposite signs. Equality needs the product to vanish almost surely, which for strictly monotone $$\ell$$ means $$\sigma$$ is constant. Hence $$d\ln\rho_\Lambda/d\eta_p\le0$$, and since $$\eta_p$$ increases with the proper time of a comoving observer, $$d\rho_\Lambda/dt\le0$$, which is $$w_{\rm eff}\ge-1$$. $$\square$$
</div>

The proof is Chebyshev's association inequality, and the content of the theorem is the structure (6), in which one monotone function of time, $$\sigma$$, is paired with the monotone log-derivative of the kernel. The hypothesis on $$\sigma$$ is where the cone enters: for $$c\ge0$$, $$\sigma=c(\rho+p)+p$$ is a sum of non-increasing functions. For $$-\tfrac14<c<0$$ the dust contribution to $$\sigma$$ is negative and $$\sigma$$ need not be monotone, and the lemma does not apply. In that window the radiation channel is the positive part.

The consequence is twofold. No constraint of this form assigns one constant to all observers: the value at $$\eta_p$$ is the value for an observer at that epoch, and it is larger for earlier observers. If instead the prediction is read as a running vacuum energy density, then $$\dot\rho_\Lambda\le0$$ means energy is passing from the vacuum to matter, and $$w_{\rm eff}\ge-1$$ at all times, so the predicted dark energy cannot show the crossing of $$-1$$ that the DESI fits with $$w_0+w_a<-1$$ describe. One caveat concerns the word "inferred". Observational reconstructions of $$w$$ assume that matter is conserved and fix its density at an early epoch, and a vacuum energy that decreases while matter increases changes the reconstructed history. With a late-time normalisation of the matter density, an interacting vacuum that feeds matter can be reconstructed as phantom [[DCK]](#ref-dck). The theorem constrains the vacuum energy density itself; the page [The sign of energy exchange between vacuum and matter](/quantum-gravity/vacuum-matter-sign/) treats the reconstruction.

<div class="example" markdown="1">
**Example 3** (the pressure weight on $$\Lambda$$CDM). Take the dust-blind weight $$c=0$$, $$B=1$$, with $$\mu$$ the four-volume, so that $$b=a^4$$, $$K=\tfrac43\pi s^3$$, $$\sigma=p$$. Then $$\rho_\Lambda=\tfrac13\rho_{r,0}\langle a^{-4}\rangle$$, where $$\rho_{r,0}$$ is today's radiation density and $$\langle a^{-4}\rangle$$ is the average of $$a^{-4}$$ over $$J^-(p)$$ with the four-volume weight. Our computation for the $$\Lambda$$CDM background of section 1 gives $$H_0\eta_0=3.179$$ and $$\langle a^{-4}\rangle=870$$ for an observer today, hence $$\rho_\Lambda=0.039\,\rho_\Lambda^{\rm obs}$$. To match $$\rho_\Lambda^{\rm obs}$$ the weighted average of $$a^{-4}$$ would have to equal $$3\Omega_\Lambda/\Omega_r=2.2\times10^{4}$$, which a delta-function weight achieves at $$a_*=0.082$$, $$z_*=11.2$$. The prediction as a function of the observer's scale factor $$a_p$$, relative to its value today, is:

| $$a_p$$ | 0.4 | 0.6 | 0.8 | 1 | 1.25 | 2 | 5 |
|---|---|---|---|---|---|---|---|
| $$\rho_\Lambda(a_p)/\rho_\Lambda(1)$$ | 21.3 | 4.87 | 1.90 | 1 | 0.573 | 0.228 | 0.073 |

The logarithmic slope at $$a_p=1$$ is $$d\ln\rho_\Lambda/d\ln a_p=-2.69$$, an effective equation of state $$w_{\rm eff}=-0.10$$, far from $$w=-1$$. The monotone fall is Theorem 3. As a check of the theorem, we evaluated (6) for 60 random kernels, with $$c\in\{0,\tfrac13,\tfrac7{10},1\}$$, power-law and saturating $$K$$, and weights $$b\propto a^{4-m}$$ with $$m\in[-2,0.9]$$ and with oscillating modulation, at 25 epochs from $$a_p=0.05$$ to 20. The prediction fell monotonically in every case.
</div>

## 7. Constraints linear in the matter Lagrangian

A linear constraint need not be built from the stress tensor. A condition on the average of the Lagrangian density is linear in the Lagrangian, and a condition of this kind is proposed in [[CR]](#ref-cr). With the vacuum contributing $$L=-\rho_\Lambda$$, it reads $$\rho_\Lambda=\langle L_m\rangle$$ with $$L_m$$ the on-shell matter Lagrangian density. This constraint is outside Proposition 1, because the on-shell value of $$L_m$$ is not a function of $$T_{ab}$$. For a perfect fluid, the value is $$p$$ in one description and $$-\rho$$ in another. For a collection of particles with fixed structure, $$\langle L_m\rangle=\langle T\rangle$$ [[AS]](#ref-as), which for dust is $$-\rho$$. The two conventions give $$\rho_\Lambda=+\tfrac13\langle\rho_r\rangle$$ and $$\rho_\Lambda=-\langle\rho\rangle$$ for the same radiation and dust, opposite in sign for the dust-and-radiation mixture of Example 1 ($$+0.039$$ against $$-24.0$$ on $$J^-(p)$$), so the fluid description does not determine the answer.

A quantum field theory fixes the Lagrangian value of a state. The following identity is the tool.

<div class="proposition" markdown="1">
**Proposition 4** (Feynman–Hellmann for the Lagrangian). Let the action be $$S=\lambda S_1+S_2$$ with $$S_2$$ independent of $$\lambda$$, and let $$\lvert\Psi\rangle$$ be a normalised energy eigenstate with energy $$E(\lambda)$$, with all other parameters and conserved charges fixed. Then the spatial integral of the expectation value of the Lagrangian density is $$\int d^3x\,\langle L\rangle=-\lambda\,\partial E/\partial\lambda+\int d^3x\,\langle L_2\rangle$$.
</div>

<div class="proof" markdown="1">
*Proof.* For a Hamiltonian $$H(q,p;\lambda)$$ obtained by Legendre transform of a Lagrangian $$L(q,\dot q;\lambda)$$, $$\partial H/\partial\lambda\vert_{q,p}=-\partial L/\partial\lambda\vert_{q,\dot q}$$. The Feynman–Hellmann theorem gives $$\partial E/\partial\lambda=\langle\partial H/\partial\lambda\rangle$$, and here $$\partial L/\partial\lambda=L_1$$ with $$L=\lambda L_1+L_2$$. So $$\langle\lambda L_1\rangle=-\lambda\,\partial E/\partial\lambda$$, and adding $$\langle L_2\rangle$$ gives the claim. $$\square$$
</div>

Three cases follow. For a free field, $$\lambda$$ multiplies the whole action, $$L_2=0$$, and $$E$$ does not depend on $$\lambda$$ because rescaling the action does not change the frequencies of the modes; the single-particle states and the vacuum then have $$\langle L\rangle=0$$. One can see this directly: for a mode of a free scalar with $$\omega^2=k^2+m^2$$, the Lagrangian $$\tfrac12(\dot\phi^2-k^2\phi^2-m^2\phi^2)$$ has vacuum expectation value $$\tfrac12(\omega^2-k^2-m^2)/(2\omega)=0$$, while the energy density is $$\tfrac12\omega$$ per mode. The Lagrangian does not see the zero-point energy of free fields. For a classical static scalar soliton, where the Lagrangian is the negative of the energy because the kinetic terms vanish, the value is $$-1$$ in units of $$E$$, and $$E\propto\lambda$$ is consistent. For a gauge theory with the gauge-field action multiplied by $$\lambda=1/g^2(M)$$ at an ultraviolet scale $$M$$, and the matter Lagrangian assumed to vanish on shell by the equations of motion (matrix elements of equation-of-motion operators between physical states vanish), a hadron mass is proportional to the dynamically generated scale $$\Lambda_{\rm QCD}$$. At one loop, $$\Lambda_{\rm QCD}=M\exp[-2\pi/(b_3\alpha_s(M))]$$ with $$b_3=11-\tfrac23n_f=9$$ the coefficient for the three light flavours (the derivative with respect to $$1/\alpha_s(M)$$ does not depend on the thresholds between $$M$$ and the hadronic scale at one loop), and rescaling $$1/\alpha_s(M)\to\lambda/\alpha_s(M)$$ gives

$$
\frac{\int\langle L\rangle}{E}=\frac{2\pi}{9\,\alpha_s(M)}\qquad(\text{chiral limit, one loop}). \tag{7}
$$

For the nucleon this is positive, and it grows with the scale $$M$$: with two-loop running from $$\alpha_s(M_Z)=0.118$$ as input, it is $$5.9$$ at $$M=M_Z$$, $$6.5$$ at $$172.5$$ GeV, $$31$$ at $$10^{16}$$ GeV and $$36$$ at the reduced Planck mass (our computation). The ratio is not an observable of the low-energy theory, since it depends on where the multiplier acts, and the quark-mass part of the nucleon mass, of order ten per cent, adds a correction.

Combining the cases, a constraint $$\rho_\Lambda=\langle L_m\rangle$$ sees baryons with weight $$\ell_b\sim5$$ to $$36$$, the value depending on a scale $$M$$ that has to be chosen, and cold dark matter with weight $$0$$ if it is an elementary particle and $$-1$$ if it is a static soliton. On the past light cone with $$\langle\rho_b\rangle=3.7\,\rho_\Lambda^{\rm obs}$$ and elementary dark matter, the value is $$\ell_b\times3.7\,\rho_\Lambda^{\rm obs}$$, that is of order $$20$$ to $$130\,\rho_\Lambda^{\rm obs}$$, with the number set by $$M$$. The sign can be positive. The magnitude contains one free number.

The second difficulty is that the multiplier is not radiatively stable, by the following argument, which we give as a heuristic. A multiplier that acts as an overall factor on the matter action leaves free-field vacuum energy untouched, as the mode sum above shows, so the zero-point energy of free fields shifts the net vacuum energy without shifting the right side of the constraint. Oda [[Oda 1]](#ref-oda1) found the corresponding instability in the nonlocal constraint of [[CR]](#ref-cr) by a one-loop calculation. By contrast, the metric-rescaling multiplier of section 5 couples to the trace of the stress tensor, and the zero-point energy contributes $$-4V$$ to the trace, which is why (5) is independent of $$V$$ when $$\sum n_s=1$$. In this family, sensitivity to the sector content and radiative stability do not occur together.

## 8. What the theorems do not exclude

Each item lists the hypothesis that is dropped and what then holds.

Weights with a positive spatial-stress component. If $$\sigma(A+B/4)>0$$ on a set of positive measure, Theorem 1 gives no sign. Such a weight has a traceless part with positive energy component in the fluid frame, such as the spatial projector or a boosted spatial stress, and it needs a preferred frame $$u$$ or a preferred direction. Two classes are quantified. For the dust-blind weight ($$A=0$$) the value is $$\langle p\rangle$$, and matching $$\rho_\Lambda^{\rm obs}$$ needs a weighted average of $$a^{-4}$$ equal to $$3\Omega_\Lambda/\Omega_r$$, which a sharply peaked weight at redshift $$z_*=11.2$$ delivers (Example 3). For a weight tilted by rapidity $$\zeta$$ to the fluid ($$A=\sinh^2\zeta$$, $$B=1$$) the value is $$\approx\sinh^2\zeta\,\langle\rho_m\rangle$$, so matching needs $$\sinh^2\zeta\approx\rho_\Lambda^{\rm obs}/\langle\rho_m\rangle=0.042$$ on the past cone of Example 1. The CMB dipole of amplitude $$3362\,\mu$$K [[Planck I]](#ref-planck1) corresponds to a velocity $$\beta\approx1.2\times10^{-3}$$ relative to the CMB frame, for which $$\sinh^2\zeta\approx1.5\times10^{-6}$$.

Varying weights. Theorem 1 holds pointwise, so a ratio $$A/B$$ that varies in spacetime does not enlarge the class of sign-definite weights. It does permit a family of positive values with a free measure. For the dust-blind kernel $$B\propto a^{-n}$$ on the four-volume of $$J^-(p)$$, the average of $$p$$ converges at the big bang only for $$n<1$$. With $$n=1$$ the integral grows logarithmically with the small-scale cutoff, and the value is $$55\,\rho_\Lambda^{\rm obs}$$ for a cutoff at the scale factor of a TeV-temperature universe ($$a_c=2.3\times10^{-16}$$) and $$125\,\rho_\Lambda^{\rm obs}$$ for a cutoff at the Planck density ($$a_c=2\times10^{-32}$$) (our computation). An exponent $$n=0.670$$ gives $$1.0\,\rho_\Lambda^{\rm obs}$$; it is a fitted value, and Theorem 3 applies: the prediction has $$w_{\rm eff}=-0.18$$ at the present epoch and is $$1.8$$ times larger for an observer at $$a_p=0.8$$.

Stiff matter and negative pressure. Outside $$p\in[0,\rho/3]$$ the cone argument fails. The trace weight over a collapsing universe is positive when the kinetic energy of a scalar field dominates near the crunch [[KP 2014]](#ref-kp2), and for a region that extends into such a future the sign is not fixed by Theorem 1. On the past cone of an observer today, an early kinetic epoch does not change the sign (section 3). Matter with $$p<0$$, such as a scalar field on a potential plateau, enters the trace weight as $$\tfrac14(3p-\rho)<0$$, which pushes the sign further negative.

Regions that are not past-causal. A constraint averaged over the collapsing patches of a halo model [[Lombriser]](#ref-lombriser) has a measure that is not of the form considered in Theorem 3. We have not recomputed that construction. Theorem 1 applies to any region on which its hypotheses hold.

Non-linear constraints. Theorem 1 uses linearity in the stress tensor. A constraint quadratic in the total stress tensor, such as $$\langle(\mathrm{tr}\,T_{\rm tot})^2-2\,T_{\rm tot}^{ab}T^{\rm tot}_{ab}\rangle_\mu=0$$ with $$T^{\rm tot}_{ab}=T_{ab}-\rho_\Lambda g_{ab}$$, is a scalar condition with no preferred frame and no action behind it. It reduces to $$8\rho_\Lambda^2-4\langle s\rangle\rho_\Lambda+\langle s^2-2q\rangle=0$$ with $$s=3p-\rho$$ and $$q=\rho^2+3p^2$$ (we confirmed the expansion symbolically). For matter in $$\mathcal T$$, $$s^2-2q=3p^2-6p\rho-\rho^2<0$$ whenever $$\rho>0$$, so the product of the roots is negative and there is exactly one positive root, for every positive measure and every region. For dust it is $$\rho_\Lambda=\tfrac14(\sqrt3-1)\rho$$. The magnitude is tied to $$\langle\rho_m\rangle$$, and no radiative stability has been shown. The example shows only that the sign statement of Theorem 1 depends on linearity. The constructions of [[SB]](#ref-sb) and [[Oda 2]](#ref-oda2) are not covered by the theorems above; we have not checked whether they can be put in the form (3).

Lagrangian-linear constraints. They can give a positive value (section 7), with the free number $$M$$ and without radiative stability.

The hypotheses (H1) to (H3) cannot be dropped: positive spatial stress, matter outside the cone, and constraints linear in the Lagrangian each give a positive $$\rho_\Lambda$$ in the examples above, so the exclusion is not a statement about all linear global constraints.

## 9. Open points

The classification of weights is exhaustive for a perfect fluid on FRW. Inhomogeneities were not included, so a weight that responds to the anisotropic stress or to the velocity field of structure is not covered; it would add terms of the order of the density contrast to the numerator of (4). The drift theorem assumes a separable measure; a measure that couples $$\eta$$ and $$\chi$$ was not treated. For $$-\tfrac14<A/B<0$$ the monotonicity of $$\sigma$$ fails and Theorem 3 does not apply, and we have no replacement statement for that window. Whether a sequestering mechanism with a single global variable can be made radiatively stable and still distinguish sectors by anything other than the trace is not settled by the identity (5). The relation between the vacuum-energy flow implied by Theorem 3 and the reconstruction of the dark-energy equation of state from distance data depends on how matter is normalised; the page on [the sign of energy exchange between vacuum and matter](/quantum-gravity/vacuum-matter-sign/) gives the cases where the sign is fixed. The structural reason that a value of $$\Lambda$$ is not selected by such constraints in general is discussed on [the cosmological constant and the dark-matter mass as free parameters](/quantum-gravity/dark-sector-moduli/), and the place of these results among other approaches is on the [landscape page](/quantum-gravity/landscape/).

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
