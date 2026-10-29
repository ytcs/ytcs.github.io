---
title: Kernel-scale kinematics of a single realisation
date: 2026-10-29
category: substrate
order: 6
description: On a Poisson sprinkling, a layer-weighted d'Alembertian of kernel scale has a standard deviation that exceeds the signal on a smooth wave packet by a factor growing like the cube of the packet width in units of the kernel scale, with an explicit lower bound from the first chaos of the Poisson process. The cocompact orbit lattice of an anisotropic form reproduces the mean dictionary with retuned weights, yet on each realisation the operator maps every finitely supported field out of the square-summable functions, and numerically its errors are as large as the Poisson ones.
status: The Poisson variance inequality is cited and its add-one cost is derived; the large-width constants are an asymptotic derivation checked numerically; the facts about the orbit lattice are proved; single-realisation errors, number variances and Palm laws are numerical, with sizes stated.
excludes: Layer-weighted kernels at the kernel scale, on a Poisson sprinkling or on the cocompact orbit lattice of an anisotropic integral form, as operators whose individual realisations reproduce the continuum d'Alembertian on smooth fields; a lattice-invariant smearing of the kernel as a remedy on the orbit lattice.
---

A causal-set d'Alembertian is a sum over the causal past of a point, with a weight that depends on the number $$m$$ of points in the causal interval between the two. Averaged over a Poisson sprinkling of density $$\ell^{-4}$$, it reproduces the d'Alembertian $$\Box\phi$$ of a smooth field $$\phi$$ [[Benincasa–Dowker]](#ref-bd) [[Aslanbeigi–Saravani–Sorkin]](#ref-ass). The question here is whether one realisation of the point set, with the kernel at the scale $$\ell$$ of the discreteness itself, computes $$\Box\phi$$ when $$\phi$$ varies on a scale $$L\gg\ell$$. For a Poisson sprinkling the answer is no: the standard deviation of the operator on a Gaussian packet is bounded below by a multiple of $$(L/\ell)^3\,\lvert \Box\phi\rvert $$, with a constant computed below, and a Monte Carlo estimate shows the standard deviation itself reaching $$3\cdot10^2$$ to $$8\cdot10^3$$ times $$\lvert \Box\phi\rvert $$ at $$L=5\ell$$, depending on the kernel. This is the quantitative form of Sorkin's observation that Lorentz-invariant discreteness needs a non-locality scale $$\ell_k\gg\ell$$ [[Sorkin]](#ref-sorkin). One might hope a lattice does better, because a lattice has no Poisson randomness. For the cocompact orbit lattice of the anisotropic form $$7x_0^2-x_1^2-x_2^2-x_3^2$$, which is Poincaré invariant in law, the weights can be retuned to the lattice's own interval statistics so that the mean reproduces $$\Box\phi$$, but a single realisation fails for a structural reason: its kernel is invariant under a Zariski-dense group, and by [Exact discrete Lorentz symmetry cannot carry propagation](/quantum-gravity/discrete-lorentz-group/) the operator maps every non-zero finitely supported field out of $$\ell^2$$. Numerically the errors are of the same size as for Poisson. The conditions on the weights that make the mean work are derived on [Mixed-Poisson and signed substrates and the four-dimensional d'Alembertian](/quantum-gravity/signed-substrates/); the invariant random lattices themselves are described on [Lorentz-invariant point sets](/quantum-gravity/invariant-point-sets/).

The lower bound for Poisson uses the first term of the chaos expansion of the variance of a Poisson functional [[Last–Penrose]](#ref-lp). The statement that the Poisson variance needs a mesoscale is due to Sorkin [[Sorkin]](#ref-sorkin), who gave the qualitative argument; Saravani and Aslanbeigi studied number fluctuations of Poisson sprinklings and of two-dimensional Lorentzian lattices [[Saravani–Aslanbeigi]](#ref-sa). What is added here is the explicit constant for the 4D kernels, the proofs for the orbit lattice, and the comparison of the two.

## 1. Layer operators at the kernel scale

Let $$\Phi$$ be a locally finite set in $$\mathbb R^{1,3}$$ and $$x\in\Phi$$. For $$y\prec x$$ in the causal past, let $$m(y,x)$$ be the number of points of $$\Phi$$ in the open interval between $$y$$ and $$x$$. A *layer operator* with weights $$(w_0,\dots,w_M)$$, local coefficient $$a$$ and normalisation $$N$$ acts on a function $$\phi$$ by

$$
(B\phi)(x)=N\Big[a\,\phi(x)+\sum_{y\in\Phi,\ y\prec x}w_{m(y,x)}\,\phi(y)\Big],\qquad w_m=0\ \ (m>M). \tag{1}
$$

For a Poisson process of unit density the interval between $$y$$ and $$x$$ has volume $$V=\pi\tau^4/24$$ ($$\tau$$ the proper time), and $$m$$ is Poisson distributed with mean $$V$$. Hence $$\mathbb E\,(B\phi)(x)=N[a\phi(x)+\int_{y\prec x}\psi(V(y,x))\phi(y)\,d^4y]$$ with

$$
\psi(V)=\sum_mw_m\,e^{-V}\frac{V^m}{m!}. \tag{2}
$$

Put $$Q(s)=\sum_mw_m\Gamma(s+m)/m!$$, the Mellin transform of $$\psi$$. In this notation the infrared conditions derived by Aslanbeigi, Saravani and Sorkin [[Aslanbeigi–Saravani–Sorkin]](#ref-ass) amount to $$Q(\tfrac12)=Q(1)=Q(\tfrac32)=0$$ together with $$a=-3Q'(1)$$, with $$N$$ fixed by $$Q'(\tfrac32)$$ so that the mean tends to $$\Box\phi=\partial_t^2\phi-\nabla^2\phi$$ for smooth $$\phi$$; we use them as given and check them numerically below. The derivation is on the page [Mixed-Poisson and signed substrates and the four-dimensional d'Alembertian](/quantum-gravity/signed-substrates/). We use two weight sets. The Benincasa–Dowker weights are $$(1,-9,16,-8)$$ with $$\lvert N\rvert =4/\sqrt6$$. A five-layer set with $$Q(\tfrac12)=Q(1)=Q(\tfrac32)=Q(\tfrac54)=0$$ is $$(5,-85,280,-328,128)$$, also with $$\lvert N\rvert =4/\sqrt6$$. A direct evaluation gives $$Q(\tfrac12),Q(1),Q(\tfrac32)$$ equal to zero to $$10^{-29}$$ for both, $$Q'(1)=\tfrac13$$ (so $$a=-1$$) for both, and $$Q(\tfrac54)=0.0567$$ for the first, $$0$$ for the second.

The test field is a Gaussian packet,

$$
\phi(x)=e^{-\lvert x\rvert _{\rm E}^2/2L^2}\cos(p\cdot x),\qquad p=3/L, \tag{3}
$$

with the carrier along the time axis ("timelike", $$\cos pt$$) or along $$x_1$$ ("spacelike", $$\cos px_1$$). Lengths are in units of $$\ell=1$$. Then $$\Box\phi(0)=2/L^2-p^2=-7/L^2$$ (timelike) or $$2/L^2+p^2=11/L^2$$ (spacelike).

*Mean control.* The mean of (1) at a Poisson point, computed by quadrature in hyperbolic coordinates, divided by $$\Box\phi(0)$$ (with the overall sign of $$N$$ fixed so that the ratio tends to $$+1$$), is, for $$L=3,5,10,20,40$$:

| kernel and carrier | $$L=3$$ | $$L=5$$ | $$L=10$$ | $$L=20$$ | $$L=40$$ |
|---|---|---|---|---|---|
| Benincasa–Dowker, timelike | 0.82 | 1.15 | 1.12 | 1.05 | 1.02 |
| Benincasa–Dowker, spacelike | 0.48 | 0.67 | 0.86 | 0.95 | 0.98 |
| five-layer, timelike | 0.47 | 0.98 | 1.24 | 1.12 | 1.04 |
| five-layer, spacelike | 0.08 | 0.33 | 0.66 | 0.86 | 0.95 |

The mean dictionary works when $$L\gg\ell$$, with corrections that fall like a power of $$\ell/L$$. The question is the fluctuation around this mean.

## 2. The fluctuation of a Poisson sprinkling

We use one cited result. For a Poisson process $$\eta$$ with $$\sigma$$-finite intensity measure $$\lambda$$ and a square-integrable functional $$F(\eta)$$, put $$D_zF(\eta)=F(\eta+\delta_z)-F(\eta)$$ (the cost of adding the point $$z$$). The Wiener–Itô chaos expansion gives $$\mathrm{Var}\,F=\sum_{n\ge1}\frac1{n!}\int\lvert \mathbb E\,D^n_{z_1\dots z_n}F\rvert ^2\lambda^n(dz)$$ [[Last–Penrose]](#ref-lp), so that

$$
\mathrm{Var}\,F\ \ge\ \int\big\lvert \mathbb E\,D_zF\big\rvert ^2\lambda(dz). \tag{4}
$$

Take $$F=X(\Phi)$$, the bracket in (1) evaluated at the Palm point $$0$$ with $$\Phi$$ a unit-density Poisson process, so that $$B\phi(0)=NX$$.

<div class="proposition" markdown="1">

**Proposition 1** (the expected add-one cost). For the operator (1) at the Palm point $$0$$,

$$
\mathbb E\,D_zX=\mathbf 1_{\{z\prec0\}}\Big[\phi(z)\,\psi(V_z)+\int_{y\prec z}\phi(y)\,\psi'(V_y)\,d^4y\Big], \tag{5}
$$

where $$V_z=V(z,0)$$ and $$V_y=V(y,0)$$ are the volumes of the intervals ending at $$0$$.

</div>

The idea is that adding a point $$z$$ in the past of $$0$$ contributes its own term, and raises by one the interior count of every earlier point $$y$$ for which $$z$$ lies in the interval between $$y$$ and $$0$$.

<div class="proof" markdown="1">

*Proof.* If $$z\not\prec0$$, the point $$z$$ does not enter any term and does not lie in any interval ending at $$0$$, so $$D_zX=0$$. If $$z\prec0$$, the new point contributes $$w_{m(z,0)}\phi(z)$$, where $$m(z,0)$$ is the number of points of $$\Phi$$ in the interval between $$z$$ and $$0$$, Poisson with mean $$V_z$$; its expectation is $$\phi(z)\psi(V_z)$$. For each $$y\in\Phi$$ with $$y\prec z$$ (equivalently $$z\in I(y,0)$$), the count $$m(y,0)$$ increases by one and the term changes from $$w_m\phi(y)$$ to $$w_{m+1}\phi(y)$$. By the Mecke formula, given a point at $$y$$ the rest of $$\Phi$$ is Poisson, so the interior count excluding $$z$$ is Poisson with mean $$V_y$$, and $$\mathbb E[w_{M+1}-w_M]=\psi'(V_y)$$ for $$M\sim\mathrm{Poisson}(V_y)$$ (differentiate (2)). Campbell's formula gives the integral over $$y\prec z$$. $$\square$$

</div>

<div class="corollary" markdown="1">

**Corollary 2** (variance lower bound). With $$\Sigma(\phi)=\int_{z\prec0}\big[\phi(z)\psi(V_z)+\int_{y\prec z}\phi(y)\psi'(V_y)d^4y\big]^2d^4z$$,

$$
\mathrm{Var}\,\big(B\phi(0)\big)\ \ge\ N^2\,\Sigma(\phi). \tag{6}
$$

</div>

*Large-width evaluation (derivation under an approximation).* For $$L\gg\ell$$ the field varies little over the region $$\{y\prec z\}$$ that matters, because $$\psi'(V_y)$$ decays like $$e^{-V_y}$$. Replace $$\phi(y)$$ by $$\phi(z)$$ in (5). The region $$\{y\prec z\}$$, at fixed $$V_y$$, is a hyperbolic ball of radius $$d=\tfrac14\ln(V_y/V_z)$$ in the space of unit future vectors, with volume $$\pi(\sinh2d-2d)$$, and $$\tau^3d\tau=(6/\pi)\,dV$$. Then $$\mathbb E\,D_zX\approx\phi(z)h(V_z)$$ with

$$
h(V)=\psi(V)+6\int_V^\infty\psi'(V')\,\big(\sinh2d-2d\big)\,dV',\qquad d=\tfrac14\ln(V'/V). \tag{7}
$$

For the packet (3), $$\int_{H^3}e^{-\tau^2\cosh2\eta/L^2}\,d\mu(\eta)\approx\pi L^2/\tau^2$$ and the carrier contributes the average $$\langle\cos^2\rangle=\tfrac12$$, so $$\Sigma\approx3L^2I$$ with

$$
I=\int_0^\infty\frac{h(V)^2}{\tau(V)^2}\,dV. \tag{8}
$$

The function $$h$$ diverges only logarithmically as $$V\to0$$ (for the Benincasa–Dowker weights $$h=29.7,\,16.1,\,4.0$$ at $$V=10^{-6},10^{-4},10^{-2}$$), so (8) converges. Numerically, by quadrature of (7) and (8), $$I=9.25$$ for the Benincasa–Dowker weights and $$I=183.3$$ for the five-layer weights. Hence $$\mathrm{std}(B\phi(0))\ge\lvert N\rvert \sqrt{3I}\,L$$ at leading order, and with $$\lvert \Box\phi(0)\rvert =\kappa/L^2$$, $$\kappa=7$$ (timelike) or $$11$$ (spacelike),

$$
\frac{\mathrm{std}\,B\phi(0)}{\lvert \Box\phi(0)\rvert }\ \gtrsim\ \frac{\lvert N\rvert \sqrt{3I}}{\kappa}\,L^3\ =\ \begin{cases}1.23\,L^3\ (\text{timelike}),\ 0.78\,L^3\ (\text{spacelike}) & \text{Benincasa–Dowker},\\ 5.5\,L^3\ (\text{timelike}),\ 3.5\,L^3\ (\text{spacelike}) & \text{five-layer}.\end{cases} \tag{9}
$$

These are lower bounds up to the error made in replacing $$\phi(y)$$ by $$\phi(z)$$, which we do not bound. They say that a $$5\%$$ accuracy of a single realisation at kernel scale is out of reach for every $$L\gtrsim2\ell$$, and that the obstruction grows as $$L^3$$.

*Numerical check by sampling.* We sampled the Palm configuration of a unit-density Poisson process in the past cone of $$0$$, keeping the points whose interval volume to $$0$$ is at most 16 and whose Euclidean norm is at most $$3.3L$$ (the omitted points have weights supported on large $$m$$ with total probability of order $$10^{-3}$$ or less; the truncation is not quantified further). The interior counts $$m(y,0)$$ were computed exactly, and $$B\phi(0)$$ evaluated for the packet (3). The table gives $$\mathrm{std}(B\phi(0))/\lvert \Box\phi(0)\rvert $$ from the sample, followed by the approximate bound $$\lvert N\rvert \sqrt{\Sigma_{\rm approx}}/\lvert \Box\phi(0)\rvert $$ evaluated numerically from (7) with the exact $$\cos^2$$ and Gaussian factors (not the leading-order form (9)). The number of independent samples was $$200,\,200,\,120,\,80,\,30$$ for $$L=2,3,4,5,7$$; the relative uncertainty of a sample standard deviation is $$1/\sqrt{2(n-1)}$$, from 5% to 13%.

| $$L$$ | Benincasa–Dowker, timelike | Benincasa–Dowker, spacelike | five-layer, timelike | five-layer, spacelike |
|---|---|---|---|---|
| 2 | 17.0 / 8.3 | 9.8 / 6.2 | 298 / 36 | 172 / 26 |
| 3 | 88.6 / 28.9 | 51.8 / 21.6 | 1787 / 127 | 969 / 93 |
| 4 | 238 / 69 | 155 / 52 | 4981 / 304 | 3066 / 225 |
| 5 | 372 / 136 | 307 / 102 | 8016 / 596 | 6107 / 445 |
| 7 | 1317 / 377 | 758 / 283 | 23920 / 1652 | 15980 / 1240 |

Each entry is the sampled value over the approximate bound. The sampled standard deviation exceeds the approximate bound by a factor 1.6 to 3.5 for the Benincasa–Dowker weights and 6.5 to 16 for the five-layer weights, at every $$L$$; the excess is carried by the higher chaoses, which (6) discards. The growth between $$L=3$$ and $$L=7$$ corresponds to exponents $$3.19,\,3.17,\,3.06,\,3.31$$ in the four columns, each with an uncertainty of about $$\pm0.3$$ from the sample sizes, consistent with the exponent 3 of (9).

The exact inequality (6) holds for every $$L$$. The explicit constants (9) are asymptotic. Their finite-width evaluation (the numbers after the slash) is below the sampled value in every case, as it must be if the approximation is faithful.

*Consequence.* At kernel scale the single-realisation operator is dominated by fluctuations for any smooth $$\phi$$ of width $$L\gtrsim2\ell$$, and the relative size is at least of order $$(L/\ell)^3$$. Reducing it needs a kernel with a non-locality scale $$\ell_k\gg\ell$$, by averaging over many points as in Sorkin's smeared operators [[Sorkin]](#ref-sorkin) [[Aslanbeigi–Saravani–Sorkin]](#ref-ass). How the bound scales with $$\ell_k$$ is not derived here.

## 3. The cocompact orbit lattice

Let $$Q_7(x)=7x_0^2-x_1^2-x_2^2-x_3^2$$ on $$\mathbb Z^4$$ and $$e(x)=7^{-1/8}(\sqrt7x_0,x_1,x_2,x_3)$$, so that $$L_7=e(\mathbb Z^4)\subset\mathbb R^{1,3}$$ has covolume 1 and Minkowski norm $$7^{-1/4}Q_7$$. Let $$\Gamma_7=SO^+(Q_7;\mathbb Z)$$, acting on $$L_7$$. A *realisation* is $$\Lambda=gL_7+\xi$$ with $$g\in SO^+(1,3)$$ and $$\xi\in\mathbb R^4$$; the random lattice is the one with $$g$$ Haar distributed on $$SO^+(1,3)/\Gamma_7$$ and $$\xi$$ uniform on $$\mathbb R^4/gL_7$$.

Two results are cited. *Borel and Harish-Chandra*: for a semisimple $$\mathbb Q$$-group the group of integral points is a lattice. *Godement's compactness criterion* (see Borel's *Introduction aux groupes arithmétiques*): this lattice is cocompact if the $$\mathbb Q$$-group is anisotropic, that is, if the form has no non-trivial rational zero. *Mahler's compactness criterion*: the set of unimodular lattices in $$\mathbb R^n$$ whose non-zero vectors all have Euclidean norm at least $$\delta>0$$ is compact.

<div class="theorem" markdown="1">

**Theorem 3** (the orbit lattice). (i) $$Q_7$$ has no non-trivial zero in $$\mathbb Z^4$$, so $$L_7$$ has no null vectors, and $$\lvert Q_7(v)\rvert \ge1$$ for $$v\in\mathbb Z^4\setminus\{0\}$$. (ii) $$\Gamma_7$$ is a cocompact lattice in $$SO^+(1,3)$$, hence Zariski dense. (iii) The random lattice $$\Lambda$$ is Poincaré invariant in law and ergodic under the Poincaré group, though not under translations alone. (iv) There is $$R_*<\infty$$ such that every lattice $$gL_7$$ has covering radius at most $$R_*$$. Consequently, for every realisation and every pair of lattice points $$x\prec y$$ with proper time $$\tau(x,y)>2\sqrt2R_*$$, the interval contains a lattice point, and the interior count satisfies $$m(x,y)\ge cV-C$$ for constants $$c>0,C$$ independent of the realisation. (v) $$m(x,y)\equiv\mathbf 1[\,y-x\in2gL_7\,]\pmod2$$ for $$x,y\in\Lambda=gL_7+\xi$$. (vi) There is $$C$$ such that for every realisation and every causal interval $$S$$ of volume $$V\ge1$$,

$$
\lvert N_S-V\rvert \le C\,V^{3/4}. \tag{10}
$$

</div>

The idea of (i) to (iv) is that an anisotropic form keeps the lattice away from the light cone in every frame, so the whole boost orbit is a compact family of lattices; (v) is a point reflection, and (vi) is the geometry-of-numbers statement that bounded covering radius forces the lattice count to follow the volume up to a boundary term.

<div class="proof" markdown="1">

*Proof.* (i) Suppose $$7a^2=b^2+c^2+d^2$$ with integers not all zero, divided by their greatest common divisor so that they are coprime. If $$a$$ is odd, the left side is $$7\bmod 8$$, but squares mod 8 are $$0,1,4$$ and no sum of three of them is $$7$$. If $$a$$ is even, the right side is $$0\bmod4$$, which forces $$b,c,d$$ even, contradicting coprimality.

(ii) By (i) the $$\mathbb Q$$-group $$SO(Q_7)$$ is anisotropic; apply the Borel–Harish-Chandra theorem, Godement's criterion and Borel density.

(iii) The group of affine Lorentz maps acts transitively on the space $$(SO^+(1,3)\ltimes\mathbb R^4)/(\Gamma_7\ltimes L_7)$$ of realisations, which carries a finite invariant measure by (ii); a transitive action preserves the invariant measure and is ergodic. Translations alone fix the orientation of the lattice and are not ergodic.

(iv) In coordinates where the metric is Euclidean, $$\lvert v\rvert _{\rm E}^2=v_0^2+\lvert \vec v\rvert ^2\ge\lvert v_0^2-\rvert \vec v\lvert ^2\rvert $$, so by (i) every non-zero vector of every lattice $$gL_7$$ has Euclidean norm at least $$7^{-1/8}$$. By Mahler's criterion the orbit lies in a compact set of lattices, on which the covering radius is bounded by some $$R_*$$. In the rest frame of an interval of proper time $$\tau$$, which is a point of the same orbit, the interval contains the Euclidean ball of radius $$\tau/(2\sqrt2)$$ about its centre (a point at distance $$\rho$$ from the centre with time offset $$s$$ has $$\lvert \vec x\rvert <\tau/2-s$$ iff $$\rho^2<\tau^2/4-\tau s+2s^2$$, whose minimum over $$s$$ is $$\tau^2/8$$). This ball contains a lattice point once its radius $$\rho$$ exceeds $$R_*$$. More precisely, the Voronoi cells of the lattice points have volume 1 and radius at most $$R_*$$ and cover the concentric ball of radius $$\rho-R_*$$, so the ball contains at least $$\mathrm{vol}\,B_{\rho-R_*}$$ lattice points; with $$V=\pi\tau^4/24$$ this gives $$m\ge cV-C$$.

(v) The map $$z\mapsto x+y-z$$ is an involution of the lattice points in the open interval between $$x$$ and $$y$$, with the only possible fixed point $$(x+y)/2$$, which is a lattice point iff $$(y-x)/2\in gL_7$$.

(vi) Work in the rest frame of $$S$$, where $$S$$ is a convex body of diameter $$\sim\tau$$ with $$V\propto\tau^4$$ and boundary area $$\propto\tau^3$$. The Voronoi cells of the lattice points have volume 1 and radius at most $$R_*$$. The cells of points in $$S$$ lie in the parallel body $$S\oplus B_{R_*}$$, and the cells covering the inner parallel body $$S\ominus B_{R_*}$$ belong to points of $$S$$. Hence $$\mathrm{vol}(S\ominus B_{R_*})\le N_S\le\mathrm{vol}(S\oplus B_{R_*})$$. The collar of width $$R_*$$ inside a convex body has volume at most $$R_*\cdot\mathrm{area}(\partial S)$$, and by Steiner's formula $$\mathrm{vol}(S\oplus B_{R_*})-\mathrm{vol}(S)\le C(R_*\tau^3+R_*^2\tau^2+\dots)$$. Both are $$O(\tau^3)=O(V^{3/4})$$ for $$\tau\ge1$$. $$\square$$

</div>

For a Poisson sprinkling every realisation contains empty intervals of arbitrarily large volume, so no bound like (10) holds. Bound (10) gives $$\mathrm{Var}\,N_S\le C^2V^{3/2}$$, which is weaker than the Poisson value $$V$$ and decides nothing about the variance.

*Number variance (numerical).* We sampled $$g$$ by a uniform draw from the ball of radius $$R_b$$ in hyperbolic space with a uniform rotation (an approximation to Haar measure on the quotient that improves with $$R_b$$) and $$\xi$$ uniform, and counted lattice points in a causal diamond of volume $$V$$. With $$3\cdot10^4$$ samples per entry the mean of $$N/V$$ was $$1.000\pm0.002$$ in each case, and $$\mathrm{Var}\,N/V$$ was:

| $$V$$ | $$R_b=3$$ | $$R_b=4$$ |
|---|---|---|
| 4 | 0.525 | 0.456 |
| 16 | 0.351 | 0.313 |
| 64 | 0.211 | 0.195 |

The ratio is below the Poisson value 1, and it falls roughly like $$V^{-0.34}$$ to $$V^{-0.37}$$ between $$V=16$$ and $$64$$ (variance growing like $$V^{0.63}$$ to $$V^{0.66}$$). The difference between $$R_b=3$$ and $$4$$ is the bias of the ball average, of order $$10\%$$ at $$V=4$$ and $$8\%$$ at $$V=64$$. The maximum of $$\lvert N-V\rvert /V^{3/4}$$ over the samples was $$1.3$$ to $$1.5$$ at $$V=64$$. These are values up to $$V=64$$ only. Saravani and Aslanbeigi conjecture that no process has $$\mathrm{Var}\,N_S\le\alpha V$$ with $$\alpha<1$$ for all large intervals [[Saravani–Aslanbeigi]](#ref-sa); the conjecture is asymptotic, the orbit lattice is an admissible test of it, and data up to $$V=64$$ do not decide it.

## 4. The mean dictionary on the orbit lattice

For a realisation through $$0$$, let $$c_n$$ be the density, per unit hyperbolic measure, of lattice vectors $$v\in\Lambda$$ in the future cone with $$7^{-1/4}Q_7(g^{-1}v)=\tau_n^2$$, where $$\tau_n^2=7^{-1/4}n$$; by invariance it does not depend on $$g$$. Let $$P(m\vert n)$$ be the probability that such a vector has interior count $$m$$. The layer-resolved expected number of lattice vectors with interior count $$m$$ on shell $$n$$ is $$c_nP(m\vert n)$$ per unit hyperbolic measure, and unit density requires $$\sum_{n\le N}c_n\approx N^2/(4\sqrt7)$$ (the integral of $$\tau^3d\tau$$ up to $$\tau_N$$); with the values listed below, $$\sum_{n\le16}c_n=24.12$$ against $$16^2/(4\sqrt7)=24.19$$. The mean of (1) is $$N[a\phi(0)+\sum_{n,m}c_nP(m\vert n)w_m\int_{H^3}\phi(-\tau_nu)\,d\mu(u)]$$, which has the same form as the Poisson mean with the continuous measure $$\psi(V)\,dV$$ replaced by a discrete sum over the shells $$n$$. Because the layer laws have compact support (Theorem 3(iv)), the Mellin transform of the layer densities is a finite Dirichlet polynomial, entire, and the infrared conditions are a finite linear system for the weights, with no obstruction from the tail that spoilt the Haar-random lattice (see [Lorentz-invariant point sets](/quantum-gravity/invariant-point-sets/)).

*Palm data (numerical).* We counted lattice vectors on the shells $$Q_7=n$$ with Euclidean norm up to a cutoff and, for the shells $$n=4,6,7,9,10$$, the interior counts of all vectors with $$x_0\le28$$ (6700 to 8500 vectors per shell) by exact enumeration. The fractions are

| $$n$$ | measured fractions | rational values consistent with them |
|---|---|---|
| 4 | $$m=0$$: 0.658; $$m=1$$: 0.342 | 2/3, 1/3 |
| 6 | $$m=0$$: 0.322; $$m=2$$: 0.678 | 1/3, 2/3 |
| 7 | $$m=0$$: 0.036; $$m=2$$: 0.964 | 1/25, 24/25 |
| 9 | $$m=2$$: 0.152; $$m=4$$: 0.848 | 1/7, 6/7 |
| 10 | $$m=4$$: 0.678; $$m=6$$: 0.322 | 2/3, 1/3 |

The parity rule (v) holds for every vector examined in the shells $$n\le6$$ with $$x_0\le10$$. The shell densities $$c_n$$, estimated by counting vectors in hyperbolic balls (cutoffs $$x_0\le60$$ and $$x_0\le120$$, averaged), are, for $$n=1,\dots,16$$: $$0.250,\,0.755,\,0.385,\,0.757,\,0.760,\,1.137,\,1.582,\,0.761,\,1.745,\,2.295,\,3.040,\,1.136,\,2.286,\,4.721,\,2.074,\,0.777$$, against the Siegel-type values $$0.25,\,0.75,\,0.375,\,0.75,\,0.75,\,1.125,\,1.5625,\,0.75,\,1.75,\,2.25,\,3.0,\,1.125,\,2.25,\,4.6874,\,2.0,\,0.75$$ (agreement to 4% or better, with deviations set by the fluctuation of hyperbolic lattice-point counts). The values $$16c_n$$ are close to integers. The closed form of these densities in terms of local densities and the exact orbit stabiliser orders was identified numerically in the source of these values and is not derived here.

*Retuned weights and their sensitivity.* The weights $$(2.940032,-194.990930,104.166648,-409.998336,28.942005)$$ with local coefficient $$a=-0.798641$$ and $$\lvert N\rvert =4/\sqrt6$$ are the solution of the infrared conditions with the exact rational layer probabilities and the Siegel-type shell densities; they, and the exact densities and probabilities, are quoted from an independent solution of that system and not re-derived here (our enumeration above agrees with the densities to 4% and the probabilities to the stated accuracy). We evaluated the mean ratio $$\mathbb E[B\phi]/\Box\phi$$ from these data; for the timelike packet at $$L=5,10,20,40$$: $$0.971,\,1.257,\,1.127,\,1.047$$; for the spacelike packet $$0.300,\,0.645,\,0.855,\,0.949$$. The ratio tends to 1, as on the Poisson process (section 1): the mean dictionary holds on the orbit lattice. It is sensitive to the Palm law: changing $$P(0\vert 6)$$ from $$1/3$$ to $$0.353$$ with the same weights changes the timelike ratio at $$L=10,20,40$$ from $$1.26,\,1.13,\,1.05$$ to $$-3.0\cdot10^3,\,-4.3\cdot10^4,\,-6.6\cdot10^5$$. A shift of $$0.02$$ in one probability breaks the cancellation of the constant mode, and the error is amplified by about $$L^4$$. A working dictionary therefore needs the exact (rational) Palm law, not an estimate.

## 5. A single realisation of the orbit lattice

Take a realisation $$\Lambda$$ through $$0$$, apply a Lorentz transformation if needed so that $$\Lambda=gL_7$$, and form the retarded layer kernel $$c(v)=N\,w_{m(v)}$$ on $$\Lambda$$ for $$v$$ in the future cone with $$n(v):=Q_7(g^{-1}v)\in\{1,\dots,10\}$$, and $$c(v)=0$$ otherwise. Here $$m(v)$$ is the number of points of $$\Lambda$$ in the interval $$(0,v)$$. Then $$(B\phi)(x)=N a\phi(x)+\sum_vc(v)\phi(x-v)$$.

<div class="proposition" markdown="1">

**Proposition 4** (a layer operator on the orbit lattice has no symbol). The stabiliser $$\Gamma_g=g\Gamma_7g^{-1}$$ of $$\Lambda$$ in $$SO^+(1,3)$$ is Zariski dense, and the kernel $$c$$ is $$\Gamma_g$$-invariant, supported on finitely many shells, and non-zero off the origin (if $$w_0\ne0$$). Therefore, by Proposition 1 of [Exact discrete Lorentz symmetry cannot carry propagation](/quantum-gravity/discrete-lorentz-group/), the Fourier series $$\hat c$$ of the kernel coincides with an integrable function on no open set of the Brillouin torus; and for every finitely supported $$\phi\ne0$$ on $$\Lambda$$, the function $$B\phi-Na\phi$$ is not square summable. Moreover no bounded operator on $$\ell^2(\Lambda)$$ that commutes with the translations of $$\Lambda$$ and with $$\Gamma_g$$ is anything but a multiple of the identity, so a smearing of the kernel by any $$\Gamma_g$$-invariant function of the interval data cannot give a bounded operator other than a multiple of the identity.

</div>

<div class="proof" markdown="1">

*Proof.* $$\Gamma_g$$ is conjugate to $$\Gamma_7$$, which is Zariski dense by Theorem 3(ii). It preserves $$\Lambda$$, the form, and the time orientation, hence the interval counts: $$m(\gamma v)=m(v)$$. So $$c$$ is $$\Gamma_g$$-invariant, and it vanishes outside the shells $$n(v)\in\{1,\dots,10\}$$. It is non-zero at the vectors of shell 1, which are links ($$m=0$$) with weight $$w_0\ne0$$. Proposition 1 applies and gives the first statement. Suppose $$\phi\ne0$$ is finitely supported and $$\widehat{B\phi-Na\phi}=\hat c\,\hat\phi$$ were in $$L^2$$, where $$\hat\phi$$ is a non-zero trigonometric polynomial. On the open set $$U=\{\hat\phi\ne0\}$$, which is non-empty, $$\hat c=(\hat c\hat\phi)/\hat\phi$$ would be locally square integrable, contradicting Proposition 1. The last statement is Theorem 1(a) of the same page. $$\square$$

</div>

So the response to a localised field is not a localised field: a retarded layer sum with weights tuned to give the correct mean maps any compactly supported test field to a function of infinite $$\ell^2$$ norm, whose values do not decay along the hyperboloidal shells around the support of the field. This is the structural content. It does not by itself give the size of the error for a Gaussian packet, which depends on cancellations among the shell sums.

*Numerical single-realisation error.* We took the unboosted realisation ($$g=1$$, $$\xi=0$$) with the weights, $$a$$ and $$N$$ of section 4, computed the interior count $$m(v)$$ of every vector $$v$$ with $$1\le Q_7(v)\le10$$ in the packet's window by exact enumeration of the lattice points of the interval, and evaluated $$B\phi(0)$$ for the packet (3) at rest ($$\beta=0$$) and boosted by speed $$\beta=0.6$$ along $$(1,2,3)/\sqrt{14}$$. The relative error $$\lvert B\phi(0)/\Box\phi(0)-1\rvert $$ is:

| packet | $$L=5$$ | $$L=10$$ | $$L=20$$ |
|---|---|---|---|
| timelike, $$\beta=0$$ | $$5.9\cdot10^3$$ | $$9.9\cdot10^3$$ | $$2.6\cdot10^5$$ |
| spacelike, $$\beta=0$$ | $$5.9\cdot10^3$$ | $$3.1\cdot10^4$$ | $$1.0\cdot10^5$$ |
| timelike, $$\beta=0.6$$ | $$3.8\cdot10^2$$ | $$3.9\cdot10^4$$ | not run |
| spacelike, $$\beta=0.6$$ | $$2.9\cdot10^3$$ | $$1.8\cdot10^4$$ | not run |

One realisation, one frame, and four packets give no estimate of the typical error or of its growth exponent: the values range over three orders of magnitude and fall in the range of the standard deviations of the Poisson sprinkling in section 2 ($$10^3$$ to $$10^4$$ at $$L=5$$ to $$7$$). The error is not small at any $$L$$ tried, and for the packets at rest it is larger at $$L=20$$ than at $$L=5$$ by factors of $$44$$ and $$17$$. Quoted from an independent computation and not repeated here: with the same weights the median error over three boosts and two carriers grows roughly like $$L^{3.3}$$ between $$L=5$$ and $$L=40$$.

## 6. Superposing independent lattices

One may ask whether more randomness repairs the lattice. Let $$\Lambda_K$$ be the union of $$K$$ independent copies of the random orbit lattice, each of density $$1/K$$ (a copy is the density-one random lattice scaled by $$K^{1/4}$$), so that the density is 1.

<div class="proposition" markdown="1">

**Proposition 5** (number variance of a superposition). For every bounded region $$D$$ of volume $$V$$ that is a causal diamond, $$\mathrm{Var}\,N_K(D)/\mathbb E\,N_K(D)=r(V/K)$$, where $$r(v)=\mathrm{Var}\,N_1(D')/\mathbb E\,N_1(D')$$ is the ratio for a single density-one copy and a diamond $$D'$$ of volume $$v$$. Moreover $$r(v)\to1$$ as $$v\to0$$, so for fixed $$V$$ the ratio tends to 1 as $$K\to\infty$$.

</div>

<div class="proof" markdown="1">

*Proof.* The counts $$N_j(D)$$ of the independent copies are independent. Scaling by $$K^{-1/4}$$ maps a copy to a density-one copy and $$D$$ to a diamond of volume $$V/K$$, and the law of the random lattice is invariant under Lorentz transformations, so $$N_j(D)$$ has the law of $$N_1(D')$$ with $$\mathrm{vol}\,D'=V/K$$. Then $$\mathrm{Var}\,N_K=K\,\mathrm{Var}\,N_1(D')=K\,r(V/K)\,\mathbb E N_1(D')$$, and $$\mathbb E N_K=V=K\,\mathbb E N_1(D')$$. For the limit: a diamond of proper time $$T$$ has Euclidean diameter at most $$T$$ in its rest frame, where every lattice of the orbit has no non-zero vector shorter than $$\delta=7^{-1/8}$$ (proof of Theorem 3(iv)). For $$T<\delta$$ the diamond contains at most one lattice point, so $$N_1(D')\in\{0,1\}$$ and $$r=1-\mathbb E N_1(D')\to1$$. $$\square$$

</div>

As $$K\to\infty$$ at fixed $$V$$ the number statistics approach the Poisson ones, but the per-realisation errors need not follow, because the failure of section 2 is already present for Poisson. Quoted from an independent computation and not repeated here: for $$K$$ between 1 and 64 the single-realisation error of the retuned kernel at $$L=5,10,20$$ varied with $$K$$ by less than a factor of about 5, with no systematic trend.

## 7. What the results do not exclude

1. Smeared kernels on a Poisson sprinkling, with a non-locality scale $$\ell_k\gg\ell$$. Sorkin's argument and the generalised d'Alembertians of [[Aslanbeigi–Saravani–Sorkin]](#ref-ass) address this regime. The bound (9) is for $$\ell_k=\ell$$, and its dependence on $$\ell_k$$ is not derived here.
2. Orbit lattices with a smearing that is not a function of $$\Gamma_g$$-invariant data. Proposition 4 excludes bounded $$\Gamma_g$$-invariant operators other than multiples of the identity. A mesoscale smearing that uses a frame, for instance a time direction, breaks the invariance under $$\Gamma_g$$ of the operator and is not covered; it brings back the preferred-frame questions of [Where a preferred frame leaks into low-energy physics](/quantum-gravity/lorentz-violation-leakage/).
3. Superpositions of $$K\ge2$$ orbit lattices with a mesoscale. Nothing here shows that they fail or succeed.
4. Lattices that are not arithmetic orbits. The proofs use cocompactness (anisotropy) and the Zariski density of the stabiliser.
5. Observables other than the d'Alembertian on a Gaussian packet at one point, for example averages of $$B\phi$$ over many points of the realisation, which are not treated here.

## 8. Open points

The large-width constants (9) rest on replacing $$\phi(y)$$ by $$\phi(z)$$ in (5) and the error of that replacement is not bounded. The exponent of the single-realisation error on the orbit lattice, and whether it differs from that of Poisson, is not determined by four packets in one frame. The asymptotic law of $$\mathrm{Var}\,N_S$$ on the orbit lattice, between the bounds $$V^{0.66}$$ (observed up to $$V=64$$) and $$V^{3/2}$$ (proved), is open. The closed form of the shell densities and Palm laws is identified numerically, not derived, and the exactness of the rational layer probabilities listed above is an observation from enumeration of up to $$8500$$ vectors per shell.

## References

1. <a id="ref-bd"></a>Benincasa, Dowker, "The scalar curvature of a causal set," Phys. Rev. Lett. 104 (2010) 181301, [arXiv:1001.2725](https://arxiv.org/abs/1001.2725).
2. <a id="ref-ass"></a>Aslanbeigi, Saravani, Sorkin, "Generalized causal set d'Alembertians," JHEP 06 (2014) 024, [arXiv:1403.1622](https://arxiv.org/abs/1403.1622).
3. <a id="ref-sorkin"></a>Sorkin, "Does locality fail at intermediate length-scales," [arXiv:gr-qc/0703099](https://arxiv.org/abs/gr-qc/0703099).
4. <a id="ref-sa"></a>Saravani, Aslanbeigi, "On the causal set-continuum correspondence," [arXiv:1403.6429](https://arxiv.org/abs/1403.6429).
5. <a id="ref-lp"></a>Last, Penrose, "Poisson process Fock space representation, chaos expansion and covariance inequalities," [arXiv:0909.3205](https://arxiv.org/abs/0909.3205).
