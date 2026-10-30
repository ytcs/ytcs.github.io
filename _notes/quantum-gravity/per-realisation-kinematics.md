---
title: Kernel-scale kinematics of a single realisation
date: 2026-10-29
category: substrate
order: 6
description: At the discreteness scale, the Benincasa–Dowker d'Alembertian on a Poisson sprinkling has standard deviation at least about one third of the cube of the width times the signal for timelike Gaussian packets (computed), and a layer operator with non-zero nearest-neighbour weight on the orbit lattice of one anisotropic integral form has no symbol (proved).
status: "Proved: the first-chaos inequality (6), the structure of the orbit lattice (Theorem 3), the absence of a symbol (Proposition 4) and the number variance of superposed lattices (Proposition 5). Computed: the size of (6), sampled standard deviations, shell densities, weights restricted to shells 1 to 10, single-realisation errors. Identified, not proved: the interior-count laws of shells 6 to 10."
excludes: "Kernel scale equal to the discreteness scale; standard deviation (root-mean-square accuracy) in one Poisson realisation, for timelike Gaussian packets of eight widths from 2 to 32 (Benincasa–Dowker) or seven from 2 to 20 (five-layer) discreteness lengths (computed). Orbit lattice of section 3 only, kernels supported on finitely many shells (proved)."
---

A causal-set d'Alembertian is a sum over the causal past of a point, with a weight that depends on the number $$m$$ of points in the causal interval between the two. Its expectation on a Poisson sprinkling of density $$\ell^{-4}$$ is the d'Alembertian $$\Box\phi$$ of a smooth field $$\phi$$ [[Benincasa–Dowker]](#ref-bd) [[Aslanbeigi–Saravani–Sorkin]](#ref-ass). Does one realisation compute $$\Box\phi$$ when the weights act at the scale $$\ell$$ of the discreteness and $$\phi$$ varies on a scale $$L\gg\ell$$? For the Benincasa–Dowker weights and for a five-layer variant, on the Gaussian packets examined, it does not, in root-mean-square error. For a timelike packet of width $$L$$ between $$2\ell$$ and $$32\ell$$ (eight widths tabulated), the standard deviation of the Benincasa–Dowker operator at a point is at least about $$0.33\,(L/\ell)^3\,\lvert\Box\phi\rvert$$: the inequality (6) behind this is proved and its value is computed. Sampled standard deviations are $$3\cdot10^2$$ to $$8\cdot10^3$$ times $$\lvert\Box\phi\rvert$$ at $$L=5\ell$$, depending on the kernel and the carrier.

A lattice has no Poisson randomness within a realisation. We take $$\mathbb Z^4$$ with the anisotropic form $$7x_0^2-x_1^2-x_2^2-x_3^2$$, embedded in Minkowski space and randomly boosted and translated (the orbit lattice of section 3), which is Poincaré invariant in law. Its weights can be retuned to the interval statistics of the lattice so that the mean reproduces $$\Box\phi$$ on the packets examined (section 4, numerical). A single realisation nevertheless carries an operator with no symbol, and for a kernel with non-zero nearest-neighbour weight the operator maps every non-zero finitely supported field outside the square-summable functions (Proposition 4, proved). The relative errors computed on four Gaussian packets lie between $$4\cdot10^2$$ and $$4\cdot10^6$$; Proposition 4 does not bound them. Superposing independent copies of the lattice brings the variance-to-mean ratio of the counts in a diamond to 1, the Poisson value (Proposition 5); the operator on superposed lattices is not computed.

Sorkin argued that Lorentz-invariant discreteness requires a non-locality scale $$\ell_k\gg\ell$$ [[Sorkin]](#ref-sorkin), and the generalised d'Alembertians of Aslanbeigi, Saravani and Sorkin [[Aslanbeigi–Saravani–Sorkin]](#ref-ass) are families with such a scale. Saravani and Aslanbeigi studied number fluctuations of Poisson sprinklings and of two-dimensional Lorentzian lattices [[Saravani–Aslanbeigi]](#ref-sa). The lower bound below starts from the first term of the chaos expansion of a Poisson functional [[Last–Penrose]](#ref-lp); the evaluation of that term for four-dimensional layer kernels is new here. The conditions on the weights that make the mean work are derived in [mixed-Poisson and signed substrates](/quantum-gravity/signed-substrates/), the invariant random lattices are described in [Lorentz-invariant point sets](/quantum-gravity/invariant-point-sets/), and the argument that a Zariski-dense stabiliser forbids a symbol is given in [lattices with an arithmetic Lorentz group](/quantum-gravity/discrete-lorentz-group/).

## 1. Layer operators at the kernel scale

Let $$\Phi$$ be a locally finite set in $$\mathbb R^{1,3}$$ with signature $$(+,-,-,-)$$, so that $$\Box=\partial_t^2-\nabla^2$$ (with signature $$(-,+,+,+)$$ the sign of $$\Box$$ and that of $$\nu$$ change), and let $$x\in\Phi$$. For $$y\prec x$$ in the causal past, let $$m(y,x)$$ be the number of points of $$\Phi$$ in the open interval between $$y$$ and $$x$$. A *layer operator* with weights $$(w_0,\dots,w_M)$$, local coefficient $$a$$ and normalisation $$\nu$$ acts on a function $$\phi$$ by

$$
(B\phi)(x)=\nu\Big[a\,\phi(x)+\sum_{y\in\Phi,\ y\prec x}w_{m(y,x)}\,\phi(y)\Big],\qquad w_m=0\ \ (m>M). \tag{1}
$$

For a Poisson process of unit density the interval between $$y$$ and $$x$$ has volume $$V=\pi\tau^4/24$$, with $$\tau$$ the proper time, and $$m$$ is Poisson distributed with mean $$V$$. Hence $$\mathbb E\,(B\phi)(x)=\nu[a\phi(x)+\int_{y\prec x}\psi(V(y,x))\phi(y)\,d^4y]$$ with

$$
\psi(V)=\sum_mw_m\,e^{-V}\frac{V^m}{m!}. \tag{2}
$$

The Mellin transform of $$\psi$$ is $$\hat\psi(s)=\int_0^\infty\psi(V)V^{s-1}dV=\sum_mw_m\Gamma(s+m)/m!$$. The conditions of Aslanbeigi, Saravani and Sorkin for the mean to tend to $$\Box\phi$$ are $$\hat\psi(\tfrac12)=\hat\psi(1)=\hat\psi(\tfrac32)=0$$ and $$a=-3\hat\psi'(1)$$, with $$\nu$$ fixed by the value of $$\hat\psi'(\tfrac32)$$ [[Aslanbeigi–Saravani–Sorkin]](#ref-ass). They are derived in [mixed-Poisson and signed substrates](/quantum-gravity/signed-substrates/), and we use them as given. We use two weight sets, both with $$\lvert\hat\psi'(\tfrac32)\rvert=\sqrt\pi/3$$ and $$\lvert\nu\rvert=4/\sqrt6$$ ($$\nu=-4/\sqrt6$$ for the Benincasa–Dowker set and $$\nu=4/\sqrt6$$ for the five-layer set below).

The Benincasa–Dowker weights are $$(1,-9,16,-8)$$, as in the Remark on the causal-set d'Alembertian in [horizon counts and the area law](/quantum-gravity/area-projector/), where the signature is $$(-,+,+,+)$$ and the normalisation is $$\nu=+4/\sqrt6$$. For them $$\hat\psi(s)=-\tfrac13\Gamma(s)(2s-1)(s-1)(2s-3)$$, so that $$\hat\psi'(1)=\tfrac13$$ (hence $$a=-1$$) and $$\hat\psi(\tfrac54)=0.0567$$. In [causal-set counts and dimensionless constants](/quantum-gravity/no-number-from-kinematics/) and [boundary and corner terms](/quantum-gravity/boundary-terms/) the sign of $$\nu$$ is absorbed into the weights, so that the normalisation is $$4/\sqrt6$$ and the Benincasa–Dowker weights are $$(-1,9,-16,8)$$ with the symbol $$Q_{\rm BD}(s)=\tfrac13(2s-1)(s-1)(2s-3)=-\hat\psi(s)/\Gamma(s)$$. The five-layer set $$(5,-85,280,-328,128)$$ has $$\hat\psi(s)=\tfrac{16}3\Gamma(s)(s-\tfrac12)(s-1)(s-\tfrac54)(s-\tfrac32)$$ and also $$\hat\psi'(1)=\tfrac13$$, so $$a=-1$$. Its extra zero at $$\tfrac54$$ removes the term of order $$\ell^{-1}$$, proportional to the volume of the timelike boundary, that the Benincasa–Dowker action has on the regions of [boundary and corner terms](/quantum-gravity/boundary-terms/). We use it to test how the fluctuation depends on the weights.

The test field is a Gaussian packet,

$$
\phi(x)=e^{-\lvert x\rvert _{\rm E}^2/2L^2}\cos(p\cdot x),\qquad p=3/L, \tag{3}
$$

where $$\lvert x\rvert_{\rm E}^2=t^2+\lvert\vec x\rvert^2$$, with the carrier along the time axis (timelike, $$\cos pt$$) or along $$x_1$$ (spacelike, $$\cos px_1$$). Lengths are in units of $$\ell=1$$. Then $$\Box\phi(0)=2/L^2-p^2=-7/L^2$$ (timelike) or $$2/L^2+p^2=11/L^2$$ (spacelike).

The mean of (1) at a point of a Poisson sprinkling, computed by quadrature in hyperbolic coordinates and divided by $$\Box\phi(0)$$ is, for $$L=3,5,10,20,40$$:

| kernel and carrier | $$L=3$$ | $$L=5$$ | $$L=10$$ | $$L=20$$ | $$L=40$$ |
|---|---|---|---|---|---|
| Benincasa–Dowker, timelike | 0.82 | 1.15 | 1.12 | 1.05 | 1.02 |
| Benincasa–Dowker, spacelike | 0.48 | 0.67 | 0.86 | 0.95 | 0.98 |
| five-layer, timelike | $$-0.47$$ | 0.98 | 1.24 | 1.12 | 1.04 |
| five-layer, spacelike | 0.08 | 0.33 | 0.66 | 0.86 | 0.95 |

The mean ratio tends to 1 as $$L$$ grows, with corrections that fall like a power of $$\ell/L$$; for the five-layer kernel at $$L=3$$ it has the opposite sign. The mean does not fix the error of one realisation; the fluctuation around it is bounded in section 2.

## 2. The variance on a Poisson sprinkling

We evaluate $$B\phi$$ at a point of the sprinkling placed at the origin. By the Mecke formula the other points then form a Poisson process $$\Phi$$ of unit density (the Palm configuration), and $$B\phi(0)=\nu X$$ with $$X=a\phi(0)+\sum_{y\in\Phi,\,y\prec0}w_{m(y,0)}\phi(y)$$. For a Poisson process $$\eta$$ with $$\sigma$$-finite intensity measure $$\lambda$$ and a square-integrable functional $$F(\eta)$$, put $$D_zF(\eta)=F(\eta+\delta_z)-F(\eta)$$, the cost of adding the point $$z$$. The Wiener–Itô chaos expansion gives $$\mathrm{Var}\,F=\sum_{n\ge1}\frac1{n!}\int\lvert \mathbb E\,D^n_{z_1\dots z_n}F\rvert ^2\lambda^n(dz)$$ [[Last–Penrose]](#ref-lp), so that

$$
\mathrm{Var}\,F\ \ge\ \int\big\lvert \mathbb E\,D_zF\big\rvert ^2\lambda(dz). \tag{4}
$$

In words, the variance is at least the mean square response of $$F$$ to the addition of one point. The terms with $$n\ge2$$ are discarded in (4); the first term is the only one we compute. The functional $$X$$ is square integrable for the packet (3), since its first and second absolute moments are bounded by integrals of $$\lvert\phi\rvert$$ and $$\phi^2$$ against $$\mathbb P(m(y,0)\le M)$$, which converge by the Gaussian decay of $$\phi$$.

<div class="proposition" markdown="1">

**Proposition 1** (the expected add-one cost). For the operator (1) at the origin,

$$
\mathbb E\,D_zX=\mathbf 1_{\{z\prec0\}}\Big[\phi(z)\,\psi(V_z)+\int_{y\prec z}\phi(y)\,\psi'(V_y)\,d^4y\Big], \tag{5}
$$

where $$V_z=V(z,0)$$ and $$V_y=V(y,0)$$ are the volumes of the intervals ending at $$0$$.

</div>

A new point $$z$$ in the past of $$0$$ adds a term of its own and raises by one the interior count of every earlier point $$y$$ for which $$z$$ lies in the interval between $$y$$ and $$0$$; the first term of (5) is the former and the integral is the latter.

<div class="proof" markdown="1">

*Proof.* If $$z\not\prec0$$, the point $$z$$ enters no term and lies in no interval ending at $$0$$, so $$D_zX=0$$. If $$z\prec0$$, the new point contributes $$w_{m(z,0)}\phi(z)$$, where $$m(z,0)$$ is Poisson with mean $$V_z$$, and the expectation of this term is $$\phi(z)\psi(V_z)$$. For each $$y\in\Phi$$ with $$y\prec z$$ (equivalently $$z\in I(y,0)$$), the count $$m(y,0)$$ increases by one and the term changes from $$w_m\phi(y)$$ to $$w_{m+1}\phi(y)$$. By the Mecke formula, given a point at $$y$$ the rest of $$\Phi$$ is Poisson, so the interior count excluding $$z$$ is Poisson with mean $$V_y$$, and $$\mathbb E[w_{M'+1}-w_{M'}]=\psi'(V_y)$$ for $$M'\sim\mathrm{Poisson}(V_y)$$ (differentiate (2)). Campbell's formula gives the integral over $$y\prec z$$. $$\square$$

</div>

<div class="corollary" markdown="1">

**Corollary 2** (variance lower bound). With $$\Sigma(\phi)=\int_{z\prec0}\big[\phi(z)\psi(V_z)+\int_{y\prec z}\phi(y)\psi'(V_y)d^4y\big]^2d^4z$$,

$$
\mathrm{Var}\,\big(B\phi(0)\big)\ \ge\ \nu^2\,\Sigma(\phi). \tag{6}
$$

</div>

Inequality (6) holds for every $$L$$ and every weight set; it is (4) with $$\lambda$$ the Lebesgue measure of the past cone and (5) inserted.

### Size of the bound for a timelike packet

For the timelike packet, write $$z=-\tau_zu$$ and $$y=-\tau_yu'$$ with $$u,u'$$ unit future vectors, that is, points of the hyperbolic space $$H^3$$. Then $$y\prec z$$ holds exactly when $$\tau_y>\tau_z$$ and the hyperbolic distance of $$u$$ and $$u'$$ is smaller than $$\ln(\tau_y/\tau_z)$$. The packet depends on $$u'$$ only through its rapidity, and the integral over the angle at $$u$$ is that of a Gaussian times a cosine, which is a difference of complex error functions. What remains is a four-dimensional quadrature, accurate to about $$1\%$$. The table gives $$\lvert\nu\rvert\sqrt{\Sigma}/\lvert\Box\phi(0)\rvert$$, which by (6) is a lower bound for the standard deviation of $$B\phi(0)$$ in units of the signal, and the same divided by $$L^3$$.

| $$L$$ | Benincasa–Dowker | divided by $$L^3$$ | five-layer | divided by $$L^3$$ |
|---|---|---|---|---|
| 2 | 2.69 | 0.34 | 16.3 | 2.0 |
| 3 | 8.95 | 0.33 | 62.5 | 2.3 |
| 4 | 21.8 | 0.34 | 153 | 2.4 |
| 5 | 43.7 | 0.35 | 303 | 2.4 |
| 7 | 125 | 0.37 | 859 | 2.5 |
| 12 | 667 | 0.39 | 4550 | 2.6 |
| 20 | 3180 | 0.40 | 21700 | 2.7 |
| 32 | 13200 | 0.40 | | |

The ratio to $$L^3$$ lies between $$0.33$$ and $$0.40$$ for the Benincasa–Dowker weights over $$2\le L\le32$$ and between $$2.0$$ and $$2.7$$ for the five-layer weights over $$2\le L\le20$$, and it increases slowly with $$L$$; the table is not extrapolated. The first-chaos term was not evaluated for the spacelike carrier.

A heuristic for the cube: for fixed $$\tau_z$$ the Gaussian confines $$u$$ to a region of hyperbolic volume proportional to $$L^2/\tau_z^2$$. If the bracket in $$\Sigma$$ stays of order one there, which the zeros of $$\hat\psi$$ at $$\tfrac12$$, $$1$$ and $$\tfrac32$$ would arrange by cancelling the growth of the $$y$$-integral, then $$\Sigma\propto L^2$$, and with $$\lvert\Box\phi\rvert\propto L^{-2}$$ the ratio grows like $$L^3$$.

### Standard deviation by sampling

We sampled Palm configurations of a unit-density Poisson process in the past cone of $$0$$, keeping the points with interval volume to $$0$$ at most 16 and Euclidean norm at most $$3.3L$$ (the omitted points have total probability of order $$10^{-3}$$ or less), computed the interior counts exactly and evaluated $$B\phi(0)$$ for (3). The table gives $$\mathrm{std}(B\phi(0))/\lvert\Box\phi(0)\rvert$$ and the number of samples. The relative uncertainty of each entry is about 5%: resampling the samples (a bootstrap) gives 1% to 4%, and the laws are heavy tailed (rare very large values dominate the variance), so repeated runs scatter somewhat more.

| $$L$$ | samples | Benincasa–Dowker, timelike | Benincasa–Dowker, spacelike | five-layer, timelike | five-layer, spacelike |
|---|---|---|---|---|---|
| 2 | 4000 | 19 | 10 | 324 | 190 |
| 3 | 3000 | 87 | 50 | 1730 | 964 |
| 4 | 2000 | 209 | 136 | 4240 | 2680 |
| 5 | 1500 | 407 | 294 | 8340 | 6030 |
| 7 | 600 | 1110 | 906 | 22700 | 18200 |

For the timelike packet the standard deviation exceeds the first-chaos bound of the table above by a factor of 9 to 10 for the Benincasa–Dowker weights and 26 to 28 for the five-layer weights at $$L=3,5,7$$. The first chaos therefore carries about 1% of the variance of the Benincasa–Dowker operator and 0.1% of the five-layer operator, and the remainder is in the higher chaoses that (6) discards. The growth from $$L=3$$ to $$L=7$$ corresponds to exponents $$3.0,\,3.4,\,3.0,\,3.5$$ in the four columns, each uncertain by about $$0.1$$. The timelike columns follow the cube of the bound, and the spacelike ones grow slightly faster over this range.

A large standard deviation does not by itself exclude a heavy-tailed law with a small typical error, so we also examine the distribution of the error. For the Benincasa–Dowker weights and the timelike packet, the fraction of samples with $$\lvert B\phi(0)/\Box\phi(0)-1\rvert<0.05$$ is about $$3\cdot10^{-3}$$ at $$L=2$$ and below $$10^{-3}$$ at $$L=3,\dots,7$$. The median relative error is about 12, 58, 140 and 270 at $$L=2,3,4,5$$, uncertain by about 10%, which is 0.6 to 0.7 times the standard deviation.

On the Gaussian packets examined, with widths from $$2\ell$$ to $$7\ell$$ and both carriers, the single-realisation operator at the kernel scale is therefore dominated by fluctuations. The bounds and samples above are for $$\ell_k=\ell$$. A kernel with a non-locality scale $$\ell_k\gg\ell$$, or an average over many points of one realisation as in Sorkin's smeared operators [[Sorkin]](#ref-sorkin) [[Aslanbeigi–Saravani–Sorkin]](#ref-ass), may reduce the fluctuations; neither is evaluated here.

## 3. The cocompact orbit lattice

Let $$Q_7(x)=7x_0^2-x_1^2-x_2^2-x_3^2$$ on $$\mathbb Z^4$$ with polar form $$B_7(x,y)=7x_0y_0-\vec x\cdot\vec y$$, and let $$e(x)=7^{-1/8}(\sqrt7x_0,x_1,x_2,x_3)$$, so that $$\Lambda_7=e(\mathbb Z^4)\subset\mathbb R^{1,3}$$ has covolume 1 and squared proper time $$7^{-1/4}Q_7$$. Let $$\Gamma_7=SO^+(Q_7;\mathbb Z)$$, acting on $$\Lambda_7$$. A *realisation* is $$\Lambda=g\Lambda_7+\xi$$ with $$g\in SO^+(1,3)$$ and $$\xi\in\mathbb R^4$$; the *orbit lattice* is the random lattice with $$g$$ Haar distributed on $$SO^+(1,3)/\Gamma_7$$ and $$\xi$$ uniform on $$\mathbb R^4/g\Lambda_7$$. A vector $$k\in\mathbb Z^4$$ with $$Q_7(k)=n>0$$ and $$k_0>0$$ is said to lie on shell $$n$$.

Four classical results are used. The theorem of Borel and Harish-Chandra: for a semisimple $$\mathbb Q$$-group the group of integral points is a lattice. Godement's compactness criterion (see Borel's *Introduction aux groupes arithmétiques*): this lattice is cocompact if the $$\mathbb Q$$-group is anisotropic, that is, if the form has no non-trivial rational zero. Borel density: a lattice in a simple group without compact factors is Zariski dense. Mahler's compactness criterion: the set of unimodular lattices in $$\mathbb R^n$$ whose non-zero vectors all have Euclidean norm at least $$\delta>0$$ is compact.

<div class="theorem" markdown="1">

**Theorem 3** (the orbit lattice). (i) $$Q_7$$ has no non-trivial zero in $$\mathbb Z^4$$, so $$\Lambda_7$$ has no null vectors, and $$\lvert Q_7(v)\rvert \ge1$$ for $$v\in\mathbb Z^4\setminus\{0\}$$. (ii) $$\Gamma_7$$ is a cocompact lattice in $$SO^+(1,3)$$, hence Zariski dense. (iii) The random lattice $$\Lambda$$ is Poincaré invariant in law and ergodic under the Poincaré group, though not under translations alone. (iv) There is $$R_*<\infty$$ such that every lattice $$g\Lambda_7$$ has covering radius at most $$R_*$$. Consequently, for every realisation and every pair of lattice points $$x\prec y$$ with proper time $$\tau(x,y)>2\sqrt2R_*$$, the interval contains a lattice point, and the interior count satisfies $$m(x,y)\ge c_0V-C$$ with $$c_0=3\pi/256$$ and a constant $$C$$ independent of the realisation. (v) $$m(x,y)\equiv\mathbf 1[\,y-x\in2g\Lambda_7\,]\pmod2$$ for $$x,y\in\Lambda=g\Lambda_7+\xi$$. (vi) There is $$C$$ such that for every realisation and every causal interval $$S$$ of volume $$V\ge1$$,

$$
\lvert N_S-V\rvert \le C\,V^{3/4}, \tag{7}
$$

where $$N_S$$ is the number of lattice points in $$S$$. (vii) For $$k\in\mathbb Z^4$$ on shell $$n\in\{1,2,3,5\}$$ the interior count is $$m(k)=0$$; on shell 4 it is $$m(k)=1$$ if $$k\in2\mathbb Z^4$$ and $$m(k)=0$$ otherwise.

</div>

An anisotropic form keeps the lattice away from the light cone in every frame, so the whole boost orbit is a compact family of lattices; this is behind (i) to (iv). Item (v) is a point reflection, (vi) says that a bounded covering radius forces the lattice count to follow the volume up to a boundary term, and (vii) comes from the reverse triangle inequality for lattice vectors.

<div class="proof" markdown="1">

*Proof.* (i) Suppose $$7a^2=b^2+c^2+d^2$$ with integers not all zero, divided by their greatest common divisor so that they are coprime. If $$a$$ is odd, the left side is $$7\bmod 8$$, but squares mod 8 are $$0,1,4$$ and no sum of three of them is $$7$$. If $$a$$ is even, the right side is $$0\bmod4$$, which forces $$b,c,d$$ even, contradicting coprimality.

(ii) By (i) the $$\mathbb Q$$-group $$SO(Q_7)$$ is anisotropic; apply the Borel–Harish-Chandra theorem, Godement's criterion and Borel density.

(iii) The group of affine Lorentz maps acts transitively on the space $$(SO^+(1,3)\ltimes\mathbb R^4)/(\Gamma_7\ltimes\Lambda_7)$$ of realisations, which carries a finite invariant measure by (ii); a transitive action preserves the invariant measure and is ergodic. Translations alone fix the orientation of the lattice and are not ergodic.

(iv) In coordinates where the metric is Euclidean, $$\lvert v\rvert_{\rm E}^2=v_0^2+\lvert\vec v\rvert^2\ge\big\lvert v_0^2-\lvert\vec v\rvert^2\big\rvert$$, so by (i) every non-zero vector of every lattice $$g\Lambda_7$$ has Euclidean norm at least $$7^{-1/8}$$. By Mahler's criterion the orbit lies in a compact set of lattices, on which the covering radius is bounded by some $$R_*$$. In the rest frame of an interval of proper time $$\tau$$, which is a point of the same orbit, the interval contains the Euclidean ball of radius $$q=\tau/(2\sqrt2)$$ about its centre (a point at distance $$d$$ from the centre with time offset $$s$$ has $$\lvert\vec x\rvert<\tau/2-s$$ iff $$d^2<\tau^2/4-\tau s+2s^2$$, whose minimum over $$s$$ is $$\tau^2/8$$). This ball contains a lattice point once $$q>R_*$$. More precisely, the Voronoi cells of the lattice points have volume 1 and radius at most $$R_*$$, and they cover the concentric ball of radius $$q-R_*$$; the points belonging to these cells lie in the ball of radius $$q$$, so the ball contains at least $$\mathrm{vol}\,B_{q-R_*}=\tfrac{\pi^2}{2}(q-R_*)^4$$ lattice points. For $$q\ge2R_*$$ we have $$(q-R_*)^4\ge q^4/16$$, and $$q^4=\tau^4/64=3V/(8\pi)$$ with $$V=\pi\tau^4/24$$, which gives $$m\ge3\pi V/256$$. For $$q<2R_*$$ the volume is bounded, and the bound $$m\ge c_0V-C$$ holds with $$C=c_0\pi(4\sqrt2R_*)^4/24$$.

(v) The map $$z\mapsto x+y-z$$ is an involution of the lattice points in the open interval between $$x$$ and $$y$$, with the only possible fixed point $$(x+y)/2$$, which is a lattice point iff $$(y-x)/2\in g\Lambda_7$$.

(vi) Work in the rest frame of $$S$$, where $$S$$ is a convex body of diameter $$\sim\tau$$ with $$V\propto\tau^4$$ and boundary area $$\propto\tau^3$$. The Voronoi cells of the lattice points have volume 1 and radius at most $$R_*$$. The cells of points in $$S$$ lie in the parallel body $$S\oplus B_{R_*}$$, and the cells covering the inner parallel body $$S\ominus B_{R_*}$$ belong to points of $$S$$. Hence $$\mathrm{vol}(S\ominus B_{R_*})\le N_S\le\mathrm{vol}(S\oplus B_{R_*})$$. The collar of width $$R_*$$ inside a convex body has volume at most $$R_*\cdot\mathrm{area}(\partial S)$$, and by Steiner's formula $$\mathrm{vol}(S\oplus B_{R_*})-\mathrm{vol}(S)\le C(R_*\tau^3+R_*^2\tau^2+\dots)$$. Both are $$O(\tau^3)=O(V^{3/4})$$ for $$\tau\ge1$$.

(vii) For future timelike $$z,z'\in\mathbb Z^4$$, the reverse Cauchy–Schwarz inequality gives $$B_7(z,z')\ge\sqrt{Q_7(z)Q_7(z')}$$, with equality iff $$z,z'$$ are proportional. Hence $$\sqrt{Q_7(z+z')}\ge\sqrt{Q_7(z)}+\sqrt{Q_7(z')}$$. If $$z$$ lies in the interval between $$0$$ and $$k$$, then $$z'=k-z$$ is also future timelike, and $$Q_7(z),Q_7(z')\ge1$$ by (i). Also $$Q_7(k)=Q_7(z)+Q_7(z')+2B_7(z,z')$$ with $$B_7(z,z')$$ an integer, so $$Q_7(z)+Q_7(z')\equiv Q_7(k)\pmod2$$. For $$n\le3$$ the sum $$\sqrt{Q_7(z)}+\sqrt{Q_7(z')}$$ is at least $$2>\sqrt n$$, so no such $$z$$ exists. For $$n=5$$ the parity forces $$Q_7(z)+Q_7(z')$$ to be odd, and the smallest such pair $$(1,2)$$ has $$1+\sqrt2>\sqrt5$$. For $$n=4$$ the only pair with $$\sqrt{Q_7(z)}+\sqrt{Q_7(z')}\le2$$ is $$(1,1)$$, with equality, so $$z=z'=k/2$$, which lies in the lattice iff $$k\in2\mathbb Z^4$$. $$\square$$

</div>

For a Poisson sprinkling, almost every realisation contains empty intervals of arbitrarily large volume, since disjoint intervals are independent and each of volume $$V$$ is empty with probability $$e^{-V}$$; so no bound like (7) holds. Bound (7) gives $$\mathrm{Var}\,N_S\le C^2V^{3/2}$$, which is weaker than the Poisson value $$V$$.

### Number variance of the orbit lattice

For a causal diamond of volume $$V$$ with a uniformly random centre in a cell of $$\Lambda_7$$ and a random rest-frame velocity $$u$$, the lattice count has the law of the count of the orbit lattice in a fixed diamond. We drew $$u$$ with the hyperbolic density uniform on the ball of radius $$R_b$$ (an approximation to the invariant probability measure on $$\Gamma_7\backslash H^3$$ that improves as $$R_b$$ grows) and counted the points of $$\Lambda_7$$ in the diamond, with $$2\cdot10^6$$ samples per entry. The mean of $$N/V$$ was $$1.000\pm0.001$$ in each case, and $$\mathrm{Var}\,N/V$$ was:

| $$V$$ | $$R_b=3$$ | $$R_b=4$$ | $$R_b=5$$ | $$R_b=6$$ |
|---|---|---|---|---|
| 4 | 0.521 | 0.478 | 0.485 | 0.481 |
| 16 | 0.358 | 0.325 | 0.331 | 0.328 |
| 64 | 0.218 | 0.203 | 0.205 | 0.203 |

The ratio is below the Poisson value 1 and falls roughly like $$V^{-0.34}$$ between $$V=16$$ and $$64$$ (variance growing like $$V^{0.66}$$). The dependence on $$R_b$$ is the bias of the ball average, of about 8% between $$R_b=3$$ and $$4$$, 1% to 2% between $$R_b=4$$ and $$5$$, and below 1% between $$R_b=5$$ and $$6$$. Saravani and Aslanbeigi conjecture that no process has $$\mathrm{Var}\,N_S\le\alpha V$$ with $$\alpha<1$$ for all large intervals [[Saravani–Aslanbeigi]](#ref-sa). The conjecture is asymptotic, the orbit lattice is an admissible test of it, and data up to $$V=64$$ do not decide it. The same data bear on the question left open in [quantum sprinklings](/quantum-gravity/quantum-sprinklings/), whether a point process invariant in law under translations and boosts can have a number variance that grows more slowly than the volume. The orbit lattice is invariant in law, so the question applies to it; its variance-to-mean ratio is $$0.2$$ at $$V=64$$, the data do not show whether it tends to $$0$$, Corollary 8 there (pair correlations with integrable $$g-1$$) does not apply to it, and the proved bound (7) gives only $$\mathrm{Var}\,N_S\le C^2V^{3/2}$$.

## 4. Interval statistics and weights on the orbit lattice

Take a realisation through $$0$$, $$\Lambda=g\Lambda_7$$. Its future vectors are $$v=g\,e(k)$$ with $$k\in\mathbb Z^4$$ on a shell $$n$$, of proper time $$\tau_n$$ with $$\tau_n^2=7^{-1/4}n$$ and interval volume $$V_n=\pi\tau_n^4/24=\pi n^2/(24\sqrt7)$$. The interior count $$m(v)$$ depends on $$k$$ alone, since $$g$$ maps the lattice and the interval together. Let $$\rho_n$$ be the density, per unit hyperbolic measure (the volume element of the unit hyperboloid $$H^3$$, in which a ball of radius $$r$$ has volume $$\pi(\sinh2r-2r)$$), of vectors on shell $$n$$; by invariance it does not depend on $$g$$. Let $$P_n(m)$$ be the fraction of those vectors with interior count $$m$$. Since $$d^4y=\tau^3d\tau\,d\mu(u)=\tfrac6\pi\,dV\,d\mu(u)$$, unit density of the point set requires $$\sum_{n\le n_{\max}}\rho_n\approx\int_0^{\tau_{n_{\max}}}\tau^3d\tau=n_{\max}^2/(4\sqrt7)$$.

### Shell densities and interior-count laws

We enumerated the lattice vectors on the shells $$Q_7=n$$ with $$k_0\le k_{\max}$$ and computed the interior count of each by exact enumeration of the lattice points of the interval. For $$n=1,\dots,16$$ the densities $$\rho_n$$, counted in hyperbolic balls up to $$k_{\max}=1000$$, are $$1/16$$ times the integers $$4,12,6,12,12,18,25,12,28,36,48,18,36,75,32,12$$ to within $$0.3\%$$ for $$n\le15$$ and $$0.5\%$$ for $$n=16$$ (the counts converge slowly, because the error in hyperbolic lattice-point counts decays only like a power of the radius). The sum of the first sixteen is $$386/16=24.125$$, against $$16^2/(4\sqrt7)=24.19$$ from the unit-density requirement.

A $$\Gamma_7$$-orbit of vectors with stabiliser $$\Gamma_v$$ has hyperbolic density $$1/(\lvert\Gamma_v\rvert\,\mathrm{vol}\,\mathcal F)$$, with $$\mathcal F$$ a fundamental domain of $$\Gamma_7$$ in $$H^3$$, so every density $$\rho_nP_n(m)$$ is a sum of such terms. We have not computed $$\mathrm{vol}\,\mathcal F$$ and the stabilisers. The sixteen densities and the measured class densities in the table are the evidence that $$16\rho_nP_n(m)$$ is an integer for every class $$(n,m)$$, and we use this as an identification rule; the factor 16 is the smallest integer multiplier that makes the sixteen measured densities integral, and it is not derived.

The table gives the classes on shells 1 to 10. For shells 1, 2, 3 and 5 and for the split of shell 4 the laws are proved by Theorem 3(vii). For shells 6, 8, 9 and 10 the laws are the rational values for which every $$16\rho_nP_n(m)$$ is an integer and which agree with the enumeration; shell 7 is treated below. The last column is the measured $$16\rho_nP_n(m)$$ at the cutoff $$k_{\max}=300$$, from at least $$3.7\cdot10^5$$ vectors per shell.

| $$n$$ | $$16\rho_n$$ | $$m$$ and $$16\rho_nP_n(m)$$ | $$P_n(m)$$ | measured $$16\rho_nP_n(m)$$ |
|---|---|---|---|---|
| 1, 2, 3, 5 | 4, 12, 6, 12 | $$m=0$$: all | 1 | proved, Theorem 3(vii) |
| 4 | 12 | $$m=0$$: 8; $$m=1$$: 4 | 2/3, 1/3 | 8.01, 4.02 |
| 6 | 18 | $$m=0$$: 6; $$m=2$$: 12 | 1/3, 2/3 | 6.03, 12.06 |
| 7 | 25 | $$m=0$$: 1; $$m=2$$: 24 | 1/25, 24/25 | 1.00, 24.07 |
| 8 | 12 | $$m=3$$: 12 | 1 | 12.11 |
| 9 | 28 | $$m=2$$: 4; $$m=4$$: 24 | 1/7, 6/7 | 3.96, 23.92 |
| 10 | 36 | $$m=4$$: 24; $$m=6$$: 12 | 2/3, 1/3 | 24.22, 12.03 |

On shell 4 the counts are proved by Theorem 3(vii), and $$P_4(1)=\rho_1/\rho_4$$, since the vectors with $$m=1$$ are the doubles of the vectors on shell 1.

The laws on shell 7 follow from the structure of the interval and two computed inputs. An interior lattice point $$z$$ of a vector $$k$$ on shell 7 has $$Q_7(z)$$ and $$Q_7(z')$$, with $$z'=k-z$$, in $$\{1,2\}$$: both are at most $$(\sqrt7-1)^2<3$$ by the reverse triangle inequality of the proof of Theorem 3(vii), and their sum is odd. So $$m(k)=2N_1(k)$$, where $$N_1(k)$$ is the number of vectors $$z$$ on shell 1 with $$B_7(z,k)=3$$. Two distinct such vectors $$z_1,z_2$$ would give $$Q_7(z_1-z_2)=2-2B_7(z_1,z_2)\le-2$$ (since $$B_7(z_1,z_2)$$ is an integer, at least 1 by the reverse Cauchy–Schwarz inequality, with equality only for proportional vectors), with $$z_1-z_2$$ orthogonal to $$k$$; but the components of $$z_1$$ and $$z_2$$ orthogonal to $$k$$ have $$Q_7=1-9/7=-2/7$$ each, which gives $$-Q_7(z_1-z_2)\le8/7$$ by the triangle inequality for the negative-definite form on the orthogonal complement of $$k$$. Hence $$N_1\le1$$ and $$m\in\{0,2\}$$ on shell 7.

Each vector $$z$$ on shell 1 has exactly six vectors $$z'$$ on shell 2 with $$B_7(z,z')=2$$. Write $$z'=2z+u$$. Then $$B_7(z,u)=0$$ and $$Q_7(u)=-2$$, and $$z'$$ is future timelike because $$Q_7(z')=2$$ and $$B_7(z,z')>0$$. So the partners of $$z$$ are the vectors of norm 2 for $$-Q_7$$ in $$L_z=\{u\in\mathbb Z^4:B_7(z,u)=0\}$$. Since $$Q_7(z)=1$$, $$\mathbb Z^4=\mathbb Zz\oplus L_z$$, so $$L_z$$ is a positive definite ternary lattice of determinant 7. A reduced Gram matrix of such a lattice has $$a_{11}a_{22}a_{33}\le2\det$$, which leaves finitely many candidates, and three classes remain, with Gram matrices $$\mathrm{diag}(1,1,7)$$, $$(1)\oplus\begin{pmatrix}2&-1\\-1&4\end{pmatrix}$$ and $$\begin{pmatrix}2&-1&-1\\-1&2&0\\-1&0&3\end{pmatrix}$$. Over the 7-adic integers $$Q_7$$ is $$\langle7\rangle\oplus\langle-1\rangle^3$$, and the determinant of the part prime to 7 (an invariant of the form, modulo squares) is $$-1$$, which is not a square mod 7. That part is $$\langle1\rangle$$ plus the rank-2 part of $$L_z$$ prime to 7, and the sign of the form does not change a determinant of rank 2, so the latter has non-square determinant for $$-Q_7$$. For the three lattices it is $$\langle1,1\rangle$$, $$\langle1,2\rangle$$ and $$\langle2,\tfrac32\rangle$$, with determinants $$1$$, $$2$$ and $$3$$ mod 7, and the squares mod 7 are $$1,2,4$$. Hence $$L_z$$ is the third lattice, which has six vectors of norm 2. The point $$k=z+z'$$ then lies on shell 7 with $$B_7(z,k)=3$$, so $$m(k)=2$$. Counting the $$\Gamma_7$$-orbits of these pairs by $$z$$ and by $$k$$ gives $$\rho_7P_7(2)=6\rho_1$$, that is $$P_7(0)=1-6\rho_1/\rho_7$$.

The ratio $$\rho_7/\rho_1$$ decides between $$P_7(0)=1/25$$ (ratio $$25/4=6.25$$) and $$1/24$$ (ratio $$6.261$$). At single cutoffs $$500\le k_{\max}\le1000$$ it lies between $$6.242$$ and $$6.264$$, which is the size of the gap, so we average over cutoffs; the average is $$6.250$$. This is numerical support without an error bound. The Siegel product of the local densities of $$Q_7$$ at $$2$$, $$7$$ and the primes dividing $$n$$ gives $$25/4$$, and all sixteen integers above for $$n\le16$$; we have not proved that it applies to these hyperbolic densities. A count of the $$m=0$$ vectors alone does not separate the two values.

On shells 11 to 60 every vector examined with $$k_0\le24$$ has $$m\ge6$$. Theorem 3(iv) gives $$m\ge c_0V-C$$ for all large $$n$$, uniformly in the realisation, but with no explicit threshold, so shells above 60 are not examined. Since $$w_m=0$$ for $$m\ge5$$, only shells 1 to 10 enter the sums below, on this numerical basis.

### Weights that reproduce the mean

The mean of (1) at $$0$$ is $$\nu[a\phi(0)+\sum_{n,m}\rho_nP_n(m)w_m\int_{H^3}\phi(-\tau_nu)\,d\mu(u)]$$. It has the form of the Poisson mean with the measure $$\psi(V)\,dV$$ replaced by $$\tfrac\pi6\sum_n\rho_n\langle w\rangle_n\delta(V-V_n)$$, where $$\langle w\rangle_n=\sum_mP_n(m)w_m$$. Because $$w_m=0$$ for $$m\ge5$$ and the interior counts on shells above 10 are at least 6 in the range examined, the Mellin transform is a finite Dirichlet polynomial,

$$
\hat\psi_\Lambda(s)=\frac\pi6\sum_{n=1}^{10}\rho_n\sum_mP_n(m)\,w_m\,V_n^{\,s-1}, \tag{8}
$$

which is entire. On the Haar-random lattice in two dimensions, by contrast, the empty-interval density decays like $$1/V$$ and the zeroth moment diverges logarithmically ([Lorentz-invariant point sets](/quantum-gravity/invariant-point-sets/)). The derivation of the infrared conditions uses only the Mellin transform of the radial density, and we assume that it carries over. The conditions become

$$
\hat\psi_\Lambda(\tfrac12)=\hat\psi_\Lambda(1)=\hat\psi_\Lambda(\tfrac54)=\hat\psi_\Lambda(\tfrac32)=0,\qquad \hat\psi_\Lambda'(\tfrac32)=\frac{\sqrt\pi}3,\qquad a=-3\hat\psi_\Lambda'(1), \tag{9}
$$

with $$\nu=4/\sqrt6$$. The condition at $$\tfrac54$$, the extra zero of the five-layer set, is added so that there are five linear conditions for the five weights $$w_0,\dots,w_4$$; the system has condition number $$1.8\cdot10^5$$ in the 1-norm. With the laws of the table above, its solution is

$$
(w_0,\dots,w_4)=(2.93994598436,\,-194.985218449,\,104.163596525,\,-409.986325849,\,28.9411577030),\qquad a=-0.798641063627.
$$

The mean ratio $$\mathbb E[B\phi(0)]/\Box\phi(0)$$, computed by the quadrature of section 1, is for $$L=5,10,20,40$$:

| packet | $$L=5$$ | $$L=10$$ | $$L=20$$ | $$L=40$$ |
|---|---|---|---|---|
| timelike | 0.971 | 1.256 | 1.127 | 1.047 |
| spacelike | 0.300 | 0.645 | 0.855 | 0.948 |

The ratio tends to 1 and is close to the values of the five-layer Poisson kernel in section 1. This follows from the construction when the laws are those of the lattice, so the table does not test the laws. The weights depend on the laws sensitively. Weights solved for $$P_6(0)=1/3$$ and applied with $$P_6(0)=0.353$$ change the timelike ratio at $$L=10,20,40$$ from $$1.26,\,1.13,\,1.05$$ to $$-3.0\cdot10^3,\,-4.3\cdot10^4,\,-6.6\cdot10^5$$, because a shift of $$0.02$$ in one probability breaks the cancellation of the constant mode and the error is amplified by $$L^4$$. Weights solved for $$P_7(0)=1/24$$ and applied with $$P_7(0)=1/25$$, a shift of $$0.0017$$, give $$3.1\cdot10^2$$, $$4.4\cdot10^3$$ and $$6.7\cdot10^4$$ if the shell density $$16\rho_7=25$$ is kept and only the split into the classes $$m=0$$ and $$m=2$$ changes; if instead the class $$m=2$$ is kept at $$24$$, as the pair count requires, they give $$-8.0$$, $$-1.3\cdot10^2$$ and $$-2.0\cdot10^3$$. Rounding the weights does the same: at nine significant digits the ratio at $$L=40$$ is $$1.37$$, and twelve digits reproduce $$1.047$$. Reproducing the mean on the orbit lattice therefore needs the exact laws and weights to twelve digits, and the identification of the laws in the table is the input.

## 5. A single realisation of the orbit lattice

Take a realisation $$\Lambda$$ through $$0$$ and apply a Lorentz transformation so that $$\Lambda=g\Lambda_7$$. For a future vector $$v\in\Lambda$$ let $$\kappa(v)=\nu\,w_{m(v)}$$, where $$m(v)$$ is the number of points of $$\Lambda$$ in the interval $$(0,v)$$, and let $$\kappa(v)=0$$ for other $$v$$. Then $$(B\phi)(x)=\nu a\phi(x)+\sum_v\kappa(v)\phi(x-v)$$, and the kernel $$\kappa$$ is bounded and supported on finitely many shells (Theorem 3(iv); numerically on shells 1 to 10 for the weights of section 4).

<div class="proposition" markdown="1">

**Proposition 4** (a layer operator on the orbit lattice has no symbol). The stabiliser $$\Gamma_g=g\Gamma_7g^{-1}$$ of $$\Lambda$$ in $$SO^+(1,3)$$ is Zariski dense, and the kernel $$\kappa$$ is bounded, $$\Gamma_g$$-invariant, supported on finitely many shells, and non-zero at the vectors of shell 1 if $$w_0\ne0$$. Therefore, by Proposition 1 of [lattices with an arithmetic Lorentz group](/quantum-gravity/discrete-lorentz-group/), the Fourier series $$\hat\kappa$$ of the kernel coincides with an integrable function on no open set of the Brillouin torus; and for every finitely supported $$\phi\ne0$$ on $$\Lambda$$, the function $$B\phi-\nu a\phi$$ is not square summable. Moreover every bounded operator on $$\ell^2(\Lambda)$$ that commutes with the translations of $$\Lambda$$ and with $$\Gamma_g$$ is a multiple of the identity, so it cannot have the symbol of $$\Box$$ on any open set of momenta.

</div>

<div class="proof" markdown="1">

*Proof.* $$\Gamma_g$$ is conjugate to $$\Gamma_7$$, which is Zariski dense by Theorem 3(ii). It preserves $$\Lambda$$, the form and the time orientation, hence the interval counts: $$m(\gamma v)=m(v)$$. So $$\kappa$$ is $$\Gamma_g$$-invariant; it is bounded because only the finitely many values $$\nu w_m$$ occur, and it vanishes outside finitely many shells by Theorem 3(iv). The vectors of shell 1 have $$m=0$$ by Theorem 3(vii), and shell 1 is non-empty (it contains $$g\,e(1,2,1,1)$$), so $$\kappa\ne0$$ there if $$w_0\ne0$$. Proposition 1 of that page applies and gives the first statement. Suppose $$\phi\ne0$$ is finitely supported and $$\widehat{B\phi-\nu a\phi}=\hat\kappa\hat\phi$$ were in $$L^2$$, where $$\hat\phi$$ is a non-zero trigonometric polynomial. On the open set $$U=\{\hat\phi\ne0\}$$, which is non-empty, $$\hat\kappa=(\hat\kappa\hat\phi)/\hat\phi$$ would be locally square integrable, contradicting Proposition 1. The last statement is Theorem 1(a) of the same page, applied to $$\Lambda$$ and $$\Gamma_g$$. $$\square$$

</div>

So the response to a localised field is not a localised field: a retarded layer sum with weights tuned to give the correct mean maps any compactly supported test field to a function of infinite $$\ell^2$$ norm. A Gaussian packet is not finitely supported and the sums converge, so Proposition 4 does not give the size of the error for it; that depends on cancellations among the shell sums, which we evaluate.

### Errors on four packets

We took the unboosted realisation ($$g=1$$, $$\xi=0$$) with the weights, $$a$$ and $$\nu$$ of section 4, computed the interior count of every vector $$v$$ with $$1\le Q_7(v)\le10$$ in the packet's window by exact enumeration, and evaluated $$B\phi(0)$$ for the packet (3) at rest and boosted to speed $$\beta=0.6$$ along $$(1,2,3)/\sqrt{14}$$. The relative error $$\lvert B\phi(0)/\Box\phi(0)-1\rvert$$ is:

| packet | $$L=5$$ | $$L=10$$ | $$L=20$$ | $$L=40$$ |
|---|---|---|---|---|
| timelike, $$\beta=0$$ | $$5.9\cdot10^3$$ | $$9.9\cdot10^3$$ | $$2.6\cdot10^5$$ | $$4.2\cdot10^6$$ |
| spacelike, $$\beta=0$$ | $$5.9\cdot10^3$$ | $$3.1\cdot10^4$$ | $$1.0\cdot10^5$$ | $$3.0\cdot10^6$$ |
| timelike, $$\beta=0.6$$ | $$3.8\cdot10^2$$ | $$3.9\cdot10^4$$ | $$3.4\cdot10^5$$ | not computed |
| spacelike, $$\beta=0.6$$ | $$2.9\cdot10^3$$ | $$1.8\cdot10^4$$ | $$1.3\cdot10^4$$ | not computed |

The windows $$k_0\le100$$ and $$150$$ give the same values to the digits shown. One realisation, one lattice frame and four packets (two carriers, two velocities) give no estimate of the typical error or of its growth exponent, and the values range over four orders of magnitude. The error is not small at any $$L$$ tried, and at rest it is larger at $$L=40$$ than at $$L=5$$ by factors of $$7\cdot10^2$$ and $$5\cdot10^2$$. At $$L=5$$ the errors lie in the range of the Poisson standard deviations of section 2 ($$3\cdot10^2$$ to $$8\cdot10^3$$). At $$L=20$$ they lie between $$10^4$$ and $$4\cdot10^5$$, against the first-chaos bounds $$3\cdot10^3$$ and $$2\cdot10^4$$ of the two Poisson kernels.

## 6. Superposed lattices

Independent randomness can be added to the lattice by superposition, and Proposition 5 gives its effect on the number statistics only. Let $$\Lambda_K$$ be the union of $$K$$ independent copies of the random orbit lattice, each of density $$1/K$$ (a copy is the density-one random lattice scaled by $$K^{1/4}$$), so that the density is 1.

<div class="proposition" markdown="1">

**Proposition 5** (number variance of a superposition). For every causal diamond $$D$$ of volume $$V$$, $$\mathrm{Var}\,N_K(D)/\mathbb E\,N_K(D)=\theta(V/K)$$, where $$\theta(v)=\mathrm{Var}\,N_1(D')/\mathbb E\,N_1(D')$$ is the ratio for a single density-one copy and a diamond $$D'$$ of volume $$v$$. Moreover $$\theta(v)\to1$$ as $$v\to0$$, so for fixed $$V$$ the ratio tends to 1 as $$K\to\infty$$.

</div>

<div class="proof" markdown="1">

*Proof.* The counts $$N_j(D)$$ of the independent copies are independent. Scaling by $$K^{-1/4}$$ maps a copy to a density-one copy and $$D$$ to a diamond of volume $$V/K$$, and the law of the random lattice is invariant under Lorentz transformations, so $$N_j(D)$$ has the law of $$N_1(D')$$ with $$\mathrm{vol}\,D'=V/K$$. Then $$\mathrm{Var}\,N_K=K\,\mathrm{Var}\,N_1(D')=K\,\theta(V/K)\,\mathbb E N_1(D')$$, and $$\mathbb E N_K=V=K\,\mathbb E N_1(D')$$. For the limit: a diamond of proper time $$T$$ has Euclidean diameter at most $$T$$ in its rest frame, where every lattice of the orbit has no non-zero vector shorter than $$\delta=7^{-1/8}$$ (proof of Theorem 3(iv)). For $$T<\delta$$ the diamond contains at most one lattice point, so $$N_1(D')\in\{0,1\}$$ and $$\theta=1-\mathbb E N_1(D')\to1$$. $$\square$$

</div>

As $$K\to\infty$$ at fixed $$V$$ the variance-to-mean ratio of the counts approaches the Poisson value 1. The operator error is large already for the Poisson sprinkling (section 2), so the counts do not determine it, and the operator on $$\Lambda_K$$ for $$K\ge2$$ is not computed.

## 7. Frames and open quantities

A smearing on the orbit lattice that uses a frame, for instance a time direction, breaks the invariance of the operator under $$\Gamma_g$$, and Proposition 4 does not apply to it. It brings back the questions of [frame leakage](/quantum-gravity/lorentz-violation-leakage/).

The number variance of the orbit lattice grows like $$V^{0.66}$$ between $$V=16$$ and $$V=64$$, and the proved bound is $$V^{3/2}$$; larger diamonds, or a proof, would fix the asymptotic law. For the Poisson operator, (6) accounts for about 1% of the variance for the Benincasa–Dowker weights and the timelike carrier, so an analytic form of the standard deviation needs the higher chaoses; the first chaos is not evaluated for the spacelike carrier, and the growth with $$L$$ beyond the sampled range is not known. For a single realisation of the orbit lattice, four packets in one frame give errors from $$4\cdot10^2$$ to $$4\cdot10^6$$ and fix no growth exponent; packets of more widths and in more frames would show whether the exponent differs from the value 3.0 to 3.5 found for the Poisson sprinkling.

## References

1. <a id="ref-bd"></a>Benincasa, Dowker, "The scalar curvature of a causal set," Phys. Rev. Lett. 104 (2010) 181301, [arXiv:1001.2725](https://arxiv.org/abs/1001.2725). Used for: the layer weights $$(1,-9,16,-8)$$.
2. <a id="ref-ass"></a>Aslanbeigi, Saravani, Sorkin, "Generalized causal set d'Alembertians," JHEP 06 (2014) 024, [arXiv:1403.1622](https://arxiv.org/abs/1403.1622). Used for: the conditions on the weights in section 1 and the smeared operators with a non-locality scale.
3. <a id="ref-sorkin"></a>Sorkin, "Does Locality Fail at Intermediate Length-Scales," [arXiv:gr-qc/0703099](https://arxiv.org/abs/gr-qc/0703099). Used for: the non-locality scale $$\ell_k$$ and smeared operators.
4. <a id="ref-sa"></a>Saravani, Aslanbeigi, "On the causal set-continuum correspondence," [arXiv:1403.6429](https://arxiv.org/abs/1403.6429). Used for: the number-variance conjecture of section 3.
5. <a id="ref-lp"></a>Last, Penrose, "Poisson process Fock space representation, chaos expansion and covariance inequalities," [arXiv:0909.3205](https://arxiv.org/abs/0909.3205). Used for: the chaos expansion of the variance, inequality (4).
