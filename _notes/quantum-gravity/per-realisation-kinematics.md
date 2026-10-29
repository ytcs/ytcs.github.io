---
title: Kernel-scale kinematics of a single realisation
date: 2026-10-29
category: substrate
order: 6
description: On a Poisson sprinkling, the standard deviation of a layer-weighted d'Alembertian of kernel scale on a smooth wave packet exceeds the signal by at least about one third of the cube of the packet width in kernel units (a proved bound, evaluated numerically), and on the cocompact orbit lattice of an anisotropic integral form the same kind of operator can have the correct mean but has no symbol on a single realisation.
status: The variance inequality (6), the structure of the orbit lattice (Theorem 3), the absence of a symbol (Proposition 4) and the superposition identity (Proposition 5) are proved. The size of (6), the sampled standard deviations, the shell densities, interior-count laws and weights of the orbit lattice and the single-realisation errors are numerical, with sample sizes and ranges stated; the rational values of the interior-count laws for shells 6 to 10 are identified from enumeration.
excludes: For the Benincasa–Dowker and a five-layer kernel on a Poisson sprinkling and a timelike Gaussian packet of width 2 to 32 discreteness lengths, accuracy of a single realisation (numerical evaluation of a proved bound); on the orbit lattice of the integral form with coefficients (7, -1, -1, -1), a symbol for the layer kernel, and a bounded smearing that commutes with the translations and the stabiliser of the realisation and is not a multiple of the identity (proved). Kernels with a non-locality scale much larger than the discreteness scale, and smearings that use a frame, are not covered.
---

A causal-set d'Alembertian is a sum over the causal past of a point, with a weight that depends on the number $$m$$ of points in the causal interval between the two. Its expectation on a Poisson sprinkling of density $$\ell^{-4}$$ is the d'Alembertian $$\Box\phi$$ of a smooth field $$\phi$$ [[Benincasa–Dowker]](#ref-bd) [[Aslanbeigi–Saravani–Sorkin]](#ref-ass). We ask whether one realisation computes $$\Box\phi$$ when the kernel sits at the scale $$\ell$$ of the discreteness and $$\phi$$ varies on a scale $$L\gg\ell$$. For the Benincasa–Dowker weights and for a five-layer variant the answer is no. For a Gaussian packet of width $$L$$ between $$2\ell$$ and $$32\ell$$, the standard deviation of the operator at a point is at least $$0.33\,(L/\ell)^3\,\lvert\Box\phi\rvert$$ (Benincasa–Dowker weights, packet oscillating in time): the inequality (6) behind this is proved, and its value is computed numerically. Sampled standard deviations are $$3\cdot10^2$$ to $$8\cdot10^3$$ times $$\lvert\Box\phi\rvert$$ at $$L=5\ell$$, depending on the kernel.

We also ask whether a lattice, which carries no Poisson randomness within a realisation, does better. We take the lattice $$\mathbb Z^4$$ with the anisotropic form $$7x_0^2-x_1^2-x_2^2-x_3^2$$, embedded in Minkowski space and randomly boosted and translated (the orbit lattice of section 3), which is Poincaré invariant in law. Its weights can be retuned to the interval statistics of the lattice so that the mean reproduces $$\Box\phi$$. A single realisation, however, carries an operator with no symbol, and the operator maps every non-zero finitely supported field outside the square-summable functions (Proposition 4, proved). The relative errors computed on a few Gaussian packets lie between $$4\cdot10^2$$ and $$4\cdot10^6$$; Proposition 4 does not bound them.

Sorkin argued that Lorentz-invariant discreteness requires a non-locality scale $$\ell_k\gg\ell$$ [[Sorkin]](#ref-sorkin), and the generalised d'Alembertians of Aslanbeigi, Saravani and Sorkin [[Aslanbeigi–Saravani–Sorkin]](#ref-ass) are families with such a scale. Saravani and Aslanbeigi studied number fluctuations of Poisson sprinklings and of two-dimensional Lorentzian lattices [[Saravani–Aslanbeigi]](#ref-sa). The lower bound below uses the first term of the chaos expansion of a Poisson functional [[Last–Penrose]](#ref-lp), and its evaluation for the four-dimensional kernels is what is added here, together with the structure of the orbit lattice. The conditions on the weights that make the mean work are derived on [the page on signed and mixed Poisson substrates](/quantum-gravity/signed-substrates/); the invariant random lattices are described on [the page on Lorentz-invariant point sets](/quantum-gravity/invariant-point-sets/); the argument that a Zariski-dense stabiliser forbids a symbol is on [the page on exact discrete Lorentz symmetry](/quantum-gravity/discrete-lorentz-group/).

## 1. Layer operators at the kernel scale

Let $$\Phi$$ be a locally finite set in $$\mathbb R^{1,3}$$ with signature $$(+,-,-,-)$$, so that $$\Box=\partial_t^2-\nabla^2$$ (with signature $$(-,+,+,+)$$ the sign of $$\Box$$ and that of $$\nu$$ change), and let $$x\in\Phi$$. For $$y\prec x$$ in the causal past, let $$m(y,x)$$ be the number of points of $$\Phi$$ in the open interval between $$y$$ and $$x$$. A *layer operator* with weights $$(w_0,\dots,w_M)$$, local coefficient $$a$$ and normalisation $$\nu$$ acts on a function $$\phi$$ by

$$
(B\phi)(x)=\nu\Big[a\,\phi(x)+\sum_{y\in\Phi,\ y\prec x}w_{m(y,x)}\,\phi(y)\Big],\qquad w_m=0\ \ (m>M). \tag{1}
$$

For a Poisson process of unit density the interval between $$y$$ and $$x$$ has volume $$V=\pi\tau^4/24$$, with $$\tau$$ the proper time, and $$m$$ is Poisson distributed with mean $$V$$. Hence $$\mathbb E\,(B\phi)(x)=\nu[a\phi(x)+\int_{y\prec x}\psi(V(y,x))\phi(y)\,d^4y]$$ with

$$
\psi(V)=\sum_mw_m\,e^{-V}\frac{V^m}{m!}. \tag{2}
$$

The Mellin transform of $$\psi$$ is $$\hat\psi(s)=\int_0^\infty\psi(V)V^{s-1}dV=\sum_mw_m\Gamma(s+m)/m!$$. The conditions of Aslanbeigi, Saravani and Sorkin for the mean to tend to $$\Box\phi$$ are $$\hat\psi(\tfrac12)=\hat\psi(1)=\hat\psi(\tfrac32)=0$$ and $$a=-3\hat\psi'(1)$$, with $$\nu$$ fixed by the value of $$\hat\psi'(\tfrac32)$$ [[Aslanbeigi–Saravani–Sorkin]](#ref-ass). They are derived on the page on signed and mixed Poisson substrates linked above, and we use them as given. We use two weight sets, both with $$\lvert\hat\psi'(\tfrac32)\rvert=\sqrt\pi/3$$ and $$\lvert\nu\rvert=4/\sqrt6$$.

The Benincasa–Dowker weights are $$(1,-9,16,-8)$$, for which $$\hat\psi(s)=-\tfrac13\Gamma(s)(2s-1)(s-1)(2s-3)$$, so that $$\hat\psi'(1)=\tfrac13$$ (hence $$a=-1$$) and $$\hat\psi(\tfrac54)=0.0567$$. The five-layer set $$(5,-85,280,-328,128)$$ has $$\hat\psi(s)=\tfrac{16}3\Gamma(s)(s-\tfrac12)(s-1)(s-\tfrac54)(s-\tfrac32)$$; it has the extra zero at $$\tfrac54$$, which removes the term of order $$\ell^{-1}$$ proportional to the timelike boundary volume that the Benincasa–Dowker action has on a region (see [the page on boundary terms](/quantum-gravity/boundary-terms/)), and it is used here to test how the fluctuation depends on the weights. It has $$\hat\psi'(1)=\tfrac13$$, so $$a=-1$$ as well.

The test field is a Gaussian packet,

$$
\phi(x)=e^{-\lvert x\rvert _{\rm E}^2/2L^2}\cos(p\cdot x),\qquad p=3/L, \tag{3}
$$

where $$\lvert x\rvert_{\rm E}^2=t^2+\lvert\vec x\rvert^2$$, with the carrier along the time axis (timelike, $$\cos pt$$) or along $$x_1$$ (spacelike, $$\cos px_1$$). Lengths are in units of $$\ell=1$$. Then $$\Box\phi(0)=2/L^2-p^2=-7/L^2$$ (timelike) or $$2/L^2+p^2=11/L^2$$ (spacelike).

The mean of (1) at a point of a Poisson sprinkling, computed by quadrature in hyperbolic coordinates and divided by $$\Box\phi(0)$$ (the sign of $$\nu$$ is chosen so that the ratio tends to $$+1$$), is, for $$L=3,5,10,20,40$$:

| kernel and carrier | $$L=3$$ | $$L=5$$ | $$L=10$$ | $$L=20$$ | $$L=40$$ |
|---|---|---|---|---|---|
| Benincasa–Dowker, timelike | 0.82 | 1.15 | 1.12 | 1.05 | 1.02 |
| Benincasa–Dowker, spacelike | 0.48 | 0.67 | 0.86 | 0.95 | 0.98 |
| five-layer, timelike | $$-0.47$$ | 0.98 | 1.24 | 1.12 | 1.04 |
| five-layer, spacelike | 0.08 | 0.33 | 0.66 | 0.86 | 0.95 |

The mean ratio tends to 1 as $$L$$ grows, with corrections that fall like a power of $$\ell/L$$; for the five-layer kernel at $$L=3$$ it has the opposite sign. The question is the fluctuation around this mean.

## 2. The variance on a Poisson sprinkling

We evaluate $$B\phi$$ at a point of the sprinkling placed at the origin. By the Mecke formula the other points then form a Poisson process $$\Phi$$ of unit density (the Palm configuration), and $$B\phi(0)=\nu X$$ with $$X=a\phi(0)+\sum_{y\in\Phi,\,y\prec0}w_{m(y,0)}\phi(y)$$. For a Poisson process $$\eta$$ with $$\sigma$$-finite intensity measure $$\lambda$$ and a square-integrable functional $$F(\eta)$$, put $$D_zF(\eta)=F(\eta+\delta_z)-F(\eta)$$, the cost of adding the point $$z$$. The Wiener–Itô chaos expansion gives $$\mathrm{Var}\,F=\sum_{n\ge1}\frac1{n!}\int\lvert \mathbb E\,D^n_{z_1\dots z_n}F\rvert ^2\lambda^n(dz)$$ [[Last–Penrose]](#ref-lp), so that

$$
\mathrm{Var}\,F\ \ge\ \int\big\lvert \mathbb E\,D_zF\big\rvert ^2\lambda(dz). \tag{4}
$$

The terms with $$n\ge2$$ are discarded in (4); the first term is the only one we compute. The functional $$X$$ is square integrable for the packet (3), since its first and second absolute moments are bounded by integrals of $$\lvert\phi\rvert$$ and $$\phi^2$$ against $$\mathbb P(m(y,0)\le M)$$, which converge by the Gaussian decay of $$\phi$$.

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

Inequality (6) holds for every $$L$$ and every weight set; it is (4) with $$\lambda$$ the Lebesgue measure of the past cone and (5) inserted. Its size is a numerical question.

*Evaluation of the bound.* For the timelike packet, write $$z=-\tau_zu$$ and $$y=-\tau_yu'$$ with $$u,u'$$ unit future vectors, that is, points of the hyperbolic space $$H^3$$. Then $$y\prec z$$ holds exactly when $$\tau_y>\tau_z$$ and the hyperbolic distance of $$u$$ and $$u'$$ is smaller than $$\ln(\tau_y/\tau_z)$$. The packet depends on $$u'$$ only through its rapidity, and the integral over the angle at $$u$$ is that of a Gaussian times a cosine, which is a difference of complex error functions. What remains is a four-dimensional quadrature, which we carried out on two grids that agree to $$0.6\%$$. The table gives $$\lvert\nu\rvert\sqrt{\Sigma}/\lvert\Box\phi(0)\rvert$$, which by (6) is a lower bound for the standard deviation of $$B\phi(0)$$ in units of the signal, and the same divided by $$L^3$$.

| $$L$$ | Benincasa–Dowker | divided by $$L^3$$ | five-layer | divided by $$L^3$$ |
|---|---|---|---|---|
| 2 | 2.69 | 0.34 | 16.3 | 2.0 |
| 3 | 8.95 | 0.33 | 62.5 | 2.3 |
| 4 | 21.8 | 0.34 | 154 | 2.4 |
| 5 | 43.7 | 0.35 | 305 | 2.4 |
| 7 | 125 | 0.37 | 859 | 2.5 |
| 12 | 667 | 0.39 | 4570 | 2.6 |
| 20 | 3180 | 0.40 | 21700 | 2.7 |
| 32 | 13200 | 0.40 | | |

The ratio to $$L^3$$ lies between $$0.33$$ and $$0.40$$ for the Benincasa–Dowker weights over $$2\le L\le32$$ and between $$2.0$$ and $$2.7$$ for the five-layer weights over $$2\le L\le20$$, and it increases slowly with $$L$$; the table is not extrapolated. The spacelike carrier was not evaluated. A heuristic for the cube: for fixed $$\tau_z$$ the Gaussian confines $$u$$ to a region of hyperbolic volume proportional to $$L^2/\tau_z^2$$; if the bracket in $$\Sigma$$ stays of order one there, as the zeros of $$\hat\psi$$ at $$\tfrac12$$, $$1$$ and $$\tfrac32$$ arrange by cancelling the growth of the $$y$$-integral, then $$\Sigma\propto L^2$$, and with $$\lvert\Box\phi\rvert\propto L^{-2}$$ the ratio grows like $$L^3$$.

*Sampled standard deviations.* We sampled Palm configurations of a unit-density Poisson process in the past cone of $$0$$, keeping the points with interval volume to $$0$$ at most 16 and Euclidean norm at most $$3.3L$$ (the omitted points carry weight $$w_m$$ with $$m$$ large, of total probability of order $$10^{-3}$$ or less), computed the interior counts exactly and evaluated $$B\phi(0)$$ for (3). The table gives $$\mathrm{std}(B\phi(0))/\lvert\Box\phi(0)\rvert$$, with the number of samples; the relative standard error of each entry, from a bootstrap, is 1% to 4% (the laws are heavy tailed, so the Gaussian estimate of the error would be too small).

| $$L$$ | samples | Benincasa–Dowker, timelike | Benincasa–Dowker, spacelike | five-layer, timelike | five-layer, spacelike |
|---|---|---|---|---|---|
| 2 | 4000 | 19 | 10 | 324 | 190 |
| 3 | 3000 | 87 | 50 | 1730 | 964 |
| 4 | 2000 | 209 | 136 | 4240 | 2680 |
| 5 | 1500 | 407 | 294 | 8340 | 6030 |
| 7 | 600 | 1110 | 906 | 22700 | 18200 |

For the timelike packet the standard deviation exceeds the first-chaos bound of the table above by a factor of 9 to 10 for the Benincasa–Dowker weights and 26 to 28 for the five-layer weights at $$L=3,5,7$$. The first chaos therefore carries about 1% of the variance of the Benincasa–Dowker operator and 0.1% of the five-layer operator; the remainder is in the higher chaoses, which (6) discards. The growth from $$L=3$$ to $$L=7$$ corresponds to exponents $$3.0,\,3.4,\,3.0,\,3.5$$ in the four columns, each with an uncertainty of about $$0.1$$: the timelike columns follow the cube of the bound, the spacelike ones grow slightly faster over this range.

A large standard deviation does not by itself exclude a heavy-tailed law with a small typical error, we also examine the distribution of the error. The fraction of samples with $$\lvert B\phi(0)/\Box\phi(0)-1\rvert<0.05$$ for the Benincasa–Dowker weights and the timelike packet is $$3\cdot10^{-3}$$ at $$L=2$$, $$3\cdot10^{-4}$$ at $$L=3$$ and zero in all samples at $$L\ge4$$ ($$2000$$, $$1500$$ and $$600$$ samples at $$L=4,5,7$$); the median relative error is 12, 58, 144 and 257 at $$L=2,3,4,5$$. The typical error is of the order of the standard deviation.

At kernel scale, then, the single-realisation operator is dominated by fluctuations for any smooth $$\phi$$ of width $$L\gtrsim2\ell$$. Reducing them needs a kernel with a non-locality scale $$\ell_k\gg\ell$$, averaging over many points as in Sorkin's smeared operators [[Sorkin]](#ref-sorkin) [[Aslanbeigi–Saravani–Sorkin]](#ref-ass); the dependence of the bound on $$\ell_k$$ is open.

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

(vii) For future timelike $$z,w\in\mathbb Z^4$$, the reverse Cauchy–Schwarz inequality gives $$B_7(z,w)\ge\sqrt{Q_7(z)Q_7(w)}$$, with equality iff $$z,w$$ are proportional. Hence $$\sqrt{Q_7(z+w)}\ge\sqrt{Q_7(z)}+\sqrt{Q_7(w)}$$. If $$z$$ lies in the interval between $$0$$ and $$k$$, then $$w=k-z$$ is also future timelike, and $$Q_7(z),Q_7(w)\ge1$$ by (i). Also $$Q_7(k)=Q_7(z)+Q_7(w)+2B_7(z,w)$$ with $$B_7(z,w)$$ an integer, so $$Q_7(z)+Q_7(w)\equiv Q_7(k)\pmod2$$. For $$n\le3$$ the left side is at least $$2>\sqrt n$$, so no such $$z$$ exists. For $$n=5$$ the parity forces $$Q_7(z)+Q_7(w)$$ to be odd, and the smallest such pair $$(1,2)$$ has $$1+\sqrt2>\sqrt5$$. For $$n=4$$ the only pair with $$\sqrt{Q_7(z)}+\sqrt{Q_7(w)}\le2$$ is $$(1,1)$$, with equality, so $$z=w=k/2$$, which lies in the lattice iff $$k\in2\mathbb Z^4$$. $$\square$$

</div>

For a Poisson sprinkling, almost every realisation contains empty intervals of arbitrarily large volume, since disjoint intervals are independent and each of volume $$V$$ is empty with probability $$e^{-V}$$; so no bound like (7) holds. Bound (7) gives $$\mathrm{Var}\,N_S\le C^2V^{3/2}$$, which is weaker than the Poisson value $$V$$.

*Number variance (numerical).* For a causal diamond of volume $$V$$ with a uniformly random centre in a cell of $$\Lambda_7$$ and a random rest-frame velocity $$u$$, the lattice count has the law of the count of the orbit lattice in a fixed diamond. We drew $$u$$ with the hyperbolic density uniform on the ball of radius $$R_b$$ (an approximation to the invariant probability measure on $$\Gamma_7\backslash H^3$$ that improves as $$R_b$$ grows) and counted the points of $$\Lambda_7$$ in the diamond, with $$2\cdot10^6$$ samples per entry. The mean of $$N/V$$ was $$1.000\pm0.001$$ in each case, and $$\mathrm{Var}\,N/V$$ was:

| $$V$$ | $$R_b=3$$ | $$R_b=4$$ | $$R_b=5$$ | $$R_b=6$$ |
|---|---|---|---|---|
| 4 | 0.521 | 0.478 | 0.485 | 0.481 |
| 16 | 0.358 | 0.325 | 0.331 | 0.328 |
| 64 | 0.218 | 0.203 | 0.205 | 0.203 |

The ratio is below the Poisson value 1 and falls roughly like $$V^{-0.34}$$ between $$V=16$$ and $$64$$ (variance growing like $$V^{0.66}$$). The dependence on $$R_b$$ is the bias of the ball average, of about 8% between $$R_b=3$$ and $$4$$, 1% to 2% between $$R_b=4$$ and $$5$$, and below 1% between $$R_b=5$$ and $$6$$. The largest value of $$\lvert N-V\rvert/V^{3/4}$$ among the samples was $$3.9$$ to $$4.2$$ at $$V=4$$, $$2.9$$ to $$3.1$$ at $$V=16$$ and $$2.0$$ to $$2.3$$ at $$V=64$$. These values reach $$V=64$$ only. Saravani and Aslanbeigi conjecture that no process has $$\mathrm{Var}\,N_S\le\alpha V$$ with $$\alpha<1$$ for all large intervals [[Saravani–Aslanbeigi]](#ref-sa); the conjecture is asymptotic, the orbit lattice is an admissible test of it, and data up to $$V=64$$ do not decide it.

## 4. The mean dictionary on the orbit lattice

Take a realisation through $$0$$, $$\Lambda=g\Lambda_7$$. Its future vectors are $$v=g\,e(k)$$ with $$k\in\mathbb Z^4$$ on a shell $$n$$, of proper time $$\tau_n$$ with $$\tau_n^2=7^{-1/4}n$$ and interval volume $$V_n=\pi\tau_n^4/24=\pi n^2/(24\sqrt7)$$. The interior count $$m(v)$$ depends on $$k$$ alone, since $$g$$ maps the lattice and the interval together. Let $$\rho_n$$ be the density, per unit hyperbolic measure (the volume element of the unit hyperboloid $$H^3$$, in which a ball of radius $$\eta$$ has volume $$\pi(\sinh2\eta-2\eta)$$), of vectors on shell $$n$$; by invariance it does not depend on $$g$$. Let $$P_n(m)$$ be the fraction of those vectors with interior count $$m$$. Since $$d^4y=\tau^3d\tau\,d\mu(u)=\tfrac6\pi\,dV\,d\mu(u)$$, unit density of the point set requires $$\sum_{n\le n_{\max}}\rho_n\approx\int_0^{\tau_{n_{\max}}}\tau^3d\tau=n_{\max}^2/(4\sqrt7)$$.

*Shell data (numerical, except where Theorem 3(vii) applies).* We enumerated the lattice vectors on the shells $$Q_7=n$$ with $$k_0\le X$$ and computed the interior count of each by exact enumeration of the lattice points of the interval. For $$n=1,\dots,16$$ the densities, counted in hyperbolic balls up to $$X=1000$$ (they converge slowly, because the error in hyperbolic lattice-point counts decays only like a power of the radius), satisfy $$16\rho_n=4.005,\,12.016,\,6.001,\,11.990,\,12.020,\,18.025,\,25.000,\,12.017,\,27.993,\,36.121,\,48.139,\,18.011,\,36.020,\,75.158,\,32.166,\,12.197$$, within $$0.6\%$$ of the integers $$4,12,6,12,12,18,25,12,28,36,48,18,36,75,32,12$$ for $$n\le15$$ and within $$2\%$$ for $$n=16$$. Their sum is $$386/16=24.125$$ against $$16^2/(4\sqrt7)=24.19$$. The interior counts on shells 1 to 10 are as follows; $$16\rho_nP_n(m)$$ is again an integer in every case, and the last column is the measured fraction at the cutoff $$X$$ stated.

| $$n$$ | $$16\rho_n$$ | $$m$$ and $$16\rho_nP_n(m)$$ | $$P_n(m)$$ | measured |
|---|---|---|---|---|
| 1, 2, 3, 5 | 4, 12, 6, 12 | $$m=0$$: all | 1 | proved, Theorem 3(vii) |
| 4 | 12 | $$m=0$$: 8; $$m=1$$: 4 | 2/3, 1/3 | $$0.666,\ 0.334$$ ($$X=300$$) |
| 6 | 18 | $$m=0$$: 6; $$m=2$$: 12 | 1/3, 2/3 | $$0.330,\ 0.670$$ ($$X=150$$) |
| 7 | 25 | $$m=0$$: 1; $$m=2$$: 24 | 1/25, 24/25 | $$0.0398,\ 0.960$$ ($$X=300$$) |
| 8 | 12 | $$m=3$$: 12 | 1 | $$1$$ ($$X=150$$) |
| 9 | 28 | $$m=2$$: 4; $$m=4$$: 24 | 1/7, 6/7 | $$0.142,\ 0.858$$ ($$X=300$$) |
| 10 | 36 | $$m=4$$: 24; $$m=6$$: 12 | 2/3, 1/3 | $$0.668,\ 0.332$$ ($$X=150$$) |

Between $$9\cdot10^4$$ and $$2.2\cdot10^5$$ vectors per shell enter the fractions at $$X=150$$ and about $$8\cdot10^5$$ at $$X=300$$, and the fractions fluctuate by about $$1\%$$ between cutoffs. The rational values in the table are those for which every $$16\rho_nP_n(m)$$ is an integer and which lie within this fluctuation of the measured fractions. On shell 4 the counts are proved by Theorem 3(vii), and $$P_4(1)=\rho_1/\rho_4$$, since the vectors with $$m=1$$ are the doubles of the vectors on shell 1. For $$m=0$$ on shell 7 the counts at $$X=300$$ ($$0.0398$$) and the integrality of $$16\rho_7P_7(m)$$ both select $$1/25$$ rather than the neighbouring $$1/24$$. On shells 11 to 60 every vector examined with $$k_0\le24$$ has $$m\ge6$$, and by Theorem 3(iv) $$m$$ grows with $$n$$ uniformly in the realisation; since $$w_m=0$$ for $$m\ge5$$ only shells 1 to 10 enter the sums below.

*The linear system.* The mean of (1) at $$0$$ is $$\nu[a\phi(0)+\sum_{n,m}\rho_nP_n(m)w_m\int_{H^3}\phi(-\tau_nu)\,d\mu(u)]$$. It has the form of the Poisson mean with the measure $$\psi(V)\,dV$$ replaced by $$\tfrac\pi6\sum_n\rho_n\langle w\rangle_n\delta(V-V_n)$$, where $$\langle w\rangle_n=\sum_mP_n(m)w_m$$. Because the shell laws have compact support in $$m$$ (Theorem 3(iv)), the Mellin transform is a finite Dirichlet polynomial,

$$
\hat\psi_\Lambda(s)=\frac\pi6\sum_{n=1}^{10}\rho_n\sum_mP_n(m)\,w_m\,V_n^{\,s-1}, \tag{8}
$$

which is entire. On the Haar-random lattice in two dimensions, by contrast, the empty-interval density decays like $$1/V$$ and the zeroth moment diverges logarithmically (see the page on Lorentz-invariant point sets). The derivation of the infrared conditions uses only the Mellin transform of the radial density, and we assume it carries over: the conditions become

$$
\hat\psi_\Lambda(\tfrac12)=\hat\psi_\Lambda(1)=\hat\psi_\Lambda(\tfrac54)=\hat\psi_\Lambda(\tfrac32)=0,\qquad \hat\psi_\Lambda'(\tfrac32)=\frac{\sqrt\pi}3,\qquad a=-3\hat\psi_\Lambda'(1), \tag{9}
$$

with $$\nu=4/\sqrt6$$. The zero at $$\tfrac54$$ is the one of the five-layer set; with the other four conditions it fixes the five weights $$w_0,\dots,w_4$$ (the system has condition number $$1.8\cdot10^5$$). Its solution, with the table above, is

$$
(w_0,\dots,w_4)=(2.93994598436,\,-194.985218449,\,104.163596525,\,-409.986325849,\,28.9411577030),\qquad a=-0.798641063627.
$$

The mean ratio $$\mathbb E[B\phi(0)]/\Box\phi(0)$$ computed from these data (the same quadrature as in section 1) is, for $$L=5,10,20,40$$:

| packet | $$L=5$$ | $$L=10$$ | $$L=20$$ | $$L=40$$ |
|---|---|---|---|---|
| timelike | 0.971 | 1.256 | 1.127 | 1.047 |
| spacelike | 0.300 | 0.645 | 0.855 | 0.948 |

The ratio tends to 1, close to the values of the five-layer Poisson kernel in section 1: the mean dictionary holds on the orbit lattice. It is sensitive to the data. Changing $$P_6(0)$$ from $$1/3$$ to $$0.353$$ with the same weights changes the timelike ratio at $$L=10,20,40$$ from $$1.26,\,1.13,\,1.05$$ to $$-3.0\cdot10^3,\,-4.3\cdot10^4,\,-6.6\cdot10^5$$: a shift of $$0.02$$ in one probability breaks the cancellation of the constant mode, and the error is amplified by $$L^4$$. Rounding the weights does the same: at nine significant digits the ratio at $$L=40$$ is $$1.37$$, and twelve digits reproduce $$1.047$$. A working dictionary therefore needs the exact rational laws and weights to twelve digits.

## 5. A single realisation of the orbit lattice

Take a realisation $$\Lambda$$ through $$0$$ and apply a Lorentz transformation so that $$\Lambda=g\Lambda_7$$. For a future vector $$v\in\Lambda$$ let $$\kappa(v)=\nu\,w_{m(v)}$$, where $$m(v)$$ is the number of points of $$\Lambda$$ in the interval $$(0,v)$$, and let $$\kappa(v)=0$$ for other $$v$$. Then $$(B\phi)(x)=\nu a\phi(x)+\sum_v\kappa(v)\phi(x-v)$$, and the kernel $$\kappa$$ is bounded and supported on finitely many shells (Theorem 3(iv); numerically on shells 1 to 10 for the weights of section 4).

<div class="proposition" markdown="1">

**Proposition 4** (a layer operator on the orbit lattice has no symbol). The stabiliser $$\Gamma_g=g\Gamma_7g^{-1}$$ of $$\Lambda$$ in $$SO^+(1,3)$$ is Zariski dense, and the kernel $$\kappa$$ is bounded, $$\Gamma_g$$-invariant, supported on finitely many shells, and non-zero at the vectors of shell 1 if $$w_0\ne0$$. Therefore, by Proposition 1 of [the page on exact discrete Lorentz symmetry](/quantum-gravity/discrete-lorentz-group/), the Fourier series $$\hat\kappa$$ of the kernel coincides with an integrable function on no open set of the Brillouin torus; and for every finitely supported $$\phi\ne0$$ on $$\Lambda$$, the function $$B\phi-\nu a\phi$$ is not square summable. Moreover every bounded operator on $$\ell^2(\Lambda)$$ that commutes with the translations of $$\Lambda$$ and with $$\Gamma_g$$ is a multiple of the identity, so it cannot have the symbol of $$\Box$$ on any open set of momenta.

</div>

<div class="proof" markdown="1">

*Proof.* $$\Gamma_g$$ is conjugate to $$\Gamma_7$$, which is Zariski dense by Theorem 3(ii). It preserves $$\Lambda$$, the form and the time orientation, hence the interval counts: $$m(\gamma v)=m(v)$$. So $$\kappa$$ is $$\Gamma_g$$-invariant; it is bounded because only the finitely many values $$\nu w_m$$ occur, and it vanishes outside finitely many shells by Theorem 3(iv). The vectors of shell 1 have $$m=0$$ by Theorem 3(vii), and shell 1 is non-empty (it contains $$g\,e(1,2,1,1)$$), so $$\kappa\ne0$$ there if $$w_0\ne0$$. Proposition 1 of that page applies and gives the first statement. Suppose $$\phi\ne0$$ is finitely supported and $$\widehat{B\phi-\nu a\phi}=\hat\kappa\hat\phi$$ were in $$L^2$$, where $$\hat\phi$$ is a non-zero trigonometric polynomial. On the open set $$U=\{\hat\phi\ne0\}$$, which is non-empty, $$\hat\kappa=(\hat\kappa\hat\phi)/\hat\phi$$ would be locally square integrable, contradicting Proposition 1. The last statement is Theorem 1(a) of the same page, applied to $$\Lambda$$ and $$\Gamma_g$$. $$\square$$

</div>

So the response to a localised field is not a localised field: a retarded layer sum with weights tuned to give the correct mean maps any compactly supported test field to a function of infinite $$\ell^2$$ norm. A Gaussian packet is not finitely supported and the sums converge, so Proposition 4 does not give the size of the error for it; that depends on cancellations among the shell sums, which we evaluate.

*Single-realisation errors (numerical).* We took the unboosted realisation ($$g=1$$, $$\xi=0$$) with the weights, $$a$$ and $$\nu$$ of section 4, computed the interior count of every vector $$v$$ with $$1\le Q_7(v)\le10$$ in the packet's window by exact enumeration, and evaluated $$B\phi(0)$$ for the packet (3) at rest and boosted to speed $$\beta=0.6$$ along $$(1,2,3)/\sqrt{14}$$. The relative error $$\lvert B\phi(0)/\Box\phi(0)-1\rvert$$ is:

| packet | $$L=5$$ | $$L=10$$ | $$L=20$$ | $$L=40$$ |
|---|---|---|---|---|
| timelike, $$\beta=0$$ | $$5.9\cdot10^3$$ | $$9.9\cdot10^3$$ | $$2.6\cdot10^5$$ | $$4.2\cdot10^6$$ |
| spacelike, $$\beta=0$$ | $$5.9\cdot10^3$$ | $$3.1\cdot10^4$$ | $$1.0\cdot10^5$$ | $$3.0\cdot10^6$$ |
| timelike, $$\beta=0.6$$ | $$3.8\cdot10^2$$ | $$3.9\cdot10^4$$ | $$3.4\cdot10^5$$ | not computed |
| spacelike, $$\beta=0.6$$ | $$2.9\cdot10^3$$ | $$1.8\cdot10^4$$ | $$1.3\cdot10^4$$ | not computed |

The windows used ($$k_0\le100$$ and $$150$$) give the same values to the digits shown. One realisation, one lattice frame and four packets (two carriers, two velocities) give no estimate of the typical error or of its growth exponent: the values range over four orders of magnitude. The error is not small at any $$L$$ tried; at rest it is larger at $$L=40$$ than at $$L=5$$ by factors of $$7\cdot10^2$$ and $$5\cdot10^2$$. At $$L=5$$ the errors lie in the range of the Poisson standard deviations of section 2 ($$3\cdot10^2$$ to $$8\cdot10^3$$); at $$L=20$$ they lie between $$10^4$$ and $$4\cdot10^5$$, against the first-chaos bounds $$3\cdot10^3$$ and $$2\cdot10^4$$ of the two Poisson kernels.

## 6. Superposed lattices

Independent randomness can be added to the lattice by superposition. Let $$\Lambda_K$$ be the union of $$K$$ independent copies of the random orbit lattice, each of density $$1/K$$ (a copy is the density-one random lattice scaled by $$K^{1/4}$$), so that the density is 1.

<div class="proposition" markdown="1">

**Proposition 5** (number variance of a superposition). For every causal diamond $$D$$ of volume $$V$$, $$\mathrm{Var}\,N_K(D)/\mathbb E\,N_K(D)=\theta(V/K)$$, where $$\theta(v)=\mathrm{Var}\,N_1(D')/\mathbb E\,N_1(D')$$ is the ratio for a single density-one copy and a diamond $$D'$$ of volume $$v$$. Moreover $$\theta(v)\to1$$ as $$v\to0$$, so for fixed $$V$$ the ratio tends to 1 as $$K\to\infty$$.

</div>

<div class="proof" markdown="1">

*Proof.* The counts $$N_j(D)$$ of the independent copies are independent. Scaling by $$K^{-1/4}$$ maps a copy to a density-one copy and $$D$$ to a diamond of volume $$V/K$$, and the law of the random lattice is invariant under Lorentz transformations, so $$N_j(D)$$ has the law of $$N_1(D')$$ with $$\mathrm{vol}\,D'=V/K$$. Then $$\mathrm{Var}\,N_K=K\,\mathrm{Var}\,N_1(D')=K\,\theta(V/K)\,\mathbb E N_1(D')$$, and $$\mathbb E N_K=V=K\,\mathbb E N_1(D')$$. For the limit: a diamond of proper time $$T$$ has Euclidean diameter at most $$T$$ in its rest frame, where every lattice of the orbit has no non-zero vector shorter than $$\delta=7^{-1/8}$$ (proof of Theorem 3(iv)). For $$T<\delta$$ the diamond contains at most one lattice point, so $$N_1(D')\in\{0,1\}$$ and $$\theta=1-\mathbb E N_1(D')\to1$$. $$\square$$

</div>

As $$K\to\infty$$ at fixed $$V$$ the number statistics approach the Poisson ones. The failure of section 2 is already present for the Poisson sprinkling, so the number statistics do not determine the operator error, and the operator on $$\Lambda_K$$ for $$K\ge2$$ is not computed.

## 7. What is not excluded, and open points

1. Smeared kernels on a Poisson sprinkling, with a non-locality scale $$\ell_k\gg\ell$$ [[Aslanbeigi–Saravani–Sorkin]](#ref-ass). The evaluation in section 2 is for $$\ell_k=\ell$$.
2. Orbit lattices with a smearing that uses a frame, for instance a time direction. It breaks the invariance of the operator under $$\Gamma_g$$, and Proposition 4 does not apply to it; it brings back the preferred-frame questions of [the page on leakage of a preferred frame](/quantum-gravity/lorentz-violation-leakage/).
3. Superpositions of $$K\ge2$$ orbit lattices with a mesoscale.
4. Lattices that are not arithmetic orbits. The proofs use cocompactness (anisotropy) and the Zariski density of the stabiliser.
5. Observables other than the d'Alembertian on a Gaussian packet at one point, for example averages of $$B\phi$$ over many points of the realisation.

Open points. The first-chaos bound (6) was evaluated for the timelike carrier only; the first chaos carries about 1% of the variance of the Benincasa–Dowker operator, so (6) is far from sharp, and the standard deviation itself is known only from sampling. The growth exponent of the single-realisation error on the orbit lattice, and whether it differs from that of the Poisson sprinkling, is not determined by four packets in one frame. The asymptotic law of $$\mathrm{Var}\,N_S$$ on the orbit lattice, between the observed growth $$V^{0.66}$$ up to $$V=64$$ and the proved bound $$V^{3/2}$$, is open. The shell densities and the interior-count laws of section 4 are identified numerically from enumeration, and their exactness is an observation, not a theorem, beyond the cases of Theorem 3(vii).

## References

1. <a id="ref-bd"></a>Benincasa, Dowker, "The scalar curvature of a causal set," Phys. Rev. Lett. 104 (2010) 181301, [arXiv:1001.2725](https://arxiv.org/abs/1001.2725).
2. <a id="ref-ass"></a>Aslanbeigi, Saravani, Sorkin, "Generalized causal set d'Alembertians," JHEP 06 (2014) 024, [arXiv:1403.1622](https://arxiv.org/abs/1403.1622).
3. <a id="ref-sorkin"></a>Sorkin, "Does locality fail at intermediate length-scales," [arXiv:gr-qc/0703099](https://arxiv.org/abs/gr-qc/0703099).
4. <a id="ref-sa"></a>Saravani, Aslanbeigi, "On the causal set-continuum correspondence," [arXiv:1403.6429](https://arxiv.org/abs/1403.6429).
5. <a id="ref-lp"></a>Last, Penrose, "Poisson process Fock space representation, chaos expansion and covariance inequalities," [arXiv:0909.3205](https://arxiv.org/abs/0909.3205).
