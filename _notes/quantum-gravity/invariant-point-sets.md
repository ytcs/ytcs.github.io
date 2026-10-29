---
title: Lorentz-invariant point sets and the structures they cannot carry
date: 2026-10-29
category: substrate
order: 5
description: Haar-random lattices are Poincaré invariant in law and not Poisson, whereas invariant random sets of null lines or null hyperplanes with locally finite mean measure are empty, and a Poisson process with frames known only modulo an arithmetic lattice admits no equivariant finite non-empty neighbour rule.
status: "Proved from cited results: Siegel's mean value theorem, Moore's ergodicity theorem, Weil's criterion, the Borel density and Borel–Harish-Chandra theorems, the classification of odd unimodular lattices, Cartan–Dieudonné and generation of p-adic orthogonal groups by reflections. Proposition 2 is for two dimensions. Equation (5) is proved for real part of s above 2 and continued under the usual continuation of Eisenstein series."
excludes: "Under stated hypotheses, non-empty invariant random sets of null lines or hyperplanes with Radon mean measure; invariant sets of non-null lines with finite expected incidence with bounded sets; complexified point processes with finite intensity of real parts and non-real points; finite non-empty neighbour rules with lattice-modulo frames on Poisson points; a hand on Poisson causal orders."
---

Which random point sets in Minkowski space have a law invariant under all Poincaré transformations, and what further structure can such a set carry? Poisson point sets are invariant, and so are Haar-random lattices, which are not Poisson (Theorem 1). Further structure is mostly excluded, and the obstruction in each case is the non-compactness of the Lorentz group. An invariant random set of null lines or null hyperplanes whose mean measure is locally finite is empty (Theorem 4), and so is an invariant random set of timelike or spacelike lines that meets bounded regions in a finite expected number of lines (Proposition 5). If each point of a Poisson process carries a frame, a Lorentz transformation known only modulo an arithmetic lattice, no rule that commutes with Poincaré transformations selects a finite non-empty set of neighbours of a point (Theorem 6).

Bombelli, Henson and Sorkin proved that no equivariant measurable rule assigns a spacetime direction to a Poisson sprinkling [[Bombelli–Henson–Sorkin]](#ref-bhs); Theorem 6 carries their argument to frames that are defined only modulo a lattice. Dowker, Henson and Sorkin showed that the discreteness of a sprinkling is compatible with local Lorentz invariance [[Dowker–Henson–Sorkin]](#ref-dhs), and Saravani and Aslanbeigi found that lattices in 1+1 dimensions give a closer correspondence between number and volume than Poisson sprinklings at large volumes, and argued that this advantage should not persist in higher dimensions [[Saravani–Aslanbeigi]](#ref-sa).

Section 2 shows that the counts of a Haar-random lattice have the Poisson mean and variance (Theorem 1) and computes, in two dimensions, the law of empty causal intervals (Proposition 2). Three further results concern less central structures. An invariant point process with complexified positions and a locally finite intensity measure either has only real points or the projection of its points to real parts has infinite intensity (Proposition 3). Hecke correspondences between integral Lorentzian lattices commute with the Lorentz group and have finite valence, but a step replaces the lattice in all of spacetime at once (Propositions 8 to 10). A Poisson process has the same law as its mirror image, whereas a Poincaré-invariant random lattice whose law differs from that of its mirror image exists (Propositions 13 and 14). The page [lattice translations with a discrete Lorentz group](/quantum-gravity/discrete-lorentz-group/) proves the lemma on finite orbits that is used for lattice stabilisers in section 5. The page [kernel-scale kinematics of a single realisation](/quantum-gravity/per-realisation-kinematics/) treats the cocompact orbit lattices, which are invariant in law and whose realisations each have a translation lattice that varies from realisation to realisation.

## 1. Point processes and invariance

A *point process* in $$\mathbb R^d$$ is a random locally finite subset $$\Phi$$, equivalently a random counting measure; $$N(D)=\#(\Phi\cap D)$$ is its count in a set $$D$$. Its *intensity measure* is $$I(D)=\mathbb E\,N(D)$$, and Campbell's formula says $$\mathbb E\sum_{x\in\Phi}f(x)=\int f\,dI$$. A *Poisson process* with a locally finite mean measure $$I$$ has independent Poisson counts in disjoint sets; if the mean of a set is infinite, its count is infinite almost surely. A *mixed Poisson* or *Cox* process is a Poisson process whose mean measure is itself random. A process is *invariant in law* under a group if the transformed process has the same law, and *ergodic* if every invariant event has probability 0 or 1. The Lorentz group $$G=SO^+(1,d-1)$$ acts on $$\mathbb R^d$$ with the form $$B(x,y)=x_0y_0-\vec x\cdot\vec y$$ and $$Q(x)=B(x,x)$$; the Poincaré group $$P=G\ltimes\mathbb R^d$$ adds translations. The reflection of one spatial axis is denoted $$\Pi$$ and the reversal of time $$T$$.

The *Palm law* of a process at a point is its law conditioned on having a point there. For a Poisson process the Palm law at $$0$$ is the law of $$\Phi\cup\{0\}$$ (Slivnyak–Mecke), and for a random lattice with a uniformly random shift it is the law of the lattice through $$0$$. Mecke's formula states that for a Poisson process with mean measure $$I$$, $$\mathbb E\sum_{x_1,\dots,x_k\in\Phi\ \mathrm{distinct}}f(x_1,\dots,x_k)=\int f\,dI^{\otimes k}$$ [[Last–Penrose]](#ref-lp).

A fixed lattice is not invariant under boosts, and randomising its position by translations does not change that. Section 2 shows that randomising the lattice itself, with the invariant measure on the space of lattices, does produce an invariant process.

## 2. Haar-random lattices

Let $$d\ge2$$, let $$X_d=ASL(d,\mathbb R)/ASL(d,\mathbb Z)$$ be the space of affine lattices of covolume 1, and let $$\sigma$$ be its Haar probability measure, which is finite by Siegel's work. A point $$(g,\xi)$$ of $$X_d$$ is the lattice $$g\mathbb Z^d+\xi$$. Two results are used.

<div class="theorem" markdown="1">

**Theorem S** (Siegel's mean value theorem [[Siegel]](#ref-siegel)). For $$d\ge2$$ and $$f\in L^1(\mathbb R^d)$$, $$\int\sum_{v\in g\mathbb Z^d\setminus\{0\}}f(v)\,d\sigma_0(g)=\int f$$, where $$\sigma_0$$ is the Haar probability measure on $$SL(d,\mathbb R)/SL(d,\mathbb Z)$$.

</div>

<div class="theorem" markdown="1">

**Theorem M** (Moore's ergodicity theorem [[Moore]](#ref-moore)). A closed non-compact subgroup of a connected simple Lie group with finite centre acts ergodically, by left multiplication, on the quotient by a lattice with respect to Haar measure.

</div>

<div class="theorem" markdown="1">

**Theorem 1** (Haar-random lattices). Let $$\Lambda=g\mathbb Z^d+\xi$$ with $$(g,\xi)$$ distributed by $$\sigma$$ on $$X_d$$, $$d\ge2$$. Then:

(i) the law of $$\Lambda$$ is invariant under $$ASL(d,\mathbb R)$$, hence under the Poincaré group $$P$$;

(ii) $$\Lambda$$ is ergodic under $$P$$;

(iii) $$\mathbb E\,N(D)=\mathrm{Var}\,N(D)=\mathrm{vol}(D)$$ for every bounded Borel set $$D$$, where $$N(D)=\#(\Lambda\cap D)$$;

(iv) $$\Lambda$$ is neither a Poisson process nor a mixed Poisson process with an almost surely diffuse random mean measure;

(v) the law of $$\Lambda$$ is also invariant under $$\Pi$$ and $$T$$, hence under the full group $$O(1,d-1)\ltimes\mathbb R^d$$.

</div>

Invariance is inherited from Haar measure, and the counting statistics follow from Siegel's formula applied to the pair correlation. The failure of the Poisson property comes from the symmetry of a lattice under the point reflection $$x\mapsto2y-x$$ about any of its points.

<div class="proof" markdown="1">

*Proof.* (i) Haar measure on $$X_d$$ is invariant under left multiplication by $$ASL(d,\mathbb R)$$, which contains $$P$$.

(ii) A $$P$$-invariant event is invariant under the translation subgroup, which acts transitively on each torus fibre $$\mathbb R^d/g\mathbb Z^d$$, so by Fubini its indicator is almost everywhere a function of $$g$$ alone. The subgroup $$SO^+(1,d-1)$$ is closed and non-compact in the simple group $$SL(d,\mathbb R)$$, so by Moore's theorem it acts ergodically on $$SL(d,\mathbb R)/SL(d,\mathbb Z)$$, and the function is constant.

(iii) For fixed $$g$$, uniform averaging over $$\xi$$ gives $$\mathbb E_\xi N(D)=\mathrm{vol}(D)$$. For the second factorial moment,

$$
\mathbb E\sum_{x\ne y\in\Lambda}\mathbf 1_D(x)\mathbf 1_D(y)=\int dx\,\mathbf 1_D(x)\ \mathbb E_g\!\!\sum_{v\in g\mathbb Z^d\setminus\{0\}}\mathbf 1_D(x+v)=\mathrm{vol}(D)^2, \tag{1}
$$

where the average over $$\xi$$ of the sum over $$x\in\Lambda$$ turned it into $$\int dx$$ (covolume 1) and the last step is Siegel's theorem applied to $$f=\mathbf 1_{D-x}$$. Then $$\mathrm{Var}\,N=\mathbb E N(N-1)+\mathbb E N-(\mathbb E N)^2=\mathrm{vol}(D)$$.

(iv) Almost surely, $$x,y\in\Lambda$$ implies $$2y-x\in\Lambda$$, and $$\Lambda$$ has infinitely many pairs $$x\ne y$$. For a Poisson process with diffuse mean measure $$I$$, Mecke's formula gives $$\mathbb E\sum_{x,y,z\ \mathrm{distinct}}\mathbf 1[z=2y-x]=\iint I(\{2y-x\})\,dI(x)\,dI(y)=0$$, and the same holds conditionally for a mixed Poisson process whose random mean measure is almost surely diffuse. So with probability 1 the lattice contains triples $$(x,y,2y-x)$$ with $$x\ne y$$, an event of probability zero for those processes.

(v) The matrices $$\Pi$$ and $$T$$ are diagonal with entries $$\pm1$$, so they lie in $$GL(d,\mathbb Z)$$ and conjugation by either is an automorphism of $$ASL(d,\mathbb R)$$ that maps $$ASL(d,\mathbb Z)$$ to itself. Since $$\Pi\mathbb Z^d=\mathbb Z^d$$, $$\Pi(g\mathbb Z^d+\xi)=(\Pi g\Pi^{-1})\mathbb Z^d+\Pi\xi$$, so $$\Pi\Lambda$$ is the image of $$\Lambda$$ under the map $$\varphi$$ of $$X_d$$ induced by this conjugation. The map $$\varphi$$ intertwines left multiplication by $$k$$ with left multiplication by $$c(k)=\Pi k\Pi^{-1}$$, and $$c$$ maps $$ASL(d,\mathbb R)$$ onto itself, so $$\varphi_*\sigma$$ is an $$ASL(d,\mathbb R)$$-invariant probability measure on $$X_d$$ and by uniqueness equals $$\sigma$$. The same holds for $$T$$, and $$O(1,d-1)$$ is generated by $$SO^+(1,d-1)$$, $$\Pi$$ and $$T$$. $$\square$$

</div>

The counts of a Haar-random lattice have the Poisson mean and variance in every bounded region, and the two processes still differ. The difference bears on the causal-set d'Alembertian. In two dimensions this operator, evaluated at a point $$x$$ of the point set, sums $$w_m\varphi(y)$$ over the earlier points $$y$$, where $$m$$ is the number of points of the set in the causal interval between $$y$$ and $$x$$ (the sets with fixed $$m$$ are the layers). The weights $$(w_0,w_1,w_2)=(1,-2,1)$$ are the two-dimensional Benincasa–Dowker weights, in the convention of the page [mixed-Poisson and signed substrates](/quantum-gravity/signed-substrates/): signature $$(-,+)$$, $$\Box=-\partial_t^2+\partial_x^2$$; the overall sign and the prefactor of the sum play no role below. On a Poisson sprinkling of unit density the expected weight of the earlier points at interval volume $$V$$ is $$\sum_mw_me^{-V}V^m/m!$$, and the mean of the operator follows from the Mellin transform of this function. For a Haar-random lattice the corresponding function is $$\sum_mw_mf_m(V)$$ with the densities $$f_m$$ defined next, and the question is how $$f_0$$, the density of earlier points with an empty interval, behaves at large $$V$$.

We use null coordinates $$(u,v)$$ in which the area element is $$du\,dv$$ and boosts act by $$(u,v)\mapsto(e^su,e^{-s}v)$$. Let $$L$$ be the Haar-random lattice through $$0$$, that is $$L=g\mathbb Z^2$$ with $$g$$ distributed by $$\sigma_0$$; this is the Palm law of the affine Haar lattice at one of its points. Since $$L=-L$$, future and past points have the same densities. For a future lattice point $$y=(u,v)$$, $$u,v>0$$, the causal interval between $$0$$ and $$y$$ is the open rectangle $$(0,u)\times(0,v)$$ of area $$V=uv$$. Write $$\zeta=\tfrac12\ln(u/v)$$, so that $$du\,dv=dV\,d\zeta$$, and let $$m(y)$$ be the number of lattice points in the rectangle. Boost invariance of the law of $$L$$ implies that the expected number of future lattice points $$y$$ with $$m(y)=m$$ and $$(V,\zeta)$$ in a small set is $$\int f_m(V)\,dV\,d\zeta$$ over that set, with a density $$f_m$$ that does not depend on $$\zeta$$. By Siegel's theorem $$\sum_mf_m(V)=1$$ for every $$V$$, so $$0\le f_m\le1$$. For a Poisson process of unit density, $$f_m(V)=e^{-V}V^m/m!$$.

<div class="proposition" markdown="1">

**Proposition 2** (empty causal intervals in a Haar-random lattice, $$d=2$$). For the Haar-random lattice through $$0$$ in $$\mathbb R^{1,1}$$,

$$
f_0(V)=\frac6{\pi^2}\min\Big(1,\frac1V\Big),\tag{2}
$$

$$f_1=f_2=0$$ for $$V>6$$, and $$f_3=0$$ for $$V>8$$. Consequently, for any weights $$w_0,w_1,w_2,w_3$$ with $$w_0\ne0$$,

$$
\sum_{m\le3}w_m\int_0^Xf_m(V)\,dV=\frac{6}{\pi^2}\,w_0\ln X+O(1)\qquad(X\to\infty), \tag{3}
$$

while for the Poisson process the same quantity converges to $$\sum_mw_m$$.

</div>

The threshold 8 for $$f_3$$ cannot be replaced by 6. Take $$V=7$$, $$y=(2,3.5)=2y_0$$ with $$y_0=(1,1.75)$$, and the lattice generated by $$y_0$$ and $$w=(0.43,1.7525)$$, which has $$\det(y_0,w)=1$$. The open rectangle $$(0,2)\times(0,3.5)$$ contains exactly the three points $$w$$, $$y_0$$ and $$2y_0-w=(1.57,1.7475)$$, so $$m(y)=3$$ at $$V=7$$. The same count holds for the lattices generated by $$y_0$$ and $$w+\theta y_0$$ with $$\theta$$ in an interval of positive length containing $$0$$, so $$f_3(7)>0$$.

The proof unfolds Haar measure over the primitive lattice vectors, which leaves the lattice determined by a single angle, and then reads the emptiness of the rectangle off the fractional parts $$\{n\alpha\}$$.

<div class="proof" markdown="1">

*Proof.* Unfolding Haar measure on $$SL(2,\mathbb R)/SL(2,\mathbb Z)$$ over pairs (lattice, primitive vector) gives

$$
\mathbb E\sum_{y\ \mathrm{primitive}}F(y,L)=c_\ast\int dy\int_0^1d\theta\,F(y,L_{y,\theta}),\qquad L_{y,\theta}=\mathbb Zy+\mathbb Z(w_y+\theta y),
$$

where $$w_y$$ is any vector with $$\det(y,w_y)=1$$ (the stabiliser of a primitive vector in $$SL(2,\mathbb Z)$$ is the group of unipotent translations, whose quotient in the real group is the circle of $$\theta$$). The constant $$c_\ast$$ follows from Siegel's theorem. Take $$F=f(y)$$; since every non-zero lattice vector is $$qy_0$$ with $$q\ge1$$ and $$y_0$$ primitive, and $$\int f(qy_0)\,dy_0=q^{-2}\int f$$, we get $$\int f=c_\ast\sum_{q\ge1}q^{-2}\int f=c_\ast\frac{\pi^2}6\int f$$, so $$c_\ast=6/\pi^2$$. A vector $$y=qy_0$$ is obtained from the primitive vector $$y_0=y/q$$, with $$dy_0=q^{-2}dy$$, and therefore

$$
f_m(V)=\frac6{\pi^2}\sum_{q\ge1}\frac1{q^2}\,\mathbb P_q(m;V),
$$

where $$\mathbb P_q(m;V)$$ is the probability, over $$\theta$$, that the rectangle of $$y=qy_0$$ with area $$V$$ contains $$m$$ points of $$L_{y_0,\theta}$$.

Rescale the rectangle of $$y=(u,v)$$, $$V=uv$$, to the unit square by dividing the first coordinate by $$u$$ and the second by $$v$$, which multiplies areas by $$1/V$$. The lattice becomes generated by $$y_0'=(1/q,1/q)$$ and a vector $$w'=(\alpha,\alpha+c)$$ with $$\det(y_0',w')=c/q=1/V$$, so $$c=q/V$$. Shifting $$\theta$$ by 1 shifts $$w'$$ by $$y_0'=(1/q,1/q)$$, so $$\alpha$$ is uniform modulo $$1/q$$. The lattice points are $$(n\alpha+k/q,\ n\alpha+k/q+nq/V)$$, $$n,k\in\mathbb Z$$, and the point lies in the open unit square iff $$s=n\alpha+k/q$$ satisfies $$0<s<1-nq/V$$ for $$n\ge1$$. The point reflection $$p\mapsto y-p$$ maps the lattice into itself and line $$n$$ onto line $$-n$$, so the lines $$\pm n$$ contain the same number of points. Line $$0$$ contains the $$q-1$$ points $$ky_0'$$, $$0<k<q$$, on the diagonal.

For the void law, note that a vector with $$q\ge2$$ has $$q-1\ge1$$ points on its diagonal, so only primitive vectors contribute to $$f_0$$ and $$f_0(V)=\frac6{\pi^2}\mathbb P_1(0;V)$$. For $$q=1$$, line $$n\ge1$$ contains a point iff $$n<V$$ and $$\{n\alpha\}\in(0,1-n/V)$$, with $$\alpha$$ uniform modulo 1. Put $$\delta=1-\{\alpha\}\in(0,1)$$. If $$V\le1$$ there is no $$n$$ with $$1\le n<V$$ and the rectangle is empty. If $$V>1$$, the case $$n=1$$ shows that emptiness forces $$\delta\le1/V$$; conversely, if $$\delta\le1/V$$ then for $$1\le n<V$$, $$n\delta\le n/V<1$$, so $$\{n\alpha\}=1-n\delta\ge1-n/V$$ and every line is empty. Hence $$\mathbb P_1(0;V)=\min(1,1/V)$$, which is (2).

To show that $$f_1,f_2,f_3$$ vanish, let first $$q=1$$ and $$V>6$$, and suppose $$\delta>1/V$$, so that line 1 is occupied. If $$\delta<\tfrac12$$, then $$\{2\alpha\}=1-2\delta<1-2/V$$, so line 2 is occupied too. If $$\delta>\tfrac12$$, then $$\{\alpha\}=1-\delta<\tfrac12$$; line 2 is empty only if $$\{\alpha\}\ge\tfrac12-1/V$$, and then $$\{3\alpha\}=3\{\alpha\}-1\in[\tfrac12-3/V,\tfrac12)$$, which lies in $$(0,1-3/V)$$ because $$V>6$$, so line 3 is occupied. In each case two lines and their reflections give $$m\ge4$$. For $$q=1$$ and $$V>6$$ we therefore have $$m=0$$ or $$m\ge4$$.

Let $$q\ge2$$. The set of values $$n\alpha+k/q$$ is a coset of $$\tfrac1q\mathbb Z$$, so an open interval of length larger than $$1/q$$ contains one of them. The interval $$(0,1-nq/V)$$ is long enough iff $$n<V(q-1)/q^2$$, and then line $$\pm n$$ is occupied. With the diagonal this gives $$m\ge(q-1)+2\#\{n\ge1:n<V(q-1)/q^2\}$$. For $$q=2$$ and $$V>4$$ the line $$n=1$$ qualifies, so $$m\ge1+2=3$$; for $$V>8$$ the line $$n=2$$ qualifies as well, so $$m\ge1+4=5$$. For $$q=3$$ and $$V>9/2$$, $$m\ge2+2=4$$. For $$q=4$$ and $$V>16/3$$, $$m\ge3+2=5$$. For $$q\ge5$$ the diagonal alone gives $$m\ge4$$. Collecting the cases: for $$V>6$$ every vector has $$m=0$$ or $$m\ge3$$, so $$f_1=f_2=0$$; for $$V>8$$ every vector has $$m=0$$ or $$m\ge4$$, so $$f_3=0$$.

For (3), take $$X\ge1$$; then $$\int_0^Xf_0\,dV=\frac6{\pi^2}(1+\ln X)$$. For $$m=1,2,3$$ the density $$f_m$$ is bounded by 1 and vanishes for $$V>8$$, so $$\int_0^Xf_m\,dV\le8$$. $$\square$$

</div>

For the Poisson process, $$\int_0^\infty\sum_mw_mf_m\,dV=\sum_mw_m\int e^{-V}V^m/m!\,dV=\sum_mw_m$$, which vanishes for the weights $$(1,-2,1)$$. This is the value at $$s=1$$ of the Mellin transform $$\int\sum_mw_mf_m(V)V^{s-1}dV$$, whose zeros at $$s=1$$ and $$s=2$$ are the conditions on the weights that [mixed-Poisson and signed substrates](/quantum-gravity/signed-substrates/) lists for the Poisson mean of the operator. On a Haar-random lattice, by (2), the Mellin transform of $$f_0$$ is $$\frac{6}{\pi^2}\frac1{s(1-s)}$$ for $$0<\mathrm{Re}\,s<1$$, with a pole at $$s=1$$ where the Poisson transform $$\Gamma(s)$$ is regular. By (3), a layer-weighted kernel supported on $$m\le3$$ with $$w_0\ne0$$, in particular the weights $$(1,-2,1)$$, has a zeroth moment that diverges logarithmically on a Haar-random lattice, so the reading of the operator from the Poisson Mellin transform does not apply to it. On the cocompact orbit lattices of [kernel-scale kinematics of a single realisation](/quantum-gravity/per-realisation-kinematics/) every realisation has interior counts $$m\ge c_0V-C$$ (its Theorem 3(iv)), so each $$f_m$$ has compact support and the corresponding moments are finite.

## 3. Marks and extended objects

One may ask for more than points. Null lines and null hyperplanes would build the light cone into the substrate, timelike lines would be world lines of massive particles, and a complex position $$x+iy$$ would supply the imaginary part used in analytic continuation. A random set of extended objects (lines, hyperplanes) or of points with additional data (a complex position, a frame) is Poincaré invariant only if its intensity measure is an invariant measure on the space of objects. We ask which spaces of objects carry such a measure, with the further requirement that the set of objects meeting a compact region has finite measure. Boosts decide the answer in each case; Lemma 1 below is the form of the argument for measures on $$\mathbb R^d$$, and Weil's criterion is its form for measures on spaces of lines.

### 3.1 Wandering shells

A Lorentz transformation is *hyperbolic* if it has an eigenvalue $$\lambda>1$$; the eigenvalue $$\lambda^{-1}$$ then occurs as well, with null eigenvectors $$w_+$$ and $$w_-$$ ($$\gamma w_\pm=\lambda^{\pm1}w_\pm$$).

<div class="lemma" markdown="1">

**Lemma 1** (wandering shells). Let $$\Gamma\subset SO^+(1,d-1)$$ contain a hyperbolic element, and suppose that either $$d=2$$ or the Zariski closure of $$\Gamma$$ acts irreducibly on $$\mathbb R^d$$ (for example $$\Gamma=SO^+(1,d-1)$$, or a Zariski-dense subgroup). Let $$\nu$$ be a $$\Gamma$$-invariant Borel measure on $$\mathbb R^d$$ and $$A$$ a $$\Gamma$$-invariant Borel set with $$\nu(A\setminus\{0\})<\infty$$. Then $$\nu(A\setminus\{0\})=0$$.

</div>

A hyperbolic element permutes an infinite family of disjoint shells, which then have the same mass; if the total mass is finite, that mass is zero. For one linear map of determinant $$\pm1$$ acting on a finite measure the same argument is Lemma 1 of the page [quantum sprinklings](/quantum-gravity/quantum-sprinklings/), where it is used on momentum space.

<div class="proof" markdown="1">

*Proof.* Put $$\nu_A=\nu\vert _{A\setminus\{0\}}$$, which is finite and $$\Gamma$$-invariant. Let $$\gamma\in\Gamma$$ be hyperbolic, with eigenvectors $$w_\pm$$ as above. Every $$y$$ decomposes as $$y=\varphi_+(y)w_++\varphi_-(y)w_-+y_\perp$$ with $$y_\perp\in\{w_+,w_-\}^\perp$$, where $$\varphi_+(y)=B(y,w_-)/B(w_+,w_-)$$, and $$\varphi_+(\gamma y)=\lambda\varphi_+(y)$$. The shells $$W_k=\{y:\lambda^k\le\lvert \varphi_+(y)\rvert <\lambda^{k+1}\}$$, $$k\in\mathbb Z$$, are disjoint, $$\gamma W_k=W_{k+1}$$, and so $$\nu_A(W_k)$$ is independent of $$k$$. Since their union has finite mass, each has mass 0, and $$\nu_A(\varphi_+\ne0)=0$$. The same argument with $$\gamma^{-1}$$ and $$\varphi_-$$ gives $$\nu_A(\varphi_-\ne0)=0$$. So $$\nu_A$$ is carried by $$Z_\gamma=\{w_+,w_-\}^\perp$$, a subspace of dimension $$d-2$$; if $$d=2$$ this is $$\{0\}$$ and we are done. In general, $$g\gamma g^{-1}$$ is hyperbolic with eigenvectors $$gw_\pm$$ for every $$g\in\Gamma$$, so $$\nu_A$$ is carried by $$gZ_\gamma$$ for every $$g$$, hence by $$\bigcap_{g\in\Gamma}gZ_\gamma$$. This is a $$\Gamma$$-invariant, hence Zariski-closure-invariant, proper subspace of $$\mathbb R^d$$, so it is $$\{0\}$$ by irreducibility. Thus $$\nu(A\setminus\{0\})=0$$. $$\square$$

</div>

### 3.2 Complexified and hyperboloid-marked points

<div class="proposition" markdown="1">

**Proposition 3** (complexified positions). Let $$\Phi$$ be a point process on $$\mathbb R^d\times\mathbb R^d$$, with points $$z=(x,y)$$ (read as $$x+iy$$), whose law is invariant under $$(x,y)\mapsto(gx+a,gy)$$, $$g\in SO^+(1,d-1)$$, $$a\in\mathbb R^d$$, and whose intensity measure $$I$$ is Radon, $$d\ge2$$. Then $$I=\mathrm{Leb}_x\otimes\nu$$ with $$\nu$$ a Radon Lorentz-invariant measure on $$\mathbb R^d$$, and the projected process $$\{x\}$$ has intensity $$\nu(\mathbb R^d)\,\mathrm{Leb}$$. This is finite only if $$\nu=c\,\delta_0$$, that is, only if all points are real. If $$\Phi$$ lives in the forward tube $$y\in V_+$$ (the open forward cone) it is almost surely empty, or its projection has infinite intensity. If $$\Phi$$ is a Poisson process, an infinite projected intensity means that every non-empty box contains infinitely many projected points almost surely.

</div>

The imaginary part $$y$$ plays the role of a mark taking values in $$\mathbb R^d$$; a mark in a hyperboloid $$\{y^2=r^2\}$$ is the special case in which $$\nu$$ is carried by the hyperboloid.

<div class="proof" markdown="1">

*Proof.* Translation invariance in $$x$$ makes $$I(\cdot\times C)$$ a translation-invariant Radon measure for each bounded $$C$$, hence $$c(C)\,\mathrm{Leb}$$; then $$\nu(C)=c(C)$$ is additive in $$C$$ and Radon, and $$I(gB\times gC)=I(B\times C)$$ gives $$\nu(gC)=\nu(C)$$. The projected intensity of a box $$B$$ is $$I(B\times\mathbb R^d)=\mathrm{vol}(B)\nu(\mathbb R^d)$$. By Lemma 1 with $$A=\mathbb R^d$$ and $$\Gamma=SO^+(1,d-1)$$, $$\nu(\mathbb R^d\setminus\{0\})$$ is either 0 or infinite, so finiteness of $$\nu(\mathbb R^d)$$ forces $$\nu=c\delta_0$$. In the forward tube $$0\notin V_+$$ and the same lemma with $$A=V_+$$ gives $$\nu=0$$ or $$\nu(V_+)=\infty$$. A Poisson count with infinite mean is infinite almost surely. $$\square$$

</div>

A Lorentz-invariant sprinkling cannot be given a small, invariantly defined imaginary part. Every invariant open neighbourhood $$U$$ of $$0$$ in $$y$$-space contains the whole null cone, because a boost carries any null vector arbitrarily close to $$0$$ and $$U$$ is invariant, and it contains every hyperboloid $$\{y^2=r^2\}$$ with $$\lvert r\rvert$$ small enough that it meets $$U$$. Such a set has infinite volume.

Smearing the imaginary part does not help either. For an invariant function $$F\ge0$$ the sets $$\{F\ge c\}$$ are invariant, so Lemma 1 gives $$\nu(\{F\ge c\}\setminus\{0\})\in\{0,\infty\}$$ for every $$c>0$$, and so $$\int_{\mathbb R^d\setminus\{0\}}F\,d\nu\in\{0,\infty\}$$.

### 3.3 Null lines and null hyperplanes

The space of null lines is the quotient $$P/H$$ of the Poincaré group by the stabiliser of one null line. Weil's criterion decides whether such a quotient carries an invariant measure.

<div class="theorem" markdown="1">

**Theorem W** (Weil's criterion). Let $$\mathcal G$$ be a locally compact group, $$H$$ a closed subgroup, and $$\Delta_{\mathcal G},\Delta_H$$ the modular functions. Then $$\mathcal G/H$$ carries a non-zero $$\mathcal G$$-invariant Radon measure if and only if $$\Delta_{\mathcal G}\vert _H=\Delta_H$$; such a measure is unique up to a constant factor (see, for instance, chapter 2 of Folland [[Folland]](#ref-folland)).

</div>

The Poincaré group $$P$$ is unimodular. For a Lie subgroup $$H$$, $$\Delta_H(h)=\lvert \det\mathrm{Ad}_{\mathfrak h}(h)\rvert ^{\pm1}$$, so $$P/H$$ carries an invariant measure if and only if $$\lvert \det\mathrm{Ad}_{\mathfrak h}(h)\rvert =1$$ for all $$h\in H$$.

<div class="theorem" markdown="1">

**Theorem 4** (no invariant measure on null lines or null hyperplanes). Let $$d\ge2$$. The space of null lines in $$\mathbb R^{1,d-1}$$ and the space of null hyperplanes each carry no non-zero Poincaré-invariant Radon measure. Consequently a Poincaré-invariant point process of null lines, or of null hyperplanes, whose intensity measure is Radon is almost surely empty; in particular there is no invariant Poisson process of null lines or null hyperplanes.

</div>

The obstruction is the boost along the null direction. It stretches the null direction and every null rotation fixing it by the same factor, so the stabiliser is not unimodular. For null lines in $$d=4$$ the obstruction can be seen in coordinates. Describe a line by its spatial direction $$n\in S^2$$ and its intercept $$y\in\mathbb R^3$$ at $$x_0=0$$, so that its points are $$(s,y+sn)$$. Suppose an invariant measure has a density $$\rho(n,y)\,d\Omega\,d^3y$$. Translations shift $$y$$ by arbitrary vectors, so $$\rho$$ does not depend on $$y$$. A boost of rapidity $$\eta$$ along a direction $$n_0$$ fixes the lines with $$n=n_0$$ and acts near them as follows: it contracts the angular coordinates by $$e^{-\eta}$$ each (the aberration $$\theta'\approx e^{-\eta}\theta$$ for $$\theta$$ small) and the component of $$y$$ along $$n_0$$ by $$e^{-\eta}$$, while the components of $$y$$ transverse to $$n_0$$ are unchanged. The measure element is multiplied by $$e^{-3\eta}$$, invariance would give $$\rho(n_0)=e^{3\eta}\rho(n_0)$$, and so $$\rho=0$$. The proof below removes the assumption of a density.

<div class="proof" markdown="1">

*Proof.* Take the null vector $$k=e_0+e_{d-1}$$, $$k'=e_0-e_{d-1}$$, and $$e_i$$ ($$1\le i\le d-2$$) transverse. The Lie algebra of the Poincaré group consists of pairs $$(X,a)$$, $$X\in\mathfrak{so}(1,d-1)$$, $$a\in\mathbb R^d$$, with $$\mathrm{Ad}(g)(X,a)=(gXg^{-1},ga)$$. Write $$x\wedge y$$ for the element $$z\mapsto B(y,z)x-B(x,z)y$$ of $$\mathfrak{so}(1,d-1)$$.

The null line $$\mathbb Rk$$ has stabiliser $$H$$ with $$\mathfrak h=\{(X,a):Xk\in\mathbb Rk,\ a\in\mathbb Rk\}$$. A basis is: $$J=k\wedge k'$$ (the boost along $$k$$), the rotations $$e_i\wedge e_j$$, the null rotations $$N_i=k\wedge e_i$$ (they satisfy $$N_ik=B(e_i,k)k-B(k,k)e_i=0$$), and the translation $$k$$. (The elements $$k'\wedge e_i$$ do not stabilise $$k$$.) Let $$b_\eta$$ be the boost in the $$(0,d-1)$$ plane, $$b_\eta k=e^\eta k$$, $$b_\eta k'=e^{-\eta}k'$$, $$b_\eta e_i=e_i$$. Then $$\mathrm{Ad}(b_\eta)J=J$$, the rotations are fixed, $$\mathrm{Ad}(b_\eta)N_i=(b_\eta k)\wedge(b_\eta e_i)=e^\eta N_i$$ for each of the $$d-2$$ transverse directions, and the translation $$k$$ is multiplied by $$e^\eta$$. So $$\det\mathrm{Ad}_{\mathfrak h}(b_\eta)=e^{(d-1)\eta}\ne1$$, and by Weil's criterion there is no invariant measure on $$P/H$$.

The null hyperplane $$\{x:B(k,x)=0\}$$ has the same Lorentz part, and its translation part is $$\{a:B(k,a)=0\}=\mathrm{span}\{k,e_1,\dots,e_{d-2}\}$$, on which $$b_\eta$$ acts by $$e^\eta$$ on $$k$$ and trivially on the $$e_i$$. Again $$\det=e^{(d-1)\eta}$$. In $$d=2$$ the two spaces coincide.

If $$\Phi$$ is an invariant process with Radon intensity measure, the intensity is an invariant Radon measure, hence zero, so $$\Phi$$ is empty almost surely. $$\square$$

</div>

The set of null lines meeting a given compact set is compact (for a point, it is the celestial sphere). For null lines, therefore, a Radon intensity is the same as a finite expected number of lines through every bounded region.

### 3.4 Timelike and spacelike lines

For lines that are not null the stabiliser is unimodular, so an invariant measure exists. It gives infinite mass to the lines through any open set.

<div class="proposition" markdown="1">

**Proposition 5** (timelike and spacelike lines). Let $$d\ge2$$ and let $$\epsilon=+1$$ for timelike and $$\epsilon=-1$$ for spacelike lines. A line is $$\{y+su:s\in\mathbb R\}$$ with $$B(u,u)=\epsilon$$ and $$y\in u^\perp$$, which carries the induced volume $$d^{d-1}y$$. The measure $$\kappa=d\sigma_\epsilon(u)\,d^{d-1}y$$, with $$d\sigma_\epsilon$$ the Lorentz-invariant measure on the hyperboloid $$\{B(u,u)=\epsilon\}$$ (modulo $$u\to-u$$ for $$\epsilon=-1$$, and with $$u$$ future-directed for $$\epsilon=+1$$), is Poincaré invariant and is the unique invariant measure up to a constant factor. The $$\kappa$$-measure of the set of lines meeting a given non-empty open set is infinite. Consequently an invariant point process of timelike lines, or of spacelike lines, for which the expected number of lines meeting every bounded set is finite, is almost surely empty.

</div>

<div class="proof" markdown="1">

*Proof.* The stabiliser of a timelike line is $$SO(d-1)\times\mathbb R$$ and that of a spacelike line is $$SO(1,d-2)\times\mathbb R$$ (up to a finite factor); both are unimodular, so by Weil's criterion an invariant measure exists and is unique up to a constant factor. The product measure $$\kappa$$ is Lorentz invariant, and translations act on $$y$$ by translations of $$u^\perp$$, so $$\kappa$$ is the invariant measure.

A non-empty open set contains a spatial ball $$D$$ of some radius in the hyperplane $$t=0$$. A line with direction $$u$$ meets $$D$$ iff $$y$$ lies in the image of $$D$$ under the projection along $$u$$ onto $$u^\perp$$. A region of the hyperplane $$t=0$$ and its projection span, together with the segment $$[0,1]u$$, prisms of equal volume, and the volume of the prism over a region $$S$$ of the hyperplane is $$\lvert B(e_0,u)\rvert\,\mathrm{vol}(S)$$ because $$e_0$$ is the unit normal of the hyperplane. The projection has $$(d-1)$$-volume $$\lvert B(e_0,u)\rvert\,\mathrm{vol}(D)$$, and

$$
\kappa(\text{lines meeting }D)=\mathrm{vol}(D)\int\lvert B(e_0,u)\rvert\,d\sigma_\epsilon(u).
$$

For timelike $$u=(\cosh\eta,\sinh\eta\,\hat n)$$ this is $$\int\cosh\eta\,\sinh^{d-2}\eta\,d\eta\,d\Omega$$, and for spacelike $$u=(\sinh\eta,\cosh\eta\,\hat n)$$ it is $$\int\lvert\sinh\eta\rvert\cosh^{d-2}\eta\,d\eta\,d\Omega$$; both diverge. Let the intensity measure $$I$$ of the process be invariant, with finite expected number of lines meeting every bounded set. A compact set of lines has bounded directions in the hyperboloid and bounded intercepts, so all its lines meet one bounded set of spacetime; hence $$I$$ is Radon, and by uniqueness $$I=c\kappa$$ with $$c\ge0$$. Finiteness for $$D$$ forces $$c=0$$. $$\square$$

</div>

## 4. Frames modulo a lattice on Poisson points

The Bombelli–Henson–Sorkin theorem says that a Poisson sprinkling carries no equivariant choice of a preferred timelike direction [[Bombelli–Henson–Sorkin]](#ref-bhs). One may try to evade it by attaching to every point a frame, a Lorentz transformation known only modulo an arithmetic subgroup $$\Gamma$$, as a weak local structure that could supply a Wick axis or a finite stencil.

Let $$\Gamma$$ be a lattice in $$G=SO^+(1,d-1)$$: for $$d=2$$ the group $$\{\beta_{ks_0}:k\in\mathbb Z\}$$ generated by the boost $$\beta_{s_0}$$ of rapidity $$s_0$$, for $$d=4$$ for instance the proper orthochronous automorphisms of $$7x_0^2-x_1^2-x_2^2-x_3^2$$, a cocompact arithmetic group. Let $$\Phi$$ be a Poisson process of unit intensity on $$\mathbb R^{1,d-1}$$ whose points $$x$$ carry independent marks $$h_x\in G/\Gamma$$, drawn from the $$G$$-invariant probability measure of the quotient, and write $$\tilde\Phi$$ for the marked process. The law of $$\tilde\Phi$$ is invariant under the Poincaré group, which acts on positions by $$x\mapsto gx+t$$ and on marks by left multiplication. A *rule* is a measurable assignment $$S_x(\tilde\Phi)\subset\Phi\setminus\{x\}$$ to every point $$x$$ of a finite or infinite set of other points, equivariant under $$P$$.

<div class="theorem" markdown="1">

**Theorem 6** (a neighbour rule that sees frames only modulo $$\Gamma$$ is empty or infinite). Under the Palm law of the marked process at a point $$(0,h)$$, for almost every mark $$h$$,

$$
\mathbb P^0_h\big(0<\#S_0<\infty\big)=0 .
$$

Moreover, no equivariant measurable choice of a timelike unit vector (a Wick axis) $$u(\tilde\Phi)$$ exists.

</div>

Given the mark of the point at the origin, the Palm law is invariant under the stabiliser $$\Gamma_h=g\Gamma g^{-1}$$ of that mark ($$h=g\Gamma$$), which is a lattice and so contains hyperbolic elements and is Zariski dense. The normalised counting measure of the neighbours is then a finite $$\Gamma_h$$-invariant measure, which Lemma 1 excludes.

<div class="proof" markdown="1">

*Proof.* By the Slivnyak–Mecke theorem the Palm law at $$(0,h)$$ is the law of $$\tilde\Phi\cup\{(0,h)\}$$. The marked process law is $$P$$-invariant, and $$\gamma\in\Gamma_h$$ fixes the pointed mark $$h$$ (because $$\gamma g\Gamma=g\Gamma$$) and the position $$0$$. So $$\gamma$$ maps the Palm law to itself. Define, for Borel $$A\subset\mathbb R^d\setminus\{0\}$$,

$$
\nu(A)=\mathbb E^0_h\Big[\frac{\#(S_0\cap A)}{\#S_0};\ 0<\#S_0<\infty\Big].
$$

By equivariance, $$S_0(\gamma\tilde\Phi)=\gamma S_0(\tilde\Phi)$$, and by the invariance of the Palm law under $$\gamma$$, $$\nu(\gamma A)=\nu(A)$$. Thus $$\nu$$ is a $$\Gamma_h$$-invariant measure on $$\mathbb R^d\setminus\{0\}$$ of total mass at most 1. The group $$\Gamma_h$$ is a lattice in $$G$$, so it contains hyperbolic elements and, by the Borel density theorem [[Borel]](#ref-borel), is Zariski dense in $$G$$ when $$d\ge3$$; for $$d=2$$ it is generated by a boost. Lemma 1 with $$A=\mathbb R^d$$ gives $$\nu=0$$, that is $$\mathbb P^0_h(0<\#S_0<\infty)=0$$.

For the Wick axis, if $$u(\tilde\Phi)$$ were equivariant, its law would be a $$G$$-invariant probability measure on the hyperboloid $$H^{d-1}\subset\mathbb R^d\setminus\{0\}$$, which Lemma 1 (with $$\Gamma=G$$) excludes. $$\square$$

</div>

A rule that sees frames only through $$\Gamma$$-cosets therefore selects, for almost every mark and almost surely, either no neighbours or infinitely many.

In two dimensions the second alternative occurs in a simple rule. For a point at the origin with mark rapidity $$\theta$$ and a future timelike point $$y=(u,v)$$, $$u,v>0$$, define the $$\Gamma$$-invariant distance $$d(y)=\min_{k}\big\lvert \beta_{-(\theta+ks_0)}y\big\rvert _{\rm E}$$, with $$u=t+x$$, $$v=t-x$$ and the Euclidean norm of $$(t,x)$$. It satisfies $$d(y)^2=uv\cosh(2\delta)$$, with $$\delta$$ the distance from $$\zeta=\frac12\ln(u/v)$$ to $$\theta+s_0\mathbb Z$$. In the coordinates $$(V,\zeta)$$ with $$V=uv$$ the region $$d(y)<1$$ is $$\{V<1/\cosh2\delta(\zeta)\}$$, of area $$\int d\zeta\,(\cosh2\delta(\zeta))^{-1}$$ with a periodic integrand of positive mean, so each strip $$\zeta\in[\zeta_0,\zeta_0+s_0]$$ contains the same area. A Poisson process of unit density with respect to $$du\,dv$$ puts in $$\{d<1\}$$ a number of points that grows like $$\ln L$$ when the Euclidean radius is limited to $$L$$ (the cut $$V\cosh2\zeta\le L^2$$ limits $$\lvert \zeta\rvert $$ to about $$\ln L$$), so the rule "all points with $$d(y)<1$$" selects an infinite neighbour set.

Frames used by fermion regulators on random points, and the consequences for Wilson terms, are discussed on the page [frames, Wilson terms and reflection positivity](/quantum-gravity/frames-and-reflection-positivity/).

## 5. Hecke correspondences between lattices

Sections 2 to 4 concern structure at the real place. A different proposal replaces "a neighbouring point in spacetime" by "a neighbouring lattice": two lattices are neighbours at a prime $$p$$ when they differ by a Kneser step [[Kneser]](#ref-kneser), and a Hecke correspondence moves from a lattice to its neighbours. The question is whether such steps can serve as dynamics of a Lorentz-invariant substrate.

Take $$\mathbb R^4$$ with $$Q=x_0^2-x_1^2-x_2^2-x_3^2$$, and let $$\mathcal Y$$ be the set of lattices $$L\subset\mathbb R^4$$ on which the bilinear form $$B$$ takes integer values and which have Gram determinant $$-1$$. An even unimodular lattice of signature $$(1,3)$$ does not exist (the signature must be divisible by 8), so each $$L\in\mathcal Y$$ is odd, and by the classification of indefinite odd unimodular lattices [[Serre]](#ref-serre) it is isometric to $$\mathbb Z^4$$ with the form $$Q$$. The group $$G=SO^+(1,3)$$ acts transitively on $$\mathcal Y$$ (the automorphism group of $$\mathbb Z^4$$ meets all four components of $$O(1,3)$$), and $$\mathbb Z^4\in\mathcal Y$$ has stabiliser $$\Gamma_0=SO^+(1,3;\mathbb Z)$$, a lattice in $$G$$ by the Borel–Harish-Chandra theorem [[Borel–Harish-Chandra]](#ref-bhc). So $$\mathcal Y\cong G/\Gamma_0$$ carries a $$G$$-invariant probability measure. For an odd prime $$p$$ and $$L\in\mathcal Y$$, a vector $$w\in L\setminus pL$$ with $$Q(w)\equiv0\pmod{p^2}$$ defines the neighbour

$$
L_w'=\{x\in L:B(x,w)\equiv0\ (\mathrm{mod}\ p)\}+\mathbb Z\,\frac wp, \tag{4}
$$

and $$N_p(L)$$ is the set of these lattices. Proposition 7 shows why one passes to a prime: at the real place a lattice supports no equivariant structure.

<div class="proposition" markdown="1">

**Proposition 7** (no summable equivariant kernel on a lattice). Let $$L\in\mathcal Y$$ and $$\Gamma_L=\mathrm{Stab}_G(L)$$. Every $$\Gamma_L$$-invariant summable function $$c:L\to\mathbb C$$ vanishes off $$0$$. Consequently an equivariant rule assigning to each point of $$L$$ a finite set of other points, or a summable kernel, is trivial.

</div>

<div class="proof" markdown="1">

*Proof.* $$\Gamma_L$$ is an arithmetic subgroup of the semisimple group $$G$$ without compact factors, hence Zariski dense (Borel). By the lemma on finite orbits (Lemma 1) of the page [lattice translations with a discrete Lorentz group](/quantum-gravity/discrete-lorentz-group/) every orbit $$\Gamma_Lv$$, $$v\ne0$$, is infinite, and an invariant summable function is constant on infinite orbits, hence zero there. Equivariance of a rule under lattice translations determines it by its value at $$0$$, which is a $$\Gamma_L$$-invariant finite set or kernel. $$\square$$

</div>

<div class="proposition" markdown="1">

**Proposition 8** (Kneser neighbours). For odd $$p$$ and $$L\in\mathcal Y$$: (i) the lattice $$L_w'$$ of (4) lies in $$\mathcal Y$$, depends only on the line $$\langle w\bmod p\rangle$$ of isotropic vectors of $$L/pL$$, and distinct lines give distinct lattices; (ii) $$\#N_p(L)=p^2+1$$ if $$p\equiv3\pmod4$$ and $$(p+1)^2$$ if $$p\equiv1\pmod4$$; (iii) $$N_p(gL)=gN_p(L)$$ for $$g\in G$$, and $$N_p(L)$$ is the set of $$L'\in\mathcal Y$$ with $$[L:L\cap L']=[L':L\cap L']=p$$, so that $$L'\in N_p(L)$$ iff $$L\in N_p(L')$$.

</div>

<div class="proof" markdown="1">

*Proof.* Write $$L_w:=\{x\in L:B(x,w)\equiv0\ (p)\}$$; it has index $$p$$ in $$L$$ because $$L$$ is unimodular and $$w\notin pL$$, so that $$x\mapsto B(x,w)\bmod p$$ is a non-zero functional. A line of isotropic vectors modulo $$p$$ lifts to $$w$$ with $$Q(w)\equiv0\pmod{p^2}$$: if $$Q(w_0)\equiv0\pmod p$$, the linear form $$r\mapsto2B(w_0,r)$$ is non-zero modulo $$p$$ ($$L$$ is unimodular and $$w_0\notin pL$$), so $$w_0+pr$$ with a suitable $$r$$ has $$Q\equiv0\pmod{p^2}$$ (Hensel). Another lift of the same line is $$w'=\lambda w+pr$$ with $$\lambda$$ a unit modulo $$p$$ and $$r\in L$$. Then $$L_{w'}=L_w$$, and $$Q(w')\equiv0\pmod{p^2}$$ gives $$2\lambda pB(w,r)\equiv0\pmod{p^2}$$, so $$r\in L_w$$; hence $$w'/p\in\lambda w/p+L_w$$, and since $$w\in L_w$$ and $$\lambda$$ is a unit, the lattice (4) is the same. Integrality: for $$x,x'\in L_w$$ and integers $$k,k'$$, $$B(x+kw/p,\,x'+k'w/p)=B(x,x')+\big(k'B(x,w)+kB(x',w)\big)/p+kk'Q(w)/p^2\in\mathbb Z$$. Since $$L_w$$ has index $$p$$ in $$L$$ and $$L'_w/L_w\cong\mathbb Z/p$$, the Gram determinant is $$-1\cdot p^2/p^2=-1$$. Two distinct lines give distinct lattices because $$L'_w\cap L=L_w$$ (an element $$x+kw/p$$ lies in $$L$$ iff $$p\mid k$$), and $$L_w/pL$$ is the orthogonal complement of the line. This proves (i).

For (ii) the neighbours correspond to the points of the quadric $$Q=0$$ in $$\mathbb P^3(\mathbb F_p)$$, of which a nondegenerate quaternary form has $$p^2+(1+\varepsilon)p+1$$ with $$\varepsilon=+1$$ if the form is hyperbolic and $$-1$$ if elliptic; for $$\mathrm{diag}(1,-1,-1,-1)$$ the determinant is $$-1$$, and the form is hyperbolic iff $$-1$$ is a square mod $$p$$, that is $$p\equiv1\pmod4$$.

For (iii), equivariance is clear, and $$L'_w\in\mathcal Y$$ with $$L\cap L'_w=L_w$$ of index $$p$$ in both lattices. Conversely let $$L'\in\mathcal Y$$ have $$[L:M]=[L':M]=p$$ for $$M=L\cap L'$$. Choose $$z\in L'\setminus M$$ with $$pz\in M$$ (possible since $$L'/M\cong\mathbb Z/p$$) and put $$w=pz\in L$$. Then $$w\notin pL$$, since otherwise $$z\in L\cap L'=M$$. For $$x\in M\subset L'$$, $$B(x,z)\in\mathbb Z$$, so $$B(x,w)\in p\mathbb Z$$; hence $$M\subset L_w$$, and both have index $$p$$ in $$L$$, so $$M=L_w$$. Also $$Q(w)=p^2Q(z)\in p^2\mathbb Z$$, and $$L'=M+\mathbb Zz=L_w+\mathbb Zw/p=L'_w$$. The index condition is symmetric in $$L$$ and $$L'$$. $$\square$$

</div>

<div class="proposition" markdown="1">

**Proposition 9** (a Hecke step is global in spacetime). Let $$L'\in N_p(L)$$ and $$M=L\cap L'$$. Then $$L'\setminus L=\bigcup_{j=1}^{p-1}(jw/p+M)$$, a union of $$p-1$$ translates of a full-rank lattice. It meets every ball of radius larger than the covering radius of $$M$$. Every lattice on every trajectory of Hecke steps lies in $$\mathcal Y$$, so its Lorentzian form is the fixed form $$Q$$.

</div>

<div class="proof" markdown="1">

*Proof.* $$L'/M\cong\mathbb Z/p$$ is generated by $$w/p$$, and $$jw/p\notin L$$ for $$p\nmid j$$. The other statements follow. $$\square$$

</div>

A Hecke step thus changes the set of points in every region of spacetime at once, so the correspondence has no finite propagation speed. The only Lorentzian metric available is $$Q$$, which is an input; the dynamics produces none. On a different space the correspondence does act as a wave propagator, as Proposition 10 shows.

For $$L\in\mathcal Y$$ set $$E_s(L)=\sum_u(u_0)^{-s}$$, the sum over primitive null vectors $$u$$ of $$L$$ with $$u_0>0$$ (absolutely convergent for $$\mathrm{Re}\,s>2$$). Since $$E_s(gL)=\sum_uB(u,g^{-1}e_0)^{-s}$$ with $$u$$ running over the primitive null vectors of $$L$$, $$E_s(gL)$$ for $$L=\mathbb Z^4$$ is a function of the rest frame $$g^{-1}e_0$$ in the hyperbolic space $$H^3$$ of unit future timelike vectors, invariant under $$\Gamma_0$$.

<div class="proposition" markdown="1">

**Proposition 10** (the Eisenstein sector). For odd $$p$$ and $$\mathrm{Re}\,s>2$$,

$$
\sum_{L'\in N_p(L)}E_s(L')=\big(p^s+c_p+p^{2-s}\big)E_s(L),\qquad c_p=\begin{cases}0&p\equiv3\ (4)\\2p&p\equiv1\ (4).\end{cases} \tag{5}
$$

</div>

To prove (5) we classify, for a fixed primitive null vector $$u$$ of $$L$$, how the neighbour lattices contain multiples of it.

<div class="proof" markdown="1">

*Proof.* Let $$u$$ be primitive null in $$L$$ and $$L'=L'_w$$. (a) If $$w\equiv\lambda u\pmod p$$, then $$u\in L_w$$, and $$u/p\in L'$$: writing $$w=\lambda u+pr$$, $$Q(w)\equiv0\pmod{p^2}$$ gives $$B(u,r)\equiv0\pmod p$$, hence $$B(r,w)=\lambda B(r,u)+pQ(r)\equiv0$$, so $$r\in L_w$$; with $$\mu\lambda\equiv1\pmod p$$, $$\mu\lambda=1+pk$$, one finds $$u/p=\mu w/p-\mu r-ku\in L'$$. The vector $$u/p$$ is primitive in $$L'$$, and contributes $$p^s(u_0)^{-s}$$. There is exactly one such line. (b) If $$B(u,w)\equiv0$$ and $$\bar w\ne\langle\bar u\rangle$$ (bars denote reduction mod $$p$$), then $$u\in L_w$$ and $$u$$ is primitive in $$L'$$, contributing $$(u_0)^{-s}$$. Such lines are the isotropic lines of $$\bar u^\perp$$ other than $$\langle\bar u\rangle$$; the quotient $$\bar u^\perp/\bar u$$ is a two-dimensional quadratic space that is split iff $$p\equiv1\pmod4$$, with two isotropic lines, each lifting to $$p$$ lines of $$\bar u^\perp$$ besides $$\langle\bar u\rangle$$ itself, so there are $$c_p$$ of them. (c) The remaining $$p^2$$ lines have $$B(u,w)\not\equiv0$$; then $$u\notin L'$$, $$pu\in L_w$$ is primitive in $$L'$$, and contributes $$p^{-s}(u_0)^{-s}$$. The total count is $$p^2+1$$ or $$(p+1)^2$$ lines, consistent with $$1+c_p+p^2$$. Conversely every primitive null vector of $$L'$$ is one of $$u/p$$, $$u$$, $$pu$$ for a unique primitive null $$u\in L$$. Summing over $$u$$ gives (5). $$\square$$

</div>

Each function $$B(u,x)^{-s}$$ on $$H^3$$ is an eigenfunction of the Laplacian: in the upper half-space model with $$u$$ at infinity it is proportional to $$y^{s}$$, and $$\Delta y^s=s(2-s)y^s$$ for $$\Delta=-y^2(\partial_x^2+\partial_y^2+\partial_z^2)+y\partial_y$$. Hence $$E_s$$ has Laplace eigenvalue $$s(2-s)$$, which equals $$1+t^2$$ at $$s=1+it$$. Assuming the standard meromorphic continuation of the Eisenstein series $$E_s$$ in $$s$$, (5) continues with it. At $$s=1+it$$ the right side of (5) is $$\big(2p\cos(t\ln p)+c_p\big)E_s(L)$$, so on this sector the operator $$T_p=\sum_{L'\in N_p(L)}$$ acts as

$$
T_p=2p\cos\big(\ln p\,\sqrt{\Delta-1}\big)+c_p .
$$

The cosine is the propagator at time $$\tau=\ln p$$ of the conformally coupled wave equation $$(\partial_\tau^2+\Delta-1)\phi=0$$ on the ultra-static metric $$-d\tau^2+g_{H^3}$$ (the scalar curvature of $$H^3$$ is $$-6$$, so the conformal coupling term is $$-1$$). Its spatial slice is the space of rest frames, which is not spacetime.

<div class="proposition" markdown="1">

**Proposition 11** (trajectory point sets are singular with respect to Poisson). Let $$L_0$$ be distributed by the invariant probability measure on $$\mathcal Y\cong G/\Gamma_0$$, and let $$L_{k+1}$$ be uniformly distributed in $$N_p(L_k)$$, $$k<n$$ (a random walk, which commutes with $$G$$). Let $$M=\bigcap_{k\le n}L_k$$, a lattice of finite index in each $$L_k$$, and let $$t$$ be uniform on $$\mathbb R^4/M$$, independent of the walk given $$M$$. Then $$\Psi=\bigcup_{k\le n}(L_k+t)$$ is a Poincaré-invariant point process, and its law is mutually singular with every Poisson or mixed Poisson process with almost surely diffuse mean measure.

</div>

<div class="proof" markdown="1">

*Proof.* The Lorentz group acts on the walk by $$L_k\mapsto gL_k$$, which preserves the law by the equivariance of $$N_p$$ and the invariance of the measure on $$\mathcal Y$$, and maps $$M$$ to $$gM$$ and the uniform law on $$\mathbb R^4/M$$ to that on $$\mathbb R^4/gM$$. A translation by $$a$$ replaces $$t$$ by $$t+a$$, which preserves the uniform law. (The shift must be uniform modulo $$M$$ and not modulo $$L_0$$: translations by elements of $$M$$ leave $$\Psi$$ unchanged, and a shift uniform modulo the smaller lattice $$L_0$$ would not give a translation-invariant law.) So $$\Psi$$ is invariant.

Every pair $$x,y$$ of distinct points of $$L_0+t$$ has $$2y-x\in L_0+t$$, so almost surely $$\Psi$$ contains infinitely many triples $$(x,y,2y-x)$$ with $$x\ne y$$. As in the proof of Theorem 1(iv), such triples occur with probability zero for a Poisson process with diffuse mean measure, and for a mixed Poisson process whose random mean measure is almost surely diffuse. The event "contains such a triple" has probability 1 for $$\Psi$$ and 0 for those processes. $$\square$$

</div>

The correspondence therefore generates invariant random lattices (as in section 2) and acts as a wave propagator on rest-frame space in the Eisenstein sector. A step is global, the metric is the input form, and the union of the lattices along a trajectory is not a sprinkling.

## 6. Parity and the causal order

Parity $$\Pi$$ is an automorphism of the Minkowski causal order, so the causal order of a point set $$X$$, taken as an abstract relation, is isomorphic to that of $$\Pi X$$. Any handedness must sit in the law of the embedded process. We say that a Poincaré-invariant point process *has a hand* if some ergodic component of its law is not invariant under $$\Pi$$; for an ergodic process this means that its law is mutually singular with the law of its mirror image. By Theorem 1(ii) and (v) the Haar-random lattices of section 2 have no hand.

<div class="proposition" markdown="1">

**Proposition 12** (rules on few points are parity blind, $$d=4$$). Let a rule decide whether $$x\prec y$$ from the positions of at most three points ($$x$$, $$y$$ and at most one auxiliary point), equivariantly under the proper orthochronous Poincaré group. Then it is invariant under parity as well. The first parity-odd invariant needs four points: three difference vectors spanning a spacelike three-plane, together with the time orientation.

</div>

<div class="proof" markdown="1">

*Proof.* After a translation one point is at the origin, and the invariants of a configuration $$c$$ are those of its difference vectors, which span a subspace $$W$$ of dimension at most 2. Then $$W^\perp$$ has dimension at least 2 and contains a spacelike vector $$s$$ (a subspace without spacelike vectors has dimension at most 1). The reflection $$R_s$$ in $$s^\perp$$ is improper, orthochronous, and fixes $$W$$ pointwise. The product $$\Pi R_s$$ is proper orthochronous and maps $$c$$ to $$\Pi c$$, so a rule invariant under the proper group gives the same value on $$c$$ and $$\Pi c$$. With four points the three difference vectors can span a spacelike three-plane $$W$$; the elements of $$O(1,3)$$ fixing $$W$$ pointwise are $$\pm1$$ on the timelike line $$W^\perp$$, and the non-trivial one, $$-1$$ on $$W^\perp$$, reverses time and is not orthochronous. If $$\Pi c=gc$$ for a proper orthochronous $$g$$, then $$g^{-1}\Pi$$ would be an improper orthochronous element fixing $$W$$ pointwise, and there is none. So the proper orthochronous orbit of the configuration differs from that of its parity image, and the sign of the determinant of the three vectors with the future normal is an invariant that is odd under parity. $$\square$$

</div>

The causal relation $$x\prec y$$ itself is a rule on two points, so it is parity blind; a chiral relation defined on a point set by an equivariant rule needs at least four points. The argument applies to any function of at most three points that is invariant under the proper orthochronous Poincaré group, in particular to the factorial moment densities of order at most three of an invariant point process, when they exist: these are parity even, and a parity-odd correlation function first appears at order four.

<div class="proposition" markdown="1">

**Proposition 13** (no hand on Poisson point sets). A Poisson process of constant intensity has no hand, and neither has a mixed Poisson process with a random constant intensity.

</div>

<div class="proof" markdown="1">

*Proof.* The law of a Poisson process of intensity $$\rho$$ is invariant under every affine isometry, including $$\Pi$$; it is ergodic under translations, hence under $$P$$. The ergodic decomposition of a mixed Poisson process with random constant intensity is the mixture of the Poisson laws over $$\rho$$, and each component is $$\Pi$$-invariant. $$\square$$

</div>

The example below uses an arithmetic lattice that is not equivalent to its mirror image.

<div class="proposition" markdown="1">

**Proposition 14** (chiral random lattices exist). Let $$L=\mathbb Z^4$$ carry the integral quadratic form with Gram matrix

$$
\begin{pmatrix}289&119\\119&48\end{pmatrix}\oplus-\begin{pmatrix}83521&18496\\18496&4097\end{pmatrix},
$$

which has signature $$(1,3)$$ and determinant $$-17^6$$, and identify $$L\otimes\mathbb R$$ with $$\mathbb R^{1,3}$$ by an isometry. Then $$L$$ has no improper automorphism. Consequently the random lattice $$\Lambda=gL+t$$, with $$g$$ distributed by the invariant probability measure of $$SO^+(1,3)/SO^+(L)$$ (finite volume by the Borel–Harish-Chandra theorem) and $$t$$ uniform on the torus $$\mathbb R^4/gL$$, is, after rescaling to unit density, a Poincaré-invariant, ergodic point process whose law is mutually singular with its parity image: it has a hand.

</div>

An improper automorphism would have a single rational spinor norm $$r$$. Away from 2 and 17 the lattice is unimodular, which forces $$r$$ to be a unit there, and at 17 this leaves $$r$$ in the class of $$1$$ or $$17$$ modulo squares; but the 17-adic structure of $$L$$ allows an improper automorphism only the classes $$u$$ and $$17u$$ with $$u$$ a non-residue. The spinor norm is the homomorphism $$\theta$$ from an orthogonal group to $$\mathbb Q_p^\times/(\mathbb Q_p^\times)^2$$ that takes a reflection in a vector $$v$$ to the class of $$Q(v)$$. We use two cited facts. For a lattice $$L_p$$ over $$\mathbb Z_p$$ with $$p$$ odd, the orthogonal group $$O(L_p)$$ is generated by reflections (O'Meara [[O'Meara]](#ref-omeara), §92). An isometry of a rational quadratic space is a product of rational reflections, and it is improper iff the number of reflections is odd (Cartan–Dieudonné, in [[O'Meara]](#ref-omeara)), so an improper automorphism $$\sigma$$ of $$L$$ has a spinor norm $$r\in\mathbb Q^\times/(\mathbb Q^\times)^2$$; it is also an improper isometry of $$L_p$$ for every prime $$p$$, with spinor norm the image of $$r$$ in $$\mathbb Q_p^\times/(\mathbb Q_p^\times)^2$$.

<div class="proof" markdown="1">

*Proof.* Over $$\mathbb Z_{17}$$, $$L$$ is diagonalised by an orthogonal change of basis into the form with diagonal entries of $$17$$-adic valuations $$0,1,2,3$$, each with a unit part of the same square class $$u$$, a non-residue modulo 17. To see this, orthogonalise successively, always with a pivot of minimal valuation, so that the new basis vectors stay in $$L\otimes\mathbb Z_{17}$$; the diagonal is $$48,\,-4097,\,-289/48,\,-4913/241$$, with valuations $$0,1,2,3$$ and Legendre symbol $$-1$$ for all four unit parts. Write $$Q(e_i)=\varepsilon_i17^{a_i}$$ with $$a_i=0,1,2,3$$.

Let $$v=\sum c_ie_i$$ be primitive with $$\tau_v\in O(L_{17})$$, which means $$2B(e_i,v)/Q(v)\in\mathbb Z_{17}$$ for all $$i$$, that is $$\mathrm{val}\,Q(v)\le a_i+\mathrm{val}\,c_i$$ for all $$i$$. Put $$\mu=\min_i(a_i+2\mathrm{val}\,c_i)\le\mathrm{val}\,Q(v)$$ and let $$j$$ attain the minimum. Then $$\mu\le\mathrm{val}\,Q(v)\le a_j+\mathrm{val}\,c_j\le a_j+2\mathrm{val}\,c_j=\mu$$, so $$\mathrm{val}\,c_j=0$$, $$\mathrm{val}\,Q(v)=a_j=\mu$$, and $$j$$ is the only index attaining the minimum: if $$i$$ attains it, then $$a_i+2\,\mathrm{val}\,c_i=a_j\le a_i+\mathrm{val}\,c_i$$ forces $$\mathrm{val}\,c_i=0$$ and $$a_i=a_j$$, so $$i=j$$ because the $$a_i$$ are distinct. Then $$Q(v)\equiv\varepsilon_j17^{a_j}c_j^2\pmod{17^{a_j+1}}$$, so the class of $$Q(v)$$ is that of $$\varepsilon_j17^{a_j}$$: $$u$$ for $$a_j$$ even and $$17u$$ for $$a_j$$ odd. Reflections in $$O(L_{17})$$ therefore have spinor norm in $$\{u,17u\}$$ modulo squares. By the cited generation theorem, an improper element of $$O(L_{17})$$ is a product of an odd number of such reflections, and its spinor norm is in $$u\{1,17\}\cdot(\mathbb Q_{17}^\times)^2=\{u,17u\}$$ (as $$u^2$$ is a square). At a prime $$p\ne2,17$$, $$L_p$$ is unimodular (the determinant is a power of 17), so a reflection in $$O(L_p)$$ is in a primitive vector with $$Q(v)$$ a unit, and every element of $$O(L_p)$$ has unit spinor norm.

Suppose $$\sigma\in O(L)$$ is improper, with spinor norm $$r\in\mathbb Q^\times/(\mathbb Q^\times)^2$$. Then $$r$$ has even valuation at every prime $$p\ne2,17$$, so $$r\in\{\pm2^a17^b\}$$ modulo squares. In $$\mathbb Q_{17}$$, $$-1$$ and $$2$$ are squares ($$17\equiv1\pmod8$$), so $$r\equiv17^b$$ is $$1$$ or $$17$$ there. But $$\sigma$$ is an improper element of $$O(L_{17})$$, so its spinor norm is $$u$$ or $$17u$$, and $$u\notin\{1,17\}$$ modulo squares because $$u$$ is a non-residue. This contradicts the equality of the rational spinor norm with its $$17$$-adic image, so no improper automorphism exists.

The law of $$\Lambda$$ is the $$P$$-invariant probability measure on the single orbit $$P\cdot L\cong P/(L\rtimes SO^+(L))$$ of affine lattices, on which $$P$$ acts transitively; it is therefore ergodic. The law of $$\Pi\Lambda$$ is carried by the orbit $$P\cdot\Pi L$$. If the two orbits met, we would have $$\Pi L=AL$$ for some proper orthochronous $$A$$, and $$A^{-1}\Pi$$ would be an improper automorphism of $$L$$. Hence the orbits are disjoint, and the two ergodic laws are mutually singular. $$\square$$

</div>

The lattice $$L$$ is odd, because the diagonal entries of its Gram matrix include $$289$$, $$-83521$$ and $$-4097$$. The map $$x\mapsto-x$$ sends every linear lattice to itself and reverses its causal order. The law of $$\Lambda$$ is invariant under the product of full spatial inversion and time reversal, and a violation of parity invariance of the law coincides with a violation of time-reversal invariance. The same map shows that the abstract causal order of a lattice is isomorphic to its converse, so the notion of chirality of a relation used in section 7 does not apply to it: the hand of Proposition 14 is a property of the embedded process.

## 7. Chirality of finite relations

Section 6 places a hand in the law of an embedded process. A finite relation carries a hand in a different sense, with no embedding at all, and the construction does not bear on orders embedded in Minkowski space, where parity is an automorphism of the Minkowski order (section 6). A *relation* $$R$$ on a finite set $$X$$ is a directed graph with adjacency matrix $$A$$, its converse $$R^T$$ has matrix $$A^T$$, $$K=\mathrm{Aut}(R)$$ is its automorphism group, and $$\mathrm{Nor}(K)$$ the normaliser of $$K$$ in the symmetric group of $$X$$. Every $$\varpi\in\mathrm{Nor}(K)$$ induces an automorphism of $$K$$ and so permutes its irreducible characters. Call $$R$$ *rigidly chiral at $$\chi$$* if $$\chi$$ occurs in the permutation module $$\mathbb C[X]$$ and $$\bar\chi$$ is not in the $$\mathrm{Nor}(K)$$-orbit of $$\chi$$. Let $$U$$ be the sum of the isotypic components of $$\mathbb C[X]$$ over the $$\mathrm{Nor}(K)$$-orbit of $$\chi$$. This notion differs from a hand of a point process: it asks whether the relation is isomorphic to its converse, not whether a law is invariant under a reflection.

<div class="proposition" markdown="1">

**Proposition 15** (a rigid hand on a finite relation). Suppose $$R$$ is rigidly chiral at $$\chi$$. (i) Every permutation $$\varpi$$ with $$\varpi R\varpi^{-1}\in\{R,R^T\}$$ preserves $$U$$ and the complex conjugate $$\bar U$$, and $$U\cap\bar U=0$$. (ii) If $$\varpi R\varpi^{-1}=R^T$$, then for every word $$w$$ in $$A$$ and $$A^T$$, the spectrum of $$w\vert _U$$ equals the complex conjugate of the spectrum of the reversed word $$w^{\rm rev}\vert _U$$. In particular, if some word violates this identity then $$R$$ is not isomorphic to its converse, and no relabelling, with or without reversal of the relation, exchanges $$U$$ with its mirror $$\bar U$$.

</div>

<div class="proof" markdown="1">

*Proof.* (i) $$\mathrm{Aut}(R^T)=\mathrm{Aut}(R)=K$$, so such a $$\varpi$$ normalises $$K$$ and induces an automorphism $$\alpha$$ of $$K$$. It maps the $$\psi$$-isotypic part to the $$\psi\circ\alpha^{-1}$$-isotypic part, which stays in the orbit, so $$U$$ is preserved; permutation matrices are real, so $$\bar U$$ is preserved too. If the orbit of $$\chi$$ and the conjugate of that orbit shared a character, $$\bar\chi$$ would lie in the orbit of $$\chi$$, contradicting rigidity; hence $$U\cap\bar U=0$$. (ii) $$\varpi A\varpi^{-1}=A^T$$ gives $$\varpi\,w(A,A^T)\,\varpi^{-1}=w(A^T,A)=(w^{\rm rev}(A,A^T))^T$$. $$A$$ and $$A^T$$ commute with $$K$$ and so preserve $$U$$ and its orthogonal complement. For a real matrix $$M$$ with this property, $$M^T\vert _U$$ is the adjoint of $$M\vert _U$$ in an orthonormal basis of $$U$$, with the complex-conjugate spectrum. Since $$\varpi$$ preserves $$U$$, the spectrum of $$w\vert _U$$ equals that of $$w(A^T,A)\vert _U$$, which is the conjugate of that of $$w^{\rm rev}\vert _U$$. $$\square$$

</div>

<div class="proposition" markdown="1">

**Proposition 16** (examples and a lower bound on height). (i) Let $$K=C_\ell\rtimes C_q$$ be a Frobenius group with $$\ell$$ a prime and $$q$$ an odd prime dividing $$\ell-1$$. Every automorphism of $$K$$ fixes every linear character $$\omega$$ of $$K$$ and $$\bar\omega\ne\omega$$ for $$\omega\ne1$$; so any relation with $$\mathrm{Aut}(R)\cong K$$ in which a non-trivial $$\omega$$ occurs is rigidly chiral at $$\omega$$. (ii) If every arc of $$R$$ runs from a set $$X_0$$ to a disjoint set $$X_1$$ (a relation of height two), every word satisfies the identity of Proposition 15(ii), so such a relation cannot show a spectral hand.

</div>

<div class="proof" markdown="1">

*Proof.* (i) $$C_\ell$$ is characteristic, so an automorphism $$\alpha$$ acts on it by multiplication by some unit. For $$g\in K$$, $$\alpha\,\mathrm{conj}_g\,\alpha^{-1}=\mathrm{conj}_{\alpha(g)}$$ as maps of $$C_\ell$$; multiplications commute, so $$\alpha(g)$$ acts on $$C_\ell$$ as $$g$$ does. Since $$K/C_\ell$$ acts faithfully on $$C_\ell$$, $$\alpha$$ induces the identity on $$K/C_\ell$$ and fixes every character pulled back from it, which are the linear characters. For odd $$q$$ a non-trivial linear character takes non-real values. (ii) $$A^2=(A^T)^2=0$$, so a non-zero word alternates its letters; words of odd length move between the two sets and square to zero, so their spectrum is $$0$$; words of even length are $$(AA^T)^k$$ or $$(A^TA)^k$$, real symmetric, and $$AA^T$$ and $$A^TA$$ have the same non-zero spectrum. $$\square$$

</div>

Let $$K=F_{21}=\{x\mapsto ax+b:\ a\in\{1,2,4\},\,b\in\mathbb F_7\}$$ and $$X=K$$ with the relation $$x\to sx$$ for $$s\in S=\{x+1,\,x+2,\,2x\}$$ (63 arcs). Right multiplications are automorphisms. An exhaustive backtracking search over bijections of the 21 vertices, extended along arcs from a fixed vertex, finds that the automorphism group of the digraph has order 21, so that it is exactly the right regular action of $$F_{21}$$, and that no bijection maps the digraph to its converse.

The linear character $$\omega$$ of $$F_{21}$$ of order 3 occupies a one-dimensional isotypic component, on which the matrix acts by $$\sum_{s\in S}\omega(s)=2+e^{2\pi i/3}=1.5+0.866i$$. The spectrum on $$U$$ is not closed under complex conjugation, which a rigid hand allows: the matrix is real, so its full spectrum is closed under conjugation, but its restriction to $$U$$ is not.

The 3-dimensional characters $$\psi,\bar\psi$$ of $$F_{21}$$ are exchanged by an automorphism of $$K$$, for instance conjugation by $$x\mapsto3x$$, which sends the translation $$x\mapsto x+1$$ to $$x\mapsto x+3$$ in the other class of elements of order 7, so the pair $$\psi,\bar\psi$$ is not rigidly chiral. Only the linear characters, which every automorphism fixes (Proposition 16(i)), carry a rigid hand. A finite relation with no decoration can therefore have a rigid hand.

## 8. Open: infinite-intensity processes, other lattices, higher dimensions

Poisson processes and Haar-random lattices are not the only invariant point sets. Orbit lattices of anisotropic forms (see [kernel-scale kinematics of a single realisation](/quantum-gravity/per-realisation-kinematics/)), mixtures, and thinnings or superpositions of Haar-random lattices are invariant as well. Theorem 1 and Proposition 2 concern Haar-random lattices only, and the empty-interval law of the others requires a separate computation.

Theorem 4 and Proposition 3 need a Radon first moment. Invariant processes of null lines, null hyperplanes or complexified points with an infinite mean measure but almost surely locally finite realisations are outside their hypotheses. Whether a non-empty invariant hyperboloid-marked process with locally finite ground process exists is not determined. It is also not proved that every Poincaré-homogeneous space of affine subspaces other than points fails to carry an invariant Radon measure together with finite incidence with compact sets; null lines, null hyperplanes, and timelike and spacelike lines are covered by Theorem 4 and Proposition 5. Imaginary parts of positions can be supplied by the state instead of the law, in which case the law is not invariant, and complexified correlation functions can be defined as limits $$y\to0$$. In the Euclidean real form the rotation group $$SO(d)$$ is compact and carries invariant probability measures on the sphere.

Theorem 6 concerns independent marks drawn from the invariant measure on $$G/\Gamma$$ and an equivariant measurable rule. Marks with a law that is not $$G$$-invariant, and rules that sum over infinitely many neighbours with cancelling weights, are outside it. A non-equivariant finite rule, for instance one built from a fixed frame, gives an output whose law is not Lorentz invariant.

For Haar-random lattices in $$d\ge3$$ the empty-interval law is not determined. In $$d=2$$ it is not determined whether the mean of the full causal-set action diverges; Proposition 2 gives the divergence of the zeroth moment of the layers $$m\le3$$. The rescaled limit of Hecke trajectories and the behaviour of $$T_p$$ on the cuspidal part are not determined, and no kernel kinematics has been computed on the chiral lattice of Proposition 14.

## References

1. <a id="ref-bhs"></a>Bombelli, Henson, Sorkin, "Discreteness without symmetry breaking: a theorem," Mod. Phys. Lett. A 24 (2009) 2579, [arXiv:gr-qc/0605006](https://arxiv.org/abs/gr-qc/0605006). Proves that no equivariant measurable map sends a sprinkling to a spacetime direction.
2. <a id="ref-dhs"></a>Dowker, Henson, Sorkin, "Quantum gravity phenomenology, Lorentz invariance and discreteness," Mod. Phys. Lett. A 19 (2004) 1829, [arXiv:gr-qc/0311055](https://arxiv.org/abs/gr-qc/0311055). States that the discreteness of a causal set is locally Lorentz invariant.
3. <a id="ref-sa"></a>Saravani, Aslanbeigi, "On the causal set-continuum correspondence," [arXiv:1403.6429](https://arxiv.org/abs/1403.6429). Compares lattices and sprinklings as discretisations of volume in 1+1 dimensions.
4. <a id="ref-siegel"></a>Siegel, "A mean value theorem in geometry of numbers," Ann. of Math. 46 (1945) 340. The mean value formula used in Theorem 1 and Proposition 2.
5. <a id="ref-moore"></a>Moore, "Ergodicity of flows on homogeneous spaces," Amer. J. Math. 88 (1966) 154. Ergodicity of the Lorentz group on the space of lattices.
6. <a id="ref-borel"></a>Borel, "Density properties for certain subgroups of semi-simple groups without compact components," Ann. of Math. 72 (1960) 179. Zariski density of lattices, used in Theorem 6 and Proposition 7.
7. <a id="ref-bhc"></a>Borel, Harish-Chandra, "Arithmetic subgroups of algebraic groups," Ann. of Math. 75 (1962) 485. Finite covolume of arithmetic subgroups, used in sections 5 and 6.
8. <a id="ref-kneser"></a>Kneser, "Klassenzahlen definiter quadratischer Formen," Arch. Math. 8 (1957) 241. The neighbour construction of section 5.
9. <a id="ref-omeara"></a>O'Meara, *Introduction to Quadratic Forms*, Springer (reprint 2000), §92. Generation of the orthogonal group of a lattice over the p-adic integers (p odd) by reflections (§92), and the Cartan–Dieudonné theorem on isometries of a quadratic space as products of reflections.
10. <a id="ref-folland"></a>Folland, *A Course in Abstract Harmonic Analysis*, CRC Press (2016), chapter 2. Weil's criterion for invariant measures on quotients.
11. <a id="ref-lp"></a>Last, Penrose, *Lectures on the Poisson Process*, Cambridge University Press (2017). Mecke's formula and the Palm law of a Poisson process.
12. <a id="ref-serre"></a>Serre, *A Course in Arithmetic*, Springer Graduate Texts in Mathematics 7 (1973), chapter V. Classification of indefinite unimodular integral quadratic forms, used in section 5 to identify an odd unimodular lattice of signature (1,3) with the integer lattice carrying the form Q.
