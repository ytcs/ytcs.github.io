---
title: Frames, Wilson terms and reflection positivity for fermions on random points
date: 2026-10-29
category: matter
order: 3
description: Fermions on a Poisson sprinkling meet three obstacles when a frame is allowed. A Wilson term on a link operator leaves exact zero modes unless its range exceeds a fixed multiple of the discreteness length. A random point set cannot be mirror symmetric in each realisation and invariant in law under translations; the Gaussian kernel of the Euclidean overlap violates reflection positivity, while a Gabriel-graph kernel satisfies it on mirror-symmetric sets. A frame supplied by a cluster of particles gives a covariant but non-local regulator that violates crossing at one loop.
status: Two theorems proved in full (isolation floor in four dimensions, no random point set that is mirror symmetric in every realisation and translation invariant, in any dimension), several lemmas and propositions proved, one cited criterion for reflection positivity, one-loop statements derived by explicit integrals and partial power counting, numerical checks in two and four dimensions at up to 3000 points.
excludes: Wilson terms with row-sum diagonal and range below 1.09 discreteness lengths on four-dimensional link operators; non-negative link weights as an infrared Weyl operator; per-realisation reflection positivity for translation-invariant point processes; the Gaussian-kernel Euclidean overlap as a reflection-positive theory; untuned cluster-frame regulators as a route to a local Lorentz-invariant effective theory.
---

A fermion on a lattice or a random point set needs three things: a kinetic operator with the right continuum limit, a way to remove the extra species that discretisation produces, and a route from the Euclidean or lattice definition to a Lorentzian theory with positive energy. On a Poisson sprinkling with exact Lorentz invariance each step meets an obstacle once a preferred frame is admitted into the definition. The companion page [Frame-free Weyl operators on random points carry no chirality index](/quantum-gravity/frame-free-chirality/) shows that without a frame the overlap index vanishes in four classes of constructions and that a Euclidean overlap on the same points works. This page examines what the frame costs. Its answers, with their conditions, are the following.

(i) For link operators on a four-dimensional sprinkling, a Wilson term whose diagonal is the row sum leaves an exact zero mode at every element whose range-limited double cone is empty; these are a fraction $$e^{-2\pi (t_{\max}/\ell)^4/3}$$ of the elements, so a fraction below five per cent needs $$t_{\max}\ge1.094\,\ell$$ (Theorem 1). Non-negative link weights give an infrared Weyl operator of the wrong helicity with speed at most $$1/(d-1)$$ (Lemma 1). A covariant scalar Wilson lift has a singular point on the time axis below the band edge in the plane-wave-averaged symbol (Lemma 2 and a numerical check).

(ii) For the Euclidean overlap operator on Poisson points, per-realisation reflection positivity requires each realisation to be mirror symmetric, and no non-empty point process that is invariant in law under translations along a direction is mirror symmetric in every realisation for the reflection in the orthogonal plane, in any dimension (Theorem 2). The Gaussian-kernel Wilson term of the companion page violates one-particle positivity even on a regular lattice, while the Wilson term on the Gabriel graph is positive on every mirror-symmetric configuration, given a cited criterion (Proposition 2). The massless Gabriel overlap has exact zero-mode pairs at vertices of degree one in our numerical sample, and Proposition 3 gives the underlying exact eigenvector; if such pairs occur with positive density, the disorder average of the massless propagator is undefined. Whether the disorder-averaged correlators are reflection positive is open.

(iii) A frame equal to the velocity of a cluster of particles gives a regulator that is covariant for the self-energy, non-local for the vacuum polarisation, and not crossing symmetric at one loop (Propositions 4 to 6).

The statements about link operators (Section 2) concern four dimensions except Lemma 1, which holds in every dimension. The reflection-positivity statements (Section 3) hold in every dimension where they are proved, and all numerical checks there are two-dimensional. Section 4 concerns a four-dimensional Yukawa theory at one loop.

Prior work. Nielsen and Ninomiya showed that a local, translation-invariant, Hermitian lattice fermion with a conserved chiral charge has equally many left and right species [[NN]](#ref-nn). Random-lattice fermions with Voronoi or Delaunay geometry were introduced by Christ, Friedberg and Lee [[CFL]](#ref-cfl) and studied in two dimensions by Friedberg, Lee and Ren [[FLR]](#ref-flr); they are Euclidean and carry no light cone. Bombelli, Henson and Sorkin proved that a sprinkling carries no preferred direction and no finite-valency graph [[BHS]](#ref-bhs), which is why links on a sprinkling have unbounded range. Osterwalder and Schrader showed that Euclidean correlation functions with reflection positivity reconstruct a relativistic theory [[OS1]](#ref-os1), [[OS2]](#ref-os2); Menotti and Pelissetto gave a general proof of reflection positivity for the Wilson action on the lattice [[MP]](#ref-mp), and Kikukawa and Usui proved that the free overlap operator with the nearest-neighbour Wilson kernel is reflection positive for link reflections [[KU1]](#ref-ku1), [[KU2]](#ref-ku2). Jaffe and Janssens give necessary and sufficient conditions on couplings for reflection positivity of Majorana and spin systems [[JJ]](#ref-jj). That a frame in a regulator feeds order-one Lorentz violation into dimension-four operators is the argument of Collins, Perez, Sudarsky, Urrutia and Vucetich [[CPSUV]](#ref-cpsuv); the loop coefficient is computed on [Where a preferred frame leaks into low-energy physics](/quantum-gravity/lorentz-violation-leakage/) and is used below without repeating the computation. Bakamjian and Thomas constructed Poincaré-invariant interacting few-body dynamics in which interactions are specified in the rest frame of a cluster [[BT]](#ref-bt); cluster separability requires Sokolov's packing operators, reviewed for few-body systems by Schweiger, Reichelt and Klink [[SRK]](#ref-srk), and Polyzou constructed cluster-separable models with particle production for a finite number of bare degrees of freedom [[P]](#ref-p).

What is added is a set of exact statements that fix the scope of each obstacle: the isolation floor, the incompatibility theorem, the mirror lemma with its corollary, the degree-one zero modes, the non-local term of a transverse cutoff, and the crossing defect, each with its hypotheses.

## 1. Fermions on points, doubling and reflection positivity

*Lattice fermions and Wilson's term.* The naive discretisation of $$\gamma\cdot\partial$$ on a lattice of spacing $$a$$ has symbol $$i\gamma\cdot\sin(pa)/a$$, which vanishes at the $$2^d$$ corners of the Brillouin zone, so it describes $$2^d$$ species. The Nielsen–Ninomiya theorem says that this cannot be avoided by a local deformation that keeps chiral symmetry [[NN]](#ref-nn). A Wilson term $$r\,(1-\cos pa)/a$$ gives the species at the corners a mass of order $$2r/a$$ and breaks chiral symmetry, at the price that the lattice operator no longer has an exact chiral symmetry; the overlap construction of the companion page restores one in the Ginsparg–Wilson form. On a random point set there is no Brillouin zone, the symbol of a single realisation is not defined, and one works with the mean symbol or with the spectrum. The extra low-lying structure of the operator then has to be looked for in the spectrum, not in the zeros of a periodic symbol.

*Reflection positivity.* Let $$H$$ be the hyperplane $$t=0$$ in Euclidean $$\mathbb R^d$$, $$\theta(t,x)=(-t,x)$$ the reflection, and let a Euclidean field theory be given by correlation functions $$\langle\cdot\rangle$$. Let $$\Theta$$ be the antilinear map that sends a field $$\psi(y)$$ at a point $$y$$ in the upper half-space to $$\bar\psi(\theta y)\gamma_t$$ at its mirror image (with the corresponding rule for $$\bar\psi$$ and the reversal of order in products). The measure is reflection positive (RP) if $$\langle\Theta(F)\,F\rangle\ge0$$ for every polynomial $$F$$ in the fields supported in $$t>0$$. The Osterwalder–Schrader reconstruction theorem says that Euclidean correlation functions that are invariant under Euclidean motions, reflection positive for every reflection and satisfy growth conditions are the Schwinger functions of a Wightman theory: there is a Hilbert space with a unitary representation of the Poincaré group, a vacuum, and a positive Hamiltonian $$P^0\ge0$$ obtained from the Euclidean time translations, which act as $$e^{-tP^0}$$ [[OS1]](#ref-os1), [[OS2]](#ref-os2). Positivity of the energy comes from reflection positivity and nothing else. For a free fermion with propagator $$S=D^{-1}$$, taking $$F$$ linear in the fields shows that a necessary condition is that the matrix

$$
K_{yx}=S(y,\theta x)\,\gamma_t,\qquad x,y\ \text{in the upper half-space}, \tag{1}
$$

is positive semidefinite (the one-particle kernel). For Gaussian measures it is also sufficient; we use only the necessary direction, so that a negative eigenvalue of $$K$$ is a violation of RP. The ordering of indices and the factor $$\gamma_t$$ in (1) are fixed by requiring $$K\succeq0$$ for the nearest-neighbour Wilson operator with $$0<r\le1$$ and the reflection through the midpoints of links, which is the known case [[MP]](#ref-mp), [[KU1]](#ref-ku1); we verified this calibration numerically (Section 3).

On a random point set there are two notions. *Per-realisation* RP requires that for each realisation $$\mathcal P$$ the lower points are the mirror images of the upper points, $$\theta\mathcal P=\mathcal P$$, so that $$\Theta$$ acts on the fields of that realisation. *Averaged-law* RP asks that the disorder-averaged Schwinger functions $$\mathbb E\langle\cdot\rangle_{\mathcal P}$$ be reflection positive although no single realisation is reflection symmetric. Averaging does not preserve RP automatically, because the average of $$\langle\Theta(F)F\rangle_{\mathcal P}$$ is not defined on realisations without a mirror image.

## 2. A frame in the operator

The link-Weyl operator is the nearest analogue of a derivative on a four-dimensional causal set. Take a finite set $$\mathcal P\subset\mathbb R^{1,3}$$ (a sprinkling of density $$\rho=\ell^{-4}$$), its set of links (pairs $$y\prec x$$ with no point between them), and a future-timelike unit vector $$u$$. Write $$\bar\sigma\cdot\Delta=\Delta^0+\vec\sigma\cdot\vec\Delta$$. For each link with displacement $$\Delta=x-y$$ and weight $$w\ge0$$, the bare link operator has entries $$Q_{yx}=iw\,\bar\sigma\cdot\Delta$$ and $$Q_{xy}=Q_{yx}^\dagger$$, so $$Q$$ is Hermitian. Its plane-wave-averaged symbol is $$F(p)=-2\sum_{\rm links}w\,\bar\sigma\cdot\Delta\,\sin(p\cdot\Delta)$$ per element, with the sum over the links incident on an element in the interior of the region.

<div class="lemma" markdown="1">
**Lemma 1** (non-negative weights; any dimension $$d$$). Let $$\Delta_n$$ be finitely many future-causal vectors and $$w_n\ge0$$ not all zero. The matrix $$M_{\mu\nu}=\sum_nw_n\Delta_{n\mu}\Delta_{n\nu}$$ is positive semidefinite as a real matrix, hence never proportional to the Minkowski metric $$\eta_{\mu\nu}$$. The infrared part of $$F$$ is $$-2\bar\sigma^\mu M_{\mu\nu}p^\nu$$. If $$M=\mathrm{diag}(a,b,\dots,b)$$ in some frame, then $$F\simeq-2\,(a\,p^0-b\,\vec\sigma\cdot\vec p)$$, which has the helicity opposite to $$\bar\sigma\cdot p=p^0+\vec\sigma\cdot\vec p$$ and speed $$b/a\le1/(d-1)$$, with equality only if every $$\Delta_n$$ is null.
</div>

<div class="proof" markdown="1">
*Proof.* $$v^\mu M_{\mu\nu}v^\nu=\sum w_n(v\cdot\Delta_n)^2\ge0$$, and $$\eta$$ takes negative values, so $$M\ne c\,\eta$$ unless $$M=0$$, which would need all $$\Delta_n^0=0$$. Expanding $$\sin(p\cdot\Delta)\simeq p\cdot\Delta$$ and averaging the odd terms away, $$F\simeq-2\sum w\,(\Delta^0+\vec\sigma\cdot\vec\Delta)(p^0\Delta^0-\vec p\cdot\vec\Delta)$$, whose time-time part is $$a\,p^0$$ with $$a=\sum w(\Delta^0)^2$$ and whose spatial part is $$-b\,\vec\sigma\cdot\vec p$$ with $$b=\tfrac1{d-1}\sum w\lvert\vec\Delta\rvert^2$$. Causality gives $$\lvert\vec\Delta_n\rvert\le\Delta_n^0$$, so $$(d-1)b\le a$$. $$\square$$
</div>

The point is that a covariant infrared term needs a second moment proportional to $$\eta$$, which has both signs, while a non-negative combination of squares cannot. Signed weights, such as the Benincasa–Dowker layers, can have $$M_{00}$$ and the spatial trace of the same sign, and nothing here excludes them. *Numerical check* (four dimensions, links among interior points, $$u$$ the time axis of the diamond): the quantity $$b/a$$ was $$0.205,0.203$$ at $$N=1000$$ and $$0.210,0.211$$ at $$N=2000$$ for two sprinklings each, below $$1/3$$ as required; the mean time extent of a link was $$0.14$$ to $$0.15$$ of the diamond height at both sizes, so links are not short compared with the region.

<div class="proposition" markdown="1">
**Proposition 1** (no sign change of $$f^0$$ below the range limit). Let $$w_n\ge0$$ vanish unless $$0<u\cdot\Delta_n\le t_{\max}$$ and let $$t_n=u\cdot\Delta_n$$. Then $$f^0(p_0)=-2\sum_nw_nt_n\sin(p_0t_n)<0$$ for $$0<p_0<\pi/t_{\max}$$.
</div>

Every term has $$w_nt_n>0$$ and $$\sin(p_0t_n)>0$$ in that interval. For $$t_{\max}\le\ell$$ the interval contains the whole resolved band $$\lvert p\rvert<\pi/\ell$$. An earlier reading of the unrestricted link operator took zeros of its symbol at momenta of order the inverse region size for species; those positions scale with the region size at fixed density, because links have extent comparable to the region, and the exact eigenvectors do not show them, so they are not species. What remains is that the spectrum of the bare link operator has a large near-zero part: the fraction of eigenvalues with $$\lvert\lambda\rvert$$ below $$2\%$$ of the maximum was $$0.350$$ and $$0.374$$ for two sprinklings of $$N=1000$$ points in a four-dimensional diamond. This is a numerical observation about one size.

### The isolation floor

A hard cutoff removes the unbounded links. Let $$t_{\max}>0$$ and call a pair of related points $$x,y$$ *cut* if $$\lvert u\cdot(x-y)\rvert\le t_{\max}$$.

<div class="theorem" markdown="1">
**Theorem 1** (isolation floor; four dimensions). Let $$\mathcal P\subset\mathbb R^{1,3}$$ be finite, $$u$$ a future-timelike unit vector, $$t_{\max}>0$$.

(H1) Matrices $$Q,W$$ on $$\mathbb C^{\mathcal P}\otimes\mathbb C^2$$ have off-diagonal blocks $$Q_{xy},W_{xy}$$ that vanish unless $$x,y$$ are a cut pair (or a cut link). 

(H2) $$Q_{xx}=0$$ and $$W_{xx}=-\sum_{y\ne x}W_{xy}$$ (the row sum of $$W$$ is zero).

Then every point $$x$$ with no cut pair satisfies $$Q(\delta_x\otimes e)=W(\delta_x\otimes e)=0$$ for all $$e\in\mathbb C^2$$, so $$\ker(aQ+bW)$$ has dimension at least $$2\,\#\{x\text{ without a cut pair}\}$$ for all $$a,b$$. A point has a cut pair if and only if it has a cut link. If $$\mathcal P$$ is a Poisson process of density $$\rho$$ and the double cone $$\{y:\ y\text{ related to }x,\ \lvert u\cdot(x-y)\rvert\le t_{\max}\}$$ of a point $$x$$ lies in the region, then $$x$$ has no cut pair with probability $$e^{-2\pi\rho\,t_{\max}^4/3}$$. This fraction is below $$5\%$$ only if $$t_{\max}\ge(3\ln20/2\pi)^{1/4}\rho^{-1/4}=1.0936\,\ell$$.
</div>

<div class="proof" markdown="1">
*Proof.* If $$x$$ has no cut pair, its row and column vanish in $$Q$$ and, off the diagonal, in $$W$$; (H2) makes the diagonal blocks zero too, so $$\delta_x\otimes e$$ is in both kernels. Suppose $$x$$ has a related point $$y$$ in its future with $$u\cdot(y-x)\le t_{\max}$$. The finite set of such $$y$$ has a minimal element $$z$$ for the causal order. If $$w$$ lay strictly between $$x$$ and $$z$$ then $$0<u\cdot(w-x)<u\cdot(z-x)\le t_{\max}$$, since $$z-w$$ is future causal and $$u$$ is future timelike, so $$w$$ would be a smaller element of the set; hence $$z$$ is linked to $$x$$ by a cut link. The past is the same. In the rest frame of $$u$$ the future part of the double cone is $$\{0<s\le t_{\max},\ \lvert\vec x\rvert\le s\}$$ of volume $$\int_0^{t_{\max}}\tfrac{4\pi}3s^3ds=\pi t_{\max}^4/3$$, and the double cone has twice that. By the Mecke formula the Poisson probability that a given point has no other point in a set of volume $$V$$ is $$e^{-\rho V}$$. Solving $$e^{-2\pi\rho t^4/3}\le\tfrac1{20}$$ gives the stated bound. $$\square$$
</div>

The hypotheses are met by every hard-cutoff link operator with a Wilson term whose diagonal is its row sum, for any weights and any $$r$$; the bound is therefore independent of $$r$$ and of the weights. The theorem concerns exact zero modes only. It does not say that a range above $$1.094\,\ell$$ removes the near-zero part of the spectrum, and it does not bound other kinds of near-zero modes.

*Numerical check.* We sprinkled $$N=3000$$ points in a four-dimensional diamond, three times, and counted the points whose double cone of radius $$t_{\max}$$ lies inside the diamond. For $$t_{\max}/\ell=1.0,1.094,1.25$$ the fractions without a cut pair were $$0.126$$, $$0.043$$, $$0.013$$ (pooled over $$1836$$, $$1529$$ and $$1079$$ interior points), against $$0.123$$, $$0.050$$, $$0.006$$ predicted; the statistical errors are about $$0.008$$, $$0.006$$ and $$0.003$$, so the largest deviation, at $$1.25$$, is about two and a half standard errors. Testing the equivalence in the theorem, the set of points without a cut pair coincided exactly with the set without a cut link in all nine cases.

### A covariant Wilson term on the time axis

In Euclidean signature a Wilson-type term enters the determinant of the kinetic operator as a sum of squares and cannot make it vanish. In Lorentzian signature it enters with the sign of a mass.

<div class="lemma" markdown="1">
**Lemma 2** (time-axis shells). Let $$f^0,\ \vec f,\ W$$ be real continuous functions of $$p_0\in[0,p_s]$$ and let the Hermitian Nambu lift of a Weyl operator be the block matrix $$\begin{pmatrix}f^0+\vec f\cdot\vec\sigma&-W\\-W&f^0-\vec f\cdot\vec\sigma\end{pmatrix}$$, with eigenvalues $$f^0\pm\sqrt{\lvert\vec f\rvert^2+W^2}$$, each doubly degenerate. It is singular exactly where $$D_1=(f^0)^2-\lvert\vec f\rvert^2-W^2$$ vanishes. If $$D_1>0$$ near $$p_0=0^+$$ and $$D_1(p_s)<0$$, then $$D_1$$ has a zero in $$(0,p_s)$$.
</div>

The proof is the intermediate value theorem. In the continuum limit $$f^0=p_0$$, $$\vec f=\vec k$$ and $$D_1=p_0^2-k^2-W^2$$ is the mass-shell condition. If $$W$$ depends on the spatial momentum only, as in a Hamiltonian lattice, $$W(0)=0$$ and the shell at $$\vec k=0$$ is $$p_0=0$$. A covariant Wilson term depends on $$p_0$$ as well: on the time axis $$W\simeq c_1r\ell\,p_0^2$$ and $$\lvert f^0\rvert\simeq c_2p_0$$ for small $$p_0$$, with $$c_1,c_2>0$$ constants of the link distribution, so $$D_1$$ has a second zero at $$p_0\simeq c_2/(c_1r\ell)$$: a pole of the lifted operator at zero spatial momentum that is not the physical particle. The lemma describes a plane-wave-averaged symbol, not the exact spectrum of a sprinkling, and says nothing about eigenvectors.

*Numerical check.* For the bare link set with $$t_{\max}=\ell$$ on $$N=2000$$ points (interior: the central half-height diamond), with $$W(p)=r\ell\,\langle\sum_{\rm links}(1-\cos p\cdot\Delta)\rangle$$ (constant diagonal) and $$f^0$$, $$\vec f$$ the averaged symbol on the time axis, $$D_1$$ is positive at small $$p_0$$ and has its first zero at $$p_0=27.1,\ 18.3,\ 10.5$$ for $$r=0.5,1,2$$ (second sprinkling: $$26.9,\ 18.3,\ 10.5$$), against the band edge $$\pi/\ell=34.9$$. The apex moves roughly like $$r^{-0.7}$$. By Proposition 1, $$f^0$$ itself has no sign change below $$\pi/\ell$$, so each zero is where $$W$$ overtakes $$\lvert f^0\rvert$$. For small $$r$$ the zero leaves the band; we did not look at exact spectra of the lifted operator, so this check does not decide whether the lifted operator has low modes.

## 3. The Euclidean overlap and reflection positivity

Example 1 of the companion page is a Wilson–Dirac operator with a Gaussian kernel on Poisson points of a torus, whose overlap operator has index $$-Q$$ in a flux $$Q$$. Its distances use the Euclidean metric of a chosen axis, and the law of the Poisson process is invariant under every volume-preserving affine map, so the law of the operator is the same for every axis. The question for this section is whether the Euclidean correlation functions are reflection positive, so that the Osterwalder–Schrader theorem produces a Lorentzian theory in which the axis is only the direction chosen for the Wick rotation.

### No ensemble is both reflection positive per realisation and translation invariant

<div class="theorem" markdown="1">
**Theorem 2.** Let $$\nu$$ be the law of a random locally finite subset $$\mathcal P\subset\mathbb R^d$$ ($$d\ge1$$). Let $$n$$ be a unit vector and $$\theta$$ the reflection in the hyperplane through the origin orthogonal to $$n$$.

(H1) $$\nu(\theta\mathcal P=\mathcal P)=1$$.

(H2) $$\nu$$ is invariant under the translations $$x\mapsto x+sn$$ for all real $$s$$.

Then $$\nu(\mathcal P=\emptyset)=1$$.
</div>

<div class="proof" markdown="1">
*Proof.* The idea is that a reflection-symmetric set stays reflection symmetric under translation of the plane, and two reflections in parallel planes compose to a translation. For $$s\in\mathbb R$$ let $$T_s(x)=x+sn$$. The event $$E=\{\theta\mathcal P=\mathcal P\}$$ has probability one, and by (H2) so has $$\{T_s\mathcal P\in E\}=\{\theta_s\mathcal P=\mathcal P\}$$, where $$\theta_s=T_s^{-1}\theta T_s$$ acts as $$x\mapsto\theta x-2sn$$ (because $$\theta n=-n$$). On the intersection of these events over a countable dense set $$S\subset\mathbb R$$, which has probability one, $$\mathcal P$$ is invariant under $$\theta$$ and all $$\theta_s$$, hence under $$\theta\theta_s:x\mapsto x+2sn$$, $$s\in S$$. If $$\mathcal P\neq\emptyset$$ and $$p\in\mathcal P$$, then $$p+2sn\in\mathcal P$$ for all $$s\in S$$, and these points accumulate, contradicting local finiteness. $$\square$$
</div>

Hence a translation-invariant ensemble that is reflection symmetric in every realisation does not exist: the unsymmetrised Poisson process is translation invariant but $$\theta\mathcal P\cap\mathcal P=\emptyset$$ almost surely, since the reflected point set coincides with the original with probability zero, and per-realisation RP is not defined for it. The symmetrised process (a Poisson process in the upper half-space together with its mirror images) is reflection symmetric and, by the theorem, not translation invariant: its pair correlation has a singular part supported on mirror pairs. The theorem uses only translation invariance along $$n$$, a weaker condition than Euclidean invariance. It leaves open lattices, which are invariant only under a discrete group, and the averaged-law RP of the unsymmetrised ensemble.

### The Gaussian kernel violates positivity

For a point $$x$$ at distance $$\delta$$ above the plane, with mirror image $$\theta x$$, the displacement from $$\theta x$$ to $$x$$ is $$2\delta\,e_t$$ and the block of the Gaussian Wilson–Dirac operator (4) of the companion page between them is $$D_W(\theta x,x)=w\,[(2\delta/Z_1)\gamma_t-r/Z_0]$$ with $$w=e^{-2\delta^2/s^2}$$.

<div class="lemma" markdown="1">
**Lemma 3** (cross-plane block of the Gaussian kernel). The block $$\gamma_tD_W(\theta x,x)=w\,[2\delta/Z_1-(r/Z_0)\gamma_t]$$ has eigenvalues $$w(2\delta/Z_1\mp r/Z_0)$$. With $$Z_1/Z_0=s^2$$ it is indefinite exactly when $$\delta<rs^2/2$$.
</div>

This is a computation, using $$\gamma_t^2=1$$. For $$s=1.5$$ and $$r=1$$ the threshold is $$\delta<1.125$$, which almost every Poisson configuration and every lattice of spacing below $$2.25$$ contains. For the nearest-neighbour Wilson operator the same block is $$\tfrac12(1-r\gamma_t)$$, positive semidefinite exactly for $$\lvert r\rvert\le1$$, which is the known range for reflection positivity of the Wilson action (see [[MP]](#ref-mp)). Raising $$r$$ in the Gaussian kernel widens the indefinite strip. The block criterion used here, that positivity of the cross-plane coupling implies RP for a quadratic fermion action, is the standard transfer-matrix argument for quadratic lattice actions (see [[OSe]](#ref-ose), [[JJ]](#ref-jj)); we have not re-derived it for point sets with antiperiodic boundary conditions, and failure of the criterion does not by itself prove a violation. The test that does prove one is the one-particle kernel (1).

*Numerical check.* We computed $$K$$ for the free operator on a $$12\times12$$ square lattice with the plane between two rows. For nearest-neighbour Wilson with $$r=1$$ and $$r=0.6$$ the ratio $$\lambda_{\min}/\lambda_{\max}$$ of the Hermitian part was $$-4\times10^{-16}$$ and $$-3\times10^{-16}$$ (positive semidefinite); for $$r=1.4$$ it was $$-3.0\times10^{-2}$$. For the Gaussian kernel it was $$-3.5\times10^{-2}$$ ($$r=1$$) and $$-0.29$$ ($$r=4$$) on the same lattice, so the violation is not due to disorder. On twenty mirror-symmetric Poisson configurations of about $$144$$ points ($$\rho=1$$, free boundary), the Gaussian kernel with $$r=1$$ violated positivity in $$20$$ of $$20$$ with median ratio $$-7\times10^{-2}$$ and smallest $$-0.17$$.

### The Gabriel graph is positive on mirror-symmetric sets

The *Gabriel graph* of a finite point set joins $$a,b$$ if no other point lies in the closed ball with diameter $$[a,b]$$, that is if $$(a-w)\cdot(b-w)>0$$ for all other $$w$$. For a unit Poisson process it is a.s. the same as the version with open balls.

<div class="proposition" markdown="1">
**Proposition 2** (mirror lemma; any dimension). Let $$\mathcal P\subset\mathbb R^d$$ be finite, with $$\theta\mathcal P=\mathcal P$$ and no point on the hyperplane. Every edge of the Gabriel graph of $$\mathcal P$$ with endpoints on opposite sides of the hyperplane joins a point $$x$$ to its mirror image $$\theta x$$.
</div>

<div class="proof" markdown="1">
*Proof.* Use coordinates $$(\tau,\xi)$$ with the hyperplane at $$\tau=0$$. Take an edge between $$\theta x$$ and $$v$$ with $$x,v$$ in the upper half-space and $$x\ne v$$; we show that it is not Gabriel. A point $$w$$ lies in the closed ball with diameter $$[a,b]$$ exactly when $$(a-w)\cdot(b-w)\le0$$. For $$w=x$$ one finds $$(\theta x-x)\cdot(v-x)=-2\tau_x(\tau_v-\tau_x)$$, and for $$w=\theta v$$ one finds $$(\theta x-\theta v)\cdot(v-\theta v)=2\tau_v(\tau_v-\tau_x)$$. If $$\tau_v\ge\tau_x$$ the first is $$\le0$$, and if $$\tau_v\le\tau_x$$ the second is $$\le0$$. In either case a point of $$\mathcal P$$ lies in the ball, and it is not an endpoint: $$x\ne\theta x,v$$ and $$\theta v\neq\theta x,v$$. $$\square$$
</div>

For the Gabriel operator $$D_G(x,y)=\tfrac12(\gamma\cdot e_{xy}-r)$$ on edges, with $$e_{xy}$$ the unit vector from $$x$$ to $$y$$, and $$D_G(x,x)=\tfrac r2\deg x$$ (no gauge field), the cross-plane coupling matrix $$C_{xy}=\gamma_tD_G(\theta x,y)$$, $$x,y$$ in the upper half-space, is diagonal by the lemma, with blocks $$0$$ or $$\tfrac12(1-r\gamma_t)$$, since $$e_{\theta x,x}=n$$ and $$\gamma\cdot n=\gamma_t$$. It is positive semidefinite for $$\lvert r\rvert\le1$$. The operator is covariant under reflections, because $$\gamma_t(\gamma\cdot e)\gamma_t=-\gamma\cdot\theta e$$. With the criterion cited above, the free Gabriel Wilson operator is therefore RP on every mirror-symmetric finite configuration; the nearest-neighbour Wilson operator on the square lattice is the special case in which the Gabriel graph is the lattice graph.

*Numerical check.* In $$200$$ mirror-symmetric Poisson configurations in the plane (30 to 120 upper points), there were $$1680$$ plane-crossing Gabriel edges and all were mirror pairs. For the free Gabriel operator with $$r=1$$ on twenty configurations of $$144$$ points the kernel (1) was positive semidefinite in all twenty ($$\lambda_{\min}/\lambda_{\max}\ge-6\times10^{-16}$$). For the overlap operator built on it, $$D_{\rm ov}+0.1$$, the kernel was positive semidefinite in $$18$$ of $$20$$; in the other two $$H=\gamma_5(D_G-1)$$ had an exact zero eigenvalue (smallest $$\lvert\mathrm{eig}\,H\rvert<10^{-4}$$), where the sign function is undefined and the numerical value meaningless. Positivity of the overlap kernel is therefore a numerical observation, not a theorem; for the lattice case it is proved in [[KU1]](#ref-ku1), [[KU2]](#ref-ku2).

### Zero modes at vertices of degree one

The disorder average of the massless propagator exists only if the operator is invertible almost surely. On the Gabriel graph it is not.

<div class="proposition" markdown="1">
**Proposition 3** (dangling vertex; even $$d$$). Let $$D_G$$ be the Gabriel Wilson operator with $$r=1$$ and no gauge field, and let $$i$$ be a vertex of degree one with unique neighbour $$j$$ and $$e=e_{ij}$$. Let $$u\in\mathbb C^{2^{d/2}}$$ satisfy $$(1+\gamma\cdot e)u=0$$ (a $$-1$$ eigenvector, which exists since $$\gamma\cdot e$$ has eigenvalues $$\pm1$$). Then $$\psi=\delta_i\otimes u$$ satisfies $$D_G\psi=\tfrac12\psi$$ and $$\psi^\dagger\gamma_5\psi=0$$.
</div>

<div class="proof" markdown="1">
*Proof.* The column of $$D_G$$ at $$i$$ has the diagonal block $$\tfrac r2\deg i=\tfrac12$$ and the block $$D_G(j,i)=\tfrac12(\gamma\cdot e_{ji}-r)=-\tfrac12(\gamma\cdot e+1)$$ at $$j$$, and $$(\gamma\cdot e+1)u=0$$. Since $$\gamma_5$$ anticommutes with $$\gamma\cdot e$$ it maps the $$-1$$ eigenspace of $$\gamma\cdot e$$ to the $$+1$$ eigenspace, which is orthogonal to it. $$\square$$
</div>

Such a vertex is a local event of positive probability for a Poisson process, so its density per unit volume is positive. That the overlap operator then has an exact pair of zero modes is an observation, not a consequence of the proposition. *Numerical observation* (torus of side $$20$$, $$N=400$$, antiperiodic spinors, $$r=1$$, no flux, the same $$80$$ configurations at two values of $$m_0$$): $$8$$ configurations contained a vertex of degree one. At $$m_0=0.9$$ each of these had exactly two zero modes of $$D_{\rm ov}$$ with net chirality zero (index $$0$$), and none of the other $$72$$ had a zero mode. At $$m_0=1$$ the same $$8$$ had two zero modes, $$68$$ had none, and in $$4$$ configurations $$H$$ had an exact zero eigenvalue, where the sign function and hence the construction is undefined; an exact eigenvalue of $$D_G$$ equal to $$m_0$$ would produce this, and none occurs at $$m_0=0.9$$ in the same configurations, so $$m_0=1$$ appears to be a special value for $$r=1$$; we did not identify the eigenvector. The zero modes persist in the cases we looked at; we did not locate them on the vertex.

<div class="lemma" markdown="1">
**Lemma 4** (the mass bound). Let $$H$$ be Hermitian and invertible, $$V=\gamma_5\,\mathrm{sign}\,H$$ and $$D_{\rm ov}=m_0(1+V)$$. For $$m>0$$ every singular value of $$D_{\rm ov}+m$$ is at least $$m$$, and $$\lVert(D_{\rm ov}+m)^{-1}\rVert\le1/m$$.
</div>

*Proof.* $$V$$ is unitary, so $$D_{\rm ov}+m=(m_0+m)+m_0V$$ is normal with eigenvalues $$m_0+m+m_0e^{i\varphi}$$, whose real part is at least $$m$$. For a normal matrix the singular values are the moduli of the eigenvalues. $$\square$$

If $$\nu(\ker D_{\rm ov}\ne0)>0$$, the expectation of $$\lVert D_{\rm ov}^{-1}\rVert$$ is infinite and the averaged massless propagator is not defined; this happens for the Gabriel overlap, if the observation above persists with the size, since degree-one vertices have a density proportional to the volume. With a mass $$m>0$$, Lemma 4 bounds every realisation, and the average of every bounded probe exists.

### Averaged-law positivity

What remains is whether the disorder averages $$\mathbb E\langle\Theta(F)F\rangle_{\mathcal P}$$ on the unsymmetrised ensemble are non-negative. A reflection acts on none of the realisations, so positivity is a linear condition on the averaged state that is not inherited from the realisations by convexity. If it holds, together with Euclidean invariance in law and an integrability condition of the form $$\mathbb E\lvert\langle\Theta(F)\,\tau_t G\rangle_{\mathcal P}\rvert<\infty$$ uniformly in the time translation $$t$$, the Osterwalder–Schrader construction gives a Hilbert space, a positive Hamiltonian and a vacuum for the averaged fields [[OS1]](#ref-os1), [[OS2]](#ref-os2). The reconstructed theory is not that of a free fermion: the averaged state is not quasi-free, since $$\mathbb E[S\otimes S]\ne\mathbb ES\otimes\mathbb ES$$, and the Hamiltonian is not an average of Hamiltonians, because no realisation has a transfer matrix. A test of averaged positivity needs a set of probes and a normalisation for which the averaged diagonal entries are resolved away from zero; we have no such test, and the question is open.

## 4. A frame that belongs to a cluster of particles

Lorentzian averaging over frames is impossible, as shown on [Where a preferred frame leaks into low-energy physics](/quantum-gravity/lorentz-violation-leakage/): there is no Lorentz-invariant probability measure on unit timelike vectors, and a frame in a matter regulator at the discreteness scale produces a dimension-four Lorentz-violating coefficient of order $$g^2/16\pi^2$$ (for a Yukawa coupling $$g$$ and a regulator that depends on the loop momentum orthogonal to the frame $$u$$). A frame can instead be the velocity $$u=P_C/\sqrt{P_C^2}$$ of a cluster $$C$$ of particles, as in the Bakamjian–Thomas construction, where the interaction of a cluster is defined in its rest frame [[BT]](#ref-bt). Then $$u$$ is a function of the state, not a field. Cluster separability, the requirement that distant clusters do not influence each other, needs Sokolov's packing operators [[SRK]](#ref-srk), which are constructed for a fixed number of particles; models with particle production exist for a finite number of bare degrees of freedom [[P]](#ref-p), and the case with infinitely many is open. We examine the one-loop consequences in the Euclidean Yukawa model $$g\bar\psi\psi\phi$$ with a massless fermion, a scalar of mass $$M$$ and a Gaussian regulator $$f(\lvert k_\perp\rvert/\Lambda)=e^{-k_\perp^2/\Lambda^2}$$ on the scalar line, where $$k_\perp$$ is the part of the loop momentum orthogonal to $$u$$. Everything in this section is a four-dimensional one-loop statement.

<div class="proposition" markdown="1">
**Proposition 4** (a line's own frame). Let the regulator frame be $$u=\hat p$$, the direction of the external momentum of the self-energy. Then $$\Sigma(p)=\not p\,A(p^2)$$ with $$A$$ analytic for $$\lvert p^2\rvert<M^2$$. If instead $$u$$ is independent of $$p$$, the vector part is $$\Sigma_\mu=a_\parallel(p\cdot u)u_\mu+a_\perp p_{\perp\mu}$$, and by the loop computation of the page linked above $$16\pi^2(a_\perp-a_\parallel)/g^2\to\tfrac13$$ as $$\Lambda\to\infty$$ at fixed $$M>0$$; so $$\Sigma$$ is covariant at this order only if $$u$$ is parallel or orthogonal to $$p$$.
</div>

<div class="proof" markdown="1">
*Proof of the first statement.* For $$u=\hat p$$ the only vectors are $$u$$ and $$p$$, which are parallel, so the components of $$\Sigma_\mu$$ orthogonal to $$p$$ are odd under $$k_\perp\to-k_\perp$$ and vanish, and $$\Sigma_\mu=\hat p_\mu B(\lvert p\rvert)$$. On the line $$p=s\,e$$, with $$k'=s-k_\parallel$$, the parallel component is

$$
\Sigma_\parallel(s)=g^2\!\int\!\frac{dk'\,d^3k_\perp}{(2\pi)^4}\,\frac{k'}{k'^2+k_\perp^2}\;\frac{f(\lvert k_\perp\rvert/\Lambda)}{(s-k')^2+k_\perp^2+M^2},
$$

and the first factor does not depend on $$s$$. The integral is analytic in $$s$$ for $$\lvert\mathrm{Im}\,s\rvert<M$$, because the poles of the last factor in $$k'$$ are at $$s\pm i\sqrt{k_\perp^2+M^2}$$ and cannot cross the real axis. The change $$k'\to-k'$$ shows $$\Sigma_\parallel(-s)=-\Sigma_\parallel(s)$$, so $$\Sigma_\parallel(s)=s\,A(s^2)$$ with $$A$$ analytic. $$\square$$
</div>

The second statement is the one-loop transfer of a frame in a matter regulator, proved on the linked page for a wider class of regulators. The infrared mass $$M$$ has to be kept until the end: integrating $$k_\parallel$$ first at $$M=0$$ gives $$\tfrac23$$, because the integral is conditionally convergent.

*Numerical check.* With the Gaussian regulator and $$M/\Lambda=10^{-2}$$, direct quadrature of the radial integral gave $$16\pi^2A(0)/g^2=4.26019$$, and the anisotropy coefficient was $$0.33318$$, $$0.333331$$ and $$0.33333330$$ at $$M/\Lambda=10^{-2},10^{-3},10^{-4}$$.

A regulator whose frame is the line's own momentum thus gives no Lorentz violation in the self-energy. The same regulator is not harmless for the vector two-point function.

<div class="proposition" markdown="1">
**Proposition 5** (non-local term of a transverse cutoff). Let a Dirac fermion of mass $$m$$ and charge $$e$$ run in a loop with two vector vertices, and multiply the loop integrand by $$f(\lvert k_\perp\rvert/\Lambda)$$ with $$f\ge0$$, $$k_\perp$$ orthogonal to $$u=\hat q$$, where $$q$$ is the external momentum. As $$q\to0$$ along a fixed direction, $$\Pi^{\mu\nu}\to\alpha\,g^{\mu\nu}+\beta\,\hat q^\mu\hat q^\nu$$, and as $$m\to0$$

$$
\beta=-\frac{2e^2}{3\pi^2}\,\Lambda^2\!\int_0^\infty x\,f(x)\,dx\ \neq0, \tag{2}
$$

up to the sign convention for $$\Pi$$. The limit $$q\to0$$ depends on the direction of $$q$$: $$\Pi^{\mu\nu}$$ contains the non-local term $$\beta\,q^\mu q^\nu/q^2$$.
</div>

<div class="proof" markdown="1">
*Proof.* At $$q=0$$ the loop integrand is $$\mathrm{tr}[\gamma^\mu(\not k+m)\gamma^\nu(\not k+m)]/(k^2+m^2)^2$$ weighted by $$f$$. The difference between the component along $$\hat q$$ and a transverse component is $$8\int\frac{d^4k}{(2\pi)^4}f\,(k_\parallel^2-\lvert k_\perp\rvert^2/3)/(k^2+m^2)^2$$, where $$\lvert k_\perp\rvert^2/3$$ is the average over the three transverse directions. With $$b^2=\lvert k_\perp\rvert^2+m^2$$ the $$k_\parallel$$ integrals are $$\int k_\parallel^2(k_\parallel^2+b^2)^{-2}dk_\parallel=\pi/(2b)$$ and $$\int(k_\parallel^2+b^2)^{-2}dk_\parallel=\pi/(2b^3)$$, so that for $$m\to0$$, $$\lvert k_\perp\rvert=r$$, the combination is $$\pi/(2r)-(r^2/3)\pi/(2r^3)=\pi/(3r)$$ (the two terms are in the ratio $$3:1$$ for every $$f$$), and $$\int d^3k_\perp$$ gives $$4\pi\int r^2\,f\,\pi/(3r)\,dr=(4\pi^2/3)\int r f\,dr$$. Collecting $$8/(2\pi)^4$$ gives $$2/(3\pi^2)$$ times $$\Lambda^2\int xf\,dx$$, which is (2). $$\square$$
</div>

Because $$q_\mu\Pi^{\mu\nu}\propto\alpha+\beta\neq0$$ the term violates the Ward identity, as any momentum cutoff does; what is particular here is that its direction dependence is non-local, so no local counterterm removes it, and that it does not cancel against a gauge-covariant completion unless the completion is built so that it does. Such a completion was not examined. *Numerical check*: for the Gaussian regulator and $$m=0.01\Lambda$$, quadrature gave $$\beta/e^2=-0.0337720$$ (in units $$\Lambda=1$$), against $$-1/(3\pi^2)=-0.0337737$$ from (2).

The remaining requirement of an effective local theory is crossing symmetry: the amplitude for $$\psi\phi\to\psi\phi$$ in the channel with total momentum $$P_s$$ and the amplitude in the crossed channel with total momentum $$P_u$$ must be boundary values of one analytic function. With cluster frames, the first amplitude is computed with $$u=P_s/\sqrt s$$, the second with $$u=P_u/\sqrt u$$.

<div class="proposition" markdown="1">
**Proposition 6** (crossing defect at one loop; derivation). Let the frame-dependent part of the one-loop $$\psi\phi\to\psi\phi$$ amplitude be taken, as $$\Lambda\to\infty$$, to be the sum over internal fermion lines of $$B\,(q\cdot u)\,\not u/u^2$$ with $$B=g^2/(48\pi^2)$$ (Proposition 4), the contributions of vertex and box subgraphs being $$O(p/\Lambda)$$. Then the difference between the amplitude of the $$s$$-channel frame continued to the $$u$$-channel region and the amplitude of the $$u$$-channel frame is

$$
\Delta=g^2B\ \bar u(p')\Big\{S_s\big[\not P_s-(m^2-M^2)\not P_u/u\big]S_s+S_u\big[(m^2-M^2)\not P_s/s-\not P_u\big]S_u\Big\}u(p),\qquad S_q=\frac{\not q+m}{q^2-m^2}, \tag{3}
$$

with the on-shell identity $$P_s\cdot P_u=m^2-M^2$$. The defect is independent of $$\Lambda$$, is generally non-zero, and has a pole at $$s=0$$ with residue proportional to $$m^2-M^2$$.
</div>

The derivation of (3) is as follows. In the $$s$$-channel region the internal line of the $$s$$-diagram has momentum $$P_s$$ and its insertion $$B(P_s\cdot u_s)\not u_s/u_s^2$$ equals $$B\not P_s$$ for $$u_s=P_s/\sqrt s$$, a covariant renormalisation of the line; the line of the $$u$$-diagram has momentum $$P_u$$ and its insertion in the frame $$u_s$$ is $$B(P_u\cdot P_s)\not P_s/s$$, not proportional to $$\not P_u$$. In the $$u$$-channel region the roles are exchanged. The difference of the two assignments gives (3), and $$P_s\cdot P_u=(p'-K)\cdot(p+K)=m^2-M^2$$ on shell. The step that is not derived here is the statement that vertex and box subgraphs contribute only $$O(p/\Lambda)$$ frame dependence; it rests on power counting (their frame-dependent parts are ultraviolet convergent, and the logarithmic part of the vertex is a frame-independent scalar because $$\not u\not u=1$$), not on a computation of those graphs.

*Numerical evaluation of (3).* We implemented the spin-summed ratio $$R=2\mathrm{Re}\,(\Delta A_{\rm tree}^*)/\lvert A_{\rm tree}\rvert^2$$ with $$m=1$$, $$M=0.5$$ in the $$u$$-channel physical region and checked $$P_s\cdot P_u=m^2-M^2$$. In units of $$g^2/16\pi^2$$ (with $$B\to g^2/(16\pi^2)\cdot\tfrac13$$) $$R$$ is $$+12.87$$ at $$\sqrt u=1.55$$, $$\cos\theta=0.3$$; $$-3.26$$ at $$\sqrt u=2$$; $$-0.796$$ at $$\sqrt u=3$$; $$+0.087$$ at $$\sqrt u=5$$, $$\cos\theta=-0.5$$; and $$+0.32$$ at $$\sqrt u=10$$. At $$\sqrt u=2.2$$ and $$\cos\theta=-0.9054$$, where $$s=0$$ inside the physical region, $$sR/B$$ is $$-0.46$$ on both sides, so $$R$$ diverges like $$1/s$$; for $$m=M$$ the residue vanishes by (3). Since $$\Delta$$ is a sum of insertions of the operator $$B(q\cdot u)\not u$$ on internal lines, a counterterm $$-B(q\cdot u_C)\not u_C$$ on every internal line removes it exactly; this is one dimension-four coefficient per order, the tuning of [[CPSUV]](#ref-cpsuv).

Graphs with zero total momentum, tadpoles and vacuum bubbles, have no cluster frame, and the Lorentzian orbit of frames is non-compact, so no normalisable average over it exists (see the linked page). Our computation does not extend to those graphs.

## 5. What the results do not exclude

Each hypothesis marks a route that remains open. We list the routes and their conditions without ranking them.

- Other ranges and signed weights for link operators. Theorem 1 excludes exact zero modes only when $$t_{\max}<1.094\,\ell$$ and the Wilson diagonal is the row sum. A larger range, a non-row-sum diagonal, or signed covariant weights (Lemma 1 needs $$w\ge0$$) are not covered. Whether signed weights reach a covariant infrared Weyl term is not settled here.
- A frame-dependent Wilson term. With coefficient depending on $$u$$, such as one with the sign of $$f^0$$, Lemma 2 does not apply. A frame in the matter sector is subject to the loop transfer of the linked page, and our check of the time-axis zero says nothing about exact spectra.
- Lattices and structured ensembles. Theorem 2 needs translation invariance along $$n$$. Lattices, random shifts of lattices restricted to symmetry-compatible shifts, and reflection planes at special positions are outside it.
- Averaged-law reflection positivity of the unsymmetrised ensemble, with a mass (Lemma 4) or on a graph without degree-one vertices. No violation and no proof are available.
- Kernels satisfying the cross-plane condition other than the Gabriel graph, and gauge fields. Proposition 2 and its corollary are for the free operator; a gauge field with non-trivial cross-plane links needs a separate reflection-symmetric measure for the links.
- Frames in the state. A frame supplied by the state of an exactly Lorentz-invariant theory, and per-particle frames with each particle using its own axis on one shared point set, are not addressed by Propositions 4 to 6. A gauge-covariant cluster-frame completion that cancels the non-local term of Proposition 5, and a subgraph-frame prescription in which each divergent subgraph uses the momentum through it, are not examined.
- Counterterms. The crossing defect (3) is removed by a frame counterterm with one tuned dimension-four coefficient per order.

## 6. Open points

Whether the disorder-averaged correlators of the massive Gabriel overlap on a Poisson process are reflection positive is open, as is the four-dimensional version of the Euclidean operator and of every two-dimensional numerical statement above. The statement that the pair of zero modes of the massless Gabriel overlap is attached to degree-one vertices is an observation at $$N=400$$ and is not proved. The one-loop statements of Section 4 are for a Yukawa model and a Gaussian regulator on the scalar line, with the vertex and box contributions treated by power counting; a gauge theory and the full two-loop structure are not covered. Cluster separability with particle production for infinitely many degrees of freedom is not established.

## References

1. <a id="ref-nn"></a>Nielsen, Ninomiya, "Absence of neutrinos on a lattice," Nucl. Phys. B 185 (1981) 20.
2. <a id="ref-cfl"></a>Christ, Friedberg, Lee, "Random lattice field theory: general formulation," Nucl. Phys. B 202 (1982) 89.
3. <a id="ref-flr"></a>Friedberg, Lee, Ren, "Fermion field on a random lattice," Prog. Theor. Phys. Suppl. 86 (1986) 322.
4. <a id="ref-bhs"></a>Bombelli, Henson, Sorkin, "Discreteness without symmetry breaking: a theorem," [arXiv:gr-qc/0605006](https://arxiv.org/abs/gr-qc/0605006), Mod. Phys. Lett. A 24 (2009) 2579.
5. <a id="ref-os1"></a>Osterwalder, Schrader, "Axioms for Euclidean Green's functions," Commun. Math. Phys. 31 (1973) 83.
6. <a id="ref-os2"></a>Osterwalder, Schrader, "Axioms for Euclidean Green's functions II," Commun. Math. Phys. 42 (1975) 281.
7. <a id="ref-ose"></a>Osterwalder, Seiler, "Gauge field theories on a lattice," Ann. Phys. 110 (1978) 440.
8. <a id="ref-ku1"></a>Kikukawa, Usui, "Reflection positivity of free overlap fermions," [arXiv:1005.3751](https://arxiv.org/abs/1005.3751), Phys. Rev. D 82 (2010) 114503.
9. <a id="ref-ku2"></a>Kikukawa, Usui, "Direct proof of reflection positivity of free overlap Dirac fermion," [arXiv:1012.0152](https://arxiv.org/abs/1012.0152).
10. <a id="ref-jj"></a>Jaffe, Janssens, "Characterization of reflection positivity: Majoranas and spins," [arXiv:1506.04197](https://arxiv.org/abs/1506.04197).
11. <a id="ref-cpsuv"></a>Collins, Perez, Sudarsky, Urrutia, Vucetich, "Lorentz invariance and quantum gravity: an additional fine-tuning problem?," [arXiv:gr-qc/0403053](https://arxiv.org/abs/gr-qc/0403053), Phys. Rev. Lett. 93 (2004) 191301.
12. <a id="ref-bt"></a>Bakamjian, Thomas, "Relativistic particle dynamics. II," Phys. Rev. 92 (1953) 1300.
13. <a id="ref-srk"></a>Schweiger, Reichelt, Klink, "The problem of cluster separability in relativistic few-body systems," [arXiv:1811.09405](https://arxiv.org/abs/1811.09405).
14. <a id="ref-p"></a>Polyzou, "Cluster properties and particle production in relativistic quantum mechanics," [arXiv:nucl-th/0308061](https://arxiv.org/abs/nucl-th/0308061).
15. <a id="ref-mp"></a>Menotti, Pelissetto, "General proof of Osterwalder–Schrader positivity for the Wilson action," Commun. Math. Phys. 113 (1987) 369.
