---
title: Frames, Wilson terms and reflection positivity for fermions on random points
date: 2026-10-29
category: matter
order: 3
description: On a four-dimensional Poisson sprinkling, a link operator cut off at a finite time extent along a frame, with a Wilson term whose rows sum to zero, has exact zero modes at a positive fraction of the interior points; no non-empty translation-invariant random point set is mirror symmetric in every realisation; and a frame supplied by a cluster of particles leaves a crossing defect at one loop in a Yukawa model under a locality hypothesis.
status: Two theorems proved, on exact zero modes of range-cut link operators in four dimensions and on the incompatibility of translation invariance with mirror symmetry for non-empty point sets in any dimension. Lemmas and propositions proved. Reflection positivity of the Gabriel-graph Wilson term conditional on a cited criterion, tested numerically in two dimensions. One-loop cluster-frame statements derived under a locality hypothesis.
excludes: "Zero-row-sum Wilson terms against exact zero modes of range-cut four-dimensional link operators; non-negative frame-isotropic link weights as an infrared Weyl operator; per-realisation reflection positivity of non-empty translation-invariant point processes; reflection positivity of the Gaussian-kernel Wilson-Dirac operator and its overlap on mirror-symmetric sets (numerical, two dimensions); cluster-frame regulators at one loop without a tuned counterterm, given the locality hypothesis."
---

A preferred frame in the definition of a fermion on a Poisson sprinkling has consequences in three places. In four dimensions, a link operator that couples only points less than a time $$t_{\max}$$ apart along the frame, with a Wilson term whose rows sum to zero (its diagonal entries are minus the sums of the others), has exact zero modes at a fraction $$e^{-2\pi (t_{\max}/\ell)^4/3}$$ of the points away from the boundary, $$\ell$$ being the mean spacing (Theorem 1). In any dimension, a non-empty random point set with translation-invariant law is never mirror symmetric in every realisation, so reflection positivity cannot be tested realisation by realisation on a Poisson process (Theorem 2). When the frame is the velocity of a cluster of particles, the one-loop amplitude of a Yukawa model is not crossing symmetric, with a defect fixed by one dimension-four coefficient, provided the frame dependence of internal lines is local (hypothesis (Loc), section 4).

The three parts are independent, and section 1 collects what sections 2 and 3 use. Section 2 also shows that non-negative link weights that are isotropic about a frame give an infrared Weyl operator of the wrong helicity (Lemma 1). Section 3 shows that the Wilson term on the Gabriel graph of a mirror-symmetric set satisfies a cited criterion for reflection positivity, whereas the Wilson-Dirac operator with a Gaussian kernel of [chirality obstructions for frame-free Weyl operators](/quantum-gravity/frame-free-chirality/), and the overlap operator built on it, fail a numerical one-particle positivity test in two dimensions. It also shows exact zero modes of the Gabriel Wilson operator at vertices of degree one (Proposition 3) and a mass bound for the overlap operator (Lemma 4).

Nielsen and Ninomiya showed that a local, translation-invariant lattice fermion with a conserved chiral charge has equally many left and right species [[NN]](#ref-nn), which is why a Wilson term or an overlap construction is needed. Random-lattice fermions on Voronoi or Delaunay geometries [[CFL]](#ref-cfl), [[FLR]](#ref-flr) are Euclidean and have no light cone. Bombelli, Henson and Sorkin proved that a sprinkling has no preferred direction and no finite-valency graph [[BHS]](#ref-bhs). Links of a sprinkling therefore have unbounded extent, and an operator built from them needs a cutoff, which here is a time extent measured along a frame. The Osterwalder–Schrader theorem turns reflection-positive Euclidean correlation functions into a Lorentzian theory with positive energy [[OS1]](#ref-os1), [[OS2]](#ref-os2). Reflection positivity is known on a lattice for the Wilson action [[MP]](#ref-mp) and for the free overlap operator with the nearest-neighbour kernel [[KU1]](#ref-ku1), [[KU2]](#ref-ku2). Collins, Perez, Sudarsky, Urrutia and Vucetich argued that a frame in a regulator feeds order-one Lorentz violation into dimension-four operators [[CPSUV]](#ref-cpsuv), and the one-loop coefficient is computed on the page on [frame leakage](/quantum-gravity/lorentz-violation-leakage/). Frames supplied by clusters are taken in the sense of Bakamjian and Thomas [[BT]](#ref-bt). The new result is the zero-mode count of Theorem 1. Theorem 2, the Gabriel-graph computation and the crossing defect apply the arguments above to this setting.

## 1. Doubling and reflection positivity

The naive lattice derivative of spacing $$h$$ has symbol $$i\gamma\cdot\sin(ph)/h$$, which vanishes at the $$2^d$$ corners of the Brillouin zone and so describes $$2^d$$ species. A Wilson term $$r\,(1-\cos ph)/h$$ gives the extra species a mass $$2r/h$$ at the price of exact chiral symmetry. The overlap construction restores a Ginsparg–Wilson form of chiral symmetry (see [chirality obstructions for frame-free Weyl operators](/quantum-gravity/frame-free-chirality/)). A random point set has no Brillouin zone and the symbol of one realisation is not defined. One works with a plane-wave-averaged symbol (defined in section 2) or with the spectrum.

Let $$\mathcal H$$ be the hyperplane $$t=0$$ in Euclidean $$\mathbb R^d$$ and $$\theta(t,x)=(-t,x)$$ the reflection in it. The antilinear map $$\Theta$$ sends a field $$\psi(y)$$ at a point $$y$$ with $$t>0$$ to $$\bar\psi(\theta y)\gamma_t$$, reverses the order of products and conjugates numbers. A Euclidean fermion theory is reflection positive (RP) if $$\langle\Theta(\mathcal A)\,\mathcal A\rangle\ge0$$ for every polynomial $$\mathcal A$$ in the fields at points with $$t>0$$. By the Osterwalder–Schrader reconstruction theorem, Euclidean-invariant RP correlation functions with suitable growth bounds are the Schwinger functions of a Wightman theory with a unitary Poincaré representation, a vacuum and a Hamiltonian $$P^0\ge0$$, Euclidean time translation acting as $$e^{-tP^0}$$ [[OS1]](#ref-os1), [[OS2]](#ref-os2).

For a free fermion with propagator $$S=D^{-1}$$, taking $$\mathcal A$$ linear in the fields gives a necessary condition: the matrix

$$
\mathcal K_{yx}=S(y,\theta x)\,\gamma_t,\qquad x,y\ \text{in the upper half-space}, \tag{1}
$$

is positive semidefinite. We call this the one-particle test. For Gaussian measures it is also sufficient, but only the necessary direction is used here, so a negative eigenvalue of $$\mathcal K$$ is a violation of RP. The index order and the sign convention of the fermionic pairing in (1) are not derived. They are fixed by the requirement that $$\mathcal K\succeq0$$ for the nearest-neighbour Wilson operator with $$0<r\le1$$ and the reflection through the midpoints of links, the case proved in [[MP]](#ref-mp) and [[KU1]](#ref-ku1). The numerical results of section 3 reproduce that case.

On a random point set there are two notions. RP in every realisation requires that the lower points are the mirror images of the upper points, $$\theta\mathcal P=\mathcal P$$, so that $$\Theta$$ acts on the fields of that realisation. Averaged-law RP asks that the disorder-averaged Schwinger functions $$\mathbb E\langle\cdot\rangle_{\mathcal P}$$ be reflection positive although no single realisation is mirror symmetric. Averaging does not preserve RP automatically, because $$\langle\Theta(\mathcal A)\mathcal A\rangle_{\mathcal P}$$ is not defined on a realisation without a mirror image.

## 2. Link operators and a frame

A four-dimensional causal set has no derivative, and the nearest analogue is an operator on links. Let $$\mathcal P\subset\mathbb R^{1,3}$$ be a finite set (a sprinkling of density $$\rho=\ell^{-4}$$). A link is a pair $$y\prec x$$ of related points with no point between them, $$\Delta=x-y$$ is its displacement and $$w\ge0$$ its weight. We use the signature $$(+,-,-,-)$$ and write $$\bar\sigma\cdot\Delta=\Delta^0+\vec\sigma\cdot\vec\Delta$$ and $$p\cdot\Delta=p^0\Delta^0-\vec p\cdot\vec\Delta$$, so that the covariant Weyl symbol is $$\bar\sigma\cdot p=p^0+\vec\sigma\cdot\vec p$$. The bare link operator $$L$$ has the entries $$L_{yx}=iw\,\bar\sigma\cdot\Delta$$ and $$L_{xy}=L_{yx}^\dagger$$, so $$L$$ is Hermitian. Its plane-wave-averaged symbol is the mean over the elements of the ratio $$(L\psi)(x)/\psi(x)$$ for $$\psi(x)=e^{ip\cdot x}$$, that is

$$
F(p)=F^0(p)+\vec F(p)\cdot\vec\sigma=-2\sum_{\rm links}w\,\bar\sigma\cdot\Delta\,\sin(p\cdot\Delta)\quad\text{per element,}
$$

the sum running over the links in the interior of the region and being divided by the number of elements. Links of a sprinkling have unbounded extent, so a range cutoff is needed, and we measure it along a future-timelike unit vector $$u$$, the frame.

### Non-negative weights

Non-negative weights are the simplest choice, and the question is whether they can give a covariant infrared Weyl term. Lemma 1 is Proposition 4 of [fermions that see only the causal order](/quantum-gravity/order-only-fermions/) in any dimension. That page uses the signature $$(-,+,+,+)$$ and $$\bar\sigma^\mu=(1,-\vec\sigma)$$, and the conclusions agree.

<div class="lemma" markdown="1">
**Lemma 1** (non-negative weights; any dimension $$d$$). Let $$\Delta_1,\dots,\Delta_N$$ be non-zero future-causal vectors in $$\mathbb R^{1,d-1}$$ and $$w_n\ge0$$ not all zero. The matrix $$T^{\mu\nu}=\sum_nw_n\Delta_n^\mu\Delta_n^\nu$$ is positive semidefinite and non-zero, hence not proportional to the Minkowski metric $$\eta$$. Suppose the links are distributed isotropically about the time axis of some frame, so that $$\sum_nw_n\Delta_n^0\vec\Delta_n=0$$ and $$\sum_nw_n\Delta_n^i\Delta_n^j=c_1\delta^{ij}$$. Then the leading term of $$F$$ is

$$
-2\,(c_0\,p^0-c_1\,\vec\sigma\cdot\vec p),\qquad c_0=\sum_nw_n(\Delta_n^0)^2,\quad c_1=\sum_nw_n\lvert\vec\Delta_n\rvert^2/(d-1),
$$

which has the helicity opposite to $$\bar\sigma\cdot p$$ (or no spatial term, if $$c_1=0$$) and speed $$c_1/c_0\le1/(d-1)$$, with equality only if every $$\Delta_n$$ is null.
</div>

<div class="proof" markdown="1">
*Proof.* For every vector $$v$$, $$v_\mu T^{\mu\nu}v_\nu=\sum_nw_n(v\cdot\Delta_n)^2\ge0$$, so $$T$$ is positive semidefinite, whereas $$\eta$$ is indefinite. Hence $$T=c\,\eta$$ forces $$c=0$$ and $$T=0$$, which is excluded because $$T^{00}=\sum_nw_n(\Delta_n^0)^2>0$$ (a non-zero causal vector has $$\Delta^0\ne0$$). Expanding $$\sin(p\cdot\Delta)\simeq p\cdot\Delta$$ gives $$F\simeq-2\sum_nw_n(\Delta_n^0+\vec\sigma\cdot\vec\Delta_n)(p^0\Delta_n^0-\vec p\cdot\vec\Delta_n)$$. By the isotropy assumption the cross terms vanish, the time part is $$c_0\,p^0$$, and the spatial part is $$-c_1\,\vec\sigma\cdot\vec p$$, where the trace of $$\sum_nw_n\Delta_n^i\Delta_n^j=c_1\delta^{ij}$$ fixes $$(d-1)c_1=\sum_nw_n\lvert\vec\Delta_n\rvert^2$$. Causality gives $$\lvert\vec\Delta_n\rvert\le\Delta_n^0$$, so $$(d-1)c_1\le c_0$$, with equality only when all $$\lvert\vec\Delta_n\rvert=\Delta_n^0$$. $$\square$$
</div>

A covariant infrared term needs $$T\propto\eta$$, that is $$T^{00}$$ and the spatial diagonal $$T^{ii}$$ of opposite signs, which a sum of squares cannot supply. Signed weights, such as the layer weights of Benincasa and Dowker [[BD]](#ref-bd), can supply it, and the lemma says nothing about them. For links among interior points of a sprinkled four-dimensional causal diamond, the ratio $$c_1/c_0$$ was between $$0.19$$ and $$0.22$$ at $$N=1000$$ and $$N=2000$$ points, below the bound $$1/3$$. The mean time extent of a link was about $$0.14$$ of the diamond height, so the links are not short compared with the region.

### Exact zero modes of range-cut operators

Let $$t_{\max}>0$$ and call two related points $$x\ne y$$ a *cut pair* if $$\lvert u\cdot(x-y)\rvert\le t_{\max}$$. A cut link is a link that is a cut pair. 
<div class="proposition" markdown="1">
**Proposition 1** (no sign change of $$F^0$$ below the range limit). Let $$w_n\ge0$$ vanish unless $$0<t_n:=u\cdot\Delta_n\le t_{\max}$$. Then $$F^0=-2\sum_nw_nt_n\sin(p_0t_n)<0$$ for $$0<p_0<\pi/t_{\max}$$ on the time axis $$\vec p=0$$.
</div>

Every term has $$w_nt_n>0$$ and $$\sin(p_0t_n)>0$$ in that interval. For $$t_{\max}\le\ell$$ the interval contains the whole resolved band $$\lvert p\rvert<\pi/\ell$$, so a zero of the time-axis symbol below the band edge, as in Lemma 2 below, cannot come from $$F^0$$ alone.

The range cut also decouples every point that has no partner within the cut.

<div class="theorem" markdown="1">
**Theorem 1** (isolation floor; four dimensions). Let $$\mathcal P\subset\mathbb R^{1,3}$$ be finite, $$u$$ a future-timelike unit vector and $$t_{\max}>0$$. Let $$L,W$$ be matrices on $$\mathbb C^{\mathcal P}\otimes\mathbb C^2$$ such that

(a) the off-diagonal blocks $$L_{xy},W_{xy}$$ vanish unless $$x,y$$ are a cut pair (this includes operators supported on cut links);

(b) $$L_{xx}=0$$ and $$W_{xx}=-\sum_{y\ne x}W_{xy}$$ (the rows of $$W$$ sum to zero).

Then every point $$x$$ with no cut pair satisfies $$L(\delta_x\otimes\zeta)=W(\delta_x\otimes\zeta)=0$$ for all $$\zeta\in\mathbb C^2$$, so $$\dim\ker(\alpha L+\beta W)\ge2\,\#\{x\text{ without a cut pair}\}$$ for all $$\alpha,\beta$$. A point has a cut pair if and only if it has a cut link. If $$\mathcal P$$ is a Poisson process of density $$\rho=\ell^{-4}$$ and the double cone $$C_x=\{y:\ y\text{ related to }x,\ \lvert u\cdot(x-y)\rvert\le t_{\max}\}$$ of a point $$x$$ lies in the region, then $$x$$ has no cut pair with probability $$e^{-2\pi\rho\,t_{\max}^4/3}$$. This is at most $$5\%$$ if and only if $$t_{\max}\ge(3\ln20/2\pi)^{1/4}\ell=1.0936\,\ell$$.
</div>

<div class="proof" markdown="1">
*Proof.* If $$x$$ has no cut pair, its row and column vanish in $$L$$ and, off the diagonal, in $$W$$. Condition (b) makes the diagonal blocks zero too, so $$\delta_x\otimes\zeta$$ is in both kernels. For the equivalence, suppose $$x$$ has a related point $$y$$ in its future with $$u\cdot(y-x)\le t_{\max}$$. The finite set of such $$y$$ has a minimal element $$y_0$$ for the causal order. If $$w$$ lay strictly between $$x$$ and $$y_0$$, then $$0<u\cdot(w-x)<u\cdot(y_0-x)\le t_{\max}$$, because $$y_0-w$$ is future causal and $$u$$ is future timelike, so $$w$$ would be a smaller element of the set. Hence $$y_0$$ is linked to $$x$$, and the pair is a cut link. The past is the same. In the rest frame of $$u$$ the future part of the double cone is $$\{0<t\le t_{\max},\ \lvert\vec x\rvert\le t\}$$, of volume $$\int_0^{t_{\max}}\tfrac{4\pi}3t^3dt=\pi t_{\max}^4/3$$, and the double cone has twice that volume. By the Mecke formula the Poisson probability that a given point has no other point in a set of volume $$\mathrm{vol}$$ is $$e^{-\rho\,\mathrm{vol}}$$. Solving $$e^{-2\pi\rho t^4/3}\le\tfrac1{20}$$ gives the stated bound. $$\square$$
</div>

The hypotheses hold for every range-cut link operator with a Wilson term whose rows sum to zero, for any weights and any $$r$$, so the fraction depends on $$t_{\max}/\ell$$ alone. It is positive for every finite $$t_{\max}$$. A sprinkling of a region of fixed shape contains such points once it has enough points, and no value of $$t_{\max}/\ell$$ removes them. The value $$1.094\,\ell$$ is only the range at which the fraction is five per cent, and at $$1.5\,\ell$$ it is $$2.5\times10^{-5}$$. Among $$N$$ points in the bulk the expected number is of order one when $$t_{\max}^4\simeq(3/2\pi)\,\ell^4\ln N$$. The theorem concerns exact zero modes only. It does not bound near-zero modes, and it does not apply to a Wilson term whose rows do not sum to zero.

### A covariant Wilson term on the time axis

A Wilson term couples the two chiral blocks of a Dirac operator, so a Weyl operator with symbol $$F$$ is paired with a second block of opposite chirality, whose spatial part has the opposite sign. Let $$F^0,\vec F$$ be the components of the averaged symbol on the time axis and let

$$
W(p)=2r\ell\sum_{\rm links}w\,\bigl(1-\cos(p\cdot\Delta)\bigr)\quad\text{per element}
$$

be the covariant Wilson term of coefficient $$r$$, averaged in the same way as $$F$$; the factor $$2$$ is the one in $$F$$. Together they give the Hermitian block matrix

$$
\begin{pmatrix}F^0+\vec F\cdot\vec\sigma&-W\\-W&F^0-\vec F\cdot\vec\sigma\end{pmatrix},
$$

which for $$F^0=p_0$$, $$\vec F=\vec k$$ and $$W=m$$ is the Dirac operator of mass $$m$$ in the chiral basis. Its eigenvalues are $$F^0\pm\sqrt{\lvert\vec F\rvert^2+W^2}$$, each doubly degenerate. The question is whether such a term leaves the lifted symbol with no zero of the determinant in the resolved band other than the physical one at $$p_0=0$$, without a frame-dependent coefficient.

<div class="lemma" markdown="1">
**Lemma 2** (time-axis zero). Let $$F^0,\ \vec F,\ W$$ be real continuous functions of $$p_0\in[0,p_s]$$. The block matrix is singular exactly where $$D_1=(F^0)^2-\lvert\vec F\rvert^2-W^2$$ vanishes. If $$D_1>0$$ near $$p_0=0^+$$ and $$D_1(p_s)<0$$, then $$D_1$$ has a zero in $$(0,p_s)$$.
</div>

The lemma is the intermediate value theorem, and what has to be checked is the sign of $$D_1$$ at the two ends. In the continuum limit $$D_1=p_0^2-k^2-m^2$$ is the mass-shell condition. If $$W$$ depends on the spatial momentum only, as in a Hamiltonian lattice, $$W(0)=0$$ and the shell at $$\vec k=0$$ is $$p_0=0$$. A covariant Wilson term depends on $$p_0$$ as well. For small $$p_0$$ and isotropic links, with $$c_0=\sum w(\Delta^0)^2$$ per element as in Lemma 1, $$F^0\simeq-2c_0\,p_0$$, $$\vec F\simeq0$$ and $$W\simeq r\ell\,c_0\,p_0^2$$, so that $$D_1\simeq c_0^2p_0^2\,(4-r^2\ell^2p_0^2)$$. Then $$D_1$$ has a second zero near $$p_0=2/(r\ell)$$, independently of the link distribution at this order. The estimate lies below the band edge $$\pi/\ell$$ when $$r>2/\pi$$. Since $$p_0t_{\max}$$ is of order one there, the expansion is only an estimate, and we located the zero numerically. For the link set with $$t_{\max}=\ell$$ on $$N=2000$$ points of a four-dimensional diamond (interior elements: the central half-height diamond, three sprinklings), $$D_1$$ is positive at small $$p_0$$ and its first zero is at

| $$r$$ | $$0.5$$ | $$1$$ | $$2$$ |
|---|---|---|---|
| estimate $$p_0\ell=2/r$$ | $$4.0$$ | $$2.0$$ | $$1.0$$ |
| first zero of $$D_1$$, $$p_0\ell$$ | $$2.43$$ | $$1.65$$ | $$0.94$$ |

against a band edge $$p_0\ell=\pi$$. By Proposition 1, $$F^0$$ has no sign change below $$\pi/\ell$$, so each zero is where $$W$$ overtakes $$\lvert F^0\rvert$$.

This covariant Wilson term therefore leaves a second zero of the determinant of the lifted symbol, at zero spatial momentum and a non-zero frequency inside the resolved band, in addition to $$p_0=0$$. The leading-order location $$2/(r\ell)$$ does not depend on the link distribution. The lemma and the table concern the plane-wave-averaged symbol. The exact spectrum of the lifted operator on a sprinkling was not examined, so they do not decide whether that operator has low modes.

## 3. The Euclidean overlap and reflection positivity

Example 1 of [chirality obstructions for frame-free Weyl operators](/quantum-gravity/frame-free-chirality/) is a Wilson–Dirac operator with a Gaussian kernel on Poisson points of a torus. Its overlap operator has index $$+Q$$ in a flux $$Q$$, in the orientation convention of that example. The distances use the Euclidean metric of a chosen axis, and the law of the Poisson process is invariant under every volume-preserving affine map, so the law of the operator is the same for every axis. The question here is whether its Euclidean correlation functions are reflection positive, so that the Osterwalder–Schrader theorem produces a Lorentzian theory in which the axis is only the direction of the Wick rotation.

### Mirror symmetry and translation invariance

<div class="theorem" markdown="1">
**Theorem 2** (mirror symmetry excludes translation invariance). Let $$\nu$$ be the law of a random locally finite subset $$\mathcal P\subset\mathbb R^d$$ ($$d\ge1$$), $$n$$ a unit vector and $$\theta$$ the reflection in the hyperplane through the origin orthogonal to $$n$$. Assume

(a) $$\nu(\theta\mathcal P=\mathcal P)=1$$;

(b) $$\nu$$ is invariant under the translations $$x\mapsto x+bn$$ for all real $$b$$.

Then $$\nu(\mathcal P=\emptyset)=1$$.
</div>

<div class="proof" markdown="1">
*Proof.* A reflection-symmetric set stays reflection symmetric when the plane is translated, and two reflections in parallel planes compose to a translation. A locally finite set cannot be invariant under a dense set of translations unless it is empty. In detail, let $$U_b(x)=x+bn$$. The event $$E=\{\theta\mathcal P=\mathcal P\}$$ has probability one, and by (b) so has $$\{U_b\mathcal P\in E\}=\{\theta_b\mathcal P=\mathcal P\}$$, where $$\theta_b=U_b^{-1}\theta U_b$$ acts as $$x\mapsto\theta x-2bn$$ (because $$\theta n=-n$$). On the intersection of these events over all rational $$b$$, which has probability one, $$\mathcal P$$ is invariant under $$\theta$$ and all $$\theta_b$$, hence under $$\theta\theta_b:x\mapsto x+2bn$$. If $$y\in\mathcal P$$, then $$y+2bn\in\mathcal P$$ for all rational $$b$$, and these points accumulate, contradicting local finiteness. $$\square$$
</div>

No non-empty translation-invariant ensemble is therefore mirror symmetric in every realisation. The Poisson process is translation invariant, and almost surely no point of $$\mathcal P$$ has its mirror image in $$\mathcal P$$, because the graph of $$\theta$$ is a null set in $$\mathbb R^d\times\mathbb R^d$$. So $$\theta\mathcal P\cap\mathcal P=\emptyset$$, and RP in every realisation is not defined for it. The symmetrised process (a Poisson process in the upper half-space together with its mirror images) is mirror symmetric and, by the theorem, not translation invariant: its pair correlation has a singular part supported on mirror pairs. The theorem uses translation invariance along $$n$$ only, a weaker condition than Euclidean invariance. It leaves open lattices, which are invariant under a discrete group, and averaged-law RP of the unsymmetrised ensemble.

### The Gaussian kernel

The standard criterion for RP of a quadratic fermion action is the transfer-matrix argument of Osterwalder and Seiler for lattice actions [[OSe]](#ref-ose), in a form given by Jaffe and Janssens [[JJ]](#ref-jj). If the action is covariant under $$\theta$$ and the cross-plane coupling matrix $$C_{xy}=\gamma_tD(\theta x,y)$$, $$x,y$$ in the upper half-space, is positive semidefinite, the free fermion is RP. We cite it and have not re-derived it for point sets without translation symmetry. Failure of the criterion does not by itself prove a violation, and the test that does is the one-particle kernel (1).

For a point $$x$$ at distance $$\delta$$ above the plane, with mirror image $$\theta x$$, the displacement from $$\theta x$$ to $$x$$ is $$2\delta\,e_t$$. The Gaussian Wilson–Dirac operator has the entries $$D_W(x,y)=w\,[\gamma\cdot(y-x)/Z_1-r/Z_0]$$ for $$x\ne y$$, with $$w=e^{-\lvert y-x\rvert^2/2s^2}$$, $$Z_1=2\pi s^4$$ and $$Z_0=2\pi s^2$$, and the diagonal $$(r/Z_0)\sum_yw$$. Its block between $$\theta x$$ and $$x$$ is $$D_W(\theta x,x)=w\,[(2\delta/Z_1)\gamma_t-r/Z_0]$$ with $$w=e^{-2\delta^2/s^2}$$.

<div class="lemma" markdown="1">
**Lemma 3** (cross-plane block of the Gaussian kernel). The block $$\gamma_tD_W(\theta x,x)=w\,[2\delta/Z_1-(r/Z_0)\gamma_t]$$ has eigenvalues $$w(2\delta/Z_1\mp r/Z_0)$$. With $$Z_1/Z_0=s^2$$ it is indefinite exactly when $$\delta<rs^2/2$$.
</div>

This follows from $$\gamma_t^2=1$$. For $$s=1.5$$ and $$r=1$$ the threshold is $$\delta<1.125$$, which is met by every mirror-symmetric configuration with a point that close to the plane and by every lattice of spacing below $$2.25$$. For the nearest-neighbour Wilson operator the same block is $$\tfrac12(1-r\gamma_t)$$, positive semidefinite exactly for $$\lvert r\rvert\le1$$, the known range [[MP]](#ref-mp). Raising $$r$$ in the Gaussian kernel widens the indefinite strip.

The one-particle test (1) gives the following on a $$12\times12$$ square lattice with the plane between two rows (open boundaries, $$s=1.5$$, overlap built with $$m_0$$ and a mass $$\mu$$ added). The entries are $$\lambda_{\min}/\lambda_{\max}$$ of the Hermitian part of $$\mathcal K$$, and values of order $$10^{-12}$$ or smaller in modulus are zero to rounding.

| operator | $$\lambda_{\min}/\lambda_{\max}$$ |
|---|---|
| nearest-neighbour Wilson, $$r=1$$, $$0.6$$ | $$-4\times10^{-16}$$, $$-3\times10^{-16}$$ |
| nearest-neighbour Wilson, $$r=1.4$$ | $$-3.0\times10^{-2}$$ |
| nearest-neighbour overlap, $$m_0=1$$ and $$0.5$$, $$\mu=0.1$$ and $$0.5$$ | at most $$5\times10^{-12}$$ in modulus |
| Gaussian Wilson, $$r=1$$ | $$-3.5\times10^{-2}$$ |
| Gaussian Wilson, $$r=4$$ | $$-0.29$$ |
| Gaussian overlap, $$r=1$$, $$m_0=0.5$$, $$\mu=0.1$$ and $$0.5$$ | $$-4.7\times10^{-2}$$, $$-0.12$$ |

The nearest-neighbour entries reproduce the known positive cases and the negative value at $$r=1.4$$, which calibrates (1). The Gaussian kernel violates the test on a regular lattice, so the violation is not an effect of disorder. On twenty mirror-symmetric Poisson configurations of about $$144$$ points ($$\rho=1$$, free boundary) the Gaussian Wilson operator with $$r=1$$ violated positivity in all twenty (median $$\lambda_{\min}/\lambda_{\max}=-6\times10^{-2}$$), and so did the Gaussian overlap with $$m_0=0.5$$ and $$\mu=0.1$$ (median $$-9\times10^{-2}$$). In these configurations the smallest eigenvalue of $$H$$ in modulus is of order $$10^{-4}$$ to $$10^{-2}$$, so the sign function is defined but sensitive.

### The Gabriel graph on mirror-symmetric sets

The *Gabriel graph* of a finite point set joins $$a,b$$ if no other point lies in the closed ball with diameter $$[a,b]$$, that is if $$(a-w)\cdot(b-w)>0$$ for all other $$w$$. For a Poisson process this is almost surely the same as the version with open balls.

<div class="proposition" markdown="1">
**Proposition 2** (mirror lemma; any dimension). Let $$\mathcal P\subset\mathbb R^d$$ be finite, with $$\theta\mathcal P=\mathcal P$$ and no point on the hyperplane. Every edge of the Gabriel graph of $$\mathcal P$$ with endpoints on opposite sides of the hyperplane joins a point $$x$$ to its mirror image $$\theta x$$.
</div>

<div class="proof" markdown="1">
*Proof.* Use coordinates $$(\tau,\xi)$$ with the hyperplane at $$\tau=0$$. Take an edge between $$\theta x$$ and $$v$$ with $$x,v$$ in the upper half-space and $$x\ne v$$, and we show that it is not Gabriel. A point $$w$$ lies in the closed ball with diameter $$[a,b]$$ exactly when $$(a-w)\cdot(b-w)\le0$$. For $$w=x$$ one finds $$(\theta x-x)\cdot(v-x)=-2\tau_x(\tau_v-\tau_x)$$, and for $$w=\theta v$$ one finds $$(\theta x-\theta v)\cdot(v-\theta v)=2\tau_v(\tau_v-\tau_x)$$. If $$\tau_v\ge\tau_x$$ the first is $$\le0$$, and if $$\tau_v\le\tau_x$$ the second is $$\le0$$. In either case a point of $$\mathcal P$$ lies in the ball, and it is not an endpoint: $$x\ne\theta x,v$$ and $$\theta v\neq\theta x,v$$. $$\square$$
</div>

Let $$D_G(x,y)=\tfrac12(\gamma\cdot e_{xy}-r)$$ on the edges of the graph, with $$e_{xy}$$ the unit vector from $$x$$ to $$y$$, and $$D_G(x,x)=\tfrac r2\deg x$$ (no gauge field). By the proposition the cross-plane coupling matrix $$C_{xy}=\gamma_tD_G(\theta x,y)$$ is diagonal, with blocks $$0$$ or $$\tfrac12(1-r\gamma_t)$$, because $$e_{\theta x,x}=n$$ and $$\gamma\cdot n=\gamma_t$$. It is positive semidefinite for $$\lvert r\rvert\le1$$. The operator is covariant under reflections, since $$\gamma_t(\gamma\cdot e)\gamma_t=-\gamma\cdot\theta e$$. With the criterion above, the free Gabriel Wilson operator is therefore RP on every mirror-symmetric finite configuration without gauge field. The nearest-neighbour Wilson operator on the square lattice is the special case in which the Gabriel graph is the lattice graph.

The one-particle test agrees. For the free Gabriel operator with $$r=1$$ on twenty mirror-symmetric Poisson configurations of $$144$$ points the kernel (1) was positive semidefinite in all twenty. For the overlap operator built on it ($$m_0=1$$, plus a mass $$0.1$$) it was positive semidefinite in $$19$$ of $$20$$. In the other configuration $$H=\gamma_5(D_G-1)$$ had an exact zero eigenvalue, where the sign function is undefined. Positivity of the Gabriel overlap kernel is thus a numerical observation, not a theorem, and for the lattice case it is proved in [[KU1]](#ref-ku1), [[KU2]](#ref-ku2).

### Zero modes at vertices of degree one

The disorder average of the massless propagator exists only if the operator is invertible almost surely. On the Gabriel graph it is not invertible in the sample below.

<div class="proposition" markdown="1">
**Proposition 3** (dangling vertex; even $$d$$). Let $$D_G$$ be the Gabriel Wilson operator with $$r=1$$ and no gauge field, and let $$i$$ be a vertex of degree one with unique neighbour $$j$$ and $$e=e_{ij}$$. Let $$\chi\in\mathbb C^{2^{d/2}}$$ satisfy $$(1+\gamma\cdot e)\chi=0$$ (a $$-1$$ eigenvector, which exists since $$\gamma\cdot e$$ has eigenvalues $$\pm1$$). Then $$\psi=\delta_i\otimes\chi$$ satisfies $$D_G\psi=\tfrac12\psi$$ and $$\psi^\dagger\gamma_5\psi=0$$.
</div>

<div class="proof" markdown="1">
*Proof.* The column of $$D_G$$ at $$i$$ has the diagonal block $$\tfrac r2\deg i=\tfrac12$$ and the block $$D_G(j,i)=\tfrac12(\gamma\cdot e_{ji}-r)=-\tfrac12(\gamma\cdot e+1)$$ at $$j$$, and $$(\gamma\cdot e+1)\chi=0$$. Since $$\gamma_5$$ anticommutes with $$\gamma\cdot e$$ it maps the $$-1$$ eigenspace of $$\gamma\cdot e$$ to the $$+1$$ eigenspace, which is orthogonal to it. $$\square$$
</div>

A vertex of degree one is a local event of positive probability for a Poisson process, so its density per unit volume is positive. The proposition gives an eigenvector of $$D_G$$ with chirality zero. It does not by itself give a zero mode of the overlap operator, since $$\mathrm{span}\{\psi,\gamma_5\psi\}$$ is not invariant under $$H=\gamma_5(D_G-m_0)$$. We looked for such zero modes on a torus of side $$20$$ with $$N=400$$ points (antiperiodic spinors, $$r=1$$, no flux). Of $$300$$ random configurations, $$18$$ contained a vertex of degree one. Of ten of these, nine had exactly two overlap zero modes at $$m_0=0.9$$, of total chirality zero, and the tenth, which had two degree-one vertices, had none. None of ten configurations without a degree-one vertex had a zero mode. At $$m_0=1$$ the matrix $$H$$ had an exact zero eigenvalue in one of the ten, where the sign function and hence the construction is undefined. Zero modes of the overlap operator therefore accompany degree-one vertices in most of this sample without following from Proposition 3, and they do not come one pair per vertex. We did not locate them on a vertex.

For the disorder average the relevant bound is the following.

<div class="lemma" markdown="1">
**Lemma 4** (mass bound). Let $$H$$ be Hermitian and invertible, $$V=\gamma_5\,\mathrm{sign}\,H$$ and $$D_{\rm ov}=m_0(1+V)$$. For $$\mu>0$$ every singular value of $$D_{\rm ov}+\mu$$ is at least $$\mu$$, and $$\lVert(D_{\rm ov}+\mu)^{-1}\rVert\le1/\mu$$.
</div>

<div class="proof" markdown="1">
*Proof.* $$V$$ is unitary, so $$D_{\rm ov}+\mu=(m_0+\mu)+m_0V$$ is normal with eigenvalues $$m_0+\mu+m_0e^{i\varphi}$$, whose real part is at least $$\mu$$. For a normal matrix the singular values are the moduli of the eigenvalues. $$\square$$
</div>

If $$\nu(\ker D_{\rm ov}\ne0)>0$$, the expectation of $$\lVert D_{\rm ov}^{-1}\rVert$$ is infinite and the averaged massless propagator is not defined. This happens for the Gabriel overlap if the observation above persists with the size, since degree-one vertices have a density proportional to the volume. With a mass $$\mu>0$$, Lemma 4 bounds every realisation, and the average of every bounded probe exists.

### Averaged-law positivity

Whether the disorder averages $$\mathbb E\langle\Theta(\mathcal A)\mathcal A\rangle_{\mathcal P}$$ on the unsymmetrised ensemble are non-negative is open. A reflection acts on none of the realisations, so positivity is a linear condition on the averaged state that realisations do not inherit by convexity. If it holds, together with Euclidean invariance in law and an integrability condition of the form $$\mathbb E\lvert\langle\Theta(\mathcal A)\,\tau_t\mathcal B\rangle_{\mathcal P}\rvert<\infty$$ uniformly in the time translation $$t$$, the Osterwalder–Schrader construction gives a Hilbert space, a positive Hamiltonian and a vacuum for the averaged fields. The reconstructed theory is not that of a free fermion. The averaged state is not quasi-free, since $$\mathbb E[S\otimes S]\ne\mathbb ES\otimes\mathbb ES$$, and the Hamiltonian is not an average of Hamiltonians, because no realisation has a transfer matrix. A test needs probes and a normalisation for which the averaged diagonal entries are resolved away from zero.

## 4. A frame that belongs to a cluster of particles

The page on [frame leakage](/quantum-gravity/lorentz-violation-leakage/) proves that no Lorentz-invariant probability measure on future unit timelike vectors exists (Proposition 6 there), so a frame cannot be averaged away. It also shows, for a Yukawa coupling $$g$$ and a regulator of the scalar line that depends on the loop momentum orthogonal to the frame, that the frame shifts the fermion speed by an amount of order $$g^2/16\pi^2$$ (Theorem 8 there). A frame can instead be the velocity $$u=P_C/\sqrt{P_C^2}$$ of a cluster $$C$$ of particles with total momentum $$P_C$$, so that $$u$$ is a function of the state and not a field. In the construction of Bakamjian and Thomas the mass operator of a system is a function of the internal variables in its rest frame, and interactions depend on the total momentum only through this velocity [[BT]](#ref-bt). Cluster separability, the requirement that distant clusters do not influence each other, then needs Sokolov's packing operators [[SRK]](#ref-srk), which are constructed for a fixed number of particles. Models with particle production exist for a finite number of bare degrees of freedom [[P]](#ref-p), and the case with infinitely many is open.

The one-loop consequences are examined in the Euclidean Yukawa model $$g\bar\psi\psi\phi$$ with a massless fermion, a scalar of mass $$M$$ and the regulated scalar propagator $$G(q)=f(\lvert q_\perp\rvert/\Lambda)/(q^2+M^2)$$, with $$f(x)=e^{-x^2}$$ and $$q_\perp$$ the part of $$q$$ orthogonal to $$u$$. The inverse propagator of the fermion is $$\Gamma(p)=i\gamma\cdot(p+\Sigma(p))$$ with

$$
\Sigma_\mu(p)=g^2\!\int\!\frac{d^4k}{(2\pi)^4}\,\frac{k_\mu}{k^2}\,G(p-k),
$$

as on the linked page. Everything in this section is a four-dimensional one-loop statement. Two choices of frame matter in what follows: the frame equal to the momentum of the line itself, treated in Propositions 4(a) and 5, and a frame independent of the line's momentum, treated in Proposition 4(b). Both occur in the crossing amplitude of Derivation 1, where the frame of the direct channel is the momentum of one exchanged line and is independent of the momentum of the other.

<div class="proposition" markdown="1">
**Proposition 4** (a line's own frame). (a) Let the regulator frame be $$u=\hat p$$, the direction of the external momentum. Then $$\Sigma_\mu(p)=p_\mu\,\omega(p^2)$$ with $$\omega$$ analytic for $$\lvert p^2\rvert<M^2$$. (b) If $$u$$ is independent of $$p$$, then to first order in $$p$$ the self-energy is $$\Sigma_\mu=a_\parallel(p\cdot u)u_\mu+a_\perp p_{\perp\mu}$$ with $$p_\perp=p-(p\cdot u)u$$, and $$16\pi^2(a_\perp-a_\parallel)/g^2\to\tfrac13$$ as $$M/\Lambda\to0$$ (proved on the page on [frame leakage](/quantum-gravity/lorentz-violation-leakage/), Theorem 8, for a wider class of regulators). So $$\Sigma$$ is covariant at this order only if $$u$$ is parallel or orthogonal to $$p$$.
</div>

<div class="proof" markdown="1">
*Proof of (a).* For $$u=\hat p$$ the only vectors are $$u$$ and $$p$$, which are parallel, so the components of $$\Sigma_\mu$$ orthogonal to $$p$$ are odd under $$k_\perp\to-k_\perp$$ and vanish, and $$\Sigma_\mu=\hat p_\mu B(\lvert p\rvert)$$. On the line $$p=z\,\hat p$$, with $$k'$$ the component of $$k$$ along $$\hat p$$, the parallel component is

$$
\Sigma_\parallel(z)=g^2\!\int\!\frac{dk'\,d^3k_\perp}{(2\pi)^4}\,\frac{k'}{k'^2+k_\perp^2}\;\frac{f(\lvert k_\perp\rvert/\Lambda)}{(z-k')^2+k_\perp^2+M^2},
$$

and the first factor does not depend on $$z$$. The integral is analytic in $$z$$ for $$\lvert\mathrm{Im}\,z\rvert<M$$, because the poles of the last factor in $$k'$$ are at $$z\pm i\sqrt{k_\perp^2+M^2}$$ and cannot cross the real axis. The change $$k'\to-k'$$ shows $$\Sigma_\parallel(-z)=-\Sigma_\parallel(z)$$, so $$\Sigma_\parallel(z)=z\,\omega(z^2)$$ with $$\omega$$ analytic. $$\square$$
</div>

The scalar mass has to be kept until the end. At $$M=0$$ the integral in (b) is conditionally convergent, and integrating over the component along $$u$$ first gives a different value. The coefficient in (b) was obtained at momenta $$\lvert p\rvert<M$$. Evaluating the same integrals at finite $$\lvert p\rvert$$, with $$p$$ along $$u$$ and orthogonal to $$u$$, gives for $$M=0.01\Lambda$$ the values $$0.3320,\ 0.3241,\ 0.2828$$ of $$16\pi^2(a_\perp-a_\parallel)/g^2$$ at $$\lvert p\rvert=0.03,\ 0.1,\ 0.3\,\Lambda$$, and for $$M=0.05\Lambda$$ the values $$0.3227,\ 0.2823$$ at $$\lvert p\rvert=0.1,\ 0.3\,\Lambda$$. The coefficient therefore stays close to $$1/3$$ for $$M\ll\lvert p\rvert\ll\Lambda$$, falling below it by about $$0.001$$, $$0.009$$ and $$0.05$$ at $$\lvert p\rvert=0.03,\ 0.1,\ 0.3\,\Lambda$$ for $$M=0.01\Lambda$$. A fermion mass $$m=0.05\Lambda$$ in the loop changes the values at $$\lvert p\rvert=0.03,\ 0.1\,\Lambda$$ to $$0.3279$$ and $$0.3210$$.

A regulator whose frame is the line's own momentum thus gives no Lorentz violation in the self-energy. The next proposition applies the same regulator, with the frame equal to the line's own momentum, to the vector two-point function, and shows that the absence of frame dependence in the self-energy does not carry over to it. The charge $$\kappa$$ below is unrelated to the Yukawa coupling.

<div class="proposition" markdown="1">
**Proposition 5** (non-local term of a transverse cutoff). Let a Dirac fermion of mass $$m$$ and charge $$\kappa$$ run in a loop with two vector vertices in Euclidean signature, and multiply the loop integrand by $$f(\lvert k_\perp\rvert/\Lambda)$$ with $$f\ge0$$, $$k_\perp$$ orthogonal to $$u=\hat q$$, where $$q$$ is the external momentum. As $$q\to0$$ along a fixed direction, $$\Pi^{\mu\nu}\to\Pi_0\,(\delta^{\mu\nu}-\hat q^\mu\hat q^\nu)$$ with $$q_\mu\Pi^{\mu\nu}=0$$, and as $$m\to0$$

$$
\Pi_0=\frac{2\kappa^2}{3\pi^2}\,\Lambda^2\!\int_0^\infty x\,f(x)\,dx\ \neq0, \tag{2}
$$

up to the sign convention for $$\Pi$$. The limit $$q\to0$$ depends on the direction of $$q$$: $$\Pi^{\mu\nu}$$ contains the non-local term $$-\Pi_0\,q^\mu q^\nu/q^2$$.
</div>

<div class="proof" markdown="1">
*Proof.* At $$q=0$$ the integrand is $$\mathrm{tr}[\gamma^\mu(-i\not k+m)\gamma^\nu(-i\not k+m)]/(k^2+m^2)^2$$ weighted by $$f$$. Put $$b^2=\lvert k_\perp\rvert^2+m^2$$. The component along $$\hat q$$ has the numerator $$4(b^2-k_\parallel^2)$$, and $$\int(b^2-k_\parallel^2)(k_\parallel^2+b^2)^{-2}dk_\parallel=[k_\parallel/(k_\parallel^2+b^2)]_{-\infty}^{\infty}=0$$ for every $$f$$, so $$\Pi^{\parallel\parallel}=0$$. This reflects the invariance of the regulator under shifts of $$k$$ along $$u$$, which gives $$q_\mu\Pi^{\mu\nu}(q)=0$$ at finite $$q$$ as well. A transverse component equals $$\Pi^{\parallel\parallel}$$ minus the difference between the longitudinal and the transverse component, so it equals minus that difference, since $$\Pi^{\parallel\parallel}=0$$. The difference is $$-8\int\frac{d^4k}{(2\pi)^4}f\,(k_\parallel^2-\lvert k_\perp\rvert^2/3)/(k^2+m^2)^2$$, where $$\lvert k_\perp\rvert^2/3$$ is the average over the three transverse directions. The $$k_\parallel$$ integrals are $$\int k_\parallel^2(k_\parallel^2+b^2)^{-2}dk_\parallel=\pi/(2b)$$ and $$\int(k_\parallel^2+b^2)^{-2}dk_\parallel=\pi/(2b^3)$$. For $$m\to0$$ and $$\lvert k_\perp\rvert=\varrho$$ the combination is $$\pi/(2\varrho)-(\varrho^2/3)\pi/(2\varrho^3)=\pi/(3\varrho)$$, the two terms being in the ratio $$3:1$$ for every $$f$$. Then $$\int d^3k_\perp$$ gives $$4\pi\int\varrho^2f\,\pi/(3\varrho)\,d\varrho=(4\pi^2/3)\int\varrho f\,d\varrho$$. Collecting $$8/(2\pi)^4$$ gives $$2/(3\pi^2)$$ times $$\Lambda^2\int xf\,dx$$, which is (2). $$\square$$
</div>

The regulator preserves the Ward identity, but the transverse part survives as a direction-dependent photon-mass-like term. It is non-analytic at $$q=0$$ and no local counterterm removes it.

### Crossing symmetry of the one-loop amplitude

A further requirement of an effective local theory is crossing symmetry: the amplitude for $$\psi\phi\to\psi\phi$$ in the direct channel and in the crossed channel must be boundary values of one analytic function of the invariants. We label the process $$\psi(p)+\phi(K)\to\psi(p')+\phi(K')$$, with $$p^2=p'^2=m^2$$ and $$K^2=K'^2=M^2$$, and set

$$
P=p+K=p'+K',\qquad P_\times=p-K'=p'-K,\qquad s_P=P^2,\quad s_\times=P_\times^2 .
$$

In Minkowski signature the tree amplitude is the sum of the exchange of a fermion of momentum $$P$$ and of one of momentum $$P_\times$$, $$A_{\rm tree}=g^2\bar\chi(p')(S_P+S_\times)\chi(p)$$ with $$S_q=(\not q+m)/(q^2-m^2)$$, $$S_\times=S_{P_\times}$$ and $$\chi,\bar\chi$$ the external spinors. Since $$P+P_\times=p+p'$$ and $$P-P_\times=K+K'$$, one has $$(P+P_\times)^2-(P-P_\times)^2=4m^2-4M^2$$, that is

$$
P\cdot P_\times=m^2-M^2 .
$$

In the crossed region $$s_\times\ge(m+M)^2$$ the same amplitude describes the elastic process $$\psi(p)+\phi(-K')\to\psi(p')+\phi(-K)$$ with total momentum $$P_\times$$. With cluster frames, the amplitude in the direct region is computed with $$u_P=P/\sqrt{s_P}$$ and in the crossed region with $$u_\times=P_\times/\sqrt{s_\times}$$. The coefficient of Proposition 4(b) is the input. It is the Euclidean self-energy of a line, and it is used here at the off-shell momenta of the internal lines, in Minkowski signature.

<div class="remark" markdown="1">
**Hypothesis (Loc)** (locality of the frame dependence). For momenta $$q$$, $$m$$, $$M$$ much smaller than $$\Lambda$$, the part of the one-loop self-energy of a fermion line that depends on the frame $$u$$ is the insertion

$$
\Sigma_u(q)=B\,\frac{(q\cdot u)\not u}{u^2},\qquad B=\frac{g^2}{48\pi^2},
$$

into the inverse propagator $$\not q-m-\Sigma_u$$, up to corrections suppressed by $$q^2/\Lambda^2$$. The expression is the part of $$\not q$$ along $$u$$, has the same form in both signatures, and is continued from Euclidean to Minkowski momenta as a local operator. The frame dependence of the vertex and box subgraphs of the $$\psi\phi\to\psi\phi$$ amplitude is $$O(q/\Lambda)$$. Frame-dependent corrections on the external legs are not included in $$A_{\rm d}$$ below.
</div>

Proposition 4(b) fixes $$B=a_\perp-a_\parallel$$ at $$\lvert p\rvert<M$$: removing the frame-independent factor $$1+a_\perp$$ from $$\Gamma/i=(1+a_\perp)\not p-(a_\perp-a_\parallel)(p\cdot u)\not u$$ leaves $$\not p-B(p\cdot u)\not u$$. The finite-momentum values above support the extension to $$M\ll\lvert q\rvert\ll\Lambda$$ in the Euclidean model. The angular dependence is produced by loop momenta of order $$\Lambda$$ (it comes from the $$-\ln\sin\chi$$ term of the radial integral on the linked page), so for external momenta well below $$\Lambda$$ it is carried by a local operator. This is the content of (Loc), and beyond the Euclidean check it is not proved. The statement about vertex and box subgraphs rests on power counting. Their frame-dependent parts are ultraviolet convergent, and the logarithmic part of the vertex is a frame-independent scalar because $$\not u\not u=1$$. Those graphs were not computed.

<div class="remark" markdown="1">
**Derivation 1** (crossing defect at one loop, under (Loc)). Under (Loc) the frame-dependent part of the one-loop amplitude is the sum of the two tree diagrams with $$S_q\to S_q\Sigma_uS_q$$ on the internal line. Let $$A_{\rm d}$$ be the difference between the amplitude computed with the frame $$u_P$$ in both diagrams and the amplitude computed with $$u_\times$$ in both diagrams, at the same momenta. Then

$$
A_{\rm d}=g^2B\ \bar\chi(p')\Big\{S_P\Big[\not P-\frac{m^2-M^2}{s_\times}\not P_\times\Big]S_P+S_\times\Big[\frac{m^2-M^2}{s_P}\not P-\not P_\times\Big]S_\times\Big\}\chi(p). \tag{3}
$$

It is independent of $$\Lambda$$, is non-zero in general, and has a pole at $$s_P=0$$ with residue proportional to $$m^2-M^2$$ (and one at $$s_\times=0$$).
</div>

To derive (3), insert $$\Sigma_u$$ on each line in each frame. On the line of momentum $$P$$, the frame $$u_P$$ gives $$B\,(P\cdot u_P)\not u_P=B\not P$$, a covariant renormalisation of the line, and the frame $$u_\times$$ gives $$B\,(P\cdot P_\times)\not P_\times/s_\times=B\,(m^2-M^2)\not P_\times/s_\times$$, which is not proportional to $$\not P$$. On the line of momentum $$P_\times$$ the roles are exchanged: $$B\,(m^2-M^2)\not P/s_P$$ in the frame $$u_P$$ and $$B\not P_\times$$ in the frame $$u_\times$$. Subtracting on each line the insertion in the frame $$u_\times$$ from the insertion in the frame $$u_P$$, and attaching $$S_q$$ on both sides and the factor $$g^2$$ of the two Yukawa vertices, gives (3). If $$A_{\rm d}\ne0$$, the amplitude with the frame of its own channel in each region is not the boundary value of one analytic function, because the continuation of the direct-channel amplitude to the crossed region is the amplitude with $$u_P$$, not with $$u_\times$$.

To see the size of $$A_{\rm d}$$ we take $$m=1$$, $$M=0.5$$ and evaluate (3) in the crossed physical region (centre-of-mass frame of $$P_\times$$, scattering angle $$\theta$$ between $$p$$ and $$p'$$). We form $$R=2\,\mathrm{Re}\sum_{\rm spins}A_{\rm d}A_{\rm tree}^*\big/\sum_{\rm spins}\lvert A_{\rm tree}\rvert^2$$, the relative change of the spin-summed squared tree amplitude due to $$A_{\rm d}$$. In units of $$g^2/16\pi^2$$ (so $$B=\tfrac13$$):

| $$\sqrt{s_\times}$$ | $$1.55$$ | $$2$$ | $$3$$ | $$5$$ | $$10$$ |
|---|---|---|---|---|---|
| $$\cos\theta$$ | $$0.3$$ | $$0.3$$ | $$0.3$$ | $$-0.5$$ | $$-0.5$$ |
| $$R$$ | $$+12.87$$ | $$-3.26$$ | $$-0.796$$ | $$+0.087$$ | $$+0.32$$ |

In the centre-of-mass frame of $$P_\times$$, with $$E=(s_\times+m^2-M^2)/(2\sqrt{s_\times})$$ and $$k^2=E^2-m^2$$, one finds $$s_P=2M^2-s_\times+2(E^2-k^2\cos\theta)$$. At $$\sqrt{s_\times}=2.2$$ and $$\cos\theta=-0.9054$$ this vanishes, so $$s_P=0$$ lies inside the physical region, and there $$s_PR=-0.154$$ on both sides, so $$R$$ diverges like $$1/s_P$$. For $$m=M$$ the residue vanishes by (3). The defect is removed by a counterterm $$-B\,(q\cdot u_C)\not u_C/u_C^2$$ on every internal line, with $$u_C$$ the frame of the regulator in the amplitude considered ($$u_P$$ in the direct region, $$u_\times$$ in the crossed region). This is one dimension-four coefficient per order, the tuning of [[CPSUV]](#ref-cpsuv).

Graphs with zero total momentum, tadpoles and vacuum bubbles, have no cluster frame, and the Lorentzian orbit of frames is non-compact, so no normalisable average over it exists (see the page on [frame leakage](/quantum-gravity/lorentz-violation-leakage/)). The computation above does not extend to those graphs.

## 5. Open: averaged-law positivity, signed weights and the locality hypothesis

Whether the unsymmetrised ensemble is reflection positive in the averaged sense, with a mass (Lemma 4) or on a graph without degree-one vertices, has neither a proof nor a violation. The numerical statements of section 3 are two-dimensional, and the four-dimensional version of the Euclidean operator has not been tested. Other kernels with the cross-plane property are not examined. Gauge fields with non-trivial cross-plane links need a reflection-symmetric measure on the links, and the Gabriel result is for the free operator.

For link operators, the cases outside Lemma 1 and Theorem 1 are signed weights and a Wilson term whose rows do not sum to zero. Whether signed weights give a covariant infrared Weyl term is not settled, and neither is whether such a Wilson term avoids exact zero modes.

For cluster frames, the vertex and box graphs and the external-leg corrections are not computed, so (Loc) rests on the one-loop self-energy and its finite-momentum check. Gauge theories, two loops, per-particle frames (each particle with its own axis on one shared point set) and frames supplied by the state of an exactly Lorentz-invariant theory are not treated. Cluster separability with particle production for infinitely many degrees of freedom is not established.

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
16. <a id="ref-bd"></a>Benincasa, Dowker, "The scalar curvature of a causal set," [arXiv:1001.2725](https://arxiv.org/abs/1001.2725), Phys. Rev. Lett. 104 (2010) 181301.
