---
title: Frames, Wilson terms and reflection positivity for fermions on random points
date: 2026-10-29
category: matter
order: 3
description: On a four-dimensional Poisson sprinkling, a link operator with a range cutoff along a frame and a row-sum Wilson term has exact zero modes at a positive fraction of the points for every finite range, no translation-invariant random point set is mirror symmetric in every realisation, and a frame supplied by a cluster of particles leaves a crossing defect at one loop in a Yukawa model under a stated locality hypothesis.
status: Two theorems proved (exact zero modes of range-cut link operators in four dimensions; incompatibility of translation invariance with mirror symmetry in every realisation, any dimension); lemmas and propositions proved; reflection positivity of the Gabriel-graph Wilson term conditional on a cited criterion and tested numerically in two dimensions; one-loop cluster-frame statements derived under a locality hypothesis, with numerical checks.
excludes: Row-sum Wilson terms on range-cut four-dimensional link operators as a way to remove exact zero modes; non-negative link weights as an infrared Weyl operator; reflection positivity in every realisation for translation-invariant point processes; the Gaussian-kernel Wilson-Dirac operator and its overlap as reflection positive on mirror-symmetric sets (numerical, two dimensions); cluster-frame regulators in the Yukawa model at one loop without a tuned frame counterterm (under the locality hypothesis of section 4).
---

What does it cost to let a preferred frame enter the definition of a fermion on a Poisson sprinkling, and what does the lack of one cost in the passage to a Lorentzian theory? Three answers are given, each with its own hypotheses. In four dimensions, a link operator whose links are limited to a time extent $$t_{\max}$$ along the frame, with a Wilson term whose diagonal is the row sum, has exact zero modes at a fraction $$e^{-2\pi (t_{\max}/\ell)^4/3}$$ of the points ($$\ell$$ the mean spacing); the fraction is positive for every finite $$t_{\max}$$ and equals five per cent at $$t_{\max}=1.094\,\ell$$ (Theorem 1). For the Euclidean overlap operator, reflection positivity needs each realisation to be mirror symmetric, and in any dimension no point process invariant under translations along a direction is mirror symmetric in every realisation for the reflection in the orthogonal plane (Theorem 2); on mirror-symmetric sets the Wilson term on the Gabriel graph passes a one-particle positivity test, while the Gaussian-kernel Wilson term of [the page on frame-free Weyl operators](/quantum-gravity/frame-free-chirality/) fails it. When the frame is the velocity of a cluster of particles, the one-loop amplitude of a Yukawa model is not crossing symmetric, with a defect fixed by one dimension-four coefficient, provided the frame dependence of an internal line is local in the sense of hypothesis (L) of section 4.

The three parts are independent. Section 2 concerns link operators in four dimensions, section 3 reflection positivity (the statements hold in every dimension where they are proved, and the numerical checks are two-dimensional), and section 4 a four-dimensional Yukawa model at one loop. Section 1 collects what sections 2 and 3 use.

Nielsen and Ninomiya showed that a local, translation-invariant lattice fermion with a conserved chiral charge has equally many left and right species [[NN]](#ref-nn), which is why a Wilson term or an overlap construction is needed. Random-lattice fermions on Voronoi or Delaunay geometries [[CFL]](#ref-cfl), [[FLR]](#ref-flr) are Euclidean and have no light cone. Bombelli, Henson and Sorkin proved that a sprinkling has no preferred direction and no finite-valency graph [[BHS]](#ref-bhs); links of a sprinkling therefore have unbounded extent, and an operator built from them needs a cutoff, which here is a time extent measured along a frame. The Osterwalder–Schrader theorem turns reflection-positive Euclidean correlation functions into a Lorentzian theory with positive energy [[OS1]](#ref-os1), [[OS2]](#ref-os2); reflection positivity is known on a lattice for the Wilson action [[MP]](#ref-mp) and for the free overlap operator with the nearest-neighbour kernel [[KU1]](#ref-ku1), [[KU2]](#ref-ku2). That a frame in a regulator feeds order-one Lorentz violation into dimension-four operators is the argument of Collins, Perez, Sudarsky, Urrutia and Vucetich [[CPSUV]](#ref-cpsuv); the one-loop coefficient is computed on [the page on frame leakage](/quantum-gravity/lorentz-violation-leakage/) and used below. Added here are the exact zero-mode count of Theorem 1, the non-existence statement of Theorem 2 together with the mirror property of the Gabriel graph, and a crossing defect for frames supplied by clusters in the sense of Bakamjian and Thomas [[BT]](#ref-bt).

## 1. Doubling and reflection positivity

The naive lattice derivative of spacing $$a$$ has symbol $$i\gamma\cdot\sin(pa)/a$$, which vanishes at the $$2^d$$ corners of the Brillouin zone and so describes $$2^d$$ species. A Wilson term $$r\,(1-\cos pa)/a$$ gives the extra species a mass $$2r/a$$ at the price of exact chiral symmetry, and the overlap construction of the page linked above restores a Ginsparg–Wilson form of chiral symmetry. A random point set has no Brillouin zone and the symbol of one realisation is not defined; one works with the plane-wave-averaged symbol or with the spectrum.

Let $$\mathcal H$$ be the hyperplane $$t=0$$ in Euclidean $$\mathbb R^d$$ and $$\theta(t,x)=(-t,x)$$ the reflection in it. The antilinear map $$\Theta$$ sends a field $$\psi(y)$$ at a point $$y$$ with $$t>0$$ to $$\bar\psi(\theta y)\gamma_t$$, reverses the order of products and conjugates numbers. A Euclidean fermion theory is reflection positive (RP) if $$\langle\Theta(\mathcal A)\,\mathcal A\rangle\ge0$$ for every polynomial $$\mathcal A$$ in the fields at points with $$t>0$$. By the Osterwalder–Schrader reconstruction theorem, Euclidean-invariant RP correlation functions with suitable growth bounds are the Schwinger functions of a Wightman theory with a unitary Poincaré representation, a vacuum and a Hamiltonian $$P^0\ge0$$, Euclidean time translation acting as $$e^{-tP^0}$$ [[OS1]](#ref-os1), [[OS2]](#ref-os2).

For a free fermion with propagator $$S=D^{-1}$$, taking $$\mathcal A$$ linear in the fields gives a necessary condition: the matrix

$$
\mathcal K_{yx}=S(y,\theta x)\,\gamma_t,\qquad x,y\ \text{in the upper half-space}, \tag{1}
$$

is positive semidefinite. We call this the one-particle test. For Gaussian measures it is also sufficient, but only the necessary direction is used here, so that a negative eigenvalue of $$\mathcal K$$ is a violation of RP. The index order and the sign convention of the fermionic pairing in (1) are not derived; they are fixed by the requirement that $$\mathcal K\succeq0$$ for the nearest-neighbour Wilson operator with $$0<r\le1$$ and the reflection through the midpoints of links, the case proved in [[MP]](#ref-mp) and [[KU1]](#ref-ku1), and the numerical results of section 3 reproduce that case.

On a random point set there are two notions. RP in every realisation requires that the lower points are the mirror images of the upper points, $$\theta\mathcal P=\mathcal P$$, so that $$\Theta$$ acts on the fields of that realisation. Averaged-law RP asks that the disorder-averaged Schwinger functions $$\mathbb E\langle\cdot\rangle_{\mathcal P}$$ be reflection positive although no single realisation is mirror symmetric; averaging does not preserve RP automatically, because $$\langle\Theta(\mathcal A)\mathcal A\rangle_{\mathcal P}$$ is not defined on a realisation without a mirror image.

## 2. Link operators and a frame

A four-dimensional causal set has no derivative; the nearest analogue is an operator on links. Let $$\mathcal P\subset\mathbb R^{1,3}$$ be a finite set (a sprinkling of density $$\rho=\ell^{-4}$$). A link is a pair $$y\prec x$$ of related points with no point between them, $$\Delta=x-y$$ is its displacement and $$w\ge0$$ its weight. We use the signature $$(+,-,-,-)$$ and write $$\bar\sigma\cdot\Delta=\Delta^0+\vec\sigma\cdot\vec\Delta$$ and $$p\cdot\Delta=p^0\Delta^0-\vec p\cdot\vec\Delta$$, so that the covariant Weyl symbol is $$\bar\sigma\cdot p=p^0+\vec\sigma\cdot\vec p$$. The bare link operator has the entries $$Q_{yx}=iw\,\bar\sigma\cdot\Delta$$ and $$Q_{xy}=Q_{yx}^\dagger$$, so $$Q$$ is Hermitian, and its plane-wave-averaged symbol is

$$
F(p)=F^0(p)+\vec F(p)\cdot\vec\sigma=-2\sum_{\rm links}w\,\bar\sigma\cdot\Delta\,\sin(p\cdot\Delta)\quad\text{per element,}
$$

the sum running over the links incident on an element in the interior of the region. Links of a sprinkling have unbounded extent, so a range cutoff is needed, and we measure it along a future-timelike unit vector $$u$$, the frame. Three questions arise. Can non-negative weights give a covariant infrared Weyl term? What does the range cutoff do to the spectrum? Can a covariant Wilson term remove the extra low-lying structure without introducing a frame-dependent coefficient?

<div class="lemma" markdown="1">
**Lemma 1** (non-negative weights; any dimension $$d$$). Let $$\Delta_1,\dots,\Delta_N$$ be non-zero future-causal vectors in $$\mathbb R^{1,d-1}$$ and $$w_n\ge0$$ not all zero. The matrix $$M^{\mu\nu}=\sum_nw_n\Delta_n^\mu\Delta_n^\nu$$ is positive semidefinite and non-zero, hence not proportional to the Minkowski metric $$\eta$$. Suppose the links are distributed isotropically about the time axis of some frame, so that $$\sum_nw_n\Delta_n^0\vec\Delta_n=0$$ and $$\sum_nw_n\Delta_n^i\Delta_n^j=b'\delta^{ij}$$. Then the leading term of $$F$$ is

$$
-2\,(a\,p^0-b\,\vec\sigma\cdot\vec p),\qquad a=\sum_nw_n(\Delta_n^0)^2,\quad b=\sum_nw_n\lvert\vec\Delta_n\rvert^2/(d-1)\ \ (=b'),
$$

which has the helicity opposite to $$\bar\sigma\cdot p$$ (or no spatial term, if $$b=0$$) and speed $$b/a\le1/(d-1)$$, with equality only if every $$\Delta_n$$ is null.
</div>

<div class="proof" markdown="1">
*Proof.* For every vector $$v$$, $$v_\mu M^{\mu\nu}v_\nu=\sum_nw_n(v\cdot\Delta_n)^2\ge0$$, so $$M$$ is positive semidefinite, whereas $$\eta$$ is indefinite. Hence $$M=c\,\eta$$ forces $$c=0$$ and $$M=0$$, which is excluded because $$M^{00}=\sum_nw_n(\Delta_n^0)^2>0$$ (a non-zero causal vector has $$\Delta^0\ne0$$). Expanding $$\sin(p\cdot\Delta)\simeq p\cdot\Delta$$ gives $$F\simeq-2\sum_nw_n(\Delta_n^0+\vec\sigma\cdot\vec\Delta_n)(p^0\Delta_n^0-\vec p\cdot\vec\Delta_n)$$. By the isotropy assumption the cross terms vanish, the time part is $$a\,p^0$$, and the spatial part is $$-b'\,\vec\sigma\cdot\vec p$$, with $$(d-1)b'=\sum_nw_n\lvert\vec\Delta_n\rvert^2$$. Causality gives $$\lvert\vec\Delta_n\rvert\le\Delta_n^0$$, so $$(d-1)b\le a$$, with equality only when all $$\lvert\vec\Delta_n\rvert=\Delta_n^0$$. $$\square$$
</div>

A covariant infrared term needs $$M\propto\eta$$, that is $$M^{00}$$ and the spatial diagonal $$M^{ii}$$ of opposite signs, which a sum of squares cannot supply. Signed weights, such as the layer weights of Benincasa and Dowker [[BD]](#ref-bd), can, and the lemma says nothing about them. For links among interior points of a sprinkled four-dimensional causal diamond we found $$b/a$$ between $$0.19$$ and $$0.22$$ at $$N=1000$$ and $$N=2000$$, below the bound $$1/3$$; the mean time extent of a link was $$0.13$$ to $$0.15$$ of the diamond height, so the links are not short compared with the region. The four-dimensional case is also treated on [the page on order-only fermions](/quantum-gravity/order-only-fermions/).

<div class="proposition" markdown="1">
**Proposition 1** (no sign change of $$F^0$$ below the range limit). Let $$w_n\ge0$$ vanish unless $$0<t_n:=u\cdot\Delta_n\le t_{\max}$$. Then $$F^0=-2\sum_nw_nt_n\sin(p_0t_n)<0$$ for $$0<p_0<\pi/t_{\max}$$ on the time axis $$\vec p=0$$.
</div>

Every term has $$w_nt_n>0$$ and $$\sin(p_0t_n)>0$$ in that interval. For $$t_{\max}\le\ell$$ the interval contains the whole resolved band $$\lvert p\rvert<\pi/\ell$$, so a zero of the time-axis symbol below the band edge, as in Lemma 2 below, cannot come from $$F^0$$ alone. 

### Exact zero modes of range-cut operators

Let $$t_{\max}>0$$ and call two related points $$x\ne y$$ a *cut pair* if $$\lvert u\cdot(x-y)\rvert\le t_{\max}$$. A cut link is a link that is a cut pair.

<div class="theorem" markdown="1">
**Theorem 1** (isolation floor; four dimensions). Let $$\mathcal P\subset\mathbb R^{1,3}$$ be finite, $$u$$ a future-timelike unit vector and $$t_{\max}>0$$. Let $$Q,W$$ be matrices on $$\mathbb C^{\mathcal P}\otimes\mathbb C^2$$ such that

(a) the off-diagonal blocks $$Q_{xy},W_{xy}$$ vanish unless $$x,y$$ are a cut pair (this includes operators supported on cut links);

(b) $$Q_{xx}=0$$ and $$W_{xx}=-\sum_{y\ne x}W_{xy}$$ (the row sum of $$W$$ is zero).

Then every point $$x$$ with no cut pair satisfies $$Q(\delta_x\otimes e)=W(\delta_x\otimes e)=0$$ for all $$e\in\mathbb C^2$$, so $$\dim\ker(\alpha Q+\beta W)\ge2\,\#\{x\text{ without a cut pair}\}$$ for all $$\alpha,\beta$$. A point has a cut pair if and only if it has a cut link. If $$\mathcal P$$ is a Poisson process of density $$\rho=\ell^{-4}$$ and the double cone $$C_x=\{y:\ y\text{ related to }x,\ \lvert u\cdot(x-y)\rvert\le t_{\max}\}$$ of a point $$x$$ lies in the region, then $$x$$ has no cut pair with probability $$e^{-2\pi\rho\,t_{\max}^4/3}$$. This is below $$5\%$$ if and only if $$t_{\max}\ge(3\ln20/2\pi)^{1/4}\ell=1.0936\,\ell$$.
</div>

<div class="proof" markdown="1">
*Proof.* If $$x$$ has no cut pair, its row and column vanish in $$Q$$ and, off the diagonal, in $$W$$; (b) makes the diagonal blocks zero too, so $$\delta_x\otimes e$$ is in both kernels. For the equivalence, suppose $$x$$ has a related point $$y$$ in its future with $$u\cdot(y-x)\le t_{\max}$$. The finite set of such $$y$$ has a minimal element $$z$$ for the causal order. If $$w$$ lay strictly between $$x$$ and $$z$$, then $$0<u\cdot(w-x)<u\cdot(z-x)\le t_{\max}$$, because $$z-w$$ is future causal and $$u$$ is future timelike, so $$w$$ would be a smaller element of the set; hence $$z$$ is linked to $$x$$, and the pair is a cut link. The past is the same. In the rest frame of $$u$$ the future part of the double cone is $$\{0<s\le t_{\max},\ \lvert\vec x\rvert\le s\}$$, of volume $$\int_0^{t_{\max}}\tfrac{4\pi}3s^3ds=\pi t_{\max}^4/3$$, and the double cone has twice that volume. By the Mecke formula the Poisson probability that a given point has no other point in a set of volume $$\mathrm{vol}$$ is $$e^{-\rho\,\mathrm{vol}}$$. Solving $$e^{-2\pi\rho t^4/3}\le\tfrac1{20}$$ gives the stated bound. $$\square$$
</div>

The hypotheses hold for every range-cut link operator with a Wilson term whose diagonal is its row sum, for any weights and any $$r$$, so the fraction depends on $$t_{\max}/\ell$$ alone. It is positive for every finite $$t_{\max}$$: a sprinkling of a region of fixed shape contains such points once it has enough points, and no value of $$t_{\max}/\ell$$ removes them; $$1.094\,\ell$$ is only the range at which the fraction is five per cent, and at $$1.5\,\ell$$ it is $$2.5\times10^{-5}$$. Among $$N$$ points in the bulk the expected number is of order one when $$t_{\max}^4\simeq(3/2\pi)\,\ell^4\ln N$$. The theorem concerns exact zero modes only. It does not bound near-zero modes, and it does not apply to a Wilson diagonal that is not the row sum.

On sprinklings of $$N=3000$$ points in a four-dimensional causal diamond (30 sprinklings, counting the points whose double cone lies in the diamond), the fractions without a cut pair at $$t_{\max}/\ell=1,\ 1.094,\ 1.25$$ were @@T1A@@, against $$0.123,\ 0.050,\ 0.006$$ from the formula. The set of points without a cut pair coincided with the set without a cut link in every sprinkling, as the theorem states.

### A covariant Wilson term on the time axis

A Wilson term couples the two chiral blocks of a Dirac operator, so a Weyl operator with symbol $$F$$ is paired with a second block of opposite chirality, whose spatial part has the opposite sign. Let $$F^0,\vec F$$ be the components of the averaged symbol on the time axis and let

$$
W(p)=2r\ell\sum_{\rm links}w\,\bigl(1-\cos(p\cdot\Delta)\bigr)\quad\text{per element}
$$

be the covariant Wilson term of coefficient $$r$$; the factor $$2$$ is the one in $$F$$. Together they give the Hermitian block matrix

$$
\begin{pmatrix}F^0+\vec F\cdot\vec\sigma&-W\\-W&F^0-\vec F\cdot\vec\sigma\end{pmatrix},
$$

which for $$F^0=p_0$$, $$\vec F=\vec k$$ and $$W=m$$ is the Dirac operator of mass $$m$$ in the chiral basis. Its eigenvalues are $$F^0\pm\sqrt{\lvert\vec F\rvert^2+W^2}$$, each doubly degenerate.

<div class="lemma" markdown="1">
**Lemma 2** (time-axis zero). Let $$F^0,\ \vec F,\ W$$ be real continuous functions of $$p_0\in[0,p_s]$$. The block matrix is singular exactly where $$D_1=(F^0)^2-\lvert\vec F\rvert^2-W^2$$ vanishes. If $$D_1>0$$ near $$p_0=0^+$$ and $$D_1(p_s)<0$$, then $$D_1$$ has a zero in $$(0,p_s)$$.
</div>

The lemma is the intermediate value theorem; the content is in the sign of $$D_1$$ at the two ends. In the continuum limit $$D_1=p_0^2-k^2-m^2$$ is the mass-shell condition. If $$W$$ depends on the spatial momentum only, as in a Hamiltonian lattice, $$W(0)=0$$ and the shell at $$\vec k=0$$ is $$p_0=0$$. A covariant Wilson term depends on $$p_0$$ as well. For small $$p_0$$ and isotropic links, with $$a=\sum w(\Delta^0)^2$$ per element as in Lemma 1, $$F^0\simeq-2a\,p_0$$, $$\vec F\simeq0$$ and $$W\simeq r\ell\,a\,p_0^2$$, so that $$D_1\simeq a^2p_0^2\,(4-r^2\ell^2p_0^2)$$ and $$D_1$$ has a second zero near $$p_0=2/(r\ell)$$, independently of the link distribution at this order. The point is a pole of the lifted operator at zero spatial momentum that is not the physical particle. It lies below the band edge $$\pi/\ell$$ when $$r>2/\pi$$. Since $$p_0t_{\max}$$ is of order one there, the expansion is an estimate, and the zero was located numerically. For the link set with $$t_{\max}=\ell$$ on $$N=2000$$ points of a four-dimensional diamond (interior elements: the central half-height diamond, three sprinklings), $$D_1$$ is positive at small $$p_0$$ and its first zero is at

| $$r$$ | $$0.5$$ | $$1$$ | $$2$$ |
|---|---|---|---|
| estimate $$p_0\ell=2/r$$ | $$4.0$$ | $$2.0$$ | $$1.0$$ |
| first zero of $$D_1$$, $$p_0\ell$$ | $$2.43$$ | $$1.65$$ | $$0.94$$ |

against a band edge $$p_0\ell=\pi$$; the three sprinklings agree to better than $$0.01$$ in $$p_0\ell$$. By Proposition 1, $$F^0$$ has no sign change below $$\pi/\ell$$, so each zero is where $$W$$ overtakes $$\lvert F^0\rvert$$. The lemma and the table concern the plane-wave-averaged symbol; the exact spectrum of the lifted operator on a sprinkling was not examined, so they do not decide whether that operator has low modes.

## 3. The Euclidean overlap and reflection positivity

Example 1 of the page on frame-free Weyl operators is a Wilson–Dirac operator with a Gaussian kernel on Poisson points of a torus, whose overlap operator has index $$-Q$$ in a flux $$Q$$. Its distances use the Euclidean metric of a chosen axis, and the law of the Poisson process is invariant under every volume-preserving affine map, so the law of the operator is the same for every axis. The question here is whether its Euclidean correlation functions are reflection positive, so that the Osterwalder–Schrader theorem produces a Lorentzian theory in which the axis is only the direction of the Wick rotation.

### Mirror symmetry and translation invariance

<div class="theorem" markdown="1">
**Theorem 2** (mirror symmetry excludes translation invariance). Let $$\nu$$ be the law of a random locally finite subset $$\mathcal P\subset\mathbb R^d$$ ($$d\ge1$$), $$n$$ a unit vector and $$\theta$$ the reflection in the hyperplane through the origin orthogonal to $$n$$. Assume

(a) $$\nu(\theta\mathcal P=\mathcal P)=1$$;

(b) $$\nu$$ is invariant under the translations $$x\mapsto x+sn$$ for all real $$s$$.

Then $$\nu(\mathcal P=\emptyset)=1$$.
</div>

<div class="proof" markdown="1">
*Proof.* A reflection-symmetric set stays reflection symmetric when the plane is translated, and two reflections in parallel planes compose to a translation; a locally finite set cannot be invariant under a dense set of translations unless it is empty. In detail, let $$T_s(x)=x+sn$$. The event $$E=\{\theta\mathcal P=\mathcal P\}$$ has probability one, and by (b) so has $$\{T_s\mathcal P\in E\}=\{\theta_s\mathcal P=\mathcal P\}$$, where $$\theta_s=T_s^{-1}\theta T_s$$ acts as $$x\mapsto\theta x-2sn$$ (because $$\theta n=-n$$). On the intersection of these events over a countable dense set $$S\subset\mathbb R$$, which has probability one, $$\mathcal P$$ is invariant under $$\theta$$ and all $$\theta_s$$, hence under $$\theta\theta_s:x\mapsto x+2sn$$, $$s\in S$$. If $$p\in\mathcal P$$, then $$p+2sn\in\mathcal P$$ for all $$s\in S$$, and these points accumulate, contradicting local finiteness. $$\square$$
</div>

No translation-invariant ensemble is therefore mirror symmetric in every realisation. The Poisson process is translation invariant, and almost surely no point of $$\mathcal P$$ has its mirror image in $$\mathcal P$$, because the graph of $$\theta$$ is a null set in $$\mathbb R^d\times\mathbb R^d$$; so $$\theta\mathcal P\cap\mathcal P=\emptyset$$ and RP in every realisation is not defined for it. The symmetrised process (a Poisson process in the upper half-space together with its mirror images) is mirror symmetric and, by the theorem, not translation invariant: its pair correlation has a singular part supported on mirror pairs. The theorem uses translation invariance along $$n$$ only, a weaker condition than Euclidean invariance. It leaves open lattices, which are invariant under a discrete group, and averaged-law RP of the unsymmetrised ensemble.

### The Gaussian kernel

The standard criterion for RP of a quadratic fermion action is the transfer-matrix argument of Osterwalder and Seiler for lattice actions [[OSe]](#ref-ose), in a form given by Jaffe and Janssens [[JJ]](#ref-jj): if the action is covariant under $$\theta$$ and the cross-plane coupling matrix $$C_{xy}=\gamma_tD(\theta x,y)$$, $$x,y$$ in the upper half-space, is positive semidefinite, the free fermion is RP. We cite it and have not re-derived it for point sets without translation symmetry. Failure of the criterion does not by itself prove a violation; the test that does is the one-particle kernel (1).

For a point $$x$$ at distance $$\delta$$ above the plane, with mirror image $$\theta x$$, the displacement from $$\theta x$$ to $$x$$ is $$2\delta\,e_t$$. The Gaussian Wilson–Dirac operator has the entries $$D_W(x,y)=w\,[\gamma\cdot(y-x)/Z_1-r/Z_0]$$ for $$x\ne y$$, with $$w=e^{-\lvert y-x\rvert^2/2s^2}$$, $$Z_1=2\pi s^4$$ and $$Z_0=2\pi s^2$$, and the diagonal $$(r/Z_0)\sum_yw$$. Its block between $$\theta x$$ and $$x$$ is $$D_W(\theta x,x)=w\,[(2\delta/Z_1)\gamma_t-r/Z_0]$$ with $$w=e^{-2\delta^2/s^2}$$.

<div class="lemma" markdown="1">
**Lemma 3** (cross-plane block of the Gaussian kernel). The block $$\gamma_tD_W(\theta x,x)=w\,[2\delta/Z_1-(r/Z_0)\gamma_t]$$ has eigenvalues $$w(2\delta/Z_1\mp r/Z_0)$$. With $$Z_1/Z_0=s^2$$ it is indefinite exactly when $$\delta<rs^2/2$$.
</div>

This follows from $$\gamma_t^2=1$$. For $$s=1.5$$ and $$r=1$$ the threshold is $$\delta<1.125$$, which is met by every mirror-symmetric configuration with a point that close to the plane and by every lattice of spacing below $$2.25$$. For the nearest-neighbour Wilson operator the same block is $$\tfrac12(1-r\gamma_t)$$, positive semidefinite exactly for $$\lvert r\rvert\le1$$, the known range [[MP]](#ref-mp). Raising $$r$$ in the Gaussian kernel widens the indefinite strip.

The one-particle test (1) gives the following on a $$12\times12$$ square lattice with the plane between two rows (open boundaries, $$s=1.5$$, overlap built with $$m_0$$ and a mass $$\mu$$ added). The entries are $$\lambda_{\min}/\lambda_{\max}$$ of the Hermitian part of $$\mathcal K$$; values of order $$10^{-12}$$ or smaller in modulus are zero to rounding.

| operator | $$\lambda_{\min}/\lambda_{\max}$$ |
|---|---|
| nearest-neighbour Wilson, $$r=1$$, $$0.6$$ | $$-4\times10^{-16}$$, $$-3\times10^{-16}$$ |
| nearest-neighbour Wilson, $$r=1.4$$ | $$-3.0\times10^{-2}$$ |
| nearest-neighbour overlap, $$m_0=1$$ and $$0.5$$, $$\mu=0.1$$ and $$0.5$$ | at most $$5\times10^{-12}$$ in modulus |
| Gaussian Wilson, $$r=1$$ | $$-3.5\times10^{-2}$$ |
| Gaussian Wilson, $$r=4$$ | $$-0.29$$ |
| Gaussian overlap, $$r=1$$, $$m_0=0.5$$, $$\mu=0.1$$ and $$0.5$$ | $$-4.7\times10^{-2}$$, $$-0.12$$ |

The nearest-neighbour entries reproduce the known positive cases and the negative value at $$r=1.4$$, which calibrates (1). The Gaussian kernel violates the test on a regular lattice, so the violation is not an effect of disorder. On twenty mirror-symmetric Poisson configurations of about $$144$$ points ($$\rho=1$$, free boundary) the Gaussian Wilson operator with $$r=1$$ violated positivity in @@GP@@, and the Gaussian overlap with $$m_0=0.5$$ and $$\mu=0.1$$ in @@GOP@@.

### The Gabriel graph on mirror-symmetric sets

The *Gabriel graph* of a finite point set joins $$a,b$$ if no other point lies in the closed ball with diameter $$[a,b]$$, that is if $$(a-w)\cdot(b-w)>0$$ for all other $$w$$. For a Poisson process this is almost surely the same as the version with open balls.

<div class="proposition" markdown="1">
**Proposition 2** (mirror lemma; any dimension). Let $$\mathcal P\subset\mathbb R^d$$ be finite, with $$\theta\mathcal P=\mathcal P$$ and no point on the hyperplane. Every edge of the Gabriel graph of $$\mathcal P$$ with endpoints on opposite sides of the hyperplane joins a point $$x$$ to its mirror image $$\theta x$$.
</div>

<div class="proof" markdown="1">
*Proof.* Use coordinates $$(\tau,\xi)$$ with the hyperplane at $$\tau=0$$. Take an edge between $$\theta x$$ and $$v$$ with $$x,v$$ in the upper half-space and $$x\ne v$$; we show that it is not Gabriel. A point $$w$$ lies in the closed ball with diameter $$[a,b]$$ exactly when $$(a-w)\cdot(b-w)\le0$$. For $$w=x$$ one finds $$(\theta x-x)\cdot(v-x)=-2\tau_x(\tau_v-\tau_x)$$, and for $$w=\theta v$$ one finds $$(\theta x-\theta v)\cdot(v-\theta v)=2\tau_v(\tau_v-\tau_x)$$. If $$\tau_v\ge\tau_x$$ the first is $$\le0$$, and if $$\tau_v\le\tau_x$$ the second is $$\le0$$. In either case a point of $$\mathcal P$$ lies in the ball, and it is not an endpoint: $$x\ne\theta x,v$$ and $$\theta v\neq\theta x,v$$. $$\square$$
</div>

Let $$D_G(x,y)=\tfrac12(\gamma\cdot e_{xy}-r)$$ on the edges of the graph, with $$e_{xy}$$ the unit vector from $$x$$ to $$y$$, and $$D_G(x,x)=\tfrac r2\deg x$$ (no gauge field). By the proposition the cross-plane coupling matrix $$C_{xy}=\gamma_tD_G(\theta x,y)$$ is diagonal, with blocks $$0$$ or $$\tfrac12(1-r\gamma_t)$$, because $$e_{\theta x,x}=n$$ and $$\gamma\cdot n=\gamma_t$$; it is positive semidefinite for $$\lvert r\rvert\le1$$. The operator is covariant under reflections, since $$\gamma_t(\gamma\cdot e)\gamma_t=-\gamma\cdot\theta e$$. With the criterion above, the free Gabriel Wilson operator is therefore RP on every mirror-symmetric finite configuration without gauge field. The nearest-neighbour Wilson operator on the square lattice is the special case in which the Gabriel graph is the lattice graph.

In @@GE@@ mirror-symmetric Poisson configurations in the plane (30 to 120 upper points) there were @@GEN@@ plane-crossing Gabriel edges, and all were mirror pairs. For the free Gabriel operator with $$r=1$$ on twenty configurations of $$144$$ points the kernel (1) was positive semidefinite in all twenty. For the overlap operator built on it ($$m_0=1$$, plus a mass $$0.1$$), @@GOV@@. Positivity of the Gabriel overlap kernel is thus a numerical observation, not a theorem; for the lattice case it is proved in [[KU1]](#ref-ku1), [[KU2]](#ref-ku2).

### Zero modes at vertices of degree one

The disorder average of the massless propagator exists only if the operator is invertible almost surely. On the Gabriel graph it is not invertible in the sample below.

<div class="proposition" markdown="1">
**Proposition 3** (dangling vertex; even $$d$$). Let $$D_G$$ be the Gabriel Wilson operator with $$r=1$$ and no gauge field, and let $$i$$ be a vertex of degree one with unique neighbour $$j$$ and $$e=e_{ij}$$. Let $$\chi\in\mathbb C^{2^{d/2}}$$ satisfy $$(1+\gamma\cdot e)\chi=0$$ (a $$-1$$ eigenvector, which exists since $$\gamma\cdot e$$ has eigenvalues $$\pm1$$). Then $$\psi=\delta_i\otimes\chi$$ satisfies $$D_G\psi=\tfrac12\psi$$ and $$\psi^\dagger\gamma_5\psi=0$$.
</div>

<div class="proof" markdown="1">
*Proof.* The column of $$D_G$$ at $$i$$ has the diagonal block $$\tfrac r2\deg i=\tfrac12$$ and the block $$D_G(j,i)=\tfrac12(\gamma\cdot e_{ji}-r)=-\tfrac12(\gamma\cdot e+1)$$ at $$j$$, and $$(\gamma\cdot e+1)\chi=0$$. Since $$\gamma_5$$ anticommutes with $$\gamma\cdot e$$ it maps the $$-1$$ eigenspace of $$\gamma\cdot e$$ to the $$+1$$ eigenspace, which is orthogonal to it. $$\square$$
</div>

A vertex of degree one is a local event of positive probability for a Poisson process, so its density per unit volume is positive. The proposition gives an eigenvector of $$D_G$$ with chirality zero. It does not by itself give a zero mode of the overlap operator, since $$\mathrm{span}\{\psi,\gamma_5\psi\}$$ is not invariant under $$H=\gamma_5(D_G-m_0)$$. @@TOR@@

For the disorder average the relevant bound is the following.

<div class="lemma" markdown="1">
**Lemma 4** (mass bound). Let $$H$$ be Hermitian and invertible, $$V=\gamma_5\,\mathrm{sign}\,H$$ and $$D_{\rm ov}=m_0(1+V)$$. For $$m>0$$ every singular value of $$D_{\rm ov}+m$$ is at least $$m$$, and $$\lVert(D_{\rm ov}+m)^{-1}\rVert\le1/m$$.
</div>

<div class="proof" markdown="1">
*Proof.* $$V$$ is unitary, so $$D_{\rm ov}+m=(m_0+m)+m_0V$$ is normal with eigenvalues $$m_0+m+m_0e^{i\varphi}$$, whose real part is at least $$m$$. For a normal matrix the singular values are the moduli of the eigenvalues. $$\square$$
</div>

If $$\nu(\ker D_{\rm ov}\ne0)>0$$, the expectation of $$\lVert D_{\rm ov}^{-1}\rVert$$ is infinite and the averaged massless propagator is not defined. This happens for the Gabriel overlap if the observation above persists with the size, since degree-one vertices have a density proportional to the volume. With a mass $$m>0$$, Lemma 4 bounds every realisation, and the average of every bounded probe exists.

### Averaged-law positivity

Whether the disorder averages $$\mathbb E\langle\Theta(\mathcal A)\mathcal A\rangle_{\mathcal P}$$ on the unsymmetrised ensemble are non-negative is open. A reflection acts on none of the realisations, so positivity is a linear condition on the averaged state that realisations do not inherit by convexity. If it holds, together with Euclidean invariance in law and an integrability condition of the form $$\mathbb E\lvert\langle\Theta(\mathcal A)\,\tau_t\mathcal B\rangle_{\mathcal P}\rvert<\infty$$ uniformly in the time translation $$t$$, the Osterwalder–Schrader construction gives a Hilbert space, a positive Hamiltonian and a vacuum for the averaged fields. The reconstructed theory is not that of a free fermion: the averaged state is not quasi-free, since $$\mathbb E[S\otimes S]\ne\mathbb ES\otimes\mathbb ES$$, and the Hamiltonian is not an average of Hamiltonians, because no realisation has a transfer matrix. A test needs probes and a normalisation for which the averaged diagonal entries are resolved away from zero.

## 4. A frame that belongs to a cluster of particles

The [page on frame leakage](/quantum-gravity/lorentz-violation-leakage/) shows that a frame cannot be averaged away, because there is no Lorentz-invariant probability measure on unit timelike vectors, and that a frame in a matter regulator at the discreteness scale produces a dimension-four Lorentz-violating coefficient of order $$g^2/16\pi^2$$ for a Yukawa coupling $$g$$ and a regulator depending on the loop momentum orthogonal to the frame. A frame can instead be the velocity $$u=P_C/\sqrt{P_C^2}$$ of a cluster $$C$$ of particles with total momentum $$P_C$$, so that $$u$$ is a function of the state and not a field. In the construction of Bakamjian and Thomas the mass operator of a system is a function of the internal variables in its rest frame, and interactions depend on the total momentum only through this velocity [[BT]](#ref-bt). Cluster separability, the requirement that distant clusters do not influence each other, then needs Sokolov's packing operators [[SRK]](#ref-srk), which are constructed for a fixed number of particles; models with particle production exist for a finite number of bare degrees of freedom [[P]](#ref-p), and the case with infinitely many is open.

The one-loop consequences are examined in the Euclidean Yukawa model $$g\bar\psi\psi\phi$$ with a massless fermion, a scalar of mass $$M$$ and the regulator $$D(q)=f(\lvert q_\perp\rvert/\Lambda)/(q^2+M^2)$$ on the scalar line, with $$f(x)=e^{-x^2}$$ and $$q_\perp$$ the part of $$q$$ orthogonal to $$u$$. The inverse propagator of the fermion is $$\Gamma(p)=i\gamma\cdot(p+\Sigma(p))$$ with

$$
\Sigma_\mu(p)=g^2\!\int\!\frac{d^4k}{(2\pi)^4}\,\frac{k_\mu}{k^2}\,D(p-k),
$$

as on the linked page. Everything in this section is a four-dimensional one-loop statement.

<div class="proposition" markdown="1">
**Proposition 4** (a line's own frame). (a) Let the regulator frame be $$u=\hat p$$, the direction of the external momentum. Then $$\Sigma_\mu(p)=p_\mu A(p^2)$$ with $$A$$ analytic for $$\lvert p^2\rvert<M^2$$. (b) If $$u$$ is independent of $$p$$, then to first order in $$p$$ the self-energy is $$\Sigma_\mu=a_\parallel(p\cdot u)u_\mu+a_\perp p_{\perp\mu}$$ with $$p_\perp=p-(p\cdot u)u$$, and $$16\pi^2(a_\perp-a_\parallel)/g^2\to\tfrac13$$ as $$M/\Lambda\to0$$. So $$\Sigma$$ is covariant at this order only if $$u$$ is parallel or orthogonal to $$p$$.
</div>

<div class="proof" markdown="1">
*Proof of (a).* For $$u=\hat p$$ the only vectors are $$u$$ and $$p$$, which are parallel, so the components of $$\Sigma_\mu$$ orthogonal to $$p$$ are odd under $$k_\perp\to-k_\perp$$ and vanish, and $$\Sigma_\mu=\hat p_\mu B(\lvert p\rvert)$$. On the line $$p=s\,\hat p$$, with $$k'=s-k_\parallel$$, the parallel component is

$$
\Sigma_\parallel(s)=g^2\!\int\!\frac{dk'\,d^3k_\perp}{(2\pi)^4}\,\frac{k'}{k'^2+k_\perp^2}\;\frac{f(\lvert k_\perp\rvert/\Lambda)}{(s-k')^2+k_\perp^2+M^2},
$$

and the first factor does not depend on $$s$$. The integral is analytic in $$s$$ for $$\lvert\mathrm{Im}\,s\rvert<M$$, because the poles of the last factor in $$k'$$ are at $$s\pm i\sqrt{k_\perp^2+M^2}$$ and cannot cross the real axis. The change $$k'\to-k'$$ shows $$\Sigma_\parallel(-s)=-\Sigma_\parallel(s)$$, so $$\Sigma_\parallel(s)=s\,A(s^2)$$ with $$A$$ analytic. $$\square$$
</div>

Part (b) is the one-loop transfer of a frame in a matter regulator, proved on the linked page for a wider class of regulators ($$a_\parallel=Z_\parallel-1$$ and $$a_\perp=Z_\perp-1$$ there, and $$Z_\perp-Z_\parallel\to g^2/48\pi^2$$). The scalar mass has to be kept until the end: at $$M=0$$ the integral is conditionally convergent, and integrating over the component along $$u$$ first gives a different value. For the Gaussian regulator, direct quadrature gives $$16\pi^2(a_\perp-a_\parallel)/g^2=0.33318,\ 0.333331,\ 0.33333330$$ at $$M/\Lambda=10^{-2},10^{-3},10^{-4}$$.

The coefficient in (b) was obtained at momenta $$\lvert p\rvert<M$$. Evaluating the same integrals at finite $$\lvert p\rvert$$, with $$p$$ along $$u$$ and orthogonal to $$u$$, gives for $$M=0.01\Lambda$$ the values $$0.3320,\ 0.3241,\ 0.2828$$ of $$16\pi^2(a_\perp-a_\parallel)/g^2$$ at $$\lvert p\rvert=0.03,\ 0.1,\ 0.3\,\Lambda$$, and for $$M=0.05\Lambda$$ the values $$0.3227,\ 0.2823$$ at $$\lvert p\rvert=0.1,\ 0.3\,\Lambda$$. The coefficient therefore stays at $$1/3$$ for $$M\ll\lvert p\rvert\ll\Lambda$$, with corrections that grow roughly like $$p^2/\Lambda^2$$. A fermion mass $$m=0.05\Lambda$$ in the loop changes the values at $$\lvert p\rvert=0.03,\ 0.1\,\Lambda$$ to $$0.3279$$ and $$0.3210$$.

A regulator whose frame is the line's own momentum thus gives no Lorentz violation in the self-energy. It is not harmless for the vector two-point function.

<div class="proposition" markdown="1">
**Proposition 5** (non-local term of a transverse cutoff). Let a Dirac fermion of mass $$m$$ and charge $$e$$ run in a loop with two vector vertices in Euclidean signature, and multiply the loop integrand by $$f(\lvert k_\perp\rvert/\Lambda)$$ with $$f\ge0$$, $$k_\perp$$ orthogonal to $$u=\hat q$$, where $$q$$ is the external momentum. As $$q\to0$$ along a fixed direction, $$\Pi^{\mu\nu}\to\alpha\,(\delta^{\mu\nu}-\hat q^\mu\hat q^\nu)$$ with $$q_\mu\Pi^{\mu\nu}=0$$, and as $$m\to0$$

$$
\alpha=\frac{2e^2}{3\pi^2}\,\Lambda^2\!\int_0^\infty x\,f(x)\,dx\ \neq0, \tag{2}
$$

up to the sign convention for $$\Pi$$. The limit $$q\to0$$ depends on the direction of $$q$$: $$\Pi^{\mu\nu}$$ contains the non-local term $$-\alpha\,q^\mu q^\nu/q^2$$.
</div>

<div class="proof" markdown="1">
*Proof.* At $$q=0$$ the integrand is $$\mathrm{tr}[\gamma^\mu(-i\not k+m)\gamma^\nu(-i\not k+m)]/(k^2+m^2)^2$$ weighted by $$f$$. Put $$b^2=\lvert k_\perp\rvert^2+m^2$$. The component along $$\hat q$$ has the numerator $$4(b^2-k_\parallel^2)$$, and $$\int(b^2-k_\parallel^2)(k_\parallel^2+b^2)^{-2}dk_\parallel=[k_\parallel/(k_\parallel^2+b^2)]_{-\infty}^{\infty}=0$$ for every $$f$$, so $$\Pi^{\parallel\parallel}=0$$. This reflects the invariance of the regulator under shifts of $$k$$ along $$u$$, which gives $$q_\mu\Pi^{\mu\nu}(q)=0$$ at finite $$q$$ as well. It remains to compute a transverse component, which equals minus the difference between the longitudinal and the transverse component. That difference is $$-8\int\frac{d^4k}{(2\pi)^4}f\,(k_\parallel^2-\lvert k_\perp\rvert^2/3)/(k^2+m^2)^2$$, where $$\lvert k_\perp\rvert^2/3$$ is the average over the three transverse directions. The $$k_\parallel$$ integrals are $$\int k_\parallel^2(k_\parallel^2+b^2)^{-2}dk_\parallel=\pi/(2b)$$ and $$\int(k_\parallel^2+b^2)^{-2}dk_\parallel=\pi/(2b^3)$$, so for $$m\to0$$, $$\lvert k_\perp\rvert=\varrho$$, the combination is $$\pi/(2\varrho)-(\varrho^2/3)\pi/(2\varrho^3)=\pi/(3\varrho)$$ (the two terms are in the ratio $$3:1$$ for every $$f$$), and $$\int d^3k_\perp$$ gives $$4\pi\int\varrho^2f\,\pi/(3\varrho)\,d\varrho=(4\pi^2/3)\int\varrho f\,d\varrho$$. Collecting $$8/(2\pi)^4$$ gives $$2/(3\pi^2)$$ times $$\Lambda^2\int xf\,dx$$, which is (2). $$\square$$
</div>

The regulator preserves the Ward identity, but the transverse part survives as a direction-dependent photon-mass-like term; it is non-analytic at $$q=0$$ and no local counterterm removes it. For the Gaussian regulator and $$m=0.01\Lambda$$, quadrature gave $$\alpha/e^2=0.0337720$$ in units $$\Lambda=1$$, against $$1/(3\pi^2)=0.0337737$$ from (2).

### Crossing symmetry of the one-loop amplitude

A further requirement of an effective local theory is crossing symmetry: the amplitude for $$\psi\phi\to\psi\phi$$ in the direct channel and in the crossed channel must be boundary values of one analytic function of the invariants. We label the process $$\psi(p)+\phi(K)\to\psi(p')+\phi(K')$$, with $$p^2=p'^2=m^2$$ and $$K^2=K'^2=M^2$$, and set

$$
P=p+K=p'+K',\qquad Q=p-K'=p'-K,\qquad s=P^2,\quad\tilde s=Q^2,\quad t=(p-p')^2 .
$$

In Minkowski signature the tree amplitude is the sum of the exchange of a fermion of momentum $$P$$ and of one of momentum $$Q$$, $$A_{\rm tree}=g^2\bar\chi(p')(S_P+S_Q)\chi(p)$$ with $$S_q=(\not q+m)/(q^2-m^2)$$ and $$\chi,\bar\chi$$ the external spinors. Since $$P+Q=p+p'$$ and $$P-Q=K+K'$$, one has $$(P+Q)^2-(P-Q)^2=4m^2-4M^2$$, that is

$$
P\cdot Q=m^2-M^2 .
$$

In the crossed region $$\tilde s\ge(m+M)^2$$ the same amplitude describes the elastic process $$\psi(p)+\phi(-K')\to\psi(p')+\phi(-K)$$ with total momentum $$Q$$. With cluster frames, the amplitude in the direct region is computed with $$u_P=P/\sqrt s$$ and in the crossed region with $$u_Q=Q/\sqrt{\tilde s}$$.

The coefficient of Proposition 4(b) is the input. It is a statement about the Euclidean self-energy of a line, and it is used here at the off-shell momenta of the internal lines, in Minkowski signature. We state what is assumed.

<div class="remark" markdown="1">
**Hypothesis (L)** (locality of the frame dependence). For momenta $$q$$, $$m$$, $$M$$ much smaller than $$\Lambda$$, the part of the one-loop self-energy of a fermion line that depends on the frame $$u$$ is the insertion

$$
\Sigma_u(q)=B\,\frac{(q\cdot u)\not u}{u^2},\qquad B=\frac{g^2}{48\pi^2},
$$

into the inverse propagator $$\not q-m-\Sigma_u$$, up to corrections suppressed by $$q^2/\Lambda^2$$; the expression is the part of $$\not q$$ along $$u$$, has the same form in both signatures, and is continued from Euclidean to Minkowski momenta as a local operator. The frame dependence of the vertex and box subgraphs of the $$\psi\phi\to\psi\phi$$ amplitude is $$O(q/\Lambda)$$. Frame-dependent corrections on the external legs are not included in $$\Delta$$ below.
</div>

Proposition 4(b) fixes $$B=a_\perp-a_\parallel$$ at $$\lvert p\rvert<M$$: removing the frame-independent factor $$1+a_\perp$$ from $$\Gamma/i=(1+a_\perp)\not p-(a_\perp-a_\parallel)(p\cdot u)\not u$$ leaves $$\not p-B(p\cdot u)\not u$$. The finite-momentum values above support the extension to $$M\ll\lvert q\rvert\ll\Lambda$$ in the Euclidean model; the angular dependence is produced by loop momenta of order $$\Lambda$$ (it comes from the $$-\ln\sin\chi$$ term of the radial integral on the linked page), so that for external momenta well below $$\Lambda$$ it is carried by a local operator. This is the content of (L); beyond the Euclidean check it is not proved. The statement about vertex and box subgraphs rests on power counting (their frame-dependent parts are ultraviolet convergent, and the logarithmic part of the vertex is a frame-independent scalar because $$\not u\not u=1$$); those graphs were not computed.

<div class="remark" markdown="1">
**Derivation 1** (crossing defect at one loop, under (L)). Under (L) the frame-dependent part of the one-loop amplitude is the sum of the two tree diagrams with $$S_q\to S_q\Sigma_uS_q$$ on the internal line. Let $$\Delta$$ be the difference between the amplitude computed with the frame $$u_P$$ in both diagrams and the amplitude computed with $$u_Q$$ in both diagrams, at the same momenta. Then

$$
\Delta=g^2B\ \bar\chi(p')\Big\{S_P\Big[\not P-\frac{m^2-M^2}{\tilde s}\not Q\Big]S_P+S_Q\Big[\frac{m^2-M^2}{s}\not P-\not Q\Big]S_Q\Big\}\chi(p). \tag{3}
$$

It is independent of $$\Lambda$$, is non-zero in general, and has a pole at $$s=0$$ with residue proportional to $$m^2-M^2$$ (and one at $$\tilde s=0$$).
</div>

To derive (3), insert $$\Sigma_u$$ on each line in each frame. On the line of momentum $$P$$, the frame $$u_P$$ gives $$B\,(P\cdot u_P)\not u_P=B\not P$$, a covariant renormalisation of the line, and the frame $$u_Q$$ gives $$B\,(P\cdot Q)\not Q/\tilde s=B\,(m^2-M^2)\not Q/\tilde s$$, which is not proportional to $$\not P$$. On the line of momentum $$Q$$ the roles are exchanged: $$B\,(m^2-M^2)\not P/s$$ in the frame $$u_P$$ and $$B\not Q$$ in the frame $$u_Q$$. Subtracting on each line the insertion in the frame $$u_Q$$ from the insertion in the frame $$u_P$$, and attaching $$S_q$$ on both sides and the factor $$g^2$$ of the two Yukawa vertices, gives (3). If $$\Delta\ne0$$, the amplitude with the frame of its own channel in each region is not the boundary value of one analytic function, because the continuation of the direct-channel amplitude to the crossed region is the amplitude with $$u_P$$, not with $$u_Q$$.

To see the size of $$\Delta$$ we take $$m=1$$, $$M=0.5$$, evaluate (3) in the crossed physical region (centre-of-mass frame of $$Q$$, scattering angle $$\theta$$ between $$p$$ and $$p'$$) and form $$R=2\,\mathrm{Re}\sum_{\rm spins}\Delta A_{\rm tree}^*\big/\sum_{\rm spins}\lvert A_{\rm tree}\rvert^2$$, the relative change of the spin-summed squared tree amplitude due to $$\Delta$$. In units of $$g^2/16\pi^2$$ (so $$B=\tfrac13$$):

| $$\sqrt{\tilde s}$$ | $$1.55$$ | $$2$$ | $$3$$ | $$5$$ | $$10$$ |
|---|---|---|---|---|---|
| $$\cos\theta$$ | $$0.3$$ | $$0.3$$ | $$0.3$$ | $$-0.5$$ | $$-0.5$$ |
| $$R$$ | $$+12.87$$ | $$-3.26$$ | $$-0.796$$ | $$+0.087$$ | $$+0.32$$ |

At $$\sqrt{\tilde s}=2.2$$ and $$\cos\theta=-0.9054$$ the point $$s=0$$ lies inside the physical region; there $$sR=-0.154$$ on both sides, so $$R$$ diverges like $$1/s$$. For $$m=M$$ the residue vanishes by (3). The defect is removed by a counterterm $$-B\,(q\cdot u_C)\not u_C/u_C^2$$ on every internal line, with $$u_C$$ the frame of the channel in which the line carries the momentum $$q$$: one dimension-four coefficient per order, the tuning of [[CPSUV]](#ref-cpsuv).

Graphs with zero total momentum, tadpoles and vacuum bubbles, have no cluster frame, and the Lorentzian orbit of frames is non-compact, so no normalisable average over it exists (see the linked page). The computation above does not extend to those graphs.

## 5. Limits and open points

The results above have the following limits. Each statement lists the hypothesis that marks the boundary.

- Theorem 1 needs a Wilson diagonal equal to the row sum and a hard range cut along a frame. Other diagonals, signed covariant weights (Lemma 1 needs $$w\ge0$$), and a Wilson term with a coefficient that depends on $$u$$ (Lemma 2 treats the covariant term) are not covered. Whether signed weights reach a covariant infrared Weyl term is not settled.
- Theorem 2 needs translation invariance along $$n$$. Lattices, random shifts of lattices restricted to symmetry-compatible shifts, and reflection planes at special positions lie outside it.
- Averaged-law reflection positivity of the unsymmetrised ensemble, with a mass (Lemma 4) or on a graph without degree-one vertices, has neither a proof nor a violation. The same holds for the four-dimensional version of the Euclidean operator, for which every numerical statement above is two-dimensional.
- Reflection positivity of the Gabriel Wilson term is conditional on the cited criterion and is for the free operator. Other kernels with the cross-plane property, and gauge fields with non-trivial cross-plane links, which need a reflection-symmetric measure for the links, are not examined. That the zero-mode pair of the massless Gabriel overlap is attached to a degree-one vertex is an observation at $$N=400$$ and is not proved.
- The cluster-frame results are for a Yukawa model with a Gaussian regulator on the scalar line, under hypothesis (L). A gauge theory, two loops, the vertex and box graphs, external-leg corrections, per-particle frames (each particle with its own axis on one shared point set) and frames supplied by the state of an exactly Lorentz-invariant theory are not treated. Cluster separability with particle production for infinitely many degrees of freedom is not established.

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
