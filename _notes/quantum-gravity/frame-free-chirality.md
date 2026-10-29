---
title: Four classes of frame-free Weyl operators carry no overlap index
date: 2026-10-29
category: matter
order: 2
description: An overlap operator built on a Lorentz-invariant random point set from the causal order and the invariant interval alone has overlap index zero or undefined when it is retarded, has a Hermitian or anti-Hermitian Weyl block, is self-adjoint for a chirality-swapping Krein involution, or defines a finite Atiyah-Patodi-Singer index. A Euclidean overlap operator on the same points carries the index, at the price of a Wick axis.
status: Linear-algebra theorems proved in full in any dimension; mean symbol and noise estimates proved for one two-dimensional kernel; numerical checks in two dimensions at up to 300 points.
excludes: Integer chirality from overlap or Ginsparg-Wilson operators on a Poisson sprinkling that use no frame, when the operator is retarded, (anti-)Hermitian in its Weyl block, Krein-self-adjoint for a chirality-swapping involution, or a finite boundary-projected index.
---

A Weyl fermion on a lattice or on a random point set is certified chiral by an integer. The integer is the index of a Ginsparg–Wilson (overlap) Dirac operator, and for a background gauge field it equals minus the flux through the torus. We ask whether such an integer exists for an operator on a Poisson sprinkling in Minkowski space that is built from the causal order and the Lorentz-invariant interval only, so that no preferred frame enters. The answer is no for four classes of constructions: operators that are retarded, operators whose Weyl block is Hermitian or anti-Hermitian, operators that are self-adjoint for an involution that swaps the two chiralities, and indices defined by projecting initial and final data onto the positive or negative spectral subspace of a boundary operator on finite antichains. In each class the overlap index is zero, undefined, or a function of boundary data that cannot see the bulk, whatever the weights, Wilson term, gauge field and dimension. The proofs are finite-dimensional linear algebra and do not use the sprinkling at all. A Euclidean overlap operator on the same Poisson points does carry the index (Section 2), at the price of a Wick axis.

The overlap construction is due to Neuberger [[N1]](#ref-n1), [[N2]](#ref-n2) and rests on the Ginsparg–Wilson relation [[GW]](#ref-gw); Lüscher showed that the relation implies an exact index theorem on the lattice [[L]](#ref-l). A frame-free operator avoids the doubling of the Nielsen–Ninomiya theorem [[NN]](#ref-nn) because its symbol is not periodic (Proposition 2), so doubling is not what obstructs chirality here. Lorentz-invariant retarded kernels on causal sets were studied by Aslanbeigi, Saravani and Sorkin [[ASS]](#ref-ass), and Bombelli, Henson and Sorkin proved that a sprinkling carries no preferred direction and no finite-valency graph [[BHS]](#ref-bhs). In the continuum, Bär and Strohmaier proved an index theorem for the Lorentzian Dirac operator on a compact globally hyperbolic region with Atiyah–Patodi–Singer conditions on spacelike boundaries [[BS]](#ref-bs). On the lattice, the Atiyah–Patodi–Singer index has been formulated through domain-wall eta invariants [[F]](#ref-f). Kimura and Watanabe showed that the symmetry classes of the tenfold way constrain the overlap operator and its index [[KW]](#ref-kw), which is the mechanism behind Theorem 2 below.

What the page adds is the classification of the frame-free constructions into the four classes above, with proofs that apply to any weights on any causal set; the exact reduction of an internal compact index to a two-dimensional one-handed propagation problem (Proposition 4); and, for the two-dimensional signed Benincasa–Dowker kernel, the mean symbol, a row-noise law and a smearing estimate (Propositions 1, 5 and 6). Theorems 1 and 2 and Propositions 3 and 4 hold in every dimension. Proposition 1, Proposition 2, Propositions 5 and 6 and every numerical statement refer to two dimensions. Section 8 lists what remains open.

## 1. The frame-free retarded Weyl operator

Let $$\mathcal P$$ be a Poisson process of density $$\rho$$ in two-dimensional Minkowski space, ordered by $$y\prec x$$ when $$x-y$$ is future timelike. In null coordinates $$u=t-x$$, $$v=t+x$$ one has $$dt\,dx=\tfrac12\,du\,dv$$, and $$y\prec x$$ exactly when $$U=u_x-u_y>0$$ and $$V=v_x-v_y>0$$. For $$y\prec x$$ let $$n_{xy}$$ be the number of points of $$\mathcal P$$ strictly between $$y$$ and $$x$$. The pair $$(n_{xy},\,(U,V))$$ is the data we call frame-free: the count is a property of the order, and the null displacements are what a boost multiplies by $$e^{\mp\eta}$$. No timelike unit vector is used.

<div class="definition" markdown="1">
**Definition 1** (signed-BD retarded Weyl operator). Let $$w_0=1$$, $$w_1=-2$$, $$w_2=1$$ and $$w_n=0$$ for $$n\ge 3$$; these are the two-dimensional Benincasa–Dowker layer weights (see [signed substrates](/quantum-gravity/signed-substrates/)). For a point set $$\mathcal P$$ with $$N$$ elements put, for $$y\prec x$$,

$$
D_+[x,y]=\rho\,w_{n_{xy}}\,(u_x-u_y),\qquad D_-[x,y]=\rho\,w_{n_{xy}}\,(v_x-v_y),
$$

and zero when $$y\not\prec x$$. On $$\mathbb C^N\otimes\mathbb C^2$$, with the two components called $$R$$ and $$L$$ and $$\gamma_5=\mathrm{diag}(1,-1)$$, the retarded Dirac operator is $$D=\begin{pmatrix}0&D_+\\ D_-&0\end{pmatrix}$$. A smeared version replaces $$w_{n}$$ by $$\varepsilon^2 f(n,\varepsilon)$$ with $$f(n,\varepsilon)=\sum_k w_k\binom nk\varepsilon^k(1-\varepsilon)^{n-k}$$, the Poisson-thinning form of the generalised causal-set d'Alembertians of [[ASS]](#ref-ass); $$\varepsilon=1$$ is the unsmeared operator.
</div>

The operator $$D_+$$ approximates $$\partial_v$$ and $$D_-$$ approximates $$\partial_u$$. The next proposition makes this precise on plane waves, in the limit in which the region is large compared with the discreteness scale $$\rho^{-1/2}$$. It is a computation with the Mecke formula, which says that, given points at $$x$$ and $$y$$, the number of other points in the order interval between them is Poisson with mean $$\rho UV/2$$ [[LP]](#ref-lp).

<div class="proposition" markdown="1">
**Proposition 1** (mean symbol; two dimensions). Let $$\psi(y)=e^{i(k_uu_y+k_vv_y)}$$ with $$k_v\ne0$$, and let $$S_\varepsilon(k)$$ be the expectation of $$(D^{\varepsilon}_+\psi)(x)/\psi(x)$$ given a point at $$x$$, the sum over the past of $$x$$ being taken first over $$V$$ and then over $$U$$. Then

$$
S_\varepsilon(k)=i\,k_v\,h\!\Big(\frac{-2k_uk_v}{\varepsilon\rho}\Big),\qquad h(w)=1+w-w(2+w)\,e^{w}E_1(w)=2\int_0^\infty\frac{t\,e^{-wt}}{(1+t)^3}\,dt, \tag{1}
$$

where $$E_1$$ is the exponential integral. For $$k_uk_v<0$$ the argument $$w$$ is positive and (1) holds as written. For $$k_uk_v>0$$ one takes the boundary value $$w\to w+i0\,\mathrm{sgn}(k_u+k_v)$$ on the cut $$w<0$$. Moreover $$h(0)=1$$, $$h$$ has no zero on the real line (on either side of the cut), and $$\lvert h(w)\rvert \le C/(1+w^2)$$ for a constant $$C$$. In particular $$S_\varepsilon(k)=0$$ exactly on the null line $$k_v=0$$, and $$S_\varepsilon(k)\to i k_v$$ as $$k\to0$$.
</div>

<div class="proof" markdown="1">
*Proof.* The idea is to reduce the sum over the past to a one-dimensional integral with the Mecke formula, and to evaluate the integral along a rotated contour. Rescaling $$V$$ by $$\varepsilon\rho$$ shows that it suffices to treat $$\varepsilon=\rho=1$$ and to replace $$k_v$$ by $$k_v/(\varepsilon\rho)$$ at the end, so we do that.

By the Mecke formula the expectation is $$\tfrac12\iint U\,f_0(UV/2)\,e^{-i(k_uU+k_vV)}\,dU\,dV$$, where $$f_0(z)=e^{-z}(1-2z+z^2/2)$$ is the Poisson average of $$w_n$$ (for general $$\varepsilon$$ one averages $$f(n,\varepsilon)$$ over $$n\sim\mathrm{Poisson}(\lambda)$$ and obtains $$f_0(\varepsilon\lambda)$$). Write this as $$i\partial_{k_u}G$$ with $$G=\tfrac12\iint f_0(UV/2)\,e^{-i(k_uU+k_vV)}dU\,dV$$. With $$s=U/2$$ and $$\beta=ik_v$$ the $$V$$-integral of $$z^n e^{-z}$$ terms gives $$\sum_n a_n n!\,s^n/(s+\beta)^{n+1}$$ with $$(a_0,a_1,a_2)=(1,-2,\tfrac12)$$, and the sum is $$\beta^2/(s+\beta)^3$$. Hence

$$
G=\int_0^\infty e^{-2ik_us}\,\frac{\beta^2}{(s+\beta)^3}\,ds ,
$$

which converges absolutely. If $$k_uk_v<0$$ one rotates the contour to $$s=\beta t$$, which sweeps a quadrant that contains neither the pole $$s=-\beta$$ nor a contribution from the arc at infinity, and finds $$G=\int_0^\infty e^{-wt}(1+t)^{-3}dt$$ with $$w=-2k_uk_v>0$$. Since $$S=i\partial_{k_u}G=ik_v\,(-2G'(w))$$ one gets the integral form of $$h$$, and integrating by parts twice gives the closed form in (1). For $$k_v>0$$, $$G$$ is analytic in $$k_u$$ in the lower half plane, which corresponds to $$\mathrm{Im}\,w>0$$; the formula, analytic in $$w$$ off the cut, agrees with $$G$$ on the real half line and therefore everywhere in that half plane, and the real timelike values are the boundary values $$w+i0$$. For $$k_v<0$$ the same argument gives $$w-i0$$. This is the stated rule, since for $$k_uk_v>0$$ the sign of $$k_u+k_v$$ equals the sign of $$k_v$$.

For $$w>0$$ the integral form gives $$h>0$$ and $$h(w)\le 2\int_0^\infty t e^{-wt}dt=2/w^2$$. For $$w=-a\pm i0$$ with $$a>0$$ the identity $$E_1(-a\pm i0)=-\mathrm{Ei}(a)\mp i\pi$$ gives $$\mathrm{Im}\,h=\mp\pi a(2-a)e^{-a}$$, which vanishes only at $$a=2$$, where $$w(2+w)=0$$ and $$h=1+w=-1\neq0$$. So $$h$$ has no real zero. For the decay on the cut, rotate the integral form to the ray $$t=\tau e^{\pm 2\pi i/3}$$, which is allowed because the pole of $$(1+t)^{-3}$$ at $$t=-1$$ is not crossed; there $$\lvert e^{-wt}\rvert =e^{-a\tau/2}$$ and $$\lvert 1+t\rvert \ge\sqrt3/2$$, so $$\lvert h(-a\pm i0)\rvert \le 2(2/\sqrt3)^3\,(4/a^2)=64/(3\sqrt3\,a^2)$$. Continuity of $$h$$ near $$w=0$$ gives the uniform bound. $$\square$$
</div>

We checked (1) against a direct numerical evaluation of the $$s$$-integral at five momenta, one spacelike and four timelike, including both signs of $$k_u+k_v$$; the values of $$ik_vh$$ agree with the integral to eight digits, and they select the sign $$w+i0\,\mathrm{sgn}(k_u+k_v)$$ as stated. The statement that $$h$$ has no real zero means that the mean operator has one chirality of zero on the null line and no further zeros: no species doubling. That is a general feature of covariant symbols.

<div class="proposition" markdown="1">
**Proposition 2** (covariant symbols do not double; two dimensions). Let $$D(p)$$ be a $$2\times2$$ matrix function of $$p=(p_u,p_v)$$, continuous at $$0$$ with $$D(0)=0$$, covariant under boosts: $$S_\eta D(\Lambda_\eta p)S_\eta^{-1}=D(p)$$ with $$\Lambda_\eta(p_u,p_v)=(e^{-\eta}p_u,e^{\eta}p_v)$$ and $$S_\eta=\mathrm{diag}(e^{\eta/2},e^{-\eta/2})$$. Then on the half-lines $$p_v=0$$ only the entry $$D_{12}$$ can be non-zero and it is linear in $$p_u$$; on the half-lines $$p_u=0$$ only $$D_{21}$$ can be non-zero and it is linear in $$p_v$$. Every other zero set of $$\det D$$ is a union of boost orbits, that is of hyperbolas $$p_up_v=\mathrm{const}\neq0$$.
</div>

<div class="proof" markdown="1">
*Proof.* Conjugation by $$S_\eta$$ leaves the diagonal entries unchanged and multiplies $$D_{12}$$ by $$e^{\eta}$$ and $$D_{21}$$ by $$e^{-\eta}$$. On the ray $$p=\kappa(1,0)$$ one has $$\Lambda_\eta p=e^{-\eta}p$$. Covariance gives $$D_{11}(e^{-\eta}p)=D_{11}(p)$$, and the left side tends to $$D_{11}(0)=0$$ as $$\eta\to\infty$$, so $$D_{11}=0$$ on the ray; the same holds for $$D_{22}$$. It gives $$D_{21}(e^{-\eta}p)=e^{\eta}D_{21}(p)$$, whose left side tends to $$0$$, so $$D_{21}(p)=0$$. It gives $$D_{12}(e^{-\eta}p)=e^{-\eta}D_{12}(p)$$, which is linearity along the ray. The other null ray is treated with $$\eta\to-\infty$$. The determinant is boost invariant, so its zero set is a union of orbits, and the orbits of the boost group in the plane are the hyperbolas $$p_up_v=\mathrm{const}$$, the four open null half-lines and the origin. $$\square$$
</div>

The symbol of Definition 1, $$D(k)=\begin{pmatrix}0&ik_vh\\ ik_uh&0\end{pmatrix}$$ with $$h$$ evaluated at $$-2k_uk_v/\rho$$, satisfies the hypotheses, and its determinant $$k_uk_vh^2$$ vanishes only on the two null lines by Proposition 1. Positive covariant weights cannot do this. A function $$\phi(\Delta^2)$$ of the invariant interval alone is never integrable over the plane unless it vanishes: in rapidity coordinates $$d^2\Delta=\tfrac12\,d\sigma\,d\theta$$ with $$\sigma=\Delta^2$$ and $$\theta\in\mathbb R$$, so $$\int\lvert \phi\rvert \,d^2\Delta=\infty$$ whenever $$\phi\ne0$$; the same holds on every hyperboloid in higher dimensions. A covariant kernel therefore has a symbol only through cancellations, and the layer weights $$(1,-2,1)$$ are such a cancellation: the zeroth and first moments of $$f_0$$ vanish.

## 2. The overlap operator and its index

The tools in this section are standard; we collect them so that the later statements are checkable. Let $$X$$ be a linear operator on a finite-dimensional space $$\mathcal V=\mathcal V_R\oplus\mathcal V_L$$ with grading $$\gamma_5=\pm1$$ on $$\mathcal V_{R/L}$$, and let $$m_0>0$$ be a real number. Put

$$
H=\gamma_5\,(X-m_0),\qquad D_{\rm ov}=m_0\,\big(1+\gamma_5\,\varepsilon\big),\qquad \varepsilon=\mathrm{sign}\,H . \tag{2}
$$

For a matrix $$H$$ with no eigenvalue on the imaginary axis, $$\mathrm{sign}\,H=P_+-P_-$$ where $$P_\pm$$ are the spectral projections onto the generalised eigenspaces with $$\mathrm{Re}\,\lambda>0$$ and $$<0$$. Then $$\varepsilon^2=1$$, $$\varepsilon$$ commutes with $$H$$, $$\mathrm{Tr}\,\varepsilon=N_+-N_-$$ is the difference of the numbers of eigenvalues (counted with algebraic multiplicity) in the two half planes, and for Hermitian $$H$$ it is the usual sign function. Here $$X$$ is the sum of a Dirac part and an optional Wilson term, and $$m_0$$ plays the role of the Wilson mass. We call the integer

$$
\mathrm{ind}\,D_{\rm ov}=-\tfrac12\,\mathrm{Tr}\,\mathrm{sign}\,H \tag{3}
$$

the overlap index; it is defined exactly when $$H$$ has no purely imaginary eigenvalue, and it is an integer because $$N_++N_-=\dim\mathcal V$$ is even when $$\mathrm{Tr}\,\gamma_5=0$$.

<div class="lemma" markdown="1">
**Lemma 1** (Ginsparg–Wilson relation and index). (a) For every $$X$$ for which $$\varepsilon$$ is defined, $$D_{\rm ov}$$ satisfies $$\gamma_5D_{\rm ov}+D_{\rm ov}\gamma_5=m_0^{-1}D_{\rm ov}\gamma_5D_{\rm ov}$$. (b) If $$\dim\mathcal V_R=\dim\mathcal V_L$$ and $$H$$ is Hermitian, then $$\mathrm{ind}\,D_{\rm ov}=n_+-n_-$$, where $$n_\pm$$ is the number of zero modes of $$D_{\rm ov}$$ with $$\gamma_5=\pm1$$.
</div>

<div class="proof" markdown="1">
*Proof.* (a) Using only $$\varepsilon^2=1$$, $$m_0^{-1}D_{\rm ov}\gamma_5D_{\rm ov}=m_0(1+\gamma_5\varepsilon)\gamma_5(1+\gamma_5\varepsilon)=m_0(2\gamma_5+\gamma_5\varepsilon\gamma_5+\varepsilon)$$, and $$\gamma_5D_{\rm ov}+D_{\rm ov}\gamma_5=m_0(2\gamma_5+\varepsilon+\gamma_5\varepsilon\gamma_5)$$.

(b) Let $$V=\gamma_5\varepsilon$$. For Hermitian $$H$$, $$\varepsilon$$ is Hermitian, so $$V$$ is unitary and $$D_{\rm ov}=m_0(1+V)$$ is normal, with $$\gamma_5D_{\rm ov}\gamma_5=D_{\rm ov}^\dagger$$. Its eigenvalues lie on the circle $$\lvert \lambda-m_0\rvert =m_0$$, so a real eigenvalue is $$0$$ or $$2m_0$$. The map $$\gamma_5$$ takes the eigenspace of a non-real $$\lambda$$ to that of $$\bar\lambda$$, which is orthogonal to it, so every eigenvector with non-real eigenvalue has $$\langle\gamma_5\rangle=0$$. Therefore $$\mathrm{Tr}\,\gamma_5(1-D_{\rm ov}/2m_0)$$ receives a contribution only from the eigenvalue $$0$$, namely $$n_+-n_-$$. On the other hand it equals $$\mathrm{Tr}\,\gamma_5-\tfrac12\mathrm{Tr}\,\gamma_5(1+\gamma_5\varepsilon)=-\tfrac12\mathrm{Tr}\,\varepsilon$$, because $$\mathrm{Tr}\,\gamma_5=0$$. $$\square$$
</div>

So for a $$\gamma_5$$-Hermitian kernel the index counts the net chirality of exact zero modes, and Lemma 1(a) is why the construction is exactly chiral at finite cutoff. The theorems below show that when the kernel is built without a frame the right side of (3) is forced to vanish by an extra structure, so that nothing is left to count.

<div class="example" markdown="1">
**Example 1** (the Euclidean overlap on Poisson points). Take a unit-density Poisson process on the torus of side $$L=\sqrt N$$ with antiperiodic spinors, Pauli matrices $$\gamma_1=\sigma_1$$, $$\gamma_2=\sigma_2$$, $$\gamma_5=\sigma_3$$, and the Gaussian-kernel Wilson–Dirac operator

$$
D_W[x,y]=w(\Delta_{xy})\Big[\frac{\gamma\cdot\Delta_{xy}}{Z_1}-\frac{r}{Z_0}\Big]U_{xy},\quad D_W[x,x]=\frac{r}{Z_0}\sum_{y\ne x}w(\Delta_{xy}), \tag{4}
$$

where $$\Delta_{xy}$$ is the displacement from $$x$$ to the nearest image of $$y$$, $$w(\Delta)=e^{-\lvert \Delta\rvert ^2/2s^2}$$ for $$\lvert \Delta\rvert <6s$$, $$Z_1=2\pi s^4$$, $$Z_0=2\pi s^2$$, and $$U_{xy}$$ are the link phases of a background U(1) field of total flux $$2\pi Q$$ (a uniform field in Landau gauge with transition functions). The mean symbol of $$D_W$$ is $$i\gamma\cdot p\,e^{-s^2p^2/2}+r(1-e^{-s^2p^2/2})$$. The operator uses the Euclidean metric $$\lvert \Delta\rvert $$, that is a Wick axis. With $$s=1.5$$, $$r=1$$, $$m_0=\tfrac12$$ we built $$D_{\rm ov}$$ by (2) with $$X=D_W$$ at $$N=300$$ for three point sets and $$Q=0,1,2,-1$$ each. The Ginsparg–Wilson residual was below $$3\times10^{-15}$$ in all twelve cases, $$\min\lvert \mathrm{eig}\,H\rvert $$ lay between $$0.05$$ and $$0.16$$, and the index was $$-Q$$ in every case, carried by exactly $$\lvert Q\rvert $$ zero modes, all of chirality $$-\mathrm{sgn}\,Q$$.
</div>

The index $$-Q$$ is the lattice index theorem. Because the law of a unit-density Poisson process is invariant under every volume-preserving affine map (the image of a Poisson process under such a map is Poisson with the image intensity, which is again Lebesgue measure), it is invariant under both the Euclidean rotations and the Lorentz boosts; so the law of $$D_W$$ built with the Euclidean metric of any unit timelike vector is the same, and the frame is a choice of Wick axis and not data carried by the points. Whether this operator defines a Lorentzian theory is the subject of [Frames, Wilson terms and reflection positivity for fermions on random points](/quantum-gravity/frames-and-reflection-positivity/). What the Euclidean operator has and the operators below lack is a pairing of the two chiral blocks by the adjoint: in Euclidean signature the adjoint of $$\partial_z$$ is $$-\partial_{\bar z}$$, so the Hermitian operator $$H=\gamma_5X$$ has no further symmetry.

## 3. Retarded operators carry no index

The first class is the most direct: the operator of Definition 1, with any weights, links, gauge phases and in any dimension, is retarded.

<div class="theorem" markdown="1">
**Theorem 1** (retarded operators). Let $$\mathcal C$$ be a finite partially ordered set with a linear extension $$x_1,\dots,x_N$$, and let the space $$\mathcal V$$ be the direct sum of spaces $$\mathcal V_x$$, $$x\in\mathcal C$$, each graded by $$\gamma_5$$. Assume

(H1) $$X$$ is block lower triangular for this ordering: the block $$X_{xy}$$ vanishes unless $$y\preceq x$$ in the linear extension. The diagonal blocks $$X_{xx}$$ are arbitrary.

(H2) $$m_0>0$$ is such that no $$H_{xx}=\gamma_5(X_{xx}-m_0)$$ has a purely imaginary eigenvalue.

Then $$H=\gamma_5(X-m_0)$$ is block lower triangular, its spectrum is the union of the spectra of the $$H_{xx}$$, the overlap operator (2) exists, satisfies the Ginsparg–Wilson relation exactly, and

$$
\mathrm{ind}\,D_{\rm ov}=-\tfrac12\sum_{x}\mathrm{Tr}\,\mathrm{sign}\,H_{xx}. \tag{5}
$$

In particular, if the diagonal blocks $$X_{xx}$$ vanish and each $$\mathcal V_x$$ has equally many $$R$$ and $$L$$ components, then $$\mathrm{ind}\,D_{\rm ov}=0$$, $$H$$ has the two eigenvalues $$\pm m_0$$ only, and $$D_{\rm ov}$$ is strictly block lower triangular, hence nilpotent. The index does not depend on any off-diagonal block, so it is independent of the weights, the links and the background gauge field.
</div>

<div class="proof" markdown="1">
*Proof.* The idea is that a triangular matrix has the spectrum of its diagonal, and that a sign function sees only the spectrum. Since $$\gamma_5$$ is block diagonal, $$H$$ is block lower triangular with diagonal blocks $$H_{xx}$$. The characteristic polynomial of a block triangular matrix is the product of those of the diagonal blocks, so the eigenvalues of $$H$$, with algebraic multiplicity, are those of the $$H_{xx}$$, and (H2) says none lies on the imaginary axis. Then $$\mathrm{Tr}\,\mathrm{sign}\,H=N_+-N_-$$ is the sum of the same numbers for the blocks, which is (5). Lemma 1(a) is algebra and applies.

For the special case the diagonal blocks are $$H_{xx}=-m_0\gamma_5$$, with eigenvalues $$-m_0$$ on $$\mathcal V_{x,R}$$ and $$+m_0$$ on $$\mathcal V_{x,L}$$. Equal dimensions give $$N_+=N_-$$. For the structure of $$D_{\rm ov}$$, let $$P_+$$ be the Riesz projection $$\frac1{2\pi i}\oint(z-H)^{-1}dz$$ around $$+m_0$$. The resolvent $$(z-H)^{-1}$$ is block lower triangular with diagonal blocks $$(z+m_0\gamma_5)^{-1}$$, so $$P_+$$ is block lower triangular with diagonal blocks $$\tfrac12(1-\gamma_5)$$. Hence $$\varepsilon=2P_+-1$$ is block lower triangular with diagonal blocks $$-\gamma_5$$, and $$D_{\rm ov}=m_0(1+\gamma_5\varepsilon)$$ has diagonal blocks $$m_0(1-\gamma_5\gamma_5)=0$$. A block lower triangular matrix with zero diagonal blocks satisfies $$D_{\rm ov}^N=0$$. $$\square$$
</div>

The step that carries the content is the first one: whatever the off-diagonal part does, it cannot move an eigenvalue of a triangular matrix. The Ginsparg–Wilson relation holds, so the operator is exactly chiral, but the structure it is chiral about is empty. In particular a nonzero index of a retarded operator can only come from on-site terms: formula (5) is a sum of local contributions that see neither the order nor the gauge field.

The theorem covers more than the signed-BD kernel. It applies to every retarded operator in every dimension: a Weyl block $$\sum_y a_{xy}\,\bar\sigma\cdot(x-y)$$ with $$y\prec x$$, any weights $$a_{xy}$$, any link variables, and a retarded Wilson term with no diagonal. It applies to advanced operators by reversing the order.

*Numerical check.* We sprinkled $$72$$ points in a two-dimensional causal diamond, built $$D_\pm$$ as in Definition 1 with independent random U(1) phases on every relation, and formed $$H=\gamma_5(D-m_0)$$ for $$m_0=0.5,1,2$$. The matrix sign function (Newton iteration) gave an index $$-\tfrac12\mathrm{Tr}\,\mathrm{sign}\,H$$ of modulus below $$10^{-13}$$ in all three cases, the Ginsparg–Wilson residual was $$\le10^{-13}$$, the largest entry of $$D_{\rm ov}$$ on or above the block diagonal was $$\le10^{-11}$$, and adding a chirally even retarded Wilson term with different strengths on the two chiralities left the index at $$3\times10^{-14}$$. Because $$H$$ is defective, its eigenvalues returned by a numerical eigensolver are accurate only to a fractional power of machine precision.

## 4. Self-adjointness for a chirality-swapping involution

The second and third classes come from the Lorentzian structure of the adjoint. For the signed-BD kernel the adjoint of the retarded operator is advanced and has the same functional form: since the entries are real,

$$
D_+^\dagger[x,y]=D_+[y,x]=\rho\,w_{n_{yx}}(u_y-u_x)\quad(x\prec y),\qquad\text{so}\qquad D_+^\dagger=-\widetilde D_+ , \tag{6}
$$

where $$\widetilde D_+[x,y]=\rho\,w_{n_{xy}}(u_x-u_y)$$ for $$y\succ x$$ is the advanced kernel of the same type. The kernel $$D_++\widetilde D_+=D_+-D_+^\dagger$$ is the symmetrised $$\partial_v$$ kernel on all causally related pairs (the Pauli–Jordan type), and it is anti-Hermitian. Nothing built from $$D_+$$ and its adjoint produces $$\partial_u$$: in Lorentzian signature the formal adjoint of $$\partial_v$$ is $$-\partial_v$$, whereas in Euclidean signature the adjoint of $$\partial_z$$ is $$-\partial_{\bar z}$$. A $$\gamma_5$$-Hermitian Dirac operator $$\begin{pmatrix}0&A\\-A^\dagger&0\end{pmatrix}$$ assembled from a frame-free block $$A\approx\partial_v$$ therefore has the same derivative in both off-diagonal blocks when $$A$$ is anti-Hermitian. The following theorem shows that any construction of this type has a spectrum symmetric about the imaginary axis.

<div class="theorem" markdown="1">
**Theorem 2** (Krein-self-adjoint operators). Let $$\gamma_5$$ be the grading on $$\mathcal V$$ and let $$J$$ be a matrix with

(K1) $$J=J^\dagger$$, $$J^2=1$$ and $$J\gamma_5=-\gamma_5J$$ (an involution that swaps the two chiralities).

Let $$X=X_0-m_0$$ with $$m_0$$ real and $$X_0$$ any matrix that is $$J$$-self-adjoint:

(K2) $$JX_0^\dagger J=X_0$$.

Put $$H=\gamma_5X$$ and $$J'=iJ\gamma_5$$. Then:

(a) $$J'$$ is a Hermitian involution and $$J'H^\dagger J'=-H$$. Consequently the spectrum of $$H$$ satisfies $$\mathrm{spec}\,H=-\overline{\mathrm{spec}\,H}$$ with equal algebraic multiplicities. If $$H$$ has no purely imaginary eigenvalue then $$\mathrm{Tr}\,\mathrm{sign}\,H=0$$ and $$\mathrm{ind}\,D_{\rm ov}=0$$. Otherwise the index (3) is undefined.

(b) For every polynomial or convergent power series $$f$$ with real coefficients, $$\mathrm{Tr}\,\gamma_5 f(X)\in i\mathbb R$$. Every real spectral asymmetry of $$X$$ vanishes.

(c) On any $$\gamma_5$$-graded subspace $$\mathcal W=\mathcal W_R\oplus\mathcal W_L$$ the Hermitian form $$\langle w,Jw'\rangle$$ has signature zero.
</div>

<div class="proof" markdown="1">
*Proof.* (a) Since $$J\gamma_5=-\gamma_5J$$, $$(iJ\gamma_5)^\dagger=-i\gamma_5J=iJ\gamma_5$$ and $$(iJ\gamma_5)^2=-J\gamma_5J\gamma_5=J^2\gamma_5^2=1$$. Then

$$
J'H^\dagger J'=-J\gamma_5X^\dagger\gamma_5J\gamma_5=J\gamma_5X^\dagger J=-\gamma_5JX^\dagger J=-\gamma_5X=-H ,
$$

where we used $$\gamma_5J\gamma_5=-J$$, then $$J\gamma_5=-\gamma_5J$$, then (K2) for $$X=X_0-m_0$$ (real $$m_0$$ does not affect it). So $$H^\dagger$$ is similar to $$-H$$ and the two have the same eigenvalues with multiplicity. The eigenvalues of $$H^\dagger$$ are the conjugates of those of $$H$$, which gives the symmetry. If no eigenvalue is purely imaginary, the map $$\lambda\mapsto-\bar\lambda$$ exchanges the eigenvalues in $$\mathrm{Re}\,\lambda>0$$ with those in $$\mathrm{Re}\,\lambda<0$$, so $$N_+=N_-$$.

(b) Let $$t=\mathrm{Tr}\,\gamma_5f(X)$$. By cyclicity and $$J^2=1$$, $$t=\mathrm{Tr}\,J\gamma_5f(X)J=-\mathrm{Tr}\,\gamma_5f(JXJ)$$, and (K2) says $$JXJ=X^\dagger$$. For real coefficients $$f(X^\dagger)=f(X)^\dagger$$, so $$t=-\mathrm{Tr}\,\gamma_5f(X)^\dagger=-\bar t$$.

(c) In an orthonormal basis adapted to $$\mathcal W_R\oplus\mathcal W_L$$, $$J$$ is off-diagonal because it swaps the chiralities, so the form has the matrix $$\begin{pmatrix}0&P\\P^\dagger&0\end{pmatrix}$$ with eigenvalues $$\pm$$ the singular values of $$P$$ and zeros. Its signature is zero. $$\square$$
</div>

The heart of the proof is the choice of $$J'$$ in (a); the rest is bookkeeping. Part (a) is a spectral-symmetry statement of the kind familiar from Krein-space theory [[S]](#ref-s) and from the symmetry classes of overlap fermions [[KW]](#ref-kw). Its use here is to identify the right involution for a frame-free Lorentzian kernel.

<div class="corollary" markdown="1">
**Corollary 1** (Hermitian and anti-Hermitian Weyl blocks). Let $$A$$ be any square matrix with $$A^\dagger=-A$$ or $$A^\dagger=A$$, let $$W$$ be any Hermitian matrix, and let $$m_0$$ be real. Let $$\mathcal V=\mathbb C^2\otimes\mathbb C^n$$ with $$\gamma_5=\sigma_z\otimes1$$, and put

$$
H=\sigma_z\otimes(W-m_0)+\begin{cases}i\sigma_y\otimes A,&A^\dagger=-A,\\ \sigma_x\otimes A,&A^\dagger=A.\end{cases}
$$

This is $$\gamma_5(X-m_0)$$ for the Dirac operator $$\begin{pmatrix}0&A\\-A^\dagger&0\end{pmatrix}$$ with a Hermitian Wilson term $$W$$ that acts identically on both chiralities. Then $$H$$ is Hermitian, $$\mathrm{spec}\,H=-\mathrm{spec}\,H$$ and, if $$H$$ is invertible, $$\mathrm{ind}\,D_{\rm ov}=0$$ for every $$m_0$$ and $$W$$.
</div>

<div class="proof" markdown="1">
*Proof.* Both $$\sigma_x\otimes1$$ and $$\sigma_y\otimes1$$ anticommute with $$\gamma_5=\sigma_z\otimes1$$, so (K1) holds for either. For $$A^\dagger=-A$$ the Dirac operator is $$\sigma_x\otimes A$$, $$X=1\otimes(W-m_0)+\sigma_x\otimes A$$, $$X^\dagger=1\otimes(W-m_0)-\sigma_x\otimes A$$, and with $$J=\sigma_y\otimes1$$ one has $$JX^\dagger J=X$$ because $$\sigma_y\sigma_x\sigma_y=-\sigma_x$$. For $$A^\dagger=A$$ the Dirac operator is $$i\sigma_y\otimes A$$, $$X^\dagger$$ contains $$-i\sigma_y\otimes A$$, and with $$J=\sigma_x\otimes1$$ one has $$\sigma_x(-i\sigma_y)\sigma_x=i\sigma_y$$, so again $$JX^\dagger J=X$$. Thus (K2) holds and Theorem 2(a) applies; the involution $$J'=iJ\gamma_5$$ is $$\mp\sigma_x\otimes1$$ resp. $$\sigma_y\otimes 1$$ up to sign, and it anticommutes with $$H$$. Because $$H$$ is Hermitian its spectrum is real, and the symmetry of the spectrum gives $$\mathrm{Tr}\,\mathrm{sign}\,H=0$$ whenever $$H$$ is invertible. $$\square$$
</div>

The hypothesis that $$W$$ acts identically on both chiralities is used: a one-sided Wilson term adds $$\sigma_z\sigma_{\rm diag}\otimes W$$ which commutes with $$J$$ in the wrong way, no involution exists, and the theorem says nothing. This is the case in which the Wilson term is chirally asymmetric, and we make no claim about it.

The two corollaries cover the frame-free Lorentzian Dirac operators one can write down that have the adjoint structure (6). For the signed-BD kernel, $$A=D_+-D_+^\dagger$$ is anti-Hermitian, and the Krein-self-adjoint part of $$iK$$ for $$K=\begin{pmatrix}0&D_+\\D_-&0\end{pmatrix}$$ and $$J=\sigma_x$$ is $$\tfrac i2\begin{pmatrix}0&D_+-D_+^\dagger\\D_--D_-^\dagger&0\end{pmatrix}$$, which is of the form in Theorem 2.

*Why the Lorentzian spectrum meets the imaginary axis.* The symbol of $$H$$ for the Krein-self-adjoint operator with Wilson symbol $$a(k)=r\widehat W(k)-m_0$$ is $$\begin{pmatrix}a&-k_v\mathrm{Re}\,h\\k_u\mathrm{Re}\,h&-a\end{pmatrix}$$, from the mean symbols of Proposition 1 (the symbol of $$D_+-D_+^\dagger$$ is $$2i\,\mathrm{Im}\,S=2ik_v\mathrm{Re}\,h$$), and its eigenvalues satisfy $$\lambda^2=a^2-k_uk_v(\mathrm{Re}\,h)^2$$. For timelike momenta with $$k_uk_v(\mathrm{Re}\,h)^2>a^2$$ (for small momenta $$\mathrm{Re}\,h\simeq1$$) the eigenvalues are purely imaginary, so the sign function is undefined on an open set of momenta; in Euclidean signature $$\lambda^2=a^2+\lvert k\rvert ^2>0$$ and there is no such set. This is a derivation at the level of mean symbols, and it explains the numerical observation below.

*Numerical check.* On the same $$72$$-point diamond, with $$W=\tfrac12(B+B^\dagger)$$ for $$B$$ the signed-BD scalar kernel with random link phases, $$r=\pm1$$ and $$m_0=0.45$$: for both signs of the Hermitian case in Corollary 1, $$JHJ+H$$ vanished to machine precision and the index was below $$2\times10^{-14}$$ in modulus for $$(r,m_0)=(1,0.45),(-1,0.45),(0.6,1.3)$$. For the Krein-self-adjoint operator of the previous paragraph, $$J'H^\dagger J'+H$$ vanished, the spectrum of $$H$$ was symmetric under $$\lambda\mapsto-\bar\lambda$$ with a maximal mismatch of $$9\times10^{-14}$$ between the two multisets, $$8$$ of the $$144$$ eigenvalues lay on the imaginary axis to within $$10^{-9}$$ (so the index is undefined), and $$\mathrm{Tr}\,\gamma_5X^3=\mp14.69\,i$$ was purely imaginary for $$r=\pm1$$, as part (b) requires.

*Four dimensions.* Both statements are algebra and do not refer to the dimension. What depends on the dimension is which operators fall under them. A Weyl block $$A[x,y]=c_{xy}\,\bar\sigma\cdot(x-y)$$ with real weights $$c_{xy}$$ has $$A^\dagger[x,y]=-c_{yx}\,\bar\sigma\cdot(x-y)$$ because $$\bar\sigma^\mu$$ is Hermitian, so it is anti-Hermitian when the weights are symmetric, $$c_{xy}=c_{yx}$$, and Hermitian when they are antisymmetric. A retarded kernel is neither, but its symmetrisation $$A_{\rm ret}-A_{\rm ret}^\dagger$$ is anti-Hermitian and falls under Corollary 1 in four dimensions exactly as in two. The retarded block alone falls under Theorem 1. We did not compute four-dimensional spectra of these operators.

## 5. Finite Atiyah–Patodi–Singer indices

In the continuum, the Dirac operator on a Lorentzian region between two Cauchy surfaces is Fredholm when the initial data are restricted to the positive spectral subspace of a boundary Dirac operator on one surface and the final data to the negative subspace on the other, and its index is the Atiyah–Patodi–Singer expression [[BS]](#ref-bs). A discrete analogue replaces the surfaces by antichains of the causal set. The proposition below shows that, in the simplest finite definition, the index does not depend on the bulk operator.

<div class="proposition" markdown="1">
**Proposition 3** (finite APS reduction; any dimension). Let $$D:\mathbb C^N\to\mathbb C^N$$ be any linear map. Let $$S_0,S_1$$ be disjoint subsets of the $$N$$ points, of sizes $$n_0,n_1$$, with Hermitian operators $$A_0$$ on $$\mathbb C^{S_0}$$ and $$A_1$$ on $$\mathbb C^{S_1}$$. Let $$\mathcal D=\{\psi\in\mathbb C^N: P_{<0}(A_0)\psi\!\upharpoonright_{S_0}=0,\ P_{>0}(A_1)\psi\!\upharpoonright_{S_1}=0\}$$ and $$\mathrm{ind}_{\rm APS}=\dim\ker(D\!\upharpoonright_{\mathcal D})-\dim\mathrm{coker}(D\!\upharpoonright_{\mathcal D})$$, the cokernel being taken in the full codomain $$\mathbb C^N$$. Write $$n_\pm(A)$$ for the numbers of positive and negative eigenvalues, $$h(A)=\dim\ker A$$ and $$\eta(A)=n_+-n_-$$. Then

(a) $$\mathrm{ind}_{\rm APS}=-n_-(A_0)-n_+(A_1)$$, independent of $$D$$.

(b) $$\mathrm{ind}_{\rm APS}+\tfrac12(n_0+n_1)=\tfrac12[\eta(A_0)-\eta(A_1)]+\tfrac12[h(A_0)+h(A_1)]$$.

(c) Exchanging the roles of $$S_0$$ and $$S_1$$ (order reversal) changes the sign of the $$\eta$$ terms.

(d) Suppose $$A_i$$ is a function of link variables $$U$$ on a neighbourhood $$\mathcal N_i$$ of $$S_i$$ that is covariant, $$A_i(U^g)=gA_i(U)g^{-1}$$ for diagonal unitary $$g$$, and that the background field is pure gauge on $$\mathcal N_0\cup\mathcal N_1$$ (every closed path there has trivial holonomy). Then the spectra of $$A_0$$ and $$A_1$$, and hence $$\mathrm{ind}_{\rm APS}$$, are independent of the flux $$Q$$ through the region between them.
</div>

<div class="proof" markdown="1">
*Proof.* (a) The conditions defining $$\mathcal D$$ are $$r=n_-(A_0)+n_+(A_1)$$ linear constraints, each a nonzero linear functional supported on $$S_0$$ or $$S_1$$; the supports are disjoint and, within each set, the functionals are the orthonormal eigenvectors, so the constraints are independent and $$\dim\mathcal D=N-r$$. For any linear map from a space of dimension $$N-r$$ to $$\mathbb C^N$$, rank–nullity gives $$\dim\ker-\dim\mathrm{coker}=(N-r)-N=-r$$. (b) Substitute $$n_0=n_++n_-+h$$ for each set. (c) Reversal imposes $$P_{<0}(A_1)$$ and $$P_{>0}(A_0)$$. (d) Pure gauge on $$\mathcal N_i$$ means that the link variables there are those of the trivial field transformed by a diagonal unitary $$g$$, which exists because every holonomy is trivial (define $$g$$ by transport from a base point); covariance gives $$A_i(U)=gA_i(1)g^{-1}$$, which has the spectrum of $$A_i(1)$$. $$\square$$
</div>

The proposition concerns this definition, with the boundary condition imposed on the domain only and the full space as the codomain. It does not show that no discrete APS-type index exists. It shows that on finite antichains the index counts boundary eigenvalues and is blind to the bulk, so a bulk-sensitive index needs an additional ingredient: in the continuum it is the evolution operator identifying two infinite-dimensional boundary spaces through the bulk. A closed loop of Hermitian matrices also has zero spectral flow in finite dimension, so the continuum relation between index and spectral flow cannot be reproduced by a finite family. If the flux is put into the boundary operator by hand, through a holonomy lift, the number $$Q$$ appears, but it appears equally when the bulk block is replaced by zero. The time-domain-wall eta invariant used for the lattice index [[F]](#ref-f) is a spectral asymmetry of a Hermitian operator, so for a chirality-blind Hermitian mass and an anti-Hermitian Weyl block the argument of Corollary 1 makes it vanish as well; the variant with a non-Hermitian retarded block is not covered, and we have not computed it.

Numerically, we checked (a) on five random maps with a random sub-domain structure (some of the maps retarded), with $$N=40$$, $$n_0=7$$, $$n_1=9$$; the computed difference $$\dim\ker-\dim\mathrm{coker}$$ equalled $$-n_-(A_0)-n_+(A_1)$$ each time.

## 6. An internal compact index

A chirality index can be carried by an internal compact space $$K$$, as in Kaluza–Klein compactification: the index of the Dirac operator on $$K$$ in a flux fixes how many one-handed zero modes the external space sees. Take $$M$$ a causal set with a retarded block $$D_M$$ odd under a grading $$\gamma_M$$, and $$K$$ a point set with a Ginsparg–Wilson operator $$D_K$$ and grading $$\gamma_K$$, for instance the operator of Example 1 on a flux torus. Define

$$
D=D_M\otimes1+\gamma_M\otimes D_K,\qquad \Gamma=\gamma_M\otimes\gamma_K . \tag{7}
$$

<div class="proposition" markdown="1">
**Proposition 4** (exact reduction; any dimension). Assume $$\gamma_KD_K+D_K\gamma_K=m_0^{-1}D_K\gamma_KD_K$$ and $$\gamma_MD_M=-D_M\gamma_M$$. Then $$\ker D_K$$ is invariant under $$\gamma_K$$, so it splits into $$n_+$$ and $$n_-$$ modes of chirality $$+1$$ and $$-1$$. The subspace $$\mathcal V_M\otimes\ker D_K$$ is invariant under $$D$$, and on it $$D=D_M\otimes1$$. On $$\mathcal V_M\otimes\ker D_K$$ the operator $$\Gamma$$ acts as $$\pm\gamma_M$$ on the $$\gamma_K=\pm1$$ modes. Hence the zero-mode sector of $$D$$ consists of $$n_+$$ copies of $$D_M$$ with $$\Gamma=\gamma_M$$ and $$n_-$$ copies of $$D_M$$ with $$\Gamma=-\gamma_M$$, in every realisation, and $$n_+-n_-=\mathrm{ind}\,D_K$$ when $$H_K$$ is Hermitian (Lemma 1).
</div>

<div class="proof" markdown="1">
*Proof.* If $$D_K\phi=0$$ the Ginsparg–Wilson relation gives $$D_K\gamma_K\phi=-\gamma_KD_K\phi+m_0^{-1}D_K\gamma_KD_K\phi=0$$, so $$\gamma_K\phi\in\ker D_K$$ and $$\gamma_K$$ is diagonalisable there. On $$\psi\otimes\phi$$ with $$\phi\in\ker D_K$$ the second term of $$D$$ vanishes, leaving $$(D_M\psi)\otimes\phi$$; $$\Gamma$$ acts as $$\gamma_M\otimes\gamma_K$$ and for $$\gamma_K\phi=\pm\phi$$ this is $$\pm\gamma_M\otimes1$$. $$\square$$
</div>

For (7) with $$K$$ a flux torus carrying $$Q$$ units, the index bookkeeping is exact: $$K$$ contributes $$\lvert Q\rvert $$ zero modes of one chirality. For the operator of Example 1 on a torus of $$60$$ points with $$Q=2$$ and a random strictly lower-triangular $$D_M$$ on $$5$$ elements, we verified the three statements numerically: the Ginsparg–Wilson residual was $$1.6\times10^{-15}$$, the index was $$-2$$ with two zero modes of chirality $$-1$$, $$\lVert D_K\gamma_K\phi\rVert $$ was $$5\times10^{-16}$$, $$\lVert D(\psi\otimes\phi)-(D_M\psi)\otimes\phi\rVert $$ was $$4\times10^{-15}$$, and $$\Gamma$$ acted as $$-\gamma_M$$ on the zero-mode sector to $$7\times10^{-15}$$.

Proposition 4 moves the whole chirality problem to $$M$$: the zero-mode sector sees only the one-handed block $$D_M$$ of the retarded operator, so the internal factor fixes the multiplicity and the sign and contributes nothing to the propagation of the mode along $$M$$. Whether $$D_M$$ propagates a massless one-handed mode is then a question about the two-dimensional block, which Section 7 addresses for the signed-BD kernel. The retarded operator $$D_M$$ has overlap index zero by Theorem 1, so in this construction the chirality of the product theory is carried by $$K$$ alone.

## 7. Noise and smearing of the two-dimensional kernel

Proposition 1 is a statement about the mean of $$D_+$$. A single sprinkling gives a random operator whose entries fluctuate, because $$n_{xy}$$ does. We compute the size of the fluctuation of one row, and the effect of smearing, for the kernel of Definition 1. Everything in this section is two-dimensional and specific to the layer weights $$(1,-2,1)$$.

Take $$\rho=1$$ and a rectangular region, and let $$x$$ be a point whose past in the region is the rectangle $$[0,a]\times[0,b]$$ in the displacements $$(U,V)$$. Given points at $$x$$ and $$y$$, the entry $$D^\varepsilon_+[x,y]=\varepsilon^2f(n_{xy},\varepsilon)U$$ is a function of $$n_{xy}\sim\mathrm{Poisson}(UV/2)$$. Let $$N_\varepsilon(x)^2=\int\mathrm{Var}\big(D_+^\varepsilon[x,y]\mid x,y\in\mathcal P\big)\,\tfrac12\,dU\,dV$$ be the total variance carried by the entries of the row, and $$M_\varepsilon(x)^2$$ the squared norm of the mean row, $$\int\big(\mathbb E[D_+^\varepsilon[x,y]\,\lvert \,x,y]\big)^2\tfrac12\,dU\,dV$$.

<div class="proposition" markdown="1">
**Proposition 5** (row noise; two dimensions). As $$b\to\infty$$ with $$a$$ fixed,

$$
\mathbb E\,\lVert{\rm row}_x\rVert^2=\tfrac12\varepsilon^3\,\frac{6}{(2-\varepsilon)^5}\,a^2+O(b^{-2}),\qquad M_\varepsilon^2=\tfrac12\varepsilon^3\,\frac{3}{16}\,a^2+O(b^{-2}),\qquad N_\varepsilon^2=\tfrac12\varepsilon^3\Big[\frac{6}{(2-\varepsilon)^5}-\frac3{16}\Big]a^2+O(b^{-2}). \tag{8}
$$

For $$\varepsilon=1$$ these are $$3a^2$$, $$\tfrac3{32}a^2$$ and $$\tfrac{93}{32}a^2$$; for small $$\varepsilon$$, $$N_\varepsilon^2\simeq\tfrac{15}{64}\varepsilon^4a^2$$.
</div>

<div class="proof" markdown="1">
*Proof.* By the Mecke formula $$\mathbb E\lVert {\rm row}_x\rVert ^2=\tfrac12\varepsilon^4\int_0^a dU\,U^2\int_0^b dV\,\mathbb E[f(n,\varepsilon)^2]$$, with $$n\sim\mathrm{Poisson}(UV/2)$$. The substitution $$\lambda=UV/2$$ turns the $$V$$ integral into $$(2/U)\int_0^{Ub/2}\mathbb E[f^2](\lambda)\,d\lambda$$, and for $$b\to\infty$$ the upper limit may be removed with an error of order $$b^{-2}$$, since $$\mathbb E[f^2](\lambda)$$ decays exponentially. Because $$\int_0^\infty\frac{e^{-\lambda}\lambda^n}{n!}d\lambda=1$$, $$\int_0^\infty\mathbb E[f^2](\lambda)\,d\lambda=\sum_nf(n,\varepsilon)^2$$. Writing $$q=1-\varepsilon$$, $$f(n,\varepsilon)=q^{n-2}\big(q^2-2n\varepsilon q+\tfrac12 n(n-1)\varepsilon^2\big)$$, and the sum is a combination of $$\sum_n n^{\underline j}q^{2n}=j!\,q^{2j}/(1-q^2)^{j+1}$$; a symbolic evaluation of this combination gives $$\sum_nf^2=6/[\varepsilon(2-\varepsilon)^5]$$ exactly (we confirmed it also by direct summation at $$\varepsilon=1,0.7,0.3,0.1$$ to $$10^{-15}$$). Then $$\int_0^adU\,U=a^2/2$$ gives the first formula. For the mean row, $$\mathbb E[f(n,\varepsilon)]=f_0(\varepsilon\lambda)$$ with $$f_0(z)=e^{-z}(1-2z+z^2/2)$$, and $$\int_0^\infty f_0(z)^2dz=\tfrac3{16}$$ gives the second. The variance is the difference. $$\square$$
</div>

The sums, not the geometry, produce the numbers $$3$$ and $$\tfrac3{32}$$: at $$\varepsilon=1$$ the entries of a row have a total variance about thirty times the squared norm of the mean row, and the ratio $$N_\varepsilon^2/M_\varepsilon^2=32/(2-\varepsilon)^5-1$$ decreases to zero like $$5\varepsilon/2$$ as $$\varepsilon\to0$$. We checked the first formula by sampling: at $$a=b=10$$, $$\varepsilon=1$$ the mean of $$\lVert {\rm row}_x\rVert ^2$$ over $$3000$$ sprinklings was $$292.6\pm3.8$$, against $$299.2$$ from the exact finite-$$b$$ integral and $$300$$ from (8); at $$a=b=10$$, $$\varepsilon=0.5$$ we found $$4.88\pm0.06$$ against $$4.93$$ and $$4.94$$; at $$a=b=14$$, $$\varepsilon=0.3$$ (1500 sprinklings) $$1.094\pm0.020$$ against $$1.117$$ and $$1.118$$.

How large is this noise compared with the signal? A plane wave of unit amplitude and wave number $$k\sim\pi/a$$ has mean action of order $$k$$. If correlations between entries are ignored, the fluctuation of $$(D_+^\varepsilon\psi)(x)$$ is of order $$N_\varepsilon\simeq\sqrt{15/64}\,\varepsilon^2a$$ for small $$\varepsilon$$, so the ratio of fluctuation to signal is of order $$\varepsilon^2a^2$$ in units of the discreteness length. This comparison is heuristic: it keeps only the diagonal part of the variance and drops the correlations between entries that share interval points. Taming the noise would require $$\varepsilon\lesssim 1/a$$, that is a nonlocality length $$(\varepsilon\rho)^{-1/2}$$ of order $$a^{1/2}$$ discreteness lengths, growing with the region.

<div class="proposition" markdown="1">
**Proposition 6** (smearing squeeze; two dimensions). Fix $$K>0$$ and $$\delta>0$$. The fraction of the box $$[-K,K]^2$$ of momenta (in units $$\rho=1$$) on which $$\lvert S_\varepsilon(k)\rvert <\delta$$ tends to $$1$$ as $$\varepsilon\to0$$.
</div>

<div class="proof" markdown="1">
*Proof.* By Proposition 1, $$\lvert S_\varepsilon(k)\rvert =\lvert k_v\rvert \,\lvert h(-2k_uk_v/\varepsilon)\rvert \le C\lvert k_v\rvert \,\varepsilon^2/(4k_u^2k_v^2)$$ as soon as $$2\lvert k_uk_v\rvert /\varepsilon\ge1$$, which tends to zero at every point off the axes. The indicator of $$\{\lvert S_\varepsilon\rvert <\delta\}$$ is bounded by $$1$$ and tends to $$1$$ almost everywhere; dominated convergence gives the claim. $$\square$$
</div>

Smearing at a fixed momentum resolution therefore removes the noise (Proposition 5) only by sending the mean symbol to zero over the whole resolved box. The statement depends on the box being held at the discreteness scale. If momenta are resolved only down to the nonlocality scale $$(\varepsilon\rho)^{-1/2}$$, the box is rescaled with $$\varepsilon$$ and the squeeze disappears; this is the route listed in Section 8.

*A numerical observation on propagation.* For Proposition 4 the relevant question is whether the unsmeared block $$D_+$$ annihilates a packet of the left-moving type that the continuum annihilates. On two-dimensional sprinklings of the square $$[0,30]^2$$ in $$(u,v)$$ we applied $$D_+$$, with its prefactor $$\rho$$, to the Gaussian packet $$f(u)$$ of width $$30/8$$ (annihilated by $$\partial_v$$) and to the packet $$g(v)$$ of the same width ($$\partial_vg\ne0$$). The quantity $$\lVert D_+\phi\rVert ^2/\lVert \phi\rVert ^2$$ was, for $$f$$ and $$g$$ respectively and averaged over five sprinklings (three at $$\rho=4$$): $$773\pm22$$ and $$892\pm95$$ at $$\rho=1$$; $$2665\pm170$$ and $$3124\pm130$$ at $$\rho=2$$; $$11430\pm240$$ and $$16040\pm590$$ at $$\rho=4$$. The continuum values are $$0$$ and $$1/(2\cdot(30/8)^2)=0.036$$. The block does not distinguish the two packets, both are acted on with a norm about $$10^4$$ times the continuum signal, and the squared norm grows roughly like $$\rho^2$$. We interpret this as the noise of Proposition 5 acting on the packet; the numbers depend on the packet width and on the normalisation.

## 8. What the theorems do not exclude

The theorems have hypotheses, and each hypothesis marks a route that remains open. We state the routes and the condition under which each would apply, without ranking them.

- Operators outside the four classes. A $$\gamma_5$$-Hermitian completion $$\begin{pmatrix}0&A\\-A^\dagger&0\end{pmatrix}$$ of a retarded block $$A=D_+$$ has an advanced block of the same type in the second slot; it is not retarded, its Weyl block is not (anti-)Hermitian, and it is not Krein-self-adjoint for $$J=\sigma_x$$ or $$\sigma_y$$. Neither the theorems nor any computation of ours constrain its index. In four dimensions the literal pairing of a $$\bar\sigma\cdot\Delta$$ block with a $$\sigma\cdot\Delta$$ block, symmetrised or not, is covered only when both blocks are retarded (Theorem 1) or when the blocks are (anti-)Hermitian.
- Chirally asymmetric Wilson terms (Corollary 1 needs the same Wilson term on both chiralities). The index of such an operator is not constrained by the involution argument.
- An involution that commutes with $$\gamma_5$$. Theorem 2 needs $$J\gamma_5=-\gamma_5J$$. With $$J=\gamma_5$$ one has $$\gamma_5$$-Hermiticity, which is the Euclidean overlap structure of Example 1; there the question is whether a frame-free kernel with this property exists.
- Indices that are not spectral asymmetries of one operator, in particular a non-Hermitian time-domain-wall variant and an APS-type index in which the bulk identifies infinite-dimensional boundary spaces. Proposition 3 covers only the finite definition with domain-only constraints.
- A frame. The Euclidean overlap of Example 1 uses a Wick axis. Whether a frame in the matter sector of a Lorentzian theory is acceptable is the subject of [Where a preferred frame leaks into low-energy physics](/quantum-gravity/lorentz-violation-leakage/) and of [Frames, Wilson terms and reflection positivity for fermions on random points](/quantum-gravity/frames-and-reflection-positivity/).
- Nonlocality scales that grow with the region (Proposition 6). The estimates of Section 7 are for a fixed resolution box. With $$\varepsilon\sim1/a$$, the near-zero fraction and the propagation of a one-handed packet are not decided by anything here.
- Smeared $$D_M$$ and genuinely product-free substrates in Proposition 4. The reduction is exact for a product of two point sets. A Poisson process on $$M\times K$$ is not of this form.
- Four-dimensional spectra. The algebra is dimension-free, but we computed nothing about the near-zero spectrum or the propagation of four-dimensional retarded blocks.

## 9. Open points

Whether there is a frame-free operator on a sprinkling, necessarily outside the four classes, with a non-zero index is open; no theorem excludes it and no example is known. The Euclidean operator of Example 1 shows that the lattice index theorem holds on Poisson points once a Wick axis is used, and whether its Euclidean correlators continue to a Lorentzian theory depends on reflection positivity, discussed on the companion page. For the internal-index route the open point is whether a smeared block with a nonlocality length growing with the region propagates a one-handed massless mode and is quiet and clean on its own scale. A Lorentzian index in which the bulk identifies infinite-dimensional boundary spaces across a causal interval of a sprinkling has no discrete definition that we know of.

## References

1. <a id="ref-gw"></a>Ginsparg, Wilson, "A remnant of chiral symmetry on the lattice," Phys. Rev. D 25 (1982) 2649.
2. <a id="ref-nn"></a>Nielsen, Ninomiya, "Absence of neutrinos on a lattice," Nucl. Phys. B 185 (1981) 20.
3. <a id="ref-n1"></a>Neuberger, "Exactly massless quarks on the lattice," [arXiv:hep-lat/9707022](https://arxiv.org/abs/hep-lat/9707022), Phys. Lett. B 417 (1998) 141.
4. <a id="ref-n2"></a>Neuberger, "More about exactly massless quarks on the lattice," [arXiv:hep-lat/9801031](https://arxiv.org/abs/hep-lat/9801031), Phys. Lett. B 427 (1998) 353.
5. <a id="ref-l"></a>Lüscher, "Exact chiral symmetry on the lattice and the Ginsparg–Wilson relation," [arXiv:hep-lat/9802011](https://arxiv.org/abs/hep-lat/9802011), Phys. Lett. B 428 (1998) 342.
6. <a id="ref-ass"></a>Aslanbeigi, Saravani, Sorkin, "Generalized causal set d'Alembertians," [arXiv:1403.1622](https://arxiv.org/abs/1403.1622), JHEP 06 (2014) 024.
7. <a id="ref-bhs"></a>Bombelli, Henson, Sorkin, "Discreteness without symmetry breaking: a theorem," [arXiv:gr-qc/0605006](https://arxiv.org/abs/gr-qc/0605006), Mod. Phys. Lett. A 24 (2009) 2579.
8. <a id="ref-bs"></a>Bär, Strohmaier, "An index theorem for Lorentzian manifolds with compact spacelike Cauchy boundary," [arXiv:1506.00959](https://arxiv.org/abs/1506.00959), Amer. J. Math. 141 (2019) 1421.
9. <a id="ref-f"></a>Fukaya, Kawai, Matsuki, Mori, Nakayama, Onogi, Yamaguchi, "Atiyah–Patodi–Singer index on a lattice," [arXiv:1910.09675](https://arxiv.org/abs/1910.09675), Prog. Theor. Exp. Phys. (2020).
10. <a id="ref-kw"></a>Kimura, Watanabe, "Band flattening and overlap fermion," [arXiv:2309.12174](https://arxiv.org/abs/2309.12174).
11. <a id="ref-s"></a>Strohmaier, "On noncommutative and semi-Riemannian geometry," [arXiv:math-ph/0110001](https://arxiv.org/abs/math-ph/0110001), J. Geom. Phys. 56 (2006) 175.
12. <a id="ref-lp"></a>Last, Penrose, *Lectures on the Poisson Process*, Cambridge University Press (2017), chapter 4 (the Mecke formula).
