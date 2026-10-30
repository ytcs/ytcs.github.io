---
title: Obstructions to a chirality index for frame-free Weyl operators
date: 2026-10-29
category: matter
order: 2
description: "The overlap index of an operator on a causal set, with any weights, is zero or undefined when the operator is retarded with a chirally even on-site part or self-adjoint for a matrix that exchanges the two chiralities; a finite boundary-projected index is nonzero in general but depends on boundary data alone."
status: "Lemma 1, Theorems 1 and 2, Corollary 1 and Propositions 3 and 4 proved in any dimension. Propositions 1, 2, 5 and 6 (mean symbol, boost covariance, row noise, smearing) proved for one two-dimensional kernel. Computed in two dimensions: the Euclidean overlap index (twelve cases), eigenvalues near the imaginary axis on a 72-point diamond, and a propagation table."
excludes: "A nonzero overlap index for a retarded operator with chirally even on-site part, or for an operator self-adjoint for a chirality-exchanging involution (for example a Hermitian or anti-Hermitian Weyl block with a Hermitian chirally even Wilson term); a bulk-sensitive integer from the finite boundary-projected index of Proposition 3. Chirally asymmetric Wilson terms are not excluded."
---

A Weyl fermion on a lattice or on a random point set is certified chiral by an integer: the index of a Ginsparg–Wilson (overlap) Dirac operator, the number of zero modes of one handedness minus those of the other, which in a gauge field equals the flux through a torus up to sign. We ask whether such an integer exists for an operator on a Poisson sprinkling in Minkowski space that is built from the causal order and the invariant interval alone, so that no preferred frame enters. For two classes of operator, with any weights on any finite causal set, it is zero or undefined: the retarded operators, supported on pairs of points in causal order (Theorem 1), and the operators self-adjoint for a matrix that exchanges the two chiralities, which include those whose off-diagonal (Weyl) block is Hermitian or anti-Hermitian (Theorem 2 and Corollary 1). Both statements are made for a Wilson term, which gives the doublers a mass, that acts identically on the two chiralities.

For retarded operators only the on-site part of the Wilson term enters, and Theorem 2 asks only that the whole operator be self-adjoint for the exchanging matrix, which holds for the Wilson-term part if that term is Hermitian and acts identically on both chiralities. A third construction, an index defined by projecting initial and final data onto spectral subspaces of boundary operators on finite antichains (sets of mutually unrelated points), gives an integer that is generally nonzero but depends on the boundary operators alone (Proposition 3). A product with an internal point set whose overlap operator has zero modes, for example a flux torus, gives copies of a retarded block whose grading is set by those zero modes, so that the chirality of the product is carried by the internal factor (Proposition 4). The proofs are finite-dimensional linear algebra and use neither the sprinkling nor a frame. The two classes are defined algebraically; the signed Benincasa–Dowker kernel of Definition 1, which is frame-free, falls in the first, and its symmetrisation (Section 4) in the second. A Euclidean overlap operator on a Poisson torus shows the expected index in the twelve cases we computed (Example 1), at the price of a choice of Euclidean time axis.

The overlap construction is due to Neuberger [[N1]](#ref-n1), [[N2]](#ref-n2) and gives an exact solution of the Ginsparg–Wilson relation [[GW]](#ref-gw). Lüscher showed that the relation yields an exact index theorem on the lattice [[L]](#ref-l). Lorentz-invariant retarded kernels on causal sets were constructed by Aslanbeigi, Saravani and Sorkin [[ASS]](#ref-ass), and Bombelli, Henson and Sorkin proved that a sprinkling carries no preferred direction and no finite-valency graph [[BHS]](#ref-bhs). In the continuum, Bär and Strohmaier proved an index theorem for the Lorentzian Dirac operator on a compact globally hyperbolic region, with Atiyah–Patodi–Singer conditions on the two spacelike boundaries [[BS]](#ref-bs), and a lattice version of the Atiyah–Patodi–Singer index has been given through the eta invariant of a domain-wall operator [[F]](#ref-f). Kimura and Watanabe built the overlap operator for each symmetry class of the tenfold way and established the Ginsparg–Wilson relation for the charge-conjugation and time-reversal symmetries and a mod-two index theorem [[KW]](#ref-kw).

The theorems and Propositions 3 and 4 of Sections 2 to 6 hold in every dimension. Sections 1 and 7 concern the two-dimensional kernel of Definition 1, and Propositions 1, 2, 5 and 6, Example 1, the numerical statements and the symbol paragraph of Section 4 refer to two dimensions.

## 1. The frame-free retarded Weyl operator

Let $$\mathcal P$$ be a Poisson process of density $$\rho$$ in two-dimensional Minkowski space, ordered by $$y\prec x$$ when $$x-y$$ is future timelike. In null coordinates $$u=t-x$$, $$v=t+x$$ one has $$dt\,dx=\tfrac12\,du\,dv$$, and $$y\prec x$$ exactly when $$U=u_x-u_y>0$$ and $$V=v_x-v_y>0$$. A boost of rapidity $$\xi$$ multiplies $$(U,V)$$ by $$(e^{-\xi},e^{\xi})$$. For $$y\prec x$$ let $$n_{xy}$$ be the number of points of $$\mathcal P$$ strictly between $$y$$ and $$x$$. The pair $$(n_{xy},(U,V))$$ is what we call frame-free: the count is a property of the order, and the null displacements are only rescaled by a boost. No timelike unit vector is used.

<div class="definition" markdown="1">
**Definition 1** (signed-BD retarded Weyl operator). Let $$w_0=1$$, $$w_1=-2$$, $$w_2=1$$ and $$w_n=0$$ for $$n\ge 3$$. These are the layer coefficients of the two-dimensional Benincasa–Dowker d'Alembertian, in the convention of [mixed-Poisson and signed substrates](/quantum-gravity/signed-substrates/). Only the coefficients are used, so the signature and the overall sign of the d'Alembertian do not enter. For a point set $$\mathcal P$$ with $$N$$ elements put, for $$y\prec x$$,

$$
D_+[x,y]=\rho\,w_{n_{xy}}\,(u_x-u_y),\qquad D_-[x,y]=\rho\,w_{n_{xy}}\,(v_x-v_y),
$$

and zero when $$y\not\prec x$$. On $$\mathbb C^N\otimes\mathbb C^2$$, with the two components called $$R$$ and $$L$$ and $$\gamma_5=\mathrm{diag}(1,-1)$$, the retarded Dirac operator is $$D=\begin{pmatrix}0&D_+\\ D_-&0\end{pmatrix}$$. A smeared version replaces $$w_{n}$$ by $$\varepsilon^2 f(n,\varepsilon)$$ with $$f(n,\varepsilon)=\sum_k w_k\binom nk\varepsilon^k(1-\varepsilon)^{n-k}$$, the Poisson-thinning form of the generalised causal-set d'Alembertians of [[ASS]](#ref-ass). The value $$\varepsilon=1$$ is the unsmeared operator.
</div>

The operator $$D_+$$ is meant to approximate $$\partial_v$$ and $$D_-$$ to approximate $$\partial_u$$. The weights are chosen so that the Poisson average of $$w_n$$ over the number of points in an interval of mean size $$z=\rho UV/2$$, namely $$f_0(z)=e^{-z}(1-2z+z^2/2)$$, has vanishing zeroth and first moments, $$\int_0^\infty f_0\,dz=\int_0^\infty zf_0\,dz=0$$. Without this cancellation the sum over the past of a point does not converge. The zeroth moment gives $$\int f_0(\rho UV/2)\,dV=0$$ at fixed $$U$$, so in mean $$D_+$$ annihilates every function of $$u$$ alone, as $$\partial_v$$ does. The first moment makes the $$V$$-integral against $$e^{-ik_vV}$$ vanish quadratically in $$k_v/(\rho U)$$ at large $$U$$. The term linear in $$k_v$$ comes from the region $$U\sim k_v/\rho$$, and Proposition 1 gives its coefficient exactly, in the limit in which the region is large compared with the discreteness scale $$\rho^{-1/2}$$. The computation uses the Mecke formula, $$\mathbb E\sum_{y\in\mathcal P}g(y,\mathcal P)=\rho\int\mathbb E\,g(y,\mathcal P\cup\{y\})\,dy$$ [[LP]](#ref-lp). Applied to the points $$x$$ and $$y$$, it says that, given points at $$x$$ and $$y\prec x$$, the number of other points in the order interval between them is Poisson with mean $$\rho UV/2$$.

<div class="proposition" markdown="1">
**Proposition 1** (mean symbol; two dimensions). Let $$\psi(y)=e^{i(k_uu_y+k_vv_y)}$$ and let $$S_\varepsilon(k)$$ be the expectation of $$(D^{\varepsilon}_+\psi)(x)/\psi(x)$$ given a point at $$x$$, the sum over the past of $$x$$ being taken first over $$V$$ and then over $$U$$. Then

$$
S_\varepsilon(k)=i\,k_v\,h\!\Big(\frac{-2k_uk_v}{\varepsilon\rho}\Big),\qquad h(\omega)=1+\omega-\omega(2+\omega)\,e^{\omega}E_1(\omega)=2\int_0^\infty\frac{t\,e^{-\omega t}}{(1+t)^3}\,dt, \tag{1}
$$

where $$E_1$$ is the exponential integral. For $$k_uk_v<0$$ the argument $$\omega$$ is positive and (1) holds as written. For $$k_uk_v>0$$ one takes the boundary value $$\omega\to \omega+i0\,\mathrm{sgn}(k_u+k_v)$$ on the cut $$\omega<0$$. Moreover $$h(0)=1$$, $$h$$ has no zero on the real line (on either side of the cut), and $$\lvert h(\omega)\rvert \le C/(1+\omega^2)$$ for a constant $$C$$. In particular $$S_\varepsilon(k)=0$$ exactly on the null line $$k_v=0$$, and $$S_\varepsilon(k)\to i k_v$$ as $$k\to0$$.
</div>

<div class="proof" markdown="1">
*Proof.* Rescaling $$V$$ by $$\varepsilon\rho$$ reduces the claim to $$\varepsilon=\rho=1$$, with $$k_v$$ replaced by $$k_v/(\varepsilon\rho)$$ and an overall factor $$\varepsilon\rho$$ that cancels against it, so we take $$\varepsilon=\rho=1$$. The sum over the past becomes a one-dimensional integral by the Mecke formula, and the integral is evaluated on a rotated contour.

By the Mecke formula the expectation is $$\tfrac12\iint U\,f_0(UV/2)\,e^{-i(k_uU+k_vV)}\,dU\,dV$$ (for general $$\varepsilon$$ one averages $$f(n,\varepsilon)$$ over $$n\sim\mathrm{Poisson}(\lambda)$$ and obtains $$f_0(\varepsilon\lambda)$$). Write this as $$i\partial_{k_u}G$$ with $$G=\tfrac12\iint f_0(UV/2)\,e^{-i(k_uU+k_vV)}dU\,dV$$. With $$s=U/2$$ and $$\beta=ik_v$$ the $$V$$-integral of the term $$a_n(sV)^ne^{-sV}$$ of $$f_0$$, where $$(a_0,a_1,a_2)=(1,-2,\tfrac12)$$, equals $$a_n\,n!\,s^n/(s+\beta)^{n+1}$$, and

$$
\sum_n \frac{a_n\,n!\,s^n}{(s+\beta)^{n+1}}=\frac{(s+\beta)^2-2s(s+\beta)+s^2}{(s+\beta)^3}=\frac{\beta^2}{(s+\beta)^3}.
$$

Hence

$$
G=\int_0^\infty e^{-2ik_us}\,\frac{\beta^2}{(s+\beta)^3}\,ds ,
$$

which converges absolutely. For $$k_v=0$$ the integrand vanishes and $$S=0$$. If $$k_uk_v<0$$ one rotates the contour to $$s=\beta t$$, which sweeps a quadrant that contains neither the pole $$s=-\beta$$ nor a contribution from the arc at infinity, and finds $$G=\int_0^\infty e^{-\omega t}(1+t)^{-3}dt$$ with $$\omega=-2k_uk_v>0$$. Since $$S=i\partial_{k_u}G=ik_v\,(-2G'(\omega))$$ one gets the integral form of $$h$$, and integrating by parts twice gives the closed form in (1). For $$k_v>0$$, $$G$$ is analytic in $$k_u$$ in the lower half plane, which corresponds to $$\mathrm{Im}\,\omega>0$$; the formula, analytic in $$\omega$$ off the cut, agrees with $$G$$ on the real half line and therefore everywhere in that half plane. The real timelike values are the boundary values $$\omega+i0$$. For $$k_v<0$$ the same argument gives $$\omega-i0$$. This is the stated rule, since for $$k_uk_v>0$$ the sign of $$k_u+k_v$$ equals the sign of $$k_v$$.

For $$\omega>0$$ the integral form gives $$h>0$$ and $$h(\omega)\le 2\int_0^\infty t e^{-\omega t}dt=2/\omega^2$$. For $$\omega=-a\pm i0$$ with $$a>0$$ the identity $$E_1(-a\pm i0)=-\mathrm{Ei}(a)\mp i\pi$$ gives $$\mathrm{Im}\,h=\mp\pi a(2-a)e^{-a}$$, which vanishes only at $$a=2$$, where $$\omega(2+\omega)=0$$ and $$h=1+\omega=-1\neq0$$. So $$h$$ has no real zero. For the decay on the cut, rotate the integral form to the ray $$t=\tau e^{\mp 2\pi i/3}$$ for $$\omega=-a\pm i0$$, which is allowed because the pole of $$(1+t)^{-3}$$ at $$t=-1$$ is not crossed; there $$\lvert e^{-\omega t}\rvert =e^{-a\tau/2}$$ and $$\lvert 1+t\rvert \ge\sqrt3/2$$, so $$\lvert h(-a\pm i0)\rvert \le 2(2/\sqrt3)^3\,(4/a^2)=64/(3\sqrt3\,a^2)$$. Continuity of $$h$$ near $$\omega=0$$ gives the uniform bound. $$\square$$
</div>

The mean symbol therefore vanishes on the null line $$k_v=0$$ in the entry $$D_+$$ (and on $$k_u=0$$ in $$D_-$$) and has no other zero. Boost covariance alone restricts the further zeros of a frame-free symbol, which would be doublers.

<div class="proposition" markdown="1">
**Proposition 2** (boost covariance fixes the symbol on the null half-lines; two dimensions). Let $$D(p)$$ be a $$2\times2$$ matrix function of $$p=(p_u,p_v)$$, continuous at $$0$$ with $$D(0)=0$$, covariant under boosts: $$\Omega_\xi D(\Lambda_\xi p)\Omega_\xi^{-1}=D(p)$$ with $$\Lambda_\xi(p_u,p_v)=(e^{\xi}p_u,e^{-\xi}p_v)$$ and $$\Omega_\xi=\mathrm{diag}(e^{\xi/2},e^{-\xi/2})$$. Here $$\Lambda_\xi$$ is the action on momenta of the boost that multiplies $$(U,V)$$ by $$(e^{-\xi},e^{\xi})$$, so that $$p_uU+p_vV$$ is invariant. Then on the half-lines $$p_v=0$$ only the entry $$D_{21}$$ can be non-zero and it is linear in $$p_u$$; on the half-lines $$p_u=0$$ only $$D_{12}$$ can be non-zero and it is linear in $$p_v$$. Every other zero set of $$\det D$$ is a union of boost orbits, that is of hyperbolas $$p_up_v=\mathrm{const}\neq0$$.
</div>

<div class="proof" markdown="1">
*Proof.* Conjugation by $$\Omega_\xi$$ leaves the diagonal entries unchanged, multiplies $$D_{12}$$ by $$e^{\xi}$$ and $$D_{21}$$ by $$e^{-\xi}$$. On the ray $$p=\kappa(1,0)$$ one has $$\Lambda_\xi p=e^{\xi}p$$. Covariance gives $$D_{11}(e^{\xi}p)=D_{11}(p)$$, and the left side tends to $$D_{11}(0)=0$$ as $$\xi\to-\infty$$, so $$D_{11}=0$$ on the ray; the same holds for $$D_{22}$$. It gives $$D_{12}(e^{\xi}p)=e^{-\xi}D_{12}(p)$$, whose left side tends to $$0$$ as $$\xi\to-\infty$$ while the right side diverges unless $$D_{12}(p)=0$$. It gives $$D_{21}(e^{\xi}p)=e^{\xi}D_{21}(p)$$, which is linearity along the ray. The other null ray is treated with $$\xi\to+\infty$$, and exchanges the roles of $$D_{12}$$ and $$D_{21}$$. The determinant is boost invariant, so its zero set is a union of orbits, and the orbits of the boost group in the plane are the hyperbolas $$p_up_v=\mathrm{const}$$, the four open null half-lines and the origin. $$\square$$
</div>

The symbol of Definition 1, $$D(k)=\begin{pmatrix}0&ik_vh\\ ik_uh&0\end{pmatrix}$$ with $$h$$ evaluated at $$-2k_uk_v/\rho$$, satisfies the hypotheses (the invariant $$k_uk_v$$ is unchanged by $$\Lambda_\xi$$), with $$D_{21}\propto k_u$$ surviving on $$k_v=0$$ and $$D_{12}\propto k_v$$ on $$k_u=0$$. Its determinant $$k_uk_vh^2$$ vanishes only on the two null lines, because $$h$$ has no zero (Proposition 1). Doubling is therefore not what obstructs chirality for this kernel. The Nielsen–Ninomiya theorem [[NN]](#ref-nn) forces extra zeros through the periodicity of the Brillouin zone of a lattice, and the symbol of a frame-free kernel is not periodic.

Positive covariant weights cannot be used instead. A function $$\phi(\Delta^2)$$ of the invariant interval alone is never integrable over the plane unless it vanishes: in rapidity coordinates $$d^2\Delta=\tfrac12\,d\sigma\,d\theta$$ with $$\sigma=\Delta^2$$ and $$\theta\in\mathbb R$$, so $$\int\lvert \phi\rvert \,d^2\Delta=\infty$$ whenever $$\phi\ne0$$. The same holds on every hyperboloid in higher dimensions. A covariant kernel therefore has a symbol only through cancellations, and the layer weights $$(1,-2,1)$$ are such a cancellation.

## 2. The overlap operator and its index

Let $$X$$ be a linear operator on a finite-dimensional space $$\mathcal V=\mathcal V_R\oplus\mathcal V_L$$ with grading $$\gamma_5=\pm1$$ on $$\mathcal V_{R/L}$$, and let $$m_0>0$$ be a real number. Put

$$
H=\gamma_5\,(X-m_0),\qquad D_{\rm ov}=m_0\,\big(1+\gamma_5\,\Sigma\big),\qquad \Sigma=\mathrm{sgn}\,H . \tag{2}
$$

For a matrix $$H$$ with no eigenvalue on the imaginary axis, $$\mathrm{sgn}\,H=P_+-P_-$$ where $$P_\pm$$ are the spectral projections onto the generalised eigenspaces with $$\mathrm{Re}\,\lambda>0$$ and $$<0$$. Then $$\Sigma^2=1$$, $$\Sigma$$ commutes with $$H$$, $$\mathrm{Tr}\,\Sigma=N_+-N_-$$ is the difference of the numbers of eigenvalues (counted with algebraic multiplicity) in the two half planes, and for Hermitian $$H$$ it is the usual sign function. Here $$X$$ is the sum of a Dirac part and an optional Wilson term, and $$m_0$$ plays the role of the Wilson mass. When $$\mathcal V=\mathbb C^2\otimes\mathbb C^n$$ with $$\gamma_5=\sigma_z\otimes1$$, a Wilson term is called chirally even if it has the form $$1\otimes W$$, that is, if it acts by the same matrix on the $$R$$ and $$L$$ components. We call the integer

$$
\mathrm{ind}\,D_{\rm ov}=-\tfrac12\,\mathrm{Tr}\,\mathrm{sgn}\,H \tag{3}
$$

the overlap index. It is defined exactly when $$H$$ has no purely imaginary eigenvalue, and it is an integer because $$N_++N_-=\dim\mathcal V$$ is even when $$\mathrm{Tr}\,\gamma_5=0$$.

<div class="lemma" markdown="1">
**Lemma 1** (Ginsparg–Wilson relation and index). (a) For every $$X$$ for which $$\Sigma$$ is defined, $$D_{\rm ov}$$ satisfies $$\gamma_5D_{\rm ov}+D_{\rm ov}\gamma_5=m_0^{-1}D_{\rm ov}\gamma_5D_{\rm ov}$$. (b) If $$\dim\mathcal V_R=\dim\mathcal V_L$$ and $$H$$ is Hermitian, then $$\mathrm{ind}\,D_{\rm ov}=\nu_+-\nu_-$$, where $$\nu_\pm$$ is the number of zero modes of $$D_{\rm ov}$$ with $$\gamma_5=\pm1$$. (c) If $$\mathrm{Tr}\,\gamma_5=0$$, then for every $$X$$ for which $$\Sigma$$ is defined, $$\mathrm{Tr}\,\gamma_5(1-D_{\rm ov}/2m_0)=-\tfrac12\mathrm{Tr}\,\Sigma$$.
</div>

<div class="proof" markdown="1">
*Proof.* (a) Using only $$\Sigma^2=1$$, $$m_0^{-1}D_{\rm ov}\gamma_5D_{\rm ov}=m_0(1+\gamma_5\Sigma)\gamma_5(1+\gamma_5\Sigma)=m_0(2\gamma_5+\gamma_5\Sigma\gamma_5+\Sigma)$$, and $$\gamma_5D_{\rm ov}+D_{\rm ov}\gamma_5=m_0(2\gamma_5+\Sigma+\gamma_5\Sigma\gamma_5)$$.

(c) $$\mathrm{Tr}\,\gamma_5(1-D_{\rm ov}/2m_0)=\mathrm{Tr}\,\gamma_5-\tfrac12\mathrm{Tr}\,\gamma_5(1+\gamma_5\Sigma)=-\tfrac12\mathrm{Tr}\,\Sigma$$, because $$\mathrm{Tr}\,\gamma_5=0$$.

(b) Let $$T=\gamma_5\Sigma$$. For Hermitian $$H$$, $$\Sigma$$ is Hermitian, so $$T$$ is unitary and $$D_{\rm ov}=m_0(1+T)$$ is normal, with $$\gamma_5D_{\rm ov}\gamma_5=D_{\rm ov}^\dagger$$. Its eigenvalues lie on the circle $$\lvert \lambda-m_0\rvert =m_0$$, so a real eigenvalue is $$0$$ or $$2m_0$$. The map $$\gamma_5$$ takes the eigenspace of a non-real $$\lambda$$ to that of $$\bar\lambda$$, which is orthogonal to it, so every eigenvector with non-real eigenvalue has $$\langle\gamma_5\rangle=0$$. Therefore $$\mathrm{Tr}\,\gamma_5(1-D_{\rm ov}/2m_0)$$ receives a contribution only from the eigenvalue $$0$$, namely $$\nu_+-\nu_-$$, and by (c) it equals $$-\tfrac12\mathrm{Tr}\,\Sigma$$. $$\square$$
</div>

For a Hermitian $$H$$ the index therefore counts the net chirality of exact zero modes, and Lemma 1(a) is why the construction is exactly chiral at finite cutoff. For a non-Hermitian $$H$$ only (a) and (c) survive. The integer (3) is then the trace of the chiral density $$\gamma_5(1-D_{\rm ov}/2m_0)$$, and it need not count zero modes: in the zero-diagonal case of Theorem 1 below $$D_{\rm ov}$$ is nilpotent, so every vector is annihilated by some power of it, while (3) vanishes. The theorems of Sections 3 and 4 concern this trace, which is the quantity that equals the flux in the Hermitian Euclidean case.

<div class="example" markdown="1">
**Example 1** (the Euclidean overlap on Poisson points). Take a unit-density Poisson process on the torus of side $$L=\sqrt N$$ with antiperiodic spinors, Pauli matrices $$\gamma_1=\sigma_1$$, $$\gamma_2=\sigma_2$$, $$\gamma_5=\sigma_3$$, and the Gaussian-kernel Wilson–Dirac operator

$$
D_W[x,y]=w(\Delta_{xy})\Big[\frac{\gamma\cdot\Delta_{xy}}{Z_1}-\frac{r}{Z_0}\Big]\Phi_{xy},\quad D_W[x,x]=\frac{r}{Z_0}\sum_{y\ne x}w(\Delta_{xy}), \tag{4}
$$

where $$\Delta_{xy}$$ is the displacement from $$x$$ to the nearest image of $$y$$, $$w(\Delta)=e^{-\lvert \Delta\rvert ^2/2s^2}$$ for $$\lvert \Delta\rvert <6s$$, $$Z_1=2\pi s^4$$, $$Z_0=2\pi s^2$$, and $$\Phi_{xy}$$ are the link phases of a background U(1) field. The field is uniform, $$F_{12}=\partial_1\mathcal A_2-\partial_2\mathcal A_1=2\pi Q/L^2$$ with total flux $$2\pi Q$$, given in Landau gauge $$\mathcal A=(0,F_{12}x_1)$$ with transition functions, and $$\Phi_{xy}=\exp(i\int_y^x\mathcal A\cdot dl)$$ along the straight segment, the parallel transport for $$D=\partial-i\mathcal A$$. The mean symbol of $$D_W$$ is $$i\gamma\cdot p\,e^{-s^2p^2/2}+r(1-e^{-s^2p^2/2})$$. The operator uses the Euclidean metric $$\lvert \Delta\rvert $$, which is a choice of Euclidean time axis (a Wick axis; see below). With $$s=1.5$$, $$r=1$$, $$m_0=\tfrac12$$ we built $$D_{\rm ov}$$ by (2) with $$X=D_W$$ at $$N=300$$ for three point sets and $$Q=0,1,2,-1$$ each. In every one of the twelve cases $$H$$ had no eigenvalue of modulus below $$0.10$$ and the index was $$+Q$$ in this convention, carried by exactly $$\lvert Q\rvert $$ zero modes, all of chirality $$\mathrm{sgn}\,Q$$ (none for $$Q=0$$). Reversing the orientation of the flux, or the sign of the link phase, gives $$-Q$$.
</div>

These twelve cases are a numerical observation. Lüscher's lattice index theorem [[L]](#ref-l) is proved for local, gauge-covariant, doubler-free operators on regular lattices with smooth admissible fields. We prove no analogue for the Gaussian kernel on Poisson points, only integrality and the Ginsparg–Wilson relation (Lemma 1).

The unit-density Poisson law is invariant under every volume-preserving affine map, because the image of a Poisson process under such a map is Poisson with the image intensity, which is again Lebesgue measure. A boost is such a map, and it carries the Euclidean metric attached to one unit timelike vector $$e$$ to the one attached to the boosted vector. The metric attached to $$e$$ is $$\lvert \Delta\rvert _e^2=2(\Delta\cdot e)^2-\Delta^2$$, the Minkowski interval with the sign of its spatial part reversed, which equals $$t^2+x^2$$ in the frame where $$e=(1,0)$$; a Poisson set in the plane with this metric is what Example 1 places on a torus. So the law of $$D_W$$ built with the Euclidean metric of $$e$$ is the same for every unit timelike $$e$$: the frame is a choice of Wick axis, which the points do not carry. [Frames, Wilson terms and reflection positivity](/quantum-gravity/frames-and-reflection-positivity/) reports that the Gaussian-kernel Wilson term of Example 1 and its overlap fail a one-particle reflection-positivity test on mirror-symmetric sets in two dimensions, a numerical result, so that the Euclidean operator is not shown to define a Lorentzian theory. The operator $$D_W$$ has $$\gamma_5$$-Hermiticity, $$D_W^\dagger=\gamma_5D_W\gamma_5$$, which makes $$H=\gamma_5(D_W-m_0)$$ Hermitian. The retarded operators of Section 3 lack it, and so does the symmetrised operator of Section 4 built from Definition 1, because their two off-diagonal blocks do not satisfy $$D_{21}=-D_{12}^\dagger$$. The operators of Corollary 1 have it, with a Hermitian or anti-Hermitian Weyl block, and their index vanishes. What separates the Euclidean overlap of Example 1 from them is the pairing of the two blocks: the adjoint of $$\partial_z$$ is $$-\partial_{\bar z}$$ in Euclidean signature, so the blocks approximate two different derivatives, whereas a block $$A$$ and $$-A^\dagger$$ with $$A$$ (anti-)Hermitian approximate the same derivative twice.

## 3. Retarded operators carry no index

An operator of the form of Definition 1, with any weights, links and gauge phases and in any dimension, is retarded, and Theorem 1 applies to it.

<div class="theorem" markdown="1">
**Theorem 1** (retarded operators). Let $$\mathcal C$$ be a finite partially ordered set with a linear extension $$x_1,\dots,x_N$$, and let the space $$\mathcal V$$ be the direct sum of spaces $$\mathcal V_x$$, $$x\in\mathcal C$$, each graded by $$\gamma_5$$. Assume

(R1) $$X$$ is block lower triangular for this ordering: the block $$X_{xy}$$ vanishes unless $$y\preceq x$$ in the linear extension. The diagonal blocks $$X_{xx}$$ are arbitrary.

(R2) $$m_0>0$$ is such that no $$H_{xx}=\gamma_5(X_{xx}-m_0)$$ has a purely imaginary eigenvalue.

Then $$H=\gamma_5(X-m_0)$$ is block lower triangular, its spectrum is the union of the spectra of the $$H_{xx}$$, the overlap operator (2) exists, satisfies the Ginsparg–Wilson relation exactly, and

$$
\mathrm{ind}\,D_{\rm ov}=-\tfrac12\sum_{x}\mathrm{Tr}\,\mathrm{sgn}\,H_{xx}. \tag{5}
$$

In particular, if each diagonal block is chirally even, $$X_{xx}=1\otimes a_x$$ on $$\mathcal V_x=\mathbb C^2\otimes\mathbb C^{\ell_x}$$, then $$\mathrm{ind}\,D_{\rm ov}=0$$. If the diagonal blocks vanish, $$H$$ has the two eigenvalues $$\pm m_0$$ only, and $$D_{\rm ov}$$ is strictly block lower triangular, hence nilpotent. The index does not depend on any off-diagonal block, so it is independent of the weights, the links and the background gauge field.
</div>

<div class="proof" markdown="1">
*Proof.* Since $$\gamma_5$$ is block diagonal, $$H$$ is block lower triangular with diagonal blocks $$H_{xx}$$. The characteristic polynomial of a block triangular matrix is the product of those of the diagonal blocks, so the eigenvalues of $$H$$, with algebraic multiplicity, are those of the $$H_{xx}$$, and (R2) says none lies on the imaginary axis. Then $$\mathrm{Tr}\,\mathrm{sgn}\,H=N_+-N_-$$ is the sum of the same numbers for the blocks, which is (5). Lemma 1(a) is algebra and applies.

For a chirally even diagonal block, $$H_{xx}=\sigma_z\otimes(a_x-m_0)$$ has the eigenvalues $$\pm\lambda$$ for each eigenvalue $$\lambda$$ of $$a_x-m_0$$, so $$N_+=N_-$$ and the block contributes zero to (5).

For vanishing diagonal blocks, $$H_{xx}=-m_0\gamma_5$$ has the eigenvalue $$-m_0$$ on $$\mathcal V_{x,R}$$ and $$+m_0$$ on $$\mathcal V_{x,L}$$. Let $$P_+$$ be the Riesz projection $$\frac1{2\pi i}\oint(z-H)^{-1}dz$$ around $$+m_0$$. The resolvent $$(z-H)^{-1}$$ is block lower triangular with diagonal blocks $$(z+m_0\gamma_5)^{-1}$$, so $$P_+$$ is block lower triangular with diagonal blocks $$\tfrac12(1-\gamma_5)$$. Hence $$\Sigma=2P_+-1$$ is block lower triangular with diagonal blocks $$-\gamma_5$$, and $$D_{\rm ov}=m_0(1+\gamma_5\Sigma)$$ has diagonal blocks $$m_0(1-\gamma_5\gamma_5)=0$$. A block lower triangular matrix with zero diagonal blocks satisfies $$D_{\rm ov}^N=0$$. $$\square$$
</div>

A triangular matrix has the spectrum of its diagonal, and the trace of a sign function depends only on the spectrum, so the off-diagonal part cannot enter. The Ginsparg–Wilson relation holds, so the operator is exactly chiral, but formula (5) is a sum of on-site contributions that see neither the order nor the gauge field. A nonzero index of a retarded operator therefore requires a chirally asymmetric on-site term. For scalar diagonal blocks $$X_{xx}=\mathrm{diag}(c_R,c_L)$$ with real $$c_R,c_L\neq m_0$$, the contribution of the site to (5) is $$-\tfrac12[\mathrm{sgn}(c_R-m_0)-\mathrm{sgn}(c_L-m_0)]$$, which is $$\mp1$$ when $$m_0$$ lies between $$c_R$$ and $$c_L$$ and zero otherwise.

The theorem covers more than the signed-BD kernel. It applies to every retarded operator in every dimension: a Weyl block $$\sum_y a_{xy}\,\bar\sigma\cdot(x-y)$$ with $$y\prec x$$, any weights $$a_{xy}$$, any link variables, and a retarded Wilson term. A retarded Wilson term whose diagonal is chirally even (in particular zero) keeps the index at zero. The theorem applies to advanced operators by reversing the order.

Numerically, the sign function of such a defective $$H$$ is ill conditioned, and eigenvalues from a numerical eigensolver are accurate only to a fractional power of machine precision, so formula (5) is the reliable way to evaluate the index of a retarded operator.

## 4. Self-adjointness for a chirality-exchanging involution

For the signed-BD kernel the adjoint of the retarded operator is advanced and has the same functional form: since the entries are real,

$$
D_+^\dagger[x,y]=D_+[y,x]=\rho\,w_{n_{yx}}(u_y-u_x)\quad(x\prec y),\qquad\text{so}\qquad D_+^\dagger=-\widetilde D_+ , \tag{6}
$$

where $$\widetilde D_+[x,y]=\rho\,w_{n_{xy}}(u_x-u_y)$$ for $$y\succ x$$ is the advanced kernel of the same type. The sum $$D_++\widetilde D_+=D_+-D_+^\dagger$$ is the symmetrised $$\partial_v$$ kernel on all causally related pairs (the Pauli–Jordan type), and it is anti-Hermitian. In Lorentzian signature the formal adjoint of $$\partial_v$$ is $$-\partial_v$$, whereas in Euclidean signature the adjoint of $$\partial_z$$ is $$-\partial_{\bar z}$$. A $$\gamma_5$$-Hermitian Dirac operator $$\begin{pmatrix}0&A\\-A^\dagger&0\end{pmatrix}$$ assembled from a frame-free block $$A\approx\partial_v$$ therefore has the same derivative in both off-diagonal blocks when $$A$$ is anti-Hermitian: the second block is $$-A^\dagger=A$$, not an approximation to $$\partial_u$$. Theorem 2 treats this case together with the larger class of operators that are self-adjoint for a matrix exchanging the two chiralities.

<div class="theorem" markdown="1">
**Theorem 2** (Krein-self-adjoint operators). Let $$\gamma_5$$ be the grading on $$\mathcal V$$ and let $$J$$ be a matrix with

(K1) $$J=J^\dagger$$, $$J^2=1$$ and $$J\gamma_5=-\gamma_5J$$ (an involution that exchanges the two chiralities).

Let $$X$$ be a matrix that is $$J$$-self-adjoint, that is, self-adjoint for the indefinite form $$\langle w,Jw'\rangle$$:

(K2) $$JX^\dagger J=X$$.

Let $$m_0$$ be real and put $$H=\gamma_5(X-m_0)$$ and $$J'=iJ\gamma_5$$. Then $$J'$$ is a Hermitian involution and $$J'H^\dagger J'=-H$$. Consequently the spectrum of $$H$$ satisfies $$\mathrm{spec}\,H=-\overline{\mathrm{spec}\,H}$$ with equal algebraic multiplicities. If $$H$$ has no purely imaginary eigenvalue then $$\mathrm{Tr}\,\mathrm{sgn}\,H=0$$ and $$\mathrm{ind}\,D_{\rm ov}=0$$. Otherwise the index (3) is undefined.
</div>

<div class="proof" markdown="1">
*Proof.* Since $$J\gamma_5=-\gamma_5J$$, $$(iJ\gamma_5)^\dagger=-i\gamma_5J=iJ\gamma_5$$ and $$(iJ\gamma_5)^2=-J\gamma_5J\gamma_5=J^2\gamma_5^2=1$$. Because $$m_0$$ is real, $$X_m=X-m_0$$ also satisfies $$JX_m^\dagger J=X_m$$, and $$H=\gamma_5X_m$$. Then

$$
J'H^\dagger J'=-J\gamma_5X_m^\dagger\gamma_5J\gamma_5=J\gamma_5X_m^\dagger J=-\gamma_5JX_m^\dagger J=-\gamma_5X_m=-H ,
$$

where we used $$\gamma_5J\gamma_5=-J$$, then $$J\gamma_5=-\gamma_5J$$, then (K2) for $$X_m$$. So $$H^\dagger$$ is similar to $$-H$$ and the two have the same eigenvalues with multiplicity. The eigenvalues of $$H^\dagger$$ are the conjugates of those of $$H$$, which gives the symmetry. If no eigenvalue is purely imaginary, the map $$\lambda\mapsto-\bar\lambda$$ exchanges the eigenvalues in $$\mathrm{Re}\,\lambda>0$$ with those in $$\mathrm{Re}\,\lambda<0$$, so $$N_+=N_-$$. $$\square$$
</div>

The involution $$J'=iJ\gamma_5$$ turns the exchange of chiralities into a symmetry of $$H$$ itself. The symmetry of the spectrum is of the kind familiar from Krein-space theory, in which Dirac operators are self-adjoint for an indefinite form [[S]](#ref-s).

<div class="corollary" markdown="1">
**Corollary 1** (Hermitian and anti-Hermitian Weyl blocks). Let $$A$$ be any square matrix with $$A^\dagger=-A$$ or $$A^\dagger=A$$, let $$W$$ be any Hermitian matrix, and let $$m_0$$ be real. Let $$\mathcal V=\mathbb C^2\otimes\mathbb C^\ell$$ with $$\gamma_5=\sigma_z\otimes1$$, and put

$$
H=\sigma_z\otimes(W-m_0)+\begin{cases}i\sigma_y\otimes A,&A^\dagger=-A,\\ \sigma_x\otimes A,&A^\dagger=A.\end{cases}
$$

This is $$\gamma_5(X-m_0)$$ for the Dirac operator $$\begin{pmatrix}0&A\\-A^\dagger&0\end{pmatrix}$$ with a chirally even Hermitian Wilson term $$W$$. Then $$H$$ is Hermitian, $$\mathrm{spec}\,H=-\mathrm{spec}\,H$$ and, if $$H$$ is invertible, $$\mathrm{ind}\,D_{\rm ov}=0$$ for every $$m_0$$ and $$W$$.
</div>

<div class="proof" markdown="1">
*Proof.* Both $$\sigma_x\otimes1$$ and $$\sigma_y\otimes1$$ anticommute with $$\gamma_5=\sigma_z\otimes1$$, so (K1) holds for either. For $$A^\dagger=-A$$ the Dirac operator is $$\sigma_x\otimes A$$, $$X=1\otimes W+\sigma_x\otimes A$$, $$X^\dagger=1\otimes W-\sigma_x\otimes A$$, and with $$J=\sigma_y\otimes1$$ one has $$JX^\dagger J=X$$ because $$\sigma_y\sigma_x\sigma_y=-\sigma_x$$. For $$A^\dagger=A$$ the Dirac operator is $$i\sigma_y\otimes A$$, $$X^\dagger$$ contains $$-i\sigma_y\otimes A$$, and with $$J=\sigma_x\otimes1$$ one has $$\sigma_x(-i\sigma_y)\sigma_x=i\sigma_y$$, so again $$JX^\dagger J=X$$. Thus (K2) holds and Theorem 2 applies, with $$J'=-\sigma_x\otimes1$$ for $$J=\sigma_y\otimes 1$$ and $$J'=\sigma_y\otimes1$$ for $$J=\sigma_x\otimes1$$; in both cases $$J'$$ anticommutes with $$H$$. Because $$H$$ is Hermitian its spectrum is real, and the symmetry of the spectrum gives $$\mathrm{Tr}\,\mathrm{sgn}\,H=0$$ whenever $$H$$ is invertible. $$\square$$
</div>

Corollary 1 uses that $$W$$ is Hermitian and chirally even. Conjugation by either $$J=\sigma_x\otimes1$$ or $$J=\sigma_y\otimes1$$ exchanges $$W_R$$ and $$W_L$$, so $$J\,\mathrm{diag}(W_R,W_L)^\dagger J=\mathrm{diag}(W_L^\dagger,W_R^\dagger)$$. A Hermitian Wilson term $$\mathrm{diag}(W_R,W_L)$$ with $$W_R\ne W_L$$ therefore fails (K2) for these two $$J$$, and Theorem 2 makes no statement about it. For a general chirality-exchanging $$J$$ the condition on $$\mathrm{diag}(W_R,W_L)$$ differs, and we do not treat it. A non-Hermitian one with $$W_L=W_R^\dagger$$ satisfies the Wilson-term part of (K2), and Theorem 2 then applies when the Dirac part does as well.

Corollary 1 does not exhaust Theorem 2. The retarded operator $$D$$ of Definition 1 is not $$J$$-self-adjoint, but for $$J=\sigma_x$$ the $$J$$-self-adjoint part of $$iD$$ is $$\tfrac i2\begin{pmatrix}0&D_+-D_+^\dagger\\D_--D_-^\dagger&0\end{pmatrix}$$, the symmetrised kernel of (6) in both blocks. Its two off-diagonal blocks are different Hermitian matrices, so Theorem 2 applies while Corollary 1 (which needs equal blocks up to sign) does not. In four dimensions the same holds for a Weyl block $$A[x,y]=c_{xy}\,\bar\sigma\cdot(x-y)$$ with real weights $$c_{xy}$$, where $$\bar\sigma^\mu=(1,-\vec\sigma)$$. Since each $$\bar\sigma^\mu$$ is Hermitian, $$A^\dagger[x,y]=-c_{yx}\,\bar\sigma\cdot(x-y)$$, so $$A$$ is anti-Hermitian when the weights are symmetric, $$c_{xy}=c_{yx}$$, and Hermitian when they are antisymmetric. A retarded kernel is neither, but its symmetrisation $$A_{\rm ret}-A_{\rm ret}^\dagger$$ is anti-Hermitian and falls under Corollary 1, as in two dimensions. The retarded block alone falls under Theorem 1. We did not compute four-dimensional spectra of these operators.

For the Krein-self-adjoint operator built from the signed-BD kernel, the mean symbols of Proposition 1 show why the spectrum meets the imaginary axis. Let the chirally even Wilson term have symbol $$r\widehat W(k)$$ and put $$\alpha(k)=r\widehat W(k)-m_0$$. The symbol of $$D_+-D_+^\dagger$$ is $$2i\,\mathrm{Im}\,S=2ik_v\mathrm{Re}\,h$$, so the symbol of $$H$$ is $$\begin{pmatrix}\alpha&-k_v\mathrm{Re}\,h\\k_u\mathrm{Re}\,h&-\alpha\end{pmatrix}$$ and its eigenvalues satisfy $$\lambda^2=\alpha^2-k_uk_v(\mathrm{Re}\,h)^2$$. At timelike momenta with $$k_uk_v(\mathrm{Re}\,h)^2>\alpha^2$$ the eigenvalues are purely imaginary, so the sign function is undefined on that open set of momenta. The set is non-empty only if $$\alpha^2$$ is below the maximum of $$k_uk_v(\mathrm{Re}\,h)^2$$ over the timelike momenta, which is $$1.009\,\rho$$ (reached at $$-2k_uk_v/\rho\simeq-1.9$$, from a numerical evaluation of $$\mathrm{Re}\,h$$ on the cut), so it requires $$\lvert r\widehat W(k)-m_0\rvert\lesssim\sqrt\rho$$ at the momenta concerned. In Euclidean signature $$\lambda^2=\alpha^2+\lvert k\rvert ^2>0$$ and there is no such set. This is a derivation from mean symbols, and an individual sprinkling is not covered by it.

On a two-dimensional causal diamond of $$72$$ points, with $$W=\tfrac12(B+B^\dagger)$$ for $$B$$ the signed-BD scalar kernel with random link phases and $$(r,m_0)=(1,0.45),(-1,0.45),(0.6,1.3)$$, the Krein-self-adjoint operator above has a few to about twenty of the $$144$$ eigenvalues of $$H$$ within $$10^{-6}$$ of the imaginary axis, depending on the sprinkling and the parameters, so the index is undefined for these sprinklings. Such eigenvalues also occur for some sprinklings at $$(r,m_0)=(0,6)$$, where the set of momenta defined above is empty, so the mean-symbol estimate does not account for them.

## 5. Finite Atiyah–Patodi–Singer indices

In the continuum, the Dirac operator on a Lorentzian region between two Cauchy surfaces is Fredholm when the data on each surface are restricted to a spectral subspace of a boundary Dirac operator, with complementary subspaces on the two surfaces (which one goes with the initial surface depends on the sign convention of the boundary operator), and its index is the Atiyah–Patodi–Singer expression [[BS]](#ref-bs). A discrete analogue replaces the surfaces by antichains. In the simplest finite definition the index is a dimension count that does not involve the bulk operator, and it is generally nonzero.

<div class="proposition" markdown="1">
**Proposition 3** (finite APS reduction; any dimension). Let $$D:\mathbb C^N\to\mathbb C^N$$ be any linear map. Let $$B_0,B_1$$ be disjoint subsets of the $$N$$ points, with Hermitian operators $$\Theta_0$$ on $$\mathbb C^{B_0}$$ and $$\Theta_1$$ on $$\mathbb C^{B_1}$$. Let $$\mathcal D=\{\psi\in\mathbb C^N: P_{<0}(\Theta_0)\psi\!\upharpoonright_{B_0}=0,\ P_{>0}(\Theta_1)\psi\!\upharpoonright_{B_1}=0\}$$ and $$\mathrm{ind}_{\rm APS}=\dim\ker(D\!\upharpoonright_{\mathcal D})-\dim\mathrm{coker}(D\!\upharpoonright_{\mathcal D})$$, the cokernel being taken in the full codomain $$\mathbb C^N$$. Write $$\nu_\pm(\Theta)$$ for the numbers of positive and negative eigenvalues, $$\nu_0(\Theta)=\dim\ker\Theta$$ and $$\eta(\Theta)=\nu_+-\nu_-$$. Then

(a) $$\mathrm{ind}_{\rm APS}=-\nu_-(\Theta_0)-\nu_+(\Theta_1)$$, independent of $$D$$.

(b) $$\mathrm{ind}_{\rm APS}+\tfrac12(\lvert B_0\rvert+\lvert B_1\rvert)=\tfrac12[\eta(\Theta_0)-\eta(\Theta_1)]+\tfrac12[\nu_0(\Theta_0)+\nu_0(\Theta_1)]$$.

(c) Exchanging the roles of $$B_0$$ and $$B_1$$ (order reversal) changes the sign of the $$\eta$$ terms.

(d) Suppose $$\Theta_i$$ is a function of link variables $$\Phi$$ on a neighbourhood $$\mathcal N_i$$ of $$B_i$$ that is covariant, $$\Theta_i(\Phi^g)=g\Theta_i(\Phi)g^{-1}$$ for diagonal unitary $$g$$, and that the background field is pure gauge on $$\mathcal N_0\cup\mathcal N_1$$ (every closed path there has trivial holonomy). Then the spectra of $$\Theta_0$$ and $$\Theta_1$$, and hence $$\mathrm{ind}_{\rm APS}$$, are independent of the flux $$Q$$ through the region between them.
</div>

<div class="proof" markdown="1">
*Proof.* (a) The conditions defining $$\mathcal D$$ are $$\nu_-(\Theta_0)+\nu_+(\Theta_1)$$ linear constraints, each a nonzero linear functional supported on $$B_0$$ or $$B_1$$. The supports are disjoint and, within each set, the functionals are orthonormal eigenvectors, so the constraints are independent and $$\dim\mathcal D=N-\nu_-(\Theta_0)-\nu_+(\Theta_1)$$. For any linear map from a space of dimension $$N-j$$ to $$\mathbb C^N$$, rank–nullity gives $$\dim\ker-\dim\mathrm{coker}=(N-j)-N=-j$$, here with $$j=\nu_-(\Theta_0)+\nu_+(\Theta_1)$$. (b) Substitute $$\lvert B_i\rvert=\nu_++\nu_-+\nu_0$$ for each set. (c) Reversal imposes $$P_{<0}(\Theta_1)$$ and $$P_{>0}(\Theta_0)$$. (d) Pure gauge on $$\mathcal N_i$$ means that on each connected component the link variables are those of the trivial field transformed by a diagonal unitary $$g$$, which exists because every holonomy is trivial (define $$g$$ by transport from a base point in each component). Covariance gives $$\Theta_i(\Phi)=g\Theta_i(1)g^{-1}$$, which has the spectrum of $$\Theta_i(1)$$. $$\square$$
</div>

The proposition concerns this definition, with the boundary condition imposed on the domain only and the full space as the codomain. It does not show that no discrete APS-type index exists. It shows that for this definition on finite antichains the index counts boundary eigenvalues and does not see the bulk, so a bulk-sensitive index needs an additional ingredient. In the continuum that ingredient is the evolution operator, which identifies two infinite-dimensional boundary spaces through the bulk. A closed loop of Hermitian matrices also has zero spectral flow in finite dimension, so the continuum relation between index and spectral flow cannot be reproduced by a finite family. A boundary operator whose spectrum depends on $$Q$$, chosen by hand, would put $$Q$$ into (a), and by the independence of $$D$$ it would do so equally with the bulk block replaced by zero.

The lattice index of [[F]](#ref-f) is the eta invariant of a domain-wall Dirac operator, a spectral asymmetry of a Hermitian operator. If that operator has the form of Corollary 1 for some chirally even $$W$$, its spectrum is symmetric and the eta invariant vanishes. Whether the domain-wall operator can be given that form with a frame-free kernel, and the variant with a non-Hermitian retarded block, are open.

## 6. An internal compact index

A chirality index can be carried by an internal compact space $$Y$$, as in Kaluza–Klein compactification: the index of the Dirac operator on $$Y$$ in a flux fixes how many one-handed zero modes the external space sees. Take $$\mathcal C$$ a causal set with a retarded block $$D_{\mathcal C}$$ odd under a grading $$\gamma_{\mathcal C}$$, and $$Y$$ a point set with a Ginsparg–Wilson operator $$D_Y$$ and grading $$\gamma_Y$$, for instance the operator of Example 1 on a flux torus. Define

$$
D=D_{\mathcal C}\otimes1+\gamma_{\mathcal C}\otimes D_Y,\qquad \Gamma=\gamma_{\mathcal C}\otimes\gamma_Y . \tag{7}
$$

<div class="proposition" markdown="1">
**Proposition 4** (exact reduction; any dimension). Assume $$\gamma_YD_Y+D_Y\gamma_Y=m_0^{-1}D_Y\gamma_YD_Y$$ and $$\gamma_{\mathcal C}D_{\mathcal C}=-D_{\mathcal C}\gamma_{\mathcal C}$$. Then $$\ker D_Y$$ is invariant under $$\gamma_Y$$, so it splits into $$\nu_+$$ and $$\nu_-$$ modes of chirality $$+1$$ and $$-1$$. The subspace $$\mathcal V_{\mathcal C}\otimes\ker D_Y$$ is invariant under $$D$$, and on it $$D=D_{\mathcal C}\otimes1$$. On $$\mathcal V_{\mathcal C}\otimes\ker D_Y$$ the operator $$\Gamma$$ acts as $$\pm\gamma_{\mathcal C}$$ on the $$\gamma_Y=\pm1$$ modes. Hence this sector consists of $$\nu_+$$ copies of $$D_{\mathcal C}$$ with $$\Gamma=\gamma_{\mathcal C}$$ and $$\nu_-$$ copies of $$D_{\mathcal C}$$ with $$\Gamma=-\gamma_{\mathcal C}$$, in every realisation, and $$\nu_+-\nu_-=\mathrm{ind}\,D_Y$$ when $$D_Y$$ is the overlap operator (2) of an operator $$X_Y$$ for which $$H_Y=\gamma_Y(X_Y-m_0)$$ is Hermitian (Lemma 1(b)).
</div>

<div class="proof" markdown="1">
*Proof.* If $$D_Y\phi=0$$ the Ginsparg–Wilson relation gives $$D_Y\gamma_Y\phi=-\gamma_YD_Y\phi+m_0^{-1}D_Y\gamma_YD_Y\phi=0$$, so $$\gamma_Y\phi\in\ker D_Y$$ and $$\gamma_Y$$ is diagonalisable there. On $$\psi\otimes\phi$$ with $$\phi\in\ker D_Y$$ the second term of $$D$$ vanishes, leaving $$(D_{\mathcal C}\psi)\otimes\phi$$; $$\Gamma$$ acts as $$\gamma_{\mathcal C}\otimes\gamma_Y$$ and for $$\gamma_Y\phi=\pm\phi$$ this is $$\pm\gamma_{\mathcal C}\otimes1$$. $$\square$$
</div>

The sector $$\mathcal V_{\mathcal C}\otimes\ker D_Y$$ is not the whole kernel of $$D$$, which also contains the kernel of $$D_{\mathcal C}$$ in other sectors. In the basis $$\gamma_{\mathcal C}=\pm1$$ write $$D_{\mathcal C}=\begin{pmatrix}0&D_+\\D_-&0\end{pmatrix}$$, as in Definition 1. A zero mode of a copy has $$D_-\psi_R=0$$ and $$D_+\psi_L=0$$, so the $$\Gamma=+1$$ components obey $$D_-\psi=0$$ in the $$\nu_+$$ copies with $$\Gamma=\gamma_{\mathcal C}$$ and $$D_+\psi=0$$ in the $$\nu_-$$ copies with $$\Gamma=-\gamma_{\mathcal C}$$. For $$\nu_-=0$$ the sector contains $$\nu_+$$ copies of a single Weyl block. For $$Y$$ a flux torus carrying $$Q$$ units, $$Y$$ contributes $$\lvert Q\rvert$$ zero modes of one chirality in the cases computed in Example 1.

Proposition 4 moves the question of chirality to $$\mathcal C$$: the multiplicity and the handedness are fixed by $$D_Y$$, and the propagation of the mode along $$\mathcal C$$ is governed by the one-handed block $$D_+$$ or $$D_-$$ of the retarded operator. The retarded operator $$D_{\mathcal C}$$ has overlap index zero by Theorem 1, so in this construction the chirality of the product theory is carried by $$Y$$. Whether $$D_+$$ propagates a massless one-handed mode is a question about the two-dimensional block, which Section 7 addresses for the signed-BD kernel. By the symmetry $$u\leftrightarrow v$$ the statements for $$D_-$$ are the same.

## 7. Noise and smearing of the two-dimensional kernel

Proposition 1 gives the mean of $$D_+$$. A single sprinkling gives a random operator whose entries fluctuate, because $$n_{xy}$$ does. This section computes the fluctuation of one row, and the effect of smearing, for the kernel of Definition 1. The computation is two-dimensional and specific to the layer weights $$(1,-2,1)$$.

Take $$\rho=1$$ and a rectangular region, and let $$x$$ be a point whose past in the region is the rectangle $$[0,a]\times[0,b]$$ in the displacements $$(U,V)$$. Given points at $$x$$ and $$y$$, the entry $$D^\varepsilon_+[x,y]=\varepsilon^2f(n_{xy},\varepsilon)U$$ is a function of $$n_{xy}\sim\mathrm{Poisson}(UV/2)$$. Let $$F_\varepsilon(x)^2=\int\mathrm{Var}\big(D_+^\varepsilon[x,y]\mid x,y\in\mathcal P\big)\,\tfrac12\,dU\,dV$$ be the total variance carried by the entries of the row, and $$M_\varepsilon(x)^2$$ the squared norm of the mean row, $$\int\big(\mathbb E[D_+^\varepsilon[x,y]\,\lvert \,x,y]\big)^2\tfrac12\,dU\,dV$$.

<div class="proposition" markdown="1">
**Proposition 5** (row noise; two dimensions). As $$b\to\infty$$ with $$a$$ fixed,

$$
\mathbb E\,\lVert{\rm row}_x\rVert^2=\tfrac12\varepsilon^3\,\frac{6}{(2-\varepsilon)^5}\,a^2+O(b^{-2}),\qquad M_\varepsilon^2=\tfrac12\varepsilon^3\,\frac{3}{16}\,a^2+O(b^{-2}),\qquad F_\varepsilon^2=\tfrac12\varepsilon^3\Big[\frac{6}{(2-\varepsilon)^5}-\frac3{16}\Big]a^2+O(b^{-2}). \tag{8}
$$

For $$\varepsilon=1$$ these are $$3a^2$$, $$\tfrac3{32}a^2$$ and $$\tfrac{93}{32}a^2$$; for small $$\varepsilon$$, $$F_\varepsilon^2\simeq\tfrac{15}{64}\varepsilon^4a^2$$.
</div>

<div class="proof" markdown="1">
*Proof.* By the Mecke formula $$\mathbb E\lVert {\rm row}_x\rVert ^2=\tfrac12\varepsilon^4\int_0^a dU\,U^2\int_0^b dV\,\mathbb E[f(n,\varepsilon)^2]$$, with $$n\sim\mathrm{Poisson}(UV/2)$$. The substitution $$\lambda=UV/2$$ turns the $$V$$ integral into $$(2/U)\int_0^{Ub/2}\mathbb E[f^2](\lambda)\,d\lambda$$, and for $$b\to\infty$$ the upper limit may be removed with an error of order $$b^{-2}$$, since $$\mathbb E[f^2](\lambda)$$ decays exponentially. Because $$\int_0^\infty\frac{e^{-\lambda}\lambda^n}{n!}d\lambda=1$$, $$\int_0^\infty\mathbb E[f^2](\lambda)\,d\lambda=\sum_nf(n,\varepsilon)^2$$. Writing $$q=1-\varepsilon$$, $$f(n,\varepsilon)=q^{n-2}\big(q^2-2n\varepsilon q+\tfrac12 n(n-1)\varepsilon^2\big)$$ for $$0<\varepsilon<2$$ (at $$\varepsilon=1$$ read $$f$$ as the weights $$w_n$$), the sum is a combination of $$\sum_n n^{\underline j}q^{2n}=j!\,q^{2j}/(1-q^2)^{j+1}$$, where $$n^{\underline j}$$ is the falling factorial. Expanding $$f^2$$ in falling factorials and summing gives $$\sum_nf^2=6/[\varepsilon(2-\varepsilon)^5]$$ exactly. Then $$\int_0^adU\,U=a^2/2$$ gives the first formula. For the mean row, $$\mathbb E[f(n,\varepsilon)]=f_0(\varepsilon\lambda)$$ with $$f_0(z)=e^{-z}(1-2z+z^2/2)$$, and $$\int_0^\infty f_0(z)^2dz=\tfrac3{16}$$ gives the second. The variance is the difference. $$\square$$
</div>

The numbers $$3$$ and $$\tfrac3{32}$$ are sums over the weights, and the geometry of the region only supplies the factor $$a^2/2$$. At $$\varepsilon=1$$ the entries of a row have a total variance about thirty times the squared norm of the mean row. The ratio $$F_\varepsilon^2/M_\varepsilon^2=32/(2-\varepsilon)^5-1$$ decreases to zero like $$5\varepsilon/2$$ as $$\varepsilon\to0$$.

A plane wave of unit amplitude and wave number $$k\sim\pi/a$$ has mean action of order $$k$$. If correlations between entries are ignored, the fluctuation of $$(D_+^\varepsilon\psi)(x)$$ is of order $$F_\varepsilon\simeq\sqrt{15/64}\,\varepsilon^2a$$ for small $$\varepsilon$$, so the ratio of fluctuation to signal is of order $$\varepsilon^2a^2$$ in units of the discreteness length. This comparison is heuristic: it keeps only the diagonal part of the variance and drops the correlations between entries that share interval points. Requiring this ratio to be at most one gives $$\varepsilon\lesssim 1/a$$, that is a nonlocality length $$(\varepsilon\rho)^{-1/2}$$ of order $$a^{1/2}$$ discreteness lengths, which grows with the region.

<div class="proposition" markdown="1">
**Proposition 6** (smearing squeeze; two dimensions). Fix $$\kappa>0$$ and $$\delta>0$$. The fraction of the box $$[-\kappa,\kappa]^2$$ of momenta (in units $$\rho=1$$) on which $$\lvert S_\varepsilon(k)\rvert <\delta$$ tends to $$1$$ as $$\varepsilon\to0$$.
</div>

<div class="proof" markdown="1">
*Proof.* By Proposition 1, $$\lvert S_\varepsilon(k)\rvert =\lvert k_v\rvert \,\lvert h(-2k_uk_v/\varepsilon)\rvert \le C\lvert k_v\rvert \,\varepsilon^2/(4k_u^2k_v^2)$$ as soon as $$2\lvert k_uk_v\rvert /\varepsilon\ge1$$, which tends to zero at every point off the axes. The indicator of $$\{\lvert S_\varepsilon\rvert <\delta\}$$ is bounded by $$1$$ and tends to $$1$$ almost everywhere; dominated convergence gives the claim. $$\square$$
</div>

Smearing at a fixed momentum resolution therefore removes the noise of Proposition 5 only by sending the mean symbol to zero over the whole resolved box. The statement depends on the box being held at the discreteness scale. If momenta are resolved only down to the nonlocality scale $$(\varepsilon\rho)^{-1/2}$$, the box is rescaled with $$\varepsilon$$ and the squeeze disappears.

Section 6 needs to know whether the unsmeared block $$D_+$$ annihilates a function of $$u$$ alone, as the continuum $$\partial_v$$ does. To test this we applied $$D_+$$, with its prefactor $$\rho$$, on two-dimensional sprinklings of the square $$[0,30]^2$$ in $$(u,v)$$ to two Gaussian packets, $$\chi_u(u)$$ and $$\chi_v(v)$$, both centred at $$15$$ with standard deviation $$30/8$$. The first is annihilated by $$\partial_v$$ and the second is not. The table gives the ratio of the sum of $$\lvert D_+\phi\rvert^2$$ over the points to the sum of $$\lvert\phi\rvert^2$$, averaged over five sprinklings (three at $$\rho=4$$).

| $$\rho$$ | $$\chi_u$$ | $$\chi_v$$ |
|---|---|---|
| 1 | $$722\pm39$$ | $$1092\pm139$$ |
| 2 | $$2856\pm197$$ | $$3886\pm347$$ |
| 4 | $$10946\pm508$$ | $$14667\pm1132$$ |

The continuum values are $$0$$ and $$1/(2\cdot(30/8)^2)=0.036$$, so the squared norm for $$\chi_v$$ is between $$3\times10^4$$ and $$4\times10^5$$ times its continuum value. It exceeds the continuum contrast between the packets by orders of magnitude, since $$\chi_v$$ gives a value larger than $$\chi_u$$ only by a factor between $$1.3$$ and $$1.5$$, and it grows roughly like $$\rho^2$$. We interpret this as the noise of Proposition 5 acting on the packet. The numbers depend on the packet width and on the normalisation.

## 8. Whether a bulk-sensitive frame-free index exists

A $$\gamma_5$$-Hermitian completion $$\begin{pmatrix}0&A\\-A^\dagger&0\end{pmatrix}$$ of a retarded block $$A=D_+$$ has an advanced block of the same type in the second slot. It is not retarded, its Weyl block is not (anti-)Hermitian, and it is not Krein-self-adjoint for $$J=\sigma_x$$ or $$\sigma_y$$, so none of the theorems constrains its index. In four dimensions the literal pairing of a $$\bar\sigma\cdot\Delta$$ block with a $$\sigma\cdot\Delta$$ block falls under Theorem 1 when both blocks are retarded and under Corollary 1 when the blocks are (anti-)Hermitian.

By formula (5), a retarded operator with a chirally asymmetric on-site term can have a nonzero index. For Theorem 2 the Wilson term must satisfy (K2) together with the rest of the operator, and a non-Hermitian term with $$W_L=W_R^\dagger$$ does. A construction of either kind of asymmetric term from the causal order and the invariant interval, with a mean symbol that has the correct limit, would settle whether such terms exist.

An involution that commutes with $$\gamma_5$$ is outside Theorem 2, which needs $$J\gamma_5=-\gamma_5J$$. With $$J=\gamma_5$$ one has $$\gamma_5$$-Hermiticity, the structure of the Euclidean overlap of Example 1. Corollary 1 is the case of this structure in which $$A$$ is (anti-)Hermitian, and its index vanishes. The open case is a frame-free kernel $$A$$, built from the causal order and the invariant interval, with $$A$$ approximating $$\partial_v$$ and $$-A^\dagger$$ approximating $$\partial_u$$.

Two indices of another kind remain: a non-Hermitian time-domain-wall variant, and an APS-type index in which the bulk identifies infinite-dimensional boundary spaces. Proposition 3 covers only the finite definition with constraints on the domain, and a discrete definition of the second kind across a causal interval of a sprinkling is not known to us.

The Euclidean overlap of Example 1 uses a Wick axis. A frame in the regulator of the scalar line of a Yukawa model shifts the fermion speed at one loop by an amount independent of the cutoff when the scalar mass is small compared with the cutoff ([frame leakage](/quantum-gravity/lorentz-violation-leakage/)), and the reflection-positivity status of the Gaussian-kernel overlap on a Poisson process is described in [frames, Wilson terms and reflection positivity](/quantum-gravity/frames-and-reflection-positivity/).

The estimates of Section 7 hold for a fixed momentum resolution. With a nonlocality scale that grows like $$\varepsilon\sim1/a$$, the fraction of momenta on which the mean symbol is small, and the propagation of a one-handed packet, remain to be determined. Proposition 4 is exact for a product of two point sets; a Poisson process on $$\mathcal C\times Y$$ and a smeared $$D_{\mathcal C}$$ fall outside it.

A frame-free operator on a sprinkling with a nonzero bulk-sensitive index would lie outside the classes treated here. No theorem excludes it, and no example is known.

## References

1. <a id="ref-gw"></a>Ginsparg, Wilson, "A remnant of chiral symmetry on the lattice," Phys. Rev. D 25 (1982) 2649.
2. <a id="ref-nn"></a>Nielsen, Ninomiya, "Absence of neutrinos on a lattice. (I). Proof by homotopy theory," Nucl. Phys. B 185 (1981) 20.
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
