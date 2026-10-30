---
title: "Boundary and corner terms of the causal-set action"
date: 2026-10-29
category: entropy
order: 2
description: In four-dimensional flat space, on product regions in the embedded regime, the mean causal-set action diverges like the inverse discreteness length on a timelike boundary, kernels with an extra Mellin zero remove the divergence, and on static regions the finite part that remains is not of Gibbons–Hawking–York form.
status: "Proved for flat four-dimensional product regions (a convex polygon times a plane, or an interval times a convex body whose averaged covariogram has the small-separation expansion of Theorem 3 and is smooth near the interval length), for the embedded-regime mean and layer kernels with the zero conditions of Theorem 1. Section 8 (isolated regime) states unproved expectations."
excludes: "Under Theorem 3 (flat static products, embedded regime, zero conditions on the kernel): a Gibbons–Hawking–York or Hayward term in the finite part. Under Proposition 6 (interval functionals, exponentially decaying kernel, flat spacelike or null joints, symbol analytic at 3/2): a half-angle function in the leading joint term. Not excluded: the isolated regime, curved or non-static boundaries, boost-conical hinges."
---

Does the mean of the causal-set action on a region contain the Gibbons–Hawking–York boundary term, and what does it assign to a corner where two boundary pieces meet? We answer for four-dimensional flat regions of two kinds, a convex polygon times a plane and a time interval times a convex body, inside a sprinkling of all of Minkowski space (the embedded regime of section 1). With $$\rho$$ the sprinkling density, the plain Benincasa–Dowker action has a mean that diverges like $$\rho^{1/4}$$ in proportion to the volume of the timelike boundary, and kernels whose Mellin symbol (section 1) has one further zero, at $$5/4$$, remove the divergence. The finite part that remains is not of Gibbons–Hawking–York form. For a time interval outside a finite exceptional set it vanishes on balls and on every convex body with a $$C^{1,1}$$ boundary (a ball of fixed radius rolls freely inside it) that satisfies the remaining hypotheses of Theorem 3 (section 6), although the extrinsic curvature does not vanish.

At the edges of prisms the finite part is a function of the dihedral angle that no combination of an integrated extrinsic curvature and a Hayward corner term (proportional to the exterior angle times the edge area) reproduces, and a joint between two spacelike edges that point the same way in space carries $$\coth\theta$$, with $$\theta$$ the boost angle (sections 5 and 6). A second question is whether the factor $$2\pi$$ of the area law can come from the action at a joint. In the Euclidean treatment that factor comes from a conical singularity, where the angle around the apex runs through $$2\pi$$ before the geometry closes up, so a joint term that carries it should, continued to an imaginary boost angle, have least period $$2\pi$$ (section 7 gives the precise criterion). For functionals of the causal interval alone, on flat joints between spacelike or null edges, the leading joint term is proportional to $$\coth\theta$$. The half-angle function $$\coth(\theta/2)$$ is not a multiple of $$\coth\theta$$ and does not occur. After continuation, $$\coth\theta$$ has least period $$\pi$$, while $$\coth(\theta/2)$$ has least period $$2\pi$$. This is proved for the finite part when the symbol vanishes at $$3/2$$ and for the coefficient of $$\log\rho$$ when it does not. The constant part in that second case is not a sum over joints and is not analysed (section 7). Boost-conical hinges (codimension-two surfaces around which the geometry is a cone in a boost angle), the isolated regime, non-static boundaries and curved backgrounds lie outside the proofs (sections 8 and 9).

The action is due to Benincasa and Dowker [[BD]](#ref-bd), extended to all dimensions by Dowker and Glaser [[Dowker–Glaser]](#ref-dg) and Glaser [[Glaser]](#ref-glaser). Aslanbeigi, Saravani and Sorkin [[ASS]](#ref-ass) introduced the family of generalised kernels and Belenchia [[Belenchia]](#ref-belenchia) showed that its members share the curvature limit. Buck, Dowker, Jubb and Surya [[BDJS]](#ref-bdjs) derived the Gibbons–Hawking–York term for a spacelike boundary and found that the mean action of a causal diamond tends to the volume of its joint, a result that Dowker [[Dowker]](#ref-dowker) and Machet and Wang [[Machet–Wang]](#ref-mw) extended to causally convex regions. For timelike boundaries Dowker, Liu and Lloyd-Jones [[DLLJ]](#ref-dllj) gave numerical and analytic evidence for a divergence of order $$\rho^{1/4}$$ and proposed the weight $$\coth\theta$$ for a joint between spacelike boundary pieces. The two regimes and the integral over the position of the first point of a pair, which becomes the covariogram below, come from the dissertation of Chevalier [[Chevalier]](#ref-chevalier). Beyond [[DLLJ]](#ref-dllj), Theorem 2 gives the corner value for every pair of edge types and Theorem 1 proves the divergence for product regions. Mellin symbols of the same form describe horizon counts on the pages [heavy-tailed horizon-count kernels](/quantum-gravity/nonlocal-weights/) and [horizon counts and the area law](/quantum-gravity/area-projector/), and the page showing that the [causal-set d'Alembertian has no positive spectral representation](/quantum-gravity/causal-box-spectrum/) treats the same layer kernels.

## 1. The action and its mean value

A *causal set* is a locally finite partially ordered set. A *Poisson sprinkling* of density $$\rho$$ into a region $$\mathcal R$$ of Minkowski space is a Poisson point process with intensity $$\rho$$ times the volume, the points being ordered by the causal relation $$\prec$$. We work in four dimensions with the Planck length set to one, and write $$\ell=\rho^{-1/4}$$ for the discreteness length. Let $$N$$ be the number of points and, for $$k\ge0$$, $$N_k$$ the number of pairs $$p\prec q$$ with exactly $$k$$ points strictly between them ($$N_0$$ counts links). A *layer action* is

$$
S=\kappa\,\rho^{-1/2}\Big[b\,N+\sum_{k}c_kN_k\Big],\qquad\kappa=\frac4{\sqrt6}, \tag{1}
$$

with real numbers $$b$$ and $$c_k$$. The Benincasa–Dowker action in four dimensions [[BD]](#ref-bd) has $$b=1$$ and $$(c_0,c_1,c_2,c_3)=(-1,9,-16,8)$$. The sign of $$R$$ is that of [[BD]](#ref-bd), for which the mean of $$S$$ tends to the Einstein–Hilbert action $$\tfrac12\int R\sqrt{-g}\,d^4x$$; the signature enters only through this sign. The causal-set d'Alembertian $$B$$ of [[BD]](#ref-bd) has mean $$\Box\phi-\tfrac12R\phi$$ on smooth functions $$\phi$$, and acts on the constant function as $$(B1)(x)=\kappa\rho^{1/2}\big[-1+\sum_kc^B_kN_k(x)\big]$$, where $$N_k(x)$$ is the number of points $$p\prec x$$ with exactly $$k$$ points between $$p$$ and $$x$$, so that $$\sum_xN_k(x)=N_k$$, and $$c^B=(1,-9,16,-8)$$. Hence $$S=-\rho^{-1}\sum_x(B1)(x)$$ and $$c_k=-c^B_k$$. The pages on the d'Alembertian write the weights as $$c^B$$; the Remark in section 5 of [horizon counts and the area law](/quantum-gravity/area-projector/) fixes the signature and normalisation of that operator.

Two conventions for the points between $$p$$ and $$q$$ are in use, following [[Chevalier]](#ref-chevalier) and [[DLLJ]](#ref-dllj). In the *embedded regime* the region sits inside a sprinkling of the whole of Minkowski space, all points of which that lie in the causal interval of $$p$$ and $$q$$ are counted whether or not they lie in $$\mathcal R$$, and only pairs with both endpoints in $$\mathcal R$$ enter the sum. In the *isolated regime* only the points in $$\mathcal R$$ exist, so the boundary cuts the interval. For causally convex regions the two coincide. Sections 2 to 7 are in the embedded regime and concern the mean of $$S$$ over sprinklings.

By the Mecke formula for the Poisson process, the expectation of a sum over pairs of distinct points is the integral of the product of the two intensities, and given two points of the process the others are still Poisson. The number of other points in an interval of volume $$V_{pq}$$ is therefore Poisson with mean $$\rho V_{pq}$$, so $$\langle N_k\rangle=\rho^2\iint_{p\prec q}(\rho V_{pq})^ke^{-\rho V_{pq}}/k!$$ and

$$
\langle S\rangle=\kappa\rho^{-1/2}\Big[b\,\rho\,{\rm Vol}(\mathcal R)+\rho^2\iint_{p\prec q;\;p,q\in \mathcal R}f(\rho V_{pq})\,dV_p\,dV_q\Big],\qquad f(v)=e^{-v}\sum_k\frac{c_k}{k!}v^k. \tag{2}
$$

The Mellin transform of the kernel defines the *Mellin symbol* $$Q$$,

$$
\int_0^\infty v^{s-1}f(v)\,dv=\Gamma(s)\,Q(s),\qquad Q(s)=\sum_kc_k\frac{(s)_k}{k!}, \tag{3}
$$

and the weights are recovered from $$c_m=\sum_{j=0}^m(-1)^j\binom mjQ(-j)$$. The Benincasa–Dowker symbol is $$Q_{\rm BD}(s)=\tfrac13(2s-1)(s-1)(2s-3)$$, with $$Q_{\rm BD}'(\tfrac32)=\tfrac23$$, $$Q_{\rm BD}'(1)=-\tfrac13$$ and $$Q_{\rm BD}(\tfrac54)=-\tfrac1{16}$$. The pair integral in (2) is a sum of terms $$\rho^{2-s}$$ with geometric coefficients, and the kernel multiplies the term at $$s$$ by $$Q(s)$$ (Lemmas 1 and 2 below). Four values of $$s$$ occur: $$s=\tfrac12$$ controls the decay of the reduced kernel $$\Psi$$ of Lemma 1, and for a region with timelike edges $$s=1$$ carries a volume term, $$s=\tfrac54$$ a boundary term and $$s=\tfrac32$$ the joint term.

## 2. Reduction to the covariogram

A pair $$p\prec q$$ in a flat region is described by the separation $$q-p$$. All intervals of equal volume are related by Poincaré transformations, so the sum over pairs depends on the region only through the volume of the set of pairs with a given separation, that is, through its covariogram. For a product with a transverse plane this reduces the four-dimensional problem to a two-dimensional one.

Let $$\mathcal R=P\times\mathbb R^2$$ with $$P$$ a convex polygon in the $$(t,x)$$ plane and $$\mathbb R^2$$ the transverse plane. The mean is computed per unit transverse area. Use null coordinates $$u=t-x$$, $$w=t+x$$, so that $$dt\,dx=\tfrac12du\,dw$$. A point $$q$$ is in the causal future of $$p$$ when $$q-p$$ has null components $$(\xi,\zeta)$$ with $$\xi,\zeta\ge0$$ and transverse part $$\boldsymbol\Delta$$ with $$\xi\zeta>\lvert\boldsymbol\Delta\rvert^2$$. The proper time then satisfies $$\tau^2=\xi\zeta-\lvert\boldsymbol\Delta\rvert^2$$ and the interval has volume $$V=\pi\tau^4/24$$. The overlap function of the polygon in the null components of the separation is

$$
\mathcal O(\xi,\zeta)=\int du\,dw\;\mathbf 1_P(u,w)\,\mathbf 1_P(u+\xi,w+\zeta),
$$

which is twice the area, in $$(t,x)$$ measure, of the overlap of $$P$$ with its translate. It is the covariogram of $$P$$ at future-directed separations. Write $$H(\mu)=\int_0^{\infty}\mathcal O(\sqrt\mu\,y,\sqrt\mu/y)\,\frac{dy}y$$ for the integral of $$\mathcal O$$ over the hyperbola $$\xi\zeta=\mu$$ with respect to $$d\xi/\xi$$.

<div class="lemma" markdown="1">
**Lemma 1** (reduction). Let $$f$$ be bounded with $$f(v)=O(e^{-\epsilon v})$$ for some $$\epsilon>0$$, as for a layer kernel. Let $$\sigma=(24/\pi\rho)^{1/2}$$ and $$\Psi(t)=\int_0^tf(t'^2)\,dt'$$. If $$Q(\tfrac12)=0$$, then $$\Psi$$ decays faster than any power, and the pair integral per unit transverse area is

$$
\rho^2\iint f(\rho V_{pq})\,dV_pdV_q=6\rho\int_0^\infty\Psi(t)\,H(\sigma t)\,dt. \tag{4}
$$

Moreover $$m_\Psi(s):=\int_0^\infty t^{s-1}\Psi(t)\,dt=-\dfrac1{2s}\,\Gamma\big(\tfrac{s+1}2\big)\,Q\big(\tfrac{s+1}2\big)$$ for $$s>0$$.
</div>

<div class="proof" markdown="1">
*Proof.* The measure of a pair is $$\tfrac12du\,dw\,d^2x_\perp$$ for $$p$$ and likewise for $$q$$, so integrating over the position of $$p$$ gives $$\tfrac14\iint d\xi\,d\zeta\,\mathcal O(\xi,\zeta)\,d^2\Delta$$ per unit transverse area. The transverse integral is exact: with $$Y=\xi\zeta-\lvert\boldsymbol\Delta\rvert^2$$ one has $$d^2\Delta=\pi\,dY$$ and $$\rho V=(Y/\sigma)^2$$, so $$\int d^2\Delta\,f(\rho V)=\pi\sigma\,\Psi(\xi\zeta/\sigma)$$. Grouping the separations by $$\mu=\xi\zeta$$ gives $$\iint d\xi\,d\zeta\,\mathcal O\,g(\xi\zeta)=\int d\mu\,H(\mu)g(\mu)$$, hence $$\rho^2\cdot\tfrac14\pi\sigma\int d\mu\,H(\mu)\Psi(\mu/\sigma)=\rho^2\tfrac14\pi\sigma^2\int dt\,H(\sigma t)\Psi(t)$$, and $$\rho^2\pi\sigma^2/4=6\rho$$. For the decay, $$\Psi(\infty)=\tfrac12\int_0^\infty v^{-1/2}f\,dv=\tfrac12\Gamma(\tfrac12)Q(\tfrac12)=0$$, so $$\Psi(t)=-\int_t^\infty f(t'^2)dt'$$ and the tail of $$f$$ is exponentially small. The Mellin transform follows by integrating by parts, using $$\Psi(0)=\Psi(\infty)=0$$: $$\int t^{s-1}\Psi\,dt=-\frac1s\int t^sf(t^2)dt=-\frac1{2s}\int v^{(s-1)/2}f(v)dv$$.
</div>

Equation (4) says that the four-dimensional problem on a product region is the two-dimensional covariogram integrated along hyperbolas and then smeared with a fixed function. The next lemma turns an expansion of $$H$$ at small argument into an expansion of the action at large density, using only the moments $$m_\Psi$$.

<div class="lemma" markdown="1">
**Lemma 2** (moments). Let $$H$$ be bounded on compact subsets of $$(0,\infty)$$, vanish for large argument, and satisfy for all $$\mu>0$$

$$
H(\mu)=\sum_{j}\mu^{\beta_j}\Big(A_j\log\frac1\mu+B_j\Big)+E(\mu),\qquad\lvert E(\mu)\rvert\le M_*\mu^{\beta_*},
$$

with finitely many exponents $$\beta_j>-1$$ below $$\beta_*$$. Then, as $$\rho\to\infty$$ with $$\sigma=(24/\pi\rho)^{1/2}$$,

$$
6\rho\int_0^\infty\Psi(t)H(\sigma t)\,dt=6\rho\sum_j\sigma^{\beta_j}\Big[A_j\big(-m_\Psi(\beta_j+1)\log\sigma-m_\Psi'(\beta_j+1)\big)+B_jm_\Psi(\beta_j+1)\Big]+O(\rho\,\sigma^{\beta_*}).
$$
</div>

<div class="proof" markdown="1">
*Proof.* Substitute $$\mu=\sigma t$$ in each term: $$\int\Psi(t)(\sigma t)^\beta dt=\sigma^\beta m_\Psi(\beta+1)$$ and $$\int\Psi(t)(\sigma t)^\beta\log\frac1{\sigma t}\,dt=\sigma^\beta\big(-\log\sigma\,m_\Psi(\beta+1)-m_\Psi'(\beta+1)\big)$$. The remainder is at most $$M_*\sigma^{\beta_*}\int t^{\beta_*}\lvert\Psi\rvert dt$$. The global bound on $$E$$ follows from a bound near zero, because $$H$$ is bounded on $$[\mu_1,\infty)$$ and vanishes for large argument while the sum grows only polynomially.
</div>

A term $$\mu^{\beta}$$ in $$H$$ contributes at order $$\rho^{(1-\beta)/2}$$ to $$\langle S\rangle$$, with a coefficient proportional to $$Q((\beta+2)/2)$$ when $$A_j=0$$: for $$\beta=0$$ this is $$s=1$$, for $$\beta=\tfrac12$$ it is $$s=\tfrac54$$, and for $$\beta=1$$ it is $$s=\tfrac32$$. A term with a logarithm has $$Q'$$ in place of $$Q$$ where $$Q$$ vanishes.

## 3. The covariogram of a polygon

We need the small-argument expansion of $$H$$ for every convex polygon $$P$$, whose edges may be spacelike, null or timelike. Proposition 1 gives the covariogram near the origin and Proposition 2 turns it into the expansion of $$H$$. Both enter Theorem 1, and Proposition 1 is used again in section 6. Write the plane with coordinates $$(t,x)$$ and let $$\det(p,q)=p_tq_x-p_xq_t$$. For each edge $$i$$ of $$P$$ let $$\nu_i$$ be the outward normal whose length equals the length of the edge, so that $$\nu_i\cdot h$$ is a determinant with the edge vector. At a vertex $$v$$ let $$e_v,e'_v$$ be the edge vectors, in $$(t,x)$$ components, that point away from $$v$$ along the two edges that meet there.

<div class="proposition" markdown="1">
**Proposition 1** (covariogram near the origin). Let $$C(h)=\lvert P\cap(P-h)\rvert$$ be the area in $$(t,x)$$ measure, so $$\mathcal O=2C$$. There is $$r_P>0$$ such that for $$\lvert h\rvert<r_P$$

$$
C(h)=V_P-\sum_i(\nu_i\cdot h)_-+\sum_v\frac{\big[\det(h,e_v)\det(h,e'_v)\big]_+}{2\,\lvert\det(e_v,e'_v)\rvert}, \tag{5}
$$

where $$(y)_\pm=\max(0,\pm y)$$ and $$V_P$$ is the area of $$P$$.
</div>

Translating a convex polygon by a short vector $$h$$ removes a strip along each edge facing $$-h$$, which costs area linear in $$h$$. The strips cut off at the two vertices where a line parallel to $$h$$ touches the polygon overlap, which gives a quadratic correction.

<div class="proof" markdown="1">
*Proof.* Take $$h=\lvert h\rvert\hat n$$ with $$\hat n$$ a unit vector. Let $$\Lambda_{\hat n}(z)$$ be the length of the chord of $$P$$ along $$\hat n$$ at transverse coordinate $$z$$ and $$\Pi_{\hat n}$$ the projection of $$P$$ along $$\hat n$$, of length $$\lvert\Pi_{\hat n}\rvert$$. Then $$C(h)=\int(\Lambda_{\hat n}(z)-\lvert h\rvert)_+dz=V_P-\lvert h\rvert\lvert\Pi_{\hat n}\rvert+\int(\lvert h\rvert-\Lambda_{\hat n}(z))_+dz$$, the last integral over $$\Pi_{\hat n}$$. For $$\lvert h\rvert$$ small the chords shorter than $$\lvert h\rvert$$ occur only near the two extreme vertices of $$\Pi_{\hat n}$$. At such a vertex with edge vectors $$e,e'$$ pointing away from it, at transverse distance $$z$$ from it the chord has length $$z\,\lvert\det(e,e')\rvert/(\det(\hat n,e)\det(\hat n,e'))$$, so the last integral equals $$\lvert h\rvert^2\det(\hat n,e)\det(\hat n,e')/(2\lvert\det(e,e')\rvert)$$, and these two vertices are exactly those at which $$\det(\hat n,e)$$ and $$\det(\hat n,e')$$ have the same sign. Finally $$\lvert h\rvert\lvert\Pi_{\hat n}\rvert=\sum_i(\nu_i\cdot h)_-$$, since $$\sum_i\nu_i=0$$ gives $$\sum_i(\nu_i\cdot h)_+=\sum_i(\nu_i\cdot h)_-$$ and the left side is the projected extent of the boundary.
</div>

Equation (5) holds for every direction of $$h$$ with $$\lvert h\rvert<r_P$$. Inside the future quadrant $$\xi,\zeta>0$$ the function $$C$$ is quadratic except along rays through the origin in the directions of the timelike edges of $$P$$, where the term $$(\nu_i\cdot h)_-$$ has a kink: a direction $$h$$ with $$h_t>\lvert h_x\rvert$$ is timelike, so these rays are the directions of timelike edges.

<div class="proposition" markdown="1">
**Proposition 2** (expansion of $$H$$). Let $$P$$ be a convex polygon with area $$V_P$$ and let $$L_P$$ be the total proper length of its timelike edges. Let $$m_u$$ and $$m_w$$ be the values of $$\partial_u\partial_wC$$ on the sectors of the future quadrant adjacent to the $$u$$-axis and to the $$w$$-axis, respectively. There are $$\mu_0>0$$ and constants $$\eta_0,\eta_1$$ such that for $$0<\mu<\mu_0$$

$$
H(\mu)=2V_P\log\frac1\mu+\eta_0+2L_P\,\mu^{1/2}+(m_u+m_w)\,\mu\log\frac1\mu+\eta_1\mu+O(\mu^2),
$$

with a remainder that is an analytic function of $$\mu$$ times $$\mu^2$$ and carries no logarithm and no half-integer power. In particular there is no term $$\mu^{3/2}$$ and no term $$\mu^2\log\mu$$.
</div>

The function $$\mathcal O$$ is piecewise quadratic. The hyperbola crosses its cells at points whose positions are analytic in $$\sqrt\mu$$ or in $$1/\sqrt\mu$$, except at the kink rays, and the integral of a quadratic along the hyperbola gives powers of $$\mu$$ at the end points and logarithms only through the constant and cross terms. The logarithms telescope in step (4) of the proof, which is where the structure of $$H$$ is fixed.

<div class="proof" markdown="1">
*Proof.* (1) *Cells.* For a convex polygon the intersection $$P\cap(P-h)$$ is a convex polygon whose vertices are vertices of $$P$$ lying in $$P-h$$, vertices of $$P-h$$ lying in $$P$$, and intersections of an edge of $$P$$ with an edge of $$P-h$$. In a region of $$h$$ where the pattern of which vertices lie in which half-planes is fixed, these vertices are affine functions of $$h$$ and the area is a quadratic polynomial. The pattern changes only when a vertex of one polygon lies on the line through an edge of the other, that is, on one of the finitely many lines $$\nu_i\cdot h=\pm\nu_i\cdot(v_j-v_k)$$. So $$\mathcal O$$ is continuous and piecewise quadratic on the cells of a finite line arrangement, and vanishes outside a bounded set containing the origin in its interior.

(2) *Breakpoints.* Parametrise the hyperbola by $$\xi=\sqrt\mu\,y$$, $$\zeta=\sqrt\mu/y$$ with $$y>0$$. A line $$\nu^\xi\xi+\nu^\zeta\zeta=\Upsilon$$ meets it where $$\nu^\xi\sqrt\mu\,y^2-\Upsilon\,y+\nu^\zeta\sqrt\mu=0$$. For $$\Upsilon\neq0$$ the roots are $$y=[\Upsilon\pm(\Upsilon^2-4\nu^\xi\nu^\zeta\mu)^{1/2}]/(2\nu^\xi\sqrt\mu)$$ (or one root if $$\nu^\xi=0$$ or $$\nu^\zeta=0$$): a large root $$y=\mu^{-1/2}\chi_+(\mu)$$ and a small root $$y=\mu^{1/2}\chi_-(\mu)$$ with $$\chi_\pm$$ analytic and nonzero at $$\mu=0$$. For $$\Upsilon=0$$ the root is $$y_0=(-\nu^\zeta/\nu^\xi)^{1/2}$$, independent of $$\mu$$, and exists when $$\nu^\xi\nu^\zeta<0$$, which means that the line is the ray along a timelike edge. The order in which the hyperbola meets the finitely many lines changes only at values of $$\mu$$ where two crossing points coincide or two roots merge; each such condition is a nontrivial algebraic equation in $$\mu$$ and has finitely many positive roots, so the ordering is fixed below some $$\mu_0>0$$.

(3) *Integration.* On a cell, $$\mathcal O=\varpi_{00}+\varpi_{10}\xi+\varpi_{01}\zeta+\varpi_{20}\xi^2+\varpi_{11}\xi\zeta+\varpi_{02}\zeta^2$$, and along the hyperbola $$\xi\zeta=\mu$$, so

$$
\int \mathcal O\frac{dy}{y}=(\varpi_{00}+\varpi_{11}\mu)\log y+\sqrt\mu\,(\varpi_{10}y-\varpi_{01}/y)+\tfrac12\mu\,(\varpi_{20}y^2-\varpi_{02}/y^2).
$$

At a large root every term other than the logarithm is an analytic function of $$\mu$$ (for example $$\sqrt\mu\,y=\chi_+(\mu)$$ and $$\mu y^2=\chi_+^2$$), and $$\log y=-\tfrac12\log\mu+\log\chi_+$$. At a small root likewise, with $$\log y=+\tfrac12\log\mu+\log\chi_-$$. At a kink root $$y_0$$ the terms $$\sqrt\mu\,\varpi_{10}y_0$$ and $$-\sqrt\mu\,\varpi_{01}/y_0$$ carry $$\mu^{1/2}$$ and the others are analytic or constant. Hence $$H(\mu)$$ is the sum of an analytic function of $$\mu$$, a term in $$\mu^{1/2}$$ with no higher half-integer power, and logarithms of the form $$(A+B\mu)\log\mu$$.

(4) *Logarithms.* Because $$\mathcal O$$ is continuous, the sum of the logarithmic terms over the large roots telescopes: it equals $$-\tfrac12\log\mu$$ times the value of $$\varpi_{00}+\varpi_{11}\mu$$ on the innermost cell on that side (the cell adjacent to the origin) minus its value on the outermost cell, where $$\mathcal O\equiv0$$. The same holds for the small roots. On the innermost cells $$\varpi_{00}=\mathcal O(0)=2V_P$$ and $$\varpi_{11}=2m_u$$ or $$2m_w$$, by (5), since on the sector adjacent to an axis $$\mathcal O$$ is the quadratic polynomial from (5). Adding the two sides gives $$-2V_P\log\mu-(m_u+m_w)\mu\log\mu$$.

(5) *The term in $$\mu^{1/2}$$.* It comes from the kink roots, one for each timelike edge. By Proposition 1 the part of $$\mathcal O$$ that is linear in $$h$$ near the origin is $$-2\sum_i(\nu_i\cdot h)_-=-\sum_i\lvert\nu_i\cdot h\rvert$$ (using $$\sum_i\nu_i=0$$). For a timelike edge $$i$$ write $$\nu_i\cdot h=\nu^\xi\xi+\nu^\zeta\zeta$$ in null components, so that $$\nu^\xi=(\nu_{i,t}-\nu_{i,x})/2$$ and $$\nu^\zeta=(\nu_{i,t}+\nu_{i,x})/2$$; for a timelike edge $$\nu^\xi\nu^\zeta<0$$ and $$4\lvert\nu^\xi\nu^\zeta\rvert=\lvert\nu_{i,t}^2-\nu_{i,x}^2\rvert=\tau_i^2$$, the square of the proper length of the edge. The kink is at $$y_0^2=\lvert\nu^\zeta/\nu^\xi\rvert$$. For $$y<y_0$$ the term $$\nu^\zeta\zeta$$ dominates $$\nu_i\cdot h$$ and for $$y>y_0$$ the term $$\nu^\xi\xi$$ does, so the gradient of $$-\lvert\nu_i\cdot h\rvert$$ is $$(-\mathrm{sgn}(\nu^\zeta)\nu^\xi,-\lvert \nu^\zeta\rvert)$$ below the kink and $$(\mathrm{sgn}(\nu^\zeta)\nu^\xi,\lvert \nu^\zeta\rvert)$$ above it, a jump of $$J=(-2\lvert \nu^\xi\rvert,2\lvert \nu^\zeta\rvert)$$ in $$(\varpi_{10},\varpi_{01})$$. The contribution of the kink to $$H$$ is the difference of the antiderivative $$\sqrt\mu\,(\varpi_{10}y-\varpi_{01}/y)$$ across it, $$\sqrt\mu\,[-J_1y_0+J_2/y_0]=\sqrt\mu\,[2\lvert \nu^\xi\rvert y_0+2\lvert \nu^\zeta\rvert/y_0]=4\sqrt{\lvert \nu^\xi\nu^\zeta\rvert}\,\sqrt\mu=2\tau_i\sqrt\mu$$, since $$\lvert \nu^\zeta\rvert/y_0=\lvert \nu^\xi\rvert y_0$$. Summing over the timelike edges gives the coefficient $$2L_P$$. The remaining terms are analytic because the series in (3) converge for $$\mu<\mu_0$$.
</div>

As a worked example, take the triangle $$P_\triangle$$ with vertices $$(t,x)=(0,-1),(-0.3,1),(0.5,1.1)$$. Its area is $$V_P=163/200$$, it has one timelike edge, of proper length $$\sqrt{0.63}$$, and (5) gives $$m_u+m_w=807/652$$.

## 4. The action on a product region

<div class="theorem" markdown="1">
**Theorem 1** (expansion of the mean action). Let $$P$$ be a convex polygon with timelike edges of total proper length $$L_P$$, and consider the layer action (1) on $$\mathcal R=P\times\mathbb R^2$$ in the embedded regime, per unit transverse area. Suppose the kernel satisfies

(K1) $$Q(\tfrac12)=0$$, (K2) $$Q(1)=0$$ and $$b=-3Q'(1)$$, (K3) $$Q(\tfrac32)=0$$.

Then as $$\rho\to\infty$$

$$
\langle S\rangle=-16\,a_*\,Q(\tfrac54)\,L_P\,\rho^{1/4}+3\,Q'(\tfrac32)\,(m_u+m_w)+O(\rho^{-1/2}),\qquad a_*=\frac{\Gamma(\frac54)}{\sqrt6}\Big(\frac{24}{\pi}\Big)^{1/4}=0.615192\ldots \tag{6}
$$

If (K1) holds but (K2) or (K3) fails there are in addition a term of order $$\rho^{1/2}$$ proportional to $$V_P$$ (when $$Q(1)=0$$ but $$b\neq-3Q'(1)$$), a term of order $$\rho^{1/2}\log\rho$$ (when $$Q(1)\neq0$$), and a term of order $$\log\rho$$ (when $$Q(\tfrac32)\neq0$$; section 7).
</div>

*Proof.* By (2) and Lemma 1, $$\langle S\rangle=\kappa\rho^{-1/2}[b\rho V_P+6\rho\int\Psi(t)H(\sigma t)dt]$$. Insert the expansion of Proposition 2 and use Lemma 2. The term $$2V_P\log(1/\mu)$$ gives $$-6\rho\cdot2V_Pm_\Psi'(1)$$, and (K1)-(K2) give $$m_\Psi(1)=-\tfrac12\Gamma(1)Q(1)=0$$ and $$m_\Psi'(1)=-\tfrac14Q'(1)$$, so the contribution is $$3\rho V_PQ'(1)$$; with the element term $$b\rho V_P$$ the sum is $$\rho V_P(b+3Q'(1))=0$$. The constant $$\eta_0$$ meets $$m_\Psi(1)=0$$ and drops out. The term $$2L_P\mu^{1/2}$$ gives $$6\rho\cdot2L_P\sigma^{1/2}m_\Psi(\tfrac32)$$ with $$m_\Psi(\tfrac32)=-\tfrac13\Gamma(\tfrac54)Q(\tfrac54)$$, and $$\kappa\rho^{-1/2}\rho\sigma^{1/2}=\kappa\rho^{1/4}(24/\pi)^{1/4}$$ gives the first term of (6). The term $$(m_u+m_w)\mu\log(1/\mu)$$ gives $$6\rho\sigma(m_u+m_w)[-m_\Psi(2)\log\sigma-m_\Psi'(2)]$$; by (K3) $$m_\Psi(2)=-\tfrac14\Gamma(\tfrac32)Q(\tfrac32)=0$$ and $$m_\Psi'(2)=-\tfrac18\Gamma(\tfrac32)Q'(\tfrac32)$$, so the contribution is $$\kappa\rho^{-1/2}\cdot6\rho\sigma\cdot\tfrac18\Gamma(\tfrac32)Q'(\tfrac32)(m_u+m_w)$$; with $$\rho\sigma=(24\rho/\pi)^{1/2}$$, $$\kappa=4/\sqrt6$$ and $$\Gamma(\tfrac32)=\sqrt\pi/2$$ the numerical factor is $$\tfrac4{\sqrt6}\cdot6\sqrt{24}\cdot\tfrac1{16}=3$$. The constant $$\eta_1$$ meets $$m_\Psi(2)=0$$. The remainder $$O(\mu^2)$$ gives $$O(\rho\sigma^2)\kappa\rho^{-1/2}=O(\rho^{-1/2})$$. If (K2) or (K3) fails, the same computation leaves the terms listed.

The simplest region with corners is a two-dimensional diamond times a plane.

<div class="example" markdown="1">
**Example 1** (a two-dimensional diamond times a plane). Let $$P$$ be the causal diamond with null sides of length $$\ell_0$$, so that $$\mathcal O(\xi,\zeta)=(\ell_0-\xi)_+(\ell_0-\zeta)_+$$ on the future quadrant, $$L_P=0$$ and $$m_u=m_w=\tfrac12$$. For the Benincasa–Dowker kernel, which satisfies (K1)-(K3), Theorem 1 gives $$\langle S\rangle\to3\cdot\tfrac23\cdot1=2$$ per unit transverse area, for every $$\ell_0$$. For $$\ell_0=1$$ one has $$H(\mu)=(1+\mu)\log(1/\mu)-2+2\mu$$ for $$\mu<1$$, so that $$\eta_1=2$$.
</div>

This region is a product of a two-dimensional diamond with a plane, not the four-dimensional causal interval of [[BDJS]](#ref-bdjs), whose joint is a two-sphere. Its joint consists of the leftmost and rightmost vertices of $$P$$, each a plane of unit transverse area, so the volume of the joint is $$2$$, and the value $$2$$ agrees with the law found in [[BDJS]](#ref-bdjs) for the interval, that the mean tends to the volume of the joint.

For the triangle $$P_\triangle$$ and the Benincasa–Dowker kernel, (6) gives $$a_*L_P\rho^{1/4}+807/326=0.488294\,\rho^{1/4}+2.47546$$.

### Plain action on a timelike edge

For the Benincasa–Dowker kernel $$b=1$$, $$Q'_{\rm BD}(1)=-\tfrac13$$ and $$Q_{\rm BD}(\tfrac12)=Q_{\rm BD}(1)=Q_{\rm BD}(\tfrac32)=0$$, so (K1)-(K3) hold and (6) applies with $$Q_{\rm BD}(\tfrac54)=-\tfrac1{16}$$.

<div class="corollary" markdown="1">
**Corollary 1** (divergence at a timelike boundary). For a convex polygon $$P$$ with at least one timelike edge, the mean Benincasa–Dowker action per unit transverse area is

$$
\langle S\rangle=a_*\,L_P\,\rho^{1/4}+3Q'_{\rm BD}(\tfrac32)(m_u+m_w)+O(\rho^{-1/2}),
$$

so it diverges like $$\rho^{1/4}=\ell^{-1}$$ with a coefficient equal to $$a_*=0.615192$$ times the volume $$L_P$$ of the timelike boundary per unit transverse area.
</div>

The coefficient $$a_*$$ equals the slab coefficient given in closed form in [[DLLJ]](#ref-dllj) for four dimensions, which they compared with simulations.

### The family of finite kernels

The zero of $$Q$$ at $$\tfrac54$$ removes the divergence. The remaining freedom is the normalisation. By Theorem 1 the corner values are proportional to $$Q'(\tfrac32)$$. In flat space the coefficient of the d'Alembertian in the smooth limit of the kernel is also proportional to $$Q'(\tfrac32)$$ [[ASS]](#ref-ass), and the same universality holds in curved space [[Belenchia]](#ref-belenchia). We use these two statements as input. Fixing $$Q'(\tfrac32)=\tfrac23$$, the Benincasa–Dowker value, therefore keeps the curvature term at its Benincasa–Dowker coefficient and makes the joint of the diamond equal to $$2$$.

<div class="proposition" markdown="1">
**Proposition 3** (the family). Let $$D\ge3$$. The layer weights of degree at most $$D$$ with $$Q(\tfrac12)=Q(1)=Q(\tfrac54)=Q(\tfrac32)=0$$ and $$Q'(\tfrac32)=\tfrac23$$ are the polynomials

$$
Q=Q_\star+(s-\tfrac12)(s-1)(s-\tfrac54)(s-\tfrac32)^2\,r(s),\qquad Q_\star(s)=\tfrac{16}3(s-\tfrac12)(s-1)(s-\tfrac54)(s-\tfrac32),
$$

with $$\deg r\le D-5$$. There is none for $$D=3$$, exactly one for $$D=4$$, and an affine family of dimension $$D-4$$ for $$D\ge4$$. The element-count coefficient $$b=-3Q'(1)$$ of the minimal member is $$-1$$, and $$b$$ changes with $$r$$.
</div>

<div class="proof" markdown="1">
*Proof.* The five conditions are linear. On polynomials of degree at most $$4$$ the functionals evaluating at $$\tfrac12,1,\tfrac54$$ and evaluating the value and derivative at $$\tfrac32$$ are independent (Hermite interpolation), so $$Q_\star$$ is the unique solution of degree $$4$$ and the solutions of degree $$\le D$$ form an affine space of dimension $$D+1-5$$. A cubic with four zeros vanishes. The difference of two solutions has simple zeros at $$\tfrac12,1,\tfrac54$$ and a double zero at $$\tfrac32$$, hence is divisible by the quintic. For $$Q_\star$$, $$Q'_\star(1)=\tfrac13$$ so $$b=-1$$.
</div>

In the weights, $$Q_\star$$ corresponds to $$(c_0,\dots,c_4)=(5,-85,280,-328,128)$$ with $$b=-1$$, to be compared with $$b=+1$$ and $$(-1,9,-16,8)$$ for Benincasa–Dowker. For the minimal member $$Q_\star'(1)=\tfrac13$$ has the same sign as $$Q_\star'(\tfrac32)=\tfrac23$$, whereas $$Q'_{\rm BD}(1)=-\tfrac13$$ has the opposite sign, which is why $$b=-3Q'(1)$$ is $$-1$$ here and $$+1$$ for Benincasa–Dowker. The condition on $$b$$ is part of finiteness: with $$b=1$$ the minimal member would diverge as $$\rho^{1/2}V_P$$. The minimal member is minimal in degree and is not otherwise distinguished. The $$O(\rho^{-1/2})$$ term of Theorem 1 is proportional to $$Q(2)$$ times the coefficient of $$\mu^2$$ in $$H$$, so it differs between members.

## 5. Corner values

By Theorem 1 the finite part of the mean is $$3Q'(\tfrac32)(m_u+m_w)$$. We write it as $$\tfrac32Q'(\tfrac32)\sum_vj_v$$, where $$j_v$$ is the contribution of vertex $$v$$ to $$2(m_u+m_w)$$, and read $$j_v$$ off from the vertex terms of (5).

Some notation is needed to state the result. At a vertex $$v$$ with edge vectors $$e,e'$$ put $$\langle e,e'\rangle=e_te'_t-e_xe'_x$$. A spacelike edge vector is written $$e=s_e(\sinh\alpha,\cosh\alpha)$$ and a timelike one $$e=s_e(\cosh\alpha,\sinh\alpha)$$, with $$s_e=\pm1$$ and $$\alpha$$ the rapidity of the edge; $$e'$$ has sign $$s_{e'}$$ and rapidity $$\alpha'$$. Let $$n_v\in\{0,1,2\}$$ be the number of the two sectors of the future quadrant adjacent to the null axes (those next to the null directions $$(1,1)$$ and $$(1,-1)$$) on which $$\det(h,e)\det(h,e')>0$$. The sign of $$\det(h,e)$$ on these sectors is: for $$e$$ spacelike or null, $$\mathrm{sgn}\,e_x$$ on both; for $$e$$ timelike, $$-\mathrm{sgn}\,e_t$$ on the sector next to $$(1,1)$$ and $$+\mathrm{sgn}\,e_t$$ on the sector next to $$(1,-1)$$.

<div class="theorem" markdown="1">
**Theorem 2** (corner values). In the setting of Theorem 1, the finite part of the mean is $$\tfrac32Q'(\tfrac32)\sum_vj_v$$ with $$j_v=-\tfrac12\,n_v\,\langle e,e'\rangle/\lvert\det(e,e')\rvert$$. Explicitly:

| edges at the vertex | $$j_v$$ |
|---|---|
| spacelike, spacelike, same direction in space | $$\coth\theta$$, with $$\theta=\lvert\alpha-\alpha'\rvert$$ |
| spacelike, spacelike, opposite directions in space | $$0$$ |
| timelike $$e$$, spacelike $$e'$$ | $$-\tfrac12s_es_{e'}\tanh(\alpha'-\alpha)$$, which is $$0$$ for Lorentz-orthogonal edges |
| timelike, timelike, same time direction | $$-\coth\theta$$, with $$\theta=\lvert\alpha-\alpha'\rvert$$ |
| timelike, timelike, opposite time directions | $$0$$ |
| null, spacelike, same direction in space | $$+1$$ |
| null, spacelike, otherwise | $$0$$ |
| null, timelike, same time direction | $$-\tfrac12$$ |
| null, timelike, opposite time directions | $$+\tfrac12$$ |
| null, null, same direction in space | $$+1$$ |
| null, null, otherwise | $$0$$ |

</div>

<div class="proof" markdown="1">
*Proof.* By (5), $$m_u$$ and $$m_w$$ are sums over vertices of the mixed derivative $$\partial_u\partial_w$$ of $$[\det(h,e)\det(h,e')]_+/(2\lvert\det(e,e')\rvert)$$ on the sector in question, where $$\det(h,e)=\tfrac12(h_ue_w-h_we_u)$$ in null components. On a sector where the product is positive it is a quadratic form with $$\partial_u\partial_w[\det(h,e)\det(h,e')]=-\tfrac14(e_we'_u+e_ue'_w)=-\tfrac12\langle e,e'\rangle$$, so the vertex contributes $$-\langle e,e'\rangle/(4\lvert\det(e,e')\rvert)$$; where the product is not positive it contributes $$0$$. Hence $$m_u+m_w=\sum_vn_v\,(-\langle e,e'\rangle/4\lvert\det(e,e')\rvert)$$ and $$3Q'(\tfrac32)(m_u+m_w)=\tfrac32Q'(\tfrac32)\sum_vj_v$$ with $$j_v=-\tfrac12n_v\langle e,e'\rangle/\lvert\det(e,e')\rvert$$.

The signs on the sectors are those of $$\det(h,e)$$ at $$h=(1,\pm1)$$ moved slightly in the direction $$(1,0)$$, into the sector. For $$e=s_e(\sinh\alpha,\cosh\alpha)$$ (spacelike) one finds $$\mathrm{sgn}\,\det=\mathrm{sgn}\,e_x$$ on both sectors, for a null edge the same by direct computation, and for $$e=s_e(\cosh\alpha,\sinh\alpha)$$ (timelike) $$-s_e$$ at $$(1,1)$$ and $$+s_e$$ at $$(1,-1)$$. Now classify. For a spacelike pair, $$n_v=2[e_xe'_x>0]$$, and with $$e=s_e(\sinh\alpha,\cosh\alpha)$$, $$e'=s_{e'}(\sinh\alpha',\cosh\alpha')$$ one has $$\langle e,e'\rangle=-s_es_{e'}\cosh(\alpha-\alpha')$$ and $$\lvert\det(e,e')\rvert=\lvert\sinh(\alpha-\alpha')\rvert$$, so $$j_v=\coth\lvert\alpha-\alpha'\rvert$$ when $$s_es_{e'}>0$$. For a timelike pair $$e=s_e(\cosh\alpha,\sinh\alpha)$$, $$e'=s_{e'}(\cosh\alpha',\sinh\alpha')$$: $$n_v=2[s_es_{e'}>0]$$, $$\langle e,e'\rangle=s_es_{e'}\cosh(\alpha-\alpha')$$, $$j_v=-\coth\lvert\alpha-\alpha'\rvert$$. For timelike $$e$$ with spacelike $$e'$$ the products of signs on the two sectors are $$-s_es_{e'}$$ and $$+s_es_{e'}$$, so $$n_v=1$$; $$\langle e,e'\rangle=s_es_{e'}\sinh(\alpha'-\alpha)$$ and $$\lvert\det(e,e')\rvert=\cosh(\alpha'-\alpha)$$, giving $$j_v=-\tfrac12s_es_{e'}\tanh(\alpha'-\alpha)$$. For a null edge $$e=s_e(1,\pm1)$$ the same computations with $$e'$$ spacelike, timelike or null give $$\langle e,e'\rangle/\lvert\det(e,e')\rvert=-s_es_{e'}$$ (spacelike $$e'$$ with $$e_xe'_x>0$$), $$s_es_{e'}$$ (timelike $$e'$$) and $$-1$$ (null with $$e_xe'_x>0$$), which yields the last six rows.
</div>

The diamond of Example 1 has two vertices joining two null edges that point the same way in space (value $$1$$ each) and two that point opposite ways (value $$0$$), so $$\sum_vj_v=2$$. For the triangle $$P_\triangle$$ the vertices are one spacelike–spacelike joint with value $$\coth\theta=2.66871$$ and two timelike–spacelike joints with values $$-0.05828$$ and $$-0.13497$$, and the sum is $$807/326$$, the finite part found in section 4.

The corner values are universal: apart from the factor $$Q'(\tfrac32)$$, which the normalisation fixes, they do not depend on the kernel. A joint between two spacelike edges has the value $$\coth\theta$$ only when the two edges point the same way in space, which is the case at the leftmost and rightmost vertices of the region, where the past and the future boundary meet. A kink inside the past or future boundary, where the two edges point in opposite directions in space, has value $$0$$, so a rule that assigns $$\coth\theta$$ to every spacelike–spacelike corner is false.

Letting the direction of a timelike edge tend to null does not commute with $$\rho\to\infty$$. At fixed $$\rho$$ the mean is continuous in the polygon. As the edge direction of a timelike–timelike vertex tends to null, however, $$-\coth\theta\to-1$$, while the null–timelike value is $$-\tfrac12$$. A spacelike edge tilting to null changes nothing. How the approach to the null limit depends on $$\rho$$ is open (section 9).

## 6. Static timelike boundaries and the Gibbons–Hawking–York term

The variational principle of general relativity with a timelike boundary needs a boundary term $$\gamma_K\oint\mathcal K\,d\Sigma$$ with $$\gamma_K=1$$ in the normalisation in which the bulk term is $$\tfrac12\int R\sqrt{-g}$$, which is the normalisation of the Benincasa–Dowker action whose mean tends to this bulk term; here $$\mathcal K$$ is the trace of the extrinsic curvature of the boundary and $$d\Sigma$$ its three-dimensional volume element. Does the mean action contain such a term? For the regions of the previous sections the boundary is flat and $$\mathcal K=0$$, so a test needs a curved boundary. We take the static product regions $$\mathcal R=[0,T]\times\Omega$$, with $$\Omega\subset\mathbb R^3$$ a convex body. Let $$C_\Omega(x)=\lvert\Omega\cap(\Omega-x)\rvert$$ be the covariogram of $$\Omega$$ and $$\bar C(r)$$ its average over directions, expanded near $$r=0$$ as $$\bar C(r)=g_0+g_1r+g_2r^2+g_3r^3+\dots$$

<div class="theorem" markdown="1">
**Theorem 3** (static product regions). Let $$\Omega$$ be a convex body with $$\bar C(r)=g_0+g_1r+g_2r^2+O(r^3)$$ as $$r\to0$$, and suppose that $$\bar C$$ is smooth except at finitely many values of $$r$$. Assume that $$T$$ is neither the diameter of $$\Omega$$ nor one of those values. For the layer action in the embedded regime with $$Q(\tfrac12)=Q(1)=Q(\tfrac32)=0$$ and $$b=-3Q'(1)$$,

$$
\langle S\rangle=-16\,a_*\,Q(\tfrac54)\,{\rm Vol}_3\,\rho^{1/4}-9\,Q'(\tfrac32)\,g_2\,T+O(\rho^{-1/4}), \tag{7}
$$

where $${\rm Vol}_3=T\,{\rm Area}(\partial\Omega)$$ is the three-volume of the timelike boundary and $$a_*$$ is as in (6). If moreover $$\bar C(r)=g_0+g_1r+g_2r^2+g_3r^3+O(r^4)$$ then the next term is $$-\tfrac{8\pi}{15}\kappa\,(24/\pi)^{7/4}\Gamma(\tfrac74)\,Q(\tfrac74)\,g_3\,T\,\rho^{-1/4}$$.
</div>

As in section 3, the action depends on the covariogram of the region near zero separation. Here the region factorises into time and space, and the only non-analytic dependence on the proper-time variable comes from the small-distance expansion of $$\bar C$$.

<div class="proof" markdown="1">
*Proof.* The covariogram of $$\mathcal R$$ at the separation $$(\Delta t,\Delta x)$$ is $$(T-\Delta t)_+\,C_\Omega(\Delta x)$$. Writing $$W=\tau^2=\Delta t^2-r^2$$ with $$r=\lvert\Delta x\rvert$$, the pair integral in (2) is $$\rho^2\int dW\,f(\rho\tfrac\pi{24}W^2)G(W)$$ with

$$
G(W)=2\pi\int_0^{r_m(W)}r^2\bar C(r)\Big[\frac T{\sqrt{r^2+W}}-1\Big]dr,\qquad r_m=\min\big(r_{\max},\sqrt{T^2-W}\big),
$$

where $$r_{\max}$$ is the diameter of $$\Omega$$, since $$\Delta t=\sqrt{r^2+W}$$ at fixed $$r$$ and $$W$$, so that $$d\Delta t=dW/(2\Delta t)$$, and $$d^3\Delta x=4\pi r^2dr$$ after averaging over directions. The non-analytic dependence of $$G$$ on $$W$$ comes from small $$r$$, and the region $$r\ge r_1$$ and the endpoint $$r_m$$ give smooth functions of $$W$$ near $$0$$. For the endpoint this needs $$\bar C$$ to be smooth near $$r=T$$, the value of $$r_m$$ at $$W=0$$ when $$T$$ is below the diameter. Above the diameter $$r_m$$ is constant. For the terms $$g_nr^n$$ of $$\bar C$$ the integrals $$\int_0^{r_1}r^{n+2}(r^2+W)^{-1/2}dr$$ have singular parts $$\tfrac W4\ln W$$, $$\tfrac23W^{3/2}$$, $$-\tfrac3{16}W^2\ln W$$, $$-\tfrac8{15}W^{5/2}$$ for $$n=0,1,2,3$$ (from the antiderivatives, for example $$\int r^3(r^2+W)^{-1/2}=\tfrac13(r^2-2W)\sqrt{r^2+W}$$). For the remainder $$\varepsilon(r)=O(r^3)$$ one expands $$(1+x)^{-1/2}=1-\tfrac x2+\tfrac38x^2+O(x^3)$$ with $$x=W/r^2$$ for $$r>\sqrt W$$ and bounds the region $$r<\sqrt W$$ directly. This gives a smooth function of $$W$$, which by Taylor's theorem is a polynomial of degree $$2$$ plus $$O(W^{5/2})$$, with no logarithm. So

$$
G(W)=a_0+a_1W+a_2W^2+2\pi T\Big[\tfrac14g_0\,W\ln W+\tfrac23g_1\,W^{3/2}-\tfrac3{16}g_2\,W^2\ln W\Big]+O(W^{5/2}).
$$

The Mellin integral of $$W^\beta\ln^mW$$ against $$f(\rho\tfrac\pi{24}W^2)$$ is $$\tfrac12(\rho\pi/24)^{-s}\Gamma(s)Q(s)$$ with $$s=(\beta+1)/2$$, and with a logarithm the derivative in $$\beta$$. The polynomial terms $$W^0,W^1,W^2$$ sit at $$s=\tfrac12,1,\tfrac32$$ and vanish. The $$W\ln W$$ term gives $$3\rho\,{\rm Vol}_4Q'(1)$$ with $${\rm Vol}_4=Tg_0$$, cancelled by the element term. The $$W^{3/2}$$ term gives $$\tfrac{2\pi}3g_1T\rho^{3/4}(24/\pi)^{5/4}\Gamma(\tfrac54)Q(\tfrac54)$$; for a convex body $$g_1=-{\rm Area}(\partial\Omega)/4$$ (Cauchy's projection formula: the mean area of the shadow of a convex body is a quarter of its surface area, and $$C_\Omega(\lvert x\rvert\hat n)=\mathrm{Vol}(\Omega)-\lvert x\rvert\,\lvert\Pi_{\hat n}\Omega\rvert+O(\lvert x\rvert^2)$$ by the chord identity of Proposition 1), which with $$\kappa\rho^{-1/2}$$ gives the first term of (7). The $$W^2\ln W$$ term gives $$-\tfrac{3\pi}{32}Tg_2\rho^{1/2}(24/\pi)^{3/2}\Gamma(\tfrac32)Q'(\tfrac32)$$ (because $$Q(\tfrac32)=0$$), and $$\kappa\rho^{-1/2}$$ times this equals $$-9Q'(\tfrac32)g_2T$$. The $$W^{5/2}$$ term is $$O(\rho^{-1/4})$$, with the explicit coefficient when the expansion of $$\bar C$$ extends to $$r^3$$.
</div>

For the ball $$\Omega=\{\lvert x\rvert\le R\}$$ the covariogram is the lens volume $$\tfrac{4\pi}{3}R^3-\pi R^2r+\tfrac\pi{12}r^3$$ for $$r<2R$$, so $$g_2=0$$ and $$g_3=\pi/12$$. By (7) the finite part vanishes, and the first correction is the $$\rho^{-1/4}$$ term with $$g_3$$. The ball therefore does not test the coefficient of $$g_2$$ in (7). The cube, treated below, does.

### When $$g_2$$ vanishes

The finite part is proportional to $$g_2$$. For which bodies is it zero?

<div class="proposition" markdown="1">
**Proposition 4** ($$C^{1,1}$$ bodies). If $$\Omega\subset\mathbb R^3$$ is a convex body whose boundary is $$C^{1,1}$$ (a ball of some radius $$r_c>0$$ rolls freely inside $$\Omega$$ at every boundary point), then $$\bar C(r)=g_0+g_1r+O(r^3)$$, so $$g_2=0$$. Hence, whenever the remaining hypotheses of Theorem 3 hold for $$\Omega$$ and $$T$$, the finite part in (7) vanishes for every kernel of Theorem 3.
</div>

<div class="proof" markdown="1">
*Proof.* By the chord identity, for a unit vector $$\hat n$$, $$C_\Omega(r\hat n)=\mathrm{Vol}(\Omega)-r\lvert\Pi_{\hat n}\Omega\rvert+\int(r-\Lambda_{\hat n}(z))_+dz$$, where $$\Lambda_{\hat n}(z)$$ is the length of the chord of $$\Omega$$ along $$\hat n$$ through the point $$z$$ of the shadow $$\Pi_{\hat n}\Omega\subset\hat n^\perp$$. Let $$z$$ be a point of the shadow at distance $$a$$ from the shadow boundary, and $$z_0$$ the nearest boundary point of the shadow. The corresponding point $$x_0\in\partial\Omega$$, at which the tangent plane is parallel to $$\hat n$$, has its inward normal along $$z-z_0$$, so the plane through $$x_0$$ spanned by $$\hat n$$ and that normal cuts the rolling ball in a disc of radius $$r_c$$ tangent to the line through $$x_0$$ parallel to $$\hat n$$. The chord through $$z$$ contains the chord of this disc at distance $$a$$ from the tangent line, so $$\Lambda_{\hat n}(z)\ge2\sqrt{2r_ca-a^2}$$ for $$a\le r_c$$. Let $$r<2r_c$$ and $$a_r=r_c-\sqrt{r_c^2-r^2/4}=r^2/(8r_c)+O(r^4)$$, the solution of $$2\sqrt{2r_ca-a^2}=r$$ in $$(0,r_c]$$. The chord length $$\Lambda_{\hat n}$$ is concave on the shadow, since the chord over a convex combination of two points contains the convex combination of the two chords. On the convex set $$A$$ of points of the shadow at distance at least $$a_r$$ from its boundary, a concave function attains its minimum at an extreme point, and every extreme point of $$A$$ is at distance exactly $$a_r$$ from the boundary. By the bound above, $$\Lambda_{\hat n}\ge r$$ on $$A$$. Hence $$\Lambda_{\hat n}(z)<r$$ only within distance $$a_r$$ of the shadow boundary, a collar of area at most $${\rm Per}(\Pi_{\hat n}\Omega)\,a_r$$, and the last integral is at most $$r\cdot{\rm Per}(\Pi_{\hat n}\Omega)\cdot r^2/(8r_c)\,(1+O(r^2))=O(r^3)$$ uniformly in $$\hat n$$, the perimeter of the shadow being bounded by that of a ball containing $$\Omega$$.
</div>

Balls and ellipsoids, and every convex body with a $$C^{1,1}$$ boundary for which Theorem 3 applies, therefore have no finite term in (7), although the lateral boundary of $$[0,T]\times\{\lvert x\rvert\le R\}$$ has $$\mathcal K=2/R$$ and $$\oint\mathcal K\,d\Sigma=8\pi RT$$. A Gibbons–Hawking–York term with coefficient $$\gamma_K$$ would add $$\gamma_K\cdot8\pi RT$$ to the finite part, a quantity that grows linearly with $$R$$, while the computed finite part is zero for every $$R$$. The faces $$t=0$$ and $$t=T$$ meet the lateral boundary at right angles in a sphere, and Theorem 2, which is proved for planar joints, does not cover these spherical joints. This does not affect the conclusion. A term attached to the joint at $$t=0$$ or $$t=T$$ is built from the same sphere for every $$T$$ and does not depend on $$T$$, whereas $$\gamma_K\cdot8\pi RT$$ is proportional to $$T$$, and the finite part vanishes for every admissible $$T$$. So on these regions the finite part of the mean is not of the form $$\gamma_K\oint\mathcal K\,d\Sigma$$ plus $$T$$-independent joint terms with $$\gamma_K\neq0$$, whereas the variational principle needs $$\gamma_K=1$$.

For bodies with edges, $$g_2\neq0$$. For the cube $$\Omega=[0,X]^3$$ one has $$C_\Omega(r\hat n)=\prod_i(X-r\lvert n_i\rvert)$$ for $$r<X$$, so $$g_1=-\tfrac32X^2$$ (equal to $$-{\rm Area}/4$$ since $$\langle\lvert n_i\rvert\rangle=\tfrac12$$) and $$g_2=3X\langle\lvert n_1n_2\rvert\rangle=2X/\pi$$, with $$\langle\lvert n_1n_2\rvert\rangle=2/(3\pi)$$ on the sphere. Since $$\bar C$$ changes form at $$r=X$$, the exceptional values of $$T$$ in Theorem 3 include $$X$$ and the diameter $$\sqrt3X$$; for the others the finite part is $$-9\cdot\tfrac23\cdot\tfrac{2X}\pi T=-12XT/\pi$$, that is $$-1/\pi$$ for each of the twelve edges of the cube, per unit area of the edge swept in time.

<div class="proposition" markdown="1">
**Proposition 5** (edges of prisms). Let $$\Omega=P_2\times[0,Z]$$ with $$P_2$$ a convex polygon in the plane and $$Z\to\infty$$. Per unit length of the prism and for the kernels of Theorem 3, the finite part of (7) equals $$-\sum_v\big(1-\delta_v\cot\delta_v\big)\,T/\pi$$, the sum over the vertices of $$P_2$$ with exterior angles $$\delta_v$$.
</div>

<div class="proof" markdown="1">
*Proof.* Per unit length, $$C_\Omega$$ at the separation $$x$$ is the planar covariogram of $$P_2$$ at the projection of $$x$$ onto the plane of $$P_2$$. By Proposition 1 applied to $$P_2$$ (with the Euclidean metric), the $$r^2$$ term of the planar covariogram is the sum of the vertex terms $$[\det(h,e_v)\det(h,e'_v)]_+/(2\lvert\det(e_v,e'_v)\rvert)$$. The projection of a uniformly random unit vector onto the plane has mean square length $$\tfrac23$$, so $$g_2=\tfrac23\sum_v\Phi(\delta_v)$$, where $$\Phi$$ is the average over the direction of $$h$$ in the plane of the vertex term for unit edge vectors, $$\Phi(\delta)=\frac1{2\pi}\int_0^{2\pi}\frac{[\det(u_\omega,e)\det(u_\omega,e')]_+}{2\lvert\det(e,e')\rvert}d\omega=\frac{1-\delta\cot\delta}{4\pi}$$ for edges meeting at the interior angle $$\pi-\delta$$, with $$u_\omega=(\cos\omega,\sin\omega)$$. The finite part is $$-9\cdot\tfrac23g_2T=-4\sum_v\Phi(\delta_v)T$$.
</div>

For a square prism the formula gives $$-4T/\pi=-1.27324\,T$$ per unit length, in agreement with the cube. For an equilateral triangle it gives $$-3(1+\tfrac{2\pi}{3\sqrt3})T/\pi=-2.1096\,T$$, and for a regular $$n$$-gon $$-n(1-\delta\cot\delta)T/\pi\approx-4\pi T/(3n)$$, which tends to zero as the polygon approaches a disc, consistent with Proposition 4.

The edge values bear on corner terms of the usual form. Consider a boundary term $$\gamma_K\oint\mathcal K\,d\Sigma+\gamma_J\sum_e\delta_e\,{\rm Area}(e)$$, the first term integrated over the smooth part of the boundary and the second a Hayward-type corner term (exterior angle $$\delta_e$$ times edge area). Rounding an edge of a prism with a radius $$r_c>0$$ produces a $$C^{1,1}$$ boundary, for which the finite part is zero however small $$r_c$$ is (Proposition 4, applied per unit length to the planar cross-section, where the same chord argument holds with a rolling disc), while the sharp edge has the value just computed. At fixed density the mean is continuous in $$r_c$$, so the limits $$\rho\to\infty$$ and $$r_c\to0$$ do not commute. We expect the crossover at $$r_c$$ of the order of the discreteness length.

On a rounded prism the corner term is absent and the integral of $$\mathcal K$$ over each rounded arc equals the exterior angle, so the boundary term equals $$2\pi\gamma_KT$$ per unit length. The mean action gives $$0$$, so $$\gamma_K=0$$. On a sharp prism $$\mathcal K=0$$ on the faces and the boundary term is $$2\pi\gamma_JT$$ whatever the polygon, because the exterior angles sum to $$2\pi$$, while the mean action gives $$-4T/\pi$$ for the square and $$-2.1096\,T$$ for the triangle. So no pair $$(\gamma_K,\gamma_J)$$ reproduces the finite part of the mean action on prisms. The corner function that the mean does produce, $$-(1-\delta\cot\delta)/\pi$$, is not proportional to the exterior angle $$\delta$$ that the Hayward term carries.

<div class="remark" markdown="1">
**Remark** (curved edges and corners). On regions of the form $$P\times\mathbb R^2$$ a term $$\gamma_K\oint\mathcal K\,d\Sigma$$ equals $$\gamma_K$$ times the transverse area times $$\int_{\partial P}k\,ds$$, with $$k$$ the curvature of the boundary curve. Along a smooth timelike arc in the Minkowski plane the integral of $$k$$ is the change of the rapidity of the tangent between the end points of the arc, which is the same quantity that a corner with that boost angle carries. On such regions a curved timelike edge therefore cannot be distinguished from corner terms, and this is why the test above uses the ball, for which $$\oint\mathcal K\,d\Sigma=8\pi RT$$ grows with $$R$$ while the finite part of the mean is zero for every $$R$$.
</div>

## 7. Interval functionals at a flat joint

The area law has a factor $$2\pi$$ that arises, in the Euclidean treatment, from a conical singularity of opening $$2\pi$$. A causal-set action could reproduce it through a joint function that, continued to imaginary boost angle, is a nonzero multiple of the deficit angle plus a function of period $$2\pi$$, as the Regge and Hayward joint terms are, or has least period $$2\pi$$, the period of a cone. We test this for functionals of the interval alone: a pair functional $$F$$ that depends only on the induced order of the interval of $$p\prec q$$. In flat Minkowski space the Poincaré group maps every timelike interval of volume $$V$$ to the standard interval of that volume, so $$\mathbb E[F]=F_{\rm eff}(\rho V_{pq})$$ for a function $$F_{\rm eff}$$, and the mean has the form (2) with $$f=F_{\rm eff}$$. Chain counts, longest-chain statistics and layer counts are of this kind. We assume that $$F_{\rm eff}$$ is bounded with $$F_{\rm eff}(v)=O(e^{-\epsilon v})$$ at large $$v$$, so that its Mellin transform $$M_F(s)=\Gamma(s)Q_F(s)$$ is analytic for $${\rm Re}\,s>0$$ (this excludes kernels with power-law tails, whose transforms have poles), that $$Q_F(\tfrac12)=Q_F(1)=0$$, and that the element count is chosen to cancel the volume term.

<div class="proposition" markdown="1">
**Proposition 6** (flat joints). For $$P\times\mathbb R^2$$ with $$P$$ a convex polygon whose edges are spacelike or null, the mean of such a functional is

$$
\langle S_F\rangle=3(m_u+m_w)\big[Q_F'(\tfrac32)-Q_F(\tfrac32)\log\rho\big]+Q_F(\tfrac32)\,\Theta_P+O(\rho^{-1/2}),\qquad \Theta_P=3(m_u+m_w)\Big[\psi^{(0)}(\tfrac32)-1+\log\frac{24}\pi\Big]-6\eta_1, \tag{8}
$$

where $$\psi^{(0)}$$ is the digamma function and $$\eta_1$$ is the coefficient of $$\mu$$ in the expansion of Proposition 2. In particular, if $$Q_F(\tfrac32)=0$$ the joint contribution is that of Theorem 2, $$\tfrac32Q_F'(\tfrac32)j_v$$ with $$j_v=\coth\theta_v$$ at the joints between spacelike edges of the same spatial direction and $$1$$ at the corresponding joints with a null edge. If $$Q_F(\tfrac32)\neq0$$ the logarithmic term is still proportional to $$m_u+m_w=\tfrac12\sum_vj_v$$, and the constant $$\Theta_P$$ is not a sum over joints: under $$P\to\lambda P$$ one finds $$\eta_1\to\eta_1+2(m_u+m_w)\log\lambda$$, so $$\Theta_P$$ depends on the size of the polygon.
</div>

*Proof.* The derivation of Theorem 1 used the functional only through the moments $$m_\Psi$$. Without (K3) the term $$(m_u+m_w)\mu\log(1/\mu)$$ of $$H$$ contributes $$6\rho\sigma\,(m_u+m_w)[-m_\Psi(2)\log\sigma-m_\Psi'(2)]$$ with $$m_\Psi(2)=-\tfrac14\Gamma(\tfrac32)Q_F(\tfrac32)\neq0$$, and the term $$\eta_1\mu$$ contributes $$6\rho\sigma\eta_1m_\Psi(2)$$. Using $$m_\Psi'(2)=-\tfrac18\Gamma(\tfrac32)[Q_F'(\tfrac32)+(\psi^{(0)}(\tfrac32)-1)Q_F(\tfrac32)]$$, $$\sigma=(24/\pi\rho)^{1/2}$$, and the numerical factor $$3$$ of Theorem 1 gives (8). For the scaling, $$\mathcal O_{\lambda P}(\xi,\zeta)=\lambda^2\mathcal O_P(\xi/\lambda,\zeta/\lambda)$$ gives $$H_{\lambda P}(\mu)=\lambda^2H_P(\mu/\lambda^2)$$, in which the terms $$(m_u+m_w)\mu\log(1/\mu)+\eta_1\mu$$ become $$(m_u+m_w)\mu\log(1/\mu)+[\eta_1+2(m_u+m_w)\log\lambda]\mu$$.

For the diamond of Example 1 with null sides of length $$\ell_0$$, the scaling gives $$H(\mu)=(\ell_0^2+\mu)\log(\ell_0^2/\mu)-2\ell_0^2+2\mu$$, so that $$m_u+m_w=1$$ for every $$\ell_0$$ and $$\eta_1=2+2\log\ell_0$$: doubling the sides adds $$2\log2$$ to $$\eta_1$$ and $$-12\log2$$ to $$\Theta_P$$.

For $$Q_F(\tfrac32)=0$$, and for the coefficient of $$\log\rho$$ otherwise, the leading joint term is therefore a multiple of $$\coth\theta$$, with the value $$1$$ for a null edge, whatever the interval functional. What is proved is this form, a multiple of $$\coth\theta$$. The half-angle function $$\coth(\theta/2)=\coth\theta+\operatorname{csch}\theta$$ is not a multiple of $$\coth\theta$$ and does not occur. The two criteria above are the continuation of this statement to imaginary angle. There $$\coth(i\varphi)=-i\cot\varphi$$ has least period $$\pi$$, not $$2\pi$$. A function $$g$$ of period $$\pi$$ that equals $$c\varphi$$ plus a function of period $$2\pi$$ has $$c=0$$, because $$g(\varphi+2\pi)-g(\varphi)$$ is $$2\pi c$$ in the one description and $$0$$ in the other. The half-angle function continues to $$-i\cot(\varphi/2)$$, whose least period is the $$2\pi$$ of a cone. In this sense the factor $$2\pi$$ is not carried by the leading joint term of these actions at a flat joint.

When $$Q_F(\tfrac32)\neq0$$ the $$O(1)$$ part is $$Q_F(\tfrac32)\Theta_P$$, with $$\Theta_P$$ containing $$\eta_1$$, a global property of the polygon that is not a sum over joints. How $$\eta_1$$ depends on a joint angle is not determined, so a half-angle piece in that part is not excluded by the analysis. The statement also requires a functional of the interval alone, flat joints with spacelike or null edges, and a symbol analytic at $$\tfrac32$$. Functionals that use data outside the interval (a slice, a horizon, spacelike pairs), kernels with a pole at $$s=\tfrac32$$, curved backgrounds and boost-conical hinges, at which the reduction of section 2 fails, are outside it. The molecule count of the page on [horizon counts and the area law](/quantum-gravity/area-projector/) is defined relative to a slice and is not an interval functional.

## 8. The isolated regime

The proofs above use the embedded regime, in which the volume of the interval of a pair does not depend on the boundary. In the isolated regime the interval is intersected with the region, so a boundary near the pair reduces the volume used in the kernel. The statements of this section are expectations, and the results of sections 2 to 7 do not depend on them.

For a flat timelike face the stabiliser $$SO(2,1)$$ of the face leaves the cut volume invariant, so the pair integral would factorise into an integral over the orbit of the stabiliser and a Mellin integral at $$s=1,\tfrac54,\tfrac32$$. If the orbit integral converges absolutely, the difference between the isolated and embedded means per unit area of the face is proportional to $$Q(\tfrac54)$$, members of the family of Proposition 3 have no face term in either regime, and corners with one or two timelike faces take their embedded values. Absolute convergence in the presence of two faces is not shown.

For a curved face, a pair moving tangentially to a boundary of radius of curvature $$R_{\rm c}$$ has its interval cut by the sagitta $$\tau^2u_t^2/(2R_{\rm c})$$ of the boundary, with $$u_t$$ the tangential boost. This cuts off the boost integral at $$u_t\sim(R_{\rm c}/\tau)^{1/2}$$. The conjecture is that the Mellin points at which boundary terms sit then move from the single value $$s=\tfrac54$$ to a tower $$s=1+k/8$$, $$k\ge1$$, of which only $$s=\tfrac98,\tfrac54,\tfrac{11}8$$ give divergent powers $$\rho^{3/2-s}$$, namely $$\rho^{3/8}$$, $$\rho^{1/4}$$ and $$\rho^{1/8}$$. The term at $$s=\tfrac98$$ would be proportional to $$Q(\tfrac98)\oint d\Sigma\int d\phi\,k(\phi)^{-1/2}$$, with $$k(\phi)$$ the normal curvature along the direction of motion, so it is not analytic in the extrinsic curvature. For the minimal member of Proposition 3, $$Q_\star(\tfrac98)=\tfrac5{256}\neq0$$. If the conjecture holds, the family with a zero at $$\tfrac54$$ is not finite on a curved timelike boundary in the isolated regime.

The zero $$Q(\tfrac98)=0$$ would not suffice, since a term proportional to $$Q(\tfrac{11}8)$$ would remain and, for a boundary with a flat direction such as a cylinder, one proportional to $$Q'(\tfrac98)$$. Kernels with zeros at $$\tfrac12,1,\tfrac98,\tfrac54,\tfrac{11}8,\tfrac32$$ and $$Q'(\tfrac32)=\tfrac23$$ (degree six), or with double zeros at $$\tfrac98$$ and $$\tfrac{11}8$$ in addition to the zeros at $$\tfrac12,1,\tfrac54,\tfrac32$$ (degree eight), would then be finite. Whether logarithms occur in the tower and whether the exponent set is complete are open.

## 9. Boundaries and backgrounds beyond static products

Theorem 3 gives the finite part on static products only. A proof that $$H$$ keeps the form of Proposition 2 for curved faces, where the covariogram is not piecewise polynomial, would settle whether the finite part on a general curved timelike boundary is a local integral over the boundary, and which one; the same extension would be needed for boosted, accelerated or otherwise non-static boundaries. The $$\rho^{-1/4}$$ term of (7), proportional to $$Q(\tfrac74)g_3T$$, depends on the boundary through $$g_3$$. For the ball it is independent of $$R$$, as is $$\ell\oint\mathcal K^2d\Sigma=16\pi\ell T$$, and its form for general boundaries is not determined. The dependence on $$\rho$$ of the corner values near the null limit (section 5) is also undetermined.

On curved backgrounds the reduction of section 2 to a covariogram is unavailable, and at a boost-conical hinge the structure used in sections 2 and 3 does not describe the geometry, so the corner function there is not fixed by the flat-joint analysis. Extending Proposition 6 would require treating kernels outside the layer family (continuum kernels with a pole at $$s=\tfrac32$$ or with power-law tails), functionals that use data outside the interval, and non-Poisson substrates. A factor $$2\pi$$ could also arise from the state of the matter fields rather than from the action, and the results here concern the action only.

The action $$S$$ is a random variable, and the absence of a Gibbons–Hawking–York term in its mean says nothing about its variance, which is not examined here.

## References

1. <a id="ref-bd"></a>Benincasa, Dowker, “The scalar curvature of a causal set,” Phys. Rev. Lett. 104, 181301 (2010), [arXiv:1001.2725](https://arxiv.org/abs/1001.2725).
2. <a id="ref-dg"></a>Dowker, Glaser, “Causal set d'Alembertians for various dimensions,” Class. Quantum Grav. 30, 195016 (2013), [arXiv:1305.2588](https://arxiv.org/abs/1305.2588).
3. <a id="ref-glaser"></a>Glaser, “A closed form expression for the causal set d'Alembertian,” [arXiv:1311.1701](https://arxiv.org/abs/1311.1701).
4. <a id="ref-ass"></a>Aslanbeigi, Saravani, Sorkin, “Generalized causal set d'Alembertians,” JHEP 1406, 024 (2014), [arXiv:1403.1622](https://arxiv.org/abs/1403.1622).
5. <a id="ref-belenchia"></a>Belenchia, “Universal behaviour of generalized causal set d'Alembertians in curved spacetime,” Class. Quantum Grav. 33, 135011 (2016), [arXiv:1510.04665](https://arxiv.org/abs/1510.04665).
6. <a id="ref-bdjs"></a>Buck, Dowker, Jubb, Surya, “Boundary terms for causal sets,” Class. Quantum Grav. 32, 205004 (2015), [arXiv:1502.05388](https://arxiv.org/abs/1502.05388).
7. <a id="ref-dowker"></a>Dowker, “Boundary contributions in the causal set action,” Class. Quantum Grav. 38, 075018 (2021), [arXiv:2007.13206](https://arxiv.org/abs/2007.13206).
8. <a id="ref-mw"></a>Machet, Wang, “On the continuum limit of Benincasa–Dowker–Glaser causal set action,” Class. Quantum Grav. 38, 015010 (2021), [arXiv:2007.13192](https://arxiv.org/abs/2007.13192).
9. <a id="ref-dllj"></a>Dowker, Liu, Lloyd-Jones, “Timelike boundary and corner terms in the causal set action,” Class. Quantum Grav. 42, 205006 (2025), [arXiv:2501.00139](https://arxiv.org/abs/2501.00139).
10. <a id="ref-chevalier"></a>Chevalier, “The Discrete Causal Action and Holes in Spacetime,” MSc dissertation, Imperial College London, Department of Physics (2023), as cited in [[DLLJ]](#ref-dllj).
