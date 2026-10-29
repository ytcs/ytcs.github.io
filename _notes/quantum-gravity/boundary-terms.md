---
title: "Boundary and corner terms of the causal-set action"
date: 2026-10-29
category: entropy
order: 2
description: The mean Benincasa–Dowker action of a sprinkling into a region with a timelike boundary diverges with the density, a family of generalised kernels with extra Mellin zeros is finite, its corner values do not depend on the kernel, and on static boundaries the finite part contains no Gibbons–Hawking–York term. For interval functionals on flat joints the corner function is proportional to the hyperbolic cotangent of the boost angle, so the factor of two pi in the area law is not produced by the action.
status: "Theorems proved in full for flat product regions in four dimensions in the embedded regime, checked by exact arithmetic and by numerical quadrature of the exact mean; the isolated-regime statements for curved boundaries are derivation sketches with numerical evidence that is quoted and not reproduced."
excludes: "A Gibbons–Hawking–York term (or any term proportional to the integrated extrinsic curvature, alone or combined with a Hayward corner term) in the finite part of the mean causal-set action on static timelike boundaries; a half-angle corner function, and so the conical factor of two pi, from interval-intrinsic functionals on flat joints."
---

Does the mean of the causal-set action on a region contain the Gibbons–Hawking–York boundary term, what does it assign to a corner where two boundary pieces meet, and does it produce the factor $$2\pi$$ of the area law? The Benincasa–Dowker action is a linear combination of counts of pairs of sprinkled points, classified by the number of points between them. Its mean is expected to equal the Einstein–Hilbert action of the region plus terms carried by its boundary and by the codimension-two surfaces (joints) where boundary pieces meet.

For four-dimensional flat regions that are products of a convex polygon with a transverse plane, or static regions $$[0,T]\times S$$ with $$S$$ a convex body, for the mean over sprinklings, and with the points outside the region allowed to lie between the endpoints of a counted pair, the answers are as follows. The plain action has a term proportional to the inverse discreteness length $$\rho^{1/4}$$ and to the volume of the timelike boundary. A zero of the Mellin symbol of the kernel at $$5/4$$ removes it, and the kernels of this family have a finite limit whose corner values do not depend on the kernel; at a joint between two spacelike edges that point the same way in space the value is $$\coth\theta$$, with $$\theta$$ the boost angle. On a static boundary the finite part is $$-9Q'(\tfrac32)a_2T$$, with $$a_2$$ the coefficient of $$r^2$$ in the covariogram of $$S$$. It vanishes for every body with a $$C^{1,1}$$ boundary, although the extrinsic curvature of the boundary does not, so the finite part contains no Gibbons–Hawking–York term; polygonal prisms have edge terms that depend on the dihedral angle in a way that no combination of an extrinsic-curvature integral and a Hayward corner term reproduces. For functionals of the causal interval alone, the dependence on the angle at a flat joint is always $$\coth\theta$$ plus a constant, so a half-angle function, and with it the conical $$2\pi$$, is not reachable. Two points are not proved: that a curved boundary in the isolated regime (where the boundary also cuts the intervals) produces a divergence that the family does not remove, which is a derivation sketch with numerical support, and anything about boost-conical hinges.

The action was introduced by Benincasa and Dowker [[BD]](#ref-bd), in all dimensions by Dowker and Glaser [[Dowker–Glaser]](#ref-dg) and Glaser [[Glaser]](#ref-glaser). Aslanbeigi, Saravani and Sorkin [[ASS]](#ref-ass) introduced the family of generalised kernels, and Belenchia [[Belenchia]](#ref-belenchia) showed that its members share the curvature limit. Buck, Dowker, Jubb and Surya [[BDJS]](#ref-bdjs) proposed separate boundary terms for a spacelike boundary whose mean gives the Gibbons–Hawking–York term, and found that the mean action of a causal diamond tends to the volume of its joint. Dowker [[Dowker]](#ref-dowker) and Machet and Wang [[Machet–Wang]](#ref-mw) gave evidence and a computation that, on causally convex regions, the mean is the Einstein–Hilbert action plus the volume of the codimension-two joint. Dowker, Liu and Lloyd-Jones [[DLLJ]](#ref-dllj) defined the embedded and isolated regimes, gave evidence that the mean diverges like the inverse discreteness length on a timelike boundary, and proposed the weight $$\coth\theta$$ for a joint between spacelike boundaries; the covariogram method used below is theirs. This page proves the divergence for static product regions in the embedded regime, proves the corner values for arbitrary convex polygons with edges of any causal type, which show that the weight $$\coth\theta$$ holds at joints between spacelike edges that point the same way in space and that other edge types have different values, determines the finite part on static boundaries, and states what the interval-kernel structure forbids. The non-locality of exact weights, and the smooth limit of the same kernels, are treated in [Exact causal-set horizon weights must be non-local](/quantum-gravity/nonlocal-weights/) and [The causal-set d'Alembertian has negative spectral weight](/quantum-gravity/causal-box-spectrum/); weighted horizon counts, which use the same Mellin symbols, are in [Horizon counts: which weights give the area law](/quantum-gravity/area-projector/).

## 1. The action and its mean value

A *causal set* is a locally finite partially ordered set; a *Poisson sprinkling* of density $$\rho$$ into a region $$R$$ of Minkowski space is a Poisson point process with intensity $$\rho$$ times the volume, ordered by the causal relation $$\prec$$. We write $$\ell=\rho^{-1/4}$$ for the discreteness length and work in four dimensions, with Planck length $$\ell_p$$ set to one in the displayed formulas (the action is then in units of $$\ell_p^{-2}$$). Let $$N$$ be the number of points and, for $$k\ge0$$, $$N_k$$ the number of pairs $$p\prec q$$ with exactly $$k$$ points of the sprinkling strictly between them ($$N_0$$ counts links). A *layer action* is

$$
S=\kappa\,\rho^{-1/2}\Big[b\,N+\sum_{k}c_kN_k\Big],\qquad\kappa=\frac4{\sqrt6}, \tag{1}
$$

with real numbers $$b$$ and $$c_k$$. The Benincasa–Dowker action in four dimensions [[BD]](#ref-bd) has $$b=1$$ and $$(c_0,c_1,c_2,c_3)=(-1,9,-16,8)$$. In the *embedded regime* the points counted between $$p$$ and $$q$$ are all points of a sprinkling of the whole of Minkowski space that lie in the causal interval of $$p$$ and $$q$$, whether or not they lie in $$R$$, and only pairs with both endpoints in $$R$$ are counted. In the *isolated regime* only the points in $$R$$ exist, so the interval is cut by the boundary. For causally convex regions the two coincide. Everything in sections 2 to 6 is in the embedded regime and about the mean of $$S$$ over sprinklings.

By the Mecke equation for the Poisson process the number of points in an interval of volume $$V_{pq}$$ other than the endpoints is Poisson with mean $$\rho V_{pq}$$, so $$\langle N_k\rangle=\rho^2\iint_{p\prec q}(\rho V_{pq})^ke^{-\rho V_{pq}}/k!$$ and

$$
\langle S\rangle=\kappa\rho^{-1/2}\Big[b\,\rho\,{\rm Vol}(R)+\rho^2\iint_{p\prec q;\;p,q\in R}f(\rho V_{pq})\,dV_p\,dV_q\Big],\qquad f(v)=e^{-v}\sum_k\frac{c_k}{k!}v^k. \tag{2}
$$

The Mellin transform of the kernel defines the *Mellin symbol*

$$
\int_0^\infty\lambda^{s-1}f(\lambda)\,d\lambda=\Gamma(s)\,Q(s),\qquad Q(s)=\sum_kc_k\frac{(s)_k}{k!}, \tag{3}
$$

and the weights are recovered from $$c_m=\sum_{j=0}^m(-1)^j\binom mjQ(-j)$$. The Benincasa–Dowker symbol is $$Q_{\rm BD}(s)=\tfrac13(2s-1)(s-1)(2s-3)$$, with $$Q_{\rm BD}'(\tfrac32)=\tfrac23$$, $$Q_{\rm BD}'(1)=-\tfrac13$$ and $$Q_{\rm BD}(\tfrac54)=-\tfrac1{16}$$. The symbol records how the kernel acts on each power of the density: the pair integral in (2) is a sum of terms $$\rho^{2-s}$$ with coefficients determined by the geometry, and the kernel multiplies the term at $$s$$ by $$Q(s)$$ (this is the content of the lemmas below). The points $$s=\tfrac12,1,\tfrac54,\tfrac32$$ turn out to be the exponents at which the geometry of a region with timelike edges produces non-local terms, a volume term, a boundary term and the joint term.

As a first example, take the causal diamond between two points, which is the product of a two-dimensional diamond with a transverse plane. We computed the mean (2) for the Benincasa–Dowker kernel by the procedure described at the end of section 3 and found $$2.000000$$ per unit transverse area at $$\rho=10^4,10^6,10^8$$, in agreement with the joint value $$2$$ found in [[BDJS]](#ref-bdjs).

## 2. Reduction to the covariogram

Let $$R=P\times\mathbb R^2$$, with $$P$$ a convex polygon in the $$(t,x)$$ plane and $$\mathbb R^2$$ the transverse plane; we compute the mean per unit transverse area. Use null coordinates $$u=t-x$$, $$w=t+x$$, so that $$dt\,dx=\tfrac12du\,dw$$. A point $$q$$ is in the causal future of $$p$$ when $$q-p=(a,b,\Delta)$$ with $$a,b\ge0$$ the null components in the plane, $$\Delta$$ the transverse separation and $$ab>\lvert\Delta\rvert^2$$; then the proper time satisfies $$\tau^2=ab-\lvert\Delta\rvert^2$$ and the interval has volume $$V=\pi\tau^4/24$$. Define the overlap function of the polygon in the null components of the separation,

$$
K(a,b)=\int du\,dw\;\mathbf 1_P(z)\mathbf 1_P\big(z+(a,b)\big),
$$

which is twice the area, in $$(t,x)$$ measure, of the overlap of $$P$$ with its translate; it is the covariogram of $$P$$, evaluated at future-directed separations. Write $$H(\mu)=\int_{-\infty}^{\infty}K(\sqrt\mu\,e^s,\sqrt\mu\,e^{-s})\,ds$$, the integral of $$K$$ over the hyperbola $$ab=\mu$$ with respect to $$da/a$$.

<div class="lemma" markdown="1">
**Lemma 1** (reduction). Let $$\sigma=(24/\pi\rho)^{1/2}$$ and $$\Psi(t)=\int_0^tf(y^2)\,dy$$. If $$Q(\tfrac12)=0$$, then $$\Psi$$ decays faster than any power, and the pair integral per unit transverse area is

$$
\rho^2\iint f(\rho V_{pq})\,dV_pdV_q=6\rho\int_0^\infty\Psi(t)\,H(\sigma t)\,dt. \tag{4}
$$

Moreover $$m_\Psi(s):=\int_0^\infty t^{s-1}\Psi(t)\,dt=-\dfrac1{2s}\,\Gamma\big(\tfrac{s+1}2\big)\,Q\big(\tfrac{s+1}2\big)$$.
</div>

<div class="proof" markdown="1">
*Proof.* The measure of a pair is $$\tfrac12du\,dw\,d^2x_\perp$$ for $$p$$ and likewise for $$q$$, so integrating over the position of $$p$$ gives $$\tfrac14\iint da\,db\,K(a,b)\,d^2\Delta$$ per unit transverse area. The transverse integral is exact: with $$Y=ab-\lvert\Delta\rvert^2$$ one has $$d^2\Delta=\pi\,dY$$ and $$\rho V=(Y/\sigma)^2$$, so $$\int d^2\Delta\,f(\rho V)=\pi\sigma\,\Psi(ab/\sigma)$$. Grouping the separations by $$\mu=ab$$ gives $$\iint da\,db\,K\,g(ab)=\int d\mu\,H(\mu)g(\mu)$$ and then $$\rho^2\cdot\tfrac14\pi\sigma\int d\mu\,H(\mu)\Psi(\mu/\sigma)=\rho^2\tfrac14\pi\sigma^2\int dt\,H(\sigma t)\Psi(t)$$, and $$\rho^2\pi\sigma^2/4=6\rho$$. For the decay, $$\Psi(\infty)=\tfrac12\int_0^\infty\lambda^{-1/2}f=\tfrac12\Gamma(\tfrac12)Q(\tfrac12)=0$$ and $$f(y^2)$$ is a polynomial times $$e^{-y^2}$$. The Mellin transform follows by integrating by parts, using $$\Psi(0)=\Psi(\infty)=0$$: $$\int t^{s-1}\Psi=-\frac1s\int t^sf(t^2)dt=-\frac1{2s}\int\lambda^{(s-1)/2}f(\lambda)d\lambda$$.
</div>

The point of (4) is that the four-dimensional problem for a product region is the two-dimensional covariogram integrated along hyperbolas and then smeared with a fixed function. The next lemma turns an expansion of $$H$$ at small argument into an expansion of the action at large density; it uses nothing but the moments $$m_\Psi$$.

<div class="lemma" markdown="1">
**Lemma 2** (moments). Let $$H$$ be bounded on compact subsets of $$(0,\infty)$$, vanish for large argument, and satisfy for all $$\mu>0$$

$$
H(\mu)=\sum_{j}\mu^{\beta_j}\Big(\alpha_j\log\frac1\mu+\gamma_j\Big)+R(\mu),\qquad\lvert R(\mu)\rvert\le C\mu^{\beta_*},
$$

with finitely many exponents $$\beta_j>-1$$ below $$\beta_*$$. Then, as $$\rho\to\infty$$ with $$\sigma=(24/\pi\rho)^{1/2}$$,

$$
6\rho\int_0^\infty\Psi(t)H(\sigma t)\,dt=6\rho\sum_j\sigma^{\beta_j}\Big[\alpha_j\big(-m_\Psi(\beta_j+1)\log\sigma-m_\Psi'(\beta_j+1)\big)+\gamma_jm_\Psi(\beta_j+1)\Big]+O(\rho\,\sigma^{\beta_*}).
$$
</div>

<div class="proof" markdown="1">
*Proof.* Substitute $$\mu=\sigma t$$ in each term: $$\int\Psi(t)(\sigma t)^\beta dt=\sigma^\beta m_\Psi(\beta+1)$$ and $$\int\Psi(t)(\sigma t)^\beta\log\frac1{\sigma t}\,dt=\sigma^\beta\big(-\log\sigma\,m_\Psi(\beta+1)-m_\Psi'(\beta+1)\big)$$. The remainder is at most $$C\sigma^{\beta_*}\int t^{\beta_*}\lvert\Psi\rvert dt$$. The global bound on $$R$$ follows from a bound near zero, because $$H$$ is bounded on $$[\mu_1,\infty)$$ and vanishes for large argument while the sum grows only polynomially.
</div>

In practice each exponent $$\beta_j$$ corresponds to a power of the density: the term $$\mu^{\beta}$$ in $$H$$ contributes at order $$\rho^{(1-\beta)/2}$$ to $$\langle S\rangle$$, with coefficient proportional to $$Q((\beta+2)/2)$$ when $$\alpha_j=0$$ (for $$\beta=0$$ this is $$s=1$$, for $$\beta=\tfrac12$$ it is $$s=\tfrac54$$, for $$\beta=1$$ it is $$s=\tfrac32$$). A term with a logarithm has the derivative $$Q'$$ in place of $$Q$$ where $$Q$$ vanishes.

## 3. The covariogram of a polygon

We need the small-argument expansion of $$H$$ for every convex polygon $$P$$, with edges that may be spacelike, null or timelike. Two facts about $$K$$ are used. Write the plane with coordinates $$(t,x)$$ and let $$\det(p,q)=p_tq_x-p_xq_t$$. For each edge $$e$$ of $$P$$ let $$N_e$$ be the outward normal of length equal to the length of $$e$$, so that $$N_e\cdot h$$ is a determinant with the edge vector. At a vertex $$v$$ let $$a_v,b_v$$ be the edge vectors pointing away from $$v$$ along the two edges that meet there, in $$(t,x)$$ components.

<div class="proposition" markdown="1">
**Proposition 1** (covariogram near the origin). Let $$C(h)=\lvert P\cap(P-h)\rvert$$ be the area in $$(t,x)$$ measure, so $$K=2C$$. There is $$r_P>0$$ such that for $$\lvert h\rvert<r_P$$

$$
C(h)=V_P-\sum_e(N_e\cdot h)_-+\sum_v\frac{\big[\det(h,a_v)\det(h,b_v)\big]_+}{2\,\lvert\det(a_v,b_v)\rvert}, \tag{5}
$$

where $$(y)_\pm=\max(0,\pm y)$$ and $$V_P$$ is the area of $$P$$.
</div>

The picture is that translating a convex polygon by a short vector $$h$$ removes a strip along each edge facing $$-h$$, which costs area linear in $$h$$, and that the strips cut off at the two vertices where a line parallel to $$h$$ touches the polygon overlap, which gives a quadratic correction.

<div class="proof" markdown="1">
*Proof.* Take $$h=su$$ with $$u$$ a unit vector and $$s>0$$. Let $$L_u(y)$$ be the length of the chord of $$P$$ along $$u$$ at transverse coordinate $$y$$ and $$\Pi_u$$ the projection of $$P$$ along $$u$$, of length $$\lvert\Pi_u\rvert$$. Then $$C(su)=\int(L_u(y)-s)_+dy=V_P-s\lvert\Pi_u\rvert+\int(s-L_u(y))_+dy$$, the last integral over $$\Pi_u$$. For $$s$$ small the chords shorter than $$s$$ occur only near the two extreme vertices of $$\Pi_u$$. At such a vertex with edge vectors $$a,b$$ pointing away from it, at transverse distance $$y$$ from it the chord has length $$y\,\lvert\det(a,b)\rvert/(\det(u,a)\det(u,b))$$ (in units of the length of $$u$$), so the last integral equals $$s^2\det(u,a)\det(u,b)/(2\lvert\det(a,b)\rvert)$$, and these two vertices are exactly those at which $$\det(u,a)$$ and $$\det(u,b)$$ have the same sign. Finally $$s\lvert\Pi_u\rvert=\sum_e(N_e\cdot h)_-$$, since $$\sum_eN_e=0$$ gives $$\sum_e(N_e\cdot h)_+=\sum_e(N_e\cdot h)_-$$ and the left side is the projected extent of the boundary.
</div>

We checked (5) against an independent computation of $$\lvert P\cap(P-h)\rvert$$ by polygon clipping, for $$300$$ random convex polygons with rational vertices (edges of all causal types) at $$5$$ random displacements of size $$10^{-3}$$ each: the largest difference is $$7\times10^{-15}$$.

Equation (5) holds for any separation $$h$$ in the plane. Inside the future quadrant $$a,b>0$$ the function $$C$$ is quadratic except along rays through the origin in the directions of the timelike edges of $$P$$, where the term $$(N_e\cdot h)_-$$ has a kink. A direction $$h$$ with $$h_t>\lvert h_x\rvert$$ is timelike, so these rays are the directions of timelike edges.

<div class="proposition" markdown="1">
**Proposition 2** (expansion of $$H$$). Let $$P$$ be a convex polygon with area $$V_P$$ and let $$L_P$$ be the total proper length of its timelike edges. Let $$m_u$$ and $$m_w$$ be the values of $$\partial_u\partial_wC$$ on the sectors of the future quadrant adjacent to the $$u$$-axis and to the $$w$$-axis, respectively. There are $$\sigma_0>0$$ and constants $$h_0,h_1$$ such that for $$0<\mu<\sigma_0$$

$$
H(\mu)=2V_P\log\frac1\mu+h_0+2L_P\,\mu^{1/2}+(m_u+m_w)\,\mu\log\frac1\mu+h_1\mu+O(\mu^2),
$$

with a remainder that is an analytic function of $$\mu$$ times $$\mu^2$$ and carries no logarithm and no half-integer power. In particular there is no term $$\mu^{3/2}$$ and no term $$\mu^2\log\mu$$.
</div>

The idea is that $$K$$ is piecewise quadratic, the hyperbola crosses its cells at points whose positions are analytic in $$\sqrt\mu$$ or in $$1/\sqrt\mu$$ except at the kink rays, and integration over a cell of a quadratic in $$(a,b)$$ along the hyperbola gives powers of $$\mu$$ at the end points and logarithms only through the constant and cross terms.

<div class="proof" markdown="1">
*Proof.* (1) *Cells.* For a convex polygon the intersection $$P\cap(P-h)$$ is a convex polygon whose vertices are vertices of $$P$$ lying in $$P-h$$, vertices of $$P-h$$ lying in $$P$$, and intersections of an edge of $$P$$ with an edge of $$P-h$$. In a region of $$h$$ where the pattern of which vertices lie in which half-planes is fixed, these vertices are affine functions of $$h$$ and the area is a quadratic polynomial. The pattern changes only when a vertex of one polygon lies on the line through an edge of the other, that is, on one of the finitely many lines $$N_e\cdot h=\pm N_e\cdot(v_j-v_k)$$. So $$K$$ is continuous and piecewise quadratic on the cells of a finite line arrangement, and vanishes outside a bounded set containing the origin in its interior.

(2) *Breakpoints.* Parametrise the hyperbola by $$a=\sqrt\mu\,y$$, $$b=\sqrt\mu/y$$, $$y=e^s>0$$. The line $$n_1a+n_2b=c$$ meets it where $$n_1\sqrt\mu\,y^2-c\,y+n_2\sqrt\mu=0$$. For $$c\neq0$$ the roots are $$y=[c\pm(c^2-4n_1n_2\mu)^{1/2}]/(2n_1\sqrt\mu)$$ (or one root if $$n_1=0$$ or $$n_2=0$$): a large root $$y=\mu^{-1/2}g(\mu)$$ and a small root $$y=\mu^{1/2}\tilde g(\mu)$$ with $$g,\tilde g$$ analytic and nonzero at $$\mu=0$$. For $$c=0$$ the root is $$y_0=(-n_2/n_1)^{1/2}$$, independent of $$\mu$$, and exists when $$n_1n_2<0$$, which means that the line is the ray along a timelike edge. The order in which the hyperbola meets the finitely many lines changes only at values of $$\mu$$ where two crossing points coincide or two roots merge; each such condition is a nontrivial algebraic equation in $$\mu$$ and has finitely many positive roots, so the ordering is fixed below some $$\sigma_0>0$$.

(3) *Integration.* On a cell, $$K=c_{00}+c_{10}a+c_{01}b+c_{20}a^2+c_{11}ab+c_{02}b^2$$, and along the hyperbola $$ab=\mu$$, so
$$
\int K\frac{dy}{y}=(c_{00}+c_{11}\mu)\log y+\sqrt\mu\,(c_{10}y-c_{01}/y)+\tfrac12\mu\,(c_{20}y^2-c_{02}/y^2).
$$
At a large root every term other than the logarithm is an analytic function of $$\mu$$ (for example $$\sqrt\mu\,y=g(\mu)$$ and $$\mu y^2=g^2$$), and $$\log y=-\tfrac12\log\mu+\log g$$. At a small root likewise, with $$\log y=+\tfrac12\log\mu+\log\tilde g$$. At a kink root $$y_0$$ the terms $$\sqrt\mu\,c_{10}y_0$$ and $$-\sqrt\mu\,c_{01}/y_0$$ carry $$\mu^{1/2}$$ and the others are analytic or constant. Hence $$H(\mu)$$ is the sum of an analytic function of $$\mu$$, a term in $$\mu^{1/2}$$ with no higher half-integer power, and logarithms of the form $$(A+B\mu)\log\mu$$.

(4) *Logarithms.* Because $$K$$ is continuous, the sum of the logarithmic terms over the large roots telescopes: it equals $$-\tfrac12\log\mu$$ times the value of $$c_{00}+c_{11}\mu$$ on the innermost cell on that side (the cell adjacent to the origin) minus its value on the outermost cell, where $$K\equiv0$$. The same holds for the small roots. On the innermost cells $$c_{00}=K(0)=2V_P$$ and $$c_{11}=2m_u$$ or $$2m_w$$, by (5), since on the sector adjacent to an axis $$K$$ is the quadratic polynomial from (5). Adding the two sides gives $$-2V_P\log\mu-(m_u+m_w)\mu\log\mu$$.

(5) *The term in $$\mu^{1/2}$$.* It comes from the kink roots, one for each timelike edge. By Proposition 1 the part of $$K$$ that is linear in $$h$$ near the origin is $$-2\sum_e(N_e\cdot h)_-=-\sum_e\lvert N_e\cdot h\rvert$$ (using $$\sum_eN_e=0$$). For a timelike edge $$e$$ write $$N_e\cdot h=N_1a+N_2b$$ in null components, so that $$N_1=(N_t-N_x)/2$$ and $$N_2=(N_t+N_x)/2$$; for a timelike edge $$N_1N_2<0$$ and $$4\lvert N_1N_2\rvert=\lvert N_t^2-N_x^2\rvert=\tau_e^2$$, the square of the proper length of $$e$$. The kink is at $$y_0^2=\lvert N_2/N_1\rvert$$. For $$y<y_0$$ the term $$N_2b$$ dominates $$N_e\cdot h$$ and for $$y>y_0$$ the term $$N_1a$$ does, so the gradient of $$-\lvert N_e\cdot h\rvert$$ is $$(-\mathrm{sgn}(N_2)N_1,-\lvert N_2\rvert)$$ below the kink and $$(\mathrm{sgn}(N_2)N_1,\lvert N_2\rvert)$$ above it, a jump of $$\Delta=(-2\lvert N_1\rvert,2\lvert N_2\rvert)$$ in $$(c_{10},c_{01})$$. The contribution of the kink to $$H$$ is the difference of the antiderivative $$\sqrt\mu\,(c_{10}y-c_{01}/y)$$ across it, $$\sqrt\mu\,[-\Delta_1y_0+\Delta_2/y_0]=\sqrt\mu\,[2\lvert N_1\rvert y_0+2\lvert N_2\rvert/y_0]=4\sqrt{\lvert N_1N_2\rvert}\,\sqrt\mu=2\tau_e\sqrt\mu$$, since $$\lvert N_2\rvert/y_0=\lvert N_1\rvert y_0$$. Summing over the timelike edges gives the coefficient $$2L_P$$. The remaining terms are analytic because the series in (3) converge for $$\mu<\sigma_0$$.
</div>

The proof of step (5) was checked by evaluating $$H$$ numerically (with the exact overlap function, quadrature on each cell, and a least-squares fit of the structure above) for the triangle with vertices $$(t,x)=(0,-1),(-0.3,1),(0.5,1.1)$$, which has one timelike edge of proper length $$\sqrt{0.63}$$: the fitted coefficients of $$\mu^{1/2}$$ and $$\mu\log\mu$$ are $$1.587451$$ and $$-1.237730$$, against the predictions $$2L_P=1.587451$$ and $$-(m_u+m_w)=-807/652=-1.237730$$; the fitted coefficient of $$\mu^2$$ is $$-0.23521$$ and no term $$\mu^{3/2}$$ is needed.

The numerical values quoted below for the exact mean (2) were obtained as follows. For a given polygon, $$K(h)$$ is computed by clipping $$P$$ against $$P-h$$. For each $$\mu$$ the breakpoints of the hyperbola on the lines of step (1) are found from the quadratic of step (2); on each interval between breakpoints $$K$$ restricted to the hyperbola is a combination of $$1,e^{\pm s},e^{\pm2s}$$ whose five coefficients are fixed by five evaluations of $$K$$ and integrated exactly. The outer integral (4) is then done by Gauss–Legendre quadrature on geometric panels in $$t$$. For the triangle the integral agrees with the exact expansion of Proposition 2 (see the fit above) and with the value $$807/326$$ in the limit.

## 4. The action on a product region

<div class="theorem" markdown="1">
**Theorem 1** (expansion of the mean action). Let $$P$$ be a convex polygon with timelike edges of total proper length $$L_P$$, and consider the layer action (1) on $$R=P\times\mathbb R^2$$ in the embedded regime, per unit transverse area. Suppose the kernel satisfies

(K1) $$Q(\tfrac12)=0$$, (K2) $$Q(1)=0$$ and $$b=-3Q'(1)$$, (K3) $$Q(\tfrac32)=0$$.

Then as $$\rho\to\infty$$

$$
\langle S\rangle=-16\,a_*\,Q(\tfrac54)\,L_P\,\rho^{1/4}+3\,Q'(\tfrac32)\,(m_u+m_w)+O(\rho^{-1/2}),\qquad a_*=\frac{\Gamma(\frac54)}{\sqrt6}\Big(\frac{24}{\pi}\Big)^{1/4}=0.615192\ldots \tag{6}
$$

If (K1) holds but (K2) or (K3) fails there are in addition a term of order $$\rho^{1/2}$$ proportional to $$V_P$$ (when $$Q(1)=0$$ but $$b\neq-3Q'(1)$$), a term of order $$\rho^{1/2}\log\rho$$ (when $$Q(1)\neq0$$), and a term of order $$\log\rho$$ (when $$Q(\tfrac32)\neq0$$; section 7).
</div>

*Proof.* By (2) and Lemma 1, $$\langle S\rangle=\kappa\rho^{-1/2}[b\rho V_P+6\rho\int\Psi(t)H(\sigma t)dt]$$. Insert the expansion of Proposition 2 and use Lemma 2. The term $$2V_P\log(1/\mu)$$ gives $$-6\rho\cdot2V_Pm_\Psi'(1)$$, and (K1)-(K2) give $$m_\Psi(1)=-\tfrac12\Gamma(1)Q(1)=0$$ and $$m_\Psi'(1)=-\tfrac14Q'(1)$$, so the contribution is $$3\rho V_PQ'(1)$$; with the element term $$b\rho V_P$$ the sum is $$\rho V_P(b+3Q'(1))=0$$. The constant $$h_0$$ meets $$m_\Psi(1)=0$$ and drops out. The term $$2L_P\mu^{1/2}$$ gives $$6\rho\cdot2L_P\sigma^{1/2}m_\Psi(\tfrac32)$$ with $$m_\Psi(\tfrac32)=-\tfrac13\Gamma(\tfrac54)Q(\tfrac54)$$, and $$\kappa\rho^{-1/2}\rho\sigma^{1/2}=\kappa\rho^{1/4}(24/\pi)^{1/4}$$ gives the first term of (6). The term $$(m_u+m_w)\mu\log(1/\mu)$$ gives $$6\rho\sigma(m_u+m_w)[-m_\Psi(2)\log\sigma-m_\Psi'(2)]$$; by (K3) $$m_\Psi(2)=-\tfrac14\Gamma(\tfrac32)Q(\tfrac32)=0$$ and $$m_\Psi'(2)=-\tfrac18\Gamma(\tfrac32)Q'(\tfrac32)$$, so the contribution is $$\kappa\rho^{-1/2}\cdot6\rho\sigma\cdot\tfrac18\Gamma(\tfrac32)Q'(\tfrac32)(m_u+m_w)$$; with $$\rho\sigma=(24\rho/\pi)^{1/2}$$, $$\kappa=4/\sqrt6$$ and $$\Gamma(\tfrac32)=\sqrt\pi/2$$ the numerical factor is $$\tfrac4{\sqrt6}\cdot6\sqrt{24}\cdot\tfrac1{16}=3$$. The constant $$h_1$$ meets $$m_\Psi(2)=0$$. The remainder $$O(\mu^2)$$ gives $$O(\rho\sigma^2)\kappa\rho^{-1/2}=O(\rho^{-1/2})$$. If (K2) or (K3) fails, the same computation leaves the terms listed.

The steps with content are the structure of $$H$$ (Proposition 2); the rest is bookkeeping of moments. As checks of (6): for the causal diamond ($$L_P=0$$, $$m_u+m_w=1$$, see section 5) the exact quadrature of (2) gives $$2.000000$$ at $$\rho=10^4,10^6,10^8$$. For the lens-shaped region with vertices $$(t,x)=(0,-1),(0.25,0.1),(0,1),(-0.35,0.2)$$, whose edges are spacelike and whose joint angles are $$0.5317$$ and $$0.7544$$, the Benincasa–Dowker mean is $$3.63097$$ and $$3.62353$$ at $$\rho=10^6$$ and $$10^8$$, against the limit $$\coth0.5317+\coth0.7544=3.62271$$, with a difference decaying like $$\rho^{-1/2}$$ (the difference times $$\rho^{1/2}$$ is $$8.26$$ and $$8.22$$).

### Plain action on a timelike edge

For the Benincasa–Dowker kernel $$b=1$$, $$Q'_{\rm BD}(1)=-\tfrac13$$ and $$Q_{\rm BD}(\tfrac12)=Q_{\rm BD}(1)=Q_{\rm BD}(\tfrac32)=0$$, so (K1)-(K3) hold and (6) applies with $$Q_{\rm BD}(\tfrac54)=-\tfrac1{16}$$.

<div class="corollary" markdown="1">
**Corollary 1** (divergence at a timelike boundary). For a convex polygon $$P$$ with at least one timelike edge, the mean Benincasa–Dowker action per unit transverse area is

$$
\langle S\rangle=a_*\,L_P\,\rho^{1/4}+3Q'_{\rm BD}(\tfrac32)(m_u+m_w)+O(\rho^{-1/2}),
$$

so it diverges like $$\rho^{1/4}=\ell^{-1}$$ with a coefficient equal to $$a_*=0.615192$$ times the volume $$L_P$$ of the timelike boundary per unit transverse area.
</div>

For the triangle above, the quadrature gives $$51.305$$ at $$\rho=10^8$$, and $$a_*L_P\rho^{1/4}+2.4754601=0.488294\cdot100+2.47546=51.305$$ with $$L_P=\sqrt{0.63}$$. The coefficient $$a_*$$ agrees with the coefficient found numerically for slabs in [[DLLJ]](#ref-dllj).

### The family of finite kernels

The divergence is removed by the zero of $$Q$$ at $$\tfrac54$$. The remaining condition is a normalisation. Theorem 1 shows that the corner values are proportional to $$Q'(\tfrac32)$$, and we fix $$Q'(\tfrac32)=\tfrac23$$, the Benincasa–Dowker value, so that the joint of the diamond is $$2$$. The coefficient of the Einstein–Hilbert term of the same members is also proportional to $$Q'(\tfrac32)$$ according to [[ASS]](#ref-ass) and [[Belenchia]](#ref-belenchia); we have not rederived that statement here.

<div class="proposition" markdown="1">
**Proposition 3** (the family). Let $$L\ge3$$. The layer weights of degree at most $$L$$ with $$Q(\tfrac12)=Q(1)=Q(\tfrac54)=Q(\tfrac32)=0$$ and $$Q'(\tfrac32)=\tfrac23$$ are the polynomials
$$
Q=Q_\star+(s-\tfrac12)(s-1)(s-\tfrac54)(s-\tfrac32)^2\,r(s),\qquad Q_\star(s)=\tfrac{16}3(s-\tfrac12)(s-1)(s-\tfrac54)(s-\tfrac32),
$$
with $$\deg r\le L-5$$. There is none for $$L=3$$, exactly one for $$L=4$$, and an affine family of dimension $$L-4$$ for $$L\ge4$$. The element-count coefficient $$b=-3Q'(1)$$ of the minimal member is $$-1$$, and $$b$$ changes with $$r$$.
</div>

<div class="proof" markdown="1">
*Proof.* The five conditions are linear. On polynomials of degree at most $$4$$ the functionals evaluating at $$\tfrac12,1,\tfrac54$$ and evaluating the value and derivative at $$\tfrac32$$ are independent (Hermite interpolation), so $$Q_\star$$ is the unique solution of degree $$4$$ and the solutions of degree $$\le L$$ form an affine space of dimension $$L+1-5$$. A cubic with four zeros vanishes. The difference of two solutions has simple zeros at $$\tfrac12,1,\tfrac54$$ and a double zero at $$\tfrac32$$, hence is divisible by the quintic. We checked the dimensions for $$L=3,\dots,7$$ with exact linear algebra. For $$Q_\star$$, $$Q'_\star(1)=\tfrac13$$ so $$b=-1$$.
</div>

In the weights, $$Q_\star$$ corresponds to $$(c_0,\dots,c_4)=(5,-85,280,-328,128)$$ with $$b=-1$$ (to be compared with $$b=+1$$ and $$(-1,9,-16,8)$$ for Benincasa–Dowker); the sign of $$b$$ is opposite because $$Q'(1)$$ and $$Q'(\tfrac32)$$ have the same sign for the family and opposite signs for $$Q_{\rm BD}$$. The condition $$b=-3Q'(1)$$ is part of the requirement of finiteness: with the Benincasa–Dowker value $$b=1$$ the minimal member would diverge as $$\rho^{1/2}V_P$$. The minimal member is minimal in degree and is not otherwise distinguished. The $$O(\rho^{-1/2})$$ term of Theorem 1 is proportional to $$Q(2)$$ times the coefficient of $$\mu^2$$ in $$H$$, so it differs between members. For $$Q_\star$$ and the triangle above, the quadrature gives $$2.57$$, $$2.484$$, $$2.4763$$ at $$\rho=10^4,10^6,10^8$$, converging to $$807/326=2.4754601$$ like $$\rho^{-1/2}$$.

## 5. Corner values

By Theorem 1 the finite part of the mean is $$\tfrac32Q'(\tfrac32)\sum_vj_v$$, where $$j_v=2(\mu_v^{(u)}+\mu_v^{(w)})$$ is the sum over vertices $$v$$ of the contributions to the Hessians $$m_u,m_w$$ in (5). We read these off from the vertex terms of (5).

<div class="theorem" markdown="1">
**Theorem 2** (corner values). In the setting of Theorem 1 let $$a,b$$ be the edge vectors pointing away from a vertex $$v$$, $$\eta(a,b)=a_tb_t-a_xb_x$$ and $$d(a,b)=a_tb_x-a_xb_t$$. The finite part of the mean is $$\tfrac32Q'(\tfrac32)\sum_vj_v$$ with $$j_v=-\tfrac12\,n_v\,\eta(a,b)/\lvert d(a,b)\rvert$$, where a spacelike edge has rapidity $$\alpha$$ if it is parallel to $$(\sinh\alpha,\cosh\alpha)$$ and a timelike edge if it is parallel to $$(\cosh\alpha,\sinh\alpha)$$, and $$n_v\in\{0,1,2\}$$ is the number of the two sectors adjacent to the null axes in which $$\det(h,a)\det(h,b)>0$$. For the sign of $$\det(h,a)$$ on a sector one uses the rule: for $$a$$ spacelike or null it is $$\mathrm{sgn}\,a_x$$ on both sectors; for $$a$$ timelike it is $$-\mathrm{sgn}\,a_t$$ on the sector adjacent to the future null direction $$(1,1)$$ and $$+\mathrm{sgn}\,a_t$$ on the sector adjacent to $$(1,-1)$$. Explicitly:

| edges at the vertex | $$j_v$$ |
|---|---|
| spacelike, spacelike, same direction in space | $$\coth\theta$$, $$\theta$$ the difference of the rapidities of the two edges |
| spacelike, spacelike, opposite directions in space | $$0$$ |
| timelike, spacelike | $$-\tfrac12\eta/\lvert d\rvert$$, of absolute value $$\tfrac12\tanh$$ of the boost between the timelike edge and the normal of the spacelike edge |
| timelike, timelike, same time direction | $$-\coth\chi$$, $$\chi$$ the difference of rapidities |
| timelike, timelike, opposite time directions | $$0$$ |
| null, spacelike, same direction in space | $$+1$$ |
| null, spacelike, otherwise | $$0$$ |
| null, timelike, same time direction | $$-\tfrac12$$ |
| null, timelike, opposite time directions | $$+\tfrac12$$ |
| null, null, same direction in space | $$+1$$ |
| null, null, otherwise | $$0$$ |
</div>

<div class="proof" markdown="1">
*Proof.* By (5), $$m_u$$ and $$m_w$$ are sums over vertices of the mixed derivative $$\partial_u\partial_w$$ of $$[\det(h,a)\det(h,b)]_+/(2\lvert d\rvert)$$ on the sector in question, where $$\det(h,a)=\tfrac12(h_ua_w-h_wa_u)$$ in null components. On a sector where the product is positive it is a quadratic form with $$\partial_u\partial_w[\det(h,a)\det(h,b)]=-\tfrac14(a_wb_u+a_ub_w)=-\tfrac12\eta(a,b)$$, so the vertex contributes $$-\eta(a,b)/(4\lvert d(a,b)\rvert)$$; where the product is not positive it contributes $$0$$. Hence $$m_u+m_w=\sum_vn_v\,(-\eta/4\lvert d\rvert)$$ and $$3Q'(\tfrac32)(m_u+m_w)=\tfrac32Q'(\tfrac32)\sum_vj_v$$ with $$j_v=-\tfrac12n_v\eta/\lvert d\rvert$$. The signs on the sectors are those of $$\det(h,a)$$ at $$h=n_\pm+\delta e_t$$ with $$n_\pm=(1,\pm1)$$ and $$\delta\to0^+$$: for $$a=s(\sinh\alpha,\cosh\alpha)$$ (spacelike) one finds $$\mathrm{sgn}\,\det=\mathrm{sgn}\,a_x$$ on both sectors, for a null edge the same by direct computation, and for $$a=s(\cosh\alpha,\sinh\alpha)$$ (timelike) $$-s$$ on $$n_+$$ and $$+s$$ on $$n_-$$. Now classify. Spacelike pair: $$n_v=2[a_xb_x>0]$$, and with $$a=s_a(\sinh\alpha,\cosh\alpha)$$, $$b=s_b(\sinh\beta,\cosh\beta)$$ one has $$\eta=-s_as_b\cosh(\alpha-\beta)$$ and $$\lvert d\rvert=\lvert\sinh(\alpha-\beta)\rvert$$, so $$j_v=\coth\lvert\alpha-\beta\rvert$$ when $$s_as_b>0$$. Timelike pair $$a=s_a(\cosh\alpha,\sinh\alpha)$$, $$b=s_b(\cosh\beta,\sinh\beta)$$: $$n_v=2[s_as_b>0]$$, $$\eta=s_as_b\cosh(\alpha-\beta)$$, $$j_v=-\coth\lvert\alpha-\beta\rvert$$. Timelike $$a$$ with spacelike $$b$$: the products of signs on the two sectors are $$-s_as_b$$ and $$+s_as_b$$, so $$n_v=1$$; $$\eta=s_as_b\sinh(\beta-\alpha)$$ and $$\lvert d\rvert=\cosh(\beta-\alpha)$$, giving $$j_v=-\tfrac12s_as_b\tanh(\beta-\alpha)$$, which vanishes when the edges are Lorentz-orthogonal. For a null edge $$a=s_a(1,\pm1)$$ the same computations with $$b$$ spacelike, timelike or null give $$\eta/\lvert d\rvert=-s_as_b$$ (spacelike $$b$$ with $$a_xb_x>0$$), $$s_as_b$$ (timelike $$b$$) and $$-1$$ (null with $$a_xb_x>0$$), which yields the last six rows.
</div>

The diamond has two vertices joining two null edges that point the same way in space (value $$1$$ each) and two that point opposite ways (value $$0$$), so $$\sum_vj_v=2$$. We checked the whole table, including the null rows, against the finite part computed directly from (5) with exact rational arithmetic (mixed second differences of (5) on the two sectors) for $$400$$ random convex polygons with integer vertices, which realise all of the classes; there was no discrepancy. For the triangle above, the vertices are one spacelike–spacelike joint, with value $$\coth\theta=2.66871$$, and two timelike–spacelike joints with values $$-0.05828$$ and $$-0.13497$$, and the sum is $$807/326=2.47546$$.

Three consequences are worth separating. First, the corner values are universal: apart from the factor $$Q'(\tfrac32)$$, which the normalisation fixes, they do not depend on the kernel. Second, the joint between two spacelike edges is $$\coth\theta$$ only when the two edges point the same way in space, that is when the vertex is a leftmost or rightmost vertex of the region, where the past and the future boundary meet. A kink inside the past or future boundary, where the two edges point in opposite directions in space, has value $$0$$, so the rule "$$\coth\theta$$ at every spacelike–spacelike corner" is false. Third, ending a timelike edge on the null cone does not commute with $$\rho\to\infty$$: as the edge direction of a timelike–timelike vertex tends to null, $$-\coth\chi\to-1$$, while the null–timelike value is $$-\tfrac12$$, and a spacelike edge tilting to null changes nothing; at finite $$\rho$$ an edge counts as null until its causal interval contains about one point. This statement is about the exact limits of the closed forms. Its uniformity at finite density was checked for one null polygon only and is not proved.

## 6. Static timelike boundaries and the absence of a Gibbons–Hawking–York term

The variational principle of general relativity with a timelike boundary needs a boundary term $$\gamma\oint K\sqrt{h}$$ with $$\gamma=1$$ in the normalisation in which the bulk term is $$\tfrac12\int R\sqrt{-g}$$ (the normalisation of the Benincasa–Dowker action, whose mean tends to this bulk term), with $$K$$ the trace of the extrinsic curvature of the boundary. Does the mean action contain such a term? For the regions of the previous sections the boundary is flat, so $$K=0$$; a test needs a curved boundary. We take the static product regions $$R=[0,T]\times S$$, with $$S\subset\mathbb R^3$$ a convex body. Let $$g_S(x)=\lvert S\cap(S-x)\rvert$$ be the covariogram of $$S$$ and $$\bar g(r)$$ its average over directions, expanded as $$\bar g(r)=a_0+a_1r+a_2r^2+a_3r^3+\dots$$ near $$r=0$$.

<div class="theorem" markdown="1">
**Theorem 3** (static product regions). Let $$S$$ be a convex body and suppose that $$\bar g(r)=a_0+a_1r+a_2r^2+O(r^3)$$ as $$r\to0$$, that $$\bar g$$ is smooth away from finitely many points, and that $$T$$ is not one of them. For the layer action in the embedded regime with $$Q(\tfrac12)=Q(1)=Q(\tfrac32)=0$$ and $$b=-3Q'(1)$$,

$$
\langle S\rangle=-16\,a_*\,Q(\tfrac54)\,{\rm Vol}_3\,\rho^{1/4}-9\,Q'(\tfrac32)\,a_2\,T+O(\rho^{-1/4}), \tag{7}
$$

where $${\rm Vol}_3=T\,{\rm Area}(\partial S)$$ is the three-volume of the timelike boundary and $$a_*$$ is as in (6). If moreover $$\bar g(r)=a_0+a_1r+a_2r^2+a_3r^3+O(r^4)$$ then the next term is $$-\tfrac{8\pi}{15}\kappa\,(24/\pi)^{7/4}\Gamma(\tfrac74)\,Q(\tfrac74)\,a_3\,T\,\rho^{-1/4}$$.
</div>

The idea is that, as in section 3, the action depends on the covariogram of the region near zero separation; here the region factorises in time and space, and the only non-analytic dependence on the proper-time variable comes from the small-distance expansion of $$\bar g$$.

<div class="proof" markdown="1">
*Proof.* The covariogram of $$R$$ at the separation $$(\Delta t,\Delta x)$$ is $$(T-\Delta t)_+\,g_S(\Delta x)$$. Writing $$w=\tau^2=\Delta t^2-r^2$$ with $$r=\lvert\Delta x\rvert$$, the pair integral in (2) is $$\rho^2\int dw\,f(\rho c_Vw^2)G(w)$$ with $$c_V=\pi/24$$ and
$$
G(w)=2\pi\int_0^{r_m(w)}r^2\bar g(r)\Big[\frac T{\sqrt{r^2+w}}-1\Big]dr,\qquad r_m=\min\big(r_{\max},\sqrt{T^2-w}\big),
$$
since $$d\Delta t\,\delta(\Delta t^2-r^2-w)=1/(2\Delta t)$$ and $$d^3\Delta x=4\pi r^2dr$$ after averaging over directions. The non-analytic dependence of $$G$$ on $$w$$ comes from small $$r$$; the region $$r\ge r_1$$ and the endpoint $$r_m$$ give analytic functions of $$w$$ near $$0$$. For the terms $$a_nr^n$$ of $$\bar g$$ the integrals $$\int_0^{r_1}r^{n+2}(r^2+w)^{-1/2}dr$$ have singular parts $$\tfrac w4\ln w$$, $$\tfrac23w^{3/2}$$, $$-\tfrac3{16}w^2\ln w$$, $$-\tfrac8{15}w^{5/2}$$ for $$n=0,1,2,3$$ (from the antiderivatives, e.g. $$\int r^3(r^2+w)^{-1/2}=\tfrac13(r^2-2w)\sqrt{r^2+w}$$). For the remainder $$E(r)=O(r^3)$$ one expands $$(1+x)^{-1/2}=1-\tfrac x2+\tfrac38x^2+O(x^3)$$ with $$x=w/r^2$$ for $$r>\sqrt w$$ and bounds the region $$r<\sqrt w$$ directly; this gives an analytic function with terms up to $$w^2$$ plus $$O(w^{5/2})$$, with no logarithm. So
$$
G(w)=\text{analytic}+2\pi T\Big[\tfrac14a_0\,w\ln w+\tfrac23a_1\,w^{3/2}-\tfrac3{16}a_2\,w^2\ln w\Big]+O(w^{5/2}).
$$
The Mellin integral of $$w^\beta\ln^mw$$ against $$f(\rho c_Vw^2)$$ is $$\tfrac12(\rho c_V)^{-s}\Gamma(s)Q(s)$$ with $$s=(\beta+1)/2$$, and with a logarithm the derivative in $$\beta$$. The analytic terms $$w^0,w^1,w^2$$ sit at $$s=\tfrac12,1,\tfrac32$$ and vanish. The $$w\ln w$$ term gives $$3\rho\,{\rm Vol}_4Q'(1)$$ with $${\rm Vol}_4=Ta_0$$, cancelled by the element term. The $$w^{3/2}$$ term gives $$\tfrac{2\pi}3a_1T\rho^{3/4}c_V^{-5/4}\Gamma(\tfrac54)Q(\tfrac54)$$; for a convex body $$a_1=-{\rm Area}(\partial S)/4$$ (Cauchy's projection formula: the mean area of the shadow of a convex body is a quarter of its surface area, and $$C(su)=V-s\lvert\Pi_u\rvert+O(s^2)$$ by the chord identity of Proposition 1), which with $$\kappa\rho^{-1/2}$$ gives the first term of (7). The $$w^2\ln w$$ term gives $$-\tfrac{3\pi}{32}Ta_2\rho^{1/2}c_V^{-3/2}\Gamma(\tfrac32)Q'(\tfrac32)$$ (because $$Q(\tfrac32)=0$$), and $$\kappa\rho^{-1/2}$$ times this equals $$-9Q'(\tfrac32)a_2T$$. The $$w^{5/2}$$ term is $$O(\rho^{-1/4})$$, with the explicit coefficient when the expansion of $$\bar g$$ extends to $$r^3$$.
</div>

We tested the theorem for the ball $$S=B_R$$ with $$R=1$$ and $$T=1$$, whose covariogram is the lens volume $$\tfrac{4\pi}{3}R^3-\pi R^2r+\tfrac\pi{12}r^3$$ for $$r<2R$$ (so $$a_2=0$$, $$a_3=\pi/12$$). We evaluated the exact pair integral (2) numerically, with $$G(w)$$ computed from the full lens formula. For the Benincasa–Dowker kernel the mean minus the divergent term of (7) is $$-0.22187$$, $$-0.07157$$ and $$-0.02277$$ at $$\rho=10^6,10^8,10^{10}$$, against the predicted $$\rho^{-1/4}$$ term $$-0.22838$$, $$-0.07222$$ and $$-0.02284$$; the differences, $$0.0065$$, $$0.00065$$ and $$0.000066$$, fall by a factor ten for each factor $$100$$ in $$\rho$$, as an $$O(\rho^{-1/2})$$ remainder does, so the finite part is zero. For the kernel $$Q_\star$$ of Proposition 3 (for which $$Q(\tfrac54)=0$$ and the divergent term is absent) the mean is $$-0.4373$$, $$-0.1425$$ and $$-0.04548$$, against $$-0.4568$$, $$-0.1444$$ and $$-0.04568$$ from the $$\rho^{-1/4}$$ term, with the same $$\rho^{-1/2}$$ pattern of differences.

### When $$a_2$$ vanishes

The finite part is proportional to $$a_2$$. For which bodies is it zero?

<div class="proposition" markdown="1">
**Proposition 4** ($$C^{1,1}$$ bodies). If $$S\subset\mathbb R^3$$ is a convex body whose boundary is $$C^{1,1}$$ (a ball of some radius $$r_c>0$$ rolls freely inside $$S$$ at every boundary point), then $$\bar g(r)=a_0+a_1r+O(r^3)$$, so $$a_2=0$$ and the finite part in (7) vanishes for every kernel of Theorem 3.
</div>

<div class="proof" markdown="1">
*Proof.* By the chord identity, for a unit vector $$u$$, $$g_S(ru)=V-r\lvert\Pi_uS\rvert+\int(r-L_u(y))_+dy$$, where $$L_u(y)$$ is the length of the chord of $$S$$ along $$u$$ through the point $$y$$ of the shadow $$\Pi_uS\subset u^\perp$$. A point $$y$$ of the shadow at distance $$\delta$$ from the shadow boundary lies on the inward normal of the nearest boundary point $$x_0$$; the plane through $$x_0$$ spanned by $$u$$ and that normal contains the normal section, and the rolling ball gives $$L_u(y)\ge2\sqrt{2r_c\delta-\delta^2}$$. So $$L_u(y)<r$$ only within $$\delta\le r^2/(8r_c)(1+o(1))$$ of the shadow boundary, and the last integral is at most $$r\cdot{\rm Per}(\Pi_uS)\cdot r^2/(8r_c)=O(r^3)$$ uniformly in $$u$$.
</div>

So balls, ellipsoids and every convex body with a $$C^{1,1}$$ boundary have no finite term in (7), although the boundary of the ball $$I\times B_R$$ has $$K=2/R$$ and $$\oint K\sqrt h=8\pi RT$$. A Gibbons–Hawking–York term with coefficient $$\gamma$$ would add $$\gamma\cdot8\pi RT$$ to the finite part, a quantity that grows linearly with $$R$$, while the computed finite part is zero for every $$R$$. The faces $$t=0$$ and $$t=T$$ meet the lateral boundary at right angles, and the corresponding joint values vanish in Theorem 2 (a timelike–spacelike joint with Lorentz-orthogonal edges), in agreement with the absence of a further finite term in (7). Hence on these regions the finite part of the mean contains no term proportional to the integrated extrinsic curvature, $$\gamma=0$$, whereas the variational principle needs $$\gamma=1$$.

For bodies with edges, $$a_2\neq0$$. For the cube $$S=[0,s]^3$$ one has $$g_S(ru)=\prod_i(s-r\lvert u_i\rvert)$$ for small $$r$$, so $$a_1=-\tfrac32s^2$$ (equal to $$-{\rm Area}/4$$ since $$\langle\lvert u_i\rvert\rangle=\tfrac12$$) and $$a_2=3s\langle\lvert u_1u_2\rvert\rangle=2s/\pi$$, with $$\langle\lvert u_1u_2\rvert\rangle=2/(3\pi)$$ on the sphere. The finite part is $$-9\cdot\tfrac23\cdot\tfrac{2s}\pi T=-12sT/\pi$$, that is $$-1/\pi$$ for each of the twelve edges of the cube, per unit area of the edge swept in time.

<div class="proposition" markdown="1">
**Proposition 5** (edges of prisms). Let $$S=P_2\times[0,L]$$ with $$P_2$$ a convex polygon in the plane and $$L\to\infty$$. Per unit length of the prism and for the kernels of Theorem 3, the finite part of (7) equals $$-\sum_v\big(1-\delta_v\cot\delta_v\big)\,T/\pi$$, the sum over the vertices of $$P_2$$ with exterior angles $$\delta_v=\pi-\beta_v$$.
</div>

<div class="proof" markdown="1">
*Proof.* Per unit length, $$\bar g$$ is the average over $$S^2$$ of the planar covariogram of $$P_2$$ at $$r\sin\theta\,(\cos\phi,\sin\phi)$$. By Proposition 1 applied to $$P_2$$ (with the Euclidean metric), the $$r^2$$ term of the planar covariogram is the sum of the vertex terms $$[\det(h,a_v)\det(h,b_v)]_+/(2\lvert\det(a_v,b_v)\rvert)$$, so $$a_2=\langle\sin^2\theta\rangle\sum_v\varphi(\beta_v)=\tfrac23\sum_v\varphi(\beta_v)$$ with $$\varphi(\beta)=\frac1{2\pi}\int_0^{2\pi}\frac{[\det(u_\phi,a)\det(u_\phi,b)]_+}{2\lvert\det(a,b)\rvert}d\phi=\frac{1+(\pi-\beta)\cot\beta}{4\pi}$$ for unit edge vectors at the angle $$\beta$$. The finite part is $$-9\cdot\tfrac23a_2T=-4\sum_v\varphi(\beta_v)T$$, and $$\cot\beta=-\cot\delta$$.
</div>

We verified the closed form of $$\varphi$$ by numerical integration at five angles (agreement to $$10^{-15}$$). For a square prism the formula gives $$-4/\pi=-1.27324$$ per unit length and unit $$T$$; for an equilateral triangle $$-3(1+\tfrac{2\pi}{3\sqrt3})/\pi=-2.1095$$; for a regular $$n$$-gon $$-n(1-\delta\cot\delta)/\pi\approx-4\pi/(3n)$$, which tends to zero as the polygon approaches a disc, consistent with Proposition 4.

The edge values also bear on corner terms of the usual form. Rounding the edge of a prism with a radius $$r_c>0$$ produces a $$C^{1,1}$$ boundary, for which the finite part is zero by Proposition 4, however small $$r_c$$ is; the sharp edge has the value just computed, so the mean action is not continuous under rounding at fixed density (the crossover is at $$r_c$$ of the order of the discreteness length). Consider a boundary term $$\alpha\oint K\sqrt h+\beta\sum_e\delta_eA_e$$, the first term integrated over the smooth part of the boundary and the second a Hayward-type corner term (exterior angle $$\delta_e$$ times edge area $$A_e$$). On a rounded prism the corner term is absent and the integral of $$K$$ over each rounded arc equals the exterior angle, so the boundary term equals $$2\pi\alpha T$$ per unit length; the mean action gives $$0$$, so $$\alpha=0$$. On a sharp prism $$K=0$$ on the faces and the boundary term is $$2\pi\beta T$$ whatever the polygon, because the exterior angles sum to $$2\pi$$, while the mean action gives $$-4/\pi$$ for the square and $$-2.1095$$ for the triangle. So no pair $$(\alpha,\beta)$$ reproduces the finite part of the mean action on prisms, and no assumption about locality was used.

<div class="remark" markdown="1">
**Remark** (Gauss–Bonnet degeneracy). On regions of the form $$P\times\mathbb R^2$$ a term $$\gamma\oint K\sqrt h$$ cannot be separated from joint terms: $$\oint K\sqrt h=A_\perp\int_{\partial P}k\,ds$$ and by the Gauss–Bonnet relation for closed curves in the Minkowski plane the integral of the curvature of the boundary curve is fixed by the boost angles at its corners (up to constants that depend only on the causal types of the pieces), so a curved timelike edge can be traded for corner terms. This is why the tests above use the ball, for which $$\oint K\sqrt h=8\pi RT$$ grows with $$R$$ while the finite part of the mean is zero for every $$R$$.
</div>

The regime is the embedded one, the background is flat, the boundary is static (a product in time) and only the mean is computed. A boundary that is boosted or accelerated, a curved background, and the isolated regime are not covered by Theorem 3. A numerical evaluation for an accelerated, non-product boundary reported a finite part equal to the joint at the top of the region and a Gibbons–Hawking–York coefficient consistent with zero at the level $$10^{-4}$$; that evaluation is quoted here without having been reproduced.

## 7. What interval functionals cannot give at a flat joint

The area law has a factor $$2\pi$$ that arises, in the Euclidean treatment, from a conical singularity of opening $$2\pi$$. One could hope that a causal-set action reproduces this through a joint function that, continued to imaginary angle, contains a term linear in the deficit angle, as the Regge and Hayward joint terms do. Consider a functional of the interval alone: a pair functional $$F$$ that depends only on the induced order of the interval of $$p\prec q$$. In flat Minkowski space the Poincaré group maps every timelike interval of volume $$V$$ to the standard interval of that volume, so $$\mathbb E[F]=F_{\rm eff}(\rho V_{pq})$$ for a function $$F_{\rm eff}$$, and the mean has the form (2) with $$f=F_{\rm eff}$$. Chain counts, longest-chain statistics and layer counts are of this kind. Assume the Mellin transform $$M_F(s)=\Gamma(s)Q_F(s)$$ of $$F_{\rm eff}$$ is meromorphic and regular at $$s=\tfrac32$$, with $$Q_F(\tfrac12)=Q_F(1)=0$$ and the element count chosen to cancel the volume term.

<div class="proposition" markdown="1">
**Proposition 6** (flat joints). For $$P\times\mathbb R^2$$ with $$P$$ a convex polygon whose edges are spacelike or null, the mean of such a functional is

$$
\langle S_F\rangle=3(m_u+m_w)\big[Q_F'(\tfrac32)-Q_F(\tfrac32)\log\rho\big]+Q_F(\tfrac32)\,c(P)+O(\rho^{-1/2}),\qquad c(P)=3(m_u+m_w)\Big[\psi(\tfrac32)-1+\log\frac{24}\pi\Big]-6h_1, \tag{8}
$$

where $$\psi$$ is the digamma function and $$h_1$$ is the coefficient of $$\mu$$ in the expansion of Proposition 2. In particular, if $$Q_F(\tfrac32)=0$$ the joint contribution is that of Theorem 2, $$\tfrac32Q_F'(\tfrac32)j_v$$ with $$j_v=\coth\theta_v$$ at the joints between spacelike edges of the same spatial direction and $$1$$ at the corresponding joints with a null edge. If $$Q_F(\tfrac32)\neq0$$ the logarithmic term is still proportional to $$m_u+m_w=\tfrac12\sum_vj_v$$, and the constant $$c(P)$$ is not a sum over joints: under $$P\to\lambda P$$ one finds $$h_1\to h_1+2(m_u+m_w)\log\lambda$$, so $$c(P)$$ depends on the size of the polygon.
</div>

*Proof.* The derivation of Theorem 1 used the functional only through the moments $$m_\Psi$$. Without (K3) the term $$(m_u+m_w)\mu\log(1/\mu)$$ of $$H$$ contributes $$6\rho\sigma\,(m_u+m_w)[-m_\Psi(2)\log\sigma-m_\Psi'(2)]$$ with $$m_\Psi(2)=-\tfrac14\Gamma(\tfrac32)Q_F(\tfrac32)\neq0$$, and the term $$h_1\mu$$ contributes $$6\rho\sigma h_1m_\Psi(2)$$. Using $$m_\Psi'(2)=-\tfrac18\Gamma(\tfrac32)[Q_F'(\tfrac32)+(\psi(\tfrac32)-1)Q_F(\tfrac32)]$$, $$\sigma=(24/\pi\rho)^{1/2}$$, and the numerical factor $$3$$ of Theorem 1 gives (8). For the scaling, $$K_{\lambda P}(a,b)=\lambda^2K_P(a/\lambda,b/\lambda)$$ gives $$H_{\lambda P}(\mu)=\lambda^2H_P(\mu/\lambda^2)$$.

We checked (8) for the kernel $$Q_F=(s-\tfrac12)(s-1)(s-2)$$ (so $$Q_F(\tfrac32)=Q_F'(\tfrac32)=-\tfrac14$$, $$b=\tfrac32$$) on the causal diamond ($$m_u+m_w=1$$, $$h_1=3.3863$$ from a fit of $$H$$): the exact mean is $$15.61565284$$ and $$19.06953048$$ at $$\rho=10^7$$ and $$10^9$$, against $$15.61565284$$ and $$19.06953048$$ from (8). On a lens-shaped polygon the increase of the mean per factor $$100$$ in $$\rho$$ is $$6.2562$$, against $$3\cdot\tfrac14(m_u+m_w)\log100=6.2562$$.

The dependence of the leading local term on the angle is therefore always $$\coth\theta$$ (with the value $$1$$ for a null edge), whatever the interval functional. Continued to imaginary angle, $$\coth(i\varphi)=-i\cot\varphi$$, a function of period $$\pi$$ with no term linear in $$\varphi$$. The function $$\coth(\theta/2)=\coth\theta+\operatorname{csch}\theta$$, which would continue to $$-i\cot(\varphi/2)$$ with the period $$2\pi$$ of a cone, is linearly independent of $$\{\coth\theta,1\}$$ and is not reachable. This is the sense in which the factor $$2\pi$$ is not produced by these actions at a flat joint. The statement is limited. It concerns functionals of the interval alone, on flat joints with spacelike or null edges, with a symbol regular at $$\tfrac32$$. It does not say that no causal-set action can carry a conical factor: functionals that use data outside the interval (a slice, a horizon, spacelike pairs), kernels with a pole at $$s=\tfrac32$$, curved backgrounds and boost-conical hinges, at which the reduction of section 2 fails, are outside it. The molecule count of [Horizon counts: which weights give the area law](/quantum-gravity/area-projector/) is of the first kind of exception: it is defined relative to a slice and is not an interval functional.

## 8. The isolated regime

In the isolated regime the interval of a counted pair is intersected with the region, so a boundary reduces the volume used in the kernel. The statements of this section are not proved here and the numbers in it are quoted from numerical work that we have not reproduced.

*Flat faces.* For a flat timelike face and a kernel with $$Q(1)=0$$ the difference between the isolated and embedded means per unit area of the face is proportional to $$Q(\tfrac54)$$. The mechanism is that the stabiliser $$SO(2,1)$$ of the face leaves the cut volume invariant, so that the pair integral factorises into an integral over the stabiliser orbit and a Mellin integral at $$s=1,\tfrac54,\tfrac32$$; the absolute convergence of the orbit integral in the presence of two faces is the one step that is argued and not proved. If this holds, members of the family of Proposition 3 have no face term in either regime, and corners with one or two timelike faces take their embedded values.

*Curved faces.* We state the following as a derivation sketch with numerical support, not as a theorem. In the isolated regime a pair moving tangentially to a boundary of radius of curvature $$\mathcal R$$ has its interval cut by the sagitta $$\tau^2u_t^2/(2\mathcal R)$$ of the boundary. This cuts off the tangential boost integral at $$u_t\sim(\mathcal R/\tau)^{1/2}$$ and moves the Mellin points at which boundary terms sit from the single value $$s=\tfrac54$$ to a tower $$s=1+k/8$$, $$k\ge1$$, of which only $$s=\tfrac98,\tfrac54,\tfrac{11}8$$ give divergent powers $$\rho^{3/2-s}$$, namely $$\rho^{3/8}$$, $$\rho^{1/4}$$ and $$\rho^{1/8}$$. The term at $$s=\tfrac98$$ would be proportional to $$Q(\tfrac98)\int\sqrt h\oint d\phi\,k(\phi)^{-1/2}$$, with $$k(\phi)$$ the normal curvature along the direction of motion, so it is not analytic in the extrinsic curvature. For the minimal member of Proposition 3, $$Q_\star(\tfrac98)=\tfrac5{256}\neq0$$. If the sketch is correct, the family with a zero at $$\tfrac54$$ is not finite on a curved timelike boundary in the isolated regime; adding the zero $$Q(\tfrac98)=0$$ does not suffice, because a term proportional to $$Q(\tfrac{11}8)$$ remains and, for a boundary with a flat direction such as a cylinder, a term proportional to $$Q'(\tfrac98)$$; kernels of degree six (zeros at $$\tfrac98,\tfrac54,\tfrac{11}8$$) and eight (double zeros at $$\tfrac98$$ and $$\tfrac{11}8$$) would be finite. Numerical evaluations of a parabolic local model for the ball and the cylinder gave a coefficient of the $$s=\tfrac98$$ term compatible with this form to about one percent and the divergent terms of the minimal members at the percent level; direct finite-density fits in four dimensions were inconclusive. The absence of logarithms in the tower, and the completeness of the exponent set, are the unproved steps. The embedded-regime results above do not depend on this section.

## 9. What the results do not exclude

The theorems concern the mean, in flat space, for product regions in the embedded regime. The following are not excluded.

1. A boundary term in the isolated regime or on curved boundaries that are not static products. Theorem 3 does not cover boosted or accelerated boundaries or curved backgrounds. The isolated regime and the variance of the action are not examined for such a term.
2. Terms at order the discreteness length. The curvature of the boundary enters at order $$\rho^{-1/4}$$ in (7), proportional to $$Q(\tfrac74)a_3$$. For the ball the coefficient does not depend on the radius, as a term of the form (discreteness length) times $$\oint K^2\sqrt h$$ would not; the form of this term for general boundaries was not determined.
3. Kernels outside the layer family: continuum kernels with a pole at $$s=\tfrac32$$ or with power-law tails, functionals that use data other than the interval, and non-Poisson substrates.
4. The mean is not the action. The mean of $$S$$ is expected to tend to the Einstein–Hilbert action plus corner terms, but $$S$$ is a random variable, and the absence of a Gibbons–Hawking–York term in the mean says only that the mean does not contain one.
5. Conical hinges. The structure used in sections 2 and 3 does not describe a boost-conical hinge, and the corner function there is not determined by the flat-joint analysis.
6. The factor $$2\pi$$ from the state of the matter fields rather than from the action; the results above concern the action only.

## 10. Open points

The expansion of $$H$$ is proved for polygons; for curved faces the covariogram is not piecewise polynomial and the proof does not apply, so triple poles of the Mellin transform at $$s=\tfrac32$$ are excluded for polygons only. The coefficient $$-9Q'(\tfrac32)a_2T$$ is proved for static products; whether the finite part of a general curved timelike boundary in the embedded regime is a local boundary integral, and if so which, is open. The isolated-regime divergence on curved boundaries and the finiteness of the degree-six and degree-eight kernels there rest on the sketch of section 8. Whether the null-edge values of Theorem 2 are uniform in the density as an edge tilts toward null is not proved.

## References

1. <a id="ref-bd"></a>Benincasa, Dowker, “The scalar curvature of a causal set,” Phys. Rev. Lett. 104, 181301 (2010), [arXiv:1001.2725](https://arxiv.org/abs/1001.2725).
2. <a id="ref-dg"></a>Dowker, Glaser, “Causal set d'Alembertians for various dimensions,” Class. Quantum Grav. 30, 195016 (2013), [arXiv:1305.2588](https://arxiv.org/abs/1305.2588).
3. <a id="ref-glaser"></a>Glaser, “A closed form expression for the causal set d'Alembertian,” [arXiv:1311.1701](https://arxiv.org/abs/1311.1701).
4. <a id="ref-ass"></a>Aslanbeigi, Saravani, Sorkin, “Generalized causal set d'Alembertians,” JHEP 1406, 024 (2014), [arXiv:1403.1622](https://arxiv.org/abs/1403.1622).
5. <a id="ref-belenchia"></a>Belenchia, “Universal behaviour of generalized causal set d'Alembertians in curved spacetime,” Class. Quantum Grav. 33, 135011 (2016), [arXiv:1510.04665](https://arxiv.org/abs/1510.04665).
6. <a id="ref-bdjs"></a>Buck, Dowker, Jubb, Surya, “Boundary terms for causal sets,” Class. Quantum Grav. 32, 205004 (2015), [arXiv:1502.05388](https://arxiv.org/abs/1502.05388).
7. <a id="ref-dowker"></a>Dowker, “Boundary contributions in the causal set action,” [arXiv:2007.13206](https://arxiv.org/abs/2007.13206).
8. <a id="ref-mw"></a>Machet, Wang, “On the continuum limit of Benincasa–Dowker–Glaser causal set action,” Class. Quantum Grav. 38, 015010 (2021), [arXiv:2007.13192](https://arxiv.org/abs/2007.13192).
9. <a id="ref-dllj"></a>Dowker, Liu, Lloyd-Jones, “Timelike boundary and corner terms in the causal set action,” Class. Quantum Grav. 42, 205006 (2025), [arXiv:2501.00139](https://arxiv.org/abs/2501.00139).
