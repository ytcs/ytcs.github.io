---
title: "Horizon counts: which weights give the area law"
date: 2026-10-29
category: entropy
order: 1
description: Weighting causal-set horizon molecules by the number of elements to their future can make the expected count respond to curvature exactly as the horizon area does, at every size of the region, but only if the weights take both signs. No non-negative weighting does it, and the Einstein form with the area-law Newton constant is automatic once the non-area terms are removed.
status: "Theorems proved in full for Poisson sprinklings at linear order in uniform curvature; exact coefficients computed by computer algebra for 3 ≤ d ≤ 12; Monte Carlo checks in d = 3 and 4."
excludes: "Entropy functionals that assign a non-negative amount to each horizon molecule and reproduce the Einstein equation with the area-law Newton constant at every ball size (Poisson sprinkling, expected count, linear order in uniform curvature)."
---

In a causal set the entropy of a horizon is modelled by counting horizon molecules, pairs of elements that straddle the horizon just below a spacelike slice. Barton, Counsell, Dowker, Gould, Jubb and Taylor [[Barton et al.]](#ref-barton) showed that the expected number of such molecules in a Poisson sprinkling tends, as the discreteness length goes to zero, to a constant times the area of the horizon cross-section, and that the count is local to the cross-section. At finite density the count also contains terms suppressed by powers of the discreteness length, which for a ball a few discreteness lengths across make its response to curvature differ from that of the area. The question here is which weightings of the molecules by their type $$m$$ (the number of further elements in the future of the lower element, below the slice) give an expected count that responds to curvature exactly as the area does, at every size of the ball, and whether such a weighting can be non-negative.

The answer, for Poisson sprinklings, for the expected count, and at linear order in curvature that is uniform over a geodesic ball, is as follows. A weighting works if and only if a polynomial built from the weights, the Mellin symbol, vanishes at $$d$$ prescribed points. The solution of lowest degree is unique up to normalisation and has integer weights that alternate in sign; in four dimensions they are $$(210,-791,1268,-928,256)$$. No non-negative weighting works: for $$3\le d\le12$$ a non-negative weighting produces a Newton constant strictly smaller than the one defined by its own flat-space area law, at every ball size. The Einstein form of the response and the equality of the two Newton constants are not independent checks. Both hold for every weighting that removes the non-area terms, so they test the counting object and not the choice of weights. The signed count is a Richardson extrapolation: its mean equals the mean of a combination of non-negative counts of thinned sprinklings, so the sign of the count on one sprinkling depends on the representation of that mean.

Horizon molecules in the form used here, their flat-space coefficient and the locality of the leading term are due to [[Barton et al.]](#ref-barton), following the links of Dou and Sorkin [[Dou–Sorkin]](#ref-dou-sorkin). Machet and Wang [[Machet–Wang]](#ref-mw) found that a version of the molecule defined by a null hypersurface through the horizon is not local in a curved spacetime; the molecules below are defined by a spacelike slice, as in [[Barton et al.]](#ref-barton). Jacobson [[Jacobson]](#ref-jacobson) derives the Einstein equation from the stationarity of the entropy of small geodesic balls at fixed volume, assuming an area-law entropy. Signed combinations of layer counts with prescribed Mellin zeros define the causal-set d'Alembertian and action [[BD]](#ref-bd) [[Dowker–Glaser]](#ref-dg) [[Glaser]](#ref-glaser) [[ASS]](#ref-ass); the weights here have the same form and a different purpose (section 4). This page adds the exact density dependence of the weighted count and the curvature response in every dimension from 3 to 12, the characterisation of the exact weights, the proof that non-negative weights are excluded, the observation that the Einstein form and the Newton constant are automatic, and the representation of the signed count through thinning, together with what thinning does to the signs of the weights. How exact weights behave when every order in the gradients of the curvature is required is treated in [Exact causal-set horizon weights must be non-local](/quantum-gravity/nonlocal-weights/).

## 1. Causal sets, sprinklings and horizon molecules

A *causal set* is a locally finite partially ordered set, read as a discretisation of a Lorentzian manifold in which the order is the causal relation and the number of elements in a region is its volume. A *Poisson sprinkling* of density $$\rho$$ into a $$d$$-dimensional spacetime is a Poisson point process with intensity $$\rho$$ times the volume measure, ordered by the causal relation $$\prec$$. Its law is invariant under the isometries of Minkowski space, and it is the only random discretisation considered on this page. We write $$\ell_\rho=\rho^{-1/d}$$ for the discreteness length and set $$\rho=1$$ except where the density dependence matters. Two properties of the Poisson process are used throughout. The number of points in a region of volume $$V$$ is Poisson distributed with mean $$\rho V$$, and the multivariate Mecke equation [[Last]](#ref-last) says that the expectation of a sum over $$k$$-tuples of distinct points equals the integral over the positions of the tuple of the expectation, with the process in the integrand replaced by the process plus the points of the tuple.

Let $$d\ge3$$ and $$n=d-1$$. Let $$\Sigma$$ be a spacelike slice, $$B\subset\Sigma$$ a geodesic ball and $$o$$ the top tip of its domain of dependence. The horizon is the part of $$\partial I^-(o)$$ below $$\Sigma$$. It meets $$\Sigma$$ in $$\partial B$$, and as one goes to the past its generators move away from $$B$$, so it has no caustic below $$\Sigma$$. Write $$P=I^-(o)\cap I^-(\Sigma)$$ for the region inside the horizon. In a drawing with time upward, $$P$$ is the region below $$\Sigma$$ bounded by the two lines that spread outward from the ends of $$B$$.

For an element $$p\in P$$ let $$F(p)$$ be the set of elements in the causal future of $$p$$ and the past of $$\Sigma$$. A *horizon molecule* is a pair $$p\prec q$$ with $$p\in P$$ and $$q\in F(p)\setminus P$$; its *type* is $$m=\lvert F(p)\rvert-1$$, the number of elements of $$F(p)$$ other than $$q$$, wherever they lie. A molecule of type $$0$$ is a link $$p\prec q$$ across the horizon, the molecule of [[Barton et al.]](#ref-barton). Given real weights $$w_0,w_1,\dots,w_K$$, the *weighted count* $$H_w$$ is the sum of $$w_m$$ over all molecules, and $$H_m$$ is the number of molecules of type $$m$$.

Let $$V(p)$$ be the volume of $$J^+(p)\cap I^-(\Sigma)$$ and $$V_{\rm out}(p)$$ the volume of its part outside $$P$$. Given $$p$$ and $$q$$, the remaining elements of $$F(p)$$ form a Poisson process of mean $$V(p)$$, so the probability that there are $$m$$ of them is $$V^me^{-V}/m!$$. The Mecke equation gives

$$
\langle H_w\rangle=\int_P dV_p\;V_{\rm out}(p)\,\phi\big(V(p)\big),\qquad \phi(v)=e^{-v}\sum_{m}w_m\frac{v^m}{m!}. \tag{1}
$$

The weights enter only through the kernel $$\phi$$. Its Mellin transform defines the *Mellin symbol* of the weighting,

$$
\Gamma(\alpha)\,Q(\alpha)=\int_0^\infty\phi(v)\,v^{\alpha-1}\,dv,\qquad Q(\alpha)=\sum_m w_m\frac{(\alpha)_m}{m!}, \tag{2}
$$

where $$(\alpha)_m=\alpha(\alpha+1)\cdots(\alpha+m-1)$$; the second form follows from $$\int_0^\infty v^{m+\alpha-1}e^{-v}dv=\Gamma(\alpha)(\alpha)_m$$. A weighting with $$w_m=0$$ for $$m>K$$ has a polynomial symbol of degree $$K$$, and $$w\mapsto Q$$ is a linear bijection onto polynomials of degree at most $$K$$, because $$(\alpha)_m/m!$$ has degree exactly $$m$$. The weights are recovered from the values of $$Q$$ at the non-positive integers, using $$(-j)_m/m!=(-1)^m\binom jm$$ and binomial inversion:

$$
w_m=\sum_{j=0}^{m}(-1)^j\binom mj\,Q(-j). \tag{3}
$$

The type-0 count has $$Q\equiv1$$. The Mellin symbol is the only way the weights will enter; the idea is that weighting by $$m$$ acts on the density dependence of the count, and section 2 makes this exact.

## 2. The mean count in a conformally static background

All computations use one exact reduction. Let the spacetime be

$$
g=W(t)^{2/d}\big(-dt^2+\gamma_A\big),
$$

with $$\gamma_A$$ the round metric on the $$n$$-sphere of radius $$A$$ ($$A=\infty$$ gives flat space) and $$W(0)=1$$. The power $$2/d$$ makes the volume element $$W\,dt\,dV_\gamma$$. Take $$\Sigma=\{t=0\}$$ and $$B$$ the geodesic ball of radius $$r_0$$ about a pole. Write $$\omega$$ for the volume of the unit $$n$$-ball, $${\rm cap}(r)$$ for the $$\gamma_A$$-volume of a geodesic ball of radius $$\min(r,\pi A)$$, and $$\ell$$ for the radius of the flat ball of the same volume as $$B$$: $${\rm cap}(r_0)=\omega\ell^n$$. The case $$W=1$$ is the Einstein static universe and $$W=\cos^{-d}(t/A)$$ is de Sitter space of radius $$A$$ in conformal time; the slice $$t=0$$ is the same round sphere in both, with vanishing extrinsic curvature.

<div class="lemma" markdown="1">
**Lemma 1** (reduction). With $$\tau$$ the coordinate depth of $$p$$ below $$\Sigma$$,

$$
\langle H_w\rangle=\int_0^\infty d\tau\,W(-\tau)\,\phi\big(V(\tau)\big)\,U(\tau), \tag{4}
$$

$$
U(\tau)=\int_0^\tau ds\,W(s-\tau)\,{\rm cap}(s)\big[{\rm cap}(r_0+\tau)-{\rm cap}(r_0+\tau-s)\big],\qquad V(\tau)=\int_0^\tau ds\,W(s-\tau)\,{\rm cap}(s).
$$

</div>

<div class="proof" markdown="1">
*Proof.* Conformal factors leave the causal relation unchanged, so $$(t_1,y_1)\prec(t_2,y_2)$$ exactly when the $$\gamma_A$$-distance from $$y_1$$ to $$y_2$$ is at most $$t_2-t_1$$. The tip $$o$$ is at time $$r_0$$ above the pole, so $$P$$ at time $$-\tau$$ is the cap $$C_1$$ of radius $$r_0+\tau$$ about the pole, and $$V(p)=V(\tau)$$ depends only on the depth. An element $$q$$ at time $$s-\tau$$ lies in $$F(p)$$ for $$p=(-\tau,y_p)$$ when its position is in the cap of radius $$s$$ about $$y_p$$, and lies outside $$P$$ when it is outside the cap $$C_2$$ of radius $$r_0+\tau-s$$ about the pole. Integrating over $$y_p\in C_1$$, the first volume gives $$\mathrm{cap}(s)\,\mathrm{cap}(r_0+\tau)$$. The part inside $$C_2$$ is computed by exchanging the roles of $$y_p$$ and $$y_q$$: for $$y_q\in C_2$$ the cap of radius $$s$$ about $$y_q$$ lies inside $$C_1$$ by the triangle inequality, so it contributes $$\mathrm{cap}(s)\,\mathrm{cap}(r_0+\tau-s)$$. The volume elements are $$W(-\tau)d\tau\,dV_\gamma$$ for $$p$$ and $$W(s-\tau)ds\,dV_\gamma$$ for $$q$$.
</div>

Equation (4) is exact, with no wrapping restriction beyond the definition of $$\mathrm{cap}$$. As a check we sprinkled the Einstein static universe with $$d=3$$, $$A=3$$, $$r_0=0.9$$ (non-perturbatively), and compared the mean number of molecules of type $$m=0,1,2,3$$ over $$2\times10^4$$ realisations with a numerical quadrature of (4): the Monte Carlo means $$2.047,\,3.654,\,5.150,\,6.579$$ (standard errors $$0.011,\,0.019,\,0.026,\,0.034$$) against $$2.051,\,3.687,\,5.180,\,6.592$$.

Substituting $$v=\omega\tau^d/d$$ turns every $$\tau$$-integral that will occur into a value of the symbol.

<div class="lemma" markdown="1">
**Lemma 2** (moment rule). Let $$b=(k+1)/d$$ and let $$r\ge0$$ be an integer with $$b>r$$. Then

$$
\int_0^\infty\tau^k\,\phi^{(r)}\!\Big(\frac{\omega\tau^d}{d}\Big)d\tau=(-1)^r\,\frac1d\Big(\frac d\omega\Big)^{b}\,\Gamma(b)\,Q(b-r). \tag{5}
$$

</div>

<div class="proof" markdown="1">
*Proof.* The substitution gives $$\frac1d(d/\omega)^b\int_0^\infty v^{b-1}\phi^{(r)}(v)\,dv$$. Integrating by parts $$r$$ times, the boundary terms vanish because $$b-j>0$$ for $$j=1,\dots,r$$, and $$\int v^{b-1}\phi^{(r)}=(-1)^r(b-1)\cdots(b-r)\int v^{b-r-1}\phi=(-1)^r\Gamma(b)Q(b-r)$$ by (2).
</div>

The remaining input is how the exponents of the symbol are fixed. Expand (4) in the curvature. Every term of the expansion has the form $$c\,\ell^e\,\tau^k\,\phi^{(r)}(\omega\tau^d/d)$$ with $$r\ge0$$ derivatives of $$\phi$$ (they come from expanding $$\phi(V)$$ about the flat volume $$V_0=\omega\tau^d/d$$ in the corrections to $$V$$), multiplied by a curvature factor of length dimension $$-D$$: $$A^{-2\nu}$$ has $$D=2\nu$$, and the coefficient of $$t^k$$ in $$W$$ has $$D=k$$. Each factor of (4) is homogeneous under a simultaneous rescaling of $$\tau$$, $$s$$, $$\ell$$ and $$A$$, and each correction to $$V$$ contributes a factor of length dimension $$d+D_i$$. Counting powers of length gives $$e+k+1=2d+D+rd$$, where the $$1$$ comes from $$d\tau$$. By (5) the term is multiplied by $$Q(b-r)$$ with $$b=(k+1)/d$$, that is, by $$Q(\beta)$$ with

$$
\beta=2-\frac{e-D}{d}. \tag{6}
$$

In terms of the density the term is proportional to $$\rho^{2-\beta}$$: a weighting multiplies each power of the density by the value of $$Q$$ at the matching exponent, and a weighting whose symbol vanishes at all exponents but one keeps a single term of the density expansion. For example the flat area term $$\ell^{d-2}$$ has $$D=0$$ and $$\beta=1+2/d$$, which we call the *area exponent* $$\alpha_A$$.

## 3. The flat count and the curvature response

In flat space $$W=1$$, $${\rm cap}(r)=\omega r^n$$ and $$V=\omega\tau^d/d$$. Expanding the bracket in (4) binomially in $$\ell$$, the term of order $$\ell^n$$ cancels and

$$
U(\tau)=\omega^2\sum_{k=1}^{n}\binom nkI_k\,\ell^{\,n-k}\tau^{d+k},\qquad I_k=\int_0^1\zeta^n\big[1-(1-\zeta)^k\big]d\zeta=\frac1d-\frac{n!\,k!}{(n+k+1)!}.
$$

<div class="proposition" markdown="1">
**Proposition 1** (flat count). For a flat ball of radius $$\ell$$,

$$
\langle H_w\rangle=\sum_{k=1}^{n}\binom nkI_k\,\frac{\omega^2}{d}\Big(\frac d\omega\Big)^{\beta_k}\Gamma(\beta_k)\,Q(\beta_k)\,\ell^{\,n-k},\qquad\beta_k=\frac{d+1+k}{d}. \tag{7}
$$

There is no term proportional to $$\ell^n$$, the volume of $$B$$. The leading term is $$Q(\alpha_A)\,a_d\,\lvert\partial B\rvert$$ with $$\lvert\partial B\rvert=n\omega\ell^{n-1}$$ and

$$
a_d=\frac{\omega}{d(d+1)}\Big(\frac d\omega\Big)^{\alpha_A}\Gamma(\alpha_A),\qquad\alpha_A=\frac{d+2}{d}. \tag{8}
$$

</div>

The proof is the computation above followed by (5) with $$r=0$$, and $$I_1=1/(d+1)$$ gives (8). The exponents present are $$\alpha_A,\alpha_A+\tfrac1d,\dots,2$$, that is $$\alpha_A+j/d$$ for $$0\le j\le d-2$$. Equation (8) gives $$a_3=0.218853$$, $$a_4=\sqrt3/10$$, $$a_5=0.148656$$, $$a_6=0.133258$$, $$a_7=0.122656$$, $$a_8=0.114883$$; the values for $$d=3$$ and $$4$$ are those found in [[Barton et al.]](#ref-barton). In four dimensions

$$
\langle H_w\rangle=\frac{2\sqrt3\,\pi}{5}\,Q(\tfrac32)\,\ell^2+\frac{7\cdot3^{3/4}\pi^{1/4}\Gamma(\frac34)}{10}\,Q(\tfrac74)\,\ell+\frac{34}{35}\,Q(2). \tag{9}
$$

For the type-0 count this is $$2.17656\,\ell^2+2.60322\,\ell+0.97143$$. We checked (9) by sprinkling a flat ball of radius $$\ell=1$$ in four dimensions, $$3000$$ realisations, depth cut $$2.6$$ (beyond which $$e^{-V}<10^{-20}$$). The mean numbers of molecules of types $$0,\dots,5$$ were $$5.800\pm0.047$$, $$9.788\pm0.081$$, $$13.445\pm0.114$$, $$16.303\pm0.143$$, $$19.286\pm0.175$$, $$22.458\pm0.203$$, against $$5.751$$, $$9.763$$, $$13.259$$, $$16.477$$, $$19.512$$, $$22.413$$ from (9) with $$Q(\alpha)=(\alpha)_m/m!$$; all six pulls are below $$1.7$$.

### The response to curvature

In curved space we hold the volume of $$B$$ fixed at $$\omega\ell^n$$, so that $$\ell$$ labels its size, and expand in $$\varepsilon=A^{-2}$$ and in the Taylor coefficients of $$W$$. The part of the expected count that sits at the area exponent is the exact area of $$\partial B$$.

<div class="theorem" markdown="1">
**Theorem 1** (area exponent). Let $$W(t)=1+\sum_{k\ge1}w_kt^k$$ and expand $$\langle H_w\rangle$$ in (4) as a formal power series in $$\varepsilon$$ and the $$w_k$$, with $$B$$ of fixed volume $$\omega\ell^n$$. Every coefficient is a polynomial in $$\ell$$, and each monomial $$\ell^e$$ multiplies $$Q(\beta)$$ with $$\beta$$ given by (6). Collecting the terms with $$\beta=\alpha_A$$ over all orders gives exactly $$a_d\,Q(\alpha_A)$$ times the area of $$\partial B$$. Every other term has $$\beta=\alpha_A+j/d$$ with an integer $$j\ge1$$.
</div>

The idea is that $$\ell$$ enters (4) only through $$r_0$$ inside the bracket, and at each order the highest power of $$\ell$$ comes from the first derivative $${\rm cap}'(r_0)$$, which is the area of $$\partial B$$, with every other factor at its flat value.

<div class="proof" markdown="1">
*Proof.* The cap volume is $${\rm cap}(r)=\sum_\nu\omega_\nu\varepsilon^\nu r^{n+2\nu}$$ with $$\omega_0=\omega$$, and the fixed-volume condition gives $$r_0=\ell\,\psi(\varepsilon\ell^2)$$ for a power series $$\psi$$ with $$\psi(0)=1$$. Expand the bracket of $$U$$ as $$\sum_{i\ge1}{\rm cap}^{(i)}(r_0)\,[\tau^i-(\tau-s)^i]/i!$$. The part of order $$\varepsilon^\nu$$ of $${\rm cap}^{(i)}(r_0)$$ is a constant times $$\ell^{\,n+2\nu-i}$$, since each power of $$\varepsilon$$ that comes through $$r_0$$ or the cap series carries $$\ell^2$$. No other factor of (4), including the expansion of $$\phi(V)$$ about $$\omega\tau^d/d$$ and the factors $$W$$, depends on $$\ell$$, and their $$\tau$$- and $$s$$-integrals are finite polynomials. So each coefficient is a polynomial in $$\ell$$. In a term of order $$\varepsilon^\nu$$ with total curvature dimension $$D=2\nu+D'$$, where $$D'$$ comes from the $$w_k$$, the power of $$\ell$$ is at most $$n-1+2\nu$$, with equality only if $$i=1$$, the other factors of that term contain no $$\varepsilon$$, and $$D'=0$$ (no $$w_k$$ and no volume correction $$V-V_0$$, which carries one). Then (6) gives $$\beta\ge 2-\frac{n-1+2\nu-2\nu-D'}{d}=\alpha_A+D'/d$$, with equality exactly in the case just described. That term is $$[{\rm cap}'(r_0)]_\nu\int_0^\infty d\tau\,\phi(\omega\tau^d/d)\int_0^\tau ds\,\omega s^{n}\,s$$, which by the flat computation equals $$a_dQ(\alpha_A)[{\rm cap}'(r_0)]_\nu$$, and $${\rm cap}'(r_0)=\lvert\partial B\rvert$$ is the exact area of the boundary of the geodesic ball. The values of $$\beta$$ lie on the lattice $$\alpha_A+j/d$$ because $$e$$ and $$D$$ are integers.
</div>

At linear order in the curvature and for curvature that is uniform over the ball two backgrounds suffice. Write $$u$$ for the unit normal of $$\Sigma$$. The static universe ($$W=1$$) has scalar curvature of the slice $$R^\Sigma=n(n-1)/A^2$$ and $$R_{uu}=0$$; de Sitter space in conformal time has the same slice and $$R_{uu}=-n/A^2$$. We define the response at fixed volume by

$$
\delta\langle H_w\rangle\big|_V=-h_\Sigma(\ell)\,R^\Sigma-h_0(\ell)\,R_{uu}, \tag{10}
$$

where $$h_\Sigma$$ and $$h_0$$ are determined by the two backgrounds. We assume that for any perturbation whose curvature is uniform over the ball and whose slice has vanishing extrinsic curvature at the centre, the linear response is a rotation-invariant linear function of the Riemann tensor at the centre. The only such scalars are $$R^\Sigma$$ and $$R_{uu}$$, because the Weyl tensor enters through its electric part, which is traceless and averages to zero over directions. This assumption is used only to say that (10) describes all uniform curvatures; all statements about $$h_\Sigma$$ and $$h_0$$ below are about the two backgrounds.

A geodesic sphere of radius $$r$$ has area $$n\omega r^{n-1}\big(1-(n-1)R^\Sigma r^2/(2n(n+2))\big)$$ and encloses volume $$\omega r^n\big(1-R^\Sigma r^2/(6(n+2))\big)$$ to first order. Eliminating $$r$$ at fixed volume gives the area deficit used by Jacobson [[Jacobson]](#ref-jacobson),

$$
\delta\lvert\partial B\rvert\big|_V=-c_d\,\ell^dR^\Sigma,\qquad c_d=\frac{\omega}{2(d+1)}\quad(c_4=2\pi/15). \tag{11}
$$

By Theorem 1 the part of $$h_\Sigma$$ at the area exponent is $$a_dQ(\alpha_A)c_d\ell^d$$. The coefficient $$h_0$$ has no part at the area exponent, because the two backgrounds have the same slice and the same ball.

The remaining coefficients follow from (4) by expanding to first order in $$\varepsilon$$ with $$W=1+\tfrac d2\varepsilon t^2$$ for de Sitter space. The computation is mechanical: each monomial in $$\tau$$ is integrated with (5), a term $$\phi'$$ from the change of $$V$$ giving $$-\Gamma(b)Q(b-1)$$. We did it in exact arithmetic for $$d=3,\dots,12$$. One check applies to the expansion itself: in $$d=4$$ and $$\ell=1$$ we integrated (4) numerically with the exact caps and exact $$W=\cos^{-4}(t/A)$$ at finite $$A$$ and extrapolated in $$A^{-2}$$; the first-order coefficients are $$-4.2464118$$ (static) and $$8.08122$$ (de Sitter), against $$-4.2464119$$ and $$8.081227$$ from the expansion. In four dimensions, with each entry multiplied by $$Q$$ at the exponent in the first column, the result is

| exponent $$\alpha$$ | flat count | $$h_\Sigma$$ | $$h_0$$ |
|---|---|---|---|
| $$3/2$$ | $$\frac{2\sqrt3\pi}{5}\ell^2$$ | $$\frac{\sqrt3\pi}{75}\ell^4$$ | $$0$$ |
| $$7/4$$ | $$\frac{7\cdot3^{3/4}\pi^{1/4}\Gamma(3/4)}{10}\ell$$ | $$\frac{7\cdot3^{3/4}\pi^{1/4}\Gamma(3/4)}{100}\ell^3$$ | $$0$$ |
| $$2$$ | $$\frac{34}{35}$$ | $$\frac{48}{175}\ell^2$$ | $$\frac{256}{175}\ell^2$$ |
| $$9/4$$ | | $$\frac{31\cdot3^{1/4}\Gamma(1/4)}{1120\,\pi^{1/4}}\ell$$ | $$\frac{37\cdot3^{1/4}\Gamma(1/4)}{70\,\pi^{1/4}}\ell$$ |
| $$5/2$$ | | $$\frac{\sqrt3}{1260}$$ | $$\frac{547\sqrt3}{1260}$$ |

and in three dimensions the flat count is $$\frac{3^{2/3}\pi^{1/3}\Gamma(2/3)}{3}Q(\tfrac53)\ell+\frac9{10}Q(2)$$, with $$h_\Sigma=\frac{3^{2/3}\pi^{1/3}\Gamma(2/3)}{48}Q(\tfrac53)\ell^3+\frac9{40}Q(2)\ell^2+\frac{3^{1/3}\Gamma(1/3)}{20\pi^{1/3}}Q(\tfrac73)\ell+\frac{3^{2/3}\Gamma(2/3)}{210\pi^{2/3}}Q(\tfrac83)$$ and $$h_0=\frac{5\cdot3^{1/3}\Gamma(1/3)}{12\pi^{1/3}}Q(\tfrac73)\ell+\frac{64\cdot3^{2/3}\Gamma(2/3)}{105\pi^{2/3}}Q(\tfrac83)$$.

<div class="proposition" markdown="1">
**Proposition 2** (structure of the response, $$3\le d\le12$$). Let $$3\le d\le12$$. With exponents $$\alpha_A+j/d$$ and each coefficient multiplying $$Q$$ at its exponent, the flat count has nonzero coefficients exactly for $$0\le j\le d-2$$, $$h_\Sigma$$ for $$0\le j\le d$$, and $$h_0$$ for $$2\le j\le d$$, and every one of these coefficients is positive. The coefficient of $$h_\Sigma$$ at $$j=0$$ is $$a_dc_d\ell^d$$.
</div>

This is the exact computation described above; the four-dimensional and three-dimensional cases are written out in full, and the coefficients in the other dimensions are the output of the same code. We did not find a closed form for the coefficients as functions of $$d$$ and $$j$$, so the range $$d\le12$$ is the range computed. A convenient consequence for four dimensions is the large-ball expansion of the type-0 count, for $$\lambda=0$$ (defined below):

$$
\frac{G_N}{G}=\frac{1}{1+3.588/\ell+3.781/\ell^2+1.367/\ell^3+0.019/\ell^4}\qquad(\text{type-0 weights}).
$$

## 4. Exact weights

We model the use of the count as in Jacobson's argument [[Jacobson]](#ref-jacobson), with a causal-set count in place of the entanglement entropy of short-distance degrees of freedom. The model has three parts. The entropy of the region is $$s_0\langle H_w\rangle$$ for a constant $$s_0$$. In flat space this defines the Newton constant $$G$$ through $$S=\lvert\partial B\rvert/4G$$ at leading order, so $$1/(4G)=s_0Q(\alpha_A)a_d\,\rho^{(d-2)/d}$$. And entanglement equilibrium holds: at fixed volume the change of the short-distance entropy balances the change of the matter entropy. In Jacobson's argument the balance, imposed for every unit timelike $$u$$, is equivalent to the Einstein equation with coupling $$G_N$$ if the short-distance entropy varies by $$\delta\lvert\partial B\rvert_V/4G_N$$; on a slice with vanishing extrinsic curvature the Gauss equation gives $$R^\Sigma=2G_{uu}$$, so by (11) $$\delta\lvert\partial B\rvert_V=-2c_d\ell^dG_{uu}$$. Comparing with (10), the count reproduces the linearised Einstein equation with $$G_N=G$$ at size $$\ell$$ exactly when

(i) $$h_0(\ell)=0$$, and (ii) $$h_\Sigma(\ell)=Q(\alpha_A)\,a_d\,c_d\,\ell^d$$.

If (i) fails the response depends on $$R_{uu}$$ separately and the equation is not Einstein's. For matter with $$R_{uu}=\lambda R^\Sigma$$ one defines $$G_N/G=Q(\alpha_A)a_dc_d\ell^d/(h_\Sigma+\lambda h_0)$$; in four dimensions the Einstein equation gives $$\lambda=0$$ for a pure slice-curvature channel, $$\lambda=1/4$$ for dust and $$\lambda=1/2$$ for radiation. This is the definition of $$G_N/G$$ used below. It is a model: only the expected count enters, and fluctuations of the count are ignored.

For the type-0 count the large-ball expansion in section 3 shows that both conditions hold up to relative corrections of order $$\ell_\rho/\ell$$, and they fail for balls a few discreteness lengths across. As an illustration take the balls spanned by links. A causal diamond of half-height $$\ell$$ has volume $$2\pi\ell^4/3$$ in four dimensions, a pair is a link when this diamond is empty, and the number of related pairs with half-height in $$d\ell$$ grows as $$\ell^3d\ell$$ per unit boost volume, so the half-heights of links are distributed as $$\ell^3e^{-2\pi\ell^4/3}$$. Averaging the numerator and the denominator of $$G_N/G$$ over this distribution, the type-0 count gives $$G_N/G=0.0832$$, $$0.0292$$ and $$0.0177$$ for $$\lambda=0$$, $$1/4$$ and $$1/2$$. Other populations of sizes give other values.

<div class="theorem" markdown="1">
**Theorem 2** (exact weights). Let $$3\le d\le12$$ and let $$w$$ be a finite weighting with $$Q(\alpha_A)\neq0$$.

(a) If $$Q(\alpha_A+j/d)=0$$ for $$j=1,\dots,d$$, then for every $$\ell>0$$ the flat count equals $$Q(\alpha_A)a_d\lvert\partial B\rvert$$ and conditions (i) and (ii) hold.

(b) Condition (ii) alone implies these $$d$$ zeros.

(c) A weighting with these zeros has $$w_m\neq0$$ for some $$m\ge d$$. Among weightings with $$w_m=0$$ for $$m>d$$ it is unique up to normalisation, and

$$
Q_d(\alpha)=\binom{2d+2-d\alpha}{d},\qquad w_m=\sum_{j=0}^{m}(-1)^j\binom mj\binom{2d+2+dj}{d}. \tag{12}
$$

</div>

<div class="proof" markdown="1">
*Proof.* (a) By Proposition 2 the flat count, $$h_\Sigma$$ and $$h_0$$ are sums over $$j\le d$$ of coefficients times $$Q(\alpha_A+j/d)$$. The zeros leave the $$j=0$$ terms: $$Q(\alpha_A)a_d\lvert\partial B\rvert$$, $$Q(\alpha_A)a_dc_d\ell^d$$ and zero. (b) By Proposition 2 the function $$h_\Sigma-Q(\alpha_A)a_dc_d\ell^d=\sum_{j=1}^dC^\Sigma_jQ(\alpha_A+j/d)\ell^{d-j}$$ has all $$C^\Sigma_j\neq0$$; it vanishes identically in $$\ell$$ only if each $$Q(\alpha_A+j/d)$$ vanishes. (c) A polynomial of degree $$N$$ with $$d$$ prescribed zeros that is not identically zero has $$N\ge d$$, and for $$N=d$$ it is fixed up to a constant by its zeros. With $$z=2d+2-d\alpha$$, $$\binom zd=z(z-1)\cdots(z-d+1)/d!$$ vanishes at $$z=0,\dots,d-1$$, which are the points $$\alpha=\alpha_A+j/d$$, $$j=d,\dots,1$$, and equals $$1$$ at $$z=d$$, which is $$\alpha=\alpha_A$$. The weights follow from (3), since $$Q_d(-j)=\binom{2d+2+dj}{d}$$.
</div>

Part (a) is routine given Proposition 2; the content of the theorem is that the same $$d$$ zeros remove every non-area term in all three quantities. The weights are integers, with $$w_0=\binom{2d+2}{d}$$ and $$w_d=(-d)^d$$:

| $$d$$ | zeros of $$Q_d$$ | weights $$w_0,\dots,w_d$$ | $$a_d$$ |
|---|---|---|---|
| 3 | $$2,\ \tfrac73,\ \tfrac83$$ | $$56,\ -109,\ 90,\ -27$$ | $$0.218853$$ |
| 4 | $$\tfrac74,\ 2,\ \tfrac94,\ \tfrac52$$ | $$210,\ -791,\ 1268,\ -928,\ 256$$ | $$0.173205$$ |
| 5 | $$\tfrac85,\ \tfrac95,\ 2,\ \tfrac{11}5,\ \tfrac{12}5$$ | $$792,\ -5396,\ 14750,\ -19500,\ 12500,\ -3125$$ | $$0.148656$$ |
| 6 | $$\tfrac32,\dots,\tfrac73$$ | $$3003,\ -35757,\ 155713,\ -328779,\ 365256,\ -206064,\ 46656$$ | $$0.133258$$ |

For $$d=7$$ and $$8$$ the weights are $$(11440,-233717,1556926,-4912103,8384292,-7983325,4000066,-823543)$$ and $$(43758,-1518517,15075412,-68204640,167489792,-238911488,198377472,-89128960,16777216)$$. We verified in exact arithmetic that $$\sum_mw_m(\alpha)_m/m!=Q_d(\alpha)$$ for these weights and that the flat count, $$h_\Sigma$$ and $$h_0$$ take the values of Theorem 2(a) in every dimension $$3\le d\le12$$. For $$K>d$$ there is a family of dimension $$K-d$$; "unique" means unique at minimal degree. For example in $$d=4$$ with $$K=5$$ the solutions are the minimal one plus a multiple of $$Q_4(\alpha)(\alpha-\tfrac32)$$.

A weighting that keeps one of the non-area exponents fails. In four dimensions, keeping $$\alpha=2$$ with $$Q(2)=q\neq0$$ and the other three zeros gives $$h_0=\tfrac{256}{175}q\,\ell^2\neq0$$.

<div class="remark" markdown="1">
**Remark** (relation to the causal-set d'Alembertian). The four-dimensional d'Alembertian of Benincasa and Dowker [[BD]](#ref-bd) has the same kind of kernel, a polynomial times $$e^{-v}$$, with layer coefficients $$(1,-9,16,-8)$$; by (2) its symbol is proportional to $$(\alpha-\tfrac12)(\alpha-1)(\alpha-\tfrac32)$$. Its zeros are spaced by $$2/d$$ and lie below the exponents that are kept; they make the continuum limit the local operator. The zeros of $$Q_d$$ are spaced by $$1/d$$ and lie above the exponent that is kept; they remove finite-density corrections to a limit that is already local.
</div>

## 5. No non-negative weighting

Theorem 2 asks $$Q$$ to vanish at positive points. A non-negative kernel cannot do that.

<div class="theorem" markdown="1">
**Theorem 3** (no non-negative weighting). Let $$\phi\ge0$$ be a measurable function on $$(0,\infty)$$, not almost everywhere zero, with convergent Mellin integrals in the range used; in particular any non-negative weights $$w_m$$ not all zero. Then $$Q(\alpha)>0$$ for all $$\alpha>0$$, and no zero required in Theorem 2 can occur. For $$3\le d\le12$$, moreover, $$h_0(\ell)>0$$ and $$h_\Sigma(\ell)>Q(\alpha_A)a_dc_d\ell^d$$ for every $$\ell>0$$, so that $$G_N<G$$ for every $$\lambda\ge0$$, at every ball size, and for every population of sizes with non-negative weight.
</div>

<div class="proof" markdown="1">
*Proof.* The first statement is (2): $$\Gamma(\alpha)Q(\alpha)=\int\phi\,v^{\alpha-1}dv>0$$. The second follows because $$h_0$$ and $$h_\Sigma-Q(\alpha_A)a_dc_d\ell^d$$ are sums of positive coefficients (Proposition 2) times positive values $$Q(\alpha_A+j/d)$$. The ratio $$G_N/G$$ for a population is the ratio of two averages, and each is bounded by the corresponding bound pointwise.
</div>

The positivity of $$Q$$ is routine; the content is Proposition 2, which says that every non-area term contributes with the same sign. Theorem 3 covers all kernels of the form (1) with $$\phi\ge0$$, not only finite weightings, and it covers positive mixtures of dilates of such kernels, since $$M[\phi(c\,\cdot)](\alpha)=c^{-\alpha}M[\phi](\alpha)$$. It does not cover substrates with signed correlations, where the expectation (1) is not an integral against a non-negative measure.

An entropy that counts molecules with a positive amount each therefore gives a weaker gravitational coupling than its own area law at finite density, for $$d\le12$$ and $$\lambda\ge0$$. The number of sign changes of the kernel is bounded below by the number of zeros of $$Q$$. The following lemma is classical.

<div class="lemma" markdown="1">
**Lemma 3** (sign changes). Let $$\phi$$ be continuous on $$(0,\infty)$$ with finitely many sign changes, and suppose $$\int_0^\infty\phi(v)v^{\alpha_k-1}dv=0$$, with absolute convergence, for $$Z$$ distinct real $$\alpha_k$$. Then $$\phi\equiv0$$ or $$\phi$$ changes sign at least $$Z$$ times.
</div>

<div class="proof" markdown="1">
*Proof.* Suppose $$\phi$$ changes sign exactly at $$v_1<\dots<v_r$$ with $$r<Z$$. A nonzero combination $$\sum_{k=1}^{r+1}c_kv^{\alpha_k-1}$$ has at most $$r$$ positive zeros counted with multiplicity (Descartes' rule for generalised polynomials), and the $$r$$ linear conditions that it vanish at the $$v_i$$ have a nonzero solution. That solution changes sign exactly at the $$v_i$$, so its product with $$\phi$$ has one sign, and its integral against $$\phi$$ is zero by hypothesis, which forces $$\phi\equiv0$$.
</div>

<div class="corollary" markdown="1">
**Corollary 1** (alternation). If $$w_m=0$$ for $$m>N$$ and $$Q$$ vanishes at $$N$$ positive points, then $$w_0,\dots,w_N$$ are all nonzero and strictly alternate in sign. In particular the weights (12) alternate.
</div>

<div class="proof" markdown="1">
*Proof.* By Lemma 3 the function $$\phi=e^{-v}p(v)$$, $$p(v)=\sum w_mv^m/m!$$, changes sign at least $$N$$ times, so $$p$$ has at least $$N$$ positive zeros. By Descartes' rule its $$N+1$$ coefficients then have at least $$N$$ sign changes, which forces all of them to be nonzero with alternating signs.
</div>

For $$d=4$$, $$\phi(v)=e^{-v}\big(210-791v+634v^2-\tfrac{464}{3}v^3+\tfrac{32}{3}v^4\big)$$ changes sign at $$v=0.361$$, $$1.583$$, $$4.052$$ and $$8.504$$, and ranges between $$-41.8$$ and $$209$$. The area term is the moment $$\int\phi\,v^{1/2}dv/\Gamma(\tfrac32)=1$$, while $$\int\lvert\phi\rvert v^{1/2}dv/\Gamma(\tfrac32)=114.9$$. For the symbols $$\binom{N+6-4\alpha}{N}$$ that cancel $$N$$ exponents in four dimensions, the same ratio is $$929$$, $$7.2\times10^3$$ and $$4.0\times10^5$$ for $$N=6,8,12$$, about a factor $$2.8$$ per added zero. The weights themselves grow at least geometrically: from $$Q(-j)=\sum_mw_m(-1)^m\binom jm$$ and $$\binom jm\le2^j$$ one gets $$\sum_m\lvert w_m\rvert\ge Q(-j)/2^j$$ for every $$j\ge0$$, and $$j=N$$ gives $$\sum_m\lvert w_m\rvert\ge\binom{5N+6}{N}2^{-N}$$, which grows like $$6.1^N$$. The actual values for $$N=4,6,8,12$$ are $$3453$$, $$1.4\times10^5$$, $$5.4\times10^6$$ and $$8.4\times10^9$$, against the bound $$934$$, $$3.0\times10^4$$, $$1.0\times10^6$$ and $$1.2\times10^9$$. Poisson fluctuations of the signed count are amplified by the same cancellation; the variance is not treated here.

## 6. The Einstein form and the Newton constant are automatic

Theorem 1 has a consequence that bears on how the exact weights should be read.

<div class="proposition" markdown="1">
**Proposition 3** (automatic Einstein form and $$G_N=G$$). Let $$\phi$$ be any kernel for which the moments in (2) converge, and suppose $$Q$$ vanishes at every exponent $$\alpha_A+j/d$$, $$j\ge1$$, that occurs in the flat count, $$h_\Sigma$$ and $$h_0$$ of a given background class, and $$Q(\alpha_A)\neq0$$. Then in that class the weighted flat count is $$Q(\alpha_A)a_d\lvert\partial B\rvert$$, the weighted response is $$Q(\alpha_A)a_d\,\delta\lvert\partial B\rvert_V$$, hence $$h_0\equiv0$$ and $$G_N/G\equiv1$$.
</div>

<div class="proof" markdown="1">
*Proof.* By Theorem 1 the terms at the area exponent sum to $$a_dQ(\alpha_A)$$ times the exact area of $$\partial B$$, and every other term carries $$Q$$ at one of the zeros. The static universe and de Sitter space have the same slice and the same ball, so the area-exponent part of the response is the same in both, and equals $$a_dQ(\alpha_A)\,\delta\lvert\partial B\rvert_V$$. Hence $$h_0$$ has no surviving part, and $$h_\Sigma=a_dQ(\alpha_A)c_d\ell^d$$.
</div>

Whatever principle selects the weights, a weighting that removes every non-area exponent gives the Einstein form with $$G_N=G$$. Equality of the two Newton constants therefore does not test a principle for choosing weights; it tests the property of the counting object stated in Theorem 1, that the term at the area exponent is the area of the curved ball. Counting objects that do not have this property can fail it; the molecules defined by a null hypersurface through the horizon in [[Machet–Wang]](#ref-mw) are an example in the literature. The only way a weighting can fail is to keep a non-area exponent. The same argument shows that the weights that remove non-area exponents are the Richardson extrapolation described next.

## 7. The signed count as an extrapolation of positive counts

Let $$C_p$$ be the type-0 count of the $$p$$-thinned sprinkling, in which each element is kept independently with probability $$p$$. It is a positive count.

<div class="proposition" markdown="1">
**Proposition 4** (Richardson representation). (i) For every sprinkling $$\eta$$, $$\mathbb E[C_p\mid\eta]=p^2\sum_m(1-p)^mH_m(\eta)$$. (ii) Suppose the expected molecule numbers have the form $$\langle H_m\rangle=\sum_{j=0}^dA_j\frac{(\alpha_j)_m}{m!}$$ with $$\alpha_j=\alpha_A+j/d$$; by Lemma 2 this is the form of the flat count and of every term of the curvature response at first order, with $$A_j$$ the coefficient of $$Q(\alpha_j)$$. Then $$\mathbb E\,C_p=\sum_jA_jp^{2-\alpha_j}$$. For distinct $$p_0,\dots,p_d\in(0,1]$$ the system $$\sum_ic_ip_i^{2-\alpha_j}=\delta_{j0}$$, $$j=0,\dots,d$$, has a unique solution, and
$$
\mathbb E\,H_{Q_d}=\sum_{i=0}^dc_i\,\mathbb E\,C_{p_i}=A_0.
$$

</div>

<div class="proof" markdown="1">
*Proof.* (i) A pair $$x\prec y$$ with $$x$$ inside and $$y$$ outside survives the thinning with probability $$p^2$$, and it is a type-0 molecule of the thinned sprinkling exactly when the other $$m$$ elements of $$F(x)$$ are all deleted, which has probability $$(1-p)^m$$ independently of the rest. (ii) The binomial series gives $$\sum_m(1-p)^m(\alpha)_m/m!=p^{-\alpha}$$, so $$\mathbb E\,C_p=p^2\sum_m(1-p)^m\langle H_m\rangle=\sum_jA_jp^{2-\alpha_j}$$. The matrix $$p_i^{2-\alpha_j}=p_i^{1-2/d}\,(p_i^{-1/d})^j$$ is a row-scaled Vandermonde matrix and is nonsingular. Finally $$\mathbb E\,H_{Q_d}=\sum_jA_jQ_d(\alpha_j)=A_0$$.
</div>

So the expected signed count equals the expected value of a combination of positive counts of thinned sprinklings. The mean of $$H_{Q_d}$$ is a Richardson extrapolation of the type-0 count to infinite density at fixed $$\ell$$, performed at one density by thinning. Moreover, the coefficients $$c_i$$ cannot all be non-negative: $$\sum_ic_ip_i^{2-\alpha_j}=0$$ for $$j\ge1$$ with positive terms $$p_i^{2-\alpha_j}$$ would force $$c=0$$. This is Theorem 3 seen from the other side.

The identity is about means. Three quantities with the same mean are available: $$H_{Q_d}(\eta)$$ itself, the Rao–Blackwellised representation $$\sum_i c_i\,\mathbb E[C_{p_i}\mid\eta]=\sum_m\tilde w_mH_m(\eta)$$ with $$\tilde w_m=\sum_ic_ip_i^2(1-p_i)^m$$, and the combination $$\sum_ic_iC_{p_i}(\eta_{p_i})$$ of independently thinned copies. Each takes both signs, by Theorem 3, and they have different distributions on a given sprinkling. We checked the identity (i) and the exact mean in four dimensions with $$\ell=1$$ and $$p=1/2$$: the mean type-0 count of the thinned process was $$4.656\pm0.047$$ and the mean of $$p^2\sum_m(1-p)^mH_m$$ was $$4.697\pm0.021$$, against $$4.700$$ from $$\sum_jA_jp^{2-\alpha_j}$$ with the flat coefficients of (9).

A statement of the form "the weighted count is negative on a fraction $$f$$ of sprinklings, so it is not the entropy of any state" is a statement about one representation. It does not follow from the mean, and the fraction changes between equal-mean representations. We do not use any such statement here. The statements that concern the mean functional are Theorems 2 and 3 and Corollary 1, together with Proposition 5(d) below, which says that the number of sign changes of the kernel is invariant under thinning and at least the number of zeros of $$Q$$.

## 8. Thinning

Thinning acts on a weighting through a matrix, and its action on the Mellin symbol is a rescaling. Let $$(B_pw)(m)=\sum_{k}\binom mkp^k(1-p)^{m-k}w_k$$ be the weights of a molecule of type $$m$$ after each of its $$m$$ extra elements is kept independently with probability $$p$$, so that for a given sprinkling $$\mathbb E[H_w(\eta_p)\mid\eta]=p^2\sum_{\text{molecules}}(B_pw)(m)$$.

<div class="proposition" markdown="1">
**Proposition 5** (thinning duality and sign class). For a finite weighting $$w$$ and $$p\in(0,1]$$:

(a) $$\sum_m(B_pw)(m)\dfrac{(\alpha)_m}{m!}=p^{-\alpha}Q_w(\alpha)$$.

(b) The kernel of $$B_pw$$ is $$\phi_{B_pw}(v)=\phi_w(pv)$$.

(c) If $$B_pw\ge0$$ for some $$p\in(0,1]$$ then $$\phi_w\ge0$$ and $$Q_w$$ has no positive zero. Consequently, if $$Q_w$$ has a positive zero, as every weighting of Theorem 2 does, then $$B_pw$$ takes a negative value for every $$p\in(0,1]$$.

(d) The number of sign changes of the kernel $$\phi_{B_pw}$$ is independent of $$p$$, and by Lemma 3 it is at least the number of positive zeros of $$Q_w$$.
</div>

<div class="proof" markdown="1">
*Proof.* (a) Write $$\binom mk\frac{(\alpha)_m}{m!}=\frac{(\alpha)_k}{k!}\frac{(\alpha+k)_{m-k}}{(m-k)!}$$. Summing over $$m$$ first, $$\sum_{j\ge0}\frac{(\alpha+k)_j}{j!}(1-p)^j=p^{-\alpha-k}$$, so the sum is $$\sum_kw_kp^k\frac{(\alpha)_k}{k!}p^{-\alpha-k}=p^{-\alpha}Q_w(\alpha)$$. (b) $$e^{-\lambda}\sum_m(B_pw)(m)\lambda^m/m!=e^{-\lambda}e^{(1-p)\lambda}\sum_kw_k(p\lambda)^k/k!=\phi_w(p\lambda)$$. (c) If $$B_pw\ge0$$ the left side of (b) is non-negative, so $$\phi_w(pv)\ge0$$ for all $$v$$, hence $$\phi_w\ge0$$, and Theorem 3 applies. (d) follows from (b), since $$v\mapsto pv$$ is increasing.
</div>

Part (a) says that the thinned weights $$p^{\alpha_A}B_pw$$, which have infinite support, have symbol $$p^{\alpha_A-\alpha}Q_w(\alpha)$$: the same zeros and the same value at the area exponent. They are therefore again exact weightings in the sense of Theorem 2(a), now with a kernel $$\phi_w(pv)$$ that has the same number of sign changes. Their counts are related to the thinned sprinkling by $$H_{p^{\alpha_A}B_pw}(\eta)=p^{\alpha_A-2}\,\mathbb E[H_w(\eta_p)\mid\eta]$$. Averaging an exact weighting over thinnings, that is, coarse graining the substrate by thinning, therefore produces another exact weighting and never a non-negative one (part (c)). This is the sense in which the sign structure survives coarse graining by thinning.

<div class="proposition" markdown="1">
**Proposition 6** (no exact finite-order homogeneity). Let $$F$$ be an isomorphism-invariant real function on finite marked posets $$(P,D)$$, with $$D$$ a down-set, and $$F(\emptyset)=0$$. If $$\mathbb E\,F(P_p)=p^\gamma F(P)$$ for all finite $$(P,D)$$ and all $$p\in(0,1]$$, where $$P_p$$ is the $$p$$-thinning, then either $$\gamma$$ is a positive integer, or $$F\equiv0$$. In particular, for $$d\ge2$$ the area degree $$\gamma=(d-2)/d\in[0,1)$$ is not a positive integer, and no nonzero functional of finite marked posets is exactly homogeneous of that degree.
</div>

<div class="proof" markdown="1">
*Proof.* The sub-poset kept by the thinning is a subset $$S\subseteq P$$ with probability $$p^{\lvert S\rvert}(1-p)^{\lvert P\rvert-\lvert S\rvert}$$, so $$\mathbb E\,F(P_p)=\sum_{S\subseteq P}p^{\lvert S\rvert}(1-p)^{\lvert P\rvert-\lvert S\rvert}F(S)$$ is a polynomial in $$p$$ that tends to $$F(\emptyset)=0$$ as $$p\to0$$. If $$\gamma$$ is not an integer, $$p^\gamma F(P)$$ is a polynomial on $$(0,1]$$ only if $$F(P)=0$$. If $$\gamma=0$$, the polynomial equals the constant $$F(P)$$ and its limit at $$p=0$$ gives $$F(P)=0$$.
</div>

The statement is about exact homogeneity of a functional of a finite order. It says nothing about the asymptotic degree of the count on large sprinklings, or about its law on an unbounded domain, which is where the area law lives: in Lemma 1 the integral over the depth $$\tau$$ is over the unbounded past of the horizon. For finite orders the count of a fixed marked poset is a polynomial in $$p$$ and the area degree is an asymptotic property.

Thinning also acts at the level of single orders. For a marked poset $$(P,D)$$ let $$H_w(P,D)=\sum_{x\in D}n_O(x)\,w_{m(x)}$$, where $$n_O(x)$$ is the number of elements of $$P\setminus D$$ above $$x$$ and $$m(x)$$ is the number of elements of $$P$$ above $$x$$ minus one. On a three-element order the value of $$H_{Q_d}$$ is one of $$2w_1$$, $$w_1$$ or $$w_0+w_1$$, according to the order. With $$w_1=\binom{2d+2}d-\binom{3d+2}d$$ and $$\binom{3d+2}{d}/\binom{2d+2}{d}>2$$ for $$d\ge3$$ (each of the $$d$$ factors of the ratio is at least $$(3d+2)/(2d+2)\ge11/8$$, and $$(11/8)^3>2$$), all three values are negative for $$d\ge3$$. For $$d=2$$ the weights are $$(15,-13,4)$$ and $$w_0+w_1=2>0$$. So in $$d\ge3$$ the signed count is negative on some three-element marked order; the sign of the count on a datum is not controlled by the sign of the mean.

## 9. What the theorems do not exclude

The statements above concern expected counts of Poisson sprinklings at linear order in uniform curvature, for the sheet that opens into the past. The following are not excluded.

1. Signed weights. They exist (Theorem 2) and give the Einstein form with $$G_N=G$$ at every ball size at this order. Their mean is a Richardson extrapolation of positive counts of thinned sprinklings (Proposition 4). Whether a signed count can be an entropy is a question about the interpretation of the count, which the theorems do not decide.
2. Large balls. For $$\ell\gg\ell_\rho$$ a non-negative weighting satisfies conditions (i) and (ii) up to relative corrections of order $$\ell_\rho/\ell$$ (section 3), and $$G_N/G\to1$$.
3. Substrates with signed correlations. Theorem 3 uses that the expectation is an integral against a non-negative measure. A non-Poisson substrate with repulsive correlations is outside it.
4. Gradients of the curvature. Theorem 2 concerns curvature uniform over the ball. In the conformally static class with $$W=\exp(d\,\varepsilon_3t^3)$$ the slice $$t=0$$ is flat with vanishing extrinsic curvature and the scalar curvature vanishes on it, while its time derivative does not. At first order in $$\varepsilon_3$$ the area of $$\partial B$$ does not change, and the $$Q_d$$-weighted count does not vanish: its coefficient is $$2511/(175\pi)\approx4.567$$ for $$d=3$$ and $$-3377\cdot3^{3/4}\Gamma(\tfrac34)/(700\,\pi^{3/4})\approx-5.711$$ for $$d=4$$ (computed here by the method of section 3). The weights $$Q=\binom{2d+4-d\alpha}{d+2}$$, which remove the exponents up to $$\alpha_A+(d+2)/d$$, are $$(252,-1035,2046,-2133,1134,-243)$$ for $$d=3$$ and $$(924,-7084,23668,-41416,39808,-19968,4096)$$ for $$d=4$$, and with them this response vanishes in both dimensions. Each order in the gradients needs new weights; the consequences for exactness to all orders are in [Exact causal-set horizon weights must be non-local](/quantum-gravity/nonlocal-weights/).
5. The other null sheet. The second null sheet through $$\partial B$$ below $$\Sigma$$ is the lower boundary of the domain of dependence, which contracts into the past and has a caustic. The computation of section 2 does not apply to it as written, and the weights of Theorem 2 are not exact for small balls on that sheet (a numerical observation that is not repeated here). The theorems are about the caustic-free sheet.
6. Non-linear Einstein equations. Jacobson's argument uses balls in slices whose extrinsic curvature vanishes at the centre. In an expanding background these are not slices $$t=$$ const, and Lemma 1 does not reach them. The non-linear statement requires the term of the count at the area exponent to be $$a_d$$ times the area of $$\partial B$$ for such slices. This has not been verified, and the first law for the matter, the conformal Killing vector of the diamond and the vacuum state remain Jacobson's assumptions.
7. Weyl and transverse-traceless sources, and other molecule definitions. Only the conformally static class and the two scalar responses were computed.

## 10. Open points

The variance of the signed count is not computed here, and the cancellation described in section 5 suggests that it is large. The coefficients of Proposition 2 are computed for $$3\le d\le12$$ and no closed form in $$(d,j)$$ is known to us, so positivity for all $$d$$ is open. Whether the first correction to the area law, of order $$\ell_\rho/\ell$$ in the type-0 count, can be used to read off the extrinsic geometry of the horizon, as proposed in [[Barton et al.]](#ref-barton), survives for weighted counts is not examined.

## References

1. <a id="ref-barton"></a>Barton, Counsell, Dowker, Gould, Jubb, Taylor, “Horizon molecules in causal set theory,” Phys. Rev. D 100, 126008 (2019), [arXiv:1909.08620](https://arxiv.org/abs/1909.08620).
2. <a id="ref-dou-sorkin"></a>Dou, Sorkin, “Black hole entropy as causal links,” Found. Phys. 33, 279 (2003), [arXiv:gr-qc/0302009](https://arxiv.org/abs/gr-qc/0302009).
3. <a id="ref-jacobson"></a>Jacobson, “Entanglement equilibrium and the Einstein equation,” Phys. Rev. Lett. 116, 201101 (2016), [arXiv:1505.04753](https://arxiv.org/abs/1505.04753).
4. <a id="ref-bd"></a>Benincasa, Dowker, “The scalar curvature of a causal set,” Phys. Rev. Lett. 104, 181301 (2010), [arXiv:1001.2725](https://arxiv.org/abs/1001.2725).
5. <a id="ref-dg"></a>Dowker, Glaser, “Causal set d'Alembertians for various dimensions,” Class. Quantum Grav. 30, 195016 (2013), [arXiv:1305.2588](https://arxiv.org/abs/1305.2588).
6. <a id="ref-glaser"></a>Glaser, “A closed form expression for the causal set d'Alembertian,” [arXiv:1311.1701](https://arxiv.org/abs/1311.1701).
7. <a id="ref-ass"></a>Aslanbeigi, Saravani, Sorkin, “Generalized causal set d'Alembertians,” JHEP 1406, 024 (2014), [arXiv:1403.1622](https://arxiv.org/abs/1403.1622).
8. <a id="ref-mw"></a>Machet, Wang, “On the horizon entropy of a causal set,” [arXiv:2012.06212](https://arxiv.org/abs/2012.06212).
9. <a id="ref-last"></a>Last, “Stochastic analysis for Poisson processes,” [arXiv:1405.4416](https://arxiv.org/abs/1405.4416) (the Mecke equation is eq. (1.7) of that survey).
