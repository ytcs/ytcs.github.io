---
title: The four-dimensional causal-set d'Alembertian has no positive spectral representation
date: 2026-10-29
category: gravity
order: 5
description: "For the mean-field causal-set d'Alembertian in four dimensions, with a retarded Lorentz-invariant kernel bounded by a power of proper time, the Euclidean propagator is not a positive superposition of free propagators, and in two and four dimensions the same holds for such kernels whose symbol agrees with the continuum one to all orders at low momentum."
status: "Proved for the mean-field operator of a sprinkling: the four-dimensional theorem for polynomially bounded kernels, its matrix-valued extension under an assumed form of the mean kernel, the low-momentum obstruction, the tail-exponent threshold, the existence of flat operators for exponents below one half, the light-cone bound, and non-negativity of the two-dimensional density. Computed, not proved: the negative density, complex zero pair and sum rule of the four-dimensional Benincasa–Dowker operator, and the Stieltjes form in two dimensions."
excludes: "With a Lorentz-invariant kernel bounded by a power of proper time: a positive spectral representation of the propagator in four dimensions (also nonzero matrix kernels of the assumed form with property (Q)), and in two and four dimensions when the symbol agrees with the continuum through fourth order in momentum. Not covered: light-cone terms, faster growth, Lorentz violation, other unitarity readings."
---

We ask whether the free field defined by the d'Alembertian of a causal set has a positive Källén–Lehmann spectral representation, that is, whether its Euclidean propagator $$D(Z)$$, with $$Z$$ the squared momentum, is a positive superposition of free propagators (the Stieltjes form (4) below). For the mean-field operator of a Poisson sprinkling (section 1), a retarded Lorentz-invariant operator whose kernel depends on proper time, the answer in four dimensions is no whenever the kernel is bounded by a power of the proper time, which includes layer weights that are finite in number or polynomially bounded (Theorem 8, Proposition 9). A positive representation forces $$D$$ to approach its limit at large $$Z$$ no faster than $$1/Z$$, and such a kernel makes it approach like $$Z^{-2}$$. Positive weight is the standard criterion for the absence of ghosts. It differs from unitarity of a retarded non-local theory, and section 6 names the readings of the latter that remain open.

Two further results concern low momentum. In two and four dimensions a kernel bounded by a power whose symbol agrees with $$-Z$$ at low momentum up to terms smaller than $$Z^2$$ has no positive representation either (Corollary 3), and a kernel whose symbol agrees with $$-Z$$ to all orders cannot be bounded by $$e^{-\lambda\tau^\beta}$$ with $$\beta>\tfrac12$$ in the proper time $$\tau$$ (Theorem 6); such kernels exist for every $$\beta<\tfrac12$$ (Theorem 7). Theorem 11 extends Theorem 8 to matrix-valued kernels, which cover non-commutative points when their mean kernel has the form assumed in section 5, and section 4 gives computed spectral data of the four-dimensional Benincasa–Dowker operator.

The operator in two and four dimensions is due to Sorkin [[Sorkin]](#ref-sorkin) and to Benincasa and Dowker [[BD]](#ref-bd), and Dowker and Glaser [[Dowker–Glaser]](#ref-dg) gave it in other dimensions. Aslanbeigi, Saravani and Sorkin [[ASS]](#ref-ass) generalised it to a family with free layer weights and imposed the low-momentum conditions order by order. They found that the four-dimensional Benincasa–Dowker symbol has a pair of complex zeros, which signals a classical instability, that the two-dimensional symbol does not, and that a numerical search did not find a choice of weights in the family that removed the pair. Belenchia, Benincasa and Liberati [[BBL]](#ref-bbl) found that the discontinuity of the two-dimensional propagator across its cut is positive and the four-dimensional one is not, and asked whether the better ultraviolet behaviour in four dimensions is related. Theorem 8 is the new result: the faster ultraviolet decay is incompatible with positive weight for every member of the family with finitely many or polynomially bounded layer weights.

## 1. Causal sets and the mean-field operator

A *causal set* is a set with a partial order $$\prec$$ in which every interval $$I(x,y)=\{z:x\prec z\prec y\}$$ is finite [[BLMS]](#ref-blms). A *sprinkling* into Minkowski space $$\mathbb R^{1,d-1}$$ draws points from a Poisson process of density $$\ell^{-d}$$ and orders them by the causal order of the spacetime, so that the number of points in a region of volume $$V_0$$ is Poisson distributed with mean $$V_0\ell^{-d}$$. A sprinkling has no preferred frame or lattice in distribution, which is why it is the standard model of a Lorentz-invariant discrete structure [[Surya]](#ref-surya).

The Benincasa–Dowker operators act on a function $$\phi$$ on the causal set by

$$
(B\phi)(x)=\ell^{-2}\Big[a\,\phi(x)+\sum_{k\ge0}c_k\!\!\sum_{y\prec x,\ \lvert I(y,x)\rvert=k}\!\!\phi(y)\Big], \tag{1}
$$

the inner sum running over the elements $$y$$ to the past of $$x$$ with exactly $$k$$ elements between $$y$$ and $$x$$ (the $$k$$-th layer of $$x$$). In two dimensions $$(a;c_0,c_1,c_2)=2(-1;2,-4,2)$$ and in four dimensions $$(a;c_0,\dots,c_3)=\tfrac4{\sqrt6}(-1;1,-9,16,-8)$$; these coefficients make the expectation of $$B\phi$$ over sprinklings tend to $$\Box\phi$$ for smooth $$\phi$$ as the density grows [[BD]](#ref-bd) [[Dowker–Glaser]](#ref-dg). The signature is $$(-,+,+,+)$$, so that $$\Box=-\partial_t^2+\nabla^2$$, and the four-dimensional coefficients are those of the Remark in section 5 of [horizon counts and the area law](/quantum-gravity/area-projector/). The *generalised family* of [[ASS]](#ref-ass) allows finitely many arbitrary coefficients subject to conditions at low momentum. We set $$\ell=1$$ from here on.

Given $$x$$, the other points form a Poisson process (the Mecke formula), so the probability that a point at $$y$$ has exactly $$k$$ points between it and $$x$$ is $$e^{-V}V^k/k!$$, with $$V=V(y,x)$$ the volume of the interval. The expectation of (1) is the *mean-field operator*

$$
(\bar B\phi)(x)=a\,\phi(x)+\int_{y\prec x}F(\tau(x,y))\,\phi(y)\,d^dy,\qquad F(\tau)=\sum_kc_k\,e^{-V}\frac{V^k}{k!}, \tag{2}
$$

where $$\tau$$ is the proper time from $$y$$ to $$x$$ and $$V=\tau^2/2$$ in two dimensions, $$V=\pi\tau^4/24$$ in four. This is a retarded, Lorentz-invariant operator with a contact term $$a\phi(x)$$ and a kernel $$F$$ that depends on $$\tau$$ only. Everything below concerns operators of the form (2) with more general $$F$$; the sprinkling enters only through this form.

Operators of the form (2) are diagonalised by exponentials in time. For $$\phi(t,\vec x)=e^{Et}$$ with $$E>0$$ the continuum operator gives $$\Box\phi=-E^2\phi$$, so it has *symbol* $$-E^2=-Z$$ at the Euclidean point $$Z=E^2$$.

<div class="lemma" markdown="1">
**Lemma 1** (symbol). Let $$F$$ be measurable with $$\lvert F(\tau)\rvert\le C(1+\tau)^p$$. Then $$\bar B e^{Et}=b(E)e^{Et}$$ for $$E>0$$, with

$$
b(E)=a+\Pi(E),\qquad \Pi(E)=\int_0^\infty F(\tau)\,\tau^{d-1}k_d(E\tau)\,d\tau,\qquad k_2(z)=2K_0(z),\quad k_4(z)=\frac{4\pi K_1(z)}{z}, \tag{3}
$$

where $$K_0,K_1$$ are modified Bessel functions. In any dimension $$k_d(z)=\Omega_{d-2}\int_0^\infty\sinh^{d-2}\eta\,e^{-z\cosh\eta}\,d\eta>0$$, with $$\Omega_{d-2}$$ the area of the unit $$(d-2)$$-sphere ($$\Omega_0=2$$). The function $$b$$ extends holomorphically to $$\operatorname{Re}E>0$$, that is to $$Z=E^2\in\mathbb C\setminus(-\infty,0]$$. For $$d\in\{2,4\}$$ and $$E\ge1$$, $$\lvert\Pi(E)\rvert\le C'E^{-d}$$, so $$b$$ is bounded on $$Z\ge1$$ and $$b\to a$$ as $$Z\to\infty$$.
</div>

<div class="proof" markdown="1">
*Proof.* Writing $$y=x-s$$ with $$s$$ in the future cone, $$\bar Be^{Et}=e^{Et}\big(a+\int_{J^+}F(\tau(s))e^{-Es^0}d^ds\big)$$, and the integral converges absolutely because of the factor $$e^{-Es^0}$$. In four dimensions it equals $$\int_0^\infty dt\,e^{-Et}\int_0^t4\pi r^2F(\sqrt{t^2-r^2})dr$$. At fixed $$t$$ substitute $$\tau=\sqrt{t^2-r^2}$$, so $$r\,dr=-\tau\,d\tau$$ and $$r=\sqrt{t^2-\tau^2}$$, and exchange the order of integration: the integral is $$\int_0^\infty4\pi\tau F(\tau)\,d\tau\int_\tau^\infty e^{-Et}\sqrt{t^2-\tau^2}\,dt$$. The inner integral is $$\tau K_1(E\tau)/E$$, which gives (3). In two dimensions the same steps give $$2\tau K_0$$ with the inner integral $$\int_\tau^\infty e^{-Et}(t^2-\tau^2)^{-1/2}dt=K_0(E\tau)$$. The general formula follows from $$t=\tau\cosh\eta$$, $$r=\tau\sinh\eta$$, with Jacobian $$\tau$$. The Bessel functions decay like $$e^{-\operatorname{Re}(E\tau)}$$, which gives the analytic continuation. For the bound, substitute $$\tau=s/E$$: $$\Pi(E)=E^{-d}\int_0^\infty F(s/E)s^{d-1}k_d(s)\,ds$$, and for $$E\ge1$$ the integrand is at most $$C(1+s)^ps^{d-1}k_d(s)$$, which is integrable because $$s^{d-1}k_d(s)$$ is $$O(s\ln(1/s))$$ at the origin and decays exponentially.
</div>

Evaluating (3) for the two operators above gives $$b/(-E^2)=0.99962,\,0.99736,\,0.98001$$ at $$E=0.01,\,0.03,\,0.1$$ in four dimensions and $$0.99953,\,0.99679,\,0.97625$$ in two, so both reproduce $$-E^2$$ at low momentum. The relative deviation grows like $$E^2\ln(1/E)$$, because the first correction is a term in $$E^4\ln E$$ (section 3).

Timelike momenta correspond to the boundary values of $$b$$ at $$Z=-\mu\pm i0$$. We use $$Z=-\mu+i0$$, that is $$E=+i\sqrt\mu$$, approached from $$\operatorname{Re}E>0$$.

## 2. Spectral weight

The Euclidean propagator of the free field defined by $$\bar B$$ is $$D(Z)=-1/b$$; for $$b=-Z$$ it is $$1/Z$$. A function $$D$$ on $$Z>0$$ has *positive spectral weight* if it is a *Stieltjes function*,

$$
D(Z)=\frac{r_0}{Z}+c+\int_{(0,\infty)}\frac{\rho(d\mu)}{Z+\mu},\qquad r_0,c\ge0,\quad\rho\ge0,\quad\int\frac{\rho(d\mu)}{1+\mu}<\infty. \tag{4}
$$

This is the Källén–Lehmann form [[Källén]](#ref-kallen) [[Lehmann]](#ref-lehmann): in a unitary quantum field theory the two-point function of a scalar field is a superposition of free propagators of mass squared $$\mu$$ with weight $$\rho(\mu)$$ equal to the norm of the states of that invariant mass, which is non-negative. A negative $$\rho$$ would mean states of negative norm in the standard quantisation that uses this Euclidean propagator. For a retarded non-local operator that quantisation is a choice, and the theorems below concern only the function $$D$$ on $$Z>0$$: they say that it cannot have the form (4). Section 6 returns to what this does and does not say about unitarity.

For a $$D$$ that is analytic in the cut plane away from isolated poles, the density is its discontinuity across the negative axis. Since $$\operatorname{Im}\,1/(Z+\mu')=-\pi\delta(\mu-\mu')$$ at $$Z=-\mu+i0$$, (4) gives $$\rho(\mu)=-\pi^{-1}\operatorname{Im}D(-\mu+i0)=\pi^{-1}\operatorname{Im}(1/b)(-\mu+i0)$$, with the boundary value $$E=+i\sqrt\mu$$ of section 1.

The constant $$c=D(\infty)=-1/a$$ is nonzero for every operator with a contact term; the page on [positive spectral weight and spectral dimension](/quantum-gravity/positivity-dimension-lemma/) excludes the case $$c>0$$, because the return amplitude of its diffusion diverges, and leaves a regularised definition open. Three elementary properties of (4) are used repeatedly. First, $$D$$ is positive and nonincreasing on $$Z>0$$, so $$D(\infty)=c$$ exists. Second, the function $$T(Z)=Z\,[D(Z)-c]=r_0+\int\rho(d\mu)\,\frac{Z}{Z+\mu}$$ is nondecreasing, at least $$r_0$$, and constant only if $$\rho=0$$. Third, $$D-r_0/Z$$ is nonincreasing with limit $$c+\int\rho/\mu\in[0,\infty]$$ at $$Z\to0^+$$ by monotone convergence.

In two dimensions the density of the Benincasa–Dowker operator is non-negative. With $$F=4e^{-V}(1-2V+\tfrac12V^2)$$, $$V=\tau^2/2$$, the boundary value of (3) has $$\operatorname{Im}\Pi=-\pi\int_0^\infty F\,\tau J_0(\sqrt\mu\,\tau)\,d\tau$$ because $$\operatorname{Im}2K_0(ix)=-\pi J_0(x)$$, and the Gaussian integrals $$\int_0^\infty\tau^{2n+1}e^{-\tau^2/2}J_0(\sqrt\mu\,\tau)d\tau=2^nn!\,L_n(u)e^{-u}$$ with $$u=\mu/2$$ and the Laguerre polynomials $$L_n$$ give $$\rho(\mu)=2u^2e^{-u}/\lvert b\rvert^2\ge0$$, with $$\rho\to\tfrac12$$ as $$\mu\to0$$.

The Stieltjes form itself needs in addition that $$b$$ has no zero off the cut and a contour argument on large circles as in section 4. Numerically, $$b$$ has no zero in $$10^{-4}\le\lvert Z\rvert\le400$$ off the cut, and $$1/Z+\tfrac12+\int\rho/(Z+\mu)$$ with $$\int\rho\,d\mu=1.0000$$ reproduces $$-1/b$$ at $$Z=0.5,\,2,\,10$$; the value $$1$$ of the integral is the one required by $$T(\infty)=r_0+\int\rho\,d\mu=2$$ (section 4). This is numerical evidence that the two-dimensional operator has positive weight, not a proof.

## 3. The infrared: the fourth-order coefficient

Positive weight constrains the small-momentum expansion of $$b$$.

<div class="proposition" markdown="1">
**Proposition 2** (fourth-order coefficient). Suppose $$D=-1/b=1/Z+S(Z)$$ with $$S(Z)=c+\int\rho(d\mu)/(Z+\mu)$$, $$c\ge0$$, $$\rho\ge0$$, and $$S\not\equiv0$$. Put $$\kappa=c+\int\rho(d\mu)/\mu\in(0,\infty]$$. Then for all $$Z>0$$

$$
0\le b(Z)+Z=\frac{Z^2S}{1+ZS}\le\kappa Z^2,\qquad \frac{b(Z)+Z}{Z^2}\longrightarrow\kappa\quad(Z\to0^+). \tag{5}
$$

</div>

<div class="proof" markdown="1">
*Proof.* From $$b=-1/D=-Z/(1+ZS)$$ we get $$b+Z=Z^2S/(1+ZS)$$, which is nonnegative. Since $$S$$ is nonincreasing, $$S\le S(0^+)=\kappa$$, so $$b+Z\le Z^2S\le\kappa Z^2$$. As $$Z\to0^+$$, $$S\to\kappa$$ and $$(b+Z)/Z^2=1/(1/S+Z)\to\kappa$$, also when $$\kappa=\infty$$.
</div>

Positive spectral weight with a massless pole of unit residue therefore forces $$b=-Z+\kappa Z^2+o(Z^2)$$ with $$\kappa>0$$, or a logarithmically divergent coefficient when $$\kappa=\infty$$. The bound runs in one direction: $$b+Z$$ lies below $$\kappa Z^2$$ and approaches it only as $$Z\to0$$. For the four-dimensional Benincasa–Dowker operator $$(b+Z)/Z^2$$ equals $$2.00,\,3.79,\,5.58,\,7.38$$ at $$E=10^{-1},10^{-2},10^{-3},10^{-4}$$. It is positive and grows by about $$1.8$$ per decade of $$E$$, which means $$\kappa=\infty$$ with a logarithm: the spectral density of section 4 tends to $$0.39$$ as $$\mu\to0$$, and $$0.39\ln100\approx1.8$$. This operator passes the infrared test and fails in the ultraviolet (section 4).

At the other extreme, an operator whose symbol agrees with the continuum one up to terms smaller than $$Z^2$$ cannot have positive weight, in the dimensions where Lemma 1 applies.

<div class="corollary" markdown="1">
**Corollary 3.** Let $$d\in\{2,4\}$$, let $$F$$ be measurable with $$\lvert F(\tau)\rvert\le C(1+\tau)^p$$, and suppose $$b(Z)=-Z+o(Z^2)$$ as $$Z\to0^+$$. In particular this holds for every *flat* operator, one with $$b(Z)=-Z+O(Z^N)$$ for every $$N$$. Then $$D=-1/b$$ does not have the form (4).
</div>

<div class="proof" markdown="1">
*Proof.* Suppose $$D$$ has the form (4). From $$b=-Z(1+o(Z))$$ we get $$ZD(Z)=1/(1+o(Z))\to1$$ as $$Z\to0^+$$. In (4), $$ZD=r_0+cZ+\int\rho(d\mu)\,Z/(Z+\mu)$$, and for $$Z\le1$$ the factor $$Z/(Z+\mu)$$ is at most $$1/(1+\mu)$$ and tends to $$0$$, so the integral tends to $$0$$ by dominated convergence and $$r_0=1$$. Hence $$D=1/Z+S$$ with $$S=c+\int\rho/(Z+\mu)\ge0$$. By Lemma 1, $$b$$ is bounded on $$Z\ge1$$, so $$b\ne-Z$$ identically, $$S\not\equiv0$$, and Proposition 2 gives $$(b+Z)/Z^2\to\kappa>0$$. This contradicts $$(b+Z)/Z^2=o(1)$$.
</div>

Corollary 3 is the only exclusion of positive weight proved for $$d=2$$ (section 4 explains why Theorem 8 is silent there). The question is then whether flat operators exist at all, and with what kernel.

### Flat operators and the tail exponent

The Mellin transform of the kernel, $$\widetilde F(s)=\int_0^\infty F(\tau)\tau^{s-1}d\tau$$, controls the small-$$E$$ expansion of the symbol.

<div class="lemma" markdown="1">
**Lemma 4** (Mellin structure of the symbol). Let $$d\in\{2,4\}$$ and $$\lvert F(\tau)\rvert\le Ce^{-\lambda\tau^\beta}$$ for all $$\tau>0$$, with $$\lambda,\beta>0$$. Then $$\widetilde F$$ is holomorphic in $$\operatorname{Re}s>0$$, and for $$E>0$$

$$
b(E)-a=\frac1{2\pi i}\int_{w_0-i\infty}^{w_0+i\infty}\widetilde k_d(w)\,\widetilde F(d-w)\,E^{-w}dw, \tag{6}
$$

with $$\widetilde k_2(w)=2^{w-1}\Gamma(w/2)^2$$ and $$0<w_0<2$$ for $$d=2$$, and $$\widetilde k_4(w)=\pi2^{w-1}\Gamma(w/2-1)\Gamma(w/2)$$ and $$2<w_0<4$$ for $$d=4$$. The function $$\widetilde k_d$$, the Mellin transform of $$k_d$$, has double poles at $$w=-2m$$ for $$m\ge0$$, with $$\widetilde k_d(w)=A_m(w+2m)^{-2}+B_m(w+2m)^{-1}+O(1)$$ and $$A_m\ne0$$ (for $$d=4$$, $$A_m=-2\pi4^{-m}/(m!(m+1)!)$$; for $$d=2$$, $$A_m=2\cdot4^{-m}/(m!)^2$$), and for $$d=4$$ a simple pole at $$w=2$$ with residue $$4\pi$$. Shifting the contour to the left gives, for every $$n$$, a small-$$E$$ expansion in which the coefficient of $$E^{2m}\ln E$$ is $$-A_m\widetilde F(d+2m)$$ and the coefficient of $$E^{2m}$$ is $$-A_m\widetilde F'(d+2m)+B_m\widetilde F(d+2m)$$ (plus $$a$$ for $$m=0$$), for $$m<n$$; for $$d=4$$ there is also the term $$4\pi\widetilde F(2)E^{-2}$$, and the remainder is $$O(E^{2n-1})$$.
</div>

<div class="proof" markdown="1">
*Proof.* Equation (6) is the Mellin–Parseval formula for (3), since $$\widetilde k_d$$ is the Mellin transform of $$k_d(z)$$ (from $$\int_0^\infty K_\nu(z)z^{\mu-1}dz=2^{\mu-2}\Gamma((\mu-\nu)/2)\Gamma((\mu+\nu)/2)$$) and $$\int F(\tau)\tau^{d-1}(E\tau)^{-w}d\tau=E^{-w}\widetilde F(d-w)$$. The strip for $$w_0$$ is where both transforms converge. The function $$\widetilde F(d-w)$$ is holomorphic for all $$\operatorname{Re}w<d$$ and bounded on vertical lines, while $$\widetilde k_d$$ decays like $$e^{-\pi\lvert\operatorname{Im}w\rvert/2}$$, so the contour can be moved left through the poles of $$\widetilde k_d$$ with vanishing contributions from the horizontal segments. The coefficients are the residues of the double poles, computed from the Laurent expansion of $$\widetilde k_d$$ and the Taylor expansion of $$\widetilde F(d-w)E^{-w}$$ at $$w=-2m$$.
</div>

A flat operator needs, order by order, the logarithmic and the pure-power coefficients to vanish except for the constant $$a$$ and the term $$-E^2$$.

<div class="corollary" markdown="1">
**Corollary 5.** If $$\bar B$$ is flat and $$F$$ satisfies the hypothesis of Lemma 4, then $$\widetilde F(d+2m)=0$$ for all $$m\ge0$$ (for $$d=4$$ also $$\widetilde F(2)=0$$), $$\widetilde F'(d+2m)=0$$ for all $$m\ge2$$, $$\widetilde F'(d+2)\ne0$$, and $$a=A_0\widetilde F'(d)$$. That is, $$\widetilde F$$ has double zeros at $$d+2m$$ for $$m\ge2$$, a simple zero at $$d+2$$, a zero at $$d$$ that is simple exactly when $$a\ne0$$, and for $$d=4$$ a zero at $$2$$.
</div>

The functions $$E^{2m}\ln E$$ and $$E^{2m}$$ are independent as $$E\to0$$, so each coefficient must match that of $$-E^2$$ separately. The coefficient of $$E^2$$ equals $$-1$$, which gives $$\widetilde F'(d+2)\ne0$$. The generalised family illustrates the point. Its kernels have $$F=e^{-V}\times$$ polynomial in $$V$$ with $$V=\omega_d\tau^d$$ ($$\omega_2=\tfrac12$$, $$\omega_4=\pi/24$$), so $$\lvert F\rvert\le Ce^{-\lambda\tau^d}$$ and Lemma 4 applies with $$\beta=d$$, and

$$
\widetilde F(s)=\frac1d\,\omega_d^{-s/d}\,\Gamma(s/d)\,P(s/d),\qquad P(x)=\sum_kc_k\frac{(x)_k}{k!},
$$

with $$P$$ a polynomial. It has finitely many zeros, so the expansion of $$b$$ contains terms in $$E^{2m}$$ and $$E^{2m}\ln E$$ from some order on, and these operators are exact to finite order only.

<div class="theorem" markdown="1">
**Theorem 6** (tail exponent of a flat operator). Let $$d\in\{2,4\}$$ and let $$F\not\equiv0$$ satisfy $$\lvert F(\tau)\rvert\le Ce^{-\lambda\tau^\beta}$$ with $$\lambda>0$$. If $$\bar B$$ is flat, then $$\beta\le\tfrac12$$.
</div>

By Corollary 5, $$\widetilde F$$ has double zeros at the integers $$d+2m$$, $$m\ge2$$, while its growth to the right is at most that of $$\Gamma(\sigma/\beta)$$. Carleman's formula for the half-plane relates the density of zeros to the size of $$\ln\lvert\widetilde F\rvert$$ on large half-circles; the zeros contribute $$\ln R$$ and the growth allows only $$\ln R/(2\beta)$$, which is enough for $$\beta\le\tfrac12$$ and no more.

<div class="proof" markdown="1">
*Proof.* Put $$f(z)=\widetilde F(z+\delta_0)$$ with $$0<\delta_0<d$$ small enough that $$f$$ is holomorphic in a neighbourhood of $$\operatorname{Re}z\ge0$$. By Corollary 5, $$f$$ has zeros of multiplicity at least $$2$$ at $$z_m=d+2m-\delta_0$$ for $$m\ge2$$. Carleman's formula [[Levin]](#ref-levin) [[Peloso–Salvatori]](#ref-ps) states that, for zeros $$z_j=\lvert z_j\rvert e^{i\theta_j}$$ with $$1\le\lvert z_j\rvert\le R$$,

$$
\sum_{\lvert z_j\rvert\le R}\Big(\frac1{\lvert z_j\rvert}-\frac{\lvert z_j\rvert}{R^2}\Big)\cos\theta_j=\frac1{\pi R}\int_{-\pi/2}^{\pi/2}\ln\lvert f(Re^{i\theta})\rvert\cos\theta\,d\theta+\frac1{2\pi}\int_1^R\Big(\frac1{y^2}-\frac1{R^2}\Big)\ln\lvert f(iy)f(-iy)\rvert\,dy+A(R), \tag{7}
$$

with $$A(R)$$ bounded. All terms on the left are nonnegative, so the zeros $$z_m$$ alone give a lower bound. They have multiplicity two and spacing two, hence the left side is at least $$\ln R-O(1)$$. On the right, $$\lvert f(iy)\rvert\le\int\lvert F\rvert\tau^{\delta_0-1}d\tau<\infty$$, so the second term is $$O(1)$$ from above. For the arc, $$\lvert f(\sigma+it)\rvert\le\int\lvert F\rvert\tau^{\sigma+\delta_0-1}d\tau\le(C/\beta)\,\Gamma\big((\sigma+\delta_0)/\beta\big)\lambda^{-(\sigma+\delta_0)/\beta}$$, so on $$\lvert z\rvert=R$$, with $$\sigma=R\cos\theta$$, $$\ln\lvert f\rvert\le(R\cos\theta/\beta)\ln R+O(R)$$. The first term of the right side is then at most $$\frac{1}{\pi R}\int(R\cos\theta/\beta)\ln R\cos\theta\,d\theta+O(1)=\ln R/(2\beta)+O(1)$$. Comparing, $$\ln R-O(1)\le\ln R/(2\beta)+O(1)$$ for all large $$R$$, which requires $$1\le1/(2\beta)$$.
</div>

The same argument for a function with zeros of multiplicity $$q$$ on a progression of step $$g$$ gives $$\beta\le g/(2q)$$. With simple zeros at every integer, as in the problem treated in [heavy-tailed horizon-count kernels](/quantum-gravity/nonlocal-weights/), it gives $$\beta\le\tfrac12$$ again (the tail exponent is called $$\kappa$$ there, while $$\kappa$$ here is the coefficient of Proposition 2). The boundary $$\beta=\tfrac12$$ with a fixed $$\lambda>0$$ is excluded there for kernels with the required zeros by a Fourier argument (its Theorem 1). For the d'Alembertian the Carleman comparison is an equality at $$\beta=\tfrac12$$ and gives no contradiction, so this case is open.

<div class="theorem" markdown="1">
**Theorem 7** (flat operators exist for every smaller exponent). Let $$d\in\{2,4\}$$ and $$0<\beta<\tfrac12$$. Put $$h(t)=\tfrac12\big[e^{-t^\beta}-e^{-t^\beta\cos\pi\beta}\cos(t^\beta\sin\pi\beta)\big]$$, $$g_\alpha(\tau)=\tau^{-\alpha}\int_\tau^\infty t^{\alpha-1}h(t)\,dt$$, $$(\alpha_1,\alpha_2)=(2,4)$$ for $$d=2$$ and $$(4,6)$$ for $$d=4$$, and $$F=\gamma\,(g_{\alpha_2}-g_{\alpha_1})/(\alpha_2-\alpha_1)$$ with a constant $$\gamma$$. For a suitable real $$\gamma$$ (fixed by the normalisation of the $$E^2$$ term) and a real $$a$$ (fixed by $$b(0)=0$$), $$F$$ is bounded and continuous with $$F(0)=0$$ and $$\lvert F(\tau)\rvert\le C(1+\tau)^{-\beta}e^{-\cos(\pi\beta)\tau^\beta}$$, and the operator (2) is flat: for every $$N$$ there is $$C_N$$ with $$\lvert b(E)+E^2\rvert\le C_N\lvert E\rvert^N$$ for all $$E$$ with $$\lvert E\rvert\le1$$ and $$\lvert\arg E\rvert\le\pi/2$$, which includes the timelike boundary values.
</div>

The function $$h$$ combines $$e^{-t^\beta}$$ with its values on the rays $$\arg t^\beta=\pm\pi\beta$$ in a way that produces the factor $$\sin^2(\pi s/2)$$ in its Mellin transform, and with it double zeros at all even integers; the contour rotation behind this stays in the half-plane of convergence because $$\beta<\tfrac12$$. A variant with simple zeros at all integers beyond some point is Theorem 4 of [heavy-tailed horizon-count kernels](/quantum-gravity/nonlocal-weights/), where the kernels are not shown to come from weights. The division by $$(s-\alpha_1)(s-\alpha_2)$$ in $$\widetilde F$$ reduces the double zeros at $$d$$ and $$d+2$$ to simple ones, which is the pattern of Corollary 5.

<div class="proof" markdown="1">
*Proof.* Rotating the contour gives $$\int_0^\infty t^{s-1}e^{-t^\beta e^{\pm i\pi\beta}}dt=\beta^{-1}\Gamma(s/\beta)e^{\mp i\pi s}$$ since $$\cos\pi\beta>0$$. Hence $$\widetilde h(s)=\beta^{-1}\Gamma(s/\beta)\sin^2(\pi s/2)$$. By Fubini, $$\widetilde g_\alpha(s)=\widetilde h(s)/(s-\alpha)$$, and for even $$\alpha$$ the right side is regular at $$s=\alpha$$, so $$g_\alpha$$ is bounded with $$g_\alpha(\tau)=O(\tau^\beta)$$ at the origin. Therefore

$$
\widetilde F(s)=\frac{\gamma\,\Gamma(s/\beta)\sin^2(\pi s/2)}{\beta\,(s-\alpha_1)(s-\alpha_2)}.
$$

It has simple zeros at $$\alpha_1,\alpha_2$$ and double zeros at every other even integer, including $$s=2$$ for $$d=4$$, where Corollary 5 asks only for a zero. With $$\gamma$$ chosen so that the coefficient of $$E^2$$ is $$-1$$ and $$a$$ chosen to cancel the constant (both are nonzero real numbers determined by the residues in Lemma 4), every residue in the shift of (6) to the left of $$w=-2$$ vanishes, because a double pole of $$\widetilde k_d$$ meets a double zero of $$\widetilde F(d-w)$$. Hence $$b(E)+E^2$$ equals the integral in (6) on the line $$\operatorname{Re}w=-L$$ for any $$L>2$$. On that line $$\lvert\widetilde k_d(w)\widetilde F(d-w)E^{-w}\rvert\le C_L(1+\lvert y\rvert)^{m}\lvert E\rvert^{L}e^{(\pi/2-\pi/(2\beta)+\lvert\arg E\rvert)\lvert y\rvert}$$, with $$y=\operatorname{Im}w$$, from the growth $$e^{-\pi\lvert y\rvert/2}$$ of $$\widetilde k_d$$, $$e^{-\pi\lvert y\rvert/(2\beta)}$$ of $$\Gamma((d-w)/\beta)$$, $$e^{\pi\lvert y\rvert}$$ of $$\sin^2$$, and $$e^{\lvert\arg E\rvert\lvert y\rvert}$$ of $$E^{-w}$$. For $$\lvert\arg E\rvert\le\pi/2$$ the exponent is at most $$\pi-\pi/(2\beta)<0$$ precisely when $$\beta<\tfrac12$$, so the integral converges and is $$O(\lvert E\rvert^L)$$; the identity extends from $$E>0$$ to the sector by analytic continuation in $$E$$. The decay of $$F$$ follows from the integral defining $$g_\alpha$$.
</div>

For $$d=4$$ and $$\beta=0.3$$ the residue conditions give $$\gamma=-2.5452\times10^{-18}$$ and $$a=-7.3438\times10^{-8}$$, and the ratio $$b/(-E^2)$$ is $$0.073,\ 0.505,\ 0.963,\ 0.995,\ 1.000$$ at $$E=10^{-3},\,3\times10^{-4},\,10^{-4},\,6.5\times10^{-5},\,10^{-5}$$. The operator therefore reproduces $$-E^2$$ to within four percent of $$E^2$$ only below $$E\approx10^{-4}$$, that is at wavelengths longer than about $$10^{4}$$ discreteness lengths; at shorter wavelengths the symbol is close to the constant $$a$$, which is tiny in these units.

The threshold $$\tfrac12$$ is the same as for the horizon-count kernels of that page, except that the boundary case is open here, while there it is excluded for a fixed constant in the exponent. On the finer scale, section 6 of that page excludes the tails $$\exp(-c\sqrt x/\log x)$$ for every fixed $$c>0$$ and, at $$\exp(-c\sqrt x/\log^2x)$$, sketches functions with the vanishing moments but does not construct a kernel from them. Its Theorem 4 covers the tails $$\exp(-c\,x^{\kappa})$$ with $$\kappa<\tfrac12$$. The tails in between are therefore not decided by the results of this page and that one. For $$\beta>\tfrac12$$ the exponent in the proof is positive at $$\arg E=\pi/2$$, so this proof gives no bound on the timelike axis.

## 4. The ultraviolet: superconvergence

Theorem 8 uses only the large-momentum behaviour of the symbol and assumes no special structure at low momentum.

<div class="theorem" markdown="1">
**Theorem 8** (no positive weight in four dimensions). Let $$d=4$$ and let $$\bar B$$ be of the form (2) with $$F$$ measurable, $$\lvert F(\tau)\rvert\le C(1+\tau)^p$$ for some $$p\ge0$$, and $$F$$ not vanishing almost everywhere. Then no function $$D$$ of the form (4) satisfies $$D\,b=-1$$ on $$Z>0$$; in particular, if $$b$$ has no zero on $$Z>0$$, then $$D=-1/b$$ does not have the form (4).
</div>

A Stieltjes function cannot approach its limit at infinity faster than $$1/Z$$, because $$T(Z)=Z[D-D(\infty)]$$ is nondecreasing. The symbol of any four-dimensional kernel with polynomially bounded $$F$$ approaches its limit like $$Z^{-2}$$. The first statement is the monotonicity of $$T$$, and the second is dominated convergence.

<div class="proof" markdown="1">
*Proof.* Write $$b=a+\Pi$$. Substituting $$\tau=s/E$$ in (3),

$$
\Pi(E)=\frac{4\pi}{E^4}\int_0^\infty F(s/E)\,s^2K_1(s)\,ds,
$$

and for $$E\ge1$$ the integrand is bounded by $$C(1+s)^ps^2K_1(s)$$, which is integrable, so $$E^4\Pi(E)$$ is bounded: $$\Pi=O(Z^{-2})$$. (If $$F(0^+)$$ exists, $$E^4\Pi\to8\pi F(0^+)$$, since $$\int_0^\infty s^2K_1(s)ds=2$$.)

Suppose $$D$$ has the form (4) and $$Db=-1$$. If $$a>0$$, then $$D=-1/b\to-1/a<0$$ as $$Z\to\infty$$, contradicting $$D\ge0$$. If $$a=0$$, then $$b=\Pi$$ and $$D=-1/\Pi$$ has $$\lvert D\rvert\ge c'Z^2\to\infty$$, contradicting that a function of the form (4) is bounded on $$[1,\infty)$$ (by $$D(1)$$). If $$a<0$$, then $$D(\infty)=-1/a$$ and

$$
D-D(\infty)=\frac{\Pi}{ab},\qquad T(Z)=Z\,[D-D(\infty)]=O(Z^{-1})\to0,
$$

because $$b\to a\ne0$$. But $$T$$ is nondecreasing and nonnegative, so $$T\equiv0$$, $$\rho=0$$ and $$r_0=0$$, and $$D$$ is constant, so $$b$$ is constant. Then $$\Pi$$ is constant, hence zero since $$\Pi\to0$$. Now $$\Pi(E)=\int_0^\infty e^{-Et}G(t)\,dt$$ with $$G(t)=4\pi\int_0^t\tau\sqrt{t^2-\tau^2}\,F(\tau)\,d\tau$$ (proof of Lemma 1), so $$G=0$$ almost everywhere by uniqueness of the Laplace transform. In the variable $$u=t^2$$, $$w=\tau^2$$ this reads $$\int_0^u\sqrt{u-w}\,F(\sqrt w)\,dw=0$$, a convolution of $$u^{1/2}$$ with a locally integrable function, and Titchmarsh's convolution theorem gives $$F=0$$ almost everywhere, contrary to the assumption.
</div>

The hypotheses hold for the whole generalised family.

<div class="proposition" markdown="1">
**Proposition 9** (finite-density operators). Let $$F=e^{-V}\sum_kc_kV^k/k!$$ with finitely many nonzero coefficients, or with $$\lvert c_k\rvert\le C(1+k)^p$$ for all $$k$$, and not all $$c_k$$ zero. Then $$F$$ satisfies the hypotheses of Theorem 8: it is bounded by a power of $$\tau$$ and does not vanish almost everywhere. So does the operator obtained by replacing $$V$$ by $$\varepsilon V$$ in the layer weights, for any $$\varepsilon>0$$.
</div>

<div class="proof" markdown="1">
*Proof.* $$\lvert F\rvert\le e^{-V}\sum C(1+k)^pV^k/k!=C\,\mathbb E[(1+N_V)^p]\le C'(1+V)^p$$ for a Poisson variable $$N_V$$ of mean $$V$$, which is polynomial in $$\tau$$. The sum $$\sum c_kV^k/k!$$ is an entire function of $$V$$, not identically zero, so $$F$$ vanishes only on a discrete set of $$\tau$$. Replacing $$V$$ by $$\varepsilon V$$ changes neither property.
</div>

So no operator in the generalised family, whatever its nonzero coefficients, has a propagator of the form (4) in four dimensions. The same scaling works for $$d\ge3$$: $$E^2\Pi=E^{2-d}\int F(s/E)s^{d-1}k_d(s)\,ds=O(E^{2-d})\to0$$, since $$\int_0^\infty s^{d-1}k_d(s)ds=\Omega_{d-2}\Gamma(d)\int_0^\infty\sinh^{d-2}\eta\cosh^{-d}\eta\,d\eta<\infty$$, and the case analysis of the proof applies to $$T=Z[D-D(\infty)]=O(E^{2-d})$$. For $$d=2$$ it gives only $$T(\infty)=2F(0^+)/a^2$$, which for the two-dimensional Benincasa–Dowker operator is $$2$$; since $$T\ge r_0=1$$ there is no contradiction, and the theorem is silent in two dimensions.

### Computed spectral data of the four-dimensional Benincasa–Dowker operator

We take $$a=-4/\sqrt6=-1.63299$$ and $$F(\tau)=\tfrac4{\sqrt6}e^{-V}(1-9V+8V^2-\tfrac43V^3)$$ with $$V=\pi\tau^4/24$$, and evaluate (3) for complex $$E$$.

The function $$T(Z)=Z[D-D(\infty)]$$ equals $$0.1507,\,0.01539,\,0.001539$$ at $$Z=10^2,10^3,10^4$$, so it decreases like $$1/Z$$, which no function of the form (4) does.

Off the cut, $$b$$ has a pair of zeros at $$Z_*=-25.384\pm3.795\,i$$, with residues of $$D=-1/b$$ equal to $$r=-1/b'(Z_*)=-0.8954\pm3.2805\,i$$, so the pair has $$2\operatorname{Re}r=-1.7907$$. The argument principle on a contour around the cut plane between $$\lvert Z\rvert=10^{-3}$$ and $$400$$ gives winding number $$2$$, so these are the only zeros there. This is the complex pair, of mass squared of order $$25$$ in units of the discreteness scale, that signals the classical instability found in [[ASS]](#ref-ass).

On the cut, $$\rho(\mu)=\pi^{-1}\operatorname{Im}(1/b)$$ tends to $$0.39$$ as $$\mu\to0$$, which accounts for the logarithm of section 3. It is positive up to $$\mu\approx9.4$$ and changes sign at $$\mu\approx9.40,\ 24.31,\ 43.18,\ 65.29,\ 90.24,\ 117.73,\ 147.54$$ (and further out, with a rapidly falling amplitude, $$\rho=-5\times10^{-3}$$ at $$\mu=100$$ and $$-4\times10^{-7}$$ at $$300$$). Its integral over $$\mu<400$$ is $$0.79072$$, made up of positive weight $$2.578$$ and negative weight $$-1.788$$.

The sum rule follows from Cauchy's theorem for $$\tilde D=D-D(\infty)$$ on a large circle $$\lvert Z\rvert=R$$ that encloses the pole at $$Z=0$$ (residue $$1$$, since $$b\approx-Z$$), the complex poles $$Z_j$$ with residues $$r_j$$, and the cut, which contributes $$\int\rho\,d\mu$$ as it does for a Stieltjes function. If $$Z\tilde D\to0$$ uniformly on the circles, including where they cross the cut, then

$$
1+\int\rho\,d\mu+\sum_jr_j=0. \tag{8}
$$

For a function of the form (4) the contour integral tends to $$r_0+\int\rho\,d\mu=T(\infty)>0$$. Equation (8) says that the total weight, counted with the unit weight of the massless pole, vanishes. Theorem 8 gives the decay of $$Z\tilde D$$ along the positive axis only, which does not by itself give it uniformly in the cut plane. For the four-dimensional Benincasa–Dowker operator, $$1+0.790724+(-1.790724)$$ vanishes to $$10^{-7}$$, which supports the uniform decay for this operator; it is not proved. In this operator the unit weight is cancelled by the complex pair together with the negative part of the cut density. A stable operator of this type, without zeros off the cut, would need $$\int\rho\,d\mu=-1$$, which no nonnegative density has.

### The light-cone escape

The proof of Theorem 8 used that $$\Pi$$ comes from a function $$F$$ bounded near the light cone. A term supported on the light cone changes the decay of $$\Pi$$.

<div class="proposition" markdown="1">
**Proposition 10** (weight on the light cone). Let $$d=4$$ and add to (2) a term $$\nu\,\theta(t)\delta(\tau^2)$$, $$\nu\in\mathbb R$$, with $$a\ne0$$ and $$F$$ as in Theorem 8. Its symbol is $$2\pi\nu/Z$$, so $$b=a+2\pi\nu/Z+O(Z^{-2})$$ and $$T(\infty)=2\pi\nu/a^2$$. If $$D$$ has the form (4) with $$r_0=1$$, then $$2\pi\nu\ge a^2$$. The bound is attained by $$B_M=-M^2\delta^4+M^4G^{(M)}_{\rm ret}$$, where $$G^{(M)}_{\rm ret}=\frac1{2\pi}\theta(t)\delta(\tau^2)-\frac M{4\pi\tau}J_1(M\tau)\theta(t)\theta(\tau^2)$$ is the retarded Green function of $$-\Box+M^2$$: its symbol is $$b=-M^2Z/(Z+M^2)$$, and $$D=1/Z+1/M^2$$ is a Stieltjes function.
</div>

<div class="proof" markdown="1">
*Proof.* The symbol of $$\theta(t)\delta(\tau^2)$$ is $$\int_0^\infty dt\,e^{-Et}\int4\pi r^2\delta(t^2-r^2)dr=\int_0^\infty te^{-Et}dt\cdot2\pi=2\pi/E^2$$. Then $$D-D(\infty)=(b-a)/(ab)=2\pi\nu/(a^2Z)(1+o(1))$$, so $$T(\infty)=2\pi\nu/a^2$$. A function of the form (4) has $$a<0$$ (so that $$D(\infty)=-1/a\ge0$$) and $$T\ge r_0=1$$, so $$2\pi\nu\ge a^2$$. For $$B_M$$, $$G^{(M)}_{\rm ret}$$ has symbol $$1/(Z+M^2)$$, so $$b=-M^2+M^4/(Z+M^2)=-M^2Z/(Z+M^2)$$, and $$\nu=M^4/(2\pi)$$, $$a=-M^2$$ give $$2\pi\nu=a^2$$.
</div>

The operator $$B_M$$ is retarded, Lorentz invariant and non-local, has a bounded kernel away from the light cone, and carries positive weight. The mean operator of a sprinkling with layer weights is (2), which has no light-cone term (Proposition 9), and the interval-local matrix kernels of section 5 have none either. A light-cone term would have to come from a construction outside these classes.

## 5. Non-commutative points

Theorem 8 concerns a scalar field. A generalisation of a causal set replaces each point $$x$$ by a matrix algebra $$M_{n_x}$$ and the order by a *quantum relation* [[Weaver]](#ref-weaver) [[KLM]](#ref-klm) [[Weaver 2021]](#ref-weaver2), a family of subspaces $$\mathcal V_{xy}\subseteq B(\mathbb C^{n_y},\mathbb C^{n_x})$$ that is reflexive, transitive and antisymmetric in a sense that reduces to the usual axioms when every $$n_x=1$$. We use a sprinkled quantum causal set only through the form of the mean-field kernel it produces, and state that form as a hypothesis. Whether a given construction of the interval operator, which replaces the integer $$\lvert I(x,y)\rvert$$, has a mean kernel of this form is a separate verification, as in section 6 of [mixed-Poisson and signed substrates](/quantum-gravity/signed-substrates/), where Theorem 17 states the corresponding hypotheses (Q1)–(Q3).

Suppose the atoms carry independent marks with $$n_x\le n_{\max}$$, and that the link data $$\mathcal V_{xy}$$ depend only on the marks inside the interval $$[x,y]$$ and on their order (*interval-local* link data). Let $$N$$ be the number of components of the field on which the operator acts, the total dimension of the multiplet attached to a point. We assume that the mean operator then takes the form of an $$N\times N$$ matrix kernel $$F_{ij}(\tau)=e^{-V}\sum_kg^{ij}_kV^k/k!$$, where $$g^{ij}_k$$ is the mean, given $$k$$ interior atoms and the endpoint components $$i,j$$, of the corresponding entry of a weight function applied to the interval operator, plus a contact matrix $$a$$, with the $$g_k$$ not all zero (if all vanish, $$F=0$$ and $$b=a$$ is constant, and Theorem 11 does not apply). We say the kernel has *property (Q)* if $$\lVert g_k\rVert\le C(1+k)^p$$; then $$\lVert F(\tau)\rVert\le C'(1+\tau)^p$$ by the argument of Proposition 9. For the scalar weights of (1), $$g_k=c_k$$ and (Q) is the condition of Proposition 9. Property (Q) is weaker than the boundedness (Q2) of the cited page, and no infrared normalisation is assumed.

A matrix-valued $$D$$ is *operator-Stieltjes* if it has the form (4) with $$r_0,c$$ positive semidefinite matrices and $$\rho$$ a positive-semidefinite-valued measure. Then $$D(Z)$$ is Hermitian for $$Z>0$$ and nonincreasing in the Loewner order.

<div class="theorem" markdown="1">
**Theorem 11** (matrix-valued kernels). Let $$d=4$$ and let $$\bar B$$ act on $$\mathbb R^N$$-valued fields by $$a\phi(x)+\int_{y\prec x}F(\tau)\phi(y)d^4y$$, with $$a$$ a real $$N\times N$$ matrix and $$F$$ an $$N\times N$$ matrix of measurable functions bounded by $$C(1+\tau)^p$$. Let $$b(E)=a+\Pi(E)$$ be its symbol, $$b$$ not constant. Then no operator-Stieltjes $$D$$ satisfies $$D\,b=-1$$ on $$Z>0$$; in particular, if $$b$$ is invertible for $$E>0$$, then $$D=-b^{-1}$$ is not operator-Stieltjes.
</div>

<div class="proof" markdown="1">
*Proof.* Each entry of $$\Pi$$ is $$O(Z^{-2})$$ by the proof of Theorem 8. Suppose $$D$$ is operator-Stieltjes with $$Db=-1$$. For every vector $$v$$, $$\langle v,Dv\rangle$$ is a scalar function of the form (4).

If $$a$$ is invertible, $$D(\infty)=-a^{-1}$$ and $$D-D(\infty)=a^{-1}-(a+\Pi)^{-1}=(a+\Pi)^{-1}\Pi a^{-1}=O(\Pi)$$. So $$T_v(Z)=Z\langle v,(D-D(\infty))v\rangle\to0$$ for each $$v$$; being nondecreasing and nonnegative, it vanishes, so $$r_0=0$$, $$\rho=0$$ and $$D$$ is constant, and so is $$b$$, contradicting the assumption.

If $$a$$ is singular, pick a unit $$v$$ with $$av=0$$. Then $$b(E)v=\Pi(E)v\to0$$, so $$\lVert b^{-1}\rVert\ge1/\lVert\Pi v\rVert\to\infty$$ and $$\lVert D(Z)\rVert\to\infty$$. But an operator-Stieltjes function satisfies $$0\preceq D(Z)\preceq D(1)$$ for $$Z\ge1$$, which is bounded. Contradiction.
</div>

No symmetry of $$F$$ is assumed: a non-symmetric $$b$$ gives a non-Hermitian $$D=-b^{-1}$$, which cannot be operator-Stieltjes in any case.

If the mean kernel of non-commutative points of bounded size with interval-local link data has the form assumed above (with the $$g_k$$ not all zero) and satisfies (Q), it is of the form assumed in Theorem 11, and such a structure has no operator-Stieltjes propagator in four dimensions. The statement concerns the matrix $$D$$; it does not decide whether the physical field is the whole multiplet or a compression of it. The same conclusion, with the infrared normalisation $$b=-WZ+o(Z)$$, $$W$$ positive definite, assumed and the boundedness (Q2) in place of (Q), is Theorem 17 of the cited page; Theorem 11 here does not need that normalisation.

## 6. Kernels and quantisations outside the hypotheses

Theorem 8 is silent for $$d=2$$: for the two-dimensional Benincasa–Dowker operator $$T(\infty)=2\ge r_0=1$$, which leaves room for positive weight. Corollary 3 is the only exclusion proved in two dimensions, and the density of the two-dimensional Benincasa–Dowker operator is non-negative (section 2). Whether its propagator is a Stieltjes function, which needs the absence of zeros off the cut and the decay of $$D-D(\infty)$$ on large circles, is open. The same uniform decay is the unproved step behind the sum rule (8) in four dimensions.

A light-cone term with $$2\pi\nu\ge a^2$$ is necessary for positive weight with a unit massless pole and is attained by the continuum construction $$B_M$$ (Proposition 10), while neither layer-weight operators nor interval-local matrix kernels can carry one. Whether link data that depend on the whole set rather than on an interval produce such a term has been neither excluded nor exhibited. Kernels that grow faster than a power, for instance from layer weights that grow geometrically so that $$F$$ grows exponentially in $$V$$, kernels with other singular behaviour at the light cone, and Lorentz-violating operators, whose kernels do not depend on $$\tau$$ alone, lie outside Theorem 8.

For a retarded non-local operator the propagator that defines the Hilbert space may be built differently from the Euclidean continuation used here, for example from the Pauli–Jordan function as in the quantisation of [[BBL]](#ref-bbl), or with a prescription that removes the complex pair of zeros from the physical spectrum. Whether a negative $$\rho$$ of the kind found in section 4 then means negative norm or a different pathology depends on that quantisation, and the theorems do not address it.

Two questions about the flat operators are open. Theorem 6 leaves the exponent $$\beta=\tfrac12$$ undecided, since the Carleman comparison is an equality there. The negative weight carried by the operators of Theorem 7 at the length scale where the symbol departs from $$-E^2$$ has not been computed.

## References

1. <a id="ref-blms"></a>Bombelli, Lee, Meyer, Sorkin, “Space-time as a causal set,” Phys. Rev. Lett. 59, 521 (1987).
2. <a id="ref-surya"></a>Surya, “The causal set approach to quantum gravity,” Living Rev. Relativ. 22, 5 (2019), [arXiv:1903.11544](https://arxiv.org/abs/1903.11544).
3. <a id="ref-sorkin"></a>Sorkin, “Does locality fail at intermediate length-scales?,” in *Approaches to Quantum Gravity*, ed. Oriti, Cambridge University Press (2009), [arXiv:gr-qc/0703099](https://arxiv.org/abs/gr-qc/0703099).
4. <a id="ref-bd"></a>Benincasa, Dowker, “The scalar curvature of a causal set,” Phys. Rev. Lett. 104, 181301 (2010), [arXiv:1001.2725](https://arxiv.org/abs/1001.2725).
5. <a id="ref-dg"></a>Dowker, Glaser, “Causal set d'Alembertians for various dimensions,” Class. Quantum Grav. 30, 195016 (2013), [arXiv:1305.2588](https://arxiv.org/abs/1305.2588).
6. <a id="ref-ass"></a>Aslanbeigi, Saravani, Sorkin, “Generalized causal set d'Alembertians,” JHEP 06 (2014) 024, [arXiv:1403.1622](https://arxiv.org/abs/1403.1622).
7. <a id="ref-bbl"></a>Belenchia, Benincasa, Liberati, “Nonlocal scalar quantum field theory from causal sets,” JHEP 03 (2015) 036, [arXiv:1411.6513](https://arxiv.org/abs/1411.6513).
8. <a id="ref-kallen"></a>Källén, “On the definition of the renormalization constants in quantum electrodynamics,” Helv. Phys. Acta 25, 417 (1952).
9. <a id="ref-lehmann"></a>Lehmann, “Über Eigenschaften von Ausbreitungsfunktionen und Renormierungskonstanten quantisierter Felder,” Nuovo Cimento 11, 342 (1954).
10. <a id="ref-levin"></a>Levin, *Lectures on Entire Functions*, Translations of Mathematical Monographs 150, American Mathematical Society (1996).
11. <a id="ref-ps"></a>Peloso, Salvatori, “Functions of exponential growth in a half-plane, sets of uniqueness and the Müntz–Szász problem for the Bergman space,” [arXiv:1511.05718](https://arxiv.org/abs/1511.05718), Theorem 4.3.
12. <a id="ref-weaver"></a>Weaver, “Quantum relations,” Mem. Amer. Math. Soc. 215 (2012), [arXiv:1005.0354](https://arxiv.org/abs/1005.0354).
13. <a id="ref-klm"></a>Kornell, Lindenhovius, Mislove, “A category of quantum posets,” Indag. Math. 33, 1137 (2022), [arXiv:2101.11184](https://arxiv.org/abs/2101.11184).
14. <a id="ref-weaver2"></a>Weaver, “Hereditarily antisymmetric operator algebras,” J. Inst. Math. Jussieu 20, 1039 (2021), [arXiv:1902.09989](https://arxiv.org/abs/1902.09989).
