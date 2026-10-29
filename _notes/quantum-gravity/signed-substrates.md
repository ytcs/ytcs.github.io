---
title: Mixed-Poisson and signed substrates and the four-dimensional d'Alembertian
date: 2026-10-29
category: substrate
order: 3
description: A sprinkling whose density is random, or a signed or complex combination of sprinklings, changes the coefficient of the causal-set d'Alembertian by a factor that is less than one in four dimensions unless the process is Poisson, and a probabilistic reading of its counts forces it to be a classical mixture. Finite-order perturbations of a Poisson process are exactly Poisson above a fixed number of points, and quantum-poset decorations of a sprinkling do not remove the obstruction to a ghost-free operator.
status: The mean of the operator is computed exactly and its limit is proved for exponential fields in two and four dimensions; the classification statements are proved under the hypotheses listed; numerical checks of every closed form; the case of lattice combinations is shown to lie outside the closure and is not resolved.
excludes: Mixed-Poisson substrates with a positive random density as carriers of the four-dimensional causal-set d'Alembertian normalised by the mean density; signed or complex invariant processes with a Born reading of their counts; finite-order Poincaré-invariant perturbations of a Poisson process as sources of large-scale order content, under the stated form of the positivity axiom; interval-local quantum-poset decorations as a route to a ghost-free mean operator.
---

A causal set is a set of events with a partial order, and a *sprinkling* produces one from Minkowski space by choosing points at random and keeping the causal order between them. For a Poisson sprinkling there is a retarded operator on functions on the points, due to Benincasa and Dowker, that approximates the d'Alembertian $$\Box$$ of the continuum when the density of points is large. We ask what happens to this operator when the point process is changed in ways that keep Poincaré invariance: when the density is random but the same everywhere, when the process is a signed or complex combination of Poisson processes, or when it is a Poisson process with a finite number of its correlation functions modified. The answers have the following form. In four dimensions the mean of the operator on a mixed Poisson process, normalised by the mean density, is $$C\,\Box$$ with $$C=\mathbb E[\Theta^{1/2}]$$ for a density $$\rho\Theta$$ of mean $$\rho$$, which is less than 1 unless the density is deterministic (Theorem 3 and Corollary 4); in two dimensions $$C=1$$ for every mixing law. If the counts of such a process are required to be probabilities in the usual sense, a signed or complex mixing law is a positive one (Proposition 5), so the signed constructions in which $$C=1$$ (Example 1) have no such reading. Perturbing finitely many correlation functions leaves the process exactly Poisson on all configurations with more than a fixed number of points, and the operator sees only the timelike part of a pair amplitude (Propositions 9 and 10 and Theorem 11). Under an axiom that the macroscopic densities of the process have a probability law, no such perturbation carries large-scale order content (Proposition 12). The axiom has several forms, and which one is adopted decides how much is closed; a bounded-variation combination of a Poisson process and a Haar-random lattice satisfies the strongest form and is not covered (Proposition 15). The mean operator of a sprinkling decorated by quantum-poset data inherits the obstruction of the four-dimensional operator (Theorem 17).

The operator is due to Benincasa and Dowker [[BD]](#ref-bd) in four dimensions, and to Sorkin [[Sorkin 2007]](#ref-sorkin07) in two. Dowker and Glaser [[DG]](#ref-dg) give the family in every dimension, and Aslanbeigi, Saravani and Sorkin [[ASS]](#ref-ass) derive its action on plane waves and the conditions on the coefficients; Belenchia, Benincasa and Liberati [[BBL]](#ref-bbl) study the non-local field theory it defines. Mixed Poisson processes are the Cox processes of point-process theory [[Last–Penrose]](#ref-lp). Complex point processes and their classification by total variation are on [Lorentz-invariant quantum sprinklings](/quantum-gravity/quantum-sprinklings/), and the non-positivity of the four-dimensional operator that Theorem 17 uses is on [The causal-set d'Alembertian has negative spectral weight](/quantum-gravity/causal-box-spectrum/). The page adds the exact scaling identity for the mean on a mixed Poisson process and its limit, the Born-reading argument for signed mixing laws, the exact structure of finite-order perturbations, and the statement of which closure holds under which positivity axiom.

## 1. Causal sets, sprinklings and the Benincasa–Dowker operator

A *causal set* is a locally finite partially ordered set: every interval $$\{z:y\prec z\prec x\}$$ is finite. A causal set is *sprinkled* into Minkowski space $$M=\mathbb R^{1,d-1}$$ (signature $$-+\cdots+$$, $$\Box=-\partial_t^2+\Delta$$, $$d=2$$ or $$4$$) by taking the points of a point process $$S$$ and $$y\prec x$$ when $$y$$ is in the causal past of $$x$$. The *Poisson sprinkling* of density $$\rho$$ has independent Poisson counts of mean $$\rho V(D)$$ in disjoint sets. A pair $$y\prec x$$ determines the causal diamond, or *interval*, $$I(y,x)$$, whose volume is $$V(y,x)=v_d\tau^d$$, with $$\tau$$ the proper time from $$y$$ to $$x$$, $$v_2=1/2$$ and $$v_4=\pi/24$$. The number $$N(y,x)$$ of points of $$S$$ in $$I(y,x)$$ is, for a Poisson sprinkling, Poisson distributed with mean $$\rho V(y,x)$$. These counts of interval points are the only data the operator below uses.

For a function $$\varphi$$ on the points, the Benincasa–Dowker operator is

$$
(B_\rho\varphi)(x)=\kappa_d\,\rho^{2/d}\Big[\alpha_0\,\varphi(x)+\sum_{n\ge0}c_n\sum_{y\prec x,\ N(y,x)=n}\varphi(y)\Big], \tag{1}
$$

with $$\kappa_2=4$$, $$\alpha_0=-\tfrac12$$, $$(c_0,c_1,c_2)=(1,-2,1)$$ in two dimensions, and $$\kappa_4=4/\sqrt6$$, $$\alpha_0=-1$$, $$(c_0,\dots,c_3)=(1,-9,16,-8)$$ in four [[BD]](#ref-bd), [[Sorkin 2007]](#ref-sorkin07), [[DG]](#ref-dg). The sum over $$n=0$$ runs over the nearest neighbours of $$x$$ (links), $$n=1$$ over elements with exactly one point between, and so on. The operator is retarded, Lorentz invariant, and non-local: it couples $$x$$ to elements arbitrarily far from $$x$$ along the light cone. We write $$f(V)=e^{-V}\sum_nc_nV^n/n!$$ for the function that the Poisson average of the weights produces, and $$\mathcal Mf(s)=\int_0^\infty f(V)V^{s-1}dV$$ for its Mellin transform. With $$\int_0^\infty e^{-V}V^{n+s-1}dV=\Gamma(s)(s)_n$$,

$$
\mathcal Mf(s)=-\tfrac43\,\Gamma(s)\,(s-\tfrac12)(s-1)(s-\tfrac32)\ \ (d=4),\qquad \mathcal Mf(s)=\tfrac12\,\Gamma(s)(s-1)(s-2)\ \ (d=2). \tag{2}
$$

(Checked by direct quadrature at $$s=0.7$$ ($$d=4$$) and $$s=0.6$$ ($$d=2$$) to 17 digits.) The zeros of (2) at $$s=\tfrac12,1,\tfrac32$$ and $$s=1,2$$ are the conditions on the coefficients that remove the terms of the mean that would otherwise diverge as $$\rho\to\infty$$ [[ASS]](#ref-ass).

*The mean at an element.* Let $$x$$ be an element of a sprinkling. For a Poisson process of density $$\lambda$$, conditioning on $$x\in S$$ leaves the remaining points Poisson of density $$\lambda$$ (the Mecke formula [[Last–Penrose]](#ref-lp)). The mean of (1) at $$x$$ is then, with the sprinkled points of density $$\lambda$$ and the coefficient set by a reference density $$\rho$$,

$$
\mathbb E\big[B_\rho\varphi(x)\mid x\in S\big]=\kappa_d\rho^{2/d}\Big[\alpha_0\varphi(x)+G_\lambda[\varphi](x)\Big],\qquad G_\lambda[\varphi](x)=\lambda\int_{J^-(x)}f\big(\lambda V(x,y)\big)\varphi(y)\,dy, \tag{3}
$$

where $$J^-(x)$$ is the causal past. The Benincasa–Dowker theorem is that (3) with $$\lambda=\rho$$ tends to $$\Box\varphi(x)$$ as $$\rho\to\infty$$ for suitable fields $$\varphi$$ [[BD]](#ref-bd), [[DG]](#ref-dg). We call this evaluation at an element *placement A*. In *placement B* the point $$x$$ is a fixed point of space that is not an element of the sprinkling: $$y$$ runs over the elements in the past of $$x$$, the intervals $$I(y,x)$$ are counted as before, and there is no conditioning. For a Poisson process of density $$\lambda$$ the mean has the same form (3), because the intensity of $$y$$ is $$\lambda$$ and the interval count is Poisson; the two placements differ for a non-Poisson process, where the pair $$(x,y)$$ carries a Palm weight only in A.

## 2. The Poisson limit, proved on exponential fields

To fix the normalisation and to have a field class on which every limit below is a computation, we prove the Benincasa–Dowker limit for the fields $$\varphi_E(y)=e^{E(t_y-t_x)}$$, $$E>0$$, which satisfy $$\Box\varphi_E=-E^2\varphi_E$$. By Lorentz invariance of the sprinkling and of (1), the same holds for $$e^{\kappa\cdot(y-x)}$$ with $$\kappa$$ any future timelike vector with $$-\kappa\cdot\kappa=E^2$$. For $$\varphi_E$$ the integral in (3) over the past of $$x$$ converges absolutely, and with $$z=x-y$$, $$d^4z=\tau^3d\tau\,d\mu_{H}$$ ($$\mu_H$$ the measure on the unit hyperboloid) and $$\int_{H}e^{-E\tau\cosh\eta}d\mu_H=4\pi K_1(E\tau)/(E\tau)$$, one finds

$$
G_\lambda[\varphi_E](x)=\tilde I(\eta)\ \ \text{with}\ \ \tilde I(\eta)=96\,\eta\int_0^\infty x^2f(\eta x^4)K_1(x)\,dx,\qquad \eta=\frac{\pi\lambda}{24E^4}, \tag{4}
$$

where $$K_1$$ is the modified Bessel function. The function $$\tilde I$$ is analytic in $$\mathrm{Re}\,\eta>0$$, because $$f(\eta x^4)$$ decays like $$e^{-\eta x^4}$$, with $$\tilde I(\eta)\to0$$ as $$\eta\to0$$ (for small $$\lambda$$, $$G\approx\lambda\int_{J^+}e^{-Ez^0}=8\pi\lambda/E^4$$). In two dimensions the same computation, with $$\int_He^{-E\tau\cosh\eta}d\mu_H=2K_0(E\tau)$$, gives $$G_\lambda[\varphi_E]=\tilde I_2(\eta)$$ with $$\eta=\lambda/(2E^2)$$ and $$\tilde I_2(\eta)=4\eta\int_0^\infty x\,f(\eta x^2)K_0(x)dx$$.

<div class="proposition" markdown="1">

**Proposition 1** (Poisson limit on exponential fields). For $$\eta\to\infty$$ in any closed sector $$\lvert\arg\eta\rvert\le\pi/2-\delta$$,

$$
\tilde I(\eta)=1-\frac{\sqrt\pi}{8}\,\eta^{-1/2}+O(\eta^{-1}\ln\lvert\eta\rvert), \tag{5}
$$

uniformly in the sector. Consequently, for $$\lambda=\rho$$ real, $$\mathbb E[B_\rho\varphi_E(x)\mid x\in S]\to-E^2=\Box\varphi_E(x)$$ as $$\rho\to\infty$$. In two dimensions $$\tilde I_2(\eta)=\tfrac12-\tfrac1{8\eta}+O(\eta^{-2}\ln\eta)$$ and the limit is again $$-E^2$$.

</div>

The idea is that $$f(\eta x^4)$$ confines the integral to $$x\lesssim\eta^{-1/4}$$, where $$K_1$$ can be replaced by its small-argument expansion; each power of $$x$$ then integrates against $$f$$ to a value of its Mellin transform, and the zeros in (2) kill every term except two.

<div class="proof" markdown="1">

*Proof.* The expansion of the Bessel function is

$$
K_1(x)=\frac1x+\frac x2\Big(\ln\frac x2+\gamma-\frac12\Big)+\frac{x^3}{16}\Big(\ln\frac x2+\gamma-\frac54\Big)+R(x),\qquad \lvert R(x)\rvert\le Cx^5(1+\lvert\ln x\rvert)\ \ (0<x\le1),
$$

with $$\gamma$$ Euler's constant. We need $$\int_0^\infty x^{a}f(\eta x^4)\,dx=\frac14\eta^{-(a+1)/4}\mathcal Mf(\tfrac{a+1}4)$$ and, differentiating in the exponent, $$\int_0^\infty x^a\ln x\,f(\eta x^4)\,dx=\frac1{16}\eta^{-(a+1)/4}\big[\mathcal M'f(\tfrac{a+1}4)-\ln\eta\,\mathcal Mf(\tfrac{a+1}4)\big]$$. These are first proved for real $$\eta>0$$ by the substitution $$V=\eta x^4$$, and then hold for complex $$\eta$$ in the right half plane by analytic continuation of both sides. Apply them to the three terms of the expansion multiplied by $$x^2$$:

- the term $$1/x$$ gives $$\frac14\eta^{-1/2}\mathcal Mf(\frac12)=0$$;
- the term in $$x$$ gives $$\frac12\cdot\frac1{16}\eta^{-1}\mathcal M'f(1)$$, the logarithm-free part being $$\propto\mathcal Mf(1)=0$$, so that $$96\eta$$ times it is $$3\mathcal M'f(1)=1$$, because $$\mathcal M'f(1)=\frac13$$ from (2);
- the term in $$x^3$$ gives $$\frac1{16}\cdot\frac1{16}\eta^{-3/2}\mathcal M'f(\frac32)$$, and $$96\eta$$ times it is $$\frac38\mathcal M'f(\frac32)\eta^{-1/2}=-\frac{\sqrt\pi}8\eta^{-1/2}$$, because $$\mathcal M'f(\frac32)=-\frac43\Gamma(\frac32)\cdot1\cdot\frac12=-\frac{\sqrt\pi}3$$.

The remainder is bounded by $$C\int x^7(1+\lvert\ln x\rvert)\lvert f(\eta x^4)\rvert dx=O(\eta^{-2}\ln\lvert\eta\rvert)$$, using $$\lvert f(\eta x^4)\rvert\le C(1+\lvert\eta\rvert x^4)^3e^{-\cos\delta\,\lvert\eta\rvert x^4}$$ in the sector, and the part of the integral over $$x>1$$ is $$O(e^{-c\lvert\eta\rvert})$$. This is (5). Then, with $$\eta=\pi\rho/(24E^4)$$,

$$
\kappa_4\rho^{1/2}\big(-1+\tilde I(\eta)\big)=-\frac4{\sqrt6}\,\rho^{1/2}\frac{\sqrt\pi}8\Big(\frac{24E^4}{\pi\rho}\Big)^{1/2}+o(1)=-E^2+o(1).
$$

In two dimensions $$K_0(x)=-(\ln\frac x2+\gamma)+\frac{x^2}4(1-\gamma-\ln\frac x2)+O(x^4\ln x)$$, $$\mathcal Mf(1)=\mathcal Mf(2)=0$$, $$\mathcal M'f(1)=-\frac12$$, $$\mathcal M'f(2)=\frac12$$, and the same steps give $$\tilde I_2=\frac12-\frac1{8\eta}+O(\eta^{-2}\ln\eta)$$, so that $$4\rho(-\frac12+\tilde I_2)=-\rho/(2\eta)=-E^2$$ with $$\eta=\rho/2E^2$$. $$\square$$

</div>

*Check.* We evaluated (4) by quadrature at $$E=1$$ with 30-digit arithmetic. At $$\eta=10^2,10^4,10^6$$ the difference between $$\tilde I(\eta)$$ and the two terms of (5), multiplied by $$\eta$$, is $$0.121,\ 0.192,\ 0.265$$, growing like $$\ln\eta$$ as the error term in (5) allows; in two dimensions the same quantity multiplied by $$\eta^2$$ is $$0.170,\ 0.313,\ 0.424$$. The values of $$\mathbb E[B_\rho\varphi_E(x)\mid x\in S]$$ at $$\rho=10^2,10^4,10^6,10^8$$ are $$-0.88575$$, $$-0.98001$$, $$-0.99711$$, $$-0.99955$$, converging to $$-E^2=-1$$ at the rate $$O(\rho^{-1/2}\ln\rho)$$ that (5) predicts. In two dimensions the corresponding values at $$\rho=10^2,10^4,10^6$$ are $$-0.97625$$, $$-0.99953$$, $$-0.999993$$.

## 3. Mixed Poisson substrates

A *mixed Poisson* process is a point process that is Poisson of random density $$\lambda$$, the same at every point, with $$\lambda$$ drawn from a *mixing law* $$m$$; this is the Cox process of the quantum-sprinkling classification, where $$\lambda=\lvert z\rvert^2$$. Its correlation functions are $$\rho_n=\int\lambda^ndm(\lambda)$$ and its count generating function is $$\mathbb E\,u^{N(D)}=\int e^{(u-1)\lambda V(D)}dm(\lambda)$$. We admit mixing laws that are not positive: a finite complex measure or a finite-order distribution with compact support in $$\{\mathrm{Re}\,\lambda>0\}$$ defines, through these formulas, a complex point process in the sense of the page on quantum sprinklings (the growth condition on $$\rho_n$$ needed for existence is trivial for compact support, see Theorem 7). The pairing of $$m$$ with a function $$h$$ analytic near the support is denoted $$\langle m,h\rangle$$. We take $$\langle m,1\rangle=1$$ (normalisation) and write $$\bar\rho=\langle m,\lambda\rangle$$ for the mean density.

At placement A the mean of (1) is, by the Palm computation of section 1 applied to the correlation functions $$\rho_{2+j}=\langle m,\lambda^{2+j}\rangle$$,

$$
\mathbb E_A\big[B_\rho\varphi(x)\big]=\kappa_d\rho^{2/d}\Big[\alpha_0\varphi(x)+\frac1{\bar\rho}\big\langle m,\lambda\,G_\lambda[\varphi](x)\big\rangle\Big], \tag{6}
$$

since the pair $$(x,y)$$ is weighted by $$\rho_2=\langle m,\lambda^2\rangle$$ and divided by $$\rho_1=\bar\rho$$. At placement B it is $$\kappa_d\rho^{2/d}[\alpha_0\varphi(x)+\langle m,G_\lambda[\varphi](x)\rangle]$$. The coefficient in (1) is set by $$\rho$$, which we take equal to the mean density $$\bar\rho$$: the operator is built knowing only the mean.

### 3.1 Exact scaling

<div class="theorem" markdown="1">

**Theorem 2** (a different density is a dilated field). For every bounded measurable $$\varphi$$ with bounded support, every $$\rho,\lambda>0$$ and $$s=(\rho/\lambda)^{1/d}$$,

$$
G_\lambda[\varphi](x)=G_\rho[\varphi_s](x),\qquad \varphi_s(y)=\varphi\big(x+s(y-x)\big). \tag{7}
$$

Consequently the mean of (1) over a Poisson sprinkling of density $$\lambda$$, with coefficient set by $$\rho$$, equals the mean over a Poisson sprinkling of density $$\rho$$ acting on the field $$\varphi_s$$ dilated about $$x$$.

</div>

<div class="proof" markdown="1">

*Proof.* Substitute $$y=x+s\,(y'-x)$$ in the integral of (3). Then $$dy=s^d\,dy'$$ and $$V(x,y)=s^dV(x,y')$$, because volumes scale with the $$d$$th power of lengths. Since $$\lambda s^d=\rho$$, the integrand becomes $$\rho f(\rho V(x,y'))\,\varphi(x+s(y'-x))\,dy'$$, which is the integrand of $$G_\rho[\varphi_s]$$. $$\square$$

</div>

The identity says that a Poisson sprinkling of density $$\lambda\ne\rho$$ read with the operator of density $$\rho$$ sees the field through a conformal rescaling by $$s$$, and since $$\Box\varphi_s=s^2(\Box\varphi)_s$$, the continuum limit of its mean is $$(\rho/\lambda)^{2/d}\Box\varphi$$ whenever the Poisson limit holds for $$\varphi_s$$. For exponential fields this is explicit: $$G_\lambda[\varphi_E]=G_\rho[\varphi_{sE}]$$, which is (4) depending on $$\lambda/E^4$$ only.

### 3.2 The limit and its sign

<div class="theorem" markdown="1">

**Theorem 3** (normalisation of the operator on a mixed Poisson process). Let $$d\in\{2,4\}$$ and let $$m_1$$ be a mixing law with $$\langle m_1,1\rangle=\langle m_1,\theta\rangle=1$$, and for each $$\rho$$ let $$m_\rho$$ be the law of $$\lambda=\rho\theta$$, so that $$\bar\rho=\rho$$. Assume either

(M1) $$m_1$$ is a finite-order distribution with compact support in $$\{\mathrm{Re}\,\theta>0\}$$ (this includes finite complex combinations of point masses and of their derivatives), or

(M2) $$m_1$$ is a positive probability measure on $$(0,\infty)$$ of mean 1.

Then for the field $$\varphi_E$$, as $$\rho\to\infty$$:

$$
\mathbb E_A\big[B_\rho\varphi_E(x)\big]\to C_A\,\Box\varphi_E(x),\quad C_A=\begin{cases}\langle m_1,\theta^{1/2}\rangle,&d=4,\\ 1,&d=2,\end{cases}
$$

and, if in case (M2) $$\int\theta^{-2/d}dm_1<\infty$$,

$$
\mathbb E_B\big[B_\rho\varphi_E(x)\big]\to C_B\,\Box\varphi_E(x),\quad C_B=\langle m_1,\theta^{-2/d}\rangle .
$$

Powers of $$\theta$$ are principal branches. No mass term, logarithm or other correction survives in the limit.

</div>

The idea is that (6), (7) and Proposition 1 turn the mean into an average over the mixing law of the Poisson means of the dilated fields; the $$\alpha_0$$ term cancels against the constant in the Poisson limit because $$\langle m,\lambda\rangle=\rho$$, and what is left is the average of $$\theta^{-2/d}$$ weighted by the Palm factor $$\theta$$.

<div class="proof" markdown="1">

*Proof.* Set $$H_\rho(\theta)=\rho^{1/2}\big(\tilde I(\rho\theta c_E)-1\big)$$ with $$c_E=\pi/(24E^4)$$ in four dimensions. By (6), $$\alpha_0=-1$$ and $$\bar\rho=\rho$$, and writing $$\lambda=\rho\theta$$,

$$
\mathbb E_A[B_\rho\varphi_E(x)]=\kappa_4\,\big\langle m_1,\ \theta\,H_\rho(\theta)\big\rangle,\qquad \mathbb E_B[B_\rho\varphi_E(x)]=\kappa_4\,\langle m_1,H_\rho\rangle,
$$

because $$-1=-\langle m_1,\theta\rangle$$ and $$-1=-\langle m_1,1\rangle$$ respectively. By Proposition 1, $$H_\rho(\theta)\to H_\infty(\theta)=-\frac{\sqrt\pi}8\big(\frac{24E^4}{\pi\theta}\big)^{1/2}=-\frac{\sqrt{24}}8E^2\theta^{-1/2}$$, locally uniformly on $$\{\mathrm{Re}\,\theta>0\}$$ (compact subsets lie in a closed sector, and $$\rho\theta c_E\to\infty$$ in it). The functions $$H_\rho$$ are analytic, so the derivatives of all orders converge locally uniformly as well. Under (M1) the pairing with $$m_1$$ involves finitely many derivatives on a compact set, so the limit may be taken inside. Under (M2) we use dominated convergence: $$\tilde I$$ is continuous on $$(0,\infty)$$, tends to 0 at 0 and satisfies (5) at infinity, so there is $$C$$ with $$\lvert\tilde I(\eta)-1\rvert\le C\min(1,\eta^{-1/2})$$ for $$\eta>0$$ and hence $$\lvert H_\rho(\theta)\rvert\le C'\theta^{-1/2}$$ uniformly in $$\rho\ge1$$: for $$\rho\theta c_E\ge1$$ use the decay, and otherwise $$\rho^{1/2}\le(c_E\theta)^{-1/2}$$. The dominating function $$\theta\cdot\theta^{-1/2}=\theta^{1/2}$$ is integrable against $$m_1$$ since $$\int\theta^{1/2}dm_1\le(\int\theta\,dm_1)^{1/2}=1$$. For placement B the dominating function is $$\theta^{-1/2}$$. With $$\kappa_4=4/\sqrt6$$, $$\kappa_4\cdot(-\frac{\sqrt{24}}8E^2)=-E^2$$, which gives $$C_A=\langle m_1,\theta\cdot\theta^{-1/2}\rangle=\langle m_1,\theta^{1/2}\rangle$$ and $$C_B=\langle m_1,\theta^{-1/2}\rangle$$.

In two dimensions the same argument with $$\tilde I_2$$ and $$H_\rho(\theta)=\rho(\tilde I_2(\rho\theta/2E^2)-\frac12)\to-\frac{E^2}{4\theta}$$ and $$\kappa_2=4$$ gives $$\mathbb E_A\to-E^2\langle m_1,\theta\cdot\theta^{-1}\rangle=-E^2$$ and $$\mathbb E_B\to-E^2\langle m_1,\theta^{-1}\rangle$$. $$\square$$

</div>

<div class="corollary" markdown="1">

**Corollary 4** (positive mixtures are mis-normalised in four dimensions). Under (M2) with $$d=4$$ and $$m_1\ne\delta_1$$: $$C_A<1$$ and, when finite, $$C_B>1$$. In $$d=2$$, $$C_A=1$$ and $$C_B>1$$.

</div>

<div class="proof" markdown="1">

*Proof.* Jensen's inequality for the strictly concave $$\theta^{1/2}$$ and the strictly convex $$\theta^{-1/2}$$, $$\theta^{-1}$$, with $$\mathbb E\theta=1$$. $$\square$$

</div>

The mixed Poisson states of [Lorentz-invariant quantum sprinklings](/quantum-gravity/quantum-sprinklings/), which have a random constant intensity $$\Lambda=\rho\Theta$$, are covered by (M2) with $$\mathrm{Law}(\Theta)=m_1$$, and so are mis-normalised by the factor $$\mathbb E\Theta^{1/2}<1$$ unless $$\Lambda$$ is deterministic. Two readings of this factor are consistent with the statements above. If the coefficient $$\rho^{2/d}$$ in (1) is a fixed constant of the theory, the ensemble-averaged operator is $$C_A\Box$$, not $$\Box$$. If, on the other hand, the coefficient could be set from the realised density of the same sprinkling, each realisation would give $$\Box$$, but that density is a non-local functional of the causal set (an average over a large region), and the operator would no longer be a local function of the neighbourhood of $$x$$. The theorem uses the first reading; the second is listed among the routes in section 7.

*Checks.* (i) Positive law: atoms $$\theta=(0.25,1.75)$$ with weights $$(\frac12,\frac12)$$ have $$\langle\theta^{1/2}\rangle=0.91144$$ and $$\langle\theta^{-1/2}\rangle=1.37796$$. The exact mean (6), computed by quadrature, gives $$C_A=0.89223,0.90863,0.91098$$ and $$C_B=1.33725,1.37186,1.37708$$ at $$\rho=10^4,10^6,10^8$$. (ii) In two dimensions, $$C_A$$ for the Poisson law, the three-atom law of Example 1 and the two-atom law above is $$0.97625,\ 0.97177,\ 0.95821$$ at $$\rho=10^2$$ and $$0.999993,\ 0.999991,\ 0.999986$$ at $$\rho=10^6$$, all tending to 1.

### 3.3 Signed mixing laws, and what a Born reading forces

Positivity of $$m$$ is not needed for Theorem 3, so signed laws can have $$C_A=1$$ in four dimensions: the condition $$\langle m_1,\theta^{1/2}\rangle=1$$ is a single real equation.

<div class="example" markdown="1">

**Example 1** (a signed mixing law with the correct normalisation). Take atoms $$\theta_i=(0.25,\,1.44,\,4)$$, so $$\sqrt{\theta_i}=(0.5,1.2,2)$$, and solve $$\sum a_i=\sum a_i\theta_i=\sum a_i\sqrt{\theta_i}=1$$. The solution is $$a=(4/21,\ 25/28,\ -1/12)=(0.190476,\ 0.892857,\ -0.083333)$$, with total variation $$S=\sum\lvert a_i\rvert=7/6$$. One weight is negative, as it must be: by Corollary 4 a positive law with $$C_A=1$$ is a point mass. By Theorem 3, $$C_A=1$$ exactly, with no other correction. The exact mean (6) gives $$C_A=0.88657,\ 0.98011,\ 0.99712,\ 0.99956$$ at $$\rho=10^2,10^4,10^6,10^8$$, and placement B gives $$C_B=\langle\theta^{-1/2}\rangle=1.08333$$ (computed $$1.08276$$ at $$\rho=10^8$$). The process is a signed combination of three Poisson processes; each of them is positive, so it is a complex point process of bounded total variation, and by Theorem 10 of the page on quantum sprinklings it is a phase mixture of classical processes (here with weights of both signs).

</div>

What the example does not have is a probabilistic reading. The next statement covers every mixed Poisson process whose counts could be probabilities.

<div class="proposition" markdown="1">

**Proposition 5** (a Born reading forces a positive mixing law). Let $$m$$ be a complex measure or finite-order distribution with compact support in $$\mathbb C$$ and $$\langle m,1\rangle=1$$, and let $$P$$ be the mixed Poisson complex point process with $$\rho_n=\langle m,\lambda^n\rangle$$. Suppose that $$\Pr(N(D)=k)$$ is a real number in $$[0,1]$$ for every bounded $$D$$ and every $$k\ge0$$. Then there is a positive probability measure $$\mu$$ on $$[0,\infty)$$ with $$\langle\mu,\lambda^n\rangle=\langle m,\lambda^n\rangle$$ for all $$n$$. Hence $$P$$ is the classical Cox process with mixing law $$\mu$$, its correlation functions and all of its means (6) are those of $$\mu$$, and when $$\mu$$ satisfies (M2), $$C_A\le1$$ in $$d=4$$ with equality if and only if $$P$$ is Poisson.

</div>

<div class="proof" markdown="1">

*Proof.* Let $$L(V)=\langle m,e^{-\lambda V}\rangle$$, an entire function of $$V$$. The count probabilities are $$\Pr(N(D)=k)=\langle m,e^{-\lambda V}(\lambda V)^k\rangle/k!=(-V)^kL^{(k)}(V)/k!$$ with $$V=V(D)$$, and $$V(D)$$ takes every value in $$(0,\infty)$$. The hypothesis that all of them are $$\ge0$$ says $$(-1)^kL^{(k)}\ge0$$ on $$(0,\infty)$$, that is, $$L$$ is completely monotone. By the Bernstein–Widder theorem [[Widder]](#ref-widder) $$L(V)=\int_{[0,\infty)}e^{-\lambda V}d\mu(\lambda)$$ for a positive measure $$\mu$$, of total mass $$L(0^+)=\langle m,1\rangle=1$$. Since $$L$$ is entire, $$(-1)^nL^{(n)}(0)=\langle m,\lambda^n\rangle$$, and $$(-1)^nL^{(n)}(0^+)=\int\lambda^nd\mu$$ by monotone convergence, so the moments agree. The correlation functions determine the complex point process (Theorem 7), and the means (6) depend on $$m$$ only through them. The last statement is Corollary 4. $$\square$$

</div>

The proof shows that the mixing "law" of a mixed Poisson process is determined only through its holomorphic moments: adding to $$m$$ a derivative $$\partial_{\bar\lambda}\nu$$ of any compactly supported distribution $$\nu$$ changes nothing, since $$\langle\partial_{\bar\lambda}\nu,h\rangle=0$$ for holomorphic $$h$$. The three-atom law of Example 1 has $$\mathrm{Var}_m(\lambda)=\sum a_i\theta_i^2-1=-0.47$$, so it is not the law of a random variable, and its count probabilities are not all non-negative: the smallest over $$k\le39$$ and $$V\in\{0.1,\dots,39.9\}$$ is $$-0.0087$$, at $$k=11$$, $$V=2.6$$ (for $$\rho=1$$). A Born reading needs a state and a positive operator-valued measure, and every probability it assigns to an event lies in $$[0,1]$$; the negative value excludes it.

*Weak values.* The signed process can be given an operational reading in which the counts are weak values after post-selection. For a complex measure $$\mathsf P_D$$ with density $$u=d\mathsf P_D/d\lvert\mathsf P_D\rvert$$ one has $$\mathsf P_D(A)=\langle\psi_f,E(A)\psi_i\rangle/\langle\psi_f,\psi_i\rangle$$ on $$L^2(\lvert\mathsf P_D\rvert)$$ with $$E$$ the multiplication projection-valued measure, $$\psi_f=\lVert\mathsf P_D\rVert^{-1/2}$$ and $$\psi_i=\lVert\mathsf P_D\rVert^{-1/2}u$$. For any representation of this kind with unit vectors $$\psi_f,\psi_i$$ and a positive-operator-valued measure $$E_j$$, Cauchy–Schwarz gives $$\sum_j\lvert\langle\psi_f,E_j\psi_i\rangle\rvert\le1$$; since $$\lVert\mathsf P_D\rVert=\sup\sum_j\lvert\mathsf P_D(A_j)\rvert$$ over partitions, the post-selection probability $$\lvert\langle\psi_f,\psi_i\rangle\rvert^2$$ is at most $$\lVert\mathsf P_D\rVert^{-2}$$, and the construction attains it. For Example 1 the total variation is $$\lVert\mathsf P_D\rVert=\sum_n\lvert\sum_ia_i\mathrm{Poi}_{\lambda_iV}(n)\rvert\le S$$ (given $$N=n$$ the configuration is uniform in $$D$$ for every atom), so the post-selection probability is at least $$1/S^2=36/49=0.735$$ for every region. For a mixing law with a non-real atom, $$\lVert\mathsf P_D\rVert$$ grows exponentially: for a finite atomic law with $$\kappa=\max_j(\lvert\lambda_j\rvert-\mathrm{Re}\lambda_j)>0$$ one has $$\limsup_Ve^{-\kappa V}\lVert\mathsf P_D\rVert>0$$, because $$\lVert\mathsf P_D\rVert\ge\lvert\mathbb Eu^N\rvert$$ for $$\lvert u\rvert=1$$ and the atoms that attain $$\kappa$$ at the maximising phase give an almost periodic sum whose limsup is at least $$(\sum\lvert a_j\rvert^2)^{1/2}>0$$ by the Bohr–Parseval identity. So the weak-value cost of a complex mixing law is exponential in the volume.

### 3.4 The Feynman kernel as a distributional mixture

A determinantal complex process with kernel $$K(x,y)=\rho g/(s^2/\ell^2-i)$$ for $$x\ne y$$ and $$K(x,x)=\rho$$, where $$s^2$$ is the squared interval (positive for timelike separation), has correlation functions $$\rho_n=\det[K(x_i,x_j)]$$ that depend only on the invariant intervals and are bounded by $$(\rho\max(1,g))^nn^{n/2}$$ (Hadamard), so Theorem 7 gives a Poincaré-invariant complex point process. It is the process with the Feynman-propagator form of two-point amplitude regularised at the length $$\ell$$.

<div class="proposition" markdown="1">

**Proposition 6** (local form). Let $$\rho\to\infty$$ at fixed $$\ell$$ and $$g$$. On every region of diameter $$\ll\ell$$ the correlation functions are those of the distributional mixed Poisson process with

$$
m=\delta_{\lambda_0}-i\rho g\,\delta'_{\lambda_0},\qquad \lambda_0=\rho(1-ig), \tag{8}
$$

up to corrections of relative order $$\ell^{-2}\times(\text{diameter})^2$$. For the field $$\varphi_E$$ in four dimensions, (8) has $$\langle m_1,1\rangle=1$$, $$\langle m_1,\theta\rangle=1$$ and

$$
C_A=\frac{1-ig/2}{(1-ig)^{1/2}},\quad C_B\ \text{as in Theorem 3}.
$$

</div>

<div class="proof" markdown="1">

*Proof.* In a region of diameter $$r\ll\ell$$ the kernel is $$K(x,y)=i\rho g\,(1+is^2/\ell^2)^{-1}\approx i\rho g$$ off the diagonal, and $$K(x,x)=\rho$$. The matrix $$[K(x_i,x_j)]$$ is then $$\rho(1-ig)\mathbb 1+i\rho g\,\mathbf J$$ with $$\mathbf J$$ the all-ones matrix, whose determinant is $$\lambda_0^{n-1}(\lambda_0+i\rho gn)$$ with $$\lambda_0=\rho(1-ig)$$. Thus $$\rho_n=\lambda_0^n+i\rho g\,n\lambda_0^{n-1}$$. For the distribution (8), $$\langle\delta'_{\lambda_0},h\rangle=-h'(\lambda_0)$$, so $$\langle m,\lambda^n\rangle=\lambda_0^n+i\rho g\,n\lambda_0^{n-1}$$, the same. The sum in (1) probes intervals with $$\tau^4\sim1/\rho$$, so every separation that enters satisfies $$\lvert s^2\rvert/\ell^2\lesssim(\rho\ell^4)^{-1/2}\to0$$. For the value of $$C_A$$, $$\langle m_1,\theta^{1/2}\rangle=(1-ig)^{1/2}+\frac{ig}2(1-ig)^{-1/2}$$ gives the formula. $$\square$$

</div>

This is a derivation in the stated limit, not a statement about the exact process at finite $$\rho\ell^4$$. For $$g=1$$ the formula gives $$C_A=0.93779-0.06665i$$. The exact mean (6) for (8) (a distribution, evaluated by differentiating the analytic function $$\lambda G_\lambda$$ at $$\lambda_0$$) gives $$0.91809-0.06610i$$ at $$\rho=10^4$$ and $$0.93485-0.06650i$$ at $$\rho=10^6$$, approaching it. The correction $$C_A-1$$ is complex, and a complex normalisation of the operator is not a rescaling of the continuum d'Alembertian. The count probabilities are those of (8): $$\Pr(N(D)=0)=e^{-\mu}e^{ix}(1-ix)$$ with $$\mu=\rho V$$ and $$x=g\mu$$, whose imaginary part $$e^{-\mu}(\sin x-x\cos x)$$ is positive for $$0<x<4.4934$$ (the first positive root of $$\tan x=x$$), and $$\mathrm{Re}\Pr(N=3)=-0.3282$$ at $$g=1$$, $$\mu=2$$. No Born reading is possible there either.

## 4. Signed and complex invariant processes

A complex point process is determined by its correlation functions, and existence needs only a growth bound. The statement is elementary, but the family it describes is the one used in section 5, so we prove it.

<div class="theorem" markdown="1">

**Theorem 7** (realisability without positivity). Let $$\rho_0=1$$ and let $$\rho_n$$, $$n\ge1$$, be symmetric measurable functions on $$M^n$$ with $$\lvert\rho_n\rvert\le C^nn^{n/2}$$ for some $$C$$. Then there is a unique simple complex point process $$\{\mathsf P_D\}$$ whose Janossy densities are

$$
j_n^D(x_1,\dots,x_n)=\sum_{m\ge0}\frac{(-1)^m}{m!}\int_{D^m}\rho_{n+m}(x,y)\,dy,
$$

and whose correlation functions are $$\rho_n$$. If the $$\rho_n$$ are Poincaré invariant, so is the process.

</div>

<div class="proof" markdown="1">

*Proof.* The series converges absolutely: $$\sum_m\lvert D\rvert^mC^{n+m}(n+m)^{(n+m)/2}/m!<\infty$$ because $$k^{k/2}/k!$$ decays faster than any exponential. With $$k=n+m$$ and $$\binom kn$$ counting the ways to split, $$\sum_n\frac1{n!}\int\lvert j_n\rvert\le\sum_k\frac{(C\lvert D\rvert)^kk^{k/2}}{k!}\sum_{n}\binom kn=\sum_k\frac{(2C\lvert D\rvert)^kk^{k/2}}{k!}<\infty$$, so $$\mathsf P_D$$ has finite total variation. Normalisation: $$\sum_n\frac1{n!}\int j_n=\sum_k\frac1{k!}\int\rho_k\sum_n\binom kn(-1)^{k-n}=\rho_0=1$$ (only $$k=0$$ survives). For $$D\subset D'$$, restricting the configuration of $$D'$$ to $$D$$ integrates out $$r$$ points in $$D'\setminus D$$; the coefficient of the terms with $$r$$ such points is $$\frac1{r!}\sum_j\binom rj(-1)^j=\delta_{r0}$$, which is consistency. The same identity inverts the formula and returns $$\rho_n=\sum_m\frac1{m!}\int j_{n+m}$$. Uniqueness: the $$j_n$$ determine $$\mathsf P_D$$. Invariance of the $$\rho_n$$ gives invariance of $$j_n^D$$ with $$D$$ transported. $$\square$$

</div>

So non-positive invariant correlations exist in abundance, for example $$\rho_1=\rho$$, $$\rho_2=\rho^2(1+h(s^2))$$ and $$\rho_n=\rho^n$$ for $$n\ge3$$ with any bounded $$h$$; the obstruction of the classification of quantum sprinklings (Bochner positivity forcing a point measure) has no analogue at the level of correlation functions.

*Variance.* The simplest observable of such a process is the number variance of a causal diamond $$D_T$$ of height $$T$$ and volume $$V=\pi T^4/24$$. Write $$\rho_2(x,y)-\rho^2=c(s^2(x-y))$$ (Poincaré invariance makes the left side a function of the squared interval, and the symmetry of $$\rho_2$$ makes it even), with $$c\in L^1(\mathbb R)$$, and let $$a$$ be the density of $$s^2$$ over ordered pairs of the unit diamond, $$\int\!\!\int_{D_1^2}F(s^2)dx\,dy=\int F(\sigma)a(\sigma)d\sigma$$.

<div class="proposition" markdown="1">

**Proposition 8** (the $$V^{3/2}$$ law). The function $$a$$ is bounded and continuous at $$0$$ with $$a(0)=\pi^2/288=2(\pi/24)^2$$. For the invariant complex process above,

$$
\mathrm{Var}\,N(D_T)=\rho V+T^6\!\int c(\sigma)\,a(\sigma/T^2)\,d\sigma=\rho V+a(0)K\,T^6+o(T^6),\qquad K=\int c(\sigma)d\sigma, \tag{9}
$$

with $$V=\pi T^4/24$$ the geometric volume, so that $$a(0)T^6=\frac{\pi^2}{288}\big(\frac{24V}{\pi}\big)^{3/2}\approx0.7236\,V^{3/2}$$. In particular, for an invariant process with integrable pair amplitude $$c$$ and $$K\ne0$$, the variance exceeds the Poisson value by a term of order $$V^{3/2}$$.

</div>

<div class="proof" markdown="1">

*Proof.* The first equality in (9) is $$\mathrm{Var}N=\int_D\rho_1+\int\!\!\int_{D^2}(\rho_2-\rho^2)$$ and scaling $$D_T=TD_1$$ gives $$\int\!\!\int_{D_T^2}c(s^2)=T^8\!\int\!\!\int_{D_1^2}c(T^2s^2)=T^6\int c(\sigma)a(\sigma/T^2)d\sigma$$. For fixed $$x\in D_1$$ the density of $$s^2=\sigma>0$$ over $$y\in D_1$$ is $$\sum_\pm\int d^3u\,\mathbb 1[\dots]/(2\sqrt{\lvert u\rvert^2+\sigma})\le\int_{\lvert u\rvert<1}d^3u/\lvert u\rvert$$, and for $$\sigma=-\varrho^2<0$$ it is $$2\pi\int d\Delta t\sqrt{\Delta t^2+\varrho^2}\,\mathbb 1[\dots]$$, both bounded and continuous at $$\sigma=0$$ by dominated convergence. Dominated convergence in (9) then gives the limit with $$a(0)=\int_{D_1}dx\int dy\,\delta(s^2)\mathbb 1_{D_1}(y)$$. For $$x=(t,r)$$ the null rays from $$x$$ leave the diamond after the parameters $$\tau_\pm=(A_\pm^2-r^2)/(2(A_\pm+r\mu))$$, with $$A_\pm=\frac12\mp t$$ and $$\mu$$ the cosine of the direction, and $$\int d^4y\,\delta(s^2)\mathbb 1=\sum_\pm\int d\Omega\,\tau_\pm^2/4=\sum_\pm\frac\pi4(A_\pm^2-r^2)$$ after integrating over $$\mu$$ ($$\int_{-1}^1d\mu\,(A+r\mu)^{-2}=2/(A^2-r^2)$$). Integrating over $$D_1$$ (with $$\int_{D_1}=\int dt\int4\pi r^2dr$$, $$r<\frac12-\lvert t\rvert$$) gives $$\frac\pi4\int_{D_1}[(\frac12-t)^2+(\frac12+t)^2-2r^2]d^4x=\pi^2/288$$, confirmed symbolically. $$\square$$

</div>

*Check.* We drew $$10^8$$ pairs uniformly in the unit diamond and measured the fraction with $$\lvert s^2\rvert<\epsilon$$; it equals $$2\epsilon a(0)/V_1^2$$ for small $$\epsilon$$. The estimate of $$a(0)/V_1^2$$ is $$1.985,\ 1.996,\ 2.000\pm0.0014$$ at $$\epsilon=0.02,0.01,0.005$$, against the exact $$2$$.

The coefficient $$K=\int c$$ is the integral of the pair amplitude over the squared interval. It vanishes, for example, for any kernel whose amplitude $$c(\sigma)$$ extends to a function holomorphic and decaying in the lower half $$\sigma$$-plane: closing the contour gives $$\int c=0$$. The Feynman kernel above is of this type, since $$c(\sigma)=-\rho^2g^2/(\sigma/\ell^2-i)^2$$ has its pole in the upper half plane, and $$\int c\,d\sigma=-\rho^2g^2\ell^2\int dw\,(w-i)^{-2}=0$$ exactly, because the antiderivative $$-1/(w-i)$$ vanishes at $$\pm\infty$$. For such processes the $$V^{3/2}$$ term in (9) is absent; (9) does not determine the next term.

## 5. Pair and finite-order amplitudes

The simplest invariant complex process that is not a mixture is the *pair family*: $$\rho_1=\rho$$, $$\rho_2(x,y)=\rho^2(1+h(x,y))$$ with $$h=h(s^2/\ell^2)$$ bounded and invariant, and $$\rho_n=\rho^n$$ for $$n\ge3$$. More generally the *finite-order family* has $$\rho_k=\rho^k(1+h_k)$$ for $$2\le k\le M$$, with $$h_k$$ bounded, symmetric and Poincaré invariant, and $$\rho_k=\rho^k$$ for $$k>M$$. We write $$h_\pm$$ for the restriction of a pair amplitude to timelike ($$s^2>0$$) and spacelike separations.

<div class="proposition" markdown="1">

**Proposition 9** (Janossy densities, total variation, Born reading). For the pair family on a bounded region $$D$$, with $$V=\rho\lvert D\rvert$$:

(a) $$j_n=\rho^ne^{-V}$$ for $$n\ge3$$, $$j_2=\rho^2(e^{-V}+h)$$, $$j_1(x)=\rho e^{-V}-\rho^2\int_Dh(x,y)dy$$, $$j_0=e^{-V}+\frac12\rho^2\int\!\!\int_{D^2}h$$.

(b) With $$I=\rho^2\int\!\!\int_{D^2}\lvert h\rvert$$: $$\frac12I-1\le\lVert\mathsf P_D\rVert\le1+2I$$. For diamonds and $$h\in L^1(\mathbb R,dw)$$ with $$w=s^2/\ell^2$$, $$I=\rho^2\ell^2T^6\int\lvert h(w)\rvert\,a(w\ell^2/T^2)dw\to a(0)\rho^2\ell^2\lVert h\rVert_1T^6$$, so $$\lVert\mathsf P_D\rVert=\Theta(T^6)=\Theta(V_{\rm geom}^{3/2})$$ for every $$h\not\equiv0$$.

(c) If $$h\not\equiv0$$, $$\mathsf P_D$$ is not a positive measure for all large $$D$$: no Born reading exists.

</div>

<div class="proof" markdown="1">

*Proof.* (a) is Theorem 7: only terms with $$n+m=2$$ differ from the Poisson ones. (b) For the upper bound, the Poisson parts have total variation 1 and the perturbations contribute at most $$\frac12\rho^2\!\int\!\!\int\lvert h\rvert+\rho^2\!\int\!\!\int\lvert h\rvert+\frac12\rho^2\!\int\!\!\int\lvert h\rvert=2I$$. For the lower bound, $$\frac12\int\!\!\int\lvert j_2\rvert\ge\frac12(I-\rho^2\lvert D\rvert^2e^{-V})\ge\frac12I-\frac12$$, using $$V^2e^{-V}\le4e^{-2}<1$$. The asymptotics are dominated convergence as in Proposition 8. (c) Positivity of $$\mathsf P_D$$ would need $$j_2\ge0$$ almost everywhere, hence $$h\ge-e^{-V}$$, and for $$D$$ large $$h\ge0$$ almost everywhere. Then $$j_1(x)\ge0$$ needs $$\rho\int_Dh(x,y)dy\le e^{-V}$$ for almost every $$x\in D$$. But $$\int_Dh(x,y)dy$$ increases with $$D$$ to $$\int_Mh>0$$ while $$e^{-V}\to0$$. $$\square$$

</div>

### 5.1 What the operator sees

<div class="proposition" markdown="1">

**Proposition 10** (the operator sees only the timelike pair amplitude). For the pair family and every choice of layer weights $$(c_n)$$ in (1):

(a) At placement A,

$$
\mathbb E_A\big[B_\rho\varphi(x)\big]=\mathbb E_A^{\rm Poisson}\big[B_\rho\varphi(x)\big]+\kappa_d\rho^{2/d+1}\,c_0\int_{J^-(x)}h_+\!\Big(\frac{\tau^2(x,y)}{\ell^2}\Big)\varphi(y)\,dy. \tag{10}
$$

(b) At placement B,

$$
\mathbb E_B\big[B_\rho\varphi(x)\big]=\mathbb E_B^{\rm Poisson}\big[B_\rho\varphi(x)\big]+\kappa_d\rho^{2/d+2}(c_1-c_0)\int_{J^-(x)}\varphi(y)\int_{I(y,x)}h_+\!\Big(\frac{\tau^2(y,z)}{\ell^2}\Big)dz\,dy. \tag{11}
$$

(c) If $$h_+=0$$ almost everywhere, both means equal the Poisson means at every $$\rho$$. If $$h_+\ne0$$ on a set of positive measure, then at fixed $$\ell$$ the placement-A mean does not have a limit as $$\rho\to\infty$$ for all $$\varphi\in C_c^\infty$$: the extra term is $$\rho^{2/d+1}$$ times a $$\rho$$-independent functional that is non-zero for some $$\varphi$$ and $$x$$.

</div>

The idea is that the weights depend on a single interval count, and the only correlation function that enters the Palm weighting and is not Poisson is $$\rho_2$$ of a pair that is causally related, hence timelike.

<div class="proof" markdown="1">

*Proof.* (a) The Palm weight of a pair $$x,y$$ with $$k$$ points in $$I(y,x)$$ is $$\rho_2P^{xy}(N=k)=\sum_{j\ge k}\frac{(-1)^{j-k}}{k!(j-k)!}\int_{I^j}\rho_{2+j}(x,y,z)\,dz$$. Only $$\rho_2$$ is perturbed, which is the term $$j=k=0$$, so $$\rho_2P^{xy}(N=k)=\rho^2[\mathrm{Poi}_\mu(k)+h\,\delta_{k0}]$$ with $$\mu=\rho V(y,x)$$. Dividing by $$\rho_1=\rho$$ and summing against $$c_k$$ and $$\varphi(y)$$ gives the extra term in (a), with $$h$$ evaluated on the timelike pair $$y\prec x$$.

(b) Without the point $$x$$ in the configuration, the weight of $$y$$ with $$k$$ points in the interval is $$\rho_1P^y(N=k)=\sum_{j\ge k}\frac{(-1)^{j-k}}{k!(j-k)!}\int_{I^j}\rho_{1+j}(y,z)\,dz$$. Only $$\rho_2$$, the term $$j=1$$, is perturbed; it enters $$k=0$$ with the sign $$-1$$ and $$k=1$$ with the sign $$+1$$, each as $$\rho^2\int_{I(y,x)}h(y,z)dz$$. The pair $$(y,z)$$ with $$z\in I(y,x)$$ satisfies $$y\prec z$$ and is timelike. Summing against $$c_k$$ gives $$(c_1-c_0)$$ times this integral.

(c) If $$h_+=0$$ both extra terms vanish. If not, the term in (a) is $$\kappa_d\rho^{2/d+1}$$ times $$T_\ell\varphi(x)=\int_{J^-(x)}h_+(\tau^2/\ell^2)\varphi(y)dy$$, a retarded Lorentz-invariant functional independent of $$\rho$$. A function $$h_+(\tau^2/\ell^2)$$ that is not zero almost everywhere on $$J^-(0)$$ has a non-zero integral against some $$\varphi\in C_c^\infty$$ at some $$x$$, and the Poisson part of the mean converges. $$\square$$

</div>

A pair amplitude supported on spacelike separations is therefore invisible to the operator and to every generalised layer kernel whose weights depend on interval counts. Its signature is elsewhere: by Proposition 8 the number variance contains the term $$\rho^2\!\int\!\!\int_{D^2}h$$, over spacelike pairs, and for an amplitude with $$\int h\,d\sigma=0$$ the leading $$V^{3/2}$$ term cancels.

### 5.2 Exactly Poisson above $$M$$ points

<div class="theorem" markdown="1">

**Theorem 11** (order pushforward). Let $$P$$ be a finite-order family. (a) On every bounded $$D$$, the signed measure $$\mathsf P_D$$ and the Poisson measure $$\mathrm{Poi}_D$$ coincide on the configurations with more than $$M$$ points. (b) Let $$F$$ be a function of finite configurations in $$D$$ with $$\lvert F(Y)\rvert\le Ca^{\lvert Y\rvert}$$ for some $$C,a$$, and put $$\mathsf DF(Z)=\sum_{Y\subseteq Z}(-1)^{\lvert Z\setminus Y\rvert}F(Y)$$. Then

$$
\mathbb E_PF-\mathbb E_{\rm Poi}F=\sum_{k=2}^M\frac{\rho^k}{k!}\int_{D^k}h_k(z)\,\mathsf DF(z)\,dz. \tag{12}
$$

If $$F$$ depends only on the causal order of the configuration, $$\mathsf DF(z)$$ depends only on the order type of $$z$$, and the right side is a finite signed combination of expectations over posets with at most $$M$$ elements.

</div>

<div class="proof" markdown="1">

*Proof.* (a) For $$n>M$$ all $$\rho_{n+m}$$ are Poisson, so $$j_n$$ equals the Poisson Janossy density. (b) Absolute convergence: $$\lvert j_n\rvert\le C'\rho^ne^{\rho\lvert D\rvert}$$, so $$\sum_n\frac1{n!}\int\lvert j_nF\rvert\le C''\sum_n(a\rho\lvert D\rvert)^n/n!<\infty$$, which allows rearrangement. Inserting the formula for $$j_n$$ and grouping $$z=(x,y)$$ by $$k=n+m$$,

$$
\mathbb E_PF=\sum_n\frac1{n!}\int j_nF=\sum_k\frac1{k!}\int\rho_k(z)\sum_{Y\subseteq z}(-1)^{\lvert z\setminus Y\rvert}F(Y)\,dz,
$$

by the symmetry of $$\rho_k$$ and $$F$$ (the binomial counts the choices of the subset $$Y$$ of size $$n$$). Subtract the same identity for $$\rho_k=\rho^k$$; only $$k\le M$$ survive. If $$F$$ depends on the order only, the order on $$Y\subseteq z$$ is the restriction of that on $$z$$. $$\square$$

</div>

(The expansion (12) was checked on a finite analogue: five sites, an occupation process with $$\rho=0.3$$ and random bounded $$h_{ij}\in[-0.7,0.7]$$ (signed Janossy probabilities with minimum $$-0.030$$), and a random $$F$$; the difference of expectations is $$-0.0490807$$ and the right side of (12) is $$-0.0490807$$ to $$10^{-14}$$. For $$F$$ vanishing on configurations with at most two points the difference is exactly 0.)

Theorem 11 is the statement that finite-order perturbations are exactly Poisson above $$M$$ points, in law and for every order functional. Two consequences are used below. First, for a $$k$$-point order type $$\tau$$ let $$U_\tau$$ be the number of $$k$$-subsets of type $$\tau$$. Then $$\mathsf DU_\tau(z)=\mathbb 1[z\text{ has type }\tau]$$ for $$\lvert z\rvert=k$$ and $$0$$ for $$\lvert z\rvert\ne k$$, so exactly

$$
\mathbb E_PU_\tau=\mathbb E_{\rm Poi}U_\tau+\frac{\rho^k}{k!}H_k^\tau(D),\qquad H_k^\tau(D)=\int_{D^k}h_k\mathbb 1_\tau, \tag{13}
$$

and the normalised statistic $$u_\tau=U_\tau/\mathbb E_{\rm Poi}U_\tau$$ has $$\mathbb E_Pu_\tau=1+a_D^\tau$$ with $$a_D^\tau=\rho^kH_k^\tau(D)/(k!\,\mathbb E_{\rm Poi}U_\tau)$$. Second, for $$p>M$$ let $$u_p=(N)_p/(\rho\lvert D\rvert)^p$$, where $$(N)_p$$ is the falling factorial; $$u_p$$, $$u_p^2$$ and $$u_\tau u_p$$ vanish on configurations with fewer than $$p$$ points, hence have $$\mathsf DF(z)=0$$ for $$\lvert z\rvert\le M$$, and their expectations under $$P$$ and under Poisson are identical; in particular $$\mathbb E_Pu_p=1$$ exactly. 

### 5.3 Positivity axioms and the two-density witness

Consider the scaled regions $$D=L\,D_0$$ for a fixed bounded $$D_0$$ and $$L\to\infty$$, for which $$\rho\lvert D\rvert\to\infty$$. The statistics $$u_\tau$$ are the *macroscopic densities* of order type: for a Poisson process they converge to 1. A signed process can be given a macroscopic meaning only if these densities have some kind of law. We consider three forms of this requirement, from weakest to strongest.

(V1) *Single-variance form.* The limiting variance of each $$u_\tau$$ is non-negative.

(V2) *Joint-variance form.* For every finite family of the densities (the $$u_\tau$$ and the $$u_p$$), the limiting covariance matrix is positive semi-definite.

(V3) *Law form.* The limiting joint law of the densities exists as a probability measure; equivalently, no non-negative polynomial of the densities has a negative limiting expectation.

(V3) implies (V2), which implies (V1). (V2) does not follow from (V1), as the example after Proposition 12 shows. These are statements about the limit along the chosen sequence of regions, taken along subsequences if the limits do not exist.

<div class="proposition" markdown="1">

**Proposition 12** (two-density witness). Let $$P$$ be a finite-order family, $$k\le M$$, $$\tau$$ a $$k$$-point order type, $$p>M$$, and suppose $$\limsup\lvert a_D^\tau\rvert>0$$ along the scaled regions. Then along those regions

$$
\mathrm{Cov}_P(u_\tau,u_p)=-a_D^\tau+o(1),\qquad \mathrm{Var}_P(u_p)\to0,\qquad \mathrm{Var}_P(u_\tau)=O(1),
$$

so $$\det\mathrm{Cov}_P(u_\tau,u_p)=-(a_D^\tau)^2+o(1)$$, and for suitable real $$t$$ the variance $$\mathrm{Var}_P(u_\tau+t\,u_p)$$ is negative in the limit. Thus (V2), hence (V3), fails whenever some type amplitude $$a^\tau$$ is non-zero in the limit.

</div>

<div class="proof" markdown="1">

*Proof.* By (13), $$\mathbb E_Pu_\tau=1+a_D^\tau$$. By the second consequence of Theorem 11, $$\mathbb E_Pu_p=1$$, $$\mathrm{Var}_Pu_p=\mathrm{Var}_{\rm Poi}u_p$$ and $$\mathbb E_P[u_\tau u_p]=\mathbb E_{\rm Poi}[u_\tau u_p]$$. For the Poisson process on $$LD_0$$ the densities $$u_\tau$$ and $$u_p$$ converge in $$L^2$$ to 1: the variance of a $$U$$-statistic with bounded kernel on $$\rho\lvert D\rvert$$ points is $$O((\rho\lvert D\rvert)^{2k-1})$$ against a squared mean of order $$(\rho\lvert D\rvert)^{2k}$$, the kernel being the indicator of a scale-invariant set of positive measure. Hence $$\mathrm{Cov}_{\rm Poi}(u_\tau,u_p)\to0$$ and $$\mathrm{Var}_{\rm Poi}(u_p)\to0$$. This gives the covariance and the variance of $$u_p$$. $$\mathrm{Var}_P(u_\tau)$$ is bounded because, by (12) with $$F=U_\tau^2$$, the correction involves $$\mathsf DF(z)$$ for $$\lvert z\rvert\le\min(M,2k)$$, bounded by $$2^{2k}\binom{\lvert z\rvert}{k}^2$$, times integrals of $$h_j$$ of order $$(\rho\lvert D\rvert)^{j}$$ against a denominator $$(\rho\lvert D\rvert)^{2k}$$, i.e. a bounded quantity. Then $$\det=\mathrm{Var}u_\tau\mathrm{Var}u_p-\mathrm{Cov}^2\to-a^2$$, and $$\mathrm{Var}(u_\tau+tu_p)=\mathrm{Var}u_\tau+2t\mathrm{Cov}+t^2\mathrm{Var}u_p$$ is, for $$t=\mathrm{sign}(a)T$$ with $$T$$ large, negative when the regions are large.  $$\square$$

</div>

The proposition shows that (V2) already excludes every pair amplitude with a non-zero asymptotic average on any order type of pairs, of either sign. The weaker (V1) does not. Take $$M=2$$ and $$h_2=\epsilon\,\mathbb 1[\text{spacelike}]$$ with $$\epsilon\in(-2,0)$$. The density $$u_{A_2}$$ of antichain pairs has $$\mathbb E_Pu_{A_2}=1+\epsilon$$, and since the second moment shifts only by a term that tends to zero, $$\mathrm{Var}(u_{A_2})\to-2\epsilon-\epsilon^2>0$$ for $$-2<\epsilon<0$$. (V1) holds, (V2) fails because $$\det\mathrm{Cov}(u_{A_2},u_3)\to-\epsilon^2$$. (For $$\epsilon>0$$ even (V1) fails.) So the form of the axiom decides which members survive.

### 5.4 Ergodic components are order-blind, and a lattice that is not covered

The two-density witness concerns finite-order families. For processes with bounded total variation, the classification on [Lorentz-invariant quantum sprinklings](/quantum-gravity/quantum-sprinklings/) reduces everything to positive ergodic processes, whose large-scale order content is that of a Poisson process of the same intensity.

<div class="proposition" markdown="1">

**Proposition 13** (order-blindness of ergodic processes). Let $$Q$$ be a translation-ergodic positive stationary point process on $$\mathbb R^{1,3}$$ with finite intensity $$\lambda$$ (it need not be Poisson or Lorentz invariant). For every order type $$\tau$$ of $$k$$ points and every bounded $$D_0$$ whose boundary has measure zero, almost surely

$$
\frac{U_\tau(L\,D_0)}{\mathbb E_{{\rm Poi}(\lambda)}U_\tau(L\,D_0)}\longrightarrow1\qquad(L\to\infty).
$$

</div>

<div class="proof" markdown="1">

*Proof.* By the spatial ergodic theorem [[Nguyen–Zessin]](#ref-nz), applied to countably many rational boxes, the rescaled counting measures $$\mu_L=L^{-4}\sum_{x\in S}\delta_{x/L}$$ converge vaguely to $$\lambda\,\mathrm{Leb}$$ almost surely. The causal relation between points is invariant under the rescaling $$x\mapsto x/L$$, so $$U_\tau(LD_0)$$ is, up to terms from coinciding points that are $$O(L^{-4})$$ relative, $$L^{4k}\mu_L^{\otimes k}(\mathbb 1_\tau\mathbb 1_{D_0^k})/k!$$. The set $$\{\mathbb 1_\tau=1\}\cap D_0^k$$ has a boundary of $$\mathrm{Leb}^k$$-measure zero (it is contained in the set where some pair is null separated, or lies on $$\partial D_0$$), so by the portmanteau theorem $$\mu_L^{\otimes k}$$ of it converges to $$\lambda^k\mathrm{Leb}^k(\cdots)$$, which is the Poisson value since $$\mathbb E_{{\rm Poi}(\lambda)}U_\tau(LD_0)=\lambda^kL^{4k}\mathrm{Leb}^k(\mathbb 1_\tau\mathbb 1_{D_0^k})/k!$$. $$\square$$

</div>

<div class="corollary" markdown="1">

**Corollary 14** (bounded total variation: no macroscopic order content). Let $$P=\int Q_\theta\,dm(\theta)$$ be a signed or complex combination, with $$\lvert m\rvert$$ finite, of translation-ergodic positive processes $$Q_\theta$$ of finite intensities $$\lambda_\theta$$ (this is the form of a Poincaré-invariant process of bounded total variation after decomposing its ergodic components further under translations, provided they have finite intensity). Then for every finite family of order types $$\tau_1,\dots,\tau_q$$ (with $$k_i$$ points), every bounded continuous $$g$$ on $$\mathbb R^q$$ and $$\rho=\int\lambda_\theta dm$$, $$\mathbb E_Pg(u_{\tau_1},\dots,u_{\tau_q})\to\int g\big((\lambda_\theta/\rho)^{k_1},\dots,(\lambda_\theta/\rho)^{k_q}\big)dm(\theta)$$, where $$u_\tau=u_\tau(LD_0)$$.

</div>

The corollary follows from Proposition 13, $$\mathbb E_{{\rm Poi}(\lambda)}U_\tau/\mathbb E_{{\rm Poi}(\rho)}U_\tau=(\lambda/\rho)^k$$, bounded convergence under each $$Q_\theta$$ and dominated convergence against $$\lvert m\rvert$$. The limit depends on the type only through the number of points. If the intensities $$\lambda_\theta$$ of the components are all equal to $$\rho$$, then every $$u_\tau$$ tends to 1 whatever the signs of $$m$$, jointly for any finite family of types.

This last situation is realised by a combination that is not a mixed Poisson process. Let $$L_\rho$$ be the Haar-random affine lattice of density $$\rho$$ in four dimensions (Proposition 9 of the page on quantum sprinklings), and take $$\epsilon>0$$.

<div class="proposition" markdown="1">

**Proposition 15** (a bounded-variation combination outside the closure). $$P_\epsilon=(1+\epsilon)\,\mathrm{Poi}(\rho)-\epsilon\,L_\rho$$ is a Poincaré-invariant complex point process of total variation $$1+2\epsilon$$. Its densities satisfy $$u_\tau\to1$$ in law for every order type, so it satisfies (V3). It is not a mixed Poisson process and is not a member of any finite-order family, because its third factorial moment measure charges a Lebesgue-null set.

</div>

<div class="proof" markdown="1">

*Proof.* Invariance is that of the two components. The ergodic components of $$\mathrm{Poi}(\rho)$$ (itself) and of $$L_\rho$$ (fixed lattices shifted uniformly, which are translation ergodic) all have intensity $$\rho$$, so Corollary 14 gives $$u_\tau\to1$$ in law. Let $$A$$ be a bounded set of positive volume, $$A'$$ a bounded set containing $$\{2y-x:x,y\in A\}$$, $$S=\{(x,y,z):z=2y-x\}$$, and $$\Theta=\sum^{\ne}_{x,y,z}\mathbb 1_A(x)\mathbb 1_A(y)\mathbb 1_{A'}(z)\mathbb 1_S(x,y,z)$$. On the lattice $$\Theta=N(A)(N(A)-1)$$, with expectation $$V(A)^2$$ (proof of Proposition 9 of the page on quantum sprinklings). On a Poisson process the triples in $$S$$ form a Lebesgue-null set of $$M^3$$, so $$\mathbb E_{\rm Poi}\Theta=0$$. Hence $$\mathbb E_{P_\epsilon}\Theta=-\epsilon V(A)^2\ne0$$. For a process with function-valued correlation functions (a mixed Poisson process, or a member of a finite-order family as in Theorem 7) $$\mathbb E\Theta=\int\mathbb 1_A\mathbb 1_A\mathbb 1_{A'}\mathbb 1_S\rho_3=0$$. $$\square$$

</div>

The mean of the operator on $$P_\epsilon$$ is $$(1+\epsilon)$$ times the Poisson mean minus $$\epsilon$$ times the mean on the lattice, by linearity. The mean on the lattice component is not computed here; whether it is finite and whether $$C_A=1$$ can be reached by adding a further Poisson component is open.

### 5.5 Which closure holds under which hypotheses

The statements proved above, with the hypotheses under which each holds exactly:

| Class of invariant substrate | Hypothesis | Conclusion | Where |
|---|---|---|---|
| Mixed Poisson, positive $$m$$ | (M2), $$d=4$$ | $$C_A=\mathbb E\theta^{1/2}\le1$$, equality iff Poisson | Theorem 3, Corollary 4 |
| Mixed Poisson, $$d=2$$ | (M1) or (M2) | $$C_A=1$$ for every mixing law | Theorem 3 |
| Mixed Poisson, signed or complex $$m$$ with compact support | Born reading of all count probabilities | the process is the classical Cox process of a positive $$\mu$$ | Proposition 5 |
| Same, without a Born reading | none | $$C_A=\langle m,\theta^{1/2}\rangle$$ can equal 1; weak-value post-selection probability at least $$1/S^2$$ (real atoms) or exponentially small (non-real atoms) | Example 1, section 3.3 |
| Bounded total variation | finite intensities of ergodic components | phase mixture of positive ergodic processes; order-blind at large scales | Corollary 14 and the page on quantum sprinklings |
| Pair family | at fixed $$\ell$$ | operator sees only $$h_+$$; the placement-A mean has a limit iff $$h_+=0$$; no Born reading for any $$h\not\equiv0$$ | Propositions 9, 10 |
| Finite-order family | any | exactly Poisson on configurations with more than $$M$$ points | Theorem 11 |
| Finite-order family | (V2) | all type amplitudes vanish | Proposition 12 |
| Finite-order family | (V1) only | members with $$\epsilon\in(-2,0)$$ survive | section 5.3 |
| $$P_\epsilon$$ | (V3) holds | not covered by any of the above | Proposition 15 |

## 6. Quantum causal sets

A quantum poset replaces the points of a poset by matrix algebras and the order by a quantum relation. In the version of Weaver [[Weaver 2012]](#ref-weaver12) and Kornell, Lindenhovius and Mislove [[KLM]](#ref-klm), restricted to the finite-dimensional case needed here, the algebra is $$\bigoplus_xM_{n_x}$$ acting on $$\bigoplus_x\mathbb C^{n_x}$$, and a relation is a family of subspaces $$\mathcal V_{xy}\subset\mathrm{Hom}(\mathbb C^{n_y},\mathbb C^{n_x})$$ that contains the identities, is closed under composition and satisfies an antisymmetry condition; Weaver [[Weaver 2019]](#ref-weaver19) studies the hereditarily antisymmetric case. A sprinkled quantum causal set assigns to each element of a Poisson sprinkling an independent random matrix size $$n_x\le n_{\max}$$ and to each related pair a relation space $$\mathcal V_{xy}$$. The counting of interval elements, $$N(y,x)=\lvert I(y,x)\rvert$$ in (1), is replaced by a positive operator $$\mathcal N^{xy}$$ on $$\mathrm{Hom}(\mathbb C^{n_y},\mathbb C^{n_x})$$ that reduces to $$\lvert I(y,x)\rvert$$ when all $$n_x=1$$, and the layer weights act through $$f(\mathcal N^{xy})$$. The resulting operator acts on fields valued in $$\bigoplus_\alpha M_{n_\alpha}$$ and its mean over the sprinkling and the marks is a matrix-valued Lorentz-invariant retarded kernel. We do not reproduce the definition of $$\mathcal N^{xy}$$. The statement below uses only three of its properties, listed as hypotheses.

The notion of a ghost-free mean operator used here is the following. Write the mean operator in the exponential basis of Proposition 1 as $$b(E)$$, an $$r\times r$$ matrix function of $$Z=E^2$$ with $$b(E)=a\mathbb 1+K(E)$$, where $$a\mathbb 1$$ is the contact term at $$x$$ and $$K(E)=\int F(V)\ldots$$ is the Laplace transform of the kernel. Put $$D(Z)=-b(E)^{-1}$$. We say $$D$$ is *Stieltjes* if $$D(Z)=R_0/Z+C+\int\rho(d\mu)/(Z+\mu)$$ with $$R_0,C\ge0$$ and $$\rho$$ a positive-matrix-valued measure with $$\int(1+\mu)^{-1}\mathrm{tr}\,\rho<\infty$$; for a scalar function this says that the Euclidean propagator is a superposition of free propagators with non-negative weights. Identifying that with the absence of ghosts for a retarded non-local operator is a convention, see [The causal-set d'Alembertian has negative spectral weight](/quantum-gravity/causal-box-spectrum/).

<div class="lemma" markdown="1">

**Lemma 16** (a Stieltjes propagator with a massless pole does not decay fast). Let $$D$$ be Stieltjes with $$R_0$$ positive definite and let $$v\ne0$$. Then $$S_v(Z)=Z\langle v,(D(Z)-C)v\rangle$$ satisfies $$S_v(Z)\ge\langle v,R_0v\rangle>0$$ for all $$Z>0$$, and $$S_v$$ is non-decreasing.

</div>

<div class="proof" markdown="1">

*Proof.* $$S_v(Z)=\langle v,R_0v\rangle+\int\frac Z{Z+\mu}\langle v,\rho(d\mu)v\rangle$$, and $$Z/(Z+\mu)$$ is non-negative and non-decreasing in $$Z$$. $$\square$$

</div>

<div class="theorem" markdown="1">

**Theorem 17** (interval-local decorations do not give a Stieltjes mean operator). Let the mean operator on a sprinkled quantum causal set have the Lorentz-invariant retarded form $$a\mathbb 1\,\delta^4+F(V(x,y))\,\theta(t)$$ on $$\mathbb R^{1,3}$$, with $$F$$ an $$r\times r$$ matrix function and the following properties.

(Q1) *Interval-locality.* $$F(V)=e^{-V}\sum_{k\ge0}g_kV^k/k!$$ where $$g_k$$ is the mean, over the marks and relation spaces, of the entry of $$f(\mathcal N^{xy})$$ given $$k$$ interior points, and depends only on the data of the points in the closed interval.

(Q2) *Boundedness.* $$f$$ is bounded and $$\sup_k\lVert g_k\rVert<\infty$$, and $$\int^\infty\lVert F\rVert\tau\,d\tau<\infty$$ (the last holds, for example, if $$F$$ decays exponentially in $$V$$).

(Q3) *Infrared normalisation.* $$a<0$$ and $$b(E)=-G\,Z+o(Z)$$ as $$Z\to0$$ with $$G$$ positive definite.

Then $$D=-b^{-1}$$ is not Stieltjes.

</div>

The idea is the same as in the scalar case: the contact term and the infrared normalisation fix the constant and the residue of $$D$$, and the decay of the kernel at the apex ($$V\to0$$) makes $$D-C$$ decay faster than $$1/Z$$, which Lemma 16 forbids.

<div class="proof" markdown="1">

*Proof.* The kernel $$F(V)\theta(t)$$ has symbol $$K(E)=4\pi\int_0^\infty F(V(\tau))\tau^3\frac{K_1(E\tau)}{E\tau}d\tau$$ (as in section 2), so $$E^2K(E)=4\pi\int F\,\tau\,[(E\tau)K_1(E\tau)]\,d\tau$$. The function $$zK_1(z)$$ is decreasing from 1 to 0 on $$(0,\infty)$$, so by (Q1)–(Q2) and dominated convergence $$E^2K(E)\to0$$, that is $$K=o(1/Z)$$ entrywise. (Under (Q1), $$F$$ is bounded near $$V=0$$ with $$F(0)=g_0$$; no term concentrated on the light cone is present.) With $$a<0$$, $$D=-(a+K)^{-1}$$ tends to $$C:=-a^{-1}\mathbb 1\ge0$$ as $$Z\to\infty$$, and $$D-C=a^{-2}K(1+K/a)^{-1}=o(1/Z)$$. If $$D$$ were Stieltjes, the residue at $$Z=0$$ would be $$R_0=G^{-1}$$ by (Q3) and $$D(\infty)=C$$ would be its constant; Lemma 16 gives $$S_v(Z)\ge\langle v,G^{-1}v\rangle>0$$, whereas $$S_v(Z)=Z\langle v,(D-C)v\rangle\to0$$. $$\square$$

</div>

In the scalar case with the four-dimensional weights (1), (Q1)–(Q3) hold with $$a=-\kappa_4\rho^{1/2}$$, and $$S(Z)=Z[D(Z)-D(\infty)]$$ evaluated numerically from (4) at $$\rho=1$$ is $$0.15069,\ 0.015387,\ 0.0015391$$ at $$Z=10^2,10^3,10^4$$, falling as $$1/Z$$, so the operator is not Stieltjes. The decoration by quantum data enters only through $$g_k$$, so interval-local quantum decorations have the same property provided (Q1)–(Q2) hold: the mean kernel is bounded at the apex. The argument does not apply to a kernel with a term $$\nu\,\delta(\tau^2)\theta(t)$$ concentrated on the light cone, for which $$K\sim\nu/Z$$; the page on [the causal-box spectrum](/quantum-gravity/causal-box-spectrum/) discusses such kernels. A kernel that depends on a pair $$(x,y)$$ only through the interval-local data and the volume $$V$$ cannot carry such a term, since its value at $$V=0$$ is the finite number $$g_0$$. Decorations that depend on global data, for example a relation angle fixed by the Wightman function of a state built from the whole order, are not covered.

*What the argument does not show.* The mean over the sprinkling is not the per-realisation operator. The quantum structure also changes the second moment: if the matrix sizes are random, the sum $$\sum_zn_z^2$$ over the points of a region is a compound Poisson variable with $$\mathrm{Var}/\mathrm{mean}=\mathbb E[n^4]/\mathbb E[n^2]\ge1$$, and the ratio exceeds 1 as soon as the sizes are not all 1.

## 7. What the theorems do not exclude

Each item states the hypothesis that is dropped and the status.

- *A coefficient set by the realised density.* The factor $$C_A<1$$ is the mean over the mixing law of an operator normalised by $$\bar\rho$$. If the coefficient is chosen from the density of each realisation, every branch of a Cox process gives $$\Box$$. This replaces the local operator (1) by one that depends on a density estimated from a large region of the causal set.
- *Signed or complex mixing laws with $$C_A=1$$.* Example 1 and its relatives exist. They are not mixtures of Poisson processes with a Born reading (Proposition 5); their count statistics are weak values with post-selection probability $$1/S^2$$ for bounded variation, and exponentially small for non-real atoms. The page states no preference.
- *Regimes other than fixed $$\ell$$.* Proposition 10 concerns pair amplitudes with a correlation length $$\ell$$ fixed as $$\rho\to\infty$$. If $$\ell$$ is tied to the discreteness scale ($$\rho\ell^4$$ fixed), the extra term in (10) has the same scaling as the Poisson part; this regime is not analysed here.
- *Pair amplitudes supported on spacelike separations.* They change no mean of the operator (Proposition 10); they change the number variance and the count of antichain pairs, and are excluded as sources of large-scale order content only under (V2) or (V3) (Proposition 12). Under (V1) alone, members with a small negative amplitude survive (section 5.3).
- *Non-clustering amplitudes.* A constant order-type amplitude $$h_2=\epsilon\mathbb 1[\text{spacelike}]$$ has total variation polynomial in $$V$$ and leaves an $$O(1)$$ relative change of the antichain count at every volume; it is in the family treated in Proposition 12 and is excluded there only by the axiom.
- *Bounded-variation combinations of distinct invariant processes.* $$P_\epsilon$$ of Proposition 15 satisfies every form of the positivity axiom and is not a mixed Poisson process; the mean of the operator on its lattice component is not known.
- *Processes of unbounded variation that are not finite-order.* The classification above does not reach processes with correlation functions of all orders modified, for instance the Feynman determinantal kernel, for which only the local form of Proposition 6 is derived.
- *Non-local dressing of quantum posets.* Theorem 17 covers interval-local data.

## 8. Open points

1. The mean of the four-dimensional operator on the Haar-random lattice, and whether a combination of it with Poisson components can reach $$C_A=1$$ while satisfying (V3).
2. Whether a signed mixing law with $$C_A=1$$ can be realised as a state on a Hilbert space with a positive-operator-valued measure for the counts, that is, whether the failure of the Born reading in Proposition 5 can be avoided by a representation in which the counts are not the diagonal observables of the mixing parameter.
3. The behaviour of the pair family when $$\ell$$ scales with the discreteness length.
4. Whether non-local order-intrinsic dressing of quantum-poset data produces a term concentrated on the light cone, and whether the resulting operator can have a Stieltjes propagator.

## References

1. <a id="ref-bd"></a>Benincasa, Dowker, "The scalar curvature of a causal set," Phys. Rev. Lett. 104 (2010) 181301, [arXiv:1001.2725](https://arxiv.org/abs/1001.2725). Used for: the four-dimensional operator (1) and its continuum limit.
2. <a id="ref-sorkin07"></a>Sorkin, "Does locality fail at intermediate length-scales?," [arXiv:gr-qc/0703099](https://arxiv.org/abs/gr-qc/0703099). Used for: the two-dimensional operator.
3. <a id="ref-dg"></a>Dowker, Glaser, "Causal set d'Alembertians for various dimensions," Class. Quantum Grav. 30 (2013) 195016, [arXiv:1305.2588](https://arxiv.org/abs/1305.2588). Used for: the operator in each dimension.
4. <a id="ref-ass"></a>Aslanbeigi, Saravani, Sorkin, "Generalized causal set d'Alembertians," JHEP 06 (2014) 024, [arXiv:1403.1622](https://arxiv.org/abs/1403.1622). Used for: the action on plane waves, the infrared conditions on the coefficients, and the generalised family.
5. <a id="ref-bbl"></a>Belenchia, Benincasa, Liberati, "Nonlocal scalar quantum field theory from causal sets," JHEP 03 (2015) 036, [arXiv:1411.6513](https://arxiv.org/abs/1411.6513). Used for: the non-local field theory defined by the operator and its instability in four dimensions.
6. <a id="ref-lp"></a>Last, Penrose, *Lectures on the Poisson Process*, Cambridge University Press, 2017. Used for: the Mecke formula and Cox processes.
7. <a id="ref-widder"></a>Widder, *The Laplace Transform*, Princeton University Press, 1941. Used for: Bernstein's theorem that a completely monotone function is the Laplace transform of a positive measure.
8. <a id="ref-nz"></a>Nguyen, Zessin, "Ergodic theorems for spatial processes," Z. Wahrscheinlichkeitstheorie verw. Gebiete 48 (1979) 133–158, [doi:10.1007/bf01886869](https://doi.org/10.1007/bf01886869). Used for: the spatial ergodic theorem for stationary point processes.
9. <a id="ref-weaver12"></a>Weaver, "Quantum relations," [arXiv:1005.0354](https://arxiv.org/abs/1005.0354). Used for: quantum relations and quantum posets.
10. <a id="ref-klm"></a>Kornell, Lindenhovius, Mislove, "A category of quantum posets," Indag. Math. 33 (2022) 1137–1171, [arXiv:2101.11184](https://arxiv.org/abs/2101.11184). Used for: the category of quantum posets.
11. <a id="ref-weaver19"></a>Weaver, "Hereditarily antisymmetric operator algebras," J. Inst. Math. Jussieu 20 (2021) 1039–1074, [arXiv:1902.09989](https://arxiv.org/abs/1902.09989). Used for: hereditarily antisymmetric algebras as quantum posets.
