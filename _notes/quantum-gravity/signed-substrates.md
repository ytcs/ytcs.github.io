---
title: Mixed-Poisson and signed substrates and the four-dimensional d'Alembertian
date: 2026-10-29
category: substrate
order: 3
description: In four dimensions the mean of the causal-set d'Alembertian on a sprinkling with a random positive density is the continuum operator times a factor below one unless the density is deterministic, and signed mixtures that restore the factor have no probabilistic reading.
status: The limit theorems (Proposition 1, Theorems 2 and 3) are proved for exponential test fields in two and four dimensions; the Born-reading result needs a compactly supported mixing law; the statements on pair and finite-order perturbations are proved for Poincaré-invariant families with bounded amplitudes; the spectral statement of section 6 is proved for kernels satisfying (Q1)-(Q3). Numerical values evaluate the closed forms. The mean operator on a Haar-random lattice, and the moment forms of the positivity condition for combinations containing it, are not determined.
excludes: Positive mixed Poisson processes as carriers of the four-dimensional causal-set d'Alembertian normalised by the mean density; mixed Poisson processes with a compactly supported signed or complex mixing law and count probabilities in [0,1]; finite-order perturbations of a Poisson process with a non-zero asymptotic type average, under the joint-variance positivity condition of section 5.3; interval-local matrix-valued mean kernels satisfying (Q1)-(Q3) as sources of a Stieltjes propagator.
---

A causal set is a set of events with a partial order, and a *sprinkling* produces one from Minkowski space by choosing points at random and keeping the causal order between them. The Benincasa–Dowker operator is a retarded sum over such a set whose weights are tuned so that, averaged over a Poisson sprinkling of density $$\rho$$, it reproduces the continuum d'Alembertian $$\Box$$ as $$\rho\to\infty$$. The question here is what the same operator gives when the Poisson process is replaced by another Poincaré-invariant point process. Take a mixed Poisson process: a Poisson process whose density $$\rho\theta$$ is drawn once from a law with $$\mathbb E\theta=1$$ and is the same everywhere. In four dimensions the mean of the operator at a point of the process, on the test fields $$e^{Et}$$, tends to $$C\,\Box$$ with $$C=\mathbb E[\theta^{1/2}]$$ (Theorem 3), and $$C<1$$ unless $$\theta$$ is constant (Corollary 4); in two dimensions $$C=1$$ for every law. A signed or complex mixing law can restore $$C=1$$ in four dimensions (Example 1), but the count probabilities of the resulting process are then not all non-negative, and for a compactly supported mixing law, requiring every count probability to lie in $$[0,1]$$ makes the mixing law positive (Proposition 5).

The operator is due to Sorkin in two dimensions [[Sorkin 2007]](#ref-sorkin07) and to Benincasa and Dowker in four [[BD]](#ref-bd); Dowker and Glaser give it in every dimension [[DG]](#ref-dg), and Aslanbeigi, Saravani and Sorkin derive its action on plane waves and the conditions on the weights that keep its mean finite [[ASS]](#ref-ass). Belenchia, Benincasa and Liberati study the non-local field theory it defines [[BBL]](#ref-bbl). Mixed Poisson processes are the Cox processes of point-process theory [[Last–Penrose]](#ref-lp), and the invariant complex point processes used below, together with the classification of those of bounded total variation, are on [Lorentz-invariant quantum sprinklings](/quantum-gravity/quantum-sprinklings/). What this page adds is the exact scaling of the mean under a change of density, which gives the factor $$C$$, the Born-reading statement for signed mixing laws, and the structure of finite-order perturbations of a Poisson process.

Sections 5 and 6 leave the mixed Poisson class. A finite-order perturbation of the Poisson process modifies the correlation functions of at most a fixed number of points; the page shows that such a process is exactly Poisson on configurations with more points, that for pair perturbations the operator sees only the part of the amplitude on causally related pairs, and that requiring the large-scale statistics of causal order to behave like random variables (in the joint-variance form of section 5.3) forces the asymptotic order content of such a perturbation to vanish. Section 6 proves that a matrix-valued mean kernel of interval-local form, with the infrared normalisation of $$\Box$$, has no Stieltjes (positive-spectral-weight) propagator; decorating the points with quantum-poset data is the intended source of such kernels, and whether a given decoration produces one is a separate check.

## 1. The operator, its mean, and where it is evaluated

A causal set is *locally finite* if every interval $$I(y,x)=\{z:y\prec z\prec x\}$$ is finite. A causal set is sprinkled into Minkowski space $$\mathbb M=\mathbb R^{1,d-1}$$ (signature $$-+\cdots+$$, $$\Box=-\partial_t^2+\Delta$$, $$d=2$$ or $$4$$) by taking the points of a point process $$S$$ and setting $$y\prec x$$ when $$y$$ is in the causal past of $$x$$. The Poisson sprinkling of density $$\rho$$ has independent Poisson counts of mean $$\rho V(D)$$ in disjoint sets $$D$$. A pair $$y\prec x$$ determines a causal diamond of volume $$V(y,x)=v_d\tau^d$$, with $$\tau$$ the proper time from $$y$$ to $$x$$, $$v_2=1/2$$ and $$v_4=\pi/24$$, and the number $$N(y,x)$$ of points of $$S$$ in it is Poisson distributed with mean $$\rho V(y,x)$$ for a Poisson sprinkling. These interval counts are the only data the operator below uses.

For a function $$\varphi$$ on the points, the Benincasa–Dowker operator is

$$
(B_\rho\varphi)(x)=\kappa_d\,\rho^{2/d}\Big[\alpha_0\,\varphi(x)+\sum_{n\ge0}c_n\sum_{y\prec x,\ N(y,x)=n}\varphi(y)\Big], \tag{1}
$$

with $$\kappa_2=4$$, $$\alpha_0=-\tfrac12$$, $$(c_0,c_1,c_2)=(1,-2,1)$$ in two dimensions, and $$\kappa_4=4/\sqrt6$$, $$\alpha_0=-1$$, $$(c_0,\dots,c_3)=(1,-9,16,-8)$$ in four [[BD]](#ref-bd), [[Sorkin 2007]](#ref-sorkin07), [[DG]](#ref-dg). The term $$n=0$$ sums over the nearest neighbours of $$x$$ (links), $$n=1$$ over elements with exactly one point between, and so on. The operator is retarded, Lorentz invariant and non-local: it couples $$x$$ to elements arbitrarily far from $$x$$ along the light cone. The sign of the weights is fixed here by requiring $$B_\rho\to\Box$$ in the signature above; other pages of this wiki use other signatures and overall signs, and the formulas below are used only within this page.

For a Poisson process the average of the weights produces the function $$f(V)=e^{-V}\sum_nc_nV^n/n!$$, whose Mellin transform $$\mathcal Mf(s)=\int_0^\infty f(V)V^{s-1}dV$$ follows from $$\int_0^\infty e^{-V}V^{n+s-1}dV=\Gamma(s)(s)_n$$:

$$
\mathcal Mf(s)=-\tfrac43\,\Gamma(s)\,(s-\tfrac12)(s-1)(s-\tfrac32)\ \ (d=4),\qquad \mathcal Mf(s)=\tfrac12\,\Gamma(s)(s-1)(s-2)\ \ (d=2). \tag{2}
$$

The zeros of (2), at $$s=\tfrac12,1,\tfrac32$$ and $$s=1,2$$, are the conditions on the weights that remove the terms of the mean that would otherwise diverge as $$\rho\to\infty$$ or alter its limit (see the proof of Proposition 1) [[ASS]](#ref-ass).

Let $$x$$ be an element of a sprinkling. For a Poisson process of density $$\lambda$$, conditioning on $$x\in S$$ leaves the remaining points Poisson of density $$\lambda$$ (the Mecke formula [[Last–Penrose]](#ref-lp)). The mean of (1) at $$x$$, with the points of density $$\lambda$$ and the coefficient set by a reference density $$\rho$$, is then

$$
\mathbb E\big[B_\rho\varphi(x)\mid x\in S\big]=\kappa_d\rho^{2/d}\Big[\alpha_0\varphi(x)+G_\lambda[\varphi](x)\Big],\qquad G_\lambda[\varphi](x)=\lambda\int_{J^-(x)}f\big(\lambda V(x,y)\big)\varphi(y)\,dy, \tag{3}
$$

where $$J^-(x)$$ is the causal past. The Benincasa–Dowker theorem is that (3) with $$\lambda=\rho$$ tends to $$\Box\varphi(x)$$ as $$\rho\to\infty$$ for suitable fields $$\varphi$$ [[BD]](#ref-bd), [[DG]](#ref-dg). We call this evaluation at an element of the process *placement A*. It is the physical one, since the operator acts on the elements of the causal set. In *placement B* the point $$x$$ is a fixed point of space that is not an element of the process: $$y$$ runs over the elements in the past of $$x$$, the intervals $$I(y,x)$$ are counted as before, and there is no conditioning. For a Poisson process the intensity of $$y$$ is $$\lambda$$ and the interval count is Poisson, so B has the same mean (3). For a non-Poisson process the pair $$(x,y)$$ carries a Palm weight (the ratio of pair to single-point correlation functions) only in A, and B serves as a control that separates the effect of the density from the effect of that weight.

## 2. The Poisson limit on exponential fields

To fix the normalisation, and to have a field class on which every limit below is a computation, we prove the Benincasa–Dowker limit for $$\varphi_E(y)=e^{E(t_y-t_x)}$$ with $$E>0$$, which satisfies $$\Box\varphi_E=-E^2\varphi_E$$. By Lorentz invariance of the sprinkling and of (1) the same holds for $$e^{\xi\cdot(y-x)}$$ with $$\xi$$ any future timelike vector and $$-\xi\cdot\xi=E^2$$. For $$\varphi_E$$ the integral in (3) converges absolutely. Write $$z=x-y$$, $$d^4z=\tau^3d\tau\,d\mu_H$$ with $$\mu_H$$ the measure on the unit hyperboloid, and use $$\int_He^{-E\tau\cosh\eta}d\mu_H=4\pi K_1(E\tau)/(E\tau)$$ with $$K_1$$ the modified Bessel function. With $$V=\pi\tau^4/24$$ this gives

$$
G_\lambda[\varphi_E](x)=\Psi(\eta),\qquad \Psi(\eta)=96\,\eta\int_0^\infty x^2f(\eta x^4)K_1(x)\,dx,\qquad \eta=\frac{\pi\lambda}{24E^4}. \tag{4}
$$

The function $$\Psi$$ is analytic in $$\mathrm{Re}\,\eta>0$$ because $$f(\eta x^4)$$ decays like $$e^{-\eta x^4}$$, and $$\Psi(\eta)\to0$$ as $$\eta\to0$$ (for small $$\lambda$$, $$G\approx\lambda\int_{J^+}e^{-Ez^0}=8\pi\lambda/E^4$$). In two dimensions the same computation, with $$\int_He^{-E\tau\cosh\eta}d\mu_H=2K_0(E\tau)$$, gives $$G_\lambda[\varphi_E]=\Psi_2(\eta)$$ with $$\eta=\lambda/(2E^2)$$ and $$\Psi_2(\eta)=4\eta\int_0^\infty x\,f(\eta x^2)K_0(x)dx$$.

<div class="proposition" markdown="1">

**Proposition 1** (Poisson limit on exponential fields). For $$\eta\to\infty$$ in any closed sector $$\lvert\arg\eta\rvert\le\pi/2-\delta$$,

$$
\Psi(\eta)=1-\frac{\sqrt\pi}{8}\,\eta^{-1/2}+O(\eta^{-1}\ln\lvert\eta\rvert), \tag{5}
$$

uniformly in the sector. Consequently, for $$\lambda=\rho$$ real, $$\mathbb E[B_\rho\varphi_E(x)\mid x\in S]\to-E^2=\Box\varphi_E(x)$$ as $$\rho\to\infty$$. In two dimensions $$\Psi_2(\eta)=\tfrac12-\tfrac1{8\eta}+O(\eta^{-2}\ln\eta)$$ and the limit is again $$-E^2$$.

</div>

For large $$\eta$$ the factor $$f(\eta x^4)$$ confines the integral to $$x\lesssim\eta^{-1/4}$$, where $$K_1$$ can be replaced by its small-argument expansion. Each power of $$x$$ then integrates against $$f$$ to a value of its Mellin transform, and the zeros in (2) remove every term except two.

<div class="proof" markdown="1">

*Proof.* The expansion of the Bessel function is

$$
K_1(x)=\frac1x+\frac x2\Big(\ln\frac x2+\gamma-\frac12\Big)+\frac{x^3}{16}\Big(\ln\frac x2+\gamma-\frac54\Big)+R(x),\qquad \lvert R(x)\rvert\le c\,x^5(1+\lvert\ln x\rvert)\ \ (0<x\le1),
$$

with $$\gamma$$ Euler's constant and $$c$$ a constant. The substitution $$V=\eta x^4$$ gives, for real $$\eta>0$$, $$\int_0^\infty x^{a}f(\eta x^4)\,dx=\frac14\eta^{-(a+1)/4}\mathcal Mf(\tfrac{a+1}4)$$ and, differentiating in the exponent, $$\int_0^\infty x^a\ln x\,f(\eta x^4)\,dx=\frac1{16}\eta^{-(a+1)/4}\big[\mathcal M'f(\tfrac{a+1}4)-\ln\eta\,\mathcal Mf(\tfrac{a+1}4)\big]$$. Both sides are analytic in $$\eta$$, so the identities hold in the right half plane. Applied to the three terms of the expansion multiplied by $$x^2$$:

- the term $$1/x$$ gives $$\frac14\eta^{-1/2}\mathcal Mf(\frac12)=0$$;
- the term in $$x$$ gives $$\frac12\cdot\frac1{16}\eta^{-1}\mathcal M'f(1)$$, the part containing $$\ln\eta$$ being proportional to $$\mathcal Mf(1)=0$$; $$96\eta$$ times it is $$3\mathcal M'f(1)=1$$, because $$\mathcal M'f(1)=\frac13$$ by (2);
- the term in $$x^3$$ gives $$\frac1{16}\cdot\frac1{16}\eta^{-3/2}\mathcal M'f(\frac32)$$, and $$96\eta$$ times it is $$\frac38\mathcal M'f(\frac32)\eta^{-1/2}=-\frac{\sqrt\pi}8\eta^{-1/2}$$, because $$\mathcal M'f(\frac32)=-\frac43\Gamma(\frac32)\cdot1\cdot\frac12=-\frac{\sqrt\pi}3$$.

In the sector $$\lvert f(\eta x^4)\rvert\le c\,(1+\lvert\eta\rvert x^4)^3e^{-\cos\delta\,\lvert\eta\rvert x^4}$$, so the remainder contributes at most a constant times $$\int x^7(1+\lvert\ln x\rvert)\lvert f(\eta x^4)\rvert dx=O(\eta^{-2}\ln\lvert\eta\rvert)$$, and the part of the integral over $$x>1$$ is $$O(e^{-c'\lvert\eta\rvert})$$. This is (5). With $$\eta=\pi\rho/(24E^4)$$,

$$
\kappa_4\rho^{1/2}\big(-1+\Psi(\eta)\big)=-\frac4{\sqrt6}\,\rho^{1/2}\frac{\sqrt\pi}8\Big(\frac{24E^4}{\pi\rho}\Big)^{1/2}+o(1)=-E^2+o(1).
$$

In two dimensions $$K_0(x)=-(\ln\frac x2+\gamma)+\frac{x^2}4(1-\gamma-\ln\frac x2)+O(x^4\ln x)$$, $$\mathcal Mf(1)=\mathcal Mf(2)=0$$, $$\mathcal M'f(1)=-\frac12$$ and $$\mathcal M'f(2)=\frac12$$, and the same steps give $$\Psi_2=\frac12-\frac1{8\eta}+O(\eta^{-2}\ln\eta)$$, so that $$4\rho(-\frac12+\Psi_2)=-\rho/(2\eta)=-E^2$$ with $$\eta=\rho/2E^2$$. $$\square$$

</div>

Numerically, with $$E=1$$ and 30-digit quadrature of (4), the remainder in (5) multiplied by $$\eta$$ is $$0.121,\ 0.192,\ 0.263$$ at $$\eta=10^2,10^4,10^6$$, growing like $$\ln\eta$$ as the error term allows; in two dimensions the remainder multiplied by $$\eta^2$$ is $$0.170,\ 0.313,\ 0.457$$. The mean at an element approaches $$-E^2=-1$$ as follows.

| $$\rho$$ | $$10^2$$ | $$10^4$$ | $$10^6$$ | $$10^8$$ |
|---|---|---|---|---|
| $$d=4$$ | $$-0.88575$$ | $$-0.98001$$ | $$-0.99711$$ | $$-0.99962$$ |
| $$d=2$$ | $$-0.97625$$ | $$-0.99953$$ | $$-0.999993$$ | |

The four-dimensional deviations fall like $$\rho^{-1/2}\ln\rho$$, as (5) predicts.

## 3. Mixed Poisson substrates

A *mixed Poisson* process is a point process that is Poisson of random density $$\lambda$$, the same at every point, with $$\lambda$$ drawn from a *mixing law* $$m$$. The mixed Poisson states of [Lorentz-invariant quantum sprinklings](/quantum-gravity/quantum-sprinklings/) are of this kind, with a random intensity $$\lambda=\lvert z\rvert^2$$ (called $$\Lambda$$ there). The correlation functions are $$\rho_n=\int\lambda^ndm(\lambda)$$ and the count generating function is $$\mathbb E\,\mathsf u^{N(D)}=\int e^{(\mathsf u-1)\lambda V(D)}dm(\lambda)$$.

We also admit mixing laws that are not positive: a finite complex measure, or a finite-order distribution with compact support in $$\{\mathrm{Re}\,\lambda>0\}$$, defines through these formulas a *complex point process*, a family of complex measures on configurations specified by its correlation functions $$\rho_n$$. Section 4 (Theorem 7) shows that such a process exists whenever $$\rho_n$$ obeys a growth bound, which holds for compact support. The pairing of $$m$$ with a function $$h$$ analytic near the support is denoted $$\langle m,h\rangle$$. We take $$\langle m,1\rangle=1$$ and write $$\bar\rho=\langle m,\lambda\rangle$$ for the mean density.

At placement A the mean of (1) is, by the Palm computation of section 1 applied to the correlation functions $$\rho_{2+j}=\langle m,\lambda^{2+j}\rangle$$,

$$
\mathbb E_A\big[B_\rho\varphi(x)\big]=\kappa_d\rho^{2/d}\Big[\alpha_0\varphi(x)+\frac1{\bar\rho}\big\langle m,\lambda\,G_\lambda[\varphi](x)\big\rangle\Big], \tag{6}
$$

since the pair $$(x,y)$$ is weighted by $$\rho_2=\langle m,\lambda^2\rangle$$ and divided by $$\rho_1=\bar\rho$$. At placement B it is $$\kappa_d\rho^{2/d}[\alpha_0\varphi(x)+\langle m,G_\lambda[\varphi](x)\rangle]$$. The coefficient in (1) is set by $$\rho$$, which we take equal to the mean density $$\bar\rho$$: the operator is built knowing only the mean.

### 3.1 Exact scaling

<div class="theorem" markdown="1">

**Theorem 2** (a different density is a dilated field). Let $$\varphi$$ be measurable and such that the integral in (3) converges absolutely, for instance bounded with bounded support, or $$\varphi_E$$. For every $$\rho,\lambda>0$$ and $$s=(\rho/\lambda)^{1/d}$$,

$$
G_\lambda[\varphi](x)=G_\rho[\varphi_s](x),\qquad \varphi_s(y)=\varphi\big(x+s(y-x)\big). \tag{7}
$$

Consequently the mean of (1) over a Poisson sprinkling of density $$\lambda$$, with coefficient set by $$\rho$$, equals the mean over a Poisson sprinkling of density $$\rho$$ acting on the field $$\varphi_s$$ dilated about $$x$$.

</div>

<div class="proof" markdown="1">

*Proof.* Substitute $$y=x+s\,(y'-x)$$ in the integral of (3). Then $$dy=s^d\,dy'$$ and $$V(x,y)=s^dV(x,y')$$, because volumes scale with the $$d$$th power of lengths. Since $$\lambda s^d=\rho$$, the integrand becomes $$\rho f(\rho V(x,y'))\,\varphi(x+s(y'-x))\,dy'$$, which is the integrand of $$G_\rho[\varphi_s]$$. $$\square$$

</div>

A Poisson sprinkling of density $$\lambda\ne\rho$$ read with the operator of density $$\rho$$ therefore sees the field through a rescaling by $$s$$. Since $$\Box\varphi_s=s^2(\Box\varphi)_s$$, the continuum limit of its mean is $$(\rho/\lambda)^{2/d}\Box\varphi$$ whenever the Poisson limit holds for $$\varphi_s$$. For exponential fields this is explicit: $$G_\lambda[\varphi_E]=G_\rho[\varphi_{sE}]$$, which is (4) depending on $$\lambda/E^4$$ only.

### 3.2 The limit and its sign

The operator (1) carries one coefficient $$\rho^{2/d}$$, fixed by the mean density. A coefficient read off from the density of the realisation instead would be a non-local functional of the causal set, an average over a large region, and the operator would no longer be a local function of the neighbourhood of $$x$$. With a coefficient set from the realised density $$\rho\theta$$, each branch of a mixed Poisson process would give $$\Box$$ by Proposition 1, and the mean would be $$\Box$$. With the coefficient fixed, the ensemble mean is what the next theorem computes.

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

Powers of $$\theta$$ are principal branches. The limit contains no other term.

</div>

The mean (6) is an average over the mixing law of Poisson means of dilated fields. The term $$\alpha_0\varphi(x)$$ cancels against the constant in the Poisson limit because $$\langle m,\lambda\rangle=\rho$$, and what remains is the average of $$\theta^{-2/d}$$ weighted by the Palm factor $$\theta$$.

<div class="proof" markdown="1">

*Proof.* Set $$H_\rho(\theta)=\rho^{1/2}\big(\Psi(\rho\theta w)-1\big)$$ with $$w=\pi/(24E^4)$$ in four dimensions. By (6), $$\alpha_0=-1$$ and $$\bar\rho=\rho$$, and writing $$\lambda=\rho\theta$$,

$$
\mathbb E_A[B_\rho\varphi_E(x)]=\kappa_4\,\big\langle m_1,\ \theta\,H_\rho(\theta)\big\rangle,\qquad \mathbb E_B[B_\rho\varphi_E(x)]=\kappa_4\,\langle m_1,H_\rho\rangle,
$$

because $$-1=-\langle m_1,\theta\rangle$$ and $$-1=-\langle m_1,1\rangle$$ respectively. By Proposition 1, $$H_\rho(\theta)\to H_\infty(\theta)=-\frac{\sqrt\pi}8\big(\frac{24E^4}{\pi\theta}\big)^{1/2}=-\frac{\sqrt{24}}8E^2\theta^{-1/2}$$, locally uniformly on $$\{\mathrm{Re}\,\theta>0\}$$: compact subsets lie in a closed sector, and $$\rho\theta w\to\infty$$ in it. The functions $$H_\rho$$ are analytic, so their derivatives of all orders converge locally uniformly as well. Under (M1) the pairing with $$m_1$$ involves finitely many derivatives on a compact set, so the limit may be taken inside. Under (M2) we use dominated convergence. The function $$\Psi$$ is continuous on $$(0,\infty)$$, tends to 0 at 0 and satisfies (5) at infinity, so $$\lvert\Psi(\eta)-1\rvert\le c\min(1,\eta^{-1/2})$$ for $$\eta>0$$ and a constant $$c$$, and hence $$\lvert H_\rho(\theta)\rvert\le c'\theta^{-1/2}$$ uniformly in $$\rho\ge1$$: for $$\rho\theta w\ge1$$ use the decay, and otherwise $$\rho^{1/2}\le(w\theta)^{-1/2}$$. The dominating function $$\theta\cdot\theta^{-1/2}=\theta^{1/2}$$ is integrable against $$m_1$$ since $$\int\theta^{1/2}dm_1\le(\int\theta\,dm_1)^{1/2}=1$$. For placement B the dominating function is $$\theta^{-1/2}$$. With $$\kappa_4=4/\sqrt6$$, $$\kappa_4\cdot(-\frac{\sqrt{24}}8E^2)=-E^2$$, which gives $$C_A=\langle m_1,\theta\cdot\theta^{-1/2}\rangle=\langle m_1,\theta^{1/2}\rangle$$ and $$C_B=\langle m_1,\theta^{-1/2}\rangle$$.

In two dimensions the same argument with $$\Psi_2$$, $$H_\rho(\theta)=\rho(\Psi_2(\rho\theta/2E^2)-\frac12)\to-\frac{E^2}{4\theta}$$ and $$\kappa_2=4$$ gives $$\mathbb E_A\to-E^2\langle m_1,\theta\cdot\theta^{-1}\rangle=-E^2$$ and $$\mathbb E_B\to-E^2\langle m_1,\theta^{-1}\rangle$$. $$\square$$

</div>

<div class="corollary" markdown="1">

**Corollary 4** (positive mixtures are mis-normalised in four dimensions). Under (M2) with $$d=4$$ and $$m_1\ne\delta_1$$: $$C_A<1$$ and, when finite, $$C_B>1$$. In $$d=2$$, $$C_A=1$$ and $$C_B>1$$.

</div>

<div class="proof" markdown="1">

*Proof.* Jensen's inequality for the strictly concave $$\theta^{1/2}$$ and the strictly convex $$\theta^{-1/2}$$ and $$\theta^{-1}$$, with $$\mathbb E\theta=1$$. $$\square$$

</div>

The mixed Poisson states of the quantum-sprinkling classification are covered by (M2) with $$\mathrm{Law}(\theta)=m_1$$, so their mean operator is $$\mathbb E[\theta^{1/2}]\,\Box<\Box$$ unless the intensity is deterministic. For the positive two-atom law with $$\theta=0.25,1.75$$ and weights $$\frac12,\frac12$$, $$\langle\theta^{1/2}\rangle=0.91144$$ and $$\langle\theta^{-1/2}\rangle=1.37796$$.

### 3.3 Signed mixing laws, and what a Born reading forces

Theorem 3 does not need positivity of $$m$$, and for a signed law the condition $$\langle m_1,\theta^{1/2}\rangle=1$$ is a single real equation.

<div class="example" markdown="1">

**Example 1** (a signed mixing law with the correct normalisation). Take atoms $$\theta_i=(0.25,\,1.44,\,4)$$, so $$\sqrt{\theta_i}=(0.5,1.2,2)$$, and solve $$\sum q_i=\sum q_i\theta_i=\sum q_i\sqrt{\theta_i}=1$$. The solution is $$q=(4/21,\ 25/28,\ -1/12)=(0.190476,\ 0.892857,\ -0.083333)$$, with total variation $$\sum\lvert q_i\rvert=7/6$$. One weight is negative, as it must be: by Corollary 4 a positive law with $$C_A=1$$ is a point mass. By Theorem 3 (case M1) $$C_A=1$$ in the limit, and $$C_B=\langle\theta^{-1/2}\rangle=1.08333$$. The process is a signed combination of three Poisson processes, hence a complex point process of bounded total variation, and by Theorem 10 of the page on quantum sprinklings a phase mixture of classical processes with weights of both signs.

</div>

The finite-$$\rho$$ values of the ratio of the exact mean to $$\Box\varphi_E$$ approach these limits as follows.

| $$\rho$$ | two-atom law, $$C_A$$ | two-atom law, $$C_B$$ | Example 1, $$C_A$$ | Example 1, $$C_B$$ |
|---|---|---|---|---|
| $$10^4$$ | $$0.89223$$ | $$1.33725$$ | $$0.98011$$ | $$1.05776$$ |
| $$10^6$$ | $$0.90863$$ | $$1.37187$$ | $$0.99712$$ | $$1.07957$$ |
| $$10^8$$ | $$0.91107$$ | $$1.37715$$ | $$0.99962$$ | $$1.08283$$ |
| limit | $$0.91144$$ | $$1.37796$$ | $$1$$ | $$1.08333$$ |

In two dimensions $$C_A$$ tends to 1 for every law: for the Poisson law, the law of Example 1 and the two-atom law the exact values are $$0.97625,\ 0.97177,\ 0.95821$$ at $$\rho=10^2$$ and $$0.999993,\ 0.999991,\ 0.999985$$ at $$\rho=10^6$$.

The weights of Example 1 are not those of a probability measure, and the process has no probabilistic reading. The next statement shows that this holds for every mixed Poisson process with compactly supported mixing law.

<div class="proposition" markdown="1">

**Proposition 5** (a Born reading forces a positive mixing law). Let $$m$$ be a complex measure or finite-order distribution with compact support in $$\mathbb C$$ and $$\langle m,1\rangle=1$$, and let $$P$$ be the mixed Poisson complex point process with $$\rho_n=\langle m,\lambda^n\rangle$$. Suppose that $$\Pr(N(D)=k)$$ is a real number in $$[0,1]$$ for every bounded $$D$$ and every $$k\ge0$$. Then there is a positive probability measure $$\mu$$ on $$[0,\infty)$$ with $$\langle\mu,\lambda^n\rangle=\langle m,\lambda^n\rangle$$ for all $$n$$. Hence $$P$$ is the classical Cox process with mixing law $$\mu$$; its correlation functions, and with them the means of the operator, are those of $$\mu$$. If $$\mu(\{0\})=0$$, so that (M2) holds after the rescaling $$\theta=\lambda/\bar\rho$$, then $$C_A\le1$$ in $$d=4$$ with equality if and only if $$P$$ is Poisson.

</div>

<div class="proof" markdown="1">

*Proof.* Let $$L(V)=\langle m,e^{-\lambda V}\rangle$$, an entire function of $$V$$. The count probabilities are $$\Pr(N(D)=k)=\langle m,e^{-\lambda V}(\lambda V)^k\rangle/k!=(-V)^kL^{(k)}(V)/k!$$ with $$V=V(D)$$, and $$V(D)$$ takes every value in $$(0,\infty)$$. Non-negativity of all of them says $$(-1)^kL^{(k)}\ge0$$ on $$(0,\infty)$$, that is, $$L$$ is completely monotone. By the Bernstein–Widder theorem [[Widder]](#ref-widder), $$L(V)=\int_{[0,\infty)}e^{-\lambda V}d\mu(\lambda)$$ for a positive measure $$\mu$$ of total mass $$L(0^+)=\langle m,1\rangle=1$$. Since $$L$$ is entire, $$(-1)^nL^{(n)}(0)=\langle m,\lambda^n\rangle$$, and $$(-1)^nL^{(n)}(0^+)=\int\lambda^nd\mu$$ by monotone convergence, so the moments agree. The correlation functions determine the complex point process (Theorem 7), and the mean of the operator at an element is computed from them. The last statement is Corollary 4. $$\square$$

</div>

The mixing "law" of a mixed Poisson process is determined only through its holomorphic moments: adding to $$m$$ a derivative $$\partial_{\bar\lambda}\Phi$$ of any compactly supported distribution $$\Phi$$ changes nothing, since $$\langle\partial_{\bar\lambda}\Phi,h\rangle=0$$ for holomorphic $$h$$. A signed $$m$$ is therefore not a measure to which positivity can be assigned directly, and positivity has to be formulated for the counts, as in Proposition 5. The law of Example 1 has $$\mathrm{Var}_m(\lambda)=\sum q_i\theta_i^2-1=-0.47$$, so it is not the law of a random variable, and its count probabilities are not all non-negative: for $$\rho V$$ in the grid $$0.1,0.2,\dots,39.9$$ and $$k\le39$$, the smallest is $$-0.0087$$, at $$k=11$$ and $$\rho V=2.6$$.

Weak values give an operational reading of such a process at a price. A weak value of a projection $$E$$ for an initial vector $$\psi_i$$ and a final vector $$\psi_f$$ is $$\langle\psi_f,E\psi_i\rangle/\langle\psi_f,\psi_i\rangle$$, the value of $$E$$ in an experiment pre-selected in $$\psi_i$$ and kept only when the final measurement gives $$\psi_f$$. A complex measure $$\mathsf P_D$$ on count configurations, with phase density $$\vartheta=d\mathsf P_D/d\lvert\mathsf P_D\rvert$$ and total variation $$\lVert\mathsf P_D\rVert$$, is the weak value of the multiplication projection-valued measure $$E$$ on $$L^2(\lvert\mathsf P_D\rvert)$$ for $$\psi_f=\lVert\mathsf P_D\rVert^{-1/2}$$ and $$\psi_i=\lVert\mathsf P_D\rVert^{-1/2}\vartheta$$. For any representation $$\mathsf P_D(A_j)=\langle\psi_f,E_j\psi_i\rangle/\langle\psi_f,\psi_i\rangle$$ by unit vectors and a positive-operator-valued measure $$E_j$$, Cauchy–Schwarz gives $$\sum_j\lvert\langle\psi_f,E_j\psi_i\rangle\rvert\le1$$. Since $$\lVert\mathsf P_D\rVert$$ is the supremum of $$\sum_j\lvert\mathsf P_D(A_j)\rvert$$ over partitions, $$\lvert\langle\psi_f,\psi_i\rangle\rvert^2\le\lVert\mathsf P_D\rVert^{-2}$$, and the construction above attains this post-selection probability. For Example 1, $$\lVert\mathsf P_D\rVert=\sum_n\lvert\sum_iq_i\mathrm{Poi}_{\lambda_iV}(n)\rvert\le7/6$$ (given $$N=n$$ the configuration is uniform in $$D$$ for every atom), which rises from $$1.012$$ at $$\rho V=0.5$$ to $$1.1665$$ at $$\rho V=20$$; the post-selection probability is therefore at least $$36/49=0.735$$ for every region. For a finite atomic law with a non-real atom the cost is exponential in the volume. Write $$q_j$$ for the weights and let $$\beta=\max_j(\lvert\lambda_j\rvert-\mathrm{Re}\lambda_j)>0$$, attained by the atom $$\lambda_j$$ at the unit phase $$\mathsf u_j=\lvert\lambda_j\rvert/\lambda_j$$, where $$\mathrm{Re}\,(\mathsf u_j-1)\lambda_j=\beta$$ and every other atom has a strictly smaller real part of $$(\mathsf u_j-1)\lambda_i$$. Then $$\lVert\mathsf P_D\rVert\ge\lvert\mathbb E\mathsf u_j^N\rvert$$ because $$\lvert\mathsf u_j\rvert=1$$, and $$e^{-\beta V}\lvert\mathbb E\mathsf u_j^N\rvert\to\lvert q_j\rvert>0$$ as $$V\to\infty$$, since the other atoms contribute strictly smaller exponents (equality would need $$\lambda_i$$ on the same ray from the origin as $$\lambda_j$$ and the same value of $$\lvert\lambda\rvert-\mathrm{Re}\lambda$$, hence $$\lambda_i=\lambda_j$$). The post-selection probability is then at most a constant times $$e^{-2\beta V}$$.

### 3.4 The Feynman kernel as a distributional mixture

A determinantal complex process with kernel $$\mathcal K(x,y)=\rho g/(s^2/\ell^2-i)$$ for $$x\ne y$$ and $$\mathcal K(x,x)=\rho$$, where $$s^2$$ is the squared interval (positive for timelike separation), has correlation functions $$\rho_n=\det[\mathcal K(x_i,x_j)]$$ that depend only on the invariant intervals. They are bounded by $$(\rho\max(1,g))^nn^{n/2}$$ (Hadamard's inequality, since every entry has modulus at most $$\rho\max(1,g)$$), so Theorem 7 gives a Poincaré-invariant complex point process. It has the Feynman-propagator form of two-point amplitude, regularised at the length $$\ell$$.

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

*Proof.* In a region of diameter $$r\ll\ell$$ the kernel is $$\mathcal K(x,y)=i\rho g\,(1+is^2/\ell^2)^{-1}\approx i\rho g$$ off the diagonal, and $$\mathcal K(x,x)=\rho$$. The matrix $$[\mathcal K(x_i,x_j)]$$ is then $$\rho(1-ig)\mathbb 1+i\rho g\,\mathbf J$$ with $$\mathbf J$$ the all-ones matrix, whose determinant is $$\lambda_0^{n-1}(\lambda_0+i\rho gn)$$. Thus $$\rho_n=\lambda_0^n+i\rho g\,n\lambda_0^{n-1}$$. For the distribution (8), $$\langle\delta'_{\lambda_0},h\rangle=-h'(\lambda_0)$$, so $$\langle m,\lambda^n\rangle=\lambda_0^n+i\rho g\,n\lambda_0^{n-1}$$, the same. The sum in (1) probes intervals with $$\tau^4\sim1/\rho$$, so every separation that enters satisfies $$\lvert s^2\rvert/\ell^2\lesssim(\rho\ell^4)^{-1/2}\to0$$. For $$C_A$$, $$\langle m_1,\theta^{1/2}\rangle=(1-ig)^{1/2}+\frac{ig}2(1-ig)^{-1/2}$$ gives the formula. $$\square$$

</div>

This is a derivation in the stated limit, not a statement about the exact process at finite $$\rho\ell^4$$. For $$g=1$$ the formula gives $$C_A=0.93779-0.06665i$$, and the exact mean (6) for (8), evaluated by differentiating the analytic function $$\lambda G_\lambda$$ at $$\lambda_0$$, gives $$0.91809-0.06610i$$ at $$\rho=10^4$$ and $$0.93492-0.06659i$$ at $$\rho=10^6$$. The correction $$C_A-1$$ is complex, and a complex normalisation of the operator is not a rescaling of the continuum d'Alembertian. The probability of an empty region of expected count $$\bar N=\rho V$$ is $$\Pr(N=0)=e^{-\bar N}e^{ig\bar N}(1-ig\bar N)$$, whose imaginary part $$e^{-\bar N}(\sin g\bar N-g\bar N\cos g\bar N)$$ is positive for $$0<g\bar N<4.4934$$ (the first positive root of $$\tan x=x$$), and $$\mathrm{Re}\Pr(N=3)=-0.3282$$ at $$g=1$$, $$\bar N=2$$. No Born reading is possible there either.

## 4. Signed and complex invariant processes

The Janossy density $$j_n^D(x_1,\dots,x_n)$$ of a point process is the density, with respect to Lebesgue measure on $$D^n$$, of the event that $$D$$ contains exactly the points $$x_1,\dots,x_n$$; for a positive process $$\frac1{n!}\int j_n^D$$ is the probability that $$N(D)=n$$. A complex point process is a consistent family of complex measures $$\mathsf P_D$$ built from signed Janossy densities. Existence needs only a growth bound on the correlation functions, and the family it describes is the one used in section 5.

<div class="theorem" markdown="1">

**Theorem 7** (realisability without positivity). Let $$\rho_0=1$$ and let $$\rho_n$$, $$n\ge1$$, be symmetric measurable functions on $$\mathbb M^n$$ with $$\lvert\rho_n\rvert\le\varsigma^nn^{n/2}$$ for some $$\varsigma$$. Then there is a unique simple complex point process $$\{\mathsf P_D\}$$ whose Janossy densities are

$$
j_n^D(x_1,\dots,x_n)=\sum_{l\ge0}\frac{(-1)^l}{l!}\int_{D^l}\rho_{n+l}(x,y)\,dy,
$$

and whose correlation functions are $$\rho_n$$. If the $$\rho_n$$ are Poincaré invariant, so is the process.

</div>

<div class="proof" markdown="1">

*Proof.* The series converges absolutely: $$\sum_l\lvert D\rvert^l\varsigma^{n+l}(n+l)^{(n+l)/2}/l!<\infty$$ because $$k^{k/2}/k!$$ decays faster than any exponential. With $$k=n+l$$ and $$\binom kn$$ counting the ways to split, $$\sum_n\frac1{n!}\int\lvert j_n\rvert\le\sum_k\frac{(\varsigma\lvert D\rvert)^kk^{k/2}}{k!}\sum_{n}\binom kn=\sum_k\frac{(2\varsigma\lvert D\rvert)^kk^{k/2}}{k!}<\infty$$, so $$\mathsf P_D$$ has finite total variation. Normalisation: $$\sum_n\frac1{n!}\int j_n=\sum_k\frac1{k!}\int\rho_k\sum_n\binom kn(-1)^{k-n}=\rho_0=1$$, since only $$k=0$$ survives. For $$D\subset D'$$, restricting the configuration of $$D'$$ to $$D$$ integrates out $$r$$ points in $$D'\setminus D$$; the coefficient of the terms with $$r$$ such points is $$\frac1{r!}\sum_j\binom rj(-1)^j=\delta_{r0}$$, which is consistency. The same identity inverts the formula and returns $$\rho_n=\sum_l\frac1{l!}\int j_{n+l}$$. The $$j_n$$ determine $$\mathsf P_D$$, which gives uniqueness, and invariance of the $$\rho_n$$ gives invariance of $$j_n^D$$ with $$D$$ transported. $$\square$$

</div>

Non-positive invariant correlations therefore exist for any invariant correlation functions of moderate growth, for example $$\rho_1=\rho$$, $$\rho_2=\rho^2(1+h(s^2))$$ and $$\rho_n=\rho^n$$ for $$n\ge3$$ with any bounded $$h$$. The obstruction of the classification of quantum sprinklings (Bochner positivity forcing a point measure) has no analogue at the level of correlation functions.

The simplest observable of such a process is the number variance of a causal diamond $$D_T$$ of height $$T$$ and volume $$V=\pi T^4/24$$. Poincaré invariance makes $$\rho_2(x,y)-\rho^2$$ a function $$\chi(s^2(x-y))$$ of the squared interval, and the symmetry of $$\rho_2$$ makes it independent of which point of the pair is earlier. Assume $$\chi\in L^1(\mathbb R)$$. Let $$\omega$$ be the density of $$s^2$$ over ordered pairs of the unit diamond $$D_1$$, defined by $$\int\!\!\int_{D_1^2}F(s^2)dx\,dy=\int F(\sigma)\omega(\sigma)d\sigma$$.

<div class="proposition" markdown="1">

**Proposition 8** (the $$V^{3/2}$$ law). The function $$\omega$$ is bounded and continuous at $$0$$ with $$\omega(0)=\pi^2/288=2(\pi/24)^2$$. For the invariant complex process above,

$$
\mathrm{Var}\,N(D_T)=\rho V+T^6\!\int \chi(\sigma)\,\omega(\sigma/T^2)\,d\sigma=\rho V+\omega(0)\chi_*T^6+o(T^6),\qquad \chi_*=\int \chi(\sigma)d\sigma, \tag{9}
$$

with $$V=\pi T^4/24$$ the geometric volume, so that $$\omega(0)T^6=\frac{\pi^2}{288}\big(\frac{24V}{\pi}\big)^{3/2}\approx0.7236\,V^{3/2}$$. In particular, for an invariant process with integrable pair amplitude $$\chi$$ and $$\chi_*\ne0$$, the variance differs from the Poisson value by a term of order $$V^{3/2}$$.

</div>

The variance is the integral of the pair amplitude over pairs of points. Pairs at squared interval $$\sigma$$ fill a set that, in a diamond of height $$T$$, scales as $$T^8\omega(\sigma/T^2)$$, and an integrable $$\chi$$ confines $$\sigma$$ to a bounded range, so that only $$\omega$$ near $$0$$ matters: the density of null-separated pairs, which grows like $$T^6=V^{3/2}$$ instead of $$V^2$$.

<div class="proof" markdown="1">

*Proof.* The first equality in (9) is $$\mathrm{Var}N=\int_D\rho_1+\int\!\!\int_{D^2}(\rho_2-\rho^2)$$, and scaling $$D_T=TD_1$$ gives $$\int\!\!\int_{D_T^2}\chi(s^2)=T^8\!\int\!\!\int_{D_1^2}\chi(T^2s^2)=T^6\int \chi(\sigma)\omega(\sigma/T^2)d\sigma$$. For fixed $$x\in D_1$$, write $$y-x=(\Delta t,\mathbf d)$$. For $$\sigma>0$$ the density of $$s^2=\Delta t^2-\lvert\mathbf d\rvert^2$$ is $$\sum_\pm\int d^3\mathbf d/(2\sqrt{\lvert\mathbf d\rvert^2+\sigma})$$ over the $$\mathbf d$$ with $$x+(\pm\sqrt{\lvert\mathbf d\rvert^2+\sigma},\mathbf d)\in D_1$$, which is at most $$\int_{\lvert\mathbf d\rvert<1}d^3\mathbf d/\lvert\mathbf d\rvert=2\pi$$ since $$D_1$$ has diameter 1. For $$\sigma=-\varrho^2<0$$ it is $$2\pi\int d\Delta t\sqrt{\Delta t^2+\varrho^2}$$ over the admissible $$\Delta t$$, at most $$4\pi\sqrt2$$. Both expressions are continuous at $$\sigma=0$$ by dominated convergence, because $$1/\lvert\mathbf d\rvert$$ is integrable. Dominated convergence in (9) then gives the limit, with $$\omega(0)=\int_{D_1}dx\int dy\,\delta(s^2)\mathbb 1_{D_1}(y)$$.

To evaluate $$\omega(0)$$, take $$x=(t,r)$$ in the diamond $$\lvert t\rvert+\lvert\vec x\rvert<\frac12$$, and a future null ray from $$x$$ in a direction making cosine $$\mu$$ with $$\vec x$$. It leaves the diamond at parameter $$\tau_+=(\Delta_+^2-r^2)/(2(\Delta_++r\mu))$$ with $$\Delta_+=\frac12-t$$, from the condition $$t+\tau+\lvert\vec x+\tau\hat n\rvert=\frac12$$; for the past ray the same holds with $$\Delta_-=\frac12+t$$. The measure $$\delta(s^2)d^4y$$ on the cone is $$\tau\,d\tau\,d\Omega/2$$ per sheet, so a sheet contributes $$\int d\Omega\,\tau_\pm^2/4=\frac\pi4(\Delta_\pm^2-r^2)$$ after integrating over $$\mu$$ with $$\int_{-1}^1d\mu\,(\Delta+r\mu)^{-2}=2/(\Delta^2-r^2)$$. Integrating over $$D_1$$ with $$\int_{D_1}=\int dt\int4\pi r^2dr$$ and $$r<\frac12-\lvert t\rvert$$,

$$
\omega(0)=\frac\pi4\int_{D_1}\Big[\tfrac12+2t^2-2r^2\Big]d^4x=\frac\pi4\cdot\frac\pi{72}=\frac{\pi^2}{288}. \qquad\square
$$

</div>

The coefficient $$\chi_*$$ vanishes, for example, for any pair amplitude $$\chi(\sigma)$$ that extends to a function holomorphic and decaying in the lower half $$\sigma$$-plane, by closing the contour. The Feynman kernel above is of this type: $$\chi(\sigma)=-\rho^2g^2/(\sigma/\ell^2-i)^2$$ has its pole in the upper half plane, and $$\int\chi\,d\sigma=-\rho^2g^2\ell^2\int dw\,(w-i)^{-2}=0$$ because the antiderivative $$-1/(w-i)$$ vanishes at $$\pm\infty$$. For such processes the $$V^{3/2}$$ term in (9) is absent, and (9) does not determine the next term.

## 5. Pair and finite-order amplitudes

The simplest invariant complex process that is not a mixture is the *pair family*: $$\rho_1=\rho$$, $$\rho_2(x,y)=\rho^2(1+h(x,y))$$ with $$h=h(s^2/\ell^2)$$ bounded and invariant, and $$\rho_n=\rho^n$$ for $$n\ge3$$. The *finite-order family* of order $$\bar n$$ has $$\rho_k=\rho^k(1+h_k)$$ for $$2\le k\le\bar n$$, with $$h_k$$ bounded, symmetric and Poincaré invariant, and $$\rho_k=\rho^k$$ for $$k>\bar n$$. We write $$h_+$$ for the restriction of a pair amplitude to timelike ($$s^2>0$$) separations.

<div class="proposition" markdown="1">

**Proposition 9** (Janossy densities, total variation, Born reading). For the pair family on a bounded region $$D$$, with $$\bar N=\rho\lvert D\rvert$$:

(a) $$j_n=\rho^ne^{-\bar N}$$ for $$n\ge3$$, $$j_2=\rho^2(e^{-\bar N}+h)$$, $$j_1(x)=\rho e^{-\bar N}-\rho^2\int_Dh(x,y)dy$$, $$j_0=e^{-\bar N}+\frac12\rho^2\int\!\!\int_{D^2}h$$.

(b) With $$\mathcal H_D=\rho^2\int\!\!\int_{D^2}\lvert h\rvert$$: $$\frac12\mathcal H_D-1\le\lVert\mathsf P_D\rVert\le1+2\mathcal H_D$$. For diamonds and $$h\in L^1(\mathbb R,dw)$$ with $$w=s^2/\ell^2$$, $$\mathcal H_D=\rho^2\ell^2T^6\int\lvert h(w)\rvert\,\omega(w\ell^2/T^2)dw\to\omega(0)\rho^2\ell^2\lVert h\rVert_1T^6$$, so $$\lVert\mathsf P_D\rVert\asymp T^6\asymp V^{3/2}$$ for every $$h\not\equiv0$$.

(c) If $$h\not\equiv0$$, $$\mathsf P_D$$ is not a positive measure for all large $$D$$: no Born reading exists.

</div>

<div class="proof" markdown="1">

*Proof.* (a) is Theorem 7: only terms with $$n+l=2$$ differ from the Poisson ones. (b) For the upper bound, the Poisson parts have total variation 1 and the perturbations contribute at most $$\frac12\rho^2\!\int\!\!\int\lvert h\rvert+\rho^2\!\int\!\!\int\lvert h\rvert+\frac12\rho^2\!\int\!\!\int\lvert h\rvert=2\mathcal H_D$$. For the lower bound, $$\frac12\int\!\!\int\lvert j_2\rvert\ge\frac12(\mathcal H_D-\rho^2\lvert D\rvert^2e^{-\bar N})\ge\frac12\mathcal H_D-\frac12$$, using $$\bar N^2e^{-\bar N}\le4e^{-2}<1$$. The asymptotics follow by dominated convergence as in Proposition 8. (c) Positivity of $$\mathsf P_D$$ would need $$j_2\ge0$$ almost everywhere, hence $$h\ge-e^{-\bar N}$$, and for $$D$$ large $$h\ge0$$ almost everywhere. Then $$j_1(x)\ge0$$ needs $$\rho\int_Dh(x,y)dy\le e^{-\bar N}$$ for almost every $$x\in D$$. But $$\int_Dh(x,y)dy$$ increases with $$D$$ to $$\int_{\mathbb M}h>0$$ while $$e^{-\bar N}\to0$$. $$\square$$

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

(c) If $$h_+=0$$ almost everywhere, both means equal the Poisson means at every $$\rho$$. If $$c_0\ne0$$ and $$h_+\ne0$$ on a set of positive measure, then at fixed $$\ell$$ the placement-A mean does not have a limit as $$\rho\to\infty$$ for all $$\varphi\in C_c^\infty$$: the extra term is $$\rho^{2/d+1}$$ times a $$\rho$$-independent functional that is non-zero for some $$\varphi$$ and $$x$$.

</div>

The weights depend on a single interval count, and the only correlation function that enters the Palm weighting and is not Poisson is $$\rho_2$$ of a pair that is causally related, hence timelike.

<div class="proof" markdown="1">

*Proof.* (a) Write $$P^{xy}(N=k)$$ for the probability that $$I(y,x)$$ contains exactly $$k$$ points, given that $$x$$ and $$y$$ are points. The Palm weight of the pair is $$\rho_2P^{xy}(N=k)=\sum_{j\ge k}\frac{(-1)^{j-k}}{k!(j-k)!}\int_{I^j}\rho_{2+j}(x,y,z)\,dz$$ with $$I=I(y,x)$$. Only $$\rho_2$$ is perturbed, which is the term $$j=k=0$$, so $$\rho_2P^{xy}(N=k)=\rho^2[\mathrm{Poi}_{\bar\mu}(k)+h\,\delta_{k0}]$$ with $$\bar\mu=\rho V(y,x)$$. Dividing by $$\rho_1=\rho$$ and summing against $$c_k$$ and $$\varphi(y)$$ gives the extra term in (a), with $$h$$ evaluated on the timelike pair $$y\prec x$$.

(b) Without the point $$x$$ in the configuration, the weight of $$y$$ with $$k$$ points in the interval is $$\rho_1P^y(N=k)=\sum_{j\ge k}\frac{(-1)^{j-k}}{k!(j-k)!}\int_{I^j}\rho_{1+j}(y,z)\,dz$$. Only $$\rho_2$$, the term $$j=1$$, is perturbed; it enters $$k=0$$ with the sign $$-1$$ and $$k=1$$ with the sign $$+1$$, each as $$\rho^2\int_{I(y,x)}h(y,z)dz$$. The pair $$(y,z)$$ with $$z\in I(y,x)$$ satisfies $$y\prec z$$ and is timelike. Summing against $$c_k$$ gives $$(c_1-c_0)$$ times this integral.

(c) If $$h_+=0$$ both extra terms vanish. If not, the term in (a) is $$\kappa_d\rho^{2/d+1}c_0$$ times $$T_\ell\varphi(x)=\int_{J^-(x)}h_+(\tau^2/\ell^2)\varphi(y)dy$$, a retarded Lorentz-invariant functional independent of $$\rho$$. A function $$h_+(\tau^2/\ell^2)$$ that is not zero almost everywhere on $$J^-(0)$$ has a non-zero integral against some $$\varphi\in C_c^\infty$$ at some $$x$$, while the Poisson part of the mean converges. $$\square$$

</div>

A pair amplitude supported on spacelike separations is therefore invisible to the operator, and to every generalised layer kernel whose weights depend on interval counts. Its signature is elsewhere: by the first equality in (9) the number variance contains the term $$\rho^2\!\int\!\!\int_{D^2}h$$, and for an amplitude with $$\int h\,d\sigma=0$$ the leading $$V^{3/2}$$ term cancels. For $$\bar n>2$$ the $$k$$-point amplitudes also enter the means through configurations of the form (endpoint, interval points, endpoint); this page does not analyse them, so Proposition 10 is a statement about the pair family.

### 5.2 Exactly Poisson above $$\bar n$$ points

<div class="theorem" markdown="1">

**Theorem 11** (order pushforward). Let $$P$$ be a finite-order family of order $$\bar n$$. (a) On every bounded $$D$$, the signed measure $$\mathsf P_D$$ and the Poisson measure $$\mathrm{Poi}_D$$ coincide on the configurations with more than $$\bar n$$ points. (b) Let $$F$$ be a function of finite configurations in $$D$$ with $$\lvert F(Y)\rvert\le c_Fa^{\lvert Y\rvert}$$ for some $$c_F,a$$, and put $$F^\sharp(Z)=\sum_{Y\subseteq Z}(-1)^{\lvert Z\setminus Y\rvert}F(Y)$$. Then

$$
\mathbb E_PF-\mathbb E_{\rm Poi}F=\sum_{k=2}^{\bar n}\frac{\rho^k}{k!}\int_{D^k}h_k(z)\,F^\sharp(z)\,dz. \tag{12}
$$

If $$F$$ depends only on the causal order of the configuration, $$F^\sharp(z)$$ depends only on the order type of $$z$$, and the right side is a finite signed combination of expectations over posets with at most $$\bar n$$ elements.

</div>

Here the *order type* of a $$k$$-point configuration is the isomorphism class of the causal order it induces; for $$k=2$$ there are two, the two-element chain (a related pair) and the two-element antichain (an unrelated pair).

<div class="proof" markdown="1">

*Proof.* (a) For $$n>\bar n$$ all $$\rho_{n+m}$$ are Poisson, so $$j_n$$ equals the Poisson Janossy density. (b) Absolute convergence: $$\lvert j_n\rvert\le c\,\rho^ne^{\rho\lvert D\rvert}$$ for a constant $$c$$, so $$\sum_n\frac1{n!}\int\lvert j_nF\rvert\le c'\sum_n(a\rho\lvert D\rvert)^n/n!<\infty$$, which allows rearrangement. Inserting the formula for $$j_n$$ and grouping $$z=(x,y)$$ by $$k=n+l$$,

$$
\mathbb E_PF=\sum_n\frac1{n!}\int j_nF=\sum_k\frac1{k!}\int\rho_k(z)\sum_{Y\subseteq z}(-1)^{\lvert z\setminus Y\rvert}F(Y)\,dz,
$$

by the symmetry of $$\rho_k$$ and $$F$$ (the binomial counts the choices of the subset $$Y$$ of size $$n$$). Subtracting the same identity for $$\rho_k=\rho^k$$ leaves only $$k\le\bar n$$. If $$F$$ depends on the order only, the order on $$Y\subseteq z$$ is the restriction of that on $$z$$. $$\square$$

</div>

A finite analogue of (12) (five sites, an occupation process with $$\rho=0.3$$, random bounded $$h_{ij}\in[-0.7,0.7]$$ giving signed Janossy weights down to $$-0.04$$, and a random $$F$$) has the two sides of (12) equal to $$14$$ digits, and for an $$F$$ that vanishes on configurations with at most two points the difference of expectations is exactly $$0$$.

Two consequences are used in section 5.3. For a $$k$$-point order type $$\tau$$ let $$U_\tau$$ be the number of $$k$$-subsets of type $$\tau$$. Then $$U_\tau^\sharp(z)=\mathbb 1[z\text{ has type }\tau]$$ for $$\lvert z\rvert=k$$ and $$0$$ for $$\lvert z\rvert\ne k$$, so for $$k\le\bar n$$

$$
\mathbb E_PU_\tau=\mathbb E_{\rm Poi}U_\tau+\frac{\rho^k}{k!}H_k^\tau(D),\qquad H_k^\tau(D)=\int_{D^k}h_k\mathbb 1_\tau, \tag{13}
$$

and the normalised statistic $$u_\tau=U_\tau/\mathbb E_{\rm Poi}U_\tau$$ has $$\mathbb E_Pu_\tau=1+\zeta_D^\tau$$ with $$\zeta_D^\tau=\rho^kH_k^\tau(D)/(k!\,\mathbb E_{\rm Poi}U_\tau)$$. Second, for $$p>\bar n$$ let $$u_p=(N)_p/(\rho\lvert D\rvert)^p$$, where $$(N)_p$$ is the falling factorial. The statistics $$u_p$$, $$u_p^2$$ and $$u_\tau u_p$$ vanish on configurations with fewer than $$p$$ points, hence have $$F^\sharp(z)=0$$ for $$\lvert z\rvert\le\bar n$$, and their expectations under $$P$$ and under the Poisson process are identical; in particular $$\mathbb E_Pu_p=1$$ exactly.

### 5.3 Positivity conditions on the order statistics

Consider the scaled regions $$D=L\,D_0$$ for a fixed bounded $$D_0$$ and $$L\to\infty$$, for which $$\rho\lvert D\rvert\to\infty$$. The statistic $$u_\tau$$ is a *macroscopic density of order type*: the number of $$k$$-subsets of type $$\tau$$ relative to its Poisson expectation, for example the fraction of pairs that are related. For a Poisson process $$u_\tau\to1$$. The question is whether a finite-order perturbation can leave a different imprint on these large-scale order statistics, that is, whether $$\zeta_D^\tau$$ can stay non-zero as $$L\to\infty$$.

For a positive process the densities are random variables, so their variances and covariances are non-negative and their joint law exists. A signed process has no Born reading and no such guarantee. We take the requirement that the densities behave like random variables as a model assumption on the signed process, in three forms of increasing strength, and ask which perturbations survive each. Limits are along the chosen sequence of regions, or along subsequences if they do not exist, and variances are computed formally as $$\mathbb E[u^2]-(\mathbb Eu)^2$$.

(V1) *Single-variance form.* The limiting variance of each $$u_\tau$$ is non-negative.

(V2) *Joint-variance form.* For every finite family of the densities (the $$u_\tau$$ and the $$u_p$$), the limiting covariance matrix is positive semi-definite.

(V3) *Moment form.* There is a probability measure $$\pi$$ on $$\mathbb R^J$$ such that $$\mathbb E_Pg(u)\to\int g\,d\pi$$ for every polynomial $$g$$ of the densities $$u=(u_1,\dots,u_J)$$.

(V3) implies (V2), by taking $$g$$ the square of a linear combination of the centred densities, and (V2) implies (V1). The converses fail: the example after Proposition 12 satisfies (V1) and not (V2). A weaker *in-law form* asks only that $$\mathbb E_Pg(u)\to\int g\,d\pi$$ for bounded continuous $$g$$, and needs no moments.

<div class="proposition" markdown="1">

**Proposition 12** (two-density witness). Let $$P$$ be a finite-order family of order $$\bar n$$, $$k\le\bar n$$, $$\tau$$ a $$k$$-point order type, $$p>\bar n$$, and suppose $$\limsup\lvert\zeta_D^\tau\rvert>0$$ along the scaled regions. Then along those regions

$$
\mathrm{Cov}_P(u_\tau,u_p)=-\zeta_D^\tau+o(1),\qquad \mathrm{Var}_P(u_p)\to0,\qquad \mathrm{Var}_P(u_\tau)=O(1),
$$

so $$\det\mathrm{Cov}_P(u_\tau,u_p)=-(\zeta_D^\tau)^2+o(1)$$, and for suitable real $$t$$ the variance $$\mathrm{Var}_P(u_\tau+t\,u_p)$$ is negative in the limit. Thus (V2), and hence (V3), fails whenever some type average $$\zeta^\tau$$ is non-zero in the limit.

</div>

By Theorem 11 the perturbation does not affect $$u_p$$, which keeps mean 1 and a variance tending to zero, while $$u_\tau$$ has its mean shifted by $$\zeta$$ and a variance of order one. For two random variables Cauchy–Schwarz gives $$\lvert\mathrm{Cov}(u_\tau,u_p)\rvert\le\sqrt{\mathrm{Var}\,u_\tau\,\mathrm{Var}\,u_p}\to0$$, whereas here the covariance tends to $$-\zeta$$.

<div class="proof" markdown="1">

*Proof.* By (13), $$\mathbb E_Pu_\tau=1+\zeta_D^\tau$$. By the second consequence of Theorem 11, $$\mathbb E_Pu_p=1$$, $$\mathrm{Var}_Pu_p=\mathrm{Var}_{\rm Poi}u_p$$ and $$\mathbb E_P[u_\tau u_p]=\mathbb E_{\rm Poi}[u_\tau u_p]$$. For the Poisson process on $$LD_0$$ the densities $$u_\tau$$ and $$u_p$$ converge in $$L^2$$ to 1: the variance of a $$U$$-statistic with bounded kernel on $$\rho\lvert D\rvert$$ points is $$O((\rho\lvert D\rvert)^{2k-1})$$ against a squared mean of order $$(\rho\lvert D\rvert)^{2k}$$, the kernel being the indicator of a scale-invariant set of positive measure. Hence $$\mathrm{Cov}_{\rm Poi}(u_\tau,u_p)\to0$$ and $$\mathrm{Var}_{\rm Poi}(u_p)\to0$$. This gives the covariance and the variance of $$u_p$$. $$\mathrm{Var}_P(u_\tau)$$ is bounded because, by (12) with $$F=U_\tau^2$$, the correction involves $$F^\sharp(z)$$ for $$\lvert z\rvert\le\min(\bar n,2k)$$, bounded by $$2^{2k}\binom{\lvert z\rvert}{k}^2$$, times integrals of $$h_j$$ of order $$(\rho\lvert D\rvert)^{j}$$ against a denominator $$(\rho\lvert D\rvert)^{2k}$$, a bounded quantity. Then $$\det=\mathrm{Var}\,u_\tau\mathrm{Var}\,u_p-\mathrm{Cov}^2\to-\zeta^2$$, and $$\mathrm{Var}(u_\tau+tu_p)=\mathrm{Var}\,u_\tau+2t\mathrm{Cov}+t^2\mathrm{Var}\,u_p$$ is, for $$t=\mathrm{sign}(\zeta)T$$ with $$T$$ large, negative when the regions are large. $$\square$$

</div>

The content of Proposition 12 lies in amplitudes that do not decay with separation. For the pair family with $$h\in L^1$$ in $$w=s^2/\ell^2$$ and $$D_0$$ a diamond, $$\lvert H_2^\tau(LD_0)\rvert\le\int\!\!\int\lvert h\rvert=O(L^6)$$ by the scaling in Proposition 8, while $$\mathbb E_{\rm Poi}U_\tau\asymp L^8$$, so $$\zeta^\tau=O(L^{-2})$$ and the hypothesis of the proposition is not met. A constant amplitude on an order type, such as $$h_2=\epsilon\,\mathbb 1[\text{spacelike}]$$, has $$\zeta\ne0$$ at every volume.

That example also separates (V1) from (V2). Take $$\bar n=2$$ and $$h_2=\epsilon\,\mathbb 1[\text{spacelike}]$$. The density $$u_{A_2}$$ of antichain pairs has $$\mathbb E_Pu_{A_2}=1+\epsilon$$. By (12) with $$F=U_{A_2}^2$$, whose $$F^\sharp(z)$$ for $$\lvert z\rvert=2$$ is the indicator of an antichain, the second moment shifts by $$\epsilon\,\mathbb E_{\rm Poi}U_{A_2}$$, which is $$o((\mathbb E_{\rm Poi}U_{A_2})^2)$$. Hence $$\mathrm{Var}(u_{A_2})\to-2\epsilon-\epsilon^2$$, which is positive for $$-2<\epsilon<0$$ and negative otherwise. For such $$\epsilon$$ (V1) holds, while (V2) fails because $$\det\mathrm{Cov}(u_{A_2},u_3)\to-\epsilon^2$$.

### 5.4 Ergodic components are order-blind, and a lattice that is not covered

Proposition 12 concerns finite-order families. For processes with bounded total variation, the classification on [Lorentz-invariant quantum sprinklings](/quantum-gravity/quantum-sprinklings/) reduces everything to positive ergodic processes, whose large-scale order content is that of a Poisson process of the same intensity.

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

**Corollary 14** (bounded total variation: no macroscopic order content). Let $$P=\int Q_\theta\,dm(\theta)$$ be a signed or complex combination, with $$\lvert m\rvert$$ finite, of translation-ergodic positive processes $$Q_\theta$$ of finite intensities $$\lambda_\theta$$. Then for every finite family of order types $$\tau_1,\dots,\tau_J$$ (with $$k_i$$ points), every bounded continuous $$g$$ on $$\mathbb R^J$$ and $$\rho=\int\lambda_\theta dm$$, $$\mathbb E_Pg(u_{\tau_1},\dots,u_{\tau_J})\to\int g\big((\lambda_\theta/\rho)^{k_1},\dots,(\lambda_\theta/\rho)^{k_J}\big)dm(\theta)$$, where $$u_\tau=u_\tau(LD_0)$$.

</div>

This is the form taken by a Poincaré-invariant process of bounded total variation: by Theorem 10 of the page on quantum sprinklings it is a combination of positive ergodic processes, and decomposing further into translation-ergodic components gives the stated form when those components have finite intensity. The corollary follows from Proposition 13, $$\mathbb E_{{\rm Poi}(\lambda)}U_\tau/\mathbb E_{{\rm Poi}(\rho)}U_\tau=(\lambda/\rho)^k$$, bounded convergence under each $$Q_\theta$$ and dominated convergence against $$\lvert m\rvert$$. The limit depends on the type only through the number of points. If the intensities $$\lambda_\theta$$ of the components are all equal to $$\rho$$, every $$u_\tau$$ tends to 1 in law whatever the signs of $$m$$, jointly for any finite family of types.

This last situation is realised by a combination that is not a mixed Poisson process. Let $$L_\rho$$ be the Haar-random affine lattice of density $$\rho$$ in four dimensions (Proposition 9 of the page on quantum sprinklings), and take $$\epsilon>0$$.

<div class="proposition" markdown="1">

**Proposition 15** (a bounded-variation combination outside the classes above). $$P_\epsilon=(1+\epsilon)\,\mathrm{Poi}(\rho)-\epsilon\,L_\rho$$ is a Poincaré-invariant complex point process of total variation $$\sup_D\lVert\mathsf P_D\rVert=1+2\epsilon$$. Its densities satisfy $$u_\tau\to1$$ in law for every order type, so it satisfies the in-law form of the positivity condition. It is not a mixed Poisson process and not a member of any finite-order family, because its third factorial moment measure charges a Lebesgue-null set, and it is not a positive measure.

</div>

<div class="proof" markdown="1">

*Proof.* Invariance is that of the two components. The ergodic components of $$\mathrm{Poi}(\rho)$$ (itself) and of $$L_\rho$$ (fixed lattices shifted uniformly, which are translation ergodic) all have intensity $$\rho$$, so Corollary 14 gives $$u_\tau\to1$$ in law. Let $$\Omega$$ be a bounded set of positive volume, $$\Omega'$$ a bounded set containing $$\{2y-x:x,y\in\Omega\}$$, $$\mathcal R=\{(x,y,z):z=2y-x\}$$, and $$\Xi=\sum^{\ne}_{x,y,z}\mathbb 1_\Omega(x)\mathbb 1_\Omega(y)\mathbb 1_{\Omega'}(z)\mathbb 1_{\mathcal R}(x,y,z)$$. On the lattice $$\Xi=N(\Omega)(N(\Omega)-1)$$, with expectation $$V(\Omega)^2$$ (proof of Proposition 9 of the page on quantum sprinklings). On a Poisson process the triples in $$\mathcal R$$ form a Lebesgue-null set of $$\mathbb M^3$$, so $$\mathbb E_{\rm Poi}\Xi=0$$. Hence $$\mathbb E_{P_\epsilon}\Xi=-\epsilon V(\Omega)^2\ne0$$. For a process with function-valued correlation functions (a mixed Poisson process, or a member of a finite-order family as in Theorem 7) $$\mathbb E\,\Xi=\int\mathbb 1_\Omega\mathbb 1_\Omega\mathbb 1_{\Omega'}\mathbb 1_{\mathcal R}\rho_3=0$$. Since $$\Xi\ge0$$ and $$\mathbb E_{P_\epsilon}\Xi<0$$, $$P_\epsilon$$ is not a positive measure. $$\square$$

</div>

The moment forms (V1)-(V3) are not shown for $$P_\epsilon$$: the variance of $$u_\tau$$ involves moments of the lattice counts of order $$2k$$, and the moment formulas available for Siegel transforms of random lattices cover orders up to $$d-1=3$$ only. The mean of the operator on $$P_\epsilon$$ is $$(1+\epsilon)$$ times the Poisson mean minus $$\epsilon$$ times the mean on the lattice component, by linearity, and the latter is not computed here.

## 6. Matrix-valued kernels and quantum posets

A quantum poset replaces the points of a poset by matrix algebras and the order by a quantum relation. In the version of Weaver [[Weaver 2012]](#ref-weaver12) and Kornell, Lindenhovius and Mislove [[KLM]](#ref-klm), restricted to the finite-dimensional case, the algebra is $$\bigoplus_xM_{n_x}$$ acting on $$\bigoplus_x\mathbb C^{n_x}$$, and a relation is a family of subspaces $$\mathcal V_{xy}\subset\mathrm{Hom}(\mathbb C^{n_y},\mathbb C^{n_x})$$ that contains the identities, is closed under composition and satisfies an antisymmetry condition; Weaver [[Weaver 2021]](#ref-weaver21) studies the hereditarily antisymmetric case. A sprinkled quantum causal set assigns to each element of a Poisson sprinkling a random matrix size $$n_x\le n_{\max}$$ and to each related pair a relation space. A d'Alembertian on it acts on fields valued in $$\bigoplus_xM_{n_x}$$, and its mean over the sprinkling and the marks is a matrix-valued, Lorentz-invariant, retarded kernel. The argument below uses nothing about quantum posets beyond three properties of that kernel, stated as hypotheses; whether a given construction of the interval operator satisfies them is a separate verification, and the case of bounded sizes and interval-local link data is discussed on [the four-dimensional causal-set d'Alembertian has no positive spectral representation](/quantum-gravity/causal-box-spectrum/).

Let the mean operator act on $$\mathbb R^r$$-valued fields as $$(\bar B\phi)(x)=a\,\phi(x)+\int_{J^-(x)}F(V(x,y))\phi(y)\,dy$$ in $$\mathbb R^{1,3}$$, with $$a$$ a real number and $$F$$ an $$r\times r$$ matrix function of the expected interval count $$V=\rho v_4\tau^4$$. On an exponential field $$\bar B$$ acts as multiplication by the matrix $$b(E)=a\mathbb 1+\hat F(E)$$ of $$Z=E^2$$, where $$\hat F(E)=4\pi\int_0^\infty F(V(\tau))\tau^3K_1(E\tau)/(E\tau)\,d\tau$$ by the computation of section 2. Put $$\mathcal G(Z)=-b(E)^{-1}$$, assuming $$b(E)$$ invertible for $$E>0$$. We say $$\mathcal G$$ is *Stieltjes* if $$\mathcal G(Z)=R_0/Z+C_0+\int\Pi(d\mu)/(Z+\mu)$$ with $$R_0,C_0\ge0$$ and $$\Pi$$ a positive-matrix-valued measure with $$\int(1+\mu)^{-1}\mathrm{tr}\,\Pi<\infty$$. For a scalar function this says that the Euclidean propagator is a superposition of free propagators with non-negative weights; identifying that with the absence of ghosts for a retarded non-local operator is a convention, discussed on the page just cited.

<div class="lemma" markdown="1">

**Lemma 16** (a Stieltjes propagator with a massless pole does not decay fast). Let $$\mathcal G$$ be Stieltjes with $$R_0$$ positive definite and let $$v\ne0$$. Then $$R_v(Z)=Z\langle v,(\mathcal G(Z)-C_0)v\rangle$$ satisfies $$R_v(Z)\ge\langle v,R_0v\rangle>0$$ for all $$Z>0$$, and $$R_v$$ is non-decreasing.

</div>

<div class="proof" markdown="1">

*Proof.* $$R_v(Z)=\langle v,R_0v\rangle+\int\frac Z{Z+\mu}\langle v,\Pi(d\mu)v\rangle$$, and $$Z/(Z+\mu)$$ is non-negative and non-decreasing in $$Z$$. $$\square$$

</div>

<div class="theorem" markdown="1">

**Theorem 17** (interval-local matrix kernels do not give a Stieltjes propagator). Let $$\bar B$$ be as above, with the following properties.

(Q1) *Interval-locality.* $$F(V)=e^{-V}\sum_{k\ge0}g_kV^k/k!$$, where the matrix $$g_k$$ is the mean weight attached to a related pair with $$k$$ interior points, and depends only on the data of the points in the closed interval.

(Q2) *Boundedness.* $$\sup_k\lVert g_k\rVert<\infty$$ and $$\int^\infty\lVert F\rVert\tau\,d\tau<\infty$$ (the last holds, for example, if $$F$$ decays exponentially in $$V$$).

(Q3) *Infrared normalisation.* $$a<0$$ and $$b(E)=-GZ+o(Z)$$ as $$Z\to0$$ with $$G$$ positive definite.

Then $$\mathcal G=-b^{-1}$$ is not Stieltjes.

</div>

The contact term and the infrared normalisation fix the constant and the residue of $$\mathcal G$$, while the boundedness of the kernel at the apex ($$V\to0$$) makes $$\mathcal G-C_0$$ decay faster than $$1/Z$$, which Lemma 16 forbids.

<div class="proof" markdown="1">

*Proof.* We have $$E^2\hat F(E)=4\pi\int F\,\tau\,[(E\tau)K_1(E\tau)]\,d\tau$$. The function $$zK_1(z)$$ decreases from 1 to 0 on $$(0,\infty)$$, and by (Q1)-(Q2) $$F$$ is bounded, with $$F(0)=g_0$$, so dominated convergence gives $$E^2\hat F(E)\to0$$, that is, $$\hat F=o(1/Z)$$ entrywise. No term concentrated on the light cone is present. With $$a<0$$, $$\mathcal G=-(a+\hat F)^{-1}$$ tends to $$C_0:=-a^{-1}\mathbb 1\ge0$$ as $$Z\to\infty$$, and $$\mathcal G-C_0=a^{-2}\hat F(1+\hat F/a)^{-1}=o(1/Z)$$. If $$\mathcal G$$ were Stieltjes, the residue at $$Z=0$$ would be $$R_0=G^{-1}$$ by (Q3) and $$\mathcal G(\infty)=C_0$$ would be its constant. Lemma 16 gives $$R_v(Z)\ge\langle v,G^{-1}v\rangle>0$$, whereas $$R_v(Z)=Z\langle v,(\mathcal G-C_0)v\rangle\to0$$. $$\square$$

</div>

For the scalar four-dimensional weights (1), (Q1)-(Q3) hold with $$a=-\kappa_4\rho^{1/2}$$ and $$G=1$$, and $$R(Z)=Z[\mathcal G(Z)-\mathcal G(\infty)]$$ evaluated from (4) at $$\rho=1$$ is $$0.15069,\ 0.015387,\ 0.0015391$$ at $$Z=10^2,10^3,10^4$$, falling as $$1/Z$$. The theorem differs from the matrix-valued theorem on the causal-box page (Theorem 11 there) in using the normalisation (Q3) in place of a polynomial bound on $$F$$ and a symmetry of $$F$$.

The conclusion applies to a decoration by quantum-poset data exactly when the mean kernel of that decoration has the form (Q1)-(Q3). A kernel that depends on a pair $$(x,y)$$ only through interval-local data and the volume $$V$$ has the finite value $$g_0$$ at $$V=0$$ and so cannot carry a term $$\nu\,\delta(\tau^2)\theta(t)$$ concentrated on the light cone, which would give $$\hat F\sim\nu/Z$$ and is excluded from the argument; the page on the causal-box spectrum treats such terms. Decorations that depend on global data, for example a relation angle fixed by the Wightman function of a state built from the whole order, are not covered. The mean over the sprinkling is also not the per-realisation operator: if the matrix sizes are random, the sum $$\sum_zn_z^2$$ over the points of a region is a compound Poisson variable with $$\mathrm{Var}/\mathrm{mean}=\mathbb E[n^4]/\mathbb E[n^2]$$, which exceeds 1 as soon as the sizes are not all 1.

## 7. Open points

The mean of the four-dimensional operator on the Haar-random lattice is not known, and with it the mean on $$P_\epsilon$$ of Proposition 15. The page on [invariant point sets](/quantum-gravity/invariant-point-sets/) reports that in two dimensions the zeroth moment of the first four layers diverges on this lattice; in four dimensions it is open whether the mean is finite, and whether $$C_A=1$$ can be reached by adding Poisson components while keeping the in-law positivity condition.

Proposition 5 concerns mixed Poisson processes whose counts are probabilities. Whether a signed mixing law with $$C_A=1$$ can be realised as a state on a Hilbert space with a positive-operator-valued measure for the counts in which the counts are not the diagonal observables of the mixing parameter is open; the weak-value representation of section 3.3 is the one considered here, with the post-selection cost stated there.

Proposition 10 is proved for a correlation length $$\ell$$ fixed as $$\rho\to\infty$$. If $$\ell$$ scales with the discreteness length ($$\rho\ell^4$$ fixed), the extra term in (10) has the same order, $$\rho^{1/2}$$, as the Poisson part, and the regime is not analysed. The classification of finite-order perturbations does not reach processes with correlation functions of all orders modified, such as the Feynman determinantal kernel, for which only the local form of Proposition 6 is derived.

For quantum posets, Theorem 17 covers interval-local data. Whether non-local, order-intrinsic dressing of the data produces a term concentrated on the light cone, and whether the resulting operator can have a Stieltjes propagator, is open.

## References

1. <a id="ref-bd"></a>Benincasa, Dowker, "The scalar curvature of a causal set," Phys. Rev. Lett. 104 (2010) 181301, [arXiv:1001.2725](https://arxiv.org/abs/1001.2725). Used for: the four-dimensional operator (1) and its continuum limit.
2. <a id="ref-sorkin07"></a>Sorkin, "Does Locality Fail at Intermediate Length-Scales," [arXiv:gr-qc/0703099](https://arxiv.org/abs/gr-qc/0703099). Used for: the two-dimensional operator.
3. <a id="ref-dg"></a>Dowker, Glaser, "Causal set d'Alembertians for various dimensions," Class. Quantum Grav. 30 (2013) 195016, [arXiv:1305.2588](https://arxiv.org/abs/1305.2588). Used for: the operator in each dimension.
4. <a id="ref-ass"></a>Aslanbeigi, Saravani, Sorkin, "Generalized causal set d'Alembertians," JHEP 06 (2014) 024, [arXiv:1403.1622](https://arxiv.org/abs/1403.1622). Used for: the action on plane waves, the infrared conditions on the coefficients, and the generalised family.
5. <a id="ref-bbl"></a>Belenchia, Benincasa, Liberati, "Nonlocal scalar quantum field theory from causal sets," JHEP 03 (2015) 036, [arXiv:1411.6513](https://arxiv.org/abs/1411.6513). Used for: the non-local field theory defined by the operator and its instability in four dimensions.
6. <a id="ref-lp"></a>Last, Penrose, *Lectures on the Poisson Process*, Cambridge University Press, 2017. Used for: the Mecke formula and Cox processes.
7. <a id="ref-widder"></a>Widder, *The Laplace Transform*, Princeton University Press, 1941. Used for: Bernstein's theorem that a completely monotone function is the Laplace transform of a positive measure.
8. <a id="ref-nz"></a>Nguyen, Zessin, "Ergodic theorems for spatial processes," Z. Wahrscheinlichkeitstheorie verw. Gebiete 48 (1979) 133–158, [doi:10.1007/bf01886869](https://doi.org/10.1007/bf01886869). Used for: the spatial ergodic theorem for stationary point processes.
9. <a id="ref-weaver12"></a>Weaver, "Quantum relations," [arXiv:1005.0354](https://arxiv.org/abs/1005.0354). Used for: quantum relations and quantum posets.
10. <a id="ref-klm"></a>Kornell, Lindenhovius, Mislove, "A category of quantum posets," Indag. Math. 33 (2022) 1137–1171, [arXiv:2101.11184](https://arxiv.org/abs/2101.11184). Used for: the category of quantum posets.
11. <a id="ref-weaver21"></a>Weaver, "Hereditarily antisymmetric operator algebras," J. Inst. Math. Jussieu 20 (2021) 1039–1074, [arXiv:1902.09989](https://arxiv.org/abs/1902.09989). Used for: hereditarily antisymmetric algebras as quantum posets.
