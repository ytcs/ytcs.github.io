---
title: The causal-set d'Alembertian has negative spectral weight in four dimensions
date: 2026-10-29
category: gravity
order: 5
description: "A retarded, Lorentz-invariant d'Alembertian built from a sprinkling that agrees with the continuum operator to all orders needs a non-local kernel with tail exponent at most one half and never has a positive spectral representation; in four dimensions no operator of the generalised Benincasa-Dowker type has one, and non-commutative points do not change this."
status: "Theorems proved in full for retarded kernels in two and four dimensions; the numbers for the four-dimensional Benincasa-Dowker operator (complex zero pair, spectral density, sum rule) are numerical; the quantum-poset case is proved for a stated class of matrix-valued kernels."
excludes: "Positive Kallen-Lehmann spectral weight for retarded Lorentz-invariant d'Alembertians whose fourth-order coefficient in momentum vanishes or is negative, including every one that agrees with the continuum operator to all orders, and for every four-dimensional such operator with a polynomially bounded kernel and no light-cone term, including all finite-layer causal-set operators with commutative or bounded non-commutative points."
---

A causal set replaces spacetime by a discrete set of points ordered by causality. Sorkin's construction of a d'Alembertian on it [[Sorkin]](#ref-sorkin), completed by Benincasa and Dowker [[BD]](#ref-bd), sums a field over the elements to the past of a point with integer weights that depend on how many elements lie between the two. On a random discretisation of Minkowski space the expectation of this operator is a retarded, Lorentz-invariant, non-local operator, and one asks what the free field it defines looks like. Two requirements are natural. At low momentum the operator should reproduce $$-p^2$$ to every order, so that the discreteness is invisible. And the free propagator should have a positive spectral representation, the Källén–Lehmann form that a unitary theory has. We show that the first requirement forces a non-local kernel with a tail no lighter than $$e^{-c\sqrt\tau}$$ in proper time and is never compatible with the second; that in four dimensions no operator in the generalised Benincasa–Dowker family has the second property at all, whatever its weights, because the difference between its propagator and its large-momentum limit decays like $$p^{-4}$$, faster than the $$p^{-2}$$ that a positive spectral representation permits; that the exception we find among Lorentz-invariant retarded operators is a term supported on the light cone, which a finite-density causal set cannot carry; and that replacing points by finite-dimensional non-commutative atoms of bounded size does not change this.

Prior work gives the pieces. The operator in two and four dimensions is due to Sorkin and to Benincasa and Dowker, and Dowker and Glaser extended it to other dimensions [[Dowker–Glaser]](#ref-dg). Aslanbeigi, Saravani and Sorkin [[ASS]](#ref-ass) generalised it to a family with free layer weights, imposed the infrared conditions on them order by order, and found that the four-dimensional Benincasa–Dowker operator has a pair of complex zeros of the symbol, which signals a classical instability, while the two-dimensional one does not; they could not find weights in the family that remove the pair in four dimensions. Belenchia, Benincasa and Liberati [[BBL]](#ref-bbl) showed that the discontinuity of the two-dimensional propagator across the cut is positive and the four-dimensional one is not, and asked whether the improved ultraviolet behaviour of the four-dimensional operator is related to that. Section 3 proves that an operator which agrees with $$-p^2$$ to all orders has non-local kernel tails with exponent at most $$\tfrac12$$, constructs such operators for every smaller exponent, and shows that positivity forces a strictly positive fourth-order coefficient. Section 4 proves a superconvergence theorem which shows that in four dimensions, and in higher dimensions, positivity fails for every operator with a polynomially bounded kernel; it identifies the light-cone term as the only escape and gives an explicit example, and it recomputes the four-dimensional Benincasa–Dowker spectral data so that the sum rule behind the theorem can be seen in numbers. Section 5 treats matrix-valued kernels, which cover non-commutative points of bounded size [[Weaver]](#ref-weaver) [[KLM]](#ref-klm).

## 1. Causal sets and the mean-field operator

A *causal set* is a set with a partial order $$\prec$$ in which every interval $$I(x,y)=\{z:x\prec z\prec y\}$$ is finite [[BLMS]](#ref-blms). A *sprinkling* of density $$\rho$$ into Minkowski space $$\mathbb R^{1,d-1}$$ draws points from a Poisson process of density $$\rho$$ and orders them by the causal order of the spacetime; the number of points in a region of volume $$V$$ is Poisson distributed with mean $$\rho V$$. A sprinkling has no preferred frame or lattice in distribution, which is why it is the standard model of a Lorentz-invariant discrete structure [[Surya]](#ref-surya). We take units with $$\rho=1$$, so the discreteness length is one.

The Benincasa–Dowker operators act on a function $$\phi$$ on the causal set by

$$
(B\phi)(x)=\ell^{-2}\Big[a\,\phi(x)+\sum_{k\ge0}c_k\!\!\sum_{y\prec x,\ \lvert I(y,x)\rvert=k}\!\!\phi(y)\Big], \tag{1}
$$

the sum running over the elements $$y$$ to the past of $$x$$ with exactly $$k$$ elements between $$y$$ and $$x$$. In two dimensions $$(a;c_0,c_1,c_2)=2(-1;2,-4,2)$$ and in four dimensions $$(a;c_0,\dots,c_3)=\tfrac4{\sqrt6}(-1;1,-9,16,-8)$$, and the coefficients are chosen so that the expectation of $$B\phi$$ over sprinklings tends to $$\Box\phi$$ for smooth $$\phi$$ as the density grows [[BD]](#ref-bd) [[Dowker–Glaser]](#ref-dg). Aslanbeigi, Saravani and Sorkin [[ASS]](#ref-ass) allow arbitrary finitely many coefficients subject to conditions at low momentum; we call this the *generalised family*.

Given $$x$$, the other points form a Poisson process (the Mecke formula), so the probability that a point at $$y$$ has exactly $$k$$ points between it and $$x$$ is $$e^{-V}V^k/k!$$ with $$V=V(y,x)$$ the volume of the interval. The expectation of (1) is the *mean-field operator*

$$
(\bar B\phi)(x)=a\,\phi(x)+\int_{y\prec x}F(\tau(x,y))\,\phi(y)\,d^dy,\qquad F(\tau)=\sum_kc_k\,e^{-V}\frac{V^k}{k!}, \tag{2}
$$

where $$\tau$$ is the proper time from $$y$$ to $$x$$ and $$V=\tau^2/2$$ in two dimensions, $$V=\pi\tau^4/24$$ in four. This is a retarded, Lorentz-invariant operator with a delta term and a kernel $$F$$ that depends on $$\tau$$ only. Everything in sections 3 to 5 concerns operators of the form (2) with more general $$F$$, so we do not need the details of the sprinkling again.

Operators of the form (2) are diagonalised by exponentials in time. Take $$\phi(t,\vec x)=e^{Et}$$ with $$E>0$$; then $$\Box\phi=-E^2\phi$$, so the continuum operator has *symbol* $$-E^2=-p^2$$ at the Euclidean point $$p^2=E^2$$.

<div class="lemma" markdown="1">
**Lemma 1** (symbol). Let $$F$$ be measurable with $$\lvert F(\tau)\rvert\le C(1+\tau)^p$$. Then $$\bar B e^{Et}=b(E)e^{Et}$$ for $$E>0$$, with

$$
b(E)=a+\int_0^\infty F(\tau)\,\tau^{d-1}k_d(E\tau)\,d\tau,\qquad k_2(z)=2K_0(z),\quad k_4(z)=\frac{4\pi K_1(z)}{z}, \tag{3}
$$

where $$K_0,K_1$$ are modified Bessel functions. In any dimension $$k_d(z)=\Omega_{d-2}\int_0^\infty\sinh^{d-2}\eta\,e^{-z\cosh\eta}\,d\eta>0$$, with $$\Omega_{d-2}$$ the area of the unit $$(d-2)$$-sphere ($$\Omega_0=2$$). The function $$b$$ extends holomorphically to $$\operatorname{Re}E>0$$, that is to $$Z=E^2\in\mathbb C\setminus(-\infty,0]$$.
</div>

<div class="proof" markdown="1">
*Proof.* Writing $$y=x-s$$ with $$s$$ in the future cone, $$\bar Be^{Et}=e^{Et}\big(a+\int_{J^+}F(\tau(s))e^{-Es^0}d^ds\big)$$, and the integral converges absolutely because of the factor $$e^{-Es^0}$$. In four dimensions it equals $$\int_0^\infty dt\,e^{-Et}\int_0^t4\pi r^2F(\sqrt{t^2-r^2})dr$$. At fixed $$t$$ substitute $$\tau=\sqrt{t^2-r^2}$$, so $$r\,dr=-\tau\,d\tau$$ and $$r=\sqrt{t^2-\tau^2}$$, and exchange the order of integration: the integral is $$\int_0^\infty4\pi\tau F(\tau)\,d\tau\int_\tau^\infty e^{-Et}\sqrt{t^2-\tau^2}\,dt$$. The inner integral is $$\tau K_1(E\tau)/E$$, which gives (3). In two dimensions the same steps give $$2\tau K_0$$ with the inner integral $$\int_\tau^\infty e^{-Et}(t^2-\tau^2)^{-1/2}dt=K_0(E\tau)$$. The Bessel functions decay like $$e^{-\operatorname{Re}(E\tau)}$$, which gives the analytic continuation.
</div>

*Check.* For the two operators above we evaluated (3) by quadrature at 30 digits and, independently, the position-space integral of the proof at $$E=0.7$$: in four dimensions $$b=-0.3393737835$$ from (3) and $$-0.3393737838$$ from the position-space integral, in two dimensions $$-0.3274102579$$ and $$-0.3274102582$$. The ratio $$b/(-E^2)$$ at $$E=0.01,\,0.03,\,0.1$$ is $$0.99962,\,0.99736,\,0.98001$$ in four dimensions and $$0.99953,\,0.99679,\,0.97625$$ in two, so the operators reproduce $$-E^2$$ at low momentum, with a deviation that grows as $$E^2$$ (units of the discreteness length).

Timelike momenta correspond to the boundary values of $$b$$ at $$Z=-m^2\pm i0$$, with the sign fixed by the retarded prescription.

## 2. Spectral weight

The Euclidean propagator of the free field defined by $$\bar B$$ is $$D(Z)=-1/b$$. For $$b=-Z$$ it is $$1/Z$$. A function $$D$$ on $$Z>0$$ has *positive spectral weight* if it is a *Stieltjes function*,

$$
D(Z)=\frac{r_0}{Z}+c+\int_{(0,\infty)}\frac{\rho(d\mu)}{Z+\mu},\qquad r_0,c\ge0,\quad\rho\ge0,\quad\int\frac{\rho(d\mu)}{1+\mu}<\infty. \tag{4}
$$

This is the Källén–Lehmann form [[Källén]](#ref-kallen) [[Lehmann]](#ref-lehmann): in a unitary quantum field theory the two-point function of a scalar field is a superposition of free propagators of mass squared $$\mu$$ with weight $$\rho(\mu)$$ equal to the norm of the states of that invariant mass, which is non-negative. A negative $$\rho$$ marks states of negative norm, a ghost. For a function $$D$$ that is analytic in the cut plane away from isolated poles, the density is recovered from its discontinuity across the negative axis, $$\rho(\mu)=-\pi^{-1}\operatorname{Im}D(-\mu+i0)=+\pi^{-1}\operatorname{Im}(1/b)(-\mu+i0)$$. The sign is fixed by the example of the two-dimensional Benincasa–Dowker operator below, where this $$\rho$$ is nonnegative and integrates to one.

Three elementary properties of (4) are used repeatedly. First, $$D$$ is positive and nonincreasing on $$Z>0$$, so $$D(\infty)=c$$ exists. Second, the function $$T(Z)=Z\,[D(Z)-c]=r_0+\int\rho(d\mu)\,\frac{Z}{Z+\mu}$$ is nondecreasing, at least $$r_0$$, and is constant only if $$\rho=0$$. Third, $$D-r_0/Z$$ is nonincreasing with limit $$c+\int\rho/\mu\in[0,\infty]$$ at $$Z\to0^+$$ by monotone convergence.

Whether (4) is the right notion of unitarity for a retarded non-local operator is a separate matter taken up in section 6. The results below concern the function $$D$$ on $$Z>0$$ and say that it cannot have the form (4).

*Example.* In two dimensions the propagator of the Benincasa–Dowker operator has the form (4) to the accuracy of our computation. We computed $$\rho=\pi^{-1}\operatorname{Im}(1/b)$$ on $$Z=-\mu+i0$$ for $$10^{-7}\le\mu\le10^{4}$$: it is nonnegative (its minimum is $$7\times10^{-19}$$, at the level of quadrature error), and $$\int\rho\,d\mu=0.9999999$$ up to the cut at $$10^{-7}$$. With $$r_0=1$$, $$c=-1/a=\tfrac12$$, the representation $$1/Z+\tfrac12+\int\rho/(Z+\mu)$$ reproduces $$-1/b$$ at $$Z=0.5,2,10$$ to eight digits ($$3.0085203,\ 1.2386888,\ 0.67618227$$). This is numerical evidence that the two-dimensional operator has positive weight, not a proof.

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

The proposition says that positive spectral weight forces $$b=-Z+\kappa Z^2+o(Z^2)$$ with $$\kappa>0$$, or a logarithmically divergent coefficient when $$\kappa=\infty$$. Note the direction of the bound: $$b+Z$$ lies below $$\kappa Z^2$$ and approaches it only as $$Z\to0$$. For an operator in the class of Lemma 1 the symbol is bounded on $$Z\ge0$$, so $$b\ne-Z$$ and $$S\not\equiv0$$; such an operator cannot have positive weight if $$(b+Z)/Z^2$$ tends to zero, or to a negative number, as $$Z\to0^+$$.

*Check.* For the four-dimensional Benincasa–Dowker operator the quantity $$(b+Z)/Z^2$$ is $$2.00,\,3.79,\,5.58,\,7.38$$ at $$E=10^{-1},10^{-2},10^{-3},10^{-4}$$. It is positive and grows by $$1.795$$ per decade of $$E$$, which means $$\kappa=\infty$$ with a logarithm; the spectral density of section 4 tends to $$0.39$$ as $$\mu\to0$$, and $$0.39\ln100=1.795$$. So this operator passes the infrared test.

### Flat operators and the tail exponent

Call $$\bar B$$ *flat* if $$b(Z)=-Z+O(Z^N)$$ as $$Z\to0^+$$ for every $$N$$. Proposition 2, together with the boundedness of $$b$$, gives at once the following.

<div class="corollary" markdown="1">
**Corollary 3.** No flat operator of the form (2) has positive spectral weight.
</div>

The question is then whether flat operators exist at all, and with what kernel. The Mellin transform of the kernel, $$M_F(s)=\int_0^\infty F(\tau)\tau^{s-1}d\tau$$, answers it.

<div class="lemma" markdown="1">
**Lemma 4** (Mellin structure of the symbol). Let $$d\in\{2,4\}$$ and $$\lvert F(\tau)\rvert\le Ce^{-c\tau^\beta}$$ for all $$\tau>0$$, with $$c,\beta>0$$. Then $$M_F$$ is holomorphic in $$\operatorname{Re}s>0$$, and for $$E>0$$

$$
b(E)-a=\frac1{2\pi i}\int_{c_0-i\infty}^{c_0+i\infty}M_k(w)\,M_F(d-w)\,E^{-w}dw, \tag{6}
$$

with $$M_k(w)=2^{w-1}\Gamma(w/2)^2$$ and $$0<c_0<2$$ for $$d=2$$, and $$M_k(w)=\pi2^{w-1}\Gamma(w/2-1)\Gamma(w/2)$$ and $$2<c_0<4$$ for $$d=4$$. The function $$M_k$$ has double poles at $$w=-2k$$ for $$k\ge0$$, with $$M_k(w)=A_k(w+2k)^{-2}+B_k(w+2k)^{-1}+O(1)$$ and $$A_k\ne0$$ (for $$d=4$$, $$A_k=-2\pi4^{-k}/(k!(k+1)!)$$; for $$d=2$$, $$A_k=2\cdot4^{-k}/(k!)^2$$), and for $$d=4$$ a simple pole at $$w=2$$ with residue $$4\pi$$. Shifting the contour to the left gives, for every $$n$$, a small-$$E$$ expansion in which the coefficient of $$E^{2k}\ln E$$ is $$-A_kM_F(d+2k)$$ and the coefficient of $$E^{2k}$$ is $$-A_kM_F'(d+2k)+B_kM_F(d+2k)$$ (plus $$a$$ for $$k=0$$), preceded for $$d=4$$ by $$4\pi M_F(2)E^{-2}$$, with a remainder $$O(E^{2n-1})$$.
</div>

<div class="proof" markdown="1">
*Proof.* Equation (6) is the Mellin–Parseval formula for (3), since $$M_k$$ is the Mellin transform of $$k_d(z)$$ (checked from $$\int_0^\infty K_\nu(z)z^{\mu-1}dz=2^{\mu-2}\Gamma((\mu-\nu)/2)\Gamma((\mu+\nu)/2)$$) and $$\int F(\tau)\tau^{d-1}(E\tau)^{-w}d\tau=E^{-w}M_F(d-w)$$. The strip of $$c_0$$ is where both transforms converge. The function $$M_F(d-w)$$ is holomorphic for all $$\operatorname{Re}w<d$$ and bounded on vertical lines, while $$M_k$$ decays like $$e^{-\pi\lvert\operatorname{Im}w\rvert/2}$$, so the contour can be moved left through the poles of $$M_k$$ with vanishing contributions from the horizontal segments. The coefficients are the residues of the double poles, computed from the Laurent expansion of $$M_k$$ and the Taylor expansion of $$M_F(d-w)E^{-w}$$ at $$w=-2k$$.
</div>

Flatness therefore requires, order by order, that the logarithmic and the pure-power coefficients vanish except for the constant $$a$$ and the term $$-E^2$$.

<div class="corollary" markdown="1">
**Corollary 5.** If $$\bar B$$ is flat and $$F$$ satisfies the hypothesis of Lemma 4, then $$M_F(d+2k)=0$$ for all $$k\ge0$$ (for $$d=4$$ also $$M_F(2)=0$$), $$M_F'(d+2k)=0$$ for all $$k\ge2$$, $$M_F'(d+2)\ne0$$, and $$a=A_0M_F'(d)$$. That is, $$M_F$$ has zeros at $$d$$ and $$d+2$$, simple at $$d+2$$, and double zeros at $$d+2k$$ for $$k\ge2$$.
</div>

The expansions in $$E^{2k}\ln E$$ and $$E^{2k}$$ are asymptotically independent, so each coefficient must vanish separately; the coefficient of $$E^2$$ equals $$-1$$, which gives $$M_F'(d+2)\ne0$$. Kernels of the generalised family have $$F=e^{-V}\times$$ polynomial in $$V$$, so $$\lvert F\rvert\le Ce^{-c\tau^d}$$ and Lemma 4 applies with $$\beta=d$$. Here, in four dimensions, $$M_F(s)=\tfrac14C^{-s/4}\Gamma(s/4)P(s/4)$$ with $$C=\pi/24$$ and $$P(x)=\sum_kc_k(x)_k/k!$$ a polynomial (in two dimensions $$M_F(s)=\tfrac12C^{-s/2}\Gamma(s/2)P(s/2)$$ with $$C=\tfrac12$$), so $$M_F$$ has finitely many zeros and the expansion contains $$E^{2k}$$ and $$E^{2k}\ln E$$ terms from some order on. This is the sense in which these operators are exact to finite order only.

The double zeros at the even integers are a strong requirement on the decay.

<div class="theorem" markdown="1">
**Theorem 6** (tail exponent of a flat operator). Let $$d\in\{2,4\}$$ and let $$F\not\equiv0$$ satisfy $$\lvert F(\tau)\rvert\le Ce^{-c\tau^\beta}$$ with $$c>0$$. If $$\bar B$$ is flat, then $$\beta\le\tfrac12$$.
</div>

The idea is that a function of exponential type cannot have too many zeros in the half-plane, and the double zeros of $$M_F$$ at every second integer exceed what the growth of $$M_F$$ allows once $$\beta>\tfrac12$$. The tool is Carleman's formula for the half-plane, which relates the density of the zeros to the growth on the boundary and on the arcs; the heart of the argument is the comparison of the two logarithms.

<div class="proof" markdown="1">
*Proof.* Put $$f(z)=M_F(z+\delta_0)$$ with $$0<\delta_0<d$$ small enough that $$f$$ is holomorphic in a neighbourhood of $$\operatorname{Re}z\ge0$$. By Corollary 5, $$f$$ has zeros of multiplicity at least $$2$$ at $$z_k=d+2k-\delta_0$$ for $$k\ge2$$. Carleman's formula [[Levin]](#ref-levin) [[Peloso–Salvatori]](#ref-ps) states that, for zeros $$z_j=r_je^{i\theta_j}$$ with $$1\le r_j\le R$$,

$$
\sum_{r_j\le R}\Big(\frac1{r_j}-\frac{r_j}{R^2}\Big)\cos\theta_j=\frac1{\pi R}\int_{-\pi/2}^{\pi/2}\ln\lvert f(Re^{i\theta})\rvert\cos\theta\,d\theta+\frac1{2\pi}\int_1^R\Big(\frac1{y^2}-\frac1{R^2}\Big)\ln\lvert f(iy)f(-iy)\rvert\,dy+A(R), \tag{7}
$$

with $$A(R)$$ bounded. All terms on the left are nonnegative, so the zeros $$z_k$$ alone give a lower bound. They have multiplicity two and spacing two, hence the left side is at least $$\ln R-O(1)$$. On the right, $$\lvert f(iy)\rvert\le\int\lvert F\rvert\tau^{\delta_0-1}d\tau<\infty$$, so the second term is $$O(1)$$ from above. For the arc, $$\lvert f(\sigma+it)\rvert\le\int\lvert F\rvert\tau^{\sigma+\delta_0-1}d\tau\le C\,\Gamma\big((\sigma+\delta_0)/\beta\big)c^{-(\sigma+\delta_0)/\beta}$$, so on $$\lvert z\rvert=R$$, with $$\sigma=R\cos\theta$$, $$\ln\lvert f\rvert\le(R\cos\theta/\beta)\ln R+O(R)$$. The first term of the right side is then at most $$\frac{1}{\pi R}\int(R\cos\theta/\beta)\ln R\cos\theta\,d\theta+O(1)=\ln R/(2\beta)+O(1)$$. Comparing, $$\ln R-O(1)\le\ln R/(2\beta)+O(1)$$ for all large $$R$$, which requires $$1\le1/(2\beta)$$.
</div>

The same argument for a function with zeros of multiplicity $$\mu$$ on a progression of step $$g$$ gives $$\beta\le g/(2\mu)$$. With simple zeros at every integer (the horizon-count problem of [Exact causal-set horizon weights must be non-local](/quantum-gravity/nonlocal-weights/)) it gives $$\beta\le\tfrac12$$ again; there the Fourier argument of that page also excludes the boundary $$\beta=\tfrac12$$ with a fixed $$c>0$$. For the d'Alembertian the Carleman comparison is an equality at $$\beta=\tfrac12$$ and the argument above gives no contradiction. We do not know whether a flat operator with $$\beta=\tfrac12$$ exists.

<div class="theorem" markdown="1">
**Theorem 7** (flat operators exist for every smaller exponent). Let $$d\in\{2,4\}$$ and $$0<\beta<\tfrac12$$. Put $$h(t)=\tfrac12\big[e^{-t^\beta}-e^{-t^\beta\cos\pi\beta}\cos(t^\beta\sin\pi\beta)\big]$$, $$g_a(\tau)=\tau^{-a}\int_\tau^\infty t^{a-1}h(t)\,dt$$, $$(a_1,a_2)=(2,4)$$ for $$d=2$$ and $$(4,6)$$ for $$d=4$$, and $$F=\gamma\,(g_{a_2}-g_{a_1})/(a_2-a_1)$$ with a constant $$\gamma$$. For a suitable real $$\gamma$$ (fixed by the normalisation of the $$E^2$$ term) and a real $$a$$ (fixed by $$b(0)=0$$), $$F$$ is bounded and continuous with $$F(0)=0$$ and $$\lvert F(\tau)\rvert\le C(1+\tau)^{-\beta}e^{-\cos(\pi\beta)\tau^\beta}$$, and the operator (2) is flat: for every $$N$$ there is $$C_N$$ with $$\lvert b(E)+E^2\rvert\le C_N\lvert E\rvert^N$$ for all $$E$$ with $$\lvert E\rvert\le1$$ and $$\lvert\arg E\rvert\le\pi/2$$, which includes the timelike boundary values.
</div>

The construction uses the same Stieltjes-type building block as the horizon-weight kernels; the factor $$\sin^2(\pi s/2)$$ supplies the double zeros of $$M_F$$ at the even integers, and $$\beta<\tfrac12$$ is what lets the rotation of the contour in $$\int e^{-t^\beta e^{i\pi\beta}}$$ stay in the half-plane of convergence.

<div class="proof" markdown="1">
*Proof.* Rotating the contour as in the horizon-weight kernels gives $$\int_0^\infty t^{s-1}e^{-t^\beta e^{\pm i\pi\beta}}dt=\beta^{-1}\Gamma(s/\beta)e^{\mp i\pi s}$$ since $$\cos\pi\beta>0$$. Hence $$M_h(s)=\beta^{-1}\Gamma(s/\beta)\sin^2(\pi s/2)$$. By Fubini, $$M[g_a](s)=M_h(s)/(s-a)$$, and for even $$a$$ the right side is regular at $$s=a$$, so $$g_a$$ is bounded with $$g_a(\tau)=O(\tau^\beta)$$ at the origin. Therefore

$$
M_F(s)=\frac{\gamma\,\Gamma(s/\beta)\sin^2(\pi s/2)}{\beta\,(s-a_1)(s-a_2)}.
$$

It has simple zeros at $$a_1,a_2$$ and double zeros at every other even integer, including $$s=2$$ for $$d=4$$, as Corollary 5 requires. With $$\gamma$$ chosen so that the coefficient of $$E^2$$ is $$-1$$ and $$a$$ chosen to cancel the constant (both are nonzero real numbers determined by the residues in Lemma 4), every residue in the shift of (6) to the left of $$w=-2$$ vanishes, because a double pole of $$M_k$$ meets a double zero of $$M_F(d-w)$$. Hence $$b(E)+E^2$$ equals the integral in (6) on the line $$\operatorname{Re}w=-L$$ for any $$L>2$$. On that line $$\lvert M_k(w)M_F(d-w)E^{-w}\rvert\le C_L(1+\lvert y\rvert)^{m}\lvert E\rvert^{L}e^{(\pi/2-\pi/(2\beta)+\lvert\arg E\rvert)\lvert y\rvert}$$, from the growth $$e^{-\pi\lvert y\rvert/2}$$ of $$M_k$$, $$e^{-\pi\lvert y\rvert/(2\beta)}$$ of $$\Gamma((d-w)/\beta)$$, $$e^{\pi\lvert y\rvert}$$ of $$\sin^2$$, and $$e^{\lvert\arg E\rvert\lvert y\rvert}$$ of $$E^{-w}$$. For $$\lvert\arg E\rvert\le\pi/2$$ the exponent is at most $$\pi-\pi/(2\beta)<0$$ precisely when $$\beta<\tfrac12$$, so the integral converges and is $$O(\lvert E\rvert^L)$$; the identity extends from $$E>0$$ to the sector by analytic continuation in $$E$$. The decay of $$F$$ follows from the integral defining $$g_a$$.
</div>

*Numerical check* ($$d=4$$, $$\beta=0.3$$). The residues give $$\gamma=-2.5452\times10^{-18}$$ and $$a=-7.3438\times10^{-8}$$. We evaluated $$\delta(E)=b(E)+E^2$$ from the line integral at 40 digits: $$\log_{10}\lvert\delta\rvert=-2.00,\,-6.03,\,-21.59,\,-169.97$$ at $$E=10^{-1},10^{-3},10^{-5},10^{-7}$$. As an independent test of the whole chain we built $$F$$ in position space from $$h$$ by quadrature and evaluated (3) at $$E=1$$: it gives $$-7.343782852449\times10^{-8}$$, against $$-7.343782852449\times10^{-8}$$ from $$-E^2+\delta$$ (the two differ by $$3\times10^{-21}$$). The ratio $$b/(-E^2)$$ is $$0.073,\ 0.505,\ 0.963,\ 0.995,\ 1.000$$ at $$E=10^{-3},\,3\times10^{-4},\,10^{-4},\,6.5\times10^{-5},\,10^{-5}$$, so the operator reproduces the continuum symbol within four percent only below $$E\approx10^{-4}$$, that is at wavelengths longer than about $$10^{4}$$ discreteness lengths. Above that scale the symbol is the constant $$a$$, tiny in these units.

Theorems 6 and 7 together say that the threshold for the d'Alembertian coincides with the one for horizon weights. The condition $$\beta<\tfrac12$$ in Theorem 7 is also the condition for flatness on the timelike axis; for $$\beta>\tfrac12$$ the exponent in the proof is positive at $$\arg E=\pi/2$$.

## 4. The ultraviolet: superconvergence

Proposition 2 and Corollary 3 constrain operators with a special infrared structure. The next theorem needs no such structure.

<div class="theorem" markdown="1">
**Theorem 8** (no positive weight in four dimensions). Let $$d=4$$ and let $$\bar B$$ be of the form (2) with $$F$$ measurable, $$\lvert F(\tau)\rvert\le C(1+\tau)^p$$ for some $$p\ge0$$, and $$b$$ not constant. Suppose $$b(E)\ne0$$ for $$E>0$$. Then $$D=-1/b$$ does not have the form (4).
</div>

The argument has two halves. A Stieltjes function cannot approach its limit at infinity faster than $$1/Z$$, because $$T(Z)=Z[D-D(\infty)]$$ is nondecreasing. The symbol of any four-dimensional kernel with a polynomially bounded $$F$$ approaches its limit like $$Z^{-2}$$. The first half is the same property that gives a sum rule in the Källén–Lehmann representation, and the second is dominated convergence.

<div class="proof" markdown="1">
*Proof.* Write $$b=a+K$$. Substituting $$\tau=s/E$$ in (3),

$$
K(E)=\frac{4\pi}{E^4}\int_0^\infty F(s/E)\,s^2K_1(s)\,ds.
$$

For $$E\ge1$$ the integrand is bounded by $$C(1+s)^ps^2K_1(s)$$, which is integrable, so $$E^4K(E)$$ is bounded: $$K=O(Z^{-2})$$. (If $$F(0^+)$$ exists, $$E^4K\to8\pi F(0^+)$$, since $$\int_0^\infty s^2K_1(s)ds=2$$.)

Suppose $$D$$ had the form (4). If $$a>0$$, then $$D(\infty)=-1/a<0$$, contradicting $$D>0$$. If $$a=0$$, then $$\lvert D\rvert=1/\lvert K\rvert\ge cZ^2\to\infty$$, contradicting that a function of the form (4) is bounded on $$[1,\infty)$$. If $$a<0$$, then $$D(\infty)=-1/a$$ and

$$
D-D(\infty)=\frac{K}{ab},\qquad T(Z)=Z\,[D-D(\infty)]=O(Z^{-1})\to0,
$$

because $$b\to a\ne0$$. But $$T$$ is nondecreasing and nonnegative, so $$T\equiv0$$, $$\rho=0$$ and $$r_0=0$$, and $$D$$ is constant, so $$b$$ is constant, contrary to assumption.
</div>

The hypotheses hold for the whole generalised family.

<div class="proposition" markdown="1">
**Proposition 9** (finite-density operators). Every operator (2) with $$F=e^{-V}\sum_kc_kV^k/k!$$, for finitely many coefficients, or with $$\lvert c_k\rvert\le C(1+k)^p$$ for all $$k$$, satisfies the hypothesis of Theorem 8. So does the operator obtained by replacing $$V$$ by $$\varepsilon V$$ in the layer weights, for any $$\varepsilon>0$$.
</div>

<div class="proof" markdown="1">
*Proof.* $$\lvert F\rvert\le e^{-V}\sum C(1+k)^pV^k/k!\le C'(1+V)^p$$, which is polynomial in $$\tau$$. Replacing $$V$$ by $$\varepsilon V$$ keeps this bound.
</div>

So no operator in the generalised family, whatever its coefficients, has a propagator of the form (4). The proof works for $$d\ge3$$ with the same scaling: $$E^2K=E^{2-d}\int F(s/E)s^{d-1}k_d(s)\,ds=O(E^{2-d})\to0$$, since $$\int_0^\infty s^{d-1}k_d(s)ds=\Omega_{d-2}\Gamma(d)\int_0^\infty\sinh^{d-2}\eta\cosh^{-d}\eta\,d\eta<\infty$$. For $$d=2$$ it gives $$T(\infty)=2F(0^+)/a^2$$, which for the two-dimensional Benincasa–Dowker operator is $$2$$; since $$T\ge r_0=1$$ there is no contradiction, and the theorem is silent in two dimensions.

### The four-dimensional Benincasa–Dowker operator in numbers

The sum rule behind the theorem is visible in the spectral data of $$D$$. We take $$a=-4/\sqrt6=-1.63299$$ and $$F(\tau)=\tfrac4{\sqrt6}e^{-V}(1-9V+8V^2-\tfrac43V^3)$$ with $$V=\pi\tau^4/24$$, and evaluate (3) for complex $$E$$ by quadrature in double precision.

*Large-momentum behaviour.* At $$E=100$$, $$E^4K/(8\pi F(0))=0.999997$$ with $$8\pi F(0)=41.04$$. The function $$T(Z)=Z[D-D(\infty)]$$ equals $$0.1507,\,0.01539,\,0.001539$$ at $$Z=10^2,10^3,10^4$$, so it decreases like $$1/Z$$ and cannot be that of a function of the form (4).

*Zeros of $$b$$ off the cut.* Newton's method finds $$b(Z_*)=0$$ at $$Z_*=-25.384\pm3.795\,i$$. The residue of $$D=-1/b$$ at $$Z_*$$ is $$r=-1/b'(Z_*)=-0.8954\pm3.2805\,i$$, so the pair has $$2\operatorname{Re}r=-1.7907$$. An argument-principle count of the zeros of $$b$$ in $$10^{-3}<\lvert Z\rvert<400$$, away from the cut, gives winding number $$2.00$$ (largest phase step along the contour $$0.06$$ rad), so these two are the only zeros there. They are the complex pair of mass squared of order $$25$$ in units of the discreteness scale that signals the classical instability found in [[ASS]](#ref-ass).

*The cut.* On $$Z=-\mu+i0$$ the density $$\rho(\mu)=\pi^{-1}\operatorname{Im}(1/b)$$ tends to $$0.39$$ as $$\mu\to0$$, which accounts for the logarithm in section 3. It is positive up to $$\mu\approx9.4$$ and then changes sign, at $$\mu\approx9.4,\ 24.3,\ 43.0,\ 65.1,\ 90.2,\ 117.5,\ 146.4$$, with an amplitude that decreases quickly ($$\lvert\rho\rvert\approx5\times10^{-3}$$ at $$\mu=100$$, $$10^{-4}$$ at $$150$$, $$4\times10^{-7}$$ at $$300$$). Its integral is $$\int\rho\,d\mu=0.79072$$, made up of positive weight $$2.578$$ and negative weight $$-1.788$$ at $$\mu<400$$.

*Sum rule.* If $$D-D(\infty)=O(Z^{-2})$$ holds uniformly on large circles in the cut plane, Cauchy's theorem gives $$1+\int\rho\,d\mu+\sum_jr_j=0$$, the sum over the poles off the cut. Here $$1+0.790724+(-1.790724)=-7\times10^{-8}$$. The numerical closure, at the level of the quadrature error, is the check of the uniformity assumption, which we have not proved. Equivalently, $$D-D(\infty)$$ falls off faster than $$1/Z$$ only if the total weight, counted with the unit weight of the massless pole, is zero. For this operator the unit weight is cancelled by the complex pair together with the negative part of the cut density. Under the same assumption, a stable operator of this type, without zeros off the cut, would need $$\int\rho\,d\mu=-1$$, which no nonnegative density has.

### The light-cone escape

The proof of Theorem 8 used that $$K$$ comes from a function $$F$$ bounded near the light cone. A term supported on the light cone changes the decay of $$K$$.

<div class="proposition" markdown="1">
**Proposition 10** (weight on the light cone). Let $$d=4$$ and add to (2) a term $$\nu\,\theta(t)\delta(\tau^2)$$, $$\nu\in\mathbb R$$, with $$a\ne0$$ and $$F$$ as in Theorem 8. Its symbol is $$2\pi\nu/Z$$, so $$b=a+2\pi\nu/Z+O(Z^{-2})$$ and $$T(\infty)=2\pi\nu/a^2$$. If $$D$$ has the form (4) with $$r_0=1$$, then $$2\pi\nu\ge a^2$$. The bound is attained by $$B_M=-M^2\delta^4+M^4G^{(M)}_{\rm ret}$$, where $$G^{(M)}_{\rm ret}=\frac1{2\pi}\theta(t)\delta(\tau^2)-\frac M{4\pi\tau}J_1(M\tau)\theta(t)\theta(\tau^2)$$ is the retarded Green function of $$-\Box+M^2$$: its symbol is $$b=-M^2Z/(Z+M^2)$$, and $$D=1/Z+1/M^2$$ is a Stieltjes function.
</div>

<div class="proof" markdown="1">
*Proof.* The symbol of $$\theta(t)\delta(\tau^2)$$ is $$\int_0^\infty dt\,e^{-Et}\int4\pi r^2\delta(t^2-r^2)dr=\int_0^\infty te^{-Et}dt\cdot2\pi=2\pi/E^2$$. Then $$D-D(\infty)=(b-a)/(ab)=2\pi\nu/(a^2Z)(1+o(1))$$, so $$T(\infty)=2\pi\nu/a^2$$. Since $$T\ge r_0=1$$ for a function of the form (4), $$2\pi\nu\ge a^2$$. For $$B_M$$, $$G^{(M)}_{\rm ret}$$ has symbol $$1/(Z+M^2)$$, so $$b=-M^2+M^4/(Z+M^2)=-M^2Z/(Z+M^2)$$, and $$\nu=M^4/(2\pi)$$, $$a=-M^2$$ give $$2\pi\nu=a^2$$.
</div>

*Check.* The symbol of the interior part $$-\frac M{4\pi\tau}J_1(M\tau)$$, from the Laplace integral of Lemma 1 at $$M=1.3$$, $$E=0.9$$, is $$-0.83456790123$$, which equals $$1/(E^2+M^2)-1/E^2$$ to the twenty digits computed. The operator $$B_M$$ is retarded, Lorentz invariant and non-local, has a bounded kernel away from the light cone, and carries positive weight. It is not an operator that a sprinkling can produce: the mean operator of a sprinkling is (2), which has no light-cone term (Proposition 9). At any finite density the mean operator stays in the class (2); a light-cone term could arise only in a limit of infinite density.

## 5. Non-commutative points

Theorem 8 concerns a scalar field. A generalisation of a causal set replaces each point $$x$$ by a matrix algebra $$M_{n_x}$$ and the order by a *quantum relation* [[Weaver]](#ref-weaver) [[KLM]](#ref-klm) [[Weaver 2021]](#ref-weaver2), a family of subspaces $$V_{xy}\subseteq B(\mathbb C^{n_y},\mathbb C^{n_x})$$ that is reflexive, transitive and antisymmetric in a sense that reduces to the usual axioms when every $$n_x=1$$. The construction of a sprinkled quantum causal set, with an interval operator that replaces the integer $$\lvert I(x,y)\rvert$$ and layer weights that act through a function $$f$$ of it, is given in section 6 of [Mixed-Poisson and signed substrates and the four-dimensional d'Alembertian](/quantum-gravity/signed-substrates/), together with the conditions under which its mean operator is covered by the next theorem; we do not repeat it. What matters here is the form of the mean-field kernel when the atoms carry independent marks $$n_x\le n_{\max}$$ and the link data $$V_{xy}$$ depend only on the marks inside the interval $$[x,y]$$ and on their order: it is an $$N\times N$$ matrix of functions $$F_{ij}(\tau)=e^{-V}\sum_kg^{ij}_kV^k/k!$$, with $$g^{ij}_k$$ the mean, given $$k$$ interior atoms and endpoint marks $$i,j$$, of a finite-dimensional operator. If these operators are bounded polynomially in $$k$$, which we call property (Q), then $$\lvert F_{ij}\rvert\le C(1+\tau)^{p}$$ for the same reason as in Proposition 9. Property (Q) holds, for instance, when the interval operator is a sum over the at most $$k+2$$ atoms through which the link data are composed, each term of norm at most a constant depending on $$n_{\max}$$, and $$f$$ grows at most polynomially.

A matrix-valued $$D$$ is *operator-Stieltjes* if it has the form (4) with $$r_0,c$$ positive semidefinite matrices and $$\rho$$ a positive-semidefinite-valued measure.

<div class="theorem" markdown="1">
**Theorem 11** (matrix-valued kernels). Let $$d=4$$ and let $$\bar B$$ act on $$\mathbb R^N$$-valued fields by $$A\phi(x)+\int_{y\prec x}F(\tau)\phi(y)d^4y$$, with $$A$$ a real symmetric matrix and $$F$$ a symmetric matrix of measurable functions bounded by $$C(1+\tau)^p$$. Let $$b(E)=A+K(E)$$ be its symbol, assumed invertible for $$E>0$$ and not constant, and $$D=-b^{-1}$$. Then $$D$$ is not operator-Stieltjes.
</div>

<div class="proof" markdown="1">
*Proof.* Each entry of $$K$$ is $$O(Z^{-2})$$ by the proof of Theorem 8. Suppose $$D$$ is operator-Stieltjes. For every vector $$v$$, $$\langle v,Dv\rangle$$ is a scalar function of the form (4).

If $$A$$ is invertible, $$D(\infty)=-A^{-1}$$ and $$D-D(\infty)=A^{-1}-(A+K)^{-1}=(A+K)^{-1}KA^{-1}=O(K)$$. So $$T_v(Z)=Z\langle v,(D-D(\infty))v\rangle\to0$$ for each $$v$$; being nondecreasing and nonnegative, it vanishes, and $$D$$ is constant, contradicting that $$b$$ is not.

If $$A$$ is singular, pick a unit $$v$$ with $$Av=0$$. Then $$b(E)v=K(E)v\to0$$, so $$\lVert b^{-1}\rVert\ge1/\lVert Kv\rVert\to\infty$$ and $$\lVert D(Z)\rVert\to\infty$$. But an operator-Stieltjes function satisfies $$0\preceq D(Z)\preceq D(1)$$ for $$Z\ge1$$, which is bounded. Contradiction.
</div>

With property (Q), the mean-field kernel of non-commutative points of bounded size with interval-local link data is of the form assumed in Theorem 11, so such a structure has no operator-Stieltjes propagator in four dimensions. The statement concerns the matrix $$D$$; it does not decide whether the physical field is the whole multiplet or a compression of it. It assumes bounded atom size and link data that depend only on the interval. The same conclusion, with the infrared normalisation $$b=-GZ+o(Z)$$ assumed, is Theorem 17 of the page cited above; Theorem 11 here does not need that normalisation.

## 6. What is not excluded

Each item is outside the hypotheses of the theorems above; the condition under which it applies is stated, and none is preferred over the others.

- Two dimensions. Theorem 8 is silent for $$d=2$$ ($$T(\infty)=2$$). The two-dimensional Benincasa–Dowker operator has positive weight in the numerical test of section 2; a proof is not given here.
- A light-cone term with $$2\pi\nu\ge a^2$$ (Proposition 10). Such an operator is a continuum construction. It is not produced by the mean of a sprinkling, and the question of whether non-local, order-intrinsic data on a causal set can produce one is open (below).
- Kernels outside the hypotheses of Theorem 8: kernels growing faster than polynomially, kernels with singular behaviour at the light cone other than the $$\theta(t)\delta(\tau^2)$$ term, and layer weights that grow geometrically.
- Lorentz-violating operators, which do not have a kernel depending on $$\tau$$ alone.
- A reading of unitarity other than (4). The theorems say that $$D$$ is not a Stieltjes function. For a retarded non-local operator the propagator that defines the Hilbert space may be built differently, for example from the Pauli–Jordan function, as in the quantisation of [[BBL]](#ref-bbl), or by treating the complex pair of zeros with a fakeon or Wheeler-type prescription. Whether a negative $$\rho$$ of the kind found above then means negative norm or a different pathology depends on the quantisation, and the theorems do not address it.
- Flat operators with $$\beta=\tfrac12$$ exactly (Theorem 6), where the Carleman comparison is an equality.
- Operators that do not agree with $$-p^2$$ to all orders. The generalised family has the right low-momentum behaviour only to finite order, and passes the infrared test of Proposition 2 (section 3). Its failure is in the ultraviolet.

## 7. Open points

Whether a flat operator with tail exponent exactly $$\tfrac12$$ exists is not decided. For the operators of Theorem 7 we have not computed a lower bound on the negative spectral weight at fixed length $$\ell_{\rm eff}$$ where the symbol departs from $$-E^2$$. A proof that the two-dimensional Benincasa–Dowker propagator is a Stieltjes function would replace the numerical evidence of section 2. The sum rule of section 4 is derived under a uniform decay of $$D-D(\infty)$$ in the cut plane, which we checked numerically for the four-dimensional operator and did not prove. For non-commutative points with non-local order-intrinsic dressing, for example link data that depend on a Sorkin–Johnston-type Wightman function of the whole set, a light-cone term of the size required in Proposition 10 has not been excluded or exhibited.

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
12. <a id="ref-weaver"></a>Weaver, “Quantum relations,” Mem. Amer. Math. Soc. 215, no. 1010 (2012), [arXiv:1005.0354](https://arxiv.org/abs/1005.0354).
13. <a id="ref-klm"></a>Kornell, Lindenhovius, Mislove, “A category of quantum posets,” Indag. Math. 33, 1137 (2022), [arXiv:2101.11184](https://arxiv.org/abs/2101.11184).
14. <a id="ref-weaver2"></a>Weaver, “Hereditarily antisymmetric operator algebras,” J. Inst. Math. Jussieu 20, 1039 (2021), [arXiv:1902.09989](https://arxiv.org/abs/1902.09989).
