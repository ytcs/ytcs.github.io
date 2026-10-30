---
title: Horizon-count kernels that follow the area law at every gradient order
date: 2026-10-29
category: gravity
order: 4
description: "Under a hypothesis on the gradient expansion, a causal-set horizon count that follows the area law at every gradient order needs a weight kernel with infinitely many sign changes and a tail that decays more slowly than the exponential of any multiple of the square root of the linear size."
status: "Proved for kernels integrable near zero: Theorems 1 and 4, Lemma 5, Propositions 8 and 10. Reading them as statements about the horizon count assumes hypothesis (H) on the gradient expansion, established only for the lowest orders. Section 6 excludes tails exp(-c sqrt x/log x) and constructs tails exp(-c sqrt x/log^2 x) (sketch). Realisation by weights is open."
excludes: "Under (H), area-law horizon counts at all orders with finitely many or geometrically decaying weights, or with a kernel, integrable near zero, of bounded support, finitely many sign changes or stretched-exponential decay with exponent at least one half in linear size; finite-variation retarded kernels (not the Lorentz-invariant sprinkling kernel) with a symbol vanishing on an open set."
---

A causal set estimates the entropy of a horizon by counting horizon molecules, pairs of elements that straddle the horizon, each weighted by a number that depends on how many further elements lie to its future. Finitely many weights reach the area law only to a finite order in the gradients of the curvature. We ask what the kernel, the function of the volume of that future region through which the weights enter, must look like if the area law holds at every gradient order. The answer rests on hypothesis (H) of section 1, that every gradient order contributes a new term to the expected count; (H) is established only for the lowest orders. Under (H), no such kernel that is integrable near zero decays at least as fast as the exponential of a fixed multiple of the square root of the linear size of the future region, and every such kernel changes sign infinitely often (Theorem 1 and Corollary 6).

Write $$v$$ for the volume of the future region and $$x=v^{1/d}$$ for its linear size in $$d$$ dimensions. Functions with the required zeros and tails $$\exp(-c\,x^{\kappa})$$ exist for every $$\kappa<\tfrac12$$ (Theorem 4). The kernels constructed there are not of the form produced by weights, and no sequence of weights realising a heavy-tailed kernel is known. Finite weights that remove the terms up to a given order exist, are unique at lowest degree and alternate in sign (Proposition 8 and Corollary 9); for weights that depend only on the molecule type and uniform curvature at linear order, the exactness conditions are derived on [horizon counts and the area law](/quantum-gravity/area-projector/), with necessity shown for $$3\le d\le12$$.

Dou and Sorkin [[Dou–Sorkin]](#ref-dou-sorkin) and Barton, Counsell, Dowker, Gould, Jubb and Taylor [[Barton et al.]](#ref-barton) introduced the count of horizon molecules and found its leading area term. Jacobson's entanglement-equilibrium argument [[Jacobson]](#ref-jacobson) turns an entropy proportional to area into the linearised Einstein equation, which is the reason to ask for an area law at every size. Signed combinations of layer counts also define the causal-set d'Alembertian of Benincasa and Dowker [[BD]](#ref-bd) and its generalisations [[ASS]](#ref-ass); the same threshold for the tail exponent (called $$\beta$$ there), with the boundary exponent open there and excluded here, governs a d'Alembertian that agrees with the continuum one to all orders, in two and four dimensions, as shown in [the causal-set d'Alembertian has no positive spectral representation](/quantum-gravity/causal-box-spectrum/). The mathematics is classical: Stieltjes' functions with vanishing moments [[Stieltjes]](#ref-stieltjes) and Carlson's theorem [[Boas]](#ref-boas). Theorem 1, the decay threshold for kernels that follow the area law at every order, is new; the other statements adapt the classical tools to horizon-count kernels.

## 1. Kernels of weighted counts and exactness

Work in units of the discreteness length, so that the sprinkling has density one. The statements about kernels hold for any spacetime dimension $$d\ge2$$; the area-law setting is that of [horizon counts and the area law](/quantum-gravity/area-projector/), with $$d\ge3$$. A horizon molecule of type $$m$$ has $$m$$ further elements to the future of its lower element and to the past of the slice. Given weights $$w_0,w_1,\dots$$, the weighted count adds $$w_m$$ for each molecule. Because the further elements form a Poisson process, the expected count depends on the weights only through the kernel

$$
\phi(v)=e^{-v}\sum_{m\ge0}w_m\frac{v^m}{m!},\qquad \int_0^\infty\phi(v)\,v^{\alpha-1}dv=\Gamma(\alpha)\,Q(\alpha),\quad Q(\alpha)=\sum_m w_m\frac{(\alpha)_m}{m!}, \tag{1}
$$

where $$v$$ is the volume of the region to the future of the lower element and to the past of the slice, and $$(\alpha)_m=\alpha(\alpha+1)\cdots(\alpha+m-1)$$. The second equation in (1) holds when sum and integral can be interchanged, for instance when $$\sum_m\lvert w_m\rvert(\alpha)_m/m!<\infty$$, which is the case for finitely many weights and for $$\lvert w_m\rvert\le Cs^m$$ with $$s<1$$, the two cases used below. For a general kernel we write $$M[\phi](\alpha)=\int_0^\infty\phi(v)v^{\alpha-1}dv$$ for the Mellin transform and $$Q(\alpha)=M[\phi](\alpha)/\Gamma(\alpha)$$ for the Mellin symbol. Since $$\Gamma>0$$ on the positive axis, the zeros of $$M[\phi]$$ and of $$Q$$ there coincide.

The link from the kernel to the area law is derived on [horizon counts and the area law](/quantum-gravity/area-projector/), and we recall what is used. The expected count of a ball $$B$$ in a slice is an integral $$\int_P dV_p\,V_{\rm out}(p)\,\phi(V(p))$$ over the region $$P$$ inside the horizon, where $$V(p)$$ is the volume of the future of $$p$$ below the slice and $$V_{\rm out}(p)$$ the volume of its part outside $$P$$. Expanding it in the curvature and its derivatives at fixed ball volume, every term is a power of the discreteness length times an integral of a derivative of $$\phi$$, and integration by parts turns that integral into a constant times $$Q(\alpha')$$, where dimensional counting fixes the exponent $$\alpha'$$ from the power of the length. The flat area term, proportional to the area of $$\partial B$$, sits at $$\alpha'=\alpha_A=1+2/d$$, and every other term at $$\alpha'=\alpha_A+j/d$$ with an integer $$j\ge1$$. A kernel is therefore *exact to order $$J$$* if $$M[\phi]$$ converges absolutely at $$\alpha_A+j/d$$ for $$0\le j\le J$$, $$M[\phi](\alpha_A)\ne0$$ and $$M[\phi](\alpha_A+j/d)=0$$ for $$1\le j\le J$$. It is *exact to all orders* if this holds for every $$J$$. Which orders appear in the expansion depends on the geometry alone, and this is what the following hypothesis asserts.

(H) For every integer $$j\ge1$$ the expansion of the expected count in the gradients of the curvature, for the conformally static backgrounds treated on that page, contains a term multiplying $$Q(\alpha_A+j/d)$$ with a nonzero coefficient that does not depend on the weights.

Under (H) the expected count follows the area law at every size and at every gradient order exactly when $$\phi$$ is exact to all orders. The coefficients are nonzero, by computation on that page, for $$1\le j\le d$$ in the dimensions $$3\le d\le12$$, and for $$j=d+1$$ on a time-dependent background in $$d=3$$ and $$d=4$$. No argument for general $$j$$ is available. Every statement below about the count depends on (H); the theorems about kernels do not.

As an example take $$d=4$$. The weights of lowest degree $$N$$ that are exact to the first $$N$$ orders are the coefficients of $$Q_N(\alpha)=\binom{N+6-4\alpha}{N}$$, whose zeros are $$7/4,2,\dots,(N+6)/4$$ and with $$Q_N(3/2)=1$$. For $$N=4$$ they are $$(210,-791,1268,-928,256)$$ and for $$N=6$$ they are $$(924,-7084,23668,-41416,39808,-19968,4096)$$.

For a kernel with $$Q(3/2)=1$$ the ratio $$\int\lvert\phi\rvert v^{1/2}dv/\Gamma(3/2)$$ equals $$\int\lvert\phi\rvert v^{\alpha_A-1}dv/\lvert M[\phi](\alpha_A)\rvert$$. It is $$1$$ for a positive kernel, and otherwise it measures how far the area term is a small remainder of larger positive and negative contributions. For $$N=4,6,8,12$$ direct evaluation gives $$115$$, $$929$$, $$7.2\times10^{3}$$ and $$4.0\times10^{5}$$, a factor of about $$2.8$$ per extra zero. Since $$\binom{N+z}{N}\sim N^{z}/\Gamma(z+1)$$, the rescaled symbols $$N^{4\alpha-6}Q_N(\alpha)$$ tend to $$1/\Gamma(7-4\alpha)$$. This vanishes at every $$\alpha=7/4+j/4$$, and by Theorem 1 below it is not the symbol of a kernel, integrable near zero, with decay $$\exp(-c\,v^{1/8})$$.

## 2. The decay threshold

Put $$\chi(x)=\phi(x^d)$$, the kernel as a function of the linear size. Substituting $$v=x^d$$ gives $$M[\phi](\alpha)=d\int_0^\infty\chi(x)x^{d\alpha-1}dx$$, and at $$\alpha=\alpha_A+j/d$$ the exponent is $$d\alpha-1=d+1+j$$. So the conditions $$M[\phi](\alpha_A+j/d)=0$$ say that the integer moments of $$\chi$$ of order $$d+2,d+3,\dots$$ vanish. How fast can a nonzero function on the half-line decay when all its moments from some order on are zero? This is the indeterminacy question of Stieltjes for the half-line, and $$e^{-c\sqrt x}$$ is the borderline between decay that forces the function to vanish and decay that does not, on the scale of powers of $$x$$.

<div class="theorem" markdown="1">
**Theorem 1** (decay threshold). Let $$\phi$$ be measurable on $$(0,\infty)$$ and $$j_0\ge1$$ an integer, with $$\int_0^1\lvert\phi(v)\rvert v^{\alpha_A+j_0/d-1}dv<\infty$$ and $$\lvert\phi(v)\rvert\le C\exp(-c\,v^{1/(2d)})$$ for $$v\ge1$$ and some $$C,c>0$$. If $$M[\phi](\alpha_A+j/d)=0$$ for all integers $$j\ge j_0$$, then $$\phi=0$$ almost everywhere.
</div>

Vanishing moments are the vanishing of all Taylor coefficients at the origin of a Fourier transform, and a decay of the form $$e^{-c\sqrt x}$$ is just enough to make that transform analytic.

<div class="proof" markdown="1">
*Proof.* With $$\chi$$ as above, $$\lvert\chi(x)\rvert\le Ce^{-c\sqrt x}$$ for $$x\ge1$$. Let $$i_0=d+1+j_0$$ and $$\chi_0(x)=x^{i_0}\chi(x)$$. The integrability hypothesis says $$\int_0^1\lvert\chi_0\rvert dx<\infty$$, so $$\int_0^\infty\lvert\chi_0(x)\rvert x^{m}dx<\infty$$ for every integer $$m\ge0$$, and by the vanishing hypothesis $$\int_0^\infty\chi_0(x)x^{m}dx=0$$. Define the even function $$f(y)=\lvert y\rvert\chi_0(y^2)$$ on $$\mathbb R$$. The substitution $$x=y^2$$ gives $$\int_{\mathbb R}f(y)y^{2m}dy=\int_0^\infty\chi_0(x)x^{m}dx=0$$, and the odd moments of $$f$$ vanish by symmetry. For $$\lvert y\rvert\ge1$$ we have $$\lvert f(y)\rvert\le C\lvert y\rvert^{1+2i_0}e^{-c\lvert y\rvert}\le C'e^{-c\lvert y\rvert/2}$$, and $$f\in L^1(\mathbb R)$$. Hence $$\hat f(\eta)=\int f(y)e^{-i\eta y}dy$$ is holomorphic in the strip $$\lvert\operatorname{Im}\eta\rvert<c/2$$, and its derivatives at $$\eta=0$$ are, up to factors $$(-i)^k$$, the moments of $$f$$, all zero. A holomorphic function on a connected strip with all derivatives zero at one point vanishes identically. So $$\hat f=0$$, hence $$f=0$$ almost everywhere, hence $$\chi_0=0$$ and $$\phi=0$$ almost everywhere.
</div>

Because $$c>0$$ is arbitrary, the exponent $$\tfrac12$$ in the length $$x$$ is excluded together with every faster decay. Corollary 2 lists the weighted cases.

<div class="corollary" markdown="1">
**Corollary 2.** Under (H), a kernel $$\phi$$ that is integrable against $$v^{\alpha_A+j_0/d-1}$$ near $$v=0$$ for some $$j_0\ge1$$ is not exact to all orders in each of the following cases: (a) $$\phi$$ comes from weights with only finitely many $$w_m$$ nonzero; (b) $$\phi$$ comes from weights with $$\lvert w_m\rvert\le Cs^m$$ for some $$s<1$$; (c) $$\lvert\phi(v)\rvert\le Ce^{-cv}$$ for $$v\ge1$$ and some $$c>0$$; (d) $$\phi$$ is bounded and vanishes outside a bounded range of $$v$$.
</div>

<div class="proof" markdown="1">
*Proof.* In case (b), $$\lvert\phi(v)\rvert\le Ce^{-v}\sum(sv)^m/m!=Ce^{-(1-s)v}$$, which is case (c). In case (a), $$\phi=e^{-v}\times$$ polynomial, so $$\lvert\phi\rvert\le C'e^{-v/2}$$. Cases (c) and (d) satisfy the decay hypothesis of Theorem 1, because $$e^{-cv}\le e^{-c\,v^{1/(2d)}}$$ for $$v\ge1$$, and for (d) any $$c$$ will do. For weights, $$\phi=e^{-v}E(v)$$ with $$E$$ entire, so $$\phi$$ is bounded near $$0$$ and the integrability hypothesis holds.
</div>

Exponentially decaying kernels admit a second proof that uses only Carlson's theorem and shows where the exponent $$1$$ enters; it is not needed for what follows.

<div class="remark" markdown="1">
**Remark 3** (Carlson's theorem). Let $$\lvert\phi(v)\rvert\le Ce^{-cv}$$ and write $$Q=M[\phi]/\Gamma$$. For $$\sigma\ge\sigma_0>0$$ the product formula $$\Gamma(\sigma)^2/\lvert\Gamma(\sigma+iy)\rvert^2=\prod_{n\ge0}\big(1+y^2/(\sigma+n)^2\big)$$ shows that $$\Gamma(\sigma)/\lvert\Gamma(\sigma+iy)\rvert\le\Gamma(\sigma_0)/\lvert\Gamma(\sigma_0+iy)\rvert\lesssim(1+\lvert y\rvert)^{1/2-\sigma_0}e^{\pi\lvert y\rvert/2}$$, uniformly in $$\sigma$$. With $$\lvert M[\phi](\sigma+iy)\rvert\le C\Gamma(\sigma)c^{-\sigma}$$ this gives $$\lvert Q(\sigma+iy)\rvert\le C'c^{-\sigma}(1+\lvert y\rvert)^{1/2-\sigma_0}e^{\pi\lvert y\rvert/2}$$ on $$\operatorname{Re}\alpha\ge\sigma_0$$. Then $$f(z)=Q(\alpha_0+z/d)$$, with $$\alpha_0=\alpha_A+j_0/d$$, is of exponential type in $$\operatorname{Re}z\ge0$$ with growth $$e^{\pi\lvert y\rvert/(2d)}$$ on the imaginary axis, and $$\pi/(2d)<\pi$$. It vanishes at $$z=0,1,2,\dots$$, so $$f\equiv0$$ by Carlson's theorem [[Boas]](#ref-boas), hence $$M[\phi]\equiv0$$ and $$\phi=0$$. For $$\lvert\phi\rvert\le Ce^{-cv^{\theta}}$$ with $$\theta<1$$ the bound on $$M[\phi]$$ is of order $$\Gamma(\sigma/\theta)$$, which makes $$Q$$ of infinite exponential type, and the argument stops there.
</div>

The zeros can also be demanded at only some of the exponents. If $$M[\phi](\alpha_A+j/d)=0$$ only for $$j\in j_0+\Delta\mathbb N$$ with a fixed integer step $$\Delta\ge1$$, the same proof in the variable $$u=x^\Delta$$ shows $$\phi=0$$ whenever $$\lvert\phi\rvert\le C\exp(-c\,v^{\Delta/(2d)})$$ at large $$v$$. For $$\Delta=2$$ the threshold in $$v$$ is $$v^{1/d}$$; exponential decay in $$v$$ is excluded whenever $$\Delta\le2d$$.

## 3. Kernels below the threshold

<div class="theorem" markdown="1">
**Theorem 4** (existence below the threshold). Let $$0<\kappa<\tfrac12$$ and $$d\ge2$$. There is a continuous function $$\phi$$ on $$(0,\infty)$$ with $$\lvert\phi(v)\rvert\le C\exp(-c\,v^{\kappa/d})$$ for any $$c<\cos\pi\kappa$$, such that $$M[\phi](\alpha_A+j/d)=0$$ for all $$j\ge1$$ and $$M[\phi](\alpha_A)=(-1)^d\pi\,\Gamma\big((d+2)/\kappa\big)/\kappa\neq0$$. Near $$v=0$$, $$\phi(v)=-\sin(\pi\kappa)\,v^{\kappa/d}/\big(d\,(d+2+\kappa)\big)+O(v^{2\kappa/d})$$.
</div>

The construction follows Stieltjes' example of a function with vanishing moments, which is the case $$\kappa=\tfrac14$$ up to the rescaling $$x\to x/4$$ [[Stieltjes]](#ref-stieltjes). The zeros of the Mellin transform come from a factor $$\sin\pi z$$, and the condition $$\kappa<\tfrac12$$ is what allows a contour rotation by the angle $$\pi\kappa<\pi/2$$.

<div class="proof" markdown="1">
*Proof.* Let $$f_1(x)=e^{-x^{\kappa}\cos\pi\kappa}\sin(x^{\kappa}\sin\pi\kappa)$$. Rotating the contour in $$\int_0^\infty t^{a-1}e^{-te^{-i\vartheta}}dt=\Gamma(a)e^{ia\vartheta}$$ for $$\lvert\vartheta\rvert<\pi/2$$ and taking imaginary parts gives $$\int_0^\infty t^{a-1}e^{-t\cos\vartheta}\sin(t\sin\vartheta)\,dt=\Gamma(a)\sin(a\vartheta)$$. With $$t=x^\kappa$$, $$a=z/\kappa$$ and $$\vartheta=\pi\kappa$$ this yields

$$
M[f_1](z)=\int_0^\infty f_1(x)x^{z-1}dx=\frac{\Gamma(z/\kappa)\sin\pi z}{\kappa}, \tag{2}
$$

which vanishes at every positive integer $$z$$. Define $$f_2(x)=x^{-(d+2)}\int_x^\infty t^{d+1}f_1(t)\,dt$$. By Fubini, for $$\operatorname{Re}z>d+2$$,

$$
M[f_2](z)=\frac{M[f_1](z)}{z-d-2}=\frac{\Gamma(z/\kappa)\sin\pi z}{\kappa\,(z-d-2)}. \tag{3}
$$

Since $$M[f_1](d+2)=0$$, we have $$\int_x^\infty t^{d+1}f_1=-\int_0^x t^{d+1}f_1$$. Near the origin $$f_1(t)=\sin(\pi\kappa)\,t^\kappa+O(t^{2\kappa})$$, so $$f_2(x)=-\sin(\pi\kappa)\,x^{\kappa}/(d+2+\kappa)+O(x^{2\kappa})$$ as $$x\to0$$, and $$f_2$$ is continuous. Its Mellin transform therefore converges for $$\operatorname{Re}z>-\kappa$$ and equals (3) there by analytic continuation. At $$z=d+2+j$$ with $$j\ge1$$ the right side of (3) vanishes; at $$z=d+2$$ it equals $$(-1)^d\pi\Gamma((d+2)/\kappa)/\kappa\ne0$$. For large $$x$$, with $$A=(d+2)/\kappa$$ and $$c_\kappa=\cos\pi\kappa$$, $$\lvert f_2(x)\rvert\le x^{-(d+2)}\kappa^{-1}\int_{x^\kappa}^\infty s^{A-1}e^{-c_\kappa s}ds\lesssim x^{-\kappa}e^{-c_\kappa x^\kappa}$$. Finally $$\phi(v)=f_2(v^{1/d})/d$$ has $$M[\phi](\alpha)=M[f_2](d\alpha)$$, and $$d\alpha=d+2+j$$ at $$\alpha=\alpha_A+j/d$$.
</div>

The bound of Theorem 4 contains the factor $$\exp(-x^{\kappa}\cos\pi\kappa)$$, which becomes small only at very large $$x$$: it falls to $$e^{-30}$$ at $$x=9.0\times10^{4}$$ for $$\kappa\approx0.41$$, the smallest value over $$\kappa$$, and at larger $$x$$ for other $$\kappa$$ (for $$\kappa=0.25$$ at $$3.2\times10^{6}$$). The exponential tail thus sets in only at about $$10^{5}$$ discreteness lengths. Here $$x=v^{1/d}$$ is the linear size of the future region of a molecule in units of the discreteness length. In flat $$d=4$$ the future region of a molecule at depth $$\tau$$ below the slice has volume $$\pi\tau^4/3$$, so $$x$$ corresponds to a depth $$\tau=(3/\pi)^{1/4}x\approx0.99\,x$$.

Weights give no such kernel directly. A weighting by molecule type has $$\phi=e^{-v}E(v)$$ with $$E$$ entire, so $$\phi$$ is analytic at $$v=0$$, whereas the kernel of Theorem 4 behaves like a nonzero multiple of the non-integer power $$v^{\kappa/d}$$ there. Whether any sequence $$w_m$$ gives a kernel with a tail of this type is open. The conditions of sections 2 and 4 are necessary conditions on any exact kernel, and Theorem 4 shows that the decay condition alone does not rule out exactness.

## 4. Oscillation: finitely many sign changes bound the exactness

A positive kernel has $$M[\phi]>0$$ everywhere and cancels nothing, so a heavy tail is not the only requirement: the kernel must also change sign, and the number of sign changes limits how many zeros of $$M[\phi]$$ it can have.

<div class="lemma" markdown="1">
**Lemma 5** (sign changes). Let $$\phi$$ be measurable on $$(0,\infty)$$ and suppose $$M[\phi](\alpha_k)=0$$ with absolute convergence for $$Z$$ distinct real numbers $$\alpha_1,\dots,\alpha_Z$$. Then either $$\phi=0$$ almost everywhere or $$\phi$$ has at least $$Z$$ sign changes. Here $$\phi$$ has $$\nu$$ sign changes if there are points $$0<v_1<\dots<v_\nu$$ such that, after fixing one overall sign, $$(-1)^i\phi\ge0$$ almost everywhere on $$(v_i,v_{i+1})$$ for $$i=0,\dots,\nu$$ (with $$v_0=0$$, $$v_{\nu+1}=\infty$$), and $$\nu$$ is the smallest number for which such points exist.
</div>

The proof compares $$\phi$$ with a generalised polynomial that has the same sign pattern.

<div class="proof" markdown="1">
*Proof.* We first show that a nonzero sum $$g(v)=\sum_{k=1}^{n}c_kv^{b_k}$$ with distinct real $$b_k$$ has at most $$n-1$$ zeros in $$(0,\infty)$$, counted with multiplicity. For $$n=1$$ there are none. For $$n>1$$, divide by $$v^{b_1}$$, which does not change the zeros, and differentiate. This removes the first term, leaves a nonzero sum of $$n-1$$ terms with distinct exponents, and by Rolle's theorem loses at most one zero; induction finishes.

Suppose $$\phi\ne0$$ has $$\nu<Z$$ sign changes at $$v_1<\dots<v_\nu$$. Take the first $$\nu+1$$ exponents and the sum $$g(v)=\sum_{k=1}^{\nu+1}c_kv^{\alpha_k-1}$$. The $$\nu$$ linear conditions $$g(v_i)=0$$ on $$\nu+1$$ coefficients have a nonzero solution. By the first paragraph $$g$$ has no zeros other than the $$v_i$$, each simple, so $$g$$ changes sign exactly at the $$v_i$$, and after fixing the overall sign of $$c$$ we have $$g\phi\ge0$$ almost everywhere. But $$\int g\phi\,dv=\sum_kc_kM[\phi](\alpha_k)=0$$, so $$g\phi=0$$ almost everywhere, and since $$g$$ vanishes only at finitely many points, $$\phi=0$$ almost everywhere. This contradicts $$\phi\ne0$$.
</div>

<div class="corollary" markdown="1">
**Corollary 6.** A nonzero kernel that is exact to order $$J$$ has at least $$J$$ sign changes, and a kernel that is exact to all orders has infinitely many.
</div>

The corollary is Lemma 5 with $$Z=J$$ and with arbitrarily large $$Z$$. Oscillation is thus a requirement separate from decay. A positive heavy-tailed kernel has the tail without the sign changes, and $$e^{-v}\sin v$$ has the sign changes with exponential decay; neither meets Theorem 1 and Corollary 6 together.

A class of functions with finitely many sign changes is the class of *tame* functions, those definable in an o-minimal expansion of the real field, a structure in which every definable subset of the line is a finite union of points and intervals. Polynomials, exponentials, logarithms and restricted analytic functions are definable in one such structure, the real field with restricted analytic functions and exponentiation [[van den Dries–Macintyre–Marker]](#ref-dmm); the o-minimality of the real exponential field alone is Wilkie's theorem [[Wilkie]](#ref-wilkie). By the monotonicity theorem [[van den Dries]](#ref-vdd), the half-line splits into finitely many points and intervals on each of which a definable function is continuous and either constant or strictly monotone. Such a function has finitely many sign changes.

<div class="theorem" markdown="1">
**Theorem 7** (tame kernels are finitely exact). Let $$\phi$$ be definable in an o-minimal structure and not zero. Then $$M[\phi]$$ has only finitely many zeros among the points $$\alpha>0$$ at which the Mellin integral converges absolutely. In particular no tame kernel is exact to all orders.
</div>

<div class="proof" markdown="1">
*Proof.* By the monotonicity theorem $$\phi$$ has finitely many sign changes, say $$\nu$$. If $$M[\phi]$$ had $$\nu+1$$ distinct zeros of this kind, Lemma 5 would give $$\phi=0$$. A kernel that is exact to all orders has, by definition, absolutely convergent Mellin integrals and zeros at the infinitely many points $$\alpha_A+j/d$$, $$j\ge1$$.
</div>

Finite combinations of exponentials with polynomial coefficients are tame, and for them the bound can be made exact. For $$q$$ distinct rates $$\lambda_i>0$$ and degrees $$D_i$$ let $$\mathcal E(D_1,\dots,D_q)$$ be the span of $$v^ke^{-\lambda_iv}$$ with $$0\le k\le D_i$$. Weighted counts with $$N+1$$ weights are the case $$q=1$$, $$\lambda_1=1$$, $$D_1=N$$.

<div class="proposition" markdown="1">
**Proposition 8** (exactness index of exponential polynomials). Put $$K=\sum_i(D_i+1)-1$$. (i) A nonzero $$\phi\in\mathcal E(D_1,\dots,D_q)$$ has at most $$K$$ zeros of $$M[\phi]$$ on $$(0,\infty)$$. (ii) For any $$K$$ distinct points $$\alpha_1,\dots,\alpha_K>0$$ and any $$\alpha_A>0$$ different from them, there is, up to a factor, exactly one $$\phi\in\mathcal E$$ with $$M[\phi](\alpha_k)=0$$ for all $$k$$, and it has $$M[\phi](\alpha_A)\ne0$$.
</div>

<div class="proof" markdown="1">
*Proof.* (i) We bound the number of positive zeros of $$\phi=\sum_ip_i(v)e^{-\lambda_iv}$$ by induction on $$q$$. For $$q=1$$, $$\phi=p_1e^{-\lambda_1v}$$ has at most $$D_1$$ zeros. For $$q>1$$ multiply by $$e^{\lambda_1v}$$, which keeps the zeros, and differentiate $$D_1+1$$ times. This removes $$p_1$$ and turns each other term into a polynomial of the same degree times $$e^{-(\lambda_i-\lambda_1)v}$$, because $$(pe^{-\delta v})'=(p'-\delta p)e^{-\delta v}$$ and $$\delta=\lambda_i-\lambda_1\ne0$$. By Rolle's theorem each differentiation loses at most one zero, so the number of zeros of $$\phi$$ is at most the bound for $$q-1$$ rates plus $$D_1+1$$, which is $$K$$. The function $$\phi$$ is continuous, so its sign changes occur at zeros and there are at most $$K$$ of them. By Lemma 5, $$M[\phi]$$ then has at most $$K$$ zeros.

(ii) The $$K$$ conditions $$M[\phi](\alpha_k)=0$$ are linear in the $$K+1$$ coefficients, so a nonzero solution exists. If $$M[\phi](\alpha_A)$$ also vanished, $$M[\phi]$$ would have $$K+1$$ zeros, contradicting (i). The same argument shows the solution is unique up to a factor: two independent solutions would have a combination vanishing at $$\alpha_A$$ as well.
</div>

For example, two rates with $$(D_1,D_2)=(1,1)$$ have $$K=3$$: three prescribed zeros of $$M[\phi]$$ fix $$\phi$$ up to a factor and leave $$M[\phi](\alpha_A)\ne0$$. For weighted counts the proof gives more.

<div class="corollary" markdown="1">
**Corollary 9** (alternation). Let $$\phi=e^{-v}p(v)$$ with $$p(v)=\sum_{m=0}^Nw_mv^m/m!$$ not zero, and suppose $$M[\phi]$$ vanishes at $$N$$ distinct positive points. Then all $$w_m$$ are nonzero and alternate in sign, and $$p$$ has exactly $$N$$ positive zeros, all simple.
</div>

<div class="proof" markdown="1">
*Proof.* By Lemma 5, $$\phi$$ has at least $$N$$ sign changes, so $$p$$ has at least $$N$$ positive zeros of odd multiplicity. Descartes' rule of signs bounds the number of positive zeros, with multiplicity, by the number of sign changes in the coefficient sequence, which is at most $$N$$. Both inequalities are equalities.
</div>

The polynomial $$p$$ for the weights $$(210,-791,1268,-928,256)$$ has simple positive zeros at $$0.361$$, $$1.583$$, $$4.052$$ and $$8.504$$, and for the $$N=6$$ weights of section 1 at $$0.178,\,0.846,\,2.217,\,4.497,\,8.002,\,13.51$$.

The layer coefficients of the Benincasa–Dowker d'Alembertian, proportional to $$(1,-2,1)$$ in two dimensions and equal to $$(1,-9,16,-8)$$ in four, are quoted in the convention of [[BD]](#ref-bd) and of the Remark in section 5 of [horizon counts and the area law](/quantum-gravity/area-projector/): signature $$-+\cdots+$$, so that the operator tends to $$-\partial_t^2+\nabla^2$$ on smooth functions. The local term and the overall normalisation are set aside. Their Mellin symbols have zeros at $$\{1,2\}$$ and $$\{\tfrac12,1,\tfrac32\}$$, and the weights alternate in sign. The zero sets contain $$\alpha_A$$ ($$2$$ in $$d=2$$, $$\tfrac32$$ in $$d=4$$), so these are weights of the d'Alembertian and not of an area-law count. Each of the four weightings has degree $$N$$ and exactly $$N$$ zeros of its symbol, so each is extremal in the sense of Proposition 8.

In $$d=4$$ the minimal weights give $$Q_N\big(\tfrac32+\tfrac{N+k}4\big)=\binom{-k}{N}=(-1)^N\binom{N+k-1}{N}$$. The case $$k=0$$ is the last removed exponent, and at $$k=1$$ the first gradient order beyond those removed enters with the factor $$(-1)^N$$ relative to the area term. For $$k=2,3$$ the absolute values are $$5,15$$ for $$N=4$$ and $$7,28$$ for $$N=6$$. The factor multiplies a coefficient that depends on the background, so it gives the weight of the uncancelled term relative to the area term and leaves its size in a given geometry open. Adding weights thus removes the lower orders but leaves the first uncancelled term at relative weight $$\pm1$$.

## 5. No band limit for retarded kernels

On a periodic lattice an identity can be made exact by letting the vertices vanish outside a band of momenta; this device is used in section 6 of [a conserved lattice momentum](/quantum-gravity/lattice-diffeomorphisms/). For a kernel the corresponding device is a transform that vanishes outside a bounded set, hence on an open set of frequencies or momenta. We consider two kinds of kernel, and both statements are about Fourier transforms of the kernels, not about the Mellin symbol $$Q$$ of section 1. The horizon-count kernel $$\phi(v)\,dv$$ is a measure on the half-line $$v\ge0$$, and its frequency transform is its Fourier transform in $$v$$ (part (a)). The retarded kernel of a convolution operator on $$\mathbb R^d$$ is a measure $$\mu$$ supported in the closed future light cone, and its symbol is $$\hat\mu(k)$$ (part (b)); part (a) is also the step on which (b) rests. We take $$\mu$$ of finite total variation, for instance a kernel with locally integrable density that is cut off outside a bounded part of the future cone, such as a truncation, in a preferred frame, of the mean kernel of a causal-set d'Alembertian. The Lorentz-invariant mean kernel $$F(\tau)$$ of a sprinkling, a function of the proper time $$\tau$$, is not of this type: the surfaces of constant $$\tau$$ in the future cone have infinite volume, so $$F$$ is not a finite measure and (b) does not apply to it. For measures of finite total variation neither transform can vanish on an open set.

<div class="proposition" markdown="1">
**Proposition 10** (no band limit). (a) Let $$h$$ be a finite complex measure on $$[0,\infty)$$, for example $$h(s)\,ds$$ with $$h\in L^1(0,\infty)$$, and $$\hat h(\omega)=\int e^{-i\omega s}h(ds)$$ its frequency transform. If $$\hat h$$ vanishes on an interval of the real line, then $$h=0$$. This applies to $$\phi(v)\,dv$$ in the variable $$v$$ and to $$\chi(x)\,dx$$ in the length $$x$$ when these are integrable. (b) Let $$\mu$$ be a finite complex measure on $$\mathbb R^d$$, $$d\ge2$$, supported in the closed future light cone, with $$\hat\mu(k)=\int e^{-ik\cdot\xi}\mu(d\xi)$$. If $$\hat\mu$$ vanishes on a nonempty open set of real momenta, then $$\mu=0$$. (c) A bounded kernel $$\phi$$ that vanishes outside a bounded range of $$v$$ and satisfies the integrability hypothesis of Theorem 1 cannot be exact to all orders; and $$M[\phi]$$ cannot vanish on an interval of $$\alpha$$ inside its strip of absolute convergence unless $$\phi=0$$.
</div>

<div class="proof" markdown="1">
*Proof.* (a) The function $$\hat h$$ is bounded and holomorphic for $$\operatorname{Im}\omega<0$$, since $$\lvert e^{-i\omega s}\rvert=e^{s\operatorname{Im}\omega}\le1$$ for $$s\ge0$$, and continuous on the closed half-plane. If it vanishes on a real interval, the function equal to $$\hat h$$ below and $$0$$ above is continuous across the interval and holomorphic on each side, hence holomorphic by Morera's theorem, and it vanishes on an open set. By the identity theorem $$\hat h\equiv0$$, and $$h=0$$ by uniqueness of the Fourier transform of finite measures.

(b) Let $$u$$ be a future-directed timelike vector, so $$u\cdot\xi\le0$$ for $$\xi$$ in the cone (signature $$-+\dots+$$). For complex $$\zeta$$, $$\lvert e^{-i\zeta u\cdot\xi}\rvert=e^{\operatorname{Im}\zeta\,u\cdot\xi}\le1$$ when $$\operatorname{Im}\zeta\ge0$$. For fixed $$k$$, the function $$\zeta\mapsto\hat\mu(k+\zeta u)$$ is the transform of the push-forward of $$e^{-ik\cdot\xi}\mu(d\xi)$$ under $$\xi\mapsto-u\cdot\xi\ge0$$, a finite measure on $$[0,\infty)$$, evaluated at $$\omega=-\zeta$$. If $$k$$ lies in the open set $$U$$ where $$\hat\mu=0$$, the function vanishes for small real $$\zeta$$, hence by (a) for all real $$\zeta$$. So $$\hat\mu=0$$ on $$U+\mathbb Ru$$ for every timelike $$u$$, that is on $$U_1=U+T$$, with $$T$$ the open set of all timelike vectors. The same argument applied to the open set $$U_1$$ gives vanishing on $$U_1+T=U+T+T$$. Every vector $$a$$ is a sum of two timelike vectors: with $$e_0$$ the unit time direction and $$b>\lvert a\rvert$$, $$a=(a/2+be_0)+(a/2-be_0)$$. So $$T+T=\mathbb R^d$$, $$\hat\mu\equiv0$$, and $$\mu=0$$ by uniqueness of the Fourier transform of finite measures.

(c) The first statement is the case of bounded support in Theorem 1 (Corollary 2(d)). The second is the identity theorem for the holomorphic function $$M[\phi]$$ in its strip of absolute convergence, together with uniqueness of the Mellin transform.
</div>

Part (b) says that a retarded kernel of finite total variation is determined by its symbol on any open set of momenta, so it cannot agree with a prescribed symbol on a band and vanish outside it. Extending (b) to the Lorentz-invariant kernels, as tempered distributions supported in the cone, through boundary values of holomorphic functions in the forward tube and an edge-of-the-wedge argument, is an open problem. Bandlimited quantum field theory [[Pye–Donnelly–Kempf]](#ref-pdk) restricts the momentum content of the field, whereas (b) concerns the symbol of a retarded kernel. For kernels of finite total variation, then, exactness cannot be obtained by confining the nonvanishing part of the symbol to a bounded set of momenta; the zeros at $$\alpha_A+j/d$$ are governed by Theorem 1.

## 6. Realisation by weights and the finer tail scale

A kernel produced by weights that is exact to all orders would have a tail heavier than $$\exp(-c\sqrt x/\log x)$$ for every $$c>0$$ (Theorem 1 and the next paragraph) and, by Corollary 6, infinitely many sign changes. Whether such weights $$w_m$$ exist is open, and the kernels of Theorem 4 are not of that form (section 3).

Theorem 1 and Theorem 4 settle the tail only on the scale of powers of $$x$$. The proof of Theorem 1 uses that $$\hat f$$ is analytic; the Denjoy–Carleman theorem [[Rudin]](#ref-rudin) replaces analyticity by quasi-analyticity. Let $$\lvert f(y)\rvert\le Ce^{-\omega(\lvert y\rvert)}$$ and $$M_n=\int\lvert y\rvert^ne^{-\omega(\lvert y\rvert)}dy$$. If $$\sum_nM_{n-1}/M_n=\infty$$, then $$\hat f$$ lies in a quasi-analytic class, and vanishing of all moments of $$f$$ forces $$f=0$$. For $$\omega(y)=c\,y/\log(e+y)$$ the integrand of $$M_n$$ peaks at $$\lvert y\rvert\approx n\log n/c$$, so $$M_n^{1/n}\asymp n\log n$$ and $$M_{n-1}/M_n\approx c/(n\log n)$$ (a Laplace estimate); the series diverges, and a tail $$\exp(-c\sqrt x/\log x)$$ with fixed $$c>0$$ is excluded as in Theorem 1. For $$\omega(y)=c\,y/\log^2(e+y)$$ the series converges, and for some $$c>0$$ there are nonzero even functions $$f$$ with all moments zero and $$\lvert f(y)\rvert\le Ce^{-\omega(\lvert y\rvert)}$$ (sketch): take $$f(y)=\cos y\prod_{k\ge1}\operatorname{sinc}(a_ky)$$ with $$a_k=\varepsilon/(k\log^2(k+2))$$ and $$\varepsilon$$ small enough that $$\sum_ka_k<1$$. The Fourier transform of $$f$$ is then supported away from the origin, and $$\chi(x)=f(\sqrt x)/\sqrt x$$ is a function on the half-line with all integer moments zero and decay $$\exp(-c'\sqrt x/\log^2x)$$ for some $$c'>0$$.

Such a $$\chi$$ can be shaped into a kernel by the second step of the proof of Theorem 4. Put $$f_2(x)=x^{-(d+2)}\int_x^\infty t^{d+1}\chi(t)\,dt$$ and $$\phi(v)=f_2(v^{1/d})/d$$. Since $$M[\chi](z)$$ vanishes at $$z=1,2,\dots$$ and $$\lvert\chi(x)\rvert\le Cx^{-1/2}$$ near $$0$$, the argument of that proof gives $$M[f_2](z)=M[\chi](z)/(z-d-2)$$ for $$\operatorname{Re}z>\tfrac12$$, $$\lvert f_2(x)\rvert\le Cx^{-1/2}$$ near $$0$$, and $$\lvert f_2(x)\rvert\le C\exp(-c''\sqrt x/\log^2x)$$ at large $$x$$ for some $$c''>0$$. Hence $$\phi$$ is continuous, satisfies the integrability condition of Theorem 1, and has $$M[\phi](\alpha_A+j/d)=0$$ for all $$j\ge1$$.

It remains to show $$M[\phi](\alpha_A)\ne0$$. Here $$M[\phi](\alpha_A)=M[\chi]'(d+2)$$. Now $$M[\chi](z)=\int_{\mathbb R}f(y)\lvert y\rvert^{2z-2}dy$$, and $$f$$ is a Schwartz function, because $$\hat f$$ is smooth (all moments of $$f$$ are finite) and has compact support. The Fourier transform of $$\lvert y\rvert^{2z-2}$$ is $$2\sin(\pi z)\,\Gamma(2z-1)\lvert\eta\rvert^{1-2z}$$ for $$\operatorname{Re}z>\tfrac12$$, and $$\hat f$$ vanishes near $$\eta=0$$, so

$$
M[\chi](z)=\frac{\sin(\pi z)\,\Gamma(2z-1)}{\pi}\int_{\mathbb R}\hat f(\eta)\lvert\eta\rvert^{1-2z}d\eta .
$$

The sine vanishes at the integers, and at $$z=d+2$$ the derivative is

$$
M[\phi](\alpha_A)=(-1)^d\,(2d+2)!\int_{\mathbb R}\hat f(\eta)\lvert\eta\rvert^{-2d-3}d\eta .
$$

The integral is positive: $$\hat f(\eta)=\tfrac12[\hat g(\eta-1)+\hat g(\eta+1)]$$ for $$g=\prod_k\operatorname{sinc}(a_ky)$$, and $$\hat g$$ is $$2\pi$$ times the density of $$\sum_ka_kU_k$$ with independent $$U_k$$ uniform on $$[-1,1]$$, which is nonnegative and not identically zero. So kernels with the required zeros, a nonzero area term and tail $$\exp(-c\sqrt x/\log^2x)$$ exist. The same construction with $$a_k=\varepsilon/(k\log^{\rho}(k+2))$$ gives the tail $$\exp(-c\sqrt x/\log^{\rho}x)$$ for every $$\rho>1$$, the series $$\sum_k1/(k\log^{\rho}k)$$ converging exactly then. On the logarithmic scale the tails $$\exp(-c\sqrt x/\log^{\rho}x)$$ are therefore excluded for $$\rho\le1$$ and attained for $$\rho>1$$.

A proof of (H) for general $$j$$ would make the statements about the count unconditional. If the gradient expansion contained terms at the exponents $$\alpha_A+j/d$$ only for $$j$$ in a progression $$j_0+\Delta\mathbb N$$, the threshold would be $$\Delta/(2d)$$ in $$v$$ (section 2); a zero set that contains no such progression is outside Theorem 1, and a failure of (H) for some $$j$$ would shrink the zero set that an exact kernel must have.

The theorems concern the expected count of a Poisson sprinkling. The variance of a count with the cancellations of section 1 and a heavy tail is not known. Counts that depend on the shape of the future region and not on its volume alone, and non-Poisson substrates, change the kernel and lie outside the statements above.

## References

1. <a id="ref-barton"></a>Barton, Counsell, Dowker, Gould, Jubb, Taylor, “Horizon molecules in causal set theory,” Phys. Rev. D 100, 126008 (2019), [arXiv:1909.08620](https://arxiv.org/abs/1909.08620).
2. <a id="ref-dou-sorkin"></a>Dou, Sorkin, “Black hole entropy as causal links,” Found. Phys. 33, 279 (2003), [arXiv:gr-qc/0302009](https://arxiv.org/abs/gr-qc/0302009).
3. <a id="ref-jacobson"></a>Jacobson, “Entanglement equilibrium and the Einstein equation,” Phys. Rev. Lett. 116, 201101 (2016), [arXiv:1505.04753](https://arxiv.org/abs/1505.04753).
4. <a id="ref-bd"></a>Benincasa, Dowker, “The scalar curvature of a causal set,” Phys. Rev. Lett. 104, 181301 (2010), [arXiv:1001.2725](https://arxiv.org/abs/1001.2725).
5. <a id="ref-ass"></a>Aslanbeigi, Saravani, Sorkin, “Generalized causal set d'Alembertians,” JHEP 06 (2014) 024, [arXiv:1403.1622](https://arxiv.org/abs/1403.1622).
6. <a id="ref-stieltjes"></a>Stieltjes, “Recherches sur les fractions continues,” Ann. Fac. Sci. Toulouse 8 (1894) J1–J122 and 9 (1895) A5–A47.
7. <a id="ref-boas"></a>Boas, *Entire Functions*, Academic Press, New York (1954).
8. <a id="ref-rudin"></a>Rudin, *Real and Complex Analysis*, 3rd edition, McGraw-Hill, New York (1987), chapter 19 (Denjoy–Carleman theorem).
9. <a id="ref-vdd"></a>van den Dries, *Tame Topology and O-minimal Structures*, London Mathematical Society Lecture Note Series 248, Cambridge University Press (1998).
10. <a id="ref-dmm"></a>van den Dries, Macintyre, Marker, “The elementary theory of restricted analytic fields with exponentiation,” Ann. of Math. 140, 183 (1994).
11. <a id="ref-wilkie"></a>Wilkie, “Model completeness results for expansions of the ordered field of real numbers by restricted Pfaffian functions and the exponential function,” J. Amer. Math. Soc. 9, 1051 (1996).
12. <a id="ref-pdk"></a>Pye, Donnelly, Kempf, “Locality and entanglement in bandlimited quantum field theory,” Phys. Rev. D 92, 105022 (2015), [arXiv:1508.05953](https://arxiv.org/abs/1508.05953).
