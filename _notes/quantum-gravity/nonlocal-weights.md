---
title: Exact causal-set horizon weights must be non-local
date: 2026-10-29
category: gravity
order: 4
description: "A weighted count of horizon molecules on a sprinkling can follow the area law to any finite order in the curvature gradients, but an area law at every order needs a weight kernel with a tail heavier than a stretched exponential with exponent one half in the interval length, with infinitely many sign changes, and without any band limit."
status: "Theorems proved in full for the weight kernel; their reading as statements about the horizon count rests on a stated hypothesis about the gradient expansion that has been checked only for the lowest orders."
excludes: "Weighted horizon-molecule counts on a Poisson sprinkling that follow the area law to all orders in the gradients, when the weight kernel has finite support in the molecule type, geometric or exponential decay, a stretched-exponential tail with exponent at least one half in length, finitely many sign changes, or compact support."
---

A causal set models a horizon entropy by counting horizon molecules, pairs of elements that straddle the horizon just below a spacelike slice. Weighting each molecule by a number that depends on how many further elements lie to its future can make the expected count respond to curvature like the horizon area, at every diamond size, up to a chosen order in the gradients of the curvature. The weights that do this at finite order are finite in number and alternate in sign; they are derived on [Horizon counts: which weights give the area law](/quantum-gravity/area-projector/). Here we ask whether weights exist that do it at every order. If each new gradient order brings a new correction term (hypothesis (H) of section 1), the answer is a statement about the tail of the weight kernel $$\phi(v)$$, a function of the volume $$v$$ of the region to the future of the molecule. Write $$x=v^{1/d}$$ for the corresponding length. Then there are kernels with $$\lvert\phi\rvert\lesssim\exp(-c\,x^{\kappa})$$ for every $$\kappa<\tfrac12$$, and there is none with $$\lvert\phi\rvert\le C\exp(-c\sqrt x)$$ for any $$c>0$$. The threshold does not depend on the dimension. Independently of the tail, an exact kernel changes sign infinitely often, so no kernel definable in an o-minimal structure is exact to all orders, and a kernel built from finitely many polynomials times exponentials is exact to an order that can be written down. A retarded kernel cannot be band-limited in any variable, so the lattice device of restricting exactness to a band of momenta is unavailable. The cost of exactness on a causal substrate is therefore paid in non-locality of the entropy count: for the explicit kernels below, a suppression by $$e^{-30}$$ of the tail needs lengths of at least about $$9\times10^{4}$$ discreteness lengths.

The count of horizon molecules and its leading area term are due to Dou and Sorkin [[Dou–Sorkin]](#ref-dou-sorkin) and to Barton, Counsell, Dowker, Gould, Jubb and Taylor [[Barton et al.]](#ref-barton). Jacobson's entanglement-equilibrium argument [[Jacobson]](#ref-jacobson) turns an entropy proportional to area into the linearised Einstein equation, which is the reason to ask for an area law at every size. Signed combinations of layer counts define the causal-set d'Alembertian of Benincasa and Dowker [[BD]](#ref-bd) and its generalisations [[ASS]](#ref-ass); the weights here have the same form with a different purpose, and the same threshold governs the d'Alembertian (see [The causal-set d'Alembertian has negative spectral weight in four dimensions](/quantum-gravity/causal-box-spectrum/)). The wish for a functional that does not run with the discreteness scale is the idea of a perfect action [[Hasenfratz–Niedermayer]](#ref-hn). The mathematics is classical: Stieltjes' indeterminate moment problem on the half-line [[Stieltjes]](#ref-stieltjes), reviewed in [[Berg]](#ref-berg), and Carlson's theorem [[Boas]](#ref-boas). What we add is the translation to horizon counts, the dimension-independent threshold with its explicit kernels, the proof that a causal kernel admits no band limit, the exactness index of kernels with finitely many sign changes, and the observation that the Benincasa–Dowker weights and the minimal horizon weights are extremal for it.

## 1. Kernels, symbols and the hypothesis

Let $$d\ge3$$ be the spacetime dimension and work in units of the discreteness length, so the sprinkling has density one. A horizon molecule of type $$m$$ has $$m$$ further elements to the future of its lower element and to the past of the slice; given weights $$w_0,w_1,\dots$$ the weighted count adds $$w_m$$ for each molecule. Because the further elements form a Poisson process, the expected count depends on the weights only through the kernel

$$
\phi(v)=e^{-v}\sum_{m\ge0}w_m\frac{v^m}{m!},\qquad \int_0^\infty\phi(v)\,v^{\alpha-1}dv=\Gamma(\alpha)\,Q(\alpha),\quad Q(\alpha)=\sum_m w_m\frac{(\alpha)_m}{m!}, \tag{1}
$$

where $$v$$ is the volume of the region to the future of the lower element and to the past of the slice, and $$(\alpha)_m=\alpha(\alpha+1)\cdots(\alpha+m-1)$$. We write $$M[\phi](\alpha)=\int_0^\infty\phi(v)v^{\alpha-1}dv$$ for the Mellin transform of a general kernel; for weights it equals $$\Gamma(\alpha)Q(\alpha)$$, and since $$\Gamma>0$$ on the positive axis the zeros of $$M[\phi]$$ and of $$Q$$ there are the same.

The page [Horizon counts: which weights give the area law](/quantum-gravity/area-projector/) shows that a term of the gradient expansion of the expected count whose length scale carries the density exponent $$\alpha$$ is multiplied by $$Q(\alpha)=M[\phi](\alpha)/\Gamma(\alpha)$$, and that the area term sits at $$\alpha_A=1+2/d$$. Terms at higher gradient order sit at $$\alpha_A+j/d$$ with integer $$j\ge1$$. A kernel is therefore *exact to order $$J$$* if $$M[\phi](\alpha_A)\ne0$$ and $$M[\phi](\alpha_A+j/d)=0$$ for $$1\le j\le J$$, and *exact to all orders* if this holds for every $$j\ge1$$. The hypothesis that turns this into a statement about the count is the following.

(H) For every integer $$j\ge1$$ the expansion of the expected count in the gradients of the curvature contains, for some admissible background, a term at the exponent $$\alpha_A+j/d$$ with a nonzero coefficient that does not depend on the weights.

Under (H), an area law at every size and to every gradient order holds exactly when $$\phi$$ is exact to all orders. Hypothesis (H) concerns the geometry of the horizon, not the kernel. It has been verified by explicit computation for the lowest exponents, including a time-dependent background in which the first gradient order beyond uniform curvature appears (section 9 of the area-projector page), and we have no argument for all $$j$$. Every statement below about the count depends on (H); the theorems about kernels do not.

*Example.* In $$d=4$$ the weights of lowest degree $$N$$ that are exact to the first $$N$$ orders are the coefficients of $$Q_N(\alpha)=\binom{N+6-4\alpha}{N}$$, whose zeros are $$7/4,2,\dots,(N+6)/4$$ and with $$Q_N(3/2)=1$$. For $$N=4$$ they are $$(210,-791,1268,-928,256)$$ and for $$N=6$$ they are $$(924,-7084,23668,-41416,39808,-19968,4096)$$; we recomputed both from the binomial form. Each further gradient order adds exponents at the next steps of the lattice $$\alpha_A+j/4$$ (see the area-projector page) and costs a larger cancellation. With $$Q(3/2)=1$$ the ratio $$\int\lvert\phi\rvert v^{1/2}dv/\Gamma(3/2)$$, the weight of the area term against the size of the terms that cancel it, is $$115$$, $$929$$, $$7.2\times10^{3}$$ and $$4.0\times10^{5}$$ for $$N=4,6,8,12$$, about a factor $$2.8$$ per extra zero. Since $$\binom{N+z}{N}\sim N^{z}/\Gamma(z+1)$$, after rescaling the density by $$N^{4}$$ the symbols tend to $$1/\Gamma(7-4\alpha)$$, which vanishes at every $$\alpha=7/4+j/4$$ but is not the Mellin symbol of a function with exponential decay; the next section makes this precise.

## 2. The decay threshold

Put $$\chi(x)=\phi(x^d)$$. Substituting $$v=x^d$$ in $$M[\phi]$$ gives $$M[\phi](\alpha)=d\int_0^\infty\chi(x)x^{d\alpha-1}dx$$, and at $$\alpha=\alpha_A+j/d$$ the exponent is $$d\alpha-1=d+1+j$$. So the conditions $$M[\phi](\alpha_A+j/d)=0$$ say that the integer moments of $$\chi$$ of order $$d+2,d+3,\dots$$ vanish. The question is how fast a nonzero function on the half-line can decay while all its moments from some order on are zero. That is the classical indeterminacy question of Stieltjes, with the threshold $$e^{-c\sqrt x}$$ separating the determinate from the indeterminate half-line problem.

<div class="theorem" markdown="1">
**Theorem 1** (decay threshold). Let $$\phi$$ be measurable on $$(0,\infty)$$, bounded on $$(0,1)$$, with $$\lvert\phi(v)\rvert\le C\exp(-c\,v^{1/(2d)})$$ for $$v\ge1$$ and some $$C,c>0$$. If $$M[\phi](\alpha_A+j/d)=0$$ for all integers $$j\ge j_0$$, for some $$j_0\ge1$$, then $$\phi=0$$ almost everywhere.
</div>

The idea is to turn the vanishing moments into the vanishing of all Taylor coefficients at the origin of a Fourier transform, and to use that a decay of the form $$e^{-c\sqrt x}$$ is just enough to make that transform analytic. The only step that needs care is the symmetrisation that converts the half-line $$\sqrt x$$-decay into exponential decay on the line.

<div class="proof" markdown="1">
*Proof.* With $$\chi$$ as above, $$\lvert\chi(x)\rvert\le Ce^{-c\sqrt x}$$ for $$x\ge1$$ and $$\chi$$ is bounded on $$(0,1)$$. Let $$i_0=d+1+j_0$$ and $$\chi_0(x)=x^{i_0}\chi(x)$$. By the hypothesis $$\int_0^\infty\chi_0(x)x^{m}dx=0$$ for every integer $$m\ge0$$. Define the even function $$f(y)=\lvert y\rvert\chi_0(y^2)$$ on $$\mathbb R$$. The substitution $$x=y^2$$ gives $$\int_{\mathbb R}f(y)y^{2m}dy=\int_0^\infty\chi_0(x)x^{m}dx=0$$, and the odd moments of $$f$$ vanish by symmetry. For $$\lvert y\rvert\ge1$$ we have $$\lvert f(y)\rvert\le C\lvert y\rvert^{1+2i_0}e^{-c\lvert y\rvert}\le C'e^{-c\lvert y\rvert/2}$$. Hence $$\hat f(\eta)=\int f(y)e^{-i\eta y}dy$$ is holomorphic in the strip $$\lvert\operatorname{Im}\eta\rvert<c/2$$, and its derivatives at $$\eta=0$$ are, up to factors $$(-i)^k$$, the moments of $$f$$, all zero. A holomorphic function on a connected strip with all derivatives zero at one point vanishes identically. So $$\hat f=0$$, hence $$f=0$$ almost everywhere, hence $$\chi_0=0$$ and $$\phi=0$$ almost everywhere.
</div>

Because $$c>0$$ is arbitrary, the exponent $$\tfrac12$$ in length is excluded outright, together with every faster decay. The theorem has the following consequences for weighted counts, each by checking the decay hypothesis.

<div class="corollary" markdown="1">
**Corollary 2.** Under (H) no weighting by the molecule type $$m$$ is exact to all orders if (a) only finitely many $$w_m$$ are nonzero, or (b) $$\lvert w_m\rvert\le Cr^m$$ for some $$r<1$$, or (c) $$\phi$$ decays like $$e^{-cv}$$ for some $$c>0$$, or (d) $$\phi$$ is bounded and vanishes outside a bounded range of $$v$$.
</div>

<div class="proof" markdown="1">
*Proof.* In case (b), $$\lvert\phi(v)\rvert\le Ce^{-v}\sum(rv)^m/m!=Ce^{-(1-r)v}$$, which is case (c). In case (a), $$\phi=e^{-v}\times$$ polynomial, so $$\lvert\phi\rvert\le C'e^{-v/2}$$. Cases (c) and (d) satisfy the hypothesis of Theorem 1 for $$v\ge1$$, because $$e^{-cv}\le e^{-c\,v^{1/(2d)}}$$ there; for (d) take any $$c$$. Boundedness of $$\phi$$ on $$(0,1)$$ holds in all cases.
</div>

Cases (a) and (b) are the statement that no finite or geometrically decaying family of weights realises an area law at every order. For exponentially decaying $$\phi$$ there is a second proof that uses only Carlson's theorem and shows where the exponent $$1$$ is used.

<div class="remark" markdown="1">
**Remark 3** (Carlson's theorem). Let $$\lvert\phi(v)\rvert\le Ce^{-cv}$$ and write $$Q=M[\phi]/\Gamma$$. For $$x\ge x_0>0$$ the product formula $$\Gamma(x)^2/\lvert\Gamma(x+iy)\rvert^2=\prod_{n\ge0}\big(1+y^2/(x+n)^2\big)$$ shows that $$\Gamma(x)/\lvert\Gamma(x+iy)\rvert\le\Gamma(x_0)/\lvert\Gamma(x_0+iy)\rvert\lesssim(1+\lvert y\rvert)^{1/2-x_0}e^{\pi\lvert y\rvert/2}$$, uniformly in $$x$$. With $$\lvert M[\phi](x+iy)\rvert\le C\Gamma(x)c^{-x}$$ this gives $$\lvert Q(x+iy)\rvert\le C'c^{-x}(1+\lvert y\rvert)^{1/2-x_0}e^{\pi\lvert y\rvert/2}$$ on $$\operatorname{Re}\alpha\ge x_0$$. Then $$f(z)=Q(\alpha_0+z/d)$$, with $$\alpha_0=\alpha_A+j_0/d$$, is of exponential type in $$\operatorname{Re}z\ge0$$ with growth $$e^{\pi\lvert y\rvert/(2d)}$$ on the imaginary axis, and $$\pi/(2d)<\pi$$. It vanishes at $$z=0,1,2,\dots$$, so $$f\equiv0$$ by Carlson's theorem [[Boas]](#ref-boas), hence $$M[\phi]\equiv0$$ and $$\phi=0$$. The route does not reach stretched exponentials: for $$\lvert\phi\rvert\le Ce^{-cv^{\theta}}$$ with $$\theta<1$$ the bound on $$M[\phi]$$ is $$\Gamma(x/\theta)$$, which makes $$Q$$ of infinite exponential type.
</div>

The weights can also be demanded to vanish at only some of the exponents. If $$M[\phi](\alpha_A+j/d)=0$$ only for $$j\in j_0+g\mathbb N$$ with a fixed integer $$g\ge1$$, the same proof in the variable $$u=x^g$$ shows $$\phi=0$$ whenever $$\lvert\phi\rvert\le C\exp(-c\,v^{g/(2d)})$$. For example, if only every second exponent is demanded ($$g=2$$), the threshold in $$v$$ is $$v^{1/d}$$, and exponential decay in $$v$$ is still excluded for $$d\ge2$$.

## 3. Kernels below the threshold

<div class="theorem" markdown="1">
**Theorem 4** (existence below the threshold). Let $$0<\kappa<\tfrac12$$ and $$d\ge2$$. There is a continuous function $$\phi$$ on $$(0,\infty)$$ with $$\lvert\phi(v)\rvert\le C\exp(-c\,v^{\kappa/d})$$ for any $$c<\cos\pi\kappa$$, such that $$M[\phi](\alpha_A+j/d)=0$$ for all $$j\ge1$$ and $$M[\phi](\alpha_A)=(-1)^d\pi\,\Gamma\big((d+2)/\kappa\big)/\kappa\neq0$$.
</div>

The construction is Stieltjes' classical example of a function with vanishing moments. The zeros of the Mellin transform come from a factor $$\sin\pi z$$, and the condition $$\kappa<\tfrac12$$ is what allows a contour rotation by the angle $$\pi\kappa<\pi/2$$.

<div class="proof" markdown="1">
*Proof.* Let $$f_1(x)=e^{-x^{\kappa}\cos\pi\kappa}\sin(x^{\kappa}\sin\pi\kappa)$$. Rotating the contour in $$\int_0^\infty t^{a-1}e^{-te^{-i\vartheta}}dt=\Gamma(a)e^{ia\vartheta}$$ for $$\lvert\vartheta\rvert<\pi/2$$ and taking imaginary parts gives $$\int_0^\infty t^{a-1}e^{-t\cos\vartheta}\sin(t\sin\vartheta)\,dt=\Gamma(a)\sin(a\vartheta)$$. With $$t=x^\kappa$$, $$a=z/\kappa$$ and $$\vartheta=\pi\kappa$$ this yields

$$
M[f_1](z)=\int_0^\infty f_1(x)x^{z-1}dx=\frac{\Gamma(z/\kappa)\sin\pi z}{\kappa}, \tag{2}
$$

which vanishes at every positive integer $$z$$. Define $$f_2(x)=x^{-(d+2)}\int_x^\infty t^{d+1}f_1(t)\,dt$$. By Fubini, for $$\operatorname{Re}z>d+2$$,

$$
M[f_2](z)=\frac{M[f_1](z)}{z-d-2}=\frac{\Gamma(z/\kappa)\sin\pi z}{\kappa\,(z-d-2)}. \tag{3}
$$

Since $$M[f_1](d+2)=0$$, we have $$\int_x^\infty t^{d+1}f_1=-\int_0^x t^{d+1}f_1=O(x^{d+2+\kappa})$$ as $$x\to0$$, so $$f_2$$ is continuous with $$f_2(x)=O(x^\kappa)$$ near the origin. Its Mellin transform therefore converges for $$\operatorname{Re}z>-\kappa$$ and equals (3) there by analytic continuation. At $$z=d+2+j$$ with $$j\ge1$$ the right side of (3) vanishes; at $$z=d+2$$ it equals $$(-1)^d\pi\Gamma((d+2)/\kappa)/\kappa\ne0$$. For large $$x$$, with $$A=(d+2)/\kappa$$ and $$c_\kappa=\cos\pi\kappa$$, $$\lvert f_2(x)\rvert\le x^{-(d+2)}\kappa^{-1}\int_{x^\kappa}^\infty s^{A-1}e^{-c_\kappa s}ds\lesssim x^{-\kappa}e^{-c_\kappa x^\kappa}$$. Finally $$\phi(v)=f_2(v^{1/d})/d$$ has $$M[\phi](\alpha)=M[f_2](d\alpha)$$, and $$d\alpha=d+2+j$$ at $$\alpha=\alpha_A+j/d$$.
</div>

*Numerical check.* We evaluated the integral (2) by quadrature at 40 digits for $$\kappa=0.25,0.4,0.45$$ and $$z=6,7,9$$ and compared the result with the integral of the absolute value of the integrand. The ratio is below $$3\times10^{-47}$$ in all nine cases, so the moments vanish to the working precision. At the non-integer $$z=6.5$$ the quadrature agrees with the closed form (2) to twelve digits for all three values of $$\kappa$$ (for $$\kappa=0.4$$, both give $$6.49997948568\times10^{12}$$).

The decay is slow in the sense that matters. The factor $$e^{-x^{\kappa}\cos\pi\kappa}$$ falls below $$e^{-30}$$ at $$x=3.2\times10^{6},\ 4.9\times10^{5},\ 9.3\times10^{4},\ 1.2\times10^{5}$$ for $$\kappa=0.25,\,0.3,\,0.4,\,0.45$$; the smallest value over $$\kappa$$ is $$9.0\times10^{4}$$, at $$\kappa\approx0.41$$. Molecules whose future region has a length of the order of $$10^{5}$$ discreteness lengths keep contributing to the count at that level. This is the non-locality in the title: it concerns the volume of the region to the future of the molecule, not a coordinate distance in the horizon.

The two theorems leave one question open. Theorem 1 excludes every tail of the form $$e^{-c\sqrt x}$$ with $$c>0$$, and Theorem 4 provides tails $$e^{-cx^{\kappa}}$$ with $$\kappa<\tfrac12$$. Whether tails between these two, such as $$\exp(-\sqrt x/\log^{2}x)$$, admit such kernels is not decided here; it is a question about moment problems with logarithmic corrections to the exponent.

## 4. Oscillation: finitely many sign changes bound the exactness

A heavy tail is necessary but not sufficient. A positive kernel has $$M[\phi]>0$$ everywhere, so it cancels nothing. The relevant second condition is on the number of sign changes of $$\phi$$.

<div class="lemma" markdown="1">
**Lemma 5** (sign changes). Let $$\phi$$ be measurable on $$(0,\infty)$$ and suppose $$M[\phi](\alpha_k)=0$$ with absolute convergence for $$Z$$ distinct real numbers $$\alpha_1,\dots,\alpha_Z$$. Then either $$\phi=0$$ almost everywhere or $$\phi$$ has at least $$Z$$ sign changes. Here $$\phi$$ has $$r$$ sign changes if there are points $$0<v_1<\dots<v_r$$ such that, after fixing one overall sign, $$(-1)^i\phi\ge0$$ almost everywhere on $$(v_i,v_{i+1})$$ for $$i=0,\dots,r$$ (with $$v_0=0$$, $$v_{r+1}=\infty$$), and $$r$$ is the smallest number for which such points exist.
</div>

The proof compares $$\phi$$ with a generalised polynomial that has the same sign pattern.

<div class="proof" markdown="1">
*Proof.* We first show that a nonzero sum $$g(v)=\sum_{k=1}^{n}c_kv^{\beta_k}$$ with distinct real $$\beta_k$$ has at most $$n-1$$ zeros in $$(0,\infty)$$, counted with multiplicity. For $$n=1$$ there are none. For $$n>1$$, divide by $$v^{\beta_1}$$, which does not change the zeros, and differentiate. This removes the first term, leaves a nonzero sum of $$n-1$$ terms with distinct exponents, and by Rolle's theorem loses at most one zero; induction finishes.

Suppose $$\phi\ne0$$ has $$r<Z$$ sign changes at $$v_1<\dots<v_r$$. Take the first $$r+1$$ exponents and the sum $$g(v)=\sum_{k=1}^{r+1}c_kv^{\alpha_k-1}$$. The $$r$$ linear conditions $$g(v_i)=0$$ on $$r+1$$ coefficients have a nonzero solution. By the first paragraph $$g$$ has no zeros other than the $$v_i$$, each simple, so $$g$$ changes sign exactly at the $$v_i$$, and after fixing the overall sign of $$c$$ we have $$g\phi\ge0$$ almost everywhere. But $$\int g\phi\,dv=\sum_kc_kM[\phi](\alpha_k)=0$$, so $$g\phi=0$$ almost everywhere, and since $$g$$ vanishes only at finitely many points, $$\phi=0$$ almost everywhere. This contradicts $$\phi\ne0$$.
</div>

<div class="corollary" markdown="1">
**Corollary 6.** A kernel that is exact to order $$J$$ (and not zero) has at least $$J$$ sign changes, and a kernel that is exact to all orders has infinitely many.
</div>

The Corollary is Lemma 5 with $$Z=J$$ and with arbitrarily large $$Z$$. It makes oscillation a separate requirement from decay. A positive heavy-tailed kernel has the tail without the sign changes, and $$e^{-v}\sin v$$ has the sign changes with exponential decay; neither satisfies Theorem 1 and Corollary 6 together.

A class of functions in which sign changes are finite in number is the class of *tame* functions: those definable in an o-minimal expansion of the real field [[van den Dries]](#ref-vdd), which includes every finite combination of polynomials, exponentials, logarithms and restricted analytic functions. By the monotonicity theorem of o-minimal geometry, the half-line splits into finitely many points and intervals on each of which a definable function is continuous and either constant or strictly monotone. Such a function has finitely many sign changes.

<div class="theorem" markdown="1">
**Theorem 7** (tame kernels are finitely exact). Let $$\phi$$ be definable in an o-minimal structure and not zero, with $$M[\phi]$$ absolutely convergent on an interval $$I$$. Then $$M[\phi]$$ has only finitely many zeros in $$I$$. In particular no tame kernel is exact to all orders.
</div>

<div class="proof" markdown="1">
*Proof.* By the monotonicity theorem $$\phi$$ has finitely many sign changes, say $$r$$. If $$M[\phi]$$ had $$r+1$$ distinct zeros in $$I$$, Lemma 5 would give $$\phi=0$$.
</div>

The finite combinations of exponentials with polynomial coefficients are tame, and for them the bound can be made exact. For $$q$$ distinct rates $$\lambda_i>0$$ and degrees $$D_i$$ let $$\mathcal E(D_1,\dots,D_q)$$ be the span of $$v^ke^{-\lambda_iv}$$ with $$0\le k\le D_i$$. Weighted counts with $$N+1$$ weights are the case $$q=1$$, $$\lambda_1=1$$, $$D_1=N$$.

<div class="proposition" markdown="1">
**Proposition 8** (exactness index of exponential polynomials). Put $$K=\sum_i(D_i+1)-1$$. (i) A nonzero $$\phi\in\mathcal E(D_1,\dots,D_q)$$ has at most $$K$$ zeros of $$M[\phi]$$ on $$(0,\infty)$$. (ii) For any $$K$$ distinct points $$\alpha_1,\dots,\alpha_K>0$$ and any $$\alpha_A>0$$ different from them, there is, up to a factor, exactly one $$\phi\in\mathcal E$$ with $$M[\phi](\alpha_k)=0$$ for all $$k$$, and it has $$M[\phi](\alpha_A)\ne0$$.
</div>

<div class="proof" markdown="1">
*Proof.* (i) We bound the number of positive zeros of $$\phi=\sum_ip_i(v)e^{-\lambda_iv}$$ by induction on $$q$$. For $$q=1$$, $$\phi=p_1e^{-\lambda_1v}$$ has at most $$D_1$$ zeros. For $$q>1$$ multiply by $$e^{\lambda_1v}$$, which keeps the zeros, and differentiate $$D_1+1$$ times. This removes $$p_1$$ and turns each other term into a polynomial of the same degree times $$e^{-(\lambda_i-\lambda_1)v}$$, because $$(pe^{-\mu v})'=(p'-\mu p)e^{-\mu v}$$ and $$\mu\ne0$$. By Rolle's theorem each differentiation loses at most one zero, so the number of zeros of $$\phi$$ is at most the bound for $$q-1$$ rates plus $$D_1+1$$, which is $$K$$. The function $$\phi$$ is continuous, so its sign changes occur at zeros and there are at most $$K$$ of them. By Lemma 5, $$M[\phi]$$ then has at most $$K$$ zeros.

(ii) The $$K$$ conditions $$M[\phi](\alpha_k)=0$$ are linear in the $$K+1$$ coefficients, so a nonzero solution exists. If $$M[\phi](\alpha_A)$$ also vanished, $$M[\phi]$$ would have $$K+1$$ zeros, contradicting (i). The same argument shows the solution is unique up to a factor: two independent solutions would have a combination vanishing at $$\alpha_A$$ as well.
</div>

For weighted counts the proof gives more.

<div class="corollary" markdown="1">
**Corollary 9** (alternation). Let $$\phi=e^{-v}p(v)$$ with $$p(v)=\sum_{m=0}^Nw_mv^m/m!$$, and suppose $$M[\phi]$$ vanishes at $$N$$ distinct positive points. Then all $$w_m$$ are nonzero and alternate in sign, and $$p$$ has exactly $$N$$ positive zeros, all simple.
</div>

<div class="proof" markdown="1">
*Proof.* By Lemma 5, $$\phi$$ has at least $$N$$ sign changes, so $$p$$ has at least $$N$$ positive zeros of odd multiplicity. Descartes' rule of signs bounds the number of positive zeros, with multiplicity, by the number of sign changes in the coefficient sequence, which is at most $$N$$. Both inequalities are equalities.
</div>

We checked the corollary on the cases that occur. The weights $$(210,-791,1268,-928,256)$$ have $$p$$ with simple positive zeros at $$0.361$$, $$1.583$$, $$4.052$$ and $$8.504$$; the $$N=6$$ weights listed in section 1 have six simple positive zeros, at $$0.178,\,0.846,\,2.217,\,4.497,\,8.002,\,13.51$$. The Benincasa–Dowker weights $$(1,-2,1)$$ in two dimensions and $$(1,-9,16,-8)$$ in four have Mellin symbols with zeros at $$\{1,2\}$$ and $$\{\tfrac12,1,\tfrac32\}$$ and also alternate; each of these weightings has degree $$N$$ and exactly $$N$$ zeros of its symbol, so each is extremal in the sense of Proposition 8. A further consequence concerns the first term that the minimal weights leave uncancelled. For $$Q_N$$ in $$d=4$$ one finds $$Q_N\big(\tfrac32+\tfrac{N+k}4\big)=\binom{-k}{N}=(-1)^N\binom{N+k-1}{N}$$; at $$k=1$$ the first gradient order beyond those removed carries weight $$(-1)^N$$ (we confirmed $$0,1,5,15$$ for $$N=4$$, $$k=0,\dots,3$$, and $$0,1,7,28$$ for $$N=6$$). Finite weights do not suppress the next correction, they only move it to higher order.

For exponential polynomials Proposition 8 also fixes how many orders a given complexity buys. We checked its sharpness numerically for $$(D_1,D_2)=(1,1)$$ with rates $$(1,2)$$ and zeros at $$\alpha=1,2,3.5$$; $$(D_1,D_2)=(2,0)$$ with rates $$(1,3)$$ and zeros at $$0.6,1.7,2.9$$; and $$(D_1,D_2,D_3)=(1,1,0)$$ with rates $$(1,2,5)$$ and zeros at $$1,2,3,4$$, each with $$\alpha_A=3/2$$. In each case the solution has exactly $$K=3,3,4$$ sign changes on a fine grid and $$M[\phi](\alpha_A)\ne0$$. For nonzero kernels that are only tame in a larger structure no such formula is proved here.

## 5. No band limit on a causal substrate

One way to make an identity hold exactly on a periodic lattice is to impose it on a band of momenta only; the role of this device for lattice gravitons is discussed on [Lattice gravitons, exact diffeomorphisms and lattice momentum](/quantum-gravity/lattice-diffeomorphisms/). A causal substrate has no such device in any of the variables of the problem.

<div class="proposition" markdown="1">
**Proposition 10** (no band limit). (a) Let $$h\in L^1(0,\infty)$$. Its Fourier transform $$\hat h(\omega)=\int_0^\infty h(s)e^{-i\omega s}ds$$ cannot vanish on an interval of the real line unless $$h=0$$. This applies to $$\phi$$ in the variable $$v$$ and to $$\chi$$ in the length $$x$$. (b) Let $$K$$ be a finite signed measure on $$\mathbb R^d$$, $$d\ge2$$, supported in the closed future light cone, with $$\hat K(q)=\int e^{-iq\cdot x}K(dx)$$. If $$\hat K$$ vanishes on a nonempty open set of real momenta, then $$K=0$$. (c) A kernel $$\phi$$ that is exact to all orders cannot have compact support in $$v$$, by Theorem 1; and $$M[\phi]$$ cannot vanish on an interval of $$\alpha$$ unless $$\phi=0$$, because it is holomorphic in a half-plane.
</div>

<div class="proof" markdown="1">
*Proof.* (a) The function $$\hat h$$ is holomorphic for $$\operatorname{Im}\omega<0$$ and continuous on the closed half-plane. If it vanishes on a real interval, the function equal to $$\hat h$$ below and $$0$$ above is continuous across the interval and holomorphic on each side, hence holomorphic by Morera's theorem, and it vanishes on an open set. By the identity theorem $$\hat h\equiv0$$ and $$h=0$$.

(b) Let $$e$$ be a future-directed timelike vector, so $$e\cdot x\le0$$ for $$x$$ in the cone (signature $$-+\dots+$$). For complex $$\zeta$$, $$\lvert e^{-i\zeta e\cdot x}\rvert=e^{\operatorname{Im}\zeta\,e\cdot x}\le1$$ when $$\operatorname{Im}\zeta\ge0$$. So $$\zeta\mapsto\hat K(q+\zeta e)$$ is bounded, holomorphic in the upper half-plane and continuous up to the real axis. If $$q$$ lies in the open set $$U$$ where $$\hat K=0$$, it vanishes for small real $$\zeta$$, hence by (a) for all real $$\zeta$$. So $$\hat K=0$$ on $$U+\mathbb R e$$ for every timelike $$e$$, that is on $$U_1=U+T$$, with $$T$$ the open set of all timelike vectors. The same argument applied to the open set $$U_1$$ gives vanishing on $$U_1+T=U+T+T$$. Every vector $$v$$ is a sum of two timelike vectors: with $$e_0$$ the unit time direction and $$\lambda>\lvert v\rvert$$, $$v=(v/2+\lambda e_0)+(v/2-\lambda e_0)$$. So $$T+T=\mathbb R^d$$, $$\hat K\equiv0$$, and $$K=0$$ by uniqueness of the Fourier transform of finite measures.

(c) The first statement is the case of bounded support in Theorem 1. The second is the identity theorem for the holomorphic function $$M[\phi]$$ in $$\operatorname{Re}\alpha>0$$.
</div>

Part (b) says that a retarded kernel with finite total variation is determined by its symbol on any open set of momenta, so it cannot agree with a prescribed symbol on a band and be free outside it. Bandlimited quantum field theory [[Pye–Donnelly–Kempf]](#ref-pdk) restricts the momentum content of the field; the proposition concerns the kernel of a retarded operator, which is a different object. Hence the condition of exactness on the discrete infinite set $$\{\alpha_A+j/d\}$$ cannot be relaxed to a condition on an interval, and the cost has to be paid in the tail class of Theorem 1.

## 6. What the theorems do not exclude

Each item below is outside the hypotheses of the results above, with the condition under which it applies.

- Exactness to a finite order $$J$$. Theorem 1 does not apply, the weights of section 1 realise it, and Proposition 8 and Corollary 9 describe their structure. The first uncancelled term enters with weight $$(-1)^N$$.
- Failure of (H). If the expansion has terms at the exponents $$\alpha_A+j/d$$ only for $$j$$ in a sparser set, the required zero set is smaller. For a progression of step $$g$$ the threshold is $$g/(2d)$$ in $$v$$ (section 2); a zero set that contains no such progression is not covered by Theorem 1.
- Heavy-tailed oscillating kernels as in Theorem 4. They are exact to all orders in the sense of section 1. A weighting by the molecule type has $$\phi=e^{-v}E(v)$$ with $$E$$ entire, so $$\phi$$ is analytic at $$v=0$$, whereas the kernel of Theorem 4 behaves like $$v^{\kappa/d}$$ there. Whether any sequence $$w_m$$ realises a kernel with the required tail is not settled here.
- Functionals that are not of the form weighted molecule count with a mean depending on the type $$m$$ alone, for example counts that depend on the shape of the future region, and non-Poisson substrates.
- Statements about fluctuations. The theorems concern the expected count. A kernel with the cancellations of section 1 and a tail as in Theorem 4 may have a large variance; it has not been computed.
- Lorentz-violating or lattice substrates, where band-limiting is available at a different cost.

## 7. Open points

The coefficient claim (H) is established for the lowest exponents only. A proof or counterexample for general $$j$$ would turn the conditional statements about the count into unconditional ones. The determinacy question at the boundary, for tails between $$e^{-c\sqrt x}$$ and $$e^{-cx^{\kappa}}$$ with $$\kappa<\tfrac12$$, is open. It is not known whether a sequence of weights $$w_m$$ can have a kernel with a tail of Theorem 4 type. The bound on the order of exactness of a tame kernel in terms of its complexity is proved here only for exponential polynomials; for kernels definable in larger o-minimal structures, explicit bounds of the Pfaffian or sharply o-minimal kind exist in the literature but are not used here.

## References

1. <a id="ref-barton"></a>Barton, Counsell, Dowker, Gould, Jubb, Taylor, “Horizon molecules in causal set theory,” Phys. Rev. D 100, 126008 (2019), [arXiv:1909.08620](https://arxiv.org/abs/1909.08620).
2. <a id="ref-dou-sorkin"></a>Dou, Sorkin, “Black hole entropy as causal links,” Found. Phys. 33, 279 (2003), [arXiv:gr-qc/0302009](https://arxiv.org/abs/gr-qc/0302009).
3. <a id="ref-jacobson"></a>Jacobson, “Entanglement equilibrium and the Einstein equation,” Phys. Rev. Lett. 116, 201101 (2016), [arXiv:1505.04753](https://arxiv.org/abs/1505.04753).
4. <a id="ref-bd"></a>Benincasa, Dowker, “The scalar curvature of a causal set,” Phys. Rev. Lett. 104, 181301 (2010), [arXiv:1001.2725](https://arxiv.org/abs/1001.2725).
5. <a id="ref-ass"></a>Aslanbeigi, Saravani, Sorkin, “Generalized causal set d'Alembertians,” JHEP 06 (2014) 024, [arXiv:1403.1622](https://arxiv.org/abs/1403.1622).
6. <a id="ref-hn"></a>Hasenfratz, Niedermayer, “Perfect lattice action for asymptotically free theories,” Nucl. Phys. B 414, 785 (1994), [arXiv:hep-lat/9308004](https://arxiv.org/abs/hep-lat/9308004).
7. <a id="ref-stieltjes"></a>Stieltjes, “Recherches sur les fractions continues,” Ann. Fac. Sci. Toulouse 8 (1894) J1–J122 and 9 (1895) A5–A47.
8. <a id="ref-berg"></a>Berg, “Indeterminate Stieltjes moment problems revisited,” Arab. J. Math. (2025), [arXiv:2407.08511](https://arxiv.org/abs/2407.08511).
9. <a id="ref-boas"></a>Boas, *Entire Functions*, Academic Press, New York (1954).
10. <a id="ref-vdd"></a>van den Dries, *Tame Topology and O-minimal Structures*, London Mathematical Society Lecture Note Series 248, Cambridge University Press (1998).
11. <a id="ref-pdk"></a>Pye, Donnelly, Kempf, “Locality and entanglement in bandlimited quantum field theory,” Phys. Rev. D 92, 105022 (2015), [arXiv:1508.05953](https://arxiv.org/abs/1508.05953).
