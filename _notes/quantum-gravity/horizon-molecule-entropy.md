---
title: Weighted horizon-molecule counts in causal sets
date: 2026-10-28
order: 1
description: Weighting causal-set horizon molecules by the number of elements to their future can make the expected count respond to curvature exactly as the horizon area does, at every diamond size, at linear order and for uniform curvature. The weights are unique at minimal degree and must take both signs, and exactness to all orders in gradients needs weights with very slowly decaying tails.
---

In causal set theory the entropy of a horizon is modelled by counting horizon molecules, pairs of elements that straddle the horizon just below a spacelike slice. In the continuum limit the expected number of molecules is proportional to the area of the horizon cross-section [[Barton et al.]](#ref-barton). At finite density the expected count also contains terms suppressed by powers of the discreteness length, and in a small causal diamond these terms make its response to curvature differ from that of the area. We ask whether some weighting of the molecules by $$m$$, the number of further elements to the causal future of the lower element and to the past of the slice, has an expected count that responds to curvature exactly as the area does at every diamond size. Jacobson's entanglement-equilibrium argument would then return the linearised Einstein equation with the Newton constant fixed by the flat-space area law. At linear order in curvature that is uniform over the diamond, and for the expected count, such weightings exist: a weighting has the property exactly when a polynomial built from its weights, its Mellin symbol, vanishes at $$d$$ points fixed by dimensional analysis, where $$d$$ is the spacetime dimension (section 5). In four dimensions the weighting of lowest degree is unique up to normalisation, with integer weights $$(210,-791,1268,-928,256)$$ on $$m=0,\dots,4$$. No weighting with nonnegative weights has the property; for such weightings the effective Newton constant is always smaller than the one fixed by the area law (section 6).

The result has three limits. First, it fixes the entropy functional but does not derive gravity. Once the leading term of the count is the local area, which is known at leading order, every weighting that removes the subleading terms gives the same Newton constant. The non-linear Einstein equation follows only under a condition, stated in section 7 and not proved here, on the count for geodesic balls in slices whose extrinsic curvature vanishes at the centre. Second, exactness holds for the horizon sheet that meets no caustic below the slice; for the lower null boundary of the diamond itself it fails for diamonds of a few discreteness lengths (section 7). Third, a weighting of finite degree is exact only to finite order in gradients of the curvature, and a time derivative of the scalar curvature already leaves a residual at linear order. Exactness to all orders is possible only for weight kernels that decay more slowly than $$\exp(-c\,(\rho V)^{1/(2d)})$$, where $$\rho$$ is the sprinkling density and $$V$$ the volume to the future of the lower element; this threshold is sharp (section 8).

Horizon molecules in the form used here, their flat-space coefficient, and the leading-order locality of the count in curved spacetime are due to Barton, Counsell, Dowker, Gould, Jubb and Taylor [[Barton et al.]](#ref-barton), following Dou and Sorkin [[Dou–Sorkin]](#ref-dou-sorkin). Barton et al. also form signed combinations of counts of different molecule types, there to cancel the area term and isolate the first correction. Signed combinations of Poisson-weighted layer counts define the causal-set d'Alembertian and action of Benincasa and Dowker [[BD]](#ref-bd) and their generalisations [[Dowker–Glaser]](#ref-dg) [[Glaser]](#ref-glaser) [[ASS]](#ref-ass); the weights here have the same form and a different purpose (section 5). Jacobson [[Jacobson 2015]](#ref-jacobson15) derives the Einstein equation from the area deficit of small geodesic balls at fixed volume, assuming an entropy proportional to area. This note adds the exact density dependence of the weighted count, the characterisation of the exact weights in every dimension, the sign constraints, the role of the horizon's orientation, and the decay threshold for exactness to all orders. The mathematics in sections 6 and 8 is classical (Descartes' rule of signs, the Stieltjes moment problem); its application to these counts is not.

Familiarity with Poisson sprinklings of a Lorentzian manifold and with the outline of Jacobson's argument is assumed.

## 1. Horizon molecules and weighted counts

Let $$d\ge3$$ be the spacetime dimension and $$n=d-1$$. A causal set is obtained by sprinkling points into spacetime by a Poisson process of density $$\rho$$, ordered by the causal relation $$\prec$$; we write $$\ell_\rho=\rho^{-1/d}$$ for the discreteness length. Let $$\Sigma$$ be a spacelike slice, $$B\subset\Sigma$$ a geodesic ball, and $$o$$ the top tip of its domain of dependence. The horizon we count across is $$\partial I^-(o)$$ below $$\Sigma$$: the past light cone of $$o$$, continued to the past of $$\Sigma$$. It meets $$\Sigma$$ in $$\partial B$$ and its generators move away from $$B$$ as one goes to the past, so it has no caustic below $$\Sigma$$. Near $$\partial B$$ it is a horizon of the type considered by Barton et al., the boundary of the past of a timelike curve. Write $$P=I^-(o)\cap I^-(\Sigma)$$ for the region inside it.

For an element $$p\in P$$ let $$F(p)$$ be the set of elements that lie to the past of $$\Sigma$$ and to the causal future of $$p$$, excluding $$p$$. A *horizon molecule* is a pair $$p\prec q$$ with $$p\in P$$ and $$q\in F(p)$$ outside $$P$$. Its *type* is $$m=\lvert F(p)\rvert-1$$, the number of elements of $$F(p)$$ other than $$q$$. A molecule of type 0 is Barton et al.'s horizon molecule: $$q$$ is the only element to the future of $$p$$ and the past of $$\Sigma$$, so $$p\prec q$$ is a link. (Their $$n$$-molecules, with all $$n$$ future elements outside the horizon, are a different family.) Given real weights $$w_0,w_1,\dots$$, the weighted count $$H_w$$ is the sum of $$w_m$$ over all molecules.

The expectation of $$H_w$$ is an integral over the position of the pair. Given $$p$$ and $$q$$, the remaining elements in $$J^+(p)\cap I^-(\Sigma)$$ form a Poisson process (the Slivnyak–Mecke property), so the probability that there are exactly $$m$$ of them is $$(\rho V)^me^{-\rho V}/m!$$, with $$V(p)$$ the volume of $$J^+(p)\cap I^-(\Sigma)$$. Writing $$V_{\rm out}(p)$$ for the volume of the part of this region outside $$P$$,

$$
\langle H_w\rangle=\rho^2\int_P dV_p\;V_{\rm out}(p)\,\phi\big(\rho V(p)\big),\qquad \phi(v)=e^{-v}\sum_m w_m\frac{v^m}{m!}. \tag{1}
$$

The weights enter only through the kernel $$\phi$$. We call

$$
Q(\alpha)=\sum_m w_m\frac{(\alpha)_m}{m!},\qquad \Gamma(\alpha)\,Q(\alpha)=\int_0^\infty\phi(v)\,v^{\alpha-1}\,dv, \tag{2}
$$

the *Mellin symbol* of the weighting; $$(\alpha)_m=\alpha(\alpha+1)\cdots(\alpha+m-1)$$, and the second equality follows from $$\int_0^\infty v^{m+\alpha-1}e^{-v}dv/m!=\Gamma(\alpha)(\alpha)_m/m!$$. Weights with $$w_m=0$$ for $$m>N$$ give a polynomial $$Q$$ of degree $$N$$. The weights are recovered from $$Q$$ through $$(-j)_m/m!=(-1)^m\binom{j}{m}$$ and binomial inversion:

$$
w_m=\sum_{j=0}^{m}(-1)^j\binom{m}{j}\,Q(-j). \tag{3}
$$

The type-0 count has $$Q\equiv1$$.

## 2. A weighting acts on the density dependence

The point of (2) is that weighting by $$m$$ is the same as acting on the dependence of the count on $$\rho$$.

<div class="lemma" markdown="1">
**Lemma 1** (density dependence). Fix the geometry and suppose that the type-0 count is a finite sum of powers of the density, $$\langle H_{0}\rangle(\rho)=\sum_\beta c_\beta\,\rho^{2-\beta}$$. Then for every weighting

$$
\langle H_w\rangle(\rho)=\sum_\beta c_\beta\,Q(\beta)\,\rho^{2-\beta}. \tag{4}
$$
</div>

<div class="proof" markdown="1">
*Proof.* Let $$G$$ be the measure on $$(0,\infty)$$ defined by $$\int f\,dG=\int_P dV_p\,V_{\rm out}(p)\,f(V(p))$$, so that $$\langle H_w\rangle=\rho^2\int\phi(\rho V)\,dG(V)$$, and let $$F(\rho)=\int e^{-\rho V}dG(V)$$, so that $$\langle H_0\rangle=\rho^2F(\rho)$$. Differentiating under the integral, $$\int(\rho V)^me^{-\rho V}dG/m!=(-\rho)^mF^{(m)}(\rho)/m!$$. On a power, $$(-\rho)^m\partial_\rho^m\rho^{-\beta}/m!=\big((\beta)_m/m!\big)\rho^{-\beta}$$. Summing over $$m$$ with weights $$w_m$$ gives (4).
</div>

In every case below the count is, order by order in the curvature, a polynomial in the size $$\ell$$ of the ball, and dimensional analysis turns powers of $$\ell$$ into powers of $$\rho$$. The count is dimensionless and depends on $$\rho$$ only through $$\rho\ell^d$$ and $$\rho$$ times $$d$$-th powers of the curvature lengths. A term $$\ell^e$$ multiplying a curvature quantity of length dimension $$-D$$ therefore carries $$\rho^{(e-D)/d}$$, that is,

$$
\beta=2-\frac{e-D}{d}. \tag{5}
$$

Equation (4) says that a weighting multiplies each such term by its own number $$Q(\beta)$$. A weighting whose symbol vanishes at every $$\beta$$ except that of the leading term keeps only the term that survives as $$\rho\to\infty$$ at fixed $$\ell$$. This is Richardson extrapolation of the type-0 count to infinite density, carried out at a single density by using the information in $$m$$.

## 3. The flat count

From here on we set $$\rho=1$$, so lengths are in units of $$\ell_\rho$$. The computations below use one exact reduction, which holds for spacetimes conformal to a static one with round spatial sections. Let

$$
g=W(t)^{2/d}\big(-dt^2+h\big),
$$

with $$h$$ the round metric on the $$n$$-sphere of radius $$A$$ ($$A=\infty$$ gives flat space), $$W(0)=1$$, $$\Sigma=\{t=0\}$$ and $$B$$ the geodesic ball of radius $$r_0$$ about a pole. Write $$\mathrm{cap}(r)$$ for the $$h$$-volume of a geodesic ball of radius $$r$$ on the sphere.

<div class="lemma" markdown="1">
**Lemma 2** (reduction). With $$\tau$$ the coordinate depth of $$p$$ below $$\Sigma$$,

$$
\langle H_w\rangle=\int_0^\infty d\tau\,W(-\tau)\,\phi\big(V(\tau)\big)\,U(\tau),\qquad
U(\tau)=\int_0^\tau du\,W(u-\tau)\,\mathrm{cap}(u)\big[\mathrm{cap}(r_0+\tau)-\mathrm{cap}(r_0+\tau-u)\big], \tag{6}
$$

with $$V(\tau)=\int_0^\tau du\,W(u-\tau)\,\mathrm{cap}(u)$$, as long as no cap wraps around the sphere (in particular to every order in $$1/A$$).
</div>

<div class="proof" markdown="1">
*Proof.* Conformal factors do not change causal relations, so $$(t_1,x_1)\prec(t_2,x_2)$$ exactly when the $$h$$-distance from $$x_1$$ to $$x_2$$ is at most $$t_2-t_1$$; the volume element is $$W\,dt\,dvol_h$$. The tip $$o$$ sits at $$t=r_0$$ above the pole, so $$P$$ at time $$-\tau$$ is the cap $$C_1$$ of radius $$r_0+\tau$$ about the pole, and $$V(p)$$ depends only on $$\tau$$. An element $$q$$ at time $$-(\tau-u)$$ is to the future of $$p=(-\tau,x_p)$$ and outside $$P$$ when it lies in the cap of radius $$u$$ about $$x_p$$ and outside the cap $$C_2$$ of radius $$r_0+\tau-u$$ about the pole. Integrating over $$x_p\in C_1$$, the volume of the cap about $$x_p$$ gives $$\mathrm{cap}(u)\,\mathrm{cap}(r_0+\tau)$$. In the part inside $$C_2$$ we exchange the roles of $$x_p$$ and $$x_q$$: for $$x_q\in C_2$$ the cap of radius $$u$$ about $$x_q$$ lies inside $$C_1$$ by the triangle inequality, so this part is $$\mathrm{cap}(u)\,\mathrm{cap}(r_0+\tau-u)$$.
</div>

In flat space $$W=1$$ and $$\mathrm{cap}(r)=\omega r^n$$, with $$\omega$$ the volume of the unit $$n$$-ball. Then $$V=\omega\tau^d/d$$, and expanding the bracket in (6) binomially in $$\ell=r_0$$, the term of order $$\ell^n$$ cancels:

$$
U(\tau)=\omega^2\sum_{k=1}^{n}\binom nk\gamma_k\,\ell^{\,n-k}\tau^{d+k},\qquad \gamma_k=\int_0^1x^n\big[1-(1-x)^k\big]dx=\frac1d-\frac{n!\,k!}{(n+k+1)!}.
$$

Substituting $$v=\omega\tau^d/d$$ turns each $$\tau$$-integral into (2).

<div class="proposition" markdown="1">
**Proposition 1** (flat count). For a flat ball of radius $$\ell$$,

$$
\langle H_w\rangle=\sum_{k=1}^{n}\binom nk\gamma_k\,\frac{\omega^2}{d}\Big(\frac d\omega\Big)^{\beta_k}\Gamma(\beta_k)\,Q(\beta_k)\,\ell^{\,n-k},\qquad\beta_k=\frac{d+1+k}{d}. \tag{7}
$$

There is no term proportional to $$\ell^{n}$$, the volume of $$B$$. The leading term is $$Q(\alpha_A)\,a_d\,\lvert\partial B\rvert$$, with $$\lvert\partial B\rvert=n\omega\ell^{n-1}$$, area exponent $$\alpha_A=1+2/d$$, and

$$
a_d=\frac{\omega}{d(d+1)}\Big(\frac d\omega\Big)^{\alpha_A}\Gamma(\alpha_A). \tag{8}
$$
</div>

The proof is the computation above; $$\gamma_1=1/(d+1)$$ gives (8). The exponents present are $$\alpha_A,\alpha_A+\tfrac1d,\dots,2$$. Equation (8) gives $$a_3=\tfrac14(3/\pi)^{2/3}\Gamma(5/3)=0.218853$$ and $$a_4=\sqrt3/10$$, the coefficients found by Barton et al., and $$a_5=0.148656$$, $$a_6=0.133258$$. In four dimensions,

$$
\langle H_w\rangle=\frac{2\sqrt3\,\pi}{5}\,Q(\tfrac32)\,\ell^2+\frac{7\cdot3^{3/4}\pi^{1/4}\Gamma(\frac34)}{10}\,Q(\tfrac74)\,\ell+\frac{34}{35}\,Q(2). \tag{9}
$$

For the type-0 count ($$Q\equiv1$$) this is $$2.17656\,\ell^2+2.60322\,\ell+0.97143$$. A direct sprinkling agrees with (9) for each type separately: at $$\ell=1$$ the mean counts of types 0, 1, 2 over 600 sprinklings are $$5.78\pm0.11$$, $$9.78\pm0.18$$, $$13.15\pm0.24$$, against $$5.751$$, $$9.763$$, $$13.259$$ from (9), and at $$\ell=6$$ the type-0 count is $$95.4\pm1.3$$ (60 sprinklings) against $$94.95$$.

<div class="remark" markdown="1">
**Remark** (the first correction to the area law). Barton et al. write the first correction for the type-0 count as $$b^{(4)}_{3,1}\,\theta\,\ell_\rho$$ per unit area, with $$\theta$$ the null expansion of the horizon at $$\partial B$$, and give $$b^{(4)}_{3,1}=-\sqrt2\,\Gamma(\tfrac{11}4)/(35\cdot3^{1/4}\pi^{3/4})\approx-0.0209$$ (their eq. 5.51). For a sphere of radius $$\ell$$, $$\lvert\theta\rvert=\sqrt2/\ell$$ in their normalisation, and the linear term of (9) gives $$b^{(4)}_{3,1}=-7\cdot3^{3/4}\Gamma(\tfrac34)/(40\sqrt2\,\pi^{3/4})\approx-0.1465$$, exactly seven times their value. Their value is reproduced exactly from the integrand of their eq. (5.47) if the factor $$(1+X^1/r)^2$$, the ratio between the area of the sphere through the point and the area of $$\partial B$$, is set to one; with that factor included the integral gives the value from (9). The sprinkling at $$\ell=6$$ quoted above distinguishes the two: the shorter coefficient predicts $$80.6$$ up to the constant term $$0.97$$.
</div>

## 4. Curvature response of a ball of fixed volume

In curved space we hold the volume of $$B$$ fixed at $$\omega\ell^n$$, so that $$\ell$$ labels its size, and expand in the curvature. The central fact is that the term of $$\langle H_w\rangle$$ at the area exponent is the exact area of $$\partial B$$, to all orders, in the class of Lemma 2.

<div class="theorem" markdown="1">
**Theorem 1** (area exponent). In the setting of Lemma 2, with $$W$$ analytic, $$W(0)=1$$, and $$B$$ of fixed volume $$\omega\ell^n$$, expand $$\langle H_w\rangle$$ as a formal power series in $$1/A^2$$ and in the Taylor coefficients of $$W$$. At every order, the term multiplying $$Q(\alpha_A)$$ is $$a_d\,Q(\alpha_A)$$ times the corresponding term of $$\lvert\partial B\rvert$$, the area of the boundary of the fixed-volume ball. All other terms carry $$Q$$ at points $$\alpha_A+j/d$$ with $$j\ge1$$.
</div>

The idea is that $$\ell$$ enters (6) only through $$r_0$$ inside the bracket. At each order the highest power of $$\ell$$ comes from the first derivative $$\mathrm{cap}'(r_0)$$ with every other factor at its flat value, and $$\mathrm{cap}'(r_0)$$ is the area of $$\partial B$$.

<div class="proof" markdown="1">
*Proof.* Expand the bracket in (6) as $$\sum_{a\ge1}\mathrm{cap}^{(a)}(r_0)\,[\tau^a-(\tau-u)^a]/a!$$. The cap volume is a series $$\sum_N b_NA^{-2N}r^{n+2N}$$ with $$b_0=\omega$$, and the fixed-volume condition gives $$r_0=\ell\,\psi(\ell^2/A^2)$$ for a power series $$\psi$$; hence the order-$$A^{-2N}$$ part of $$\mathrm{cap}^{(a)}(r_0)$$ is a constant times $$\ell^{\,n+2N-a}$$. No other factor in (6), nor the expansion of $$\phi(V)$$ about $$\omega\tau^d/d$$, involves $$\ell$$. Consider a term of order $$A^{-2N}$$ that also contains Taylor coefficients of $$W$$ of total length dimension $$-D'$$ (the coefficient of $$t^k$$ has dimension $$-k$$). Its power of $$\ell$$ is at most $$n-1+2N$$, so by (5) with $$D=2N+D'$$ its exponent is at least $$2-(n-1-D')/d=\alpha_A+D'/d$$. Equality requires $$D'=0$$, $$a=1$$, and the flat values of $$\mathrm{cap}(u)$$ and $$V$$. That term is $$[\mathrm{cap}'(r_0)]_{N}\int_0^\infty d\tau\,\phi(\omega\tau^d/d)\int_0^\tau du\,\omega u^{n}\,u$$, which by the flat computation equals $$a_d\,Q(\alpha_A)\,[\mathrm{cap}'(r_0)]_N$$. Finally $$\mathrm{cap}'(r_0)=\lvert\partial B\rvert$$. All other terms have $$\beta>\alpha_A$$, and by (5) $$\beta$$ lies on the lattice $$\alpha_A+j/d$$.
</div>

At linear order and for curvature that is uniform over the diamond, two backgrounds suffice. The Einstein static universe ($$W=1$$) has spatial scalar curvature $$R^\Sigma=n(n-1)/A^2$$ and $$R_{uu}=0$$, with $$u$$ the unit normal to $$\Sigma$$. De Sitter space in conformal time ($$W=\cos^{-d}(t/A)$$) has the same slice and $$R_{uu}=-n/A^2$$. On a slice with vanishing extrinsic curvature, the linear response is a rotation-invariant linear function of the Riemann tensor at the centre, and the only such scalars are $$R^\Sigma$$ and $$R_{uu}$$: the Weyl tensor enters through $$C_{uiuj}$$, which is traceless and averages to zero over directions. We write the response at fixed volume as

$$
\delta\langle H_w\rangle\big|_V=-h_\Sigma(\ell)\,R^\Sigma-h_0(\ell)\,R_{uu}. \tag{10}
$$

The area of a geodesic ball at fixed volume changes by $$\delta\lvert\partial B\rvert_V=-c_d\,\ell^d R^\Sigma$$ with $$c_d=\omega/(2(d+1))$$ ($$c_4=2\pi/15$$). Theorem 1 then gives two facts. The part of $$h_\Sigma$$ at $$\alpha_A$$ is $$a_dQ(\alpha_A)c_d\ell^d$$. And $$h_0$$, the difference between de Sitter and the static universe, has no part at $$\alpha_A$$, because the two have the same slice and the same ball. By (5) with $$D=2$$, a term $$\ell^e$$ in $$h_\Sigma$$ or $$h_0$$ carries $$Q(2+(2-e)/d)$$. In four dimensions the result is the following table: each response is the sum of the entries in its column, and each entry is multiplied by $$Q$$ at the exponent in the first column.

| exponent $$\alpha$$ | flat count | $$h_\Sigma$$ | $$h_0$$ |
|---|---|---|---|
| $$3/2$$ | $$\frac{2\sqrt3\pi}{5}\ell^2$$ | $$\frac{\sqrt3\pi}{75}\ell^4$$ | $$0$$ |
| $$7/4$$ | $$\frac{7\cdot3^{3/4}\pi^{1/4}\Gamma(3/4)}{10}\ell$$ | $$\frac{7\cdot3^{3/4}\pi^{1/4}\Gamma(3/4)}{100}\ell^3$$ | $$0$$ |
| $$2$$ | $$\frac{34}{35}$$ | $$\frac{48}{175}\ell^2$$ | $$\frac{256}{175}\ell^2$$ |
| $$9/4$$ | | $$\frac{31\cdot3^{1/4}\Gamma(1/4)}{1120\,\pi^{1/4}}\ell$$ | $$\frac{37\cdot3^{1/4}\Gamma(1/4)}{70\,\pi^{1/4}}\ell$$ |
| $$5/2$$ | | $$\frac{\sqrt3}{1260}$$ | $$\frac{547\sqrt3}{1260}$$ |

The first entry of $$h_\Sigma$$ is $$a_4c_4\ell^4$$, as Theorem 1 requires. We computed the same tables exactly in $$d=3,\dots,8$$. In each, $$h_\Sigma$$ has nonzero coefficients at all $$d+1$$ exponents $$\alpha_A+j/d$$, $$j=0,\dots,d$$, $$h_0$$ at $$j=2,\dots,d$$, and the flat count at $$j=0,\dots,d-2$$; every coefficient is positive. Two independent calculations agree on the four-dimensional table.

## 5. The linearised Einstein equation with the area-law coefficient

We now say what is required of the count. We take the following model, which is Jacobson's setting [[Jacobson 2015]](#ref-jacobson15) with a causal-set count in place of the entanglement entropy of the short-distance degrees of freedom.

1. The short-distance entropy of the diamond over $$B$$ is $$s_0\langle H_w\rangle$$ for a constant $$s_0$$ (for instance $$\ln2$$ per molecule). Only the expected count enters; fluctuations are ignored.
2. In flat space this entropy defines Newton's constant through $$S=\lvert\partial B\rvert/4G$$ at leading order, so $$1/(4G)=s_0\,Q(\alpha_A)\,a_d\,\rho^{(d-2)/d}$$.
3. Entanglement equilibrium holds: at fixed volume the change of the short-distance entropy balances the change of the matter entropy.

Jacobson's argument shows that the balance in item 3, for every unit timelike $$u$$, is equivalent to the Einstein equation with coupling $$G_N$$ exactly when the short-distance entropy changes by $$\delta\lvert\partial B\rvert_V/4G_N$$, and that $$R^\Sigma=2G_{uu}$$ on a slice with vanishing extrinsic curvature. Comparing with (10), the count gives the linearised Einstein equation with $$G_N=G$$ at every size $$\ell$$ exactly when

(i) $$h_0(\ell)=0$$, and (ii) $$h_\Sigma(\ell)=Q(\alpha_A)\,a_d\,c_d\,\ell^d$$.

If (i) fails, the response depends on $$R_{uu}$$ separately and the equation is not Einstein's. For matter with $$R_{uu}=\lambda R^\Sigma$$ the balance then defines an effective $$G_N/G=Q(\alpha_A)a_dc_d\ell^d/(h_\Sigma+\lambda h_0)$$; in four dimensions, under the Einstein equation, $$\lambda=1/4$$ for dust and $$1/2$$ for radiation.

For the type-0 count the table gives $$G_N/G=1/(1+3.59/\ell+3.78/\ell^2+\dots)$$ for $$\lambda=0$$, so for $$\ell\gg\ell_\rho$$ both conditions hold up to relative corrections of order $$\ell_\rho/\ell$$. The conditions matter for diamonds of a few discreteness lengths. If, for example, $$G_N$$ is read off from the diamonds spanned by links, whose half-heights are distributed as $$\ell^3e^{-2\pi\ell^4/3}$$ in four dimensions, the type-0 count gives $$G_N/G=0.0832$$, $$0.0292$$ and $$0.0177$$ for $$\lambda=0$$, dust and radiation.

<div class="theorem" markdown="1">
**Theorem 2** (exact weights). Let $$d\ge3$$ and let $$w$$ be a finite weighting with $$Q(\alpha_A)\neq0$$.

(a) If $$Q(\alpha_A+j/d)=0$$ for $$j=1,\dots,d$$, then for every $$\ell>0$$ the flat count is $$Q(\alpha_A)a_d\lvert\partial B\rvert$$ and conditions (i) and (ii) hold.

(b) For $$3\le d\le8$$, condition (ii) alone implies these $$d$$ zeros.

(c) A weighting with these zeros has $$w_m\neq0$$ for some $$m\ge d$$. Among weightings with $$w_m=0$$ for $$m>d$$ it is unique up to normalisation, with

$$
Q_d(\alpha)=\binom{2d+2-d\alpha}{d},\qquad w_m=\sum_{j=0}^{m}(-1)^j\binom mj\binom{2d+2+dj}{d}. \tag{11}
$$
</div>

<div class="proof" markdown="1">
*Proof.* (a) By (4) and the structure in section 4, the weighted flat count and $$h_\Sigma$$, $$h_0$$ are sums over $$j$$ of $$Q(\alpha_A+j/d)$$ times the type-0 coefficients, with $$j\le d$$. The zeros leave only $$j=0$$, which is $$Q(\alpha_A)a_d$$ times $$\lvert\partial B\rvert$$, times $$c_d\ell^d$$, and times zero respectively. (b) Condition (ii) says that $$\sum_{j=1}^dC_j\,Q(\alpha_A+j/d)\,\ell^{d-j}$$ vanishes identically in $$\ell$$, where $$C_j$$ are the type-0 coefficients of $$h_\Sigma$$; these are nonzero for $$d\le8$$. (c) If $$w_m=0$$ for $$m>N$$, $$Q$$ is a polynomial of degree at most $$N$$, so $$d$$ zeros require $$N\ge d$$ unless $$Q\equiv0$$; for $$N=d$$ it is fixed up to a constant by its zeros. With $$x=2d+2-d\alpha$$, $$\binom xd$$ vanishes at $$x=0,\dots,d-1$$, which are the points $$\alpha=\alpha_A+j/d$$, $$j=d,\dots,1$$, and equals 1 at $$x=d$$, which is $$\alpha=\alpha_A$$. The weights follow from (3).
</div>

The weights are integers, with $$w_0=\binom{2d+2}{d}$$ and $$w_d=(-d)^d$$:

| $$d$$ | zeros of $$Q_d$$ | weights $$w_0,\dots,w_d$$ | $$a_d$$ |
|---|---|---|---|
| 3 | $$2,\ \tfrac73,\ \tfrac83$$ | $$56,\ -109,\ 90,\ -27$$ | $$0.218853$$ |
| 4 | $$\tfrac74,\ 2,\ \tfrac94,\ \tfrac52$$ | $$210,\ -791,\ 1268,\ -928,\ 256$$ | $$0.173205$$ |
| 5 | $$\tfrac85,\dots,\tfrac{12}5$$ | $$792,\ -5396,\ 14750,\ -19500,\ 12500,\ -3125$$ | $$0.148656$$ |
| 6 | $$\tfrac32,\dots,\tfrac73$$ | $$3003,\ -35757,\ 155713,\ -328779,\ 365256,\ -206064,\ 46656$$ | $$0.133258$$ |

We checked with the exact responses that these weights give a flat count equal to $$a_d\lvert\partial B\rvert$$, $$h_\Sigma=a_dc_d\ell^d$$ and $$h_0=0$$ identically in $$\ell$$, in $$d=3,\dots,8$$. With more than $$d+1$$ weights there are $$N-d$$ free parameters; "unique" means unique at minimal degree.

The theorem does not derive the Einstein equation. Its physical input is Theorem 1, that the part of the count at the area exponent is the local area of $$\partial B$$, including its curvature correction at fixed volume. At leading order in $$\ell_\rho$$ this is the locality of the continuum limit established by Barton et al. Given it, every weighting whose symbol vanishes at the $$d$$ other exponents satisfies (i) and (ii), whatever principle is used to choose it, and a weighting that keeps one of them fails. Keeping the exponent $$\alpha=2$$, for example, with $$Q(2)=q\neq0$$ and the other three zeros in four dimensions, gives $$h_0=\tfrac{256}{175}q\,\ell^2$$. What the weights achieve is that the continuum statement holds without corrections in $$\ell_\rho/\ell$$.

<div class="remark" markdown="1">
**Remark** (relation to the causal-set d'Alembertian). The four-dimensional d'Alembertian of Benincasa and Dowker [[BD]](#ref-bd) uses the same kind of kernel, a polynomial times $$e^{-v}$$, with layer coefficients $$(1,-9,16,-8)$$; by (2) its Mellin symbol vanishes at $$\alpha=\tfrac12,1,\tfrac32$$. Its zeros lie at and below the exponent it keeps and are spaced by $$2/d$$, and they remove the non-local terms of the continuum limit. The zeros of $$Q_d$$ lie above the kept exponent, are spaced by $$1/d$$, and remove the finite-density corrections.
</div>

<div class="remark" markdown="1">
**Remark** (two other readings of $$Q_d$$). Since $$\partial_x^d\,x^{d(s+1)}=d^d\prod_{j=1}^d(s+j/d)\,x^{ds}$$, the polynomial $$b(s)=\prod_{j=1}^d(s+j/d)$$ is the Bernstein–Sato polynomial of $$x^d$$, and $$Q_d(\alpha)=b(\alpha_A-\alpha)/b(0)$$. With $$t=d+2-d\alpha$$, $$Q_d=\binom{t+d}{d}$$ is the Ehrhart polynomial of the standard $$d$$-simplex; its zeros $$t=-1,\dots,-d$$ are, by Ehrhart reciprocity, the dilates of the simplex without interior lattice points. These are identities and add no constraint.
</div>

## 6. Signs, minimal degree and the cost of cancellation

<div class="proposition" markdown="1">
**Proposition 2** (no nonnegative weighting). If $$\phi\ge0$$ and $$\phi\not\equiv0$$, in particular if every $$w_m\ge0$$, then $$Q(\alpha)>0$$ for all $$\alpha>0$$, and no zero required in Theorem 2 can occur. For $$3\le d\le8$$, moreover, $$h_0(\ell)>0$$ and $$h_\Sigma(\ell)>Q(\alpha_A)a_dc_d\ell^d$$ for every $$\ell$$, so that $$G_N<G$$ for every $$\lambda\ge0$$, at every size and for every population of sizes.
</div>

<div class="proof" markdown="1">
*Proof.* The first statement is (2). The second follows because $$h_0$$ and $$h_\Sigma-Q(\alpha_A)a_dc_d\ell^d$$ are sums of positive coefficients times positive values $$Q(\alpha_A+j/d)$$ (section 4).
</div>

An entropy that counts molecules, each contributing a positive amount, therefore always gives a weaker gravitational coupling than its own area law at finite density. Signed weights are required, and the number of sign changes is bounded below by the number of zeros. The following lemma is classical; its proof is short.

<div class="lemma" markdown="1">
**Lemma 3** (sign changes). Let $$\phi$$ be continuous on $$(0,\infty)$$ with finitely many sign changes, and suppose $$\int_0^\infty\phi(v)v^{\alpha_k-1}dv=0$$, with absolute convergence, for $$K$$ distinct real $$\alpha_k$$. Then $$\phi\equiv0$$ or $$\phi$$ changes sign at least $$K$$ times.
</div>

<div class="proof" markdown="1">
*Proof.* Suppose $$\phi$$ changes sign exactly at $$v_1<\dots<v_r$$ with $$r<K$$. A nonzero combination $$\sum_{k=1}^{r+1}c_kv^{\alpha_k-1}$$ has at most $$r$$ positive zeros counted with multiplicity (Descartes' rule for generalised polynomials), and the $$r$$ conditions that it vanish at the $$v_i$$ have a nonzero solution. That solution changes sign exactly at the $$v_i$$, so its product with $$\phi$$ has one sign. Its integral is zero by hypothesis, so $$\phi\equiv0$$.
</div>

<div class="corollary" markdown="1">
**Corollary 1.** (a) Weights with $$w_m=0$$ for $$m>N$$ make $$Q$$ vanish at no more than $$N$$ positive points, because $$\phi=e^{-v}p(v)$$ with $$p$$ of degree $$N$$ has at most $$N$$ positive zeros. (b) If $$Q$$ vanishes at $$N$$ positive points, the weights $$w_0,\dots,w_N$$ are all nonzero and strictly alternate in sign. (c) More generally, a kernel $$\sum_{i=1}^{q}p_i(v)e^{-\mu_iv}$$ with distinct $$\mu_i>0$$ and $$\deg p_i\le D_i$$ makes $$Q$$ vanish at no more than $$\sum_i(D_i+1)-1$$ points, and this number is attained.
</div>

<div class="proof" markdown="1">
*Proof.* (a) follows from Lemma 3. For (b), $$p(v)=\sum w_mv^m/m!$$ then has at least $$N$$ positive zeros, so by Descartes' rule its $$N+1$$ coefficients have at least $$N$$ sign changes, which forces every coefficient to be nonzero and the signs to alternate. For (c), a nonzero exponential polynomial of this form has at most $$\sum_i(D_i+1)-1$$ real zeros [Pólya–Szegő, Part V]; the bound is attained because that many linear conditions on $$\sum_i(D_i+1)$$ coefficients have a nonzero solution, which by the same count cannot also vanish at $$\alpha_A$$.
</div>

The cancellation is heavy. In four dimensions $$\phi_4(v)=e^{-v}\big(210-791v+634v^2-\tfrac{464}{3}v^3+\tfrac{32}{3}v^4\big)$$ changes sign at $$v\approx0.36,\ 1.58,\ 4.05,\ 8.5$$ and ranges between $$-41.8$$ and $$210$$. The area term is the moment $$\int\phi_4v^{1/2}dv/\Gamma(\tfrac32)=1$$, while $$\int\lvert\phi_4\rvert v^{1/2}dv/\Gamma(\tfrac32)=114.9$$. For the symbols $$Q=\binom{N+6-4\alpha}{N}$$, which cancel $$N$$ exponents, this ratio is $$929$$, $$7.2\times10^3$$ and $$4.0\times10^5$$ for $$N=6$$, $$8$$, $$12$$, about a factor $$2.8$$ per additional zero. The weights themselves grow at least geometrically: from $$Q(-j)=\sum_mw_m(-1)^m\binom jm$$ one gets $$\sum_m\lvert w_m\rvert\ge Q(-j)/2^j$$ for every $$j$$, and $$j=N$$ gives $$\sum_m\lvert w_m\rvert\ge\binom{5N+6}{N}2^{-N}$$, which grows like $$6.1^N$$. The variance of $$H_w$$ has not been computed. Its Poisson fluctuations are amplified by the same cancellation, so whether the weighted count is close to its mean in a single sprinkling is open.

## 7. Scope: gradients, orientation and non-linear order

### Gradients of the curvature

Theorem 2 concerns curvature that is uniform over the diamond. Gradients add exponents beyond $$\alpha_A+1$$. The class of Lemma 2 contains time-dependent examples: $$W=e^{d\,c_kt^k}$$ with $$k\ge2$$ is conformally flat, its slice $$t=0$$ is flat with vanishing extrinsic curvature, and for $$k=3$$ the scalar curvature vanishes on $$\Sigma$$ while its time derivative does not. At first order in $$c_k$$ the count contains $$\ell^1$$ and $$\ell^0$$ terms, at exponents $$2+(k-1)/d$$ and $$2+k/d$$ by (5). For $$k=3$$ the $$Q_d$$-weighted count is not zero but

$$
\frac{2511}{175\pi}\,c_3\quad(d=3),\qquad -\frac{3377\cdot3^{3/4}\Gamma(\frac34)}{700\,\pi^{3/4}}\,c_3\approx-5.71\,c_3\quad(d=4),
$$

in units $$\rho=1$$, while the area of $$\partial B$$ does not change. Theorem 2 therefore holds for sources without gradients, and a first time derivative of the scalar curvature already leaves a residual at linear order.

At the next order in gradients the exponents run up to $$\alpha_A+(d+2)/d$$. The symbol $$\binom{2d+4-d\alpha}{d+2}$$ removes them; in four dimensions its weights are $$(924,-7084,23668,-41416,39808,-19968,4096)$$. With these weights the $$k=3$$ and $$k=4$$ responses above vanish identically in $$d=3$$ and $$d=4$$. For static perturbations whose spatial Laplacian of the curvature is nonzero, the same weights have also been checked to leave only the area term in four dimensions; we do not reproduce that computation. Each order in gradients needs new weights. The first exponent a weighting does not cancel enters with an unsuppressed coefficient: $$Q_d(\alpha_A+(d+k)/d)=(-1)^d\binom{d+k-1}{d}$$, which is $$\pm1$$ for $$k=1$$.

### Orientation of the horizon sheet

Two null sheets pass through $$\partial B$$ below $$\Sigma$$. The one used so far, $$\partial I^-(o)$$, moves away from $$B$$ into the past. The other is the lower boundary of the diamond itself, $$\partial I^+(o')$$ with $$o'$$ the bottom tip; it contracts into the past and has a caustic at $$o'$$, a depth $$\ell$$ below $$\Sigma$$. Molecules are defined for it in the same way, with $$p$$ outside $$I^+(o')$$ and $$q$$ inside. Barton et al.'s horizons are of the first kind. In flat four-dimensional space, Lemma 2 applies to the second sheet with the bracket replaced by $$\mathrm{cap}(\ell-\tau+u)-\mathrm{cap}(\ell-\tau)$$, caps of negative radius being empty. For $$\tau<\ell$$ this is the bracket of the first sheet evaluated at $$-\ell$$, since $$\mathrm{cap}(r)=\omega r^3$$ is odd. Hence

$$
\langle H_w\rangle_{\rm lower}(\ell)=\langle H_w\rangle(-\ell)-\mathcal D_w(\ell),
$$

where the second sheet's count differs from the polynomial (9) evaluated at $$-\ell$$ by a defect $$\mathcal D_w$$ that comes from molecules with $$p$$ below the caustic. For each type,

$$
\mathcal D_m(\ell)=\int_0^\infty d\delta\;\frac{V^m e^{-V}}{m!}\Big|_{V=\pi(\ell+\delta)^4/3}\;\frac{4\pi^2}{315}\,\delta^3\big(35(\ell+\delta)^4-\delta^4\big)>0,
$$

and Laplace's method gives $$\mathcal D_m(\ell)=\frac{27}{32\pi^2}\frac{(\pi\ell^4/3)^m}{m!}\ell^{-8}e^{-\pi\ell^4/3}\big(1+O(\ell^{-4})\big)$$; we checked the asymptotic form numerically to $$2\times10^{-4}$$ at $$\ell=12$$. Here $$\pi\ell^4/3$$ is the volume of the half of the diamond below $$\Sigma$$, which lies to the future of every $$p$$ below the caustic.

<div class="proposition" markdown="1">
**Proposition 3** (no exact weighting for the lower sheet). In flat four-dimensional space no finite nonzero weighting makes the count across the lower sheet proportional to $$\ell^2$$ for all $$\ell>0$$.
</div>

<div class="proof" markdown="1">
*Proof.* If it did, $$\mathcal D_w(\ell)$$ would equal a polynomial in $$\ell$$, and since $$\mathcal D_w$$ tends to zero faster than any power, the polynomial and $$\mathcal D_w$$ would vanish identically. But if $$M$$ is the largest $$m$$ with $$w_m\neq0$$, the asymptotic form shows $$\mathcal D_w(\ell)\sim w_M\mathcal D_M(\ell)\neq0$$ for large $$\ell$$.
</div>

For $$\ell\gtrsim2.2\,\ell_\rho$$ the defect is negligible, and because reflecting $$\ell$$ changes no exponent, $$Q_4$$ gives the area law and conditions (i) and (ii) on this sheet as well. At Planckian sizes it does not. With the weights of Theorem 2 the flat count across the lower sheet, divided by $$a_4\lvert\partial B\rvert$$, is $$1.113$$, $$0.959$$ and $$0.99998$$ at $$\ell=1$$, $$1.36$$ and $$1.9$$, and $$h_0$$ is $$-0.253$$, $$-0.364$$ and $$-8\times10^{-5}$$. Averaged over the link population of section 5, $$G_N/G=1.350$$, $$8.450$$ and $$-1.984$$ for $$\lambda=0$$, dust and radiation. The difference between the sheets is the caustic, not the direction of time: reversing time maps the first sheet, counted to the past of $$\Sigma$$, to the upper boundary of the diamond counted to the future, and the same weights are exact there.

### Beyond linear order

Theorem 1 is not restricted to linear order: in the class of Lemma 2 the term at the area exponent is the exact area of the fixed-volume ball to all orders in $$1/A^2$$ and in the derivatives of $$W$$, and with $$Q_d$$ every other term up to $$j=d$$ is removed. At second order in $$1/A^2$$ in $$d=3$$ the weighted count is

$$
a_3\lvert\partial B\rvert+\frac{1}{A^4}\Big(\frac{27\,\ell}{560\pi}+\frac{2\cdot3^{1/3}\Gamma(\frac13)}{225\,\pi^{4/3}}\Big)\ \text{(static universe)},\qquad
a_3\lvert\partial B\rvert-\frac{1}{A^4}\Big(\frac{17307\,\ell}{2800\pi}+\frac{4109\cdot3^{1/3}\Gamma(\frac13)}{450\,\pi^{4/3}}\Big)\ \text{(de Sitter)},
$$

with residuals at $$j=4,5$$ only. The residuals are suppressed by $$(\ell_\rho/\ell)^{d+1}$$ relative to the area term but have large coefficients: in de Sitter the residual exceeds the second-order part of $$a_3\lvert\partial B\rvert$$, which is $$-0.0107\,\ell^5/A^4$$, for $$\ell<4.3\,\ell_\rho$$.

This does not by itself give the non-linear Einstein equation. When $$W'(0)\neq0$$ the slice $$t=0$$ has extrinsic curvature $$K_{ij}\propto h_{ij}$$, the area deficit of $$B$$ follows $$R^\Sigma$$, and by the Gauss equation $$R^\Sigma=2G_{uu}-(K^2-K_{ij}K^{ij})\neq2G_{uu}$$. Jacobson's argument uses, for each point $$x$$ and unit timelike $$u$$, geodesic balls in the slice $$\Sigma_u$$ generated by geodesics orthogonal to $$u$$ at $$x$$, whose extrinsic curvature vanishes at $$x$$. In an expanding universe these slices are not of the form $$t=$$ const, and Lemma 2 does not reach them. The non-linear statement is therefore conditional on the following property.

(J) For every $$x$$ and $$u$$, and for geodesic balls $$B\subset\Sigma_u$$ with $$\ell_\rho\ll\ell\ll L$$, $$L$$ the curvature length, the term of the count at the area exponent is $$a_d$$ times the area of $$\partial B$$.

If (J) holds, the count supplies $$\delta S=\eta\,\delta\lvert\partial B\rvert_V$$ with $$\eta=s_0Q(\alpha_A)a_d\rho^{(d-2)/d}$$, where at leading order in $$\ell/L$$, $$\delta\lvert\partial B\rvert_V=-\Omega_{d-2}\ell^dG_{uu}/(d^2-1)$$ for all $$u$$. Jacobson's argument then gives $$G_{ab}+\Lambda g_{ab}=8\pi G\,T_{ab}$$ with $$G=1/(4\eta)$$, including its non-linear terms. Two facts support (J). The leading term of the count is local [[Barton et al.]](#ref-barton). And in flat space the coefficient of a corner, a null plane meeting a spacelike hyperplane in a codimension-two plane, does not depend on the tilt of the hyperplane: a boost along the null plane maps one tilted hyperplane through the corner to any other and preserves the sprinkling, the order and all volumes. Curvature, the extrinsic curvature of $$\Sigma_u$$ and that of $$\partial B$$ enter at $$j\ge1$$ by (5), where $$Q_d$$ removes them up to $$j=d$$. Property (J) has not been verified by computing the count on a curved slice $$\Sigma_u$$. Even if it holds, the count supplies only the entropy term. The first law for the matter, the conformal Killing vector of the diamond and the vacuum-state hypothesis remain Jacobson's assumptions, and the temperature is not addressed.

## 8. Exactness to all orders requires slowly decaying kernels

A weighting of finite degree is exact only to finite order in gradients (Corollary 1). We now ask what kernel could be exact to all orders. Every order in gradients adds new exponents. In the time-dependent family of section 7 the $$\ell^0$$ term of the $$c_k$$-response sits at $$2+k/d$$, and its coefficient is nonzero for every $$k=2,\dots,8$$ in $$d=3$$ and $$d=4$$; together with section 4 this covers $$\alpha_A+j/d$$ for $$1\le j\le10$$ in four dimensions. We assume this continues.

(H) For every $$j\ge1$$ some linear response in the class of Lemma 2 has a nonzero type-0 coefficient at $$\alpha_A+j/d$$.

Under (H), exactness to all orders means $$Q(\alpha_A+j/d)=0$$ for all $$j\ge1$$ with $$Q(\alpha_A)\neq0$$. For the kernels of section 1, $$\phi(v)=e^{-v}\sum w_mv^m/m!$$. We state the next result for general kernels $$\phi$$ on $$(0,\infty)$$, with $$Q$$ defined by (2), and use the length variable $$x=v^{1/d}$$, which is of the order of the proper-time extent of $$J^+(p)\cap I^-(\Sigma)$$ in units of $$\ell_\rho$$.

<div class="theorem" markdown="1">
**Theorem 3** (decay threshold). Let $$\phi$$ be bounded on $$(0,1)$$ with $$\lvert\phi(v)\rvert\le C\exp(-c\,v^{1/(2d)})$$ for $$v\ge1$$ and some $$c,C>0$$. If $$Q(\alpha_A+j/d)=0$$ for all $$j\ge j_0$$, then $$\phi=0$$ almost everywhere. Conversely, for every $$\gamma<1/(2d)$$ there is a continuous $$\phi$$ with $$\lvert\phi(v)\rvert\le C\exp(-c\,v^{\gamma})$$ such that $$Q(\alpha_A+j/d)=0$$ for all $$j\ge1$$ and $$Q(\alpha_A)\neq0$$.
</div>

In the variable $$x$$ the conditions are vanishing moments, and the threshold $$e^{-c\sqrt x}$$ is the classical boundary of determinacy for the Stieltjes moment problem on the half-line.

<div class="proof" markdown="1">
*Proof.* Put $$h(x)=\phi(x^d)$$. By (2), $$\Gamma(\alpha)Q(\alpha)=d\int_0^\infty h(x)\,x^{d\alpha-1}dx$$, and $$d\alpha-1=d+1+j$$ at $$\alpha=\alpha_A+j/d$$. So the hypothesis says $$\int_0^\infty h(x)x^kdx=0$$ for all integers $$k\ge k_0=d+1+j_0$$, and $$\lvert h(x)\rvert\le Ce^{-c\sqrt x}$$ for $$x\ge1$$. Let $$H(x)=x^{k_0}h(x)$$, so that all moments of $$H$$ vanish, and define the even function $$g(y)=\lvert y\rvert H(y^2)$$ on $$\mathbb R$$. Substituting $$x=y^2$$ gives $$\int_{\mathbb R}g(y)y^{2k}dy=\int_0^\infty H(x)x^kdx=0$$, and the odd moments vanish by symmetry. Since $$\lvert g(y)\rvert\le C'e^{-c\lvert y\rvert/2}$$, its Fourier transform is analytic in the strip $$\lvert{\rm Im}\,\xi\rvert<c/2$$, and its Taylor coefficients at $$\xi=0$$ are the moments of $$g$$, all zero. Hence the transform vanishes, $$g=0$$ and $$\phi=0$$ almost everywhere.

For the converse let $$0<\beta<\tfrac12$$ and $$s(x)=e^{-x^\beta\cos\pi\beta}\sin(x^\beta\sin\pi\beta)$$. Substituting $$t=x^\beta$$ and using $$\int_0^\infty t^{a-1}e^{-t\cos\theta}\sin(t\sin\theta)\,dt=\Gamma(a)\sin(a\theta)$$ for $$\lvert\theta\rvert<\pi/2$$ gives $$\int_0^\infty s(x)x^{z-1}dx=\Gamma(z/\beta)\sin(\pi z)/\beta$$, which vanishes at every positive integer $$z$$ (Stieltjes' example). Let $$g(x)=x^{-(d+2)}\int_x^\infty t^{d+1}s(t)\,dt$$. By Fubini, $$\int_0^\infty g(x)x^{z-1}dx=\Gamma(z/\beta)\sin(\pi z)/(\beta(z-d-2))$$. This vanishes at $$z=d+2+j$$ for $$j\ge1$$ and equals $$(-1)^d\pi\Gamma((d+2)/\beta)/\beta\neq0$$ at $$z=d+2$$. The function $$g$$ is continuous on $$[0,\infty)$$ (the integral from 0 to $$\infty$$ of $$t^{d+1}s$$ vanishes, so $$g(x)=O(x^{\beta})$$ near 0) and decays like $$x^{-\beta}e^{-x^\beta\cos\pi\beta}$$. Then $$\phi(v)=g(v^{1/d})/d$$ has $$\Gamma(\alpha)Q(\alpha)=\int_0^\infty g(x)x^{d\alpha-1}dx$$, so it has the required zeros and $$Q(\alpha_A)\neq0$$, and it decays like $$\exp(-c\,v^{\beta/d})$$. Take $$\beta=d\gamma$$.
</div>

The theorem includes exponentially decaying kernels, kernels supported on a bounded range of $$v$$, and weights with $$\lvert w_m\rvert\le Cr^m$$ for some $$r<1$$, which give $$\lvert\phi(v)\rvert\le Ce^{-(1-r)v}$$; none of these is exact to all orders. (For exponentially decaying kernels the same conclusion follows from Carlson's theorem applied to $$Q$$.) Lemma 3 adds a second, independent condition: an exact kernel changes sign infinitely often. Kernels with finitely many sign changes, which include every kernel definable in an o-minimal structure, are therefore exact only to finite order. Decay and oscillation are separate requirements. A positive kernel can have a heavy tail, and $$e^{-v}\sin v$$ oscillates but decays exponentially.

Theorem 3 describes how non-local an exact count must be. A kernel at the threshold weights molecules with $$x$$ up to $$(30/c)^2$$ by more than $$e^{-30}$$ of their typical weight. For the kernel of the proof with $$\beta=0.41$$, weights fall below $$e^{-30}$$ only for $$x\gtrsim9\times10^4$$, so molecules whose lower element lies some $$10^5$$ discreteness lengths below $$\Sigma$$ still contribute. The kernels of the proof are functions of $$v$$, not of $$m$$. A weighting by $$m$$ gives $$\phi(v)=e^{-v}\times$$ (an entire function), and whether a kernel of that form can reach the threshold is open.

Two simpler escapes are also closed. Because the conditions fix $$Q$$ on a discrete set of points that geometry determines, they cannot be relaxed to vanishing on an interval of $$\alpha$$. And a kernel supported in a bounded range of $$v$$ decays faster than the threshold. The one-sided support of $$\phi$$, which comes from causality, is what makes the half-line threshold $$e^{-c\sqrt x}$$ apply rather than the two-sided threshold $$e^{-cx}$$.

## 9. Discussion

The results of sections 4–6 are theorems about the expected count in a model: Poisson sprinkling, molecules defined relative to a slice and a caustic-free horizon sheet, curvature at linear order and uniform over the diamond. Within that model the area term and its curvature response at fixed volume can be made exact at every size by signed weights, and the weights are unique at minimal degree in every dimension. The Newton constant that results is the one fixed by the area law, but this follows from the locality of the leading term and holds for any weighting that removes the subleading ones. The Einstein equation itself is an input through Jacobson's argument, linearised unless property (J) holds.

The model leaves several questions open. The variance of the signed count has not been computed, and the cancellation in section 6 suggests it is large. The responses to time-dependent and Weyl-type gradient sources have been computed only in the families described in section 7, and hypothesis (H) is verified only up to $$j=10$$ in four dimensions. Property (J) requires the count on curved slices whose extrinsic curvature vanishes at a point, which has not been done. On the lower null boundary of the diamond no finite weighting is exact at Planckian sizes, so the choice of horizon sheet matters exactly where the weights are meant to act. It is not known whether a weighting by $$m$$ alone, rather than a kernel in the continuum volume, can be exact to all orders in gradients. Finally, the count fixes only the entropy; the temperature and the heat flux in Jacobson's argument are not derived from the causal set here.

## References

1. <a id="ref-barton"></a>Barton, Counsell, Dowker, Gould, Jubb, Taylor, “Horizon molecules in causal set theory,” [arXiv:1909.08620](https://arxiv.org/abs/1909.08620).
2. <a id="ref-dou-sorkin"></a>Dou, Sorkin, “Black hole entropy as causal links,” [arXiv:gr-qc/0302009](https://arxiv.org/abs/gr-qc/0302009).
3. <a id="ref-jacobson15"></a>Jacobson, “Entanglement equilibrium and the Einstein equation,” [arXiv:1505.04753](https://arxiv.org/abs/1505.04753).
4. <a id="ref-bd"></a>Benincasa, Dowker, “The scalar curvature of a causal set,” [arXiv:1001.2725](https://arxiv.org/abs/1001.2725).
5. <a id="ref-dg"></a>Dowker, Glaser, “Causal set d'Alembertians for various dimensions,” [arXiv:1305.2588](https://arxiv.org/abs/1305.2588).
6. <a id="ref-glaser"></a>Glaser, “A closed form expression for the causal set d'Alembertian,” [arXiv:1311.1701](https://arxiv.org/abs/1311.1701).
7. <a id="ref-ass"></a>Aslanbeigi, Saravani, Sorkin, “Generalized causal set d'Alembertians,” [arXiv:1403.1622](https://arxiv.org/abs/1403.1622).
