---
title: Structures that leave the cosmological constant and the dark-matter mass undetermined
date: 2026-10-29
category: dark-sector
order: 3
description: For the vacuum energy of a free scalar on a flat expanding background, horizon quantisations of Kodama and Verlinde type, a group-field-theory condensate, a clock Hamiltonian, causal-set counts and relations between dark-matter and baryon number densities, the defining conditions leave the cosmological constant or the dark-matter mass undetermined.
status: A lemma and propositions proved under stated hypotheses. The de Sitter energy density is derived from the Bunch–Davies closed form for a conformally coupled scalar, and the sixth-order estimate on flat LambdaCDM is for that coupling. The structure of the adiabatic expansion on general backgrounds and the horizon conditions of the Kodama treatments are taken from the literature. The group-field-theory, clock and gauge-coupling statements are derivations within the stated models.
excludes: Fixing the cosmological constant from the vacuum energy of a free scalar on flat FRW in an adiabatic state, with one renormalisation representative (interacting fields and other curvature couplings only by citation); from integrality conditions homogeneous in the de Sitter entropy, as in the Kodama and Verlinde treatments; from the coefficient of the logarithm in a causal-set count; and, within the model assumptions of Proposition 6, a positive value from the induced energy of a clock in a passive state. Fixing the dark-matter to baryon density ratio from relations between number densities alone.
---

The vacuum energy density of the cosmological constant, $$\rho_\Lambda=\Lambda/8\pi G\approx(2.24\ {\rm meV})^4$$, and the dark-matter to baryon density ratio $$\Omega_c/\Omega_b=5.36$$ [[Planck VI]](#ref-planck6) are not derived from the Standard Model. We ask whether several proposed structures fix either number, and find that those examined do not: a transformation that preserves the defining conditions moves the number, or the conditions admit a lattice of values, or the number depends on a choice that the structure does not make. For a free massive scalar on a flat expanding background, the slowly varying part of the vacuum energy is a free renormalisation constant, and the term that survives decoupling is at most about $$2\times10^{-134}\rho_\Lambda$$ today for a field of mass at least $$H(z=1100)$$. The Kodama-state and Verlinde-count conditions on the horizon admit an arithmetic progression of values of $$\Lambda$$, or every value. A construction that is unchanged when $$m_X$$ is rescaled leaves the dark-matter mass free. Section 6 lists the routes that remain.

The freedom in the renormalised stress tensor is due to Wald and to Hollands and Wald [[Wald]](#ref-wald), [[HW1]](#ref-hw1), [[HW2]](#ref-hw2), adiabatic subtraction to Parker and Fulling [[PF]](#ref-pf), and the de Sitter energy density of a massive scalar to Bunch and Davies [[BD]](#ref-bd). The Kodama state and the horizon counts at finite level come from Randono, Witten and Engle, Noui, Perez and Pranzetti, and the group-field-theory background from Gielen and Polaczek, de Cesare, Pithis and Sakellariadou, and Granata and Oriti; each is described where it is used. This page adds the orbit lemma as a common formulation, the sixth-order estimate on a flat $$\Lambda$$CDM history, and Propositions 3 to 8. How these results sit among other approaches is on the [landscape page](/quantum-gravity/landscape/).

## 1. The orbit lemma

A structure often produces conditions of the form "a functional $$F$$ vanishes", "$$F$$ is extremal" or "two functionals of the same weight are equal". Suppose a group $$\mathcal T$$ acts on the space $$X$$ of data and the functional transforms as $$F(h\cdot\vartheta)=\alpha(h)F(\vartheta)$$ with $$\alpha(h)\ne0$$. Suppose the number sought is a function $$q:X\to\mathbb R$$ that transforms either multiplicatively, $$q(h\cdot\vartheta)=\psi(h)\,q(\vartheta)$$, or additively, $$q(h\cdot\vartheta)=q(\vartheta)+\tau(h)$$.

<div class="lemma" markdown="1">
**Lemma 1** (orbit lemma). Let $$Z\subset X$$ be the zero set of $$F$$, or the set of critical points of $$F$$ (for differentiable $$F$$ and an action by diffeomorphisms), or the set of extrema of $$F$$ (when $$\alpha>0$$), or the set where two functionals $$F_1,F_2$$ of the same weight $$\alpha$$ satisfy $$F_1=cF_2$$. Then $$Z$$ is invariant under $$\mathcal T$$. In the multiplicative case $$q(Z)\subset\mathbb R$$ is invariant under multiplication by $$\psi(\mathcal T)$$, and in the additive case under translation by $$\tau(\mathcal T)$$. In particular: if $$\psi(\mathcal T)\supseteq\{e^t:t\in\mathbb R\}$$ and $$Z\cap\{q>0\}\ne\emptyset$$, then $$q(Z)\supseteq(0,\infty)$$; if $$\psi(\mathcal T)=\{\psi_0^n:n\in\mathbb Z\}$$, then $$q(Z)$$ is a union of geometric progressions of ratio $$\psi_0$$; if $$\psi(\mathcal T)=\{\pm1\}$$, then $$q(Z)$$ is symmetric under a change of sign; if $$\tau(\mathcal T)=\mathbb R$$ and $$Z\ne\emptyset$$, then $$q(Z)=\mathbb R$$.
</div>

<div class="proof" markdown="1">
*Proof.* If $$F(\vartheta)=0$$ then $$F(h\cdot\vartheta)=\alpha(h)F(\vartheta)=0$$. If $$\vartheta$$ is a critical point, the chain rule gives $$dF_{h\cdot\vartheta}\circ dh_\vartheta=\alpha(h)\,dF_\vartheta=0$$, and $$dh_\vartheta$$ is invertible. For extrema, $$\alpha>0$$ preserves the order of values. For $$F_1=cF_2$$, $$F_1(h\cdot\vartheta)=\alpha F_1(\vartheta)=c\alpha F_2(\vartheta)=cF_2(h\cdot\vartheta)$$. Hence $$\vartheta\in Z$$ implies $$h\cdot\vartheta\in Z$$, and the transformation law of $$q$$ shows that $$\psi(h)q(\vartheta)$$, respectively $$q(\vartheta)+\tau(h)$$, lies in $$q(Z)$$. $$\square$$
</div>

The lemma is used in sections 2, 3 and 5, where it removes every condition built from $$F$$ alone. The other statements of sections 4 and 5 are of different kinds: an arithmetic statement about integrality conditions (Proposition 3), the vanishing of a deformation on a horizon of small area (Proposition 4), the dependence of a model on the choice of clock (Proposition 5), a sign (Proposition 6) and a sensitivity bound (Proposition 8). The lemma says nothing about structures whose admissible data are discrete and rigid, such as a data set selected by an extremal property among a discrete family that no symmetry of the problem moves; such data have no orbit. It does not apply to conditions that are not built from a covariant functional, such as the integrality conditions $$a s+b\in\mathbb Z$$ with fixed $$b$$ that Proposition 3 treats.

## 2. Counting statistics that cannot see the discreteness scale

A causal set is a locally finite partially ordered set, used as a discrete model of spacetime. A sprinkling of density $$\varrho$$ into a Lorentzian region $$D$$ with metric $$g$$ is a Poisson-distributed random point set, ordered by the causal relation of $$g$$. A functional of the order, such as the number of pairs of points related by a given number of intermediate points, has an expectation value $$\Xi(\varrho,g,D)$$, and $$\ell\propto\varrho^{-1/4}$$ is the discreteness length. The coefficient of $$\ln\varrho$$ in the large-$$\varrho$$ expansion of $$\Xi$$ is the first term that does not change when $$\varrho$$ is rescaled, which makes it a candidate for a condition that could fix $$\Lambda\ell^2$$. Proposition 1 shows that it cannot.

<div class="proposition" markdown="1">
**Proposition 1** (homothety). For a constant $$\zeta>0$$, $$\Xi(\varrho,\zeta^2g,D)=\Xi(\zeta^4\varrho,g,D)$$ for every functional of the causal order. Suppose $$\Xi=\sum_na_n(g)\varrho^{n/4}+a_{\ln}(g)\ln\varrho+a_0(g)+o(1)$$ as $$\varrho\to\infty$$. Then $$a_n(\zeta^2g)=\zeta^na_n(g)$$, $$a_{\ln}(\zeta^2g)=a_{\ln}(g)$$ and $$a_0(\zeta^2g)=a_0(g)+4a_{\ln}(g)\ln\zeta$$. In particular, $$a_{\ln}$$ depends on $$(g,D)$$ only through the dimensionless shape of $$D$$; for a diamond of proper height $$\tau$$ in de Sitter space of Hubble rate $$H$$ it is a function of $$H\tau$$ alone.
</div>

<div class="proof" markdown="1">
*Proof.* A constant rescaling of the metric does not change the causal relations, and it multiplies the volume measure by $$\zeta^4$$. The Poisson process with intensity $$\varrho\,dV_{\zeta^2g}=\varrho\zeta^4dV_g$$ is the process of density $$\zeta^4\varrho$$ for $$g$$, so the sprinkled causal sets have the same law and the first claim follows. Insert the expansion on both sides, use $$\ln(\zeta^4\varrho)=\ln\varrho+4\ln\zeta$$, and compare coefficients of $$\varrho^{n/4}$$ and $$\ln\varrho$$; an asymptotic expansion in this scale is unique. The statement about the shape follows because $$\zeta$$ is arbitrary. $$\square$$
</div>

A condition that sets $$a_{\ln}$$ equal to a number, or equates two such coefficients computed for different functionals or regions of the same geometry, is therefore an equation $$F(H\tau)=0$$ for the dimensionless shape of the region. It contains neither $$\varrho$$ nor the Planck length, so it cannot fix $$\Lambda\ell^2$$. If the height is tied to the horizon, $$\tau=\kappa_0/H$$ with a number $$\kappa_0$$, the condition reads $$F(\kappa_0)=0$$, which holds for every $$\Lambda$$ or for none. If it is tied to the discreteness scale, $$\tau=n\ell$$, a solution has $$\Lambda\ell^2=O(n^{-2})$$, which is of order one for $$n$$ of order one and is $$10^{-122}$$ only for $$n\sim10^{61}$$. The scale of the cosmological constant would have to come from a second relation containing $$\ell$$ explicitly. Other results on what causal-set kinematics can and cannot fix are on the page on [pure numbers from causal-set kinematics](/quantum-gravity/no-number-from-kinematics/). The discussion of this section is schematic: it applies to any condition of the stated form and does not examine a particular published construction.

## 3. The cosmological constant as a coordinate on the counterterm space

Consider a free scalar field of mass $$m$$ on a spatially flat Friedmann–Robertson–Walker (FRW) background with scale factor $$a(t)$$ and Hubble rate $$H=\dot a/a$$, coupled to the curvature with coupling $$\xi$$, in a state of adiabatic-vacuum type. The energy density has an ultraviolet divergence, and the renormalised value $$\rho_{\rm ren}$$ is defined only up to the following freedom.

<div class="proposition" markdown="1">
**Proposition 2** (renormalisation ambiguity). Any two prescriptions for the renormalised stress tensor that obey the axioms of Wald (locality, covariance, conservation, standard normalisation) differ by a finite linear combination of local, conserved tensors built from the metric and $$m$$, of dimension at most four: $$m^4g_{ab}$$, $$m^2G_{ab}$$ (with $$G_{ab}$$ the Einstein tensor) and the two independent conserved tensors that are quadratic in the curvature, with coefficients that may depend on $$\ln(m/M_{\rm R})$$ for a renormalisation scale $$M_{\rm R}$$ [[Wald]](#ref-wald), [[HW1]](#ref-hw1), [[HW2]](#ref-hw2). Call this span the counterterm space $$\mathcal C$$. In a flat FRW background the energy-density components of these tensors are $$m^4$$, $$m^2H^2$$ and two combinations of $$H^4$$, $$H^2\dot H$$, $$\dot H^2$$ and $$H\ddot H$$, each with a coefficient that can be changed freely. Let $$\mathcal F$$ be the set of renormalised energy densities obtained from admissible prescriptions. Then $$\mathcal C$$ acts on $$\mathcal F$$ by addition, and a property of $$\rho_{\rm ren}$$ that is invariant under $$\rho_{\rm ren}\to\rho_{\rm ren}+c$$ for all $$c\in\mathcal C$$ is a property of the class of $$\rho_{\rm ren}$$ in $$\mathcal F/\mathcal C$$. Being "Lambda-like", that is, constant in time to within a given tolerance, is not such a property: adding $$c_0m^4$$ with $$\vert c_0\vert $$ large makes any class Lambda-like.
</div>

The last statement is immediate, since $$m^4$$ does not depend on time. It is the additive case of Lemma 1: $$\mathcal T=\mathcal C$$ acts on $$\mathcal F$$ by translation, $$q$$ is the coefficient $$c_0$$ of $$m^4$$ relative to a fixed reference prescription, and $$\tau(c)=c_0$$. Any condition built from a functional that is invariant under these translations is satisfied by representatives with every value of $$c_0$$. A statement about the value of $$\rho_\Lambda$$ in this setting must therefore be a statement about a choice of representative in each class.

A natural choice exists. Among the representatives whose counterterm coefficients are numbers independent of $$m$$, exactly one has $$\rho_{\rm ren}\to0$$ as $$m\to\infty$$ at fixed geometry: a nonzero coefficient of $$m^4$$ or $$m^2H^2$$ diverges in that limit, and a nonzero coefficient of an $$H^4$$-type combination tends to a nonzero limit on histories where that combination does not vanish. We call it the decoupled representative. Adiabatic subtraction through fourth order selects it [[PF]](#ref-pf): the mode energy per comoving wavenumber $$k$$ is expanded in the adiabatic parameter $$\epsilon\sim H/\omega_k$$, with $$\omega_k=\sqrt{k^2/a^2+m^2}$$, and the terms through $$\epsilon^4$$ are subtracted before the $$k$$-integral. The subtracted terms are local in the metric and of dimension at most four, so the result differs from any other prescription by an element of $$\mathcal C$$.

<div class="remark" markdown="1">
**Remark 1** (structure of the adiabatic expansion; cited inputs). Take the decoupled representative, an adiabatic vacuum of order $$N_{\rm ad}\ge6$$, and a history with $$H(t)\le H_{\max}\ll m$$. Then $$\rho_{\rm ren}\sim\sum_{n\ge3}\rho_{2n}$$ through adiabatic order $$N_{\rm ad}$$, where $$\rho_{2n}$$ is a local polynomial in $$a^{-1}d^ja/dt^j$$ with $$2n$$ time derivatives in total, divided by $$m^{2n-4}$$. No term of lower derivative order than six appears, and there is no term of the form $$m^4\ln(m/H)$$.

The argument rests on two inputs from the literature on adiabatic regularisation [[PF]](#ref-pf), [[FNS]](#ref-fns), which we cite for general $$\xi$$ and $$a(t)$$: the per-mode energy of an adiabatic vacuum of order $$N_{\rm ad}$$ equals its adiabatic expansion through that order, and the adiabatic term of each order is a local polynomial in the scale factor and its derivatives. The order-$$2n$$ term falls as $$\omega_k^{1-2n}$$ at large $$k$$, so its integral with measure $$k^2dk$$ diverges for $$n\le2$$ (quartically, quadratically and logarithmically for $$n=0,1,2$$, for general $$\xi$$) and converges for $$n\ge3$$. After the subtraction only $$n\ge3$$ remains, and the remainder of the expansion is controlled by $$\epsilon\le H_{\max}/m$$. The $$k$$-integral of a term with $$2n$$ derivatives scales as $$m^{4-2n}$$ by dimensional analysis, and no logarithm $$\ln(m/M_{\rm R})$$ occurs because the subtraction acts on the integrand and involves no renormalisation scale. For interacting fields the local ambiguity is as in Proposition 2 with coefficients that depend on the couplings, and the decoupling statement is taken from the cited work.
</div>

The following theorem confirms the remark in the one background where the energy density is known in closed form. It also gives the coefficient of the leading term.

<div class="theorem" markdown="1">
**Theorem 1** (energy density in de Sitter space). Let $$\phi$$ be a free scalar field of mass $$m$$ with curvature coupling $$\xi=1/6$$ in the Bunch–Davies vacuum of de Sitter space with Hubble rate $$H$$, and let $$\rho_{\rm BD}$$ be the renormalised energy density (1) of [[BD]](#ref-bd). Put $$t=H^2/m^2$$. Then, as $$t\to0$$,

(a) $$\rho_{\rm BD}=-\dfrac{H^6}{2520\pi^2m^2}\Big(1+\dfrac34t+\dfrac{12}{11}t^2+O(t^3)\Big)$$, with no terms proportional to $$m^4\ln(m/H)$$, $$m^4$$, $$m^2H^2$$ or $$H^4$$;

(b) $$\rho_{\rm BD}$$ is the decoupled representative;

(c) the sixth-order term of the adiabatic expansion of Remark 1, evaluated in de Sitter space, equals $$-H^6/(2520\pi^2m^2)$$, the leading term in (a).
</div>

<div class="proof" markdown="1">
*Proof.* (a) For $$\xi=1/6$$, the energy density in the Bunch–Davies vacuum is

$$
\rho_{\rm BD}=\frac{1}{64\pi^2}\Big[m^4\big(\psi(\tfrac32+\nu)+\psi(\tfrac32-\nu)-\ln\tfrac{m^2}{H^2}\big)-\tfrac23m^2H^2+\tfrac1{15}H^4\Big],\quad \nu^2=\tfrac14-\tfrac{m^2}{H^2}, \tag{1}
$$

where $$\psi$$ is the digamma function. This is the point-splitting result of [[BD]](#ref-bd) for conformal coupling at $$R=12H^2$$. For $$m>H/2$$ put $$\nu=iy$$ with $$y^2=m^2/H^2-\tfrac14$$. The two digamma terms are complex conjugates, so the combination in parentheses is $$2\,\mathrm{Re}\,\psi(\tfrac32+iy)-\ln(m^2/H^2)$$. The recursion $$\psi(z+1)=\psi(z)+1/z$$ gives $$\mathrm{Re}\,\psi(\tfrac32+iy)=\mathrm{Re}\,\psi(\tfrac12+iy)+\tfrac12t$$, since $$\mathrm{Re}\,(\tfrac12+iy)^{-1}=\tfrac12/(y^2+\tfrac14)=\tfrac12t$$. The asymptotic series of the digamma function,

$$\mathrm{Re}\,\psi(\tfrac12+iy)\sim\ln y+\sum_{k\ge1}(-1)^k\Big(1-2^{1-2k}\Big)\frac{B_{2k}}{2k\,y^{2k}},$$

with Bernoulli numbers $$B_{2k}$$, combined with $$y^2=1/t-\tfrac14$$ and $$2\ln y-\ln(m^2/H^2)=\ln(1-t/4)$$, gives

$$
2\,\mathrm{Re}\,\psi(\tfrac32+iy)-\ln\frac{m^2}{H^2}\sim\frac23t-\frac1{15}t^2-\frac8{315}t^3-\frac2{105}t^4-\frac{32}{1155}t^5-\dots \tag{2}
$$

Multiplying by $$m^4=H^4t^{-2}$$, the first two terms of (2) are $$\tfrac23m^2H^2-\tfrac1{15}H^4$$ and cancel exactly against the two explicit terms in (1). What remains in the brackets is $$-H^4(\tfrac8{315}t+\tfrac2{105}t^2+\tfrac{32}{1155}t^3+\dots)$$. Division by $$64\pi^2$$ gives (a), using $$64\cdot315/8=2520$$ and the ratios $$\tfrac2{105}/\tfrac8{315}=\tfrac34$$ and $$\tfrac{32}{1155}/\tfrac8{315}=\tfrac{12}{11}$$.

(b) By Proposition 2, $$\rho_{\rm BD}$$ and the decoupled representative differ by an element of $$\mathcal C$$, which in de Sitter space has the energy density $$c_0m^4+c_1m^2H^2+c_2H^4$$ with coefficients that may depend on $$m$$. By (a) and Remark 1, both have expansions in powers of $$H$$ at fixed $$m$$ that begin at $$H^6$$. Their difference is therefore a polynomial in $$H^2$$ of degree two that is $$O(H^6)$$ as $$H\to0$$, so it vanishes.

(c) For $$\xi=1/6$$ the mode function $$\chi_k=a\phi_k$$ obeys $$\chi_k''+(k^2+U)\chi_k=0$$ in conformal time $$\eta$$, with $$U=m^2a^2$$ and primes denoting $$d/d\eta$$, and the energy density is $$\rho=(4\pi^2a^4)^{-1}\int k^2dk\,(\vert \chi_k'\vert^2+\omega^2\vert \chi_k\vert^2)$$ with $$\omega^2=k^2+U$$, up to a spatial total derivative. In the adiabatic expansion $$\vert \chi_k\vert^2=1/2W$$ and $$\vert \chi_k'\vert^2=W/2+W'^2/8W^3$$, where $$W^2=\omega^2-W''/2W+3W'^2/4W^2$$ is solved order by order in the number of time derivatives. Taking the terms with six derivatives and integrating over $$k$$ gives, with $$U_j=d^jU/d\eta^j$$ and $$\rho_6=I_6/4\pi^2a^4$$,

$$
\begin{aligned}
I_6={}&\frac{U_1U_5}{3360\,U^3}-\frac{U_2U_4}{3360\,U^3}+\frac{U_3^2}{6720\,U^3}-\frac{U_1^2U_4}{560\,U^4}-\frac{U_1U_2U_3}{336\,U^4}+\frac{U_2^3}{1008\,U^4}\\
&+\frac{13\,U_1^3U_3}{1680\,U^5}+\frac{9\,U_1^2U_2^2}{1120\,U^5}-\frac{U_1^4U_2}{48\,U^6}+\frac{5\,U_1^6}{576\,U^7}.
\end{aligned}\tag{3}
$$

In de Sitter space $$a=-1/H\eta$$, so $$U=m^2/H^2\eta^2$$, and (3) evaluates to $$\rho_6=-H^6/(2520\pi^2m^2)$$. $$\square$$
</div>

Part (c) is an independent route to the coefficient in (a): the exact closed form and the adiabatic expansion agree on the leading term and on its normalisation. Numerically, $$\rho_{\rm BD}$$ divided by the leading term of (a) is $$1.0076$$, $$1.00083$$ and $$1.000075$$ at $$m/H=10,30,100$$, as the factor $$1+\tfrac34t+\tfrac{12}{11}t^2$$ predicts. The series (2) is asymptotic. For $$m/H=3,5,7,9$$, the difference between the exact bracket and the series truncated at its smallest term is $$-0.43$$, $$-0.78$$, $$-1.1$$ and $$0.14$$ times $$m^4e^{-2\pi m/H}$$, so the remainder is of order $$m^4e^{-2\pi m/H}$$ with a prefactor of order one that depends on the truncation point.

### The sixth-order term on a ΛCDM history

Formula (3) applies to any flat FRW history once $$a(\eta)$$ is known. For a flat $$\Lambda$$CDM universe, $$a'^2=H_0^2(\Omega_ma+\Omega_r+\Omega_\Lambda a^4)$$ with $$a=1$$ today, which determines all $$U_j$$. We take $$H_0=67.36$$ km s$$^{-1}$$ Mpc$$^{-1}$$, $$\Omega_m=0.3153$$, $$\Omega_\Lambda=0.6847$$ [[Planck VI]](#ref-planck6) and $$\Omega_r=9.2\times10^{-5}$$. With $$M_P=2.435\times10^{27}$$ eV one has $$\hbar H_0=1.437\times10^{-33}$$ eV, $$\rho_\Lambda=3\Omega_\Lambda H_0^2M_P^2=2.51\times10^{-11}\ {\rm eV}^4$$ (fourth root $$2.24$$ meV) and $$\rho_\Lambda/H_0^4=5.9\times10^{120}$$. At $$z=1100$$, $$H/H_0=2.36\times10^4$$, so $$H(1100)=3.4\times10^{-29}$$ eV and $$H(1100)^4=5.2\times10^{-104}\rho_\Lambda$$.

Because a quantum field on a fixed background contains no Newton constant, every energy density it produces is $$H^4$$ times a function of $$m/H$$. Hence for a field with $$m\ge H(1100)$$, any contribution of the form $$H^4f(H/m)$$ with $$\vert f\vert\le C$$ satisfies $$\vert\rho\vert\le C\,H(1100)^4=5.2\times10^{-104}\,C\,\rho_\Lambda$$ on $$0\le z\le1100$$. For the sixth-order term, (3) gives

| quantity | value |
|---|---|
| $$\rho_6(z=0)\,m^2$$ | $$-5.6\times10^{-5}H_0^6$$ |
| $$\rho_6(z=0)$$ at $$m=H(1100)$$ | $$-1.7\times10^{-134}\rho_\Lambda$$ |
| $$\rho_6(z=1100)/\rho_6(z=0)$$ | $$8.5\times10^{25}$$ |

The value at $$z=0$$ lies within the regime of the expansion, since $$H_0/m=4\times10^{-5}$$ there, and for larger $$m$$ it falls as $$m^{-2}$$. The ratio does not depend on $$m$$, because it is the ratio of the same function of the history at two epochs; at $$z=1100$$ the sixth-order term is a term of the expansion only for $$m\gg H(1100)$$, and for $$m\sim H(1100)$$ the ratio is the ratio of formal terms. For comparison, the de Sitter expression $$H_0^6/(2520\pi^2m^2)$$ at $$m=H(1100)$$ has magnitude $$1.2\times10^{-134}\rho_\Lambda$$. This computation is for $$\xi=1/6$$. A term of definite degree $$d$$ in $$H$$ changes by a factor $$(2.36\times10^4)^d$$ over $$0\le z\le1100$$. An equation of state with $$\vert1+w\vert<10^{-2}$$ over that range allows a change by a factor of at most $$1101^{0.03}=1.23$$, which means $$d<0.021$$. The local terms of the expansion have $$d=6,8,\dots$$, and the sixth-order term changes by the factor in the table.

Since every energy density of the field is $$H^4$$ times a function of $$m/H$$, while the observed $$\rho_\Lambda$$ is $$5.9\times10^{120}H_0^4$$, an exponent argument follows. Suppose that $$\rho_\Lambda=m^aH_0^{4-a}$$ for the mass $$m$$ of a known particle. Then $$a=\ln(\rho_\Lambda/H_0^4)/\ln(m/H_0)$$, which is $$4.05$$ for $$m=1$$ meV, $$3.83$$ for $$50$$ meV and $$3.13$$ for the electron mass. The local terms of the adiabatic expansion are $$m^{4-2n}H^{2n}$$, that is $$a=4-2n\in\{4,2,0,-2,\dots\}$$: only even powers of $$H$$ occur. The exponents required lie between $$3.1$$ and $$4.05$$, and the only local term near them is $$a=4$$, the cosmological counterterm $$c_0m^4$$. Its coefficient would have to be $$c_0=\rho_\Lambda/m^4$$, which is $$25$$ for $$m=1$$ meV, $$4\times10^{-6}$$ for $$50$$ meV and $$3.7\times10^{-34}$$ for the electron. Among the local terms of a free massive field, the closeness of $$\rho_\Lambda^{1/4}=2.24$$ meV to neutrino mass scales is thus a statement about a coefficient in $$\mathcal C$$.

### Non-local terms and excitations

The adiabatic expansion is local, so the question arises whether a non-local response could supply a slowly varying term. At second order in a perturbation of the mass, the zero-momentum retarded response of $$\langle\phi^2\rangle$$ to a time-dependent mass shift has the kernel $$I(T)=\int_1^\infty\sqrt{x^2-1}\,x^{-1}\sin(2xT)\,dx$$ in units of $$m$$. Its large-$$T$$ behaviour comes from the branch point at $$x=1$$ and is $$(\sqrt\pi/4)\,T^{-3/2}\cos(2T+\pi/4)$$. The tail oscillates at frequency $$2m$$ with zero mean, and a smooth feature of width $$w$$ in the history (a $$\mathrm{sech}^2$$ profile, for instance) excites it with a weight of order $$e^{-\pi mw}$$, up to a power of $$mw$$, which is exponentially small for cosmological $$w\gtrsim1/H$$. Excitations of the field above the adiabatic vacuum are particles and are not part of $$\rho_{\rm ren}$$.

### Light fields during inflation

Theorem 1 and the estimates above require $$m\gg H$$ over the interval. A field lighter than the Hubble rate during inflation is outside them. The stationary distribution of a light spectator field $$\sigma$$ with potential $$V$$ in de Sitter space of rate $$H_I$$ is $$P(\sigma)\propto\exp[-8\pi^2V(\sigma)/3H_I^4]$$ [[SY]](#ref-sy). For $$V=\tfrac12m^2\sigma^2$$ the mean energy density is $$\langle V\rangle=\tfrac12m^2\langle\sigma^2\rangle=3H_I^4/16\pi^2$$, independent of $$m$$. Setting this equal to $$\rho_\Lambda$$ requires $$H_I=(16\pi^2\rho_\Lambda/3)^{1/4}=6.03$$ meV. The variance approaches its stationary value as $$1-\exp(-2m^2N/3H_I^2)$$ after $$N$$ e-folds, so equilibrium needs $$N\gtrsim3H_I^2/2m^2$$, which for $$m\lesssim H_0$$ is at least $$2.6\times10^{61}$$ e-folds. The de Sitter entropy at that rate is $$8\pi^2M_P^2/H_I^2=1.3\times10^{61}$$, so $$N$$ would exceed it by a factor $$2.05$$. If a bound $$N\lesssim S_{\rm dS}$$ on the duration of inflation is imposed as an assumption, a fraction $$1-e^{-0.49}=0.39$$ of the stationary variance is reached for $$m=H_0$$. In either case the value of $$\rho_\Lambda$$ is exchanged for the value of $$H_I$$, and the field value in our patch is a random draw from the distribution, so one input replaces one input.

## 4. Horizon quantisation, condensates and clocks

### The Kodama state

The Kodama state is a wave functional of the Ashtekar connection that satisfies the Hamiltonian constraint of gravity with cosmological constant $$\Lambda$$ in the connection formulation. For the self-dual connection $$A_+=\Gamma+iK$$ (the spin connection and the extrinsic curvature) on a closed three-manifold $$\Sigma$$ it is the exponential of a multiple of the Chern–Simons functional $$CS(A_+)$$, with a coefficient proportional to $$1/\Lambda$$. Randono generalised it to a real Immirzi parameter $$\gamma$$ [[Randono]](#ref-randono). The Immirzi parameter is the real constant that enters the definition of the real connection variables and the area spectrum $$a_j=8\pi\gamma\ell_P^2\sqrt{j(j+1)}$$ of a puncture of spin $$j$$. In the generalised state the exponent is $$i$$ times the Chern–Simons functional with coefficients proportional to $$1/\Lambda$$ for the Euler (imaginary-part) combination and $$1/\gamma\Lambda$$ for the Pontryagin (real-part) combination. A large gauge transformation is a gauge transformation $$g:\Sigma\to SL(2,\mathbb C)$$ that is not homotopic to the identity; the classes are labelled by the degree of $$g$$ in $$\pi_3(SL(2,\mathbb C))=\mathbb Z$$, and under such a transformation the Chern–Simons functional shifts by a Wess–Zumino term.

<div class="lemma" markdown="1">
**Lemma 2** (real shift). For $$g:\Sigma\to SL(2,\mathbb C)$$ on a closed oriented three-manifold, $$CS(A^g)-CS(A)$$ is a real constant times the degree of $$g$$ in $$\pi_3(SL(2,\mathbb C))=\mathbb Z$$.
</div>

<div class="proof" markdown="1">
*Proof.* The shift is a multiple of $$\int_\Sigma g^*\mathrm{tr}(\theta^3)$$ with $$\theta=g^{-1}dg$$, since the remaining terms are exact. The form $$\mathrm{tr}(\theta^3)$$ is closed on $$SL(2,\mathbb C)$$, so the integral depends only on the homotopy class of $$g$$; $$SL(2,\mathbb C)$$ deformation retracts onto $$SU(2)$$, and a representative of degree $$n$$ is an $$SU(2)$$-valued map. For $$\theta=\theta^aT_a$$ with $$T_a=-i\sigma_a/2$$, $$\mathrm{tr}(T_aT_bT_c)=-\epsilon_{abc}/4$$ is real, so the integral is real and proportional to $$n$$. $$\square$$
</div>

Consequently the Euler part, the one with the "pure $$\Lambda$$" coefficient, is invariant under large gauge transformations in the Lorentzian theory for every $$\Lambda$$. The cosmological constant enters the large-gauge phase through the Pontryagin coefficient only, which is proportional to $$1/\gamma\Lambda$$. For real $$\gamma$$ the state picks up the phase $$\exp(2\pi i\kappa_\gamma n)$$ under a transformation of degree $$n$$, with

$$
\kappa_\gamma=\frac{3}{2\gamma\Lambda\ell_P^2}=\frac{S_{\rm dS}}{2\pi\gamma},\qquad S_{\rm dS}=\frac{3\pi}{\Lambda\ell_P^2}, \tag{4}
$$

the level of the Chern–Simons theory in [[Randono]](#ref-randono), who notes that the coupling constants can be adjusted to make $$\kappa_\gamma$$ an integer. For the self-dual state ($$\gamma=i$$) the shift in the exponent is real, so a transformation of degree $$n$$ multiplies the modulus by $$e^{nS_{\rm dS}}$$ and the state is not invariant up to a phase for any $$\Lambda$$ [[Witten]](#ref-witten). With $$H_0=67.36$$ km s$$^{-1}$$ Mpc$$^{-1}$$ and $$\Omega_\Lambda=0.6847$$, $$\Lambda=3\Omega_\Lambda H_0^2/c^2=1.09\times10^{-52}\ {\rm m}^{-2}$$, $$\Lambda\ell_P^2=2.85\times10^{-122}$$ and $$S_{\rm dS}=3.3\times10^{122}$$, so $$\kappa_\gamma=5.3\times10^{121}$$ for $$\gamma=1$$. Consecutive integers differ in $$\Lambda$$ by the relative amount $$1/\kappa_\gamma\approx1.9\times10^{-122}$$.

<div class="proposition" markdown="1">
**Proposition 3** (allowed values form an arithmetic progression). Let $$a_i\ne0$$ and $$b_i$$, $$i=1,\dots,r$$, be constants independent of $$\Lambda$$, and let $$W$$ be the set of $$s\in\mathbb R$$ with $$a_is+b_i\in\mathbb Z$$ for all $$i$$. If all ratios $$a_i/a_j$$ are rational, $$W$$ is empty or an arithmetic progression. If some ratio is irrational, $$W$$ has at most one point.
</div>

<div class="proof" markdown="1">
*Proof.* Each condition defines a shifted lattice $$(\mathbb Z-b_i)/a_i$$. If $$s,s'\in W$$, then $$s-s'\in\bigcap_ia_i^{-1}\mathbb Z$$. This intersection is $$\{0\}$$ if some ratio $$a_i/a_j$$ is irrational, and a lattice $$\Delta s\,\mathbb Z$$ otherwise, with $$\Delta s$$ the smallest positive common element. So $$W$$ has at most one point in the first case. In the second, if $$s_0\in W$$ then $$W\subset s_0+\Delta s\,\mathbb Z$$, and every element of $$s_0+\Delta s\,\mathbb Z$$ lies in $$W$$ because $$a_i\Delta s\in\mathbb Z$$ for all $$i$$; so $$W$$ is an arithmetic progression. $$\square$$
</div>

In the treatments of the Kodama state by Randono [[Randono]](#ref-randono) and of horizon states at finite level by Engle, Noui, Perez and Pranzetti [[ENPP]](#ref-enpp), the conditions that involve $$\Lambda$$ are the integrality of $$\kappa_\gamma$$ in (4) and of rational multiples of it: the level of an isolated-horizon Chern–Simons theory, the maximal spin $$j_{\max}=\kappa_\gamma/2$$, a larger gauge group. They are of the form of Proposition 3 in the variable $$s=S_{\rm dS}$$, with $$b_i=0$$ and $$a_i\in\mathbb Q/\gamma$$ by (4), so the allowed set is a progression with no preferred element. This identification of the conditions is taken from the cited treatments.

A single allowed point requires an exact relation between the offsets: with two conditions of irrational ratio, $$s^*=(n_1-b_1)/a_1$$ and $$b_2\equiv-a_2(n_1-b_1)/a_1\pmod 1$$ for an integer $$n_1$$. A value $$s^*\sim10^{122}$$ then needs $$n_1\sim10^{122}$$, and the relation between the offsets must hold exactly for that integer. The proposition does not exclude a second, non-homogeneous integrality condition with an irrational ratio. By Lemma 2 it cannot come from the $$SL(2,\mathbb C)$$ part of the gauge group.

### The q-deformed horizon

The horizon of the universe is modelled as a sphere pierced by $$n$$ punctures, each carrying an $$SU(2)$$ spin $$j_i$$ and contributing the area $$a_{j_i}$$ above. The states are the invariants of the tensor product of the spins. If the level $$\kappa$$ of a bulk Chern–Simons theory is tied to the horizon, the invariants are replaced by the conformal blocks of $$SU(2)_\kappa$$, and their number is given by the Verlinde formula

$$
\mathcal V_\kappa(j)=\frac2K\sum_{l=1}^{K-1}\sin^2\!\frac{\pi l}{K}\prod_{i=1}^n\frac{\sin\big((2j_i+1)\pi l/K\big)}{\sin(\pi l/K)},\qquad K=\kappa+2. \tag{5}
$$

Let $$C(j)$$ denote the classical ($$\kappa\to\infty$$) number of $$SU(2)$$ invariants in the product of the spins $$j_1,\dots,j_n$$.

<div class="proposition" markdown="1">
**Proposition 4** (q-transparency). (i) If $$\sum_ij_i\le\kappa$$, then $$\mathcal V_\kappa(j)=C(j)$$. (ii) If $$\sum_ij_i=\kappa+1$$ and $$C(j)\ge1$$, then $$\mathcal V_\kappa(j)=C(j)-1$$. (iii) If the area of the horizon satisfies $$A\le8\pi\gamma\ell_P^2(\kappa+1)$$, every configuration of punctures that fits has $$\mathcal V_\kappa=C$$. With the level of the bulk theory tied to the cosmological constant by $$\kappa=c/(\gamma\Lambda\ell_P^2)$$ and the area of the de Sitter horizon $$A_{\rm dS}=12\pi/\Lambda$$, (iii) holds for all $$\kappa$$ when $$c\ge3/2$$, which includes the level (4) ($$c=3/2$$) and the value $$c=3$$ that follows from matching $$a_{j_{\max}}$$ with $$j_{\max}=\kappa/2$$ to $$A_{\rm dS}$$. The entropy condition $$S_{\rm Verlinde}(A_{\rm dS})=A_{\rm dS}/4$$ is then the classical one and contains no information on $$\Lambda$$.
</div>

<div class="proof" markdown="1">
*Proof.* The sum (5) is a discrete Fourier average of the character product, and averaging over $$K$$-th roots of unity keeps only the Fourier modes divisible by $$K$$. Let $$z=e^{2i\theta}$$ and $$\prod_i\chi_{j_i}=\sum_mc_mz^m$$ with $$\chi_j=\sum_{m=-j}^jz^m$$, $$c_{-m}=c_m$$ and $$c_m=0$$ for $$\vert m\vert >J\equiv\sum_ij_i$$. If $$J$$ is a half-integer, the substitution $$l\to K-l$$ multiplies each summand of (5) by $$(-1)^{2J}=-1$$, so $$\mathcal V_\kappa=0$$, and $$C=0$$ since no invariant exists; we take $$J$$ an integer from here on. With $$\sin^2\theta=(2-z-z^{-1})/4$$ and $$\sum_{l=0}^{K-1}z_l^m=K\,[K\mid m]$$ for $$z_l=e^{2\pi il/K}$$ (the $$l=0$$ term vanishes), (5) becomes

$$\mathcal V_\kappa=\sum_{r\in K\mathbb Z}(c_r-c_{r+1}),$$

using the symmetry of $$c$$. The classical count is $$C=c_0-c_1$$ by the Weyl integration formula. For $$J\le\kappa=K-2$$ only $$r=0$$ contributes, which proves (i); for $$J=K-1$$ the term $$r=-K$$ gives $$-c_{K-1}=-c_J=-1$$, which proves (ii). For (iii), $$a_j>8\pi\gamma\ell_P^2j$$, so $$\sum_ia_{j_i}\le A\le8\pi\gamma\ell_P^2(\kappa+1)$$ forces $$\sum_ij_i<\kappa+1$$, and a non-zero invariant has integer $$\sum_ij_i$$, so $$\sum_ij_i\le\kappa$$. With $$A_{\rm dS}=12\pi\gamma\ell_P^2\kappa/c$$ the condition $$A_{\rm dS}\le8\pi\gamma\ell_P^2(\kappa+1)$$ is $$12\kappa/c\le8(\kappa+1)$$, which holds for $$c\ge3/2$$. $$\square$$
</div>

The proposition says that the horizon of the universe holds so few maximal quanta that the deformation cannot change any count, so the Verlinde condition is the classical one and leaves $$\Lambda$$ free. Direct evaluation of (5) for all configurations of at most four punctures with $$\kappa\le8$$ agrees with (i) and (ii). The configuration $$\{\kappa/2,\kappa/2,1\}$$ has $$C=1$$ and $$\mathcal V_\kappa=0$$ for $$\kappa=2,4,10,20,40$$ (in each case $$\sum j=\kappa+1$$), and its area is $$8\pi\gamma\ell_P^2(\kappa+1+\sqrt2)$$ up to terms of order $$1/\kappa$$, so the threshold in (iii) cannot be raised by more than $$\sqrt2$$ in the factor $$(\kappa+1)$$. A value $$c<3/2$$ would open a window where the deformation is visible. The value of $$c$$ is a modelling input, taken here from the two normalisations above.

### A group-field-theory condensate

Group field theory (GFT) is a quantum field theory on copies of a group, whose perturbative expansion generates spin-foam amplitudes. A condensate is a state in which many quanta occupy one mode, and its mean-field equation gives an effective cosmological dynamics in which a massless scalar field $$\phi_c$$ serves as a relational clock. We take from the cited treatments [[dCPS]](#ref-dcps), [[GO]](#ref-go) the equation for a single mode $$\sigma=x\,e^{i\theta}$$ in the clock time $$\phi_c$$, with units $$\hbar=1$$ and primes denoting $$d/d\phi_c$$,

$$x''=Q^2/x^3+E^2x-\sum_n\lambda_nx^{n-1},$$

with conserved charge $$Q=x^2\theta'$$. The volume is $$V=vx^2$$, and $$d\phi_c/dt=Q/V$$, so that $$H=(Q/3V)(V'/V)$$. Substituting the conserved energy $$E_{\rm tot}=x'^2+Q^2/x^2-E^2x^2+\sum_n(2\lambda_n/n)x^n$$ gives the identity

$$
H^2=\frac{4Q^2}{9}\Big[\frac{E^2}{V^2}+\frac{vE_{\rm tot}}{V^3}-\frac{v^2Q^2}{V^4}-\sum_n\frac{2\lambda_n}{n}\,v^{-2}\Big(\frac{V}{v}\Big)^{(n-6)/2}\Big], \tag{6}
$$

so that each interaction term is a perfect fluid with $$\rho_n\propto V^{(n-6)/2}$$ and equation of state $$w_n=(4-n)/2$$: dust for $$n=4$$ and a cosmological constant for $$n=6$$, with $$\Lambda=3H^2\to(4/9)Q^2\vert \lambda_6\vert /v^2$$. With two massless scalars $$\chi$$ and $$\eta$$, either can serve as the clock. Gielen and Polaczek [[GP]](#ref-gp) showed that in the free theory with several scalars the clock and the non-clock scalars are treated differently.

<div class="proposition" markdown="1">
**Proposition 5** (clock dependence; derivation within the model). Let the same couplings $$\lambda_n$$, the same volume parameter $$v$$ and the same classical data $$(\pi_\chi,\pi_\eta)$$ be described by a condensate peaked on $$\chi$$ (frame $$\chi$$) or on $$\eta$$ (frame $$\eta$$), where the charge $$Q$$ in (6) is the momentum $$\pi_c$$ of the clock. Then the cosmological constant and the dust density of the interaction terms satisfy $$\Lambda_\eta/\Lambda_\chi=\rho_{\rm dust,\eta}/\rho_{\rm dust,\chi}=(\pi_\eta/\pi_\chi)^2$$, while the stiff term, fixed by matching to general relativity, is proportional to $$\pi_\chi^2+\pi_\eta^2$$ in both frames.
</div>

<div class="proof" markdown="1">
*Proof.* In frame $$c$$, (6) holds with $$Q=\pi_c$$. The interaction term with $$n=6$$ gives $$\Lambda_c=(4/9)\pi_c^2\vert \lambda_6\vert /v^2$$, and the term with $$n=4$$ gives a contribution $$(2/9)\pi_c^2\vert \lambda_4\vert /(vV)$$ to $$H^2$$ (dust). Both are proportional to $$\pi_c^2$$ with all other factors equal, and taking the ratio of the two frames gives the stated law. The coefficient $$E^2$$ in frame $$c$$ includes the non-clock momentum; requiring that the stiff term be the general-relativistic one, $$(4\pi G/3)(\pi_\chi^2+\pi_\eta^2)/V^2$$, fixes it, as in [[GP]](#ref-gp). $$\square$$
</div>

Hence the interaction-induced dust density and cosmological constant are not functions of the clock-independent data $$(V,\pi_\chi^2+\pi_\eta^2)$$. Each is proportional to $$\pi_c^2$$, while the stiff term is not, so the ratio of $$\Lambda_c$$ to the coefficient of the stiff term is proportional to $$\pi_c^2/(\pi_\chi^2+\pi_\eta^2)$$ and differs between the frames. The ratio of the dust coefficient to $$\Lambda_c$$ is the same in both frames: it equals $$\vert \lambda_4\vert v/2\vert \lambda_6\vert $$, independent of $$\pi_c$$. A frame-independent statement can use only ratios among coefficients with the same power of $$\pi_c$$, such as this one. The proposition depends on the matching rule that identifies the non-clock momentum and on the use of the same interaction in both frames; both are modelling choices in the cited treatments. A frame-independent value of $$\Lambda$$ would need a rule that selects the clock, and the model supplies none.

### Clock Hamiltonians

In the Page–Wootters construction [[PW83]](#ref-pw83) the quantum state of a closed system is stationary, and time evolution of the rest of the system is read off by conditioning on the reading of a clock. A finite-dimensional clock with Hamiltonian $$H_C(v)$$ then contributes to the minisuperspace Hamiltonian constraint of a flat universe with volume variable $$v=a^3$$. In the Born–Oppenheimer regime, with $$v$$ slow, conditioning on a clock reading in the adiabatic branch $$n$$ gives that branch an induced energy density $$\rho_{{\rm ind},n}(v)=\lambda_n(v)/v$$, where $$\lambda_n$$ is the corresponding eigenvalue of $$H_C(v)$$. This definition is a model assumption. A term $$c\,v^{\gamma}$$ in $$\lambda_n$$ is then a fluid with $$\rho\propto v^{\gamma-1}$$, and since $$\rho\propto a^{-3(1+w)}=v^{-(1+w)}$$ for an equation of state $$w$$, it has $$w=-\gamma$$. Take $$H_C(v)=H_{C0}+v^\beta B$$ with a non-degenerate $$H_{C0}$$ having eigenvalues $$E_m$$ and a perturbation matrix $$B$$ with $$B_{nn}=0$$.

<div class="proposition" markdown="1">
**Proposition 6** (sign of the induced vacuum energy). By Rayleigh–Schrödinger perturbation theory in $$u=v^\beta$$, $$\lambda_n=E_n+u^2\Delta_n+O(u^3)$$ with $$\Delta_n=\sum_{m\ne n}\vert B_{nm}\vert ^2/(E_n-E_m)$$. The term $$u^2/v=v^{2\beta-1}$$ is a cosmological constant ($$w=-1$$) iff $$\beta=1/2$$, with energy density $$\Delta_n$$. (i) $$\Delta_n$$ depends only on the gaps of $$H_{C0}$$, and $$\sum_n\Delta_n=0$$. (ii) For any population $$p_n$$ of the branches that is passive, meaning non-increasing in $$E_n$$ (every thermal population at positive temperature is of this kind), the mean induced energy is $$\sum_np_n\Delta_n=\tfrac12\sum_{n,m}(p_n-p_m)\vert B_{nm}\vert ^2/(E_n-E_m)\le0$$, with strict inequality if some $$B_{nm}\ne0$$ links levels of different population. It is non-negative for populations that are non-decreasing in energy. In particular, a clock in its ground branch has $$\Delta_0=-\sum_{m\ne0}\vert B_{0m}\vert ^2/(E_m-E_0)<0$$.
</div>

<div class="proof" markdown="1">
*Proof.* The expansion is the second-order energy shift. (i) A shift $$H_{C0}\to H_{C0}+c$$ changes every $$\lambda_n$$ by $$c$$, which changes only the constant (dust-like, $$w=0$$) term. The sum vanishes because the summand $$\vert B_{nm}\vert ^2/(E_n-E_m)$$ is antisymmetric under $$n\leftrightarrow m$$. (ii) Symmetrise the double sum: $$\sum_{n,m}p_n\vert B_{nm}\vert ^2/(E_n-E_m)=\tfrac12\sum_{n,m}(p_n-p_m)\vert B_{nm}\vert ^2/(E_n-E_m)$$, and each pair has $$(p_n-p_m)/(E_n-E_m)\le0$$ for a passive population. $$\square$$
</div>

Within the model assumptions $$\rho_{\rm ind}=\lambda_n/v$$ and $$\beta=1/2$$, a clock in a thermal state therefore cannot supply a positive cosmological constant. The proof uses the ordering inequality $$(p_n-p_m)(E_n-E_m)\le0$$ for passive populations, which also underlies Theorem 1(c) on the page on [the sign of energy exchange between vacuum and matter](/quantum-gravity/vacuum-matter-sign/). There it shows that a unital map cannot lower the energy of a passive state, and here that second-order level repulsion at fixed populations lowers the mean energy. A clock that stores work, with populations rising in energy, is not covered, and the induced term adds to a bare $$\Lambda$$ that nothing in the construction fixes. Whether a physical clock couples to the volume with exponent $$\beta=1/2$$ is not determined here; for other $$\beta$$ the induced term is a fluid with $$w=-2\beta\ne-1$$.

### A multiple-point condition

A degeneracy condition on the Standard Model potential could in principle constrain the vacuum term, and the following check shows that it does not. The multiple-point principle of Froggatt and Nielsen [[FN]](#ref-fn) requires the effective potential of the Standard Model to have two degenerate minima, one at the electroweak scale and one at a large field value $$\phi_*$$. The condition fixes couplings, and the top-quark mass is the one checked against the measured value; the Standard Model is close to this situation [[Buttazzo]](#ref-buttazzo). A bare vacuum term $$\Lambda_b$$ enters the potential additively and drops out of every difference of values and every derivative, so degeneracy imposes no condition on it. If one asks instead that the difference of the two minima be the observed $$\rho_\Lambda$$, then with $$V=\tfrac14\lambda_{\rm eff}(\phi)\phi^4$$ at large field the effective quartic coupling at $$\phi_*$$ must equal $$4\rho_\Lambda/\phi_*^4$$. For an illustrative $$\phi_*=5\times10^{17}$$ GeV and $$\rho_\Lambda=2.5\times10^{-47}\ {\rm GeV}^4$$ this is $$1.6\times10^{-117}$$. The cosmological-constant problem becomes the problem of fixing $$\lambda_{\rm eff}(\phi_*)$$ to this precision. The renormalisation-group running of $$\lambda_{\rm eff}$$ is taken from [[Buttazzo]](#ref-buttazzo) and not reconsidered.

## 5. The ratio of dark-matter to baryon density

Write $$R=\Omega_c/\Omega_b=r\,\mu$$, where $$r=n_X/n_B$$ is the ratio of the number densities of dark-matter and baryon particles and $$\mu=m_X/m_B$$ the ratio of their masses (with $$m_B$$ the mean baryon mass). The Planck 2018 values $$\Omega_ch^2=0.1200\pm0.0012$$ and $$\Omega_bh^2=0.02237\pm0.00015$$ [[Planck VI]](#ref-planck6) give $$R=5.364\pm0.065$$, with the correlation between the two densities ignored, and $$\sigma(\ln R)=0.012$$.

Constructions that relate $$r$$ to a count, for instance by a conserved asymmetry, a symmetry or a group-theory ratio, leave $$m_X$$ as a parameter of the Lagrangian. Lemma 1 makes the consequence precise. Let $$\vartheta$$ denote the data of a construction and suppose its selection conditions are built from a functional $$F$$ that is invariant under the dark scaling $$D_t:m_X\mapsto e^tm_X$$ with all other data fixed.

<div class="proposition" markdown="1">
**Proposition 7** (dark-mass modulus). If $$F\circ D_t=F$$ and $$R$$ is $$C^1$$ with weight $$w=\partial_t\ln R(D_t\vartheta)\ne0$$ on an open set $$U$$, then no condition built from $$F$$ alone fixes $$R$$ on $$U$$: the solution set is $$D_t$$-invariant and $$\ln R$$ moves strictly monotonically along each orbit. If $$r$$ does not depend on $$m_X$$, $$w=1$$.
</div>

<div class="proof" markdown="1">
*Proof.* The invariance of the solution set is the first statement of Lemma 1 with $$\mathcal T=\{D_t\}$$ and $$\alpha=1$$. The motion of $$R$$ along an orbit is computed directly: $$\partial_t\ln R=\partial_t\ln\mu+\partial_t\ln r=1+\partial_t\ln r$$, so $$\ln R(D_t\vartheta)=\ln R(\vartheta)+\int_0^tw$$, which is strictly monotone along the part of the orbit inside $$U$$, where $$w\ne0$$. If $$r$$ does not depend on $$m_X$$, $$w=1$$. $$\square$$
</div>

The obstruction is the independence of $$m_X$$ from $$m_B$$: scaling the overall mass scale of the whole theory has weight zero on $$R$$. The cases without a dark modulus are those that fix $$\mu$$ by an additional relation. If the relation is $$\mu=1$$ (an exact $$\mathbb Z_2$$ copy of the visible sector, or two species of equal mass), then $$R=r$$ is a number: $$R=1$$ is $$67.6\sigma$$ below the measured value and $$R=2$$, for a doubling of the number of states, is $$52.1\sigma$$ below it. An exact mirror sector also has a modulus, the temperature ratio of the two sectors, which appears in $$r$$; its value is limited by nucleosynthesis and by the effective number of relativistic species.

The remaining way to fix $$\mu$$ is to link the dark and visible confinement scales. Bai and Schwaller [[BS]](#ref-bs) relate them through infrared fixed points of the two gauge couplings, and Lonsdale and Volkas [[LV]](#ref-lv) relate the dark and visible couplings at a high scale in a grand-unified setting. Suppose the dark baryon mass is proportional to the dark confinement scale $$M_{\rm d}$$ and the dark gauge coupling equals a unified coupling at a scale $$M_U$$ fixed by the measured visible couplings. Then $$\mu$$ has no free modulus, and the question is how precisely it is fixed.

<div class="proposition" markdown="1">
**Proposition 8** (threshold floor; one-loop derivation). Let the dark coupling at $$M_U$$ be $$\alpha_d^{-1}(M_U)=\alpha_U^{-1}+\delta$$, with $$\delta$$ an unknown ultraviolet threshold correction, and let the one-loop running of $$\alpha_d^{-1}$$ have coefficient $$b_d>0$$, so that $$\alpha_d^{-1}(M)=\alpha_d^{-1}(M_U)-(b_d/2\pi)\ln(M_U/M)$$. Then $$\partial\ln\mu/\partial\delta=-2\pi/b_d$$, and an uncertainty $$\vert \delta\vert \ge\delta_0$$ gives $$\sigma(\ln\mu)\ge2\pi\delta_0/b_d$$. Reaching the precision of the measured ratio, $$\sigma(\ln R)=0.012$$, needs $$\delta_0\le0.012\,b_d/2\pi$$, which is $$0.0086$$ for the illustrative value $$b_d=4.5$$.
</div>

<div class="proof" markdown="1">
*Proof.* At one loop the confinement scale, where $$\alpha_d^{-1}$$ reaches zero, satisfies $$\ln M_{\rm d}=\ln M_U-2\pi\alpha_d^{-1}(M_U)/b_d$$. The visible confinement scale and $$M_U$$ do not depend on $$\delta$$, and $$\mu\propto M_{\rm d}$$. Differentiate. $$\square$$
</div>

The sensitivity is present unless the construction suppresses it. Ritter and Volkas [[RV]](#ref-rv) implement the dependence on the ultraviolet initial conditions of the couplings in the infrared-fixed-point construction and find a reduced number of viable models. We read this as a residual sensitivity of the type in the proposition. Reaching $$\sigma(\ln R)=0.012$$ with $$b_d\approx4.5$$ needs a threshold knowledge of $$0.009$$ in $$\alpha^{-1}$$, a relative precision of order $$2\times10^{-4}$$ for $$\alpha^{-1}$$ of order $$40$$. Whether an infrared fixed point can make $$\partial\ln\mu/\partial\delta$$ small at two loops is open.

### Unification and the mass of an electroweak multiplet

The route above ties the dark-matter mass to the scale $$M_U$$ at which the visible couplings meet. A colourless $$SU(2)$$ multiplet with zero hypercharge, the minimal electroweak dark-matter candidate, does not contribute to the one-loop running of $$\alpha_1$$ or $$\alpha_3$$, since the one-loop coefficient of a gauge coupling receives contributions proportional to the Dynkin index of the multiplet under that factor and these vanish. The scale at which $$\alpha_1=\alpha_3$$ is therefore independent of the multiplet's mass at one loop, and the multiplet cannot move the crossing.

With the inputs $$\alpha_{\rm em}^{-1}(M_Z)=127.95$$, $$\sin^2\theta_W=0.2312$$, $$\alpha_s(M_Z)=0.118$$ and the Standard Model coefficients $$(b_1,b_2,b_3)=(41/10,-19/6,-7)$$, the couplings satisfy $$\alpha_1=\alpha_3$$ at $$2.4\times10^{14}$$ GeV with $$\alpha^{-1}=40.4$$, $$\alpha_1=\alpha_2$$ at $$1.0\times10^{13}$$ GeV and $$\alpha_2=\alpha_3$$ at $$9.7\times10^{16}$$ GeV. If a grand-unified gauge boson has mass of the order of the first crossing scale, the dimension-six estimate $$\tau_p\sim M_X^4/(\alpha_G^2m_p^5)$$ with $$\alpha_G^{-1}=40$$ gives $$\tau_p\sim10^{29}$$ years at $$M_X=2.4\times10^{14}$$ GeV, to be compared with the Super-Kamiokande bound $$\tau/B(p\to e^+\pi^0)>2.4\times10^{34}$$ years at $$90\%$$ confidence level [[SK]](#ref-sk). The estimate is of order of magnitude only; its prefactor depends on hadronic matrix elements, and models with suppressed gauge-boson couplings change the relation between $$M_X$$ and the crossing scale. This is an observation about Standard Model running, not a theorem. It shows that any dependence of the unification scale on the dark sector needs charged states beyond a minimal electroweak multiplet, and that the proton-decay bound then constrains the structure.

## 6. Routes that remain open

A global condition, such as a sequestering or unimodular constraint, can assign $$\Lambda$$ a value outside quantum field theory on a fixed background, because the cosmological constant of section 3 is a coordinate on the counterterm space. The properties of the linear versions are on the page on [linear global constraints](/quantum-gravity/linear-lambda-constraints/).

A principle that selects the number of degrees of freedom, or the state, by a measure on histories is not a condition built from a covariant functional of the kind in Lemma 1. Likewise a discrete rigid datum selected by an extremal property is not moved by a symmetry, and the lemma says nothing about it.

A stochastic spectator from inflation fixes $$\rho_\Lambda$$ from $$H_I=6.03$$ meV (section 3), which replaces one input by another and requires $$N\gtrsim2.6\times10^{61}$$ e-folds for $$m\lesssim H_0$$.

For the Kodama and Verlinde conditions of Propositions 3 and 4, a second condition with an irrational ratio, or a bulk level $$\kappa$$ not tied to the area ($$c<3/2$$), would lift the degeneracy. The pure Lorentzian theory supplies neither.

Proposition 6 gives a non-positive mean induced energy for passive populations only, so a clock that stores work is outside it.

A dark-mass modulus can be fixed by another relation. A symmetry equating the dark and visible scales (a mirror sector) leaves the temperature-ratio modulus. A single confinement scale for dark matter and baryons [[Farrar]](#ref-farrar) gives $$\Omega_{\rm DM}/\Omega_b=5.3$$ within about $$15\%$$ for a sexaquark mass in the range $$1860$$ to $$1880$$ MeV, and is disputed on the grounds of the relic abundance [[KT]](#ref-kt), [[MS]](#ref-ms). A common ultraviolet coupling fixes $$\mu$$ at the precision of Proposition 8.

## 7. Open points

The decoupling statement of Remark 1 for interacting fields and for states other than the adiabatic vacuum is taken from the literature. The coefficient of $$H^6/m^2$$ is derived here for the conformal coupling $$\xi=1/6$$ only, in de Sitter space and on the $$\Lambda$$CDM history; its value for other couplings and for histories with other contents remains to be derived. Proposition 5 is a derivation within the mean-field treatment, and it is open whether a condensate state exists that is covariant under the choice of clock. The weight $$\beta$$ with which a physical clock, such as the cosmic microwave background photons or a neutrino flavour oscillation, couples to the volume decides whether the clock-Hamiltonian route in Proposition 6 gives a cosmological constant at all. The ratio $$R$$ in a common-coupling model with an infrared fixed point at two loops is also open; Proposition 8 gives the quantity to be computed.

## References

1. <a id="ref-wald"></a>Wald, “Trace anomaly of a conformally invariant quantum field in curved spacetime,” Phys. Rev. D 17, 1477 (1978).
2. <a id="ref-hw1"></a>Hollands, Wald, “Local Wick Polynomials and Time Ordered Products of Quantum Fields in Curved Spacetime,” Commun. Math. Phys. 223, 289 (2001), [arXiv:gr-qc/0103074](https://arxiv.org/abs/gr-qc/0103074).
3. <a id="ref-hw2"></a>Hollands, Wald, “Existence of Local Covariant Time Ordered Products of Quantum Fields in Curved Spacetime,” Commun. Math. Phys. 231, 309 (2002), [arXiv:gr-qc/0111108](https://arxiv.org/abs/gr-qc/0111108).
4. <a id="ref-pf"></a>Parker, Fulling, “Adiabatic regularization of the energy-momentum tensor of a quantized field in homogeneous spaces,” Phys. Rev. D 9, 341 (1974).
5. <a id="ref-fns"></a>Ferreiro, Navarro-Salas, “Running gravitational couplings, decoupling, and curved spacetime renormalization,” Phys. Rev. D 102, 045021 (2020), [arXiv:2005.05258](https://arxiv.org/abs/2005.05258).
6. <a id="ref-bd"></a>Bunch, Davies, “Quantum field theory in de Sitter space: renormalization by point-splitting,” Proc. R. Soc. Lond. A 360, 117 (1978).
7. <a id="ref-sy"></a>Starobinsky, Yokoyama, “Equilibrium state of a massless self-interacting scalar field in the de Sitter background,” Phys. Rev. D 50, 6357 (1994), [arXiv:astro-ph/9407016](https://arxiv.org/abs/astro-ph/9407016).
8. <a id="ref-randono"></a>Randono, “Generalizing the Kodama State II: Properties and Physical Interpretation,” [arXiv:gr-qc/0611074](https://arxiv.org/abs/gr-qc/0611074).
9. <a id="ref-witten"></a>Witten, “A Note On The Chern-Simons And Kodama Wavefunctions,” [arXiv:gr-qc/0306083](https://arxiv.org/abs/gr-qc/0306083).
10. <a id="ref-enpp"></a>Engle, Noui, Perez, Pranzetti, “The SU(2) Black Hole entropy revisited,” JHEP 1105, 016 (2011), [arXiv:1103.2723](https://arxiv.org/abs/1103.2723).
11. <a id="ref-pw83"></a>Page, Wootters, “Evolution without evolution: Dynamics described by stationary observables,” Phys. Rev. D 27, 2885 (1983).
12. <a id="ref-gp"></a>Gielen, Polaczek, “Hamiltonian group field theory with multiple scalar matter fields,” Phys. Rev. D 103, 086011 (2021), [arXiv:2009.00615](https://arxiv.org/abs/2009.00615).
13. <a id="ref-dcps"></a>de Cesare, Pithis, Sakellariadou, “Cosmological implications of interacting Group Field Theory models: cyclic Universe and accelerated expansion,” Phys. Rev. D 94, 064051 (2016), [arXiv:1606.00352](https://arxiv.org/abs/1606.00352).
14. <a id="ref-go"></a>Granata, Oriti, “Dynamical dark energy from quantum gravity condensates,” [arXiv:2609.34917](https://arxiv.org/abs/2609.34917).
15. <a id="ref-fn"></a>Froggatt, Nielsen, “Standard Model Criticality Prediction: Top mass 173 +/- 5 GeV and Higgs mass 135 +/- 9 GeV,” Phys. Lett. B 368, 96 (1996), [arXiv:hep-ph/9511371](https://arxiv.org/abs/hep-ph/9511371).
16. <a id="ref-buttazzo"></a>Buttazzo et al., “Investigating the near-criticality of the Higgs boson,” JHEP 12 (2013) 089, [arXiv:1307.3536](https://arxiv.org/abs/1307.3536).
17. <a id="ref-bs"></a>Bai, Schwaller, “The Scale of Dark QCD,” Phys. Rev. D 89, 063522 (2014), [arXiv:1306.4676](https://arxiv.org/abs/1306.4676).
18. <a id="ref-rv"></a>Ritter, Volkas, “Exploring the cosmological dark matter coincidence using infrared fixed points,” [arXiv:2210.11011](https://arxiv.org/abs/2210.11011).
19. <a id="ref-lv"></a>Lonsdale, Volkas, “Grand unified hidden-sector dark matter,” Phys. Rev. D 90, 083501 (2014), [arXiv:1407.4192](https://arxiv.org/abs/1407.4192).
20. <a id="ref-farrar"></a>Farrar, “A precision test of the nature of Dark Matter and a probe of the QCD phase transition,” [arXiv:1805.03723](https://arxiv.org/abs/1805.03723).
21. <a id="ref-kt"></a>Kolb, Turner, “Dibaryons cannot be the dark matter,” Phys. Rev. D 99, 063519 (2019), [arXiv:1809.06003](https://arxiv.org/abs/1809.06003).
22. <a id="ref-ms"></a>Moore, Slatyer, “On the cosmology and terrestrial signals of sexaquark dark matter,” Phys. Rev. D 110, 023515 (2024), [arXiv:2403.03972](https://arxiv.org/abs/2403.03972).
23. <a id="ref-sk"></a>Super-Kamiokande Collaboration, “Search for proton decay via $$p\to e^+\pi^0$$ and $$p\to\mu^+\pi^0$$ with an enlarged fiducial volume in Super-Kamiokande I-IV,” Phys. Rev. D 102, 112011 (2020), [arXiv:2010.16098](https://arxiv.org/abs/2010.16098).
24. <a id="ref-planck6"></a>Planck Collaboration, “Planck 2018 results. VI. Cosmological parameters,” Astron. Astrophys. 641, A6 (2020), [arXiv:1807.06209](https://arxiv.org/abs/1807.06209).
