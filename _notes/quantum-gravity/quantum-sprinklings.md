---
title: Lorentz-invariant quantum sprinklings are condensates
date: 2026-10-29
category: substrate
order: 1
description: A state of a bosonic field over spacetime that is invariant under the Poincaré group, has a finite mean number of quanta in every bounded region and gives sharp counts there is a mixture of coherent states of constant amplitude, so its counts are Poisson with a random intensity that is the same everywhere. No sub-Poisson, number-squeezed or fermionic version exists, and the classical analogue of this statement is false.
status: Theorems proved in full for the Fock and CAR settings and for classical determinantal processes; complex processes of bounded total variation are classified; the failure of the classical analogue is proved for random lattices and checked by simulation in two dimensions.
excludes: Poincaré-invariant states of finite sharp density on the Weyl algebra over spacetime that are not mixtures of coherent condensates, in particular sub-Poisson, number-squeezed, Fock-number and fermionic states, and invariant determinantal point processes.
---

A sprinkling of Minkowski space is a random discrete set of points whose law is invariant under the Poincaré group. The standard one is the Poisson process, whose number variance in a region equals its mean. We ask whether a *quantum* state of a field on spacetime, whose quanta are the points, can be exactly Poincaré invariant with a finite density of points and still count better than Poisson: with a variance below the mean, or with a fixed number of quanta in a region. The answer is no, under three hypotheses made precise in section 1: the counts are the particle numbers of a bosonic field whose one-particle space is $$L^2$$ of spacetime, the state is locally normal, and the mean number of quanta in every bounded region is finite. Every such state is a mixture of coherent states with a constant amplitude. Its counts are those of a Poisson process whose intensity is a random number, the same at every point, so the number variance is at least the mean (Theorem 4). For fermions the same hypotheses leave only the vacuum (Theorem 6), and for classical determinantal point processes, the closest classical model of a fermionic sprinkling, invariance under translations and one boost subgroup leaves only the empty process (Theorem 7). The classical analogue of the bosonic statement is false. Haar-random lattices are Poincaré invariant, ergodic and not Poisson, and their counts have variance equal to the mean (Proposition 9). Complex-valued point processes of bounded total variation do not escape either (Theorem 10).

A Poisson sprinkling is Poincaré invariant, and Bombelli, Henson and Sorkin [[BHS]](#ref-bhs) prove that no Lorentz-equivariant choice of a preferred direction at a point of a Poisson sprinkling exists. Dowker, Henson and Sorkin [[DHS]](#ref-dhs) explain why boosted regular lattices are not invariant, and Saravani and Aslanbeigi [[SA]](#ref-sa) show that Poisson sprinkling gives the best number-volume correspondence in arbitrarily small regions, while two-dimensional Lorentzian lattices do better in large ones. That a Poisson process is the coherent state of a Fock-space field is due to Hudson and Parthasarathy [[HP]](#ref-hp). The surveys [[Surya]](#ref-surya) and [[Sorkin]](#ref-sorkin) introduce causal sets and sprinklings. This page adds the classification of all invariant states in the Fock setting, including the phase of the condensate, the fermionic and determinantal versions, the role of each hypothesis, and the example showing that the classical analogue fails.

The same classification matters in two other places. The mean of the causal-set d'Alembertian on a mixed Poisson process is not the continuum d'Alembertian when the operator is normalised by the mean density; this is worked out in [Mixed-Poisson and signed substrates and the four-dimensional d'Alembertian](/quantum-gravity/signed-substrates/). The condensate states are never ground states of a positive-energy dynamics, which is shown in [Integer counts of a positive-energy field](/quantum-gravity/count-algebras/).

## 1. Sprinklings, counts and the setting

Let $$M=\mathbb R^{1,3}$$ with Lebesgue measure $$dV$$, let $$V(D)$$ be the volume of a bounded Borel set $$D$$, and let $$P$$ be the proper orthochronous Poincaré group. The same arguments work in $$\mathbb R^{1,d-1}$$ for every $$d\ge 2$$. A *sprinkling* is a random locally finite subset of $$M$$; a causal set is obtained by keeping the points and the causal order between them. Everything below depends only on the *counts* $$N(D)$$, the number of points in $$D$$. The Poisson process of intensity $$\lambda$$ is the sprinkling in which $$N(D)$$ is Poisson with mean $$\lambda V(D)$$ and counts in disjoint sets are independent. Its mean and variance are both $$\lambda V(D)$$, and its law is invariant under $$P$$ because Lebesgue measure is.

*A quantum version.* Let the one-particle space be $$L^2(M,dV)$$: a one-particle state is a wave function on spacetime, and the points of the sprinkling are the positions of the quanta. Let $$\Gamma(h)$$ be the bosonic Fock space over a Hilbert space $$h$$, with annihilation operators $$a(f)$$ (antilinear in $$f$$), creation operators $$a^*(f)$$ and $$[a(f),a^*(g)]=\langle f,g\rangle$$, where the inner product is antilinear in the first slot. Weyl operators are $$W(f)=\exp(a^*(f)-a(f))$$ and satisfy $$W(f)W(g)=e^{-i\,\mathrm{Im}\langle f,g\rangle}W(f+g)$$. For a bounded Borel set $$D$$ of positive volume let $$L^2(D)\subset L^2(M)$$ be the functions vanishing outside $$D$$, let $$\mathfrak A(D)$$ be the C$$^*$$-algebra generated by $$W(f)$$ with $$f\in L^2(D)$$, and let $$\mathfrak A$$ be the algebra generated by all of them. The group $$P$$ acts by automorphisms $$\alpha_g W(f)=W(f\circ g^{-1})$$. The count of $$D$$ is the number operator of $$\Gamma(L^2(D))$$,

$$
N(D)=d\Gamma(1_D)=\sum_k a^*(e_k)a(e_k), \tag{1}
$$

for any orthonormal basis $$(e_k)$$ of $$L^2(D)$$. It has spectrum $$\{0,1,2,\dots\}$$, and for disjoint $$D_1,D_2$$ the Fock space of $$D_1\cup D_2$$ is the tensor product of those of $$D_1$$ and $$D_2$$, so counts in disjoint sets commute. We call (1) a *sharp* count: it assigns an integer to every region and is not smeared against a profile.

*Coherent states are Poisson processes.* For $$z\in\mathbb C$$ let $$\omega_z$$ be the state on $$\mathfrak A(D)$$ given by the normalised exponential vector $$e^{-\lvert z\rvert ^2V(D)/2}e(z1_D)$$ of $$\Gamma(L^2(D))$$. Then

$$
\omega_z(W(f))=\exp\!\big(-\tfrac12\lVert f\rVert ^2+2i\,\mathrm{Im}(\bar z\!\int\! f)\big),\qquad f\in L^2(D), \tag{2}
$$

which is consistent in $$D$$ and extends to every $$f\in L^1\cap L^2$$. The exponential vector factorises over a partition of $$D$$, and each factor is a coherent state of the corresponding Fock space, so the counts of disjoint sets are independent and Poisson with means $$\lvert z\rvert ^2V(D_i)$$ [[HP]](#ref-hp). The state $$\omega_z$$ is Poincaré invariant, since $$\lVert f\circ g^{-1}\rVert =\lVert f\rVert $$ and $$\int f\circ g^{-1}=\int f$$. It is not a vector in the Fock space of $$L^2(M)$$ when $$z\ne0$$, because the constant $$z\,1_M$$ is not square integrable. For this reason we work with the local algebras $$\mathfrak A(D)$$ and not with the Weyl algebra over $$L^2(M)$$, on which $$\int f$$ is undefined. Working over $$L^1\cap L^2$$ is possible, but then the state on functions of unbounded support is an additional datum that the counts do not see.

*Hypotheses.* A state $$\omega$$ on $$\mathfrak A$$ is a *quantum sprinkling* if it satisfies:

(H1) *Invariance.* $$\omega\circ\alpha_g=\omega$$ for all $$g\in P$$.

(H2) *Local normality.* For every bounded $$D$$ there is a density matrix $$\rho_D$$ on $$\Gamma(L^2(D))$$ with $$\omega(A)=\mathrm{tr}(\rho_D A)$$ for $$A\in\mathfrak A(D)$$.

(H3) *Finite density.* $$\mathrm{tr}(\rho_D N(D))<\infty$$ for every bounded $$D$$. We write $$\omega(N(D))$$ for this number.

Local normality implies regularity, so that the field operators exist. Finite density is needed: Section 7 gives invariant states without it. The one-body density is the operator $$T$$ on $$L^2(M)$$ with

$$
\langle f,Tg\rangle=\omega\big(a^*(g)\,a(f)\big),\qquad f,g\in L^2_c(M), \tag{3}
$$

where $$L^2_c$$ is the set of square-integrable functions with bounded support. The trace in (3) is finite: (H3) puts the range of $$\rho_D^{1/2}$$ in the domain of $$N(D)^{1/2}$$, and $$\lVert a(f)\psi\rVert \le\lVert f\rVert \,\lVert N(D)^{1/2}\psi\rVert $$. For a coherent state, (3) reads $$\langle f,Tg\rangle=\lvert z\rvert ^2\,\overline{\textstyle\int f}\int g$$, a rank-one operator with constant kernel. The first result is that this is the only possibility.

## 2. The one-body density is a constant

The tool is a statement about measures that are invariant under a boost.

<div class="lemma" markdown="1">

**Lemma 1** (wandering sets). Let $$\mu$$ be a finite positive Borel measure on $$\mathbb R^n$$, and let $$B$$ be a linear map of determinant $$\pm1$$ with $$B_*\mu=\mu$$. If $$W$$ is a Borel set such that the sets $$B^kW$$, $$k\in\mathbb Z$$, are pairwise disjoint, then $$\mu(W)=0$$. The same holds for a function $$G\in L^1(\mathbb R^n)$$ with $$G\circ B=G$$ almost everywhere, with $$\int_W\lvert G\rvert =0$$ in place of $$\mu(W)=0$$, and then $$G=0$$ almost everywhere on $$\bigcup_kB^kW$$.

</div>

<div class="proof" markdown="1">

*Proof.* The sets $$B^kW$$ all have the same measure, and they are disjoint, so $$\sum_k\mu(B^kW)\le\mu(\mathbb R^n)<\infty$$ forces $$\mu(W)=0$$. For $$G$$ the same argument applies to $$\int_{B^kW}\lvert G\rvert =\int_W\lvert G\circ B^k\rvert =\int_W\lvert G\rvert $$, which uses that $$B$$ preserves Lebesgue measure. $$\square$$

</div>

A boost with rapidity $$\theta$$ along $$x^1$$ acts in null coordinates $$u=p^0+p^1$$, $$v=p^0-p^1$$ as $$(u,v)\mapsto(e^\theta u,e^{-\theta}v)$$ and fixes $$p^2,p^3$$. The sets $$W=\{e^{0}\le\lvert u\rvert <e^{\theta}\}$$ and $$W'=\{1\le\lvert v\rvert <e^\theta\}$$ are wandering, and their images cover $$\{u\ne0\}$$ and $$\{v\ne0\}$$. This is the mechanism behind every result below.

<div class="theorem" markdown="1">

**Theorem 2** (constant one-body density). Under (H1)–(H3) there is a number $$\rho\ge0$$ with

$$
\omega\big(a^*(g)a(f)\big)=\rho\,\overline{\textstyle\int f}\,\textstyle\int g\qquad\text{for all } f,g\in L^2_c(M), \tag{4}
$$

and $$\omega(N(D))=\rho\,V(D)$$ for every bounded $$D$$.

</div>

The idea is that translation invariance writes the density as a positive measure $$\mu$$ on momentum space; finite density says $$\mu$$ has finite total mass; and a boost-invariant finite measure has no room except at the origin.

<div class="proof" markdown="1">

*Proof.* *Step 1: a spectral measure.* The form $$\tau(f,g)=\langle f,Tg\rangle$$ is positive and translation invariant. For $$f,g\in L^2(D)$$ it is bounded: $$a^*(f)a(f)\le\lVert f\rVert ^2N(D)$$, because $$a^*(e)a(e)$$ with $$\lVert e\rVert =1$$ is one term of (1) in a basis containing $$e$$, so Cauchy–Schwarz gives $$\lvert \tau(f,g)\rvert \le\lVert f\rVert \lVert g\rVert \,\omega(N(D))$$. By the Schwartz kernel theorem and the Bochner–Schwartz theorem [[RS2]](#ref-rs2) there is a unique positive tempered measure $$\mu$$ on $$\mathbb R^4$$ with

$$
\tau(f,g)=\int\overline{\hat f(p)}\,\hat g(p)\,d\mu(p),\qquad \hat f(p)=\int e^{-ip\cdot x}f(x)\,dx,\quad f,g\in C_c^\infty(M), \tag{5}
$$

where $$p\cdot x$$ is the Minkowski pairing. The products $$\overline{\hat f}\hat g$$ are the Fourier transforms of the convolutions $$\tilde f*g\in C_c^\infty$$, with $$\tilde f(x)=\overline{f(-x)}$$, and these are dense, so $$\tau$$ determines $$\mu$$.

*Step 2: finite density is finite mass.* Let $$D$$ be an open ball and let $$(e_k)$$ be an orthonormal basis of $$L^2(D)$$ made of functions in $$C_c^\infty(D)$$. By (1), (3), (5) and Tonelli,

$$
\omega(N(D))=\sum_k\tau(e_k,e_k)=\int\sum_k\lvert \hat e_k(p)\rvert ^2\,d\mu(p)=V(D)\,\mu(\mathbb R^4).
$$

The last step is Parseval: $$\hat e_k(p)=\langle e^{ip\cdot}1_D,e_k\rangle$$ up to conjugation, so $$\sum_k\lvert \hat e_k(p)\rvert ^2=\lVert e^{ip\cdot}1_D\rVert ^2=V(D)$$ for every $$p$$. Hence (H3) gives $$\mu(\mathbb R^4)<\infty$$.

*Step 3: boosts.* A boost $$\Lambda$$ acts on test functions by $$f\mapsto f\circ\Lambda^{-1}$$ and on transforms by $$\hat f\mapsto\hat f\circ\Lambda^{-1}$$, since the pairing is invariant. Invariance (H1) of $$\tau$$ therefore says $$\mu$$ is invariant under $$\Lambda$$. Apply Lemma 1 to the boost along $$x^1$$ with the wandering sets above: $$\mu(u\ne0)=\mu(v\ne0)=0$$, so $$\mu$$ lives on $$\{p^0=p^1=0\}$$. The boosts along $$x^2$$ and $$x^3$$ give $$\{p^0=p^2=0\}$$ and $$\{p^0=p^3=0\}$$. The intersection is the origin, so $$\mu=\rho\,\delta_0$$ with $$\rho=\mu(\mathbb R^4)$$, and (5) becomes (4) because $$\hat f(0)=\int f$$. Since $$\mu$$ is finite, dominated convergence extends (5) from $$C_c^\infty$$ to $$f,g\in L^2_c$$ (approximate in $$L^2$$ and in $$L^1$$ simultaneously, so that $$\hat f_n\to\hat f$$ boundedly), which gives (4) as stated; the Parseval computation of Step 2 with an orthonormal basis of $$L^2(D)$$ then gives $$\omega(N(D))=\rho V(D)$$ for every bounded $$D$$. $$\square$$

</div>

Only translations and boosts were used, not rotations and not positivity of the energy. Translation invariance alone allows any finite measure $$\mu$$: a Fermi ball in momentum space gives a translation-invariant, hyperuniform, fermionic state of finite density (section 7). It is the boosts that force the condensation.

<div class="remark" markdown="1">

**Remark** (finitely additive spectral measures). Step 1 produced a $$\sigma$$-additive measure from the continuity of $$\tau$$. Suppose instead that one only had $$\tau(f,g)=M(\overline{\hat f}\hat g)$$ for a boost-invariant positive linear functional $$M$$ on the bounded continuous functions, that is, a finitely additive "measure". Its restriction to $$C_0(\mathbb R^4)$$ is a positive functional, hence by the Riesz–Markov theorem a finite Radon measure, which is boost invariant and therefore a multiple of $$\delta_0$$. For $$f,g\in L^1$$ the product $$\overline{\hat f}\hat g$$ lies in $$C_0$$, so (4) still holds: whatever $$M$$ has "at infinity" contributes to no correlation $$\omega(a^*(g)a(f))$$.

</div>

## 3. All the counting happens in one mode

Fix a bounded set $$D$$ of positive volume, and let $$u_D=1_D/\sqrt{V(D)}$$ be the normalised constant on $$D$$. By (4), $$a(f)$$ has zero second moment in $$\omega$$ whenever $$\int f=0$$, and the next step turns this into a statement about the state.

<div class="proposition" markdown="1">

**Proposition 3** (single-mode reduction and attenuation). Assume (H1)–(H3). Then:

(a) For every bounded $$D$$ of positive volume, $$\Gamma(L^2(D))=\Gamma(\mathbb Cu_D)\otimes\Gamma(u_D^\perp)$$ and $$\rho_D=\sigma_D\otimes\lvert \Omega\rangle\langle\Omega\rvert $$, where $$\sigma_D$$ is a density matrix on the one-mode space $$\Gamma(\mathbb Cu_D)\cong\ell^2(\mathbb N_0)$$ and $$\Omega$$ is the vacuum of $$\Gamma(u_D^\perp)$$. In particular $$N(D)=a^*(u_D)a(u_D)$$ on the support of $$\rho_D$$.

(b) If $$D\subset D'$$ and $$\eta=V(D)/V(D')$$, then $$\sigma_D$$ is the image of $$\sigma_{D'}$$ under the pure-loss channel of transmissivity $$\eta$$, equivalently

$$
\Phi_D(c)=\Phi_{D'}(\sqrt\eta\,c),\qquad \Phi_D(c):=\mathrm{tr}\big(\sigma_D\,e^{c a^\dagger}e^{-\bar c a}\big), \tag{6}
$$

where $$a=a(u_D)$$ on the left and $$a=a(u_{D'})$$ on the right.

</div>

<div class="proof" markdown="1">

*Proof.* (a) Let $$f\in L^2(D)$$ with $$\int f=0$$, that is, $$f\perp u_D$$. By (4), $$\mathrm{tr}(\rho_Da^*(f)a(f))=0$$, which is $$\lVert a(f)\rho_D^{1/2}\rVert ^2_{HS}=0$$, so the support of $$\rho_D$$ lies in $$\ker a(f)$$. The common kernel of $$a(f)$$ over $$f\in u_D^\perp$$ is $$\Gamma(\mathbb Cu_D)\otimes\Omega$$, the only vectors of $$\Gamma(u_D^\perp)$$ annihilated by all $$a(f)$$ being multiples of the vacuum.

(b) Write $$u_D=\sqrt\eta\,u_{D'}+w$$ with $$w=u_D-\sqrt\eta\,u_{D'}\in L^2(D')$$. Then $$\langle u_{D'},w\rangle=0$$, $$\int w=0$$ and $$\lVert w\rVert ^2=1-\eta$$. For $$c\in\mathbb C$$ the Weyl operator $$W(cu_D)=W(c\sqrt\eta\,u_{D'})W(cw)$$ is a product of commuting factors, one in the mode $$u_{D'}$$ and one in $$u_{D'}^\perp$$. By (a) for $$D'$$,

$$
\omega(W(cu_D))=\mathrm{tr}\big(\sigma_{D'}W(c\sqrt\eta\,u_{D'})\big)\,e^{-(1-\eta)\lvert c\rvert ^2/2}.
$$

By (a) for $$D$$ the left side is $$\mathrm{tr}(\sigma_DW(cu_D))$$. Passing to the normally ordered function with $$W(cu)=e^{ca^\dagger}e^{-\bar ca}e^{-\lvert c\rvert ^2/2}$$ turns this into (6). $$\square$$

</div>

The relation (6) is the attenuator: the state of the mode $$u_D$$ is what remains of the state of $$u_{D'}$$ after a fraction $$1-\eta$$ of the quanta has been lost. It holds for every pair of nested regions, so the state of a mode in a small region is obtained from the state in a large region by loss, for all loss rates at once. Normally ordered moments scale as $$\langle a^{\dagger j}a^k\rangle_D=\eta^{(j+k)/2}\langle a^{\dagger j}a^k\rangle_{D'}$$ (with the appropriate mode on each side). In particular Mandel's parameter $$Q=(\langle a^{\dagger2}a^2\rangle-\langle a^\dagger a\rangle^2)/\langle a^\dagger a\rangle=(\mathrm{Var}\,N-\langle N\rangle)/\langle N\rangle$$ obeys $$Q_D=\eta\,Q_{D'}$$. Since $$Q\ge-1$$ for every state, $$Q_D\ge-V(D)/V(D')$$, and letting $$D'$$ grow gives $$Q_D\ge0$$. This already shows $$\mathrm{Var}\,N(D)\ge\langle N(D)\rangle$$ when the second moments are finite, before any classification.

*Checks.* (i) We applied the pure-loss channel numerically to random mixed states of a mode truncated at 59 quanta. For four random states and transmissivities between 0.44 and 0.83, $$Q_{\rm out}-\eta Q_{\rm in}$$ was below $$2.2\times10^{-15}$$ in absolute value, with $$Q_{\rm in}$$ ranging over $$-0.46,\dots,5.65$$. (ii) A number state cannot occur. Suppose the mode of a region has exactly one quantum, $$\sigma_D=\lvert 1\rangle\langle1\rvert $$. By (6) with $$\eta=1/k$$, $$k>1$$, the state $$\sigma_{D'}$$ of the region $$k$$ times larger has $$\Phi_{D'}(c)=1-k\lvert c\rvert ^2$$. The normally ordered function of the number state $$\lvert n\rangle$$ is the Laguerre polynomial $$L_n(\lvert c\rvert ^2)$$, so $$1-k\lvert c\rvert ^2=p_0L_0+p_1L_1$$ forces $$p_1=k>1$$ and $$p_0=1-k<0$$, which is not a state. For example $$k=1.5$$ gives $$(p_0,p_1)=(-0.5,1.5)$$. Pure-loss evolution of this operator returns $$\lvert 1\rangle\langle 1\rvert $$, so the failure is positivity of the preimage and not an algebraic slip.

## 4. Classification

<div class="theorem" markdown="1">

**Theorem 4** (invariant quantum sprinklings are coherent condensates). Let $$\omega$$ satisfy (H1)–(H3). There is a unique Borel probability measure $$\mathsf P$$ on $$\mathbb C$$ with $$\int\lvert z\rvert ^2d\mathsf P(z)=\rho<\infty$$ such that

$$
\omega(A)=\int\omega_z(A)\,d\mathsf P(z),\qquad A\in\mathfrak A(D),\ D\ \text{bounded}. \tag{7}
$$

Conversely every such mixture satisfies (H1)–(H3). Consequently, for disjoint bounded sets $$D_1,\dots,D_k$$ the counts have the joint law

$$
\Pr\big(N(D_i)=n_i,\ i\le k\big)=\mathbb E\prod_{i}e^{-\Lambda V(D_i)}\frac{(\Lambda V(D_i))^{n_i}}{n_i!},\qquad \Lambda=\lvert z\rvert ^2, \tag{8}
$$

so $$N$$ is a Cox (mixed Poisson) process with a spacetime-constant random intensity $$\Lambda$$ of mean $$\rho$$, and $$\mathrm{Var}\,N(D)=\rho V(D)+V(D)^2\,\mathrm{Var}\,\Lambda\ \ge\ \mathbb E N(D)$$, with equality for all $$D$$ if and only if $$\Lambda=\rho$$ almost surely.

</div>

The idea is that (6) says the one-mode state of a small region is a loss-image of the state of every larger region, and a state that is a loss-image at every loss rate has a positive Husimi function at every scale; the limit of large regions is a positive measure on coherent amplitudes.

<div class="proof" markdown="1">

*Proof.* Let $$\Psi(\zeta)=\Phi_D(\zeta/\sqrt{V(D)})$$. By (6) with $$c=\zeta/\sqrt{V(D)}$$ and $$\sqrt\eta\,c=\zeta/\sqrt{V(D')}$$, $$\Psi$$ is the same for nested regions, hence for all bounded regions (any two lie in a third). It is continuous with $$\Psi(0)=1$$, since $$\sigma_D$$ is a normal state.

The Husimi function $$Q_D(\alpha)=\pi^{-1}\langle\alpha\lvert \sigma_D\rvert \alpha\rangle$$ is a probability density on $$\mathbb C$$. Its Fourier transform is the anti-normally ordered function $$\mathrm{tr}(\sigma_De^{-\bar\xi a}e^{\xi a^\dagger})=\Phi_D(\xi)e^{-\lvert \xi\rvert ^2}$$. A Fourier transform of a probability density is positive definite, so with $$\zeta=\sqrt{V(D)}\,\xi$$ the function $$\Psi(\zeta)e^{-\lvert \zeta\rvert ^2/V(D)}$$ is positive definite on $$\mathbb R^2\cong\mathbb C$$ for every $$V(D)\in(0,\infty)$$. Let $$V(D)\to\infty$$. A pointwise limit of positive-definite functions is positive definite, so $$\Psi$$ is a continuous positive-definite function with $$\Psi(0)=1$$, and Bochner's theorem gives a probability measure $$\mathsf P$$ on $$\mathbb C$$ with

$$
\Psi(\zeta)=\int e^{\bar z\zeta-z\bar\zeta}\,d\mathsf P(z).
$$

A coherent state $$\lvert \beta\rangle$$ has $$\mathrm{tr}(\lvert \beta\rangle\langle\beta\rvert e^{\xi a^\dagger}e^{-\bar\xi a})=e^{\xi\bar\beta-\bar\xi\beta}$$, and the normally ordered function determines the state, so $$\sigma_D=\int\lvert z\sqrt{V(D)}\rangle\langle z\sqrt{V(D)}\rvert \,d\mathsf P(z)$$. By (2) and Proposition 3(a), $$\lvert z\sqrt{V(D)}\rangle\otimes\Omega$$ is the exponential vector defining $$\omega_z$$ on $$\mathfrak A(D)$$, which gives (7). Uniqueness of $$\mathsf P$$ follows from that of the Fourier transform. Finite density gives $$\int\lvert z\rvert ^2d\mathsf P=\rho$$.

For the converse, (2) shows that each $$\omega_z$$, hence the mixture, is invariant; local normality and finite density are clear. The law (8) follows from the independence and Poisson counts under each $$\omega_z$$. The mean and variance of a mixed Poisson variable with random mean $$\Lambda V$$ are $$\mathbb E\Lambda V$$ and $$\mathbb E\Lambda V+\mathrm{Var}(\Lambda)V^2$$, which is $$+\infty$$ if $$\mathbb E\Lambda^2=\infty$$. $$\square$$

</div>

The counts determine only the law of $$\Lambda=\lvert z\rvert ^2$$, whereas the state also contains the phase of $$z$$. For $$\mathsf P=\delta_{z_0}$$ the state is a single coherent condensate. For $$\mathsf P$$ uniform on the circle $$\lvert z\rvert ^2=\rho$$ the counts are exactly Poisson and the state is the phase average of a condensate; this is the state whose counting statistics coincide with the classical sprinkling.

<div class="corollary" markdown="1">

**Corollary 5** (what cannot occur). Under (H1)–(H3), no state has: counts with $$\mathrm{Var}\,N(D)<\mathbb E N(D)$$ for any $$D$$ (sub-Poisson, number-squeezed); a fixed number of quanta in a region of positive volume (a Fock number state of the mode $$u_D$$); or an area-law number variance $$\mathrm{Var}\,N(D)\propto$$ area of $$\partial D$$. The only fluctuation beyond Poisson allowed is a random but spacetime-constant intensity, which contributes a term $$\mathrm{Var}(\Lambda)V^2$$ growing like the square of the volume.

</div>

The statement about the area law follows from the formula for $$\mathrm{Var}\,N$$: the variance is $$\rho V+V^2\mathrm{Var}\Lambda$$ for every region, with no term depending on the boundary.

*Fermions.* Replace the Weyl algebra by the CAR algebra generated by $$a(f)$$, $$f\in L^2_c(M)$$, with $$\{a(f),a^*(g)\}=\langle f,g\rangle$$. Every state of this algebra gives $$\lvert \omega(a^*(g)a(f))\rvert \le\lVert f\rVert \lVert g\rVert $$, because $$\lVert a(f)\rVert =\lVert f\rVert $$, so $$T$$ is a bounded operator with $$0\le T\le1$$. No normality or regularity is needed. For a state we define $$\omega(N(D))=\sum_k\omega(a^*(e_k)a(e_k))\in[0,\infty]$$ for an orthonormal basis of $$L^2(D)$$, which does not depend on the basis.

<div class="theorem" markdown="1">

**Theorem 6** (no invariant fermionic sprinkling). Let $$\omega$$ be a state on the CAR algebra over $$L^2_c(M)$$ that is invariant under translations and boosts and has $$\omega(N(D))<\infty$$ for some open ball $$D$$. Then $$\omega$$ is the Fock vacuum: $$\omega(N(D))=0$$ for every $$D$$ and $$\omega(a^*(f)a(g))=0$$.

</div>

<div class="proof" markdown="1">

*Proof.* The argument of Theorem 2 uses nothing about the statistics except the bound on $$\tau$$, which here comes from $$\lVert a(f)\rVert =\lVert f\rVert $$. Steps 1 to 3 give $$\tau(f,g)=\rho\,\overline{\int f}\int g$$, with finite $$\rho$$ since $$\omega(N(D))=V(D)\mu(\mathbb R^4)<\infty$$ (Step 2 applies to the one ball $$D$$ for which the density is finite, and the extension of (4) from $$C_c^\infty$$ to $$L^2_c$$ is by continuity, as in the proof of Theorem 2). Now take $$f=1_{D'}$$ for any ball $$D'$$: $$\tau(f,f)=\omega(a^*(f)a(f))\le\lVert f\rVert ^2$$ gives $$\rho V(D')^2\le V(D')$$, that is $$\rho\le1/V(D')$$. Since $$D'$$ can have arbitrarily large volume, $$\rho=0$$. Then $$\omega(a^*(f)a(f))=0$$ for all $$f\in L^2_c$$, and $$\omega(N(D))=0$$ for every bounded $$D$$. For any $$B$$ in the algebra, Cauchy–Schwarz gives $$\lvert \omega(Ba(f))\rvert ^2\le\omega(BB^*)\,\omega(a^*(f)a(f))=0$$, and by taking adjoints $$\omega(a^*(f)B)=0$$. Every monomial in the generators that contains a factor $$a(f)$$ or $$a^*(f)$$ is of this form after normal ordering, so $$\omega$$ vanishes on all of them: $$\omega$$ is the vacuum. $$\square$$

</div>

There are Poincaré-invariant quasi-free CAR states with $$0\le T\le 1$$ and sub-Poisson smeared counts, namely those whose $$T$$ is a bounded invariant multiplier, but they have infinite density (section 7).

## 5. Classical determinantal sprinklings

The CAR theorem concerns a quantum state. The point process of a quasi-free fermionic state is determinantal, and the classical statement is simpler and needs no Fock space. A point process on $$M$$ is *determinantal* with Hermitian kernel $$\mathcal K$$ if $$\mathcal K$$ is a self-adjoint operator on $$L^2(M)$$ with $$0\le\mathcal K\le1$$ and $$1_B\mathcal K1_B$$ trace class for bounded $$B$$, and the process has correlation functions $$\rho_n(x_1,\dots,x_n)=\det[K(x_i,x_j)]$$, where $$K(x,y)$$ is a measurable kernel of $$\mathcal K$$ with $$\int_BK(x,x)dx=\mathrm{tr}(1_B\mathcal K1_B)$$ [[Soshnikov]](#ref-soshnikov). In particular $$\rho_1(x)=K(x,x)$$ and $$\rho_2(x,y)=\rho_1(x)\rho_1(y)-\lvert K(x,y)\rvert ^2$$.

<div class="theorem" markdown="1">

**Theorem 7** (no invariant determinantal process). Let $$\Xi$$ be a determinantal point process on $$\mathbb R^{1,d-1}$$, $$d\ge2$$, whose law is invariant under translations and under one boost subgroup. Then $$\Xi$$ is empty almost surely.

</div>

<div class="proof" markdown="1">

*Proof.* Invariance of the law makes $$\rho_1$$ invariant under all translations almost everywhere, hence equal to a constant $$\rho$$ almost everywhere (by Fubini, a locally integrable function that is invariant a.e. under every translation is a.e. constant). Likewise $$\rho_2(x+a,y+a)=\rho_2(x,y)$$ gives $$\lvert K(x,y)\rvert ^2=G(x-y)$$ for a measurable $$G\ge0$$, and invariance under a boost $$B$$ gives $$G\circ B=G$$ almost everywhere.

Next, $$\mathcal K^2\le\mathcal K$$. For bounded $$B$$ the operator $$\mathcal K1_B$$ is Hilbert–Schmidt, with $$\lVert \mathcal K1_B\rVert _{HS}^2=\mathrm{tr}(1_B\mathcal K^2 1_B)\le\mathrm{tr}(1_B\mathcal K1_B)=\rho V(B)$$. The left side is $$\int\!\!\int\lvert K(x,y)\rvert ^21_B(y)\,dx\,dy=V(B)\int G$$ by the form of $$\lvert K\rvert ^2$$. Hence $$G\in L^1(\mathbb R^d)$$ with $$\int G\le\rho$$. Lemma 1, applied to the wandering sets of $$B$$ (the regions $$e^{k\theta}\le\lvert z^0+z^1\rvert <e^{(k+1)\theta}$$), gives $$G=0$$ almost everywhere. Then $$K=0$$ almost everywhere, so $$\mathcal K1_B$$ has zero Hilbert–Schmidt norm for every $$B$$ and $$\mathcal K=0$$. Finally $$\rho V(B)=\mathrm{tr}(1_B\mathcal K1_B)=0$$, so $$\Xi=\emptyset$$. $$\square$$

</div>

Hilbert–Schmidt control of $$\mathcal K1_B$$ replaces the pointwise inequality $$(\mathcal K^2)(x,x)\le K(x,x)$$, which would need a choice of the diagonal values of the kernel. The same lemma gives a statement about hyperuniformity.

<div class="corollary" markdown="1">

**Corollary 8.** Let $$\Xi$$ be a simple point process on $$\mathbb R^{1,d-1}$$ with locally integrable intensity $$\rho_1=\rho$$ and pair correlation $$\rho_2(x,y)=\rho^2g(x-y)$$, invariant under translations and one boost subgroup. If $$g-1\in L^1(\mathbb R^d)$$ then $$g=1$$ almost everywhere, and $$\mathrm{Var}\,N(D)=\rho V(D)$$ for every bounded $$D$$. In particular such a process is not hyperuniform ($$\mathrm{Var}\,N(D)/V(D)\to0$$ along a growing sequence of regions is impossible).

</div>

<div class="proof" markdown="1">

*Proof.* $$g-1$$ is an integrable boost-invariant function, so Lemma 1 gives $$g-1=0$$ almost everywhere. Then $$\mathrm{Var}\,N(D)=\int_D\rho_1+\int_D\!\!\int_D(\rho_2-\rho_1\otimes\rho_1)=\rho V(D)$$. $$\square$$

</div>

The hypothesis $$g-1\in L^1$$ is the one that hyperuniformity normally implies, via the sum rule $$1+\rho\int(g-1)=0$$. A Lorentz-invariant, hyperuniform process must therefore have correlations that are not integrable. Whether such processes exist, and what they look like, is not settled by this page; the null cone carries an invariant measure of infinite mass, which suggests, without proving it, that non-integrable invariant correlations are carried by the null cone.

## 6. What the hypotheses are used for

Each of the three hypotheses (Fock structure with sharp counts, positivity of the state, locality of the algebra) does work, and the following examples show that dropping them can produce invariant sprinklings that are not mixed Poisson.

### 6.1 Classical invariant processes need not be Poisson

Let $$d\ge2$$, let $$X=ASL(d,\mathbb R)/ASL(d,\mathbb Z)$$ with its invariant probability measure (finite by Siegel's theorem [[Siegel]](#ref-siegel)), and let $$\Lambda=g\mathbb Z^d+\xi$$ be the random affine lattice of density 1 with $$(g,\xi)$$ distributed by this measure. Concretely, $$g$$ is distributed by the invariant probability measure on $$SL(d,\mathbb R)/SL(d,\mathbb Z)$$ and, given $$g$$, $$\xi$$ is uniform in the torus $$\mathbb R^d/g\mathbb Z^d$$.

<div class="proposition" markdown="1">

**Proposition 9** (Haar-random lattices). The point process $$\Lambda$$ on $$\mathbb R^{1,d-1}$$ has the following properties.

(a) Its law is invariant under $$P$$ and under all of $$ASL(d,\mathbb R)$$.

(b) For every bounded $$D$$: $$\mathbb E N(D)=V(D)$$ and $$\mathbb E[N(D)(N(D)-1)]=V(D)^2$$, so $$\mathrm{Var}\,N(D)=\mathbb E N(D)$$. The pair correlation is $$\rho_2\equiv1$$.

(c) $$\Lambda$$ is not a Cox process directed by any random measure without atoms; in particular it is neither Poisson nor mixed Poisson.

(d) $$\Lambda$$ is ergodic under $$P$$ [[Moore]](#ref-moore).

</div>

<div class="proof" markdown="1">

*Proof.* (a) The measure on $$X$$ is invariant under left multiplication, and $$P\subset ASL(d,\mathbb R)$$ acts by affine maps $$x\mapsto Ax+a$$ with $$\det A=1$$.

(b) Averaging over $$\xi$$ first gives $$\mathbb E_\xi\sum_{x\in\Lambda}h(x)=\int h$$ for $$h\ge0$$, so $$\mathbb EN(D)=V(D)$$. Write $$\sum_{x\ne y\in\Lambda}1_D(x)1_D(y)=\sum_{v\in g\mathbb Z^d\setminus0}\sum_{x\in\Lambda}1_D(x)1_D(x+v)$$. Averaging over $$\xi$$ turns the inner sum into $$H(v)=\int1_D(x)1_D(x+v)dx$$, and Siegel's mean value theorem [[Siegel]](#ref-siegel), $$\mathbb E_g\sum_{v\in g\mathbb Z^d\setminus0}H(v)=\int H$$ for $$d\ge2$$, gives $$\int H=V(D)^2$$. Hence $$\mathbb E[N(N-1)]=V^2$$ and $$\mathrm{Var}N=V^2+V-V^2=V$$. The same computation with two test functions gives $$\rho_2\equiv1$$.

(c) Almost surely, $$x\ne y\in\Lambda$$ implies $$2y-x\in\Lambda$$ and $$2y-x\notin\{x,y\}$$. Let $$S=\{(x,y,z):z=2y-x\}$$ and $$A$$ a bounded set with $$V(A)>0$$. Then $$\sum^{\ne}_{x,y,z\in\Lambda}1_A(x)1_A(y)1_S(x,y,z)$$ is at least $$N(A)(N(A)-1)$$, which has positive expectation $$V(A)^2$$ by (b). For a Cox process directed by a random measure $$\Xi$$ without atoms, the third factorial moment measure is $$\mathbb E\,\Xi^{\otimes3}$$, and $$\Xi^{\otimes3}(S)=\iint\Xi(dx)\Xi(dy)\,\Xi(\{2y-x\})=0$$. The two expectations disagree.

(d) $$P$$ contains the translations, whose invariant functions on $$X$$ depend on $$g$$ only (they move $$\xi$$ transitively in its torus), and $$SO^+(1,d-1)$$, a closed non-compact subgroup of the simple group $$SL(d,\mathbb R)$$, acts ergodically on $$SL(d,\mathbb R)/SL(d,\mathbb Z)$$ by Moore's theorem [[Moore]](#ref-moore). $$\square$$

</div>

Proposition 9 shows that "Lorentz invariance and finite density force Poisson" is false for classical point processes. The random lattice has variance equal to the mean; the Lorentzian lattices of [[SA]](#ref-sa) have a variance well below the mean, but are invariant only under a discrete group. What fails for the lattice is the Fock structure. In a Fock space the counts in disjoint regions are independent in every coherent component (section 1), whereas in a lattice they are rigidly dependent: the pair $$x,y$$ forces the point $$2y-x$$. Because the second-order statistics agree with Poisson, they cannot see the difference, and neither can Corollary 8, which is consistent with $$\rho_2\equiv1$$. Void probabilities do see it.

*Check in two dimensions.* In null coordinates $$(u,v)$$ the boost is $$(u,v)\mapsto(e^\theta u,e^{-\theta}v)$$ and $$SL(2,\mathbb R)$$ contains it. We sampled $$2\times10^6$$ lattices with $$g$$ from the invariant measure (the shape of the lattice hyperbolically distributed in the modular fundamental domain, a uniform rotation, covolume 1) and $$\xi$$ uniform, and counted points in the square $$[0,a]^2$$ of area $$V=a^2$$.

| $$V$$ | $$\mathbb E N$$ | $$\mathrm{Var}N/\mathbb EN$$ | $$\Pr(N=0)$$ | Poisson $$e^{-V}$$ | $$V\cdot\Pr(N=0)$$ |
|---|---|---|---|---|---|
| 1 | 1.0002 | 1.007 | 0.3044 | $$3.7\times10^{-1}$$ | 0.304 |
| 4 | 4.0005 | 1.10 | 0.0539 | $$1.8\times10^{-2}$$ | 0.216 |
| 16 | 16.0006 | 1.07 | 0.0130 | $$1.1\times10^{-7}$$ | 0.208 |

The mean agrees with $$V$$ to a relative accuracy of $$2\times10^{-4}$$. The ratio of variance to mean is 1 within the sampling error, which is large because the tail of the count distribution is heavy (the largest count seen was 1544 at $$V=16$$). The void probability does not decay exponentially: $$V\Pr(N=0)$$ is close to constant, about 0.21, for $$V\ge4$$, in contrast to $$e^{-V}$$. This is the measurable difference between the random lattice and a Poisson process with the same first and second moments.

### 6.2 Complex amplitudes of bounded total variation are classical

Positivity of the state was used in the Husimi argument of section 4. One can ask whether replacing the positive measure that defines a sprinkling by a complex-valued one removes the obstruction. A *complex point process* (CPP) is a family $$\{\mathsf P_D\}$$, indexed by bounded Borel sets, of complex measures on the space $$\Gamma_D$$ of finite counting measures on $$D$$, with $$\mathsf P_D(\Gamma_D)=1$$, finite total variation $$\lVert \mathsf P_D\rVert <\infty$$, and consistency: the image of $$\mathsf P_{D'}$$ under restriction to $$D\subset D'$$ is $$\mathsf P_D$$. It is *Poincaré invariant* if $$\mathsf P_{gD}=g_*\mathsf P_D$$. A positive process has $$\lVert \mathsf P_D\rVert =1$$. Examples of invariant CPPs with $$\lVert \mathsf P_D\rVert \to\infty$$ are given on [Mixed-Poisson and signed substrates and the four-dimensional d'Alembertian](/quantum-gravity/signed-substrates/).

<div class="theorem" markdown="1">

**Theorem 10** (bounded total variation is classical). Let $$\mathsf P$$ be a Poincaré-invariant CPP with $$S=\sup_D\lVert \mathsf P_D\rVert <\infty$$. Then $$\mathsf P$$ extends to a complex measure on the space $$\Gamma_M$$ of locally finite configurations, and

$$
\mathsf P=S\int e^{i\theta(e)}\,Q_e\,\kappa(de), \tag{9}
$$

where $$\kappa$$ is a probability measure, each $$Q_e$$ is an ergodic Poincaré-invariant classical (positive) point process, and $$\theta$$ is a real measurable function with $$S\int e^{i\theta}d\kappa=1$$. In particular, if $$S=1$$ then $$\theta=0$$ almost everywhere and $$\mathsf P$$ is a classical point process.

</div>

The idea is that the total variations of the restrictions define a positive invariant measure $$\nu$$ dominating $$\mathsf P$$; the density of $$\mathsf P$$ with respect to $$\nu$$ is invariant, hence constant on ergodic components, and its modulus is 1 because the total variation is exhausted.

<div class="proof" markdown="1">

*Proof.* Let $$B_n$$ be the ball of radius $$n$$ and $$\pi_{D'\to D}$$ the restriction of configurations.

*Step 1 (a dominating positive family).* For a measurable map $$\pi$$ and a complex measure $$\mu$$, $$\lvert \pi_*\mu\rvert \le\pi_*\lvert \mu\rvert $$. Hence $$\lvert \mathsf P_D\rvert \le(\pi_{D'\to D})_*\lvert \mathsf P_{D'}\rvert $$ for $$D\subset D'$$, and the right side is nondecreasing in $$D'$$ and has mass at most $$S$$. The setwise limit $$\nu_{B_m}$$ of $$(\pi_{B_n\to B_m})_*\lvert \mathsf P_{B_n}\rvert $$ as $$n\to\infty$$ is then a measure of mass $$S=\lim\lVert \mathsf P_{B_n}\rVert $$, the family $$(\nu_{B_m})$$ is consistent, and $$\lvert \mathsf P_{B_m}\rvert \le\nu_{B_m}$$. The limit is the same along any increasing exhaustion of $$M$$ by bounded sets, so $$g_*\nu_D=\nu_{gD}$$.

*Step 2 (Kolmogorov).* The spaces $$\Gamma_{B_n}$$ are standard Borel, and their projective limit is $$\Gamma_M$$. By Kolmogorov's extension theorem for projective systems on standard Borel spaces [[Parthasarathy]](#ref-parthasarathy) there is a unique finite measure $$\nu$$ on $$\Gamma_M$$ with marginals $$\nu_{B_m}$$. It is invariant by Step 1, and has mass $$S$$.

*Step 3 (martingale).* Since $$\lvert \mathsf P_{B_n}\rvert \le\nu_{B_n}$$ the density $$h_n=d\mathsf P_{B_n}/d\nu_{B_n}$$ exists with $$\lvert h_n\rvert \le1$$. Consistency says that $$h_n$$ is the conditional expectation of $$h_{n+1}$$ on the configuration in $$B_n$$, so $$(h_n)$$ is a bounded martingale for the filtration generated by the restrictions to $$B_n$$ and converges $$\nu$$-almost everywhere and in $$L^1$$ to a function $$h$$ with $$\lvert h\rvert \le1$$. The complex measure $$h\nu$$ has the marginals $$\mathsf P_{B_n}$$, and is the unique complex measure with these marginals, because cylinder sets form a $$\pi$$-system that generates the $$\sigma$$-algebra. This is the extension of $$\mathsf P$$, in the spirit of [[Thomas]](#ref-thomas); the complex-measure extension criterion $$\sup\lVert \mathsf P_D\rVert <\infty$$ is also treated in [[Albeverio–Mazzucchi]](#ref-am).

*Step 4 (invariance).* For $$g\in P$$, $$g_*\mathsf P$$ and $$\mathsf P$$ have the same marginals, so $$g_*\mathsf P=\mathsf P$$, and likewise $$g_*\nu=\nu$$. Therefore $$h\circ g=h$$ $$\nu$$-almost everywhere for each $$g$$, and since $$P$$ is locally compact and second countable, $$h$$ agrees almost everywhere with a strictly invariant function [[Zimmer]](#ref-zimmer).

*Step 5 (ergodic decomposition).* By Varadarajan's theorem on ergodic decompositions [[Varadarajan]](#ref-varadarajan), $$\nu/S=\int Q_e\,\kappa(de)$$ with ergodic invariant probability measures $$Q_e$$. A strictly invariant function is constant on $$Q_e$$-almost every point for $$\kappa$$-almost every $$e$$; call the constant $$h(e)$$. Then $$\mathsf P=S\int h(e)Q_e\kappa(de)$$.

*Step 6 (the modulus is 1).* We have $$\lvert \mathsf P\rvert (\Gamma_M)=\int\lvert h\rvert d\nu\le S$$. Conversely $$\lvert \mathsf P\rvert (\Gamma_M)\ge\lVert \mathsf P_{B_n}\rVert \to S$$. Therefore $$\int(1-\lvert h\rvert )d\nu=0$$ with $$1-\lvert h\rvert \ge0$$, so $$\lvert h\rvert =1$$ $$\nu$$-almost everywhere, and $$h=e^{i\theta}$$. Normalisation $$\mathsf P(\Gamma_M)=1$$ gives $$S\int e^{i\theta}d\kappa=1$$. If $$S=1$$, a unimodular function with average 1 equals 1 almost everywhere. $$\square$$

</div>

A complex process that is invariant, has bounded total variation, and is genuinely non-classical (is not a positive process) is therefore a unimodular-weighted mixture of classical invariant ergodic processes with $$S>1$$ and cancellations. Note that $$Q_e$$ in (9) is any ergodic invariant classical process, not necessarily Poisson; Proposition 9 shows that non-Poisson ergodic components exist. Positivity is sharp in the following sense: dropping it while keeping bounded total variation gives nothing new at the level of ergodic components, and a genuine escape needs $$\lVert \mathsf P_D\rVert \to\infty$$. Processes with unbounded total variation exist in abundance, for any prescribed invariant correlation functions of moderate growth; they are constructed and studied on [Mixed-Poisson and signed substrates and the four-dimensional d'Alembertian](/quantum-gravity/signed-substrates/), where it is also shown that they have no probabilistic (Born) reading of their count probabilities.

## 7. What the theorems do not exclude

Each item lists the hypothesis that is dropped and what is known without it.

- *Broken translation or boost invariance.* A fixed frame is a legitimate datum, for instance the rest frame of the cosmic microwave background. A quasi-free fermionic state with $$T$$ the multiplier by the indicator of a Euclidean ball in momentum space is translation invariant, has finite density and is not boost invariant. Steps 1 and 2 of Theorem 2 hold when boosts are not imposed, with $$\mu$$ any finite measure. Which counting statistics such states give, and what they imply for a cosmological application, is a separate calculation.
- *Infinite local density.* A bounded Lorentz-invariant multiplier, for example the indicator of $$\{p^0>0,\ 1\le-p^2\le4\}$$, defines a Poincaré-invariant quasi-free CAR state with $$0\le T\le1$$ and sub-Poisson smeared counts. Here $$\mu(\mathbb R^4)=\infty$$, so $$\omega(N(D))=V(D)\mu(\mathbb R^4)=\infty$$ for every $$D$$ by the computation in Step 2 of Theorem 2. Sharp counts have no finite mean in such states, and (H3) cannot be weakened to finiteness of smeared counts.
- *Counts that are not $$d\Gamma(1_D)$$.* A different notion of "number of quanta in a region", for example a positive operator-valued measure on the one-particle space, is outside the setting. Page [Integer counts of a positive-energy field](/quantum-gravity/count-algebras/) shows what integer additive counts of a positive-energy field can be.
- *Classical invariant processes.* Proposition 9 gives invariant non-Poisson processes. They are not Fock states, and their dependence between regions is rigid. Their behaviour as a substrate for the causal-set d'Alembertian is treated on [Lorentz-invariant point sets and the structures they cannot carry](/quantum-gravity/invariant-point-sets/).
- *Complex processes of unbounded total variation.* Excluded neither by Theorem 10 nor by any result on this page.
- *Other symmetry groups.* The proofs use the non-compactness of the boosts. For the de Sitter group $$SO(4,1)$$ acting on de Sitter space, the analogue of Theorem 2 would concern positive-definite invariant kernels on $$dS_4$$ with finite diagonal; whether a non-constant such kernel exists is not decided here.
- *States that are not locally normal.* Without (H2) the single-mode reduction in Proposition 3 is not available.

## 8. Open points

The classification of invariant states under (H1)–(H3) is complete. The following remain open on this page.

1. Whether an invariant, hyperuniform classical point process exists, and, if so, how its correlations decay along the null cone (Corollary 8 shows they cannot be integrable).
2. Whether a non-constant invariant positive-definite kernel with finite diagonal exists on de Sitter space.
3. Whether, among classical invariant simple point processes with second moments, the mixed Poisson processes are characterised by the independence of counts in regions that are far apart, a property that Fock states have and the random lattice lacks.

## References

1. <a id="ref-bhs"></a>Bombelli, Henson, Sorkin, "Discreteness without symmetry breaking: a theorem," Mod. Phys. Lett. A 24 (2009) 2579, [arXiv:gr-qc/0605006](https://arxiv.org/abs/gr-qc/0605006). Used for: no equivariant choice of a preferred direction exists on a Poisson sprinkling.
2. <a id="ref-dhs"></a>Dowker, Henson, Sorkin, "Quantum gravity phenomenology, Lorentz invariance and discreteness," Mod. Phys. Lett. A 19 (2004) 1829, [arXiv:gr-qc/0311055](https://arxiv.org/abs/gr-qc/0311055). Used for: local Lorentz invariance of the discreteness of a sprinkled causal set.
3. <a id="ref-sa"></a>Saravani, Aslanbeigi, "On the causal set-continuum correspondence," [arXiv:1403.6429](https://arxiv.org/abs/1403.6429). Used for: Poisson sprinkling is optimal for the number-volume correspondence in arbitrarily small regions; two-dimensional Lorentzian lattices do better in large regions.
4. <a id="ref-hp"></a>Hudson, Parthasarathy, "Quantum Itô's formula and stochastic evolutions," Commun. Math. Phys. 93 (1984) 301–323. Used for: a Poisson process is a coherent state.
5. <a id="ref-surya"></a>Surya, "The causal set approach to quantum gravity," Living Rev. Relativ. (2019), [doi:10.1007/s41114-019-0023-1](https://doi.org/10.1007/s41114-019-0023-1), [arXiv:1903.11544](https://arxiv.org/abs/1903.11544). Review of causal sets and sprinklings.
6. <a id="ref-sorkin"></a>Sorkin, "Causal sets: discrete gravity (notes for the Valdivia summer school)," [arXiv:gr-qc/0309009](https://arxiv.org/abs/gr-qc/0309009). Introduction to causal sets.
7. <a id="ref-rs2"></a>Reed, Simon, *Methods of Modern Mathematical Physics II: Fourier Analysis, Self-Adjointness*, Academic Press, 1975, section IX.2. Used for: the Bochner–Schwartz theorem (a positive-definite tempered distribution is the Fourier transform of a positive tempered measure).
8. <a id="ref-soshnikov"></a>Soshnikov, "Determinantal random point fields," Russian Math. Surveys (2000), [doi:10.1070/RM2000v055n05ABEH000321](https://doi.org/10.1070/RM2000v055n05ABEH000321), [arXiv:math/0002099](https://arxiv.org/abs/math/0002099). Used for: determinantal processes with locally trace-class Hermitian kernels.
9. <a id="ref-siegel"></a>Siegel, "A mean value theorem in geometry of numbers," Ann. of Math. 46 (1945) 340, [doi:10.2307/1969027](https://doi.org/10.2307/1969027). Used for: the mean value theorem for lattice sums over the space of unimodular lattices and the finiteness of its invariant measure.
10. <a id="ref-moore"></a>Moore, "Ergodicity of flows on homogeneous spaces," Amer. J. Math. 88 (1966) 154, [doi:10.2307/2373052](https://doi.org/10.2307/2373052). Used for: ergodicity of closed non-compact subgroups on the quotient of a simple group by a lattice.
11. <a id="ref-thomas"></a>Thomas, "Projective limits of complex measures and martingale convergence," Probab. Theory Related Fields 119 (2001) 579–588, [doi:10.1007/pl00008772](https://doi.org/10.1007/pl00008772). Used for: the martingale argument that extends a bounded-variation projective family of complex measures.
12. <a id="ref-am"></a>Albeverio, Mazzucchi, "A unified approach to infinite-dimensional integration," Rev. Math. Phys. 28 (2016) 1630001, [arXiv:1411.2853](https://arxiv.org/abs/1411.2853). Used for: bounded variation as the condition for extending a projective family of complex measures.
13. <a id="ref-parthasarathy"></a>Parthasarathy, *Probability Measures on Metric Spaces*, Academic Press, 1967, Theorem V.3.2. Used for: Kolmogorov extension of projective systems of Borel measures on standard Borel spaces.
14. <a id="ref-varadarajan"></a>Varadarajan, "Groups of automorphisms of Borel spaces," Trans. Amer. Math. Soc. 109 (1963) 191–220, [doi:10.1090/s0002-9947-1963-0159923-5](https://doi.org/10.1090/s0002-9947-1963-0159923-5). Used for: the ergodic decomposition of an invariant measure.
15. <a id="ref-zimmer"></a>Zimmer, *Ergodic Theory and Semisimple Groups*, Birkhäuser, 1984. Used for: an almost-everywhere invariant function for an action of a second-countable locally compact group has a strictly invariant version.
