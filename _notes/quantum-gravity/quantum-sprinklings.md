---
title: Lorentz-invariant quantum sprinklings are condensates
date: 2026-10-29
category: substrate
order: 1
description: A Poincaré-invariant, locally normal state of a bosonic field on spacetime with a finite mean number of quanta in every bounded region is a mixture of coherent states of constant amplitude, so its counts are Poisson with a random intensity that is the same everywhere; the fermionic and determinantal analogues with finite density are trivial, and invariant classical processes need not be Poisson.
status: "Proved: Theorems 2 and 4 (bosonic Weyl algebra, hypotheses (H1)–(H3)); Theorem 6 (CAR algebra, translations, boosts, finite density on a ball); Theorem 7 (determinantal processes invariant under translations and one boost subgroup); Theorem 10 (complex processes of bounded total variation); Proposition 11 (continuous invariant kernels on de Sitter space). Computed: void probabilities of random lattices in two dimensions."
excludes: "Under (H1)–(H3): bosonic states that are not mixtures of coherent condensates, including sub-Poisson, number-squeezed and Fock-number states of non-zero density. Fermionic CAR states other than the vacuum that are invariant under translations and boosts with finite density on a ball. Non-empty determinantal processes invariant under translations and one boost subgroup."
---

A sprinkling of Minkowski space is a random discrete set of points whose law is invariant under the Poincaré group; the standard one is the Poisson process, whose number variance in a region equals its mean. We ask whether a quantum field on spacetime, with the quanta as the points, can be in an invariant state of finite density and count more regularly than Poisson, with a number variance below the mean or a fixed number of quanta in a region. For a bosonic field with wave functions on spacetime the answer is no, if the state is locally normal (each bounded region has a density matrix) and the mean number of quanta in every bounded region is finite (section 1.4). Every such state is a mixture of coherent states, each with one constant complex wave function $$z$$. Its counts are Poisson with random intensity $$\lvert z\rvert ^2$$, the same at every point, so the variance is at least the mean (Theorem 4).

Poisson sprinklings are the standard model of a causal set [[Surya]](#ref-surya), [[Sorkin]](#ref-sorkin). Bombelli, Henson and Sorkin [[BHS]](#ref-bhs) prove that no Lorentz-equivariant choice of a preferred direction at a point of a Poisson sprinkling exists. Dowker, Henson and Sorkin [[DHS]](#ref-dhs) explain why a boosted regular lattice is not a sprinkling with the same law as the original, and Saravani and Aslanbeigi [[SA]](#ref-sa) show that Poisson sprinkling gives the best number-volume correspondence in arbitrarily small regions, while two-dimensional Lorentzian lattices do better in large ones. Hudson and Parthasarathy [[HP]](#ref-hp) realise Poisson processes inside Fock space. Theorem 4 classifies all invariant, locally normal states of finite density in that setting, including the phase of the condensate.

The same boost argument leaves only the vacuum for fermions (Theorem 6) and only the empty process for determinantal point processes, the classical model of a fermionic sprinkling (Theorem 7). The classical analogue of the bosonic statement fails, since Haar-random lattices are invariant, ergodic and not Poisson (Proposition 9). A point process with complex weights of bounded total variation is a combination of classical ergodic processes (Theorem 10). On [Mixed-Poisson and signed substrates and the four-dimensional d'Alembertian](/quantum-gravity/signed-substrates/), the mean of the causal-set d'Alembertian on a mixed Poisson process of non-constant intensity, normalised by the mean density, tends at high density to a factor below one times the continuum d'Alembertian in four dimensions. Theorem 1 of [Integer counts and finite-dimensional algebras in a positive-energy vacuum](/quantum-gravity/count-algebras/), applied to the counts of a condensate in the Remark after Corollary 5, shows that such a state is not a positive-energy vacuum.

## 1. Sprinklings, counts and the setting

### 1.1 Classical counts

Let $$M=\mathbb R^{1,3}$$ with Lebesgue measure $$dV$$, let $$V(D)$$ be the volume of a bounded Borel set $$D$$, and let $$P$$ be the proper orthochronous Poincaré group. The same arguments work in $$\mathbb R^{1,d-1}$$ for every $$d\ge 2$$. The signature is $$(-,+,+,+)$$, so the Minkowski pairing of a momentum $$p$$ and a point $$x$$ is $$p\cdot x=-p^0x^0+\vec p\cdot\vec x$$. A *sprinkling* is a random locally finite subset of $$M$$, and a causal set is obtained by keeping the points and the causal order between them. Everything below depends only on the counts $$N(D)$$, the number of points in $$D$$. The Poisson process of intensity $$\lambda$$ has $$N(D)$$ Poisson with mean $$\lambda V(D)$$ and independent counts in disjoint sets. Its mean and variance are both $$\lambda V(D)$$, and its law is invariant under $$P$$ because Lebesgue measure is.

### 1.2 The quantum version

Take the one-particle space to be $$L^2(M,dV)$$: a one-particle state is a wave function on spacetime, and the points of the sprinkling are the positions of the quanta. Let $$\Gamma(h)$$ be the bosonic Fock space over a Hilbert space $$h$$, with annihilation operators $$a(f)$$ (antilinear in $$f$$), creation operators $$a^*(f)$$ and $$[a(f),a^*(g)]=\langle f,g\rangle$$, where the inner product is antilinear in the first slot. The Weyl operators $$W(f)=\exp(a^*(f)-a(f))$$ satisfy $$W(f)W(g)=e^{-i\,\mathrm{Im}\langle f,g\rangle}W(f+g)$$. For a bounded Borel set $$D$$ of positive volume let $$L^2(D)\subset L^2(M)$$ be the functions vanishing outside $$D$$, let $$\mathfrak A(D)$$ be the C$$^*$$-algebra generated by $$W(f)$$ with $$f\in L^2(D)$$, and let $$\mathfrak A$$ be the algebra generated by all of them. The group $$P$$ acts by automorphisms $$\alpha_gW(f)=W(f\circ g^{-1})$$. The count of $$D$$ is the number operator of $$\Gamma(L^2(D))$$,

$$
N(D)=d\Gamma(1_D)=\sum_k a^*(e_k)a(e_k), \tag{1}
$$

for any orthonormal basis $$(e_k)$$ of $$L^2(D)$$. It has spectrum $$\{0,1,2,\dots\}$$, and for disjoint $$D_1,D_2$$ the Fock space of $$D_1\cup D_2$$ is the tensor product of those of $$D_1$$ and $$D_2$$, so counts in disjoint sets commute. This is a *sharp* count: it assigns an integer to every region and is not smeared against a profile.

### 1.3 Coherent states are Poisson processes

For $$z\in\mathbb C$$ let $$\omega_z$$ be the state on $$\mathfrak A(D)$$ given by the normalised exponential vector $$e^{-\lvert z\rvert ^2V(D)/2}e(z1_D)$$ of $$\Gamma(L^2(D))$$. Then

$$
\omega_z(W(f))=\exp\!\big(-\tfrac12\lVert f\rVert ^2+2i\,\mathrm{Im}(\bar z\!\int\! f)\big),\qquad f\in L^2(D), \tag{2}
$$

which is consistent in $$D$$ and extends to every $$f\in L^1\cap L^2$$. The operator $$e^{itN(D)}$$ maps the exponential vector $$e(g)$$ to $$e(e^{it}g)$$, so $$\omega_z(e^{itN(D)})=\exp\big(\lvert z\rvert ^2V(D)(e^{it}-1)\big)$$, the generating function of a Poisson variable of mean $$\lvert z\rvert ^2V(D)$$. The exponential vector also factorises over a partition of $$D$$, with a coherent state in each factor, so counts of disjoint sets are independent. Hence $$\omega_z$$ is a Poisson sprinkling of intensity $$\lvert z\rvert ^2$$. It is Poincaré invariant, since $$\lVert f\circ g^{-1}\rVert =\lVert f\rVert $$ and $$\int f\circ g^{-1}=\int f$$. It is not a vector in the Fock space of $$L^2(M)$$ when $$z\ne0$$, because the constant $$z\,1_M$$ is not square integrable. For this reason we work with the local algebras $$\mathfrak A(D)$$ and not with the Weyl algebra over $$L^2(M)$$, on which $$\int f$$ is undefined. Working over $$L^1\cap L^2$$ is possible, but then the state on functions of unbounded support is an additional datum that the counts do not see.

### 1.4 Hypotheses

A state $$\omega$$ on $$\mathfrak A$$ is a *quantum sprinkling* if it satisfies:

(H1) *Invariance.* $$\omega\circ\alpha_g=\omega$$ for all $$g\in P$$.

(H2) *Local normality.* For every bounded $$D$$ there is a density matrix $$\varrho_D$$ on $$\Gamma(L^2(D))$$ with $$\omega(A)=\mathrm{tr}(\varrho_D A)$$ for $$A\in\mathfrak A(D)$$.

(H3) *Finite density.* $$\mathrm{tr}(\varrho_D N(D))<\infty$$ for every bounded $$D$$. We write $$\omega(N(D))$$ for this number.

Local normality makes $$t\mapsto\omega(W(tf))$$ continuous, so the field operators $$a(f)$$ exist. The hypothesis (H3) is used in Theorem 2 to make a certain measure finite; section 8 discusses what happens without it. The one-body density is the operator $$T$$ on $$L^2(M)$$ with

$$
\langle f,Tg\rangle=\omega\big(a^*(g)\,a(f)\big),\qquad f,g\in L^2_c(M), \tag{3}
$$

where $$L^2_c$$ is the set of square-integrable functions with bounded support. The right side is finite: (H3) puts the range of $$\varrho_D^{1/2}$$ in the domain of $$N(D)^{1/2}$$, and $$\lVert a(f)\psi\rVert \le\lVert f\rVert \,\lVert N(D)^{1/2}\psi\rVert $$. For a coherent state, (3) reads $$\langle f,Tg\rangle=\lvert z\rvert ^2\,\overline{\textstyle\int f}\int g$$, a rank-one operator with constant kernel. Theorem 2 shows that this is the only possibility.

## 2. The one-body density is a constant

<div class="lemma" markdown="1">

**Lemma 1** (wandering sets). Let $$\mu$$ be a finite positive Borel measure on $$\mathbb R^n$$, and let $$B$$ be a linear map of determinant $$\pm1$$ with $$B_*\mu=\mu$$. If $$Y$$ is a Borel set such that the sets $$B^kY$$, $$k\in\mathbb Z$$, are pairwise disjoint, then $$\mu(Y)=0$$. The same holds for a function $$G\in L^1(\mathbb R^n)$$ with $$G\circ B=G$$ almost everywhere, with $$\int_Y\lvert G\rvert =0$$ in place of $$\mu(Y)=0$$, and then $$G=0$$ almost everywhere on $$\bigcup_kB^kY$$.

</div>

<div class="proof" markdown="1">

*Proof.* The sets $$B^kY$$ all have the same measure, and they are disjoint, so $$\sum_k\mu(B^kY)\le\mu(\mathbb R^n)<\infty$$ forces $$\mu(Y)=0$$. For $$G$$ the same argument applies to $$\int_{B^kY}\lvert G\rvert =\int_Y\lvert G\circ B^k\rvert =\int_Y\lvert G\rvert $$, which uses that $$B$$ preserves Lebesgue measure. $$\square$$

</div>

A boost with rapidity $$\theta$$ along $$x^1$$ acts on momenta in the null coordinates $$p^\pm=p^0\pm p^1$$ as $$(p^+,p^-)\mapsto(e^\theta p^+,e^{-\theta}p^-)$$ and fixes $$p^2,p^3$$. The sets $$Y=\{1\le\lvert p^+\rvert <e^{\theta}\}$$ and $$Y'=\{1\le\lvert p^-\rvert <e^\theta\}$$ are wandering, and their images cover $$\{p^+\ne0\}$$ and $$\{p^-\ne0\}$$. A finite boost-invariant measure is therefore carried by $$\{p^+=p^-=0\}$$, and this is what makes the one-body density constant. The same wandering-set argument, for a group containing a hyperbolic element, is Lemma 1 of [Lorentz-invariant point sets and the structures they cannot carry](/quantum-gravity/invariant-point-sets/).

<div class="theorem" markdown="1">

**Theorem 2** (constant one-body density). Under (H1)–(H3) there is a number $$\rho\ge0$$ with

$$
\omega\big(a^*(g)a(f)\big)=\rho\,\overline{\textstyle\int f}\,\textstyle\int g\qquad\text{for all } f,g\in L^2_c(M), \tag{4}
$$

and $$\omega(N(D))=\rho\,V(D)$$ for every bounded $$D$$.

</div>

Translation invariance writes the density as a positive measure $$\mu$$ on momentum space, finite density makes the total mass of $$\mu$$ finite, and a finite boost-invariant measure sits at the origin.

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

*Step 3: boosts.* A boost $$B$$ acts on test functions by $$f\mapsto f\circ B^{-1}$$ and on transforms by $$\hat f\mapsto\hat f\circ B^{-1}$$, since the pairing is invariant. Invariance (H1) of $$\tau$$ therefore says that $$\mu$$ is invariant under $$B$$. Lemma 1 for the boost along $$x^1$$ with the wandering sets above gives $$\mu(p^+\ne0)=\mu(p^-\ne0)=0$$, so $$\mu$$ lives on $$\{p^0=p^1=0\}$$. The boosts along $$x^2$$ and $$x^3$$ give $$\{p^0=p^2=0\}$$ and $$\{p^0=p^3=0\}$$, and the intersection is the origin. Thus $$\mu=\rho\,\delta_0$$ with $$\rho=\mu(\mathbb R^4)$$, and (5) becomes (4) because $$\hat f(0)=\int f$$. Since $$\mu$$ is finite, dominated convergence extends (5) from $$C_c^\infty$$ to $$f,g\in L^2_c$$ (approximate in $$L^2$$ and in $$L^1$$ simultaneously, so that $$\hat f_n\to\hat f$$ boundedly), which gives (4) as stated. The Parseval computation of Step 2 with an orthonormal basis of $$L^2(D)$$ then gives $$\omega(N(D))=\rho V(D)$$ for every bounded $$D$$. $$\square$$

</div>

The proof uses translations and boosts only, not rotations and not positivity of the energy. With translations alone any finite measure $$\mu$$ is possible (section 8).

## 3. Reduction to a single mode

Fix a bounded set $$D$$ of positive volume and let $$u_D=1_D/\sqrt{V(D)}$$ be the normalised constant on $$D$$. By (4), $$a(f)$$ has zero second moment in $$\omega$$ whenever $$\int f=0$$. Proposition 3 turns this into a statement about the state: all quanta of $$D$$ are in the single mode $$u_D$$, and the single-mode states of nested regions are related by loss. The relation (6) is what lets us pass from small regions to large ones in section 4.

<div class="proposition" markdown="1">

**Proposition 3** (single-mode reduction and attenuation). Assume (H1)–(H3). Then:

(a) For every bounded $$D$$ of positive volume, $$\Gamma(L^2(D))=\Gamma(\mathbb Cu_D)\otimes\Gamma(u_D^\perp)$$ and $$\varrho_D=\sigma_D\otimes\lvert \Omega\rangle\langle\Omega\rvert $$, where $$\sigma_D$$ is a density matrix on the one-mode space $$\Gamma(\mathbb Cu_D)\cong\ell^2(\mathbb N_0)$$ and $$\Omega$$ is the vacuum of $$\Gamma(u_D^\perp)$$. In particular $$N(D)=a^*(u_D)a(u_D)$$ on the support of $$\varrho_D$$.

(b) If $$D\subset D'$$ and $$\eta=V(D)/V(D')$$, then $$\sigma_D$$ is the image of $$\sigma_{D'}$$ under the pure-loss channel of transmissivity $$\eta$$, equivalently

$$
\Phi_D(c)=\Phi_{D'}(\sqrt\eta\,c),\qquad \Phi_D(c):=\mathrm{tr}\big(\sigma_D\,e^{c a^\dagger}e^{-\bar c a}\big), \tag{6}
$$

where $$a=a(u_D)$$ on the left and $$a=a(u_{D'})$$ on the right.

</div>

In (a) a mode orthogonal to the constant carries no quanta. In (b) the constant on $$D$$ is a mixture of the constant on $$D'$$, with weight $$\sqrt\eta$$, and a function orthogonal to the constant on $$D'$$, which carries no quanta, so the mode $$u_D$$ sees a fraction $$\eta$$ of the quanta of the mode $$u_{D'}$$.

<div class="proof" markdown="1">

*Proof.* (a) Let $$f\in L^2(D)$$ with $$\int f=0$$, that is, $$f\perp u_D$$. By (4), $$\mathrm{tr}(\varrho_Da^*(f)a(f))=0$$, which is $$\lVert a(f)\varrho_D^{1/2}\rVert ^2_{HS}=0$$, so the support of $$\varrho_D$$ lies in $$\ker a(f)$$. The common kernel of $$a(f)$$ over $$f\in u_D^\perp$$ is $$\Gamma(\mathbb Cu_D)\otimes\Omega$$, the only vectors of $$\Gamma(u_D^\perp)$$ annihilated by all $$a(f)$$ being multiples of the vacuum.

(b) Write $$u_D=\sqrt\eta\,u_{D'}+w$$ with $$w=u_D-\sqrt\eta\,u_{D'}\in L^2(D')$$. Then $$\langle u_{D'},w\rangle=0$$, $$\int w=0$$ and $$\lVert w\rVert ^2=1-\eta$$. For $$c\in\mathbb C$$ the Weyl operator $$W(cu_D)=W(c\sqrt\eta\,u_{D'})W(cw)$$ is a product of commuting factors, one in the mode $$u_{D'}$$ and one in $$u_{D'}^\perp$$. By (a) for $$D'$$,

$$
\omega(W(cu_D))=\mathrm{tr}\big(\sigma_{D'}W(c\sqrt\eta\,u_{D'})\big)\,e^{-(1-\eta)\lvert c\rvert ^2/2}.
$$

By (a) for $$D$$ the left side is $$\mathrm{tr}(\sigma_DW(cu_D))$$. Passing to the normally ordered function with $$W(cu)=e^{ca^\dagger}e^{-\bar ca}e^{-\lvert c\rvert ^2/2}$$ turns this into (6). $$\square$$

</div>

The relation (6) holds for every pair of nested regions, so the state of a mode in a small region is obtained from the state in a large region by loss, for all loss rates at once. Normally ordered moments scale as $$\langle a^{\dagger j}a^k\rangle_D=\eta^{(j+k)/2}\langle a^{\dagger j}a^k\rangle_{D'}$$ (with the appropriate mode on each side). Mandel's parameter $$Q_{\rm M}=(\langle a^{\dagger2}a^2\rangle-\langle a^\dagger a\rangle^2)/\langle a^\dagger a\rangle=(\mathrm{Var}\,N-\langle N\rangle)/\langle N\rangle$$ therefore obeys $$Q_{\rm M}(D)=\eta\,Q_{\rm M}(D')$$. Every state has $$Q_{\rm M}\ge-1$$, so $$Q_{\rm M}(D)\ge-V(D)/V(D')$$, and letting $$D'$$ grow gives $$Q_{\rm M}(D)\ge0$$. Whenever the second moments are finite, $$\mathrm{Var}\,N(D)\ge\langle N(D)\rangle$$ follows already at this point.

<div class="example" markdown="1">

**Example** (a number state cannot occur). Suppose the mode of a region has exactly one quantum, $$\sigma_D=\lvert 1\rangle\langle1\rvert $$. By (6) with $$\eta=1/k$$, $$k>1$$, the state $$\sigma_{D'}$$ of the region $$k$$ times larger has $$\Phi_{D'}(c)=1-k\lvert c\rvert ^2$$. The normally ordered function of the number state $$\lvert n\rangle$$ is the Laguerre polynomial $$L_n(\lvert c\rvert ^2)$$, so $$1-k\lvert c\rvert ^2=p_0L_0+p_1L_1$$ forces $$p_1=k>1$$ and $$p_0=1-k<0$$, which is not a state. For $$k=3/2$$ the weights are $$(p_0,p_1)=(-\tfrac12,\tfrac32)$$, and the pure-loss channel with $$\eta=2/3$$ maps this operator back to $$\lvert 1\rangle\langle 1\rvert $$. The obstruction is the positivity of the preimage.

</div>

## 4. Classification

### 4.1 Bosons

<div class="theorem" markdown="1">

**Theorem 4** (invariant quantum sprinklings are coherent condensates). Let $$\omega$$ satisfy (H1)–(H3). There is a unique Borel probability measure $$\mathsf m$$ on $$\mathbb C$$ with $$\int\lvert z\rvert ^2d\mathsf m(z)=\rho<\infty$$ such that

$$
\omega(A)=\int\omega_z(A)\,d\mathsf m(z),\qquad A\in\mathfrak A(D),\ D\ \text{bounded}. \tag{7}
$$

Conversely every such mixture satisfies (H1)–(H3). Consequently, for disjoint bounded sets $$D_1,\dots,D_k$$ the counts have the joint law

$$
\Pr\big(N(D_i)=n_i,\ i\le k\big)=\mathbb E\prod_{i}e^{-\Lambda V(D_i)}\frac{(\Lambda V(D_i))^{n_i}}{n_i!},\qquad \Lambda=\lvert z\rvert ^2, \tag{8}
$$

so $$N$$ is a Cox (mixed Poisson) process with a spacetime-constant random intensity $$\Lambda$$ of mean $$\rho$$, and $$\mathrm{Var}\,N(D)=\rho V(D)+V(D)^2\,\mathrm{Var}\,\Lambda\ \ge\ \mathbb E N(D)$$, with equality for all $$D$$ if and only if $$\Lambda=\rho$$ almost surely.

</div>

By (6) the one-mode state of a small region is obtained from the state of every larger region by losing each quantum independently with probability $$1-\eta$$ (a pure-loss channel). The Husimi function of a one-mode state is the probability density of the outcomes of a measurement in coherent states, so its Fourier transform is positive definite; it equals the normally ordered characteristic function $$\Phi_D$$ of (6) times a Gaussian whose width grows with the volume. In the limit of large regions the Gaussian drops out, and $$\Phi_D$$ becomes the Fourier transform of a positive measure on coherent amplitudes. This limit $$V(D)\to\infty$$ is where the content of the theorem lies.

<div class="proof" markdown="1">

*Proof.* Let $$\Psi(\zeta)=\Phi_D(\zeta/\sqrt{V(D)})$$. By (6) with $$c=\zeta/\sqrt{V(D)}$$ and $$\sqrt\eta\,c=\zeta/\sqrt{V(D')}$$, $$\Psi$$ is the same for nested regions, hence for all bounded regions (any two lie in a third). It is continuous with $$\Psi(0)=1$$, since $$\sigma_D$$ is a normal state.

The Husimi function $$q_D(\alpha)=\pi^{-1}\langle\alpha\lvert \sigma_D\rvert \alpha\rangle$$ is a probability density on $$\mathbb C$$. Its Fourier transform is the anti-normally ordered function $$\mathrm{tr}(\sigma_De^{-\bar\xi a}e^{\xi a^\dagger})=\Phi_D(\xi)e^{-\lvert \xi\rvert ^2}$$. A Fourier transform of a probability density is positive definite, so with $$\zeta=\sqrt{V(D)}\,\xi$$ the function $$\Psi(\zeta)e^{-\lvert \zeta\rvert ^2/V(D)}$$ is positive definite on $$\mathbb R^2\cong\mathbb C$$ for every $$V(D)\in(0,\infty)$$. Let $$V(D)\to\infty$$. A pointwise limit of positive-definite functions is positive definite, so $$\Psi$$ is a continuous positive-definite function with $$\Psi(0)=1$$, and Bochner's theorem gives a probability measure $$\mathsf m$$ on $$\mathbb C$$ with

$$
\Psi(\zeta)=\int e^{\bar z\zeta-z\bar\zeta}\,d\mathsf m(z).
$$

A coherent state $$\lvert \beta\rangle$$ has $$\mathrm{tr}(\lvert \beta\rangle\langle\beta\rvert e^{\xi a^\dagger}e^{-\bar\xi a})=e^{\xi\bar\beta-\bar\xi\beta}$$, and the normally ordered function determines the state, so $$\sigma_D=\int\lvert z\sqrt{V(D)}\rangle\langle z\sqrt{V(D)}\rvert \,d\mathsf m(z)$$. By (2) and Proposition 3(a), $$\lvert z\sqrt{V(D)}\rangle\otimes\Omega$$ is the exponential vector defining $$\omega_z$$ on $$\mathfrak A(D)$$, which gives (7). Uniqueness of $$\mathsf m$$ follows from that of the Fourier transform. Finite density gives $$\int\lvert z\rvert ^2d\mathsf m=\rho$$.

For the converse, (2) shows that each $$\omega_z$$, hence the mixture, is invariant; local normality and finite density are clear. The law (8) follows from the independence and Poisson counts under each $$\omega_z$$. The mean and variance of a mixed Poisson variable with random mean $$\Lambda V$$ are $$\mathbb E\Lambda V$$ and $$\mathbb E\Lambda V+\mathrm{Var}(\Lambda)V^2$$, which is $$+\infty$$ if $$\mathbb E\Lambda^2=\infty$$. $$\square$$

</div>

The counts determine only the law of $$\Lambda=\lvert z\rvert ^2$$, whereas the state also contains the phase of $$z$$. For $$\mathsf m=\delta_{z_0}$$ the state is a single coherent condensate. For $$\mathsf m$$ uniform on the circle $$\lvert z\rvert ^2=\rho$$ the counts are exactly Poisson and the state is the phase average of a condensate, whose counting statistics coincide with those of the classical Poisson sprinkling.

<div class="corollary" markdown="1">

**Corollary 5** (what cannot occur). Let $$\omega$$ satisfy (H1)–(H3) with $$\rho>0$$. Then no region has $$\mathrm{Var}\,N(D)<\mathbb E N(D)$$ (sub-Poisson or number-squeezed counts). No region of positive volume has a fixed number of quanta (a Fock number state of the mode $$u_D$$). The variance does not follow an area law, $$\mathrm{Var}\,N(D)\propto$$ the area of $$\partial D$$. The only fluctuation beyond Poisson is a random but spacetime-constant intensity, which contributes a term $$V^2\,\mathrm{Var}(\Lambda)$$ growing like the square of the volume.

</div>

The case $$\rho=0$$ is the vacuum, in which $$N(D)=0$$ with certainty. The variance is $$\rho V+V^2\mathrm{Var}\Lambda$$ for every region, with no term that depends on the boundary, which excludes an area law.

<div class="remark" markdown="1">

**Remark** (positive energy). Let $$\omega$$ be as in Theorem 4 with $$\mathbb E\Lambda^2<\infty$$. By local normality the counts (1) act in the GNS space of $$\omega$$ as commuting self-adjoint operators with integer spectrum, additive on disjoint sets and covariant under the unitary translations that implement (H1); the second moment of every count in the GNS vector is finite by the variance formula of Theorem 4. If the translations are strongly continuous in the GNS representation, these data satisfy the hypotheses (C1)–(C4) and (R) of [Integer counts and finite-dimensional algebras in a positive-energy vacuum](/quantum-gravity/count-algebras/), and its Theorem 1 gives $$\rho=0$$ whenever the spectrum of the translation generator lies in the closed forward cone. Hence a condensate with $$\rho>0$$ is not the vacuum of a positive-energy translation group.

</div>

### 4.2 Fermions

Replace the Weyl algebra by the CAR algebra generated by $$a(f)$$, $$f\in L^2_c(M)$$, with $$\{a(f),a^*(g)\}=\langle f,g\rangle$$. Every state of this algebra satisfies $$\lvert \omega(a^*(g)a(f))\rvert \le\lVert f\rVert \lVert g\rVert $$, because $$\lVert a(f)\rVert =\lVert f\rVert $$, so $$T$$ is a bounded operator with $$0\le T\le1$$, and no normality or regularity is needed. For a state we define $$\omega(N(D))=\sum_k\omega(a^*(e_k)a(e_k))\in[0,\infty]$$ for an orthonormal basis of $$L^2(D)$$, which does not depend on the basis.

<div class="theorem" markdown="1">

**Theorem 6** (no invariant fermionic sprinkling). Let $$\omega$$ be a state on the CAR algebra over $$L^2_c(M)$$ that is invariant under translations and boosts and has $$\omega(N(D))<\infty$$ for some open ball $$D$$. Then $$\omega$$ is the Fock vacuum: $$\omega(N(D))=0$$ for every $$D$$ and $$\omega(a^*(f)a(g))=0$$.

</div>

<div class="proof" markdown="1">

*Proof.* Theorem 2 uses nothing about the statistics except the bound on $$\tau$$, which here comes from $$\lVert a(f)\rVert =\lVert f\rVert $$. Steps 1 to 3 give $$\tau(f,g)=\rho\,\overline{\int f}\int g$$, with finite $$\rho$$ since $$\omega(N(D))=V(D)\mu(\mathbb R^4)<\infty$$ (Step 2 applies to the one ball $$D$$ for which the density is finite, and the extension of (4) from $$C_c^\infty$$ to $$L^2_c$$ is by continuity, as in the proof of Theorem 2). Now take $$f=1_{D'}$$ for any ball $$D'$$: $$\tau(f,f)=\omega(a^*(f)a(f))\le\lVert f\rVert ^2$$ gives $$\rho V(D')^2\le V(D')$$, that is $$\rho\le1/V(D')$$. Since $$D'$$ can have arbitrarily large volume, $$\rho=0$$. Then $$\omega(a^*(f)a(f))=0$$ for all $$f\in L^2_c$$, and $$\omega(N(D))=0$$ for every bounded $$D$$. For any $$B$$ in the algebra, Cauchy–Schwarz gives $$\lvert \omega(Ba(f))\rvert ^2\le\omega(BB^*)\,\omega(a^*(f)a(f))=0$$, and by taking adjoints $$\omega(a^*(f)B)=0$$. Hence $$\omega$$ vanishes on every normally ordered monomial $$a^*(f_1)\cdots a^*(f_m)a(g_1)\cdots a(g_n)$$ with $$m+n\ge1$$. These monomials and the identity span the algebra, so $$\omega$$ is the vacuum state. $$\square$$

</div>

For fermions the bound $$T\le1$$ rules out the constant density that the boost argument allows: on a region of volume $$V(D')$$ the constant density gives $$\rho V(D')^2$$ for the smeared number of the function $$1_{D'}$$, while $$T\le1$$ allows at most $$V(D')$$. Poincaré-invariant quasi-free CAR states with $$0\le T\le1$$ whose single-mode numbers are sub-Poisson do exist, namely those whose $$T$$ is a bounded invariant Fourier multiplier, but they have infinite density (section 8).

## 5. Classical determinantal sprinklings

The point process of a quasi-free fermionic state is determinantal, and the classical statement is simpler and needs no Fock space. A point process on $$M$$ is *determinantal* with Hermitian kernel $$\mathcal K$$ if $$\mathcal K$$ is a self-adjoint operator on $$L^2(M)$$ with $$0\le\mathcal K\le1$$ and $$1_D\mathcal K1_D$$ trace class for bounded $$D$$, and the process has correlation functions $$\rho_n(x_1,\dots,x_n)=\det[K(x_i,x_j)]$$, where $$K(x,y)$$ is a measurable kernel of $$\mathcal K$$ with $$\int_DK(x,x)dx=\mathrm{tr}(1_D\mathcal K1_D)$$ [[Soshnikov]](#ref-soshnikov). In particular $$\rho_1(x)=K(x,x)$$ and $$\rho_2(x,y)=\rho_1(x)\rho_1(y)-\lvert K(x,y)\rvert ^2$$.

<div class="theorem" markdown="1">

**Theorem 7** (no invariant determinantal process). Let $$\Xi$$ be a determinantal point process on $$\mathbb R^{1,d-1}$$, $$d\ge2$$, whose law is invariant under translations and under one boost subgroup. Then $$\Xi$$ is empty almost surely.

</div>

<div class="proof" markdown="1">

*Proof.* Invariance of the law makes $$\rho_1$$ invariant under all translations almost everywhere, hence equal to a constant $$\rho$$ almost everywhere (by Fubini, a locally integrable function that is invariant a.e. under every translation is a.e. constant). Likewise $$\rho_2(x+a,y+a)=\rho_2(x,y)$$ gives $$\lvert K(x,y)\rvert ^2=G(x-y)$$ for a measurable $$G\ge0$$, and invariance under a boost $$B$$ gives $$G\circ B=G$$ almost everywhere.

Next, $$\mathcal K^2\le\mathcal K$$. For bounded $$D$$ the operator $$\mathcal K1_D$$ is Hilbert–Schmidt, with $$\lVert \mathcal K1_D\rVert _{HS}^2=\mathrm{tr}(1_D\mathcal K^2 1_D)\le\mathrm{tr}(1_D\mathcal K1_D)=\rho V(D)$$. The left side is $$\int\!\!\int\lvert K(x,y)\rvert ^21_D(y)\,dx\,dy=V(D)\int G$$ by the form of $$\lvert K\rvert ^2$$. Hence $$G\in L^1(\mathbb R^d)$$ with $$\int G\le\rho$$. Lemma 1, applied to the wandering sets of the boost $$B$$ (the regions $$e^{k\theta}\le\lvert r^0+r^1\rvert <e^{(k+1)\theta}$$ in the difference variable $$r=x-y$$), gives $$G=0$$ almost everywhere. Then $$K=0$$ almost everywhere, so $$\mathcal K1_D$$ has zero Hilbert–Schmidt norm for every $$D$$ and $$\mathcal K=0$$. Finally $$\rho V(D)=\mathrm{tr}(1_D\mathcal K1_D)=0$$, so $$\Xi=\emptyset$$. $$\square$$

</div>

Hilbert–Schmidt control of $$\mathcal K1_D$$ replaces the pointwise inequality $$(\mathcal K^2)(x,x)\le K(x,x)$$, which would need a choice of the diagonal values of the kernel. The same lemma gives a statement about hyperuniformity, the property that the number variance grows more slowly than the volume, $$\mathrm{Var}\,N(D)/V(D)\to0$$ along a growing sequence of regions.

<div class="corollary" markdown="1">

**Corollary 8** (pair correlations in $$L^1$$). Let $$\Xi$$ be a simple point process on $$\mathbb R^{1,d-1}$$ with locally integrable intensity $$\rho_1=\rho$$ and pair correlation $$\rho_2(x,y)=\rho^2g(x-y)$$, invariant under translations and one boost subgroup. If $$g-1\in L^1(\mathbb R^d)$$ then $$g=1$$ almost everywhere, and $$\mathrm{Var}\,N(D)=\rho V(D)$$ for every bounded $$D$$. In particular such a process is not hyperuniform.

</div>

<div class="proof" markdown="1">

*Proof.* $$g-1$$ is an integrable boost-invariant function, so Lemma 1 gives $$g-1=0$$ almost everywhere. Then $$\mathrm{Var}\,N(D)=\int_D\rho_1+\int_D\!\!\int_D(\rho_2-\rho_1\otimes\rho_1)=\rho V(D)$$. $$\square$$

</div>

The hypothesis $$g-1\in L^1$$ is what hyperuniformity normally implies, through the sum rule $$1+\rho\int(g-1)=0$$. A hyperuniform process invariant under translations and one boost subgroup must therefore have pair correlations that are not integrable.

## 6. Classical lattices and complex weights

Sections 2 to 4 use the Fock structure of the counts and the positivity of the state. Without the Fock structure there are invariant classical processes that are not mixed Poisson (section 6.1). Without positivity, a process with complex weights of bounded total variation is a combination of classical ergodic processes (section 6.2).

### 6.1 Haar-random lattices are invariant and not Poisson

Let $$d\ge2$$, let $$X=ASL(d,\mathbb R)/ASL(d,\mathbb Z)$$ with its invariant probability measure (finite by Siegel's theorem [[Siegel]](#ref-siegel)), and let $$\mathcal L=g\mathbb Z^d+\xi$$ be the random affine lattice of density 1 with $$(g,\xi)$$ distributed by this measure. Concretely, $$g$$ is distributed by the invariant probability measure on $$SL(d,\mathbb R)/SL(d,\mathbb Z)$$ and, given $$g$$, $$\xi$$ is uniform in the torus $$\mathbb R^d/g\mathbb Z^d$$. A typical lattice of this ensemble has no unusually short vector, but the measure gives positive weight to lattices with arbitrarily short ones, which consist of long, widely separated rows of closely spaced points. These produce the heavy tail of the counts seen below.

<div class="proposition" markdown="1">

**Proposition 9** (Haar-random lattices). The point process $$\mathcal L$$ on $$\mathbb R^{1,d-1}$$ has the following properties.

(a) Its law is invariant under $$P$$ and under all of $$ASL(d,\mathbb R)$$.

(b) For every bounded $$D$$: $$\mathbb E N(D)=V(D)$$ and $$\mathbb E[N(D)(N(D)-1)]=V(D)^2$$, so $$\mathrm{Var}\,N(D)=\mathbb E N(D)$$. The pair correlation is $$\rho_2\equiv1$$.

(c) $$\mathcal L$$ is not a Cox process directed by any random measure without atoms; in particular it is neither Poisson nor mixed Poisson.

(d) $$\mathcal L$$ is ergodic under $$P$$ [[Moore]](#ref-moore).

</div>

The proof is Theorem 1 of [Lorentz-invariant point sets and the structures they cannot carry](/quantum-gravity/invariant-point-sets/), which also describes the construction. It rests on three inputs. For (b), averaging over $$\xi$$ and applying Siegel's mean value theorem [[Siegel]](#ref-siegel) gives $$\mathbb E[N(N-1)]=V(D)^2$$. For (c), $$x\ne y\in\mathcal L$$ forces $$2y-x\in\mathcal L\setminus\{x,y\}$$, so the triples $$(x,y,2y-x)$$ with $$x,y$$ in a bounded set $$A$$ number at least $$N(A)(N(A)-1)$$, with expectation $$V(A)^2>0$$, while a Cox process with non-atomic directing measure gives zero probability to a third point at $$2y-x$$. For (d), Moore's theorem [[Moore]](#ref-moore) applies to $$SO^+(1,d-1)$$ acting on $$SL(d,\mathbb R)/SL(d,\mathbb Z)$$.

Proposition 9 shows that "Lorentz invariance and finite density force Poisson" is false for classical point processes. In a Fock state the counts in disjoint regions are independent in every coherent component (section 1.3), whereas in a lattice they are rigidly related: the pair $$x,y$$ forces the point $$2y-x$$. The second-order statistics of the random lattice coincide with those of a Poisson process ($$\rho_2\equiv1$$), so they cannot detect the difference, and neither can Corollary 8. The Lorentzian lattices of [[SA]](#ref-sa) have smaller fluctuations of the number-volume correspondence in large regions, but are invariant only under a discrete group.

Void probabilities do separate the random lattice from a Poisson process. In two dimensions the law of $$N(D)$$ is the same for every parallelogram $$D$$ of area $$V$$, because the lattice law is invariant under $$SL(2,\mathbb R)$$ and translations, so it suffices to count the points in the square $$[0,a]^2$$ with $$V=a^2$$. We drew $$1.6\times10^{7}$$ lattices (eight batches of $$2\times10^6$$), with the shape distributed by the hyperbolic measure on the modular fundamental domain, a uniform rotation, covolume 1 and $$\xi$$ uniform. The entries are means over the batches, with the batch-to-batch standard error in the last digits in parentheses.

| $$V$$ | $$\Pr(N=0)$$ | Poisson $$e^{-V}$$ | $$V\cdot\Pr(N=0)$$ |
|---|---|---|---|
| 1 | 0.30378 (9) | $$3.7\times10^{-1}$$ | 0.304 |
| 4 | 0.05432 (5) | $$1.8\times10^{-2}$$ | 0.217 |
| 16 | 0.01286 (3) | $$1.1\times10^{-7}$$ | 0.206 |

The void probability decays roughly like $$1/V$$ and not exponentially: the product $$V\Pr(N=0)$$ stays near 0.2 for $$V\ge4$$, while $$e^{-V}$$ falls by five orders of magnitude between $$V=4$$ and $$V=16$$. A heuristic accounts for the power law. A lattice avoids the square when its points lie on widely spaced rows, which requires a lattice vector of length at most of order $$1/a$$, and the invariant measure of such lattices is of order $$a^{-2}=1/V$$. The same short vectors make the tail of the counts heavy. A lattice with a short vector has rows of order $$n$$ points in the square, and the data give $$\Pr(N>n)$$ of order $$n^{-3}$$, so that the variance of $$N$$ is finite and its third moment is infinite. No formula for the constant 0.2 is available.

### 6.2 Complex weights of bounded total variation

A complex measure on configurations is a generalisation of a sprinkling in which the weights of configurations may be negative or complex; such weights occur in the study of the causal-set d'Alembertian on [Mixed-Poisson and signed substrates and the four-dimensional d'Alembertian](/quantum-gravity/signed-substrates/). A *complex point process* (CPP) is a family $$\{\mathsf P_D\}$$, indexed by bounded Borel sets, of complex measures on the space $$\Gamma_D$$ of finite counting measures on $$D$$, with $$\mathsf P_D(\Gamma_D)=1$$, finite total variation $$\lVert \mathsf P_D\rVert <\infty$$, and consistency: the image of $$\mathsf P_{D'}$$ under restriction to $$D\subset D'$$ is $$\mathsf P_D$$. It is *Poincaré invariant* if $$\mathsf P_{gD}=g_*\mathsf P_D$$. A positive process has $$\lVert \mathsf P_D\rVert =1$$. Positivity entered the proof of Theorem 4 through the Husimi function, and a complex process does not have it. Theorem 10 describes the invariant complex processes whose total variation is bounded: they are combinations, with complex phases, of classical ergodic invariant processes, and they are classical when the bound is 1.

<div class="theorem" markdown="1">

**Theorem 10** (bounded total variation: classical ergodic components). Let $$\mathsf P$$ be a Poincaré-invariant CPP with $$S=\sup_D\lVert \mathsf P_D\rVert <\infty$$. Then $$\mathsf P$$ extends to a complex measure on the space $$\Gamma_M$$ of locally finite configurations, and

$$
\mathsf P=S\int e^{i\varphi(s)}\,R_s\,\kappa(ds), \tag{9}
$$

where $$\kappa$$ is a probability measure, each $$R_s$$ is an ergodic Poincaré-invariant classical (positive) point process, and $$\varphi$$ is a real measurable function with $$S\int e^{i\varphi}d\kappa=1$$. In particular, if $$S=1$$ then $$\varphi=0$$ almost everywhere and $$\mathsf P$$ is a classical point process.

</div>

The total variations of the restrictions define a positive invariant measure $$\nu$$ that dominates $$\mathsf P$$. The density of $$\mathsf P$$ with respect to $$\nu$$ is then invariant, hence constant on ergodic components, and its modulus is 1 because the total variation is exhausted by the restrictions.

<div class="proof" markdown="1">

*Proof.* Let $$D_n$$ be the ball of radius $$n$$ and $$\pi_{D'\to D}$$ the restriction of configurations.

*Step 1 (a dominating positive family).* For a measurable map $$\pi$$ and a complex measure $$\mu$$, $$\lvert \pi_*\mu\rvert \le\pi_*\lvert \mu\rvert $$. Hence $$\lvert \mathsf P_D\rvert \le(\pi_{D'\to D})_*\lvert \mathsf P_{D'}\rvert $$ for $$D\subset D'$$, and the right side is nondecreasing in $$D'$$ and has mass at most $$S$$. The setwise limit $$\nu_{D_m}$$ of $$(\pi_{D_n\to D_m})_*\lvert \mathsf P_{D_n}\rvert $$ as $$n\to\infty$$ is then a measure of mass $$S=\lim\lVert \mathsf P_{D_n}\rVert $$, the family $$(\nu_{D_m})$$ is consistent, and $$\lvert \mathsf P_{D_m}\rvert \le\nu_{D_m}$$. The limit is the same along any increasing exhaustion of $$M$$ by bounded sets, so $$g_*\nu_D=\nu_{gD}$$.

*Step 2 (Kolmogorov).* The spaces $$\Gamma_{D_n}$$ are standard Borel, and their projective limit is $$\Gamma_M$$. By Kolmogorov's extension theorem for projective systems on standard Borel spaces [[Parthasarathy]](#ref-parthasarathy) there is a unique finite measure $$\nu$$ on $$\Gamma_M$$ with marginals $$\nu_{D_m}$$. It is invariant by Step 1, and has mass $$S$$.

*Step 3 (martingale).* This step carries the weight of the extension. Since $$\lvert \mathsf P_{D_n}\rvert \le\nu_{D_n}$$ the density $$h_n=d\mathsf P_{D_n}/d\nu_{D_n}$$ exists with $$\lvert h_n\rvert \le1$$. Consistency says that $$h_n$$ is the conditional expectation of $$h_{n+1}$$ on the configuration in $$D_n$$, so $$(h_n)$$ is a bounded martingale for the filtration generated by the restrictions to $$D_n$$ and converges $$\nu$$-almost everywhere and in $$L^1$$ to a function $$h$$ with $$\lvert h\rvert \le1$$. The complex measure $$h\nu$$ has the marginals $$\mathsf P_{D_n}$$, and is the unique complex measure with these marginals, because cylinder sets form a $$\pi$$-system that generates the $$\sigma$$-algebra. This is the extension of $$\mathsf P$$. For projective families of complex Radon measures, uniform boundedness of the variation is necessary for such an extension to exist [[Albeverio–Mazzucchi]](#ref-am), and together with a tightness condition it is sufficient [[Thomas]](#ref-thomas). The argument above does not need a separate tightness condition because the family is dominated by the positive projective family $$\nu$$, which extends by Kolmogorov's theorem.

*Step 4 (invariance).* For $$g\in P$$, $$g_*\mathsf P$$ and $$\mathsf P$$ have the same marginals, so $$g_*\mathsf P=\mathsf P$$, and likewise $$g_*\nu=\nu$$. Therefore $$h\circ g=h$$ $$\nu$$-almost everywhere for each $$g$$, and since $$P$$ is locally compact and second countable, $$h$$ agrees almost everywhere with a strictly invariant function [[Zimmer]](#ref-zimmer).

*Step 5 (ergodic decomposition).* By Varadarajan's theorem on ergodic decompositions [[Varadarajan]](#ref-varadarajan), $$\nu/S=\int R_s\,\kappa(ds)$$ with ergodic invariant probability measures $$R_s$$. A strictly invariant function is constant on $$R_s$$-almost every point for $$\kappa$$-almost every $$s$$; call the constant $$h(s)$$. Then $$\mathsf P=S\int h(s)R_s\kappa(ds)$$.

*Step 6 (the modulus is 1).* We have $$\lvert \mathsf P\rvert (\Gamma_M)=\int\lvert h\rvert d\nu\le S$$. Conversely $$\lvert \mathsf P\rvert (\Gamma_M)\ge\lVert \mathsf P_{D_n}\rVert \to S$$. Therefore $$\int(1-\lvert h\rvert )d\nu=0$$ with $$1-\lvert h\rvert \ge0$$, so $$\lvert h\rvert =1$$ $$\nu$$-almost everywhere, and $$h=e^{i\varphi}$$. Normalisation $$\mathsf P(\Gamma_M)=1$$ gives $$S\int e^{i\varphi}d\kappa=1$$. If $$S=1$$, a unimodular function with average 1 equals 1 almost everywhere. $$\square$$

</div>

In (9) a process that is not classical has $$S>1$$: its ergodic components are positive processes, and the phases $$e^{i\varphi}$$ produce cancellations between them. The simplest case is $$\mathsf P=(1+\epsilon)R_1-\epsilon R_2$$ for two mutually singular ergodic invariant processes $$R_1,R_2$$, which has $$S=1+2\epsilon$$, $$\kappa$$ the probability measure with weights $$(1+\epsilon)/S$$ and $$\epsilon/S$$, and $$e^{i\varphi}=\pm1$$. The components $$R_s$$ are arbitrary ergodic invariant classical processes, not necessarily Poisson, and Proposition 9 gives non-Poisson examples. Bounded total variation therefore adds no ergodic components other than classical ones. The cancellations can still change the counting statistics. For a signed combination of Poisson processes with mean intensity $$\rho$$, the second cumulant of $$N(D)$$ computed from the correlation functions is $$\rho V+V^2\mu_2$$, where $$\mu_2$$ is the second central moment of the intensity under the signed mixing law. For the three-atom law of Example 1 on [Mixed-Poisson and signed substrates and the four-dimensional d'Alembertian](/quantum-gravity/signed-substrates/), whose total variation is at most $$7/6$$, $$\mu_2=-0.47\rho^{2}$$, so the second cumulant is $$\rho V-0.47\rho^2V^2<\rho V$$. The inequality $$\mathrm{Var}\,N\ge\mathbb EN$$ of Theorem 4 therefore fails for this cumulant, and some count probabilities of the process are negative.

Complex processes with prescribed invariant correlation functions of moderate growth are constructed on the same page (Theorem 7), and for mixed Poisson processes with compactly supported complex mixing law it proves that requiring every count probability to lie in $$[0,1]$$ makes the mixing law positive (Proposition 5).

## 7. Invariant kernels on de Sitter space

The proofs above use the non-compactness of the boosts, and a non-compact stabiliser plays the same role for other symmetry groups. On de Sitter space $$dS_4=G/H$$ with $$G=SO_0(4,1)$$ and $$H=SO_0(3,1)$$ the one-body density of an invariant state would be a positive-definite $$G$$-invariant kernel. We do not set up states on de Sitter space and prove only the statement about kernels.

<div class="proposition" markdown="1">

**Proposition 11** (invariant positive-definite kernels on de Sitter space). Let $$k:dS_4\times dS_4\to\mathbb C$$ be a continuous positive-definite kernel with $$k(gx,gy)=k(x,y)$$ for all $$g\in G$$. Then $$k$$ is constant.

</div>

<div class="proof" markdown="1">

*Proof.* Let $$\mathcal H$$ be the Hilbert space obtained from finitely supported functions on $$dS_4$$ with the inner product $$\langle\delta_x,\delta_y\rangle=k(x,y)$$. By invariance, $$\pi(g)\delta_x=\delta_{gx}$$ defines a unitary representation of $$G$$, and it is strongly continuous because $$\lVert\pi(g)\delta_x-\delta_x\rVert ^2=2k(x,x)-2\mathrm{Re}\,k(gx,x)\to0$$ as $$g\to e$$. Fix $$x_0$$ with stabiliser $$H$$ and put $$v=\delta_{x_0}$$, so $$\pi(h)v=v$$ for $$h\in H$$. Write $$v=v_0+v'$$ with $$v_0$$ invariant under all of $$G$$ and $$v'$$ in the orthogonal complement, which contains no non-zero $$G$$-invariant vector. Then $$v'$$ is $$H$$-invariant. The group $$G$$ is connected, simple and has trivial centre, so by the Howe–Moore theorem [[Howe–Moore]](#ref-hm) the matrix coefficient $$\langle\pi(g)v',v'\rangle$$ tends to zero as $$g$$ leaves every compact set. For $$g=h_n$$ in the closed non-compact subgroup $$H$$ it equals $$\lVert v'\rVert ^2$$ for every $$n$$, so $$v'=0$$. Hence $$v$$ is $$G$$-invariant, $$\delta_x=\pi(g)v=v$$ for all $$x$$, and $$k(x,y)=\lVert v\rVert ^2$$. $$\square$$

</div>

## 8. Counting statistics without boost invariance or finite density

A fixed frame is a legitimate datum, for instance the rest frame of the cosmic microwave background. A quasi-free fermionic state whose $$T$$ is the multiplier by the indicator of a Euclidean ball in momentum space is translation invariant, has finite density and is not boost invariant. Steps 1 and 2 of Theorem 2 hold without boosts, with $$\mu$$ any finite measure, so a Fermi-ball state is allowed. Which counting statistics such states give is open.

Without finite density, a bounded Lorentz-invariant multiplier, for example the indicator of $$\{p^0>0,\ 1\le(p^0)^2-\lvert\vec p\rvert^2\le4\}$$, defines a Poincaré-invariant quasi-free CAR state with $$0\le T\le1$$. For a unit vector $$h\in L^2_c$$ the number $$a^*(h)a(h)$$ of the single mode $$h$$ has variance $$t(1-t)$$ with mean $$t=\langle h,Th\rangle$$, so single-mode numbers are sub-Poisson. Here $$\mu(\mathbb R^4)=\infty$$, so $$\omega(N(D))=V(D)\mu(\mathbb R^4)=\infty$$ for every $$D$$ by the computation in Step 2 of Theorem 2: sharp counts have no finite mean, and the conclusion of Theorem 6 fails without finite density. For bosons the mixtures (7) with $$\int\lvert z\rvert ^2d\mathsf m=\infty$$ are invariant and locally normal, with infinite mean counts and $$\mathrm{Var}\,N(D)=\infty$$.

Counts other than $$d\Gamma(1_D)$$, for example a positive operator-valued measure on the one-particle space, are outside the setting; [Integer counts and finite-dimensional algebras in a positive-energy vacuum](/quantum-gravity/count-algebras/) treats integer additive counts of a positive-energy field. A state that is not locally normal has no single-mode reduction, and Proposition 3 does not apply to it. Complex processes of unbounded total variation lie outside Theorem 10.

Three questions remain open.

1. Whether a Poincaré-invariant classical point process can be hyperuniform. Corollary 8 excludes it when the pair correlation is $$\rho^2g(x-y)$$ with $$g-1$$ integrable. The orbit lattice of [Kernel-scale kinematics of a single realisation](/quantum-gravity/per-realisation-kinematics/) is invariant in law and has a pair correlation carried by hyperboloids of fixed proper time, so Corollary 8 does not apply to it. The variance-to-mean ratio computed there for causal diamonds is about 0.48, 0.33 and 0.20 at volumes 4, 16 and 64, and the proved bound $$\mathrm{Var}\,N\le C^2V^{3/2}$$ does not settle whether it tends to 0.
2. Whether, among classical invariant simple point processes with second moments, the mixed Poisson processes are characterised by the independence, given the intensity, of counts in disjoint regions. Fock states have this property and the random lattice lacks it.
3. Whether the classification of Theorem 4 holds without (H3), for locally normal invariant states of infinite mean count. (H3) enters the proof of Theorem 2 through the finiteness of $$\mu$$.

## References

1. <a id="ref-bhs"></a>Bombelli, Henson, Sorkin, "Discreteness without symmetry breaking: a theorem," Mod. Phys. Lett. A 24 (2009) 2579, [arXiv:gr-qc/0605006](https://arxiv.org/abs/gr-qc/0605006). Used for: no equivariant choice of a preferred direction exists on a Poisson sprinkling.
2. <a id="ref-dhs"></a>Dowker, Henson, Sorkin, "Quantum gravity phenomenology, Lorentz invariance and discreteness," Mod. Phys. Lett. A 19 (2004) 1829, [arXiv:gr-qc/0311055](https://arxiv.org/abs/gr-qc/0311055). Used for: local Lorentz invariance of the discreteness of a sprinkled causal set, and the non-invariance of boosted regular lattices.
3. <a id="ref-sa"></a>Saravani, Aslanbeigi, "On the causal set-continuum correspondence," Class. Quantum Grav. 31 (2014) 205013, [arXiv:1403.6429](https://arxiv.org/abs/1403.6429). Used for: Poisson sprinkling is optimal for the number-volume correspondence in arbitrarily small regions; two-dimensional Lorentzian lattices do better in large regions.
4. <a id="ref-hp"></a>Hudson, Parthasarathy, "Quantum Itô's formula and stochastic evolutions," Commun. Math. Phys. 93 (1984) 301–323. Used for: the realisation of Poisson processes by operators on Fock space.
5. <a id="ref-surya"></a>Surya, "The causal set approach to quantum gravity," Living Rev. Relativ. 22 (2019) 5, [doi:10.1007/s41114-019-0023-1](https://doi.org/10.1007/s41114-019-0023-1), [arXiv:1903.11544](https://arxiv.org/abs/1903.11544). Review of causal sets and sprinklings.
6. <a id="ref-sorkin"></a>Sorkin, "Causal sets: discrete gravity (notes for the Valdivia summer school)," [arXiv:gr-qc/0309009](https://arxiv.org/abs/gr-qc/0309009). Introduction to causal sets.
7. <a id="ref-rs2"></a>Reed, Simon, *Methods of Modern Mathematical Physics II: Fourier Analysis, Self-Adjointness*, Academic Press, 1975, section IX.2. Used for: the Bochner–Schwartz theorem (a positive-definite tempered distribution is the Fourier transform of a positive tempered measure).
8. <a id="ref-soshnikov"></a>Soshnikov, "Determinantal random point fields," Russian Math. Surveys 55 (2000) 923–975, [doi:10.1070/RM2000v055n05ABEH000321](https://doi.org/10.1070/RM2000v055n05ABEH000321), [arXiv:math/0002099](https://arxiv.org/abs/math/0002099). Used for: determinantal processes with locally trace-class Hermitian kernels.
9. <a id="ref-siegel"></a>Siegel, "A mean value theorem in geometry of numbers," Ann. of Math. 46 (1945) 340, [doi:10.2307/1969027](https://doi.org/10.2307/1969027). Used for: the mean value theorem for lattice sums over the space of unimodular lattices and the finiteness of its invariant measure.
10. <a id="ref-moore"></a>Moore, "Ergodicity of flows on homogeneous spaces," Amer. J. Math. 88 (1966) 154, [doi:10.2307/2373052](https://doi.org/10.2307/2373052). Used for: ergodicity of closed non-compact subgroups on the quotient of a simple group by a lattice.
11. <a id="ref-thomas"></a>Thomas, "Projective limits of complex measures and martingale convergence," Probab. Theory Related Fields 119 (2001) 579–588, [doi:10.1007/pl00008772](https://doi.org/10.1007/pl00008772). Used for: the extension of a projective family of complex Radon measures of bounded variation under a tightness condition, in the form presented in [[Albeverio–Mazzucchi]](#ref-am).
12. <a id="ref-am"></a>Albeverio, Mazzucchi, "A unified approach to infinite-dimensional integration," Rev. Math. Phys. 28 (2016) 1650005, [arXiv:1411.2853](https://arxiv.org/abs/1411.2853). Used for: uniform boundedness of the total variation as a necessary condition for a projective family of complex measures to arise from a measure on the limit.
13. <a id="ref-parthasarathy"></a>Parthasarathy, *Probability Measures on Metric Spaces*, Academic Press, 1967, Theorem V.3.2. Used for: Kolmogorov extension of projective systems of Borel measures on standard Borel spaces.
14. <a id="ref-varadarajan"></a>Varadarajan, "Groups of automorphisms of Borel spaces," Trans. Amer. Math. Soc. 109 (1963) 191–220, [doi:10.1090/s0002-9947-1963-0159923-5](https://doi.org/10.1090/s0002-9947-1963-0159923-5). Used for: the ergodic decomposition of an invariant measure.
15. <a id="ref-zimmer"></a>Zimmer, *Ergodic Theory and Semisimple Groups*, Birkhäuser, 1984. Used for: an almost-everywhere invariant function for an action of a second-countable locally compact group has a strictly invariant version.
16. <a id="ref-hm"></a>Howe, Moore, "Asymptotic properties of unitary representations," J. Funct. Anal. 32 (1979) 72–96, [doi:10.1016/0022-1236(79)90078-8](https://doi.org/10.1016/0022-1236(79)90078-8). Used for: matrix coefficients of a unitary representation of a connected simple Lie group with finite centre, without non-zero invariant vectors, vanish at infinity.
