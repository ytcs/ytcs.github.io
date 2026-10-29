---
title: Exact discrete Lorentz symmetry cannot carry propagation
date: 2026-10-29
category: substrate
order: 4
description: A lattice in Minkowski space that is exactly invariant under lattice translations and under a Zariski-dense group of lattice-preserving Lorentz transformations admits no dispersion relation, because every translation-commuting operator is a constant and every positive invariant state has a flat spectrum plus periodic parts. Spontaneous breaking of the discrete Lorentz group restores propagation only when the unbroken subgroup is no longer Zariski dense, and then a Lorentz-violating kinetic term is allowed.
status: Theorems for operators are proved in full; the statements for invariant states and for orbit closures rest on a cited measure-classification theorem; the packet-regulator values are exact for one shell and numerical otherwise.
excludes: Exact lattice translations together with an exactly preserved Zariski-dense arithmetic subgroup of the Lorentz group (for example O(1,3;Z) or the automorphism group of an anisotropic integral form), taken as a mechanism that protects low-energy Lorentz invariance of a scalar lattice dynamics or of a positive state.
---

Take a lattice $$\Lambda\subset\mathbb R^{1,3}$$, such as $$\mathbb Z^4$$ with the form $$Q(x)=x_0^2-x_1^2-x_2^2-x_3^2$$, and a group $$\Gamma$$ of Lorentz transformations that map $$\Lambda$$ onto itself, such as $$O(1,3;\mathbb Z)$$. We ask whether a dynamics or a state that is invariant under both the lattice translations and $$\Gamma$$ can carry a dispersion relation. It cannot when $$\Gamma$$ is Zariski dense in the Lorentz group, which holds for $$O(1,3;\mathbb Z)$$, for its finite-index subgroups, and for the automorphism group of an anisotropic integral form such as $$7x_0^2-x_1^2-x_2^2-x_3^2$$. A translation-invariant operator on $$\ell^2(\Lambda)$$ that also commutes with $$\Gamma$$ and has a measurable symbol is a constant (Theorem 1). The two-point function of any positive translation- and $$\Gamma$$-invariant state is a multiple of the Kronecker delta plus a sum of periodic functions with rational wave vectors (Theorem 2), which covers interacting states. Both results follow from one fact: every non-zero lattice vector has an infinite $$\Gamma$$-orbit, so $$\Gamma$$ acts ergodically on the Brillouin torus.

If $$\Gamma$$ is only a symmetry of the action and the state breaks it, the same conclusion holds while the unbroken subgroup stays Zariski dense. Once it does not, the remaining symmetry no longer forces the quadratic action to be Lorentz invariant: apart from one case left open, there is an invariant tensor different from the metric, and a term $$T^{\mu\nu}\partial_\mu\phi\,\partial_\nu\phi$$ is allowed (Theorem 3). Low-energy Lorentz invariance is then possible but unprotected.

Integral Lorentz transformations of $$\mathbb Z^4$$ and the covariance of lattice wave equations under them were studied by Lorente [[Lorente]](#ref-lorente), who obtains covariance by letting the group act on non-linear functions of the lattice momentum; section 8 asks whether that action is a symmetry of a point set. The mathematical inputs are three cited theorems, stated in section 2: the Borel density and Borel–Harish-Chandra theorems for the arithmetic groups, the Benoist–Quint classification of invariant measures and closed invariant sets on the torus [[Benoist–Quint]](#ref-bq) (the proximal case is due to Bourgain, Furman, Lindenstrauss and Mozes [[BFLM]](#ref-bflm)), and the classification of irreducible connected groups preserving a Lorentzian form [[Di Scala–Leistner–Neukirchner]](#ref-dln). What is added here is their use as a statement about lattice dynamics: the constancy of symbols (Theorem 1), the flat-plus-periodic form of invariant two-point functions (Theorem 2), the density of low momenta in the zone (Corollary 1), the exact packet response (Proposition 2), the dichotomy for a broken group (Theorem 3), and the closure of three neighbouring loopholes in sections 8 to 10: non-linear momentum actions, spectral-triple geometries on the Brillouin quotient, and states centred on one lattice point.

Lorentz-invariant point sets without a lattice structure are treated on other pages: the Poisson case in [Lorentz-invariant quantum sprinklings](/quantum-gravity/quantum-sprinklings/), and random and orbit lattices, which have no translation subgroup, in [Lorentz-invariant point sets](/quantum-gravity/invariant-point-sets/) and [Kernel-scale kinematics of a single realisation](/quantum-gravity/per-realisation-kinematics/). The last of these uses Proposition 1 below.

## 1. Conventions

Write $$B(x,y)=x_0y_0-\vec x\cdot\vec y$$ and $$Q(x)=B(x,x)$$. The lattice $$\Lambda\subset\mathbb R^4$$ has rank 4 and $$\Gamma\subset O(1,3)$$ is a group with $$g\Lambda=\Lambda$$ for $$g\in\Gamma$$, so that $$\Gamma$$ acts on $$\Lambda$$ through $$GL(\Lambda)\cong GL(4,\mathbb Z)$$. The Brillouin torus is $$\mathbb T=\mathbb R^4/2\pi\Lambda^\vee$$ with $$\Lambda^\vee=\{k: k\cdot v\in\mathbb Z\ \forall v\in\Lambda\}$$, and its characters are $$\chi_v(k)=e^{ik\cdot v}$$, $$v\in\Lambda$$. The group acts on $$\mathbb T$$ by $$g\cdot k=(g^{-1})^{T}k$$, the action for which

$$
\chi_v(g\cdot k)=\chi_{g^{-1}v}(k). \tag{1}
$$

For $$g\in O(1,3)$$ one has $$(g^{-1})^T=\eta g\eta$$ with $$\eta=\mathrm{diag}(1,-1,-1,-1)$$, so the action on $$\mathbb T$$ is equivalent to the vector action.

The Fourier transform $$\hat f(k)=\sum_xf(x)e^{-ik\cdot x}$$ identifies $$\ell^2(\Lambda)\otimes V$$ with $$L^2(\mathbb T;V)$$. Here $$V=\mathbb C^N$$ is the space of field components and $$\sigma:\Gamma\to GL(V)$$ a finite-dimensional representation, not necessarily unitary (for spinors, the spin representation). Translation $$(T_af)(x)=f(x-a)$$ becomes multiplication by $$e^{-ik\cdot a}$$, and the Lorentz action $$(U_gf)(x)=\sigma(g)f(g^{-1}x)$$ becomes

$$
(U_gF)(k)=\sigma(g)F(g^{-1}\cdot k). \tag{2}
$$

A *symbol* is a measurable map $$d:\mathbb T\to M_N(\mathbb C)$$, defining $$(DF)(k)=d(k)F(k)$$. Every operator commuting with all $$T_a$$ is of this form (proof of Theorem 1). The dispersion relations are the eigenvalues of $$d(k)$$ as functions of $$k$$.

Two hypotheses on $$\Gamma$$ are used.

(F) Every $$\Gamma$$-orbit on $$\Lambda\setminus\{0\}$$ is infinite.

(H1) The Zariski closure of $$\Gamma$$ in $$GL(4,\mathbb R)$$ contains the identity component $$SO(1,3)^\circ$$ of the Lorentz group.

The Zariski closure of a set of matrices is the smallest set, cut out by polynomial equations in the matrix entries, that contains it; $$\Gamma$$ satisfies (H1) when every polynomial that vanishes on $$\Gamma$$ vanishes on $$SO(1,3)^\circ$$. A group that is Zariski dense in this sense is not contained in any proper algebraic subgroup, so it fixes no non-zero vector and no proper subspace that $$SO(1,3)^\circ$$ does not. Arithmetic groups such as $$O(1,3;\mathbb Z)$$, the integer matrices preserving $$Q$$, are discrete subgroups of the Lorentz group and nevertheless Zariski dense; this is the content of the Borel density theorem (Theorem C1).

Lemma 1 shows that (H1) implies (F). Condition (F) alone suffices for Lemma 2, Theorem 1 and Theorem 4, and it also holds in lower dimension. A worked limiting case: for $$\Lambda=\mathbb Z^2$$ with the form $$t^2-x^2$$, the group $$\Gamma$$ is finite (signed permutations of the two axes), the orbit of $$e_0$$ is $$\{\pm e_0,\pm e_1\}$$, (F) fails, and the usual lattice Klein–Gordon operator with a non-trivial dispersion relation is an invariant example. For the form $$t^2-2x^2$$, the matrix $$A=\begin{pmatrix}3&4\\2&3\end{pmatrix}$$ preserves it, has eigenvalues $$3\pm2\sqrt2$$, and no non-zero vector of $$\mathbb Z^2$$ has a finite $$\langle A\rangle$$-orbit; (F) holds, and Theorem 1 forbids a dispersion relation even in 1+1 dimensions.

## 2. The cited theorems

Three results are used without proof, in the form in which they are applied.

<div class="theorem" markdown="1">

**Theorem C1** (Borel density; Borel–Harish-Chandra). If $$G$$ is the group of real points of a semisimple algebraic group over $$\mathbb Q$$ without compact factors, then $$G\cap GL(n,\mathbb Z)$$ is a lattice in $$G$$ (Borel and Harish-Chandra, 1962), and every lattice in $$G$$ is Zariski dense in $$G$$ (Borel, 1960). Consequently (H1) holds for every finite-index subgroup of $$O(1,3;\mathbb Z)$$ and, more generally, of the automorphism group of any integral quadratic form of signature $$(1,3)$$, such as $$7x_0^2-x_1^2-x_2^2-x_3^2$$. The same applies to signature $$(1,2)$$.

</div>

<div class="theorem" markdown="1">

**Theorem C2** (Benoist–Quint [[Benoist–Quint]](#ref-bq), Theorem 1.3 and Corollary 1.4). Let $$\Gamma_0$$ be a subsemigroup of $$SL(d,\mathbb Z)$$ acting strongly irreducibly on $$\mathbb R^d$$, meaning that every finite-index subgroup of the group it generates acts irreducibly. On the torus $$\mathbb R^d/\mathbb Z^d$$: (a) the only $$\Gamma_0$$-invariant Borel probability measure without atoms is the Haar measure; (b) the only closed infinite $$\Gamma_0$$-invariant subset is the whole torus. Part (b) for such groups is due to Muchnik and to Guivarc'h and Starkov, and part (a) under a proximality assumption to Bourgain, Furman, Lindenstrauss and Mozes [[BFLM]](#ref-bflm).

</div>

<div class="theorem" markdown="1">

**Theorem C3** (Di Scala, Leistner, Neukirchner [[Di Scala–Leistner–Neukirchner]](#ref-dln)). A connected subgroup of $$GL(n,\mathbb R)$$ that acts irreducibly on $$\mathbb R^n$$ and leaves invariant a bilinear form of Lorentzian signature is conjugate to $$SO(1,n-1)^\circ$$.

</div>

Under (H1) the group $$\Gamma^+=\Gamma\cap SL(4,\mathbb Z)$$, of index at most 2, still has Zariski closure containing $$SO(1,3)^\circ$$ and acts strongly irreducibly: a finite-index subgroup has Zariski closure of finite index in that of $$\Gamma$$, hence containing $$SO(1,3)^\circ$$, which acts irreducibly because it contains boosts in every $$(0,j)$$-plane and all rotations. So C2 applies to $$\Gamma^+$$ acting on $$\mathbb T$$, and a fortiori to measures and sets invariant under $$\Gamma$$.

The results that use C1 to C3 are Corollary 1, Proposition 1, Theorem 2, Theorem 3 and part of Proposition 7. Everything else on the page is proved in full from (F) or (H1).

## 3. No finite orbits, no finite-dimensional invariant subspaces

The point of this section is that the Brillouin torus carries no small $$\Gamma$$-invariant structure at all.

<div class="lemma" markdown="1">

**Lemma 1** (no finite orbits). Assume (H1). Then (F) holds. A point of $$\mathbb T$$ has a finite $$\Gamma$$-orbit if and only if it is a torsion point, that is, a point of finite order in the group $$\mathbb T$$.

</div>

<div class="proof" markdown="1">

*Proof.* If $$\Gamma v$$ is finite, a finite-index subgroup $$\Gamma'$$ fixes $$v$$. The condition $$gv=v$$ is polynomial in $$g$$, so the Zariski closure of $$\Gamma'$$ also fixes $$v$$. That closure contains $$SO(1,3)^\circ$$, which fixes no non-zero vector: a vector with $$v_0\ne0$$ is moved by a boost, and one with $$v_0=0$$, $$\vec v\ne0$$ by a rotation about an axis orthogonal to $$\vec v$$. Hence $$v=0$$.

Torsion points of a given order are finitely many and form a $$\Gamma$$-invariant set, so they have finite orbits. Conversely, let $$k$$ have a finite orbit and let $$\Gamma'$$ be the finite-index subgroup fixing it. The fixed set $$\mathrm{Fix}(\Gamma')\ni k$$ is a closed subgroup of $$\mathbb T$$ whose identity component is a subtorus $$S=\exp\mathfrak s$$, $$\mathfrak s\subset\mathbb R^4$$. Since $$\Gamma'$$ fixes $$S$$ pointwise, it fixes every vector of $$\mathfrak s$$ under the linear action $$g\mapsto(g^{-1})^T=\eta g\eta$$. By the argument of the first paragraph, applied to this equivalent representation, $$\mathfrak s=0$$. A compact group with trivial identity component is finite, so $$k$$ has finite order. $$\square$$

</div>

<div class="lemma" markdown="1">

**Lemma 2** (weak mixing and ergodicity). Assume (F). Let $$L^2_0(\mathbb T)$$ be the orthogonal complement of the constants in $$L^2(\mathbb T)$$, and $$(K_gf)(k)=f(g^{-1}\cdot k)$$. (i) $$L^2_0(\mathbb T)$$ contains no non-zero finite-dimensional $$K$$-invariant subspace. (ii) Every measurable function $$h:\mathbb T\to\mathbb C$$ with $$h(g\cdot k)=h(k)$$ almost everywhere for each $$g\in\Gamma$$ is almost everywhere constant.

</div>

<div class="proof" markdown="1">

*Proof.* (i) By (1), the Fourier transform turns $$K_g$$ into the permutation operator $$f\mapsto f\circ g^{-1}$$ on $$\ell^2(\Lambda)$$, and $$L^2_0$$ into $$\ell^2(\Lambda\setminus\{0\})$$. Suppose $$F\subset\ell^2(\Lambda\setminus\{0\})$$ is a non-zero finite-dimensional invariant subspace. Pick $$f\in F$$ and $$v_0$$ with $$f(v_0)\ne0$$. By (F) there are $$g_n\in\Gamma$$ with the points $$g_nv_0$$ pairwise distinct. The vectors $$f\circ g_n^{-1}$$ lie on a sphere in $$F$$, which is compact, so a subsequence converges in norm. But $$(f\circ g_n^{-1})(g_nv_0)=f(v_0)$$, while for fixed $$m$$ the values $$(f\circ g_m^{-1})(g_nv_0)=f(g_m^{-1}g_nv_0)$$ tend to 0 as $$n\to\infty$$, because $$f\in\ell^2$$ and the points $$g_m^{-1}g_nv_0$$ are pairwise distinct. Passing inductively to a subsequence with $$\lvert f(g_m^{-1}g_nv_0)\rvert \le\lvert f(v_0)\rvert /2$$ for $$m<n$$ gives $$\lVert f\circ g_n^{-1}-f\circ g_m^{-1}\rVert \ge\lvert f(v_0)\rvert /2$$ for all $$m<n$$, so no subsequence is Cauchy. This contradicts compactness.

(ii) The functions $$\arctan\operatorname{Re}h$$ and $$\arctan\operatorname{Im}h$$ are bounded, hence in $$L^2$$, and invariant. An invariant function $$u\in L^2$$ gives the one-dimensional invariant subspace spanned by $$u-\int u$$ in $$L^2_0$$, which is zero by (i). Since $$\arctan$$ is injective, $$h$$ is constant almost everywhere. $$\square$$

</div>

## 4. Operators that commute with translations and with the group

<div class="theorem" markdown="1">

**Theorem 1** (constant symbols). Assume (F). Let $$D$$ be a bounded operator, or a self-adjoint operator, on $$\ell^2(\Lambda)\otimes V$$ that commutes with every $$T_a$$, $$a\in\Lambda$$, and every $$U_g$$, $$g\in\Gamma$$. Then $$D$$ is the multiplication by a symbol $$d$$ with $$d(g\cdot k)=\sigma(g)d(k)\sigma(g)^{-1}$$ almost everywhere, and:

(a) if $$N=1$$ ($$V=\mathbb C$$, $$\sigma$$ trivial), then $$d$$ is almost everywhere a constant $$c$$ and $$D=c\cdot1$$;

(b) for general $$N$$, the characteristic polynomial of $$d(k)$$ is almost everywhere independent of $$k$$, so every band of the dispersion relation is flat;

(c) if $$D$$ is bounded, then $$d$$ is almost everywhere a constant matrix $$d_0$$, and $$d_0=\sigma(g)d_0\sigma(g)^{-1}$$ for all $$g\in\Gamma$$.

</div>

Translation invariance turns the operator into a function on the torus, and invariance under $$\Gamma$$ makes that function constant on the orbits of an ergodic action. The heart of the argument is (F); the remaining steps are routine.

<div class="proof" markdown="1">

*Proof.* The operators $$T_a$$ generate, on $$L^2(\mathbb T)\otimes V$$, the von Neumann algebra $$L^\infty(\mathbb T)\otimes1$$ of multiplications by characters and their limits. Its commutant is $$L^\infty(\mathbb T)\otimes M_N(\mathbb C)$$, because $$L^\infty(\mathbb T)$$ is maximal abelian on $$L^2(\mathbb T)$$. A bounded operator commuting with all $$T_a$$ is therefore a bounded matrix symbol, and a self-adjoint one has its spectral projections in the commutant, so it is a multiplier by a measurable Hermitian symbol. Commutation with $$U_g$$ and (2) give $$d(k)\sigma(g)F(g^{-1}\cdot k)=\sigma(g)d(g^{-1}\cdot k)F(g^{-1}\cdot k)$$ for all $$F$$, which is the stated covariance.

(a) $$d$$ is a measurable function with $$d(g\cdot k)=d(k)$$, so it is constant by Lemma 2(ii).

(b) For each $$j$$ the function $$k\mapsto\operatorname{tr}d(k)^j$$ is measurable and invariant, hence constant by Lemma 2(ii). The coefficients of the characteristic polynomial are polynomials in these traces (Newton's identities).

(c) The entries of $$d$$ are bounded, hence in $$L^2(\mathbb T)$$. By the covariance, the entries of $$d\circ g$$ are linear combinations of the entries of $$d$$, so the span of the entries is a finite-dimensional invariant subspace of $$L^2(\mathbb T)$$. Its projection onto $$L^2_0$$ is again finite-dimensional and invariant, hence zero by Lemma 2(i). $$\square$$

</div>

For bounded $$D$$ and $$N=1$$ there is a direct argument without ergodic theory: the kernel $$c=D\delta_0$$ lies in $$\ell^2(\Lambda)$$ and is constant on $$\Gamma$$-orbits, and the orbits of non-zero vectors are infinite, so $$c=c(0)\delta_0$$. The ergodic form covers unbounded operators and the matrix case.

<div class="remark" markdown="1">

**Remark 1** (scope of "$$D=c\cdot1$$"). Translation invariance under all of $$\Lambda$$ is used. If only a $$\Gamma$$-stable sublattice $$\Lambda'$$ of finite index is a symmetry, the symbol is a matrix of size $$[\Lambda:\Lambda']$$ on the dual torus of $$\Lambda'$$, with $$\Gamma$$ acting through a finite permutation representation. Parts (b) and (c) then apply with $$\sigma$$ that permutation representation, and the conclusion is a constant matrix, not a multiple of the identity. An example is the staggered mass for $$\Lambda=\mathbb Z^4$$: since $$Q(x)\equiv x_0+x_1+x_2+x_3\pmod2$$, the function $$m(x)=(-1)^{Q(x)}$$ is $$\Gamma$$-invariant, commutes with translations by the $$\Gamma$$-stable even sublattice $$\{Q\ \text{even}\}$$, and does not commute with translation by $$e_0$$. It is ultra-local and independent of $$k$$. Operators that are $$\Gamma$$-invariant without translation invariance exist (any multiplication by a function of $$Q(x)$$), and none of them has a $$k$$-dependent spectrum.

</div>

<div class="corollary" markdown="1">

**Corollary 1** (low momenta are dense in the zone). Assume (H1) and C2. For every non-torsion $$k\in\mathbb T$$ the orbit $$\Gamma\cdot k$$ is dense in $$\mathbb T$$.

</div>

<div class="proof" markdown="1">

*Proof.* By Lemma 1 the orbit is infinite, and so is the orbit under $$\Gamma^+$$, which satisfies (H1) as well. Its closure is a closed infinite $$\Gamma^+$$-invariant set, hence the whole torus by C2(b). $$\square$$

</div>

A lattice momentum of small norm is non-torsion as soon as one of its coordinate ratios is irrational, and then some $$g\in\Gamma$$ maps it arbitrarily close to any prescribed point of the zone, the zone boundary included. A symmetry that identifies low momenta with momenta at the cutoff leaves no separate low-energy description. The Borel density theorem makes every $$\Gamma$$-invariant polynomial tensor $$SO(1,3)$$-invariant, so a continuum effective theory with $$\Gamma$$-invariant coefficients has Lorentz-invariant coefficients; Theorem 1 shows that a $$\Gamma$$-invariant lattice theory has no such effective description for the protection to act on.

<div class="proposition" markdown="1">

**Proposition 1** (shell sums have no dispersion on any open set). Assume (H1) and C2. Let $$c:\Lambda\to\mathbb C$$ be a $$\Gamma$$-invariant function that vanishes outside finitely many shells $$\{Q=q\}$$ and is non-zero at some $$v\ne0$$. Then its Fourier series $$\hat c$$, a distribution on $$\mathbb T$$, coincides with an integrable function on no non-empty open subset of $$\mathbb T$$.

</div>

<div class="proof" markdown="1">

*Proof.* Suppose $$\hat c=h$$ on an open set $$U\ne\emptyset$$ with $$h\in L^1_{\rm loc}(U)$$. Since $$\hat c$$ is $$\Gamma$$-invariant, it equals $$h\circ g^{-1}$$ on $$gU$$, and these local functions agree on overlaps because $$\hat c$$ is one distribution; so $$\hat c=\tilde h$$ on the open invariant set $$\Gamma U$$, with $$\tilde h\in L^1_{\rm loc}$$ invariant. The indicator of $$\Gamma U$$ is invariant, hence almost everywhere constant by Lemma 2(ii), and equal to 1 since $$U$$ has positive measure; so $$\Gamma U$$ is conull and $$\tilde h$$ is almost everywhere a constant $$\kappa$$. The closed invariant set $$Z=\mathbb T\setminus\Gamma U$$ is not the whole torus, so by C2(b) it is finite, and by Lemma 1 it consists of torsion points. The distribution $$\hat c-\kappa$$ is supported on $$Z$$, hence is a finite sum of derivatives of delta functions at points of $$Z$$. Inverting the Fourier series, $$c(v)=\kappa\delta_{v,0}+\sum_{z\in Z}p_z(v)e^{iz\cdot v}$$ with polynomials $$p_z$$.

Let $$v_0\ne0$$ have $$c(v_0)\ne0$$. Let $$M\ge2$$ be an integer such that $$Mz\in2\pi\Lambda^\vee$$ for all $$z\in Z$$ and $$v_0\notin M\Lambda$$. On the coset $$v_0+M\Lambda$$, which does not contain 0, each factor $$e^{iz\cdot v}$$ is constant, so $$c$$ coincides there with a single polynomial $$p$$. The points of the coset outside the finitely many shells form a Zariski-dense subset of $$\mathbb R^4$$ (a polynomial vanishing on them, multiplied by $$\prod_q(Q-q)$$, vanishes on the whole coset, which is Zariski dense), and $$p$$ vanishes on them. So $$p\equiv0$$ and $$c(v_0)=0$$, a contradiction. $$\square$$

</div>

The proposition applies to the kernels of causal-set-inspired lattice operators that sum interior-count weights over finitely many shells; on a lattice with a Zariski-dense $$\Gamma$$ such a kernel is a genuine distribution rather than a function.

## 5. Invariant states

An operator is a special case of what a quantum theory provides. The two-point function of a state is more general, because it covers interacting and non-perturbative states.

<div class="theorem" markdown="1">

**Theorem 2** (invariant two-point functions are flat plus periodic). Assume (H1) and C2. Let $$\omega$$ be a positive linear functional on a $$*$$-algebra containing elements $$\phi(x)$$, $$x\in\Lambda$$, and set $$W(x,y)=\omega(\phi(x)^*\phi(y))$$. Suppose $$W(x+a,y+a)=W(x,y)$$ for $$a\in\Lambda$$ and $$W(gx,gy)=W(x,y)$$ for $$g\in\Gamma$$. Then

$$
W(x,y)=a_0\,\delta_{x,y}+\sum_{O}c_O\,\frac1{\lvert O\rvert }\sum_{k\in O}e^{ik\cdot(y-x)}, \tag{3}
$$

where $$a_0\ge0$$, the sum runs over the countably many finite $$\Gamma$$-orbits $$O$$ of torsion points of $$\mathbb T$$, $$c_O\ge0$$, and $$a_0+\sum_Oc_O=W(0,0)<\infty$$.

</div>

The spectral measure of the two-point function is a multiple of Haar measure plus atoms at torsion points. No part of it depends continuously on $$k$$, and the atoms at rational wave vectors give correlations that do not decay with distance.

<div class="proof" markdown="1">

*Proof.* Positivity gives $$\sum\bar\lambda_x\lambda_yW(x,y)=\omega\big((\sum\lambda_x\phi(x))^*(\sum\lambda_y\phi(y))\big)\ge0$$. Put $$w(v)=W(0,v)$$; translation invariance gives $$W(x,y)=w(y-x)$$, and $$w$$ is a positive-definite function on $$\Lambda$$. By Bochner's theorem for the discrete group $$\Lambda$$, $$w(v)=\int_{\mathbb T}e^{ik\cdot v}d\mu(k)$$ for a unique finite positive measure $$\mu$$ with $$\mu(\mathbb T)=w(0)$$. From $$w(gv)=W(g0,gv)=w(v)$$ and (1), the measure $$g_*\mu$$ has the same Fourier coefficients as $$\mu$$, so $$\mu$$ is $$\Gamma$$-invariant.

Decompose $$\mu=\mu_{\rm c}+\mu_{\rm p}$$ into non-atomic and purely atomic parts; both are $$\Gamma$$-invariant because $$\Gamma$$ acts by homeomorphisms. By C2(a), $$\mu_{\rm c}=a_0\cdot\text{Haar}$$ (Haar normalised to total mass 1). The atomic part is $$\sum_km_k\delta_k$$ with $$m_k$$ constant on orbits and summable, so an orbit carrying positive mass is finite, and by Lemma 1 its points are torsion. With $$c_O=\lvert O\rvert m_k$$ for $$k\in O$$ this is (3), and evaluation at $$x=y$$ gives the sum rule. $$\square$$

</div>

If only a $$\Gamma$$-stable finite-index sublattice $$\Lambda'$$ is a translation symmetry, the theorem applies to the sub-algebra generated by $$\phi(x)$$, $$x\in\Lambda'$$.

Positivity is used in an essential way. The shell function $$c=\mathbf 1_{\{Q=1\}}$$ is $$\Gamma$$-invariant, defines a translation-invariant kernel, and has $$c(0)=0$$; in (3) this would force $$a_0=c_O=0$$ and so $$w\equiv0$$. It is therefore the two-point function of no positive state, and a kernel of this kind can arise only from a state on an indefinite-metric space. For such kernels Proposition 1 still forbids a dispersion relation on an open set of momenta.

<div class="remark" markdown="1">

**Remark 2** (infinite invariant measures). The finiteness of $$\mu$$ is essential. Infinite $$\Gamma$$-invariant measures on $$\mathbb T$$ exist, for example counting measure on a dense orbit, but they are not Radon on the compact torus and cannot be spectral measures of a state, for which $$\mu(\mathbb T)=W(0,0)$$.

</div>

## 6. The wave packet is itself a regulator

Section 4 shows that a $$\Gamma$$-invariant lattice operator has no symbol to expand. One can ask instead what a localised probe sees. A finite-norm packet $$\phi(x)=e^{ip\cdot x}f(x)$$, with $$f$$ real and decaying, is what a dispersion relation is read off from. Let $$D$$ have kernel $$c$$ with $$c(-v)=c(v)$$, so $$\langle\phi,D\phi\rangle=\sum_{x,v}\bar\phi(x)c(v)\phi(x+v)$$. Summing over $$x$$ gives

$$
\frac{\langle\phi,D\phi\rangle}{\lVert \phi\rVert ^2}=\frac{1}{E(0)}\sum_{v}c(v)\cos(p\cdot v)E(v),\qquad E(v)=\sum_xf(x)f(x+v), \tag{4}
$$

where $$E$$ is the envelope autocorrelation. For a Gaussian envelope of width $$L$$ in the Euclidean norm, $$E(v)\propto e^{-\lvert v\rvert _{\rm E}^2/4L^2}$$ up to corrections exponentially small in $$L^2$$; this form is used below. Expanding the cosine, the packet sees $$S_0-\tfrac12p_\mu p_\nu M^{\mu\nu}+O(p^4)$$ with

$$
M^{\mu\nu}=\sum_vc(v)E(v)\,v^\mu v^\nu,\qquad S_0=\sum_vc(v)E(v). \tag{5}
$$

A Lorentz-invariant low-energy dispersion requires $$M^{\mu\nu}\propto\eta^{\mu\nu}$$, that is $$\rho:=M^{00}/(-M^{11})=+1$$ with the three spatial entries equal.

<div class="proposition" markdown="1">

**Proposition 2** (positive kernels give an elliptic packet response). (i) If $$c\ge0$$ and $$c(v)\ne0$$ for some $$v\ne0$$, then $$M$$ is positive semidefinite and non-zero, hence not proportional to $$\eta^{\mu\nu}$$, and $$\rho\le0$$. (ii) For a shell kernel $$c=\mathbf 1_{\{Q=q\}}$$ on a lattice for which $$c$$ and $$E$$ are symmetric under permutations of the three spatial axes,

$$
\rho=-\frac{3\langle t^2\rangle}{\langle t^2\rangle-q},\qquad \langle t^2\rangle=\frac{\sum_{v}c(v)E(v)v_0^2}{S_0}, \tag{6}
$$

exactly. (iii) If the shell contains a non-zero vector and (F) holds, then $$\rho\to-3$$ as $$L\to\infty$$. For $$q=0$$, $$\rho=-3$$ for every $$L$$.

</div>

<div class="proof" markdown="1">

*Proof.* (i) $$x^TMx=\sum cE\,(x\cdot v)^2\ge0$$ for every real $$x$$, and the trace $$\sum cE\,\lvert v\rvert _{\rm E}^2$$ is positive. A non-zero positive semidefinite form is not proportional to the indefinite $$\eta^{\mu\nu}$$. Moreover $$M^{11}\ge0$$ and $$M^{00}>0$$, so $$\rho\le0$$ where defined.

(ii) On the shell $$v_0^2=q+\lvert \vec v\rvert ^2$$, and by the permutation symmetry $$M^{11}=\tfrac13\sum cE\,\lvert \vec v\rvert ^2=\tfrac13(M^{00}-qS_0)$$. Hence $$\rho=-3M^{00}/(M^{00}-qS_0)$$, which is (6).

(iii) Let $$T_0>0$$ and let $$n(T_0)$$ be the number of shell points with $$\lvert v_0\rvert \le T_0$$; it is finite because $$\lvert \vec v\rvert ^2=v_0^2-q$$. The shell is infinite by (F), and $$E_L\to1$$ pointwise as $$L\to\infty$$, so $$S_0\to\infty$$ by Fatou's lemma. The weight carried by $$\lvert v_0\rvert \le T_0$$ is at most $$n(T_0)$$, so $$\langle t^2\rangle\ge T_0^2(1-n(T_0)/S_0)\to T_0^2$$. As $$T_0$$ is arbitrary, $$\langle t^2\rangle\to\infty$$ and (6) gives $$\rho\to-3$$. For $$q=0$$, (6) equals $$-3$$ identically. $$\square$$

</div>

The tensor tends to $$M^{00}\,\mathrm{diag}(1,\tfrac13,\tfrac13,\tfrac13)$$, that of null dust: boosted copies of the shell lie almost along the light cone and dominate the sum, and the packet sees a Euclidean-elliptic quadratic form $$(k\cdot u)^2+\lvert k_\perp\rvert ^2/3$$ instead of $$\omega^2-\lvert \vec k\rvert ^2$$. The vector $$u$$ is the time direction of the packet's rest frame; a packet boosted by a different rapidity sees a different $$M$$. The regulator therefore picks a frame, and the resulting Lorentz-violating coefficient is of order one rather than suppressed (see [Where a preferred frame leaks into low-energy physics](/quantum-gravity/lorentz-violation-leakage/) for what an order-one coefficient does).

*Numerical check.* For $$\Lambda=\mathbb Z^4$$ and the Gaussian envelope above, direct enumeration of the shell $$Q=1$$ gives $$\rho=-3.0931,\,-3.0232,\,-3.0059$$ at $$L=4,8,16$$ (sums over 1777, 6985 and 28677 shell points with $$t>0$$), equal to the closed form (6) to all digits printed. The shell $$Q=0$$ gives $$-3$$ to rounding at the same $$L$$. The number of points of the shell $$Q=1$$ in $$\mathbb Z^4$$ with $$t>0$$ and Euclidean norm at most $$R=25,50,100,200$$ is $$473,\,2257,\,8617,\,33861$$, growing like $$R^2$$ (an observation here; asymptotic counts of integer points on hyperboloids are known equidistribution statements). For kernels with weights of both signs, such as retuned d'Alembertians, part (i) does not apply, and a value near $$-3$$ requires that the leading shell contributions do not cancel; the values found for an orbit lattice are on the page on [single realisations](/quantum-gravity/per-realisation-kinematics/).

## 7. Spontaneous breaking of the discrete Lorentz group

Theorem 2 assumes an exactly invariant state. Suppose instead that the action is invariant under $$\Gamma\ltimes\Lambda$$ and the state is not. Let $$S\subset\Gamma\ltimes\Lambda$$ be the group of affine maps that leave the state invariant, $$\Lambda'=S\cap\Lambda$$ the unbroken translations, and $$H=S\cap\Gamma$$ the unbroken linear part. The group $$H$$ preserves $$\Lambda'$$, because $$h\,t_a\,h^{-1}=t_{ha}$$. Write $$\bar H$$ for the Zariski closure of $$H$$ in $$GL(4,\mathbb R)$$ and $$L_H$$ for the identity component, in the Lie-group topology, of its real points. Then $$L_H\subset SO(1,3)^\circ$$, it has finite index in $$\bar H(\mathbb R)$$, and it is normalised by $$H$$. Either $$L_H=SO(1,3)^\circ$$, which is (H1) for $$H$$, or $$L_H$$ is a proper connected subgroup of $$SO(1,3)^\circ$$.

<div class="theorem" markdown="1">

**Theorem 3** (dichotomy for the unbroken group). Assume C2 and C3.

(a) If $$\Lambda'$$ has finite index in $$\Lambda$$ and $$H$$ satisfies (H1) as a group acting on $$\Lambda'$$, then the two-point function of the field restricted to $$\Lambda'$$ has the form (3) with $$\Lambda'$$ and $$H$$ in place of $$\Lambda$$ and $$\Gamma$$. The same holds if $$S$$ has finite index in $$\Gamma\ltimes\Lambda$$.

(b) If $$L_H\ne SO(1,3)^\circ$$, then $$L_H$$ leaves invariant a proper non-zero subspace of $$\mathbb R^4$$, and at least one of the following holds. (i) $$L_H$$ leaves invariant a non-degenerate proper subspace, or two distinct null lines. (ii) If (i) fails, $$L_H$$ leaves invariant exactly one null line $$\ell$$, and $$\ell\cap\Lambda\ne0$$. (iii) If (i) fails, $$L_H$$ leaves invariant exactly one null line, and it contains no lattice vector. In cases (i) and (ii) there is an $$H$$-invariant symmetric tensor $$T^{\mu\nu}$$ that is not a multiple of $$\eta^{\mu\nu}$$, so the term $$T^{\mu\nu}\partial_\mu\phi\,\partial_\nu\phi$$ is invariant under all unbroken symmetries. Case (iii) occurs only if $$L_H$$ does not act completely reducibly.

</div>

Part (a) says that breaking to a finite-index subgroup changes nothing. Part (b) says that when the unbroken group is thin, the remaining symmetry no longer forces the quadratic action to be Lorentz invariant; whether it is, at low energy, then depends on couplings the symmetry leaves free.

<div class="proof" markdown="1">

*Proof.* (a) Apply Theorem 2 to the lattice $$\Lambda'$$ and the group $$H$$, with the field algebra generated by $$\phi(x)$$, $$x\in\Lambda'$$. If $$S$$ has finite index, then $$\Lambda'$$ and $$H$$ have finite index in $$\Lambda$$ and $$\Gamma$$, and $$H$$ inherits (H1) from $$\Gamma$$.

(b) *Reducibility.* $$L_H$$ is a connected subgroup of $$SO(1,3)^\circ$$ that preserves the Lorentzian form. If it acted irreducibly, C3 would make it conjugate to $$SO(1,3)^\circ$$, and being contained in $$SO(1,3)^\circ$$ with the same dimension it would equal it. So it preserves a proper non-zero subspace $$U$$.

*Non-degenerate case.* Suppose some invariant proper subspace $$U$$ is non-degenerate, or two distinct null lines are invariant (they span a Lorentzian plane, which is non-degenerate). Put $$W=U^\perp$$. A non-degenerate subspace of $$\mathbb R^{1,3}$$ is either negative definite, or has a negative definite orthogonal complement. Let $$b_U(p,q)=B(P_Up,P_Uq)$$ with $$P_U$$ the orthogonal projection onto $$U$$; then $$b_U+b_W=B$$. Each $$h\in H$$ maps $$U$$ to another $$L_H$$-invariant subspace, because $$H$$ normalises $$L_H$$. The orbit $$HU$$ is finite, since the stabiliser of $$U$$ contains $$H\cap L_H$$, of finite index in $$H$$. Put $$T=\sum_{U'\in HU}b_{U'}$$, which is $$H$$-invariant. If $$U$$ is negative definite, each $$b_{U'}$$ is negative semidefinite and non-zero, so $$T$$ is also, and cannot be a multiple of the indefinite $$B$$. Otherwise $$W$$ is negative definite, $$b_{U'}=B-b_{W'}$$ with $$-b_{W'}$$ positive semidefinite and non-zero, and $$T=\lvert HU\rvert \,B+P$$ with $$P$$ positive semidefinite and non-zero. If $$T=cB$$ then $$P=(c-\lvert HU\rvert )B$$, impossible for a non-zero positive semidefinite $$P$$ and an indefinite $$B$$.

*Degenerate case.* Suppose every invariant proper subspace is degenerate and at most one null line is invariant. A degenerate invariant subspace $$U$$ has a radical $$U\cap U^\perp$$ that is totally null, hence a null line, invariant under $$L_H$$. So there is exactly one invariant null line $$\ell$$, and $$\bar H(\mathbb R)$$, which normalises $$L_H$$, preserves it. If $$\ell=\mathbb Rn_0$$ with $$n_0\in\Lambda$$, then $$H\subset GL(\Lambda)$$ preserves the rank-one lattice $$\ell\cap\Lambda$$ and acts on it by $$\pm1$$, so $$T=n_0\otimes n_0$$ is $$H$$-invariant. It is not a multiple of $$\eta^{\mu\nu}$$: $$T^{00}>0$$ and $$T^{11}\ge0$$, while a multiple $$c\eta^{\mu\nu}$$ has $$T^{00}$$ and $$T^{11}$$ of opposite sign. This is case (ii). Otherwise the line contains no lattice vector: case (iii).

Finally, suppose $$L_H$$ acts completely reducibly and preserves a null line $$\ell$$. An invariant complement $$C$$ of $$\ell$$ gives an invariant two-dimensional subspace $$C\cap\ell^\perp$$ complementary to $$\ell$$ in $$\ell^\perp$$, and its orthogonal complement is an invariant Lorentzian plane containing $$\ell$$, which contains a second null line, also invariant because $$L_H$$ is connected. So (iii) requires that $$L_H$$ does not act completely reducibly. $$\square$$

</div>

The very special relativity groups of Cohen and Glashow [[Cohen–Glashow]](#ref-cg), which fix a null direction up to scale, are the case in which the invariant two-tensors reduce to the metric. For a rational null direction this escape is closed by integrality, because $$H$$ cannot rescale a primitive lattice vector (case (ii)). The case of an irrational unique null line, (iii), is not settled here.

<div class="example" markdown="1">

**Example 1** (an explicitly broken model). Let $$\Gamma=O^+(1,3;\mathbb Z)$$ and let $$u$$ run over the orbit $$\Gamma e_0$$; for each $$u$$ choose $$\gamma_u\in\Gamma$$ with $$\gamma_ue_0=u$$. For each $$u$$ let $$S_u[\phi^{(u)}]$$ be a nearest-neighbour lattice action for a field $$\phi^{(u)}$$, written in the frame $$\gamma_u\{e_\mu\}$$ of $$\mathbb Z^4$$, with a gradient coefficient $$c_s$$ for each species $$s$$ that is the same for the three spatial directions. The stabiliser of $$e_0$$ in $$\Gamma$$ is the finite group of signed spatial permutations, which leaves each $$c_s$$ free. The sum over $$u$$ of these actions is invariant under $$\Gamma\ltimes\Lambda$$, with $$g$$ permuting the labels $$u$$. No state is $$\Gamma$$-invariant, since that would need a $$\Gamma$$-invariant probability measure on the infinite set $$\Gamma/\mathrm{Stab}(e_0)$$. Each sector has the dispersion relation of the nearest-neighbour lattice in the frame $$u$$, which is Lorentz invariant at leading order in momentum when all $$c_s=1$$. The residual symmetry does not tie the $$c_s$$ to 1, so equality of the species' speeds is a condition on couplings. This is an example of propagation with a Zariski-dense symmetry group of the action; it does not contradict Theorem 2, because every state breaks $$\Gamma$$.

</div>

## 8. Cayley-momentum covariance is not a symmetry of a point set

Lorente obtains a covariant lattice Klein–Gordon operator by letting Lorentz transformations act on the variables $$\varphi(k)_\mu=2\tan(k_\mu/2)$$, the Cayley momentum, through $$T_\Lambda=\varphi^{-1}\circ\Lambda\circ\varphi$$ [[Lorente]](#ref-lorente). The symbol $$\eta(\varphi(k),\varphi(k))-m^2$$ is invariant under this action. The question is whether the action is induced by a transformation of the points of a lattice.

<div class="proposition" markdown="1">

**Proposition 3** (point maps induce only automorphisms). Let $$\sigma$$ be a bijection of $$\mathbb Z^4$$, and suppose there are a measurable map $$T:\mathbb T^4\to\mathbb T^4$$ and unit complex numbers $$c(k)$$ with $$\chi_k(\sigma^{-1}(x))=c(k)\chi_{T(k)}(x)$$ for all $$k$$ and $$x$$; that is, the composition operator of $$\sigma$$ maps each character to a multiple of a character. Then $$\sigma$$ is affine with linear part in $$GL(4,\mathbb Z)$$, and $$T$$ is a linear automorphism of the torus.

</div>

<div class="proof" markdown="1">

*Proof.* Evaluating at $$x=0$$ gives $$c(k)=\chi_k(\sigma^{-1}(0))$$. With $$\sigma'(x)=\sigma^{-1}(x)-\sigma^{-1}(0)$$ we get $$e^{ik\cdot\sigma'(x)}=e^{iT(k)\cdot x}$$ for all $$k,x$$. The left side is multiplicative in $$k$$, so $$e^{iT(k+k')\cdot x}=e^{iT(k)\cdot x}e^{iT(k')\cdot x}$$ for all $$x\in\mathbb Z^4$$, and characters separate points of the torus: $$T$$ is a homomorphism. A measurable homomorphism of the torus is continuous, hence $$T(k)=Ak$$ for an integer matrix $$A$$. Then $$e^{ik\cdot\sigma'(x)}=e^{ik\cdot A^Tx}$$ for all $$k$$, so $$\sigma'(x)=A^Tx$$. Since $$\sigma$$ is a bijection, $$A^T\in GL(4,\mathbb Z)$$. $$\square$$

</div>

<div class="proposition" markdown="1">

**Proposition 4** ($$T_\Lambda$$ is additive only for monomial $$\Lambda$$). Let $$\Lambda\in O(1,3)$$. Then $$T_\Lambda(k+k')=T_\Lambda(k)+T_\Lambda(k')$$ for all small $$k,k'$$ if and only if $$\Lambda$$ has exactly one non-zero entry in each row and column.

</div>

<div class="proof" markdown="1">

*Proof.* Expand $$\varphi(k)=k+k^{(3)}/12+O(k^5)$$ and $$\varphi^{-1}(y)=y-y^{(3)}/12+O(y^5)$$, where $$(\cdot)^{(3)}$$ is the componentwise cube. Then $$T_\Lambda(k)=\Lambda k+c(k)+O(k^5)$$ with $$c(k)=\tfrac1{12}\big(\Lambda k^{(3)}-(\Lambda k)^{(3)}\big)$$, homogeneous of degree 3. If $$T_\Lambda$$ is additive then $$T_\Lambda(2k)=2T_\Lambda(k)$$, and the cubic terms give $$6c(k)=0$$, that is $$\Lambda k^{(3)}=(\Lambda k)^{(3)}$$ for all $$k$$. If some row $$i$$ of $$\Lambda$$ has two non-zero entries $$\Lambda_{ij},\Lambda_{il}$$, the coefficient of $$k_j^2k_l$$ in $$(\Lambda k)_i^3$$ is $$3\Lambda_{ij}^2\Lambda_{il}\ne0$$, while $$(\Lambda k^{(3)})_i=\sum_j\Lambda_{ij}k_j^3$$ has no such monomial. So $$\Lambda$$ is monomial (a matrix in $$O(1,3)$$ is invertible, so no row or column vanishes, and a monomial matrix has one non-zero entry in each). Conversely, a monomial $$\Lambda$$ is a signed permutation, which commutes with the odd componentwise function $$\varphi$$, so $$T_\Lambda=\Lambda$$ is linear. $$\square$$

</div>

By Proposition 3, a genuine boost gives a $$T_\Lambda$$ that is induced by no point map: it is a symmetry of the dispersion relation only. Numerically, in 1+1 dimensions for a boost of rapidity $$1/2$$ and $$k=r\,(0.6,0.8)$$, $$\lvert T_\Lambda(2k)-2T_\Lambda(k)\rvert =8.5\cdot10^{-5},\ 6.8\cdot10^{-4},\ 5.3\cdot10^{-3},\ 3.9\cdot10^{-2}$$ at $$r=0.05,0.1,0.2,0.4$$, with ratio to $$r^3$$ equal to $$0.68,0.68,0.66,0.61$$, as the cubic term predicts, and the invariant $$\eta(\varphi(k),\varphi(k))$$ is preserved to rounding.

<div class="proposition" markdown="1">

**Proposition 5** (vertices). Suppose an interaction is invariant under lattice translations, so that its vertices conserve $$\sum_jk_j$$ modulo $$2\pi$$, and suppose that for a boost $$\Lambda$$ with non-additive $$T_\Lambda$$, the map $$T_\Lambda$$ sends every triple $$(k,k',-k-k')$$ of momenta into a triple with vanishing sum. Then $$T_\Lambda$$ is additive, a contradiction. Hence no lattice-translation-invariant three-point vertex is covariant under $$T_\Lambda$$.

</div>

<div class="proof" markdown="1">

*Proof.* Applied to $$(k,-k,0)$$ the hypothesis gives $$T(0)=0$$ and $$T(-k)=-T(k)$$; applied to $$(k,k',-k-k')$$ it gives $$T(k)+T(k')+T(-k-k')=0$$, that is $$T(k+k')=T(k)+T(k')$$. This contradicts Proposition 4. $$\square$$

</div>

<div class="remark" markdown="1">

**Remark 3** (the covariant alternative). A vertex that is covariant must instead conserve $$\sum_j\varphi(k_j)$$, and it cannot conserve $$\sum_jk_j$$ as well, because $$\varphi$$ is not additive. The composition law $$k\oplus k'=\varphi^{-1}(\varphi(k)+\varphi(k'))$$ is commutative, associative and covariant; it is the composition law of the doubly special relativity of Judes and Visser [[Judes–Visser]](#ref-jv), obtained from a non-linear realisation of the Lorentz group by a change of variables. Identifying $$\varphi(k)$$ with the conserved momentum turns the theory into one with continuous translations and the linear Lorentz action, with the lattice as a labelling of a basis of $$L^2$$; we do not prove this identification in detail. It follows the distinction between momentum and translation generator emphasised by Mignemi [[Mignemi]](#ref-mignemi). A theory with two inequivalent notions of momentum is not excluded, only shown to carry no lattice.

</div>

If the lattice translations are kept as symmetries, covariance fails at the vertex and the lattice theory is Lorentz violating. The free massless dispersion obtained from $$\varphi(\omega)=\lvert \varphi(\vec k)\rvert $$ is, to cubic order, $$\omega=\lvert \vec k\rvert -\frac{\lvert \vec k\rvert ^3}{12}\big(1-\sum_in_i^4\big)$$ with $$n=\vec k/\lvert \vec k\rvert $$, a direction-dependent correction that vanishes along the axes.

<div class="proposition" markdown="1">

**Proposition 6** (automorphic actions are linear lattice actions or continuum actions). Let $$M$$ be a connected four-dimensional Lie group on which a Zariski-dense subgroup $$\Gamma$$ of $$O(1,3)$$ acts by group automorphisms, with differential at the identity equal to the vector representation. Then $$M\cong\mathbb R^4$$ or $$M\cong\mathbb T^4$$.

</div>

<div class="proof" markdown="1">

*Proof.* The Lie bracket on $$\mathfrak m\cong\mathbb R^4$$ is a $$\Gamma$$-equivariant linear map $$\Lambda^2\mathbb R^4\to\mathbb R^4$$. Equivariance under $$g$$ is a polynomial condition, so the set of $$g$$ for which the bracket is equivariant is Zariski closed and contains $$\Gamma$$, hence $$SO(1,3)^\circ$$. As a representation of $$SO(1,3)^\circ$$, $$\Lambda^2\mathbb R^4=\mathfrak{so}(1,3)$$ is the adjoint representation of a simple real Lie algebra, hence irreducible of dimension 6. It contains no copy of the four-dimensional vector representation, so by Schur's lemma the equivariant bracket vanishes. Thus $$M=\mathbb R^4/K$$ with $$K=\ker\exp$$ discrete. Its real span is a $$\Gamma$$-invariant, hence $$SO(1,3)^\circ$$-invariant, subspace, so it is $$0$$ or $$\mathbb R^4$$, and a discrete subgroup spanning $$\mathbb R^4$$ is a lattice. $$\square$$

</div>

In the torus case $$\Gamma$$ acts linearly through $$GL(4,\mathbb Z)$$, which is the setting of Theorems 1 and 2. In the $$\mathbb R^4$$ case there is no discreteness. What remains is a Lorentz action on momentum space that is not by group automorphisms, as in $$\kappa$$-Poincaré constructions, which are not realised on a point set.

## 9. Spectral triples on the Brillouin quotient

Proposition 6 leaves open whether the non-commutative quotient $$C(\mathbb T^n)\rtimes\Gamma$$ of the Brillouin torus by the integral Lorentz group could carry a geometry of finite spectral dimension. On $$H=L^2(\mathbb T^n)\otimes V$$ the algebra is generated by the translation unitaries $$T_x$$ and the operators $$U_g$$ of (2), taken with $$\sigma$$ unitary. A *covariant Brillouin triple* is a self-adjoint $$D$$ with compact resolvent on $$H$$ such that $$T_{e_i}$$ and $$U_g$$, for $$g$$ in a finite generating set of $$\Gamma$$, and their inverses, preserve $$\mathrm{dom}\,D$$ and have bounded commutators with $$D$$. It is *exactly equivariant* if $$U_gDU_g^{-1}=D$$ for all $$g\in\Gamma$$.

<div class="theorem" markdown="1">

**Theorem 4** (exact equivariance excludes compact resolvent). Assume (F) for $$\Gamma$$ acting on $$\Lambda=\mathbb Z^n$$. There is no self-adjoint operator $$D$$ with compact resolvent on $$L^2(\mathbb T^n)\otimes V$$, with $$V$$ finite-dimensional and $$\sigma$$ a representation of $$\Gamma$$ on $$V$$ (not necessarily unitary), that satisfies $$U_gDU_g^{-1}=D$$ for all $$g\in\Gamma$$.

</div>

<div class="proof" markdown="1">

*Proof.* Let $$E$$ be an eigenspace of $$D$$; it is finite-dimensional because the resolvent is compact, and invariant under every $$U_g$$. Let $$F\subset L^2(\mathbb T^n)$$ be the span of the components $$(\mathrm{id}\otimes\ell)(e)$$, $$e\in E$$, $$\ell\in V^*$$; it is finite-dimensional. For $$f=(\mathrm{id}\otimes\ell\sigma(g))e$$ one has $$K_gf=(\mathrm{id}\otimes\ell)(U_ge)\in F$$, and as $$\ell$$ ranges over $$V^*$$ so does $$\ell\sigma(g)$$, because $$\sigma(g)$$ is invertible; so $$F$$ is $$K$$-invariant. Its projection onto $$L^2_0$$ is invariant and finite-dimensional, hence zero by Lemma 2(i). Thus every component of every eigenvector is constant, $$E\subset1\otimes V$$. The eigenvectors of a self-adjoint operator with compact resolvent span $$H$$, so $$H=1\otimes V$$ is finite-dimensional, which contradicts $$\dim L^2(\mathbb T^n)=\infty$$. $$\square$$

</div>

The Lorentzian Dirac-type operator $$D_L=\gamma^\mu x_\mu$$ in position space is exactly covariant, but each eigenvalue has infinite multiplicity (the value $$\sqrt Q$$ is attained on a whole orbit), so its resolvent is not compact; this is consistent with the theorem.

If covariance is only required up to bounded terms, the obstruction is a different one: the translations are exponentially distorted in $$G=\Lambda\rtimes\Gamma$$. Fix a finite generating set of $$G$$, and write $$\lvert x\rvert _G$$ for the word length of the translation $$t_x$$, $$x\in\Lambda$$.

<div class="lemma" markdown="1">

**Lemma 3** (exponential distortion of translations). (i) Let $$A\in SL(2,\mathbb Z)$$ with $$\operatorname{tr}A>2$$ and $$G_A=\mathbb Z^2\rtimes\langle A\rangle$$. There is $$C$$ with $$\lvert x\rvert _{G_A}\le C\log(2+\lvert x\rvert )$$ for all $$x\in\mathbb Z^2$$. (ii) Let $$\Gamma$$ be a subgroup of finite index in $$O(1,3;\mathbb Z)$$ and $$G=\mathbb Z^4\rtimes\Gamma$$. Then $$\lvert x\rvert _G\le C\log(2+\lvert x\rvert )$$ for all $$x\in\mathbb Z^4$$.

</div>

<div class="proof" markdown="1">

*Proof.* (i) Let $$\lambda>1$$ be the larger eigenvalue of $$A$$. Write $$x=u(x)e_u+s(x)e_s$$ in the eigenbasis, so that $$A$$ acts by $$(u,s)\mapsto(\lambda u,s/\lambda)$$. Let $$\rho$$ be the covering radius of $$\mathbb Z^2$$ in the sup norm of these coordinates, $$M=\rho$$ and $$\sigma_0=\lambda\rho$$.

*Step 1 (points near the unstable axis).* Suppose $$\lvert s(y)\rvert \le\sigma_0$$. The rectangle $$\{\lvert u\rvert \le M,\ \lvert s-s(y)\rvert \le\sigma_0/\lambda\}$$ contains a ball of sup-radius $$\rho$$, hence a lattice point $$v$$ of the finite set $$F=\{v\in\mathbb Z^2:\lvert u(v)\rvert \le M,\ \lvert s(v)\rvert \le\sigma_0+\sigma_0/\lambda\}$$. Put $$y'=A^{-1}(y-v)$$, an integer vector because $$\det A=1$$. Then $$\lvert s(y')\rvert =\lambda\lvert s(y)-s(v)\rvert \le\sigma_0$$ and $$\lvert u(y')\rvert \le(\lvert u(y)\rvert +M)/\lambda$$. Iterating, after $$k$$ steps $$\lvert u\rvert \le\lvert u(y)\rvert \lambda^{-k}+M/(\lambda-1)$$, so after $$k=\log_\lambda(2+\lvert u(y)\rvert )+O(1)$$ steps the point lies in a fixed finite set $$B_0$$. This gives $$y=\sum_{j<k}A^jv_j+A^ky_k$$ with $$v_j\in F$$ and $$y_k\in B_0$$.

*Step 2 (the word).* Let $$a$$ be the generator acting as $$A$$, so $$a\,t_za^{-1}=t_{Az}$$. Then $$t_{\sum_{j<k}A^jv_j}=t_{v_0}\,a\,t_{v_1}\,a\,t_{v_2}\cdots a\,t_{v_{k-1}}\,a^{-(k-1)}$$ and $$t_{A^ky_k}=a^kt_{y_k}a^{-k}$$. The elements $$t_v$$ with $$v\in F\cup B_0$$ have bounded length, so $$\lvert t_y\rvert _{G_A}\le C_1k+C_2$$.

*Step 3 (general $$x$$).* The same construction with $$A^{-1}$$ treats points with $$\lvert u\rvert \le\sigma_0$$ and large $$\lvert s\rvert $$. Given $$x\in\mathbb Z^2$$, choose a lattice point $$p$$ within sup-distance $$\rho$$ of $$u(x)e_u$$, then a lattice point $$q$$ within $$\rho$$ of the stable-axis point of $$x-p$$. Then $$x=p+q+r$$ with $$\lvert s(p)\rvert \le\rho$$, $$\lvert u(q)\rvert \le\rho$$, $$r$$ in a finite set, and $$\lvert p\rvert ,\lvert q\rvert \le C\lvert x\rvert $$. Hence $$\lvert x\rvert _{G_A}\le C\log(2+\lvert x\rvert )$$.

(ii) Let $$A=\begin{pmatrix}3&4\\2&3\end{pmatrix}$$, which preserves $$a^2-2b^2$$ since $$(3a+4b)^2-2(2a+3b)^2=a^2-2b^2$$. Let $$g_0\in GL(4,\mathbb R)$$ act as $$A$$ on the coordinates $$(a,b)$$ of $$ae_0+b(e_1+e_2)$$ and trivially on $$e_1-e_2$$ and $$e_3$$. The lattice $$\mathbb Z^4$$ contains $$P_0\oplus P_0^\perp$$, with $$P_0=\mathbb Z e_0+\mathbb Z(e_1+e_2)$$, with index 2. A power $$h=g_0^N$$ with $$A^N\equiv1\pmod2$$ acts trivially on the quotient of index 2, so $$h\in O(1,3;\mathbb Z)$$, and for suitable $$N$$ also $$h\in\Gamma$$. For $$\gamma\in\Gamma$$ the plane $$\gamma P$$, $$P=\mathbb RP_0$$, carries the element $$\gamma h\gamma^{-1}\in\Gamma$$, which acts on $$\gamma P_0=\mathbb Z^4\cap\gamma P$$ as a hyperbolic element of $$SL(2,\mathbb Z)$$ in the basis $$\gamma(e_0,e_1+e_2)$$. Part (i) applies on each such plane: it uses only the generator $$h$$-conjugate and translations by a basis, and the relation $$a\,t_za^{-1}=t_{az}$$ holds in $$G$$ for all $$z\in\mathbb Z^4$$. So $$\lvert z\rvert _G\le C\log(2+\lvert z\rvert )$$ for $$z\in P_0\cup\gamma P_0$$. The condition $$P\oplus\gamma P=\mathbb R^4$$ is the non-vanishing of a polynomial in the entries of $$\gamma$$ that is not identically zero on $$SO(1,3)^\circ$$ (rotate $$P$$ to a transverse plane), so by (H1) some $$\gamma\in\Gamma$$ satisfies it. Then $$P_0\oplus\gamma P_0$$ has finite index in $$\mathbb Z^4$$, and every $$x$$ is $$x_1+x_2+r$$ with $$x_1\in P_0$$, $$x_2\in\gamma P_0$$, $$\lvert x_i\rvert \le C\lvert x\rvert $$ and $$r$$ in a finite set of coset representatives. Then $$\lvert x\rvert _G\le\lvert x_1\rvert _G+\lvert x_2\rvert _G+\lvert r\rvert _G$$. $$\square$$

</div>

<div class="theorem" markdown="1">

**Theorem 5** (exponential Weyl law). Let $$G=\mathbb Z^n\rtimes\Gamma$$ with either $$n=2$$ and $$\Gamma=\langle A\rangle$$ as in Lemma 3(i), or $$n=4$$ and $$\Gamma$$ of finite index in $$O(1,3;\mathbb Z)$$. Let $$D$$ be a covariant Brillouin triple on $$L^2(\mathbb T^n)\otimes V$$ with $$\sigma$$ unitary. Then there are $$c,\kappa,R_0>0$$ with

$$
N_D(R):=\operatorname{rank}\mathbf 1_{[-R,R]}(D)\ \ge\ c\,e^{\kappa R}\qquad(R\ge R_0). \tag{7}
$$

In particular $$D$$ is $$p$$-summable for no $$p<\infty$$.

</div>

Bounded commutators with the generators give a commutator with a word of length $$m$$ that grows only linearly in $$m$$, so exponentially many translates of one vector fit into a spectral window of width linear in $$m$$. Lemma 3 supplies the exponential count.

<div class="proof" markdown="1">

*Proof.* The vectors $$\zeta_x=T_x\zeta$$, $$x\in\Lambda$$, with $$\zeta=1\otimes v_0$$ and $$\lVert v_0\rVert =1$$, are orthonormal. Let $$L$$ bound the commutators of $$D$$ with the generators and their inverses. For a product $$u$$ of $$m$$ generators or inverses, all partial products preserve $$\mathrm{dom}\,D$$, and the Leibniz rule gives $$\lVert [D,u]\rVert \le Lm$$. Choose $$\xi\in\mathrm{dom}\,D$$ with $$\lVert \xi-\zeta\rVert \le\tfrac14$$ and put $$K=\lVert D\xi\rVert $$. For $$\lvert x\rvert _G\le m$$, $$\lVert DT_x\xi\rVert \le\lVert T_xD\xi\rVert +\lVert [D,T_x]\xi\rVert \le K+\tfrac54Lm$$. Put $$R=4K+5Lm$$ and $$P=\mathbf 1_{[-R,R]}(D)$$. Then $$\lVert (1-P)T_x\xi\rVert \le\lVert DT_x\xi\rVert /R\le\tfrac14$$, so $$\lVert (1-P)\zeta_x\rVert \le\tfrac14+\lVert \zeta-\xi\rVert \le\tfrac12$$ and $$\langle\zeta_x,P\zeta_x\rangle=1-\lVert (1-P)\zeta_x\rVert ^2\ge\tfrac34$$. Bessel's inequality for the orthonormal family gives $$\operatorname{rank}P\ge\sum_x\langle\zeta_x,P\zeta_x\rangle\ge\tfrac34\,\#\{x:\lvert x\rvert _G\le m\}$$. By Lemma 3 every $$x$$ with $$\lvert x\rvert _\infty\le r$$ has $$\lvert x\rvert _G\le C\log(2+r)+c_0$$, so for $$r=e^{(m-c_0)/C}-2$$ the count is at least $$(2r+1)^n\ge e^{n(m-c_0)/C}$$. Since $$R$$ is affine in $$m$$, this is (7) with $$\kappa=n/(5LC)$$. A $$p$$-summable $$D$$ has $$N_D(R)\le C'(1+R)^p$$, which (7) violates. $$\square$$

</div>

The estimate is sharp in the following sense. The operator $$D=\lvert x\rvert _G$$ acting by multiplication in position space has bounded commutators with all generators (since $$\lvert x+a\rvert _G\le\lvert x\rvert _G+\lvert a\rvert _G$$ and $$\lvert gx\rvert _G\le\lvert x\rvert _G+2\lvert g\rvert _G$$), compact resolvent, and $$\operatorname{Tr}e^{-tD^2}=\sum_xe^{-t\lvert x\rvert _G^2}<\infty$$ for every $$t>0$$, because the number of elements of $$G$$ of length at most $$R$$ is at most exponential in $$R$$. It is therefore $$\theta$$-summable, and by Theorem 5 not finitely summable.

Numerically, with $$A=\begin{pmatrix}3&4\\2&3\end{pmatrix}$$ and $$\lambda=3+2\sqrt2=5.83$$, the digit algorithm of Lemma 3 reproduces random $$x\in\mathbb Z^2$$ with $$\lvert x\rvert _\infty\le r$$ exactly (checked by reconstruction), using at most $$2,3,4,5,7,8,11$$ digits in each of the two expansions for $$r=10,10^2,\dots,10^6,10^8$$, against $$\log_\lambda r=1.3,2.6,3.9,5.2,6.5,7.8,10.5$$.

Whittaker has built finitely summable spectral triples for the stable-groupoid algebras of Smale spaces such as a hyperbolic toral automorphism [[Whittaker]](#ref-whit). These replace $$C(\mathbb T^n)\rtimes\Gamma$$ by a different algebra and are not covered by the theorems above.

## 10. States centred on one lattice point

Theorem 2 removes translations from the discussion only by assuming them. Take instead $$\Gamma$$ to fix the origin of $$\Lambda=\mathbb Z^4$$ and impose no translation symmetry. A positive $$\Gamma$$-invariant state is then the arithmetic version of a Milne universe about the origin.

<div class="proposition" markdown="1">

**Proposition 7** (a dark centre). Let $$\Gamma$$ be a finite-index subgroup of $$O^+(1,3;\mathbb Z)$$, $$\Lambda=\mathbb Z^4$$, and let $$W(x,y)=\omega(\phi(x)^*\phi(y))$$ for a positive state $$\omega$$ with $$W(gx,gy)=W(x,y)$$, $$g\in\Gamma$$. Assume

(WC) if $$y\in\Lambda$$ and $$\gamma_n\in\Gamma$$ satisfy $$\lvert B(y,\gamma_ny)\rvert \to\infty$$, then $$W(y,\gamma_ny)\to0$$.

Use two facts from C1 and its consequences. For a primitive spacelike $$v\in\Lambda$$, $$\mathrm{Stab}_\Gamma(v)$$ is a lattice, hence Zariski dense, in $$SO(v^\perp)\cong SO(1,2)$$, and it contains a hyperbolic element. For a primitive null $$v\in\Lambda$$, $$\mathrm{Stab}_\Gamma(v)$$ contains a power of a null rotation about $$v$$. Then:

(a) $$W(0,y)=0$$ for all $$y\in\Lambda\setminus\{0\}$$;

(b) $$W(v,y)=0$$ for spacelike $$v\in\Lambda$$ and all $$y\in\Lambda\setminus\mathbb Rv$$;

(c) $$W(v,y)=0$$ for null $$v\in\Lambda$$ and all $$y\in\Lambda$$ with $$B(v,y)\ne0$$.

</div>

<div class="proof" markdown="1">

*Proof.* Let $$\xi_x=\phi(x)\Omega$$ be the vectors of the Gelfand–Naimark–Segal construction, so $$W(x,y)=\langle\xi_x,\xi_y\rangle$$. Since $$W$$ is $$\Gamma$$-invariant, $$U(g)\xi_x=\xi_{gx}$$ defines an isometry on the closed span of the $$\xi_x$$.

*Mean ergodic step.* Let $$\ell\in\Gamma$$ fix $$v$$ (in (a) take $$v=0$$, fixed by every $$\ell$$), so $$U(\ell)\xi_v=\xi_v$$. By the mean ergodic theorem for isometries on a Hilbert space, $$\frac1N\sum_{k<N}U(\ell)^k\xi_y\to P\xi_y$$, with $$P$$ the projection onto the fixed vectors of $$U(\ell)$$. If $$\lvert B(y,\ell^ky)\rvert \to\infty$$, then (WC) gives $$\langle\xi_y,U(\ell)^k\xi_y\rangle=W(y,\ell^ky)\to0$$, so $$\lVert P\xi_y\rVert ^2=\lim\frac1N\sum_k\langle\xi_y,U(\ell)^k\xi_y\rangle=0$$. As $$\xi_v$$ is fixed, $$W(v,y)=\langle P\xi_v,\xi_y\rangle=\langle\xi_v,P\xi_y\rangle=0$$. It remains to find, in each case, $$\ell$$ fixing $$v$$ with $$\lvert B(y,\ell^ky)\rvert \to\infty$$.

*Existence in (a).* Let $$\ell_0\in\Gamma$$ be hyperbolic (the element $$h$$ of Lemma 3(ii) is one), with null eigenvectors $$w_\pm$$ of eigenvalues $$\lambda^{\pm1}$$; it acts as a rotation on the plane $$\{w_+,w_-\}^\perp$$. For $$\gamma\in\Gamma$$, $$\ell=\gamma\ell_0\gamma^{-1}$$ has eigenvectors $$\gamma w_\pm$$. Write $$y=\alpha\gamma w_++\beta\gamma w_-+y_0$$ with $$y_0\perp\gamma w_\pm$$; then $$\alpha\propto B(y,\gamma w_-)$$, $$\beta\propto B(y,\gamma w_+)$$, and $$B(y,\ell^ky)=\alpha\beta(\lambda^k+\lambda^{-k})B(\gamma w_+,\gamma w_-)+B(y_0,\ell^ky_0)$$ tends to $$\pm\infty$$ if $$\alpha\beta\ne0$$, because the last term is bounded. The function $$\gamma\mapsto B(\gamma^{-1}y,w_+)B(\gamma^{-1}y,w_-)$$ is a polynomial on $$SO(1,3)$$ that is not identically zero: if $$B(\cdot,w_+)$$ vanished on the orbit of $$y$$, which spans $$\mathbb R^4$$ by irreducibility, then $$w_+=0$$. Since $$\Gamma^+$$ is Zariski dense in $$SO(1,3)^\circ$$, some $$\gamma$$ gives $$\alpha\beta\ne0$$.

*Existence in (b).* Write $$y=cv+y'$$ with $$y'\in v^\perp$$; $$y'\ne0$$ because $$y\notin\mathbb Rv$$. The same argument inside $$v^\perp\cong\mathbb R^{1,2}$$ with the Zariski-dense group $$\mathrm{Stab}_\Gamma(v)$$ gives a hyperbolic $$\ell$$ fixing $$v$$ with $$B(y',\ell^ky')\to\infty$$, and $$B(y,\ell^ky)=c^2Q(v)+B(y',\ell^ky')$$.

*Existence in (c).* For $$b\perp v$$ define the null rotation $$N_b(y)=y+B(y,v)b-B(y,b)v-\tfrac12Q(b)B(y,v)v$$. It is an isometry fixing $$v$$, $$N_bN_{b'}=N_{b+b'}$$, and $$B(y,N_{kb}y)=Q(y)-\tfrac12k^2Q(b)B(y,v)^2$$, which tends to $$+\infty$$ when $$B(y,v)\ne0$$ since $$Q(b)<0$$ for spacelike $$b$$. Take $$\ell=N_b^m\in\Gamma$$. $$\square$$

</div>

Part (a) says that a clustering invariant state has no propagation at the centre, and (b) and (c) that points spacelike or null related to the centre propagate to nothing else, apart from their own rational multiples in the spacelike case. Correlations are possible between points inside the light cone of the centre. Timelike vectors have finite stabilisers in $$\Gamma$$ (they lie in a compact $$O(3)$$), so the argument constrains nothing for them. Condition (WC) is a hypothesis; it is not a consequence of positivity and invariance.

<div class="proposition" markdown="1">

**Proposition 8** (a state that propagates inside the cone). Let $$\Phi$$ be a free massive scalar field on Minkowski space with Fock vacuum $$\Omega$$, let $$f_0$$ be a real Schwartz function on $$\mathbb R^4$$ invariant under spatial rotations, let $$\Lambda_x$$ be a boost taking $$e_0$$ to $$u(x)=x/\sqrt{Q(x)}$$, and put $$f_x(y)=f_0(\Lambda_x^{-1}(y-x))$$ for $$x$$ in the open future cone. The functional $$\omega(\phi(x)\phi(y))=\langle\Omega,\Phi(f_x)\Phi(f_y)\Omega\rangle$$ on $$\Lambda\cap\{Q>0,\ x_0>0\}$$ is positive and invariant under $$O^+(1,3)$$, hence under $$\Gamma$$; its spectral support is the mass shell; and it satisfies (WC) on pairs of points in the open cone.

</div>

<div class="proof" markdown="1">

*Proof.* Positivity: $$W$$ is the Gram matrix of the vectors $$\Phi(f_x)\Omega$$. Invariance: for $$g\in O^+(1,3)$$, $$u(gx)=gu(x)$$, so $$\Lambda_{gx}=g\Lambda_xR$$ with $$R$$ a spatial rotation, and since $$f_0$$ is rotation invariant, $$f_{gx}=f_x\circ g^{-1}$$. The vacuum is invariant, so $$\langle\Phi(f_{gx})\Phi(f_{gy})\rangle=\langle\Phi(f_x)\Phi(f_y)\rangle$$. The spectral support is that of the vacuum two-point function, the mass shell. For (WC), $$W(y,\gamma_ny)=\langle\psi,\pi(h_n)\psi\rangle$$ for a fixed vector $$\psi$$ in the irreducible unitary representation $$\pi$$ of $$SO^+(1,3)$$ on $$L^2$$ of the mass shell and $$h_n=\Lambda_y^{-1}\gamma_n\Lambda_y$$ up to rotations; $$h_n\to\infty$$ in the group when $$B(y,\gamma_ny)\to\infty$$, and matrix coefficients of irreducible unitary representations vanish at infinity (Howe–Moore). $$\square$$

</div>

In the Milne frame $$u(x)$$ the symmetry at the point $$x$$ is the compact $$O(3)_{u(x)}$$ only, so a separate $$(u\cdot\partial\phi)^2$$ term is allowed for each field species; the frame $$u(x)$$, which the symmetry forces to be the Milne flow, plays the role of the aether of Einstein–aether theories. A centred $$\Gamma$$ therefore does not protect local Lorentz invariance; the dynamics inside the cone is Lorentz invariant to the extent that couplings are tuned, and the preferred frame is that of the cosmic expansion. Whether a state on all of $$\Lambda$$ can combine this interior with the dark exterior of Proposition 7 is not decided here.

## 11. What the theorems do not exclude

The theorems apply to scalar field algebras, to states or operators with exact translation invariance, and to Zariski-dense $$\Gamma$$. The following are outside them, each with its condition.

1. Indefinite-metric states. Theorem 2 uses positivity. For $$\Gamma$$-invariant kernels that are not positive definite, only Proposition 1 holds: there is no dispersion relation as a function on an open set of momenta.
2. States that break $$\Gamma$$ to a subgroup that is not Zariski dense. Theorem 3(b) shows that Lorentz-invariant propagation can occur there, and Example 1 has it after one condition on the couplings per species. Whether such a condition is stable under radiative corrections is a question about the couplings, which the symmetry does not decide.
3. Translations broken to a subgroup of infinite index, with $$\Gamma$$ unbroken. The point-centred states of section 10 propagate inside the cone of the centre; any Lorentz-violating frame is then the Milne frame of the cosmic expansion.
4. Point sets with $$\Gamma$$-symmetry and no translation subgroup, such as Haar-random lattices and orbit lattices; for a generic lattice $$g\mathbb Z^4$$ with $$g\in SL(4,\mathbb R)$$ the Lorentz stabiliser is finite and (F) fails. See [Lorentz-invariant point sets](/quantum-gravity/invariant-point-sets/) and [the page on single realisations](/quantum-gravity/per-realisation-kinematics/).
5. Non-automorphic Lorentz actions on non-abelian momentum groups of the $$\kappa$$-Poincaré kind, which have no point-set realisation.
6. For the statement about states (Theorem 2), non-scalar fields in non-unitary representations $$\sigma$$ of $$\Gamma$$. Theorem 1 covers them for operators. For states the trace of the matrix-valued spectral measure is only quasi-invariant and the argument of Theorem 2 does not apply as written.
7. Spectral-triple geometries of infinite spectral dimension ($$\theta$$-summable), geometries supported on torsion points, and geometries on a different algebra such as the Smale-space groupoid algebra.

## 12. Open points

Clustering (WC) is assumed in Proposition 7, not derived from the state axioms, and the statements (a) to (c) say nothing for states that do not cluster, including superselected and indefinite-metric ones. Whether a null-rotation argument also controls the pairs $$(v,y)$$ with $$B(v,y)=0$$ in (c) is not settled. Case (iii) of Theorem 3(b), an irrational unique invariant null line, is open and requires a non-reductive $$L_H$$. The distortion Lemma 3(ii) is proved for finite-index subgroups of $$O(1,3;\mathbb Z)$$; for other Zariski-dense arithmetic groups, such as the automorphism group of an anisotropic form, it is not proved here, and Theorem 5 for them is open. The packet response of Proposition 2 is exact for positive single-shell kernels; its value for signed retuned kernels is a numerical question taken up on the page about [single realisations](/quantum-gravity/per-realisation-kinematics/).

## References

1. <a id="ref-bq"></a>Benoist, Quint, "Mesures stationnaires et fermés invariants des espaces homogènes," Ann. of Math. 174 (2011) 1111–1162; English translation by B. Weiss, [arXiv:1610.05912](https://arxiv.org/abs/1610.05912).
2. <a id="ref-bflm"></a>Bourgain, Furman, Lindenstrauss, Mozes, "Stationary measures and equidistribution for orbits of nonabelian semigroups on the torus," J. Amer. Math. Soc. 24 (2011) 231–280.
3. <a id="ref-dln"></a>Di Scala, Leistner, Neukirchner, "Irreducibly acting subgroups of $$Gl(n,\mathbb R)$$," IRMA Lect. Math. Theor. Phys. 16 (2010) 629–651, [arXiv:math/0507047](https://arxiv.org/abs/math/0507047).
4. <a id="ref-lorente"></a>Lorente, "Discrete reflection groups and induced representations of Poincare group on the lattice," [arXiv:hep-lat/0401006](https://arxiv.org/abs/hep-lat/0401006).
5. <a id="ref-jv"></a>Judes, Visser, "Conservation laws in doubly special relativity," Phys. Rev. D 68 (2003) 045001, [arXiv:gr-qc/0205067](https://arxiv.org/abs/gr-qc/0205067).
6. <a id="ref-mignemi"></a>Mignemi, "Doubly special relativity and translation invariance," Phys. Lett. B 672 (2009) 186, [arXiv:0808.1628](https://arxiv.org/abs/0808.1628).
7. <a id="ref-cg"></a>Cohen, Glashow, "Very special relativity," Phys. Rev. Lett. 97 (2006) 021601, [arXiv:hep-ph/0601236](https://arxiv.org/abs/hep-ph/0601236).
8. <a id="ref-whit"></a>Whittaker, "Spectral triples for hyperbolic dynamical systems," J. Noncommut. Geom. 7 (2013) 563–582, [arXiv:1011.3292](https://arxiv.org/abs/1011.3292).
