---
title: Exact discrete Lorentz symmetry cannot carry propagation
date: 2026-10-29
category: substrate
order: 4
description: A scalar lattice theory with exact lattice translations and an exactly preserved Zariski-dense group of lattice-preserving Lorentz transformations has no dispersion relation, and propagation requires that the unbroken translations have infinite index, that the unbroken linear group is not Zariski dense, or that the number of fields is infinite.
status: Every statement is proved from the cited theorems of section 2 (Borel density, Benoist–Quint, Di Scala–Leistner–Neukirchner, and Howe–Moore in section 10); Proposition 7 assumes a clustering condition; the packet-response values of section 6 are computed numerically.
excludes: Exact lattice translations together with an exactly preserved Zariski-dense arithmetic subgroup of the Lorentz group (for example O(1,3;Z) or the automorphism group of an anisotropic integral form), as a mechanism that protects low-energy Lorentz invariance of a lattice dynamics with finitely many fields or of a positive state of a scalar field.
---

We ask whether a lattice theory that keeps exact lattice translations and an exactly preserved discrete group of Lorentz transformations can have a dispersion relation. The proposal behind the question is that such a group, for example the integral Lorentz group $$O(1,3;\mathbb Z)$$ acting on $$\mathbb Z^4$$ with $$Q(x)=x_0^2-x_1^2-x_2^2-x_3^2$$, might protect Lorentz invariance at low energy. Let $$\Lambda\subset\mathbb R^{1,3}$$ be the lattice and $$\Gamma$$ the group of Lorentz transformations that map $$\Lambda$$ onto itself and are kept as symmetries. If $$\Gamma$$ is Zariski dense in the Lorentz group, which holds for $$O(1,3;\mathbb Z)$$, for its finite-index subgroups, and for the automorphism group of an anisotropic integral form such as $$7x_0^2-x_1^2-x_2^2-x_3^2$$, the answer is no for a field with finitely many components. A translation-invariant operator on $$\ell^2(\Lambda)\otimes\mathbb C^N$$ that commutes with $$\Gamma$$ has the same characteristic polynomial at every momentum, so its bands are flat; for one component, or for a bounded operator, the symbol is constant (Theorem 1). The two-point function of a positive translation- and $$\Gamma$$-invariant state of a scalar field is a multiple of the Kronecker delta plus a sum of periodic functions with rational wave vectors (Theorem 2), which covers interacting states because only the two-point function enters. Theorem 1 follows from the fact that every non-zero lattice vector has an infinite $$\Gamma$$-orbit. Theorem 2 needs in addition the classification of $$\Gamma$$-invariant measures on the Brillouin torus.

If $$\Gamma$$ is a symmetry of the action and the state breaks it, the conclusion persists as long as the unbroken translations have finite index in $$\Lambda$$ and the linear parts of the unbroken symmetries stay Zariski dense (Theorem 3(a)). When they do not, the remaining symmetry no longer forces an effective quadratic action to be Lorentz invariant: apart from one case left open, there is an invariant symmetric tensor different from the metric, so a term $$\Theta^{\mu\nu}\partial_\mu\phi\,\partial_\nu\phi$$ is allowed (Theorem 3(b)). Low-energy Lorentz invariance is then possible but unprotected. Two further routes to propagation are an infinite family of fields permuted by $$\Gamma$$ (Example 1) and translations broken to infinite index, where correlations survive inside the light cone of a centre (section 10).

Integral Lorentz transformations of $$\mathbb Z^4$$ and the covariance of lattice wave equations under them were studied by Lorente [[Lorente]](#ref-lorente), who obtains covariance by letting the group act on non-linear functions of the lattice momentum; section 8 shows that this action is not induced by any map of the lattice points. The proofs rest on the Borel density theorem for arithmetic groups, the Benoist–Quint classification of invariant measures and closed invariant sets on the torus [[Benoist–Quint]](#ref-bq) (the proximal case is due to Bourgain, Furman, Lindenstrauss and Mozes [[BFLM]](#ref-bflm)), and the classification of irreducible connected groups preserving a Lorentzian form [[Di Scala–Leistner–Neukirchner]](#ref-dln), all stated in section 2. Their use as statements about lattice dynamics is what the page adds, together with the response of a localised packet to a $$\Gamma$$-invariant kernel (section 6) and an examination of two constructions that might evade the theorems: spectral-triple geometries on the Brillouin quotient (section 9) and states centred on one lattice point (section 10).

Lorentz-invariant point sets without a fixed lattice structure are treated on other pages: the Poisson case in [Lorentz-invariant quantum sprinklings](/quantum-gravity/quantum-sprinklings/), and random and orbit lattices in [Lorentz-invariant point sets](/quantum-gravity/invariant-point-sets/) and [Kernel-scale kinematics of a single realisation](/quantum-gravity/per-realisation-kinematics/). A single realisation of an orbit lattice is a lattice with its own translations and a Zariski-dense Lorentz stabiliser, so Theorem 1 and Proposition 1 below apply to it directly.

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

Lemma 1 shows that (H1) implies (F). Condition (F) alone suffices for Lemma 2, Theorem 1 and Theorem 4, and it also makes sense in lower dimension. For $$\Lambda=\mathbb Z^2$$ with the form $$t^2-x^2$$, an integer matrix preserving the form has first column $$(a,c)$$ with $$a^2-c^2=1$$, so $$a=\pm1$$, $$c=0$$, and $$\Gamma=\{\mathrm{diag}(\pm1,\pm1)\}$$ is finite. The orbit of $$e_0$$ is $$\{\pm e_0\}$$, (F) fails, and the usual lattice Klein–Gordon operator with a non-trivial dispersion relation is an invariant example. For the form $$t^2-2x^2$$, the matrix $$A=\begin{pmatrix}3&4\\2&3\end{pmatrix}$$ preserves it, has eigenvalues $$3\pm2\sqrt2$$, and no non-zero vector of $$\mathbb Z^2$$ has a finite $$\langle A\rangle$$-orbit; (F) holds, and Theorem 1 forbids a dispersion relation even in 1+1 dimensions.

## 2. The cited theorems

Four results are used without proof, in the form in which they are applied. The first three are used in sections 3 to 9, the fourth only in section 10.

<div class="theorem" markdown="1">

**Theorem C1** (Borel density; Borel–Harish-Chandra). If $$\mathcal G$$ is the group of real points of a semisimple algebraic group over $$\mathbb Q$$ without compact factors, then $$\mathcal G\cap GL(n,\mathbb Z)$$ is a lattice in $$\mathcal G$$ (Borel and Harish-Chandra [[BHC]](#ref-bhc)), and every lattice in $$\mathcal G$$ is Zariski dense in $$\mathcal G$$ (Borel [[Borel]](#ref-borel)). Consequently (H1) holds for every finite-index subgroup of $$O(1,3;\mathbb Z)$$ and, more generally, of the automorphism group of any integral quadratic form of signature $$(1,3)$$, such as $$7x_0^2-x_1^2-x_2^2-x_3^2$$. The same applies to signature $$(1,2)$$.

</div>

<div class="theorem" markdown="1">

**Theorem C2** (Benoist–Quint [[Benoist–Quint]](#ref-bq), Theorem 1.3 and Corollary 1.4). Let $$\Gamma_0$$ be a subsemigroup of $$SL(d,\mathbb Z)$$ acting strongly irreducibly on $$\mathbb R^d$$, meaning that every finite-index subgroup of the group it generates acts irreducibly. On the torus $$\mathbb R^d/\mathbb Z^d$$: (a) the only $$\Gamma_0$$-invariant Borel probability measure without atoms is the Haar measure; (b) the only closed infinite $$\Gamma_0$$-invariant subset is the whole torus. As recorded in that paper, part (b) for such groups is due to Muchnik and to Guivarc'h and Starkov, and part (a) under a proximality assumption to Bourgain, Furman, Lindenstrauss and Mozes [[BFLM]](#ref-bflm).

</div>

<div class="theorem" markdown="1">

**Theorem C3** (Di Scala, Leistner, Neukirchner [[Di Scala–Leistner–Neukirchner]](#ref-dln)). A connected subgroup of $$GL(n,\mathbb R)$$ that acts irreducibly on $$\mathbb R^n$$ and leaves invariant a bilinear form of Lorentzian signature is conjugate to $$SO(1,n-1)^\circ$$.

</div>

<div class="theorem" markdown="1">

**Theorem C4** (Howe–Moore [[Howe–Moore]](#ref-hm)). If $$\pi$$ is an irreducible unitary representation of a connected non-compact simple Lie group with finite centre, such as $$SO^+(1,3)\cong PSL(2,\mathbb C)$$, then every matrix coefficient $$\langle\psi,\pi(h)\psi'\rangle$$ tends to 0 as $$h\to\infty$$ in the group.

</div>

Under (H1) the group $$\Gamma^+=\Gamma\cap SL(4,\mathbb Z)$$, of index at most 2, still has Zariski closure containing $$SO(1,3)^\circ$$ and acts strongly irreducibly: a finite-index subgroup has Zariski closure of finite index in that of $$\Gamma$$, hence containing $$SO(1,3)^\circ$$, which acts irreducibly because it contains boosts in every $$(0,j)$$-plane and all rotations. So C2 applies to $$\Gamma^+$$ acting on $$\mathbb T$$, and a fortiori to measures and sets invariant under $$\Gamma$$.

C2 enters Corollary 1, Proposition 1 and Theorem 2 (and Theorem 3(a) through Theorem 2), C3 enters Theorem 3(b), and C4 enters Proposition 8. Lemmas 1 and 2, Theorems 1, 4 and 5 and Propositions 2 to 6 need only (F) or (H1). Proposition 7 uses two consequences of C1, stated in section 10.

## 3. No finite orbits, no finite-dimensional invariant subspaces

A $$\Gamma$$-orbit on the Brillouin torus is either finite, in which case its points are torsion points, or infinite, and no finite-dimensional space of functions on the torus is invariant apart from the constants.

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

**Theorem 1** (constant symbols). Assume (F). Let $$D$$ be a bounded operator, or a self-adjoint operator, on $$\ell^2(\Lambda)\otimes V$$ with $$V=\mathbb C^N$$, $$N<\infty$$, that commutes with every $$T_a$$, $$a\in\Lambda$$, and every $$U_g$$, $$g\in\Gamma$$. Then $$D$$ is the multiplication by a symbol $$d$$ with $$d(g\cdot k)=\sigma(g)d(k)\sigma(g)^{-1}$$ almost everywhere, and:

(a) if $$N=1$$ ($$\sigma$$ is then a character), $$d$$ is almost everywhere a constant $$d_0$$ and $$D=d_0\cdot1$$;

(b) for general $$N$$, the characteristic polynomial of $$d(k)$$ is almost everywhere independent of $$k$$, so every band of the dispersion relation is flat;

(c) if $$D$$ is bounded, then $$d$$ is almost everywhere a constant matrix $$d_0$$, and $$d_0=\sigma(g)d_0\sigma(g)^{-1}$$ for all $$g\in\Gamma$$.

</div>

Translation invariance turns the operator into a function on the torus, and invariance under $$\Gamma$$ makes that function constant on the orbits of an ergodic action. The input is (F); the remaining steps are routine.

<div class="proof" markdown="1">

*Proof.* The operators $$T_a$$ generate, on $$L^2(\mathbb T)\otimes V$$, the von Neumann algebra $$L^\infty(\mathbb T)\otimes1$$ of multiplications by characters and their limits. Its commutant is $$L^\infty(\mathbb T)\otimes M_N(\mathbb C)$$, because $$L^\infty(\mathbb T)$$ is maximal abelian on $$L^2(\mathbb T)$$. A bounded operator commuting with all $$T_a$$ is therefore a bounded matrix symbol, and a self-adjoint one has its spectral projections in the commutant, so it is a multiplier by a measurable Hermitian symbol. Commutation with $$U_g$$ and (2) give $$d(k)\sigma(g)F(g^{-1}\cdot k)=\sigma(g)d(g^{-1}\cdot k)F(g^{-1}\cdot k)$$ for all $$F$$, which is the stated covariance.

(a) $$d$$ is a measurable function with $$d(g\cdot k)=d(k)$$, so it is constant by Lemma 2(ii).

(b) For each $$j$$ the function $$k\mapsto\operatorname{tr}d(k)^j$$ is measurable and invariant, hence constant by Lemma 2(ii). The coefficients of the characteristic polynomial are polynomials in these traces (Newton's identities).

(c) The entries of $$d$$ are bounded, hence in $$L^2(\mathbb T)$$. By the covariance, the entries of $$d\circ g$$ are linear combinations of the entries of $$d$$, so the span of the entries is a finite-dimensional invariant subspace of $$L^2(\mathbb T)$$. Its projection onto $$L^2_0$$ is again finite-dimensional and invariant, hence zero by Lemma 2(i). $$\square$$

</div>

For bounded $$D$$ and $$N=1$$ there is a direct argument without ergodic theory: the kernel $$c=D\delta_0$$ lies in $$\ell^2(\Lambda)$$ and is constant on $$\Gamma$$-orbits, and the orbits of non-zero vectors are infinite, so $$c=c(0)\delta_0$$. The ergodic form covers unbounded operators and the matrix case.

<div class="remark" markdown="1">

**Remark 1** (a sublattice of translations). Translation invariance under all of $$\Lambda$$ is used. Suppose only a $$\Gamma$$-stable sublattice $$\Lambda'$$ of finite index is a translation symmetry, and let $$D$$ be a bounded scalar operator ($$N=1$$) commuting with the translations by $$\Lambda'$$ and with $$\Gamma$$, with kernel $$D(x,y)$$. The column $$f_y=D\delta_y$$ is square summable. Let $$\Gamma_y\subset\Gamma$$ be the finite-index subgroup of elements $$g$$ with $$gy-y\in\Lambda'$$. For $$g\in\Gamma_y$$ the affine map $$h(x)=gx-(gy-y)$$ fixes $$y$$ and is a product of $$g$$ and a translation by $$\Lambda'$$, so $$f_y(y+z)$$ is invariant under $$z\mapsto gz$$ for $$g\in\Gamma_y$$. Orbits of $$\Gamma_y$$ on $$\Lambda\setminus\{0\}$$ are infinite (they are finite unions of those of $$\Gamma$$ are infinite, and a finite-index subgroup has orbits that are infinite whenever those of $$\Gamma$$ are), so $$f_y=D(y,y)\delta_y$$. Hence $$D$$ is multiplication by a function of the class of $$x$$ modulo $$\Lambda'$$, constant on $$\Gamma$$-orbits of classes: a constant matrix, diagonal in the coset basis, and not in general a multiple of the identity. An example is the staggered mass for $$\Lambda=\mathbb Z^4$$: since $$Q(x)\equiv x_0+x_1+x_2+x_3\pmod2$$, the function $$m(x)=(-1)^{Q(x)}$$ is $$\Gamma$$-invariant, commutes with translations by the $$\Gamma$$-stable even sublattice $$\{Q\ \text{even}\}$$, and does not commute with translation by $$e_0$$. Operators that are $$\Gamma$$-invariant without translation invariance exist (any multiplication by a function of $$Q(x)$$).

</div>

<div class="corollary" markdown="1">

**Corollary 1** (low momenta are dense in the zone). Assume (H1). For every non-torsion $$k\in\mathbb T$$ the orbit $$\Gamma\cdot k$$ is dense in $$\mathbb T$$.

</div>

<div class="proof" markdown="1">

*Proof.* By Lemma 1 the orbit is infinite, and so is the orbit under $$\Gamma^+$$, which satisfies (H1) as well. Its closure is a closed infinite $$\Gamma^+$$-invariant set, hence the whole torus by C2(b). $$\square$$

</div>

A lattice momentum $$k$$ is non-torsion as soon as $$k\cdot v/2\pi$$ is irrational for some $$v\in\Lambda$$. For such a $$k$$ of small norm, some $$g\in\Gamma$$ maps it arbitrarily close to any prescribed point of the zone, the zone boundary included. A symmetry that identifies low momenta with momenta at the cutoff leaves no separate low-energy description. By Borel density a continuum effective action with $$\Gamma$$-invariant polynomial coefficients has Lorentz-invariant coefficients, since an invariant tensor is fixed by the Zariski closure; Theorem 1 and Corollary 1 show that the lattice theory itself has no low-energy sector on which that protection could act.

<div class="proposition" markdown="1">

**Proposition 1** (shell sums have no dispersion on any open set). Assume (H1). Let $$c:\Lambda\to\mathbb C$$ be a bounded $$\Gamma$$-invariant function that vanishes outside finitely many shells $$\{Q=q\}$$ and is non-zero at some $$v\ne0$$. Then its Fourier series $$\hat c$$, a distribution on $$\mathbb T$$, coincides with an integrable function on no non-empty open subset of $$\mathbb T$$.

</div>

The point is that invariance carries a local integrable representation of $$\hat c$$ over the whole torus, where ergodicity makes it constant; what is left is supported on finitely many torsion points, where the Fourier coefficients are polynomials times periodic functions, and vanishing on all but finitely many shells then forces them to vanish.

<div class="proof" markdown="1">

*Proof.* Suppose $$\hat c=h$$ on an open set $$U\ne\emptyset$$ with $$h\in L^1_{\rm loc}(U)$$. Since $$\hat c$$ is $$\Gamma$$-invariant, it equals $$h\circ g^{-1}$$ on $$gU$$, and these local functions agree on overlaps because $$\hat c$$ is one distribution; so $$\hat c=\tilde h$$ on the open invariant set $$\Gamma U$$, with $$\tilde h\in L^1_{\rm loc}$$ invariant. The indicator of $$\Gamma U$$ is invariant, hence almost everywhere constant by Lemma 2(ii), and equal to 1 since $$U$$ has positive measure; so $$\Gamma U$$ is conull and $$\tilde h$$ is almost everywhere a constant $$\kappa$$. The closed invariant set $$Z=\mathbb T\setminus\Gamma U$$ is not the whole torus, so by C2(b) it is finite, and by Lemma 1 it consists of torsion points. The distribution $$\hat c-\kappa$$ is supported on $$Z$$, hence is a finite sum of derivatives of delta functions at points of $$Z$$. Inverting the Fourier series, $$c(v)=\kappa\delta_{v,0}+\sum_{z\in Z}p_z(v)e^{iz\cdot v}$$ with polynomials $$p_z$$.

Let $$v_0\ne0$$ have $$c(v_0)\ne0$$. Let $$M\ge2$$ be an integer such that $$Mz\in2\pi\Lambda^\vee$$ for all $$z\in Z$$ and $$v_0\notin M\Lambda$$. On the coset $$v_0+M\Lambda$$, which does not contain 0, each factor $$e^{iz\cdot v}$$ is constant, so $$c$$ coincides there with a single polynomial $$p$$. The points of the coset outside the finitely many shells form a Zariski-dense subset of $$\mathbb R^4$$ (a polynomial vanishing on them, multiplied by $$\prod_q(Q-q)$$, vanishes on the whole coset, which is Zariski dense), and $$p$$ vanishes on them. So $$p\equiv0$$ and $$c(v_0)=0$$, a contradiction. $$\square$$

</div>

The proposition applies to the kernels of causal-set-inspired lattice operators that sum interior-count weights over finitely many shells; on a lattice with a Zariski-dense $$\Gamma$$ such a kernel is a distribution that is not a function.

## 5. Invariant states

An operator is a special case of what a quantum theory provides. The two-point function of a state is more general, because it covers interacting and non-perturbative states.

<div class="theorem" markdown="1">

**Theorem 2** (invariant two-point functions are flat plus periodic). Assume (H1). Let $$\omega$$ be a positive linear functional on a $$\ast$$-algebra containing elements $$\phi(x)$$, $$x\in\Lambda$$, and set $$W(x,y)=\omega(\phi(x)^\ast\phi(y))$$. Suppose $$W(x+a,y+a)=W(x,y)$$ for $$a\in\Lambda$$ and $$W(gx,gy)=W(x,y)$$ for $$g\in\Gamma$$. Then

$$
W(x,y)=a_0\,\delta_{x,y}+\sum_{O}b_O\,\frac1{\lvert O\rvert }\sum_{k\in O}e^{ik\cdot(y-x)}, \tag{3}
$$

where $$a_0\ge0$$, the sum runs over the countably many finite $$\Gamma$$-orbits $$O$$ of torsion points of $$\mathbb T$$, $$b_O\ge0$$, and $$a_0+\sum_Ob_O=W(0,0)<\infty$$.

</div>

The spectral measure of the two-point function is a multiple of Haar measure plus atoms at torsion points. The Haar part is the same at every momentum, and the atoms at rational wave vectors give correlations that do not decay with distance; no part has a non-trivial dependence on $$k$$.

<div class="proof" markdown="1">

*Proof.* Positivity gives $$\sum\bar\lambda_x\lambda_yW(x,y)=\omega\big((\sum\lambda_x\phi(x))^\ast(\sum\lambda_y\phi(y))\big)\ge0$$. Put $$w(v)=W(0,v)$$; translation invariance gives $$W(x,y)=w(y-x)$$, and $$w$$ is a positive-definite function on $$\Lambda$$. By Bochner's theorem for the discrete group $$\Lambda$$, $$w(v)=\int_{\mathbb T}e^{ik\cdot v}d\mu(k)$$ for a unique finite positive measure $$\mu$$ with $$\mu(\mathbb T)=w(0)$$. From $$w(gv)=W(g0,gv)=w(v)$$ and (1), the measure $$g_\ast\mu$$ has the same Fourier coefficients as $$\mu$$, so $$\mu$$ is $$\Gamma$$-invariant.

Decompose $$\mu=\mu_{\rm c}+\mu_{\rm p}$$ into non-atomic and purely atomic parts; both are $$\Gamma$$-invariant because $$\Gamma$$ acts by homeomorphisms. By C2(a), $$\mu_{\rm c}=a_0\cdot\text{Haar}$$ (Haar normalised to total mass 1). The atomic part is $$\sum_km_k\delta_k$$ with $$m_k$$ constant on orbits and summable, so an orbit carrying positive mass is finite, and by Lemma 1 its points are torsion. With $$b_O=\lvert O\rvert m_k$$ for $$k\in O$$ this is (3), and evaluation at $$x=y$$ gives the sum rule. $$\square$$

</div>

If only a $$\Gamma$$-stable finite-index sublattice $$\Lambda'$$ is a translation symmetry, the theorem applies to the sub-algebra generated by $$\phi(x)$$, $$x\in\Lambda'$$.

Positivity is used in an essential way. The shell function $$c=\mathbf 1_{\{Q=1\}}$$ is $$\Gamma$$-invariant, defines a translation-invariant kernel, and has $$c(0)=0$$; in (3) this would force $$a_0=b_O=0$$ and so $$w\equiv0$$. It is therefore the two-point function of no positive state, and a kernel of this kind can arise only from a state on an indefinite-metric space. For such kernels Proposition 1 still forbids a dispersion relation on an open set of momenta.

For a field with several components in a non-unitary representation $$\sigma$$ the trace of the matrix-valued spectral measure is only quasi-invariant, and the argument above does not apply as written; Theorem 1 covers such fields for operators.

<div class="remark" markdown="1">

**Remark 2** (infinite invariant measures). The finiteness of $$\mu$$ is essential. Infinite $$\Gamma$$-invariant measures on $$\mathbb T$$ exist, for example counting measure on a dense orbit, but they are not Radon on the compact torus and cannot be spectral measures of a state, for which $$\mu(\mathbb T)=W(0,0)$$.

</div>

## 6. Packet response of shell kernels

Section 4 shows that a $$\Gamma$$-invariant lattice operator has no symbol to expand in momentum. One can ask instead what a localised probe sees. A finite-norm packet $$\phi(x)=e^{ip\cdot x}f(x)$$, with $$f$$ real, decaying and normalised to $$\lVert f\rVert=1$$, is what a dispersion relation is read off from. A kernel supported on a shell $$\{Q=q\}$$ sums over infinitely many lattice vectors, so the sum has to be cut off, and here the packet envelope cuts it off. Let $$D$$ have kernel $$c$$ with $$c(-v)=c(v)$$, so $$\langle\phi,D\phi\rangle=\sum_{x,v}\bar\phi(x)c(v)\phi(x+v)$$. Summing over $$x$$ gives

$$
\langle\phi,D\phi\rangle=\sum_{v}c(v)\cos(p\cdot v)E(v),\qquad E(v)=\sum_xf(x)f(x+v), \tag{4}
$$

where $$E$$ is the envelope autocorrelation, $$E(0)=1$$. For the Gaussian envelope $$f(x)\propto e^{-\lvert x\rvert_{\rm E}^2/2L^2}$$ of width $$L$$ in the Euclidean norm, $$E(v)\propto e^{-\lvert v\rvert _{\rm E}^2/4L^2}$$ up to corrections exponentially small in $$L^2$$, with the constant fixed by $$E(0)=1$$; this form is used below. Expanding the cosine, the packet sees $$S_0-\tfrac12p_\mu p_\nu \Pi^{\mu\nu}+O(p^4)$$ with

$$
\Pi^{\mu\nu}=\sum_vc(v)E(v)\,v^\mu v^\nu,\qquad S_0=\sum_vc(v)E(v). \tag{5}
$$

A Lorentz-invariant low-energy dispersion requires $$\Pi^{\mu\nu}\propto\eta^{\mu\nu}$$, that is $$\rho:=\Pi^{00}/(-\Pi^{11})=+1$$ with the three spatial entries equal.

<div class="proposition" markdown="1">

**Proposition 2** (positive kernels give an elliptic packet response). Let $$c(-v)=c(v)$$ and $$E>0$$. (i) If $$c\ge0$$ and $$c(v)\ne0$$ for some $$v\ne0$$, then $$\Pi$$ is positive semidefinite and non-zero, hence not proportional to $$\eta^{\mu\nu}$$, and $$\rho\le0$$ where defined. (ii) For a shell kernel $$c=\mathbf 1_{\{Q=q\}}$$ on a lattice for which $$c$$ and $$E$$ are symmetric under permutations of the three spatial axes,

$$
\rho=-\frac{3\langle t^2\rangle}{\langle t^2\rangle-q},\qquad \langle t^2\rangle=\frac{\sum_{v}c(v)E(v)v_0^2}{S_0}, \tag{6}
$$

exactly. (iii) Let $$\Lambda=\mathbb Z^4$$ with the standard form, $$E=E_L$$ the Gaussian above, and let the shell $$\{Q=q\}$$ contain a non-zero vector. Then $$\rho\to-3$$ as $$L\to\infty$$. For $$q=0$$, $$\rho=-3$$ for every $$L$$.

</div>

<div class="proof" markdown="1">

*Proof.* (i) $$x^T\Pi x=\sum cE\,(x\cdot v)^2\ge0$$ for every real $$x$$, and the trace $$\sum cE\,\lvert v\rvert _{\rm E}^2$$ is positive. A non-zero positive semidefinite form is not proportional to the indefinite $$\eta^{\mu\nu}$$. Moreover $$\Pi^{11}\ge0$$ and $$\Pi^{00}>0$$, so $$\rho\le0$$ where defined.

(ii) On the shell $$v_0^2=q+\lvert \vec v\rvert ^2$$, and by the permutation symmetry $$\Pi^{11}=\tfrac13\sum cE\,\lvert \vec v\rvert ^2=\tfrac13(\Pi^{00}-qS_0)$$. Hence $$\rho=-3\Pi^{00}/(\Pi^{00}-qS_0)$$, which is (6).

(iii) The symmetry hypotheses of (ii) hold for $$\mathbb Z^4$$ and the Euclidean Gaussian. Let $$T_0>0$$ and let $$n(T_0)$$ be the number of shell points with $$\lvert v_0\rvert \le T_0$$; it is finite because $$\lvert \vec v\rvert ^2=v_0^2-q$$. The shell is infinite by (F), which holds for $$\Gamma=O(1,3;\mathbb Z)$$, and $$E_L\to1$$ pointwise as $$L\to\infty$$, so $$S_0\to\infty$$ by Fatou's lemma. The weight carried by $$\lvert v_0\rvert \le T_0$$ is at most $$n(T_0)$$, so $$\langle t^2\rangle\ge T_0^2(1-n(T_0)/S_0)\to T_0^2$$. As $$T_0$$ is arbitrary, $$\langle t^2\rangle\to\infty$$ and (6) gives $$\rho\to-3$$. For $$q=0$$, (6) equals $$-3$$ identically. $$\square$$

</div>

The tensor tends to $$\Pi^{00}\,\mathrm{diag}(1,\tfrac13,\tfrac13,\tfrac13)$$, that of null dust: boosted copies of the shell lie almost along the light cone and dominate the sum. In the rest frame of the envelope, with time direction $$u$$, the packet sees the elliptic quadratic form $$p_0^2+\lvert\vec p\rvert^2/3$$ instead of $$p_0^2-\lvert \vec p\rvert ^2$$. A packet boosted by a different rapidity sees a different $$\Pi$$, so the regulator picks a frame, and the resulting Lorentz-violating coefficient is of order one rather than suppressed (see [Where a preferred frame leaks into low-energy physics](/quantum-gravity/lorentz-violation-leakage/) for what an order-one coefficient does).

We computed $$\rho$$ for $$\Lambda=\mathbb Z^4$$ and the shell $$Q=1$$ by summing over all shell points inside a ball of radius about $$11L$$ (beyond which the weight is below $$10^{-12}$$), and found

| $$L$$ | 4 | 8 | 16 |
|---|---|---|---|
| $$\rho$$ | $$-3.0931$$ | $$-3.0232$$ | $$-3.0059$$ |

in agreement with (6), with the three spatial entries of $$\Pi$$ equal to rounding error. For the shell $$Q=0$$ the same sums give $$\rho=-3$$ to rounding. The number of points of the shell $$Q=1$$ with $$t>0$$ and Euclidean norm at most $$R=25,50,100,200$$ is $$473,\,2257,\,8617,\,33861$$, which grows like $$R^2$$. For kernels with weights of both signs, such as retuned d'Alembertians, part (i) does not apply, and whether $$\rho$$ is near $$-3$$ depends on cancellations between shells that Proposition 2 does not determine.

## 7. Spontaneous breaking of the discrete Lorentz group

Theorem 2 assumes an invariant state. Suppose instead that the action is invariant under $$\Gamma\ltimes\Lambda$$, the group of affine maps $$x\mapsto gx+a$$ with $$g\in\Gamma$$, $$a\in\Lambda$$, and that the state is invariant only under a subgroup $$S$$. Let $$\Lambda'=S\cap\Lambda$$ be the unbroken translations and

$$
H=\{g\in\Gamma:\ (g,a)\in S\ \text{for some }a\in\Lambda\}
$$

the group of linear parts of the unbroken symmetries. An unbroken symmetry can be a linear map composed with a translation that is not itself a symmetry, so $$H$$ can be larger than $$S\cap\Gamma$$; the next paragraphs show which of the two matters where.

The group $$H$$ preserves $$\Lambda'$$, because $$(g,a)\,t_b\,(g,a)^{-1}=t_{gb}$$. For $$h\in H$$ choose $$a_h$$ with $$(h,a_h)\in S$$; it is determined modulo $$\Lambda'$$, and $$a_{hk}\equiv ha_k+a_h$$. So $$H$$ acts on the quotient $$\Lambda/\Lambda'$$ by $$h\cdot[x]=[hx+a_h]$$, and $$S\cap\Gamma=\{h:(h,0)\in S\}$$ is the stabiliser of the class $$[0]$$. If $$\Lambda'$$ has finite index in $$\Lambda$$, $$S\cap\Gamma$$ therefore has finite index in $$H$$, at most $$[\Lambda:\Lambda']$$. If $$\Lambda'$$ has infinite index the two can differ completely: for a state centred at a lattice point $$p$$, $$S=t_p\Gamma t_p^{-1}$$, so $$H=\Gamma$$ while $$S\cap\Gamma=\mathrm{Stab}_\Gamma(p)$$ is finite when $$p$$ is timelike.

Write $$\bar H$$ for the Zariski closure of $$H$$ in $$GL(4,\mathbb R)$$ and $$L_H$$ for the identity component, in the Lie-group topology, of its real points. Since $$H\subset O(1,3)$$ and $$\bar H$$ is algebraic, $$L_H\subset SO(1,3)^\circ$$. A real algebraic group has finitely many connected components, so $$L_H$$ has finite index in $$\bar H(\mathbb R)$$ and $$H\cap L_H$$ has finite index in $$H$$. Conjugation by $$h\in H$$ preserves $$\bar H(\mathbb R)$$, so $$H$$ normalises $$L_H$$. Finally $$L_H=SO(1,3)^\circ$$ if and only if $$\bar H\supset SO(1,3)^\circ$$, which is (H1) for $$H$$.

An effective quadratic action for the fluctuations about the state is invariant under $$S$$. When $$\Lambda'$$ has finite index it is invariant under translations by $$\Lambda'$$, so its gradient term has constant coefficients $$\Theta^{\mu\nu}$$, and an affine map $$(g,a)$$ acts on a constant-coefficient term through $$g$$ alone. The term $$\Theta^{\mu\nu}\partial_\mu\phi\,\partial_\nu\phi$$ for a scalar $$\phi$$ is therefore invariant under $$S$$ exactly when $$\Theta$$ is invariant under $$H$$. Since $$H$$ preserves $$\eta$$, an $$H$$-invariant symmetric bilinear form on $$\mathbb R^4$$ corresponds to an $$H$$-invariant symmetric tensor by raising indices with $$\eta$$, and the two are not distinguished below.

<div class="theorem" markdown="1">

**Theorem 3** (dichotomy for the unbroken group). (a) Assume that $$\Lambda'$$ has finite index in $$\Lambda$$ and that $$H$$ satisfies (H1). Then the two-point function of a positive state on an algebra containing the fields $$\phi(x)$$, $$x\in\Lambda'$$, has the form (3) with $$\Lambda'$$ and $$S\cap\Gamma$$ in place of $$\Lambda$$ and $$\Gamma$$. This applies when $$S$$ has finite index in $$\Gamma\ltimes\Lambda$$, since (H1) for $$H$$ then follows from (H1) for $$\Gamma$$.

(b) Let $$H$$ be any subgroup of $$\Gamma$$ with $$L_H\ne SO(1,3)^\circ$$. Then exactly one of the following holds.

1. $$L_H$$ leaves invariant a non-degenerate subspace $$Y$$ with $$0\ne Y\ne\mathbb R^4$$.
2. $$L_H$$ leaves invariant no such subspace; it then leaves invariant exactly one null line $$\ell$$, and $$\ell$$ contains a non-zero lattice vector.
3. As in 2, but $$\ell$$ contains no non-zero lattice vector.

In cases 1 and 2 there is an $$H$$-invariant symmetric tensor $$\Theta^{\mu\nu}$$ that is not a multiple of $$\eta^{\mu\nu}$$, so the term $$\Theta^{\mu\nu}\partial_\mu\phi\,\partial_\nu\phi$$ is invariant under all unbroken symmetries when $$\Lambda'$$ has finite index. Cases 2 and 3 occur only if $$L_H$$ does not act completely reducibly.

</div>

Part (a) says that breaking to a finite-index subgroup changes nothing. Part (b) says that when the unbroken group is thin, the remaining symmetry no longer forces the quadratic action to be Lorentz invariant; whether it is, at low energy, then depends on couplings that the symmetry leaves free.

<div class="proof" markdown="1">

*Proof.* (a) If $$\Lambda'$$ has finite index, $$S\cap\Gamma$$ has finite index in $$H$$ by the paragraph above, and so inherits (H1): the Zariski closure of a finite-index subgroup has finite index in that of the group, hence contains $$SO(1,3)^\circ$$. Both $$\Lambda'$$ and $$S\cap\Gamma$$ preserve $$\Lambda'$$, and the restricted two-point function is invariant under translations by $$\Lambda'$$ and under $$S\cap\Gamma$$. Theorem 2 applies to $$\Lambda'$$ and $$S\cap\Gamma$$. If $$S$$ has finite index in $$\Gamma\ltimes\Lambda$$, then $$[\Lambda:\Lambda']\le[\Gamma\ltimes\Lambda:S]$$ and $$[\Gamma:H]\le[\Gamma\ltimes\Lambda:S]$$, since $$H$$ is the image of $$S$$ under the projection to $$\Gamma$$; so $$H$$ has finite index in $$\Gamma$$ and inherits (H1).

(b) First, $$L_H$$ is reducible. It is a connected subgroup of $$SO(1,3)^\circ$$ and preserves the Lorentzian form. If it acted irreducibly, C3 would make it conjugate to $$SO(1,3)^\circ$$, and a connected Lie subgroup of $$SO(1,3)^\circ$$ of the same dimension is open, hence equal to $$SO(1,3)^\circ$$; this is excluded. So $$L_H$$ preserves a subspace $$Y_0$$ with $$0\ne Y_0\ne\mathbb R^4$$.

Suppose $$L_H$$ preserves two distinct null lines. They span a plane on which the Gram matrix has the form $$\begin{pmatrix}0&\beta\\\beta&0\end{pmatrix}$$ with $$\beta\ne0$$, since a totally null plane does not exist in $$\mathbb R^{1,3}$$; the plane is Lorentzian and non-degenerate. So two invariant null lines imply case 1. If case 1 fails, $$Y_0$$ is degenerate, and its radical $$Y_0\cap Y_0^\perp$$ is a non-zero totally null subspace, hence a null line, invariant under $$L_H$$. By the previous sentence it is the only invariant null line $$\ell$$.

Case 1. Let $$Y$$ be as in case 1 and $$b_Y(p,q)=B(P_Yp,P_Yq)$$, with $$P_Y$$ the orthogonal projection onto $$Y$$, so that $$b_Y+b_{Y^\perp}=B$$. A non-degenerate subspace of $$\mathbb R^{1,3}$$ is negative definite, or its orthogonal complement is. For $$h\in H$$ the subspace $$hY$$ is again $$L_H$$-invariant, because $$H$$ normalises $$L_H$$, and $$b_{hY}=b_Y(h^{-1}\cdot,h^{-1}\cdot)$$. The orbit $$HY$$ is finite, since the stabiliser of $$Y$$ contains $$H\cap L_H$$, of finite index in $$H$$. Put $$\Theta=\sum_{Y'\in HY}b_{Y'}$$, which is $$H$$-invariant. If $$Y$$ is negative definite, every $$b_{Y'}$$ is negative semidefinite and non-zero, so $$\Theta$$ is also, and it cannot be a multiple of the indefinite $$B$$. Otherwise $$Y^\perp$$ is negative definite, $$b_{Y'}=B-b_{Y'^\perp}$$ with $$-b_{Y'^\perp}$$ positive semidefinite and non-zero, and $$\Theta=\lvert HY\rvert \,B+P$$ with $$P$$ positive semidefinite and non-zero. If $$\Theta=\alpha B$$ then $$P=(\alpha-\lvert HY\rvert )B$$, which is impossible for a non-zero positive semidefinite $$P$$ and an indefinite $$B$$.

Cases 2 and 3. The group $$H$$ maps $$\ell$$ to an $$L_H$$-invariant null line, hence to $$\ell$$. If $$\ell=\mathbb Rn_0$$ with $$n_0\in\Lambda$$ primitive, then $$H\subset GL(\Lambda)$$ preserves the rank-one lattice $$\ell\cap\Lambda$$ and acts on it by $$\pm1$$, so $$\Theta=n_0\otimes n_0$$ is $$H$$-invariant. It is not a multiple of $$\eta^{\mu\nu}$$: $$\Theta^{00}>0$$ and $$\Theta^{11}\ge0$$, while a multiple $$\alpha\eta^{\mu\nu}$$ has $$\Theta^{00}$$ and $$\Theta^{11}$$ of opposite sign unless $$\alpha=0$$. This gives the tensor in case 2. If $$\ell$$ contains no lattice vector, we are in case 3.

Finally, suppose $$L_H$$ acts completely reducibly and preserves a null line $$\ell$$. An invariant complement $$C$$ of $$\ell$$ gives an invariant two-dimensional subspace $$C\cap\ell^\perp$$ complementary to $$\ell$$ in $$\ell^\perp$$; it is negative definite, since $$\ell^\perp/\ell$$ is. Its orthogonal complement is an invariant Lorentzian plane containing $$\ell$$, which contains a second null line, also invariant because $$L_H$$ is connected. So case 1 holds. Hence cases 2 and 3 require that $$L_H$$ does not act completely reducibly. $$\square$$

</div>

The very special relativity groups of Cohen and Glashow [[Cohen–Glashow]](#ref-cg), such as $$SIM(2)$$, fix a null direction up to scale. A boost along that direction rescales it, so $$n_0\otimes n_0$$ is not invariant under such a group. The group $$H$$ here is discrete, and it cannot rescale a primitive lattice vector, which is why case 2 carries the invariant tensor $$n_0\otimes n_0$$. For an irrational null line, case 3, this argument is unavailable, and whether an $$H$$-invariant tensor other than $$\eta$$ exists is not decided here.

<div class="example" markdown="1">

**Example 1** (infinitely many fields). Let $$\Gamma=O^+(1,3;\mathbb Z)$$ and $$\Lambda=\mathbb Z^4$$, and let $$\mathcal O=\Gamma e_0$$ be the orbit of $$e_0$$, an infinite set. For $$u\in\mathcal O$$ choose $$\gamma_u\in\Gamma$$ with $$\gamma_ue_0=u$$, so that $$f^u_\mu=\gamma_ue_\mu$$ is a basis of $$\mathbb Z^4$$ with $$Q(f^u_\mu)=\eta_{\mu\mu}$$ and $$f^u_0=u$$. The model has one real scalar field $$\phi^{u}$$ for each $$u\in\mathcal O$$ and the action $$\sum_u\mathcal A_u[\phi^u]$$ with

$$
\mathcal A_u[\phi]=\tfrac12\sum_{x\in\Lambda}\Big[(\phi(x+f^u_0)-\phi(x))^2-\nu\sum_{i=1}^3(\phi(x+f^u_i)-\phi(x))^2-m^2\phi(x)^2\Big],
$$

a nearest-neighbour action in the frame $$f^u$$ with the same coefficient $$\nu$$ for the three spatial directions. An element $$(g,a)$$ of $$\Gamma\ltimes\Lambda$$ acts by $$\phi^u(x)\mapsto\phi^{gu}(gx+a)$$, permuting the labels. The action is invariant: $$\gamma_{gu}^{-1}g\gamma_u$$ lies in $$\mathrm{Stab}_\Gamma(e_0)$$, the finite group of signed spatial permutations, so $$g$$ maps the neighbour directions $$\pm f^u_i$$ to $$\pm f^{gu}_{\pi(i)}$$ for a permutation $$\pi$$, and the equal coefficients $$\nu$$ are mapped to each other.

For $$\nu$$ and $$m$$ such that the frame-$$u$$ dispersion relation $$4\sin^2(\omega/2)=4\nu\sum_i\sin^2(p_i/2)+m^2$$ is real, let $$W_u$$ be the positive-frequency (vacuum) two-point function of $$\mathcal A_u$$, defined by the same formula in each frame $$f^u$$. It is positive and invariant under translations and under $$\mathrm{Stab}_\Gamma(u)$$. Put $$W((u,x),(u',y))=\delta_{uu'}W_u(x,y)$$. This is a positive functional on the algebra of all the fields, because it is block diagonal with positive blocks, and it is invariant under $$\Gamma\ltimes\Lambda$$ by the same frame argument as for the action. Each sector has the dispersion relation of the nearest-neighbour lattice in the frame $$u$$, which is Lorentz invariant at leading order in momentum when $$\nu=1$$. The residual symmetry of a single sector is the finite group $$\mathrm{Stab}_\Gamma(u)$$, which does not tie $$\nu$$ to 1, so $$\nu=1$$ is a condition on the coupling.

The example has propagation, a symmetric action and a symmetric state with a Zariski-dense $$\Gamma$$. It does not contradict the theorems because the field has infinitely many components. Theorem 2 concerns a single field $$\phi(x)$$, and the single field $$\phi^u$$ is invariant only under $$\mathrm{Stab}_\Gamma(u)$$. Theorem 1 requires $$V=\mathbb C^N$$ with $$N<\infty$$; here $$V=\ell^2(\mathcal O)$$, the symbol is an operator-valued function with $$d_{gu}(g\cdot k)=d_u(k)$$, and this covariance does not force it to be constant, because $$\Gamma$$ acts on $$\mathbb T\times\mathcal O$$ with non-constant invariant functions.

</div>

## 8. Cayley-momentum covariance is not a symmetry of a point set

Lorente obtains a covariant lattice Klein–Gordon operator by letting Lorentz transformations act on the variables $$\varphi(k)_\mu=2\tan(k_\mu/2)$$, which we call the Cayley momentum, through $$T_g=\varphi^{-1}\circ g\circ\varphi$$ for $$g\in O(1,3)$$ [[Lorente]](#ref-lorente). The symbol $$\eta(\varphi(k),\varphi(k))-m^2$$ is invariant under this action. The question is whether the action is induced by a transformation of the points of a lattice.

<div class="proposition" markdown="1">

**Proposition 3** (point maps induce only automorphisms). Let $$\psi$$ be a bijection of $$\mathbb Z^4$$, and suppose there are a measurable map $$T:\mathbb T^4\to\mathbb T^4$$ and unit complex numbers $$\epsilon(k)$$ with $$\chi_k(\psi^{-1}(x))=\epsilon(k)\chi_{T(k)}(x)$$ for all $$k$$ and $$x$$; that is, the composition operator of $$\psi$$ maps each character to a multiple of a character. Then $$\psi$$ is affine with linear part in $$GL(4,\mathbb Z)$$, and $$T$$ is a linear automorphism of the torus.

</div>

<div class="proof" markdown="1">

*Proof.* Evaluating at $$x=0$$ gives $$\epsilon(k)=\chi_k(\psi^{-1}(0))$$. With $$\psi'(x)=\psi^{-1}(x)-\psi^{-1}(0)$$ we get $$e^{ik\cdot\psi'(x)}=e^{iT(k)\cdot x}$$ for all $$k,x$$. The left side is multiplicative in $$k$$, so $$e^{iT(k+k')\cdot x}=e^{iT(k)\cdot x}e^{iT(k')\cdot x}$$ for all $$x\in\mathbb Z^4$$, and characters separate points of the torus: $$T$$ is a homomorphism. A measurable homomorphism of the torus is continuous, hence $$T(k)=Ak$$ for an integer matrix $$A$$. Then $$e^{ik\cdot\psi'(x)}=e^{ik\cdot A^Tx}$$ for all $$k$$, so $$\psi'(x)=A^Tx$$. Since $$\psi'$$ is a bijection of $$\mathbb Z^4$$ (a translate of the bijection $$\psi^{-1}$$), $$A^T\in GL(4,\mathbb Z)$$. $$\square$$

</div>

<div class="proposition" markdown="1">

**Proposition 4** ($$T_g$$ is additive only for monomial $$g$$). Let $$g\in O(1,3)$$. Then $$T_g(k+k')=T_g(k)+T_g(k')$$ for all small $$k,k'$$ if and only if $$g$$ has exactly one non-zero entry in each row and column.

</div>

<div class="proof" markdown="1">

*Proof.* Expand $$\varphi(k)=k+k^{(3)}/12+O(k^5)$$ and $$\varphi^{-1}(y)=y-y^{(3)}/12+O(y^5)$$, where $$(\cdot)^{(3)}$$ is the componentwise cube. Then $$T_g(k)=gk+\Delta(k)+O(k^5)$$ with $$\Delta(k)=\tfrac1{12}\big(gk^{(3)}-(gk)^{(3)}\big)$$, homogeneous of degree 3. If $$T_g$$ is additive then $$T_g(2k)=2T_g(k)$$, and the cubic terms give $$6\Delta(k)=0$$, that is $$gk^{(3)}=(gk)^{(3)}$$ for all $$k$$. If some row $$i$$ of $$g$$ has two non-zero entries $$g_{ij},g_{il}$$, the coefficient of $$k_j^2k_l$$ in $$(gk)_i^3$$ is $$3g_{ij}^2g_{il}\ne0$$, while $$(gk^{(3)})_i=\sum_jg_{ij}k_j^3$$ has no such monomial. So $$g$$ is monomial (a matrix in $$O(1,3)$$ is invertible, so no row or column vanishes, and a monomial matrix has one non-zero entry in each). Conversely, a monomial $$g$$ is a signed permutation, which commutes with the odd componentwise function $$\varphi$$, so $$T_g=g$$ is linear. $$\square$$

</div>

By Proposition 3, a non-monomial Lorentz transformation, such as a boost, gives a $$T_g$$ that is induced by no point map: it is a symmetry of the dispersion relation only. We checked the cubic term in 1+1 dimensions for a boost of rapidity $$1/2$$ and $$k=r\,(0.6,0.8)$$. The ratio $$\lvert T_g(2k)-2T_g(k)\rvert/r^3$$ is $$0.68,\,0.68,\,0.66,\,0.60$$ at $$r=0.05,\,0.1,\,0.2,\,0.4$$ and tends to $$6\lvert\Delta(k)\rvert/r^3=0.680$$ as $$r\to0$$, and $$\eta(\varphi(k),\varphi(k))$$ is unchanged by $$k\mapsto T_g(k)$$.

<div class="proposition" markdown="1">

**Proposition 5** (vertices). Let $$g\in O(1,3)$$ be non-monomial. Then there are small momenta $$k,k'$$ with $$k+k'+(-k-k')=0$$ such that $$T_g(k)+T_g(k')+T_g(-k-k')\ne0$$ modulo $$2\pi$$. Consequently no three-point vertex that conserves $$\sum_jk_j$$ modulo $$2\pi$$ is covariant under $$T_g$$.

</div>

<div class="proof" markdown="1">

*Proof.* The map $$T_g=\varphi^{-1}g\varphi$$ is odd, because $$\varphi$$ and $$\varphi^{-1}$$ are odd and $$g$$ is linear; hence $$T_g(0)=0$$ and $$T_g(-k-k')=-T_g(k+k')$$. By Proposition 4 there are small $$k,k'$$ with $$T_g(k+k')\ne T_g(k)+T_g(k')$$, so $$T_g(k)+T_g(k')+T_g(-k-k')=T_g(k)+T_g(k')-T_g(k+k')\ne0$$. This vector is small by continuity of $$T_g$$ at 0, so it is non-zero modulo $$2\pi$$. A vertex supported on the triples with zero sum would be mapped by a covariant action to a vertex supported on the images, which do not have zero sum. $$\square$$

</div>

<div class="remark" markdown="1">

**Remark 3** (the covariant alternative). A vertex that is covariant must instead conserve $$\sum_j\varphi(k_j)$$, and it cannot conserve $$\sum_jk_j$$ as well, because $$\varphi$$ is not additive. The composition law $$k\oplus k'=\varphi^{-1}(\varphi(k)+\varphi(k'))$$ is commutative, associative and covariant. It has the structure of the composition laws of the doubly special relativity models of Judes and Visser [[Judes–Visser]](#ref-jv), which are obtained from a non-linear realisation of the Lorentz group by a change of variables. Identifying $$\varphi(k)$$ with the conserved momentum turns the theory into one with continuous translations and the linear Lorentz action, with the lattice as a labelling of a basis of $$L^2$$; we do not prove this identification in detail. It follows the distinction between momentum and translation generator emphasised by Mignemi [[Mignemi]](#ref-mignemi). A theory with two inequivalent notions of momentum is not excluded, only shown to carry no lattice.

</div>

If the lattice translations are kept as symmetries, covariance fails at the vertex and the lattice theory is Lorentz violating. The free massless dispersion obtained from $$\varphi(\omega)=\lvert \varphi(\vec k)\rvert $$ is, to cubic order, $$\omega=\lvert \vec k\rvert -\frac{\lvert \vec k\rvert ^3}{12}\big(1-\sum_in_i^4\big)$$ with $$n=\vec k/\lvert \vec k\rvert $$, a direction-dependent correction that vanishes along the axes.

<div class="proposition" markdown="1">

**Proposition 6** (automorphic actions are linear lattice actions or continuum actions). Let $$\mathcal M$$ be a connected four-dimensional Lie group, the group on which momenta (or translations) live, and let a subgroup $$\Gamma$$ of $$O(1,3)$$ satisfying (H1) act on it by group automorphisms, with differential at the identity equal to the vector representation. Then $$\mathcal M\cong\mathbb R^4$$ or $$\mathcal M\cong\mathbb T^4$$.

</div>

<div class="proof" markdown="1">

*Proof.* The Lie bracket on $$\mathfrak m\cong\mathbb R^4$$ is a $$\Gamma$$-equivariant linear map $$\wedge^2\mathbb R^4\to\mathbb R^4$$. Equivariance under $$g$$ is a polynomial condition, so the set of $$g$$ for which the bracket is equivariant is Zariski closed and contains $$\Gamma$$, hence $$SO(1,3)^\circ$$. As a representation of $$SO(1,3)^\circ$$, $$\wedge^2\mathbb R^4=\mathfrak{so}(1,3)$$ is the adjoint representation of a simple real Lie algebra, hence irreducible of dimension 6. It contains no copy of the four-dimensional vector representation, so by Schur's lemma the equivariant bracket vanishes. Thus $$\mathcal M=\mathbb R^4/\mathcal K$$ with $$\mathcal K=\ker\exp$$ discrete. Its real span is a $$\Gamma$$-invariant, hence $$SO(1,3)^\circ$$-invariant, subspace, so it is $$0$$ or $$\mathbb R^4$$, and a discrete subgroup spanning $$\mathbb R^4$$ is a lattice. $$\square$$

</div>

In the torus case $$\Gamma$$ acts linearly through $$GL(4,\mathbb Z)$$, which is the setting of Theorems 1 and 2. In the $$\mathbb R^4$$ case there is no discreteness. What remains is a Lorentz action on momentum space that is not by group automorphisms, as in $$\kappa$$-Poincaré constructions, which are not realised on a point set.

## 9. Spectral triples on the Brillouin quotient

A spectral triple, in the sense of non-commutative geometry, is an algebra acting on a Hilbert space together with a self-adjoint operator $$D$$ with compact resolvent whose commutators with the algebra are bounded. The operator $$D$$ plays the role of a Dirac operator, and the dimension of the geometry is read from the growth of its spectrum: for a Dirac operator on a compact $$d$$-dimensional manifold the number $$N_D(R)$$ of eigenvalues in $$[-R,R]$$ grows like $$R^d$$. A triple is called $$p$$-summable when $$N_D(R)\le C(1+R)^p$$, and $$\theta$$-summable when $$\operatorname{Tr}e^{-tD^2}<\infty$$ for every $$t>0$$.

Proposition 6 leaves open whether the non-commutative quotient $$C(\mathbb T^n)\rtimes\Gamma$$ of the Brillouin torus by the integral Lorentz group could carry a geometry of finite spectral dimension. On $$\mathcal H=L^2(\mathbb T^n)\otimes V$$ the algebra is generated by the translation unitaries $$T_x$$ and the operators $$U_g$$ of (2), taken with $$\sigma$$ unitary. A *covariant Brillouin triple* is a self-adjoint $$D$$ with compact resolvent on $$\mathcal H$$ such that $$T_{e_i}$$ and $$U_g$$, for $$g$$ in a finite generating set of $$\Gamma$$, and their inverses, preserve $$\mathrm{dom}\,D$$ and have bounded commutators with $$D$$. It is *exactly equivariant* if $$U_gDU_g^{-1}=D$$ for all $$g\in\Gamma$$.

<div class="theorem" markdown="1">

**Theorem 4** (exact equivariance excludes compact resolvent). Assume (F) for $$\Gamma$$ acting on $$\Lambda=\mathbb Z^n$$. There is no self-adjoint operator $$D$$ with compact resolvent on $$L^2(\mathbb T^n)\otimes V$$, with $$V$$ finite-dimensional and $$\sigma$$ a representation of $$\Gamma$$ on $$V$$ (not necessarily unitary), that satisfies $$U_gDU_g^{-1}=D$$ for all $$g\in\Gamma$$.

</div>

<div class="proof" markdown="1">

*Proof.* Let $$\mathcal E$$ be an eigenspace of $$D$$; it is finite-dimensional because the resolvent is compact, and invariant under every $$U_g$$. Let $$F\subset L^2(\mathbb T^n)$$ be the span of the components $$(\mathrm{id}\otimes\ell)(\vartheta)$$, $$\vartheta\in\mathcal E$$, $$\ell\in V^\ast$$; it is finite-dimensional. For $$f=(\mathrm{id}\otimes\ell\sigma(g))\vartheta$$ one has $$K_gf=(\mathrm{id}\otimes\ell)(U_g\vartheta)\in F$$, and as $$\ell$$ ranges over $$V^\ast$$ so does $$\ell\sigma(g)$$, because $$\sigma(g)$$ is invertible; so $$F$$ is $$K$$-invariant. Its projection onto $$L^2_0$$ is invariant and finite-dimensional, hence zero by Lemma 2(i). Thus every component of every eigenvector is constant, $$\mathcal E\subset1\otimes V$$. The eigenvectors of a self-adjoint operator with compact resolvent span $$\mathcal H$$, so $$\mathcal H=1\otimes V$$ is finite-dimensional, which contradicts $$\dim L^2(\mathbb T^n)=\infty$$. $$\square$$

</div>

The Lorentzian Dirac-type operator $$D_L=\gamma^\mu x_\mu$$ in position space is exactly covariant, but each eigenvalue has infinite multiplicity (the value $$\sqrt Q$$ is attained on a whole orbit), so its resolvent is not compact; this is consistent with the theorem.

If covariance is only required up to bounded terms, the obstruction is a different one: the translations are exponentially distorted in $$G=\Lambda\rtimes\Gamma$$. Fix a finite generating set of $$G$$, and write $$\lvert x\rvert _G$$ for the word length of the translation $$t_x$$, $$x\in\Lambda$$.

<div class="lemma" markdown="1">

**Lemma 3** (exponential distortion of translations). (i) Let $$A\in SL(2,\mathbb Z)$$ with $$\operatorname{tr}A>2$$ and $$G_A=\mathbb Z^2\rtimes\langle A\rangle$$. There is $$C$$ with $$\lvert x\rvert _{G_A}\le C\log(2+\lvert x\rvert )$$ for all $$x\in\mathbb Z^2$$. (ii) Let $$\Gamma$$ be a subgroup of finite index in $$O(1,3;\mathbb Z)$$ and $$G=\mathbb Z^4\rtimes\Gamma$$. Then $$\lvert x\rvert _G\le C\log(2+\lvert x\rvert )$$ for all $$x\in\mathbb Z^4$$.

</div>

The difficulty is that a translation by a large vector costs a long word in the translation generators alone. Conjugation by $$A$$ stretches one eigendirection by $$\lambda>1$$, so a large vector can be written as a short sum of integer vectors from a bounded set, each conjugated by a power of $$A$$, in the manner of a base-$$\lambda$$ expansion.

<div class="proof" markdown="1">

*Proof.* (i) Let $$\lambda>1$$ be the larger eigenvalue of $$A$$. Write $$x=u(x)e_u+s(x)e_s$$ in the eigenbasis, so that $$A$$ acts by $$(u,s)\mapsto(\lambda u,s/\lambda)$$. Let $$r_c$$ be the covering radius of $$\mathbb Z^2$$ in the sup norm of these coordinates and $$\tau_0=\lambda r_c$$.

Step 1 (points near the unstable axis). Suppose $$\lvert s(y)\rvert \le\tau_0$$. The rectangle $$\{\lvert u\rvert \le r_c,\ \lvert s-s(y)\rvert \le\tau_0/\lambda\}$$ contains a ball of sup-radius $$r_c$$, hence a lattice point $$v$$ of the finite set $$F=\{v\in\mathbb Z^2:\lvert u(v)\rvert \le r_c,\ \lvert s(v)\rvert \le\tau_0+\tau_0/\lambda\}$$. Put $$y'=A^{-1}(y-v)$$, an integer vector because $$\det A=1$$. Then $$\lvert s(y')\rvert =\lambda\lvert s(y)-s(v)\rvert \le\tau_0$$ and $$\lvert u(y')\rvert \le(\lvert u(y)\rvert +r_c)/\lambda$$. Iterating, after $$k$$ steps $$\lvert u\rvert \le\lvert u(y)\rvert \lambda^{-k}+r_c/(\lambda-1)$$, so after $$k=\log_\lambda(2+\lvert u(y)\rvert )+O(1)$$ steps the point lies in a fixed finite set $$\mathcal B_0$$. This gives $$y=\sum_{j<k}A^jv_j+A^ky_k$$ with $$v_j\in F$$ and $$y_k\in \mathcal B_0$$.

Step 2 (the word). Let $$a$$ be the generator acting as $$A$$, so $$a\,t_za^{-1}=t_{Az}$$. Then $$t_{\sum_{j<k}A^jv_j}=t_{v_0}\,a\,t_{v_1}\,a\,t_{v_2}\cdots a\,t_{v_{k-1}}\,a^{-(k-1)}$$ and $$t_{A^ky_k}=a^kt_{y_k}a^{-k}$$. The elements $$t_v$$ with $$v\in F\cup \mathcal B_0$$ have bounded length, so $$\lvert t_y\rvert _{G_A}\le C_1k+C_2$$.

Step 3 (general $$x$$). The same construction with $$A^{-1}$$ treats points with $$\lvert u\rvert \le\tau_0$$ and large $$\lvert s\rvert $$. Given $$x\in\mathbb Z^2$$, choose a lattice point $$p$$ within sup-distance $$r_c$$ of $$u(x)e_u$$, then a lattice point $$q$$ within $$r_c$$ of the stable-axis point of $$x-p$$. Then $$x=p+q+r$$ with $$\lvert s(p)\rvert \le r_c$$, $$\lvert u(q)\rvert \le r_c$$, $$r$$ in a finite set, and $$\lvert p\rvert ,\lvert q\rvert \le C\lvert x\rvert $$. Hence $$\lvert x\rvert _{G_A}\le C\log(2+\lvert x\rvert )$$.

(ii) Let $$A=\begin{pmatrix}3&4\\2&3\end{pmatrix}$$, which preserves $$a^2-2b^2$$ since $$(3a+4b)^2-2(2a+3b)^2=a^2-2b^2$$. Let $$g_0\in GL(4,\mathbb R)$$ act as $$A$$ on the coordinates $$(a,b)$$ of $$ae_0+b(e_1+e_2)$$ and trivially on $$e_1-e_2$$ and $$e_3$$. These vectors are mutually orthogonal, with $$Q(e_0)=1$$, $$Q(e_1+e_2)=Q(e_1-e_2)=-2$$, so $$g_0$$ preserves $$Q$$. It maps $$e_1$$ to $$2e_0+2e_1+e_2$$ and $$e_2$$ to $$2e_0+e_1+2e_2$$, so $$g_0\in O^+(1,3;\mathbb Z)$$, and a power $$h=g_0^N$$, $$N\ge1$$, lies in the finite-index subgroup $$\Gamma$$ (two powers of $$g_0$$ lie in the same coset). Put $$P=\mathbb R e_0+\mathbb R(e_1+e_2)$$ and $$P_0=\mathbb Z^4\cap P=\mathbb Ze_0+\mathbb Z(e_1+e_2)$$. For $$\gamma\in\Gamma$$ the element $$\gamma h\gamma^{-1}\in\Gamma$$ preserves $$\gamma P$$ and acts on $$\gamma P_0=\mathbb Z^4\cap\gamma P$$ as a hyperbolic element of $$SL(2,\mathbb Z)$$ in the basis $$\gamma(e_0,e_1+e_2)$$. Part (i) applies on each such plane, with the translations of $$\gamma P_0$$ and the element $$\gamma h\gamma^{-1}$$ as generators, whose lengths in $$G$$ are bounded; the relation $$a\,t_za^{-1}=t_{az}$$ holds in $$G$$ for all $$z\in\mathbb Z^4$$. So $$\lvert z\rvert _G\le C\log(2+\lvert z\rvert )$$ for $$z\in P_0\cup\gamma P_0$$.

The condition $$P\oplus\gamma P=\mathbb R^4$$ is the non-vanishing of a determinant, a polynomial in the entries of $$\gamma$$. It is not identically zero on $$SO(1,3)^\circ$$: take $$\gamma$$ to be a boost of rapidity $$s\ne0$$ in the $$e_0e_3$$ plane composed with a rotation by $$\pi/2$$ in the $$e_1e_2$$ plane, so that $$\gamma e_0=\cosh s\,e_0+\sinh s\,e_3$$ and $$\gamma(e_1+e_2)=-(e_1-e_2)$$. The four vectors $$e_0$$, $$e_1+e_2$$, $$\gamma e_0$$, $$\gamma(e_1+e_2)$$ are independent. By (H1) some $$\gamma\in\Gamma$$ satisfies the condition. Then $$P_0\oplus\gamma P_0$$ has finite index in $$\mathbb Z^4$$, and every $$x$$ is $$x_1+x_2+r$$ with $$x_1\in P_0$$, $$x_2\in\gamma P_0$$, $$\lvert x_i\rvert \le C\lvert x\rvert $$ and $$r$$ in a finite set of coset representatives. Then $$\lvert x\rvert _G\le\lvert x_1\rvert _G+\lvert x_2\rvert _G+\lvert r\rvert _G$$. $$\square$$

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

*Proof.* The vectors $$\zeta_x=T_x\zeta$$, $$x\in\Lambda$$, with $$\zeta=1\otimes v_0$$ and $$\lVert v_0\rVert =1$$, are orthonormal. Let $$C_D$$ bound the commutators of $$D$$ with the generators and their inverses. For a product $$u$$ of $$m$$ generators or inverses, all partial products preserve $$\mathrm{dom}\,D$$, and the Leibniz rule gives $$\lVert [D,u]\rVert \le C_Dm$$. Choose $$\xi\in\mathrm{dom}\,D$$ with $$\lVert \xi-\zeta\rVert \le\tfrac14$$ and put $$K_\xi=\lVert D\xi\rVert $$. For $$\lvert x\rvert _G\le m$$, $$\lVert DT_x\xi\rVert \le\lVert T_xD\xi\rVert +\lVert [D,T_x]\xi\rVert \le K_\xi+\tfrac54C_Dm$$. Put $$R=4K_\xi+5C_Dm$$ and $$P=\mathbf 1_{[-R,R]}(D)$$. Then $$\lVert (1-P)T_x\xi\rVert \le\lVert DT_x\xi\rVert /R\le\tfrac14$$, so $$\lVert (1-P)\zeta_x\rVert \le\tfrac14+\lVert \zeta-\xi\rVert \le\tfrac12$$ and $$\langle\zeta_x,P\zeta_x\rangle=1-\lVert (1-P)\zeta_x\rVert ^2\ge\tfrac34$$. Bessel's inequality for the orthonormal family gives $$\operatorname{rank}P\ge\sum_x\langle\zeta_x,P\zeta_x\rangle\ge\tfrac34\,\#\{x:\lvert x\rvert _G\le m\}$$. By Lemma 3 every $$x$$ with $$\lvert x\rvert _\infty\le r$$ has $$\lvert x\rvert _G\le C\log(2+r)+c_0$$, so for $$r=e^{(m-c_0)/C}-2$$ the count is at least $$(2r+1)^n\ge e^{n(m-c_0)/C}$$. Since $$R$$ is affine in $$m$$, this is (7) with $$\kappa=n/(5C_DC)$$. A $$p$$-summable $$D$$ has $$N_D(R)\le C'(1+R)^p$$, which (7) violates. $$\square$$

</div>

The estimate is sharp in the following sense. The operator $$D=\lvert x\rvert _G$$ acting by multiplication in position space has bounded commutators with all generators (since $$\lvert x+a\rvert _G\le\lvert x\rvert _G+\lvert a\rvert _G$$ and $$\lvert gx\rvert _G\le\lvert x\rvert _G+2\lvert g\rvert _G$$), compact resolvent, and $$\operatorname{Tr}e^{-tD^2}=\sum_xe^{-t\lvert x\rvert _G^2}<\infty$$ for every $$t>0$$, because the number of elements of $$G$$ of length at most $$R$$ is at most exponential in $$R$$. It is therefore $$\theta$$-summable, and by Theorem 5 not finitely summable.

Whittaker has built finitely summable spectral triples for the stable-groupoid algebras of Smale spaces such as a hyperbolic toral automorphism [[Whittaker]](#ref-whit). These replace $$C(\mathbb T^n)\rtimes\Gamma$$ by a different algebra and are not covered by the theorems above.

## 10. States centred on one lattice point

Theorem 2 removes translations from the discussion only by assuming them. Take instead $$\Gamma$$ to fix the origin of $$\Lambda=\mathbb Z^4$$ and impose no translation symmetry. A positive $$\Gamma$$-invariant state is then the arithmetic version of a Milne universe about the origin.

Two facts about stabilisers in $$\Gamma$$, a finite-index subgroup of $$O^+(1,3;\mathbb Z)$$, are used. They are consequences of Theorem C1.

(S1) For a primitive spacelike $$v\in\Lambda$$, $$\mathrm{Stab}_\Gamma(v)$$ is a Zariski-dense subgroup of $$SO(v^\perp)^\circ\cong SO(1,2)^\circ$$ and contains a hyperbolic element, that is, one with null eigenvectors of eigenvalues $$\lambda^{\pm1}$$, $$\lambda>1$$. Indeed, $$\mathrm{Stab}_{O(1,3;\mathbb Z)}(v)$$ acts on the lattice $$\Lambda\cap v^\perp$$ with the integral form of signature $$(1,2)$$ induced by $$Q$$; it is contained in the automorphism group of that lattice and contains a finite-index subgroup of it (those automorphisms that act trivially on the finite quotient of $$\Lambda$$ by $$\mathbb Zv\oplus(\Lambda\cap v^\perp)$$). That automorphism group is a lattice by C1, hence $$\mathrm{Stab}_\Gamma(v)$$, of finite index in $$\mathrm{Stab}_{O^+(1,3;\mathbb Z)}(v)$$, is a lattice and Zariski dense in $$SO(1,2)^\circ$$. A Zariski-dense discrete subgroup of $$SO(1,2)^\circ\cong PSL(2,\mathbb R)$$ is non-elementary (an elementary group lies in a proper algebraic subgroup: the stabiliser of a point of the closed hyperbolic disc or of a pair of boundary points), and a non-elementary Fuchsian group contains hyperbolic elements.

(S2) For a primitive null $$v\in\Lambda$$ and $$b\in\Lambda\cap v^\perp$$, the null rotation $$N_b(y)=y+B(y,v)b-B(y,b)v-\tfrac12Q(b)B(y,v)v$$ (an isometry fixing $$v$$, with $$N_bN_{b'}=N_{b+b'}$$) has $$N_{2b}$$ integral, so $$N_{2b}\in O^+(1,3;\mathbb Z)$$ and a power of it lies in $$\Gamma$$.

<div class="proposition" markdown="1">

**Proposition 7** (correlations of points outside the cone of the centre). Let $$\Gamma$$ be a finite-index subgroup of $$O^+(1,3;\mathbb Z)$$, $$\Lambda=\mathbb Z^4$$, and let $$W(x,y)=\omega(\phi(x)^\ast\phi(y))$$ for a positive state $$\omega$$ on a $$\ast$$-algebra containing the fields $$\phi(x)$$, $$x\in\Lambda$$, with $$W(gx,gy)=W(x,y)$$, $$g\in\Gamma$$. Assume

(WC) if $$y\in\Lambda$$ and $$\gamma_n\in\Gamma$$ satisfy $$\lvert B(y,\gamma_ny)\rvert \to\infty$$, then $$W(y,\gamma_ny)\to0$$.

Then:

(a) $$W(0,y)=0$$ for all $$y\in\Lambda\setminus\{0\}$$;

(b) $$W(v,y)=0$$ for spacelike $$v\in\Lambda$$ and all $$y\in\Lambda\setminus\mathbb Rv$$;

(c) $$W(v,y)=0$$ for null $$v\in\Lambda$$ and all $$y\in\Lambda$$ with $$B(v,y)\ne0$$.

</div>

<div class="proof" markdown="1">

*Proof.* Let $$\xi_x=\phi(x)\Omega$$ be the vectors of the Gelfand–Naimark–Segal construction, so $$W(x,y)=\langle\xi_x,\xi_y\rangle$$. Since $$W$$ is $$\Gamma$$-invariant, $$U(g)\xi_x=\xi_{gx}$$ defines an isometry on the closed span of the $$\xi_x$$.

Let $$\ell\in\Gamma$$ fix $$v$$ (in (a) take $$v=0$$, fixed by every $$\ell$$), so $$U(\ell)\xi_v=\xi_v$$. By the mean ergodic theorem for isometries on a Hilbert space, $$\frac1N\sum_{k<N}U(\ell)^k\xi_y\to P\xi_y$$, with $$P$$ the projection onto the fixed vectors of $$U(\ell)$$. If $$\lvert B(y,\ell^ky)\rvert \to\infty$$, then (WC) gives $$\langle\xi_y,U(\ell)^k\xi_y\rangle=W(y,\ell^ky)\to0$$, so $$\lVert P\xi_y\rVert ^2=\lim\frac1N\sum_k\langle\xi_y,U(\ell)^k\xi_y\rangle=0$$. As $$\xi_v$$ is fixed, $$W(v,y)=\langle P\xi_v,\xi_y\rangle=\langle\xi_v,P\xi_y\rangle=0$$. It remains to find, in each case, $$\ell$$ fixing $$v$$ with $$\lvert B(y,\ell^ky)\rvert \to\infty$$.

In (a), let $$\ell_0\in\Gamma$$ be hyperbolic (the element $$h$$ of Lemma 3(ii) is one), with null eigenvectors $$w_\pm$$ of eigenvalues $$\lambda^{\pm1}$$; it acts as a rotation on the plane $$\{w_+,w_-\}^\perp$$. For $$\gamma\in\Gamma$$, $$\ell=\gamma\ell_0\gamma^{-1}$$ has eigenvectors $$\gamma w_\pm$$. Write $$y=\alpha\gamma w_++\beta\gamma w_-+y_0$$ with $$y_0\perp\gamma w_\pm$$; then $$\alpha\propto B(y,\gamma w_-)$$, $$\beta\propto B(y,\gamma w_+)$$, and $$B(y,\ell^ky)=\alpha\beta(\lambda^k+\lambda^{-k})B(\gamma w_+,\gamma w_-)+B(y_0,\ell^ky_0)$$ tends to $$\pm\infty$$ if $$\alpha\beta\ne0$$, because the last term is bounded. The function $$\gamma\mapsto B(\gamma^{-1}y,w_+)B(\gamma^{-1}y,w_-)$$ is a polynomial on $$SO(1,3)$$ that is not identically zero: if $$B(\cdot,w_+)$$ vanished on the orbit of $$y$$, which spans $$\mathbb R^4$$ by irreducibility, then $$w_+=0$$. Since $$\Gamma^+$$ is Zariski dense in $$SO(1,3)^\circ$$, some $$\gamma$$ gives $$\alpha\beta\ne0$$.

In (b), write $$y=cv+y'$$ with $$y'\in v^\perp$$; $$y'\ne0$$ because $$y\notin\mathbb Rv$$. The same argument inside $$v^\perp\cong\mathbb R^{1,2}$$ with the Zariski-dense group $$\mathrm{Stab}_\Gamma(v)$$ of (S1) gives a hyperbolic $$\ell$$ fixing $$v$$ with $$B(y',\ell^ky')\to\infty$$, and $$B(y,\ell^ky)=c^2Q(v)+B(y',\ell^ky')$$.

In (c), take $$b\in\Lambda\cap v^\perp$$ not proportional to $$v$$; since $$v^\perp/\mathbb Rv$$ is negative definite, $$Q(b)<0$$. We have $$B(y,N_{kb}y)=Q(y)-\tfrac12k^2Q(b)B(y,v)^2$$, which tends to $$+\infty$$ when $$B(y,v)\ne0$$. By (S2) take $$\ell=N_{2b}^m\in\Gamma$$ for a suitable $$m\ge1$$. $$\square$$

</div>

Part (a) says that a clustering invariant state has no propagation from the centre, and (b) and (c) that points spacelike or null related to the centre propagate to nothing else, apart from their own rational multiples in the spacelike case. Correlations are possible between points inside the light cone of the centre. Timelike vectors have finite stabilisers in $$\Gamma$$ (they lie in a compact $$O(3)$$), so the argument constrains nothing for them. Condition (WC) is a hypothesis; it is not a consequence of positivity and invariance.

<div class="proposition" markdown="1">

**Proposition 8** (a state that propagates inside the cone). Let $$\Phi$$ be a free massive scalar field on Minkowski space with Fock vacuum $$\Omega$$, let $$f_0$$ be a real Schwartz function on $$\mathbb R^4$$ invariant under spatial rotations, let $$\Xi_x$$ be a boost taking $$e_0$$ to $$u(x)=x/\sqrt{Q(x)}$$, and put $$f_x(y)=f_0(\Xi_x^{-1}(y-x))$$ for $$x$$ in the open future cone. The functional $$\omega(\phi(x)\phi(y))=\langle\Omega,\Phi(f_x)\Phi(f_y)\Omega\rangle$$ on $$\Lambda\cap\{Q>0,\ x_0>0\}$$ is positive and invariant under $$O^+(1,3)$$, hence under $$\Gamma$$; its spectral support is the mass shell; and it satisfies (WC) on pairs of points in the open cone.

</div>

<div class="proof" markdown="1">

*Proof.* Positivity: $$W$$ is the Gram matrix of the vectors $$\Phi(f_x)\Omega$$. Invariance: for $$g\in O^+(1,3)$$, $$u(gx)=gu(x)$$, so $$\Xi_{gx}=g\Xi_xR$$ with $$R$$ a spatial rotation, and since $$f_0$$ is rotation invariant, $$f_{gx}=f_x\circ g^{-1}$$. The vacuum is invariant, so $$\langle\Phi(f_{gx})\Phi(f_{gy})\rangle=\langle\Phi(f_x)\Phi(f_y)\rangle$$. The spectral support is that of the vacuum two-point function, the mass shell. For (WC), $$W(y,\gamma_ny)=\langle\psi,\pi(h_n)\psi\rangle$$ for a fixed vector $$\psi$$ in the irreducible unitary representation $$\pi$$ of $$SO^+(1,3)$$ on $$L^2$$ of the mass shell and $$h_n=\Xi_y^{-1}\gamma_n\Xi_y$$ up to rotations; $$h_n\to\infty$$ in the group when $$B(y,\gamma_ny)\to\infty$$, and matrix coefficients of irreducible unitary representations vanish at infinity (C4). $$\square$$

</div>

This state is defined on the open cone only, not on all of $$\Lambda$$, so it does not meet the hypothesis of Proposition 7; it shows that the inside of the cone can carry correlations of a Lorentz-invariant field.

In the Milne frame $$u(x)$$ the symmetry fixing the point $$x$$ is the compact $$O(3)_{u(x)}$$ only, so a separate $$(u\cdot\partial\phi)^2$$ term is allowed for each field species without breaking the symmetry. The unit timelike field $$u(x)=x/\sqrt{Q(x)}$$ then plays the role of the aether of Einstein–aether theories, and local Lorentz invariance inside the cone is a property of the couplings and not of the symmetry.

## 11. Cases outside the theorems

The theorems concern fields with finitely many components, states and operators with exact translation invariance, and Zariski-dense $$\Gamma$$. Three further cases are not covered by them and are not excluded by them.

Point sets with $$\Gamma$$-symmetry and no fixed translation lattice are outside the statements as written. A single realisation of an orbit lattice is covered, as noted in the introduction. What is not covered is an ensemble whose realisations have different translation groups, and a generic lattice $$g\mathbb Z^4$$ with $$g\in SL(4,\mathbb R)$$, whose Lorentz stabiliser is finite, so that (F) fails. See [Lorentz-invariant point sets](/quantum-gravity/invariant-point-sets/) for random lattices.

Spectral-triple geometries of infinite spectral dimension ($$\theta$$-summable) exist by the sharpness remark after Theorem 5. Geometries supported on torsion points, and geometries on a different algebra such as Whittaker's Smale-space groupoid algebra, are not covered by Theorems 4 and 5.

Whether the tuning of couplings in Example 1 and in the models allowed by Theorem 3(b) is stable under radiative corrections is a question about the couplings, which the symmetry does not decide.

Several points are open. Condition (WC) is assumed in Proposition 7, not derived from the state axioms, and (a) to (c) say nothing for states that do not cluster, including superselected and indefinite-metric ones. Whether a null-rotation argument also controls the pairs $$(v,y)$$ with $$B(v,y)=0$$ in (c) is not settled. Whether a state on all of $$\Lambda$$ can combine the interior of the cone of Proposition 8 with the vanishing correlations of Proposition 7 is not decided. Case 3 of Theorem 3(b), an irrational unique invariant null line, requires a non-reductive $$L_H$$ and is open. The distortion Lemma 3(ii) is proved for finite-index subgroups of $$O(1,3;\mathbb Z)$$; for other Zariski-dense arithmetic groups, such as the automorphism group of an anisotropic form, it is not proved here, and Theorem 5 for them is open. For signed retuned kernels the value of the packet ratio $$\rho$$ of Proposition 2 is not determined.

## References

1. <a id="ref-bq"></a>Benoist, Quint, "Mesures stationnaires et fermés invariants des espaces homogènes," Ann. of Math. 174 (2011) 1111–1162; English translation by B. Weiss, [arXiv:1610.05912](https://arxiv.org/abs/1610.05912).
2. <a id="ref-bflm"></a>Bourgain, Furman, Lindenstrauss, Mozes, "Stationary measures and equidistribution for orbits of nonabelian semigroups on the torus," J. Amer. Math. Soc. 24 (2011) 231–280.
3. <a id="ref-dln"></a>Di Scala, Leistner, Neukirchner, "Geometric applications of irreducible representations of Lie groups," IRMA Lect. Math. Theor. Phys. 16 (2010) 629–651; arXiv version titled "Irreducibly acting subgroups of $$Gl(n,\mathbb R)$$", [arXiv:math/0507047](https://arxiv.org/abs/math/0507047).
4. <a id="ref-borel"></a>Borel, "Density properties for certain subgroups of semi-simple groups without compact components," Ann. of Math. 72 (1960) 179.
5. <a id="ref-bhc"></a>Borel, Harish-Chandra, "Arithmetic subgroups of algebraic groups," Ann. of Math. 75 (1962) 485.
6. <a id="ref-hm"></a>Howe, Moore, "Asymptotic properties of unitary representations," J. Funct. Anal. 32 (1979) 72–96.
7. <a id="ref-lorente"></a>Lorente, "Discrete reflection groups and induced representations of Poincare group on the lattice," [arXiv:hep-lat/0401006](https://arxiv.org/abs/hep-lat/0401006).
8. <a id="ref-jv"></a>Judes, Visser, "Conservation laws in doubly special relativity," Phys. Rev. D 68 (2003) 045001, [arXiv:gr-qc/0205067](https://arxiv.org/abs/gr-qc/0205067).
9. <a id="ref-mignemi"></a>Mignemi, "Doubly special relativity and translation invariance," Phys. Lett. B 672 (2009) 186, [arXiv:0808.1628](https://arxiv.org/abs/0808.1628).
10. <a id="ref-cg"></a>Cohen, Glashow, "Very special relativity," Phys. Rev. Lett. 97 (2006) 021601, [arXiv:hep-ph/0601236](https://arxiv.org/abs/hep-ph/0601236).
11. <a id="ref-whit"></a>Whittaker, "Spectral triples for hyperbolic dynamical systems," J. Noncommut. Geom. 7 (2013) 563–582, [arXiv:1011.3292](https://arxiv.org/abs/1011.3292).
