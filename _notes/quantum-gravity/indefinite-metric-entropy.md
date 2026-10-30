---
title: Entropy of states in indefinite-metric theories
date: 2026-10-29
category: continuum
order: 1
description: For a generalised free field with a ghost, the Lorentz-invariant positive-energy vacuum is not positive on the algebra of any non-empty open region and so defines no entropy through a positive state, while positive states remain if the ghost has negative energy or is absent from the algebra.
status: "Proved: finite Krein algebras have no nonzero positive functional (Theorem 1); the vacuum with signed spectral measure is not positive (Theorem 2, Proposition 3); boost scaling in a finite-trace factor (Proposition 4); prescription independence of the static response, for a scalar and for spin 2 under a stated assumption (Proposition 5). The mass bound is an estimate."
excludes: "The entropy of the Lorentz-invariant, translation-invariant, positive-energy vacuum of a generalised free field with a ghost, on the algebra of any non-empty open region. Exact boost covariance with positive null energy, affiliated to the algebra, in a factor with a finite trace. Not excluded: positive states with a negative-energy ghost, algebras without the ghost, infinite traces."
---

A ghost, a mode of negative norm as in Lee–Wick models and quadratic gravity, obstructs the definition of the entropy of a region from the Lorentz-invariant vacuum. For a generalised free field whose two-point function contains the ghost, that vacuum has positive energy and its functional takes negative values on the algebra of every non-empty open region (Theorem 2), and no translation-invariant functional with positive energy and a tempered two-point function is positive on the algebra of the whole space (Proposition 3). A positive state exists if the ghost carries negative energy, and Theorem 2 does not apply to a physical algebra from which the ghost is absent, as in the fakeon prescription (section 4).

Independently of the ghost, a factor with a finite trace cannot carry the exact boost symmetry of a horizon if the null energy is positive and affiliated to the algebra (Proposition 4), which limits the algebras on which an entropy can be built. Whether an entropy exists depends on the reading of the ghost chosen in section 4, so the last two sections treat what does not: the static response is the same for every Lorentz-invariant treatment of the mass shell, in a scalar model and, under an assumption stated in section 6, for spin 2 (Proposition 5), and a torsion-balance estimate of the ghost mass follows.

Krein spaces, the state spaces with an indefinite inner product, are treated in [[Bognar]](#ref-bognar). Theorem 1 shows directly that the algebra of all operators of a finite-dimensional Krein space has no positive functional other than 0, hence no state, and Theorem 2 combines the Reeh–Schlieder theorem [[RS61]](#ref-rs61) with the sign of the spectral measure. Proposition 4 follows from the uniqueness of the trace on a factor, and Proposition 5 from standard properties of Green functions of the Klein–Gordon equation, applied to the prescriptions proposed for ghosts [[LW]](#ref-lw) [[AP]](#ref-ap) [[Ans]](#ref-ans) [[DM]](#ref-dm). Applied to the entropy constructions of recent algebraic treatments of gravity [[Witten]](#ref-witten) [[CLPW]](#ref-clpw) [[CPW]](#ref-cpw), these facts show which of the constructions can start from the vacuum of a ghost theory.

The page on [positive spectral weight and spectral dimension](/quantum-gravity/positivity-dimension-lemma/) shows that, under its hypotheses, a spectral dimension below that of space requires a negative spectral weight or one of three other departures; this page treats what a negative weight does to the state. The place of the result among other approaches is shown in the [map of the quantum-gravity families examined here](/quantum-gravity/landscape/). The regulator dependence of [the induced Newton constant](/quantum-gravity/induced-gravity-number/), which includes Pauli–Villars regulators of indefinite weight, is treated on that page.

## 1. What an entropy of a region requires

The observables of a region form an algebra. In the algebraic formulation this is a von Neumann algebra $$\mathcal N$$ of operators on a Hilbert space $$\mathcal H$$, and a state is a linear functional $$\omega$$ with $$\omega(1)=1$$ and $$\omega(A^*A)\ge0$$ for all $$A$$. A state is *faithful* if $$\omega(A^*A)=0$$ only for $$A=0$$, and *normal* if it is continuous under increasing limits of positive operators. A self-adjoint operator is *affiliated* to $$\mathcal N$$ if all its spectral projections belong to $$\mathcal N$$. A *trace* on $$\mathcal N$$ is a positive functional $$\tau$$ with $$\tau(AB)=\tau(BA)$$, and a *factor* is an algebra with trivial centre. Whether a trace exists, and what it looks like, sorts factors into three types.

The algebra $$B(\mathcal H)$$ of all bounded operators is the factor of type $$I_n$$ if $$\dim\mathcal H=n<\infty$$ and of type $$I_\infty$$ otherwise. It has minimal projections and a trace $$\mathrm{Tr}$$ equal to $$1$$ on them, and every normal state is a density matrix, $$\omega=\mathrm{Tr}(\rho\,\cdot)$$. The entropy $$-\mathrm{Tr}\,\rho\ln\rho$$ is absolute and bounded by $$\ln n$$ in $$I_n$$, and $$e^S$$ counts states in the sense that the maximally mixed state of an $$m$$-dimensional subspace has $$S=\ln m$$. A factor of type $$II$$ has a trace $$\tau$$ but no minimal projections; it is $$II_1$$ if $$\tau(1)<\infty$$ and $$II_\infty$$ otherwise. A state is still $$\omega=\tau(\rho\,\cdot)$$ with $$\rho$$ affiliated to $$\mathcal N$$, and the entropy $$-\tau(\rho\ln\rho)$$ is defined, but $$\tau$$ is fixed only up to a positive factor $$c$$, which shifts the entropy by $$\ln c$$. In a $$II_1$$ factor with $$\tau(1)=1$$ the entropy is at most $$0$$, attained at the tracial state, and only differences are meaningful. A factor of type $$III$$ has no non-zero trace, so the entropy of a state must be regularised or replaced by a relative entropy.

In relativistic quantum field theory the algebra of a wedge is of type $$III_1$$ (see the discussion in [[Witten]](#ref-witten)), which is why the entropy of a region is infinite and regulator dependent. Including gravitational dressing turns the algebra into one of type $$II$$. The crossed product of a type $$III_1$$ algebra by its modular group is of type $$II_\infty$$, and describes a black hole at large $$N$$ in AdS/CFT [[Witten]](#ref-witten) [[CPW]](#ref-cpw); for the static patch of de Sitter space with an observer the algebra is of type $$II_1$$ [[CLPW]](#ref-clpw).

Each of these constructions starts from a faithful normal state. Tomita–Takesaki theory attaches to such a state $$\omega$$ a modular operator $$\Delta$$, and for the vacuum and a wedge the Bisognano–Wichmann theorem [[BW]](#ref-bw) identifies $$\Delta^{it}$$ with the boost, which is where the temperature $$1/2\pi$$ of a horizon comes from. The question for a ghost theory is therefore whether a positive state of the algebra of the region exists, and which one.

## 2. Krein spaces and finite algebras

A harmonic oscillator whose operators obey $$[b,b^\dagger]=-1$$ has a one-particle state $$b^\dagger\lvert0\rangle$$ of norm $$\langle0\rvert bb^\dagger\lvert0\rangle=-1$$. The probability interpretation fails and the space of states carries an indefinite form. Such a mode is the ghost of a Pauli–Villars regulator, a Lee–Wick partner [[LW]](#ref-lw), or the massive spin-2 mode of quadratic gravity [[Stelle77]](#ref-stelle77).

A *Krein space* is a complex vector space $$\mathcal K$$ with a non-degenerate Hermitian form $$\langle\cdot,\cdot\rangle$$ that splits as $$\mathcal K=\mathcal K_+\oplus\mathcal K_-$$, the form being positive definite on $$\mathcal K_+$$ and negative definite on $$\mathcal K_-$$, each a Hilbert space for $$\pm\langle\cdot,\cdot\rangle$$ [[Bognar]](#ref-bognar). The *sign operator* $$\Theta$$ is $$+1$$ on $$\mathcal K_+$$ and $$-1$$ on $$\mathcal K_-$$, so that $$\langle x,y\rangle=(x,\Theta y)$$ with $$(\cdot,\cdot)$$ the Hilbert inner product. The *Krein adjoint* of an operator $$A$$ is $$A^\#=\Theta A^*\Theta$$, which satisfies $$\langle Ax,y\rangle=\langle x,A^\#y\rangle$$. A linear functional $$\omega$$ on a $$\#$$-algebra is *positive* if $$\omega(A^\#A)\ge0$$ for all $$A$$.

The simplest Krein algebra is $$\mathrm{Mat}_n(\mathbb C)$$ with $$\mathcal K=\mathbb C^p\oplus\mathbb C^q$$, $$n=p+q$$ and $$\Theta=\mathrm{diag}(1_p,-1_q)$$. If $$p,q\ge1$$ the algebra is not a $$C^*$$-algebra: for a null vector $$u$$, $$\langle u,u\rangle=0$$, the matrix $$A=\lvert u\rangle\langle v\rvert$$ has $$A^\#A=0$$ with $$A\neq0$$.

<div class="theorem" markdown="1">
**Theorem 1** (Finite Krein algebras have no state). Let $$p,q\ge1$$ and let $$\omega$$ be a linear functional on $$\mathrm{Mat}_{p+q}(\mathbb C)$$ with $$\omega(A^\#A)\ge0$$ for all $$A$$. Then $$\omega=0$$. In particular no such functional has $$\omega(1)=1$$.
</div>

Rank-one matrices suffice, because for them $$A^\#A$$ factorises into the Krein norm of one vector times a quantity depending on the other.

<div class="proof" markdown="1">
*Proof.* Write $$\omega=\mathrm{tr}(Y\,\cdot)$$ and take $$A=\lvert u\rangle\langle v\rvert$$, the operator $$x\mapsto u\,(v,x)$$. Then $$A^\#=\Theta\lvert v\rangle\langle u\rvert \Theta$$ and $$A^\#A=(u,\Theta u)\,\Theta\lvert v\rangle\langle v\rvert$$, where $$(u,\Theta u)=\langle u,u\rangle$$ is the Krein norm of $$u$$. Hence $$\omega(A^\#A)=\langle u,u\rangle\,(v,Y\Theta v)$$. The first factor is positive for $$u\in\mathcal K_+$$ and negative for $$u\in\mathcal K_-$$, so positivity requires $$(v,Y\Theta v)=0$$ for every $$v$$. A complex quadratic form that vanishes identically is zero, so $$Y\Theta=0$$ and $$Y=0$$. $$\square$$
</div>

Theorem 1 is the finite-dimensional model of the field-theory statement of the next section, where the vacuum functional fails to be positive on the algebra of every region.

## 3. The vacuum of a generalised free field with a ghost

Let $$\sigma$$ be a signed measure on $$[0,\infty)$$ with Jordan decomposition $$\sigma=\sigma_+-\sigma_-$$ into mutually singular positive measures, such that $$\lvert\sigma\rvert=\sigma_++\sigma_-$$ is tempered: $$\int\lvert\sigma\rvert(d\mu)(1+\mu)^{-N}<\infty$$ for some $$N$$. The *generalised free field with signed spectral measure* $$\sigma$$ is the field $$\phi$$ with $$\phi(f)^\#=\phi(\bar f)$$ and Wightman two-point function

$$
W_\sigma(x-y)=\int\sigma(d\mu)\int d\nu_\mu(p)\,e^{ip\cdot(x-y)},\tag{1}
$$

where $$d\nu_\mu(p)$$ is the Lorentz-invariant measure on the mass shell $$p^2=-\mu$$, $$p^0>0$$, in the mostly-plus signature. For $$\sigma=\delta_{m^2}$$ this is the free field of mass $$m$$. For $$\sigma=\delta_{m^2}-\delta_{M^2}$$, the Lee–Wick or Pauli–Villars pair, the two-point function is a difference of two Wightman functions and the Feynman propagator is $$1/(p^2+m^2)-1/(p^2+M^2)$$.

The state space is built as follows. The one-particle space is $$\mathcal H_1=L^2(\lvert\sigma\rvert(d\mu)\,d\nu_\mu)$$, with Hahn decomposition $$\mathcal H_1=\mathcal H_1^+\oplus\mathcal H_1^-$$ according to the sets where $$\sigma$$ is positive and negative. The Krein form on $$\mathcal H_1$$ is $$\langle h,h'\rangle=(h,\Theta h')$$ with $$\Theta=\pm1$$ on $$\mathcal H_1^\pm$$, and the state space is the Fock space over this Krein space, with vacuum $$\Omega$$. The vector $$\phi(f)\Omega$$ is the restriction of the Fourier transform $$\hat f$$ to the mass shells, with the sign convention for which it has positive energy. The vacuum functional is $$\omega_0(A)=\langle\Omega,A\Omega\rangle$$, and on a test function

$$
\omega_0\big(\phi(f)^\#\phi(f)\big)=\int\sigma(d\mu)\int d\nu_\mu(p)\,\lvert\hat f(p)\rvert^2=:Q(f).\tag{2}
$$

The algebra $$\mathfrak A(O)$$ of an open region $$O\subseteq\mathbb R^d$$ is the $$\#$$-algebra generated by $$\phi(f)$$ with $$f\in C_c^\infty(O)$$.

<div class="theorem" markdown="1">
**Theorem 2** (The Krein vacuum is not positive on any region). Let $$\sigma$$ be as above with $$\sigma_-((0,\infty))>0$$, and let $$O\subseteq\mathbb R^d$$ be any non-empty open set. Then $$Q(f)<0$$ for some $$f\in C_c^\infty(O)$$, so $$\omega_0$$ is not positive on $$\mathfrak A(O)$$. More precisely,

$$
\inf_{f\in C_c^\infty(O),\,f\neq0}\ \frac{Q(f)}{Q_{\rm abs}(f)}=-1,\qquad Q_{\rm abs}(f)=\int\lvert\sigma\rvert(d\mu)\int d\nu_\mu(p)\lvert\hat f(p)\rvert^2.\tag{3}
$$

</div>

$$Q$$ is the Krein form on the one-particle space. The vectors $$\phi(f)\Omega$$ with $$f$$ supported in a small region are dense in the Hilbert topology of that space, and an indefinite form is negative somewhere on every dense subspace. The density is the Reeh–Schlieder theorem; the rest is a continuity argument.

<div class="proof" markdown="1">
*Proof.* By the Reeh–Schlieder theorem [[RS61]](#ref-rs61) [[SW]](#ref-sw), applied to the generalised free field with the positive measure $$\lvert\sigma\rvert$$, which satisfies the Wightman axioms, the polynomial algebra of $$O$$ applied to the vacuum is dense in the Fock space. Only its one-particle consequence is used, so the difference between the Krein and the Hilbert Fock space is immaterial. The one-particle component of a product $$\phi(f_1)\cdots\phi(f_k)\Omega$$ is a linear combination of vectors $$\phi(f_j)\Omega$$ (the contractions give scalar coefficients), that is a vector $$\phi(f)\Omega$$ with $$f=\sum_j\lambda_jf_j$$. Hence the restrictions $$\hat f\vert_{\rm shells}$$ with $$f\in C_c^\infty(O)$$ are dense in $$\mathcal H_1$$ in the Hilbert norm. Choose a bounded Borel set $$\Lambda\subset(0,\infty)$$ with $$\sigma_+(\Lambda)=0$$ and $$0<\sigma_-(\Lambda)<\infty$$, and let $$\eta\in\mathcal H_1^-$$ equal $$1$$ on the part of the shells with $$\mu\in\Lambda$$ and $$\lvert p\rvert\le1$$, and $$0$$ elsewhere. It is non-zero and $$\langle \eta,\eta\rangle=-\lVert \eta\rVert^2<0$$. The Krein form is continuous in the Hilbert norm, so a sequence $$f_n$$ with $$\hat f_n\vert_{\rm shells}\to \eta$$ has $$Q(f_n)\to\langle \eta,\eta\rangle<0$$. The same argument with an arbitrary $$\eta\in\mathcal H_1^-$$ gives the infimum $$-1$$ in (3), because $$Q(f)/Q_{\rm abs}(f)=(\lVert\Pi_+\hat f\rVert^2-\lVert\Pi_-\hat f\rVert^2)/\lVert\hat f\rVert^2$$, with $$\Pi_\pm$$ the projections on $$\mathcal H_1^\pm$$. $$\square$$
</div>

The size of the region matters for how negative a given family of test functions gets, as the following example shows. For $$d=2$$ and $$\sigma=\delta_{m^2}-\delta_{M^2}$$ with $$m=0.1$$ and $$M=1$$ we minimise $$Q/Q_{\rm abs}$$ over the span of 30 test functions supported in a disc of radius $$R$$ (the bump $$\exp(-1/(1-r^2/R^2))$$ times monomials $$t^ax^b$$ of total degree at most $$3$$ times $$e^{i\varpi t}$$, $$\varpi\in\{0,\pm M\}$$), discarding the combinations whose $$Q_{\rm abs}$$ is below $$10^{-4}$$ of the largest eigenvalue. For $$R=0.25,\,0.5,\,1,\,2,\,4$$ the minimum is $$-0.017,\ -0.050,\ -0.22,\ -1.00,\ -1.00$$. The infimum $$-1$$ of (3) is reached in this basis once $$RM\gtrsim2$$; the smaller values at small radius belong to the finite basis and the cut, because (3) holds at every radius.

<div class="proposition" markdown="1">
**Proposition 3** (No positive state with positive energy). Let $$\sigma$$ be as in Theorem 2, and let $$\mathfrak A$$ be the $$\#$$-algebra generated by $$\phi(f)$$, $$f\in\mathcal S(\mathbb R^d)$$, with the commutation relations $$[\phi(f),\phi(g)]=\big(W_\sigma(f,g)-W_\sigma(g,f)\big)1$$, where $$W_\sigma(f,g)=\int dx\,dy\,f(x)g(y)W_\sigma(x-y)$$ and $$W_\sigma$$ is the function (1). There is no linear functional $$\omega$$ on $$\mathfrak A$$ with all of the following properties: it is positive, $$\omega(A^\#A)\ge0$$; its two-point function $$(f,g)\mapsto\omega(\phi(f)\phi(g))$$ is a tempered distribution; it is translation invariant; and the Fourier transform of its two-point function is supported in the closed forward cone $$\bar V_+$$ (positive energy).
</div>

<div class="proof" markdown="1">
*Proof.* By translation invariance the two-point function has the form $$w(x-y)$$ with $$w\in\mathcal S'$$. Positivity on $$\phi(f)^\#\phi(f)$$ says that $$w$$ is positive definite, so by the Bochner–Schwartz theorem $$\hat w$$ is a positive measure, supported in $$\bar V_+$$. The commutation relation gives $$w(x)-w(-x)=\int\sigma(d\mu)\int d\nu_\mu(p)\,(e^{ip\cdot x}-e^{-ip\cdot x})$$, that is $$\hat w-\mathcal R\hat w=m_\sigma-\mathcal Rm_\sigma$$ with $$m_\sigma=\int\sigma(d\mu)\,d\nu_\mu$$ and $$\mathcal R$$ the reflection $$p\mapsto-p$$. On the open forward cone $$V_+$$ the measures $$\mathcal R\hat w$$ and $$\mathcal Rm_\sigma$$ vanish, because they are supported in $$-\bar V_+$$, which meets $$V_+$$ in no point. Hence $$\hat w=m_\sigma$$ on $$V_+$$. The measure $$\hat w$$ is positive and the shells with $$\mu>0$$ lie in $$V_+$$ and are disjoint, so $$\sigma\ge0$$ on $$(0,\infty)$$, contradicting $$\sigma_-((0,\infty))>0$$. $$\square$$
</div>

Theorem 2 and Proposition 3 exclude one state: the Lorentz-invariant vacuum, which keeps the energy positive. They say nothing about other states, which the next section takes up.

## 4. Where a positive state remains

The three constructions of section 1 (the density matrix of a type $$I$$ factor, the faithful normal state of Tomita–Takesaki theory, and the trace obtained from a modular group in a crossed product) all start from a positive state of the algebra. For the Lorentz-invariant positive-energy vacuum of a Krein theory none is available, whatever algebra is built from it. There are two ways in which a positive state can still be present.

If the requirement of positive energy is dropped, positive states exist. The functional with two-point function $$w'(x-y)=\int\sigma_+(d\mu)\int d\nu_\mu\,e^{ip\cdot(x-y)}+\int\sigma_-(d\mu)\int d\nu_\mu\,e^{-ip\cdot(x-y)}$$, extended to a quasi-free state, is positive, because each term is the Wightman function of a positive-norm field (the second with the roles of creation and annihilation exchanged). It is Poincaré invariant and satisfies the commutation relations of $$\mathfrak A$$. Its Fourier transform has a part supported in the backward cone, so the ghost carries negative energy. By Proposition 3, positivity and positive energy cannot both hold for a translation-invariant functional with a tempered two-point function, and the vacuum of Theorem 2 keeps the second. The state with two-point function $$w'$$ is positive, so the constructions of section 1 can be applied to it (with a relative entropy where the algebra is of type $$III$$); its energy is unbounded below.

The second way is that the ghost is absent from the algebra that generates the region. In the fakeon prescription [[AP]](#ref-ap) [[Ans]](#ref-ans) the pole of the ghost is treated by the average of the two analytic continuations around it, and the ghost becomes a purely virtual particle, never an asymptotic state. In that formulation the physical Hilbert space is positive, only the Feynman propagator carries the signed weight (in loops), and the Wightman function of the physical algebra is positive. Lee–Wick formulations with an unstable ghost are argued in [[DM]](#ref-dm) to be similar in that the ghost does not appear as an external state. Theorem 2 does not apply to that algebra.

What fails in this reading is the passage from Euclidean correlators to states. A set of Euclidean correlation functions is *reflection positive* if their pairing with time-reflected test functions is positive, and the Osterwalder–Schrader theorem [[OS]](#ref-os) then reconstructs from them a positive Wightman function, of which the Euclidean functions are the analytic continuation. If the Euclidean two-point function with weight $$\sigma$$ were reflection positive, reconstruction would give a positive Wightman function whose continuation is the same function, namely $$W_\sigma$$; this contradicts Theorem 2. The Euclidean correlators of the signed weight therefore do not determine the Wightman functions of the physical states by this route. The *replica* entropy, which is computed from the Euclidean path integral on $$n$$-sheeted covers of the space and continued to $$n\to1$$, uses only Euclidean correlators. Theorem 2 shows that this quantity is not the entropy of the Krein vacuum on a region. Whether it equals the entropy of a state of the physical algebra is not decided by the theorems of this page. Which reading a given theory such as quadratic gravity uses is an input of the theory.

## 5. Finite trace and exact boost covariance

Both readings of section 4 lead, when gravity is included, to the type $$II$$ algebras of section 1. The result of this section involves no ghost: it constrains which algebras of that kind can carry an exact boost symmetry with positive null energy, and so applies to an entropy built from a positive state of a gravitational algebra, with or without a ghost.

In the algebraic description of a horizon the boost acts as an automorphism group $$\alpha_s$$ of the algebra, and the null energy $$P\ge0$$ is rescaled by it: translating along a null direction by $$\ell$$ and boosting by $$s$$ is translating by $$e^{s}\ell$$. The algebraic expression is $$\alpha_s(P)=e^{s}P$$ (Borchers' relation between the modular group and the translations, with the sign convention that gives $$e^{s}$$). For a state of a gravitational algebra we take this relation as a supposition, together with the supposition that $$P$$ is affiliated to the algebra, as is the case if the algebra contains the generator of the translations along the horizon.

<div class="proposition" markdown="1">
**Proposition 4** (Boost scaling of the null energy under a finite trace). Let $$\mathcal N$$ be a factor with a faithful normal tracial state $$\tau$$ (type $$I_n$$ or $$II_1$$), let $$(\alpha_s)_{s\in\mathbb R}$$ be a one-parameter group of $$*$$-automorphisms of $$\mathcal N$$, and let $$P\ge0$$ be a self-adjoint operator affiliated to $$\mathcal N$$ with $$\alpha_s(P)=e^{s}P$$ for all $$s$$. Then $$P=0$$. The same holds for any von Neumann algebra with a faithful normal tracial state $$\tau$$ that satisfies $$\tau\circ\alpha_s=\tau$$.
</div>

<div class="proof" markdown="1">
*Proof.* On a factor the finite tracial state is unique, so $$\tau\circ\alpha_s$$, which is again a faithful normal tracial state, equals $$\tau$$; for a general algebra this is the extra hypothesis. Let $$E_c$$ be the spectral projection of $$P$$ on $$(c,\infty)$$ and $$g(c)=\tau(E_c)$$. The relation $$\alpha_s(P)=e^{s}P$$ gives $$\alpha_s(E_c)=E_{e^{-s}c}$$, so $$g(c)=g(e^{-s}c)$$ for all $$s$$ and $$c>0$$, and $$g$$ is constant on $$(0,\infty)$$. Because $$P$$ is a self-adjoint operator, $$E_c\to0$$ strongly as $$c\to\infty$$, and by normality of $$\tau$$ the constant is $$0$$. Faithfulness gives $$E_c=0$$ for all $$c>0$$, so $$P=0$$. $$\square$$
</div>

In type $$I_n$$ the proposition is a matrix fact: an automorphism of $$\mathrm{Mat}_n$$ is inner and preserves the spectrum of $$P$$, which cannot be multiplied by $$e^{s}$$ unless it is $$\{0\}$$.

The hypothesis that $$\mathcal N$$ be a factor, or at least that $$\tau$$ be invariant, cannot be dropped. On the commutative algebra $$L^\infty(\mathbb R_+,e^{-x}dx)$$ with the finite faithful normal trace $$\tau(f)=\int f\,e^{-x}dx$$, the maps $$(\alpha_sf)(x)=f(e^{s}x)$$ are automorphisms and $$P=x$$ satisfies $$\alpha_s(P)=e^{s}P$$ with $$P\ne0$$; here $$\tau\circ\alpha_s\neq\tau$$.

With an infinite trace the relation holds exactly. In $$B(L^2(\mathbb R))$$ (type $$I_\infty$$) let $$P$$ be multiplication by $$e^{x}$$ and $$\alpha_s=\mathrm{Ad}\,U_{-s}$$ with $$(U_s\psi)(x)=\psi(x-s)$$; then $$\alpha_s(P)=e^{s}P$$ for all real $$s$$. In the crossed product of a type $$III$$ algebra by its modular group, of type $$II_\infty$$, the algebra is generated by the original algebra and $$h+\xi$$, with $$h$$ the modular Hamiltonian and $$\xi$$ the real variable of the crossed product; the dual action shifts $$\xi$$, and $$e^{h+\xi}$$, which is affiliated to the crossed product, is rescaled by it, up to the sign convention for the shift.

For a finite algebra the relation can hold on part of the spectrum. In $$\mathrm{Mat}_n$$ take $$P=\mathrm{diag}(e^{j\theta})_{j=0}^{n-1}$$ and the cyclic shift $$V$$ ($$Ve_j=e_{j+1}$$, $$Ve_{n-1}=e_0$$), and put $$\alpha=\mathrm{Ad}\,V^{-1}$$. Then $$\alpha(P)e_j=e^{\theta}Pe_j$$ for $$j<n-1$$ while $$\alpha(P)e_{n-1}=e_{n-1}$$, so the relation of the proposition holds for one step $$s=\theta$$ on every state except the one at the top of the spectrum, and

$$
\alpha(P)-e^{\theta}P=\big(1-e^{n\theta}\big)\,\lvert e_{n-1}\rangle\langle e_{n-1}\rvert.\tag{4}
$$

The trace of the left side is $$(1-e^{\theta})\,\mathrm{tr}\,P$$, because $$\alpha$$ preserves the trace, and this equals the trace of the right side. The whole defect therefore sits on one state, and its size is fixed by the trace. Covariance that is exact below a cutoff on the spectrum of $$P$$ is thus compatible with a finite trace, at the price of a defect at the cutoff.

The proposition applies to $$I_n$$ and $$II_1$$ alike; its hypothesis is the finiteness of the trace. For the algebras of section 1, the black-hole algebra at large $$N$$ is of type $$II_\infty$$ and has an infinite trace, whereas the de Sitter static-patch algebra with an observer is of type $$II_1$$, with a finite trace and the tracial state as maximum-entropy state. In type $$II_1$$ the entropy is defined up to an additive constant and $$e^S$$ is not an integer count; the range of $$\tau$$ on projections is $$\{k/n\}$$ for $$I_n$$ and $$[0,1]$$ for $$II_1$$. A $$II_1$$ algebra can arise as a limit of $$I_n$$ algebras with $$\ln n$$ subtracted: under the embedding $$\mathrm{Mat}_n\to\mathrm{Mat}_{2n}$$, $$X\mapsto X\otimes1$$, a state with density $$\rho$$ goes to the state with density $$\rho\otimes\tfrac12$$, whose entropy is $$S(\rho)+\ln2$$, so that $$S-\ln n$$ is unchanged. Whether a given gravitational algebra contains an operator with the properties assumed in Proposition 4 is a separate question.

## 6. The response to a ghost under different prescriptions

Theorem 2 and Proposition 3 show that the on-shell part of a ghost cannot be treated as an ordinary positive-energy state, and the treatments proposed in the literature differ in how that part enters. We ask which classical observables depend on the choice among the real Green functions of the mass shell defined below.

A ghost of mass $$M$$ in the graviton propagator contributes a term of opposite sign to a Yukawa potential. Its Green function equals $$1/(p^2+M^2)$$ away from the mass shell $$p^2=-M^2$$, which every Lorentz-invariant treatment agrees on; the treatments differ in how the shell is handled: as a retarded or advanced pole, as a principal value, or as a pole with a width of the wrong sign. We take the scalar problem $$(-\partial_t^2+\nabla^2-M^2)\chi=-j(t)\delta^3(\vec x)$$ for a point source with time profile $$j\in L^2(\mathbb R)$$, and the response at distance $$r$$. The real Green functions are $$G_a=(1-a)G_{\rm ret}+aG_{\rm adv}$$ with $$a\in\mathbb R$$: $$a=0$$ is the retarded response, in which the ghost radiates with the usual causal orientation, $$a=\tfrac12$$ is the time-symmetric (principal-value) response, and $$a=1$$ is the advanced response. The average continuation of the fakeon [[Ans]](#ref-ans) corresponds in the classical limit to $$a=\tfrac12$$. The retarded Green function of a ghost with a width of the wrong sign, in the limit of vanishing width, has its ghost contribution at negative times [[DM]](#ref-dm), which is the advanced response $$a=1$$ for the ghost term. We use these identifications as working assumptions; the proposition concerns the family and does not depend on them.

With the Fourier convention $$\chi(t)=\int dk_0\,e^{-ik_0t}\tilde\chi(k_0)$$ and $$\kappa^2=\vec k^2+M^2$$, the retarded and advanced functions are $$1/(\kappa^2-(k_0\pm i0)^2)$$ and their difference is the Pauli–Jordan function, supported on the mass shell:

$$
\tilde G_a(k_0,\vec k)=\mathrm{p.v.}\,\frac{1}{\kappa^2-k_0^2}+(1-2a)\,i\pi\,\mathrm{sgn}(k_0)\,\delta\big(\kappa^2-k_0^2\big).\tag{5}
$$

All members of the family share the principal-value part, and the parameter $$a$$ multiplies a term concentrated on the shell. For a source $$\hat j(k_0)$$ the response at distance $$r$$ is obtained by the radial Fourier transform. For $$\lvert k_0\rvert>M$$ the radial kernel of $$G_{\rm ret}-G_{\rm adv}$$ is $$2i\,\mathrm{sgn}(k_0)\sin(qr)/(4\pi r)$$ with $$q=\sqrt{k_0^2-M^2}$$, and for $$\lvert k_0\rvert<M$$ it vanishes. This kernel is the Fourier transform in $$t$$ of the time-domain function $$[\delta(t-r)-\delta(t+r)]/(4\pi r)-(M/4\pi)\,\mathrm{sgn}(t)\,\mathbf 1_{\{t^2>r^2\}}J_1(M\sqrt{t^2-r^2})/\sqrt{t^2-r^2}$$, with $$J_1$$ the Bessel function.

<div class="proposition" markdown="1">
**Proposition 5** (Prescription independence). For every $$a\in\mathbb R$$:

(i) The response at every frequency $$\lvert k_0\rvert<M$$, in particular at zero frequency, is independent of $$a$$. The static potential is $$e^{-Mr}/(4\pi r)$$ for all $$a$$.

(ii) The difference of two responses has Fourier support in $$\lvert k_0\rvert\ge M$$.

(iii) Let $$\chi_0$$ be the response of the massless field to the same source and $$\Delta_a=\chi_a-\chi_{\rm ret}$$. Then, at distance $$r$$,

$$
\frac{\lVert\Delta_a(\cdot,r)\rVert^2}{\lVert \chi_0(\cdot,r)\rVert^2}=4a^2\,\frac{\int_{\vert k_0\vert>M}\vert\hat j\vert^2\sin^2(qr)\,dk_0}{\int\vert\hat j\vert^2\,dk_0}\ \le\ 4a^2\,w(M),\qquad w(M)=\frac{\int_{\vert k_0\vert>M}\vert\hat j\vert^2dk_0}{\int\vert\hat j\vert^2dk_0}.\tag{6}
$$

</div>

<div class="proof" markdown="1">
*Proof.* (i) By (5) the members of the family differ by multiples of $$\delta(\kappa^2-k_0^2)$$, which vanishes when $$\lvert k_0\rvert<M$$, since $$\kappa^2\ge M^2$$. (ii) The term multiplying $$a$$ in (5) is supported on the shell, which requires $$\lvert k_0\rvert\ge M$$. (iii) $$\Delta_a=a(G_{\rm adv}-G_{\rm ret})*j$$ has radial kernel of modulus $$2a\vert\sin(qr)\vert/(4\pi r)$$ on $$\vert k_0\vert>M$$, while the massless retarded kernel is $$e^{ik_0r}/(4\pi r)$$, of modulus $$1/(4\pi r)$$ at every frequency. Parseval's theorem in the variable $$t$$ gives the ratio in (6), and $$\sin^2\le1$$ gives the bound. $$\square$$
</div>

For a Gaussian pulse $$j(t)=e^{-t^2/2T_0^2}$$ one has $$\vert\hat j\vert^2\propto e^{-k_0^2T_0^2}$$ and $$w(M)=\mathrm{erfc}(MT_0)$$, which is $$0.16$$ at $$MT_0=1$$ and $$2.2\times10^{-5}$$ at $$MT_0=3$$. For $$r$$ large compared with $$1/M$$ and a smooth $$\hat j$$, the factor $$\sin^2(qr)$$ in (6) oscillates rapidly and averages to $$\tfrac12$$, so that the ratio is close to $$2a^2w(M)$$, half of the bound.

The part of the response that depends on $$a$$ is $$a\,(G_{\rm adv}-G_{\rm ret})*j$$, a band-pass of the source above the ghost mass that is supported partly before the signal could arrive. It does not change the propagation speed. A Lorentz-invariant prescription changes the propagator only on the mass shell $$p^2=-M^2$$ of the ghost, so the pole of the massless graviton at $$p^2=0$$, and with it the dispersion relation $$k_0=\vert\vec k\vert$$ in flat space, is unchanged, provided the Lorentz and diffeomorphism Ward identities keep the self-energy transverse and a function of $$p^2$$ only.

The proposition is proved for the scalar equation. For the massive spin-2 ghost we assume that the tensor propagator is the scalar denominator multiplied by a numerator polynomial in $$k_0$$ and $$\vec k$$, and that the prescription enters only through the treatment of the pole of that denominator. Under this assumption the prescription-dependent terms are polynomials times $$\delta(\kappa^2-k_0^2)$$, which are again supported on the shell and vanish for $$\lvert k_0\rvert<M$$, so (i) and (ii) carry over. The bound (6) applies to the scalar model only.

## 7. A bound on the ghost mass from torsion-balance data

The static potential between two masses in quadratic gravity is [[Stelle78]](#ref-stelle78)

$$
V(r)=-\frac{Gm_1m_2}{r}\Big(1+\frac13e^{-m_0r}-\frac43e^{-m_2r}\Big),\tag{7}
$$

with a massive scalar of mass $$m_0$$ and a massive spin-2 ghost of mass $$m_2$$. By Proposition 5(i), and under the assumption stated there for spin 2, the term with $$m_2$$ has the same form for all Lorentz-invariant prescriptions of the ghost, so a bound on it does not depend on the prescription.

Torsion-balance tests of the inverse-square law [[Lee20]](#ref-lee20) report, with 95% confidence, that gravitational-strength Yukawa interactions ($$\lvert\beta\rvert=1$$, with $$\beta$$ the strength relative to Newtonian gravity) with ranges above $$38.6\ \mu$$m are excluded. A range of $$38.6\ \mu$$m corresponds to a mass $$\hbar c/\lambda=5.1$$ meV. If $$m_0\ge m_2$$, the combination of the two terms in (7) has magnitude at least $$\tfrac43e^{-m_2r}-\tfrac13e^{-m_0r}\ge e^{-m_2r}$$, that is at least that of a $$\lvert\beta\rvert=1$$ Yukawa of range $$1/m_2$$ at every $$r$$. For $$m_0\ge m_2$$ we therefore estimate that a spin-2 ghost with $$m_2<5.1$$ meV is excluded. The estimate treats the combination as a single Yukawa, which the published limit does not strictly license; it takes the limit at $$\lvert\beta\rvert=1$$ to apply to the term of negative sign in (7); and it gives no restriction for masses above $$5.1$$ meV or for $$m_0<m_2$$. It is an empirical estimate and not part of any theorem of this page.

## 8. Replica entropy, interacting and curved cases, and covariance with a cutoff

Extending Theorem 2 to the Krein vacuum of an interacting theory, or to a curved background such as a de Sitter static patch, needs a Reeh–Schlieder density statement there, which is open. A state defined through a Hilbert majorant, a norm dominating the Krein form, is a different notion from positivity under the Krein adjoint, and Theorems 1 and 2 and Proposition 3 do not apply to it.

If the ghost is absent from the physical algebra, whether the Euclidean replica entropy of the signed weight agrees with the entropy of a type $$I$$ truncation of the physical algebra can be tested on a lattice wedge for a Lee–Wick pair; this has not been done. Whether the fluctuation-dissipation relation for an accelerated detector holds at the Unruh temperature when the response is computed with the fakeon prescription, where the response function differs from the commutator by the on-shell band, is also open.

Equation (4) gives covariance that is exact below a cutoff in a finite algebra; whether the defect at the cutoff can be kept from feeding down to lower energies is undecided. The range of the trace on projections of a gravitational algebra, discrete as for $$I_n$$ or continuous as for $$II_1$$, is a non-perturbative input. Prescriptions outside the family (5), such as those that are not Lorentz invariant, those with a finite ghost width, and complex-conjugate pairs of ghost poles, for which unitarity violations are reported in the operator formalism [[KK]](#ref-kk), need a separate calculation of the response.

## References

1. <a id="ref-bognar"></a>Bognár, *Indefinite Inner Product Spaces*, Ergebnisse der Mathematik und ihrer Grenzgebiete 78, Springer (1974).
2. <a id="ref-lw"></a>Lee, Wick, “Negative metric and the unitarity of the S-matrix,” Nucl. Phys. B 9 (1969) 209.
3. <a id="ref-stelle77"></a>Stelle, “Renormalization of higher-derivative quantum gravity,” Phys. Rev. D 16 (1977) 953.
4. <a id="ref-stelle78"></a>Stelle, “Classical gravity with higher derivatives,” Gen. Relativ. Gravit. 9 (1978) 353.
5. <a id="ref-ap"></a>Anselmi, Piva, “A new formulation of Lee-Wick quantum field theory,” JHEP 06 (2017) 066, [arXiv:1703.04584](https://arxiv.org/abs/1703.04584).
6. <a id="ref-ans"></a>Anselmi, “Fakeons and Lee-Wick models,” JHEP 02 (2018) 141, [arXiv:1801.00915](https://arxiv.org/abs/1801.00915).
7. <a id="ref-dm"></a>Donoghue, Menezes, “Unitarity, stability and loops of unstable ghosts,” Phys. Rev. D 100, 105006 (2019), [arXiv:1908.02416](https://arxiv.org/abs/1908.02416).
8. <a id="ref-kk"></a>Kubo, Kugo, “Unitarity violation in field theories of Lee-Wick's complex ghost,” [arXiv:2308.09006](https://arxiv.org/abs/2308.09006).
9. <a id="ref-rs61"></a>Reeh, Schlieder, “Bemerkungen zur Unitäräquivalenz von Lorentzinvarianten Feldern,” Nuovo Cimento 22 (1961) 1051.
10. <a id="ref-sw"></a>Streater, Wightman, *PCT, Spin and Statistics, and All That*, Benjamin (1964).
11. <a id="ref-os"></a>Osterwalder, Schrader, “Axioms for Euclidean Green's functions,” Commun. Math. Phys. 31 (1973) 83.
12. <a id="ref-bw"></a>Bisognano, Wichmann, “On the duality condition for a Hermitian scalar field,” J. Math. Phys. 16 (1975) 985.
13. <a id="ref-witten"></a>Witten, “Gravity and the crossed product,” [arXiv:2112.12828](https://arxiv.org/abs/2112.12828).
14. <a id="ref-clpw"></a>Chandrasekaran, Longo, Penington, Witten, “An algebra of observables for de Sitter space,” [arXiv:2206.10780](https://arxiv.org/abs/2206.10780).
15. <a id="ref-cpw"></a>Chandrasekaran, Penington, Witten, “Large N algebras and generalized entropy,” [arXiv:2209.10454](https://arxiv.org/abs/2209.10454).
16. <a id="ref-lee20"></a>Lee, Adelberger, Cook, Fleischer, Heckel, “New test of the gravitational $$1/r^2$$ law at separations down to 52 $$\mu$$m,” [arXiv:2002.11761](https://arxiv.org/abs/2002.11761).
