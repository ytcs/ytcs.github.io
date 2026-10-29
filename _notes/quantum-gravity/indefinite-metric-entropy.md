---
title: Entropy of states in indefinite-metric theories
date: 2026-10-29
category: continuum
order: 1
description: If a ghost with negative norm is present in the Wightman function of the fields that generate a region's algebra, the Lorentz-invariant vacuum functional is not positive on the algebra of any open region, so no entropy defined from a state of that algebra is available. Exact boost covariance with positive energy also forces the energy to vanish when the trace is finite. The response to a ghost is independent of the Lorentz-invariant prescription for all static quantities.
status: Theorems proved in full for finite Krein algebras, for the vacuum of a generalised free field with a signed spectral measure, and for finite-trace boost covariance; the prescription-independence statement is proved for a scalar toy response; the Eot-Wash mass bound is an empirical input and not part of any theorem.
excludes: State-based entropies (density matrices, modular theory, crossed-product traces) of the Krein vacuum of a theory whose indefinite spectral weight is carried by the Wightman function of the fields generating the region's algebra; exact boost covariance with positive null energy inside an algebra with a finite trace.
---

The entropy of a region in quantum field theory is a function of a state on the algebra of observables of that region. Theories with ghosts, which include Lee–Wick models and quadratic gravity, assign a negative norm to some states. This page asks what happens to the entropy of a region in such a theory, and answers in four statements with the conditions under which each holds. (i) The algebra of all operators on a finite-dimensional space with an indefinite inner product, with the adjoint adapted to that product (the Krein adjoint), has no positive linear functional at all, so there is no density matrix (Theorem 1). (ii) In a generalised free field whose Källén–Lehmann spectral measure has a negative part, the vacuum functional, which is the Lorentz-invariant two-point function with positive energy, takes negative values on the algebra of every non-empty open region (Theorem 2), and no translation-invariant state with positive energy and a tempered two-point function is positive on the algebra of the whole space (Proposition 3). The entropy defined through a density matrix, a modular operator or a crossed-product trace has no state to start from. (iii) Independently of ghosts, an algebra with a finite trace cannot carry exact boost covariance with a positive energy operator inside it (Proposition 4); this separates algebras by whether the trace is finite and not by whether they count states or integrate a density. (iv) For a ghost of mass $$M$$ whose Green function equals the Feynman one off the mass shell, every static quantity is independent of how the shell is treated, and the part of the response that depends on the treatment is confined to frequencies above $$M$$ (Proposition 5). An empirical lower bound on the ghost mass from torsion-balance experiments is recorded separately in section 5.

The positivity statements are elementary and close to folklore. Theorem 1 is the finite-dimensional case of the absence of positive functionals on algebras with an indefinite involution; Reeh–Schlieder density [[RS61]](#ref-rs61) is the ingredient of Theorem 2. Spectral positivity is also what the positivity-dimension lemma of the neighbouring page [Positive spectral weight forbids dimensional reduction](/quantum-gravity/positivity-dimension-lemma/) uses from the opposite side: a negative spectral weight is needed to reduce the spectral dimension, and this page treats what a negative weight does to the state. Proposition 4 is a corollary of the uniqueness of the finite trace on a factor; its use here is to separate the two descriptions of horizon entropy that appear in the recent algebraic treatments of gravity [[Witten]](#ref-witten) [[CLPW]](#ref-clpw) [[CPW]](#ref-cpw). Proposition 5 packages standard facts about Green functions of the Klein–Gordon equation, with the prescriptions that have been proposed for ghosts in Lee–Wick and fakeon models [[LW]](#ref-lw) [[AP]](#ref-ap) [[Ans]](#ref-ans) [[DM]](#ref-dm). The place of this result among other approaches is mapped on [The landscape of quantum-gravity theories](/quantum-gravity/landscape/), and the entropy of induced gravity is the subject of [Can induced gravity fix Newton's constant?](/quantum-gravity/induced-gravity-number/).

## 1. What an entropy of a region requires

The observables of a region form an algebra. In the algebraic formulation this is a von Neumann algebra $$\mathcal M$$ of operators on a Hilbert space, and a state is a positive linear functional $$\omega$$ with $$\omega(1)=1$$; positivity means $$\omega(A^*A)\ge0$$ for all $$A$$. A *trace* on $$\mathcal M$$ is a positive functional $$\tau$$ with $$\tau(AB)=\tau(BA)$$. Whether a trace exists, and what it looks like, sorts the algebras that are factors, those with trivial centre, into three types.

A factor of *type I* is the algebra $$B(\mathcal H)$$ of all bounded operators on a Hilbert space; it is $$I_n$$ if $$\dim\mathcal H=n<\infty$$ and $$I_\infty$$ otherwise. It has minimal projections, a trace $$\mathrm{Tr}$$ that takes the value $$1$$ on them, and every state is a density matrix: $$\omega=\mathrm{Tr}(\rho\,\cdot)$$. The entropy $$-\mathrm{Tr}\,\rho\ln\rho$$ is absolute, bounded by $$\ln n$$ in $$I_n$$, and $$e^S$$ counts states in the sense that the maximally mixed state of a subspace of dimension $$m$$ has $$S=\ln m$$. A factor of *type II* has a trace $$\tau$$ but no minimal projections; it is $$II_1$$ if $$\tau(1)<\infty$$ and $$II_\infty$$ otherwise. A state is still $$\omega=\tau(\rho\,\cdot)$$, with $$\rho$$ affiliated to $$\mathcal M$$, and the entropy $$-\tau(\rho\ln\rho)$$ is defined, but the trace is fixed only up to a positive factor $$c$$, which shifts the entropy by $$\ln c$$. In a $$II_1$$ factor with $$\tau(1)=1$$ the entropy is at most $$0$$, with the maximum at the tracial state; only differences of entropies are meaningful. A factor of *type III* has no non-zero trace, and an entropy of a state has to be defined by a regularisation or replaced by a relative entropy. In relativistic quantum field theory the algebra of a wedge is of type $$III_1$$ (see the discussion in [[Witten]](#ref-witten)), which is why the entropy of a region is infinite and regulator-dependent. Including gravitational dressing turns the algebra into one of type II: the crossed product of a type $$III_1$$ algebra by its modular group, of type $$II_\infty$$, for a black hole at large $$N$$ in AdS/CFT [[Witten]](#ref-witten) [[CPW]](#ref-cpw), and of type $$II_1$$ for the static patch of de Sitter space with an observer [[CLPW]](#ref-clpw). In all cases the construction starts from a faithful normal positive state: Tomita–Takesaki theory attaches a modular operator $$\Delta$$ to a faithful normal state on a von Neumann algebra, and for the vacuum and a wedge the Bisognano–Wichmann theorem [[BW]](#ref-bw) identifies $$\Delta^{it}$$ with the boost, which is where the temperature $$1/2\pi$$ of a horizon comes from.

So three things are needed: an algebra, a positive state on it, and a trace or a modular structure derived from the state. The next section shows that the second fails for the Lorentz-invariant vacuum of a theory with an indefinite spectral weight.

## 2. Krein spaces and the vacuum of a ghost theory

*The picture.* A harmonic oscillator whose creation and annihilation operators obey $$[a,a^\dagger]=-1$$ has a one-particle state $$a^\dagger\lvert0\rangle$$ of norm $$\langle0\rvert aa^\dagger\lvert0\rangle=-1$$. The Hilbert-space probability interpretation fails, and the space of states carries an indefinite form instead. Such a mode is the ghost of a Pauli–Villars regulator, a Lee–Wick partner [[LW]](#ref-lw), or the massive spin-2 mode of quadratic gravity [[Stelle77]](#ref-stelle77).

*Definitions.* A *Krein space* is a complex vector space $$\mathcal K$$ with a non-degenerate Hermitian form $$\langle\cdot,\cdot\rangle$$ that splits as $$\mathcal K=\mathcal K_+\oplus\mathcal K_-$$ with $$\langle\cdot,\cdot\rangle$$ positive definite on $$\mathcal K_+$$ and negative definite on $$\mathcal K_-$$, each being a Hilbert space for $$\pm\langle\cdot,\cdot\rangle$$ [[Bognar]](#ref-bognar). The *sign operator* $$J$$ is $$+1$$ on $$\mathcal K_+$$ and $$-1$$ on $$\mathcal K_-$$, so $$\langle x,y\rangle=(x,Jy)$$ with $$(\cdot,\cdot)$$ the Hilbert inner product. The *Krein adjoint* of an operator $$A$$ is $$A^\#=JA^*J$$, which satisfies $$\langle Ax,y\rangle=\langle x,A^\#y\rangle$$. A linear functional $$\omega$$ on a $$\#$$-algebra is *positive* if $$\omega(A^\#A)\ge0$$ for all $$A$$.

The simplest Krein algebra is $$M_n(\mathbb C)$$ with $$\mathcal K=\mathbb C^p\oplus\mathbb C^q$$, $$n=p+q$$, $$J=\mathrm{diag}(1_p,-1_q)$$ and $$A^\#=JA^*J$$. If $$p,q\ge1$$ the algebra is not a $$C^*$$-algebra: for a null vector $$u$$, $$\langle u,u\rangle=0$$, the matrix $$A=\lvert u\rangle\langle v\rvert$$ satisfies $$A^\#A=0$$ with $$A\neq0$$.

<div class="theorem" markdown="1">
**Theorem 1** (Finite Krein algebras have no positive functionals). Let $$p,q\ge1$$ and let $$\omega$$ be a linear functional on $$M_{p+q}(\mathbb C)$$ with $$\omega(A^\#A)\ge0$$ for all $$A$$. Then $$\omega=0$$.
</div>

The idea is to test positivity on rank-one matrices, for which $$A^\#A$$ factorises into the Krein norm of one vector times a quantity depending on the other.

<div class="proof" markdown="1">
*Proof.* Write $$\omega=\mathrm{tr}(\sigma\,\cdot)$$ and take $$A=\lvert u\rangle\langle v\rvert$$. Here $$\lvert u\rangle\langle v\rvert$$ is the operator $$x\mapsto u\,(v,x)$$ in the Hilbert pairing $$(\cdot,\cdot)$$. Then $$A^\#=J\lvert v\rangle\langle u\rvert J$$ and $$A^\#A=(u,Ju)\,J\lvert v\rangle\langle v\rvert$$, where $$(u,Ju)=\langle u,u\rangle$$ is the Krein norm of $$u$$. Hence $$\omega(A^\#A)=\langle u,u\rangle\,(v,\sigma Jv)$$. The first factor is positive for $$u\in\mathcal K_+$$ and negative for $$u\in\mathcal K_-$$, so positivity requires $$(v,\sigma Jv)=0$$ for every $$v$$. A complex quadratic form that vanishes identically is zero, so $$\sigma J=0$$ and $$\sigma=0$$. $$\square$$
</div>

One consequence concerns counting. The signed dimension $$\mathrm{tr}\,J=p-q$$ is invariant under $$J$$-unitary changes of basis, and for a Lee–Wick pair, one normal mode and one ghost per mode, $$p=q$$ and it vanishes. A count of states read as the logarithm of a dimension therefore gives either $$\ln n$$, which is not the entropy of any state since there is none, or $$\ln(p-q)=-\infty$$.

*The field-theory setting.* Let $$\rho$$ be a signed measure on $$[0,\infty)$$ with Jordan decomposition $$\rho=\rho_+-\rho_-$$ into mutually singular positive measures, such that $$\lvert\rho\rvert=\rho_++\rho_-$$ is tempered: $$\int\lvert\rho\rvert(d\mu)(1+\mu)^{-N}<\infty$$ for some $$N$$. A *generalised free field with signed spectral measure* $$\rho$$ is the field $$\phi$$ with $$\#$$-structure $$\phi(f)^\#=\phi(\bar f)$$ and Wightman two-point function

$$
W_\rho(x-y)=\int\rho(d\mu)\int d\Omega_\mu(p)\,e^{ip\cdot(x-y)},\tag{1}
$$

where $$d\Omega_\mu(p)$$ is the Lorentz-invariant measure on the mass shell $$p^2=-\mu$$, $$p^0>0$$, and the mostly-plus signature is used. Its Krein space is built as follows. The one-particle space is $$H_1=L^2(\lvert\rho\rvert(d\mu)\,d\Omega_\mu)$$, the Hilbert space for the positive measure $$\lvert\rho\rvert$$, with Hahn decomposition $$H_1=H_1^+\oplus H_1^-$$ according to the sets where $$\rho$$ is positive and negative. The Krein form on $$H_1$$ is $$\langle h,h'\rangle=(h,Jh')$$ with $$J=\pm1$$ on $$H_1^\pm$$, and the whole state space is the Fock space over this Krein space, with vacuum $$\Omega$$. The vector $$\phi(f)\Omega$$ is the restriction to the mass shells of the Fourier transform of $$f$$, with the sign convention for which it has positive energy. The vacuum functional is $$\omega_0(A)=\langle\Omega,A\Omega\rangle$$, and on a test function

$$
\omega_0\big(\phi(f)^\#\phi(f)\big)=\int\rho(d\mu)\int d\Omega_\mu(p)\,\lvert\hat f(p)\rvert^2=:Q(f).\tag{2}
$$

The algebra $$\mathfrak A(O)$$ of an open region $$O\subseteq\mathbb R^d$$ is the $$\#$$-algebra generated by $$\phi(f)$$ with $$f\in C_c^\infty(O)$$. For $$\rho=\delta_{m^2}$$ everything reduces to the free field of mass $$m$$. For $$\rho=\delta_{m^2}-\delta_{M^2}$$, the Lee–Wick or Pauli–Villars pair, the two-point function is the difference of two Wightman functions and the Feynman propagator is $$1/(p^2+m^2)-1/(p^2+M^2)$$.

<div class="theorem" markdown="1">
**Theorem 2** (The Krein vacuum is not positive on any region). Let $$\rho$$ be as above with $$\rho_-((0,\infty))>0$$, and let $$O\subseteq\mathbb R^d$$ be any non-empty open set. Then $$Q(f)<0$$ for some $$f\in C_c^\infty(O)$$; that is, $$\omega_0$$ is not positive on $$\mathfrak A(O)$$. More precisely,

$$
\inf_{f\in C_c^\infty(O),\,f\neq0}\ \frac{Q(f)}{Q_{\rm abs}(f)}=-1,\qquad Q_{\rm abs}(f)=\int\lvert\rho\rvert(d\mu)\int d\Omega_\mu(p)\lvert\hat f(p)\rvert^2.\tag{3}
$$

</div>

The idea is that $$Q$$ is the Krein form on the one-particle space, that the vectors $$\phi(f)\Omega$$ with $$f$$ supported in a small region are dense in the Hilbert topology of that space, and that an indefinite form is negative somewhere on every dense subspace. The density is the Reeh–Schlieder theorem; the rest is a continuity argument.

<div class="proof" markdown="1">
*Proof.* By the Reeh–Schlieder theorem [[RS61]](#ref-rs61) [[SW]](#ref-sw) for the generalised free field with the positive measure $$\lvert\rho\rvert$$, which satisfies the Wightman axioms, the polynomial algebra of $$O$$ applied to the vacuum is dense in the Fock space. The one-particle component of a product $$\phi(f_1)\cdots\phi(f_k)\Omega$$ is a sum of terms of the form $$c\,\phi(f_j)\Omega$$ with scalar $$c$$ (the contractions), so the vectors $$\hat f\vert_{\rm shells}$$, $$f\in C_c^\infty(O)$$, are dense in $$H_1$$ in the Hilbert norm. Choose a bounded Borel set $$A\subset(0,\infty)$$ with $$\rho_+(A)=0$$ and $$0<\rho_-(A)<\infty$$, and let $$h\in H_1^-$$ be the function equal to $$1$$ on the part of the shells with $$\mu\in A$$ and $$\lvert p\rvert\le1$$, and $$0$$ elsewhere. It is non-zero and $$\langle h,h\rangle=-\lVert h\rVert^2<0$$. The form $$\langle\cdot,\cdot\rangle$$ is continuous in the Hilbert norm, so a sequence $$f_n$$ with $$\hat f_n\vert_{\rm shells}\to h$$ has $$Q(f_n)\to\langle h,h\rangle<0$$. The same argument with $$h\in H_1^-$$ arbitrary gives the infimum $$-1$$ in (3), because $$Q(f)/Q_{\rm abs}(f)=(\lVert P_+\hat f\rVert^2-\lVert P_-\hat f\rVert^2)/\lVert\hat f\rVert^2$$, with $$P_\pm$$ the projections on $$H_1^\pm$$. $$\square$$
</div>

*Check.* For $$d=2$$ take the pair $$\rho=\delta_{m^2}-\delta_{M^2}$$ with $$m=0.1$$ and $$M=1$$. We computed the minimum of $$Q/Q_{\rm abs}$$ over a 30-dimensional space of test functions on a disc of radius $$R$$ centred at the origin, spanned by the bump function $$\exp(-1/(1-r^2/R^2))$$ times the monomials $$t^ax^b$$ of total degree at most $$3$$ times $$e^{i\nu t}$$ with $$\nu\in\{0,M,-M\}$$, by solving the generalised eigenvalue problem for the Gram matrices of $$Q$$ and $$Q_{\rm abs}$$ after discarding directions with $$Q_{\rm abs}$$ below $$10^{-4}$$ of its maximum. For $$R=0.25,0.5,1,2,4$$ the minimum is $$-0.031,-0.082,-0.17,-0.99,-1.00$$. It is negative at every radius and approaches the infimum $$-1$$ of (3) once $$RM\gtrsim1$$; for small radii the finite basis only reaches a small negative value. The values depend on the cut and the basis; the sign does not.

The vacuum is not positive on the algebra of any region, however small. It remains to ask whether some other state could take its place.

<div class="proposition" markdown="1">
**Proposition 3** (No positive state with positive energy). Let $$\rho$$ be as in Theorem 2, and let $$\mathfrak A$$ be the $$\#$$-algebra generated by $$\phi(f)$$, $$f\in\mathcal S(\mathbb R^d)$$, with the canonical commutation relations $$[\phi(f),\phi(g)]=\big(W_\rho(f,g)-W_\rho(g,f)\big)1$$, where $$W_\rho(f,g)=\int dx\,dy\,f(x)g(y)W_\rho(x-y)$$ and $$W_\rho$$ is the function (1). There is no linear functional $$\omega$$ on $$\mathfrak A$$ with all of the following properties: it is positive, $$\omega(A^\#A)\ge0$$; its two-point function $$(f,g)\mapsto\omega(\phi(f)\phi(g))$$ is a tempered distribution; it is translation invariant; and the Fourier transform of its two-point function is supported in the closed forward cone $$\bar V_+$$ (positive energy).
</div>

<div class="proof" markdown="1">
*Proof.* By translation invariance the two-point function has the form $$w(x-y)$$ with $$w\in\mathcal S'$$. Positivity on $$\phi(f)^\#\phi(f)$$ says that $$w$$ is positive definite, so by the Bochner–Schwartz theorem $$\hat w$$ is a positive measure, supported in $$\bar V_+$$. The commutation relation gives $$w(x)-w(-x)=\int\rho(d\mu)\int d\Omega_\mu(p)\,(e^{ip\cdot x}-e^{-ip\cdot x})$$, that is $$\hat w-R\hat w=m_\rho-Rm_\rho$$ with $$m_\rho=\int\rho(d\mu)\,d\Omega_\mu$$ and $$R$$ the reflection $$p\mapsto-p$$. On the open forward cone $$V_+$$ the measures $$R\hat w$$ and $$Rm_\rho$$ vanish, because they are supported in $$-\bar V_+$$, which meets $$V_+$$ in no point. Hence $$\hat w=m_\rho$$ on $$V_+$$. The measure $$\hat w$$ is positive and the shells with $$\mu>0$$ lie in $$V_+$$ and are disjoint, so $$\rho\ge0$$ on $$(0,\infty)$$, contradicting $$\rho_-((0,\infty))>0$$. $$\square$$
</div>

If the requirement of positive energy is dropped, positive states exist. The functional with two-point function $$w'(x-y)=\int\rho_+(d\mu)\int d\Omega_\mu\,e^{ip\cdot(x-y)}+\int\rho_-(d\mu)\int d\Omega_\mu\,e^{-ip\cdot(x-y)}$$, extended to a quasi-free state, is positive, because each term is the Wightman function of a positive-norm field (the second with the roles of creation and annihilation exchanged), it is Poincaré invariant, and it satisfies the commutation relations of $$\mathfrak A$$. Its Fourier transform has a part supported in the backward cone, so the ghost carries negative energy. Positivity and positive energy cannot both hold, and the vacuum of Theorem 2 is the one that keeps positive energy.

*Consequences for entropy.* The three constructions of section 1 start from a positive state on the algebra: the density matrix of a type I factor, the faithful normal state of Tomita–Takesaki theory, and the trace obtained from a modular group in a crossed product. For the Lorentz-invariant positive-energy vacuum of a Krein theory none is available. On the algebra of all operators of a finite-dimensional Krein space with $$q\ge1$$ there is no positive functional at all, and in the field theory the vacuum functional is negative on the algebra of every non-empty open region, whatever the type of the von Neumann algebra that might be generated from it. A positive state does exist on the abstract algebra if the ghost has negative energy, and an entropy can be defined from it; the vacuum is then not an energy minimum.

*The positive-quotient reading.* In the fakeon and Lee–Wick formulations [[AP]](#ref-ap) [[Ans]](#ref-ans) [[DM]](#ref-dm) the physical Hilbert space is positive: the ghost is not an asymptotic state, and only the Feynman propagator, in loops, carries the signed weight. The Wightman function of the physical algebra is then positive and Theorem 2 does not apply to it. What fails instead is the Euclidean reconstruction. If the Euclidean two-point function with weight $$\rho$$ were reflection positive, the Osterwalder–Schrader reconstruction theorem [[OS]](#ref-os) would produce a positive Wightman function whose analytic continuation is the same function, namely $$W_\rho$$; this contradicts Theorem 2. So the Euclidean correlators of the signed weight do not determine the Wightman functions of the physical states by the usual route, and the finiteness of the Euclidean (replica) area term, which is a Pauli–Villars-regulated quantity, need not be the entropy of the positive state. Which reading a given theory such as quadratic gravity uses is an input of the theory.

## 3. Finite trace and exact boost covariance

*The picture.* In the algebraic description of a horizon the boost acts as an automorphism group $$\alpha_s$$ of the algebra, and the null energy $$H\ge0$$ is rescaled by the boost: translating along a null direction by $$a$$ and boosting by $$s$$ gives translating by $$e^{s}a$$. The algebraic expression is $$\alpha_s(H)=e^{s}H$$ (Borchers' relation between the modular group and the translations, with the sign convention that makes it $$e^{s}$$). Suppose $$H$$ is affiliated to the algebra, which is the case if the algebra contains the generator of the translations along the horizon. If the algebra has a finite trace, the relation cannot hold unless $$H=0$$.

<div class="proposition" markdown="1">
**Proposition 4** (A finite trace forbids a scaling energy). Let $$\mathcal M$$ be a factor with a faithful normal tracial state $$\tau$$ (type $$I_n$$ or $$II_1$$), let $$(\alpha_s)_{s\in\mathbb R}$$ be a one-parameter group of $$*$$-automorphisms of $$\mathcal M$$, and let $$H\ge0$$ be a self-adjoint operator affiliated to $$\mathcal M$$ with $$\alpha_s(H)=e^{s}H$$ for all $$s$$. Then $$H=0$$.
</div>

<div class="proof" markdown="1">
*Proof.* A factor with a faithful normal tracial state has only one, so $$\tau\circ\alpha_s=\tau$$ for each $$s$$. Let $$E_a$$ be the spectral projection of $$H$$ on $$(a,\infty)$$ and $$f(a)=\tau(E_a)$$. The relation $$\alpha_s(H)=e^{s}H$$ gives $$\alpha_s(E_a)=E_{e^{-s}a}$$, so $$f(a)=f(e^{-s}a)$$ for all $$s$$ and $$a>0$$, and $$f$$ is constant on $$(0,\infty)$$. Because $$H$$ is a self-adjoint operator, $$E_a\to0$$ strongly as $$a\to\infty$$, and by normality of $$\tau$$ the constant is $$0$$. Faithfulness of $$\tau$$ gives $$E_a=0$$ for all $$a>0$$, so $$H=0$$. $$\square$$
</div>

The result is the observation that a finite trace is preserved by every automorphism, while the relation asks for a trace scaled by $$e^{-s}$$ on the spectral projections of $$H$$. In type $$I_n$$ it is a matrix fact: an automorphism of $$M_n$$ is inner and preserves the spectrum of $$H$$, which cannot be multiplied by $$e^{s}$$ unless it is $$\{0\}$$.

With an infinite trace the relation can hold exactly. In $$B(L^2(\mathbb R))$$ (type $$I_\infty$$) let $$H$$ be multiplication by $$e^{x}$$ and $$\alpha_s=\mathrm{Ad}\,T_{-s}$$ with $$(T_s\psi)(x)=\psi(x-s)$$; then $$\alpha_s(H)=e^{s}H$$ for all real $$s$$. In the crossed product of a type III algebra by its modular group, a type $$II_\infty$$ algebra, the dual action shifts the generator $$Q$$ of the crossed-product variable and $$H=e^{Q}$$ has the same property.

*Windowed covariance.* In $$M_n$$ with $$H=\mathrm{diag}(e^{j\delta})_{j=0}^{n-1}$$ and the cyclic shift $$S$$ ($$Se_j=e_{j+1}$$, $$Se_{n-1}=e_0$$), put $$\alpha=\mathrm{Ad}\,S^{-1}$$. Then $$\alpha(H)e_j=e^{\delta}He_j$$ for $$j<n-1$$ while $$\alpha(H)e_{n-1}=e_{n-1}$$, so the relation of the proposition holds for one step $$s=\delta$$ on every state except the one at the top of the spectrum, and

$$
\alpha(H)-e^{\delta}H=\big(1-e^{n\delta}\big)\,\lvert e_{n-1}\rangle\langle e_{n-1}\rvert.\tag{4}
$$

The trace of the left side is $$(1-e^{\delta})\,\mathrm{tr}\,H$$, because $$\alpha$$ preserves the trace, and this equals the trace of the right side. The whole defect therefore sits on one state, and its size is fixed by the trace. Covariance that is exact below a cutoff on the spectrum of $$H$$ is thus compatible with a finite trace, at the price of a defect at the cutoff; whether such a defect feeds down to energies below the cutoff in a given model has not been examined here.

*Reading.* The proposition applies to $$I_n$$ and $$II_1$$ alike, so the distinction between a count of states and a trace does not decide whether exact covariance is possible; finiteness of the trace does. The algebras obtained by including gravitational dressing are of type $$II_\infty$$ for a black hole in the large-$$N$$ limit of AdS/CFT [[Witten]](#ref-witten) [[CPW]](#ref-cpw), where the trace is infinite, and of type $$II_1$$ for the de Sitter static patch with an observer [[CLPW]](#ref-clpw), where the trace is finite and the maximum-entropy state is the tracial state. In type $$II_1$$ the entropy is defined up to an additive constant and $$e^S$$ is not an integer count. The range of $$\tau$$ on projections is $$\{k/n\}$$ for $$I_n$$ and $$[0,1]$$ for $$II_1$$. A $$II_1$$ algebra can arise as a limit of $$I_n$$ algebras with the logarithm of $$n$$ subtracted: under the embedding $$M_n\to M_{2n}$$, $$x\mapsto x\otimes1$$, a state with density $$\rho$$ is carried to the state with density $$\rho\otimes\tfrac12$$, whose entropy is $$S(\rho)+\ln2$$, so that $$S-\ln n$$ does not change. Whether a given gravitational algebra contains an operator with the properties assumed in Proposition 4 is a separate question.

## 4. The response to a ghost under different prescriptions

*The picture.* A ghost of mass $$M$$ in the graviton propagator contributes a term with the opposite sign to a Yukawa potential. Its Green function equals $$1/(p^2+M^2)$$ away from the mass shell $$p^2=-M^2$$, which every Lorentz-invariant treatment agrees on; the treatments differ in how the shell is handled: as a retarded or advanced pole, as a principal value, as a pole with a width of the wrong sign. We ask which observables depend on the choice.

Consider the scalar toy problem $$(-\partial_t^2+\nabla^2-M^2)\chi=-J(t)\delta^3(\vec x)$$ for a point source with time profile $$J\in L^2(\mathbb R)$$, and the response at distance $$r$$. The real Green functions are $$G_a=(1-a)G_{\rm ret}+aG_{\rm adv}$$ with $$a\in\mathbb R$$: $$a=0$$ is the retarded response in which the ghost radiates with the usual causal orientation, $$a=\tfrac12$$ is the time-symmetric (principal-value) response, and $$a=1$$ is the advanced response. The classical limits of prescriptions proposed in the literature are members of this family: the average continuation of the fakeon [[Ans]](#ref-ans) corresponds to $$a=\tfrac12$$ and the Lee–Wick contour with a ghost width tending to zero [[DM]](#ref-dm) to $$a=1$$. We use these identifications as working assumptions from the literature; the proposition concerns the family and does not depend on them. With the Fourier convention $$\chi(t)=\int d\omega\,e^{-i\omega t}\tilde\chi(\omega)$$ and $$\kappa^2=\vec k^2+M^2$$, the retarded and advanced functions are $$1/(\kappa^2-(\omega\pm i0)^2)$$, and their difference is the Pauli–Jordan function, supported on the mass shell:

$$
\tilde G_a(\omega,\vec k)=\mathrm P\,\frac{1}{\kappa^2-\omega^2}+(1-2a)\,i\pi\,\mathrm{sgn}(\omega)\,\delta\big(\kappa^2-\omega^2\big).\tag{5}
$$

Equation (5) says that all the members of the family share the principal-value part, and that the parameter $$a$$ multiplies a term concentrated on the shell. For a source $$\hat J(\omega)$$ the response at distance $$r$$ is obtained by the radial Fourier transform. For $$\vert\omega\vert>M$$ the radial kernel of $$G_{\rm ret}-G_{\rm adv}$$ is $$2i\,\mathrm{sgn}(\omega)\sin(kr)/(4\pi r)$$ with $$k=\sqrt{\omega^2-M^2}$$, and for $$\vert\omega\vert<M$$ it vanishes.

<div class="proposition" markdown="1">
**Proposition 5** (Prescription independence). For every $$a\in\mathbb R$$:

(i) The response at zero frequency, and at any imaginary frequency $$\omega=i\gamma$$, is independent of $$a$$. In particular the static potential is $$e^{-Mr}/(4\pi r)$$ for all $$a$$.

(ii) The difference of two responses has Fourier support in $$\vert\omega\vert\ge M$$.

(iii) Let $$h_0$$ be the response of the massless field to the same source and $$\Delta_a=\chi_a-\chi_{\rm ret}$$. Then, at distance $$r$$,

$$
\frac{\lVert\Delta_a(\cdot,r)\rVert^2}{\lVert h_0(\cdot,r)\rVert^2}=4a^2\,\frac{\int_{\vert\omega\vert>M}\vert\hat J\vert^2\sin^2(kr)\,d\omega}{\int\vert\hat J\vert^2\,d\omega}\ \le\ 4a^2\,w(M),\qquad w(M)=\frac{\int_{\vert\omega\vert>M}\vert\hat J\vert^2d\omega}{\int\vert\hat J\vert^2d\omega}.\tag{6}
$$

</div>

<div class="proof" markdown="1">
*Proof.* (i) By (5) the members of the family differ by multiples of $$\delta(\kappa^2-\omega^2)$$, which vanishes when $$\vert\omega\vert<M$$ and for imaginary $$\omega$$. (ii) is the statement that the term multiplying $$a$$ in (5) is supported on the shell, which requires $$\vert\omega\vert\ge M$$. (iii) $$\Delta_a=a(G_{\rm adv}-G_{\rm ret})*J$$ has radial kernel of modulus $$2a\vert\sin(kr)\vert/(4\pi r)$$ on $$\vert\omega\vert>M$$, while the massless retarded kernel is $$e^{i\omega r}/(4\pi r)$$, of modulus $$1/(4\pi r)$$ at every frequency. Parseval's theorem in the variable $$t$$ gives the ratio in (6), and $$\sin^2\le1$$ gives the bound. $$\square$$
</div>

*Check.* For a Gaussian pulse $$J(t)=e^{-t^2/2\tau^2}$$, $$\vert\hat J\vert^2\propto e^{-\omega^2\tau^2}$$ and $$w(M)=\mathrm{erfc}(M\tau)$$. At $$r=60/M$$ we evaluated the ratio in (6) by numerical quadrature:

| $$M\tau$$ | $$w(M)$$ | ratio, $$a=\tfrac12$$ | ratio, $$a=1$$ | bound $$4a^2w$$ for $$a=1$$ |
|---|---|---|---|---|
| 0.5 | 0.4795 | 0.2398 | 0.9591 | 1.918 |
| 1 | 0.1573 | 0.07866 | 0.3147 | 0.6292 |
| 2 | 0.004678 | 0.002340 | 0.009361 | 0.01871 |
| 3 | 2.209e-5 | 1.106e-5 | 4.424e-5 | 8.84e-5 |

The ratio equals $$2a^2w(M)$$ to four digits, because $$\sin^2(kr)$$ averages to $$\tfrac12$$ over the rapid oscillation at this $$r$$, so the bound is saturated up to this factor of $$2$$. As an independent check of the kernel we evaluated the Fourier transform of the time-domain function $$G_{\rm ret}-G_{\rm adv}$$, namely $$[\delta(t-r)-\delta(t+r)]/(4\pi r)-(M/4\pi)\,\mathrm{sgn}(t)\,\theta(t^2-r^2)J_1(M\sqrt{t^2-r^2})/\sqrt{t^2-r^2}$$ at $$r=60/M$$, on a grid with step $$0.02/M$$: at $$\omega=0.3M,0.6M,0.9M$$ the imaginary part is smaller than $$3\times10^{-5}$$ against $$2\times10^{-3}$$ for the delta term alone, and at $$\omega=1.5M,2M$$ it equals $$2i\sin(kr)/(4\pi r)$$ to $$2\%$$, with the residual consistent with the first-order error of the Riemann sum across the jump of the kernel at $$t=\pm r$$.

Two further statements follow from Lorentz invariance and are derivations, not computations. A Lorentz-invariant prescription changes the propagator only on the mass shell $$p^2=-M^2$$ of the ghost, so the pole of the massless graviton at $$p^2=0$$, and with it the dispersion relation $$\omega=\vert\vec k\vert$$ in flat space, is unchanged, provided the Lorentz and diffeomorphism Ward identities keep the self-energy transverse and a function of $$p^2$$ only. And the part of the response that depends on $$a$$ is $$a\,(G_{\rm adv}-G_{\rm ret})*J$$, a band-pass of the source above the ghost mass supported partly before the signal could arrive: it is not a change of the propagation speed. Prescriptions that give the ghost a finite width, and prescriptions that are not Lorentz invariant, are outside the family $$G_a$$.

## 5. An empirical bound on the ghost mass

The static potential between two masses in quadratic gravity is [[Stelle78]](#ref-stelle78)

$$
V(r)=-\frac{Gm_1m_2}{r}\Big(1+\frac13e^{-m_0r}-\frac43e^{-m_2r}\Big),\tag{7}
$$

with a massive scalar of mass $$m_0$$ and a massive spin-2 ghost of mass $$m_2$$. By Proposition 5(i) the term with $$m_2$$ has the same form for all Lorentz-invariant prescriptions of the ghost. Torsion-balance tests of the inverse-square law [[Lee20]](#ref-lee20) report, with 95% confidence, that gravitational-strength Yukawa interactions ($$\vert\alpha\vert=1$$) with ranges above $$38.6\ \mu$$m are excluded. A Yukawa range of $$38.6\ \mu$$m corresponds to a mass $$\hbar c/\lambda=5.1$$ meV. Reading the reported limit as a bound on $$\vert\alpha\vert$$ at fixed range, which applies to a coupling of relative strength $$4/3$$ a fortiori, a spin-2 ghost with $$m_2<5.1$$ meV is excluded. This is an empirical statement about the Yukawa term (7) and is not part of any theorem of this page. It does not restrict ghost masses above $$5.1$$ meV.

## 6. What the theorem does not exclude

* A theory in which the ghost is removed from the physical states, as in the positive-quotient (fakeon) reading, has a positive Wightman function on the physical algebra, and Theorem 2 does not apply to that algebra. The Euclidean correlators with the signed weight are not reflection positive (section 2), and the replica entropy computed from them need not be the entropy of a state of the physical algebra. Whether the two agree for a given theory is a separate calculation.
* A positive state exists on the abstract algebra of Theorem 2 if the ghost has negative energy. Its entropy is defined, and the vacuum is not an energy minimum.
* Notions of positivity that use a Hilbert norm dominating the Krein form (Hilbert majorants) in place of the Krein adjoint are not covered. They define a different notion of state.
* Proposition 4 does not exclude exact boost covariance in an algebra with an infinite trace ($$I_\infty$$, $$II_\infty$$, or type III), nor covariance that is exact below a cutoff in a finite algebra if the defect at the cutoff does not feed down.
* Proposition 5 does not cover prescriptions that are not Lorentz invariant, prescriptions with a finite ghost width, nor complex-conjugate pairs of ghost poles, for which unitarity violations are reported in the operator formalism [[KK]](#ref-kk).
* The bound of section 5 does not exclude a ghost of mass above $$5.1$$ meV.

## 7. Open points

The Reeh–Schlieder density used in Theorem 2 is for the Minkowski vacuum of a generalised free field; the corresponding statement for the Krein vacuum of an interacting theory, or on a curved background such as a de Sitter static patch, is not covered here. For the positive-quotient reading, whether the Euclidean replica entropy of the signed weight agrees with the entropy of a type I truncation of the physical algebra can be tested on a lattice wedge for a Lee–Wick pair; this has not been done here. Whether the fluctuation-dissipation relation for an accelerated detector holds at the Unruh temperature when the response is computed with the fakeon prescription, where the response function differs from the commutator by the on-shell band, is also open. Whether the range of the trace on projections in a gravitational algebra is discrete, as for $$I_n$$, or continuous, as for $$II_1$$, is a non-perturbative question that Proposition 4 does not address.

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
