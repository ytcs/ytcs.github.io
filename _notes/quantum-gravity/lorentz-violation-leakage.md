---
title: Where a preferred frame leaks into low-energy physics
date: 2026-10-29
category: gravity
order: 7
description: A frame, or a compact residual symmetry group, in the substrate does not make gravitons, photons and matter share one limiting speed. The page proves the symmetry counts, shows that averaging over frames is impossible, computes the one-loop transfer from a frame in a matter regulator, and shows that code distance does not equalise speeds.
status: Symmetry counts proved with an exact finite computation; helicity, supersymmetry and no-averaging statements proved; one-loop coefficient proved for a stated class of regulators in a Yukawa model; the estimate for gauge theories is a replacement of couplings, not a computation.
excludes: Protection of equal graviton, photon and matter speeds by spacetime symmetry, by averaging over frames or by code distance, when the residual symmetry group is compact or a frame enters the matter regulator. Exact supersymmetry with one algebra and exact Euclidean hypercubic symmetry are not excluded.
---

Suppose a substrate singles out a frame, or leaves only a compact subgroup of the Lorentz group unbroken. Do gravitons, photons and matter then still share one limiting speed at low energy? Under the conditions below they do not, and the mismatch is not suppressed by the cutoff. (i) If the residual group is compact, the quadratic action of each species contains its own speed parameter (section 1). A group that acts absolutely irreducibly on $$\mathbb R^4$$, such as the hypercubic group, is the exception: it allows one speed for all two-derivative bosonic actions. (ii) If the vacuum has no boost, no bosonic symmetry that commutes with energy, momentum and rotations relates the graviton speed to the photon speed, whereas exact supersymmetry with a single algebra does (section 2). (iii) A frame cannot be averaged away, because there is no Lorentz-invariant probability measure on the set of frames (section 3). (iv) In a Yukawa model where a frame enters the regulator of the scalar line, the fermion speed is shifted at one loop by $$g^2/48\pi^2$$, for every regulator in a stated class and independently of the cutoff (section 4). (v) For stabiliser codes with generic local perturbations the code distance does not equalise the speeds of two species; it multiplies a relative asymmetry in the couplings by the order at which the speed-setting coupling first appears (section 5).

That a frame in a regulator feeds order-one Lorentz violation into the marginal operators of the low-energy theory is the fine-tuning argument of Collins, Perez, Sudarsky, Urrutia and Vucetich [[Collins et al.]](#ref-collins). Pospelov and Shang [[Pospelov–Shang]](#ref-ps) showed that if a Lorentz-violating sector (Hořava–Lifshitz gravity [[Hořava]](#ref-horava)) couples to Lorentz-invariant matter only through Planck-suppressed operators, the transmitted violation is suppressed by the squared ratio of the two scales. Coates, Melby-Thompson and Mukohyama [[Coates et al.]](#ref-cmm) found power-law divergences in the difference of sound speeds of a gauge field and a scalar in Hořava gravity, which spoil that suppression unless the matter sector is also modified at high energy. Groot Nibbelink and Pospelov [[Groot Nibbelink–Pospelov]](#ref-gnp) analysed supersymmetry as a protection, and Redigolo [[Redigolo]](#ref-redigolo) showed that in renormalisable Lorentz-violating supersymmetric gauge theories with linear supercharges the only violation sits in a constant $$c$$ in the supersymmetry algebra, which does not renormalise. Renormalisation-group flow towards a common speed [[Anber–Donoghue]](#ref-ad) [[Iengo–Russo–Serone]](#ref-irs) [[Bednik–Pujolàs–Sibiryakov]](#ref-bps) is not analysed here. What is added is the following: a proof that gauge-invariant two-derivative forms are classified by symmetric tensors, which turns the counting of free speeds into representation theory; the helicity argument and the supersymmetry statement in a form with explicit hypotheses; the no-averaging statement; a proof that the one-loop coefficient $$1/3$$ in units of $$g^2/16\pi^2$$ does not depend on the shape of the regulator; and the monomial-support statement for code Hamiltonians.

The question for a discrete substrate with exact Lorentz symmetry is treated elsewhere: [Lorentz-invariant quantum sprinklings](/quantum-gravity/quantum-sprinklings/) and [Lorentz-invariant point sets](/quantum-gravity/invariant-point-sets/) discuss what such substrates can carry, and [Exact discrete Lorentz symmetry cannot carry propagation](/quantum-gravity/discrete-lorentz-group/) treats non-compact residual groups, which are not considered below. Frames that are properties of a state, such as the velocity of a cluster of particles, are treated in [Frames, Wilson terms and reflection positivity for fermions on random points](/quantum-gravity/frames-and-reflection-positivity/). The same question for gravitons on a lattice appears in [Lattice gravitons, exact diffeomorphisms and lattice momentum](/quantum-gravity/lattice-diffeomorphisms/).

## 1. What a residual symmetry group fixes

We begin with a scalar. A free scalar with a frame has the action $$\tfrac12 a\,\dot\phi^2-\tfrac12 b\,(\nabla\phi)^2$$ and speed $$c^2=b/a$$. Lorentz invariance sets $$a=b$$. If only a subgroup $$G$$ of the Lorentz group survives, the allowed coefficient tensors are the $$G$$-invariant symmetric bilinear forms on momentum space, and a compact $$G$$ always admits two independent ones: the Minkowski form $$\eta$$ and a positive-definite form obtained by averaging over $$G$$. The ratio $$b/a$$ is then a free parameter of the quadratic action, one per species. For spin 2 gauge invariance restricts the allowed forms, and the question is how many survive.

Write a real quadratic form in a symmetric tensor $$h_{\mu\nu}$$ and a momentum $$k_\mu$$ on $$\mathbb R^4$$ as a polynomial $$Q(h,k)$$ of degree 2 in the ten components of $$h$$ and degree 2 in $$k$$; this is a two-derivative action $$\int h(-k)^{T}M(k)\,h(k)$$ with $$M$$ quadratic in $$k$$. It is *gauge invariant* if

$$
Q\bigl(h+k\otimes\xi+\xi\otimes k,\;k\bigr)=Q(h,k)\qquad\text{for all }\xi\in\mathbb R^4. \tag{1}
$$

Let $$V$$ be the space of gauge-invariant forms. A matrix $$g\in GL(4,\mathbb R)$$ acts by $$k\mapsto gk$$, $$h\mapsto ghg^{T}$$, and (1) is preserved, so $$V$$ is a representation of $$GL(4,\mathbb R)$$. For a symmetric invertible matrix $$G^{\mu\nu}$$ define the Fierz–Pauli form with all contractions made with $$G$$:

$$
Q_G(h,k)=-\tfrac12\,(k\!\cdot\!k)\,h_{\mu\nu}h^{\mu\nu}+k^{\mu}h_{\mu\nu}\,k_{\lambda}h^{\lambda\nu}-(k^{\mu}k^{\nu}h_{\mu\nu})\,h+\tfrac12\,(k\!\cdot\!k)\,h^{2}, \tag{2}
$$

where indices are raised and contracted with $$G$$, $$k\!\cdot\!k=G^{\mu\nu}k_\mu k_\nu$$ and $$h=G^{\mu\nu}h_{\mu\nu}$$. For $$G=\eta$$ this is the linearised Einstein–Hilbert action. For every $$G$$ it satisfies (1): it is the Einstein–Hilbert form written in linear coordinates adapted to $$G$$, and a linear change of coordinates maps linearised gauge transformations to linearised gauge transformations. Each term of (2) contains three factors of $$G$$, so $$Q_{\lambda G}=\lambda^{3}Q_G$$. In the eigenvalue basis $$G=\mathrm{diag}(-1,c^2,c^2,c^2)$$ a transverse-traceless mode with polarisation $$h_{12}$$ and momentum $$k=(\omega,0,0,k_z)$$ has $$Q_G=c^{4}(\omega^{2}-c^{2}k_z^{2})$$, so $$Q_G$$ describes helicity-2 waves with speed $$c$$.

<div class="proposition" markdown="1">
**Proposition 1** (gauge-invariant two-derivative forms). Let $$\mathrm{Sym}^2$$ be the space of symmetric $$4\times4$$ matrices with the action $$s\mapsto gsg^{T}$$ of $$O(4)$$. The map $$\Phi:\mathrm{Sym}^2\to V$$, $$\Phi(s)=\frac{d}{dt}Q_{1+ts}\Big\rvert_{t=0}$$, is an isomorphism of $$O(4)$$-representations. In particular $$\dim V=10$$, and for every subgroup $$G\subset O(4)$$ the number of $$G$$-invariant gauge-invariant forms is

$$
n(G)=\dim V^{G}=\dim(\mathrm{Sym}^2)^{G}. \tag{3}
$$

</div>

<div class="proof" markdown="1">
*Proof.* The idea is to exhibit ten independent elements of $$V$$ with the right transformation law and to show by a finite computation that there are no others. We use the identity $$Q_{G}(ghg^{T},gk)=Q_{g^{T}Gg}(h,k)$$, which holds because every contraction in (2) is a contraction with $$G$$ of vectors transformed by $$g$$. Differentiating at $$G=1$$ gives $$\Phi(g^{T}sg)(h,k)=\Phi(s)(ghg^{T},gk)$$, so $$\Phi$$ is $$O(4)$$-equivariant.

Next, $$\Phi$$ is injective. The space $$\mathrm{Sym}^2=\mathbb R\,1\oplus\mathrm{Sym}^2_0$$ splits into the trace part and the nine-dimensional traceless part, which are irreducible and not isomorphic, so the kernel of $$\Phi$$ is a sum of a subspace of each. By homogeneity $$\Phi(1)=3Q_1\neq0$$. On the traceless part, take the transverse-traceless test mode above with $$G=1+ts$$ diagonal, $$s=\mathrm{diag}(3,-1,-1,-1)$$: then $$Q_G=-(G^{00}\omega^{2}+G^{33}k_z^{2})\,G^{11}G^{22}$$ and $$\frac{d}{dt}Q_G\Big\rvert_{t=0}=-(3\omega^{2}-k_z^{2})+2(\omega^{2}+k_z^{2})=3k_z^{2}-\omega^{2}\neq0$$. Hence $$\ker\Phi=0$$.

Finally $$\dim V\le10$$. Condition (1) is a system of linear equations for the $$55\times10=550$$ coefficients of $$Q$$. Evaluating it at 590 random integer points $$(h,k,\xi)$$ gives a matrix with integer entries whose rank modulo each of the primes $$2147483629$$ and $$1000003$$ is 540. A rank modulo a prime does not exceed the rank over $$\mathbb Q$$, which does not exceed the rank of the full system, so the solution space has dimension at most $$550-540=10$$. Since $$\Phi$$ is injective, $$\dim V=10$$ and $$\Phi$$ is an isomorphism. $$\square$$
</div>

The heart of the proof is the identification of a ten-dimensional space of solutions with $$\mathrm{Sym}^2$$; the equivariance and injectivity are routine, and the upper bound on $$\dim V$$ is a computation, not an argument by hand. Table 1 lists $$n(G)$$ computed directly from the defining equations (1) and the invariance conditions, for the graviton and for a Maxwell field with $$Q(A,k)$$ quadratic in $$A_\mu$$ and gauge transformations $$A\mapsto A+k\,\xi$$, together with the number of invariant forms when gauge invariance is not imposed. The groups are: $$O(4)$$; the hypercubic group $$H(4)$$ of signed permutations of the four axes; $$O(3)\times T$$, spatial rotations together with time reversal $$k_0\mapsto-k_0$$ (with $$h_{0i}\mapsto-h_{0i}$$); and the cubic group of signed permutations of the three spatial axes times $$T$$. The computations use floating-point singular values with a gap of $$10^{-13}$$ against $$0.4$$ or larger; the gauge-invariant graviton entries agree with $$\dim(\mathrm{Sym}^2)^G$$ counted by hand ($$1$$ for $$O(4)$$ and $$H(4)$$; $$2$$ for $$O(3)\times T$$ and for the cubic group times $$T$$, the invariants being $$\delta_{ij}$$ and $$u_\mu u_\nu$$ with $$u$$ along the time axis).

| group | graviton, any form | graviton, gauge invariant | Maxwell, any form | Maxwell, gauge invariant |
|---|---|---|---|---|
| none | 550 | 10 | 100 | 20 |
| $$O(4)$$ | 4 | 1 | 2 | 1 |
| $$H(4)$$ | 9 | 1 | 3 | 1 |
| $$O(3)\times T$$ | 17 | 2 | 6 | 2 |
| cubic $$\times\,T$$ | 26 | 2 | 7 | 2 |

*Table 1. Dimensions of the spaces of two-derivative quadratic forms invariant under the group.*

Two consequences follow.

<div class="corollary" markdown="1">
**Corollary 2** (absolutely irreducible groups). Let $$G\subset O(4)$$ act on $$\mathbb R^4$$ absolutely irreducibly, as $$H(4)$$ does. Then $$(\mathrm{Sym}^2)^G=\mathbb R\,1$$, so every $$G$$-invariant gauge-invariant two-derivative graviton action is a multiple of $$Q_1$$, the Fierz–Pauli action with a single metric. The same holds for the scalar action $$c^{\mu\nu}k_\mu k_\nu$$ and, by the computation in Table 1, for a Maxwell field when $$G=H(4)$$.
</div>

<div class="proof" markdown="1">
*Proof.* A symmetric invariant bilinear form on an absolutely irreducible real representation is unique up to a factor, by Schur's lemma applied to the complexification. For $$H(4)$$ the commutant of the signed permutations is scalar: commuting with all sign changes forces a diagonal matrix, and commuting with all permutations forces equal entries. Equation (3) gives the graviton statement, and for scalars the allowed coefficients are exactly $$(\mathrm{Sym}^2)^G$$. $$\square$$
</div>

All species then share the same tensor $$1$$, so in Euclidean signature the relative weight of $$\partial_0^2$$ and $$\partial_i^2$$ is the same for all of them. Since $$-1\in H(4)$$, every odd-rank invariant also vanishes, which removes all odd-dimension Lorentz-violating operators; the first operator that distinguishes $$H(4)$$ from $$O(4)$$ has four derivatives. Passing to a common limiting speed in real time requires that the Euclidean theory be reflection positive and that the linearised gauge invariance persist at interacting level. When the gauge invariance is only approximate, Table 1 shows what opens up: the $$H(4)$$-invariant graviton forms number nine, only one of which is gauge invariant.

<div class="corollary" markdown="1">
**Corollary 3** (compact groups in real time). Let $$G\subset O(3,1)$$ be compact. Then $$n(G)\ge2$$ for the graviton, and the scalar kinetic term likewise has at least two independent coefficients. The speed of each species is then a free parameter of its quadratic action, and no relation among the speeds of different species follows from the symmetry.
</div>

<div class="proof" markdown="1">
*Proof.* Averaging $$g^{T}g$$ over $$G$$ with Haar measure gives a $$G$$-invariant positive-definite form $$P$$. In a basis in which $$P=1$$ the group lies in $$O(4)$$, and the Minkowski form becomes a symmetric matrix $$\eta'$$ that is invariant under $$G$$ and indefinite, hence not proportional to $$1$$. Both $$1$$ and $$\eta'$$ lie in $$(\mathrm{Sym}^2)^G$$, so (3) gives $$n(G)\ge2$$. The dimension $$n(G)$$ does not depend on the basis, because $$V$$ is stable under $$GL(4,\mathbb R)$$. For scalars the invariant coefficient tensors are the same two matrices. $$\square$$
</div>

Corollary 3 is a statement about the most general action allowed by the symmetry: the group does not prevent the two species from having different speeds, and the following sections show that interactions generically make them different. Whether the free parameters are in fact tuned equal is a different question; Corollary 3 only says that equality would be an input and not a consequence. The case of a non-compact residual group, which contains unbounded boosts, is treated in [Exact discrete Lorentz symmetry cannot carry propagation](/quantum-gravity/discrete-lorentz-group/).

## 2. Vacua without boosts, and supersymmetry

A Hořava-type vacuum is invariant under time and space translations and spatial rotations, possibly with time reversal, and has no boost. Call this residual group $$\Gamma$$. We assume (B1) that the graviton has a linearised gauge invariance between foliation-preserving diffeomorphisms and the full linearised diffeomorphisms, (B2) that the photon has a $$U(1)$$ gauge invariance, and (B3) that the quadratic action has no cross term between $$h$$ and $$A$$, as follows for instance from charge conjugation $$A\mapsto-A$$.

<div class="proposition" markdown="1">
**Proposition 4** (free speeds without boosts). Under (B1) to (B3):

(a) For every pair $$c_T,c_\gamma>0$$ the quadratic action $$Q_{G_{c_T}}(h,k)+Q^{\mathrm{Max}}_{G_{c_\gamma}}(A,k)$$, with $$G_c=\mathrm{diag}(-1,c^2,c^2,c^2)$$ and $$Q^{\mathrm{Max}}_G=-\tfrac14F_{\mu\nu}F_{\rho\sigma}G^{\mu\rho}G^{\nu\sigma}$$, is $$\Gamma$$-invariant and gauge invariant under the full linearised diffeomorphisms, hence under the foliation-preserving subgroup. Its helicity-2 modes propagate with speed $$c_T$$ and its helicity-1 modes with speed $$c_\gamma$$. So $$c_T/c_\gamma$$ is a free parameter of the most general quadratic action.

(b) Let $$U$$ be a unitary operator that commutes with the Hamiltonian $$H$$, the momentum $$\vec P$$ and the angular momentum $$\vec J$$. Then $$U$$ maps the one-particle states of momentum $$\vec p$$ and helicity $$\lambda$$ to states with the same $$\vec p$$ and $$\lambda$$. It therefore cannot map graviton states ($$\lvert\lambda\rvert=2$$) to photon states ($$\lvert\lambda\rvert=1$$), and equality of the two dispersion relations is not implied by any such $$U$$.
</div>

<div class="proof" markdown="1">
*Proof.* (a) Invariance under translations and rotations holds because $$G_c$$ is rotation invariant and constant, and $$T$$ acts trivially on it. Gauge invariance of $$Q_G$$ was explained after (2), and the Maxwell form is invariant under $$A\mapsto A+k\xi$$ because it depends on $$F$$ only. The dispersion relation of transverse modes is $$G^{\mu\nu}k_\mu k_\nu=0$$, that is $$\omega=c\lvert\vec k\rvert$$, for each sector with its own $$c$$.

(b) The helicity of a one-particle state with momentum $$\vec p$$ is the eigenvalue of $$\vec J\cdot\hat p$$. If $$U$$ commutes with $$\vec P$$ and $$\vec J$$ it commutes with $$\vec J\cdot\hat P$$ as well, and with $$H$$ it preserves the energy. $$\square$$
</div>

Part (a) is routine and part (b) is a statement that the symmetry classes considered cannot relate the sectors. In a Lorentz-invariant vacuum the relation between the two speeds comes from the boosts, which do not commute with $$\vec J$$, and this is the only structure that has been removed. The proposition does not address symmetries that transform non-trivially under rotations; those would extend the algebra beyond $$\Gamma$$ and are not constructed here. A fermionic symmetry is an exception to the argument, because it can change the helicity by one half.

Exact supersymmetry is of this type. Suppose that a single algebra

$$
\{Q_\alpha,Q^\dagger_\beta\}=2\bigl(H\,\delta_{\alpha\beta}+c\,\vec P\cdot\vec\sigma_{\alpha\beta}\bigr) \tag{4}
$$

acts on all fields, with a real constant $$c>0$$ and Pauli matrices $$\vec\sigma$$, and that $$[J_z,Q_{1,2}]=\pm\tfrac12Q_{1,2}$$ for a momentum along the $$z$$ axis.

<div class="proposition" markdown="1">
**Proposition 5** (short multiplets have speed $$c$$). On the sector of fixed energy $$E$$ and momentum $$\vec p=p\hat z$$, the operators $$Q_1,Q_2$$ satisfy $$\{Q_1,Q_1^\dagger\}=2(E+cp)$$ and $$\{Q_2,Q_2^\dagger\}=2(E-cp)$$, with $$\{Q_1,Q_2^\dagger\}=0$$. Hence every state has $$E\ge cp$$, and a multiplet generated by a single fermionic oscillator, with helicities $$(\lambda,\lambda-\tfrac12)$$, exists exactly when $$E=cp$$. If the graviton and the photon each lie in such two-state multiplets, with helicities $$(2,\tfrac32)$$ and $$(1,\tfrac12)$$, then both have the dispersion relation $$E=c\lvert\vec p\rvert$$ and $$c_{\mathrm{GW}}=c_\gamma=c$$.
</div>

<div class="proof" markdown="1">
*Proof.* With $$\vec p$$ along $$z$$, $$\vec p\cdot\vec\sigma=p\,\sigma_z$$ is diagonal, which gives the two anticommutators and shows that $$Q_1,Q_2$$ anticommute as stated. In a positive-norm Hilbert space $$\lVert Q_2^\dagger\psi\rVert^2+\lVert Q_2\psi\rVert^2=2(E-cp)\lVert\psi\rVert^2\ge0$$, so $$E\ge cp$$. If $$E=cp$$ then $$Q_2=Q_2^\dagger=0$$ on the sector, and the multiplet is built by $$Q_1^\dagger$$ alone, which changes $$J_z$$ by one half: two states. If $$E>cp$$, two oscillators act and the multiplet has four states with helicities $$(\lambda,\lambda-\tfrac12,\lambda-\tfrac12,\lambda-1)$$. $$\square$$
</div>

The proof is the Clifford-oscillator count and is routine; the content is that the constant $$c$$ in the algebra, not a coefficient in the action, sets the speed of every short multiplet. Soft supersymmetry breaking is expected to leave a residual difference of order $$m_{\mathrm{soft}}^{2}/\Lambda_{\mathrm{LV}}^{2}$$ [[Coates et al.]](#ref-cmm), where $$\Lambda_{\mathrm{LV}}$$ is the scale at which the Lorentz-violating terms appear; that estimate is quoted and not derived here. Whether a boost-free gravity theory with a supersymmetry of the form (4) acting on the graviton exists is not settled by this analysis.

## 3. A frame cannot be averaged away

A construction on a point set that uses a unit timelike vector $$u$$ breaks Lorentz invariance unless its observables do not depend on $$u$$. One may try to restore invariance by drawing $$u$$ at random. The following shows that no random choice is Lorentz invariant.

<div class="proposition" markdown="1">
**Proposition 6** (no invariant measure on frames). Let $$d\ge2$$ and let $$H^{d-1}$$ be the set of future unit timelike vectors in $$\mathbb R^{1,d-1}$$. There is no $$SO^{+}(1,d-1)$$-invariant Borel probability measure on $$H^{d-1}$$.
</div>

<div class="proof" markdown="1">
*Proof.* For $$u\in H^{d-1}$$ we have $$(u^{0})^{2}-(u^{1})^{2}=1+\lvert u_\perp\rvert^{2}>0$$, so the rapidity $$\theta(u)=\operatorname{artanh}(u^{1}/u^{0})$$ is defined and continuous. A boost $$B_t$$ in the $$(0,1)$$ plane maps $$(u^0,u^1)$$ to $$(\cosh t\,u^{0}+\sinh t\,u^{1},\ \sinh t\,u^{0}+\cosh t\,u^{1})$$ and so shifts $$\theta$$ by $$t$$. If $$\mu$$ were invariant, the image measure $$\nu=\theta_*\mu$$ on $$\mathbb R$$ would satisfy $$\nu(A+t)=\nu(A)$$ for all $$t$$. Then $$\nu([n,n+1))$$ takes the same value $$a$$ for every integer $$n$$, and $$\sum_n\nu([n,n+1))=1$$ is impossible for a constant $$a$$, whether $$a=0$$ or $$a>0$$. $$\square$$
</div>

<div class="corollary" markdown="1">
**Corollary 7** (no equivariant frame from a Lorentz-invariant point process). Let $$\mathcal P$$ be a random locally finite subset of $$\mathbb R^{1,d-1}$$ whose law is invariant under $$SO^{+}(1,d-1)$$, for example a Poisson process, and let $$u(\mathcal P)\in H^{d-1}$$ be a measurable function of the point set with $$u(g\mathcal P)=g\,u(\mathcal P)$$ almost surely for each $$g$$. There is no such function.
</div>

<div class="proof" markdown="1">
*Proof.* The law $$\mu$$ of $$u(\mathcal P)$$ satisfies $$\mu(gA)=\Pr[g^{-1}u(\mathcal P)\in A]=\Pr[u(g^{-1}\mathcal P)\in A]=\mu(A)$$, because $$g^{-1}\mathcal P$$ has the same law as $$\mathcal P$$. It is therefore an invariant probability measure, which contradicts Proposition 6. $$\square$$
</div>

The corollary is the argument of Bombelli, Henson and Sorkin [[Bombelli–Henson–Sorkin]](#ref-bhs), who prove related statements for sprinklings, in its simplest form. It gives a trichotomy for any matter construction on a Lorentz-invariant point set that uses a timelike axis: the axis is random and independent of the points, in which case its law is not invariant and a frame is selected; or the axis is a function of the points, which Corollary 7 forbids when the function is equivariant and measurable; or the observables are independent of the axis in each realisation. Averaging over the axis cannot turn the first case into a Lorentz-invariant theory. In Euclidean signature the corresponding group $$SO(d)$$ is compact, a Haar average over the axis exists, and the obstruction does not arise; whether such an averaged Euclidean construction defines a Lorentz-invariant real-time theory is the question of reflection positivity, which is not decided here. The construction of fermions on a sprinkling, where a frame is needed to define a kinetic term, appears in [Fermions that see only the causal order](/quantum-gravity/order-only-fermions/).

## 4. A frame in the matter regulator

We now compute how a frame in a regulator is transferred to a matter species. The model is the simplest in which a fermion line meets a regulated line, and the result is independent of the regulator shape within a class.

*The model.* Work in Euclidean $$\mathbb R^4$$ with a unit vector $$u$$ along the time axis. A massless Dirac fermion $$\psi$$ with propagator $$S(p)=1/(i(\gamma\!\cdot\!p))=-i(\gamma\!\cdot\!p)/p^{2}$$ couples to a real scalar through $$g\,\phi\bar\psi\psi$$. The scalar propagator is replaced by

$$
D(q)=\frac{f(\lvert q_\perp\rvert/\Lambda)}{q^{2}+M^{2}},\qquad q_\perp=q-(q\!\cdot\!u)u, \tag{5}
$$

so that the frame enters only through the cutoff on the momentum orthogonal to $$u$$. This is a prescription for a propagator and not an action; it is all that a one-loop self-energy needs. We take $$f(x)=F(x^{2})$$ with $$F$$ continuously differentiable on $$[0,\infty)$$, $$F(0)=1$$, and $$F(s)$$, $$sF'(s)$$ bounded by $$C(1+s)^{-a}$$ for some $$a>0$$. The examples used below are $$e^{-x^{2}}$$, $$e^{-x^{4}}$$ and $$(1+x^{2})^{-2}$$. The scalar has speed 1 for momenta much below $$\Lambda$$, and the question is the speed of the fermion.

The one-loop inverse propagator of $$\psi$$ is

$$
\Gamma(p)=i(\gamma\!\cdot\!p)-g^{2}\!\int\!\frac{d^{4}q}{(2\pi)^{4}}\,S(p-q)\,D(q)=i(\gamma\!\cdot\!p)+ig^{2}\!\int\!\frac{d^{4}k}{(2\pi)^{4}}\,\frac{(\gamma\!\cdot\!k)}{k^{2}}\,D(p-k). \tag{6}
$$

For $$M>0$$ the integral converges absolutely and is continuously differentiable in $$p$$. We need its derivative at $$p=0$$. Differentiating $$D(p-k)$$, using that $$D$$ is even, and integrating by parts (the boundary terms at $$k=0$$ and at infinity vanish) gives

$$
\partial_{p_\nu}\Gamma\Big\rvert_{p=0}=i\gamma_\nu+ig^{2}\!\int\!\frac{d^{4}k}{(2\pi)^{4}}\,D(k)\Bigl(\frac{\gamma_\nu}{k^{2}}-\frac{2(\gamma\!\cdot\!k)\,k_\nu}{k^{4}}\Bigr). \tag{7}
$$

Define $$A=\int D/k^{2}$$, $$B_\parallel=\int D\,(k\!\cdot\!u)^{2}/k^{4}$$ and $$B_\perp=\tfrac13\int D\,k_\perp^{2}/k^{4}$$, all with measure $$d^{4}k/(2\pi)^{4}$$. Rotation invariance about $$u$$ turns (7) into

$$
\Gamma(p)=i\bigl[Z_\parallel\,(p\!\cdot\!u)\,(\gamma\!\cdot\!u)+Z_\perp\,(\gamma\!\cdot\!p_\perp)\bigr]+O(p^{2}),\qquad Z_\parallel=1+g^{2}(A-2B_\parallel),\quad Z_\perp=1+g^{2}(A-2B_\perp). \tag{8}
$$

For $$u$$ the time axis, the pole of the propagator is at $$Z_\parallel^{2}p_0^{2}+Z_\perp^{2}\vec p^{\,2}=0$$, that is at $$E=(Z_\perp/Z_\parallel)\lvert\vec p\rvert$$ for $$p_0=iE$$. The fermion speed relative to the scalar is therefore

$$
\delta c=\frac{Z_\perp}{Z_\parallel}-1=2g^{2}\,(B_\parallel-B_\perp)+O(g^{4}). \tag{9}
$$

As a check, for an isotropic $$D$$ one has $$B_\parallel=B_\perp=A/4$$, so $$\delta c=0$$ and $$Z=1+g^{2}A/2$$, with $$A=\frac{1}{16\pi^{2}}\ln(\Lambda^{2}/M^{2})$$, which is the standard logarithmic wave-function factor $$g^{2}/32\pi^{2}\,\ln\Lambda^{2}$$ of the Yukawa theory.

<div class="theorem" markdown="1">
**Theorem 8** (universal transfer). Let $$f$$ be in the class above and $$M>0$$. Then

$$
\delta c=\frac{g^{2}}{16\pi^{2}}\;C\!\left(\tfrac{M}{\Lambda};f\right)+O(g^{4}),\qquad \lim_{M/\Lambda\to0}C\!\left(\tfrac{M}{\Lambda};f\right)=\frac13 . \tag{10}
$$

Thus $$\delta c\to g^{2}/48\pi^{2}$$ for every such $$f$$. For the Gaussian $$f=e^{-x^{2}}$$ the correction is $$O\bigl((M/\Lambda)^{2}\ln(\Lambda/M)\bigr)$$. For a regulator that depends on the full four-momentum, $$f=f(\lvert q\rvert/\Lambda)$$, $$C=0$$.
</div>

<div class="proof" markdown="1">
*Proof.* The idea is to write (9) in four-dimensional polar coordinates about $$u$$, where the regulator breaks isotropy only through the angle, and to note that the shape of $$f$$ enters the radial integral through an additive constant that multiplies an angular average equal to zero.

*Step 1 (reduction to one angle and one radius).* Let $$\chi$$ be the angle between $$k$$ and $$u$$, so that $$k\!\cdot\!u=\lvert k\rvert\cos\chi$$, $$\lvert k_\perp\rvert=\lvert k\rvert\sin\chi$$ and $$d^{4}k=4\pi\,\lvert k\rvert^{3}\sin^{2}\chi\,d\chi\,d\lvert k\rvert$$. Then $$B_\parallel-B_\perp=\int D\,(k\!\cdot\!u)^{2}/k^{4}-\tfrac13\int D\,k_\perp^{2}/k^{4}$$ gives

$$
\delta c=\frac{2g^{2}}{4\pi^{3}}\int_{0}^{\pi}\!d\chi\;w(\chi)\,\Phi(\sin\chi),\qquad w(\chi)=\sin^{2}\chi\Bigl(\cos^{2}\chi-\tfrac13\sin^{2}\chi\Bigr), \tag{11}
$$

with $$\Phi(s)=\int_{0}^{\infty}\frac{\kappa\,d\kappa\,f(\kappa s/\Lambda)}{\kappa^{2}+M^{2}}$$. The weight $$w$$ is the quadrupole component of the angular distribution and has zero mean: $$\int_0^\pi w\,d\chi=\pi/8-\tfrac13\cdot3\pi/8=0$$.

*Step 2 (the radial integral).* Put $$t=\kappa s/\Lambda$$ and $$\varepsilon=Ms/\Lambda$$, so that $$\Phi(s)=\int_0^\infty t f(t)\,(t^{2}+\varepsilon^{2})^{-1}dt$$. Split the range at $$t=1$$. On $$(0,1)$$ write $$f=1+(f-1)$$; since $$f(t)-1=O(t^{2})$$, the second term gives a convergent limit as $$\varepsilon\to0$$ by dominated convergence, and the first gives $$\tfrac12\ln\bigl((1+\varepsilon^{2})/\varepsilon^{2}\bigr)$$. On $$(1,\infty)$$ the integral converges to $$\int_{1}^{\infty}f(t)\,dt/t$$, finite because $$f$$ decays. Hence

$$
\Phi(s)=\ln\frac{\Lambda}{M}-\ln s+c_f+\rho(\varepsilon),\qquad c_f=\int_0^1\frac{f-1}{t}\,dt+\int_1^\infty\frac{f}{t}\,dt, \tag{12}
$$

with $$\rho(\varepsilon)\to0$$ as $$\varepsilon\to0$$. Since $$\varepsilon\le M/\Lambda$$, the remainder is uniformly small on $$0<s\le1$$ when $$M/\Lambda$$ is small. The constant $$c_f$$ and the term $$\ln(\Lambda/M)$$ do not depend on $$s$$, hence do not depend on $$\chi$$.

*Step 3 (the angular integral).* Insert (12) in (11). The $$\chi$$-independent terms multiply $$\int w\,d\chi=0$$ and drop out. What remains is $$-\int_0^\pi w(\chi)\ln\sin\chi\,d\chi$$ plus a term that tends to zero. With $$w=\sin^{2}\chi-\tfrac43\sin^{4}\chi$$ and $$J(a)=\int_0^\pi\sin^{a}\chi\,d\chi=\sqrt\pi\,\Gamma(\tfrac{a+1}{2})/\Gamma(\tfrac a2+1)$$ one has $$\int_0^\pi\sin^{a}\chi\ln\sin\chi\,d\chi=J'(a)$$. Using $$\psi(\tfrac32)=2-\gamma-2\ln2$$, $$\psi(2)=1-\gamma$$, $$\psi(\tfrac52)=\tfrac83-\gamma-2\ln2$$ and $$\psi(3)=\tfrac32-\gamma$$,

$$
J'(2)=\frac\pi4\,(1-2\ln2),\qquad J'(4)=\frac{3\pi}{16}\Bigl(\frac76-2\ln2\Bigr),\qquad \int_0^\pi w\ln\sin\chi\,d\chi=J'(2)-\tfrac43J'(4)=-\frac{\pi}{24}.
$$

Therefore $$\delta c\to\frac{2g^{2}}{4\pi^{3}}\cdot\frac{\pi}{24}=\frac{g^{2}}{48\pi^{2}}$$, which is (10) with $$C=1/3$$. For the Gaussian, $$\Phi(s)=\tfrac12e^{\varepsilon^{2}}E_1(\varepsilon^{2})$$ exactly, and its expansion gives $$\rho=O(\varepsilon^{2}\ln\varepsilon)$$. If $$f$$ depends on $$\lvert k\rvert$$ the radial integral does not depend on $$\chi$$ and (11) vanishes by $$\int w=0$$. $$\square$$
</div>

The argument shows why the coefficient does not depend on the regulator. The integrand of (9) is a quadrupole, so with an isotropic regulator its angular average vanishes at each radius. A regulator that cuts off only the momentum orthogonal to $$u$$ removes more of the large-$$\lvert q_\perp\rvert$$ region at angles near $$\pi/2$$ than near the axis, and the induced angular dependence is the universal $$-\ln\sin\chi$$ of the infrared-sensitive radial integral. The shape of $$f$$ only changes the constant $$c_f$$, which the zero-mean weight removes.

*Checks.* The reduction (11) and the limit were checked in two independent ways. Direct quadrature of (11) with the radial integral done numerically gives, for $$C=\delta c/(g^2/16\pi^2)$$ at $$M/\Lambda=10^{-1},10^{-2},10^{-3},10^{-4}$$: $$0.3256,\,0.3332,\,0.3333,\,0.3333$$ for $$e^{-x^{2}}$$; $$0.3305,\,0.3333,\,0.3333,\,0.3333$$ for $$e^{-x^{4}}$$; and $$0.3209,\,0.3331,\,0.3333,\,0.3333$$ for $$(1+x^{2})^{-2}$$. A two-dimensional quadrature in cylindrical coordinates $$(k\!\cdot\!u,\lvert k_\perp\rvert)$$ for the Gaussian at $$M/\Lambda=10^{-2}$$ gives $$0.33318$$, in agreement with the polar value $$0.33318$$ at the same ratio. The isotropic control gives $$0$$ to machine precision.

Two remarks on the scope of the theorem. First, with $$M=0$$ the double integral defining $$\delta c$$ is only conditionally convergent: integrating over the component along $$u$$ first at fixed $$\lvert k_\perp\rvert$$ gives zero in the form (7), while the limit $$M/\Lambda\to0$$ of the absolutely convergent integral gives $$1/3$$. The mass must be kept until the end. Second, the result is for a regulator on the scalar line only. Regulators placed on other lines, or on vertices, give other coefficients: the value is regulator dependent in general [[Collins et al.]](#ref-collins), and the universality proved here is universality in the shape of $$f$$ for this placement.

*What this means for a gauge theory (heuristic).* The coefficient multiplies a dimension-four operator, $$c^{\mu\nu}\bar\psi\gamma_\mu\partial_\nu\psi$$ with $$c^{\mu\nu}\propto u^{\mu}u^{\nu}-\tfrac14\delta^{\mu\nu}$$, and it does not depend on $$\Lambda$$. A cutoff at the Planck scale therefore gives no suppression. Replacing $$g^{2}$$ by $$e^{2}$$ gives $$\delta c\sim\alpha/12\pi=1.9\times10^{-4}$$ for $$\alpha=1/137.036$$. This is an order-of-magnitude replacement: the factor from the gauge structure of the photon line is not computed here, and in a theory with several couplings the coefficients of different species need not coincide. The GW170817 observation bounds the difference between the speeds of gravity and light to between $$-3\times10^{-15}$$ and $$+7\times10^{-16}$$ of the speed of light [[LIGO–Virgo et al.]](#ref-gw170817), and the data tables of [[Kostelecký–Russell]](#ref-kr) list bounds on dimension-four coefficients of electrons, protons and neutrons that lie many orders of magnitude below $$10^{-10}$$. A coefficient near $$10^{-4}$$ would therefore need a cancellation by many orders of magnitude between independently renormalised parameters. No custodial symmetry is available: the operator is a singlet under every internal symmetry, and only exact Lorentz symmetry of the regulator or a fermionic generator forbids it, in line with section 2.

## 5. Code distance does not equalise speeds

A different proposal is that Lorentz invariance in the infrared might be protected by an error-correcting structure: the Lorentz-violating terms would be logical errors and so suppressed by the code distance, while the kinetic terms are not. We test this for stabiliser codes.

*Setting.* Let $$\mathcal S$$ be an abelian subgroup of the Pauli group on $$n$$ qubits with $$-1\notin\mathcal S$$, with generators $$S_s$$, code space $$\mathcal C$$ (the joint $$+1$$ eigenspace) and projector $$\Pi$$. The distance $$d$$ is the least weight of an element of the normaliser $$N(\mathcal S)$$ that is not in $$\mathcal S$$. Let $$H_0=-\Delta\sum_sS_s$$ with $$\Delta>0$$ and $$V=\sum_t\lambda_t P_t$$ with $$P_t$$ Pauli strings and $$\lambda_t$$ arbitrary couplings. The low-energy Hamiltonian on $$\mathcal C$$ is the Schrieffer–Wolff series $$H_{\mathrm{eff}}=\Pi e^{\mathcal T}(H_0+V)e^{-\mathcal T}\Pi$$ [[Bravyi–DiVincenzo–Loss]](#ref-bdl), defined as a formal power series in the $$\lambda_t$$. We use only the following property of that series, shared by the Schrieffer–Wolff, Kato and Rayleigh–Schrödinger expansions: the term of order $$N$$ is a linear combination, with coefficients independent of the $$\lambda_t$$, of operators

$$
\Pi\,P_{t_1}f_1(\mathbf S)\,P_{t_2}f_2(\mathbf S)\cdots f_{N-1}(\mathbf S)\,P_{t_N}\,\Pi, \tag{13}
$$

where each $$f_i$$ is a function of the commuting stabilisers (energy denominators and projectors on syndrome sectors). Expand $$H_{\mathrm{eff}}=\sum_La_L(\lambda)\,L$$ in a basis of logical Pauli operators, one representative $$L$$ for each class in $$N(\mathcal S)/\langle\mathcal S\rangle$$.

<div class="proposition" markdown="1">
**Proposition 9** (disjoint monomial supports). Every monomial $$\prod_t\lambda_t^{m_t}$$ that appears in $$a_L$$ satisfies $$[\prod_tP_t^{m_t}]=[L]$$ in $$N(\mathcal S)/\langle\mathcal S\rangle$$. Hence $$a_L$$ and $$a_{L'}$$ for distinct classes have no monomial in common. If $$V$$ consists of single-qubit Paulis, the least degree of $$a_L$$ is the least weight in the class $$[L]$$, so at least $$d$$ for $$L\notin\langle\mathcal S\rangle$$.
</div>

<div class="proof" markdown="1">
*Proof.* The idea is that between code projectors the stabilisers act as numbers, so only the Pauli product of an operator (13) matters, and a Pauli product determines its logical class.

A Pauli string either commutes or anticommutes with each stabiliser, so a function $$f(\mathbf S)$$ can be moved through $$P_t$$ at the price of flipping the arguments by signs. Pushing all $$f_i$$ to the right in (13) and acting on $$\Pi$$, where each stabiliser equals $$+1$$, turns (13) into a number times $$\Pi R\Pi$$ with $$R=P_{t_1}\cdots P_{t_N}$$, a Pauli string up to phase. If $$R$$ anticommutes with some $$S_s$$, then $$\Pi R\Pi=0$$. Otherwise $$R\in N(\mathcal S)$$ and $$\Pi R\Pi=R\,\Pi=\pm L_R\Pi$$, where $$L_R$$ is the logical representative of the class $$[R]$$. The product $$R$$ depends on the multiset $$\{t_i\}$$ only up to a phase, hence so does its class, and the monomial $$\lambda_{t_1}\cdots\lambda_{t_N}$$ feeds only $$a_{L_R}$$. For single-qubit $$V$$ the number of factors needed to build a string in a class is at least the weight of that class. $$\square$$
</div>

The step with content is the observation that the stabilisers drop out; the rest is bookkeeping. For a speed, the relevant objects are the logical couplings. A species whose dynamics is carried by logical qubits has a speed determined by the coefficients $$a_L$$ of its logical Hamiltonian, and Proposition 9 says which microscopic couplings feed which $$a_L$$.

<div class="corollary" markdown="1">
**Corollary 10** (decoupled species). Suppose $$V=V^{(1)}+V^{(2)}$$, with the two terms acting on disjoint blocks of qubits and containing disjoint sets of couplings $$\lambda^{(1)},\lambda^{(2)}$$. Then $$H_{\mathrm{eff}}=H^{(1)}_{\mathrm{eff}}(\lambda^{(1)})+H^{(2)}_{\mathrm{eff}}(\lambda^{(2)})$$, and the speeds are $$c_\alpha=F_\alpha(\lambda^{(\alpha)})$$. If $$c_1=c_2$$ holds on an open set of couplings, then both $$F_1$$ and $$F_2$$ are constant there. A relation between non-constant speeds is a restriction on the couplings, that is a tuning, or follows from a symmetry that maps the couplings of one species onto those of the other.
</div>

The proof of the splitting is that $$e^{\mathcal T}$$ factorises over commuting blocks. The distance $$d$$ enters only through the functions $$F_\alpha$$. For the amplification statement, let a coupling $$a_{L_*}$$ of minimal degree $$k$$ set the speed through $$c_\alpha=\kappa\,a_{L_*}(\lambda^{(\alpha)})\bigl(1+O(\epsilon^{2})\bigr)$$, with $$\kappa$$ the same for both species and $$\epsilon$$ the ratio of the perturbation to the gap $$\Delta$$.

<div class="proposition" markdown="1">
**Proposition 11** (amplification). Under the assumptions just stated, if the couplings of species 2 are those of species 1 multiplied by $$1+\delta$$, then

$$
\ln\frac{c_2}{c_1}=k\,\ln(1+\delta)+O(\epsilon^{2}\delta), \tag{14}
$$

with $$k\ge d$$ when $$V$$ is a sum of single-qubit terms and $$L_*\notin\langle\mathcal S\rangle$$.
</div>

*Proof.* The leading part of $$a_{L_*}$$ is homogeneous of degree $$k$$ in the $$\lambda$$, hence is multiplied by $$(1+\delta)^{k}$$, while higher-degree parts change the relative value by $$O(\epsilon^{2}\delta)$$. $$\square$$

So a 1% difference in the physical couplings of the two species produces a difference of about $$k\%$$ in their speeds. The distance reduces the absolute size of the logical couplings, here as $$\epsilon^{d}$$, and does not reduce their ratios.

*Example (exact).* Each species is a repetition-code chain: blocks of $$d$$ qubits, an intra-block coupling $$-\Delta\,Z_iZ_{i+1}$$ (the stabilisers), a field $$-h\,X_i$$ on every qubit, and an inter-block coupling $$-J\,Z Z$$. The logical $$\bar X=\prod X_i$$ has weight $$d$$ and is generated at order $$d$$. The Jordan–Wigner transform gives a free Majorana chain, and the velocity follows from the smallest singular value of the $$d\times d$$ Bloch matrix with diagonal $$h$$, superdiagonal $$-\Delta$$ and corner entry $$-Je^{ik}$$, at the critical point $$h^{d}=\Delta^{d-1}J$$ (the tuning that the speed-setting coupling needs in either scenario). With $$\epsilon=h/\Delta=0.3$$ and $$\Delta=1$$ the speed per unit cell is $$c=0.4553,\,0.4550,\,0.4550,\,0.4550$$ times $$2\epsilon^{d}\Delta$$ for $$d=3,5,7,9$$, that is $$c\simeq(1-\epsilon^{2})\epsilon^{d}\Delta$$ for $$d\ge5$$. Species 2 differs by 1% in $$h$$ only, with the inter-block coupling re-tuned in both species. Then $$\ln(c_2/c_1)/\ln1.01=1.000,\,2.805,\,4.800,\,6.800,\,8.800$$ for $$d=1,3,5,7,9$$, in agreement with (14) for $$k=d$$ up to the factor $$d-2\epsilon^{2}/(1-\epsilon^{2})$$ that follows from $$c\propto(1-\epsilon^{2})\epsilon^{d}$$. The relative speed difference $$\Delta c/\bar c$$ is $$9.95\times10^{-3},\,2.79\times10^{-2},\,4.78\times10^{-2},\,6.76\times10^{-2},\,8.75\times10^{-2}$$. If instead species 2 differs by 1% in both $$h$$ and $$\Delta$$, then $$h^{d}/\Delta^{d-1}$$ changes by 1% for every $$d$$ and $$\Delta c/\bar c=9.95\times10^{-3}$$ at every $$d$$: the ratio is protected only when the asymmetry is a common rescaling.

*Example (quantum code).* For the $$[[5,1,3]]$$ code with $$H_0=-\Delta\sum S_s$$ ($$\Delta=1$$) and $$V=\epsilon\sum_{q,a}\lambda_{q,a}\sigma^a_q$$ with 15 independent Gaussian couplings, the splitting of the two lowest levels is proportional to $$\epsilon^{3}$$: the logarithmic slopes between $$\epsilon=10^{-3}$$ and $$1.6\times10^{-2}$$ are $$3.0000,\,3.0000,\,3.0000,\,3.0001$$. Two independent draws of the couplings give a splitting ratio $$13.10$$ that is stable in $$\epsilon$$ to $$0.1\%$$ over that range. Multiplying the couplings of the second copy by $$1.01$$ changes the ratio by $$3.030\%$$ and by $$1.02$$ by $$6.121\%$$, equal to $$(1+\delta)^{3}-1=3.0301\%,\,6.1208\%$$. This is a check at $$d=3$$ only.

Neither example is a proof for all codes. The proposition is order by order in the series, and it concerns stabiliser codes with Pauli perturbations. The stability theorem for topologically ordered Hamiltonians [[Bravyi–Hastings–Michalakis]](#ref-bhm) protects quantities proportional to the identity on the code space, such as the degeneracy of the ground space. A speed is a ratio of two non-trivial logical couplings, and the theorem is silent about it.

## 6. What the results do not exclude

The statements above exclude protection by the mechanisms named. They leave the following routes, each with its condition.

- A residual group acting absolutely irreducibly on the Euclidean momentum space, such as $$H(4)$$, with exact linearised diffeomorphism invariance that survives interactions, reflection positivity so that the continuation to real time exists, and no doublers. Corollary 2 gives one speed for the quadratic bosonic actions of such a theory. Whether a substrate with these properties exists is not shown here; see [Lattice gravitons, exact diffeomorphisms and lattice momentum](/quantum-gravity/lattice-diffeomorphisms/) for the obstruction to exact diffeomorphism invariance in Hamiltonian lattices.
- Exact supersymmetry with one algebra of the form (4) acting on all fields. Proposition 5 gives equal speeds for short multiplets. The existence of such a theory including gravity, and the size of the residual difference after soft breaking, are inputs.
- A gravity-only frame that does not enter the matter regulator, so that the transmitted violation is suppressed as in [[Pospelov–Shang]](#ref-ps). Proposition 4 then says that $$c_T/c_\gamma$$ is an input parameter of the gravity sector, which must be small enough to respect the GW170817 window; the page gives no symmetry reason for it to be.
- A tuned counterterm, with one dimension-four coefficient per species and per order of perturbation theory. Theorem 8 gives the size of the coefficient that must be cancelled in the model considered.
- Emergent Lorentz invariance by renormalisation-group flow, for which [[Anber–Donoghue]](#ref-ad), [[Iengo–Russo–Serone]](#ref-irs) and [[Bednik–Pujolàs–Sibiryakov]](#ref-bps) give analyses that this page does not examine.
- A frame that is a property of a state or a cluster of particles and not of the regulator: [Frames, Wilson terms and reflection positivity for fermions on random points](/quantum-gravity/frames-and-reflection-positivity/).
- An averaged axis in Euclidean signature (section 3), subject to reflection positivity.
- Codes outside the stabiliser class (holographic or approximate codes), non-Pauli perturbations, and regimes in which the Schrieffer–Wolff series does not converge. Proposition 9 is not proved there.
- Non-compact residual groups: [Exact discrete Lorentz symmetry cannot carry propagation](/quantum-gravity/discrete-lorentz-group/).

## 7. Open points

The gauge-structure factor that converts the Yukawa coefficient $$1/3$$ into the photon–electron speed difference in a regulated gauge theory is not computed. The universality in the shape of $$f$$ is proved for a regulator acting on one line, and the analogous statement for regulators acting on the fermion line, or on all lines in a gauge-invariant way, has not been established. A two-loop calculation would show whether the one-loop coefficient is stable. The dimension count $$\dim V=10$$ rests on a finite exact computation, and a derivation by hand, for example by writing gauge-invariant forms in terms of the linearised Riemann tensor, would extend more transparently to other spins. Proposition 5 for a graviton multiplet requires a boost-free supergravity with an algebra (4), whose existence is not decided here. It is also not known whether Proposition 9 extends to codes beyond the stabiliser class.

## References

1. <a id="ref-collins"></a>Collins, Perez, Sudarsky, Urrutia, Vucetich, “Lorentz invariance and quantum gravity: an additional fine-tuning problem?,” Phys. Rev. Lett. 93 (2004) 191301, [arXiv:gr-qc/0403053](https://arxiv.org/abs/gr-qc/0403053).
2. <a id="ref-ps"></a>Pospelov, Shang, “On Lorentz violation in Hořava–Lifshitz type theories,” [arXiv:1010.5249](https://arxiv.org/abs/1010.5249).
3. <a id="ref-horava"></a>Hořava, “Quantum gravity at a Lifshitz point,” Phys. Rev. D 79 (2009) 084008, [arXiv:0901.3775](https://arxiv.org/abs/0901.3775).
4. <a id="ref-cmm"></a>Coates, Melby-Thompson, Mukohyama, “Revisiting Lorentz violation in Hořava gravity,” Phys. Rev. D 100 (2019) 064046, [arXiv:1805.10299](https://arxiv.org/abs/1805.10299).
5. <a id="ref-gnp"></a>Groot Nibbelink, Pospelov, “Lorentz violation in supersymmetric field theories,” Phys. Rev. Lett. 94 (2005) 081601, [arXiv:hep-ph/0404271](https://arxiv.org/abs/hep-ph/0404271).
6. <a id="ref-redigolo"></a>Redigolo, “On Lorentz-violating supersymmetric quantum field theories,” Phys. Rev. D 85 (2012) 085009, [arXiv:1106.2035](https://arxiv.org/abs/1106.2035).
7. <a id="ref-ad"></a>Anber, Donoghue, “The emergence of a universal limiting speed,” Phys. Rev. D 83 (2011) 105027, [arXiv:1102.0789](https://arxiv.org/abs/1102.0789).
8. <a id="ref-irs"></a>Iengo, Russo, Serone, “Renormalization group in Lifshitz-type theories,” JHEP 0911 (2009) 020, [arXiv:0906.3477](https://arxiv.org/abs/0906.3477).
9. <a id="ref-bps"></a>Bednik, Pujolàs, Sibiryakov, “Emergent Lorentz invariance from strong dynamics: holographic examples,” [arXiv:1305.0011](https://arxiv.org/abs/1305.0011).
10. <a id="ref-bhs"></a>Bombelli, Henson, Sorkin, “Discreteness without symmetry breaking: a theorem,” Mod. Phys. Lett. A 24 (2009) 2579, [arXiv:gr-qc/0605006](https://arxiv.org/abs/gr-qc/0605006).
11. <a id="ref-gw170817"></a>LIGO Scientific Collaboration, Virgo Collaboration, Fermi Gamma-ray Burst Monitor, INTEGRAL, “Gravitational waves and gamma-rays from a binary neutron star merger: GW170817 and GRB 170817A,” Astrophys. J. Lett. 848 (2017) L13, [arXiv:1710.05834](https://arxiv.org/abs/1710.05834).
12. <a id="ref-kr"></a>Kostelecký, Russell, “Data tables for Lorentz and CPT violation,” Rev. Mod. Phys. 83 (2011) 11, [arXiv:0801.0287](https://arxiv.org/abs/0801.0287).
13. <a id="ref-bdl"></a>Bravyi, DiVincenzo, Loss, “Schrieffer–Wolff transformation for quantum many-body systems,” Ann. Phys. 326 (2011) 2793, [arXiv:1105.0675](https://arxiv.org/abs/1105.0675).
14. <a id="ref-bhm"></a>Bravyi, Hastings, Michalakis, “Topological quantum order: stability under local perturbations,” J. Math. Phys. 51 (2010) 093512, [arXiv:1001.0344](https://arxiv.org/abs/1001.0344).
