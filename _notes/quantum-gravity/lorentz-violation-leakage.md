---
title: Where a preferred frame leaks into low-energy physics
date: 2026-10-29
category: gravity
order: 7
description: A compact residual symmetry group does not force gravitons, photons and matter to share one limiting speed, a frame cannot be averaged away, and in a Yukawa model a frame in the regulator of the scalar line shifts the fermion speed at one loop by an amount independent of the cutoff.
status: Proved, with the bound on the number of gauge-invariant forms and the entries of Table 1 obtained by exact linear algebra over the rationals, the symmetry counts for compact groups, the helicity statement and the supersymmetry statement for the stated algebra, the absence of an invariant probability measure on frames, the one-loop coefficient for a regulator on the scalar line of a Yukawa model, and the monomial supports of effective code Hamiltonians. Calculation under stated hypotheses, checked in two examples, the amplification of coupling asymmetries by the code distance. Heuristic, the estimate for gauge theories.
excludes: That a compact residual symmetry group or a vacuum without boosts enforces equal graviton, photon and matter speeds; that averaging over frames with a probability measure restores Lorentz invariance; that a frame in the scalar-line regulator of a Yukawa model leaves the fermion speed unshifted; that the distance of a stabiliser code with Pauli perturbations equalises the speeds of two species, order by order. Exact supersymmetry with one algebra and exact Euclidean hypercubic symmetry are not excluded.
---

Suppose a substrate selects a frame, or leaves only a compact subgroup of the Lorentz group unbroken. Does the residual symmetry then force gravitons, photons and matter to share one limiting speed at low energy? It does not. For a compact residual group, and in a vacuum without boosts, the quadratic action of each species carries its own speed parameter; the exception considered is a supersymmetry algebra with a single constant (sections 1 and 2). In Euclidean signature a group acting absolutely irreducibly, such as the hypercubic group, leaves a single coefficient for the scalar, graviton and Maxwell two-derivative actions, but reaching a real-time theory then requires reflection positivity. A frame cannot be removed by averaging, because the set of frames carries no Lorentz-invariant probability measure (section 3). In a Yukawa model with a frame in the regulator of the scalar line, the fermion speed shifts at one loop by $$g^2/48\pi^2$$ in the limit of a small scalar mass, for every regulator shape in a stated class and independently of the cutoff (section 4). For stabiliser codes with Pauli perturbations, the code distance multiplies a relative asymmetry between the couplings of two species by the order at which the speed-setting coupling first appears, in perturbation theory and under the hypotheses of Proposition 11 (section 5).

The fine-tuning argument of Collins, Perez, Sudarsky, Urrutia and Vucetich [[Collins et al.]](#ref-collins) is that a frame in a regulator feeds order-one Lorentz violation into the marginal operators of the low-energy theory; they estimate it with a Yukawa self-energy in which the regulator multiplies the fermion propagator. Pospelov and Shang [[Pospelov–Shang]](#ref-ps) showed that when a Lorentz-violating sector (Hořava–Lifshitz gravity [[Hořava]](#ref-horava)) couples to Lorentz-invariant matter only through Planck-suppressed operators, the transmitted violation is suppressed by the squared ratio of the two scales, provided the quadratic divergences from the instantaneous gravity modes are removed by higher-dimension terms. Coates, Melby-Thompson and Mukohyama [[Coates et al.]](#ref-cmm) found power-law divergences in the difference of the sound speeds of a gauge field and a scalar in Hořava gravity, which spoil that suppression unless the matter sector is also modified at high energy. Groot Nibbelink and Pospelov [[Groot Nibbelink–Pospelov]](#ref-gnp) showed that in the supersymmetric Standard Model supersymmetric Lorentz-violating operators have dimension at least five, and Redigolo [[Redigolo]](#ref-redigolo) that in renormalisable Lorentz-violating supersymmetric gauge theories with linear supercharges the only violation sits in a constant $$c$$ of the supersymmetry algebra, which does not renormalise. This page adds a classification of gauge-invariant two-derivative forms by symmetric tensors, which reduces the counting of free speeds to representation theory; a proof that the one-loop coefficient is independent of the shape of the regulator; and the structure of effective Hamiltonians on stabiliser codes.

Exact Lorentz symmetry of a discrete substrate is the subject of [quantum sprinklings](/quantum-gravity/quantum-sprinklings/) and [Lorentz-invariant point sets](/quantum-gravity/invariant-point-sets/); [exact discrete Lorentz groups](/quantum-gravity/discrete-lorentz-group/) treat non-compact residual groups, which are not considered below. Frames that are properties of a state, such as the velocity of a cluster of particles, appear in [frames and reflection positivity](/quantum-gravity/frames-and-reflection-positivity/), and the same question for gravitons on a lattice in [lattice diffeomorphisms](/quantum-gravity/lattice-diffeomorphisms/).

## 1. What a residual symmetry group fixes

A free scalar with a frame has the action $$\tfrac12 a\,\dot\phi^2-\tfrac12 b\,(\nabla\phi)^2$$ and speed $$c^2=b/a$$. Lorentz invariance sets $$a=b$$. If only a subgroup $$K$$ of the Lorentz group survives, the allowed coefficient tensors are the $$K$$-invariant symmetric bilinear forms on momentum space. A compact $$K$$ always admits two independent ones, the Minkowski form $$\eta$$ and a positive-definite form obtained by averaging over $$K$$, so $$b/a$$ is a free parameter of the quadratic action of each scalar species. For spin 2, gauge invariance restricts the allowed forms, and the question is how many survive.

Write a real quadratic form in a symmetric tensor $$h_{\mu\nu}$$ and a momentum $$k_\mu$$ on $$\mathbb R^4$$ as a polynomial $$Q(h,k)$$ of degree 2 in the ten components of $$h$$ and degree 2 in $$k$$; this is a two-derivative action $$\int h(-k)^{T}M(k)\,h(k)$$ with $$M$$ quadratic in $$k$$. It is *gauge invariant* if

$$
Q\bigl(h+k\otimes\xi+\xi\otimes k,\;k\bigr)=Q(h,k)\qquad\text{for all }\xi\in\mathbb R^4. \tag{1}
$$

Let $$\mathcal V$$ be the space of gauge-invariant forms. A matrix $$g\in GL(4,\mathbb R)$$ acts by $$k\mapsto gk$$, $$h\mapsto ghg^{T}$$, and (1) is preserved, so $$\mathcal V$$ is a representation of $$GL(4,\mathbb R)$$. For a symmetric invertible matrix $$G^{\mu\nu}$$ define the Fierz–Pauli form with all contractions made with $$G$$:

$$
Q_G(h,k)=-\tfrac12\,(k\!\cdot\!k)\,h_{\mu\nu}h^{\mu\nu}+k^{\mu}h_{\mu\nu}\,k_{\lambda}h^{\lambda\nu}-(k^{\mu}k^{\nu}h_{\mu\nu})\,h+\tfrac12\,(k\!\cdot\!k)\,h^{2}, \tag{2}
$$

where indices are raised and contracted with $$G$$, $$k\!\cdot\!k=G^{\mu\nu}k_\mu k_\nu$$ and $$h=G^{\mu\nu}h_{\mu\nu}$$. For $$G=\eta$$ this is the linearised Einstein–Hilbert action, which satisfies (1). For a Lorentzian $$G$$ the form $$Q_G$$ is the Einstein–Hilbert form written in linear coordinates adapted to $$G$$, and a linear change of coordinates maps linearised gauge transformations to linearised gauge transformations. Both sides of (1) for $$Q_G$$ are polynomials in the entries of $$G$$, so the identity, valid on the open set of Lorentzian $$G$$, holds for every symmetric $$G$$. Each term of (2) contains three factors of $$G$$, so $$Q_{\lambda G}=\lambda^{3}Q_G$$. In the eigenvalue basis $$G=\mathrm{diag}(-1,c^2,c^2,c^2)$$, a transverse-traceless mode with polarisation $$h_{12}$$ and momentum $$k=(\omega,0,0,k_z)$$ has $$Q_G=c^{4}(\omega^{2}-c^{2}k_z^{2})$$, so $$Q_G$$ describes helicity-2 waves with speed $$c$$.

<div class="proposition" markdown="1">
**Proposition 1** (gauge-invariant two-derivative forms). Let $$\mathrm{Sym}^2$$ be the space of symmetric $$4\times4$$ matrices, on which $$O(4)$$ acts by $$s\mapsto g^{T}sg$$. The map $$\Phi:\mathrm{Sym}^2\to\mathcal V$$, $$\Phi(s)=\frac{d}{dt}Q_{1+ts}\Big\rvert_{t=0}$$, is an isomorphism of $$O(4)$$-representations. In particular $$\dim\mathcal V=10$$, and for every subgroup $$K\subset O(4)$$ the number of $$K$$-invariant gauge-invariant forms is

$$
n(K)=\dim\mathcal V^{K}=\dim(\mathrm{Sym}^2)^{K}. \tag{3}
$$

</div>

<div class="proof" markdown="1">
*Proof.* The identity $$Q_{G}(ghg^{T},gk)=Q_{g^{T}Gg}(h,k)$$ holds because every contraction in (2) is a contraction with $$G$$ of vectors transformed by $$g$$. For $$g\in O(4)$$ and $$G=1+ts$$ it gives $$\Phi(g^{T}sg)(h,k)=\Phi(s)(ghg^{T},gk)$$, so $$\Phi$$ is equivariant.

Next, $$\Phi$$ is injective. The space $$\mathrm{Sym}^2=\mathbb R\,1\oplus\mathrm{Sym}^2_0$$ splits into the trace part and the nine-dimensional traceless part, which are irreducible and not isomorphic, so the kernel of $$\Phi$$ is a sum of a subspace of each. By homogeneity $$\Phi(1)=3Q_1\neq0$$. On the traceless part, label the components of $$k$$ by $$(0,1,2,3)$$, call the first one $$\omega$$, and take the test mode above with the diagonal $$G=1+ts$$, $$s=\mathrm{diag}(3,-1,-1,-1)$$. Then $$Q_G=-(G^{00}\omega^{2}+G^{33}k_z^{2})\,G^{11}G^{22}$$ and $$\frac{d}{dt}Q_G\Big\rvert_{t=0}=-(3\omega^{2}-k_z^{2})+2(\omega^{2}+k_z^{2})=3k_z^{2}-\omega^{2}\neq0$$. Hence $$\ker\Phi=0$$.

Finally $$\dim\mathcal V=10$$. Expanding (1) in monomials of $$(h,k,\xi)$$ gives a homogeneous linear system with 1150 distinct equations for the $$55\times10=550$$ coefficients of $$Q$$ (55 monomials quadratic in the components of $$h$$, 10 quadratic in $$k$$). Its rank over $$\mathbb Q$$, computed in exact rational arithmetic, is 540, so the solution space has dimension $$550-540=10$$. Since $$\Phi$$ is injective and $$\dim\mathrm{Sym}^2=10$$, it is an isomorphism. $$\square$$
</div>

The identification of a ten-dimensional space of solutions with $$\mathrm{Sym}^2$$ is the content of the proposition; the dimension of $$\mathcal V$$ itself is a linear-algebra computation and no argument by hand is given here. For $$O(4)$$ the count can be done by hand: the $$O(4)$$-invariant forms of the required degree are spanned by $$k^2\,h_{\mu\nu}h_{\mu\nu}$$, $$k^2h^2$$, $$k_\mu h_{\mu\nu}h_{\nu\lambda}k_\lambda$$ and $$(k_\mu h_{\mu\nu}k_\nu)\,h$$, and (1) fixes their coefficients, up to a common factor, to $$-\tfrac12,\tfrac12,1,-1$$, which is (2) with $$G=1$$.

Table 1 lists $$n(K)$$ for the graviton and for a Maxwell field. For Maxwell, $$Q(A,k)$$ is quadratic in $$A_\mu$$ and in $$k$$ (100 coefficients), and the gauge transformation is $$A\mapsto A+k\,\xi$$ with scalar $$\xi$$. The columns "any form" count the invariants of the group without imposing gauge invariance. The groups are $$O(4)$$; the hypercubic group $$H(4)$$ of signed permutations of the four axes; $$O(3)\times T$$, spatial rotations together with time reversal $$k_0\mapsto-k_0$$ (with $$h_{0i}\mapsto-h_{0i}$$); and the cubic group of signed permutations of the three spatial axes times $$T$$. Every entry is the exact rank over $$\mathbb Q$$ of the linear conditions (generators of the Lie algebra for the continuous groups, generators of the group for the discrete ones). The gauge-invariant graviton entries agree with $$\dim(\mathrm{Sym}^2)^K$$ counted by hand: $$1$$ for $$O(4)$$ and $$H(4)$$; $$2$$ for $$O(3)\times T$$ and for the cubic group times $$T$$, the invariants being $$\delta_{ij}$$ and $$u_\mu u_\nu$$ with $$u$$ along the time axis. The Maxwell entry 20 for the trivial group is the number of quadratic forms in $$F_{\mu\nu}=k_\mu A_\nu-k_\nu A_\mu$$ (21) less the one proportional to $$F\wedge F$$, which vanishes for $$F=k\wedge A$$.

| group | graviton, any form | graviton, gauge invariant | Maxwell, any form | Maxwell, gauge invariant |
|---|---|---|---|---|
| none | 550 | 10 | 100 | 20 |
| $$O(4)$$ | 4 | 1 | 2 | 1 |
| $$H(4)$$ | 9 | 1 | 3 | 1 |
| $$O(3)\times T$$ | 17 | 2 | 6 | 2 |
| cubic $$\times\,T$$ | 26 | 2 | 7 | 2 |

*Table 1. Dimensions of the spaces of two-derivative quadratic forms invariant under the group.*

<div class="corollary" markdown="1">
**Corollary 2** (absolutely irreducible groups). Let $$K\subset O(4)$$ act on $$\mathbb R^4$$ absolutely irreducibly, as $$H(4)$$ does. Then $$(\mathrm{Sym}^2)^K=\mathbb R\,1$$, so every $$K$$-invariant gauge-invariant two-derivative graviton action is a multiple of $$Q_1$$, the Fierz–Pauli action with a single metric. The same holds for the scalar action $$c^{\mu\nu}k_\mu k_\nu$$, and, from Table 1, for a Maxwell field when $$K=H(4)$$.
</div>

<div class="proof" markdown="1">
*Proof.* A symmetric invariant bilinear form on an absolutely irreducible real representation is unique up to a factor, by Schur's lemma applied to the complexification. For $$H(4)$$ the commutant of the signed permutations is scalar: commuting with all sign changes forces a diagonal matrix, and commuting with all permutations forces equal entries. Equation (3) gives the graviton statement, and for scalars the allowed coefficients are exactly $$(\mathrm{Sym}^2)^K$$. The Maxwell statement is the entry of Table 1. $$\square$$
</div>

All three species then share the tensor $$1$$, so in Euclidean signature the relative weight of $$\partial_0^2$$ and $$\partial_i^2$$ is the same for each of them. Since $$-1\in H(4)$$, every odd-rank invariant tensor vanishes, which removes all operators with an odd number of indices; the only rank-two invariant is $$\delta_{\mu\nu}$$, so there is no dimension-four operator $$c^{\mu\nu}\bar\psi\gamma_\mu\partial_\nu\psi$$ with traceless $$c^{\mu\nu}$$. The first operators that distinguish $$H(4)$$ from $$O(4)$$ are built from the rank-four invariant $$\sum_\mu e_\mu^{\otimes4}$$: $$\phi\sum_\mu\partial_\mu^4\phi$$ for a scalar (four derivatives) and $$\bar\psi\sum_\mu\gamma_\mu\partial_\mu^3\psi$$ for a Dirac fermion (three). A common limiting speed in real time requires in addition that the Euclidean theory be reflection positive and that the linearised gauge invariance persist at interacting level. If the gauge invariance is only approximate, Table 1 shows what opens up: the $$H(4)$$-invariant graviton forms number nine, of which one is gauge invariant.

<div class="corollary" markdown="1">
**Corollary 3** (compact groups in real time). Let $$K\subset O(3,1)$$ be compact, acting on momentum space $$\mathbb R^4$$. Then $$n(K)\ge2$$ for the graviton, and the scalar and Maxwell quadratic actions likewise have at least two independent forms. The quadratic action of each species then has a parameter in addition to its normalisation, and nothing in the symmetry relates these parameters for different species.
</div>

<div class="proof" markdown="1">
*Proof.* Averaging $$g^{T}g$$ over $$K$$ with Haar measure gives a $$K$$-invariant positive-definite form $$\Omega$$. In a basis in which $$\Omega=1$$ the group lies in $$O(4)$$, and the Minkowski form becomes a symmetric matrix $$\eta'$$ that is invariant under $$K$$ and indefinite, hence not proportional to $$1$$. Both $$1$$ and $$\eta'$$ lie in $$(\mathrm{Sym}^2)^K$$, so (3) gives $$n(K)\ge2$$; the number $$n(K)$$ does not depend on the basis, because $$\mathcal V$$ is stable under $$GL(4,\mathbb R)$$. For scalars the invariant coefficient tensors are the same two matrices. For Maxwell, rotate within $$O(4)$$ so that $$\eta'=\mathrm{diag}(g_0,\dots,g_3)$$; the forms $$Q^{\mathrm{Max}}_{1}$$ and $$Q^{\mathrm{Max}}_{\eta'}$$ (defined in Proposition 4) are $$K$$-invariant, and they are not proportional, because the coefficient of $$F_{\mu\nu}^2$$ in $$Q^{\mathrm{Max}}_{\eta'}$$ is $$-\tfrac12g_\mu g_\nu$$ and equal products $$g_\mu g_\nu$$ for all six pairs force all $$g_\mu$$ equal, which an indefinite $$\eta'$$ excludes. $$\square$$
</div>

For $$K=SO(3)$$, with or without $$T$$, the additional parameter is the speed (Proposition 4). Corollary 3 describes what the symmetry allows in the free action. It does not say that the speeds differ, only that their equality would be an input. Interactions and perturbations then act on that input, as sections 4 and 5 compute. A symmetry that mixes species, such as supersymmetry, is outside the corollary.

## 2. Vacua without boosts, and supersymmetry

A Hořava-type vacuum is invariant under time and space translations and spatial rotations, possibly with time reversal. Call this residual group $$\Gamma$$. We assume (B1) that the graviton has a linearised gauge invariance between foliation-preserving diffeomorphisms and the full linearised diffeomorphisms, (B2) that the photon has a $$U(1)$$ gauge invariance, and (B3) that the quadratic action has no cross term between $$h$$ and $$A$$, as follows for instance from charge conjugation $$A\mapsto-A$$.

<div class="proposition" markdown="1">
**Proposition 4** (speeds without boosts). Under (B1) to (B3):

(a) For every pair $$c_T,c_\gamma>0$$ the quadratic action $$Q_{G_{c_T}}(h,k)+Q^{\mathrm{Max}}_{G_{c_\gamma}}(A,k)$$, with $$G_c=\mathrm{diag}(-1,c^2,c^2,c^2)$$ and $$Q^{\mathrm{Max}}_G=-\tfrac14F_{\mu\nu}F_{\rho\sigma}G^{\mu\rho}G^{\nu\sigma}$$, is $$\Gamma$$-invariant and gauge invariant under the full linearised diffeomorphisms, hence under the foliation-preserving subgroup. Its helicity-2 modes propagate with speed $$c_T$$ and its helicity-1 modes with speed $$c_\gamma$$. The assumed symmetries therefore leave $$c_T/c_\gamma$$ unconstrained.

(b) Let $$U$$ be a unitary operator that commutes with the Hamiltonian $$H$$, the momentum $$\vec P$$ and the angular momentum $$\vec J$$. Then $$U$$ maps the one-particle states of momentum $$\vec p$$ and helicity $$\lambda$$ to states with the same $$\vec p$$ and $$\lambda$$. It therefore cannot map graviton states ($$\lvert\lambda\rvert=2$$) to photon states ($$\lvert\lambda\rvert=1$$): a symmetry that preserves helicity cannot intertwine the two sectors.
</div>

<div class="proof" markdown="1">
*Proof.* (a) Invariance under translations and rotations holds because $$G_c$$ is rotation invariant and constant, and $$T$$ acts trivially on it. Gauge invariance of $$Q_G$$ was shown in section 1, and the Maxwell form is invariant under $$A\mapsto A+k\xi$$ because it depends on $$F$$ only. The dispersion relation of transverse modes is $$G^{\mu\nu}k_\mu k_\nu=0$$, that is $$\omega=c\lvert\vec k\rvert$$, for each sector with its own $$c$$.

(b) The helicity of a one-particle state with momentum $$\vec p$$ is the eigenvalue of $$\vec J\cdot\hat p$$. If $$U$$ commutes with $$\vec P$$ and $$\vec J$$ it commutes with $$\vec J\cdot\hat P$$ as well, and with $$H$$ it preserves the energy. $$\square$$
</div>

In a Lorentz-invariant vacuum the relation between the two speeds comes from the boosts, which do not commute with $$\vec J$$; part (b) records that the symmetries remaining here cannot replace them. Part (b) does not exclude equal speeds as a coincidence of the couplings; it excludes them as a consequence of such a symmetry. Symmetries that transform non-trivially under rotations would extend the algebra beyond $$\Gamma$$ and are not constructed here. A fermionic symmetry is an exception to the argument, because it can change the helicity by one half.

Exact supersymmetry is of this type. Suppose that a single algebra

$$
\{\mathcal Q_\alpha,\mathcal Q^\dagger_\beta\}=2\bigl(H\,\delta_{\alpha\beta}+c\,\vec P\cdot\vec\sigma_{\alpha\beta}\bigr),\qquad \{\mathcal Q_\alpha,\mathcal Q_\beta\}=0, \tag{4}
$$

acts on all fields, with a real constant $$c>0$$ and Pauli matrices $$\vec\sigma$$. This is the supersymmetry algebra with the speed of light replaced by $$c$$, the form in which [[Redigolo]](#ref-redigolo) finds the only Lorentz violation of supersymmetric gauge theories with linear supercharges. The second relation of (4), the absence of a central charge, is used below. We also take $$[J_z,\mathcal Q_1]=-\tfrac12\mathcal Q_1$$ and $$[J_z,\mathcal Q_2]=+\tfrac12\mathcal Q_2$$, the transformation of a spinor, for momentum along the $$z$$ axis.

<div class="proposition" markdown="1">
**Proposition 5** (short multiplets have speed $$c$$). Assume (4). On the sector of fixed energy $$E$$ and momentum $$\vec p=p\hat z$$ with $$p>0$$, the operators satisfy $$\{\mathcal Q_1,\mathcal Q_1^\dagger\}=2(E+cp)$$ and $$\{\mathcal Q_2,\mathcal Q_2^\dagger\}=2(E-cp)$$, and all other anticommutators among $$\mathcal Q_1,\mathcal Q_2,\mathcal Q_1^\dagger,\mathcal Q_2^\dagger$$ vanish. Hence every state has $$E\ge cp$$. An irreducible multiplet has two states, with helicities $$(\lambda,\lambda-\tfrac12)$$, if and only if $$E=cp$$, and four states otherwise. If the graviton and the photon each lie in two-state multiplets, with helicities $$(2,\tfrac32)$$ and $$(1,\tfrac12)$$, then both have the dispersion relation $$E=c\lvert\vec p\rvert$$, and $$c_{\mathrm{GW}}=c_\gamma=c$$.
</div>

<div class="proof" markdown="1">
*Proof.* With $$\vec p$$ along $$z$$, $$\vec p\cdot\vec\sigma=p\,\sigma_z$$ is diagonal, which gives the two anticommutators and shows that $$\{\mathcal Q_1,\mathcal Q_2^\dagger\}=0$$; the relations $$\{\mathcal Q_\alpha,\mathcal Q_\beta\}=0$$ are the second part of (4). In a positive-norm Hilbert space $$\lVert \mathcal Q_2^\dagger\psi\rVert^2+\lVert \mathcal Q_2\psi\rVert^2=2(E-cp)\lVert\psi\rVert^2\ge0$$, so $$E\ge cp$$. If $$E=cp$$ then $$\mathcal Q_2=\mathcal Q_2^\dagger=0$$ on the sector, and an irreducible multiplet is spanned by a state $$\lvert\lambda\rangle$$ annihilated by $$\mathcal Q_1^\dagger$$ and its image $$\mathcal Q_1\lvert\lambda\rangle$$, which is non-zero because $$\lVert\mathcal Q_1\lvert\lambda\rangle\rVert^2=4cp\lVert\lambda\rVert^2$$ and has helicity lower by one half: two states. If $$E>cp$$, the operators $$\mathcal Q_1/\sqrt{2(E+cp)}$$ and $$\mathcal Q_2^\dagger/\sqrt{2(E-cp)}$$ are the creation operators of two fermionic oscillators, since (4) gives the canonical anticommutation relations. An irreducible multiplet is spanned by a state $$\lvert\lambda\rangle$$ annihilated by $$\mathcal Q_1^\dagger$$ and $$\mathcal Q_2$$ together with its three images, with helicities $$(\lambda,\lambda-\tfrac12,\lambda-\tfrac12,\lambda-1)$$. A graviton in a two-state multiplet therefore has $$E=cp$$, and so does a photon. $$\square$$
</div>

The speed of every short multiplet is the constant $$c$$ of the algebra and does not depend on a coefficient in the action. Soft supersymmetry breaking leaves a residual difference of order $$m_{\mathrm{soft}}^{2}/\Lambda_{\mathrm{LV}}^{2}$$, where $$\Lambda_{\mathrm{LV}}$$ is the scale at which the Lorentz-violating terms appear; this estimate is quoted in [[Coates et al.]](#ref-cmm) and is not derived here. Whether a boost-free gravity theory with a supersymmetry of the form (4) acting on the graviton exists is not settled by this analysis.

## 3. A frame cannot be averaged away

A construction on a point set that uses a unit timelike vector $$u$$ breaks Lorentz invariance unless its observables do not depend on $$u$$. One may try to restore invariance by drawing $$u$$ at random. No normalised random choice is Lorentz invariant.

<div class="proposition" markdown="1">
**Proposition 6** (no invariant measure on frames). Let $$d\ge2$$ and let $$H^{d-1}$$ be the set of future unit timelike vectors in $$\mathbb R^{1,d-1}$$. There is no $$SO^{+}(1,d-1)$$-invariant Borel probability measure on $$H^{d-1}$$.
</div>

<div class="proof" markdown="1">
*Proof.* For $$u\in H^{d-1}$$ we have $$(u^{0})^{2}-(u^{1})^{2}=1+\lvert u_\perp\rvert^{2}>0$$, so the rapidity $$\theta(u)=\operatorname{artanh}(u^{1}/u^{0})$$ is defined and continuous. A boost $$B_t$$ in the $$(0,1)$$ plane maps $$(u^0,u^1)$$ to $$(\cosh t\,u^{0}+\sinh t\,u^{1},\ \sinh t\,u^{0}+\cosh t\,u^{1})$$ and so shifts $$\theta$$ by $$t$$. If $$\mu$$ were invariant, the image measure $$\nu=\theta_*\mu$$ on $$\mathbb R$$ would satisfy $$\nu(A+t)=\nu(A)$$ for all $$t$$. Then $$\nu([n,n+1))$$ takes the same value $$a$$ for every integer $$n$$, and $$\sum_n\nu([n,n+1))=1$$ is impossible for a constant $$a$$, whether $$a=0$$ or $$a>0$$. $$\square$$
</div>

The invariant (Haar) measure on $$H^{d-1}$$ is the hyperbolic volume; it exists and has infinite total mass. An average over it is defined only for observables that are integrable over the rapidity, and the proposition says that no choice of normalisation turns it into a probability measure.

<div class="corollary" markdown="1">
**Corollary 7** (no equivariant frame from a Lorentz-invariant point process). Let $$\mathcal P$$ be a random locally finite subset of $$\mathbb R^{1,d-1}$$ whose law is invariant under $$SO^{+}(1,d-1)$$, for example a Poisson process. There is no measurable function $$u(\mathcal P)\in H^{d-1}$$ of the point set with $$u(g\mathcal P)=g\,u(\mathcal P)$$ almost surely for each $$g$$.
</div>

<div class="proof" markdown="1">
*Proof.* If there were one, the law $$\mu$$ of $$u(\mathcal P)$$ would satisfy $$\mu(gA)=\Pr[g^{-1}u(\mathcal P)\in A]=\Pr[u(g^{-1}\mathcal P)\in A]=\mu(A)$$, because $$g^{-1}\mathcal P$$ has the same law as $$\mathcal P$$. It would therefore be an invariant probability measure, which contradicts Proposition 6. $$\square$$
</div>

Corollary 7 is the argument of Bombelli, Henson and Sorkin [[Bombelli–Henson–Sorkin]](#ref-bhs), who prove related statements for sprinklings, in its simplest form. It gives a trichotomy for any matter construction on a Lorentz-invariant point set that uses a timelike axis. The axis may be drawn independently of the points; its law is then not invariant by Proposition 6, and the construction selects a frame. The axis may be a function of the points; Corollary 7 forbids this when the function is equivariant and measurable. Or the observables may be independent of the axis in each realisation. In Euclidean signature the corresponding group $$SO(d)$$ is compact, a Haar average over the axis exists, and the obstruction does not arise; whether such an averaged Euclidean construction defines a Lorentz-invariant real-time theory is the question of reflection positivity, which is not decided here. The construction of fermions on a sprinkling, where a frame is needed to define a kinetic term, appears in [order-only fermions](/quantum-gravity/order-only-fermions/).

## 4. A frame in the matter regulator

We compute how a frame in a regulator is transferred to a matter species. The model is a Yukawa theory in which the scalar line carries the regulator.

Work in Euclidean $$\mathbb R^4$$ with a unit vector $$u$$ along the time axis. A massless Dirac fermion $$\psi$$ with propagator $$S(p)=1/(i(\gamma\!\cdot\!p))=-i(\gamma\!\cdot\!p)/p^{2}$$ couples to a real scalar through $$g\,\phi\bar\psi\psi$$. The scalar propagator is replaced by

$$
D(q)=\frac{f(\lvert q_\perp\rvert/\Lambda)}{q^{2}+M^{2}},\qquad q_\perp=q-(q\!\cdot\!u)u, \tag{5}
$$

so that the frame enters only through the cutoff on the momentum orthogonal to $$u$$. This is a prescription for a propagator and not an action; it is all that a one-loop self-energy needs. We take $$f(x)=F(x^{2})$$ with $$F$$ continuously differentiable on $$[0,\infty)$$, $$F(0)=1$$, and $$F(s)$$, $$sF'(s)$$ bounded by $$C(1+s)^{-a}$$ for some $$a>0$$. Examples are $$e^{-x^{2}}$$, $$e^{-x^{4}}$$ and $$(1+x^{2})^{-2}$$. The scalar has speed 1 for momenta much below $$\Lambda$$, and the question is the speed of the fermion.

The one-loop inverse propagator of $$\psi$$ is

$$
W(p)=i(\gamma\!\cdot\!p)-g^{2}\!\int\!\frac{d^{4}q}{(2\pi)^{4}}\,S(p-q)\,D(q)=i(\gamma\!\cdot\!p)+ig^{2}\!\int\!\frac{d^{4}k}{(2\pi)^{4}}\,\frac{(\gamma\!\cdot\!k)}{k^{2}}\,D(p-k). \tag{6}
$$

For $$M>0$$ the integral converges absolutely and is continuously differentiable in $$p$$. We need its derivative at $$p=0$$. Differentiating $$D(p-k)$$, using that $$D$$ is even, and integrating by parts (the boundary terms at $$k=0$$ and at infinity vanish) gives

$$
\partial_{p_\nu}W\Big\rvert_{p=0}=i\gamma_\nu+ig^{2}\!\int\!\frac{d^{4}k}{(2\pi)^{4}}\,D(k)\Bigl(\frac{\gamma_\nu}{k^{2}}-\frac{2(\gamma\!\cdot\!k)\,k_\nu}{k^{4}}\Bigr). \tag{7}
$$

Define $$A=\int D/k^{2}$$, $$B_\parallel=\int D\,(k\!\cdot\!u)^{2}/k^{4}$$ and $$B_\perp=\tfrac13\int D\,k_\perp^{2}/k^{4}$$, all with measure $$d^{4}k/(2\pi)^{4}$$. Rotation invariance about $$u$$ turns (7) into

$$
W(p)=i\bigl[Z_\parallel\,(p\!\cdot\!u)\,(\gamma\!\cdot\!u)+Z_\perp\,(\gamma\!\cdot\!p_\perp)\bigr]+o(\lvert p\rvert),\qquad Z_\parallel=1+g^{2}(A-2B_\parallel),\quad Z_\perp=1+g^{2}(A-2B_\perp). \tag{8}
$$

For $$u$$ the time axis, the pole of the propagator is at $$Z_\parallel^{2}p_0^{2}+Z_\perp^{2}\vec p^{\,2}=0$$, that is at $$E=(Z_\perp/Z_\parallel)\lvert\vec p\rvert$$ for $$p_0=iE$$. The fermion speed relative to the scalar is therefore

$$
\delta c=\frac{Z_\perp}{Z_\parallel}-1=2g^{2}\,(B_\parallel-B_\perp)+O(g^{4}). \tag{9}
$$

For an isotropic $$D$$ one has $$B_\parallel=B_\perp=A/4$$, so $$\delta c=0$$ and $$Z=1+g^{2}A/2$$, with $$A=\frac{1}{16\pi^{2}}\ln(\Lambda^{2}/M^{2})$$. This is the standard logarithmic wave-function factor $$(g^{2}/32\pi^{2})\ln\Lambda^{2}$$ of the Yukawa theory.

<div class="theorem" markdown="1">
**Theorem 8** (one-loop transfer from a frame in the scalar regulator). In the model (5), (6), let $$f$$ be in the class above and $$M>0$$. Then

$$
\delta c=\frac{g^{2}}{16\pi^{2}}\;C\!\left(\tfrac{M}{\Lambda};f\right)+O(g^{4}),\qquad \lim_{M/\Lambda\to0}C\!\left(\tfrac{M}{\Lambda};f\right)=\frac13 . \tag{10}
$$

Thus $$\delta c\to g^{2}/48\pi^{2}$$ for every such $$f$$. For the Gaussian $$f=e^{-x^{2}}$$ the correction is $$O\bigl((M/\Lambda)^{2}\ln(\Lambda/M)\bigr)$$. For a regulator that depends on the full four-momentum, $$f=f(\lvert q\rvert/\Lambda)$$, $$C=0$$.
</div>

<div class="proof" markdown="1">
*Proof.* We write (9) in four-dimensional polar coordinates about $$u$$. The regulator breaks isotropy only through the angle, and the shape of $$f$$ enters the radial integral through an additive constant that multiplies an angular average equal to zero.

*Step 1 (reduction to one angle and one radius).* Let $$\chi$$ be the angle between $$k$$ and $$u$$, so that $$k\!\cdot\!u=\lvert k\rvert\cos\chi$$, $$\lvert k_\perp\rvert=\lvert k\rvert\sin\chi$$ and $$d^{4}k=4\pi\,\lvert k\rvert^{3}\sin^{2}\chi\,d\chi\,d\lvert k\rvert$$. Then $$B_\parallel-B_\perp=\int D\,(k\!\cdot\!u)^{2}/k^{4}-\tfrac13\int D\,k_\perp^{2}/k^{4}$$ gives

$$
\delta c=\frac{2g^{2}}{4\pi^{3}}\int_{0}^{\pi}\!d\chi\;w(\chi)\,R(\sin\chi),\qquad w(\chi)=\sin^{2}\chi\Bigl(\cos^{2}\chi-\tfrac13\sin^{2}\chi\Bigr), \tag{11}
$$

with $$R(s)=\int_{0}^{\infty}\frac{\varrho\,d\varrho\,f(\varrho s/\Lambda)}{\varrho^{2}+M^{2}}$$. The weight $$w$$ is the quadrupole component of the angular distribution and has zero mean: $$\int_0^\pi w\,d\chi=\pi/8-\tfrac13\cdot3\pi/8=0$$.

*Step 2 (the radial integral).* Put $$t=\varrho s/\Lambda$$ and $$\tau=Ms/\Lambda$$, so that $$R(s)=\int_0^\infty t f(t)\,(t^{2}+\tau^{2})^{-1}dt$$. Split the range at $$t=1$$. On $$(0,1)$$ write $$f=1+(f-1)$$; since $$f(t)-1=O(t^{2})$$, the second term gives a convergent limit as $$\tau\to0$$ by dominated convergence, and the first gives $$\tfrac12\ln\bigl((1+\tau^{2})/\tau^{2}\bigr)$$. On $$(1,\infty)$$ the integral converges to $$\int_{1}^{\infty}f(t)\,dt/t$$, finite because $$f$$ decays. Hence

$$
R(s)=\ln\frac{\Lambda}{M}-\ln s+b_f+r(\tau),\qquad b_f=\int_0^1\frac{f-1}{t}\,dt+\int_1^\infty\frac{f}{t}\,dt, \tag{12}
$$

with $$r(\tau)\to0$$ as $$\tau\to0$$. Since $$\tau\le M/\Lambda$$, the remainder is uniformly small on $$0<s\le1$$ when $$M/\Lambda$$ is small. The constant $$b_f$$ and the term $$\ln(\Lambda/M)$$ do not depend on $$s$$, hence do not depend on $$\chi$$.

*Step 3 (the angular integral).* Insert (12) in (11). The $$\chi$$-independent terms multiply $$\int w\,d\chi=0$$ and drop out. What remains is $$-\int_0^\pi w(\chi)\ln\sin\chi\,d\chi$$ plus a term that tends to zero. With $$w=\sin^{2}\chi-\tfrac43\sin^{4}\chi$$ and $$J(a)=\int_0^\pi\sin^{a}\chi\,d\chi=\sqrt\pi\,\Gamma(\tfrac{a+1}{2})/\Gamma(\tfrac a2+1)$$ one has $$\int_0^\pi\sin^{a}\chi\ln\sin\chi\,d\chi=J'(a)$$. Using the digamma values $$\Psi(\tfrac32)=2-\gamma_{\mathrm E}-2\ln2$$, $$\Psi(2)=1-\gamma_{\mathrm E}$$, $$\Psi(\tfrac52)=\tfrac83-\gamma_{\mathrm E}-2\ln2$$ and $$\Psi(3)=\tfrac32-\gamma_{\mathrm E}$$,

$$
J'(2)=\frac\pi4\,(1-2\ln2),\qquad J'(4)=\frac{3\pi}{16}\Bigl(\frac76-2\ln2\Bigr),\qquad \int_0^\pi w\ln\sin\chi\,d\chi=J'(2)-\tfrac43J'(4)=-\frac{\pi}{24}.
$$

Therefore $$\delta c\to\frac{2g^{2}}{4\pi^{3}}\cdot\frac{\pi}{24}=\frac{g^{2}}{48\pi^{2}}$$, which is (10) with $$C=1/3$$. For the Gaussian, $$R(s)=\tfrac12e^{\tau^{2}}E_1(\tau^{2})$$ exactly, and its expansion gives $$r=O(\tau^{2}\ln\tau)$$. If $$f$$ depends on $$\lvert k\rvert$$ the radial integral does not depend on $$\chi$$ and (11) vanishes by $$\int w=0$$. $$\square$$
</div>

An isotropic regulator gives zero because the integrand of (9) is a quadrupole, whose angular average vanishes at each radius. A regulator that cuts off only the momentum orthogonal to $$u$$ removes more of the large-$$\lvert q_\perp\rvert$$ region at angles near $$\pi/2$$ than near the axis, and the induced angular dependence is the logarithm $$-\ln\sin\chi$$ of the infrared-sensitive radial integral. The shape of $$f$$ only changes the constant $$b_f$$, which the zero-mean weight removes.

Numerically, $$C=\delta c/(g^2/16\pi^2)$$ from (11) with the radial integral done by quadrature is

| $$M/\Lambda$$ | $$10^{-1}$$ | $$10^{-2}$$ | $$10^{-3}$$ | $$10^{-4}$$ |
|---|---|---|---|---|
| $$e^{-x^{2}}$$ | 0.3256 | 0.3332 | 0.3333 | 0.3333 |
| $$e^{-x^{4}}$$ | 0.3305 | 0.3333 | 0.3333 | 0.3333 |
| $$(1+x^{2})^{-2}$$ | 0.3209 | 0.3331 | 0.3333 | 0.3333 |

*Table 2. The coefficient $$C(M/\Lambda;f)$$.*

A two-dimensional quadrature of the definitions of $$B_\parallel$$ and $$B_\perp$$ in cylindrical coordinates $$(k\!\cdot\!u,\lvert k_\perp\rvert)$$ for the Gaussian at $$M/\Lambda=10^{-2}$$ gives $$0.33318$$, the value that (11) gives at the same ratio.

The theorem has two limits. With $$M=0$$ the double integral defining $$B_\parallel-B_\perp$$ is only conditionally convergent: integrating over the component along $$u$$ first at fixed $$\lvert k_\perp\rvert$$ gives zero, while the limit $$M/\Lambda\to0$$ of the absolutely convergent integral gives $$1/3$$. The mass must be kept until the end. The result also holds for a regulator on the scalar line only. The model of [[Collins et al.]](#ref-collins), in which the regulator multiplies the fermion propagator and the quantity computed is the scalar self-energy, gives a Lorentz-violating coefficient $$\xi=(g^2/6\pi^2)\bigl[1+2\int_0^\infty x f'(x)^2dx\bigr]$$ that depends on the shape of $$f$$. Regulators placed on other lines, or on vertices, give other coefficients, and the universality proved here is universality in the shape of $$f$$ for this placement.

For a gauge theory only an estimate is available. The coefficient multiplies a dimension-four operator, $$c^{\mu\nu}\bar\psi\gamma_\mu\partial_\nu\psi$$ with $$c^{\mu\nu}\propto u^{\mu}u^{\nu}-\tfrac14\delta^{\mu\nu}$$, and it does not depend on $$\Lambda$$, so a cutoff at the Planck scale gives no suppression. Replacing $$g^{2}$$ by $$e^{2}$$ gives $$\delta c\sim\alpha/12\pi=1.9\times10^{-4}$$ for $$\alpha=1/137.036$$. This is an order-of-magnitude replacement and not a computation: the factor from the gauge structure of the photon line is not included, and in a theory with several couplings the coefficients of different species need not coincide. The data tables of [[Kostelecký–Russell]](#ref-kr) list bounds on dimension-four coefficients of electrons, protons and neutrons that lie many orders of magnitude below $$10^{-10}$$, so a coefficient near $$10^{-4}$$ would need a cancellation by many orders of magnitude between independently renormalised parameters. The GW170817 observation bounds the difference between the speeds of gravity and light to between $$-3\times10^{-15}$$ and $$+7\times10^{-16}$$ of the speed of light [[LIGO–Virgo et al.]](#ref-gw170817); the Yukawa computation concerns a fermion and a scalar, and applying it to the photon and the graviton would need the corresponding computation on those lines.

No internal symmetry forbids the operator, since it is a singlet. Exact Lorentz invariance of the regulator removes it; so does exact $$H(4)$$ invariance in Euclidean signature (section 1, where the only rank-two invariant is $$\delta_{\mu\nu}$$), and a supersymmetry with one algebra (4) fixes the speed within short multiplets (section 2). Other symmetries are not examined here.

## 5. Code distance does not equalise speeds

A different proposal is that Lorentz invariance in the infrared might be protected by an error-correcting structure: Lorentz-violating terms would be logical errors and so suppressed by the code distance, while the kinetic terms are not. We test this for stabiliser codes. A species is pictured as a lattice field whose degrees of freedom are logical qubits of a code, with hopping amplitudes given by logical couplings; its speed is a function of those couplings, and the question is how the code distance enters that function. The example at the end of the section is a chain of this kind.

Let $$\mathcal S$$ be an abelian subgroup of the Pauli group on $$n_{\mathrm q}$$ qubits with $$-1\notin\mathcal S$$, with generators $$S_s$$, code space $$\mathcal C$$ (the joint $$+1$$ eigenspace) and projector $$\Pi$$. The distance $$d_{\mathrm c}$$ is the least weight of an element of the normaliser $$N(\mathcal S)$$ that is not in $$\mathcal S$$. Let $$H_0=-\Delta\sum_sS_s$$ with $$\Delta>0$$ and $$V=\sum_t\lambda_t P_t$$ with $$P_t$$ Pauli strings and $$\lambda_t$$ arbitrary couplings. The low-energy Hamiltonian on $$\mathcal C$$ is the Schrieffer–Wolff series $$H_{\mathrm{eff}}=\Pi e^{\mathcal T}(H_0+V)e^{-\mathcal T}\Pi$$ [[Bravyi–DiVincenzo–Loss]](#ref-bdl), defined as a formal power series in the $$\lambda_t$$. We use only the following property of that series, shared by the Schrieffer–Wolff, Kato and Rayleigh–Schrödinger expansions: the term of order $$N$$ is a linear combination, with coefficients independent of the $$\lambda_t$$, of operators

$$
\Pi\,P_{t_1}f_1(\mathbf S)\,P_{t_2}f_2(\mathbf S)\cdots f_{N-1}(\mathbf S)\,P_{t_N}\,\Pi, \tag{13}
$$

where each $$f_i$$ is a function of the commuting stabilisers (energy denominators and projectors on syndrome sectors). Expand $$H_{\mathrm{eff}}=\sum_La_L(\lambda)\,L$$ in a basis of logical Pauli operators, one representative $$L$$ for each class in $$N(\mathcal S)/\langle\mathcal S\rangle$$.

<div class="proposition" markdown="1">
**Proposition 9** (disjoint monomial supports). Every monomial $$\prod_t\lambda_t^{m_t}$$ that appears in $$a_L$$ satisfies $$[\prod_tP_t^{m_t}]=[L]$$ in $$N(\mathcal S)/\langle\mathcal S\rangle$$. Hence $$a_L$$ and $$a_{L'}$$ for distinct classes have no monomial in common. If $$V$$ consists of single-qubit Paulis, every monomial in $$a_L$$ has degree at least the least weight in the class $$[L]$$, so at least $$d_{\mathrm c}$$ for $$L\notin\langle\mathcal S\rangle$$.
</div>

<div class="proof" markdown="1">
*Proof.* Between code projectors the stabilisers act as numbers, so only the Pauli product of an operator (13) matters, and a Pauli product determines its logical class.

A Pauli string either commutes or anticommutes with each stabiliser, so a function $$f(\mathbf S)$$ can be moved through $$P_t$$ at the price of flipping the arguments by signs. Pushing all $$f_i$$ to the right in (13) and acting on $$\Pi$$, where each stabiliser equals $$+1$$, turns (13) into a number times $$\Pi R\Pi$$ with $$R=P_{t_1}\cdots P_{t_N}$$, a Pauli string up to phase. If $$R$$ anticommutes with some $$S_s$$, then $$\Pi R\Pi=0$$. Otherwise $$R\in N(\mathcal S)$$ and $$\Pi R\Pi=R\,\Pi=(\text{phase})\,L_R\Pi$$, where $$L_R$$ is the logical representative of the class $$[R]$$. The product $$R$$ depends on the multiset $$\{t_i\}$$ only up to a phase, hence so does its class, and the monomial $$\lambda_{t_1}\cdots\lambda_{t_N}$$ feeds only $$a_{L_R}$$. For single-qubit $$V$$ the number of factors needed to build a string in a class is at least the weight of that class. $$\square$$
</div>

Proposition 9 gives a lower bound on the degree. Whether the coefficient at exactly that degree is non-zero depends on the code and on the perturbation, because sums over orderings with energy denominators can cancel; it is an assumption in Proposition 11 below. The statement about speeds concerns the logical couplings: a species whose dynamics is carried by logical qubits has a speed determined by the coefficients $$a_L$$ of its logical Hamiltonian, and Proposition 9 says which microscopic couplings feed which $$a_L$$.

<div class="corollary" markdown="1">
**Corollary 10** (decoupled species). Suppose that the qubits split into two blocks, that $$\mathcal S$$ is generated by stabilisers each supported in one block, and that $$V=V^{(1)}+V^{(2)}$$ with the two terms acting on the respective blocks and containing disjoint sets of couplings $$\lambda^{(1)},\lambda^{(2)}$$. Then $$H_{\mathrm{eff}}=H^{(1)}_{\mathrm{eff}}(\lambda^{(1)})+H^{(2)}_{\mathrm{eff}}(\lambda^{(2)})$$, and the speeds are $$c_\alpha=F_\alpha(\lambda^{(\alpha)})$$. If $$c_1=c_2$$ holds on an open set of couplings, then both $$F_1$$ and $$F_2$$ are constant there. A relation between non-constant speeds is a restriction on the couplings, that is a tuning, or follows from a symmetry that maps the couplings of one species onto those of the other.
</div>

The splitting holds because $$H_0$$ and $$V$$ are sums of commuting block terms, so the Schrieffer–Wolff unitary is a product of unitaries each acting on its own block, and $$\Pi$$ is the product of the block projectors. An open set contains a product neighbourhood, which gives the statement about constants. The distance $$d_{\mathrm c}$$ enters only through the functions $$F_\alpha$$.

<div class="proposition" markdown="1">
**Proposition 11** (amplification). Let species 1 and 2 be as in Corollary 10, with $$\epsilon$$ the ratio of the perturbation to the gap $$\Delta$$. Assume that for each species the speed is set by the coefficient $$a=a_{L_*}$$ of one logical operator $$L_*\notin\langle\mathcal S\rangle$$,

$$
c_\alpha=\kappa\,a(\lambda^{(\alpha)})\bigl(1+\rho(\lambda^{(\alpha)})\bigr),
$$

where $$\kappa$$ is the same constant for both species, $$\rho$$ is smooth with $$\rho=O(\epsilon^{2})$$ and $$\lambda\cdot\nabla\rho=O(\epsilon^{2})$$, and $$a$$ has a non-zero homogeneous part $$a^{(k)}$$ of least degree $$k$$ with $$a=a^{(k)}(1+O(\epsilon^2))$$. If the couplings of species 2 are those of species 1 multiplied by $$1+\zeta$$, then

$$
\ln\frac{c_2}{c_1}=k\,\ln(1+\zeta)+O(\epsilon^{2}\zeta), \tag{14}
$$

and $$k\ge d_{\mathrm c}$$ when $$V$$ is a sum of single-qubit terms.
</div>

<div class="proof" markdown="1">
*Proof.* The leading part $$a^{(k)}$$ is homogeneous of degree $$k$$ in the $$\lambda$$, hence is multiplied by $$(1+\zeta)^{k}$$. The higher-degree parts of $$a$$ and the factor $$1+\rho$$ change the ratio by a relative amount $$O(\epsilon^{2}\zeta)$$ by the bounds on $$\rho$$ and on $$a/a^{(k)}-1$$. The bound $$k\ge d_{\mathrm c}$$ is Proposition 9. $$\square$$
</div>

So a 1% difference in the physical couplings of the two species produces a difference of about $$k\%$$ in their speeds. The distance reduces the absolute size of the logical couplings, here as $$\epsilon^{d_{\mathrm c}}$$, and does not reduce their ratios.

*Example (repetition-code chain).* Each species is a chain of blocks of $$d_{\mathrm c}$$ qubits with an intra-block coupling $$-\Delta\,Z_iZ_{i+1}$$ (the stabilisers), a field $$-h\,X_i$$ on every qubit, and an inter-block coupling $$-J\,Z Z$$. The logical $$\bar X=\prod X_i$$ has weight $$d_{\mathrm c}$$ and is generated at order $$d_{\mathrm c}$$ in $$h$$, so the effective chain is a transverse-field Ising chain on logical qubits with field of order $$h^{d_{\mathrm c}}/\Delta^{d_{\mathrm c}-1}$$. The Jordan–Wigner transform gives a free Majorana chain whose quasiparticle energy is $$2\sigma(k)$$, with $$\sigma(k)$$ the smallest singular value of the $$d_{\mathrm c}\times d_{\mathrm c}$$ Bloch matrix with diagonal $$h$$, superdiagonal $$-\Delta$$ and corner entry $$-Je^{ik}$$. The chain is gapless at $$h^{d_{\mathrm c}}=\Delta^{d_{\mathrm c}-1}J$$, which is the tuning that the speed-setting coupling needs in either scenario, and the speed per unit cell is $$v=2\,d\sigma/dk$$ at $$k=0$$. The speed-setting coupling is the order-$$d_{\mathrm c}$$ coefficient of $$\bar X$$, proportional to $$h^{d_{\mathrm c}}$$, and criticality ties $$J$$ to it; the asymmetry between the species is therefore introduced in $$h$$, with $$J$$ re-tuned. With $$\epsilon=h/\Delta=0.3$$ we find $$v=2\,\epsilon^{d_{\mathrm c}}\Delta$$ times $$0.9107,\,0.9100,\,0.9100,\,0.9100$$ for $$d_{\mathrm c}=3,5,7,9$$, so $$v\simeq2(1-\epsilon^{2})\epsilon^{d_{\mathrm c}}\Delta$$ for $$d_{\mathrm c}\ge5$$. Species 2 differs by 1% in $$h$$ only, with the inter-block coupling re-tuned in both species ($$\zeta=0.01$$). Then $$\ln(v_2/v_1)/\ln1.01=2.805,\,4.800,\,6.800,\,8.800$$ and $$(v_2-v_1)/\bar v=2.79\times10^{-2},\,4.78\times10^{-2},\,6.76\times10^{-2},\,8.75\times10^{-2}$$, with $$\bar v$$ the mean speed. This is (14) with $$k=d_{\mathrm c}$$: the exponent is $$d_{\mathrm c}-2\epsilon^{2}/(1-\epsilon^{2})\approx d_{\mathrm c}-0.2$$, which follows from $$v\propto(1-\epsilon^{2})\epsilon^{d_{\mathrm c}}$$, and the offset from $$d_{\mathrm c}$$ is the $$O(\epsilon^{2}\zeta)$$ term. If instead species 2 differs by 1% in both $$h$$ and $$\Delta$$ (and in $$J$$), the whole Hamiltonian is rescaled and $$(v_2-v_1)/\bar v=9.95\times10^{-3}$$ at every $$d_{\mathrm c}$$: the factor $$d_{\mathrm c}$$ is absent because the gap, against which the perturbative couplings are measured, scales with them.

*Example (quantum code).* For the $$[[5,1,3]]$$ code with $$H_0=-\Delta\sum S_s$$ ($$\Delta=1$$) and $$V=\epsilon\sum_{q,a}\lambda_{q,a}\sigma^a_q$$ with 15 independent Gaussian couplings, the splitting of the two lowest levels is proportional to $$\epsilon^{3}$$: the logarithmic slope between $$\epsilon=10^{-3}$$ and $$1.6\times10^{-2}$$ is $$3.000$$ to four digits. Since multiplying all couplings by $$1+\zeta$$ is a rescaling of $$\epsilon$$, the splitting then changes by $$(1+\zeta)^3-1$$, in agreement with (14) for $$k=d_{\mathrm c}=3$$. This is a check at $$d_{\mathrm c}=3$$ only.

Neither example is a proof for all codes. The proposition is order by order in the series, it concerns stabiliser codes with Pauli perturbations, and it rests on the stated assumption about how the speed depends on the logical coupling. The stability theorem for topologically ordered Hamiltonians [[Bravyi–Hastings–Michalakis]](#ref-bhm) protects quantities proportional to the identity on the code space, such as the degeneracy of the ground space. A speed is the ratio of two non-trivial logical couplings, which that theorem does not address. Holographic or approximate codes, non-Pauli perturbations and regimes in which the Schrieffer–Wolff series does not converge are outside the proposition.

## 6. What the results leave open

The results exclude protection by the mechanisms named in the opening. They leave the following routes, each with its conditions.

- A residual group acting absolutely irreducibly on Euclidean momentum space, such as $$H(4)$$, with linearised diffeomorphism invariance that survives interactions, reflection positivity so that the continuation to real time exists, and no doublers. Corollary 2 gives one coefficient for each of the quadratic bosonic actions of such a theory, and a Haar average over the frame in Euclidean signature (section 3) falls under the same conditions. Whether a substrate with these properties exists is open; [lattice diffeomorphisms](/quantum-gravity/lattice-diffeomorphisms/) discusses the obstruction to exact diffeomorphism invariance in Hamiltonian lattices.
- Exact supersymmetry with one algebra of the form (4) acting on all fields. Proposition 5 gives equal speeds for short multiplets. The existence of such a theory including gravity, and the size of the residual difference after soft breaking, are inputs.
- A gravity-only frame that does not enter the matter regulator, so that the transmitted violation is suppressed as in [[Pospelov–Shang]](#ref-ps). Proposition 4 then says that $$c_T/c_\gamma$$ is a parameter of the gravity sector, which must respect the GW170817 window; no symmetry argument for it to be small is given here.
- A tuned counterterm, with one dimension-four coefficient per species and per order of perturbation theory. Theorem 8 gives the size of the coefficient that must be cancelled in the model considered.
- Emergent Lorentz invariance by renormalisation-group flow, for which [[Anber–Donoghue]](#ref-ad), [[Iengo–Russo–Serone]](#ref-irs) and [[Bednik–Pujolàs–Sibiryakov]](#ref-bps) give analyses that are not examined here.
- A frame that is a property of a state or a cluster of particles and not of the regulator: [frames and reflection positivity](/quantum-gravity/frames-and-reflection-positivity/).

The gauge-structure factor that converts the Yukawa coefficient $$1/3$$ into a photon–electron speed difference is not computed. The shape independence of Theorem 8 holds for a regulator on the scalar line; for a regulator on the fermion line the coefficient of [[Collins et al.]](#ref-collins) depends on the shape, and regulators acting on all lines in a gauge-invariant way have not been treated. A two-loop calculation would show whether the one-loop coefficient is stable. For Proposition 1, a derivation of $$\dim\mathcal V=10$$ by hand, for example through the linearised Riemann tensor, would extend more transparently to other spins than the exact linear-algebra computation used here. For codes, the non-vanishing of the minimal-degree coefficient in Proposition 11 has to be checked for each code and perturbation, as it is in the two examples, and Proposition 9 is not proved beyond the stabiliser class.

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
