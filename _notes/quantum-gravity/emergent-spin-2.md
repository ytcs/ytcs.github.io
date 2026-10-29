---
title: "Gravitons from qubits, rotors and composite tetrads"
date: 2026-10-29
category: gravity
order: 3
description: Three routes to a massless spin-2 mode from local degrees of freedom are tested. A stable, local, finite-dimensional qubit or rotor model with emergent linearised diffeomorphisms has gravitons that disperse at least as k squared. A condensed composite-tetrad model has no massless graviton at leading order in the number of flavours. An entanglement-geometry state whose metric responds only at short range carries no gravitating mass.
status: "Propositions proved under hypotheses stated for each one; the composite-tetrad statement is a derivation inside a large-flavour mean-field model, and the entanglement-geometry statement is conditional on one relation of the framework it criticises. Numerical checks are included for small models, not for interacting lattices."
excludes: "Finite-dimensional local models whose gapless sector is a stable, analytic, harmonic spin-2 field with emergent linearised spatial diffeomorphisms and linear dispersion; condensed composite-tetrad mean-field phases with a massless graviton at leading order in the flavour number; entanglement-geometry states with exponentially local metric response as sources of Newtonian gravity."
---

The question is whether a massless spin-2 mode with linear dispersion can emerge from local qubits or rotors, from fermions with a composite tetrad, or from the entanglement structure of a state. The answers, under explicit hypotheses, are as follows. For a stable, local, analytic harmonic theory of a symmetric tensor with emergent linearised spatial diffeomorphisms, the gapless modes have $$\omega\le Ck^2$$ (Proposition 1), and $$\omega\le Ck^3$$ if the scalar constraint also emerges and the variables are compact (Proposition 2). Linear dispersion needs an indefinite potential and a kinetic term that is invariant under the gauge transformations only up to terms proportional to constraints, and in a finite-dimensional local model the second cannot occur (Proposition 3). In a large-flavour mean field for fermions with a composite tetrad, the quadratic form for uniform shear is negative definite for every regulator that does not single out special directions, so no massless transverse traceless mode exists at leading order in $$1/N_f$$ (Proposition 4). In the entanglement-geometry framework of Cao, Carroll and Michalakis, a state whose metric response to a local perturbation decays faster than the Newtonian tail has zero gravitating mass (Proposition 6).

The emergence of linearised gauge symmetry and gravitons from local qubit models is due to Gu and Wen [[Gu–Wen 2006]](#ref-gw06) [[Gu–Wen 2009]](#ref-gw09), who found $$\omega\propto k^3$$ in a model with both constraints; soft gravitons with $$\omega\propto k^2$$ and $$z=3,2$$ Lifshitz gravitons come from compact rotor liquids [[Xu]](#ref-xu) [[Xu–Hořava]](#ref-xh), and tensor gauge theories with fractons give gravitons with linear dispersion but a gauge group smaller than the diffeomorphisms [[Pretko 2017a]](#ref-pretko1) [[Pretko 2017b]](#ref-pretko2). Marolf argued that dynamical gravity with universal coupling to energy requires kinematic non-locality [[Marolf]](#ref-marolf). Weinberg and Witten excluded a covariant conserved energy-momentum tensor for massless particles of helicity above one [[Weinberg–Witten]](#ref-ww). Spinor gravity with a composite tetrad is due to Diakonov and, in lattice mean field, Vladimirov and Diakonov, who found that the graviton mass is generically of the order of the cutoff and proposed to sit at a phase transition [[Diakonov]](#ref-diak) [[Vladimirov–Diakonov]](#ref-vd); the vanishing of the vacuum energy at equilibrium goes back to Klinkhamer and Volovik [[Klinkhamer–Volovik]](#ref-kv). Space from entanglement is the proposal of Cao, Carroll and Michalakis [[CCM]](#ref-ccm).

This page adds a bound on the dispersion of emergent gravitons that depends only on stability, locality, analyticity and the structure of the gauge symmetry; the observation that the linear-dispersion mechanism of general relativity is excluded in finite-dimensional local models; the sign-definite shear Hessian of the composite-tetrad mean field for an arbitrary regulator, with a sum rule; and the combination of a divergence-theorem argument with the exponential locality of the response of gapped ground states, which shows that the framework of [[CCM]](#ref-ccm) with gapped states has zero gravitating mass. How the same locality arguments enter the lattice problem for a gravity with exact diffeomorphisms is on the page [Lattice gravitons, exact diffeomorphisms and lattice momentum](/quantum-gravity/lattice-diffeomorphisms/), and the soft-theorem side is on [Exact universal graviton coupling and periodic dispersion](/quantum-gravity/spin-2-soft-locking/).

## 1. Linearised gauge structure and the hypotheses

The linearised spatial gauge transformation (3) of the companion page, $$A_{ij}\to A_{ij}+\partial_i\xi_j+\partial_j\xi_i$$ for a symmetric tensor $$A_{ij}$$, is accompanied by the vector Gauss law $$\partial_jE_{ij}=0$$ for the conjugate momentum $$E_{ij}$$; this is the momentum constraint. In general relativity there is also the Hamiltonian constraint $$\partial_i\partial_jA_{ij}-\partial^2A_{ii}=0$$, which acts on the conjugate variable by $$E_{ij}\to E_{ij}+(\partial_i\partial_j-\delta_{ij}\partial^2)\theta$$. In Fourier space these two gauge directions are

$$
g(k)\xi=i\,(k_i\xi_j+k_j\xi_i),\qquad s(k)=k\otimes k-\lvert k\rvert^2\mathbb 1. \tag{1}
$$

We call them the *vector* and the *scalar* constraint. The vector one acts on $$A$$, the scalar one on $$E$$ (and as a constraint on $$A$$). They commute: $$k_is_{ij}=0$$, that is $$g^\dagger s=0$$.

The statements of sections 2 and 3 concern a harmonic Hamiltonian $$H_2=\tfrac12\sum_k(E,A)^\dagger\mathbf H(k)(E,A)$$ with

$$
\mathbf H(k)=\begin{pmatrix}K(k)&N(k)\\ N(k)^\dagger&M(k)\end{pmatrix}.
$$

The hypotheses used by each proposition are collected here, and each proposition says which of them it uses.

(H1) *Locality.* The entries of $$\mathbf H(k)$$ are real-analytic and $$2\pi$$-periodic in $$k$$ (exponentially local couplings) on a lattice, and $$A,E$$ take values in the six-dimensional space of symmetric tensors.

(H2) *Vector constraint.* The vector gauge transformation is $$g(k)=g_1(k)+O(k^2)$$, with $$g_1(k)\xi=i(k_i\xi_j+k_j\xi_i)$$, and $$\mathbf H(k)$$ is invariant under it: $$\mathbf H(k)(0,g\xi)=0$$, that is $$Mg=0$$ and $$Ng=0$$. Physical states obey $$g^\dagger E=0$$.

(H3) *Stability.* $$\mathbf H(k)$$ is positive semidefinite on the subspace $$\{(E,A):g(k)^\dagger E=0\}$$. In particular $$M\ge0$$, because $$(0,A)$$ lies in that subspace.

(H4) *Scalar constraint, strong invariance.* There is a scalar gauge direction $$s(k)=s_2(k)+O(k^3)$$ in $$E$$ with $$\mathbf H(k)(s\theta,0)=0$$, that is $$Ks=0$$ and $$N^\dagger s=0$$; physical $$A$$ obeys $$s^\dagger A=0$$ and physical $$E$$ is taken modulo $$s\theta$$.

(H5) *Compactness.* The potential is $$V(A)=\sum_{x,\alpha}c_\alpha(1-\cos(n_\alpha\cdot A)_x)$$ with finitely many integer vectors $$n_\alpha$$ of finite support and $$c_\alpha>0$$, and the vacuum is $$A=0$$, so $$M(k)=\sum c_\alpha\,n_\alpha(k)n_\alpha(k)^\dagger$$, where $$n_\alpha(k)$$ is the symbol of $$n_\alpha$$.

(H6) *Finite local dimension.* The local Hilbert spaces are finite-dimensional, and the constraints are commuting Hermitian operators $$C_x$$ supported near $$x$$.

The quantities that the propositions bound are the frequencies of the harmonic theory restricted to physical states. With the gauge directions removed, the space of physical $$(E,A)$$ at fixed $$k$$ is a symplectic vector space, and $$\mathbf H$$ induces a quadratic form on it. The frequencies are its symplectic eigenvalues: if $$\mathbf H$$ is positive definite there is a symplectic basis in which it is $$\mathrm{diag}(\Omega,\Omega)$$ with $$\Omega=\mathrm{diag}(\omega_1,\omega_2,\dots)$$ (Williamson's theorem), and the $$\omega_j$$ are the mode frequencies. For a Hamiltonian $$\tfrac12E^TKE+\tfrac12A^TMA$$ with no cross term they are the square roots of the eigenvalues of $$KM$$. We say $$z\ge z_0$$ if $$\omega(k)=O(\lvert k\rvert^{z_0})$$ for all modes.

## 2. Worked examples

*The Fierz–Pauli form is the only gauge-invariant quadratic form of degree two.* We solved the linear equations for a real symmetric $$6\times6$$ matrix $$M(k)$$ whose entries are homogeneous polynomials of degree $$d$$ in $$k$$ with $$M(k)g_1(k)=0$$ for all $$k$$ (no isotropy assumed). For $$d=0$$ and $$d=1$$ the solution space is zero-dimensional; for $$d=2$$ it is one-dimensional, spanned by the Fierz–Pauli form $$\mathcal U$$ of section 1.1 of the companion page. On the three-dimensional space of physical $$A$$ at $$k=(0,0,1)$$ (the space orthogonal to $$g_1\xi$$), its signature is $$(+,+,-)$$: positive on the transverse traceless pair, negative on the transverse-trace mode. It is therefore indefinite, and no non-zero multiple of it is positive semidefinite.

*Four small models.* Take $$k=\kappa(0,0,1)$$ in the continuum symbols, $$\mathrm{inc}(A)_{ij}=\varepsilon_{iab}\varepsilon_{jcd}k_ak_cA_{bd}$$ (the linearised Einstein-like operator, which is gauge invariant and of degree two) and $$\mathrm{symcurl}(E)_{ij}=\tfrac12(\varepsilon_{iab}k_aE_{bj}+\varepsilon_{jab}k_aE_{bi})$$. The frequencies of the physical modes, from the eigenvalues of $$KM$$ on the physical subspaces, are the following.

| model | $$K$$ | $$M$$ | constraints | physical modes | frequencies | $$z$$ |
|---|---|---|---|---|---|---|
| A | $$\mathbb 1$$ | Fierz–Pauli | vector | 3 | $$\omega^2=\pm\kappa^2/2$$: two real, one imaginary | 1, unstable |
| B | $$\mathbb 1$$ | $$\mathrm{inc}^T\mathrm{inc}$$ | vector | 3 | $$\omega=\kappa^2$$, three equal | 2 |
| C | $$\mathrm{symcurl}^T\mathrm{symcurl}$$ | $$\mathrm{inc}^T\mathrm{inc}$$ | vector and scalar | 2 | $$\omega=\kappa^3$$, two equal | 3 |
| D | DeWitt form | Fierz–Pauli | vector and scalar | 2 | $$\omega=\kappa/\sqrt2$$, two equal | 1 |

At $$\kappa=0.02$$ and $$0.04$$ the computed frequencies in model B are $$4.0\times10^{-4}$$ and $$1.6\times10^{-3}$$, in model C $$8.0\times10^{-6}$$ and $$6.4\times10^{-5}$$, in model D $$0.0141$$ and $$0.0283$$, so the exponents are $$2$$, $$3$$ and $$1$$. In model A the physical subspace contains the transverse-trace direction with negative Fierz–Pauli eigenvalue, which gives one imaginary frequency. Model D uses the DeWitt kinetic form, which is indefinite on the transverse-trace direction, and its stability depends on the scalar constraint removing that direction; its kinetic term does not annihilate $$s$$, so it is invariant only up to a multiple of the constraint. Model C is the pair of strongly invariant constraints and has the $$k^3$$ dispersion of Gu and Wen's L-type model [[Gu–Wen 2009]](#ref-gw09).

## 3. Qubit and rotor gravitons

### 3.1 Stability bounds the dispersion

<div class="proposition" markdown="1">
**Proposition 1** (stability gives $$z\ge2$$). Assume (H1), (H2) and (H3). Then every physical mode has $$\omega(k)\le C\lvert k\rvert^2$$ for small $$k$$.
</div>

<div class="proof" markdown="1">
*Proof.* The idea is that gauge invariance of the potential block $$M$$ forces its low-order terms to be the Fierz–Pauli form, which is indefinite, and positivity then removes them; the frequencies are then controlled by the sizes of $$K$$ and $$M$$ alone.

Expand $$M=\sum_{n\ge0}M_n$$ in homogeneous polynomials, each Hermitian. Order by order, $$Mg=0$$ reads $$\sum_{a+b=n}M_ag_b=0$$ with $$g_b$$ of degree $$b\ge1$$. At $$n=1$$, $$M_0g_1=0$$; the vectors $$g_1(k)\xi$$ span all symmetric tensors, so $$M_0=0$$. At $$n=2$$, $$M_1g_1=0$$. The matrix $$M_1(k)$$ is odd in $$k$$, and $$M(k)/\lvert k\rvert\to M_1(\hat k)$$ is positive semidefinite for $$\hat k$$ and for $$-\hat k$$; so $$M_1(\hat k)$$ and $$-M_1(\hat k)$$ are both positive semidefinite and $$M_1=0$$. At $$n=3$$, $$M_2g_1=0$$. Because $$g_1$$ is a real matrix times $$i$$, the real symmetric part $$S$$ and the imaginary antisymmetric part of $$M_2$$ both annihilate $$g_1$$ separately. By the computation of section 2, $$S=\lambda\,\mathrm{FP}$$ for a scalar $$\lambda$$. The quadratic form of the positive semidefinite $$M_2$$ on real vectors is that of $$S$$, so $$\lambda\,\mathrm{FP}\ge0$$, and since FP is indefinite on the physical subspace, $$\lambda=0$$. A Hermitian positive semidefinite matrix with zero real part has zero diagonal, hence vanishes: $$M_2=0$$. At $$n=4$$, $$M_3g_1=0$$, and $$M_3$$ is odd, so as for $$M_1$$ it vanishes. Thus $$\lVert M(k)\rVert=O(k^4)$$.

To pass to frequencies, let $$W=\{(E,A):g^\dagger E=0\}$$ and $$D(E,A)=(E,-A)$$. The map $$D$$ preserves $$W$$, and $$\mathbf H+D\mathbf HD=2\,\mathrm{diag}(K,M)$$. On $$W$$ both $$\mathbf H$$ and $$D\mathbf HD$$ are positive semidefinite by (H3), so $$\mathbf H\le2\,\mathrm{diag}(K,M)$$ on $$W$$. The gauge directions $$(0,g\xi)$$ are null vectors of $$\mathbf H$$ in $$W$$ and in the kernel of the symplectic form restricted to $$W$$, so everything descends to the symplectic quotient. Symplectic eigenvalues are monotone in the order of positive definite forms, which follows from the min–max principle for them [[Bhatia–Jain]](#ref-bj); for semidefinite forms add $$\epsilon\mathbb 1$$ and let $$\epsilon\to0$$. The symplectic eigenvalues of $$2\,\mathrm{diag}(K,M)$$ are the square roots of the eigenvalues of $$4KM$$, so $$\omega\le2\sqrt{\lVert K\rVert\lVert M\rVert}$$, where $$K$$ is restricted to $$\{g^\dagger E=0\}$$ and is bounded because its entries are analytic and periodic. With $$\lVert M\rVert=O(k^4)$$ this gives $$\omega=O(k^2)$$.
</div>

The step that carries the content is the use of positivity to remove the Fierz–Pauli term; the rest is routine bookkeeping. General relativity at $$z=1$$ evades the proposition because its full potential, restricted only by the vector constraint, is indefinite (model A); stability holds there only after the Hamiltonian constraint removes the negative direction. The proposition shows that a finite-dimensional local model cannot rely on that: if all physical modes have $$z=1$$, the scalar constraint must also be present.

### 3.2 Both constraints, strongly invariant

<div class="proposition" markdown="1">
**Proposition 2** (the pair of constraints). Assume (H1)–(H4).

(a) Every physical mode has $$\omega\le C\lvert k\rvert^2$$, and the physical modes are the transverse traceless pair.

(b) If in addition (H5) holds, then $$\omega\le C\lvert k\rvert^3$$.
</div>

<div class="proof" markdown="1">
*Proof.* Physical $$E$$ lies in $$\{g^\dagger E=0\}$$ modulo $$s\theta$$, and physical $$A$$ in $$\{s^\dagger A=0\}$$ modulo gauge; at leading order the intersection is the two-dimensional transverse traceless space, because $$s(k)$$ spans the transverse-trace direction.

(a) We bound $$K$$ on the physical $$E$$. From $$Ks=0$$ and $$s=s_2+O(k^3)$$: at order 2, $$K_0s_2(k)=0$$ for all $$k$$; the tensors $$s_2(k)=k\otimes k-\lvert k\rvert^2\mathbb 1$$ span all symmetric tensors (the three diagonal ones for $$k=e_1,e_2,e_3$$ give all diagonal tensors, and $$k=e_i+e_j$$ the off-diagonal ones), so $$K_0=0$$. At order 1, $$K_1(k)$$ restricted to the transverse tensors is odd and positive semidefinite for $$\pm\hat k$$ (H3), hence zero there. So $$K$$ restricted to the physical $$E$$ is $$O(k^2)$$. For $$M$$ the proof of Proposition 1 shows $$M=\lambda\,\mathrm{FP}+O(k^3)$$ with $$\lambda$$ no longer forced to vanish, since FP is positive on the transverse traceless pair; hence $$\lVert M_{\rm phys}\rVert=O(k^2)$$. The comparison argument of Proposition 1 then gives $$\omega\le2\sqrt{\lVert K_{\rm phys}\rVert\lVert M_{\rm phys}\rVert}=O(k^2)$$.

(b) Under (H5), gauge invariance of $$V$$ under $$A\to A+g\xi$$ for every continuous $$\xi$$ requires $$n_\alpha\cdot(g\xi)=0$$, that is $$g(k)^\dagger n_\alpha(k)=0$$ for each $$\alpha$$. The symbol $$n(k)$$ of a vector with finite support is a trigonometric polynomial with values in the symmetric tensors, and $$g^\dagger n=0$$ reads $$v_i(k)n_{ij}(k)=0$$ with $$v=k+O(k^2)$$. Expand $$n=n_0+n_1(k)+\dots$$. At order 1, $$k_in_{0,ij}=0$$ for all $$k$$, so $$n_0=0$$. At order 2 write $$n_{1,ij}=c_{ij,l}k_l$$ with $$c_{ij,l}=c_{ji,l}$$; then $$k_ik_lc_{ij,l}=0$$ for all $$k$$ says that $$c_{ij,l}+c_{lj,i}=0$$, and a tensor that is symmetric in its first two indices and antisymmetric under exchange of its first and third vanishes: $$c_{ij,l}=-c_{lj,i}=-c_{jl,i}=c_{il,j}=c_{li,j}=-c_{ji,l}=-c_{ij,l}$$. Hence $$n_\alpha(k)=O(k^2)$$, $$M=\sum c_\alpha n_\alpha n_\alpha^\dagger=O(k^4)$$, and $$\omega\le2\sqrt{\lVert K_{\rm phys}\rVert\lVert M\rVert}=O(k^3)$$.
</div>

The algebraic lemma in (b) is the dual of the one in Theorem 2 of the companion page, where it is applied to the symbol of the integer matrix $$L$$. It is the point that makes compact gravitons soft: a local integer-valued gauge-invariant combination of the fields has at least two derivatives, as the linearised curvature does.

### 3.3 Weak invariance cannot occur in finite dimensions

General relativity has $$z=1$$ with both constraints because its kinetic term is invariant under the scalar and vector constraints only weakly. The generator $$C$$ of the vector constraint maps the kinetic term $$T=\pi_{ij}\pi_{ij}-\tfrac12\pi_{kk}^2$$ to a multiple of the Hamiltonian constraint $$G$$, which commutes with $$C$$: $$\mathrm{ad}_C T=G$$ and $$\mathrm{ad}_CG=0$$. In linear-algebra terms $$\mathrm{ad}_C$$ acts on $$T$$ as a $$2\times2$$ nilpotent Jordan block.

<div class="proposition" markdown="1">
**Proposition 3** (finite dimension forces strong invariance). Assume (H6). Let $$H$$ be any local Hamiltonian and $$P$$ the projector onto the common kernel of the $$C_x$$. Then $$PHP=PH_0P$$, where $$H_0$$ is the part of $$H$$ that commutes with every $$C_x$$; $$H_0$$ is local and strongly invariant. In particular, if $$\mathrm{ad}_{C}^2H=0$$ for a Hermitian $$C=C_x$$, then $$\mathrm{ad}_CH=0$$: no nilpotent Jordan block exists.
</div>

<div class="proof" markdown="1">
*Proof.* On the space of operators on a finite-dimensional Hilbert space with the Hilbert–Schmidt inner product $$\langle A,B\rangle=\mathrm{tr}(A^\dagger B)$$, the superoperator $$\mathrm{ad}_C$$ is self-adjoint when $$C$$ is Hermitian, since $$\langle A,[C,B]\rangle=\mathrm{tr}(A^\dagger CB-A^\dagger BC)=\langle[C,A],B\rangle$$. A self-adjoint operator is diagonalisable with real eigenvalues, so $$\mathrm{ad}_C^2H=0$$ implies $$\mathrm{ad}_CH=0$$. For the commuting family, decompose $$H=\sum_qH_q$$ into joint eigen-operators, $$[C_x,H_q]=q_xH_q$$. Then $$0=P[C_x,H_q]P=q_xPH_qP$$ because $$C_xP=PC_x=0$$, so $$PH_qP=0$$ unless $$q=0$$, and $$PHP=PH_0P$$ with $$H_0=H_{q=0}$$. A term of $$H$$ supported in a region $$Z$$ commutes with $$C_x$$ for $$x$$ far from $$Z$$, so its charge-zero part involves only the $$C_x$$ near $$Z$$ and is supported in a slightly larger region.
</div>

The proposition applies the usual remark that gauge invariance of an effective Hamiltonian on the constraint surface is automatic after twirling. It excludes the specific mechanism by which general relativity reaches $$z=1$$ with both constraints, because that mechanism needs $$\mathrm{ad}_C^2T=0$$ and $$\mathrm{ad}_CT\ne0$$ for the generators of local constraints that are bounded operators on a finite-dimensional space. It does not exclude a model in which the constraints are generated by unbounded operators (rotors with non-compact angles), for which the Hilbert space is infinite-dimensional; the linear dispersion of a non-compact rotor model with both constraints exists at the harmonic level, and its nonlinear completion is obstructed by [[Marolf]](#ref-marolf).

<div class="remark" markdown="1">
**Remark** (a gauge group smaller than the diffeomorphisms). The scalar-charge tensor gauge theory [[Pretko 2017b]](#ref-pretko2) has the gauge redundancy $$A_{ij}\to A_{ij}+\partial_i\partial_j\alpha$$ and Gauss law $$\partial_i\partial_jE_{ij}=\rho$$, which is not of the symmetrised-gradient type of (H2). Take $$K=\mathbb 1$$ and a potential built from the field strength $$B_{ij}=\varepsilon_{iab}k_aA_{bj}$$, namely $$V=\alpha\lvert B_0\rvert^2+\beta\lvert B_A\rvert^2+\gamma(\mathrm{tr}B)^2$$ with $$B_0$$ the traceless symmetric part and $$B_A$$ the antisymmetric part. We computed the Hessian symbolically at $$k=(0,0,1)$$: its eigenvalues are $$0$$ for the gauge direction $$A_{33}$$, $$2\alpha$$ twice (the transverse traceless pair $$A_{11}-A_{22},A_{12}$$), $$2\beta$$ once ($$A_{11}+A_{22}$$) and $$(\alpha+\beta)/2$$ twice ($$A_{13},A_{23}$$). All five physical modes have $$\omega\propto\lvert k\rvert$$, with $$\omega^2=2\alpha k^2$$ for the pair that carries helicity two and $$\omega^2=\tfrac12(\alpha+\beta)k^2$$ for the pair of helicity one, so the speed ratio is $$(\alpha+\beta)/(4\alpha)\ge\tfrac14$$ for $$\beta\ge0$$. This gauge group does not contain the transport term of the diffeomorphisms, and the extra modes of helicity one and zero are present; whether such a theory can couple universally to energy is not decided here.
</div>

## 4. Composite tetrads

### 4.1 The mean-field model

In spinor gravity of the Akama–Diakonov type the tetrad is a composite of fermions, $$e^A_\mu=\sum_f\bar\psi_f\gamma^A\nabla_\mu\psi_f$$ over $$N_f$$ flavours, and the bare action is built from differential forms [[Diakonov]](#ref-diak). At zero derivatives the only such term is the volume $$\det e$$. A Hubbard–Stratonovich field $$J^\mu{}_A$$ (the inverse-tetrad density, a $$4\times4$$ matrix per site) is introduced, and at large $$N_f$$ the mean-field action per site is

$$
\frac{\Gamma(J)}{N_f}=c\,\Phi(\det J)-2\int d\mu(n)\,\log\big(n^TJJ^Tn\big)+\text{const}, \tag{2}
$$

where $$\Phi$$ is an arbitrary function, $$c$$ a constant, and $$\mu$$ a probability measure on unit four-vectors $$n$$, the directions of the fermion modes: for naive lattice fermions with action $$\sum_\mu J^\mu{}_A\gamma^A i\sin p_\mu$$ it is the distribution of $$\sin p/\lvert\sin p\rvert$$ for $$p$$ uniform in the Brillouin zone. The measure is independent of $$J$$ because the number of modes is fixed per coordinate cell, not per proper volume; this is what we call a background-free regulator. We assume:

(T1) The effective action of the uniform field is (2), and $$J_0=j\mathbb 1$$ is a stationary point.

(T2) The measure $$\mu$$ is a $$J$$-independent probability measure.

(T3) The quadratic form of fluctuations of the shear $$J=j(1+\delta\epsilon)$$, $$\epsilon$$ symmetric and traceless, at wave vector $$k\to0$$ equals its value at $$k=0$$ (continuity at zero momentum).

(T1) and (T2) define the model; (T3) is used only for the final consequence about masslessness. The result is a derivation inside this mean-field model.

### 4.2 The shear form

Write $$C_\mu(\epsilon)=\langle\lvert\epsilon n\rvert^2-(n^T\epsilon n)^2\rangle_\mu$$ for a symmetric traceless $$4\times4$$ matrix $$\epsilon$$. By Cauchy–Schwarz, $$\lvert n^T\epsilon n\rvert\le\lvert\epsilon n\rvert$$ for unit $$n$$, so $$C_\mu\ge0$$.

<div class="proposition" markdown="1">
**Proposition 4** (shear no-go for condensed composite tetrads). Assume (T1) and (T2). Then:

1. $$\langle nn^T\rangle_\mu=\tfrac14\mathbb 1$$.

2. The coefficient of $$\delta^2$$ in $$\Gamma(j(1+\delta\epsilon))/N_f$$ is $$Q(\epsilon)=-4\,C_\mu(\epsilon)\le0$$, for every function $$\Phi$$ and every $$c$$ compatible with the gap equation. It vanishes only if $$\mu$$-almost every $$n$$ is an eigenvector of $$\epsilon$$; in particular $$Q(\epsilon)<0$$ for every $$\epsilon\ne0$$ if $$\mu$$ gives zero weight to the unit spheres of the eigenspaces of every $$\epsilon\ne0$$ (for example if $$\mu$$ has a density on the sphere).

3. For an orthonormal basis $$\epsilon_1,\dots,\epsilon_9$$ of the traceless symmetric matrices with $$\mathrm{tr}(\epsilon_a\epsilon_b)=\delta_{ab}$$,

$$
\sum_{a=1}^9Q(\epsilon_a)=-6\qquad\text{for every }\mu,\ \Phi\text{ and }c, \tag{3}
$$

so some shear eigenvalue is at most $$-2/3$$.

Under (T3) the transverse traceless fluctuations of the composite metric therefore have a non-zero value of their quadratic form at $$k\to0$$, and no massless transverse traceless mode exists at order $$N_f$$.
</div>

<div class="proof" markdown="1">
*Proof.* (1) Stationarity at $$J_0=j\mathbb 1$$ in a traceless direction: the bare term varies at order $$\delta^2$$ only (see below), and $$\log(n^TJJ^Tn)$$ varies at first order by $$2\delta\,n^T\epsilon n$$. So $$\langle n^T\epsilon n\rangle=0$$ for all traceless symmetric $$\epsilon$$, that is $$\langle nn^T\rangle\propto\mathbb 1$$, and the trace of $$nn^T$$ is 1.

(2) Expand with $$J=j(1+\delta\epsilon)$$ and $$\epsilon$$ traceless symmetric. Then $$JJ^T=j^2(1+2\delta\epsilon+\delta^2\epsilon^2)$$ and

$$
\log(n^TJJ^Tn)=2\log j+2\delta\,n^T\epsilon n+\delta^2\big(\lvert\epsilon n\rvert^2-2(n^T\epsilon n)^2\big)+O(\delta^3).
$$

Also $$\det J=j^4(1-\tfrac12\delta^2\mathrm{tr}\,\epsilon^2+O(\delta^3))$$. The gap equation, the stationarity of (2) in the scale $$j$$ at $$J=j\mathbb 1$$, is $$c\,\Phi'(D_0)D_0=1$$ with $$D_0=j^4$$, so the bare term contributes $$-\tfrac12\mathrm{tr}\,\epsilon^2$$ to the coefficient of $$\delta^2$$. The fermion term contributes $$-2\langle\lvert\epsilon n\rvert^2-2(n^T\epsilon n)^2\rangle$$. With (1), $$\langle\lvert\epsilon n\rvert^2\rangle=\mathrm{tr}(\epsilon^2\langle nn^T\rangle)=\tfrac14\mathrm{tr}\,\epsilon^2$$, so the total is

$$
Q(\epsilon)=-\tfrac12\mathrm{tr}\,\epsilon^2-\tfrac12\mathrm{tr}\,\epsilon^2+4\langle(n^T\epsilon n)^2\rangle=-4\Big[\tfrac14\mathrm{tr}\,\epsilon^2-\langle(n^T\epsilon n)^2\rangle\Big]=-4\,C_\mu(\epsilon),
$$

independent of $$\Phi$$ and $$c$$. Equality $$C_\mu(\epsilon)=0$$ requires $$\lvert\epsilon n\rvert=\lvert n^T\epsilon n\rvert$$ for $$\mu$$-almost every $$n$$, that is $$\epsilon n\parallel n$$. For $$\epsilon\ne0$$ traceless the eigenvectors of $$\epsilon$$ with a given eigenvalue span a proper subspace, so the unit eigenvectors form a finite union of lower-dimensional spheres.

(3) The completeness relation for traceless symmetric matrices is $$\sum_a(\epsilon_a)_{bc}(\epsilon_a)_{de}=\tfrac12(\delta_{bd}\delta_{ce}+\delta_{be}\delta_{cd})-\tfrac14\delta_{bc}\delta_{de}$$. Hence $$\sum_a\lvert\epsilon_an\rvert^2=\tfrac12(4+1)-\tfrac14=\tfrac94$$ and $$\sum_a(n^T\epsilon_an)^2=\tfrac12(1+1)-\tfrac14=\tfrac34$$ for every unit $$n$$, so $$\sum_aC_\mu(\epsilon_a)=\tfrac32$$ and the sum of $$Q$$ is $$-6$$. The largest eigenvalue in magnitude is at least $$6/9$$.

For the consequence, the massless transverse traceless mode would need the quadratic form of shear fluctuations to vanish at $$k\to0$$ in the direction of its polarisation; by (T3) that is $$Q(\epsilon)$$ for the polarisation $$\epsilon$$, which is non-zero under the condition in (2).
</div>

*Checks.* For $$\mu$$ uniform on the three-sphere, $$\langle n_1^4\rangle=1/8$$ and $$\langle n_1^2n_2^2\rangle=1/24$$, which gives $$Q=-2/3$$ in every direction. For naive hypercubic fermions we averaged $$n=\sin p/\lvert\sin p\rvert$$ over a midpoint grid of $$48^4$$ points in the zone. The diagonal traceless channel (multiplicity three) has $$Q_E=-1+2\langle(n_1^2-n_2^2)^2\rangle=-0.8263$$ and the off-diagonal channel (multiplicity six) has $$Q_T=-1+8\langle n_1^2n_2^2\rangle=-0.5868$$; the identity $$4\langle n_1^4\rangle+12\langle n_1^2n_2^2\rangle=1$$ holds to eight digits, and $$3Q_E+6Q_T=-6.0000$$. A finite-difference second derivative of the fermion part of (2) in the $$E$$ channel, plus the bare $$-\tfrac12$$, gives $$-0.8263$$ as well.

The sign of $$Q$$ depends on the contour of the Hubbard–Stratonovich integral, which for a purely interacting Grassmann action has no canonical real form; "tachyonic" is the real-$$J$$ reading. What is independent of the contour is that the shear form is of order $$N_f$$ and non-zero in every direction, which is all that the conclusion about masslessness uses. Where the shear form is negative, the model also has no stable uniform tetrad: at fixed volume the shape $$\hat G=\mathrm{diag}(\lambda^3,\lambda^{-1},\lambda^{-1},\lambda^{-1})$$ has $$\Gamma_{\rm shape}\simeq-6N_f\log\lambda$$ in the fermion term, which suggests a runaway towards degenerate tetrads; we have not analysed this beyond the leading expansion.

<div class="remark" markdown="1">
**Remark** (what is held fixed). Corollaries of Proposition 4 within the model: replacing $$\mu$$ by a mixture of regulators (a quenched average over complexes) gives another probability measure and the same sum rule (3), because $$Q$$ is linear in $$\mu$$. Replacing $$\Phi$$ or $$c$$ changes nothing, since they cancel in $$Q$$. Bare terms built from forms with derivatives (curvature or torsion) vanish on a constant $$J$$ with flat links, so they do not contribute to the uniform shear form either; this last statement is part of the definition of the class of bare actions, not proved here.
</div>

## 5. Entanglement geometry

### 5.1 The framework and a relation it uses

Cao, Carroll and Michalakis [[CCM]](#ref-ccm) decompose a Hilbert space into factors $$A_p$$ and consider states that are *redundancy-constrained*: the entropy of a union $$B$$ of factors is the sum of the mutual informations across its boundary,

$$
S(B)=\tfrac12\sum_{p\in B,\ q\notin B}I(A_p:A_q), \tag{4}
$$

where $$I(A_p:A_q)=S(A_p)+S(A_q)-S(A_pA_q)$$. Mutual information defines distances on the graph of factors; multidimensional scaling extracts an emergent manifold, and they argue that a perturbation of the state that changes the entropy of a small region by $$\delta S$$ induces scalar curvature $$R_p=-\zeta\,\delta S_{A_p}=\zeta\varepsilon_p$$ with $$\zeta>0$$ a constant fixed by the construction and $$\varepsilon_p=-\delta\langle K_{A_p}\rangle$$ an effective modular energy density, using the first law of entanglement. They note that perturbations that entangle the region further would have negative modular energy and give negative curvature, and state that conditions limiting such effects would be needed. We use the relation in the form

(E1) in the linearised regime, $$R^{(1)}=\zeta\,\varepsilon$$ with $$\zeta>0$$ and $$\varepsilon\ge0$$,

where $$R^{(1)}=\partial_i\partial_jh_{ij}-\partial^2h_{ii}$$ is the linearised scalar curvature of the emergent metric $$g=\delta+h$$.

### 5.2 Which states are redundancy-constrained

<div class="proposition" markdown="1">
**Proposition 5** (pair states). Let a finite graph with vertices $$p$$ and weights $$e_{pq}\in[0,\ln\chi]$$ on its edges be given. Place on each edge a pure two-party state with entanglement entropy $$e_{pq}$$, one party in $$A_p$$ and the other in $$A_q$$ (each factor a tensor product of the parties it holds). Then the product state satisfies (4) exactly for every subset $$B$$, with $$I(A_p:A_q)=2e_{pq}$$ on edges and $$0$$ elsewhere.
</div>

<div class="proof" markdown="1">
*Proof.* The state is a product over edges, so the entropy of any union of factors is the sum of $$e_{pq}$$ over the edges cut by its boundary. For two factors, $$S(A_p)=\sum_{q}e_{pq}$$ and $$S(A_pA_q)=S(A_p)+S(A_q)-2e_{pq}$$, since the pure state on the edge $$pq$$ is inside $$A_pA_q$$; so $$I=2e_{pq}$$. The sum in (4) is then $$\sum_{\rm cut}e_{pq}=S(B)$$.
</div>

So the map from redundancy-constrained states to graph metrics is onto: no property of the emergent geometry follows from the dimension of the Hilbert space, the entropies of the factors or the redundancy condition alone. For a graph that is the one-skeleton of a triangulation of a closed three-manifold with equilateral edges, curvature has a sign set by the mean degree. Euler's relation for the triangulation, $$V-E+F-T=0$$, with $$F=2T$$, gives $$T=E-V$$, and since each tetrahedron has six edges the mean number of tetrahedra per edge is $$\bar n=6T/E=6(1-2/\langle\deg\rangle)$$, where $$\langle\deg\rangle=2E/V$$. The deficit angle per edge for regular tetrahedra is $$2\pi-\bar n\arccos(1/3)$$, which vanishes for $$\langle\deg\rangle=2/(1-2\pi/(6\arccos(1/3)))=13.397$$ and is positive below it. The dimension of the Hilbert space does not enter. A curvature of order one in units of the factor size, against a cosmological scale, would then require $$\langle\deg\rangle$$ to be tuned to the corresponding accuracy; we state this as arithmetic, not as a result about observed cosmology.

### 5.3 Short-range response carries no mass

<div class="proposition" markdown="1">
**Proposition 6** (short-range metric response). Let $$D\ge3$$ and let $$g=\delta+h$$ on $$\mathbb R^D$$ with $$h$$ smooth, $$h=o(r^{-(D-2)})$$ and $$\partial h=o(r^{-(D-1)})$$ as $$r\to\infty$$. If $$R^{(1)}\ge0$$ everywhere then $$R^{(1)}\equiv0$$. Consequently, under (E1) such a metric has $$\varepsilon\equiv0$$.
</div>

<div class="proof" markdown="1">
*Proof.* The idea is that the linearised scalar curvature is a divergence, so its integral is a surface term, and the decay assumed makes the surface term vanish. We have $$R^{(1)}=\partial_i(\partial_jh_{ij}-\partial_ih_{jj})$$. By the divergence theorem, $$\int_{r<\mathcal R}R^{(1)}=\oint_{r=\mathcal R}(\partial_jh_{ij}-\partial_ih_{jj})n^i\,dS$$. The surface has area proportional to $$\mathcal R^{D-1}$$ and $$\partial h=o(\mathcal R^{-(D-1)})$$, so the right side tends to zero. (It is proportional to the linearised ADM mass.) Since $$R^{(1)}\ge0$$, the integral over the ball increases to $$\int R^{(1)}$$, which is therefore zero, and the continuous non-negative function $$R^{(1)}$$ vanishes.
</div>

A source of positive mass $$M$$ has $$h\sim GM/r^{D-2}$$, so it requires a far-field response of the mutual information of order $$r^{-(D-2)}$$ [[CCM]](#ref-ccm). The application is therefore as follows. Ground states of weakly interacting spin systems with gapped on-site terms respond to a localised perturbation, even one that closes the gap, only locally: expectation values of observables at distance $$d$$ change by at most $$\mathrm{const}\,e^{-c\,d}$$ [[Henheik–Teufel–Wessel]](#ref-htw); for general gapped systems when the gap stays open along the perturbation, the locality of the response is the principle of local perturbations perturbing locally [[BMNS]](#ref-bmns). The mutual information of a finite number of finite-dimensional factors is a continuous function of their reduced density matrices, so the induced change of the emergent metric far from a perturbation is exponentially small, and Proposition 6 gives zero gravitating mass: any positive modular energy is cancelled by negative modular energy elsewhere.

Whether the response of a state used in the framework is of this kind is a property of the state. As an illustration we computed, by exact diagonalisation of the ring $$H=-\sum_iZ_iZ_{i+1}-h\sum_iX_i$$ with periodic boundary conditions, the quantity $$\lvert\delta\ln I/\varepsilon\rvert$$ for the mutual information $$I$$ of the neighbouring qubits $$(r,r+1)$$ when the field on qubit 0 is changed by $$\pm\varepsilon$$, $$\varepsilon=10^{-4}$$. In the gapped phase, $$h=1.5$$, $$N=14$$, it is $$0.494,\ 0.0143,\ 3.9\times10^{-4}$$ for $$r=0,1,2$$, a fall by a factor of about 35 per site, and then a floor near $$9.5\times10^{-4}$$. The floor is a finite-size effect: its values are $$5.6\times10^{-3},\ 2.3\times10^{-3},\ 9.5\times10^{-4},\ 4.0\times10^{-4}$$ for $$N=10,12,14,16$$. At the critical field, $$h=1$$, $$N=14$$, the response for $$r=0,\dots,6$$ is $$0.558,\ 0.152,\ 0.098,\ 0.071,\ 0.057,\ 0.049,\ 0.046$$, a slow decay consistent with a power law at these sizes. The framework requires (4), which gapped states approximately satisfy and critical states do not at the level of single-qubit factors; we did not test (4) here.

## 6. Two statements about related models

### 6.1 Gauss-law constraints are invisible to typical states

Gauge-constrained qubit models are the natural place for emergent gauge fields, and a recurring idea is that the constraint structure of a closed universe leaves a trace in the weights of different charge sectors. A finite group $$G$$, a finite connected graph with $$N$$ vertices and $$L=\lvert\text{edges}\rvert-N+1$$ independent loops, a copy of a unitary representation $$H$$ of $$G$$ at each vertex (dimension $$D$$, character $$\chi_H$$) and the group algebra $$\mathbb C[G]$$ on each edge define the Hilbert space $$\bigotimes_vH_v\otimes\bigotimes_e\mathbb C[G]$$, with $$h\in G^V$$ acting on matter at $$v$$ by $$R_v(h_v)$$ and on an edge $$(s,t)$$ by $$g\mapsto h_sgh_t^{-1}$$. The physical space is the subspace invariant under all $$h$$ (Gauss law at every vertex), with projector $$P_{\rm phys}$$.

<div class="proposition" markdown="1">
**Proposition 7** (typicality is blind to the Gauss law). (a) For a gauge-invariant operator $$X$$ on $$H_v$$,

$$
\frac{\mathrm{tr}(P_{\rm phys}X)}{\mathrm{tr}\,P_{\rm phys}}=\frac{\sum_g\mathrm{tr}_{H}(R(g)X)\,\chi_H(g)^{N-1}\,\lvert C_G(g)\rvert^{L}}{\sum_g\chi_H(g)^{N}\,\lvert C_G(g)\rvert^{L}}, \tag{5}
$$

with $$C_G(g)$$ the centraliser of $$g$$.

(b) Suppose the identity is the only element of $$G$$ that acts as a scalar on $$H$$, and let $$\epsilon=\max_{g\ne e}\lvert\chi_H(g)/D\rvert^{N-1}(\lvert C_G(g)\rvert/\lvert G\rvert)^L$$ with $$(\lvert G\rvert-1)\epsilon<1$$. Then for every irreducible representation $$\lambda$$ of $$G$$, with multiplicity $$m_\lambda$$ and dimension $$d_\lambda$$ in $$H$$, the weight $$p_v(\lambda)=\mathrm{tr}(P_{\rm phys}\Pi^v_\lambda)/\mathrm{tr}P_{\rm phys}$$ of the $$\lambda$$-isotypic component at a vertex satisfies

$$
\Big\lvert\frac{p_v(\lambda)}{m_\lambda d_\lambda/D}-1\Big\rvert\le\frac{2(\lvert G\rvert-1)\epsilon}{1-(\lvert G\rvert-1)\epsilon}. \tag{6}
$$

</div>

<div class="proof" markdown="1">
*Proof.* (a) Choose a spanning tree and a root. The gauge transformations at the $$N-1$$ non-root vertices act freely and transitively on the configurations of the tree edges, so setting every tree edge to the identity is a unitary isomorphism of the physical space with the invariants of the residual diagonal action of $$G$$ on $$\bigotimes_vH_v\otimes\mathbb C[G]^{\otimes L}$$, where $$G$$ acts on matter by $$\bigotimes_vR_v$$ and on each of the $$L$$ non-tree edges by conjugation. The projector onto invariants is $$\lvert G\rvert^{-1}\sum_hU(h)$$, the trace of conjugation on $$\mathbb C[G]$$ at $$h$$ is $$\lvert C_G(h)\rvert$$, and an operator $$X$$ on $$H_v$$ that commutes with $$R_v$$ is unchanged by the isomorphism. Taking the trace gives (5).

(b) Divide numerator and denominator of (5) by $$D^N\lvert G\rvert^L$$ and put $$X=\Pi_\lambda$$, for which $$\mathrm{tr}_H(R(g)\Pi_\lambda)=m_\lambda\chi_\lambda(g)$$. The term $$g=e$$ gives $$m_\lambda d_\lambda/D=:a_0$$ in the numerator and $$1$$ in the denominator. The terms $$g\ne e$$ are bounded by $$a_0\epsilon$$ each in the numerator (using $$\lvert\chi_\lambda(g)\rvert\le d_\lambda$$) and by $$\epsilon$$ each in the denominator, so numerator $$=a_0(1+x)$$ and denominator $$=1+y$$ with $$\lvert x\rvert,\lvert y\rvert\le(\lvert G\rvert-1)\epsilon$$, and $$\lvert(1+x)/(1+y)-1\rvert\le\lvert x-y\rvert/(1-\lvert y\rvert)$$.
</div>

The weights tend to the values $$m_\lambda d_\lambda/D$$ of the unconstrained space exponentially in $$N$$ and in $$L$$, since a loop contributes $$\lvert C_G(g)\rvert/\lvert G\rvert\le\tfrac12$$ for non-central $$g$$. Elements that act as scalars carry the only memory of the constraint, namely the conservation of the total central charge (the $$N$$-ality); (6) does not cover that case. The proposition is a statement about the maximally mixed state on the physical space, which is the Haar average of pure physical states; that an individual random state is close to it when $$\dim H_{\rm phys}$$ is large is the usual concentration of measure and is not shown here.

*Example and check.* Take $$G=S_3$$ and $$H=\mathrm{triv}\oplus\mathrm{sign}\oplus\mathrm{std}$$ (dimension 4, isotypic weights $$\tfrac14,\tfrac14,\tfrac12$$). The classes have sizes $$1,3,2$$, $$\chi_H=4,0,1$$ and centraliser orders $$6,2,3$$. The formula (5) gives the following weights of the trivial and the standard component at one vertex of the maximally mixed physical state.

| graph | $$N$$ | $$L$$ | $$\dim H_{\rm phys}$$ | $$w_{\rm triv}$$ | $$w_{\rm std}$$ |
|---|---|---|---|---|---|
| path | 3 | 0 | 11 | 0.27273 | 0.45455 |
| triangle | 3 | 1 | 65 | 0.26154 | 0.47692 |
| square | 4 | 1 | 257 | 0.25292 | 0.49416 |
| theta | 4 | 2 | 1539 | 0.25146 | 0.49708 |
| $$K_4$$ | 4 | 3 | 9225 | 0.25073 | 0.49854 |

For the path, $$w_{\rm triv}=18/66=3/11$$ by hand (numerator $$16+2\cdot1$$, denominator $$64+0+2$$). On a tree, $$\lvert w_{\rm triv}-\tfrac14\rvert$$ is $$2.3\times10^{-2},1.5\times10^{-3},1.4\times10^{-6},1.4\times10^{-12}$$ for $$N=3,5,10,20$$, a decay by a factor $$\tfrac14$$ per vertex, and $$1.3\times10^{-15}$$ for $$N=10$$, $$L=30$$.

### 6.2 An automaton: forced Lorentz violation and the lapse coupling

A quantum cellular automaton is a local unitary on a lattice, and its Bloch symbol is a finite unitary matrix, so the results of the companion page apply to its bands. We record two exact statements about standard examples.

*A Weyl automaton on a cubic lattice.* Take the $$2\times2$$ Bloch matrix $$A_\sigma(k)=d_I-i\,\vec d\cdot\vec\sigma$$ with $$c_i=\cos k_i$$, $$s_i=\sin k_i$$, $$d_I=c_xc_yc_z-\sigma s_xs_ys_z$$ and $$\vec d=(s_xc_yc_z+\sigma c_xs_ys_z,\ c_xs_yc_z-\sigma s_xc_ys_z,\ c_xc_ys_z+\sigma s_xs_yc_z)$$, $$\sigma=\pm1$$; automata of this type on the body-centred cubic lattice are due to D'Ariano and Perinotti [[D'Ariano–Perinotti]](#ref-dap), and we do not claim that this form coincides with their conventions. The matrix is unitary ($$d_I^2+\lvert\vec d\rvert^2=1$$ identically, checked symbolically), and its eigenphases $$\pm\omega$$ satisfy $$\cos\omega=d_I$$. A series expansion along a unit vector $$n$$ gives

$$
\omega^2=\lvert k\rvert^2+2\sigma\,k_xk_yk_z-\tfrac13\big(k_x^2k_y^2+k_y^2k_z^2+k_z^2k_x^2\big)+O(k^5). \tag{7}
$$

The leading correction is cubic in $$k$$, anisotropic and odd under $$k\to-k$$: $$\omega=\lvert k\rvert+\sigma\,n_xn_yn_z\lvert k\rvert^2+\dots$$, a relative correction linear in the energy. In the notation of the companion page the lowest non-zero term of $$\omega^2-\lvert k\rvert^2$$ has degree $$p=3$$; a spin-2 band with a leading correction of this form would have a universality order $$\nu=1$$ in the sense of Proposition 5 there. In one space dimension the corresponding massless walk is exactly linear. Bounds on the lattice spacing of automata of this kind from Lorentz-violation data are derived in [[Mlodinow–Brun]](#ref-mb).

*The lapse coupling.* In one space dimension the Dirac automaton $$U(k)=\begin{pmatrix}n\,e^{ik}&-im\\-im&n\,e^{-ik}\end{pmatrix}$$ with $$n^2+m^2=1$$ and $$m=\sin\mu$$ has $$\cos\omega=n\cos k$$ [[Bisio–D'Ariano–Tosini]](#ref-bdt). Then $$\omega(0)=\mu$$ and $$\omega''(0)=\cot\mu$$.

<div class="proposition" markdown="1">
**Proposition 8** (equivalence principle for a tick-rate coupling). Model a static gravitational field by a local tick rate: the cell at $$x$$ performs $$N(x)$$ updates per coordinate time, so the generator is $$H=\tfrac12\{N(x),\Omega\}$$ with $$\Omega=i\log U$$. Assume a wave packet narrow in $$k$$ and a slowly varying $$N$$, so that the ray equations $$\dot x=N\omega'(k)$$, $$\dot k=-N'\omega(k)$$ hold. A packet at rest ($$k=0$$) at a point where $$N=1$$, $$N'=g$$ accelerates with

$$
a=-g\,\omega(0)\,\omega''(0)=-g\,\mu\cot\mu=-g\Big(1-\frac{\mu^2}3-\frac{\mu^4}{45}-\dots\Big). \tag{8}
$$

</div>

<div class="proof" markdown="1">
*Proof.* The ray Hamiltonian is $$H=N(x)\omega(k)$$. At $$k=0$$, $$\omega'(0)=0$$ (the band is even), so $$\ddot x=N\omega''(k)\dot k+N'\dot x\omega'=-NN'\omega(0)\omega''(0)$$, which is (8) with $$\omega(0)=\mu$$ and $$\omega''(0)=\cot\mu$$. The force $$\dot k=-g\mu$$ defines a gravitational mass $$\mu$$ and the curvature of the band an inertial mass $$\tan\mu$$, whose ratio is $$\mu\cot\mu$$.
</div>

The weak equivalence principle holds to first order in $$\mu$$ and fails at second order, at relative size $$\mu^2/3$$. At $$\mu\to\pi/2$$ the band is flat: the inertial mass diverges while the gravitational mass stays finite. We checked (8) by evolving a Gaussian wave packet of width $$40$$ lattice sites in the positive band on a ring of $$2048$$ sites with $$N=1+gx$$, $$g=2\times10^{-5}$$, for a time $$500$$, by fourth-order Runge–Kutta. The measured force over $$-g$$ is $$0.5238$$ and $$1.1160$$ for $$m=0.5$$ and $$0.9$$ against $$\mu=0.5236$$ and $$1.1198$$; the measured curvature $$\langle\omega''\rangle$$ is $$1.7292$$ and $$0.4842$$ against $$\cot\mu=1.7321$$ and $$0.4843$$; and the product, the acceleration over $$-g$$, is $$0.9057$$ and $$0.5403$$ against $$\mu\cot\mu=0.9069$$ and $$0.5423$$. For a proton with $$\mu\approx m_pa$$ and $$a$$ the inverse Planck mass, $$\mu\approx7.7\times10^{-20}$$ and $$\mu^2/3\approx2\times10^{-39}$$, which is twenty-four orders of magnitude below the sensitivity $$\lvert\eta\rvert\lesssim3\times10^{-15}$$ of torsion-balance and space-based tests; the application to composites whose mass is mostly binding energy is an assumption.

## 7. What the propositions do not exclude

Each route below drops a hypothesis. They are stated without a choice among them.

1. *Gauge redundancy that is not a symmetrised gradient.* (H2) fails. The scalar-charge tensor gauge theory of the remark in section 3.3 has $$z=1$$ and extra modes of helicity one and zero; other tensor gauge theories with fractons also have linear dispersion. Whether any of these couples universally to energy is not decided here.
2. *Non-analytic effective symbols.* (H1) fails. Gapless charged matter integrated out can produce terms such as $$k^{-2}$$ in the effective symbols, which would allow $$z=1$$ for the transverse traceless modes in Proposition 2; this is not excluded.
3. *Approximate gauge structure.* (H2) or (H4) holds only up to a breaking scale. The spectrum then has a gap at that scale, as in the composite-tetrad mean field of section 4.
4. *Non-compact variables.* (H5) and (H6) fail for rotors with non-compact angles, whose Hilbert space per site is infinite-dimensional. Such models have a linear dispersion with both constraints at the harmonic level, and a nonlinear completion with universal coupling to energy is obstructed by [[Marolf]](#ref-marolf). The companion page proves that integer charges in Villain form force $$z\ge2$$.
5. *Beyond the mean-field class of Proposition 4.* The composite-tetrad statement uses a fermion operator linear in a single site-valued $$J$$ and a $$J$$-independent mode set. Dynamical gauge links, simplicial actions with several composite channels, and phases in which the metric condenses without the tetrad (vestigial phases, see [[Volovik]](#ref-vol)) add channels with zero-derivative shape dependence and are not covered. Corrections of order $$N_f^0$$ to the sum rule (3) are not computed; if the one-loop fluctuation of $$J$$ shifts the trace by a positive amount $$\kappa_1$$, the shear trace can vanish at $$N_f=\kappa_1/6$$.
6. *Coarse-grained gapless states.* Proposition 6 excludes Newtonian gravity from states with exponentially local response. States that are redundancy-constrained at a coarse scale $$\ell$$ and have a mutual-information response of order $$(\ell/r)^{D-2}$$ are not excluded; whether such a family exists is open. Negative modular energy, which (E1) excludes by assumption, is also not excluded by the proposition.
7. *Matter in automata.* Proposition 8 and (7) concern one model each. Whether a different coupling of automata to a gravitational field, or an interacting automaton, changes the equivalence-principle statement is not addressed.

## 8. Open points

For Proposition 2, a model of the type in the first escape with a stable potential and the full set of constraints of general relativity at the harmonic level has not been constructed in finite dimensions, and Proposition 3 says it cannot use weak invariance. For the composite-tetrad model, the continuity assumption (T3) concerns the shear two-point function at small wave vector, which was not computed here; only the zero-momentum Hessian was. For entanglement geometry, a family of states that is redundancy-constrained at a coarse scale and has a Newtonian tail in the mutual-information response, with a response coefficient that is the same for all perturbation types, would decide whether Proposition 6 can be evaded.

## References

1. <a id="ref-gw06"></a>Gu, Wen, “A lattice bosonic model as a quantum theory of gravity,” [arXiv:gr-qc/0606100](https://arxiv.org/abs/gr-qc/0606100).
2. <a id="ref-gw09"></a>Gu, Wen, “Emergence of helicity $$\pm2$$ modes (gravitons) from qubit models,” Nucl. Phys. B 863, 90 (2012), [arXiv:0907.1203](https://arxiv.org/abs/0907.1203).
3. <a id="ref-xu"></a>Xu, “Novel algebraic boson liquid phase with soft graviton excitations,” [arXiv:cond-mat/0602443](https://arxiv.org/abs/cond-mat/0602443).
4. <a id="ref-xh"></a>Xu, Hořava, “Emergent gravity at a Lifshitz point from a Bose liquid on the lattice,” Phys. Rev. D 81, 104033 (2010), [arXiv:1003.0009](https://arxiv.org/abs/1003.0009).
5. <a id="ref-pretko1"></a>Pretko, “Generalized electromagnetism of subdimensional particles: a spin liquid story,” Phys. Rev. B 96, 035119 (2017), [arXiv:1606.08857](https://arxiv.org/abs/1606.08857).
6. <a id="ref-pretko2"></a>Pretko, “Emergent gravity of fractons: Mach's principle revisited,” Phys. Rev. D 96, 024051 (2017), [arXiv:1702.07613](https://arxiv.org/abs/1702.07613).
7. <a id="ref-marolf"></a>Marolf, “Emergent gravity requires (kinematic) non-locality,” [arXiv:1409.2509](https://arxiv.org/abs/1409.2509).
8. <a id="ref-ww"></a>Weinberg, Witten, “Limits on massless particles,” Phys. Lett. B 96, 59 (1980).
9. <a id="ref-bj"></a>Bhatia, Jain, “On symplectic eigenvalues of positive definite matrices,” J. Math. Phys. 56, 112201 (2015), [arXiv:1803.04647](https://arxiv.org/abs/1803.04647).
10. <a id="ref-diak"></a>Diakonov, “Towards lattice-regularized quantum gravity,” [arXiv:1109.0091](https://arxiv.org/abs/1109.0091).
11. <a id="ref-vd"></a>Vladimirov, Diakonov, “Phase transitions in spinor quantum gravity on a lattice,” [arXiv:1208.1254](https://arxiv.org/abs/1208.1254).
12. <a id="ref-kv"></a>Klinkhamer, Volovik, “Self-tuning vacuum variable and cosmological constant,” Phys. Rev. D 77, 085015 (2008), [arXiv:0711.3170](https://arxiv.org/abs/0711.3170).
13. <a id="ref-vol"></a>Volovik, “Fermionic quartet and vestigial gravity,” [arXiv:2312.09435](https://arxiv.org/abs/2312.09435).
14. <a id="ref-ccm"></a>Cao, Carroll, Michalakis, “Space from Hilbert space: recovering geometry from bulk entanglement,” Phys. Rev. D 95, 024031 (2017), [arXiv:1606.08444](https://arxiv.org/abs/1606.08444).
15. <a id="ref-htw"></a>Henheik, Teufel, Wessel, “Local stability of ground states in locally gapped and weakly interacting quantum spin systems,” Lett. Math. Phys. 112, 9 (2022), [arXiv:2106.13780](https://arxiv.org/abs/2106.13780).
16. <a id="ref-bmns"></a>Bachmann, Michalakis, Nachtergaele, Sims, “Automorphic equivalence within gapped phases of quantum lattice systems,” Commun. Math. Phys. 309, 835 (2012), [arXiv:1102.0842](https://arxiv.org/abs/1102.0842).
17. <a id="ref-dap"></a>D'Ariano, Perinotti, “Derivation of the Dirac equation from principles of information processing,” Phys. Rev. A 90, 062106 (2014), [arXiv:1306.1934](https://arxiv.org/abs/1306.1934).
18. <a id="ref-bdt"></a>Bisio, D'Ariano, Tosini, “Quantum field as a quantum cellular automaton: the Dirac free evolution in one dimension,” Ann. Phys. 354, 244 (2015), [arXiv:1212.2839](https://arxiv.org/abs/1212.2839).
19. <a id="ref-mb"></a>Mlodinow, Brun, “Bounds on QCA lattice spacing from data on Lorentz violation,” Phys. Rev. D 112, 074513 (2025), [arXiv:2506.20136](https://arxiv.org/abs/2506.20136).
