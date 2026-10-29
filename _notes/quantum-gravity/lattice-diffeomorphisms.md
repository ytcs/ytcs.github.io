---
title: "Lattice gravitons, exact diffeomorphisms and lattice momentum"
date: 2026-10-29
category: gravity
order: 1
description: In a classical, perturbative Hamiltonian lattice theory of gravitons without three-wave resonances, an exactly conserved lattice momentum with an analytic symbol is incompatible with non-zero low-energy graviton scattering, and making the momentum exact by band-limiting the vertices singles out a frame and, in a one-loop model of scalar electrodynamics with a ball-shaped band, shifts the speed of light by a few parts in ten thousand.
status: "Theorems 1 and 2 and Propositions 1 to 4 are proved under the stated hypotheses. That exact deformed diffeomorphisms supply the conserved charge, the form of the low-energy amplitude and the gauge-closure condition (G) of section 6.2 are assumptions. The one-loop coefficients are numerical, for scalar electrodynamics with a ball-shaped band."
excludes: "Classical Hamiltonian theories of gravitons in continuous time on a fixed cubic lattice, expanded order by order in the coupling, with no three-wave resonances and coefficient functions that are continuous away from vanishing momenta, that carry a conserved lattice momentum with a real-analytic symbol equal to the wave vector at low energy and the same for both helicities, while the low-energy two-to-two graviton interaction does not vanish near the massless resonant set; and modified-Villain lattices in which every component of the field is compact, with a finite-range integer curvature matrix and bounded kinetic and potential symbols, that have integer momentum-constraint charges and a linear graviton dispersion."
---

The question is whether an interacting graviton theory on a spatial lattice can have an exactly conserved lattice momentum built from exponentially local terms. In the continuum, constant diffeomorphisms translate the fields, so exact, deformed diffeomorphism invariance on a lattice would be expected to bring such a charge with it. Section 2 takes that passage as a hypothesis and does not derive it. The answer is no when the charge has a real-analytic symbol $$P(k)$$ (Theorem 1). The theory is a classical Hamiltonian theory in continuous time on a fixed cubic lattice, expanded order by order in the gravitational coupling. The free dispersion has no three-wave resonances, as the nearest-neighbour graviton has none. The charge acts in the same way on both helicities, with $$P(k)=k+O(k^2)$$ at small wave vector, and the coefficient functions of the interaction and of the charge are continuous away from vanishing momenta. The low-energy $$2\to2$$ graviton interaction is non-zero near the massless resonant set. Conservation of the charge in low-energy scattering then forces $$P(k)=k$$ near the origin, hence everywhere by analyticity, and that contradicts the periodicity of $$P$$ on the Brillouin zone.

The theorem does not apply to a charge with a non-analytic symbol. If every vertex vanishes when one of its legs is near the zone boundary, no scattering process is an Umklapp process, one in which the total momentum wraps around the zone, and a charge equal to $$k$$ near the origin is conserved exactly. Section 6 gives the cost of this option. No band of momenta is invariant under the Lorentz group of any metric (Proposition 3). If the band must be closed under the nonlinear gauge transformations, it is the whole zone (Proposition 2 and the closure requirement (G)). In a one-loop model of band-limited scalar electrodynamics with a ball-shaped band, the photon speed shifts by $$\delta c_\gamma=(1.8\text{–}7.8)\times10^{-4}$$ independently of the band edge, the charged scalar has a different shift, and the photon acquires a mass of a few hundredths of the band edge. Section 7 treats a separate charge, the integer momentum-constraint charge of a compact (Villain) lattice, and shows that it forces $$\omega\propto k^2$$ (Theorem 2).

Linearised Regge calculus has exact linear gauge modes [[Dittrich–Freidel–Speziale]](#ref-dfs) [[Höhn]](#ref-hoehn). Beyond linear order discretisation generically breaks diffeomorphism symmetry [[Bahr–Dittrich 2009a]](#ref-bd1), and exact symmetry has been proposed to be recoverable with non-local "perfect" actions [[Bahr–Dittrich 2009b]](#ref-bd2) [[Dittrich]](#ref-dittrich). That crystal momentum is a conserved momentum when no Umklapp process occurs is due to Peierls [[Peierls]](#ref-peierls). Lewis, Kempf and Menicucci showed that bandlimited lattice models carry observables that generate continuous translations [[LKM]](#ref-lkm), and Proposition 1 states this observation in the present language. Spohn determined the collision invariants of the phonon Boltzmann equation with pair collisions over the whole zone [[Spohn]](#ref-spohn), which gives a second route to Theorem 1 (a remark in section 5). Collins et al. showed that in a Yukawa theory a preferred-frame cutoff produces Lorentz violation of order $$g^2/6\pi^2$$ times a factor fixed by the cutoff profile, which does not decrease as the cutoff is raised [[Collins et al.]](#ref-collins). Compact lattice tensor gauge theories in Villain form have soft gravitons with $$\omega\propto k^2$$ [[Xu]](#ref-xu) [[Xu–Hořava]](#ref-xh) [[GLSS 2021]](#ref-glss1) [[GLSS 2022]](#ref-glss2).

This page adds Theorem 1, whose proof uses only low-energy data, the connectedness of the massless resonant set and the identity theorem for analytic functions; the lemma on collision invariants of massless kinematics in section 4, which [the page on exact universal graviton coupling](/quantum-gravity/spin-2-soft-locking/) uses in a second setting; Theorem 2 for the whole Villain class; the proof that no band is Lorentz invariant; and the one-loop light-cone shifts and photon mass produced by band-limited vertices. The lattice spacing is 1 and the low-energy speed of light is 1.

## 1. Linearised gravity in ADM form and on a lattice

### 1.1 The continuum theory

In the canonical (ADM) description of gravity the dynamical variables are the spatial metric $$\gamma_{ij}$$ and its conjugate momentum $$\pi^{ij}$$. The lapse $$N$$ and the shift $$N^i$$ are Lagrange multipliers, and the Hamiltonian is $$\int(N\mathcal H+N^i\mathcal H_i)$$ with the Hamiltonian constraint $$\mathcal H$$ and the momentum constraint $$\mathcal H_i$$. Expand around flat space, $$\gamma_{ij}=\delta_{ij}+\kappa h_{ij}$$, where $$\kappa$$ is the gravitational coupling. Rescaling $$h$$ and $$\pi$$ by $$\kappa$$ removes $$\kappa$$ from the quadratic theory, and equations (1)–(3) are written with this rescaling, in units with $$16\pi G=1$$. To linear order the constraints are

$$
C_i=-2\,\partial_j\pi_{ij}\approx0,\qquad C=\partial_i\partial_jh_{ij}-\partial_k\partial_kh_{ii}\approx0, \tag{1}
$$

where $$\approx$$ means equal to zero on physical states and repeated indices are summed. The Hamiltonian at second order is

$$
H_2=\int d^3x\,\Big[\pi_{ij}\pi_{ij}-\tfrac12\pi_{kk}\pi_{ll}+\mathcal U(h)\Big],\qquad
\mathcal U(h)=\tfrac14\partial_kh_{ij}\partial_kh_{ij}-\tfrac12\partial_ih_{ij}\partial_kh_{kj}+\tfrac12\partial_ih_{ij}\partial_jh_{kk}-\tfrac14\partial_kh_{ii}\partial_kh_{jj}. \tag{2}
$$

The kinetic term is the DeWitt form. The potential $$\mathcal U$$ is minus the second-order part of $$\sqrt\gamma R$$, which is the Fierz–Pauli form. The smeared momentum constraint $$D[\xi]=\int\xi_iC_i=\int\pi_{ij}(\partial_i\xi_j+\partial_j\xi_i)$$ generates the gauge transformations

$$
\delta_\xi h_{ij}=(S\xi)_{ij}\equiv\partial_i\xi_j+\partial_j\xi_i, \tag{3}
$$

and $$\mathcal U$$ is invariant under (3) because the linearised curvature of a pure-gauge $$h$$ vanishes. In Fourier space the shift is $$h\to h+k\otimes\xi+\xi\otimes k$$, and $$\mathcal U$$ is unchanged. The Hamiltonian constraint generates shifts of $$\pi_{ij}$$ by $$(\partial_i\partial_j-\delta_{ij}\partial_k\partial_k)\theta$$; the kinetic term is invariant under these only up to terms proportional to the constraints.

At each wave vector $$k\ne0$$ the phase space has twelve dimensions, and the four first-class constraints (1) remove eight, leaving the transverse traceless components. For them $$H_2=\int[\pi^{\rm TT}_{ij}\pi^{\rm TT}_{ij}+\tfrac14(\partial_kh^{\rm TT}_{ij})^2]$$, so there are two modes, of helicity $$\pm2$$, with $$\omega=\lvert k\rvert$$. For $$k$$ along the third axis, $$\mathcal U$$ takes the value $$+\tfrac12$$ on $$h=\mathrm{diag}(1,-1,0)$$ and $$-\tfrac12$$ on the transverse-trace mode $$h=\mathrm{diag}(1,1,0)$$. The potential is therefore indefinite on the gauge-invariant modes, and it is the Hamiltonian constraint that removes the negative direction. The page on [gravitons from qubits, rotors and composite tetrads](/quantum-gravity/emergent-spin-2/) uses this fact.

Beyond linear order the structure persists with deformations. The full momentum constraint generates the Lie derivative of $$\gamma_{ij}=\delta_{ij}+\kappa h_{ij}$$:

$$
\delta_\xi h_{ij}=(S\xi)_{ij}+\kappa\big(\xi_k\partial_kh_{ij}+h_{kj}\partial_i\xi_k+h_{ik}\partial_j\xi_k\big)+O(\kappa^2). \tag{4}
$$

For constant $$\xi$$ the linear term vanishes and what remains is the translation $$\xi_k\partial_kh_{ij}$$ of the field. This is the reason to expect that an exact diffeomorphism symmetry on a lattice comes with an exactly conserved momentum, and section 2 makes the expectation into a hypothesis.

### 1.2 Fourier analysis on the lattice

Let the sites be $$\mathbb Z^3$$ with spacing 1. A field $$\phi_x$$ has the Fourier coefficient $$\phi(k)=\sum_xe^{-ik\cdot x}\phi_x$$ with $$k$$ in the Brillouin torus $$T^3=(\mathbb R/2\pi\mathbb Z)^3$$. A translation-invariant linear operator acts on $$\phi(k)$$ as multiplication by a matrix-valued function of $$k$$, its *symbol*. Three facts are used below. A symbol of finite range is a trigonometric polynomial. If the kernel of the operator decays like $$e^{-\mu\lvert x\rvert}$$, the symbol extends holomorphically to the strip $$\lvert\mathrm{Im}\,k\rvert<\mu$$; conversely a real-analytic function on the compact torus extends to some strip, so its Fourier coefficients decay exponentially. *Exponentially local* therefore means *real-analytic symbol*. Finally, translation invariance conserves momentum only modulo $$2\pi$$: a process with incoming momenta $$k_1,k_2$$ and outgoing momenta $$k_3,k_4$$ is *normal* if $$k_1+k_2-k_3-k_4=0$$ and *Umklapp* if the difference is a non-zero multiple of $$2\pi$$.

To put (1)–(3) on the lattice, place the diagonal components $$h_{ii}$$ on sites, the off-diagonal components $$h_{ij}$$ on plaquette centres and $$\xi_i$$ on links, and replace every derivative by the centred difference $$d_i\phi(x)=\phi(x+\tfrac12e_i)-\phi(x-\tfrac12e_i)$$. Each $$d_i$$ then maps a field on one sublattice to a field on another. In the potential (2) the two differentiated fields in every term live on the same sublattice. For $$i\ne j$$ the component $$h_{ij}$$ sits at $$x+\tfrac12e_i+\tfrac12e_j$$ and for $$i=j$$ the component $$h_{jj}$$ sits at $$x$$, so in both cases $$d_ih_{ij}$$ sits, modulo lattice translations, on the link $$x+\tfrac12e_j$$, and so does $$d_jh_{kk}$$. This covers the terms $$d_ih_{ij}\,d_kh_{kj}$$ and $$d_ih_{ij}\,d_jh_{kk}$$. The remaining terms $$(d_kh_{ij})^2$$ and $$d_kh_{ii}\,d_kh_{jj}$$ pair derivatives that sit at the same offset by construction. Likewise $$d_i\xi_j+d_j\xi_i$$ lands on the plaquette centre for $$i\ne j$$ and on the site for $$i=j$$, which is where $$h_{ij}$$ lives. After absorbing the phase of each sublattice into the field, $$d_i$$ has symbol $$i\,v_i(k)$$ with

$$
v_i(k)=2\sin(k_i/2). \tag{5}
$$

Every identity of section 1.1 is a polynomial identity in the commuting variables $$\partial_i$$, so it holds on the lattice with $$k$$ replaced by the real vector $$v(k)$$. The linear gauge invariance is therefore exact, and at each $$k\ne0$$ there are two physical modes with the common frequency

$$
\omega(k)=\lvert v(k)\rvert=2\Big(\sum_i\sin^2(k_i/2)\Big)^{1/2}. \tag{6}
$$

Since $$v(k)$$ vanishes in the zone $$(-\pi,\pi]^3$$ only at $$k=0$$, there are no doublers, that is, no further zeros of the symbol that would give extra low-energy modes, and $$\omega(k)=\lvert k\rvert\,(1+O(k^2))$$. We call this the *nearest-neighbour graviton*. The question of sections 2–6 is what happens when interactions are added. Section 7 returns to the free theory and asks whether its gauge group can be made compact.

## 2. The conserved lattice momentum: setting and hypotheses

Write the physical graviton modes through complex amplitudes $$a_\sigma(k)$$, with helicity $$\sigma=\pm$$, $$k\in T^3$$ and Poisson bracket $$\{a_\sigma(k),a^*_{\sigma'}(k')\}=-i\delta_{\sigma\sigma'}\delta(k-k')$$. We work with the classical field theory, which is the tree-level content of the quantum theory, and with formal power series in the coupling $$\kappa$$:

$$
H=H_2+\kappa H_3+\kappa^2H_4+\dots,\qquad H_2=\sum_\sigma\int\omega(k)\,\lvert a_\sigma(k)\rvert^2\,dk, \tag{7}
$$

where $$H_r$$ is a polynomial of degree $$r$$ in $$a$$ and $$a^*$$ whose monomials conserve momentum modulo $$2\pi$$. For a monomial $$w$$ let $$\Omega(w)$$ and $$\mathcal P(w)$$ be the sums of $$\omega(k)$$ and of $$P(k)$$ over its factors, with sign $$+$$ for each $$a$$ and $$-$$ for each $$a^*$$. Then

$$
\{w,H_2\}=-i\,\Omega(w)\,w,\qquad \{w,Q_2\}=-i\,\mathcal P(w)\,w, \tag{8}
$$

where $$Q_2=\sum_\sigma\int P(k)\lvert a_\sigma(k)\rvert^2dk$$ is the quadratic part of the charge introduced below.

The gauge generators suggest the charge. The deformed momentum constraint is $$D[\xi]=\langle\pi,S\xi\rangle+\kappa D_3[\xi]+\dots$$, with $$(S\xi)_{ij}=d_i\xi_j+d_j\xi_i$$ the lattice symmetrised gradient. For constant $$\xi$$ the linear term vanishes, because a lattice difference of a constant is zero, and $$D[\xi]/\kappa$$ starts at quadratic order. Its quadratic part generates the transport term $$\xi\cdot\partial h$$ of (4), which for constant $$\xi$$ translates the long-wavelength fields, and its symbol is periodic because the lattice is. If the generators are exponentially local, this symbol is real-analytic by section 1.2.

This is a motivation and not a derivation. $$D[\xi]$$ with constant $$\xi$$ is itself a constraint, so it vanishes on physical states, and a conserved graviton momentum appears only after the constraints are solved, with the total momentum carried by a boundary term as in the ADM construction. On a lattice that identification needs an argument that is not given here. That the resulting charge acts diagonally on the physical modes, with the symbol $$k+O(k^2)$$ and the same for both helicities, is a further assumption, and the continuity of its higher terms must be inherited from the generators. Theorem 1 is therefore stated for a conserved charge with the following properties.

(H1) The dispersion $$\omega$$ is continuous on $$T^3$$, positive away from $$k=0$$, the same for both helicities, and relativistic at low energy: $$\omega(\lambda q)/\lambda\to\lvert q\rvert$$ as $$\lambda\to0$$, in $$C^1$$ on compact subsets of $$q\ne0$$.

(H2) There are no three-wave resonances: $$\omega(k_1)+\omega(k_2)>\omega(k_1+k_2)$$ whenever $$k_1\ne0$$ and $$k_2\ne0$$.

(H3) There is a charge $$Q=Q_2+\kappa Q_3+\kappa^2Q_4+\dots$$ with $$\{Q,H\}=0$$ order by order in $$\kappa$$. Its quadratic part is $$Q_2$$ above, with $$P:T^3\to\mathbb R^3$$ continuous, $$P(0)=0$$ and $$P(k)=k+O(k^2)$$. The coefficient functions of $$Q_3,Q_4,H_3,H_4$$ are continuous, in any continuous local choice of polarisation basis, at all momenta at which no internal or external momentum vanishes. A continuous choice of helicity basis exists only locally on the sphere of directions, so global continuity cannot be asked of helicity-basis coefficients. Exponentially local terms have this property.

(H4) $$P$$ is real-analytic on $$T^3$$.

A fifth hypothesis (H5), that the low-energy $$2\to2$$ interaction does not vanish near the massless resonant set, refers to the sets defined in section 4 and is stated there. Theorem 1 restates all five.

The nearest-neighbour graviton satisfies (H1) by (6), and (H2) by the following lemma.

<div class="lemma" markdown="1">
**Lemma 1.** The dispersion (6) satisfies (H2).
</div>

<div class="proof" markdown="1">
*Proof.* For real $$\phi,\psi$$, $$\lvert\sin\tfrac{\phi+\psi}2\rvert\le\lvert\sin\tfrac\phi2\rvert\lvert\cos\tfrac\psi2\rvert+\lvert\cos\tfrac\phi2\rvert\lvert\sin\tfrac\psi2\rvert\le\lvert\sin\tfrac\phi2\rvert+\lvert\sin\tfrac\psi2\rvert$$, and equality in the last step requires $$\phi$$ or $$\psi$$ to lie in $$2\pi\mathbb Z$$. Applied componentwise, the vector $$(\lvert v_j(k_1+k_2)\rvert)_j$$ is dominated entry by entry by $$(\lvert v_j(k_1)\rvert+\lvert v_j(k_2)\rvert)_j$$, so the triangle inequality gives $$\omega(k_1+k_2)\le\omega(k_1)+\omega(k_2)$$. Equality needs equality in every component, so that in each direction $$j$$ one of $$v_j(k_1),v_j(k_2)$$ vanishes, and equality in the triangle inequality for the vectors $$(\lvert v_j(k_1)\rvert)_j$$ and $$(\lvert v_j(k_2)\rvert)_j$$, which then must be parallel. Two parallel non-negative vectors with disjoint supports cannot both be non-zero, so $$k_1=0$$ or $$k_2=0$$.
</div>

The gap in (H2) is not bounded below by a multiple of $$\min(\lvert k_1\rvert,\lvert k_2\rvert)$$. For parallel $$k_1,k_2$$ of size $$s$$ along an axis it equals $$4\sin(s/2)-2\sin s\approx s^3/4$$. It is non-zero for every non-zero pair, and that is all the next section uses.

## 3. The resonance condition

With (H2), the cubic part of $$H$$ can be removed by a canonical change of variables, the first step of Birkhoff normal form: a canonical map that eliminates the cubic term, which is non-resonant, and leaves an effective quartic interaction. By (8) and positivity of $$\omega$$, $$\Omega(w)\ne0$$ for every cubic monomial whose momenta are non-zero: a monomial with three factors of the same kind has $$\lvert\Omega\rvert$$ equal to a sum of positive frequencies, and one with two factors of one kind and one of the other has $$\Omega\ne0$$ by (H2) together with momentum conservation modulo $$2\pi$$. Let $$\Gamma_3$$ be the cubic functional whose coefficient on each such monomial is that of $$H_3$$ divided by $$-i\,\Omega(w)$$, so that $$\{H_2,\Gamma_3\}=-H_3$$ by (8). The Lie series $$F\mapsto F+\kappa\{F,\Gamma_3\}+\tfrac12\kappa^2\{\{F,\Gamma_3\},\Gamma_3\}+\dots$$ is a canonical transformation and maps $$H$$ to $$H'=H_2+\kappa^2H_4'+O(\kappa^3)$$ with

$$
H_4'=H_4+\tfrac12\{H_3,\Gamma_3\}.
$$

The factor $$\tfrac12$$ is the net of the first-order term $$\{H_3,\Gamma_3\}$$ and the second-order term $$\tfrac12\{\{H_2,\Gamma_3\},\Gamma_3\}=-\tfrac12\{H_3,\Gamma_3\}$$. The same transformation maps $$Q$$ to $$Q'=Q_2+\kappa Q_3'+\kappa^2Q_4'+\dots$$ with the same $$Q_2$$, where $$Q_3'=Q_3+\{Q_2,\Gamma_3\}$$ and $$Q_4'=Q_4+\{Q_3,\Gamma_3\}+\tfrac12\{\{Q_2,\Gamma_3\},\Gamma_3\}$$. On the resonant monomials $$a_1^*a_2^*a_3a_4$$ with $$\omega_1+\omega_2=\omega_3+\omega_4$$ the coefficient $$\mathcal T(1,2;3,4)$$ of $$H_4'$$ is the effective $$2\to2$$ interaction: contact term plus graviton exchange, with the energy denominators of second-order perturbation theory. It carries helicity labels, which we suppress. At tree level it is the $$2\to2$$ scattering amplitude up to non-vanishing normalisation factors; this identification enters through (H5).

<div class="lemma" markdown="1">
**Lemma 2** (resonance condition). Under (H1)–(H3), at every resonant quartet at which all $$k_i\ne0$$ and none of the exchanged momenta $$k_1-k_3$$, $$k_1-k_4$$, $$k_1+k_2$$ vanishes (modulo $$2\pi$$),

$$
\mathcal T(1,2;3,4)\,\big(P(k_1)+P(k_2)-P(k_3)-P(k_4)\big)=0. \tag{9}
$$

</div>

<div class="proof" markdown="1">
*Proof.* Canonical transformations preserve brackets, so $$\{Q',H'\}=0$$ order by order. At order $$\kappa$$ this reads $$\{Q_3',H_2\}=0$$. By (8) every coefficient of $$Q_3'$$ is multiplied by $$\Omega(w)\ne0$$ and so vanishes: $$Q_3'=0$$. At order $$\kappa^2$$ it reads $$\{Q_4',H_2\}+\{Q_2,H_4'\}=0$$. By (8) the coefficient of a quartic monomial $$w$$ in this sum is $$-i\,\Omega(w)$$ times its coefficient $$c_Q(w)$$ in $$Q_4'$$ plus $$i\,\mathcal P(w)$$ times its coefficient $$c_H(w)$$ in $$H_4'$$, so

$$
\Omega(w)\,c_Q(w)=\mathcal P(w)\,c_H(w).
$$

The coefficients of $$\Gamma_3$$ have denominators $$\Omega\ne0$$, and the coefficient functions of $$H_3,H_4,Q_3,Q_4$$ are continuous by (H3), so $$c_Q$$ and $$c_H$$ are continuous at the quartets of the statement: the contractions in $$\{H_3,\Gamma_3\}$$, $$\{Q_3,\Gamma_3\}$$ and $$\{\{Q_2,\Gamma_3\},\Gamma_3\}$$ run over one internal line whose momentum is one of the exchanged momenta, which is non-zero there. A continuous identity between coefficient functions holds at every point of its domain and may be evaluated on the resonant set, where $$\Omega(w)=0$$ and the left side vanishes. (Without continuity the resonant set, which has measure zero, would not be reached: a distributional coefficient proportional to $$\mathcal P\,\mathcal T/\Omega$$ satisfies the identity for any $$P$$.) This leaves $$\mathcal P(w)\,c_H(w)=0$$ with $$\mathcal P(a_1^*a_2^*a_3a_4)=P(k_3)+P(k_4)-P(k_1)-P(k_2)$$. The helicity labels do not enter, because $$P$$ is the same for both.
</div>

Equation (9) says that on every resonant quartet either the effective $$2\to2$$ interaction vanishes or the charge is additive.

## 4. Collisions of massless particles

At low energy the resonant quartets of the lattice approach those of massless particles. Let $$\Lambda=\{q\in\mathbb R^{12}:q_1+q_2=q_3+q_4\}$$ and let

$$
\mathcal M_\lambda=\{q\in\Lambda:\ \omega(\lambda q_1)+\omega(\lambda q_2)=\omega(\lambda q_3)+\omega(\lambda q_4)\}
$$

be the rescaled resonant set of normal quartets. Let $$\mathcal M_0=\{q\in\Lambda:\ \lvert q_1\rvert+\lvert q_2\rvert=\lvert q_3\rvert+\lvert q_4\rvert\}$$ be its limit, the massless resonant set, and $$\mathcal M_0^\ast$$ its part on which $$\lvert q_1\rvert+\lvert q_2\rvert>\lvert q_1+q_2\rvert$$. The fifth hypothesis is about the coefficient $$\mathcal T$$ of section 3 on these sets.

(H5) There are an open set $$U\subset\Lambda$$ meeting $$\mathcal M_0^\ast$$, a number $$\lambda_0>0$$, and a helicity configuration such that $$\mathcal T(\lambda q)\ne0$$ for all $$q\in\mathcal M_\lambda\cap U$$ and $$0<\lambda<\lambda_0$$.

For gravity this holds if the low-energy limit is general relativity, whose tree-level $$2\to2$$ graviton amplitude does not vanish on an open set of physical kinematics: with all momenta outgoing and Mandelstam invariants $$\mathsf s,\mathsf t,\mathsf u$$, the helicity configuration $$(-,-,+,+)$$ has modulus proportional to $$\kappa^2\mathsf s^3/(\mathsf t\mathsf u)$$. The identification of $$\mathcal T$$ with this amplitude for the lattice vertex is assumed and not computed.

Theorem 1 rests on two facts about $$\mathcal M_0$$. The first is its geometry. The second says that a function that adds up in every massless collision is, among homogeneous functions, at most linear in the momentum (degree 0 and 1 are the exceptions). The proof of the second fact does not depend on the lattice at all.

<div class="lemma" markdown="1">
**Lemma 3** (the massless resonant set). $$\mathcal M_0^\ast$$ is a connected real-analytic submanifold of $$\Lambda$$ of dimension 8. On it all $$q_i\ne0$$, and the gradient of $$F_0(q)=\lvert q_1\rvert+\lvert q_2\rvert-\lvert q_3\rvert-\lvert q_4\rvert$$ along $$\Lambda$$ does not vanish. For every $$z\in\mathbb R^3\setminus\{0\}$$ the quartet $$(3z,z;2z,2z)$$ lies in the closure of $$\mathcal M_0^\ast$$.
</div>

<div class="proof" markdown="1">
*Proof.* Write $$K=q_1+q_2$$ and $$E=\lvert q_1\rvert+\lvert q_2\rvert$$. If $$q\in\mathcal M_0^\ast$$, then $$q_3$$ lies on the spheroid $$\lvert q_3\rvert+\lvert K-q_3\rvert=E$$ with foci $$0$$ and $$K$$. It is a 2-sphere and $$E>\lvert K\rvert$$, so it avoids both foci: $$q_3\ne0$$ and $$q_4=K-q_3\ne0$$. Also $$E>\lvert K\rvert$$ forces $$q_1,q_2\ne0$$. Along $$\Lambda$$, with coordinates $$(q_1,q_2,q_3)$$ and $$q_4=q_1+q_2-q_3$$, the derivative of $$F_0$$ with respect to $$q_3$$ is $$-\hat q_3+\hat q_4$$ (hats denote unit vectors). It vanishes only if $$q_3$$ and $$q_4$$ point the same way, and then $$\lvert q_3\rvert+\lvert q_4\rvert=\lvert K\rvert<E$$, a contradiction. So $$F_0$$ is analytic with non-vanishing gradient near $$\mathcal M_0^\ast$$, and $$\mathcal M_0^\ast$$ is a real-analytic submanifold of dimension $$9-1=8$$.

Connectedness. Let $$\mathcal G$$ be the set of pairs $$(q_1,q_2)$$ of non-zero vectors that are not positively parallel; these are the pairs with $$E>\lvert K\rvert$$. The group $$SO(3)\times\mathbb R_{>0}$$ acts on $$\mathcal G$$, and every orbit contains a pair $$(e_3,\ r(\sin\theta\,e_1+\cos\theta\,e_3))$$ with $$r>0$$ and $$\theta\in(0,\pi]$$. So $$\mathcal G$$ is the image of the connected set $$SO(3)\times\{(r,\theta)\}$$ and is connected. The map $$\pi:\mathcal M_0^\ast\to\mathcal G$$, $$q\mapsto(q_1,q_2)$$, is surjective with fibres the spheroids, which are connected. It is proper, because $$\lvert q_3\rvert,\lvert q_4\rvert\le E$$ and the defining conditions are closed, hence closed as a map. A closed continuous surjection with connected fibres onto a connected space has connected domain: if $$\mathcal M_0^\ast=A\cup A'$$ were a splitting into disjoint non-empty open and closed sets, every fibre would lie in one of them, so $$\pi(A)$$ and $$\pi(A')$$ would be disjoint, non-empty, closed and cover $$\mathcal G$$.

The collinear quartet. Take $$z\ne0$$ and a unit vector $$z_\perp\perp z$$. For small $$\varepsilon\ne0$$ let $$q_1=3z$$ and $$q_2=z+\varepsilon z_\perp$$. Then $$K=4z+\varepsilon z_\perp$$ and $$E-\lvert K\rvert>0$$, so $$(q_1,q_2)\in\mathcal G$$. Put $$t=\tfrac12\sqrt{E^2-\lvert K\rvert^2}$$ and $$q_3=\tfrac12K+t\,u$$, $$q_4=\tfrac12K-t\,u$$ with $$u\perp K$$ a unit vector; then $$\lvert q_3\rvert+\lvert q_4\rvert=2\sqrt{\lvert K\rvert^2/4+t^2}=E$$, so $$q\in\mathcal M_0^\ast$$. As $$\varepsilon\to0$$, $$t\to0$$ and $$q\to(3z,z,2z,2z)$$.
</div>

<div class="lemma" markdown="1">
**Lemma 4** (homogeneous collision invariants). Let $$g:\mathbb R^3\setminus\{0\}\to\mathbb R^n$$ be real-analytic and positively homogeneous of a real degree $$m$$, $$g(tq)=t^mg(q)$$ for $$t>0$$. Put $$\Delta g(q)=g(q_1)+g(q_2)-g(q_3)-g(q_4)$$. If $$\Delta g$$ vanishes on a non-empty relatively open subset of $$\mathcal M_0^\ast$$, then $$g\equiv0$$ unless $$m\in\{0,1\}$$. For $$m=1$$ the linear functions $$g(q)=Aq$$ and for $$m=0$$ the constants do satisfy $$\Delta g=0$$ on all of $$\mathcal M_0$$, and so does $$g(q)=\lvert q\rvert$$ for $$m=1$$.
</div>

<div class="proof" markdown="1">
*Proof.* The function $$\Delta g$$ is real-analytic on the connected analytic manifold $$\mathcal M_0^\ast$$ (Lemma 3, and all $$q_i\ne0$$ there), so it vanishes identically on it by the identity theorem, and by continuity on the closure within the region where all $$q_i\ne0$$. The quartet $$(3z,z;2z,2z)$$ has all vectors non-zero, so $$\Delta g=0$$ there, and homogeneity gives

$$
\big(3^m+1-2^{m+1}\big)\,g(z)=0 .
$$

The function $$\psi(m)=(3^m+1-2^{m+1})/2^m=(3/2)^m+2^{-m}-2$$ is strictly convex in $$m$$, so it has at most two zeros, and $$m=0$$ and $$m=1$$ are zeros. For every other $$m$$ the prefactor is non-zero and $$g(z)=0$$. The last sentence follows from momentum conservation, $$\sum\pm q_i=0$$, and energy conservation, $$\sum\pm\lvert q_i\rvert=0$$ on $$\mathcal M_0$$.
</div>

For integer degree $$m=2,3,\dots$$ the factor $$3^m+1-2^{m+1}$$ is $$2,12,50,180,\dots$$. A numerical rank test on $$6000$$ sampled massless collisions agrees with Lemma 4 for polynomials: the matrix of $$\Delta g$$ on the monomials of degree one has exactly three zero singular values, the linear functions, and for degrees two to six it has none.

## 5. Analytic symbols are excluded

Equation (9) at low energy sees only normal quartets, on which $$P(k)=k$$ satisfies it exactly. Any analytic correction to $$k$$ is a sum of homogeneous polynomials, and the lowest one would have to be a collision invariant of massless $$2\to2$$ kinematics. By Lemma 4 none exists in degree two or more. So $$P$$ equals $$k$$ near $$0$$, hence everywhere by analyticity, which is incompatible with periodicity.

<div class="theorem" markdown="1">
**Theorem 1** (no conserved lattice momentum with an analytic symbol). Let $$H=H_2+\kappa H_3+\kappa^2H_4+\dots$$ have the form (7) and let $$Q=Q_2+\kappa Q_3+\kappa^2Q_4+\dots$$ satisfy $$\{Q,H\}=0$$ order by order in $$\kappa$$, with $$Q_2$$ as in (8). The following five hypotheses cannot hold together.

(H1) $$\omega$$ is continuous, positive away from $$k=0$$, the same for both helicities, and $$\omega(\lambda q)/\lambda\to\lvert q\rvert$$ in $$C^1$$ on compact subsets of $$q\ne0$$.

(H2) $$\omega(k_1)+\omega(k_2)>\omega(k_1+k_2)$$ for $$k_1,k_2\ne0$$.

(H3) $$P$$ is continuous with $$P(k)=k+O(k^2)$$ at $$0$$, and the coefficient functions of $$Q_3,Q_4,H_3,H_4$$ are continuous, in a continuous local choice of polarisation basis, wherever no internal or external momentum vanishes.

(H4) $$P$$ is real-analytic on $$T^3$$.

(H5) For some open $$U\subset\Lambda$$ meeting $$\mathcal M_0^\ast$$, some $$\lambda_0>0$$ and some helicity configuration, $$\mathcal T(\lambda q)\ne0$$ for all $$q\in\mathcal M_\lambda\cap U$$ and $$0<\lambda<\lambda_0$$.

Equivalently, if (H1)–(H4) hold, then for every open $$U\subset\Lambda$$ meeting $$\mathcal M_0^\ast$$, every $$\lambda_0>0$$ and every helicity configuration there are $$0<\lambda<\lambda_0$$ and $$q\in\mathcal M_\lambda\cap U$$ with $$\mathcal T(\lambda q)=0$$.
</div>

<div class="proof" markdown="1">
*Proof.* Write $$P(k)=k+R(k)$$, with $$R$$ real-analytic near $$0$$ and $$R=\sum_{p\ge2}R_p$$ its expansion in vector-valued homogeneous polynomials. Suppose $$R\not\equiv0$$ near $$0$$ and let $$p_0\ge2$$ be the least $$p$$ with $$R_p\ne0$$. For a function $$f$$ write $$\Delta f=f(k_1)+f(k_2)-f(k_3)-f(k_4)$$.

*Step 1 (low-energy quartets are normal).* If every component of every $$k_i$$ lies in $$(-\pi/2,\pi/2)$$, then $$k_1+k_2-k_3-k_4$$ has components in $$(-2\pi,2\pi)$$ and is a multiple of $$2\pi$$, so it vanishes. For small $$\lambda$$ the quartets $$\lambda q$$ with $$q$$ in a bounded set are therefore normal, and $$\Delta k=0$$ there. Hence $$\Delta P(\lambda q)=\Delta R(\lambda q)=\lambda^{p_0}\,\Delta R_{p_0}(q)+O(\lambda^{p_0+1})$$.

*Step 2 (limit to the massless resonant set).* Set $$F_\lambda(q)=\lambda^{-1}[\omega(\lambda q_1)+\omega(\lambda q_2)-\omega(\lambda q_3)-\omega(\lambda q_4)]$$ on $$\Lambda$$, so that $$\mathcal M_\lambda=\{F_\lambda=0\}$$. By (H1), $$F_\lambda\to F_0$$ in $$C^1$$ near any point where all $$q_i\ne0$$. Choose a point $$q^\ast\in U\cap\mathcal M_0^\ast$$ at which also $$q_1\ne q_3$$, $$q_1\ne q_4$$ and $$q_1+q_2\ne0$$, so that the exchanged momenta of Lemma 2 do not vanish at the nearby quartets $$\lambda q$$. These conditions fail only on lower-dimensional subsets, so such points exist, and we may shrink $$U$$ to a neighbourhood of $$q^\ast$$ on which they hold. By Lemma 3 the gradient of $$F_0$$ at $$q^\ast$$ is non-zero, so the implicit function theorem shows that every $$q\in\mathcal M_0$$ near $$q^\ast$$ is a limit of points $$q_\lambda\in\mathcal M_\lambda\cap U$$. By Lemma 2 and (H5), $$\Delta P(\lambda q_\lambda)=0$$, so Step 1 gives $$\Delta R_{p_0}(q_\lambda)=O(\lambda)$$ and, in the limit, $$\Delta R_{p_0}(q)=0$$. The polynomial map $$R_{p_0}$$ is real-analytic and homogeneous of degree $$p_0\ge2$$, so Lemma 4 gives $$R_{p_0}\equiv0$$. This is a contradiction. The weight of the proof is in this step: it uses the lattice only through the $$C^1$$ convergence in (H1), and everything else is the geometry of massless collisions.

*Step 3 (periodicity).* Hence $$R_p=0$$ for all $$p$$, so the Taylor series of $$R$$ at $$0$$ vanishes, and $$R\equiv0$$ near $$0$$ by analyticity. Lift $$P$$ to a real-analytic periodic function on $$\mathbb R^3$$. Then $$P(k)-k$$ is real-analytic on the connected set $$\mathbb R^3$$ and vanishes on an open set, so $$P(k)=k$$ everywhere, and $$P(k+2\pi e_1)=P(k)+2\pi e_1\ne P(k)$$ contradicts periodicity.
</div>

For the nearest-neighbour dispersion and the analytic charge $$P=\sin k$$ (componentwise), exactly resonant normal quartets $$\lambda q$$ can be constructed for fixed $$q_1,q_2$$ and a fixed direction of $$q_3$$. In six such configurations $$\max_a\lvert\Delta P_a\rvert/\lambda^3$$ changes by less than $$9\%$$ between $$\lambda=0.2$$ and $$\lambda=0.0125$$ (for instance from $$1.464$$ to $$1.506$$ and from $$8.633$$ to $$9.392$$), so $$\Delta P$$ scales as $$\lambda^3$$ with a non-zero coefficient, as Step 1 predicts for $$p_0=3$$. For the first configuration at $$\lambda=0.0125$$ the three components of $$\Delta P/\lambda^3$$ are $$(-0.20139,-1.50591,-0.00120)$$, against $$(-0.20140,-1.50606,-0.00120)$$ from the leading term $$-\tfrac16\sum\pm k_i^3$$.

<div class="remark" markdown="1">
**Remark** (where the hypotheses enter). Steps 1 and 2 need only smoothness of $$P$$, with $$R_p$$ read as the Taylor terms of $$R$$; for a smooth $$P$$ they show that the Taylor series of $$R$$ at $$0$$ vanishes. Analyticity enters in Step 3, twice: a vanishing Taylor series forces $$R\equiv0$$ near $$0$$, and a function equal to $$k$$ on an open set equals $$k$$ everywhere. A smooth $$P$$ that equals $$k$$ on a neighbourhood of $$0$$ passes Steps 1 and 2, so the theorem says nothing against smooth non-analytic charges. Both uses of analyticity are uses of the identity theorem, and they hold verbatim in any class of smooth periodic functions in which vanishing of the Taylor series at a point forces vanishing on a connected domain (a quasi-analytic class in the sense of Denjoy and Carleman). The theorem therefore also covers charges in such a class; which Fourier decay rates define one is not characterised here. Only low-energy normal quartets are used, so Umklapp processes play no role. The proof is an identity-theorem argument on a nonlinear manifold, and finite range is a special case of analytic.
</div>

<div class="remark" markdown="1">
**Remark** (the vertex side). A second route uses the vertex instead of the charge. Spohn [[Spohn]](#ref-spohn) proved that a measurable periodic function $$f$$ that is locally integrable and satisfies $$f(k_1)+f(k_2)=f(k_3)+f(k_4)$$ almost everywhere on the full resonant manifold of the lattice, Umklapp quartets included, has the form $$c_0+c_1\omega(k)$$, under regularity assumptions on the dispersion relation (twice differentiable off a codimension-one set, with degenerate Hessian only on such a set) and in dimension two or more. We do not verify those assumptions for (6). If $$\mathcal T\ne0$$ almost everywhere on that manifold, (9) makes each component of $$P$$ such an invariant, and $$c_0+c_1\omega$$ is even in $$k$$ for the nearest-neighbour dispersion, so it cannot equal $$k+O(k^2)$$. This route needs no analyticity of $$P$$ but needs $$\mathcal T\ne0$$ almost everywhere on the whole zone. That would follow from analyticity of $$\mathcal T$$ together with connectedness of the lattice resonant manifold after removal of its degenerate subsets; the connectedness is not proved here.
</div>

## 6. Band-limited vertices

### 6.1 An exact momentum from Peierls' observation

Theorem 1 excludes analytic charges only. For a non-analytic $$P$$, (9) requires $$\mathcal T$$ to vanish on every resonant quartet with $$\Delta P\ne0$$: on Umklapp quartets, and on normal quartets outside the region where $$P$$ is linear. One way to meet this is to combine a non-analytic charge with vertices that never produce Umklapp processes.

<div class="proposition" markdown="1">
**Proposition 1** (no Umklapp; Peierls). Suppose that for each $$r\ge3$$ the coefficients of $$H_r$$ vanish unless every momentum $$k_i$$ of the monomial has all its components in $$(-2\pi/r,2\pi/r)$$. Let $$P$$ be any periodic function with values in $$\mathbb R^3$$ such that $$P(k)=k$$ when all components of $$k$$ lie in $$(-2\pi/3,2\pi/3)$$. Then $$\{Q_2,H_r\}=0$$ for every $$r$$, so $$Q_2$$ is conserved exactly, to all orders in $$\kappa$$.
</div>

<div class="proof" markdown="1">
*Proof.* By (8), the coefficient of a monomial $$w$$ in $$\{Q_2,H_r\}$$ is $$i\,\mathcal P(w)$$ times its coefficient in $$H_r$$. On the support of $$H_r$$ the signed sum of the $$r$$ momenta has components in $$(-2\pi,2\pi)$$ and is a multiple of $$2\pi$$, so it vanishes, and $$\mathcal P(w)$$ equals that signed sum because $$P(k)=k$$ there.
</div>

The free dispersion does not enter, so the nearest-neighbour $$H_2$$ can be used. Nor does the decay class of $$P$$ matter beyond non-analyticity: the sawtooth $$P(k)=k$$ on $$(-\pi,\pi]$$, which is discontinuous and decays like $$1/n$$, works as well as a smooth $$P$$ that equals $$k$$ on a box. Smooth examples exist in every Gevrey class $$s>1$$, the smooth functions whose $$n$$-th derivatives are bounded by $$C^{n+1}(n!)^s$$; class $$s=1$$ is the real-analytic functions. The vertices must vanish on an open set of momenta and $$P$$ is linear on an open set, so neither is real-analytic. With cutoffs of Gevrey class $$s>1$$ the vertices decay with distance as a stretched exponential. The band of $$H_r$$ shrinks as $$2\pi/r$$, so at fixed lattice spacing higher-order vertices act on ever smaller sets of momenta. An analytic charge is not rescued by band-limiting, because the low-energy quartets of Theorem 1 lie inside every band.

### 6.2 Gauge invariance

Let $$B\subset T^3$$ be the band of the cubic vertex $$H_3$$, the set of momenta that may enter it; the bands of $$H_r$$ for $$r>3$$ are smaller. The transport term $$\kappa\,\xi\cdot\partial h$$ of (4) maps a gauge parameter and a field with momenta in $$B$$ to a field with momentum in $$B+B$$. We require the nonlinear gauge transformation to keep band-limited fields band-limited:

(G) At order $$\kappa$$, the gauge transformation maps a gauge parameter and a field with momenta in $$B$$ to a field with momentum in $$B$$, so that $$B+B\subset B$$.

(G) is a requirement on the deformation of the gauge transformations, imposed so that the band-limited vertices have a gauge-invariant meaning. It is not derived from gauge invariance, and a deformation that lets band-limited fields leave the band, for example by acting differently on modes outside the band, is not covered by it.

<div class="proposition" markdown="1">
**Proposition 2.** Let $$B\subset T^3$$ be closed, symmetric ($$B=-B$$), contain a neighbourhood of $$0$$, and satisfy $$B+B\subset B$$. Then $$B=T^3$$.
</div>

<div class="proof" markdown="1">
*Proof.* $$B$$ contains $$0$$ and is closed under addition and negation, so it is a subgroup. A subgroup containing a neighbourhood $$\mathcal N$$ of $$0$$ is open, since $$b+\mathcal N\subset B$$ for each $$b\in B$$. An open subgroup is also closed, its complement being a union of cosets, and $$T^3$$ is connected.
</div>

Under (G) the band is therefore the whole zone, where Proposition 1 is unavailable and Umklapp processes are present. Keeping band-limited vertices of the form of Proposition 1 thus requires dropping (G), that is, a deformation of the gauge transformations that does not close on band-limited fields. Whether some deformation exists under which a band-limited cubic vertex is gauge invariant is open.

### 6.3 Lorentz invariance

A band limit is a statement about spatial momenta in the lattice frame, while the energy is unrestricted in a continuous-time theory. It singles out that frame.

<div class="proposition" markdown="1">
**Proposition 3** (no Lorentz-invariant band). Let $$\tilde B\subset\mathbb R^4$$ be a set of momenta $$(\omega,\mathbf k)$$ that contains a neighbourhood of $$0$$ and has bounded projection onto the spatial momenta $$\mathbf k$$. Then $$\tilde B$$ is not invariant under the identity component of the Lorentz group of any Lorentzian metric $$g$$ on $$\mathbb R^4$$.
</div>

<div class="proof" markdown="1">
*Proof.* Choose $$v\in\tilde B$$ with $$g(v,v)>0$$ (spacelike in signature $$-+++$$); one exists because $$\tilde B$$ contains a neighbourhood of $$0$$. If $$\tilde B$$ were invariant it would contain the orbit of $$v$$, which is the whole one-sheeted hyperboloid $$g(x,x)=g(v,v)$$, since the identity component of the Lorentz group acts transitively on it. The orthogonal complement of $$v$$ is a three-dimensional subspace of Lorentzian signature, and it contains a one-parameter family of null directions $$\nu$$. For each of them $$g(v+\tau\nu,v+\tau\nu)=g(v,v)$$, so the line $$v+\tau\nu$$ lies on the hyperboloid. At most one of these directions, up to sign and scale, has vanishing spatial part, so for some $$\nu$$ the spatial part of $$v+\tau\nu$$ is unbounded in $$\tau$$.
</div>

A band is therefore a preferred-frame cutoff of the kind treated by Collins et al. [[Collins et al.]](#ref-collins), and loop corrections need not respect Lorentz invariance. A cutoff on the spatial momentum, as in the bandlimited field theory of Pye, Donnelly and Kempf [[Pye–Donnelly–Kempf]](#ref-pdk), is of this kind. A cutoff on the spectrum of the d'Alembertian, proposed by Kempf, Chatwin-Davies and Martin as a covariant cutoff [[Kempf et al.]](#ref-kcm), does not restrict the spatial momentum to a bounded set and is outside Proposition 3. We compute the loop effects of a band in a model.

### 6.4 One-loop speeds and a photon mass

Exact conservation of the total momentum requires that no sector has Umklapp vertices, so the model band-limits the vertices of matter as well as those of gravity; this is a hypothesis of the model. Every field has the exactly relativistic dispersion $$\omega=\lvert k\rvert$$ inside the band, which requires a non-analytic lattice derivative equal to $$k$$ there, so the only departure from the continuum is a factor $$\chi(\lvert k\rvert/b)$$ on every leg of every vertex. Here $$b$$ is the band edge; $$\chi=1$$ for $$\lvert k\rvert\le y_0b$$ and $$\chi=0$$ for $$\lvert k\rvert\ge b$$, with a Gevrey step in between, $$\chi=u(1-\eta)/[u(1-\eta)+u(\eta)]$$ with $$u(x)=\exp(-x^{-\beta})$$ for $$x>0$$, $$u(x)=0$$ for $$x\le0$$ and $$\eta=(\lvert k\rvert/b-y_0)/(1-y_0)$$. The exponent $$\beta=1$$ gives Gevrey class 2 and $$\beta=\tfrac12$$ class 3. The same cutoff is used for the cubic and the seagull vertices, which satisfies Proposition 1 for $$r=3,4$$ when $$b\le\pi/2$$; a vertex with $$r$$ legs needs $$b\le2\pi/r$$. The matter is scalar electrodynamics: a complex scalar $$\Phi$$ of charge $$e$$, with the infrared regulator $$m_{\rm IR}$$ on the internal lines later taken to $$0$$.

A vertex vanishes if any of its legs lies outside the band, and inside the band the lattice propagators equal the continuum ones. A one-loop diagram with external momenta inside the band is therefore a continuum Euclidean loop in which each internal line carries the weight $$\varrho=\chi^2$$ of its spatial momentum, one factor $$\chi$$ from the vertex at each end, while the loop energy is integrated over the real line without restriction. Integrating out the canonical momenta of a Hamiltonian lattice theory would also generate an $$A_4A_4$$ seagull whose legs need not carry $$\chi$$ in this way; that vertex is independent of the external momentum and affects $$\Pi_{44}(0)$$ only, not the speeds or $$\Pi_{xx}(0)$$ below.

Write the Euclidean inverse propagator of a field as $$p^2+\Sigma(p)$$ and expand $$\Sigma(p)=\Sigma(0)+A_4p_4^2+A_s\lvert\mathbf p\rvert^2+\dots$$. After the continuation $$p_4\to i\omega$$ the low-energy speed is $$c=[(1+A_s)/(1+A_4)]^{1/2}$$, so

$$
\delta c=\tfrac12(A_s-A_4). \tag{10}
$$

For a marginal coupling $$\delta c=(e^2/16\pi^2)\,C$$, with $$C$$ a number that depends on the shape of the profile but not on $$b$$, because $$b$$ is the only scale. For the photon $$\gamma$$ (a transverse component, from the $$\Phi$$ loop) the coefficient $$C_\gamma$$ is independent of the gauge fixing, since no photon propagates in the loop. For $$\Phi$$ (from the photon loop) we use Feynman gauge; because the cut breaks the $$U(1)$$ gauge symmetry, $$C_\Phi$$ depends on this choice, which is then part of the definition of the model. Without the cut, both vanish by Lorentz invariance.

The cut also gives the photon a mass. With $$q^2=q_4^2+\lvert\mathbf q\rvert^2$$ and $$\int\equiv\int d^4q/(2\pi)^4$$, the seagull contributes $$2e^2\delta_{\mu\nu}\int\varrho/q^2$$ to the vacuum polarisation $$\Pi_{\mu\nu}$$, and the bubble contributes $$-e^2\int(2q+p)_\mu(2q+p)_\nu\,\varrho\varrho'/(q^2(q+p)^2)$$, where the prime refers to the line with momentum $$q+p$$.

<div class="proposition" markdown="1">
**Proposition 4** (photon mass). Let $$\varrho$$ be a function of the spatial momentum $$\mathbf q$$ only, with $$0\le\varrho\le1$$, compact support, and $$\varrho>0$$ on a set of positive measure. At one loop,

$$
\Pi_{44}(0)=e^2\!\int\!\frac{d^3q}{(2\pi)^3}\,\frac{\varrho(1-\varrho)}{\lvert\mathbf q\rvert},\qquad
\Pi_{xx}(0)=e^2\!\int\!\frac{d^3q}{(2\pi)^3}\,\frac{\varrho-\varrho^2\cos^2\theta}{\lvert\mathbf q\rvert}, \tag{11}
$$

where $$\theta$$ is the angle between $$\mathbf q$$ and the $$x$$ axis. Both are non-negative and $$\Pi_{xx}(0)>0$$. For a ball band, $$\varrho=\varrho(\lvert\mathbf q\rvert/b)$$, the photon mass squared is

$$
m_\gamma^2\equiv\Pi_{xx}(0)=\frac{e^2b^2}{16\pi^2}\Big[\frac{16}3\int_0^\infty y\,\varrho^2\,dy+8\int_0^\infty y\,(\varrho-\varrho^2)\,dy\Big], \tag{12}
$$

with $$y=\lvert\mathbf q\rvert/b$$.
</div>

<div class="proof" markdown="1">
*Proof.* At $$p=0$$ the lines $$q$$ and $$q+p$$ coincide, so $$\varrho\varrho'=\varrho^2$$ and, with no sum over $$\mu$$, $$\Pi_{\mu\mu}(0)=2e^2\int\varrho/q^2-4e^2\int q_\mu^2\varrho^2/q^4$$. The weight $$\varrho$$ does not depend on $$q_4$$, and with $$s=\lvert\mathbf q\rvert$$ the energy integrals are $$\int\frac{dq_4}{2\pi}(q_4^2+s^2)^{-1}=\frac1{2s}$$, $$\int\frac{dq_4}{2\pi}(q_4^2+s^2)^{-2}=\frac1{4s^3}$$ and $$\int\frac{dq_4}{2\pi}q_4^2(q_4^2+s^2)^{-2}=\frac1{4s}$$. For $$\mu=4$$ this gives $$e^2(\varrho-\varrho^2)/s$$ per unit $$d^3q/(2\pi)^3$$, and for $$\mu=x$$ it gives $$e^2(\varrho-\varrho^2\cos^2\theta)/s$$, which is (11). Since $$0\le\varrho\le1$$, $$\varrho-\varrho^2\ge0$$ and $$\varrho-\varrho^2\cos^2\theta=\varrho(1-\varrho\cos^2\theta)\ge\varrho(1-\varrho)\ge0$$. The second integrand is positive wherever $$\varrho>0$$ except where $$\varrho=1$$ and $$\cos^2\theta=1$$, a set of measure zero. For a ball band, $$d^3q/(2\pi)^3=s^2ds\,d\Omega/(2\pi)^3$$ and the angular average of $$\cos^2\theta$$ is $$\tfrac13$$, so $$\Pi_{xx}(0)=(e^2/2\pi^2)\int s\,(\varrho-\varrho^2/3)\,ds$$. In the units of (12) the prefactor is $$8$$ and $$8(\varrho-\varrho^2/3)=8(\varrho-\varrho^2)+\tfrac{16}3\varrho^2$$.
</div>

Without the cut, $$\Pi_{\mu\nu}(0)$$ vanishes in a gauge-invariant regularisation such as dimensional regularisation, as gauge invariance requires. With the cut, the seagull carries the weight $$\varrho$$ and the bubble carries $$\varrho^2$$, and the mismatch is the photon mass. Since $$\Pi_{44}(0)\ne\Pi_{xx}(0)$$, the mass term itself is not Lorentz invariant.

The coefficients $$C_\gamma$$ and $$C_\Phi$$ need the second derivatives of the bubbles in $$p$$; they were evaluated numerically at $$m_{\rm IR}=0.003\,b$$. In the table, $$C_\gamma$$ and $$C_\Phi$$ are defined by $$\delta c=(e^2/16\pi^2)C$$, and $$m_\gamma^2$$ and $$\Pi_{44}(0)$$ from (11) and (12) are in units of $$e^2b^2/16\pi^2$$.

| profile ($$y_0$$, Gevrey class) | $$C_\gamma$$ | $$C_\Phi$$ | $$m_\gamma^2$$ | $$\Pi_{44}(0)$$ |
|---|---|---|---|---|
| 0.2, 2 | +0.3086 | +0.6049 | 0.8517 | 0.2614 |
| 0.5, 2 | +0.5144 | +1.1194 | 1.4016 | 0.2169 |
| 0.8, 2 | +1.3376 | +3.1772 | 2.1084 | 0.1081 |
| 0.5, 3 | +0.3850 | +0.7959 | 1.3394 | 0.3843 |
| 0.8, 3 | +1.0118 | +2.3629 | 2.0663 | 0.1941 |

For the profile $$(0.5,2)$$ the coefficients change by less than $$0.05\%$$ between $$m_{\rm IR}=0.01\,b$$ and $$0.001\,b$$: $$C_\gamma=0.51424,\ 0.51442,\ 0.51443$$ at $$m_{\rm IR}/b=0.01,\ 0.003,\ 0.001$$, and $$C_\Phi=1.11913,\ 1.11940,\ 1.11942$$. A second evaluation of $$C_\gamma$$ for this profile by finite differences in $$p$$ of the integrated bubble gives $$0.51424$$ at $$m_{\rm IR}=0.01\,b$$, in agreement. The photon-mass column was obtained both from (12) and from the angular form of (11), with identical results to four digits.

All coefficients are positive, $$C_\Phi/C_\gamma$$ lies between $$1.96$$ and $$2.38$$, and the coefficients vary with the shape of the profile by factors of $$4.3$$ ($$C_\gamma$$), $$5.3$$ ($$C_\Phi$$) and $$2.5$$ ($$m_\gamma^2$$); none comes near zero.

With $$\alpha=e^2/4\pi=1/137.036$$, (10) gives a photon speed shift $$\delta c_\gamma=C_\gamma\,\alpha/4\pi=(1.8\text{–}7.8)\times10^{-4}$$. The graviton has no $$O(e^2)$$ shift in this model, so this is also $$c_\gamma-c_{\rm GW}$$ at this order. The arrival times of GW170817 and GRB 170817A bound $$(c_{\rm GW}-c_\gamma)/c$$ between $$-3\times10^{-15}$$ and $$+7\times10^{-16}$$ [[GW170817]](#ref-gw170817). Since $$c_{\rm GW}-c_\gamma=-\delta c_\gamma<0$$, the relevant bound is $$3\times10^{-15}$$, which is $$4\times10^{-12}$$ to $$1.7\times10^{-11}$$ times $$\delta c_\gamma$$. The shift does not depend on $$b$$, so no choice of band edge reduces it. The two species of the same theory also disagree, $$\lvert\delta c_\gamma-\delta c_\Phi\rvert=(1.7\text{–}11)\times10^{-4}$$, with the gauge caveat above. The photon mass is $$m_\gamma=(0.022\text{–}0.035)\,b$$. For a band edge at the Planck mass, $$1.2\times10^{28}$$ eV, this is $$(2.7\text{–}4.3)\times10^{26}$$ eV, some $$10^{44}$$ times the bound $$m_\gamma<10^{-18}$$ eV [[PDG]](#ref-pdg). Chankowski, Lewandowski and Meißner found that a physical momentum cutoff gives the photon a mass [[CLM]](#ref-clm), as here. Each of these effects can be cancelled by a bare Lorentz-violating counterterm, since no symmetry forbids one once gauge invariance is broken; agreement with the bounds then requires the bare parameters to be tuned species by species, the speeds to between one part in $$10^{11}$$ and one part in $$10^{12}$$ and the photon mass term to a much higher accuracy. This is the fine-tuning problem of Collins et al. [[Collins et al.]](#ref-collins), realised here by the vertices of Proposition 1. The calculation is for a ball-shaped band, one toy matter sector, and Feynman gauge for $$C_\Phi$$. Graviton loops add contributions of order $$\kappa^2b^2/16\pi^2$$, which depend on $$b$$ and could cancel the photon shift only for one tuned value of $$b$$; cubic bands, fermions and graviton loops are not computed.

## 7. Integer charges in Villain form

Sections 2–6 concern the interacting theory. The free theory of section 1 raises a separate question that does not enter the earlier argument: whether its gauge group can be made compact, so that the momentum-constraint charges become integers. Compact $$U(1)$$ makes electric charges integers; compactifying $$h$$ would likewise make these charges integers. The standard device is the modified Villain formulation [[GLSS 2021]](#ref-glss1). The components of the lattice field $$h$$, labelled by an index $$a$$ that runs over the tensor components at all sites, have conjugate momenta $$\varpi_a$$. A translation-invariant integer matrix $$L$$ of finite range, with rows labelled by $$b$$, forms the curvature-like combinations $$Lh$$, and each row is paired with an integer field $$N_b$$ whose conjugate angle is $$\theta_b$$. The potential depends on $$h$$ only through $$X=Lh-2\pi N$$. The integer shifts $$h\to h+2\pi n$$, $$N\to N+Ln$$, for every integer-valued $$n$$ on the components of $$h$$, are gauged by imposing

$$
\mathcal W[n]\equiv\exp\big(2\pi i\langle\varpi,n\rangle+i\langle\theta,Ln\rangle\big)=1 . \tag{13}
$$

This makes every component of $$h$$ an angle. The symmetrised gradient $$S$$ of section 2 is an integer matrix from link variables $$\xi$$ to the components of $$h$$, and the generators of the linearised gauge transformations $$h\to h+S\xi$$ are the momentum-constraint charges $$\mathsf q=S^{\rm T}\varpi$$, one per link.

<div class="theorem" markdown="1">
**Theorem 2** (integer momentum-constraint charges). Consider the modified Villain formulation above, with every component of $$h$$ compact, $$L$$ of finite range, and a quadratic potential $$\tfrac12X^{\rm T}VX$$ whose symbol is bounded. Then:

(a) the charges $$\mathsf q$$ have integer spectrum on the states with $$\mathcal W[n]=1$$ for all $$n$$ if and only if $$LS=0$$;

(b) if $$LS=0$$, the symbol of $$L$$ is $$O(k^2)$$, so the quadratic potential is $$O(k^4)$$; if moreover the kinetic term has a bounded symbol, every mode has $$\omega=O(k^2)$$.

Hence integer charges and a graviton with $$\omega\propto\lvert k\rvert$$ are incompatible in this formulation.
</div>

<div class="proof" markdown="1">
*Proof.* (a) The operators $$\varpi$$ and $$\theta$$ commute, and $$\mathcal W[n]=\exp(2\pi i\langle\zeta,n\rangle)$$ with $$\zeta\equiv\varpi+L^{\rm T}\theta/2\pi$$. So $$\mathcal W[n]=1$$ for all integer $$n$$ means that $$\zeta$$ is integer-valued, while $$\theta$$ takes all values in its torus. Then $$\mathsf q=S^{\rm T}\zeta-(LS)^{\rm T}\theta/2\pi$$. The first term is an integer. The second sweeps a continuum as $$\theta$$ varies unless $$LS=0$$.

(b) $$LS=0$$ is an identity between symbols, and its expansion in $$k$$ kills the low orders of $$L$$ by an algebraic lemma. Let $$\ell(k)$$ and $$S(k)$$ be the symbols of $$L$$ and $$S$$, and expand $$\ell=\ell_0+\ell_1(k)+O(k^2)$$, $$S(k)\xi=i(k_i\xi_j+k_j\xi_i)+O(k^3)$$. The columns of $$\ell$$ are indexed by the symmetric pair $$(ij)$$ and each row $$b$$ is treated separately, with $$b$$ suppressed. At first order in $$k$$, $$\ell_0S_1(k)=0$$ requires $$\ell_0$$ to annihilate every symmetrised product $$k_i\xi_j+k_j\xi_i$$; these span all symmetric tensors, so $$\ell_0=0$$. At second order, write the row of $$\ell_1$$ as $$c_{ij,l}k_l$$ with $$c_{ij,l}=c_{ji,l}$$. The condition $$c_{ij,l}k_l(k_i\xi_j+k_j\xi_i)=0$$ for all $$k,\xi$$ says that $$c_{ij,l}+c_{lj,i}=0$$, that is, $$c$$ is antisymmetric under $$i\leftrightarrow l$$. A tensor symmetric in its first two indices and antisymmetric in its first and third vanishes: $$c_{ij,l}=-c_{lj,i}=-c_{jl,i}=c_{il,j}=c_{li,j}=-c_{ji,l}=-c_{ij,l}$$. So $$\ell=O(k^2)$$, the quadratic potential $$\ell(k)^\dagger V\ell(k)$$ is $$O(k^4)$$ for a bounded matrix $$V$$, and $$\omega^2$$, an eigenvalue of the product of the kinetic and potential symbols, is $$O(k^4)$$ when the kinetic symbol is bounded.
</div>

The theorem is narrow. Its hypothesis is that the gauge lattice is all of $$\mathbb Z$$ on every component of $$h$$. If instead only the gauge directions are compactified, by imposing $$\exp(2\pi i\langle\varpi,S\iota\rangle)=1$$ for integer-valued $$\iota$$ on links, then the charges $$S^{\rm T}\varpi$$ are integers, the potential of section 1 is invariant under $$h\to h+S\iota$$ because the linearised curvature of $$S\iota$$ vanishes identically, and the dispersion is (6). Integer charges and $$\omega\propto\lvert k\rvert$$ then coexist at the free level. That compact lattice tensor gauge theories have $$\omega\propto k^2$$ was known in examples [[Xu]](#ref-xu) [[Xu–Hořava]](#ref-xh) [[GLSS 2022]](#ref-glss2); Theorem 2 gives the reason for the whole Villain class. The "only if" in (a) also has limited reach: when $$LS\ne0$$, a quadratic potential $$\tfrac12X^{\rm T}VX$$ changes under $$h\to h+S\xi$$ with continuous $$\xi$$ by terms such as $$-2\pi N^{\rm T}VLS\xi$$, so in that branch the theory generally has no continuous linear gauge symmetry to begin with. The same tensor lemma, applied to local integer charges of a general compact potential, gives $$z\ge2$$ with the vector constraint and $$z\ge3$$ when the scalar constraint also holds, in the page on [gravitons from qubits, rotors and composite tetrads](/quantum-gravity/emergent-spin-2/) (Propositions 1 and 2).

## 8. What the theorems do not exclude

Each setting below drops one hypothesis of the results above.

1. If $$P$$ is not analytic, Theorem 1 does not apply, except in a quasi-analytic class by the remark in section 5. Band-limited vertices with a charge that is linear on the band conserve it exactly (Proposition 1), at the costs of section 6: a preferred frame, the closure requirement (G), and the loop effects of section 6.4 in the model computed there. Whether a deformation of the gauge transformations exists that closes on band-limited fields is open.
2. If the theory is not a continuous-time Hamiltonian on a fixed Bravais lattice, Theorem 1 does not apply. Spacetime lattices, on which energy is also periodic, dynamical lattices, and non-local actions are outside its scope, and perfect actions are non-local in this sense.
3. If the charge acts on the helicity index by a $$2\times2$$ matrix symbol, (9) becomes a matrix condition, which is not analysed here.
4. If (H2) fails, as it does for an exactly linear dispersion where collinear three-wave processes are resonant, the cubic step of section 3 needs separate treatment, and the on-shell three-graviton amplitude would have to vanish on the collinear set.
5. If the low-energy $$2\to2$$ interaction vanishes on the resonant set, (H5) fails and Theorem 1 gives no information. If it is non-zero but differs from that of general relativity, the theorem applies as long as it is non-zero.

The obstruction in Theorem 1 first appears at order $$\kappa^2$$. Whether an exponentially local $$Q_3$$ exists at order $$\kappa$$ is open. Whether a Lorentz-invariant structure other than a momentum band, for example a random rather than a regular lattice, avoids both the identity-theorem obstruction and the preferred frame of a band is not addressed here; see [point sets with no preferred frame](/quantum-gravity/quantum-sprinklings/) for point sets of that kind.

## References

1. <a id="ref-dfs"></a>Dittrich, Freidel, Speziale, “Linearized dynamics from the 4-simplex Regge action,” Phys. Rev. D 76, 104020 (2007), [arXiv:0707.4513](https://arxiv.org/abs/0707.4513).
2. <a id="ref-hoehn"></a>Höhn, “Canonical linearized Regge calculus: counting lattice gravitons with Pachner moves,” Phys. Rev. D 91, 124034 (2015), [arXiv:1411.5672](https://arxiv.org/abs/1411.5672).
3. <a id="ref-bd1"></a>Bahr, Dittrich, “(Broken) gauge symmetries and constraints in Regge calculus,” Class. Quantum Grav. 26, 225011 (2009), [arXiv:0905.1670](https://arxiv.org/abs/0905.1670).
4. <a id="ref-bd2"></a>Bahr, Dittrich, “Breaking and restoring of diffeomorphism symmetry in discrete gravity,” [arXiv:0909.5688](https://arxiv.org/abs/0909.5688).
5. <a id="ref-dittrich"></a>Dittrich, “Diffeomorphism symmetry in quantum gravity models,” [arXiv:0810.3594](https://arxiv.org/abs/0810.3594).
6. <a id="ref-peierls"></a>Peierls, “Zur kinetischen Theorie der Wärmeleitung in Kristallen,” Ann. Phys. (Leipzig) 395, 1055 (1929).
7. <a id="ref-lkm"></a>Lewis, Kempf, Menicucci, “Quantum lattice models that preserve continuous translation symmetry,” Phys. Rev. D 108, 096024 (2023), [arXiv:2303.07649](https://arxiv.org/abs/2303.07649).
8. <a id="ref-spohn"></a>Spohn, “Collisional invariants for the phonon Boltzmann equation,” [arXiv:math-ph/0605069](https://arxiv.org/abs/math-ph/0605069).
9. <a id="ref-collins"></a>Collins, Perez, Sudarsky, Urrutia, Vucetich, “Lorentz invariance and quantum gravity: an additional fine-tuning problem?,” Phys. Rev. Lett. 93, 191301 (2004), [arXiv:gr-qc/0403053](https://arxiv.org/abs/gr-qc/0403053).
10. <a id="ref-xu"></a>Xu, “Novel algebraic boson liquid phase with soft graviton excitations,” [arXiv:cond-mat/0602443](https://arxiv.org/abs/cond-mat/0602443).
11. <a id="ref-xh"></a>Xu, Hořava, “Emergent gravity at a Lifshitz point from a Bose liquid on the lattice,” Phys. Rev. D 81, 104033 (2010), [arXiv:1003.0009](https://arxiv.org/abs/1003.0009).
12. <a id="ref-glss1"></a>Gorantla, Lam, Seiberg, Shao, “A modified Villain formulation of fractons and other exotic theories,” J. Math. Phys. 62, 102301 (2021), [arXiv:2103.01257](https://arxiv.org/abs/2103.01257).
13. <a id="ref-glss2"></a>Gorantla, Lam, Seiberg, Shao, “Global dipole symmetry, compact Lifshitz theory, tensor gauge theory, and fractons,” [arXiv:2201.10589](https://arxiv.org/abs/2201.10589).
14. <a id="ref-pdk"></a>Pye, Donnelly, Kempf, “Locality and entanglement in bandlimited quantum field theory,” Phys. Rev. D 92, 105022 (2015), [arXiv:1508.05953](https://arxiv.org/abs/1508.05953).
15. <a id="ref-kcm"></a>Kempf, Chatwin-Davies, Martin, “A fully covariant information-theoretic ultraviolet cutoff for scalar fields in expanding Friedmann Robertson Walker spacetimes,” J. Math. Phys. 54, 022301 (2013), [arXiv:1210.0750](https://arxiv.org/abs/1210.0750).
16. <a id="ref-clm"></a>Chankowski, Lewandowski, Meißner, “Ultraviolet cutoffs and the photon mass,” [arXiv:1608.01214](https://arxiv.org/abs/1608.01214).
17. <a id="ref-gw170817"></a>LIGO Scientific Collaboration, Virgo Collaboration, Fermi Gamma-ray Burst Monitor, INTEGRAL, “Gravitational waves and gamma-rays from a binary neutron star merger: GW170817 and GRB 170817A,” Astrophys. J. Lett. 848, L13 (2017), [arXiv:1710.05834](https://arxiv.org/abs/1710.05834).
18. <a id="ref-pdg"></a>Navas et al. (Particle Data Group), “Review of particle physics,” Phys. Rev. D 110, 030001 (2024).
