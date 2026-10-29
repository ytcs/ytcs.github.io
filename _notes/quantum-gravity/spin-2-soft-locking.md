---
title: "Exact universal graviton coupling and periodic dispersion"
date: 2026-10-29
category: gravity
order: 2
description: Under stated assumptions on the soft expansion of the amplitude, a graviton whose coupling to energy and momentum is exactly universal must have an exactly conical dispersion near zero momentum, and no band of a finite-dimensional real-analytic periodic symbol is an exact cone.
status: "Theorem 2 (no exact cone) is proved for finite-dimensional real-analytic periodic symbols with no further assumption. Theorem 1 (locking of the dispersion) is proved under the hypotheses (B) and (S1)–(S4) on the soft expansion, which are assumed here and not derived from a lattice action. Proposition 4 concerns an algebraic necessary condition and is accompanied by numerical checks. Proposition 5 is proved in a model in which the one-particle space is a spin-s representation. Two steps are given as sketches: the Zariski-density step in Lemma A and the Weinberg–Witten argument of section 6."
excludes: "Lattice Hamiltonian and Floquet (cellular-automaton) gravitons with a finite-dimensional, real-analytic, periodic Bloch symbol that satisfy (B) and (S1)–(S4) of section 2: a relativistic low-energy band, simple soft poles with a leg-dependent coupling that is a multiple of the identity on helicity, exact soft gauge invariance, non-zero coupling to energy and momentum, and non-vanishing hard scattering. Not excluded: soft factors that are universal only up to a finite order in the lattice spacing, and models in which any of these hypotheses fails (section 7)."
---

On a lattice the graviton dispersion is periodic in the wave vector, so it cannot be $$\omega=\lvert k\rvert$$ everywhere. The question is whether the graviton can still couple to energy and momentum as Weinberg's soft theorem demands, with one coupling for every species and an exactly gauge-invariant soft factor. Under the hypotheses of section 2 on the soft expansion of the amplitude, the answer is no, for lattice Hamiltonians and for Floquet cellular automata whose Bloch symbol is a finite matrix that is real-analytic in the wave vector. Exact universality forces the graviton band to be an exact cone, $$\Omega(k)=\sqrt{Q(k)}$$ with $$Q$$ a positive quadratic form, on a neighbourhood of $$k=0$$ (Theorem 1), and no band of such a symbol is an exact cone on any open set (Theorem 2). The hypotheses are assumptions about the soft expansion; they are not derived from a lattice action.

A necessary condition for universality can be met to finite order in the lattice spacing: if $$\Omega^2-\lvert k\rvert^2$$ begins at degree $$d\ge3$$, the condition fails at relative order $$(Ea)^{d-2}$$ in the energy $$E$$ and the lattice spacing $$a$$, and no admissible choice of coupling does better (Proposition 4). Whether a lattice amplitude exists whose soft factor is gauge invariant to that order is not decided here. In a model where the one-particle space is a spin-$$s$$ representation, a local analytic energy density has off-forward matrix elements of order one between neighbouring momenta, so the covariance hypothesis of the Weinberg–Witten theorem is not met (Proposition 5).

Weinberg derived the universality of the soft graviton coupling, and with it the equality of gravitational and inertial mass, from the pole structure of amplitudes with a soft graviton in a Lorentz-covariant S-matrix [[Weinberg 1964]](#ref-w64) [[Weinberg 1965]](#ref-w65). Weinberg and Witten showed that a massless particle of helicity above one has no Lorentz-covariant conserved energy-momentum tensor [[Weinberg–Witten]](#ref-ww). Porrati extended that argument to theories without a gauge-invariant stress tensor and concluded that massless particles interacting with gravity have spin at most two [[Porrati]](#ref-porrati), and Benincasa and Cachazo excluded spin above two from consistency conditions on four-particle amplitudes [[Benincasa–Cachazo]](#ref-bc). All of these assume exact Poincaré covariance. For supersymmetry, no lattice field theory in infinite volume with finitely many flavours has translation invariance, locality and the Leibniz rule together [[Kato–Sakamoto–So]](#ref-kss); the obstruction below has a similar shape, with lattice translations and locality on one side and an exact soft Ward identity on the other. The page [Lattice gravitons, exact diffeomorphisms and lattice momentum](/quantum-gravity/lattice-diffeomorphisms/) shows that a conserved lattice momentum with an analytic symbol is incompatible with low-energy graviton scattering. That is the charge side of the obstruction; here the dispersion is locked through the soft theorem, and two lemmas of that page are restated in section 3. This page adds the transposition of Weinberg's argument to a non-covariant band with explicit hypotheses, the locking theorem, the theorem on periodic symbols, and the order of the defect. The lattice spacing is $$a$$ throughout, set to 1 where a statement concerns a symbol periodic in $$2\pi$$, and the low-energy speed of light is 1.

## 1. Weinberg's argument on the light cone

Consider a process with hard particles $$i=1,\dots,n$$ of momenta $$k_i$$ and energies $$\Omega_i$$, incoming ($$\sigma_i=+1$$) or outgoing ($$\sigma_i=-1$$), and add one soft graviton of momentum $$K=(K_0,\mathbf K)$$ and polarisation $$\varepsilon_{\mu\nu}$$. The amplitude has poles in $$K$$ from the propagators of the hard legs next to the emission. In the continuum the leading term is

$$
\mathcal M_{n+1}\simeq\kappa\sum_i\sigma_i\frac{p_i^\mu p_i^\nu\varepsilon_{\mu\nu}}{p_i\cdot K}\,\mathcal M_n ,\qquad p_i=(\Omega_i,k_i). \tag{1}
$$

Under a gauge transformation $$\varepsilon_{\mu\nu}\to\varepsilon_{\mu\nu}+K_\mu\xi_\nu+\xi_\mu K_\nu$$ with $$K\cdot\xi=0$$, the expression (1) changes by $$2\kappa\,\xi\cdot\sum_i\sigma_ip_i\,\mathcal M_n$$, and this vanishes because energy-momentum is conserved. The coupling $$\kappa$$ is the same for every species because it multiplies the conserved four-vector $$\sum_i\sigma_ip_i$$ and nothing else. If two species had couplings $$\kappa_A\ne\kappa_B$$, the variation would be proportional to $$\kappa_A\sum_Ap_i+\kappa_B\sum_Bp_i$$, which does not vanish in general. This is the universality of the gravitational coupling.

For a photon the same reasoning, with $$\varepsilon_\mu\to\varepsilon_\mu+K_\mu$$, gives $$\sum_i\sigma_ie_i=0$$ for the charges $$e_i$$. The soft charge is then an internal scalar, conserved on any lattice with an exact $$U(1)$$ whatever the dispersion. For the graviton the conserved soft charge is the four-vector $$\sum_i\sigma_ip_i$$, which is tied to the kinematics of the hard legs. On a lattice the question is therefore whether the dispersion allows the charge that gauge invariance requires to be the lattice energy-momentum. Sections 2 to 4 show that it does only for an exact cone. For spin above two the corresponding identity would require conserved tensor charges of higher rank, which Lorentz-covariant scattering does not have.

## 2. The soft expansion off the cone

Let the graviton band be a function $$\Omega(k)$$ on a punctured neighbourhood of the origin. In a Hamiltonian lattice theory $$\Omega$$ is an eigenvalue of the Bloch Hamiltonian $$H(k)$$. In a Floquet theory with time step 1, $$e^{-i\Omega(k)}$$ is an eigenvalue of the unitary Bloch matrix $$W(k)$$, and near $$k=0$$ the branch with $$\Omega\to0$$ is used, so that energies are conserved as real numbers with no wrap-around. A hard leg with momentum $$k_i$$ that emits a soft graviton has the denominator $$\Omega(k_i+\mathbf K)-\Omega(k_i)-K_0\simeq-\mathsf s_i\cdot K$$, where

$$
\mathsf s_i\cdot K=K_0-v_i\cdot\mathbf K,\qquad v_i=\nabla\Omega(k_i). \tag{2}
$$

The covector $$\mathsf s_i=(1,-v_i)$$ is the normal to the on-shell variety $$\{K_0=\Omega(\mathbf K)\}$$ at $$k_i$$. On the cone it is proportional to $$p_i$$, and (2) reduces to $$-p_i\cdot K/\Omega_i$$.

The lattice theories considered here have no covariant S-matrix from which the soft expansion could be derived, so we take its form as hypothesis. The first hypothesis concerns the band.

(B) After a linear change of the spatial axes, $$\Omega(k)=\lvert k\rvert\,\tilde\omega(\lvert k\rvert,\hat k)$$ for $$0<\lvert k\rvert<\rho_0$$, where $$\tilde\omega$$ is real-analytic in a neighbourhood of $$[0,\rho_0)\times S^2$$ and $$\tilde\omega(0,\hat k)=1$$. Then $$\Omega=\sum_{m\ge1}\Omega_m$$ near $$k=0$$, with $$\Omega_m$$ real-analytic on $$\mathbb R^3\setminus\{0\}$$ and positively homogeneous of degree $$m$$, and $$\Omega_1=\lvert k\rvert$$. This says that the dispersion is relativistic at low energy and has a Taylor-type expansion in the scale. It holds, for example, when $$\Omega^2$$ is real-analytic at $$0$$ with $$\Omega^2=\lvert k\rvert^2+O(k^3)$$.

Write $$\eta=\mathrm{diag}(-1,1,1,1)$$ for the metric whose null cone $$K_0=\lvert\mathbf K\rvert$$ is the leading cone of the band, and let $$\mathsf s_i^\sharp=-(1,v_i)$$ be the vector with $$\eta(\mathsf s_i^\sharp,x)=\mathsf s_i\cdot x$$ for all $$x$$; then $$\eta(\mathsf s_i^\sharp,\mathsf s_i^\sharp)=\lvert v_i\rvert^2-1$$. The remaining hypotheses concern the soft factor.

(S1) For a hard $$2\to2$$ graviton process at small momenta with $$\mathcal M_4\ne0$$, the amplitude with an extra soft graviton of polarisation $$\varepsilon_{\mu\nu}$$ and momentum $$K$$ on the leading cone is, as $$K\to0$$,

$$
\mathcal M_5=\Big[\sum_i\sigma_i\frac{\varepsilon_{\mu\nu}J_i^{\mu\nu}}{\mathsf s_i\cdot K}+O(K^0)\Big]\mathcal M_4 ,
$$

where $$J_i^{\mu\nu}=J^{\mu\nu}(k_i)$$ is a symmetric tensor that depends on the leg $$i$$ only, acts as a multiple of the identity on the helicity index, and has the form $$\hat J(\lvert k\rvert,\hat k)$$ with $$\hat J$$ real-analytic near $$[0,\rho_0)\times S^2$$.

(S2) The pole term of (S1) is invariant under $$\varepsilon_{\mu\nu}\to\varepsilon_{\mu\nu}+K_\mu\xi_\nu+\xi_\mu K_\nu$$ for every $$K$$ on the leading cone and every $$\xi$$ with $$\eta(K,\xi)=0$$ (indices lowered with $$\eta$$). The variation of the pole term is of order $$K^0$$ and that of the next term is of order $$K$$, so the invariance of the amplitude implies that of the pole term separately.

The next hypothesis needs the set of low-energy collisions. Let $$\Lambda=\{q=(q_1,\dots,q_4)\in(\mathbb R^3)^4:\ q_1+q_2=q_3+q_4\}$$ be the space of momentum-conserving quartets, with $$\sigma=(+,+,-,-)$$, and for $$\rho>0$$ let

$$
\mathcal M_\rho=\Big\{q\in\Lambda:\ \sum_i\sigma_i\Omega(\rho q_i)=0\Big\}
$$

be the rescaled energy shell. Its limit as $$\rho\to0$$ is the massless resonant set $$\mathcal M_0=\{q\in\Lambda:\ \lvert q_1\rvert+\lvert q_2\rvert=\lvert q_3\rvert+\lvert q_4\rvert\}$$, and $$\mathcal M_0^\ast$$ is its part on which $$\lvert q_1\rvert+\lvert q_2\rvert>\lvert q_1+q_2\rvert$$.

(S3) There are an open set $$U\subset\Lambda$$ meeting $$\mathcal M_0^\ast$$ and a number $$\rho_1>0$$ such that for $$0<\rho<\rho_1$$ the quartets $$\rho q$$ with $$q\in\mathcal M_\rho\cap U$$ have $$\mathcal M_4(\rho q)\ne0$$ and pairwise distinct velocities $$v_i$$.

(S4) The tensor $$J$$ is not identically a multiple of $$\eta^{\mu\nu}$$. (By Lemma A this means that the coefficient $$\beta$$ of $$\mathsf s^\sharp\mathsf s^\sharp$$ in $$J$$ is not identically zero: the graviton couples to energy and momentum.)

In general relativity $$\Omega=\lvert k\rvert$$ and $$J^{\mu\nu}=-\kappa\Omega\,\mathsf s^{\sharp\mu}\mathsf s^{\sharp\nu}$$, with $$\mathsf s^\sharp=-p/\Omega$$ null, which is (1) rewritten with $$p\cdot K=-\Omega\,\mathsf s\cdot K$$; so $$\beta=-\kappa\Omega$$ there. The hypotheses hold in that case. We do not assert that they hold for a lattice graviton; Theorem 1 states what they imply.

## 3. Exact universality locks the dispersion

Let $$\hat J_i$$ be the operator $$(\hat J_i\xi)^\mu=J_i^{\mu\nu}\eta_{\nu\rho}\xi^\rho$$, which is self-adjoint for $$\eta$$ because $$J_i$$ is symmetric. Lemma A says what (S1) and (S2) leave of $$J_i$$, and what condition on the hard momenta results.

<div class="lemma" markdown="1">
**Lemma A** (the soft vertex). Assume (S1) and (S2), and that for the process considered the velocities $$v_i$$ are pairwise distinct with $$\lvert v_i\rvert\ne1$$. Then $$J_i^{\mu\nu}=\alpha_i\eta^{\mu\nu}+\beta_i\,\mathsf s_i^{\sharp\mu}\mathsf s_i^{\sharp\nu}$$ for scalars $$\alpha_i,\beta_i$$ that depend on $$k_i$$ only, and

$$
\sum_i\sigma_i\beta_i\,\mathsf s_i^\sharp=0,\qquad\text{that is}\qquad\sum_i\sigma_i\,\beta(k_i)\,\big(1,\nabla\Omega(k_i)\big)=0. \tag{3}
$$

Moreover, on a small punctured ball $$0<\lvert k\rvert<\rho_0'$$, $$\beta$$ and $$\alpha$$ extend to functions of $$(\lvert k\rvert,\hat k)$$ that are real-analytic near $$\lvert k\rvert=0$$, and the decomposition of $$J$$ holds at every $$k$$ there, including where $$\lvert v\rvert=1$$.
</div>

<div class="proof" markdown="1">
*Proof.* The leg $$i$$ is the only term of the gauge variation with a pole at $$\mathsf s_i\cdot K=0$$, so its numerator must vanish there. This fixes $$J_i$$ up to two scalars, and what remains of the variation is (3).

With the operator $$\hat J_i$$ the variation in (S2) is

$$
\sum_i\sigma_i\frac{\eta(K,\hat J_i\xi)}{\mathsf s_i\cdot K}=0\qquad\text{for real null }K,\ \eta(K,\xi)=0. \tag{4}
$$

Multiplying by $$\prod_j(\mathsf s_j\cdot K)$$ turns (4) into a polynomial identity in $$(K,\xi)$$ that holds on the real points of the complex variety $$\mathcal V=\{(K,\xi)\in\mathbb C^4\times\mathbb C^4:\ \eta(K,K)=0,\ \eta(K,\xi)=0\}$$. We need it on the complex points, because the real null vectors in the hyperplane $$\mathsf s_i\cdot K=0$$ may be absent when $$\lvert v_i\rvert<1$$. The passage from real to complex points is the standard fact that a polynomial vanishing on a real-analytic submanifold of an irreducible complex variety, of real dimension equal to the complex dimension of the variety, vanishes on the whole variety. Here $$\mathcal V$$ is irreducible (a rank-3 bundle over the irreducible complex null cone minus the origin; the part with $$K=0$$ has dimension 4, less than 6, and is not a component), and its real points $$K\ne0$$ null, $$\xi$$ real with $$\eta(K,\xi)=0$$ form a manifold of real dimension 6, the complex dimension of $$V$$. We state this step as a sketch of the standard argument and do not reprove it.

Fix $$i$$. Because $$\lvert v_i\rvert\ne1$$, $$\mathsf s_i^\sharp$$ is not null, and the hyperplane $$\mathsf s_i\cdot K=0$$ is a three-dimensional space on which $$\eta$$ is non-degenerate. Its complex null vectors form a quadric cone $$C_i$$ that spans it, and $$C_i\setminus\{0\}$$ is connected. Because the velocities are distinct, the hyperplanes $$\mathsf s_j\cdot K=0$$ for $$j\ne i$$ do not contain $$C_i$$, so at a generic point of $$C_i$$ only the term $$i$$ of the multiplied identity survives. It gives $$\eta(K,\hat J_i\xi)=0$$ for $$K\in C_i$$ and all $$\xi$$ orthogonal to $$K$$. By self-adjointness $$\eta(\hat J_iK,\xi)=0$$ for all $$\xi\perp K$$, so $$\hat J_iK$$ is a multiple of $$K$$: $$\hat J_iK=\lambda(K)K$$ on $$C_i$$. The eigenvalue $$\lambda(K)$$ is continuous on the connected set $$C_i\setminus\{0\}$$ and takes values in the finite spectrum of $$\hat J_i$$, so it is a constant $$\lambda_i$$, and $$\hat J_i=\lambda_i$$ on the span of $$C_i$$, the hyperplane $$(\mathsf s_i^\sharp)^\perp$$. A self-adjoint operator that is $$\lambda_i$$ on $$(\mathsf s_i^\sharp)^\perp$$ maps the $$\eta$$-orthogonal complement of that hyperplane, the line through the non-null vector $$\mathsf s_i^\sharp$$, into itself: $$\hat J_i\mathsf s_i^\sharp=\mu_i\mathsf s_i^\sharp$$. Hence $$\hat J_i=\lambda_i+(\mu_i-\lambda_i)\,\mathsf s_i^\sharp\,\eta(\mathsf s_i^\sharp,\cdot)/\eta(\mathsf s_i^\sharp,\mathsf s_i^\sharp)$$, which is the stated form with $$\alpha_i=\lambda_i$$ and $$\beta_i=(\mu_i-\lambda_i)/\eta(\mathsf s_i^\sharp,\mathsf s_i^\sharp)$$.

Insert this form into (4). The $$\alpha_i$$ term gives $$\alpha_i\eta(K,\xi)=0$$. The $$\beta_i$$ term gives $$\beta_i\,\eta(K,\mathsf s_i^\sharp)\,\eta(\mathsf s_i^\sharp,\xi)/(\mathsf s_i\cdot K)=\beta_i\,\eta(\mathsf s_i^\sharp,\xi)$$, because $$\eta(K,\mathsf s_i^\sharp)=\mathsf s_i\cdot K$$. So $$\eta(y,\xi)=0$$ for $$y=\sum_i\sigma_i\beta_i\mathsf s_i^\sharp$$, every real null $$K$$ and every $$\xi\perp K$$. Then $$y$$ is parallel to every real null vector, hence $$y=0$$. Since $$\mathsf s_i^\sharp=-(1,v_i)$$ this is (3).

For the last statement, take a covector $$\mathsf n=(1,\hat k)$$, which is null for the inverse metric, and contract the decomposition with $$\mathsf n_\mu\mathsf n_\nu$$: $$\beta=J^{\mu\nu}\mathsf n_\mu\mathsf n_\nu/(\mathsf s^\sharp\!\cdot\mathsf n)^2$$ with $$\mathsf s^\sharp\!\cdot\mathsf n=-(1+v\cdot\hat k)$$. For small $$\lvert k\rvert$$, $$v\to\hat k$$ and the denominator tends to $$-2$$, so this expression is real-analytic in $$(\lvert k\rvert,\hat k)$$ near $$\lvert k\rvert=0$$, equals $$\beta$$ where $$\lvert v\rvert\ne1$$, and defines $$\beta$$ everywhere. The coefficient $$\alpha=(J^{\mu\nu}\eta_{\mu\nu}-\beta\,\eta(\mathsf s^\sharp,\mathsf s^\sharp))/4$$ is analytic in the same way. Both sides of $$J=\alpha\eta+\beta\mathsf s^\sharp\mathsf s^\sharp$$ are continuous, so the decomposition holds also where $$\lvert v\rvert=1$$. In particular $$\beta=\sum_{m\ge0}\beta_m$$ with $$\beta_m$$ real-analytic on $$\mathbb R^3\setminus\{0\}$$ and homogeneous of degree $$m$$.
</div>

For the exact cone $$\Omega=\lvert k\rvert$$, (3) reads $$\sum_i\sigma_i\beta(k_i)(1,\hat k_i)=0$$, which is energy-momentum conservation when $$\beta\propto\lvert k\rvert$$; this is Weinberg's case. On a general band, (3) says that the vector $$\beta\,(1,\nabla\Omega)$$ is conserved in every hard process, and it is not clear in advance that any $$\beta$$ achieves this.

The proof of Theorem 1 expands (3) in the scale of the hard momenta, and at each order it needs to know which functions are conserved in every massless collision. The answer is in two lemmas of [Lattice gravitons, exact diffeomorphisms and lattice momentum](/quantum-gravity/lattice-diffeomorphisms/), restated here with $$\Delta g(q)=g(q_1)+g(q_2)-g(q_3)-g(q_4)$$.

<div class="lemma" markdown="1">
**Lemma B** (massless collisions). Parts (i) and (ii) are Lemmas 3 and 4 of the page cited above; part (iii) follows from (i) and (B) and is proved below.

(i) $$\mathcal M_0^\ast$$ is a connected real-analytic submanifold of $$\Lambda$$ of dimension 8. On it all $$q_i\ne0$$, and the gradient along $$\Lambda$$ of $$F_0(q)=\lvert q_1\rvert+\lvert q_2\rvert-\lvert q_3\rvert-\lvert q_4\rvert$$ does not vanish.

(ii) Let $$g:\mathbb R^3\setminus\{0\}\to\mathbb R^{\ell}$$ be real-analytic and positively homogeneous of a real degree $$m$$. If $$\Delta g$$ vanishes on a non-empty relatively open subset of $$\mathcal M_0^\ast$$, then $$g\equiv0$$ unless $$m\in\{0,1\}$$. In degree 1, the linear functions and $$\lvert q\rvert$$ satisfy $$\Delta g=0$$ on all of $$\mathcal M_0$$; in degree 0, so do the constants.

(iii) Assume (B). Every $$q^\ast\in\mathcal M_0^\ast$$ is a limit of points $$q_\rho\in\mathcal M_\rho$$ with $$q_\rho-q^\ast=O(\rho)$$.
</div>

Part (ii) rests on the quartet $$(3z,z;2z,2z)$$, which lies in the closure of $$\mathcal M_0^\ast$$ and gives, for homogeneous $$g$$ of degree $$m$$, the factor $$3^m+1-2^{m+1}$$ that is convex in $$m$$ and vanishes only at $$m=0,1$$. For part (iii), set $$F_\rho(q)=\rho^{-1}\sum_i\sigma_i\Omega(\rho q_i)$$, so that $$\mathcal M_\rho=\{F_\rho=0\}$$. By (B), $$F_\rho\to F_0$$ in $$C^1$$ with an error $$O(\rho)$$ near points where all $$q_i\ne0$$, and by (i) the gradient of $$F_0$$ does not vanish there, so the implicit function theorem gives (iii).

<div class="theorem" markdown="1">
**Theorem 1** (locking of the dispersion). Assume (B) and (S1)–(S4). Then $$\Omega(k)=\lvert k\rvert$$ for $$0<\lvert k\rvert<\rho_0$$, in the coordinates of (B): the band is an exact cone.
</div>

<div class="proof" markdown="1">
*Proof.* We expand (3) in the scale $$\rho$$ of the hard momenta. At each order the conserved vector is a collision invariant of massless kinematics, and by Lemma B(ii) the only homogeneous invariants are the energy-momentum itself and constants. This forces $$\beta=\gamma\Omega$$ for a constant $$\gamma$$, and then removes every correction to the cone. Step 2 contains the geometric input, and Steps 3 and 4 repeat the same use of Lemma B(ii) in higher degree.

*Step 0 (the case $$\lvert\nabla\Omega\rvert\equiv1$$).* The function $$\lvert\nabla\Omega\rvert^2-1$$ is real-analytic on the punctured ball. Suppose first that it vanishes identically. Its part of degree $$m-1$$, $$m\ge2$$, is $$2\hat k\cdot\nabla\Omega_m+\sum\nabla\Omega_{m_1}\cdot\nabla\Omega_{m_2}$$ summed over $$m_1+m_2=m+1$$ with $$m_1,m_2\ge2$$. If $$\Omega_2=\dots=\Omega_{m-1}=0$$ the sum is empty, and Euler's relation $$k\cdot\nabla\Omega_m=m\Omega_m$$ gives $$2m\,\Omega_m/\lvert k\rvert=0$$. By induction $$\Omega_m=0$$ for all $$m\ge2$$, so $$\Omega=\lvert k\rvert$$ near $$0$$ and, by analytic continuation, on $$0<\lvert k\rvert<\rho_0$$.

Assume from now on that the zero set $$Z$$ of $$\lvert\nabla\Omega\rvert^2-1$$ is nowhere dense. For small $$\rho$$, $$\mathcal M_\rho$$ is a smooth hypersurface near $$\mathcal M_0^\ast\cap U$$, and the projection to the momentum $$q_i$$ of any one leg is a submersion: with $$q_1$$ fixed, $$\partial F_\rho/\partial q_3\to-\hat q_3+\hat q_4\ne0$$ by Lemma B(i), and likewise for the other legs. The preimage of the nowhere dense set $$Z$$ under the open map $$q\mapsto\rho q_i$$ is nowhere dense, so the quartets with $$\lvert v_i\rvert\ne1$$ for all four legs are open and dense in $$\mathcal M_\rho\cap U$$. By (S3) their velocities are pairwise distinct. Lemma A applies there, and (3) holds on a dense subset of $$\mathcal M_\rho\cap U$$, hence by continuity on all of it, for each small $$\rho$$.

*Step 1 (expansion).* Put $$\Psi(k)=\beta(k)\,(1,\nabla\Omega(k))=\sum_m\Psi_m(k)$$ with $$\Psi_m$$ homogeneous of degree $$m$$ and real-analytic off the origin. For $$q\in\mathcal M_\rho\cap U$$, (3) reads $$\sum_i\sigma_i\Psi(\rho q_i)=0$$. Momentum is conserved exactly, $$\sum_i\sigma_iq_i=0$$ on $$\Lambda$$, and by Lemma B(iii) every $$q^\ast\in\mathcal M_0^\ast\cap U$$ is a limit of points $$q_\rho\in\mathcal M_\rho\cap U$$ with $$q_\rho-q^\ast=O(\rho)$$.

*Step 2 (the lowest order fixes the coupling).* Let $$m_\ast$$ be the lowest degree of $$\beta$$, so that $$\beta_{m_\ast}\not\equiv0$$ by (S4) and Lemma A. Since $$\nabla\Omega=\hat q+O(\rho)$$, we have $$\Psi(\rho q)=\rho^{m_\ast}\beta_{m_\ast}(q)(1,\hat q)+O(\rho^{m_\ast+1})$$, and the limit $$\rho\to0$$ gives $$\sum_i\sigma_i\beta_{m_\ast}(q_i)(1,\hat q_i)=0$$ on $$\mathcal M_0^\ast\cap U$$. This extends to all of $$\mathcal M_0^\ast$$ by analyticity and Lemma B(i). The energy component is a homogeneous collision invariant of degree $$m_\ast$$, so $$m_\ast\in\{0,1\}$$ by Lemma B(ii).

Write $$\beta_{m_\ast}(q)=b(q)\lvert q\rvert$$ and $$p_i=(\lvert q_i\rvert,q_i)$$ for the null four-momenta. Then $$\sum_i\sigma_ib(q_i)\,p_i=0$$, and also $$\sum_i\sigma_ip_i=0$$. For a quartet in $$\mathcal M_0^\ast$$ with $$q_1,q_2,q_3$$ linearly independent, $$p_1,p_2,p_3$$ are independent and $$p_4=p_1+p_2-p_3$$, so the four vectors span a three-dimensional space and satisfy exactly one linear relation up to scale. Comparing the two relations gives $$b(q_1)=b(q_2)=b(q_3)=b(q_4)$$.

Given two linearly independent non-zero vectors $$t_1,t_2$$ there is such a quartet with $$q_1=t_1$$ and $$q_3=t_2$$. Take $$q_4=q_2+t_1-t_2$$, so that momentum is conserved, and require $$q_2=x$$ to lie on the surface $$\lvert x\rvert-\lvert x+t_1-t_2\rvert=\lvert t_2\rvert-\lvert t_1\rvert$$, which expresses energy conservation. This surface is a sheet of a hyperboloid with foci $$0$$ and $$t_2-t_1$$ and is not contained in a plane, because the right-hand side lies strictly between $$\pm\lvert t_1-t_2\rvert$$ when $$t_1,t_2$$ are not positively parallel. A point $$x$$ off the plane spanned by $$t_1,t_2$$ completes the quartet. So $$b(t_1)=b(t_2)$$ for independent $$t_1,t_2$$. For any two vectors there is a third one independent of both, so $$b$$ is constant, $$b\equiv\gamma$$, and $$\gamma\ne0$$ because $$\beta_{m_\ast}\not\equiv0$$. Then $$\beta_{m_\ast}=\gamma\lvert q\rvert$$ has degree one, so $$m_\ast=1$$ and $$\beta_1=\gamma\Omega_1$$.

*Step 3 (the energy component gives $$\beta=\gamma\Omega$$).* On $$\mathcal M_\rho$$ the energy is conserved exactly, $$\sum_i\sigma_i\Omega(\rho q_i)=0$$, so the energy component of Step 1 gives $$\sum_i\sigma_i(\beta-\gamma\Omega)(\rho q_i)=0$$. The function $$g=\beta-\gamma\Omega=\sum_{m\ge2}g_m$$ has no term of degree one. If $$g\not\equiv0$$, let $$g_m$$ be its lowest non-zero term. Then $$\rho^m\sum_i\sigma_ig_m(q_{\rho,i})+O(\rho^{m+1})=0$$, so $$\sum_i\sigma_ig_m(q^\ast_i)=0$$ on $$\mathcal M_0^\ast\cap U$$, and Lemma B(ii) gives $$g_m\equiv0$$, a contradiction. Hence $$\beta=\gamma\Omega$$ as series.

*Step 4 (the momentum components give the cone).* Now $$\Psi^j=\gamma\Omega\,\partial_j\Omega=\tfrac\gamma2\partial_j(\Omega^2)$$. Write $$\Omega^2=\lvert k\rvert^2+R$$ with $$R=\sum_{d\ge3}R_d$$, $$R_d$$ homogeneous of degree $$d$$ and real-analytic off the origin. Exact momentum conservation gives $$\sum_i\sigma_i\partial_j\lvert k_i\rvert^2=2\sum_i\sigma_ik_{i,j}=0$$, so (3) becomes $$\sum_i\sigma_i\partial_jR(\rho q_i)=0$$ on $$\mathcal M_\rho\cap U$$. Suppose $$R\not\equiv0$$ and let $$R_d$$ be its lowest non-zero term. Then $$\rho^{d-1}\sum_i\sigma_i\partial_jR_d(q_{\rho,i})+O(\rho^d)=0$$, so $$\partial_jR_d$$ is a homogeneous collision invariant of degree $$d-1\ge2$$ on $$\mathcal M_0^\ast\cap U$$ and vanishes by Lemma B(ii). A homogeneous function of positive degree with zero gradient vanishes, so $$R_d\equiv0$$, a contradiction. So $$R\equiv0$$ and $$\Omega^2=\lvert k\rvert^2$$ near $$0$$, and by analytic continuation on the punctured ball.
</div>

The energy row fixes the coupling to be proportional to the energy. The momentum rows say that $$\Omega\nabla\Omega=\tfrac12\nabla\Omega^2$$ is conserved in every hard process, which for a band that is not the cone cannot hold on all the collisions that the lattice allows at low energy.

## 4. A periodic symbol has no exact cone

<div class="theorem" markdown="1">
**Theorem 2** (no exact cone). Let $$A(k)$$ be an $$N\times N$$ matrix function on $$\mathbb R^3$$, real-analytic and $$2\pi$$-periodic in each coordinate. Let $$Q$$ be a positive-definite quadratic form and $$M\ge0$$.

(a) If $$A$$ is Hermitian (a lattice Hamiltonian), no continuous eigenvalue branch $$\Omega$$ of $$A(k)$$ satisfies $$\Omega(k)^2=Q(k)+M^2$$ on a non-empty open set.

(b) If $$A$$ is unitary (a Floquet or cellular-automaton symbol), no branch with $$e^{-i\Omega(k)}\in\mathrm{spec}\,A(k)$$ satisfies $$\Omega(k)^2=Q(k)+M^2$$ on a non-empty open set.

In both cases the open set is taken not to contain $$k=0$$ when $$M=0$$.
</div>

<div class="proof" markdown="1">
*Proof.* A determinant identity on a ball spreads by analyticity to all of space, and periodicity then places an unbounded sequence in a finite spectrum. On a small ball in the open set, $$\Omega\ne0$$ and is continuous, so $$\Omega=\chi\sqrt{Q+M^2}$$ with a fixed sign $$\chi=\pm1$$. Put $$\phi_0(k)=\sqrt{Q(k)+M^2}$$, real-analytic on $$\mathbb R^3$$ (on $$\mathbb R^3\setminus\{0\}$$ if $$M=0$$).

(a) The function $$\Xi(k)=\det(\chi\phi_0(k)-A(k))$$ is real-analytic on the connected set $$\mathbb R^3\setminus\{0\}$$ and vanishes on the ball, so $$\Xi\equiv0$$: $$\chi\phi_0(k)\in\mathrm{spec}\,A(k)$$ for all $$k\ne0$$. By periodicity $$\chi\phi_0(k+2\pi ne_1)\in\mathrm{spec}\,A(k)$$ for every integer $$n$$. But $$\phi_0(k+2\pi ne_1)\to\infty$$ and the spectrum of a fixed $$N\times N$$ matrix is bounded. Contradiction.

(b) The same argument with $$\Xi(k)=\det(e^{-i\chi\phi_0(k)}-A(k))$$ gives $$e^{-i\chi\phi_n}\in\mathrm{spec}\,A(k)$$ for all $$n$$, where $$\phi_n=\phi_0(k+2\pi ne_1)$$. The sign $$\chi$$ is the same for all $$n$$ because it was fixed in $$\Xi$$. The spectrum has at most $$N$$ points on the unit circle, so $$\chi\phi_n$$ lies, modulo $$2\pi$$, in a finite set $$\Theta$$ of phases, and the increments $$\delta_n=\phi_{n+1}-\phi_n$$ satisfy $$\chi\delta_n\in\Gamma=(\Theta-\Theta)+2\pi\mathbb Z$$, a set that meets every bounded interval in finitely many points. Now choose $$k$$ not parallel to $$e_1$$ and set $$\phi(t)=\sqrt{Q_{11}t^2+2Q_{1k}t+Q_{kk}+M^2}$$, with $$Q_{11}=Q(e_1)$$, $$Q_{1k}=Q(e_1,k)$$ the bilinear form and $$Q_{kk}=Q(k)$$, so that $$\phi_n=\phi(2\pi n)$$. Then

$$
\phi''(t)=\frac{Q_{11}(Q_{kk}+M^2)-Q_{1k}^2}{\big(Q_{11}t^2+2Q_{1k}t+Q_{kk}+M^2\big)^{3/2}}>0,
$$

by the strict Cauchy–Schwarz inequality $$Q_{1k}^2<Q_{11}Q_{kk}$$ for non-parallel vectors. So $$\phi$$ is strictly convex and the increments $$\delta_n=\phi(2\pi(n+1))-\phi(2\pi n)$$ are strictly increasing, and they are bounded above by $$2\pi\sqrt{Q_{11}}$$. A strictly increasing bounded sequence takes infinitely many values in a bounded interval, but $$\chi\Gamma$$ meets that interval in finitely many points. Contradiction.
</div>

For $$Q=\lvert k\rvert^2$$, $$M=0$$ and $$k=(0,1,0)$$ the increments are $$\delta_n=\sqrt{4\pi^2(n+1)^2+1}-\sqrt{4\pi^2n^2+1}$$; for $$n=1,2,3,10$$ they equal $$6.2438$$, $$6.2700$$, $$6.2766$$, $$6.2825$$, rising towards $$2\pi=6.2832$$ as the proof requires.

<div class="remark" markdown="1">
**Remark** (how much analyticity is used). The proof uses real-analyticity of $$A$$ only on the complement of a discrete set that is invariant under the translations by $$2\pi\mathbb Z^3$$, such as $$2\pi\mathbb Z^3$$ itself, because that complement is connected. Smoothness is not enough. Let $$g$$ be smooth with $$g=1$$ for $$\lvert k\rvert\le1$$ and $$g=0$$ for $$\lvert k\rvert\ge2$$, and let $$A(k)$$ be the $$2\pi$$-periodic extension of $$\lvert k\rvert\,g(\lvert k\rvert)$$ from the cube $$[-\pi,\pi]^3$$; this is a scalar ($$N=1$$) symbol, smooth away from $$2\pi\mathbb Z^3$$, equal to the exact cone $$\lvert k\rvert$$ near the origin. The conical point at $$k=0$$ makes the hopping amplitudes, the Fourier coefficients of $$A$$, decay as the power law $$-1/(\pi^2\lvert x\rvert^4)$$ in the lattice distance $$\lvert x\rvert$$; the transition region of $$g$$ is where analyticity fails. This is a toy band, not a graviton. Whether crystal-momentum periodicity alone excludes an exact cone for $$C^\infty$$ symbols with non-vanishing hard scattering is not decided here.
</div>

<div class="remark" markdown="1">
**Remark** (periodicity at complex momentum; conditional). In a formulation with a Euclidean time direction the on-shell variety is the complex cone $$\sum_\mu p_\mu^2=0$$, and the kinetic symbol is not a finite Hermitian or unitary matrix, so Theorem 2 does not apply. Let $$P(p)$$ be a function that is $$2\pi$$-periodic in $$p_1$$ (a lattice-momentum charge) and real-analytic in the closed strip $$\lvert\mathrm{Im}\,p_\mu\rvert\le\pi/\sqrt3$$, and assume $$P(p)=p_1$$ on the complex cone inside the strip. This assumption is the Euclidean analogue of Theorem 1 and is not derived here. If $$p$$ and $$p-2\pi e_1$$ are both on the cone, subtracting the two conditions gives $$4\pi p_1-4\pi^2=0$$, so $$p_1=\pi$$ and $$p_2^2+p_3^2+p_4^2=-\pi^2$$. Writing $$p_j=x_j+iy_j$$ gives $$\sum_jy_j^2=\pi^2+\sum_jx_j^2\ge\pi^2$$, hence $$\max_j\lvert y_j\rvert\ge\pi/\sqrt3$$, with equality at $$p=(\pi,i\pi/\sqrt3,i\pi/\sqrt3,i\pi/\sqrt3)$$. At this point $$P(p)=\pi$$ and $$P(p-2\pi e_1)=-\pi$$, which contradicts $$P(p)=P(p-2\pi e_1)$$. A strip of smaller height does not contain such a pair of points, and the argument gives nothing there.
</div>

<div class="corollary" markdown="1">
**Corollary 3** (no analytic Bloch symbol satisfies the hypotheses of Theorem 1). There is no lattice Hamiltonian or Floquet model with a finite-dimensional, real-analytic, periodic Bloch symbol whose graviton band satisfies (B) and (S1)–(S4).
</div>

<div class="proof" markdown="1">
*Proof.* Theorem 1 makes the band an exact cone on a punctured ball, which is a non-empty open set not containing $$0$$; in the original axes it is $$\Omega^2=Q$$ with $$Q$$ positive definite. Theorem 2 forbids this for any such symbol at any lattice spacing.
</div>

The corollary concerns exact soft universality in the sense of (S1)–(S4). It does not say that lattice theories cannot have gravitons, and section 5 shows how close a band can come to satisfying the necessary condition (3).

## 5. The order of the defect

The condition (3) is necessary for a gauge-invariant soft factor of the form (S1). For a band satisfying (B) and a coupling that is admissible in the following sense, we measure how far a band is from satisfying it. A coupling is admissible if $$\beta=\sum_{m\ge1}\beta_m$$ with $$\beta_m$$ real-analytic on $$\mathbb R^3\setminus\{0\}$$ and homogeneous of degree $$m$$, and $$\beta_1=\gamma\lvert q\rvert$$ with $$\gamma\ne0$$, which is the lowest-order form that Step 2 of Theorem 1 forces. The *defect* of a hard quartet $$q\in\mathcal M_\rho$$ is

$$
\mathcal D_\beta(\rho,q)=\sum_i\sigma_i\,\beta(\rho q_i)\,\big(1,\nabla\Omega(\rho q_i)\big). \tag{5}
$$

It vanishes identically for the exact cone with $$\beta=\gamma\lvert k\rvert$$, and it is the failure of the pole term of the soft factor to be gauge invariant.

<div class="proposition" markdown="1">
**Proposition 4** (order of the defect). Assume (B) and let $$\Omega^2=\lvert k\rvert^2+R_d+O(\lvert k\rvert^{d+1})$$ with $$R_d\not\equiv0$$ homogeneous of degree $$d\ge3$$.

(i) For $$\beta=\gamma\Omega$$ the energy component of $$\mathcal D_\beta$$ vanishes and its momentum components are $$\tfrac\gamma2\rho^{d-1}\sum_i\sigma_i\partial_jR_d(q_i)+O(\rho^d)$$.

(ii) Let $$\mathcal O\subset\mathcal M_0^\ast$$ be non-empty and relatively open. For no admissible $$\beta$$ is $$\mathcal D_\beta(\rho,q_\rho)=o(\rho^{d-1})$$ for all families $$q_\rho\in\mathcal M_\rho$$ with $$q_\rho\to q\in\mathcal O$$.

Relative to $$\sum_i\lvert\rho q_i\rvert\propto\rho$$, the defect is therefore of order $$\rho^{d-2}$$, and no admissible $$\beta$$ gives a smaller one. We call $$r=d-2$$ the order of the defect. For a separable band $$\Omega^2=\sum_{j=1}^3f_w(k_j)$$ built from the order-$$2w$$ central-difference function $$f_w$$ for $$k^2$$, $$d=2w+2$$ and $$r=2w$$.
</div>

<div class="proof" markdown="1">
*Proof.* (i) The energy component is $$\sum_i\sigma_i\gamma\Omega(\rho q_i)=0$$ on the energy shell. The momentum components are $$\tfrac\gamma2\sum_i\sigma_i\partial_j\Omega^2(\rho q_i)=\tfrac\gamma2\sum_i\sigma_i\partial_jR(\rho q_i)$$ by $$\sum_i\sigma_ik_i=0$$, and the leading term of $$\partial_jR$$ is $$\partial_jR_d$$.

(ii) Suppose $$\mathcal D_\beta=o(\rho^{d-1})$$ along all such families. By Lemma B(iii) every $$q\in\mathcal O$$ is the limit of such a family. The energy component of $$\mathcal D_\beta$$ equals $$\sum_i\sigma_ig(\rho q_i)$$ for $$g=\beta-\gamma\Omega$$, as in Step 3 of Theorem 1. If $$g_m$$ is the lowest non-zero term of $$g$$ and $$m\le d-1$$, then $$\sum_i\sigma_ig_m=0$$ on $$\mathcal O$$, and Lemma B(ii) gives $$g_m=0$$. Hence $$\beta_m=\gamma\Omega_m$$ for $$m\le d-1$$. Here $$\Omega_m=0$$ for $$2\le m\le d-2$$ and $$\Omega_{d-1}=R_d/(2\lvert k\rvert)$$. The degree-$$(d-1)$$ part of $$\beta\,\partial_j\Omega$$ is then $$\gamma\lvert q\rvert\partial_j\Omega_{d-1}+\gamma\Omega_{d-1}\hat q_j=\tfrac\gamma2\partial_jR_d$$, and the lower parts are either conserved exactly ($$\gamma q_j$$ in degree one) or zero. So the momentum components of $$\mathcal D_\beta$$ are $$\tfrac\gamma2\rho^{d-1}\sum_i\sigma_i\partial_jR_d(q_i)+O(\rho^d)$$, and $$o(\rho^{d-1})$$ would make $$\partial_jR_d$$ a collision invariant of degree $$d-1\ge2$$ on $$\mathcal O$$. By Lemma B(ii) it vanishes, hence $$R_d=0$$, a contradiction.

For the separable bands we use $$f_w(k)=2\sum_{\iota=1}^{w}\dfrac{(2\sin(k/2))^{2\iota}}{\iota^2\binom{2\iota}{\iota}}$$, the truncation at $$\iota=w$$ of the series $$k^2=2\sum_{\iota\ge1}(2\sin(k/2))^{2\iota}/(\iota^2\binom{2\iota}{\iota})$$ (the series of $$(2\arcsin y)^2$$ with $$y=\sin(k/2)$$). It is a trigonometric polynomial of degree $$w$$ and for $$w=1,2,3$$ it equals $$2-2\cos k$$, $$\tfrac52-\tfrac83\cos k+\tfrac16\cos2k$$ and $$\tfrac{49}{18}-3\cos k+\tfrac3{10}\cos2k-\tfrac1{45}\cos3k$$. The omitted terms ($$\iota>w$$) give $$f_w(k)-k^2=c_wk^{2w+2}+O(k^{2w+4})$$ with $$c_w=-2(w!)^2/(2w+2)!$$, which is non-zero for every $$w$$; for $$w=1,2,3$$ it is $$-\tfrac1{12}$$, $$-\tfrac1{90}$$, $$-\tfrac1{560}$$. So $$R_{2w+2}=c_w\sum_jk_j^{2w+2}\not\equiv0$$.
</div>

For these three bands, exact on-shell quartets were constructed at $$\rho$$ between $$0.05$$ and $$0.4$$ for twelve randomly chosen configurations, and the momentum-row defect of $$\beta=\Omega$$, divided by $$\sum_i\lvert k_i\rvert$$, was computed. Its median over the configurations falls as $$\rho^{2}$$, $$\rho^{4}$$ and $$\rho^{6}$$ for $$w=1,2,3$$, with fitted exponents $$2.0$$, $$4.0$$ and $$6.0$$ to within $$0.05$$, as part (i) predicts. This confirms the expansion in (i); it does not test the soft factor.

Proposition 4 is a statement about the algebraic condition (3). It shows that the condition can be met to relative order $$\rho^{d-2}$$ by a band whose $$\Omega^2-\lvert k\rvert^2$$ starts at degree $$d$$, with $$\beta=\gamma\Omega$$, and to no higher order by any admissible $$\beta$$. It does not construct an amplitude whose soft factor is gauge invariant to that order, since (S1) and (S2) are hypotheses of exact invariance.

*Example.* The following finite-cell automaton has $$d=4$$ and $$r=2$$. Let $$\Sigma_j$$ be the Pauli matrices, $$\mathsf R_j(t)=\cos t-i\sin t\,\Sigma_j$$ and

$$
V(k)=\mathsf R_x(k_x/2)\,\mathsf R_y(k_y/2)\,\mathsf R_z(k_z)\,\mathsf R_y(k_y/2)\,\mathsf R_x(k_x/2)\in SU(2),
$$

which is $$2\pi$$-periodic in each $$k_j$$. Write $$V=\cos\lambda-i\sin\lambda\,\hat n\cdot\Sigma$$. A direct computation gives $$\cos\lambda=\cos k_x\cos k_y\cos k_z$$. Let $$D^{(s)}$$ be the $$(2s+1)$$-dimensional irreducible representation of $$SU(2)$$ and take the one-step symbol $$W(k)=D^{(s)}(V(k/2s))$$, which acts on $$2s+1$$ internal states per cell. Since $$D^{(s)}(V)=e^{-2i\lambda\,\hat n\cdot S}$$, its eigenphases are $$2\varsigma\lambda(k/2s)$$ for $$\varsigma=-s,\dots,s$$, and the helicity-$$s$$ band is

$$
\Omega_s(k)=2s\,\arccos\!\Big(\prod_{j=1}^{3}\cos\frac{k_j}{2s}\Big),\qquad
\Omega_s^2=\lvert k\rvert^2-\frac1{12s^2}\sum_{i<j}k_i^2k_j^2+O(k^6). \tag{6}
$$

The symbol $$W$$ has period $$4\pi s$$ in each $$k_j$$, so in these units the lattice spacing is $$a=1/(2s)$$ and the quartic coefficient is $$-a^2/3$$; for $$s=2$$ it is $$-1/48$$. The band is not a cone for any $$s$$ (Theorem 2), $$R_4=-\frac1{12s^2}\sum_{i<j}k_i^2k_j^2\not\equiv0$$, and Proposition 4 gives $$d=4$$, $$r=2$$.

<div class="remark" markdown="1">
**Remark** (equivalence-principle estimate; not a theorem). Weinberg's theorem and its S-matrix sharpenings assume an exactly Poincaré-covariant S-matrix. If the S-matrix is covariant only up to relative corrections of order $$(Ea)^r$$, the argument above gives universality up to such corrections at best; no result is known that forces exact universality from an ultraviolet theory with a preferred frame. As an order of magnitude, suppose that the relative violation of the equivalence principle for nucleon-bound matter is $$\delta_{\rm EP}\sim(p_Fa)^r$$, with the nucleon Fermi momentum $$p_F\approx0.25$$ GeV and a composition-dependent coefficient of order one. The MICROSCOPE experiment found $$\eta(\mathrm{Ti},\mathrm{Pt})=[-1.5\pm2.3\,(\mathrm{stat})\pm1.5\,(\mathrm{syst})]\times10^{-15}$$, the statistical errors at $$1\sigma$$ [[Touboul et al.]](#ref-touboul). Adding the two uncertainties in quadrature and taking the central value plus two standard deviations gives $$\lvert\eta\rvert<7\times10^{-15}$$. Setting $$\delta_{\rm EP}$$ equal to this bound gives $$1/a\gtrsim p_F/(7\times10^{-15})^{1/r}=3\times10^6$$ GeV for $$r=2$$, $$9\times10^2$$ GeV for $$r=4$$ and $$57$$ GeV for $$r=6$$. The assumption $$\delta_{\rm EP}\sim(p_Fa)^r$$ is dimensional, and the other costs of a preferred frame, such as the per-species speed shifts discussed in [Lattice gravitons, exact diffeomorphisms and lattice momentum](/quantum-gravity/lattice-diffeomorphisms/), are not included.
</div>

## 6. The Weinberg–Witten theorem and a local lattice energy density

The Weinberg–Witten theorem could give a second obstruction. Its hypothesis is a Lorentz-covariant, conserved, local energy-momentum tensor $$T^{\mu\nu}$$ with the Poincaré charges as its integrals. The standard argument for a massless particle of helicity $$h$$ runs as follows; we state it as a sketch. Take the momenta $$p$$ and $$-p$$ along an axis, where the one-particle states have angular momentum $$h$$ and $$-h$$ about it. A rank-2 tensor operator changes the angular momentum about that axis by at most 2 in absolute value, so $$\langle -p,h\vert T^{\mu\nu}\vert p,h\rangle=0$$ if $$\lvert2h\rvert>2$$. Boosts map any pair of non-parallel massless momenta to a back-to-back pair, so covariance gives $$\langle p',h\vert T^{\mu\nu}\vert p,h\rangle=0$$ for all non-parallel $$p,p'$$ when $$\lvert h\rvert>1$$. If the matrix element is continuous at $$p'=p$$, this contradicts the normalisation $$\langle p,h\vert T^{00}\vert p,h\rangle=\Omega(p)$$. We call the vanishing for non-parallel momenta the covariant benchmark.

On a lattice the energy density is a local operator that need not be covariant, so the benchmark need not hold. The following model quantifies how far from the benchmark a local analytic energy density is. In the model the one-particle space at small wave vector is the spin-$$s$$ irreducible representation of $$SU(2)$$, and the helicity-$$s$$ band has the eigenvector $$u(k)=\lvert\hat n(k)\rangle$$, the spin coherent state along $$\hat n(k)=\hat k+O(\lvert k\rvert)$$. A graviton has only the helicities $$\pm2$$, so the model is a stand-in with the same angular-momentum content, as in the automaton of section 5; a statement about it is not a statement about every possible graviton.

<div class="proposition" markdown="1">
**Proposition 5** (a local energy density is not covariant on helicity-$$s$$ states). In the model above, let the energy density be a bilinear form with kernel $$\tau(k',k)$$, real-analytic at $$(0,0)$$ (as for an exponentially local lattice operator). Let $$T(k',k)=u(k')^\dagger\tau(k',k)u(k)$$ and assume that on the diagonal it is the band energy, $$T(k,k)=\Omega(k)=\lvert k\rvert(1+o(1))$$. Then for $$k=\rho\hat n$$, $$k'=\rho\hat n'$$ the limit of $$\lvert T(k',k)\rvert/\Omega(k)$$ as $$\rho\to0$$ exists and equals $$\lvert G(\hat n',\hat n)\rvert$$, where

$$
G(\hat n',\hat n)=\langle\hat n'\vert L(\hat n',\hat n)\vert\hat n\rangle
$$

($$L$$ is the linear part of $$\tau$$ at the origin), $$\lvert G\rvert$$ is continuous on $$S^2\times S^2$$, and $$\lvert G(\hat n,\hat n)\rvert=1$$. In particular there is $$\theta_0>0$$, independent of $$\rho$$, with $$\lvert T(k',k)\rvert\ge\tfrac12\Omega(k)$$ for all small $$\rho$$ and all directions with $$\lvert\hat n'-\hat n\rvert<\theta_0$$.
</div>

<div class="proof" markdown="1">
*Proof.* Expand $$\tau(k',k)=\tau_0+L(k',k)+O(\rho^2)$$ with $$L$$ linear. On the diagonal, $$T(k,k)=\langle\hat n\vert\tau_0\vert\hat n\rangle+O(\rho)$$ must be $$O(\rho)$$, so $$\langle\hat n\vert\tau_0\vert\hat n\rangle=0$$ for every unit $$\hat n$$. The diagonal coherent-state symbol is injective on operators on the spin-$$s$$ representation (it is a polynomial in the components of $$\hat n$$ that determines the matrix), so $$\tau_0=0$$. Then $$T(k',k)=\rho\,\langle\hat n'\vert L(\hat n',\hat n)\vert\hat n\rangle+O(\rho^2)$$ up to the arbitrary phases of $$u(k),u(k')$$. The modulus of $$G$$ does not depend on these phases and is continuous, and $$\lvert G(\hat n,\hat n)\rvert=\lim T(k,k)/\Omega(k)=1$$. The last statement follows from the uniform continuity of $$\lvert G\rvert$$ on $$S^2\times S^2$$ and the uniformity of the $$O(\rho^2)$$ remainder.
</div>

For $$s\ge\frac32$$ and non-parallel momenta $$k,k'$$ of equal length with a small angle, the covariant benchmark is $$T=0$$ and the proposition gives $$\lvert T\rvert\ge\tfrac12\Omega$$. The hypothesis $$s\ge\tfrac32$$ enters only in this comparison and not in the proof. Within the model, a local analytic energy density is therefore not covariant on helicity-$$s$$ states, not even approximately: the violation is of relative order one and does not decrease with the scale, so the Weinberg–Witten hypothesis is not met. This is the lattice form of the fact that a free massless spin-2 field has no covariant conserved rank-2 energy-momentum tensor; the covariant conserved bilinear is the Bel–Robinson tensor, of rank four.

An explicit family has the one-particle generator $$h(k)=\frac{\Omega(k)}{s}\,\hat n(k)\cdot S$$ in the spin-$$s$$ representation, whose helicity-$$s$$ eigenvector is $$u(k)=\lvert\hat n(k)\rangle$$ with eigenvalue $$\Omega(k)$$. Take the energy density $$\tau^{00}(k',k)=\tfrac12[\psi(h(k'))+\psi(h(k))]$$ for a function $$\psi$$. Because $$u(k)$$ is an eigenvector of $$\psi(h(k))$$ with eigenvalue $$\psi(\Omega(k))$$, and the overlap of two spin-$$s$$ coherent states is $$\lvert\langle\hat n'\vert\hat n\rangle\rvert=((1+\hat n\cdot\hat n')/2)^s=\cos^{2s}(\theta/2)$$ for $$\theta$$ the angle between $$\hat n$$ and $$\hat n'$$, the matrix element is exactly

$$
\lvert T^{00}(k',k)\rvert=\tfrac12\lvert E(k)+E(k')\rvert\Big(\frac{1+\hat n(k)\cdot\hat n(k')}2\Big)^{s},\qquad E=\psi(\Omega). \tag{7}
$$

If $$\Omega(k)\hat n(k)=k$$ and $$\psi$$ is real-analytic with $$\psi(0)=0$$, $$\psi'(0)=1$$, then $$h(k)=k\cdot S/s$$ and $$\tau^{00}$$ is analytic at the origin, so the hypotheses of Proposition 5 hold. For $$s=2$$ and equal energies, (7) gives $$\lvert T^{00}\rvert/E=\cos^4(\theta/2)=0.8705,\ 0.5625,\ 0.25,\ 0.0625$$ at $$\theta=\pi/6,\pi/3,\pi/2,2\pi/3$$, independent of $$\rho$$. For $$s=1$$ it gives $$\cos^2(\theta/2)$$, the overlap of two helicity-one polarisation vectors; no contradiction arises there, because the photon has a covariant conserved energy-momentum tensor, Maxwell's, and the Weinberg–Witten argument does not require the matrix element to vanish for $$\lvert h\rvert\le1$$. For $$s=\tfrac12$$ it gives $$\cos(\theta/2)$$.

## 7. Escape routes and open points

Theorem 1 and Corollary 3 concern exact universality of the soft factor, and each hypothesis of Theorem 1 marks a way out.

- If the soft coupling vanishes ((S4) fails) there is no leading soft charge and no Newtonian limit. Cubic vertices built from the linearised curvature are of this kind: they are invariant under the linear gauge transformations by themselves, can be finite-range, and carry no coupling to energy.
- If the nonlinear gauge invariance is broken explicitly ((S2) fails), Lemma A does not apply. The band-limited vertices of [Lattice gravitons, exact diffeomorphisms and lattice momentum](/quantum-gravity/lattice-diffeomorphisms/) are an instance, by the heuristic criterion stated there.
- If the hard legs do not have simple poles $$1/(\mathsf s_i\cdot K)$$ ((S1) fails), the argument has no starting point. For interacting cellular automata the existence of such poles in discrete time is not established here.
- If $$J$$ is a matrix in helicity or species space rather than a multiple of the identity ((S1) fails in that clause), a matrix-valued $$\beta$$ is not covered by Lemma A.
- If the band is non-analytic in the direction $$\hat k$$ at the origin, which can occur when helicity bands are degenerate at leading order, (B) fails and the jet-level argument of Theorem 1 does not apply.
- Theorem 2 uses a finite-dimensional fibre. A continuum free massless field regarded as a lattice theory with one unit cell has all bands exactly conical, $$\Omega=\lvert k+2\pi n\rvert$$, and an unbounded fibre. Whether bounded infinite-dimensional Hamiltonian fibres, and unitary ones, escape is not settled here.
- Theorem 2 uses analyticity; the first remark of section 4 gives a smooth scalar toy band with an exact cone, and the second gives a conditional obstruction at complex momentum for a charge analytic in a strip of height $$\pi/\sqrt3$$ or more.

Spin one is not an instance of any of these: the soft charge is an internal scalar, and the corresponding condition holds for any dispersion.

The hypotheses (S1)–(S3) are assumptions on the soft expansion of lattice amplitudes. Deriving them for an interacting lattice or automaton graviton, including simple poles in discrete time, is open. The order $$r$$ of Proposition 4 measures the failure of an algebraic condition; how it translates into observable violations of the equivalence principle depends on the dynamics, and the estimate of section 5 is dimensional only. It is also open whether the non-covariant energy density of section 6 can be the source to which a cubic graviton vertex couples with a soft limit of defect order $$r$$, as the gravitational pseudotensor is in general relativity.

## References

1. <a id="ref-w64"></a>Weinberg, “Photons and gravitons in S-matrix theory: derivation of charge conservation and equality of gravitational and inertial mass,” Phys. Rev. 135, B1049 (1964).
2. <a id="ref-w65"></a>Weinberg, “Infrared photons and gravitons,” Phys. Rev. 140, B516 (1965).
3. <a id="ref-ww"></a>Weinberg, Witten, “Limits on massless particles,” Phys. Lett. B 96, 59 (1980).
4. <a id="ref-porrati"></a>Porrati, “Universal limits on massless high-spin particles,” Phys. Rev. D 78, 065016 (2008), [arXiv:0804.4672](https://arxiv.org/abs/0804.4672).
5. <a id="ref-bc"></a>Benincasa, Cachazo, “Consistency conditions on the S-matrix of massless particles,” [arXiv:0705.4305](https://arxiv.org/abs/0705.4305).
6. <a id="ref-kss"></a>Kato, Sakamoto, So, “No-go theorem of Leibniz rule and supersymmetry on the lattice,” PoS LATTICE2008, 233, [arXiv:0810.2360](https://arxiv.org/abs/0810.2360).
7. <a id="ref-touboul"></a>Touboul et al. (MICROSCOPE collaboration), “MICROSCOPE mission: final results of the test of the Equivalence Principle,” Phys. Rev. Lett. 129, 121102 (2022), [arXiv:2209.15487](https://arxiv.org/abs/2209.15487).
