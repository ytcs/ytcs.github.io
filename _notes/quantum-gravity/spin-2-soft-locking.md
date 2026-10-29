---
title: "Exact universal graviton coupling and periodic dispersion"
date: 2026-10-29
category: gravity
order: 2
description: A graviton whose soft coupling to energy and momentum is exactly Weinberg's must have an exactly conical dispersion near zero momentum, and no band of a finite-dimensional real-analytic periodic symbol is exactly conical. Universality to a finite order in the lattice spacing is possible, and no better than that.
status: "The no-exact-cone theorem for finite-fibre analytic symbols is proved without further assumptions. The locking theorem is proved under explicitly stated hypotheses on the soft expansion, which are assumed here and not derived from a lattice action. The universality order and the Weinberg–Witten statement are proved for the stated classes; numerical checks are included."
excludes: "Lattice Hamiltonian and Floquet (cellular-automaton) gravitons with a periodic, real-analytic, finite-dimensional symbol whose soft graviton coupling is exactly universal, that is, exactly gauge invariant with a non-zero Weinberg charge, and whose hard graviton scattering is non-zero."
---

On a lattice the graviton dispersion is periodic in the wave vector, so it cannot be exactly $$\omega=\lvert k\rvert$$ everywhere. The question is whether the graviton can nevertheless couple to energy and momentum exactly as Weinberg's soft theorem demands, with a universal coupling and exact gauge invariance of the soft factor. The answer is no, for lattice Hamiltonians and for Floquet cellular automata whose Bloch symbol is real-analytic with finite-dimensional fibres, under the hypotheses of section 2 on the soft expansion. The argument has two steps. Exact universality forces the free graviton band to be an exact cone, $$\Omega(k)=\sqrt{Q(k)}$$ with $$Q$$ a positive quadratic form, on a neighbourhood of $$k=0$$ (Theorem 1). No band of a finite-dimensional, real-analytic, periodic symbol is an exact cone on any open set (Theorem 2). The word *exact* carries the statement. The failure of the exact-universality condition can be made of relative order $$(Ea)^\nu$$, with $$a$$ the lattice spacing and $$E$$ the energy, for every even $$\nu\ge2$$ by choosing the contact of the band with the cone, and it cannot be made smaller than the order set by that contact; $$\nu=\infty$$ is the exact cone (Proposition 5). The Weinberg–Witten theorem does not bind a graviton with a local lattice energy density (Proposition 6).

Weinberg derived the universality of the soft graviton coupling, and with it the equality of gravitational and inertial mass, from the pole structure of amplitudes with a soft graviton and Lorentz-covariant kinematics [[Weinberg 1964]](#ref-w64) [[Weinberg 1965]](#ref-w65). Porrati, and Benincasa and Cachazo, gave the S-matrix versions that exclude massless higher-spin particles with Poincaré-covariant interactions [[Porrati]](#ref-porrati) [[Benincasa–Cachazo]](#ref-bc). Weinberg and Witten showed that a massless particle of helicity above one cannot carry a Lorentz-covariant conserved energy-momentum tensor [[Weinberg–Witten]](#ref-ww). For supersymmetry, no lattice has translation invariance, locality and the Leibniz rule together [[Kato–Sakamoto–So]](#ref-kss), and the present argument is its spin-2 relative, with translations in the place of supersymmetry. The companion page [Lattice gravitons, exact diffeomorphisms and lattice momentum](/quantum-gravity/lattice-diffeomorphisms/) proves that a conserved lattice momentum with an analytic symbol is incompatible with low-energy graviton scattering. That is the charge side of the same obstruction; here it is the dispersion that is locked, through the soft theorem. The lemma on collision invariants of massless kinematics proved there (its Lemma 4) is used below without repetition.

This page adds the transposition of Weinberg's argument to a non-covariant on-shell variety, with explicit hypotheses; the proof that the locking leaves only an exact cone; the theorem that finite-dimensional analytic periodic symbols have no exact cone; the order $$\nu$$ at which universality can hold; and a statement of what Weinberg–Witten does and does not give on a lattice. The units are those of the lattice spacing, and the low-energy speed of light is 1.

## 1. Weinberg's argument on the light cone

Consider a process with hard particles $$i=1,\dots,n$$ of momenta $$k_i$$ and energies $$\Omega_i$$, incoming ($$\sigma_i=+1$$) or outgoing ($$\sigma_i=-1$$), and add one soft graviton of momentum $$q=(q_0,\mathbf q)$$ and polarisation $$\varepsilon_{\mu\nu}$$. The amplitude has poles in $$q$$ from the propagators of the hard legs next to the emission. In the continuum, the leading term is

$$
\mathcal M_{n+1}\simeq\kappa\sum_i\sigma_i\frac{p_i^\mu p_i^\nu\varepsilon_{\mu\nu}}{p_i\cdot q}\,\mathcal M_n ,\qquad p_i=(\Omega_i,k_i). \tag{1}
$$

A gauge transformation $$\varepsilon_{\mu\nu}\to\varepsilon_{\mu\nu}+q_\mu\xi_\nu+\xi_\mu q_\nu$$ with $$q\cdot\xi=0$$ changes (1) by $$2\kappa\,\xi\cdot\sum_i\sigma_ip_i\,\mathcal M_n$$, which vanishes because of energy-momentum conservation. The coupling $$\kappa$$ is the same for every species, because it multiplies the conserved four-vector $$\sum\sigma_ip_i$$ and nothing else: if two species had different couplings $$\kappa_A,\kappa_B$$, the sum would be $$\kappa_A\sum_Ap_i+\kappa_B\sum_Bp_i$$, which does not vanish in general. This is the universality of the gravitational coupling.

For a photon the same reasoning, with $$\varepsilon_\mu\to\varepsilon_\mu+q_\mu$$, gives $$\sum_i\sigma_ie_i=0$$ for the charges $$e_i$$. The soft charge is a scalar, an internal quantum number; it is conserved on any lattice that has an exact $$U(1)$$, whatever the dispersion. For the graviton the conserved soft charge is the four-vector $$\sum\sigma_ip_i$$, the energy-momentum, and it is tied to the kinematics of the hard legs. On a lattice the question is therefore whether the dispersion allows the soft charge that gauge invariance requires to be the lattice momentum. The sections below show that it does only for an exact cone. In Weinberg's analysis spin three and higher is excluded because the corresponding conservation law would involve conserved tensor charges of higher rank, which Lorentz-covariant scattering does not allow.

## 2. The argument off the cone

Let the graviton band be a function $$\Omega(k)$$ on a punctured neighbourhood of the origin. In a Hamiltonian lattice theory $$\Omega$$ is an eigenvalue of the Bloch Hamiltonian $$H(k)$$; in a Floquet theory with time step 1 it is defined by an eigenvalue $$e^{-i\Omega(k)}$$ of the unitary Bloch matrix $$U(k)$$, and at low energy the energies are conserved as real numbers, with no wrap-around. A hard leg with momentum $$k_i$$ now emits a soft graviton with the denominator $$\Omega(k_i+q)-\Omega(k_i)-q_0\simeq-\mathsf s_i\cdot q$$, where

$$
\mathsf s_i\cdot q=q_0-v_i\cdot\mathbf q,\qquad v_i=\nabla\Omega(k_i). \tag{2}
$$

The covector $$\mathsf s_i=(1,-v_i)$$ is the normal to the on-shell variety $$\{q_0=\Omega(\mathbf q)\}$$ at $$k_i$$; on the cone it is proportional to $$p_i$$ and (2) reduces to $$p_i\cdot q/\Omega_i$$ up to sign.

We state the hypotheses on the soft expansion, because the lattice theories considered here have no covariant S-matrix from which they could be derived. They are assumptions about the theory, and Theorem 1 is conditional on them.

(B) *Band.* After a linear change of the spatial axes, $$\Omega(k)=\lvert k\rvert\,\tilde\omega(\lvert k\rvert,\hat k)$$ for $$0<\lvert k\rvert<\rho_0$$, where $$\tilde\omega$$ is real-analytic in a neighbourhood of $$[0,\rho_0)\times S^2$$ with $$\tilde\omega(0,\hat k)=1$$. Equivalently, $$\Omega=\sum_{m\ge1}\Omega_m$$ with $$\Omega_m$$ real-analytic on $$\mathbb R^3\setminus\{0\}$$ and positively homogeneous of degree $$m$$, $$\Omega_1=\lvert k\rvert$$, and the series converges with its first derivatives for $$\lvert k\rvert<\rho_0$$. This is the statement that the dispersion is relativistic at low energy and has a Taylor-type expansion in the scale; it holds, for example, when $$\Omega^2$$ is real-analytic at $$0$$ with $$\Omega^2=\lvert k\rvert^2+O(k^3)$$. Write $$\eta=\mathrm{diag}(-1,1,1,1)$$ for the metric whose null cone in $$q$$ is $$q_0=\lvert\mathbf q\rvert$$, the leading cone of the band.

(S1) *Pole structure.* For a hard $$2\to2$$ graviton process at small momenta with $$\mathcal M_4\ne0$$, the amplitude with an extra soft graviton of polarisation $$\varepsilon_{\mu\nu}$$ and momentum $$q$$ on the leading cone is, as $$q\to0$$,

$$
\mathcal M_5=\Big[\sum_i\sigma_i\frac{\varepsilon_{\mu\nu}J_i^{\mu\nu}}{\mathsf s_i\cdot q}+O(q^0)\Big]\mathcal M_4 ,
$$

with $$J_i^{\mu\nu}=J^{\mu\nu}(k_i)$$ a symmetric tensor that depends on the leg $$i$$ only, acts as a multiple of the identity on the helicity index, and is of the form $$\hat J(\lvert k\rvert,\hat k)$$ with $$\hat J$$ real-analytic near $$[0,\rho_0)\times S^2$$.

(S2) *Exact soft gauge invariance.* The pole term of (S1) is invariant under $$\varepsilon_{\mu\nu}\to\varepsilon_{\mu\nu}+q_\mu\xi_\nu+\xi_\mu q_\nu$$ for every $$q$$ on the leading cone and every $$\xi$$ with $$\eta(q,\xi)=0$$ (indices lowered with $$\eta$$). The variation of the pole term is of order $$q^0$$ and the variation of the next term is of order $$q$$, so the invariance of the amplitude implies that of the pole term separately.

(S3) *Non-degeneracy.* There are an open set $$U$$ in the space $$\Lambda$$ of momentum-conserving quartets, meeting the massless resonant set $$\mathcal M_0^\ast$$ of the companion page, and $$\rho_1>0$$, such that for $$0<\rho<\rho_1$$ the quartets $$\rho q$$ with $$q\in U$$ on the exact energy shell $$\mathcal M_\rho$$ have $$\mathcal M_4(\rho q)\ne0$$ and pairwise distinct velocities $$v_i$$.

(S4) *Non-zero coupling.* The function $$J$$ is not identically zero in the sense that the coefficient $$\beta$$ of Lemma A is not identically zero.

Here $$\mathcal M_\rho=\{q\in\Lambda:\ \sum\sigma_i\Omega(\rho q_i)=0\}$$, and $$\mathcal M_0^\ast$$ and $$\Lambda$$ are as in section 3 of the companion page; the $$2\to2$$ process has $$\sigma=(+,+,-,-)$$. The hypotheses hold in general relativity with $$\Omega=\lvert k\rvert$$ and $$J^{\mu\nu}=-\kappa\Omega\,\mathsf s^{\sharp\mu}\mathsf s^{\sharp\nu}$$ (see Lemma A for $$\mathsf s^\sharp$$). We do not assert that they hold for a lattice graviton; the theorem says what they imply.

## 3. Exact universality locks the dispersion

Let $$\mathsf s_i^\sharp$$ be the vector with $$\eta(\mathsf s_i^\sharp,x)=\mathsf s_i\cdot x$$ for all $$x$$, so $$\mathsf s_i^\sharp=-(1,v_i)$$ and $$\eta(\mathsf s_i^\sharp,\mathsf s_i^\sharp)=\lvert v_i\rvert^2-1$$. Let $$\hat J_i$$ be the operator $$(\hat J_i\xi)^\mu=J_i^{\mu\nu}\eta_{\nu\rho}\xi^\rho$$, which is self-adjoint for $$\eta$$ because $$J_i$$ is symmetric.

<div class="lemma" markdown="1">
**Lemma A** (the soft vertex). Assume (S1), (S2) and that, for the process considered, the velocities $$v_i$$ are pairwise distinct and $$\lvert v_i\rvert\ne1$$. Then $$J_i^{\mu\nu}=\alpha_i\eta^{\mu\nu}+\beta_i\,\mathsf s_i^{\sharp\mu}\mathsf s_i^{\sharp\nu}$$ for scalars $$\alpha_i,\beta_i$$ depending on $$k_i$$ only, and

$$
\sum_i\sigma_i\beta_i\,\mathsf s_i^\sharp=0,\qquad\text{that is}\qquad\sum_i\sigma_i\,\beta(k_i)\,\big(1,\nabla\Omega(k_i)\big)=0. \tag{3}
$$

</div>

<div class="proof" markdown="1">
*Proof.* The idea is that the leg $$i$$ is the only term of the gauge variation with a pole at $$\mathsf s_i\cdot q=0$$, so its numerator must vanish there; this fixes $$J_i$$ up to two scalars, and what remains of the variation is (3).

The gauge variation in (S2) is, with the operator $$\hat J_i$$,

$$
\sum_i\sigma_i\frac{\eta(q,\hat J_i\xi)}{\mathsf s_i\cdot q}=0\qquad\text{for real null }q,\ \eta(q,\xi)=0. \tag{4}
$$

Multiplying by $$\prod_j(\mathsf s_j\cdot q)$$ turns (4) into a polynomial identity in $$(q,\xi)$$. It holds on the real points of the variety $$V=\{(q,\xi)\in\mathbb C^4\times\mathbb C^4:\ \eta(q,q)=0,\ \eta(q,\xi)=0\}$$. The variety $$V$$ is irreducible (a rank-3 bundle over the irreducible null cone minus the origin), and its real points form a real-analytic submanifold of real dimension equal to its complex dimension, so they are Zariski dense; the identity therefore holds on all of $$V$$.

Fix $$i$$. Because $$\lvert v_i\rvert\ne1$$, $$\mathsf s_i^\sharp$$ is not null, and the hyperplane $$\mathsf s_i\cdot q=0$$ is a three-dimensional space on which $$\eta$$ is non-degenerate. The complex null vectors in it form a quadric cone $$N_i$$ (the complex null cone of the restriction of $$\eta$$, which is why the Zariski step above was needed: the real null vectors in it may be absent when $$\lvert v_i\rvert<1$$) that spans it, and $$N_i\setminus\{0\}$$ is connected. Because the velocities are distinct, the hyperplanes $$\mathsf s_j\cdot q=0$$ for $$j\ne i$$ do not contain $$N_i$$, so at a generic point of $$N_i$$ only the term $$i$$ of the multiplied identity survives. It gives $$\eta(q,\hat J_i\xi)=0$$ for $$q\in N_i$$ and all $$\xi$$ orthogonal to $$q$$. By self-adjointness, $$\eta(\hat J_iq,\xi)=0$$ for all $$\xi\perp q$$, so $$\hat J_iq$$ is a multiple of $$q$$: $$\hat J_iq=\lambda(q)q$$ for $$q\in N_i$$. The eigenvalue $$\lambda(q)$$ is continuous on the connected set $$N_i\setminus\{0\}$$ and takes values in the finite spectrum of $$\hat J_i$$, so it is a constant $$\lambda_i$$, and $$\hat J_i=\lambda_i$$ on the span of $$N_i$$, which is the hyperplane $$(\mathsf s_i^\sharp)^\perp$$. A self-adjoint operator that is $$\lambda_i$$ on $$(\mathsf s_i^\sharp)^\perp$$ maps its $$\eta$$-orthogonal complement, the line through $$\mathsf s_i^\sharp$$ (non-null, so complementary), into itself: $$\hat J_i\mathsf s_i^\sharp=\mu_i\mathsf s_i^\sharp$$. Hence $$\hat J_i=\lambda_i+(\mu_i-\lambda_i)\,\mathsf s_i^\sharp\,\eta(\mathsf s_i^\sharp,\cdot)/\eta(\mathsf s_i^\sharp,\mathsf s_i^\sharp)$$, which is the stated form with $$\alpha_i=\lambda_i$$ and $$\beta_i=(\mu_i-\lambda_i)/\eta(\mathsf s_i^\sharp,\mathsf s_i^\sharp)$$.

Insert this form into (4). The $$\alpha_i$$ term gives $$\alpha_i\eta(q,\xi)=0$$. The $$\beta_i$$ term gives $$\beta_i\,\eta(q,\mathsf s_i^\sharp)\,\eta(\mathsf s_i^\sharp,\xi)/(\mathsf s_i\cdot q)=\beta_i\,\eta(\mathsf s_i^\sharp,\xi)$$, because $$\eta(q,\mathsf s_i^\sharp)=\mathsf s_i\cdot q$$. So $$\eta(w,\xi)=0$$ for $$w=\sum_i\sigma_i\beta_i\mathsf s_i^\sharp$$ and every $$\xi\perp q$$, for every real null $$q$$. Then $$w$$ is parallel to every real null vector, hence $$w=0$$. Since $$\mathsf s_i^\sharp=-(1,v_i)$$ this is (3). The coefficient $$\beta_i$$ is a function of $$k_i$$ because $$J_i$$ and $$\mathsf s_i$$ are, and it has an expansion $$\beta=\sum_{m\ge0}\beta_m$$ in homogeneous real-analytic functions of degree $$m\ge0$$: the operators $$\eta$$ and $$\mathsf s^\sharp\mathsf s^\sharp$$ are uniformly independent as $$k\to0$$, and the analytic form of $$\hat J$$ is inherited.
</div>

For the exact cone $$\Omega=\lvert k\rvert$$, (3) reads $$\sum\sigma_i\beta(k_i)(1,\hat k_i)=0$$, which is energy-momentum conservation when $$\beta=a\lvert k\rvert$$: this is Weinberg's case. The lemma says that on a general band the vector $$\beta\,(1,\nabla\Omega)$$ must be conserved in every hard process, and it is not clear a priori that any $$\beta$$ achieves this. The remaining work is to show that only $$\beta\propto\Omega$$ and an exact cone do.

<div class="theorem" markdown="1">
**Theorem 1** (locking of the dispersion). Assume (B) and (S1)–(S4). Then $$\Omega(k)=\lvert k\rvert$$ for $$0<\lvert k\rvert<\rho_0$$, in the coordinates of (B): the band is an exact cone.
</div>

<div class="proof" markdown="1">
*Proof.* The idea is to expand (3) in the scale $$\rho$$ of the hard momenta. At each order the conserved vector must be a collision invariant of massless kinematics, and by Lemma 4 of the companion page the only homogeneous invariants are the energy-momentum itself and constants. This forces $$\beta=a\Omega$$ and then kills every correction to the cone.

*Step 0 (an eikonal case).* The function $$\lvert\nabla\Omega\rvert^2-1$$ is real-analytic on the punctured ball. Suppose first that it vanishes identically. The part of $$\lvert\nabla\Omega\rvert^2$$ of degree $$m-1$$, $$m\ge2$$, is $$2\hat k\cdot\nabla\Omega_m+\sum\nabla\Omega_a\cdot\nabla\Omega_b$$ over $$a+b=m+1$$, $$a,b\ge2$$. If $$\Omega_2=\dots=\Omega_{m-1}=0$$, the sum is empty, and Euler's relation $$k\cdot\nabla\Omega_m=m\Omega_m$$ gives $$2m\,\Omega_m/\lvert k\rvert=0$$. By induction $$\Omega_m=0$$ for all $$m\ge2$$ and $$\Omega=\lvert k\rvert$$, the claim. So assume from now on that the zero set of $$\lvert\nabla\Omega\rvert^2-1$$ is a nowhere dense analytic subset. Because the map from $$\mathcal M_\rho$$ to the momentum of one leg is a submersion, the quartets whose four velocities satisfy $$\lvert v_i\rvert\ne1$$ are open and dense in $$\mathcal M_\rho\cap\rho U$$, and by (S3) the velocities are pairwise distinct. Lemma A applies there and (3) holds on a dense subset of $$\mathcal M_\rho\cap\rho U$$, hence, being continuous, on all of it, for each small $$\rho$$.

*Step 1 (expansion of the invariance).* Put $$f(k)=\beta(k)\,(1,\nabla\Omega(k))$$. For $$q\in\mathcal M_\rho\cap U$$, (3) reads $$\sum_i\sigma_if(\rho q_i)=0$$ with $$\sigma=(+,+,-,-)$$, and $$f=\sum_mF_m$$ with $$F_m$$ homogeneous of degree $$m$$ and real-analytic off the origin. Momentum is conserved exactly at small momenta, $$\sum\sigma_iq_i=0$$. Since $$\mathcal M_\rho\to\mathcal M_0$$ in $$C^1$$ near points with all $$q_i\ne0$$ (the argument of Step 2 of Theorem 1 on the companion page), every $$q^\ast\in\mathcal M_0^\ast\cap U$$ is a limit of points $$q_\rho\in\mathcal M_\rho\cap U$$ with $$q_\rho-q^\ast=O(\rho)$$.

*Step 2 (the lowest order fixes the coupling).* Let $$m_0$$ be the lowest degree of $$\beta$$, so that $$\beta_{m_0}\not\equiv0$$ by (S4). Then $$f=\beta_{m_0}(q)(1,\hat q)+O(\rho)$$ at leading order and $$\sum\sigma_i\beta_{m_0}(q_i)(1,\hat q_i)=0$$ on $$\mathcal M_0^\ast\cap U$$, hence on all of $$\mathcal M_0^\ast$$ by analyticity and Lemma 3 of the companion page. The energy component is a homogeneous collision invariant of degree $$m_0$$, so $$m_0\in\{0,1\}$$ by Lemma 4. Write $$\beta_{m_0}(q)=B(q)\lvert q\rvert$$ and $$p_i=(\lvert q_i\rvert,q_i)$$, the null four-momenta. Then $$\sum\sigma_iB(q_i)\,p_i=0$$ and also $$\sum\sigma_ip_i=0$$. For a quartet in $$\mathcal M_0^\ast$$ with $$q_1,q_2,q_3$$ linearly independent, $$p_1,p_2,p_3$$ are independent, so the four vectors $$p_i$$ span a three-dimensional space and have exactly one linear relation, up to scale. Comparing the two relations gives $$B(q_1)=B(q_2)=B(q_3)=B(q_4)$$. Given two linearly independent non-zero vectors $$a,b$$ there is such a quartet with $$q_1=a$$ and $$q_3=b$$: choose $$q_2=x$$ on the surface $$\lvert x\rvert-\lvert x+a-b\rvert=\lvert b\rvert-\lvert a\rvert$$, a sheet of a hyperboloid that is not contained in a plane, and $$q_4=x+a-b$$; the right-hand side lies strictly between $$\pm\lvert a-b\rvert$$ because $$a,b$$ are not positively parallel. So $$B$$ is constant on pairs of independent vectors, hence, using a third independent vector, constant: $$B\equiv a$$ with $$a\ne0$$. Then $$\beta_{m_0}=a\lvert q\rvert$$ has degree one, $$m_0=1$$, and $$\beta_1=a\lvert q\rvert=a\Omega_1$$.

*Step 3 (the energy component gives $$\beta=a\Omega$$).* On $$\mathcal M_\rho$$ the energy is conserved exactly, $$\sum\sigma_i\Omega(\rho q_i)=0$$, and so $$\sum\sigma_i(\beta-a\Omega)(\rho q_i)=0$$ by the energy component of Step 1. The function $$g=\beta-a\Omega=\sum_{m\ge2}g_m$$ has no term of degree one. If $$g\not\equiv0$$ let $$g_m$$ be its lowest non-zero term, $$m\ge2$$. Then $$\rho^m\sum\sigma_ig_m(q_{\rho,i})+O(\rho^{m+1})=0$$, so $$\sum\sigma_ig_m(q^\ast_i)=0$$ on $$\mathcal M_0^\ast\cap U$$, and Lemma 4 gives $$g_m\equiv0$$, a contradiction. Hence $$\beta=a\Omega$$ as series.

*Step 4 (the momentum components give the cone).* Now $$f^j=a\Omega\,\partial_j\Omega=\tfrac a2\partial_j(\Omega^2)$$. Write $$\Omega^2=\lvert k\rvert^2+R$$ with $$R=\sum_{p\ge3}R_p$$, $$R_p$$ homogeneous of degree $$p$$ and real-analytic off the origin. The conserved momentum gives $$\sum\sigma_i\partial_j\lvert k_i\rvert^2=2\sum\sigma_ik_{ij}=0$$ exactly, so (3) becomes $$\sum_i\sigma_i\partial_jR(\rho q_i)=0$$ on $$\mathcal M_\rho\cap U$$. Suppose $$R\not\equiv0$$ and let $$R_p$$ be its lowest non-zero term. Then $$\rho^{p-1}\sum\sigma_i\partial_jR_p(q_{\rho,i})+O(\rho^p)=0$$, so $$\partial_jR_p$$ is a homogeneous collision invariant of degree $$p-1\ge2$$ on $$\mathcal M_0^\ast\cap U$$, hence $$\partial_jR_p\equiv0$$ by Lemma 4, and a homogeneous function of positive degree with zero gradient vanishes: $$R_p\equiv0$$, a contradiction. So $$R\equiv0$$ and $$\Omega^2=\lvert k\rvert^2$$ on the punctured ball.
</div>

Step 2 is where non-trivial structure of massless kinematics enters (the four null momenta have a single linear relation); Steps 3 and 4 are the same application of Lemma 4 at higher degree, routine once the lemma is available. The energy row fixes the coupling to be proportional to the energy, and the momentum rows say that the velocity $$\nabla\Omega$$ times the energy must itself be conserved, which on the lattice is the statement that velocity is parallel to momentum with the correct magnitude.

## 4. A periodic symbol has no exact cone

<div class="theorem" markdown="1">
**Theorem 2** (no exact cone). Let $$A(k)$$ be an $$N\times N$$ matrix function on $$\mathbb R^3$$, real-analytic and $$2\pi$$-periodic in each coordinate. Let $$Q$$ be a positive-definite quadratic form and $$m\ge0$$.

(a) If $$A$$ is Hermitian (a lattice Hamiltonian), no continuous eigenvalue branch $$\Omega$$ of $$A(k)$$ satisfies $$\Omega(k)^2=Q(k)+m^2$$ on a non-empty open set.

(b) If $$A$$ is unitary (a Floquet or cellular-automaton symbol), no branch with $$e^{-i\Omega(k)}\in\mathrm{spec}\,A(k)$$ satisfies $$\Omega(k)^2=Q(k)+m^2$$ on a non-empty open set.

In both cases the open set is taken not to contain $$k=0$$ when $$m=0$$.
</div>

<div class="proof" markdown="1">
*Proof.* On a small ball in the open set, $$\Omega\ne0$$ and is continuous, so $$\Omega=\sigma\sqrt{Q+m^2}$$ with a fixed sign $$\sigma$$. Put $$a(k)=\sqrt{Q(k)+m^2}$$, real-analytic on $$\mathbb R^3$$ (on $$\mathbb R^3\setminus\{0\}$$ if $$m=0$$). The idea is that a determinant identity on the ball spreads by analyticity to all of space, and periodicity then puts an unbounded sequence into a finite spectrum.

(a) The function $$h(k)=\det(\sigma a(k)-A(k))$$ is real-analytic on the connected set $$\mathbb R^3\setminus\{0\}$$ and vanishes on the ball, so $$h\equiv0$$: $$\sigma a(k)\in\mathrm{spec}\,A(k)$$ for all $$k\ne0$$. By periodicity $$\sigma a(k+2\pi ne_1)\in\mathrm{spec}\,A(k)$$ for every integer $$n$$. But $$a(k+2\pi ne_1)\to\infty$$ and the spectrum of a fixed $$N\times N$$ matrix is bounded. Contradiction.

(b) The same argument with $$h(k)=\det(e^{-i\sigma a(k)}-A(k))$$ gives $$e^{-i\sigma a(k+2\pi ne_1)}\in\mathrm{spec}\,A(k)$$ for all $$n$$, with the same $$\sigma$$ for all $$n$$ because $$h$$ was defined with it. The spectrum has at most $$N$$ points on the unit circle, so the numbers $$a_n=a(k+2\pi ne_1)$$ satisfy $$a_n\bmod2\pi\in E$$ for a finite set $$E$$, and the increments $$d_n=a_{n+1}-a_n$$ lie in the set $$D=(E-E)+2\pi\mathbb Z$$, which meets every bounded interval in a finite set. Now choose $$k$$ not parallel to $$e_1$$ and set $$\phi(t)=\sqrt{Q_{11}t^2+2Bt+C+m^2}$$ with $$Q_{11}=Q(e_1)$$, $$B=Q(e_1,k)$$ the bilinear form and $$C=Q(k)$$. Then

$$
\phi''(t)=\frac{Q_{11}(C+m^2)-B^2}{\big(Q_{11}t^2+2Bt+C+m^2\big)^{3/2}}>0,
$$

by the strict Cauchy–Schwarz inequality $$B^2<Q_{11}Q(k)$$ for non-parallel vectors. So $$\phi$$ is strictly convex, the increments $$d_n=\phi(2\pi(n+1))-\phi(2\pi n)$$ are strictly increasing, and they are bounded above by $$2\pi\sqrt{Q_{11}}$$. A strictly increasing bounded sequence takes infinitely many values in a bounded interval, but $$D$$ meets that interval in finitely many points. Contradiction.
</div>

The sign $$\sigma$$ in (b) is the point at which an argument that counts "distinct eigenvalues $$e^{\pm ia_n}$$" has a gap: for irrational $$\sqrt{Q_{11}}$$ the signs of the eigenvalues that occur could in principle vary with $$n$$. Fixing $$\sigma$$ through the determinant avoids the issue, and the convexity argument does not need the asymptotics of $$a_n$$.

*A numerical illustration.* For $$Q=\lvert k\rvert^2$$, $$m=0$$ and $$k=(0,1,0)$$, the increments are $$d_n=\sqrt{4\pi^2(n+1)^2+1}-\sqrt{4\pi^2n^2+1}$$; for $$n=1,2,3,10$$ they equal $$6.2438$$, $$6.2700$$, $$6.2766$$, $$6.2825$$, increasing towards $$2\pi=6.2832$$ as the proof requires.

<div class="corollary" markdown="1">
**Corollary 3** (no lattice or automaton graviton with exact universal soft coupling). There is no lattice Hamiltonian or Floquet graviton, with a real-analytic, periodic, finite-dimensional Bloch symbol, for which (B) and (S1)–(S4) hold.
</div>

<div class="proof" markdown="1">
*Proof.* Theorem 1 makes the band an exact cone on a punctured ball, which is a non-empty open set not containing $$0$$; in the original axes it is $$\Omega^2=Q$$ with $$Q$$ positive definite. Theorem 2 forbids this for any such symbol at any lattice spacing.
</div>

The corollary is about exact soft universality. It does not say that lattice theories cannot have gravitons, and it does not say that the soft factor of a lattice graviton fails to be universal at low energy; section 5 quantifies the second point.

*Example.* The free Sym$$^{2s}$$ field built from the Weyl-type walk $$U(k)=R_x(k_x/2)R_y(k_y/2)R_z(k_z)R_y(k_y/2)R_x(k_x/2)$$, $$R_j(t)=\cos t-i\sin t\,\sigma_j$$, with the one-step symbol $$W(k)=D^{(s)}(U(k/2s))$$ in the spin-$$s$$ representation $$D^{(s)}$$, is a finite-cell automaton whose helicity-$$s$$ band is $$\Omega_s(k)=2s\lambda(k/2s)$$, where $$U=\cos\lambda-i\sin\lambda\,\hat n\cdot\sigma$$. A series expansion by computer algebra gives $$\lambda^2=\lvert q\rvert^2-\tfrac13\sum_{i<j}q_i^2q_j^2+O(q^6)$$, so

$$
\Omega_s^2=\lvert k\rvert^2-\frac1{12s^2}\sum_{i<j}k_i^2k_j^2+O(k^6),
$$

which is not a cone for any $$s$$: at $$s=2$$ the quartic coefficient is $$-1/48$$. In the notation of section 5, $$p=4$$ and the universality order is $$\nu=2$$.

## 5. Universality to finite order

Theorem 1 suggests measuring how far a band is from an exact cone by the order at which the conserved vector of (3) fails. For a band satisfying (B) and a coupling $$\beta=\sum_{m\ge1}\beta_m$$ with $$\beta_1=a\lvert q\rvert$$, $$a\ne0$$, define the *defect* of a hard quartet $$q\in\mathcal M_\rho$$ by

$$
D_\beta(\rho,q)=\sum_i\sigma_i\,\beta(\rho q_i)\,\big(1,\nabla\Omega(\rho q_i)\big). \tag{5}
$$

It vanishes identically for the exact cone with $$\beta=a\lvert k\rvert$$. In general it is the failure of the pole term of the soft factor to be gauge invariant.

<div class="proposition" markdown="1">
**Proposition 5** (universality order). Assume (B) and let $$\Omega^2=\lvert k\rvert^2+R_p+O(\lvert k\rvert^{p+1})$$ with $$R_p\not\equiv0$$ homogeneous of degree $$p\ge3$$.

(i) For $$\beta=a\Omega$$ the energy component of $$D_\beta$$ vanishes and its momentum components are $$\tfrac a2\rho^{p-1}\sum_i\sigma_i\partial_jR_p(q_i)+O(\rho^p)$$.

(ii) For every admissible $$\beta$$ and every non-empty relatively open set $$V\subset\mathcal M_0^\ast$$, $$D_\beta$$ is not $$o(\rho^{p-1})$$ uniformly on points $$q_\rho\in\mathcal M_\rho$$ converging to the points of $$V$$.

Relative to $$\sum_i\lvert\rho q_i\rvert\propto\rho$$, the defect is therefore of order $$\rho^{p-2}$$, and cannot be made smaller: the universality order is $$\nu=p-2$$. For a separable band $$\Omega^2=\sum_if_m(k_i)$$ with the order-$$2m$$ central stencil $$f_m$$ for $$k^2$$, $$p=2m+2$$ and $$\nu=2m$$.
</div>

<div class="proof" markdown="1">
*Proof.* (i) The energy component is $$\sum\sigma_ia\Omega(\rho q_i)=0$$ on the energy shell. The momentum components are $$\tfrac a2\sum\sigma_i\partial_j\Omega^2(\rho q_i)=\tfrac a2\sum\sigma_i\partial_jR(\rho q_i)$$ by $$\sum\sigma_ik_i=0$$, and the leading term of $$\partial_jR$$ is $$\partial_jR_p$$.

(ii) Suppose $$D_\beta=o(\rho^{p-1})$$ uniformly near $$V$$. The energy component of $$D_\beta$$ equals $$\sum\sigma_ig(\rho q_i)$$ for $$g=\beta-a\Omega$$, as in Step 3 of the proof of Theorem 1. If $$g_m$$ is its lowest non-zero term with $$m\le p-1$$, then $$\sum\sigma_ig_m=0$$ on $$V$$ is forced, and Lemma 4 gives $$g_m=0$$. Hence $$\beta_m=a\Omega_m$$ for $$m\le p-1$$. Here $$\Omega_m=0$$ for $$2\le m\le p-2$$ and $$\Omega_{p-1}=R_p/(2\lvert k\rvert)$$. The degree-$$(p-1)$$ part of $$\beta\,\partial_j\Omega$$ is then $$a\lvert q\rvert\partial_j\Omega_{p-1}+a\Omega_{p-1}\hat q_j=\tfrac a2\partial_jR_p$$, all lower parts being either conserved exactly ($$a\,q_j$$) or zero. So the momentum components of $$D_\beta$$ are $$\tfrac a2\rho^{p-1}\sum\sigma_i\partial_jR_p(q_i)+O(\rho^p)$$, and $$o(\rho^{p-1})$$ would make $$\partial_jR_p$$ a collision invariant of degree $$p-1\ge2$$ on $$V$$. By Lemma 4 it vanishes, hence $$R_p=0$$, a contradiction.

For the stencils: $$f_m(k)-k^2=c_mk^{2m+2}+\dots$$ with $$c_m\ne0$$ (checked below), so $$R_{2m+2}=c_m\sum_ik_i^{2m+2}\not\equiv0$$.
</div>

*Numerical check.* For $$\Omega^2=\sum_if_m(k_i)$$ with $$f_1=2-2\cos k$$ (error $$-k^4/12$$), $$f_2=\tfrac52-\tfrac83\cos k+\tfrac16\cos2k$$ (error $$-0.0111\,k^6\approx-k^6/90$$) and $$f_3=\tfrac{49}{18}-3\cos k+\tfrac3{10}\cos2k-\tfrac1{45}\cos3k$$ (error $$-0.0018\,k^8$$), we constructed exact on-shell quartets (momentum conserved and the energy residual below $$10^{-14}$$) at scales $$\rho=0.4,0.2,0.1,0.05$$ for twelve fixed configurations, and computed the momentum-row defect $$\lvert\tfrac12\sum\sigma_i\nabla\Omega^2(k_i)\rvert/\sum\lvert k_i\rvert$$ for $$\beta=\Omega$$. Its median over the configurations has slope in $$\rho$$ of $$1.99$$, $$3.97$$ and $$5.94$$ for $$m=1,2,3$$, against the predicted $$2$$, $$4$$ and $$6$$. At $$\rho=0.05$$ the medians are $$2.4\times10^{-4}$$, $$4.4\times10^{-7}$$ and $$1.0\times10^{-9}$$.

<div class="remark" markdown="1">
**Remark** (physical reading; an estimate, not a theorem). Weinberg's theorem and its S-matrix sharpenings assume an exactly Poincaré-covariant S-matrix. If one assumes only that the S-matrix is covariant up to relative corrections of order $$(Ea)^\nu$$, the same argument gives universality up to such corrections and no more: this is the content of Proposition 5, and no infrared theorem we know of forces exact universality from an ultraviolet theory with a preferred frame. As an order of magnitude, if the relative violation of the equivalence principle for nucleon-bound matter were $$\eta\sim(p_Fa)^\nu$$ with the Fermi momentum $$p_F\approx0.25$$ GeV, then $$\lvert\eta\rvert<3\times10^{-15}$$ gives $$1/a\gtrsim4.6\times10^6$$ GeV for $$\nu=2$$, $$1.1\times10^3$$ GeV for $$\nu=4$$ and $$66$$ GeV for $$\nu=6$$. The assumption $$\eta\sim(p_Fa)^\nu$$ is a dimensional estimate; the other costs of a preferred frame, such as the per-species speed shifts of the companion page, are not included.
</div>

## 6. The Weinberg–Witten theorem and a local lattice energy density

One may ask whether the Weinberg–Witten theorem gives a second obstruction. Its hypothesis is a Lorentz-covariant, conserved, local energy-momentum tensor $$T^{\mu\nu}$$ with Poincaré charges as its integrals. For a massless particle of helicity $$h$$ the argument runs as follows. Take the momenta $$p$$ and $$-p$$ along an axis, where the two one-particle states have angular momentum $$h$$ and $$-h$$ about it. A rank-2 tensor operator carries angular momentum $$\lvert j_z\rvert\le2$$ about that axis, so $$\langle -p,h\vert T^{\mu\nu}\vert p,h\rangle=0$$ if $$\lvert2h\rvert>2$$. Boosts map any pair of non-parallel massless momenta to a back-to-back pair, so covariance gives $$\langle p',h\vert T^{\mu\nu}\vert p,h\rangle=0$$ for all non-parallel $$p,p'$$ when $$\lvert h\rvert>1$$. If the matrix element is continuous at $$p'=p$$, this contradicts the normalisation $$\langle p,h\vert T^{00}\vert p,h\rangle=\Omega(p)$$. We call the vanishing for non-parallel momenta the covariant benchmark.

<div class="proposition" markdown="1">
**Proposition 6** (a local lattice energy density is not covariant on helicity-$$s$$ states). Let the one-particle space at small wave vector be the spin-$$s$$ irreducible representation, $$s\ge\tfrac32$$, with the helicity-$$s$$ band eigenvector $$u(k)=\lvert\hat k\rangle+O(\lvert k\rvert)$$, the spin coherent state along $$\hat k$$. Let the energy density be a bilinear form with kernel $$\tau(k',k)$$, real-analytic at $$(0,0)$$ (as for an exponentially local lattice operator), and let the matrix element $$M(k',k)=u(k')^\dagger\tau(k',k)u(k)$$ on the diagonal be the band energy, $$M(k,k)=\Omega(k)=\lvert k\rvert(1+o(1))$$. Then for $$k=\rho\hat n$$, $$k'=\rho\hat n'$$ the limit of $$\lvert M(k',k)\rvert/\Omega(k)$$ as $$\rho\to0$$ exists and equals $$\lvert G(\hat n',\hat n)\rvert$$, where

$$
G(\hat n',\hat n)=\langle\hat n'\vert L(\hat n',\hat n)\vert\hat n\rangle
$$

is continuous on $$S^2\times S^2$$ ($$L$$ is the linear part of $$\tau$$ at the origin) and $$G(\hat n,\hat n)=1$$. In particular $$\lvert M(k',k)\rvert\ge\tfrac12\Omega(k)$$ for all small $$\rho$$ and all directions with $$\lvert\hat n'-\hat n\rvert<\theta_0$$, where $$\theta_0>0$$ does not depend on $$\rho$$.
</div>

<div class="proof" markdown="1">
*Proof.* Expand $$\tau(k',k)=\tau_0+L(k',k)+O(\rho^2)$$ with $$L$$ linear. On the diagonal, $$M(k,k)=\langle\hat n\vert\tau_0\vert\hat n\rangle+O(\rho)$$ must be $$O(\rho)$$, so $$\langle\hat n\vert\tau_0\vert\hat n\rangle=0$$ for every unit $$\hat n$$. The diagonal coherent-state symbol is injective on operators on the spin-$$s$$ representation (the coherent states are overcomplete, and the symbol is a polynomial in the components of $$\hat n$$ that determines the matrix), so $$\tau_0=0$$. Then $$M(k',k)=\rho\langle\hat n'\vert L(\hat n',\hat n)\vert\hat n\rangle+O(\rho^2)$$ up to the arbitrary phases of $$u(k),u(k')$$, which drop out of the modulus; $$G$$ is continuous in $$(\hat n',\hat n)$$ with $$G(\hat n,\hat n)=\lim M(k,k)/\Omega(k)=1$$.
</div>

For $$s\ge\frac32$$ and non-parallel momenta $$k,k'$$ of equal length with small angle, the covariant benchmark is $$M=0$$, and the proposition shows $$\lvert M\rvert\ge\tfrac12\Omega$$. A local lattice energy density is therefore not covariant on helicity-two states, not even approximately: the violation is of relative order one and does not decrease with the scale. The Weinberg–Witten theorem has a hypothesis that the lattice does not meet, so it excludes nothing here. This is the lattice form of the fact that a free massless spin-2 field has no covariant conserved rank-2 energy-momentum tensor (the covariant conserved bilinear is the Bel–Robinson tensor, of rank four).

*An explicit family.* For a one-particle generator $$h(k)=\Omega(k)\,\hat n(k)\cdot S$$ in the spin-$$s$$ representation, with the helicity-$$s$$ eigenvector $$u(k)=\lvert\hat n(k)\rangle$$ and the energy density $$\tau^{00}(k',k)=\tfrac12[f(h(k'))+f(h(k))]$$ for a function $$f$$, the matrix element is exactly

$$
\lvert M^{00}(k',k)\rvert=\tfrac12\lvert E(k)+E(k')\rvert\Big(\frac{1+\hat n(k)\cdot\hat n(k')}2\Big)^{s},\qquad E=f(\Omega), \tag{6}
$$

because $$u(k)$$ is an eigenvector of $$f(h(k))$$ with eigenvalue $$f(\Omega(k))$$, and the overlap of two spin-$$s$$ coherent states is $$\lvert\langle\hat n'\vert\hat n\rangle\rvert=((1+\hat n\cdot\hat n')/2)^s=\cos^{2s}(\theta/2)$$. We checked the overlap formula for $$s=\tfrac12,1,\tfrac32,2$$ on 200 random pairs of directions each: the largest deviation is $$9\times10^{-16}$$. For $$s=2$$ and equal energies, (6) gives $$\lvert M^{00}\rvert/E=\cos^4(\theta/2)=0.8705,\ 0.5625,\ 0.25,\ 0.0625$$ at $$\theta=\pi/6,\pi/3,\pi/2,2\pi/3$$, independent of $$\rho$$. For $$s=1$$ it gives $$\cos^2(\theta/2)$$, the overlap of two helicity-one polarisation vectors; no contradiction arises there, because the photon has a covariant conserved energy-momentum tensor, Maxwell's, and the argument above does not require the matrix element to vanish for $$\lvert h\rvert\le1$$. For $$s=\tfrac12$$ it gives $$\cos(\theta/2)$$. Whether such a non-covariant energy density can be the source to which the graviton couples universally, as the gravitational pseudotensor is in general relativity, is not decided here.

## 7. What the theorems do not exclude

Theorem 1 and Corollary 3 are statements about exact universality of the soft factor. Each escape below drops a hypothesis. They are listed without a choice among them.

E1. *Vanishing soft coupling.* If $$\beta\equiv0$$ there is no leading soft charge and no Newtonian limit. All cubic vertices built from the linearised curvature are of this kind: they are invariant under the linear gauge transformations by themselves, can be finite-range, and carry no coupling to energy.

E2. *Explicit gauge breaking.* (S2) fails. The band-limited vertices of the companion page break the nonlinear gauge invariance by the heuristic criterion stated there, and are an instance.

E3. *No asymptotic poles.* (S1) fails: the hard legs do not have simple poles $$1/(\mathsf s_i\cdot q)$$. For interacting cellular automata, the existence of such poles in discrete time is not established here.

E4. *Unbounded or infinite-dimensional fibres.* Theorem 2 uses a finite-dimensional fibre. A continuum free massless field regarded as a lattice theory with one unit cell has all bands exactly conical, $$\Omega=\lvert k+2\pi n\rvert$$, and an unbounded fibre. Whether bounded infinite-dimensional Hamiltonian fibres and unitary ones escape is not settled here.

E5. *Euclidean-time non-locality.* In a Euclidean formulation the on-shell variety is the complex null cone. A periodic kinetic symbol that is exactly zero on the periodised cone near the origin is not excluded by Theorem 2, which is about Hamiltonian and Floquet fibres with finite matrices. If the lattice momentum charge $$f$$ of such a theory is analytic on a strip of imaginary height above $$\pi/\sqrt3$$ and equals the momentum on the cone, periodicity gives a contradiction, by the following elementary argument. A point $$p$$ and its translate $$p-2\pi e_1$$ are both on the cone $$\sum_\mu p_\mu^2=0$$ only if $$p_1=\pi$$ and $$p_2^2+p_3^2+p_4^2=-\pi^2$$; writing $$p_j=x_j+iy_j$$ gives $$\sum y_j^2\ge\pi^2$$, so $$\max\lvert y_j\rvert\ge\pi/\sqrt3\approx1.814$$, with equality at $$p=(\pi,i\pi/\sqrt3,i\pi/\sqrt3,i\pi/\sqrt3)$$. Then $$f(p)=p$$ and $$f(p-2\pi e_1)=p-2\pi e_1$$ contradict $$f(p)=f(p-2\pi e_1)$$. The equality $$f=k$$ on the cone is the Euclidean analogue of Theorem 1 and was not derived here; its derivation needs the soft identity at complex momenta. Charges analytic only in strips of smaller height are not covered.

E6. *Non-analytic symbols.* Theorem 2 needs analyticity. A scalar fibre $$A(k)=\lvert k\rvert\,g(\lvert k\rvert)$$ with $$g$$ smooth, $$g=1$$ for $$\lvert k\rvert\le1$$ and $$g=0$$ for $$\lvert k\rvert\ge2$$, extended periodically, is $$C^\infty$$ away from the origin, has an exact cone near $$0$$ and hopping that decays faster than any power but not exponentially. This is a toy band, not a graviton; it shows that analyticity cannot be dropped from Theorem 2. Whether crystal-momentum periodicity alone excludes an exact cone for $$C^\infty$$ symbols with non-vanishing hard scattering is not decided here.

E7. *Matrix-valued coupling.* (S1) assumes that $$J$$ acts as a multiple of the identity on the helicity index. A helicity- or species-mixing matrix $$\beta$$ is not covered by Lemma A.

Spin one is not an instance of any of these: there the soft charge is an internal scalar, and the constant solution of the corresponding condition ignores the dispersion.

## 8. Open points

The hypotheses (S1)–(S3) are assumptions on the soft expansion of lattice amplitudes, not consequences of a lattice action; deriving them for an interacting lattice or automaton graviton, including simple poles in discrete time, is open. The jet-level argument of Theorem 1 uses analyticity of the homogeneous pieces of $$\Omega$$ and $$\beta$$; bands with non-analytic dependence on the direction, which can occur when helicity bands are degenerate at leading order, are not covered. The order $$\nu$$ of Proposition 5 measures the failure of an algebraic condition; how it translates into observable violations of the equivalence principle depends on the dynamics, and the estimate in the remark is dimensional only. Whether the non-covariant energy density of section 6 can saturate a cubic graviton vertex, that is, whether a local cubic vertex exists whose soft limit is the coupling to that energy density with a defect of order $$\rho^3$$, has not been tested.

## References

1. <a id="ref-w64"></a>Weinberg, “Photons and gravitons in S-matrix theory: derivation of charge conservation and equality of gravitational and inertial mass,” Phys. Rev. 135, B1049 (1964).
2. <a id="ref-w65"></a>Weinberg, “Infrared photons and gravitons,” Phys. Rev. 140, B516 (1965).
3. <a id="ref-ww"></a>Weinberg, Witten, “Limits on massless particles,” Phys. Lett. B 96, 59 (1980).
4. <a id="ref-porrati"></a>Porrati, “Universal limits on massless high-spin particles,” Phys. Rev. D 78, 065016 (2008), [arXiv:0804.4672](https://arxiv.org/abs/0804.4672).
5. <a id="ref-bc"></a>Benincasa, Cachazo, “Consistency conditions on the S-matrix of massless particles,” [arXiv:0705.4305](https://arxiv.org/abs/0705.4305).
6. <a id="ref-kss"></a>Kato, Sakamoto, So, “No-go theorem of Leibniz rule and supersymmetry on the lattice,” PoS LATTICE2008, 233, [arXiv:0810.2360](https://arxiv.org/abs/0810.2360).
