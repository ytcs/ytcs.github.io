---
title: Lattice gravitons, exact diffeomorphisms and lattice momentum
date: 2026-10-28
order: 4
description: In a Hamiltonian lattice theory of gravitons, a conserved lattice momentum built from exponentially local terms is incompatible with graviton scattering at low energy, which excludes exact deformed diffeomorphism invariance if such a symmetry supplies that charge. Switching off interactions near the zone boundary conserves lattice momentum exactly, but by a heuristic closure criterion is not preserved by the nonlinear gauge transformations, and at one loop it shifts the speed of light by a few parts in ten thousand.
---

We ask whether an interacting theory of gravitons on a spatial lattice can keep an exact, conserved lattice momentum built from exponentially local terms, as an exact deformed diffeomorphism symmetry would require. Such a charge assigns to each graviton of wavevector $$k$$ a momentum $$P(k)$$, its symbol, which is periodic on the Brillouin zone and, for exponentially local terms, real-analytic. The answer is no, if $$P$$ is the same for both helicities and gravitons scatter at low energy (Theorem 1): conservation in low-energy $$2\to2$$ scattering forces $$P(k)=k$$, which is not periodic. The setting is a Hamiltonian theory in continuous time on a fixed cubic lattice, treated order by order in the gravitational coupling, with a free dispersion that has no three-wave resonances, as the nearest-neighbour graviton has. That an exact deformed diffeomorphism symmetry with exponentially local generators supplies such a charge is assumed, not derived; under that assumption the symmetry is excluded.

The theorem does not exclude non-analytic charges. A process whose total momentum leaves the Brillouin zone and is folded back into it is called Umklapp. If every vertex is switched off when one of its legs is near the zone boundary, no process is Umklapp, and a charge equal to $$k$$ near the origin is conserved exactly, as Peierls observed for phonons. This costs Lorentz invariance and, by a heuristic closure criterion (section 5), gauge invariance: the band limit is not preserved by gauge transformations whose leading nonlinear part moves the field along the gauge parameter $$\xi$$, and no band is invariant under the Lorentz group of any metric. In a one-loop model of band-limited scalar electrodynamics the speed of light shifts by $$\delta c_\gamma=(1.8\text{–}7.8)\times10^{-4}$$, independently of the band edge and differently for each species, and the photon acquires a positive mass of 2–4% of the band edge.

Lattice versions of linearised gravity with exact linear gauge modes are known, for example linearised Regge calculus [[Dittrich–Freidel–Speziale]](#ref-dfs) [[Höhn]](#ref-hoehn); beyond linear order discretisation generically breaks diffeomorphism symmetry, which can be restored only by non-local "perfect" actions [[Bahr–Dittrich 2009a]](#ref-bd1) [[Bahr–Dittrich 2009b]](#ref-bd2) [[Dittrich]](#ref-dittrich). For fermions, the Ginsparg–Wilson relation [[Ginsparg–Wilson]](#ref-gw) gives an exact deformed lattice chiral symmetry [[Lüscher]](#ref-luscher), and its Dirac operators can be exponentially local but not ultralocal [[Horváth]](#ref-horvath) [[Bietenholz]](#ref-bietenholz); the question here is whether gravity has an analogue. For supersymmetry, the failure of the Leibniz rule for lattice derivatives obstructs exact symmetry with local operators [[Kato–Sakamoto–So 2008a]](#ref-kss1) [[Kato–Sakamoto–So 2008b]](#ref-kss2). In the continuum, consistent deformations of linearised diffeomorphism invariance lead to general relativity [[Deser]](#ref-deser) [[Barnich–Henneaux]](#ref-bh). Spohn characterised the collision invariants of the phonon Boltzmann equation with pair collisions over the whole zone [[Spohn]](#ref-spohn). That crystal momentum is a true conserved momentum in the absence of Umklapp processes goes back to Peierls [[Peierls]](#ref-peierls), and Lewis, Kempf and Menicucci showed that bandlimited lattice models carry conserved observables for continuous translations [[LKM]](#ref-lkm); the construction of section 4 is theirs in substance. Collins et al. showed that a preferred-frame cutoff produces Lorentz violation of order $$g^2/16\pi^2$$ that does not decrease as the cutoff is raised [[Collins et al.]](#ref-collins), and Chankowski, Lewandowski and Meissner that a physical momentum cutoff gives the photon a mass [[CLM]](#ref-clm). Compact lattice tensor gauge theories in Villain form have soft, $$\omega\propto k^2$$ gravitons [[Xu]](#ref-xu) [[Xu–Hořava]](#ref-xh) [[GLSS 2021]](#ref-glss1) [[GLSS 2022]](#ref-glss2).

The results that are new here are Theorem 1, which uses only low-energy data and the identity theorem for analytic functions, and the one-loop light-cone shifts and photon mass produced by band-limited vertices. Along the way we show that no band is Lorentz invariant, that no proper band is closed under addition, and that in Villain lattices integer charges force $$\omega\propto k^2$$ (Theorem 2).

Familiarity with the canonical (ADM) form of linearised gravity and with lattice Fourier analysis is assumed. The lattice spacing is 1 and the low-energy speed of light is 1.

## 1. Free lattice gravitons with exact linear gauge invariance

The linearised ADM theory has a symmetric tensor $$h_{ij}$$ with conjugate momentum $$\varpi^{ij}$$, the gauge transformation $$\delta h_{ij}=\partial_i\xi_j+\partial_j\xi_i$$, the linearised momentum and Hamiltonian constraints, and a Hamiltonian quadratic in $$h$$ and $$\varpi$$. Every operator in it is a polynomial in the commuting derivatives $$\partial_i$$, and the identities that make it consistent, such as the vanishing of the linearised curvature of a pure-gauge $$h$$, are polynomial identities in commuting variables.

To put it on the cubic lattice $$\mathbb Z^3$$ we place the diagonal components $$h_{ii}$$ on sites, the off-diagonal ones on plaquette centres and $$\xi_i$$ on links, so that every lattice difference $$d_i$$ that occurs is centred and has Fourier symbol $$i\,v_i(k)$$ with

$$
v_i(k)=2\sin(k_i/2). \tag{1}
$$

Replacing each $$\partial_i$$ by $$d_i$$ gives a lattice theory whose Fourier transform at wavevector $$k$$ is the continuum theory at the real wavevector $$v(k)$$. All the identities therefore hold exactly on the lattice, the linear gauge invariance is exact, and at each $$k\neq0$$ there are two physical modes, transverse and traceless with respect to $$v(k)$$, with the common frequency

$$
\omega(k)=\lvert v(k)\rvert=2\Big(\sum_i\sin^2(k_i/2)\Big)^{1/2}. \tag{2}
$$

Since $$v(k)$$ vanishes in the zone $$(-\pi,\pi]^3$$ only at $$k=0$$, there are no doublers, and $$\omega(k)=\lvert k\rvert\,(1+O(k^2))$$ at small $$k$$. We call this the nearest-neighbour lattice graviton. Whether its gauge group can be made compact, with integer charges, is a separate question, taken up in section 6. The question here is what happens when interactions are added.

## 2. From an exact symmetry to a conserved lattice momentum

Write the physical graviton modes in terms of complex amplitudes $$a_\sigma(k)$$, with helicity $$\sigma=\pm$$, $$k$$ in the zone $$T^3=(\mathbb R/2\pi\mathbb Z)^3$$ and Poisson bracket $$\{a_\sigma(k),a^*_{\sigma'}(k')\}=-i\delta_{\sigma\sigma'}\delta(k-k')$$. We work with the classical field theory, which is the tree-level content of the quantum one, and expand the Hamiltonian in the gravitational coupling $$\kappa$$,

$$
H=H_2+\kappa H_3+\kappa^2H_4+\dots,\qquad H_2=\sum_\sigma\int\omega(k)\,\lvert a_\sigma(k)\rvert^2\,dk,
$$

where $$H_r$$ is a polynomial of degree $$r$$ in the $$a$$ and $$a^*$$. Lattice translation invariance means that each monomial in $$H_r$$ conserves momentum modulo $$2\pi$$. A $$2\to2$$ process $$k_1,k_2\to k_3,k_4$$ is normal if $$k_1+k_2-k_3-k_4=0$$ and Umklapp if this difference is a non-zero multiple of $$2\pi$$.

The hypotheses below are motivated by the gauge generators. In the canonical theory the deformed momentum constraint $$D[\xi]=\langle\varpi,S\xi\rangle+\kappa D_3[\xi]+\dots$$ generates the gauge transformations, where $$(S\xi)_{ij}=d_i\xi_j+d_j\xi_i$$ is the lattice symmetrised gradient, and the Hamiltonian is invariant under all of them. For a constant gauge parameter $$\xi$$ the linear term vanishes, because a lattice difference of a constant is zero, and $$D[\xi]/\kappa$$ starts at quadratic order. Its quadratic part generates the leading nonlinear term of the deformed transformation, the transport term $$\xi\cdot\partial h$$, which for constant $$\xi$$ translates the long-wavelength fields. Its symbol is periodic because the lattice is. If the generators $$D[\xi]$$ are exponentially local, the Fourier coefficients of this symbol decay exponentially, and a periodic function has exponentially decaying Fourier coefficients exactly when it is real-analytic.

This motivation is not a derivation. $$D[\xi]$$ with constant $$\xi$$ is itself a constraint, so it vanishes on physical states, and a conserved graviton momentum appears only after the constraints are solved, with the total momentum carried by a boundary term as in the ADM construction; on the lattice that identification needs an argument. That the resulting charge acts on the physical modes diagonally, with symbol $$k+O(k^2)$$ and the same for both helicities, is a further assumption, and the continuity of its higher terms must be inherited from the generators. We therefore state the hypotheses directly in terms of the charge.

(H1) The dispersion $$\omega$$ is continuous on $$T^3$$, positive away from $$k=0$$, the same for both helicities, and relativistic at low energy: $$\omega(\lambda q)/\lambda\to\lvert q\rvert$$ as $$\lambda\to0$$, in $$C^1$$ on compact sets of $$q\neq0$$.

(H2) There are no three-wave resonances: $$\omega(k_1)+\omega(k_2)>\omega(k_1+k_2)$$ whenever $$k_1,k_2\neq0$$.

(H3) There is a charge $$Q=Q_2+\kappa Q_3+\kappa^2Q_4+\dots$$ with $$\{Q,H\}=0$$ order by order in $$\kappa$$, whose quadratic part is
$$Q_2=\sum_\sigma\int P(k)\,\lvert a_\sigma(k)\rvert^2\,dk$$ with $$P:T^3\to\mathbb R^3$$ continuous and $$P(k)=k+O(k^2)$$, and whose higher terms $$Q_3,Q_4$$ have coefficient functions that are continuous away from zero momenta (as they are if $$Q$$ is built from exponentially local terms). The coefficient functions of the vertices $$H_3,H_4$$ are likewise continuous, as they are for exponentially local vertices.

(H4) $$P$$ is real-analytic on $$T^3$$.

(H5) The low-energy $$2\to2$$ graviton interaction is non-zero on an open set of kinematics; the precise form is given after (4).

The nearest-neighbour graviton satisfies (H1) and (H2).

<div class="lemma" markdown="1">
**Lemma 1.** The dispersion (2) satisfies (H2).
</div>

<div class="proof" markdown="1">
*Proof.* For real $$\phi,\psi$$, $$\lvert\sin\tfrac{\phi+\psi}2\rvert\le\lvert\sin\tfrac\phi2\rvert\lvert\cos\tfrac\psi2\rvert+\lvert\cos\tfrac\phi2\rvert\lvert\sin\tfrac\psi2\rvert\le\lvert\sin\tfrac\phi2\rvert+\lvert\sin\tfrac\psi2\rvert$$, with equality only if $$\phi$$ or $$\psi$$ lies in $$2\pi\mathbb Z$$. Applied componentwise, the vector $$\lvert v(k_1+k_2)\rvert$$ (components in absolute value) is dominated by $$\lvert v(k_1)\rvert+\lvert v(k_2)\rvert$$, so by the triangle inequality $$\omega(k_1+k_2)\le\omega(k_1)+\omega(k_2)$$. Equality needs equality in every component, so that in each direction $$j$$ one of $$v_j(k_1),v_j(k_2)$$ vanishes, and equality in the triangle inequality, so that the vectors $$(\lvert v_j(k_1)\rvert)_j$$ and $$(\lvert v_j(k_2)\rvert)_j$$ are parallel. Two parallel non-negative vectors with disjoint supports cannot both be non-zero, so $$k_1=0$$ or $$k_2=0$$.
</div>

With (H2), the cubic part of $$H$$ can be removed by a canonical change of variables, the first step of Birkhoff normal form, which leaves an effective quartic interaction. For a monomial $$w$$ in the amplitudes, let $$\Omega(w)$$ and $$\mathcal P(w)$$ be the sums of $$\omega(k)$$ and of $$P(k)$$ over its factors, with sign $$+$$ for each $$a$$ and $$-$$ for each $$a^*$$. Then

$$
\{w,H_2\}=-i\,\Omega(w)\,w,\qquad \{w,Q_2\}=-i\,\mathcal P(w)\,w. \tag{3}
$$

By (H2) and positivity of $$\omega$$, $$\Omega(w)\neq0$$ for every cubic monomial whose momenta are non-zero. Let $$\Gamma_3$$ be the cubic functional whose coefficient on each such monomial is that of $$H_3$$ divided by $$-i\Omega(w)$$, so that $$\{H_2,\Gamma_3\}=-H_3$$ by (3), and apply the canonical transformation generated by $$\kappa \Gamma_3$$. Its Lie series $$H\mapsto H+\kappa\{H,\Gamma_3\}+\tfrac12\kappa^2\{\{H,\Gamma_3\},\Gamma_3\}+\dots$$ maps $$H$$ to $$H'=H_2+\kappa^2H_4'+O(\kappa^3)$$ with $$H_4'=H_4+\tfrac12\{H_3,\Gamma_3\}$$; the factor $$\tfrac12$$ is the net of the first-order term $$\{H_3,\Gamma_3\}$$ and the second-order term $$\tfrac12\{\{H_2,\Gamma_3\},\Gamma_3\}=-\tfrac12\{H_3,\Gamma_3\}$$, and maps $$Q$$ to $$Q'=Q_2+\kappa Q_3'+\kappa^2Q_4'+\dots$$ with the same $$Q_2$$. On the resonant monomials $$a_1^*a_2^*a_3a_4$$ with $$\omega_1+\omega_2=\omega_3+\omega_4$$, the coefficient $$\mathcal T(1,2;3,4)$$ of $$H_4'$$ is the effective $$2\to2$$ interaction, which includes graviton exchange; at tree level it is the $$2\to2$$ scattering amplitude up to normalisation factors that do not vanish. Exact conservation of $$Q$$ constrains $$P$$ through the condition

$$
\mathcal T(1,2;3,4)\,\big(P(k_1)+P(k_2)-P(k_3)-P(k_4)\big)=0. \tag{4}
$$

<div class="lemma" markdown="1">
**Lemma 2** (resonance condition). Under (H1)–(H3), (4) holds on every resonant quartet, normal or Umklapp, at which all $$k_i\ne0$$ and no exchanged momentum $$k_1-k_3$$, $$k_1-k_4$$ or $$k_1+k_2$$ vanishes.
</div>

<div class="proof" markdown="1">
*Proof.* Canonical transformations preserve brackets, so $$\{Q',H'\}=0$$ order by order. At order $$\kappa$$ this reads $$\{Q_3',H_2\}=0$$, so by (3) every coefficient of $$Q_3'$$ is multiplied by $$\Omega(w)\neq0$$ and vanishes: $$Q_3'=0$$. At order $$\kappa^2$$ it reads $$\{Q_4',H_2\}+\{Q_2,H_4'\}=0$$, where $$Q_4'=Q_4+\{Q_3,\Gamma_3\}+\tfrac12\{\{Q_2,\Gamma_3\},\Gamma_3\}$$. By (H3) and (H2), which makes the denominators $$\Omega$$ of $$\Gamma_3$$ non-zero whenever the momenta are non-zero, the coefficients of $$\Gamma_3$$, $$H_4'$$ and $$Q_4'$$ are continuous away from vanishing internal momenta, so the following identity between coefficients holds pointwise and may be evaluated where $$\Omega=0$$. (Without continuity, a distributional coefficient proportional to $$\mathcal P\,\mathcal T/\Omega$$ would solve it for any $$P$$.) By (3) the coefficient of a quartic monomial $$w$$ in this sum is $$-i\Omega(w)$$ times its coefficient in $$Q_4'$$ plus $$i\mathcal P(w)$$ times its coefficient in $$H_4'$$. On a resonant monomial $$\Omega(w)=0$$, which leaves $$\mathcal P(w)\mathcal T=0$$, and $$\mathcal P(a_1^*a_2^*a_3a_4)=P_3+P_4-P_1-P_2$$. At a resonant quartet with non-zero exchanged momenta the internal momenta of the contractions in $$\{H_3,\Gamma_3\}$$ and $$\{Q_3,\Gamma_3\}$$ are non-zero, so the coefficients are finite there. The helicity labels do not enter because $$P$$ is the same for both.
</div>

The higher corrections $$Q_3,Q_4,\dots$$ drop out of (4); they enter only through the continuity of their coefficients. To state (H5) precisely, let $$\Lambda=\{q\in\mathbb R^{12}:q_1+q_2=q_3+q_4\}$$, let

$$
\mathcal M_\lambda=\{q\in\Lambda:\ \omega(\lambda q_1)+\omega(\lambda q_2)=\omega(\lambda q_3)+\omega(\lambda q_4)\}
$$

be the rescaled resonant set of normal quartets, and let $$\mathcal M_0=\{q\in\Lambda:\ \lvert q_1\rvert+\lvert q_2\rvert=\lvert q_3\rvert+\lvert q_4\rvert\}$$ be its continuum limit.

(H5, precise form) There are an open set $$U\subset\mathbb R^{12}$$ and $$\lambda_0>0$$ such that $$U$$ meets $$\mathcal M_0$$ at a point where all $$q_i\neq0$$ and $$q_1,q_2$$ are not parallel, and $$\mathcal T(\lambda q)\neq0$$ for all $$q\in\mathcal M_\lambda\cap U$$ and $$0<\lambda<\lambda_0$$.

For gravity this holds if the low-energy limit is general relativity, whose tree-level $$2\to2$$ graviton amplitude does not vanish on an open set of physical kinematics (with all momenta outgoing, the helicity configuration $$(-,-,+,+)$$ has modulus proportional to $$\kappa^2\mathsf s^3/(\mathsf t\mathsf u)$$ in terms of the Mandelstam invariants $$\mathsf s,\mathsf t,\mathsf u$$; for the physical process the incoming pair then has equal helicities). We assume this identification and do not prove it for a specific lattice vertex.

## 3. Analytic charges are excluded

The idea is that (4) at low energy only sees normal quartets, on which $$P(k)=k$$ satisfies it exactly. Any analytic correction to $$k$$ is a sum of homogeneous polynomials. The lowest one would have to be a collision invariant of massless $$2\to2$$ kinematics, that is, a function $$f$$ with $$f(q_1)+f(q_2)=f(q_3)+f(q_4)$$ on every allowed collision, and no homogeneous polynomial of degree two or more is one. So $$P$$ equals $$k$$ near $$0$$, hence everywhere by analyticity, which is incompatible with periodicity.

<div class="theorem" markdown="1">
**Theorem 1** (no exponentially local exact translation charge). Hypotheses (H1)–(H5) cannot hold together. Equivalently, if (H1)–(H4) hold, the effective $$2\to2$$ interaction of low-energy gravitons vanishes in the sense that (H5) fails.
</div>

<div class="proof" markdown="1">
*Proof.* Work with one component of $$P$$ at a time and write $$P(k)=k+R(k)$$, with $$R$$ real-analytic near $$0$$ and $$R=\sum_{p\ge2}R_p$$ its expansion in homogeneous polynomials. Suppose $$R\not\equiv0$$ near $$0$$ and let $$p_0\ge2$$ be the least $$p$$ with $$R_p\neq0$$.

*Step 1 (only normal quartets at low energy).* If every component of every $$k_i$$ lies in $$(-\pi/2,\pi/2)$$, then $$k_1+k_2-k_3-k_4$$ has components in $$(-2\pi,2\pi)$$ and is a multiple of $$2\pi$$, so it vanishes. For small $$\lambda$$ the quartets $$\lambda q$$ with $$q$$ in a bounded set are therefore normal, and $$\Delta k=0$$ there. Writing $$\Delta f=f(k_1)+f(k_2)-f(k_3)-f(k_4)$$, we get $$\Delta P(\lambda q)=\Delta R(\lambda q)=\lambda^{p_0}\,\Delta R_{p_0}(q)+O(\lambda^{p_0+1})$$.

*Step 2 (the continuum resonant set is a connected analytic manifold).* On $$\Lambda$$ set $$F_0(q)=\lvert q_1\rvert+\lvert q_2\rvert-\lvert q_3\rvert-\lvert q_4\rvert$$, so that $$\mathcal M_0=\{F_0=0\}$$. Let $$\mathcal M_0^\ast$$ be the part of $$\mathcal M_0$$ with $$E\equiv\lvert q_1\rvert+\lvert q_2\rvert>\lvert q_1+q_2\rvert$$. Given $$(q_1,q_2)$$ with this property, set $$K=q_1+q_2$$; the allowed $$q_3$$ form the spheroid $$\lvert q_3\rvert+\lvert K-q_3\rvert=E$$ with foci $$0$$ and $$K$$, a 2-sphere that avoids its foci, so $$q_3,q_4\neq0$$. The base, the set of pairs $$(q_1,q_2)\in\mathbb R^6$$ that are not positively parallel and not zero, is the complement of a closed set contained in finitely many submanifolds of dimension at most 4, hence connected. A continuous surjection from (base)$$\times S^2$$ onto $$\mathcal M_0^\ast$$ shows that $$\mathcal M_0^\ast$$ is connected. On $$\mathcal M_0^\ast$$ all $$q_i\neq0$$, so $$F_0$$ is real-analytic, and its gradient in $$q_3$$ (with $$q_4=K-q_3$$) is $$-\hat q_3+\hat q_4$$, which vanishes only if $$q_3,q_4$$ point the same way; then $$\lvert q_3\rvert+\lvert q_4\rvert=\lvert K\rvert<E$$, which is excluded. So $$\mathcal M_0^\ast$$ is a connected real-analytic manifold of dimension 8 on which the gradient of $$F_0$$ along $$\Lambda$$ never vanishes. The point provided by (H5) lies on it.

*Step 3 (limit to the continuum resonant set).* Set $$F_\lambda(q)=\lambda^{-1}\big[\omega(\lambda q_1)+\omega(\lambda q_2)-\omega(\lambda q_3)-\omega(\lambda q_4)\big]$$ on $$\Lambda$$, so that $$\mathcal M_\lambda=\{F_\lambda=0\}$$. By (H1), $$F_\lambda\to F_0$$ in $$C^1$$ near any point where all $$q_i\neq0$$. Since the gradient of $$F_0$$ is non-zero at the point of (H5) (Step 2), the implicit function theorem shows that every $$q\in\mathcal M_0$$ near it is a limit of points $$q_\lambda\in\mathcal M_\lambda\cap U$$. By Lemma 2 (after shrinking $$U$$ to avoid the closed set of lower dimension on which an exchanged momentum vanishes, so that $$U\cap\mathcal M_0^\ast$$ remains a non-empty open set) and (H5), $$\Delta P(\lambda q_\lambda)=0$$, so Step 1 gives $$\Delta R_{p_0}(q_\lambda)=O(\lambda)$$ and, in the limit, $$\Delta R_{p_0}(q)=0$$. Thus the polynomial $$\Delta R_{p_0}$$ vanishes on a non-empty open subset of $$\mathcal M_0^\ast$$, hence on all of it by the identity theorem, and by continuity on its closure.

*Step 4 (a collinear quartet).* For any vector $$z$$, the quartet $$(3z,z)\to(2z,2z)$$ is in the closure of $$\mathcal M_0^\ast$$: perturb $$q_2=z$$ to $$z+\varepsilon z_\perp$$ with $$z_\perp\perp z$$; the spheroid then collapses onto the segment from $$0$$ to $$4z$$ as $$\varepsilon\to0$$, and contains points converging to $$2z$$. There, by homogeneity,

$$
\Delta R_{p_0}=\big(3^{p_0}+1-2^{p_0+1}\big)R_{p_0}(z).
$$

The bracket is positive for $$p_0\ge2$$ by strict convexity of $$x\mapsto x^{p_0}$$ ($$2,12,50,\dots$$). So $$R_{p_0}(z)=0$$ for every $$z$$, contradicting the choice of $$p_0$$.

*Step 5 (periodicity).* Hence $$R\equiv0$$ near $$0$$, that is, $$P(k)=k$$ near $$0$$. Lift $$P$$ to a real-analytic periodic function on $$\mathbb R^3$$. Then $$P(k)-k$$ is real-analytic on the connected set $$\mathbb R^3$$ and vanishes on an open set, so $$P(k)=k$$ everywhere, and $$P(k+2\pi\hat n)=P(k)+2\pi\hat n\neq P(k)$$ for $$\hat n=(1,0,0)$$, contradicting periodicity.
</div>

As a consistency check we evaluated (4) for the nearest-neighbour dispersion and the analytic charge $$P=\sin k$$ on numerically constructed resonant quartets $$\lambda q$$: for fixed $$q$$, $$\lvert\Delta P(\lambda q)\rvert/\lambda^3$$ tends to a non-zero constant as $$\lambda\to0$$, the $$\lambda^3$$ scaling that Steps 1 and 4 predict for $$p_0=3$$.

<div class="remark" markdown="1">
**Remark** (where the hypotheses enter). Steps 1–4 need only smoothness of $$P$$, with $$R_p$$ read as the Taylor terms of $$R$$; for a smooth $$P$$ they show that $$R$$ has a vanishing Taylor series at $$0$$. Analyticity enters in Step 5, twice: a vanishing Taylor series forces $$R\equiv0$$ near $$0$$, and a function equal to $$k$$ on an open set equals $$k$$ everywhere. A smooth $$P$$ that equals $$k$$ on a neighbourhood of $$0$$ passes Steps 1–4, so the theorem says nothing against smooth non-analytic charges; section 4 shows that they can be conserved. Both uses of analyticity remain valid in any quasi-analytic class in the sense of Denjoy and Carleman (a class of smooth functions in which a function with vanishing Taylor series at a point vanishes identically), so the argument extends to charges whose Fourier coefficients decay somewhat more slowly than exponentially; we do not make the class precise. Only low-energy normal quartets are used, so Umklapp processes play no role.
</div>

<div class="remark" markdown="1">
**Remark** (the vertex side). A second route uses the vertex instead of the charge. Spohn [[Spohn]](#ref-spohn) proved that, in at least two dimensions, for a continuous periodic dispersion relation that is twice differentiable away from a set of codimension at least one and whose Hessian is non-degenerate away from a set of codimension at least one, every measurable periodic function $$f$$, integrable on a unit cell, with $$f(k_1)+f(k_2)=f(k_3)+f(k_4)$$ almost everywhere on the full resonant manifold, Umklapp quartets included, has the form $$c_1\,\omega+c_0$$ with constants $$c_0,c_1$$. If $$\mathcal T\neq0$$ almost everywhere on that manifold, (4) makes each component of $$P$$ such an invariant, and $$c_1\,\omega+c_0$$ is even in $$k$$ at small $$k$$ (it behaves as $$c_0+c_1\lvert k\rvert$$), so it cannot equal $$k+O(k^2)$$. This route needs no analyticity of $$P$$, but needs $$\mathcal T$$ non-zero almost everywhere on the whole zone, which would follow from analyticity of $$\mathcal T$$ together with connectedness of the lattice resonant manifold; that connectedness is not proved here.
</div>

## 4. Band-limited vertices restore an exact momentum

Theorem 1 excludes analytic charges only. For a non-analytic $$P$$, (4) requires $$\mathcal T$$ to vanish on every resonant quartet with $$\Delta P\neq0$$: on Umklapp quartets, and on normal quartets outside the region where $$P$$ is linear. Whether this can be met other than by switching vertices off remains open. One way to meet it is to combine a non-analytic charge with vertices that never produce Umklapp processes. This is Peierls' observation [[Peierls]](#ref-peierls) that, without Umklapp, crystal momentum is a conserved momentum; Lewis, Kempf and Menicucci make the corresponding point for bandlimited fields [[LKM]](#ref-lkm). In the present language it reads as follows.

<div class="proposition" markdown="1">
**Proposition 1** (no Umklapp; Peierls). Suppose that for each $$r\ge3$$ the coefficients of $$H_r$$ vanish unless every momentum $$k_i$$ of the monomial has all components in $$(-2\pi/r,2\pi/r)$$. Let $$P$$ be any periodic function, with values in $$\mathbb R^3$$, such that $$P(k)=k$$ when all components of $$k$$ lie in $$(-2\pi/3,2\pi/3)$$. Then $$\{Q_2,H_r\}=0$$ for every $$r$$, so $$Q_2$$ is conserved exactly, to all orders in $$\kappa$$.
</div>

<div class="proof" markdown="1">
*Proof.* By (3), the coefficient of a monomial $$w$$ in $$\{Q_2,H_r\}$$ is $$i\mathcal P(w)$$ times its coefficient in $$H_r$$. On the support of $$H_r$$ the signed sum of the $$r$$ momenta has components in $$(-2\pi,2\pi)$$ and is a multiple of $$2\pi$$, so it vanishes, and $$\mathcal P(w)$$ is that signed sum because $$P(k)=k$$ there.
</div>

The free dispersion does not enter, so the nearest-neighbour $$H_2$$ can be used. Nor does the smoothness class of $$P$$ matter, provided it is not analytic. A convenient class is Gevrey class $$s$$: smooth functions whose $$n$$-th derivatives are bounded by $$C^{n+1}(n!)^s$$. Class $$s=1$$ is the real-analytic functions, while for $$s>1$$ there are functions of compact support, so that a smooth $$P$$ equal to $$k$$ on a box exists in every Gevrey class $$s>1$$; the discontinuous sawtooth $$P(k)=k$$ on $$(-\pi,\pi]$$ also works. The vertices must vanish on an open set of momenta, so they are not real-analytic, and $$P$$ is linear on an open set, so it is not real-analytic either. With smooth cutoffs of Gevrey class $$s>1$$ the vertices still decay with distance faster than any power, as a stretched exponential with exponent $$1/s$$. The band of $$H_r$$ shrinks as $$2\pi/r$$, so at fixed lattice spacing higher-order vertices act on ever smaller sets of momenta. An analytic charge is not rescued by band-limiting, because the low-energy quartets of Theorem 1 lie inside every band.

## 5. What the band costs

### 5.1 Gauge invariance

Let $$B\subset T^3$$ be the set of momenta on which the vertices act. The transport term $$\kappa\,\xi\cdot\partial h$$ maps a gauge parameter and a field with momenta in $$B$$ to momenta in $$B+B$$. At order $$\kappa$$, the transport term varies the free Hamiltonian $$H_2$$, which is not band-limited, and this variation contains momenta in $$B+B$$. Invariance requires it to be cancelled by the linear variation of the cubic vertex, which lives only in $$B$$. This suggests that the band can be preserved at all orders only if $$B+B\subset B$$. We have not turned this into a proof for general deformed transformations, and treat it as a heuristic criterion.

<div class="proposition" markdown="1">
**Proposition 2.** Let $$B\subset T^3$$ be closed, symmetric ($$B=-B$$), contain a neighbourhood of $$0$$, and satisfy $$B+B\subset B$$. Then $$B=T^3$$.
</div>

<div class="proof" markdown="1">
*Proof.* $$B$$ contains $$0$$ and is closed under addition and negation, so it is a subgroup. A subgroup containing a neighbourhood $$\mathcal N$$ of $$0$$ is open, since $$b+\mathcal N\subset B$$ for each $$b\in B$$. An open subgroup is also closed, its complement being a union of cosets, and $$T^3$$ is connected.
</div>

A band that satisfies the closure criterion is therefore the whole zone, where Proposition 1 is unavailable and Umklapp processes are present. This indicates that band-limited vertices are not invariant under deformed diffeomorphisms whose nonlinear part is the transport term. A different deformation of the gauge transformations, for example one that acts differently on modes outside the band, is not excluded, and whether one exists is open.

### 5.2 Lorentz invariance

A band limit is a statement about spatial momenta in the lattice frame; the energy is unrestricted in a continuous-time theory. It singles out that frame.

<div class="proposition" markdown="1">
**Proposition 3** (no Lorentz-invariant band). Let $$\tilde B\subset\mathbb R^4$$ contain a neighbourhood of $$0$$ and have bounded projection onto the spatial momenta. Then $$\tilde B$$ is not invariant under the identity component of the Lorentz group of any Lorentzian metric $$g$$ on $$\mathbb R^4$$.
</div>

<div class="proof" markdown="1">
*Proof.* Choose $$v\in\tilde B$$ with $$g(v,v)>0$$ (spacelike in signature $$-+++$$); one exists because $$\tilde B$$ contains a neighbourhood of $$0$$. If $$\tilde B$$ were invariant, it would contain the orbit of $$v$$, which is the hyperboloid $$g(x,x)=g(v,v)$$. The orthogonal complement of $$v$$ is a three-dimensional subspace of Lorentzian signature and contains a one-parameter family of null directions $$\nu$$. For each of them, $$g(v+\tau\nu,v+\tau\nu)=g(v,v)$$, so the line $$v+\tau\nu$$ lies on the hyperboloid. Not all these $$\nu$$ are multiples of the time axis of the lattice frame, so for some $$\nu$$ the spatial part of $$v+\tau\nu$$ is unbounded in $$\tau$$.
</div>

So no band can be a Lorentz-invariant regulator in any emergent metric, and loops are expected to produce the preferred-frame effects of [[Collins et al.]](#ref-collins). We now compute them.

### 5.3 One-loop speeds and a photon mass

In the construction of Proposition 1, exact conservation of the total momentum is obtained by band-limiting every vertex, including those of matter. We take the most favourable case for Lorentz invariance at tree level: every field has the exactly relativistic dispersion $$\omega=\lvert k\rvert$$ inside the band, which requires a non-analytic lattice derivative equal to $$k$$ there, so that the only departure from the continuum is a cutoff $$\chi(\lvert k\rvert/b)$$ on every leg of every vertex. Here $$b$$ is the band edge, and $$\chi=1$$ for $$\lvert k\rvert\le y_0b$$, $$\chi=0$$ for $$\lvert k\rvert\ge b$$, with a Gevrey step in between: $$\chi=\varphi(1-\eta)/[\varphi(1-\eta)+\varphi(\eta)]$$ with $$\varphi(x)=\exp(-x^{-\beta})$$ for $$x>0$$, $$\varphi(x)=0$$ for $$x\le0$$, and $$\eta=(\lvert k\rvert/b-y_0)/(1-y_0)$$, where $$\beta=1$$ gives Gevrey class 2 and $$\beta=\tfrac12$$ class 3. We use the same cutoff for the cubic and the seagull (quartic) vertices, which is consistent with Proposition 1 when $$b\le\pi/2$$. A one-loop diagram with external momenta inside the band is then a continuum Euclidean loop in which each internal line carries the weight $$\varrho=\chi^2$$ of its spatial momentum, one factor $$\chi$$ from the vertex at each of its ends, while the loop energy is integrated over the real line.

The model is scalar electrodynamics: a complex scalar $$\Phi$$ of charge $$e$$, massless (the infrared regulator $$m_{\rm IR}$$ is taken to $$0$$), with the cubic and the seagull vertex cut in this way. Write the Euclidean inverse propagator of a field as $$p^2+\Sigma(p)$$ and expand $$\Sigma(p)=\Sigma(0)+A_4p_4^2+A_s\lvert\mathbf p\rvert^2+\dots$$. After continuing $$p_4\to i\omega$$, the low-energy speed is $$c=\big[(1+A_s)/(1+A_4)\big]^{1/2}$$, so

$$
\delta c=\tfrac12(A_s-A_4). \tag{5}
$$

For a marginal coupling, $$\delta c=(e^2/16\pi^2)\,C[\varrho]$$ with $$C$$ a number that depends on the shape of the profile but not on $$b$$, because $$b$$ is the only scale. For the photon $$\gamma$$ (a transverse component, from the $$\Phi$$ loop) $$C_\gamma$$ is independent of the gauge fixing, since no photon propagates in the loop. For $$\Phi$$ (from the photon loop) we use Feynman gauge; because the cut breaks the $$U(1)$$ gauge symmetry, $$C_\Phi$$ depends on this choice, which is part of the definition of the model. Without the cut, both vanish by Lorentz invariance.

The cut also gives the photon a mass. Write $$q^2=q_4^2+\lvert\mathbf q\rvert^2$$ and $$\int\equiv\int d^4q/(2\pi)^4$$. The seagull contributes $$2e^2\delta_{\mu\nu}\int\varrho/q^2$$ to the vacuum polarisation $$\Pi_{\mu\nu}$$, and the bubble $$-e^2\int(2q+p)_\mu(2q+p)_\nu\,\varrho\varrho'/(q^2(q+p)^2)$$, where primes refer to the line with momentum $$q+p$$.

<div class="proposition" markdown="1">
**Proposition 4** (photon mass). At one loop,

$$
\Pi_{xx}(0)=2e^2\!\int\frac{\varrho-\varrho^2}{q^2}-2e^2\!\int\frac{q_x}{q^2}\,\partial_x(\varrho^2),\qquad
\Pi_{44}(0)=2e^2\!\int\frac{\varrho-\varrho^2}{q^2}.
$$

If $$\varrho$$ has compact support in the spatial momenta, $$0\le\varrho\le1$$, and $$\varrho$$ is non-increasing along rays, both are non-negative, and $$\Pi_{xx}(0)>0$$ unless $$\varrho\equiv0$$. For a ball band, $$\varrho=\varrho(\lvert\mathbf q\rvert/b)$$, the photon mass is given by (6) below.
</div>

<div class="proof" markdown="1">
*Proof.* At $$p=0$$, $$\Pi_{\mu\mu}(0)=2e^2\int\varrho/q^2-4e^2\int q_\mu^2\varrho^2/q^4$$ (no sum). Split $$\varrho=(\varrho-\varrho^2)+\varrho^2$$ in the first term and use $$2/q^2-4q_\mu^2/q^4=2\partial_\mu(q_\mu/q^2)$$. Integrating by parts, the boundary terms vanish: in the spatial directions because $$\varrho$$ has compact support, in the energy direction because $$\varrho$$ does not depend on $$q_4$$ and $$\int dq_4\,\partial_4(q_4/q^2)=0$$. This gives both formulas, and the sign conditions give $$\varrho-\varrho^2\ge0$$ and $$-q_x\partial_x\varrho^2\ge0$$. For a ball band, $$\int dq_4/2\pi\;q^{-2}=1/(2\lvert\mathbf q\rvert)$$, $$\partial_x\varrho^2=(\varrho^2)'q_x/\lvert\mathbf q\rvert$$ and the angular average of $$q_x^2$$ is $$\lvert\mathbf q\rvert^2/3$$. The $$\varrho-\varrho^2$$ term then gives the second term of

$$
M^2\equiv\Pi_{xx}(0)=\frac{e^2b^2}{16\pi^2}\Big[\frac{16}3\int_0^\infty y\,\varrho^2\,dy+8\int_0^\infty y\,(\varrho-\varrho^2)\,dy\Big], \tag{6}
$$

where $$y=\lvert\mathbf q\rvert/b$$, and the boundary term gives the first after one integration by parts in $$y$$.
</div>

Without the cut, $$\Pi_{\mu\nu}(0)$$ vanishes, as gauge invariance requires; with the cut, the mismatch between the seagull weight $$\varrho$$ and the bubble weight $$\varrho^2$$, and the derivative of the cutoff, source $$M^2$$. Since $$\Pi_{44}(0)\neq\Pi_{xx}(0)$$, the mass term itself is not Lorentz invariant.

The light-cone coefficients require the second derivatives of the bubbles in $$p$$ and are computed numerically. Two independent calculations agree: one with symmetric routing ($$q\pm p/2$$) and numerical energy integration, the other with routing ($$q,q+p$$), the energy integral done in closed form by residues, and a different infrared regulator on the photon line for $$C_\Phi$$. At $$m_{\rm IR}=0.003\,b$$ they agree to five digits for $$C_\gamma$$ and $$M^2$$ and to $$5\times10^{-5}$$ in relative terms for $$C_\Phi$$, and the numerical $$M^2$$ agrees with (6) to $$10^{-5}$$. The table gives the second calculation at $$m_{\rm IR}=0.001\,b$$; between $$m_{\rm IR}=0.01\,b$$ and $$0.001\,b$$ the coefficients change by less than 0.2%.

| profile ($$y_0$$, Gevrey class) | $$C_\gamma$$ | $$C_\Phi$$ | $$M^2/(e^2b^2/16\pi^2)$$ |
|---|---|---|---|
| 0.2, 2 | +0.309 | +0.605 | 0.852 |
| 0.5, 2 | +0.514 | +1.119 | 1.402 |
| 0.8, 2 | +1.338 | +3.177 | 2.108 |
| 0.5, 3 | +0.385 | +0.796 | 1.339 |
| 0.8, 3 | +1.012 | +2.363 | 2.066 |

All coefficients are positive, $$C_\Phi/C_\gamma$$ lies between 1.96 and 2.38, and the coefficients vary with the shape of the profile by factors of 2.5–5 ($$C_\gamma$$ by 4.3, $$C_\Phi$$ by 5.3, $$M^2$$ by 2.5); none comes near zero.

With $$\alpha=e^2/4\pi=1/137.036$$, (5) gives a photon speed shift $$\delta c_\gamma=C_\gamma\,\alpha/4\pi=(1.8\text{–}7.8)\times10^{-4}$$. The graviton has no $$O(e^2)$$ shift, so this is also $$c_\gamma-c_{\rm GW}$$ at this order. The arrival times of GW170817 and GRB 170817A bound $$(c_{\rm GW}-c_\gamma)/c$$ between $$-3\times10^{-15}$$ and $$+7\times10^{-16}$$ [[GW170817]](#ref-gw170817). Since $$\delta c_\gamma>0$$, the relevant bound, $$3\times10^{-15}$$, is $$10^{-11}$$–$$10^{-12}$$ times $$\delta c_\gamma$$. The shift does not depend on $$b$$, so no choice of band edge reduces it. The two species of the same theory also disagree, $$\lvert\delta c_\gamma-\delta c_\Phi\rvert=(1.7\text{–}11)\times10^{-4}$$, with the gauge caveat above. The photon mass is $$m_\gamma=(0.022\text{–}0.035)\,b$$; for a band edge at the Planck mass, $$1.2\times10^{28}$$ eV, this is about $$3\times10^{26}$$ eV, some $$10^{44}$$ times the bound $$m_\gamma<10^{-18}$$ eV [[PDG]](#ref-pdg). Each of these can be cancelled by a bare Lorentz-violating counterterm, since no symmetry forbids one once gauge invariance is broken; agreement with the bounds then requires the bare parameters to be tuned species by species, the speeds to between one part in $$10^{11}$$ and one part in $$10^{12}$$. This is the fine-tuning problem of [[Collins et al.]](#ref-collins), realised here by the vertices of Proposition 1. Graviton loops add contributions of order $$\kappa^2b^2/16\pi^2$$, which depend on $$b$$ and could cancel the photon shift only for one tuned value of $$b$$; we have not evaluated them.

## 6. Integer charges in Villain form

A separate question about the free theory of section 1 is whether its gauge group can be made compact. Compact $$U(1)$$ makes electric charges integers; compactifying $$h$$ would likewise make the momentum-constraint charges integers. The standard device is the modified Villain formulation [[GLSS 2021]](#ref-glss1). Each component $$h_a$$ of the lattice field, with conjugate $$\varpi_a$$, is paired with integer fields $$N_b$$ with conjugate angles $$\theta_b$$. The potential depends on $$h$$ only through $$X=Lh-2\pi N$$, where $$L$$ is a translation-invariant integer matrix of finite range. Finally the integer shifts $$h\to h+2\pi n$$, $$N\to N+Ln$$, for every integer-valued $$n$$ on the components of $$h$$, are gauged by imposing

$$
\mathcal W[n]\equiv\exp\big(2\pi i\langle\varpi,n\rangle+i\langle\theta,Ln\rangle\big)=1 .
$$

This makes every component of $$h$$ an angle. The symmetrised gradient $$S$$ of section 2 is an integer matrix, and the generators of the linearised gauge transformations $$h\to h+S\xi$$ are the momentum-constraint charges $$\mathsf q=S^{\rm T}\varpi$$, one per link.

<div class="theorem" markdown="1">
**Theorem 2** (Villain dichotomy). In the modified Villain formulation above:

(a) the charges $$\mathsf q$$ have integer spectrum on the states with $$\mathcal W[n]=1$$ for all $$n$$ if and only if $$LS=0$$;

(b) if $$LS=0$$, the symbol of $$L$$ is $$O(k^2)$$, so the quadratic potential is $$O(k^4)$$; if moreover the kinetic term has a bounded symbol, every mode has $$\omega=O(k^2)$$.

Hence integer charges and a graviton with $$\omega\propto\lvert k\rvert$$ are incompatible in this formulation.
</div>

<div class="proof" markdown="1">
*Proof.* (a) $$\varpi$$ and $$\theta$$ commute, and $$\mathcal W[n]=1$$ for all integer $$n$$ means that $$\varpi+L^{\rm T}\theta/2\pi=\zeta$$ with $$\zeta$$ integer-valued, while $$\theta$$ is unrestricted. Then $$\mathsf q=S^{\rm T}\zeta-(LS)^{\rm T}\theta/2\pi$$. The first term is an integer; the second sweeps a continuum as $$\theta$$ varies unless $$LS=0$$.

(b) Let $$\ell(k)$$ and $$S(k)$$ be the symbols of $$L$$ and $$S$$, and expand $$\ell=\ell_0+\ell_1(k)+O(k^2)$$ and $$S(k)\xi=i(k_i\xi_j+k_j\xi_i)+O(k^3)$$. At first order in $$k$$, $$\ell(k)S(k)=0$$ requires $$\ell_0$$ to annihilate every symmetrised product $$k_i\xi_j+k_j\xi_i$$; these span all symmetric tensors, so $$\ell_0=0$$. At second order, write $$\ell_1(k)_{ij}=\sum_l c_{ij,l}k_l$$ for each row of $$\ell_1$$, with $$c$$ symmetric in $$ij$$. The condition $$\sum c_{ij,l}k_l(k_i\xi_j+k_j\xi_i)=0$$ for all $$k,\xi$$ says that $$c_{ij,l}$$ is antisymmetric under $$i\leftrightarrow l$$. A tensor symmetric in its first two indices and antisymmetric in its first and third vanishes: applying the two symmetries alternately three times each gives $$c_{ij,l}=-c_{ij,l}$$. So $$\ell=O(k^2)$$, the quadratic potential $$\ell(k)^\dagger V\,\ell(k)$$ is $$O(k^4)$$, and with a bounded kinetic symbol $$\omega^2=O(k^4)$$.
</div>

The theorem is narrow. Its hypothesis is that the integer gauge lattice is all of $$\mathbb Z$$ on every component of $$h$$. If instead only the gauge directions are compactified, by imposing $$\exp(2\pi i\langle\varpi,S\iota\rangle)=1$$ for integer $$\iota$$ on links, then the charges $$S^{\rm T}\varpi$$ are integers, the potential of section 1 is invariant under $$h\to h+S\iota$$ because the linearised curvature of $$S\iota$$ vanishes identically, and the dispersion is (2). Integer charges and $$\omega\propto\lvert k\rvert$$ then coexist. That compact lattice tensor gauge theories have $$\omega\propto k^2$$ was known in examples [[Xu]](#ref-xu) [[Xu–Hořava]](#ref-xh) [[GLSS 2022]](#ref-glss2); Theorem 2 gives the reason for the whole Villain class. The "only if" in (a) also has limited reach: when $$LS\neq0$$, a quadratic potential $$\tfrac12X^{\rm T}VX$$ changes under $$h\to h+S\xi$$ with continuous $$\xi$$ by terms such as $$-2\pi N^{\rm T}VLS\xi$$, so in that branch the theory generally has no continuous linear gauge symmetry to begin with.

## 7. Limitations and open questions

Theorem 1 concerns a Hamiltonian theory in continuous time on a fixed Bravais lattice, at tree level and order by order in $$\kappa$$. Spacetime lattices, on which energy is also periodic, dynamical lattices and non-local actions are outside its scope. It assumes a translation charge that acts identically on both helicities; a charge that mixes them, with a $$2\times2$$ matrix symbol, is not covered. It assumes no three-wave resonances (H2), which the nearest-neighbour graviton satisfies but an exactly linear dispersion does not, since collinear three-wave processes are then resonant; with such dispersions the cubic step needs separate treatment, and the on-shell three-graviton amplitude would have to vanish on the collinear set. The passage from exact gauge invariance to the conserved charge of (H3) and the identification of $$\mathcal T$$ with the general-relativity amplitude in (H5) are assumptions about the lattice theory. The obstruction first becomes visible at order $$\kappa^2$$; whether an exponentially local $$Q_3$$ exists at order $$\kappa$$ is open; the lattice Leibniz defect obstructs the obvious construction.

Proposition 2 rests on a heuristic closure criterion and concerns only gauge transformations whose nonlinear part is the transport term; whether band-limited vertices admit some other exact deformation of linearised diffeomorphisms is unknown. The one-loop results use ball-shaped bands, a single toy matter sector and Feynman gauge for $$C_\Phi$$; cubic bands, fermions and the graviton's own shift were not computed. They show that a band-limited theory needs bare Lorentz-violating parameters tuned to $$10^{-11}$$–$$10^{-12}$$; they do not exclude such parameters. Theorem 2 is limited to Villain lattices in which every component of $$h$$ is compact. Whether some Lorentz-invariant structure other than a momentum band, for example a random rather than a regular lattice, avoids both the identity-theorem obstruction and the band is not addressed here.

## References

1. <a id="ref-dfs"></a>Dittrich, Freidel, Speziale, “Linearized dynamics from the 4-simplex Regge action,” [arXiv:0707.4513](https://arxiv.org/abs/0707.4513).
2. <a id="ref-hoehn"></a>Höhn, “Canonical linearized Regge calculus: counting lattice gravitons with Pachner moves,” [arXiv:1411.5672](https://arxiv.org/abs/1411.5672).
3. <a id="ref-bd1"></a>Bahr, Dittrich, “(Broken) gauge symmetries and constraints in Regge calculus,” [arXiv:0905.1670](https://arxiv.org/abs/0905.1670).
4. <a id="ref-bd2"></a>Bahr, Dittrich, “Breaking and restoring of diffeomorphism symmetry in discrete gravity,” [arXiv:0909.5688](https://arxiv.org/abs/0909.5688).
5. <a id="ref-dittrich"></a>Dittrich, “Diffeomorphism symmetry in quantum gravity models,” [arXiv:0810.3594](https://arxiv.org/abs/0810.3594).
6. <a id="ref-gw"></a>Ginsparg, Wilson, “A remnant of chiral symmetry on the lattice,” Phys. Rev. D 25 (1982) 2649.
7. <a id="ref-luscher"></a>Lüscher, “Exact chiral symmetry on the lattice and the Ginsparg–Wilson relation,” [arXiv:hep-lat/9802011](https://arxiv.org/abs/hep-lat/9802011).
8. <a id="ref-horvath"></a>Horváth, “Ginsparg–Wilson relation and ultralocality,” [arXiv:hep-lat/9808002](https://arxiv.org/abs/hep-lat/9808002).
9. <a id="ref-bietenholz"></a>Bietenholz, “The absence of ultralocal Ginsparg–Wilson fermions,” [arXiv:hep-lat/9901005](https://arxiv.org/abs/hep-lat/9901005).
10. <a id="ref-kss1"></a>Kato, Sakamoto, So, “Taming the Leibniz rule on the lattice,” [arXiv:0803.3121](https://arxiv.org/abs/0803.3121).
11. <a id="ref-kss2"></a>Kato, Sakamoto, So, “No-go theorem of Leibniz rule and supersymmetry on the lattice,” [arXiv:0810.2360](https://arxiv.org/abs/0810.2360).
12. <a id="ref-deser"></a>Deser, “Self-interaction and gauge invariance,” [arXiv:gr-qc/0411023](https://arxiv.org/abs/gr-qc/0411023).
13. <a id="ref-bh"></a>Barnich, Henneaux, “Consistent couplings between fields with a gauge freedom and deformations of the master equation,” [arXiv:hep-th/9304057](https://arxiv.org/abs/hep-th/9304057).
14. <a id="ref-spohn"></a>Spohn, “Collisional invariants for the phonon Boltzmann equation,” [arXiv:math-ph/0605069](https://arxiv.org/abs/math-ph/0605069).
15. <a id="ref-peierls"></a>Peierls, “Zur kinetischen Theorie der Wärmeleitung in Kristallen,” Ann. Phys. (Leipzig) 395 (1929) 1055.
16. <a id="ref-lkm"></a>Lewis, Kempf, Menicucci, “Quantum lattice models that preserve continuous translation symmetry,” [arXiv:2303.07649](https://arxiv.org/abs/2303.07649).
17. <a id="ref-collins"></a>Collins, Perez, Sudarsky, Urrutia, Vucetich, “Lorentz invariance and quantum gravity: an additional fine-tuning problem?,” [arXiv:gr-qc/0403053](https://arxiv.org/abs/gr-qc/0403053).
18. <a id="ref-clm"></a>Chankowski, Lewandowski, Meissner, “Ultraviolet cutoffs and the photon mass,” [arXiv:1608.01214](https://arxiv.org/abs/1608.01214).
19. <a id="ref-xu"></a>Xu, “Novel algebraic boson liquid phase with soft graviton excitations,” [arXiv:cond-mat/0602443](https://arxiv.org/abs/cond-mat/0602443).
20. <a id="ref-xh"></a>Xu, Hořava, “Emergent gravity at a Lifshitz point from a Bose liquid on the lattice,” [arXiv:1003.0009](https://arxiv.org/abs/1003.0009).
21. <a id="ref-glss1"></a>Gorantla, Lam, Seiberg, Shao, “A modified Villain formulation of fractons and other exotic theories,” [arXiv:2103.01257](https://arxiv.org/abs/2103.01257).
22. <a id="ref-glss2"></a>Gorantla, Lam, Seiberg, Shao, “Global dipole symmetry, compact Lifshitz theory, tensor gauge theory, and fractons,” [arXiv:2201.10589](https://arxiv.org/abs/2201.10589).
23. <a id="ref-gw170817"></a>LIGO Scientific Collaboration, Virgo Collaboration, Fermi Gamma-Ray Burst Monitor, INTEGRAL, “Gravitational waves and gamma-rays from a binary neutron star merger: GW170817 and GRB 170817A,” [arXiv:1710.05834](https://arxiv.org/abs/1710.05834).
24. <a id="ref-pdg"></a>Navas et al. (Particle Data Group), “Review of particle physics,” Phys. Rev. D 110 (2024) 030001.
