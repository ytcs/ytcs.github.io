---
title: The sign of energy exchange between vacuum and matter
date: 2026-10-29
category: dark-sector
order: 2
description: If a non-propagating vacuum component tracks the minimum of its free energy, or relaxes passively towards it, or exchanges energy with matter through a unital map on a passive state, energy flows from vacuum to matter and dark-matter or electron masses are smaller in the past; this is opposite to what some fits of cosmological data prefer.
status: Theorems proved in full under stated hypotheses (global minimiser tracking, linear passive relaxation with constant Hessian, unital maps on passive states); numerical checks on random models; the electron-mass bounds use quoted experimental limits.
excludes: Mass-coupled vacuum-matter models that follow the minimum of a free energy or relax passively towards it, as an explanation of a dark-matter density or electron mass that decreases with time; a scalar field that follows its effective minimum as the origin of a heavier electron at recombination, with the stated proviso on the laboratory environment.
---

Some analyses of cosmological data favour a dark-energy equation of state with $$w_0>-1$$ and $$w_a<0$$, which for $$w_0+w_a<-1$$ lies below $$-1$$ at high redshift, and give central values of the electron mass at recombination that are about one per cent above the laboratory value, with uncertainties of about half a per cent. If the dark matter is conserved at recombination, the first corresponds to a dark-matter density per comoving volume that was larger then than today (Lemma 1), that is, to energy flowing from matter to the vacuum; the second asks for a particle mass that decreases with time. We ask which dynamics of a vacuum component can produce this direction. The answer is a sign theorem. If the vacuum component has no kinetic energy and its state tracks the global minimum of its free energy (Theorem 1a), or relaxes towards that minimum through a passive linear dynamics (Theorem 1b), or exchanges energy with matter through a unital map acting on a passive state of the matter (Theorem 1c), then energy flows from the vacuum to the matter. Under tracking the mass of the matter particles does not decrease with time, and a cosmological reconstruction that fixes the dark-matter density at recombination returns a dark-energy equation of state $$w\ge-1$$; under passive relaxation the particle mass does not fall below its value at the start of the relaxation, and under a unital map the matter gains energy at each step. A field that tracks its minimum in a bath of Standard Model particles makes the electron lighter in the hot, dense past, whatever its potential (Theorem 2). A field that does not track, with a coupling constrained by tests of the equivalence principle, shifts the electron mass by at most $$0.96\%$$ times the square root of its average kinetic share of the energy density (Theorem 3). The sections on what the theorems do not exclude list the departures that reverse the sign.

Prior work covers parts of this. Pavón and Wang [[PW]](#ref-pw) argued from the Le Chatelier–Braun principle that energy flows from dark energy to dark matter. The sign of the mass shift for a field that minimises its effective potential in a matter background was found for neutrinos by Fardon, Nelson and Weiner [[FNW]](#ref-fnw) and for dark matter by Afshordi, Zaldarriaga and Kohri [[AZK]](#ref-azk) and Bean, Flanagan and Trodden [[BFT 1]](#ref-bft1), [[BFT 2]](#ref-bft2), who also derived the accompanying instability of the sound speed. The comparison argument in Theorem 1a is a form of monotone comparative statics [[Topkis]](#ref-topkis), [[MS]](#ref-ms). The statement that a fit with a conserved-matter reconstruction can turn an interacting vacuum into an apparent phantom component is due to Das, Corasaniti and Khoury [[DCK]](#ref-dck). For the electron mass, Baryakhtar, Simon and Weiner [[BSW 1]](#ref-bsw1), [[BSW 2]](#ref-bsw2) computed the in-medium potential of a hyperlight scalar that modulates $$m_e$$ and found that it drives the field towards lower electron mass. What this page adds is a single statement that covers mass-coupled tracking, passive relaxation and unital heating, with the hypotheses stated so that a violation can be located, the extension of the electron-mass argument to any potential and to any number of sources, and the excursion bound for a linearly coupled field.

## 1. Vacuum, matter and what a reconstruction sees

We work in a spatially flat Friedmann–Robertson–Walker universe with scale factor $$a(t)$$, Hubble rate $$H=\dot a/a>0$$, and signature $$(-,+,+,+)$$. A component with energy density $$\rho$$ and pressure $$p$$ that exchanges no energy with others obeys $$\dot\rho+3H(\rho+p)=0$$. The vacuum component $$V$$ has $$T^V_{ab}=-\rho_vg_{ab}$$, so $$p_v=-\rho_v$$, and it is non-propagating: it has no kinetic term, and its energy density is a function of auxiliary variables that carry no energy of their own. The matter component $$M$$ is dust, with a conserved particle number density $$n$$ ($$na^3$$ constant) and mass per particle $$\mu$$, so that $$\rho_M=n\mu$$; heat deposited in matter is treated separately where it occurs. Total energy is conserved, which defines the exchange rate $$Q$$:

$$
\dot\rho_v=-Q,\qquad \dot\rho_M+3H\rho_M=Q. \tag{1}
$$

A positive $$Q$$ means that energy flows from the vacuum to matter. For dust, $$Q=n\dot\mu$$ when the particle number is conserved, so $$Q\ge0$$ is the statement that the particle mass does not decrease.

A cosmological analysis typically assumes that cold dark matter is conserved, fixes its density at some epoch $$a_*$$, and attributes the rest of the energy budget to dark energy. We call the result the reconstruction anchored at $$a_*$$: $$\rho_c^{\rm rec}(a)=\rho_M(a_*)(a_*/a)^3$$ and $$\rho_{\rm DE,eff}=\rho_v+\rho_M-\rho_c^{\rm rec}$$. Early anchors correspond to normalising the dark-matter density to the cosmic microwave background (CMB), late anchors to a low-redshift measurement.

<div class="lemma" markdown="1">
**Lemma 1** (reconstruction). Under (1), with $$\Delta(t)=a^{-3}\int_{t_*}^{t}Q\,a^3\,dt'$$,

$$
\dot\rho_{\rm DE,eff}=-3H\Delta,\qquad 1+w_{\rm eff}=\frac{\Delta}{\rho_{\rm DE,eff}}, \tag{2}
$$

where $$w_{\rm eff}$$ is defined by $$\dot\rho_{\rm DE,eff}=-3H(1+w_{\rm eff})\rho_{\rm DE,eff}$$ and $$\rho_{\rm DE,eff}>0$$. If $$Q\ge0$$ at all times, then for an anchor in the past of $$t$$ ($$t_*<t$$) $$\Delta\ge0$$ and $$w_{\rm eff}\ge-1$$, and for an anchor in the future of $$t$$ ($$t_*>t$$) $$\Delta\le0$$ and $$w_{\rm eff}\le-1$$. The ratio of the comoving dark-matter densities satisfies $$(a^3\rho_M)(t_1)\le(a^3\rho_M)(t_2)$$ for $$t_1<t_2$$.
</div>

<div class="proof" markdown="1">
*Proof.* From (1), $$a^3\rho_M$$ has derivative $$a^3Q$$, so $$\rho_M-\rho_c^{\rm rec}=\Delta$$. Then $$\rho_{\rm DE,eff}=\rho_v+\Delta$$ and $$\dot\rho_{\rm DE,eff}=-Q+(-3H\Delta+Q)=-3H\Delta$$. The sign statements follow from the sign of the integral over $$[t_*,t]$$, which is negative if the limits are reversed. The last statement integrates $$a^3Q\ge0$$. $$\square$$
</div>

The lemma is bookkeeping, and it fixes the meaning of the data. A dark-energy equation of state below $$-1$$ at high redshift in a fit anchored to the CMB is, by (2), the same as $$\Delta<0$$: matter must have lost comoving energy after the anchor, which means $$Q<0$$ over some interval and $$(a^3\rho_M)$$ larger at the anchor than later. With a late anchor, a vacuum that feeds matter looks phantom in the past, the effect described in [[DCK]](#ref-dck). The comparison with observations therefore needs the anchor to be stated. The dark-energy preference of the DESI second data release combined with CMB and supernova data is a preference for $$w_0>-1$$ and $$w_a<0$$ in $$w(a)=w_0+w_a(1-a)$$, at $$2.8$$–$$4.2\sigma$$ over a cosmological constant depending on the supernova sample [[DESI]](#ref-desi); the question of the sign is whether the physics of an exchange can produce the corresponding reconstruction.

## 2. The thermodynamic sign theorem

We start from the simplest case, then generalise. Take a single auxiliary variable $$\xi$$, a vacuum energy $$g(\xi)=\tfrac12h\xi^2$$ with $$h>0$$, and dust of particle number density $$n$$ whose particle mass is $$\mu(\xi)=\mu_0+b\,\xi$$. The energy density available to $$\xi$$ is $$\Phi=\tfrac12h\xi^2+n(\mu_0+b\xi)$$. Its minimum is at $$\xi=-nb/h$$ and the mass there is $$\mu=\mu_0-nb^2/h$$. Matter lowers its energy by making its particles lighter, so the denser the dust, the lighter the particles; as the universe expands $$n$$ falls and $$\mu$$ rises towards $$\mu_0$$. The vacuum energy $$g=\tfrac12 n^2b^2/h$$ falls with $$n$$ as well, and the difference is paid to matter: with $$\rho_M=n\mu$$ and $$na^3$$ constant, $$Q=n\dot\mu=\tfrac{\,3Hn^2b^2}{h}>0$$. The sign does not depend on the sign of $$b$$.

The general statement replaces the quadratic and the linear coupling by arbitrary functions, and replaces the instantaneous minimum by the situations in which it is reached.

<div class="theorem" markdown="1">
**Theorem 1** (thermodynamic sign). Let $$\xi$$ take values in a set $$X$$, let the vacuum energy density be $$\rho_v=g(\xi)$$ with $$g:X\to\mathbb R$$ arbitrary, and let the matter depend on $$\xi$$ only through a mass parameter $$\mu(\xi)>0$$, an arbitrary function of $$\xi$$. The free energy density of the matter in state $$j$$ is $$F_j(\mu)$$, differentiable, with $$\partial_\mu F_j>0$$; for dust of conserved particle number density $$n_j$$ it is $$F_j(\mu)=n_j\mu$$, and for a plasma it is the free energy of section 4. We call $$\Phi_j(\xi)=g(\xi)+F_j(\mu(\xi))$$ the effective potential in state $$j$$. The conclusion is stated in each case.

(a) Global tracking. Let $$A$$ be an earlier state and $$B$$ a later one, and suppose $$\partial_\mu F_B(\mu)<\partial_\mu F_A(\mu)$$ for all $$\mu$$ in the interval between $$\mu_A$$ and $$\mu_B$$ (for dust, $$n_B<n_A$$; for a plasma, the electron scalar density of section 4 is smaller in $$B$$). If $$\xi_A$$ and $$\xi_B$$ are global minimisers of $$\Phi_A$$ and $$\Phi_B$$, then $$\mu(\xi_B)\ge\mu(\xi_A)$$. In particular a field that follows the global minimiser of $$\Phi_t$$ through an expansion has $$\mu(t)$$ non-decreasing, also across first-order jumps.

(b) Passive linear relaxation. Let $$X=\mathbb R^N$$, $$g(\xi)=\tfrac12\xi^{\mathsf T}H\xi$$ with $$H$$ symmetric positive definite and constant, $$\mu(\xi)=\mu_0+b^{\mathsf T}\xi$$, dust with $$n(t)$$ non-increasing, and $$\dot\xi=-\Gamma(t)\nabla\Phi_t(\xi)$$ with $$\Gamma(t)$$ any matrix function whose symmetric part is positive semi-definite (it may be non-reciprocal and time dependent). If $$\xi$$ starts at the minimiser of $$\Phi_{t_0}$$, then $$\mu(t)\ge\mu(t_0)$$ for all $$t\ge t_0$$, and the energy dissipated into the matter, $$\nabla\Phi^{\mathsf T}\Gamma\nabla\Phi$$, is non-negative.

(c) Unital exchange on a passive state. Let matter be a finite-dimensional quantum system with Hamiltonian $$H_M$$ and state $$\varrho$$, and let the exchange over a time step act as a unital completely positive trace-preserving map $$\mathcal E$$ on $$\varrho$$, jointly energy conserving with the vacuum. If $$\varrho$$ is passive, that is diagonal in an energy eigenbasis with occupations non-increasing in energy (every thermal state at positive temperature is of this kind), then $$\mathrm{Tr}[H_M\,\mathcal E(\varrho)]\ge\mathrm{Tr}[H_M\varrho]$$, and the vacuum loses the corresponding energy.
</div>

Idea of the proofs. In (a) the two minimality conditions are added so that $$g$$ cancels, and what remains is a comparison of the matter free energies only. In (b) the response of the mass to a step in the source is positive because a contraction cannot overshoot its equilibrium, and a decreasing source is a positive superposition of steps. In (c) a unital map can only mix, and mixing raises the energy of a passive state. The steps are elementary; the content is in locating the hypotheses.

<div class="proof" markdown="1">
*Proof of (a).* Minimality of $$\xi_A$$ for $$\Phi_A$$ and of $$\xi_B$$ for $$\Phi_B$$ gives
$$g(\xi_A)+F_A(\mu_A)\le g(\xi_B)+F_A(\mu_B)$$ and $$g(\xi_B)+F_B(\mu_B)\le g(\xi_A)+F_B(\mu_A)$$, with $$\mu_j=\mu(\xi_j)$$.
Adding the two, $$g$$ cancels, and $$G(\mu_B)\le G(\mu_A)$$ for $$G=F_B-F_A$$. By hypothesis $$G'<0$$ on the interval between $$\mu_A$$ and $$\mu_B$$, so $$G$$ is strictly decreasing there, and $$G(\mu_B)\le G(\mu_A)$$ forces $$\mu_B\ge\mu_A$$. No convexity of $$g$$ or of $$\Phi$$ is used, and the argument holds when the minimiser jumps between distant points of $$X$$. For dust, $$G(\mu)=(n_B-n_A)\mu$$. For the exchange rate, total energy is $$\rho_v+\rho_M=g+n\mu=\Phi_t(\xi(t))$$, and it is continuous across a jump between two degenerate minima; with $$na^3$$ constant, $$a^3\rho_M=a^3n\mu$$ is non-decreasing, so $$Q\ge0$$ in the sense of measures. $$\square$$

*Proof of (b).* Put $$y=H^{1/2}\xi$$, $$c=H^{-1/2}b$$ and $$A(t)=H^{1/2}\Gamma(t)H^{1/2}$$. The potential is $$\Phi_t=\tfrac12y^{\mathsf T}y+n(t)\,c^{\mathsf T}y+\text{const}$$, so $$\dot y=-A(t)\,(y+n(t)c)$$ and the instantaneous minimiser is $$y_{\rm eq}(t)=-n(t)c$$. The mass is $$\mu=\mu_0+c^{\mathsf T}y$$. Write $$\theta(t)=n(t_0)-n(t)\ge0$$, non-decreasing, so that $$y_{\rm eq}(t)=y_0+\theta(t)c$$ with $$y_0=y_{\rm eq}(t_0)=y(t_0)$$. For $$\delta=y-y_0$$, $$\dot\delta=-A(\delta-\theta c)$$. Let $$U(t,\tau)$$ be the propagator of $$\dot z=-Az$$. Since $$\tfrac{d}{dt}\lVert z\rVert^2=-2z^{\mathsf T}(\mathrm{sym}\,A)z\le0$$ and $$\mathrm{sym}\,A=H^{1/2}(\mathrm{sym}\,\Gamma)H^{1/2}\ge0$$, $$\lVert U\rVert\le1$$. For $$\theta$$ equal to a unit step at time $$\tau$$, the solution is $$\delta(t)=(1-U(t,\tau))c$$, because $$\partial_\tau U(t,\tau)=U(t,\tau)A(\tau)$$. A non-decreasing $$\theta$$ is a positive superposition of steps, $$\theta(t)=\int_{t_0}^t d\theta(\tau)$$, so
$$c^{\mathsf T}\delta(t)=\int d\theta(\tau)\,\big[c^{\mathsf T}c-c^{\mathsf T}U(t,\tau)c\big]\ge0$$
by the Cauchy–Schwarz inequality and $$\lVert U\rVert\le1$$. Hence $$\mu(t)-\mu(t_0)=c^{\mathsf T}\delta\ge0$$. The rate at which the effective potential energy is dissipated is $$-\nabla\Phi\cdot\dot\xi=\nabla\Phi^{\mathsf T}\Gamma\nabla\Phi\ge0$$, which by energy conservation is the heat received by the matter. $$\square$$

*Proof of (c).* Let $$\varrho=\sum_jp_j\vert j\rangle\langle j\vert $$ in the energy eigenbasis, with energies $$E_1\le E_2\le\dots$$ and $$p_1\ge p_2\ge\dots$$. Define $$W_{ij}=\langle i\vert \mathcal E(\vert j\rangle\langle j\vert )\vert i\rangle\ge0$$. Trace preservation gives $$\sum_iW_{ij}=1$$ and unitality gives $$\sum_jW_{ij}=\langle i\vert \mathcal E(\mathbb 1)\vert i\rangle=1$$, so $$W$$ is doubly stochastic. The energy after the step is $$\sum_iE_iq_i$$ with $$q=Wp$$ (diagonal elements of $$\mathcal E(\varrho)$$ in the energy basis; off-diagonal elements do not contribute to the trace). Doubly stochastic maps preserve the sum and satisfy $$Q_k\equiv\sum_{i\le k}q_i\le P_k\equiv\sum_{i\le k}p_i$$ for every $$k$$, because $$p$$ is sorted in decreasing order and $$\sum_{i\le k}q_i=\sum_jp_j\sum_{i\le k}W_{ij}$$ with $$\sum_{i\le k}W_{ij}\in[0,1]$$ and $$\sum_j\sum_{i\le k}W_{ij}=k$$. Summation by parts then gives $$\sum_iE_iq_i=E_N-\sum_{k<N}(E_{k+1}-E_k)Q_k\ge E_N-\sum_{k<N}(E_{k+1}-E_k)P_k=\sum_iE_ip_i$$ ($$N$$ is the dimension). This is passivity in the sense of [[PW78]](#ref-pw78), [[Lenard]](#ref-lenard): no unital mixing lowers the energy of a passive state. $$\square$$
</div>

Hypotheses are used as follows. Global minimality is used only in (a), and the monotonicity of the source only through $$\partial_\mu F_B<\partial_\mu F_A$$. In (b) the constancy of $$H$$ is used in the substitution $$y=H^{1/2}\xi$$ that turns the dynamics into a contraction, and starting on the equilibrium branch is used in $$\delta(t_0)=0$$. The condition that the vacuum is non-propagating means that its energy is $$g(\xi)$$ only: with kinetic energy, $$w_v\ne-1$$ and the reconstruction of Lemma 1 changes character. The sign of the coupling $$b$$ never enters.

Two consequences connect the theorem to the reconstruction of section 1.

<div class="corollary" markdown="1">
**Corollary 1.** Under (a) of Theorem 1, $$Q\ge0$$ at all times, so the comoving dark-matter density $$a^3\rho_M$$ at recombination is not larger than today, $$\omega_c(\mathrm{rec})/\omega_c(0)\le1$$, and a reconstruction anchored at recombination returns $$w_{\rm eff}\ge-1$$ at all later times whenever $$\rho_{\rm DE,eff}>0$$. Under (b), the particle mass satisfies $$\mu(t)\ge\mu(t_0)$$, so the rest-mass part $$a^3n\mu$$ of the comoving matter density at the start $$t_0$$ of the relaxation is not larger than at any later time; the heat deposited in the matter is positive in addition. Under (c), the matter gains energy at each step and the vacuum loses it, $$\dot\rho_v\le0$$.
</div>

The proof of the first statement is Lemma 1 together with $$Q\ge0$$. A fit that prefers $$\omega_c(\mathrm{rec})/\omega_c(0)>1$$ is therefore asking for $$Q<0$$ over some interval, which Theorem 1 forbids for the three classes.

We checked the three parts numerically. For (a), 500 random one-dimensional models with a non-convex quartic-plus-sextic $$g$$, a non-monotone $$\mu(\xi)$$ and a grid minimiser over 200 values of $$n$$ falling from 5 to 0.01 had no violation of the monotonicity of $$\mu$$; 161 of the models had first-order jumps. For (b), 40 random runs with random positive-definite $$H$$ in two to four dimensions, a time-dependent symmetric $$\Gamma$$ plus a time-dependent antisymmetric part, and $$n(t)=e^{-0.15t}$$ had $$\mu(t)\ge\mu(t_0)$$ throughout; with $$\Gamma$$ replaced by $$-\Gamma_{\rm sym}$$ (an active medium), the mass fell below its initial value in 30 of 30 runs. For (c), random mixtures of Haar-random unitaries acting on random passive states in two to five dimensions lowered the energy in none of 2000 trials.

## 3. Where the theorem applies

### Mass-coupled static fluids

Take $$\rho=g(\xi)+nf(\xi)$$ for dust whose mass per particle is $$f(\xi)$$, with $$\xi$$ at a strict minimum at fixed $$n$$: $$g'+nf'=0$$ with Hessian $$\mathcal H=g''+nf''>0$$.

<div class="proposition" markdown="1">
**Proposition 1** (Hessian identity). Along the branch of strict minima, $$d\xi/dn=-\mathcal H^{-1}f'$$, and

$$
\frac{dm}{dn}=-f'^{\mathsf T}\mathcal H^{-1}f'\le0,\qquad \frac{d\rho_v}{dn}=n\,f'^{\mathsf T}\mathcal H^{-1}f'\ge0, \tag{3}
$$

where $$m=f(\xi)$$ and $$\rho_v=g(\xi)$$. With $$\chi=n\,f'^{\mathsf T}\mathcal H^{-1}f'/m\ge0$$, the mass changes as $$d\ln m/d\ln a=3\chi\ge0$$, and the adiabatic sound speed squared of the tracking fluid is $$c_s^2=-\chi\le0$$.
</div>

<div class="proof" markdown="1">
*Proof.* Differentiate $$g'(\xi)+nf'(\xi)=0$$ along the branch: $$\mathcal H\,\xi'+f'=0$$. Then $$m'=f'^{\mathsf T}\xi'=-f'^{\mathsf T}\mathcal H^{-1}f'$$ and $$\rho_v'=g'^{\mathsf T}\xi'=-nf'^{\mathsf T}\xi'$$ gives the second relation. With $$na^3$$ constant, $$dn/d\ln a=-3n$$, so $$d\ln m/d\ln a=-3n\,m'/m=3\chi$$. The total energy density on the branch is $$\rho_{\rm eq}(n)=g+nf$$ with $$\rho_{\rm eq}'=f=m$$ (envelope theorem) and $$\rho_{\rm eq}''=m'$$, the pressure is $$p=n\rho_{\rm eq}'-\rho_{\rm eq}=-g$$, and $$c_s^2=dp/d\rho=n\,m'/m=-\chi$$. $$\square$$
</div>

The sign $$c_s^2<0$$ is the adiabatic instability of mass-varying dark sectors found in [[AZK]](#ref-azk), [[BFT 1]](#ref-bft1), [[BFT 2]](#ref-bft2). It does not enter the sign theorem, but it means that a non-propagating auxiliary variable gives a fluid with a gradient instability; a propagating variable removes it only on scales larger than its Compton wavelength. For the global minimiser, the concavity of $$\rho_{\rm eq}(n)=\min_\xi[g+nf]$$, a minimum of functions affine in $$n$$, gives the monotonicity of $$m$$ at once, including across jumps.

### Unimodular heating

In unimodular gravity the cosmological term is an integration function. Write the field equations as $$G_{ab}+\Lambda g_{ab}=8\pi GT_{ab}$$ with $$\Lambda(x)$$ determined by the Bianchi identity, $$\nabla_b\Lambda=8\pi G\nabla^aT_{ab}$$ [[JPS]](#ref-jps). In a homogeneous universe $$u^b\nabla^aT_{ab}=-(\dot\rho_M+3H(\rho_M+p_M))=-Q$$, so $$\dot\rho_{\rm vac}=-Q$$ with $$\rho_{\rm vac}=\Lambda/8\pi G$$, which is (1). Consider matter whose reduced dynamics is a Gorini–Kossakowski–Sudarshan–Lindblad generator $$\dot\varrho=-i[H_M,\varrho]+\sum_k(L_k\varrho L_k^\dagger-\tfrac12\{L_k^\dagger L_k,\varrho\})$$ with self-adjoint $$L_k$$, as in spontaneous-collapse models.

<div class="proposition" markdown="1">
**Proposition 2** (heating drains the vacuum). For self-adjoint $$L_k$$ the heating rate satisfies $$\dot E=-\tfrac12\sum_k\langle[L_k,[L_k,H_M]]\rangle$$. For $$H_M=p^2/2m+V(x)$$ and $$L_k=L_k(x)$$ a function of position, it equals $$\dot E=\tfrac{\hbar^2}{2m}\sum_k\langle(\nabla L_k)^2\rangle\ge0$$ in every state. In unimodular gravity this gives $$\dot\rho_{\rm vac}\le0$$, hence $$w_{\rm vac}\ge-1$$. The map generated by self-adjoint $$L_k$$ is unital, so (c) of Theorem 1 gives the same sign for any passive state and any energy function. For massive particles undergoing Lorentz-invariant momentum diffusion on the mass shell $$\mathbb H^3$$, with energy $$E=m\cosh\chi$$ and diffusion $$\partial_\tau f=\kappa\Delta_{\mathbb H^3}f$$, $$\Delta_{\mathbb H^3}\cosh\chi=3\cosh\chi$$ gives $$\langle E\rangle^\cdot=3\kappa\langle E\rangle>0$$.
</div>

<div class="proof" markdown="1">
*Proof.* In the Heisenberg picture the dissipator acts on an operator $$O$$ as $$\sum_k(L_kOL_k-\tfrac12L_k^2O-\tfrac12OL_k^2)=-\tfrac12\sum_k[L_k,[L_k,O]]$$. For $$O=H_M$$ only the kinetic term fails to commute with $$L_k(x)$$. From $$[L,p_j]=i\hbar\,\partial_jL$$ one finds $$[L,p^2]=i\hbar(\nabla L\cdot p+p\cdot\nabla L)$$ and $$[L,[L,p^2]]=-2\hbar^2(\nabla L)^2$$. Hence $$[L,[L,H_M]]=-(\hbar^2/m)(\nabla L)^2$$ and the dissipator gives $$(\hbar^2/2m)(\nabla L)^2$$, a positive operator. The Laplacian on $$\mathbb H^3$$ acts on a radial function as $$f''+2\coth\chi\,f'$$, which for $$f=\cosh\chi$$ is $$3\cosh\chi$$. $$\square$$
</div>

A model with this property, and with the heating rate fixed by a collapse model, can only lower the vacuum energy. This is stated for $$w_{\rm vac}$$. The equation of state that an observer reconstructs is a different quantity, by Lemma 1: if the heated matter is counted as cold dark matter at a late anchor, the reconstruction has $$w_{\rm eff}<-1$$ at earlier times, as in [[DCK]](#ref-dck). The sign statement of the proposition therefore does not by itself exclude a reconstruction that looks phantom, only a phantom vacuum energy. The diffusion of massive particles in energy-momentum from spacetime discreteness is the model class of [[DPS]](#ref-dps); the most general Lorentz-invariant massless diffusion in that reference contains a drift in the energy of the particle in addition to the diffusion, and the sign of the drift is not fixed by Lorentz invariance. Energy diffusion from matter into the discrete structure, which is the sign of the proposition, is the scenario of [[PS]](#ref-ps). A drift of the opposite sign lies outside the proposition.

### What the theorem requires of the vacuum

The sign depends on $$w_v=-1$$. A scalar field with a kinetic term can reverse it. Take dust with mass $$m(\phi)=m_0e^{\beta\phi}$$ (with $$\phi$$ in Planck units) and a scalar field of negligible potential gradient. The matter action gives $$\rho_c=n\,m(\phi)$$ and the field equation $$\ddot\phi+3H\dot\phi+V'(\phi)=-\beta\rho_c$$. If $$V'=0$$ and $$\dot\phi=0$$ initially, then $$(a^3\dot\phi)^\cdot=-\beta a^3\rho_c$$ gives $$\beta\dot\phi=-(\beta^2/a^3)\int a^3\rho_c\,dt<0$$, so $$\dot m/m=\beta\dot\phi<0$$ for either sign of $$\beta$$: the particle mass falls with time and energy flows from matter into the kinetic energy of the field. Here the effective potential $$nm_0e^{\beta\phi}$$ has no minimum, so (a) does not apply, and the field has $$w=+1$$, so the vacuum is not of the type in section 1. Models of this class (coupled quintessence [[Amendola]](#ref-amendola)) that start away from the tracking branch, or that have no minimiser, are outside Theorem 1. A field with a potential minimum and a large mass tracks the minimum adiabatically and then falls under Theorem 1 up to corrections from its kinetic energy; we have not computed those corrections.

## 4. A field that tracks a minimum makes the electron lighter in the past

Fits of CMB and baryon acoustic oscillation (BAO) data in which the electron mass at recombination is free return central values above the laboratory one. Hydrogen recombines when the photon temperature reaches a fixed fraction of the binding energy, which is proportional to $$m_e$$; a heavier electron makes recombination earlier and shrinks the sound horizon. Write $$\delta\equiv m_e(z_*)/m_e^{\rm lab}-1$$. The published values are in the table.

| data | $$\delta$$ (%) |
|---|---|
| Planck + ACT DR6 + CMB lensing + DESI DR1 BAO [[ACT DR6]](#ref-act) | $$0.96\pm0.60$$ |
| the same + type Ia supernovae [[ACT DR6]](#ref-act) | $$0.63\pm0.56$$ |
| ACT DR6 + Planck ($$\ell<1000$$) + lensing + DESI DR2 BAO + supernovae [[Toda–Seto]](#ref-toda-seto) | $$0.81\pm0.46$$ |
| CMB + DESI DR2 BAO [[Weiner]](#ref-weiner) | $$0.76\pm0.48$$ |

The analyses in [[ACT DR6]](#ref-act) state that the time variation is not physical, in the sense that no field drives it; here a scalar $$\phi$$ on which $$m_e$$ depends supplies the dynamics.

Let $$m_e=m_e(\phi)$$ depend on a homogeneous scalar field with bare potential $$V(\phi)$$. Before the general statement, consider the simplest case.

<div class="example" markdown="1">
**Example 1.** Take $$m_e=m_0e^{\beta\phi/M_P}$$, with the reduced Planck mass $$M_P=(8\pi G)^{-1/2}$$, and $$V=\tfrac12m_\phi^2\phi^2$$ with $$m_\phi$$ large enough that the field stays at the minimum of its effective potential. Non-relativistic electrons of number density $$n$$ add the energy density $$n\,m_e(\phi)$$, so $$U(\phi)=\tfrac12m_\phi^2\phi^2+n\,m_0e^{\beta\phi/M_P}$$. At the minimum $$m_\phi^2\phi=-(\beta/M_P)\,n\,m_e$$, hence $$\phi$$ has the sign opposite to $$\beta$$ and
$$\ln(m_e/m_0)=\beta\phi/M_P=-\beta^2n\,m_e/(m_\phi^2M_P^2)<0.$$
The sign of the coupling drops out because it enters squared, and the denser the gas, the lighter the electron.
</div>

The field moves so as to lower the energy of the matter it is immersed in, and matter of given composition has lower energy when its constituents are lighter. The general statement needs the derivative of the free energy of a relativistic plasma with respect to a mass.

For one species of mass $$M$$ with $$g$$ internal states in kinetic equilibrium at temperature $$T$$, define $$\int\equiv g\int d^3k/(2\pi)^3$$, $$E=\sqrt{k^2+M^2}$$, and occupation numbers $$f$$. Then $$\rho=\int Ef$$ and $$p=\int\frac{k^2}{3E}f$$ (summed over particles and antiparticles when both are present), and we call $$n_{\rm sc}=\int(M/E)f=(\rho-3p)/M$$ the scalar density of the species, using $$E-k^2/E=M^2/E$$. It is positive wherever the species is present, and $$n_{\rm sc}\to n$$ for non-relativistic particles.

<div class="lemma" markdown="1">
**Lemma 2** (mass derivative of the free energy). Let $$\mathcal F(M;T,\mathcal N)$$ be the Helmholtz free energy density of an ideal gas at fixed temperature and fixed net number density $$\mathcal N$$ (particle minus antiparticle number for a gas that contains both). Then

$$
\left.\frac{\partial\mathcal F}{\partial M}\right\vert_{T,\mathcal N}=n_{\rm sc}=\frac{\rho-3p}{M}>0. \tag{4}
$$

For a mixture whose masses depend on $$m_e$$, $$\partial\mathcal F/\partial m_e=\sum_in_{{\rm sc},i}\,dM_i/dm_e$$, at fixed composition and at chemical equilibrium.
</div>

<div class="proof" markdown="1">
*Proof.* The grand potential density is $$-p(M;T,\mu_c)$$ with $$p=\pm gT\int\frac{d^3k}{(2\pi)^3}\ln(1\pm e^{-(E-\mu_c)/T})$$ for the particle species, plus the antiparticle term for $$-\mu_c$$. Differentiating under the integral with $$\partial E/\partial M=M/E$$ gives $$\partial p/\partial M\vert _{T,\mu_c}=-n_{\rm sc}$$. An integration by parts in $$k$$ turns the logarithm into the kinetic pressure $$\int k^2f/3E$$. The free energy is $$\mathcal F=-p+\mu_c\mathcal N$$ with $$\mu_c$$ fixed by $$\mathcal N$$; since $$\partial\mathcal F/\partial\mu_c\vert _{M}=-\partial p/\partial\mu_c+\mathcal N=0$$, the derivative at fixed $$\mathcal N$$ equals that of $$-p$$ at fixed $$\mu_c$$, which is $$n_{\rm sc}$$. For a mixture the chain rule gives the sum, and at chemical equilibrium the composition minimises $$\mathcal F$$, so by the envelope theorem its variation does not contribute. $$\square$$
</div>

As a numerical check of (4), we computed $$\mathcal F$$ for an ideal fermion gas with antiparticles at $$T=1$$, $$M=1.3$$ and net density $$0.4$$ (natural units) and differentiated numerically in $$M$$: $$\partial\mathcal F/\partial M=0.19735667$$, $$n_{\rm sc}=0.19735667$$ and $$(\rho-3p)/M=0.19735667$$.

In the cosmological plasma the species that depend on $$m_e$$ are free electrons and positrons, with $$dM/dm_e=1$$, and hydrogen and helium with bound electrons. A bound level of binding energy $$B\propto m_e$$ gives an atom the mass $$M=M_{\rm nucleus}+m_e-B$$ and $$dM/dm_e=1-B/m_e$$, which is $$1-2.7\times10^{-5}$$ for ground-state hydrogen ($$B=13.6$$ eV). Coulomb corrections to the ideal-gas free energy are neglected. We call $$\nu\equiv\partial\mathcal F/\partial m_e=\sum_in_{{\rm sc},i}\,dM_i/dm_e$$ the electron scalar density; by Lemma 2 it is positive wherever electrons are present.

Let state $$j$$ of the plasma be specified by its temperature, its conserved densities and its composition (ionisation fractions in equilibrium or frozen out). Let $$\mathcal F_j(\phi)$$ be its free energy density with masses evaluated at $$\phi$$ and $$U_j(\phi)=V(\phi)+\mathcal F_j(\phi)$$. This treats $$\phi$$ as a background parameter of the plasma; its derivative reproduces the source term $$-V'-\sum_in_{{\rm sc},i}\partial M_i/\partial\phi$$ in the field equation. Cosmological evolution conserves entropy rather than temperature, which does not change the stationarity condition, since $$\partial E/\partial M$$ at fixed entropy and particle number also equals $$n_{\rm sc}$$. The hypotheses are three.

(T1) Tracking: in each state $$j$$ compared, $$\phi$$ is at a global minimiser of $$U_j$$.

(T2) Standard Model sources only: apart from $$V$$, the $$\phi$$-dependence of $$U_j$$ comes from Standard Model species whose masses are non-decreasing functions of $$m_e$$. A field that changes only $$m_e$$ qualifies, and so does a Higgs portal provided every mass it changes increases with the Higgs vacuum value.

(T3) More matter in the past: at every $$m$$ between the two values of $$m_e$$ compared, $$\nu(m)$$ is larger in the earlier state than in the later.

<div class="theorem" markdown="1">
**Theorem 2** (electron mass sign). Let $$A$$ be an earlier state and $$B$$ a later one, and let $$\phi_A,\phi_B$$ be global minimisers of $$U_A,U_B$$. Under (T2) and (T3), $$m_e(\phi_A)\le m_e(\phi_B)$$. In particular, for the recombination plasma ($$A$$) and a laboratory environment ($$B$$) whose electron scalar density is smaller than that of the recombination plasma, $$\delta\le0$$.
</div>

<div class="proof" markdown="1">
*Proof.* This is Theorem 1(a) with $$X$$ the set of field values, $$g=V$$, $$\mu(\xi)=m_e(\phi)$$ and $$F_j=\mathcal F_j$$ as a function of $$m_e$$. Under (T2) the difference $$\mathcal F_B-\mathcal F_A$$ depends on $$\phi$$ only through $$m_e$$, and its derivative is $$\nu^{(B)}-\nu^{(A)}<0$$ by Lemma 2 and (T3). The conclusion of (a) is $$m_e(\phi_B)\ge m_e(\phi_A)$$. $$\square$$
</div>

Nothing in the proof uses the shape of $$V$$, the size of the coupling, or the number of components of $$\phi$$. The content is in Lemma 2, which fixes the sign of $$\partial\mathcal F/\partial m_e$$, and in (T3), which says which state has more of the matter that prefers a light electron. A weaker reading of tracking, a continuous branch of local minima, gives the same sign.

<div class="proposition" markdown="1">
**Proposition 3** (local tracking). Let $$\lambda\in[0,1]$$ label a continuous sequence of states from today ($$\lambda=0$$) back to an earlier epoch, and let $$\phi(\lambda)$$ be a $$C^1$$ branch of non-degenerate local minima of $$U_\lambda$$ for a single-component field. Under (T2), if $$\nu(m;\lambda)$$ does not decrease with $$\lambda$$ at fixed $$m$$, then $$m_e(\phi(\lambda))$$ is non-increasing in $$\lambda$$.
</div>

<div class="proof" markdown="1">
*Proof.* Differentiate $$U'_\lambda(\phi(\lambda))=0$$ along the branch: $$U''_\lambda\phi'+\partial_\lambda U'_\lambda=0$$. By Lemma 2 and (T2), $$U'_\lambda=V'+\nu(m_e(\phi);\lambda)\,m_e'(\phi)$$, so $$\partial_\lambda U'_\lambda=m_e'\partial_\lambda\nu$$ and $$dm_e/d\lambda=m_e'\phi'=-(m_e')^2\partial_\lambda\nu/U''_\lambda\le0$$. $$\square$$
</div>

The branch can end where $$U''_\lambda\to0$$, after which the field jumps; Theorem 2 covers jumps between global minima.

Hypothesis (T3) holds in the cosmological plasma. After electron–positron annihilation the net electron number is conserved, so the number density scales as $$(1+z)^3$$ and the scalar density is $$n_{\rm sc}=n\langle m_e/E\rangle\approx n(1-3T/2m_e)$$; at recombination $$T/m_e\approx5\times10^{-7}$$, so the correction is negligible against the factor $$(1+z)^3$$. Before and during annihilation the pairs add to it. We checked this for an ideal electron–positron gas with the baryon-to-photon ratio $$\eta=273.9\,\Omega_bh^2\times10^{-10}=6.1\times10^{-10}$$ (with $$\Omega_bh^2=0.02237$$ [[Planck VI]](#ref-planck6)) and helium fraction $$Y_p=0.245$$ (so $$1-Y_p/2$$ electrons per baryon): the scalar density at $$m=m_e$$ (we did not scan $$m$$) increases monotonically with temperature along 60 temperatures from $$T_0=2.7255$$ K to 10 MeV. The mean electron density at last scattering is then $$n_e(z_*=1090)\approx2.9\times10^2\ {\rm cm}^{-3}$$ (the mean today is $$2.6\times10^{-7}\,{\rm cm}^{-3}$$ for the same inputs).

### Which environment is today

The theorem compares tracked values in two homogeneous states. The laboratory value is the tracked value for whatever environment the field responds to there. Theorem 2 gives $$\delta\le0$$ provided that environment has an electron scalar density below $$n_e(z_*)\approx290\ {\rm cm}^{-3}$$. This holds if the field responds to the cosmic mean or to the mean density of the Galaxy, which near the Sun is a few nucleons per cubic centimetre. It fails if the field tracks the density of the Earth or of the laboratory, as screened fields of chameleon [[KW]](#ref-chameleon) or symmetron type [[HK]](#ref-symmetron) do. Applied to pairs of environments of similar composition, the theorem orders them by electron scalar density: $$m_e$$ in any environment less dense than the recombination plasma is at least as large as at recombination.

<div class="corollary" markdown="1">
**Corollary 2.** Assume (T1) to (T3) and that the laboratory value of $$m_e$$ is set by an environment less dense than the recombination plasma. Then every field that changes $$m_e$$ gives $$\delta\le0$$. This includes couplings of symmetron or chameleon type to the electron density with either sign of the coupling, the response of the field to the trace $$\rho-3p$$ of the electron–positron plasma during annihilation, and a Higgs portal under the proviso in (T2).
</div>

The fits in the table place $$\delta=0$$ between $$1.1$$ and $$1.8$$ standard deviations below their central values ($$0.96/0.60=1.6$$, $$0.63/0.56=1.1$$, $$0.81/0.46=1.8$$, $$0.76/0.48=1.6$$), so Theorem 2 is in tension with the central values at a level that is not decisive; the fits impose no field dynamics. Its content is the sign, which would be significant only if the preference for $$\delta>0$$ strengthened.

## 5. A field that does not track: the excursion bound

A light field that is frozen at early times and released later can move in either direction, so Theorem 2 does not apply. The obstruction is then different: the field must still arrive at today's value, and if its coupling to electrons is the same along the way, that value is constrained by tests of the equivalence principle.

Write $$\beta_e\equiv M_P\,d\ln m_e/d\phi$$ today. If $$\phi$$ is light enough to mediate a force over the size of the Earth, a body $$b$$ of mass $$M_b$$ containing $$N_b$$ electrons has scalar charge $$Q_b=\partial M_b/\partial\phi=\beta_eN_bm_e/M_P$$, when only the electron rest mass depends on $$\phi$$. Scalar exchange between bodies $$b$$ and $$c$$ gives an attractive force $$Q_bQ_c/(4\pi r^2)$$, gravity gives $$M_bM_c/(8\pi M_P^2r^2)$$, and the ratio is $$2\beta_e^2f_bf_c$$ with $$f_b=N_bm_e/M_b\approx(m_e/m_u)(Z/A)_b$$, where $$Z$$ is the atomic number and $$A$$ the mass number. The Eötvös parameter for two test masses falling towards the Earth ($$E$$) is

$$
\eta\simeq2\beta_e^2f_E(f_{\rm Ti}-f_{\rm Pt}). \tag{5}
$$

The MICROSCOPE experiment measured $$\eta({\rm Ti,Pt})=[-1.5\pm2.3\,({\rm stat})\pm1.5\,({\rm syst})]\times10^{-15}$$ [[MICROSCOPE]](#ref-microscope). We use $$\lvert\eta\rvert<5.5\times10^{-15}$$, twice the quadrature sum of the errors, which ignores the negative central value. With $$m_e/m_u=5.486\times10^{-4}$$, $$(Z/A)_{\rm Ti}=22/47.867=0.4596$$, $$(Z/A)_{\rm Pt}=78/195.084=0.3998$$ and a bulk-Earth value $$(Z/A)_E\approx0.487$$ (an assumed input), (5) gives $$\lvert\beta_e\rvert<5.6\times10^{-4}$$ at this level, and $$4.0\times10^{-4}$$ at one sigma (our computation). Varying $$(Z/A)_E$$ between $$0.48$$ and $$0.50$$ changes the bound by less than 2%; the bound scales as $$\sqrt{\eta_{\max}}$$.

If $$\lvert d\ln m_e/d\phi\rvert$$ never exceeds its present value along the path, then $$\ln(1+\delta)\le\lvert\beta_e\Delta\phi\rvert/M_P$$ and the excursion must satisfy $$\lvert\Delta\phi\rvert/M_P\ge\ln(1+\delta)/(5.6\times10^{-4})\approx18\,\delta/(1\%)$$. A field cannot move arbitrarily far. Its kinetic energy density is $$\dot\phi^2/2$$, and its share of the total is $$\Omega_{\rm kin}=\dot\phi^2/(6H^2M_P^2)$$. For a flat or negatively curved universe whose other components have non-negative energy density the Friedmann equation gives $$\Omega_{\rm kin}\le1$$. With $$N=\ln a$$ as the time variable, $$d\phi/dN=\pm\sqrt{6\Omega_{\rm kin}}\,M_P$$, and over the $$N_*=\ln(1+z_*)\approx\ln1090=6.99$$ e-folds from last scattering to today, the Cauchy–Schwarz inequality gives

$$
\frac{\lvert\Delta\phi\rvert}{M_P}\le\int_0^{N_*}\sqrt{6\Omega_{\rm kin}}\,dN\le N_*\sqrt{6\langle\Omega_{\rm kin}\rangle}=17.1\sqrt{\langle\Omega_{\rm kin}\rangle}, \tag{6}
$$

with $$\langle\Omega_{\rm kin}\rangle$$ the average of $$\Omega_{\rm kin}$$ over e-folds.

<div class="theorem" markdown="1">
**Theorem 3** (excursion bound). Suppose $$m_e$$ depends on one field $$\phi$$ whose range is at least the size of the Earth, with $$\lvert d\ln m_e/d\phi\rvert\le\lvert\beta_e\rvert/M_P$$ along the path from last scattering to today, $$\lvert\beta_e\rvert<5.6\times10^{-4}$$ at the level above, the universe is flat or negatively curved and other components have non-negative energy density. Then $$\delta\le\exp(5.6\times10^{-4}\times17.1\sqrt{\langle\Omega_{\rm kin}\rangle})-1\approx0.96\%\times\sqrt{\langle\Omega_{\rm kin}\rangle}$$.
</div>

The proof is the chain of the two inequalities above. The table shows what the central values of the fits would require. The required $$\langle\Omega_{\rm kin}\rangle$$ is $$(\lvert\Delta\phi\rvert/17.1)^2$$ (our computation).

| $$\delta$$ (%) | required $$\lvert\Delta\phi\rvert/M_P$$ | required $$\langle\Omega_{\rm kin}\rangle$$ |
|---|---|---|
| 0.96 | 17.1 | 0.99 |
| 0.81 | 14.4 | 0.71 |
| 0.76 | 13.5 | 0.62 |
| 0.63 | 11.2 | 0.43 |

A required $$\langle\Omega_{\rm kin}\rangle$$ above 1 is impossible under the stated assumptions, so a value of $$\delta$$ above $$0.96\%$$ is excluded at this level for the class. The central values in the table need the field's kinetic energy to carry between $$40\%$$ and $$100\%$$ of the total energy density on average over the e-folds since recombination, in an interval that the CMB and BAO describe as dominated by matter. If the kinetic energy is confined to the share of a cosmological constant, $$\Omega_{\rm kin}\le\Omega_\Lambda(z)$$ with $$\Omega_m=0.3$$, the middle expression of (6) gives $$\lvert\Delta\phi\rvert\le1.97M_P$$ and $$\delta\le0.11\%$$ (heuristic, for a $$\Lambda$$CDM history).

## 6. What the theorems do not exclude

Each item drops a hypothesis, and states what remains.

Fields that start off the tracking branch, or that have no minimiser. The rolling field of section 3 pushes the particle mass down for either sign of the coupling, so that $$\omega_c(\mathrm{rec})/\omega_c(0)>1$$ is produced by a coupled scalar with kinetic energy [[Amendola]](#ref-amendola). The vacuum is then not of the type in Theorem 1, since $$w\ne-1$$. The required coupling and the cost in the fits to the CMB are not computed here.

Active media. A negative $$\Gamma$$ or a concave $$g+F$$ (population inversion, a negative temperature) reverses the sign; the random runs above flip in 30 of 30 cases. The generalised second law and the source of the free energy then have to be specified.

Hysteresis. A field that follows a local minimum to a spinodal and then jumps is outside the global-minimum argument. In a scan of 2000 random one-dimensional models with the structure of the check above, 377 had a spinodal jump along the local-minimum branch; the jump lowered $$\mu$$ in one of them, and the net change of $$\mu$$ from the start to the end of the history was never negative. A proof for this case is missing.

Non-linear passive relaxation. Theorem 1(b) is proved for a constant Hessian. Relaxation $$\dot\xi=-\Gamma(\xi,t)\nabla\Phi_t$$ with a non-quadratic $$\Phi_t$$ was not tested here and is not covered.

Matter creation with a chemical potential [[CCL]](#ref-ccl), non-unital maps, and reservoirs at negative temperature are outside (c).

Fields that do not track and have a quadratic coupling. For $$m_e=m_0(1+\phi^2/2M_e^2)$$ with $$\phi=0$$ today there is no linear coupling and (5) is empty. A displacement from zero cannot come from a Standard Model source, since any displacement raises $$m_e$$, which Theorem 2 forbids. A positive $$\delta=\phi_i^2/2M_e^2$$ requires a field held away from zero by Hubble friction that later oscillates about zero. This class is studied in [[BSW 1]](#ref-bsw1), [[BSW 2]](#ref-bsw2) and [[GBY]](#ref-gby). The oscillating field is matter-like at late times, and its residual oscillation modulates $$m_e$$ in time, which atomic clocks constrain; we have not computed those limits.

Fields also sourced by dark matter. If $$\phi$$ also changes the mass $$m_X$$ of a non-relativistic dark-matter particle, (T2) fails but the comparison inequality in the proof of Theorem 1(a) still holds under (T1). With $$\mathcal F_j=n_{e,j}m_e+n_{X,j}m_X$$ and both densities scaling as $$(1+z)^3$$, it reads $$-\Delta m_X/m_X\ge\delta\,\rho_{e,0}/\rho_{X,0}$$ with $$\rho_{e,0}/\rho_{X,0}=(1-Y_p/2)(m_e/m_p)\,\Omega_bh^2/\Omega_ch^2=8.9\times10^{-5}$$. A heavier electron at recombination then requires the dark matter to have been lighter by a fraction of at least $$8.9\times10^{-5}\delta$$, that is $$7\times10^{-7}$$ at $$\delta=0.81\%$$. A field of this kind whose coupling saturates at high density would produce a step in $$m_e$$; whether such a model fits the full CMB, and its consequences for nucleosynthesis, are open.

Fields that track the local density today. Theorem 2 then compares the recombination plasma with a far denser environment and allows $$\delta>0$$. Spectroscopy of absorbing gas measures $$m_p/m_e$$ in other environments: methanol absorption in the lens towards PKS1830−211 at $$z=0.88582$$ limits its fractional change to $$4\times10^{-7}$$ at $$2\sigma$$ [[Kanekar]](#ref-kanekar), and ammonia and cyanopolyyne lines in Galactic molecular cores limit it to $$2\times10^{-8}$$ at $$3\sigma$$ [[Levshakov]](#ref-levshakov), both far below $$\delta\sim10^{-2}$$. They bear on this route when the absorbing gas has an electron density below about $$290\ {\rm cm}^{-3}$$ counting bound electrons; molecular clouds are typically denser, and then the theorem only gives $$m_e\le m_e(z_*)$$ there. Whether such a field can satisfy fifth-force and equivalence-principle bounds has not been checked.

## 7. Open points

Theorem 1(b) assumes a constant Hessian; the nonlinear case, hysteresis, and a proof that includes a propagating but heavy auxiliary field are open. The condition (T3) was verified for an ideal plasma only; Coulomb corrections and the full recombination history were not included. The excursion bound assumes that only the electron mass depends on $$\phi$$ and that the coupling does not grow along the path; a path along which $$d\ln m_e/d\phi$$ grows from a small value today interpolates between Theorem 3 and the quadratic class, and was not scanned. The numerical values of $$\delta$$ come from fits that vary $$m_e$$ by hand at recombination, with no field dynamics, and a field model changes $$m_e$$ at other epochs, in particular during nucleosynthesis, and adds its own energy density; fitting any of the models of section 6 to the data needs all of this. The vacuum-energy analogue of the sign statement, for constraints that fix the cosmological constant globally, is on the page on [linear global constraints](/quantum-gravity/linear-lambda-constraints/), where the monotone decrease along worldlines rests on a positivity argument of a different kind and is not an instance of Theorem 1. The conditions under which a dark-sector component can mimic a lighter or negative neutrino mass in distance data are the subject of [the neutrino-mimic page](/quantum-gravity/neutrino-mimic/). The place of these results among the other approaches is on the [landscape page](/quantum-gravity/landscape/).

## References

1. <a id="ref-pw"></a>Pavón, Wang, “Le Chatelier–Braun principle in cosmological physics,” Gen. Relativ. Gravit. 41, 1 (2009), [arXiv:0712.0565](https://arxiv.org/abs/0712.0565).
2. <a id="ref-fnw"></a>Fardon, Nelson, Weiner, “Dark Energy from Mass Varying Neutrinos,” JCAP 0410, 005 (2004), [arXiv:astro-ph/0309800](https://arxiv.org/abs/astro-ph/0309800).
3. <a id="ref-azk"></a>Afshordi, Zaldarriaga, Kohri, “On the stability of Dark Energy with Mass-Varying Neutrinos,” Phys. Rev. D 72, 065024 (2005), [arXiv:astro-ph/0506663](https://arxiv.org/abs/astro-ph/0506663).
4. <a id="ref-bft1"></a>Bean, Flanagan, Trodden, “The Adiabatic Instability on Cosmology's Dark Side,” New J. Phys. 10, 033006 (2008), [arXiv:0709.1124](https://arxiv.org/abs/0709.1124).
5. <a id="ref-bft2"></a>Bean, Flanagan, Trodden, “Adiabatic instability in coupled dark energy–dark matter models,” Phys. Rev. D 78, 023009 (2008), [arXiv:0709.1128](https://arxiv.org/abs/0709.1128).
6. <a id="ref-dck"></a>Das, Corasaniti, Khoury, “Super-acceleration as Signature of Dark Sector Interaction,” Phys. Rev. D 73, 083509 (2006), [arXiv:astro-ph/0510628](https://arxiv.org/abs/astro-ph/0510628).
7. <a id="ref-amendola"></a>Amendola, “Coupled Quintessence,” Phys. Rev. D 62, 043511 (2000), [arXiv:astro-ph/9908023](https://arxiv.org/abs/astro-ph/9908023).
8. <a id="ref-topkis"></a>Topkis, “Minimizing a Submodular Function on a Lattice,” Operations Research 26, 305 (1978).
9. <a id="ref-ms"></a>Milgrom, Shannon, “Monotone Comparative Statics,” Econometrica 62, 157 (1994).
10. <a id="ref-pw78"></a>Pusz, Woronowicz, “Passive states and KMS states for general quantum systems,” Commun. Math. Phys. 58, 273 (1978).
11. <a id="ref-lenard"></a>Lenard, “Thermodynamical proof of the Gibbs formula for elementary quantum systems,” J. Stat. Phys. 19, 575 (1978).
12. <a id="ref-jps"></a>Josset, Perez, Sudarsky, “Dark energy as the weight of violating energy conservation,” Phys. Rev. Lett. 118, 021102 (2017), [arXiv:1604.04183](https://arxiv.org/abs/1604.04183).
13. <a id="ref-dps"></a>Dowker, Philpott, Sorkin, “Energy-momentum diffusion from spacetime discreteness,” Phys. Rev. D 79, 124047 (2009), [arXiv:0810.5591](https://arxiv.org/abs/0810.5591).
14. <a id="ref-ps"></a>Perez, Sudarsky, “Dark energy from quantum gravity discreteness,” Phys. Rev. Lett. 122, 221302 (2019), [arXiv:1711.05183](https://arxiv.org/abs/1711.05183).
15. <a id="ref-ccl"></a>Cárdenas, Cruz, Lepe, “Generalized second law of thermodynamics for the matter creation scenario and emergence of phantom regime,” Eur. Phys. J. Plus 139, 642 (2024), [arXiv:2302.10155](https://arxiv.org/abs/2302.10155).
16. <a id="ref-desi"></a>DESI Collaboration, “DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints,” Phys. Rev. D 112, 083515 (2025), [arXiv:2503.14738](https://arxiv.org/abs/2503.14738).
17. <a id="ref-planck6"></a>Planck Collaboration, “Planck 2018 results. VI. Cosmological parameters,” Astron. Astrophys. 641, A6 (2020), [arXiv:1807.06209](https://arxiv.org/abs/1807.06209).
18. <a id="ref-act"></a>Calabrese et al. (ACT Collaboration), “The Atacama Cosmology Telescope: DR6 Constraints on Extended Cosmological Models,” [arXiv:2503.14454](https://arxiv.org/abs/2503.14454).
19. <a id="ref-toda-seto"></a>Toda, Seto, “Constraints on the varying electron mass and early dark energy in light of ACT DR6 and DESI DR2 and the implications for inflation,” JCAP 2026, 019, [arXiv:2508.09025](https://arxiv.org/abs/2508.09025).
20. <a id="ref-weiner"></a>Weiner, “High-redshift physics from the acoustic scale,” JCAP 08 (2026) 072, [arXiv:2603.18131](https://arxiv.org/abs/2603.18131).
21. <a id="ref-bsw1"></a>Baryakhtar, Simon, Weiner, “Cosmology with varying fundamental constants from hyperlight, coupled scalars,” Phys. Rev. D 110, 083505 (2024), [arXiv:2405.10358](https://arxiv.org/abs/2405.10358).
22. <a id="ref-bsw2"></a>Baryakhtar, Simon, Weiner, “Searching for coupled, hyperlight scalars across cosmic history,” Phys. Rev. D 111, 115026 (2025), [arXiv:2502.04432](https://arxiv.org/abs/2502.04432).
23. <a id="ref-gby"></a>Ghosh, Boddy, Yu, “Early Universe Constraints on Variations in Fundamental Constants Induced by Ultralight Scalar Dark Matter,” [arXiv:2511.14532](https://arxiv.org/abs/2511.14532).
24. <a id="ref-microscope"></a>Touboul et al. (MICROSCOPE Collaboration), “MICROSCOPE mission: final results of the test of the Equivalence Principle,” Phys. Rev. Lett. 129, 121102 (2022), [arXiv:2209.15487](https://arxiv.org/abs/2209.15487).
25. <a id="ref-chameleon"></a>Khoury, Weltman, “Chameleon Fields: Awaiting Surprises for Tests of Gravity in Space,” Phys. Rev. Lett. 93, 171104 (2004), [arXiv:astro-ph/0309300](https://arxiv.org/abs/astro-ph/0309300).
26. <a id="ref-symmetron"></a>Hinterbichler, Khoury, “Symmetron Fields: Screening Long-Range Forces Through Local Symmetry Restoration,” Phys. Rev. Lett. 104, 231301 (2010), [arXiv:1001.4525](https://arxiv.org/abs/1001.4525).
27. <a id="ref-kanekar"></a>Kanekar et al., “Constraints on changes in the proton-electron mass ratio using methanol lines,” [arXiv:1412.7757](https://arxiv.org/abs/1412.7757).
28. <a id="ref-levshakov"></a>Levshakov et al., “Limits on the spatial variations of the electron-to-proton mass ratio in the Galactic plane,” [arXiv:1310.1850](https://arxiv.org/abs/1310.1850).
