---
title: The sign of energy exchange between vacuum and matter
date: 2026-10-29
category: dark-sector
order: 2
description: A vacuum energy without kinetic term that tracks the minimum of its free energy, relaxes passively towards it, or exchanges energy with matter through a unital map on a passive state transfers energy to matter, and a scalar that tracks its minimum in a Standard Model plasma makes the electron lighter in the denser past provided the laboratory value is set by an environment less dense than the recombination plasma.
status: Theorems proved under the stated hypotheses (global-minimum tracking, linear passive relaxation with constant Hessian, unital maps on passive states, Standard Model sources with more matter in the past). The monotonicity of the electron scalar density in the early plasma was checked numerically for an ideal gas. The excursion bound uses the 2 sigma range of the MICROSCOPE result and assumes that the electron coupling does not grow along the path.
excludes: Mass-coupled vacuum-matter models that follow the global minimum of a free energy or relax passively towards it, as an explanation of a particle mass that decreases with time; a scalar that follows the minimum of its effective potential in a Standard Model plasma, as the origin of a heavier electron at recombination, when the laboratory value is set by an environment less dense than the recombination plasma; a non-tracking scalar with an electron coupling that never exceeds the MICROSCOPE limit, as the origin of a shift above about 1.1 per cent.
---

Some analyses of cosmological data favour a dark-energy equation of state with $$w_0>-1$$ and $$w_a<0$$, which lies below $$-1$$ at high redshift when $$w_0+w_a<-1$$, and an electron mass at recombination between 0.6 and 1 per cent above the laboratory value. With dark matter conserved, the first pattern asks for energy to flow from matter into the vacuum during part of the history after recombination, and the second for a particle mass that falls with time. Which dynamics of a vacuum component can do this? Three classes give the opposite sign: a vacuum without kinetic energy that tracks the global minimum of its free energy, one that relaxes towards that minimum under a passive linear dynamics, and one that exchanges energy with matter through a unital map acting on a passive state of the matter. In each, energy flows from the vacuum to the matter, in the sense made precise in Theorem 1. For electrons the same sign holds for any scalar that tracks its minimum in a Standard Model plasma, provided the laboratory value is set by an environment less dense than the recombination plasma. A field that does not track is limited by equivalence-principle tests to a shift of about $$1.1\%\sqrt{\langle\Omega_{\rm kin}\rangle}$$, where $$\langle\Omega_{\rm kin}\rangle\le1$$ is its average kinetic share of the energy density.

Pavón and Wang argued from the Le Chatelier–Braun principle that energy flows from dark energy to dark matter [[PW]](#ref-pw). Fardon, Nelson and Weiner considered neutrinos whose mass is set by a field at the minimum of its effective potential in the neutrino background [[FNW]](#ref-fnw); the gradient instability that accompanies such a field was found for neutrinos by Afshordi, Zaldarriaga and Kohri [[AZK]](#ref-azk) and for coupled dark matter by Bean, Flanagan and Trodden [[BFT 1]](#ref-bft1), [[BFT 2]](#ref-bft2). Das, Corasaniti and Khoury showed that a reconstruction with conserved matter can turn an interacting vacuum into an apparent phantom component [[DCK]](#ref-dck). Baryakhtar, Simon and Weiner computed the in-medium force on a hyperlight scalar that modulates the electron mass [[BSW 1]](#ref-bsw1), [[BSW 2]](#ref-bsw2). The comparison argument below is monotone comparative statics [[Topkis]](#ref-topkis), [[MS]](#ref-ms). What is added is one comparison that covers tracking, passive relaxation and unital exchange, a version of the electron-mass argument that holds for any potential and any number of sources, and the excursion bound for a linearly coupled field that does not track. The analogous question for the cosmological constant under linear global constraints is treated on the page on [linear global constraints](/quantum-gravity/linear-lambda-constraints/), where the monotone decrease along worldlines rests on a positivity argument of a different kind, and the question of a dark-sector component mimicking a lighter neutrino on the page on the [neutrino mimic](/quantum-gravity/neutrino-mimic/).

## 1. Vacuum, matter and what a reconstruction sees

We work in a spatially flat Friedmann–Robertson–Walker universe with scale factor $$a(t)$$, Hubble rate $$H=\dot a/a>0$$, and signature $$(-,+,+,+)$$. A component with energy density $$\rho$$ and pressure $$p$$ that exchanges no energy with others obeys $$\dot\rho+3H(\rho+p)=0$$. The vacuum component has $$T^V_{ab}=-\rho_vg_{ab}$$, so $$p_v=-\rho_v$$, and it is non-propagating: it has no kinetic term, and its energy density is a function of auxiliary variables that carry no energy of their own. The matter component is dust, with conserved particle number density $$n$$ ($$na^3$$ constant) and mass per particle $$\mu$$, so that $$\rho_M=n\mu$$. Total energy is conserved, which defines the exchange rate $$Q$$:

$$
\dot\rho_v=-Q,\qquad \dot\rho_M+3H\rho_M=Q. \tag{1}
$$

A positive $$Q$$ means that energy flows from the vacuum to matter. For dust, $$Q=n\dot\mu$$, so $$Q\ge0$$ is the statement that the particle mass does not decrease. Where the transferred energy is deposited as heat (Theorem 1(b) and (c), Proposition 2), $$Q$$ is still defined by $$Q=-\dot\rho_v$$, and the second relation in (1) holds if the heat is counted as dust-like energy that dilutes as $$a^{-3}$$; the statements below say which reading they use.

A cosmological analysis typically assumes that cold dark matter is conserved, fixes its density at some epoch $$a_*$$, and attributes the rest of the energy budget to dark energy. We call the result the reconstruction anchored at $$a_*$$: $$\rho_c^{\rm rec}(a)=\rho_M(a_*)(a_*/a)^3$$ and $$\rho_{\rm DE,eff}=\rho_v+\rho_M-\rho_c^{\rm rec}$$. Early anchors correspond to normalising the dark-matter density to the cosmic microwave background (CMB), late anchors to a low-redshift measurement.

<div class="lemma" markdown="1">
**Lemma 1** (reconstruction). Under (1), with $$\Delta(t)=a^{-3}\int_{t_*}^{t}Q\,a^3\,dt'$$,

$$
\dot\rho_{\rm DE,eff}=-3H\Delta,\qquad 1+w_{\rm eff}=\frac{\Delta}{\rho_{\rm DE,eff}}, \tag{2}
$$

where $$w_{\rm eff}$$ is defined by $$\dot\rho_{\rm DE,eff}=-3H(1+w_{\rm eff})\rho_{\rm DE,eff}$$ and $$\rho_{\rm DE,eff}>0$$. If $$Q\ge0$$ at all times, then for an anchor in the past of $$t$$ ($$t_*<t$$) $$\Delta\ge0$$ and $$w_{\rm eff}\ge-1$$, and for an anchor in the future of $$t$$ ($$t_*>t$$) $$\Delta\le0$$ and $$w_{\rm eff}\le-1$$. The comoving dark-matter densities satisfy $$(a^3\rho_M)(t_1)\le(a^3\rho_M)(t_2)$$ for $$t_1<t_2$$.
</div>

<div class="proof" markdown="1">
*Proof.* From (1), $$a^3\rho_M$$ has derivative $$a^3Q$$, so $$\rho_M-\rho_c^{\rm rec}=\Delta$$. Then $$\rho_{\rm DE,eff}=\rho_v+\Delta$$ and $$\dot\rho_{\rm DE,eff}=-Q+(-3H\Delta+Q)=-3H\Delta$$. The sign statements follow from the sign of the integral over $$[t_*,t]$$, which is negative if the limits are reversed. The last statement integrates $$a^3Q\ge0$$. $$\square$$
</div>

The sign statements use only the sign of $$\int_{t_*}^tQ\,a^3dt'$$, not the sign of $$Q$$ at each time. A value $$w_{\rm eff}<-1$$ at time $$t$$ in a fit anchored at $$t_*<t$$ is, by (2), the same as $$\Delta(t)<0$$: the comoving matter density at $$t$$ is lower than at the anchor, so $$Q<0$$ on part of $$[t_*,t]$$. With a late anchor, a vacuum that feeds matter looks phantom in the past, the effect described in [[DCK]](#ref-dck), so a comparison with observations needs the anchor to be stated. The dark-energy preference of the DESI second data release combined with CMB and supernova data is a preference for $$w_0>-1$$ and $$w_a<0$$ in $$w(a)=w_0+w_a(1-a)$$, at $$2.8$$–$$4.2\sigma$$ over a cosmological constant depending on the supernova sample [[DESI]](#ref-desi). The question for the next section is whether the physics of an exchange can produce the corresponding reconstruction.

## 2. The thermodynamic sign theorem

We start with the simplest case. Take a single auxiliary variable $$\xi$$, a vacuum energy $$g(\xi)=\tfrac12h\xi^2$$ with $$h>0$$, and dust of particle number density $$n$$ whose particle mass is $$\mu(\xi)=\mu_0+b\,\xi$$. The energy density available to $$\xi$$ is $$\Phi=\tfrac12h\xi^2+n(\mu_0+b\xi)$$. Its minimum is at $$\xi=-nb/h$$ and the mass there is $$\mu=\mu_0-nb^2/h$$. Matter lowers its energy by making its particles lighter, so the denser the dust, the lighter the particles; as the universe expands $$n$$ falls and $$\mu$$ rises towards $$\mu_0$$. The vacuum energy $$g=\tfrac12 n^2b^2/h$$ falls with $$n$$ as well, and the difference is paid to matter: with $$\rho_M=n\mu$$ and $$na^3$$ constant, $$Q=n\dot\mu=3Hn^2b^2/h>0$$. The sign does not depend on the sign of $$b$$.

The general argument replaces the quadratic and the linear coupling by arbitrary functions. Its first step compares two states. Let $$g:X\to\mathbb R$$ and $$F_1,F_2:X\to\mathbb R$$ be arbitrary, and let $$\xi_j$$ be a global minimiser of $$\Phi_j=g+F_j$$. Adding $$\Phi_1(\xi_1)\le\Phi_1(\xi_2)$$ and $$\Phi_2(\xi_2)\le\Phi_2(\xi_1)$$ cancels $$g$$ and leaves

$$
D(\xi_2)\le D(\xi_1),\qquad D=F_2-F_1. \tag{3}
$$

The later minimiser lowers the difference of the matter energies, whatever $$g$$ is. No convexity of $$g$$ or of $$\Phi_j$$ enters, and the two minimisers may lie far apart in $$X$$. The theorem turns (3) into a statement about the mass, for the situations in which a minimiser is reached.

<div class="theorem" markdown="1">
**Theorem 1** (thermodynamic sign). Let $$\xi$$ take values in a set $$X$$, let the vacuum energy density be $$\rho_v=g(\xi)$$ with $$g:X\to\mathbb R$$ arbitrary, and let the matter depend on $$\xi$$ only through a mass parameter $$\mu(\xi)>0$$, an arbitrary function of $$\xi$$. The energy density of the matter in state $$j$$, at the entropy and particle numbers that state fixes, is $$F_j(\mu)$$, differentiable, with $$\partial_\mu F_j>0$$; for dust of conserved particle number density $$n_j$$ it is $$F_j(\mu)=n_j\mu$$, and section 4 gives it for a plasma. We call $$\Phi_j(\xi)=g(\xi)+F_j(\mu(\xi))$$ the effective potential in state $$j$$.

(a) Global tracking. Let state 1 be earlier and state 2 later, let $$\xi_1,\xi_2$$ be global minimisers of $$\Phi_1,\Phi_2$$, and let $$J$$ be an interval containing $$\mu(\xi_1)$$ and $$\mu(\xi_2)$$ on which $$\partial_\mu F_2<\partial_\mu F_1$$ (for dust, $$n_2<n_1$$). Then $$\mu(\xi_2)\ge\mu(\xi_1)$$. For dust, a field that follows the global minimiser of $$\Phi_t$$ through an expansion has $$a^3n\mu$$ non-decreasing, also across first-order jumps, so $$Q\ge0$$ (as a measure).

(b) Passive linear relaxation. Let $$X=\mathbb R^r$$, $$g(\xi)=\tfrac12\xi^{\mathsf T}K\xi$$ with $$K$$ symmetric positive definite and constant, $$\mu(\xi)=\mu_0+b^{\mathsf T}\xi$$, dust with $$n(t)$$ non-increasing, and $$\dot\xi=-\Gamma(t)\nabla\Phi_t(\xi)$$ with $$\Gamma(t)$$ any matrix function whose symmetric part is positive semi-definite (it may be non-reciprocal and time dependent). If $$\xi$$ starts at the minimiser of $$\Phi_{t_0}$$, then $$\mu(t)\ge\mu(t_0)$$ for all $$t\ge t_0$$, the dissipation $$\Pi=\nabla\Phi^{\mathsf T}\Gamma\nabla\Phi$$ is non-negative, and $$\int_{t_0}^ta^3Q\,dt'\ge0$$ with $$Q=-\dot\rho_v=n\dot\mu+\Pi$$.

(c) Unital exchange on a passive state. Let matter be a finite-dimensional quantum system with energy operator $$\hat E$$ and state $$\varrho$$, and let the exchange over a time step act as a unital completely positive trace-preserving map $$\mathcal T$$ on $$\varrho$$ ($$\mathcal T(\mathbb 1)=\mathbb 1$$; random unitary mixtures and dephasing are examples), jointly energy conserving with the vacuum. If $$\varrho$$ is passive, that is diagonal in an energy eigenbasis with occupations non-increasing in energy (every thermal state at positive temperature is of this kind), then $$\mathrm{Tr}[\hat E\,\mathcal T(\varrho)]\ge\mathrm{Tr}[\hat E\varrho]$$, and the vacuum loses the corresponding energy.
</div>

In (a) the conclusion is (3) read through the hypothesis on the derivative. In (b) the mass responds to a step in the source with a contraction that cannot overshoot its equilibrium by more than it started away from it, and a decreasing source is a positive superposition of steps. In (c) a unital map can only mix, and mixing raises the energy of a passive state.

<div class="proof" markdown="1">
*Proof of (a).* Apply (3) with $$F_j$$ evaluated at $$\mu(\xi)$$: $$G(\mu_2)\le G(\mu_1)$$ for $$G=F_2-F_1$$ and $$\mu_j=\mu(\xi_j)$$. By hypothesis $$G'<0$$ on $$J$$, so $$G$$ is strictly decreasing there, and $$G(\mu_2)\le G(\mu_1)$$ forces $$\mu_2\ge\mu_1$$. For dust, $$G(\mu)=(n_2-n_1)\mu$$. For the exchange rate, total energy is $$\rho_v+\rho_M=g+n\mu=\Phi_t(\xi(t))$$, the minimum of $$\Phi_t$$, and it is continuous across a jump between two degenerate minima. With $$na^3$$ constant, $$a^3\rho_M=a^3n\mu$$ is non-decreasing by the first part, so $$Q\ge0$$ in the sense of measures. $$\square$$

*Proof of (b).* Put $$y=K^{1/2}\xi$$, $$c=K^{-1/2}b$$ and $$A(t)=K^{1/2}\Gamma(t)K^{1/2}$$. The potential is $$\Phi_t=\tfrac12y^{\mathsf T}y+n(t)\,c^{\mathsf T}y+\text{const}$$, so $$\dot y=-A(t)\,(y+n(t)c)$$ and the instantaneous minimiser is $$y_{\rm eq}(t)=-n(t)c$$. The mass is $$\mu=\mu_0+c^{\mathsf T}y$$. Write $$\theta(t)=n(t_0)-n(t)\ge0$$, non-decreasing, so that $$y_{\rm eq}(t)=y_0+\theta(t)c$$ with $$y_0=y_{\rm eq}(t_0)=y(t_0)$$. For $$u=y-y_0$$, $$\dot u=-A(u-\theta c)$$. Let $$Y(t,\tau)$$ be the propagator of $$\dot z=-Az$$. Since $$\tfrac{d}{dt}\lVert z\rVert^2=-2z^{\mathsf T}(\mathrm{sym}\,A)z\le0$$ and $$\mathrm{sym}\,A=K^{1/2}(\mathrm{sym}\,\Gamma)K^{1/2}\ge0$$, $$\lVert Y\rVert\le1$$. For $$\theta$$ equal to a unit step at time $$\tau$$, the solution is $$u(t)=(1-Y(t,\tau))c$$, because $$\partial_\tau Y(t,\tau)=Y(t,\tau)A(\tau)$$. A non-decreasing $$\theta$$ is a positive superposition of steps, $$\theta(t)=\int_{t_0}^t d\theta(\tau)$$, so
$$c^{\mathsf T}u(t)=\int d\theta(\tau)\,\big[c^{\mathsf T}c-c^{\mathsf T}Y(t,\tau)c\big]\ge0$$
by the Cauchy–Schwarz inequality and $$\lVert Y\rVert\le1$$. Hence $$\mu(t)-\mu(t_0)=c^{\mathsf T}u\ge0$$. The rate at which the effective potential energy is dissipated is $$-\nabla\Phi\cdot\dot\xi=\nabla\Phi^{\mathsf T}\Gamma\nabla\Phi=\Pi\ge0$$ (only the symmetric part of $$\Gamma$$ contributes). Since $$\Phi=g+n\mu$$, $$-\dot\rho_v-n\dot\mu=-\nabla\Phi\cdot\dot\xi=\Pi$$, so $$Q=-\dot\rho_v=n\dot\mu+\Pi$$. With $$na^3$$ constant, $$\int_{t_0}^ta^3n\dot\mu\,dt'=a^3n\,(\mu(t)-\mu(t_0))\ge0$$, and $$\int_{t_0}^ta^3\Pi\,dt'\ge0$$. $$\square$$

*Proof of (c).* Let $$\varrho=\sum_jr_j\vert j\rangle\langle j\vert $$ in the energy eigenbasis, with energies $$E_1\le E_2\le\dots\le E_d$$ and $$r_1\ge r_2\ge\dots\ge r_d$$. Define $$W_{ij}=\langle i\vert \mathcal T(\vert j\rangle\langle j\vert )\vert i\rangle\ge0$$. Trace preservation gives $$\sum_iW_{ij}=1$$ and unitality gives $$\sum_jW_{ij}=\langle i\vert \mathcal T(\mathbb 1)\vert i\rangle=1$$, so $$W$$ is doubly stochastic. The energy after the step is $$\sum_iE_iq_i$$ with $$q=Wr$$ (the diagonal elements of $$\mathcal T(\varrho)$$ in the energy basis; off-diagonal elements do not contribute to the trace). Let $$R_k=\sum_{i\le k}r_i$$ and $$R'_k=\sum_{i\le k}q_i$$. Then $$R'_k=\sum_jr_jc_j$$ with $$c_j=\sum_{i\le k}W_{ij}\in[0,1]$$ and $$\sum_jc_j=\sum_{i\le k}\sum_jW_{ij}=k$$. Because $$r$$ is sorted in decreasing order, a linear form $$\sum_jr_jc_j$$ under the constraints $$0\le c_j\le1$$, $$\sum_jc_j=k$$ is largest at $$c=(1,\dots,1,0,\dots,0)$$, where it equals $$R_k$$. Hence $$R'_k\le R_k$$ for every $$k$$. Summation by parts gives $$\sum_iE_iq_i=E_d-\sum_{k<d}(E_{k+1}-E_k)R'_k\ge E_d-\sum_{k<d}(E_{k+1}-E_k)R_k=\sum_iE_ir_i$$. This is passivity in the sense of [[PW78]](#ref-pw78), [[Lenard]](#ref-lenard): no unital mixing lowers the energy of a passive state. $$\square$$
</div>

Global minimality is used only in (a), and the monotonicity of the source only through $$\partial_\mu F_2<\partial_\mu F_1$$. In (b) the constancy of $$K$$ is used in the substitution $$y=K^{1/2}\xi$$ that turns the dynamics into a contraction, and starting on the equilibrium branch is used in $$u(t_0)=0$$. The condition that the vacuum is non-propagating means that its energy is $$g(\xi)$$ only; with kinetic energy, $$w_v\ne-1$$ and the reconstruction of Lemma 1 changes character. The sign of the coupling $$b$$ never enters.

<div class="corollary" markdown="1">
**Corollary 1.** (a) Under Theorem 1(a) for dust, $$Q\ge0$$ at all times. The comoving dark-matter density $$a^3\rho_M$$ at recombination is then not larger than at any later time, in particular $$\omega_c(\mathrm{rec})/\omega_c(0)\le1$$, where $$\omega_c(t)=a(t)^3\rho_M(t)/a_0^3$$ is the comoving dark-matter density expressed as a density today, and a reconstruction anchored at recombination returns $$w_{\rm eff}\ge-1$$ at all later times whenever $$\rho_{\rm DE,eff}>0$$.

(b) Under Theorem 1(b), $$a^3n\mu$$ at any $$t\ge t_0$$ is not smaller than at the start $$t_0$$ of the relaxation, and the transfer weighted by $$a^3$$ and accumulated since $$t_0$$ is non-negative. A reconstruction anchored at $$t_0$$ in which the transferred energy is counted as dust-like then has $$\Delta\ge0$$.

(c) Under Theorem 1(c), the matter gains energy at each step and the vacuum loses it, $$\dot\rho_v\le0$$. Nothing follows for $$\omega_c$$, since the added energy is not an $$a^{-3}$$-diluting rest mass.
</div>

The first statement is Lemma 1 together with $$Q\ge0$$. A fit that returns $$w_{\rm eff}<-1$$ at some time after an early anchor, or $$\omega_c(\mathrm{rec})/\omega_c(0)>1$$, needs $$Q<0$$ over some interval, which Theorem 1(a) forbids. For (b) and (c) the theorem restricts the sign of the energy transfer in the senses stated, and translates into the dark-matter density only to the extent that the transferred energy dilutes as $$a^{-3}$$.

<div class="remark" markdown="1">
**Remark 1.** In (b) the transfer is non-negative only cumulatively. Take $$r=2$$, $$K=\mathbb 1$$, $$c=(1,0)$$, $$\Gamma=\begin{pmatrix}\epsilon&\omega\\-\omega&\epsilon\end{pmatrix}$$ with $$\epsilon,\omega>0$$ (symmetric part $$\epsilon\mathbb 1$$), and let $$n$$ drop by one unit at $$t=0$$, the limit of a drop that is fast compared with the relaxation time. The solution of $$\dot u=-\Gamma(u-c)$$ with $$u(0)=0$$ has $$\mu(t)-\mu(0)=1-e^{-\epsilon t}\cos\omega t\ge0$$ and $$\dot\mu=e^{-\epsilon t}(\epsilon\cos\omega t+\omega\sin\omega t)$$, which is negative during part of each period. The mass overshoots its new equilibrium value and returns, so $$n\dot\mu<0$$ at those times, although $$\mu(t)\ge\mu(0)$$ throughout.
</div>

## 3. Where the theorem applies

### Mass-coupled static fluids

Take $$\rho=g(\xi)+n\mu(\xi)$$ for dust whose mass per particle is $$\mu(\xi)$$, with $$\xi$$ at a strict minimum at fixed $$n$$: $$\nabla g+n\nabla\mu=0$$ with Hessian $$\mathcal K=\nabla^2g+n\nabla^2\mu>0$$. This is the local version of Theorem 1(a), and it gives the rate of change of the mass.

<div class="proposition" markdown="1">
**Proposition 1** (Hessian identity). Along the branch of strict minima, $$d\xi/dn=-\mathcal K^{-1}\nabla\mu$$, and

$$
\frac{d\mu}{dn}=-\nabla\mu^{\mathsf T}\mathcal K^{-1}\nabla\mu\le0,\qquad \frac{d\rho_v}{dn}=n\,\nabla\mu^{\mathsf T}\mathcal K^{-1}\nabla\mu\ge0, \tag{4}
$$

where $$\mu$$ and $$\rho_v=g$$ are evaluated on the branch. With $$\chi=n\,\nabla\mu^{\mathsf T}\mathcal K^{-1}\nabla\mu/\mu\ge0$$, the mass changes as $$d\ln \mu/d\ln a=3\chi\ge0$$, and the adiabatic sound speed squared of the tracking fluid is $$c_s^2=-\chi\le0$$.
</div>

<div class="proof" markdown="1">
*Proof.* Differentiate $$\nabla g+n\nabla\mu=0$$ along the branch: $$\mathcal K\,d\xi/dn+\nabla\mu=0$$. Then $$d\mu/dn=\nabla\mu^{\mathsf T}d\xi/dn=-\nabla\mu^{\mathsf T}\mathcal K^{-1}\nabla\mu$$, and $$d\rho_v/dn=\nabla g^{\mathsf T}d\xi/dn=-n\nabla\mu^{\mathsf T}d\xi/dn$$ gives the second relation. With $$na^3$$ constant, $$dn/d\ln a=-3n$$, so $$d\ln\mu/d\ln a=-3n\,(d\mu/dn)/\mu=3\chi$$. The total energy density on the branch is $$\rho_{\rm eq}(n)=g+n\mu$$ with $$\rho_{\rm eq}'=\mu$$ (envelope theorem), the pressure is $$p=n\rho_{\rm eq}'-\rho_{\rm eq}=-g$$, and $$c_s^2=dp/d\rho=n\,(d\mu/dn)/\mu=-\chi$$. $$\square$$
</div>

The sign $$c_s^2<0$$ is the adiabatic instability of mass-varying dark sectors found in [[AZK]](#ref-azk), [[BFT 1]](#ref-bft1), [[BFT 2]](#ref-bft2). It does not enter the sign theorem, but it means that a fluid obtained from a non-propagating auxiliary variable has a gradient instability. For the global minimiser, the concavity of $$\rho_{\rm eq}(n)=\min_\xi[g+n\mu]$$, a minimum of functions affine in $$n$$, gives the monotonicity of $$\mu$$ at once, including across jumps.

### Unimodular heating

In unimodular gravity the cosmological term is an integration function. Write the field equations as $$G_{ab}+\Lambda g_{ab}=8\pi GT_{ab}$$ with $$\Lambda(x)$$ determined by the Bianchi identity, $$\nabla_b\Lambda=8\pi G\nabla^aT_{ab}$$ [[JPS]](#ref-jps). In a homogeneous universe $$u^b\nabla^aT_{ab}=-(\dot\rho_M+3H(\rho_M+p_M))=-Q$$, so $$\dot\rho_{\rm vac}=-Q$$ with $$\rho_{\rm vac}=\Lambda/8\pi G$$, which is (1). A vacuum density that changes by exchange has the effective equation of state $$w_{\rm vac}=-1-\dot\rho_{\rm vac}/(3H\rho_{\rm vac})$$, the one it would need if it were conserved separately. Consider matter whose reduced dynamics is a Gorini–Kossakowski–Sudarshan–Lindblad (GKSL) generator, the most general form of a Markovian completely positive evolution of a density operator, $$\dot\varrho=-\tfrac{i}{\hbar}[\hat h,\varrho]+\sum_k(L_k\varrho L_k^\dagger-\tfrac12\{L_k^\dagger L_k,\varrho\})$$ with self-adjoint $$L_k$$, as in spontaneous-collapse models.

<div class="proposition" markdown="1">
**Proposition 2** (heating by self-adjoint Lindblad operators). For self-adjoint $$L_k$$ the heating rate satisfies $$\dot E=-\tfrac12\sum_k\langle[L_k,[L_k,\hat h]]\rangle$$. For $$\hat h=\hat p^2/2m+V(\hat x)$$ and $$L_k=L_k(\hat x)$$ a function of position, it equals $$\dot E=\tfrac{\hbar^2}{2m}\sum_k\langle(\nabla L_k)^2\rangle\ge0$$ in every state.
</div>

<div class="proof" markdown="1">
*Proof.* In the Heisenberg picture the dissipator acts on an operator $$O$$ as $$\sum_k(L_kOL_k-\tfrac12L_k^2O-\tfrac12OL_k^2)=-\tfrac12\sum_k[L_k,[L_k,O]]$$, and the unitary part does not change the energy. For $$O=\hat h$$ only the kinetic term fails to commute with $$L_k(\hat x)$$. From $$[L,\hat p_j]=i\hbar\,\partial_jL$$ one finds $$[L,\hat p^2]=i\hbar(\nabla L\cdot\hat p+\hat p\cdot\nabla L)$$ and $$[L,[L,\hat p^2]]=-2\hbar^2(\nabla L)^2$$. Hence $$[L,[L,\hat h]]=-(\hbar^2/m)(\nabla L)^2$$ and the dissipator gives $$(\hbar^2/2m)(\nabla L)^2$$, a positive operator. $$\square$$
</div>

In unimodular gravity (a model assumption) the heating is the energy $$Q\ge0$$ received by matter, so $$\dot\rho_{\rm vac}\le0$$, and $$w_{\rm vac}\ge-1$$ if $$\rho_{\rm vac}>0$$. The map generated by self-adjoint $$L_k$$ is unital, so Theorem 1(c) gives the same sign for any passive state and any energy function, while the computation above holds in every state. The quantity $$w_{\rm vac}$$ refers to the vacuum itself. The equation of state that an observer reconstructs is a different quantity, by Lemma 1: if the heated matter is counted as cold dark matter at a late anchor, the reconstruction has $$w_{\rm eff}<-1$$ at earlier times, as in [[DCK]](#ref-dck). The proposition therefore does not exclude a reconstruction that looks phantom, only a phantom vacuum energy.

<div class="remark" markdown="1">
**Remark 2** (momentum diffusion on the mass shell). The diffusion of massive particles in energy-momentum from spacetime discreteness is the model class of [[DPS]](#ref-dps). The momentum of a particle of mass $$m$$ lies on the mass shell $$\Sigma_m\cong\mathbb H^3$$, with energy $$E=m\cosh\zeta$$ in terms of the rapidity $$\zeta$$, and the Lorentz-invariant diffusion is $$\partial_\tau f=\kappa\nabla^2_\Sigma f$$ with $$\tau$$ the proper time of the particle. The Laplacian acts on a radial function as $$f''+2\coth\zeta\,f'$$, which for $$f=\cosh\zeta$$ is $$3\cosh\zeta$$. Hence $$d\langle E\rangle/d\tau=3\kappa\langle E\rangle>0$$, assuming that boundary terms vanish. The most general Lorentz-invariant massless diffusion in [[DPS]](#ref-dps) contains a drift in the energy of the particle in addition to the diffusion, and the sign of the drift is not fixed by Lorentz invariance. Energy diffusion from matter into the discrete structure is the scenario of [[PS]](#ref-ps); in unimodular gravity it gives $$Q<0$$ and $$\dot\rho_{\rm vac}>0$$, the direction opposite to Proposition 2, so it lies outside the proposition and is not excluded by the theorems. A drift that removes energy from the particle lies outside as well.
</div>

### What the theorem requires of the vacuum

The sign depends on $$w_v=-1$$, and a scalar field with a kinetic term can reverse it. As a derivation within the model, take dust with mass $$m(\phi)=m_0e^{\beta\phi}$$ (with $$\phi$$ in Planck units) and a scalar field of negligible potential gradient. The matter action gives $$\rho_c=n\,m(\phi)$$ and the field equation $$\ddot\phi+3H\dot\phi+V'(\phi)=-\beta\rho_c$$. If $$V'=0$$ and $$\dot\phi=0$$ initially, then $$(a^3\dot\phi)^\cdot=-\beta a^3\rho_c$$ gives $$\beta\dot\phi=-(\beta^2/a^3)\int a^3\rho_c\,dt<0$$, so $$\dot m/m=\beta\dot\phi<0$$ for either sign of $$\beta$$: the particle mass falls with time and energy flows from matter into the kinetic energy of the field. The effective potential $$nm_0e^{\beta\phi}$$ has no minimum, so (a) does not apply, and the field has $$w=+1$$, so the vacuum is not of the type in section 1. Models of this class (coupled quintessence [[Amendola]](#ref-amendola)) that start away from the tracking branch, or that have no minimiser, are outside Theorem 1.

## 4. A field that tracks a minimum makes the electron lighter in the past

Fits of CMB and baryon acoustic oscillation (BAO) data in which the electron mass at recombination is free return central values above the laboratory one. Hydrogen recombines when the photon temperature reaches a fixed fraction of the binding energy, which is proportional to $$m_e$$; a heavier electron makes recombination earlier and shrinks the sound horizon. Write $$\delta\equiv m_e(z_*)/m_e^{\rm lab}-1$$. The published values are in the table.

| data | $$\delta$$ (%) |
|---|---|
| Planck + ACT DR6 + CMB lensing + DESI DR1 BAO [[ACT DR6]](#ref-act) | $$0.96\pm0.60$$ |
| the same + type Ia supernovae [[ACT DR6]](#ref-act) | $$0.63\pm0.56$$ |
| ACT DR6 + Planck ($$\ell<1000$$) + lensing + DESI DR2 BAO + supernovae [[Toda–Seto]](#ref-toda-seto) | $$0.78\pm0.47$$ |
| CMB + DESI DR2 BAO [[Weiner]](#ref-weiner) | $$0.76\pm0.48$$ |

These analyses treat $$m_e$$ at recombination as a free parameter with no field dynamics. Here a scalar $$\phi$$ on which $$m_e$$ depends, with bare potential $$V(\phi)$$, supplies the dynamics. The vacuum of section 1 is replaced by a scalar with a kinetic term whose homogeneous value is assumed to follow the minimum of its effective potential (hypothesis (T1) below); this is a model assumption, and it holds when the field mass is large compared with the Hubble rate, so that the kinetic energy and the oscillations about the minimum are negligible. 

<div class="example" markdown="1">
**Example 1.** Take $$m_e=m_0e^{\beta\phi/M_P}$$, with the reduced Planck mass $$M_P=(8\pi G)^{-1/2}$$, and $$V=\tfrac12m_\phi^2\phi^2$$ with $$m_\phi\gg H$$. Non-relativistic electrons of number density $$n$$ add the energy density $$n\,m_e(\phi)$$, so $$U(\phi)=\tfrac12m_\phi^2\phi^2+n\,m_0e^{\beta\phi/M_P}$$. At the minimum $$m_\phi^2\phi=-(\beta/M_P)\,n\,m_e$$, hence $$\phi$$ has the sign opposite to $$\beta$$ and
$$\ln(m_e/m_0)=\beta\phi/M_P=-\beta^2n\,m_e/(m_\phi^2M_P^2)<0.$$
The sign of the coupling drops out because it enters squared, and the denser the gas, the lighter the electron.
</div>

The field moves so as to lower the energy of the matter it is immersed in, and matter of given composition has lower energy when its constituents are lighter. The general statement needs the derivative of the energy of a relativistic plasma with respect to a mass.

For one species of mass $$M$$ with $$g_s$$ internal states in kinetic equilibrium at temperature $$T$$, define $$\int_k\equiv g_s\int d^3k/(2\pi)^3$$, $$E=\sqrt{k^2+M^2}$$, and occupation numbers $$f$$. Then $$\rho=\int_kEf$$ and $$p=\int_k\frac{k^2}{3E}f$$ (summed over particles and antiparticles when both are present), and we call $$n_{\rm sc}=\int_k(M/E)f=(\rho-3p)/M$$ the scalar density of the species, using $$E-k^2/E=M^2/E$$. It is positive wherever the species is present, and $$n_{\rm sc}\to n$$ for non-relativistic particles.

<div class="lemma" markdown="1">
**Lemma 2** (mass derivative of the free energy). For an ideal gas let $$\mathcal F(M;T,\mathcal N)$$ be the Helmholtz free energy density at fixed temperature and fixed net number density $$\mathcal N$$ (particle minus antiparticle number for a gas that contains both), and $$\mathcal E(M;s,\mathcal N)$$ the energy density at fixed entropy density $$s$$ and $$\mathcal N$$. Then

$$
\left.\frac{\partial\mathcal F}{\partial M}\right\vert_{T,\mathcal N}=\left.\frac{\partial\mathcal E}{\partial M}\right\vert_{s,\mathcal N}=n_{\rm sc}=\frac{\rho-3p}{M}>0. \tag{5}
$$

For a mixture whose masses depend on $$m_e$$, $$\partial\mathcal F/\partial m_e=\partial\mathcal E/\partial m_e=\sum_in_{{\rm sc},i}\,dM_i/dm_e$$, at fixed composition and at chemical equilibrium.
</div>

<div class="proof" markdown="1">
*Proof.* The grand potential density is $$-p(M;T,\mu_c)$$ with $$p=\pm g_sT\int\frac{d^3k}{(2\pi)^3}\ln(1\pm e^{-(E-\mu_c)/T})$$ for the particle species, plus the antiparticle term for $$-\mu_c$$. Differentiating under the integral with $$\partial E/\partial M=M/E$$ gives $$\partial p/\partial M\vert _{T,\mu_c}=-n_{\rm sc}$$. An integration by parts in $$k$$ turns the logarithm into the kinetic pressure $$\int_kk^2f/3E$$. The free energy is $$\mathcal F=-p+\mu_c\mathcal N$$ with $$\mu_c$$ fixed by $$\mathcal N$$; since $$\partial\mathcal F/\partial\mu_c\vert _{M}=-\partial p/\partial\mu_c+\mathcal N=0$$, the derivative at fixed $$\mathcal N$$ equals that of $$-p$$ at fixed $$\mu_c$$, which is $$n_{\rm sc}$$. The energy density satisfies $$d\mathcal E=T\,ds+\mu_c\,d\mathcal N+\nu\,dM$$ with $$\nu=\partial\mathcal E/\partial M\vert_{s,\mathcal N}$$, and $$\mathcal F=\mathcal E-Ts$$ gives $$d\mathcal F=-s\,dT+\mu_c\,d\mathcal N+\nu\,dM$$, so $$\nu=\partial\mathcal F/\partial M\vert_{T,\mathcal N}$$. For a mixture the chain rule gives the sum, and at chemical equilibrium the composition extremises the energy at fixed entropy, so by the envelope theorem its variation does not contribute. $$\square$$
</div>

In the cosmological plasma the species that depend on $$m_e$$ are free electrons and positrons, with $$dM/dm_e=1$$, and hydrogen and helium with bound electrons. A bound level of binding energy $$B\propto m_e$$ gives an atom the mass $$M=M_{\rm nucleus}+m_e-B$$ and $$dM/dm_e=1-B/m_e$$, which is $$1-2.7\times10^{-5}$$ for ground-state hydrogen ($$B=13.6$$ eV). Coulomb corrections to the ideal-gas thermodynamics are neglected. We call $$\nu\equiv\partial\mathcal E/\partial m_e=\sum_in_{{\rm sc},i}\,dM_i/dm_e$$ the electron scalar density; by Lemma 2 it is positive wherever electrons are present.

Let state $$j$$ of the plasma be specified by its entropy density, its conserved number densities and its composition (ionisation fractions in equilibrium or frozen out). Cosmological evolution conserves entropy per comoving volume, so the energy density $$e_j(m)$$ of state $$j$$ at fixed entropy and numbers, with masses evaluated at electron mass $$m$$, is the function the field responds to, and $$U_j(\phi)=V(\phi)+e_j(m_e(\phi))$$. Its derivative $$\nu_j(m)=e_j'(m)$$ is the electron scalar density of state $$j$$. A homogeneous field at rest is stationary where $$U_j'(\phi)=0$$, since by Lemma 2 the derivative of $$e_j$$ reproduces the source term $$-\sum_in_{{\rm sc},i}\partial M_i/\partial\phi$$ of the field equation, and the minima of $$U_j$$ are the stable positions.

(T1) Tracking: in each state $$j$$ compared, $$\phi$$ is at a global minimiser of $$U_j$$.

(T2) Standard Model sources only: apart from $$V$$, the $$\phi$$-dependence of $$U_j$$ comes from Standard Model species whose masses are non-decreasing functions of $$m_e$$. A field that changes only $$m_e$$ qualifies, and so does a Higgs portal provided every mass it changes increases with the Higgs vacuum value.

(T3) More matter in the past: on an interval of $$m$$ containing the two values of $$m_e$$ compared, $$\nu_1(m)>\nu_2(m)$$ for the earlier state 1 and the later state 2.

<div class="theorem" markdown="1">
**Theorem 2** (electron mass sign). Let state 1 be earlier and state 2 later, and let $$\phi_1,\phi_2$$ satisfy (T1) in the two states. Under (T2) and (T3), $$m_e(\phi_1)\le m_e(\phi_2)$$. In particular, for the recombination plasma (state 1) and a laboratory environment (state 2) whose electron scalar density, counting bound electrons, is smaller than that of the recombination plasma over the interval of (T3), $$\delta\le0$$.
</div>

<div class="proof" markdown="1">
*Proof.* This is Theorem 1(a) with $$X$$ the set of field values, $$g=V$$, $$\mu(\xi)=m_e(\phi)$$ and $$F_j=e_j$$ as a function of $$m_e$$. Under (T2) the difference $$e_2-e_1$$ depends on $$\phi$$ only through $$m_e$$, and its derivative is $$\nu_2-\nu_1<0$$ on the interval of (T3), by Lemma 2. The conclusion of (a) is $$m_e(\phi_2)\ge m_e(\phi_1)$$. For $$\delta$$, $$1+\delta=m_e(\phi_1)/m_e(\phi_2)\le1$$. $$\square$$
</div>

Nothing in the proof uses the shape of $$V$$, the size of the coupling, or the number of components of $$\phi$$. The input is Lemma 2, which fixes the sign of $$\partial\mathcal E/\partial m_e$$, and (T3), which says which state has more of the matter that prefers a light electron. A weaker reading of tracking, a continuous branch of local minima, gives the same sign.

<div class="proposition" markdown="1">
**Proposition 3** (local tracking). Let $$\lambda\in[0,1]$$ label a continuous sequence of states from today ($$\lambda=0$$) back to an earlier epoch, and let $$\phi(\lambda)$$ be a $$C^1$$ branch of non-degenerate local minima of $$U_\lambda$$ for a single-component field. Under (T2), if $$\nu(m;\lambda)$$ does not decrease with $$\lambda$$ at fixed $$m$$, then $$m_e(\phi(\lambda))$$ is non-increasing in $$\lambda$$.
</div>

<div class="proof" markdown="1">
*Proof.* Differentiate $$U'_\lambda(\phi(\lambda))=0$$ along the branch: $$U''_\lambda\phi'+\partial_\lambda U'_\lambda=0$$. By Lemma 2 and (T2), $$U'_\lambda=V'+\nu(m_e(\phi);\lambda)\,m_e'(\phi)$$, so $$\partial_\lambda U'_\lambda=m_e'\partial_\lambda\nu$$ and $$dm_e/d\lambda=m_e'\phi'=-(m_e')^2\partial_\lambda\nu/U''_\lambda\le0$$. $$\square$$
</div>

The branch can end where $$U''_\lambda\to0$$, after which the field jumps; Theorem 2 covers jumps between global minima.

For a field that changes only $$m_e$$, (T3) holds between recombination and any later state of the universe at its mean density. After electron–positron annihilation the net electron number is conserved, so the number density scales as $$(1+z)^3$$ and the scalar density is $$n_{\rm sc}=n\langle m_e/E\rangle\approx n(1-3T/2m_e)$$; at recombination $$T/m_e\approx5\times10^{-7}$$, so the correction is negligible against the factor $$(1+z)^3$$, for every $$m$$ within a few per cent of $$m_e$$. Before and during annihilation the pairs add to the scalar density. An evaluation for an ideal electron–positron gas with the baryon-to-photon ratio $$\eta_b=273.9\,\Omega_bh^2\times10^{-10}=6.1\times10^{-10}$$ (with $$\Omega_bh^2=0.02237$$ [[Planck VI]](#ref-planck6)) and helium fraction $$Y_p=0.245$$ (so $$1-Y_p/2$$ electrons per baryon) finds the scalar density at $$m=m_e$$ increasing monotonically with temperature from $$T_0=2.7255$$ K to 10 MeV; (T3) in the annihilation era, used by Proposition 3 and by the response to the plasma trace, is therefore established at $$m=m_e$$ only. The mean electron density at last scattering is $$n_e(z_*=1090)\approx2.9\times10^2\ {\rm cm}^{-3}$$, and the mean today is $$2.2\times10^{-7}\,{\rm cm}^{-3}$$ for the same inputs.

### Which environment is today

The theorem compares tracked values in two homogeneous states. The laboratory value is the tracked value for whatever environment the field responds to there. Theorem 2 gives $$\delta\le0$$ provided that environment has an electron scalar density below $$n_e(z_*)\approx290\ {\rm cm}^{-3}$$. This holds if the field responds to the cosmic mean or to the mean density of the Galaxy, which near the Sun is a few nucleons per cubic centimetre. It fails if the field tracks the density of the Earth or of the laboratory, as screened fields of chameleon [[KW]](#ref-chameleon) or symmetron [[HK]](#ref-symmetron) type do. Applied to pairs of environments of similar composition, the theorem orders them by electron scalar density: $$m_e$$ in any environment less dense than the recombination plasma is at least as large as at recombination.

<div class="corollary" markdown="1">
**Corollary 2.** Assume (T1) to (T3) and that the laboratory value of $$m_e$$ is set by an environment less dense than the recombination plasma. Then every field that changes $$m_e$$ gives $$\delta\le0$$, whatever its potential and the sign and number of its couplings. This includes the response of the field to the trace $$\rho-3p$$ of the electron–positron plasma during annihilation, a Higgs portal under the proviso in (T2), and potentials of chameleon or symmetron form if the screening is not active at the density of the laboratory.
</div>

Screened fields whose laboratory value is set by the density of the Earth or of the apparatus do not satisfy the hypothesis on the environment, and are treated in section 6, under tracking of the local density.

The fits in the table place $$\delta=0$$ between $$1.1$$ and $$1.7$$ standard deviations below their central values ($$0.96/0.60=1.6$$, $$0.63/0.56=1.1$$, $$0.78/0.47=1.7$$, $$0.76/0.48=1.6$$). Theorem 2 is therefore in tension with the central values at a level that is not decisive, and the fits impose no field dynamics. How much the sign statement constrains the data depends on how far the preference for $$\delta>0$$ strengthens beyond this range.

## 5. A field that does not track: the excursion bound

A light field that is frozen at early times and released later can move in either direction, so Theorem 2 does not apply. The obstruction is then different: the field must still arrive at today's value, and if its coupling to electrons is the same along the way, that value is constrained by tests of the equivalence principle.

Write $$\beta_e\equiv M_P\,d\ln m_e/d\phi$$ today. If $$\phi$$ is light enough to mediate a force over the size of the Earth, a body $$b$$ of mass $$M_b$$ containing $$N_b$$ electrons has scalar charge $$q_b=\partial M_b/\partial\phi=\beta_eN_bm_e/M_P$$, when only the electron rest mass depends on $$\phi$$. Scalar exchange between bodies $$b$$ and $$c$$ gives an attractive force $$q_bq_c/(4\pi r^2)$$, gravity gives $$M_bM_c/(8\pi M_P^2r^2)$$, and the ratio is $$2\beta_e^2f_bf_c$$ with $$f_b=N_bm_e/M_b\approx(m_e/m_u)(Z/A)_b$$, where $$Z$$ is the atomic number and $$A$$ the mass number. The Eötvös parameter for two test masses falling towards the Earth ($$E$$) is

$$
\eta_{\rm EP}\simeq2\beta_e^2f_E(f_{\rm Ti}-f_{\rm Pt}). \tag{6}
$$

The MICROSCOPE experiment measured $$\eta_{\rm EP}({\rm Ti,Pt})=[-1.5\pm2.3\,({\rm stat})\pm1.5\,({\rm syst})]\times10^{-15}$$ [[MICROSCOPE]](#ref-microscope). The errors combine in quadrature to $$2.75\times10^{-15}$$, so the 2 sigma range of $$\lvert\eta_{\rm EP}\rvert$$ extends to $$1.5\times10^{-15}+2\times2.75\times10^{-15}=7.0\times10^{-15}$$. With $$m_e/m_u=5.486\times10^{-4}$$, $$(Z/A)_{\rm Ti}=22/47.867=0.4596$$, $$(Z/A)_{\rm Pt}=78/195.084=0.3998$$ and a bulk-Earth value $$(Z/A)_E\approx0.487$$ (an assumed input), (6) gives $$\lvert\beta_e\rvert<6.3\times10^{-4}$$ at this level. Varying $$(Z/A)_E$$ between $$0.48$$ and $$0.50$$ changes the bound by less than 2%; the bound scales as $$\sqrt{\eta_{\max}}$$.

Suppose now that $$\lvert d\ln m_e/d\phi\rvert$$ never exceeds a value $$\beta_{\max}/M_P$$ along the path from last scattering to today; the identification $$\beta_{\max}=6.3\times10^{-4}$$ is the assumption that the coupling does not grow towards the past. Then $$\ln(1+\delta)\le\beta_{\max}\lvert\Delta\phi\rvert/M_P$$. A field cannot move arbitrarily far, because its kinetic energy density $$\dot\phi^2/2$$ is a share $$\Omega_{\rm kin}=\dot\phi^2/(6H^2M_P^2)$$ of the total. For a flat or negatively curved universe in which every non-kinetic energy density is non-negative, the Friedmann equation gives $$\Omega_{\rm kin}\le1$$. With $$s=\ln a$$ as the time variable, $$d\phi/ds=\pm\sqrt{6\Omega_{\rm kin}}\,M_P$$, and over the $$s_*=\ln(1+z_*)\approx\ln1091=6.99$$ e-folds from last scattering to today the Cauchy–Schwarz inequality gives

$$
\frac{\lvert\Delta\phi\rvert}{M_P}\le\int_0^{s_*}\sqrt{6\Omega_{\rm kin}}\,ds\le s_*\sqrt{6\langle\Omega_{\rm kin}\rangle}=17.1\sqrt{\langle\Omega_{\rm kin}\rangle}, \tag{7}
$$

with $$\langle\Omega_{\rm kin}\rangle$$ the average of $$\Omega_{\rm kin}$$ over e-folds.

<div class="theorem" markdown="1">
**Theorem 3** (excursion bound). Suppose $$m_e$$ depends on one field $$\phi$$ with $$\lvert d\ln m_e/d\phi\rvert\le\beta_{\max}/M_P$$ along the path from last scattering to today, that the universe is flat or negatively curved, and that all non-kinetic energy densities, including $$V(\phi)$$, are non-negative. Then $$\ln(1+\delta)\le\sqrt6\,s_*\,\beta_{\max}\sqrt{\langle\Omega_{\rm kin}\rangle}$$. For $$\beta_{\max}=6.3\times10^{-4}$$ this is $$\delta\lesssim1.1\%\times\sqrt{\langle\Omega_{\rm kin}\rangle}$$.
</div>

The proof is the chain of the two inequalities above. The table shows what the central values of the fits would require at $$\beta_{\max}=6.3\times10^{-4}$$: the required excursion is $$\ln(1+\delta)/\beta_{\max}$$ and the required $$\langle\Omega_{\rm kin}\rangle$$ is $$(\lvert\Delta\phi\rvert/17.1\,M_P)^2$$.

| $$\delta$$ (%) | required $$\lvert\Delta\phi\rvert/M_P$$ | required $$\langle\Omega_{\rm kin}\rangle$$ |
|---|---|---|
| 0.96 | 15.2 | 0.78 |
| 0.78 | 12.3 | 0.52 |
| 0.76 | 12.0 | 0.49 |
| 0.63 | 10.0 | 0.34 |

A required $$\langle\Omega_{\rm kin}\rangle$$ above 1 is impossible under the stated assumptions, so a value of $$\delta$$ above $$1.1\%$$ is excluded at this level for the class. The central values in the table need the field's kinetic energy to carry between $$34\%$$ and $$78\%$$ of the total energy density on average over the e-folds since recombination, in an interval that the CMB and BAO describe as dominated by matter. As an illustration, if the kinetic energy never exceeded the share of a cosmological constant, $$\Omega_{\rm kin}\le\Omega_\Lambda(z)$$ with $$\Omega_m=0.3$$ in a $$\Lambda$$CDM history, the middle expression of (7) gives $$\lvert\Delta\phi\rvert\le1.98M_P$$ and $$\delta\le0.12\%$$.

## 6. Hypotheses whose failure changes the sign

### Fields that start off the tracking branch, or have no minimiser

The rolling field of section 3 pushes the particle mass down for either sign of the coupling, so that $$\omega_c(\mathrm{rec})/\omega_c(0)>1$$ is produced by a coupled scalar with kinetic energy [[Amendola]](#ref-amendola). The vacuum is then not of the type in Theorem 1, since $$w\ne-1$$. The required coupling and the cost in the fits to the CMB are not computed here.

### Active media

A negative $$\Gamma$$ or a concave $$g+F$$ (population inversion, a negative temperature) reverses the sign. In one dimension with $$K=1$$, $$\Gamma=-\gamma<0$$ and a drop of $$n$$ by $$\theta$$ at $$t_0$$, the equation $$\dot y=\gamma(y-y_{\rm eq})$$ gives $$\mu(t)-\mu(t_0)=\theta c^2(1-e^{\gamma(t-t_0)})<0$$. The generalised second law and the source of the free energy then have to be specified. Matter creation with a chemical potential [[CCL]](#ref-ccl), non-unital maps, and reservoirs at negative temperature are likewise outside (c).

### Hysteresis

A field that follows a local minimum to a spinodal and then jumps is outside the global-minimum argument. In random one-dimensional models with the structure of Theorem 1(a), a jump along the local-minimum branch can lower $$\mu$$, while the net change of $$\mu$$ from the start to the end of the history was never negative. A proof for this case is missing.

### Quadratic couplings

For $$m_e=m_0(1+\phi^2/2M_e^2)$$ with $$\phi=0$$ today there is no linear coupling, $$\beta_e=0$$, and (6) is empty. A displacement from zero cannot come from a Standard Model source, since any displacement raises $$m_e$$, which Theorem 2 forbids. A positive $$\delta=\phi_i^2/2M_e^2$$ requires a field held away from zero by Hubble friction that later oscillates about zero. This class is studied in [[BSW 1]](#ref-bsw1), [[BSW 2]](#ref-bsw2) and [[GBY]](#ref-gby). The oscillating field is matter-like at late times, and its residual oscillation modulates $$m_e$$ in time, which atomic clocks constrain; those limits are not computed here.

### Fields also sourced by dark matter

If $$\phi$$ also changes the mass $$m_X$$ of a non-relativistic dark-matter particle, (T2) fails but inequality (3) still holds under (T1). With $$F_j=n_{e,j}m_e+n_{X,j}m_X$$ and both densities scaling as $$(1+z)^3$$, it reads

$$
\frac{m_X^{\rm lab}-m_X^{\rm rec}}{m_X}\ge\delta\,\frac{\rho_{e,0}}{\rho_{X,0}},\qquad \frac{\rho_{e,0}}{\rho_{X,0}}=(1-Y_p/2)\frac{m_e}{m_p}\frac{\Omega_bh^2}{\Omega_ch^2}=8.9\times10^{-5}\quad(\Omega_ch^2=0.120).
$$

A heavier electron at recombination then requires the dark matter to have been lighter by a fraction of at least $$8.9\times10^{-5}\delta$$, that is $$7\times10^{-7}$$ at $$\delta=0.78\%$$. A field of this kind whose coupling saturates at high density would produce a step in $$m_e$$; whether such a model fits the full CMB, and its consequences for nucleosynthesis, are open.

### Fields that track the local density today

Theorem 2 then compares the recombination plasma with a far denser environment and allows $$\delta>0$$. Spectroscopy of absorbing gas measures $$m_p/m_e$$ in other environments, and it constrains this route in two steps. If the gas has an electron scalar density below that of the recombination plasma, Theorem 2 gives $$m_e^{\rm gas}\ge m_e(z_*)=(1+\delta)m_e^{\rm lab}$$, so a limit on $$\lvert\Delta(m_p/m_e)/(m_p/m_e)\rvert$$ between gas and laboratory, attributed to $$m_e$$ alone, limits $$\delta$$ by the same amount. Methanol absorption in the lens towards PKS1830−211 at $$z=0.88582$$ limits the fractional change to $$4\times10^{-7}$$ at $$2\sigma$$ [[Kanekar]](#ref-kanekar), and ammonia and cyanopolyyne lines in Galactic molecular cores limit it to $$2\times10^{-8}$$ at $$3\sigma$$ [[Levshakov]](#ref-levshakov), both far below $$\delta\sim10^{-2}$$. The step applies only if the absorbing gas has an electron density below about $$290\ {\rm cm}^{-3}$$ counting bound electrons. Molecular clouds are typically denser, and then the theorem gives only $$m_e\le m_e(z_*)$$ there and no limit on $$\delta$$. Whether such a field can satisfy fifth-force and equivalence-principle bounds has not been checked.

## 7. Open points

Theorem 1(b) assumes a constant Hessian $$K$$; relaxation $$\dot\xi=-\Gamma(\xi,t)\nabla\Phi_t$$ with a non-quadratic $$\Phi_t$$ is not covered. The corrections to Theorem 1 from the kinetic energy of a heavy propagating auxiliary field, which tracks the minimum adiabatically, are not computed, and a proof for hysteresis is missing. Condition (T3) was verified at $$m=m_e$$ for an ideal plasma; Coulomb corrections and the full recombination history were not included. The excursion bound assumes that only the electron mass depends on $$\phi$$ and that the coupling does not grow along the path; a path along which $$d\ln m_e/d\phi$$ grows from a small value today interpolates between Theorem 3 and the quadratic class, and was not scanned. The numerical values of $$\delta$$ come from fits that vary $$m_e$$ by hand at recombination, with no field dynamics, and a field model changes $$m_e$$ at other epochs, in particular during nucleosynthesis, and adds its own energy density; fitting any of the models of section 6 to the data needs all of this.

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
12. <a id="ref-jps"></a>Josset, Perez, Sudarsky, “Dark Energy from Violation of Energy Conservation,” Phys. Rev. Lett. 118, 021102 (2017), [arXiv:1604.04183](https://arxiv.org/abs/1604.04183).
13. <a id="ref-dps"></a>Dowker, Philpott, Sorkin, “Energy-momentum diffusion from spacetime discreteness,” Phys. Rev. D 79, 124047 (2009), [arXiv:0810.5591](https://arxiv.org/abs/0810.5591).
14. <a id="ref-ps"></a>Perez, Sudarsky, “Dark energy from quantum gravity discreteness,” Phys. Rev. Lett. 122, 221302 (2019), [arXiv:1711.05183](https://arxiv.org/abs/1711.05183).
15. <a id="ref-ccl"></a>Cárdenas, Cruz, Lepe, “Generalized second law of thermodynamics for the matter creation scenario and emergence of phantom regime,” Eur. Phys. J. Plus 139, 642 (2024), [arXiv:2302.10155](https://arxiv.org/abs/2302.10155).
16. <a id="ref-desi"></a>DESI Collaboration, “DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints,” Phys. Rev. D 112, 083515 (2025), [arXiv:2503.14738](https://arxiv.org/abs/2503.14738).
17. <a id="ref-planck6"></a>Planck Collaboration, “Planck 2018 results. VI. Cosmological parameters,” Astron. Astrophys. 641, A6 (2020), [arXiv:1807.06209](https://arxiv.org/abs/1807.06209).
18. <a id="ref-act"></a>Calabrese et al. (ACT Collaboration), “The Atacama Cosmology Telescope: DR6 Constraints on Extended Cosmological Models,” [arXiv:2503.14454](https://arxiv.org/abs/2503.14454).
19. <a id="ref-toda-seto"></a>Toda, Seto, “Constraints on the varying electron mass and early dark energy in light of ACT DR6 and DESI DR2 and the implications for inflation,” JCAP 02 (2026) 019, [arXiv:2508.09025](https://arxiv.org/abs/2508.09025).
20. <a id="ref-weiner"></a>Weiner, “High-redshift physics from the acoustic scale,” JCAP 08 (2026) 072, [arXiv:2603.18131](https://arxiv.org/abs/2603.18131).
21. <a id="ref-bsw1"></a>Baryakhtar, Simon, Weiner, “Cosmology with varying fundamental constants from hyperlight, coupled scalars,” Phys. Rev. D 110, 083505 (2024), [arXiv:2405.10358](https://arxiv.org/abs/2405.10358).
22. <a id="ref-bsw2"></a>Baryakhtar, Simon, Weiner, “Searching for coupled, hyperlight scalars across cosmic history,” Phys. Rev. D 111, 115026 (2025), [arXiv:2502.04432](https://arxiv.org/abs/2502.04432).
23. <a id="ref-gby"></a>Ghosh, Boddy, Yu, “Early Universe Constraints on Variations in Fundamental Constants Induced by Ultralight Scalar Dark Matter,” [arXiv:2511.14532](https://arxiv.org/abs/2511.14532).
24. <a id="ref-microscope"></a>Touboul et al. (MICROSCOPE Collaboration), “MICROSCOPE mission: final results of the test of the Equivalence Principle,” Phys. Rev. Lett. 129, 121102 (2022), [arXiv:2209.15487](https://arxiv.org/abs/2209.15487).
25. <a id="ref-chameleon"></a>Khoury, Weltman, “Chameleon Fields: Awaiting Surprises for Tests of Gravity in Space,” Phys. Rev. Lett. 93, 171104 (2004), [arXiv:astro-ph/0309300](https://arxiv.org/abs/astro-ph/0309300).
26. <a id="ref-symmetron"></a>Hinterbichler, Khoury, “Symmetron Fields: Screening Long-Range Forces Through Local Symmetry Restoration,” Phys. Rev. Lett. 104, 231301 (2010), [arXiv:1001.4525](https://arxiv.org/abs/1001.4525).
27. <a id="ref-kanekar"></a>Kanekar et al., “Constraints on changes in the proton-electron mass ratio using methanol lines,” [arXiv:1412.7757](https://arxiv.org/abs/1412.7757).
28. <a id="ref-levshakov"></a>Levshakov et al., “Limits on the spatial variations of the electron-to-proton mass ratio in the Galactic plane,” Astron. Astrophys. 559, A91 (2013), [arXiv:1310.1850](https://arxiv.org/abs/1310.1850).
