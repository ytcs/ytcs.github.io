---
title: A matter-induced instability in Freezing Gravity
date: 2026-10-28
order: 9
description: Adding pressureless matter to the Freezing Gravity dark-energy model produces a sub-horizon instability whose growth rate does not depend on wavenumber. With constant parameters it affects every phantom phase and every nonzero beyond-Horndeski coupling, even on a cosmological-constant background, and no constant choice of that coupling stabilises the DESI DR2 fits.
---

Freezing Gravity is a dark-energy model in which the expansion history, including a crossing of the phantom divide $$w=-1$$, can be prescribed freely, while the linear perturbations are controlled by separate effective-field-theory (EFT) parameters $$\alpha_L$$, $$\alpha_T$$, $$\alpha_H$$ and $$\beta_3$$ [[YYS]](#ref-yys). Its stability conditions were derived without matter. We ask whether they survive once pressureless matter (dust) is added. They do not. For $$\alpha_L=0$$, $$0<\beta_3\ll1$$ and wavenumbers $$k\gg aH$$, the scalar mode, dressed by the dust, obeys an oscillator equation whose squared frequency $$B_mH^2/\beta_3$$ does not depend on $$k$$, with

$$
B_m=2(\alpha_H-\alpha_T)+3(1+\alpha_H)(1+w)\,\Omega_{\rm de}-3\alpha_H\,\Omega_m+\frac{2\dot\alpha_H}{H}. \tag{1}
$$

Here $$w$$ and $$\Omega_{\rm de}$$ are the equation of state and density fraction of the dark energy, $$\Omega_m$$ that of the dust, and a dot is a derivative in cosmic time. Where $$B_m<0$$ the perturbations grow at the rate $$\Gamma=H\sqrt{-B_m/\beta_3}$$. This is derived at leading order in $$\sqrt{\beta_3}$$ and $$aH/k$$ and confirmed by exact numerical evolution of the linear equations; for small $$\alpha_L>0$$ it holds in a band $$aH\ll k<k_{\max}$$ with $$k_{\max}\propto\alpha_L^{-1/2}$$, a statement supported numerically.

With constant EFT parameters the result has three consequences. At $$\alpha_H=\alpha_T=0$$ every phantom phase ($$w<-1$$) is unstable. Any constant $$\alpha_H\neq0$$ is unstable on a cosmological-constant background, and any constant $$\alpha_H>0$$ is unstable during matter domination whatever the dark energy. No constant $$\alpha_H$$ removes the instability from the phantom phase of the four DESI DR2 $$w_0w_a$$ fits; at the value $$\beta_3=10^{-4}$$ used in [[YYS]](#ref-yys), modes grow by $$23$$ to $$28$$ e-folds between $$z=2.33$$ and the crossing. The instability is avoided if $$\alpha_H$$ runs in time, if $$\alpha_T$$ is negative around $$z\sim1$$, if $$\beta_3\gtrsim\lvert B_m\rvert$$, or if $$\alpha_L$$ is large enough to close the band; the last case leaves near-horizon scales unresolved.

Yao, Ye and Silvestri [[YYS]](#ref-yys) introduced the model, derived its ghost and gradient conditions in vacuum, and noted that matter would modify them. Yao [[Yao]](#ref-yao) added a minimally coupled perfect fluid, derived the no-ghost conditions at all wavenumbers and the gradient condition at high wavenumber, and concluded that the model can realise arbitrary background evolutions, including phantom crossing, while maintaining stable perturbation dynamics. That the scalar's gradient energy is set by the dark-energy combination $$\rho_{\rm de}+p_{\rm de}$$, and not by the total, is a familiar feature of the EFT of dark energy: in the Horndeski sound speed of Bellini and Sawicki [[BS]](#ref-bs), eq. (3.13), the matter contribution to $$\dot H$$ cancels against an explicit $$\rho_m+p_m$$ term, and Creminelli et al. [[CDNV]](#ref-cdnv) showed that a stable $$w<-1$$ phase needs operators beyond those of quintessence. De Felice, Frusciante and Papadomanolakis [[DFP]](#ref-dfp) classified the instabilities of modified gravity with matter, including tachyonic ones whose rate greatly exceeds $$H$$. This note adds the coefficient (1) for Freezing Gravity with dust, in particular the terms $$-3\alpha_H\Omega_m$$ and $$2\dot\alpha_H/H$$, its consequences for constant parameters and for the DESI fits, and a set of parameters that satisfies every stability condition stated explicitly in [[Yao]](#ref-yao) and is nonetheless unstable (section 7).

Units are $$M_{\rm Pl}^2/2=1$$, as in [[YYS]](#ref-yys), so that the Friedmann equation reads $$6H^2=\rho_{\rm tot}$$. We write $$\kappa\equiv k/(aH)$$ for a comoving wavenumber in units of the comoving Hubble rate.

## 1. Freezing Gravity with dust

In the unitary gauge of the dark-energy scalar, the model is the action [[YYS]](#ref-yys), eq. (3.27),

$$
S_g=\int dt\,d^3x\,N\sqrt h\Big[\sigma_1\Big(K_{ij}K^{ij}-\tfrac13K^2\Big)+\rho_1R+\rho_2\,a_ia^i-\tfrac23K^2+4\mathcal F K-6\mathcal G^2\Big],
$$

where $$N$$ is the lapse, $$h_{ij}$$ the spatial metric, $$K_{ij}$$ its extrinsic curvature, $$R$$ its Ricci scalar and $$a_i=\partial_i\ln N$$. The coefficients $$\sigma_1,\rho_1,\rho_2$$ are functions of $$t$$ and $$N$$; $$\mathcal F$$ and $$\mathcal G$$ are functions of $$t$$ alone. The perturbations depend on the coefficients only through their values at $$N=1$$ and one derivative, which define the EFT parameters ([[YYS]](#ref-yys), eq. (4.2)):

$$
\sigma_1=\frac{1}{1+\alpha_L},\qquad \rho_1=\frac{1+\alpha_T}{1+\alpha_L},\qquad \rho_1+\partial_N\rho_1=\frac{1+\alpha_H}{1+\alpha_L},\qquad \rho_2=\frac{\beta_3}{1+\alpha_L}.
$$

Here $$\alpha_T$$ is the tensor speed excess, $$\alpha_H$$ the beyond-Horndeski coupling of $$R$$ to $$\delta N$$, $$\alpha_L$$ the detuning of the two $$K^2$$ terms, and $$\beta_3$$ the coefficient of the gradient energy $$(\partial_i\delta N)^2/a^2$$ of the lapse, in the basis of Gleyzes, Langlois and Vernizzi [[GLV]](#ref-glv) and Langlois et al. [[LMNV]](#ref-lmnv). Throughout we take $$\alpha_L\ge0$$ and $$\beta_3>0$$.

We add dust in the form

$$
S_m=-\frac12\int d^4x\sqrt{-g}\;\rho_m\,\big(g^{\mu\nu}\partial_\mu\phi\,\partial_\nu\phi+1\big),
$$

where $$\rho_m(x)$$ and $$\phi(x)$$ are independent fields. Varying $$\rho_m$$ makes $$\partial_\mu\phi$$ a unit timelike vector, so that $$\phi$$ is the proper time along geodesic worldlines; varying $$\phi$$ gives $$\nabla_\mu(\rho_m\partial^\mu\phi)=0$$; and on shell the stress tensor is $$\rho_m\,\partial_\mu\phi\,\partial_\nu\phi$$, which is pressureless matter of density $$\rho_m$$.

On a flat FRW background with $$N=1$$ and $$\phi=t$$, varying the lapse and the scale factor gives

$$
6H^2=6\mathcal G^2+\rho_m,\qquad \dot H=\dot{\mathcal F}-\frac{\rho_m}{4},\qquad \dot\rho_m=-3H\rho_m. \tag{2}
$$

The dark energy therefore has $$\rho_{\rm de}=6\mathcal G^2$$ and $$\rho_{\rm de}+p_{\rm de}=-4\dot{\mathcal F}$$, as in vacuum ([[YYS]](#ref-yys), eq. (3.25)), and

$$
-\frac{2\dot{\mathcal F}}{H^2}=3(1+w)\,\Omega_{\rm de}.
$$

In vacuum $$\dot H=\dot{\mathcal F}$$, and [[YYS]](#ref-yys), eq. (4.17), finds at $$\alpha_L=0$$ and $$\alpha_T=0$$ a kinetic coefficient $$\beta_3\kappa^2$$ and a gradient coefficient $$2\alpha_H-2(1+\alpha_H)\dot H/H^2+2\dot\alpha_H/H$$. At $$\alpha_H=0$$ the gradient coefficient is $$-2\dot H/H^2$$, negative in any phantom phase. With dust the two derivatives differ, and the question is which of them replaces $$\dot H$$. For the DESI+CMB+DESY5 fit (section 5) at $$z=0.93$$, on the phantom side of the crossing, $$-2\dot H/H^2=2.18$$, while $$-2\dot{\mathcal F}/H^2=3(1+w)\Omega_{\rm de}=-0.117$$. If the total $$\dot H$$ entered, matter would stabilise the phantom phase; if $$\dot{\mathcal F}$$ enters, it would not. Section 3 shows that $$\dot{\mathcal F}$$ enters, together with a new term $$-3\alpha_H\Omega_m$$.

## 2. The quadratic action

We perturb in unitary gauge, $$N=1+\delta N$$, $$N_i=\partial_i\psi$$, $$h_{ij}=a^2e^{2\zeta}\delta_{ij}$$, $$\phi=t+v$$, and $$\rho_m\to\rho_m+\delta\rho_m$$. The variable $$v$$ is the shift between the dust's proper time and the dark-energy clock $$t$$. For one Fourier mode of comoving wavenumber $$k$$, a direct expansion gives the following.

<div class="lemma" markdown="1">
**Lemma 1** (quadratic Lagrangian). Up to total time derivatives, after use of (2), and up to an overall positive constant, the second-order Lagrangian of Freezing Gravity with dust is

$$
\begin{aligned}
\frac{\ell}{a^3}={}&\frac{k^2}{a^2}\Big[\frac{\beta_3\,\delta N^2+4(1+\alpha_H)\,\zeta\,\delta N+2(1+\alpha_T)\,\zeta^2}{1+\alpha_L}-\frac{\rho_m}{2}v^2-4\psi\Big(\dot\zeta-H\delta N+\frac{\rho_m}{4}v\Big)\Big]-\frac{2\alpha_L}{3(1+\alpha_L)}\frac{k^4}{a^4}\psi^2\\
&-6(\dot\zeta-H\delta N)^2+3\rho_m\,\zeta\,\dot v+\delta\rho_m(\dot v-\delta N)+\frac{\rho_m}{2}(\dot v-\delta N)^2 .
\end{aligned} \tag{3}
$$

It holds for time-dependent $$\alpha_L,\alpha_T,\alpha_H,\beta_3$$.
</div>

<div class="proof" markdown="1">
*Proof.* Three facts reduce the gravitational part to the terms shown. The combination $$K_{ij}K^{ij}-\tfrac13K^2$$ is the square of the traceless part of $$K_{ij}$$, which vanishes on FRW; at second order it receives only the shift, $$\tfrac23(k^2/a^2)^2\psi^2$$, so $$\sigma_1$$ enters only through the $$\psi^2$$ term and its $$N$$-dependence not at all. Next, $$R$$ and $$a_i$$ vanish on FRW, so $$\rho_1$$ enters through $$\rho_1(1)\,\delta_2(\sqrt hR)$$ and $$(\rho_1+\partial_N\rho_1)(1)\,\delta N\,\delta_1R$$, and $$\rho_2$$ through $$\rho_2(1)(\partial\delta N)^2$$; higher $$N$$-derivatives multiply terms of cubic order. Finally $$N\sqrt h\,K=\partial_t\sqrt h-\partial_i(\sqrt hN^i)$$ identically, so $$4\mathcal FK$$ contributes $$-4\dot{\mathcal F}\sqrt h$$, independent of the lapse. The dust term is expanded from $$-g^{\mu\nu}\partial_\mu\phi\partial_\nu\phi=(\dot\phi-N^i\partial_i\phi)^2/N^2-h^{ij}\partial_i\phi\,\partial_j\phi$$. The remaining algebra is routine; we did it symbolically, starting from the full ADM expressions, and checked that the difference between that expansion and (3) has vanishing Euler–Lagrange derivative with respect to every field once (2) holds.
</div>

Two independent checks fix the normalisation and the signs. Without dust and at $$\alpha_L=0$$, the $$\psi$$ equation gives $$\delta N=\dot\zeta/H$$, and substituting it and integrating the $$\zeta\dot\zeta$$ term by parts reproduces the kinetic and gradient coefficients of [[YYS]](#ref-yys), eq. (4.17), including the $$2\dot\alpha_H/H$$ term. With dust, the $$\psi$$ and $$\delta N$$ equations of (3) coincide with the constraint equations (3.8) and (3.9) of [[Yao]](#ref-yao) under the dictionary $$A=\delta N$$, $$aB=\psi$$, $$v_m=v$$, $$\alpha_A=\beta_3$$, $$M^2=2/(1+\alpha_L)$$, $$\bar\rho_m\delta_m=\delta\rho_m+\rho_m(\dot v-\delta N)$$.

## 3. The fast mode and the coefficient $$B_m$$

The idea is that two constraints remove all but one fast degree of freedom, and that both remaining terms of its Lagrangian carry two spatial derivatives. The dust fixes the lapse: $$\delta\rho_m$$ appears only multiplying $$\dot v-\delta N$$, so

$$
\delta N=\dot v .
$$

At $$\alpha_L=0$$ the shift $$\psi$$ appears linearly, and its equation, the momentum constraint, reads $$\dot\zeta-H\dot v+\rho_mv/4=0$$. It is natural to trade $$\zeta$$ for

$$
\mathcal R\equiv\zeta-Hv ,
$$

the curvature perturbation on the slices of constant dust proper time, since $$a^2(t)e^{2\zeta}=a^2(t+v)e^{2\mathcal R}$$ to first order. With (2) the momentum constraint becomes

$$
\dot{\mathcal R}=-\dot{\mathcal F}\,v. \tag{4}
$$

On a cosmological-constant background $$\dot{\mathcal F}=0$$ and $$\mathcal R$$ is conserved on all scales, as in general relativity with dust and $$\Lambda$$. Otherwise $$\mathcal R$$ is sourced by the dark energy's $$\rho+p$$, and this is where $$\dot{\mathcal F}$$, not the total $$\dot H$$, enters.

The kinetic term of $$v$$ comes from the lapse gradient energy, $$\beta_3(k^2/a^2)\delta N^2=\beta_3(k^2/a^2)\dot v^2$$. The terms in (3) without time derivatives on $$v$$ also carry $$k^2/a^2$$ once $$\zeta=Hv+\mathcal R$$ is inserted, and the $$\zeta\,\delta N$$ term, after integration by parts, adds more of the same kind.

<div class="lemma" markdown="1">
**Lemma 2** (exact reduction). Let $$\alpha_L=0$$. After $$\delta N=\dot v$$, $$\zeta=Hv+\mathcal R$$ and $$\tilde\psi\equiv\psi+(1+\alpha_H)v$$, the Lagrangian (3) equals, up to a total time derivative,

$$
\begin{aligned}
\ell={}&a\,k^2\Big\{\beta_3\dot v^2-H^2B_m\,v^2+2(1+\alpha_T)\mathcal R^2-4\big[(\alpha_H-\alpha_T)H+\dot\alpha_H\big]\mathcal R\,v-4\tilde\psi\big(\dot{\mathcal R}+\dot{\mathcal F}v\big)\Big\}\\
&+a^3\Big\{3\rho_m(Hv+\mathcal R)\,\dot v-6(\dot{\mathcal R}+\dot Hv)^2\Big\},
\end{aligned} \tag{5}
$$

with $$B_m=2(\alpha_H-\alpha_T)-2(1+\alpha_H)\dot{\mathcal F}/H^2-3\alpha_H\Omega_m+2\dot\alpha_H/H$$, which is (1).
</div>

<div class="proof" markdown="1">
*Proof.* Solving the $$\delta\rho_m$$ equation for $$\delta N$$ and substituting is legitimate, and kills the last two terms of (3). In the $$k^2$$ bracket, $$4(1+\alpha_H)\zeta\dot v=4(1+\alpha_H)Hv\dot v+4(1+\alpha_H)\mathcal R\dot v$$. With the measure $$a^3\cdot k^2/a^2=ak^2$$ and $$\dot a=aH$$, integration by parts gives

$$
4a(1+\alpha_H)Hv\dot v\simeq-2\big[aH^2(1+\alpha_H)+aH\dot\alpha_H+a(1+\alpha_H)\dot H\big]v^2,
$$

$$
4a(1+\alpha_H)\mathcal R\dot v\simeq-4a\big[H(1+\alpha_H)+\dot\alpha_H\big]\mathcal Rv-4a(1+\alpha_H)\dot{\mathcal R}\,v .
$$

The $$\psi$$ term is $$-4\psi(\dot\zeta-H\dot v+\rho_mv/4)=-4\psi(\dot{\mathcal R}+\dot{\mathcal F}v)$$ by (2), and the last term above equals $$-4(1+\alpha_H)v(\dot{\mathcal R}+\dot{\mathcal F}v)+4(1+\alpha_H)\dot{\mathcal F}v^2$$; its first part is absorbed by $$\tilde\psi$$. Collecting the $$v^2$$ terms, including $$2(1+\alpha_T)H^2v^2$$ from $$\zeta^2$$ and $$-\rho_mv^2/2$$, and using $$\dot H=\dot{\mathcal F}-\rho_m/4$$ and $$\rho_m=6H^2\Omega_m$$:

$$
2(\alpha_T-\alpha_H)H^2-2H\dot\alpha_H+2(1+\alpha_H)\dot{\mathcal F}+\tfrac12\alpha_H\rho_m=-H^2B_m .
$$

The $$\mathcal Rv$$ terms are $$4(1+\alpha_T)H\mathcal Rv$$ from $$\zeta^2$$ and $$-4[H(1+\alpha_H)+\dot\alpha_H]\mathcal Rv$$ from above. The terms without $$k^2$$ are rewritten with $$\dot\zeta-H\dot v=\dot{\mathcal R}+\dot Hv$$. We also verified (5) symbolically against (3), with $$\alpha_T$$, $$\alpha_H$$ and $$\beta_3$$ arbitrary functions of time.
</div>

The step with content is the cancellation between the $$\dot H$$ produced by integrating $$Hv\dot v$$ by parts and the $$\dot{\mathcal F}$$ produced by the momentum constraint: matter's share of $$\dot H$$ is traded for $$\dot{\mathcal F}$$, leaving $$\tfrac12\alpha_H\rho_m$$ behind. At $$\alpha_H=0$$ that remainder vanishes and the $$v^2$$ coefficient is the quintessence-type gradient energy $$2\dot{\mathcal F}=-\tfrac12(\rho_{\rm de}+p_{\rm de})$$.

In (5) the dynamics separates by time scale. The variable $$v$$ has kinetic coefficient $$ak^2\beta_3$$ and potential coefficient $$ak^2H^2B_m$$, so it oscillates or grows at a rate of order $$H/\sqrt{\beta_3}$$. The variable $$\mathcal R$$ is slow: by (4) its rate of change is $$\dot{\mathcal F}v$$, of order $$H^2v$$.

<div class="proposition" markdown="1">
**Proposition 1** (fast mode). Let $$\alpha_L=0$$, $$0<\beta_3\ll\lvert B_m\rvert$$ and $$\kappa\gg1$$, and let the background and the EFT parameters vary on the Hubble time. Then the linear system (5) has a pair of fast solutions, with $$\lvert\dot v\rvert\gg H\lvert v\rvert$$, and along them

$$
\ddot v=-\frac{B_mH^2}{\beta_3}\,v\,\Big[1+O\big(\sqrt{\beta_3}\big)+O\big(\kappa^{-2}\big)\Big]. \tag{6}
$$

Where $$B_m<0$$ they grow at the rate $$\Gamma=H\sqrt{-B_m/\beta_3}$$, independent of $$k$$; $$\zeta=Hv+\mathcal R$$ grows with them.
</div>

<div class="proof" markdown="1">
*Proof.* This is a leading-order (WKB) statement, and the proof consists of size estimates. Suppose $$v$$ varies at a rate $$\Gamma\sim H/\sqrt{\beta_3}$$. The constraint (4) gives $$\mathcal R\sim(\dot{\mathcal F}/\Gamma)\,v\sim\sqrt{\beta_3}\,Hv$$. The $$\mathcal R$$ equation of (5), $$\tfrac{d}{dt}(4ak^2\tilde\psi)=-4ak^2(1+\alpha_T)\mathcal R+4ak^2[(\alpha_H-\alpha_T)H+\dot\alpha_H]v+(\text{terms of order }a^3)$$, then gives $$\tilde\psi=O(\sqrt{\beta_3})\,v+O(\kappa^{-2})\,v$$; the second piece comes from the terms of order $$a^3$$, which are of order $$a^3H^2\Gamma v$$ there. In the $$v$$ equation,

$$
\frac{d}{dt}\big(2ak^2\beta_3\dot v\big)+2ak^2H^2B_mv=-4ak^2\big[(\alpha_H-\alpha_T)H+\dot\alpha_H\big]\mathcal R-4ak^2\dot{\mathcal F}\tilde\psi+(\text{terms of order }a^3),
$$

the two terms on the left are of order $$ak^2H^2v$$ and the first two on the right of order $$\sqrt{\beta_3}\,ak^2H^2v+\kappa^{-2}ak^2H^2v$$. In the terms of order $$a^3$$, the $$v\dot v$$ contributions from $$3\rho_mHv\dot v$$ cancel identically in the Euler–Lagrange derivative, leaving terms of order $$a^3H^4v$$, a fraction $$\kappa^{-2}$$ of the left side. Dividing by $$2ak^2\beta_3$$ gives (6); the friction term $$(\dot a/a)\dot v$$ is a relative correction $$H/\Gamma=O(\sqrt{\beta_3})$$.
</div>

In the language of the EFT of dark energy, $$v$$ is minus the Goldstone time shift $$\pi$$ of the dark energy, seen from the dust's rest frame, and the $$v$$ part of (5) is $$a^3\big[\beta_3(\partial\dot\pi)^2-B_mH^2(\partial\pi)^2\big]/a^2$$. The kinetic term comes from $$a_ia^i$$ and carries two spatial gradients, which is why the rate does not depend on $$k$$. The instability is thus neither a gradient instability, whose rate grows with $$k$$, nor a mass term in the usual sense; it falls in the class that [[DFP]](#ref-dfp) call tachyonic, with $$\lvert\omega^2\rvert=\lvert B_m\rvert H^2/\beta_3\gg H^2$$. The derivation keeps $$\dot\alpha_H$$, which enters through an integration by parts and would be lost in a frozen-coefficient treatment of the action. For constant parameters, (6) agrees with the roots of the frozen-coefficient dispersion relation of the full equations of motion, computed to 50 digits at the parameter points we tried; for instance at $$(\alpha_H,\Omega_m,\dot H/H^2)=(-0.3,0.3,-0.9)$$, where $$B_m=0.300$$, the root is $$\beta_3\Gamma^2/H^2=-0.3004$$ at $$\beta_3\kappa^2=10^{-2}$$ and $$-0.3000$$ at $$\beta_3\kappa^2=10^{2}$$.

## 4. Checks and the finite-$$\alpha_L$$ band

**Exact evolution.** For $$\alpha_L>0$$ the shift appears quadratically in (3) and can be integrated out, which gives a regular two-variable Lagrangian in $$(\zeta,v)$$ with the penalty $$\tfrac{6(1+\alpha_L)}{\alpha_L}(\dot{\mathcal R}+\dot{\mathcal F}v)^2$$ in place of the constraint (4). We integrate its Euler–Lagrange equations with full time dependence, no frozen coefficients, on $$w_0w_a$$ (CPL) backgrounds with $$\beta_3=10^{-4}$$, and measure the local growth rate of $$(\zeta^2+H^2v^2)^{1/2}$$. Comoving wavenumbers are quoted in units of $$H_0$$; $$k=10H_0$$ is $$3.3\times10^{-3}\,h\,{\rm Mpc}^{-1}$$.

| background, $$\alpha_H$$ | $$z$$ | $$\kappa$$ | measured $$\Gamma/H$$ | $$\sqrt{-B_m/\beta_3}$$ |
|---|---|---|---|---|
| DESI+CMB+DESY5, $$0$$ | 1.5 | 10.6 | 26.1 | 29.3 |
| DESI+CMB+DESY5, $$0$$ | 1.0 | 11.1 | 32.2 | 34.1 |
| DESI+CMB+DESY5, $$0$$ | 0.8 | 11.2 | 32.0 | 33.7 |
| DESI+CMB+DESY5, $$0$$ | 0.5 | 11.1 | 22.0 | 21.9 |
| DESI+CMB+DESY5, $$+0.3$$ | 1.2 | 10.9 | 53.5 | 54.2 |
| DESI+CMB+DESY5, $$-0.3$$ | 0.6 | 11.1 | 28.5 | 29.5 |
| $$\Lambda$$CDM, $$+0.3$$ | 1.5 | 3.2 | 40.5 | 42.9 |
| $$\Lambda$$CDM, $$-0.3$$ | 0.2 | 2.2 | 44.3 | 46.3 |

The first four rows use $$\alpha_L=10^{-4}$$ and $$k=10H_0$$, the next two $$\alpha_L=10^{-5}$$, the last two $$\alpha_L=10^{-3}$$. Apart from the row at $$z=1.5$$, which is still affected by the initial transient of a run started at $$z=2.33$$, the measured rates lie up to 6% below the prediction, consistent with the $$O(\sqrt{\beta_3})$$ corrections of (6), which include the friction term $$-H/2$$ in the rate. On the $$\Lambda$$CDM background with $$\alpha_H=0$$, where $$B_m=0$$, all modes decay.

**The band edge.** For $$0<\alpha_L\ll1$$ the measured rates follow (6) with $$B_m$$ replaced by

$$
B_m+\tfrac23\alpha_L(1+\alpha_H)^2\kappa^2 . \tag{7}
$$

The added term is $$\beta_3c_s^2\kappa^2$$ at leading order in $$\beta_3$$, where $$c_s^2=\alpha_L[2(1+\alpha_H)^2-\beta_3(1+\alpha_T)]/[3\beta_3(1+\alpha_L)]$$ is the high-$$k$$ sound speed of the scalar ([[Yao]](#ref-yao), eq. (3.38)), so the squared frequency interpolates as $$B_mH^2/\beta_3+c_s^2k^2/a^2$$. A heuristic derivation produces exactly this term: at $$\alpha_L>0$$ the term $$-4ak^2(1+\alpha_H)\dot{\mathcal R}\,v$$ is no longer absorbed by a multiplier, and treating $$\dot{\mathcal R}$$ as an auxiliary variable with the penalty $$6a^3\alpha_L^{-1}\dot{\mathcal R}^2$$ and eliminating it adds $$-\tfrac23\alpha_L(1+\alpha_H)^2\kappa^2\,ak^2H^2v^2$$ to (5). On $$\Lambda$$CDM with $$\alpha_H=0.3$$ at $$z=1.5$$ ($$B_m=-0.184$$) and $$\alpha_L=10^{-3}$$, the measured rates at $$\kappa=5.4,\,8.6,\,10.7$$ are $$38.0,\,31.1,\,23.4$$ against $$39.0,\,31.8,\,23.4$$ from (7), and the mode at $$\kappa=12.9$$, just beyond the edge, grows at $$10.2$$ instead of the leading-order $$42.9$$. The instability therefore occupies the band

$$
aH\ll k<k_{\max},\qquad k_{\max}=\frac{aH}{1+\alpha_H}\sqrt{\frac{3\lvert B_m\rvert}{2\alpha_L}} .
$$

This statement is numerical; (7) is not derived to the standard of Proposition 1.

**Radiation.** Adding radiation as a fluid with $$w=c^2=\tfrac13$$ changes $$B_m$$ by a $$k$$-dependent amount: $$-4\alpha_H(1+\alpha_H)\Omega_r$$ for $$\beta_3\kappa^2\gg1$$ and $$-4\alpha_H\Omega_r$$ for $$\beta_3\kappa^2\ll1$$. We obtained these from the frozen-coefficient dispersion relation of the system with dust and radiation at leading order in $$\beta_3$$; for example at $$(\alpha_H,\Omega_m,\Omega_r,\dot H/H^2)=(-0.3,\,0.3,\,0.05,\,-0.9)$$ the roots are $$\beta_3\Gamma^2/H^2=-0.2206$$ and $$-0.2019$$ in the two regimes, against $$-0.220$$ and $$-0.202$$ from the formulae. The correction vanishes at $$\alpha_H=0$$ and is below $$2\times10^{-3}$$ for $$\lvert\alpha_H\rvert\le0.3$$ at $$z\le3$$, so it is neglected below.

## 5. Constant parameters

With $$\dot\alpha_H=0$$, three statements follow from (1) by inspection. Each holds in the band of section 4, at leading order.

<div class="corollary" markdown="1">
**Corollary 1** (phantom phases). If $$\alpha_H=\alpha_T=0$$, then $$B_m=3(1+w)\Omega_{\rm de}$$, and the band is stable exactly when $$w\ge-1$$.
</div>

<div class="corollary" markdown="1">
**Corollary 2** (constant $$\alpha_H$$ on $$\Lambda$$CDM). If $$w=-1$$ and $$\alpha_T=0$$, then $$B_m=\alpha_H(2-3\Omega_m)$$. A constant $$\alpha_H>0$$ is unstable while $$\Omega_m>\tfrac23$$, which includes the whole matter era, and a constant $$\alpha_H<0$$ is unstable while $$\Omega_m<\tfrac23$$. For $$\Omega_{m,0}=0.3027$$ the transition is at $$z=0.66$$.
</div>

The part concerning $$\alpha_H>0$$ does not need $$w=-1$$: whenever $$\Omega_m\to1$$ and $$(1+w)\Omega_{\rm de}\to0$$, (1) gives $$B_m\to-\alpha_H$$ (at $$\alpha_T=0$$). The vacuum condition of [[YYS]](#ref-yys), eqs. (4.15)–(4.16), allows constant $$\alpha_H\ge0$$ on a $$\Lambda$$CDM background; with dust the matter era excludes every $$\alpha_H>0$$. The exact evolution confirms it: from $$z=3$$ to $$0$$ at $$k=20H_0$$ and $$\alpha_L=10^{-5}$$, the amplitude grows by $$10^{15.0}$$, $$10^{8.4}$$, $$10^{4.1}$$ and $$10^{7.8}$$ for $$\alpha_H=0.3,\,0.1,\,-0.1,\,-0.3$$, against $$10^{15.3}$$, $$10^{8.8}$$, $$10^{4.9}$$ and $$10^{8.5}$$ from $$\exp\int\Gamma\,dt$$, and not at all for $$\alpha_H=0$$.

For a general background, write $$B_m=\alpha_H\,C(z)+3(1+w)\Omega_{\rm de}-2\alpha_T$$ with $$C\equiv2-3\Omega_m+3(1+w)\Omega_{\rm de}$$.

<div class="corollary" markdown="1">
**Corollary 3** (no constant $$\alpha_H$$). Let $$\alpha_T=0$$. If $$C(z_*)=0$$ at a redshift $$z_*$$ where $$w(z_*)<-1$$, then $$B_m(z_*)=3(1+w)\Omega_{\rm de}<0$$ for every constant $$\alpha_H$$.
</div>

We evaluate this on the DESI DR2 $$w_0w_a$$ fits that combine BAO, CMB and supernovae, using the posterior means of $$(\Omega_m,w_0,w_a)$$ in Table V of [[DESI]](#ref-desi) and $$w(a)=w_0+w_a(1-a)$$. In each, $$w$$ crosses $$-1$$ at $$z_c$$ and is phantom above it.

| fit | $$z_c$$ | min $$B_m$$ at $$\alpha_H=0$$ (at $$z$$) | $$z_*$$ | $$B_m(z_*)$$ | e-folds, $$z=2.33\to z_c$$ |
|---|---|---|---|---|---|
| DESI+CMB+DESY5 | 0.405 | $$-0.117$$ (0.93) | 0.61 | $$-0.085$$ | 24.2 (23.0) |
| DESI+CMB+Union3 | 0.440 | $$-0.137$$ (0.96) | 0.61 | $$-0.089$$ | 25.3 (24.2) |
| DESI+CMB+Pantheon+ | 0.354 | $$-0.096$$ (0.89) | 0.61 | $$-0.078$$ | 22.9 (21.6) |
| DESI+CMB | 0.496 | $$-0.192$$ (0.99) | 0.63 | $$-0.104$$ | 28.2 (27.3) |

The last column is $$\int\Gamma\,dt$$ at $$\alpha_H=\alpha_T=0$$ and $$\beta_3=10^{-4}$$, from $$z=2.33$$, the effective redshift of the highest-redshift DESI BAO measurement, to the crossing; in parentheses the growth measured in the exact evolution at $$k=20H_0$$, $$\alpha_L=10^{-5}$$. In all four fits $$z_*$$ lies on the phantom side, so Corollary 3 applies: no constant $$\alpha_H$$ stabilises them. A constant $$\alpha_T$$ below about $$-0.04$$ to $$-0.05$$ would make $$B_m(z_*)\ge0$$, but the speed of gravitational waves measured with GW170817 [[GW170817]](#ref-gw) bounds a constant $$\lvert\alpha_T\rvert$$ far below that. The growth scales as $$\beta_3^{-1/2}$$, so it is $$e^{24}$$ at $$\beta_3=10^{-4}$$ and $$e^{2.4}$$ at $$\beta_3=10^{-2}$$ for the first fit. Extending the CPL form beyond the data to $$z=20$$ raises the first entry to $$34$$ e-folds.

## 6. Ways out

The hypotheses of Corollaries 1–3 are constant parameters, $$\beta_3\ll\lvert B_m\rvert$$ and a band that exists. Relaxing each gives a way out; we list them with the conditions under which they work, all on the DESI+CMB+DESY5 background.

*A running $$\alpha_H$$.* The term $$2\dot\alpha_H/H$$ in (1) can make $$B_m$$ positive throughout. For $$\alpha_H=s\,(\ln a+0.5)$$, which runs from $$-0.89\,s$$ at $$z=3$$ to $$0.5\,s$$ today, the exact evolution ($$k=20H_0$$, $$\alpha_L=10^{-5}$$, $$z=3\to0$$) gives

| $$s$$ | min $$B_m$$ | growth, measured | from (1) | from (1) without $$2\dot\alpha_H/H$$ |
|---|---|---|---|---|
| 0 | $$-0.117$$ | $$10^{10.9}$$ | $$10^{11.9}$$ | $$10^{11.9}$$ |
| 0.03 | $$-0.056$$ | $$10^{4.5}$$ | $$10^{4.4}$$ | $$10^{10.5}$$ |
| 0.06 | $$+0.006$$ | $$10^{0.3}$$ | $$10^{0}$$ | $$10^{9.3}$$ |
| 0.10 | $$+0.088$$ | $$10^{0.1}$$ | $$10^{0}$$ | $$10^{8.3}$$ |

The comparison confirms the $$2\dot\alpha_H/H$$ term. A rate $$\dot\alpha_H/H\gtrsim0.06$$ satisfies the vacuum condition $$\dot\alpha_H\ge0$$ of [[YYS]](#ref-yys), eq. (4.16). This route is closed if $$\alpha_H$$ must vanish: the decay of gravitational waves into dark-energy fluctuations forces $$\alpha_H=0$$ in the EFT of dark energy [[CLTV]](#ref-cltv). Whether that argument carries over to Freezing Gravity, whose scalar has the additional operator $$a_ia^i$$, is not settled here. With $$\alpha_H\equiv0$$, Corollary 1 leaves only the remaining routes.

*A negative $$\alpha_T$$ at intermediate redshift.* A constant $$\alpha_T=-0.03$$ ($$-0.07$$) reduces the growth from $$10^{10.9}$$ to $$10^{5.8}$$ ($$10^{0.8}$$). GW170817 constrains $$\alpha_T$$ at $$z\approx0$$ only, so an $$\alpha_T(z)$$ that is about $$-0.06$$ near $$z\sim1$$ and vanishes today is not excluded by it.

*A large $$\beta_3$$.* For $$\beta_3\gtrsim\lvert B_m\rvert_{\max}\approx0.1$$–$$0.2$$ the rate drops to order $$H$$, Proposition 1 no longer applies, and the total growth $$\exp\big(\int\sqrt{-B_m}\,H\,dt/\sqrt{\beta_3}\big)$$ is about a factor 3 at $$\beta_3=0.1$$, even with the CPL form extended to $$z=20$$. The vacuum condition $$\beta_3\le2(1+\alpha_H)^2$$ of [[YYS]](#ref-yys), eq. (4.15), still holds there.

*A large $$\alpha_L$$.* For $$\alpha_L\gtrsim\tfrac32\lvert B_m\rvert\approx0.2$$–$$0.3$$, $$k_{\max}$$ falls below $$aH$$ and the band closes. The near-horizon modes are then not described by Proposition 1, and they still grow in the exact evolution: at $$\alpha_L=0.05$$ the amplitude at $$k=H_0$$ grows by $$10^{14.6}$$ between $$z=3$$ and $$0$$, while at $$k=3H_0$$ and $$10H_0$$ it does not grow. Whether a large $$\alpha_L$$ is a genuine way out therefore depends on the scales $$k\lesssim aH$$, which this note does not resolve.

## 7. Relation to the stability conditions with matter

For $$\alpha_L>0$$, Yao [[Yao]](#ref-yao) states the conditions $$\alpha_L>-1$$ (its eq. (3.24)), $$\alpha_A>0$$ (3.25), the no-ghost conditions (3.26), which for $$\alpha_L>0$$ hold at all $$k$$ when $$\epsilon_m<(\sqrt{1+\alpha_L}+\sqrt{\alpha_L})^2$$, and the high-$$k$$ gradient condition (3.39), $$(1+\alpha_H)^2>\tfrac12\alpha_A(1+\alpha_T)>0$$. Its abstract concludes that the separation of background and perturbation parameters allows arbitrary background evolutions, including phantom crossing, while maintaining stable perturbation dynamics.

Take the DESI+CMB+DESY5 background with $$\alpha_A=\beta_3=10^{-4}$$, $$\alpha_H=\alpha_T=0$$ and $$\alpha_L=10^{-4}$$. Then (3.24) and (3.25) hold, $$\epsilon_m=\tfrac34\beta_3\Omega_m<10^{-4}$$, so (3.26) holds at all $$k$$, and (3.39) reads $$1>5\times10^{-5}$$. The tensor condition holds, and since all parameters are constant the vacuum conditions (4.15)–(4.16) of [[YYS]](#ref-yys) also hold if the total $$\dot H$$, which is negative, is inserted into them. Yet at $$k=10H_0$$ the exact evolution grows by $$22.5$$ e-folds between $$z=2.33$$ and the crossing ($$24.2$$ predicted by (1)). The same holds at $$\alpha_L=10^{-3}$$, where the amplitude at $$k=3H_0$$ grows by $$10^{11.8}$$ from $$z=3$$ to $$0$$, and at $$k=5H_0$$ by $$21.5$$ e-folds between $$z=2.33$$ and the crossing. The no-ghost results of [[Yao]](#ref-yao) are not affected; the growth is a property of the gradient sector.

The reason the explicit conditions miss it is the order of limits. The high-$$k$$ expansion of [[Yao]](#ref-yao), eq. (3.34), is an expansion in inverse powers of $$k$$ at fixed $$\alpha_L\alpha_A\neq0$$, and its next-to-leading coefficients carry inverse powers of $$\alpha_L$$ (its Appendix C). Its leading term is the $$c_s^2k^2/a^2$$ of (7), which dominates only for $$\tfrac23\alpha_L(1+\alpha_H)^2\kappa^2\gg\lvert B_m\rvert$$, beyond $$k_{\max}$$. Inside the band, where the $$k$$-independent term $$B_mH^2/\beta_3$$ dominates, (3.39) says nothing; as $$\alpha_L\to0$$ the band extends to all sub-horizon $$k$$. The general dispersion condition (3.43) of [[Yao]](#ref-yao) is stated without being evaluated on a phantom-crossing background, and we have not evaluated it in that paper's variables. The example shows that the explicitly stated conditions are not sufficient, and that stability with matter depends on the background through $$w$$ and $$\Omega_m$$, so the background and perturbation sectors are not separated as far as stability is concerned.

## 8. Limitations and open questions

Proposition 1 is a leading-order statement in $$\sqrt{\beta_3}$$ and $$1/\kappa$$, obtained by size estimates on the exact reduced Lagrangian (5), not a bound on solutions; the exact evolution supports it to a few per cent for $$\beta_3=10^{-4}$$ and $$\kappa\approx10$$. The band edge (7) is established only numerically. The matter is pressureless and minimally coupled; the radiation correction is computed with frozen coefficients and constant parameters. The DESI numbers use CPL backgrounds with posterior means; they illustrate the size of the effect, while Corollaries 1–3 depend only on the sign structure of (1).

Several questions are open. The scales $$k\lesssim aH$$ are not resolved: the exact evolution shows growth there even when the band is closed, and [[Yao]](#ref-yao) finds that the kinetic matrix degenerates as $$k\to0$$. The branch $$-1<\alpha_L<0$$ is not treated. It is not known whether the gravitational-wave decay bound $$\alpha_H=0$$ applies to Freezing Gravity, which decides whether a running $$\alpha_H$$ is available. The growth factors exceed $$10^{10}$$, so in an unstable model linear theory fails at a time set by the initial amplitude of $$v$$; what happens after that is not addressed. Within the band, a Freezing Gravity realisation of the DESI phantom crossing requires one of the conditions of section 6, for instance $$\beta_3\gtrsim0.1$$, a running $$\alpha_H$$ or an $$\alpha_T$$ that departs from zero near $$z\sim1$$; which of them, if any, is compatible with large-scale-structure and lensing data has not been examined.

## References

1. <a id="ref-yys"></a>Yao, Ye, Silvestri, “A general model for dark energy crossing the phantom divide,” JCAP 10 (2025) 078, [arXiv:2508.01378](https://arxiv.org/abs/2508.01378).
2. <a id="ref-yao"></a>Yao, “Effective field theory for Freezing Gravity with minimally coupled matter,” [arXiv:2608.20085](https://arxiv.org/abs/2608.20085).
3. <a id="ref-bs"></a>Bellini, Sawicki, “Maximal freedom at minimum cost: linear large-scale structure in general modifications of gravity,” JCAP 07 (2014) 050, [arXiv:1404.3713](https://arxiv.org/abs/1404.3713).
4. <a id="ref-cdnv"></a>Creminelli, D’Amico, Noreña, Vernizzi, “The effective theory of quintessence: the w<−1 side unveiled,” JCAP 02 (2009) 018, [arXiv:0811.0827](https://arxiv.org/abs/0811.0827).
5. <a id="ref-dfp"></a>De Felice, Frusciante, Papadomanolakis, “On the stability conditions for theories of modified gravity in presence of matter fields,” JCAP 03 (2017) 027, [arXiv:1609.03599](https://arxiv.org/abs/1609.03599).
6. <a id="ref-glv"></a>Gleyzes, Langlois, Vernizzi, “A unifying description of dark energy,” [arXiv:1411.3712](https://arxiv.org/abs/1411.3712).
7. <a id="ref-lmnv"></a>Langlois, Mancarella, Noui, Vernizzi, “Effective description of higher-order scalar-tensor theories,” JCAP 05 (2017) 033, [arXiv:1703.03797](https://arxiv.org/abs/1703.03797).
8. <a id="ref-desi"></a>DESI Collaboration, “DESI DR2 results II: measurements of baryon acoustic oscillations and cosmological constraints,” [arXiv:2503.14738](https://arxiv.org/abs/2503.14738).
9. <a id="ref-gw"></a>LIGO Scientific Collaboration, Virgo Collaboration, Fermi GBM, INTEGRAL, “Gravitational waves and gamma-rays from a binary neutron star merger: GW170817 and GRB 170817A,” [arXiv:1710.05834](https://arxiv.org/abs/1710.05834).
10. <a id="ref-cltv"></a>Creminelli, Lewandowski, Tambalo, Vernizzi, “Gravitational wave decay into dark energy,” [arXiv:1809.03484](https://arxiv.org/abs/1809.03484).
