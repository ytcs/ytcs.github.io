---
title: A matter-induced instability in Freezing Gravity
date: 2026-10-29
category: dark-sector
order: 6
description: At leading order in a small lapse-gradient coefficient and for small kinetic detuning, Freezing Gravity with minimally coupled dust has a sub-horizon scalar instability whose growth rate does not depend on the wavenumber, in phantom phases without beyond-Horndeski coupling or tensor-speed excess, and no constant beyond-Horndeski coupling removes it on the phantom side of four DESI DR2 fits.
status: The quadratic action and an exact reduction of the dust system at zero kinetic detuning are derived by direct expansion. The growth rate and the edge of the unstable band are derivations at leading order in the lapse-gradient coefficient, compared with numerical integration of the linear equations. DESI numbers use CPL posterior means.
excludes: At leading order in a small lapse-gradient coefficient, for minimally coupled dust without radiation, constant parameters and modes below the band edge, excluded are phantom phases with zero beyond-Horndeski coupling and tensor-speed excess, and constant coupling on the phantom side of four DESI DR2 fits at zero tensor-speed excess. Larger coefficients and running couplings are not excluded.
---

Freezing Gravity is a dark-energy model whose expansion history, including a crossing of the phantom divide $$w=-1$$, can be prescribed freely, while four parameters control the linear perturbations [[YYS]](#ref-yys). Without matter the perturbations are stable under conditions on those parameters. Once pressureless matter is added they are not, if the lapse-gradient coefficient $$\beta_3$$ and the detuning $$\alpha_L$$ of the two kinetic terms are small (both defined in section 1). Between the Hubble scale and a wavenumber $$k_{\max}$$ set by $$\alpha_L$$ (section 5), the scalar mode, once the dust variable is eliminated, obeys an oscillator equation whose squared frequency $$B_mH^2/\beta_3$$ does not depend on the wavenumber. At leading order in $$\sqrt{\beta_3}$$ it grows at the rate $$\Gamma=H\sqrt{-B_m/\beta_3}$$ wherever $$B_m<0$$. With zero beyond-Horndeski coupling and zero tensor-speed excess, $$B_m<0$$ in every phantom phase.

The coefficient $$B_m$$ is

$$
B_m=2(\alpha_H-\alpha_T)+3(1+\alpha_H)(1+w)\,\Omega_{\rm de}-3\alpha_H\,\Omega_m+\frac{2\dot\alpha_H}{H}. \tag{1}
$$

Here $$H$$ is the Hubble rate and a dot a derivative in cosmic time; $$w$$ and $$\Omega_{\rm de}$$ are the equation of state and the density fraction of the dark energy, and $$\Omega_m$$ is the density fraction of the dust; $$\alpha_T$$ is the excess of the tensor-wave speed over the speed of light; and $$\alpha_H$$ couples the spatial curvature to the lapse perturbation, a coupling that is absent in Horndeski theories. For constant parameters and $$\alpha_T=0$$, $$B_m<0$$ also for every $$\alpha_H>0$$ in a matter era, and at some epoch for every $$\alpha_H\neq0$$ on a cosmological-constant background (Corollary 2). On the phantom side of the four DESI DR2 $$w_0w_a$$ fits, no constant $$\alpha_H$$ makes $$B_m$$ non-negative throughout (Corollary 3). At $$\beta_3=10^{-4}$$ the computed leading-order growth between $$z=2.33$$ and the crossing of $$w=-1$$ is 23 to 28 e-folds at $$\alpha_H=0$$ on the four fits, and at least 4.7 e-folds for any constant $$\alpha_H$$ between $$-6$$ and $$6$$.

These statements are leading order in $$\sqrt{\beta_3}$$ and require $$\beta_3\ll\lvert B_m\rvert$$, where $$\lvert B_m\rvert\lesssim0.2$$ on the DESI backgrounds. At $$\beta_3=10^{-2}$$ the integrated growth in the exact linear evolution is already a fraction of an e-fold at the wavenumbers tested (section 8). A running $$\alpha_H$$ and a negative $$\alpha_T$$ near $$z\sim1$$ also lie outside the result, and for a large $$\alpha_L$$ the question is unresolved; section 8 gives the conditions.

The model and its vacuum stability conditions are due to Yao, Ye and Silvestri [[YYS]](#ref-yys). Yao [[Yao]](#ref-yao) added a minimally coupled perfect fluid, derived the no-ghost conditions at all wavenumbers and a gradient condition at high wavenumber, and presented the model as realising arbitrary backgrounds with stable perturbations. Those conditions hold at the parameters used here (section 7); the instability lies below the wavenumber $$k_{\max}$$ above which the high-wavenumber gradient condition applies. The dependence of the scalar's gradient energy on $$\rho_{\rm de}+p_{\rm de}$$ rather than on the total $$\rho+p$$ already appears in the effective field theory of dark energy: in the Horndeski sound speed of Bellini and Sawicki [[BS]](#ref-bs), eq. (3.13), the matter contribution to $$\dot H$$ cancels against an explicit $$\tilde\rho_m+\tilde p_m$$ term, and Creminelli et al. [[CDNV]](#ref-cdnv) showed that a stable $$w<-1$$ phase needs operators beyond those of quintessence. De Felice, Frusciante and Papadomanolakis [[DFP]](#ref-dfp) classified the instabilities of modified gravity with matter, including tachyonic ones with rates far above $$H$$. The new result is the coefficient (1) for Freezing Gravity with dust, which follows from an exact reduction of the dust system (Lemma 2).

DESI data are also the starting point of [negative neutrino mass in BAO data](/quantum-gravity/neutrino-mimic/), which asks what a negative neutrino mass sum inferred from them requires of the dark sector, and of [the sign of vacuum-matter energy exchange](/quantum-gravity/vacuum-matter-sign/), which shows that three classes of vacuum dynamics transfer energy from the vacuum to matter in the senses stated there (at each time for a vacuum that tracks the global minimum of its effective potential, cumulatively since the start for passive linear relaxation from equilibrium, and over one step from a passive state for a unital map), the sign opposite to the flow from matter into the vacuum that $$w_0>-1$$, $$w_a<0$$ asks for during part of the history when dark matter is otherwise conserved.

Units are $$M_{\rm Pl}^2/2=1$$, as in [[YYS]](#ref-yys), so that the Friedmann equation reads $$6H^2=\rho_{\rm tot}$$.

## 1. Background, perturbations and the model

The universe is described by a flat Friedmann–Robertson–Walker metric $$ds^2=-N^2dt^2+a(t)^2d\mathbf x^2$$ with lapse $$N=1$$ on the background, scale factor $$a(t)$$ and Hubble rate $$H=\dot a/a$$. Two components are present: pressureless matter (dust) with density $$\rho_m\propto a^{-3}$$, and a dark-energy component with density $$\rho_{\rm de}$$ and pressure $$p_{\rm de}=w\rho_{\rm de}$$. The density fractions are $$\Omega_m=\rho_m/(6H^2)$$ and $$\Omega_{\rm de}=\rho_{\rm de}/(6H^2)$$, with $$\Omega_m+\Omega_{\rm de}=1$$. A cosmological constant has $$w=-1$$. The line $$w=-1$$ is the phantom divide: a component with $$w<-1$$ is called phantom, and in the models considered here its density grows with time. Baryon acoustic oscillation (BAO) measurements by DESI, combined with the cosmic microwave background (CMB) and a supernova sample (DES-Y5, written DESY5, Union3 or Pantheon+), prefer a dark-energy equation of state $$w(a)=w_0+w_a(1-a)$$ (the CPL form) with $$w_0>-1$$ and $$w_a<0$$ [[DESI]](#ref-desi). With $$w_0+w_a<-1$$, $$w$$ is phantom at early times and crosses $$-1$$ at $$a_c=1+(1+w_0)/w_a$$, that is at redshift $$z_c=1/a_c-1$$. The fit to DESI, CMB and DESY5 data is written DESI+CMB+DESY5 below; its parameters are in section 6.

A small perturbation is decomposed into Fourier modes of comoving wavenumber $$k$$. A mode is sub-horizon when $$\kappa\equiv k/(aH)\gg1$$, that is, when its wavelength is much shorter than the Hubble radius. We work to linear order, for one mode at a time, in the gauge in which the dark-energy scalar is uniform on constant-time slices (unitary gauge), so that the scalar's own perturbation is carried by the metric.

In unitary gauge Freezing Gravity is defined by the action [[YYS]](#ref-yys), eq. (3.27),

$$
S_g=\int dt\,d^3x\,N\sqrt h\Big[\sigma_1\Big(K_{ij}K^{ij}-\tfrac13K^2\Big)+r_1\,{}^{(3)}\!R+r_2\,\mathfrak a_i\mathfrak a^i-\tfrac23K^2+4\mathcal F K-6\mathcal G^2\Big], \tag{2}
$$

where $$h_{ij}$$ is the spatial metric with determinant $$h$$, $$K_{ij}$$ its extrinsic curvature with trace $$K$$, $${}^{(3)}\!R$$ its Ricci scalar, and $$\mathfrak a_i=\partial_i\ln N$$ the acceleration of the normal to the slices. The coefficients $$\sigma_1,r_1,r_2$$ (written $$\sigma_1,\rho_1,\rho_2$$ in [[YYS]](#ref-yys)) are functions of $$t$$ and $$N$$; $$\mathcal F$$ and $$\mathcal G$$ are functions of $$t$$ alone. For $$\sigma_1=r_1=1$$, $$r_2=0$$ and $$\mathcal F=0$$ the bracket is the Einstein–Hilbert Lagrangian plus a cosmological constant $$\propto\mathcal G^2$$. The terms $$4\mathcal FK$$ and $$-6\mathcal G^2$$ fix the background: on the flat background $$\rho_{\rm de}=6\mathcal G^2$$ and $$\rho_{\rm de}+p_{\rm de}=-4\dot{\mathcal F}$$ (shown below), so any $$w(t)$$ can be realised by choosing $$\mathcal F(t),\mathcal G(t)$$. The perturbations depend on the coefficients only through their values at $$N=1$$ and one derivative, which define the effective-field-theory parameters ([[YYS]](#ref-yys), eq. (4.2)):

$$
\sigma_1=\frac{1}{1+\alpha_L},\qquad r_1=\frac{1+\alpha_T}{1+\alpha_L},\qquad r_1+\partial_Nr_1=\frac{1+\alpha_H}{1+\alpha_L},\qquad r_2=\frac{\beta_3}{1+\alpha_L}.
$$

The parameter $$\alpha_T$$ is the excess of the tensor speed over the speed of light, $$\alpha_H$$ the beyond-Horndeski coupling of the curvature to the lapse perturbation (it vanishes in Horndeski theories), $$\alpha_L$$ the detuning of the two $$K^2$$ terms, and $$\beta_3$$ the coefficient of the lapse gradient energy $$(\partial_i\delta N)^2/a^2$$, in the notation of Gleyzes, Langlois and Vernizzi [[GLV]](#ref-glv) and Langlois et al. [[LMNV]](#ref-lmnv). Throughout, $$\alpha_L\ge0$$ and $$\beta_3>0$$. In vacuum the scalar mode of this model has kinetic coefficient $$\beta_3\kappa^2$$ and gradient coefficient $$2\alpha_H-2(1+\alpha_H)\dot H/H^2+2\dot\alpha_H/H$$ at $$\alpha_L=\alpha_T=0$$ ([[YYS]](#ref-yys), eq. (4.17)). At $$\alpha_H=0$$ the gradient coefficient is $$-2\dot H/H^2$$, negative in any phantom phase, so the vacuum conditions of [[YYS]](#ref-yys), eqs. (4.15)–(4.16), require a sufficiently large $$\alpha_H$$ there. With dust, the derivative that enters is $$\dot{\mathcal F}$$, the dark-energy part of $$\dot H$$ (section 3). For the DESI+CMB+DESY5 fit at $$z=0.93$$, on the phantom side, the vacuum formula evaluated with the total $$\dot H$$ of the background with dust would give $$-2\dot H/H^2=2.18$$, whereas $$-2\dot{\mathcal F}/H^2=3(1+w)\Omega_{\rm de}=-0.117$$. The dust therefore does not stabilise the phantom phase.

Pressureless matter is added in the form

$$
S_m=-\frac12\int d^4x\sqrt{-g}\;\rho_m\,\big(g^{\mu\nu}\partial_\mu\phi\,\partial_\nu\phi+1\big),
$$

where $$\rho_m(x)$$ and $$\phi(x)$$ are independent fields. Varying $$\rho_m$$ makes $$\partial_\mu\phi$$ a unit timelike vector, so $$\phi$$ is proper time along geodesic worldlines; varying $$\phi$$ gives $$\nabla_\mu(\rho_m\partial^\mu\phi)=0$$; and on shell the stress tensor is $$\rho_m\partial_\mu\phi\,\partial_\nu\phi$$, pressureless matter of density $$\rho_m$$. On the flat background with $$\phi=t$$, the lapse equation of (2) plus $$S_m$$ is $$6H^2-6\mathcal G^2-\rho_m=0$$, and the scale-factor equation then gives

$$
6H^2=6\mathcal G^2+\rho_m,\qquad \dot H=\dot{\mathcal F}-\frac{\rho_m}{4},\qquad \dot\rho_m=-3H\rho_m. \tag{3}
$$

So $$\rho_{\rm de}=6\mathcal G^2$$ and $$\rho_{\rm de}+p_{\rm de}=-4\dot{\mathcal F}$$ as claimed, and $$-2\dot{\mathcal F}/H^2=3(1+w)\Omega_{\rm de}$$. Radiation is not included. Its density fraction is about $$10^{-3}$$ in the redshift range used below ($$1.1\times10^{-3}$$ at $$z=3$$ and $$1.4\times10^{-3}$$ at $$z=4$$ for a present-day radiation fraction of $$9\times10^{-5}$$), and its effect on (1) is not derived.

The results below are stated under the following hypotheses, and the conditions of section 8 are the failures of each.

- (H1) $$\alpha_L=0$$ in the exact reduction of section 3; section 5 gives the leading-order modification for small $$\alpha_L>0$$.
- (H2) The only matter is the dust $$S_m$$, minimally coupled, in a flat background (3).
- (H3) $$\alpha_T,\alpha_H,\beta_3$$ are smooth functions of time that change on the Hubble time, and $$\beta_3>0$$.
- (H4) $$\varepsilon\equiv\sqrt{\beta_3}$$ is small compared with $$\sqrt{|B_m|}$$.

## 2. The quadratic action

We perturb in unitary gauge: $$N=1+\delta N$$, $$N_i=\partial_i\psi$$, $$h_{ij}=a^2e^{2\zeta}\delta_{ij}$$, $$\phi=t+v$$, $$\rho_m\to\rho_m+\delta\rho_m$$. The variable $$v$$ is the shift between the dust's proper time and the dark-energy clock $$t$$. For one Fourier mode of comoving wavenumber $$k$$:

<div class="lemma" markdown="1">
**Lemma 1** (quadratic Lagrangian). Up to total time derivatives, after use of (3), and up to an overall positive constant, the second-order Lagrangian of Freezing Gravity with dust is

$$
\begin{aligned}
\frac{\ell}{a^3}={}&\frac{k^2}{a^2}\Big[\frac{\beta_3\,\delta N^2+4(1+\alpha_H)\,\zeta\,\delta N+2(1+\alpha_T)\,\zeta^2}{1+\alpha_L}-\frac{\rho_m}{2}v^2-4\psi\Big(\dot\zeta-H\delta N+\frac{\rho_m}{4}v\Big)\Big]-\frac{2\alpha_L}{3(1+\alpha_L)}\frac{k^4}{a^4}\psi^2\\
&-6(\dot\zeta-H\delta N)^2+3\rho_m\,\zeta\,\dot v+\delta\rho_m(\dot v-\delta N)+\frac{\rho_m}{2}(\dot v-\delta N)^2 .
\end{aligned} \tag{4}
$$

It holds for time-dependent $$\alpha_L,\alpha_T,\alpha_H,\beta_3$$.
</div>

<div class="proof" markdown="1">
*Proof.* We expand each term of (2) and of $$S_m$$ to second order for fields proportional to $$\cos kx$$. The average of $$\cos^2$$ gives a common factor $$\tfrac12$$, which we drop; then $$\partial_x^2\to-k^2$$ and $$\partial_xf\,\partial_xg\to k^2fg$$. We use $$\sqrt h=a^3e^{3\zeta}$$, $$N^i=h^{ij}\partial_j\psi$$, $$K=(3(H+\dot\zeta)-\nabla_iN^i)/N$$ with $$\nabla_iN^i=a^{-2}e^{-2\zeta}(\partial^2\psi+\partial\zeta\cdot\partial\psi)$$, and $$\sqrt h\,{}^{(3)}\!R=-ae^{\zeta}\big(4\partial^2\zeta+2(\partial\zeta)^2\big)$$.

The traceless part $$E_{ij}=K_{ij}-\tfrac13Kh_{ij}$$ vanishes on the background and equals $$-(\nabla_i\nabla_j-\tfrac13h_{ij}\nabla^2)\psi$$ at first order. Since $$K_{ij}K^{ij}-\tfrac13K^2=E_{ij}E^{ij}$$ is already second order, only $$\sigma_1(1)$$ enters, and the term is $$\tfrac23\sigma_1k^4\psi^2/a$$. The term $$-\tfrac23N\sqrt h\,K^2$$ gives

$$
-6a^3(\dot\zeta-H\delta N)^2-a^3\big(36H\zeta\dot\zeta-18H^2\zeta\,\delta N+27H^2\zeta^2\big)-4ak^2\psi(\dot\zeta-H\delta N)-\tfrac23\frac{k^4}{a}\psi^2 ,
$$

so that the $$\psi^2$$ coefficient of the two terms together is $$\tfrac23(\sigma_1-1)=-\tfrac{2\alpha_L}{3(1+\alpha_L)}$$. Since $$N\sqrt h\,K=\partial_t\sqrt h-\partial_i(\sqrt hN^i)$$ identically, $$4\mathcal FK$$ contributes $$-4\dot{\mathcal F}\sqrt h$$, independent of the lapse, which is $$-18\dot{\mathcal F}a^3\zeta^2$$ at second order. The term $$-6\mathcal G^2N\sqrt h$$ gives $$-27\mathcal G^2a^3\zeta^2-18\mathcal G^2a^3\zeta\,\delta N$$.

The curvature $${}^{(3)}\!R$$ vanishes on the background, so the product $$N r_1(N)\sqrt h\,{}^{(3)}\!R$$ needs $$r_1$$ only to first order in $$\delta N$$. It contributes $$2ak^2\frac{1+\alpha_T}{1+\alpha_L}\zeta^2+4ak^2\frac{1+\alpha_H}{1+\alpha_L}\zeta\,\delta N$$, with $$r_1(1)$$ multiplying $$\zeta^2$$ and $$(r_1+\partial_Nr_1)(1)$$ multiplying $$\zeta\,\delta N$$. Since $$\mathfrak a_i\mathfrak a^i$$ is already second order, $$r_2$$ enters at $$N=1$$ and the term is $$ak^2\beta_3\delta N^2/(1+\alpha_L)$$. The dust action, with $$-g^{\mu\nu}\partial_\mu\phi\partial_\nu\phi=(\dot\phi-N^i\partial_i\phi)^2/N^2-h^{ij}\partial_i\phi\,\partial_j\phi$$, gives

$$
a^3\Big[\tfrac12\rho_m(\dot v-\delta N)^2+3\rho_m\zeta(\dot v-\delta N)+\delta\rho_m(\dot v-\delta N)\Big]-ak^2\rho_m\big(\psi v+\tfrac12v^2\big).
$$

Add these terms and integrate $$-36a^3H\zeta\dot\zeta$$ by parts to $$+18(a^3H)^{\boldsymbol\cdot}\,\zeta^2=18a^3(3H^2+\dot H)\zeta^2$$. With (3) in the form $$6\mathcal G^2=6H^2-\rho_m$$ and $$\dot H-\dot{\mathcal F}=-\rho_m/4$$, the coefficient of $$\zeta^2$$ is $$a^3[27H^2+18(\dot H-\dot{\mathcal F})-27\mathcal G^2]=a^3[27H^2-\tfrac92\rho_m-27H^2+\tfrac92\rho_m]=0$$, and the coefficient of $$\zeta\,\delta N$$ is $$a^3[18(H^2-\mathcal G^2)-3\rho_m]=0$$. What remains is (4).
</div>

Without dust and at $$\alpha_L=\alpha_T=0$$ the $$\psi$$ equation gives $$\delta N=\dot\zeta/H$$; substituting it and integrating the $$\zeta\dot\zeta$$ term by parts gives the kinetic and gradient coefficients of [[YYS]](#ref-yys), eq. (4.17), quoted in section 1, including the $$2\dot\alpha_H/H$$ term. With dust, (4) is the quadratic action behind the constraint equations of [[Yao]](#ref-yao), eqs. (3.8) and (3.9): the $$\psi$$ equation of (4) is the first of them, with $$M^2=2/(1+\alpha_L)$$ in that paper's notation, and the $$\delta N$$ equation is the second.

## 3. An exact reduction at $$\alpha_L=0$$

The dust equation and the momentum constraint remove all but one fast degree of freedom, and both remaining terms of its Lagrangian carry two spatial derivatives. The dust fixes the lapse: $$\delta\rho_m$$ appears only multiplying $$\dot v-\delta N$$, so $$\delta N=\dot v$$. At $$\alpha_L=0$$ the shift $$\psi$$ appears linearly in (4) and is a multiplier. We change variables to

$$
\mathcal R\equiv\zeta-Hv,
$$

the curvature perturbation on the slices of constant dust proper time, since $$a^2(t)e^{2\zeta}=a^2(t+v)e^{2\mathcal R}$$ to first order, and shift the multiplier to $$\tilde\psi\equiv\psi+(1+\alpha_H)v$$. The new variables remove the cross term between $$\zeta$$ and $$\delta N$$ at the price of a total time derivative. Define

$$
D\equiv(\alpha_H-\alpha_T)H+\dot\alpha_H ,
$$

so that $$B_m$$ of (1) can be written $$B_m=2(\alpha_H-\alpha_T)-2(1+\alpha_H)\dot{\mathcal F}/H^2-3\alpha_H\Omega_m+2\dot\alpha_H/H$$ using $$3(1+w)\Omega_{\rm de}=-2\dot{\mathcal F}/H^2$$ and $$\rho_m=6H^2\Omega_m$$.

<div class="lemma" markdown="1">
**Lemma 2** (exact reduction). Let $$\alpha_L=0$$ and $$\delta N=\dot v$$, $$\zeta=Hv+\mathcal R$$, $$\psi=\tilde\psi-(1+\alpha_H)v$$. Then, up to a total time derivative, (4) equals

$$
\begin{aligned}
\ell={}&a\,k^2\Big\{\beta_3\dot v^2-H^2B_m\,v^2+2(1+\alpha_T)\mathcal R^2-4D\,\mathcal R\,v-4\tilde\psi\big(\dot{\mathcal R}+\dot{\mathcal F}v\big)\Big\}\\
&+a^3\Big\{3\rho_m(Hv+\mathcal R)\,\dot v-6(\dot{\mathcal R}+\dot Hv)^2\Big\},
\end{aligned} \tag{5}
$$

and its Euler–Lagrange equations are equivalent to the closed linear system

$$
\dot{\mathcal R}=-\dot{\mathcal F}\,v,\qquad \frac{d}{dt}\big(a\tilde\psi\big)=-a(1+\alpha_T)\mathcal R+aD\,v, \tag{6}
$$

$$
\frac1a\frac{d}{dt}\big(a\beta_3\dot v\big)+H^2B_m\,v=-2D\,\mathcal R-2\dot{\mathcal F}\,\tilde\psi+\frac{9\Omega_m\dot{\mathcal F}}{\kappa^2}\,v. \tag{7}
$$

</div>

<div class="proof" markdown="1">
*Proof.* Solving the $$\delta\rho_m$$ equation for $$\delta N$$ and substituting is legitimate and removes the last two terms of (4). In the $$k^2$$ bracket, $$4(1+\alpha_H)\zeta\dot v=4(1+\alpha_H)Hv\dot v+4(1+\alpha_H)\mathcal R\dot v$$. With the measure $$a^3\cdot k^2/a^2=ak^2$$ and $$\dot a=aH$$, integration by parts gives

$$
4a(1+\alpha_H)Hv\dot v\simeq-2\big[aH^2(1+\alpha_H)+aH\dot\alpha_H+a(1+\alpha_H)\dot H\big]v^2,
$$

$$
4a(1+\alpha_H)\mathcal R\dot v\simeq-4a\big[H(1+\alpha_H)+\dot\alpha_H\big]\mathcal Rv-4a(1+\alpha_H)\dot{\mathcal R}\,v .
$$

The $$\psi$$ term is $$-4\psi(\dot\zeta-H\dot v+\rho_mv/4)=-4\psi(\dot{\mathcal R}+\dot{\mathcal F}v)$$ by (3). The shift $$\psi=\tilde\psi-(1+\alpha_H)v$$ turns it into $$-4\tilde\psi(\dot{\mathcal R}+\dot{\mathcal F}v)+4(1+\alpha_H)v(\dot{\mathcal R}+\dot{\mathcal F}v)$$. The $$\dot{\mathcal R}$$ part of the second term cancels the term $$-4a(1+\alpha_H)\dot{\mathcal R}\,v$$ of the integration by parts above, and $$+4(1+\alpha_H)\dot{\mathcal F}v^2$$ remains. Collecting the $$v^2$$ terms, including $$2(1+\alpha_T)H^2v^2$$ from $$\zeta^2$$ and $$-\rho_mv^2/2$$, and using $$\dot H=\dot{\mathcal F}-\rho_m/4$$,

$$
2(\alpha_T-\alpha_H)H^2-2H\dot\alpha_H+2(1+\alpha_H)\dot{\mathcal F}+\tfrac12\alpha_H\rho_m=-H^2B_m ,
$$

and the $$\mathcal Rv$$ terms, $$4(1+\alpha_T)H$$ from $$\zeta^2$$ and $$-4[H(1+\alpha_H)+\dot\alpha_H]$$ from the line above, add to $$-4D$$. The terms without $$k^2$$ are rewritten with $$\dot\zeta-H\dot v=\dot{\mathcal R}+\dot Hv$$. This is (5).

The Euler–Lagrange equations of (5) follow. The variation with respect to $$\tilde\psi$$ gives the first equation of (6). With it, $$\dot{\mathcal R}+\dot Hv=-\tfrac14\rho_mv$$, so $$12a^3(\dot{\mathcal R}+\dot Hv)=-3a^3\rho_mv$$, and the variation with respect to $$\mathcal R$$ reads

$$
\frac{d}{dt}\Big[4ak^2\tilde\psi-3a^3\rho_mv\Big]=-4ak^2(1+\alpha_T)\mathcal R+4ak^2D\,v-3a^3\rho_m\dot v .
$$

Since $$a^3\rho_m$$ is constant, $$d(3a^3\rho_mv)/dt=3a^3\rho_m\dot v$$ cancels the last term, which is the second equation of (6). The variation with respect to $$v$$ gives $$\frac{d}{dt}(2ak^2\beta_3\dot v)+2ak^2H^2B_mv=-4ak^2D\mathcal R-4ak^2\dot{\mathcal F}\tilde\psi+S$$ with $$S$$ the contribution of the $$a^3$$ terms,

$$
S=a^3\big[3\rho_mH\dot v-12\dot H(\dot{\mathcal R}+\dot Hv)\big]-\frac{d}{dt}\big[3a^3\rho_m(Hv+\mathcal R)\big]=-3a^3(\rho_m+4\dot H)(\dot{\mathcal R}+\dot Hv)=3a^3\dot{\mathcal F}\rho_m v ,
$$

using $$(a^3\rho_m)^{\cdot}=0$$, $$\rho_m+4\dot H=4\dot{\mathcal F}$$ and $$\dot{\mathcal R}+\dot Hv=-\rho_mv/4$$. Dividing by $$2ak^2$$ and using $$3a^2\rho_m/(2k^2)=9\Omega_m/\kappa^2$$ gives (7).
</div>

The cancellation between the $$\dot H$$ produced by integrating $$Hv\dot v$$ by parts and the $$\dot{\mathcal F}$$ produced by the momentum constraint is what turns the total $$\dot H$$ into $$\dot{\mathcal F}$$. It leaves $$\tfrac12\alpha_H\rho_m$$ behind, which is the origin of the term $$-3\alpha_H\Omega_m$$. At $$\alpha_H=\alpha_T=0$$ that remainder vanishes and the coefficient of $$v^2$$ in (5) is the quintessence-type gradient energy $$2\dot{\mathcal F}=-\tfrac12(\rho_{\rm de}+p_{\rm de})$$. At $$\Omega_m=0$$, $$B_m$$ is the vacuum gradient coefficient of section 1 with $$\dot{\mathcal F}$$ in place of $$\dot H$$; with dust the same combination appears as a squared frequency independent of $$k$$. The system (6)–(7) contains $$k$$ only through $$\kappa^{-2}$$ in the last term of (7), so it is valid for every $$\kappa$$ at $$\alpha_L=0$$, not only for $$\kappa\gg1$$.

## 4. The fast mode

In (6)–(7) the dynamics separates by time scale. The variable $$v$$ has inertia $$\beta_3$$ and stiffness $$H^2B_m$$, so it oscillates or grows at a rate of order $$H/\sqrt{\beta_3}$$. The variables $$\mathcal R$$ and $$\tilde\psi$$ are slow: by (6) their rates are $$\dot{\mathcal F}v$$ and $$Dv$$, of order $$H^2v$$ and $$Hv$$. Define

$$
B_{\rm eff}\equiv B_m-\frac{9\Omega_m\dot{\mathcal F}}{H^2\kappa^2}=B_m+\frac{27(1+w)\Omega_m\Omega_{\rm de}}{2\kappa^2}.
$$

<div class="proposition" markdown="1">
**Proposition 1** (fast mode; derivation at leading order in $$\sqrt{\beta_3}$$). Assume (H1)–(H4), and let $$[t_1,t_2]$$ be an interval on which $$B_{\rm eff}\le-b_0<0$$ with $$b_0$$ independent of $$\varepsilon$$. As $$\varepsilon\to0$$, the system (6)–(7) has a solution in which $$v$$ grows with

$$
\frac{d\ln\lvert v\rvert}{dt}=H\sqrt{\frac{-B_{\rm eff}}{\beta_3}}+O(H),\qquad \mathcal R,\ \tilde\psi=O(\varepsilon)\,v, \tag{8}
$$

and $$\zeta=Hv+\mathcal R$$ grows with it. Where $$B_{\rm eff}>0$$ the corresponding solutions oscillate at frequency $$H\sqrt{B_{\rm eff}/\beta_3}$$ without growth. For $$\kappa\gg1$$, $$B_{\rm eff}\to B_m$$, the quantity (1), and the growth rate is $$\Gamma=H\sqrt{-B_m/\beta_3}$$, independent of $$k$$.
</div>

The proof combines exact size estimates with the standard WKB theory of linear systems with a large parameter, which we use without reproving; the remainder in (8) is not bounded.

<div class="proof" markdown="1">
*Proof.* Suppose $$v$$ varies at a rate $$\Gamma\sim H/\varepsilon$$. Then (6) gives $$\mathcal R\simeq-\dot{\mathcal F}v/\Gamma=O(\varepsilon H)v$$ and $$\tilde\psi\simeq Dv/\Gamma=O(\varepsilon)v$$. The first two terms on the right of (7) are therefore of order $$\varepsilon H^2v$$, and the third is the explicit $$\kappa^{-2}$$ term already moved into $$B_{\rm eff}$$. Equation (7) becomes

$$
\beta_3\big(\ddot v+H\dot v+(\dot\beta_3/\beta_3)\dot v\big)+H^2B_{\rm eff}\,v=O(\varepsilon H^2)\,v .
$$

The friction and $$\dot\beta_3$$ terms are smaller than $$\ddot v$$ by $$H/\Gamma=O(\varepsilon)$$. The remaining oscillator equation has local frequency squared $$H^2B_{\rm eff}/\beta_3$$ and varies on the Hubble time, so its WKB solutions have $$d\ln\lvert v\rvert/dt=H\sqrt{-B_{\rm eff}/\beta_3}+O(H)$$, the correction collecting the friction, the slow variation of the amplitude factor and the $$O(\varepsilon)$$ shift of $$B_{\rm eff}$$. The last statement follows from $$B_{\rm eff}-B_m=O(\kappa^{-2})$$.
</div>

In the language of the effective field theory of dark energy, $$v$$ is minus the Goldstone time shift of the dark energy seen from the dust's rest frame, and the $$v$$ part of (5) is $$a\,k^2[\beta_3\dot v^2-B_mH^2v^2]$$. The kinetic term comes from $$\mathfrak a_i\mathfrak a^i$$ and carries two spatial gradients, as does the potential term, which is why $$k^2$$ cancels in the rate. The instability is of the type that De Felice et al. [[DFP]](#ref-dfp) call tachyonic: a negative squared frequency $$B_mH^2/\beta_3$$ that is independent of $$k$$ and much larger than $$H^2$$ in magnitude, as opposed to a gradient instability, whose rate grows with $$k$$. The term $$2\dot\alpha_H/H$$ enters through an integration by parts and would be lost in a frozen-coefficient treatment of the action.

The linear equations were integrated without approximating the time dependence, for a single Fourier mode, on $$w_0w_a$$ backgrounds with $$H_0=1$$. The amplitude is $$\mathcal X=(\zeta^2+H^2v^2)^{1/2}$$, its local growth rate is $$d\ln\mathcal X/d\ln a$$, and the growth between $$z_a$$ and $$z_b$$ is the logarithm of the maximum of $$\mathcal X$$ over $$z_b\le z\le z_a$$ divided by $$\mathcal X(z_a)$$. Unless stated otherwise the initial data are imposed at $$z=4$$ and are $$Hv=\zeta=10^{-3}$$ ($$\mathcal R=0$$), $$\dot v=0$$, $$\dot\zeta=-\rho_mv/4$$, which satisfy the constraint, and $$\tilde\psi=0$$ where $$\tilde\psi$$ is evolved. Comoving wavenumbers are in units of $$H_0$$; $$k=10H_0$$ is $$3.3\times10^{-3}\,h\,{\rm Mpc}^{-1}$$ for $$H_0=100\,h\,{\rm km\,s^{-1}Mpc^{-1}}$$. The first test is the exact $$\alpha_L=0$$ system (6)–(7) on the DESI+CMB+DESY5 background of section 6 at $$z=1$$, with $$\beta_3=10^{-4}$$, $$\alpha_H=\alpha_T=0$$ and initial data at $$z=2.33$$, and isolates the $$k$$-dependence. The rate of $$\lvert v\rvert$$, to which (8) refers, and that of $$\mathcal X$$ are both listed.

| $$k/H_0$$ | $$\kappa$$ | rate of $$\lvert v\rvert$$, $$d\ln\lvert v\rvert/d\ln a$$ | rate of $$\mathcal X$$ | $$\sqrt{-B_m/\beta_3}$$ | $$\sqrt{-B_{\rm eff}/\beta_3}$$ |
|---|---|---|---|---|---|
| 3 | 3.3 | 39.19 | 38.07 | 34.14 | 39.21 |
| 10 | 11.1 | 34.55 | 33.43 | 34.14 | 34.63 |
| 30 | 33.3 | 34.11 | 32.99 | 34.14 | 34.19 |
| 100 | 111 | 34.06 | 32.94 | 34.14 | 34.14 |
| 1000 | 1110 | 34.05 | 32.93 | 34.14 | 34.14 |

For $$\kappa\gtrsim30$$ both rates are independent of $$k$$ to 0.2%. The rate of $$\lvert v\rvert$$ is within 0.3% of $$\sqrt{-B_{\rm eff}/\beta_3}$$ in every row, so the explicit $$\kappa^{-2}$$ term of $$B_{\rm eff}$$ accounts for the departure from $$\sqrt{-B_m/\beta_3}$$ at $$\kappa\lesssim11$$. The rate of $$\mathcal X$$ is lower by $$1.12=-d\ln H/d\ln a$$ at $$z=1$$, because $$\mathcal X\simeq\sqrt2H\lvert v\rvert$$ up to $$O(\varepsilon)$$, and at $$\kappa\to\infty$$ it is 3.5% below $$\sqrt{-B_m/\beta_3}$$. Both offsets are of the order $$H$$ allowed by (8).

The second test uses the small positive value $$\alpha_L=10^{-5}$$. For $$\alpha_L>0$$ the shift $$\psi$$ is no longer a multiplier, so it can be eliminated and the equations of the two-variable Lagrangian in $$(\mathcal R,v)$$ integrated directly. The band edge (11) of section 5 lies beyond $$\kappa\approx80$$ in every row below, and the correction (10) at the measured $$\kappa\approx11$$ is below 2% of $$\lvert B_m\rvert$$. Here $$k=10H_0$$, $$\beta_3=10^{-4}$$, and the rate is the slope of $$\ln\mathcal X$$ over $$\pm0.02$$ in $$\ln a$$. The $$\Lambda$$CDM background has $$\Omega_{m,0}=0.3027$$, the DESI+CMB value ([[DESI]](#ref-desi), eq. (21)).

| background | $$\alpha_H$$ | $$z$$ | $$\kappa$$ | $$B_m$$ | measured $$\Gamma/H$$ | $$\sqrt{-B_m/\beta_3}$$ |
|---|---|---|---|---|---|---|
| DESY5 | 0 | 1.0 | 11.1 | $$-0.117$$ | 33.3 | 34.1 |
| DESY5 | 0 | 0.8 | 11.2 | $$-0.113$$ | 33.2 | 33.7 |
| DESY5 | 0 | 0.5 | 11.1 | $$-0.048$$ | 23.4 | 21.9 |
| DESY5 | $$+0.3$$ | 1.2 | 10.9 | $$-0.294$$ | 53.5 | 54.2 |
| DESY5 | $$-0.3$$ | 0.6 | 11.1 | $$-0.087$$ | 27.5 | 29.5 |
| $$\Lambda$$CDM | $$+0.3$$ | 1.5 | 10.7 | $$-0.184$$ | 42.0 | 42.9 |
| $$\Lambda$$CDM | $$+0.3$$ | 1.0 | 11.3 | $$-0.099$$ | 31.2 | 31.4 |
| $$\Lambda$$CDM | $$-0.3$$ | 0.2 | 10.9 | $$-0.214$$ | 44.6 | 46.3 |
| $$\Lambda$$CDM | $$-0.3$$ | 0.4 | 11.3 | $$-0.111$$ | 30.5 | 33.3 |

The predicted rate differs from the measured one by at most 9.2% of the measured rate (30.5 against 33.3). The four rows with differences above 3% are those in which $$B_m$$ changes quickly in time: DESY5 at $$z=0.5$$ (6.4%) and at $$z=0.6$$ with $$\alpha_H=-0.3$$ (7.3%, $$B_m$$ vanishing near $$z=0.9$$), and the $$\Lambda$$CDM rows with $$\alpha_H=-0.3$$ (3.8% and 9.2%, $$B_m$$ vanishing at $$z=0.66$$). There the WKB assumption of slow variation fails. The other five rows agree within 2.5%. Friction lowers the rate by $$H/2$$ (1.5%), the $$O(\varepsilon)$$ terms of (8) are of the same order, and the explicit $$\kappa^{-2}$$ term raises it. On $$\Lambda$$CDM with $$\alpha_H=0$$, where $$B_m=0$$, no mode grows.

The local rates do not depend on the initial data. Integrated growth does, because the growing mode needs a fraction of an e-fold to dominate: for the first fit of section 6, imposing the initial data at $$z=2.33$$ rather than $$z=4$$ gives 19.6 instead of 23.0 e-folds between $$z=2.33$$ and the crossing.

As a worked value, take $$\Lambda$$CDM with $$\alpha_H=0.3$$ and $$\alpha_T=0$$ at $$z=1.5$$: $$\Omega_m=0.3027\cdot15.625/(0.3027\cdot15.625+0.6973)=0.8715$$, so $$B_m=0.3(2-3\cdot0.8715)=-0.184$$ and $$\Gamma/H=\sqrt{0.184/10^{-4}}=42.9$$, against 42.0 measured. In the vacuum limit ($$\Omega_m\to0$$), (1) reduces to the coefficient of [[YYS]](#ref-yys), eq. (4.17), at $$\alpha_T=0$$; for $$\alpha_H=\alpha_T=0$$ and $$w=-1$$, $$B_m=0$$, as it must for general relativity with dust and a cosmological constant.

## 5. Finite $$\alpha_L$$: the band

For $$\alpha_L>0$$ the shift enters (4) quadratically. The exact difference between (4) and its $$\alpha_L=0$$ counterpart, in the variables of Lemma 2, is

$$
\ell(\alpha_L)-\ell(0)=\frac{\alpha_L}{1+\alpha_L}\Big[-\frac23\frac{k^4}{a}\big(\tilde\psi-(1+\alpha_H)v\big)^2-a k^2\big(\beta_3\dot v^2+4(1+\alpha_H)\zeta\dot v+2(1+\alpha_T)\zeta^2\big)\Big], \tag{9}
$$

which follows by substituting the variables of Lemma 2 into (4). The second part of the bracket changes the coefficients in (5) by relative amounts of order $$\alpha_L$$. The first part, with $$\tilde\psi=O(\varepsilon)v$$ as in Proposition 1, contributes $$-\tfrac23\alpha_L(1+\alpha_H)^2\kappa^2\cdot ak^2H^2v^2$$ at leading order, which is enhanced by $$\kappa^2$$. Its effect on (7), a derivation at leading order in $$\varepsilon$$ and $$\alpha_L$$ for $$\alpha_L\ll1$$ and $$\alpha_L\kappa^2=O(1)$$, is to replace

$$
B_{\rm eff}\to B_{\rm eff}+\tfrac23\alpha_L(1+\alpha_H)^2\kappa^2 \tag{10}
$$

The added term equals $$\beta_3c_s^2\kappa^2$$, where $$c_s^2=\tfrac23\alpha_L(1+\alpha_H)^2/\beta_3$$ is the leading term of the high-$$k$$ sound speed $$\alpha_L[2(1+\alpha_H)^2-\beta_3(1+\alpha_T)]/[3\beta_3(1+\alpha_L)]$$ of the scalar in vacuum ([[YYS]](#ref-yys), large-$$k$$ limit of the ratio of the coefficients of $$\zeta^2$$ and $$\dot\zeta^2$$), so the squared frequency interpolates as $$B_mH^2/\beta_3+c_s^2k^2/a^2$$. The growth therefore occupies a band

$$
aH\ll k<k_{\max},\qquad k_{\max}=\frac{aH}{1+\alpha_H}\sqrt{\frac{3\lvert B_m\rvert}{2\alpha_L}} , \tag{11}
$$

where $$\kappa_{\max}=k_{\max}/(aH)$$. The band extends to all sub-horizon scales as $$\alpha_L\to0$$ and closes (falls to $$k\lesssim aH$$) for $$\alpha_L\gtrsim 3\lvert B_m\rvert/[2(1+\alpha_H)^2]$$. On the DESI backgrounds at $$\alpha_H=0$$ the maximum of $$\lvert B_m\rvert$$ is 0.096 to 0.192, so the band closes for $$\alpha_L\gtrsim0.14$$ to $$0.29$$ (0.18 for DESI+CMB+DESY5).

As a test, take $$\Lambda$$CDM with $$\alpha_H=0.3$$, $$\alpha_L=10^{-3}$$, $$z=1.5$$ ($$B_m=-0.184$$, $$\kappa_{\max}=12.8$$) and initial data at $$z=3$$.

| $$\kappa$$ | measured $$\Gamma/H$$ | from (10) | leading order |
|---|---|---|---|
| 5.4 | 38.0 | 39.0 | 42.9 |
| 8.6 | 31.1 | 31.8 | 42.9 |
| 10.7 | 23.2 | 23.4 | 42.9 |

Beyond the edge the integrated growth falls. For the same background and parameters, with initial data at $$z=4$$, the growth between $$z=3$$ and $$0$$ is 30.6, 18.5, 4.1, 1.2 and 0.0 e-folds at $$k=5,10,15,20,30\,H_0$$. The local rate is not quoted there, since the amplitude oscillates and its logarithmic slope is not defined. The derivation of (10) assumes $$\kappa\gg1$$ and does not cover scales near the horizon.

## 6. Constant parameters

With $$\dot\alpha_H=0$$ the following statements follow from (1) by inspection. Each assumes (H2)–(H4) and $$\alpha_L$$ small enough that the band (11) exists, and holds at leading order in $$\sqrt{\beta_3}$$, for modes $$aH\ll k<k_{\max}$$.

<div class="corollary" markdown="1">
**Corollary 1** (phantom phases, leading order). If $$\alpha_H=\alpha_T=0$$, then $$B_m=3(1+w)\Omega_{\rm de}$$. There is no fast growth exactly when $$w\ge-1$$, with $$w=-1$$ marginal at this order. In every phantom phase ($$w<-1$$) with $$\beta_3\ll3\lvert1+w\rvert\Omega_{\rm de}$$ the mode grows at the rate $$H\sqrt{3\lvert1+w\rvert\Omega_{\rm de}/\beta_3}$$.
</div>

<div class="corollary" markdown="1">
**Corollary 2** (constant $$\alpha_H$$ on $$\Lambda$$CDM, leading order). If $$w=-1$$ and $$\alpha_T=0$$, then $$B_m=\alpha_H(2-3\Omega_m)$$. A constant $$\alpha_H>0$$ is unstable while $$\Omega_m>\tfrac23$$, which includes the whole matter era, and a constant $$\alpha_H<0$$ is unstable while $$\Omega_m<\tfrac23$$, in both cases where $$\beta_3\ll\lvert B_m\rvert$$. For $$\Omega_{m,0}=0.3027$$ the transition is at $$z=0.664$$.
</div>

The statement for $$\alpha_H>0$$ does not need $$w=-1$$: whenever $$\Omega_m\to1$$ and $$(1+w)\Omega_{\rm de}\to0$$, (1) gives $$B_m\to-\alpha_H$$ at $$\alpha_T=0$$. The vacuum condition of [[YYS]](#ref-yys), eq. (4.15), allows constant $$\alpha_H\ge0$$ on a $$\Lambda$$CDM background; with dust, a matter era makes every $$\alpha_H>0$$ with $$\beta_3\ll\alpha_H$$ unstable. In the exact evolution for $$\Lambda$$CDM with $$\Omega_{m,0}=0.3027$$, $$k=20H_0$$, $$\alpha_L=10^{-5}$$, $$\beta_3=10^{-4}$$ and initial data at $$z=4$$, the growth between $$z=3$$ and $$0$$ is as follows, with $$\int\Gamma\,dt$$ for comparison:

| $$\alpha_H$$ | $$+0.3$$ | $$+0.1$$ | $$0$$ | $$-0.1$$ | $$-0.3$$ |
|---|---|---|---|---|---|
| exact growth (e-folds) | 34.5 | 19.4 | 0.0 | 9.4 | 17.9 |
| $$\int\Gamma\,dt$$ (e-folds) | 35.1 | 20.3 | 0.0 | 11.2 | 19.5 |

For $$\alpha_H<0$$ the unstable epoch $$z<0.66$$ starts late, and the exact growth depends more on the initial data: starting from $$\zeta=10^{-3}$$ with $$v=\dot\zeta=\dot v=0$$ instead gives 12.7 and 21.3 e-folds for $$\alpha_H=-0.1,-0.3$$.

For a general background write $$B_m=\alpha_H\,C(z)+3(1+w)\Omega_{\rm de}-2\alpha_T$$ at constant $$\alpha_H$$, with $$C\equiv2-3\Omega_m+3(1+w)\Omega_{\rm de}$$.

<div class="corollary" markdown="1">
**Corollary 3** (no constant $$\alpha_H$$, leading order). Let $$\alpha_T=0$$. If $$C(z_*)=0$$ at a redshift $$z_*$$ where $$w(z_*)<-1$$, then $$B_m(z_*)=3(1+w)\Omega_{\rm de}<0$$ for every constant $$\alpha_H$$. The local growth rate is then positive at $$z_*$$ for every constant $$\alpha_H$$, and the mode grows there if $$\beta_3\ll\lvert B_m(z_*)\rvert$$.
</div>

We evaluate this on the DESI DR2 $$w_0w_a$$ fits that combine BAO, CMB and supernovae, with the posterior means of $$(\Omega_m,w_0,w_a)$$ from Table V of [[DESI]](#ref-desi). In each fit $$w$$ crosses $$-1$$ at $$z_c$$ and is phantom above it, and $$C$$ has one zero $$z_*$$ for $$z\in(0.05,3)$$, which lies on the phantom side.

| fit | $$\Omega_m,\ w_0,\ w_a$$ | $$z_c$$ | $$\min B_m$$ at $$\alpha_H=0$$ (at $$z$$) | $$z_*$$ | $$B_m(z_*)$$ |
|---|---|---|---|---|---|
| DESI+CMB+DESY5 | 0.3191, $$-0.752$$, $$-0.86$$ | 0.405 | $$-0.117$$ (0.93) | 0.611 | $$-0.085$$ |
| DESI+CMB+Union3 | 0.3275, $$-0.667$$, $$-1.09$$ | 0.440 | $$-0.137$$ (0.96) | 0.615 | $$-0.089$$ |
| DESI+CMB+Pantheon+ | 0.3114, $$-0.838$$, $$-0.62$$ | 0.354 | $$-0.096$$ (0.89) | 0.607 | $$-0.078$$ |
| DESI+CMB | 0.353, $$-0.42$$, $$-1.75$$ | 0.496 | $$-0.192$$ (0.99) | 0.627 | $$-0.104$$ |

Integrated over the phantom epoch at $$\beta_3=10^{-4}$$, the growth in e-folds is as follows. The leading-order columns are $$\int\Gamma\,dt$$ with $$\alpha_H=\alpha_T=0$$. The exact column is $$\ln[\mathcal X(z_c)/\mathcal X(2.33)]$$ in the exact evolution at $$k=20H_0$$, $$\alpha_L=10^{-5}$$ with initial data at $$z=4$$. The last column is the minimum of the leading-order integral over constant $$\alpha_H\in[-6,6]$$.

| fit | leading order, $$2.33\to z_c$$ | exact, $$2.33\to z_c$$ | leading order, $$3\to z_c$$ | minimum over constant $$\alpha_H$$ | at $$\alpha_H$$ |
|---|---|---|---|---|---|
| DESI+CMB+DESY5 | 24.2 | 23.0 | 27.3 | 6.8 | $$-0.58$$ |
| DESI+CMB+Union3 | 25.3 | 24.2 | 28.6 | 5.9 | $$-0.72$$ |
| DESI+CMB+Pantheon+ | 22.9 | 21.6 | 25.8 | 8.2 | $$-0.43$$ |
| DESI+CMB | 28.2 | 27.3 | 31.7 | 4.7 | $$-1.02$$ |

The redshift $$z=2.33$$ is the effective redshift of the highest-redshift DESI BAO measurement; extending the CPL form to $$z=20$$ raises the first entry to 34 e-folds, but that range lies beyond the data. The growth scales as $$\beta_3^{-1/2}$$ in the exponent at the level of $$\int\Gamma dt$$, so for the first fit it is $$e^{24}$$ at $$\beta_3=10^{-4}$$ and $$e^{2.4}$$ at $$\beta_3=10^{-2}$$, and the exact evolution at $$\beta_3=10^{-2}$$ gives less (section 8). These e-fold counts are growth factors of linear perturbation theory, up to $$e^{28}\sim10^{12}$$, and not predictions: linear theory fails at a time set by the initial amplitude of $$v$$, and what happens after that is not addressed.

The posterior means are not best-fit points. The DESY5 supernova sample has since been recalibrated; with the recalibrated sample, the CMB and DESI DR2 data the reported CPL values are $$w_0=-0.803$$, $$w_a=-0.72$$ [[Popovic]](#ref-popovic), for which the crossing is at $$z_c=0.377$$. Corollary 3 also applies to these values: for $$w_0=-0.803$$, $$w_a=-0.72$$ and $$\Omega_{m,0}$$ between 0.30 and 0.33, $$C$$ has one zero at $$z_*=0.58$$ to $$0.64$$, on the phantom side of $$z_c$$, with $$B_m(z_*)=-0.07$$ to $$-0.09$$.

In all four fits $$z_*$$ lies on the phantom side, so Corollary 3 applies and no constant $$\alpha_H$$ makes $$B_m$$ non-negative throughout. A constant $$\alpha_T$$ of at most $$-0.039$$ to $$-0.052$$ (depending on the fit) would make $$B_m(z_*)\ge0$$ (it enters as $$-2\alpha_T$$), but GW170817 [[GW170817]](#ref-gw) bounds a constant $$\lvert\alpha_T\rvert$$ far below that.

## 7. Comparison with the stability conditions of the matter-coupled model

The conditions (3.24) to (3.26) and (3.39) that [[Yao]](#ref-yao) derives for the matter-coupled model hold at parameters where the exact evolution grows. For $$\alpha_L>0$$ they read as follows, with $$\alpha_A$$ the coefficient written $$\beta_3$$ here. The detuning must satisfy $$\alpha_L>-1$$ (eq. (3.24)), and $$\alpha_A>0$$ (3.25). The no-ghost conditions (3.26) hold at all wavenumbers when $$\chi_m<(\sqrt{1+\alpha_L}+\sqrt{\alpha_L})^2$$, with $$\chi_m=\alpha_A(1+w_m)\bar\rho_m/(4M^2H^2(1+\alpha_L))$$ and $$M^2=2/(1+\alpha_L)$$, which for dust in our units is $$\chi_m=\tfrac34\alpha_A\Omega_m$$. The high-$$k$$ gradient condition (3.39) is $$(1+\alpha_H)^2>\tfrac12\alpha_A(1+\alpha_T)>0$$.

Take the DESI+CMB+DESY5 background with $$\beta_3=\alpha_A=10^{-4}$$, $$\alpha_H=\alpha_T=0$$ and $$\alpha_L=10^{-4}$$. Then (3.24) and (3.25) hold, $$\chi_m<7.4\times10^{-5}$$ for $$z\le3$$ so that the no-ghost bound is satisfied by a wide margin, and (3.39) reads $$1>5\times10^{-5}$$. The scalar has the real sound speed $$c_s^2=\tfrac23$$. In the exact linear evolution of section 4 for these parameters, the amplitude at $$k=10H_0$$ grows by 22.5 e-folds between $$z=2.33$$ and the crossing (19.5 with the initial data imposed at $$z=2.33$$), against 24.2 from (1) at leading order. At $$\alpha_L=10^{-3}$$ it grows by 21.5 e-folds over the same interval at $$k=5H_0$$, and by 28.1 e-folds between $$z=3$$ and $$0$$ at $$k=3H_0$$ (initial data at $$z=3$$). For $$\alpha_L=10^{-4}$$ the band (11) extends to $$\kappa_{\max}\approx27$$ to $$42$$ for $$0.5\le z\le1$$, and the evolved modes have $$\kappa\approx10$$.

The gradient condition (3.39) is the $$k\to\infty$$ limit of the WKB condition (3.43), and the high-$$k$$ expansion of [[Yao]](#ref-yao), eq. (3.34), is an expansion in inverse powers of $$k$$ at fixed $$\alpha_L\alpha_A\neq0$$ whose leading term is the sound speed $$c_s^2k^2/a^2$$ that appears in (10). The unstable band lies below $$k_{\max}$$, where the $$k$$-independent term $$B_mH^2/\beta_3$$ is not small against it: the fast mode has $$\omega^2=c_s^2k^2/a^2+B_mH^2/\beta_3$$, which is negative for $$k<k_{\max}$$, with $$\lvert\omega\rvert\simeq30H$$ well inside the WKB regime $$\lvert\omega\rvert\gg H$$. The finite-$$k$$ condition (3.43) is therefore the one that applies in the band. It has not been evaluated from the closed-form coefficients of that paper; the dispersion relation above is the result of Proposition 1 and (10) in the variables of section 3, and it has $$\omega^2<0$$ throughout the band.

## 8. Parameter ranges outside the result

Corollaries 1 to 3 assume constant parameters, $$\beta_3\ll\lvert B_m\rvert$$, an existing band, and dust as the only matter. Outside those assumptions the instability can be weaker or absent. The conditions below refer to the DESI+CMB+DESY5 background.

The vacuum conditions of [[YYS]](#ref-yys) are sufficient for stability without matter. At $$\alpha_T=0$$ they are, in eq. (4.15), $$\alpha_L\ge0$$, $$2(1+\alpha_H)^2\ge\beta_3>0$$ and $$\alpha_H\ge\dot H/(H^2-\dot H)$$, the last being the positivity of the gradient coefficient $$2\alpha_H-2(1+\alpha_H)\dot H/H^2$$ of section 1 at constant $$\alpha_H$$, and in eq. (4.16) $$\dot\alpha_L\le0$$, $$\dot\beta_3\le0$$, $$\dot\alpha_H\ge0$$. Take $$\dot H$$ to be the total derivative of the background with dust, as in section 1. On the four DESI backgrounds $$\dot H/H^2\le-0.63$$ for $$0\le z\le4$$, and $$q/(1-q)$$ increases with $$q$$, so the third condition holds for every $$\alpha_H\ge-0.63/1.63=-0.386$$. All of (4.15) therefore holds at $$\alpha_H=\alpha_T=0$$, where $$B_m<0$$ nevertheless. With dust the condition for stability of the fast mode is $$B_m\ge0$$ (Lemma 2), and (4.15)–(4.16) do not imply it. The routes below are therefore stated in terms of $$B_m$$, and the conditions of section 7 and of (4.15)–(4.16) are noted where they bear on a route.

A running $$\alpha_H$$ can make $$B_m$$ positive throughout through the term $$2\dot\alpha_H/H$$ in (1). For $$\alpha_H=s\,(\ln a+0.5)$$, which runs from $$-0.89\,s$$ at $$z=3$$ to $$0.5\,s$$ today, the exact evolution ($$k=20H_0$$, $$\alpha_L=10^{-5}$$, initial data at $$z=4$$; growth between $$z=3$$ and $$0$$) gives

| $$s$$ | min $$B_m$$ | exact growth (e-folds) | $$\int\Gamma dt$$ with (1) | $$\int\Gamma dt$$ without $$2\dot\alpha_H/H$$ |
|---|---|---|---|---|
| 0 | $$-0.117$$ | 25.2 | 27.3 | 27.3 |
| 0.03 | $$-0.056$$ | 10.4 | 10.2 | 24.2 |
| 0.06 | $$+0.006$$ | 0.7 | 0.0 | 21.4 |
| 0.10 | $$+0.088$$ | 0.3 | 0.0 | 19.0 |

The last two columns show that the $$2\dot\alpha_H/H$$ term accounts for the reduction. A rate $$d\alpha_H/d\ln a\gtrsim0.06$$ suffices on this background and in this linear parametrisation. It satisfies $$\dot\alpha_H\ge0$$ of (4.16), and $$\alpha_H\ge-0.089$$ for $$s\le0.1$$, so the third condition of (4.15) holds by the bound $$-0.386$$ above. The route is closed if $$\alpha_H$$ must vanish: the decay of gravitational waves into dark-energy fluctuations forces $$\alpha_H=0$$ in the effective field theory of dark energy [[CLTV]](#ref-cltv). Whether that argument carries over to Freezing Gravity, whose scalar has the additional operator $$\mathfrak a_i\mathfrak a^i$$, is open (section 9). With $$\alpha_H\equiv0$$, Corollary 1 leaves the other routes.

A negative $$\alpha_T$$ at intermediate redshift enters (1) as $$-2\alpha_T$$. A constant $$\alpha_T=-0.03$$ ($$-0.07$$) reduces the exact growth from 25.2 to 13.5 (1.7) e-folds, against $$\int\Gamma dt=11.5$$ (0.0). GW170817 constrains $$\alpha_T$$ at $$z\approx0$$ only, so an $$\alpha_T(z)$$ that is about $$-0.06$$ near $$z\sim1$$ and vanishes today is not excluded by it. The conditions of section 7 hold for $$\alpha_T\ge-0.06$$ at $$\beta_3=10^{-4}$$; (4.15)–(4.16) are stated at $$\alpha_T=0$$.

A larger $$\beta_3$$ leaves Proposition 1, which requires $$\beta_3\ll\lvert B_m\rvert$$. The exact growth between $$z=3$$ and $$0$$ ($$\alpha_H=\alpha_T=0$$, $$\alpha_L=10^{-5}$$, initial data at $$z=4$$) compared with $$\int\Gamma dt$$ is

| $$\beta_3$$ | exact, $$k=5H_0$$ | exact, $$k=20H_0$$ | $$\int\Gamma dt$$ |
|---|---|---|---|
| $$10^{-3}$$ | 7.3 | 6.0 | 8.6 |
| $$3.3\times10^{-3}$$ | 2.9 | 2.1 | 4.7 |
| $$10^{-2}$$ | 0.64 | 0.22 | 2.7 |
| $$10^{-1}$$ | 0.0 | 0.0 | 0.86 |

At $$\beta_3=10^{-3}$$, where $$\varepsilon/\sqrt{\lvert B_m\rvert}\lesssim0.1$$, the leading-order integral exceeds the exact growth by 19% at $$k=5H_0$$ and by 43% at $$k=20H_0$$ (by 5% at $$\beta_3=10^{-4}$$, section 6). From $$\beta_3=10^{-2}$$ it exceeds the exact growth by a factor of 4 to 12. At $$k=100H_0$$, which lies near or beyond the band edge at $$\alpha_L=10^{-5}$$ ($$\kappa_{\max}\approx85$$ to $$130$$ for $$0.5\le z\le1$$), no growth is found for any of these $$\beta_3$$. For $$\beta_3\gtrsim\lvert B_m\rvert_{\max}\approx0.1$$ to $$0.2$$ the rate drops to order $$H$$ and the proposition no longer applies. The vacuum condition $$\beta_3\le2(1+\alpha_H)^2$$ of [[YYS]](#ref-yys), eq. (4.15), and the condition (3.39) are still satisfied there.

A large $$\alpha_L$$ closes the band when $$\alpha_L\gtrsim3\lvert B_m\rvert/[2(1+\alpha_H)^2]$$, which is about 0.18 at $$\alpha_H=0$$ on this background, because $$k_{\max}$$ in (11) then falls below $$aH$$. Near-horizon modes are outside Proposition 1, and in the exact evolution they can still grow. On the DESY5 background with $$\alpha_H=\alpha_T=0$$, $$\beta_3=10^{-4}$$ and initial data at $$z=3$$, the growth between $$z=3$$ and $$0$$ in e-folds is

| $$\alpha_L$$ | $$k=H_0$$ | $$k=3H_0$$ | $$k=10H_0$$ |
|---|---|---|---|
| $$10^{-3}$$ | 55.4 | 28.1 | 3.6 |
| $$10^{-2}$$ | 52.5 | 9.3 | 0.0 |
| $$0.05$$ | 35.3 | 0.0 | 0.0 |

Whether a large $$\alpha_L$$ removes the instability therefore depends on scales $$k\lesssim aH$$ that this page does not resolve. A non-minimal coupling of dust to the scalar, a time-dependent $$\alpha_L$$, radiation with $$\alpha_H\neq0$$ and the branch $$-1<\alpha_L<0$$ are not treated.

## 9. Modes near the horizon, the remainder and the running coupling

The remainder in (8) and the correction to the band edge (10) are not bounded, so the leading-order rates are derivations whose accuracy is known only from the integrations of sections 4 and 5. Scales $$k\lesssim aH$$ are not resolved either. The exact evolution shows growth there even when the band is closed, and the kinetic matrix of the matter-coupled model degenerates as $$k\to0$$ ([[Yao]](#ref-yao), eq. (3.33)).

The running-$$\alpha_H$$ route of section 8 is available only if the argument that gravitational-wave decay forces $$\alpha_H=0$$ in the effective field theory of dark energy [[CLTV]](#ref-cltv) does not carry over to Freezing Gravity. Whether it does has not been examined.

A Freezing Gravity realisation of the DESI phantom crossing therefore needs one of the conditions of section 8. Which of them, if any, is compatible with large-scale-structure and lensing data has not been examined.

## References

1. <a id="ref-yys"></a>Yao, Ye, Silvestri, "A General Model for Dark Energy Crossing the Phantom Divide," JCAP 10 (2025) 078, [arXiv:2508.01378](https://arxiv.org/abs/2508.01378).
2. <a id="ref-yao"></a>Yao, "Effective Field Theory for Freezing Gravity with Minimally Coupled Matter," [arXiv:2608.20085](https://arxiv.org/abs/2608.20085).
3. <a id="ref-bs"></a>Bellini, Sawicki, "Maximal freedom at minimum cost: linear large-scale structure in general modifications of gravity," JCAP 07 (2014) 050, [arXiv:1404.3713](https://arxiv.org/abs/1404.3713).
4. <a id="ref-cdnv"></a>Creminelli, D'Amico, Noreña, Vernizzi, "The Effective Theory of Quintessence: the w<-1 Side Unveiled," JCAP 02 (2009) 018, [arXiv:0811.0827](https://arxiv.org/abs/0811.0827).
5. <a id="ref-dfp"></a>De Felice, Frusciante, Papadomanolakis, "On the stability conditions for theories of modified gravity in presence of matter fields," JCAP 03 (2017) 027, [arXiv:1609.03599](https://arxiv.org/abs/1609.03599).
6. <a id="ref-glv"></a>Gleyzes, Langlois, Vernizzi, "A unifying description of dark energy," [arXiv:1411.3712](https://arxiv.org/abs/1411.3712).
7. <a id="ref-lmnv"></a>Langlois, Mancarella, Noui, Vernizzi, "Effective Description of Higher-Order Scalar-Tensor Theories," [arXiv:1703.03797](https://arxiv.org/abs/1703.03797).
8. <a id="ref-desi"></a>DESI Collaboration, "DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints," Phys. Rev. D 112, 083515 (2025), [arXiv:2503.14738](https://arxiv.org/abs/2503.14738).
9. <a id="ref-gw"></a>LIGO Scientific Collaboration, Virgo Collaboration, Fermi Gamma-ray Burst Monitor, INTEGRAL, "Gravitational Waves and Gamma-rays from a Binary Neutron Star Merger: GW170817 and GRB 170817A," Astrophys. J. Lett. 848, L13 (2017), [arXiv:1710.05834](https://arxiv.org/abs/1710.05834).
10. <a id="ref-cltv"></a>Creminelli, Lewandowski, Tambalo, Vernizzi, "Gravitational Wave Decay into Dark Energy," [arXiv:1809.03484](https://arxiv.org/abs/1809.03484).
11. <a id="ref-popovic"></a>Popovic et al. (DES Collaboration), "The Dark Energy Survey Supernova Program: A Reanalysis Of Cosmology Results And Evidence For Evolving Dark Energy With An Updated Type Ia Supernova Calibration," Mon. Not. R. Astron. Soc. (2026), [arXiv:2511.07517](https://arxiv.org/abs/2511.07517).
