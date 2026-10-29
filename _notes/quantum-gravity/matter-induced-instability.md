---
title: A matter-induced instability in Freezing Gravity
date: 2026-10-29
category: dark-sector
order: 6
description: Adding pressureless matter to the Freezing Gravity dark-energy model produces a sub-horizon instability whose growth rate does not depend on wavenumber. It affects every phantom phase and every constant non-zero beyond-Horndeski coupling, and no constant coupling stabilises the DESI DR2 fits, unless one of the escape routes of section 7 applies.
status: The quadratic action and its exact reduction are proved by direct expansion; the growth rate is derived at leading order in the square root of the lapse-gradient coefficient and compared with exact numerical integration of the linear equations. DESI numbers use CPL posterior means.
excludes: Freezing Gravity with minimally coupled dust, constant effective-field-theory parameters and a small lapse-gradient coefficient, in any phantom phase with zero beyond-Horndeski coupling and zero tensor-speed excess, and for any non-zero constant beyond-Horndeski coupling on a cosmological-constant background.
---

Freezing Gravity is a dark-energy model in which the expansion history, including a crossing of the phantom divide $$w=-1$$, can be prescribed freely, while the linear perturbations are controlled by four separate parameters [[YYS]](#ref-yys). The question is whether those perturbations remain stable once pressureless matter is added to the model. For a small lapse-gradient coefficient $$\beta_3$$ they do not. On sub-horizon scales the scalar mode, with the dust slaved to it, obeys an oscillator equation whose squared frequency $$B_mH^2/\beta_3$$ does not depend on the wavenumber, with

$$
B_m=2(\alpha_H-\alpha_T)+3(1+\alpha_H)(1+w)\,\Omega_{\rm de}-3\alpha_H\,\Omega_m+\frac{2\dot\alpha_H}{H}. \tag{1}
$$

Where $$B_m<0$$ the perturbations grow at the rate $$\Gamma=H\sqrt{-B_m/\beta_3}$$. The symbols are defined in section 1: $$H$$ is the Hubble rate, $$w$$ and $$\Omega_{\rm de}$$ the equation of state and density fraction of the dark energy, $$\Omega_m$$ the density fraction of the dust, $$\alpha_T$$ the tensor-speed excess, $$\alpha_H$$ a beyond-Horndeski coupling, and a dot is a derivative in cosmic time. With constant parameters and $$\alpha_T=0$$, every phantom phase is unstable at $$\alpha_H=0$$, and every constant $$\alpha_H\neq0$$ is unstable on a cosmological-constant background, so the instability is not tied to the phantom crossing. No constant $$\alpha_H$$ stabilises the phantom phase of the four DESI DR2 $$w_0w_a$$ fits; at $$\beta_3=10^{-4}$$ the leading-order growth between $$z=2.33$$ and the crossing is 23 to 28 e-folds. The conditions under which this fails are listed in section 7: a running $$\alpha_H$$, a negative $$\alpha_T$$ near $$z\sim1$$, a $$\beta_3$$ of order 0.1, or a large $$\alpha_L$$.

The model and its vacuum stability conditions are due to Yao, Ye and Silvestri [[YYS]](#ref-yys). Yao [[Yao]](#ref-yao) added a minimally coupled perfect fluid, derived the no-ghost conditions at all wavenumbers and a gradient condition at high wavenumber, and presented the model as realising arbitrary backgrounds with stable perturbations. That the scalar's gradient energy involves the dark-energy combination $$\rho_{\rm de}+p_{\rm de}$$ and not the total $$\rho+p$$ is a familiar feature of the effective field theory of dark energy: in the Horndeski sound speed of Bellini and Sawicki [[BS]](#ref-bs), eq. (3.13), the matter contribution to $$\dot H$$ cancels against an explicit $$\tilde\rho_m+\tilde p_m$$ term, and Creminelli et al. [[CDNV]](#ref-cdnv) showed that a stable $$w<-1$$ phase needs operators beyond those of quintessence. De Felice, Frusciante and Papadomanolakis [[DFP]](#ref-dfp) classified the instabilities of modified gravity with matter, including tachyonic ones with rates far above $$H$$. What this page adds is the coefficient (1) for Freezing Gravity with dust, in particular its terms $$-3\alpha_H\Omega_m$$ and $$2\dot\alpha_H/H$$, an exact reduction of the dust system (Lemma 2) from which the rate follows, and the consequences for constant parameters and for the DESI fits. The conclusion bears on the DESI-motivated dark-energy interpretations discussed in [What a negative neutrino mass in BAO data would require](/quantum-gravity/neutrino-mimic/) and [The sign of energy exchange between vacuum and matter](/quantum-gravity/vacuum-matter-sign/), which address the same data with different mechanisms.

Units are $$M_{\rm Pl}^2/2=1$$, as in [[YYS]](#ref-yys), so that the Friedmann equation reads $$6H^2=\rho_{\rm tot}$$.

## 1. Background, perturbations and the model

*Background.* The universe is described by a flat Friedmann–Robertson–Walker metric $$ds^2=-N^2dt^2+a(t)^2d\mathbf x^2$$ with lapse $$N=1$$ on the background, scale factor $$a(t)$$ and Hubble rate $$H=\dot a/a$$. Two components are present: pressureless matter (dust) with density $$\rho_m\propto a^{-3}$$, and a dark-energy component with density $$\rho_{\rm de}$$ and pressure $$p_{\rm de}=w\rho_{\rm de}$$. The density fractions are $$\Omega_m=\rho_m/(6H^2)$$ and $$\Omega_{\rm de}=\rho_{\rm de}/(6H^2)$$, with $$\Omega_m+\Omega_{\rm de}=1$$. A cosmological constant has $$w=-1$$ and $$\rho_{\rm de}$$ constant. The line $$w=-1$$ is the phantom divide: a component with $$w<-1$$ is called phantom, and in the models considered here it is a dark-energy density that grows with time. Recent baryon acoustic oscillation (BAO) measurements by DESI, combined with the cosmic microwave background and supernovae, prefer a dark-energy equation of state $$w(a)=w_0+w_a(1-a)$$ (the CPL form) with $$w_0>-1$$ and $$w_a<0$$ [[DESI]](#ref-desi). Then $$w$$ is phantom at early times and crosses $$-1$$ at $$a_c=1+(1+w_0)/w_a$$, that is at redshift $$z_c=1/a_c-1$$.

*Perturbations.* A small perturbation is decomposed into Fourier modes of comoving wavenumber $$k$$. A mode is sub-horizon when $$\kappa\equiv k/(aH)\gg1$$, that is, when its wavelength is much shorter than the Hubble radius. We work to linear order, for one mode at a time, in the gauge in which the dark-energy scalar is uniform on constant-time slices (unitary gauge), so that the scalar's own perturbation is carried by the metric.

*The model.* In unitary gauge Freezing Gravity is defined by the action [[YYS]](#ref-yys), eq. (3.27),

$$
S_g=\int dt\,d^3x\,N\sqrt h\Big[\sigma_1\Big(K_{ij}K^{ij}-\tfrac13K^2\Big)+r_1R+r_2\,\mathfrak a_i\mathfrak a^i-\tfrac23K^2+4\mathcal F K-6\mathcal G^2\Big], \tag{2}
$$

where $$h_{ij}$$ is the spatial metric with determinant $$h$$, $$K_{ij}$$ its extrinsic curvature with trace $$K$$, $$R$$ its Ricci scalar, and $$\mathfrak a_i=\partial_i\ln N$$ the acceleration of the normal to the slices. The coefficients $$\sigma_1,r_1,r_2$$ (written $$\sigma_1,\rho_1,\rho_2$$ in [[YYS]](#ref-yys)) are functions of $$t$$ and $$N$$; $$\mathcal F$$ and $$\mathcal G$$ are functions of $$t$$ alone. For $$\sigma_1=r_1=1$$, $$r_2=0$$ and $$\mathcal F=0$$ the bracket is the Einstein–Hilbert Lagrangian plus a cosmological constant $$\propto\mathcal G^2$$. The terms $$4\mathcal FK$$ and $$-6\mathcal G^2$$ fix the background: on the flat background $$\rho_{\rm de}=6\mathcal G^2$$ and $$\rho_{\rm de}+p_{\rm de}=-4\dot{\mathcal F}$$ (shown below), so any $$w(t)$$ can be realised by choosing $$\mathcal F(t),\mathcal G(t)$$. The perturbations depend on the coefficients only through their values at $$N=1$$ and one derivative, which define the effective-field-theory parameters ([[YYS]](#ref-yys), eq. (4.2)):

$$
\sigma_1=\frac{1}{1+\alpha_L},\qquad r_1=\frac{1+\alpha_T}{1+\alpha_L},\qquad r_1+\partial_Nr_1=\frac{1+\alpha_H}{1+\alpha_L},\qquad r_2=\frac{\beta_3}{1+\alpha_L}.
$$

The parameter $$\alpha_T$$ is the excess of the tensor speed over the speed of light, $$\alpha_H$$ the beyond-Horndeski coupling of $$R$$ to the lapse perturbation, $$\alpha_L$$ the detuning of the two $$K^2$$ terms, and $$\beta_3$$ the coefficient of the lapse gradient energy $$(\partial_i\delta N)^2/a^2$$, in the notation of Gleyzes, Langlois and Vernizzi [[GLV]](#ref-glv) and Langlois et al. [[LMNV]](#ref-lmnv). Throughout, $$\alpha_L\ge0$$ and $$\beta_3>0$$. In vacuum the scalar mode of this model has kinetic coefficient $$\beta_3\kappa^2$$ and gradient coefficient $$2\alpha_H-2(1+\alpha_H)\dot H/H^2+2\dot\alpha_H/H$$ at $$\alpha_L=\alpha_T=0$$ ([[YYS]](#ref-yys), eq. (4.17)); the gradient coefficient is $$-2\dot H/H^2$$ at $$\alpha_H=0$$, negative in any phantom phase, which is why the vacuum conditions of [[YYS]](#ref-yys), eqs. (4.15)–(4.16), require a sufficiently large $$\alpha_H$$ there. The question is which of the two derivatives $$\dot H$$ and $$\dot{\mathcal F}$$ takes the place of $$\dot H$$ once dust is present. For the DESI+CMB+DESY5 fit at $$z=0.93$$, on the phantom side, $$-2\dot H/H^2=2.18$$ while $$-2\dot{\mathcal F}/H^2=3(1+w)\Omega_{\rm de}=-0.117$$: if the total $$\dot H$$ entered, matter would stabilise the phantom phase, and if $$\dot{\mathcal F}$$ enters it would not. Section 3 shows that $$\dot{\mathcal F}$$ enters.

*Dust.* We add pressureless matter in the form

$$
S_m=-\frac12\int d^4x\sqrt{-g}\;\rho_m\,\big(g^{\mu\nu}\partial_\mu\phi\,\partial_\nu\phi+1\big),
$$

where $$\rho_m(x)$$ and $$\phi(x)$$ are independent fields. Varying $$\rho_m$$ makes $$\partial_\mu\phi$$ a unit timelike vector, so $$\phi$$ is proper time along geodesic worldlines; varying $$\phi$$ gives $$\nabla_\mu(\rho_m\partial^\mu\phi)=0$$; and on shell the stress tensor is $$\rho_m\partial_\mu\phi\,\partial_\nu\phi$$, pressureless matter of density $$\rho_m$$. On the flat background with $$\phi=t$$, varying the lapse and the scale factor in (2) gives

$$
6H^2=6\mathcal G^2+\rho_m,\qquad \dot H=\dot{\mathcal F}-\frac{\rho_m}{4},\qquad \dot\rho_m=-3H\rho_m. \tag{3}
$$

So $$\rho_{\rm de}=6\mathcal G^2$$ and $$\rho_{\rm de}+p_{\rm de}=-4\dot{\mathcal F}$$ as claimed, and $$-2\dot{\mathcal F}/H^2=3(1+w)\Omega_{\rm de}$$. Radiation is not included. In the redshift range used below its density fraction is below $$10^{-3}$$, and we do not derive its effect on (1).

*Hypotheses.* Results below are stated under the following hypotheses, which are collected here because the escape routes of section 7 are the failures of each.

- (H1) $$\alpha_L=0$$ (section 5 treats small $$\alpha_L>0$$).
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
*Proof.* The expansion is routine, and four facts organise it. First, $$K_{ij}K^{ij}-\tfrac13K^2$$ is the square of the traceless part of $$K_{ij}$$, which vanishes on the background; at second order it receives only the shift, $$\tfrac23(k^2/a^2)^2\psi^2$$, so the $$N$$-dependence of $$\sigma_1$$ never enters, and writing $$\sigma_1=1-\alpha_L/(1+\alpha_L)$$ the part with coefficient 1 combines with $$-\tfrac23K^2$$ into the Einstein–Hilbert kinetic term. Second, $$R$$ and $$\mathfrak a_i$$ vanish on the background, so $$r_1$$ enters through $$r_1(1)\,\delta_2(\sqrt hR)$$ and $$(r_1+\partial_Nr_1)(1)\,\delta N\,\delta_1R$$, and $$r_2$$ through $$r_2(1)(\partial\delta N)^2$$; higher $$N$$-derivatives multiply terms of cubic order. Third, $$N\sqrt h\,K=\partial_t\sqrt h-\partial_i(\sqrt hN^i)$$ identically, so $$4\mathcal FK$$ contributes $$-4\dot{\mathcal F}\sqrt h$$, independent of the lapse. Fourth, the dust term is expanded from $$-g^{\mu\nu}\partial_\mu\phi\partial_\nu\phi=(\dot\phi-N^i\partial_i\phi)^2/N^2-h^{ij}\partial_i\phi\,\partial_j\phi$$. We carried out the expansion by computer algebra for fields proportional to $$\cos kx$$, starting from the ADM expressions, and checked that the Euler–Lagrange expressions of the result with respect to each of $$\delta N,\psi,\zeta,v,\delta\rho_m$$ equal those of (4) up to the overall factor $$\tfrac12$$ (from the average of $$\cos^2$$) once (3) is used.
</div>

Two checks. Without dust and at $$\alpha_L=\alpha_T=0$$ the $$\psi$$ equation gives $$\delta N=\dot\zeta/H$$; substituting it and integrating the $$\zeta\dot\zeta$$ term by parts reproduces the kinetic and gradient coefficients of [[YYS]](#ref-yys), eq. (4.17), quoted in section 1, including the $$2\dot\alpha_H/H$$ term. With dust, (4) is the quadratic action behind the constraint equations written in [[Yao]](#ref-yao), eqs. (3.8) and (3.9), which we did not rederive from that paper's variables.

## 3. An exact reduction at $$\alpha_L=0$$

The idea is that the dust equation and the momentum constraint remove all but one fast degree of freedom, and that both remaining terms of its Lagrangian carry two spatial derivatives. The dust fixes the lapse: $$\delta\rho_m$$ appears only multiplying $$\dot v-\delta N$$, so $$\delta N=\dot v$$. At $$\alpha_L=0$$ the shift $$\psi$$ appears linearly in (4) and is a multiplier. It is natural to trade $$\zeta$$ for

$$
\mathcal R\equiv\zeta-Hv,
$$

the curvature perturbation on the slices of constant dust proper time, since $$a^2(t)e^{2\zeta}=a^2(t+v)e^{2\mathcal R}$$ to first order, and to shift the multiplier to $$\tilde\psi\equiv\psi+(1+\alpha_H)v$$. The new variables remove the cross term between $$\zeta$$ and $$\delta N$$ at the price of a total time derivative. Define

$$
D\equiv(\alpha_H-\alpha_T)H+\dot\alpha_H ,
$$

so that $$B_m$$ of (1) can be written $$B_m=2(\alpha_H-\alpha_T)-2(1+\alpha_H)\dot{\mathcal F}/H^2-3\alpha_H\Omega_m+2\dot\alpha_H/H$$ using $$\rho_m=6H^2\Omega_m$$.

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

The $$\psi$$ term is $$-4\psi(\dot\zeta-H\dot v+\rho_mv/4)=-4\psi(\dot{\mathcal R}+\dot{\mathcal F}v)$$ by (3), and $$-4(1+\alpha_H)v(\dot{\mathcal R}+\dot{\mathcal F}v)$$ is absorbed by the shift $$\psi\to\tilde\psi$$, leaving $$+4(1+\alpha_H)\dot{\mathcal F}v^2$$. Collecting the $$v^2$$ terms, including $$2(1+\alpha_T)H^2v^2$$ from $$\zeta^2$$ and $$-\rho_mv^2/2$$, and using $$\dot H=\dot{\mathcal F}-\rho_m/4$$,

$$
2(\alpha_T-\alpha_H)H^2-2H\dot\alpha_H+2(1+\alpha_H)\dot{\mathcal F}+\tfrac12\alpha_H\rho_m=-H^2B_m ,
$$

and the $$\mathcal Rv$$ terms, $$4(1+\alpha_T)H$$ from $$\zeta^2$$ and $$-4[H(1+\alpha_H)+\dot\alpha_H]$$ from the line above, add to $$-4D$$. The terms without $$k^2$$ are rewritten with $$\dot\zeta-H\dot v=\dot{\mathcal R}+\dot Hv$$. This is (5); the equality was also confirmed symbolically, with $$\alpha_T,\alpha_H,\beta_3$$ arbitrary functions of time.

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

The step with content is the cancellation between the $$\dot H$$ produced by integrating $$Hv\dot v$$ by parts and the $$\dot{\mathcal F}$$ produced by the momentum constraint. Matter's share of $$\dot H$$ is traded for $$\dot{\mathcal F}$$, leaving $$\tfrac12\alpha_H\rho_m$$ behind, which is the origin of the term $$-3\alpha_H\Omega_m$$. At $$\alpha_H=0$$ that remainder vanishes and the coefficient of $$v^2$$ is the quintessence-type gradient energy $$2\dot{\mathcal F}=-\tfrac12(\rho_{\rm de}+p_{\rm de})$$. The system (6)–(7) contains $$k$$ only through $$\kappa^{-2}$$ in the last term of (7), so it is valid for every $$\kappa$$ at $$\alpha_L=0$$, not only for $$\kappa\gg1$$.

## 4. The fast mode

In (6)–(7) the dynamics separates by time scale. The variable $$v$$ has inertia $$\beta_3$$ and stiffness $$H^2B_m$$, so it oscillates or grows at a rate of order $$H/\sqrt{\beta_3}$$. The variables $$\mathcal R$$ and $$\tilde\psi$$ are slow: by (6) their rates are $$\dot{\mathcal F}v$$ and $$Dv$$, of order $$H^2v$$ and $$Hv$$. Define

$$
B_{\rm eff}\equiv B_m-\frac{9\Omega_m\dot{\mathcal F}}{H^2\kappa^2}=B_m+\frac{27(1+w)\Omega_m\Omega_{\rm de}}{2\kappa^2}.
$$

<div class="proposition" markdown="1">
**Proposition 1** (fast mode at leading order). Assume (H1)–(H4), and let $$[t_1,t_2]$$ be an interval on which $$B_{\rm eff}\le-b_0<0$$ with $$b_0$$ independent of $$\varepsilon$$. As $$\varepsilon\to0$$, the system (6)–(7) has a solution in which $$v$$ grows with

$$
\frac{d\ln\lvert v\rvert}{dt}=H\sqrt{\frac{-B_{\rm eff}}{\beta_3}}+O(H),\qquad \mathcal R,\ \tilde\psi=O(\varepsilon)\,v, \tag{8}
$$

and $$\zeta=Hv+\mathcal R$$ grows with it. Where $$B_{\rm eff}>0$$ the corresponding solutions oscillate at frequency $$H\sqrt{B_{\rm eff}/\beta_3}$$ without growth. For $$\kappa\gg1$$, $$B_{\rm eff}\to B_m$$, the quantity (1), and the growth rate is $$\Gamma=H\sqrt{-B_m/\beta_3}$$, independent of $$k$$.
</div>

This is a derivation at leading order, not a bound on solutions: it combines exact size estimates with the standard WKB theory of linear systems with a large parameter, which we use without reproving, and the numerical evidence of this section.

<div class="proof" markdown="1">
*Proof.* Suppose $$v$$ varies at a rate $$\Gamma\sim H/\varepsilon$$. Then (6) gives $$\mathcal R\simeq-\dot{\mathcal F}v/\Gamma=O(\varepsilon H)v$$ and $$\tilde\psi\simeq Dv/\Gamma=O(\varepsilon)v$$. The first two terms on the right of (7) are therefore of order $$\varepsilon H^2v$$, and the third is the explicit $$\kappa^{-2}$$ term already moved into $$B_{\rm eff}$$. Equation (7) becomes

$$
\beta_3\big(\ddot v+H\dot v+(\dot\beta_3/\beta_3)\dot v\big)+H^2B_{\rm eff}\,v=O(\varepsilon H^2)\,v .
$$

The friction and $$\dot\beta_3$$ terms are smaller than $$\ddot v$$ by $$H/\Gamma=O(\varepsilon)$$. The remaining oscillator equation has local frequency squared $$H^2B_{\rm eff}/\beta_3$$ and varies on the Hubble time, so its WKB solutions have $$d\ln\lvert v\rvert/dt=H\sqrt{-B_{\rm eff}/\beta_3}+O(H)$$, the correction collecting the friction, the slow variation of the amplitude factor and the $$O(\varepsilon)$$ shift of $$B_{\rm eff}$$. The last statement follows from $$B_{\rm eff}-B_m=O(\kappa^{-2})$$.
</div>

In the language of the effective field theory of dark energy, $$v$$ is minus the Goldstone time shift of the dark energy seen from the dust's rest frame, and the $$v$$ part of (5) is $$a\,k^2[\beta_3\dot v^2-B_mH^2v^2]$$. The kinetic term comes from $$\mathfrak a_i\mathfrak a^i$$ and carries two spatial gradients, as does the potential term, which is why $$k^2$$ cancels in the rate. The instability is of the type that De Felice et al. [[DFP]](#ref-dfp) call tachyonic: a negative squared frequency $$B_mH^2/\beta_3$$ that is independent of $$k$$ and much larger than $$H^2$$ in magnitude, as opposed to a gradient instability, whose rate grows with $$k$$. The term $$2\dot\alpha_H/H$$ enters through an integration by parts and would be lost in a frozen-coefficient treatment of the action.

*Checks.* We integrate the linear equations of the reduced system without approximating the time dependence: the Euler–Lagrange equations of the two-variable Lagrangian in $$(\mathcal R,v)$$ obtained from (4) with $$\delta N=\dot v$$ and the shift eliminated (for $$\alpha_L>0$$), on $$w_0w_a$$ backgrounds with $$H_0=1$$. The amplitude is $$\mathcal X=(\zeta^2+H^2v^2)^{1/2}$$ and its local growth rate is $$d\ln\mathcal X/d\ln a$$. Initial data at $$z=4$$ are $$Hv=\zeta=10^{-3}$$ ($$\mathcal R=0$$), $$\dot v=0$$ and $$\dot\zeta=-\rho_mv/4$$, which satisfy the constraint. Comoving wavenumbers are in units of $$H_0$$; $$k=10H_0$$ is $$3.3\times10^{-3}\,h\,{\rm Mpc}^{-1}$$. The first test is the exact $$\alpha_L=0$$ system (6)–(7), which isolates the $$k$$-dependence.

| $$k/H_0$$ | $$\kappa$$ | measured rate $$/H$$ | $$\sqrt{-B_m/\beta_3}$$ | $$\sqrt{-B_{\rm eff}/\beta_3}$$ |
|---|---|---|---|---|
| 3 | 3.3 | 39.18 | 34.14 | 39.21 |
| 10 | 11.1 | 34.54 | 34.14 | 34.63 |
| 30 | 33.3 | 34.10 | 34.14 | 34.19 |
| 100 | 111 | 34.05 | 34.14 | 34.14 |
| 1000 | 1110 | 34.05 | 34.14 | 34.14 |

The background is the DESI+CMB+DESY5 fit of section 6 at $$z=1$$, with $$\beta_3=10^{-4}$$, $$\alpha_H=\alpha_T=0$$, and initial data at $$z=2.33$$. The rate at fixed $$z$$ is independent of $$k$$ for $$\kappa\gtrsim30$$ to 0.3%, and the explicit $$\kappa^{-2}$$ term of $$B_{\rm eff}$$ accounts for the departure at $$\kappa\lesssim11$$.

The second test is the case $$\alpha_L=10^{-5}$$, the small positive value that makes the system regular and for which the band edge of section 5 lies beyond $$\kappa\approx100$$. Here $$k=10H_0$$, $$\beta_3=10^{-4}$$, and the rate is the slope of $$\ln\mathcal X$$ over $$\pm0.02$$ in $$\ln a$$.

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

Agreement is within 8% in all rows, and the largest deviations occur where $$B_m$$ changes quickly in time (DESY5 at $$z=0.5$$; the $$\Lambda$$CDM rows with $$\alpha_H=-0.3$$, where $$B_m$$ vanishes at $$z=0.66$$), which is outside the WKB regime. The expected size of the corrections is a few per cent: friction lowers the rate by $$H/2$$ (1.5%), the $$O(\varepsilon)$$ terms of (8) are of the same order, and the explicit $$\kappa^{-2}$$ term raises it. The $$\Lambda$$CDM background has $$\Omega_{m,0}=0.3027$$, the DESI+CMB value ([[DESI]](#ref-desi), eq. (21)). On $$\Lambda$$CDM with $$\alpha_H=0$$, where $$B_m=0$$, no mode grows.

The local rates do not depend on the initial data. Integrated growth does, because the growing mode needs a fraction of an e-fold to dominate: for the first fit of section 6, imposing the initial data at $$z=2.33$$ rather than $$z=4$$ gives 19.6 instead of 23.0 e-folds between $$z=2.33$$ and the crossing.

*A worked value.* On $$\Lambda$$CDM with $$\alpha_H=0.3$$ and $$\alpha_T=0$$ at $$z=1.5$$, $$\Omega_m=0.3027\cdot15.625/(0.3027\cdot15.625+0.6973)=0.8715$$, so $$B_m=0.3(2-3\cdot0.8715)=-0.184$$ and $$\Gamma/H=\sqrt{0.184/10^{-4}}=42.9$$, against 42.0 measured. In the vacuum limit ($$\Omega_m\to0$$), (1) reduces to the coefficient of [[YYS]](#ref-yys), eq. (4.17), at $$\alpha_T=0$$; for $$\alpha_H=\alpha_T=0$$ and $$w=-1$$, $$B_m=0$$, as it must for general relativity with dust and a cosmological constant.

## 5. Finite $$\alpha_L$$: the band

For $$\alpha_L>0$$ the shift enters (4) quadratically. The exact difference between (4) and its $$\alpha_L=0$$ counterpart, in the variables of Lemma 2, is

$$
\ell(\alpha_L)-\ell(0)=\frac{\alpha_L}{1+\alpha_L}\Big[-\frac23\frac{k^4}{a}\big(\tilde\psi-(1+\alpha_H)v\big)^2-a k^2\big(\beta_3\dot v^2+4(1+\alpha_H)\zeta\dot v+2(1+\alpha_T)\zeta^2\big)\Big], \tag{9}
$$

which can be checked by expansion. The second part of the bracket changes the coefficients in (5) by relative amounts of order $$\alpha_L$$. The first part, with $$\tilde\psi=O(\varepsilon)v$$ as in Proposition 1, contributes $$-\tfrac23\alpha_L(1+\alpha_H)^2\kappa^2\cdot ak^2H^2v^2$$ at leading order, which is enhanced by $$\kappa^2$$. Its effect on (7) is to replace

$$
B_{\rm eff}\to B_{\rm eff}+\tfrac23\alpha_L(1+\alpha_H)^2\kappa^2 \tag{10}
$$

when $$\alpha_L\ll1$$ and $$\alpha_L\kappa^2=O(1)$$. This is a derivation at leading order in $$\varepsilon$$ and $$\alpha_L$$; we have not bounded the remainder. The added term equals $$\beta_3c_s^2\kappa^2$$, where $$c_s^2=\tfrac23\alpha_L(1+\alpha_H)^2/\beta_3$$ is the leading term of the high-$$k$$ sound speed $$\alpha_L[2(1+\alpha_H)^2-\beta_3(1+\alpha_T)]/[3\beta_3(1+\alpha_L)]$$ of the scalar in vacuum ([[YYS]](#ref-yys), large-$$k$$ limit of the ratio of the coefficients of $$\zeta^2$$ and $$\dot\zeta^2$$), so the squared frequency interpolates as $$B_mH^2/\beta_3+c_s^2k^2/a^2$$. The growth therefore occupies a band

$$
aH\ll k<k_{\max},\qquad k_{\max}=\frac{aH}{1+\alpha_H}\sqrt{\frac{3\lvert B_m\rvert}{2\alpha_L}} , \tag{11}
$$

which extends to all sub-horizon scales as $$\alpha_L\to0$$ and closes (falls to $$k\lesssim aH$$) for $$\alpha_L\gtrsim\tfrac32\lvert B_m\rvert$$. Numerical test: $$\Lambda$$CDM, $$\alpha_H=0.3$$, $$\alpha_L=10^{-3}$$, $$z=1.5$$ ($$B_m=-0.184$$, $$\kappa_{\max}=12.8$$), initial data at $$z=3$$.

| $$\kappa$$ | measured $$\Gamma/H$$ | from (10) | leading order |
|---|---|---|---|
| 5.4 | 38.0 | 39.0 | 42.9 |
| 8.6 | 31.1 | 31.8 | 42.9 |
| 10.7 | 23.2 | 23.4 | 42.9 |

Beyond the edge the integrated growth falls: the logarithm of the maximum of $$\mathcal X$$ between $$z=3$$ and $$0$$, divided by its value at $$z=3$$, is 30.6, 18.5, 4.1, 1.2 and 0.0 e-folds at $$k=5,10,15,20,30\,H_0$$ (initial data at $$z=4$$). The local rate is not quoted there, since the amplitude oscillates and its logarithmic slope is not defined. The derivation of (10) assumes $$\kappa\gg1$$ and does not cover scales near the horizon.

## 6. Constant parameters

With $$\dot\alpha_H=0$$ the following statements follow from (1) by inspection; each holds in the band and at leading order, for $$\kappa\gg1$$.

<div class="corollary" markdown="1">
**Corollary 1** (phantom phases). If $$\alpha_H=\alpha_T=0$$, then $$B_m=3(1+w)\Omega_{\rm de}$$. There is no fast growth exactly when $$w\ge-1$$, with $$w=-1$$ marginal at this order. Every phantom phase ($$w<-1$$) is unstable.
</div>

<div class="corollary" markdown="1">
**Corollary 2** (constant $$\alpha_H$$ on $$\Lambda$$CDM). If $$w=-1$$ and $$\alpha_T=0$$, then $$B_m=\alpha_H(2-3\Omega_m)$$. A constant $$\alpha_H>0$$ is unstable while $$\Omega_m>\tfrac23$$, which includes the whole matter era, and a constant $$\alpha_H<0$$ is unstable while $$\Omega_m<\tfrac23$$. For $$\Omega_{m,0}=0.3027$$ the transition is at $$z=0.664$$.
</div>

The statement for $$\alpha_H>0$$ does not need $$w=-1$$: whenever $$\Omega_m\to1$$ and $$(1+w)\Omega_{\rm de}\to0$$, (1) gives $$B_m\to-\alpha_H$$ at $$\alpha_T=0$$. The vacuum condition of [[YYS]](#ref-yys), eqs. (4.15)–(4.16), allows constant $$\alpha_H\ge0$$ on a $$\Lambda$$CDM background; with dust, the matter era excludes every $$\alpha_H>0$$. The exact evolution confirms this. With $$k=20H_0$$, $$\alpha_L=10^{-5}$$, $$\beta_3=10^{-4}$$ and initial data at $$z=4$$, the growth between $$z=3$$ and $$0$$ (logarithm of the maximum of $$\mathcal X$$ over $$0\le z\le3$$ divided by $$\mathcal X(z=3)$$) is compared with $$\int\Gamma\,dt$$:

| $$\alpha_H$$ | $$+0.3$$ | $$+0.1$$ | $$0$$ | $$-0.1$$ | $$-0.3$$ |
|---|---|---|---|---|---|
| exact growth (e-folds) | 34.5 | 19.4 | 0.0 | 9.4 | 17.9 |
| $$\int\Gamma\,dt$$ (e-folds) | 35.1 | 20.3 | 0.0 | 11.2 | 19.5 |

For $$\alpha_H<0$$ the unstable epoch $$z<0.66$$ starts late, and the exact growth depends more on the initial data: starting from $$\zeta=10^{-3}$$ with $$v=\dot\zeta=\dot v=0$$ instead gives 12.7 and 21.3 e-folds for $$\alpha_H=-0.1,-0.3$$.

For a general background write $$B_m=\alpha_H\,C(z)+3(1+w)\Omega_{\rm de}-2\alpha_T$$ at constant $$\alpha_H$$, with $$C\equiv2-3\Omega_m+3(1+w)\Omega_{\rm de}$$.

<div class="corollary" markdown="1">
**Corollary 3** (no constant $$\alpha_H$$). Let $$\alpha_T=0$$. If $$C(z_*)=0$$ at a redshift $$z_*$$ where $$w(z_*)<-1$$, then $$B_m(z_*)=3(1+w)\Omega_{\rm de}<0$$ for every constant $$\alpha_H$$.
</div>

We evaluate this on the DESI DR2 $$w_0w_a$$ fits that combine BAO, CMB and supernovae, with the posterior means of $$(\Omega_m,w_0,w_a)$$ from Table V of [[DESI]](#ref-desi). In each fit $$w$$ crosses $$-1$$ at $$z_c$$ and is phantom above it, and $$C$$ has one zero $$z_*$$ for $$z\in(0.05,3)$$, which lies on the phantom side.

| fit | $$\Omega_m,\ w_0,\ w_a$$ | $$z_c$$ | $$\min B_m$$ at $$\alpha_H=0$$ (at $$z$$) | $$z_*$$ | $$B_m(z_*)$$ | $$\int\Gamma dt$$, $$2.33\to z_c$$ (exact) | $$3\to z_c$$ |
|---|---|---|---|---|---|---|---|
| DESI+CMB+DESY5 | 0.3191, $$-0.752$$, $$-0.86$$ | 0.405 | $$-0.117$$ (0.93) | 0.611 | $$-0.085$$ | 24.2 (23.0) | 27.3 |
| DESI+CMB+Union3 | 0.3275, $$-0.667$$, $$-1.09$$ | 0.440 | $$-0.137$$ (0.96) | 0.615 | $$-0.089$$ | 25.3 (24.2) | 28.6 |
| DESI+CMB+Pantheon+ | 0.3114, $$-0.838$$, $$-0.62$$ | 0.354 | $$-0.096$$ (0.89) | 0.607 | $$-0.078$$ | 22.9 (21.6) | 25.8 |
| DESI+CMB | 0.353, $$-0.42$$, $$-1.75$$ | 0.496 | $$-0.192$$ (0.99) | 0.627 | $$-0.104$$ | 28.2 (27.3) | 31.7 |

Growth columns are e-folds at $$\alpha_H=\alpha_T=0$$ and $$\beta_3=10^{-4}$$ from $$\int\Gamma\,dt$$; the parenthesis is the growth $$\ln[\mathcal X(z_c)/\mathcal X(2.33)]$$ in the exact evolution at $$k=20H_0$$, $$\alpha_L=10^{-5}$$, with initial data at $$z=4$$ (and dependence on initial data as above). The redshift $$z=2.33$$ is the effective redshift of the highest-redshift DESI BAO measurement; extending the CPL form to $$z=20$$ raises the first entry to 34 e-folds, but that range lies beyond the data. The growth scales as $$\beta_3^{-1/2}$$ in the exponent, so for the first fit it is $$e^{24}$$ at $$\beta_3=10^{-4}$$ and $$e^{2.4}$$ at $$\beta_3=10^{-2}$$ at the level of $$\int\Gamma dt$$. The posterior means are not best-fit points, and the DESY5 supernova sample has been recalibrated since the release; the sign structure of (1), on which Corollaries 1 to 3 rest, does not depend on these values.

In all four fits $$z_*$$ lies on the phantom side, so Corollary 3 applies and no constant $$\alpha_H$$ stabilises them. A constant $$\alpha_T$$ of at most $$-0.04$$ to $$-0.05$$ would make $$B_m(z_*)\ge0$$ (it enters as $$-2\alpha_T$$), but GW170817 [[GW170817]](#ref-gw) bounds a constant $$\lvert\alpha_T\rvert$$ far below that.

## 7. What the theorem does not exclude

The hypotheses of Corollaries 1 to 3 are constant parameters, $$\beta_3\ll\lvert B_m\rvert$$, a band that exists, and dust as the only matter. Relaxing each gives a route out. We list the routes with the conditions under which they work on the DESI+CMB+DESY5 background, without choosing among them.

*A running $$\alpha_H$$.* The term $$2\dot\alpha_H/H$$ in (1) can make $$B_m$$ positive throughout. For $$\alpha_H=s\,(\ln a+0.5)$$, which runs from $$-0.89\,s$$ at $$z=3$$ to $$0.5\,s$$ today, the exact evolution ($$k=20H_0$$, $$\alpha_L=10^{-5}$$, initial data at $$z=4$$; growth is the logarithm of the maximum of $$\mathcal X$$ over $$0\le z\le3$$ divided by $$\mathcal X(z=3)$$) gives

| $$s$$ | min $$B_m$$ | exact growth (e-folds) | $$\int\Gamma dt$$ with (1) | $$\int\Gamma dt$$ without $$2\dot\alpha_H/H$$ |
|---|---|---|---|---|
| 0 | $$-0.117$$ | 25.2 | 27.3 | 27.3 |
| 0.03 | $$-0.056$$ | 10.4 | 10.2 | 24.2 |
| 0.06 | $$+0.006$$ | 0.7 | 0.0 | 21.4 |
| 0.10 | $$+0.088$$ | 0.3 | 0.0 | 19.0 |

The comparison confirms the $$2\dot\alpha_H/H$$ term. A rate $$d\alpha_H/d\ln a\gtrsim0.06$$ suffices on this background and in this linear parametrisation; it satisfies the vacuum condition $$\dot\alpha_H\ge0$$ of [[YYS]](#ref-yys), eq. (4.16). This route is closed if $$\alpha_H$$ must vanish: the decay of gravitational waves into dark-energy fluctuations forces $$\alpha_H=0$$ in the effective field theory of dark energy [[CLTV]](#ref-cltv). Whether that argument carries over to Freezing Gravity, whose scalar has the additional operator $$\mathfrak a_i\mathfrak a^i$$, is not examined here. With $$\alpha_H\equiv0$$, Corollary 1 leaves the other routes.

*A negative $$\alpha_T$$ at intermediate redshift.* A constant $$\alpha_T=-0.03$$ ($$-0.07$$) reduces the exact growth from 25.2 to 13.5 (1.7) e-folds, against $$\int\Gamma dt=11.5$$ (0.0). GW170817 constrains $$\alpha_T$$ at $$z\approx0$$ only, so an $$\alpha_T(z)$$ that is about $$-0.06$$ near $$z\sim1$$ and vanishes today is not excluded by it.

*A large $$\beta_3$$.* Proposition 1 requires $$\beta_3\ll\lvert B_m\rvert$$. For $$\beta_3\gtrsim\lvert B_m\rvert_{\max}\approx0.1$$ to $$0.2$$ the rate drops to order $$H$$ and the proposition no longer applies. At $$\beta_3=0.1$$, $$\int\Gamma dt$$ from $$z=3$$ gives $$e^{0.86}$$, but the exact evolution shows no growth of $$\mathcal X$$ beyond its initial value for $$k=5,20,100\,H_0$$; at $$\beta_3=10^{-2}$$ the exact growth is $$e^{0.64}$$, $$e^{0.22}$$ and none at those wavenumbers, against $$e^{2.7}$$ from $$\int\Gamma dt$$. The vacuum condition $$\beta_3\le2(1+\alpha_H)^2$$ of [[YYS]](#ref-yys), eq. (4.15), is still satisfied there.

*A large $$\alpha_L$$.* For $$\alpha_L\gtrsim\tfrac32\lvert B_m\rvert\approx0.2$$ to $$0.3$$, $$k_{\max}$$ in (11) falls below $$aH$$ and the band closes. Near-horizon modes are then outside Proposition 1, and in the exact evolution they can still grow. On the DESY5 background with $$\alpha_H=\alpha_T=0$$, $$\beta_3=10^{-4}$$ and initial data at $$z=3$$, the growth between $$z=3$$ and $$0$$ is

| $$\alpha_L$$ | $$k=H_0$$ | $$k=3H_0$$ | $$k=10H_0$$ |
|---|---|---|---|
| $$10^{-3}$$ | 55.4 | 28.1 | 3.6 |
| $$10^{-2}$$ | 52.5 | 9.3 | 0.0 |
| $$0.05$$ | 35.3 | 0.0 | 0.0 |

in e-folds. Whether a large $$\alpha_L$$ removes the instability therefore depends on scales $$k\lesssim aH$$ that this page does not resolve.

*Other matter content or couplings.* The page treats minimally coupled dust with $$\alpha_L\ge0$$. A non-minimal coupling of dust to the scalar, radiation with $$\alpha_H\neq0$$, and the branch $$-1<\alpha_L<0$$ are not treated.

## 8. The closed-form conditions of the matter-coupled model

For $$\alpha_L>0$$, Yao [[Yao]](#ref-yao) states the conditions $$\alpha_L>-1$$ (its eq. (3.24)), $$\alpha_A>0$$ (3.25), the no-ghost conditions (3.26), and the high-$$k$$ gradient condition (3.39), $$(1+\alpha_H)^2>\tfrac12\alpha_A(1+\alpha_T)>0$$, where $$\alpha_A$$ is our $$\beta_3$$. In that paper the no-ghost conditions hold at all wavenumbers when $$\epsilon_m<(\sqrt{1+\alpha_L}+\sqrt{\alpha_L})^2$$, with $$\epsilon_m=\alpha_A(1+w_m)\bar\rho_m/(4M^2H^2(1+\alpha_L))$$ and $$M^2=2/(1+\alpha_L)$$, which for dust in our units is $$\epsilon_m=\tfrac34\alpha_A\Omega_m$$.

Consider the DESI+CMB+DESY5 background with $$\beta_3=\alpha_A=10^{-4}$$, $$\alpha_H=\alpha_T=0$$ and $$\alpha_L=10^{-4}$$. Then (3.24) and (3.25) hold, $$\epsilon_m<7.4\times10^{-5}$$ for $$z\le3$$ so that the bound above is satisfied by a wide margin, and (3.39) reads $$1>5\times10^{-5}$$. We evaluated the exact linear evolution of section 4 for these parameters: at $$k=10H_0$$ the amplitude grows by 22.5 e-folds between $$z=2.33$$ and the crossing (19.5 with the initial data imposed at $$z=2.33$$), against 24.2 from (1) at leading order. At $$\alpha_L=10^{-3}$$ the amplitude at $$k=5H_0$$ grows by 21.5 e-folds over the same interval, and at $$k=3H_0$$ by 28.1 e-folds between $$z=3$$ and $$0$$ (initial data at $$z=3$$). For these parameters the band (11) extends to $$\kappa_{\max}\approx27$$ to $$42$$ at $$\alpha_L=10^{-4}$$ for $$0.5\le z\le1$$, and the evolved modes have $$\kappa\approx10$$. The high-$$k$$ expansion of [[Yao]](#ref-yao), eq. (3.34), is an expansion in inverse powers of $$k$$ at fixed $$\alpha_L\alpha_A\neq0$$, and its leading term is the $$c_s^2k^2/a^2$$ of (10); we have not evaluated its general WKB condition (3.43) on a phantom-crossing background.

## 9. Open points

- Proposition 1 is a leading-order statement in $$\sqrt{\beta_3}$$, obtained from the exact system (6)–(7) by size estimates and WKB; the remainder is not bounded. The numerical agreement is a few per cent where $$B_m$$ varies slowly. The band edge (10) is derived at leading order only.
- The scales $$k\lesssim aH$$ are not resolved. The exact evolution shows growth there even when the band is closed, and the kinetic matrix of the matter-coupled model degenerates as $$k\to0$$ ([[Yao]](#ref-yao), eq. (3.33)).
- The branch $$-1<\alpha_L<0$$, a radiation component with $$\alpha_H\ne0$$, and time-dependent $$\alpha_L$$ are not treated.
- It is not known whether the gravitational-wave decay bound $$\alpha_H=0$$ applies to Freezing Gravity, which decides whether a running $$\alpha_H$$ is available.
- The growth factors exceed $$10^{10}$$. In an unstable model linear theory fails at a time set by the initial amplitude of $$v$$, and what happens after that is not addressed.
- A Freezing Gravity realisation of the DESI phantom crossing needs one of the conditions of section 7. Which of them, if any, is compatible with large-scale-structure and lensing data has not been examined.

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
