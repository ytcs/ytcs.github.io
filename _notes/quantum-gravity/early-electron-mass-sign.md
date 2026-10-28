---
title: A sign rule for an early change in the electron mass
date: 2026-10-28
order: 10
description: A scalar field that follows the minimum of its effective potential in the Standard Model plasma can only make the electron lighter in the past, while cosmological fits prefer it about one percent heavier at recombination. A field with a linear coupling allowed by the MICROSCOPE test of the equivalence principle would have to move further than its kinetic energy allows.
---

Fits of cosmic microwave background (CMB) and DESI baryon acoustic oscillation (BAO) data in which the electron mass at recombination is a free parameter prefer it heavier than in the laboratory: $$\delta\equiv m_e(z_\ast)/m_e^{\rm lab}-1$$ comes out between $$+0.6\%$$ and $$+1.2\%$$, with uncertainties of about $$0.5\%$$ (section 1). We ask whether a scalar field $$\phi$$ on which $$m_e$$ depends can produce a positive $$\delta$$, and find two obstructions. First, suppose $$\phi$$ adiabatically follows a minimum of its effective potential, and the only sources of that potential besides its bare potential are Standard Model (SM) particles whose masses do not decrease when $$m_e$$ increases. Then $$m_e$$ at any earlier epoch is at most its value today, so $$\delta\le0$$ (section 3). The statement compares cosmic-mean states and assumes that the laboratory value of $$m_e$$ is set by an environment less dense than the recombination plasma. Second, a light field whose coupling $$\beta_e\equiv M_P\,d\ln m_e/d\phi$$ does not vary along its path is limited by the MICROSCOPE test of the equivalence principle to $$\lvert\beta_e\rvert<5.6\times10^{-4}$$ at the $$2\sigma$$ level. A shift $$\delta=1\%$$ then requires the field to move by about $$18\,M_P$$ since recombination, while its kinetic energy allows at most $$17.1\,M_P\sqrt{\langle\Omega_{\rm kin}\rangle}$$, where $$\langle\Omega_{\rm kin}\rangle\le1$$ is the field's mean share of the total energy density in kinetic form. Hence $$\delta\le0.96\%\times\sqrt{\langle\Omega_{\rm kin}\rangle}$$ at $$2\sigma$$ (section 4). Three routes are not covered: fields that do not track a minimum (frozen by Hubble friction, with couplings quadratic in $$\phi$$), fields that are also sourced by dark matter, for which the same argument requires the dark matter to have been lighter at recombination by a fraction of at least $$0.9\times10^{-4}\,\delta$$, and fields that track the local density today (section 5).

Full likelihood analyses that free $$m_e$$ at recombination find values above the laboratory one once DESI BAO are included [[ACT DR6]](#ref-act) [[Toda–Seto]](#ref-toda-seto) [[Weiner]](#ref-weiner). Weiner finds a time-varying electron mass to be the most effective of the high-redshift modifications he considers at removing the geometric information on the neutrino mass sum, which is what drives the BAO preference for a negative sum. Baryakhtar, Simon and Weiner studied the cosmology of hyperlight scalars that modulate $$m_e$$ and the fine-structure constant [[BSW 2024]](#ref-bsw1). In a second paper [[BSW 2025]](#ref-bsw2) (their eqs. 2.25–2.30) they computed the in-medium potential that SM particles give such a scalar, proportional to the trace $$\rho-3p$$ of each species' stress tensor, and found that for positive quadratic couplings it gives the field a positive in-medium mass, which drives it towards the symmetric point $$\phi=0$$, where $$m_e$$ is smallest, and can do so before recombination. Ghosh, Boddy and Yu constrained ultralight dark matter whose quadratic couplings modulate $$m_e$$ and the fine-structure constant [[GBY]](#ref-gby). Smith, Mylova, van de Bruck, Burgess and Di Valentino fitted an axio-dilaton model in which particle masses depend on the dilaton, and found that the data prefer couplings large enough to have been detected in solar-system tests [[Smith et al.]](#ref-smith). This note adds the sign rule as a statement about any field that tracks a minimum, whatever its bare potential, its extension to fields also sourced by dark matter, and the excursion bound for linear couplings.

Familiarity with the thermodynamics of ideal quantum gases and with scalar fifth forces is assumed. We use natural units and the reduced Planck mass $$M_P=(8\pi G)^{-1/2}=2.435\times10^{18}\,{\rm GeV}$$.

## 1. The value of δ that the data prefer

Hydrogen recombines when the photon temperature falls to a fixed fraction of the atomic binding energy, which is proportional to $$m_e$$. A heavier electron therefore makes recombination happen earlier, at a larger redshift $$z_\ast$$, and shrinks the sound horizon that sets the BAO scale. Published fits with $$m_e(z_\ast)$$ free give:

| data | $$\delta$$ (%) |
|---|---|
| Planck + ACT DR6 + CMB lensing + DESI DR1 BAO [[ACT DR6]](#ref-act) | $$0.96\pm0.60$$ |
| the same + type Ia supernovae [[ACT DR6]](#ref-act) | $$0.63\pm0.56$$ |
| ACT DR6 + Planck ($$\ell<1000$$) + lensing + DESI DR2 BAO + supernovae [[Toda–Seto]](#ref-toda-seto) | $$0.78\pm0.47$$ |
| Planck + ACT + SPT + DESI DR2 BAO [[Weiner]](#ref-weiner) | $$0.76\pm0.48$$ |

A fit of DESI DR2 BAO with a compressed CMB prior that omits CMB lensing, of the kind used in [an earlier note](/quantum-gravity/negative-neutrino-mass/) (section 5 there), with the neutrino mass sum fixed at its minimal normal-ordering value $$0.0588\eV$$, gives $$\delta=(1.15\pm0.47)\%$$. Its $$\chi^2$$ is lower by $$0.14$$ than that of the fit in which $$m_e$$ is fixed and the mass sum is free, which returns a negative sum. The published values are lower; we have not tested which ingredients of the full likelihoods (the full CMB spectra, and in some fits CMB lensing) account for the difference. Below we keep $$\delta$$ as a parameter and evaluate the bounds over this range.

## 2. A scalar field in the plasma

Let $$m_e=m_e(\phi)$$ depend on a homogeneous scalar field with bare potential $$V(\phi)$$. Before stating anything general we look at the simplest case.

<div class="example" markdown="1">
**Example 1.** Take $$m_e=m_0\,e^{\beta\phi/M_P}$$ and $$V=\tfrac12m_\phi^2\phi^2$$, with $$m_\phi$$ large enough that the field stays at the minimum of its effective potential. A gas of non-relativistic electrons of number density $$n$$ adds its rest-mass energy $$n\,m_e(\phi)$$, so the effective potential is $$U(\phi)=\tfrac12m_\phi^2\phi^2+n\,m_0e^{\beta\phi/M_P}$$. At the minimum, $$m_\phi^2\phi=-(\beta/M_P)\,n\,m_e$$, so $$\phi$$ has the sign opposite to $$\beta$$ and

$$
\ln\frac{m_e}{m_0}=\frac{\beta\phi}{M_P}=-\frac{\beta^2\,n\,m_e}{m_\phi^2M_P^2}<0.
$$

The sign of the coupling drops out, because it enters squared. The denser the gas, the lighter the electron.
</div>

The point of the example is that the field moves so as to lower the energy of the matter it is immersed in, and matter of a given composition has less energy when its constituents are lighter. This is Le Chatelier's principle applied to a coupling constant. The general statement needs two ingredients: the derivative of the free energy of a relativistic plasma with respect to a mass, and a comparison argument that does not depend on the shape of $$V$$.

For the first, consider an ideal gas of one species of mass $$M$$ with $$g$$ internal states, in kinetic equilibrium at temperature $$T$$ and chemical potential $$\mu$$, with occupation $$f=(e^{(E-\mu)/T}\pm1)^{-1}$$, $$E=\sqrt{k^2+M^2}$$. Write $$\int\equiv g\int d^3k/(2\pi)^3$$. Its energy density and pressure are $$\rho=\int Ef$$ and $$p=\int\frac{k^2}{3E}f$$. We call

$$
n_s\equiv\int\frac{M}{E}\,f=\frac{\rho-3p}{M}
$$

the scalar density of the species; it is the thermal expectation value of $$\bar\psi\psi$$ for a fermion. The equality follows from $$E-k^2/E=M^2/E$$. The scalar density is positive whenever the species is present, and $$n_s\to n$$ in the non-relativistic limit.

<div class="lemma" markdown="1">
**Lemma 1** (mass derivative of the free energy). Let $$\mathcal F(M;T,n)$$ be the Helmholtz free energy density of the gas at fixed temperature and number density. Then

$$
\frac{\partial\mathcal F}{\partial M}\Big|_{T,n}=n_s=\frac{\rho-3p}{M}>0. \tag{1}
$$

For a mixture of ideal gases whose masses $$M_i$$ depend on $$m_e$$, $$\partial\mathcal F/\partial m_e=\sum_in_{s,i}\,dM_i/dm_e$$, both at fixed composition and at chemical equilibrium.
</div>

<div class="proof" markdown="1">
*Proof.* The grand potential density is $$-p(M;T,\mu)$$ with $$p=\pm gT\int\frac{d^3k}{(2\pi)^3}\ln\big(1\pm e^{-(E-\mu)/T}\big)$$, and differentiating under the integral with $$\partial E/\partial M=M/E$$ gives $$\partial p/\partial M\vert_{T,\mu}=-\int(M/E)f=-n_s$$. Integration by parts in $$k$$ turns the logarithm into $$\int\frac{k^2}{3E}f$$, which is the expression for $$p$$ used above. The free energy is $$\mathcal F=-p+\mu n$$ with $$\mu$$ determined by $$n$$; since $$\partial\mathcal F/\partial\mu\vert_{T,n,M}=-\partial p/\partial\mu+n=0$$, the derivative at fixed $$n$$ equals the derivative of $$-p$$ at fixed $$\mu$$, which is $$n_s$$. For a mixture, the chain rule gives the sum. At chemical equilibrium the composition minimises $$\mathcal F$$ at fixed conserved charges, so by the same envelope argument its variation with $$m_e$$ does not contribute.
</div>

In the cosmological plasma the species that depend on $$m_e$$ are free electrons and positrons, with $$dM/dm_e=1$$, and hydrogen and helium atoms and ions with bound electrons. A bound level of binding energy $$B$$ scales as $$B\propto m_e$$, so its mass $$M=M_{\rm nucleus}+m_e-B$$ has $$dM/dm_e=1-B/m_e$$, which for hydrogen in its ground state is $$1-2.7\times10^{-5}$$. We neglect Coulomb corrections to the ideal-gas free energy; at recombination the plasma is extremely weakly coupled. If nuclei, photons and neutrinos do not depend on $$\phi$$, every term in Lemma 1 is non-negative. We call the sum

$$
\nu_s\equiv\frac{\partial\mathcal F}{\partial m_e}=\sum_in_{s,i}\,\frac{dM_i}{dm_e}
$$

the electron scalar density; it counts free and bound electrons, the latter with weight $$1-B/m_e$$, and is positive whenever electrons are present.

## 3. The sign rule

We now state what "tracking" means and compare two epochs. A state $$s$$ of the plasma is specified by its temperature and its conserved densities (baryon number, charge, lepton number). We write $$\mathcal F_s(\phi)$$ for its free energy density when the masses are evaluated at $$\phi$$, and

$$
U_s(\phi)=V(\phi)+\mathcal F_s(\phi)
$$

for the effective potential of $$\phi$$ in that state. This treats $$\phi$$ as a background parameter of the plasma, as the finite-temperature effective potential does; the derivative of $$U_s$$ reproduces the source term $$-\partial V/\partial\phi-\sum_in_{s,i}\,\partial M_i/\partial\phi$$ in the field equation. Cosmological evolution conserves entropy rather than temperature, but this does not change the stationarity condition: by the same Maxwell relation, $$\partial E/\partial M$$ at fixed entropy and particle number also equals $$n_s$$. Nor does the field's response heat or cool the plasma appreciably, since the photons, which do not couple to $$\phi$$, dominate its heat capacity. The hypotheses are:

- (T1) *Tracking.* In each state $$s$$ that we compare, $$\phi$$ sits at a global minimiser $$\phi_s$$ of $$U_s$$. (Proposition 2 below replaces this by a continuous branch of local minima.)
- (T2) *SM sources only.* Apart from $$V$$, the only $$\phi$$-dependence of $$U_s$$ comes from SM species whose masses are non-decreasing functions of $$m_e$$. A field that changes only $$m_e$$ qualifies. So does a Higgs portal, which changes all masses through the Higgs vacuum value, provided every mass it affects increases with that value.
- (T3) *More matter in the past.* At every $$m$$ between the two values of $$m_e$$ being compared, $$\nu_s=\partial\mathcal F/\partial m_e$$ of the earlier state exceeds that of the later one. For a field that changes only $$m_e$$, $$\nu_s$$ is the electron scalar density; for a Higgs portal it also contains the scalar densities of the nucleons weighted by $$dM_N/dm_e$$.

The comparison itself is two lines.

<div class="theorem" markdown="1">
**Theorem 1** (comparison). Let $$\phi_1$$ and $$\phi_0$$ be global minimisers of $$U_1$$ and $$U_0$$. Then

$$
\mathcal F_1(\phi_1)-\mathcal F_0(\phi_1)\;\le\;\mathcal F_1(\phi_0)-\mathcal F_0(\phi_0). \tag{2}
$$

Under (T2) and (T3), $$m_e(\phi_1)\le m_e(\phi_0)$$.
</div>

<div class="proof" markdown="1">
*Proof.* Minimality gives $$U_1(\phi_1)\le U_1(\phi_0)$$ and $$U_0(\phi_0)\le U_0(\phi_1)$$. Adding the two inequalities cancels $$V$$ and gives (2). Under (T2), $$\mathcal F_1-\mathcal F_0$$ depends on $$\phi$$ only through $$m_e$$; call it $$G(m_e)$$. By Lemma 1, $$G'(m)=\nu^{(1)}_s(m)-\nu^{(0)}_s(m)$$, which is positive by (T3). If $$m_e(\phi_1)>m_e(\phi_0)$$, then $$G(m_e(\phi_1))>G(m_e(\phi_0))$$, contradicting (2).
</div>

Nothing in the proof uses the shape of $$V$$, the size of the coupling or the number of components of $$\phi$$; the argument is the usual one of monotone comparative statics. The physical content is in Lemma 1, which fixes the sign of $$\partial\mathcal F/\partial m_e$$, and in (T3), which says which state has more of the matter that prefers light electrons. Global minimality is a strong reading of "tracking". The same conclusion holds for a field that follows a local minimum continuously.

<div class="proposition" markdown="1">
**Proposition 2** (local tracking). Let $$s\in[0,1]$$ label a continuous sequence of states running from today ($$s=0$$) back to an earlier epoch ($$s=1$$), and let $$\phi(s)$$ be a continuously differentiable branch of non-degenerate local minima, $$U_s'(\phi(s))=0$$ and $$U_s''(\phi(s))>0$$, for a single-component field. Under (T2), if the electron scalar density $$\nu_s$$ at fixed $$m$$ does not decrease with $$s$$, then $$m_e(\phi(s))$$ is non-increasing in $$s$$.
</div>

<div class="proof" markdown="1">
*Proof.* Differentiating $$U_s'(\phi(s))=0$$ along the branch gives $$U_s''\,\phi'(s)+\partial_sU_s'=0$$. By Lemma 1 and (T2), $$U_s'(\phi)=V'(\phi)+\nu_s(m_e(\phi);s)\,m_e'(\phi)$$, so $$\partial_sU_s'=m_e'\,\partial_s\nu_s$$. Hence

$$
\frac{d\,m_e}{ds}=m_e'\,\phi'(s)=-\frac{(m_e')^2\,\partial_s\nu_s}{U_s''}\;\le\;0.
$$

</div>

The branch can end where $$U_s''\to0$$, after which the field jumps and no longer tracks; Theorem 1 covers jumps between global minima.

### Hypothesis (T3) in cosmology

After electron–positron annihilation the net number of electrons is conserved, so their number density scales as $$(1+z)^3$$, and in the non-relativistic regime $$n_s=n\langle m_e/E\rangle\approx n\,(1-3T/2m_e)$$. At recombination $$T/m_e\approx5\times10^{-7}$$, so the correction is negligible against the factor $$(1+z)^3$$ that separates any redshift $$z>0$$ from today. Before and during annihilation the pairs add to $$n_s$$. If $$\nu_s$$ also contains nucleon terms, as for a Higgs portal, these grow into the past in the same way, since nucleons are non-relativistic over the whole range considered and their number density scales as $$(1+z)^3$$. We have checked numerically, for a free electron–positron gas with the observed baryon asymmetry and $$Y_p=0.245$$, that the scalar density at $$m=m_e$$ increases monotonically with temperature from today's $$T_0=2.7255\,$$K up to $$10\,$$MeV. Today's cosmic mean is $$n_e\approx2.2\times10^{-7}\,{\rm cm^{-3}}$$; at last scattering, $$z_\ast\approx1089$$, it is $$n_e(z_\ast)\approx2.2\times10^{-7}\times1090^3\approx290\,{\rm cm^{-3}}$$ (free and bound electrons together).

### Which state is "today"

The theorem compares tracked values in two homogeneous states. The value of $$m_e$$ measured in a laboratory is the tracked value for whatever environment the field responds to there. Theorem 1 gives $$m_e(z_\ast)\le m_e^{\rm lab}$$, that is $$\delta\le0$$, provided that this environment has an electron scalar density below $$n_e(z_\ast)\approx290\,{\rm cm^{-3}}$$. This holds if the field responds to the cosmic mean, or to the mean density of the Galaxy, which near the Sun is of order a few nucleons per $${\rm cm^3}$$ counting stars and gas together. It fails if the field tracks the density of the Earth or of the laboratory itself, as screened fields of chameleon or symmetron type do [[Khoury–Weltman]](#ref-chameleon) [[Hinterbichler–Khoury]](#ref-symmetron). We return to that case in section 5.

<div class="corollary" markdown="1">
**Corollary 3.** Under (T1)–(T3), with the laboratory value set by an environment less dense than the recombination plasma, every field that changes $$m_e$$ gives $$\delta\le0$$. This includes couplings of symmetron or chameleon type to the electron density, with either sign of the coupling, the response of the field to the trace $$\rho-3p$$ of the electron–positron plasma during annihilation, and a Higgs portal under the proviso in (T2).
</div>

The fits of section 1 place $$\delta=0$$ between $$1.1$$ and $$2.4$$ standard deviations below their central values, depending on the data combination.

## 4. Linear couplings and the excursion they need

Fields that do not track a minimum escape Theorem 1. A light field that is frozen at early times and released later can move in either direction. For such fields the obstruction is different: the field must still reach today's value, and if its coupling to electrons is the same along the way, that value is constrained by tests of the equivalence principle.

Write $$\beta_e\equiv M_P\,d\ln m_e/d\phi$$ evaluated today. If $$\phi$$ is light enough to mediate a force over the size of the Earth, $$m_\phi\lesssim10^{-14}\,$$eV, it produces a composition-dependent acceleration. A body $$A$$ of mass $$M_A$$ containing $$N_A$$ electrons has scalar charge $$Q_A=\partial M_A/\partial\phi=\beta_eN_Am_e/M_P$$, if only the electron rest mass depends on $$\phi$$. Scalar exchange between two bodies gives the attractive force $$Q_AQ_B/(4\pi r^2)$$, while gravity gives $$M_AM_B/(8\pi M_P^2r^2)$$. The ratio is $$2\beta_e^2f_Af_B$$, where $$f_A\equiv N_Am_e/M_A\approx(m_e/m_u)(Z/A)_A$$ is the electron mass fraction. The Eötvös parameter for two test masses falling towards the Earth ($$E$$) is therefore

$$
\eta\simeq2\beta_e^2\,f_E\,(f_{\rm Ti}-f_{\rm Pt}). \tag{3}
$$

MICROSCOPE measured $$\eta({\rm Ti,Pt})=[-1.5\pm2.3\,({\rm stat})\pm1.5\,({\rm syst})]\times10^{-15}$$ [[MICROSCOPE]](#ref-microscope). We use $$\lvert\eta\rvert<5.5\times10^{-15}$$, twice the quadrature sum of the errors. This ignores the negative central value, which would tighten the bound, since (3) is positive. With $$m_e/m_u=5.486\times10^{-4}$$, $$(Z/A)_{\rm Ti}=0.460$$, $$(Z/A)_{\rm Pt}=0.400$$ and the bulk-Earth value $$(Z/A)_E\approx0.487$$, (3) gives

$$
\lvert\beta_e\rvert<5.6\times10^{-4}.
$$

Treating the test masses as the titanium and platinum alloys actually flown, rather than as pure elements, loosens the bound by about 3%; varying $$(Z/A)_E$$ between $$0.48$$ and $$0.50$$ changes it by less than 2%. The bound scales as $$\sqrt{\eta_{\max}}$$: at $$1\sigma$$ ($$\lvert\eta\rvert<2.75\times10^{-15}$$) it is $$\lvert\beta_e\rvert<4.0\times10^{-4}$$, and at $$3\sigma$$ it is $$6.9\times10^{-4}$$.

If $$\beta_e$$ does not vary along the field's path, $$\ln(1+\delta)=\beta_e\,\Delta\phi/M_P$$, where $$\Delta\phi=\phi(z_\ast)-\phi(0)$$. More generally, if $$\lvert d\ln m_e/d\phi\rvert$$ never exceeds its present value along the path, $$\lvert\ln(1+\delta)\rvert\le\lvert\beta_e\Delta\phi\rvert/M_P$$. The required excursion is

$$
\frac{\lvert\Delta\phi\rvert}{M_P}\ \ge\ \frac{\ln(1+\delta)}{5.6\times10^{-4}}\approx18\times\frac{\delta}{1\%}.
$$

How far can the field move? Its kinetic energy density is $$\dot\phi^2/2$$, and its share of the total energy density is $$\Omega_{\rm kin}=\dot\phi^2/(6H^2M_P^2)$$. If space is flat or negatively curved and the potential and all other components have non-negative energy density, the Friedmann equation gives $$\Omega_{\rm kin}\le1$$. With $$N=\ln a$$ as time variable, $$d\phi/dN=\dot\phi/H=\pm\sqrt{6\,\Omega_{\rm kin}}\,M_P$$. Integrating from last scattering to today, over $$N_\ast=\ln(1+z_\ast)\approx\ln1090=6.99$$ e-folds, and applying the Cauchy–Schwarz inequality,

$$
\frac{\lvert\Delta\phi\rvert}{M_P}\le\int_0^{N_\ast}\sqrt{6\,\Omega_{\rm kin}}\;dN\le N_\ast\sqrt{6\langle\Omega_{\rm kin}\rangle}=17.1\sqrt{\langle\Omega_{\rm kin}\rangle}, \tag{4}
$$

where $$\langle\Omega_{\rm kin}\rangle$$ is the average of $$\Omega_{\rm kin}$$ over e-folds. Combining (4) with the required excursion gives

$$
\delta\ \le\ \exp\!\big(5.6\times10^{-4}\times17.1\sqrt{\langle\Omega_{\rm kin}\rangle}\big)-1\approx0.96\%\times\sqrt{\langle\Omega_{\rm kin}\rangle}.
$$

For the values of section 1:

| $$\delta$$ (%) | required $$\lvert\Delta\phi\rvert/M_P$$ | required $$\langle\Omega_{\rm kin}\rangle$$ |
|---|---|---|
| 1.15 | 20.4 | 1.42 |
| 0.96 | 17.1 | 0.99 |
| 0.78 | 13.9 | 0.66 |
| 0.76 | 13.5 | 0.62 |
| 0.63 | 11.2 | 0.43 |

At the $$2\sigma$$ MICROSCOPE level, values of $$\delta$$ above $$0.96\%$$ are excluded outright for this class. Since the ceiling on $$\delta$$ scales as $$\sqrt{\eta_{\max}}$$, it is $$0.68\%$$ at $$1\sigma$$ and $$1.18\%$$ at $$3\sigma$$; the table uses the $$2\sigma$$ bound. The lower central values are not excluded by (4) alone, but require the field's kinetic energy to carry, on average over the e-folds since recombination, between 40% and 70% of the total energy density, during an interval that the CMB and BAO describe as dominated by matter. A heuristic estimate makes this concrete. Suppose the expansion history is that of $$\Lambda$$CDM with $$\Omega_m=0.3$$, and the kinetic energy of the field is confined to the share that $$\Lambda$$ occupies, $$\Omega_{\rm kin}\le\Omega_\Lambda(z)$$. Then the middle expression in (4) gives $$\lvert\Delta\phi\rvert\le1.97\,M_P$$ and $$\delta\le0.11\%$$.

The excursion bound does not use tracking, and so it also applies to the frozen-then-released fields that Theorem 1 does not cover, as long as their coupling is linear and their mass today is small enough for the MICROSCOPE bound to apply.

## 5. What remains

Each route to $$\delta>0$$ that survives sections 3 and 4 gives up one of the hypotheses.

### Fields that do not track and have no linear coupling today

If $$m_e=m_0(1+\phi^2/2M^2)$$ with $$\phi=0$$ today, then $$\beta_e=0$$ and the bound (3) is empty. An SM source cannot displace $$\phi$$ from zero, since any displacement raises $$m_e$$, which Theorem 1 forbids. A positive $$\delta$$ then needs a field that is held away from zero by Hubble friction and later oscillates about it, with $$\delta=\phi_i^2/2M^2$$ set by its initial value $$\phi_i$$. This class has been studied in detail [[BSW 2024]](#ref-bsw1) [[BSW 2025]](#ref-bsw2) [[GBY]](#ref-gby). The oscillating field contributes a matter-like energy density at late times, which BSW identify as a "late dark matter" degeneracy in the cosmological fits. Its residual oscillation today modulates $$m_e$$ in time, which atomic clocks constrain. We do not analyse these constraints here.

### Fields also sourced by dark matter

If $$\phi$$ also changes the mass $$m_X$$ of a non-relativistic dark-matter particle, (T2) fails, but under (T1) the inequality (2) still holds. Take both electrons and dark matter non-relativistic and neglect thermal corrections, so that $$\mathcal F_s(\phi)=n_{e,s}\,m_e(\phi)+n_{X,s}\,m_X(\phi)$$ up to $$\phi$$-independent terms, with both densities scaling as $$(1+z)^3$$. Then (2) reads $$\big[(1+z)^3-1\big]\big[n_{e,0}\,\Delta m_e+n_{X,0}\,\Delta m_X\big]\le0$$, where $$\Delta$$ denotes the value at redshift $$z$$ minus that today, so that

$$
-\frac{\Delta m_X}{m_X}\ \ge\ \delta\,\frac{\rho_{e,0}}{\rho_{X,0}},
$$

with $$\rho_{e,0}$$ and $$\rho_{X,0}$$ the present rest-mass densities. For $$\omega_b=0.0224$$, $$\omega_c=0.120$$ and $$Y_p=0.245$$, $$\rho_{e,0}/\rho_{X,0}=8.9\times10^{-5}$$. A heavier electron at recombination requires the dark matter to have been lighter then by a fraction of at least $$8.9\times10^{-5}\,\delta$$, which is $$1.0\times10^{-6}$$ at $$\delta=1.15\%$$. The dark matter must supply the source that the plasma cannot. A field of this kind that saturates at high density produces a step in $$m_e$$ at the redshift where the dark-matter density crosses a threshold. Whether such a model fits the full CMB, and what it implies for nucleosynthesis, where $$m_e$$ would be shifted by the same amount, is not examined here.

### Fields that track the local density today

If the field responds to the density of the Earth or of the laboratory, Theorem 1 compares the recombination plasma with that environment, which is far denser, and allows $$m_e(z_\ast)>m_e^{\rm lab}$$. Applied to pairs of environments of similar composition, the same theorem orders them by electron scalar density: $$m_e$$ in any environment less dense than the recombination plasma, $$n_e\lesssim290\,{\rm cm^{-3}}$$, is at least $$m_e(z_\ast)$$, and so at least $$(1+\delta)\,m_e^{\rm lab}$$. Spectroscopy of absorbing gas measures the proton-to-electron mass ratio $$m_p/m_e$$ in environments other than the laboratory. Methanol absorption in the lens towards PKS1830−211 at $$z=0.886$$ limits its fractional change to $$4\times10^{-7}$$ at $$2\sigma$$ [[Kanekar et al.]](#ref-kanekar), and ammonia and cyanopolyyne lines in Galactic dark clouds limit it to $$2\times10^{-8}$$ at $$3\sigma$$ [[Levshakov et al.]](#ref-levshakov). Both are far below $$\delta\approx10^{-2}$$, so they bear on this route if the absorbing gas has an electron density below about $$290\,{\rm cm^{-3}}$$, counting electrons bound in molecules, on the scale to which the field responds. Molecular clouds are typically denser than this, in which case the theorem only gives $$m_e\le m_e(z_\ast)$$ there and these limits do not apply directly. We have not settled which case holds for these absorbers, nor checked whether a field with this behaviour can satisfy fifth-force and equivalence-principle bounds.

## 6. Limitations and open questions

Theorem 1 is exact for ideal-gas thermodynamics and global tracking, and Proposition 2 for a continuous branch of local minima. Both treat $$\phi$$ as homogeneous and neglect its own gradient and kinetic energy in the definition of tracking; a field that lags behind its minimum is between the tracking and frozen cases and is not covered by either section 3 or section 4 unless its coupling is linear. Hypothesis (T2) excludes fields that also change the fine-structure constant or nucleon masses in a direction opposite to $$m_e$$; for those, (2) still holds but does not fix the sign of $$\delta$$.

The excursion bound rests on the MICROSCOPE result, on an electron-only coupling (a coupling to nucleons as well changes (3) and is not considered), and on a coupling that does not grow along the path. A path along which $$d\ln m_e/d\phi$$ grows from a small value today is the quadratic class of section 5 with a small displacement of today's minimum from the symmetric point; we have not scanned this intermediate case. The heuristic $$\delta\le0.11\%$$ assumes a $$\Lambda$$CDM expansion history and is not a bound derived from data.

The target values of $$\delta$$ come from fits that vary $$m_e$$ by hand at recombination, with no field dynamics. A field model also changes $$m_e$$ at other epochs, in particular during nucleosynthesis, and adds its own energy density; the data's preference for $$\delta>0$$ has not been recomputed for any of the field models of section 5.

## References

1. <a id="ref-act"></a>Calabrese et al. (ACT Collaboration), “The Atacama Cosmology Telescope: DR6 constraints on extended cosmological models,” [arXiv:2503.14454](https://arxiv.org/abs/2503.14454).
2. <a id="ref-toda-seto"></a>Toda, Seto, “Constraints on the varying electron mass and early dark energy in light of ACT DR6 and DESI DR2 and the implications for inflation,” [arXiv:2508.09025](https://arxiv.org/abs/2508.09025).
3. <a id="ref-weiner"></a>Weiner, “High-redshift physics from the acoustic scale,” [arXiv:2603.18131](https://arxiv.org/abs/2603.18131).
4. <a id="ref-bsw1"></a>Baryakhtar, Simon, Weiner, “Cosmology with varying fundamental constants from hyperlight, coupled scalars,” [arXiv:2405.10358](https://arxiv.org/abs/2405.10358).
5. <a id="ref-bsw2"></a>Baryakhtar, Simon, Weiner, “Searching for coupled, hyperlight scalars across cosmic history,” [arXiv:2502.04432](https://arxiv.org/abs/2502.04432).
6. <a id="ref-gby"></a>Ghosh, Boddy, Yu, “Early Universe constraints on variations in fundamental constants induced by ultralight scalar dark matter,” [arXiv:2511.14532](https://arxiv.org/abs/2511.14532).
7. <a id="ref-smith"></a>Smith, Mylova, van de Bruck, Burgess, Di Valentino, “The serendipitous axiodilaton: a self-consistent recombination-era solution to the Hubble tension,” [arXiv:2512.13544](https://arxiv.org/abs/2512.13544).
8. <a id="ref-microscope"></a>Touboul et al. (MICROSCOPE Collaboration), “MICROSCOPE mission: final results of the test of the Equivalence Principle,” [arXiv:2209.15487](https://arxiv.org/abs/2209.15487).
9. <a id="ref-chameleon"></a>Khoury, Weltman, “Chameleon fields: awaiting surprises for tests of gravity in space,” [arXiv:astro-ph/0309300](https://arxiv.org/abs/astro-ph/0309300).
10. <a id="ref-symmetron"></a>Hinterbichler, Khoury, “Symmetron fields: screening long-range forces through local symmetry restoration,” [arXiv:1001.4525](https://arxiv.org/abs/1001.4525).
11. <a id="ref-kanekar"></a>Kanekar et al., “Constraints on changes in the proton-electron mass ratio using methanol lines,” [arXiv:1412.7757](https://arxiv.org/abs/1412.7757).
12. <a id="ref-levshakov"></a>Levshakov, Reimers, Henkel, Winkel, Mignano, Centurion, Molaro, “Limits on the spatial variations of the electron-to-proton mass ratio in the Galactic plane,” [arXiv:1310.1850](https://arxiv.org/abs/1310.1850).
