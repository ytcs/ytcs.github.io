---
title: What a negative neutrino mass in BAO data would require
date: 2026-10-29
category: dark-sector
order: 4
description: In a flat universe with stable cold dark matter, no components that obey the null energy condition reproduce the exact expansion history of a smaller neutrino mass sum, and in a compressed likelihood built from DESI DR2 BAO data such components with constant density above redshift 300 leave the inferred sum at or above 0.0638 eV to first order in their amplitude.
status: Theorems 1 and 2, Proposition 1 and Corollary 1 are proved. The floor of section 5 and the curvature slope of section 6 are numerical results in a compressed likelihood, first order in the amplitude of the extra components, with a limited set of nonlinear cases.
excludes: Exact reproduction of the expansion history of a smaller neutrino mass sum on an interval of redshift, in a flat universe whose extra components respect the null energy condition and leave the stable cold dark matter density unchanged. In the numerical set-up of section 5 only, extra components of constant density above redshift 300, at first order in their amplitude.
---

Fits of flat $$\Lambda$$CDM with a neutrino mass sum $$\Seff$$ that is allowed to be negative return $$\Seff=-0.101^{+0.047}_{-0.056}\eV$$ when DESI DR2 baryon acoustic oscillation (BAO) data are combined with the cosmic microwave background (CMB) [[DESI ν]](#ref-desi-nu), below the $$0.0588\eV$$ that oscillation experiments require. We ask what the dark sector would have to do for a fit to return less than the true mass sum $$\Strue$$. Take a flat universe in which the baryon and cold dark matter (CDM) densities of the fit equal those of the truth and the CDM is stable. If the expansion history of the fit with $$\Seff<\Strue$$ is matched exactly on an interval of redshift, the other components of the truth must have a negative total $$\rho+p$$ there, which violates the null energy condition (NEC) (Theorem 1). If the CDM density of the fit exceeds the stable one by $$\delta$$, an NEC-respecting match exists if and only if $$\delta\ge\delta_{\min}$$ (Theorem 2). For the values of $$\Seff$$ tabulated in section 4, $$\delta_{\min}$$ is within 4% of $$(\Strue-\Seff)/93\eV$$, which is $$1.7\times10^{-3}$$ in the physical CDM density $$\omega_c\approx0.12$$ at $$\Seff=-0.10\eV$$. A fluid that is dust at early times and radiation at late times attains this match exactly (Proposition 1).

The BAO data fix thirteen distances, not the function $$H(z)$$. In a compressed likelihood built from the DR2 values, and to first order in their amplitude, extra NEC components whose density is constant above $$z=300$$ leave the BAO-inferred mass sum at or above $$0.0638\eV$$, the value that $$\Lambda$$CDM itself returns. An NEC component whose density increases with redshift between $$z=300$$ and recombination can lower it, and open curvature lowers it by about $$66\eV$$ per unit of $$\Omega_K$$ (sections 5 and 6).

Graham, Green and Meyers [[GGM]](#ref-ggm) separate the mass sum inferred from the expansion history from the one inferred from clustering, find that non-phantom dynamical dark energy pushes the first towards negative values, and note that spatial curvature is degenerate with it. Lynch and Knox [[LK]](#ref-lk) show that dark matter decaying exponentially to dark radiation, at a rate of order the inverse age when a neutrino becomes non-relativistic, is nearly degenerate with a signed mass sum in the background. Elbers et al. [[Elbers]](#ref-elbers) introduced the effective mass parameter that extends to negative values and that DESI uses, and Green and Meyers [[GM]](#ref-gm) trace the preference for negative values in the full data to CMB lensing. Added here are the exact comparison for an arbitrary true mass spectrum (Theorem 1), the sharp criterion $$\delta_{\min}$$ with a fluid that attains it and satisfies every energy condition (Theorem 2, Proposition 1), and, in a fully specified set-up, the floor with the redshifts at which it holds and the curvature slope. Theorem 1 compares the $$\rho+p$$ of two expansion histories in the way the [sign theorem for vacuum-matter exchange](/quantum-gravity/vacuum-matter-sign/) compares two minimisers: in both, a monotone dependence on one parameter fixes a sign.

## 1. The expansion history and what BAO measure

We work in a spatially flat Friedmann–Robertson–Walker universe with metric $$ds^2=-dt^2+a(t)^2d\mathbf x^2$$. The redshift is $$1+z=1/a$$ and the Hubble rate is $$H=\dot a/a$$. Each component $$i$$ has an energy density $$\rho_i$$ and a pressure $$p_i$$, and the Friedmann equation $$H^2=\tfrac{8\pi G}{3}\sum_i\rho_i$$ holds. Total energy is conserved, $$\dot\rho_{\rm tot}=-3H(\rho_{\rm tot}+p_{\rm tot})$$. A component that exchanges no energy with the others is separately conserved, and since $$d/dt=-(1+z)H\,d/dz$$ it obeys

$$
\frac{d\rho_i}{dz}=\frac{3(\rho_i+p_i)}{1+z}. \tag{1}
$$

Pressureless matter ("dust", $$p=0$$) has $$\rho\propto(1+z)^3$$, radiation ($$p=\rho/3$$) has $$\rho\propto(1+z)^4$$, and a cosmological constant ($$p=-\rho$$) has constant $$\rho$$. The NEC for a component is $$\rho_i+p_i\ge0$$. By (1) a separately conserved component obeys the NEC if and only if its density does not decrease with increasing redshift. A component with $$\rho+p<0$$ is phantom: its density grows as the universe expands.

We measure densities in units of $$3(100\,{\rm km\,s^{-1}\,Mpc^{-1}})^2/8\pi G$$, so that $$H^2=\sum_i\rho_i$$ with $$H$$ in units of $$100\,{\rm km\,s^{-1}\,Mpc^{-1}}$$, and the present density of a dust component is $$\omega_i=\Omega_ih^2$$, where $$H_0=100h\,{\rm km\,s^{-1}\,Mpc^{-1}}$$. Differentiating $$H^2=\sum_i\rho_i$$ and using conservation of the total gives

$$
(1+z)\,\frac{d(H^2)}{dz}=3\sum_i(\rho_i+p_i). \tag{2}
$$

Two flat models with the same $$H(z)$$ on an interval therefore have the same total energy density there and, by (2), the same total $$\rho+p$$, whatever they are made of. Only the split into components can differ. The sign of the fixed total $$\rho+p$$ is what the rest of the page examines.

BAO are the imprint, in the clustering of galaxies, of sound waves that propagated in the photon-baryon plasma before the baryons decoupled from the photons at the drag redshift $$\zd\approx1060$$ [[Planck 2018]](#ref-planck18). The comoving distance the waves travelled is the sound horizon $$r_d=\int_{\zd}^\infty c_s\,dz/H(z)$$ with sound speed $$c_s$$. The angular size of the feature at redshift $$z$$ gives the transverse comoving distance $$D_M(z)=c\int_0^zdz'/H(z')$$ in units of $$r_d$$, and its extent in redshift along the line of sight gives $$D_H(z)=c/H(z)$$ in units of $$r_d$$. Where the signal is weaker only the combination $$D_V=(zD_M^2D_H)^{1/3}$$ is constrained. DESI DR2 reports $$D_V/r_d$$ at $$z=0.295$$, and $$D_M/r_d$$ and $$D_H/r_d$$ with a correlation coefficient at each of six redshifts from $$0.510$$ to $$2.330$$, thirteen numbers in all [[DESI DR2]](#ref-desi-dr2) (their Table IV).

The CMB fixes the early universe. It constrains the physical baryon density $$\omega_b$$, the density $$\omega_{cb}=\omega_b+\omega_c$$ of baryons and CDM, and the angular scale $$\theta_\ast=r_\ast/D_M(z_\ast)$$ of the sound horizon at recombination, $$z_\ast\approx1090$$. Given $$\omega_b$$ and $$\omega_{cb}$$, the sound horizon $$r_d$$ follows from the physics before $$\zd$$, and the BAO distances probe $$H(z)$$ at $$z\lesssim2.5$$. The fit that returns a mass sum $$\Seff$$ is flat $$\Lambda$$CDM with free $$(\omega_b,\omega_c,h,\Seff)$$, three neutrino species of equal mass $$\Seff/3$$, and $$\Lambda$$ fixed by flatness. Neutrinos with mass much larger than their temperature behave as dust, so at fixed early universe a larger $$\Seff$$ means more matter at late times and a different $$H(z)$$. This is the dependence through which BAO measure the mass sum.

## 2. Neutrinos as a source of $$\rho+p$$

After decoupling, a neutrino species keeps a Fermi–Dirac distribution in comoving momentum, so its effective temperature falls as $$T(z)=T_{\nu0}(1+z)$$. We take $$T_{\nu0}=0.1683\meV$$, the instantaneous-decoupling value $$(4/11)^{1/3}T_{\gamma0}=0.1676\meV$$ rescaled by $$(N_{\rm eff}/3)^{1/4}$$ with $$N_{\rm eff}=3.044$$, so that three massless species have the standard radiation density; one massless species then has $$\rho=5.70\times10^{-6}$$ in the units of section 1 today. For one species of mass $$m\ge0$$, counting particle and antiparticle, let $$E=\sqrt{q^2+m^2}$$, $$f(q)=(e^{q/T}+1)^{-1}$$ and $$\int\equiv\pi^{-2}\int_0^\infty q^2\,dq$$. Then

$$
\rho(m)=\int E\,f,\qquad p(m)=\int\frac{q^2}{3E}\,f,\qquad j(m)\equiv\rho(m)+p(m)=\int\Big(E+\frac{q^2}{3E}\Big)f. \tag{3}
$$

Each species is separately conserved, so by (1) $$d\rho/dz=3j/(1+z)$$. For $$m\ll T$$ the species is radiation and $$j=\tfrac43\rho(0)$$. For $$m\gg T$$ it is dust with $$\rho\simeq m\,n_1(z)$$, where $$n_1=n_1(0)(1+z)^3$$ is the number density of one species, and $$j\simeq m\,n_1$$. With the temperature above, $$n_1(0)=0.010747\eV^{-1}$$ in our units, so that three species of total mass $$\Sigma$$ contribute $$\omega_\nu=\Sigma\,n_1(0)=\Sigma/93.05\eV$$ at late times. (The convention $$\Sigma/93.14\eV$$ is also in use; the 0.1% difference is immaterial here.)

A fit can return a negative mass sum only if the neutrino model is continued below zero. The continuation of [[Elbers]](#ref-elbers), implemented in the DESI analysis [[DESI ν]](#ref-desi-nu) (their appendix A), reverses the sign of the energy of the massive species and adds massless species so that the early expansion is unchanged. For the background this amounts to

$$
\rho(m)\equiv2\rho(0)-\rho(\lvert m\rvert),\quad p(m)\equiv2p(0)-p(\lvert m\rvert),\quad j(m)=2j(0)-j(\lvert m\rvert)\qquad(m<0). \tag{4}
$$

A negative mass removes from the late-time energy as much, relative to massless neutrinos, as the positive mass of the same modulus adds, and for $$T\gg\lvert m\rvert$$ all expressions reduce to the massless values. We write $$j(m)$$ for both signs.

<div class="lemma" markdown="1">
**Lemma 1.** At every $$T>0$$, $$j$$ is strictly increasing and strictly convex on $$m\ge0$$, and strictly increasing on $$m\le0$$ (hence on $$\mathbb R$$).
</div>

<div class="proof" markdown="1">
*Proof.* Differentiation under the integral sign is justified by the exponential decay of $$f$$. Since $$\partial_mE=m/E$$,

$$
\partial_mj=\int\frac mE\Big(1-\frac{q^2}{3E^2}\Big)f>0\quad(m>0),
$$

because $$q^2<E^2$$. With $$\partial_m(m/E)=q^2/E^3$$ and $$\partial_m(q^2m/3E^3)=q^2/3E^3-q^2m^2/E^5$$,

$$
\partial_m^2j=\int\Big(\frac23\frac{q^2}{E^3}+\frac{q^2m^2}{E^5}\Big)f=\int\frac{q^2(2q^2+5m^2)}{3E^5}\,f>0.
$$

At $$m=0$$ the integrand is $$\tfrac23q^{-1}f$$ against the measure $$q^2dq$$, which is integrable, so the second derivative is positive there too. For $$m<0$$, (4) gives $$j'(m)=j'(\lvert m\rvert)>0$$, and $$j'(0)=0$$ from both sides. $$\square$$
</div>

Monotonicity says that a smaller mass sum carries less $$\rho+p$$ at every temperature. Convexity, through Jensen's inequality, says that a hierarchical spectrum carries more than the degenerate spectrum of the same sum.

## 3. Exact mimicry forces a violation of the NEC

The true neutrino masses used for all numbers on this page are the minimal normal ordering with lightest mass zero, $$m=(0,\,8.655,\,50.13)\meV$$ and $$\Strue=58.78\meV$$. They follow from the squared-mass splittings $$s_{21}=7.49\times10^{-5}\eV^2$$ and $$s_{31}=2.513\times10^{-3}\eV^2$$ of the NuFit 6.0 best fit that includes Super-Kamiokande atmospheric data [[NuFit]](#ref-nufit); the value $$s_{31}=2.534\times10^{-3}\eV^2$$ of the fit without those data changes $$\Strue$$ by $$0.2\meV$$. Other spectra are named where they are used.

The statement compares the true universe with the fit, under these hypotheses.

(H1) Both are spatially flat Friedmann–Robertson–Walker universes with conserved total energy.

(H2) Photons, baryons and CDM have the same present densities $$\omega_\gamma,\omega_b,\omega_c$$ in the truth and in the fit, and the same $$N_{\rm eff}$$. Baryons and CDM are stable and separately conserved: $$\rho_b=\omega_b(1+z)^3$$ and $$\rho_c=\omega_c(1+z)^3$$ in both.

(H3) The true neutrinos are three standard Fermi–Dirac species with masses $$m_i\ge0$$ and sum $$\Strue$$.

(H4) The fit is flat $$\Lambda$$CDM with three neutrino species of equal mass $$\Seff/3$$, continued by (4) when $$\Seff<0$$, and a cosmological constant.

(H5) The true model may contain further components $$X_k$$, with arbitrary pressures and densities (negative densities are allowed), not necessarily separately conserved. Only the conservation of the total energy is used.

The quantity that decides the sign is the $$\rho+p$$ of the true neutrinos minus that of the fitted ones,

$$
\Delta(z)\equiv\sum_{i=1}^3j\big(m_i;T(z)\big)-3\,j\big(\Seff/3;T(z)\big).
$$

<div class="lemma" markdown="1">
**Lemma 2.** Under (H1)–(H5), if the Hubble rate of the truth equals that of the fit on an open interval $$I$$ of redshift, then $$\sum_k(\rho_k+p_k)(z)=-\Delta(z)$$ for every $$z\in I$$.
</div>

<div class="proof" markdown="1">
*Proof.* Equal $$H$$ on $$I$$ gives equal $$d(H^2)/dz$$ there, and by (2) equal total $$\rho+p$$. The fit contains photons, baryons, CDM, its neutrinos and $$\Lambda$$, for which $$\rho+p=0$$. The truth contains the same photons, baryons and CDM by (H2), its own neutrinos, and the $$X_k$$. The common terms cancel. $$\square$$
</div>

<div class="theorem" markdown="1">
**Theorem 1** (exact mimicry violates the NEC). Assume (H1)–(H5) and $$\Seff<\Strue$$. If the Hubble rate of the truth equals that of the fit on an open interval $$I$$, then $$\Delta(z)>0$$ and therefore $$\sum_k(\rho_k+p_k)(z)<0$$ at every $$z\in I$$. In particular some component violates the NEC at every redshift of $$I$$.
</div>

The true neutrinos, with more mass, carry more $$\rho+p$$ than the fitted ones at every temperature, and the remaining components must carry the opposite sign. The comparison needs monotonicity of $$j$$ and Jensen's inequality for it, both from Lemma 1. No assumption on the spectrum, on the redshift or on the conservation of the $$X_k$$ enters.

<div class="proof" markdown="1">
*Proof.* By Lemma 2 it suffices to show $$\Delta(z)>0$$ for each $$z\in I$$. Fix $$z$$, hence $$T$$. If $$0\le\Seff<\Strue$$, monotonicity gives $$3j(\Seff/3)<3j(\Strue/3)$$, and convexity with Jensen's inequality, applied to $$m_i\ge0$$ with mean $$\Strue/3$$, gives $$3j(\Strue/3)\le\sum_ij(m_i)$$. If $$\Seff<0$$, monotonicity of $$j$$ on $$\mathbb R$$ gives $$3j(\Seff/3)<3j(0)$$, and $$3j(0)\le\sum_ij(m_i)$$ since $$m_i\ge0$$. In both cases $$\Delta(z)>0$$. $$\square$$
</div>

The $$X_k$$ need not be separately conserved, and the redshift enters only through $$T(z)$$, so the statement holds on any interval, including $$z>\zd$$.

<div class="remark" markdown="1">
**Remark 1** (conserved extra components). If the $$X_k$$ are separately conserved, (1) shows that their total density $$\rho_X$$ obeys $$d\rho_X/dz=-3\Delta/(1+z)<0$$: it grows as the universe expands. At low redshift, where the neutrinos are non-relativistic, $$\rho_X\simeq C-(\Strue-\Seff)\,n_1(z)$$ for a constant $$C$$, a cosmological constant minus the missing neutrino rest mass.
</div>

<div class="remark" markdown="1">
**Remark 2** (other fit spectra). Let the fit have non-negative masses $$m'_1,m'_2,m'_3$$, not necessarily equal, with the sum of the $$k$$ largest never exceeding the sum of the $$k$$ largest true masses for $$k=1,2,3$$ (weak majorisation) and with $$\sum_im'_i<\Strue$$. Because $$j$$ is increasing and convex on $$m\ge0$$, the Tomić–Weyl inequality gives $$\sum_ij(m'_i)\le\sum_ij(m_i)$$, strictly since the sums differ, and the proof of Theorem 1 goes through with this inequality in place of Jensen's.
</div>

<div class="remark" markdown="1">
**Remark 3** (a fit with the mass in one species). The degenerate fit of (H4) is a convention. A fit that puts all of $$\Seff\ge0$$ in one species and leaves two massless, as some analyses do, is not weakly majorised by the truth once $$\Seff$$ exceeds the largest true mass, and for $$\Seff$$ close to $$\Strue$$ the sign of $$\Delta$$ reverses. At $$z=0$$ the truth has two non-relativistic species and one massless, the fit one non-relativistic and two massless; a massless species has $$j(0)=\tfrac43\rho(0)=4.20\,n_1T_{\nu0}$$, whereas a non-relativistic one has $$j\simeq m\,n_1$$, so $$\Delta(0)\simeq(\Strue-\Seff-4.20\,T_{\nu0})\,n_1$$ and $$\Delta(0)<0$$ when the gap is below $$4.20\,T_{\nu0}=0.71\meV$$. A numerical evaluation for the normal ordering above shows that $$\max_z\Delta<0$$, attained at $$z=0$$, if and only if $$\Strue-\Seff<0.671\meV$$; the difference from $$0.71\meV$$ is the thermal energy of the $$8.7\meV$$ species. For such gaps the fluid of the proof of Theorem 2 below, with $$\delta=0$$, is an NEC-respecting exact mimic. The conclusion of Theorem 1 therefore depends on the fit convention, at a level far below any observational resolution.
</div>

<div class="remark" markdown="1">
**Remark 4** (the case $$\Seff=\Strue$$). For a degenerate truth and $$\Seff=\Strue$$ one has $$\Delta\equiv0$$. For the hierarchical truth above, $$\Delta(z)>0$$ even at $$\Seff=\Strue$$, since Jensen's inequality is strict. In units of $$n_1(z)$$, that is, as a mass sum, the gap is $$0.70\meV$$ at $$z=0$$, $$0.91\meV$$ at $$0.3$$, $$1.4\meV$$ at $$1$$, $$4.0\meV$$ at $$5$$ and $$6.5\meV$$ at $$10$$. This growth with redshift underlies the offset of $$5\meV$$ that the fit returns for a normal-ordering truth in section 5.
</div>

<div class="example" markdown="1">
**Example 1** (size of the violation). At $$z=0$$ and for $$\Seff=0$$ we find $$\Delta=6.17\times10^{-4}$$, so by Lemma 2 the extra components must have $$\sum_k(\rho_k+p_k)=-6.17\times10^{-4}$$ in our units. For $$h=0.68$$ this is $$-1.3\times10^{-3}$$ of the critical density today. If the extra component is the dark energy, with a density of $$0.69$$ of critical, this is $$1+w=-1.9\times10^{-3}$$. For $$\Seff=-0.10\eV$$ the same computation gives $$\Delta=1.67\times10^{-3}$$, $$-3.6\times10^{-3}$$ of critical and $$1+w=-5.2\times10^{-3}$$. The required departures from $$w=-1$$ are small, and they must be present throughout the matched interval: for $$\Seff=-0.10\eV$$, $$\Delta(z)$$ grows to $$1.3\times10^{-2}$$ at $$z=1$$ and to $$1.8$$ at $$z=10$$.
</div>

## 4. When the fit has more cold matter than the truth

Theorem 1 uses (H2) in a strong form: the CDM density of the fit is the stable CDM density of the truth. The CMB constrains the density of cold matter at recombination, and if part of that matter later ceases to be matter, the fit's $$\omega_c$$ exceeds the density of the stable CDM. Let the fit have CDM density $$\omega_c+\delta$$ while the truth keeps $$\omega_c$$, with the photon and baryon densities and $$N_{\rm eff}$$ of (H2) unchanged, and otherwise assume (H1), (H3), (H4), (H5). The excess $$\delta$$ is a present-day dust density in the units of section 1, so that a mass sum $$\Sigma$$ of non-relativistic neutrinos is equivalent to $$\delta=\Sigma\,n_1(0)$$. The proof of Lemma 2 now has an extra dust term in the fit, $$\delta(1+z)^3$$, with $$\rho+p=\delta(1+z)^3$$, and gives

$$
\sum_k(\rho_k+p_k)(z)=\delta(1+z)^3-\Delta(z)\qquad(z\in I). \tag{5}
$$

<div class="theorem" markdown="1">
**Theorem 2** (criterion for an NEC-respecting mimic). Under the hypotheses just stated, there exist components $$X_k$$, each obeying $$\rho_k+p_k\ge0$$, such that the Hubble rate of the truth equals that of the fit on $$I$$, if and only if

$$
\delta\;\ge\;\delta_{\min}\equiv\sup_{z\in I}\frac{\Delta(z)}{(1+z)^3}. \tag{6}
$$

When (6) holds, a single separately conserved fluid does the job.
</div>

<div class="proof" markdown="1">
*Proof.* If each $$X_k$$ obeys the NEC, the left side of (5) is non-negative, which is (6). Conversely, assume (6) and take one fluid with $$\rho_X=\rho_\Lambda+\delta(1+z)^3+\rho_\nu^{\rm fit}-\rho_\nu^{\rm true}$$, where $$\rho_\Lambda$$ is the cosmological constant of the fit and $$\rho_\nu^{\rm fit,true}$$ are the densities of the three neutrino species of the fit and of the truth. Then the total density of the truth equals that of the fit at every redshift. The neutrino species are separately conserved, so by (1) the fluid has $$(1+z)\rho_X'/3=\delta(1+z)^3+3j(\Seff/3)-\sum_ij(m_i)=\delta(1+z)^3-\Delta$$, which is non-negative by (6), and the pressure $$p_X$$ is fixed by $$\rho_X$$ through (1). $$\square$$
</div>

The supremum can be evaluated. For the values in the table it is the limit $$z\to0$$, so that $$\delta_{\min}=\Delta(0)$$, which is $$(\Strue-\Seff)\,n_1(0)$$ up to the thermal energy of the neutrinos that are still mildly relativistic today. The interval is $$I=(0,1060)$$; the value does not change when the upper limit is moved between $$1000$$ and $$1090$$.

| $$\Seff$$ (eV) | 0 | −0.05 | −0.10 | −0.20 |
|---|---|---|---|---|
| $$\delta_{\min}$$ | $$6.17\times10^{-4}$$ | $$1.132\times10^{-3}$$ | $$1.669\times10^{-3}$$ | $$2.744\times10^{-3}$$ |
| $$(\Strue-\Seff)/93.05\eV$$ | $$6.32\times10^{-4}$$ | $$1.169\times10^{-3}$$ | $$1.707\times10^{-3}$$ | $$2.781\times10^{-3}$$ |

For $$\Seff=-0.10\eV$$ the excess $$\delta_{\min}=1.67\times10^{-3}$$ is 1.4 times the uncertainty $$0.0012$$ of $$\omega_c$$ in the Planck analysis of the temperature, polarisation and lensing spectra [[Planck 2018]](#ref-planck18). Such an excess is matter that is present at recombination and ceases to be matter later.

An explicit fluid has this property exactly. For $$\mu>0$$ let

$$
X_\mu(z)\equiv\rho(0;z)+\mu\,n_1(z)-\rho(\mu;z), \tag{7}
$$

the density of a massless neutrino species plus the rest-mass density of a species of mass $$\mu$$, minus the density of a neutrino of mass $$\mu$$, each at temperature $$T(z)$$; and $$X_0\equiv0$$.

<div class="proposition" markdown="1">
**Proposition 1** (the fluid $$X_\mu$$). Treated as a separately conserved perfect fluid, $$X_\mu$$ has non-negative density that does not decrease with redshift, and it obeys the null, weak and dominant energy conditions with $$0\le w\le\tfrac13$$. It is dust for $$T\gg\mu$$ ($$X_\mu\simeq\mu n_1$$) and radiation for $$T\ll\mu$$ ($$X_\mu\to\rho(0)$$).
</div>

While the neutrino is relativistic, its energy is close to that of a massless species, so $$X_\mu$$ reduces to the rest-mass term $$\mu n_1$$, which redshifts as dust. Once the neutrino has cooled, it carries $$\rho(\mu)\simeq\mu n_1$$ and $$X_\mu$$ is the energy of the massless species, which redshifts as radiation. The proof is a pointwise inequality in the integrand.

<div class="proof" markdown="1">
*Proof.* Each term in (7) is separately conserved, so $$p_X=p(0)-p(\mu)$$ and the dust term has no pressure. In comoving momentum $$q$$ the distribution is $$f=(e^{q/T_{\nu0}}+1)^{-1}$$, independent of $$z$$. With $$a=1/(1+z)$$ and $$x=\mu a$$,

$$
X_\mu=a^{-4}g(x),\qquad g(x)=\frac1{\pi^2}\int_0^\infty f\,q^2\big[q+x-\sqrt{q^2+x^2}\big]dq\ge0,
$$

because $$\sqrt{q^2+x^2}\le q+x$$. Monotonicity in $$z$$ means $$dX_\mu/da\le0$$, that is $$x\,g'(x)\le4g(x)$$, and this holds if it holds for each $$q$$. For fixed $$q$$ let $$s=\sqrt{q^2+x^2}$$. Then $$4(q+x-s)-x(1-x/s)=\big[(4q+3x)s-4q^2-3x^2\big]/s$$, which is non-negative because both $$(4q+3x)s$$ and $$4q^2+3x^2$$ are positive and

$$
(4q+3x)^2(q^2+x^2)-(4q^2+3x^2)^2=q^2x^2+24q^3x+24qx^3\ge0.
$$

By (1), $$\rho_X+p_X=\tfrac13(1+z)\,dX_\mu/dz\ge0$$. Next $$p(\mu)=\int\tfrac{q^2}{3E}f\le\int\tfrac q3f=p(0)$$, so $$p_X\ge0$$. Finally $$p_X\le X_\mu/3$$ is equivalent to $$\rho(\mu)-3p(\mu)\le\mu n_1$$, that is $$\int\mu^2f/E\le\mu n_1$$, which holds since $$E\ge\mu$$. With $$X_\mu\ge0$$ and $$0\le p_X\le X_\mu/3$$ the weak and dominant energy conditions follow. $$\square$$
</div>

<div class="corollary" markdown="1">
**Corollary 1** (an exact mimic of any negative mass sum). Let $$\Seff<0$$ and let the truth consist of photons, baryons, stable CDM $$\omega_c$$, a cosmological constant, neutrinos of masses $$m_i\ge0$$, and the fluids $$X_{m_i}$$ ($$i=1,2,3$$) and three copies of $$X_{\lvert\Seff\rvert/3}$$. Let the fit have the same photons, baryons and cosmological constant, the three negative-mass species of (4) and CDM density $$\omega_c+\delta$$ with $$\delta=(\Strue+\lvert\Seff\rvert)\,n_1(0)$$. Then the two have the same total energy density at every redshift $$z\ge0$$, so the same $$H(z)$$, $$r_d$$, $$\theta_\ast$$ and BAO distances, and every component of the truth other than the stable CDM and $$\Lambda$$ obeys the null, weak and dominant energy conditions.
</div>

<div class="proof" markdown="1">
*Proof.* By (7), $$\rho(m_i)+X_{m_i}=\rho(0)+m_i\,n_1(1+z)^3$$, so the true neutrinos and the $$X_{m_i}$$ together contribute $$3\rho(0)+\Strue n_1(1+z)^3$$. The three copies of $$X_{\lvert\Seff\rvert/3}$$ contribute $$3\rho(0)+\lvert\Seff\rvert n_1(1+z)^3-3\rho(\lvert\Seff\rvert/3)$$. The sum is $$6\rho(0)-3\rho(\lvert\Seff\rvert/3)+(\Strue+\lvert\Seff\rvert)n_1(1+z)^3$$, which is the density of the three negative-mass species, $$3[2\rho(0)-\rho(\lvert\Seff\rvert/3)]$$ by (4), plus the extra dust $$\delta(1+z)^3$$ of the fit. The photons, baryons and $$\Lambda$$ are common to both. The neutrinos satisfy every energy condition, and the $$X_\mu$$ satisfy the null, weak and dominant conditions by Proposition 1. $$\square$$
</div>

At $$\Seff=-0.10\eV$$ the total $$X$$ is $$1.14\%$$ of the density of baryons and CDM at $$z=1060$$, and $$\delta=1.707\times10^{-3}$$ exceeds $$\delta_{\min}=1.669\times10^{-3}$$ by $$2.3\%$$, as Theorem 2 requires. The two densities in Corollary 1 agree to a relative accuracy of $$10^{-15}$$ in a numerical evaluation for $$\Seff=-0.05,-0.10,-0.20\eV$$ at redshifts up to $$10^6$$.

The fluid $$X_\mu$$ is matter that turns into radiation on the schedule on which a neutrino of mass $$\mu$$ becomes non-relativistic, run in reverse. At the level of the background it is an exactly degenerate version of the decaying dark matter of [[LK]](#ref-lk), whose exponential decay is only nearly degenerate. It violates only the hypothesis that the CDM seen at recombination is stable. Corollary 1 is a statement about the background. A CMB analysis of this universe returns a cold matter density that depends on how $$X_\mu$$ clusters, which $$H(z)$$ does not determine.

## 5. The floor under DESI DR2 weights

The data constrain a finite set of distances, not the whole function $$H(z)$$, so the question becomes how low $$\Seff$$ can be when the truth contains only NEC-respecting extra components. We write $$\SBAO$$ for the best-fit $$\Seff$$ of the flat $$\Lambda$$CDM$$+\Seff$$ model fitted to BAO data and a compressed CMB prior, without CMB lensing. The set-up, to which every number in this section and the next refers, is as follows.

- The BAO data are the thirteen numbers of the DESI DR2 table with the correlation coefficients of the table [[DESI DR2]](#ref-desi-dr2).
- The CMB enters as a Gaussian prior on $$(\theta_\ast,\omega_b,\omega_{cb})$$, which stands in for the CMB spectra, with the covariance of appendix A of the same paper (their (A2)). It is centred on the values of the true model, so that the mock data are noiseless. The recombination redshift is from the fit of Hu and Sugiyama [[HS]](#ref-hs), with the density $$\omega_{cb}$$ of baryons and CDM and without the neutrino density.
- The sound horizon $$r_d$$ is taken from the fitting formula of Aubourg et al. [[Aubourg]](#ref-aubourg) (their (16)), with the neutrino density $$\omega_\nu=\Seff/93.14\eV$$ continued to negative $$\Seff$$ although the formula was fitted for positive masses. In a variant, $$r_d$$ is obtained by integrating $$c_s/H$$ above the drag redshift of Eisenstein and Hu [[EH]](#ref-eh), multiplied by a constant ($$0.9755$$) so that the two treatments agree at the fiducial point. This variant lets $$r_d$$ respond to any extra component present before $$\zd$$.
- The neutrinos have exact Fermi–Dirac thermodynamics with the continuation (4), and the photon temperature is $$2.7255\,$$K. The fit parameters are $$(\omega_b,\omega_c,h,\Seff)$$.
- The true model is flat with $$\omega_b=0.02223$$, $$\omega_{cb}=0.14208$$ (the centre of the prior), $$h=0.68$$ and the minimal normal ordering. When extra components are added, $$h$$ is re-solved so that $$\theta_\ast$$ of the truth is unchanged, which holds the early universe fixed.

Replacing the covariance of (A2) by uncorrelated Gaussians of Planck-like widths changes the floor below by less than $$0.1\meV$$ and the curvature slope of section 6 by 1%. A full-likelihood analysis is not part of the set-up.

For the truth without extra components, the fit returns $$\Seff=0.06379\eV$$ with the formula of Aubourg et al. and $$0.06410\eV$$ with the integrated $$r_d$$; it moves by less than $$0.1\meV$$ when $$h$$ is varied between $$0.66$$ and $$0.70$$. The mock's forecast uncertainty (the inverse Fisher matrix) is $$0.087\eV$$, so this is a statement about the central value. The excess of $$5.0\meV$$ over $$\Strue=58.78\meV$$ is an effect of the convexity gap of section 3: the fit imposes equal masses on a hierarchical spectrum (Remark 4), while a fit that uses the true spectrum returns $$\Strue$$. For other truths the same procedure gives

| truth | $$\Strue$$ (meV) | fitted $$\Seff$$ (meV) |
|---|---|---|
| normal ordering, lightest $$0$$ | 58.78 | 63.79 |
| normal ordering, lightest $$10\meV$$ | 74.34 | 75.89 |
| normal ordering, lightest $$30\meV$$ | 119.64 | 119.86 |
| inverted ordering, lightest $$0$$ | 98.92 | 103.42 |
| degenerate | any | $$\Strue$$ |

where the inverted ordering uses $$\lvert s_{32}\rvert=2.484\times10^{-3}\eV^2$$ (NuFit 6.0 with Super-Kamiokande data). In these cases the fitted value for the truth without extra components is at least $$\Strue$$; for $$\Strue=0$$ it is $$0$$. The number $$0.0638\eV$$ refers to the minimal normal ordering.

An uncoupled NEC component has a density that is non-decreasing in redshift. It is a sum of downward steps in redshift,

$$
\rho_X(z)=\rho_X(0)+\int\Theta(z-z')\,d\lambda(z'),\qquad d\lambda\ge0,
$$

with $$\Theta$$ the unit step; a constant is the case $$\lambda=0$$. To first order $$\SBAO=\SBAO[\Lambda{\rm CDM}]+\int\mathcal K(z')\,d\lambda(z')$$, where the kernel $$\mathcal K(z')$$ is the change of $$\SBAO$$ per unit step at $$z'$$. We computed $$\mathcal K$$ with smoothed steps (width $$0.4\%$$ in $$\ln(1+z)$$) of small amplitude, so that the response is linear. For $$z'<100$$ the amplitude is measured in units of the critical density today, and for $$z'\ge200$$ in units of $$\varepsilon\equiv\Delta\rho_X/\rho_m(z=1060)$$, the step as a fraction of the matter density at the drag redshift.

| $$z'$$ | 0.1 | 0.3 | 0.6 | 1.0 | 1.6 | 2.5 | 4 | 10 |
|---|---|---|---|---|---|---|---|---|
| $$\mathcal K$$, Aubourg $$r_d$$ (eV per critical density today) | 0.19 | 0.69 | 0.81 | 0.79 | 0.39 | 0.14 | 0.042 | 0.0027 |
| $$\mathcal K$$, integrated $$r_d$$ (eV per critical density today) | 0.20 | 0.74 | 0.86 | 0.82 | 0.40 | 0.14 | 0.043 | 0.0027 |

| $$z'$$ | 200 | 300 | 400 | 500 | 600 | 800 | 1000 |
|---|---|---|---|---|---|---|---|
| $$\mathcal K$$, Aubourg $$r_d$$ (eV per unit $$\varepsilon$$) | 31 | 5.1 | −0.04 | −1.6 | −2.2 | −2.6 | −2.7 |
| $$\mathcal K$$, integrated $$r_d$$ (eV per unit $$\varepsilon$$) | 32 | 6.0 | 0.84 | −0.71 | −1.3 | −1.7 | −1.8 |

The entry at $$z'=0.3$$ lies next to the $$D_V$$ point at $$z=0.295$$ and varies quickly: with the formula of Aubourg et al. $$\mathcal K=0.46$$ at $$z'=0.28$$ and $$0.72$$ at $$0.32$$. On a dense geometric set of redshifts from $$0.005$$ to $$300$$, $$\mathcal K\ge0$$ for both treatments of $$r_d$$, apart from values of order $$10^{-5}\eV$$ of either sign at $$z'\approx60$$–$$100$$, where the response is itself of that size and below the numerical resolution. Beyond $$z'=300$$ the kernel decreases through zero, at $$z'=398$$ with the formula of Aubourg et al. and at $$z'=441$$ with the integrated $$r_d$$, and its most negative values are $$-2.7\eV$$ and $$-1.9\eV$$ per unit $$\varepsilon$$, near $$z'=1050$$–$$1100$$.

<div class="remark" markdown="1">
**Remark 5** (the floor, numerical). In the set-up above, take a true model with the minimal normal ordering and extra components that are separately conserved with $$\rho+p\ge0$$ and whose total density is constant for $$z\ge300$$. Then, to first order in the amplitude of the extra components, $$\SBAO\ge0.0638\eV$$ (and $$\ge0.0641\eV$$ with the integrated $$r_d$$). If the density instead increases between $$z=300$$ and $$z=1060$$ by a fraction $$\varepsilon$$ of $$\rho_m(1060)$$, then to first order $$\SBAO\ge0.0638\eV-\kappa\,\varepsilon$$ with $$\kappa\approx2.7\eV$$ (Aubourg $$r_d$$) or $$1.9\eV$$ (integrated $$r_d$$). In the full non-linear fit a step of $$\varepsilon=0.01$$ at $$z'=800$$ gives $$\SBAO=0.0376\eV$$ and $$0.0467\eV$$ for the two treatments, and a step of $$\varepsilon=0.003$$ gives $$0.0560\eV$$ and $$0.0589\eV$$.
</div>

The condition of constant density above $$z=300$$ is sufficient with margin. The first-order result needs only $$\mathcal K\ge0$$ over the support of $$d\lambda$$, which holds up to $$z'\approx400$$ for the formula of Aubourg et al. and up to $$440$$ for the integrated $$r_d$$, but $$\mathcal K(400)$$ is already slightly negative with the first treatment.

Beyond first order, $$\SBAO$$ stays at or above the floor for single steps of up to three critical densities at redshifts from $$0.1$$ to $$250$$ and for random multi-step histories at $$z'<300$$ with amplitudes up to three critical densities. The smallest value in the multi-step histories is $$0.06384\eV$$, against $$0.06379\eV$$ without extra components, and the single-step values range from $$0.0638\eV$$ to $$1.6\eV$$. These cases are samples.

The positivity of $$\mathcal K$$ at low redshift depends on the DR2 weights. With the six $$D_H$$ measurements alone, $$\mathcal K(0.1)=-0.69\eV$$ and $$\mathcal K(0.3)=-1.78\eV$$ per unit critical density, while with the six $$D_M$$ measurements alone $$\mathcal K$$ is positive at the same redshifts ($$0.25$$ and $$0.65$$). Data weighted more towards $$D_H$$ could lie below the floor with an NEC component at low redshift.

The same set-up applied to the DR2 numbers, with the prior centred on the values of appendix A of the DESI paper and $$100\theta_\ast=1.0411$$ for the CMB acoustic scale of Planck [[Planck 2018]](#ref-planck18) (our $$\theta_\ast$$ is rescaled by $$1.0013$$ to reproduce it), returns $$\SBAO=-0.152\pm0.087\eV$$ with $$\chi^2=10.7$$ for the thirteen BAO numbers and the three prior entries; with the uncorrelated Planck-like widths the uncertainty is $$0.077\eV$$. Moving $$100\theta_\ast$$ between $$1.0410$$ and $$1.0412$$ moves the central value between $$-0.156$$ and $$-0.148\eV$$. For Gaussian errors the central value lies $$2.5\sigma$$ below the floor of $$0.0638\eV$$. The DESI result $$-0.101\eV$$ is a different quantity, since it includes CMB lensing, which is the main source of the preference for negative values [[GM]](#ref-gm), whereas $$\SBAO$$ excludes it.

The mimic of Corollary 1 is not among the truths of this section, for two reasons. At $$\Seff=-0.10\eV$$ its total extra density at $$z=300$$ is $$2\%$$ of its value at $$z=1060$$, so it is an NEC component whose density increases with redshift by $$\varepsilon=0.011$$ in the window where $$\mathcal K<0$$, and Remark 5 does not apply to it. And the prior here is centred on the stable $$\omega_{cb}$$ of the truth, whereas the CMB in Corollary 1 constrains the density $$\omega_{cb}+\delta$$ of the fit, which is displaced from the stable value by $$1.4$$ times the uncertainty of $$\omega_c$$ (section 4). The floor and Theorem 2 therefore refer to different identifications of the density that the CMB measures.

## 6. Curvature and the BAO distances

Spatial curvature acts as a fluid with $$w=-\tfrac13$$ and density $$\omega_K(1+z)^2$$, where $$\omega_K=\Omega_Kh^2$$ and $$\Omega_K>0$$ is open. It contributes $$\tfrac23\omega_K(1+z)^2$$ to the sum of $$\rho+p$$ in (2). Suppose the truth has curvature, the fit is flat, and only $$H(z)$$ is matched. For $$\Omega_K\ge0$$ the curvature term is non-negative, the sum over the $$X_k$$ is more negative than in Lemma 2, and Theorem 1 holds unchanged. For $$\Omega_K<0$$ an NEC-respecting mimic exists if and only if $$\lvert\omega_K\rvert\ge\tfrac32\sup_{z\in I}\Delta(z)/(1+z)^2$$, the proof being that of Theorem 2. The supremum is numerically at the upper end of the interval, $$z=1060$$, and for $$h=0.68$$ it gives $$\lvert\Omega_K\rvert\ge0.056,\ 0.072,\ 0.081,\ 0.186$$ for $$\Seff=0.0488,\ 0.03,\ 0,\ -0.10\eV$$. A closed universe with $$\lvert\Omega_K\rvert\gtrsim0.06$$ is needed for $$\Seff$$ below $$0.05\eV$$. DESI DR2 combined with the CMB gives $$\Omega_K=+0.0023\pm0.0011$$ when curvature is a free parameter of $$\Lambda$$CDM [[DESI DR2]](#ref-desi-dr2) (their Table V), $$2.1\sigma$$ from flat, so these values lie many standard deviations beyond it.

If the distances are matched as well as $$H(z)$$, the premise cannot be satisfied with $$\Omega_K\ne0$$: equal $$H$$ gives equal comoving distance $$D_C=c\int_0^zdz'/H(z')$$ to every redshift, but the transverse distance is $$\sinh$$ or $$\sin$$ of it for the curved model and $$D_C$$ for the flat one. For $$\Omega_K>0$$ the transverse distance is

$$
D_M=\frac{c}{H_0\sqrt{\Omega_K}}\sinh\!\Big(\sqrt{\Omega_K}\,\frac{H_0}{c}\,D_C\Big),
$$

which exceeds $$D_C$$, and a flat fit cannot absorb the hyperbolic sine into $$H(z)$$. At fixed early universe, the flat model's distances grow when its late-time matter density falls, which is the direction of a smaller mass sum. We take a curved $$\Lambda$$CDM truth with $$\omega_b$$, $$\omega_{cb}$$, $$\theta_\ast$$ and (through the formula of Aubourg et al.) $$r_d$$ held fixed, and fit flat $$\Lambda$$CDM$$+\Seff$$ in the set-up of section 5:

| $$\Omega_K$$ | −0.002 | −0.001 | −0.0005 | 0 | +0.0005 | +0.001 | +0.002 | +0.0023 |
|---|---|---|---|---|---|---|---|---|
| $$\SBAO$$ (eV) | 0.190 | 0.128 | 0.097 | 0.064 | 0.030 | −0.019 | −0.092 | −0.113 |

The derivative at $$\Omega_K=0$$ is $$d\SBAO/d\Omega_K\approx-66.6\eV$$, from symmetric differences with steps $$10^{-4}$$ and $$5\times10^{-4}$$ ($$-66.5$$ and $$-66.8\eV$$). The response is not linear over steps of $$10^{-3}$$, for which the symmetric difference is $$-73\eV$$. The continuation (4) makes $$\rho(m)=\rho(0)+c\,m\lvert m\rvert$$ for $$\lvert m\rvert\lesssim T_\nu(z)$$, which is $$C^1$$ with vanishing slope at $$m=0$$; at the redshifts that carry the BAO signal, $$z<2$$, this covers $$\lvert\Sigma\rvert$$ of about $$1\meV$$ or less, and outside it the dependence is linear in $$\lvert m\rvert$$. In that window the data lose sensitivity to $$\SBAO$$, and the curve of the table is not smooth near its zero: SCAN_PLACEHOLDER Entering the same curvature in a flat $$H(z)$$ as a fluid with $$w=-\tfrac13$$ gives a positive response, in agreement with the kernel of section 5, so the negative response is an effect of the transverse distance.

At $$\Omega_K=0.0023$$ the table gives $$\SBAO=-0.113\eV$$. Open curvature of the size DESI reports therefore accounts, in the set-up of section 5, for the whole BAO-channel value $$-0.152\eV$$ except for about $$0.04\eV$$, with a truth that satisfies every energy condition. The table is a map from a curved truth to the value a flat fit returns. DESI reports no significant preference for a non-flat universe.

## 7. What the results do not exclude

Theorem 1 and its numerical counterpart concern a flat universe with stable CDM, the same $$\omega_b$$ and $$\omega_c$$ in the truth and the fit, standard neutrinos and a degenerate fit. A negative BAO-inferred mass sum is compatible with relaxing any one of these, under the following conditions.

1. If the other hypotheses hold and $$H(z)$$ is matched exactly, Theorem 1 requires a component with $$\rho+p<0$$ throughout the matched interval. Phantom dark energy, components with negative density and the non-dynamical sources discussed in [[GGM]](#ref-ggm) (section 5.1) are of this kind, with $$\lvert1+w\rvert\sim2\times10^{-3}$$ to $$5\times10^{-3}$$ at $$z=0$$ for $$\Seff=0$$ to $$-0.10\eV$$ if the component is the dark energy (Example 1). In the Freezing Gravity model, which prescribes the expansion history freely, the page on its [matter-induced instability](/quantum-gravity/matter-induced-instability/) finds the phantom phases unstable with minimally coupled dust under the conditions stated there.

2. If the CDM density of the fit exceeds the stable one, Theorem 2 requires $$\delta\ge\delta_{\min}\approx(\Strue-\Seff)/93\eV$$, that is, about $$1.7\times10^{-3}$$ in $$\omega_c$$ for $$\Seff=-0.10\eV$$. The excess must be matter that clusters at recombination and is not stable afterwards: decay to radiation, conversion to dark energy or a coupling between dark matter and dark energy. Proposition 1 is a decay that obeys every energy condition. The condition does not fix how such matter clusters, and the lensing channel depends on that. A conversion into dark energy requires energy to flow from matter to the vacuum, the direction opposite to the one that the [sign theorem](/quantum-gravity/vacuum-matter-sign/) establishes for a vacuum component that tracks the minimum of its free energy, relaxes passively towards it or exchanges energy through a unital map.

3. If the universe is curved, Theorem 1 itself is unaffected by open curvature when only $$H(z)$$ is matched. The BAO distances of a flat fit change by an amount equivalent to $$\Delta\SBAO\approx-66\eV\times\Omega_K$$ for $$\lvert\Omega_K\rvert\lesssim5\times10^{-4}$$ (section 6).

4. If the expansion history changes between $$z\approx300$$ and recombination, or $$r_d$$ changes, the floor can be lowered. In the numerical set-up an increase of the density of an NEC component with redshift in this window lowers $$\SBAO$$ by up to $$2.7\eV$$ per unit $$\varepsilon$$ (section 5). Compatibility of such a component with the CMB spectra has not been tested.

5. If the data weight $$D_H$$ more than DR2 does, the response to an NEC component at low redshift can be negative: with the six $$D_H$$ measurements alone it is.

6. If the fit convention is not degenerate, or the true neutrinos are not standard Fermi–Dirac species, Theorem 1 changes. Remark 3 gives the former at the level of $$0.67\meV$$. Decay, cooling and time-dependent masses are discussed in [[CGMR]](#ref-cgmr).

7. Any resolution that acts through the lensing channel is outside these statements, which concern the expansion history, while the full DESI+CMB value of $$-0.101\eV$$ includes CMB lensing. Whether an exact background mimic also mimics the lensing spectrum is not determined by $$H(z)$$.

## 8. Open points

The criterion of Theorem 2 concerns the background. Whether the CMB temperature and polarisation spectra of the universe in Corollary 1 return a cold matter density $$\omega_c+\delta$$, and what lensing amplitude they have, requires a perturbation calculation with a specified sound speed and viscosity for $$X_\mu$$.

The positivity of $$\mathcal K$$ on $$0<z'<300$$ and the first-order floor rest on a numerical response computed on a finite set of redshifts, with a compressed likelihood and fitting formulae for $$z_\ast$$ and $$r_d$$. The nonlinear cases are samples, and no proof is available that no NEC history lies below the floor. The window above $$z\approx400$$ in which the response has the opposite sign is characterised only for single steps. Because the prior is centred on the true model, the floor is a property of the best-fit value and not of a statistical bound; with the forecast uncertainty of $$0.087\eV$$, a fluctuation of the data below the floor is not excluded.

## References

1. <a id="ref-desi-dr2"></a>DESI Collaboration (M. Abdul-Karim et al.), "DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints," Phys. Rev. D 112, 083515 (2025), [arXiv:2503.14738](https://arxiv.org/abs/2503.14738).
2. <a id="ref-desi-nu"></a>W. Elbers et al. (DESI Collaboration), "Constraints on Neutrino Physics from DESI DR2 BAO and DR1 Full Shape," 2025, [arXiv:2503.14744](https://arxiv.org/abs/2503.14744).
3. <a id="ref-ggm"></a>P. W. Graham, D. Green, J. Meyers, "New Interpretations of the Cosmological Preference for a Negative Neutrino Mass," Phys. Rev. D 113, 043514 (2026), [arXiv:2508.20999](https://arxiv.org/abs/2508.20999).
4. <a id="ref-lk"></a>G. P. Lynch, L. Knox, "What's the matter with $$\Sigma m_\nu$$?", 2025, [arXiv:2503.14470](https://arxiv.org/abs/2503.14470).
5. <a id="ref-elbers"></a>W. Elbers, C. S. Frenk, A. Jenkins, B. Li, S. Pascoli, "Negative neutrino masses as a mirage of dark energy," Phys. Rev. D 111, 063534 (2025), [arXiv:2407.10965](https://arxiv.org/abs/2407.10965).
6. <a id="ref-gm"></a>D. Green, J. Meyers, "Cosmological preference for a negative neutrino mass," Phys. Rev. D 111, 083507 (2025), [arXiv:2407.07878](https://arxiv.org/abs/2407.07878).
7. <a id="ref-cgmr"></a>N. Craig, D. Green, J. Meyers, S. Rajendran, "No $$\nu$$s is Good News," JHEP 2024, 97 (2024), [arXiv:2405.00836](https://arxiv.org/abs/2405.00836).
8. <a id="ref-nufit"></a>I. Esteban, M. C. Gonzalez-Garcia, M. Maltoni, I. Martinez-Soler, J. P. Pinheiro, T. Schwetz, "NuFit-6.0: Updated global analysis of three-flavor neutrino oscillations," JHEP 12 (2024) 216, [arXiv:2410.05380](https://arxiv.org/abs/2410.05380).
9. <a id="ref-planck18"></a>Planck Collaboration, "Planck 2018 results. VI. Cosmological parameters," Astron. Astrophys. 641, A6 (2020), [arXiv:1807.06209](https://arxiv.org/abs/1807.06209).
10. <a id="ref-aubourg"></a>E. Aubourg et al., "Cosmological implications of baryon acoustic oscillation measurements," Phys. Rev. D 92, 123516 (2015), [arXiv:1411.1074](https://arxiv.org/abs/1411.1074).
11. <a id="ref-hs"></a>W. Hu, N. Sugiyama, "Small Scale Cosmological Perturbations: An Analytic Approach," Astrophys. J. 471, 542 (1996), [arXiv:astro-ph/9510117](https://arxiv.org/abs/astro-ph/9510117).
12. <a id="ref-eh"></a>D. J. Eisenstein, W. Hu, "Baryonic Features in the Matter Transfer Function," Astrophys. J. 496, 605 (1998), [arXiv:astro-ph/9709112](https://arxiv.org/abs/astro-ph/9709112).
