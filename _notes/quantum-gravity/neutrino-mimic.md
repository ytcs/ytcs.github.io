---
title: What a negative neutrino mass in BAO data would require
date: 2026-10-29
category: dark-sector
order: 4
description: In a flat universe with stable cold dark matter and standard neutrinos, no components that obey the null energy condition can reproduce the expansion history of a smaller neutrino mass sum. With the weights of the DESI DR2 BAO data, the mass sum inferred from the expansion history has a numerical floor of 0.0638 eV under stated conditions.
status: The exact-mimicry theorem, its criterion for a mismatched dark matter density and the mimic fluid are proved; the floor under DESI DR2 weights and the curvature slope are numerical results in a compressed-likelihood set-up.
excludes: Explanations of a negative BAO-inferred neutrino mass sum in a flat universe by components that respect the null energy condition and leave the stable cold dark matter density unchanged.
---

Fits of flat $$\Lambda$$CDM with a neutrino mass sum $$\Seff$$ that is allowed to be negative return $$\Seff=-0.101^{+0.047}_{-0.056}\eV$$ when DESI DR2 baryon acoustic oscillation (BAO) data are combined with the cosmic microwave background (CMB) [[DESI ν]](#ref-desi-nu), below the $$0.0588\eV$$ that oscillation experiments require. We ask what the dark sector would have to do for such a fit to return less than the true mass sum. In a flat universe in which the baryon and cold dark matter (CDM) densities are the same in the fit and in the truth, and the CDM is stable, matching the expansion history of a fit with $$\Seff<\Strue$$ on any interval of redshift forces the other components to have a negative total $$\rho+p$$ throughout that interval, which is a violation of the null energy condition (NEC) (Theorem 1). If the CDM density of the fit exceeds the stable one by $$\delta$$, an NEC-respecting mimic exists if and only if $$\delta\ge\delta_{\min}$$, where $$\delta_{\min}$$ is within 4% of $$(\Strue-\Seff)/93\eV$$ for the values we tabulate, $$1.7\times10^{-3}$$ at $$\Seff=-0.10\eV$$ (Theorem 2). An explicit fluid, dust at early times and radiation at late times, realises the mimic exactly (Proposition 3).

With the weights of the DR2 data the question becomes quantitative, and the answer is numerical. In the set-up of section 6, any NEC component whose density is constant above $$z=300$$ leaves the BAO-inferred mass sum at or above $$0.0638\eV$$, the value for $$\Lambda$$CDM itself; an NEC component whose density keeps increasing with redshift between $$z=300$$ and recombination can lower it, and open curvature lowers it by about $$66\eV$$ per unit of $$\Omega_K$$ (section 7).

The physical picture is in the literature. Graham, Green and Meyers [[GGM]](#ref-ggm) separate the mass sum inferred from the expansion history from the one inferred from clustering, find that non-phantom dynamical dark energy pushes the first towards more negative values, and note that spatial curvature is degenerate with it (their section 5.1 estimates $$f_\nu^{\rm BAO}\approx-1100\,\Omega_K/\Omega_m$$). Lynch and Knox [[LK]](#ref-lk) propose dark matter that decays to dark radiation at a rate chosen so that the decay coincides with the epoch at which a neutrino becomes non-relativistic, which makes it degenerate with a signed neutrino mass at the level of the background. Elbers et al. [[Elbers]](#ref-elbers) introduced the effective mass parameter that extends consistently to negative values and that DESI uses, and Green and Meyers [[GM]](#ref-gm) trace the preference for negative values in the full data to CMB lensing. What this page adds is the exact statement with a proof valid for any true mass spectrum at every redshift (Theorem 1); the sharp criterion $$\delta_{\min}$$ and a fluid that attains it with all energy conditions satisfied (Theorem 2 and Proposition 3); and, in a set-up that is specified in full, the floor with the redshift window in which it holds and the curvature slope. The sign structure of Theorem 1 is the same as in the [sign theorem for vacuum-matter exchange](/quantum-gravity/vacuum-matter-sign/), where a comparison of two minimisers is replaced by a comparison of two expansion histories.

## 1. The expansion history and what BAO measure

We work in a spatially flat Friedmann–Robertson–Walker universe with metric $$ds^2=-dt^2+a(t)^2d\mathbf x^2$$. The redshift is $$1+z=1/a$$ and the Hubble rate is $$H=\dot a/a$$. Each component $$i$$ has an energy density $$\rho_i$$ and a pressure $$p_i$$, and the Friedmann equation $$H^2=\tfrac{8\pi G}{3}\sum_i\rho_i$$ holds. Total energy is conserved, $$\dot\rho_{\rm tot}=-3H(\rho_{\rm tot}+p_{\rm tot})$$. A component that exchanges no energy with the others is separately conserved, and since $$d/dt=-(1+z)H\,d/dz$$ it obeys

$$
\frac{d\rho_i}{dz}=\frac{3(\rho_i+p_i)}{1+z}. \tag{1}
$$

Pressureless matter ("dust", $$p=0$$) has $$\rho\propto(1+z)^3$$, radiation ($$p=\rho/3$$) has $$\rho\propto(1+z)^4$$, and a cosmological constant ($$p=-\rho$$) has constant $$\rho$$. The NEC for a component is $$\rho_i+p_i\ge0$$. By (1) a separately conserved component obeys the NEC if and only if its density does not decrease with increasing redshift. A component with $$\rho+p<0$$ is phantom: its density grows as the universe expands.

It is convenient to measure densities in units of $$3(100\,{\rm km\,s^{-1}\,Mpc^{-1}})^2/8\pi G$$, so that $$H^2=\sum_i\rho_i$$ with $$H$$ in units of $$100\,{\rm km\,s^{-1}\,Mpc^{-1}}$$ and the present density of a dust component is $$\omega_i=\Omega_ih^2$$, where $$H_0=100h\,{\rm km\,s^{-1}\,Mpc^{-1}}$$. Differentiating $$H^2=\sum_i\rho_i$$ and using conservation of the total gives

$$
(1+z)\,\frac{d(H^2)}{dz}=3\sum_i(\rho_i+p_i). \tag{2}
$$

Two flat models with the same $$H(z)$$ on an interval therefore have the same total $$\rho+p$$ there, whatever they are made of. Their energy densities may differ, but their $$\rho+p$$ may not. The sign of this fixed quantity is what the rest of the page examines.

Baryon acoustic oscillations are the imprint, in the clustering of galaxies, of sound waves that propagated in the photon-baryon plasma before the baryons decoupled from the photons at the drag redshift $$\zd\approx1060$$ [[Planck 2018]](#ref-planck18). The comoving distance the waves travelled is the sound horizon $$r_d=\int_{\zd}^\infty c_s\,dz/H(z)$$ with sound speed $$c_s$$. Measuring the angular size of the feature at redshift $$z$$ gives the transverse comoving distance $$D_M(z)=c\int_0^zdz'/H(z')$$ in units of $$r_d$$, and measuring its extent in redshift along the line of sight gives $$D_H(z)=c/H(z)$$ in units of $$r_d$$. Where the signal is weaker only the combination $$D_V=(zD_M^2D_H)^{1/3}$$ is constrained. DESI DR2 reports $$D_V/r_d$$ at one effective redshift, $$z=0.295$$, and $$D_M/r_d$$ and $$D_H/r_d$$ with a correlation coefficient at each of six redshifts from $$0.510$$ to $$2.330$$, thirteen numbers in all [[DESI DR2]](#ref-desi-dr2) (their Table IV).

The CMB fixes the early universe. It constrains the physical baryon density $$\omega_b$$, the density $$\omega_{cb}=\omega_b+\omega_c$$ of baryons and CDM, and the angular scale $$\theta_\ast=r_\ast/D_M(z_\ast)$$ of the sound horizon at recombination, $$z_\ast\approx1090$$. Given $$\omega_b$$ and $$\omega_{cb}$$, the sound horizon $$r_d$$ follows from the physics before $$\zd$$, and the BAO distances then probe $$H(z)$$ at $$z\lesssim2.5$$. The fit that returns a mass sum $$\Seff$$ is flat $$\Lambda$$CDM with free $$(\omega_b,\omega_c,h,\Seff)$$, three neutrino species of equal mass $$\Seff/3$$, and $$\Lambda$$ fixed by flatness. Neutrinos with $$m\gg T$$ behave as dust, so at fixed early universe a larger $$\Seff$$ means more matter at late times and a different $$H(z)$$; this is the dependence through which BAO measure the mass sum.

## 2. Neutrinos as a source of $$\rho+p$$

After decoupling, a neutrino species keeps a Fermi–Dirac distribution in comoving momentum, so its effective temperature falls as $$T(z)=T_{\nu0}(1+z)$$. We take $$T_{\nu0}=0.1683\meV$$, the instantaneous-decoupling value $$(4/11)^{1/3}T_{\gamma0}=0.1676\meV$$ rescaled by $$(N_{\rm eff}/3)^{1/4}$$ with $$N_{\rm eff}=3.044$$, so that three massless species have the standard radiation density; one massless species then has $$\rho=5.70\times10^{-6}$$ in the units of section 1 today. For one species of mass $$m\ge0$$, counting particle and antiparticle, let $$E=\sqrt{q^2+m^2}$$, $$f(q)=(e^{q/T}+1)^{-1}$$ and $$\int\equiv\pi^{-2}\int_0^\infty q^2\,dq$$. Then

$$
\rho(m)=\int E\,f,\qquad p(m)=\int\frac{q^2}{3E}\,f,\qquad j(m)\equiv\rho(m)+p(m)=\int\Big(E+\frac{q^2}{3E}\Big)f. \tag{3}
$$

Each species is separately conserved, so by (1) $$d\rho/dz=3j/(1+z)$$. For $$m\ll T$$ the species is radiation and $$j=\tfrac43\rho(0)$$. For $$m\gg T$$ it is dust with $$\rho\simeq m\,n_1(z)$$, where $$n_1=n_1(0)(1+z)^3$$ is the number density of one species, and $$j\simeq m\,n_1$$. With the temperature above, $$n_1(0)=0.010747\eV^{-1}$$ in our units, so that three species of total mass $$\Sigma$$ contribute $$\omega_\nu=\Sigma/93.05\eV$$ at late times. (The convention $$\Sigma/93.14\eV$$ is also in use; the 0.1% difference is immaterial here.)

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

At $$m=0$$ the integrand is $$\tfrac23q^{-1}f$$ against the measure $$q^2dq$$, which is integrable, so the second derivative is positive there too. For $$m<0$$, (4) gives $$j'(m)=j'(\lvert m\rvert)>0$$, with value $$0$$ only at $$m=0$$. We checked both integrands symbolically. $$\square$$
</div>

Both properties are what the comparison needs. Monotonicity says that a smaller mass sum carries less $$\rho+p$$ at every temperature. Convexity, through Jensen's inequality, says that a hierarchical spectrum carries more than the degenerate spectrum of the same sum. We checked numerically on $$0\le m/T\le30$$ that $$j$$ is increasing and convex, and that $$\partial_m^2j$$ at $$m=0$$ equals $$\tfrac23\int f/q$$ (both $$0.0965$$ in units of $$\rho(0)/T^2$$).

## 3. Exact mimicry forces a violation of the NEC

The statement is about two universes, the true one and the fit, and the hypotheses are as follows.

(H1) Both are spatially flat Friedmann–Robertson–Walker universes with conserved total energy.

(H2) Photons, baryons and CDM have the same present densities $$\omega_\gamma,\omega_b,\omega_c$$ in the truth and in the fit, and the same $$N_{\rm eff}$$. Baryons and CDM are stable and separately conserved: $$\rho_b=\omega_b(1+z)^3$$ and $$\rho_c=\omega_c(1+z)^3$$ in both.

(H3) The true neutrinos are three standard Fermi–Dirac species with masses $$m_i\ge0$$ and sum $$\Strue$$.

(H4) The fit is flat $$\Lambda$$CDM with three neutrino species of equal mass $$\Seff/3$$, continued by (4) when $$\Seff<0$$, and a cosmological constant.

(H5) The true model may contain further components $$X_a$$, with arbitrary pressures and densities (negative densities are allowed), not necessarily separately conserved. Only the conservation of the total energy is used.

Set

$$
\Delta(z)\equiv\sum_{i=1}^3j\big(m_i;T(z)\big)-3\,j\big(\Seff/3;T(z)\big),
$$

the $$\rho+p$$ of the true neutrinos minus that of the fitted ones.

<div class="lemma" markdown="1">
**Lemma 2.** Under (H1)–(H5), if the Hubble rate of the truth equals that of the fit on an open interval $$I$$ of redshift, then $$\sum_a(\rho_a+p_a)(z)=-\Delta(z)$$ for every $$z\in I$$.
</div>

<div class="proof" markdown="1">
*Proof.* Equal $$H$$ on $$I$$ gives equal $$d(H^2)/dz$$ there, and by (2) equal total $$\rho+p$$. The fit contains photons, baryons, CDM, its neutrinos and $$\Lambda$$, for which $$\rho+p=0$$. The truth contains the same photons, baryons and CDM by (H2), its own neutrinos, and the $$X_a$$. The common terms cancel. $$\square$$
</div>

<div class="theorem" markdown="1">
**Theorem 1** (exact mimicry violates the NEC). Assume (H1)–(H5) and $$\Seff<\Strue$$. If the Hubble rate of the truth equals that of the fit on an open interval $$I$$, then $$\Delta(z)>0$$ and therefore $$\sum_a(\rho_a+p_a)(z)<0$$ at every $$z\in I$$. In particular some component violates the NEC at every redshift of $$I$$.
</div>

The idea is that the true neutrinos, with more mass, have more $$\rho+p$$ than the fitted ones at every temperature, so the remaining components must carry the opposite sign. The proof is a comparison with two steps, monotonicity and Jensen's inequality, both routine given Lemma 1; the content of the theorem is that this is all that is needed and that no assumption on the spectrum, on the redshift or on the conservation of the $$X_a$$ enters.

<div class="proof" markdown="1">
*Proof.* By Lemma 2 it suffices to show $$\Delta(z)>0$$ for each $$z\in I$$. Fix $$z$$, hence $$T$$. If $$0\le\Seff<\Strue$$, monotonicity gives $$3j(\Seff/3)<3j(\Strue/3)$$, and convexity with Jensen's inequality, applied to $$m_i\ge0$$ with mean $$\Strue/3$$, gives $$3j(\Strue/3)\le\sum_ij(m_i)$$. If $$\Seff<0$$, monotonicity of $$j$$ on $$\mathbb R$$ gives $$3j(\Seff/3)<3j(0)$$, and $$3j(0)\le\sum_ij(m_i)$$ since $$m_i\ge0$$. In both cases $$\Delta(z)>0$$. $$\square$$
</div>

Four remarks fix the scope.

First, the $$X_a$$ need not be separately conserved, and the redshift enters only through $$T(z)$$, so the statement holds on any interval, including $$z>\zd$$. If the $$X_a$$ are separately conserved, (1) shows that their total density $$\rho_X$$ obeys $$d\rho_X/dz=-3\Delta/(1+z)<0$$: it grows as the universe expands. At low redshift, where the neutrinos are non-relativistic, $$\rho_X\simeq C-(\Strue-\Seff)\,n_1(z)$$ for a constant $$C$$, a cosmological constant minus the missing neutrino rest mass.

Second, the proof extends to any fit spectrum that is majorised by the true one at equal sum, with Karamata's inequality in place of Jensen's; section 5 shows that some condition of this kind on the fit is needed.

Third, the inequality is sharp at $$\Seff=\Strue$$: for a degenerate truth and $$\Seff=\Strue$$ one has $$\Delta\equiv0$$, and for the hierarchical truth of the fourth remark with $$\Seff=\Strue$$ the convexity gap is $$\Delta(0)=7.5\times10^{-6}$$ in the units of section 1. This residual is the reason the fit to a normal-ordering truth in section 6 returns a value $$5\meV$$ above the true sum.

Fourth, the mass spectrum we use for all numbers is the minimal normal ordering with lightest mass zero, $$m=(0,\,8.655,\,50.13)\meV$$ and $$\Strue=58.78\meV$$, from $$\Delta m^2_{21}=7.49\times10^{-5}\eV^2$$ and $$\Delta m^2_{31}=2.513\times10^{-3}\eV^2$$, the best fit of NuFit 6.0 including Super-Kamiokande atmospheric data [[NuFit]](#ref-nufit). Using the value $$2.534\times10^{-3}\eV^2$$ of the fit without those data changes $$\Strue$$ by $$0.2\meV$$.

*A worked example.* At $$z=0$$ and for $$\Seff=0$$ we find $$\Delta=6.17\times10^{-4}$$, so by Lemma 2 the extra components must have $$\sum_a(\rho_a+p_a)=-6.17\times10^{-4}$$ in our units. For $$h=0.68$$ this is $$-1.3\times10^{-3}$$ of the critical density today. If the extra component is the dark energy, with a density of $$0.69$$ of critical, this is $$1+w=-1.9\times10^{-3}$$. For $$\Seff=-0.10\eV$$ the same computation gives $$\Delta=1.67\times10^{-3}$$, $$-3.6\times10^{-3}$$ of critical and $$1+w=-5.2\times10^{-3}$$. The required phantom departures are small, and they must be present throughout the matched interval: $$\Delta(z)$$ grows to $$1.3\times10^{-2}$$ at $$z=1$$ and $$1.8$$ at $$z=10$$ for $$\Seff=-0.10\eV$$.

## 4. When the fit has more cold matter than the truth

Theorem 1 uses (H2) in a strong form: the CDM density of the fit is the stable CDM density of the truth. The CMB constrains the density of cold matter at recombination, and if part of that matter later ceases to be matter, the fit's $$\omega_c$$ exceeds the density of the stable CDM. Let the fit have $$\omega_c+\delta$$ with the truth keeping $$\omega_c$$, and otherwise assume (H1), (H3), (H4), (H5). The proof of Lemma 2 now has an extra dust term in the fit, $$\delta(1+z)^3$$, with $$\rho+p=\delta(1+z)^3$$, and gives

$$
\sum_a(\rho_a+p_a)(z)=\delta(1+z)^3-\Delta(z)\qquad(z\in I). \tag{5}
$$

<div class="theorem" markdown="1">
**Theorem 2** (criterion for an NEC-respecting mimic). Under the hypotheses just stated, there exist components $$X_a$$, each obeying $$\rho_a+p_a\ge0$$, such that the Hubble rate of the truth equals that of the fit on $$I$$, if and only if

$$
\delta\;\ge\;\delta_{\min}\equiv\sup_{z\in I}\frac{\Delta(z)}{(1+z)^3}. \tag{6}
$$

When (6) holds, a single separately conserved fluid does the job.
</div>

<div class="proof" markdown="1">
*Proof.* If each $$X_a$$ obeys the NEC, the left side of (5) is non-negative, which is (6). Conversely, assume (6) and take one fluid with $$\rho_X=\rho_\Lambda+\delta(1+z)^3+\rho_\nu^{\rm fit}-\rho_\nu^{\rm true}$$, where $$\rho_\Lambda$$ is the cosmological constant of the fit and $$\rho_\nu^{\rm fit,true}$$ are the densities of the three neutrino species of the fit and of the truth. Then the total density of the truth equals that of the fit at every redshift. The neutrino species are separately conserved, so by (1) the fluid has $$(1+z)\rho_X'/3=\delta(1+z)^3+3j(\Seff/3)-\sum_ij(m_i)=\delta(1+z)^3-\Delta$$, which is non-negative by (6), and the pressure $$p_X$$ is fixed by $$\rho_X$$ through (1). $$\square$$
</div>

The criterion is exact and the supremum can be evaluated. In the cases below the supremum is the limit $$z\to0$$, so that $$\delta_{\min}=\Delta(0)$$, which is $$(\Strue-\Seff)\,n_1(0)$$ up to the thermal energy of the neutrinos that are still mildly relativistic today.

| $$\Seff$$ (eV) | 0 | −0.05 | −0.10 | −0.20 |
|---|---|---|---|---|
| $$\delta_{\min}$$ | $$6.17\times10^{-4}$$ | $$1.132\times10^{-3}$$ | $$1.669\times10^{-3}$$ | $$2.744\times10^{-3}$$ |
| $$(\Strue-\Seff)/93.05\eV$$ | $$6.32\times10^{-4}$$ | $$1.169\times10^{-3}$$ | $$1.707\times10^{-3}$$ | $$2.781\times10^{-3}$$ |

(The interval is $$I=(0,1060)$$; the value of $$\delta_{\min}$$ does not change when the upper limit is moved between $$1000$$ and $$1090$$.) The excess needed for $$\Seff=-0.10\eV$$ is about $$1.4$$ times the uncertainty $$0.0012$$ of $$\omega_c$$ in the Planck analysis of the temperature, polarisation and lensing spectra [[Planck 2018]](#ref-planck18), so a shift of this size is not large on the scale of the CMB uncertainty. The excess must, however, be physical matter that clusters at recombination and is not stable afterwards.

The following fluid has this property exactly. For $$\mu>0$$ let

$$
X_\mu(z)\equiv\rho(0;z)+\mu\,n_1(z)-\rho(\mu;z), \tag{7}
$$

the density of a massless neutrino species plus the rest-mass density of a species of mass $$\mu$$, minus the density of a neutrino of mass $$\mu$$, each at temperature $$T(z)$$; and $$X_0\equiv0$$.

<div class="proposition" markdown="1">
**Proposition 3** (the fluid $$X_\mu$$). Treated as a separately conserved perfect fluid, $$X_\mu$$ has non-negative density that does not decrease with redshift, and it obeys the null, weak and dominant energy conditions with $$0\le w\le\tfrac13$$. It is dust for $$T\gg\mu$$ ($$X_\mu\simeq\mu n_1$$) and radiation for $$T\ll\mu$$ ($$X_\mu\to\rho(0)$$).
</div>

The idea is that $$X_\mu$$ is the difference between a neutrino's energy if it stayed relativistic with its rest mass added as dust, and its actual energy; this difference shrinks when the neutrino is relativistic, because its kinetic energy is then close to the dust-plus-massless sum, and is the whole rest mass once the neutrino has cooled. The proof is a pointwise inequality in the integrand.

<div class="proof" markdown="1">
*Proof.* Each term in (7) is separately conserved, so $$p_X=p(0)-p(\mu)$$ and the dust term has no pressure. In comoving momentum $$q$$ the distribution is $$f=(e^{q/T_{\nu0}}+1)^{-1}$$, independent of $$z$$. With $$a=1/(1+z)$$ and $$x=\mu a$$,

$$
X_\mu=a^{-4}g(x),\qquad g(x)=\frac1{\pi^2}\int_0^\infty f\,q^2\big[q+x-\sqrt{q^2+x^2}\big]dq\ge0,
$$

because $$\sqrt{q^2+x^2}\le q+x$$. Monotonicity in $$z$$ means $$dX_\mu/da\le0$$, that is $$x\,g'(x)\le4g(x)$$, and this holds if it holds for each $$q$$. For fixed $$q$$ let $$s=\sqrt{q^2+x^2}$$. Then $$4(q+x-s)-x(1-x/s)=\big[(4q+3x)s-4q^2-3x^2\big]/s$$, which is non-negative because both $$(4q+3x)s$$ and $$4q^2+3x^2$$ are positive and

$$
(4q+3x)^2(q^2+x^2)-(4q^2+3x^2)^2=q^2x^2+24q^3x+24qx^3\ge0
$$

(a symbolic identity). By (1), $$\rho_X+p_X=\tfrac13(1+z)\,dX_\mu/dz\ge0$$. Next $$p(\mu)=\int\tfrac{q^2}{3E}f\le\int\tfrac q3f=p(0)$$, so $$p_X\ge0$$. Finally $$p_X\le X_\mu/3$$ is equivalent to $$\rho(\mu)-3p(\mu)\le\mu n_1$$, that is $$\int\mu^2f/E\le\mu n_1$$, which holds since $$E\ge\mu$$. With $$X_\mu\ge0$$ and $$0\le p_X\le X_\mu/3$$ the weak and dominant energy conditions follow. $$\square$$
</div>

<div class="corollary" markdown="1">
**Corollary 2** (an exact mimic of any negative mass sum). Let $$\Seff<0$$ and let the truth consist of photons, baryons, stable CDM $$\omega_c$$, neutrinos of masses $$m_i\ge0$$, and the fluids $$X_{m_i}$$ ($$i=1,2,3$$) and three copies of $$X_{\lvert\Seff\rvert/3}$$. Let the fit have the three negative-mass species of (4) and CDM density $$\omega_c+\delta$$ with $$\delta=(\Strue+\lvert\Seff\rvert)\,n_1(0)$$. Then the two have the same total energy density at every redshift $$z\ge0$$, so the same $$H(z)$$, $$r_d$$, $$\theta_\ast$$ and BAO distances; and every component of the truth other than the stable CDM obeys all the energy conditions.
</div>

<div class="proof" markdown="1">
*Proof.* By (7), $$\rho(m_i)+X_{m_i}=\rho(0)+m_i\,n_1(1+z)^3$$, so the true neutrinos and the $$X_{m_i}$$ together contribute $$3\rho(0)+\Strue n_1(1+z)^3$$. The three copies of $$X_{\lvert\Seff\rvert/3}$$ contribute $$3\rho(0)+\lvert\Seff\rvert n_1(1+z)^3-3\rho(\lvert\Seff\rvert/3)$$. The sum is $$6\rho(0)-3\rho(\lvert\Seff\rvert/3)+(\Strue+\lvert\Seff\rvert)n_1(1+z)^3$$, which is the density of the three negative-mass species, $$3[2\rho(0)-\rho(\lvert\Seff\rvert/3)]$$ by (4), plus the extra dust $$\delta(1+z)^3$$ of the fit. $$\square$$
</div>

We checked the corollary numerically for $$\Seff=-0.05,-0.10,-0.20\eV$$ on a grid of $$1.6\times10^4$$ points in $$\ln(1+z)$$ from $$z=0$$ to $$10^6$$: the two densities agree to a relative accuracy of $$10^{-15}$$, and each $$X_\mu$$ is non-negative and non-decreasing with $$0\le w\le0.333$$ for the masses we tried ($$\mu=1$$ to $$200\meV$$). At $$\Seff=-0.10\eV$$ the total $$X$$ amounts to $$1.14\%$$ of the stable matter density at $$z=1060$$. It also exceeds $$\delta_{\min}$$ by 2.3% ($$\delta=1.707\times10^{-3}$$ against $$1.669\times10^{-3}$$), as Theorem 2 requires.

$$X_\mu$$ is matter that turns into radiation on the schedule on which a neutrino of mass $$\mu$$ becomes non-relativistic, run in reverse. At the level of the background it is the exact form of the decaying dark matter of [[LK]](#ref-lk). It violates only the hypothesis that the CDM seen at recombination is stable. The corollary is a statement about the background. It does not say that a CMB analysis of this universe returns $$\omega_c+\delta$$ for the cold matter density, since that depends on how $$X_\mu$$ clusters, which $$H(z)$$ does not determine.

## 5. Fit conventions and curvature

*The split of the fitted mass.* Theorem 1 uses the degenerate fit of (H4). A fit that puts all of $$\Seff$$ in one species and leaves two massless, as some analyses do, is not majorised by the truth. At $$z=0$$ the truth has two non-relativistic species and one massless, the fit one non-relativistic and two massless; a massless species has $$j(0)=\tfrac43\rho(0)=4.20\,n_1T_{\nu0}$$, whereas a non-relativistic one has $$j\simeq m\,n_1$$, so $$\Delta(0)\simeq(\Strue-\Seff-4.20\,T_{\nu0})\,n_1$$ and $$\Delta(0)<0$$ when the gap is below $$4.20\,T_{\nu0}=0.71\meV$$. We computed $$\max_z\Delta$$ numerically for the minimal normal ordering and a fit with $$\Seff\ge0$$ in one species: it is negative, attained at $$z=0$$, if and only if $$\Strue-\Seff<0.671\meV$$ (the difference from $$0.71\meV$$ is the thermal energy of the $$8.7\meV$$ species). For gaps below this the fluid of the proof of Theorem 2 with $$\delta=0$$ is an NEC-respecting exact mimic. The conclusion of Theorem 1 therefore depends on the fit convention, at a level far below any observational resolution.

*Curvature and the matching of $$H(z)$$.* Spatial curvature acts as a fluid with $$w=-\tfrac13$$ and density $$\omega_K(1+z)^2$$, where $$\omega_K=\Omega_Kh^2$$ and $$\Omega_K>0$$ is open, and (2) acquires the term $$\tfrac23\omega_K(1+z)^2$$ on the right. Suppose the truth has curvature, the fit is flat, and only $$H(z)$$ is matched. For $$\Omega_K\ge0$$ the curvature term is non-negative, the sum over the $$X_a$$ is more negative than in Lemma 2, and Theorem 1 holds unchanged. For $$\Omega_K<0$$ an NEC-respecting mimic exists if and only if $$\lvert\omega_K\rvert\ge\tfrac32\sup_{z\in I}\Delta(z)/(1+z)^2$$, the proof being that of Theorem 2. Numerically the supremum is at the upper end of the interval, $$z=1060$$, and for $$h=0.68$$ it gives $$\lvert\Omega_K\rvert\ge0.056,\ 0.072,\ 0.081,\ 0.186$$ for $$\Seff=0.0488,\ 0.03,\ 0,\ -0.10\eV$$. These values lie many standard deviations beyond the constraint $$\Omega_K=0.0023\pm0.0011$$ quoted in section 7. If the distances are matched as well as $$H(z)$$, the premise cannot be satisfied with $$\Omega_K\ne0$$: equal $$H$$ gives equal comoving distance $$\chi$$ to every redshift, but the transverse distance is $$\sinh$$ or $$\sin$$ of it for the curved model and $$\chi$$ for the flat one.

## 6. The floor under DESI DR2 weights

The data constrain a finite set of distances, not the whole function $$H(z)$$, so the question becomes how low $$\Seff$$ can be when the truth contains only NEC-respecting extra components. We define $$\SBAO$$ as the best-fit $$\Seff$$ of the flat $$\Lambda$$CDM$$+\Seff$$ model fitted to BAO data and a compressed CMB prior without CMB lensing. The calculation is a numerical one. The set-up, which every number in this section and the next refers to, is as follows.

The BAO data are the thirteen numbers of the DESI DR2 table with the correlation coefficients of the table [[DESI DR2]](#ref-desi-dr2). The CMB enters as a Gaussian prior on $$(\theta_\ast,\omega_b,\omega_{cb})$$ with the covariance of appendix A of the same paper (their (A2)), centred on the values of the true model, so that the mock data are noiseless. The recombination redshift is from the fit of Hu and Sugiyama [[HS]](#ref-hs). The sound horizon $$r_d$$ is taken from the fitting formula of Aubourg et al. [[Aubourg]](#ref-aubourg) (their (16)), or in a variant by integrating $$c_s/H$$ above the drag redshift of Eisenstein and Hu [[EH]](#ref-eh), rescaled by a constant factor $$1.039$$ so that the two treatments agree at the fiducial point; the variant makes $$r_d$$ respond to any extra component present before $$\zd$$. The neutrinos have exact Fermi–Dirac thermodynamics with the continuation (4). The fit parameters are $$(\omega_b,\omega_c,h,\Seff)$$. The true model is flat with $$\omega_b=0.02223$$, $$\omega_{cb}=0.14208$$ (the centre of the prior), $$h=0.68$$ and the minimal normal ordering. When extra components are added, $$h$$ is re-solved so that $$\theta_\ast$$ of the truth is unchanged, which is how the early universe is held fixed. We replaced the covariance of (A2) by uncorrelated Gaussians of width $$3.1\times10^{-6}$$, $$1.5\times10^{-4}$$ and $$1.2\times10^{-3}$$ on $$(\theta_\ast,\omega_b,\omega_c)$$ and the results quoted below change by less than $$0.1\meV$$ in the floor and by 1% in the slope of section 7. A full-likelihood analysis is not part of this set-up.

*The floor for $$\Lambda$$CDM.* For the truth without extra components, $$\Seff=0.06379\eV$$ with the formula of Aubourg et al. and $$0.06410\eV$$ with the integrated $$r_d$$; it moves by less than $$0.1\meV$$ when $$h$$ is varied between $$0.66$$ and $$0.70$$. The mock's forecast uncertainty (the inverse Fisher matrix) is $$0.087\eV$$, so this is a statement about the central value. The excess of $$5.0\meV$$ over $$\Strue=58.78\meV$$ is the convexity gap of section 3: the fit imposes equal masses on a hierarchical spectrum. A fit that uses the true spectrum returns $$\Strue$$. For other truths the same procedure gives

| truth | $$\Strue$$ (meV) | fitted $$\Seff$$ (meV) |
|---|---|---|
| normal ordering, lightest $$0$$ | 58.78 | 63.79 |
| normal ordering, lightest $$10\meV$$ | 74.34 | 75.89 |
| normal ordering, lightest $$30\meV$$ | 119.64 | 119.86 |
| inverted ordering, lightest $$0$$ | 99.51 | 104.01 |
| degenerate | any | $$\Strue$$ |

so the floor is the fitted value for the truth without extra components, which is at least $$\Strue$$; for $$\Strue=0$$ it is $$0$$, and the number $$0.0638\eV$$ refers to the minimal normal ordering.

*The response to an NEC component.* An uncoupled NEC component has a density that is non-decreasing in redshift. It is a sum of downward steps in redshift,

$$
\rho_X(z)=\rho_X(0)+\int\Theta(z-z')\,d\lambda(z'),\qquad d\lambda\ge0,
$$

with $$\Theta$$ the unit step; the case of a constant is the integral with $$\lambda=0$$. To first order $$\SBAO=\SBAO[\Lambda{\rm CDM}]+\int\mathcal K(z')\,d\lambda(z')$$, where the kernel $$\mathcal K(z')$$ is the change of $$\SBAO$$ per unit step at $$z'$$. We computed $$\mathcal K$$ with smoothed steps (width $$0.4\%$$ in $$\ln(1+z)$$), in units of the critical density today for $$z'<100$$ and of $$\varepsilon\equiv\Delta\rho_X/\rho_m(z=1060)$$, the step as a fraction of the matter density at the drag redshift, for $$z'\ge200$$ (small amplitudes, so that the response is linear):

| $$z'$$ | 0.1 | 0.3 | 0.6 | 1.0 | 1.6 | 2.5 | 4 | 10 |
|---|---|---|---|---|---|---|---|---|
| $$\mathcal K$$, Aubourg $$r_d$$ (eV) | 0.19 | 0.65 | 0.81 | 0.79 | 0.39 | 0.14 | 0.042 | 0.0027 |
| $$\mathcal K$$, integrated $$r_d$$ (eV) | 0.20 | 0.69 | 0.86 | 0.82 | 0.40 | 0.14 | 0.043 | 0.0027 |

| $$z'$$ | 200 | 300 | 400 | 500 | 600 | 800 | 1000 |
|---|---|---|---|---|---|---|---|
| $$\mathcal K$$, Aubourg $$r_d$$ (eV) | 31 | 5.1 | −0.04 | −1.6 | −2.2 | −2.6 | −2.7 |
| $$\mathcal K$$, integrated $$r_d$$ (eV) | 32 | 6.0 | 0.84 | −0.71 | −1.3 | −1.7 | −1.8 |

On 143 values of $$z'$$ spaced geometrically between $$0.005$$ and $$300$$, $$\mathcal K\ge0$$ for both treatments of $$r_d$$, apart from values of order $$10^{-5}\eV$$ of either sign at $$z'\approx60$$–$$100$$, where the response is itself of that size and below the numerical resolution. Beyond $$z'=300$$ the kernel decreases through zero, at $$z'=398$$ with the formula of Aubourg et al. and at $$z'=441$$ with the integrated $$r_d$$, and its most negative values are $$-2.7\eV$$ and $$-1.9\eV$$ per unit $$\varepsilon$$, near $$z'=1050$$–$$1100$$.

<div class="remark" markdown="1">
**Numerical result.** In the set-up above, take a true model with the minimal normal ordering and extra components that are separately conserved with $$\rho+p\ge0$$ and whose total density is constant for $$z\ge300$$. Then, to first order in the amplitude of the extra components, $$\SBAO\ge0.0638\eV$$ (and $$\ge0.0641\eV$$ with the integrated $$r_d$$). If the density instead increases between $$z=300$$ and $$z=1060$$ by a fraction $$\varepsilon$$ of $$\rho_m(1060)$$, then to first order $$\SBAO\ge0.0638\eV-\kappa\,\varepsilon$$ with $$\kappa\approx2.7\eV$$ (Aubourg $$r_d$$) or $$1.9\eV$$ (integrated $$r_d$$); a step of $$\varepsilon=0.01$$ at $$z'=800$$ gives $$\SBAO=0.0376\eV$$ and $$0.0467\eV$$ in the full non-linear fit, and a step of $$\varepsilon=0.003$$ gives $$0.0560\eV$$ and $$0.0589\eV$$.
</div>

The condition "constant above $$z=300$$" is the conservative one: the first-order result needs only $$\mathcal K\ge0$$ over the support of $$d\lambda$$, which holds up to $$z'\approx400$$ for the formula of Aubourg et al. and up to $$440$$ for the integrated $$r_d$$, but $$\mathcal K(400)$$ is already slightly negative with the first treatment. We did not test whether an extra component of the shape and size of the last example, which adds one per cent of the matter density at $$z=1060$$, is compatible with the full CMB spectra.

*Nonlinear checks.* Beyond first order we computed $$\SBAO$$ with the Aubourg formula for single steps of amplitude $$0.1$$, $$0.3$$, $$1$$ and $$3$$ times the critical density at $$z'=0.1,0.3,1,3,10,100,250$$ (in the cases where the re-solved $$h$$ stays within $$0.4$$–$$1$$), and for 44 random histories of two to five steps with $$z'$$ spread logarithmically between $$0.01$$ and $$300$$ and amplitudes up to three critical densities. All values lie at or above the floor: the smallest value in the random histories is $$0.06384\eV$$, against $$0.06379\eV$$ without extra components, and the single-step values range from $$0.0638\eV$$ to $$1.6\eV$$.

*Dependence on the weighting of the data.* The positivity of $$\mathcal K$$ at low redshift is a property of the DR2 weights and not of the NEC. If only the six $$D_H$$ measurements are used, $$\mathcal K(0.1)=-0.69\eV$$ and $$\mathcal K(0.3)=-1.78\eV$$ per unit critical density, while with the six $$D_M$$ measurements alone $$\mathcal K$$ is positive at the same redshifts (0.25 and 0.65). Data weighted more towards $$D_H$$ could lie below the floor with an NEC component at low redshift.

*The data.* The same set-up applied to the DR2 numbers, with the prior centred on the values of appendix A of the DESI paper and $$100\theta_\ast=1.0411$$ for the CMB acoustic scale of Planck [[Planck 2018]](#ref-planck18) (our $$\theta_\ast$$ is rescaled by $$1.0013$$ to reproduce it), returns $$\SBAO=-0.152\pm0.087\eV$$ with $$\chi^2=10.7$$ for the thirteen BAO numbers and the three prior entries; with the uncorrelated Planck-like widths the uncertainty is $$0.077\eV$$. Moving $$100\theta_\ast$$ between $$1.0410$$ and $$1.0412$$ moves the central value between $$-0.156$$ and $$-0.148\eV$$. The central value lies $$2.5\sigma$$ below the floor of $$0.0638\eV$$ for Gaussian errors. The comparison with the DESI result $$-0.101\eV$$ is not like for like, since that number includes CMB lensing, which is the main source of the preference for negative values [[GM]](#ref-gm), whereas $$\SBAO$$ excludes it.

## 7. Curvature and the BAO distances

Section 5 showed that a curved truth can reproduce a flat $$H(z)$$ without violating the NEC only if it is closed with $$\lvert\Omega_K\rvert\gtrsim0.06$$. BAO also measure $$D_M$$, which for $$\Omega_K>0$$ is

$$
D_M=\frac{c}{H_0\sqrt{\Omega_K}}\sinh\!\Big(\sqrt{\Omega_K}\,\frac{H_0}{c}\,\chi\Big),\qquad\chi=c\int_0^z\frac{dz'}{H(z')},
$$

and a flat fit cannot absorb the hyperbolic sine into $$H(z)$$. We take a curved $$\Lambda$$CDM truth with $$\omega_b$$, $$\omega_{cb}$$, $$\theta_\ast$$ and (through the formula of Aubourg et al.) $$r_d$$ held fixed, and fit flat $$\Lambda$$CDM$$+\Seff$$ in the set-up of section 6:

| $$\Omega_K$$ | −0.002 | −0.001 | −0.0005 | 0 | +0.0005 | +0.001 | +0.002 | +0.0023 |
|---|---|---|---|---|---|---|---|---|
| $$\SBAO$$ (eV) | 0.190 | 0.128 | 0.097 | 0.064 | 0.030 | −0.019 | −0.092 | −0.113 |

The derivative at $$\Omega_K=0$$ is $$d\SBAO/d\Omega_K\approx-66.6\eV$$ from symmetric differences with steps $$10^{-4}$$ and $$5\times10^{-4}$$ ($$-66.6$$ and $$-66.8\eV$$; with the uncorrelated prior widths $$-66.2$$ and $$-66.5\eV$$). A symmetric difference with step $$10^{-3}$$ gives $$-73\eV$$, and a step of $$2\times10^{-3}$$ gives $$-71\eV$$, because the response is not linear over such steps: it crosses $$\SBAO=0$$ at $$\Omega_K=9.2\times10^{-4}$$. The model's dependence on the mass is quadratic near $$\Sigma=0$$ (the continuation (4) has $$\rho(m)=\rho(0)+c\,m\lvert m\rvert$$, which is $$C^1$$ with vanishing slope at $$m=0$$), so the data lose sensitivity to $$\SBAO$$ there and the fitted value changes quickly with any shift in the data; the one-sided slopes on either side of $$\Omega_K=9.2\times10^{-4}$$ are $$-210$$ and $$-510\eV$$. The effect comes from the hyperbolic sine. Entering the same curvature in a flat $$H(z)$$ as a fluid with $$w=-\tfrac13$$ gives a positive response, in agreement with the kernel of section 6.

DESI DR2 combined with the CMB gives $$\Omega_K=+0.0023\pm0.0011$$ when curvature is a free parameter of $$\Lambda$$CDM [[DESI DR2]](#ref-desi-dr2) (their Table V), $$2.1\sigma$$ from flat. At that value the table gives $$\SBAO=-0.113\eV$$. Open curvature of the size DESI reports therefore accounts, in the set-up of section 6, for the whole BAO-channel value $$-0.152\eV$$ except for about $$0.04\eV$$, without violating any energy condition; this agrees with the degeneracy noted in [[GGM]](#ref-ggm). The result is a map from a curved truth to the value a flat fit would return, and it does not claim that the data prefer curvature: DESI reports no significant preference for a non-flat universe.

## 8. What the theorem does not exclude

Theorem 1 and its numerical counterpart concern a flat universe with stable CDM, the same $$\omega_b$$ and $$\omega_c$$ in the truth and the fit, standard neutrinos and a degenerate fit. A negative BAO-inferred mass sum is possible when one of these is relaxed, as follows. The page does not choose among them.

1. A violation of the null energy condition. By Theorem 1 this is required if all the other hypotheses hold and the match of $$H(z)$$ is exact. Phantom dark energy, components with negative density, and the non-dynamical sources discussed in [[GGM]](#ref-ggm) (section 5.1) are of this kind. Section 3 gives the size: $$\lvert1+w\rvert\sim2\times10^{-3}$$ to $$5\times10^{-3}$$ at $$z=0$$ for the values $$\Seff=0$$ to $$-0.10\eV$$ if the component is the dark energy.

2. A fit CDM density above the stable one. By Theorem 2 this needs $$\delta\ge\delta_{\min}\approx(\Strue-\Seff)/93\eV$$, that is, about $$1.7\times10^{-3}$$ in $$\omega_c$$ for $$\Seff=-0.10\eV$$, carried by matter that clusters at recombination and is not stable afterwards: decay to radiation, conversion to dark energy or a coupling between dark matter and dark energy. Proposition 3 is a decay that obeys every energy condition. The condition does not fix how such matter clusters, and the lensing channel depends on that.

3. Spatial curvature. With $$\Omega_K$$ of the size DESI reports, the BAO distances of a flat fit change by an amount equivalent to $$\Delta\SBAO\approx-66\eV\times\Omega_K$$ at small $$\Omega_K$$ (section 7). Theorem 1 itself is unaffected by open curvature when only $$H(z)$$ is matched.

4. A change of the expansion history between $$z\approx300$$ and recombination, or of $$r_d$$. In the numerical set-up an increase of the density of an NEC component with redshift in this window lowers $$\SBAO$$ by up to $$2.7\eV$$ per unit $$\varepsilon$$ (section 6); whether such a component is compatible with the CMB spectra has not been tested here.

5. Data that weight $$D_H$$ more than DR2 does. With the six $$D_H$$ measurements alone the response to an NEC component at low redshift is negative.

6. A non-degenerate fit convention, or true neutrinos that are not standard Fermi–Dirac species: decay, cooling or time-dependent masses, as discussed in [[CGMR]](#ref-cgmr). Section 5 gives the former at the level of $$0.67\meV$$.

7. Any resolution that acts through the lensing channel. The statements above are about the expansion history, and the full DESI+CMB value of $$-0.101\eV$$ includes CMB lensing. Whether an exact background mimic is also a mimic of the lensing spectrum is not determined by $$H(z)$$. The sign of energy exchange between a vacuum component and matter, which enters item 2 when the conversion is a coupling to dark energy, is treated in the [sign theorem](/quantum-gravity/vacuum-matter-sign/); the [dark-sector parameters page](/quantum-gravity/dark-sector-moduli/) and the [landscape](/quantum-gravity/landscape/) place this page among the other dark-sector results.

## 9. Open points

The exact criterion of Theorem 2 concerns the background. Whether the CMB temperature and polarisation spectra of the universe in Corollary 2 return a cold matter density $$\omega_c+\delta$$, and which lensing amplitude they have, requires a perturbation calculation with a specified sound speed and viscosity for $$X_\mu$$; we did not do it. The positivity of $$\mathcal K$$ on $$0<z'<300$$ and the first-order floor rest on a numerical response computed on a finite set of redshifts with a compressed likelihood and fitting formulae for $$z_\ast$$ and $$r_d$$; replacing the covariance of (A2) by uncorrelated widths changes the floor by less than $$0.1\meV$$, and a full-likelihood analysis has not been done. The nonlinear checks are samples, not a proof that no NEC history lies below the floor. The window above $$z\approx400$$ in which the response has the opposite sign is characterised only at the level of single steps. The compressed prior is centred on the true model, so the floor is a property of the best-fit value and not of a statistical bound: with the forecast uncertainty of $$0.087\eV$$, a fluctuation of the data below the floor is not excluded.

## References

1. <a id="ref-desi-dr2"></a>DESI Collaboration (M. Abdul-Karim et al.), "DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints," 2025, [arXiv:2503.14738](https://arxiv.org/abs/2503.14738).
2. <a id="ref-desi-nu"></a>W. Elbers et al. (DESI Collaboration), "Constraints on Neutrino Physics from DESI DR2 BAO and DR1 Full Shape," 2025, [arXiv:2503.14744](https://arxiv.org/abs/2503.14744).
3. <a id="ref-ggm"></a>P. W. Graham, D. Green, J. Meyers, "New Interpretations of the Cosmological Preference for a Negative Neutrino Mass," 2025, [arXiv:2508.20999](https://arxiv.org/abs/2508.20999).
4. <a id="ref-lk"></a>G. P. Lynch, L. Knox, "What's the matter with $$\Sigma m_\nu$$?", 2025, [arXiv:2503.14470](https://arxiv.org/abs/2503.14470).
5. <a id="ref-elbers"></a>W. Elbers, C. S. Frenk, A. Jenkins, B. Li, S. Pascoli, "Negative neutrino masses as a mirage of dark energy," Phys. Rev. D 111, 063534 (2025), [arXiv:2407.10965](https://arxiv.org/abs/2407.10965).
6. <a id="ref-gm"></a>D. Green, J. Meyers, "The Cosmological Preference for Negative Neutrino Mass," Phys. Rev. D 111, 083507 (2025), [arXiv:2407.07878](https://arxiv.org/abs/2407.07878).
7. <a id="ref-cgmr"></a>N. Craig, D. Green, J. Meyers, S. Rajendran, "No $$\nu$$s is Good News," JHEP 2024, 97 (2024), [arXiv:2405.00836](https://arxiv.org/abs/2405.00836).
8. <a id="ref-nufit"></a>I. Esteban, M. C. Gonzalez-Garcia, M. Maltoni, I. Martinez-Soler, J. P. Pinheiro, T. Schwetz, "NuFit-6.0: Updated global analysis of three-flavor neutrino oscillations," JHEP 12 (2024) 216, [arXiv:2410.05380](https://arxiv.org/abs/2410.05380).
9. <a id="ref-planck18"></a>Planck Collaboration, "Planck 2018 results. VI. Cosmological parameters," Astron. Astrophys. 641, A6 (2020), [arXiv:1807.06209](https://arxiv.org/abs/1807.06209).
10. <a id="ref-aubourg"></a>E. Aubourg et al., "Cosmological implications of baryon acoustic oscillation (BAO) measurements," Phys. Rev. D 92, 123516 (2015), [arXiv:1411.1074](https://arxiv.org/abs/1411.1074).
11. <a id="ref-hs"></a>W. Hu, N. Sugiyama, "Small Scale Cosmological Perturbations: An Analytic Approach," Astrophys. J. 471, 542 (1996), [arXiv:astro-ph/9510117](https://arxiv.org/abs/astro-ph/9510117).
12. <a id="ref-eh"></a>D. J. Eisenstein, W. Hu, "Baryonic Features in the Matter Transfer Function," Astrophys. J. 496, 605 (1998), [arXiv:astro-ph/9709112](https://arxiv.org/abs/astro-ph/9709112).
