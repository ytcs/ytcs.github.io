---
title: What a negative neutrino mass in the BAO data would require
date: 2026-10-28
order: 6
description: In a flat universe whose cold dark matter is stable, no component obeying the null energy condition can make the expansion history mimic a lighter neutrino. Under DESI DR2 weights the BAO-inferred mass sum has a floor of 0.0638 eV.
---

Neutrino oscillation experiments require the three neutrino masses to sum to at least $$0.0588\eV$$. When cosmological data are fitted with $$\Lambda$$CDM plus a free mass sum $$\Seff$$ that is allowed to take negative values, the preferred value is below zero: DESI DR2 combined with the CMB gives $$\Seff=-0.101^{+0.047}_{-0.056}\eV$$ [[DESI ν]](#ref-desi-nu). A large part of this preference comes from CMB lensing. This note concerns the part that comes from the expansion history alone, which is what baryon acoustic oscillations (BAO) measure; fitted to DR2 BAO with a compressed CMB prior, it is $$\SBAO=-0.153\pm0.077\eV$$ (section 5).

We ask what the dark sector would have to do for a $$\Lambda$$CDM fit to the expansion history to return a mass sum below the true one. In a flat universe in which the cold dark matter (CDM) is stable and is the same CDM that the CMB measures, no collection of components obeying the null energy condition can reproduce the expansion history of a smaller neutrino mass sum, at any redshift (section 3). Relaxing the assumptions of this statement opens four escapes: violation of the null energy condition, matter that is counted as CDM at recombination but later ceases to be matter, closed spatial curvature large enough to be excluded by CMB lensing and BAO, and a different convention for how the fit distributes the mass among the three species (section 4).

With DESI DR2’s actual BAO weights the question becomes quantitative (section 5). Adding a dark sector that obeys the null energy condition and is constant above $$z\approx300$$ raises the fitted mass sum, to first order and in all nonlinear cases we tried. The minimum is $$0.0638\eV$$, the value obtained for $$\Lambda$$CDM itself, which lies $$5\meV$$ above the true sum because the fit imposes equal masses. Open curvature, by contrast, lowers the fitted value substantially through the BAO distances (section 6).

Much of the physical picture is in the literature. Graham, Green and Meyers [[GGM]](#ref-ggm) separate the BAO-inferred mass sum from the clustering-inferred one and show that non-phantom dark energy makes the BAO-inferred value more negative. They also note that curvature is degenerate with it. Lynch and Knox [[Lynch–Knox]](#ref-lynch-knox) propose dark matter decaying into dark radiation as a way to obtain less late-time matter than the CMB infers. This note adds the exact theorem, a fluid that realises the decaying-matter escape exactly at the level of the background, and the floor $$0.0638\eV$$ with the conditions under which it holds.

The reader is assumed to be familiar with the Friedmann equations, Fermi–Dirac statistics, and the use of BAO as distances in units of the sound horizon.

## 1. Matching $$H(z)$$ fixes $$\rho+p$$

The argument rests on one identity. In a flat FRW universe the Friedmann equation $$H^2=\tfrac{8\pi G}{3}\rho_{\rm tot}$$ and energy conservation $$\dot\rho_{\rm tot}=-3H(\rho_{\rm tot}+p_{\rm tot})$$ combine to give

$$
\frac{dH^2}{dz}=\frac{8\pi G}{1+z}\sum_i(\rho_i+p_i). \tag{1}
$$

If two flat models have the same $$H(z)$$ on an interval of redshift, their derivatives agree there, and by (1) so does their total $$\rho+p$$ at every redshift in the interval. Their energy densities may differ, but their $$\rho+p$$ may not.

The null energy condition (NEC), $$\rho+p\ge0$$, is accordingly the one to test. When a dark sector makes one expansion history reproduce another, its $$\rho+p$$ is fixed by the comparison, and the only question is the sign of that fixed value.

## 2. Neutrinos and negative masses

A decoupled neutrino keeps a Fermi–Dirac momentum distribution whose temperature redshifts as $$T(z)=T_{\nu0}(1+z)$$. We take $$T_{\nu0}=0.1682\meV$$, the instantaneous-decoupling value $$(4/11)^{1/3}T_{\gamma0}=0.1676\meV$$ scaled by $$(N_{\rm eff}/3)^{1/4}$$ with $$N_{\rm eff}=3.044$$, so that the neutrino energy density matches the standard value. For one species of mass $$m\ge0$$, counting particle and antiparticle, let $$E=\sqrt{q^2+m^2}$$, $$f(q)=(e^{q/T}+1)^{-1}$$ and $$\int\equiv\frac{1}{\pi^2}\int_0^\infty q^2\,dq$$. The energy density and pressure are

$$
\rho(m)=\int E\,f,\qquad p(m)=\int\frac{q^2}{3E}\,f,
$$

and, by section 1, the relevant combination is

$$
j(m)\equiv\rho(m)+p(m)=\int\Big(E+\frac{q^2}{3E}\Big)f. \tag{2}
$$

At early times, $$T\gg m$$, the species behaves as radiation. At late times, $$T\ll m$$, it is matter with density $$m\,n_1(z)$$, where $$n_1(z)=n_1(0)(1+z)^3$$ is the number density of one species. In the units $$\omega\equiv\Omega h^2$$, three species of total mass $$\Sigma$$ contribute $$\omega_\nu=\Sigma/93.14\eV$$ at late times. At fixed early-universe parameters, a larger mass sum means more matter at low redshift, a larger $$H(z)$$ there and shorter distances. This is the dependence through which BAO measure the mass sum.

A fit can return a negative mass sum only if the model is extended below zero. The standard extension [[Craig et al.]](#ref-craig) [[Elbers et al.]](#ref-elbers) [[Green–Meyers]](#ref-green-meyers) is antisymmetric about the massless value: for $$m<0$$,

$$
\rho(m)\equiv2\rho(0)-\rho(\lvert m\rvert),\qquad p(m)\equiv2p(0)-p(\lvert m\rvert).
$$

A negative mass removes as much late-time energy, relative to massless neutrinos, as the corresponding positive mass adds. We write $$\tilde\jmath(m)=2j(0)-j(\lvert m\rvert)$$ for the extended $$j$$ when $$m<0$$.

## 3. The exact statement

We work under the following assumptions, which section 4 relaxes one at a time.

1. The universe is flat FRW and total energy is conserved.
2. The true model and the fit have the same photon density and $$N_{\rm eff}$$, and the same baryon and CDM densities $$\omega_b,\omega_c$$. Baryons and CDM are stable and separately conserved, so that $$\rho_b=\omega_b(1+z)^3$$ and $$\rho_c=\omega_c(1+z)^3$$ in both.
3. The true neutrinos are standard Fermi–Dirac species with masses $$m_i\ge0$$ and sum $$\Strue$$.
4. The fit is flat $$\Lambda$$CDM$$+\Seff$$ with three equal masses $$\Seff/3$$, extended antisymmetrically when $$\Seff<0$$.
5. The true model may contain further components $$X_a$$. They need not be separately conserved, and their densities may be negative. They satisfy the NEC if $$\sum_a(\rho_a+p_a)\ge0$$.

Densities are in units in which $$H^2=\sum_i\rho_i$$ with $$H$$ in $$100\,{\rm km\,s^{-1}Mpc^{-1}}$$. The redshift range of interest is $$0<z<\zd$$, where $$\zd\approx1060$$ is the drag redshift at which baryons decouple from photons. BAO distances are measured in this range, while earlier epochs are fixed by the CMB. The proofs below do not use the upper limit.

Comparing the two models through (1) gives the following identity.

<div class="lemma" markdown="1">
**Lemma 1.** Under assumptions 1–5, suppose $$H_{\rm true}=H_{\rm fit}$$ on an open interval $$I$$. Define the neutrino deficit

$$
\Delta(z)\equiv\sum_{i=1}^3 j\big(m_i;T(z)\big)-3\,\tilde\jmath\big(\Seff/3;T(z)\big),
$$

the true neutrinos’ $$\rho+p$$ minus that of the fitted neutrinos. Then $$\sum_a(\rho_a+p_a)(z)=-\Delta(z)$$ for every $$z\in I$$.
</div>

<div class="proof" markdown="1">
*Proof.* By (1) the two models have equal total $$\rho+p$$ on $$I$$. The fit contains baryons, CDM, photons, its neutrinos and $$\Lambda$$, for which $$\rho+p=0$$. The true model contains baryons, CDM, photons, its neutrinos and the $$X_a$$. By assumption 2 the baryon, CDM and photon terms coincide and cancel.
</div>

The extra components violate the NEC exactly when $$\Delta>0$$, that is, when the true neutrinos carry more $$\rho+p$$ than the fitted ones. Two properties of $$j$$ determine the sign of $$\Delta$$: monotonicity, which handles a fitted sum that is too small, and convexity, which handles unequal true masses compared with equal fitted ones.

<div class="lemma" markdown="1">
**Lemma 2.** At every $$T>0$$, $$j$$ is strictly increasing and strictly convex on $$m\ge0$$, and $$\tilde\jmath$$ is strictly increasing on $$m\le0$$.
</div>

<div class="proof" markdown="1">
*Proof.* Differentiation under the integral sign is justified by the exponential decay of $$f$$. Since $$\partial_mE=m/E$$,

$$
\partial_mj=\int\frac{m}{E}\Big(1-\frac{q^2}{3E^2}\Big)f,
$$

which is positive for $$m>0$$ because $$q^2<E^2$$. Using $$\partial_m(m/E)=q^2/E^3$$ and $$\partial_m(q^2m/3E^3)=q^2/3E^3-q^2m^2/E^5$$,

$$
\partial_m^2j=\int\Big(\frac23\frac{q^2}{E^3}+\frac{q^2m^2}{E^5}\Big)f=\int\frac{q^2(2q^2+5m^2)}{3E^5}\,f>0,
$$

including at $$m=0$$, where the integrand is $$\tfrac23q^{-1}f$$ against the measure $$q^2\,dq$$. For $$m<0$$, $$\tilde\jmath\,'(m)=j'(\lvert m\rvert)>0$$.
</div>

<div class="theorem" markdown="1">
**Theorem.** Under assumptions 1–5, if $$H_{\rm true}=H_{\rm fit}$$ on an open interval $$I$$ and $$\Seff<\Strue$$, then $$\sum_a(\rho_a+p_a)<0$$ at every $$z\in I$$. The extra components violate the NEC throughout $$I$$.
</div>

<div class="proof" markdown="1">
*Proof.* By Lemma 1 it suffices to show $$\Delta(z)>0$$ for each $$z\in I$$. Fix $$z$$, and hence $$T$$. If $$0\le\Seff<\Strue$$, monotonicity gives $$3j(\Seff/3)<3j(\Strue/3)$$, and convexity with Jensen’s inequality gives $$3j(\Strue/3)\le\sum_ij(m_i)$$. If $$\Seff<0$$, monotonicity of $$\tilde\jmath$$ gives $$3\tilde\jmath(\Seff/3)<3j(0)\le\sum_ij(m_i)$$, where the last step uses $$m_i\ge0$$.
</div>

The conclusion holds pointwise in $$z$$, and it does not require the extra components to be separately conserved, since only their total $$\rho+p$$ enters. If they are separately conserved, each satisfies $$d\rho_a/dz=3(\rho_a+p_a)/(1+z)$$, so their total density decreases with redshift: it grows as the universe expands, as phantom dark energy does. At low redshift, $$z\lesssim1$$, where the neutrinos are non-relativistic, it is approximately $$\rho_X=C-(\Strue-\Seff)\,n_1(z)$$ for a constant $$C$$: a cosmological constant minus the missing neutrino rest mass. The neglected terms, mainly the radiation density of the massless species, are about $$0.02(1+z)$$ times $$(\Strue-\Seff)\,n_1$$.

## 4. Relaxing the assumptions

### Matter that ceases to be matter

The second half of assumption 2 states that the cold matter the CMB measures at recombination is the same stable CDM that is present today. Suppose instead that the fit’s CDM density is $$\omega_c+\delta$$ while the true stable CDM density is $$\omega_c$$. The fit then contains an additional dust term $$\delta(1+z)^3$$, whose $$\rho+p$$ is also $$\delta(1+z)^3$$, and Lemma 1 becomes $$\sum_a(\rho_a+p_a)=\delta(1+z)^3-\Delta(z)$$.

<div class="corollary" markdown="1">
**Corollary.** Under assumptions 1, 3, 4, 5, and assumption 2 with $$\omega_c^{\rm fit}=\omega_c+\delta$$, there exists a true model whose extra components obey the NEC and which reproduces $$H_{\rm fit}$$ on $$I$$ if and only if

$$
\delta\;\ge\;\delta_{\min}\equiv\sup_{z\in I}\frac{\Delta(z)}{(1+z)^3}. \tag{3}
$$

</div>

<div class="proof" markdown="1">
*Proof.* Necessity follows from the identity above. For sufficiency, take a single separately conserved fluid with $$\rho_X=\rho_\Lambda+\delta(1+z)^3+\rho_\nu^{\rm fit}-\rho_\nu^{\rm true}$$, whose $$\rho+p$$ is $$\delta(1+z)^3-\Delta\ge0$$.
</div>

For a minimal normal-ordering spectrum, $$m=(0,\ 8.614,\ 50.150)\meV$$ with $$\Strue=58.76\meV$$, and $$I=(0,\zd)$$:

| $$\Seff$$ (eV) | 0 | −0.05 | −0.10 | −0.20 |
|---|---|---|---|---|
| $$\delta_{\min}$$ | $$6.2\times10^{-4}$$ | $$1.13\times10^{-3}$$ | $$1.67\times10^{-3}$$ | $$2.74\times10^{-3}$$ |
| $$(\Strue-\Seff)/93.14\eV$$ | $$6.3\times10^{-4}$$ | $$1.17\times10^{-3}$$ | $$1.70\times10^{-3}$$ | $$2.78\times10^{-3}$$ |

The second row is the missing neutrino rest mass in $$\omega$$ units, a late-time estimate that agrees with the exact value to within 1–3% for $$\Seff\le0$$. At $$\Seff=-0.10\eV$$ the required $$\delta$$ is about 1.4 times the Planck uncertainty on $$\omega_c$$, which is approximately $$0.0012$$. A shift of this size in a fitted $$\omega_c$$ would be unremarkable; here, however, $$\delta$$ must be physical matter that clusters at recombination and later ceases to be matter.

The following fluid has exactly this property.

<div class="definition" markdown="1">
**Definition.** For $$\mu>0$$, let

$$
X_\mu(z)\equiv\rho(0;z)+\mu\,n_1(z)-\rho(\mu;z), \tag{4}
$$

the density of a massless neutrino, plus the rest-mass density of a species of mass $$\mu$$, minus the density of a neutrino of mass $$\mu$$.
</div>

<div class="proposition" markdown="1">
**Proposition 1.** As a separately conserved perfect fluid, $$X_\mu$$ is non-negative and non-decreasing in $$z$$, and it satisfies the null, weak and dominant energy conditions, with $$0\le w\le\tfrac13$$. It behaves as dust ($$X_\mu\simeq\mu n_1$$) for $$T\gg\mu$$ and as radiation ($$X_\mu\to\rho(0)$$) for $$T\ll\mu$$.
</div>

<div class="proof" markdown="1">
*Proof.* Each term in (4) is separately conserved, so the pressure is $$p_X=p(0)-p(\mu)$$. Change to comoving momentum $$q$$, so that $$f=(e^{q/T_{\nu0}}+1)^{-1}$$ is independent of $$z$$, and let $$x=\mu a$$ with $$a=1/(1+z)$$. Then $$X_\mu=a^{-4}g(x)$$, where

$$
g(x)=\frac{1}{\pi^2}\int_0^\infty f\,q^2\big[q+x-\sqrt{q^2+x^2}\big]\,dq\;\ge\;0,
$$

because $$\sqrt{q^2+x^2}\le q+x$$. Monotonicity, $$dX_\mu/da\le0$$, is equivalent to $$x\,g'(x)\le4g(x)$$. This holds if it holds for each integrand, that is, if $$x\big(1-x/\sqrt{q^2+x^2}\big)\le4\big(q+x-\sqrt{q^2+x^2}\big)$$. Rearranged, the condition is $$4q^2+3x^2\le(4q+3x)\sqrt{q^2+x^2}$$; both sides are positive, and the inequality follows from

$$
(4q+3x)^2(q^2+x^2)-(4q^2+3x^2)^2=q^2x^2+24q^3x+24qx^3\;\ge\;0.
$$

The continuity equation then gives $$\rho_X+p_X=\tfrac13(1+z)\,dX_\mu/dz\ge0$$. Next, $$p(\mu)=\int\tfrac{q^2}{3E}f\le\int\tfrac q3f=p(0)$$, so $$p_X\ge0$$. Finally, $$p_X\le X_\mu/3$$ is equivalent to $$\rho(\mu)-3p(\mu)=\int\tfrac{\mu^2}{E}f\le\mu n_1$$, which holds because $$E\ge\mu$$.
</div>

Consider a true model with stable CDM $$\omega_c$$, minimal normal-ordering neutrinos and $$X=\sum_iX_{m_i}+3X_{\lvert\Seff\rvert/3}$$, and a fit with three masses $$\Seff/3<0$$ and $$\omega_c^{\rm fit}=\omega_c+(\Strue+\lvert\Seff\rvert)\,n_1(0)$$. Omitting the baryons, photons and $$\Lambda$$ common to both, the neutrinos and $$X$$ of the true model sum to

$$
\sum_i\big[\rho(m_i)+X_{m_i}\big]+3X_{\lvert\Seff\rvert/3}=\big[3\rho(0)+\Strue n_1\big]+\big[3\rho(0)+\lvert\Seff\rvert n_1-3\rho(\lvert\Seff\rvert/3)\big],
$$

while the neutrinos and additional CDM of the fit sum to $$\big[6\rho(0)-3\rho(\lvert\Seff\rvert/3)\big]+(\Strue+\lvert\Seff\rvert)\,n_1$$. The two are equal at every redshift. Hence $$H(z)$$ and the sound horizon $$r_d$$ coincide exactly, for any $$\Seff<0$$, and so does the CMB acoustic angle $$\theta_\ast=r_s/D_M(z_\ast)$$, where $$r_s$$ is the sound horizon at recombination, $$z_\ast\approx1090$$ (distinct from $$r_d$$ at $$\zd$$). For $$\Seff=-0.10\eV$$, $$X$$ amounts to about 1.1% of the matter density at $$\zd$$.

The background identity is exact. Whether a CMB analysis would return $$\omega_c^{\rm fit}$$ for this true model is a separate, approximate question. In the construction the fit’s excess is $$\delta=(\Strue+\lvert\Seff\rvert)\,n_1(0)$$, which is $$1.70\times10^{-3}$$ at $$\Seff=-0.10\eV$$ (the second row of the table above). At $$\zd$$ the actual density of $$X$$ equals 92–96% of this rest-mass density for $$-0.20\eV\le\Seff\le-0.05\eV$$, and 95.3% at $$\Seff=-0.10\eV$$. If the CMB measures the density of $$X$$ as clustering matter, it infers an excess of $$1.63\times10^{-3}$$, slightly below $$\delta_{\min}=1.67\times10^{-3}$$. By (3), a fit with that excess has no exact NEC-respecting mimic. Under this reading the construction reproduces the expansion history exactly but the combined CMB and BAO fit only approximately.

$$X_\mu$$ describes matter that converts into radiation on the same schedule on which a neutrino of mass $$\mu$$ becomes non-relativistic, run in reverse. At the level of the background it is the exact form of the decaying dark matter of Lynch and Knox, whose decay rate tracks the neutrino transition [[Lynch–Knox]](#ref-lynch-knox). It satisfies every energy condition and violates only assumption 2, since part of what the CMB counts as CDM is not stable.

### The fit’s mass convention

Assumption 4, that the fit uses three equal masses, also matters, although only at a level far below observational resolution. Suppose instead that the fit places all its mass in one species and leaves two massless. At $$z=0$$ a massless species has $$j(m{=}0;T_{\nu0})=\tfrac43\rho(m{=}0;T_{\nu0})$$, and $$\rho(0)/n_1=7\pi^4T/(180\zeta(3))\approx3.15\,T$$, so $$j(m{=}0;T_{\nu0})\approx4.20\,n_1T_{\nu0}$$, whereas a non-relativistic species has $$j\approx m\,n_1$$. Where the true model has a non-relativistic species and the fit a massless one, $$\Delta$$ at $$z=0$$ is positive only if $$\Strue-\Seff$$ exceeds approximately $$4.20\,T_{\nu0}\approx0.71\meV$$. For a minimal normal-ordering truth the threshold evaluates numerically to $$0.67\meV$$. Below it, $$\Delta<0$$ at every redshift (the supremum of $$\Delta$$ is approached as $$z\to0$$), so the fluid used in the proof of the Corollary, with $$\delta=0$$, reproduces the fit while respecting the NEC.

In the other direction the theorem extends beyond its stated form. The proof applies to any non-negative fit split that can be raised species by species to a spectrum of total mass $$\Strue$$ that is majorised by the true spectrum, meaning the true masses are at least as unequal. Karamata’s inequality then replaces Jensen’s.

### Curvature

Suppose flatness is dropped and only $$H(z)$$ is matched. Curvature acts as a fluid with $$w=-\tfrac13$$ and adds $$(\rho+p)_K=\tfrac23\omega_K(1+z)^2$$ to the identity of Lemma 1, where $$\omega_K=\Omega_Kh^2$$ and $$\Omega_K>0$$ corresponds to an open universe.

- For open curvature the additional term is non-negative and the theorem holds unchanged.
- If $$H(z)$$ is matched at all redshifts and $$\theta_\ast$$ is matched as well, the premise cannot be satisfied with $$\Omega_K\ne0$$. Equal $$H$$ at all redshifts gives the same $$r_s$$ and the same comoving distance $$\chi_\ast$$ to recombination, but with curvature the transverse distance $$D_M(z_\ast)$$ differs from $$\chi_\ast$$.
- For closed curvature with $$\theta_\ast$$ free, an NEC-respecting mimic requires $$\lvert\omega_K\rvert\ge\sup_z\tfrac32\Delta(z)/(1+z)^2$$, with the supremum attained at $$\zd$$. With $$h=0.68$$ this gives $$\lvert\Omega_K\rvert\ge0.056,\ 0.072,\ 0.081,\ 0.19$$ for $$\Seff=0.0488$$ (10 meV below $$\Strue$$), $$0.03$$, $$0$$ and $$-0.10\eV$$. These values are compatible with the primary CMB alone, which prefers $$\Omega_K=-0.044^{+0.018}_{-0.015}$$ [[Planck 2018]](#ref-planck18), but are excluded once CMB lensing or BAO are included.

Curvature thus produces an exact mimic only if it is closed and larger than about $$0.05$$ in magnitude ($$0.08$$ or more for $$\Seff\le0$$). Its effect on the fitted mass sum through the BAO distances is nonetheless large, as section 6 shows.

## 5. The floor under DESI DR2 weights

Data do not require an exact match of $$H(z)$$, only an acceptable fit to a finite set of BAO measurements. We now ask how low the fitted mass sum can be when the true model contains only NEC-respecting extra components.

We define $$\SBAO$$ as the best-fit $$\Seff$$ when flat $$\Lambda$$CDM$$+\Seff$$ is fitted to BAO data together with a compressed CMB prior, without CMB lensing. The BAO data are the 13 DESI DR2 measurements of $$D_M/r_d$$, $$D_H/r_d$$ and $$D_V/r_d$$, with their covariance [[DESI DR2]](#ref-desi-dr2); here $$D_H=c/H$$, $$D_M$$ is the transverse comoving distance, and $$D_V=(zD_M^2D_H)^{1/3}$$. The compressed prior is an uncorrelated Gaussian on $$(\theta_\ast,\omega_b,\omega_c)$$ with Planck-level uncertainties, in place of the full Planck likelihood. The recombination redshift and $$r_d$$ are taken from standard fitting formulae (Hu–Sugiyama and Aubourg et al., respectively); a variant integrates $$r_d$$ directly. Neutrinos are treated with exact Fermi–Dirac thermodynamics. The fitted parameters are $$(\omega_b,\omega_c,h,\Seff)$$, with three equal masses. The true model has minimal normal ordering, $$\Strue=0.0588\eV$$, and the fit is to noiseless mock data generated from it, so that any shift in $$\SBAO$$ comes from the model.

An uncoupled NEC-respecting component has $$\rho(z)$$ non-decreasing in $$z$$. It is a non-negative superposition of downward steps,

$$
\rho_X(z)=\rho_X(0)+\int\Theta(z-z')\,d\lambda(z'),\qquad d\lambda\ge0,
$$

with $$\Theta$$ the unit step function. To first order the fitted mass sum responds as $$\SBAO=\SBAO[\Lambda{\rm CDM}]+\int\mathcal K(z')\,d\lambda(z')$$, where the kernel $$\mathcal K(z')$$ is the change in $$\SBAO$$ per unit step at $$z'$$. Where $$\mathcal K\ge0$$, every such history raises the fitted mass sum.

We find $$\mathcal K\ge0$$ on two independent grids of 21 and 143 points spanning $$0.005\le z'\le300$$. At higher redshift the kernel decreases through zero and becomes negative. With steps measured in units of the matter density at $$\zd$$:

| step redshift $$z'$$ | 300 | 400 | 600 | 800 | 1000 |
|---|---|---|---|---|---|
| $$\mathcal K$$, Aubourg $$r_d$$ (eV) | +5.06 | −0.06 | −2.17 | −2.57 | −2.69 |
| $$\mathcal K$$, integrated $$r_d$$ (eV) | +6.08 | +0.93 | −1.21 | −1.61 | −1.73 |

For comparison, $$\mathcal K(200)=+30.9\eV$$ with the Aubourg formula.

<div class="remark" markdown="1">
**Numerical result.** In the setup above, let the true model be minimal normal ordering plus extra components that are separately conserved with $$\rho+p\ge0$$ and constant for $$z\ge\zd$$, so that the early universe is unchanged. Let $$\varepsilon\equiv[\rho_X(\zd)-\rho_X(300)]/\rho_m(\zd)$$ be the drop in the extra density between $$z=\zd$$ and $$z=300$$, in units of the matter density at $$\zd$$. Then, to first order in $$\varepsilon$$,

$$
\SBAO\;\ge\;0.0638\eV-\kappa\,\varepsilon,
$$

where $$\kappa\approx1.7\eV$$ (integrated $$r_d$$) to $$2.7\eV$$ (Aubourg $$r_d$$) is the magnitude of the kernel near $$z'=1000$$. For an extra sector that is constant above $$z\approx300$$, $$\SBAO\ge0.0638\eV$$, the value obtained for $$\Lambda$$CDM itself.
</div>

The floor exceeds the true value $$0.0588\eV$$ by $$5.0\meV$$ because the fit imposes three equal masses on a hierarchical spectrum; a fit with the true spectrum gives $$0.0588\eV$$. Beyond first order, we searched over single steps of up to three times the critical density, random histories of two to five steps, and binned NEC equations of state $$w(z)$$, and minimised directly. No value below the floor appears; the minimum, $$0.06377\eV$$, occurs at $$w\equiv-1$$, that is, for $$\Lambda$$CDM.

Applied to the DR2 measurements themselves, the same setup gives $$\SBAO=-0.153\pm0.077\eV$$, about $$2.8\sigma$$ below the floor for Gaussian errors.

At higher redshift the floor can be crossed without violating the NEC. A step $$\rho_X=\rho_\Lambda+\varepsilon\,\rho_m(\zd)\,\Theta(z-800)$$, such as a scalar field frozen until $$z\approx800$$ and then released into kinetic-energy domination would produce, gives $$\SBAO=0.048\eV$$ (integrated $$r_d$$) or $$0.038\eV$$ (Aubourg $$r_d$$) at $$\varepsilon=0.01$$. We have not tested whether an early component of this shape and size is compatible with the full Planck spectra.

The floor depends on DR2’s balance of $$D_M$$ and $$D_H$$: with the $$D_H$$ measurements alone, the kernel is negative for steps below their redshifts, so data weighted differently could fall below it. It also depends on the true spectrum. The result is stated for minimal normal ordering; for $$\Strue=0$$ the floor is trivially zero, and we have checked numerically, but not proved, that $$\SBAO[\Lambda{\rm CDM}]$$ increases with $$\Strue$$. The compressed likelihood adds a systematic uncertainty of approximately $$\pm3\meV$$.

## 6. Curvature and the BAO distances

Section 4 showed that curvature reproduces $$H(z)$$ exactly only for closed values excluded by lensing and BAO. BAO, however, also measure $$D_M$$, which for open curvature is

$$
D_M=\frac{c}{H_0\sqrt{\Omega_K}}\sinh\!\Big(\sqrt{\Omega_K}\,\frac{H_0}{c}\,\chi\Big),\qquad \chi=\int_0^z\frac{c\,dz'}{H(z')}.
$$

A flat fit cannot absorb the $$\sinh$$ into $$H(z)$$. Consider a curved $$\Lambda$$CDM truth with $$\omega_b$$, $$\omega_b+\omega_c$$, $$\theta_\ast$$ and $$r_d$$ held fixed, fitted with flat $$\Lambda$$CDM$$+\Seff$$:

| $$\Omega_K$$ | −0.002 | −0.001 | −0.0005 | 0 | +0.0005 | +0.001 | +0.002 |
|---|---|---|---|---|---|---|---|
| $$\SBAO$$ (eV) | +0.187 | +0.127 | +0.096 | +0.064 | +0.030 | −0.018 | −0.090 |

The slope at $$\Omega_K=0$$ is $$d\SBAO/d\Omega_K\approx-65\eV$$: symmetric differences with steps $$10^{-4}$$ and $$5\times10^{-4}$$ give $$-65.2$$ and $$-65.5\eV$$. A step of $$10^{-3}$$ gives $$-72\eV$$ because it crosses $$\SBAO=0$$, where the antisymmetric extension has a kink. The effect comes entirely from the $$\sinh$$. Including the same $$\Omega_K$$ in a flat $$H(z)$$ as a fluid with $$w=-\tfrac13$$ instead raises $$\SBAO$$, consistent with the positive kernel of section 5.

DESI DR2 combined with the CMB gives $$\Omega_K=+0.0023\pm0.0011$$ [[DESI DR2]](#ref-desi-dr2), which corresponds to $$\SBAO\approx-0.11\eV$$. Open curvature about $$2\sigma$$ from flat accounts for most of the DR2 BAO-channel value without any violation of an energy condition, as also noted by Graham, Green and Meyers [[GGM]](#ref-ggm).

## 7. Discussion

The DR2 value $$\SBAO=-0.153\pm0.077\eV$$ lies about $$2.8\sigma$$ below the floor of section 5. In the setup of that section, the cases studied produce a value below the floor only by relaxing one of the hypotheses of the numerical result: through violation of the NEC; decay or conversion, after recombination, of part of the CDM measured by the CMB, or energy exchange between dark matter and dark energy; open spatial curvature; a change to the expansion history between $$z\approx300$$ and recombination, or to $$r_d$$; or data weighted towards $$D_H$$ more heavily than DR2. The results do not discriminate among these possibilities.

The results constrain which dark sectors can make a $$\Lambda$$CDM$$+\Seff$$ fit return a value below the true mass sum; they say nothing about the true masses themselves. The numerical results of sections 5 and 6 rest on noiseless mock data, a compressed CMB prior and fitting formulae, not on the full Planck and DESI likelihoods. An exact background mimic also leaves the lensing channel undetermined, since lensing depends on how the extra sector clusters. For decaying dark matter, Graham, Green and Meyers argue that the clustering-inferred mass sum moves in the opposite direction to the BAO-inferred one [[GGM]](#ref-ggm). Whether an opposite response of the two channels holds for NEC-respecting sectors in general is open.

## References

1. <a id="ref-desi-dr2"></a>DESI Collaboration, “DESI DR2 results II: baryon acoustic oscillations and cosmological constraints,” [arXiv:2503.14738](https://arxiv.org/abs/2503.14738).
2. <a id="ref-desi-nu"></a>DESI Collaboration, “Constraints on neutrino physics from DESI DR2 BAO and DR1 full shape,” [arXiv:2503.14744](https://arxiv.org/abs/2503.14744).
3. <a id="ref-ggm"></a>Graham, Green, Meyers, [arXiv:2508.20999](https://arxiv.org/abs/2508.20999).
4. <a id="ref-lynch-knox"></a>Lynch, Knox, [arXiv:2503.14470](https://arxiv.org/abs/2503.14470).
5. <a id="ref-craig"></a>Craig, Green, Meyers, Rajendran, “No νs is good news,” [arXiv:2405.00836](https://arxiv.org/abs/2405.00836).
6. <a id="ref-elbers"></a>Elbers et al., “Negative neutrino masses as a mirage of dark energy,” [arXiv:2407.10965](https://arxiv.org/abs/2407.10965).
7. <a id="ref-planck18"></a>Planck Collaboration, “Planck 2018 results. VI. Cosmological parameters,” [arXiv:1807.06209](https://arxiv.org/abs/1807.06209).
8. <a id="ref-green-meyers"></a>Green, Meyers, “The cosmological preference for negative neutrino mass,” [arXiv:2407.07878](https://arxiv.org/abs/2407.07878).
