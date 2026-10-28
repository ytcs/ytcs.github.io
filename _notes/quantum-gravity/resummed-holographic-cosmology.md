---
title: The low multipoles of holographic cosmology at large flavour number
date: 2026-10-28
order: 7
description: In holographic cosmology with conformally coupled matter, the leading large-flavour resummation of the dual stress-tensor correlator fixes the power spectrum at the largest angular scales with no new parameter. Positivity of the spectrum then bounds the number of non-conformal scalars and excludes the explicit solution published with the model, and a fit to Planck temperature data is about two standard deviations worse than the standard cosmological model.
---

In holographic cosmology the primordial power spectrum is computed from the stress-tensor two-point function of a three-dimensional SU($$N$$) gauge theory with many scalar and fermion fields. The published two-loop form fits the Planck temperature spectrum at multipoles $$\ell\ge30$$, but it is not valid at lower $$\ell$$, which were therefore excluded from the fits. We ask what the same theory predicts for $$\ell<30$$. When all scalars are conformally coupled and quartic and Yukawa couplings are negligible, the answer is fixed at leading order in the number of fields: the gauge-field contribution to the trace correlator, with both gluon lines dressed by the chain of matter loops, can be computed exactly. Its expansion reproduces the published two-loop coefficient $$64/(3\pi^2)$$, and below a crossover scale, corresponding to $$\ell_c\approx3$$–$$4$$ for the fitted parameters, the logarithm that drives the two-loop running stops growing. The low-$$\ell$$ spectrum then follows from the three parameters already fitted at $$\ell\ge30$$, with no new parameter.

Positivity of the power spectrum at all scales bounds the number of non-conformal scalar units, in a model that extends the calculation to that case, by $$N_B\le0.864\,\beta$$, that is, $$N_B\lesssim1.6$$–$$2.9$$ over the published one-standard-deviation ranges of $$\beta$$; this excludes the explicit solution with $$N_B\approx88$$ given with the model. Fitted to Planck temperature data at all multipoles, with a compressed likelihood, the resummed spectrum is worse than $$\Lambda$$CDM by $$\Delta(-2\ln\mathcal L)\approx3$$–$$5$$, depending on the treatment of the background cosmology, and predicts a quadrupole $$0.57$$ times the $$\Lambda$$CDM value. Finally, the same parameters predict suppressed low-$$\ell$$ $$E$$-mode polarization with unsuppressed tensor modes, a sum rule among the scalar tilt and its first two runnings, a positive effective running, and a $$\mu$$-type spectral distortion about $$1.3$$ times the $$\Lambda$$CDM value.

The framework is due to McFadden and Skenderis [[MS 2009]](#ref-ms09), [[MS 2010]](#ref-ms10). Afshordi, Corianò, Delle Rose, Gould and Skenderis [[ACDGS]](#ref-acdgs) reported the two-loop coefficients, whose computation Corianò, Delle Rose and Skenderis later gave in full [[CDS]](#ref-cds), and fitted the model to Planck 2015 data; they found the fit at all multipoles disfavoured relative to $$\Lambda$$CDM with running, and attributed this to the breakdown of perturbation theory at $$\ell\lesssim30$$. Afshordi, Gould and Skenderis [[AGS]](#ref-ags) repeated the analysis in more detail. The dressing of a three-dimensional gauge field by a large number of matter fields is standard; at the resulting infrared fixed point, Giombi, Tarnopolsky and Klebanov [[GTK]](#ref-gtk) compute the stress-tensor two-point coefficient of conformal QED to next-to-leading order in $$1/N$$. This note adds the full crossover of the trace correlator between the perturbative and infrared regimes, the resulting low-$$\ell$$ spectrum and its comparison with data, the positivity bound, and the predictions of section 7.

Familiarity with the holographic cosmology dictionary is not assumed beyond eq. (1) below; familiarity with one-loop computations in three-dimensional gauge theory and with CMB power spectra is.

## 1. The two-loop spectrum and where it fails

The dual theory is SU($$N$$) Yang–Mills theory in three Euclidean dimensions coupled to $$N_\Phi$$ real scalars and $$N_\psi$$ two-component Dirac fermions, all in the adjoint representation [[ACDGS]](#ref-acdgs). Scalars couple to the background curvature with parameters $$\xi_M$$; the value $$\xi_M=1/8$$ is conformal coupling in three dimensions. The theory is super-renormalizable, and its dimensionless coupling at momentum $$q$$ is the 't Hooft coupling $$\lambda=g_{\rm YM}^2N/q$$.

The trace $$\Theta=T^i{}_i$$ of the stress tensor carries the scalar perturbations. At large $$N$$ its two-point function has the form $$\langle\Theta(q)\Theta(-q)\rangle=N^2q^3f(\lambda)$$ up to local terms, and the holographic dictionary [[MS 2009]](#ref-ms09), in the form used in [[ACDGS]](#ref-acdgs) and [[CDS]](#ref-cds), gives the curvature power spectrum

$$
\Delta^2(q)=\frac{1}{4\pi^2N^2f(\lambda)}. \tag{1}
$$

The function $$f$$ is unchanged by the analytic continuation that relates the QFT to cosmology, so $$\lambda>0$$ throughout. In perturbation theory [[ACDGS]](#ref-acdgs), [[CDS]](#ref-cds),

$$
f(\lambda)=f_0\big[1-f_1\lambda\ln\lambda+f_2\lambda+O(\lambda^2)\big],\qquad f_0=\frac{N_B}{64},\qquad N_B=1+\sum_M(1-8\xi_M)^2. \tag{2}
$$

The "1" in $$N_B$$ is the gauge field; each scalar contributes $$(1-8\xi_M)^2$$, which vanishes for conformal coupling, and fermions do not contribute to $$f_0$$. With all $$\xi_M=1/8$$ and no Yukawa couplings, eq. (8) of [[ACDGS]](#ref-acdgs) gives

$$
f_1=-\frac{4}{3\pi^2}\Big(N_\psi+\frac{N_\Phi}{2}-2\Big),\qquad N_B=1.
$$

Writing $$gq_\ast=f_1g_{\rm YM}^2N$$ and $$\ln(1/\beta)=f_2/f_1+\ln\lvert f_1\rvert$$, with the pivot $$q_\ast=0.05\,{\rm Mpc}^{-1}$$, and expanding $$1/f$$ to first order, (1) becomes the form used in the fits [[ACDGS]](#ref-acdgs),

$$
\Delta^2(q)=\frac{\Delta_0^2}{1+u\ln u+u\ln\beta},\qquad u\equiv\frac{\lvert g\rvert q_\ast}{q},\qquad\Delta_0^2=\frac{1}{4\pi^2N^2f_0}. \tag{3}
$$

The fits find $$g<0$$, which requires $$f_1<0$$; the variable $$u=\lambda\lvert f_1\rvert$$ is positive. The three parameters $$(\Delta_0^2,g,\ln\beta)$$ replace the amplitude and tilt of $$\Lambda$$CDM. Fitted to Planck 2015 data at $$\ell\ge30$$, [[ACDGS]](#ref-acdgs) (Table 1) find $$g=-0.0131^{+0.0045}_{-0.0035}$$ and $$\ln\beta=1.01^{+0.21}_{-0.27}$$; the fit at all multipoles gives $$g=-0.0070^{+0.0011}_{-0.0017}$$ and $$\ln\beta=0.88^{+0.19}_{-0.24}$$.

Multipoles and wavenumbers are related by $$\ell\approx q\chi_\ast$$, with $$\chi_\ast=13\,873$$ Mpc the comoving distance to last scattering in the Planck 2018 cosmology. At the quadrupole, the $$\ell\ge30$$ fit has $$u\approx4.5$$, so (3) is outside its range of validity. Extrapolated anyway, it predicts a quadrupole $$0.32$$ times the $$\Lambda$$CDM value and worsens the low-$$\ell$$ likelihood by $$\Delta(-2\ln\mathcal L)=+23.6$$ (section 6). The question is what replaces it.

## 2. The expansion parameter is the number of fields

The two-loop term in (2) is large at low $$\ell$$ although the 't Hooft coupling there is small. The tensor-to-scalar ratio of the model is $$r=32N_B/N_A$$ with $$N_A=1+2N_\psi+N_\Phi$$ [[ACDGS]](#ref-acdgs), so the bound $$r<0.036$$ [[BK18]](#ref-bk18) requires $$N_A>889$$ when $$N_B=1$$. Since $$N_\psi+N_\Phi/2=(N_A-1)/2$$, the coefficient above is

$$
\lvert f_1\rvert=\frac{2}{3\pi^2}(N_A-5)\gtrsim60,
$$

and $$\lambda=u/\lvert f_1\rvert\le0.1$$ at $$\ell=2$$ for every parameter set used in this note. The two-loop correction is not small because $$f_1$$ counts all the matter fields, while $$f_0$$ counts only the gauge field.

This identifies the relevant limit. Each matter field contributes to the one-loop vacuum polarisation of the gluon. In three dimensions a massless two-component fermion of unit charge gives a self-energy $$e^2k/16$$ and a complex scalar the same, so that a real adjoint scalar contributes $$g_{\rm YM}^2Nk/32$$ and an adjoint fermion $$g_{\rm YM}^2Nk/16$$ (the adjoint index factor is $$N$$). Summing the chain of these bubbles gives the dressed gluon propagator, in a normalisation where the free propagator is $$1/k^2$$,

$$
D(k)=\frac{1}{k^2+q_ck},\qquad q_c=g_{\rm YM}^2N\Big(\frac{N_\Phi}{32}+\frac{N_\psi}{16}\Big)=g_{\rm YM}^2N\,\frac{N_A-1}{32}. \tag{4}
$$

Each bubble is suppressed by $$g_{\rm YM}^2$$ and enhanced by the number of fields, so at fixed $$q_c$$ the bubble chain is leading order in $$1/N_A$$. The ratio of the three-loop term in which one bubble is inserted into the two-loop diagram to the two-loop term itself is of order $$q_c/q$$. We therefore expand in $$1/N_A$$ at fixed

$$
x\equiv\frac{q_c}{q}=\kappa u,\qquad\kappa=\frac{q_c}{\lvert g\rvert q_\ast}=\frac{(N_A-1)/32}{\lvert f_1\rvert}=\frac{3\pi^2}{64}\,\frac{N_A-1}{N_A-5}. \tag{5}
$$

For $$N_A>889$$, $$\kappa$$ equals $$3\pi^2/64=0.4626$$ to within $$0.5\%$$, and we use that value. The crossover $$x=1$$ lies at $$\ell_c=\kappa\lvert g\rvert q_\ast\chi_\ast$$, which is $$4.2$$ for the published $$\ell\ge30$$ value of $$g$$.

Three assumptions define the setting from here on. (A1) All scalars are conformally coupled, $$\xi_M=1/8$$, so that $$N_B=1$$. (A2) Yukawa couplings vanish. (A3) Quartic couplings are small enough that their own bubble chains can be neglected. At $$\xi_M=1/8$$ the quartic couplings drop out of the two-loop $$f$$ (their contribution in eq. 9 of [[ACDGS]](#ref-acdgs) is proportional to $$(8\xi_{M_1}-1)(8\xi_{M_2}-1)$$), but a flavour-singlet quartic coupling would generate a chain of scalar bubbles at a scale proportional to $$\lambda_{\rm quartic}N_\Phi$$ and could alter the low-$$\ell$$ spectrum; (A3) excludes this.

## 3. The resummed trace correlator

Under (A1)–(A3) the scalars and fermions are classically conformal, up to the neglected quartic couplings, and the parts of $$\Theta$$ built from them vanish on the equations of motion; their insertions in correlators produce only contact terms. The trace is then the gauge-field term $$\Theta_A=\tfrac14F^a_{\mu\nu}F^a_{\mu\nu}$$ (in the normalisation of (4)), whose free two-point function is the "1" in $$N_B$$. The matter fields enter only through the dressing (4). At leading order in $$1/N_A$$ the trace correlator is therefore the one-loop diagram of $$\Theta_A$$ with both gluon lines replaced by $$D$$.

That no other diagram contributes at this order is a power-counting argument, not a proof: with the propagator (4), each additional gluon exchange or matter loop attached to the two gluon lines costs a factor $$g_{\rm YM}^2N/q_c\sim1/N_A$$ at every $$q$$, including $$q\ll q_c$$, where $$D\approx1/(q_ck)$$. The pure-gauge two-loop term, the "$$-2$$" in $$f_1$$, is of this kind. The counting fails when some scalar has $$\xi_M\ne1/8$$: then $$\Theta$$ has a matter part, the mixed correlators $$\langle\Theta_\Phi\Theta_A\rangle$$ are of the same order as the gauge loop, and they are not included here. This is why the controlled case is $$N_B=1$$.

### The first two terms

Before the full function we check the expansion in $$q_c$$, which can be done in closed form.

<div class="proposition" markdown="1">
**Proposition 1** (two-loop logarithm). For one colour, the gauge-field trace correlator with the propagator (4) has, to first order in $$q_c$$, the non-analytic part

$$
\langle\Theta_A\Theta_A\rangle(q)=\frac{q^3}{64}-\frac{q_c\,q^2\ln q}{3\pi^2}+\dots,
$$

where the dots stand for terms analytic in $$q$$ and terms of order $$q_c^2$$. Equivalently, the ratio $$\mathcal F=64\langle\Theta_A\Theta_A\rangle/q^3$$ obeys $$\mathcal F=1+a_1x\ln x+O(x)$$ with $$a_1=64/(3\pi^2)$$.
</div>

<div class="proof" markdown="1">
*Proof.* The correlator of $$F_{\mu\nu}$$ involves only the transverse part of the gluon propagator, so we may use $$\langle A_\mu(r)A_\nu(0)\rangle=\delta_{\mu\nu}G(r)$$ with $$G$$ the Fourier transform of $$D$$. Expanding $$D=1/k^2-q_c/k^3+O(q_c^2)$$ and transforming,

$$
G(r)=\frac{1}{4\pi r}+\frac{q_c}{2\pi^2}\ln r+\text{const}+O(q_c^2),
$$

since $$\nabla^2\ln r=1/r^2$$ and the transform of $$1/k$$ is $$1/(2\pi^2r^2)$$; the constant drops out of $$F_{\mu\nu}$$. Then $$\langle F_{\mu\nu}(r)F_{\rho\sigma}(0)\rangle=-(\partial_\mu\partial_\rho G\,\delta_{\nu\sigma}-\partial_\mu\partial_\sigma G\,\delta_{\nu\rho}-\partial_\nu\partial_\rho G\,\delta_{\mu\sigma}+\partial_\nu\partial_\sigma G\,\delta_{\mu\rho})$$, and by Wick's theorem $$\langle\Theta_A(r)\Theta_A(0)\rangle=\tfrac18\sum\langle F_{\mu\nu}F_{\rho\sigma}\rangle^2$$. Carrying out the sums gives

$$
\langle\Theta_A(r)\Theta_A(0)\rangle=\frac{3}{16\pi^2r^6}-\frac{q_c}{2\pi^3r^5}+O(q_c^2).
$$

In three dimensions the Fourier transform of $$r^{-2a}$$ is $$\pi^{3/2}2^{3-2a}\Gamma(\tfrac32-a)\Gamma(a)^{-1}q^{2a-3}$$. At $$a=3$$ this is $$\pi^2q^3/12$$, so the first term gives $$q^3/64$$. At $$a=\tfrac52$$ the Gamma function has a pole, whose residue multiplies $$q^2\ln q$$ with coefficient $$2\pi/3$$, while the pole itself is a local $$q^2$$ term. The second term therefore gives $$-q_cq^2\ln q/(3\pi^2)$$ plus a local term. At fixed $$q_c$$, $$x\ln q=x\ln q_c-x\ln x$$, and the first piece is analytic in $$q$$. Dividing by $$q^3/64$$ gives $$a_1=64/(3\pi^2)$$.
</div>

The free term reproduces the normalisation $$f_0=1/64$$ per unit of $$N_B$$. For the logarithm, translate back to (2): with $$x=\kappa\lambda\lvert f_1\rvert$$ and $$N_B=1$$, the term $$a_1x\ln x$$ contributes $$-f_1\lambda\ln\lambda$$ with $$-f_1=a_1(N_\Phi/32+N_\psi/16)=\tfrac{4}{3\pi^2}(N_\psi+N_\Phi/2)$$. This is the published coefficient, fermion and scalar terms separately, apart from the pure-gauge $$-2$$, which is of relative order $$1/N_A$$ and is not part of the leading order. The match is a check, not an input: $$a_1$$ comes from the gauge loop alone and the bubble coefficients $$1/16$$ and $$1/32$$ from the one-loop vacuum polarisation. It also shows that at $$\xi=1/8$$ the entire matter logarithm of the two-loop result comes from the self-energy insertion. The check concerns only the first order in $$x$$; the low-$$\ell$$ spectrum depends on $$x\gtrsim1$$, where the full function is needed.

### The contact term

The full calculation produces a term independent of $$q$$, which in $$\mathcal F$$ would appear as a multiple of $$x^3$$ and would dominate at small $$q$$. It does not enter the power spectrum. The reason is that the power spectrum is computed from the spin-0 form factor $$B(q)$$ in the decomposition $$\langle T_{ij}T_{kl}\rangle=A(q)\Pi_{ijkl}+B(q)\pi_{ij}\pi_{kl}$$ of the transverse part of the correlator [[CDS]](#ref-cds), where $$\pi_{ij}=\delta_{ij}-q_iq_j/q^2$$ and $$\Pi_{ijkl}=\tfrac12(\pi_{ik}\pi_{jl}+\pi_{il}\pi_{jk}-\pi_{ij}\pi_{kl})$$; away from contact terms $$\langle\Theta\Theta\rangle=4B$$.

<div class="lemma" markdown="1">
**Lemma 1** (no constant in $$B$$). A term in $$\langle T_{ij}T_{kl}\rangle$$ independent of $$q$$ does not contribute to $$B$$.
</div>

<div class="proof" markdown="1">
*Proof.* A term independent of $$q$$ is local and has no derivatives. By diffeomorphism invariance the only such term in the generating functional is $$c\int d^3x\sqrt{g}$$, whose second variation at flat space is

$$
c\Big[\tfrac14\delta_{ij}\delta_{kl}-\tfrac14(\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk})\Big].
$$

Projecting each index with $$\pi$$ gives $$c\big[\tfrac14\pi_{ij}\pi_{kl}-\tfrac14(\pi_{ik}\pi_{jl}+\pi_{il}\pi_{jk})\big]=-\tfrac{c}{2}\Pi_{ijkl}+0\cdot\pi_{ij}\pi_{kl}$$, so its transverse part lies entirely in $$A$$.
</div>

The constant is nonetheless present in the full trace $$\delta_{ij}\delta_{kl}\langle T_{ij}T_{kl}\rangle$$, with weight $$\tfrac34c$$; it belongs to the Ward identity involving $$\langle T_{ij}\rangle$$. We therefore remove it, which amounts to $$B(0)=0$$. Keeping it would add a term proportional to $$x^3$$ to $$\mathcal F$$, which would dominate at small $$q$$.

### The full crossover

The dressed propagator has a positive spectral representation,

$$
D(k)=\int_0^\infty d\mu\,\frac{\rho(\mu)}{k^2+\mu^2},\qquad\rho(\mu)=\frac{2}{\pi}\,\frac{q_c}{\mu^2+q_c^2},
$$

which reduces the dressed loop to a superposition of loops with two massive vector propagators. The sum over polarisations gives the numerator $$4[(k\cdot p)^2+k^2p^2]$$ for gluon momenta $$k$$ and $$p=q-k$$. Writing $$k\cdot p=\tfrac12(q^2-k^2-p^2)$$ and setting $$k^2=-\mu_1^2$$, $$p^2=-\mu_2^2$$ wherever a propagator is not cancelled, the remaining terms have one propagator cancelled and are polynomials in $$q$$ of degree at most 2. With the three-dimensional bubble $$\int\frac{d^3k}{(2\pi)^3}\frac{1}{(k^2+\mu_1^2)(p^2+\mu_2^2)}=\frac{\arctan(q/(\mu_1+\mu_2))}{4\pi q}$$,

$$
\mathcal F(q)=\frac{2}{\pi q^3}\int_0^\infty\!\!\int_0^\infty d\mu_1d\mu_2\,\rho(\mu_1)\rho(\mu_2)\Big[J(q)-J(0)-q^2J'(0)\Big],\quad J=\big[(q^2+\mu_1^2+\mu_2^2)^2+4\mu_1^2\mu_2^2\big]\frac{\arctan\frac{q}{\mu_1+\mu_2}}{q}, \tag{6}
$$

where $$J'(0)$$ denotes the coefficient of $$q^2$$ in the expansion of $$J$$ about $$q=0$$. The two subtractions remove the $$q^0$$ and $$q^2$$ terms of each massive loop. The first implements $$B(0)=0$$. The second removes a local term $$\propto q_cq^2$$, which in (2) is a contribution to $$f_2\lambda$$ and so to $$\ln\beta$$; its coefficient is a renormalisation constant, logarithmically divergent, and is fixed by matching to the fitted $$\ln\beta$$ in section 4. No other counterterm is added. The integral (6) converges absolutely, and as $$q_c\to0$$, $$\rho$$ tends to a delta function at $$\mu=0$$ and $$\mathcal F\to1$$.

We evaluated (6) by trapezoidal quadrature in $$\ln\mu_1,\ln\mu_2$$, with a series expansion of the subtracted integrand for $$q\ll\mu_1+\mu_2$$; doubling the grid changes $$\mathcal F$$ by less than $$10^{-12}$$. The results are numerical:

- At small $$x$$, a fit gives $$\mathcal F=1+a_1x\ln x+b\,x+O(x^2\ln^2x)$$ with $$a_1$$ equal to $$64/(3\pi^2)$$ to $$10^{-6}$$, as Proposition 1 requires, and $$b=1.98139$$. This $$b$$ agrees with $$\tfrac{11}{12}a_1$$ to six digits; we have not derived that value.
- $$\mathcal F$$ decreases monotonically from 1 to 0 and is positive for all $$x$$. Sample values are $$\mathcal F=0.920,\ 0.644,\ 0.199,\ 0.026$$ at $$x=0.01,\ 0.1,\ 1,\ 10$$.
- At large $$x$$, $$\mathcal F\approx0.274/x$$.

In this scheme the small-$$x$$ expansion can be written $$\mathcal F=1+a_1x\ln(x/x_\ast)+\dots$$ with

$$
x_\ast=e^{-b/a_1}=0.3998,
$$

and at large $$x$$ neither the constant 1 nor a term linear in $$x$$ survives. The disappearance of the constant is expected: for $$q\ll q_c$$ the gauge field becomes part of the conformal infrared theory of the matter fields, whose trace correlator has no $$q^3$$ term. The logarithm, in turn, stops growing near $$x=x_\ast$$. The combination $$x_\ast$$ does not depend on the renormalisation constant, because a change in that constant shifts $$b$$ and the large-$$x$$ linear term equally.

## 4. The resummed spectrum

The physical $$f$$ is $$f_0\mathcal F$$ plus the local term $$\propto q_cq^2$$, that is, plus a multiple of $$u$$. With $$N_B=1$$ and $$x=\kappa u$$, the multiple is fixed by requiring that the expansion in $$u$$ reproduce (3) to first order. Allowing for later use a general $$N_B$$, with the extra non-conformal scalars contributing only to the tree term and with $$\kappa=N_B/a_1$$ chosen so that the logarithm has unit coefficient, this gives

$$
\frac{f}{f_0}=1+\frac{\mathcal F(\kappa u)-1}{N_B}+u\ln\frac{\beta x_\ast}{\kappa},\qquad\kappa=\frac{N_B}{a_1}. \tag{7}
$$

For $$u\ll1$$, (7) reduces to $$1+u\ln u+u\ln\beta$$ up to $$O(u^2\ln^2u)$$. For $$u\gg1$$,

$$
\frac{f}{f_0}\to1-\frac{1}{N_B}+u\ln\frac{\beta x_\ast}{\kappa}:
$$

the logarithm $$\ln(\beta u)$$ of (3) is frozen at its value at $$x=x_\ast$$, which for $$N_B=1$$ is the scale $$q=q_c/x_\ast\approx2.5\,q_c$$. For $$N_B=1$$ the low-$$\ell$$ spectrum is therefore determined by $$(\Delta_0^2,g,\ln\beta)$$ alone, the parameters that $$\ell\ge30$$ data already measure. The colour number $$N$$ and the split between $$N_\Phi$$ and $$N_\psi$$ drop out at leading order. Equation (7) for $$N_B>1$$ is a model: it omits the mixed diagrams discussed in section 3.

At the published $$\ell\ge30$$ parameters and $$N_B=1$$, the resummed spectrum gives a quadrupole $$0.51$$ times the $$\Lambda$$CDM value, against $$0.32$$ for (3), and worsens the low-$$\ell$$ likelihood by $$+4.6$$, against $$+23.6$$ (section 6). The resummed and two-loop forms differ by more than 10% below $$\ell\approx33$$, so the $$\ell\ge30$$ fit itself must be redone with (7); this is done in section 6.

## 5. Positivity and the number of non-conformal scalars

A power spectrum is positive, so $$f$$ must be positive at every $$q$$, including scales far outside the observed range.

<div class="proposition" markdown="1">
**Proposition 2** (positivity bound). Assume, as found numerically in section 3, that $$\mathcal F>0$$ for all $$x$$ and $$\mathcal F\to0$$ as $$x\to\infty$$. Then the spectrum (7) is positive at all $$q$$ if and only if $$\kappa\le\beta x_\ast$$, that is,

$$
N_B\le a_1x_\ast\,\beta=0.864\,\beta. \tag{8}
$$

</div>

<div class="proof" markdown="1">
*Proof.* Each term of $$f/f_0=(1-1/N_B)+\mathcal F(\kappa u)/N_B+u\ln(\beta x_\ast/\kappa)$$ is non-negative when $$\ln(\beta x_\ast/\kappa)\ge0$$, and the second is strictly positive, so $$f>0$$. If $$\ln(\beta x_\ast/\kappa)<0$$, the last term tends to $$-\infty$$ as $$u\to\infty$$ while the others stay bounded, so $$f<0$$ at small enough $$q$$.
</div>

For $$N_B=1$$, (8) requires $$\ln\beta\ge0.146$$, which every published fit satisfies. For larger $$N_B$$, (8) is a statement within the model (7); its derivation assumes the omitted mixed diagrams do not change the infrared slope. Over the one-standard-deviation ranges of $$\ln\beta$$ in the two fits of [[ACDGS]](#ref-acdgs), $$0.64\le\ln\beta\le1.22$$, it gives $$N_B\le1.6$$ to $$2.9$$. At the all-$$\ell$$ refit of section 6, $$\ln\beta=1.11^{+0.16}_{-0.15}$$ and the bound is $$2.6$$ ($$2.3$$ to $$3.1$$). The violation, when it occurs, need not be at observable scales: at the refit parameters, $$N_B=3$$ makes $$f$$ negative only below $$\ell\approx1$$, and $$N_B=4$$ below $$\ell\approx3.5$$. Positivity is a consistency requirement on the theory at all scales, not a fit to data.

The explicit solution of [[ACDGS]](#ref-acdgs), eq. (12), has $$N=2995$$, $$N_\Phi=23\,255$$ scalars with common $$\xi\approx0.133$$ chosen so that $$r=0.12$$, and no fermions. Then $$\lvert1-8\xi\rvert=0.0613$$, $$N_B\approx88$$ and $$f_1=-11.0$$, as quoted there. Within (7), positivity would require $$\ln\beta\ge\ln(88/0.864)=4.6$$, far outside the fitted range and outside the prior range $$-0.2\le\ln\beta\le3.5$$ used in [[ACDGS]](#ref-acdgs). The conclusion does not depend on how the non-conformal scalars are treated. With the actual coupling of this solution, $$\kappa=q_c/(\lvert g\rvert q_\ast)=(N_\Phi/32)/\lvert f_1\rvert=66$$ instead of $$88/a_1=41$$: the gauge bubble then carries 161% of the two-loop logarithm and the mixed diagrams $$-61\%$$. If the whole logarithm freezes at $$x=x_\ast$$, positivity needs $$\ln\beta\ge\ln(66/x_\ast)=5.1$$; if the mixed share keeps running, the spectrum turns negative at small enough $$q$$ for any $$\beta$$. At the published $$\ell\ge30$$ parameters, the spectrum turns negative below $$\ell\approx33$$ in the model (7), and below $$\ell\approx37$$ or $$\ell\approx56$$ if the mixed share of the logarithm is taken to freeze with the gauge share or to keep running. In addition, the crossover of this solution lies at $$\ell_c=\kappa\lvert g\rvert q_\ast\chi_\ast\approx600$$, so the two-loop form on which its parameters were fitted is not valid anywhere in $$\ell\lesssim600$$. Its tensor-to-scalar ratio, $$0.12$$, is also excluded by $$r<0.036$$ [[BK18]](#ref-bk18).

## 6. Comparison with Planck temperature data

We compare (7) with $$N_B=1$$ to the Planck 2018 temperature power spectrum [[Planck V]](#ref-planck-v) with a compressed likelihood. The setup is as follows. For $$2\le\ell\le29$$ we use the exact full-sky likelihood of the Commander spectrum, scaled by a sky fraction, $$-2\ln\mathcal L=\sum_\ell(2\ell+1)f_{\rm sky}(\hat C_\ell/C_\ell-\ln(\hat C_\ell/C_\ell)-1)$$ with $$f_{\rm sky}=0.86$$; for $$30\le\ell\le800$$ a diagonal Gaussian $$\chi^2$$ with the published per-multipole errors. The background is fixed at the Planck 2018 best fit [[Planck VI]](#ref-planck-vi) with $$\tau=0.0544$$; $$C_\ell$$ are computed from CAMB transfer functions, which are linear in the primordial spectrum, and lensing is included as the $$\Lambda$$CDM ratio of lensed to unlensed spectra. $$\Lambda$$CDM has $$(A_s,n_s)$$ free; the holographic models have $$(\Delta_0^2,g,\ln\beta)$$ free, with positivity at all $$q$$ imposed. This is not the Planck likelihood, and only differences between models are meaningful.

| model | $$g$$ | $$\ln\beta$$ | $$\Delta(-2\ln\mathcal L)$$ total [$$\ell<30$$, $$\ell\ge30$$] | $$D_2/D_2^{\Lambda\rm CDM}$$ |
|---|---|---|---|---|
| resummed (7), $$N_B=1$$ | $$-0.0091$$ | $$1.11$$ | $$+5.0$$ [$$+2.1$$, $$+2.9$$] | $$0.57$$ |
| resummed (7), $$N_B=2$$ | $$-0.0124$$ | $$1.33$$ | $$+3.2$$ [$$+1.1$$, $$+2.1$$] | $$0.59$$ |
| two-loop form (3), all $$\ell$$ | $$-0.0065$$ | $$0.83$$ | $$+7.6$$ [$$+3.2$$, $$+4.4$$] | $$0.53$$ |
| resummed, $$N_B=1$$, fitted to $$\ell\ge30$$ only | $$-0.0174$$ | $$1.28$$ | $$+19.8$$ [$$+18.5$$, $$+1.3$$] | $$0.36$$ |

The differences are relative to $$\Lambda$$CDM, whose fit has $$n_s=0.964$$. Without the lensing factor all entries change by at most $$1.4$$; the first row becomes $$+5.4$$. The $$N_B=2$$ row requires one non-conformal unit, for which the resummation is incomplete, and is shown only for comparison.

For $$N_B=1$$ the refit has $$\ell_c=2.9$$, and the predicted temperature spectrum relative to $$\Lambda$$CDM at $$\ell=2,\dots,8$$ is $$0.57,\ 0.62,\ 0.68,\ 0.73,\ 0.78,\ 0.82,\ 0.85$$. The observed quadrupole is about $$0.22$$ of the $$\Lambda$$CDM value, so the model moves towards it without reaching it. The last row shows why the fit at all multipoles matters. The $$\ell\ge30$$ data constrain $$g$$ and $$\ln\beta$$ only along a degenerate direction, and different points along it predict very different low-$$\ell$$ spectra; fitting to $$\ell\ge30$$ alone selects a point that overshoots the suppression at low $$\ell$$.

![Panel (a): the resummed function F(x) falling monotonically from 1 at small x to 0 at large x, together with its first-order expansion, which agrees below x of about 0.03 and then turns upward. Panel (b): the temperature spectrum relative to LambdaCDM for 2 ≤ ℓ ≤ 50, with Planck 2018 data points and three curves: the two-loop form at the ℓ ≥ 30 parameters falling to 0.32 at the quadrupole, the resummed form at the same parameters falling to about 0.5, and the resummed form refitted at all multipoles falling to 0.57.](/assets/img/quantum-gravity/holographic-lowell.svg){: .invertible}

*Figure 1.* (a) The resummed gauge-field trace correlator (6) and its first-order expansion $$1+a_1x\ln(x/x_\ast)$$, as functions of $$x=q_c/q$$. (b) Temperature spectrum relative to the $$\Lambda$$CDM fit of section 6 at $$\ell\le50$$, with the Planck 2018 data. The two curves at the published $$\ell\ge30$$ parameters [[ACDGS]](#ref-acdgs) have their amplitude matched to $$\Lambda$$CDM over $$30\le\ell\le800$$; the solid curve is the $$N_B=1$$ refit.
{: .caption}


Part of the penalty at $$\ell\ge30$$ comes from fixing the background. Refitting both models on a $$3\times3$$ grid in $$(\omega_b,\omega_c)$$, with $$\omega_b\in[0.0219,0.0228]$$, $$\omega_c\in[0.1175,0.1225]$$ and the acoustic angle held fixed, and interpolating quadratically, the difference for $$N_B=1$$ becomes $$+4.2$$ with a Gaussian prior from the Planck 2018 values $$\omega_b=0.02237\pm0.00015$$, $$\omega_c=0.1200\pm0.0012$$ [[Planck VI]](#ref-planck-vi), and $$+3.0$$ minimised over the grid without a prior; the minimum then lies on the edge $$\omega_c=0.1175$$. The prior was derived within $$\Lambda$$CDM from data that include the temperature spectrum, so it favours $$\Lambda$$CDM slightly. Outside the grid the quadratic interpolation moves both models to $$\omega_c\approx0.11$$, where the difference nearly vanishes; temperature data to $$\ell=800$$ do not constrain $$\omega_c$$ well enough to make that extrapolation meaningful. Over the treatments within the grid, the resummed model is worse than $$\Lambda$$CDM by $$\Delta(-2\ln\mathcal L)=3.0$$ to $$5.0$$ while having one more primordial parameter. As a rough Gaussian equivalent, $$\sqrt{\Delta(-2\ln\mathcal L)}$$ is $$1.7$$ to $$2.2$$.

After the refit, the cost is shared between low and high multipoles: $$+2.1$$ at $$\ell<30$$, and $$+2.9$$ at $$\ell\ge30$$, mostly from an excess of $$1.3$$–$$1.7\%$$ over $$\Lambda$$CDM at $$\ell=30$$–$$50$$. Both are in a regime where the compressed likelihood is adequate in magnitude but not in detail. A fit with the full Planck temperature and polarization likelihood and free background is needed to turn these numbers into a significance.

## 7. Further predictions of the same parameters

### Low-multipole polarization

The suppression of the scalar spectrum at $$q\lesssim q_c$$ appears in the reionisation bump of the $$E$$-mode spectrum, which probes the largest scales. At the $$N_B=1$$ refit and $$\tau=0.0544$$, $$C_\ell^{EE}$$ relative to $$\Lambda$$CDM at $$\ell=2,\dots,8$$ is $$0.45,\ 0.56,\ 0.64,\ 0.69,\ 0.74,\ 0.79,\ 0.85$$.

Tensor modes behave differently. The tensor spectrum is inversely proportional to the analogous function $$f^T$$ of the transverse-traceless part of the correlator, with $$f^T_0=N_A/256$$ [[CDS]](#ref-cds), and the gauge field is one of the $$N_A$$ units in $$f^T_0$$. The crossover changes only that unit, so at leading order the tensor spectrum at low $$\ell$$ changes by a fraction of at most $$1/N_A<1.1\times10^{-3}$$. If primordial $$B$$ modes are detected at low $$\ell$$, the model predicts an unsuppressed reionisation bump in $$BB$$ together with a suppressed one in $$EE$$.

The polarization test is weak. For a cosmic-variance-limited measurement of $$EE$$ at $$2\le\ell\le30$$ over 70% of the sky, an idealisation of the low-$$\ell$$ measurement planned by LiteBIRD [[LiteBIRD]](#ref-litebird), and with $$\tau$$ refitted in $$\Lambda$$CDM at fixed $$A_se^{-2\tau}$$, the $$N_B=1$$ spectrum differs from the best $$\Lambda$$CDM by $$\Delta\chi^2=2.0$$ (best $$\tau=0.051$$), and by $$1.5$$ using $$\ell\le10$$ only. Foregrounds and noise would reduce this.

### A sum rule for the running

Write $$n_s-1$$, $$\alpha_s$$ and $$\beta_s$$ for the first three logarithmic derivatives of $$\ln\Delta^2$$ with respect to $$\ln q$$ (the tilt, its running and the running of the running).

<div class="proposition" markdown="1">
**Proposition 3** (running sum rule). To first order in $$u$$, the spectrum (3) satisfies

$$
n_s-1=-u(L-1),\qquad\alpha_s=u(L-2),\qquad\beta_s=-u(L-3),\qquad L=\ln\frac{q}{\beta\lvert g\rvert q_\ast},
$$

and hence

$$
\beta_s+2\alpha_s+(n_s-1)=0 \tag{9}
$$

at every $$q$$, for all values of $$g$$ and $$\beta$$.
</div>

<div class="proof" markdown="1">
*Proof.* To first order, $$\ln\Delta^2=\ln\Delta_0^2-u\ln(\beta u)=\ln\Delta_0^2+uL$$. Write $$\partial=d/d\ln q$$. Then $$\partial u=-u$$ and $$\partial L=1$$, so $$(\partial+1)u=0$$ and $$(\partial+1)(uL)=u$$, hence $$(\partial+1)^2(uL)=0$$. The left side of (9) is $$(\partial^3+2\partial^2+\partial)\ln\Delta^2=\partial(\partial+1)^2\ln\Delta^2$$, which vanishes because $$(\partial+1)^2$$ annihilates $$uL$$ and $$\partial$$ annihilates the constant. The individual expressions follow by differentiating $$uL$$ three times.
</div>

The resummation does not affect (9) at the pivot, where $$x\approx0.004$$. With the full forms (3) and (7), the left side of (9) at $$q_\ast$$ is at most $$7\times10^{-4}$$ in magnitude for all the parameter sets considered here, while the individual terms are of order $$10^{-2}$$. At the pivot, the $$N_B=1$$ refit has $$n_s=0.975$$, $$\alpha_s=+0.016$$ and $$\beta_s=-0.008$$; the published $$\ell\ge30$$ parameters give $$0.968$$, $$+0.020$$ and $$-0.009$$. The ratio $$\alpha_s/(n_s-1)=-(L-2)/(L-1)$$ depends on $$L$$, which is $$3.3$$–$$3.6$$ at the pivot, so the ratio is $$-0.57$$ to $$-0.65$$ at leading order. The value $$-1$$ quoted in [[AGS]](#ref-ags) is its limit for $$L\to\infty$$.

Equation (9) is a property of the two-loop form and holds for any parameters, so a measurement of $$(n_s,\alpha_s,\beta_s)$$ tests the form directly. Current errors on $$\beta_s$$, of order $$0.01$$, are too large for this.

### Effective running

A survey that fits $$\Lambda$$CDM with running to data generated by (7) measures an effective running $$\alpha_{\rm eff}$$, a weighted average of the local running over the scales it probes. Because the local running changes sign at $$L=2$$, near $$q\approx0.01\,{\rm Mpc}^{-1}$$ ($$\ell\approx140$$–$$190$$ for the parameter sets of section 6), and is large and negative at lower $$q$$, $$\alpha_{\rm eff}$$ depends on the multipole range. We computed it as the projection of the temperature and polarization spectra of (7) onto the template $$\ln\Delta^2=\ln A_s+(n_s-1)\ln(q/q_\ast)+\tfrac12\alpha\ln^2(q/q_\ast)$$ under Gaussian Fisher metrics, with the background fixed. For a Planck-like metric at $$30\le\ell\le2500$$ (noise $$33$$ and $$70\,\mu$$K-arcmin in temperature and polarization, $$7'$$ beam, $$f_{\rm sky}=0.6$$), the $$N_B=1$$ refit gives $$\alpha_{\rm eff}=+0.016$$; including $$2\le\ell\le29$$ lowers it to $$+0.008$$. For a weighting that mimics the combination of Planck and ACT used in [[ACT]](#ref-act) (Planck temperature to $$\ell=1000$$ and polarization to $$\ell=600$$, including the low multipoles, plus an ACT-like survey at $$600\le\ell\le3000$$ with $$15\,\mu$$K-arcmin, $$1.4'$$ and $$f_{\rm sky}=0.4$$), it gives $$\alpha_{\rm eff}=+0.011$$, and $$+0.006$$ to $$+0.013$$ across the other parameter sets of section 6. The corresponding measurement, $$dn_s/d\ln k=0.0062\pm0.0052$$ from Planck, ACT, CMB lensing and BAO [[ACT]](#ref-act), lies within about one standard deviation. Planck alone gives $$-0.0045\pm0.0067$$ [[Planck X]](#ref-planck-x); that value is driven by the low multipoles and so restates the comparison of section 6.

These are template projections at fixed background, not fits, and marginalising the background could shift them. For the $$N_B=1$$ refit the effective running is positive, between $$+0.008$$ and $$+0.016$$, in every weighting considered. For the parameters fitted to $$\ell\ge30$$ alone, which section 6 disfavours, a Planck-like weighting that includes $$\ell<30$$ gives $$-0.014$$ instead. With the background fixed, a survey with $$1\,\mu$$K-arcmin noise over 40% of the sky at $$30\le\ell\le3000$$ would measure $$\alpha$$ with $$\sigma\approx0.0015$$; marginalising the background would increase this, by an amount we have not computed.

### The tensor tilt

The tensor spectrum has the two-loop form (3) with its own parameters $$g_T$$ and $$\beta_T$$. From eq. (7.10) of [[CDS]](#ref-cds), at $$\xi=1/8$$ and with no Yukawa couplings, $$f_1^T=\frac{8}{3\pi^2N_A}(N_\psi+2N_\Phi-2)$$, so $$g_T/g=f_1^T/f_1$$ lies between $$-2/N_A$$ (all fermions) and $$-8/N_A$$ (all scalars). With $$N_A>889$$ and $$\lvert g\rvert\le0.013$$ this gives $$0<g_T<1.2\times10^{-4}$$. Repeating the steps leading to (3), with $$w_T=g_Tq_\ast/q$$, the tensor tilt at first order is $$n_T=w_T(L_T-1)$$ with $$L_T=\ln(q/\beta_Tg_Tq_\ast)$$. From $$f_2^T$$ (eq. 7.11 of [[CDS]](#ref-cds)), $$L_T=c_T+\ln(q_\ast/\mu)$$ at the pivot, where $$\mu$$ is the renormalisation scale and $$c_T$$, the ratio of the non-logarithmic parts of $$f_2^T$$ and $$f_1^T$$, lies between $$-2.3$$ (all fermions) and $$-2.1$$ (all scalars) for large $$N_A$$. The choice $$\mu$$ amounts to the finite part of the $$\int\sqrt g\,R$$ counterterm. For $$\mu=q_\ast$$, the scheme of [[ACDGS]](#ref-acdgs), the tilt is red, $$n_T\approx-3.2\,w_T$$, of order $$-10^{-4}$$; it is blue if $$\mu<e^{-3.2}q_\ast$$. The scalar $$f_2$$ contains the same logarithm of $$\mu$$, but when scalars are present also an infrared-regulated term absent from $$f_2^T$$, so the fitted $$\ln\beta$$ does not fix the counterterm. The sign of the tensor tilt is therefore not predicted, and its magnitude, below about $$4\times10^{-4}$$ in the scheme $$\mu=q_\ast$$, is far from measurable.

### Small scales and spectral distortions

As $$q\to\infty$$, $$u\to0$$ and $$\Delta^2\to\Delta_0^2$$: the spectrum becomes scale invariant, whereas the red tilt of $$\Lambda$$CDM continues. At the $$N_B=1$$ refit the ratio to the $$\Lambda$$CDM fit is $$1.08$$ at $$q=1\,{\rm Mpc}^{-1}$$ and $$1.38$$ at $$10^3\,{\rm Mpc}^{-1}$$. With the $$k$$-space window for $$\mu$$-type distortions of Chluba, Erickcek and Ben-Dayan [[CEB]](#ref-ceb), $$\mu\approx2.2\int_{1\,{\rm Mpc}^{-1}}^\infty\Delta^2(q)\,[e^{-\hat q/5400}-e^{-(\hat q/31.6)^2}]\,d\ln q$$ with $$\hat q=q\,{\rm Mpc}$$, the ratio of $$\mu$$ to its $$\Lambda$$CDM value is $$1.29$$–$$1.32$$ across the parameter sets above. This extrapolates the dual description over eleven e-folds in scale and assumes that the transition out of the non-geometric phase, which requires a change of the dual theory in the ultraviolet [[CDS]](#ref-cds), occurs at $$q\gtrsim10^4\,{\rm Mpc}^{-1}$$. The excess comes from the asymptotic scale invariance of (3) and is present without the resummation.

## 8. What the fitted theory does not determine

In the holographic dictionary, time evolution in the bulk corresponds to inverse renormalisation-group flow, and late times to the ultraviolet of the dual theory [[MS 2009]](#ref-ms09), [[CDS]](#ref-cds). The resummation of this note is an infrared effect and concerns the largest scales, which left the horizon earliest. The fitted theory approaches the free theory in the ultraviolet, where $$f\to f_0$$. Its parameters $$(N,N_\Phi,N_\psi,g_{\rm YM}^2)$$ do not fix a late-time vacuum energy: the only term in $$\langle T_{ij}T_{kl}\rangle$$ independent of $$q$$ is the contact term of Lemma 1, a local counterterm that does not enter $$B$$. The fermions of the dual carry no Standard Model charges and enter the observables only through $$q_c$$, $$N_A$$ and $$f_1$$. A late-time cosmological constant, evolving dark energy, neutrino masses or a dark matter abundance would therefore have to come from deformations of the dual theory in the ultraviolet or from degrees of freedom outside it, each with couplings not determined by the CMB fit. Nastase [[Nastase 2018]](#ref-nastase18), [[Nastase 2020]](#ref-nastase20) proposed constructions of this kind and identified the need for additional ultraviolet degrees of freedom.

The size of the gap can be estimated under two further assumptions: that the dual theory matches onto a geometric de Sitter phase at reheating, so that $$\Delta_T^2=2H^2/(\pi^2M_P^2)$$ with $$M_P$$ the reduced Planck mass, and that the de Sitter entropy $$8\pi^2M_P^2/H^2=16/\Delta_T^2$$ of that phase measures the degrees of freedom of the dual. Then $$r<0.036$$ gives an entropy $$16/(r\Delta_0^2)\gtrsim2\times10^{11}$$, while the de Sitter entropy of the present cosmological constant is about $$3\times10^{122}$$. This is a statement about orders of magnitude under the stated identifications, not a theorem.

## 9. Limitations and open questions

The low-$$\ell$$ result rests on (A1)–(A3). For conformally coupled matter it is the leading term of the $$1/N_A$$ expansion, and the corrections are suppressed by $$1/N_A<1.1\times10^{-3}$$ by power counting; the counting has not been carried out diagram by diagram to all orders. For $$N_B>1$$ the mixed diagrams between the scalar trace and the gauge trace are of the same order as the gauge loop and have not been computed, so the spectrum (7) and the bound (8) are then a model. A flavour-singlet quartic coupling would add a second crossover and at least one parameter. The prescription $$B(0)=0$$ rests on Lemma 1, and the absence of finite counterterms beyond $$\int\sqrt g$$ and $$\int\sqrt g\,R$$ is a choice; a finite $$\int\sqrt g\,R^2$$-type term would change the large-$$x$$ tail $$\mathcal F\approx0.27/x$$, which is subleading in (7).

The comparison with data uses a compressed temperature likelihood at fixed or gridded background. With it, the resummed model at $$N_B=1$$ is worse than $$\Lambda$$CDM by $$\Delta(-2\ln\mathcal L)$$ between about $$3$$ and $$5$$ while having one more primordial parameter. Whether this survives the full Planck likelihood with polarization and a free background, and how the model fares against ACT and SPT data at high $$\ell$$, has not been determined. The observed quadrupole is $$0.39$$ times the prediction of the $$N_B=1$$ refit, against $$0.22$$ times that of $$\Lambda$$CDM.

The value $$b=\tfrac{11}{12}a_1$$, if exact, should follow from a closed-form evaluation of (6), which we have not found. It is not known whether a non-perturbative treatment of the dual theory at $$\lambda\gtrsim1$$, relevant only at $$\ell<1$$ for the fitted parameters, changes the positivity bound.

## References

1. <a id="ref-ms09"></a>McFadden, Skenderis, “Holography for cosmology,” [arXiv:0907.5542](https://arxiv.org/abs/0907.5542).
2. <a id="ref-ms10"></a>McFadden, Skenderis, “The holographic universe,” [arXiv:1001.2007](https://arxiv.org/abs/1001.2007).
3. <a id="ref-acdgs"></a>Afshordi, Corianò, Delle Rose, Gould, Skenderis, “From Planck data to Planck era: observational tests of holographic cosmology,” [arXiv:1607.04878](https://arxiv.org/abs/1607.04878).
4. <a id="ref-ags"></a>Afshordi, Gould, Skenderis, “Constraining holographic cosmology using Planck data,” [arXiv:1703.05385](https://arxiv.org/abs/1703.05385).
5. <a id="ref-cds"></a>Corianò, Delle Rose, Skenderis, “Two-point function of the energy-momentum tensor and generalised conformal structure,” [arXiv:2008.05346](https://arxiv.org/abs/2008.05346).
6. <a id="ref-gtk"></a>Giombi, Tarnopolsky, Klebanov, “On $$C_J$$ and $$C_T$$ in conformal QED,” [arXiv:1602.01076](https://arxiv.org/abs/1602.01076).
7. <a id="ref-bk18"></a>BICEP/Keck Collaboration, “BICEP/Keck XIII: improved constraints on primordial gravitational waves using Planck, WMAP, and BICEP/Keck observations through the 2018 observing season,” [arXiv:2110.00483](https://arxiv.org/abs/2110.00483).
8. <a id="ref-planck-v"></a>Planck Collaboration, “Planck 2018 results. V. CMB power spectra and likelihoods,” [arXiv:1907.12875](https://arxiv.org/abs/1907.12875).
9. <a id="ref-planck-vi"></a>Planck Collaboration, “Planck 2018 results. VI. Cosmological parameters,” [arXiv:1807.06209](https://arxiv.org/abs/1807.06209).
10. <a id="ref-planck-x"></a>Planck Collaboration, “Planck 2018 results. X. Constraints on inflation,” [arXiv:1807.06211](https://arxiv.org/abs/1807.06211).
11. <a id="ref-act"></a>Calabrese et al. (ACT Collaboration), “The Atacama Cosmology Telescope: DR6 constraints on extended cosmological models,” [arXiv:2503.14454](https://arxiv.org/abs/2503.14454).
12. <a id="ref-litebird"></a>LiteBIRD Collaboration, “Probing cosmic inflation with the LiteBIRD cosmic microwave background polarization survey,” [arXiv:2202.02773](https://arxiv.org/abs/2202.02773).
13. <a id="ref-ceb"></a>Chluba, Erickcek, Ben-Dayan, “Probing the inflaton: small-scale power spectrum constraints from measurements of the CMB energy spectrum,” [arXiv:1203.2681](https://arxiv.org/abs/1203.2681).
14. <a id="ref-nastase18"></a>Nastase, “Solution of the cosmological constant problem within holographic cosmology,” [arXiv:1812.07597](https://arxiv.org/abs/1812.07597).
15. <a id="ref-nastase20"></a>Nastase, “Reheating in holographic cosmology and connecting to Λ-MSSM constructions for particle physics,” [arXiv:2008.06144](https://arxiv.org/abs/2008.06144).
