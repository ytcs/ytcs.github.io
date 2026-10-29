---
title: The low multipoles of holographic cosmology at large flavour number
date: 2026-10-29
category: dark-sector
order: 7
description: In holographic cosmology with conformally coupled matter, the leading large-flavour resummation of the dual stress-tensor correlator reproduces the published two-loop coefficient and fixes the power spectrum at the largest angular scales with no new parameter. Positivity of the spectrum then bounds the number of non-conformally coupled fields at about two to three, which excludes the explicit solution published with the model.
status: The two-loop coefficient and the contact-term lemma are proved by direct computation; the crossover function, the positivity bound and the Planck comparison are numerical, the latter with a compressed temperature likelihood at fixed background.
excludes: Holographic-cosmology duals of the stated field content whose weighted count of non-conformal scalars exceeds about 2.6 (with the fitted value of the second spectral parameter), among them the published explicit solution with about 88. The extrapolated two-loop form at multipoles below about 30 is disfavoured by the same comparison.
---

In holographic cosmology the primordial power spectrum is computed from the stress-tensor two-point function of a three-dimensional gauge theory with many scalar and fermion fields. The published two-loop form of the spectrum fits the Planck temperature data at multipoles $$\ell\ge30$$, but perturbation theory fails at lower $$\ell$$, which were therefore excluded from the fits. The question here is what the same theory predicts for $$\ell<30$$. When all scalars are conformally coupled and the quartic and Yukawa couplings are negligible, the answer is fixed at leading order in the number of matter fields: the gauge-field contribution to the trace correlator, with both gluon lines dressed by the chain of matter loops, can be computed in closed form up to one numerical function. Its expansion reproduces the published two-loop coefficient $$64/(3\pi^2)$$ exactly (section 3), and below a crossover scale, corresponding to $$\ell_c\approx3$$ for the parameters fitted to all multipoles, the logarithm that drives the two-loop running stops growing. The low-$$\ell$$ spectrum then follows from the three parameters already fitted at high $$\ell$$, with no new parameter.

Positivity of the power spectrum at all scales bounds the number $$N_B=1+\sum_M(1-8\xi_M)^2$$, which counts the gauge field plus the non-conformally coupled scalars weighted by $$(1-8\xi_M)^2$$: $$N_B\le0.864\,\beta$$, where $$\beta$$ is one of the fitted parameters (section 5). The bound relies on two properties of the resummed correlator that we establish numerically, and for $$N_B>1$$ it holds within a model that extends the calculation to that case. Over the published one-standard-deviation ranges of $$\ln\beta$$ it gives $$N_B\lesssim1.6$$ to $$2.9$$, which excludes the explicit solution with $$N_B\approx88$$ given with the model. Fitted to the Planck temperature spectrum at all multipoles with a compressed likelihood and the background held at its Planck value, the resummed spectrum is worse than $$\Lambda$$CDM by $$\Delta(-2\ln\mathcal L)\approx5$$ and predicts a quadrupole $$0.57$$ times the $$\Lambda$$CDM value (section 6). These last numbers are numerical results of an approximate likelihood; the other results of the page do not depend on them.

The framework is due to McFadden and Skenderis [[MS09]](#ref-ms09), [[MS10]](#ref-ms10). Afshordi, Corianò, Delle Rose, Gould and Skenderis [[ACDGS]](#ref-acdgs) used the two-loop coefficients, computed in full by Corianò, Delle Rose and Skenderis [[CDS]](#ref-cds), to fit the model to Planck 2015 data; they found the fit at all multipoles disfavoured relative to $$\Lambda$$CDM with running, and attributed this to the breakdown of perturbation theory at $$\ell\lesssim30$$. Afshordi, Gould and Skenderis [[AGS]](#ref-ags) repeated the analysis in more detail. The dressing of a three-dimensional gauge field by a large number of matter fields is standard; Giombi, Tarnopolsky and Klebanov [[GTK]](#ref-gtk) compute the stress-tensor two-point coefficient of conformal QED at the resulting infrared fixed point to next-to-leading order in $$1/N$$. What this page adds is the crossover of the trace correlator between the perturbative and infrared regimes, an exact check of the two-loop coefficient against it, the consequences for the low-$$\ell$$ spectrum and for positivity, and a refit of the spectrum to temperature data.

The dictionary is stated in section 1. A reader needs one-loop computations in three-dimensional gauge theory and the basics of CMB power spectra.

## 1. The two-loop spectrum and where it fails

The dual theory is SU($$N$$) Yang–Mills theory in three Euclidean dimensions coupled to $$N_\Phi$$ real scalars and $$N_\psi$$ fermions, all in the adjoint representation [[ACDGS]](#ref-acdgs). Scalars couple to the background curvature with parameters $$\xi_M$$; $$\xi_M=1/8$$ is conformal coupling in three dimensions. The theory is super-renormalizable, and its dimensionless coupling at momentum $$q$$ is the 't Hooft coupling $$\lambda=g_{\rm YM}^2N/q$$, small at large $$q$$.

The trace $$\Theta=T^i{}_i$$ of the stress tensor carries the scalar perturbations. Up to local terms its two-point function has the form $$\langle\Theta(q)\Theta(-q)\rangle=N^2q^3f(\lambda)$$, and the holographic dictionary [[MS09]](#ref-ms09), in the form used in [[ACDGS]](#ref-acdgs), gives the curvature power spectrum

$$
\Delta^2(q)=\frac{1}{4\pi^2N^2f(\lambda)}. \tag{1}
$$

Here $$q$$ is the comoving wavenumber of the cosmological perturbation. In perturbation theory [[ACDGS]](#ref-acdgs), [[CDS]](#ref-cds),

$$
f(\lambda)=f_0\big[1-f_1\lambda\ln\lambda+f_2\lambda+O(\lambda^2)\big],\qquad f_0=\frac{N_B}{64},\qquad N_B=1+\sum_M(1-8\xi_M)^2. \tag{2}
$$

The "1" in $$N_B$$ is the gauge field; each scalar contributes $$(1-8\xi_M)^2$$, which vanishes for conformal coupling, and fermions do not contribute to $$f_0$$. Without Yukawa couplings the published coefficient is ([[ACDGS]](#ref-acdgs), eq. (8), with $$\Sigma_\Phi=2\sum_M\xi_M^2$$)

$$
N_Bf_1=-\frac{4}{3\pi^2}\Big(N_\psi-2+2N_\Phi-96\sum_M\xi_M^2\Big),\qquad\text{so that}\quad f_1=-\frac{4}{3\pi^2}\Big(N_\psi+\frac{N_\Phi}{2}-2\Big)\ \text{ at all }\xi_M=\tfrac18. \tag{3}
$$

Writing $$g\,q_*=f_1g_{\rm YM}^2N$$ and $$\ln(1/\beta)=f_2/f_1+\ln\lvert f_1\rvert$$ with the pivot $$q_*=0.05\,{\rm Mpc}^{-1}$$ and expanding $$1/f$$ to first order, (1) becomes the form fitted in [[ACDGS]](#ref-acdgs),

$$
\Delta^2(q)=\frac{\Delta_0^2}{1+u\ln u+u\ln\beta},\qquad u\equiv\frac{\lvert g\rvert q_*}{q},\qquad\Delta_0^2=\frac{1}{4\pi^2N^2f_0}, \tag{4}
$$

where the fits have $$g<0$$, that is $$f_1<0$$. The three parameters $$(\Delta_0^2,g,\ln\beta)$$ replace the amplitude and tilt of $$\Lambda$$CDM. The pivot is not stated in [[ACDGS]](#ref-acdgs); the value $$0.05\,{\rm Mpc}^{-1}$$ reproduces its quoted range of the effective coupling, $$3.3\times10^{-4}\le g_{\rm eff}^2\le0.41$$ between $$\ell=2500$$ and $$\ell=2$$, for the solution of section 5. The fits to Planck 2015 data in [[ACDGS]](#ref-acdgs), Table 1, give $$g=-0.01305^{+0.00452}_{-0.00345}$$ and $$\ln\beta=1.014^{+0.206}_{-0.272}$$ for $$\ell\ge30$$, and $$g=-0.00703^{+0.00105}_{-0.00167}$$ and $$\ln\beta=0.877^{+0.186}_{-0.239}$$ for all multipoles.

Multipoles and wavenumbers are related by $$\ell\approx q\chi_*$$, where $$\chi_*=13\,873\,{\rm Mpc}$$ is the comoving distance to last scattering for the Planck 2018 parameters of Table 2 of [[Planck VI]](#ref-planck-vi) (obtained here from a Boltzmann code at $$z_*=1089.9$$). At the quadrupole, the $$\ell\ge30$$ fit has $$u=4.5$$, so (4) lies outside its range of validity. Extrapolated anyway, it predicts a quadrupole $$0.32$$ times the $$\Lambda$$CDM value (section 6 describes the comparison).

## 2. The expansion parameter is the number of fields

The two-loop term in (2) is large at low $$\ell$$ although the 't Hooft coupling there is small. The tensor-to-scalar ratio of the model is $$r=32N_B/N_A$$ with $$N_A=1+2N_\psi+N_\Phi$$ [[ACDGS]](#ref-acdgs), so the bound $$r<0.036$$ [[BK18]](#ref-bk18) requires $$N_A>889$$ when $$N_B=1$$. With (3), $$\lvert f_1\rvert=\tfrac{2}{3\pi^2}(N_A-5)\gtrsim60$$ at $$\xi_M=\tfrac18$$, and $$\lambda=u/\lvert f_1\rvert\le0.08$$ at $$\ell=2$$ for the parameter sets of this page. The two-loop correction is not small because $$f_1$$ counts all the matter fields, while $$f_0$$ counts only the gauge field.

Each matter field contributes to the one-loop vacuum polarisation of the gluon, and this fixes the relevant limit. For a field of unit charge in three dimensions, a complex scalar and a two-component Dirac fermion each give the transverse self-energy $$e^2k/16$$. For the scalar, the polarisation tensor has the form $$\Pi_{\mu\nu}=(k^2\delta_{\mu\nu}-k_\mu k_\nu)\Pi(k)$$, so its trace is $$2k^2\Pi$$. In dimensional regularisation the trace is $$\Pi_{\mu\mu}=-e^2\int\frac{d^3l}{(2\pi)^3}\frac{(2l+k)^2}{l^2(l+k)^2}$$, and $$(2l+k)^2=2l^2+2(l+k)^2-k^2$$ leaves $$e^2k^2\int d^3l/[(2\pi)^3l^2(l+k)^2]=e^2k/8$$ from the bubble $$\int d^3l/[(2\pi)^3l^2(l+k)^2]=1/(8k)$$. Hence $$k^2\Pi=e^2k/16$$ in magnitude. The fermion trace, with $$\gamma_\mu\gamma_\nu\gamma_\mu=-\gamma_\nu$$ in three dimensions and $${\rm tr}\,1=2$$, gives the same. For adjoint fields the charge factor is $$N$$, and a real adjoint scalar counts half of a complex one. This reproduces the relative weights $$1:\tfrac12$$ of fermions and scalars in (3). The chain of these bubbles gives the dressed gluon propagator, in a normalisation where the free propagator is $$1/k^2$$,

$$
D(k)=\frac{1}{k^2+q_ck},\qquad q_c=g_{\rm YM}^2N\Big(\frac{N_\Phi}{32}+\frac{N_\psi}{16}\Big)=g_{\rm YM}^2N\,\frac{N_A-1}{32}. \tag{5}
$$

Each bubble is suppressed by $$g_{\rm YM}^2$$ and enhanced by the number of fields, so at fixed $$q_c$$ the bubble chain is leading order in $$1/N_A$$. We therefore expand in $$1/N_A$$ at fixed $$x\equiv q_c/q$$. With $$g\,q_*=f_1g_{\rm YM}^2N$$ and (3), $$x=\kappa u$$ with

$$
\kappa=\frac{q_c}{\lvert g\rvert q_*}=\frac{(N_A-1)/32}{\lvert f_1\rvert}=\frac{3\pi^2}{64}\,\frac{N_A-1}{N_A-5}, \tag{6}
$$

which equals $$3\pi^2/64=0.4626$$ to within $$0.5\%$$ for $$N_A>889$$. Three variables are in play: the 't Hooft coupling $$\lambda$$, which is small; the two-loop expansion parameter $$u=\lvert f_1\rvert\lambda$$, which is large at low $$\ell$$; and $$x=q_c/q$$, the variable of the resummation. The crossover $$x=1$$ lies at $$\ell_c=\kappa\lvert g\rvert q_*\chi_*$$.

We assume from here on: (A1) all scalars are conformally coupled, $$\xi_M=1/8$$, so that $$N_B=1$$; (A2) the Yukawa couplings vanish; (A3) the quartic couplings are small enough that their own bubble chains can be neglected. At $$\xi_M=1/8$$ the quartic couplings drop out of the constant term of the two-loop result at the order considered (the constant $$a_0$$ of [[ACDGS]](#ref-acdgs) contains them only through $$(8\xi_{M_1}-1)(8\xi_{M_2}-1)$$), but a flavour-singlet quartic coupling would generate a chain of scalar bubbles at a scale proportional to its coupling times $$N_\Phi$$, and could alter the low-$$\ell$$ spectrum; (A3) excludes this.

## 3. The resummed trace correlator

Under (A1)–(A3) the scalars and fermions are classically conformal, and the parts of $$\Theta$$ built from them vanish on the equations of motion; their insertions in correlators produce only contact terms. The trace is then the gauge-field term $$\Theta_A=\tfrac14F^a_{\mu\nu}F^a_{\mu\nu}$$ in the normalisation of (5), whose free two-point function is the "1" in $$N_B$$. At leading order in $$1/N_A$$ the trace correlator is the one-loop diagram of $$\Theta_A$$ with both gluon lines replaced by $$D$$. That no other diagram contributes at this order is a power-counting statement, not a proof: with the propagator (5), each additional gluon exchange or matter loop attached to the two gluon lines costs a factor $$g_{\rm YM}^2N/q_c\sim1/N_A$$ at every $$q$$, including $$q\ll q_c$$, where $$D\approx1/(q_ck)$$. The pure-gauge two-loop term, the "$$-2$$" in (3), is of this kind. The counting fails when some scalar has $$\xi_M\ne1/8$$: then $$\Theta$$ has a matter part, the mixed correlators $$\langle\Theta_\Phi\Theta_A\rangle$$ are of the same order as the gauge loop, and they are not included here. This is why the controlled case is $$N_B=1$$.

### The first two terms

The expansion in $$q_c$$ can be done in closed form, and it checks the normalisation before the full function is computed.

<div class="proposition" markdown="1">
**Proposition 1** (two-loop logarithm). Per colour component, the gauge-field trace correlator with the propagator (5) has, to first order in $$q_c$$, the non-analytic part

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

since $$\nabla^2\ln r=1/r^2$$ and the transform of $$1/k$$ is $$1/(2\pi^2r^2)$$; the constant drops out of $$F_{\mu\nu}$$. Then $$\langle F_{\mu\nu}(r)F_{\rho\sigma}(0)\rangle=-(\partial_\mu\partial_\rho G\,\delta_{\nu\sigma}-\partial_\mu\partial_\sigma G\,\delta_{\nu\rho}-\partial_\nu\partial_\rho G\,\delta_{\mu\sigma}+\partial_\nu\partial_\sigma G\,\delta_{\mu\rho})$$, and by Wick's theorem $$\langle\Theta_A(r)\Theta_A(0)\rangle=\tfrac18\sum\langle F_{\mu\nu}F_{\rho\sigma}\rangle^2$$. Carrying out the sums, which we did symbolically, gives

$$
\langle\Theta_A(r)\Theta_A(0)\rangle=\frac{3}{16\pi^2r^6}-\frac{q_c}{2\pi^3r^5}+O(q_c^2).
$$

In three dimensions the Fourier transform of $$r^{-2a}$$ is $$\pi^{3/2}2^{3-2a}\Gamma(\tfrac32-a)\Gamma(a)^{-1}q^{2a-3}$$. At $$a=3$$ this is $$\pi^2q^3/12$$, so the first term gives $$q^3/64$$. At $$a=\tfrac52+\epsilon$$ we have $$\Gamma(\tfrac32-a)=\Gamma(-1-\epsilon)=1/\epsilon+O(1)$$ and $$q^{2a-3}=q^2(1+2\epsilon\ln q+\dots)$$, so the transform is $$\pi^{3/2}2^{-2}\Gamma(\tfrac52)^{-1}q^2(1/\epsilon+2\ln q)+O(\epsilon)=\tfrac{2\pi}{3}q^2\ln q$$ plus a local $$q^2$$ term. The second term therefore gives $$-q_cq^2\ln q/(3\pi^2)$$ plus a local term. At fixed $$q_c$$, $$x\ln q=x\ln q_c-x\ln x$$, and the first piece is analytic in $$q$$. Dividing by $$q^3/64$$ gives $$a_1=64/(3\pi^2)$$.
</div>

The first term reproduces the normalisation $$f_0=1/64$$ per unit of $$N_B$$. For the logarithm, translate back to (2): with $$x=\kappa\lambda\lvert f_1\rvert$$, the term $$a_1x\ln x$$ contributes $$-f_1\lambda\ln\lambda$$ with $$-f_1=a_1\,q_c/(g_{\rm YM}^2N)=\tfrac{4}{3\pi^2}(N_\psi+N_\Phi/2)$$. This is the coefficient (3) at $$\xi_M=\tfrac18$$ apart from the pure-gauge $$-2$$, of relative order $$1/N_A$$. Since $$a_1$$ comes from the gauge loop alone and the bubble coefficients $$\tfrac1{16}$$ and $$\tfrac1{32}$$ from the one-loop vacuum polarisation, the agreement is a check of both. It shows that at $$\xi_M=1/8$$ the entire matter logarithm of the two-loop result comes from the self-energy insertion.

The agreement is specific to the conformal case. For a scalar with general $$\xi$$ the published coefficient in (3) is $$2-96\xi^2$$ per scalar, against the dressing value $$\tfrac12$$; the difference $$\tfrac32(1-64\xi^2)=12(1-8\xi)(\xi+\tfrac18)$$ vanishes only at $$\xi=\tfrac18$$ (for example it is $$\tfrac32$$ at $$\xi=0$$, where the dressing accounts for a quarter of the published coefficient, and $$-0.19$$ at $$\xi=0.133$$). The remainder comes from diagrams that mix $$\Theta_\Phi$$ and $$\Theta_A$$, which are of the same order as the gauge loop when $$N_B>1$$. The check concerns only the first order in $$x$$; the low-$$\ell$$ spectrum depends on $$x\gtrsim1$$, where the full function is needed.

### The contact term

The full calculation produces a term independent of $$q$$, which in $$\mathcal F$$ would appear as a multiple of $$x^3$$ and would dominate at small $$q$$. It does not enter the power spectrum, which is computed from the spin-0 form factor $$\mathcal B(q)$$ in the decomposition $$\langle T_{ij}T_{kl}\rangle=\mathcal A(q)\Pi_{ijkl}+\mathcal B(q)\pi_{ij}\pi_{kl}$$ of the transverse part of the correlator ([[CDS]](#ref-cds), eq. (5.3)). Here $$\pi_{ij}=\delta_{ij}-q_iq_j/q^2$$ and $$\Pi_{ijkl}=\tfrac12(\pi_{ik}\pi_{jl}+\pi_{il}\pi_{jk}-\pi_{ij}\pi_{kl})$$; away from contact terms $$\langle\Theta\Theta\rangle=4\mathcal B$$.

<div class="lemma" markdown="1">
**Lemma 1** (no constant in $$\mathcal B$$). A term in $$\langle T_{ij}T_{kl}\rangle$$ independent of $$q$$ does not contribute to $$\mathcal B$$.
</div>

<div class="proof" markdown="1">
*Proof.* A term independent of $$q$$ is local and has no derivatives. By diffeomorphism invariance the only such term in the generating functional is $$c\int d^3x\sqrt{\gamma}$$, with $$\gamma_{ij}$$ the background metric of the dual theory. Its second variation at flat space is $$c[\tfrac14\delta_{ij}\delta_{kl}-\tfrac14(\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk})]$$. Projecting each index with $$\pi$$ gives $$c[\tfrac14\pi_{ij}\pi_{kl}-\tfrac14(\pi_{ik}\pi_{jl}+\pi_{il}\pi_{jk})]=-\tfrac{c}{2}\Pi_{ijkl}+0\cdot\pi_{ij}\pi_{kl}$$, so its transverse part lies entirely in $$\mathcal A$$. We confirmed the decomposition by computer algebra at a generic momentum.
</div>

The constant is nonetheless present in the full trace $$\delta_{ij}\delta_{kl}\langle T_{ij}T_{kl}\rangle$$, with weight $$\tfrac34c$$; it belongs to the Ward identity involving $$\langle T_{ij}\rangle$$. We remove it, which amounts to $$\mathcal B(0)=0$$.

### The full crossover

The dressed propagator has a positive spectral representation over a mass $$m$$,

$$
D(k)=\int_0^\infty dm\,\frac{\rho(m)}{k^2+m^2},\qquad\rho(m)=\frac{2}{\pi}\,\frac{q_c}{m^2+q_c^2},
$$

as follows from $$\int_0^\infty dm\,[(m^2+a^2)(m^2+k^2)]^{-1}=\pi/[2ak(a+k)]$$ with $$a=q_c$$. This reduces the dressed loop to a superposition of loops with two massive vector propagators. The contraction of the two field-strength correlators gives, for gluon momenta $$k$$ and $$p=q-k$$, the numerator $$4[(k\cdot p)^2+k^2p^2]$$ (verified symbolically in three dimensions). Writing $$k\cdot p=\tfrac12(q^2-k^2-p^2)$$ and setting $$k^2=-m_1^2$$, $$p^2=-m_2^2$$ wherever a propagator is not cancelled, the remaining terms have one propagator cancelled and are polynomials in $$q$$ of degree at most 2. With the three-dimensional bubble $$\int\frac{d^3k}{(2\pi)^3}\frac{1}{(k^2+m_1^2)(p^2+m_2^2)}=\frac{\arctan(q/(m_1+m_2))}{4\pi q}$$ and the prefactor $$\tfrac18$$ from Wick's theorem, $$\langle\Theta_A\Theta_A\rangle=\frac{1}{32\pi}\iint\rho\rho\,J$$ plus polynomial terms. Dividing by $$q^3/64$$,

$$
\mathcal F(q)=\frac{2}{\pi q^3}\int_0^\infty\!\!\int_0^\infty dm_1dm_2\,\rho(m_1)\rho(m_2)\Big[J(q)-J_0-q^2J_2\Big],\quad J=\big[(q^2+m_1^2+m_2^2)^2+4m_1^2m_2^2\big]\frac{\arctan\frac{q}{m_1+m_2}}{q}, \tag{7}
$$

where $$J_0$$ and $$J_2$$ are the coefficients of $$q^0$$ and $$q^2$$ in the expansion of $$J$$ about $$q=0$$. As $$q_c\to0$$, $$\rho$$ tends to a delta function at $$m=0$$ and $$\mathcal F\to1$$. The first subtraction implements $$\mathcal B(0)=0$$; the second removes a local term $$\propto q_cq^2$$, whose coefficient is a logarithmically divergent renormalisation constant. The subtracted integrand is $$O(q^4)$$ at small $$q$$ and $$O(1/m)$$ at large masses, and the integral converges absolutely.

*Numerical evaluation.* We integrated (7) by the trapezoidal rule in $$\ln m_1,\ln m_2$$ over $$e^{-40}<m<e^{30}$$ (units $$q_c=1$$, so $$q=1/x$$), with the series for the subtracted integrand at small $$q/(m_1+m_2)$$ and the unsubtracted form at large ratio to avoid cancellations. Changing the step from $$0.06$$ to $$0.03$$ or the range changes $$\mathcal F$$ by less than $$10^{-12}$$.

| $$x$$ | $$10^{-3}$$ | $$10^{-2}$$ | $$0.1$$ | $$1$$ | $$10$$ | $$10^2$$ | $$10^4$$ |
|---|---|---|---|---|---|---|---|
| $$\mathcal F(x)$$ | 0.98704 | 0.91958 | 0.64439 | 0.19920 | 0.026346 | 0.0027272 | $$2.7378\times10^{-5}$$ |

At small $$x$$ a fit of $$(\mathcal F-1)/x$$ to $$a_1\ln x+b+O(x\ln^2x)$$ over $$10^{-6}\le x\le10^{-3}$$ gives $$a_1=2.161518$$ against $$64/(3\pi^2)=2.161519$$, as Proposition 1 requires, and $$b=1.981392$$, which agrees with $$\tfrac{11}{12}a_1=1.981392$$ to seven digits; we have not derived that value. On a grid of 161 points in $$10^{-4}\le x\le10^4$$ the function decreases monotonically and stays positive, and $$x\mathcal F(x)$$ tends to a constant, $$0.2738$$ at $$x=10^4$$ ($$0.2727$$ at $$x=10^2$$). In this subtraction scheme the large-$$x$$ expansion therefore has neither a constant nor a term linear in $$x$$. The disappearance of the constant is expected: for $$q\ll q_c$$ the gauge field becomes part of the conformal infrared theory of the matter fields, whose trace correlator has no $$q^3$$ term.

The subtraction in (7) is arbitrary up to a term linear in $$x$$, since that term is the local counterterm $$\propto q_cq^2$$; a change of scheme adds the same multiple of $$x$$ at all $$x$$. This term is absorbed in the fitted $$\ln\beta$$ in section 4. What remains predictive is the shape of $$\mathcal F$$. Let $$b_\infty$$ be the coefficient of the term linear in $$x$$ at large $$x$$ ($$b_\infty=0$$ in our scheme) and define

$$
x_*=e^{-(b-b_\infty)/a_1}=e^{-11/12}=0.39985. \tag{8}
$$

A change of scheme shifts $$b$$ and $$b_\infty$$ equally, so $$x_*$$ is scheme independent. Near $$x=x_*$$ the logarithm in $$\mathcal F=1+a_1x\ln(x/x_*)+\dots$$ stops growing.

## 4. The resummed spectrum

The physical $$f$$ is $$f_0\mathcal F$$ plus the local term, that is, plus a multiple of $$u$$. Matching to the fitted form (4) at small $$u$$, where $$\mathcal F(\kappa u)=1+u\ln u+u\ln\kappa+(b/a_1)u+\dots$$ for $$N_B=1$$ (using $$a_1\kappa=1$$), fixes the local term to $$u\ln(\beta x_*/\kappa)$$. Hence, for $$N_B=1$$,

$$
\frac{f}{f_0}=\mathcal F(\kappa u)+u\ln\frac{\beta x_*}{\kappa},\qquad\kappa=\frac1{a_1}=\frac{3\pi^2}{64}. \tag{9}
$$

For $$u\ll1$$, (9) reduces to $$1+u\ln u+u\ln\beta$$ up to $$O(u^2\ln^2u)$$. For $$u\gg1$$, $$f/f_0\to u\ln(\beta x_*/\kappa)$$: the logarithm $$\ln(\beta u)$$ of (4) is frozen at its value at $$x=x_*$$, the scale $$q=q_c/x_*\approx2.5\,q_c$$, and the tree term has switched off. The low-$$\ell$$ spectrum is therefore determined by $$(\Delta_0^2,g,\ln\beta)$$ alone, the parameters that high-$$\ell$$ data already measure. The colour number $$N$$ and the split between $$N_\Phi$$ and $$N_\psi$$ drop out at leading order.

For $$N_B>1$$ we use a model in which the extra non-conformal scalars contribute only to the tree term and the gauge bubble carries the whole two-loop logarithm. This requires replacing $$\kappa$$ by $$\kappa_B\equiv N_B/a_1$$, which equals $$\kappa$$ at $$N_B=1$$, and the same matching gives

$$
\frac{f}{f_0}=1+\frac{\mathcal F(\kappa_Bu)-1}{N_B}+u\ln\frac{\beta x_*}{\kappa_B}. \tag{10}
$$

For $$N_B>1$$ this omits the mixed diagrams discussed in section 3, so (10) is a model, not a consequence of the theory.

## 5. Positivity and the weighted number of non-conformal scalars

A power spectrum is positive, so $$f$$ must be positive at every $$q$$, including scales far outside the observed range.

<div class="proposition" markdown="1">
**Proposition 2** (positivity bound). Assume, as found numerically in section 3, that $$\mathcal F>0$$ for all $$x$$ and $$x\mathcal F(x)$$ stays bounded as $$x\to\infty$$. Then the spectrum (10) is positive at all $$q$$ if and only if $$\kappa_B\le\beta x_*$$, that is,

$$
N_B\le a_1x_*\,\beta=0.864\,\beta. \tag{11}
$$

</div>

<div class="proof" markdown="1">
*Proof.* Each term of $$f/f_0=(1-1/N_B)+\mathcal F(\kappa_Bu)/N_B+u\ln(\beta x_*/\kappa_B)$$ is non-negative when $$\ln(\beta x_*/\kappa_B)\ge0$$, and the second is strictly positive, so $$f>0$$. If $$\ln(\beta x_*/\kappa_B)<0$$, the last term tends to $$-\infty$$ as $$u\to\infty$$ while the others stay bounded, so $$f<0$$ at small enough $$q$$.
</div>

For $$N_B=1$$, (11) requires $$\ln\beta\ge\ln(1/0.864)=0.146$$, which every published fit satisfies. For larger $$N_B$$, (11) is a statement within the model (10). The bound for several values of $$\ln\beta$$:

| $$\ln\beta$$ | 0.638 | 0.877 | 1.014 | 1.112 | 1.22 | 2.0 | 3.5 |
|---|---|---|---|---|---|---|---|
| $$N_{B,\max}$$ | 1.64 | 2.08 | 2.38 | 2.63 | 2.93 | 6.4 | 28.6 |

The first and fifth entries are the lower and upper ends of the one-standard-deviation ranges of the two fits of [[ACDGS]](#ref-acdgs) combined ($$0.877-0.239$$ and $$1.014+0.206$$), so over those ranges $$N_B\le1.6$$ to $$2.9$$; the second and third are the central values of those fits, the fourth is the all-$$\ell$$ refit of section 6, and $$3.5$$ is the upper end of the prior range used in [[AGS]](#ref-ags). The violation, when it occurs, can lie outside the observed range: at the parameters of the refit, $$N_B=3$$ makes $$f$$ negative only below $$\ell\approx1.2$$, $$N_B=4$$ below $$\ell\approx3.5$$ and $$N_B=6$$ below $$\ell\approx6.2$$.

The explicit solution of [[ACDGS]](#ref-acdgs) takes $$N_\Phi$$ scalars with a common $$\xi$$ chosen so that $$r=0.12$$, no fermions and $$\lambda_{\rm quartic}=1$$, and finds from the observed $$\Delta_0^2$$ and $$\ln\beta$$ the values $$N=2995$$ and $$N_\Phi=23\,255$$. For large $$N_\Phi$$ the tensor-to-scalar ratio is $$r=32(1-8\xi)^2$$, so $$r=0.12$$ means $$\lvert1-8\xi\rvert=0.0612$$ (the paper writes $$\xi=0.133$$ for $$\xi=0.1327$$). Then $$N_B=1+23\,255\times0.00375=88.2$$ and, from (3), $$f_1=-11.06$$, as quoted there ("$$\approx-11$$"). Within the model (10), positivity requires $$\ln\beta\ge\ln(88.2/0.864)=4.63$$, far outside the fitted range and outside the prior range $$-0.2\le\ln\beta\le3.5$$ used in [[AGS]](#ref-ags).

The actual coupling of this solution is stronger than the model assumes. The ratio (6) is $$\kappa=(N_\Phi/32)/\lvert f_1\rvert=65.7$$, against $$\kappa_B=88.2/a_1=40.8$$, so the gauge bubble alone gives $$65.7/40.8=1.61$$ times the two-loop logarithm, and if the published coefficient is the sum of the gauge and mixed contributions the mixed diagrams contribute $$-0.61$$ of it. If the whole logarithm froze at $$x=x_*$$, positivity would require $$\ln\beta\ge\ln(65.7/x_*)=5.1$$; if the mixed share keeps running, the spectrum is negative at small enough $$q$$ for any $$\beta$$. The crossover of this solution lies at $$\ell_c=\kappa\lvert g\rvert q_*\chi_*\approx600$$ for $$\lvert g\rvert=0.01305$$, so the two-loop form on which its parameters were fitted is not valid anywhere in $$\ell\lesssim600$$. Its tensor-to-scalar ratio, $$0.12$$, is also excluded by $$r<0.036$$ [[BK18]](#ref-bk18).

## 6. Comparison with Planck temperature data

This section is numerical and approximate; only differences between models are meaningful.

*Inputs.* The data are the Planck 2018 temperature power spectrum from the Planck Legacy Archive (file `COM_PowerSpect_CMB-TT-full_R3.01`, described in [[Planck V]](#ref-planck-v)): for $$2\le\ell\le29$$ the Commander estimates $$\hat D_\ell$$, for $$30\le\ell\le800$$ the unbinned estimates with their per-multipole errors. The background is fixed at the base $$\Lambda$$CDM values of [[Planck VI]](#ref-planck-vi), Table 2 (TT,TE,EE+lowE+lensing): $$\Omega_bh^2=0.02237$$, $$\Omega_ch^2=0.1200$$, $$H_0=67.36$$, $$\tau=0.0544$$, with massive neutrinos of $$0.06\,{\rm eV}$$. The $$C_\ell$$ are computed from the line-of-sight transfer functions of a Boltzmann code, which are linear in the primordial spectrum, and lensing is included as the $$\Lambda$$CDM ratio of lensed to unlensed spectra. The kernel reproduces the code's own $$C_\ell$$ to $$4\times10^{-6}$$.

*Likelihood.* For $$\ell\le29$$ we use the full-sky form for a single multipole, scaled by a sky fraction $$f_{\rm sky}=0.86$$, $$-2\ln\mathcal L=\sum_\ell(2\ell+1)f_{\rm sky}\,(\hat C_\ell/C_\ell-\ln(\hat C_\ell/C_\ell)-1)$$, which approximates the Planck low-$$\ell$$ likelihood; for $$30\le\ell\le800$$ a diagonal $$\chi^2$$ with the published errors. $$\Lambda$$CDM has $$(A_s,n_s)$$ free; the holographic models have $$(\Delta_0^2,g,\ln\beta)$$ free, with positivity at all $$q$$ imposed.

| model | $$g$$ | $$\ln\beta$$ | $$\Delta(-2\ln\mathcal L)$$ total [$$\ell<30$$, $$\ell\ge30$$] | $$C_2/C_2^{\Lambda\rm CDM}$$ |
|---|---|---|---|---|
| resummed (9), $$N_B=1$$ | $$-0.00914$$ | 1.112 | $$+5.0$$ [$$+2.1$$, $$+2.9$$] | 0.565 |
| resummed (10), $$N_B=2$$ | $$-0.01238$$ | 1.326 | $$+3.2$$ [$$+1.1$$, $$+2.1$$] | 0.591 |
| two-loop form (4), all $$\ell$$ | $$-0.00648$$ | 0.832 | $$+7.6$$ [$$+3.2$$, $$+4.4$$] | 0.534 |
| resummed, $$N_B=1$$, fitted to $$\ell\ge30$$ only | $$-0.01744$$ | 1.284 | $$+19.8$$ [$$+18.5$$, $$+1.3$$] | 0.359 |

Differences are relative to the $$\Lambda$$CDM fit ($$n_s=0.9642$$, $$-2\ln\mathcal L=867.4$$). The $$N_B=2$$ row requires one non-conformally coupled scalar with $$(1-8\xi)^2=1$$, for which the resummation is incomplete, and is shown for comparison only. Using instead the published $$\ell\ge30$$ values of $$g$$ and $$\ln\beta$$ and fitting only the amplitude, the two-loop form worsens the $$\ell<30$$ fit by $$+23.6$$ with $$C_2/C_2^{\Lambda\rm CDM}=0.32$$, and the resummed form by $$+4.6$$ with $$0.51$$; the two forms differ by more than 10% below $$\ell\approx33$$, so the $$\ell\ge30$$ fit has to be redone with the resummed spectrum, as in the table.

For $$N_B=1$$ the refit has $$q_c=\kappa\lvert g\rvert q_*=2.11\times10^{-4}\,{\rm Mpc}^{-1}$$, that is $$\ell_c=2.9$$, and the predicted temperature spectrum relative to $$\Lambda$$CDM at $$\ell=2,\dots,8$$ is $$0.565,\ 0.623,\ 0.681,\ 0.734,\ 0.781,\ 0.821,\ 0.855$$. The observed quadrupole is $$0.22$$ of the $$\Lambda$$CDM value and $$0.39$$ of the resummed prediction, so the model moves towards the data without reaching it. The last row shows why the fit at all multipoles matters: the $$\ell\ge30$$ data constrain $$g$$ and $$\ln\beta$$ only along a degenerate direction, and fitting to $$\ell\ge30$$ alone selects a point that overshoots the suppression at low $$\ell$$. The profile of the total $$\Delta(-2\ln\mathcal L)$$ in $$\ln\beta$$ at $$N_B=1$$ is 14.8, 10.9, 7.8, 6.1, 5.0, 5.8, 7.9, 11.9 at $$\ln\beta=0.5,\ 0.7,\ 0.85,\ 0.95,\ 1.112,\ 1.25,\ 1.4,\ 1.6$$, which gives $$\ln\beta=1.11^{+0.16}_{-0.15}$$ at one unit above the minimum, and the bound (11) is $$2.6$$ ($$2.3$$ to $$3.1$$) there.

The cost is shared between low and high multipoles ($$+2.1$$ and $$+2.9$$), the latter mostly from an excess of the model over $$\Lambda$$CDM at $$\ell=30$$ to $$50$$. With one more primordial parameter than $$\Lambda$$CDM, a difference of $$5$$ in $$-2\ln\mathcal L$$ corresponds to a Gaussian equivalent of $$\sqrt5\approx2.2$$. The background was held fixed, and $$(\Omega_bh^2,\Omega_ch^2)$$ are partly degenerate with the spectral shape at $$\ell>30$$, so freeing them could change the gap; this was not done here. A fit to the full Planck temperature and polarization likelihood with the background free is needed to turn these numbers into a significance.

## 7. A running sum rule

Write $$n_s-1$$, $$\alpha_s$$ and $$\beta_s$$ for the first three logarithmic derivatives of $$\ln\Delta^2$$ with respect to $$\ln q$$ (the tilt, its running and the running of the running).

<div class="proposition" markdown="1">
**Proposition 3** (running sum rule). To first order in $$u$$, the spectrum (4) satisfies $$n_s-1=-u(L-1)$$, $$\alpha_s=u(L-2)$$ and $$\beta_s=-u(L-3)$$, with $$L=\ln\big(q/(\beta\lvert g\rvert q_*)\big)$$, and hence

$$
\beta_s+2\alpha_s+(n_s-1)=0 \tag{12}
$$

at every $$q$$, for all values of $$g$$ and $$\beta$$.
</div>

<div class="proof" markdown="1">
*Proof.* To first order, $$\ln\Delta^2=\ln\Delta_0^2-u\ln(\beta u)=\ln\Delta_0^2+uL$$. Write $$\partial=d/d\ln q$$. Then $$\partial u=-u$$ and $$\partial L=1$$, so $$(\partial+1)u=0$$ and $$(\partial+1)(uL)=u$$, hence $$(\partial+1)^2(uL)=0$$. The left side of (12) is $$(\partial^3+2\partial^2+\partial)\ln\Delta^2=\partial(\partial+1)^2\ln\Delta^2$$, which vanishes because $$(\partial+1)^2$$ annihilates $$uL$$ and $$\partial$$ annihilates the constant. The individual expressions follow by differentiating $$uL$$ three times.
</div>

At the pivot the full forms, evaluated by finite differences of $$\ln\Delta^2$$, give for the resummed spectrum (9) at the $$N_B=1$$ refit $$n_s-1=-0.0247$$, $$\alpha_s=+0.0161$$, $$\beta_s=-0.0077$$, with left side of (12) equal to $$-2\times10^{-4}$$; for (9) at the published $$\ell\ge30$$ values $$-0.0323$$, $$+0.0202$$, $$-0.0084$$ and $$-3\times10^{-4}$$; and for the two-loop form (4) at those values $$-0.0317$$, $$+0.0191$$, $$-0.0062$$ and $$+2\times10^{-4}$$. The resummation does not affect (12) at the pivot, where $$x\approx0.004$$. The ratio $$\alpha_s/(n_s-1)$$ is $$-(L-2)/(L-1)$$ at first order, $$-0.57$$ to $$-0.61$$ for $$L=3.3$$ to $$3.6$$, and $$-0.60$$ to $$-0.65$$ from the full forms. The value $$-1$$ quoted in [[AGS]](#ref-ags) is the limit $$L\to\infty$$. A measurement of $$(n_s,\alpha_s,\beta_s)$$ tests (12) for any parameters, but errors on $$\beta_s$$ of order $$0.01$$ are too large for this at present.

## 8. What the result does not exclude

The low-$$\ell$$ result and the bound (11) are statements within stated hypotheses. Each of the following changes them, and the conditions are listed without choosing among them.

- Non-conformal scalars ($$N_B>1$$). The mixed correlators $$\langle\Theta_\Phi\Theta_A\rangle$$ are of the same order as the gauge loop and are not computed, so (10) and (11) are then a model; a different treatment could move $$N_{B,\max}$$. The exclusion of the solution with $$N_B\approx88$$ holds in the model for any $$\ln\beta<4.6$$, and its large crossover scale ($$\ell_c\approx600$$) is independent of the model.
- A larger $$\ln\beta$$. The bound is $$N_B\le0.864\beta$$; values of $$\ln\beta$$ above the fitted range admit more non-conformal scalars (6.4 at $$\ln\beta=2$$).
- A flavour-singlet quartic coupling (violating A3) adds a second crossover scale and at least one parameter, and could alter the low-$$\ell$$ shape.
- Other finite counterterms. The prescription $$\mathcal B(0)=0$$ rests on Lemma 1, and the coefficient of the curvature counterterm is absorbed in $$\ln\beta$$; a finite curvature-squared term would change the large-$$x$$ tail $$\mathcal F\approx0.27/x$$, which is subleading in (9).
- The fit to data. The temperature comparison uses an approximate likelihood and a fixed background; with a free background and polarization the gap to $$\Lambda$$CDM could differ in either direction.
- Late times. The only term of the correlator independent of $$q$$ is the contact term of Lemma 1, which does not enter $$\mathcal B$$, so the fitted dual contains no constant that could be identified with a late-time vacuum energy; any such energy requires ingredients outside the fitted dual theory.

## 9. Open points

- The power counting of section 3 has not been carried out diagram by diagram to all orders; the corrections are of relative order $$1/N_A<1.1\times10^{-3}$$ at $$N_B=1$$.
- The value $$b=\tfrac{11}{12}a_1$$ and hence $$x_*=e^{-11/12}$$ are numerical; a closed-form evaluation of (7) is not known to us.
- The positivity of $$\mathcal F$$ and the boundedness of $$x\mathcal F$$ are established on a grid of $$10^{-4}\le x\le10^{4}$$, not proved.
- The mixed diagrams for $$N_B>1$$ remain to be computed.
- It is not known whether a non-perturbative treatment of the dual theory at $$\lambda\gtrsim1$$, relevant only at $$\ell<1$$ for the fitted parameters, changes the positivity bound.
- The comparison with data needs the full temperature and polarization likelihood with a free background, and a comparison with ACT and SPT data at high $$\ell$$.

## References

1. <a id="ref-ms09"></a>McFadden, Skenderis, "Holography for Cosmology," Phys. Rev. D 81, 021301 (2010), [arXiv:0907.5542](https://arxiv.org/abs/0907.5542).
2. <a id="ref-ms10"></a>McFadden, Skenderis, "The Holographic Universe," J. Phys. Conf. Ser. 222, 012007 (2010), [arXiv:1001.2007](https://arxiv.org/abs/1001.2007).
3. <a id="ref-acdgs"></a>Afshordi, Corianò, Delle Rose, Gould, Skenderis, "From Planck data to Planck era: Observational tests of Holographic Cosmology," Phys. Rev. Lett. 118, 041301 (2017), [arXiv:1607.04878](https://arxiv.org/abs/1607.04878).
4. <a id="ref-ags"></a>Afshordi, Gould, Skenderis, "Constraining holographic cosmology using Planck data," Phys. Rev. D 95, 123505 (2017), [arXiv:1703.05385](https://arxiv.org/abs/1703.05385).
5. <a id="ref-cds"></a>Corianò, Delle Rose, Skenderis, "Two-point function of the energy-momentum tensor and generalised conformal structure," [arXiv:2008.05346](https://arxiv.org/abs/2008.05346).
6. <a id="ref-gtk"></a>Giombi, Tarnopolsky, Klebanov, "On $$C_J$$ and $$C_T$$ in Conformal QED," [arXiv:1602.01076](https://arxiv.org/abs/1602.01076).
7. <a id="ref-bk18"></a>BICEP/Keck Collaboration, "BICEP / Keck XIII: Improved Constraints on Primordial Gravitational Waves using Planck, WMAP, and BICEP/Keck Observations through the 2018 Observing Season," Phys. Rev. Lett. 127, 151301 (2021), [arXiv:2110.00483](https://arxiv.org/abs/2110.00483).
8. <a id="ref-planck-v"></a>Planck Collaboration, "Planck 2018 results. V. CMB power spectra and likelihoods," Astron. Astrophys. 641, A5 (2020), [arXiv:1907.12875](https://arxiv.org/abs/1907.12875).
9. <a id="ref-planck-vi"></a>Planck Collaboration, "Planck 2018 results. VI. Cosmological parameters," Astron. Astrophys. 641, A6 (2020), [arXiv:1807.06209](https://arxiv.org/abs/1807.06209).
