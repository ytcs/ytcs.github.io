---
title: How a dark-dimension modulus couples to baryons
date: 2026-10-29
category: dark-sector
order: 5
description: In an unwarped dark-dimension compactification with the Standard Model on a brane and no brane Einstein-Hilbert term, a field that changes only the size of the dark circle couples to baryons with one third of its coupling to dark matter, which the Cassini bound excludes at the dark-matter coupling of the fading-dark-sector fits, so a viable field needs further light moduli with an aligned direction or one of the relaxed hypotheses listed.
status: Reduction, couplings, the radion ratio and the bound on baryon-decoupled directions proved at tree level for a flat torus with an arbitrary metric; the warped case is a numerical calculation in a linearised two-brane effective theory; the extension to other internal spaces is an argument, not a proof.
excludes: In unwarped tree-level models with Kaluza-Klein gravitons as dark matter, the Standard Model on a point-like brane with frozen couplings and no brane Einstein-Hilbert term, a light field whose only mass-relevant components are the radius of the dark circle, or the dark circle together with the common size of the other dimensions, with a dark-matter coupling near 0.05 and a baryon coupling within the Cassini bound. Not excluded are further light moduli with an aligned direction, a brane Einstein-Hilbert term, warping, brane couplings that depend on moduli, and dark matter in the tower of a bulk scalar.
---

In the dark-dimension scenario the dark matter is the Kaluza–Klein (KK) tower of a micron-sized extra circle. Fits of a fading dark sector to BAO, CMB and supernova data [[BOVW]](#ref-bovw) describe a dark-matter mass that drifts as $$m_{\rm DM}\propto e^{-c'\phi}$$ with $$c'=0.05\pm0.01$$, for a light field $$\phi$$ in reduced Planck units, and treat baryons as uncoupled to $$\phi$$. We ask what coupling $$\beta_b=-d\ln m_b/d\phi$$ to baryons, which live on the Standard Model brane, the geometry implies, and whether it can satisfy the Cassini bound on the post-Newtonian parameter $$\gamma$$, which requires $$\lvert\beta_b\rvert<2.5\times10^{-3}$$.

The answer depends on which moduli $$\phi$$ moves. We work at tree level with an unwarped product compactification, brane masses fixed in units of the higher-dimensional Planck mass, and no Einstein–Hilbert term on the brane. If the only mass-relevant component of $$\phi$$ is the radius of the dark circle, then $$\beta_b=c'/3$$ (Proposition 2). Brane masses are fixed in higher-dimensional Planck units while the four-dimensional Planck mass grows with the internal volume, so in four-dimensional Planck units a circle of radius $$R$$ gives brane masses $$\propto R^{-1/2}$$ and KK masses $$\propto R^{-3/2}$$. At $$c'=0.05$$ the value of $$\gamma-1$$ is then 44 times the Cassini bound (section 3). The same holds in the minimal model, where only the dark circle and the common size of $$n$$ further dimensions are light (Corollary 1). With more light moduli, baryons decouple exactly on the directions that leave the total internal volume fixed, and on those $$\lvert c'\rvert\le\sqrt{n/(n+1)}$$, which is $$0.71$$ to $$0.93$$ for $$n=1,\dots,6$$ (Theorem 1). At the maximal value the force that $$\phi$$ mediates between dark-matter particles, $$2c'^2$$ times gravity, would be 1.0 to 1.7 times gravity, while the satellite bound is $$c'\lesssim0.2$$. A decoupled direction with $$c'=0.05$$ exists, about $$87^\circ$$ from the extremal one, and nothing in the geometry selects it (section 5.1).

The exclusion does not apply with a brane Einstein–Hilbert term that supplies at least about 87 to 91 per cent of $$M_P^2$$, which is consistent only if the moduli of the other dimensions are not dynamical four-dimensional fields, with a warped dark dimension of $$kL\gtrsim1.4$$ for curvature scale $$k$$ and length $$L$$, with brane couplings that depend on the moduli, or with dark matter in the KK tower of a bulk scalar (sections 5.2 to 5.4).

Montero, Vafa and Valenzuela proposed the dark dimension [[MVV]](#ref-mvv), and Gonzalo, Montero, Obied and Vafa proposed its KK gravitons as dark matter [[GMOV]](#ref-gmov). Bedroya, Obied, Vafa and Wu [[BOVW]](#ref-bovw) fitted the fading dark sector and noted (their section III) that the field may change the circle either at fixed total internal volume, which keeps Newton's constant fixed in Standard Model units, or with the volume varying; they and Stewart [[Stewart]](#ref-stewart) take baryons to be uncoupled. Vafa [[Vafa]](#ref-vafa) (appendix A) observes that the fitted $$c'\simeq0.05$$ is far below the value $$\sqrt{3/2}$$ of the pure radion and writes $$\sin\theta=c'/\sqrt{3/2}\simeq0.04$$ for the projection of the light field on the radion. Mantry and Furuuchi [[MF]](#ref-mf) note that a fixed total internal volume makes Newton's constant independent of the shape modulus (their section 5; their canonical field $$\chi=\sqrt2M_P\ln\sigma$$ corresponds to $$c'=1/\sqrt2$$, the case $$n=1$$ below) and build a five-dimensional model in which an induced brane-localised Einstein–Hilbert term dominates $$M_P^2$$ (their section 4). The bound of Theorem 1 can be read off from the tower and species vectors of Etheredge et al. [[EHRRV]](#ref-ehrrv) (their eqs. (1.3), (1.6) and (1.11)), as we note after the theorem. We add the ratio $$\beta_b/c'$$ with its Cassini consequence, the bound over the full metric moduli space of a torus, the closed form for the minimal model, and quantitative conditions for the cases of section 5, including a ghost condition for the brane-term case. Schemes that fix only a density ratio leave the dark-matter mass as a free modulus, as discussed on the page on [free dark-sector parameters](/quantum-gravity/dark-sector-moduli/); the modulus considered here is the size of the dark dimension.

## 1. A circle with a brane

The mechanism is already present with one extra dimension, and the general case only adds bookkeeping. Take five dimensions with Planck mass $$M_5$$ (action $$\tfrac12M_5^3\int d^5x\sqrt{-g_5}R_5$$), one dimension compactified on a circle of radius $$R$$, and the Standard Model on a point-like brane. Write the metric as

$$
ds^2=g_{\mu\nu}(x)\,dx^\mu dx^\nu+R(x)^2dy^2,\qquad y\sim y+2\pi .
$$

Reducing on the circle gives $$\tfrac12M_P^2\,(R/R_0)\sqrt{-g}\,\mathcal R$$ with $$M_P^2=2\pi R_0M_5^3$$, where $$\mathcal R$$ is the Ricci scalar of $$g_{\mu\nu}$$ and $$R_0$$ the present radius. The brane couples only to $$g_{\mu\nu}$$ and has parameters fixed in units of $$M_5$$, so every brane mass is a constant $$m$$ in the frame of $$g_{\mu\nu}$$. We call this the brane frame; it is the Jordan frame of the brane matter. The coefficient of $$\mathcal R$$ depends on a field, which is a scalar–tensor theory (Brans–Dicke with parameter $$\omega=0$$), and masses and couplings are compared in the Einstein frame, where the coefficient of $$\mathcal R$$ is constant and equal to $$M_P^2/2$$. Rescaling $$g_{\mu\nu}=(R_0/R)\,g^E_{\mu\nu}$$ takes the brane frame to the Einstein frame, and a mass $$m$$ that is constant in the brane frame becomes $$m\,(R_0/R)^{1/2}$$. The KK masses of the circle are $$\lvert j\rvert/R$$ in the brane frame, so in the Einstein frame

$$
m_b\propto R^{-1/2},\qquad m_{\rm DM}\propto R^{-3/2}. \tag{1}
$$

Brane masses fall with the radius because the Planck mass grows with it, and KK masses fall faster because of the additional $$1/R$$.

For a light field $$\phi$$ we define the coupling of a species of Einstein-frame mass $$m_X$$ by

$$
\beta_X\equiv-\frac{d\ln m_X}{d\phi},
$$

with $$\phi$$ canonically normalised in units of the reduced Planck mass, $$\mathcal L=\tfrac12R-\tfrac12(\partial\phi)^2$$. Exchange of $$\phi$$, when it is light on the scale of the separation, adds a fraction $$2\beta_X\beta_Y$$ to the Newtonian attraction between non-relativistic bodies of species $$X$$ and $$Y$$. The dark-matter coupling of [[BOVW]](#ref-bovw) is $$c'=\beta_{\rm DM}$$, and the force between dark-matter particles, $$2c'^2$$ times gravity (their eq. (18)), uses the same normalisation.

By (1), if $$\phi$$ moves only $$R$$, then $$\beta_b/\beta_{\rm DM}=\tfrac12/\tfrac32=\tfrac13$$, whatever the normalisation of $$\phi$$ and whatever the potential. The Einstein-frame action is $$\tfrac12R-\tfrac34(\partial\ln R)^2$$, so $$\phi=\sqrt{3/2}\,\ln R$$ and the pure radion has $$\beta_b=\tfrac12\sqrt{2/3}=1/\sqrt6$$ and $$\beta_{\rm DM}=\sqrt{3/2}$$. The value $$2\beta_b^2=\tfrac13$$ is the Brans–Dicke value for $$\omega=0$$, since a linear coupling $$\beta_b$$ corresponds to $$2\omega+3=1/(2\beta_b^2)$$.

A massless scalar with a constant linear coupling $$\beta_b$$ to the Sun is a Brans–Dicke field with $$\alpha_0^2=1/(2\omega+3)$$ in the notation of Will's review [[Will]](#ref-will), where the physical metric is $$A^2(\varphi)\tilde g_{\mu\nu}$$, the Einstein-frame action is $$(16\pi G)^{-1}\int[\tilde R-2(\tilde\partial\varphi)^2]$$ (his eq. (33)) and $$\alpha_0=d\ln A/d\varphi$$. Matching $$\tfrac12(\partial\phi)^2$$ to $$M_P^2(\partial\varphi)^2$$ gives $$\phi=\sqrt2M_P\varphi$$ and $$\alpha_0^2=2\beta_b^2$$. The post-Newtonian parameter is $$\gamma-1=-1/(2+\omega)=-2\alpha_0^2/(1+\alpha_0^2)$$, that is

$$
\gamma-1=-\frac{4\beta_b^2}{1+2\beta_b^2}. \tag{2}
$$

The Cassini radio-link experiment measured $$\gamma-1=(2.1\pm2.3)\times10^{-5}$$ [[BIT]](#ref-bit), as quoted in [[Will]](#ref-will), so at two standard deviations $$\gamma-1>-2.5\times10^{-5}$$ and

$$
\lvert\beta_b\rvert<2.5\times10^{-3}. \tag{3}
$$

This applies when the range of $$\phi$$ exceeds an astronomical unit, as it does for a field that rolls on cosmological time scales, and when the coupling is linear. By (1) every brane species has the same coupling, so tests of the equivalence principle among ordinary materials do not constrain it and (3) is the relevant bound. For the pure radion $$\beta_b=0.408$$ exceeds the bound (3) by a factor 160.

## 2. Several internal dimensions

We take $$D=n+5$$ dimensions: the four-dimensional spacetime, the dark circle and $$n$$ further circles, so that the internal space is a torus $$T^{n+1}$$ (the case $$n=0$$ is section 1). The torus makes the computation explicit; the extension to other internal spaces is discussed after Theorem 1. The hypotheses are as follows.

(H1) Tree-level two-derivative gravity in $$D$$ dimensions, with action $$\tfrac12M_D^{D-2}\int d^Dx\sqrt{-g_D}R_D$$, and the product geometry
$$ds^2=g_{\mu\nu}(x)\,dx^\mu dx^\nu+h_{ab}(x)\,dy^a dy^b$$, $$y^a\sim y^a+2\pi$$, where $$h_{ab}$$ is any positive-definite matrix of moduli, constant on the torus. Potentials for the moduli are allowed; they do not enter the kinetic terms or the couplings computed here. Graviphotons and fluxes are absent.

(H2) The Standard Model lives on a point-like 3-brane and couples only to the induced metric $$g_{\mu\nu}$$. Every brane parameter (position, tension, couplings, any dilaton) is fixed in units of $$M_D$$, so every brane mass, including the nucleon mass through $$\Lambda_{\rm QCD}$$, is constant in the brane frame. The brane may carry an Einstein–Hilbert term $$\tfrac12M_b^2\sqrt{-g}\,\mathcal R$$; the case $$M_b=0$$ is the setting of the introduction.

(H3) The dark matter is the KK tower of the first circle, with momentum $$p=(j,0,\dots,0)$$, so that $$m_j^2=j^2h^{00}$$ in units of $$M_D$$ in the brane frame, where $$h^{00}$$ is the $$00$$ component of $$h^{-1}$$. If the brane carries an Einstein–Hilbert term, (H3) is understood for the modes that vanish at the brane, whose masses are unaffected by it.

(H4) A single light canonically normalised scalar $$\phi$$ moves the moduli $$h_{ab}$$ along a direction at the present point, and may also move fields on which neither $$m_b$$ nor $$m_{\rm DM}$$ depends. Its couplings are linear and it is massless on solar-system scales.

Let $$h_0$$ be the present value of $$h$$, $$e^S=\sqrt{\det h/\det h_0}$$ the relative volume of the torus, and $$V_0$$ the present volume. The four-dimensional Planck mass is $$M_P^2=M_D^{D-2}V_0+M_b^2$$, and we define the bulk fraction

$$
F\equiv\frac{M_D^{D-2}V_0}{M_P^2}\in(0,1],
$$

so that the brane term supplies the fraction $$1-F$$ of $$M_P^2$$ and $$F=1$$ means no brane term. A tangent vector to the moduli space at the present point is a symmetric matrix $$\delta h$$; we describe it by $$Q=h^{-1/2}\delta h\,h^{-1/2}$$, a symmetric matrix with $$\operatorname{tr}Q=2\,\delta S$$. In the diagonal case $$h={\rm diag}(e^{2\sigma_a})$$ we have $$Q={\rm diag}(2\delta\sigma_a)$$. Let $$e_0$$ be the unit vector of the first circle and $$\hat e\equiv h^{-1/2}e_0/\lvert h^{-1/2}e_0\rvert$$.

The computation has the shape of section 1: reduce to a scalar–tensor theory in the brane frame, rescale to the Einstein frame, read off the kinetic metric of the moduli, and convert the masses of (H2) and (H3) to the Einstein frame. Its result is the following proposition, and the derivation is given after the statement.

<div class="proposition" markdown="1">
**Proposition 1** (moduli metric and couplings). Assume (H1)–(H4). At the present point the Einstein-frame kinetic form on tangent vectors $$Q$$ is

$$
K(Q,Q)=\tfrac14F\big[\operatorname{tr}Q^2-(\operatorname{tr}Q)^2\big]+\tfrac38F^2(\operatorname{tr}Q)^2, \tag{4}
$$

and a light field moving along $$Q$$ with $$K(Q,Q)=1$$ has the couplings

$$
\beta_b=\tfrac14F\operatorname{tr}Q,\qquad c'=\tfrac12\hat e^{\sf T}Q\hat e+\tfrac14F\operatorname{tr}Q. \tag{5}
$$

In the diagonal case, with $$v=(1,\dots,1)$$, $$K=F(\mathbb 1-vv^{\sf T})+\tfrac32F^2vv^{\sf T}$$ on the vector $$u_a=\delta\sigma_a/d\phi$$. It has eigenvalue $$F$$ with multiplicity $$n$$ on $$\sum_au_a=0$$ and eigenvalue

$$
F\big[\tfrac32(n+1)F-n\big] \tag{6}
$$

on the volume direction $$u_a=1$$. For $$F=1$$ this is the standard metric $$\mathbb 1+\tfrac12vv^{\sf T}$$ of the radii of a torus.
</div>

<div class="proof" markdown="1">
*Proof.* The argument has three steps; the only structure is in the metric (4).

Step 1, reduction. The Christoffel symbols with at least one torus index are $$\Gamma^\mu_{ab}=-\tfrac12\partial^\mu h_{ab}$$ and $$\Gamma^a_{\mu b}=\tfrac12h^{ac}\partial_\mu h_{cb}$$; the others involving the torus vanish. Inserting them into the Ricci scalar gives

$$
R_D=\mathcal R-2\,\Box S-(\partial S)^2-\tfrac14\operatorname{tr}\!\big[(h^{-1}\partial h)^2\big], \tag{7}
$$

where $$\mathcal R$$, $$\Box$$ and the contractions refer to $$g_{\mu\nu}$$. For diagonal $$h$$ this reads $$\mathcal R-2\Box S-(\partial S)^2-\sum_a(\partial\sigma_a)^2$$, and for one circle it is the familiar $$\mathcal R-2e^{-\sigma}\Box e^{\sigma}$$. A direct evaluation of the Ricci scalar for a general $$2\times2$$ block $$h_{ab}$$ depending on both coordinates of a conformally flat two-dimensional base agrees with (7); the coefficients do not depend on the dimension of the base. Since $$\sqrt{-g_D}=e^S\sqrt{\det h_0}\,\sqrt{-g}$$ and $$\int e^S\Box S=-\int e^S(\partial S)^2$$, integrating over the torus gives the bulk Lagrangian $$\tfrac12M_D^{D-2}V_0\,e^S\sqrt{-g}\,[\mathcal R+(\partial S)^2-\tfrac14\operatorname{tr}(h^{-1}\partial h)^2]$$. Adding the brane term and setting $$M_P=1$$, the Lagrangian in the brane frame is

$$
\sqrt{-g}\Big[\tfrac12\Phi\,\mathcal R+\tfrac12Fe^S\big((\partial S)^2-\tfrac14\operatorname{tr}(h^{-1}\partial h)^2\big)\Big],\qquad\Phi\equiv Fe^S+1-F. \tag{8}
$$

Step 2, Einstein frame. In four dimensions the rescaling $$g_{\mu\nu}=\Phi^{-1}g^E_{\mu\nu}$$ gives $$\mathcal R=\Phi[R_E+3\Box_E\ln\Phi-\tfrac32(\partial\ln\Phi)^2_E]$$ and $$\sqrt{-g}=\Phi^{-2}\sqrt{-g_E}$$, so that, up to the total derivative $$3\Box_E\ln\Phi$$, $$\sqrt{-g}\,\Phi\,\mathcal R=\sqrt{-g_E}\,[R_E-\tfrac32(\partial\ln\Phi)_E^2]$$; for any scalar $$f$$, $$\sqrt{-g}\,(\partial f)^2=\Phi^{-1}\sqrt{-g_E}\,(\partial f)^2_E$$. With $$\partial\ln\Phi=w\,\partial S$$ and $$w\equiv Fe^S/\Phi$$, the Lagrangian (8) becomes $$\sqrt{-g_E}[\tfrac12R_E-\tfrac12K]$$ with kinetic form $$K=\tfrac14w\operatorname{tr}(h^{-1}\partial h)^2-\tfrac14w(\operatorname{tr}h^{-1}\partial h)^2+\tfrac38w^2(\operatorname{tr}h^{-1}\partial h)^2$$. At the present point $$w=F$$ and $$h^{-1}\partial h$$ becomes $$Q$$, which gives (4).

Step 3, masses. A mass $$m$$ that is constant in the brane frame is $$m\Phi^{-1/2}$$ in the Einstein frame. The brane masses are constant in the brane frame by (H2); the KK masses are $$m_j=\lvert j\rvert\sqrt{h^{00}}$$ in the brane frame by (H3). Along the direction $$Q$$ at the present point one has $$\delta\ln h^{00}=-\hat e^{\sf T}Q\hat e$$ and $$\delta\ln\Phi^{-1/2}=-\tfrac14F\operatorname{tr}Q$$. Then $$\beta_X=-d\ln m_X/d\phi$$ with $$d\phi=\sqrt{K(Q,Q)}$$ gives (5).

In the diagonal case $$\operatorname{tr}Q^2=4\sum u_a^2$$ and $$\operatorname{tr}Q=2\sum u_a$$ give $$K=F[\sum u_a^2-(\sum u_a)^2]+\tfrac32F^2(\sum u_a)^2$$. The matrix $$vv^{\sf T}$$ annihilates vectors with $$\sum u_a=0$$ and has eigenvalue $$n+1$$ on $$v$$, which gives (6). $$\square$$
</div>

Two limits fix the normalisation. For $$n=0$$ the bracket in (4) vanishes and $$K=\tfrac38F^2(\operatorname{tr}Q)^2=\tfrac32F^2(\delta\ln R)^2$$, which for $$F=1$$ is the coefficient found in section 1. For $$F=1$$ and the isotropic volume direction, (5) and (6) give $$2\beta_b^2=(n+1)/(n+3)$$, the known strength of the force of the volume modulus of $$n+1$$ equal extra dimensions. This is the largest baryon coupling of any unit direction at $$F=1$$: by (5) $$\beta_b=\tfrac12v\cdot u$$ in the diagonal case, and $$v$$ is an eigenvector of $$K$$, so the maximum is at $$u\propto v$$.

## 3. The radion direction and the Cassini bound

Suppose first that $$\phi$$ changes the radius of the dark circle and nothing else that enters a mass. We call $$Q$$ a radion direction if $$Q=a\,\hat e\hat e^{\sf T}+Q_\perp$$ with $$\operatorname{tr}Q_\perp=0$$ and $$\hat e^{\sf T}Q_\perp\hat e=0$$. The components $$Q_\perp$$ move the other moduli (shapes of the other circles, off-diagonal components, anything else that leaves the volume and $$h^{00}$$ unchanged); they affect neither coupling in (5).

<div class="proposition" markdown="1">
**Proposition 2** (radion coupling). Assume (H1)–(H4) and a radion direction. Then

$$
\frac{\beta_b}{c'}=\frac{F}{2+F},
$$

which is $$\tfrac13$$ without a brane Einstein–Hilbert term. For the pure radion ($$Q_\perp=0$$), $$\beta_b=1/\sqrt6$$ for every $$F$$ and $$c'=(2+F)/(\sqrt6\,F)$$.
</div>

<div class="proof" markdown="1">
*Proof.* For $$Q=a\hat e\hat e^{\sf T}+Q_\perp$$ one has $$\operatorname{tr}Q=a$$ and $$\hat e^{\sf T}Q\hat e=a$$, so (5) gives $$\beta_b=\tfrac14Fa/\sqrt K$$ and $$c'=(\tfrac12+\tfrac14F)a/\sqrt K$$, whose ratio is $$F/(2+F)$$ whatever the normalisation. For the pure radion, (4) gives $$K=\tfrac14F(a^2-a^2)+\tfrac38F^2a^2=\tfrac38F^2a^2$$, so $$a=\sqrt{8/3}/F$$ and $$\beta_b=\tfrac14F a=1/\sqrt6$$, $$c'=\tfrac14(2+F)a=(2+F)/(\sqrt6F)$$. $$\square$$
</div>

The observed $$c'\approx0.05$$ is far below $$\sqrt{3/2}$$, so the light field is mostly something other than the radion, as Vafa assumes [[Vafa]](#ref-vafa). If the remaining components leave the volume and the dark-circle radius unchanged, Proposition 2 says that the ratio $$\beta_b/c'$$ does not depend on how small the radion component is. Without a brane term, $$c'=0.05$$ gives $$\beta_b=0.0167$$ and, by (2), $$\gamma-1=-1.11\times10^{-3}$$, which is 44 times the two-standard-deviation bound $$2.5\times10^{-5}$$. The fitted range $$c'=0.04$$–$$0.06$$ gives factors $$28$$ to $$64$$. Equivalently, by (3), every $$c'>3\times2.5\times10^{-3}=7.5\times10^{-3}$$ is excluded for such a field.

The fitted value $$c'=0.05$$ was obtained with $$\beta_b=0$$ [[BOVW]](#ref-bovw). A fit that includes the baryon coupling $$\beta_b=c'/3$$ could return a different $$c'$$, which has not been computed; what Proposition 2 excludes is $$c'>7.5\times10^{-3}$$ for a radion-direction field, whatever the fit.

<div class="remark" markdown="1">
**Remark** (comparison with published treatments). Stewart [[Stewart]](#ref-stewart), section 2.1, takes the field to be the radion of the dark dimension and states that the Standard Model "does not couple to the radion at leading order, so the baryons are uncoupled dust", which enters his eq. (2.2c) as $$\rho_b\propto a^{-3}$$; his footnote 2 (in section 3) adds that the masses of Standard Model fields on the brane would typically not depend on $$R$$. In units of $$M_D$$ this is hypothesis (H2). As a statement about masses in four-dimensional Planck units, under (H1)–(H3) it holds on the directions that preserve the total internal volume (section 4), and for a field whose only mass-relevant component is the radion Proposition 2 gives $$\beta_b=c_2/3$$, where $$c_2$$, the logarithmic slope of the tower mass in his eq. (2.10), equals $$c'$$. Bedroya et al. [[BOVW]](#ref-bovw), section IV, state that visible matter, being localised on the Standard Model brane, "does not couple directly to $$\phi$$". Under (H1)–(H3) this holds on the fixed-volume branch that their section III allows, and fails on the branch with varying volume, where $$\beta_b=\tfrac14F\operatorname{tr}Q\ne0$$.
</div>

## 4. Directions that leave baryons uncoupled

By (5), $$\beta_b=0$$ if and only if $$\operatorname{tr}Q=0$$, that is, if the field preserves the total internal volume in units of $$M_D$$; this holds for every $$F>0$$. The question is how large $$c'$$ can be on such a direction. On the slice $$\operatorname{tr}Q=0$$ the metric (4) is the flat trace form $$\tfrac14F\operatorname{tr}Q^2$$ and $$c'$$ is linear, so the maximum is a Cauchy–Schwarz inequality in the space of symmetric matrices.

<div class="theorem" markdown="1">
**Theorem 1** (decoupled directions). Assume (H1)–(H4) with internal space $$T^{n+1}$$, $$n\ge1$$, and any flat metric $$h$$. A unit direction has $$\beta_b=0$$ if and only if $$\operatorname{tr}Q=0$$. Every such direction has

$$
\lvert c'\rvert\le\sqrt{\frac{n}{(n+1)F}}, \tag{9}
$$

with equality exactly for $$Q\propto\hat e\hat e^{\sf T}-\mathbb 1/(n+1)$$. Without a brane term the bound is $$\sqrt{n/(n+1)}$$.
</div>

<div class="proof" markdown="1">
*Proof.* The first statement is (5). On the slice $$\operatorname{tr}Q=0$$ the normalisation $$K=1$$ reads $$\operatorname{tr}Q^2=4/F$$, and by (5) $$c'=\tfrac12\hat e^{\sf T}Q\hat e=\tfrac12\operatorname{tr}(Q\,\hat e\hat e^{\sf T})=\tfrac12\operatorname{tr}(QP)$$ with $$P\equiv\hat e\hat e^{\sf T}-\mathbb 1/(n+1)$$, since $$\operatorname{tr}Q=0$$. The matrix $$P$$ is symmetric and traceless with $$\operatorname{tr}P^2=1-\tfrac2{n+1}+\tfrac1{n+1}=\tfrac n{n+1}$$. Cauchy–Schwarz for the trace inner product gives $$\lvert\operatorname{tr}(QP)\rvert\le(\operatorname{tr}Q^2)^{1/2}(\operatorname{tr}P^2)^{1/2}=2\sqrt{n/((n+1)F)}$$, with equality if and only if $$Q\propto P$$, and $$P$$ is itself traceless. $$\square$$
</div>

These are the directions on which the assumption of uncoupled baryons made in [[BOVW]](#ref-bovw) and [[Stewart]](#ref-stewart) holds exactly under (H1)–(H3). In the diagonal case the extremal direction increases the dark circle by an amount $$\delta$$ in the logarithm and decreases each of the $$n$$ other circles by $$\delta/n$$. For $$n=1$$ and $$F=1$$ it is the shape modulus of [[MF]](#ref-mf), section 5. Without a brane term the bound is:

| $$n$$ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| $$\sqrt{n/(n+1)}$$ | 0.707 | 0.816 | 0.866 | 0.894 | 0.913 | 0.926 |

The bound increases with $$n$$ and tends to $$1$$. For $$n=5$$, the value for the Standard Model on D3-branes in type IIB string theory (ten dimensions, with the dilaton and form fields fixed so that (H1) and (H2) hold), the decoupled value is $$0.913$$. For $$n=6$$ (eleven dimensions) M-theory has no point-like 3-brane, and a wrapped M5-brane has couplings that depend on the size of the wrapped cycle, which violates (H2).

*Relation to tower vectors.* In the swampland literature a limit of infinite distance in moduli space comes with a tower of states whose masses fall exponentially with the distance. The tower vector $$\zeta$$ is the gradient of $$-\ln m_{\rm tower}$$ in the moduli-space metric, with masses in four-dimensional Planck units. The species vector $$Z$$ is the same gradient for the species scale, the scale at which gravity becomes strongly coupled, which for a decompactification is $$M_D$$. Etheredge et al. [[EHRRV]](#ref-ehrrv) classify the values these vectors take in asymptotic limits. Here the metric is $$K$$ at $$F=1$$. Let $$\zeta$$ be the vector dual to $$-d\ln m_{\rm DM}$$ and $$Z$$ the one dual to $$-d\ln M_D$$. Since brane masses are proportional to $$M_D$$, $$\beta_b=\langle Z,u\rangle$$ and $$c'=\langle\zeta,u\rangle$$ for a unit direction $$u$$, and $$\beta_b=0$$ means $$u\perp Z$$. A computation with (4) gives $$\lvert\zeta\rvert^2=\tfrac32$$, $$\zeta\cdot Z=\tfrac12$$ and $$\lvert Z\rvert^2=(n+1)/(2(n+3))$$. Decompose $$\zeta=\zeta_\perp+\tfrac{\zeta\cdot Z}{\lvert Z\rvert^2}Z$$ with $$\zeta_\perp\perp Z$$. Then $$\lvert\zeta_\perp\rvert^2=\lvert\zeta\rvert^2-(\zeta\cdot Z)^2/\lvert Z\rvert^2=n/(n+1)$$, which is (9) at $$F=1$$, and $$\zeta_\perp$$ points along the extremal direction. For a general unit direction $$u$$ at an angle $$\psi$$ from the extremal direction,

$$
c'=\sqrt{\frac{n}{n+1}}\cos\psi+\frac{n+3}{n+1}\,\beta_b \qquad (F=1). \tag{10}
$$

Hence $$\cos\psi=c'\sqrt{(n+1)/n}$$ exactly when $$\beta_b=0$$, and for any $$F$$ the same relation $$\cos\psi=c'\sqrt{(n+1)F/n}$$ holds on decoupled directions by the equality case of (9). The values $$\lvert\zeta\rvert^2=\tfrac32$$ and $$\zeta\cdot Z=\tfrac12$$ are $$(\ell+d-2)/(\ell(d-2))$$ for the decompactification of $$\ell=1$$ dimension in $$d=4$$ and $$1/(d-2)$$ of that taxonomy (their eqs. (1.6) and (1.3)), and $$\lvert Z\rvert^2=1/(d-2)-1/(D-2)$$ is their eq. (1.11); here they come out of the explicit metric.

*The minimal model.* We call minimal the model in which only the dark circle and the common size of the other $$n$$ circles are light, with $$F=1$$. It is the simplest realisation of the fixed-volume branch of [[BOVW]](#ref-bovw) (their Fig. 1), in which the dark dimension grows while the other dimensions contract. The light directions form a plane, and both couplings are linear functionals on it.

<div class="corollary" markdown="1">
**Corollary 1** (the minimal model). Assume (H1)–(H4) with $$F=1$$, internal space $$T^{n+1}$$ ($$n\ge1$$) and light moduli spanning only the radius of the dark circle and the common radius of the other circles. Then the baryon-decoupled direction is unique up to sign, and it has $$\lvert c'\rvert=\sqrt{n/(n+1)}$$. For every unit direction with a given $$c'$$,

$$
\lvert\beta_b\rvert\;\ge\;\sqrt{\frac{n}{3(n+3)}}\,\sqrt{1-\tfrac23c'^2}-\frac{\lvert c'\rvert}{3}. \tag{11}
$$

For $$c'=0.05$$ the right side is $$0.272,\ 0.348,\ 0.439,\ 0.454$$ for $$n=1,2,5,6$$, which is 109 to 182 times the bound (3).
</div>

<div class="proof" markdown="1">
*Proof.* In the plane of light directions, with the metric induced by $$K$$, the functionals $$\beta_b=\langle Z,u\rangle$$ and $$c'=\langle\zeta,u\rangle$$ are given by two vectors $$Z,\zeta$$ of the plane (the dual vectors lie in the plane because $$K^{-1}=\mathbb 1-vv^{\sf T}/(n+3)$$ maps $$\operatorname{span}(e_0,v)$$ to itself) with $$\lvert\zeta\rvert^2=\tfrac32$$, $$\zeta\cdot Z=\tfrac12$$, $$\lvert Z\rvert^2=(n+1)/(2(n+3))$$ (computed above). A unit $$u$$ with $$\langle\zeta,u\rangle=c'$$ is $$u=(c'/\lvert\zeta\rvert^2)\zeta\pm\sqrt{1-c'^2/\lvert\zeta\rvert^2}\,\hat\zeta_\perp$$, so $$\beta_b=\tfrac13c'\pm\lvert Z_\perp\rvert\sqrt{1-\tfrac23c'^2}$$ with $$\lvert Z_\perp\rvert^2=\lvert Z\rvert^2-(\zeta\cdot Z)^2/\lvert\zeta\rvert^2=n/(3(n+3))$$. The smaller modulus of the two values is (11) when it is non-negative, which it is for $$\lvert c'\rvert\le\sqrt{n/(n+1)}$$. Setting $$\beta_b=0$$ gives $$c'^2=\lvert\zeta\rvert^2\big(1-\tfrac{(\zeta\cdot Z)^2}{\lvert\zeta\rvert^2\lvert Z\rvert^2}\big)=n/(n+1)$$ at the two points $$\pm u$$. $$\square$$
</div>

The fitted value $$c'=0.05\pm0.01$$ [[BOVW]](#ref-bovw) is smaller than the decoupled value by 66 to 88 times its quoted uncertainty. Conversely, (3) can be met in the minimal model only within a small angle of the decoupled direction, where $$\lvert c'\rvert$$ lies within about $$0.01$$ of $$\sqrt{n/(n+1)}$$, and there the force between dark-matter particles is $$2c'^2=1.0$$ to $$1.7$$ times gravity. This is above the bound from tidal disruption of Milky Way satellites, $$c'\lesssim0.2$$ [[BOVW]](#ref-bovw), and the stronger bound $$\lvert c_2\rvert\lesssim0.071$$ of [[Stewart]](#ref-stewart), section 6.1. At $$c'=0.2$$ the right side of (11) is still $$0.218$$ to $$0.398$$ for $$n=1,\dots,6$$, 87 to 159 times (3). In the minimal model no direction therefore satisfies both (3) and the satellite bound.

*Other internal spaces.* For a product $$S^1\times X_n$$ with $$X_n$$ Ricci-flat but not a torus we expect the same structure for the overall sizes, by the following argument, which is not a proof. The shape moduli of $$X_n$$ are traceless deformations of its metric, so they change neither the volume nor the circle, are $$K$$-orthogonal to the circle and volume directions, and enter neither mass. A light field with a shape component has a smaller $$\lvert\cos\psi\rvert$$, and (9) would still hold. Fibred geometries and a graviphoton are not treated.

## 5. Cases in which the exclusion does not apply

Proposition 2 and Corollary 1 exclude a baryon coupling within (3) at $$c'\approx0.05$$ for radion directions and in the minimal model. Beyond those two classes, one case lies inside (H1)–(H4), and the others relax a hypothesis. Each is stated with its conditions.

### 5.1 Further light moduli

Under (H1)–(H4) with light moduli beyond the dark circle and the common size of the others, Theorem 1 gives decoupled directions with $$c'$$ anywhere in $$[-c_{\max},c_{\max}]$$, $$c_{\max}=\sqrt{n/((n+1)F)}$$. The value $$c'=0.05$$ corresponds to $$\cos\psi=0.055$$ for $$n=5$$ and $$F=1$$ (from $$0.044$$ to $$0.066$$ for $$c'=0.04$$–$$0.06$$), an angle $$\psi=86.9^\circ$$ ($$87.5^\circ$$ to $$86.2^\circ$$) from the extremal direction; for $$n=1$$ it is $$85.9^\circ$$. Allowing $$\lvert\beta_b\rvert$$ up to the bound (3) moves $$\psi$$ by about $$0.4^\circ$$ at most, by (10). Equivalently, a unit direction satisfies (3) at $$F=1$$ when $$\lvert\langle Z,u\rangle\rvert<2.5\times10^{-3}$$, that is, when it lies within $$4\times10^{-3}$$ to $$5\times10^{-3}$$ radians of the fixed-volume slice for $$n=1,\dots,6$$ ($$\lvert Z\rvert=0.50$$ to $$0.62$$).

The satellite bound restricts $$\psi$$ from the other side. Since $$c'=c_{\max}\cos\psi$$ on decoupled directions, $$c'\le0.2$$ requires $$\psi\gtrsim74^\circ$$ to $$78^\circ$$ for $$n=1,\dots,6$$ at $$F=1$$, and $$\lvert c'\rvert\le0.071$$ requires $$\psi\gtrsim84^\circ$$ to $$86^\circ$$. The geometry does not fix $$\psi$$; the potential selects the light direction. Whether any potential motivated by the scenario places it near $$87^\circ$$ is not known.

### 5.2 An Einstein–Hilbert term on the brane

When $$F<1$$, Proposition 2 gives $$\beta_b=c'F/(2+F)$$ for a radion direction. The pure radion keeps its coupling $$1/\sqrt6$$ per canonical unit, so a field that is purely the radion remains excluded; at fixed observed $$c'$$, however, a smaller $$F$$ lowers $$\beta_b$$. The condition (3) becomes $$F<2r/(1-r)$$ with $$r=2.5\times10^{-3}/c'$$:

| $$c'$$ | 0.04 | 0.05 | 0.06 |
|---|---|---|---|
| largest bulk fraction $$F$$ | 0.133 | 0.105 | 0.087 |
| smallest brane share $$1-F$$ of $$M_P^2$$ | 86.7% | 89.5% | 91.3% |

The brane term must therefore supply most of $$M_P^2$$ (it is the bulk fraction $$F$$ that must be small). For $$n=0$$ the only modulus is the radion, whose pure direction has $$c'=(2+F)/(\sqrt6F)\ge1.2$$, so $$c'=0.05$$ requires a light field that is mostly a scalar on which no mass depends, with a small radion component, as (H4) allows. For $$n\ge1$$ the cap (9) on decoupled directions grows as $$F^{-1/2}$$ and is at least $$2.2$$ at $$F=0.1$$. For $$n=0$$ and a circle of circumference $$L=1\,\mu$$m, $$F=M_5^3L/M_P^2$$ equals $$0.1$$ at $$M_5\approx5\times10^8\,$$GeV (with the reduced Planck mass $$2.4\times10^{18}\,$$GeV), about half the value $$1.05\times10^9\,$$GeV at which the bulk alone supplies $$M_P^2$$ and which is within the range $$10^9$$–$$10^{10}\,$$GeV of [[MVV]](#ref-mvv). This is the regime of the induced-gravity model of [[MF]](#ref-mf), who require $$M_P^2\gg M_5^3L$$ (their eq. (4.2)), which gives $$M_5\ll3\times10^8$$–$$2\times10^9\,$$GeV for their range of $$L$$ (their eq. (4.3)).

The ghost condition comes from (6). If the radii of the $$n$$ other circles are dynamical four-dimensional fields, the volume direction has negative kinetic energy whenever

$$
F<\frac{2n}{3(n+1)}, \tag{12}
$$

that is for $$F<\tfrac13$$ when $$n=1$$ and $$F<\tfrac59$$ when $$n=5$$. The right side is at least $$\tfrac13$$ for every $$n\ge1$$, so the region allowed by (3), $$F\lesssim0.1$$, lies inside the ghost region. This case is therefore consistent only if the moduli of the other dimensions are not light four-dimensional degrees of freedom: there are no further dimensions ($$n=0$$, a five-dimensional theory), or their size is fixed at or above the cutoff. The radion itself never becomes a ghost, since $$K=\tfrac32F^2>0$$ for the circle. In warped models, Dillon, George and McDonald [[DGM]](#ref-dgm) showed that brane curvature terms change the radion's coupling to brane matter and that large ones can make the radion ghost-like.

Whether the brane term can be this large depends on the ultraviolet completion. The model of [[MF]](#ref-mf) obtains it from loops of species below the species scale, their eq. (4.1); a one-loop estimate $$N\Lambda^2/16\pi^2$$ with $$N\sim10^2$$ species and cutoff $$\Lambda=10^9$$–$$10^{10}\,$$GeV gives $$M_b^2/M_P^2\sim10^{-19}$$–$$10^{-17}$$, and without the loop factor $$10^{-17}$$–$$10^{-15}$$, so with these inputs $$N\Lambda^2$$ would have to be larger by 15 to 19 orders of magnitude for the brane term to carry 90% of $$M_P^2$$.

### 5.3 Warping

Take the Randall–Sundrum geometry $$e^{-2k\lvert y\rvert}g_{\mu\nu}dx^\mu dx^\nu+dy^2$$ on an interval $$0\le y\le L$$, with anti-de Sitter curvature scale $$k$$, a positive-tension (ultraviolet) brane at $$y=0$$ that carries the Standard Model, and a second (infrared) brane at $$y=L$$. This is the case $$n=0$$ with no brane Einstein–Hilbert term, and the dark matter is the KK gravitons. The warp factor $$\Omega\equiv e^{-kL}$$ is the ratio of mass scales on the infrared and ultraviolet branes, and $$k\to0$$ restores the flat interval. Write $$M_0^2\equiv M_5^3/k$$. Garriga and Tanaka [[GT]](#ref-gt) showed that linearised gravity on the positive-tension brane, with the other brane empty, is Brans–Dicke theory with four-dimensional Planck mass $$M_P^2=M_0^2(1-\Omega^2)$$ (their eq. (26), up to the factor relating $$G$$ and $$M_P$$), and parameter $$\omega=\tfrac32(\Omega^{-2}-1)$$ (their eq. (27)). With $$\Phi=M_0^2(1-\Omega^2)$$ and $$\partial\Phi=-2M_0^2\Omega\,\partial\Omega$$ the Brans–Dicke action $$\tfrac12\Phi\mathcal R-\tfrac{\omega}{2\Phi}(\partial\Phi)^2$$ becomes, in the frame of the ultraviolet brane,

$$
\tfrac12M_0^2(1-\Omega^2)\,\mathcal R-3M_0^2(\partial\Omega)^2 .
$$

As $$k\to0$$, $$1-\Omega^2\to2kL$$ and the second term vanishes, which is the flat circle with $$\omega=0$$. The Weyl rescaling of step 2 of section 2, with $$\Phi$$ in place of $$Fe^S+1-F$$, gives the canonical field $$d\phi=\sqrt6\,d\Omega/(1-\Omega^2)$$ in units $$M_P=1$$. Brane masses are constant in the ultraviolet frame, so $$d\ln m_b=-\tfrac12d\ln\Phi=\Omega\,d\Omega/(1-\Omega^2)$$ and $$\lvert\beta_b\rvert=\Omega/\sqrt6$$. The KK graviton masses in the same frame are $$m_j=kx_j$$, where $$x_j(\Omega)$$ are the positive roots of

$$
J_1(x)\,Y_1(x/\Omega)-Y_1(x)\,J_1(x/\Omega)=0, \tag{13}
$$

the Neumann condition at both branes. In the Einstein frame $$d\ln m_j=d\ln x_j-\tfrac12d\ln\Phi$$, so

$$
\frac{\beta_b}{\beta_{\rm DM}}=\frac{\Omega/(1-\Omega^2)}{d\ln x_j/d\Omega+\Omega/(1-\Omega^2)} .
$$

The roots of (13), found numerically, give masses that approach $$\pi j/L$$ for $$kL\to0$$ and $$k\Omega$$ times the zeros of $$J_1$$ for $$kL\gg1$$. At large $$kL$$ the Planck mass depends on $$\Omega$$ through $$1-\Omega^2$$ only, while the KK masses are proportional to $$\Omega$$, so $$\beta_b/\beta_{\rm DM}\to\Omega^2=e^{-2kL}$$. This agrees with the standard radion couplings: the radion couples to matter on the infrared brane with strength $$1/(\sqrt6\,\Lambda_W)$$, $$\Lambda_W=M_Pe^{-kL}$$ in the normalisation of the Planck mass used there [[CGK]](#ref-cgk) (their eq. (2.4)), which is $$\beta_{\rm DM}=1/(\sqrt6\,\Omega)$$ for gravitons localised there, and $$\Omega^2$$ times that for matter on the ultraviolet brane. For the lightest mode:

| $$kL$$ | 0.05 | 0.5 | 1 | 1.5 | 2 | 3 |
|---|---|---|---|---|---|---|
| $$\beta_b/\beta_{\rm DM}$$ | 0.317 | 0.189 | 0.093 | 0.041 | 0.017 | 0.0024 |
| $$e^{-2kL}$$, the large-$$kL$$ limit | 0.905 | 0.368 | 0.135 | 0.050 | 0.018 | 0.0025 |

For a light field whose mass-relevant component is the radion, (3) at $$c'=0.05$$ requires $$\beta_b/\beta_{\rm DM}<0.05$$, which holds for $$kL>1.39$$; the threshold is $$1.25$$ at $$c'=0.04$$ and $$1.50$$ at $$c'=0.06$$. Using the fifth root of (13) in place of the first lowers each threshold by $$0.03$$. A pure warped radion, with $$\lvert\beta_b\rvert=e^{-kL}/\sqrt6$$, needs $$kL>5.1$$. These numbers are a computation in the linearised two-brane effective theory of [[GT]](#ref-gt), with a massless radion; they are not a theorem, and they do not address whether the dark-matter phenomenology of a micron-sized dimension survives warping with $$kL\approx1.4$$.

### 5.4 Other departures from the hypotheses

Brane couplings that depend on the moduli violate (H2). Loop effects that make couplings depend on the species scale, brane-position moduli, and a dilaton that varies on the brane each add a baryon coupling that is not proportional to $$\delta S$$ and moves the decoupled hyperplane; none is computed here. A non-linear coupling or screening in the Solar System violates (H4). Internal spaces other than a flat torus, fibrations and a graviphoton are outside (H1).

The dark matter can also be the KK tower of a bulk scalar instead of the graviton tower, which is outside (H3). In the variant of Vafa [[Vafa]](#ref-vafa) (appendix A and footnote 6) the dark dimension is stabilised with a heavy radion, the tower is that of the massless bulk field whose zero mode is $$\phi$$, and $$c'=-d\ln m_{\phi,j}/d\phi$$ comes from the dependence on $$\phi$$ of the propagation speed of the tower along the dark dimension (his eq. (23) with $$\theta=0$$). The internal metric does not move with $$\phi$$, so with brane couplings independent of $$\phi$$ as in (H2) the baryon masses in Planck units do not depend on $$\phi$$ and $$\beta_b=0$$, with $$c'\simeq0.05$$ compatible with (3). In that variant $$\phi$$ is not a modulus of $$h_{ab}$$, so Proposition 1 does not apply to it.

## 6. Open points

The statements are classical and at two derivatives. Theorem 1 is proved for a torus with an arbitrary flat metric; for other internal spaces it relies on an argument about shape moduli that has not been completed. For the case of section 5.1, the potential that selects the light direction is not specified. For the brane-term case, it is open whether an ultraviolet completion supplies a brane term carrying about 90% of $$M_P^2$$ while fixing the moduli of the other dimensions at the cutoff. For the warped case, the computation covers only $$n=0$$ with the Standard Model on the ultraviolet brane and a massless radion. Whether the fit of [[BOVW]](#ref-bovw) with the baryon coupling $$\beta_b=c'/3$$ included still prefers $$c'\approx0.05$$ has not been computed. The coupling of other light fields of the scenario, such as the neutrino sector with a bulk right-handed tower, which is mentioned in [[Stewart]](#ref-stewart) (footnote 2), is not treated.

## References

1. <a id="ref-mvv"></a>M. Montero, C. Vafa, I. Valenzuela, "The Dark Dimension and the Swampland," JHEP 2023, 22 (2023), [arXiv:2205.12293](https://arxiv.org/abs/2205.12293).
2. <a id="ref-gmov"></a>E. Gonzalo, M. Montero, G. Obied, C. Vafa, "Dark Dimension Gravitons as Dark Matter," 2022, [arXiv:2209.09249](https://arxiv.org/abs/2209.09249).
3. <a id="ref-bovw"></a>A. Bedroya, G. Obied, C. Vafa, D. H. Wu, "Evolving Dark Sector and the Dark Dimension Scenario," 2025, [arXiv:2507.03090](https://arxiv.org/abs/2507.03090).
4. <a id="ref-vafa"></a>C. Vafa, "The Phenomenology of the Dark Dimension and Non-Supersymmetric Strings," 2026, [arXiv:2610.04909](https://arxiv.org/abs/2610.04909).
5. <a id="ref-mf"></a>R. Mantry, K. Furuuchi, "Relating Dark Dimension Size with Dark Energy via 5D Quintessence Potential and Casimir Energy," 2026, [arXiv:2610.35124](https://arxiv.org/abs/2610.35124).
6. <a id="ref-stewart"></a>A. Stewart, "Cosmological stasis and the coupled dark sector of the Dark Dimension," 2026, [arXiv:2609.25230](https://arxiv.org/abs/2609.25230).
7. <a id="ref-ehrrv"></a>M. Etheredge, B. Heidenreich, T. Rudelius, I. Ruiz, I. Valenzuela, "Taxonomy of Infinite Distance Limits," 2024, [arXiv:2405.20332](https://arxiv.org/abs/2405.20332).
8. <a id="ref-will"></a>C. M. Will, "The Confrontation between General Relativity and Experiment," Living Rev. Relativity 17, 4 (2014), [arXiv:1403.7377](https://arxiv.org/abs/1403.7377).
9. <a id="ref-bit"></a>B. Bertotti, L. Iess, P. Tortora, "A test of general relativity using radio links with the Cassini spacecraft," Nature 425, 374–376 (2003), [doi:10.1038/nature01997](https://doi.org/10.1038/nature01997).
10. <a id="ref-gt"></a>J. Garriga, T. Tanaka, "Gravity in the Randall–Sundrum Brane World," Phys. Rev. Lett. 84, 2778 (2000), [arXiv:hep-th/9911055](https://arxiv.org/abs/hep-th/9911055).
11. <a id="ref-cgk"></a>C. Csáki, M. L. Graesser, G. D. Kribs, "Radion Dynamics and Electroweak Physics," Phys. Rev. D 63, 065002 (2001), [arXiv:hep-th/0008151](https://arxiv.org/abs/hep-th/0008151).
12. <a id="ref-dgm"></a>B. M. Dillon, D. P. George, K. L. McDonald, "Regarding the Radion in Randall–Sundrum Models with Brane Curvature," Phys. Rev. D 94, 064045 (2016), [arXiv:1605.03087](https://arxiv.org/abs/1605.03087).
