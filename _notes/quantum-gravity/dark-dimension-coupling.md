---
title: How a dark-dimension modulus couples to baryons
date: 2026-10-29
category: dark-sector
order: 5
description: In an unwarped dark-dimension compactification with the Standard Model on a brane and no Einstein-Hilbert term on the brane, a field that changes only the size of the dark dimension couples to baryons with one third of its coupling to dark matter, which the Cassini bound excludes at the coupling preferred by cosmological fits. Fields that leave baryons uncoupled preserve the internal volume, and their dark-matter coupling is then capped at an order-one value.
status: Reduction, couplings and the bound proved at tree level for a flat torus with an arbitrary metric; the warped case is a numerical estimate in a linearised two-brane effective theory.
excludes: Unwarped tree-level dark-dimension models with the Standard Model on a point-like brane, frozen brane couplings and no brane Einstein-Hilbert term, as the source of a dark-matter coupling near 0.05 that is compatible with the Cassini bound; in the model where only the dark circle and the size of the other dimensions are light, no direction satisfies both.
---

In the dark-dimension scenario the dark matter is the Kaluza–Klein (KK) tower of a micron-sized extra circle, and a light scalar $$\phi$$ that changes the size of the circle makes the dark-matter mass drift. Fits of this fading dark sector to BAO, CMB and supernova data give $$m_{\rm DM}\propto e^{-c'\phi}$$ with $$c'=0.05\pm0.01$$ in reduced Planck units [[BOVW]](#ref-bovw). We ask how strongly the same field couples to baryons, which live on the Standard Model brane. We work at tree level with an unwarped product compactification, brane masses fixed in units of the higher-dimensional Planck mass, and no Einstein–Hilbert term on the brane. In this setting a field whose only mass-relevant component is the radius of the dark circle couples to baryons with $$\beta_b=c'/3$$ (Proposition 2). The reason is that brane masses are fixed in higher-dimensional Planck units while the four-dimensional Planck mass grows with the internal volume, so that in four-dimensional Planck units brane masses scale as the inverse square root of the volume (for a single circle, brane masses scale as $$R^{-1/2}$$ and KK masses as $$R^{-3/2}$$). At $$c'=0.05$$ the resulting post-Newtonian parameter $$\gamma-1$$ is about 44 times the Cassini bound (section 3). Baryons decouple exactly when the field leaves the total internal volume fixed, and on those directions $$\lvert c'\rvert\le\sqrt{n/(n+1)}$$ for $$n$$ further internal dimensions, which is $$0.71$$ to $$0.93$$ for $$n=1,\dots,6$$ (Theorem 1). In the model where only the dark circle and the common size of the other dimensions are light, no direction has both $$c'=0.05$$ and a baryon coupling within the Cassini bound (Corollary 1).

Three conditions evade the exclusion, each stated in section 5: an Einstein–Hilbert term on the brane that supplies at least about 87 to 91 per cent of $$M_P^2$$, which is consistent only if the moduli of the other dimensions are not dynamical four-dimensional fields; a warped dark dimension with $$kL\gtrsim1.4$$, where $$k$$ is the curvature scale and $$L$$ the length of the extra dimension; and brane couplings that depend on the moduli.

Montero, Vafa and Valenzuela proposed the dark dimension [[MVV]](#ref-mvv), and Gonzalo, Montero, Obied and Vafa proposed its KK gravitons as dark matter [[GMOV]](#ref-gmov). Bedroya, Obied, Vafa and Wu [[BOVW]](#ref-bovw) fitted the fading dark sector and noted (their section III) that the field may change the circle either at fixed total internal volume, which keeps Newton's constant fixed in Standard Model units, or with the volume varying. Vafa [[Vafa]](#ref-vafa) (appendix A) observes that the fitted $$c'\simeq0.05$$ is far below the value $$\sqrt{3/2}$$ of the pure radion and writes $$\sin\theta=c'/\sqrt{3/2}\simeq0.04$$ for the projection of the light field on the radion. Mantry and Furuuchi [[MF]](#ref-mf) observe that a fixed total internal volume makes Newton's constant independent of the shape modulus (their section 5; their canonical field $$\chi=\sqrt2M_P\ln\sigma$$ corresponds to $$c'=1/\sqrt2$$, the case $$n=1$$ below), and they build a five-dimensional model in which an induced brane-localised Einstein–Hilbert term dominates $$M_P^2$$ (their section 4). The bound of Theorem 1 can be read off from the tower and species vectors of Etheredge et al. [[EHRRV]](#ref-ehrrv) (their eqs. (1.3) and (1.6)), as we note after the theorem. Two analyses of this model, by Stewart [[Stewart]](#ref-stewart) and in [[BOVW]](#ref-bovw) (section IV), treat baryons as uncoupled from the field; the comparison is made after Proposition 2. What this page adds is the ratio $$\beta_b/c'$$ with its Cassini consequence in the fading-dark-sector setting, the bound over the full metric moduli space of a torus, the closed form for the minimal model, and quantitative conditions for the escapes, including the ghost condition for the brane-term escape. The related question of which dark-matter masses a given field can fix is in the [page on dark-sector parameters](/quantum-gravity/dark-sector-moduli/); the place of this result among the others is in the [landscape](/quantum-gravity/landscape/).

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

with $$\phi$$ canonically normalised in units of the reduced Planck mass, $$\mathcal L=\tfrac12R-\tfrac12(\partial\phi)^2$$. Exchange of $$\phi$$, when it is light on the scale of the separation, adds a fraction $$2\beta_X\beta_Y$$ to the Newtonian attraction between non-relativistic bodies of species $$X$$ and $$Y$$. The dark-matter coupling of [[BOVW]](#ref-bovw) is $$c'=\beta_{\rm DM}$$, and their dark fifth force, $$2c'^2$$ times gravity (their eq. (18)), uses the same normalisation.

By (1), if $$\phi$$ moves only $$R$$, then $$\beta_b/\beta_{\rm DM}=\tfrac12/\tfrac32=\tfrac13$$, whatever the normalisation of $$\phi$$ and whatever the potential. The Einstein-frame action is $$\tfrac12R-\tfrac34(\partial\ln R)^2$$, so $$\phi=\sqrt{3/2}\,\ln R$$ and the pure radion has $$\beta_b=\tfrac12\sqrt{2/3}=1/\sqrt6$$ and $$\beta_{\rm DM}=\sqrt{3/2}$$. The value $$2\beta_b^2=\tfrac13$$ is the Brans–Dicke value for $$\omega=0$$, since a linear coupling $$\beta_b$$ corresponds to $$2\omega+3=1/(2\beta_b^2)$$ (see below).

*The Solar-System bound.* A massless scalar with a constant linear coupling $$\beta_b$$ to the Sun is a Brans–Dicke field with $$\alpha_0^2=1/(2\omega+3)$$ in the notation of Will's review [[Will]](#ref-will), where the physical metric is $$A^2(\varphi)\tilde g_{\mu\nu}$$ and the Einstein-frame action is $$(16\pi G)^{-1}\int[\tilde R-2(\tilde\partial\varphi)^2]$$ (his eq. (33)) and $$\alpha_0=d\ln A/d\varphi$$. Matching $$\tfrac12(\partial\phi)^2$$ to $$M_P^2(\partial\varphi)^2$$ gives $$\phi=\sqrt2M_P\varphi$$ and $$\alpha_0^2=2\beta_b^2$$. The post-Newtonian parameter is $$\gamma-1=-1/(2+\omega)=-2\alpha_0^2/(1+\alpha_0^2)$$, that is

$$
\gamma-1=-\frac{4\beta_b^2}{1+2\beta_b^2}. \tag{2}
$$

The Cassini radio-link experiment measured $$\gamma-1=(2.1\pm2.3)\times10^{-5}$$ [[BIT]](#ref-bit), as quoted in [[Will]](#ref-will), so at two standard deviations $$\gamma-1>-2.5\times10^{-5}$$ and

$$
\lvert\beta_b\rvert<2.5\times10^{-3}. \tag{3}
$$

This applies when the range of $$\phi$$ exceeds an astronomical unit, as it does for a field that rolls on cosmological time scales, and when the coupling is linear. By (1) every brane species has the same coupling, so tests of the equivalence principle among ordinary materials do not constrain it and (3) is the relevant bound. For the pure radion $$\beta_b=0.408$$ exceeds the bound (3) by a factor 160.

## 2. Several internal dimensions

We take $$D=4+k$$ dimensions with $$k=n+1$$: the four-dimensional spacetime, the dark circle and $$n$$ further circles, so that the internal space is a torus $$T^k$$. The torus makes the computation explicit; the extension to other internal spaces is discussed after Theorem 1. The hypotheses are as follows.

(H1) Tree-level two-derivative gravity in $$D$$ dimensions, with action $$\tfrac12M_D^{D-2}\int d^Dx\sqrt{-g_D}R_D$$, and the product geometry
$$ds^2=g_{\mu\nu}(x)\,dx^\mu dx^\nu+h_{ab}(x)\,dy^a dy^b$$, $$y^a\sim y^a+2\pi$$, where $$h_{ab}$$ is any positive-definite matrix of moduli, constant on the torus. Potentials for the moduli are allowed; they do not enter the kinetic terms or the couplings computed here. Graviphotons and fluxes are absent.

(H2) The Standard Model lives on a point-like 3-brane and couples only to the induced metric $$g_{\mu\nu}$$. Every brane parameter (position, tension, couplings, any dilaton) is fixed in units of $$M_D$$, so every brane mass, including the nucleon mass through $$\Lambda_{\rm QCD}$$, is constant in the brane frame. The brane may carry an Einstein–Hilbert term $$\tfrac12M_b^2\sqrt{-g}\,\mathcal R$$; the case $$M_b=0$$ is the setting of the introduction.

(H3) The dark matter is the KK tower of the first circle, with momentum $$p=(j,0,\dots,0)$$, so that $$m_j^2=j^2h^{00}$$ in units of $$M_D$$ in the brane frame. If the brane carries an Einstein–Hilbert term, (H3) is understood for the modes that vanish at the brane, whose masses are unaffected by it.

(H4) A single light canonically normalised scalar $$\phi$$ moves the moduli $$h_{ab}$$ along a direction at the present point, and may also move fields on which neither $$m_b$$ nor $$m_{\rm DM}$$ depends. Its couplings are linear and it is massless on solar-system scales.

Let $$h_0$$ be the present value of $$h$$, $$e^S=\sqrt{\det h/\det h_0}$$ the relative volume of the torus, and $$V_0$$ the present volume. The four-dimensional Planck mass is $$M_P^2=M_D^{D-2}V_0+M_b^2$$, and we define the bulk fraction

$$
F\equiv\frac{M_D^{D-2}V_0}{M_P^2}\in(0,1],
$$

so that the brane term supplies the fraction $$1-F$$ of $$M_P^2$$ and $$F=1$$ means no brane term. A tangent vector to the moduli space at the present point is a symmetric matrix $$\delta h$$; we describe it by $$Q=h^{-1/2}\delta h\,h^{-1/2}$$, which is a symmetric matrix with $$\operatorname{tr}Q=2\,\delta S$$. In the diagonal case $$h={\rm diag}(e^{2\sigma_a})$$ we have $$Q={\rm diag}(2\delta\sigma_a)$$.

The idea of the computation is the same as in section 1. Dimensional reduction produces a scalar–tensor theory in the brane frame; a Weyl rescaling gives the Einstein frame, where the kinetic metric of the moduli can be read off and the masses (H2), (H3) acquire a factor from the rescaling. The steps are routine; the only structure is in the metric (4).

*Step 1: reduction.* A direct computation of the Christoffel symbols gives the pointwise identity

$$
R_D=\mathcal R-2\,\Box S-(\partial S)^2-\tfrac14\operatorname{tr}\!\big[(h^{-1}\partial h)^2\big], \tag{4}
$$

where $$\mathcal R$$, $$\Box$$ and the contractions refer to $$g_{\mu\nu}$$. In the diagonal case it reads $$\mathcal R-2\Box S-(\partial S)^2-\sum_a(\partial\sigma_a)^2$$, and for one circle it is the familiar $$\mathcal R-2e^{-\sigma}\Box e^{\sigma}$$. We verified (4) symbolically for a conformally flat two-dimensional base with a general $$2\times2$$ block $$h_{ab}(x)$$ depending on both coordinates; the coefficients do not depend on the dimension of the base. Since $$\sqrt{-g_D}=e^S\sqrt{\det h_0}\,\sqrt{-g}$$ and $$\int e^S\Box S=-\int e^S(\partial S)^2$$, integrating over the torus gives the bulk Lagrangian $$\tfrac12M_D^{D-2}V_0\,e^S\sqrt{-g}\,[\mathcal R+(\partial S)^2-\tfrac14\operatorname{tr}(h^{-1}\partial h)^2]$$. Adding the brane term and setting $$M_P=1$$, the Lagrangian in the brane frame is

$$
\sqrt{-g}\Big[\tfrac12\Phi\,\mathcal R+\tfrac12Fe^S\big((\partial S)^2-\tfrac14\operatorname{tr}(h^{-1}\partial h)^2\big)\Big],\qquad\Phi\equiv Fe^S+1-F. \tag{5}
$$

*Step 2: Einstein frame.* In four dimensions the rescaling $$g_{\mu\nu}=\Phi^{-1}g^E_{\mu\nu}$$ gives, up to a total derivative, $$\sqrt{-g}\,\Phi\,\mathcal R=\sqrt{-g_E}\,[R_E-\tfrac32(\partial\ln\Phi)_E^2]$$ (we checked the underlying formula $$\mathcal R=\Phi[R_E+3\Box_E\ln\Phi-\tfrac32(\partial\ln\Phi)^2_E]$$ symbolically for a diagonal metric depending on two coordinates), and $$\sqrt{-g}\,(\partial f)^2=\Phi^{-1}\sqrt{-g_E}\,(\partial f)^2_E$$ for any scalar $$f$$. With $$\partial\ln\Phi=w\,\partial S$$ and $$w\equiv Fe^S/\Phi$$, the Lagrangian (5) becomes $$\sqrt{-g_E}[\tfrac12R_E-\tfrac12G]$$ with kinetic form $$G=\tfrac14w\operatorname{tr}(h^{-1}\partial h)^2-\tfrac14w(\operatorname{tr}h^{-1}\partial h)^2+\tfrac38w^2(\operatorname{tr}h^{-1}\partial h)^2$$.

*Step 3: masses.* A mass $$m$$ that is constant in the brane frame is $$m\Phi^{-1/2}$$ in the Einstein frame. The brane masses are constant in the brane frame by (H2); the KK masses are $$m_j=\lvert j\rvert\sqrt{h^{00}}$$ in the brane frame by (H3). Along the direction $$Q$$ at the present point ($$w=F$$), with $$\hat e\equiv h^{-1/2}e_0/\lvert h^{-1/2}e_0\rvert$$ and $$e_0$$ the unit vector of the first circle, one has $$\delta\ln h^{00}=-\hat e^{\sf T}Q\hat e$$ and $$\delta\ln\Phi^{-1/2}=-\tfrac14F\operatorname{tr}Q$$.

<div class="proposition" markdown="1">
**Proposition 1** (moduli metric and couplings). Assume (H1)–(H4). At the present point the Einstein-frame kinetic form on tangent vectors $$Q$$ is

$$
G(Q,Q)=\tfrac14F\big[\operatorname{tr}Q^2-(\operatorname{tr}Q)^2\big]+\tfrac38F^2(\operatorname{tr}Q)^2, \tag{6}
$$

and a light field moving along $$Q$$ with $$G(Q,Q)=1$$ has the couplings

$$
\beta_b=\tfrac14F\operatorname{tr}Q,\qquad c'=\tfrac12\hat e^{\sf T}Q\hat e+\tfrac14F\operatorname{tr}Q. \tag{7}
$$

In the diagonal case, with $$v=(1,\dots,1)$$, $$G=F(\mathbb 1-vv^{\sf T})+\tfrac32F^2vv^{\sf T}$$ on the vector $$u_a=\delta\sigma_a/d\phi$$. It has eigenvalue $$F$$ with multiplicity $$n$$ on $$\sum_au_a=0$$ and eigenvalue

$$
F\big[\tfrac32(n+1)F-n\big] \tag{8}
$$

on the volume direction $$u_a=1$$. For $$F=1$$ this is the standard metric $$\mathbb 1+\tfrac12vv^{\sf T}$$ of the radii of a torus.
</div>

<div class="proof" markdown="1">
*Proof.* (6) is the form obtained in step 2 at $$w=F$$, written in terms of $$Q$$. (7) follows from step 3 and $$\beta_X=-\,d\ln m_X/d\phi$$ with $$d\phi=\sqrt{G(Q,Q)}$$. In the diagonal case $$\operatorname{tr}Q^2=4\sum u_a^2$$ and $$\operatorname{tr}Q=2\sum u_a$$ give $$G=F[\sum u_a^2-(\sum u_a)^2]+\tfrac32F^2(\sum u_a)^2$$. The matrix $$vv^{\sf T}$$ annihilates vectors with $$\sum u_a=0$$ and has eigenvalue $$n+1$$ on $$v$$, which gives (8). $$\square$$
</div>

Two checks. For $$k=1$$ the bracket in (6) vanishes and $$G=\tfrac38F^2(\operatorname{tr}Q)^2=\tfrac32F^2(\delta\ln R)^2$$, which for $$F=1$$ is the coefficient found in section 1. For $$F=1$$ and the isotropic volume direction, (7) and (8) give $$2\beta_b^2=k/(k+2)$$, the known strength of the force of the volume modulus of $$k$$ equal extra dimensions; this is the largest baryon coupling of any unit direction, attained when $$u\propto v$$ (see the remark after Theorem 1).

## 3. The radion direction and the Cassini bound

Suppose first that $$\phi$$ changes the radius of the dark circle and nothing else that enters a mass. We call $$Q$$ a radion direction if $$Q=a\,\hat e\hat e^{\sf T}+Q_\perp$$ with $$\operatorname{tr}Q_\perp=0$$ and $$\hat e^{\sf T}Q_\perp\hat e=0$$. The components $$Q_\perp$$ move the other moduli (shapes of the other circles, off-diagonal components, anything else that leaves the volume and $$h^{00}$$ unchanged); they affect neither coupling in (7).

<div class="proposition" markdown="1">
**Proposition 2** (radion coupling). Assume (H1)–(H4) and a radion direction. Then

$$
\frac{\beta_b}{c'}=\frac{F}{2+F},
$$

which is $$\tfrac13$$ without a brane Einstein–Hilbert term. For the pure radion ($$Q_\perp=0$$), $$\beta_b=1/\sqrt6$$ for every $$F$$ and $$c'=(2+F)/(\sqrt6\,F)$$.
</div>

<div class="proof" markdown="1">
*Proof.* For $$Q=a\hat e\hat e^{\sf T}+Q_\perp$$ one has $$\operatorname{tr}Q=a$$ and $$\hat e^{\sf T}Q\hat e=a$$, so (7) gives $$\beta_b=\tfrac14Fa/\sqrt G$$ and $$c'=(\tfrac12+\tfrac14F)a/\sqrt G$$, whose ratio is $$F/(2+F)$$ whatever the normalisation. For the pure radion, (6) gives $$G=\tfrac14F(a^2-a^2)+\tfrac38F^2a^2=\tfrac38F^2a^2$$, so $$a=\sqrt{8/3}/F$$ and $$\beta_b=\tfrac14F a=1/\sqrt6$$, $$c'=\tfrac14(2+F)a=(2+F)/(\sqrt6F)$$. $$\square$$
</div>

*The Cassini exclusion.* Since the observed $$c'\approx0.05$$ is far below $$\sqrt{3/2}$$, the light field must be mostly something other than the radion, as Vafa assumes [[Vafa]](#ref-vafa), and Proposition 2 says that the ratio $$\beta_b/c'$$ does not depend on how small the radion component is. Without a brane term, $$c'=0.05$$ gives $$\beta_b=0.0167$$ and, by (2), $$\gamma-1=-1.11\times10^{-3}$$, which is 44 times the two-standard-deviation bound $$2.5\times10^{-5}$$. The fitted range $$c'=0.04$$–$$0.06$$ gives factors $$28$$ to $$64$$. Equivalently, by (3), every $$c'>3\times2.5\times10^{-3}=7.5\times10^{-3}$$ is excluded for such a field.

<div class="remark" markdown="1">
**Remark** (comparison with published treatments). Stewart [[Stewart]](#ref-stewart), section 2.1, takes the field to be the radion of the dark dimension and states that the Standard Model "does not couple to the radion at leading order, so the baryons are uncoupled dust", which enters his eq. (2.2c) as $$\rho_b\propto a^{-3}$$; his footnote 2 adds that the masses of Standard Model fields on the brane would typically not depend on $$R$$. In units of $$M_D$$ this is hypothesis (H2). As a statement about masses in four-dimensional Planck units, under (H1)–(H3) it holds on directions that preserve the total internal volume (section 4), and for a field whose only mass-relevant component is the radion Proposition 2 gives $$\beta_b=c_2/3$$, where $$c_2$$, the logarithmic slope of the tower mass in his eq. (2.10), equals $$c'$$. Bedroya et al. [[BOVW]](#ref-bovw), section IV, state that visible matter, being localised on the Standard Model brane, "does not couple directly to $$\phi$$". By the same argument this holds on the fixed-volume branch that their section III allows, and not on the branch with varying volume.
</div>

## 4. Directions that leave baryons uncoupled

By (7), $$\beta_b=0$$ if and only if $$\operatorname{tr}Q=0$$, that is, if the field preserves the total internal volume in units of $$M_D$$; this holds for every $$F>0$$. The question is how large $$c'$$ can be on such a direction. The point is that on the slice $$\operatorname{tr}Q=0$$ the metric (6) is the flat trace form $$\tfrac14F\operatorname{tr}Q^2$$ and $$c'$$ is linear, so the maximum is a Cauchy–Schwarz inequality in the space of symmetric matrices.

<div class="theorem" markdown="1">
**Theorem 1** (decoupled directions). Assume (H1)–(H4) with internal space $$T^{n+1}$$, $$n\ge1$$, and any flat metric $$h$$. A unit direction has $$\beta_b=0$$ if and only if $$\operatorname{tr}Q=0$$. Every such direction has

$$
\lvert c'\rvert\le\sqrt{\frac{n}{(n+1)F}}, \tag{9}
$$

with equality exactly for $$Q\propto\hat e\hat e^{\sf T}-\mathbb 1/(n+1)$$. Without a brane term the bound is $$\sqrt{n/(n+1)}$$.
</div>

<div class="proof" markdown="1">
*Proof.* The first statement is (7). On the slice $$\operatorname{tr}Q=0$$ the normalisation $$G=1$$ reads $$\operatorname{tr}Q^2=4/F$$, and by (7) $$c'=\tfrac12\hat e^{\sf T}Q\hat e=\tfrac12\operatorname{tr}(Q\,\hat e\hat e^{\sf T})=\tfrac12\operatorname{tr}(QP)$$ with $$P\equiv\hat e\hat e^{\sf T}-\mathbb 1/(n+1)$$, since $$\operatorname{tr}Q=0$$. The matrix $$P$$ is symmetric and traceless with $$\operatorname{tr}P^2=1-\tfrac2{n+1}+\tfrac1{n+1}=\tfrac n{n+1}$$. Cauchy–Schwarz for the trace inner product gives $$\lvert\operatorname{tr}(QP)\rvert\le(\operatorname{tr}Q^2)^{1/2}(\operatorname{tr}P^2)^{1/2}=2\sqrt{n/((n+1)F)}$$, with equality if and only if $$Q\propto P$$, and $$P$$ is itself traceless. $$\square$$
</div>

In the diagonal case the extremal direction increases the dark circle by an amount $$\delta$$ in the logarithm and decreases each of the $$n$$ other circles by $$\delta/n$$. We checked (9) independently by maximising $$c'$$ numerically over all symmetric $$Q$$ with $$\operatorname{tr}Q=0$$ for random positive-definite metrics $$h$$, for $$n=1,2,5,6$$ and $$F=1,\,0.3$$; the maximum agrees with (9) to six digits. For $$n=1$$ and $$F=1$$ the extremal direction is the shape modulus of [[MF]](#ref-mf), section 5. For $$n=1,\dots,6$$ without a brane term the bound is:

| $$n$$ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| $$\sqrt{n/(n+1)}$$ | 0.707 | 0.816 | 0.866 | 0.894 | 0.913 | 0.926 |

The range $$0.71$$–$$0.93$$ requires $$1\le n\le6$$ (for $$n=22$$ the value is $$0.978$$). Writing $$c'=\sqrt{n/((n+1)F)}\cos\alpha$$ defines $$\alpha$$ as the angle, in the metric $$G$$, between the light direction and the extremal direction. With more light moduli than the dark circle and the volume, $$c'=0.05$$ requires $$\cos\alpha=0.055$$ for $$n=5$$ (from $$0.044$$ to $$0.066$$ for $$c'=0.04$$–$$0.06$$), an angle $$\alpha=86.9^\circ$$ ($$87.5^\circ$$ to $$86.2^\circ$$) from the extremal direction. The geometry does not fix $$\alpha$$, and the potential selects the light direction.

*Relation to tower vectors.* Let $$\zeta$$ be the vector, in the metric $$G$$ at $$F=1$$, dual to $$-d\ln m_{\rm DM}$$ and $$Z$$ the one dual to $$-d\ln M_D$$, both in four-dimensional Planck units. Then $$\beta_b=\langle Z,u\rangle$$ and $$c'=\langle\zeta,u\rangle$$ for a unit direction $$u$$, and $$\beta_b=0$$ means $$u\perp Z$$. A computation with (6) gives $$\lvert\zeta\rvert^2=\tfrac32$$, $$\zeta\cdot Z=\tfrac12$$ and $$\lvert Z\rvert^2=(n+1)/(2(n+3))$$, so the largest $$c'$$ orthogonal to $$Z$$ is $$(\lvert\zeta\rvert^2-(\zeta\cdot Z)^2/\lvert Z\rvert^2)^{1/2}=\sqrt{n/(n+1)}$$, as in (9). These are the values $$\lvert\zeta_{{\rm KK}}\rvert^2=(n+d-2)/(n(d-2))=\tfrac32$$ (for the decompactification of one dimension in $$d=4$$) and $$\zeta\cdot Z=1/(d-2)=\tfrac12$$ that the taxonomy of Etheredge et al. gives for towers in asymptotic limits [[EHRRV]](#ref-ehrrv) (their eqs. (1.6) and (1.3)); here they come out of the explicit metric.

*The minimal model.* We call minimal the model in which only the dark circle and the common size of the other $$n$$ circles are light, with $$F=1$$. It is the simplest realisation of the fixed-volume branch of [[BOVW]](#ref-bovw) (their Fig. 1), in which the dark dimension grows while the other dimensions contract. The light directions form a plane, and both couplings are linear functionals on it.

<div class="corollary" markdown="1">
**Corollary 1** (the minimal model). Assume (H1)–(H4) with $$F=1$$, internal space $$T^{n+1}$$ ($$n\ge1$$) and light moduli spanning only the radius of the dark circle and the common radius of the other circles. Then the baryon-decoupled direction is unique up to sign, and it has $$\lvert c'\rvert=\sqrt{n/(n+1)}$$. For every unit direction with a given $$c'$$,

$$
\lvert\beta_b\rvert\;\ge\;\sqrt{\frac{n}{3(n+3)}}\,\sqrt{1-\tfrac23c'^2}-\frac{\lvert c'\rvert}{3}. \tag{10}
$$

For $$c'=0.05$$ the right side is $$0.272,\ 0.348,\ 0.439,\ 0.454$$ for $$n=1,2,5,6$$, which is 109 to 182 times the bound (3).
</div>

<div class="proof" markdown="1">
*Proof.* In the plane of light directions, with the metric induced by $$G$$, the functionals $$\beta_b=\langle Z,u\rangle$$ and $$c'=\langle\zeta,u\rangle$$ are given by two vectors $$Z,\zeta$$ of the plane (the dual vectors lie in the plane because $$G^{-1}=\mathbb 1-vv^{\sf T}/(n+3)$$ maps $$\operatorname{span}(e_0,v)$$ to itself) with $$\lvert\zeta\rvert^2=\tfrac32$$, $$\zeta\cdot Z=\tfrac12$$, $$\lvert Z\rvert^2=(n+1)/(2(n+3))$$ (computed above). A unit $$u$$ with $$\langle\zeta,u\rangle=c'$$ is $$u=(c'/\lvert\zeta\rvert^2)\zeta\pm\sqrt{1-c'^2/\lvert\zeta\rvert^2}\,\hat\zeta_\perp$$, so $$\beta_b=\tfrac13c'\pm\lvert Z_\perp\rvert\sqrt{1-\tfrac23c'^2}$$ with $$\lvert Z_\perp\rvert^2=\lvert Z\rvert^2-(\zeta\cdot Z)^2/\lvert\zeta\rvert^2=n/(3(n+3))$$. The smaller modulus of the two values is (10) when it is non-negative, which it is for $$\lvert c'\rvert\le\sqrt{n/(n+1)}$$. Setting $$\beta_b=0$$ gives $$c'^2=\lvert\zeta\rvert^2\big(1-\tfrac{(\zeta\cdot Z)^2}{\lvert\zeta\rvert^2\lvert Z\rvert^2}\big)=n/(n+1)$$ at the two points $$\pm u$$. $$\square$$
</div>

We checked (10) numerically by scanning the unit circle of directions for $$n=1,2,5,6$$. The fitted value $$c'=0.05\pm0.01$$ [[BOVW]](#ref-bovw) is smaller than the decoupled value by 66 to 88 times its quoted uncertainty (counting only the data error, which is legitimate because the geometric number has no error under the hypotheses). The dark fifth force $$2c'^2$$ times gravity would be between $$1.0$$ and $$1.7$$, against the bound from tidal disruption of Milky Way satellites, $$c'\lesssim0.2$$ [[BOVW]](#ref-bovw), and the stronger bound $$\lvert c_2\rvert\lesssim0.071$$ of [[Stewart]](#ref-stewart), section 6.1. In type IIB string theory with the Standard Model on D3-branes $$n=5$$ and the decoupled value is $$0.913$$; for $$n=6$$ (eleven dimensions) M-theory has no point-like 3-brane, and a wrapped M5-brane has couplings that depend on the size of the wrapped cycle, which violates (H2).

*Other internal spaces.* For a product $$S^1\times X_n$$ with $$X_n$$ Ricci-flat but not a torus we expect the same structure for the overall sizes, by the following argument, which is not a proof. The shape moduli of $$X_n$$ are traceless deformations of its metric, so they change neither the volume nor the circle, are $$G$$-orthogonal to the circle and volume directions, and enter neither mass. A light field with a shape component has a smaller $$\lvert\cos\alpha\rvert$$, and (9) would still hold. Fibred geometries and a graviphoton are not treated.

## 5. What the theorem does not exclude

Theorem 1 and Corollary 1 are statements about (H1)–(H4). A baryon coupling compatible with (3) and $$c'\approx0.05$$ is possible when one of them is relaxed. The cases below are stated with their conditions; the page does not choose among them.

### 5.1 An Einstein–Hilbert term on the brane

When $$F<1$$, Proposition 2 gives $$\beta_b=c'F/(2+F)$$ for a radion direction. The pure radion keeps its coupling $$1/\sqrt6$$ per canonical unit, so a field that is purely the radion remains excluded; at fixed observed $$c'$$, however, a smaller $$F$$ lowers $$\beta_b$$. The condition (3) becomes $$F<2r/(1-r)$$ with $$r=2.5\times10^{-3}/c'$$:

| $$c'$$ | 0.04 | 0.05 | 0.06 |
|---|---|---|---|
| largest bulk fraction $$F$$ | 0.133 | 0.105 | 0.087 |
| smallest brane share $$1-F$$ of $$M_P^2$$ | 86.7% | 89.5% | 91.3% |

The brane term must therefore supply most of $$M_P^2$$ (it is the bulk fraction $$F$$ that must be small). The bound (9) becomes $$\sqrt{n/((n+1)F)}$$, which exceeds $$2$$ for $$F<0.1$$ and then no longer constrains the decoupled directions. For $$n=0$$ and a circle of circumference $$L=1\,\mu$$m, $$F=M_5^3L/M_P^2$$ equals $$0.1$$ at $$M_5\approx5\times10^8\,$$GeV (with the reduced Planck mass $$2.4\times10^{18}\,$$GeV), about half the value $$1.05\times10^9\,$$GeV at which the bulk alone supplies $$M_P^2$$ and which is within the range $$10^9$$–$$10^{10}\,$$GeV of [[MVV]](#ref-mvv). This is the regime of the induced-gravity model of [[MF]](#ref-mf), who require $$M_P^2\gg M_5^3L$$ (their eq. (4.2), giving $$M_5\ll3\times10^8$$–$$2\times10^9\,$$GeV for their range of $$L$$).

The ghost condition comes from (8). If the radii of the $$n$$ other circles are dynamical four-dimensional fields, the volume direction has negative kinetic energy whenever

$$
F<\frac{2n}{3(n+1)}, \tag{11}
$$

that is for $$F<\tfrac13$$ when $$n=1$$ and $$F<\tfrac59$$ when $$n=5$$ (the eigenvalues in (8) were checked numerically for $$n=1,2,5,6$$ and $$F$$ from $$1$$ to $$0.1$$). The right side is at least $$\tfrac13$$ for every $$n\ge1$$, so the region allowed by (3), $$F\lesssim0.1$$, lies inside the ghost region. This escape is therefore consistent only if the moduli of the other dimensions are not light four-dimensional degrees of freedom: there are no further dimensions ($$n=0$$, a five-dimensional theory), or their size is fixed at or above the cutoff. The radion itself never becomes a ghost, since $$G=\tfrac32F^2>0$$ for the circle. In warped models, Dillon, George and McDonald [[DGM]](#ref-dgm) showed that brane curvature terms change the radion's coupling to brane matter and that large ones can make the radion ghost-like.

Whether the brane term can be this large depends on the ultraviolet completion. The model of [[MF]](#ref-mf) obtains it from loops of species below the species scale, their eq. (4.1); a one-loop estimate $$N\Lambda^2/16\pi^2$$ with $$N\sim10^2$$ species and cutoff $$10^9$$–$$10^{10}\,$$GeV gives $$M_b^2/M_P^2\sim10^{-19}$$–$$10^{-17}$$, and without the loop factor $$10^{-17}$$–$$10^{-15}$$, so with these inputs $$N\Lambda^2$$ would have to be larger by 15 to 19 orders of magnitude for the brane term to carry 90% of $$M_P^2$$.

### 5.2 Warping

Now let the extra dimension be an interval of length $$L$$ with anti-de Sitter curvature $$k$$ (Randall–Sundrum geometry with $$n=0$$), the Standard Model on the positive-tension (ultraviolet) brane, no brane Einstein–Hilbert term, and the dark matter the KK gravitons. Write $$\Omega\equiv e^{-kL}$$ and $$M_0^2\equiv M_5^3/k$$. Garriga and Tanaka [[GT]](#ref-gt) showed that linearised gravity on the positive-tension brane, with the other brane empty, is Brans–Dicke theory with four-dimensional Planck mass $$M_P^2=M_0^2(1-\Omega^2)$$ (their eq. (26), up to the factor relating $$G$$ and $$M_P$$), and parameter $$\omega=\tfrac32(\Omega^{-2}-1)$$ (their eq. (27)). With $$\Phi=M_0^2(1-\Omega^2)$$ and $$\partial\Phi=-2M_0^2\Omega\,\partial\Omega$$ the Brans–Dicke action $$\tfrac12\Phi\mathcal R-\tfrac{\omega}{2\Phi}(\partial\Phi)^2$$ becomes, in the frame of the ultraviolet brane,

$$
\tfrac12M_0^2(1-\Omega^2)\,\mathcal R-3M_0^2(\partial\Omega)^2 .
$$

As $$k\to0$$, $$1-\Omega^2\to2kL$$ and the second term vanishes, which is the flat circle with $$\omega=0$$. The Weyl rescaling of step 2, with $$\Phi$$ in place of $$Fe^S+1-F$$, gives the canonical field $$d\phi=\sqrt6\,d\Omega/(1-\Omega^2)$$ in units $$M_P=1$$. Brane masses are constant in the ultraviolet frame, so $$d\ln m_b=-\tfrac12d\ln\Phi=\Omega\,d\Omega/(1-\Omega^2)$$ and $$\lvert\beta_b\rvert=\Omega/\sqrt6$$. The KK graviton masses in the same frame are $$m_j=kx_j$$, where $$x_j$$ are the positive roots of

$$
J_1(x)\,Y_1(x/\Omega)-Y_1(x)\,J_1(x/\Omega)=0, \tag{12}
$$

the Neumann condition at both branes; we solved it numerically (and checked that for $$kL\to0$$ the masses approach $$\pi j/L$$ and for $$kL\gg1$$ the zeros of $$J_1$$ times $$\Omega$$). At large $$kL$$ the Planck mass depends on $$\Omega$$ through $$1-\Omega^2$$ only, while the KK masses are proportional to $$\Omega$$, so $$\beta_b/\beta_{\rm DM}\to\Omega^2=e^{-2kL}$$. This is also what the standard radion couplings give: the radion couples to matter on the infrared brane with strength $$1/(\sqrt6\,\Lambda_W)$$, $$\Lambda_W=M_Pe^{-kL}$$ in the normalisation of the Planck mass used there [[CGK]](#ref-cgk) (their eq. (2.4)), which is $$\beta_{\rm DM}=1/(\sqrt6\,\Omega)$$ for gravitons localised there, and $$\Omega^2$$ times that for matter on the ultraviolet brane. Numerically for the lightest mode:

| $$kL$$ | 0.05 | 0.5 | 1 | 1.5 | 2 | 3 |
|---|---|---|---|---|---|---|
| $$\beta_b/\beta_{\rm DM}$$ | 0.317 | 0.189 | 0.093 | 0.041 | 0.017 | 0.0024 |
| $$e^{-2kL}$$ | 0.905 | 0.368 | 0.135 | 0.050 | 0.018 | 0.0025 |

For a light field whose mass-relevant component is the radion, (3) at $$c'=0.05$$ requires $$\beta_b/\beta_{\rm DM}<0.05$$, which holds for $$kL>1.39$$; the threshold is $$1.25$$ at $$c'=0.04$$ and $$1.50$$ at $$c'=0.06$$, and for the fifth KK mode it is lower by $$0.03$$. A pure warped radion, with $$\lvert\beta_b\rvert=e^{-kL}/\sqrt6$$, needs $$kL>5.1$$. These numbers are a computation in the linearised two-brane effective theory of [[GT]](#ref-gt), with a massless radion; they are not a theorem, and they do not address whether the dark-matter phenomenology of a micron-sized dimension survives warping with $$kL\approx1.4$$.

### 5.3 Other departures from the hypotheses

Brane couplings that depend on the moduli violate (H2). Loop effects that make couplings depend on the species scale, brane-position moduli, and a dilaton that varies on the brane each add a baryon coupling that is not proportional to $$\delta S$$ and moves the decoupled hyperplane; none is computed here. A non-linear coupling or screening in the Solar System violates (H4). Internal spaces other than a flat torus, fibrations and a graviphoton are outside (H1). A unit direction $$u$$ satisfies (3) in the absence of a brane term when $$\lvert\langle Z,u\rangle\rvert<2.5\times10^{-3}$$, that is, when it lies within $$4\times10^{-3}$$ to $$5\times10^{-3}$$ of the fixed-volume slice for $$n=1,\dots,6$$ ($$\lvert Z\rvert=0.50$$ to $$0.62$$), but the angle $$\alpha$$ of section 4 is set by the potential, and the geometry gives no reason for it to be near $$87^\circ$$.

## 6. Open points

The statements are classical and at two derivatives. Theorem 1 is proved for a torus with an arbitrary flat metric; for other internal spaces it relies on an argument about shape moduli that we have not completed. Whether any potential motivated by the scenario places the light direction near $$87^\circ$$ from the extremal direction is not known. For the brane-term escape, it is open whether an ultraviolet completion supplies a brane term carrying about 90% of $$M_P^2$$ while fixing the moduli of the other dimensions at the cutoff. For the warped escape, the computation covers only $$n=0$$ with the Standard Model on the ultraviolet brane and a massless radion. The coupling of other light fields of the scenario, such as the neutrino sector with a bulk right-handed tower, which is mentioned in [[Stewart]](#ref-stewart) (footnote 2), is not treated.

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
