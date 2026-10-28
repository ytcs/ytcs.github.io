---
title: How a dark-dimension modulus couples to baryons
date: 2026-10-28
order: 8
description: In an unwarped dark-dimension compactification with the Standard Model on a brane, a field that changes only the size of the dark dimension couples to baryons with one third of its coupling to dark matter, which the Cassini bound excludes at the coupling preferred by cosmological fits. Fields that leave baryons uncoupled must preserve the internal volume, and their dark-matter coupling is then capped at an order-one value.
---

In the dark-dimension scenario the dark matter is the Kaluza–Klein (KK) tower of a micron-sized extra circle, and a light scalar $$\phi$$ that changes the size of the circle makes the dark-matter mass drift. Fits of this fading-dark-sector model to BAO, CMB and supernova data give $$m_{\rm DM}\propto e^{-c'\phi}$$ with $$c'=0.05\pm0.01$$ in reduced Planck units [[BOVW]](#ref-bovw). We ask how strongly the same field couples to baryons, which live on the Standard Model brane. We work at tree level with an unwarped product compactification $$S^1\times X_n$$, brane couplings fixed in units of the higher-dimensional Planck mass, and no Einstein–Hilbert term on the brane. In this setting, if the circle radius is the only modulus that $$\phi$$ moves among those that set masses, the baryon coupling is $$\beta_b=c'/3$$. The reason is that brane masses are fixed in higher-dimensional Planck units while the four-dimensional Planck mass grows with the internal volume, so in four-dimensional Planck units brane masses scale as the inverse square root of the volume. At $$c'=0.05$$ the resulting $$\gamma-1$$ is about 45 times the Cassini bound (section 3). Baryons decouple exactly when $$\phi$$ leaves the total internal volume fixed, and on those directions $$\lvert c'\rvert\le\sqrt{n/(n+1)}$$, which is $$0.71$$–$$0.93$$ for $$n=1,\dots,6$$ (section 4).

The Cassini exclusion of the first statement is evaded by an Einstein–Hilbert term on the brane that supplies at least 87–91% of $$M_P^2$$, which is consistent only if the moduli of $$X_n$$ are not dynamical four-dimensional fields, since otherwise the volume modulus is a ghost (section 5). It is also evaded by a warped dark dimension with $$kL\gtrsim1.4$$, where $$k$$ is the curvature scale and $$L$$ the length of the extra dimension.

Montero, Vafa and Valenzuela proposed the dark dimension [[MVV]](#ref-mvv), and Gonzalo, Montero, Obied and Vafa proposed its KK gravitons as dark matter [[GMOV]](#ref-gmov). Bedroya, Obied, Vafa and Wu [[BOVW]](#ref-bovw) fitted the fading dark sector and noted (their section III) that $$\phi$$ may change the circle either at fixed total internal volume, which keeps Newton's constant fixed in Standard Model units, or with the volume varying. Vafa [[Vafa]](#ref-vafa) (appendix A) writes $$c'=\sqrt{3/2}\,\sin\theta$$ for a field with a component $$\sin\theta$$ along the radion. Mantry and Furuuchi [[MF]](#ref-mf) observe that a fixed total internal volume makes Newton's constant independent of the modulus, and the normalisation of their six-dimensional shape modulus corresponds to $$c'=1/\sqrt2$$, the case $$n=1$$ below (their section 5). They also build a five-dimensional model in which a brane-localised Einstein–Hilbert term dominates $$M_P^2$$ (their section 4). The bound of section 4 can be read off from the tower and species vectors of Etheredge et al. [[EHRRV]](#ref-taxonomy) (remark after Theorem 1). New here are the ratio $$\beta_b/c'=1/3$$ with its Cassini exclusion in the fading-dark-sector setting, the bound over the full metric moduli space of a torus with its consequence for the minimal model, and the quantitative conditions for both escapes, including the ghost condition. Two analyses of this model, by Stewart [[Stewart]](#ref-stewart) and in [[BOVW]](#ref-bovw), section IV, treat baryons as uncoupled from $$\phi$$; the comparison is made after Proposition 2.

The reader is assumed to know Kaluza–Klein reduction on a circle and the Einstein and Jordan frames of scalar–tensor gravity.

## 1. Couplings and the Cassini bound

We work in the four-dimensional Einstein frame with reduced Planck mass $$M_P=1$$ and a canonically normalised light field, $$\mathcal L=\tfrac12R-\tfrac12(\partial\phi)^2-V(\phi)$$. A species $$X$$ whose Einstein-frame mass is $$m_X(\phi)$$ has coupling

$$
\beta_X\equiv-\frac{d\ln m_X}{d\phi},
$$

and exchange of $$\phi$$, when it is light on the scale of the separation, adds a fraction $$2\beta_X\beta_Y$$ to the Newtonian force between non-relativistic bodies of species $$X$$ and $$Y$$. The dark-matter coupling of [[BOVW]](#ref-bovw) is $$c'=\beta_{\rm DM}$$, and their dark fifth force, $$2c'^2$$ times gravity (their eq. 18), uses the same normalisation.

For the Solar System the relevant quantity is the post-Newtonian parameter $$\gamma$$. A scalar with a linear coupling to the Sun is a Brans–Dicke field, with $$\alpha_0^2=1/(2\omega+3)$$ and $$\gamma-1=-2\alpha_0^2/(1+\alpha_0^2)$$ in the notation of Will's review [[Will]](#ref-will), and in our normalisation $$\alpha_0^2=2\beta_b^2$$. Cassini measured $$\gamma-1=(2.1\pm2.3)\times10^{-5}$$ [[BIT]](#ref-bit), so at two standard deviations $$\gamma-1>-2.5\times10^{-5}$$, and

$$
\lvert\beta_b\rvert<2.5\times10^{-3}. \tag{1}
$$

This applies when the range of $$\phi$$ exceeds an astronomical unit, as it does for a field that rolls on cosmological time scales. We will find that every brane species has the same coupling (see (5)), so tests of the equivalence principle among ordinary materials do not constrain it, and (1) is the relevant bound.

## 2. Reduction with a brane

We take $$D=5+n$$ dimensions with an unwarped product of four-dimensional spacetime and a torus $$T^{n+1}$$. The torus makes the computation explicit; how the results extend to a curved $$X_n$$ is noted after Theorem 1. For the diagonal case the metric is

$$
ds^2=g_{\mu\nu}(x)\,dx^\mu dx^\nu+\sum_{a=0}^{n}e^{2\sigma_a(x)}\,dy_a^2,\qquad y_a\sim y_a+2\pi,
$$

with lengths in units of $$M_D^{-1}$$, where $$M_D$$ is the $$D$$-dimensional Planck mass. The direction $$a=0$$ is the dark circle, of radius $$e^{\sigma_0}$$, and $$a=1,\dots,n$$ are the small dimensions $$X_n$$. The total internal volume is proportional to $$e^S$$, where $$S\equiv\sum_a\sigma_a$$, and we set $$\sigma_a=0$$ at the present values of the moduli.

The setting is fixed by the following hypotheses.

(H1) Tree-level two-derivative gravity in $$D$$ dimensions, with the unwarped product geometry above. Potentials for the moduli are allowed but do not enter the kinetic terms or couplings computed here.

(H2) The Standard Model lives on a point-like 3-brane and couples only to the induced metric $$g_{\mu\nu}$$. Every brane parameter (position, tension, couplings, any dilaton) is fixed in units of $$M_D$$, so every brane mass, including the nucleon mass through $$\Lambda_{\rm QCD}$$, is constant in the frame $$g_{\mu\nu}$$. The brane may carry an Einstein–Hilbert term $$\tfrac12M_b^2\sqrt{-g}\,R$$; the case $$M_b=0$$ is the setting of the introduction, without a brane term.

(H3) The dark matter is the KK tower of the dark circle, with masses $$m_j=\lvert j\rvert e^{-\sigma_0}$$ in units of $$M_D$$ in the frame $$g_{\mu\nu}$$.

A direct computation of the Christoffel symbols of the $$D$$-dimensional metric gives the exact identity

$$
R_D=R-2\,\Box S-(\partial S)^2-\sum_a(\partial\sigma_a)^2,
$$

where $$R$$, $$\Box$$ and the contractions refer to $$g_{\mu\nu}$$; for a single circle it reduces to the familiar $$R-2e^{-\sigma}\Box e^{\sigma}$$. (We checked it symbolically for two circles with moduli depending on two spacetime coordinates.) Multiplying by $$\sqrt{-g_D}=e^S\sqrt{-g}$$, integrating over the torus and integrating $$e^S\Box S$$ by parts gives the bulk action $$\tfrac12M_D^{D-2}V_0\int d^4x\sqrt{-g}\,e^S\big[R+(\partial S)^2-\sum_a(\partial\sigma_a)^2\big]$$, with $$V_0$$ the present internal volume. The four-dimensional Planck mass is $$M_P^2=M_D^{D-2}V_0+M_b^2$$. We define the bulk fraction

$$
F\equiv\frac{M_D^{D-2}V_0}{M_P^2}\in(0,1],
$$

so that the brane term supplies the fraction $$1-F$$ of $$M_P^2$$; $$F=1$$ means no brane term. In units $$M_P=1$$, the action in the frame of the induced metric $$g_{\mu\nu}$$, which we call the brane frame (the Jordan frame of brane matter, in which brane masses are constant), is

$$
\sqrt{-g}\,\Big[\tfrac12\Phi\,R+\tfrac12F e^S\Big((\partial S)^2-\sum_a(\partial\sigma_a)^2\Big)\Big],\qquad \Phi\equiv Fe^S+1-F. \tag{2}
$$

The moduli multiply $$R$$, so this is a scalar–tensor theory, and the couplings are read off after a Weyl rescaling to the Einstein frame.

<div class="proposition" markdown="1">
**Proposition 1** (moduli metric). With $$g_{\mu\nu}=\Phi^{-1}g^E_{\mu\nu}$$, the action (2) equals, up to a total derivative,

$$
\sqrt{-g_E}\,\Big[\tfrac12R_E-\tfrac12G_{ab}\,\partial\sigma_a\partial\sigma_b\Big],\qquad G=w\,(\mathbb 1-vv^{\sf T})+\tfrac32w^2\,vv^{\sf T},\qquad w\equiv\frac{Fe^S}{\Phi}, \tag{3}
$$

where $$v=(1,\dots,1)$$, so that $$vv^{\sf T}$$ is the matrix of ones. At the present point, $$w=F$$, and $$G$$ has eigenvalue $$F$$ on the $$n$$-dimensional subspace $$\sum_au_a=0$$ and eigenvalue

$$
F\big[\tfrac32(n+1)F-n\big] \tag{4}
$$

on the volume direction $$u_a=1$$ for all $$a$$.
</div>

<div class="proof" markdown="1">
*Proof.* Under $$g=\Phi^{-1}g_E$$ one has $$\sqrt{-g}\,\Phi R=\sqrt{-g_E}\,\big[R_E-\tfrac32(\partial\ln\Phi)^2_E\big]$$ up to a total derivative, and $$\sqrt{-g}\,(\partial f)^2=\Phi^{-1}\sqrt{-g_E}\,(\partial f)^2_E$$ for any scalar $$f$$. Since $$\partial\ln\Phi=w\,\partial S$$, the first term of (2) contributes $$-\tfrac34w^2(\partial S)^2=-\tfrac12\cdot\tfrac32w^2\sum_{a,b}\partial\sigma_a\partial\sigma_b$$, and the second contributes $$\tfrac12w\big(\sum_{a,b}\partial\sigma_a\partial\sigma_b-\sum_a(\partial\sigma_a)^2\big)$$, which is $$-\tfrac12w\,\partial\sigma^{\sf T}(\mathbb 1-vv^{\sf T})\,\partial\sigma$$. Together these give (3). At $$w=F$$, $$G=F\,\mathbb 1+(\tfrac32F^2-F)\,vv^{\sf T}$$; the matrix $$vv^{\sf T}$$ annihilates vectors with $$\sum_au_a=0$$ and has eigenvalue $$n+1$$ on $$v$$, which gives (4).
</div>

(On a homogeneous cosmological background with one, two and three radii, the Euler–Lagrange equations of the difference between (2) and (3) vanish identically.) For $$F=1$$, $$G=\mathbb 1+\tfrac12vv^{\sf T}$$, the standard metric for the radii of a torus. For a single circle it gives the kinetic coefficient $$\tfrac32$$ of the radion, and the brane frame (2) is Brans–Dicke theory with $$\omega=0$$.

The masses follow from (H2) and (H3). A mass $$m$$ that is constant in the frame $$g_{\mu\nu}$$ is $$m\,\Phi^{-1/2}$$ in the Einstein frame, so

$$
d\ln m_b=-\tfrac12w\,dS,\qquad d\ln m_{\rm DM}=-d\sigma_0-\tfrac12w\,dS. \tag{5}
$$

The first relation holds for every brane species, which is the universality used in section 1. Let the light field move the moduli along a direction $$u$$ normalised by $$G(u,u)=1$$, so that $$d\sigma_a/d\phi=u_a$$. At the present point (5) gives

$$
\beta_b=\tfrac12F\sum_au_a,\qquad c'=u_0+\tfrac12F\sum_au_a. \tag{6}
$$

The light direction may also move fields on which neither mass depends, such as shape moduli of $$X_n$$ or axions; those components change the normalisation of $$u$$ but do not enter (6).

In words: in four-dimensional Planck units $$m_b\propto e^{-S/2}$$ and $$m_{\rm DM}\propto e^{-\sigma_0-S/2}$$ when $$F=1$$. If only the dark circle moves, $$dS=d\sigma_0$$, so $$\beta_b=\tfrac12\,d\sigma_0/d\phi$$ and $$c'=\tfrac32\,d\sigma_0/d\phi$$, and their ratio is $$\tfrac13$$.

## 3. The radion

Suppose first that $$\phi$$ changes the radius of the dark circle and none of the other radii. This is the case considered in [[Stewart]](#ref-stewart), and in [[Vafa]](#ref-vafa) when the remaining components of the light field enter no mass.

<div class="proposition" markdown="1">
**Proposition 2** (radion coupling). Assume (H1)–(H3), and let the light direction have $$u_a=0$$ for $$a=1,\dots,n$$, with arbitrary components along fields that enter neither mass. Then

$$
\frac{\beta_b}{c'}=\frac{F}{2+F},
$$

which is $$\tfrac13$$ without a brane Einstein–Hilbert term. For the pure radion, $$u=e_0/\sqrt{G_{00}}$$ with $$G_{00}=\tfrac32F^2$$, so $$\beta_b=1/\sqrt6$$ for every $$F$$ and $$c'=(2+F)/(\sqrt6\,F)$$.
</div>

<div class="proof" markdown="1">
*Proof.* With $$\sum_au_a=u_0$$, (6) gives $$\beta_b=\tfrac12Fu_0$$ and $$c'=(1+\tfrac12F)u_0$$, whose ratio is $$F/(2+F)$$ whatever the normalisation. For the pure radion, $$G_{00}=w(1-1)+\tfrac32w^2=\tfrac32F^2$$ from (3), since the diagonal entries of $$\mathbb 1-vv^{\sf T}$$ vanish, so $$u_0=\sqrt{2/3}/F$$, and substituting gives the stated values.
</div>

For $$F=1$$ the pure radion has $$c'=\sqrt{3/2}$$ and $$\beta_b=1/\sqrt6$$. The latter is the coupling of a Brans–Dicke field with $$\omega=0$$, since $$2\beta_b^2=\tfrac13=1/(2\omega+3)$$. The observed $$c'\approx0.05$$ is far below $$\sqrt{3/2}$$, so in this case the light field must be mostly something other than the radion, as [[Vafa]](#ref-vafa) assumes; Proposition 2 says that the ratio $$\beta_b/c'$$ does not depend on how small the radion component is.

Without a brane term, $$c'=0.05$$ gives $$\beta_b=0.0167$$ and $$\gamma-1=-1.1\times10^{-3}$$, about 45 times the two-standard-deviation bound. By (1), every $$c'>3\times2.5\times10^{-3}=7.5\times10^{-3}$$ is excluded for such a field. The fitted range $$c'=0.04$$–$$0.06$$ gives factors of 28 to 64.

<div class="remark" markdown="1">
**Remark** (comparison with published treatments). Stewart [[Stewart]](#ref-stewart), section 2.1, takes $$\phi$$ to be the radion of the dark dimension and states that the Standard Model "does not couple to the radion at leading order, so the baryons are uncoupled dust", which enters his eq. (2.2c) as $$\rho_b\propto a^{-3}$$. His footnote 2 adds that brane masses typically would not depend on the radius; in higher-dimensional Planck units this is hypothesis (H2). As a statement about four-dimensional masses, under (H1)–(H3) it holds only on directions that preserve the total internal volume (section 4). For the pure radion Proposition 2 gives $$\beta_b=1/\sqrt6$$, and for any field whose only mass-relevant component is the radion it gives $$\beta_b=c_2/3$$, where $$c_2$$, the logarithmic slope of the tower mass in his eq. (2.10), equals $$c'$$. Bedroya et al. [[BOVW]](#ref-bovw), section IV, state that visible matter "does not couple directly to $$\phi$$". By the same argument this is correct on their fixed-volume branch and not on the varying-volume branch that their section III also allows.
</div>

## 4. Directions that decouple baryons

By (6), $$\beta_b=0$$ if and only if $$\sum_au_a=0$$, that is, if the light field preserves the total internal volume in units of $$M_D$$. This holds for any $$F>0$$. The question is then how large $$c'$$ can be on such a direction. On the fixed-volume slice $$G=F\,\mathbb 1$$, by Proposition 1, and $$c'=u_0$$, so we need the largest $$u_0$$ among vectors of Euclidean length $$F^{-1/2}$$ with zero component sum. The answer is the length of the projection of $$e_0$$ onto the hyperplane $$\sum_au_a=0$$, which is $$\sqrt{n/(n+1)}$$, divided by $$\sqrt F$$. The extremal direction increases the dark circle and shrinks each of the $$n$$ small radii by $$1/n$$ of that amount in the logarithm.

The same bound holds when the torus metric has off-diagonal components, and the proof is a Cauchy–Schwarz inequality. Write the internal metric as a positive matrix $$h_{ab}(x)$$, constant along $$T^{n+1}$$, so that the diagonal case is $$h={\rm diag}(e^{2\sigma_a})$$, and describe a tangent direction by the symmetric matrix $$Q\equiv h^{-1/2}\,\delta h\,h^{-1/2}$$. For the diagonal case $$Q={\rm diag}(2u_a)$$. The general torus reduction [[MS]](#ref-ms) replaces $$\sum_a(\partial\sigma_a)^2$$ by $$\tfrac14{\rm tr}(h^{-1}\partial h)^2$$ and $$\partial S$$ by $$\tfrac12{\rm tr}(h^{-1}\partial h)$$; we checked this form symbolically for a non-diagonal $$2\times2$$ block. Proposition 1 then gives

$$
G(Q,Q)=\tfrac14F\big[{\rm tr}\,Q^2-({\rm tr}\,Q)^2\big]+\tfrac38F^2({\rm tr}\,Q)^2 .
$$

The KK masses on a torus are $$m^2=p_ah^{ab}p_b$$ for integer $$p$$, and the dark-matter tower is $$p=(j,0,\dots,0)$$, so $$m_{\rm DM}^2=j^2h^{00}$$ in units of $$M_D$$. In the diagonal case this reduces to (H3).

<div class="theorem" markdown="1">
**Theorem 1** (decoupled directions). Assume (H1)–(H3) with internal space $$T^{n+1}$$, $$n\ge1$$, and any flat metric $$h$$. A light direction has $$\beta_b=0$$ if and only if $${\rm tr}\,Q=0$$, that is, if it preserves the total internal volume. Every such direction has

$$
\lvert c'\rvert\le\sqrt{\frac{n}{(n+1)F}}, \tag{7}
$$

with equality exactly for $$Q\propto\hat e\hat e^{\sf T}-\mathbb 1/(n+1)$$, where $$\hat e=h^{-1/2}e_0/\lvert h^{-1/2}e_0\rvert$$. Without a brane term the bound is $$\sqrt{n/(n+1)}$$.
</div>

<div class="proof" markdown="1">
*Proof.* The first statement is (5) with $$dS=\tfrac12{\rm tr}\,Q$$. On the slice $${\rm tr}\,Q=0$$, normalisation means $${\rm tr}\,Q^2=4/F$$, and the Einstein-frame factor in (5) drops out of $$m_{\rm DM}$$. Since $$\delta(h^{-1})=-h^{-1}\,\delta h\,h^{-1}$$ and $$h^{00}=\lvert h^{-1/2}e_0\rvert^2$$,

$$
c'=-\tfrac12\,\delta\ln h^{00}=\tfrac12\,\hat e^{\sf T}Q\,\hat e=\tfrac12\,{\rm tr}(QP),\qquad P\equiv\hat e\hat e^{\sf T}-\frac{\mathbb 1}{n+1},
$$

where the last step uses $${\rm tr}\,Q=0$$. The matrix $$P$$ is symmetric with $${\rm tr}\,P^2=1-\tfrac{2}{n+1}+\tfrac{1}{n+1}=\tfrac{n}{n+1}$$. The Cauchy–Schwarz inequality for the trace inner product gives $$\lvert{\rm tr}(QP)\rvert\le({\rm tr}\,Q^2)^{1/2}({\rm tr}\,P^2)^{1/2}=2\sqrt{n/((n+1)F)}$$, with equality iff $$Q\propto P$$, and $$P$$ is itself traceless.
</div>

Writing $$c'=\sqrt{n/((n+1)F)}\,\cos\alpha$$ defines $$\alpha$$ as the angle, in the metric $$G$$, between the light direction and the extremal direction of the theorem. For $$n=1$$ and $$F=1$$ the extremal direction is the shape modulus of [[MF]](#ref-mf), section 5.

For a product $$S^1\times X_n$$ with $$X_n$$ Ricci-flat but not a torus, we expect the same structure for the overall sizes, by the following argument, which is not a proof. The shape moduli of $$X_n$$ are traceless deformations of its metric, so they change neither the volume nor the circle; they are therefore $$G$$-orthogonal to the circle and volume directions and enter neither mass. A light field with a shape-modulus component has a smaller $$\lvert\cos\alpha\rvert$$, and (7) still holds. We have not treated fibred geometries or a graviphoton.

The bound can also be read from the tower geometry of [[EHRRV]](#ref-taxonomy), in which a tower vector is the moduli-space gradient of minus the logarithm of a tower mass and a species vector the same for the species scale, both in four-dimensional Planck units. With $$F=1$$, the tower vector of the dark circle, $$\zeta=-\nabla\ln m_{\rm DM}$$, and the species vector of the $$D$$-dimensional Planck scale, $$Z=-\nabla\ln M_D$$, satisfy $$\lvert\zeta\rvert^2=\tfrac32$$ and $$\zeta\cdot Z=\tfrac12$$ (their eqs. 1.6 and 1.3 with $$d=4$$), and the metric (3) gives $$\lvert Z\rvert^2=(n+1)/(2(n+3))$$. Since $$\beta_b$$ is the component of the light direction along $$Z$$ and $$c'$$ the component along $$\zeta$$, the largest $$c'$$ orthogonal to $$Z$$ is $$\big(\lvert\zeta\rvert^2-(\zeta\cdot Z)^2/\lvert Z\rvert^2\big)^{1/2}=\sqrt{n/(n+1)}$$.

We call minimal the model in which only the dark circle and the overall size of $$X_n$$ are light; it is the simplest realisation of the fixed-volume branch of [[BOVW]](#ref-bovw) (their Fig. 1), in which the dark dimension grows while the other dimensions contract. The fixed-volume slice is then one-dimensional, it consists of the extremal direction, and a light field that decouples baryons has $$\lvert c'\rvert=\sqrt{n/(n+1)}$$ with no free parameter:

| $$n$$ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| $$\sqrt{n/(n+1)}$$ | 0.707 | 0.816 | 0.866 | 0.894 | 0.913 | 0.926 |

The fitted value $$c'=0.05\pm0.01$$ [[BOVW]](#ref-bovw) is smaller than these numbers by 66 to 88 times its quoted uncertainty. Independently of the fit, $$c'\ge0.71$$ violates the bound $$c'\lesssim0.2$$ from tidal disruption of Milky Way satellites quoted in [[BOVW]](#ref-bovw), which corresponds to a dark fifth force $$2c'^2\lesssim0.08$$ times gravity, against $$1.0$$–$$1.7$$ here. The minimal model is therefore excluded. The range in the table assumes $$n\le6$$; for larger $$n$$ the value approaches 1. In type IIB string theory with the Standard Model on D3-branes, $$n=5$$ and the value is $$0.913$$. For $$n=6$$ (eleven dimensions) M-theory has no point-like 3-brane, and a wrapped M5-brane has couplings that depend on the size of the wrapped cycle, which violates (H2).

With more light moduli, $$c'=0.05$$ requires $$\cos\alpha=0.055\pm0.011$$ for $$n=5$$, an angle $$\alpha=86.9^\circ\pm0.6^\circ$$ from the extremal direction. The geometry does not fix $$\alpha$$; the potential selects the light direction.

## 5. Two ways around the exclusion

### 5.1 An Einstein–Hilbert term on the brane

When $$F<1$$, Proposition 2 gives $$\beta_b=c'F/(2+F)$$ for a light field whose only mass-relevant component is the radion. The coupling of the pure radion per canonical unit stays $$1/\sqrt6$$, so a field that is purely the radion remains excluded; at fixed observed $$c'$$, however, a small $$F$$ lowers $$\beta_b$$. The condition (1) becomes $$F<2r/(1-r)$$ with $$r=2.5\times10^{-3}/c'$$:

| $$c'$$ | 0.04 | 0.05 | 0.06 |
|---|---|---|---|
| largest bulk fraction $$F$$ | 0.133 | 0.105 | 0.087 |
| smallest brane share $$1-F$$ of $$M_P^2$$ | 86.7% | 89.5% | 91.3% |

The brane term must therefore supply most of $$M_P^2$$. The ghost condition comes from (4). If the radii of $$X_n$$ are dynamical four-dimensional fields, the volume direction has negative kinetic energy whenever

$$
F<\frac{2n}{3(n+1)},
$$

that is, for $$F<\tfrac13$$ when $$n=1$$ and $$F<\tfrac59$$ when $$n=5$$. The right-hand side is at least $$\tfrac13$$ for every $$n\ge1$$, so the region allowed by Cassini, $$F\lesssim0.1$$, lies inside the ghost region. The escape is consistent only if the moduli of $$X_n$$ are not light four-dimensional degrees of freedom: either there are no further dimensions ($$n=0$$, a five-dimensional theory), or the size of $$X_n$$ is fixed at or above the cutoff. The radion itself never becomes a ghost, since $$G_{00}=\tfrac32F^2>0$$. In warped models, Dillon, George and McDonald [[DGM]](#ref-dgm) showed that brane curvature terms change the radion's coupling to brane matter and that large ones can make the radion ghost-like.

With a brane at $$y_0=0$$, the KK modes odd under $$y_0\to-y_0$$ vanish on the brane and keep $$m_j\propto e^{-\sigma_0}$$ exactly, while the even modes are shifted by the brane term; (H3) is exact for the odd modes. For $$F\le0.1$$ the bound (7) exceeds 2 and no longer constrains $$c'$$. Finally, whether the brane term can be this large depends on the ultraviolet completion. A one-loop estimate $$M_b^2\sim N\Lambda^2/16\pi^2$$ with $$N\sim10^2$$ species and cutoff $$\Lambda$$ at the five-dimensional Planck scale, $$10^9$$–$$10^{10}\,$$GeV [[MVV]](#ref-mvv), gives $$M_b^2/M_P^2\sim10^{-19}$$–$$10^{-17}$$; the model of [[MF]](#ref-mf) instead assumes brane fields whose cutoff lies far above that scale.

### 5.2 Warping

Now let the extra dimension be an interval of length $$L$$ with anti-de Sitter curvature $$k$$ (Randall–Sundrum geometry with $$n=0$$), with the Standard Model on the positive-tension (ultraviolet) brane, no brane Einstein–Hilbert term, and the dark matter the KK gravitons. Write $$\Omega\equiv e^{-kL}$$ and $$M_0^2\equiv M_5^3/k$$, with $$M_5$$ the five-dimensional Planck mass. Garriga and Tanaka [[GT]](#ref-gt) showed that linearised gravity on the positive-tension brane is Brans–Dicke theory, with Brans–Dicke scalar (the Einstein-frame Planck mass) $$M_P^2=M_0^2(1-\Omega^2)$$ (their eq. 26) and parameter $$\omega=\tfrac32(\Omega^{-2}-1)$$ (their eq. 27). A Brans–Dicke theory $$\tfrac12\Phi R-\tfrac{\omega}{2\Phi}(\partial\Phi)^2$$ is fixed by these two functions. With $$\Phi=M_0^2(1-\Omega^2)$$ and $$\partial\Phi=-2M_0^2\Omega\,\partial\Omega$$, the radion sector in the frame of the ultraviolet brane is therefore

$$
\tfrac12M_0^2(1-\Omega^2)\,R-3M_0^2(\partial\Omega)^2 .
$$

As $$k\to0$$, $$1-\Omega^2\to2kL$$ and the second term vanishes, which is the flat circle with $$\omega=0$$. The Weyl rescaling of Proposition 1, with $$\Phi$$ in place of $$Fe^S+1-F$$ and $$M_P^2=\Phi$$ today, gives the canonical kinetic coefficient $$\tfrac32(\partial_\Omega\ln\Phi)^2+6M_0^2/\Phi=6/(1-\Omega^2)^2$$ in units $$M_P=1$$, so $$d\phi=\sqrt6\,d\Omega/(1-\Omega^2)$$. Brane masses are constant in the ultraviolet frame, so $$d\ln m_b=-\tfrac12d\ln\Phi=\Omega\,d\Omega/(1-\Omega^2)$$, and $$\lvert\beta_b\rvert=\Omega/\sqrt6$$. The KK graviton masses in the same frame solve

$$
J_1(m/k)\,Y_1\big(m/(k\Omega)\big)-Y_1(m/k)\,J_1\big(m/(k\Omega)\big)=0,
$$

which for $$kL\to0$$ gives $$m_j=\pi j/L$$ and for $$kL\gg1$$ gives $$m_j\approx x_jk\Omega$$ with $$x_j$$ the zeros of $$J_1$$. At large $$kL$$ the Planck mass depends on $$\Omega$$ only through $$1-\Omega^2$$, while the KK masses are proportional to $$\Omega$$, so $$\beta_b/\beta_{\rm DM}\approx\Omega^2=e^{-2kL}$$. Solving the mass condition numerically for the lightest mode:

| $$kL$$ | 0.05 | 0.5 | 1 | 1.5 | 2 | 3 |
|---|---|---|---|---|---|---|
| $$\beta_b/\beta_{\rm DM}$$ | 0.317 | 0.189 | 0.093 | 0.041 | 0.017 | 0.0024 |
| $$e^{-2kL}$$ | 0.905 | 0.368 | 0.135 | 0.050 | 0.018 | 0.0025 |

For a light field whose mass-relevant component is the radion, (1) at $$c'=0.05$$ requires $$\beta_b/\beta_{\rm DM}<0.05$$, which holds for $$kL>1.39$$; the threshold is $$1.25$$ at $$c'=0.04$$ and $$1.50$$ at $$c'=0.06$$, and the thresholds for the fifth KK mode are lower by 0.03 in $$kL$$. A pure warped radion, with $$\lvert\beta_b\rvert=e^{-kL}/\sqrt6$$, needs $$kL>5.1$$. These numbers are a computation within the linearised two-brane effective theory of [[GT]](#ref-gt).

## 6. Scope and open questions

The statements are classical and at two derivatives. Loop effects that make brane couplings depend on the moduli, for example through the species scale, are not included, nor are brane-position moduli or a dilaton varying on the brane; any of these adds a baryon coupling that is not proportional to $$dS$$ and moves the decoupled hyperplane. Theorem 1 is proved for a torus with an arbitrary flat metric. For other $$X_n$$ it relies on the shape moduli being traceless deformations, and fibred internal spaces and a graviphoton are not treated.

Within the unwarped setting without a brane term, a nonzero $$c'$$ and a value of $$\gamma$$ consistent with 1 at the $$10^{-5}$$ level require the light direction to lie close to the fixed-volume slice; any volume admixture appears as a universal coupling to brane matter and a deficit in $$\gamma$$. The angle $$\alpha$$ of section 4 is set by the potential, and it is not known whether any potential motivated by the scenario places the light direction near $$87^\circ$$ from the extremal direction. For the brane-term escape, it is open whether an ultraviolet completion supplies a brane term carrying about 90% of $$M_P^2$$ while fixing the moduli of $$X_n$$ at the cutoff. For the warped escape, the computation covers only $$n=0$$ with the Standard Model on the ultraviolet brane, and it does not address whether the dark-matter phenomenology of a micron-sized dimension survives warping with $$kL\approx1.4$$.

## References

1. <a id="ref-mvv"></a>Montero, Vafa, Valenzuela, “The dark dimension and the Swampland,” [arXiv:2205.12293](https://arxiv.org/abs/2205.12293).
2. <a id="ref-gmov"></a>Gonzalo, Montero, Obied, Vafa, “Dark dimension gravitons as dark matter,” [arXiv:2209.09249](https://arxiv.org/abs/2209.09249).
3. <a id="ref-bovw"></a>Bedroya, Obied, Vafa, Wu, “Evolving dark sector and the dark dimension scenario,” [arXiv:2507.03090](https://arxiv.org/abs/2507.03090).
4. <a id="ref-vafa"></a>Vafa, “The phenomenology of the dark dimension and non-supersymmetric strings,” [arXiv:2610.04909](https://arxiv.org/abs/2610.04909).
5. <a id="ref-mf"></a>Mantry, Furuuchi, “Relating dark dimension size with dark energy via 5D quintessence potential and Casimir energy,” [arXiv:2610.35124](https://arxiv.org/abs/2610.35124).
6. <a id="ref-stewart"></a>Stewart, “Cosmological stasis and the coupled dark sector of the Dark Dimension,” [arXiv:2609.25230](https://arxiv.org/abs/2609.25230).
7. <a id="ref-taxonomy"></a>Etheredge, Heidenreich, Rudelius, Ruiz, Valenzuela, “Taxonomy of infinite distance limits,” [arXiv:2405.20332](https://arxiv.org/abs/2405.20332).
8. <a id="ref-ms"></a>Maharana, Schwarz, “Noncompact symmetries in string theory,” [arXiv:hep-th/9207016](https://arxiv.org/abs/hep-th/9207016).
9. <a id="ref-will"></a>Will, “The confrontation between general relativity and experiment,” [arXiv:1403.7377](https://arxiv.org/abs/1403.7377).
10. <a id="ref-bit"></a>Bertotti, Iess, Tortora, “A test of general relativity using radio links with the Cassini spacecraft,” Nature 425 (2003) 374–376, [doi:10.1038/nature01997](https://doi.org/10.1038/nature01997).
11. <a id="ref-gt"></a>Garriga, Tanaka, “Gravity in the Randall–Sundrum brane world,” [arXiv:hep-th/9911055](https://arxiv.org/abs/hep-th/9911055).
12. <a id="ref-dgm"></a>Dillon, George, McDonald, “Regarding the radion in Randall–Sundrum models with brane curvature,” [arXiv:1605.03087](https://arxiv.org/abs/1605.03087).
