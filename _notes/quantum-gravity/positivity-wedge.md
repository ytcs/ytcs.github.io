---
title: A positivity wedge for inflationary non-Gaussianity
date: 2026-10-29
category: dark-sector
order: 8
description: When the primordial bispectrum comes from tree-level exchange with a unitary hidden sector coupled through the de Sitter-invariant vertex, the equilateral and orthogonal amplitudes lie in a convex cone, a narrow wedge with negative equilateral and positive orthogonal amplitude. Its edge angles are computed for stated inner products; each hypothesis that carries the wedge is identified, and the sign of the boost-breaking admixture is fixed in one class of couplings.
status: The convex-cone statement is a theorem under stated hypotheses; the edge angles are numerical, for a Fergusson-Shellard-type proxy and three other weightings, and a CMB Fisher metric is not computed. The sign result for the boost-breaking admixture is proved in the class where the hidden operator multiplies the inflaton kinetic function.
excludes: Positive equilateral amplitude from tree-level exchange of a unitary hidden sector with principal-series spectrum, no subtraction, no three-point function of the hidden operator and no boost-breaking vertex; amplitude directions outside the computed wedge for that class.
---

If the inflaton couples to a hidden sector of heavy states, exchange of those states makes the primordial fluctuations non-Gaussian. Unitarity of the hidden sector implies that its two-point function has a spectral decomposition with a non-negative density, and de Rham, Jazayeri and Tolley [[DJT]](#ref-djt) used this to show that the shape of the resulting bispectrum is confined to a limited region of shape space. They posed the corresponding question for the quantities that experiments report, the amplitudes of the equilateral and orthogonal templates. The question here is which pairs of amplitudes $$(\fe,\fo)$$ such an exchange can produce. The answer has two parts. Under the hypotheses (H1)–(H6) of sections 1 and 2, the set of reachable pairs is exactly a convex cone, a wedge with its vertex at the origin (Theorem 1), because the amplitudes are linear in the spectral density. Its two edges are the directions produced by two particular single-particle exchanges, and they are computed numerically: for the inner product of section 1 the wedge is $$164.05^\circ\le\arg(\fe+i\fo)\le176.50^\circ$$ (angles measured anticlockwise from the positive $$\fe$$ axis), so that $$\fe<0<\fo$$ and $$0.061\,\lvert\fe\rvert\le\fo\le0.286\,\lvert\fe\rvert$$. Three of the hypotheses carry the result: that there is no subtraction in the spectral integral, that the three-point function of the hidden operator is negligible, and that the vertex has no boost-breaking admixture. Without any one of them, positivity no longer implies the wedge (section 5).

The setting, the positivity argument for the shape and the template question are due to [[DJT]](#ref-djt). Cabass et al. [[Cabass24]](#ref-cabass24) searched BOSS and Planck data for spin-zero particles during inflation using the exchange bispectrum at fixed mass and coupling. Ghosh, Maity and Ullah [[GMU]](#ref-gmu) showed that further operators of the effective field theory of inflation make the sign of the equilateral bispectrum non-universal. What this page adds is the reduction of the template question to a convex cone, the explicit edges for several inner products with an independent check of the single-exchange shapes, a statement of which hypotheses carry the wedge together with the dependence of the edges on them, a proof that in one class of couplings the boost-breaking admixture has a definite sign, and the comparison with the Planck amplitudes.

Familiarity with the primordial bispectrum and its shape function, and with the mass parameter of a field in de Sitter space, is assumed. Related pages: the [dark-sector](/quantum-gravity/landscape/) map places this result among the dark-sector statements, and [The sign of energy exchange between vacuum and matter](/quantum-gravity/vacuum-matter-sign/) concerns a different positivity argument.

## 1. What is measured

Let $$\zeta$$ be the curvature perturbation and $$B_\zeta(k_1,k_2,k_3)$$ its bispectrum, defined by $$\langle\zeta_{\mathbf k_1}\zeta_{\mathbf k_2}\zeta_{\mathbf k_3}\rangle=(2\pi)^3\delta^3(\mathbf k_1+\mathbf k_2+\mathbf k_3)B_\zeta$$. The shape function $$S=(k_1k_2k_3)^2B_\zeta$$ is scale invariant in the cases considered here, so it is a function on triangle shapes. The two templates used by Planck [[Planck18]](#ref-planck18), eqs. (1) and (2), are, for a scale-invariant spectrum, the equilateral and orthogonal shapes

$$
\begin{aligned}
S_{\rm eq}&=(k_1k_2k_3)^2\Big[-\sum_{3}\frac1{k_i^3k_j^3}-\frac{2}{(k_1k_2k_3)^2}+\sum_{6}\frac{1}{k_ak_b^2k_c^3}\Big],\\
S_{\rm or}&=(k_1k_2k_3)^2\Big[-3\sum_{3}\frac1{k_i^3k_j^3}-\frac{8}{(k_1k_2k_3)^2}+3\sum_{6}\frac{1}{k_ak_b^2k_c^3}\Big],
\end{aligned} \tag{1}
$$

where $$\sum_3$$ runs over the three pairs and $$\sum_6$$ over the six permutations of $$(a,b,c)$$; both equal 1 at $$k_1=k_2=k_3$$. The sign convention is that of WMAP and Planck: $$f_{\rm NL}\propto+B_\zeta(k,k,k)$$. The wedge depends only on the ratio $$\fo/\fe$$, so the overall normalisation of $$f_{\rm NL}$$ does not enter.

Experiments fit both templates simultaneously. Given a positive-definite inner product $$\langle\cdot,\cdot\rangle$$ on shapes, we define the reported amplitudes of a shape $$S$$ as the joint least-squares coefficients

$$
f=G^{-1}b,\qquad G_{XY}=\langle S_X,S_Y\rangle,\qquad b_X=\langle S,S_X\rangle, \tag{2}
$$

with $$X,Y\in\{\rm eq,or\}$$ (hypothesis H6). The inner product reflects the experiment's noise; for the CMB it is the bispectrum Fisher matrix. We do not compute that matrix. The inner product used for the numbers below is the Fergusson–Shellard proxy

$$
\langle A,B\rangle=\int_{V}A\,B\,\frac{dk_1\,dk_2\,dk_3}{k_1+k_2+k_3}, \tag{3}
$$

with $$V$$ the tetrapyd, the set of $$(k_1,k_2,k_3)$$ satisfying the triangle inequalities with each $$k_i\le k_{\max}$$. Writing $$k_i=Kx_i$$ with $$K=k_1+k_2+k_3$$ and $$\sum x_i=1$$ and integrating over $$K$$, (3) becomes $$\int d^2x\,AB\,(\max_ix_i)^{-2}$$ up to a constant, and we integrate over the fundamental domain $$x_1\ge x_2\ge x_3$$ by Gauss–Legendre quadrature. We also report three other weightings, $$(\max x_i)^{-1}$$, $$(\max x_i)^{-3}$$ (a flat three-dimensional measure) and the inverse area of the triangle (a flat-sky weight). With (3) the cosine of the angle between the templates is $$G_{\rm eo}/\sqrt{G_{\rm ee}G_{\rm oo}}=+0.142$$, so the templates are not orthogonal and the joint and separate projections differ. The only property of (2) used in the theorem below is that $$f$$ is linear in $$S$$.

## 2. Exchange of a hidden sector

Let the hidden sector couple to the inflaton through a scalar operator $$\cO$$ with $$\langle\cO\rangle=0$$. The setting is the decoupling limit of [[DJT]](#ref-djt): the inflaton $$\phi=\dot{\bar\phi}\,t+\delta\phi$$ evolves on a fixed de Sitter background of Hubble rate $$H=1$$, with $$\delta\phi$$ massless, $$\zeta=-H\delta\phi/\dot{\bar\phi}$$, and the breaking of de Sitter boosts is due to the background $$\dot{\bar\phi}$$ only. We use the following hypotheses.

- (H1) In the Bunch–Davies state the two-point function of $$\cO$$ in the absence of the coupling has the de Sitter-invariant Källén–Lehmann representation

$$
\langle\cO(x)\,\cO(x')\rangle=\int_0^\infty d\mu\;\rho(\mu)\,G(\sigma;\mu),\qquad\rho(\mu)\ge0, \tag{4}
$$

where $$G(\sigma;\mu)$$ is the Wightman function of a free scalar of mass $$m$$, $$\mu=\sqrt{m^2/H^2-9/4}$$ is its mass index and $$\sigma$$ the de Sitter-invariant distance. Positivity of $$\rho$$ follows from unitarity. A single free heavy field is a delta function in $$\rho$$; a conformal sector is a power law. The support is on the principal series, $$m\ge3H/2$$, so $$\mu\ge0$$ is real.
- (H2) The coupling is $$-(\alpha/\Lambda)\sqrt{-g}\,(\partial\phi)^2\cO$$, invariant under the de Sitter isometries and with $$\alpha$$ real. On the rolling background it produces two vertices, $$\mathcal L_{\rm int}=\sqrt{-g}\,[-(\alpha/\Lambda)(\partial_\mu\delta\phi)^2\cO+(2\alpha\dot{\bar\phi}/\Lambda)\,\delta\dot\phi\,\cO]$$. The inflaton has sound speed $$c_s=1$$.
- (H3) The tree diagram in which $$\cO$$ is exchanged between these two vertices dominates the bispectrum; the diagrams with $$\langle\cO\cO\cO\rangle$$ and loops are negligible.
- (H4) $$\int^\infty d\mu\,\rho(\mu)\mu^{-2}<\infty$$, so that the spectral integral of the bispectrum converges without subtraction (the single-exchange bispectrum falls as $$\mu^{-2}$$, see section 3).

The tree diagram is linear in the in-in propagators of $$\cO$$, each of which is given by the same spectral integral as (4) [[DJT]](#ref-djt), where the in-in propagators of $$\cO$$ with the four orderings $$\pm\pm$$ are the corresponding time-ordered, anti-time-ordered and Wightman functions of a free field. The bispectrum is therefore a positive superposition of single exchanges,

$$
B(k_1,k_2,k_3)=\int_0^\infty d\mu\;\rho(\mu)\,B(k_1,k_2,k_3;\mu), \tag{5}
$$

where $$B(k;\mu)$$ is the bispectrum from exchanging one free scalar of index $$\mu$$ through the same two vertices. Because each vertex carries one power of $$\alpha$$, every mass enters with weight $$\alpha^2\rho(\mu)\ge0$$. This already indicates why $$\fe$$ is negative: each single exchange is negative at the equilateral point (section 3), and the same observation shows which additions can spoil the result, namely any contribution odd in a coupling, such as one proportional to $$\alpha^3\langle\cO\cO\cO\rangle$$.

## 3. The single-exchange shape

**Seed function.** For conformally coupled external fields $$\varphi$$ with cubic coupling $$\varphi^2\sigma$$ to a scalar $$\sigma$$ of index $$\mu$$, the exchange four-point function in the limit where one external momentum vanishes is determined by a function $$\mathcal F(u)$$ of $$u=s/(k_1+k_2)$$, with $$s$$ the exchanged momentum, defined for $$H=1$$ by

$$
\frac{\mathcal F(u)}{s}=\sum_{a,b=\pm}\frac{(ia)(ib)}{2}\int_{-\infty}^0\frac{d\eta}{\eta^2}e^{ia(s/u)\eta}\int_{-\infty}^0\frac{d\eta'}{\eta'^2}e^{ib\,s\,\eta'}\,G_{ab}(s;\eta,\eta'), \tag{6}
$$

where $$G_{ab}$$ are the propagators of the free scalar built from the Bunch–Davies mode functions $$f_\pm\propto(-\eta)^{3/2}H^{(1,2)}_{i\mu}(-s\eta)$$ ([[DJT]](#ref-djt), eqs. (19)–(23)). A closed form is given in [[DJT]](#ref-djt), eqs. (28)–(30) (following Qin and Xianyu): an analytic term $$\frac{1}{\mu^2+1/4}\frac{u}{1+u}\,{}_3F_2\big(1,1,1;\tfrac32+i\mu,\tfrac32-i\mu;\tfrac{2u}{1+u}\big)$$ plus a non-analytic term proportional to $$1/\cosh\pi\mu$$ that behaves as $$u^{1/2\pm i\mu}$$ at small $$u$$. We checked it in two independent ways. It satisfies the boundary differential equation

$$
u^2(1-u^2)\mathcal F''-2u^3\mathcal F'+\big(\mu^2+\tfrac14\big)\mathcal F=\frac{u}{1+u} \tag{7}
$$

to $$10^{-17}$$ at nine values of $$(\mu,u)$$. A direct evaluation of (6), with the time integrals rotated onto the imaginary axis (nested integrals for $$++$$ and $$--$$, factorised ones for $$+-$$) and Hankel functions of order $$i\mu$$, agrees with it to a relative $$5\times10^{-4}$$ or better at $$(\mu,u)=(0.5,0.3),(1,0.5),(2,0.6),(4,0.8),(1,0.95),(0.1,0.2),(3,0.1)$$. Equation (7) alone does not fix $$\mathcal F$$: it has a one-parameter family of solutions regular at the folded point $$u=1$$, and the closed form selects the one that has the Bunch–Davies behaviour, which the direct evaluation confirms. For $$u$$ close to 1 we continue $$\mathcal F$$ from $$u=0.85$$ by integrating (7) numerically (agreement with the closed form to 9 digits at $$u=0.999$$).

**From the seed to the bispectrum.** For the vertices of (H2), consider the channel in which $$\cO$$ carries $$k_c$$ and the two other legs have $$k_a,k_b$$. The mixing vertex has the time integrand $$a^3K'(k_c,\eta)\propto-\eta^{-2}e^{ik_c\eta}/(2k_c)$$, where $$K(k,\eta)=(1-ik\eta)e^{ik\eta}/(2k^3)$$ is the bulk-to-boundary propagator of $$\delta\phi$$ and $$a=-1/\eta$$; this is the right vertex of (6) with $$s=k_c$$. The cubic vertex $$-a^2[-K'_aK'_b-(\mathbf k_a\cdot\mathbf k_b)K_aK_b]$$, with $$\mathbf k_a\cdot\mathbf k_b=(k_c^2-k_a^2-k_b^2)/2$$, is proportional to $$[k_a^2k_b^2\eta^2+(\mathbf k_a\cdot\mathbf k_b)(1-ik_{ab}\eta-k_ak_b\eta^2)]\,\eta^{-2}e^{ik_{ab}\eta}/(4k_a^3k_b^3)$$ with $$k_{ab}=k_a+k_b$$. Relative to the conformally coupled vertex $$\eta^{-2}e^{ik_{ab}\eta}$$, each power of $$\eta$$ is a derivative $$\mp i\partial_{k_{ab}}$$ on the exponent, so the vertex becomes a real differential operator acting on $$\mathcal F(s/k_{ab})/s$$. Carrying this out, and writing $$u=k_c/k_{ab}$$, the channel contributes

$$
B_c=\kappa\Big\{-\frac{2\mathcal F'+u\mathcal F''}{k_ak_bk_c\,k_{ab}^3}+\frac{\mathbf k_a\cdot\mathbf k_b}{k_a^3k_b^3k_c^2}\Big[\mathcal F+u\mathcal F'+\frac{k_ak_b}{k_c^2}u^3\big(2\mathcal F'+u\mathcal F''\big)\Big]\Big\}, \tag{8}
$$

with $$\kappa>0$$, and $$B(k;\mu)=B_1+B_2+B_3$$. This is the weight-shifting operator of [[DJT]](#ref-djt), eq. (27); we derived it independently. The first term in braces is the contribution of the $$\delta\dot\phi^2\cO$$ part of the cubic vertex ("time vertex", the part of $$(\partial\delta\phi)^2$$ with two time derivatives) and the second that of the $$(\partial_i\delta\phi)^2\cO$$ part ("space vertex"). The sign of $$\kappa$$ is fixed by the heavy-mass limit: integrating out $$\cO$$ at tree level gives $$\mathcal L_{\rm eff}=(\alpha^2/2\Lambda^2m^2)\,((\partial\phi)^2)^2$$ with a positive coefficient, whose bispectrum is negative [[DJT]](#ref-djt), eq. (7).

**Checks.** (i) As $$\mu\to\infty$$, $$\mu^2\mathcal F\to u/(1+u)$$, and (8) summed over channels gives, at four triangles and symbolically, exactly $$\tfrac12$$ times the contact shape $$(k_1k_2k_3)^{-3}e_1^{-2}\,(e_1^5-3e_1^3e_2-4e_1e_2^2+11e_1^2e_3-4e_2e_3)$$ of [[DJT]](#ref-djt), eq. (7), with $$e_1=\sum k_i$$, $$e_2=\sum_{i<j}k_ik_j$$, $$e_3=k_1k_2k_3$$, with the sign of a positive coefficient; the same shape follows from the in-in integral of $$\delta\dot\phi(\partial\delta\phi)^2$$ with our vertex signs. (ii) A direct in-in evaluation of the full diagram with massless external legs and the vertices of (H2), including the sums over all four orderings, gives for $$(\mu;k_1,k_2,k_3)=(1;1,1,1),(3;1,1,1),(1;1,1,0.4),(2;1,0.8,0.5)$$ a ratio to the right side of (8) equal to the same constant, $$-\tfrac12$$, to $$2\times10^{-4}$$. (iii) $$B(1,1,1;\mu)<0$$ at all 25 values of $$\mu$$ computed in $$10^{-3}\le\mu\le40$$ and in the limit $$\mu\to\infty$$. (iv) For isosceles triangles $$(1,1,x)$$ the maximum over $$\mu$$ of $$S$$ changes sign between $$x=0.026$$ and $$0.027$$, which is the sign window quoted in [[DJT]](#ref-djt) ("$$0.027\lesssim k_3/k_1$$"). The single-exchange shape falls as $$\mu^{-2}$$ at large $$\mu$$, which is the origin of (H4).

## 4. The cone

When (2) is applied to (5), linearity of the projection lets it pass inside the integral:

$$
(\fe,\fo)[\rho]=\int_0^\infty d\mu\;\rho(\mu)\,v(\mu),\qquad v(\mu)\equiv G^{-1}b\big(S(\cdot\,;\mu)\big). \tag{9}
$$

As $$\mu\to\infty$$, $$v(\mu)\to0$$ but $$\mu^2v(\mu)$$ tends to a non-zero vector $$v_c$$, the projection of the heavy-mass contact shape. Rescaling generators by positive factors does not change the cone, so it is generated equally well by the directions of $$v(\mu)$$ for $$\mu<\infty$$ together with the direction of $$v_c$$.

<div class="lemma" markdown="1">
**Lemma** (planar cone). Let $$\{u_t\}$$ be a family of non-zero vectors in $$\mathbb R^2$$ whose directions lie in a closed arc of opening less than $$\pi$$, and let $$\theta_-$$ and $$\theta_+$$ be the smallest and largest of their angles. Then the convex cone generated by the $$u_t$$ lies in the sector $$\{re^{i\theta}:r\ge0,\ \theta_-\le\theta\le\theta_+\}$$ and contains its interior; it contains an edge ray if and only if the corresponding extreme angle is attained.
</div>

<div class="proof" markdown="1">
*Proof.* The vectors lie in an open half-plane, on which the angle is single-valued, and a positive combination of two vectors less than $$\pi$$ apart lies angularly between them. Every positive combination thus has its angle in $$[\theta_-,\theta_+]$$. Conversely, any angle strictly between $$\theta_-$$ and $$\theta_+$$ is a positive combination of two of the vectors, one on each side of it, and an extreme angle is reached only by the vectors that have it.
</div>

<div class="theorem" markdown="1">
**Theorem 1** (cone). Assume (H1)–(H4) and (H6), and that $$\mu\mapsto S(\cdot\,;\mu)$$ is continuous from $$[0,\infty)$$ into the Hilbert space of the inner product with norm bounded by $$C(1+\mu^2)^{-1}$$. Then for every admissible $$\rho$$ the integral (5) converges, $$(\fe,\fo)[\rho]=\int\rho\,v\,d\mu$$, and the set of reachable pairs is a convex cone $$\mathcal C$$ that contains every positive combination of vectors $$v(\mu)$$ and is contained in the closed convex cone generated by $$\{v(\mu)\}_{\mu\ge0}$$ and $$v_c$$. If the directions of $$v(\mu)$$, $$\mu\in[0,\infty]$$, lie in an arc of opening less than $$\pi$$, then $$\mathcal C$$ is the sector between the extreme directions (up to unattained edge rays).
</div>

<div class="proof" markdown="1">
*Proof.* By (H4) and the norm bound, $$\int\rho\,\lVert S_\mu\rVert d\mu<\infty$$, so (5) converges in the Hilbert space (Bochner integral) and the continuous linear map $$S\mapsto G^{-1}b(S)$$ commutes with it, which is (9). A finite sum of point masses $$\rho=\sum c_i\delta(\mu-\mu_i)$$ with $$c_i\ge0$$ gives every positive combination of the $$v(\mu_i)$$. Conversely, $$\int\rho\,v\,d\mu$$ is a limit of positive combinations of values of $$v$$, so it lies in the closed convex cone generated by them. The last statement is the Lemma, applied to the curve $$v$$ extended by $$v_c/\lvert v_c\rvert$$ at $$\mu=\infty$$.
</div>

The hypothesis on the arc and on the norm are checked numerically below. The question of [[DJT]](#ref-djt), which values of the template amplitudes positivity allows, is thereby reduced to the two extreme angles of one curve. The reachable set constrains the direction of $$(\fe,\fo)$$ and not its magnitude.

**The edges.** We computed $$v(\mu)$$ for 24 values of $$\mu$$ between $$10^{-3}$$ and 40 and for $$\mu\to\infty$$, with Gauss–Legendre quadrature of $$32\times32$$ nodes on the fundamental domain (changing to $$48\times48$$ changes the angles by less than $$10^{-4}$$ degrees). The angle $$\theta(\mu)=\arg v(\mu)$$ for the proxy (3):

| $$\mu$$ | $$10^{-3}$$ | 0.1 | 0.3 | 0.5 | 1 | 1.5 | 2 | 2.4 | 3 | 4 | 10 | $$\infty$$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $$\theta$$ (degrees) | 164.047 | 164.118 | 164.677 | 165.740 | 169.800 | 173.867 | 176.096 | 176.503 | 176.097 | 175.445 | 175.005 | 174.942 |

The angle increases from its value as $$\mu\to0$$ ($$164.047^\circ$$, a quadratic minimum), reaches a single maximum $$176.503^\circ$$ at $$\mu^\ast\simeq2.41$$ and then decreases to the contact direction $$174.942^\circ$$, which lies strictly inside. All directions lie in the second quadrant, in an arc of $$12.46^\circ$$, so the Lemma applies. The lower edge is attained only by a point mass at $$m=3H/2$$ and is otherwise a limit; the upper edge by a point mass at $$\mu\simeq2.41$$. The edges depend on the inner product:

| inner product | lower edge ($$\mu\to0$$) | upper edge | contact direction |
|---|---|---|---|
| $$(\max x)^{-2}$$ (proxy (3)) | 164.05° | 176.50° | 174.94° |
| $$(\max x)^{-1}$$ | 164.06° | 176.46° | 174.93° |
| $$(\max x)^{-3}$$ | 164.03° | 176.55° | 174.95° |
| inverse triangle area | 164.58° | 176.77° | 175.66° |
| proxy (3), separate projections $$f_X=b_X/G_{XX}$$ | 168.91° | 182.32° | 180.73° |

The last row shows that the joint definition (2) is load-bearing: with separate projections the wedge would extend across the $$\fo=0$$ axis. A real CMB inner product differs from these proxies; the spread among the four weightings is $$0.6^\circ$$, and we do not estimate the shift for the Planck Fisher matrix. Within the class of hypotheses, the robust content is the sign quadrant $$\fe<0<\fo$$ and an upper edge near $$176^\circ$$ to $$177^\circ$$.

## 5. Which hypotheses carry the wedge

**Lighter states.** Unitarity allows spectral weight on the complementary series, $$0<m^2<9H^2/4$$, where $$\mu=i\nu$$ with $$0<\nu<3/2$$. The seed continues smoothly to this range. Computing the exchange shapes at imaginary $$\mu$$ for the proxy (3), the angle decreases monotonically as the mass decreases and only the lower edge moves; the upper edge is unchanged:

| $$\nu$$ | 0.3 | 0.5 | 1.0 | 1.3 | 1.49 |
|---|---|---|---|---|---|
| $$m^2/H^2$$ | 2.16 | 2.00 | 1.25 | 0.56 | 0.03 |
| lower edge | 163.39° | 162.17° | 155.38° | 147.58° | 140.82° |

(quadrature of $$32\times32$$ and $$64\times64$$ nodes agree to $$0.03^\circ$$; $$\nu=0.5001$$ and $$0.999$$ stand for the points $$\nu=0.5$$ and $$1$$ at which the closed form is singular). Near the light end the shapes are enhanced in the squeezed limit, $$S\propto(k_3/k_1)^{3/2-\nu}$$, and the edge depends strongly on the weight in that corner: at $$\nu=1.49$$ the $$(\max x)^{-1}$$ and $$(\max x)^{-3}$$ weightings give $$140.0^\circ$$ and $$141.6^\circ$$, and the inverse-area weighting is dominated by the squeezed cutoff and is not quoted. For the proxy the quadrant $$\fe<0<\fo$$ is preserved. The lower edge $$164.05^\circ$$ is therefore a consequence of the assumed gap $$m\ge3H/2$$ and not of unitarity.

**A subtraction.** If (H4) fails and the spectral integral requires one subtraction, a contact term with a coefficient of either sign appears, and the reachable set becomes the cone together with the full line through $$v_c$$. A set of generators whose consecutive angular gaps are all smaller than $$180^\circ$$ spans the plane. The generators $$176.50^\circ$$, $$354.94^\circ$$ (the negative contact direction) and $$164.05^\circ$$ leave gaps of $$178.4^\circ$$ and $$169.1^\circ$$, so every direction becomes reachable. The case arises for $$\rho$$ that grows faster than $$\mu$$ at large $$\mu$$, for example the density of a conformal operator of dimension $$\delta\ge2$$ in three spatial dimensions [[DJT]](#ref-djt).

**The three-point function of $$\cO$$.** The diagram with one $$\langle\cO\cO\cO\rangle$$ insertion scales as $$\alpha^3$$, whereas the exchange diagram scales as $$\alpha^2$$. Changing $$\alpha\to-\alpha$$ leaves the exchange term and all positivity conditions on $$\rho$$ unchanged and reverses the sign of the $$\alpha^3$$ term relative to it, and unitarity does not fix the sign of a three-point function. Once this term is included, positivity of $$\rho$$ no longer implies the wedge; which directions become reachable depends on the shape of the three-point contribution, which is not computed here.

**Boost breaking.** In the effective field theory of inflation with $$\phi\to\phi(t+\pi)$$, write $$\delta g^{00}\equiv g^{00}+1=-2\dot\pi-\dot\pi^2+(\partial_i\pi)^2/a^2$$. The most general coupling to $$\cO$$ at this order is $$c_1\,\delta g^{00}\cO+c_2(\delta g^{00})^2\cO$$. The de Sitter-invariant coupling of (H2) is linear in $$\delta g^{00}$$, since $$(\partial\phi)^2=\dot{\bar\phi}^2g^{00}$$ in unitary gauge, and has $$c_2=0$$. In general the quadratic vertex is $$c_1[-(1+\varepsilon)\dot\pi^2+(\partial_i\pi)^2]\cO$$ and the linear vertex $$-2c_1\dot\pi\cO$$, where

$$
\varepsilon\equiv-\frac{4c_2}{c_1}\tag{10}
$$

measures the boost-breaking admixture (it is unrelated to a slow-roll parameter). The exchange is then $$c_1^2\int\rho\,[S^{(s)}_\mu+(1+\varepsilon)S^{(t)}_\mu]$$, with $$S^{(t)}$$ and $$S^{(s)}$$ the time-vertex and space-vertex parts of (8). For each fixed $$\varepsilon$$ the reachable set is again a cone. The edges for several values, in the proxy (3):

| $$\varepsilon$$ | $$-1.5$$ | $$-1$$ | $$-0.35$$ | 0 | 0.35 | 1.25 | 5 | 15 | 19.75 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| lower edge | 164.62° | 164.44° | 164.19° | 164.05° | 163.90° | 163.50° | 161.35° | 142.50° | 90.03° | 85.23° |
| upper edge | 181.01° | 178.94° | 177.20° | 176.50° | 175.93° | 174.81° | 172.62° | 171.06° | 170.78° | 170.77° |

The quadrant $$\fe<0<\fo$$ is preserved exactly for $$-1.28<\varepsilon<19.75$$ (the ends are where the upper edge reaches $$180^\circ$$ and the lower edge $$90^\circ$$). The union over $$|\varepsilon|\le0.35$$ is $$163.90^\circ$$ to $$177.20^\circ$$ and over $$|\varepsilon|\le1.25$$ it is $$163.50^\circ$$ to $$179.87^\circ$$. For $$\varepsilon\ge0$$ the union over single-operator cones extends continuously down to the direction of the time vertex alone, whose directions $$\arg v^{(t)}(\mu)$$ range over $$[-7.6^\circ,169.7^\circ]$$, an opening of $$177.3^\circ$$, so $$\fe>0$$ is reachable. At the equilateral point the sign of the bispectrum is negative for all $$\mu$$ if and only if $$-4.47\lesssim\varepsilon\lesssim16.8$$; the thresholds are $$\varepsilon_c=-1-S^{(s)}/S^{(t)}$$ evaluated at the equilateral configuration, whose smallest positive value is $$16.8$$ at $$\mu\to0$$ and whose largest negative value is $$-4.47$$ at $$\mu\simeq2.6$$ (this is the critical ratio discussed in [[GMU]](#ref-gmu)).

The question is whether some principle fixes the sign of $$\varepsilon$$. Constraints that involve the vertices only through diagonal products do not. The mixing correction to the power spectrum is proportional to $$c_1^2\rho$$ and the exchange trispectrum of the $$\dot\pi^2\cO$$ vertex to $$c_1^2(1+\varepsilon)^2\rho$$; both are unchanged under $$1+\varepsilon\to-(1+\varepsilon)$$, whereas the time part of the bispectrum is proportional to $$c_1^2(1+\varepsilon)$$ and is odd. Positivity of the Källén–Lehmann density, of its matrix form for several operators (which constrains $$\lvert\rho_{12}\rvert\le\sqrt{\rho_{11}\rho_{22}}$$ and leaves the sign of a cross density free), and the propagator positivity conditions of Lee and Melville [[LM]](#ref-lm) all have this property: they hold for any real $$c_1,c_2$$. This argument covers the conditions listed and does not exclude a principle that acts on the vertex coefficients directly. Positivity of $$2\to2$$ amplitudes without boosts [[GM]](#ref-gm) is sensitive to $$c_1c_2$$, but it constrains the quartic interaction of $$\pi$$ and is not evaluated here.

There is a class in which the sign is fixed.

<div class="proposition" markdown="1">
**Proposition 1** ($$P(X)\cdot\cO$$ class). Suppose $$\cO$$ multiplies the full kinetic function of the inflaton, $$\mathcal L\supset P(X)\,\cO/M$$ with $$X=-\tfrac12(\partial\phi)^2$$ and $$M$$ a constant, and that $$P_X>0$$ and $$P_{XX}\ge0$$ on the background. Then $$\varepsilon=2X_0P_{XX}/P_X=c_s^{-2}-1\ge0$$, with equality if and only if $$c_s=1$$.
</div>

<div class="proof" markdown="1">
*Proof.* In unitary gauge $$X=X_0(1-\delta g^{00})$$ with $$X_0=\tfrac12\dot{\bar\phi}^2$$. Expanding $$P$$ to second order in $$\delta g^{00}$$ gives $$c_1=-X_0P_X/M$$ and $$c_2=\tfrac12X_0^2P_{XX}/M$$, so $$\varepsilon=-4c_2/c_1=2X_0P_{XX}/P_X$$. The sound speed of a $$P(X)$$ theory is $$c_s^2=P_X/(P_X+2X_0P_{XX})$$, hence $$\varepsilon=c_s^{-2}-1$$. The sign follows from $$P_X>0$$, required for the absence of ghosts, together with $$P_{XX}\ge0$$, which is equivalent to subluminal propagation, $$c_s\le1$$.
</div>

Within this class, (H2) reduces to the single condition $$c_s=1$$, and the wedge holds exactly there. The exchange shapes for $$c_s<1$$ have not been computed, so for $$c_s<1$$ only the sign $$\varepsilon\ge0$$ is established; the inflaton's own contact bispectrum, of the same order in $$1-c_s^2$$, would also have to be included. Outside the class, the sign is free: for a de Sitter-invariant coupling $$g(X)\cO$$ with a function $$g$$ different from the inflaton's own $$P$$, for instance $$g=\alpha X+\beta X^2$$, one finds $$\varepsilon=2X_0g_{XX}/g_X=4\beta X_0/(\alpha+2\beta X_0)$$, of either sign.

## 6. Comparison with data

The comparison uses the published Planck 2018 amplitudes from temperature and polarisation, $$\fe=-26\pm47$$ and $$\fo=-38\pm24$$ (68%, statistical) [[Planck18]](#ref-planck18), which lie at direction $$235.6^\circ$$, outside the wedge. The correlation between the two amplitudes is not quoted in the abstract; we take Gaussian errors with correlation $$r$$ as a free input and show $$r=0,\pm0.1,\pm0.3$$. With $$r=0$$, $$\chi^2=2.81$$ at the origin (2 degrees of freedom, $$1.2\sigma$$) and the minimum over the wedge of the proxy is $$\chi^2=2.68$$ at the upper edge near $$(-16.9,\,1.0)$$; for $$r=+0.1$$ the values are $$2.66$$ and $$2.62$$, for $$r=-0.3$$ they are $$3.67$$ and $$2.89$$, and for $$r=+0.3$$ the nearest point is the origin ($$2.51$$). Since the exchange amplitudes are at most of order unity ([[DJT]](#ref-djt), from perturbative unitarity), the comparison amounts to a test of consistency with zero, which the Planck amplitudes pass at about $$1\sigma$$ whether or not the wedge is imposed. The galaxy-survey amplitudes of BOSS, $$\fe=940\pm600$$, $$\fo=-170\pm170$$ [[Cabass22]](#ref-cabass22) and $$\fe=207\pm292$$, $$\fo=-68\pm73$$ [[DAmico]](#ref-damico), are defined with templates adapted to large-scale structure and are not compared with the wedge here.

If the Planck central value persisted with smaller errors, the minimum $$\chi^2$$ inside the wedge would grow as the errors shrink: for errors scaled by $$0.75$$, $$0.5$$ and $$0.45$$ (uncorrelated) the minimum over the wedge is $$4.8$$, $$10.7$$ and $$13.3$$, that is $$1.7\sigma$$, $$2.8\sigma$$ and $$3.2\sigma$$. A $$3\sigma$$ detection with $$\fe>0$$ or $$\fo<0$$, or with a direction outside the wedge, would falsify the class.

## 7. What the theorem does not exclude

The wedge is a property of a model class. Each hypothesis can fail, and the consequences are stated without choosing among them.

- A spectral density with weight at $$m<3H/2$$ (complementary series) lowers the lower edge, to $$140.8^\circ$$ for a nearly massless state in the proxy (3), with a strong dependence on the weight in the squeezed corner; the quadrant is preserved in the weightings tested.
- A spectral integral that needs one subtraction (violating (H4)) adds a contact term of either sign and the reachable set is the whole plane.
- A non-negligible $$\langle\cO\cO\cO\rangle$$ contribution (violating (H3)) adds a term odd in $$\alpha$$ whose sign unitarity does not fix.
- A boost-breaking admixture $$\varepsilon\neq0$$ moves the edges by about $$0.4^\circ$$ to $$2^\circ$$ per unit $$\varepsilon$$ for $$\lvert\varepsilon\rvert\lesssim1$$, preserves the quadrant for $$-1.28<\varepsilon<19.75$$, and for values outside that range, or for $$\varepsilon$$ of unrestricted sign (several operators), reaches all directions. In the $$P(X)\cdot\cO$$ class $$\varepsilon=c_s^{-2}-1\ge0$$.
- The inner product. The edges quoted are for the proxy (3) and three other weightings; amplitudes extracted by an experiment with a different estimator correspond to a different inner product, and separate rather than joint projection moves the wedge by about $$5^\circ$$ to $$6^\circ$$ and across $$\fo=0$$.
- Exchange at inflaton sound speed $$c_s<1$$ outside the $$P(X)\cdot\cO$$ class, loops, and sectors whose two-point function does not have the Källén–Lehmann form are not covered.

## 8. Open points

- The edges for a CMB Fisher inner product, with transfer functions and noise, are not computed on this page.
- It is not known whether any principle constrains the vertex coefficients, and hence the sign of $$\varepsilon$$, beyond the class of Proposition 1.
- The exchange shapes for $$c_s<1$$ are not computed.
- Which shapes the three-point function of a unitary hidden sector can take is not known, and with it the size of the correction to the wedge from (H3).
- The norm bound in Theorem 1 and the arc condition are verified on a grid in $$\mu$$, not proved.

## Appendix: the Planck conversion formula

Planck 2018 IX [[Planck18]](#ref-planck18), eq. (58), gives the map from the effective-field-theory parameters $$(c_s,\tilde c_3)$$ to $$(\fe,\fo)$$. With $$\beta=(1-c_s^2)/c_s^2$$ it reads $$\fe=\beta[-0.275-0.0780\,c_s^2-\tfrac23\times0.780\,\tilde c_3]$$ and $$\fo=\beta[0.0159-0.0167\,c_s^2-\tfrac23\times0.0167\,\tilde c_3]$$. Write each amplitude as $$f_X=a_X\,f^{\dot\pi(\partial\pi)^2}+b_X\,f^{\dot\pi^3}$$ with $$f^{\dot\pi(\partial\pi)^2}=-\tfrac{85}{324}\beta$$ and $$f^{\dot\pi^3}=-\tfrac{10}{243}\beta(\tilde c_3+\tfrac32c_s^2)$$, the amplitudes of the two operators ($$\dot\pi(\partial_i\pi)^2$$ and $$\dot\pi^3$$, each shape normalised to 1 at the equilateral point). Then $$b_X$$ appears twice in each row, once multiplying $$c_s^2$$ and once multiplying $$\tilde c_3$$, and the two occurrences must agree. In the equilateral row the $$c_s^2$$ term gives $$b_{\rm eq}=1.264$$ and the $$\tilde c_3$$ term gives $$12.64$$; the row is consistent only if $$0.780$$ is read as $$0.0780$$. The orthogonal row is internally consistent, with $$b_{\rm or}=+0.2705$$. Planck 2013 XXIV [[Planck13]](#ref-planck13), eq. (98), writes with the same prefactor $$\fo=\beta(0.0159-0.0167A)$$ and $$\fe=\beta(-0.275+0.0780A)$$ with $$A=-(c_s^2+\tfrac23\tilde c_3)$$; expanding gives $$\fo=\beta[0.0159+0.0167\,c_s^2+\tfrac23\times0.0167\,\tilde c_3]$$, that is $$b_{\rm or}=-0.2705$$, opposite in sign to eq. (58), and an equilateral row with $$0.0780$$ in both places.

We projected the two operator shapes ($$\dot\pi(\partial_i\pi)^2$$ from the in-in integral with our vertex conventions and $$\dot\pi^3\propto e_3/e_1^3$$) onto the templates (1) with the joint definition (2). Rows are equilateral and orthogonal, columns $$\dot\pi(\partial\pi)^2$$ and $$\dot\pi^3$$:

| | $$\dot\pi(\partial\pi)^2$$ | $$\dot\pi^3$$ |
|---|---|---|
| equilateral, projection (3) | 1.056 | 1.284 |
| equilateral, Planck (58) | 1.048 | 1.264 |
| orthogonal, projection (3) | $$-0.058$$ | $$-0.266$$ |
| orthogonal, Planck 2013 (98) expanded | $$-0.061$$ | $$-0.2705$$ |
| orthogonal, Planck (58) | $$-0.061$$ | $$+0.2705$$ |

The projections agree with the printed coefficients to 1–5%, except for the sign of the $$\dot\pi^3$$ entry of the orthogonal row, where the projection agrees with the 2013 formula; the sign is the same for all four weightings of section 1. This does not use the Planck Fisher matrix, so it is a consistency check and not a determination of the printed coefficient. The projections in the rest of this page compute $$(\fe,\fo)$$ directly from the shapes and do not use eq. (58).

## References

1. <a id="ref-djt"></a>de Rham, Jazayeri, Tolley, "Bispectrum Islands," Phys. Rev. D 112, 083531 (2025), [arXiv:2506.19198](https://arxiv.org/abs/2506.19198).
2. <a id="ref-cabass24"></a>Cabass, Philcox, Ivanov, Akitsu, Chen, Simonović, Zaldarriaga, "BOSS Constraints on Massive Particles during Inflation: The Cosmological Collider in Action," [arXiv:2404.01894](https://arxiv.org/abs/2404.01894).
3. <a id="ref-gmu"></a>Ghosh, Maity, Ullah, "Massive Exchange and the Sign of the Equilateral Bispectrum," [arXiv:2604.06145](https://arxiv.org/abs/2604.06145).
4. <a id="ref-planck18"></a>Planck Collaboration, "Planck 2018 results. IX. Constraints on primordial non-Gaussianity," [arXiv:1905.05697](https://arxiv.org/abs/1905.05697).
5. <a id="ref-planck13"></a>Planck Collaboration, "Planck 2013 Results. XXIV. Constraints on primordial non-Gaussianity," Astron. Astrophys. 571, A24 (2014), [arXiv:1303.5084](https://arxiv.org/abs/1303.5084).
6. <a id="ref-cabass22"></a>Cabass, Ivanov, Philcox, Simonović, Zaldarriaga, "Constraints on Single-Field Inflation from the BOSS Galaxy Survey," [arXiv:2201.07238](https://arxiv.org/abs/2201.07238).
7. <a id="ref-damico"></a>D'Amico, Lewandowski, Senatore, Zhang, "Limits on primordial non-Gaussianities from BOSS galaxy-clustering data," Phys. Rev. D 111, 063514, [arXiv:2201.11518](https://arxiv.org/abs/2201.11518).
8. <a id="ref-lm"></a>Lee, Melville, "Propagator positivity bounds for cosmological correlators," Phys. Rev. D 114 (2026) 023504, [arXiv:2512.20706](https://arxiv.org/abs/2512.20706).
9. <a id="ref-gm"></a>Grall, Melville, "Positivity Bounds without Boosts," Phys. Rev. D 105 (2022) L121301, [arXiv:2102.05683](https://arxiv.org/abs/2102.05683).
