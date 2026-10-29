---
title: A positivity wedge for inflationary non-Gaussianity
date: 2026-10-29
category: dark-sector
order: 8
description: Tree-level exchange of a unitary hidden sector with a de Sitter-invariant coupling to an inflaton of sound speed 1 reaches a convex cone of equilateral and orthogonal bispectrum amplitudes, which for four proxy inner products is a narrow wedge with negative equilateral and positive orthogonal amplitude.
status: "Proved: the cone statement (Theorem 1), conditional on two properties of the single-exchange shapes verified numerically and not proved. Numerical: the edge angles, for four proxy inner products and inflaton sound speed 1; no CMB Fisher metric. Proved: the sign of the boost-breaking parameter for couplings to the inflaton's own kinetic function (edges not computed there)."
excludes: "Under the hypotheses of section 2, the numerical properties of section 4 and the proxy inner products of section 1: positive equilateral amplitude from tree-level exchange of a unitary principal-series hidden sector with a de Sitter-invariant vertex and inflaton sound speed 1. Nothing is excluded for the CMB Fisher inner product, sound speed below 1, or a failed hypothesis."
---

Heavy states of a hidden sector that couple to the inflaton make the primordial bispectrum non-Gaussian when they are exchanged between inflaton lines. If the hidden sector is unitary, which pairs $$(\fe,\fo)$$ of equilateral and orthogonal template amplitudes can the exchange produce? Suppose the coupling is de Sitter invariant, the hidden spectrum consists of principal-series states, no added contact terms are needed, the hidden three-point function is negligible and the inflaton sound speed is 1 (hypotheses (H1)–(H6) of section 2). Then the reachable pairs form a convex cone whose edges come from single-state exchanges (Theorem 1). Computed with the Fergusson–Shellard weight of section 1 in place of the CMB bispectrum Fisher metric, the edges of $$\fe+i\fo$$ lie at $$164.05^\circ$$ and $$176.50^\circ$$ anticlockwise from the positive $$\fe$$ axis. Hence $$\fe<0<\fo$$ and $$0.061\,\lvert\fe\rvert\le\fo\le0.286\,\lvert\fe\rvert$$ for that weight.

Theorem 1 also uses two properties of the single-exchange shapes that are verified numerically and not proved. Three other weightings move the edges by at most $$0.6^\circ$$ relative to the Fergusson–Shellard weight, whereas projecting each template separately instead of jointly moves them by $$5^\circ$$ to $$6^\circ$$ and across $$\fo=0$$ (section 4). The CMB Fisher metric is not computed. Section 5 gives the effect of relaxing the hypotheses: light hidden states lower the lower edge to $$140.8^\circ$$ under an analytic continuation, a subtraction in the spectral integral removes the wedge, and a coupling to the inflaton's own kinetic function fixes the sign of the boost-breaking admixture (Proposition 1).

de Rham, Jazayeri and Tolley [[DJT]](#ref-djt) showed that the spectral decomposition of a unitary hidden sector confines the shape of the exchange bispectrum to a limited part of shape space, and they raised the question for template amplitudes. Cabass et al. [[Cabass24]](#ref-cabass24) constrained the exchange of spin-zero particles during inflation with BOSS and Planck data. Ghosh, Maity and Ullah [[GMU]](#ref-gmu) showed that further operators of the effective field theory of inflation make the sign of the equilateral bispectrum depend on the operator coefficients. The addition here is the reduction of the template question to the two extreme directions of one curve of single-exchange shapes, with those directions computed.

The primordial bispectrum and its shape function, the mass parameter of a field in de Sitter space and the in-in formalism for correlators are assumed known.

## 1. Templates, projection and inner product

Let $$\zeta$$ be the curvature perturbation and $$B_\zeta(k_1,k_2,k_3)$$ its bispectrum, defined by $$\langle\zeta_{\mathbf k_1}\zeta_{\mathbf k_2}\zeta_{\mathbf k_3}\rangle=(2\pi)^3\delta^3(\mathbf k_1+\mathbf k_2+\mathbf k_3)B_\zeta$$. The shape function $$S=(k_1k_2k_3)^2B_\zeta$$ is scale invariant in the cases considered here, so it is a function on triangle shapes. For a scale-invariant spectrum the two templates of Planck [[Planck18]](#ref-planck18), eqs. (1) and (2), are the equilateral and orthogonal shapes

$$
\begin{aligned}
S_{\rm eq}&=(k_1k_2k_3)^2\Big[-\sum_{3}\frac1{k_i^3k_j^3}-\frac{2}{(k_1k_2k_3)^2}+\sum_{6}\frac{1}{k_ak_b^2k_c^3}\Big],\\
S_{\rm or}&=(k_1k_2k_3)^2\Big[-3\sum_{3}\frac1{k_i^3k_j^3}-\frac{8}{(k_1k_2k_3)^2}+3\sum_{6}\frac{1}{k_ak_b^2k_c^3}\Big],
\end{aligned} \tag{1}
$$

where $$\sum_3$$ runs over the three pairs and $$\sum_6$$ over the six permutations of $$(a,b,c)$$. Both equal 1 at $$k_1=k_2=k_3$$. The sign convention is that of WMAP and Planck, $$f_{\rm NL}\propto+B_\zeta(k,k,k)$$. The wedge depends only on the ratio $$\fo/\fe$$, so the overall normalisation of $$f_{\rm NL}$$ does not enter.

Experiments fit both templates at once. Given a positive-definite inner product $$\langle\cdot,\cdot\rangle$$ on shapes, which encodes the noise of the experiment (for the CMB, the bispectrum Fisher matrix), the reported amplitudes of a shape $$S$$ are the joint least-squares coefficients

$$
f=N^{-1}b,\qquad N_{XY}=\langle S_X,S_Y\rangle,\qquad b_X=\langle S,S_X\rangle, \tag{2}
$$

with $$X,Y\in\{\rm eq,or\}$$. This is hypothesis (H6) below. The only property of (2) used in Theorem 1 is that $$f$$ is a continuous linear function of $$S$$.

The numbers below use the inner product of Fergusson and Shellard [[FS]](#ref-fs), who approximate the CMB bispectrum correlator of scale-invariant shapes by

$$
\langle A,B\rangle=\int_{V}A\,B\,\frac{dk_1\,dk_2\,dk_3}{k_1+k_2+k_3}, \tag{3}
$$

with $$V$$ the tetrapyd, the set of $$(k_1,k_2,k_3)$$ that satisfy the triangle inequalities and $$k_i\le k_{\max}$$. Write $$k_i=Kx_i$$ with $$K=k_1+k_2+k_3$$ and $$\sum x_i=1$$. The integral over $$K$$ runs up to $$k_{\max}/\max_ix_i$$, so (3) becomes $$\int d^2x\,AB\,(\max_ix_i)^{-2}$$ up to a constant. To test the sensitivity to the weight we also use the weights $$K^{-2}$$ and $$K^0$$ in place of $$K^{-1}$$ in (3), which give $$(\max x_i)^{-1}$$ and $$(\max x_i)^{-3}$$ (the second is the flat three-dimensional measure), and, as a weight that emphasises nearly degenerate triangles more strongly, $$\int d^2x\,AB/\mathrm{Area}(x)$$ on the simplex $$\sum x_i=1$$, with $$\mathrm{Area}(x)$$ the area of the triangle with sides $$x_i$$. For (3) the cosine of the angle between the two templates is $$N_{\rm eo}/\sqrt{N_{\rm ee}N_{\rm oo}}=+0.142$$, so the templates are not orthogonal and joint and separate projections differ.

As an example of (1)–(3), the equilateral-normalised shape $$27e_3/e_1^3$$ of the operator $$\dot\pi^3$$ of the effective field theory of inflation ($$e_1=\sum k_i$$, $$e_3=k_1k_2k_3$$) projects to $$(\fe,\fo)=(1.28,-0.27)$$ for (3).

## 2. Exchange of a hidden sector

The hidden sector couples to the inflaton through a scalar operator $$\cO$$ with $$\langle\cO\rangle=0$$. The setting is the decoupling limit of [[DJT]](#ref-djt): the inflaton $$\phi=\dot{\bar\phi}\,t+\delta\phi$$ evolves on a fixed de Sitter background of Hubble rate $$H=1$$, with $$\delta\phi$$ massless and $$\zeta=-H\delta\phi/\dot{\bar\phi}$$. The rolling background $$\dot{\bar\phi}$$ is the only source of de Sitter boost breaking in (H2).

The two-point function of $$\cO$$ in the Bunch–Davies state (the vacuum that matches the flat-space vacuum at early times), in the absence of the coupling, is assumed to have the de Sitter-invariant Källén–Lehmann representation

$$
\langle\cO(x)\,\cO(x')\rangle=\int d\rho(\mu)\;G(\sigma;\mu),\qquad\rho\ge0, \tag{4}
$$

where $$G(\sigma;\mu)$$ is the Wightman function of a free scalar of mass $$m$$ and $$\sigma$$ is the de Sitter-invariant distance. The index $$\mu$$ is defined by $$m^2/H^2=9/4+\mu^2$$, with real $$\mu$$ on the principal series $$m\ge3H/2$$ and $$\mu=i\nu$$, $$0<\nu<3/2$$, on the complementary series. The hypotheses are the following.

- (H1) The representation (4) holds with a non-negative measure $$\rho$$, whose positivity follows from unitarity. A single free heavy field is a point mass; for a conformal sector of dimension $$\delta$$ the density behaves as $$\mu^{2\delta-3}$$ at large $$\mu$$ [[DJT]](#ref-djt).
- (H2) The coupling is $$-(\alpha/\Lambda)\sqrt{-g}\,(\partial\phi)^2\cO$$ with real $$\alpha$$ and a mass scale $$\Lambda$$, which is invariant under the de Sitter isometries. On the rolling background it produces the vertices $$\mathcal L_{\rm int}=\sqrt{-g}\,[-(\alpha/\Lambda)(\partial\delta\phi)^2\cO+(2\alpha\dot{\bar\phi}/\Lambda)\,\delta\dot\phi\,\cO]$$. The inflaton has sound speed $$c_s=1$$. A vertex with an additional boost-breaking admixture is treated in section 5.
- (H3) The tree diagram in which $$\cO$$ is exchanged between these two vertices dominates the bispectrum; diagrams with $$\langle\cO\cO\cO\rangle$$ and loops are negligible.
- (H4) The measure $$\rho$$ satisfies $$\int(1+\mu^2)^{-1}d\rho(\mu)<\infty$$, so that the spectral integral of the bispectrum converges without subtraction. Subtraction means removing a divergent part of the integral by a local counterterm, whose finite coefficient is then free. The single-exchange bispectrum falls as $$\mu^{-2}$$ (section 3), which is the origin of the condition.
- (H5) The support of $$\rho$$ lies on the principal series, $$m\ge3H/2$$, so that $$\mu\ge0$$ is real.
- (H6) The reported amplitudes are the joint projections (2) for a positive-definite inner product on shapes.

The tree diagram is linear in the in-in propagators of $$\cO$$ with the four orderings $$\pm\pm$$, each of which is given by the same spectral integral as (4) [[DJT]](#ref-djt). The bispectrum is therefore a positive superposition of single exchanges,

$$
B(k_1,k_2,k_3)=\int d\rho(\mu)\;B(k_1,k_2,k_3;\mu), \tag{5}
$$

where $$B(k;\mu)$$ is the bispectrum from exchanging one free scalar of index $$\mu$$ through the same two vertices. Each vertex carries one power of $$\alpha$$, so every mass enters with the weight $$\alpha^2\rho\ge0$$. Section 3 finds numerically (check (iii)) that each single exchange is negative at the equilateral point, consistent with the sign of $$\fe$$. A contribution odd in the coupling, such as $$\alpha^3\langle\cO\cO\cO\rangle$$, does not enter with a positive weight (section 5).

## 3. The single-exchange shape

The bispectrum of one exchange is obtained in two steps. A seed function $$\mathcal F(u)$$ describes the exchange between conformally coupled external lines. A real differential operator then turns it into the bispectrum for massless inflaton lines, whose vertices carry time derivatives and spatial momenta.

### The seed function for conformally coupled lines

Let $$s$$ be the momentum of the exchanged scalar of index $$\mu$$, $$K$$ the total energy entering the vertex at the other end, and $$u=s/K\in(0,1]$$. The seed is defined for $$H=1$$ by

$$
\frac{\mathcal F(u)}{s}=\sum_{p,q=\pm}\frac{(ip)(iq)}{2}\int_{-\infty}^0\frac{d\eta}{\eta^2}e^{ip(s/u)\eta}\int_{-\infty}^0\frac{d\eta'}{\eta'^2}e^{iq\,s\,\eta'}\,D_{pq}(s;\eta,\eta'), \tag{6}
$$

where $$p,q$$ are the signs of the two time orderings of the in-in formalism and the contours are rotated slightly off the real axis so that each exponent decays at early times. The propagators are built from the Bunch–Davies mode function $$\chi(\eta)=\tfrac{\sqrt\pi}{2}e^{-\pi\mu/2}(-\eta)^{3/2}H^{(1)}_{i\mu}(-s\eta)$$: $$D_{-+}(\eta,\eta')=\chi(\eta)\chi^*(\eta')$$, $$D_{+-}(\eta,\eta')=\chi^*(\eta)\chi(\eta')$$, $$D_{++}=\theta(\eta-\eta')D_{-+}+\theta(\eta'-\eta)D_{+-}$$ and $$D_{--}=D_{++}^*$$ ([[DJT]](#ref-djt), eqs. (19)–(23)). The function $$\mathcal F$$ satisfies the boundary differential equation

$$
u^2(1-u^2)\mathcal F''-2u^3\mathcal F'+\big(\mu^2+\tfrac14\big)\mathcal F=\frac{u}{1+u}. \tag{7}
$$

Equation (7) has a two-parameter family of solutions, and the Bunch–Davies solution is

$$
\mathcal F(u)=\frac{1}{\mu^2+\frac14}\,\frac{u}{1+u}\;{}_3F_2\Big(1,1,1;\tfrac32+i\mu,\tfrac32-i\mu;\tfrac{2u}{1+u}\Big)+C(\mu)\,\psi_{\mu}(u)+C(-\mu)\,\psi_{-\mu}(u), \tag{8}
$$

with

$$
\psi_{\pm\mu}(u)=u^{\frac12\pm i\mu}\,{}_2F_1\Big(\tfrac14\pm\tfrac{i\mu}2,\tfrac34\pm\tfrac{i\mu}2;1\pm i\mu;u^2\Big),\qquad C(\mu)=-\frac{\pi^2\,\Gamma(1-i\mu)}{2\mu\,(\sinh\pi\mu+i)\,\Gamma(\tfrac14-\tfrac{i\mu}2)\,\Gamma(\tfrac34-\tfrac{i\mu}2)}.
$$

This is the closed form of [[DJT]](#ref-djt), eqs. (28)–(30) (following Qin and Xianyu), with the coefficients simplified. The first term of (8) satisfies (7) and the two $$\psi$$ terms satisfy the homogeneous equation. The $$\psi$$ terms are the non-analytic part, which behaves as $$u^{1/2\pm i\mu}$$ at small $$u$$. For real $$\mu$$ the last two terms are complex conjugates, so $$\mathcal F$$ is real. Each of the three terms has a logarithmic singularity at the folded configuration $$u=1$$, and the singularities cancel. The cancellation fixes one real combination of the homogeneous solutions, and the Bunch–Davies condition fixes the other. As $$\mu\to\infty$$, $$\mu^2\mathcal F\to u/(1+u)$$.

Numerical evaluation of the definition (6) with the contours rotated onto the imaginary axis agrees with (8). For complex $$\mu$$ the right side of (8) defines the analytic continuation of $$\mathcal F$$, used at imaginary $$\mu$$ in section 5.

### From the seed to the bispectrum of massless inflaton lines

Consider the channel in which $$\cO$$ carries $$k_c$$ and the two other legs carry $$k_a,k_b$$; write $$k_{ab}=k_a+k_b$$ and $$u=k_c/k_{ab}$$. The bulk-to-boundary propagator of $$\delta\phi$$ is $$\mathcal K(k,\eta)=(1-ik\eta)e^{ik\eta}/(2k^3)$$, and the scale factor is $$-1/\eta$$. The mixing vertex has the time dependence $$(-\eta)^{-3}\mathcal K'(k_c,\eta)=-\eta^{-2}e^{ik_c\eta}/(2k_c)$$, which is the right-hand vertex of (6) with $$s=k_c$$ and an extra factor $$1/(2k_c)$$. The cubic vertex $$-\eta^{-2}[-\mathcal K'_a\mathcal K'_b-(\mathbf k_a\cdot\mathbf k_b)\mathcal K_a\mathcal K_b]$$, with $$\mathbf k_a\cdot\mathbf k_b=(k_c^2-k_a^2-k_b^2)/2$$, equals $$[k_a^2k_b^2\eta^2+(\mathbf k_a\cdot\mathbf k_b)(1-ik_{ab}\eta-k_ak_b\eta^2)]\,\eta^{-2}e^{ik_{ab}\eta}/(4k_a^3k_b^3)$$.

The left-hand vertex of (6) is $$\eta^{-2}e^{ik_{ab}\eta}$$ for $$p=+$$, with $$K=k_{ab}$$. Relative to it, each power of $$\eta$$ is a derivative with respect to $$K$$, since $$\eta\,e^{iK\eta}=-i\partial_Ke^{iK\eta}$$. For $$p=-$$ the exponential is $$e^{-iK\eta}$$ and the polynomial is conjugated, and the same replacement gives the same operator. The vertex therefore acts on $$T(s,K)=\mathcal F(s/K)/s$$ as the real operator

$$
-k_a^2k_b^2\,\partial_K^2+(\mathbf k_a\cdot\mathbf k_b)\big(1-K\partial_K+k_ak_b\,\partial_K^2\big).
$$

With $$\partial_K=-(u/K)\partial_u$$ on functions of $$u=s/K$$, one has $$\partial_K^2T=u(2\mathcal F'+u\mathcal F'')/(sK^2)$$ and $$K\partial_KT=-u\mathcal F'/s$$. Collecting the numerical factors $$1/4$$ and $$1/2$$ of the two vertices, the couplings and the normalisation of $$\zeta$$ into an overall constant $$\kappa$$ that does not depend on the momenta, the channel contributes

$$
B_c=\kappa\Big\{-\frac{2\mathcal F'+u\mathcal F''}{k_ak_bk_c\,k_{ab}^3}+\frac{\mathbf k_a\cdot\mathbf k_b}{k_a^3k_b^3k_c^2}\Big[\mathcal F+u\mathcal F'+\frac{k_ak_b}{k_c^2}u^3\big(2\mathcal F'+u\mathcal F''\big)\Big]\Big\}, \tag{9}
$$

and the bispectrum of one exchange is $$B(k;\mu)=B_1+B_2+B_3$$, summed over the three channels. This is the weight-shifting operator of [[DJT]](#ref-djt), eq. (27). The first term in braces comes from the $$\delta\dot\phi^2\cO$$ part of the cubic vertex (the time vertex) and the second from the $$(\partial_i\delta\phi)^2\cO$$ part (the space vertex).

The sign of $$\kappa$$ is fixed by the heavy-mass limit. Integrating out $$\cO$$ at tree level gives $$\mathcal L_{\rm eff}=(\alpha^2/2\Lambda^2m^2)\,((\partial\phi)^2)^2$$ with a positive coefficient, and the bispectrum of this operator is negative [[DJT]](#ref-djt), eq. (7). Hence $$\kappa>0$$. Four results bear on the shape (9).

(i) As $$\mu\to\infty$$, $$\mu^2\mathcal F\to u/(1+u)$$, and (9) summed over the channels becomes exactly one half of the contact shape $$(e_1^5-3e_1^3e_2-4e_1e_2^2+11e_1^2e_3-4e_2e_3)/(e_3^3e_1^2)$$ of [[DJT]](#ref-djt), eq. (7), where $$e_2=\sum_{i<j}k_ik_j$$. A direct in-in integral of $$\delta\dot\phi(\partial\delta\phi)^2$$ with the vertex signs of (H2) gives the same shape.

(ii) The in-in diagram with massless external legs and the vertices of (H2), including all four orderings, evaluated by numerical integration for equilateral, isosceles and scalene triangles and several $$\mu$$, agrees with (9) channel by channel with one common normalisation constant.

(iii) Numerically, $$B(1,1,1;\mu)<0$$ at every $$\mu$$ computed in $$10^{-3}\le\mu\le40$$ and in the limit $$\mu\to\infty$$.

(iv) For isosceles triangles $$(1,1,x)$$ the maximum over $$\mu$$ of $$S$$ changes sign between $$x=0.026$$ and $$0.027$$, in agreement with the sign window $$0.027\lesssim k_3/k_1$$ of [[DJT]](#ref-djt).

The single-exchange shape falls as $$\mu^{-2}$$ at large $$\mu$$, as (i) shows, and this is the origin of (H4).

## 4. The cone

Applying (2) to (5), the linearity of the projection lets it pass inside the integral:

$$
(\fe,\fo)[\rho]=\int d\rho(\mu)\,v(\mu),\qquad v(\mu)\equiv N^{-1}b\big(S(\cdot\,;\mu)\big), \tag{10}
$$

where $$S(\cdot\,;\mu)=(k_1k_2k_3)^2B(\cdot\,;\mu)$$. Theorem 1 uses two properties of the family of single-exchange shapes. They are properties of the function (8) with (9), verified numerically at sampled values of $$\mu$$ for the proxy inner products and not proved.

- (N1) The map $$\mu\mapsto S(\cdot\,;\mu)$$ is continuous from $$[0,\infty)$$ into the Hilbert space of shapes of finite norm for the inner product, with $$\lVert S(\cdot\,;\mu)\rVert\le C_0(1+\mu^2)^{-1}$$, and $$\mu^2S(\cdot\,;\mu)$$ converges in that space, as $$\mu\to\infty$$, to the contact shape $$S_c$$ of check (i) of section 3. We write $$v_c$$ for the projection of $$S_c$$ and assume $$v_c\ne0$$.
- (N2) The directions of $$v(\mu)$$ for $$\mu\in[0,\infty)$$ and of $$v_c$$ lie in a closed arc of opening less than $$\pi$$.

For (3) with $$\kappa=1$$ in (9), $$\lVert S(\cdot\,;\mu)\rVert(1+\mu^2)$$ lies between $$0.27$$ and $$0.57$$ at the sampled $$\mu$$, and the arc has opening $$12.46^\circ$$ (see the table below). Since $$\mu^2v(\mu)\to v_c$$, the cone generated by the $$v(\mu)$$ has the same closure as the cone generated by them together with $$v_c$$.

<div class="lemma" markdown="1">
**Lemma** (planar cone). Let $$\{w_t\}$$ be a family of non-zero vectors in $$\mathbb R^2$$ whose directions lie in a closed arc of opening less than $$\pi$$, and let $$\theta_-$$ and $$\theta_+$$ be the infimum and supremum of their angles. Then the convex cone generated by the $$w_t$$ lies in the sector $$\{re^{i\theta}:r\ge0,\ \theta_-\le\theta\le\theta_+\}$$ and contains its interior; it contains an edge ray if and only if the corresponding extreme angle is attained by some $$w_t$$.
</div>

<div class="proof" markdown="1">
*Proof.* The vectors lie in an open half-plane, on which the angle is single-valued, and a positive combination of two vectors less than $$\pi$$ apart lies angularly between them. Every positive combination thus has its angle in $$[\theta_-,\theta_+]$$. Conversely, any angle strictly between $$\theta_-$$ and $$\theta_+$$ is a positive combination of two of the vectors, one on each side of it, and an extreme angle is reached only by the vectors that have it.
</div>

<div class="theorem" markdown="1">
**Theorem 1** (cone). Assume (H1)–(H6) and (N1), and let $$\mathcal C$$ be the set of pairs $$(\fe,\fo)[\rho]$$ obtained from all measures $$\rho$$ admitted by (H1), (H4) and (H5), with $$B$$ given by (5). Then:

(a) for each such $$\rho$$ the integral (5) converges in the Hilbert space of the inner product, and $$(\fe,\fo)[\rho]=\int d\rho\,v$$;

(b) $$\mathcal C$$ is a convex cone, it contains every positive combination of vectors $$v(\mu)$$, $$\mu\ge0$$, and it is contained in the closed convex cone generated by $$\{v(\mu)\}_{\mu\ge0}$$ and $$v_c$$;

(c) if (N2) also holds, $$\mathcal C$$ is the sector between the extreme directions of $$v(\mu)$$, $$\mu\in[0,\infty)$$, up to its two edge rays: an edge ray belongs to $$\mathcal C$$ if and only if its direction is that of $$v(\mu)$$ for some finite $$\mu$$. The extreme directions are the same if $$v_c$$ is included.
</div>

<div class="proof" markdown="1">
*Proof.* By (H4) and the norm bound in (N1), $$\int d\rho\,\lVert S_\mu\rVert<\infty$$, so (5) converges in the Hilbert space as a Bochner integral (the integral of a continuous Hilbert-space-valued function of integrable norm), and the continuous linear map $$S\mapsto N^{-1}b(S)$$ commutes with it. This is (a). A finite sum of point masses $$\rho=\sum c_i\delta(\mu-\mu_i)$$ with $$c_i\ge0$$ is admitted by (H1), (H4) and (H5) and gives every positive combination of the $$v(\mu_i)$$. Conversely, $$\int d\rho\,v$$ is a limit of positive combinations of values of $$v$$, so it lies in the closed convex cone they generate, and $$\mathcal C$$ is a cone because the admitted measures form a cone. This is (b). For (c), apply the Lemma to the family $$\{v(\mu)\}_{\mu\ge0}$$ of vectors in the arc of (N2). The direction of $$v_c$$ is the limit of the directions of $$v(\mu)$$, since $$\mu^2v(\mu)\to v_c\ne0$$, so including $$v_c$$ does not change the extreme angles. Finally, $$\int d\rho\,v$$ is non-zero and lies on an edge ray only if $$v(\mu)$$ points along that edge for $$\rho$$-almost every $$\mu$$, because the component of $$v(\mu)$$ orthogonal to the edge has one sign and vanishes only along it. An edge direction that no finite $$\mu$$ attains is therefore not reached.
</div>

The question of [[DJT]](#ref-djt), which values of the template amplitudes positivity allows, is thereby reduced to the two extreme angles of one curve. The reachable set fixes the direction of $$(\fe,\fo)$$ only, since a cone contains all positive multiples.

### Edges for the proxy inner products

We computed $$v(\mu)$$ for $$\mu$$ between $$10^{-3}$$ and 40 and for $$\mu\to\infty$$. The angle $$\theta(\mu)=\arg v(\mu)$$ for the proxy (3) is:

| $$\mu$$ | $$10^{-3}$$ | 0.1 | 0.3 | 0.5 | 1 | 1.5 | 2 | 2.4 | 3 | 4 | 10 | $$\infty$$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $$\theta$$ (degrees) | 164.047 | 164.118 | 164.677 | 165.740 | 169.800 | 173.866 | 176.096 | 176.503 | 176.097 | 175.445 | 175.005 | 174.942 |

The angle increases from its value as $$\mu\to0$$ ($$164.047^\circ$$, a quadratic minimum), reaches a single maximum $$176.503^\circ$$ at $$\mu^\ast\simeq2.39$$ and then decreases to the contact direction $$174.942^\circ$$, which lies strictly inside. All directions lie in the second quadrant, in an arc of $$12.46^\circ$$, so (N2) holds for this proxy and the Lemma applies. The lower edge is attained only by a point mass at $$m=3H/2$$ ($$\mu=0$$) and the upper edge only by a point mass at $$\mu^\ast$$. The edges depend on the inner product:

| inner product | lower edge ($$\mu\to0$$) | upper edge | contact direction |
|---|---|---|---|
| $$(\max x)^{-2}$$ (proxy (3)) | 164.05° | 176.50° | 174.94° |
| $$(\max x)^{-1}$$ | 164.06° | 176.46° | 174.93° |
| $$(\max x)^{-3}$$ | 164.03° | 176.55° | 174.95° |
| inverse triangle area | 164.63° | 176.76° | 175.66° |
| proxy (3), separate projections $$f_X=b_X/N_{XX}$$ | 168.91° | 182.32° | 180.73° |

Refining the numerical integration changes the inverse-area row by up to $$0.05^\circ$$; the other rows are stable in the last digit. With separate projections the wedge would extend across the $$\fo=0$$ axis, so the joint definition (2) matters for the sign statement. The four joint-projection weightings differ by $$0.6^\circ$$ in the lower edge and $$0.3^\circ$$ in the upper edge, and all give $$\fe<0<\fo$$ with the upper edge between $$176.4^\circ$$ and $$176.8^\circ$$.

## 5. Relaxing the hypotheses

### Hidden states below the principal series (H5)

Unitarity allows spectral weight on the complementary series $$0<m^2<9H^2/4$$, where $$\mu=i\nu$$ with $$0<\nu<3/2$$. The shapes are computed from the continuation of (8) to imaginary $$\mu$$. For $$\nu<1/2$$ the time integrals in (6) converge and the direct evaluation agrees with the continuation at $$\nu=0.3$$. For $$\nu>1/2$$ the integral over the mixing vertex diverges at late times, as $$\int d\eta\,\eta^{-1/2-\nu}$$, and the shapes are defined by the analytic continuation in $$\mu$$ alone. Whether the in-in correlator at a finite final time reduces to this continuation is not examined. The rows $$\nu=1.0,1.3,1.49$$ of the table below therefore rest on that definition, and only the row $$\nu=0.3$$ is confirmed by a direct evaluation.

With the continuation, the angle of a point mass at $$\nu$$, for the proxy (3), decreases monotonically as the mass decreases and falls below the principal-series minimum $$164.05^\circ$$. The table gives this angle, which is the lower edge when the spectral weight extends down to that mass. The upper edge is unchanged.

| $$\nu$$ | 0.3 | 0.5 | 1.0 | 1.3 | 1.49 |
|---|---|---|---|---|---|
| $$m^2/H^2$$ | 2.16 | 2.00 | 1.25 | 0.56 | 0.03 |
| lower edge | 163.39° | 162.17° | 155.38° | 147.58° | 140.82° |

Individual terms of (8) have poles at $$\nu=1/2$$ and $$\nu=1$$, so the entries for these two values were computed at $$\nu=0.5001$$ and $$0.999$$. Near the light end the exchange shapes grow in the squeezed limit $$k_3\to0$$ as $$S\propto(k_3/k_1)^{1/2-\nu}$$ (the ratio $$B/(P(k_1)P(k_3))$$ of the bispectrum to the power spectra grows as $$(k_3/k_1)^{3/2-\nu}$$), which diverges for $$\nu>1/2$$. The edge therefore depends strongly on the weight in that corner. At $$\nu=1.49$$ the weights $$(\max x)^{-1}$$ and $$(\max x)^{-3}$$ give $$140.0^\circ$$ and $$141.6^\circ$$, and the inverse-area weight is dominated by the squeezed cutoff and is not quoted. For the proxy the quadrant $$\fe<0<\fo$$ is preserved. The lower edge $$164.05^\circ$$ depends on the gap $$m\ge3H/2$$ of (H5); with unitarity alone, and under the analytic continuation, the lower edge reaches the values in the table.

### A subtraction in the spectral integral (H4)

If (H4) fails and the spectral integral requires one subtraction, the counterterm adds a contact term with the shape of $$v_c$$ and a coefficient of either sign. The reachable set is then the cone plus the full line $$\mathbb R\,v_c$$. Since $$v_c$$ lies strictly inside the sector, adding any large multiple of $$v_c$$ to a point of the plane puts it in the interior of the cone, and subtracting the same multiple returns it, so the sum is the whole plane. The case arises when $$\rho$$ grows too fast at large $$\mu$$ for (H4); for a conformal operator of dimension $$\delta$$ the density grows as $$\mu^{2\delta-3}$$ and (H4) fails for $$\delta\ge2$$ ([[DJT]](#ref-djt), section on explicit scenarios).

### The hidden three-point function (H3)

The diagram with one $$\langle\cO\cO\cO\rangle$$ insertion scales as $$\alpha^3$$, whereas the exchange diagram scales as $$\alpha^2$$. Changing $$\alpha\to-\alpha$$ leaves the exchange term and all positivity conditions on $$\rho$$ unchanged and reverses the sign of the $$\alpha^3$$ term relative to it, and unitarity does not fix the sign of a three-point function. With this term included, positivity of $$\rho$$ does not by itself imply the wedge; the directions then reachable depend on the shape of the three-point contribution, which is not computed here.

### Boost breaking in the vertex (H2)

In the effective field theory of inflation with $$\phi\to\phi(t+\pi)$$, write $$\delta g^{00}\equiv g^{00}+1=-2\dot\pi-\dot\pi^2+(\partial_i\pi)^2/a^2$$. The two lowest-order couplings to $$\cO$$ built from $$\delta g^{00}$$ are $$c_1\,\delta g^{00}\cO$$ and $$c_2(\delta g^{00})^2\cO$$. The de Sitter-invariant coupling of (H2) is linear in $$\delta g^{00}$$, since $$(\partial\phi)^2=\dot{\bar\phi}^2g^{00}$$ in unitary gauge, and has $$c_2=0$$. For general $$c_2$$ the quadratic vertex is $$c_1[-(1+\varepsilon)\dot\pi^2+(\partial_i\pi)^2]\cO$$ and the linear vertex is $$-2c_1\dot\pi\cO$$, where

$$
\varepsilon\equiv-\frac{4c_2}{c_1} \tag{11}
$$

measures the boost-breaking admixture (it is unrelated to a slow-roll parameter). The exchange is then $$c_1^2\int d\rho\,[S^{(s)}_\mu+(1+\varepsilon)S^{(t)}_\mu]$$, with $$S^{(t)}$$ and $$S^{(s)}$$ the time-vertex and space-vertex parts of (9). For each fixed $$\varepsilon$$ the reachable set is again a cone, with the inflaton at $$c_s=1$$. The edges for several values, in the proxy (3):

| $$\varepsilon$$ | $$-1.5$$ | $$-1$$ | $$-0.35$$ | 0 | 0.35 | 1.25 | 5 | 15 | 19.75 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| lower edge | 164.62° | 164.44° | 164.19° | 164.05° | 163.90° | 163.50° | 161.35° | 142.50° | 90.03° | 85.23° |
| upper edge | 181.01° | 178.96° | 177.20° | 176.50° | 175.93° | 174.82° | 172.64° | 171.07° | 170.79° | 170.78° |

The quadrant $$\fe<0<\fo$$ is preserved for $$-1.28<\varepsilon<19.75$$ (thresholds $$-1.278$$ and $$19.751$$; at the ends the upper edge reaches $$180^\circ$$ and the lower edge $$90^\circ$$). The union over $$\lvert\varepsilon\rvert\le0.35$$ is $$163.90^\circ$$ to $$177.20^\circ$$, and over $$\lvert\varepsilon\rvert\le1.25$$ it is $$163.50^\circ$$ to $$179.88^\circ$$. As $$\varepsilon\to\infty$$ the cone tends to the cone of the time vertex alone, whose directions $$\arg v^{(t)}(\mu)$$ range over $$[-7.6^\circ,169.7^\circ]$$, an opening of $$177.3^\circ$$, which contains directions with $$\fe>0$$. At the equilateral point the computed bispectrum is negative at all $$\mu$$ if and only if $$-4.47\lesssim\varepsilon\lesssim16.8$$. The thresholds are $$\varepsilon_c=-1-S^{(s)}/S^{(t)}$$ evaluated at the equilateral configuration; its smallest positive value is $$16.8$$, at $$\mu\to0$$, and its largest negative value is $$-4.47$$, at $$\mu\simeq2.6$$ (the critical ratio of [[GMU]](#ref-gmu)).

The sign of $$\varepsilon$$ is not fixed by the positivity conditions that involve the vertices only through diagonal products. The mixing correction to the power spectrum is proportional to $$c_1^2\rho$$ and the exchange trispectrum of the $$\dot\pi^2\cO$$ vertex to $$c_1^2(1+\varepsilon)^2\rho$$. Both are unchanged under $$1+\varepsilon\to-(1+\varepsilon)$$, whereas the time part of the bispectrum is proportional to $$c_1^2(1+\varepsilon)$$ and is odd. Positivity of the Källén–Lehmann density, positivity of its matrix form for several operators (which bounds $$\lvert\rho_{12}\rvert\le\sqrt{\rho_{11}\rho_{22}}$$ and leaves the sign of a cross density free), and the propagator positivity conditions of Lee and Melville [[LM]](#ref-lm), which are conditions on the hidden-sector propagator, all have this property: they hold for real $$c_1,c_2$$ of either sign. A principle acting on the vertex coefficients directly could fix the sign. Positivity of $$2\to2$$ amplitudes without boosts [[GM]](#ref-gm) applies to the quartic interaction of $$\pi$$ generated by the exchange, whose $$\dot\pi^2(\partial_i\pi)^2$$ term is odd in $$1+\varepsilon$$; its consequence for $$\varepsilon$$ is not evaluated.

The sign is fixed for one class of couplings.

<div class="proposition" markdown="1">
**Proposition 1** ($$P(X)\cdot\cO$$ class). Suppose $$\cO$$ multiplies the full kinetic function of the inflaton, $$\mathcal L\supset P(X)\,\cO/M$$ with $$X=-\tfrac12(\partial\phi)^2$$ and $$M$$ a constant, and that $$P_X>0$$ and $$P_{XX}\ge0$$ on the background. Then $$\varepsilon=2X_0P_{XX}/P_X=c_s^{-2}-1\ge0$$, with equality if and only if $$c_s=1$$.
</div>

<div class="proof" markdown="1">
*Proof.* In unitary gauge $$X=X_0(1-\delta g^{00})$$ with $$X_0=\tfrac12\dot{\bar\phi}^2$$. Expanding $$P$$ to second order in $$\delta g^{00}$$ gives $$c_1=-X_0P_X/M$$ and $$c_2=\tfrac12X_0^2P_{XX}/M$$, so $$\varepsilon=-4c_2/c_1=2X_0P_{XX}/P_X$$. The sound speed of a $$P(X)$$ theory is $$c_s^2=P_X/(P_X+2X_0P_{XX})$$, hence $$\varepsilon=c_s^{-2}-1$$. The sign follows from $$P_X>0$$, required for the absence of ghosts, together with $$P_{XX}\ge0$$, which is equivalent to subluminal propagation, $$c_s\le1$$.
</div>

Within this class, (H2) reduces to the single condition $$c_s=1$$, and the computed wedge applies only there. For $$c_s<1$$ the proposition gives only the sign $$\varepsilon>0$$. The table above does not describe that case, because it keeps the inflaton at $$c_s=1$$. The exchange shapes at $$c_s<1$$ (with the changed inflaton mode functions, and with the inflaton's own contact bispectrum, of the same order in $$1-c_s^2$$) are not computed. The proposition does show that the values $$\varepsilon<-1.28$$, for which $$\fo<0$$ becomes reachable at $$c_s=1$$, do not occur in this class. Outside the class the sign is free. For a de Sitter-invariant coupling $$g(X)\cO$$ with a function $$g$$ different from the inflaton's own $$P$$, for instance $$g=\alpha X+\beta X^2$$ with a canonical inflaton, one finds $$\varepsilon=2X_0g_{XX}/g_X=4\beta X_0/(\alpha+2\beta X_0)$$, of either sign, and the table applies.

For fixed $$\varepsilon$$ outside $$(-1.28,19.75)$$ the quadrant is lost on one side but the reachable set is still a sector (for $$\varepsilon=-1.5$$ it is $$[164.62^\circ,181.01^\circ]$$, for $$\varepsilon=20$$ it is $$[85.2^\circ,170.8^\circ]$$). All directions are reachable when contributions with $$\varepsilon$$ of both signs and large size are added, as with two hidden operators: for $$\varepsilon=30$$ and $$\varepsilon=-32$$ the two cones together positively span the plane.

## 6. Comparison with data

The comparison uses the Planck 2018 amplitudes from temperature and polarisation, $$\fe=-26\pm47$$ and $$\fo=-38\pm24$$ (68%, statistical) [[Planck18]](#ref-planck18), which lie at direction $$235.6^\circ$$, outside the wedge. The correlation $$r$$ between the two amplitudes is not quoted in the abstract, so we take Gaussian errors with $$r$$ as a free input:

| $$r$$ | $$\chi^2$$ at the origin | minimum of $$\chi^2$$ over the wedge of the proxy (3) |
|---|---|---|
| 0 | 2.81 | 2.68 at $$(-16.9,\,1.0)$$ |
| 0.1 | 2.66 | 2.62 |
| $$-0.3$$ | 3.67 | 2.89 |
| $$+0.3$$ | 2.51 | 2.51 (the origin) |

For $$r=0$$ the value at the origin is $$1.2\sigma$$ (2 degrees of freedom). Since [[DJT]](#ref-djt) argue from perturbative unitarity that the amplitudes are at most of order unity, the comparison tests consistency with zero, which the Planck amplitudes pass at about $$1\sigma$$ whether or not the wedge is imposed. The galaxy-survey amplitudes of BOSS, $$\fe=940\pm600$$, $$\fo=-170\pm170$$ [[Cabass22]](#ref-cabass22) and $$\fe=207\pm292$$, $$\fo=-68\pm73$$ [[DAmico]](#ref-damico), use templates and normalisations that differ from Planck's, and the wedge is not converted to them.

If the Planck central value persisted with smaller errors, the minimum $$\chi^2$$ inside the wedge would grow as the errors shrink. For uncorrelated errors scaled by $$0.75$$, $$0.5$$ and $$0.45$$ the minimum over the wedge is $$4.8$$, $$10.7$$ and $$13.3$$, which correspond to $$1.7\sigma$$, $$2.8\sigma$$ and $$3.2\sigma$$ (two-sided Gaussian equivalents of the 2-degree-of-freedom $$p$$-values). A $$3\sigma$$ detection with $$\fe>0$$ or $$\fo<0$$, or with a direction outside the wedge, would be inconsistent with the class of section 2 for the proxy inner product.

## 7. Cases without a computed wedge

The edges depend on the inner product, and none was computed with the transfer functions and noise of the CMB. Replacing the joint by the separate projection moves the wedge by about $$5^\circ$$ to $$6^\circ$$ and across $$\fo=0$$, so the sign of $$\fo$$ depends on the estimator. The shift for the Planck Fisher matrix is unknown.

For inflaton sound speed $$c_s<1$$ the exchange shapes are not computed, which includes the $$P(X)\cdot\cO$$ class of Proposition 1 with $$\varepsilon>0$$. Loops and hidden sectors whose two-point function lacks the Källén–Lehmann form are likewise outside the result.

The shapes that a unitary hidden sector can give to $$\langle\cO\cO\cO\rangle$$ determine the correction to the wedge from (H3). Whether a principle constrains the vertex coefficients, and with them the sign of $$\varepsilon$$, outside the class of Proposition 1 is unknown.

Properties (N1) and (N2) are verified numerically and not proved. The exchange shapes for $$1/2<\nu<3/2$$ rest on analytic continuation, and whether they equal the in-in correlator at finite final time is open.

## References

1. <a id="ref-djt"></a>de Rham, Jazayeri, Tolley, "Bispectrum Islands," Phys. Rev. D 112, 083531 (2025), [arXiv:2506.19198](https://arxiv.org/abs/2506.19198).
2. <a id="ref-cabass24"></a>Cabass, Philcox, Ivanov, Akitsu, Chen, Simonović, Zaldarriaga, "BOSS Constraints on Massive Particles during Inflation: The Cosmological Collider in Action," [arXiv:2404.01894](https://arxiv.org/abs/2404.01894).
3. <a id="ref-gmu"></a>Ghosh, Maity, Ullah, "Massive Exchange and the Sign of the Equilateral Bispectrum," [arXiv:2604.06145](https://arxiv.org/abs/2604.06145).
4. <a id="ref-planck18"></a>Planck Collaboration, "Planck 2018 results. IX. Constraints on primordial non-Gaussianity," [arXiv:1905.05697](https://arxiv.org/abs/1905.05697).
5. <a id="ref-fs"></a>Fergusson, Shellard, "The shape of primordial non-Gaussianity and the CMB bispectrum," Phys. Rev. D 80, 043510 (2009), [arXiv:0812.3413](https://arxiv.org/abs/0812.3413).
6. <a id="ref-cabass22"></a>Cabass, Ivanov, Philcox, Simonović, Zaldarriaga, "Constraints on Single-Field Inflation from the BOSS Galaxy Survey," Phys. Rev. Lett. 129, 021301 (2022), [arXiv:2201.07238](https://arxiv.org/abs/2201.07238).
7. <a id="ref-damico"></a>D'Amico, Lewandowski, Senatore, Zhang, "Limits on primordial non-Gaussianities from BOSS galaxy-clustering data," Phys. Rev. D 111, 063514 (2025), [arXiv:2201.11518](https://arxiv.org/abs/2201.11518).
8. <a id="ref-lm"></a>Lee, Melville, "Propagator positivity bounds for cosmological correlators," Phys. Rev. D 114 (2026) 023504, [arXiv:2512.20706](https://arxiv.org/abs/2512.20706).
9. <a id="ref-gm"></a>Grall, Melville, "Positivity Bounds without Boosts," Phys. Rev. D 105 (2022) L121301, [arXiv:2102.05683](https://arxiv.org/abs/2102.05683).
