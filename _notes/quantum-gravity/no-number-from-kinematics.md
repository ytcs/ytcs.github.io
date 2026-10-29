---
title: "Pure numbers from causal-set kinematics"
date: 2026-10-29
category: substrate
order: 7
description: The kernels with prescribed Mellin zeros, the horizon-molecule counts built from them, and their first and second moments on a Poisson sprinkling fix only ratios. They leave the Newton constant in units of the discreteness length, and the entropy per molecule, free. A finite rigid Lorentz-invariant configuration defined over the rationals yields only algebraic numbers, and the horizon count has no volume term that could carry a cosmological constant.
status: "Propositions proved for Poisson sprinklings; one estimate (finiteness of a variance constant) proved in outline; numerical checks in two dimensions; one comparison with the observed vacuum energy density is an order-of-magnitude estimate under a stated rule."
excludes: "Deriving the Newton constant in units of the discreteness length, an entropy unit per weighted molecule, a universal second-moment number, a transcendental constant such as pi from a rigid Lorentz-invariant configuration, or a cosmological constant from the volume degree of the horizon count, using only the mean and the variance of local counts on a Poisson sprinkling."
---

A Poisson sprinkling into spacetime has one scale, the discreteness length $$\ell_\rho=\rho^{-1/d}$$, and the counts built on it have a definite mean and variance. Do these kinematics fix a dimensionless constant of nature: the Newton constant in units of the discreteness length, the entropy attached to a horizon molecule, or the size of a cosmological constant? The answer, for the counts and the conditions stated below, is no. Every output of the kernel family of the causal-set action and of the weighted horizon counts is a ratio, a zero of a Mellin symbol, or a quantity that rescales with a free normalisation; a rigid first-moment functional carries its own free scale; the one scale-free second-moment number, the ratio of the squared mean to the variance, takes arbitrarily large values on functionals that agree in every first-moment property; a finite rigid Lorentz-invariant configuration defined by rational conditions has only algebraic invariants, so no $$\pi$$, $$2\pi$$ or $$\log2$$ appears; and the horizon count has no term proportional to a volume, so it cannot supply a cosmological constant from its mean, while its fluctuation grows with the area and not with the volume.

The free normalisation of the causal-set action is the coupling $$(\ell_\rho/\ell_p)^2$$ already present in the definition of Benincasa and Dowker [[BD]](#ref-bd), and the cutoff dependence of the area coefficient of an entropy together with the renormalisation of Newton's constant is known from entanglement entropy [[Susskind–Uglum]](#ref-su). The horizon molecules and their area law are due to [[Barton et al.]](#ref-barton); the family of kernels with extra Mellin zeros is the subject of [Boundary and corner terms of the causal-set action](/quantum-gravity/boundary-terms/), and the exact weights of the horizon count of [Horizon counts: which weights give the area law](/quantum-gravity/area-projector/). Sorkin and collaborators [[Ahmed et al.]](#ref-adgs) obtain a fluctuating cosmological constant of the observed order from the number of points of a sprinkling; the comparison here replaces that number by the horizon count. This page adds the homogeneity statement for the action and the entropy together, the classification of rigid first-moment functionals, the thinning orbit that makes the variance ratio unbounded, the algebraic ceiling for rigid configurations, and the absence of a volume degree in the horizon count with the covariance identity that fixes its dependence on the number of points.

## 1. The counts and what a pure number would be

We use two counts on a Poisson sprinkling of density $$\rho$$ (a Poisson point process with intensity $$\rho$$ times the volume, ordered by the causal relation) and write $$\ell_\rho=\rho^{-1/d}$$. The first is a *layer action* in four dimensions,

$$
S=\lambda\,\kappa\Big[b\,N+\sum_kc_kN_k\Big],\qquad\lambda=\Big(\frac{\ell_\rho}{\ell_p}\Big)^2,\quad\kappa=\frac4{\sqrt6}, \tag{1}
$$

where $$N$$ is the number of points, $$N_k$$ the number of related pairs with exactly $$k$$ points between them, and $$\ell_p^2=8\pi G$$, so that the prefactor of the counts is $$\kappa\ell_\rho^2/\ell_p^2$$ with $$\ell_\rho^2=\rho^{-1/2}$$. With the Mellin symbol $$Q(s)=\sum_kc_k(s)_k/k!$$ of the kernel $$f(v)=e^{-v}\sum_kc_kv^k/k!$$ (the symbol is defined in the two pages linked in the previous paragraph), the Benincasa–Dowker action has $$b=1$$, $$c=(-1,9,-16,8)$$ and $$Q_{\rm BD}(s)=\tfrac13(2s-1)(s-1)(2s-3)$$. The second is the *weighted horizon count* $$H_w=\sum_mw_mH_m$$, where $$H_m$$ is the number of horizon molecules of type $$m$$ (pairs straddling a horizon just below a spacelike slice, with $$m$$ further elements in the future of the lower element and the past of the slice), with symbol $$Q_w(\alpha)=\sum_mw_m(\alpha)_m/m!$$. A horizon of area $$A$$ has expected count $$\langle H_w\rangle=a_dQ_w(\alpha_A)A\rho^{(d-2)/d}$$ at leading order, with $$\alpha_A=1+2/d$$ and $$a_4=\sqrt3/10$$.

A pure number here means a dimensionless quantity determined by the construction with no input beyond the Poisson law: for instance $$G/\ell_\rho^2$$ or an entropy $$s_0$$ per unit weighted count, with the horizon entropy modelled as $$S_{\rm hor}=s_0H_w$$. Four hypotheses enter the statements below.

(H1) The substrate is a Poisson sprinkling, and the quantities considered are means and variances of the counts above.

(H2) The count $$H_w$$ contains no entropy unit: $$s_0$$ is a free constant that the sprinkling does not determine. This is the weakest hypothesis. It is motivated by the absence of a state-counting reading of signed weights ([Horizon counts: which weights give the area law](/quantum-gravity/area-projector/), section 7), and an intrinsic unit for weighted molecules that we have not foreseen would remove the conclusion of section 2.

(H3) The normalisation of the weights is a convention, which is to say that a weighting $$w$$ and $$\mu w$$ describe the same functional with different units.

(H4) For the comparison with the cosmological constant in section 6, the rule that a fluctuating count $$X$$ plays the role of $$N$$ in the fluctuating cosmological constant of [[Ahmed et al.]](#ref-adgs).

## 2. The action and the entropy: only the product of a scale and a coupling

<div class="proposition" markdown="1">
**Proposition 1** (homogeneity). Under (H1)–(H3), consider the layer action (1) with kernels that satisfy the finiteness conditions of the boundary-terms page, $$Q(\tfrac12)=Q(1)=Q(\tfrac54)=Q(\tfrac32)=0$$ and $$b=-3Q'(1)$$. Let $$\mathcal O$$ be the set of numbers that the kernel conditions and the large-density expansion of $$\langle S\rangle$$ determine: the location of the zeros of $$Q$$, the relation $$b=-3Q'(1)$$, and the ratios of the coefficients of the expansion to one another (for instance a corner value divided by the Einstein–Hilbert coefficient). Then every element of $$\mathcal O$$ is unchanged by $$(b,c_k)\to\mu(b,c_k)$$ and by $$\lambda\to k\lambda$$, for all $$\mu,k>0$$, while the Newton constant defined by the coefficient of the Einstein–Hilbert term is

$$
\frac G{\ell_\rho^2}=\frac1{8\pi\,\lambda\,\mu_Q},\qquad\mu_Q=\tfrac32Q'(\tfrac32), \tag{2}
$$

which changes under either rescaling. So $$\mathcal O$$ does not determine $$G/\ell_\rho^2$$. On the entropy side, with $$S_{\rm hor}=s_0H_w$$, the area law $$S_{\rm hor}=A/4G$$ gives $$1/(4G)=s_0a_dQ_w(\alpha_A)\ell_\rho^{-(d-2)}$$, in which only the product $$s_0Q_w(\alpha_A)$$ appears. Requiring the Newton constant of the entropy to equal that of the action gives, in $$d=4$$,

$$
s_0\,Q_w(\alpha_A)=\frac{2\pi}{a_4}\,\lambda\mu_Q=\frac{20\pi}{\sqrt3}\,\lambda\mu_Q=36.276\,\lambda\mu_Q, \tag{3}
$$

a single relation between the free quantities $$s_0Q_w(\alpha_A)$$ and $$\lambda\mu_Q$$.
</div>

The idea is that the mean of the action is linear in the kernel, so the overall size of the kernel is indistinguishable from the coupling $$\lambda$$ in front. The conditions that make the action finite are zeros and ratios of values of the symbol, and cannot fix that size.

<div class="proof" markdown="1">
*Proof.* By the Mecke equation $$\langle N\rangle=\rho\,{\rm Vol}$$ and $$\langle N_k\rangle=\rho^2\iint(\rho V_{pq})^ke^{-\rho V_{pq}}/k!$$, so $$\langle S\rangle$$ is linear in $$\lambda\cdot(b,c_k)$$. The Mellin analysis of the boundary-terms page writes each coefficient of the large-density expansion as a value of $$Q$$ or $$Q'$$ at one of the points $$\tfrac12,1,\tfrac54,\tfrac32$$ times a geometric constant. The zeros and $$b=-3Q'(1)$$ are unchanged by $$Q\to\mu Q$$; a ratio of coefficients is unchanged because numerator and denominator scale alike; and $$\lambda$$ multiplies all coefficients. For the Einstein–Hilbert term we use, without rederiving it, that its coefficient is proportional to $$Q'(\tfrac32)$$ [[ASS]](#ref-ass) [[Belenchia]](#ref-belenchia), and that for Benincasa–Dowker ($$\mu_Q=1$$) the mean tends to $$\tfrac12\int R\sqrt{-g}/\ell_p^2=\lambda\,\tfrac12\int R\sqrt{-g}/\ell_\rho^2$$. With $$\ell_p^2=8\pi G$$ the coefficient of $$\tfrac12\int R\sqrt{-g}$$ is $$1/(8\pi G)=\lambda\mu_Q/\ell_\rho^2$$, which is (2). For the entropy, $$\langle H_w\rangle=a_dQ_w(\alpha_A)A\rho^{(d-2)/d}$$ is linear in $$w$$. Equating $$1/(4G)=2\pi\lambda\mu_Q/\ell_\rho^2$$ from (2) with $$s_0a_4Q_w(\alpha_A)/\ell_\rho^2$$ gives (3).
</div>

The factor $$2\pi$$ in (3) is therefore a consistency condition between two normalisations and not an output of the kinematics. The area law $$A/4G$$, with $$G$$ the coefficient of $$\tfrac12\int R$$, holds for every $$\lambda$$ once $$s_0$$ is chosen by (3).

*Example.* Suppose one postulates $$s_0=\ln2$$ with $$Q_w(\alpha_A)=\mu_Q=1$$ (one bit per unit of weighted count). Then (3) gives $$\lambda=s_0a_4/2\pi=0.0191$$ and $$G/\ell_\rho^2=1/(4a_4s_0)=2.082$$, that is $$\ell_\rho\approx0.69\sqrt G$$. The number comes from the postulate and would be different for any other $$s_0$$.

Two routes could break the homogeneity, and both lie outside (H1)–(H3). The entropy of the short-distance fields could supply an absolute unit, as in induced-gravity pictures where the area coefficient and $$1/G$$ renormalise together [[Susskind–Uglum]](#ref-su); [Can induced gravity fix Newton's constant?](/quantum-gravity/induced-gravity-number/) treats that route. A dynamical selection of $$\lambda$$, a critical value in a path integral over causal sets, is a question about dynamics and is not addressed by the kernel family.

## 3. Rigid first-moment functionals

A weighted count is *rigid* if its symbol is fixed by conditions of the type that occur above: a normalisation $$Q(t)=1$$ at an exponent $$t$$ and zeros at the other points of a set $$\mathcal S$$ of exponents.

<div class="proposition" markdown="1">
**Proposition 2** (rigid functionals are Lagrange polynomials). Let $$\mathcal S$$ be a finite set of $$n+1$$ distinct real numbers and $$t\in\mathcal S$$. The weightings $$w$$ of degree at most $$K$$ with $$Q_w(t)=1$$ and $$Q_w(s)=0$$ for $$s\in\mathcal S\setminus\{t\}$$ form an affine space of dimension $$K-n$$ if $$K\ge n$$ and are empty if $$K<n$$. For $$K=n$$ there is exactly one, with $$Q_w=L_{t,\mathcal S}$$, the Lagrange basis polynomial of the node set $$\mathcal S$$ at $$t$$.
</div>

<div class="proof" markdown="1">
*Proof.* The map $$w\mapsto Q_w$$ is a linear bijection from weightings of degree at most $$K$$ onto polynomials of degree at most $$K$$, because $$(s)_m/m!$$ has degree exactly $$m$$. The conditions prescribe the values of $$Q$$ at $$n+1$$ distinct nodes. For $$K=n$$ polynomial interpolation gives a unique solution; for $$K>n$$ the solutions form a coset of the space of polynomials of degree at most $$K$$ vanishing at all nodes, of dimension $$K-n$$; for $$K<n$$ a nonzero polynomial of degree at most $$K$$ cannot have $$n$$ zeros, so the only candidate is $$0$$, which fails the normalisation.
</div>

For the horizon count in $$d=4$$, $$\mathcal S=\{\tfrac32,\tfrac74,2,\tfrac94,\tfrac52\}$$ and $$t=\tfrac32$$ give the integer weights $$(210,-791,1268,-928,256)$$; adding the nodes $$\tfrac{11}4$$ and $$3$$ gives the next member of the tower (see [Horizon counts: which weights give the area law](/quantum-gravity/area-projector/)).

<div class="corollary" markdown="1">
**Corollary 1** (no number from first moments). Let $$F_1,\dots,F_r$$ be rigid first-moment functionals, each identified with a physical quantity $$X_i=u_i\langle F_i\rangle$$ with a unit $$u_i$$ (H2, H3). Every condition that defines them is invariant under $$w_i\to\mu_iw_i$$, and $$\langle F_i\rangle$$ is linear in $$w_i$$. So the group $$\mathbb R_+^r$$ preserves all the conditions and moves each $$u_i$$; an invariant combination of the $$X_i$$ involves one functional only, and is a ratio of outputs of that functional, such as a corner value divided by the Einstein–Hilbert coefficient.
</div>

This extends the homogeneity of section 2 to any finite number of functionals. It excludes a procedure in which two or more rigid count functionals are used to fix each other's units.

## 4. The variance ratio

The mean of $$H_w$$ is linear in $$w$$ and the variance is quadratic, so $$\kappa_w=\langle H_w\rangle^2/(A\,{\rm Var}\,H_w)$$ is invariant under $$w\to\mu w$$ and is a pure number attached to each rigid functional. We consider a planar null horizon of area $$A$$ in flat space of dimension $$d$$ with $$\rho=1$$ (for $$\rho\neq1$$ the ratio is measured per $$\ell_\rho^{d-2}$$), write $$v(w)={\rm Var}H_w/A$$, and recall that exact area projectors are the weights whose symbol has the properties of Theorem 2 of the area-projector page. If one postulates that a variance-normalised count $$K=\mu H_w$$ with $${\rm Var}K=\langle K\rangle$$ is the horizon entropy (the modular area law $$\langle\Delta K^2\rangle=\langle K\rangle=A/4G$$ discussed in [[Verlinde–Zurek]](#ref-vz)), then $$S=\kappa_wA/\ell_\rho^{d-2}$$ and $$G/\ell_\rho^{d-2}=1/(4\kappa_w)$$ in $$d=4$$. This is a postulate, and its weakest point is that a classical sprinkling variance and a modular fluctuation are different objects. The question here is only whether $$\kappa_w$$ is a universal number.

Thinning acts on weights by $$(B_pw)(m)=\sum_k\binom mkp^k(1-p)^{m-k}w_k$$. The weights

$$
w^{(p)}=p^{\alpha_A}B_pw,\qquad\text{symbol}\quad Q^{(p)}(\alpha)=p^{\alpha_A-\alpha}Q_w(\alpha), \tag{4}
$$

(the area-projector page, Proposition 5) have the same zeros as $$Q_w$$ and the same value at the area exponent, so they are again exact area projectors with the same mean. By construction, $$H_{w^{(p)}}(\eta)=p^{\alpha_A-2}\,\mathbb E[H_w(\eta_p)\mid\eta]$$, where $$\eta_p$$ is the independent $$p$$-thinning of the sprinkling $$\eta$$.

<div class="lemma" markdown="1">
**Lemma 1** (finite variance constant). Let $$w$$ be a finite weighting of degree $$K$$ and let $$D_xH_w=H_w(\eta+\delta_x)-H_w(\eta)$$ be the add-one cost. For the planar horizon with $$\rho=1$$,

$$
c_w=\frac1A\int\mathbb E\big[(D_xH_w)^2\big]\,dx<\infty.
$$

</div>

*Outline of the proof.* Write $$Z=\max_m\lvert w_m\rvert$$. Adding a point $$x$$ changes $$H_w$$ in two ways: $$x$$ may be the lower element of new molecules, which requires $$x\in P$$ and some element of its future outside $$P$$, and $$x$$ may lie in the future of an existing lower element $$p$$ (below the slice), which changes the type of the molecules of $$p$$ from $$m$$ to $$m+1$$ and creates a new one if $$x\notin P$$. Both changes vanish unless $$p$$ (respectively $$x$$) has at most $$K+1$$ elements in its future below the slice, because $$w_m=0$$ for $$m>K$$. Hence $$\lvert D_xH_w\rvert\le Z(K+2)[\mathbf 1\{x\in P,\ \lvert F(x)\rvert\le K+1,\ F(x)\not\subset P\}+N_x]$$ with $$N_x$$ the number of elements $$p\in P$$ below $$x$$ with $$\lvert F(p)\rvert\le K+1$$ and either $$F(p)\not\subset P$$ or $$x\notin P$$. For an element at depth $$u$$ below the slice the volume of its future below the slice is $$\omega u^d/d$$, so by the Poisson distribution the probability that it has at most $$K+1$$ elements in its future is at most $$C(1+u^d)^{K+1}e^{-\omega u^d/d}$$; and the first indicator requires that the point lie within a horizontal distance $$u$$ of the horizon. Integrating over the position of $$x$$ (the transverse directions give the factor $$A$$), the first term is bounded by $$A\int u\,C(1+u^d)^{K+2}e^{-\omega u^d/d}du<\infty$$. For $$N_x$$, the Mecke equation gives $$\mathbb E N_x\le\int_{p\prec x}C(1+V(p))^{K+1}e^{-V(p)}dp$$ and $$\mathbb E[N_x(N_x-1)]\le(\int_{p\prec x}C(1+V(p))^{K+2}e^{-V(p)}dp)^2$$ (conditioning on a second element adds at most one element to a future), and these are non-zero only for $$x$$ within a bounded distance of the region in which the first term lives, with the same exponential decay in the depth. We have not written out the constants.

<div class="proposition" markdown="1">
**Proposition 3** (the variance ratio is unbounded). Assume Lemma 1. For every exact area projector $$w$$ of finite degree and every $$p\in(0,1]$$,

$$
v\big(w^{(p)}\big)\le c_w\,p^{2/d},\qquad\text{hence}\qquad\kappa_{w^{(p)}}\ge\frac{a_d^2Q_w(\alpha_A)^2}{c_w}\,p^{-2/d}\longrightarrow\infty\quad(p\to0).
$$

</div>

The idea is that averaging a count over thinnings is a Rao–Blackwell step: it keeps the mean and every first-moment property and removes variance without limit, because a thinned sprinkling is a sprinkling of larger discreteness length and the area law keeps the variance proportional to the area at that larger length.

<div class="proof" markdown="1">
*Proof.* Write $$F(\eta)=\mathbb E[H_w(\eta_p)\mid\eta]$$. Adding a point $$x$$ to $$\eta$$ changes the thinned configuration only if $$x$$ is retained, so $$D_xF=p\,\mathbb E[D_xH_w(\eta_p)\mid\eta]$$ and, by the conditional Jensen inequality, $$\mathbb E(D_xF)^2\le p^2\,\mathbb E[(D_xH_w(\eta_p))^2]$$, where $$\eta_p$$ is a Poisson process of intensity $$p$$. The Poincaré inequality for Poisson functionals, $${\rm Var}\,F\le\mathbb E\int(D_xF)^2\rho\,dx$$ [[Last]](#ref-last) (Theorem 8.3), gives $${\rm Var}\,F\le p^2\int\mathbb E_p[(D_xH_w)^2]dx$$. The planar horizon and the slice are invariant under the dilation $$x\to p^{1/d}x$$, and $$H_w$$ depends only on the causal order, so a process of intensity $$p$$ on transverse area $$A$$ is a process of intensity $$1$$ on transverse area $$Ap^{(d-2)/d}$$, with volume element scaled by $$p^{-1}$$: $$\int\mathbb E_p[(D_xH_w)^2]dx=p^{-1}\cdot Ap^{(d-2)/d}c_w=Ac_wp^{-2/d}$$. So $${\rm Var}\,F\le Ac_wp^{2-2/d}$$. Finally $$H_{w^{(p)}}=p^{\alpha_A-2}F$$ and $$2(\alpha_A-2)+2-2/d=2/d$$.
</div>

*Numerical check.* In $$d=2$$ the area is a point, $$\alpha_A=2$$, $$a_2=\tfrac13$$ and $$Q_w(\alpha)=\binom{6-2\alpha}2$$ with weights $$(15,-13,4)$$. We sprinkled the plane in the window $$-5\le t\le0$$, $$-10\le z\le10$$ (about $$100$$ points per sprinkling; a molecule at depth $$5$$ has an empty-future probability below $$10^{-10}$$) with the horizon $$t+z=0$$, for $$4.24\times10^6$$ sprinklings, and computed the numbers $$H_m$$ of molecules of each type. Their means for $$m=0,\dots,5$$ are $$0.3332,0.6665,1.0000,1.3343,1.6672,2.0005$$, against $$(m+1)/3$$. The variance of $$H_w$$ for $$w=(15,-13,4)$$ is $$304.4\pm0.3$$, so $$\kappa=\tfrac19/304.4=3.65\times10^{-4}$$. For the thinned weights (4), truncated at $$m<70$$, the mean stays $$\tfrac13$$ and the variance is $$60.34\pm0.06$$, $$28.54\pm0.03$$ and $$6.80\pm0.01$$ for $$p=0.8,0.7,0.5$$ (at $$p=0.5$$ the mean is $$0.3291\pm0.0013$$, slightly low because of the depth cut), so $$\kappa$$ grows by the factors $$5.04$$, $$10.7$$ and $$44.8$$ to $$1.84\times10^{-3}$$, $$3.89\times10^{-3}$$ and $$1.63\times10^{-2}$$. Smaller $$p$$ needs a deeper window. In four dimensions a Monte Carlo evaluation of the same quantity for the weights $$(210,-791,1268,-928,256)$$ gives $$\kappa\approx1.1\times10^{-8}$$; we quote that value without having reproduced it.

So the minimal variance over exact area projectors is not attained and the ratio $$\kappa_w$$ has no universal value. Together with Corollary 1 and Proposition 1, the first and second moments of Poisson kinematics give no pure number unless a selection rule among exact projectors is added.

## 5. Rigid Lorentz-invariant configurations: algebraic numbers only

A different source of numbers is a finite configuration that is rigid, that is fixed up to a Lorentz transformation by rational conditions. Its Lorentz-invariant quantities are then determined by the conditions, with no sprinkling involved.

<div class="proposition" markdown="1">
**Proposition 4** (algebraic ceiling). Let $$C\subset(\mathbb R^{1,3})^N$$ be a nonempty set defined by finitely many polynomial equations and inequalities with rational coefficients, and suppose that the Gram matrix $$G(v)=(\langle v_i,v_j\rangle)_{ij}$$ takes a single value $$g$$ on $$C$$ (for instance because $$C$$ is a single orbit of the Lorentz group). Then every entry of $$g$$ is a real algebraic number. The field $$\mathbb Q(g_{ij})$$ is a finite extension of $$\mathbb Q$$.
</div>

<div class="proof" markdown="1">
*Proof.* The set $$\{G(v):v\in C\}$$ is the image of a semialgebraic set under a polynomial map with rational coefficients, hence semialgebraic and defined over $$\mathbb Q$$ (Tarski–Seidenberg). It is the single point $$g$$. By Tarski transfer, a statement of the form "the semialgebraic set defined over $$\mathbb Q$$ by $$\phi$$ is nonempty" that holds over $$\mathbb R$$ holds over the field $$\mathbb R_{\rm alg}$$ of real algebraic numbers, so the set contains a point with algebraic coordinates; that point lies in $$\{g\}$$.
</div>

*Examples.* A disc on the celestial sphere is a vector $$v$$ with $$\langle v,v\rangle=1$$ in $$\mathbb R^{3,1}$$; two discs are tangent when $$\langle v,w\rangle=-1$$ and disjoint when $$\langle v,w\rangle=-I$$ with the inversive distance $$I>1$$. For a triangulation $$T$$ of the sphere, Koebe–Andreev–Thurston gives a circle packing with tangency graph $$T$$, unique up to Möbius maps, that is, up to the Lorentz group; Proposition 4 applies, and the inversive distances of non-adjacent pairs are algebraic. Three families can be computed. (i) For a stacked triangulation (repeated insertion of a vertex into a face), starting from four mutually tangent discs whose Gram matrix $$2\cdot1-J$$ (diagonal $$1$$, off-diagonal $$-1$$) has signature $$(3,1)$$, the disc tangent to $$p,q,r$$ on the side away from a fourth tangent disc $$s$$ is $$w=2(p+q+r)-s$$; one checks $$\langle w,w\rangle=1$$ and $$\langle w,p\rangle=\langle w,q\rangle=\langle w,r\rangle=-1$$, so all vectors are integer combinations of the initial basis and all inversive distances are integers. For a stacked triangulation with $$10$$ vertices and $$24$$ edges we found the distances $$7,17,31,49,97,199,287,577$$. (ii) For the $$n$$-gonal bipyramid, the ring discs have angular radius $$\pi/n$$ and the two polar discs $$\pi/2-\pi/n$$, and the inversive distance between the polar discs is $$I=1+2\tan^2(\pi/n)=(3-c)/(1+c)$$ with $$c=\cos(2\pi/n)$$, a Möbius function of $$c$$ with rational coefficients, so $$\mathbb Q(I)=\mathbb Q(\cos2\pi/n)$$; for $$n=3,\dots,8$$ the values are $$7$$, $$3$$, $$2.05573$$, $$5/3$$, $$1.46383$$, $$1.34315$$. (iii) The count of parameters is exactly rigid in $$3+1$$ dimensions: $$N$$ discs have $$3N$$ parameters, a triangulation imposes $$3N-6$$ contacts, and $$6=\dim SO(3,1)=3\chi(S^2)$$. Numerical computation for asymmetric triangulations suggests fields of degree $$2$$ to $$5$$ that are not totally real; that observation is quoted and not proved.

The proposition excludes transcendental constants as invariants of a finite rigid configuration defined over $$\mathbb Q$$: $$\pi$$, $$2\pi$$, $$\log2$$ and $$\zeta(3)$$ would need an integral (a volume or a period) or an infinite limit. It does not exclude algebraic numbers from kinematics, and a single configuration is one orbit of the Lorentz group, on which an average is not normalisable. It says nothing about the Poisson sprinkling, which is not a rigid configuration. The relation to the discrete Lorentz group results is in [Exact discrete Lorentz symmetry cannot carry propagation](/quantum-gravity/discrete-lorentz-group/) and [Lorentz-invariant point sets and the structures they cannot carry](/quantum-gravity/invariant-point-sets/).

## 6. The cosmological constant and the volume degree of the horizon count

An effective cosmological constant would be a term of the count proportional to a volume. We say that a term has *volume degree* if it is proportional to $$\ell^{d+D}$$, and *null-cone degree* if it is proportional to $$\ell^{d-1+D}$$, where $$\ell$$ is the size of the ball and $$D$$ is the dimension (as a power of inverse length) of the curvature factor multiplying the term; at $$D=0$$ these are the terms proportional to the volume of the causal diamond and to the volume of the ball. The area degree is $$\ell^{d-2+D}$$.

<div class="theorem" markdown="1">
**Theorem 1** (no volume degree). Consider the expected weighted horizon count in the conformally static class of the area-projector page, $$g=W(t)^{2/d}(-dt^2+\gamma_A)$$, for a ball $$B$$ of fixed volume, expanded as a formal power series in $$A^{-2}$$ and in the Taylor coefficients of $$W$$. For every weighting $$w$$, every term of curvature dimension $$D$$ is a multiple of $$\ell^e$$ with $$e\le d-2+D$$. In particular there is no term of volume degree and none of null-cone degree, at any order.
</div>

<div class="proof" markdown="1">
*Proof.* In the proof of Theorem 1 of the area-projector page the dependence on $$\ell$$ comes only through $${\rm cap}^{(i)}(r_0)$$ with $$i\ge1$$ and $$r_0=\ell\psi(\ell^2/A^2)$$, and the part of order $$A^{-2\nu}$$ of $${\rm cap}^{(i)}(r_0)$$ is a constant times $$\ell^{n+2\nu-i}\le\ell^{n-1+2\nu}$$. All other factors are $$\ell$$-independent. A term of order $$A^{-2\nu}$$ has $$D\ge2\nu$$, so $$e\le n-1+2\nu=d-2+2\nu\le d-2+D$$.
</div>

The ball of fixed volume and the sheet $$\partial I^-(o)$$ are those of that page. The theorem is proved for this class, for all weights and to all orders, and it uses no property of the weights. For general first-order perturbations of the metric of radial degree $$s$$ the same degree bound is reached by rim localisation: a molecule exists only if its lower element lies within its own depth of the horizon, and the elements of the molecule lie within that depth of it. The absence of the volume and null-cone degrees has been checked by exact computation in 19 cases ($$d=3,4,5$$; perturbations of radial degree up to six; $$m\le1$$); we quote this result without having reproduced it, and the degree bound $$d-2+s$$ for every perturbation is not proved here.

So the mean of the horizon count contains no term that could play the role of a cosmological constant, whatever the weights; a cosmological constant enters the count only through the curvature of the background, inside the area-degree response (the response to $$R_{uu}$$ and to the area deficit of the area-projector page), and that response matters only for balls of the size of the curvature radius.

The fluctuation of the count is a different matter. For any region $$R$$ let $$N_R$$ be the number of points in $$R$$.

<div class="proposition" markdown="1">
**Proposition 5** (covariance with the number of points). For any integrable functional $$F$$ of the sprinkling and any region $$R$$,

$$
{\rm Cov}(F,N_R)=\rho\int_R\mathbb E[D_xF]\,dx.
$$

If $$D_xF=0$$ outside $$R$$, then $${\rm Cov}(F,N_R)=\rho\,\partial_\rho\mathbb E_\rho F$$. For an exact area projector in $$d$$ dimensions, $$\mathbb E H_w\propto\rho^{(d-2)/d}$$ and so $${\rm Cov}(H_w,N_R)=\tfrac{d-2}d\,\mathbb E H_w$$, and the regression coefficient of $$H_w$$ on $$N_R$$ is $$\frac{d-2}d\,\mathbb E H_w/(\rho\lvert R\rvert)$$. In particular $$H_w$$ and $$N_R$$ are uncorrelated in $$d=2$$.
</div>

<div class="proof" markdown="1">
*Proof.* By the Mecke equation, $$\mathbb E[F\,N_R]=\mathbb E\int_RF\,\eta(dx)=\rho\int_R\mathbb E[F(\eta+\delta_x)]dx=\rho\lvert R\rvert\,\mathbb E F+\rho\int_R\mathbb E[D_xF]dx$$, and $$\mathbb E N_R=\rho\lvert R\rvert$$. For the second statement, a Poisson process of intensity $$\rho+\epsilon$$ is the sum of independent processes of intensities $$\rho$$ and $$\epsilon$$; adding the second to the first adds a point at $$x$$ with probability $$\epsilon\,dx$$ to first order, so $$\partial_\rho\mathbb E_\rho F=\int\mathbb E[D_xF]dx$$. When $$D_xF$$ vanishes outside $$R$$ this equals $$\int_R\mathbb E[D_xF]dx$$. For the horizon count the dependence region is not bounded, but the contribution of depth greater than $$U$$ is of order $$e^{-\omega U^d/d}$$ by the estimate in the proof of Lemma 1, so the identity holds up to that error for a region of depth $$U$$.
</div>

The variance of the count satisfies the upper bound $${\rm Var}\,H_w\le c_wA\rho^{(d-2)/d}$$ (Lemma 1 and the Poincaré inequality, with the $$\rho$$-scaling of Proposition 3). The Cauchy–Schwarz inequality gives $${\rm Var}\,H_w\ge{\rm Cov}(H_w,N_R)^2/{\rm Var}\,N_R=\big(\tfrac{d-2}d\mathbb E H_w\big)^2/(\rho\lvert R\rvert)$$ for a region $$R$$ of depth $$U$$ around the horizon, whose volume is of order $$AU^2$$ in units of $$\ell_\rho$$; this is a lower bound of order $$A\rho^{(d-2)/d}/U^2$$ up to constants. The fluctuation is thus proportional to the square root of the area, not of the volume.

*Estimate.* Under (H4), suppose the cosmological constant fluctuates by an amount $$\delta\Lambda\sim\Delta X/V_4$$ with $$X=H_w$$ in place of the number of points $$N$$ (for which this rule gives $$\delta\Lambda\sim N^{-1/2}$$ in Planck units, the order of the observed value). Take $$V_4\sim R_H^4$$ with the Hubble radius $$R_H=c/H_0=8.4\times10^{60}\,\ell_{\rm Pl}$$ for $$H_0=68.17$$ km s$$^{-1}$$ Mpc$$^{-1}$$, and $$\Delta X\sim\sqrt{S}$$ with $$S=\pi R_H^2$$ the horizon entropy in the units of the count. Then $$\delta\Lambda\sim R_H/R_H^4=R_H^{-3}$$ while $$\Lambda_{\rm obs}\sim R_H^{-2}$$, a ratio of order $$\ell_{\rm Pl}/R_H\approx10^{-61}$$ (taking $$\ell_\rho$$ of the order of the Planck length, before any weight-dependent factor). The cancellation between the large alternating weights of an exact projector amplifies the standard deviation of $$H_w$$ relative to $$\sqrt{\langle H_w\rangle}$$ by a weight-dependent factor, of order $$5\times10^3$$ in a four-dimensional Monte Carlo evaluation that we quote without having reproduced it, giving a ratio of order $$10^{-57}$$. The projection of the fluctuation on the direction of the number of points (Proposition 5) is smaller by a further factor of order $$S/N\sim(\ell_\rho/R_H)^2$$. In all of these the ratio to the observed value is many orders of magnitude below one, because the horizon fluctuation is area-graded. The estimate assumes the rule (H4); what is proved is Theorem 1, the absence of a volume degree in the mean, together with the area-graded bounds on the variance.

## 7. What is not excluded

The statements above are about means and variances of local counts on a Poisson sprinkling, and about finite rigid configurations defined over $$\mathbb Q$$. They do not exclude the following, each with its condition.

1. A number from an external entropy unit. If the substrate or the matter fields supply an absolute unit for the entropy of a horizon molecule, or an absolute coefficient of $$1/G$$, the product in (3) is fixed. Induced-gravity pictures are of this type, and the scheme dependence of their area coefficient is discussed in [Can induced gravity fix Newton's constant?](/quantum-gravity/induced-gravity-number/).
2. A number from dynamics. A critical value of $$\lambda$$ in a path integral over causal sets would fix $$G/\ell_\rho^2$$; the kernel family says nothing about it.
3. A number from a selection rule among exact projectors. Minimal degree gives one value of $$\kappa$$; a different rule, for example minimising the variance at fixed degree, gives a degree-dependent value that decreases with the degree in the quoted computations in two and four dimensions. Whether such a family has a positive limit is not known (Proposition 3 shows that the infimum over all exact projectors, including those of infinite support, is not attained).
4. Higher cumulants, and counts that are not local molecule counts or layer actions.
5. Algebraic numbers from kinematics. Proposition 4 allows them, and a configuration that is rigid over a number field other than $$\mathbb Q$$ has invariants in that field.
6. Volume terms outside the class of Theorem 1: metrics that are not conformally static, finite-size effects at order $$A^{-4}$$ and beyond in the perturbative degree bound, and counts that are not horizon counts.

## 8. Open points

The finiteness of the variance constant $$c_w$$ (Lemma 1) is proved in outline. Whether the minimal variance over exact projectors of degree $$K$$ tends to zero as $$K\to\infty$$, as the quoted numerical sequence in $$d=2$$ and $$d=4$$ suggests, is not proved; Proposition 3 shows only that the infimum over projectors of unrestricted support is zero. The degree bound of Theorem 1 for general radial perturbations is verified in a finite number of cases and not proved. The comparison with the observed vacuum energy density rests on the rule (H4).

## References

1. <a id="ref-bd"></a>Benincasa, Dowker, “The scalar curvature of a causal set,” Phys. Rev. Lett. 104, 181301 (2010), [arXiv:1001.2725](https://arxiv.org/abs/1001.2725).
2. <a id="ref-ass"></a>Aslanbeigi, Saravani, Sorkin, “Generalized causal set d'Alembertians,” JHEP 1406, 024 (2014), [arXiv:1403.1622](https://arxiv.org/abs/1403.1622).
3. <a id="ref-belenchia"></a>Belenchia, “Universal behaviour of generalized causal set d'Alembertians in curved spacetime,” Class. Quantum Grav. 33, 135011 (2016), [arXiv:1510.04665](https://arxiv.org/abs/1510.04665).
4. <a id="ref-barton"></a>Barton, Counsell, Dowker, Gould, Jubb, Taylor, “Horizon molecules in causal set theory,” Phys. Rev. D 100, 126008 (2019), [arXiv:1909.08620](https://arxiv.org/abs/1909.08620).
5. <a id="ref-su"></a>Susskind, Uglum, “Black hole entropy in canonical quantum gravity and superstring theory,” Phys. Rev. D 50, 2700 (1994), [arXiv:hep-th/9401070](https://arxiv.org/abs/hep-th/9401070).
6. <a id="ref-adgs"></a>Ahmed, Dodelson, Greene, Sorkin, “Everpresent Lambda,” Phys. Rev. D 69, 103523 (2004), [arXiv:astro-ph/0209274](https://arxiv.org/abs/astro-ph/0209274).
7. <a id="ref-vz"></a>Verlinde, Zurek, “Modular fluctuations from shockwave geometries,” [arXiv:2208.01059](https://arxiv.org/abs/2208.01059).
8. <a id="ref-last"></a>Last, “Stochastic analysis for Poisson processes,” [arXiv:1405.4416](https://arxiv.org/abs/1405.4416).
