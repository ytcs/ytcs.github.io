---
title: "Causal-set counts and dimensionless constants"
date: 2026-10-29
category: substrate
order: 7
description: On a Poisson sprinkling the means and variances of the causal-set action and of weighted horizon counts leave the Newton constant in units of the discreteness length undetermined, and the entropy per horizon molecule too when its unit is free; in counts with exact area-law weightings of infinite support the variance can be made arbitrarily small relative to the squared mean; and in conformally static backgrounds the mean horizon count has no volume term.
status: "Propositions 1 to 3 and 5, Lemma 1 and Corollary 1 are proved for Poisson sprinklings, Proposition 1 with the Einstein–Hilbert coefficient taken from the literature. Theorem 1 holds for conformally static backgrounds with d ≥ 3, and Proposition 4 over the rationals. Variances are Monte Carlo values, and the vacuum-energy comparison is an estimate under rule (H3)."
excludes: "An entropy unit per horizon molecule from the Poisson means, when the unit is free and the weight normalisation is a convention; a Newton constant in units of the discreteness length fixed by the layer action; a universal squared-mean-to-variance ratio over exact weightings; a volume term in the mean horizon count (conformally static backgrounds); a transcendental Gram entry of a rigid rational configuration."
---

A Poisson sprinkling has one length, the discreteness length $$\ell_\rho=\rho^{-1/d}$$. Do the means and variances of counts built on it fix a dimensionless constant? We consider the causal-set action, a sum over pairs of points weighted by the number of points between them, and the horizon count, a sum over pairs straddling a horizon weighted by the number of points in the future of the lower one. Both means are linear in the weights, so the Newton constant in units of $$\ell_\rho$$ appears only with a free coupling, and the entropy per horizon molecule only with the weight normalisation if the count carries no entropy unit of its own (Proposition 1). The squared mean divided by the variance per unit area has no upper bound over weightings that give the area law exactly (Proposition 3). In conformally static backgrounds the mean horizon count has no term proportional to a volume (Theorem 1).

The free coupling is $$\lambda=(\ell_\rho/\ell_p)^2$$ in the action of Benincasa and Dowker [[BD]](#ref-bd), and the cutoff dependence of the area coefficient of an entropy, together with the renormalisation of Newton's constant, is known from entanglement entropy [[Susskind–Uglum]](#ref-su). Horizon molecules and their area law are due to [[Barton et al.]](#ref-barton). The kernels with extra Mellin zeros are those of [Boundary and corner terms of the causal-set action](/quantum-gravity/boundary-terms/) and the exact weights those of [Horizon counts: which weights give the area law](/quantum-gravity/area-projector/). Ahmed, Dodelson, Greene and Sorkin [[Ahmed et al.]](#ref-adgs) obtain a fluctuating cosmological constant of the observed order from the fluctuation of the number of points in a sprinkling, and section 6 asks what changes when that number is replaced by the horizon count. The result not contained in these references is the unbounded variance ratio, Proposition 3 with the finite variance constant of Lemma 1. Section 5 concerns a different source of numbers: a finite rigid configuration defined over the rationals has algebraic Gram entries (Proposition 4), which leaves the angles and rapidities between its vectors unconstrained.

## 1. The counts

A *Poisson sprinkling* of density $$\rho$$ into a $$d$$-dimensional spacetime is a Poisson point process with intensity $$\rho$$ times the volume, ordered by the causal relation $$\prec$$. The multivariate Mecke equation [[Last]](#ref-last) says that the expectation of a sum over $$k$$-tuples of distinct points equals the integral over the positions of the tuple of the expectation, with the process in the integrand replaced by the process plus the tuple. All statements concern means and variances of the counts below.

### The layer action

In four dimensions let $$N$$ be the number of points and $$N_k$$ the number of pairs $$\zeta\prec q$$ with exactly $$k$$ points between them. A layer action is

$$
S=\lambda\,\kappa\Big[b\,N+\sum_kc_kN_k\Big],\qquad\lambda=\Big(\frac{\ell_\rho}{\ell_p}\Big)^2,\quad\kappa=\frac4{\sqrt6},\quad\ell_p^2=8\pi G, \tag{1}
$$

with real $$b$$ and $$c_k$$. The kernel $$f(v)=e^{-v}\sum_kc_kv^k/k!$$ has the Mellin symbol $$Q(s)=\sum_kc_k(s)_k/k!$$, where $$(s)_k=s(s+1)\cdots(s+k-1)$$ and $$\Gamma(s)Q(s)=\int_0^\infty f(v)v^{s-1}dv$$. The Benincasa–Dowker action has $$b=1$$, $$c=(-1,9,-16,8)$$ and $$Q_{\rm BD}(s)=\tfrac13(2s-1)(s-1)(2s-3)$$. These are the negatives of the weights $$-1$$ on the point itself and $$(1,-9,16,-8)$$ on the layers of the Benincasa–Dowker d'Alembertian, which the pages on [mixed-Poisson and signed substrates](/quantum-gravity/signed-substrates/) and on [kernel-scale kinematics of a single realisation](/quantum-gravity/per-realisation-kinematics/) use, and which are those of the Remark on the causal-set d'Alembertian in [horizon counts and the area law](/quantum-gravity/area-projector/). For the Benincasa–Dowker weights their Mellin transform, which contains the factor $$\Gamma(s)$$, equals $$-\Gamma(s)Q_{\rm BD}(s)$$.

The mean of $$S$$ tends to a multiple of the Einstein–Hilbert action when $$Q(\tfrac12)=Q(1)=Q(\tfrac32)=0$$ and $$b=-3Q'(1)$$; these are the conditions (K1)–(K3) of [boundary and corner terms](/quantum-gravity/boundary-terms/). That page also imposes $$Q(\tfrac54)=0$$ to remove a divergence at timelike boundaries. The Benincasa–Dowker symbol has $$Q_{\rm BD}(\tfrac54)=-\tfrac1{16}$$, and the extra zero is not used below.

### Horizon molecules

Let $$\Sigma$$ be a spacelike slice, $$B\subset\Sigma$$ a geodesic ball, $$o$$ the top tip of the domain of dependence of $$B$$, and $$n=d-1$$. The horizon is the part of $$\partial I^-(o)$$ below $$\Sigma$$, and $$P=I^-(o)\cap I^-(\Sigma)$$ is the region inside it. For a point $$\zeta\in P$$ let $$F(\zeta)$$ be the set of sprinkled points in the causal future of $$\zeta$$ and the past of $$\Sigma$$. A *horizon molecule* is a pair $$\zeta\prec q$$ with $$\zeta\in P$$ and $$q\in F(\zeta)\setminus P$$, and its *type* is $$m=\lvert F(\zeta)\rvert-1$$, the number of points of $$F(\zeta)$$ other than $$q$$, wherever they lie. A molecule of type $$0$$ is a link across the horizon. In two dimensions, with $$\Sigma=\{t=0\}$$ and the horizon along $$t+z=0$$, a molecule is a related pair with the lower point under the null line $$t+z=0$$ and the upper point above it, both below $$t=0$$, and its type counts the further points in the future of the lower point below $$t=0$$. With weights $$w_0,\dots,w_K$$ the weighted count is $$H_w=\sum_mw_mH_m$$, $$H_m$$ being the number of molecules of type $$m$$, and the Mellin symbol of the weighting is $$Q_w(\alpha)=\sum_mw_m(\alpha)_m/m!$$.

The expected count depends on the weights only through $$Q_w$$. In a conformally static background the term of the expansion in the curvature and in the size of the ball that carries the density $$\rho^{2-\beta}$$ is multiplied by $$Q_w(\beta)$$. The leading flat-space term sits at the *area exponent* $$\alpha_A=1+2/d$$, and for a flat ball of radius $$\ell$$

$$
\langle H_w\rangle=a_d\,Q_w(\alpha_A)\,\lvert\partial B\rvert+\sum_{j=1}^{d-2}C_j\,Q_w\big(\alpha_A+\tfrac jd\big)\,\ell^{\,d-2-j},\qquad a_d=\frac{\omega}{d(d+1)}\Big(\frac d\omega\Big)^{\alpha_A}\Gamma(\alpha_A), \tag{2}
$$

with constants $$C_j$$ and $$\omega$$ the volume of the unit $$n$$-ball; $$a_4=\sqrt3/10$$ and $$a_2=1/3$$. In $$d=2$$ the formula is the count per horizon point and $$\lvert\partial B\rvert$$ is replaced by $$1$$. A weighting with $$Q_w(\alpha_A)\neq0$$ and $$Q_w(\alpha_A+j/d)=0$$ for $$j=1,\dots,d$$ is an *exact weighting*; it may have finite support or, like the thinned weights (5) below, infinite support with convergent sums. For it the flat count is exactly $$a_dQ_w(\alpha_A)\lvert\partial B\rvert$$ at every ball size, and the first-order response to uniform curvature has the form of the area response. The exact weighting of lowest finite degree is unique up to normalisation, with $$Q_d(\alpha)=\binom{2d+2-d\alpha}{d}$$, and in four dimensions its weights are $$(210,-791,1268,-928,256)$$. These statements are Proposition 1 and Theorem 2 of [horizon counts and the area law](/quantum-gravity/area-projector/), proved there for $$d\ge3$$; the converse, that the zeros are necessary, is proved there for $$3\le d\le12$$. In $$d=2$$ the same formulas are used below as definitions.

## 2. The action and the entropy: only a product is fixed

Two hypotheses enter this section. The first is (H1): the count $$H_w$$ contains no entropy unit, so that the constant $$s_0$$ in the horizon entropy $$S_{\rm hor}=s_0H_w$$ is not determined by the sprinkling. An intrinsic unit for weighted molecules would remove the conclusion; we do not address whether a signed count can be an entropy. The second is (H2): the normalisation of a weighting is a convention, so that $$w$$ and $$\mu w$$ with $$\mu>0$$ describe the same functional in different units.

<div class="proposition" markdown="1">
**Proposition 1** (homogeneity). Consider the layer action (1) with a kernel satisfying (K1)–(K3), and put $$\theta_Q=\tfrac32Q'(\tfrac32)$$, so that $$\theta_Q=1$$ for Benincasa–Dowker.

(a) The conditions (K1)–(K3), and the ratio of any two coefficients of the large-density expansion of $$\langle S\rangle$$ (for instance a corner value divided by the Einstein–Hilbert coefficient), are unchanged by $$(b,c_k)\to\mu(b,c_k)$$ and by $$\lambda\to\sigma\lambda$$ for $$\mu,\sigma>0$$.

(b) The coefficient of $$\tfrac12\int R\sqrt{-g}$$ in the limit of $$\langle S\rangle$$ is $$1/(8\pi G)$$ and

$$
\frac G{\ell_\rho^2}=\frac1{8\pi\,\lambda\,\theta_Q}, \tag{3}
$$

which changes by $$\mu^{-1}$$ under the first rescaling and by $$\sigma^{-1}$$ under the second. The conditions in (a) therefore do not determine $$G/\ell_\rho^2$$.

(c) Take $$d=4$$. The area law $$S_{\rm hor}=\lvert\partial B\rvert/4G$$ for $$S_{\rm hor}=s_0H_w$$ gives $$1/(4G)=s_0a_4Q_w(\alpha_A)\ell_\rho^{-2}$$, in which only the product $$s_0Q_w(\alpha_A)$$ appears. Requiring this $$G$$ to equal the one in (3) gives

$$
s_0\,Q_w(\alpha_A)=\frac{2\pi}{a_4}\,\lambda\theta_Q=\frac{20\pi}{\sqrt3}\,\lambda\theta_Q=36.276\,\lambda\theta_Q, \tag{4}
$$

a single relation between the free quantities $$s_0Q_w(\alpha_A)$$ and $$\lambda\theta_Q$$.
</div>

The proposition is bookkeeping. The mean of the action is linear in $$(b,c_k)$$ and each condition of (K1)–(K3) is either a zero of $$Q$$ or a ratio ($$b=-3Q'(1)$$), so a common factor drops out of all of them while the overall size of the kernel remains free.

<div class="proof" markdown="1">
*Proof.* (a) By the Mecke equation $$\langle N\rangle=\rho\,{\rm Vol}$$ and $$\langle N_k\rangle=\rho^2\iint(\rho V_{\zeta q})^ke^{-\rho V_{\zeta q}}/k!$$, so $$\langle S\rangle$$ is linear in $$\lambda\cdot(b,c_k)$$. Zeros of $$Q$$ and the relation $$b=-3Q'(1)$$ are unchanged by $$Q\to\mu Q$$, and a ratio of coefficients is unchanged because numerator and denominator scale alike. (b) Aslanbeigi, Saravani and Sorkin [[ASS]](#ref-ass) show that for kernels with (K1)–(K3) the coefficient of the Einstein–Hilbert term is proportional to $$Q'(\tfrac32)$$, and Belenchia [[Belenchia]](#ref-belenchia) extends the curvature limit to the whole family. For Benincasa–Dowker the mean tends to $$\tfrac12\int R\sqrt{-g}/\ell_p^2=\lambda\,\tfrac12\int R\sqrt{-g}/\ell_\rho^2$$ [[BD]](#ref-bd), and $$Q'_{\rm BD}(\tfrac32)=\tfrac23$$, so the proportionality constant is fixed by $$\theta_Q=1$$ there. With $$\ell_p^2=8\pi G$$ the coefficient of $$\tfrac12\int R\sqrt{-g}$$ is $$1/(8\pi G)=\lambda\theta_Q/\ell_\rho^2$$, which is (3). (c) By (2), $$\langle H_w\rangle=a_dQ_w(\alpha_A)\lvert\partial B\rvert\rho^{(d-2)/d}$$ at leading order, linear in $$w$$, which gives the first statement. Equating $$1/(4G)=2\pi\lambda\theta_Q/\ell_\rho^2$$ from (3) with $$s_0a_4Q_w(\alpha_A)/\ell_\rho^2$$ gives (4).
</div>

The factor $$2\pi$$ in (4) comes from the normalisations $$1/(8\pi G)$$ of the Einstein–Hilbert coefficient and $$1/(4G)$$ of the area law, and it relates the two free quantities. The area law with $$G$$ the coefficient of $$\tfrac12\int R$$ holds for every $$\lambda$$ once $$s_0$$ is chosen by (4).

As an example, suppose one postulates $$s_0=\ln2$$ with $$Q_w(\alpha_A)=\theta_Q=1$$ (one bit per unit of weighted count). Then (4) gives $$\lambda=s_0a_4/2\pi=0.0191$$ and $$G/\ell_\rho^2=1/(4a_4s_0)=2.082$$, that is $$\ell_\rho\approx0.69\sqrt G$$. Any other postulated $$s_0$$ gives another value.

Two routes lie outside (H1) and (H2). The entropy of the short-distance fields could supply an absolute unit, as in induced-gravity pictures where the area coefficient and $$1/G$$ renormalise together [[Susskind–Uglum]](#ref-su); [the induced Newton constant](/quantum-gravity/induced-gravity-number/) discusses their scheme dependence. A dynamical selection of $$\lambda$$, for example a critical value in a path integral over causal sets, is a question about dynamics that the kernel family does not address.

## 3. Weightings fixed by interpolation conditions

The exact weights of section 1 are fixed by a normalisation and a list of zeros. We call a weighting *interpolating* if its symbol is fixed by conditions of this type: $$Q(t)=1$$ at one exponent $$t$$ and zeros at the other points of a set $$\mathcal S$$ of exponents.

<div class="proposition" markdown="1">
**Proposition 2** (interpolating weightings are Lagrange polynomials). Let $$\mathcal S$$ be a finite set of $$J+1$$ distinct real numbers and $$t\in\mathcal S$$. The weightings $$w$$ of degree at most $$K$$ with $$Q_w(t)=1$$ and $$Q_w(s)=0$$ for $$s\in\mathcal S\setminus\{t\}$$ form an affine space of dimension $$K-J$$ if $$K\ge J$$ and are empty if $$K<J$$. For $$K=J$$ there is exactly one, with $$Q_w=\Pi_{t,\mathcal S}$$, the Lagrange basis polynomial of the node set $$\mathcal S$$ at $$t$$.
</div>

<div class="proof" markdown="1">
*Proof.* The map $$w\mapsto Q_w$$ is a linear bijection from weightings of degree at most $$K$$ onto polynomials of degree at most $$K$$, because $$(s)_m/m!$$ has degree exactly $$m$$. The conditions prescribe the values of $$Q$$ at $$J+1$$ distinct nodes. For $$K=J$$ polynomial interpolation gives a unique solution. For $$K>J$$ the solutions form a coset of the space of polynomials of degree at most $$K$$ vanishing at all nodes, of dimension $$K-J$$. For $$K<J$$ a nonzero polynomial of degree at most $$K$$ cannot have $$J$$ zeros, so the only candidate is $$0$$, which fails the normalisation.
</div>

For the horizon count in $$d=4$$, $$\mathcal S=\{\tfrac32,\tfrac74,2,\tfrac94,\tfrac52\}$$ and $$t=\tfrac32$$ give the integer weights $$(210,-791,1268,-928,256)$$ of section 1. Adding the nodes $$\tfrac{11}4$$ and $$3$$ gives the next member of the tower (see [horizon counts and the area law](/quantum-gravity/area-projector/)).

<div class="corollary" markdown="1">
**Corollary 1** (no relation between interpolating functionals from first moments). Let $$\Psi_1,\Psi_2,\dots$$ be finitely many first-moment functionals of interpolating weightings, each identified with a physical quantity $$X_i=u_i\langle\Psi_i\rangle$$ with a unit $$u_i$$. By (H2) the normalisations $$Q_{w_i}(t_i)=1$$ are conventions. The zero conditions that remain are invariant under $$w_i\to\mu_iw_i$$ with independent $$\mu_i>0$$, and $$\langle\Psi_i\rangle$$ is linear in $$w_i$$. The units $$u_i$$ are therefore independent free parameters, and no relation among the $$X_i$$ (for instance a value of $$X_1/X_2$$) follows from the conditions. An invariant combination involves a single functional and is a ratio of outputs of that functional, such as a corner value divided by the Einstein–Hilbert coefficient.
</div>

This extends the homogeneity of section 2 to any finite number of functionals. Under (H2) it excludes a procedure in which two or more interpolating count functionals fix each other's units.

## 4. The variance ratio

The mean of $$H_w$$ is linear in $$w$$ and the variance is quadratic, so the ratio of the squared mean to the variance is invariant under $$w\to\mu w$$ and is a pure number attached to each exact weighting. Thinning shows that it has no upper bound.

### The planar horizon

For the second moment we use the flat geometry that a large ball presents near its boundary: the horizon $$t+z=0$$ in Minkowski space of dimension $$d$$, with $$\Sigma=\{t=0\}$$, $$P=\{t<0,\ t+z<0\}$$, transverse coordinates $$y\in\mathbb R^{d-2}$$, and $$\rho=1$$ (for $$\rho\neq1$$ lengths are measured in $$\ell_\rho$$). For a box $$\Omega$$ of transverse area $$\mathcal A$$ let $$H_w^{\Omega}$$ be the weighted count of the molecules whose lower point has $$y\in\Omega$$. In $$d=2$$ there is no transverse direction, $$\mathcal A=1$$, and $$H_w^\Omega$$ is the count for the single horizon point. The mean is $$\langle H_w^\Omega\rangle=a_dQ_w(\alpha_A)\mathcal A$$ for any weighting, being the planar limit of the leading term in (2). We write $$v(w)=\limsup_{\mathcal A\to\infty}{\rm Var}\,H_w^\Omega/\mathcal A$$ (in $$d=2$$, $$v(w)={\rm Var}\,H_w^\Omega$$) and

$$
r_w=\frac{a_d^2Q_w(\alpha_A)^2}{v(w)},
$$

which is unchanged by $$w\to\mu w$$. If one postulates that a count $$\mu H_w$$ with variance equal to its mean is the horizon entropy, the modular area law $$\langle\Delta K_{\rm mod}^2\rangle=\langle K_{\rm mod}\rangle=\mathcal A/4G$$ discussed in [[Verlinde–Zurek]](#ref-vz), then $$\mu=a_dQ_w(\alpha_A)/v(w)$$, the mean of $$\mu H_w$$ is $$r_w\mathcal A$$, and $$G/\ell_\rho^{d-2}=1/(4r_w)$$. This is a postulate, and a classical sprinkling variance and a modular fluctuation are different objects. The question here is only whether $$r_w$$ is a universal number.

### Thinning

Let $$\eta_p$$ be the independent $$p$$-thinning of a sprinkling $$\eta$$, in which each point is kept with probability $$p$$. A molecule survives with probability $$p^2$$, and each of the $$m$$ further points of its type is kept independently, so that $$\mathbb E[H_w(\eta_p)\mid\eta]=p^2\sum_{\rm molecules}(B_pw)(m)$$ with $$(B_pw)(m)=\sum_k\binom mkp^k(1-p)^{m-k}w_k$$. The weights

$$
w^{(p)}=p^{\alpha_A}B_pw,\qquad\text{symbol}\quad Q^{(p)}(\alpha)=p^{\alpha_A-\alpha}Q_w(\alpha), \tag{5}
$$

(Proposition 5 of [horizon counts and the area law](/quantum-gravity/area-projector/)) have infinite support, the same zeros as $$Q_w$$ and the same value at the area exponent, so they have the same mean as $$w$$ in the flat count and in the first-order response to uniform curvature, where every other exponent is a zero. By construction $$H_{w^{(p)}}(\eta)=p^{\alpha_A-2}\,\mathbb E[H_w(\eta_p)\mid\eta]$$. They differ from $$w$$ at exponents where $$Q_w\neq0$$, which occur at higher order in the curvature.

The variance of $$H_w$$ is controlled by the add-one cost $$\Delta_xH=H(\eta+\delta_x)-H(\eta)$$ through the Poincaré inequality for Poisson functionals, $${\rm Var}\,H\le\mathbb E\int(\Delta_xH)^2\rho\,dx$$ [[Last]](#ref-last) (Theorem 8.3). The lemma shows that the right side is of order $$\mathcal A$$.

<div class="lemma" markdown="1">
**Lemma 1** (finite variance constant). Let $$w$$ be a weighting with $$w_m=0$$ for $$m>K$$. In the planar model with $$\rho=1$$,

$$
M_w=\sup_\Omega\frac1{\mathcal A}\int\mathbb E\big[(\Delta_xH_w^\Omega)^2\big]\,dx<\infty,
$$

the supremum being over boxes $$\Omega$$ of any transverse area $$\mathcal A$$ (in $$d=2$$ there is one count and $$\mathcal A=1$$). Consequently $${\rm Var}\,H_w^\Omega\le M_w\mathcal A$$ and $$v(w)\le M_w$$.
</div>

The difficulty is that the sum defining $$H_w$$ runs over an unbounded region, so the add-one cost has to be shown to be localised near the horizon. The localisation comes from two facts: a molecule requires that the cone of the lower point reach outside $$P$$ before the slice, and a weighting of degree $$K$$ ignores lower points with more than $$K+1$$ points in their future, which become exponentially rare with depth.

<div class="proof" markdown="1">
*Proof.* Put $$Z=\max_m\lvert w_m\rvert$$ and, for a point at depth $$u=-t>0$$, $$V(u)=\omega u^d/d$$, the volume of its future below $$\Sigma$$. The number of sprinkled points in that future is Poisson with mean $$V(u)$$, so the probability that there are at most $$K+1$$ of them is at most $$T(u)=(1+V(u))^{K+1}e^{-V(u)}$$, using $$\sum_{j\le k}V^j/j!\le(1+V)^k$$. Let $$\mathcal E=\{(t,z,y):t<0,\ \lvert z\rvert<-t\}$$ be the rim. A point $$\zeta\in P$$ at depth $$u$$ has $$z<u$$, and its future below $$\Sigma$$ leaves $$P$$ exactly when $$z>-u$$, so only points of $$\mathcal E$$ can be lower points of molecules. A molecule of type $$m$$ has weight $$w_m=0$$ unless $$\lvert F(\zeta)\rvert\le K+1$$.

Adding $$x$$ with $$t_x\ge0$$ changes nothing. For $$t_x<0$$ there are two effects. First, $$x$$ may be the lower point of new molecules; this contributes at most $$Z(K+1)$$ and requires $$x\in\mathcal E$$ and $$\lvert F(x)\rvert\le K+1$$. Second, $$x$$ may lie in the future of an existing lower point $$\zeta\in P$$, raising $$\lvert F(\zeta)\rvert$$ by one and the number of points of $$F(\zeta)$$ outside $$P$$ by one if $$x\notin P$$. The change in the contribution of $$\zeta$$ is at most $$Z(2K+3)$$, and it vanishes unless $$\lvert F(\zeta)\rvert\le K+1$$ and $$F(\zeta)\cup\{x\}\not\subset P$$, which forces $$\zeta\in\mathcal E$$ (for $$\zeta\in P\setminus\mathcal E$$ the whole future of $$\zeta$$ below $$\Sigma$$ lies in $$P$$). Let $$Y_x$$ be the number of $$\zeta\in\eta\cap\mathcal E$$ with $$y_\zeta\in\Omega$$, $$\zeta\prec x$$ and $$\lvert F(\zeta)\rvert\le K+1$$. Then $$\lvert\Delta_xH_w^\Omega\rvert\le Z(2K+3)\,[\mathbf 1\{x\in\mathcal E,\ y_x\in\Omega,\ \lvert F(x)\rvert\le K+1\}+Y_x]$$ and

$$
\mathbb E(\Delta_xH)^2\le2Z^2(2K+3)^2\Big[\mathbf 1\{x\in\mathcal E,\ y_x\in\Omega\}\,T(u_x)+\mathbb E Y_x+\mathbb E Y_x(Y_x-1)\Big].
$$

We integrate the three terms over $$x$$. The first gives $$\mathcal A\int_0^\infty2u\,T(u)\,du$$, since the rim has width $$2u$$ at depth $$u$$. By the Mecke equation $$\mathbb E Y_x\le\int_{\zeta\in\mathcal E,\,y_\zeta\in\Omega,\,\zeta\prec x}T(u_\zeta)\,d\zeta$$, and integrating over $$x$$ (the volume of $$x$$ above $$\zeta$$ and below $$\Sigma$$ is $$V(u_\zeta)$$) gives $$\mathcal A\int_0^\infty2u\,T(u)V(u)\,du$$. For the second factorial moment the Mecke equation gives a double integral over $$\zeta_1,\zeta_2\in\mathcal E$$ with $$y_{\zeta_i}\in\Omega$$ of the probability that both have at most $$K+1$$ further points. The extra points $$\zeta_1,\zeta_2$$ only increase the counts, and the probability of the intersection is at most $$\sqrt{T(u_1)T(u_2)}$$. Integrating over $$x$$ multiplies by the volume of the common future of $$\zeta_1,\zeta_2$$ below $$\Sigma$$, at most $$\min(V(u_1),V(u_2))$$, and it is zero unless $$\lvert y_1-y_2\rvert<u_1+u_2$$, because a common future point $$q$$ has $$t_q<0$$ and so lies at spatial distance less than $$u_i$$ from $$\zeta_i$$. The integral over $$y_2$$ is thus at most $$\omega_\perp(u_1+u_2)^{d-2}$$ with $$\omega_\perp$$ the volume of the unit ball of $$\mathbb R^{d-2}$$ (equal to $$1$$ for $$d=2$$), and over $$z_2$$ at most $$2u_2$$. The $$y_1$$ integral gives $$\mathcal A$$. The result is $$\mathcal A\int_0^\infty\!\!\int_0^\infty4u_1u_2\,\omega_\perp(u_1+u_2)^{d-2}\sqrt{T(u_1)T(u_2)}\,\min(V(u_1),V(u_2))\,du_1du_2$$. Every integrand is a polynomial in $$u$$ times $$e^{-\omega u^d/d}$$ or $$e^{-\omega u^d/(2d)}$$, so all three integrals are finite and independent of $$\Omega$$.
</div>

<div class="proposition" markdown="1">
**Proposition 3** (the variance ratio is unbounded). For every weighting $$w$$ of finite degree with $$Q_w(\alpha_A)\neq0$$ (in particular every exact weighting of finite degree) and every $$p\in(0,1]$$,

$$
v\big(w^{(p)}\big)\le M_w\,p^{2/d},\qquad\text{hence}\qquad r_{w^{(p)}}\ge\frac{a_d^2Q_w(\alpha_A)^2}{M_w}\,p^{-2/d}\longrightarrow\infty\quad(p\to0).
$$

If $$w$$ is exact, the weightings $$w^{(p)}$$ are exact weightings of infinite support, outside the finite-degree class of Proposition 2 and Lemma 1.
</div>

Averaging a count over thinnings keeps the mean and removes variance, because a point of the sprinkling influences the thinned average only when it is retained, which has probability $$p$$, and the rescaled problem is again a sprinkling with a larger discreteness length.

<div class="proof" markdown="1">
*Proof.* Write $$\Phi(\eta)=\mathbb E[H_w^\Omega(\eta_p)\mid\eta]$$. Adding a point $$x$$ to $$\eta$$ changes the thinned configuration only if $$x$$ is retained, so $$\Delta_x\Phi=p\,\mathbb E[\Delta_xH_w^\Omega(\eta_p)\mid\eta]$$, and by the conditional Jensen inequality $$\mathbb E(\Delta_x\Phi)^2\le p^2\,\mathbb E[(\Delta_xH_w^\Omega(\eta_p))^2]$$, where $$\eta_p$$ is a Poisson process of intensity $$p$$. The Poincaré inequality gives $${\rm Var}\,\Phi\le p^2\int\mathbb E_p[(\Delta_xH_w^\Omega)^2]dx$$. The slice and the planar horizon are invariant under the dilation $$x\to p^{1/d}x$$ and $$H_w$$ depends only on the causal order, so a process of intensity $$p$$ and the box $$\Omega$$ become a process of intensity $$1$$ and a box of area $$\mathcal Ap^{(d-2)/d}$$, with volume element scaled by $$p^{-1}$$. By Lemma 1, which holds for every box, $$\int\mathbb E_p[(\Delta_xH_w^\Omega)^2]dx\le p^{-1}\cdot M_w\mathcal Ap^{(d-2)/d}=M_w\mathcal Ap^{-2/d}$$. So $${\rm Var}\,\Phi\le M_w\mathcal Ap^{2-2/d}$$. Finally $$H_{w^{(p)}}=p^{\alpha_A-2}\Phi$$ and $$2(\alpha_A-2)+2-2/d=2/d$$. The mean of $$H_{w^{(p)}}$$ equals that of $$H_w$$ by (5), which gives the bound on $$r_{w^{(p)}}$$.
</div>

### Computed variances

In $$d=2$$ we have $$\alpha_A=2$$, $$a_2=\tfrac13$$, and the exact weighting is $$w=(15,-13,4)$$ with $$Q_w(\alpha)=(3-\alpha)(5-2\alpha)$$. We computed $${\rm Var}\,H_w$$ and the variance of the thinned weights (5), truncated at $$m<70$$, by Monte Carlo over $$4\times10^6$$ sprinklings of the strip $$-5\le t\le0$$, $$\lvert z\rvert\le10$$ with the horizon $$t+z=0$$; the strip is deep enough that the mean of every row is $$\tfrac13$$ to within $$2\%$$. The table gives the variance and the ratio $$r_w=a_2^2/{\rm Var}$$.

| $$p$$ | $${\rm Var}\,H$$ | $$r$$ | factor over $$p=1$$ |
|---|---|---|---|
| $$1$$ | $$304$$ | $$3.7\times10^{-4}$$ | $$1$$ |
| $$0.8$$ | $$60.3$$ | $$1.8\times10^{-3}$$ | $$5.0$$ |
| $$0.7$$ | $$28.5$$ | $$3.9\times10^{-3}$$ | $$10.7$$ |
| $$0.5$$ | $$6.80$$ | $$1.6\times10^{-2}$$ | $$44.7$$ |

Proposition 3 guarantees growth of order $$p^{-1}$$ in $$d=2$$ relative to a constant $$M_w$$, and the computed growth is faster, so the bound is not saturated. In $$d=4$$ a Monte Carlo evaluation on a transverse torus of area $$25$$ and depth cut $$2.4$$, with $$4\times10^5$$ sprinklings, gives $${\rm Var}\,H_w/\mathcal A\approx2.7\times10^{6}$$ for $$w=(210,-791,1268,-928,256)$$, whose mean per unit area is $$a_4=0.1732$$ (here $$Q_w(\alpha_A)=1$$), hence $$r_w\approx1.1\times10^{-8}$$.

So $$r_w$$ has no universal value on exact weightings: it is a different pure number for each weighting and has no upper bound over weightings of infinite support. Under (H1) and (H2), the first and second moments of Poisson kinematics therefore give no pure number without a rule that selects one weighting.

## 5. Rigid Lorentz-invariant configurations

A random substrate is not the only possible source of a number. A finite configuration of vectors that is rigid, fixed up to a Lorentz transformation by rational conditions, has Lorentz invariants that are determined with no sprinkling involved. The question is which numbers these can be. Whether a constant of a sprinkling is of this kind is not addressed here.

<div class="proposition" markdown="1">
**Proposition 4** (algebraic Gram entries). Let $$\mathcal C\subset(\mathbb R^{3,1})^L$$ be a nonempty set defined by finitely many polynomial equations and inequalities with rational coefficients, and suppose that the Gram matrix $$\mathrm{Gr}(v)=(\langle v_i,v_j\rangle)_{ij}$$ takes a single value $$\mathcal G$$ on $$\mathcal C$$ (for instance because $$\mathcal C$$ is a single orbit of the Lorentz group). The form is $$\langle x,y\rangle=x_1y_1+x_2y_2+x_3y_3-x_4y_4$$.

(a) Every entry of $$\mathcal G$$ is a real algebraic number, and so is every real number that is algebraic over the field generated by the entries.

(b) If $$v_i,v_j$$ are timelike unit vectors ($$\langle v,v\rangle=-1$$) in the same cone, the rapidity $$\xi$$ with $$\cosh\xi=-\mathcal G_{ij}$$ is transcendental unless $$\xi=0$$. If $$v_i,v_j$$ are spacelike unit vectors ($$\langle v,v\rangle=1$$) with $$\lvert\mathcal G_{ij}\rvert<1$$, the angle $$\chi\in(0,\pi)$$ with $$\cos\chi=\mathcal G_{ij}$$ is transcendental.
</div>

<div class="proof" markdown="1">
*Proof.* (a) The set $$\{\mathrm{Gr}(v):v\in\mathcal C\}$$ is the image of a semialgebraic set under a polynomial map with rational coefficients, hence is semialgebraic and defined over $$\mathbb Q$$ (Tarski–Seidenberg). By hypothesis it is the single point $$\mathcal G$$. The statement "this set is nonempty" holds over $$\mathbb R$$, hence over the real closed field $$\mathbb R_{\rm alg}$$ of real algebraic numbers by Tarski transfer, so the set contains a point with algebraic coordinates; that point is $$\mathcal G$$. (b) For timelike vectors $$e^{\xi}=\cosh\xi+\sinh\xi=-\mathcal G_{ij}+\sqrt{\mathcal G_{ij}^2-1}$$ is algebraic by (a). If $$\xi\neq0$$ were algebraic, $$e^\xi$$ would be transcendental by the Hermite–Lindemann theorem. For spacelike vectors, $$e^{i\chi}=\mathcal G_{ij}+i\sqrt{1-\mathcal G_{ij}^2}$$ is algebraic, and the same theorem excludes an algebraic $$\chi\neq0$$; $$\chi=0$$ is excluded by $$\lvert\mathcal G_{ij}\rvert<1$$.
</div>

Two timelike unit vectors with rational coordinates, $$v_1=(0,0,0,1)$$ and $$v_2=(\tfrac34,0,0,\tfrac54)$$, have $$\langle v_1,v_2\rangle=-\tfrac54$$, so $$\cosh\xi=\tfrac54$$, $$\sinh\xi=\tfrac34$$ and $$\xi=\ln2$$. The pair is a single Lorentz orbit defined over $$\mathbb Q$$ (its Gram matrix is the only datum), and its rapidity is transcendental by (b), as is every nonzero rapidity or angle between vectors with algebraic Gram entries.

A richer family comes from circle packings. A disc on the celestial sphere is represented by a unit spacelike vector $$v_i\in\mathbb R^{3,1}$$, the normal of the hyperbolic plane bounded by its circle, oriented towards the disc. Two discs are tangent when $$\langle v_i,v_j\rangle=-1$$ and disjoint when $$\langle v_i,v_j\rangle=-I$$ with the *inversive distance* $$I>1$$, the hyperbolic cosine of the distance between the corresponding hyperbolic planes. By the Koebe–Andreev–Thurston theorem a triangulation $$\mathcal T$$ of the sphere has a circle packing with tangency graph $$\mathcal T$$, unique up to Möbius maps, that is, up to the Lorentz group. The set $$\mathcal C$$ of packings with prescribed tangencies is defined by rational equations and inequalities, so Proposition 4(a) applies and the inversive distances of non-adjacent pairs are algebraic. The parameter count matches: $$L$$ discs have $$3L$$ parameters, a triangulation of the sphere has $$3L-6$$ edges and so imposes $$3L-6$$ contacts, and $$6=\dim SO(3,1)$$.

For a stacked triangulation (a vertex repeatedly inserted into a face) the inversive distances are integers. Start from four mutually tangent discs, whose Gram matrix $$2\cdot\mathbf 1-J$$ (diagonal $$1$$, off-diagonal $$-1$$) has eigenvalues $$(-2,2,2,2)$$. The disc inserted in the face $$v_1,v_2,v_3$$, on the side away from the fourth disc $$v_4$$, is $$v_*=2(v_1+v_2+v_3)-v_4$$, for which $$\langle v_*,v_*\rangle=1$$, $$\langle v_*,v_i\rangle=-1$$ for $$i=1,2,3$$ and $$\langle v_*,v_4\rangle=-7$$. All vectors are integer combinations of the initial basis, so all Gram entries are integers. Enumerating the insertion sequences that give $$10$$ vertices shows no inversive distance below $$7$$. For the bipyramid on an $$h$$-gon the ring discs have angular radius $$\pi/h$$ and the two polar discs $$\pi/2-\pi/h$$, and the inversive distance between the polar discs is

$$
I=1+2\tan^2\frac\pi h=\frac{3-\cos(2\pi/h)}{1+\cos(2\pi/h)}.
$$

This is a Möbius function of $$\cos(2\pi/h)$$ with rational coefficients, so $$\mathbb Q(I)=\mathbb Q(\cos2\pi/h)$$; for $$h=3,\dots,8$$ the values are $$7$$, $$3$$, $$11-4\sqrt5=2.0557$$, $$5/3$$, $$1.4638$$, $$7-4\sqrt2=1.3431$$.

The proposition restricts the Gram data of a rigid configuration over $$\mathbb Q$$ to algebraic numbers. It does not restrict the transcendental quantities computed from them: angles and rapidities between its vectors are transcendental by (b), and volumes and other integrals of the configuration are not covered. It also does not apply to a configuration that is rigid over a number field other than $$\mathbb Q$$, whose invariants lie in that field. A single configuration is one orbit of the Lorentz group, on which an average is not normalisable, and the proposition says nothing about the Poisson sprinkling, which is not a rigid configuration. The relation to exact discrete Lorentz symmetry is in [lattices with an arithmetic Lorentz group](/quantum-gravity/discrete-lorentz-group/) and [Lorentz-invariant point sets and the structures they cannot carry](/quantum-gravity/invariant-point-sets/).

## 6. The cosmological constant and the volume degree of the horizon count

A cosmological constant would appear as a term of the count proportional to a volume. Let $$\ell$$ be the size of the ball, so that $$\omega\ell^n$$ is the volume of $$B$$, and let a term of the expected count carry a curvature factor of dimension $$D$$ (a power of inverse length). Its *area degree* is $$\ell^{d-2+D}$$. The *ball-volume degree* is $$\ell^{d-1+D}$$, and the *spacetime-volume degree* is $$\ell^{d+D}$$. At $$D=0$$ they are the terms proportional to the area of $$\partial B$$, to the volume of $$B$$, and to the volume of its domain of dependence, and the last is the form of a cosmological-constant term.

<div class="theorem" markdown="1">
**Theorem 1** (no volume degree). Let $$d\ge3$$ and $$n=d-1$$. Take the metric $$g=W(t)^{2/d}(-dt^2+\gamma_A)$$ with $$\gamma_A$$ the round metric of radius $$A$$ on the $$n$$-sphere, $$W(t)=1+\sum_{k\ge1}W_kt^k$$, $$\Sigma=\{t=0\}$$, and $$B$$ a geodesic ball in $$\Sigma$$ of fixed volume $$\omega\ell^n$$. Expand the expected weighted horizon count $$\langle H_w\rangle$$ as a formal power series in $$A^{-2}$$ and the $$W_k$$; a term of order $$A^{-2\nu}W_{k_1}\cdots W_{k_j}$$ has curvature dimension $$D=2\nu+k_1+\dots+k_j$$. For every finite weighting $$w$$, every term of curvature dimension $$D$$ is a multiple of $$\ell^e$$ with $$e\le d-2+D$$. In particular there is no term of ball-volume or spacetime-volume degree, at any order.
</div>

The expansion is that of [horizon counts and the area law](/quantum-gravity/area-projector/), from which the proof takes its input: by its Lemma 1,

$$
\langle H_w\rangle=\int_0^\infty d\tau\,W(-\tau)\,\phi\big(V(\tau)\big)\,U(\tau),\qquad U(\tau)=\int_0^\tau ds\,W(s-\tau)\,{\rm cap}(s)\big[{\rm cap}(r_0+\tau)-{\rm cap}(r_0+\tau-s)\big],
$$

where $$\phi(v)=e^{-v}\sum_mw_mv^m/m!$$ is the kernel of the weighting. The factor $$W(-\tau)$$ is the conformal volume factor at depth $$\tau$$ below $$\Sigma$$, and $$V(\tau)$$ is the volume of the future of a point at that depth, so $$\phi(V(\tau))$$ is the weight of a lower point after averaging over the further points of its type. The function $$U(\tau)$$ is the volume of pairs, the lower point in the slice of $$P$$ at depth $$\tau$$ and the upper point in its future outside $$P$$: $${\rm cap}(r)$$ is the $$\gamma_A$$-volume of a geodesic ball of radius $$r$$, and $$r_0$$ is the radius of $$B$$. The size $$\ell$$ enters only through $$r_0$$ inside the bracket.

<div class="proof" markdown="1">
*Proof.* The cap volume is $${\rm cap}(r)=\sum_\nu\varpi_\nu A^{-2\nu}r^{n+2\nu}$$ with $$\varpi_0=\omega$$, and fixing the volume of $$B$$ gives $$r_0=\ell\,\psi(\ell^2/A^2)$$ with $$\psi(0)=1$$. The bracket expands as $$\sum_{i\ge1}{\rm cap}^{(i)}(r_0)[\tau^i-(\tau-s)^i]/i!$$, with no term $$i=0$$, and the part of order $$A^{-2\nu}$$ of $${\rm cap}^{(i)}(r_0)$$ is a constant times $$\ell^{\,n+2\nu-i}\le\ell^{\,n-1+2\nu}$$. All other factors are independent of $$\ell$$. A term of order $$A^{-2\nu}$$ has $$D\ge2\nu$$, so $$e\le n-1+2\nu=d-2+2\nu\le d-2+D$$.
</div>

The proof uses only the form of the reduction formula, so the conclusion holds for every finite weighting. Outside the conformally static class the degree bound is a conjecture; the rim localisation of Lemma 1 (a molecule exists only if its lower point lies within its own depth of the horizon) is the supporting observation.

The mean of the horizon count therefore contains no term that could play the role of a cosmological constant, whatever the weights. A cosmological constant enters the count only through the curvature of the background, inside the area-degree response (the response to $$R_{uu}$$ and to the area deficit of [horizon counts and the area law](/quantum-gravity/area-projector/)), and that response matters for balls of the size of the curvature radius.

The fluctuation of the count is a different matter. For a region $$\mathcal D$$ let $$N_{\mathcal D}$$ be the number of points in $$\mathcal D$$.

<div class="proposition" markdown="1">
**Proposition 5** (covariance with the number of points). For any integrable functional $$\Psi$$ of the sprinkling and any region $$\mathcal D$$,

$$
{\rm Cov}(\Psi,N_{\mathcal D})=\rho\int_{\mathcal D}\mathbb E[\Delta_x\Psi]\,dx.
$$

If $$\Delta_x\Psi=0$$ outside $$\mathcal D$$, then $${\rm Cov}(\Psi,N_{\mathcal D})=\rho\,\partial_\rho\mathbb E_\rho\Psi$$. For an exact weighting in $$d$$ dimensions, $$\mathbb E H_w\propto\rho^{(d-2)/d}$$, and for a region $$\mathcal D$$ of depth $$U$$ around the horizon $${\rm Cov}(H_w,N_{\mathcal D})=\tfrac{d-2}d\,\mathbb E H_w$$ up to an error that decays like $$e^{-\omega U^d/(2d)}$$ times a polynomial in $$U$$ (see the proof). The regression coefficient of $$H_w$$ on $$N_{\mathcal D}$$ is then $$\frac{d-2}d\,\mathbb E H_w/(\rho\lvert\mathcal D\rvert)$$. In particular $$H_w$$ and $$N_{\mathcal D}$$ are uncorrelated in $$d=2$$ up to that error.
</div>

<div class="proof" markdown="1">
*Proof.* By the Mecke equation, $$\mathbb E[\Psi\,N_{\mathcal D}]=\mathbb E\int_{\mathcal D}\Psi\,\eta(dx)=\rho\int_{\mathcal D}\mathbb E[\Psi(\eta+\delta_x)]dx=\rho\lvert\mathcal D\rvert\,\mathbb E \Psi+\rho\int_{\mathcal D}\mathbb E[\Delta_x\Psi]dx$$, and $$\mathbb E N_{\mathcal D}=\rho\lvert\mathcal D\rvert$$. For the second statement, a Poisson process of intensity $$\rho+\epsilon$$ is the sum of independent processes of intensities $$\rho$$ and $$\epsilon$$; adding the second to the first adds a point at $$x$$ with probability $$\epsilon\,dx$$ to first order, so $$\partial_\rho\mathbb E_\rho\Psi=\int\mathbb E[\Delta_x\Psi]dx$$. When $$\Delta_x\Psi$$ vanishes outside $$\mathcal D$$ this equals $$\int_{\mathcal D}\mathbb E[\Delta_x\Psi]dx$$. For the horizon count the dependence region is not bounded, but by the estimate in the proof of Lemma 1 the contribution of depth greater than $$U$$ is of order $$e^{-\omega U^d/(2d)}$$ up to polynomial factors, so the identity holds up to that error for a region of depth $$U$$.
</div>

By Lemma 1, $${\rm Var}\,H_w\le M_w\mathcal A\rho^{(d-2)/d}$$ (with the $$\rho$$-scaling of the proof of Proposition 3). The Cauchy–Schwarz inequality gives $${\rm Var}\,H_w\ge{\rm Cov}(H_w,N_{\mathcal D})^2/{\rm Var}\,N_{\mathcal D}=\big(\tfrac{d-2}d\mathbb E H_w\big)^2/(\rho\lvert\mathcal D\rvert)$$ for a region $$\mathcal D$$ of depth $$U$$ around the horizon, whose volume is of order $$\mathcal AU^2$$ in units of $$\ell_\rho$$, which is a lower bound of order $$\mathcal A/U^2$$ in those units. The standard deviation of the count is therefore of order the square root of the area, not of the volume.

### Comparison with the observed vacuum energy density

The comparison needs a rule, which we state as (H3): a fluctuating count $$X$$ plays the role that the number of points $$N$$ plays in the fluctuating cosmological constant of [[Ahmed et al.]](#ref-adgs), with $$\delta\Lambda\sim\Delta X/V_4$$, where $$\Delta X$$ is the standard deviation of $$X$$ and $$V_4$$ the four-volume. For $$X=N$$ this gives $$\delta\Lambda\sim N^{-1/2}$$ in units of $$\ell_\rho^{-2}$$, of the order of the observed value when the Hubble four-volume contains $$N\sim R_H^4$$ points.

For $$X=H_w$$ we set $$\ell_\rho$$ of the order of the Planck length, $$R_H=c/H_0\approx8.4\times10^{60}\,\ell_{\rm Pl}$$ for $$H_0\approx68$$ km s$$^{-1}$$ Mpc$$^{-1}$$ [[DESI]](#ref-desi), $$V_4\sim R_H^4$$, and normalise $$H_w$$ by $$Q_w(\alpha_A)=1$$, so that its mean $$a_4\mathcal A$$ is of the order of the horizon area $$R_H^2$$ in units of $$\ell_\rho^2$$. A count with Poisson-like fluctuations, $$\Delta X=\langle X\rangle^{1/2}\sim R_H$$, would give $$\delta\Lambda\sim R_H^{-3}$$, a fraction $$\ell_{\rm Pl}/R_H\approx10^{-61}$$ of $$\Lambda_{\rm obs}\sim R_H^{-2}$$. For the four-dimensional exact weights the variance per unit area is $$2.7\times10^6$$ (section 4), so the standard deviation exceeds $$\langle H_w\rangle^{1/2}$$ by the factor $$(a_4/r_w)^{1/2}\approx4\times10^3$$, and with $$\mathcal A\sim R_H^2$$ it is $$\Delta X\approx1.6\times10^3R_H$$. The fraction becomes of order $$10^{-58}$$ (the leading digit depends on order-one conventions, here $$\mathcal A=R_H^2$$ and $$a_4=0.17$$).

The component of $$H_w$$ along the number of points is smaller still. Take for $$\mathcal D$$ the Hubble four-volume, so that $${\rm Var}\,N_{\mathcal D}=N\sim R_H^4$$. By Proposition 5 the covariance is of order $$\mathbb E H_w\sim R_H^2$$ (it equals $$\tfrac12\mathbb E H_w$$ in $$d=4$$ when $$\mathcal D$$ contains the region on which $$H_w$$ depends). The part of $$H_w$$ that $$N_{\mathcal D}$$ predicts linearly then has standard deviation of order $$R_H^2/N^{1/2}\sim1$$ in counts, smaller by the factor $$\ell_{\rm Pl}/R_H$$ than the Poisson-like value $$R_H$$, and through it $$\delta\Lambda\sim V_4^{-1}$$ is a fraction of order $$(\ell_{\rm Pl}/R_H)^2\sim10^{-122}$$ of $$\Lambda_{\rm obs}$$. In each case the ratio to the observed value is far below one, because the horizon fluctuation grows with the area and not with the volume. The estimate rests on (H3) and on $$\ell_\rho\sim\ell_{\rm Pl}$$; what is proved is Theorem 1 and the area bounds on the variance.

## 7. The variance at finite degree and other counts

For exact weightings of finite degree, the infimum of $$v$$ at fixed normalisation $$Q_w(\alpha_A)=1$$ over degree at most $$K$$ is non-increasing in $$K$$, since the families are nested. Proposition 3 shows that the infimum of $$v$$ over exact weightings of infinite support is zero. Whether the infimum over finite degree tends to zero or to a positive limit as $$K\to\infty$$ is not known. Selection rules among exact weightings supply a number only once a rule is specified. Higher cumulants, and counts that are not molecule counts or layer actions, are not covered by any statement above. Theorem 1 concerns the formal expansion in the curvature, and the behaviour of the full function of $$A$$ for balls comparable to the curvature radius is open.

## References

1. <a id="ref-bd"></a>Benincasa, Dowker, “The scalar curvature of a causal set,” Phys. Rev. Lett. 104, 181301 (2010), [arXiv:1001.2725](https://arxiv.org/abs/1001.2725). Used for: the layer action and its Einstein–Hilbert limit.
2. <a id="ref-ass"></a>Aslanbeigi, Saravani, Sorkin, “Generalized causal set d'Alembertians,” JHEP 1406, 024 (2014), [arXiv:1403.1622](https://arxiv.org/abs/1403.1622). Used for: the Einstein–Hilbert coefficient proportional to $$Q'(\tfrac32)$$ in flat space.
3. <a id="ref-belenchia"></a>Belenchia, “Universal behaviour of generalized causal set d'Alembertians in curved spacetime,” Class. Quantum Grav. 33, 135011 (2016), [arXiv:1510.04665](https://arxiv.org/abs/1510.04665). Used for: the same coefficient in curved spacetime.
4. <a id="ref-barton"></a>Barton, Counsell, Dowker, Gould, Jubb, Taylor, “Horizon molecules in causal set theory,” Phys. Rev. D 100, 126008 (2019), [arXiv:1909.08620](https://arxiv.org/abs/1909.08620). Used for: horizon molecules and their area law.
5. <a id="ref-su"></a>Susskind, Uglum, “Black hole entropy in canonical quantum gravity and superstring theory,” Phys. Rev. D 50, 2700 (1994), [arXiv:hep-th/9401070](https://arxiv.org/abs/hep-th/9401070). Used for: renormalisation of the area coefficient and of Newton's constant together.
6. <a id="ref-adgs"></a>Ahmed, Dodelson, Greene, Sorkin, “Everpresent Lambda,” Phys. Rev. D 69, 103523 (2004), [arXiv:astro-ph/0209274](https://arxiv.org/abs/astro-ph/0209274). Used for: the fluctuating cosmological constant from the number of points.
7. <a id="ref-vz"></a>Verlinde, Zurek, “Modular fluctuations from shockwave geometries,” Phys. Rev. D 106, 106011 (2022), [arXiv:2208.01059](https://arxiv.org/abs/2208.01059). Used for: the modular area law.
8. <a id="ref-last"></a>Last, “Stochastic analysis for Poisson processes,” [arXiv:1405.4416](https://arxiv.org/abs/1405.4416). Used for: the Mecke equation and the Poincaré inequality.
9. <a id="ref-desi"></a>DESI Collaboration, “DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints,” Phys. Rev. D 112, 083515 (2025), [arXiv:2503.14738](https://arxiv.org/abs/2503.14738). Used for: the Hubble rate.
