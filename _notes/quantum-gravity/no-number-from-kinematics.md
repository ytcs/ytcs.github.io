---
title: "Causal-set counts and dimensionless constants"
date: 2026-10-29
category: substrate
order: 7
description: On a Poisson sprinkling the means and variances of the causal-set action and of weighted horizon counts leave the Newton constant in units of the discreteness length and the entropy per horizon molecule undetermined, the variance of exact horizon weightings can be made arbitrarily small relative to the squared mean, and the mean horizon count has no volume term.
status: "Propositions 1 to 3 and 5, Lemma 1 and Theorem 1 are proved for Poisson sprinklings under the stated hypotheses (Theorem 1 for conformally static backgrounds, as a consequence of the area-projector page); Proposition 4 is proved for rigid configurations over the rationals; Monte Carlo checks in d = 2 and d = 4; the comparison with the observed vacuum energy density is an order-of-magnitude estimate under rule (H3)."
excludes: "A value of the Newton constant in units of the discreteness length, or of an entropy unit per weighted molecule, from the Poisson means alone when the unit is free (H1) and the weight normalisation is conventional (H2); a universal value of the squared-mean-to-variance ratio of exact horizon weightings; a volume term in the mean horizon count in conformally static backgrounds; a transcendental Gram entry from a rigid Lorentz-invariant configuration defined over the rationals. Transcendental angles and rapidities of such a configuration are not excluded."
---

A Poisson sprinkling has one length, the discreteness length $$\ell_\rho=\rho^{-1/d}$$. Do the mean and the variance of the local counts built on it fix a dimensionless constant, such as the Newton constant in units of $$\ell_\rho$$, an entropy per horizon molecule, or a cosmological constant? For the layer counts of the causal-set action and for weighted horizon counts they do not, under the conditions below. When the horizon count carries no entropy unit of its own and the normalisation of the weights is a convention, the means are linear in the weights and determine $$G/\ell_\rho^2$$ only through a product with a free coupling (Proposition 1). The one scale-free second-moment number, the squared mean divided by the variance per unit area, has no upper bound over exact weightings (Proposition 3). In conformally static backgrounds the mean horizon count has no term proportional to a volume (Theorem 1), and its variance grows with the area (Lemma 1), so neither supplies a cosmological constant. A finite rigid configuration defined over the rationals has algebraic Gram entries (Proposition 4); transcendental angles and rapidities of such a configuration are functions of those entries and are not excluded.

The free coupling is $$\lambda=(\ell_\rho/\ell_p)^2$$ in the action of Benincasa and Dowker [[BD]](#ref-bd), and the cutoff dependence of the area coefficient of an entropy, together with the renormalisation of Newton's constant, is known from entanglement entropy [[Susskind–Uglum]](#ref-su). Horizon molecules and their area law are due to [[Barton et al.]](#ref-barton). The kernels with extra Mellin zeros are those of [Boundary and corner terms of the causal-set action](/quantum-gravity/boundary-terms/) and the exact weights those of [Horizon counts: which weights give the area law](/quantum-gravity/area-projector/). Ahmed, Dodelson, Greene and Sorkin [[Ahmed et al.]](#ref-adgs) obtain a fluctuating cosmological constant of the observed order from the fluctuation of the number of points in a sprinkling; section 6 asks what changes when that number is replaced by the horizon count. What is added here is the joint homogeneity of the action and the entropy, a thinning construction that makes the variance ratio unbounded together with the proof that the variance constant is finite, the degree bound on the mean horizon count with a covariance identity for its dependence on the number of points, and the statement of what a rigid Lorentz-invariant configuration can and cannot supply.

## 1. The counts

A *Poisson sprinkling* of density $$\rho$$ into a $$d$$-dimensional spacetime is a Poisson point process with intensity $$\rho$$ times the volume, ordered by the causal relation $$\prec$$. The multivariate Mecke equation [[Last]](#ref-last) says that the expectation of a sum over $$k$$-tuples of distinct points equals the integral over the positions of the tuple of the expectation, with the process in the integrand replaced by the process plus the tuple. All statements concern means and variances of the counts below.

**The layer action.** In four dimensions let $$N$$ be the number of points and $$N_k$$ the number of pairs $$p\prec q$$ with exactly $$k$$ points between them. A layer action is

$$
S=\lambda\,\kappa\Big[b\,N+\sum_kc_kN_k\Big],\qquad\lambda=\Big(\frac{\ell_\rho}{\ell_p}\Big)^2,\quad\kappa=\frac4{\sqrt6},\quad\ell_p^2=8\pi G, \tag{1}
$$

with real $$b$$ and $$c_k$$. The kernel $$f(v)=e^{-v}\sum_kc_kv^k/k!$$ has the Mellin symbol $$Q(s)=\sum_kc_k(s)_k/k!$$, where $$(s)_k=s(s+1)\cdots(s+k-1)$$ and $$\Gamma(s)Q(s)=\int_0^\infty f(v)v^{s-1}dv$$. The Benincasa–Dowker action has $$b=1$$, $$c=(-1,9,-16,8)$$ and $$Q_{\rm BD}(s)=\tfrac13(2s-1)(s-1)(2s-3)$$. The mean of $$S$$ tends to a multiple of the Einstein–Hilbert action when $$Q(\tfrac12)=Q(1)=Q(\tfrac32)=0$$ and $$b=-3Q'(1)$$; these are the conditions (K1)–(K3) of the boundary-terms page. That page also imposes $$Q(\tfrac54)=0$$ to remove a divergence at timelike boundaries. The Benincasa–Dowker symbol has $$Q_{\rm BD}(\tfrac54)=-\tfrac1{16}$$, and this extra zero is not used below.

**Horizon molecules.** Let $$\Sigma$$ be a spacelike slice, $$B\subset\Sigma$$ a geodesic ball, $$o$$ the top tip of the domain of dependence of $$B$$, and $$n=d-1$$. The horizon is the part of $$\partial I^-(o)$$ below $$\Sigma$$, and $$P=I^-(o)\cap I^-(\Sigma)$$ is the region inside it. For a point $$p\in P$$ let $$F(p)$$ be the set of sprinkled points in the causal future of $$p$$ and the past of $$\Sigma$$. A *horizon molecule* is a pair $$p\prec q$$ with $$p\in P$$ and $$q\in F(p)\setminus P$$, and its *type* is $$m=\lvert F(p)\rvert-1$$, the number of points of $$F(p)$$ other than $$q$$, wherever they lie. A molecule of type $$0$$ is a link across the horizon. In two dimensions, with $$\Sigma=\{t=0\}$$ and the horizon along $$t+z=0$$, a molecule is a related pair with the lower point under the null line $$t+z=0$$ and the upper point above it, both below $$t=0$$, and its type counts the further points in the future of the lower point below $$t=0$$. With weights $$w_0,\dots,w_K$$ the weighted count is $$H_w=\sum_mw_mH_m$$, $$H_m$$ being the number of molecules of type $$m$$, and the Mellin symbol of the weighting is $$Q_w(\alpha)=\sum_mw_m(\alpha)_m/m!$$.

The expected count depends on the weights only through $$Q_w$$. In a conformally static background the term of the expansion in the curvature and in the size of the ball that carries the density $$\rho^{2-\beta}$$ is multiplied by $$Q_w(\beta)$$. The leading flat-space term sits at the *area exponent* $$\alpha_A=1+2/d$$, and for a flat ball of radius $$\ell$$

$$
\langle H_w\rangle=a_d\,Q_w(\alpha_A)\,\lvert\partial B\rvert+\sum_{j=1}^{d-2}C_j\,Q_w\big(\alpha_A+\tfrac jd\big)\,\ell^{\,d-2-j},\qquad a_d=\frac{\omega}{d(d+1)}\Big(\frac d\omega\Big)^{\alpha_A}\Gamma(\alpha_A), \tag{2}
$$

with constants $$C_j$$ and $$\omega$$ the volume of the unit $$n$$-ball; $$a_4=\sqrt3/10$$, $$a_2=1/3$$ (in $$d=2$$ the area $$\lvert\partial B\rvert$$ is replaced by $$1$$). A finite weighting with $$Q_w(\alpha_A)\neq0$$ and $$Q_w(\alpha_A+j/d)=0$$ for $$j=1,\dots,d$$ is an *exact weighting*. For it the flat count is exactly $$a_dQ_w(\alpha_A)\lvert\partial B\rvert$$ at every ball size, and the first-order response to uniform curvature has the form of the area response. The exact weighting of lowest degree is unique up to normalisation, with $$Q_d(\alpha)=\binom{2d+2-d\alpha}{d}$$, and in four dimensions its weights are $$(210,-791,1268,-928,256)$$. (Proposition 1 and Theorem 2 of the area-projector page, where the proofs are given.)

## 2. The action and the entropy: only a product is fixed

Two hypotheses enter this section. The first is (H1): the count $$H_w$$ contains no entropy unit, so that the constant $$s_0$$ in the horizon entropy $$S_{\rm hor}=s_0H_w$$ is not determined by the sprinkling. This is the weakest hypothesis. An intrinsic unit for weighted molecules would remove the conclusion; the area-projector page leaves open whether a signed count can be read as an entropy. The second is (H2): the normalisation of a weighting is a convention, so that $$w$$ and $$\mu w$$ with $$\mu>0$$ describe the same functional in different units.

<div class="proposition" markdown="1">
**Proposition 1** (homogeneity). Consider the layer action (1) with a kernel satisfying (K1)–(K3), and put $$\theta_Q=\tfrac32Q'(\tfrac32)$$, so that $$\theta_Q=1$$ for Benincasa–Dowker.

(a) The conditions (K1)–(K3), and the ratio of any two coefficients of the large-density expansion of $$\langle S\rangle$$ (for instance a corner value divided by the Einstein–Hilbert coefficient), are unchanged by $$(b,c_k)\to\mu(b,c_k)$$ and by $$\lambda\to c\lambda$$ for $$\mu,c>0$$.

(b) The coefficient of $$\tfrac12\int R\sqrt{-g}$$ in the limit of $$\langle S\rangle$$ is $$1/(8\pi G)$$ and

$$
\frac G{\ell_\rho^2}=\frac1{8\pi\,\lambda\,\theta_Q}, \tag{3}
$$

which changes by $$\mu^{-1}$$ under the first rescaling and by $$c^{-1}$$ under the second. The conditions in (a) therefore do not determine $$G/\ell_\rho^2$$.

(c) Take $$d=4$$. The area law $$S_{\rm hor}=\lvert\partial B\rvert/4G$$ for $$S_{\rm hor}=s_0H_w$$ gives $$1/(4G)=s_0a_4Q_w(\alpha_A)\ell_\rho^{-2}$$, in which only the product $$s_0Q_w(\alpha_A)$$ appears. Requiring this $$G$$ to equal the one in (3) gives

$$
s_0\,Q_w(\alpha_A)=\frac{2\pi}{a_4}\,\lambda\theta_Q=\frac{20\pi}{\sqrt3}\,\lambda\theta_Q=36.276\,\lambda\theta_Q, \tag{4}
$$

a single relation between the free quantities $$s_0Q_w(\alpha_A)$$ and $$\lambda\theta_Q$$.
</div>

The proposition is bookkeeping. The mean of the action is linear in $$(b,c_k)$$ and each condition of (K1)–(K3) is either a zero of $$Q$$ or a ratio ($$b=-3Q'(1)$$), so a common factor drops out of all of them while the overall size of the kernel remains free.

<div class="proof" markdown="1">
*Proof.* (a) By the Mecke equation $$\langle N\rangle=\rho\,{\rm Vol}$$ and $$\langle N_k\rangle=\rho^2\iint(\rho V_{pq})^ke^{-\rho V_{pq}}/k!$$, so $$\langle S\rangle$$ is linear in $$\lambda\cdot(b,c_k)$$. Zeros of $$Q$$ and the relation $$b=-3Q'(1)$$ are unchanged by $$Q\to\mu Q$$, and a ratio of coefficients is unchanged because numerator and denominator scale alike. (In the flat product regions of the boundary-terms page each coefficient is $$Q$$ or $$Q'$$ at one of the points $$\tfrac12,1,\tfrac54,\tfrac32$$ times a geometric constant, but linearity is all that is used.) (b) Aslanbeigi, Saravani and Sorkin [[ASS]](#ref-ass) show that for kernels with (K1)–(K3) the coefficient of the Einstein–Hilbert term is proportional to $$Q'(\tfrac32)$$, and Belenchia [[Belenchia]](#ref-belenchia) extends the curvature limit to the whole family. For Benincasa–Dowker the mean tends to $$\tfrac12\int R\sqrt{-g}/\ell_p^2=\lambda\,\tfrac12\int R\sqrt{-g}/\ell_\rho^2$$ [[BD]](#ref-bd), and $$Q'_{\rm BD}(\tfrac32)=\tfrac23$$, so the proportionality constant is fixed by $$\theta_Q=1$$ there. With $$\ell_p^2=8\pi G$$ the coefficient of $$\tfrac12\int R\sqrt{-g}$$ is $$1/(8\pi G)=\lambda\theta_Q/\ell_\rho^2$$, which is (3). (c) By (2), $$\langle H_w\rangle=a_dQ_w(\alpha_A)\lvert\partial B\rvert\rho^{(d-2)/d}$$ at leading order, linear in $$w$$, which gives the first statement. Equating $$1/(4G)=2\pi\lambda\theta_Q/\ell_\rho^2$$ from (3) with $$s_0a_4Q_w(\alpha_A)/\ell_\rho^2$$ gives (4).
</div>

The factor $$2\pi$$ in (4) comes from the normalisations $$1/(8\pi G)$$ of the Einstein–Hilbert coefficient and $$1/(4G)$$ of the area law, and it relates the two free quantities. The area law with $$G$$ the coefficient of $$\tfrac12\int R$$ holds for every $$\lambda$$ once $$s_0$$ is chosen by (4).

*Example.* Suppose one postulates $$s_0=\ln2$$ with $$Q_w(\alpha_A)=\theta_Q=1$$ (one bit per unit of weighted count). Then (4) gives $$\lambda=s_0a_4/2\pi=0.0191$$ and $$G/\ell_\rho^2=1/(4a_4s_0)=2.082$$, that is $$\ell_\rho\approx0.69\sqrt G$$. Any other postulated $$s_0$$ gives another value.

Two routes lie outside (H1) and (H2). The entropy of the short-distance fields could supply an absolute unit, as in induced-gravity pictures where the area coefficient and $$1/G$$ renormalise together [[Susskind–Uglum]](#ref-su); [Can induced gravity fix Newton's constant?](/quantum-gravity/induced-gravity-number/) discusses their scheme dependence. A dynamical selection of $$\lambda$$, for example a critical value in a path integral over causal sets, is a question about dynamics that the kernel family does not address.

## 3. Rigid first-moment functionals

The exact weights of the previous section are fixed by a normalisation and a list of zeros. We call a weighted count *rigid* if its symbol is fixed by conditions of this type: $$Q(t)=1$$ at one exponent $$t$$ and zeros at the other points of a set $$\mathcal S$$ of exponents.

<div class="proposition" markdown="1">
**Proposition 2** (rigid functionals are Lagrange polynomials). Let $$\mathcal S$$ be a finite set of $$n+1$$ distinct real numbers and $$t\in\mathcal S$$. The weightings $$w$$ of degree at most $$K$$ with $$Q_w(t)=1$$ and $$Q_w(s)=0$$ for $$s\in\mathcal S\setminus\{t\}$$ form an affine space of dimension $$K-n$$ if $$K\ge n$$ and are empty if $$K<n$$. For $$K=n$$ there is exactly one, with $$Q_w=L_{t,\mathcal S}$$, the Lagrange basis polynomial of the node set $$\mathcal S$$ at $$t$$.
</div>

<div class="proof" markdown="1">
*Proof.* The map $$w\mapsto Q_w$$ is a linear bijection from weightings of degree at most $$K$$ onto polynomials of degree at most $$K$$, because $$(s)_m/m!$$ has degree exactly $$m$$. The conditions prescribe the values of $$Q$$ at $$n+1$$ distinct nodes. For $$K=n$$ polynomial interpolation gives a unique solution. For $$K>n$$ the solutions form a coset of the space of polynomials of degree at most $$K$$ vanishing at all nodes, of dimension $$K-n$$. For $$K<n$$ a nonzero polynomial of degree at most $$K$$ cannot have $$n$$ zeros, so the only candidate is $$0$$, which fails the normalisation.
</div>

For the horizon count in $$d=4$$, $$\mathcal S=\{\tfrac32,\tfrac74,2,\tfrac94,\tfrac52\}$$ and $$t=\tfrac32$$ give the integer weights $$(210,-791,1268,-928,256)$$ of section 1. Adding the nodes $$\tfrac{11}4$$ and $$3$$ gives the next member of the tower (see the area-projector page).

<div class="corollary" markdown="1">
**Corollary 1** (no relation between rigid functionals from first moments). Let $$F_1,\dots,F_r$$ be rigid first-moment functionals, each identified with a physical quantity $$X_i=u_i\langle F_i\rangle$$ with a unit $$u_i$$. By (H2) the normalisations $$Q_{w_i}(t_i)=1$$ are conventions. The zero conditions that remain are invariant under $$w_i\to\mu_iw_i$$ with independent $$\mu_i>0$$, and $$\langle F_i\rangle$$ is linear in $$w_i$$. The units $$u_i$$ are therefore independent free parameters, and no relation among $$X_1,\dots,X_r$$ (for instance a value of $$X_1/X_2$$) follows from the conditions. An invariant combination involves a single functional and is a ratio of outputs of that functional, such as a corner value divided by the Einstein–Hilbert coefficient.
</div>

This extends the homogeneity of section 2 to any finite number of functionals. It excludes a procedure in which two or more rigid count functionals fix each other's units.

## 4. The variance ratio

The mean of $$H_w$$ is linear in $$w$$ and the variance is quadratic, so the ratio of the squared mean to the variance is invariant under $$w\to\mu w$$ and is a pure number attached to each exact weighting. Thinning shows that it has no upper bound.

**The planar horizon.** For the second moment we use the flat geometry that a large ball presents near its boundary: the horizon $$t+z=0$$ in Minkowski space of dimension $$d$$, with $$\Sigma=\{t=0\}$$, $$P=\{t<0,\ t+z<0\}$$, transverse coordinates $$y\in\mathbb R^{d-2}$$, and $$\rho=1$$ (for $$\rho\neq1$$ lengths are measured in $$\ell_\rho$$). For a box $$\Lambda$$ of transverse area $$\mathcal A$$ let $$H_w^{\Lambda}$$ be the weighted count of the molecules whose lower point has $$y\in\Lambda$$; in $$d=2$$ there is no transverse direction and $$\mathcal A=1$$. The mean is $$\langle H_w^\Lambda\rangle=a_dQ_w(\alpha_A)\mathcal A$$ for any weighting, being the planar limit of the leading term in (2). We write $$v(w)=\limsup_{\mathcal A\to\infty}{\rm Var}\,H_w^\Lambda/\mathcal A$$ and

$$
r_w=\frac{a_d^2Q_w(\alpha_A)^2}{v(w)},
$$

which is unchanged by $$w\to\mu w$$. If one postulates that a count $$\mu H_w$$ with variance equal to its mean is the horizon entropy, the modular area law $$\langle\Delta K^2\rangle=\langle K\rangle=\mathcal A/4G$$ discussed in [[Verlinde–Zurek]](#ref-vz), then $$\mu=a_dQ_w(\alpha_A)/v(w)$$, the mean of $$\mu H_w$$ is $$r_w\mathcal A$$, and $$G/\ell_\rho^{d-2}=1/(4r_w)$$. This is a postulate, and a classical sprinkling variance and a modular fluctuation are different objects. The question here is only whether $$r_w$$ is a universal number.

**Thinning.** Let $$\eta_p$$ be the independent $$p$$-thinning of a sprinkling $$\eta$$, in which each point is kept with probability $$p$$. A molecule survives with probability $$p^2$$, and each of the $$m$$ further points of its type is kept independently, so that $$\mathbb E[H_w(\eta_p)\mid\eta]=p^2\sum_{\rm molecules}(B_pw)(m)$$ with $$(B_pw)(m)=\sum_k\binom mkp^k(1-p)^{m-k}w_k$$. The weights

$$
w^{(p)}=p^{\alpha_A}B_pw,\qquad\text{symbol}\quad Q^{(p)}(\alpha)=p^{\alpha_A-\alpha}Q_w(\alpha), \tag{5}
$$

(Proposition 5 of the area-projector page) have infinite support, the same zeros as $$Q_w$$ and the same value at the area exponent, so they have the same mean as $$w$$ in the flat count and in the first-order response to uniform curvature, where every other exponent is a zero. By construction $$H_{w^{(p)}}(\eta)=p^{\alpha_A-2}\,\mathbb E[H_w(\eta_p)\mid\eta]$$. They differ from $$w$$ at exponents where $$Q_w\neq0$$, which occur at higher order in the curvature.

The variance of $$H_w$$ is controlled by the add-one cost $$\Delta_xH=H(\eta+\delta_x)-H(\eta)$$ through the Poincaré inequality for Poisson functionals, $${\rm Var}\,H\le\mathbb E\int(\Delta_xH)^2\rho\,dx$$ [[Last]](#ref-last) (Theorem 8.3). The lemma shows that the right side is of order $$\mathcal A$$.

<div class="lemma" markdown="1">
**Lemma 1** (finite variance constant). Let $$w$$ be a weighting with $$w_m=0$$ for $$m>K$$. In the planar model with $$\rho=1$$,

$$
c_w=\sup_\Lambda\frac1{\mathcal A}\int\mathbb E\big[(\Delta_xH_w^\Lambda)^2\big]\,dx<\infty,
$$

the supremum being over boxes $$\Lambda$$ of any transverse area $$\mathcal A$$. Consequently $${\rm Var}\,H_w^\Lambda\le c_w\mathcal A$$ and $$v(w)\le c_w$$.
</div>

The difficulty is that the sum defining $$H_w$$ runs over an unbounded region, so the add-one cost has to be shown to be localised near the horizon. The localisation comes from two facts: a molecule requires that the cone of the lower point reach outside $$P$$ before the slice, and a weighting of degree $$K$$ ignores lower points with more than $$K+1$$ points in their future, which become exponentially rare with depth.

<div class="proof" markdown="1">
*Proof.* Put $$Z=\max_m\lvert w_m\rvert$$ and, for a point at depth $$u=-t>0$$, $$V(u)=\omega u^d/d$$, the volume of its future below $$\Sigma$$. The number of sprinkled points in that future is Poisson with mean $$V(u)$$, so the probability that there are at most $$K+1$$ of them is at most $$g(u)=(1+V(u))^{K+1}e^{-V(u)}$$, using $$\sum_{j\le k}V^j/j!\le(1+V)^k$$. Let $$R=\{(t,z,y):t<0,\ \lvert z\rvert<-t\}$$ be the rim. A point $$p\in P$$ at depth $$u$$ has $$z<u$$, and its future below $$\Sigma$$ leaves $$P$$ exactly when $$z>-u$$, so only points of $$R$$ can be lower points of molecules. A molecule of type $$m$$ has weight $$w_m=0$$ unless $$\lvert F(p)\rvert\le K+1$$.

Adding $$x$$ with $$t_x\ge0$$ changes nothing. For $$t_x<0$$ there are two effects. First, $$x$$ may be the lower point of new molecules; this contributes at most $$Z(K+1)$$ and requires $$x\in R$$ and $$\lvert F(x)\rvert\le K+1$$. Second, $$x$$ may lie in the future of an existing lower point $$p\in P$$, raising $$\lvert F(p)\rvert$$ by one and the number of points of $$F(p)$$ outside $$P$$ by one if $$x\notin P$$. The change in the contribution of $$p$$ is at most $$Z(2K+3)$$, and it vanishes unless $$\lvert F(p)\rvert\le K+1$$ and $$F(p)\cup\{x\}\not\subset P$$, which forces $$p\in R$$ (for $$p\in P\setminus R$$ the whole future of $$p$$ below $$\Sigma$$ lies in $$P$$). Let $$N_x$$ be the number of $$p\in\eta\cap R$$ with $$y_p\in\Lambda$$, $$p\prec x$$ and $$\lvert F(p)\rvert\le K+1$$. Then $$\lvert\Delta_xH_w^\Lambda\rvert\le Z(2K+3)\,[\mathbf 1\{x\in R,\ y_x\in\Lambda,\ \lvert F(x)\rvert\le K+1\}+N_x]$$ and

$$
\mathbb E(\Delta_xH)^2\le2Z^2(2K+3)^2\Big[\mathbf 1\{x\in R,\ y_x\in\Lambda\}\,g(u_x)+\mathbb E N_x+\mathbb E N_x(N_x-1)\Big].
$$

We integrate the three terms over $$x$$. The first gives $$\mathcal A\int_0^\infty2u\,g(u)\,du$$, since the rim has width $$2u$$ at depth $$u$$. By the Mecke equation $$\mathbb E N_x\le\int_{p\in R,\,y_p\in\Lambda,\,p\prec x}g(u_p)\,dp$$, and integrating over $$x$$ (the volume of $$x$$ above $$p$$ and below $$\Sigma$$ is $$V(u_p)$$) gives $$\mathcal A\int_0^\infty2u\,g(u)V(u)\,du$$. For the second factorial moment the Mecke equation gives a double integral over $$p_1,p_2\in R$$ with $$y_{p_i}\in\Lambda$$ of the probability that both have at most $$K+1$$ further points. The extra points $$p_1,p_2$$ only increase the counts, and the probability of the intersection is at most $$\sqrt{g(u_1)g(u_2)}$$. Integrating over $$x$$ multiplies by the volume of the common future of $$p_1,p_2$$ below $$\Sigma$$, at most $$\min(V(u_1),V(u_2))$$, and it is zero unless $$\lvert y_1-y_2\rvert<u_1+u_2$$, because a common future point $$q$$ has $$t_q<0$$ and so lies at spatial distance less than $$u_i$$ from $$p_i$$. The integral over $$y_2$$ is thus at most $$\omega_{d-2}(u_1+u_2)^{d-2}$$ with $$\omega_{d-2}$$ the volume of the unit ball of $$\mathbb R^{d-2}$$ (equal to $$1$$ for $$d=2$$), and over $$z_2$$ at most $$2u_2$$. The $$y_1$$ integral gives $$\mathcal A$$. The result is $$\mathcal A\int_0^\infty\!\!\int_0^\infty4u_1u_2\,\omega_{d-2}(u_1+u_2)^{d-2}\sqrt{g(u_1)g(u_2)}\,\min(V(u_1),V(u_2))\,du_1du_2$$. Every integrand is a polynomial in $$u$$ times $$e^{-\omega u^d/d}$$ or $$e^{-\omega u^d/(2d)}$$, so all three integrals are finite and independent of $$\Lambda$$.
</div>

<div class="proposition" markdown="1">
**Proposition 3** (the variance ratio is unbounded). For every weighting $$w$$ of finite degree with $$Q_w(\alpha_A)\neq0$$ (in particular every exact weighting) and every $$p\in(0,1]$$,

$$
v\big(w^{(p)}\big)\le c_w\,p^{2/d},\qquad\text{hence}\qquad r_{w^{(p)}}\ge\frac{a_d^2Q_w(\alpha_A)^2}{c_w}\,p^{-2/d}\longrightarrow\infty\quad(p\to0).
$$

The weightings $$w^{(p)}$$ have infinite support and so lie outside the finite-degree class of Proposition 2 and Lemma 1.
</div>

Averaging a count over thinnings keeps the mean and removes variance, because a point of the sprinkling influences the thinned average only when it is retained, which has probability $$p$$, and the rescaled problem is again a sprinkling with a larger discreteness length.

<div class="proof" markdown="1">
*Proof.* Write $$F(\eta)=\mathbb E[H_w^\Lambda(\eta_p)\mid\eta]$$. Adding a point $$x$$ to $$\eta$$ changes the thinned configuration only if $$x$$ is retained, so $$\Delta_xF=p\,\mathbb E[\Delta_xH_w^\Lambda(\eta_p)\mid\eta]$$, and by the conditional Jensen inequality $$\mathbb E(\Delta_xF)^2\le p^2\,\mathbb E[(\Delta_xH_w^\Lambda(\eta_p))^2]$$, where $$\eta_p$$ is a Poisson process of intensity $$p$$. The Poincaré inequality gives $${\rm Var}\,F\le p^2\int\mathbb E_p[(\Delta_xH_w^\Lambda)^2]dx$$. The slice and the planar horizon are invariant under the dilation $$x\to p^{1/d}x$$ and $$H_w$$ depends only on the causal order, so a process of intensity $$p$$ and the box $$\Lambda$$ become a process of intensity $$1$$ and a box of area $$\mathcal Ap^{(d-2)/d}$$, with volume element scaled by $$p^{-1}$$. By Lemma 1, which holds for every box, $$\int\mathbb E_p[(\Delta_xH_w^\Lambda)^2]dx\le p^{-1}\cdot c_w\mathcal Ap^{(d-2)/d}=c_w\mathcal Ap^{-2/d}$$. So $${\rm Var}\,F\le c_w\mathcal Ap^{2-2/d}$$. Finally $$H_{w^{(p)}}=p^{\alpha_A-2}F$$ and $$2(\alpha_A-2)+2-2/d=2/d$$. The mean of $$H_{w^{(p)}}$$ equals that of $$H_w$$ by (5), which gives the bound on $$r_{w^{(p)}}$$.
</div>

*Numerical check.* In $$d=2$$ we have $$\alpha_A=2$$, $$a_2=\tfrac13$$, and the exact weighting is $$w=(15,-13,4)$$ with $$Q_w(\alpha)=(3-\alpha)(5-2\alpha)$$. We sprinkled the strip $$-5\le t\le0$$, $$\lvert z\rvert\le10$$ with the horizon $$t+z=0$$ for $$4\times10^6$$ realisations. The means of $$H_m$$ for $$m=0,\dots,5$$ agree with $$(m+1)/3$$ to the statistical error of about $$10^{-3}$$. The variance of $$H_w$$, and that of the thinned weights (5) truncated at $$m<70$$, together with the ratio $$r_w=a_2^2/{\rm Var}$$, are:

| $$p$$ | $${\rm Var}\,H$$ | $$r$$ | factor over $$p=1$$ |
|---|---|---|---|
| $$1$$ | $$304$$ | $$3.7\times10^{-4}$$ | $$1$$ |
| $$0.8$$ | $$60.3$$ | $$1.8\times10^{-3}$$ | $$5.0$$ |
| $$0.7$$ | $$28.5$$ | $$3.9\times10^{-3}$$ | $$10.7$$ |
| $$0.5$$ | $$6.80$$ | $$1.6\times10^{-2}$$ | $$44.7$$ |

The mean of the count stays at $$\tfrac13$$ for all four weightings within about two standard errors; it is $$0.331\pm0.001$$ at $$p=0.5$$, slightly low because of the finite depth of the window, and smaller $$p$$ would need a deeper one.
The bound of Proposition 3 gives a factor $$p^{-1}$$ in $$d=2$$, and the observed growth is faster, so the bound is not saturated. In $$d=4$$ a Monte Carlo evaluation on a transverse torus of area $$25$$ and depth cut $$2.4$$, with $$4\times10^5$$ realisations, gives $${\rm Var}\,H_w/\mathcal A\approx2.7\times10^{6}$$ for $$w=(210,-791,1268,-928,256)$$, against a mean $$a_4=0.1732$$ per unit area, hence $$r_w\approx1.1\times10^{-8}$$. The means of $$H_m$$ per unit area for $$m=0,\dots,5$$ agree with $$a_4(\tfrac32)_m/m!$$ to about $$0.3\%$$.

So the ratio $$r_w$$ has no universal value on exact weightings. Together with Corollary 1 and Proposition 1, the first and second moments of Poisson kinematics give no pure number unless a selection rule among exact weightings is added.

## 5. Rigid Lorentz-invariant configurations

A random substrate is not the only possible source of a number. A finite configuration of vectors that is Lorentz invariant in the sense that it is rigid, fixed up to a Lorentz transformation by rational conditions, has Lorentz invariants that are determined with no sprinkling involved. The question is which numbers these can be.

<div class="proposition" markdown="1">
**Proposition 4** (algebraic Gram entries). Let $$C\subset(\mathbb R^{3,1})^N$$ be a nonempty set defined by finitely many polynomial equations and inequalities with rational coefficients, and suppose that the Gram matrix $$G(v)=(\langle v_i,v_j\rangle)_{ij}$$ takes a single value $$g$$ on $$C$$ (for instance because $$C$$ is a single orbit of the Lorentz group). The form is $$\langle x,y\rangle=x_1y_1+x_2y_2+x_3y_3-x_4y_4$$.

(a) Every entry of $$g$$ is a real algebraic number, and so is every real number that is algebraic over the field generated by the entries.

(b) If $$v_i,v_j$$ are timelike unit vectors ($$\langle v,v\rangle=-1$$) in the same cone, the rapidity $$\eta$$ with $$\cosh\eta=-g_{ij}$$ is transcendental unless $$\eta=0$$. If $$v_i,v_j$$ are spacelike unit vectors ($$\langle v,v\rangle=1$$) with $$\lvert g_{ij}\rvert<1$$, the angle $$\theta\in(0,\pi)$$ with $$\cos\theta=g_{ij}$$ is transcendental.
</div>

<div class="proof" markdown="1">
*Proof.* (a) The set $$\{G(v):v\in C\}$$ is the image of a semialgebraic set under a polynomial map with rational coefficients, hence is semialgebraic and defined over $$\mathbb Q$$ (Tarski–Seidenberg). By hypothesis it is the single point $$g$$. The statement "this set is nonempty" holds over $$\mathbb R$$, hence over the real closed field $$\mathbb R_{\rm alg}$$ of real algebraic numbers by Tarski transfer, so the set contains a point with algebraic coordinates; that point is $$g$$. (b) For timelike vectors $$e^{\eta}=\cosh\eta+\sinh\eta=-g_{ij}+\sqrt{g_{ij}^2-1}$$ is algebraic by (a). If $$\eta\neq0$$ were algebraic, $$e^\eta$$ would be transcendental by the Hermite–Lindemann theorem. For spacelike vectors, $$e^{i\theta}=g_{ij}+i\sqrt{1-g_{ij}^2}$$ is algebraic, and the same theorem excludes an algebraic $$\theta\neq0$$; $$\theta=0$$ is excluded by $$\lvert g_{ij}\rvert<1$$.
</div>

*Examples.* Two timelike unit vectors with rational coordinates, $$u=(0,0,0,1)$$ and $$v=(\tfrac34,0,0,\tfrac54)$$, have $$\langle u,v\rangle=-\tfrac54$$, so $$\cosh\eta=\tfrac54$$, $$\sinh\eta=\tfrac34$$ and $$\eta=\ln2$$. The pair is a single Lorentz orbit defined over $$\mathbb Q$$, and its rapidity is transcendental by (b), as is every nonzero rapidity or angle between vectors with algebraic Gram entries.

A richer family comes from circle packings. A disc on the celestial sphere is represented by a unit spacelike vector $$v\in\mathbb R^{3,1}$$, the normal of the hyperbolic plane bounded by its circle, oriented towards the disc. Two discs are tangent when $$\langle v,w\rangle=-1$$ and disjoint when $$\langle v,w\rangle=-I$$ with the *inversive distance* $$I>1$$, the hyperbolic cosine of the distance between the corresponding hyperbolic planes. By the Koebe–Andreev–Thurston theorem a triangulation $$T$$ of the sphere has a circle packing with tangency graph $$T$$, unique up to Möbius maps, that is, up to the Lorentz group. The set $$C$$ of packings with prescribed tangencies is defined by rational equations and inequalities, so Proposition 4(a) applies and the inversive distances of non-adjacent pairs are algebraic. The parameter count matches: $$N$$ discs have $$3N$$ parameters, a triangulation of the sphere has $$3N-6$$ edges and so imposes $$3N-6$$ contacts, and $$6=\dim SO(3,1)$$.

For a stacked triangulation (a vertex repeatedly inserted into a face) the inversive distances are integers. Start from four mutually tangent discs, whose Gram matrix $$2\cdot\mathbf 1-J$$ (diagonal $$1$$, off-diagonal $$-1$$) has eigenvalues $$(-2,2,2,2)$$. The disc inserted in the face $$p,q,r$$, on the side away from the fourth disc $$s$$, is $$w=2(p+q+r)-s$$, for which $$\langle w,w\rangle=1$$, $$\langle w,p\rangle=\langle w,q\rangle=\langle w,r\rangle=-1$$ and $$\langle w,s\rangle=-7$$. All vectors are integer combinations of the initial basis, so all Gram entries are integers. Enumerating all $$322560$$ insertion sequences that give $$10$$ vertices and $$24$$ edges, every inversive distance is an integer of at least $$7$$; in $$1536$$ of the sequences the set of distances is $$\{7,17,31,49,97,199,287,577\}$$. For the bipyramid on an $$n$$-gon the ring discs have angular radius $$\pi/n$$ and the two polar discs $$\pi/2-\pi/n$$, and the inversive distance between the polar discs is $$I=1+2\tan^2(\pi/n)=(3-c)/(1+c)$$ with $$c=\cos(2\pi/n)$$. This is a Möbius function of $$c$$ with rational coefficients, so $$\mathbb Q(I)=\mathbb Q(\cos2\pi/n)$$; for $$n=3,\dots,8$$ the values are $$7$$, $$3$$, $$11-4\sqrt5=2.0557$$, $$5/3$$, $$1.4638$$, $$7-4\sqrt2=1.3431$$.

The proposition restricts the Gram data of a rigid configuration over $$\mathbb Q$$ to algebraic numbers. It does not restrict the transcendental quantities computed from them: angles and rapidities between its vectors are transcendental by (b), and volumes and other integrals of the configuration are not covered. It also does not apply to a configuration that is rigid over a number field other than $$\mathbb Q$$, whose invariants lie in that field. A single configuration is one orbit of the Lorentz group, on which an average is not normalisable, and the proposition says nothing about the Poisson sprinkling, which is not a rigid configuration. The relation to exact discrete Lorentz symmetry is in [Exact discrete Lorentz symmetry cannot carry propagation](/quantum-gravity/discrete-lorentz-group/) and [Lorentz-invariant point sets and the structures they cannot carry](/quantum-gravity/invariant-point-sets/).

## 6. The cosmological constant and the volume degree of the horizon count

A cosmological constant would appear as a term of the count proportional to a volume. Let $$\ell$$ be the size of the ball, so that $$\omega\ell^n$$ is the volume of $$B$$, and let a term of the expected count carry a curvature factor of dimension $$D$$ (a power of inverse length). Its *area degree* is $$\ell^{d-2+D}$$. The *ball-volume degree* is $$\ell^{d-1+D}$$, and the *spacetime-volume degree* is $$\ell^{d+D}$$. At $$D=0$$ they are the terms proportional to the area of $$\partial B$$, to the volume of $$B$$, and to the volume of its domain of dependence, and the last is the form of a cosmological-constant term.

<div class="theorem" markdown="1">
**Theorem 1** (no volume degree). Let $$d\ge3$$ and $$n=d-1$$. Take the metric $$g=W(t)^{2/d}(-dt^2+\gamma_A)$$ with $$\gamma_A$$ the round metric of radius $$A$$ on the $$n$$-sphere, $$W(t)=1+\sum_{k\ge1}W_kt^k$$, $$\Sigma=\{t=0\}$$, and $$B$$ a geodesic ball in $$\Sigma$$ of fixed volume $$\omega\ell^n$$. Expand the expected weighted horizon count $$\langle H_w\rangle$$ as a formal power series in $$A^{-2}$$ and the $$W_k$$; a term of order $$A^{-2\nu}W_{k_1}\cdots W_{k_j}$$ has curvature dimension $$D=2\nu+k_1+\dots+k_j$$. For every weighting $$w$$, every term of curvature dimension $$D$$ is a multiple of $$\ell^e$$ with $$e\le d-2+D$$. In particular there is no term of ball-volume or spacetime-volume degree, at any order.
</div>

The expansion is that of the area-projector page, from which the proof takes its input: by its Lemma 1,

$$
\langle H_w\rangle=\int_0^\infty d\tau\,W(-\tau)\,\phi\big(V(\tau)\big)\,U(\tau),\qquad U(\tau)=\int_0^\tau ds\,W(s-\tau)\,{\rm cap}(s)\big[{\rm cap}(r_0+\tau)-{\rm cap}(r_0+\tau-s)\big],
$$

where $$\phi(v)=e^{-v}\sum_mw_mv^m/m!$$, $$V(\tau)$$ is the volume of the future of a point at depth $$\tau$$ below $$\Sigma$$, $${\rm cap}(r)$$ is the $$\gamma_A$$-volume of a geodesic ball of radius $$r$$, and $$r_0$$ is the radius of $$B$$. The size $$\ell$$ enters only through $$r_0$$ inside the bracket.

<div class="proof" markdown="1">
*Proof.* The cap volume is $${\rm cap}(r)=\sum_\nu\omega_\nu A^{-2\nu}r^{n+2\nu}$$ with $$\omega_0=\omega$$, and fixing the volume of $$B$$ gives $$r_0=\ell\,\psi(\ell^2/A^2)$$ with $$\psi(0)=1$$. The bracket expands as $$\sum_{i\ge1}{\rm cap}^{(i)}(r_0)[\tau^i-(\tau-s)^i]/i!$$, with no term $$i=0$$, and the part of order $$A^{-2\nu}$$ of $${\rm cap}^{(i)}(r_0)$$ is a constant times $$\ell^{\,n+2\nu-i}\le\ell^{\,n-1+2\nu}$$. All other factors are independent of $$\ell$$. A term of order $$A^{-2\nu}$$ has $$D\ge2\nu$$, so $$e\le n-1+2\nu=d-2+2\nu\le d-2+D$$.
</div>

The theorem uses no property of the weights; it is a statement about the counting object. For first-order perturbations of the metric that are not of this form, rim localisation suggests the same bound: a molecule exists only if its lower point lies within its own depth of the horizon, and all points of the molecule lie within that depth of it. This is a heuristic, and the degree bound for general perturbations is not proved here.

The mean of the horizon count therefore contains no term that could play the role of a cosmological constant, whatever the weights. A cosmological constant enters the count only through the curvature of the background, inside the area-degree response (the response to $$R_{uu}$$ and to the area deficit of the area-projector page), and that response matters for balls of the size of the curvature radius.

The fluctuation of the count is a different matter. For a region $$R$$ let $$N_R$$ be the number of points in $$R$$.

<div class="proposition" markdown="1">
**Proposition 5** (covariance with the number of points). For any integrable functional $$F$$ of the sprinkling and any region $$R$$,

$$
{\rm Cov}(F,N_R)=\rho\int_R\mathbb E[\Delta_xF]\,dx.
$$

If $$\Delta_xF=0$$ outside $$R$$, then $${\rm Cov}(F,N_R)=\rho\,\partial_\rho\mathbb E_\rho F$$. For an exact weighting in $$d$$ dimensions, $$\mathbb E H_w\propto\rho^{(d-2)/d}$$, so $${\rm Cov}(H_w,N_R)=\tfrac{d-2}d\,\mathbb E H_w$$ and the regression coefficient of $$H_w$$ on $$N_R$$ is $$\frac{d-2}d\,\mathbb E H_w/(\rho\lvert R\rvert)$$. In particular $$H_w$$ and $$N_R$$ are uncorrelated in $$d=2$$.
</div>

<div class="proof" markdown="1">
*Proof.* By the Mecke equation, $$\mathbb E[F\,N_R]=\mathbb E\int_RF\,\eta(dx)=\rho\int_R\mathbb E[F(\eta+\delta_x)]dx=\rho\lvert R\rvert\,\mathbb E F+\rho\int_R\mathbb E[\Delta_xF]dx$$, and $$\mathbb E N_R=\rho\lvert R\rvert$$. For the second statement, a Poisson process of intensity $$\rho+\epsilon$$ is the sum of independent processes of intensities $$\rho$$ and $$\epsilon$$; adding the second to the first adds a point at $$x$$ with probability $$\epsilon\,dx$$ to first order, so $$\partial_\rho\mathbb E_\rho F=\int\mathbb E[\Delta_xF]dx$$. When $$\Delta_xF$$ vanishes outside $$R$$ this equals $$\int_R\mathbb E[\Delta_xF]dx$$. For the horizon count the dependence region is not bounded, but by the estimate in the proof of Lemma 1 the contribution of depth greater than $$U$$ is of order $$e^{-\omega U^d/(2d)}$$ up to polynomial factors, so the identity holds up to that error for a region of depth $$U$$.
</div>

By Lemma 1, $${\rm Var}\,H_w\le c_w\mathcal A\rho^{(d-2)/d}$$ (with the $$\rho$$-scaling of the proof of Proposition 3). The Cauchy–Schwarz inequality gives $${\rm Var}\,H_w\ge{\rm Cov}(H_w,N_R)^2/{\rm Var}\,N_R=\big(\tfrac{d-2}d\mathbb E H_w\big)^2/(\rho\lvert R\rvert)$$ for a region $$R$$ of depth $$U$$ around the horizon, whose volume is of order $$\mathcal AU^2$$ in units of $$\ell_\rho$$; this is a lower bound of order $$\mathcal A\rho^{(d-2)/d}/U^2$$ up to constants. The standard deviation of the count is therefore of order the square root of the area, not of the volume.

*Estimate.* The comparison with the observed vacuum energy density uses a rule that we state as (H3): a fluctuating count $$X$$ plays the role that the number of points $$N$$ plays in the fluctuating cosmological constant of [[Ahmed et al.]](#ref-adgs), with $$\delta\Lambda\sim\Delta X/V_4$$ and $$V_4$$ the four-volume. For $$X=N$$ this gives $$\delta\Lambda\sim N^{-1/2}$$ in units of $$\ell_\rho$$, of the order of the observed value when the Hubble volume contains $$N\sim R_H^4$$ points. Set $$\ell_\rho$$ of the order of the Planck length, $$R_H=c/H_0\approx8.4\times10^{60}\,\ell_{\rm Pl}$$ for $$H_0\approx68$$ km s$$^{-1}$$ Mpc$$^{-1}$$ [[DESI]](#ref-desi), $$V_4\sim R_H^4$$, and $$X=H_w$$ with $$\Delta X\sim\sqrt S$$, where $$S=\pi R_H^2$$ is the horizon entropy in units of the count. Then $$\delta\Lambda\sim R_H/R_H^4=R_H^{-3}$$ while $$\Lambda_{\rm obs}\sim R_H^{-2}$$, a ratio of order $$\ell_{\rm Pl}/R_H\approx10^{-61}$$. The alternating weights of an exact weighting enlarge the standard deviation relative to $$\sqrt{\langle H_w\rangle}$$ by $$\sqrt{{\rm Var}H_w/\langle H_w\rangle}=\sqrt{a_4/r_w}\approx4\times10^3$$ for the four-dimensional weights (section 4), which brings the ratio to about $$5\times10^{-58}$$. The projection of the fluctuation on the direction of the number of points (Proposition 5) is smaller by a further factor of order $$S/N\sim(\ell_\rho/R_H)^2$$. In each case the ratio to the observed value is many orders of magnitude below one, because the horizon fluctuation is area-graded. The estimate rests on (H3) and on $$\ell_\rho\sim\ell_{\rm Pl}$$; what is proved is Theorem 1 and the area-graded bounds on the variance.

## 7. Open points

Selection rules among exact weightings supply numbers only once a rule is specified. For finite-degree weightings the minimal variance at fixed degree $$K$$ is non-increasing in $$K$$, since the families are nested; Proposition 3 shows that the infimum over weightings of infinite support has no positive lower bound on $$v$$, and whether the minimal variance of finite degree tends to zero or to a positive limit is not known. The degree bound of Theorem 1 is proved for conformally static backgrounds and only argued for general radial perturbations. Higher cumulants, and counts that are not molecule counts or layer actions, are not covered by any statement above. Theorem 1 concerns the formal expansion in the curvature; the behaviour of the full function of $$A$$ for balls comparable to the curvature radius is not covered. The comparison with the observed vacuum energy density depends on the rule (H3).

## References

1. <a id="ref-bd"></a>Benincasa, Dowker, “The scalar curvature of a causal set,” Phys. Rev. Lett. 104, 181301 (2010), [arXiv:1001.2725](https://arxiv.org/abs/1001.2725).
2. <a id="ref-ass"></a>Aslanbeigi, Saravani, Sorkin, “Generalized causal set d'Alembertians,” JHEP 1406, 024 (2014), [arXiv:1403.1622](https://arxiv.org/abs/1403.1622).
3. <a id="ref-belenchia"></a>Belenchia, “Universal behaviour of generalized causal set d'Alembertians in curved spacetime,” Class. Quantum Grav. 33, 135011 (2016), [arXiv:1510.04665](https://arxiv.org/abs/1510.04665).
4. <a id="ref-barton"></a>Barton, Counsell, Dowker, Gould, Jubb, Taylor, “Horizon molecules in causal set theory,” Phys. Rev. D 100, 126008 (2019), [arXiv:1909.08620](https://arxiv.org/abs/1909.08620).
5. <a id="ref-su"></a>Susskind, Uglum, “Black hole entropy in canonical quantum gravity and superstring theory,” Phys. Rev. D 50, 2700 (1994), [arXiv:hep-th/9401070](https://arxiv.org/abs/hep-th/9401070).
6. <a id="ref-adgs"></a>Ahmed, Dodelson, Greene, Sorkin, “Everpresent Lambda,” Phys. Rev. D 69, 103523 (2004), [arXiv:astro-ph/0209274](https://arxiv.org/abs/astro-ph/0209274).
7. <a id="ref-vz"></a>Verlinde, Zurek, “Modular fluctuations from shockwave geometries,” Phys. Rev. D 106, 106011 (2022), [arXiv:2208.01059](https://arxiv.org/abs/2208.01059).
8. <a id="ref-last"></a>Last, “Stochastic analysis for Poisson processes,” [arXiv:1405.4416](https://arxiv.org/abs/1405.4416).
9. <a id="ref-desi"></a>DESI Collaboration, “DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints,” [arXiv:2503.14738](https://arxiv.org/abs/2503.14738).
