---
title: A proof of the coth joint conjecture for polygonal regions
date: 2026-10-28
order: 2
description: For causal sets sprinkled into flat two-dimensional regions bounded by piecewise-linear spacelike or null curves, and into products of such regions with a plane in four dimensions, the mean Benincasa–Dowker action tends to the hyperbolic cotangent of the boost angle summed over the two joints. The same expansion gives the exact linear-order mean action of a curved two-dimensional causal interval.
---

The Benincasa–Dowker action is a proposed action for causal sets. For a causal set obtained by sprinkling points at density $$\rho$$ into a region of Minkowski space, its mean over sprinklings is expected to approach, as $$\rho\to\infty$$, the Einstein–Hilbert action plus a term carried by the joints, the codimension-two surfaces where the past and future boundaries of the region meet. Dowker, Liu and Lloyd-Jones conjectured that a joint at which the two boundaries meet at Lorentzian angle $$\theta$$ contributes $$\coth\theta$$ per unit joint area in Planck units. We prove this for two classes of flat regions. In two dimensions the class consists of all regions bounded below and above by piecewise-linear spacelike or null curves that meet only at their two endpoints, convex or not. In four dimensions it consists of the products of such a region with a flat transverse plane, per unit transverse area $$A$$. For these regions the mean action tends to $$\coth\theta_L+\coth\theta_R$$ in two dimensions and to $$(A/\ell_p^2)(\coth\theta_L+\coth\theta_R)$$ in four, with errors $$O(\rho^{-1})$$ and $$O(\rho^{-1/2})$$ respectively. In particular no logarithm of $$\rho$$ appears and the flat spacelike faces contribute nothing. Joints along which $$\theta$$ varies, curved faces, non-product regions in four dimensions and timelike boundaries are not covered. The same expansion gives a second result: for the two-dimensional metric $$(1+2\phi)\eta$$, at first order in $$\phi$$, every causal interval has mean action $$2+\tfrac12\int R\sqrt{-g}\,d^2x+O(\rho^{-1}\log\rho)$$, whatever the profile of the curvature inside the interval.

The action was introduced by Benincasa and Dowker [[BD]](#ref-bd). Buck, Dowker, Jubb and Surya [[BDJS]](#ref-bdjs) showed that for a causal interval in flat space of any dimension the mean tends to the volume of its joint in Planck units. Dowker [[D20]](#ref-d20) and Machet and Wang [[MW]](#ref-mw) found the Einstein–Hilbert term plus the joint volume for causal intervals in weakly curved spacetimes, at first order in a curvature that is constant over the interval to the order considered. Dowker, Liu and Lloyd-Jones [[DLLJ]](#ref-dllj) found that a joint between two spacelike boundaries contributes more than its volume and conjectured the weight $$\coth\theta$$ (their Conjecture 1′, eqs. 11 and 68, stated also for curved joints along which $$\theta$$ varies). They computed the mean by integrating over the separation vector $$c=y-x$$ of a pair of points, with weight the "volume of realisation" $$a(c)$$, the volume of the overlap of the region with its translate by $$-c$$. With this method they supported the conjecture with the two-dimensional lozenge and spacelike triangle, evaluated with an enlarged integration region valid at large $$\rho$$, and with $$d$$-dimensional cones of constant $$\theta$$, checked up to $$d=11$$. The function $$K$$ used below is their $$a(c)$$, written in null coordinates. We add an exact proof for the classes above, in which the two joints may have different angles; the identification of a joint's contribution with the mixed second derivative of $$a(c)$$ at zero separation, which is where the zeros of the action's Mellin polynomial enter; and error bounds that exclude logarithmic and face terms. The two-dimensional curved result extends [[D20]](#ref-d20) and [[MW]](#ref-mw) from approximately constant curvature to an arbitrary profile, at linear order.

The reader is assumed to know what a causal set and a Poisson sprinkling are. Signature is mostly plus, and the Lorentzian angle between two spacelike directions of rapidities $$\eta_1,\eta_2$$ is $$\lvert\eta_1-\eta_2\rvert$$.

## 1. The mean action

Sprinkle a region $$M$$ of a spacetime with a Poisson process of density $$\rho$$ and give the points the causal order. For $$k\ge0$$ let $$N_k$$ be the number of related pairs $$x\prec y$$ with exactly $$k$$ sprinkled points strictly between them, so that $$N_0$$ counts links, and let $$N$$ be the number of points. The two- and four-dimensional actions are [[DLLJ]](#ref-dllj, App. A)

$$
S_2=2\,(N-2N_0+4N_1-2N_2),\qquad S_4=\frac{4}{\sqrt6}\Big(\frac{\ell}{\ell_p}\Big)^2\big(N-N_0+9N_1-16N_2+8N_3\big),
$$

where $$\ell=\rho^{-1/4}$$ is the discreteness length and $$\ell_p^2=8\pi G$$. (The index $$k$$ here is $$i-1$$ in the notation of [[DLLJ]](#ref-dllj).) Let $$V_{xy}$$ be the volume of the causal interval between $$x\prec y$$. If $$M$$ is causally convex, which holds for every region in this note, the number of points inside that interval is Poisson with mean $$\rho V_{xy}$$, and the mean action is

$$
\langle S_d\rangle=c_d\Big[\rho V_M+\rho^2\!\int_M\!dV_x\!\int_{M\cap J^+(x)}\!dV_y\;F_d(\rho V_{xy})\Big], \tag{1}
$$

with $$c_2=2$$, $$c_4=(4/\sqrt6)(\ell/\ell_p)^2$$, and

$$
F_2(\lambda)=(-2+4\lambda-\lambda^2)\,e^{-\lambda},\qquad F_4(\lambda)=\big(-1+9\lambda-8\lambda^2+\tfrac43\lambda^3\big)\,e^{-\lambda}.
$$

Both terms in the bracket grow with $$\rho$$, and the finite limit comes from cancellations between them. Their bookkeeping is done by the Mellin transforms of the kernels,

$$
\int_0^\infty\lambda^{s-1}F_d(\lambda)\,d\lambda=\Gamma(s)\,Q_d(s),\qquad Q_2(s)=-(s-1)(s-2),\quad Q_4(s)=\tfrac13(2s-1)(s-1)(2s-3), \tag{2}
$$

which follow from $$\int_0^\infty\lambda^{s+k-1}e^{-\lambda}d\lambda=\Gamma(s)\,(s)_k$$. We use only the values

$$
Q_2'(1)=1,\quad Q_2'(2)=-1,\qquad Q_4(\tfrac12)=Q_4(1)=Q_4(\tfrac32)=0,\quad Q_4'(1)=-\tfrac13,\quad Q_4'(\tfrac32)=\tfrac23 .
$$

A term $$\rho^{p}$$ in the bracket of (1) comes with a factor $$Q_d$$ evaluated at a definite point, so terms land on zeros of $$Q_d$$ and drop out, except where the integrand also carries a logarithm, in which case $$Q_d'$$ survives. The joint terms are of this second kind: they come from logarithms produced by pairs of points that are close in both null directions at once, and the zero of $$Q_d$$ at the joint exponent ($$s=2$$ in two dimensions, $$s=\tfrac32$$ in four) removes the accompanying $$\log\rho$$ and leaves a finite coefficient. Section 2 reduces (1) to a one-variable integral, section 3 carries out this bookkeeping, and section 4 computes the coefficient.

## 2. Reduction to one variable

In two dimensions use null coordinates $$u=t-x$$ and $$w=t+x$$, so that $$dt\,dx=\tfrac12du\,dw$$. Two points are related, $$x\prec y$$, exactly when $$y-x=(a,b)$$ in null components with $$a,b\ge0$$, and then $$V_{xy}=ab/2$$. For a region $$P$$ in the plane define

$$
K(a,b)=\int du\,dw\;\mathbf 1_P(z)\,\mathbf 1_P\big(z+(a,b)\big),
$$

the $$du\,dw$$-area of the set of points $$z\in P$$ whose translate by $$(a,b)$$ is still in $$P$$. This is twice the volume of realisation $$a(c)$$ of [[DLLJ]](#ref-dllj), the factor coming from the area element. Since $$V_{xy}$$ depends only on $$y-x$$, the pair integral in (1) is $$\tfrac14\int_{a,b>0}K(a,b)\,F_2(\rho ab/2)\,da\,db$$. Grouping the separations by the product $$\mu=ab$$ gives

$$
\rho^2\!\int\!\!\int_{x\prec y}\!dV_xdV_y\,F_2(\rho V_{xy})=\frac{\rho^2}{4}\int_0^\infty H(\mu)\,F_2(\rho\mu/2)\,d\mu,\qquad H(\mu)=\int_0^\infty K\Big(a,\frac{\mu}{a}\Big)\frac{da}{a}. \tag{3}
$$

The function $$H$$ is the density of pairs whose causal interval has volume $$\mu/2$$.

In four dimensions take $$M=P\times T^2$$, where $$T^2$$ is a flat torus of area $$A$$ in the transverse coordinates $$x_\perp$$, large enough that no causal interval of $$M$$ wraps around it. (Equivalently, work per unit transverse area of $$P\times\mathbb R^2$$.) For two points with null separation $$(a,b)$$ in $$P$$ and transverse separation $$\Delta$$ the proper time satisfies $$\tau^2=ab-\lvert\Delta\rvert^2$$, and the interval volume is $$V=\tfrac{\pi}{24}\tau^4$$. The transverse integral can be done exactly: with $$Y=ab-\lvert\Delta\rvert^2$$,

$$
\int_{\lvert\Delta\rvert^2<ab}d^2\Delta\;F_4\Big(\frac{\pi\rho}{24}(ab-\lvert\Delta\rvert^2)^2\Big)=\pi\int_0^{ab}F_4\Big(\frac{\pi\rho}{24}Y^2\Big)dY .
$$

Write $$\varepsilon=(24/\pi\rho)^{1/2}$$ and $$\Psi(t)=\int_0^tF_4(y^2)\,dy$$. The right-hand side is $$\pi\varepsilon\,\Psi(ab/\varepsilon)$$, and inserting it into (1), with $$d^4x=\tfrac12du\,dw\,d^2x_\perp$$, gives the pair integral per unit transverse area as $$\tfrac{\rho^2}{4}\pi\varepsilon\int_0^\infty H(\mu)\Psi(\mu/\varepsilon)\,d\mu$$, that is,

$$
\frac{1}{A}\,\rho^2\!\int\!\!\int_{x\prec y}\!dV_xdV_y\,F_4(\rho V_{xy})=6\rho\int_0^\infty\Psi(t)\,H(\varepsilon t)\,dt, \tag{4}
$$

with the same $$H$$ as in (3). The four-dimensional problem for $$P\times T^2$$ is therefore the two-dimensional one with a different kernel. The kernel $$\Psi$$ decays like a Gaussian, because $$\Psi(\infty)=\tfrac12\Gamma(\tfrac12)Q_4(\tfrac12)=0$$ by (2).

## 3. The large-density expansion

Everything now depends on the behaviour of $$H(\mu)$$ as $$\mu\to0$$, that is, on pairs whose interval is small. The following lemma turns a small-$$\mu$$ expansion of $$H$$ into a large-$$\rho$$ expansion of the action. It uses nothing but the moments (2).

<div class="lemma" markdown="1">
**Lemma 1** (moments). Let $$f$$ be a continuous function on $$(0,\infty)$$ with $$\lvert f(\lambda)\rvert\le C(1+\lambda^n)e^{-\lambda}$$ for some $$n$$, and let $$m(s)=\int_0^\infty\lambda^{s-1}f(\lambda)\,d\lambda$$. Let $$H$$ be integrable on $$(0,\infty)$$ and satisfy, for all $$\mu>0$$,

$$
H(\mu)=\sum_{k=0}^{n_0}\mu^k\big(\alpha_k\log\tfrac1\mu+\beta_k\big)+R(\mu),\qquad\lvert R(\mu)\rvert\le C'\mu^{n_0+1}\big(1+\lvert\log\mu\rvert\big).
$$

Then, as $$Z\to\infty$$,

$$
\int_0^\infty H(\mu)f(Z\mu)\,d\mu=\sum_{k=0}^{n_0}Z^{-k-1}\Big[\alpha_k\big(m(k+1)\log Z-m'(k+1)\big)+\beta_k\,m(k+1)\Big]+O\big(Z^{-n_0-2}\log Z\big).
$$

The same holds for $$f=\Psi$$, whose Mellin transform is $$m_\Psi(s)=-\tfrac{1}{2s}\Gamma\big(\tfrac{s+1}{2}\big)Q_4\big(\tfrac{s+1}{2}\big)$$.
</div>

<div class="proof" markdown="1">
*Proof.* Substitute $$\mu=\lambda/Z$$. A term $$\mu^k\log(1/\mu)$$ gives $$Z^{-k-1}\int_0^\infty\lambda^k(\log Z-\log\lambda)f(\lambda)\,d\lambda=Z^{-k-1}\big(m(k+1)\log Z-m'(k+1)\big)$$, and $$\mu^k$$ gives $$Z^{-k-1}m(k+1)$$. The remainder is bounded by $$C'Z^{-n_0-2}\int_0^\infty\lambda^{n_0+1}(1+\lvert\log\lambda\rvert+\log Z)\lvert f(\lambda)\rvert\,d\lambda$$. For $$\Psi$$, which is bounded by a polynomial times $$e^{-t^2}$$, integrate by parts: $$\int_0^\infty t^{s-1}\Psi(t)\,dt=-\tfrac1s\int_0^\infty t^sF_4(t^2)\,dt=-\tfrac1{2s}\Gamma\big(\tfrac{s+1}2\big)Q_4\big(\tfrac{s+1}2\big)$$, the boundary terms vanishing because $$\Psi(0)=\Psi(\infty)=0$$.
</div>

The hypothesis that the bound on $$R$$ holds for all $$\mu$$, and not only for small $$\mu$$, costs nothing in our applications: there $$H$$ is bounded away from $$\mu=0$$ and vanishes for large $$\mu$$, while the sum grows only like $$\mu^{n_0}\log\mu$$.

We apply the lemma with $$n_0=1$$ to the two-dimensional pair integral (3), with $$f=F_2$$ and $$Z=\rho/2$$. By (2), $$m(1)=Q_2(1)=0$$, $$m(2)=Q_2(2)=0$$, $$m'(1)=Q_2'(1)=1$$ and $$m'(2)=Q_2'(2)=-1$$, so the constants $$\beta_0,\beta_1$$ and every $$\log\rho$$ drop out, and

$$
\frac{\rho^2}{4}\int_0^\infty H(\mu)F_2(\rho\mu/2)\,d\mu=-\frac{\rho}{2}\alpha_0+\alpha_1+O(\rho^{-1}). \tag{5}
$$

For the four-dimensional integral (4) take $$f=\Psi$$ and $$Z=1/\varepsilon$$, so that the left side of (4) is $$6\rho\varepsilon^{-1}\int_0^\infty H(\mu)\Psi(\mu/\varepsilon)\,d\mu$$. Here $$m_\Psi(1)=-\tfrac12Q_4(1)=0$$, $$m_\Psi(2)=-\tfrac14\Gamma(\tfrac32)Q_4(\tfrac32)=0$$, $$m_\Psi'(1)=-\tfrac14Q_4'(1)=\tfrac1{12}$$ and $$m_\Psi'(2)=-\tfrac18\Gamma(\tfrac32)Q_4'(\tfrac32)=-\tfrac{\sqrt\pi}{24}$$. Hence

$$
6\rho\int_0^\infty\Psi(t)H(\varepsilon t)\,dt=-\frac{\rho}{2}\alpha_0+\frac{\sqrt\pi}{4}\,\rho\,\varepsilon\,\alpha_1+O(1)=-\frac\rho2\alpha_0+\frac{\sqrt6}{2}\,\rho^{1/2}\alpha_1+O(1). \tag{6}
$$

The structure is the same in both dimensions. The constants $$\beta_k$$, which encode the size and shape of the region, never contribute, because they meet zeros of $$Q_d$$. The coefficient $$\alpha_0$$ of $$\log(1/\mu)$$ produces a term linear in $$\rho$$. The coefficient $$\alpha_1$$ of $$\mu\log(1/\mu)$$ is the only other input to the finite part. In Mellin language, $$\mu\log(1/\mu)$$ is a double pole, which the simple zero of $$Q_d$$ at the joint exponent lowers to a simple pole without a logarithm.

For the regions of this note $$\alpha_0=K(0,0)$$, which is the $$du\,dw$$-area of $$P$$, twice its volume $$V_P$$. The linear terms in (5) and (6) are then $$-\rho V_P$$ per unit transverse area, and they cancel the $$\rho V_M$$ in (1). What remains is

$$
\langle S_2\rangle=2\alpha_1+O(\rho^{-1}),\qquad \langle S_4\rangle=\frac{A}{\ell_p^2}\big(2\alpha_1+O(\rho^{-1/2})\big), \tag{7}
$$

where in four dimensions we used $$c_4\cdot\tfrac{\sqrt6}{2}\rho^{1/2}=2/\ell_p^2$$, since $$\ell^2\rho^{1/2}=1$$. These hold as soon as $$H(\mu)=\alpha_0\log\frac1\mu+\beta_0+\mu(\alpha_1\log\frac1\mu+\beta_1)+O(\mu^2)$$. It remains to show that $$H$$ has this form and to compute $$\alpha_1$$.

## 4. The overlap function near zero separation

<div class="definition" markdown="1">
**Definition** (graph region). A graph region is a set $$P=\{(t,x):x_L\le x\le x_R,\ f_-(x)\le t\le f_+(x)\}$$ where $$f_\pm$$ are continuous and piecewise linear with slopes in $$[-1,1]$$, $$f_-<f_+$$ on $$(x_L,x_R)$$ and $$f_-=f_+$$ at $$x_L$$ and $$x_R$$. The graphs of $$f_-$$ and $$f_+$$ are the past and future boundaries, and their common endpoints $$L$$ and $$R$$ are the joints. At a joint let $$v_-\ne v_+$$ be the slopes of the past and future edges that end there; the joint angle is $$\theta_J=\lvert\operatorname{artanh}v_+-\operatorname{artanh}v_-\rvert\in(0,\infty]$$, with $$\theta_J=\infty$$ if either edge is null, and $$\coth\infty=1$$.
</div>

Every edge is spacelike or null, and the joint angle is the Lorentzian angle of [[DLLJ]](#ref-dllj) between the tangents to the past and future boundaries. Every convex polygon with spacelike or null edges is a graph region, with its joints at the leftmost and rightmost vertices. The class also contains non-convex regions, such as the one in figure 1(a). A graph region is causally convex, because $$t-f_\pm(x)$$ is non-decreasing along any future-directed causal curve, so (1) applies with full causal diamonds as intervals.

The next lemma rests on a simple picture. The deficit $$\lvert P\rvert-K(h)$$ is the area of the points of $$P$$ whose translate by $$h$$ leaves $$P$$. For a future-timelike $$h$$ these points form a strip of width proportional to $$h$$ under the future boundary, whose area is linear in $$h$$, except near the two joints, where the strip runs into the past boundary and is cut off. Each cut-off piece is a triangle whose area is quadratic in $$h$$, and its mixed derivative gives the joint's contribution.

<div class="lemma" markdown="1">
**Lemma 2** (joint triangles). Let $$P$$ be a graph region and let $$\lvert P\rvert$$ be its $$du\,dw$$-area. At each joint $$J$$ let $$e_-$$ and $$e_+$$ be vectors along the past and future edges ending at $$J$$, written in null components as $$(-p_\pm,q_\pm)$$ with $$p_\pm,q_\pm\ge0$$. There are $$\delta>0$$ and a linear function $$\mathcal L(a,b)$$ such that for $$0\le a,b<\delta$$

$$
K(a,b)=\lvert P\rvert-\mathcal L(a,b)+T_L(a,b)+T_R(a,b),\qquad T_J(a,b)=\frac{(aq_++bp_+)(aq_-+bp_-)}{2\,\lvert p_+q_--q_+p_-\rvert}. \tag{8}
$$

In particular $$\partial_a\partial_bK=\tfrac12(\coth\theta_L+\coth\theta_R)$$ on $$[0,\delta)^2$$.
</div>

<div class="proof" markdown="1">
*Proof.* Fix $$h=(a,b)$$ with $$a,b>0$$, so that $$h$$ is future timelike, $$h_t>\lvert h_x\rvert$$ in $$(t,x)$$ components. Along the segment $$z+sh$$, $$s\ge0$$, both $$t-f_+(x)$$ and $$t-f_-(x)$$ increase strictly, because the slopes of $$f_\pm$$ are at most 1 in magnitude. Hence, for $$z\in P$$, the set of $$s\ge0$$ with $$z+sh\in P$$ is an interval $$[0,s^*]$$, and the exit point $$z+s^*h$$ lies on the future boundary. (If the segment reached $$x=x_R$$ inside $$P$$, then $$f_-(x_R)\le t\le f_+(x_R)=f_-(x_R)$$ there, so it would be at the joint $$R$$, which is on the future boundary.) The same argument applied to $$-h$$ shows that, for $$y$$ on the future boundary, the set of $$s\ge0$$ with $$y-sh\in P$$ is an interval $$[0,\sigma(y)]$$.

Let $$E(h)=\{z\in P:z+h\notin P\}$$, so that $$K(h)=\lvert P\rvert-\lvert E(h)\rvert$$. By the above, every $$z\in E(h)$$ is $$y-sh$$ for exactly one point $$y=(f_+(x),x)$$ of the future boundary and one $$s\in(0,1]$$ with $$s\le\sigma(y)$$, and every such point lies in $$E(h)$$. In the coordinates $$(x,s)$$ the area element is $$2\big(h_t-f_+'(x)h_x\big)\,dx\,ds$$ in $$du\,dw$$-area, which is positive. Therefore

$$
\lvert E(h)\rvert=\int_{x_L}^{x_R}2\big(h_t-f_+'(x)h_x\big)\,dx\;-\;\lvert E'(h)\rvert,\qquad E'(h)=\{y-sh:\ \sigma(y)<s\le1\}.
$$

The first term is linear in $$h$$; call it $$\mathcal L(h)$$. The set $$E'(h)$$ consists of points $$y-sh$$ that have crossed the past boundary. The two boundaries are polygonal and meet only at $$L$$ and $$R$$, at non-zero angles, so there is a constant $$\kappa$$ such that a point $$y$$ of the future boundary at distance more than $$\kappa\lvert h\rvert$$ from both joints has $$\sigma(y)>1$$. Thus $$E'(h)$$ lies within distance $$(\kappa+1)\lvert h\rvert$$ of the joints. Choose $$\delta$$ so small that within this distance of each joint, $$P$$ coincides with the wedge bounded by the two edges ending there. In the wedge at $$J$$, placed at the origin, $$E'(h)$$ is the triangle with vertices $$J$$, $$J-h$$ and the point $$X$$ where the line of the past edge meets the line through $$J-h$$ parallel to the future edge (figure 1(b)). Writing $$X=\xi e_-=-h+\zeta e_+$$ and taking the cross product $$e\times e'=e^ue'^w-e^we'^u$$ with $$e_+$$ gives $$\xi=-(h\times e_+)/(e_-\times e_+)$$, and the area is

$$
\tfrac12\lvert h\times X\rvert=\frac{\lvert h\times e_+\rvert\,\lvert h\times e_-\rvert}{2\,\lvert e_-\times e_+\rvert}.
$$

With $$e_\pm=(-p_\pm,q_\pm)$$ we have $$h\times e_\pm=aq_\pm+bp_\pm\ge0$$ and $$\lvert e_-\times e_+\rvert=\lvert p_+q_--q_+p_-\rvert$$, which is (8) for $$a,b>0$$; continuity of $$K$$ extends it to the axes. For the mixed derivative, $$\partial_a\partial_bT_J=\tfrac12(q_+p_-+p_+q_-)/\lvert p_+q_--q_+p_-\rvert$$. A spacelike edge of rapidity $$\eta$$ has $$(p,q)\propto(e^{-\eta},e^{\eta})$$, which gives $$\tfrac12(e^{\eta_+-\eta_-}+e^{\eta_--\eta_+})/\lvert e^{\eta_+-\eta_-}-e^{\eta_--\eta_+}\rvert=\tfrac12\coth\theta_J$$. A null edge has $$p=0$$ or $$q=0$$, and the ratio is then $$\tfrac12$$.
</div>

![Panel (a): a non-convex graph region P between a past boundary f− and a future boundary f+, both piecewise linear with spacelike edges, meeting at the joints L and R; the dashed curve is the future boundary translated by −h for a future-timelike vector h, and the strip between the two curves is the set of points whose translate leaves P. Panel (b): the neighbourhood of a joint J, with the past and future edges, the translated future edge (dashed) and the shaded triangle T_J with vertices J, J−h and the intersection of the translated future edge with the line of the past edge.](/assets/img/causal-set-action-corners-joint.svg){: .invertible}

*Figure 1.* (a) A graph region with a reflex kink in each boundary. Points of $$P$$ between the future boundary and its translate by $$-h$$ (dashed) leave $$P$$ when translated by $$h$$. (b) Near a joint the strip is cut off by the past edge; the excess is the triangle $$T_J$$, whose area is quadratic in $$h$$.

The lemma has two consequences used below. Kinks inside one boundary, convex or reflex, contribute nothing to $$\partial_a\partial_bK$$, because the strip passes them without being cut off. And $$K$$ is a single quadratic polynomial on a whole neighbourhood of the origin in the closed quadrant, whatever the direction of $$h$$, because no edge direction of $$P$$ lies inside the open quadrant. This is where the restriction to spacelike and null edges is used.

<div class="lemma" markdown="1">
**Lemma 3** (small intervals). For a graph region, with $$k_{11}=\tfrac12(\coth\theta_L+\coth\theta_R)$$,

$$
H(\mu)=\lvert P\rvert\log\frac1\mu+\beta_0+\mu\Big(k_{11}\log\frac1\mu+\beta_1\Big)+O(\mu^2)\qquad(\mu\to0).
$$
</div>

<div class="proof" markdown="1">
*Proof.* The overlap $$K(h)$$ is the area of the intersection of two polygons, one translated by $$h$$. On each cell of a finite decomposition of the $$h$$-plane into polygons (the cells on which the pattern of crossings between edges of $$P$$ and of $$P-h$$ is fixed), the vertices of the intersection move affinely with $$h$$, and its area is a quadratic polynomial in $$h$$. So $$K$$ is continuous, Lipschitz with some constant $$L_K$$, and piecewise quadratic; it vanishes outside a rectangle $$[0,a_{\max}]\times[0,b_{\max}]$$ in the quadrant. Write

$$
K(a,b)=K(0,0)+k_1(a)+k_2(b)+D(a,b),\qquad k_1(a)=K(a,0)-K(0,0),\quad k_2(b)=K(0,b)-K(0,0),
$$

and insert this into $$H(\mu)=\int_{\mu/b_{\max}}^{a_{\max}}K(a,\mu/a)\,da/a$$. The constant gives $$\lvert P\rvert\log(a_{\max}b_{\max}/\mu)$$. The term $$k_1$$ gives $$\int_0^{a_{\max}}k_1(a)\,da/a-\int_0^{\mu/b_{\max}}k_1(a)\,da/a$$; by Lemma 2, $$k_1$$ is a quadratic polynomial vanishing at 0 on $$[0,\delta)$$, so the second integral is a multiple of $$\mu$$ plus $$O(\mu^2)$$. The term $$k_2$$ is treated in the same way after the substitution $$b=\mu/a$$.

The mixed difference $$D$$ carries the logarithm. Split the integral at $$a=\mu/\delta$$ and $$a=\delta$$. On the middle range both $$a$$ and $$\mu/a$$ are below $$\delta$$, so $$D(a,\mu/a)=k_{11}\mu$$ by Lemma 2, giving $$k_{11}\mu\log(\delta^2/\mu)$$. On $$a>\delta$$ we have $$b=\mu/a<\mu/\delta$$. For fixed $$a$$, $$b\mapsto K(a,b)$$ is piecewise quadratic with second derivative bounded by a constant $$M_K$$, and its derivative jumps only where the vertical line through $$a$$ meets a non-vertical cell boundary. Such a meeting at a height in $$(0,b]$$ happens only for $$a$$ in a set of measure $$O(b)$$, because each of the finitely many cell boundaries either stays away from the axis $$b=0$$ near it, lies on it, or crosses it at a non-zero angle. Hence $$D(a,b)=b\,d(a)+r(a,b)$$, with $$d(a)=\partial_bK(a,0^+)-\partial_bK(0,0^+)$$ bounded and $$\lvert r(a,b)\rvert\le M_Kb^2+2L_Kb\,\mathbf 1_{S_b}(a)$$, $$\lvert S_b\rvert=O(b)$$. The range $$a>\delta$$ therefore contributes $$\mu\int_\delta^{a_{\max}}d(a)\,da/a^2+O(\mu^2)$$, and the range $$a<\mu/\delta$$ contributes the same with the roles of $$a$$ and $$b$$ exchanged. Collecting terms gives the claim.
</div>

If an edge of $$P$$ were timelike, its direction would lie inside the open quadrant, a cell boundary of $$K$$ would pass through the origin there, and $$K$$ would be quadratic only on sectors. Integrating over such a sector produces, in general, a term proportional to $$\mu^{1/2}$$ in $$H$$, and Lemma 1 then gives a contribution growing like $$\rho^{1/2}$$ in two dimensions and $$\rho^{1/4}$$ in four. This is the divergent timelike-face term of Conjecture 2 of [[DLLJ]](#ref-dllj). Regions with timelike edges are not treated here.

## 5. The theorem

<div class="theorem" markdown="1">
**Theorem 1** (coth joints for graph regions). Let $$P$$ be a graph region with joint angles $$\theta_L,\theta_R$$.

(a) For the two-dimensional action of a sprinkling into $$P$$,
$$\langle S_2\rangle=\coth\theta_L+\coth\theta_R+O(\rho^{-1})$$.

(b) For the four-dimensional action of a sprinkling into $$P\times T^2$$, with $$T^2$$ a flat torus of area $$A$$ large enough that no causal interval wraps around it,
$$\langle S_4\rangle=\dfrac{A}{\ell_p^2}\big(\coth\theta_L+\coth\theta_R+O(\rho^{-1/2})\big)$$.
</div>

<div class="proof" markdown="1">
*Proof.* By Lemma 3, $$H$$ has the expansion required in section 3 with $$\alpha_0=\lvert P\rvert$$ and $$\alpha_1=k_{11}=\tfrac12(\coth\theta_L+\coth\theta_R)$$, and the remainder bound holds for all $$\mu$$ by the remark after Lemma 1. Equation (7) gives both statements.
</div>

In the language of [[DLLJ]](#ref-dllj), the four-dimensional joints are the two flat surfaces $$\{L\}\times T^2$$ and $$\{R\}\times T^2$$, each of area $$A$$ and constant angle, so (b) is their eq. (68) for this class, with zero Einstein–Hilbert term. The routine part of the proof is Lemma 1, a bookkeeping of moments, and the tail estimate in Lemma 3. The content is in Lemma 2: the joints are the only places where the strip of departing points is cut off, and the cut-off triangle is what $$\coth\theta$$ measures.

The theorem reproduces the known cases. For the causal diamond both joints are null, and (a) gives $$2$$, the value of [[BDJS]](#ref-bdjs). For the lozenge of [[DLLJ]](#ref-dllj), their eq. (55) gives $$2n+1/(2n)$$ with $$n=\tfrac12\coth(\theta/2)$$ (their eq. 59), which equals $$2\coth\theta$$, and (a) gives the same. For their spacelike triangle the apex is a kink inside the future boundary, which contributes nothing, and each base vertex contributes $$\coth\theta$$, as found in their eq. (66).

As an independent check we computed $$K$$ exactly for several graph regions, by integrating the overlap of the vertical sections of $$P$$ and its translate, and compared the second difference at small $$(a,b)$$ with Lemma 2. The two agree to the accuracy of the finite differences ($$10^{-6}$$ or better), along directions $$b/a$$ from $$0.05$$ to $$20$$. For each region, $$K-T_L-T_R$$ is affine in $$h$$ near the origin to within $$10^{-12}$$. We also evaluated the exact finite-$$\rho$$ means (3) and (4) by quadrature, without the expansion of section 3:

| Region | $$\theta_L,\ \theta_R$$ | $$\tfrac12(\coth\theta_L+\coth\theta_R)$$ | $$\partial_a\partial_bK$$, numerical | $$\langle S_2\rangle$$ at $$\rho=10^3,\ 4\cdot10^3$$ |
|---|---|---|---|---|
| wedge closed by null edges | $$0.7,\ \infty$$ | 1.327311 | 1.327311 | |
| convex lens, two spacelike joints | $$0.5317,\ 0.7544$$ | 1.811353 | 1.811353 | 3.63334, 3.62535 |
| figure 1(a), reflex kinks in both boundaries | $$1.8921,\ 1.8921$$ | 1.046512 | 1.046512 | |
| causal diamond | $$\infty,\ \infty$$ | 1 | 1.000000 | 2.00003, 2.00000 |

For the lens, $$2k_{11}=3.62271$$ and the deviation falls by a factor of 4 when $$\rho$$ grows by 4, as the $$O(\rho^{-1})$$ error predicts.

## 6. Curved two-dimensional intervals

The bookkeeping of section 3 applies whenever the pair integral can be written as a weighted overlap function. Here we use it for a causal interval in a curved two-dimensional spacetime, at first order in the curvature. Every two-dimensional metric is locally conformally flat, so near flat space it suffices to consider $$g_\epsilon=(1+2\epsilon\phi)\eta$$ with $$\phi$$ smooth, and to differentiate in $$\epsilon$$ at $$\epsilon=0$$. The causal structure of $$g_\epsilon$$ is that of $$\eta$$, so the causal interval between two points is the null rectangle $$I=[u_0,u_0+L_u]\times[w_0,w_0+L_w]$$ for every $$\epsilon$$; we put $$u_0=w_0=0$$. Call its corners $$z_{\rm b}=(0,0)$$ (bottom), $$z_{\rm t}=(L_u,L_w)$$ (top), $$z_{\rm l}=(L_u,0)$$ (left) and $$z_{\rm r}=(0,L_w)$$ (right).

For $$g=e^{2\omega}\eta$$ the scalar curvature is $$R=-2e^{-2\omega}\Box_\eta\omega$$, positive for de Sitter space, and $$\Box_\eta=-4\partial_u\partial_w$$. At first order, therefore, $$R=8\epsilon\,\partial_u\partial_w\phi$$, $$\sqrt{-g}\,d^2x=\tfrac12du\,dw$$, and

$$
\frac12\int_IR\sqrt{-g}\,d^2x=2\epsilon\int_I\partial_u\partial_w\phi\,du\,dw=2\epsilon\big[\phi(z_{\rm t})+\phi(z_{\rm b})-\phi(z_{\rm l})-\phi(z_{\rm r})\big]+O(\epsilon^2).
$$

<div class="theorem" markdown="1">
**Theorem 2** (curved intervals at linear order). Let $$\phi$$ be smooth on a neighbourhood of $$I$$ and sprinkle $$I$$ at density $$\rho$$ with respect to the volume of $$g_\epsilon$$. Then

$$
\frac{d}{d\epsilon}\Big\vert_{\epsilon=0}\langle S_2(I)\rangle=2\big[\phi(z_{\rm t})+\phi(z_{\rm b})-\phi(z_{\rm l})-\phi(z_{\rm r})\big]+O(\rho^{-1}\log\rho).
$$

Equivalently, $$\langle S_2(I)\rangle=2+\tfrac12\int_IR\sqrt{-g}\,d^2x+O(\rho^{-1}\log\rho)+O(\epsilon^2)$$, up to terms exponentially small in $$\rho V_I$$.
</div>

The flat value is $$2(1-e^{-\rho V_I})$$ [[BDJS]](#ref-bdjs). The content of the theorem is that at linear order the curvature enters only through its integral, with coefficient $$\tfrac12$$, however it varies across the interval, and that it brings no $$\rho$$-growing term and no $$\log\rho$$. The proof has the same structure as before; the new ingredient is that the weights are smooth, so the expansion of $$H$$ is a Taylor expansion.

<div class="lemma" markdown="1">
**Lemma 4** (smooth weights). Let $$G$$ be smooth on $$[0,L_u]\times[0,L_w]$$ and $$H_G(\mu)=\int_{\mu/L_w}^{L_u}G(a,\mu/a)\,da/a$$. Then, as $$\mu\to0$$,

$$
H_G(\mu)=G(0,0)\log\frac1\mu+\beta_0+\mu\Big(\partial_a\partial_bG(0,0)\log\frac1\mu+\beta_1\Big)+O\Big(\mu^2\log\frac1\mu\Big).
$$
</div>

<div class="proof" markdown="1">
*Proof.* Decompose $$G=G(0,0)+g_1(a)+g_2(b)+ab\,\tilde G(a,b)$$ with $$g_1(a)=G(a,0)-G(0,0)$$, $$g_2(b)=G(0,b)-G(0,0)$$ and $$\tilde G(a,b)=\int_0^1\!\int_0^1\partial_a\partial_bG(\sigma a,\tau b)\,d\sigma\,d\tau$$. The first three pieces are treated as in Lemma 3 and give $$G(0,0)\log(L_uL_w/\mu)$$ plus a constant plus a multiple of $$\mu$$ plus $$O(\mu^2)$$. The last gives $$\mu H_{\tilde G}(\mu)$$. Applying the same decomposition to $$\tilde G$$ shows $$H_{\tilde G}(\mu)=\tilde G(0,0)\log(L_uL_w/\mu)+\text{const}+O(\mu\log\frac1\mu)$$, the last term because the mixed remainder of $$\tilde G$$ is bounded and $$\int_{\mu/L_w}^{L_u}da/a=O(\log\frac1\mu)$$. Finally $$\tilde G(0,0)=\partial_a\partial_bG(0,0)$$.
</div>

<div class="proof" markdown="1">
*Proof of Theorem 2.* Differentiate (1) with respect to $$\epsilon$$. The volume element is $$dV=\tfrac12(1+2\epsilon\phi)\,du\,dw$$, and the volume of the interval between $$x\prec y$$ is $$\tfrac12ab+\epsilon\,\delta V_{xy}+O(\epsilon^2)$$ with $$\delta V_{xy}=\int_{[x,y]}\phi\,du\,dw$$, the integral over the null rectangle spanned by $$x$$ and $$y$$. The derivative of the bracket in (1) is the sum of three terms:

$$
\rho\int_I\phi\,du\,dw,\qquad \frac{\rho^2}{4}\int_0^\infty H_{G_2}(\mu)F_2(\rho\mu/2)\,d\mu,\qquad \frac{\rho^3}{4}\int_0^\infty H_{G_3}(\mu)F_2'(\rho\mu/2)\,d\mu,
$$

coming from the number of points, from the volume elements of the pair, and from the change of the interval volume. Here, with $$z=(u,w)$$ and $$h=(a,b)$$,

$$
G_2(a,b)=2\!\int_{z,\,z+h\in I}\!\big[\phi(z)+\phi(z+h)\big]du\,dw,\qquad G_3(a,b)=\int_{z,\,z+h\in I}\!\delta V_{z,z+h}\,du\,dw .
$$

The weight $$G_2$$ is smooth, with $$G_2(0,0)=4\int_I\phi$$ and $$\partial_a\partial_bG_2(0,0)=2[\phi(z_{\rm t})+\phi(z_{\rm b})]$$. The second follows by differentiating the integration limits: $$\int_0^{L_u-a}\!\int_0^{L_w-b}\phi$$ contributes $$\phi(z_{\rm t})$$, and $$\int_a^{L_u}\!\int_b^{L_w}\phi$$ contributes $$\phi(z_{\rm b})$$. The weight $$G_3$$ vanishes on both axes: $$G_3=ab\,g_3$$ with $$g_3(a,b)=\int_0^1\!\int_0^1d\sigma\,d\tau\int_0^{L_u-a}\!\int_0^{L_w-b}\phi(u+\sigma a,w+\tau b)\,du\,dw$$. This is a product of an operation in $$u$$ and one in $$w$$. For a function $$\chi$$ of one variable on $$[0,L]$$, $$\frac{d}{da}\Big\vert_{a=0}\int_0^1d\sigma\int_0^{L-a}\chi(u+\sigma a)\,du=-\chi(L)+\tfrac12\big(\chi(L)-\chi(0)\big)=-\tfrac12\big(\chi(0)+\chi(L)\big)$$. Hence $$g_3(0,0)=\int_I\phi$$ and $$\partial_a\partial_bg_3(0,0)=\tfrac14\big[\phi(z_{\rm b})+\phi(z_{\rm t})+\phi(z_{\rm l})+\phi(z_{\rm r})\big]$$, and $$H_{G_3}(\mu)=\mu H_{g_3}(\mu)$$.

Apply Lemma 1 with $$Z=\rho/2$$. For the second term, $$f=F_2$$ and $$n_0=1$$, as in (5): it equals $$-\tfrac\rho2\cdot4\int_I\phi+2[\phi(z_{\rm t})+\phi(z_{\rm b})]+O(\rho^{-1}\log\rho)$$. For the third term, $$f=F_2'$$, whose Mellin transform is $$-(s-1)\Gamma(s-1)Q_2(s-1)=-\Gamma(s)Q_2(s-1)$$; it vanishes at $$s=2,3$$, with derivative $$-1$$ at $$s=2$$ and $$2$$ at $$s=3$$. With $$n_0=2$$ and the expansion $$\mu H_{g_3}(\mu)$$ from Lemma 4, the $$\mu$$ and $$\mu^2$$ terms give $$\rho\int_I\phi$$ and $$-4\,\partial_a\partial_bg_3(0,0)$$, and every constant and every $$\log\rho$$ again meets a zero. Adding the three terms, the parts linear in $$\rho$$ cancel, $$\rho-2\rho+\rho=0$$ times $$\int_I\phi$$, and the finite part is

$$
2[\phi(z_{\rm t})+\phi(z_{\rm b})]-\big[\phi(z_{\rm b})+\phi(z_{\rm t})+\phi(z_{\rm l})+\phi(z_{\rm r})\big]=\phi(z_{\rm t})+\phi(z_{\rm b})-\phi(z_{\rm l})-\phi(z_{\rm r}).
$$

Multiplying by $$c_2=2$$ gives the theorem. The neglected tails of the integrals are exponentially small in $$\rho L_uL_w$$.
</div>

For a metric that is a pure coordinate change at linear order, $$\phi=\chi_1(u)+\chi_2(w)$$, the right side vanishes, as it must: such a $$\phi$$ only changes the volume of the interval, and the mean action changes by an exponentially small amount. For $$\phi=\tfrac R8\,uw$$, constant curvature $$R$$ at linear order, the expansion terminates and gives, up to exponentially small terms,

$$
\langle S_2(I)\rangle=2+\frac{RV_I}{2}+\frac{2R}{\rho}\big(1-\gamma_E-\log\rho V_I\big)+O(R^2),
$$

with $$\gamma_E$$ Euler's constant. Since $$t^2=\tfrac14(u+w)^2$$ differs from $$\tfrac12uw$$ by a function of the form $$\chi_1(u)+\chi_2(w)$$, this is also the result for $$\phi\propto t^2$$, and it is the expression Dowker obtained for the interval with conformal factor $$1+bt^2$$ ([[D20]](#ref-d20), eqs. 3.15–3.18, with $$R=4b$$ and $$V_I=T^2/2$$), including the $$\rho^{-1}\log\rho$$ term. As an independent check of a case with non-constant curvature we took $$\phi=u^2w$$ on the unit square, for which $$R=16\epsilon u$$, and computed the full mean (1) for $$g_\epsilon$$ by direct four-dimensional quadrature at $$\epsilon=\pm10^{-3}$$, without linearising the interval volumes. At $$\rho=40$$ the central difference gives $$d\langle S_2\rangle/d\epsilon=0.970821$$, against $$0.970821$$ from the exact expansion; at this density the $$\rho^{-1}\log\rho$$ term is still more than half of the leading value 2. Exact symbolic evaluation for all monomials $$\phi=u^mw^n$$ with $$m,n\le3$$ gives no $$\rho^1$$ term, no $$\log\rho$$ at order $$\rho^0$$, and the finite part of the theorem.

## 7. Limitations and open questions

Theorem 1 covers joints that are flat, of constant angle, and adjacent to flat faces. Conjecture 1′ of [[DLLJ]](#ref-dllj) also concerns joints along which $$\theta$$ varies and boundaries with extrinsic curvature. For those, the overlap function near zero separation is no longer quadratic, and nothing in the present argument excludes a face term of order $$\rho^{1/4}$$ multiplying an integral of the extrinsic curvature, or a modification of the joint weight. The cones of [[DLLJ]](#ref-dllj), with curved joints of constant angle, remain the only evidence there. In four dimensions the reduction to two dimensions uses the product structure through the exact transverse integral; a general region requires the full four-dimensional overlap function near zero separation, and its mixed derivatives have not been computed.

Regions with timelike edges carry, besides the divergent face term, finite contributions from joints between spacelike and timelike edges. In the present framework the coefficient of $$\mu\log(1/\mu)$$ in $$H$$ is then the average of $$\partial_a\partial_bK$$ over the two sectors adjacent to the null axes (a sector bounded away from both axes contributes no logarithm), but we have not computed these averages in general or proved the remainder bounds in that case.

Theorem 2 is a statement at first order in the curvature and in two dimensions. In four dimensions the corresponding statement at first order, for small causal diamonds, is the result of [[MW]](#ref-mw); whether the joint term remains exactly the joint area for curvature that varies on the scale of the diamond, and how the subleading $$\log\rho$$ terms depend on the curvature, are not settled here.

## References

1. <a id="ref-bd"></a>Benincasa, Dowker, “The scalar curvature of a causal set,” [arXiv:1001.2725](https://arxiv.org/abs/1001.2725).
2. <a id="ref-bdjs"></a>Buck, Dowker, Jubb, Surya, “Boundary terms for causal sets,” [arXiv:1502.05388](https://arxiv.org/abs/1502.05388).
3. <a id="ref-d20"></a>Dowker, “Boundary contributions in the causal set action,” [arXiv:2007.13206](https://arxiv.org/abs/2007.13206).
4. <a id="ref-mw"></a>Machet, Wang, “On the continuum limit of Benincasa–Dowker–Glaser causal set action,” [arXiv:2007.13192](https://arxiv.org/abs/2007.13192).
5. <a id="ref-dllj"></a>Dowker, Liu, Lloyd-Jones, “Timelike boundary and corner terms in the causal set action,” [arXiv:2501.00139](https://arxiv.org/abs/2501.00139).
