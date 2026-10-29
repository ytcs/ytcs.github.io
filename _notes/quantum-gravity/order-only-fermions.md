---
title: Fermions that see only the causal order
date: 2026-10-29
category: matter
order: 1
description: A spinor field with one fixed spinor space at every element of a sprinkled causal set has no Lorentz-covariant kinetic term built from the order alone, and Kähler–Dirac fields on the span-restricted order complex have exact zero modes whose number grows faster than the number of elements (proved in expectation in two dimensions, measured in four).
status: Proved: the classification of order-natural covariant kernels (Theorem 1), the index theorem and the Euler characteristic of the span-restricted complex, the expected number of links in two dimensions, the private-face bound, the Hodge-type and degree-mass statements in two dimensions (the latter relative to the smallest nonzero eigenvalue of the vertex Laplacian), and the sign-change statement for shift-invariant Gaussian states on the order chain. Measured only: zero-mode growth in four dimensions, the vertex-Laplacian gap, the entanglement coefficient.
excludes: Covariant order-only kinetic terms for pointwise Weyl spinors with a global spinor space; Kähler–Dirac fermions with an operator that is self-adjoint for a degree-diagonal nondegenerate form and exchanges even and odd degrees, on the span-restricted order complex of a Poisson sprinkling (growth of the zero-mode count proved in expectation for d=2, measured for d=4); Hodge-type Wilson terms and a degree-one mass on the two-dimensional complex with the counting form; shift-invariant pure Gaussian states on the order chain.
---

Can a spinor field on a Poisson sprinkling have a Lorentz-covariant kinetic term built from the causal order alone? For a Weyl spinor attached to each element, with the same spinor space and the same Lorentz action at every element, the answer is no. If the quadratic form depends only on the order and is covariant, then in four dimensions every sesquilinear kernel vanishes, and the bilinear kernels are multiples of the antisymmetric matrix $$\epsilon$$, that is Majorana-type pairings without derivatives (Theorem 1). A first-order operator needs a vector-valued datum on each pair of elements, and the order supplies none.

A Kähler–Dirac field avoids this obstruction. Its components are cochains, functions on the chains $$x_0\prec\dots\prec x_k$$ of the order, and the coboundary $$\delta$$ is combinatorial. The difficulty moves to the zero modes. Any operator that exchanges even and odd degrees and is self-adjoint for a nondegenerate form diagonal in the degree has at least $$\lvert\chi\rvert$$ exact zero modes, with $$\chi$$ the Euler characteristic of the complex (Theorem 5). On the span-restricted complex, which keeps only chains with at most $$d-1$$ elements between their ends, the expected value of $$-\chi$$ on a sprinkling of a two-dimensional diamond is $$N(\ln N+\gamma-3)+O(\ln N)$$ for $$N$$ elements (Proposition 7), and sampling gives comparable growth in four dimensions. A zero mode of a kinetic operator is a massless mode, so this is a number of massless species that grows faster than $$N$$, against $$2^{d/2}$$ tastes in the continuum limit. The growth belongs to the span restriction: on the complex of all chains the mean index in two dimensions stays between $$0$$ and $$2$$ (Proposition 15).

In two dimensions, terms built from $$\delta$$ and its transpose, such as the Wilson term $$rK^{2}$$ that is added to give mass to unwanted modes, leave every harmonic zero mode exactly at zero (Proposition 10). A mass on the degree-one cochains lifts them to the mass scale (Proposition 11), and whether light modes remain then depends on the smallest nonzero eigenvalue of the vertex Laplacian, which is observed and not proved (section 4). In 1+1 dimensions a one-handed fermion exists on the order chain once a null direction is chosen, but every shift-invariant pure Gaussian state on the chain has as many upward as downward sign changes of its symbol (Proposition 13).

Matter on causal sets has been developed mostly for scalar fields, starting from Johnston's retarded propagators as sums over chains [[Johnston 2008]](#ref-johnston) (see also [[Johnston thesis]](#ref-jthesis), [[X–Dowker–Surya]](#ref-ads), and [[Surya]](#ref-surya) for a review). For spinors, Sverdlov builds vierbeins from holonomies and then spinor fields on them [[Sverdlov]](#ref-sverdlov), which supplies frame data at each element beyond the order. Dowker and Sorkin showed that a Poisson sprinkling carries no distinguished spatial or temporal orientation, no preferred lattice of timelike directions and no distinguished lattice of points [[Dowker–Sorkin]](#ref-ds). Kähler–Dirac fermions on triangulations were developed by Catterall and collaborators, who find zero modes that follow the topology of the triangulation [[Catterall–Laiho–Unmuth-Yockey]](#ref-clu) and relate the index to the Euler characteristic [[Butt et al.]](#ref-butt); a reduced version leaves mirror fermions [[Catterall]](#ref-catterall), and the Lorentzian version needs an operator incompatible with Lorentz transformations [[Catterall–Hubisz]](#ref-ch). What is added here is Theorem 1 with its hypotheses written out, the Euler characteristic of the span-restricted complex of a sprinkling with the expected number of links in two dimensions, the statements on Wilson terms and a degree mass in two dimensions, and the sign-change statement on the order chain.

Frames cannot be supplied by a sprinkling at random: [the page on preferred frames](/quantum-gravity/lorentz-violation-leakage/) shows that there is no Lorentz-invariant probability measure on frames, and [quantum sprinklings](/quantum-gravity/quantum-sprinklings/) and [Lorentz-invariant point sets](/quantum-gravity/invariant-point-sets/) describe the sprinklings themselves. Operators built without any frame, and what they can carry as chirality, are treated in [frame-free chirality](/quantum-gravity/frame-free-chirality/); frames used inside a Euclidean regulator are treated in [frames and reflection positivity](/quantum-gravity/frames-and-reflection-positivity/).

## 1. Causal sets, sprinklings and "order-only"

A causal set is a locally finite set $$P$$ with a strict partial order $$\prec$$. For $$x\prec y$$ the interval is $$I(x,y)=\{z:x\prec z\prec y\}$$, and $$x\prec y$$ is a link if $$I(x,y)=\varnothing$$. A sprinkling of density $$\rho$$ into Minkowski space $$\mathbb M^{d}$$ is a Poisson point process with intensity $$\rho$$, ordered by $$x\prec y$$ when $$y$$ lies in the causal future of $$x$$. For finite computations we sprinkle into a causal diamond. In two dimensions, with null coordinates $$(u,v)$$ scaled so that the diamond is the unit square, $$x\prec y$$ means $$u_x<u_y$$ and $$v_x<v_y$$.

Every orthochronous Poincaré transformation $$g$$ maps $$\mathbb M^{d}$$ to itself preserving the causal order, so it maps a point set $$P$$ to a point set $$gP$$ with an isomorphic causal order, the isomorphism being $$x\mapsto gx$$. A construction that uses only the order is therefore natural: what it assigns to a pair $$(x,y)$$ in $$P$$ equals what it assigns to $$(gx,gy)$$ in $$gP$$. This is the sense in which a causal-set construction is Lorentz invariant. If the construction also has to be covariant on spinors, naturality and covariance together constrain each value, and this is the question of the next section.

## 2. Weyl spinors and the Schur argument

A left-handed Weyl spinor in four dimensions is a pair $$\psi\in\mathbb C^{2}$$ on which $$S\in SL(2,\mathbb C)$$ acts as $$\psi\mapsto S\psi$$. The connection with Lorentz transformations is this. Identify a vector $$x=(x^{0},\vec x)$$ with the Hermitian matrix $$X=x^{0}\mathbb 1+\vec x\cdot\vec\sigma$$, so that $$\det X=(x^{0})^{2}-\lvert\vec x\rvert^{2}$$. The map $$X\mapsto SXS^{\dagger}$$ preserves Hermiticity and the determinant, hence is a Lorentz transformation $$\Lambda(S)$$, and $$S\mapsto\Lambda(S)$$ is a two-to-one cover of the proper orthochronous group $$SO^{+}(1,3)$$ with kernel $$\pm\mathbb 1$$. A rotation by $$2\pi$$ corresponds to $$S=-\mathbb 1$$, so spinors change sign. The matrix $$\psi\psi^{\dagger}$$ transforms as $$X$$ does, which is why the combinations $$\psi^{\dagger}\bar\sigma^{\mu}\psi$$, with $$\bar\sigma^{\mu}=(\mathbb 1,-\vec\sigma)$$, are the components of a vector. The kinetic term $$\psi^{\dagger}\bar\sigma^{\mu}\partial_\mu\psi$$ therefore requires a vector-valued object: the four Hermitian matrices $$\bar\sigma^{\mu}$$ form a basis of the space $$\mathrm{Herm}(2)\cong\mathbb R^{1,3}$$ of Hermitian $$2\times2$$ matrices. A mass term is a quadratic form in $$\psi$$ without a derivative. For anticommuting components the only candidate is $$\psi^{T}\epsilon\psi$$ with $$\epsilon$$ the antisymmetric matrix, a Majorana mass. In two dimensions a chiral Weyl spinor has one component, and a boost with rapidity $$\eta$$ acts as multiplication by $$e^{\eta/2}$$. In three dimensions the spinor group is $$SL(2,\mathbb R)$$ acting on $$\mathbb C^{2}$$.

We fix $$d\in\{2,3,4\}$$ and let $$V$$ be the spinor space, with $$S(g)$$ the matrix by which $$g$$, in the double cover of $$SO^{+}(1,d-1)$$, acts on it. A construction assigns to every locally finite point set $$P\subset\mathbb M^{d}$$ and every ordered pair $$(x,y)$$ of its elements a complex matrix $$\Phi(P)(x,y)$$ on $$V$$ with $$\Phi(P)(y,x)=\Phi(P)(x,y)^{\dagger}$$, so that the quadratic form $$\sum_{x,y}\psi_x^{\dagger}\Phi(P)(x,y)\psi_y$$ with $$\psi_x\in V$$ is real. A bilinear construction assigns instead matrices that define $$\sum_{x,y}\psi_x^{T}\Phi(P)(x,y)\psi_y$$. The construction is required to be natural for a group $$G$$ of Lorentz transformations, which is one of three: the full group $$SO^{+}(1,d-1)$$; the stabiliser of a region frame, here the two tips of the causal diamond in which we sprinkle, which is trivial for $$d=2$$, $$SO(2)$$ for $$d=3$$, and $$SO(3)$$ acting through $$SU(2)$$ for $$d=4$$; and the trivial group.

(H1) There is one copy of $$V$$ at every element, and a Lorentz transformation $$g$$ acts on all copies by the same matrix $$S(g)$$. This is a global trivialisation of the spinor bundle, that is a global frame.

(H2) $$\Phi$$ depends on the order alone: $$\Phi(gP)(gx,gy)=\Phi(P)(x,y)$$ for every $$g\in G$$.

(H3) The quadratic form is covariant: replacing $$P$$ by $$gP$$ and $$\psi_x$$ by $$S(g)\psi_x$$ at the element $$gx$$ leaves its value unchanged, which means $$S(g)^{\dagger}\Phi(gP)(gx,gy)S(g)=\Phi(P)(x,y)$$.

Combining (H2) and (H3) gives, for every $$g\in G$$, every $$P$$, $$x$$ and $$y$$,

$$
S(g)^{\dagger}\,\Phi(P)(x,y)\,S(g)=\Phi(P)(x,y), \tag{1}
$$

and the same with transposes for bilinear constructions. Each value of the kernel is therefore a $$G$$-invariant form on a single copy of $$V$$, and the problem reduces to representation theory.

<div class="theorem" markdown="1">
**Theorem 1** (order-natural covariant spinor kernels). Under (H1) to (H3) every value $$\Phi(P)(x,y)$$ is a form on $$V$$ invariant under $$G$$. The real dimensions of the spaces of invariant sesquilinear forms (complex matrices $$k$$ with $$S^{\dagger}kS=k$$), of the Hermitian ones among them, and of invariant bilinear forms ($$S^{T}kS=k$$) are

| $$d$$ | $$G$$ | sesquilinear | Hermitian | bilinear |
|---|---|---|---|---|
| 4 | $$SO^{+}(1,3)$$ | 0 | 0 | 2 |
| 4 | $$SU(2)$$ (region frame) | 2 | 1 | 2 |
| 4 | trivial | 8 | 4 | 8 |
| 3 | $$SO^{+}(1,2)$$ | 2 | 1 | 2 |
| 3 | $$SO(2)$$ (region frame) | 4 | 2 | 4 |
| 2 | $$SO^{+}(1,1)$$ | 0 | 0 | 0 |
| 2 | trivial (region frame) | 2 | 1 | 2 |

In particular, in four dimensions with $$G=SO^{+}(1,3)$$ every order-natural covariant sesquilinear kernel vanishes, and the order-natural covariant bilinear kernels are $$\Phi(P)(x,y)=b(P)(x,y)\,\epsilon$$ with $$b$$ a complex function of the pair that depends on $$P$$ through the order. With a region frame, a sesquilinear kernel is a complex scalar function times $$\mathbb 1$$ and distinguishes no helicity.
</div>

<div class="proof" markdown="1">
*Proof.* An invariant form is an intertwiner, and the Weyl representation has no invariant Hermitian pairing with itself because its conjugate is a different representation. For $$d=4$$ we give the computation for each row.

*Sesquilinear forms, $$G=SO^{+}(1,3)$$.* Write $$k=k_1+ik_2$$ with $$k_1,k_2$$ Hermitian. The map $$k\mapsto S^{\dagger}kS$$ preserves Hermiticity, so both parts are invariant. A Hermitian $$k_j$$ is a vector in $$\mathrm{Herm}(2)\cong\mathbb R^{1,3}$$, and $$k_j\mapsto S^{\dagger}k_jS$$ is the Lorentz transformation $$\Lambda(S^{\dagger})$$. As $$S$$ runs over $$SL(2,\mathbb C)$$ so does $$S^{\dagger}$$, so an invariant $$k_j$$ is a vector fixed by all of $$SO^{+}(1,3)$$. Rotations fix only vectors along the time axis, and a boost along $$x^1$$ moves $$(t,0,0,0)$$ to $$(t\cosh\eta,t\sinh\eta,0,0)$$. So $$k=0$$.

*Bilinear forms, $$G=SO^{+}(1,3)$$.* Take $$S=\mathrm{diag}(z,z^{-1})$$ with $$z\neq\pm1$$ in $$S^{T}kS=k$$: this gives $$k_{11}z^{2}=k_{11}$$ and $$k_{22}z^{-2}=k_{22}$$, so $$k_{11}=k_{22}=0$$. With $$k=\begin{pmatrix}0&b\\c&0\end{pmatrix}$$ and $$S=\begin{pmatrix}1&a\\0&1\end{pmatrix}$$ the $$(2,2)$$ entry of $$S^{T}kS$$ is $$ab+ca$$, which must vanish, so $$c=-b$$ and $$k=b\epsilon$$. Conversely $$S^{T}\epsilon S=\epsilon$$ for $$\det S=1$$. One complex parameter is two real.

*$$G=SU(2)$$.* Here $$S^{\dagger}=S^{-1}$$, so $$S^{\dagger}kS=k$$ says that $$k$$ commutes with $$SU(2)$$, hence is a complex scalar; the Hermitian ones are the real multiples of $$\mathbb 1$$. For bilinear forms $$S^{T}=\epsilon S^{-1}\epsilon^{-1}$$ for $$S\in SU(2)$$, so the condition says that $$\epsilon^{-1}k$$ commutes with $$SU(2)$$, hence is a scalar, and $$k\propto\epsilon$$.

*Trivial group.* All complex $$2\times2$$ matrices (8 real parameters), of which 4 are Hermitian.

For $$d=2$$ the space $$V=\mathbb C$$ carries the character $$S=e^{\eta/2}$$ of the boost, and invariance requires $$e^{\eta}k=k$$ for sesquilinear and for bilinear forms alike, so $$k=0$$ when $$G$$ contains boosts and $$k$$ is arbitrary otherwise. For $$d=3$$ the matrices $$S$$ are real, so sesquilinear and bilinear invariance coincide. The same computation as for $$SO^{+}(1,3)$$ with $$S=\mathrm{diag}(z,z^{-1})$$ and a real unipotent $$S$$ gives $$k=b\epsilon$$, $$b\in\mathbb C$$, which is Hermitian when $$b$$ is imaginary. For the rotation subgroup $$SO(2)$$ one has $$S^{T}=S^{-1}$$, so $$k$$ commutes with the rotations and $$k=a\mathbb 1+b\epsilon$$ with $$a,b\in\mathbb C$$, Hermitian when $$a$$ is real and $$b$$ imaginary. $$\square$$
</div>

The hypotheses carry the content of the theorem. (H1) is what fails if the spinor space at each element is allowed to depend on the element (local frames); in the continuum the same requirement makes a spinor kinetic term depend on a choice of tetrad in addition to the metric.

<div class="corollary" markdown="1">
**Corollary 2** (no order-natural Weyl operator in four dimensions). Let $$K$$ be an operator on fields given by $$(K\psi)_x=\sum_y\Phi(P)(x,y)\psi_y$$ with $$\Phi$$ satisfying (H1) to (H3) and $$G=SO^{+}(1,3)$$. Then $$\Phi=0$$ and $$K=0$$. With a region frame added, $$K$$ acts on each component of $$\psi$$ by the same scalar kernel, and no term in the expansion of $$K\psi$$ in derivatives of $$\psi$$ depends on its spin.
</div>

*Proof.* The first statement is Theorem 1. For the second, $$\Phi(P)(x,y)=\varphi(P)(x,y)\mathbb 1$$, so $$K$$ commutes with the rotation of the components, whereas a Weyl operator $$\vec\sigma\cdot\vec\nabla$$ does not. $$\square$$

*Example* (square roots). A Dirac-type operator is often sought as a square root of an order-natural scalar matrix $$\Phi$$ on the elements, for instance a retarded Green function. For any such matrix, with $$\sigma_\pm$$ the raising and lowering matrices of an auxiliary two-component space and $$\kappa\neq0$$, the matrix $$R_\kappa=-\rho\kappa\,\Phi\otimes\sigma_{+}+\kappa^{-1}\mathbb 1\otimes\sigma_{-}$$ satisfies $$R_\kappa^{2}=-\rho\,\Phi\otimes\mathbb 1$$, because $$\sigma_\pm^{2}=0$$ and $$\sigma_+\sigma_-+\sigma_-\sigma_+=\mathbb 1$$. The roots form a one-parameter family. In 1+1 dimensions a boost acts on the two components as $$\mathrm{diag}(e^{\eta/2},e^{-\eta/2})$$ and maps $$R_\kappa$$ to $$R_{\kappa e^{\eta}}$$. No root is boost invariant, and a boost moves one root along the family; fixing a root fixes a frame. This agrees with the $$d=2$$ rows of Theorem 1.

### Frames that are supplied

The obstruction is removed by data not contained in the order: a frame. Two statements concern such data.

<div class="proposition" markdown="1">
**Proposition 3** (no random global frame). Let $$\mathcal P$$ be a Poisson sprinkling of $$\mathbb M^{4}$$ and let $$F$$ be a random element of $$SL(2,\mathbb C)$$ (a global frame) jointly distributed with $$\mathcal P$$ so that the pair $$(\mathcal P,F)$$ has a law invariant under $$g:(P,F)\mapsto(gP,S(g)F)$$. There is no such law.
</div>

<div class="proof" markdown="1">
*Proof.* The map $$F\mapsto FF^{\dagger}$$ takes values in the positive Hermitian matrices of determinant 1, which form the hyperboloid $$Y$$ of future unit timelike vectors, and is equivariant: $$(S F)(SF)^{\dagger}=S(FF^{\dagger})S^{\dagger}=\Lambda(S)\cdot(FF^{\dagger})$$. The law of $$FF^{\dagger}$$ would be a Lorentz-invariant probability measure on $$Y$$, which does not exist by Proposition 6 of [the page on preferred frames](/quantum-gravity/lorentz-violation-leakage/). $$\square$$
</div>

A frame cannot therefore be a random variable attached to the sprinkling with an invariant joint law. It can be a fixed choice, which breaks Lorentz invariance, or it can be gauge data attached to each element, in which case the kernel depends on the frames at its two ends and (H1) fails. The second statement concerns link operators once a frame is given.

<div class="proposition" markdown="1">
**Proposition 4** (link operators with non-negative weights). Let $$v_n\in\mathbb M^{4}$$ be a finite set of future-causal vectors (displacements between linked elements), $$w_n\ge0$$ weights that are not all zero, and $$F(p)=-2\sum_nw_n\,\bar\sigma\!\cdot\!v_n\,\sin(p\!\cdot\!v_n)$$, which is $$i$$ times the symbol of the operator $$\psi(x)\mapsto\sum_nw_n\,\bar\sigma\!\cdot\!v_n[\psi(x+v_n)-\psi(x-v_n)]$$. Then $$T_{\mu\nu}=\sum_nw_nv_{n\mu}v_{n\nu}$$ is a non-zero positive semidefinite matrix, hence is not proportional to the Minkowski metric, and $$F(p)=-2\bar\sigma^{\mu}T_{\mu\nu}p^{\nu}+O(p^{3})$$. If the weights are rotation invariant about a unit timelike vector $$e_0$$, so that $$T=\mathrm{diag}(a,b,b,b)$$ in the frame of $$e_0$$, the infrared symbol has the relative sign of the time and space terms opposite to that of a covariant Weyl symbol $$\bar\sigma^{\mu}\eta_{\mu\nu}p^{\nu}$$, that is the opposite helicity, and its speed is $$b/a\le\tfrac13$$, with equality only if every $$v_n$$ is null.
</div>

<div class="proof" markdown="1">
*Proof.* $$T$$ is a sum of positive semidefinite rank-one matrices, and non-zero because $$v_n\neq0$$ for a causal vector. A multiple $$\lambda\eta$$ with $$\lambda\neq0$$ is indefinite, and $$\lambda=0$$ would make $$T=0$$. Expanding the sine gives the leading term. In signature $$(-+++)$$ with $$\bar\sigma^{\mu}=(\mathbb 1,-\vec\sigma)$$, $$\bar\sigma^{\mu}\eta_{\mu\nu}p^{\nu}=-(p^{0}+\vec\sigma\!\cdot\!\vec p)$$, whereas $$\bar\sigma^{\mu}T_{\mu\nu}p^{\nu}=ap^{0}-b\,\vec\sigma\!\cdot\!\vec p$$, so the relative sign of the two terms is reversed. The speed is $$b/a$$. Since each $$v_n$$ is causal, $$(v^{0}_n)^{2}\ge\lvert\vec v_n\rvert^{2}$$, so $$a=\sum w(v^{0})^{2}\ge\sum w\lvert\vec v\rvert^{2}=3b$$. $$\square$$
</div>

A covariant infrared limit therefore needs an indefinite coefficient matrix, which for link operators means signed weights or a frame-dependent twist. The proposition concerns the infrared symbol only. It makes no statement about doublers, whose location depends on the weights. Link operators on random points with a Wilson term are treated further in [frames and reflection positivity](/quantum-gravity/frames-and-reflection-positivity/).

## 3. Kähler–Dirac fields on the order complex

In flat space the exterior derivative $$\mathrm d$$ and its adjoint act on the space $$\Omega=\bigoplus_{k=0}^{d}\Omega^{k}$$ of differential forms, and $$K=\mathrm d+\mathrm d^{\dagger}$$ squares to the Hodge Laplacian. Exterior and interior products with a vector define a Clifford action on $$\Omega$$, so $$\Omega$$ is a sum of $$2^{d/2}$$ copies of the Dirac spinor representation, the tastes, and $$K$$ is the Dirac operator on this sum (Kähler). The parity $$\Gamma=(-1)^{k}$$ of the degree anticommutes with $$K$$. On a lattice or a triangulation the same construction uses cochains: $$C^{k}$$ is the space of functions on $$k$$-simplices and the coboundary $$\delta:C^{k}\to C^{k+1}$$ replaces $$\mathrm d$$ [[Catterall–Laiho–Unmuth-Yockey]](#ref-clu) [[Butt et al.]](#ref-butt). No frame is needed to define $$\delta$$, which is why Kähler–Dirac fields can escape Theorem 1: their components sit on tuples of elements and not in a fibre of a spinor bundle.

For a causal set $$P$$ the simplices are chains $$x_0\prec x_1\prec\dots\prec x_k$$, with the coboundary

$$
(\delta\omega)(x_0,\dots,x_{k+1})=\sum_{i=0}^{k+1}(-1)^{i}\,\omega(x_0,\dots,\widehat{x_i},\dots,x_{k+1}). \tag{2}
$$

The complex of all chains is not local, since a chain may contain elements far apart (section 6 gives its index). We restrict to chains whose first and last elements have at most $$n_0$$ elements between them, and call the resulting simplicial complex $$\Delta_{n_0}$$. A sub-chain of such a chain has a smaller span, so $$\Delta_{n_0}$$ is closed under faces. A chain with $$k+1$$ elements has at least $$k-1$$ elements inside its span, so $$\dim\Delta_{n_0}\le n_0+1$$ and the choice $$n_0=d-1$$ gives a complex of dimension at most $$d$$. For $$d=2$$ the complex $$\Delta_1$$ has as vertices the elements, as edges the links and the pairs $$x\prec z$$ with exactly one element $$y$$ in between, and as triangles the chains $$x\prec y\prec z$$ with $$I(x,z)=\{y\}$$.

The cochain spaces carry a nondegenerate symmetric bilinear form $$Q=\bigoplus_kQ_k$$, diagonal in the degree. The default is the counting form $$Q_k=\mathbb 1$$, for which the adjoint of $$\delta$$ is the transpose $$\delta^{T}$$. Another is the Krein form $$Q_k=(-1)^{k}\mathbb 1$$, which is indefinite and for which the adjoint is $$-\delta^{T}$$. We write $$\delta^{*}$$ for the adjoint and $$K=\delta+\delta^{*}$$.

<div class="theorem" markdown="1">
**Theorem 5** (index of parity-odd operators). Let $$\Delta$$ be a finite simplicial complex with cochain spaces $$C^{k}$$, let $$Q$$ be a nondegenerate symmetric bilinear form diagonal in the degree, and let $$K$$ be a linear operator on $$C=\bigoplus_kC^{k}$$ that maps even to odd degrees and odd to even degrees and is self-adjoint for $$Q$$. Then

$$
\dim\ker K|_{\mathrm{even}}-\dim\ker K|_{\mathrm{odd}}=\chi(\Delta)=\sum_k(-1)^{k}f_k, \tag{3}
$$

where $$f_k$$ is the number of $$k$$-simplices. Hence $$\dim\ker K\ge\lvert\chi(\Delta)\rvert$$. If $$Q$$ is positive definite and $$K=\delta+\delta^{*}$$, then $$\dim\ker K=\sum_kb_k$$, the sum of the Betti numbers.
</div>

<div class="proof" markdown="1">
*Proof.* Rank–nullity applied to the two halves of $$K$$ gives the index, and self-adjointness makes the two ranks equal. Write $$K_+=K|_{\mathrm{even}}:C^{\mathrm{ev}}\to C^{\mathrm{odd}}$$ and $$K_-=K|_{\mathrm{odd}}$$. For $$x$$ even and $$y$$ odd, self-adjointness gives $$\langle K_+x,y\rangle_Q=\langle x,K_-y\rangle_Q$$. Since $$Q$$ is nondegenerate on $$C^{\mathrm{ev}}$$, $$y\perp\mathrm{ran}\,K_+$$ if and only if $$K_-y=0$$: $$(\mathrm{ran}\,K_+)^{\perp}=\ker K_-$$. For a subspace $$W$$ of a space with a nondegenerate form $$\dim W^{\perp}=\dim C^{\mathrm{odd}}-\dim W$$, so $$\dim\ker K_-=\dim C^{\mathrm{odd}}-\mathrm{rk}K_+$$, while $$\dim\ker K_+=\dim C^{\mathrm{ev}}-\mathrm{rk}K_+$$. Subtracting gives $$\dim C^{\mathrm{ev}}-\dim C^{\mathrm{odd}}=\sum_k(-1)^{k}f_k$$, and $$\ker K=\ker K_+\oplus\ker K_-$$ gives the bound. For positive $$Q$$, $$\langle \delta\omega,\delta^{*}\eta\rangle=\langle \delta\delta\omega,\eta\rangle=0$$, so $$\lVert K\omega\rVert^{2}=\lVert \delta\omega\rVert^{2}+\lVert \delta^{*}\omega\rVert^{2}$$, $$\ker K=\ker \delta\cap\ker \delta^{*}$$, and Hodge theory identifies this with the cohomology. $$\square$$
</div>

Theorem 5 is the discrete McKean–Singer index relation. What matters for causal sets is the size of $$\chi$$. For a poset the Euler characteristic of the complexes $$\Delta_{n_0}$$ is computed from the Möbius function $$\mu$$, defined by $$\mu(x,x)=1$$ and $$\sum_{x\preceq z\preceq y}\mu(x,z)=0$$ for $$x\prec y$$. Hall's theorem says $$\mu(x,y)=\sum_{k\ge1}(-1)^{k}c_k(x,y)$$, where $$c_k$$ is the number of chains $$x=x_0\prec\dots\prec x_k=y$$ [[Wachs]](#ref-wachs).

<div class="corollary" markdown="1">
**Corollary 6** (Euler characteristic of the span-restricted complex). With $$N=\lvert P\rvert$$,

$$
\chi(\Delta_{n_0})=N+\sum_{\substack{x\prec y\\ \lvert I(x,y)\rvert\le n_0}}\mu(x,y), \tag{4}
$$

and $$\chi(\Delta_1)=N-\ell$$ where $$\ell$$ is the number of links.
</div>

<div class="proof" markdown="1">
*Proof.* The $$N$$ one-element chains contribute $$+1$$ each. Group the other simplices by their first and last elements $$x\prec y$$. If $$\lvert I(x,y)\rvert\le n_0$$ then every chain from $$x$$ to $$y$$ is a simplex of $$\Delta_{n_0}$$, since all its elements lie in $$I(x,y)$$, and the signed count is $$\mu(x,y)$$ by Hall's theorem. A link has the single chain $$x\prec y$$, so $$\mu=-1$$; a pair with one interior element $$z$$ has the chains $$x\prec y$$ and $$x\prec z\prec y$$, so $$\mu=-1+1=0$$. This gives (4) for $$n_0=1$$. The same follows from the face counts of $$\Delta_1$$: the triangles are in bijection with the one-interior pairs, because a triangle $$x\prec y\prec z$$ has $$I(x,z)=\{y\}$$, so with $$\tau$$ triangles $$f_0-f_1+f_2=N-(\ell+\tau)+\tau=N-\ell$$. $$\square$$
</div>

*Example.* Take the poset with four elements $$x\prec a,b\prec y$$, $$a$$ and $$b$$ unrelated. It has four links, and $$I(x,y)=\{a,b\}$$ has two elements. Then $$\chi(\Delta_1)=4-4=0$$. For $$n_0=2$$ the pair $$(x,y)$$ is admitted with $$\mu(x,y)=-(1-1-1)=1$$, and $$\chi(\Delta_2)=4-4+1=1$$, in agreement with the face count $$4-5+2=1$$ and with the fact that this complex is a cone, hence contractible.

### How large is the Euler characteristic on a sprinkling?

A link is a pair with an empty interval, and in a Lorentz-invariant sprinkling an element has unboundedly many links as the density grows [[Bombelli–Henson–Sorkin]](#ref-bhs). In two dimensions the expected number of links can be computed exactly.

<div class="proposition" markdown="1">
**Proposition 7** (expected links in two dimensions). Let $$N$$ points be independent and uniform in the unit square, ordered by the product order (a sprinkling of the causal diamond in null coordinates, with the number of points fixed). Then

$$
\mathbb E[\ell]=N\,H_{N-1}+H_N-2N+1,\qquad H_n=\sum_{k=1}^{n}\frac1k, \tag{5}
$$

and consequently $$\mathbb E[\chi(\Delta_1)]=-N\bigl(\ln N+\gamma-3\bigr)+O(\ln N)$$, where $$\gamma$$ is Euler's constant.
</div>

<div class="proof" markdown="1">
*Proof.* Count ordered pairs $$(i,j)$$. For $$i\neq j$$ the differences $$s=u_j-u_i$$ and $$t=v_j-v_i$$ are both positive with joint density $$(1-s)(1-t)$$ on the unit square, and given them $$(i,j)$$ is a link exactly when none of the other $$N-2$$ points falls in the rectangle of area $$st$$ between them, which has probability $$(1-st)^{N-2}$$. With $$m=N-2$$,

$$
\mathbb E[\ell]=N(N-1)\int_0^1\!\!\int_0^1(1-s)(1-t)(1-st)^{m}\,ds\,dt .
$$

Expand $$(1-s)(1-t)=1-s-t+st$$. The product $$p=st$$ of two uniform variables has density $$-\ln p$$, so $$\iint(1-st)^{m}=\int_0^1(-\ln p)(1-p)^{m}dp=H_{m+1}/(m+1)$$. Next $$\iint s(1-st)^{m}=\int_0^1\frac{1-(1-s)^{m+1}}{m+1}\,ds=\frac1{m+2}$$, and the same for $$t$$. Finally $$\iint st(1-st)^{m}=\int_0^1(-\ln p)\,p(1-p)^{m}dp=\frac{H_{m+2}-1}{(m+1)(m+2)}$$. With $$m+1=N-1$$ and $$m+2=N$$ the four terms add to

$$
\mathbb E[\ell]=N(N-1)\Bigl[\frac{H_{N-1}}{N-1}-\frac{2}{N}+\frac{H_N-1}{(N-1)N}\Bigr]=NH_{N-1}-2(N-1)+H_N-1,
$$

which is (5). The asymptotics follow from $$H_{N-1}=\ln N+\gamma+O(1/N)$$ and Corollary 6. $$\square$$
</div>

Equation (5) gives $$1/2$$ for $$N=2$$ (the two points are related with probability $$1/2$$, and then they are a link) and $$4/3$$ for $$N=3$$. Pairs of elements with an empty rectangle between them are therefore not rare, which is the discreteness-without-symmetry-breaking phenomenon in two dimensions. Table 1 gives sample means of $$-\chi/N$$ for $$\Delta_{d-1}$$ on sprinklings of a causal diamond, together with the exact expectation in two dimensions. In four dimensions $$\chi(\Delta_3)$$ was computed from (4) on points uniform in a causal diamond. The links per element are listed because they show the unbounded valency; for $$d=4$$ the quantity $$\chi(\Delta_3)$$ is not $$N-\ell$$, and its values come from (4).

| $$N$$ | 2D, sample | 2D, exact (5) | 2D links per element | 4D, sample | 4D links per element |
|---|---|---|---|---|---|
| 250 | 3.10 | 3.13 | 4.1 | 3.64 | 6.7 |
| 500 | 3.78 | 3.81 | 4.8 | 6.22 | 11.4 |
| 1000 | 4.54 | 4.49 | 5.5 | 9.01 | 17.6 |
| 2000 | 5.21 | 5.18 | 6.2 | 12.68 | 27.9 |
| 4000 | 5.90 | 5.87 | 6.9 | 16.85 | 42.1 |

*Table 1. The quantity $$-\chi(\Delta_{d-1})/N$$ on sprinklings of a causal diamond; each sample entry is the mean over five independent sprinklings.*

The standard deviation of $$\chi/N$$ between sprinklings is below 5% of its mean for $$N\ge500$$ in both dimensions. As a control, take the product order on the integer grid $$\{0,\dots,m\}^2$$, which is regular and not Lorentz invariant. It has $$N=(m+1)^2$$ points and $$\ell=2m(m+1)$$ links, so $$\chi(\Delta_1)=N-\ell=1-m^{2}$$, and $$-\chi/N\to1$$. For $$m=4,6,10$$ we find $$\chi(\Delta_2)=\chi(\Delta_3)=1$$. On the grid the span-restricted complex $$\Delta_1$$ therefore also has about $$N$$ exact zero modes, and a larger span removes them. On sprinklings of the unit square $$-\chi(\Delta_1)/N$$ grows like $$\ln N$$ instead of tending to $$1$$ (Proposition 7). For the larger spans, $$-\chi(\Delta_2)/N$$ is $$2.5$$ and $$2.9$$ at $$N=1000$$ and $$2000$$, while $$\chi(\Delta_3)/N$$ is $$0.0$$ and $$0.1$$. The growth in Table 1 refers to the span $$d-1$$, which gives a complex of dimension $$d$$.

<div class="corollary" markdown="1">
**Corollary 8** (zero modes of parity-odd operators). For every nondegenerate degree-diagonal $$Q$$ and every $$Q$$-self-adjoint operator $$K$$ that exchanges even and odd degrees on $$\Delta_{d-1}$$ of a sprinkling, in particular for $$\delta+\delta^{*}$$ with the counting or the Krein form, $$\dim\ker K\ge\lvert\chi(\Delta_{d-1})\rvert\ge-\chi(\Delta_{d-1})$$. In two dimensions, with $$N$$ uniform points, $$\mathbb E[\dim\ker K]\ge N(\ln N+\gamma-3)+O(\ln N)$$. In the samples of Table 1 at $$N=4000$$ this is at least about $$23\,600$$ exact zero modes in two dimensions, and in four dimensions, with $$-\chi/N=16.85$$, at least about $$67\,000$$.
</div>

For a region of fixed topology the continuum operator $$\mathrm d+\mathrm d^{\dagger}$$ has $$\sum_kb_k$$ zero modes, equal to $$1$$ for a contractible region, and a lattice version has $$2^{d/2}$$ tastes. The comparison with these numbers concerns the physical interpretation of the zero modes and not the theorem. The lower bound itself is Theorem 5; the growth of $$\chi$$ is proved in expectation for $$d=2$$ (Proposition 7) and measured for $$d=4$$.

## 4. Wilson terms and a degree mass in two dimensions

The index relation (3) holds for operators that anticommute with $$\Gamma$$. A Wilson term, such as $$rK^{2}$$, commutes with $$\Gamma$$, and the zero modes in (3) could in principle be lifted by it. We ask which terms of this kind lift them, and what is left. Take $$d=2$$ and the complex $$\Delta_1$$ with cochain spaces $$C^{0},C^{1},C^{2}$$, coboundaries $$\delta_0:C^{0}\to C^{1}$$ and $$\delta_1:C^{1}\to C^{2}$$, the counting form, $$K=\delta+\delta^{T}$$, and the Hodge Laplacians $$L_0=\delta_0^{T}\delta_0$$ and $$L_2=\delta_1\delta_1^{T}$$. We write $$b_k$$ for the Betti numbers of $$\Delta_1$$, $$\ell$$ for its number of links and $$\Pi_1$$ for the projector on $$C^{1}$$. Since $$K$$ has entries $$0,\pm1$$ whatever the density, lengths are measured in units of $$\rho^{-1/2}$$. A Dirac operator on a region of linear size $$R$$ in these units, with $$R^{2}\sim N$$, has its lowest nonzero eigenvalues of order $$1/R\sim N^{-1/2}$$, and we call a mode light if its eigenvalue or singular value is of this order or smaller.

The first statement is about the structure of $$\Delta_1$$ and holds for every finite poset.

<div class="proposition" markdown="1">
**Proposition 9** (private faces). For every finite poset, $$L_2=\delta_1\delta_1^{T}$$ on $$\Delta_1$$ satisfies $$L_2\ge\mathbb 1$$. Hence $$b_2=0$$, and the set of nonzero eigenvalues of $$K^{2}$$ is the union of the nonzero spectrum of $$L_0$$ and the spectrum of $$L_2$$, the latter lying in $$[1,\infty)$$.
</div>

<div class="proof" markdown="1">
*Proof.* Each triangle has an edge that belongs to no other triangle, so that $$\delta_1$$ contains an invertible block of signed permutations. Let $$t=(x,y,z)$$ be a triangle, so $$I(x,z)=\{y\}$$. The faces $$(x,y)$$ and $$(y,z)$$ are links, because $$I(x,y)$$ and $$I(y,z)$$ are subsets of $$I(x,z)\setminus\{y\}=\varnothing$$. The face $$(x,z)$$ has one interior element. A second triangle containing $$(x,z)$$ would either have that same interior element, hence be $$t$$, or have $$(x,z)$$ as one of its two short faces, impossible because $$(x,z)$$ is not a link. So one-interior edges and triangles are in bijection. Order the columns of $$\delta_1$$ as (one-interior edges, links). Then $$\delta_1=[P\,|\,B]$$ with $$P$$ a signed permutation matrix, and $$L_2=PP^{T}+BB^{T}=\mathbb 1+BB^{T}\ge\mathbb 1$$. For the second statement, $$K^{2}=L_0\oplus L_1\oplus L_2$$ with $$L_1=\delta_0\delta_0^{T}+\delta_1^{T}\delta_1$$, and since $$\delta_1\delta_0=0$$ the nonzero spectrum of $$L_1$$ consists of that of $$\delta_0\delta_0^{T}$$ (equal to that of $$L_0$$) and that of $$\delta_1^{T}\delta_1$$ (equal to that of $$L_2$$). $$\square$$
</div>

Proposition 9 says that no light mode comes from the sector of $$L_2$$. The remaining sector is that of the vertex Laplacian $$L_0$$, the graph Laplacian of the one-skeleton of $$\Delta_1$$. Its smallest nonzero eigenvalue $$\mu_0$$ is not bounded below on all posets: for a chain of $$n$$ elements it decreases like $$n^{-2}$$. On sprinklings of the unit square we computed $$\mu_0$$ for 20 sprinklings each at $$N=250$$, $$500$$ and $$1000$$ and for 6 at $$N=2000$$. The medians are $$2.8$$, $$3.0$$, $$3.3$$ and $$3.5$$, and the smallest values $$2.3$$, $$2.7$$, $$3.1$$ and $$3.3$$; one sprinkling at $$N=250$$ had an isolated element, and its value is the smallest nonzero eigenvalue. So $$\mu_0>2$$ in every sample, and it increases slowly with $$N$$. This is an observation and not a theorem, and the statements below keep $$\mu_0$$ explicit.

<div class="proposition" markdown="1">
**Proposition 10** (Hodge-type terms do not lift). Let $$W$$ be any element of the non-unital algebra generated by $$\delta$$ and $$\delta^{T}$$ (it contains $$rK^{2}$$) and $$\alpha,\beta\in\mathbb C$$. Then $$\ker(\alpha K+\beta W)\supseteq\ker K$$, and on $$\Delta_1$$ $$\dim\ker(\alpha K+\beta W)\ge\sum_kb_k=2b_0+\ell-N\ge\ell-N+2$$.
</div>

<div class="proof" markdown="1">
*Proof.* By Theorem 5, $$h\in\ker K$$ implies $$\delta h=0$$ and $$\delta^{T}h=0$$. Every word in $$\delta$$ and $$\delta^{T}$$ acts first with one of them, hence annihilates $$h$$. For the count, $$b_0-b_1+b_2=\chi=N-\ell$$ by Corollary 6, $$b_2=0$$ by Proposition 9, and $$b_0\ge1$$, so $$b_1=b_0+\ell-N$$ and $$\sum_kb_k=2b_0+\ell-N$$. $$\square$$
</div>

The proof uses that the Wilson term is built from the same operators as $$K$$. A term that is not built from them can act on the harmonic cochains, as the following mass does.

<div class="proposition" markdown="1">
**Proposition 11** (degree-one mass). Let $$A=[\delta_0\,|\,\delta_1^{T}]$$ as a map from $$C^{0}\oplus C^{2}$$ to $$C^{1}$$, and $$r>0$$. The singular values of $$iK+r\Pi_1$$ are $$0$$ with multiplicity $$b_0$$, $$r$$ with multiplicity $$\dim\mathrm{coker}A=b_1$$, and $$\tfrac12(\sqrt{r^{2}+4\mu}\mp r)$$ for each eigenvalue $$\mu$$ of $$L_0\oplus L_2$$ other than $$0$$, counted with multiplicity. Let $$\mu_*$$ be the smallest nonzero eigenvalue of $$L_0\oplus L_2$$; then $$\mu_*\ge\min\{\mu_0,1\}$$, and if $$b_1>0$$ the smallest nonzero singular value is
$$
\min\bigl\{r,\ \tfrac12\bigl(\sqrt{r^{2}+4\mu_*}-r\bigr)\bigr\}\ \ge\ \min\bigl\{r,\ \tfrac12\bigl(\sqrt{r^{2}+4\min\{\mu_0,1\}}-r\bigr)\bigr\}.
$$
</div>

<div class="proof" markdown="1">
*Proof.* Since $$\delta_1\delta_0=0$$, $$A^{T}A=L_0\oplus L_2$$. On each singular pair of $$A$$ with singular value $$s>0$$ the operator acts as $$\begin{pmatrix}0&is\\ is&r\end{pmatrix}$$ in a two-dimensional space, whose singular values have product $$s^{2}$$ and squares summing to $$2s^{2}+r^{2}$$, hence are $$\tfrac12(\sqrt{r^{2}+4s^{2}}\mp r)$$. The kernel of $$A$$ consists of the locally constant 0-cochains ($$\delta_1^{T}$$ is injective because $$L_2\ge\mathbb 1$$ by Proposition 9), and its cokernel is the space of harmonic 1-cochains, of dimension $$b_1$$, on which the operator is $$r$$. The smaller root $$\tfrac12(\sqrt{r^{2}+4\mu}-r)$$ increases with $$\mu$$, which gives the minimum, and the spectrum of $$L_2$$ lies in $$[1,\infty)$$. $$\square$$
</div>

The $$\ell-N+1$$ or more harmonic cochains that Proposition 10 leaves at zero are lifted to $$r$$, and the other singular values are at least $$\tfrac12(\sqrt{r^{2}+4\mu_*}-r)$$. If $$\mu_0\ge1$$, as observed, this is $$\tfrac12(\sqrt{r^{2}+4}-r)$$ or more; for $$r$$ of order one in the units above, no light mode remains. Without the gap on $$L_0$$, Proposition 11 gives no such lower bound.

{{CHECK}}

Four limits apply to Propositions 9 to 11.

1. The combination is an anti-Hermitian Kähler–Dirac operator $$iK$$ plus a Hermitian term. Other phases are not covered.
2. If $$W$$ is Hermitian and commutes with $$K$$, then on an eigenspace of $$K$$ with eigenvalue $$\lambda$$ the operator $$iK+W$$ is normal with singular values $$\sqrt{\lambda^{2}+w^{2}}\ge\lvert\lambda\rvert$$, so such a term cannot create light modes from modes of $$K$$ that are not light. A non-Hermitian $$\Gamma$$-even commuting $$W$$ is not covered.
3. Degree masses are covered for a mass proportional to $$\Pi_1$$ (Proposition 11). Other simplex-dependent masses are not examined.
4. The statements concern the two-dimensional complex $$\Delta_1$$ with the counting form. For the Krein form $$K$$ is replaced by $$\delta-\delta^{T}$$, and Proposition 10 holds with the same proof because $$\lVert(\delta-\delta^{T})h\rVert^{2}=\lVert\delta h\rVert^{2}+\lVert\delta^{T}h\rVert^{2}$$. Proposition 9 does not apply to the four-dimensional complex $$\Delta_3$$.

## 5. The order chain in 1+1 dimensions

In two dimensions the stabiliser of a causal diamond is trivial, and the $$d=2$$ rows of Theorem 1 allow order-natural covariant kernels once a region frame is fixed. A kinetic term for a one-component chiral spinor still needs a null direction, which is a binary choice. The kernel constructed below has entries $$\pm1$$ independent of the positions, so its quadratic form is invariant when the components transform as scalars and not with the weight $$e^{\eta/2}$$ of Theorem 1. It is therefore not an instance of (H1) to (H3), and the null direction is the extra datum.

Sprinkle $$N$$ points (with $$N$$ even) into the unit square in null coordinates and label them $$1,\dots,N$$ in increasing $$u$$. Let $$\mathcal C_{xy}=1$$ if $$x\prec y$$ and $$0$$ otherwise. The antisymmetric part $$\mathcal C-\mathcal C^{T}$$ of the causal matrix plays the role of the commutator function in the Sorkin–Johnston construction for scalar fields [[Surya]](#ref-surya). Since $$x\prec y$$ means that $$u$$ and $$v$$ both increase, $$(\mathcal C-\mathcal C^{T})_{xy}=\tfrac12[\mathrm{sgn}(u_y-u_x)+\mathrm{sgn}(v_y-v_x)]$$, that is $$\mathcal C-\mathcal C^{T}=-\tfrac12(\Sigma_u+\Sigma_v)$$, where $$(\Sigma_u)_{xy}=\mathrm{sgn}(u_x-u_y)$$ is the sign matrix of the linear order in $$u$$ and $$\Sigma_v$$ that of the order in $$v$$. The matrix splits into one piece for each null direction, each the sign matrix of a linear order. Choosing one of them, say $$\Sigma_u$$, which in the $$u$$-labelling has entries $$\mathrm{sgn}(x-y)$$, selects the chiral sector of fields depending on $$u$$. The choice is a binary datum: reflecting the diamond in space exchanges $$u$$ and $$v$$ and leaves the law of the sprinkling invariant, and no property of the order selects one side, in line with the absence of an intrinsic orientation [[Dowker–Sorkin]](#ref-ds).

Let $$U$$ be the antiperiodic shift, $$Ue_j=e_{j+1}$$ for $$j<N$$ and $$Ue_N=-e_1$$, so $$U^{N}=-\mathbb 1$$ and its eigenvalues are $$e^{i\theta_j}$$ with $$\theta_j=(2j-1)\pi/N$$. Then

$$
\Sigma_u=(\mathbb 1+U)(\mathbb 1-U)^{-1}, \tag{6}
$$

because $$U^{N}=-\mathbb 1$$ gives $$(\mathbb 1-U)^{-1}=\tfrac12\sum_{k=0}^{N-1}U^{k}$$, hence $$(\mathbb 1+U)(\mathbb 1-U)^{-1}=\sum_{k=1}^{N-1}U^{k}$$, and $$U^{k}e_j$$ is $$e_{j+k}$$ for $$j+k\le N$$ and $$-e_{j+k-N}$$ otherwise, so the $$(x,y)$$ entry is $$+1$$ for $$x>y$$ and $$-1$$ for $$x<y$$. The eigenvalues of the real antisymmetric matrix $$\Sigma_u$$ are $$i\cot(\theta_j/2)$$. The matrix $$\Sigma_u$$ acts on a function sampled at the points as $$N$$ times the integral operator with kernel $$\mathrm{sgn}(u-u')$$, which is the inverse of $$\tfrac12\partial_u$$. So $$\Sigma_u^{-1}$$ discretises $$\partial_u/2N$$, and we take $$D_u=-2iN\,\Sigma_u^{-1}$$ as the kinetic operator of a real (Majorana) one-component field with one chiral sector. Its eigenvalues are $$-2N\tan(\theta_j/2)$$, which for $$\theta_j\ll1$$ are close to $$-(2j-1)\pi$$, the spectrum of $$-i\partial_u$$ on antiperiodic functions of $$u\in[0,1]$$.

<div class="proposition" markdown="1">
**Proposition 12** (the chain derivative). For every even $$N$$ and every sprinkling, $$\mathrm{spec}\,D_u=\{-2N\tan(\theta_j/2)\}$$, with $$\theta_j$$ taken in $$(-\pi,\pi)$$. The function $$\theta\mapsto-2N\tan(\theta/2)$$ is strictly monotone on $$(-\pi,\pi)$$, so $$D_u$$ has a single zero of the symbol, at $$\theta=0$$, and $$\lambda_j=-(2j-1)\pi\,(1+O(\Lambda^{2}/N^{2}))$$ for the eigenvalues with $$\lvert\lambda_j\rvert<\Lambda\ll N$$. The inverse is $$\Sigma_u^{-1}=\sum_{k=1}^{N-1}(-1)^{k}U^{k}$$, so every off-diagonal entry of $$\Sigma_u^{-1}$$ is $$\pm1$$: all pairs of elements are coupled with the same weight.
</div>

<div class="proof" markdown="1">
*Proof.* $$U$$ is diagonalised by the Fourier vectors with eigenvalues $$e^{i\theta_j}$$, and the Cayley map $$e^{i\theta}\mapsto(1+e^{i\theta})/(1-e^{i\theta})=i\cot(\theta/2)$$ gives the spectrum of $$\Sigma_u$$. Inverting gives $$-i\tan(\theta/2)$$, and monotonicity is a property of $$\tan$$. For the inverse, $$\Sigma_u^{-1}=(\mathbb 1-U)(\mathbb 1+U)^{-1}$$ and $$(\mathbb 1+U)\sum_{k=0}^{N-1}(-U)^{k}=\mathbb 1-(-U)^{N}=2\cdot\mathbb 1$$ for even $$N$$ since $$U^{N}=-\mathbb 1$$. So $$\Sigma_u^{-1}=\tfrac12(\mathbb 1-U)\sum_{k=0}^{N-1}(-U)^{k}=\tfrac12\bigl[\mathbb 1+2\sum_{k=1}^{N-1}(-U)^{k}+(-U)^{N}\bigr]=\sum_{k=1}^{N-1}(-U)^{k}$$. $$\square$$
</div>

The operator is a non-local lattice derivative with a single zero of the symbol, at the cost of couplings of equal magnitude between all elements. The remaining question is the vacuum. A pure Gaussian state of the Majorana field is given by a real antisymmetric matrix $$J$$ with $$J^{2}=-\mathbb 1$$, through $$\langle\psi_r\psi_s\rangle=\tfrac12(\delta_{rs}+iJ_{rs})$$. The analogue of the Sorkin–Johnston prescription takes $$iJ$$ equal to the sign of the Hermitian matrix $$i\Sigma_u$$. We consider every state that is invariant under the shift $$U$$. A zero of the occupation function of such a state, a Fermi point, is a value of $$\theta$$ at which its symbol changes sign.

<div class="proposition" markdown="1">
**Proposition 13** (sign changes of shift-invariant states). Let $$J$$ be real antisymmetric with $$J^{2}=-\mathbb 1$$ and $$[J,U]=0$$. Then $$J$$ has a symbol $$f(\theta_j)\in\{+i,-i\}$$ which is an odd function of $$\theta$$. Consequently (i) $$f$$ changes sign between $$\theta=-\pi/N$$ and $$\theta=\pi/N$$ and between the two grid points adjacent to $$\theta=\pi$$; (ii) around the circle the number of upward jumps equals the number of downward jumps, so the net chirality $$c_L-c_R$$, defined as the signed number of jumps, vanishes for every such state. For the state with $$iJ=\mathrm{sgn}(i\Sigma_u)$$ the symbol is $$i\,\mathrm{sgn}\sin\theta$$, with exactly these two jumps and no others, which is the sign of the symbol $$i\sin\theta$$ of the nearest-neighbour Majorana hopping chain, whose Fermi points are at $$\theta=0$$ and $$\theta=\pi$$. The first jump is at the zero of the symbol of $$D_u$$; the second is at $$\theta=\pi$$, where $$D_u$$ has a pole and $$\Sigma_u$$ a zero.
</div>

<div class="proof" markdown="1">
*Proof.* A matrix commuting with $$U$$, which has distinct eigenvalues, is a polynomial in $$U$$ and is diagonalised by the same Fourier vectors; call its eigenvalue on the vector with phase $$\theta$$ the symbol $$f(\theta)$$. Reality of $$J$$ gives $$f(-\theta)=\overline{f(\theta)}$$, antisymmetry gives $$f(-\theta)=-f(\theta)$$, so $$f$$ is purely imaginary and odd, and $$J^{2}=-\mathbb 1$$ gives $$f\in\{\pm i\}$$. The grid $$\theta_j=(2j-1)\pi/N$$ is symmetric under $$\theta\to-\theta$$, with $$\pm\pi/N$$ adjacent at $$0$$ and $$\pm(N-1)\pi/N$$ adjacent at $$\pi$$ (using $$N$$ even), and oddness forces opposite signs at each pair. A cyclic sequence of $$\pm1$$ has as many upward as downward jumps. For the sign state, $$\mathrm{sgn}\cot(\theta/2)=\mathrm{sgn}\sin\theta$$ on $$(-\pi,\pi)\setminus\{0\}$$. $$\square$$
</div>

Proposition 13 is the lattice pairing of Fermi points for every invariant pure state, and it has a consequence for entanglement. For a conformal field theory the entanglement entropy of an interval of $$n$$ sites grows as $$\tfrac{c_L+c_R}{6}\ln n$$ [[Calabrese–Cardy]](#ref-cc), which is $$\tfrac16\ln n$$ for a non-chiral Majorana fermion ($$c_L=c_R=\tfrac12$$) and $$\tfrac1{12}\ln n$$ for a single Majorana–Weyl fermion. For the sign state at $$N=1000$$ and blocks of $$n=20,30,\dots,400$$ consecutive elements in the $$u$$-order, a fit of the entropy of the Gaussian state to $$b\ln[(N/\pi)\sin(\pi n/N)]+\text{const}$$ gives {{ENT}} The coefficient corresponds to two Fermi points, as in Proposition 13: the entanglement entropy of this state sees both chiralities. This is a numerical observation about one state. A one-handed vacuum would need a state that is not shift invariant, or interactions, which Proposition 13 does not address.

## 6. The full order complex and the Möbius function

If the span restriction is dropped, the complex $$\Delta(P)$$ of all chains is not local, but the index of Theorem 5 has a closed form, and in two dimensions its mean is bounded. This is the reason why the growth of section 3 belongs to the span restriction. Let $$\hat0$$ and $$\hat1$$ be the bottom and top of a causal diamond, $$P^{\circ}$$ the set of elements strictly between them, and $$\Delta(P^{\circ})$$ the order complex of $$P^{\circ}$$.

<div class="proposition" markdown="1">
**Proposition 14** (index of the open and closed interval). For every finite poset $$P^{\circ}$$, $$\chi(\Delta(P^{\circ}))=1+\mu(\hat0,\hat1)$$, so any operator as in Theorem 5 has $$\dim\ker K\ge\lvert1+\mu(\hat0,\hat1)\rvert$$ on $$\Delta(P^{\circ})$$. On the closed interval $$\Delta(P^{\circ}\cup\{\hat0,\hat1\})$$, a cone, $$\ker(\delta+\delta^{*})$$ is one-dimensional for the counting form.
</div>

<div class="proof" markdown="1">
*Proof.* Hall's theorem gives $$\mu(\hat0,\hat1)=\sum_{k\ge1}(-1)^{k}c_k$$, where $$c_k$$ is the number of chains $$\hat0\prec x_1\prec\dots\prec x_{k-1}\prec\hat1$$, that is the number of $$(k-1)$$-element chains of $$P^{\circ}$$ (for $$k=1$$, the empty chain). With $$a_j$$ the number of $$j$$-element chains of $$P^{\circ}$$ and $$a_0=1$$, $$\mu=\sum_{j\ge0}(-1)^{j+1}a_j$$ while $$\chi(\Delta(P^{\circ}))=\sum_{j\ge1}(-1)^{j-1}a_j=1+\mu$$. A cone has the cohomology of a point. $$\square$$
</div>

<div class="proposition" markdown="1">
**Proposition 15** (mean index in two dimensions). Sprinkle the diamond $$[0,1]^{2}$$ in null coordinates with a Poisson process of mean $$\bar N$$ points and take $$\hat0=(0,0)$$, $$\hat1=(1,1)$$. Then $$\mathbb E[\chi(\Delta(P^{\circ}))]=1-J_0(2\sqrt{\bar N})$$, with $$J_0$$ the Bessel function. Since $$\lvert J_0\rvert\le1$$, the mean index lies between $$0$$ and $$2$$ for every $$\bar N$$.
</div>

<div class="proof" markdown="1">
*Proof.* By the Mecke formula, the expected number of $$j$$-element subsets of a Poisson process with a given property is $$\bar N^{j}/j!$$ times the probability that $$j$$ independent uniform points have it. Sort $$j$$ uniform points by $$u$$; they form a chain exactly when their $$v$$-ranks are increasing, with probability $$1/j!$$. So $$\mathbb E[a_j]=\bar N^{j}/(j!)^{2}$$, and Proposition 14 gives $$\mathbb E[\mu]=-\sum_j(-\bar N)^{j}/(j!)^{2}=-J_0(2\sqrt{\bar N})$$. The bound is $$\lvert J_0\rvert\le1$$. $$\square$$
</div>

With $$\mathcal C$$ the causal matrix, $$(\mathbb 1+\mathcal C)^{-1}$$ has $$(\hat0,\hat1)$$ entry $$\mu$$ and is the sum over chains with weight $$-1$$ per step. Johnston used this matrix geometric series to define retarded propagators and showed that in two dimensions the propagator with weight $$-m^{2}/(2\rho)$$ per step has the mean $$\tfrac12J_0(m\tau)$$ between points at proper time $$\tau$$ [[Johnston 2008]](#ref-johnston); the same Bessel function is the two-dimensional massive Green function [[X–Dowker–Surya]](#ref-ads). The weight $$-1$$ corresponds to $$m^{2}=2\rho$$ and $$m\tau=2\sqrt{\bar N}$$, so $$-\mu/2$$ has the mean $$\tfrac12J_0(2\sqrt{\bar N})$$ of Proposition 15. Sampling the Poisson process gives $$\mathbb E[\mu]=-0.228\pm0.008$$ at $$\bar N=10$$ and $$0.190\pm0.020$$ at $$\bar N=30$$ (40000 samples each), against $$-0.229$$ and $$0.179$$. The root-mean-square of $$\mu$$ is $$1.5$$, $$4.1$$ and $$30$$ at $$\bar N=10,30,100$$; the last value is dominated by rare large samples and is less precise. Over this range $$\ln\mathrm{rms}\,\mu$$ is close to affine in $$\sqrt{\bar N}$$ with slope about $$0.4$$. This is an observation from three points and is not proved. On the closed interval the index is $$1$$ irrespective of $$\mu$$: the fluctuating index belongs to the open interval, which is the link of the edge $$\{\hat0,\hat1\}$$ in the closed complex.

## 7. Routes not excluded and open points

The results exclude constructions with the stated hypotheses. The following routes lie outside them, each with its condition.

- Local frames. If each element carries its own frame, as when vierbeins are built from holonomies [[Sverdlov]](#ref-sverdlov), (H1) fails and Theorem 1 does not apply. The kernel then depends on frame data attached to the elements or pairs, as gauge data. Proposition 3 excludes an invariant random global frame; whether a gauge-fixed description with frames per element is consistent with the order is not decided here.
- A given global frame or region frame. Theorem 1 with $$G=SU(2)$$ allows scalar kernels, and a kinetic term then uses the frame explicitly; the transfer of such a frame into low-energy physics is the subject of [the page on preferred frames](/quantum-gravity/lorentz-violation-leakage/).
- Constructions covariant only on average, or only approximately, are not covered by (H3).
- Kähler–Dirac fields with terms that break the parity of the degree and are not of Hodge type, Wilson terms that do not commute with $$K$$ and are not degree-diagonal, and degree masses other than $$\Pi_1$$.
- Inner products that are not degree-diagonal. A Lorentzian Whitney form is non-diagonal and indefinite, and Theorem 5 does not apply to it; Propositions 9 to 11 use positivity and the unit private-face entry and do not apply either.
- The complexes $$\Delta_{n_0}$$ with $$n_0\ge2$$ in two dimensions and the four-dimensional complex $$\Delta_3$$, for the Wilson-term statements. Proposition 10 holds on any complex, and the bound on $$b_2$$ in it uses Proposition 9, which is specific to $$\Delta_1$$.
- Regular lattices, on which $$\chi(\Delta_{n_0})=1$$ once $$n_0\ge2$$ in the control of section 3, at the cost of the Lorentz symmetry.
- Other states and interactions on the 1+1 chain, such as a mirror fermion decoupled by interactions or a state not invariant under the shift.
- Link operators with signed weights, to which Proposition 4 does not apply.

Several points are open. The growth law of the variance of the Möbius function of a random two-dimensional order, suggested by three sample values, is not known. The vertex-Laplacian gap of $$\Delta_1$$ has no proof. It is not known whether a Lorentzian inner product on the cochains of $$\Delta_1$$ yields region-scale modes, nor whether the private-face weight stays bounded below for it. The Kähler–Dirac construction on a Lorentzian manifold needs an operator that is incompatible with the Lorentz transformations [[Catterall–Hubisz]](#ref-ch), and the analogue of that condition on the order complex is not worked out. A four-dimensional version of the statements on Wilson terms, and the extension of Proposition 7 beyond two dimensions, are open.

## References

1. <a id="ref-johnston"></a>Johnston, “Particle propagators on discrete spacetime,” Class. Quantum Grav. 25 (2008) 202001, [arXiv:0806.3083](https://arxiv.org/abs/0806.3083).
2. <a id="ref-jthesis"></a>Johnston, “Quantum fields on causal sets,” Ph.D. thesis, [arXiv:1010.5514](https://arxiv.org/abs/1010.5514).
3. <a id="ref-ads"></a>Nomaan X, Dowker, Surya, “Scalar field Green functions on causal sets,” Class. Quantum Grav. 34 (2017) 124002, [arXiv:1701.07212](https://arxiv.org/abs/1701.07212).
4. <a id="ref-surya"></a>Surya, “The causal set approach to quantum gravity,” [arXiv:1903.11544](https://arxiv.org/abs/1903.11544).
5. <a id="ref-sverdlov"></a>Sverdlov, “Spinor fields in causal set theory,” [arXiv:0808.2956](https://arxiv.org/abs/0808.2956).
6. <a id="ref-ds"></a>Dowker, Sorkin, “Symmetry-breaking and zero-one laws,” [arXiv:1909.06070](https://arxiv.org/abs/1909.06070).
7. <a id="ref-bhs"></a>Bombelli, Henson, Sorkin, “Discreteness without symmetry breaking: a theorem,” Mod. Phys. Lett. A 24 (2009) 2579, [arXiv:gr-qc/0605006](https://arxiv.org/abs/gr-qc/0605006).
8. <a id="ref-clu"></a>Catterall, Laiho, Unmuth-Yockey, “Kähler–Dirac fermions on Euclidean dynamical triangulations,” Phys. Rev. D 98 (2018) 114503, [arXiv:1810.10626](https://arxiv.org/abs/1810.10626).
9. <a id="ref-butt"></a>Butt, Catterall, Pradhan, Toga, “Anomalies and symmetric mass generation for Kähler–Dirac fermions,” [arXiv:2101.01026](https://arxiv.org/abs/2101.01026).
10. <a id="ref-catterall"></a>Catterall, “Lattice regularization of reduced Kähler–Dirac fermions and connections to chiral fermions,” SciPost Phys. 16 (2024) 108, [arXiv:2311.02487](https://arxiv.org/abs/2311.02487).
11. <a id="ref-ch"></a>Catterall, Hubisz, “Lorentzian Kähler–Dirac fermions,” [arXiv:2608.19400](https://arxiv.org/abs/2608.19400).
12. <a id="ref-wachs"></a>Wachs, “Poset topology: tools and applications,” [arXiv:math/0602226](https://arxiv.org/abs/math/0602226).
13. <a id="ref-cc"></a>Calabrese, Cardy, “Entanglement entropy and quantum field theory,” J. Stat. Mech. (2004) P06002, [arXiv:hep-th/0405152](https://arxiv.org/abs/hep-th/0405152).
