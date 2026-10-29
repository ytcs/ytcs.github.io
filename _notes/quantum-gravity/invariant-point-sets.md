---
title: Lorentz-invariant point sets and the structures they cannot carry
date: 2026-10-29
category: substrate
order: 5
description: Poisson point sets and Haar-random lattices are Poincaré invariant in law, and the lattices are not Poisson. A Poincaré-invariant random set of null lines or null hyperplanes with Radon mean measure does not exist, nor do invariant substrates with complexified positions or with frame marks from a lattice of frames that support a finite neighbour rule. Hecke correspondences between lattices are Lorentz equivariant but act on rest-frame space rather than spacetime. Poisson causal orders carry no hand, while chiral random lattices exist.
status: Theorems proved in full, using the Siegel mean value theorem, Moore's ergodicity theorem, Weil's criterion for invariant measures and the spinor-norm structure of local isometry groups as cited results; the intensity screen is proved for null lines, null hyperplanes and point marks only.
excludes: Poincaré-invariant random sets of null lines or null hyperplanes with Radon mean measure; invariant point processes with complexified or hyperboloid-marked positions and finite real-slice intensity; finite neighbour rules built from lattice-of-frames marks on Poisson points; Hecke correspondences at finitely many primes as local dynamics on spacetime; a hand on the causal order of any Poisson or mixed Poisson process.
---

Which discrete structures in Minkowski space have a law that is invariant under the Poincaré group, and what can such a structure carry? Poisson point sets are invariant, and so are Haar-random lattices, which are not Poisson and have the Poisson mean and variance of every count (Theorem 1), but have a void probability that decays like $$1/V$$ rather than $$e^{-V}$$ (Proposition 2). Structures built from extended objects or from extra data attached to points are much more restricted. A Poincaré-invariant random set of null lines or of null hyperplanes whose mean measure is Radon does not exist (Theorem 4). An invariant point process with complexified positions, or with marks in a hyperboloid, has either finite real-slice intensity and trivial marks, or infinite real-slice intensity (Proposition 3). If each point of a Poisson process carries a frame modulo an arithmetic lattice $$\Gamma$$ of Lorentz transformations, no equivariant rule picks a finite non-empty set of neighbours (Theorem 6). Hecke correspondences between integral Lorentzian lattices are exactly Lorentz equivariant, positive and of finite valence, but a step changes the lattice in every region of spacetime at once, and on the Eisenstein sector the correspondence is a wave propagator on the space of rest frames (Propositions 8 to 11). The causal order of a Poisson or mixed Poisson process carries no hand, while an invariant, ergodic random lattice with no improper symmetry exists (Propositions 12 to 14).

The invariance-in-law results build on the theorem of Bombelli, Henson and Sorkin that a Poisson sprinkling admits no equivariant choice of a preferred direction [[Bombelli–Henson–Sorkin]](#ref-bhs), on the observation of Dowker, Henson and Sorkin that the discreteness of a sprinkling is compatible with local Lorentz invariance [[Dowker–Henson–Sorkin]](#ref-dhs), and on the finding of Saravani and Aslanbeigi that Lorentzian lattices in two dimensions have number fluctuations below Poisson [[Saravani–Aslanbeigi]](#ref-sa). What is added here is the construction and exact description of the invariant non-Poisson example (Haar-random lattices), the proof that the extended-object and marked substrates do not exist or carry no finite rule, the extension of the Bombelli–Henson–Sorkin argument to frames modulo a lattice, and the statement of what Hecke correspondences are and are not. The page [Exact discrete Lorentz symmetry cannot carry propagation](/quantum-gravity/discrete-lorentz-group/) proves the no-finite-orbit lemma that is used for lattice stabilisers here, and [Kernel-scale kinematics of a single realisation](/quantum-gravity/per-realisation-kinematics/) treats the cocompact orbit lattices, which are invariant in law and have no translation symmetry.

## 1. Point processes and what invariance means

A point process in $$\mathbb R^d$$ is a random locally finite subset $$\Phi$$, equivalently a random counting measure. Its *intensity measure* is $$\mu(B)=\mathbb E\,\Phi(B)$$; Campbell's formula says $$\mathbb E\sum_{x\in\Phi}f(x)=\int f\,d\mu$$. A *Poisson process* with a locally finite mean measure $$\mu$$ has independent Poisson counts in disjoint sets; if the mean of a set is infinite, its count is infinite almost surely. A *mixed Poisson* or *Cox* process is a Poisson process whose mean measure is itself random. A process is *invariant in law* under a group if the transformed process has the same law, and *ergodic* if every invariant event has probability 0 or 1. The Lorentz group $$G=SO^+(1,d-1)$$ acts on $$\mathbb R^d$$ with the form $$B(x,y)=x_0y_0-\vec x\cdot\vec y$$ and $$Q(x)=B(x,x)$$; the Poincaré group $$P$$ adds translations.

The *Palm law* of a process at a point is its law conditioned on having a point there. For a Poisson process the Palm law at $$0$$ is the law of $$\Phi\cup\{0\}$$ (Slivnyak–Mecke). For a random lattice with a random shift it is the law of the lattice through $$0$$. Mecke's formula, a consequence of independence in the Poisson case, gives for a Poisson process with diffuse mean measure $$\mu$$ the identity $$\mathbb E\sum_{x\ne y\in\Phi}f(x,y)=\iint f\,d\mu\,d\mu$$ and likewise for triples.

The simplest invariant process is the Poisson process of unit density. A fixed lattice $$\Lambda$$ is not invariant: a boost changes it, and randomising its position by translations does not make the law invariant under boosts. Section 2 shows that randomising the lattice itself, with the invariant measure on the space of lattices, does.

## 2. Haar-random lattices

Let $$d\ge2$$, $$X_d=ASL(d,\mathbb R)/ASL(d,\mathbb Z)$$ the space of affine lattices of covolume 1, and $$\mu$$ its Haar probability measure, which is finite (Siegel). A point $$(g,\xi)$$ of $$X_d$$ is the lattice $$g\mathbb Z^d+\xi$$. Two results are used.

<div class="theorem" markdown="1">

**Theorem S** (Siegel's mean value theorem, Siegel 1945). For $$f\in L^1(\mathbb R^d)$$ and $$d\ge2$$, $$\int\sum_{v\in g\mathbb Z^d\setminus\{0\}}f(v)\,d\mu(g)=\int f$$, where $$\mu$$ is the Haar probability measure on $$SL(d,\mathbb R)/SL(d,\mathbb Z)$$.

</div>

<div class="theorem" markdown="1">

**Theorem M** (Moore's ergodicity theorem, Moore 1966). A closed non-compact subgroup of a connected simple Lie group with finite centre acts ergodically on the quotient by a lattice with respect to Haar measure.

</div>

<div class="theorem" markdown="1">

**Theorem 1** (Haar-random lattices). Let $$\Lambda=g\mathbb Z^d+\xi$$ with $$(g,\xi)$$ Haar distributed on $$X_d$$, $$d\ge2$$. Then:

(i) $$\Lambda$$ is invariant in law under $$ASL(d,\mathbb R)$$, hence under the Poincaré group $$P=SO^+(1,d-1)\ltimes\mathbb R^d$$;

(ii) $$\Lambda$$ is ergodic under $$P$$;

(iii) $$\mathbb E\,N(D)=\mathrm{Var}\,N(D)=\mathrm{vol}(D)$$ for every bounded Borel set $$D$$, where $$N(D)=\#(\Lambda\cap D)$$;

(iv) $$\Lambda$$ is neither a Poisson process nor a mixed Poisson process with a.s. diffuse random mean measure.

</div>

The idea is that Haar measure is invariant, the translation part is uniform on a torus fibre, and Siegel's formula computes the pair correlation exactly. The heart of (iv) is that a lattice contains the reflection $$2y-x$$ of every pair of its points, which no diffuse random set does.

<div class="proof" markdown="1">

*Proof.* (i) Haar measure on $$X_d$$ is invariant under left multiplication by $$ASL(d,\mathbb R)$$, which contains $$P$$.

(ii) A $$P$$-invariant event is invariant under the translation subgroup. The translations act transitively on each torus fibre $$\mathbb R^d/g\mathbb Z^d$$, so an invariant event depends on $$g$$ only. $$SO^+(1,d-1)$$ is a closed non-compact subgroup of the simple group $$SL(d,\mathbb R)$$, so by Moore's theorem it acts ergodically on $$SL(d,\mathbb R)/SL(d,\mathbb Z)$$.

(iii) For fixed $$g$$, uniform averaging over $$\xi$$ gives $$\mathbb E_\xi N(D)=\mathrm{vol}(D)$$. For the second factorial moment,

$$
\mathbb E\sum_{x\ne y\in\Lambda}\mathbf 1_D(x)\mathbf 1_D(y)=\int dx\,\mathbf 1_D(x)\ \mathbb E_g\!\!\sum_{v\in g\mathbb Z^d\setminus\{0\}}\mathbf 1_D(x+v)=\mathrm{vol}(D)^2, \tag{1}
$$

where the translation average of the sum over $$x\in\Lambda$$ turned it into $$\int dx$$ (covolume 1) and the last step is Siegel's theorem applied to $$f=\mathbf 1_{D-x}$$. Then $$\mathrm{Var}\,N=\mathbb E N(N-1)+\mathbb E N-(\mathbb E N)^2=\mathrm{vol}(D)$$.

(iv) Almost surely, $$x,y\in\Lambda$$ implies $$2y-x\in\Lambda$$, and $$\Lambda$$ has infinitely many pairs $$x\ne y$$. For a Poisson process with diffuse mean measure $$m$$, Mecke's formula gives $$\mathbb E\sum_{x,y,z\ \mathrm{distinct}}\mathbf 1[z=2y-x]=\iint m(\{2y-x\})\,dm(x)\,dm(y)=0$$, and the same holds conditionally for a mixed Poisson process. So with probability 1 the lattice contains triples $$(x,y,2y-x)$$ with $$x\ne y$$, an event of probability zero for those processes. $$\square$$

</div>

Statement (iii) shows that the counts of a Haar lattice have the Poisson mean and variance in every region. They are not Poisson, and the difference is visible at the level of voids.

We work in two dimensions with null coordinates $$(u,v)$$ in which the area element is $$du\,dv$$ and boosts act by $$(u,v)\mapsto(e^su,e^{-s}v)$$. A unit-density lattice has covolume 1 in these coordinates. For two points $$0,y$$ of the lattice with $$y=(u,v)$$, $$u,v>0$$, the causal interval is the open rectangle $$(0,u)\times(0,v)$$ of area $$V=uv$$. Write $$\zeta=\tfrac12\ln(u/v)$$, so $$du\,dv=dV\,d\zeta$$, and let $$m(y)$$ be the number of lattice points in the rectangle. Boost invariance of the law of the lattice through $$0$$ implies that the expected number of future points $$y$$ with $$m(y)=m$$ and $$(V,\zeta)$$ in a small set is $$\int f_m(V)\,dV\,d\zeta$$ over that set. By Siegel's theorem $$\sum_mf_m(V)=1$$ for every $$V$$, so $$0\le f_m\le1$$. For a Poisson process of unit density, $$f_m(V)=e^{-V}V^m/m!$$.

<div class="proposition" markdown="1">

**Proposition 2** (the void law of a Haar-random lattice in two dimensions). For the Haar-random lattice through $$0$$ in $$\mathbb R^{1,1}$$,

$$
f_0(V)=\frac6{\pi^2}\min\Big(1,\frac1V\Big),\tag{2}
$$

and $$f_1=f_2=f_3=0$$ for $$V>6$$. Consequently, for any weights $$w_0,w_1,w_2,w_3$$ with $$w_0\ne0$$,

$$
\sum_{m\le3}w_m\int_0^Xf_m(V)\,dV=\frac{6}{\pi^2}\,w_0\ln X+O(1)\qquad(X\to\infty), \tag{3}
$$

while for the Poisson process the same quantity converges to $$\sum_mw_m$$.

</div>

The idea is to unfold Haar measure over the primitive lattice vectors, so that the lattice is determined by a single uniform angle, and then to read the emptiness of the rectangle off the fractional parts $$\{n\alpha\}$$. The first step is standard, the second is a short computation.

<div class="proof" markdown="1">

*Proof.* Unfolding Haar measure on $$SL(2,\mathbb R)/SL(2,\mathbb Z)$$ over the pairs (lattice, primitive vector) gives: the expected sum over primitive vectors $$y$$ of $$F(y,L)$$ is $$\zeta(2)^{-1}\int dy\int_0^1d\theta\,F(y,L_{y,\theta})$$, where $$L_{y,\theta}=\mathbb Zy+\mathbb Z(w_0+\theta y)$$ and $$w_0$$ is any vector with $$\det(y,w_0)=1$$ (the stabiliser of a primitive vector in $$SL(2,\mathbb Z)$$ is the group of unipotent translations, whose quotient in the real group is the circle of $$\theta$$). A non-primitive vector $$y=qy_0$$, $$q\ge2$$, has the points $$ky_0$$, $$0<k<q$$, on the diagonal of its rectangle, so only primitive vectors contribute to $$f_0$$, and $$f_0(V)=\zeta(2)^{-1}\,P_\theta(\text{rectangle empty})$$ with $$\zeta(2)^{-1}=6/\pi^2$$.

Rescale the rectangle of $$y=(a,b)$$, $$V=ab$$, to the unit square. The lattice becomes generated by $$(1,1)$$ and $$w'=(\alpha,\alpha+1/V)$$, with $$\alpha$$ uniform modulo 1 (the condition $$\det(y,w)=1$$ gives the offset $$1/V$$ between the two coordinates). Its points are $$(n\alpha+k,\ n\alpha+k+n/V)$$, $$n,k\in\mathbb Z$$. For $$n\ge1$$ such a point lies in the open square iff $$n\alpha+k\in(0,1-n/V)$$, which has a solution $$k$$ iff $$n<V$$ and $$\{n\alpha\}\in(0,1-n/V)$$; for $$n\le-1$$ the condition is the same with $$\lvert n\rvert $$, by the point reflection $$p\mapsto y-p$$. The line $$n=0$$ contributes only the diagonal points $$ky$$, none inside. Put $$\delta=1-\{\alpha\}\in(0,1)$$. If $$V\le1$$ there is no $$n$$ with $$1\le n<V$$ and the rectangle is empty. If $$V>1$$, the case $$n=1$$ shows that emptiness forces $$\delta\le1/V$$; conversely if $$\delta\le1/V$$ then for $$1\le n<V$$, $$n\delta\le n/V<1$$, so $$\{n\alpha\}=1-n\delta\ge1-n/V$$ and every line is empty. Hence $$P_\theta(\text{empty})=\min(1,1/V)$$, which is (2).

For the vanishing statement let $$V>6$$. A vector $$y=qy_0$$ with $$q\ge2$$: in the unit-square picture the lattice has points $$(n\alpha+k/q,\ n\alpha+k/q+n/V)$$, and for $$1\le n<V(1-1/q)$$ the interval $$(0,1-n/V)$$ has length larger than the spacing $$1/q$$ of the progression $$n\alpha+k/q$$, so line $$n$$ contains a point. For $$V>6$$ and $$q\ge2$$ the lines $$n=1,2$$ qualify, and by reflection so do $$n=-1,-2$$, so $$m\ge(q-1)+4\ge5$$. For $$q=1$$ and $$\delta>1/V$$, line 1 is occupied. If $$\delta<\tfrac12$$, then $$\{2\alpha\}=1-2\delta<1-2/V$$, so line 2 is occupied too. If $$\delta>\tfrac12$$, put $$\gamma=1-\delta<\tfrac12$$; line 2 is empty only if $$\gamma\ge\tfrac12-1/V$$, and then $$\{3\alpha\}=3\gamma-1\le\tfrac12<1-3/V$$ (because $$V>6$$), so line 3 is occupied. In each case two lines and their reflections give $$m\ge4$$. Thus $$m\in\{0\}\cup[4,\infty)$$ for $$q=1$$ and $$m\ge5$$ for $$q\ge2$$, so $$f_m=0$$ for $$m=1,2,3$$ when $$V>6$$.

For (3): $$\int_0^Xf_0\,dV=\frac6{\pi^2}(1+\ln X)$$ for $$X\ge1$$, and $$\int_0^6f_m\,dV\le6$$ for each $$m$$, since $$f_m\le1$$. $$\square$$

</div>

The Poisson process has $$\int_0^\infty\sum_mw_mf_m\,dV=\sum_mw_m\int e^{-V}V^m/m!\,dV=\sum_mw_m$$, which vanishes for the two-dimensional Benincasa–Dowker weights $$(w_0,w_1,w_2)=(1,-2,1)$$. This is the first of the conditions that make a layer-weighted sum over the causal past reproduce a d'Alembertian on a Poisson sprinkling; the complete list, in terms of the Mellin transforms $$\int f_m(V)V^{s-1}dV$$, is on the page [Mixed-Poisson and signed substrates and the four-dimensional d'Alembertian](/quantum-gravity/signed-substrates/). By (2) the Mellin transform of $$f_0$$ is $$\frac{6}{\pi^2}\frac1{s(1-s)}$$, which has a pole at $$s=1$$ where the Poisson transform $$\Gamma(s)$$ is regular. By (3), a layer-weighted kernel supported on $$m\le3$$ with $$w_0\ne0$$, in particular the Benincasa–Dowker weights, has a zeroth moment that diverges logarithmically on a Haar-random lattice. The mean action of a diamond is a weighted combination of such moments, and the corresponding divergence of the mean 2D action is not derived here. So an invariant, non-Poisson, positive point process exists and the Poisson-weighted dictionary does not transfer to it. For the cocompact orbit lattices of the page on [single realisations](/quantum-gravity/per-realisation-kinematics/) the empty-interval law has compact support and the corresponding moments are finite.

*Numerical check.* In two dimensions we sampled Haar lattices (the hyperbolic fundamental domain with density $$dx\,dy/y^2$$, a uniform rotation, a uniform shift). For rectangles of area $$V=4,16,64$$ in $$6\cdot10^4$$ lattices the mean count over $$V$$ was $$1.0004\pm0.0021$$, $$0.9985\pm0.0010$$, $$0.9990\pm0.0005$$. The variance over the mean was $$1.03$$, $$0.96$$, $$0.80$$ at $$6\cdot10^4$$ samples; its estimator has infinite fourth moment because of cusp lattices, and an independent run of $$3\cdot10^3$$ lattices gave $$1.03$$, $$0.78$$, $$1.57$$. For (2), over $$4.5\cdot10^3$$ lattices (three batches of $$1.5\cdot10^3$$) the number of lattice vectors with $$\lvert \zeta\rvert <1$$ and $$V$$ in a bin, divided by bin width times $$\zeta$$-range, was $$1.00\pm0.01$$ in each bin (Siegel), and the corresponding rate of empty rectangles was, bin by bin against the prediction (2) averaged over the bin: $$0.624\pm0.017$$ against $$0.608$$ for $$V\in(0.25,0.5)$$, $$0.608\pm0.005$$ against $$0.608$$ for $$(0.5,1)$$, $$0.409\pm0.007$$ against $$0.421$$ for $$(1,2)$$, $$0.208\pm0.002$$ against $$0.211$$ for $$(2,4)$$, $$0.101\pm0.005$$ against $$0.105$$ for $$(4,8)$$, and $$0.056\pm0.002$$ against $$0.053$$ for $$(8,16)$$ (errors from the batch spread). A direct sampling of the angle $$\alpha$$ gave $$P(\text{empty})=0.498,\,0.242,\,0.168,\,0.104$$ at $$V=2,4,6,10$$ against $$1/V=0.5,\,0.25,\,0.167,\,0.1$$.

For $$d\ge3$$ this page does not prove a void law. Projecting the lattice along a primitive timelike vector onto its orthogonal complement gives a random lattice and suggests the same $$1/V$$ decay, which is not established here.

## 3. The invariant-intensity screen

A random set of extended objects (lines, hyperplanes, frames at points) is Poincaré invariant only if its mean measure is a Poincaré-invariant measure on the space of objects. The screen asks which spaces of objects carry such a measure, with the further requirement that the set of objects meeting a compact region have finite measure.

### 3.1 Wandering shells

<div class="lemma" markdown="1">

**Lemma 1** (wandering shells). Let $$\Gamma\subset SO^+(1,d-1)$$ contain a hyperbolic element, and suppose that either $$d=2$$ or the Zariski closure of $$\Gamma$$ acts irreducibly on $$\mathbb R^d$$ (for example $$\Gamma=SO^+(1,d-1)$$, or a Zariski-dense subgroup). Let $$\nu$$ be a $$\Gamma$$-invariant Borel measure on $$\mathbb R^d$$ and $$A$$ a $$\Gamma$$-invariant Borel set with $$\nu(A\setminus\{0\})<\infty$$. Then $$\nu(A\setminus\{0\})=0$$.

</div>

The point is that a hyperbolic element permutes an infinite family of disjoint shells, which then have equal finite mass in total.

<div class="proof" markdown="1">

*Proof.* Put $$\mu=\nu\vert _{A\setminus\{0\}}$$, finite and $$\Gamma$$-invariant. Let $$\gamma\in\Gamma$$ be hyperbolic, with null eigenvectors $$w_\pm$$ and $$\gamma w_\pm=\lambda^{\pm1}w_\pm$$, $$\lambda>1$$. Write $$a(y)=B(y,w_-)/B(w_+,w_-)$$ for the coefficient of $$w_+$$; then $$a(\gamma y)=\lambda a(y)$$. The shells $$W_k=\{y:\lambda^k\le\lvert a(y)\rvert <\lambda^{k+1}\}$$, $$k\in\mathbb Z$$, are disjoint, $$\gamma W_k=W_{k+1}$$, and so $$\mu(W_k)$$ is independent of $$k$$. Since their union has finite mass, each has mass 0, and $$\mu(a\ne0)=0$$. The same argument with $$\gamma^{-1}$$ and the coefficient $$b(y)$$ of $$w_-$$ gives $$\mu(b\ne0)=0$$. So $$\mu$$ is carried by $$P_\gamma=\{w_+,w_-\}^\perp$$, a subspace of dimension $$d-2$$; if $$d=2$$ this is $$\{0\}$$ and we are done. In general, conjugating by $$g\in\Gamma$$ shows $$\mu$$ is carried by $$gP_\gamma$$ for every $$g$$, hence by $$\bigcap_{g\in\Gamma}gP_\gamma$$. This is a $$\Gamma$$-invariant, hence Zariski-closure-invariant, proper subspace of $$\mathbb R^d$$, so it is $$\{0\}$$ by irreducibility. Thus $$\mu(\mathbb R^d\setminus\{0\})=0$$. $$\square$$

</div>

### 3.2 Marks on points

<div class="proposition" markdown="1">

**Proposition 3** (complexified and hyperboloid-marked substrates). Let $$\Phi$$ be a point process on $$\mathbb R^d\times\mathbb R^d$$, with points $$z=(x,y)$$ (read as $$x+iy$$), whose law is invariant under $$(x,y)\mapsto(\Lambda x+a,\Lambda y)$$, $$\Lambda\in SO^+(1,d-1)$$, $$a\in\mathbb R^d$$, and whose intensity measure $$I$$ is Radon, $$d\ge2$$. Then $$I=\mathrm{Leb}_x\otimes\nu$$ with $$\nu$$ a Radon Lorentz-invariant measure on $$\mathbb R^d$$, and the projected process $$\{x\}$$ has intensity $$\nu(\mathbb R^d)\,\mathrm{Leb}$$. This is finite only if $$\nu=c\,\delta_0$$, that is, only if all points are real. If $$\Phi$$ lives in the forward tube $$y\in V_+$$ (the open forward cone) it is almost surely empty, or its projection has infinite intensity. If $$\Phi$$ is a Poisson process, an infinite projected intensity means that every non-empty box contains infinitely many projected points almost surely.

</div>

<div class="proof" markdown="1">

*Proof.* Translation invariance in $$x$$ makes $$I(\cdot\times C)$$ a translation-invariant Radon measure for each bounded $$C$$, hence $$c(C)\,\mathrm{Leb}$$; $$\nu(C)=c(C)$$ is additive in $$C$$ and Radon, and $$I(\Lambda B\times\Lambda C)=I(B\times C)$$ gives $$\nu(\Lambda C)=\nu(C)$$. The projected intensity of a box $$B$$ is $$I(B\times\mathbb R^d)=\mathrm{vol}(B)\nu(\mathbb R^d)$$. By Lemma 1 with $$A=\mathbb R^d$$ and $$\Gamma=SO^+(1,d-1)$$, $$\nu(\mathbb R^d\setminus\{0\})$$ is either 0 or infinite, so finiteness forces $$\nu=c\delta_0$$. In the forward tube $$0\notin V_+$$ and the same lemma with $$A=V_+$$ gives $$\nu=0$$ or $$\nu(V_+)=\infty$$. A Poisson count with infinite mean is infinite almost surely. $$\square$$

</div>

The real-slice intensity is finite only for real points, so a Lorentz-invariant sprinkling cannot be given a small, invariantly defined imaginary part: every invariant open neighbourhood $$U$$ of $$0$$ in $$y$$-space contains the whole null cone, since a boost carries any null vector arbitrarily close to $$0$$ and $$U$$ is invariant, and it contains every hyperboloid $$\{y^2=r^2\}$$ with $$\lvert r\rvert $$ small enough that it meets $$U$$; such a set has infinite volume. Smearing does not help either: for an invariant $$F\ge0$$ the sets $$\{F\ge c\}$$ are invariant, so Lemma 1 shows $$\int F\,d\nu\in\{0,\infty\}$$. A complex conjugation or $$CPT$$-type reality condition adds only a $$\mathbb Z_2$$ and leaves the action on $$y$$ intact. What escapes is a process whose imaginary parts are supplied by the state rather than by the law (so that the law is not invariant), complexified correlation functions as limits $$y\to0$$, and the Euclidean real form, where $$SO(d)$$ is compact and there are invariant probability measures on the sphere. A non-Poisson invariant law with infinite intensity but almost surely locally finite projection is not decided here: the argument needs the first moment to be Radon.

### 3.3 Null lines and null hyperplanes

The space of null lines is the quotient $$P/H$$ of the Poincaré group by the stabiliser of one null line. Weil's criterion decides whether such a quotient carries an invariant measure.

<div class="theorem" markdown="1">

**Theorem W** (Weil's criterion). Let $$G$$ be a locally compact group, $$H$$ a closed subgroup, and $$\Delta_G,\Delta_H$$ the modular functions. Then $$G/H$$ carries a non-zero $$G$$-invariant Radon measure if and only if $$\Delta_G\vert _H=\Delta_H$$ (see, for instance, section 2.6 of Folland's *A Course in Abstract Harmonic Analysis*).

</div>

The Poincaré group $$P=SO^+(1,d-1)\ltimes\mathbb R^d$$ is unimodular. For a Lie subgroup $$H$$, $$\Delta_H(h)=\lvert \det\mathrm{Ad}_{\mathfrak h}(h)\rvert ^{\pm1}$$, so $$G/H$$ carries an invariant measure if and only if $$\lvert \det\mathrm{Ad}_{\mathfrak h}(h)\rvert =1$$ for all $$h\in H$$.

<div class="theorem" markdown="1">

**Theorem 4** (no invariant measure on null lines or null hyperplanes). Let $$d\ge2$$. The space of null lines in $$\mathbb R^{1,d-1}$$ and the space of null hyperplanes each carry no non-zero Poincaré-invariant Radon measure. Consequently a Poincaré-invariant point process of null lines, or of null hyperplanes, whose intensity measure is Radon is almost surely empty; in particular there is no invariant Poisson process of null lines or null hyperplanes.

</div>

The proof is a computation of the modular function on the stabiliser. The obstruction is the boost along the null direction: it stretches the null direction and every null rotation fixing it by the same factor, so the stabiliser is not unimodular.

<div class="proof" markdown="1">

*Proof.* Take the null vector $$k=e_0+e_{d-1}$$, $$k'=e_0-e_{d-1}$$, and $$e_i$$ ($$1\le i\le d-2$$) transverse. The Lie algebra of the Poincaré group consists of pairs $$(A,a)$$, $$A\in\mathfrak{so}(1,d-1)$$, $$a\in\mathbb R^d$$, with $$\mathrm{Ad}(g)(A,a)=(gAg^{-1},ga)$$. Write $$x\wedge y$$ for the element $$z\mapsto B(y,z)x-B(x,z)y$$.

The null line $$\mathbb Rk$$ has stabiliser $$H$$ with $$\mathfrak h=\{(A,a):Ak\in\mathbb Rk,\ a\in\mathbb Rk\}$$. A basis is: $$J=k\wedge k'$$ (the boost along $$k$$), the rotations $$e_i\wedge e_j$$, the null rotations $$N_i=k\wedge e_i$$ (they satisfy $$N_ik=B(e_i,k)k-B(k,k)e_i=0$$), and the translation $$k$$. (The elements $$k'\wedge e_i$$ do not stabilise $$k$$.) Let $$B_\eta$$ be the boost in the $$(0,d-1)$$ plane, $$B_\eta k=e^\eta k$$, $$B_\eta k'=e^{-\eta}k'$$, $$B_\eta e_i=e_i$$. Then $$\mathrm{Ad}(B_\eta)J=J$$, the rotations are fixed, $$\mathrm{Ad}(B_\eta)N_i=(B_\eta k)\wedge(B_\eta e_i)=e^\eta N_i$$ for each of the $$d-2$$ transverse directions, and the translation $$k$$ is multiplied by $$e^\eta$$. So $$\det\mathrm{Ad}_{\mathfrak h}(B_\eta)=e^{(d-1)\eta}\ne1$$, and by Weil's criterion there is no invariant measure on $$P/H$$.

The null hyperplane $$\{x:B(k,x)=0\}$$ has the same Lorentz part, and its translation part is $$\{a:B(k,a)=0\}=\mathrm{span}\{k,e_1,\dots,e_{d-2}\}$$ on which $$B_\eta$$ acts by $$e^\eta$$ on $$k$$ and trivially on the $$e_i$$. Again $$\det=e^{(d-1)\eta}$$. In $$d=2$$ the two spaces coincide.

If $$\Phi$$ is an invariant process with Radon intensity measure, the intensity is an invariant Radon measure, hence zero, so $$\Phi$$ is empty almost surely. $$\square$$

</div>

The set of null lines meeting a given compact set is itself compact (a point times the celestial sphere), so finiteness of the number of lines through a region is not the obstruction; only the invariance is. For the chart $$(n,y)$$, with $$n\in S^2$$ the spatial direction of the line and $$y\in\mathbb R^3$$ the intercept $$\vec x-x_0n$$ at $$x_0=0$$, a Lorentz transformation changes the measure $$d\Omega\,d^3y$$ by the factor $$D^{-3}$$, with $$D$$ the Doppler factor of the direction. A boost fixing a direction $$n_0$$ has $$D(n_0)=e^{\pm\eta}$$, so an invariant density at $$n_0$$ would satisfy $$\rho(n_0)=e^{\pm3\eta}\rho(n_0)$$ and vanish. A numerical check of this exponent: for 200 random proper Lorentz transformations, a finite-difference Jacobian of the induced map on $$(\theta,\phi,y)$$, divided by the corresponding factor for the solid angle, fits $$\ln J=p\ln D+c$$ with $$p=-3.000000$$, $$\lvert c\rvert =3\cdot10^{-11}$$ and maximal residual $$4\cdot10^{-9}$$.

<div class="proposition" markdown="1">

**Proposition 5** (timelike lines). The space of timelike lines carries a Poincaré-invariant measure $$d\mu=d\mu_{H^{d-1}}(u)\,d^{d-1}y_\perp$$, where $$u$$ is the unit direction and $$y_\perp$$ the intercept in the Euclidean space $$u^\perp$$. The $$\mu$$-measure of the set of lines meeting a given non-empty open set is infinite.

</div>

<div class="proof" markdown="1">

*Proof.* The stabiliser of a timelike line is $$SO(d-1)\times\mathbb R$$, which is unimodular, so an invariant measure exists by Weil's criterion; the product measure is evidently Lorentz invariant, and translations act on $$y_\perp$$ by Euclidean translations. A non-empty open set contains a spatial ball of some radius $$\rho$$ in a hyperplane $$t=\text{const}$$, which we take to be $$t=0$$. A line in direction $$u=(\cosh\eta,\sinh\eta\,\hat n)$$ meets it iff $$y_\perp$$ lies in the image of the ball under the projection along $$u$$ onto $$u^\perp$$, whose $$(d-1)$$-volume is $$\cosh\eta$$ times that of the ball (the projection of a hyperplane with unit timelike normal $$e_0$$ along $$u$$ onto $$u^\perp$$ scales volume by $$\lvert B(e_0,u)\rvert $$). Hence $$\mu\ge\mathrm{vol}\cdot\int\cosh\eta\,\sinh^{d-2}\eta\,d\eta\,d\Omega=\infty$$. $$\square$$

</div>

For timelike lines the obstruction is thus the opposite one: the measure exists and the incidence with a compact region is infinite. Spacelike lines have a unimodular stabiliser $$SO(1,d-2)\times\mathbb R$$ and so an invariant measure as well; that its incidence with compact regions is also infinite is expected but is not proved here. The general statement that for every Poincaré-homogeneous space of affine subspaces, an invariant Radon measure and finite incidence with compact sets exclude each other except for points, is a conjecture that this page does not prove.

The screen has established the following, and nothing beyond it. No Poincaré-invariant Radon measure exists on the space of null lines or of null hyperplanes (Theorem 4). Lorentz-invariant marks on points are excluded when the mean measure is Radon, in the forms of Proposition 3 and of Theorem 6 below. Timelike lines are excluded as a substrate with finite incidence by Proposition 5, since the invariant measure puts infinite mass on lines through a region. Spacelike lines are not treated. Twistors enter only through the lines they represent: in the twistor space $$PT^+$$ of positive-norm twistors the Poincaré group is transitive and a point and a twistor have exactly one invariant, the phase of a spinor pairing; a frame-free incidence rule that is invariant would, for a finite rule, push forward a finite $$SL(2,\mathbb C)$$-invariant measure to $$\mathbb{CP}^1$$, which does not exist because a loxodromic element contracts $$\mathbb{CP}^1$$ onto a point. The other alternative, that each twistor meets a whole null hyperplane, is the hyperplane case of Theorem 4 only after a further argument that we only sketch. Hence we treat the twistor substrate as not excluded by a complete proof.

## 4. Frames modulo a lattice on Poisson points

The Bombelli–Henson–Sorkin theorem says that a Poisson sprinkling carries no equivariant choice of a preferred timelike direction [[Bombelli–Henson–Sorkin]](#ref-bhs). One may try to evade it by attaching to every point a frame, a Lorentz transformation, known only modulo an arithmetic subgroup $$\Gamma$$, as a weak local structure that could supply a Wick axis or a finite stencil.

*Set-up.* Let $$\Phi$$ be a Poisson process of unit intensity on $$\mathbb R^{1,d-1}$$ whose points $$x$$ carry independent marks $$m_x\in G/\Gamma$$, $$G=SO^+(1,d-1)$$, drawn from the $$G$$-invariant probability measure of the quotient by a lattice $$\Gamma$$ ($$d=2$$: the group of boosts with rapidity in $$a\mathbb Z$$; $$d=4$$: a cocompact arithmetic group such as the automorphism group of $$7x_0^2-x_1^2-x_2^2-x_3^2$$). The law of the marked process is invariant under the Poincaré group, which acts on positions by $$x\mapsto gx+t$$ and on marks by left multiplication. A *rule* is a measurable assignment $$N_x(\tilde\Phi)\subset\Phi\setminus\{x\}$$ to every point of a finite or infinite set of other points, equivariant under $$P$$.

<div class="theorem" markdown="1">

**Theorem 6** (a neighbour rule that sees frames only modulo $$\Gamma$$ is empty or infinite). Under the Palm law of the marked process at a point $$(0,m)$$, for almost every mark $$m$$,

$$
\mathbb P^0_m\big(0<\#N_0<\infty\big)=0 .
$$

Moreover, no equivariant measurable choice of a timelike unit vector (a Wick axis) $$u(\tilde\Phi)$$ exists.

</div>

The point is that, given the mark of the point at the origin, the Palm law is invariant under the stabiliser $$\Gamma_m=g\Gamma g^{-1}$$ of that mark, which is a lattice and so contains hyperbolic elements and is Zariski dense. The normalised counting measure of the neighbours is then a finite $$\Gamma_m$$-invariant measure, which Lemma 1 excludes.

<div class="proof" markdown="1">

*Proof.* By the Slivnyak–Mecke theorem the Palm law at $$(0,m)$$ is the law of $$\tilde\Phi\cup\{(0,m)\}$$. The marked process law is $$P$$-invariant, and $$\gamma\in\Gamma_m$$ fixes the pointed mark $$m$$ (because $$\gamma g\Gamma=g\Gamma$$) and the position $$0$$. So $$\gamma$$ maps the Palm law to itself. Define, for Borel $$A\subset\mathbb R^d\setminus\{0\}$$,

$$
\nu(A)=\mathbb E^0_m\Big[\frac{\#(N_0\cap A)}{\#N_0};\ 0<\#N_0<\infty\Big].
$$

By equivariance, $$N_0(\gamma\tilde\Phi)=\gamma N_0(\tilde\Phi)$$, and by the invariance of the Palm law under $$\gamma$$, $$\nu(\gamma A)=\nu(A)$$. Thus $$\nu$$ is a $$\Gamma_m$$-invariant measure on $$\mathbb R^d\setminus\{0\}$$ of total mass at most 1. The group $$\Gamma_m$$ is a lattice in $$G$$, so it contains hyperbolic elements and, by the Borel density theorem, is Zariski dense in $$G$$ when $$d\ge3$$; for $$d=2$$ it is generated by a boost. Lemma 1 with $$A=\mathbb R^d$$ gives $$\nu=0$$, that is $$\mathbb P^0_m(0<\#N_0<\infty)=0$$.

For the Wick axis, if $$u(\tilde\Phi)$$ were equivariant, its law would be a $$G$$-invariant probability measure on the hyperboloid $$H^{d-1}\subset\mathbb R^d\setminus\{0\}$$, which Lemma 1 (with $$\Gamma=G$$) excludes. $$\square$$

</div>

A rule that sees frames only through $$\Gamma$$-cosets therefore selects either no neighbours or infinitely many. In two dimensions the second alternative can be made explicit. For a point at the origin with mark rapidity $$\theta$$ and a future timelike point $$y=(u,v)$$, $$u,v>0$$, the $$\Gamma$$-invariant distance $$d(y)=\min_{k}\big\lvert \Lambda_{-(\theta+ka)}y\big\rvert _{\rm E}$$, with $$u=t+x$$, $$v=t-x$$ and the Euclidean norm of $$(t,x)$$, satisfies $$d(y)^2=uv\cosh(2\delta)$$, with $$\delta$$ the distance from $$\zeta=\frac12\ln(u/v)$$ to $$\theta+a\mathbb Z$$. In the coordinates $$(V,\zeta)$$ with $$V=uv$$ the region $$d(y)<1$$ is $$\{V<1/\cosh2\delta(\zeta)\}$$, of area $$\int d\zeta\,(\cosh2\delta(\zeta))^{-1}$$ with a periodic integrand of positive mean, so each strip $$\zeta\in[\zeta_0,\zeta_0+a]$$ contains the same area. A Poisson process of unit density with respect to $$du\,dv$$ therefore puts in $$\{d<1\}$$ a number of points that grows like $$\ln L$$ when the Euclidean radius is limited to $$L$$ (the cut $$V\cosh2\zeta\le L^2$$ limits $$\lvert \zeta\rvert $$ to about $$\ln L$$), so there is no finite neighbour set. The extension of the dichotomy to spinor-valued Wilson kernels, and the consequences for fermion regulators that use a frame, are the subject of [Frames, Wilson terms and reflection positivity for fermions on random points](/quantum-gravity/frames-and-reflection-positivity/).

The theorem does not exclude kernels defined as conditionally convergent sums over infinitely many neighbours with cancelling weights, and it does not apply when the mark law is not invariant under the stabiliser of the mark.

## 5. Hecke correspondences between lattices

The previous sections exclude local structure at the real place. A different kind of locality sits at a prime $$p$$: two lattices are neighbours when they differ by a Kneser step. Take $$\mathbb R^4$$ with $$Q=x_0^2-x_1^2-x_2^2-x_3^2$$, and let $$\mathcal Y$$ be the set of lattices $$L\subset\mathbb R^4$$ on which the bilinear form $$B$$ takes integer values and which have Gram determinant $$-1$$ (they are odd unimodular lattices of signature $$(1,3)$$, so each is isometric to $$\mathbb Z^4$$ with the form $$Q$$ by the classification of indefinite odd unimodular lattices); the group $$G=SO^+(1,3)$$ acts on $$\mathcal Y$$, and $$\mathbb Z^4\in\mathcal Y$$ has stabiliser $$\Gamma_0=SO^+(1,3;\mathbb Z)$$, a lattice in $$G$$ by the Borel–Harish-Chandra theorem. For an odd prime $$p$$ and $$L\in\mathcal Y$$, a primitive vector $$w\in L$$ with $$Q(w)\equiv0\pmod{p^2}$$ defines the neighbour

$$
L_w'=\{x\in L:B(x,w)\equiv0\ (\mathrm{mod}\ p)\}+\mathbb Z\,\frac wp, \tag{4}
$$

and $$N_p(L)$$ is the set of these lattices. (Equivalently, $$N_p(L)$$ is the set of $$L'\in\mathcal Y$$ with $$[L:L\cap L']=[L':L\cap L']=p$$.)

<div class="proposition" markdown="1">

**Proposition 7** (no summable equivariant kernel on a lattice). Let $$L\in\mathcal Y$$ and $$\Gamma_L=\mathrm{Stab}_G(L)$$. Every $$\Gamma_L$$-invariant summable function $$c:L\to\mathbb C$$ vanishes off $$0$$. Consequently an equivariant rule assigning to each point of $$L$$ a finite set of other points, or a summable kernel, is trivial.

</div>

<div class="proof" markdown="1">

*Proof.* $$\Gamma_L$$ is an arithmetic subgroup of the semisimple group $$G$$ without compact factors, hence Zariski dense (Borel). By the lemma on finite orbits of the page [Exact discrete Lorentz symmetry cannot carry propagation](/quantum-gravity/discrete-lorentz-group/) every orbit $$\Gamma_Lv$$, $$v\ne0$$, is infinite, and an invariant summable function is constant on infinite orbits, hence zero there. Equivariance of a rule under lattice translations determines it by its value at $$0$$, which is a $$\Gamma_L$$-invariant finite set or kernel. $$\square$$

</div>

<div class="proposition" markdown="1">

**Proposition 8** (Kneser neighbours). For odd $$p$$: (i) the lattice $$L_w'$$ of (4) lies in $$\mathcal Y$$, depends only on the line $$\langle w\bmod p\rangle$$ of isotropic vectors of $$L/pL$$, and distinct lines give distinct lattices; (ii) $$\#N_p(L)=p^2+1$$ if $$p\equiv3\pmod4$$ and $$(p+1)^2$$ if $$p\equiv1\pmod4$$; (iii) $$N_p(gL)=gN_p(L)$$ for $$g\in G$$, and $$L'\in N_p(L)$$ iff $$L\in N_p(L')$$.

</div>

<div class="proof" markdown="1">

*Proof.* A line of isotropic vectors modulo $$p$$ lifts to $$w$$ with $$Q(w)\equiv0\pmod{p^2}$$: if $$Q(w_0)\equiv0\pmod p$$, the gradient $$2Gw_0$$ is non-zero modulo $$p$$, so $$w_0+pr$$ with a suitable $$r$$ has $$Q\equiv0\pmod{p^2}$$ (Hensel). Another lift of the same line is $$w'=\lambda w+pr$$ with $$\lambda$$ a unit modulo $$p$$ and $$r\in L$$. Then $$L_{w'}=L_w$$ (where $$L_w:=\{x\in L:B(x,w)\equiv0\}$$), and $$Q(w')\equiv0\pmod{p^2}$$ gives $$2\lambda pB(w,r)\equiv0\pmod{p^2}$$, so $$r\in L_w$$; hence $$w'/p\in\lambda w/p+L_w$$, and since $$w\in L_w$$ and $$\lambda$$ is a unit, the lattice (4) is the same. Integrality: for $$x,x'\in L_w$$ and integers $$k,k'$$, $$B(x+kw/p,\,x'+k'w/p)=B(x,x')+\big(k'B(x,w)+kB(x',w)\big)/p+kk'Q(w)/p^2\in\mathbb Z$$. Since $$L_w$$ has index $$p$$ in $$L$$ (the functional $$x\mapsto B(x,w)\bmod p$$ is non-zero) and $$L'_w/L_w\cong\mathbb Z/p$$, the Gram determinant is $$-1\cdot p^2/p^2=-1$$. Two distinct lines give distinct lattices because $$L'_w\cap L=L_w$$ (an element $$x+kw/p$$ lies in $$L$$ iff $$p\mid k$$), and $$L_w/pL$$ is the orthogonal complement of the line. This proves (i). For (ii) the neighbours correspond to the points of the quadric $$Q=0$$ in $$\mathbb P^3(\mathbb F_p)$$, of which a nondegenerate quaternary form has $$p^2+(1+\varepsilon)p+1$$ with $$\varepsilon=+1$$ if the form is hyperbolic and $$-1$$ if elliptic; for $$\mathrm{diag}(1,-1,-1,-1)$$ the determinant is $$-1$$, and the form is hyperbolic iff $$-1$$ is a square mod $$p$$, that is $$p\equiv1\pmod4$$. For (iii), equivariance is clear; symmetry follows from the symmetric definition $$[L:L\cap L']=[L':L\cap L']=p$$. $$\square$$

</div>

*Numerical check.* Enumerating isotropic lines and computing Hermite normal forms of the lattices (4) for $$\mathbb Z^4$$ gives $$10$$, $$36$$ and $$50$$ distinct neighbours for $$p=3,5,7$$, all integral with Gram determinant $$-1$$, against $$p^2+1=10$$, $$(p+1)^2=36$$, $$p^2+1=50$$.

<div class="proposition" markdown="1">

**Proposition 9** (a Hecke step is global in spacetime). Let $$L'\in N_p(L)$$ and $$M=L\cap L'$$. Then $$L'\setminus L=\bigcup_{j=1}^{p-1}(jw/p+M)$$, a union of $$p-1$$ translates of a full-rank lattice. It meets every ball of radius larger than the covering radius of $$M$$. Every lattice on every trajectory of Hecke steps lies in $$\mathcal Y$$, so its Lorentzian form is the fixed form $$Q$$.

</div>

<div class="proof" markdown="1">

*Proof.* $$L'/M\cong\mathbb Z/p$$ is generated by $$w/p$$, and $$jw/p\notin L$$ for $$p\nmid j$$. The other statements are immediate. $$\square$$

</div>

A Hecke step thus changes the set of points in every region of spacetime at once, so the correspondence has no finite propagation speed, and the only Lorentzian metric available is $$Q$$, which is an input. This is what is meant by saying that the dynamics is not spacetime dynamics: no metric is produced by it, and it is not local in the sense of a rule that changes the configuration only near a point. The following shows that the correspondence nevertheless acts as a wave propagator on a different space.

For $$L\in\mathcal Y$$ set $$E_s(L)=\sum_u(u_0)^{-s}$$, the sum over primitive null vectors $$u$$ of $$L$$ with $$u_0>0$$ (absolutely convergent for $$\mathrm{Re}\,s>2$$).

<div class="proposition" markdown="1">

**Proposition 10** (the Eisenstein sector). For odd $$p$$,

$$
\sum_{L'\in N_p(L)}E_s(L')=\big(p^s+c_p+p^{2-s}\big)E_s(L),\qquad c_p=\begin{cases}0&p\equiv3\ (4)\\2p&p\equiv1\ (4).\end{cases} \tag{5}
$$

</div>

The idea is to classify, for a fixed primitive null vector $$u$$ of $$L$$, how the neighbour lattices contain multiples of it.

<div class="proof" markdown="1">

*Proof.* Let $$u$$ be primitive null in $$L$$ and $$L'=L'_w$$. (a) If $$w\equiv\lambda u\pmod p$$, then $$u\in L_w$$, and $$u/p\in L'$$: writing $$w=\lambda u+pr$$, $$Q(w)\equiv0\pmod{p^2}$$ gives $$B(u,r)\equiv0\pmod p$$, hence $$B(r,w)=\lambda B(r,u)+pQ(r)\equiv0$$, so $$r\in L_w$$; with $$\mu\lambda\equiv1\pmod p$$, $$\mu\lambda=1+pk$$, one finds $$u/p=\mu w/p-\mu r-ku\in L'$$. The vector $$u/p$$ is primitive in $$L'$$, and contributes $$p^s(u_0)^{-s}$$. There is exactly one such line. (b) If $$B(u,w)\equiv0$$ and $$\bar w\ne\langle\bar u\rangle$$, then $$u\in L_w$$ and $$u$$ is primitive in $$L'$$, contributing $$(u_0)^{-s}$$. Such lines are the isotropic lines of $$\bar u^\perp$$ other than $$\langle\bar u\rangle$$; the quotient $$\bar u^\perp/\bar u$$ is a two-dimensional quadratic space that is split iff $$p\equiv1\pmod4$$, with two isotropic lines, each lifting to $$p$$ lines of $$\bar u^\perp$$ besides $$\langle\bar u\rangle$$ itself, so there are $$c_p$$ of them. (c) The remaining $$p^2$$ lines have $$B(u,w)\not\equiv0$$; then $$u\notin L'$$, $$pu\in L_w$$ is primitive in $$L'$$, and contributes $$p^{-s}(u_0)^{-s}$$. The total count is $$p^2+1$$ or $$(p+1)^2$$ lines, consistent with $$1+c_p+p^2$$. Conversely every primitive null vector of $$L'$$ is one of $$u/p$$, $$u$$, $$pu$$ for a unique primitive null $$u\in L$$. Summing over $$u$$ gives (5). $$\square$$

</div>

*Numerical check.* With the cutoffs $$u_0\le40$$ ($$p=3$$) and $$u_0\le30$$ ($$p=5$$) and $$s=4$$, the ratio of $$\sum_{L'}E_4(L')$$ to $$E_4(\mathbb Z^4)$$ is $$81.122$$ against $$3^4+3^{-2}=81.111$$, and $$635.21$$ against $$5^4+10+5^{-2}=635.04$$.

At $$s=1+it$$, (5) reads $$2p\cos(t\ln p)+c_p$$. The functions $$E_s$$ are the Eisenstein series on the quotient $$H^3/\Gamma_0$$ of the space of unit future timelike vectors (rest frames) by the integral Lorentz group, with Laplace eigenvalue $$s(2-s)=1+t^2$$, so that on this sector $$T_p=\sum_{L'\in N_p(L)}$$ acts as $$2p\cos\big(\ln p\sqrt{\Delta-1}\big)+c_p$$. This is the cosine propagator for a wave equation on the ultra-static metric $$-d\tau^2+g_{H^3}$$ evaluated at the time $$\tau=\ln p$$. Its spatial slice is the space of rest frames, which is not spacetime. We take the identification of $$E_s$$ with the standard Eisenstein series as known and do not reprove it. Hecke operators commute with the Laplacian, and whether the same cosine form holds on the cuspidal part is not settled here: it would require an equidistribution statement for the Hecke eigenvalues that we do not prove.

<div class="proposition" markdown="1">

**Proposition 11** (trajectory point sets are singular with respect to Poisson). Let $$L_0$$ be a Haar-random element of $$\mathcal Y\cong G/\Gamma_0$$ with an independent uniform shift, and $$L_{k+1}$$ uniformly distributed in $$N_p(L_k)$$ (a random walk, which commutes with $$G$$). Then $$\bigcup_{k\le n}L_k$$ is a Poincaré-invariant point process each of whose points has infinitely many exactly null partners, so its law is mutually singular with every Poisson or mixed Poisson process with almost surely diffuse mean measure.

</div>

<div class="proof" markdown="1">

*Proof.* Invariance follows from equivariance. Every $$L\in\mathcal Y$$ is isometric to $$\mathbb Z^4$$ with the form $$Q$$, which contains infinitely many non-zero null vectors (for instance $$(t,x,y,z)$$ with $$t^2=x^2+y^2+z^2$$), so each point of each $$L_k$$ has infinitely many null partners in the same lattice. For a Poisson process with diffuse mean measure the probability that two points are exactly null-related is 0. $$\square$$

</div>

The correspondence therefore survives as a generator of invariant random lattices (kinematics, as in section 2) or as a wave equation on frame space, and is excluded as dynamics on spacetime in the sense above: a step is global, the metric is the input form, and the union of the lattices along a trajectory is not a sprinkling. The rescaled large-$$n$$ limit of the trajectory point sets is not computed.

## 6. Does the causal order carry a hand?

Parity $$P$$ reverses one spatial axis; it is an automorphism of the Minkowski causal order, so the causal order of a point set does not by itself change under reflection. A *hand* of a Poincaré-invariant point process, in the sense used here, is a failure of parity invariance of its law that cannot be removed by averaging: the process has a hand if some ergodic component is not invariant in law under $$P$$. This is a property of ergodic components and not of the abstract order: for any law $$\mu$$ on embedded relations, pushing forward to abstract relations gives the same law for $$\mu$$ and $$P_*\mu$$, because $$P$$ relabels the embedding and preserves the relation.

<div class="proposition" markdown="1">

**Proposition 12** (rules on few points are parity blind). Let a rule decide whether $$x\prec y$$ from the positions of at most three points ($$x$$, $$y$$ and at most one auxiliary point), equivariantly under the proper orthochronous Poincaré group. Then it is invariant under parity as well. The first parity-odd invariant needs four points: three difference vectors spanning a spacelike three-plane, together with the time orientation.

</div>

<div class="proof" markdown="1">

*Proof.* The invariants of a configuration are those of its difference vectors, which span a subspace $$W$$ of dimension at most 2. Then $$W^\perp$$ has dimension at least 2 and contains a spacelike vector $$s$$ (a subspace without spacelike vectors has dimension at most 1). The reflection $$R_s$$ in $$s^\perp$$ is improper, orthochronous, and fixes $$W$$ pointwise. For an improper orthochronous $$\Pi$$ (parity) the product $$\Pi R_s$$ is proper orthochronous and maps the configuration $$c$$ to $$\Pi c$$, so a rule invariant under the proper group gives the same value on $$c$$ and $$\Pi c$$. With four points the three difference vectors can span a spacelike three-plane $$W$$; the elements fixing $$W$$ pointwise are $$\pm1$$ on the timelike line $$W^\perp$$, and the non-trivial one, $$-1$$, reverses time and is not orthochronous. So the proper orthochronous orbit of the configuration differs from that of its parity image, and the sign of the determinant of the three vectors with the future normal is an invariant that is odd under parity. $$\square$$

</div>

<div class="proposition" markdown="1">

**Proposition 13** (no hand on Poisson causal orders). A Poisson process of constant intensity is invariant in law under $$P$$ and ergodic; a mixed Poisson process with a random constant intensity has Poisson ergodic components. Hence neither has a hand. A chiral relation built on a Poisson point set by a rule on at least four points is therefore not the causal order of the embedding.

</div>

<div class="proof" markdown="1">

*Proof.* The law of a Poisson process of intensity $$\rho$$ is invariant under every affine isometry, including $$P$$; it is ergodic under translations (and so under $$P$$). The ergodic decomposition of a mixed Poisson process with random constant intensity is the mixture of the Poisson laws. The second statement follows from Proposition 12: a parity-odd relation needs a parity-odd rule, which uses at least four points, and a relation between two points that depends on auxiliary points is not their causal relation. $$\square$$

</div>

The Haar-random lattices of section 2 are $$P$$-invariant, so they have no hand either. A hand needs an arithmetic lattice that is not equivalent to its mirror image.

<div class="proposition" markdown="1">

**Proposition 14** (chiral random lattices exist). Let $$L$$ be an integral lattice on $$\mathbb Q^4$$ with form $$\eta=\mathrm{diag}(1,-1,-1,-1)$$ and Gram matrix

$$
\begin{pmatrix}289&119\\119&48\end{pmatrix}\oplus-\begin{pmatrix}83521&18496\\18496&4097\end{pmatrix},
$$

which has signature $$(1,3)$$ and determinant $$-17^6$$. Then $$L$$ has no improper automorphism. Consequently the random lattice $$\Lambda=gL+t$$, with $$g$$ Haar on $$SO^+(1,3)/SO^+(L)$$ (finite volume by the Borel–Harish-Chandra theorem) and $$t$$ uniform on the torus $$\mathbb R^4/gL$$, is, after rescaling to unit density, a Poincaré-invariant, ergodic point process whose law is mutually singular with its parity image: it has a hand.

</div>

The proof uses the structure of the isometry groups of lattices over the $$p$$-adic integers, through the spinor norm, which is the homomorphism $$\theta$$ from the orthogonal group to $$\mathbb Q_p^\times/(\mathbb Q_p^\times)^2$$ taking a reflection in a vector $$v$$ to the class of $$Q(v)$$. We use one cited theorem: for a lattice $$L_p$$ over $$\mathbb Z_p$$ with $$p$$ odd, the orthogonal group $$O(L_p)$$ is generated by reflections (Kneser; see O'Meara, *Introduction to Quadratic Forms*, §92). An improper automorphism $$\sigma$$ of $$L$$ is an improper isometry of the rational quadratic space, hence (Cartan–Dieudonné) a product of an odd number of rational reflections, and its spinor norm is one rational number $$r$$ modulo squares; it is also an improper isometry of $$L_p$$ for every prime $$p$$, with spinor norm the image of $$r$$ in $$\mathbb Q_p^\times/(\mathbb Q_p^\times)^2$$.

<div class="proof" markdown="1">

*Proof.* *Local structure at 17.* Over $$\mathbb Z_{17}$$, $$L$$ is diagonalised by an orthogonal change of basis into the form with diagonal entries of $$17$$-adic valuations $$0,1,2,3$$, each with a unit part of the same square class $$u$$, a non-residue modulo 17 (we orthogonalised successively, always with a pivot of minimal valuation, so that the new basis vectors stay in $$L\otimes\mathbb Z_{17}$$, and read off the valuations and Legendre symbols of the unit parts: the diagonal is $$48,\,-4097,\,-289/48,\,-4913/241$$, with valuations $$0,1,2,3$$ and Legendre symbol $$-1$$ for all four unit parts). Write $$Q(e_i)=\varepsilon_i17^{a_i}$$ with $$a_i=0,1,2,3$$.

*Spinor norms of reflections in $$O(L_{17})$$.* Let $$v=\sum c_ie_i$$ be primitive and $$\tau_v\in O(L_{17})$$, which means $$2B(e_i,v)/Q(v)\in\mathbb Z_{17}$$ for all $$i$$, that is $$\mathrm{val}\,Q(v)\le a_i+\mathrm{val}\,c_i$$ for all $$i$$. Put $$\mu=\min_i(a_i+2\mathrm{val}\,c_i)\le\mathrm{val}\,Q(v)$$ and let $$j$$ attain the minimum. Then $$\mu\le\mathrm{val}\,Q(v)\le a_j+\mathrm{val}\,c_j\le a_j+2\mathrm{val}\,c_j=\mu$$, so $$\mathrm{val}\,c_j=0$$, $$\mathrm{val}\,Q(v)=a_j=\mu$$, and, as the $$a_i$$ are distinct, $$j$$ is the unique index attaining the minimum. Then $$Q(v)\equiv\varepsilon_j17^{a_j}c_j^2\pmod{17^{a_j+1}}$$, so the class of $$Q(v)$$ is that of $$\varepsilon_j17^{a_j}$$: $$u$$ for $$a_j$$ even and $$17u$$ for $$a_j$$ odd. Reflections in $$O(L_{17})$$ therefore have spinor norm in $$\{u,17u\}$$ modulo squares.

*Improper elements.* By the cited generation theorem, an improper element of $$O(L_{17})$$ is a product of an odd number of such reflections; its spinor norm is in $$u\{1,17\}\cdot(\mathbb Q_{17}^\times)^2=\{u,17u\}$$ (as $$u^2$$ is a square). At a prime $$p\ne2,17$$, $$L_p$$ is unimodular (the determinant is a power of 17) so reflections in $$O(L_p)$$ have unit spinor norm and so does every element.

*Global contradiction.* Suppose $$\sigma\in O(L)$$ is improper, with spinor norm $$r\in\mathbb Q^\times/(\mathbb Q^\times)^2$$. Then $$r$$ has even valuation at every prime $$p\ne2,17$$, so $$r\in\{\pm2^a17^b\}$$ modulo squares. In $$\mathbb Q_{17}$$, $$-1$$ and $$2$$ are squares ($$17\equiv1\pmod8$$), so $$r\equiv17^b$$ is $$1$$ or $$17$$ there. But $$\sigma$$ is an improper element of $$O(L_{17})$$, so its spinor norm is $$u$$ or $$17u$$, and $$u\notin\{1,17\}$$ modulo squares because $$u$$ is a non-residue. This contradicts the equality of the rational spinor norm with its $$17$$-adic image.

*The process.* If the laws of $$\Lambda$$ and $$\Pi\Lambda$$ had a common ergodic component they would coincide, because each is ergodic. The support of the law of $$\Lambda$$ is the single orbit $$SO^+(1,3)\cdot L$$ of lattices (plus translations), and that of $$\Pi\Lambda$$ is $$SO^+(1,3)\cdot\Pi L$$. If these orbits met, $$\Pi L=AL$$ for some proper orthochronous $$A$$, so $$A^{-1}\Pi$$ would be an improper orthochronous automorphism of $$L$$. In even dimension, $$-1$$ is an automorphism of every lattice with $$\det(-1)=1$$; if $$g$$ is any improper automorphism, one of $$g$$, $$-g$$ is orthochronous and both are improper, so the absence of an improper automorphism excludes this. Hence the orbits are disjoint, and the laws are mutually singular. Invariance and ergodicity follow as in Theorem 1, with Haar measure on the finite-volume orbit $$SO^+(1,3)/SO^+(L)$$. $$\square$$

</div>

*Numerical check.* The Gram matrix has determinant $$-17^6=-24137569$$, signature $$(1,3)$$, and diagonal entries of odd parity at the first, third and fourth places, so the lattice is odd. The $$17$$-adic diagonalisation gives the four entries stated, with Legendre symbols $$-1$$. Among $$4\cdot10^5$$ random vectors of $$L\otimes\mathbb Z_{17}$$, $$3.99\cdot10^5$$ define reflections in $$O(L_{17})$$ (the condition $$\mathrm{val}\,Q(v)\le\min_i\mathrm{val}(Gv)_i$$), and their spinor norms fall in exactly the two classes $$u$$ ($$3.77\cdot10^5$$ vectors) and $$17u$$ ($$2.2\cdot10^4$$ vectors), as predicted. A consistency check on the squares modulo 17: $$-1$$ and $$2$$ are squares, $$3$$ is not.

The causal order of the lattice $$L$$ is not shown here to be chiral in the sense of Proposition 15 below: that would require that every automorphism of the order be affine, which we do not prove. Exact parity violation of this kind coincides with time-reversal violation, because $$-1$$ is an automorphism of every lattice, so $$PT$$ is exact.

A finite relation can show the mechanism with no embedding at all. Let $$R$$ be a relation on a finite set $$X$$ with matrix $$A$$ and converse $$R^T$$ (matrix $$A^T$$), $$G=\mathrm{Aut}(R)$$, and $$N$$ the normaliser of $$G$$ in the symmetric group of $$X$$. Every $$\pi\in N$$ induces an automorphism of $$G$$ and so permutes its irreducible characters. Call $$R$$ *rigidly chiral at $$\chi$$* if $$\chi$$ occurs in the permutation module $$\mathbb C[X]$$ and $$\bar\chi$$ is not in the $$N$$-orbit of $$\chi$$. Let $$H$$ be the sum of the isotypic components of $$\mathbb C[X]$$ over the $$N$$-orbit of $$\chi$$.

<div class="proposition" markdown="1">

**Proposition 15** (a rigid hand on a finite relation). Suppose $$R$$ is rigidly chiral at $$\chi$$. (i) Every permutation $$\pi$$ with $$\pi R\pi^{-1}\in\{R,R^T\}$$ preserves $$H$$ and the complex conjugate $$\bar H$$, and $$H\cap\bar H=0$$. (ii) If $$\pi R\pi^{-1}=R^T$$, then for every word $$w$$ in $$A$$ and $$A^T$$, the spectrum of $$w\vert _H$$ equals the complex conjugate of the spectrum of the reversed word $$w^{\rm rev}\vert _H$$. In particular, if some word violates this identity then $$R$$ is not isomorphic to its converse, and no relabelling, with or without reversal of the relation, exchanges $$H$$ with its mirror $$\bar H$$.

</div>

<div class="proof" markdown="1">

*Proof.* (i) $$\mathrm{Aut}(R^T)=\mathrm{Aut}(R)=G$$, so such a $$\pi$$ normalises $$G$$ and induces an automorphism $$\alpha$$ of $$G$$. It maps the $$\psi$$-isotypic part to the $$\psi\circ\alpha^{-1}$$-isotypic part, which stays in the orbit, so $$H$$ is preserved; permutation matrices are real, so $$\bar H$$ is preserved too. If the orbit of $$\chi$$ and the conjugate of that orbit shared a character, $$\bar\chi$$ would lie in the orbit of $$\chi$$, contradicting rigidity; hence $$H\cap\bar H=0$$. (ii) $$\pi A\pi^{-1}=A^T$$ gives $$\pi\,w(A,A^T)\,\pi^{-1}=w(A^T,A)=(w^{\rm rev}(A,A^T))^T$$. $$A$$ and $$A^T$$ commute with $$G$$ and so preserve $$H$$ and its orthogonal complement. For a real matrix $$M$$ with this property, $$M^T\vert _H$$ is the adjoint of $$M\vert _H$$ in an orthonormal basis of $$H$$, with the complex-conjugate spectrum. Since $$\pi$$ preserves $$H$$, the spectrum of $$w\vert _H$$ equals that of $$w(A^T,A)\vert _H$$, which is the conjugate of that of $$w^{\rm rev}\vert _H$$. $$\square$$

</div>

<div class="proposition" markdown="1">

**Proposition 16** (examples and a lower bound on height). (i) Let $$G=\mathbb Z_p\rtimes\mathbb Z_q$$ be a Frobenius group with $$q$$ an odd prime dividing $$p-1$$. Every automorphism of $$G$$ fixes every linear character $$\omega$$ of $$G$$ and $$\bar\omega\ne\omega$$ for $$\omega\ne1$$; so any relation with $$\mathrm{Aut}(R)\cong G$$ in which a non-trivial $$\omega$$ occurs is rigidly chiral at $$\omega$$. (ii) If every arc of $$R$$ runs from a set $$X_0$$ to a disjoint set $$X_1$$ (a relation of height two), every word satisfies the identity of Proposition 15(ii), so such a relation cannot show a spectral hand.

</div>

<div class="proof" markdown="1">

*Proof.* (i) $$\mathbb Z_p$$ is characteristic, so an automorphism $$\alpha$$ acts on it by multiplication by some unit. For $$g\in G$$, $$\alpha\,\mathrm{conj}_g\,\alpha^{-1}=\mathrm{conj}_{\alpha(g)}$$ as maps of $$\mathbb Z_p$$; multiplications commute, so $$\alpha(g)$$ acts on $$\mathbb Z_p$$ as $$g$$ does. Since $$G/\mathbb Z_p$$ acts faithfully on $$\mathbb Z_p$$, $$\alpha$$ induces the identity on $$G/\mathbb Z_p$$ and fixes every character pulled back from it, which are the linear characters. For odd $$q$$ a non-trivial linear character takes non-real values. (ii) $$A^2=(A^T)^2=0$$, so a non-zero word alternates its letters; words of odd length move between the two sets and square to zero, so their spectrum is $$0$$; words of even length are $$(AA^T)^k$$ or $$(A^TA)^k$$, real symmetric, and $$AA^T$$ and $$A^TA$$ have the same non-zero spectrum. $$\square$$

</div>

*Example, computed.* Let $$G=F_{21}=\{x\mapsto ax+b:\ a\in\{1,2,4\},\,b\in\mathbb F_7\}$$ and $$X=G$$ with the relation $$x\to sx$$ for $$s\in S=\{x+1,\,x+2,\,2x\}$$ (63 arcs). Right multiplications are automorphisms. The automorphism group of the digraph, by exhaustive search, has order 21, so it is exactly the right regular action of $$F_{21}$$, and there is no isomorphism from the digraph to its converse. The linear character $$\omega$$ of $$F_{21}$$ of order 3 occupies a one-dimensional isotypic component, on which the matrix acts by $$\sum_{s\in S}\omega(s)=2+e^{2\pi i/3}=1.5+0.866i$$; the spectrum on $$H$$ is not closed under complex conjugation, which a rigid hand allows (the matrix is real, so its full spectrum is closed under conjugation, but its restriction to $$H$$ is not). The 3-dimensional characters $$\psi,\bar\psi$$ of $$F_{21}$$, by contrast, are exchanged by an automorphism of $$G$$, for instance conjugation by $$x\mapsto3x$$, which sends the translation $$x\mapsto x+1$$ to $$x\mapsto x+3$$ in the other class of elements of order 7; their chirality is removable, like that of Weyl spinors under parity. These examples are in the spirit of the type-I groups of CP-violation model building, in which a finite group has no automorphism that maps a representation to its conjugate [[Chen et al.]](#ref-chen). The existence of rigid hands on finite relations shows that "an undecorated relation cannot carry chirality" is not a theorem about relations as such. For orders embedded in Minkowski space the obstruction is that parity, which exchanges the two characters, is an automorphism of the Minkowski order.

## 7. What the theorems do not exclude

1. Non-Poisson invariant point sets with other law than Haar: orbit lattices of anisotropic forms (see the page on [single realisations](/quantum-gravity/per-realisation-kinematics/)), mixtures, and thinnings or superpositions of Haar lattices. Theorem 1 and Proposition 2 concern Haar-random lattices only; the corresponding statement for other invariant lattices is a separate computation.
2. Invariant processes of null lines, null hyperplanes or complexified positions with infinite mean measure but almost surely locally finite realisations, which Theorem 4 and Proposition 3 do not treat.
3. Spacelike lines, for which the invariant measure exists and the incidence is not computed, and twistor substrates, for which the argument above is a sketch.
4. Frame marks whose law is not invariant under the stabiliser of the mark, and non-equivariant finite rules (a rule that uses a fixed frame breaks Lorentz invariance of the law of the rule's output). Rules that sum over infinitely many neighbours with cancelling weights are also outside Theorem 6.
5. Hecke correspondences as generators of invariant random lattices, as wave equations on frame space, or at the real place combined with a state that supplies the frame.
6. Chiral lattices, which exist; whether their causal order is rigidly chiral depends on whether all automorphisms of the order are affine, and the kinematics of their kernels (the Poisson-weighted dictionary does not apply to a lattice) is not computed here.

## 8. Open points

The infinite-mean case of Theorem 4 and Proposition 3, a non-Poisson invariant law with almost surely locally finite projection, is undecided; a mass-transport argument for the non-amenable group $$SO^+(1,3)$$ might settle whether a non-empty invariant hyperboloid-marked process with locally finite ground process exists. Whether every Poincaré-homogeneous space of affine subspaces other than points fails to combine an invariant Radon measure with finite incidence is not proved. For Haar-random lattices in four dimensions the void law is expected to decay like $$1/V$$ as in two dimensions, but this page proves it only in two dimensions. The rescaled limit of Hecke trajectories and the behaviour of the cuspidal part under $$T_p$$ are not determined. The hypothesis that all order automorphisms of a lattice causal order are affine is not proved.

## References

1. <a id="ref-bhs"></a>Bombelli, Henson, Sorkin, "Discreteness without symmetry breaking: a theorem," Mod. Phys. Lett. A 24 (2009) 2579, [arXiv:gr-qc/0605006](https://arxiv.org/abs/gr-qc/0605006).
2. <a id="ref-dhs"></a>Dowker, Henson, Sorkin, "Quantum gravity phenomenology, Lorentz invariance and discreteness," Mod. Phys. Lett. A 19 (2004) 1829, [arXiv:gr-qc/0311055](https://arxiv.org/abs/gr-qc/0311055).
3. <a id="ref-sa"></a>Saravani, Aslanbeigi, "On the causal set-continuum correspondence," [arXiv:1403.6429](https://arxiv.org/abs/1403.6429).
4. <a id="ref-chen"></a>Chen, Fallbacher, Mahanthappa, Ratz, Trautner, "CP violation from finite groups," Nucl. Phys. B 883 (2014) 267, [arXiv:1402.0507](https://arxiv.org/abs/1402.0507).
