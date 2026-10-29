---
title: Local rules of change on a sprinkling
date: 2026-10-29
category: entropy
order: 3
description: A Poisson sprinkling in Minkowski space carries no neighbours, only an order. No rule that completes a field value at a sprinkled element from finitely many other sprinkled elements is exactly consistent with the wave equation, in any dimension. Laws that re-decide each pair of elements from separate invariant statistics cannot hold a symmetric manifold-like order fixed. For path sums over layered orders the suppression at fixed outer layers depends on the action only through a link phase.
status: Theorems proved in full for exact completion rules, for laws built from separate statistics and for the layered path sum at fixed outer layers; the uniform bound at Kleitman-Rothschild proportions misses by a logarithmic factor; numerical checks in two and four dimensions at sprinkled sizes up to a few thousand points.
excludes: Exactly wave-consistent completion rules on finitely many sprinkled elements (any dimension); order-revising laws that read only separate per-element statistics, on orders with a symmetry that moves a point to one with a different future; kernel-universal path-sum suppression of layered orders beyond the link phase at fixed outer layers.
---

A causal set sprinkled into Minkowski space has no coordinates and no neighbours. All it carries is a partial order. This page asks whether a dynamics can be written on it by a rule that is local in the only available sense, and treats three kinds of rule. The answers are as follows.

A rule that fills in the value of a field at a sprinkled element from its values at finitely many other sprinkled elements cannot be exactly consistent with the massless wave equation, in any dimension at least two (Theorem 3). A law that revises the relations of the order, deciding each pair of elements from separate automorphism-invariant statistics of the two elements, cannot leave fixed an order that has a symmetry moving a point to a point with a different future; the cone-count dominance map, a concrete law of this kind, relates a positive fraction of spacelike pairs on smooth configurations (Propositions 5 and 6). For a path sum over three-layer orders with a finite-depth interval action, the suppression at fixed outer layers is bounded by a rate that depends on the action only through the link phase $$\beta C_0$$. There is no suppression at all when that phase is a multiple of $$2\pi$$, and the uniform bound that would give suppression at the proportions that dominate the count of orders misses by a logarithmic factor (Theorems 8 to 10 and Proposition 11).

Sorkin [[Sorkin]](#ref-sorkin) argued that Lorentz-invariant discreteness forces non-locality; the operators that do exist on causal sets, the d'Alembertian of Benincasa and Dowker [[BD]](#ref-bd) and its generalisations [[ASS]](#ref-ass), and the propagator of Johnston [[Johnston]](#ref-johnston), are built from interval counts over mesoscopic regions. Theorem 3 is an exact statement of this kind for completion rules. In one space dimension a rewriting rule that is exactly consistent with the wave equation does exist, on the lattice of null coordinates: the rule that fills the fourth corner of a rectangle from the other three. Its algebra is that of a heap, treated as pure algebra in [[ZAB]](#ref-zab), and in the abelian case it is the parallelogram identity that characterises solutions of the one-dimensional wave equation [[KRud]](#ref-krud). Section 2 shows that this is the only rule, under two assumptions, for which the answer does not depend on how a large rectangle is cut into small ones, and section 3 shows why it cannot be moved onto the sprinkled elements. The path-sum result builds on the theorem of Kleitman and Rothschild [[KR75]](#ref-kr75) that almost all partial orders have three layers, and on the work of Loomis and Carlip [[LC]](#ref-lc), Carlip, Carlip and Surya [[CCS1]](#ref-ccs1) [[CCS2]](#ref-ccs2) and Mathur, Singh and Surya [[MSS]](#ref-mss) on whether the Benincasa–Dowker action suppresses them. What is added here is an exact transfer-matrix treatment of the sum over a three-layer class for every finite-depth action, which separates what depends on the choice of action from what does not. The place of these results among other approaches is mapped on [The landscape of quantum-gravity theories](/quantum-gravity/landscape/).

## 1. Three kinds of rule

A field theory on a sprinkling needs a way to propagate values. Three kinds of rule have been proposed or are natural.

A *completion rule* assigns to a target element $$t$$ and a finite set $$S$$ of other elements a function that predicts $$\varphi(t)$$ from $$\varphi\vert_S$$. Sections 2 and 3 treat these. A *pair law* revises the order: it takes a finite order and returns a new relation, deciding for each pair $$x,y$$ whether $$x$$ precedes $$y$$ in the new relation. Section 4 treats pair laws that read only statistics attached to $$x$$ and to $$y$$ separately. A *path-sum weight* does not change an order; it assigns a phase to each order, and the dynamics is the sum over orders. Sections 5 and 6 treat the sum over layered orders. Section 5 also states a rank-one property of the decoherence functional of the sequential-growth models that carry a single amplitude per history.

Throughout, "sprinkling" means a Poisson point process of density $$\rho$$ in a region of $$d$$-dimensional Minkowski space, with the induced causal order, and signature is mostly plus. A statement is called a theorem or proposition only if it is proved on this page; numerical statements say what was computed and with which parameters.

## 2. The null lattice: which rectangle rules are confluent

*The picture.* In $$1+1$$ dimensions use null coordinates $$u,v$$. The massless wave equation is $$\partial_u\partial_v\varphi=0$$, its solutions are $$\varphi=f(u)+g(v)$$, and on every rectangle with corners $$b=(u_1,v_1)$$, $$l=(u_1,v_2)$$, $$r=(u_2,v_1)$$, $$t=(u_2,v_2)$$ they satisfy $$\varphi(t)=\varphi(l)+\varphi(r)-\varphi(b)$$. The identity holds for rectangles of every size, so the value at $$t$$ can be computed from the three other corners directly, or after cutting the rectangle into two and computing the value at the new corner first. That the two routes agree is the property studied here. We ask which rules $$t=F(l,b,r)$$, for values in an arbitrary set $$X$$, have it.

*The setting.* Let $$\Lambda=\{0,\dots,N\}^2$$ be the null lattice. A rectangle is a quadruple $$b=(a,c)$$, $$l=(a,j)$$, $$r=(i,c)$$, $$t=(i,j)$$ with $$a<i$$ and $$c<j$$, of any size. A field is a map from a subset of $$\Lambda$$ to $$X$$. Given a map $$F:X^3\to X$$, a rewriting step picks a rectangle whose corners $$b,l,r$$ carry values and whose corner $$t$$ does not, and sets $$\varphi(t)=F(\varphi(l),\varphi(b),\varphi(r))$$. Two assumptions are made on $$F$$.

* (H1) The same $$F$$ is used for every rectangle, whatever its size.
* (H2) $$F$$ is reversible in the two outer arguments: for fixed $$b$$ and $$r$$ the map $$l\mapsto F(l,b,r)$$ is a bijection of $$X$$, and for fixed $$l$$ and $$b$$ the map $$r\mapsto F(l,b,r)$$ is a bijection.

Goursat data are the values of $$\varphi$$ on the two lines $$\{0\}\times\{0,\dots,N\}$$ and $$\{0,\dots,N\}\times\{0\}$$, agreeing at the origin. A history is a sequence of rewriting steps that fills $$\Lambda$$ starting from Goursat data. Histories exist, since a point $$t=(i,j)$$ with $$i,j\ge1$$ and $$i+j$$ minimal among the empty points has all three other corners of the unit rectangle filled. The rule $$F$$ is *confluent* if for every Goursat data every history ends with the same field on $$\Lambda$$.

A heap operation on a group $$G$$ is the map $$F(l,b,r)=l\,b^{-1}r$$. Writing the two splittings of a rectangle as identities gives, for all values in $$X$$,

$$
F\big(F(l,b,m),m,r\big)=F(l,b,r),\tag{1}
$$

$$
F\big(l,l_1,F(l_1,b,r)\big)=F(l,b,r).\tag{2}
$$

Identity (1) says that cutting a rectangle with bottom edge $$b\to r$$ along a vertical line through an intermediate bottom point $$m$$ and filling the left piece first gives the same value at the top right corner as filling the big rectangle at once. Identity (2) says the same for a cut along a horizontal line through an intermediate point $$l_1$$ of the left edge.

<div class="theorem" markdown="1">
**Theorem 1** (Confluent rectangle rules are heaps). Let $$F:X^3\to X$$ satisfy (H1) and (H2). The following are equivalent.

(a) $$F$$ is confluent.

(b) $$F$$ satisfies (1) and (2).

(c) There is a group structure on $$X$$ with $$F(l,b,r)=l\,b^{-1}r$$.

In that case, for Goursat data $$\varphi(0,j)=\alpha(j)$$ and $$\varphi(i,0)=\beta(i)$$, every history produces $$\varphi(i,j)=A(j)\,B(i)$$ with $$A(j)=\alpha(j)\alpha(0)^{-1}$$ and $$B(i)=\beta(i)$$. If $$X$$ is a manifold and $$F$$ is smooth, the group is a Lie group.
</div>

The idea is that (1) and (2) are the two ways of cutting a rectangle, and that reversibility turns them into the associativity law of the product $$x\cdot y=F(x,e,y)$$ for a fixed element $$e$$. Everything is routine algebra. The one point that carries content is that both identities are needed: each alone admits solutions that are not heaps, and neither (1) nor (2) alone forces a group.

<div class="proof" markdown="1">
*Proof.* (a)$$\Rightarrow$$(b). Take $$\Lambda=\{0,1,2\}\times\{0,1\}$$ with Goursat values $$\varphi(0,0)=b$$, $$\varphi(0,1)=l$$, $$\varphi(1,0)=m$$, $$\varphi(2,0)=r$$, which are arbitrary elements of $$X$$. One history fills $$(1,1)$$ from the unit rectangle at the origin and then $$(2,1)$$ from the rectangle with bottom edge from $$(1,0)$$ to $$(2,0)$$. It gives $$F(F(l,b,m),m,r)$$ at $$(2,1)$$. Another history fills $$(2,1)$$ at once from the rectangle with bottom corner $$(0,0)$$ and gives $$F(l,b,r)$$. Confluence forces (1). The same construction on $$\{0,1\}\times\{0,1,2\}$$ with data $$\varphi(0,0)=b$$, $$\varphi(0,1)=l_1$$, $$\varphi(0,2)=l$$, $$\varphi(1,0)=r$$ gives (2).

(b)$$\Rightarrow$$(c). Put $$r=m$$ in (1). By (H2) the element $$x=F(l,b,m)$$ runs over all of $$X$$ as $$l$$ does, so

$$
F(x,m,m)=x\qquad\text{for all }x,m.\tag{3}
$$

Put $$l=l_1$$ in (2). As $$r$$ varies, $$y=F(l_1,b,r)$$ runs over $$X$$, so $$F(l_1,l_1,y)=y$$ for all $$l_1,y$$. Fix $$e\in X$$ and define $$x\cdot y=F(x,e,y)$$. The two identities with the middle argument equal to $$e$$ give $$x\cdot e=x$$ and $$e\cdot y=y$$. By (H2), for every $$m$$ the map $$l\mapsto l\cdot m$$ is a bijection, with inverse $$x\mapsto x/m$$, and $$r\mapsto m\cdot r$$ is a bijection with inverse $$y\mapsto m\backslash y$$. Now put $$b=e$$ in (1): $$F(l\cdot m,m,r)=l\cdot r$$. Writing $$x=l\cdot m$$ this reads

$$
F(x,m,r)=(x/m)\cdot r.\tag{4}
$$

Put $$b=e$$ in (2): $$F(l,l_1,l_1\cdot r)=l\cdot r$$. Writing $$y=l_1\cdot r$$ this reads

$$
F(l,l_1,y)=l\cdot(l_1\backslash y).\tag{5}
$$

Comparing (4) and (5) for the same arguments gives $$(x/m)\cdot y=x\cdot(m\backslash y)$$. Choose $$x=a\cdot m$$ and $$y=m\cdot c$$. Then $$x/m=a$$ and $$m\backslash y=c$$, so $$a\cdot(m\cdot c)=(a\cdot m)\cdot c$$ for all $$a,m,c$$. Thus $$(X,\cdot)$$ is an associative quasigroup with identity $$e$$, that is a group, with $$x/m=x\,m^{-1}$$, and (4) is $$F(x,m,r)=x\,m^{-1}r$$.

(c)$$\Rightarrow$$(a). Let $$F$$ be the heap operation of a group. Along any history every filled point $$(i,j)$$ carries $$A(j)B(i)$$: this holds on the Goursat lines, since $$A(0)=e$$ and $$B(0)=\alpha(0)$$, and a step with corners $$A(j)B(a)$$, $$A(c)B(a)$$, $$A(c)B(i)$$ produces $$A(j)B(a)\,B(a)^{-1}A(c)^{-1}\,A(c)B(i)=A(j)B(i)$$. The final field is therefore independent of the history.

For the last statement, $$x\cdot y=F(x,e,y)$$ is smooth, and so is the inverse $$x^{-1}=F(e,x,e)$$. $$\square$$
</div>

*Check.* We enumerated all maps $$F:X^3\to X$$ that are reversible in the sense of (H2) and satisfy (1) and (2), on sets of $$n=2,3,4$$ elements, by exhaustive search. There are $$1,1,4$$ of them, and all are heap operations; the number of heap operations on an $$n$$-element set is $$\sum_G (n-1)!/\lvert\mathrm{Aut}\,G\rvert$$ over isomorphism classes of groups of order $$n$$, which is $$1,1,4$$ for $$n=2,3,4$$ ($$\mathbb Z_4$$ contributes $$3$$ and $$\mathbb Z_2^2$$ contributes $$1$$). Imposing (1) alone gives $$1,2,24$$ solutions and (2) alone gives $$1,2,24$$ for $$n=2,3,4$$, of which only $$1,1,4$$ are heaps.

*Consequences.* In a group, a field $$\varphi(i,j)=A(j)B(i)$$ has $$\partial_v\varphi\,\varphi^{-1}$$ independent of $$u$$, so in the continuum limit the rule reproduces the field equation

$$
\partial_u\big(\partial_v\varphi\;\varphi^{-1}\big)=0,\tag{6}
$$

the principal chiral field equation in its chiral-factorised form [[ZAB]](#ref-zab), which for $$G=(\mathbb R,+)$$ is the free wave equation. If $$X=\mathbb R$$ and $$F$$ is smooth, every group structure is isomorphic to $$(\mathbb R,+)$$ by a diffeomorphism $$h$$, so $$F=h^{-1}(h(l)-h(b)+h(r))$$: a free wave equation for the redefined field $$h\circ\varphi$$.

<div class="corollary" markdown="1">
**Corollary 2** (Linear confluent rules). Let $$X=\mathbb R$$ and $$F(l,b,r)=\alpha l+\gamma b+\beta r$$ with real $$\alpha,\beta\neq0$$ and $$\gamma$$. Then $$F$$ is confluent if and only if $$F(l,b,r)=l-b+r$$.
</div>

<div class="proof" markdown="1">
*Proof.* Expanding (1) gives $$\alpha^2l+\alpha\gamma b+(\alpha\beta+\gamma)m+\beta r=\alpha l+\gamma b+\beta r$$, so $$\alpha^2=\alpha$$ and $$\alpha\beta+\gamma=0$$; with $$\alpha\neq0$$ this gives $$\alpha=1$$ and $$\gamma=-\beta$$. Expanding (2) gives $$\alpha l+(\gamma+\alpha\beta)l_1+\beta\gamma b+\beta^2r=\alpha l+\gamma b+\beta r$$, so $$\beta^2=\beta$$, hence $$\beta=1$$ and $$\gamma=-1$$. Conversely $$l-b+r$$ is the heap operation of $$(\mathbb R,+)$$. $$\square$$
</div>

The absence of a mass term has a one-line reason. For $$(\partial_u\partial_v+m^2)\varphi=0$$ the combination $$\varphi(t)-\varphi(l)-\varphi(r)+\varphi(b)$$ equals $$-m^2$$ times the integral of $$\varphi$$ over the rectangle, and two solutions can share the values at the three corners $$b,l,r$$ and differ in that integral, because $$\varphi(t)$$ depends on the data along the characteristic segments from $$b$$ to $$l$$ and from $$b$$ to $$r$$ and not only on their end points. No rule that sees only the corners reproduces it.

*Scope of Theorem 1.* The rule acts on the lattice $$\Lambda$$ of all pairs of ranks. A two-dimensional sprinkling of $$n$$ points is a two-order: each element has integer null ranks $$(i(x),j(x))\in\{1,\dots,n\}^2$$ obtained by ordering the points along each null direction, and $$x\prec y$$ exactly when both ranks increase. Every rank occurs once, so no rectangle of the rank lattice has all four corners among the sprinkled elements: two corners of a rectangle share a rank. The sprinkled elements occupy a fraction $$1/n$$ of the lattice sites, and a statement about rewriting on the lattice is a statement about a regular square lattice.

## 3. Rules on the sprinkled elements

Let $$\Sigma\subset\mathbb R^{1,d-1}$$ be a finite set of points, $$d\ge2$$, and let $$t\in\Sigma$$ and $$S\subset\Sigma\setminus\{t\}$$ be such that $$t$$ is not null-separated from any point of $$S$$. A *completion rule* is any function $$F:\mathbb R^S\to\mathbb R$$ that may depend on the positions of all points and on the order; no locality is assumed on the choice of $$S$$. It is *exactly consistent with $$\Box\varphi=0$$* if $$F(\varphi\vert_S)=\varphi(t)$$ for every smooth solution $$\varphi$$ of the wave equation on Minkowski space. The same definition applies to fields with values in a group $$G$$ that contains a non-trivial one-parameter subgroup $$\exp(sX)$$, with the principal chiral equation $$\partial^\mu(g^{-1}\partial_\mu g)=0$$ in place of the wave equation, and with exact consistency meaning $$F(g\vert_S)=g(t)$$ for all smooth solutions.

The idea is to use plane waves along a null direction. A function $$f(n\cdot x)$$ with $$n$$ null is a solution of the wave equation, and it is constant on each null hyperplane $$n\cdot x=\text{const}$$. The finite set $$S$$ meets only finitely many such hyperplanes. If one can choose $$n$$ so that the hyperplane through $$t$$ contains no point of $$S$$, then $$f$$ can vanish on all the hyperplanes through $$S$$ and equal $$1$$ on the one through $$t$$.

<div class="theorem" markdown="1">
**Theorem 3** (No exact completion rule on sprinkled elements). Let $$d\ge2$$, and let $$S$$ and $$t$$ be as above. There are smooth solutions $$\varphi_1,\varphi_2$$ of $$\Box\varphi=0$$ with $$\varphi_1=0$$, $$\varphi_2\vert_S=0$$ and $$\varphi_2(t)=1$$. Consequently no completion rule on $$(S,t)$$ is exactly consistent with the wave equation. The same holds for the principal chiral equation with values in a group $$G$$ that contains a non-trivial one-parameter subgroup. For a Poisson sprinkling these hypotheses hold almost surely, simultaneously for all $$S$$ and $$t$$.
</div>

<div class="proof" markdown="1">
*Proof.* Put $$\Delta_s=x_t-x_s$$ for $$s\in S$$; each $$\Delta_s$$ is non-zero and not null by assumption. The null directions are the vectors $$n=(1,\hat n)$$ with $$\hat n$$ on the unit sphere of $$\mathbb R^{d-1}$$ (two points when $$d=2$$). For such $$n$$, $$n\cdot\Delta_s=-\Delta_s^0+\hat n\cdot\vec\Delta_s$$. This vanishes for no $$\hat n$$ if $$\Delta_s$$ is timelike, because $$\lvert\hat n\cdot\vec\Delta_s\rvert\le\lvert\vec\Delta_s\rvert<\lvert \Delta_s^0\rvert$$; and if $$\Delta_s$$ is spacelike it vanishes on a set of directions that is a proper closed subset of the sphere without interior (empty for $$d=2$$ since $$\Delta_s$$ is not null). A finite union of such sets does not cover the sphere, so there is a null $$n$$ with $$n\cdot x_t\notin\{n\cdot x_s:s\in S\}$$. Choose a smooth $$f:\mathbb R\to\mathbb R$$ that vanishes at the finitely many numbers $$n\cdot x_s$$ and equals $$1$$ at $$n\cdot x_t$$. Then $$\varphi_2(x)=f(n\cdot x)$$ satisfies $$\Box\varphi_2=n^\mu n_\mu f''=0$$, vanishes on $$S$$ and equals $$1$$ at $$t$$. If $$F$$ were exactly consistent, $$F(0)=\varphi_1(t)=0$$ and $$F(0)=\varphi_2(t)=1$$, a contradiction.

For the group-valued case put $$g_1=e$$ and $$g_2(x)=\exp\big(f(n\cdot x)X\big)$$ with $$f$$ chosen as before except that $$f(n\cdot x_t)=s_0$$ with $$\exp(s_0X)\neq e$$. Since $$\exp(fX)$$ commutes with its derivative, $$g_2^{-1}\partial_\mu g_2=f'n_\mu X$$, whose divergence is $$n^\mu n_\mu f''X=0$$. The two solutions agree on $$S$$ and differ at $$t$$.

For the Poisson process, almost surely no two points coincide or are null-separated, and this property holds for all finite subsets at once. $$\square$$
</div>

Two features of the statement deserve to be spelled out. First, the dimension plays no role, and neither does the choice of $$S$$: the rule may use any finite set of sprinkled elements, however it is selected, and may use the positions as well as the order. Second, what the proof uses is generic position. The rectangle rule of Theorem 1 evades it for a concrete reason. In $$d=2$$ the only null directions are the two lattice directions. In a rectangle the corner $$r$$ has the same $$u$$ as $$t$$ and the corner $$l$$ has the same $$v$$, so the null hyperplane through $$t$$ always contains a point of $$S=\{b,l,r\}$$ for each of the two null directions, the general solution $$f(u)+g(v)$$ is determined by the three corners, and the proof breaks down. For sprinkled elements, null separation has probability zero.

The theorem says nothing about rules whose error tends to zero as the density grows. A rule with a defect that vanishes in the continuum limit, or one that uses a growing number of elements, is not covered. A scaling identity shows what such a rule would have to do.

<div class="proposition" markdown="1">
**Proposition 4** (Scaling of the defect of an order-only rule). Let a completion rule be defined for each marked finite order $$(C,\prec,t)$$ by a set $$S$$ and an affine-equivariant function $$F$$, meaning $$F(c\,\varphi_S+c_0)=c\,F(\varphi_S)+c_0$$, and let both depend on the isomorphism class of $$(C,\prec,t)$$ only. For the linear field $$\varphi(x)=a\cdot x$$, with a fixed vector $$a$$, on a Poisson sprinkling of density $$\rho$$ in a causal interval of proper height $$T$$, let $$D_\rho(T)$$ be the mean of $$\lvert F(\varphi\vert_S)-\varphi(t)\rvert$$ over targets. Then

$$
D_\rho(T)=\rho^{-1/d}\,D_1\big(\rho^{1/d}T\big).\tag{7}
$$

</div>

<div class="proof" markdown="1">
*Proof.* The dilation $$x\mapsto\rho^{1/d}x$$ maps a sprinkling of density $$\rho$$ in an interval of height $$T$$ onto a sprinkling of density $$1$$ in an interval of height $$\rho^{1/d}T$$, preserves the order, hence the selection of $$S$$ and the function $$F$$, and turns the data $$a\cdot x$$ into $$(\rho^{-1/d}a)\cdot x'$$. By equivariance the defect is multiplied by $$\rho^{-1/d}$$. $$\square$$
</div>

Equation (7) says that the defect of an order-only rule at fixed $$T$$ tends to zero as $$\rho\to\infty$$ if and only if $$D_1(L)=o(L)$$: the defect per step in units of the discreteness scale must grow more slowly than the size of the region. If a measured slope is $$\sigma=d\ln D/d\ln\rho$$, then $$D_1(L)\propto L^{d\sigma+1}$$.

*Numerical observation.* We applied the join-heap rule, the natural transcription of the rectangle rule to a two-dimensional sprinkling. For a target $$t$$, take two maximal elements $$l,r$$ of its past (its past links), so that $$t$$ is minimal in $$J^+(l)\cap J^+(r)$$; take $$b$$ maximal in $$J^-(l)\cap J^-(r)$$; predict $$\varphi(t)=\varphi(l)+\varphi(r)-\varphi(b)$$. The data are $$\varphi=u+v$$ in null coordinates on the unit square, a solution of the wave equation. For $$N=250,500,1000,2000,4000,8000$$ points (three repetitions, 150 targets with $$u,v>0.5$$ each, 30 random admissible triples per target) the mean absolute error is $$0.174,0.169,0.156,0.151,0.142,0.134$$, and the spread of the prediction over the admissible triples for one target is $$0.168,0.179,0.173,0.175,0.173,0.171$$. A local rule would give an error falling like $$N^{-1/2}$$, that is by a factor $$5.7$$ over this range, and the observed fall is a factor $$1.3$$: $$\sigma\approx-0.04$$ and $$D_1(L)\propto L^{0.9}$$. The selected elements are not confined to a neighbourhood of $$t$$. The larger null extent of the box from $$b$$ to $$t$$ has mean $$0.45,0.39,0.35$$ for $$N=500,2000,8000$$ (the unit square has side $$1$$), and its mean area is $$0.077,0.050,0.039$$, while the mean spacing $$N^{-1/2}$$ is $$0.045,0.022,0.011$$. The boxes are small in area, because $$t$$ is minimal in the common future of $$l$$ and $$r$$, but they are long in one null direction, so that the elements involved are far from $$t$$ on the scale of the region. This is a computation for one family of rules in one dimension, with one way of selecting $$l$$ and $$r$$; it is not a theorem.

## 4. Laws that revise the order from per-element statistics

*The picture.* Suppose the relations of the order are to be re-decided by a law that looks, for each pair $$x,y$$, at numbers attached to $$x$$ and to $$y$$ separately, such as the sizes of the past and future cones, the height, or the number of chains through the element, and not at how $$x$$ and $$y$$ relate to each other. A manifold-like order should be a fixed point. We show that this cannot be the case when the order has a symmetry that moves a point to a point with a different future.

Let $$(X,<)$$ be a finite strict partial order. A *statistic* is a function $$s$$ on $$X$$ with values in a set $$\Sigma$$ that is invariant under the automorphisms of $$(X,<)$$: $$s(gx)=s(x)$$ for every automorphism $$g$$. A *separate-statistics law* is given by $$s$$ and a relation $$\Psi\subseteq\Sigma\times\Sigma$$, and returns the relation $$x\,R'\,y\iff\Psi(s(x),s(y))$$. The law *fixes* $$(X,<)$$ if $$R'$$ equals $$<$$.

<div class="proposition" markdown="1">
**Proposition 5** (A fixed point has orbit-invariant futures). If a separate-statistics law fixes $$(X,<)$$, then for every automorphism $$g$$ and all $$x,y\in X$$, $$x<y$$ implies $$gx<y$$. Consequently an order that has an automorphism $$g$$, and elements $$x<y$$ with $$gx\not<y$$, is not fixed by any separate-statistics law.
</div>

<div class="proof" markdown="1">
*Proof.* If $$x<y$$ then $$\Psi(s(x),s(y))$$ holds. Since $$s(gx)=s(x)$$, also $$\Psi(s(gx),s(y))$$, that is $$gx\,R'\,y$$, and $$R'=<$$ gives $$gx<y$$. $$\square$$
</div>

The proposition is elementary, but its hypotheses are met by the configurations one wants to be fixed. The Minkowski diamond (in the continuum, with Lebesgue measure and cone volumes as the statistics) is mapped to itself by a spatial reflection through a hyperplane containing the time axis. Take $$x=(t,\vec x)$$ with $$\vec x\neq g\vec x$$ and let $$g$$ be the reflection, so that $$gx=(t,g\vec x)$$ is spacelike to $$x$$. Let $$y=x+\epsilon(1,\hat e)$$ with $$\hat e$$ the unit vector along $$\vec x-g\vec x$$. Then $$y$$ is in the future of $$x$$, while $$y-gx=(\epsilon,\vec x-g\vec x+\epsilon\hat e)$$ is spacelike, so $$x<y$$ and $$gx\not<y$$. The same holds in every dimension $$d\ge2$$, and the proof of Proposition 5 applies verbatim to a measured order whose statistics are cone volumes and whose automorphisms preserve the measure. In the two-dimensional null lattice $$[N]^2$$ the exchange of the two coordinates is such an automorphism. For a finite sprinkling the automorphism group is almost surely trivial and the proposition does not apply to it; the continuum version is the relevant one, and the following computation shows what happens at finite size. A randomised law whose output distribution depends only on the pair of statistics has the same limitation in expectation: $$\Pr(gx\,R'\,y)=\Pr(x\,R'\,y)$$.

*An example: cone-count dominance.* For a finite strict order $$R$$ on $$n$$ elements let $$d_R(i)$$ and $$u_R(i)$$ be the numbers of elements below and above $$i$$. The cone-count dominance map is

$$
\Phi(R)=\{(i,j):\ d_R(i)<d_R(j)\ \text{and}\ u_R(i)>u_R(j)\}.\tag{8}
$$

Pairs are related when the first element has fewer predecessors and more successors than the second. It is a separate-statistics law with $$s=(d_R,u_R)$$.

<div class="proposition" markdown="1">
**Proposition 6** (Properties of the dominance map). (a) For every strict order $$R$$ on $$n$$ elements, $$R\subseteq\Phi(R)$$, $$\Phi(R)$$ is a strict order of order dimension at most two, the sequence $$R,\Phi(R),\Phi^2(R),\dots$$ is increasing and constant after at most $$n(n-1)/2$$ steps, and $$R=\Phi(R)$$ if and only if $$(d_R(i)-d_R(j))(u_R(i)-u_R(j))\ge0$$ for every pair $$i,j$$. In particular no order of dimension $$\ge3$$ is a fixed point.

(b) Let $$\mu$$ be a probability measure on the open unit square in null coordinates $$(u,v)$$ with a continuous positive density near an interior point $$x_0$$, order $$(u,v)<(u',v')$$ componentwise, and let $$p(x)$$ and $$f(x)$$ be the $$\mu$$-masses of the sets below and above $$x$$. Put $$A=\int_0^{v_0}\rho(u_0,v')dv'$$, $$B=\int_0^{u_0}\rho(u',v_0)du'$$, $$C=\int_{v_0}^1\rho(u_0,v')dv'$$ and $$D=\int_{u_0}^1\rho(u',v_0)du'$$, where $$\rho$$ is the density. For every $$a,b>0$$ with $$a/b>\max(B/A,\,D/C)$$ there is $$\delta_0>0$$ such that for $$0<\delta<\delta_0$$ the point $$y=x_0+\delta(a,-b)$$ is spacelike to $$x_0$$, and $$p(x_0)<p(y)$$ and $$f(x_0)>f(y)$$: the point $$x_0$$ is related to $$y$$ by the continuum analogue of $$\Phi$$ although they are incomparable. The same holds for $$a/b<\min(B/A,D/C)$$ with the roles exchanged.
</div>

<div class="proof" markdown="1">
*Proof.* (a) If $$i<j$$ the down-set of $$j$$ contains the down-set of $$i$$ and $$i$$ itself, so $$d(i)<d(j)$$; the same argument for up-sets gives $$u(i)>u(j)$$. Hence $$R\subseteq\Phi(R)$$. The relation $$\Phi(R)$$ is the componentwise strict order on the points $$(d(i),-u(i))\in\mathbb Z^2$$, which is irreflexive and transitive, and it is the intersection of two linear orders: order the elements lexicographically by $$(d,u)$$ ascending and by $$(-u,-d)$$ ascending, and break exact ties by the index in one order and by the reverse index in the other. So it has dimension at most two. Applying $$R\subseteq\Phi(R)$$ to each term shows the sequence is increasing; a strict order has at most $$n(n-1)/2$$ relations. A fixed point satisfies $$\Phi(R)\subseteq R$$, and a pair outside $$R$$ in either orientation is promoted exactly when $$(d(i)-d(j))(u(i)-u(j))<0$$. Fixed points are of dimension at most two because they equal $$\Phi(R)$$.

(b) The function $$p(u,v)=\int_0^u\int_0^v\rho$$ is $$C^1$$ near $$x_0$$ with $$\partial_up=A$$, $$\partial_vp=B$$ at $$x_0$$, and likewise $$\partial_uf=-C$$ and $$\partial_vf=-D$$. For $$y=x_0+\delta(a,-b)$$ the first-order changes are $$\Delta p=\delta(aA-bB)+o(\delta)$$ and $$\Delta f=\delta(-aC+bD)+o(\delta)$$, which have the stated signs when $$a/b>B/A$$ and $$a/b>D/C$$. The point $$y$$ is incomparable to $$x_0$$ because the $$u$$-coordinate increases and the $$v$$-coordinate decreases. $$\square$$
</div>

Part (a) leaves open whether the map moves a given order at all, since an order of dimension two with the right statistics could be a fixed point. Part (b) shows that on a smooth density of points it does move the order, and that the set of promoted directions is open. For the uniform measure on the unit square, $$p=uv$$ and $$f=(1-u)(1-v)$$, and a Monte Carlo evaluation with $$8\times10^7$$ pairs gives the fraction of incomparable pairs that are promoted as $$0.6666\pm0.0001$$.

*Numerical observation.* On sprinklings of a two-dimensional interval the fraction of spacelike pairs that one application of $$\Phi$$ relates is $$0.655,0.664,0.668,0.663$$ for $$n=250,500,1000,2000$$ points, and on sprinklings of a four-dimensional interval it is $$0.547,0.580,0.614,0.620$$ for $$n=250,500,1000,1500$$ (averages of three samples each; no relation of the original order was ever lost, as part (a) requires). Iterating $$\Phi$$ on a single sprinkling of $$n=600$$ points ended after $$32$$ steps in two dimensions, with the fraction of related pairs rising from $$0.518$$ to $$0.988$$ and the height from $$42$$ to $$257$$; in four dimensions it ended after $$9$$ steps, with the fraction of related pairs rising from $$0.098$$ to $$0.874$$ and the height from $$9$$ to $$163$$. The fixed points are near-chains.

The statement is about laws that read separate statistics. A law that reads data of the pair itself, such as the number of common predecessors $$\lvert\downarrow x\cap\downarrow y\rvert$$, the number of elements in the interval, or the links between them, is not covered, because a symmetry that preserves the separate statistics of $$x$$ and $$y$$ need not preserve those.

## 5. Sequential growth with one amplitude per history

Classical sequential growth builds a causal set one element at a time, with probabilities for the new element's past [[RS]](#ref-rs). Complex sequential growth replaces the probabilities by complex amplitudes [[SZ]](#ref-sz); complex transitive percolation [[DJS]](#ref-djs) is the case with a single parameter $$p$$ and $$q=1-p$$, in which a labelled order with $$L$$ links and $$R$$ relations has amplitude $$p^Lq^{\binom n2-R}$$. The amplitude $$A(E)$$ of an event is the sum of the amplitudes of the labelled orders in it, and the decoherence functional is the product $$D(\alpha,\beta)=\overline{A(\alpha)}A(\beta)$$ with $$A(\Omega)=1$$. The two cited papers discuss the extension of this functional to covariant events, which is not needed here.

<div class="proposition" markdown="1">
**Proposition 7** (Rank one means no decoherence). Let $$D(\alpha,\beta)=\overline{A(\alpha)}A(\beta)$$ with $$A(\Omega)=1$$, and let $$E$$ be an event with complement $$E^c$$. If $$A(E)A(E^c)\neq0$$ the normalised interference $$\lvert D(E,E^c)\rvert/\sqrt{D(E,E)D(E^c,E^c)}$$ equals $$1$$. It vanishes if and only if $$A(E)\in\{0,1\}$$, that is if one of $$E,E^c$$ has zero quantum measure.
</div>

<div class="proof" markdown="1">
*Proof.* $$\lvert D(E,E^c)\rvert=\lvert A(E)\rvert\lvert A(E^c)\rvert=\sqrt{D(E,E)D(E^c,E^c)}$$; this is the Cauchy–Schwarz inequality with equality in the one-dimensional space $$\mathbb C$$. Since $$A(E^c)=1-A(E)$$, the product $$A(E)A(E^c)$$ vanishes exactly when $$A(E)=0$$ or $$A(E)=1$$. $$\square$$
</div>

A binary coarse-graining of a rank-one decoherence functional is therefore never decoherent unless one of the two sides is precluded. This does not depend on the event, on the number of elements or on the value of $$p$$, and it applies to real $$p$$ (where the model is classical percolation seen as an amplitude). Decoherence between events needs a history space of dimension at least two. If the transition operators on $$\mathbb C^m$$ are commuting and normal, they can be diagonalised simultaneously and the vector-valued measure splits into $$m$$ orthogonal channels $$A_k$$, each satisfying the sum rule, with $$D(\alpha,\beta)=\sum_k\lvert c_k\rvert^2\overline{A_k(\alpha)}A_k(\beta)$$; this is the first case in which the question can be asked non-trivially, and it needs $$m\ge2$$ channels. The existence of non-commutative quantum sequential growth satisfying the natural causality conditions is a separate question that this page does not treat.

## 6. Path sums over layered orders

*The question.* A sum over orders, $$Z=\sum_{\mathcal O}e^{iS(\mathcal O)/\hbar}$$, can only be dominated by manifold-like orders if the phases suppress the rest. The difficulty is entropy. By the theorem of Kleitman and Rothschild [[KR75]](#ref-kr75), almost all partial orders on $$n$$ elements have three layers, of sizes close to $$n/4$$, $$n/2$$ and $$n/4$$, and their number is $$2^{n^2/4+o(n^2)}$$. Almost all of these orders are very far from a manifold. The question for a given action is whether its phases cancel enough in the sum over the three-layer class that the class does not dominate.

For an action built from interval counts the sum over the three-layer class can be done exactly. We define the action, the class, and the transfer matrix, then state the results, and compare with exact numbers at small sizes.

*The action.* For a finite order, let $$N_k$$ be the number of pairs $$x\prec y$$ with exactly $$k$$ elements strictly between them, so $$N_0$$ is the number of links, and let $$n$$ be the number of elements. A finite-depth action is

$$
\frac{S}{\hbar}=\beta\Big(b\,n-\sum_{k=0}^{L}C_kN_k\Big),\tag{9}
$$

with real parameters $$\beta,b,C_0,\dots,C_L$$. The four-dimensional Benincasa–Dowker action [[BD]](#ref-bd) has $$L=3$$, $$b=1$$ and $$C=(1,-9,16,-8)$$, with $$\beta=(4/\sqrt6)(\ell/\ell_p)^2$$ in terms of the discreteness length $$\ell$$ and the Planck length $$\ell_p$$; the two-dimensional one has $$L=2$$, $$b=1$$, $$C=(2,-4,2)$$ and $$\beta=2$$. The *link action* has only $$C_0\neq0$$. The link phase is $$z=e^{-i\beta C_0}$$, and $$f(k)=e^{-i\beta C_k}$$, with $$f(k)=1$$ for $$k>L$$ and $$f(0)=z$$. The coefficient $$\beta$$ is a free parameter in all that follows.

*The class.* Fix layer sizes $$(n_1,n_2,n_3)$$ and call the layers $$L_1,L_2,L_3$$. An order of the class $$\mathcal K(n_1,n_2,n_3)$$ is given by two bipartite relations $$P_{12}\subseteq L_1\times L_2$$ and $$P_{23}\subseteq L_2\times L_3$$ with $$a\prec v$$ for $$(a,v)\in P_{12}$$, $$v\prec c$$ for $$(v,c)\in P_{23}$$, and $$a\prec c$$ for all $$a\in L_1$$, $$c\in L_3$$. It has $$\lvert\mathcal K\rvert=2^{n_2(n_1+n_3)}$$ members, which is $$2^{n^2/4}$$ at proportions $$(n/4,n/2,n/4)$$, the leading exponent of the number of all orders. For $$a\in L_1$$ and $$c\in L_3$$ the number of elements strictly between them is $$k_{ac}=\lvert\{v:(a,v)\in P_{12},(v,c)\in P_{23}\}\rvert$$. The links are the elements of $$P_{12}$$ and $$P_{23}$$ together with the pairs $$(a,c)$$ with $$k_{ac}=0$$. We write

$$
Z(n_1,n_2,n_3)=\sum_{\mathcal O\in\mathcal K}e^{iS(\mathcal O)/\hbar}.
$$

Two layers are the simplest case. For the bilayer class of orders on $$L_1\cup L_2$$ with $$a\prec v$$ for $$(a,v)\in G\subseteq L_1\times L_2$$, no pair has an element between, so only links occur.

<div class="theorem" markdown="1">
**Theorem 8** (Exact bilayer sum). For layers of sizes $$m_1,m_2$$ and the action (9),

$$
Z=e^{i\beta b(m_1+m_2)}\,(1+z)^{m_1m_2},\qquad\frac{\lvert Z\rvert}{2^{m_1m_2}}=\Big\lvert\cos\frac{\beta C_0}{2}\Big\rvert^{m_1m_2}.\tag{10}
$$

For every finite-depth action the bilayer sum depends on the action only through the link phase $$\beta C_0$$ (and a global phase). If $$\beta C_0\in2\pi\mathbb Z$$ the sum is not suppressed at all.
</div>

<div class="proof" markdown="1">
*Proof.* In a bilayer every interval is empty, so $$N_k=0$$ for $$k\ge1$$ and $$N_0=\lvert G\rvert$$, and $$e^{iS/\hbar}=e^{i\beta bn}z^{\lvert G\rvert}$$. Summing over all $$G\subseteq L_1\times L_2$$ gives $$\sum_Gz^{\lvert G\rvert}=(1+z)^{m_1m_2}$$, and $$\lvert1+z\rvert=2\lvert\cos(\beta C_0/2)\rvert$$. $$\square$$
</div>

*Check.* For layers $$(4,4)$$, $$\beta=2\arccos0.4$$ and the Benincasa–Dowker coefficients (9), a brute-force sum over all $$2^{16}$$ relation patterns with the interval counts computed from the order itself gives $$Z=0.0281475$$ and $$\lvert Z\rvert/2^{16}=4.2950\times10^{-7}$$, against $$0.4^{16}=4.2950\times10^{-7}$$ from (10).

*The transfer matrix.* In the three-layer class the sum organises itself by the elements of the middle layer. Each $$v\in L_2$$ has a set $$u_v\subseteq L_1$$ of lower neighbours and $$w_v\subseteq L_3$$ of upper neighbours, chosen independently of the other middle elements, with weight $$z^{\lvert u_v\rvert+\lvert w_v\rvert}$$ for its links. The number $$k_{ac}$$ is the number of $$v$$ with $$(a,c)\in u_v\times w_v$$. The weight of the whole order is therefore

$$
e^{iS/\hbar}=e^{i\beta bn}\;z^{\lvert P_{12}\rvert+\lvert P_{23}\rvert}\prod_{a\in L_1,c\in L_3}f(k_{ac}).\tag{11}
$$

Adding the middle elements one at a time, the state is the matrix $$K=(\min(k_{ac},L+1))$$ of common-neighbour counts capped at $$L+1$$, since $$f(k)=1$$ beyond the cap. There are $$D=(L+2)^{n_1n_3}$$ states. A middle element with data $$(u,w)$$ adds $$1$$ to the entries in $$u\times w$$, with weight $$z^{\lvert u\rvert+\lvert w\rvert}$$. Call $$M$$ the matrix of these weights. Counts never decrease, so $$M$$ is triangular for the componentwise order on states, and its diagonal entries are read off from the moves that leave the state unchanged. If $$U=\{(a,c):K_{ac}\le L\}$$ is the set of uncapped entries, a move leaves $$K$$ unchanged exactly when $$(u\times w)\cap U=\varnothing$$, and summing over such moves gives

$$
\lambda_U=\sum_{u\subseteq L_1}z^{\lvert u\rvert}(1+z)^{\,n_3-\lvert N_U(u)\rvert},\qquad N_U(u)=\{c:(a,c)\in U\text{ for some }a\in u\}.\tag{12}
$$

The final weight is $$g(K)=\prod f(K_{ac})$$, and $$Z=e^{i\beta bn}\,(M^{n_2})_{0\cdot}\,g$$, where $$0$$ is the zero state. Write $$\rho_*=\max_U\lvert\lambda_U\rvert$$, the maximum over all subsets $$U\subseteq L_1\times L_3$$, and $$\kappa=\rho_*/2^{n_1+n_3}$$. The quantities $$\lambda_U$$, $$\rho_*$$ and $$\kappa$$ depend on $$z$$, $$n_1$$ and $$n_3$$ only. They do not depend on $$b$$, on $$C_1,\dots,C_L$$, or on $$L$$.

<div class="theorem" markdown="1">
**Theorem 9** (Suppression at fixed outer layers). Fix $$n_1,n_3\ge1$$ and a finite-depth action (9), and put $$\mu=\max(\lvert\cos(\beta C_0/2)\rvert,\tfrac12)$$. Then

$$
\kappa\le\mu^{\max(n_1,n_3)},\qquad\kappa\ge\Big\lvert\cos\frac{\beta C_0}{2}\Big\rvert^{n_1+n_3},\tag{13}
$$

and there is a constant $$c$$ depending on $$n_1,n_3,L$$ and $$\kappa$$ such that for all $$n_2$$

$$
\frac{\lvert Z(n_1,n_2,n_3)\rvert}{\lvert\mathcal K\rvert}\le c\,n_2^{\,D-1}\,\kappa^{\,n_2}.\tag{14}
$$

In particular, if $$\lvert\cos(\beta C_0/2)\rvert<1$$ then $$\kappa<1$$ and the sum over the class is exponentially small in $$n_2$$. The exponential factor in (14) depends on the action only through $$z=e^{-i\beta C_0}$$. The action enters the polynomial prefactor, whose degree is at most $$D-1$$.
</div>

The idea is that the triangular structure confines the dependence on the weights $$C_1,\dots,C_L$$ to the off-diagonal entries of $$M$$, which affect only a polynomial prefactor. The exponential rate is set by the diagonal entries, and those see only the links, through $$z$$. The route is routine linear algebra once the transfer matrix is set up; the two inequalities in (13) are the only place where a computation is needed.

<div class="proof" markdown="1">
*Proof.* Triangularity was shown above: moves only increase entries, so $$M_{KK'}\neq0$$ implies $$K'\ge K$$ componentwise. Consider a product of $$n_2$$ factors of $$M$$ along a path of states. The path has at most $$D-1$$ strict increases. Every other step is a diagonal entry, of modulus at most $$\rho_*$$, and each off-diagonal entry is a sum of at most $$2^{n_1+n_3}$$ unimodular terms. Counting the choice of the steps at which the strict increases occur and the choice of the intermediate states gives

$$
\big\lvert(M^{n_2})_{0K}\big\rvert\le D^{D-1}\sum_{j=0}^{D-1}\binom{n_2}{j}2^{(n_1+n_3)j}\rho_*^{\,n_2-j}.
$$

Summing over the at most $$D$$ final states, using $$\lvert g\rvert=1$$, dividing by $$\lvert\mathcal K\rvert=2^{n_2(n_1+n_3)}$$ and using $$\binom{n_2}{j}\le n_2^{\,j}$$ gives (14) (when $$\rho_*=0$$ the matrix is nilpotent and $$Z=0$$ for $$n_2\ge D$$).

For the upper bound in (13), $$\lvert1+z\rvert=2\lvert\cos(\beta C_0/2)\rvert\le2\mu$$, and $$\lvert N_U(u)\rvert\le n_3$$, so each term of (12) satisfies

$$
\frac{\lvert z^{\lvert u\rvert}(1+z)^{n_3-\lvert N_U(u)\rvert}\rvert}{2^{n_3}}=\Big\lvert\cos\frac{\beta C_0}{2}\Big\rvert^{n_3-\lvert N_U(u)\rvert}2^{-\lvert N_U(u)\rvert}\le\mu^{n_3},
$$

since both $$\lvert\cos\rvert\le\mu$$ and $$\tfrac12\le\mu$$. The sum has $$2^{n_1}$$ terms, so $$\lvert\lambda_U\rvert/2^{n_1+n_3}\le\mu^{n_3}$$. Reorganising the same sum with the roles of $$L_1$$ and $$L_3$$ exchanged gives $$\mu^{n_1}$$, hence $$\kappa\le\mu^{\max(n_1,n_3)}$$. For the lower bound, $$U=\varnothing$$ has $$N_U(u)=\varnothing$$ for all $$u$$, so $$\lambda_\varnothing=(1+z)^{n_3}\sum_uz^{\lvert u\rvert}=(1+z)^{n_3+n_1}$$. $$\square$$
</div>

*Check.* For $$\lvert\cos(\beta C_0/2)\rvert=0.4$$, so $$\mu=1/2$$, (12) gives $$\rho_*=1.510,\,2.280,\,3.986$$ for $$n_1=n_3=1,2,3$$, hence exponential rates $$-\ln\kappa=0.974,\,1.948,\,2.776$$ per middle element, inside the interval $$[\,n\ln2,\ 2n\ln(1/0.4)\,]=[0.693,1.833]$$, $$[1.386,3.665]$$, $$[2.079,5.498]$$ allowed by (13) (for $$n_1=n_3=n$$). The exact sums, computed by enumerating the middle layer by the multiset of its neighbourhood types, decrease with these local slopes $$-\Delta\ln\lvert Z\rvert/\Delta n_2$$:

| layers | action | local slopes for $$n_2=8\to24$$ (step 4) | bound rate |
|---|---|---|---|
| $$(1,n_2,1)$$ | link | $$0.974,0.974,0.974,0.974$$ | $$0.974$$ |
| $$(1,n_2,1)$$ | Benincasa–Dowker, $$d=4$$ | $$0.585,0.739,0.802,0.836$$ | $$0.974$$ |
| $$(2,n_2,2)$$ | link | $$2.10,2.07,1.77,2.03$$ | $$1.948$$ |
| $$(2,n_2,2)$$ | Benincasa–Dowker, $$d=4$$ | $$0.91,1.17,1.35,1.49$$ | $$1.948$$ |
| $$(3,n_2,3)$$ | link | $$2.95,2.91,2.90,2.83$$ (step 2, $$n_2=8\to16$$) | $$2.776$$ |
| $$(3,n_2,3)$$ | Benincasa–Dowker, $$d=4$$ | $$0.80,1.23,1.47,1.71$$ (step 2, $$n_2=8\to16$$) | $$2.776$$ |

For the link action the decay is at least as fast as the bound, as it must be. For the Benincasa–Dowker coefficients the local slopes are below the bound and rising, which is what a polynomial prefactor of large degree produces; the theorem gives the exponential rate only as an upper bound for $$\lvert Z\rvert$$, and whether it is attained for a given action depends on whether the dominant term has a non-zero coefficient. This is not decided by the theorem.

The rate in (14) is a statement for fixed $$n_1$$ and $$n_3$$. At Kleitman–Rothschild proportions all three layer sizes grow together, and the next result covers all sizes at once.

<div class="theorem" markdown="1">
**Theorem 10** (Uniform bound, and its logarithmic gap). Let $$B_L(p)=\sum_{j=0}^{L}\binom pj$$ and $$\mu$$ as above. For all $$n_1,n_2,n_3$$ and every finite-depth action,

$$
\frac{\lvert Z\rvert}{\lvert\mathcal K\rvert}\le\mu^{\,n_2n_3}\Big(\mathbb E_{p\sim\mathrm{Bin}(n_2,1/2)}\big[(1+2B_L(p))^{n_3}\big]\Big)^{n_1}.\tag{15}
$$

(a) For the link action ($$L=0$$, $$B_0=1$$) the right-hand side is $$\big(\mu^{n_2}3^{n_1}\big)^{n_3}$$. At proportions $$(n/4,n/2,n/4)$$ it is $$\exp\big[\tfrac{n^2}{8}\ln(\sqrt3\,\mu)\big]$$, exponentially small in $$n^2$$ whenever $$\lvert\cos(\beta C_0/2)\rvert<1/\sqrt3$$.

(b) For $$L\ge1$$, the bound gives $$e^{-cn^2}$$ whenever $$n_1L\ln(n_2+1)\le(1-\delta)\,n_2\ln(1/\mu)$$ for some $$\delta>0$$, for instance for $$n_1=o(n/\ln n)$$ with $$n_2,n_3$$ proportional to $$n$$. At proportions $$(n/4,n/2,n/4)$$ it exceeds $$1$$ for large $$n$$.
</div>

The difficulty is that the weights of the interval counts $$k=1,\dots,L$$ are unimodular but not summable; the bound must control them without using cancellation. The way around it is to expand the product over $$a\in L_1$$ in (11) as a sum over subsets, so that each term depends on the neighbourhood of the middle layer only through a bounded number of constraints.

<div class="proof" markdown="1">
*Proof.* Fix $$P_{12}$$ and write $$P_a\subseteq L_2$$ for the upper neighbourhood of $$a\in L_1$$, $$p_a=\lvert P_a\rvert$$. In (11) the columns $$c\in L_3$$ are independent given $$P_{12}$$, so $$Z=e^{i\beta bn}\sum_{P_{12}}z^{\lvert P_{12}\rvert}H(P_{12})^{n_3}$$ with

$$
H(P_{12})=\sum_{w\subseteq L_2}z^{\lvert w\rvert}\prod_{a\in L_1}f(\lvert P_a\cap w\rvert).
$$

Write $$f=1+g$$, where $$g(k)=0$$ for $$k>L$$ and $$\lvert g\rvert\le2$$, and expand $$\prod_a(1+g)=\sum_{T\subseteq L_1}\prod_{a\in T}g(\lvert P_a\cap w\rvert)$$. For fixed $$T$$ the product depends on $$w$$ only through $$w\cap P_T$$, with $$P_T=\bigcup_{a\in T}P_a$$. Summing over the free part of $$w$$ outside $$P_T$$ gives $$(1+z)^{n_2-\lvert P_T\rvert}$$. The sum over $$w'=w\cap P_T$$ has non-zero terms only if $$\lvert P_a\cap w'\rvert\le L$$ for all $$a\in T$$. The map $$w'\mapsto(w'\cap P_a)_{a\in T}$$ is injective, so at most $$\prod_{a\in T}B_L(p_a)$$ terms survive, each of modulus at most $$2^{\lvert T\rvert}$$. With $$\lvert1+z\rvert^{n_2-\lvert P_T\rvert}\le(2\mu)^{n_2}$$, because $$2\mu\ge1$$, we get

$$
\lvert H(P_{12})\rvert\le(2\mu)^{n_2}\sum_T\prod_{a\in T}2B_L(p_a)=(2\mu)^{n_2}\prod_{a\in L_1}\big(1+2B_L(p_a)\big).
$$

Dividing by $$2^{n_2}$$, raising to the power $$n_3$$ and averaging over uniform $$P_{12}$$, for which the row sums $$p_a$$ are independent and binomial with parameters $$n_2$$ and $$1/2$$, gives (15). For (a), $$B_0=1$$. For (b) use $$B_L(p)\le(n_2+1)^L$$, so the right-hand side is at most $$\exp[\,n_3(n_2\ln\mu+n_1\ln(1+2(n_2+1)^L))\,]$$, which is smaller than $$\exp(-\delta n_2n_3\ln(1/\mu))$$ under the stated condition. At proportions $$(n/4,n/2,n/4)$$ the condition reads $$\tfrac12L\ln(n/2+1)\le(1-\delta)\ln(1/\mu)$$, which fails for large $$n$$ because the left side grows. $$\square$$
</div>

*Check.* Evaluating (15) and the exact sum for $$\lvert\cos\rvert=0.4$$ and $$0.8$$ on layers $$(1,10,4)$$, $$(2,12,4)$$, $$(2,16,6)$$, $$(3,12,3)$$ and $$(4,8,4)$$, the bound holds in every case. For the link action it is non-trivial, for example $$7.4\times10^{-11}$$ against the exact $$5.0\times10^{-14}$$ on $$(1,10,4)$$ and $$1.0\times10^{-2}$$ against $$1.9\times10^{-8}$$ on $$(4,8,4)$$ at $$\lvert\cos\rvert=0.4$$. For the Benincasa–Dowker coefficients it is larger than $$1$$ in all the cases tried except $$(1,10,4)$$ at $$\lvert\cos\rvert=0.4$$, where it is $$1.5\times10^{-4}$$ against $$2.4\times10^{-9}$$; this is the logarithmic gap at accessible sizes.

*What the exact sums show at Kleitman–Rothschild proportions.* The exact value of $$\lvert Z\rvert/\lvert\mathcal K\rvert$$ at $$(m,2m,m)$$ with $$\lvert\cos(\beta C_0/2)\rvert=0.4$$ (so $$\beta C_0=2\arccos0.4$$) is:

| $$n=4m$$ | link action | Benincasa–Dowker, $$d=4$$ | naive $$0.4^{n^2/4}$$ |
|---|---|---|---|
| 4 | $$2.59\times10^{-1}$$ | $$1.90\times10^{-1}$$ | $$2.6\times10^{-2}$$ |
| 8 | $$9.54\times10^{-3}$$ | $$5.51\times10^{-4}$$ | $$4.3\times10^{-7}$$ |
| 12 | $$4.16\times10^{-5}$$ | $$6.48\times10^{-6}$$ | $$4.7\times10^{-15}$$ |
| 16 | $$1.86\times10^{-8}$$ | $$2.94\times10^{-8}$$ | $$3.4\times10^{-26}$$ |

All entries are suppressed, but much more weakly than the naive law $$\lvert\cos\rvert^{n^2/4}$$ obtained by applying (10) to the pairs $$(a,v)$$ alone. The data stop at $$n=16$$ and do not decide the asymptotic form.

When the link phase is trivial there is no suppression at all, whatever the other weights.

<div class="proposition" markdown="1">
**Proposition 11** (A trivial link phase leaves the layered orders unsuppressed). If $$\beta C_0\in2\pi\mathbb Z$$, then for all $$n_1,n_2,n_3$$

$$
\frac{\lvert Z\rvert}{\lvert\mathcal K\rvert}\ge1-2\,n_1n_3\,\Pr\big[\mathrm{Bin}(n_2,\tfrac14)\le L\big].
$$

At proportions $$(n/4,n/2,n/4)$$ the right-hand side tends to $$1$$.
</div>

<div class="proof" markdown="1">
*Proof.* With $$z=1$$, (11) reduces to $$e^{i\beta bn}\prod_{a,c}f(k_{ac})$$, and the product equals $$1$$ unless some $$k_{ac}\le L$$. Under the uniform measure on $$\mathcal K$$ each $$k_{ac}$$ is binomial with parameters $$n_2$$ and $$1/4$$. Then $$\lvert\mathbb E\prod f-1\rvert\le2\Pr[\exists(a,c):k_{ac}\le L]\le2n_1n_3\Pr[\mathrm{Bin}(n_2,\tfrac14)\le L]$$, and a Chernoff bound makes the probability $$e^{-cn_2}$$. $$\square$$
</div>

*Check.* With $$(C_0,C_1,C_2,C_3)=(0,-9,16,-8)$$, $$b=1$$, $$\beta=1$$ and layers $$(2,n_2,2)$$, the exact value of $$\lvert Z\rvert/\lvert\mathcal K\rvert$$ is $$0.097,0.111,0.881,0.9999$$ for $$n_2=8,16,32,64$$, and the bound of the proposition is $$0.80$$ at $$n_2=32$$ and $$0.9999$$ at $$n_2=64$$. The kernel $$(0,-9,16,-8)$$ is used only to exhibit the proposition; no continuum property is claimed for it.

Together, Theorems 8 to 10 and Proposition 11 say the following about a finite-depth action in this class. The link phase $$\beta C_0$$ decides whether the layered orders are suppressed at fixed outer layers and sets the exponential rate up to the bound (13); the remaining weights $$C_1,\dots,C_L$$ enter through a polynomial prefactor only. This is consistent with the observation in [[LC]](#ref-lc) (bilayers) and [[CCS1]](#ref-ccs1) [[CCS2]](#ref-ccs2) that for typical layered orders the leading suppression is governed by the link term, and at fixed outer layers it makes that statement exact for the exponential rate. It does not establish the reduction to the link action at Kleitman–Rothschild proportions, and the polynomial prefactor, which the theorem leaves free, is visible in the exact data above. Mathur, Singh and Surya [[MSS]](#ref-mss) study the link action in the causal set path sum; no comparison with their statements is made here.

*The variance of the unsmeared action.* A different obstacle to a path sum with a single scale is the size of the fluctuations of the action on a manifold-like sprinkling. Write $$F=bN-\sum_kC_kN_k$$ for the raw functional in (9), so that in $$d=4$$ the action is a constant times $$\rho^{-1/2}F$$. For a sprinkling of density $$\rho$$ and a nonlocality density $$\rho_k=\epsilon\rho$$ with $$0<\epsilon\le1$$, the smeared action is defined, following [[ASS]](#ref-ass), by replacing the weights $$C_k$$ with their binomial averages $$C^{(\epsilon)}(n)=\sum_{k}C_k\binom nk\epsilon^k(1-\epsilon)^{n-k}$$ over the number $$n$$ of elements in the interval:

$$
S_\epsilon=\epsilon\,b\,N-\epsilon^2\sum_{x\prec y}C^{(\epsilon)}(n_{xy}),\tag{16}
$$

up to the prefactor $$c_0\rho_k^{-1/2}$$.

<div class="proposition" markdown="1">
**Proposition 12** (Smearing is thinning, and damps the variance). Let $$\eta$$ be a Poisson process of density $$\rho$$ on a bounded region, $$\eta'$$ an independent $$\epsilon$$-thinning of $$\eta$$ (each point is kept independently with probability $$\epsilon$$), so that $$\eta'$$ is Poisson of density $$\epsilon\rho$$, and let $$F$$ be the raw functional above. Then (i) $$S_\epsilon(\eta)=\mathbb E[F(\eta')\mid\eta]$$, and (ii) $$\mathrm{Var}\,S_\epsilon\le\epsilon\,\mathrm{Var}\,F(\eta')$$.
</div>

The proof of (i) is direct and (ii) uses the chaos expansion of Last and Penrose [[LP]](#ref-lp), $$\mathrm{Var}\,G=\sum_{k\ge1}\frac1{k!}\lVert\mathbb E\,D^kG\rVert^2$$, where $$D_zG(\eta)=G(\eta+\delta_z)-G(\eta)$$ is the add-one-point difference and the norm is that of $$L^2$$ of the $$k$$-fold product of the intensity measure; it holds for every square-integrable functional.

<div class="proof" markdown="1">
*Proof.* (i) A pair of $$\eta$$ survives the thinning with probability $$\epsilon^2$$, and given that, the number of surviving points of its interval is binomial with parameters $$n_{xy}$$ and $$\epsilon$$, so the expected contribution to $$-\sum C$$ of the pair is $$-\epsilon^2C^{(\epsilon)}(n_{xy})$$; the number of points contributes $$\epsilon bN$$.

(ii) A point $$z$$ added to $$\eta$$ survives the thinning with probability $$\epsilon$$, so $$D_zG=\epsilon\,\mathbb E[D_zF(\eta')\mid\eta]$$ for $$G=\mathbb E[F(\eta')\mid\eta]$$. Iterating, with independent survival indicators for the added points, $$\mathbb E\,D^kG(z_1,\dots,z_k)=\epsilon^k\,T_kF(z_1,\dots,z_k)$$, where $$T_kF=\mathbb E\,D^kF(\eta')$$ is the corresponding quantity for the process $$\eta'$$ of intensity measure $$\epsilon\rho\,dx$$. Then $$\lVert\epsilon^kT_kF\rVert^2_{L^2((\rho dx)^k)}=\epsilon^k\lVert T_kF\rVert^2_{L^2((\epsilon\rho\,dx)^k)}$$, and summing over $$k\ge1$$ with $$\epsilon^k\le\epsilon$$ gives $$\mathrm{Var}\,G\le\epsilon\,\mathrm{Var}\,F(\eta')$$. Square integrability holds because $$\lvert F\rvert$$ is bounded by a polynomial in the number of points of the bounded region. $$\square$$
</div>

So at fixed nonlocality density $$\rho_k$$ the variance of the smeared action is at most $$\rho_k/\rho$$ times the variance of the unsmeared action at density $$\rho_k$$, for every finite-depth kernel.

*Numerical observation.* For the four-dimensional Benincasa–Dowker functional $$F=N-N_0+9N_1-16N_2+8N_3$$ on a causal interval of unit height with mean point number $$\bar N=250,500,1000,2000$$ (600, 300, 150, 60 samples), the variance of the action in units of $$T^4$$, $$\mathrm{Var}\,S/T^4=(4/\sqrt6)^2(V/\bar N)\,\mathrm{Var}\,F$$ with $$V=\pi/24$$, is $$227,\,426,\,829,\,1420$$. The statistical error of each variance estimate is about $$6\%,8\%,12\%,18\%$$ from the smallest to the largest $$\bar N$$. A fit of $$\ln\mathrm{Var}\,S$$ against $$\ln\bar N$$ has slope $$0.89$$. We do not know the asymptotic exponent: a pair-diagonal contribution suggests $$1/2$$ at large density, and the measured exponent is that of the accessible range only. For the thinning identity we checked (i) on one realisation of $$300$$ points with $$\epsilon=0.3$$: formula $$44.05$$, average over $$3000$$ independent thinnings $$43.11\pm2.18$$. For the inequality (ii), at $$\bar N=1000$$ and $$\epsilon=1/4$$ we find $$\mathrm{Var}\,S_\epsilon=598$$ against $$\epsilon\,\mathrm{Var}\,F(\eta')=3.5\times10^4$$ (80 and 400 samples), and at $$\bar N=2000$$ and $$\epsilon=1/8$$ we find $$144$$ against $$2.0\times10^4$$.

## 7. What the results do not exclude

* Theorem 3 excludes exact consistency on finitely many sprinkled elements. Rules whose error tends to zero as $$\rho\to\infty$$ are not covered, among them the interval-count d'Alembertian [[BD]](#ref-bd) with a nonlocality scale that is a free input (see [The causal-set d'Alembertian has negative spectral weight](/quantum-gravity/causal-box-spectrum/) and [Exact causal-set horizon weights must be non-local](/quantum-gravity/nonlocal-weights/) for related constraints on such operators). Proposition 4 states what such a rule must achieve on linear data: $$D_1(L)=o(L)$$.
* Theorem 1 holds in $$1+1$$ dimensions for fields defined on all pairs of ranks. It does not apply to a theory whose only objects are the sprinkled elements, since no rectangle lies in the sprinkling.
* Proposition 5 covers separate statistics. Laws that use the number of common predecessors, the contents of intervals, or the links between $$x$$ and $$y$$ are not covered, and neither are laws that are not asked to return the order they are given.
* Proposition 7 concerns a one-dimensional history space. Commuting normal transition operators on $$\mathbb C^m$$ with $$m\ge2$$, and non-commutative models, are not excluded.
* Theorems 9 and 10 concern finite-depth interval actions on three-layer classes with the complete relation $$L_1\prec L_3$$. Infinite-depth kernels, actions with an independent nonlocality scale, and sums over layer sizes with unrestricted layer assignment are not covered.
* Theorem 10 does not give $$e^{-cn^2}$$ for $$L\ge1$$ at Kleitman–Rothschild proportions when $$\beta C_0\not\equiv0$$ mod $$2\pi$$, and gives no lower bound. Whether the suppression there is of order $$n^2$$, of lower order, or absent is not settled by the bound or by the exact data up to $$n=16$$.

## 8. Open points

The uniform bound of Theorem 10 misses at Kleitman–Rothschild proportions by a factor $$(n^L)^{n_1n_3}$$ in the exponent. This comes from counting the low-overlap subsets $$w$$ with $$\lvert P_a\cap w\rvert\le L$$, and the count is far from sharp: in the exact search over capped states at $$n_1=n_3=m$$ the number of states that matter grows only linearly in $$m$$. Whether a refinement that restricts to typical $$P_{12}$$ gives $$e^{-cn^2}$$ for $$\lvert\cos(\beta C_0/2)\rvert$$ in some range, and whether the rate at proportions $$(n/4,n/2,n/4)$$ is the link rate, is open. Whether the upper bound (14) is attained for a given action, that is whether the dominant coefficient is non-zero, is known here only from the exact sums at small sizes. The asymptotic growth exponent of $$\mathrm{Var}\,S$$ for the finite-depth family, the existence of a non-commutative sequential-growth model with a decoherence functional of rank greater than one that satisfies the causality conditions, and the existence of an order-only completion rule with $$D_1(L)=o(L)$$ on linear data are also open.

## References

1. <a id="ref-sorkin"></a>Sorkin, “Does locality fail at intermediate length-scales?,” [arXiv:gr-qc/0703099](https://arxiv.org/abs/gr-qc/0703099).
2. <a id="ref-bd"></a>Benincasa, Dowker, “The scalar curvature of a causal set,” Phys. Rev. Lett. 104, 181301 (2010), [arXiv:1001.2725](https://arxiv.org/abs/1001.2725).
3. <a id="ref-ass"></a>Aslanbeigi, Saravani, Sorkin, “Generalized causal set d'Alembertians,” JHEP 06 (2014) 024, [arXiv:1403.1622](https://arxiv.org/abs/1403.1622).
4. <a id="ref-johnston"></a>Johnston, “Feynman propagator for a free scalar field on a causal set,” Phys. Rev. Lett. 103, 180401 (2009), [arXiv:0909.0944](https://arxiv.org/abs/0909.0944).
5. <a id="ref-zab"></a>Zapata-Carratalá, Arsiwalla, Beynon, “Heaps of Fish: arrays, generalized associativity and heapoids,” [arXiv:2205.05456](https://arxiv.org/abs/2205.05456).
6. <a id="ref-krud"></a>Korzyuk, Rudzko, “Curvilinear parallelogram identity and mean-value property for a semilinear hyperbolic equation of second-order,” Eurasian Math. J. 15 (2024) 61, [arXiv:2204.09408](https://arxiv.org/abs/2204.09408).
7. <a id="ref-kr75"></a>Kleitman, Rothschild, “Asymptotic enumeration of partial orders on a finite set,” Trans. Amer. Math. Soc. 205 (1975) 205.
8. <a id="ref-lc"></a>Loomis, Carlip, “Suppression of non-manifold-like sets in the causal set path integral,” [arXiv:1709.00064](https://arxiv.org/abs/1709.00064).
9. <a id="ref-ccs1"></a>Carlip, Carlip, Surya, “Path integral suppression of badly behaved causal sets,” [arXiv:2209.00327](https://arxiv.org/abs/2209.00327).
10. <a id="ref-ccs2"></a>Carlip, Carlip, Surya, “The Einstein-Hilbert action for entropically dominant causal sets,” [arXiv:2311.18238](https://arxiv.org/abs/2311.18238).
11. <a id="ref-mss"></a>Mathur, Singh, Surya, “Entropy and the link action in the causal set path-sum,” [arXiv:2009.07623](https://arxiv.org/abs/2009.07623).
12. <a id="ref-lp"></a>Last, Penrose, “Poisson process Fock space representation, chaos expansion and covariance inequalities,” [arXiv:0909.3205](https://arxiv.org/abs/0909.3205).
13. <a id="ref-rs"></a>Rideout, Sorkin, “A classical sequential growth dynamics for causal sets,” Phys. Rev. D 61, 024002 (2000), [arXiv:gr-qc/9904062](https://arxiv.org/abs/gr-qc/9904062).
14. <a id="ref-djs"></a>Dowker, Johnston, Surya, “On extending the quantum measure,” J. Phys. A 43, 505305 (2010), [arXiv:1007.2725](https://arxiv.org/abs/1007.2725).
15. <a id="ref-sz"></a>Surya, Zalel, “A criterion for covariance in complex sequential growth models,” [arXiv:2003.11311](https://arxiv.org/abs/2003.11311).
