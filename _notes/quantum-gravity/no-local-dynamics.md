---
title: Local rules of change on a sprinkling
date: 2026-10-29
category: entropy
order: 3
description: "On a Poisson sprinkling of Minkowski space no rule that completes a field value from finitely many other sprinkled elements is exactly consistent with the wave equation, and for path sums over three-layer orders the bound on the exponential suppression at fixed outer layers depends on the action only through the phase it gives to a link."
status: "Proved here: Theorems 1, 3, 8, 9 and 10, Corollary 2 and Propositions 4, 5, 6(a), 6(b), 7, 11 and 12. Computed here, and labelled so in the text: the error of a transcription of the rectangle rule (two dimensions), the fractions of pairs related by the dominance map (two and four dimensions, and the value 2/3 on the uniform square), exact sums over small layered classes, and variances of the smeared action. At Kleitman–Rothschild proportions the uniform bound exceeds 1 for large sizes whenever the action has depth at least one, and suppression there is not decided."
excludes: "Exactly wave-consistent completion rules on finitely many sprinkled elements (dimension two or higher). Order-revising laws that read only separate statistics, on a continuum or measured order with a symmetry that moves a point to one with a different future (an actual sprinkling has no such symmetry). Exponential suppression of layered orders at Kleitman–Rothschild proportions for an action beyond the link action, which the uniform bound does not give."
---

A sprinkling of Minkowski space has no coordinates and no neighbours. It is a partial order. We ask which rules of change can be written on it using finitely many elements at a time, for three kinds of rule, and state the conditions under which each answer holds.

A rule that predicts the value of a field at a sprinkled element from its values at finitely many other sprinkled elements cannot be exactly consistent with the massless wave equation, in any dimension $$d\ge2$$ (Theorem 3). A law that decides whether $$x$$ precedes $$y$$ from statistics of $$x$$ and of $$y$$ taken separately, such as the sizes of their pasts and futures, cannot leave fixed an order that has a symmetry moving a point to a point with a different future; a Minkowski diamond with a spatial reflection is such an order (Proposition 5). A sprinkling has no such symmetry. For the law that relates an element to every element with more ancestors and fewer descendants, an open set of directions around every interior point of a smooth density is related to incomparable points (Proposition 6), and the computed fraction of incomparable pairs that it relates is about 0.67 on sprinklings in two dimensions and 0.55 to 0.62 in four.

The third kind of rule is a phase assigned to each order, with the dynamics given by the sum over orders. For three-layer orders with outer layers of fixed size and a finite-depth action built from interval counts, the sum over the class is exponentially small in the size of the middle layer when the phase the action gives to one link is not a multiple of $$2\pi$$, at a rate bounded by a quantity that depends on the action only through that phase (Theorems 8 and 9). When the phase is a multiple of $$2\pi$$ the sum is not suppressed: exactly so for two layers (Theorem 8), and in the limit of a large middle layer for three (Proposition 11). At the layer proportions $$n/4,n/2,n/4$$ that almost all orders have, the uniform bound of Theorem 10 is exponentially small in $$n^2$$ for the link action in a range of the phase, and for large $$n$$ exceeds $$1$$ for every action of depth at least one (weights on intervals containing one or more elements), so it does not decide suppression there.

Two further results belong to the same programme. A decoherence functional that is a product of one amplitude per history, as in complex transitive percolation, has rank one and does not decohere a binary coarse-graining unless one side has zero measure (Proposition 7). Smearing the interval-count action over a nonlocality scale is a thinning of the sprinkling, and it bounds the variance of the smeared action by the retained fraction times the variance of the unsmeared action at the thinned density (Proposition 12).

Sorkin [[Sorkin]](#ref-sorkin) argued that Lorentz-invariant discreteness forces non-locality. The d'Alembertians of Benincasa and Dowker [[BD]](#ref-bd) and of Aslanbeigi, Saravani and Sorkin [[ASS]](#ref-ass) are built from interval counts over mesoscopic regions, and Johnston's propagator [[Johnston]](#ref-johnston) sums over chains of the order. Theorem 3 is an exact statement about completion rules. In one space dimension a rewriting rule that is exactly consistent with the wave equation does exist on the lattice of null coordinates: it fills the fourth corner of a rectangle from the other three. Its algebra is that of a heap, treated as pure algebra in [[ZAB]](#ref-zab), and in the abelian case it is the parallelogram identity satisfied by solutions of the one-dimensional wave equation [[KRud]](#ref-krud). Section 2 shows that this is the only rule, under two assumptions, for which the answer does not depend on how a large rectangle is cut into small ones, and section 3 shows why it cannot be moved onto the sprinkled elements. For path sums, Kleitman and Rothschild [[KR75]](#ref-kr75) showed that almost all partial orders have three layers. Loomis and Carlip [[LC]](#ref-lc) showed that the Benincasa–Dowker action suppresses bilayer orders, Carlip, Carlip and Surya [[CCS1]](#ref-ccs1) [[CCS2]](#ref-ccs2) that for a typical layered order it reduces to the link action to leading order, and Mathur, Singh and Surya [[MSS]](#ref-mss) that the link action suppresses the Kleitman–Rothschild orders over a range of parameters. What is added here is an exact transfer-matrix treatment of the sum over a three-layer class for every finite-depth action, which separates what depends on the choice of action from what does not. The [map of quantum-gravity theories](/quantum-gravity/landscape/) places these results among other approaches.

## 1. Setting and three kinds of rule

A sprinkling is a Poisson point process of density $$\rho$$ in a region of $$d$$-dimensional Minkowski space, with signature mostly plus and the induced causal order. A field theory on it needs a way to propagate values, and a dynamics of the order needs a way to change relations or to weight them.

A *completion rule* gives, for a target element $$t$$ and a finite set $$S$$ of other elements, a function that predicts $$\varphi(t)$$ from $$\varphi\vert_S$$. A *pair law* takes a finite order and returns a new relation, deciding for each pair $$x,y$$ whether $$x$$ precedes $$y$$ in the new relation; the laws of section 4 read only numbers attached to $$x$$ and to $$y$$ separately. A *path-sum weight* assigns a phase to each order and leaves the orders themselves unchanged; the dynamics is the sum over orders (section 6). Sections 5 and 7 concern two properties of such sums: the rank of the decoherence functional of sequential growth, and the fluctuations of the action on manifold-like orders.

## 2. Rectangle rules on the null lattice

In $$1+1$$ dimensions use null coordinates $$u,v$$. The massless wave equation is $$\partial_u\partial_v\varphi=0$$, its solutions are $$\varphi=f(u)+g(v)$$, and on every rectangle with corners $$b=(u_1,v_1)$$, $$l=(u_1,v_2)$$, $$r=(u_2,v_1)$$, $$t=(u_2,v_2)$$ they satisfy $$\varphi(t)=\varphi(l)+\varphi(r)-\varphi(b)$$. The identity holds for rectangles of every size, so the value at $$t$$ can be computed from the other three corners directly, or after cutting the rectangle in two and computing the value at the new corner first. That the two routes agree is the property studied here. We ask which rules $$t=F(l,b,r)$$, for values in an arbitrary set $$X$$, have it.

Let $$\Lambda=\{0,\dots,N\}^2$$ be the null lattice. A rectangle is a quadruple $$b=(a,c)$$, $$l=(a,j)$$, $$r=(i,c)$$, $$t=(i,j)$$ with $$a<i$$ and $$c<j$$, of any size. A field is a map from a subset of $$\Lambda$$ to $$X$$. Given a map $$F:X^3\to X$$, a rewriting step picks a rectangle whose corners $$b,l,r$$ carry values and whose corner $$t$$ does not, and sets $$\varphi(t)=F(\varphi(l),\varphi(b),\varphi(r))$$. Two assumptions are made on $$F$$.

* (H1) The same $$F$$ is used for every rectangle, whatever its size.
* (H2) $$F$$ is reversible in the two outer arguments: for fixed $$b$$ and $$r$$ the map $$l\mapsto F(l,b,r)$$ is a bijection of $$X$$, and for fixed $$l$$ and $$b$$ the map $$r\mapsto F(l,b,r)$$ is a bijection.

Goursat data are the values of $$\varphi$$ on the two lines $$\{0\}\times\{0,\dots,N\}$$ and $$\{0,\dots,N\}\times\{0\}$$, agreeing at the origin. A history is a sequence of rewriting steps that fills $$\Lambda$$ starting from Goursat data. Histories exist, since a point $$t=(i,j)$$ with $$i,j\ge1$$ and $$i+j$$ minimal among the empty points has all three other corners of the unit rectangle filled. The rule $$F$$ is *confluent* if for every Goursat data every history ends with the same field on $$\Lambda$$.

A heap operation on a group $$G$$ is the map $$F(l,b,r)=l\,b^{-1}r$$. Writing the two ways of cutting a rectangle as identities gives, for all values in $$X$$,

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

In that case, for Goursat data $$\varphi(0,j)=\psi_L(j)$$ and $$\varphi(i,0)=\psi_B(i)$$, every history produces $$\varphi(i,j)=A(j)\,B(i)$$ with $$A(j)=\psi_L(j)\psi_L(0)^{-1}$$ and $$B(i)=\psi_B(i)$$. If $$X$$ is a manifold and $$F$$ is smooth, the group is a Lie group.
</div>

Identities (1) and (2) are the two ways of cutting a rectangle, and reversibility turns them into the associativity law of the product $$x\cdot y=F(x,e,y)$$ for a fixed element $$e$$. Both identities are needed, since each alone admits operations that are not heaps (see the counts after the proof).

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

(c)$$\Rightarrow$$(a). Let $$F$$ be the heap operation of a group. Along any history every filled point $$(i,j)$$ carries $$A(j)B(i)$$: this holds on the Goursat lines, since $$A(0)=e$$ and $$B(0)=\psi_B(0)=\psi_L(0)$$, and a step with corners $$A(j)B(a)$$, $$A(c)B(a)$$, $$A(c)B(i)$$ produces $$A(j)B(a)\,B(a)^{-1}A(c)^{-1}\,A(c)B(i)=A(j)B(i)$$. The final field is therefore independent of the history.

For the last statement, $$x\cdot y=F(x,e,y)$$ is smooth, and so is the inverse $$x^{-1}=F(e,x,e)$$. $$\square$$
</div>

An exhaustive search over all maps $$F:X^3\to X$$ that are reversible in the sense of (H2) and satisfy (1) and (2), on sets of $$n=2,3,4$$ elements, finds $$1,1,4$$ of them, all heap operations. The number of heap operations on an $$n$$-element set is $$\sum_G (n-1)!/\lvert\mathrm{Aut}\,G\rvert$$ over isomorphism classes of groups of order $$n$$, which is $$1,1,4$$ for $$n=2,3,4$$ ($$\mathbb Z_4$$ contributes $$3$$ and $$\mathbb Z_2^2$$ contributes $$1$$). Imposing (1) alone leaves $$1,2,24$$ solutions and (2) alone leaves $$1,2,24$$ for $$n=2,3,4$$, of which only $$1,1,4$$ are heaps.

In a group, a field $$\varphi(i,j)=A(j)B(i)$$ has $$\partial_v\varphi\,\varphi^{-1}$$ independent of $$u$$, so in the continuum limit the rule reproduces the equation

$$
\partial_u\big(\partial_v\varphi\;\varphi^{-1}\big)=0,\tag{6}
$$

whose solutions are exactly the factorised fields $$\varphi=A(v)B(u)$$. For $$G=(\mathbb R,+)$$ it is the free wave equation. For non-abelian $$G$$ it is not the principal chiral equation $$\partial_u(\varphi^{-1}\partial_v\varphi)+\partial_v(\varphi^{-1}\partial_u\varphi)=0$$, which in null coordinates reads $$2\partial_u\partial_v\varphi=\partial_u\varphi\,\varphi^{-1}\partial_v\varphi+\partial_v\varphi\,\varphi^{-1}\partial_u\varphi$$ whereas (6) reads $$\partial_u\partial_v\varphi=\partial_v\varphi\,\varphi^{-1}\partial_u\varphi$$; the two coincide when the currents $$\partial_u\varphi\,\varphi^{-1}$$ and $$\partial_v\varphi\,\varphi^{-1}$$ commute, in particular for abelian $$G$$. If $$X=\mathbb R$$ and $$F$$ is smooth, the group is a connected, simply connected one-dimensional Lie group, hence isomorphic to $$(\mathbb R,+)$$ by a diffeomorphism $$h$$, so $$F=h^{-1}(h(l)-h(b)+h(r))$$: a free wave equation for the redefined field $$h\circ\varphi$$.

<div class="corollary" markdown="1">
**Corollary 2** (Linear confluent rules). Let $$X=\mathbb R$$ and $$F(l,b,r)=c_l\,l+c_b\,b+c_r\,r$$ with real $$c_l,c_r\neq0$$ and $$c_b$$. Then $$F$$ is confluent if and only if $$F(l,b,r)=l-b+r$$.
</div>

<div class="proof" markdown="1">
*Proof.* Expanding (1) gives $$c_l^2l+c_lc_bb+(c_lc_r+c_b)m+c_rr=c_ll+c_bb+c_rr$$, so $$c_l^2=c_l$$ and $$c_lc_r+c_b=0$$; with $$c_l\neq0$$ this gives $$c_l=1$$ and $$c_b=-c_r$$. Expanding (2) gives $$c_ll+(c_b+c_rc_l)l_1+c_rc_bb+c_r^2r=c_ll+c_bb+c_rr$$, so $$c_r^2=c_r$$, hence $$c_r=1$$ and $$c_b=-1$$. Conversely $$l-b+r$$ is the heap operation of $$(\mathbb R,+)$$. $$\square$$
</div>

A mass term is excluded for a short reason. For $$(\partial_u\partial_v+m^2)\varphi=0$$ the combination $$\varphi(t)-\varphi(l)-\varphi(r)+\varphi(b)$$ equals $$-m^2$$ times the integral of $$\varphi$$ over the rectangle, and two solutions can share the values at the three corners $$b,l,r$$ and differ in that integral, because $$\varphi(t)$$ depends on the data along the characteristic segments from $$b$$ to $$l$$ and from $$b$$ to $$r$$ and not only on their end points. No rule that sees only the corners reproduces it.

Theorem 1 concerns the lattice $$\Lambda$$ of all pairs of ranks. A two-dimensional sprinkling of $$n$$ points is a two-order: each element has integer null ranks $$(i(x),j(x))\in\{1,\dots,n\}^2$$ obtained by ordering the points along each null direction, and $$x\prec y$$ exactly when both ranks increase. Every rank occurs once, so no rectangle of the rank lattice has all four corners among the sprinkled elements, since two corners of a rectangle share a rank. The sprinkled elements occupy a fraction $$1/n$$ of the lattice sites, and a statement about rewriting on the lattice is a statement about a regular square lattice.

## 3. Completion rules on sprinkled elements

Let $$\Sigma\subset\mathbb R^{1,d-1}$$ be a finite set of points, $$d\ge2$$, and let $$t\in\Sigma$$ and $$S\subset\Sigma\setminus\{t\}$$ be such that $$t$$ is not null-separated from any point of $$S$$. A *completion rule* is any function $$F:\mathbb R^S\to\mathbb R$$ that may depend on the positions of all points and on the order; no locality is assumed on the choice of $$S$$. It is *exactly consistent with $$\Box\varphi=0$$* if $$F(\varphi\vert_S)=\varphi(t)$$ for every smooth solution $$\varphi$$ of the wave equation on Minkowski space. The same definition applies to fields with values in a group $$G$$ that contains a non-trivial one-parameter subgroup $$\exp(s\xi)$$, with the principal chiral equation $$\partial^\mu(g^{-1}\partial_\mu g)=0$$ in place of the wave equation, and with exact consistency meaning $$F(g\vert_S)=g(t)$$ for all smooth solutions.

Plane waves along a null direction give the counterexample. A function $$f(n\cdot x)$$ with $$n$$ null satisfies the wave equation and is constant on each null hyperplane $$n\cdot x=\text{const}$$. The finite set $$S$$ meets only finitely many such hyperplanes. If one can choose $$n$$ so that the hyperplane through $$t$$ contains no point of $$S$$, then $$f$$ can vanish on all the hyperplanes through $$S$$ and equal $$1$$ on the one through $$t$$.

<div class="theorem" markdown="1">
**Theorem 3** (No exact completion rule on sprinkled elements). Let $$d\ge2$$, and let $$S$$ and $$t$$ be as above. There are smooth solutions $$\varphi_1,\varphi_2$$ of $$\Box\varphi=0$$ with $$\varphi_1=0$$, $$\varphi_2\vert_S=0$$ and $$\varphi_2(t)=1$$. Consequently no completion rule on $$(S,t)$$ is exactly consistent with the wave equation. The same holds for the principal chiral equation with values in a group $$G$$ that contains a non-trivial one-parameter subgroup. For a Poisson sprinkling these hypotheses hold almost surely, simultaneously for all $$S$$ and $$t$$.
</div>

<div class="proof" markdown="1">
*Proof.* Put $$\Delta_s=x_t-x_s$$ for $$s\in S$$; each $$\Delta_s$$ is non-zero and not null by assumption. The null directions are the vectors $$n=(1,\hat n)$$ with $$\hat n$$ on the unit sphere of $$\mathbb R^{d-1}$$ (two points when $$d=2$$). For such $$n$$, $$n\cdot\Delta_s=-\Delta_s^0+\hat n\cdot\vec\Delta_s$$. This vanishes for no $$\hat n$$ if $$\Delta_s$$ is timelike, because $$\lvert\hat n\cdot\vec\Delta_s\rvert\le\lvert\vec\Delta_s\rvert<\lvert \Delta_s^0\rvert$$; and if $$\Delta_s$$ is spacelike it vanishes on a set of directions that is a proper closed subset of the sphere without interior (empty for $$d=2$$ since $$\Delta_s$$ is not null). A finite union of such sets does not cover the sphere, so there is a null $$n$$ with $$n\cdot x_t\notin\{n\cdot x_s:s\in S\}$$. Choose a smooth $$f:\mathbb R\to\mathbb R$$ that vanishes at the finitely many numbers $$n\cdot x_s$$ and equals $$1$$ at $$n\cdot x_t$$. Then $$\varphi_2(x)=f(n\cdot x)$$ satisfies $$\Box\varphi_2=n^\mu n_\mu f''=0$$, vanishes on $$S$$ and equals $$1$$ at $$t$$. If $$F$$ were exactly consistent, $$F(0)=\varphi_1(t)=0$$ and $$F(0)=\varphi_2(t)=1$$, a contradiction.

For the group-valued case put $$g_1=e$$ and $$g_2(x)=\exp\big(f(n\cdot x)\xi\big)$$ with $$f$$ chosen as before except that $$f(n\cdot x_t)=s_0$$ with $$\exp(s_0\xi)\neq e$$. Since $$\exp(f\xi)$$ commutes with its derivative, $$g_2^{-1}\partial_\mu g_2=f'n_\mu \xi$$, whose divergence is $$n^\mu n_\mu f''\xi=0$$. The two solutions agree on $$S$$ and differ at $$t$$.

For the Poisson process, almost surely no two points coincide or are null-separated, and this property holds for all finite subsets at once. $$\square$$
</div>

The dimension plays no role, and neither does the choice of $$S$$: the rule may use any finite set of sprinkled elements, however it is selected, and may use the positions as well as the order. What the proof uses is generic position. The rectangle rule of Theorem 1 evades it for a concrete reason. In $$d=2$$ the only null directions are the two lattice directions. In a rectangle the corner $$r$$ has the same $$u$$ as $$t$$ and the corner $$l$$ has the same $$v$$, so the null hyperplane through $$t$$ always contains a point of $$S=\{b,l,r\}$$ for each of the two null directions, the general solution $$f(u)+g(v)$$ is determined by the three corners, and the proof breaks down. For sprinkled elements, null separation has probability zero. In $$d=2$$ the group-valued solutions $$\exp(f(u)\xi)$$ and $$\exp(f(v)\xi)$$ are factorised fields $$A(v)B(u)$$, so they also solve (6) and the same conclusion holds for rules consistent with (6).

The theorem says nothing about rules whose error tends to zero as the density grows. A rule with a defect that vanishes in the continuum limit, or one that uses a growing number of elements, is not covered. A scaling identity shows what such a rule would have to do.

<div class="proposition" markdown="1">
**Proposition 4** (Scaling of the error of an order-only rule). Let a completion rule be defined for each marked finite order $$(C,\prec,t)$$ by a set $$S$$ and an affine-equivariant function $$F$$, meaning $$F(\lambda\,\varphi_S+\lambda_0)=\lambda\,F(\varphi_S)+\lambda_0$$ for real $$\lambda,\lambda_0$$, and let both depend on the isomorphism class of $$(C,\prec,t)$$ only. For the linear field $$\varphi(x)=a\cdot x$$, with a fixed vector $$a$$, on a Poisson sprinkling of density $$\rho$$ in a causal interval of proper height $$T$$, let $$e_\rho(T)$$ be the mean of $$\lvert F(\varphi\vert_S)-\varphi(t)\rvert$$ over targets. Then

$$
e_\rho(T)=\rho^{-1/d}\,e_1\big(\rho^{1/d}T\big).\tag{7}
$$

</div>

<div class="proof" markdown="1">
*Proof.* The dilation $$x\mapsto\rho^{1/d}x$$ maps a sprinkling of density $$\rho$$ in an interval of height $$T$$ onto a sprinkling of density $$1$$ in an interval of height $$\rho^{1/d}T$$, preserves the order, hence the selection of $$S$$ and the function $$F$$, and turns the data $$a\cdot x$$ into $$(\rho^{-1/d}a)\cdot x'$$. By equivariance the error is multiplied by $$\rho^{-1/d}$$. $$\square$$
</div>

Write $$\theta=\rho^{1/d}T$$ for the height in units of the discreteness length. Equation (7) says that the error of an order-only rule at fixed $$T$$ tends to zero as $$\rho\to\infty$$ if and only if $$e_1(\theta)=o(\theta)$$: the error per step in units of the discreteness scale must grow more slowly than the size of the region. If the error scales as $$e_\rho\propto\rho^{\gamma}$$, then $$e_1(\theta)\propto\theta^{d\gamma+1}$$, and a rule whose error per step stays bounded has $$\gamma=-1/d$$.

The rectangle rule has a transcription to order data in two dimensions, which we call the join-heap rule. For a target $$t$$, take two maximal elements $$l,r$$ of its past (its past links), so that $$t$$ is minimal in $$J^+(l)\cap J^+(r)$$; take $$b$$ maximal in $$J^-(l)\cap J^-(r)$$; and predict $$\varphi(t)=\varphi(l)+\varphi(r)-\varphi(b)$$. We evaluated it on the wave-equation solution $$\varphi=u+v$$ in null coordinates on the unit square, for targets with $$u,v>0.5$$, averaging over sprinklings, targets and admissible triples $$(l,r,b)$$.

| points $$N$$ | 250 | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---|---|---|---|---|---|
| mean absolute error | 0.174 | 0.169 | 0.156 | 0.151 | 0.142 | 0.134 |
| spread of the prediction over admissible triples for one target | 0.168 | 0.179 | 0.173 | 0.175 | 0.173 | 0.171 |

By (7) with $$d=2$$, a rule with bounded error per step would give an error falling like $$N^{-1/2}$$, a factor $$5.7$$ over this range; the observed fall is a factor $$1.3$$. A least-squares fit of $$\ln e$$ on $$\ln N$$ gives $$\gamma\approx-0.08$$, that is $$e_1(\theta)\propto\theta^{0.85}$$. The selected elements are not confined to a neighbourhood of $$t$$. The longer null side of the box from $$b$$ to $$t$$ has mean $$0.45,0.39,0.35$$ for $$N=500,2000,8000$$, while the mean spacing $$N^{-1/2}$$ is $$0.045,0.022,0.011$$. The mean area of the box is small ($$0.077,0.050,0.039$$), because $$t$$ is minimal in the common future of $$l$$ and $$r$$, but the box is long in one null direction, so the elements involved are far from $$t$$ on the scale of the region. This is a computation for one rule in two dimensions with one way of selecting $$l$$ and $$r$$, and it is not a theorem.

## 4. Order-revising laws that read separate statistics

Suppose the relations of an order are to be re-decided by a law that looks, for each pair $$x,y$$, at numbers attached to $$x$$ and to $$y$$ separately, such as the sizes of the past and future cones, the height, or the number of chains through the element, and not at how $$x$$ and $$y$$ relate to each other. A manifold-like order should be a fixed point of such a law. We show that this is impossible when the order has a symmetry that moves a point to a point with a different future, and then study a concrete law of this kind.

Let $$(Q,<)$$ be a strict partial order, possibly carrying a measure (as a Minkowski diamond carries Lebesgue measure). An automorphism is an order isomorphism that preserves the measure, if there is one. A *statistic* is a function $$s:Q\to\Gamma$$ with values in a set $$\Gamma$$ that is invariant under the automorphisms: $$s(gx)=s(x)$$ for every automorphism $$g$$. A *separate-statistics law* is given by $$s$$ and a relation $$\Psi\subseteq\Gamma\times\Gamma$$, and returns the relation $$x\,R'\,y\iff\Psi(s(x),s(y))$$. The law *fixes* $$(Q,<)$$ if $$R'$$ equals $$<$$.

<div class="proposition" markdown="1">
**Proposition 5** (A fixed point has orbit-invariant futures). If a separate-statistics law fixes $$(Q,<)$$, then for every automorphism $$g$$ and all $$x,y\in Q$$, $$x<y$$ implies $$gx<y$$. Consequently an order that has an automorphism $$g$$, and elements $$x<y$$ with $$gx\not<y$$, is not fixed by any separate-statistics law.
</div>

<div class="proof" markdown="1">
*Proof.* If $$x<y$$ then $$\Psi(s(x),s(y))$$ holds. Since $$s(gx)=s(x)$$, also $$\Psi(s(gx),s(y))$$, that is $$gx\,R'\,y$$, and $$R'=<$$ gives $$gx<y$$. $$\square$$
</div>

The hypotheses are met by the configurations one wants to be fixed. The Minkowski diamond, with Lebesgue measure and cone volumes as the statistics, is mapped to itself by a spatial reflection $$g$$ through a hyperplane containing the time axis. Take an interior point $$x=(t,\vec x)$$ with $$\vec x\neq g\vec x$$, so that $$gx=(t,g\vec x)$$ is spacelike to $$x$$, and let $$y=x+\epsilon(1,\hat e)$$ with $$\hat e$$ the unit vector along $$\vec x-g\vec x$$ and $$\epsilon>0$$ small. Then $$y$$ is in the future of $$x$$, while $$y-gx=(\epsilon,\vec x-g\vec x+\epsilon\hat e)$$ is spacelike, so $$x<y$$ and $$gx\not<y$$. The same holds in every dimension $$d\ge2$$. In the two-dimensional null lattice $$[N]^2$$ the exchange of the two coordinates is such an automorphism. A randomised law whose output distribution depends only on the pair of statistics has the same limitation in expectation, since $$\Pr(gx\,R'\,y)=\Pr(x\,R'\,y)$$.

A finite sprinkling has almost surely a trivial automorphism group, and the proposition says nothing about it. What the law does to a sprinkling is a separate question, studied next for one law.

For a finite strict order $$R$$ on $$n$$ elements let $$n^-_R(i)$$ and $$n^+_R(i)$$ be the numbers of elements below and above $$i$$. The *cone-count dominance map* relates $$i$$ to $$j$$ when $$i$$ has fewer predecessors and more successors than $$j$$:

$$
\Phi(R)=\{(i,j):\ n^-_R(i)<n^-_R(j)\ \text{and}\ n^+_R(i)>n^+_R(j)\}.\tag{8}
$$

It is a separate-statistics law with $$s=(n^-_R,n^+_R)$$. Part (a) of the next proposition gives its exact properties on finite orders, and part (b) shows that on a smooth density of points it relates incomparable pairs. The quantities $$n^\mp$$ become the masses $$\nu^\mp(x)$$ of the sets below and above $$x$$ for a probability measure $$\nu$$.

<div class="proposition" markdown="1">
**Proposition 6** (Properties of the dominance map). (a) For every strict order $$R$$ on $$n$$ elements, $$R\subseteq\Phi(R)$$, $$\Phi(R)$$ is a strict order of order dimension at most two, the sequence $$R,\Phi(R),\Phi^2(R),\dots$$ is increasing and constant after at most $$n(n-1)/2$$ steps, and $$R=\Phi(R)$$ if and only if $$(n^-_R(i)-n^-_R(j))(n^+_R(i)-n^+_R(j))\ge0$$ for every pair $$i,j$$ that is incomparable in $$R$$. In particular no order of dimension $$\ge3$$ is a fixed point.

(b) Let $$\nu$$ be a probability measure on the open unit square in null coordinates $$(u,v)$$ with a bounded density $$h$$ that is continuous and positive on the open square, with the componentwise order, and let $$\nu^-(x)$$ and $$\nu^+(x)$$ be the $$\nu$$-masses of the sets below and above $$x$$. At an interior point $$x_0=(u_0,v_0)$$ put

$$
q^-_u=\int_0^{v_0}h(u_0,v')\,dv',\quad q^-_v=\int_0^{u_0}h(u',v_0)\,du',\quad q^+_u=\int_{v_0}^1h(u_0,v')\,dv',\quad q^+_v=\int_{u_0}^1h(u',v_0)\,du'.
$$

For every slope $$\tau$$ with $$0<\tau<\min(q^-_u/q^-_v,\;q^+_u/q^+_v)$$ there is $$\delta_0>0$$ such that for $$0<\delta<\delta_0$$ the point $$y=x_0+\delta(1,-\tau)$$ is incomparable to $$x_0$$ and satisfies $$\nu^-(x_0)<\nu^-(y)$$ and $$\nu^+(x_0)>\nu^+(y)$$: the continuum analogue of $$\Phi$$ relates $$x_0$$ to $$y$$. For $$\tau>\max(q^-_u/q^-_v,\,q^+_u/q^+_v)$$ the same holds with the two inequalities reversed, so that $$y$$ is related to $$x_0$$.
</div>

<div class="proof" markdown="1">
*Proof.* (a) If $$i<j$$ the down-set of $$j$$ contains the down-set of $$i$$ and $$i$$ itself, so $$n^-(i)<n^-(j)$$; the same argument for up-sets gives $$n^+(i)>n^+(j)$$. Hence $$R\subseteq\Phi(R)$$. The relation $$\Phi(R)$$ is the strict componentwise order on the points $$(n^-(i),-n^+(i))\in\mathbb Z^2$$, which is irreflexive and transitive, and it is the intersection of two linear orders: order the elements lexicographically by $$(n^-,n^+)$$ ascending and by $$(-n^+,-n^-)$$ ascending, and break exact ties by the index in one order and by the reverse index in the other. So it has dimension at most two. Applying $$R\subseteq\Phi(R)$$ to each term shows the sequence is increasing; a strict order has at most $$n(n-1)/2$$ relations. A fixed point satisfies $$\Phi(R)\subseteq R$$. For a pair $$i,j$$, one of $$(i,j)$$ and $$(j,i)$$ lies in $$\Phi(R)$$ exactly when $$(n^-(i)-n^-(j))(n^+(i)-n^+(j))<0$$. For a pair comparable in $$R$$ the product is negative and the pair is already in $$R$$ or its reverse, so only incomparable pairs can violate $$\Phi(R)\subseteq R$$, and they do so exactly when the product is negative. Fixed points have dimension at most two because they equal $$\Phi(R)$$.

(b) Since $$h$$ is bounded and continuous, differentiation under the integral shows that $$\nu^-(u,v)=\int_0^u\int_0^vh$$ and $$\nu^+(u,v)=\int_u^1\int_v^1h$$ are $$C^1$$ on the open square, with $$\partial_u\nu^-=q^-_u$$, $$\partial_v\nu^-=q^-_v$$, $$\partial_u\nu^+=-q^+_u$$ and $$\partial_v\nu^+=-q^+_v$$ at $$x_0$$. For $$y=x_0+\delta(1,-\tau)$$ the first-order changes are $$\Delta\nu^-=\delta(q^-_u-\tau q^-_v)+o(\delta)$$ and $$\Delta\nu^+=\delta(-q^+_u+\tau q^+_v)+o(\delta)$$, which are positive and negative respectively when $$\tau<q^-_u/q^-_v$$ and $$\tau<q^+_u/q^+_v$$. The point $$y$$ is incomparable to $$x_0$$ because its $$u$$-coordinate is larger and its $$v$$-coordinate smaller. The case $$\tau$$ large reverses both signs. $$\square$$
</div>

Part (a) leaves open whether the map moves a given order at all, since an order of dimension two with the right statistics could be a fixed point. Part (b) shows that on a smooth density it does move the order, and that the set of promoted directions is open. For the uniform measure on the unit square, $$\nu^-=uv$$ and $$\nu^+=(1-u)(1-v)$$. Numerical integration of the fraction of incomparable pairs that are related in one orientation or the other gives $$0.66667$$, in agreement with direct sampling of $$8\times10^7$$ pairs ($$0.6666\pm0.0001$$); we take the value to be $$2/3$$, which is numerical.

On sprinklings the fraction of spacelike pairs that one application of $$\Phi$$ relates is, averaged over three samples each,

| dimension 2, points $$n$$ | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| fraction related | 0.655 | 0.664 | 0.668 | 0.663 |

| dimension 4, points $$n$$ | 250 | 500 | 1000 | 1500 |
|---|---|---|---|---|
| fraction related | 0.547 | 0.580 | 0.614 | 0.620 |

and no relation of the original order was lost, as part (a) requires. Iterating $$\Phi$$ on a single sprinkling of $$n=600$$ points ended after $$32$$ steps in two dimensions, with the fraction of related pairs rising from $$0.518$$ to $$0.988$$ and the height from $$42$$ to $$257$$, and after $$9$$ steps in four dimensions, with the fraction rising from $$0.098$$ to $$0.874$$ and the height from $$9$$ to $$163$$. The fixed points reached are close to chains.

A law that reads data of the pair itself, such as the number of common predecessors $$\lvert\downarrow x\cap\downarrow y\rvert$$, the number of elements in the interval, or the links between them, is not covered, because a symmetry that preserves the separate statistics of $$x$$ and $$y$$ need not preserve those.

## 5. Decoherence for a rank-one functional

Classical sequential growth builds a causal set one element at a time, with probabilities for the new element's past [[RS]](#ref-rs). Complex sequential growth replaces the probabilities by complex amplitudes [[SZ]](#ref-sz); complex transitive percolation [[DJS]](#ref-djs) is the case with a single parameter $$p$$ and $$q=1-p$$, in which a labelled order with $$N_0$$ links and $$N_{\mathrm{rel}}$$ relations has amplitude $$p^{N_0}q^{\binom n2-N_{\mathrm{rel}}}$$. The amplitude $$\mathcal A(E)$$ of an event is the sum of the amplitudes of the labelled orders in it, and the decoherence functional is the product $$\mathcal D(E,E')=\overline{\mathcal A(E)}\,\mathcal A(E')$$ with $$\mathcal A(\Omega)=1$$ for the whole space $$\Omega$$. This product form, and the one-dimensional space of histories it implies, are noted in [[DJS]](#ref-djs), which also discusses the extension of the functional to covariant events together with [[SZ]](#ref-sz). The consequence for coarse-grainings is the following.

<div class="proposition" markdown="1">
**Proposition 7** (A rank-one functional does not decohere a binary coarse-graining). Let $$\mathcal D(E,E')=\overline{\mathcal A(E)}\mathcal A(E')$$ with $$\mathcal A(\Omega)=1$$ and $$\mathcal A$$ additive on disjoint events, and let $$E$$ be an event with complement $$E^c$$. If $$\mathcal A(E)\mathcal A(E^c)\neq0$$ then $$\lvert\mathcal D(E,E^c)\rvert/\sqrt{\mathcal D(E,E)\mathcal D(E^c,E^c)}=1$$. The off-diagonal element $$\mathcal D(E,E^c)$$ vanishes if and only if $$\mathcal A(E)\in\{0,1\}$$, that is if one of $$E,E^c$$ has zero quantum measure.
</div>

<div class="proof" markdown="1">
*Proof.* $$\lvert\mathcal D(E,E^c)\rvert=\lvert\mathcal A(E)\rvert\lvert\mathcal A(E^c)\rvert=\sqrt{\mathcal D(E,E)\mathcal D(E^c,E^c)}$$; this is the Cauchy–Schwarz inequality with equality in the one-dimensional space $$\mathbb C$$. Since $$\mathcal A(E^c)=1-\mathcal A(E)$$, the product $$\mathcal A(E)\mathcal A(E^c)$$ vanishes exactly when $$\mathcal A(E)=0$$ or $$\mathcal A(E)=1$$. $$\square$$
</div>

The statement does not depend on the event, on the number of elements or on the value of $$p$$, and it applies to real $$p$$ (where the model is classical percolation seen as an amplitude). Decoherence between events needs a history space of dimension at least two. If the transition operators on $$\mathbb C^m$$ are commuting and normal, they can be diagonalised simultaneously and the vector-valued measure splits into $$m$$ orthogonal channels $$\mathcal A_k$$, each satisfying the sum rule, with $$\mathcal D(E,E')=\sum_k\lvert c_k\rvert^2\overline{\mathcal A_k(E)}\mathcal A_k(E')$$; this is the first case in which the question can be asked non-trivially, and it needs $$m\ge2$$ channels.

## 6. Path sums over three-layer orders

A sum over orders, $$Z=\sum_{\mathcal O}e^{iS(\mathcal O)/\hbar}$$, can be dominated by manifold-like orders only if the phases suppress the rest. The difficulty is entropy. By the theorem of Kleitman and Rothschild [[KR75]](#ref-kr75), almost all partial orders on $$n$$ elements have three layers, of sizes close to $$n/4$$, $$n/2$$ and $$n/4$$ (the Kleitman–Rothschild proportions), and their number is $$2^{n^2/4+o(n^2)}$$. Almost all of these orders are very far from a manifold. For a given action the question is whether its phases cancel enough in the sum over the three-layer class that the class does not dominate. For an action built from interval counts the sum over the class can be done exactly.

### 6.1 The action, the class and two layers

For a finite order, let $$N_k$$ be the number of pairs $$x\prec y$$ with exactly $$k$$ elements strictly between them, so $$N_0$$ is the number of links, and let $$n$$ be the number of elements. A finite-depth action is

$$
\frac{S}{\hbar}=\beta\Big(b\,n-\sum_{k=0}^{L}C_kN_k\Big),\tag{9}
$$

with real parameters $$\beta,b,C_0,\dots,C_L$$. The four-dimensional Benincasa–Dowker action [[BD]](#ref-bd) has $$L=3$$, $$b=1$$ and $$C=(1,-9,16,-8)$$, with $$\beta$$ proportional to $$(\ell/\ell_p)^2$$ in terms of the discreteness length $$\ell$$ and the Planck length $$\ell_p$$; the two-dimensional one has $$L=2$$, $$b=1$$ and $$C=(2,-4,2)$$. The coupling $$\beta$$ is a free parameter in all that follows. The *link action* has only $$C_0\neq0$$. The *link phase* is $$z=e^{-i\beta C_0}$$, the phase the action assigns to one link, and $$f(k)=e^{-i\beta C_k}$$, with $$f(k)=1$$ for $$k>L$$ and $$f(0)=z$$.

Fix layer sizes $$(n_1,n_2,n_3)$$ and call the layers $$Y_1,Y_2,Y_3$$. An order of the class $$\mathcal K(n_1,n_2,n_3)$$ is given by two bipartite relations $$P_{12}\subseteq Y_1\times Y_2$$ and $$P_{23}\subseteq Y_2\times Y_3$$ with $$a\prec v$$ for $$(a,v)\in P_{12}$$, $$v\prec c$$ for $$(v,c)\in P_{23}$$, and $$a\prec c$$ for all $$a\in Y_1$$, $$c\in Y_3$$. It has $$\lvert\mathcal K\rvert=2^{n_2(n_1+n_3)}$$ members, which is $$2^{n^2/4}$$ at proportions $$(n/4,n/2,n/4)$$, the leading exponent of the number of all orders. For $$a\in Y_1$$ and $$c\in Y_3$$ the number of elements strictly between them is $$k_{ac}=\lvert\{v:(a,v)\in P_{12},(v,c)\in P_{23}\}\rvert$$. The links are the elements of $$P_{12}$$ and $$P_{23}$$ together with the pairs $$(a,c)$$ with $$k_{ac}=0$$. We write

$$
Z(n_1,n_2,n_3)=\sum_{\mathcal O\in\mathcal K}e^{iS(\mathcal O)/\hbar}.
$$

With two layers there is nothing between any pair, so only links occur: the bilayer class consists of the orders on $$Y_1\cup Y_2$$ with $$a\prec v$$ for $$(a,v)\in G\subseteq Y_1\times Y_2$$.

<div class="theorem" markdown="1">
**Theorem 8** (Exact bilayer sum). For layers of sizes $$m_1,m_2$$ and the action (9),

$$
Z=e^{i\beta b(m_1+m_2)}\,(1+z)^{m_1m_2},\qquad\frac{\lvert Z\rvert}{2^{m_1m_2}}=\Big\lvert\cos\frac{\beta C_0}{2}\Big\rvert^{m_1m_2}.\tag{10}
$$

For every finite-depth action the bilayer sum depends on the action only through the link phase $$\beta C_0$$ (and a global phase). If $$\beta C_0\in2\pi\mathbb Z$$ the sum is not suppressed at all.
</div>

<div class="proof" markdown="1">
*Proof.* In a bilayer every interval is empty, so $$N_k=0$$ for $$k\ge1$$ and $$N_0=\lvert G\rvert$$, and $$e^{iS/\hbar}=e^{i\beta bn}z^{\lvert G\rvert}$$. Summing over all $$G\subseteq Y_1\times Y_2$$ gives $$\sum_Gz^{\lvert G\rvert}=(1+z)^{m_1m_2}$$, and $$\lvert1+z\rvert=2\lvert\cos(\beta C_0/2)\rvert$$. $$\square$$
</div>

A brute-force sum over all $$2^{16}$$ relation patterns on layers $$(4,4)$$, with $$\beta=2\arccos0.4$$, the four-dimensional Benincasa–Dowker coefficients and the interval counts computed from the order itself, gives $$\lvert Z\rvert=0.0281475$$ and $$\lvert Z\rvert/2^{16}=4.2950\times10^{-7}$$, against $$0.4^{16}=4.2950\times10^{-7}$$ from (10).

### 6.2 The transfer matrix

In the three-layer class the sum organises itself by the elements of the middle layer. Each $$v\in Y_2$$ has a set $$u_v\subseteq Y_1$$ of lower neighbours and $$w_v\subseteq Y_3$$ of upper neighbours, chosen independently of the other middle elements, with weight $$z^{\lvert u_v\rvert+\lvert w_v\rvert}$$ for its links. The number $$k_{ac}$$ is the number of $$v$$ with $$(a,c)\in u_v\times w_v$$. The weight of the whole order is therefore

$$
e^{iS/\hbar}=e^{i\beta bn}\;z^{\lvert P_{12}\rvert+\lvert P_{23}\rvert}\prod_{a\in Y_1,c\in Y_3}f(k_{ac}).\tag{11}
$$

Adding the middle elements one at a time, what is needed to continue is the matrix $$\bar k=(\min(k_{ac},L+1))$$ of common-neighbour counts capped at $$L+1$$, since $$f(k)=1$$ beyond the cap. It takes $$(L+2)^{n_1n_3}$$ values. A middle element with data $$(u,w)$$ adds $$1$$ to the entries in $$u\times w$$, with weight $$z^{\lvert u\rvert+\lvert w\rvert}$$. Call $$M$$ the matrix of these weights; its entries are sums of powers of $$z$$. Counts never decrease, so $$M$$ is triangular for the componentwise order on states. Its diagonal entry at a state depends only on the set $$U=\{(a,c):\bar k_{ac}\le L\}$$ of uncapped entries, because a move leaves the state unchanged exactly when $$(u\times w)\cap U=\varnothing$$. Summing over such moves, and for each $$u$$ over the allowed $$w\subseteq Y_3\setminus N_U(u)$$, gives

$$
\lambda_U=\sum_{(u,w):\,(u\times w)\cap U=\varnothing}z^{\lvert u\rvert+\lvert w\rvert}=\sum_{u\subseteq Y_1}z^{\lvert u\rvert}(1+z)^{\,n_3-\lvert N_U(u)\rvert},\qquad N_U(u)=\{c:(a,c)\in U\text{ for some }a\in u\}.\tag{12}
$$

The final weight of a state is $$g(\bar k)=\prod f(\bar k_{ac})$$, and $$Z=e^{i\beta bn}\,(M^{n_2})_{0\cdot}\,g$$, where $$0$$ is the zero state. Let $$\lambda_*=\max_U\lvert\lambda_U\rvert$$, the maximum over all subsets $$U\subseteq Y_1\times Y_3$$, and $$\kappa=\lambda_*/2^{n_1+n_3}$$. The quantities $$\lambda_U$$, $$\lambda_*$$ and $$\kappa$$ depend on $$z$$, $$n_1$$ and $$n_3$$ only. They do not depend on $$b$$, on $$C_1,\dots,C_L$$, or on $$L$$.

### 6.3 Suppression at fixed outer layers

The weights $$C_1,\dots,C_L$$ do not enter the entries of $$M$$, which depend on $$z$$ alone; they enter through the cap $$L+1$$, which fixes the set of states, and through the final weights $$g$$. The exponential rate is set by the diagonal entries of $$M$$, which see only the links through $$z$$, and the weights beyond the link affect the polynomial prefactor.

<div class="theorem" markdown="1">
**Theorem 9** (Suppression at fixed outer layers). Fix $$n_1,n_3\ge1$$ and a finite-depth action (9). Let $$\kappa=\max_U\lvert\lambda_U\rvert/2^{n_1+n_3}$$ with $$\lambda_U$$ as in (12), $$J=(L+1)n_1n_3$$, and $$\mu=\max(\lvert\cos(\beta C_0/2)\rvert,\tfrac12)$$. Then

$$
\kappa\le\mu^{\max(n_1,n_3)},\qquad\kappa\ge\Big\lvert\cos\frac{\beta C_0}{2}\Big\rvert^{n_1+n_3},\tag{13}
$$

and for $$\kappa>0$$ and all $$n_2\ge1$$

$$
\frac{\lvert Z(n_1,n_2,n_3)\rvert}{\lvert\mathcal K\rvert}\le(J+1)\,n_2^{\,J}\,\kappa^{\,n_2-J}.\tag{14}
$$

If $$\kappa=0$$ then $$Z=0$$ for $$n_2>J$$. In particular, if $$\lvert\cos(\beta C_0/2)\rvert<1$$ then $$\kappa<1$$ and the sum over the class is exponentially small in $$n_2$$. The exponential factor in (14) depends on the action only through $$z=e^{-i\beta C_0}$$. The action enters the prefactor, a polynomial in $$n_2$$ of degree at most $$J$$.
</div>

The bound (14) is below $$1$$ only once $$(n_2-J)\ln(1/\kappa)$$ exceeds $$J\ln n_2+\ln(J+1)$$. For the four-dimensional Benincasa–Dowker action ($$L=3$$) with $$n_1=n_3=3$$ and $$\lvert\cos(\beta C_0/2)\rvert=0.4$$ one has $$J=36$$ and $$\kappa=0.0623$$, and this needs $$n_2$$ of order $$10^2$$.

<div class="proof" markdown="1">
*Proof.* Triangularity was shown above: moves only increase entries, so $$M_{\sigma\sigma'}\neq0$$ implies $$\sigma'\ge\sigma$$ componentwise. A term of $$(M^{n_2})_{0\bar k}$$ is a product along a path of states $$0=\sigma_0\le\sigma_1\le\dots\le\sigma_{n_2}$$. A step that changes the state raises at least one entry by $$1$$, so it raises the sum of the entries by at least $$1$$; that sum of the entries lies between $$0$$ and $$(L+1)n_1n_3=J$$, hence a path has at most $$J$$ steps that change the state. At the $$n_2-j$$ steps that do not change the state the factor is a diagonal entry, of modulus at most $$\lambda_*$$. At a step that changes the state, the entries from the current state to all other states are sums over distinct moves of unimodular terms, so their moduli add up to at most the number $$2^{n_1+n_3}$$ of moves. There are $$\binom{n_2}{j}$$ choices of the positions of the $$j$$ changing steps, and therefore, summing over the destination of each changing step and over the final state,

$$
\sum_{\bar k}\big\lvert(M^{n_2})_{0\bar k}\big\rvert\le\sum_{j=0}^{\min(J,n_2)}\binom{n_2}{j}2^{(n_1+n_3)j}\lambda_*^{\,n_2-j}.
$$

Since $$\lvert g\rvert=1$$, $$\lvert Z\rvert$$ is at most this sum. Dividing by $$\lvert\mathcal K\rvert=2^{n_2(n_1+n_3)}$$ turns each term into $$\binom{n_2}{j}\kappa^{\,n_2-j}$$, and with $$\kappa\le1$$, $$\binom{n_2}{j}\le n_2^{\,j}\le n_2^{\,J}$$ and $$\kappa^{\,n_2-j}\le\kappa^{\,n_2-J}$$ the sum over the at most $$J+1$$ values of $$j$$ gives (14). If $$\kappa=0$$, every diagonal entry vanishes, a chain of states has at most $$J+1$$ members, and $$M^{n_2}=0$$ for $$n_2>J$$.

For the upper bound in (13), $$\lvert1+z\rvert=2\lvert\cos(\beta C_0/2)\rvert\le2\mu$$, and $$\lvert N_U(u)\rvert\le n_3$$, so each term of (12) satisfies

$$
\frac{\lvert z^{\lvert u\rvert}(1+z)^{n_3-\lvert N_U(u)\rvert}\rvert}{2^{n_3}}=\Big\lvert\cos\frac{\beta C_0}{2}\Big\rvert^{n_3-\lvert N_U(u)\rvert}2^{-\lvert N_U(u)\rvert}\le\mu^{n_3},
$$

since both $$\lvert\cos\rvert\le\mu$$ and $$\tfrac12\le\mu$$. The sum has $$2^{n_1}$$ terms, so $$\lvert\lambda_U\rvert/2^{n_1+n_3}\le\mu^{n_3}$$. The first form of $$\lambda_U$$ in (12) is symmetric in $$Y_1$$ and $$Y_3$$, so summing over $$u$$ first instead gives $$\mu^{n_1}$$, hence $$\kappa\le\mu^{\max(n_1,n_3)}$$. For the lower bound, $$U=\varnothing$$ has $$N_U(u)=\varnothing$$ for all $$u$$, so $$\lambda_\varnothing=(1+z)^{n_3}\sum_uz^{\lvert u\rvert}=(1+z)^{n_3+n_1}$$. $$\square$$
</div>

For $$\lvert\cos(\beta C_0/2)\rvert=0.4$$, so $$\mu=1/2$$, (12) gives $$\lambda_*=1.510,\,2.280,\,3.986$$ for $$n_1=n_3=1,2,3$$, hence exponential rates $$-\ln\kappa=0.974,\,1.948,\,2.776$$ per middle element, inside the intervals $$[\,n\ln2,\ 2n\ln(1/0.4)\,]=[0.693,1.833]$$, $$[1.386,3.665]$$, $$[2.079,5.498]$$ allowed by (13) for $$n_1=n_3=n$$. The exact sums, computed by enumerating the middle layer by the multiset of its neighbourhood types, decrease with the following local slopes $$-\Delta\ln\lvert Z\rvert/\Delta n_2$$.

| layers | action | local slopes for $$n_2=8\to24$$ (step 4) | rate $$-\ln\kappa$$ |
|---|---|---|---|
| $$(1,n_2,1)$$ | link | $$0.974,0.974,0.974,0.974$$ | $$0.974$$ |
| $$(1,n_2,1)$$ | Benincasa–Dowker, $$d=4$$ | $$0.585,0.739,0.802,0.836$$ | $$0.974$$ |
| $$(2,n_2,2)$$ | link | $$2.10,2.07,1.77,2.03$$ | $$1.948$$ |
| $$(2,n_2,2)$$ | Benincasa–Dowker, $$d=4$$ | $$0.91,1.17,1.35,1.49$$ | $$1.948$$ |
| $$(3,n_2,3)$$ | link | $$2.95,2.91,2.90,2.83$$ (step 2, $$n_2=8\to16$$) | $$2.776$$ |
| $$(3,n_2,3)$$ | Benincasa–Dowker, $$d=4$$ | $$0.80,1.23,1.47,1.71$$ (step 2, $$n_2=8\to16$$) | $$2.776$$ |

For the link action the local slopes lie close to the rate. Theorem 9 bounds $$\lvert Z\rvert/\lvert\mathcal K\rvert$$ with a polynomial prefactor, so a local slope can fall below the rate while the asymptotic decay is not slower; the entry $$1.77$$ in the $$(2,n_2,2)$$ row against $$1.948$$ is of this kind. For the Benincasa–Dowker coefficients the local slopes are below the rate and rising, as a prefactor of large degree produces. The theorem gives the rate only as an upper bound for $$\lvert Z\rvert$$, and whether it is attained for a given action depends on whether the dominant term has a non-zero coefficient, which the theorem does not decide. The table illustrates $$\kappa$$ as a rate. For the Benincasa–Dowker rows with layers $$(2,n_2,2)$$ and $$(3,n_2,3)$$ ($$J=16$$ and $$36$$) the bound (14) is not yet below $$1$$ at these sizes.

The rate in (14) is for fixed $$n_1$$ and $$n_3$$. At Kleitman–Rothschild proportions all three layer sizes grow together, and the next result covers all sizes at once.

### 6.4 A uniform bound and its logarithmic gap

<div class="theorem" markdown="1">
**Theorem 10** (Uniform bound, and its logarithmic gap). Let $$B_L(p)=\sum_{j=0}^{L}\binom pj$$ and $$\mu$$ as above. For all $$n_1,n_2,n_3$$ and every finite-depth action,

$$
\frac{\lvert Z\rvert}{\lvert\mathcal K\rvert}\le\mu^{\,n_2n_3}\Big(\mathbb E_{p\sim\mathrm{Bin}(n_2,1/2)}\big[(1+2B_L(p))^{n_3}\big]\Big)^{n_1}.\tag{15}
$$

(a) For the link action ($$L=0$$, $$B_0=1$$) the right-hand side is $$\big(\mu^{n_2}3^{n_1}\big)^{n_3}$$. At proportions $$(n/4,n/2,n/4)$$ it is $$\exp\big[\tfrac{n^2}{8}\ln(\sqrt3\,\mu)\big]$$, exponentially small in $$n^2$$ whenever $$\lvert\cos(\beta C_0/2)\rvert<1/\sqrt3$$.

(b) Let $$L\ge1$$ and $$\lvert\cos(\beta C_0/2)\rvert<1$$. If $$n_1\big(L\ln(n_2+1)+\ln3\big)\le(1-\delta)\,n_2\ln(1/\mu)$$ for some $$\delta>0$$, the right-hand side of (15) is at most $$\exp(-\delta\,n_2n_3\ln(1/\mu))$$. For $$n_1=o(n/\ln n)$$ with $$n_2,n_3$$ proportional to $$n$$ this gives $$e^{-cn^2}$$ for large $$n$$.

(c) Let $$L\ge1$$ and $$p_0=\lceil n_2/2\rceil$$. The right-hand side of (15) is at least $$2^{-n_1}\mu^{n_2n_3}(1+2B_L(p_0))^{n_1n_3}$$, and at proportions $$(n/4,n/2,n/4)$$ it exceeds $$1$$ for large $$n$$.
</div>

The weights of the interval counts $$k=1,\dots,L$$ are unimodular but not summable, so the bound has to control them without using cancellation. The device is to expand the product over $$a\in Y_1$$ in (11) as a sum over subsets, so that each term depends on the neighbourhood of the middle layer only through a bounded number of constraints.

<div class="proof" markdown="1">
*Proof.* Fix $$P_{12}$$ and write $$P_a\subseteq Y_2$$ for the upper neighbourhood of $$a\in Y_1$$, $$p_a=\lvert P_a\rvert$$. In (11) the columns $$c\in Y_3$$ are independent given $$P_{12}$$, so $$Z=e^{i\beta bn}\sum_{P_{12}}z^{\lvert P_{12}\rvert}W(P_{12})^{n_3}$$ with

$$
W(P_{12})=\sum_{\omega\subseteq Y_2}z^{\lvert \omega\rvert}\prod_{a\in Y_1}f(\lvert P_a\cap \omega\rvert).
$$

Here $$\omega$$ is the set of middle elements below a given top element. Write $$f=1+g$$, where $$g(k)=0$$ for $$k>L$$ and $$\lvert g\rvert\le2$$, and expand $$\prod_a(1+g)=\sum_{I\subseteq Y_1}\prod_{a\in I}g(\lvert P_a\cap \omega\rvert)$$. For fixed $$I$$ the product depends on $$\omega$$ only through $$\omega\cap P_I$$, with $$P_I=\bigcup_{a\in I}P_a$$. Summing over the free part of $$\omega$$ outside $$P_I$$ gives $$(1+z)^{n_2-\lvert P_I\rvert}$$. The sum over $$\omega'=\omega\cap P_I$$ has non-zero terms only if $$\lvert P_a\cap \omega'\rvert\le L$$ for all $$a\in I$$. The map $$\omega'\mapsto(\omega'\cap P_a)_{a\in I}$$ is injective, so at most $$\prod_{a\in I}B_L(p_a)$$ terms survive, each of modulus at most $$2^{\lvert I\rvert}$$. With $$\lvert1+z\rvert^{n_2-\lvert P_I\rvert}\le(2\mu)^{n_2}$$, because $$2\mu\ge1$$, we get

$$
\lvert W(P_{12})\rvert\le(2\mu)^{n_2}\sum_I\prod_{a\in I}2B_L(p_a)=(2\mu)^{n_2}\prod_{a\in Y_1}\big(1+2B_L(p_a)\big).
$$

Dividing by $$2^{n_2}$$, raising to the power $$n_3$$ and averaging over uniform $$P_{12}$$, for which the row sums $$p_a$$ are independent and binomial with parameters $$n_2$$ and $$1/2$$, gives (15). For (a), $$B_0=1$$.

For (b), $$B_L(p)\le(p+1)^L\le(n_2+1)^L$$, so $$1+2B_L(p)\le3(n_2+1)^L$$ and the right-hand side of (15) is at most $$\exp[\,n_3(n_2\ln\mu+n_1(L\ln(n_2+1)+\ln3))\,]$$, which is at most $$\exp(-\delta n_2n_3\ln(1/\mu))$$ under the stated condition.

For (c), $$B_L$$ is non-decreasing and $$\Pr[\mathrm{Bin}(n_2,\tfrac12)\ge p_0]\ge\tfrac12$$, so $$\mathbb E[(1+2B_L(p))^{n_3}]\ge\tfrac12(1+2B_L(p_0))^{n_3}$$, which gives the stated lower bound. Since $$\mu\ge\tfrac12$$, its logarithm is at least $$n_3\big[n_1\ln(1+2B_L(p_0))-n_2\ln2\big]-n_1\ln2$$. At proportions $$(n/4,n/2,n/4)$$ we have $$p_0=n/4$$ and $$B_L(p_0)\ge\binom{n/4}{L}\ge(n/4L)^L$$, so this is at least $$\tfrac{n^2}{16}\big[L\ln\tfrac n{4L}-2\ln2\big]-\tfrac n4\ln2$$, which tends to $$+\infty$$. $$\square$$
</div>

Part (b) states what the bound does give and part (c) what it cannot give: the exponent of the right-hand side of (15) carries a term $$n_1n_3L\ln n_2$$ against the term $$n_2n_3\ln(1/\mu)$$ that produces the decay, which is the logarithmic gap.

Evaluating (15) and the exact sum for $$\lvert\cos\rvert=0.4$$ and $$0.8$$ on layers $$(1,10,4)$$, $$(2,12,4)$$, $$(2,16,6)$$, $$(3,12,3)$$ and $$(4,8,4)$$, the bound holds in every case. For the link action it is non-trivial, for example $$7.4\times10^{-11}$$ against the exact $$5.0\times10^{-14}$$ on $$(1,10,4)$$ and $$1.0\times10^{-2}$$ against $$1.9\times10^{-8}$$ on $$(4,8,4)$$ at $$\lvert\cos\rvert=0.4$$. For the Benincasa–Dowker coefficients it is larger than $$1$$ in all the cases tried except $$(1,10,4)$$ at $$\lvert\cos\rvert=0.4$$, where it is $$1.5\times10^{-4}$$ against $$2.4\times10^{-9}$$.

The exact value of $$\lvert Z\rvert/\lvert\mathcal K\rvert$$ at $$(m,2m,m)$$ with $$\lvert\cos(\beta C_0/2)\rvert=0.4$$ (so $$\beta C_0=2\arccos0.4$$) is:

| $$n=4m$$ | link action | Benincasa–Dowker, $$d=4$$ | $$0.4^{n^2/4}$$ |
|---|---|---|---|
| 4 | $$2.59\times10^{-1}$$ | $$1.90\times10^{-1}$$ | $$2.6\times10^{-2}$$ |
| 8 | $$9.54\times10^{-3}$$ | $$5.51\times10^{-4}$$ | $$4.3\times10^{-7}$$ |
| 12 | $$4.16\times10^{-5}$$ | $$6.48\times10^{-6}$$ | $$4.7\times10^{-15}$$ |
| 16 | $$1.86\times10^{-8}$$ | $$2.94\times10^{-8}$$ | $$3.4\times10^{-26}$$ |

The last column is the value (10) would give if each of the $$n^2/4$$ relation slots $$n_2(n_1+n_3)$$ contributed a factor $$\lvert\cos\rvert$$. All entries are suppressed, much more weakly than that. The data stop at $$n=16$$ and do not decide the asymptotic form.

### 6.5 A trivial link phase

A trivial link phase removes the suppression whatever the other weights are.

<div class="proposition" markdown="1">
**Proposition 11** (Trivial link phase). If $$\beta C_0\in2\pi\mathbb Z$$, then for all $$n_1,n_2,n_3$$

$$
\frac{\lvert Z\rvert}{\lvert\mathcal K\rvert}\ge1-2\,n_1n_3\,\Pr\big[\mathrm{Bin}(n_2,\tfrac14)\le L\big].
$$

For fixed $$n_1,n_3$$ the right-hand side tends to $$1$$ as $$n_2\to\infty$$, and it also tends to $$1$$ at proportions $$(n/4,n/2,n/4)$$.
</div>

<div class="proof" markdown="1">
*Proof.* With $$z=1$$, (11) reduces to $$e^{i\beta bn}\prod_{a,c}f(k_{ac})$$, and the product equals $$1$$ unless some $$k_{ac}\le L$$. Under the uniform measure on $$\mathcal K$$ each $$k_{ac}$$ is binomial with parameters $$n_2$$ and $$1/4$$. Then $$\lvert\mathbb E\prod f-1\rvert\le2\Pr[\exists(a,c):k_{ac}\le L]\le2n_1n_3\Pr[\mathrm{Bin}(n_2,\tfrac14)\le L]$$, and a Chernoff bound makes the probability $$e^{-cn_2}$$. In both limits $$n_1n_3$$ grows at most polynomially in $$n_2$$. $$\square$$
</div>

For two layers the statement is exact (Theorem 8). For three layers the ratio is bounded below by a quantity that tends to $$1$$ only as $$n_2$$ grows. With $$(C_0,C_1,C_2,C_3)=(0,-9,16,-8)$$, $$b=1$$, $$\beta=1$$ and layers $$(2,n_2,2)$$, the exact value of $$\lvert Z\rvert/\lvert\mathcal K\rvert$$ is $$0.097,0.111,0.881,0.9999$$ for $$n_2=8,16,32,64$$, and the bound of the proposition is $$0.80$$ at $$n_2=32$$ and $$0.9999$$ at $$n_2=64$$ (it is negative at $$8$$ and $$16$$). The weights $$(0,-9,16,-8)$$ serve only to exhibit the proposition; no continuum property is claimed for them.

### 6.6 Relation to earlier results on layered orders

Theorem 9 shows that at fixed outer layers the weights $$C_1,\dots,C_L$$ do not enter the exponential rate bound, which is in line with the finding in [[LC]](#ref-lc) for bilayers and in [[CCS1]](#ref-ccs1) [[CCS2]](#ref-ccs2) for typical layered orders that the link term governs the leading suppression. The statements for typical orders evaluate the action on a typical member of the class, whereas the sums above weight every order of the class with its phase. The transfer-matrix bounds do not establish the reduction to the link action at Kleitman–Rothschild proportions, and the prefactor that Theorem 9 leaves free is visible in the exact data above. The result of [[MSS]](#ref-mss), that the link action suppresses the Kleitman–Rothschild orders over a range of parameters, concerns a different quantity from the full phase sum treated here.

## 7. Smeared actions and the variance of the action

A path sum with a single scale also depends on how much the action fluctuates among manifold-like orders. Write $$F=bN-\sum_kC_kN_k$$ for the raw functional in (9), where $$N$$ is the number of elements, so that in $$d=4$$ the action is a constant times $$\rho^{-1/2}F$$. For a sprinkling of density $$\rho$$ and a nonlocality density $$\rho_k=\epsilon\rho$$ with $$0<\epsilon\le1$$, the smeared action of [[BD]](#ref-bd) replaces the weights $$C_k$$ by their binomial averages $$C^{(\epsilon)}(m)=\sum_{k}C_k\binom mk\epsilon^k(1-\epsilon)^{m-k}$$ over the number $$m$$ of elements in the interval:

$$
S_\epsilon=\epsilon\,b\,N-\epsilon^2\sum_{x\prec y}C^{(\epsilon)}(m_{xy}),\tag{16}
$$

up to the same prefactor as the unsmeared action; $$m_{xy}$$ is the number of elements strictly between $$x$$ and $$y$$.

<div class="proposition" markdown="1">
**Proposition 12** (Smearing is thinning, and damps the variance). Let $$\eta$$ be a Poisson process of density $$\rho$$ on a bounded region, $$\eta'$$ an independent $$\epsilon$$-thinning of $$\eta$$ (each point is kept independently with probability $$\epsilon$$), so that $$\eta'$$ is Poisson of density $$\epsilon\rho$$, and let $$F$$ be the raw functional above. Then (i) $$S_\epsilon(\eta)=\mathbb E[F(\eta')\mid\eta]$$, and (ii) $$\mathrm{Var}\,S_\epsilon\le\epsilon\,\mathrm{Var}\,F(\eta')$$.
</div>

The proof of (i) is direct. The proof of (ii) uses the chaos expansion of Last and Penrose [[LP]](#ref-lp), $$\mathrm{Var}\,G=\sum_{k\ge1}\frac1{k!}\lVert\mathbb E\,D^kG\rVert^2$$, where $$D_zG(\eta)=G(\eta+\delta_z)-G(\eta)$$ is the add-one-point difference and the norm is that of $$L^2$$ of the $$k$$-fold product of the intensity measure; it holds for every square-integrable functional.

<div class="proof" markdown="1">
*Proof.* (i) A pair of $$\eta$$ survives the thinning with probability $$\epsilon^2$$, and given that, the number of surviving points of its interval is binomial with parameters $$m_{xy}$$ and $$\epsilon$$, so the expected contribution to $$-\sum C$$ of the pair is $$-\epsilon^2C^{(\epsilon)}(m_{xy})$$; the number of points contributes $$\epsilon bN$$.

(ii) A point $$z$$ added to $$\eta$$ survives the thinning with probability $$\epsilon$$, so $$D_zG=\epsilon\,\mathbb E[D_zF(\eta')\mid\eta]$$ for $$G=\mathbb E[F(\eta')\mid\eta]$$. Iterating, with independent survival indicators for the added points, $$\mathbb E\,D^kG(z_1,\dots,z_k)=\epsilon^k\,T_kF(z_1,\dots,z_k)$$, where $$T_kF=\mathbb E\,D^kF(\eta')$$ is the corresponding quantity for the process $$\eta'$$ of intensity measure $$\epsilon\rho\,dx$$. Then $$\lVert\epsilon^kT_kF\rVert^2_{L^2((\rho dx)^k)}=\epsilon^k\lVert T_kF\rVert^2_{L^2((\epsilon\rho\,dx)^k)}$$, and summing over $$k\ge1$$ with $$\epsilon^k\le\epsilon$$ gives $$\mathrm{Var}\,G\le\epsilon\,\mathrm{Var}\,F(\eta')$$. Square integrability holds because $$\lvert F\rvert$$ is bounded by a polynomial in the number of points of the bounded region. $$\square$$
</div>

So at fixed nonlocality density $$\rho_k$$ the variance of the smeared action is at most $$\rho_k/\rho$$ times the variance of the unsmeared action at density $$\rho_k$$, for every finite-depth kernel.

For the four-dimensional Benincasa–Dowker functional $$F=N-N_0+9N_1-16N_2+8N_3$$ on a causal interval of unit height with mean point number $$\bar N=250,500,1000,2000$$, the variance of the raw functional is

| $$\bar N$$ | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| $$\mathrm{Var}\,F$$ | $$1.6\times10^{5}$$ | $$6.1\times10^{5}$$ | $$2.4\times10^{6}$$ | $$8.1\times10^{6}$$ |
| statistical error | 6% | 8% | 12% | 18% |

so $$\mathrm{Var}\,F$$ grows like $$\bar N^{1.9}$$ over this range. With $$S\propto\rho^{-1/2}F$$ the variance of the action grows like $$\mathrm{Var}\,F/\bar N$$, a fitted power $$\bar N^{0.89}$$; the asymptotic exponent is not determined by these data. For the thinning identity (i), on one realisation of $$300$$ points with $$\epsilon=0.3$$ the formula gives $$44.05$$ and the average over $$3000$$ independent thinnings $$43.11\pm2.18$$. For the inequality (ii), at $$\bar N=1000$$ and $$\epsilon=1/4$$ the variance of $$S_\epsilon$$ (in units of the raw functional) is about $$6\times10^2$$ against $$\epsilon\,\mathrm{Var}\,F(\eta')\approx4\times10^4$$, so the inequality holds with a factor of about $$60$$ to spare.

## 8. What the results do not exclude

* Theorem 3 excludes exact consistency on finitely many sprinkled elements. Rules whose error tends to zero as $$\rho\to\infty$$ are not covered, among them the interval-count d'Alembertian [[BD]](#ref-bd) with a nonlocality scale that is a free input (see the [spectral representation of the causal-set d'Alembertian](/quantum-gravity/causal-box-spectrum/) and the [heavy-tailed horizon kernels](/quantum-gravity/nonlocal-weights/) for related constraints on such operators). Proposition 4 states what such a rule must achieve on linear data: $$e_1(\theta)=o(\theta)$$.
* Theorem 1 holds in $$1+1$$ dimensions for fields defined on all pairs of ranks. It does not apply to a theory whose only objects are the sprinkled elements, since no rectangle lies in the sprinkling.
* Proposition 5 covers separate statistics on orders with a symmetry of the stated kind. Laws that use the number of common predecessors, the contents of intervals, or the links between $$x$$ and $$y$$ are not covered, nor are laws that are not asked to return the order they are given, nor the behaviour of a separate-statistics law on an actual sprinkling beyond the computations of section 4.
* Proposition 7 concerns a one-dimensional history space. Commuting normal transition operators on $$\mathbb C^m$$ with $$m\ge2$$, and non-commutative models, are not excluded.
* Theorems 9 and 10 concern finite-depth interval actions on three-layer classes with the complete relation $$Y_1\prec Y_3$$. Infinite-depth kernels, actions with an independent nonlocality scale, and sums over layer sizes with unrestricted layer assignment are not covered.

Four questions are open. For $$L\ge1$$ and a non-trivial link phase, the suppression at Kleitman–Rothschild proportions may be of order $$n^2$$, of lower order, or absent; the uniform bound and the exact data up to $$n=16$$ do not decide it. Whether a refinement of (15) restricted to typical $$P_{12}$$ gives $$e^{-cn^2}$$ for $$\lvert\cos(\beta C_0/2)\rvert$$ in some range, and whether the rate at these proportions equals the link rate, is not known. Whether the rate in (14) is attained for a given action is known only from the exact sums at small sizes. The asymptotic growth exponent of $$\mathrm{Var}\,S$$ for the finite-depth family, the existence of a non-commutative sequential-growth model with a decoherence functional of rank greater than one that satisfies the causality conditions, and the existence of an order-only completion rule with $$e_1(\theta)=o(\theta)$$ on linear data are also open.

## References

1. <a id="ref-sorkin"></a>Sorkin, “Does locality fail at intermediate length-scales?,” [arXiv:gr-qc/0703099](https://arxiv.org/abs/gr-qc/0703099).
2. <a id="ref-bd"></a>Benincasa, Dowker, “The scalar curvature of a causal set,” Phys. Rev. Lett. 104, 181301 (2010), [arXiv:1001.2725](https://arxiv.org/abs/1001.2725).
3. <a id="ref-ass"></a>Aslanbeigi, Saravani, Sorkin, “Generalized causal set d'Alembertians,” JHEP 06 (2014) 024, [arXiv:1403.1622](https://arxiv.org/abs/1403.1622).
4. <a id="ref-johnston"></a>Johnston, “Feynman propagator for a free scalar field on a causal set,” Phys. Rev. Lett. 103, 180401 (2009), [arXiv:0909.0944](https://arxiv.org/abs/0909.0944).
5. <a id="ref-zab"></a>Zapata-Carratalá, Arsiwalla, Beynon, “Heaps of Fish: arrays, generalized associativity and heapoids,” [arXiv:2205.05456](https://arxiv.org/abs/2205.05456).
6. <a id="ref-krud"></a>Korzyuk, Rudzko, “Curvilinear parallelogram identity and mean-value property for a semilinear hyperbolic equation of second-order,” Eurasian Math. J. 15 (2024) 61, [arXiv:2204.09408](https://arxiv.org/abs/2204.09408).
7. <a id="ref-kr75"></a>Kleitman, Rothschild, “Asymptotic enumeration of partial orders on a finite set,” Trans. Amer. Math. Soc. 205 (1975) 205–220.
8. <a id="ref-lc"></a>Loomis, Carlip, “Suppression of non-manifold-like sets in the causal set path integral,” [arXiv:1709.00064](https://arxiv.org/abs/1709.00064).
9. <a id="ref-ccs1"></a>Carlip, Carlip, Surya, “Path integral suppression of badly behaved causal sets,” [arXiv:2209.00327](https://arxiv.org/abs/2209.00327).
10. <a id="ref-ccs2"></a>Carlip, Carlip, Surya, “The Einstein-Hilbert action for entropically dominant causal sets,” [arXiv:2311.18238](https://arxiv.org/abs/2311.18238).
11. <a id="ref-mss"></a>Mathur, Singh, Surya, “Entropy and the link action in the causal set path-sum,” [arXiv:2009.07623](https://arxiv.org/abs/2009.07623).
12. <a id="ref-lp"></a>Last, Penrose, “Poisson process Fock space representation, chaos expansion and covariance inequalities,” [arXiv:0909.3205](https://arxiv.org/abs/0909.3205).
13. <a id="ref-rs"></a>Rideout, Sorkin, “A classical sequential growth dynamics for causal sets,” Phys. Rev. D 61, 024002 (2000), [arXiv:gr-qc/9904062](https://arxiv.org/abs/gr-qc/9904062).
14. <a id="ref-djs"></a>Dowker, Johnston, Surya, “On extending the quantum measure,” J. Phys. A 43, 505305 (2010), [arXiv:1007.2725](https://arxiv.org/abs/1007.2725).
15. <a id="ref-sz"></a>Surya, Zalel, “A criterion for covariance in complex sequential growth models,” [arXiv:2003.11311](https://arxiv.org/abs/2003.11311).
