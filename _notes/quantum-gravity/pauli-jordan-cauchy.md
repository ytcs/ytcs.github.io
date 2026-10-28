---
title: The Cauchy law of the sprinkled Pauli–Jordan operator in 1+1 dimensions
date: 2026-10-28
order: 3
description: For a causal set sprinkled into a two-dimensional causal diamond, the massless Pauli–Jordan matrix is exactly a sum of two copies of one fixed matrix, one per null direction, related by a uniformly random permutation. Its eigenvalue distribution tends to a Cauchy law of width one half, and the volume law of the untruncated entanglement entropy is reproduced to within ten per cent by one null chain thinned at random.
---

On a causal set the Pauli–Jordan matrix $$i\Delta$$ of a free scalar field, the discrete counterpart of the commutator $$[\phi(x),\phi(y)]$$, determines the Sorkin–Johnston (SJ) vacuum and Sorkin's spacetime entanglement entropy. For $$N$$ points sprinkled into a causal diamond in 1+1 dimensions, a few of its eigenvalues are of order $$N$$ and match those of the continuum operator, while the great majority are of order one, in the range where Sorkin and Yazdi [[SY]](#ref-sy) found the discrete spectrum to depart from the continuum one. We ask what the distribution of these eigenvalues is and where it comes from. For the massless field and a uniform sprinkling of a causal diamond, $$i\Delta$$ is exactly the sum of two copies of one deterministic $$N\times N$$ matrix, one for each null direction, conjugated relative to each other by a uniformly random permutation (Proposition 1). All the randomness of the sprinkling sits in that permutation. Each copy has exactly the quantiles of a Cauchy distribution of width $$\tfrac14$$ as its eigenvalues (Lemma 1), the two copies are asymptotically free (Proposition 2), and the fraction of eigenvalues of $$i\Delta$$ below any $$x$$ converges almost surely to $$\tfrac12+\tfrac1\pi\arctan 2x$$, the distribution function of a Cauchy law of width $$\tfrac12$$ (Theorem 1). These statements are proved; the proof is specific to the massless field, to 1+1 dimensions and to regions that are causal diamonds.

Two consequences follow. In the continuum the two null sectors act on nearly orthogonal subspaces; on the causal set they act on the same space in free position (section 6). And no split of $$i\Delta$$ into a "signal" and a "noise" that are both functions of $$i\Delta$$ can be asymptotically free, so free deconvolution of the spectral distribution has nothing to remove (section 7). A third observation is numerical and not proved: the volume law of the untruncated entanglement entropy of a sub-diamond is reproduced to within 10% by the entropy of a single null chain restricted to a randomly thinned interval, while the same chain without thinning shows no volume law (section 8).

The discrete retarded propagator and Pauli–Jordan function on causal sets are due to Johnston [[Johnston 2009]](#ref-johnston09) [[Johnston 2010]](#ref-johnston10), and the SJ vacuum is constructed from $$i\Delta$$ as in [[Johnston 2010]](#ref-johnston10) and [[AAS]](#ref-aas). Sorkin's entropy formula [[Sorkin 2012]](#ref-sorkin12) was evaluated in 1+1 dimensions by Saravani, Sorkin and Yazdi [[SSY]](#ref-ssy), and on causal sets by Sorkin and Yazdi [[SY]](#ref-sy), who found the volume law, traced it to the many small eigenvalues of $$i\Delta$$, and removed it by truncating the spectrum below $$\sqrt N/4\pi$$. Jones and Yazdi [[JY]](#ref-jy) conjecture the asymptotic spectral density of the continuum operator. On the probability side we use the asymptotic freeness of random permutations, due to Nica [[Nica]](#ref-nica) and extended by Kammoun and Maïda [[KM]](#ref-km), and the theory of free convolution of unbounded measures of Bercovici and Voiculescu [[BV]](#ref-bv). The splitting of the 1+1 massless commutator into left- and right-moving parts is standard in the continuum, and the eigenvalues of the sign matrix are a classical computation. What is added here is the exact discrete identity, the Cauchy limit including the regime of small eigenvalues, the statement about free deconvolution, and the comparison of the volume law with a thinned chain.

## 1. The Pauli–Jordan matrix of a sprinkled diamond

We use null coordinates $$u=t-x$$ and $$v=t+x$$ on 1+1 Minkowski space. A point $$p$$ precedes $$q$$, written $$p\prec q$$, when $$u_p<u_q$$ and $$v_p<v_q$$. After a boost and a rescaling, a causal diamond is the unit square $$D=[0,1]^2$$ in these coordinates. A sprinkling of $$D$$ with $$N$$ points is a set of $$N$$ independent points uniformly distributed in $$D$$; a Poisson sprinkling conditioned on its number of points is of this form.

Label the points $$1,\dots,N$$. The causal matrix is $$C_{ij}=1$$ if $$i\prec j$$ and $$0$$ otherwise. For the massless field in 1+1 dimensions the continuum retarded Green function is $$\tfrac12$$ inside the future light cone, and its causal-set counterpart is $$\tfrac12C$$ [[Johnston 2009]](#ref-johnston09) [[SY]](#ref-sy). The Pauli–Jordan matrix is the difference of the retarded and advanced matrices,

$$
\Delta=\tfrac12\big(C-C^{\mathsf T}\big),
$$

a real antisymmetric matrix, so $$i\Delta$$ is Hermitian and its eigenvalues come in pairs $$\pm\lambda$$. With the opposite sign convention $$\Delta$$ changes sign and the spectrum is unchanged. Because the Green function is dimensionless in two dimensions, no factor of the sprinkling density appears: the eigenvalues need no rescaling with $$N$$.

We describe the spectrum by its empirical distribution $$\mu_N$$, the probability measure with mass $$1/N$$ at each eigenvalue (counted with multiplicity), and its distribution function $$F_N(x)=\mu_N((-\infty,x])$$, the fraction of eigenvalues at most $$x$$. The limit will be the Cauchy law $$\mathcal C_\gamma$$ of width $$\gamma$$, with density $$\gamma/\pi(x^2+\gamma^2)$$ and distribution function $$\tfrac12+\tfrac1\pi\arctan(x/\gamma)$$. Its tail is heavy: the fraction of mass above $$x$$ falls only as $$\gamma/\pi x$$.

## 2. One sign matrix per null direction

The causal order of two points is decided by two comparisons, one per null coordinate, and the antisymmetric combination in $$\Delta$$ turns the product of step functions into a sum. For real $$\alpha,\beta\neq0$$,

$$
\theta(\alpha)\theta(\beta)-\theta(-\alpha)\theta(-\beta)=\tfrac12\big(\operatorname{sgn}\alpha+\operatorname{sgn}\beta\big), \tag{1}
$$

as one checks on the four sign patterns: both sides equal $$1$$ if $$\alpha,\beta>0$$, $$-1$$ if $$\alpha,\beta<0$$, and $$0$$ otherwise. For a sequence $$y_1,\dots,y_N$$ of distinct reals, let $$S_y$$ be its sign matrix, $$(S_y)_{ij}=\operatorname{sgn}(y_j-y_i)$$, and let $$S$$ be the sign matrix of the sequence $$1,2,\dots,N$$.

<div class="proposition" markdown="1">
**Proposition 1** (light-cone decomposition). For any $$N$$ points of 1+1 Minkowski space with distinct $$u$$-coordinates and distinct $$v$$-coordinates,

$$
i\Delta=X_u+X_v,\qquad X_u=\tfrac i4S_u,\quad X_v=\tfrac i4S_v. \tag{2}
$$

Label the points in increasing order of $$u$$ and let $$\pi(i)$$ be the rank of $$v_i$$ among the $$v$$-coordinates. Then $$S_u=S$$ and $$S_v=PSP^{\mathsf T}$$, where $$P$$ is the permutation matrix with $$P_{i,\pi(i)}=1$$. For a sprinkling of a causal diamond, $$\pi$$ is uniformly distributed on the symmetric group $$\mathfrak S_N$$.
</div>

<div class="proof" markdown="1">
*Proof.* By definition $$(C-C^{\mathsf T})_{ij}=\theta(u_j-u_i)\theta(v_j-v_i)-\theta(u_i-u_j)\theta(v_i-v_j)$$, and (1) with $$\alpha=u_j-u_i$$, $$\beta=v_j-v_i$$ gives $$\Delta=\tfrac14(S_u+S_v)$$. In the $$u$$-ordered labelling $$\operatorname{sgn}(u_j-u_i)=\operatorname{sgn}(j-i)$$, and $$(PSP^{\mathsf T})_{ij}=S_{\pi(i)\pi(j)}=\operatorname{sgn}(\pi(j)-\pi(i))=\operatorname{sgn}(v_j-v_i)$$. In $$D=[0,1]^2$$ the coordinates $$u_i$$ and $$v_i$$ are independent, so the $$v$$-values listed in $$u$$-order are $$N$$ independent uniform variables, independent of the $$u$$-values, and all $$N!$$ orderings of them are equally likely.
</div>

The identity (2) holds for every finite subset of 1+1 Minkowski space in general position; only the uniformity of $$\pi$$ uses the diamond. The matrix $$X=\tfrac i4S$$ is deterministic. Each null direction contributes a copy of it, written in the order of that null coordinate, and the sprinkling enters only through the relative permutation $$\pi$$ of the two orders.

Two extreme cases show what $$\pi$$ controls. If all points are pairwise timelike, the causal set is a chain, $$\pi$$ is the identity and $$i\Delta=2X$$. If all points are pairwise spacelike, $$\pi$$ reverses the order, $$S_v=-S_u$$ and $$i\Delta=0$$. A sprinkled diamond lies in between: a uniform $$\pi$$ makes about half of all pairs related.

## 3. The sign matrix is a Cayley transform

The spectrum of $$X$$ can be computed in closed form, and the computation shows why a Cauchy law appears. Let $$e_1,\dots,e_N$$ be the standard basis and let $$V$$ be the signed cyclic shift

$$
Ve_r=e_{r-1}\quad(2\le r\le N),\qquad Ve_1=-e_N .
$$

It is a permutation matrix of an $$N$$-cycle with one entry replaced by $$-1$$. In particular it is orthogonal, and $$V^N=-1$$ because each basis vector passes once around the cycle and crosses the sign once.

<div class="lemma" markdown="1">
**Lemma 1** (sign matrix). The matrix $$1-V$$ is invertible and

$$
S=(1+V)(1-V)^{-1}. \tag{3}
$$

The eigenvalues of $$V$$ are the $$N$$ roots $$e^{i\theta_k}$$ of $$z^N=-1$$, with $$\theta_k=(2k-1)\pi/N$$, $$k=1,\dots,N$$. Hence $$X=h(V)$$ with $$h(z)=\tfrac i4(1+z)/(1-z)$$, and the eigenvalues of $$X$$ are

$$
h(e^{i\theta_k})=-\tfrac14\cot\frac{(2k-1)\pi}{2N},\qquad k=1,\dots,N, \tag{4}
$$

which are exactly the quantiles of $$\mathcal C_{1/4}$$ at the levels $$(2k-1)/2N$$.
</div>

<div class="proof" markdown="1">
*Proof.* Since $$(1-V)\sum_{k=0}^{N-1}V^k=1-V^N=2$$, the inverse is $$(1-V)^{-1}=\tfrac12\sum_{k=0}^{N-1}V^k$$. Multiplying by $$1+V$$ and using $$1+V^N=0$$,

$$
(1+V)(1-V)^{-1}=\tfrac12\sum_{k=0}^{N-1}\big(V^k+V^{k+1}\big)=\sum_{k=1}^{N-1}V^k .
$$

For $$1\le k\le N-1$$, $$V^ke_r=e_{r-k}$$ if $$r>k$$ and $$V^ke_r=-e_{r-k+N}$$ if $$r\le k$$. Summing over $$k$$, column $$r$$ of $$\sum_kV^k$$ has $$+1$$ in every row above the diagonal, $$-1$$ in every row below it and $$0$$ on it, which is column $$r$$ of $$S$$. For the eigenvalues, if $$\omega^N=-1$$ the vector $$f=\sum_r\omega^re_r$$ satisfies $$Vf=\sum_{r\ge2}\omega^re_{r-1}-\omega e_N=\omega f$$, the last component because $$\omega^{N+1}=-\omega$$. The $$N$$ roots are distinct, so they are all the eigenvalues. On the unit circle, $$(1+e^{i\theta})/(1-e^{i\theta})=i\cot(\theta/2)$$, which gives (4). The Cauchy quantile at level $$p$$ is $$\gamma\tan(\pi(p-\tfrac12))=-\gamma\cot(\pi p)$$, and $$p=(2k-1)/2N$$, $$\gamma=\tfrac14$$ gives (4).
</div>

For $$N=3$$, for instance, $$S$$ has eigenvalues $$0,\pm i\sqrt3$$, so $$X$$ has $$0,\pm\sqrt3/4$$, and (4) gives $$\tfrac14\cot(\pi/6)=\sqrt3/4$$ and $$\tfrac14\cot(\pi/2)=0$$. In general the spectral distribution of one null piece is, deterministically, a discretisation of $$\mathcal C_{1/4}$$ at the midpoints of $$N$$ equal steps, so its distribution function is within $$1/2N$$ of that of $$\mathcal C_{1/4}$$ everywhere. For $$N=2000$$, a fraction $$0.705$$ of the eigenvalues of $$X$$ lie in $$[-\tfrac12,\tfrac12]$$, against $$\tfrac2\pi\arctan2=0.7048$$ for $$\mathcal C_{1/4}$$.

The Cauchy law enters through a standard fact: if $$\theta$$ is uniform on the circle, then $$h(e^{i\theta})=-\tfrac14\cot(\theta/2)$$ has law $$\mathcal C_{1/4}$$. The eigenvalues of $$V$$ are spread evenly on the circle, and $$h$$ maps that even spread onto Cauchy quantiles.

Combining Lemma 1 with Proposition 1,

$$
i\Delta=h(V_u)+h(V_v),\qquad V_u=V,\quad V_v=PVP^{\mathsf T}, \tag{5}
$$

with $$P$$ a uniformly random permutation matrix. The question is now the law of a sum of two Cayley transforms whose arguments are the same signed cycle in two random relative positions. If $$V_u$$ and $$V_v$$ behaved like free Haar unitaries, the answer would be the free convolution $$\mathcal C_{1/4}\boxplus\mathcal C_{1/4}$$, which we show in section 5 to be $$\mathcal C_{1/2}$$. The next section proves that they do.

The value $$\tfrac12$$ alone does not detect freeness. For the chain of section 2, $$i\Delta=2X$$ has limiting law $$\mathcal C_{1/2}$$ as well, since doubling a Cauchy variable doubles its width. What distinguishes a sprinkled diamond from a chain is the vanishing of mixed moments, which fails for the chain and is the content of Proposition 2.

## 4. The two null orders are asymptotically free

Freeness of two unitaries $$U_1,U_2$$ in a von Neumann algebra with a faithful tracial state $$\tau$$ is a statement about words. They are free Haar unitaries when $$\tau(w(U_1,U_2))=0$$ for every word $$w$$ that is not the identity in the free group on two letters, and $$\tau(1)=1$$. Asymptotic freeness of $$V_u$$ and $$V_v$$ therefore means that the normalised trace $$\operatorname{tr}=\frac1N\operatorname{Tr}$$ of every such word in $$V_u^{\pm1},V_v^{\pm1}$$ tends to zero.

The point is that $$V_u$$ and $$V_v$$ are signed permutation matrices, so the trace of a word is a signed count of fixed points of the corresponding word in permutations, and fixed points of words in random permutations are rare. The underlying permutation of $$V_u$$ is the $$N$$-cycle $$c:r\mapsto r-1\pmod N$$, and that of $$V_v$$ is $$\pi c\pi^{-1}$$. A product of signed permutation matrices is again one, and its diagonal entry at $$i$$ is $$\pm1$$ if $$i$$ is fixed by the underlying permutation and $$0$$ otherwise. Hence

$$
\lvert\operatorname{Tr}w(V_u,V_v)\rvert\le\operatorname{Fix}\,w(c,\pi c\pi^{-1}), \tag{6}
$$

where $$\operatorname{Fix}$$ is the number of fixed points. The difficulty is that $$c$$ is deterministic, so only one of the two permutations is random. We get around it by a conjugation: if $$\rho$$ is uniform and independent of $$\pi$$, then $$\rho$$ and $$\rho\pi$$ are independent and uniform, so $$\sigma_1=\rho c\rho^{-1}$$ and $$\sigma_2=(\rho\pi)c(\rho\pi)^{-1}$$ are independent uniformly random $$N$$-cycles. Conjugating both letters by the same $$\rho$$ does not change the number of fixed points of a word. It therefore suffices to count fixed points of words in two independent uniform $$N$$-cycles.

A word of length $$L$$ in $$\sigma_1^{\pm1},\sigma_2^{\pm1}$$ is reduced if no letter is immediately followed by its inverse.

<div class="lemma" markdown="1">
**Lemma 2** (fixed points of words). Let $$\sigma_1,\sigma_2$$ be independent uniformly random $$N$$-cycles and $$w$$ a reduced word of length $$L\ge1$$, with $$N>L$$. Then for every point $$x$$,

$$
\Pr\big(w(x)=x\big)\le\frac{L^2}{N-L},\qquad\text{hence}\qquad\mathbb E\operatorname{Fix}w\le\frac{NL^2}{N-L}. \tag{7}
$$

</div>

<div class="proof" markdown="1">
*Proof.* Write $$w=s_L\cdots s_1$$ with letters $$s_t\in\{\sigma_1^{\pm1},\sigma_2^{\pm1}\}$$ and follow the orbit $$x_0=x$$, $$x_j=s_j(x_{j-1})$$, revealing the cycles only where they are evaluated. At step $$j$$ the arrow from $$x_{j-1}$$ under $$s_j$$ has either been revealed before or not. Call step $$j$$ a coincidence if the arrow is new and $$x_j\in\{x_0,\dots,x_{j-1}\}$$.

First, if no coincidence occurs, then $$x_L\neq x_0$$. Indeed, by induction the points $$x_0,\dots,x_{j-1}$$ are then distinct and the revealed arrows are exactly the path $$x_0\to x_1\to\dots\to x_{j-1}$$. The only revealed arrow at $$x_{j-1}$$ leads back to $$x_{j-2}$$ and is read by the inverse of $$s_{j-1}$$, which $$s_j$$ is not because $$w$$ is reduced. So the arrow at step $$j$$ is new, and without a coincidence $$x_j$$ is a new point. In particular $$x_L\neq x_0$$.

Second, we bound the probability of a coincidence at step $$j$$, given everything revealed before it. Suppose $$s_j=\sigma_1$$ and $$m\le j-1$$ arrows of $$\sigma_1$$ are known. They form disjoint paths, since $$m<N$$. A uniform $$N$$-cycle conditioned on containing these paths is a uniform cyclic arrangement of the $$N-m$$ path blocks, so $$\sigma_1(x_{j-1})$$ is uniform over the $$N-m-1$$ starting points of blocks other than the block of $$x_{j-1}$$. The same holds for $$\sigma_1^{-1}$$, and the arrows of $$\sigma_2$$ are independent of those of $$\sigma_1$$. The probability that the new point lies among the $$j\le L$$ visited ones is therefore at most $$L/(N-L)$$. Summing over the $$L$$ steps gives the first bound in (7), and multiplying by $$N$$ the second.
</div>

The lemma is cruder than what is known. Nica [[Nica]](#ref-nica) for uniform permutations, and Kammoun and Maïda [[KM]](#ref-km) for conjugation-invariant permutations with few short cycles, which includes uniform $$N$$-cycles, prove that $$\mathbb E\operatorname{Fix}w$$ converges to the number of divisors of $$d$$ when $$w$$ is the $$d$$-th power of a word that is not itself a power, and that the counts of short cycles have a universal joint limit. A bound of order one is all that is needed here. In a Monte Carlo of two independent uniform cycles, with words of length 2 to 8 and $$N=10^2,10^3,10^4$$, the mean number of fixed points is between $$0.97$$ and $$1.05$$ for words that are not powers and between $$1.96$$ and $$2.07$$ for $$(\sigma_1\sigma_2)^2$$, in agreement with that limit.

<div class="proposition" markdown="1">
**Proposition 2** (asymptotic freeness of the null orders). Let $$U_1,U_2$$ be free Haar unitaries in a von Neumann algebra with faithful tracial state $$\tau$$. For every word $$w$$ in $$V_u^{\pm1},V_v^{\pm1}$$ whose cyclic reduction has length $$L$$, and every $$N\ge2L$$,

$$
\mathbb E\,\big\lvert\operatorname{tr}w(V_u,V_v)-\tau\big(w(U_1,U_2)\big)\big\rvert\le\frac{2L^2}{N}.
$$

In particular every mixed moment of $$(V_u,V_u^{*},V_v,V_v^{*})$$ converges in $$L^1$$, and hence in probability, to the corresponding moment of $$(U_1,U_1^{*},U_2,U_2^{*})$$.
</div>

<div class="proof" markdown="1">
*Proof.* The trace is invariant under cyclic permutation of the letters, and cancelling adjacent inverse letters does not change the matrix, so we may replace $$w$$ by its cyclic reduction. Three cases remain. If the reduction is empty, $$w(V_u,V_v)=1$$ and both traces equal $$1$$. If it is a single power $$V_u^a$$ or $$V_v^a$$ with $$0<\lvert a\rvert\le L<N$$, then $$\operatorname{tr}V_u^a=\operatorname{tr}V_v^a=\frac1N\sum_k e^{ia\theta_k}=0$$ by Lemma 1, and $$\tau(U_i^a)=0$$. Otherwise the reduction alternates between nonzero powers of $$V_u$$ and of $$V_v$$, its letter sequence in $$c$$ and $$\pi c\pi^{-1}$$ is a reduced word of length $$L$$, and $$\tau(w(U_1,U_2))=0$$. By (6), the conjugation above and (7), $$\mathbb E\lvert\operatorname{tr}w\rvert\le L^2/(N-L)\le2L^2/N$$.
</div>

No variance estimate is needed. The normalised trace is dominated by the non-negative variable $$\operatorname{Fix}/N$$, whose mean is $$O(1/N)$$, and this already gives convergence in $$L^1$$.

## 5. The Cauchy limit

<div class="theorem" markdown="1">
**Theorem 1** (Cauchy law of the 1+1 Pauli–Jordan operator). Let $$p_1,p_2,\dots$$ be independent uniform points of a causal diamond in 1+1 Minkowski space, and let $$F_N$$ be the eigenvalue distribution function of the massless Pauli–Jordan matrix $$i\Delta$$ of $$p_1,\dots,p_N$$. Then, almost surely,

$$
\sup_{x\in\mathbb R}\Big\lvert F_N(x)-\Big(\frac12+\frac1\pi\arctan2x\Big)\Big\rvert\longrightarrow0\qquad(N\to\infty).
$$

That is, the empirical spectral distribution of $$i\Delta$$ converges to $$\mathcal C_{1/2}$$.
</div>

The idea is to pass the asymptotic freeness of $$V_u,V_v$$ through the map $$h$$ in (5). The obstacle is that $$h$$ is unbounded near $$z=1$$, where the largest eigenvalues of $$X$$, those of order $$N$$, come from. We cut $$h$$ off near $$z=1$$. The cut-off changes only a small fraction of eigenvalues, which costs little in distribution, and for the bounded remainder moment methods apply.

The proof uses two facts about Hermitian matrices. The first is the rank inequality: if $$A$$ and $$B$$ are Hermitian $$N\times N$$ matrices with $$\operatorname{rank}(A-B)=r$$, their eigenvalue distribution functions differ by at most $$r/N$$ everywhere. This follows from the interlacing inequalities $$\lambda_{k+r}(A)\le\lambda_k(B)\le\lambda_{k-r}(A)$$ for eigenvalues in decreasing order. The second is Weyl's inequality: if $$\lVert A-B\rVert\le\varepsilon$$, each eigenvalue moves by at most $$\varepsilon$$.

The input from free probability is the following. For probability measures $$\mu,\nu$$ on $$\mathbb R$$, with bounded or unbounded support, the free convolution $$\mu\boxplus\nu$$ is the distribution of $$a+b$$ for free self-adjoint $$a,b$$ with distributions $$\mu,\nu$$, which in the unbounded case are operators affiliated with a von Neumann algebra. Bercovici and Voiculescu [[BV]](#ref-bv) proved that $$\boxplus$$ is continuous for weak convergence and that it linearises under the Voiculescu transform. If $$G_\mu(z)=\int d\mu(x)/(z-x)$$ is the Cauchy transform and $$\Phi_\mu=1/G_\mu$$, then $$\varphi_\mu(\zeta)=\Phi_\mu^{-1}(\zeta)-\zeta$$, defined on a region near infinity in the upper half-plane, satisfies $$\varphi_{\mu\boxplus\nu}=\varphi_\mu+\varphi_\nu$$ and determines $$\mu$$. For the Cauchy law, $$G_{\mathcal C_\gamma}(z)=1/(z+i\gamma)$$ for $$\operatorname{Im}z>0$$, so $$\Phi(z)=z+i\gamma$$ and $$\varphi_{\mathcal C_\gamma}(\zeta)=-i\gamma$$, a constant. Hence

$$
\mathcal C_{\gamma_1}\boxplus\mathcal C_{\gamma_2}=\mathcal C_{\gamma_1+\gamma_2},\qquad\text{in particular}\qquad\mathcal C_{1/4}\boxplus\mathcal C_{1/4}=\mathcal C_{1/2}. \tag{8}
$$

<div class="proof" markdown="1">
*Proof of Theorem 1.* Distances between distributions are measured by the Kolmogorov distance $$d_K$$, the supremum of the difference of distribution functions, and by the Lévy distance $$d_L\le d_K$$, which metrises weak convergence.

*Step 1: cut-off.* Fix $$\delta\in(0,\pi)$$. Let $$h_\delta$$ be the continuous real function on the circle equal to $$-\tfrac14\cot(\theta/2)$$ at $$e^{i\theta}$$ for $$\delta\le\lvert\theta\rvert\le\pi$$ and linear in $$\theta$$ on $$\lvert\theta\rvert<\delta$$, interpolating between $$\tfrac14\cot(\delta/2)$$ and $$-\tfrac14\cot(\delta/2)$$. The matrices $$h(V_u)$$ and $$h_\delta(V_u)$$ differ only on eigenvectors of $$V_u$$ with $$\lvert\theta_k\rvert<\delta$$. Since the $$\theta_k$$ are odd multiples of $$\pi/N$$, there are at most $$N\delta/\pi+1$$ of them, and the same holds for $$V_v$$, which has the same spectrum. With $$H_\delta=h_\delta(V_u)+h_\delta(V_v)$$, the rank inequality gives

$$
d_K\big(\mu_N,\ \mu(H_\delta)\big)\le\frac{2\delta}{\pi}+\frac2N,
$$

where $$\mu(H_\delta)$$ is the eigenvalue distribution of $$H_\delta$$.

*Step 2: the bounded part.* By Weierstrass's theorem there is, for each $$\varepsilon>0$$, a real trigonometric polynomial $$q$$ with $$\lvert h_\delta-q\rvert\le\varepsilon$$ on the circle. Then $$\lVert h_\delta(V)-q(V)\rVert\le\varepsilon$$ for every unitary $$V$$, both for the matrices and in the limit algebra. For each $$j$$, $$\operatorname{tr}\big(q(V_u)+q(V_v)\big)^j$$ is a finite linear combination of normalised traces of words, which by Proposition 2 converge in probability to the same combination for $$U_1,U_2$$. The estimate $$\lVert A^j-B^j\rVert\le j\max(\lVert A\rVert,\lVert B\rVert)^{j-1}\lVert A-B\rVert$$, with $$\varepsilon$$ arbitrary, transfers this to $$h_\delta$$: every moment of $$\mu(H_\delta)$$ converges in probability to the corresponding moment of $$h_\delta(U_1)+h_\delta(U_2)$$. The operators $$h_\delta(U_i)$$ belong to the free algebras generated by $$U_i$$, so they are free, and each has distribution $$\nu_\delta$$, the image of the uniform measure on the circle under $$h_\delta$$. The limit distribution is therefore $$\nu_\delta\boxplus\nu_\delta$$. All spectra lie in $$[-2\max\lvert h_\delta\rvert,2\max\lvert h_\delta\rvert]$$, a fixed interval, and on a fixed interval convergence of moments in probability implies $$d_L\big(\mu(H_\delta),\nu_\delta\boxplus\nu_\delta\big)\to0$$ in probability.

*Step 3: removing the cut-off in the limit.* The measures $$\nu_\delta$$ and $$\mathcal C_{1/4}$$ are images of the same uniform measure under maps that agree outside an arc of normalised length $$\delta/\pi$$, so they differ by at most $$\delta/\pi$$ in total variation. As $$\delta\to0$$, $$\nu_\delta\to\mathcal C_{1/4}$$ weakly. By continuity of $$\boxplus$$ and (8), $$\eta(\delta)=d_L\big(\nu_\delta\boxplus\nu_\delta,\mathcal C_{1/2}\big)\to0$$.

*Step 4: convergence in probability.* By the triangle inequality,

$$
d_L(\mu_N,\mathcal C_{1/2})\le\frac{2\delta}\pi+\frac2N+d_L\big(\mu(H_\delta),\nu_\delta\boxplus\nu_\delta\big)+\eta(\delta).
$$

Given any tolerance, choose $$\delta$$ so that $$2\delta/\pi+\eta(\delta)$$ is below half of it; by Step 2 the third term is below the other half with probability tending to one. Hence $$d_L(\mu_N,\mathcal C_{1/2})\to0$$ in probability. Since the limit distribution function $$F$$ is continuous, $$F_N(x)\to F(x)$$ in probability for every $$x$$.

*Step 5: almost sure and uniform convergence.* Moving one point $$p_i$$ changes only row and column $$i$$ of $$\Delta$$, a perturbation of rank at most $$2$$, so by the rank inequality it changes $$F_N(x)$$ by at most $$2/N$$. McDiarmid's bounded-differences inequality for functions of $$N$$ independent variables then gives

$$
\Pr\big(\lvert F_N(x)-\mathbb EF_N(x)\rvert\ge t\big)\le2e^{-Nt^2/2}.
$$

The right side is summable in $$N$$, so by the Borel–Cantelli lemma $$F_N(x)-\mathbb EF_N(x)\to0$$ almost surely. By Step 4 and bounded convergence, $$\mathbb EF_N(x)\to F(x)$$. Thus $$F_N(x)\to F(x)$$ almost surely, simultaneously for all rational $$x$$. Since the $$F_N$$ are non-decreasing and $$F$$ is continuous and bounded, the convergence is uniform in $$x$$.
</div>

All steps are elementary except Step 3, which uses the two cited results of [[BV]](#ref-bv). The proof gives no rate. We checked the theorem by Monte Carlo. The identity (2) and the Cayley form (3) hold to machine precision, and the eigenvalues of $$X$$ agree with (4) to $$2\times10^{-12}$$ at $$N=501$$. The Kolmogorov distance between the eigenvalue distribution of $$i\Delta$$ and $$\mathcal C_{1/2}$$ decreases with $$N$$:

| $$N$$ | 250 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|
| sprinklings | 40 | 40 | 20 | 8 | 3 |
| $$d_K(\mu_N,\mathcal C_{1/2})$$, mean | 0.0109 | 0.0053 | 0.0038 | 0.0021 | 0.0014 |
| $$d_K(\mu_N,\mathcal C_{1/4})$$, mean | 0.111 | 0.110 | 0.109 | 0.109 | 0.108 |

The last row shows that the sum is not distributed like one of its summands. Over this range $$\sqrt N\,d_K$$ falls from $$0.17$$ to $$0.09$$. Replacing the random permutation $$P$$ by a Haar-random unitary at $$N=1000$$ gives $$d_K=0.0026$$, comparable to the sprinkled value. A simple consequence to check by hand: the median of $$\lvert\lambda\rvert$$ is the quartile of $$\mathcal C_{1/2}$$, which is $$\tfrac12$$. In three sprinklings each at $$N=1000$$ and $$N=2000$$, the fraction of eigenvalues with $$\lvert\lambda\rvert\le\tfrac12$$ is between $$0.500$$ and $$0.502$$.

## 6. Null sectors in the continuum and on the causal set

In the continuum the two terms of (2) act on different functions. Let $$K$$ be the operator on $$L^2[0,1]$$ with kernel $$\operatorname{sgn}(u'-u)$$ and $$J$$ the projection onto constant functions, whose kernel is $$1$$. On the diamond with density $$\rho=N$$ the continuum Pauli–Jordan operator is, by the same identity (1),

$$
i\Delta_{\rm cont}=\frac{i\rho}4\big(K\otimes J+J\otimes K\big)
$$

on $$L^2([0,1]_u\times[0,1]_v)$$. The first term annihilates every function of $$v$$ alone with zero mean, and the second every such function of $$u$$; their product $$(K\otimes J)(J\otimes K)=KJ\otimes JK$$ has rank one. Left- and right-movers thus occupy subspaces that are orthogonal apart from the constants. On the causal set no such subspaces exist. For even $$N$$, by (4), $$X_u$$ and $$X_v$$ have no zero eigenvalue, both act on all of $$\mathbb C^N$$, and Proposition 2 says that their Cayley transforms are asymptotically in free position. In this sense discreteness replaces the nearly orthogonal position of the two null sectors by free position.

The continuum spectrum survives at the top of the discrete one. At $$N=4000$$, the six largest eigenvalues of $$i\Delta$$, averaged over four sprinklings, are $$855,\ 318,\ 217,\ 160,\ 129,\ 106$$. Discretising $$i\Delta_{\rm cont}$$ on its invariant subspace of sums of a function of $$u$$ and a function of $$v$$ gives $$858,\ 318,\ 217,\ 159,\ 128,\ 106$$. The spectrum of one chain, $$\tfrac14\cot(\pi/2N)\approx637$$ and so on, is different, so the continuum modes are not those of a single null piece. For $$1\ll\lambda\ll N$$ the discretised continuum operator has about $$N/2\pi\lambda$$ eigenvalues above $$\lambda$$, and $$\mathcal C_{1/2}$$ assigns the same count, $$\frac N\pi\arctan\frac1{2\lambda}=\frac N{2\pi\lambda}\big(1+O(\lambda^{-2})\big)$$. In one sprinkling with $$N=4000$$ the observed counts above $$\lambda=3,10,30,100$$ are $$210,63,20,6$$, against $$210,64,21,6$$ for $$\mathcal C_{1/2}$$ and $$211,63,21,6$$ for the discretised continuum operator. The Cauchy law thus carries the continuum's $$1/\lambda$$ tail. The modes above the truncation level $$\sqrt N/4\pi$$ of [[SY]](#ref-sy) number about $$2\sqrt N$$ of each sign ($$126$$ at $$N=4000$$), a fraction $$4/\sqrt N$$ of the total, and have no weight in the limit. The eigenvalue count shows no feature at that level, so a criterion that separates continuum-like modes from the others there has to use more than the eigenvalues, for instance the eigenvectors.

## 7. Spectral deconvolution has nothing to remove

A natural idea is to write the discrete operator as a continuum part plus discreteness noise, assume the two free, and remove the noise from the spectrum by free deconvolution. Three facts show that at the level of the spectral distribution there is nothing for such a deconvolution to remove.

The first is an elementary property of freeness. For self-adjoint $$a,b$$ write $$\operatorname{Var}a=\tau(a^2)-\tau(a)^2$$.

<div class="lemma" markdown="1">
**Lemma 3.** If $$a$$ and $$b$$ are free self-adjoint elements of a tracial algebra, then $$\tau\big([a,b]^*[a,b]\big)=2\operatorname{Var}a\operatorname{Var}b$$. In particular, if they commute and $$\tau$$ is faithful, one of them is a multiple of the identity.
</div>

<div class="proof" markdown="1">
*Proof.* The commutator does not change if $$a,b$$ are replaced by $$a^\circ=a-\tau(a)$$, $$b^\circ=b-\tau(b)$$. Expanding $$[a^\circ,b^\circ]^2$$, freeness gives $$\tau(a^\circ b^\circ a^\circ b^\circ)=\tau(b^\circ a^\circ b^\circ a^\circ)=0$$ and, with cyclicity, $$\tau(a^\circ b^{\circ2}a^\circ)=\tau(b^\circ a^{\circ2}b^\circ)=\tau(a^{\circ2})\tau(b^{\circ2})$$. So $$\tau([a,b]^2)=-2\operatorname{Var}a\operatorname{Var}b$$, and $$[a,b]^*=-[a,b]$$.
</div>

<div class="proposition" markdown="1">
**Proposition 3.** Let $$f,g$$ be bounded continuous real functions. If $$f(i\Delta)$$ and $$g(i\Delta)$$ are asymptotically free as $$N\to\infty$$, in the sense that their mixed moments converge in probability to those of a free pair, then $$f$$ or $$g$$ is constant.
</div>

<div class="proof" markdown="1">
*Proof.* Every mixed moment of $$f(i\Delta)$$ and $$g(i\Delta)$$ is $$\int f^jg^k\,d\mu_N$$ for some $$j,k$$, which by Theorem 1 converges to $$\int f^jg^k\,d\mathcal C_{1/2}$$. The limit pair therefore has the joint moments of the commuting multiplication operators $$f,g$$ on $$L^2(\mathcal C_{1/2})$$. In particular $$\operatorname{tr}\big([f(i\Delta),g(i\Delta)]^*[f(i\Delta),g(i\Delta)]\big)=0$$ for every $$N$$, so a pair $$(a,b)$$ with the limiting moments has $$\tau([a,b]^*[a,b])=0$$. If $$a$$ and $$b$$ are free, Lemma 3 gives $$\operatorname{Var}f\operatorname{Var}g=0$$ under $$\mathcal C_{1/2}$$, and since $$\mathcal C_{1/2}$$ has full support and $$f,g$$ are continuous, one of them is constant.
</div>

So a split of $$i\Delta$$ into two functions of itself, such as modes above and below a fixed eigenvalue, is never asymptotically free. The second fact concerns a split at an $$N$$-dependent level, such as the truncation at $$\sqrt N/4\pi$$. The part above the level contains a fraction $$4/\sqrt N$$ of the eigenvalues, so its spectral distribution tends to the point mass at zero. Free convolution with that point mass is the identity, so deconvolving it leaves the whole of $$\mathcal C_{1/2}$$: at the level of the spectral distribution the continuum part has zero weight.

The third fact concerns the free decomposition that does exist, $$X_u+X_v$$. Its two summands have the same law $$\mathcal C_{1/4}$$, so it is not a split into signal and noise. Moreover a free Cauchy summand acts as an ordinary Poisson smoothing. For any probability measure $$\mu$$, $$G_{\mu*\mathcal C_\gamma}(z)=\int d\mu(y)\int d\mathcal C_\gamma(s)\,(z-y-s)^{-1}=G_\mu(z+i\gamma)$$, so $$\Phi_{\mu*\mathcal C_\gamma}(z)=\Phi_\mu(z+i\gamma)$$ and $$\varphi_{\mu*\mathcal C_\gamma}=\varphi_\mu-i\gamma=\varphi_{\mu\boxplus\mathcal C_\gamma}$$. Therefore $$\mu\boxplus\mathcal C_\gamma=\mu*\mathcal C_\gamma$$, and removing a free Cauchy summand of width $$\gamma$$ amounts to the substitution $$G(z)\mapsto G(z-i\gamma)$$. Applied to $$\mathcal C_{1/2}$$ it returns $$\mathcal C_{1/2-\gamma}$$. The limit law is described by one number, its width, and there is no further structure in it for a deconvolution to resolve.

## 8. The volume law and a randomly thinned chain

This section is numerical. Sorkin's spacetime entropy of a region $$O$$ [[Sorkin 2012]](#ref-sorkin12) is computed from the restrictions $$W_O$$ and $$i\Delta_O$$ of the Wightman matrix and the Pauli–Jordan matrix to the points in $$O$$. One solves $$W_O\psi=\xi\,i\Delta_O\psi$$ for $$\psi$$ outside the kernel of $$i\Delta_O$$ and sets $$\mathcal S=\sum\xi\ln\lvert\xi\rvert$$. For the SJ state, $$W$$ is the positive spectral part of $$i\Delta$$ on the whole causal set. Without truncation, Sorkin and Yazdi [[SY]](#ref-sy) found $$\mathcal S\approx\kappa\,n$$ for a concentric sub-diamond containing $$n$$ points, with $$\kappa=0.46$$ when its side is $$\tfrac14$$ of the outer side and $$\kappa=0.32$$ when it is $$\tfrac12$$.

Proposition 1 gives an exact description of a concentric sub-diamond $$D_\ell=I_u\times I_v$$ of side $$\ell$$ as seen from one null order. The points with $$u\in I_u$$ form a contiguous block of about $$\ell N$$ consecutive elements of the $$u$$-ordered chain. Whether each of them also has $$v\in I_v$$ is decided independently, with probability $$\ell$$, because $$u$$ and $$v$$ are independent. From the $$u$$-chain, $$D_\ell$$ is thus an interval thinned at random to a fraction $$\ell$$.

We compare two entropies on the same points. $$\mathcal S_{\rm 2D}$$ uses $$i\Delta$$ and its SJ state. $$\mathcal S_{\rm chain}$$ uses only $$X_u$$, with the SJ state built from the positive part of $$X_u$$ on the whole causal set, restricted to $$D_\ell$$: it is the entropy of the thinned interval of one null chain. Fitting $$\mathcal S=\kappa n+{\rm const}$$ over sprinklings with $$N$$ from 1000 to 3000 or 4000 gives the following. Errors are bootstrap standard errors of the slope; the ratio is the mean over sprinklings of $$\mathcal S_{\rm chain}/\mathcal S_{\rm 2D}$$ with its standard error.

| $$\ell$$ | sprinklings | $$\kappa_{\rm 2D}$$ | $$\kappa_{\rm chain}$$ | $$\mathcal S_{\rm chain}/\mathcal S_{\rm 2D}$$ | $$\kappa$$ of [[SY]](#ref-sy) |
|---|---|---|---|---|---|
| 0.25 | 13 | 0.476 ± 0.015 | 0.432 ± 0.016 | 1.06 ± 0.02 | 0.46 |
| 0.3 | 16 | 0.422 ± 0.010 | 0.417 ± 0.013 | 1.02 ± 0.01 | |
| 0.5 | 29 | 0.312 ± 0.004 | 0.283 ± 0.005 | 0.94 ± 0.01 | 0.32 |

The two-dimensional slopes agree with [[SY]](#ref-sy). The single thinned chain reproduces them to within 10%. The agreement is approximate and not an identity: at $$\ell=0.5$$ the chain value is lower in every measure, and the difference of $$0.015\pm0.003$$ in $$\mathcal S/n$$ at $$N=2000$$ is significant. A diluted chain of fixed length $$M$$ gives the same values as the chain restricted to sprinkled points, for instance $$\mathcal S/n=0.416\pm0.005$$ ($$\ell=0.3$$) and $$0.289\pm0.004$$ ($$\ell=0.5$$) at $$M=2000$$.

Without thinning there is no volume term. For a central interval containing a fraction $$f=0.1$$ of a chain of $$M=1000,2000,4000$$ elements, $$\mathcal S=2.039,\ 2.270,\ 2.501$$. Each doubling of $$n$$ adds $$0.231$$, which is $$\tfrac13\ln2$$ to three digits, so $$\mathcal S$$ grows as $$\tfrac13\ln n$$ with no linear part. Taken together, the two computations indicate that the volume term comes from the random thinning that the second null order induces on the first, and that the coupling between the two sectors changes it by about 10% or less in these runs. We have no derivation of the volume coefficient from this picture. Sorkin and Yazdi [[SY]](#ref-sy) studied a related quantity, the entropy generated by randomly decimating a causal set or a harmonic chain, and found similar behaviour in both; here the state is the SJ state of one null chain and only the region is thinned.

## 9. Limitations and open questions

Theorem 1 is proved for the massless field on sprinklings of a causal diamond in 1+1 dimensions. The decomposition (2) holds for any region, but the proof uses the uniformity of $$\pi$$, which requires the region to be a product in null coordinates. Numerically the limit does not depend on this. At $$N=2000$$ the Kolmogorov distance to $$\mathcal C_{1/2}$$ is $$0.0016$$ for a disc in the $$(t,x)$$ plane, and $$0.004$$ both for a thin strip around a timelike line, in which 94% of pairs are related, and for one around a spacelike line, in which 6.5% are. Whether Theorem 1 holds for every region with a bounded density is open. So is a rate of convergence; the observed distances fall roughly as $$0.1/\sqrt N$$.

For a massive field the causal-set retarded propagator is no longer a multiple of $$C$$, and (1) does not apply. In more than two dimensions a sprinkled causal set is not the intersection of two linear orders, and no analogue of (2) is known. Whether the Pauli–Jordan matrix in higher dimensions has a comparable free decomposition, for instance over null directions, is open.

The comparison of section 8 is numerical, uses sprinklings of at most 4000 points and leaves differences of up to about 10%. A derivation of the volume coefficient from the thinned-chain picture, or a proof that the two-sector coupling contributes only a correction, is not available. The results here concern the eigenvalue distribution. They do not determine which modes the SJ state or the entanglement entropy should retain; Proposition 3 and the discussion after it show only that the eigenvalue distribution alone cannot make that choice.

## References

1. <a id="ref-johnston09"></a>S. Johnston, “Feynman propagator for a free scalar field on a causal set,” [arXiv:0909.0944](https://arxiv.org/abs/0909.0944).
2. <a id="ref-johnston10"></a>S. Johnston, “Quantum fields on causal sets,” PhD thesis, [arXiv:1010.5514](https://arxiv.org/abs/1010.5514).
3. <a id="ref-aas"></a>N. Afshordi, S. Aslanbeigi, R. D. Sorkin, “A distinguished vacuum state for a quantum field in a curved spacetime: formalism, features, and cosmology,” [arXiv:1205.1296](https://arxiv.org/abs/1205.1296).
4. <a id="ref-sorkin12"></a>R. D. Sorkin, “Expressing entropy globally in terms of (4D) field-correlations,” [arXiv:1205.2953](https://arxiv.org/abs/1205.2953).
5. <a id="ref-ssy"></a>M. Saravani, R. D. Sorkin, Y. K. Yazdi, “Spacetime entanglement entropy in 1+1 dimensions,” [arXiv:1311.7146](https://arxiv.org/abs/1311.7146).
6. <a id="ref-sy"></a>R. D. Sorkin, Y. K. Yazdi, “Entanglement entropy in causal set theory,” [arXiv:1611.10281](https://arxiv.org/abs/1611.10281).
7. <a id="ref-jy"></a>J. Y. L. Jones, Y. K. Yazdi, “Spectral density of the causal propagator,” [arXiv:2606.00311](https://arxiv.org/abs/2606.00311).
8. <a id="ref-nica"></a>A. Nica, “Asymptotically free families of random unitaries in symmetric groups,” Pacific J. Math. 157 (1993) 295–310.
9. <a id="ref-km"></a>M. S. Kammoun, M. Maïda, “Small cycle structure for words in conjugation invariant random permutations,” [arXiv:2204.04759](https://arxiv.org/abs/2204.04759).
10. <a id="ref-bv"></a>H. Bercovici, D. Voiculescu, “Free convolution of measures with unbounded support,” Indiana Univ. Math. J. 42 (1993) 733–773.
