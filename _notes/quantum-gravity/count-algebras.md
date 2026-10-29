---
title: Integer counts and finite-dimensional algebras in a positive-energy vacuum
date: 2026-10-29
category: substrate
order: 2
description: Integer-valued, translation-covariant counts with finite variance have zero density in a positive-energy vacuum when their two-point function is real or they are confined to a light ray, and a positive-energy net of finite-dimensional local factors is trivial.
status: Theorems 1, 5, 7 and 8 and Propositions 3 and 6 are proved under the hypotheses stated in sections 1, 3 and 5. The four-dimensional light-ray statement is a remark that holds under an explicit extra reality hypothesis. The chiral-current example is an analytic and numerical computation.
excludes: Integer counts that are additive, translation covariant and of finite variance in a translation-invariant vector of a positive-energy representation, at non-zero density, when their two-point function is real (in particular for commuting counts) or when they are confined to a light ray with real two-point function on intervals of disjoint closure; finite-dimensional local factors with exact positive-energy translations, weak additivity and locality. Counts with a complex two-point function that are not confined to a light ray are not covered.
---

A classical sprinkling counts points: it assigns to each bounded region $$D$$ a non-negative integer $$N(D)$$, additively over disjoint regions, with a law that does not change when $$D$$ is translated. Can observables of a quantum field in its ground state do the same at non-zero density? We take a Hilbert space with unitary translations $$U(a)$$ whose generator has spectrum in the closed forward cone (positive energy), a translation-invariant vector $$\Omega$$ (the vacuum), and self-adjoint count operators $$N(D)$$ with integer spectrum, additive and covariant, with finite variance in $$\Omega$$. The mean count per unit volume then vanishes whenever the two-point function $$\langle N(D)\Omega,N(D+a)\Omega\rangle$$ is real, which includes commuting counts (Theorem 1). On a light ray the same holds when this function is real for intervals with disjoint closures (Theorem 5). Whether it holds for counts with a complex two-point function that are not confined to a light ray is not decided. For counts of a second-quantised field, integer spectrum and additivity make the counts commute and the Fock space factorise over disjoint regions (Proposition 6). If every region carries a finite-dimensional algebra instead, positive-energy translations bound the dimension of the whole Hilbert space, and locality then makes every local algebra trivial (Theorems 7 and 8).

Reeh and Schlieder [[RS]](#ref-rs) showed that the vacuum of a quantum field satisfying the spectrum condition is cyclic for the algebra of any open region, by analytic continuation in the translations; the same continuation is used in section 5. Halvorson and Clifton [[HC]](#ref-hc) prove several no-go theorems against localised particle observables in relativistic quantum theories. Additive integer counts at finite density differ from a projection-valued localisation of one particle in that they need not commute and need not be bounded, and the sections below treat them directly. The classification of invariant bosonic Fock states with counts $$d\Gamma(1_D)$$ is on [Lorentz-invariant quantum sprinklings](/quantum-gravity/quantum-sprinklings/). When the intensity of such a state has finite second moment and the translations in its GNS representation are strongly continuous, Theorem 1 shows that the state is not a positive-energy vacuum.

## 1. Count systems and their hypotheses

Let $$\mathcal H$$ be a Hilbert space and $$a\mapsto U(a)$$ a strongly continuous unitary representation of the translation group $$\mathbb R^4$$ (or of $$\mathbb R$$ for the light ray). Write $$U(a)=\int e^{i(p^0a^0-\vec p\cdot\vec a)}dE(p)$$ with a projection-valued measure $$E$$. Positive energy means

(PE) $$E$$ is supported in the closed forward cone $$\bar V_+=\{p:p^0\ge\lvert\vec p\rvert\}$$.

Since $$\bar V_+\cap(-\bar V_+)=\{0\}$$, a vector whose spectral measure is symmetric under $$p\mapsto-p$$ is translation invariant, and the proofs of Theorems 1 and 5 use only this property. A count system consists of the following data.

(C1) A unit vector $$\Omega$$ (the vacuum) with $$U(a)\Omega=\Omega$$ for all $$a$$.

(C2) For every bounded Borel set $$D\subset\mathbb R^4$$, a self-adjoint operator $$N(D)$$ with spectrum in $$\mathbb Z$$ (in $$\mathbb N_0=\{0,1,2,\dots\}$$ where we say so) and $$\Omega\in\mathrm{dom}\,N(D)$$. The last condition says that the second moment $$\langle\Omega,N(D)^2\Omega\rangle$$ is finite.

(C3) $$U(a)N(D)U(a)^*=N(D+a)$$ as an equality of self-adjoint operators, so that $$U(a)$$ maps $$\mathrm{dom}\,N(D)$$ onto $$\mathrm{dom}\,N(D+a)$$.

(C4) For disjoint bounded Borel sets $$A,B$$: $$N(A\cup B)\Omega=N(A)\Omega+N(B)\Omega$$.

Only the vectors $$N(D)\Omega$$ enter the theorems. We therefore do not assume that $$N(A\cup B)$$ is the closure of $$N(A)+N(B)$$ (a sum of two unbounded self-adjoint operators need not be self-adjoint) or that the $$N(D)$$ have a common core. Where operator additivity is needed, it is stated. Write $$\square=[0,1)^4$$ for the unit cube and $$\rho=\langle\Omega,N(\square)\Omega\rangle$$ for the mean count in it. By (C1), (C3) and (C4) the mean count in any translate of $$\square$$ is the same, and the mean count in a cube of side $$1/m$$ is $$\rho/m^4$$. The last hypothesis concerns the two-point function of counts at translated positions.

(R) $$\langle N(D)\Omega,N(D+a)\Omega\rangle\in\mathbb R$$ for all bounded Borel $$D$$ and all $$a$$.

If $$N(D)$$ and $$N(D+a)$$ commute strongly (their spectral projections commute), (R) holds: both vectors are functions of a common joint spectral measure of $$\Omega$$, and $$\int nm\,d\nu(n,m)$$ is real. Counts with neither commutation nor reality are treated in section 3.

Commuting counts are classical. If all $$N(D)$$ commute strongly, their joint spectral measure in $$\Omega$$ is the law of a family of integer random variables $$N(D)$$, additive on disjoint sets almost surely by (C4) and with a law invariant under translations by (C1) and (C3). For $$\mathbb N_0$$-valued counts for which the identity in (C4) is also $$\sigma$$-additive, that is $$N(\bigcup_jA_j)\Omega=\sum_jN(A_j)\Omega$$ in norm for disjoint sets with bounded union, monotone convergence of the partial sums shows that $$N$$ is almost surely a $$\sigma$$-additive integer-valued set function: a locally finite point process. (For $$\mathbb Z$$-valued counts the partial sums are not monotone, and the limit is a priori only a finitely additive signed set function.) Conversely, every translation-invariant point process with finite second moments is a count system with commuting counts. Take for $$\mathcal H$$ the space $$L^2$$ of its law, for $$N(D)$$ the multiplication by the number of points in $$D$$, for $$U(a)$$ the Koopman representation (it translates a configuration of points by $$-a$$ inside a function of configurations) and $$\Omega=1$$. Without an energy condition, commuting counts therefore are classical point processes, and a classical sprinkling when the process is Lorentz invariant. The energy condition is the extra hypothesis of Theorem 1.

A Poisson sprinkling shows how (PE) fails. In the Koopman representation of a Poisson process of intensity $$\lambda$$, the centred vector $$\psi=(N(D)-\lambda V(D))\Omega$$, with $$V(D)$$ the volume of $$D$$, has $$\langle\psi,U(a)\psi\rangle=\mathrm{Cov}(N(D),N(D+a))=\lambda V(D\cap(D+a))$$. The Fourier transform of this function is the spectral measure $$d\mu_\psi=\lambda\lvert\hat 1_D(p)\rvert^2\,dp/(2\pi)^4$$, of total mass $$\lambda V(D)$$ by Parseval. It is symmetric under $$p\mapsto-p$$, and its density is non-zero for almost every $$p\in\mathbb R^4$$ because $$\hat 1_D$$ is a non-zero entire function, so $$\mu_\psi$$ is far from supported in $$\bar V_+$$. For the mixed Poisson states of [quantum sprinklings](/quantum-gravity/quantum-sprinklings/) with $$\mathbb E[\Lambda^2]<\infty$$, the covariance of the counts has the same form plus the constant $$\mathrm{Var}(\Lambda)V(D)^2$$, which adds an atom at $$p=0$$ and leaves the rest of the measure unchanged.

## 2. A real two-point function forces zero density

<div class="theorem" markdown="1">

**Theorem 1** (no positive-energy count at non-zero density). Let $$(\mathcal H,U,\Omega,N)$$ satisfy (PE), (C1)–(C4) and (R). Then $$\rho=0$$. If the spectra lie in $$\mathbb N_0$$, then in addition $$N(D)\Omega=0$$ for every bounded Borel set $$D$$.

</div>

Reality makes the autocorrelation $$\langle\psi,U(a)\psi\rangle$$ of $$\psi=N(D)\Omega$$ an even function of $$a$$. Positive energy then leaves only $$p=0$$ in the spectral measure of $$\psi$$, so $$N(D)\Omega$$ does not depend on the position of $$D$$. The $$m^4$$ cells of side $$1/m$$ in the unit cube then carry the same vector, and these vectors add coherently: the second moment of $$N(\square)$$ is $$m^8$$ times that of one cell. For an integer-valued variable the second moment is at least the mean, which for a cell is $$\rho/m^4$$. The second moment of $$N(\square)$$ is therefore at least $$m^4\rho$$ for every $$m$$, and it is finite, so $$\rho=0$$.

<div class="proof" markdown="1">

*Proof.* *Step 1 (the spectral measure is symmetric).* Fix a bounded Borel $$D$$, let $$\psi=N(D)\Omega$$ and $$F(a)=\langle\psi,U(a)\psi\rangle$$. By (C3) and (C1), $$U(a)\psi=N(D+a)\Omega$$, so $$F(a)=\langle N(D)\Omega,N(D+a)\Omega\rangle$$, which is real by (R). Also $$F(-a)=\overline{F(a)}$$, hence $$F(-a)=F(a)$$. Write $$F(a)=\int e^{ip\cdot a}d\mu_\psi(p)$$ with the positive measure $$\mu_\psi=\langle\psi,E(\cdot)\psi\rangle$$, supported in $$\bar V_+$$ by (PE). By uniqueness of the Fourier–Stieltjes transform, $$F(-a)=F(a)$$ for all $$a$$ says that $$\mu_\psi$$ equals its reflection under $$p\mapsto-p$$. So $$\mu_\psi$$ is supported in $$\bar V_+\cap(-\bar V_+)=\{0\}$$, and $$\lVert U(a)\psi-\psi\rVert ^2=\int\lvert e^{ip\cdot a}-1\rvert^2d\mu_\psi=0$$. Therefore

$$
N(D+a)\Omega=N(D)\Omega\qquad\text{for all bounded Borel }D\text{ and all }a. \tag{1}
$$

*Step 2 (subdivision).* For an integer $$m\ge1$$ the cube $$\square$$ is the disjoint union of the $$m^4$$ translates $$\square_m+j/m$$, $$j\in\{0,\dots,m-1\}^4$$, of $$\square_m=[0,1/m)^4$$. By (C4), used finitely many times, and (1), $$N(\square)\Omega=m^4N(\square_m)\Omega$$. Taking the inner product with $$\Omega$$ gives $$\langle N(\square_m)\rangle=m^{-4}\rho$$. Let $$\nu$$ be the spectral measure of $$N(\square_m)$$ in $$\Omega$$. Since the spectrum is integer, $$n^2\ge\lvert n\rvert$$ on it, hence $$\lVert N(\square_m)\Omega\rVert ^2=\int n^2d\nu\ge\lvert\int n\,d\nu\rvert=\lvert\langle N(\square_m)\rangle\rvert$$. Therefore

$$
\lVert N(\square)\Omega\rVert ^2=m^8\lVert N(\square_m)\Omega\rVert ^2\ \ge\ m^8\,m^{-4}\lvert\rho\rvert=m^4\lvert\rho\rvert. \tag{2}
$$

The left side does not depend on $$m$$ and is finite by (C2). Letting $$m\to\infty$$ gives $$\rho=0$$.

*Step 3 (non-negative spectra).* If the spectra lie in $$\mathbb N_0$$, then $$\langle N(\square)\rangle=0$$, so the spectral measure of $$N(\square)$$ in $$\Omega$$ sits at $$n=0$$ and $$N(\square)\Omega=0$$. The same holds for every translate by (1). A bounded $$D$$ lies in a finite union of cells $$\square+j$$, $$j\in\mathbb Z^4$$. On each cell, $$0=N(\square+j)\Omega=N(D\cap(\square+j))\Omega+N((\square+j)\setminus D)\Omega$$ with both terms of non-negative mean, so both means vanish, and a non-negative integer-valued operator with zero mean annihilates $$\Omega$$. Summing over cells gives $$N(D)\Omega=0$$. $$\square$$

</div>

Lorentz invariance and commutation of counts at spacelike separation were not used, and reality of the two-point function enters in Step 1 only. The vacuum has to be an invariant vector: in a thermal state, the GNS translation generator has both signs and the conclusion fails. If $$\Omega$$ is replaced by a countable mixture of invariant vectors, each satisfying the hypotheses, each component has $$\rho=0$$.

<div class="corollary" markdown="1">

**Corollary 2** (classical sprinklings are not vacua). Let $$\Phi$$ be a translation-invariant point process with finite second moments and intensity $$\rho>0$$. There is no count system with commuting counts and positive energy (PE) whose counts have the joint law of $$\Phi$$ in the vector $$\Omega$$.

</div>

Such a system would satisfy (C1)–(C4) and (R), so Theorem 1 would give $$\rho=0$$.

Commuting counts can arise from a purely algebraic condition on projections.

<div class="proposition" markdown="1">

**Proposition 3** (projections with integer sum commute). Let $$e,f$$ be orthogonal projections on a Hilbert space such that $$e+f$$ has spectrum in $$\mathbb Z$$. Then $$ef=fe$$.

</div>

The proposition is used only for Corollary 4.

<div class="proof" markdown="1">

*Proof.* Put $$S=e+f$$. Since $$0\le S\le2$$, the spectrum lies in $$\{0,1,2\}$$ and $$S(S-1)(S-2)=0$$. With $$e^2=e$$, $$f^2=f$$ and $$T=ef+fe$$ one has $$S^2=S+T$$, so $$S^3-3S^2+2S=(S-2)T$$ and $$(S-2)T=0$$. Expanding, $$(ef+fe)(e+f-2)=efe+fef-ef-fe=0$$. Multiplying on the left by $$e$$ gives $$efef=ef$$, so $$q=ef$$ is an idempotent of norm at most 1. Such an operator is an orthogonal projection: for $$x\in\mathrm{ran}\,q$$, $$y\in\ker q$$ and any scalar $$t$$, $$q(x+ty)=x$$, so $$\lVert x\rVert \le\lVert x+ty\rVert $$ for all $$t$$, hence $$x\perp y$$. Thus $$ef=(ef)^*=fe$$. $$\square$$

</div>

The hypothesis that the operators are projections matters. Let $$e,f$$ project onto two lines of $$\mathbb C^2$$ spanned by unit vectors $$u,v$$ with $$\lvert\langle u,v\rangle\rvert=1/2$$. Then $$X=2e$$ and $$Y=2f$$ have spectrum $$\{0,2\}$$, the sum $$X+Y$$ has eigenvalues $$2(1\pm\frac12)=1,3$$, and $$[X,Y]$$ has eigenvalues $$\pm i\sqrt3$$. Integer spectrum and additivity alone do not force commutation, which is why Theorem 5 does not assume it.

<div class="corollary" markdown="1">

**Corollary 4** (counts that are projections on small sets). Suppose the counts satisfy (C2), have operator additivity $$N(A\cup B)=N(A)+N(B)$$ for disjoint bounded sets, and every bounded Borel set is a finite disjoint union of sets $$A$$ for which $$N(A)$$ is a projection. Then all the $$N(D)$$ commute, so (R) holds.

</div>

<div class="proof" markdown="1">

*Proof.* If $$A,B$$ are disjoint with $$N(A),N(B)$$ projections, then $$N(A\cup B)=N(A)+N(B)$$ has integer spectrum by (C2), so the two commute by Proposition 3. For overlapping $$A,B$$ of this kind write $$A\cap B$$, $$A\setminus B$$ and $$B\setminus A$$ as finite disjoint unions of sets with projection counts. All the pieces are pairwise disjoint, so their counts commute pairwise, and $$N(A)$$ and $$N(B)$$ are sums of these. For general $$D,D'$$ decompose each into such sets. $$\square$$

</div>

The hypothesis holds for a point set with a hard core, in which every sufficiently small set contains at most one point and $$N(A)\in\{0,1\}$$; a lattice is an example. We do not use whether a Lorentz-invariant sprinkling can have this property.

## 3. Counts on a light ray without commutation

Reality (R) is automatic when counts commute, and it can fail when the commutator of two counts has a non-zero vacuum expectation. On a light ray, positive energy together with finite variance removes that freedom for adjacent counts. A light-ray count system has $$U$$ a representation of $$\mathbb R$$ with $$U(a)=\int_{[0,\infty)}e^{ipa}dE(p)$$ (positive generator), a vector $$\Omega$$ as in (C1), and for every bounded half-open interval $$I=[s,t)$$ an operator $$N(I)$$ as in (C2)–(C4), with intervals in place of Borel sets and additivity required for adjacent intervals. The weaker form of (R) is as follows.

(R$$'$$) $$\langle N(I)\Omega,N(J)\Omega\rangle\in\mathbb R$$ whenever the closures $$\bar I$$ and $$\bar J$$ are disjoint.

Nothing is assumed for overlapping or adjacent intervals. Strong commutation of $$N(I)$$ and $$N(J)$$ for intervals with disjoint closures implies (R$$'$$).

<div class="theorem" markdown="1">

**Theorem 5** (light-ray counts have zero density). A light-ray count system satisfying (R$$'$$) has $$\rho=\langle N([0,1))\rangle=0$$. If the spectra lie in $$\mathbb N_0$$, then $$N(I)\Omega=0$$ for all $$I$$.

</div>

For overlapping intervals $$F(a)=\langle N(I)\Omega,N(I+a)\Omega\rangle$$ need not be real, and its imaginary part is the vacuum expectation of a commutator. That expectation is the same for every pair of adjacent pieces, because only a small neighbourhood of the shared endpoint can contribute. A constant imaginary part on an interval of $$a$$ values, vanishing for large $$a$$, contradicts the continuity of $$F$$ at $$0$$ unless the constant is zero. Continuity of $$F$$ is where finite variance enters; positive energy is used in the last step, as in Theorem 1.

<div class="proof" markdown="1">

*Proof.* *Step 1 (edge universality).* For adjacent intervals $$A=[s,x)$$ and $$B=[x,t)$$ let $$c(A,B)=\mathrm{Im}\langle N(A)\Omega,N(B)\Omega\rangle$$. For $$0<\delta<\min(x-s,t-x)$$ split $$A=A_1\cup A_2$$ with $$A_2=[x-\delta,x)$$ and $$B=B_1\cup B_2$$ with $$B_1=[x,x+\delta)$$. By (C4), $$\langle N(A)\Omega,N(B)\Omega\rangle$$ is a sum of four terms. The pairs $$(A_1,B)$$ and $$(A_2,B_2)$$ have disjoint closures, so by (R$$'$$) those terms are real. Hence $$c(A,B)=\mathrm{Im}\langle N(A_2)\Omega,N(B_1)\Omega\rangle=:c_\delta(x)$$. Translation covariance makes $$c_\delta(x)$$ independent of $$x$$, and the identity $$c(A,B)=c_\delta$$ for all $$\delta$$ below the shorter length shows that $$c_\delta$$ is the same for all $$\delta$$. Call this common value $$c_0$$, so $$c(A,B)=c_0$$ for every pair of adjacent intervals.

*Step 2 (a plateau in the autocorrelation).* Fix $$L>0$$, $$I=[0,L)$$, $$\psi=N(I)\Omega$$ and $$F(a)=\langle\psi,U(a)\psi\rangle=\langle N(I)\Omega,N(I+a)\Omega\rangle$$. For $$0<a<L$$ write $$I=A\cup B$$ with $$A=[0,a)$$, $$B=[a,L)$$ and $$I+a=B\cup C$$ with $$C=[L,L+a)$$. Then $$F(a)=\langle N(A)\Omega,N(B)\Omega\rangle+\langle N(A)\Omega,N(C)\Omega\rangle+\lVert N(B)\Omega\rVert ^2+\langle N(B)\Omega,N(C)\Omega\rangle$$. The second term is real because $$\bar A\cap\bar C=\emptyset$$ for $$a<L$$, the third is real, and the first and fourth contribute $$c(A,B)$$ and $$c(B,C)$$. So $$\mathrm{Im}F(a)=2c_0$$. For $$a>L$$ the intervals $$I$$ and $$I+a$$ have disjoint closures and $$\mathrm{Im}F(a)=0$$.

*Step 3 (continuity).* $$F$$ is continuous because $$U$$ is strongly continuous, and $$F(0)=\lVert \psi\rVert ^2$$ is real and finite by (C2). Letting $$a\to0^+$$ in Step 2 gives $$2c_0=0$$. Hence $$\mathrm{Im}F(a)=0$$ for all $$a\ne\pm L$$, and by continuity for all $$a$$, because $$F(-a)=\overline{F(a)}$$.

*Step 4 (zero density).* $$F$$ is real, so as in Theorem 1 the spectral measure of $$\psi$$ is symmetric. It is supported in $$[0,\infty)$$ by positivity of the generator, so it sits at $$0$$ and $$\psi$$ is invariant: $$N(I+a)\Omega=N(I)\Omega$$. The same argument applies to every length. Subdivide $$[0,L)$$ into $$2^k$$ intervals $$I_k$$ of length $$L2^{-k}$$. Integrality, used exactly as in Step 2 of Theorem 1, gives $$\lVert N(I)\Omega\rVert ^2\ge 4^k\lvert\langle N(I_k)\rangle\rvert=2^k\lvert\langle N(I)\rangle\rvert$$ for all $$k$$, so $$\langle N(I)\rangle=0$$. For $$\mathbb N_0$$-valued counts, a zero mean gives $$N(I)\Omega=0$$ for every interval $$I$$. $$\square$$

</div>

Finite variance is essential. The U(1) current $$J$$ of a chiral Dirac fermion has vacuum correlator $$\langle J(x)J(y)\rangle=-1/(4\pi^2(x-y-i0)^2)=\int_0^\infty\frac{p\,dp}{4\pi^2}e^{-ip(x-y)}$$, a positive spectral density on $$p>0$$. Smear the charge of an interval $$I=[0,L)$$ with a Gaussian $$g_\varepsilon$$ of width $$\varepsilon$$, $$\mathcal J_\varepsilon(I)=J(1_I*g_\varepsilon)$$. Its autocorrelation, with the convention $$U(a)=\int e^{ipa}dE$$ of section 1, is

$$
F_\varepsilon(a)=\int_0^\infty\frac{p\,dp}{4\pi^2}\,\frac{4\sin^2(pL/2)}{p^2}\,e^{-\varepsilon^2p^2}e^{ipa},\qquad
\mathrm{Im}\,F_\varepsilon(a)\ \xrightarrow{\ \varepsilon\to0\ }\ \frac1{\pi^2}\int_0^\infty\frac{\sin^2(pL/2)\sin(pa)}{p}\,dp=\begin{cases}\dfrac1{4\pi},&0<a<L,\\[4pt]0,&a>L.\end{cases}
$$

This is the plateau of Step 2 with $$2c_0=1/(4\pi)$$, so $$c_0=1/(8\pi)\ne0$$. It agrees with the commutator $$[J(x),J(y)]=\frac{i}{2\pi}\delta'(x-y)$$ integrated over adjacent intervals with half weight at the shared endpoint, which gives $$\langle[\mathcal J_A,\mathcal J_B]\rangle=2ic_0=i/(4\pi)$$. The plateau does not contradict Theorem 5, because the variance of the sharp charge diverges:

$$
F_\varepsilon(0)=\frac1{2\pi^2}\Big(\ln\frac{L}{\varepsilon}+\frac\gamma2\Big)+o(1),
$$

with $$\gamma$$ Euler's constant. A direct quadrature at $$L=1$$ agrees with both formulas.

| $$\varepsilon$$ | $$\mathrm{Im}F_\varepsilon$$ at $$a=0.05,\,0.3,\,0.7,\,0.95$$ | $$\mathrm{Im}F_\varepsilon$$ at $$a=1.5$$ | $$F_\varepsilon(0)$$, quadrature | $$F_\varepsilon(0)$$, formula |
|---|---|---|---|---|
| 0.02 | 0.0734, 0.0796, 0.0796, 0.0765 | 0.0000 | 0.2128 | 0.2128 |
| 0.005 | 0.0796 at all four | 0.0000 | 0.2830 | 0.2830 |

The plateau value is $$1/(4\pi)=0.0796$$. The sharp charge is outside the domain of the vacuum and the continuity used in Step 3 fails, so the example does not satisfy (C2). The smeared charge also has continuous spectrum, and the vector $$\mathcal J_\varepsilon(I)\Omega$$ is not translation invariant, because $$\mathrm{Im}F\ne0$$ and (R) fails for it.

A conserved current $$j^\mu$$ has a constant Lorentz-vector expectation in a Poincaré-invariant state, hence zero. A charge counted in a diamond therefore has zero mean. An $$\mathbb N_0$$-valued charge count then vanishes, and a $$\mathbb Z$$-valued one counts positive and negative units with equal mean densities rather than a density of points.

<div class="remark" markdown="1">

**Remark** (four dimensions). Let $$n\in\bar V_+$$ be a null vector, $$u$$ a coordinate with $$a\mapsto U(an)$$ translating $$u$$ by $$a$$, and $$K$$ a bounded Borel set in the other three coordinates. The generator of $$a\mapsto U(an)$$ is $$n\cdot P$$, which is non-negative on $$\bar V_+$$. Suppose (R$$'$$) holds for $$N(I\times K)$$ and $$N(J\times K)$$ whenever $$\bar I\cap\bar J=\emptyset$$. The four steps of the proof of Theorem 5 then apply with $$N(I\times K)$$ in place of $$N(I)$$: Steps 1 and 2 use (C4) and covariance along $$u$$ only. They give $$\langle N(I\times K)\rangle=0$$, and $$N(I\times K)\Omega=0$$ for $$\mathbb N_0$$-valued counts. If (R$$'$$) holds for $$K=[0,1)^3$$, this is $$\rho=0$$. The hypothesis is a reality condition on counts that are separated along a null direction only. It does not follow from Einstein causality: for $$K$$ of positive extent in the other null coordinate, a region $$I\times K$$ and its translate along $$u$$ contain timelike separated pairs for every non-zero shift.

</div>

## 4. Integer second-quantised counts factorise

For a one-particle Hilbert space $$h$$ and a bounded operator $$\Pi$$ on $$h$$, $$d\Gamma(\Pi)$$ is the operator on the bosonic Fock space $$\Gamma(h)=\bigoplus_{n\ge0}h^{\otimes_sn}$$ that acts on the $$n$$-particle sector as $$\sum_{j=1}^n1\otimes\cdots\otimes\Pi\otimes\cdots\otimes1$$, with $$\Pi$$ in the $$j$$th slot. On the one-particle sector it acts as $$\Pi$$. For a field with one-particle space $$h$$, the count of a region $$A$$ is $$N(A)=d\Gamma(\Pi(A))$$ with $$\Pi(A)$$ a positive contraction on $$h$$ ($$0\le\Pi(A)\le1$$), additive: $$\Pi(A\cup B)=\Pi(A)+\Pi(B)$$ for disjoint $$A,B$$. When $$\Pi(A)$$ is multiplication by $$1_A$$ on $$L^2$$ of spacetime, this is the count of [Lorentz-invariant quantum sprinklings](/quantum-gravity/quantum-sprinklings/). A different $$\Pi(A)$$, for instance a compression of $$1_A$$ to the positive-frequency subspace, would be natural for a positive-energy field.

<div class="proposition" markdown="1">

**Proposition 6** (integer counts force projections). Assume that $$N(A)=d\Gamma(\Pi(A))$$ has integer spectrum for every bounded set $$A$$ and that $$\Pi$$ is additive. Then every $$\Pi(A)$$ is an orthogonal projection, $$\Pi(A)\Pi(B)=0$$ for disjoint $$A,B$$, and all the $$N(A)$$ commute. For any finite partition $$(A_i)$$ of a bounded set $$A_0$$, the Fock space over $$h$$ is the tensor product of the Fock spaces over the subspaces $$\Pi(A_i)h$$ and over the complement $$(1-\Pi(A_0))h$$, and $$N(A_i)$$ acts on the $$i$$th factor only.

</div>

<div class="proof" markdown="1">

*Proof.* The one-particle sector $$\Gamma^1(h)=h$$ reduces $$N(A)$$, and $$N(A)$$ acts on it as $$\Pi(A)$$. The spectrum of a restriction to a reducing subspace is contained in the spectrum of the operator, so $$\mathrm{spec}\,\Pi(A)\subset\mathbb Z\cap[0,1]=\{0,1\}$$, and $$\Pi(A)$$ is a projection. For disjoint $$A,B$$, the operators $$e=\Pi(A)$$, $$f=\Pi(B)$$ and $$\Pi(A\cup B)=e+f$$ are all projections, so $$(e+f)^2=e+f$$ gives $$ef+fe=0$$. Multiplying by $$e$$ on the left and on the right gives $$ef=-efe=fe$$, hence $$ef=0$$. For overlapping $$A,B$$, write $$\Pi(A)=\Pi(A\setminus B)+\Pi(A\cap B)$$ and $$\Pi(B)=\Pi(B\setminus A)+\Pi(A\cap B)$$; the three pieces are mutually orthogonal projections, so $$\Pi(A)$$ and $$\Pi(B)$$ commute, and so do $$N(A)$$ and $$N(B)$$. Finally $$h=\bigoplus_i\Pi(A_i)h\oplus(1-\Pi(A_0))h$$, the exponential property $$\Gamma(h_1\oplus h_2)=\Gamma(h_1)\otimes\Gamma(h_2)$$ gives the tensor product, and $$d\Gamma(\Pi(A_i))$$ acts on the $$i$$th factor only. $$\square$$

</div>

A non-projection $$\Pi$$ gives non-integer counts: $$d\Gamma(\mathrm{diag}(1,0.3))$$ on $$\Gamma(\mathbb C^2)$$ has the eigenvalue $$0.3$$ on the one-particle state of the second coordinate. Integer spectrum and additivity therefore already imply the factorisation of the Fock space over disjoint regions, and the commutation of counts that Theorem 1 uses through (R).

## 5. Finite-dimensional local algebras

Sections 2 to 4 concern observables that count. A substrate can also be discrete in a quantum theory by giving each region finitely many degrees of freedom, as in a lattice of qubits made covariant. This section tests that proposal with the same two ingredients as Theorem 1, an invariant vector and positive-energy translations. No count operator enters, and the results are independent of sections 2 to 4. Theorem 7 says that if one local algebra is finite dimensional, then so is the whole Hilbert space; Theorem 8 and Corollary 9 add locality and show that finite-dimensional factors are then trivial.

Suppose each bounded open region $$O$$ of $$\mathbb R^4$$ carries an algebra $$\mathcal A(O)\subset B(\mathcal H)$$ containing the identity, with the following properties.

(H1) Isotony: $$\mathcal A(O)\subset\mathcal A(O')$$ if $$O\subset O'$$.

(H2) Covariance: $$U(a)\mathcal A(O)U(a)^*=\mathcal A(O+a)$$ with $$U$$ as in (PE), and $$U(a)\Omega=\Omega$$.

(H3) Weak additivity: for every non-empty open $$O_1$$, the vectors $$B_1(a_1)\cdots B_\ell(a_\ell)\Omega$$, with $$\ell\ge0$$, $$a_i\in\mathbb R^4$$, $$B_i\in\mathcal A(O_1)$$ and $$B_i(a)=U(a)B_iU(a)^*$$, span a dense subspace of $$\mathcal H$$.

(H4) Locality: $$\mathcal A(O_1)$$ and $$\mathcal A(O_2)$$ commute when $$O_1$$ and $$O_2$$ are spacelike separated.

(H5) Finite type I factors: each $$\mathcal A(\mathcal D)$$ for a double cone $$\mathcal D$$ is isomorphic to a full matrix algebra $$M_{\kappa(\mathcal D)}$$.

<div class="theorem" markdown="1">

**Theorem 7** (finite-dimensional collapse of the Hilbert space). Assume (PE) and (H1)–(H3). If $$\dim\mathcal A(O_0)<\infty$$ for a single non-empty open $$O_0$$, then $$\dim\mathcal H\le\dim\mathcal A(O_0)$$. Locality (H4) is not used.

</div>

This is the Reeh–Schlieder argument. A vector orthogonal to the finite-dimensional space $$\mathcal A(O_0)\Omega$$ has matrix elements with products of translated local operators that vanish for small translations. Positive energy makes those matrix elements boundary values of analytic functions of the translations, so they vanish for all translations, and by (H3) the vector is zero.

<div class="proof" markdown="1">

*Proof.* Choose a non-empty open $$O_1$$ and $$\varepsilon>0$$ with $$O_1+a\subset O_0$$ for $$\lvert a\rvert<\varepsilon$$. Let $$\psi\perp\mathcal A(O_0)\Omega$$; this space contains $$\Omega$$ because $$1\in\mathcal A(O_0)$$. For $$B_1,\dots,B_\ell\in\mathcal A(O_1)$$ let

$$
G(a_1,\dots,a_\ell)=\langle\psi,B_1(a_1)\cdots B_\ell(a_\ell)\Omega\rangle .
$$

If every $$\lvert a_i\rvert<\varepsilon$$ then $$B_i(a_i)\in\mathcal A(O_0)$$ by (H1) and (H2), so the product lies in $$\mathcal A(O_0)$$ and $$G=0$$. Using $$U\Omega=\Omega$$ we rewrite $$B_1(a_1)\cdots B_\ell(a_\ell)\Omega=U(a_1)B_1U(a_2-a_1)B_2\cdots U(a_\ell-a_{\ell-1})B_\ell\Omega$$, so in the variables $$x_0=a_1$$, $$x_j=a_{j+1}-a_j$$,

$$
G=\langle\psi,U(x_0)B_1U(x_1)B_2\cdots U(x_{\ell-1})B_\ell\Omega\rangle .
$$

For $$z=x+iy$$ with $$y$$ in the open forward cone $$V_+$$, $$U(z)=\int e^{ip\cdot z}dE(p)$$ is a contraction, because $$\lvert e^{ip\cdot z}\rvert=e^{-p\cdot y}\le1$$ for $$p,y\in\bar V_+$$. It is analytic in $$z$$ and strongly continuous up to $$y=0$$. Hence $$G$$ is the boundary value of a function $$\tilde G(z_0,\dots,z_{\ell-1})$$, bounded by $$\lVert \psi\rVert \prod\lVert B_i\rVert $$, analytic in the product of tubes $$\mathbb R^4+iV_+$$ and continuous up to the real boundary. It vanishes on an open box in $$\mathbb R^{4\ell}$$ (the set where all $$\lvert a_i\rvert<\varepsilon$$ is open in the $$x$$ coordinates).

We show that $$\tilde G\equiv0$$ one variable at a time, and the first variable carries the argument. Fix $$z_1,\dots,z_{\ell-1}$$ at real points of the box and $$e\in V_+$$. The function $$\zeta\mapsto\tilde G(x_0+\zeta e,z_1,\dots)$$ is bounded and analytic on the upper half plane, continuous up to the real axis, and zero on an interval of the real axis (the points where $$x_0+\zeta e$$ lies in the box). Extended by $$0$$ across that interval, it is continuous and analytic off the interval, hence analytic across it by Morera's theorem (the Schwarz reflection principle for zero boundary values), and therefore identically zero by the identity theorem. Take $$\zeta=i$$ and let $$x_0$$ range over the box and $$e$$ over $$V_+$$: then $$\tilde G=0$$ at every $$z_0=x_0+ie$$ with $$x_0$$ in the box and $$e\in V_+$$. This is an open subset of the connected tube, so $$\tilde G=0$$ for every $$z_0$$ in the tube. Now fix $$z_0$$ arbitrary in the tube and repeat the argument for $$z_1$$, with $$z_2,\dots$$ real in the box, and so on. After $$\ell$$ steps $$\tilde G\equiv0$$ and, taking boundary values, $$G\equiv0$$ for all real $$a_i$$.

Thus $$\psi$$ is orthogonal to all the vectors in (H3), including the case $$\ell=0$$ (the vector $$\Omega$$), so $$\psi=0$$. Therefore $$\mathcal A(O_0)\Omega=\mathcal H$$, and $$\dim\mathcal H\le\dim\mathcal A(O_0)$$. $$\square$$

</div>

The next result needs locality and the factor property, and no unitary implementation of the translations. Let $$\mathcal D_r(x)$$ be the double cone over the ball of radius $$r/2$$ centred at $$x$$ at time 0, so that $$\mathcal D_r$$ has tip separation $$r$$.

<div class="theorem" markdown="1">

**Theorem 8** (packing). Assume (H1), (H4), (H5) and translation covariance by automorphisms, $$\alpha_a(\mathcal A(O))=\mathcal A(O+a)$$ (unitarity is not needed). Then for $$\tau\ge4\sigma$$,

$$
\log_2\kappa(\mathcal D_\tau)\ \ge\ \Big(\frac{\tau}{5\sigma}\Big)^3\log_2\kappa(\mathcal D_\sigma). \tag{3}
$$

In particular, if $$\mathcal A(\mathcal D_\sigma)\ne\mathbb C$$ for arbitrarily small $$\sigma$$, then $$\kappa(\mathcal D_\tau)=\infty$$, contradicting (H5); and if $$\mathcal A(\mathcal D_\sigma)\ne\mathbb C$$ for some $$\sigma$$, the logarithm of the dimension grows at least like the cube of the size, faster than any area law.

</div>

Double cones of size $$\sigma$$ whose centres are on a grid of spacing $$\sigma$$ are spacelike separated, so by locality their algebras commute and multiply into a tensor product inside the algebra of the large cone. The number of such cones in $$\mathcal D_\tau$$ grows like $$(\tau/\sigma)^3$$, and each contributes a factor $$\kappa(\mathcal D_\sigma)$$ to the matrix size.

<div class="proof" markdown="1">

*Proof.* Place balls of radius $$\sigma/2$$ centred at the points $$\sigma k$$, $$k\in\mathbb Z^3$$, with $$\lvert\sigma k\rvert+\sigma/2\le\tau/2$$. Their double cones $$\mathcal D_\sigma(\sigma k)$$ lie in $$\mathcal D_\tau$$ (if a ball lies in a ball, so do the diamonds over them), and are pairwise spacelike separated: for points $$p,q$$ in two of them with centres at distance at least $$\sigma$$, $$\lvert t_p-t_q\rvert<\sigma-\lvert\vec p-\vec c_p\rvert-\lvert\vec q-\vec c_q\rvert\le\lvert\vec p-\vec q\rvert$$. By covariance each $$\mathcal A(\mathcal D_\sigma(\sigma k))\cong M_{\kappa(\mathcal D_\sigma)}$$, and by (H4) these algebras commute. For commuting factors the map $$a_1\otimes\cdots\otimes a_n\mapsto a_1\cdots a_n$$ is a $$*$$-homomorphism from the tensor product, which is simple, and it is non-zero, hence injective. Its image lies in $$\mathcal A(\mathcal D_\tau)\cong M_{\kappa(\mathcal D_\tau)}$$ by (H1), so $$\kappa(\mathcal D_\tau)^2\ge\kappa(\mathcal D_\sigma)^{2n}$$, where $$n$$ is the number of centres, i.e. $$\kappa(\mathcal D_\tau)\ge\kappa(\mathcal D_\sigma)^n$$. Choose centres in the cube $$\lvert k\rvert_\infty\le m$$ with $$m=\lfloor(\tau/\sigma-1)/(2\sqrt3)\rfloor$$; then $$n\ge(2m+1)^3\ge((\tau/\sigma-1)/(2\sqrt3))^3\ge(\tau/5\sigma)^3$$ for $$\tau\ge4\sigma$$ (using $$2\lfloor x\rfloor+1\ge x$$ and $$(\tau/\sigma-1)/(2\sqrt3)\ge 0.2165\,\tau/\sigma$$). $$\square$$

</div>

<div class="corollary" markdown="1">

**Corollary 9** (triviality). Under (PE) and (H1)–(H5), $$\mathcal A(\mathcal D)=\mathbb C\cdot1$$ for every double cone $$\mathcal D$$.

</div>

<div class="proof" markdown="1">

*Proof.* By (H5) every double cone algebra is finite dimensional, so Theorem 7 gives $$\dim\mathcal H<\infty$$. The algebra $$\mathcal A(\mathcal D_\tau)\cong M_{\kappa(\mathcal D_\tau)}$$ is a unital subalgebra of $$B(\mathcal H)$$, so $$\kappa(\mathcal D_\tau)^2\le(\dim\mathcal H)^2$$ for all $$\tau$$. By (3) this forces $$\log_2\kappa(\mathcal D_\sigma)=0$$ for all $$\sigma$$, so $$\kappa(\mathcal D_\sigma)=1$$. $$\square$$

</div>

If $$U$$ extends to a unitary representation $$V$$ of the Lorentz group on a finite-dimensional $$\mathcal H$$, then $$V$$ is trivial (a finite-dimensional unitary representation of $$SL(2,\mathbb C)$$ has compact image, and $$SL(2,\mathbb C)$$ has no non-trivial compact quotient), so covariance gives $$U(\Lambda a)=U(a)$$ for all $$\Lambda$$. The spectral measure of $$U$$ is then Lorentz invariant and, on a finite-dimensional space, has finitely many atoms; a non-zero momentum has an infinite Lorentz orbit, so (PE) leaves $$p=0$$ only. The translations act trivially and the theory is static. Neither Lorentz invariance nor any statement about boosts enters Theorem 7 itself; the obstruction is positive energy.

The stabiliser in the Poincaré group of a bounded double cone is compact, a group of rotations about its centre, and a compact group has non-trivial finite-dimensional unitary representations. A finite-dimensional algebra attached to a region can therefore carry a non-trivial rotation action, and finite dimension alone does not trivialise Lorentz symmetry region by region. A site-based net, built as a tensor product of finite-dimensional factors over a locally finite subset $$S\subset\mathbb R^4$$, cannot be exactly covariant: a translation-invariant subset is empty or all of $$\mathbb R^4$$, and the orbit of any non-zero point under the Lorentz group meets a bounded set in a continuum of points, so no non-empty locally finite set is Poincaré invariant. Covariance can hold only in law, with a random set.

## 6. What the hypotheses leave open

Theorem 1 needs an invariant vector in a positive-energy representation. Thermal states, states of a field on an expanding background and states that break translation invariance are outside it. Without the energy condition, commuting $$\mathbb N_0$$-valued counts are the classical point processes of section 1, among them Poisson processes and Haar-random lattices, and they have positive density.

For counts whose two-point function is neither real nor confined to a light ray, none of the results applies. An example is a set of scalar volume counts in 1+1 or 3+1 dimensions that obey Einstein locality but not (R) or (R$$'$$). Whether positive energy, integrality and finite second moments are compatible with a non-zero density for such counts is open. They cannot be second-quantised counts of a field (Proposition 6) or projections on small sets (Corollary 4), since both give (R). One possible approach is to replace (R) in Theorem 1 by a bound on the imaginary part of the autocorrelation, using the Hilbert-transform relation between the real and imaginary parts of the autocorrelation of a positive-energy vector.

Unbounded or non-integer counts are not covered either: the sharp charge of a chiral current is not in the domain of the vacuum and the smeared charge has continuous spectrum, so Theorem 5 applies to neither. Counts on an algebra that is not a Fock algebra reach Theorems 1 and 5 only through (R) or (R$$'$$).

For finite-dimensional nets, Theorem 7 does not use locality. A net with a non-trivial centre in its local algebras (gauge-type edge modes) factorises only in a modified sense, so Theorem 8 and Corollary 9 do not apply to it, while Theorem 7 still bounds the dimension of the Hilbert space for unitary positive-energy translations and weak additivity. A finite total Hilbert space with a non-trivial net is excluded under (H4) and (H5), and not otherwise. Whether (H3) can be derived from (H1), (H2), (H4) and finite dimension is open. A quantum cellular automaton on $$\mathbb Z^4$$ has translations only by integers, with the translation spectrum on a torus, so (PE) has no meaning there and Theorem 7 does not apply. Finite-dimensional diamonds that are not subalgebras of a common algebra on a common Hilbert space are outside (H1)–(H3).

## References

1. <a id="ref-rs"></a>Reeh, Schlieder, "Bemerkungen zur Unitäräquivalenz von Lorentzinvarianten Feldern," Nuovo Cimento 22 (1961) 1051–1068, [doi:10.1007/bf02787889](https://doi.org/10.1007/bf02787889). Used for: analytic continuation in the translations from the forward tube, the argument behind Theorem 7.
2. <a id="ref-hc"></a>Halvorson, Clifton, "No place for particles in relativistic quantum theories?", in Kuhlmann, Lyre, Wayne (eds.), Ontological Aspects of Quantum Field Theory, World Scientific 2002, pp. 181–213, [doi:10.1142/9789812776440_0010](https://doi.org/10.1142/9789812776440_0010); [arXiv:quant-ph/0103041](https://arxiv.org/abs/quant-ph/0103041). Used for: no-go theorems against localised particle observables in relativistic quantum theories, to which the integer-count statements are compared.
