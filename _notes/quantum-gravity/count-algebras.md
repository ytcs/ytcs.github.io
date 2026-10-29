---
title: Integer counts of a positive-energy field
date: 2026-10-29
category: substrate
order: 2
description: In a representation with translation generators of positive energy and an invariant vector, integer-valued additive counting observables with finite second moments have zero density, whether or not they commute, and finite-dimensional local algebras collapse. Integer second-quantised counts also force the underlying algebra to factorise over regions.
status: Theorems proved in full under explicitly stated domain hypotheses; the light-ray statement is proved for intervals, and its four-dimensional version only for counts that are local along a null direction; a numerical check of the edge commutator for the chiral current.
excludes: Counting observables of integer spectrum, additive and translation covariant, with finite second moments in a translation-invariant vector state of a positive-energy representation, at non-zero density, when the two-point function of counts is real (commuting counts) or the counts are local along a light ray; finite-dimensional local algebras with exact positive-energy translations.
---

A classical sprinkling counts points: it assigns to each bounded region $$D$$ a non-negative integer $$N(D)$$, additively over disjoint regions, with a law that does not change when $$D$$ is translated. We ask whether the same can be done by observables of a quantum field theory in its ground state. Concretely, take a Hilbert space with unitary translations $$U(a)$$ whose generator has spectrum in the closed forward cone (positive energy), a translation-invariant vector $$\Omega$$ (the vacuum), and self-adjoint operators $$N(D)$$ with integer spectrum, additive and covariant. The answer is that the density of counts in the vacuum must vanish. This holds when counts at translated positions have a real two-point function in $$\Omega$$, which includes the case that they commute (Theorem 1). On a light ray it holds without any commutation assumption beyond disjoint closures (Theorem 5), because positive energy forces the commutator of adjacent counts to vanish. For counts of a second-quantised field, integer spectrum and additivity already force the algebra to factorise over disjoint regions and the counts to commute (Proposition 6). If the local algebras are finite dimensional, exact positive-energy translations collapse the whole Hilbert space to the dimension of one local algebra, and with locality every local algebra is trivial (Theorems 7 and 8). The hypotheses are listed in section 1, the functional-analytic ones explicitly.

Reeh and Schlieder [[RS]](#ref-rs) showed that for a quantum field satisfying the spectrum condition the vacuum is cyclic for the algebra of any open region; the analytic continuation in the translations behind it is the one used in section 5. Halvorson and Clifton [[HC]](#ref-hc) give no-go theorems against localised particle observables in relativistic quantum theories. The statements below are about a different object: additive integer counts at finite density, which a sprinkling provides and which, unlike a projection-valued localisation of one particle, are not required to commute or to be bounded. The page adds the vacuum no-go for such counts, its extension past commutativity on a light ray, the factorisation consequence of second quantisation, and the finite-dimensional collapse in a form that does not use locality. The classification of invariant bosonic Fock states with counts $$d\Gamma(1_D)$$ is on [Lorentz-invariant quantum sprinklings](/quantum-gravity/quantum-sprinklings/): those states are never positive-energy vacua, which is a case of Theorem 1.

## 1. Count systems and their hypotheses

Let $$\mathcal H$$ be a Hilbert space and $$a\mapsto U(a)$$ a strongly continuous unitary representation of the translation group $$\mathbb R^4$$ (or of $$\mathbb R$$ for the light ray). We write $$U(a)=\int e^{i(p^0a^0-\vec p\cdot\vec a)}dE(p)$$ with a projection-valued measure $$E$$. Positive energy means

(P) $$E$$ is supported in the closed forward cone $$\bar V_+=\{p:p^0\ge\lvert\vec p\rvert\}$$.

Since $$\bar V_+\cap(-\bar V_+)=\{0\}$$, a vector whose spectral measure is symmetric under $$p\mapsto-p$$ is translation invariant. The proofs of Theorems 1 and 5 use only this property. A *count system* consists of the following data.

(C1) *Vacuum.* A unit vector $$\Omega$$ with $$U(a)\Omega=\Omega$$ for all $$a$$.

(C2) *Counts.* For every bounded Borel set $$D\subset\mathbb R^4$$ a self-adjoint operator $$N(D)$$ whose spectrum lies in $$\mathbb Z$$ (or in $$\mathbb N_0=\{0,1,2,\dots\}$$ if we say so) and with $$\Omega\in\mathrm{dom}\,N(D)$$. Equivalently, the second moment $$\langle\Omega,N(D)^2\Omega\rangle$$ is finite.

(C3) *Covariance.* $$U(a)N(D)U(a)^*=N(D+a)$$, as an equality of self-adjoint operators, so that $$U(a)$$ maps $$\mathrm{dom}\,N(D)$$ onto $$\mathrm{dom}\,N(D+a)$$.

(C4) *Additivity on the vacuum.* For disjoint bounded Borel $$A,B$$: $$N(A\cup B)\Omega=N(A)\Omega+N(B)\Omega$$.

Only the vectors $$N(D)\Omega$$ enter the theorems, so we do not have to assume that $$N(A\cup B)$$ equals the closure of $$N(A)+N(B)$$ (the sum of two unbounded self-adjoint operators need not be self-adjoint) or that the $$N(D)$$ have a common core. When we need operator additivity, we say so. The mean count in the unit cube $$Q=[0,1)^4$$ is $$\rho=\langle\Omega,N(Q)\Omega\rangle$$; by (C3), (C4) and (C1) the mean count in any translate is the same, and the mean count of a cube of side $$1/m$$ is $$\rho/m^4$$. The last hypothesis is about the two-point function of counts at translated positions.

(R) *Reality.* $$\langle N(D)\Omega,N(D+a)\Omega\rangle\in\mathbb R$$ for all bounded Borel $$D$$ and all $$a$$.

If $$N(D)$$ and $$N(D+a)$$ commute strongly (their spectral projections commute), (R) holds: the vectors are functions of a common joint spectral measure of $$\Omega$$, and $$\int nm\,d\nu(n,m)$$ is real. The case with neither commutation nor reality is the subject of section 3.

*Commuting counts are classical.* If all the $$N(D)$$ commute strongly, their joint spectral measure in $$\Omega$$ is the law of a family of integer random variables $$N(D)$$, additive on disjoint sets by (C4) (the vector identity says that the random variables satisfy $$N(A\cup B)=N(A)+N(B)$$ almost surely), and with a law invariant under translations by (C1) and (C3). If the identity is also $$\sigma$$-additive on $$\Omega$$, that is $$N(\bigcup_jA_j)\Omega=\sum_jN(A_j)\Omega$$ in norm for disjoint sets with bounded union, the monotone convergence of the partial sums shows that $$N$$ is almost surely a $$\sigma$$-additive integer-valued set function, a locally finite point process. Conversely, every translation-invariant point process with finite second moments is a count system with commuting counts: take $$\mathcal H=L^2$$ of its law, $$N(D)$$ the multiplication by the number of points in $$D$$, $$U$$ the Koopman representation and $$\Omega=1$$. So with commuting counts and no energy condition, the content is classical (and is a classical sprinkling when the process is Lorentz invariant). The energy condition is what is new.

*Example: a Poisson sprinkling has no positive energy.* In the Koopman representation of a Poisson process of intensity $$\lambda$$, the centred vector $$\psi=(N(D)-\lambda V(D))\Omega$$ has $$\langle\psi,U(a)\psi\rangle=\mathrm{Cov}(N(D),N(D+a))=\lambda V(D\cap(D+a))$$, whose Fourier transform is the spectral measure $$d\mu_\psi=\lambda\lvert\hat 1_D(p)\rvert^2\,dp/(2\pi)^4$$ (total mass $$\lambda V(D)$$, by Parseval). It is symmetric and its density is non-zero for almost every $$p\in\mathbb R^4$$, so (P) fails as badly as it can. The same is true for the mixed Poisson states of the page on quantum sprinklings, whose centred counts have the same covariance up to the term $$\mathrm{Var}(\Lambda)V^2$$.

## 2. Commuting counts cannot sit in a positive-energy vacuum

<div class="theorem" markdown="1">

**Theorem 1** (no positive-energy count at non-zero density). Let $$(\mathcal H,U,\Omega,N)$$ satisfy (P) and (C1)–(C4) and (R). Then $$\rho=0$$. If the spectra lie in $$\mathbb N_0$$, then in addition $$N(D)\Omega=0$$ for every bounded Borel set $$D$$.

</div>

The idea is that (R) makes the autocorrelation $$\langle\psi,U(a)\psi\rangle$$ of the vector $$\psi=N(D)\Omega$$ a real function, hence even; positive energy then leaves only $$p=0$$ in its spectral measure, so $$N(D)\Omega$$ cannot depend on the position of $$D$$. A vector that does not depend on position, summed over $$m^4$$ cells, grows like $$m^4$$ in norm squared faster than its mean can follow, because the spectrum is integer.

<div class="proof" markdown="1">

*Proof.* *Step 1 (the spectral measure is symmetric).* Fix a bounded Borel $$D$$, let $$\psi=N(D)\Omega$$ and $$F(a)=\langle\psi,U(a)\psi\rangle$$. By (C3) and (C1), $$U(a)\psi=N(D+a)\Omega$$, so $$F(a)=\langle N(D)\Omega,N(D+a)\Omega\rangle$$, which is real by (R). Also $$F(-a)=\overline{F(a)}$$, hence $$F(-a)=F(a)$$. Write $$F(a)=\int e^{ip\cdot a}d\mu_\psi(p)$$ with the positive measure $$\mu_\psi=\langle\psi,E(\cdot)\psi\rangle$$, supported in $$\bar V_+$$ by (P). Then $$F(-a)=F(a)$$ for all $$a$$ says that the measures $$\mu_\psi$$ and its reflection agree, by uniqueness of the Fourier–Stieltjes transform. So $$\mu_\psi$$ is supported in $$\bar V_+\cap(-\bar V_+)=\{0\}$$, and $$\lVert U(a)\psi-\psi\rVert ^2=\int\lvert e^{ip\cdot a}-1\rvert^2d\mu_\psi=0$$. Therefore

$$
N(D+a)\Omega=N(D)\Omega\qquad\text{for all bounded Borel }D\text{ and all }a. \tag{1}
$$

*Step 2 (subdivision).* For an integer $$m\ge1$$ the unit cube $$Q$$ is the disjoint union of the $$m^4$$ translates $$Q_m+j/m$$, $$j\in\{0,\dots,m-1\}^4$$, of the cube $$Q_m=[0,1/m)^4$$. By (C4) (used finitely many times) and (1), $$N(Q)\Omega=m^4N(Q_m)\Omega$$. Taking the inner product with $$\Omega$$ gives $$\langle N(Q_m)\rangle=m^{-4}\rho$$. Since the spectrum is integer, $$n^2\ge\lvert n\rvert\ge n$$ on it, hence $$\lVert N(Q_m)\Omega\rVert ^2=\int n^2d\nu\ge\lvert\int n\,d\nu\rvert=\lvert\langle N(Q_m)\rangle\rvert$$, where $$\nu$$ is the spectral measure of $$N(Q_m)$$ in $$\Omega$$. Therefore

$$
\lVert N(Q)\Omega\rVert ^2=m^8\lVert N(Q_m)\Omega\rVert ^2\ \ge\ m^8\,m^{-4}\lvert\rho\rvert=m^4\lvert\rho\rvert. \tag{2}
$$

The left side does not depend on $$m$$ and is finite by (C2). Letting $$m\to\infty$$ gives $$\rho=0$$.

*Step 3 (non-negative spectra).* If the spectra lie in $$\mathbb N_0$$ then $$N(Q)\Omega$$ has $$\langle N(Q)\rangle=0$$, so the spectral measure of $$N(Q)$$ in $$\Omega$$ sits at $$n=0$$ and $$N(Q)\Omega=0$$. The same holds for every translate by (1). Any bounded $$D$$ lies in a finite union of translates $$Q+j$$, $$j\in\mathbb Z^4$$; on each cell $$N(Q+j)\Omega=N(D\cap(Q+j))\Omega+N((Q+j)\setminus D)\Omega=0$$ with both terms having non-negative mean, so both means vanish, and a non-negative integer-valued operator with zero mean annihilates $$\Omega$$. Summing over cells gives $$N(D)\Omega=0$$. $$\square$$

</div>

Lorentz invariance was not used, nor was commutation of counts at spacelike separation. Three remarks on the hypotheses. First, the vacuum $$\Omega$$ must be an invariant *vector*. The conclusion fails for a thermal state, whose GNS representation has translation generator of both signs; the statement is about ground states. Second, if $$\Omega$$ is replaced by a countable mixture of invariant vectors, each satisfying the hypotheses, then each component has $$\rho=0$$. A general translation-invariant density matrix need not be such a mixture. Third, the reality hypothesis is used in Step 1 and nowhere else; the smeared charge of a chiral current in section 3 has an autocorrelation with a non-zero imaginary part, and the corresponding vector is not translation invariant, but that example does not have integer spectrum.

<div class="corollary" markdown="1">

**Corollary 2** (classical sprinklings are not vacua). Let $$\Phi$$ be a translation-invariant point process with finite second moments and intensity $$\rho>0$$. There is no count system with commuting counts and positive energy (P) whose counts have the joint law of $$\Phi$$ in the vector $$\Omega$$.

</div>

Such a system would satisfy (C1)–(C4) and (R), so Theorem 1 would force $$\rho=0$$. The joint law of the counts fixes the vectors $$N(D)\Omega$$ and their inner products, hence $$U$$ on the cyclic subspace they generate (because $$U(a)N(D)\Omega=N(D+a)\Omega$$), so the spectral measure of $$N(D)\Omega$$ is the symmetric one computed in section 1 for the Poisson process, whatever the ambient representation.

The simple algebra below also yields a source of commuting counts.

<div class="proposition" markdown="1">

**Proposition 3** (projections with integer sum commute). Let $$P,Q$$ be orthogonal projections on a Hilbert space such that $$P+Q$$ has spectrum in $$\mathbb Z$$. Then $$PQ=QP$$.

</div>

<div class="proof" markdown="1">

*Proof.* Put $$S=P+Q$$. Since $$0\le S\le2$$, the spectrum lies in $$\{0,1,2\}$$ and $$S(S-1)(S-2)=0$$. With $$P^2=P$$, $$Q^2=Q$$ and $$T=PQ+QP$$ one has $$S^2=S+T$$, so $$S^3-3S^2+2S=(S-2)T$$ and $$(S-2)T=0$$. Expanding, $$(PQ+QP)(P+Q-2)=PQP+QPQ-PQ-QP=0$$. Multiplying on the left by $$P$$ gives $$PQPQ=PQ$$. So $$E=PQ$$ is an idempotent of norm at most 1. Such an operator is an orthogonal projection: for $$x\in\mathrm{ran}E$$, $$y\in\ker E$$ and any scalar $$t$$, $$E(x+ty)=x$$, so $$\lVert x\rVert \le\lVert x+ty\rVert $$ for all $$t$$, hence $$x\perp y$$. Thus $$PQ=(PQ)^*=QP$$. $$\square$$

</div>

The hypothesis that the operators are projections matters: with $$P$$ and $$Q$$ the projections onto two lines of $$\mathbb C^2$$ at an angle with $$\lvert\langle u,v\rangle\rvert=1/2$$, the operators $$X=2P$$ and $$Y=2Q$$ have spectrum $$\{0,2\}$$, their sum has spectrum $$\{1,3\}$$ (computed numerically: eigenvalues $$1.000,\,3.000$$), and $$[X,Y]$$ has eigenvalues $$\pm i\sqrt3$$. Integer spectrum and additivity alone do not force commutation, which is why Theorem 5 below does not assume it.

<div class="corollary" markdown="1">

**Corollary 4** (counts that are projections on small sets). Suppose the counts satisfy (C2), have operator additivity $$N(A\cup B)=N(A)+N(B)$$ for disjoint bounded sets, and every bounded Borel set is a finite disjoint union of sets $$A$$ for which $$N(A)$$ is a projection. Then all the $$N(D)$$ commute, so (R) holds.

</div>

<div class="proof" markdown="1">

*Proof.* If $$A,B$$ are disjoint with $$N(A),N(B)$$ projections, then $$N(A\cup B)=N(A)+N(B)$$ has integer spectrum by (C2), so the two commute by Proposition 3. For overlapping $$A,B$$ of this kind write $$A\cap B$$, $$A\setminus B$$ and $$B\setminus A$$ as finite disjoint unions of sets with projection counts. All the pieces are pairwise disjoint, so their counts commute pairwise, and $$N(A)$$ and $$N(B)$$ are sums of these. For general $$D,D'$$ decompose each into such sets. $$\square$$

</div>

The hypothesis holds for a point set with a hard core, in which every sufficiently small set contains at most one point and $$N(A)\in\{0,1\}$$. A Lorentz-invariant sprinkling has no hard core, but a point set that is invariant only under a discrete group can have one (a lattice with no null vectors, for example).

## 3. Non-commuting counts on a light ray

The reality hypothesis (R) is automatic when counts commute, and it fails for counts whose commutators have a non-zero expectation. The next theorem shows that on a light ray positive energy removes that freedom. A light-ray count system has $$U$$ a representation of $$\mathbb R$$ with $$U(a)=\int_{[0,\infty)}e^{ipa}dE(p)$$ (positive generator), (C1), and for every bounded half-open interval $$I=[s,t)$$ an operator $$N(I)$$ as in (C2)–(C4) with intervals in place of Borel sets; additivity is required for adjacent intervals. The weaker form of (R) is:

(R$$'$$) $$\langle N(I)\Omega,N(J)\Omega\rangle\in\mathbb R$$ whenever the closures $$\bar I$$ and $$\bar J$$ are disjoint.

Nothing is assumed for overlapping or adjacent intervals. In particular, strong commutation of $$N(I)$$ and $$N(J)$$ for intervals with disjoint closures implies (R$$'$$).

<div class="theorem" markdown="1">

**Theorem 5** (light-ray counts have zero density). A light-ray count system satisfying (R$$'$$) has $$\rho=\langle N([0,1))\rangle=0$$. If the spectra lie in $$\mathbb N_0$$, then $$N(I)\Omega=0$$ for all $$I$$.

</div>

The difficulty is that for overlapping intervals $$F(a)=\langle N(I)\Omega,N(I+a)\Omega\rangle$$ need not be real, and the imaginary part is the expectation of the commutator. The way around it is that this expectation is the same for every pair of adjacent pieces, because only a small neighbourhood of the shared endpoint can contribute. A constant imaginary part on an interval of $$a$$ values, vanishing for large $$a$$, contradicts the continuity of $$F$$ at 0 unless the constant is zero.

<div class="proof" markdown="1">

*Proof.* *Step 1 (edge universality).* For adjacent intervals $$A=[s,x)$$ and $$B=[x,t)$$ let $$c(A,B)=\mathrm{Im}\langle N(A)\Omega,N(B)\Omega\rangle$$. For $$0<\delta<\min(x-s,t-x)$$ split $$A=A_1\cup A_2$$ with $$A_2=[x-\delta,x)$$ and $$B=B_1\cup B_2$$ with $$B_1=[x,x+\delta)$$. By (C4), $$\langle N(A)\Omega,N(B)\Omega\rangle$$ is a sum of four terms. The pairs $$(A_1,B)$$ and $$(A_2,B_2)$$ have disjoint closures, so by (R$$'$$) those terms are real. Hence $$c(A,B)=\mathrm{Im}\langle N(A_2)\Omega,N(B_1)\Omega\rangle=:c_\delta(x)$$. Translation covariance makes $$c_\delta(x)$$ independent of $$x$$, and the identity $$c(A,B)=c_\delta$$ for all $$\delta$$ below the shorter length shows that $$c_\delta$$ is the same for all $$\delta$$. We write $$c_0$$ for this common value, so $$c(A,B)=c_0$$ for every pair of adjacent intervals.

*Step 2 (a plateau in the autocorrelation).* Fix $$L>0$$, $$I=[0,L)$$, $$\psi=N(I)\Omega$$ and $$F(a)=\langle\psi,U(a)\psi\rangle=\langle N(I)\Omega,N(I+a)\Omega\rangle$$. For $$0<a<L$$ write $$I=A\cup B$$ with $$A=[0,a)$$, $$B=[a,L)$$ and $$I+a=B\cup C$$ with $$C=[L,L+a)$$. Then $$F(a)=\langle N(A)\Omega,N(B)\Omega\rangle+\langle N(A)\Omega,N(C)\Omega\rangle+\lVert N(B)\Omega\rVert ^2+\langle N(B)\Omega,N(C)\Omega\rangle$$. The middle terms are real ($$\bar A\cap\bar C=\emptyset$$ since $$a<L$$), so $$\mathrm{Im}F(a)=c(A,B)+c(B,C)=2c_0$$. For $$a>L$$ the intervals have disjoint closures and $$\mathrm{Im}F(a)=0$$.

*Step 3 (continuity).* $$F$$ is continuous because $$U$$ is strongly continuous, and $$F(0)=\lVert \psi\rVert ^2$$ is real. Letting $$a\to0^+$$ in Step 2 gives $$2c_0=0$$. Hence $$\mathrm{Im}F(a)=0$$ for all $$a\ne\pm L$$, and by continuity for all $$a$$, because $$F(-a)=\overline{F(a)}$$.

*Step 4.* $$F$$ is real, so as in Theorem 1 the spectral measure of $$\psi$$ is symmetric. It is supported in $$[0,\infty)$$ by positivity of the generator, so it sits at $$0$$ and $$\psi$$ is invariant: $$N(I+a)\Omega=N(I)\Omega$$. The same argument applies to every length. Subdividing $$[0,L)$$ into $$2^k$$ intervals of length $$L2^{-k}$$ and using integrality exactly as in Step 2 of Theorem 1 gives $$\lVert N(I)\Omega\rVert ^2\ge 4^k\lvert\langle N(I_k)\rangle\rvert=2^k\lvert\langle N(I)\rangle\rvert$$ for all $$k$$, so $$\langle N(I)\rangle=0$$. The statement for $$\mathbb N_0$$-valued counts is Step 3 of Theorem 1. $$\square$$

</div>

*Example: the edge commutator is real.* The U(1) current of a chiral Dirac fermion has $$\langle J(x)J(y)\rangle=-1/(4\pi^2(x-y-i0)^2)$$ in its vacuum, which is $$\int_0^\infty\frac{p\,dp}{4\pi^2}e^{-ip(x-y)}$$: a positive spectral density on $$p>0$$. The charge of a smeared interval, $$Q_\varepsilon(I)=J(1_I*G_\varepsilon)$$ with a Gaussian $$G_\varepsilon$$ of width $$\varepsilon$$, has autocorrelation $$F_\varepsilon(a)=\int_0^\infty\frac{p\,dp}{4\pi^2}\,\frac{4\sin^2(pL/2)}{p^2}e^{-\varepsilon^2p^2}e^{-ipa}$$. For $$\varepsilon\to0$$,

$$
\lvert\mathrm{Im}F(a)\rvert=\frac1{\pi^2}\int_0^\infty\frac{\sin^2(pL/2)\sin(pa)}{p}\,dp=\begin{cases}\dfrac1{4\pi},&0<a<L,\\[4pt]0,&a>L,\end{cases}
$$

which is the plateau of Step 2, with $$c_0=1/(8\pi)$$ (consistent with $$[J(x),J(y)]=\frac{i}{2\pi}\delta'(x-y)$$ integrated over adjacent intervals, which gives the midpoint value $$i/4\pi$$). Numerically, at $$\varepsilon=0.005$$ and $$L=1$$ the imaginary part equals $$0.07958$$ at $$a=0.05,0.3,0.7,0.95$$ (the plateau value is $$1/4\pi=0.07958$$) and $$0.0000$$ at $$a=1.5$$; at $$\varepsilon=0.02$$ it rises from $$0.022$$ at $$a=0.01$$ to $$0.0796$$ at $$a=0.3$$. The imaginary part jumps from the plateau to 0 at $$a=0$$ as $$\varepsilon\to0$$, and correspondingly the variance diverges: $$F_\varepsilon(0)=\frac1{2\pi^2}\big(\ln(L/\varepsilon)+\gamma/2\big)$$ (checked numerically: $$0.2128$$ at $$\varepsilon=0.02$$, $$0.2830$$ at $$\varepsilon=0.005$$, both within $$10^{-4}$$ of the formula). The sharp charge is therefore not in the domain of the vacuum, and the continuity used in Step 3 fails. Integer-valued counts with finite second moments are the case where it holds.

*Charges.* In a Poincaré-invariant state, the expectation of a conserved current $$j^\mu$$ is a constant Lorentz vector, hence zero. A charge counted in a diamond has zero mean, so $$\mathbb N_0$$-valued charge counts vanish in the mean, and $$\mathbb Z$$-valued ones describe a neutral plasma, not a density of points.

*Four dimensions.* The proof of Theorem 5 applies to sets of the form $$I\times K$$, with $$I$$ a half-open interval in a null coordinate $$u$$ and $$K$$ a fixed bounded set in the other three coordinates, and translations along $$u$$ with positive generator $$n\cdot P\ge0$$ ($$n$$ the null vector), provided (R$$'$$) holds for $$I\times K$$ and $$J\times K$$ whenever the closures of $$I$$ and $$J$$ are disjoint. This is a statement about counts that are local along a null direction. It does not follow from Einstein causality: a region and its translate along $$u$$ contain timelike separated pairs for every shift.

## 4. Integer second-quantised counts factorise

For a field with one-particle space $$h$$, the count of a region is $$N(A)=d\Gamma(E(A))$$ with $$E(A)$$ a positive contraction on $$h$$ ($$0\le E(A)\le1$$), additive: $$E(A\cup B)=E(A)+E(B)$$. When $$E(A)$$ is multiplication by $$1_A$$ on $$L^2$$ of spacetime, this is the count of [Lorentz-invariant quantum sprinklings](/quantum-gravity/quantum-sprinklings/). A different $$E(A)$$, for instance a compression of $$1_A$$ to the positive-frequency subspace, would be natural for a positive-energy field.

<div class="proposition" markdown="1">

**Proposition 6** (integer counts force projections). Assume that $$N(A)=d\Gamma(E(A))$$ has integer spectrum for every bounded set $$A$$ and that $$E$$ is additive. Then every $$E(A)$$ is an orthogonal projection, $$E(A)E(B)=0$$ for disjoint $$A,B$$, the Fock space over $$h$$ is the tensor product of the Fock spaces over $$E(A_i)h$$ for any finite partition $$(A_i)$$ of a bounded set, and all the $$N(A)$$ commute.

</div>

<div class="proof" markdown="1">

*Proof.* The one-particle sector $$\Gamma^1(h)=h$$ reduces $$N(A)$$, and $$N(A)$$ acts on it as $$E(A)$$. The spectrum of a restriction to a reducing subspace is contained in the spectrum of the operator, so $$\mathrm{spec}\,E(A)\subset\mathbb Z\cap[0,1]=\{0,1\}$$, and $$E(A)$$ is a projection. For disjoint $$A,B$$, the operators $$E(A)$$, $$E(B)$$ and $$E(A\cup B)=E(A)+E(B)$$ are all projections, so $$(P+Q)^2=P+Q$$ gives $$PQ+QP=0$$. Multiplying by $$P$$ on the left and on the right gives $$PQ=-PQP=QP$$, hence $$PQ=0$$. Then $$h=\bigoplus_iE(A_i)h\oplus(1-E(\bigcup A_i))h$$, the exponential property $$\Gamma(h_1\oplus h_2)=\Gamma(h_1)\otimes\Gamma(h_2)$$ gives the tensor product, and $$d\Gamma(E(A_i))$$ acts on the $$i$$th factor only. $$\square$$

</div>

A non-projection $$E$$ gives non-integer counts: $$d\Gamma(\mathrm{diag}(1,0.3))$$ on $$\Gamma(\mathbb C^2)$$ has the eigenvalue $$0.3$$ on the one-particle state of the second coordinate. So integrality and additivity already imply the factorisation of the algebra over disjoint regions, and factorisation need not be imposed separately.

## 5. Finite-dimensional local algebras

Suppose each bounded open region $$O$$ of $$\mathbb R^4$$ carries an algebra $$\mathcal A(O)\subset B(\mathcal H)$$ containing the identity, with the following properties.

(N1) *Isotony:* $$\mathcal A(O)\subset\mathcal A(O')$$ if $$O\subset O'$$.

(N2) *Covariance:* $$U(a)\mathcal A(O)U(a)^*=\mathcal A(O+a)$$ with $$U$$ as in (P), and $$U(a)\Omega=\Omega$$.

(N3) *Weak additivity:* for every non-empty open $$O_1$$, the vectors $$B_1(a_1)\cdots B_m(a_m)\Omega$$, with $$m\ge0$$, $$a_i\in\mathbb R^4$$, $$B_i\in\mathcal A(O_1)$$ and $$B_i(a)=U(a)B_iU(a)^*$$, span a dense subspace of $$\mathcal H$$.

(N4) *Locality:* $$\mathcal A(O_1)$$ and $$\mathcal A(O_2)$$ commute when $$O_1$$ and $$O_2$$ are spacelike separated.

(N5) *Finite type I factors:* each $$\mathcal A(O)$$ for a double cone is isomorphic to a full matrix algebra $$M_{d(O)}$$.

<div class="theorem" markdown="1">

**Theorem 7** (finite-dimensional collapse of the Hilbert space). Assume (P), (N1)–(N3). If $$\dim\mathcal A(O_0)<\infty$$ for a single non-empty open $$O_0$$, then $$\dim\mathcal H\le\dim\mathcal A(O_0)$$. Locality (N4) is not used.

</div>

The idea is the Reeh–Schlieder argument: a vector orthogonal to the finite-dimensional space $$\mathcal A(O_0)\Omega$$ has matrix elements with products of translated local operators that vanish for small translations; positive energy makes those matrix elements boundary values of analytic functions of the translations, so they vanish for all translations, and by (N3) the vector is zero.

<div class="proof" markdown="1">

*Proof.* Choose a non-empty open $$O_1$$ and $$\varepsilon>0$$ with $$O_1+a\subset O_0$$ for $$\lvert a\rvert<\varepsilon$$. Let $$\psi\perp\mathcal A(O_0)\Omega$$; this space contains $$\Omega$$ because $$1\in\mathcal A(O_0)$$. For $$B_1,\dots,B_m\in\mathcal A(O_1)$$ let

$$
G(a_1,\dots,a_m)=\langle\psi,B_1(a_1)\cdots B_m(a_m)\Omega\rangle .
$$

If every $$\lvert a_i\rvert<\varepsilon$$ then $$B_i(a_i)\in\mathcal A(O_0)$$ by (N1) and (N2), so the product lies in $$\mathcal A(O_0)$$ and $$G=0$$. Using $$U\Omega=\Omega$$ we rewrite $$B_1(a_1)\cdots B_m(a_m)\Omega=U(a_1)B_1U(a_2-a_1)B_2\cdots U(a_m-a_{m-1})B_m\Omega$$, so in the variables $$x_0=a_1$$, $$x_j=a_{j+1}-a_j$$,

$$
G=\langle\psi,U(x_0)B_1U(x_1)B_2\cdots U(x_{m-1})B_m\Omega\rangle .
$$

For $$z=x+iy$$ with $$y$$ in the open forward cone, $$U(z)=\int e^{ip\cdot z}dE(p)$$ is a contraction, because $$\lvert e^{ip\cdot z}\rvert=e^{-p\cdot y}\le1$$ for $$p,y\in\bar V_+$$, and it is analytic in $$z$$ and strongly continuous up to $$y=0$$. Hence $$G$$ is the boundary value of a function $$\tilde G(z_0,\dots,z_{m-1})$$, bounded by $$\lVert \psi\rVert \prod\lVert B_i\rVert $$, analytic in the product of tubes $$\mathbb R^4+iV_+$$ and continuous up to the real boundary. It vanishes on an open box in $$\mathbb R^{4m}$$ (the set where all $$\lvert a_i\rvert<\varepsilon$$ is open in the $$x$$ coordinates).

We show that $$\tilde G\equiv0$$, one variable at a time. Fix the other variables at real points of the box and fix $$e\in V_+$$. The function $$\zeta\mapsto\tilde G(x_0+\zeta e,\dots)$$ on the upper half plane is bounded and analytic, continuous to the real axis and zero on an interval of the real axis. By the Schwarz reflection principle it extends analytically across that interval and is therefore zero. Letting $$e$$ and $$\zeta$$ vary, $$\tilde G=0$$ whenever $$\mathrm{Re}\,z_0$$ lies in the box and $$\mathrm{Im}\,z_0\in V_+$$; as the tube is connected, the identity theorem gives $$\tilde G=0$$ for every $$z_0$$ in the tube. Now fix $$z_0$$ arbitrary in the tube and repeat for $$z_1$$, and so on. After $$m$$ steps $$\tilde G\equiv0$$ and, taking boundary values, $$G\equiv0$$ for all real $$a_i$$.

Thus $$\psi$$ is orthogonal to all the vectors in (N3), including the case $$m=0$$ (the vector $$\Omega$$), so $$\psi=0$$. Therefore $$\mathcal A(O_0)\Omega=\mathcal H$$, and $$\dim\mathcal H\le\dim\mathcal A(O_0)$$. $$\square$$

</div>

The next result needs locality and the factor property, and no unitary implementation of the translations. Let $$D_r(x)$$ be the double cone over the ball of radius $$r/2$$ centred at $$x$$ at time 0, so that $$D_r$$ has tip separation $$r$$, and let $$d(D)$$ be the integer in (N5).

<div class="theorem" markdown="1">

**Theorem 8** (packing). Assume (N1), (N4), (N5) and translation covariance by automorphisms, $$\alpha_a(\mathcal A(O))=\mathcal A(O+a)$$ (unitarity is not needed). Then for $$\tau\ge4\sigma$$,

$$
\log_2d(D_\tau)\ \ge\ \Big(\frac{\tau}{5\sigma}\Big)^3\log_2d(D_\sigma). \tag{3}
$$

In particular if $$\mathcal A(D_\sigma)\ne\mathbb C$$ for arbitrarily small $$\sigma$$, then $$d(D_\tau)=\infty$$, contradicting (N5); and if $$\mathcal A(D_\sigma)\ne\mathbb C$$ for some $$\sigma$$, the logarithm of the dimension grows at least like the cube of the size, faster than any area law.

</div>

<div class="proof" markdown="1">

*Proof.* Place balls of radius $$\sigma/2$$ centred at the points $$\sigma k$$, $$k\in\mathbb Z^3$$, with $$\lvert\sigma k\rvert+\sigma/2\le\tau/2$$. Their double cones $$D_\sigma(\sigma k)$$ lie in $$D_\tau$$ (if a ball lies in a ball, so do the diamonds over them), and are pairwise spacelike separated: for points $$p,q$$ in two of them with centres at distance at least $$\sigma$$, $$\lvert t_p-t_q\rvert<\sigma-\lvert\vec p-\vec c_p\rvert-\lvert\vec q-\vec c_q\rvert\le\lvert\vec p-\vec q\rvert$$. By covariance each $$\mathcal A(D_\sigma(\sigma k))\cong M_{d(D_\sigma)}$$, and by (N4) these algebras commute. For commuting factors the map $$a_1\otimes\cdots\otimes a_n\mapsto a_1\cdots a_n$$ is a $$*$$-homomorphism from the tensor product, which is simple, and it is non-zero, hence injective. Its image lies in $$\mathcal A(D_\tau)\cong M_{d(D_\tau)}$$ by (N1), so $$d(D_\tau)^2\ge d(D_\sigma)^{2n}$$, where $$n$$ is the number of centres, i.e. $$d(D_\tau)\ge d(D_\sigma)^n$$. Choose centres in the cube $$\lvert k\rvert_\infty\le m$$ with $$m=\lfloor(\tau/\sigma-1)/(2\sqrt3)\rfloor$$; then $$n\ge(2m+1)^3\ge((\tau/\sigma-1)/(2\sqrt3))^3\ge(\tau/5\sigma)^3$$ for $$\tau\ge4\sigma$$ (using $$2\lfloor x\rfloor+1\ge x$$ and $$(\tau/\sigma-1)/(2\sqrt3)\ge 0.2165\,\tau/\sigma$$). $$\square$$

</div>

<div class="corollary" markdown="1">

**Corollary 9** (triviality). Under (P), (N1)–(N5), if some double cone has $$\mathcal A(D_\sigma)\ne\mathbb C\cdot1$$ then $$\dim\mathcal A(O_0)=\infty$$ for every open $$O_0$$. Equivalently, a net of finite-dimensional factors with exact positive-energy translations, weak additivity and locality has $$\mathcal A(O)=\mathbb C\cdot1$$ for every double cone.

</div>

<div class="proof" markdown="1">

*Proof.* If one $$\mathcal A(O_0)$$ were finite dimensional, Theorem 7 would give $$\dim\mathcal H<\infty$$ and hence $$d(D_\tau)^2\le(\dim\mathcal H)^2$$ for all $$\tau$$. By (3) this forces $$\log_2d(D_\sigma)=0$$ for all $$\sigma$$, so $$d=1$$. $$\square$$

</div>

If $$U$$ extends to a unitary representation $$V$$ of the Lorentz group on a finite-dimensional $$\mathcal H$$, then $$V$$ is trivial (a finite-dimensional unitary representation of $$SL(2,\mathbb C)$$ has compact image, and $$SL(2,\mathbb C)$$ has no non-trivial compact quotient), so covariance gives $$U(\Lambda a)=U(a)$$ for all $$\Lambda$$. The spectral measure of $$U$$ is then Lorentz invariant and, on a finite-dimensional space, has finitely many atoms; a non-zero momentum has an infinite Lorentz orbit, so (P) leaves $$p=0$$ only: the translations act trivially and the theory is static. Neither Lorentz invariance nor any statement about boosts enters Theorem 7 itself; the obstruction is positive energy.

*Remarks on the symmetry.* The stabiliser in the Poincaré group of a bounded double cone is compact, a group of rotations about its centre, and a compact group has non-trivial finite-dimensional unitary representations. So a finite-dimensional algebra attached to a *region* can carry a non-trivial rotation action, and finite dimension alone does not trivialise Lorentz symmetry region by region. Second, a site-based net, built as a tensor product of finite-dimensional factors over a locally finite subset $$S\subset\mathbb R^4$$, cannot be exactly covariant: a translation-invariant subset is empty or all of $$\mathbb R^4$$, and the orbit of any non-zero point under the Lorentz group meets a bounded set in a continuum of points, so no non-empty locally finite set is Poincaré invariant. Covariance can hold only in law, with a random set.

## 6. What the theorems do not exclude

Each route is stated with the hypothesis it drops.

- *Non-vacuum states.* Theorem 1 needs an invariant vector in a positive-energy representation. Thermal states, states of a field on an expanding background, and states that break translation invariance are outside it.
- *No energy condition.* Commuting counts without positive energy are exactly classical point processes (section 1). Theorem 1 says nothing about them; Haar-random lattices and Poisson processes are examples.
- *Counts whose two-point function is neither real nor local along a null direction.* In 1+1 or 3+1 dimensions, scalar volume counts that are Einstein-local but do not satisfy (R) or (R$$'$$) are not covered by Theorems 1 and 5. Whether positive energy, integrality and finite second moments are compatible with a non-zero density for such counts is not decided here. By Proposition 6 they cannot be second-quantised counts of a field, and by Corollary 4 they cannot be projections on small sets.
- *Unbounded or non-integer counts.* The sharp charge of a chiral current is not in the domain of the vacuum, and the smeared charge has continuous spectrum. Theorem 5 applies to neither.
- *Counts that are not second quantised.* Proposition 6 covers counts of the form $$d\Gamma(E(A))$$. Counts on an algebra that is not a Fock algebra are covered by Theorems 1 and 5 only through the hypotheses (R) or (R$$'$$).
- *Non-factorising algebras.* Algebras with a centre (gauge-type edge modes) factorise only in a modified sense. Theorem 7 does not use locality, so a non-factorising net with finite-dimensional local algebras, unitary positive-energy translations and weak additivity still lives on a Hilbert space of bounded dimension; Theorem 8 and Corollary 9 do use it.
- *Discrete translations.* A quantum cellular automaton on $$\mathbb Z^4$$ has translations only by integers; the spectrum of the translation generator lives on a torus, and (P) has no meaning. Theorem 7 does not apply.
- *A finite total Hilbert space.* Theorem 7 gives $$\dim\mathcal H<\infty$$; with locality and the factor property the net is then trivial (Corollary 9). Without locality, or without (N5), a non-trivial net on a finite-dimensional space is not excluded by these statements.
- *Non-net constructions.* Finite-dimensional diamonds that are not subalgebras of a common algebra on a common Hilbert space, with no net structure, are outside (N1)–(N3).

## 7. Open points

1. Whether Einstein-local integer counts with non-zero density and finite second moments can exist in a positive-energy representation when (R$$'$$) is not available (section 6, third item).
2. Whether the hypothesis (R) of Theorem 1 can be replaced by a bound on the imaginary part of the autocorrelation of the counts, using the Hilbert-pair relation between the real and imaginary parts of a positive-energy autocorrelation.
3. Whether weak additivity (N3) can be derived from (N1), (N2), (N4) and finite-dimensionality, which would remove it from Theorem 7.

## References

1. <a id="ref-rs"></a>Reeh, Schlieder, "Bemerkungen zur Unitäräquivalenz von Lorentzinvarianten Feldern," Nuovo Cimento 22 (1961) 1051–1068, [doi:10.1007/bf02787889](https://doi.org/10.1007/bf02787889). Used for: analytic continuation in the translations from the forward tube, the argument behind Theorem 7.
2. <a id="ref-hc"></a>Halvorson, Clifton, "No place for particles in relativistic quantum theories?," 2001, [arXiv:quant-ph/0103041](https://arxiv.org/abs/quant-ph/0103041). Used for: no-go theorems against localised particle observables in relativistic quantum theories, to which the integer-count statements are compared.
