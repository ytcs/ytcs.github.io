---
title: "The landscape of quantum-gravity theories"
date: 2026-10-29
category: landscape
order: 1
description: A map of the approaches to quantum gravity that have been tested against one list of requirements, with the theorems that exclude combinations of Lorentz invariance, locality, positivity, chiral matter, horizon thermodynamics and a law of change, and the routes that remain.
status: "A classification. Each exclusion it cites is a theorem with stated hypotheses, proved on the linked page; the list of requirements and the choice of axes are modelling decisions and not theorems."
---

We ask which theories of quantum gravity can have, at the same time, one limiting speed for gravitons, photons and matter; a massless graviton with universal coupling; a positive quantum state space; the chiral fermions of the Standard Model; the area law and temperature of horizons; and a law of change that selects the geometry. The theorems collected in this wiki answer a part of the question. For each family of theories examined, at least one of these properties is excluded under the hypotheses that define the family, or is available only at a stated cost, and for none of them can all be shown to hold. Each exclusion is narrow. It is stated on its own page with its hypotheses and with the routes it leaves open, and the map below records which family loses which property and where the proof is.

Most of the families start from the same idea: spacetime is a random or regular set of points carrying some structure, and the continuum is recovered on average. The best developed example is the causal set with its Poisson sprinkling [[Bombelli–Lee–Meyer–Sorkin]](#ref-blms). A sprinkling carries the Benincasa–Dowker action [[Benincasa–Dowker]](#ref-bd) and a horizon count whose expected value reproduces linearised Einstein gravity with the Bekenstein–Hawking coefficient, and much of the wiki concerns it. The other families are regular lattices, lattices with a preferred frame, finite-dimensional quantum models, continuum theories with indefinite propagators, and routes by which gravity might fix a cosmological number. Strings and holography are not the object of any theorem here.

## 1. The requirements

The columns of the map are eight requirements, chosen so that each can be tested by a calculation in a stated family.

1. Lorentz invariance: the limiting speeds of gravitational waves, light and matter agree, and photon birefringence is absent, to observational accuracy. Gravitational waves and gamma rays from a neutron-star merger agree on the speed to about one part in $$10^{15}$$ [[GW170817]](#ref-gw170817).
2. A graviton: a massless spin-2 excitation coupled universally to energy and momentum, giving general relativity coupled to the Standard Model at low energy.
3. Locality: the action is finite and local, or its non-locality is controlled.
4. Positivity: states and spectral weights are positive, so that probabilities are positive and the energy is bounded below.
5. Chiral matter: the chiral fermions and gauge fields of the Standard Model, one species per field and no doublers.
6. Horizon thermodynamics: the entropy $$S=A/4G\hbar$$ with the correct coefficient, a temperature $$\hbar\kappa/2\pi$$, and from them the Einstein equation through a first law.
7. Dynamics: a law of change, or a measure over histories, that selects a manifold-like geometry.
8. A number: Newton's constant in units of the discreteness scale, or a dark-sector quantity, computed instead of fitted.

## 2. The axes

The families differ in which of the following properties they keep. Discreteness means that spacetime is a locally finite set of events or a lattice. Exact Lorentz invariance means that the law of the point set, or the action, is invariant under the Lorentz group and not only approximately at long distance. Positivity is split into a positive classical point process, which is a probability measure on point sets, and a positive quantum state on an algebra of counts. A fixed background means that the manifold is an input and the point set is a random discretisation of it. The place of locality is the space in which the dynamics is local. A finite-dimensional local space means that each region carries a finite-dimensional Hilbert space.

The Lorentz group is not compact, and most obstructions come from this. A Lorentz-invariant measure of finite mass on a non-compact orbit does not exist, so an invariant random structure cannot carry a frame at a point; and a lattice that is invariant under a discrete subgroup mixes infrared and ultraviolet momenta. The pages on [Lorentz-invariant point sets](/quantum-gravity/invariant-point-sets/) and on [lattice translations with a discrete Lorentz group](/quantum-gravity/discrete-lorentz-group/) make both statements precise.

## 3. The map

Table 1 lists the families in its rows and the requirements in its columns. A cell is empty when the combination was not examined. Otherwise it carries a short phrase linked to the page that decides it: *excluded* means that a theorem on the page rules the property out for the family under the hypotheses stated there, *limited* that the property is available only under a stated condition or cost, *holds* that a construction or theorem on the page achieves it in the regime stated there, and *open* that the question was examined and is undecided. A cell carries no more than the page proves, and *excluded* never means that the family as a whole is excluded.

| Family | Lorentz invariance | Graviton and locality | Positivity | Chiral matter | Horizon thermodynamics and dynamics | A number |
|---|---|---|---|---|---|---|
| Poisson sprinkling of a given manifold | invariant in law; a single realisation at the kernel scale is [limited](/quantum-gravity/per-realisation-kinematics/) | the area law at every gradient order needs [heavy-tailed kernels](/quantum-gravity/nonlocal-weights/) (weights realising them are open); the 4D d'Alembertian has [no positive spectral representation](/quantum-gravity/causal-box-spectrum/) | area-law weights must be [signed](/quantum-gravity/area-projector/) | covariant order-only kinetic terms [excluded](/quantum-gravity/order-only-fermions/); several frame-free chirality indices [excluded](/quantum-gravity/frame-free-chirality/); a Euclidean overlap with a Wick axis [holds](/quantum-gravity/frame-free-chirality/) as a numerical observation | area law at linear order [holds](/quantum-gravity/area-projector/); the leading joint term of the action is [proportional to coth](/quantum-gravity/boundary-terms/); exact wave-consistent completion rules [excluded](/quantum-gravity/no-local-dynamics/); path sums [open](/quantum-gravity/no-local-dynamics/) | scale and entropy unit [free](/quantum-gravity/no-number-from-kinematics/) |
| Quantised sprinkling (invariant bosonic or fermionic state of finite density) | | | [excluded](/quantum-gravity/quantum-sprinklings/); counts of a positive-energy field have [zero density](/quantum-gravity/count-algebras/) under stated conditions | | | |
| Signed, complex or mixed-Poisson sprinkling | | normalisation of the 4D d'Alembertian [limited](/quantum-gravity/signed-substrates/) | Born reading [classical](/quantum-gravity/signed-substrates/) | | | |
| Invariant random lattices | invariant in law; one cocompact lattice [fails](/quantum-gravity/per-realisation-kinematics/) for single realisations | Poisson-weighted dictionary [fails](/quantum-gravity/invariant-point-sets/) | | a Poincaré-invariant ergodic lattice with a hand [exists](/quantum-gravity/invariant-point-sets/) | | |
| Lattice with an exact discrete Lorentz group and lattice translations | propagation [excluded](/quantum-gravity/discrete-lorentz-group/) under stated conditions | | | | | |
| Null-line, complexified and frame-marked substrates | invariant processes [excluded](/quantum-gravity/invariant-point-sets/) | | | | | |
| Lattice gravitons with lattice translations | a frame [leaks](/quantum-gravity/lorentz-violation-leakage/) | analytic conserved momentum [excluded](/quantum-gravity/lattice-diffeomorphisms/); exact soft coupling [excluded](/quantum-gravity/spin-2-soft-locking/) under stated conditions | | | | |
| Gravity with a preferred frame (Hořava type) | speeds [not forced equal](/quantum-gravity/lorentz-violation-leakage/) | | | | | |
| Gravitons from local models | | harmonic, analytic, stable class [excluded](/quantum-gravity/emergent-spin-2/) | | | | |
| Asymptotic safety with a positive propagator | | | dimensional reduction [excluded](/quantum-gravity/positivity-dimension-lemma/) within a stated class | | | |
| Theories with a ghost | | | negative weight [needed](/quantum-gravity/positivity-dimension-lemma/) within that class | | entropy of the Lorentz-invariant Krein vacuum [excluded](/quantum-gravity/indefinite-metric-entropy/) | |
| Induced gravity with a cutoff | | | | | | regulator [dependent](/quantum-gravity/induced-gravity-number/) |

Table 1. The families examined against the requirements. The pages state the hypotheses of every cell.

Two features stand out. The exclusions in the first three columns come from properties that many families share: exact Lorentz invariance with discreteness or positivity, or exact universality with periodicity. And the right-hand columns are mostly empty: matter, horizon thermodynamics and dynamics have been examined far more for the sprinkling than for the other families, and for the sprinkling the open question is a combination of those columns.

## 4. How the theorems fit

The theorems fall into two groups, by what they assume.

The first group concerns the Lorentz-invariant discrete substrate as such and does not use the causal set. A Poincaré-invariant, locally normal state of finite density on the bosonic Weyl algebra over spacetime is a mixture of coherent condensates, so counts are Poisson with a random intensity, and no determinantal state exists ([quantum sprinklings](/quantum-gravity/quantum-sprinklings/)). Integer-valued counts with positive-energy translations have zero density under the conditions stated on the [count algebras](/quantum-gravity/count-algebras/) page. Exact lattice translations together with a Zariski-dense arithmetic Lorentz group leave no dispersion relation ([discrete Lorentz group](/quantum-gravity/discrete-lorentz-group/)). Poincaré-invariant random sets of null lines or hyperplanes with a Radon mean measure are almost surely empty ([invariant point sets](/quantum-gravity/invariant-point-sets/)). For the inverse of a positive-weight propagator taken as a diffusion generator, the spectral dimension is at least that of space ([positivity and dimension](/quantum-gravity/positivity-dimension-lemma/)). Together these say that an exactly Lorentz-invariant, discrete, positive substrate has Poisson-like counts. The qualifier matters: invariant random lattices exist, are classical and are not Poisson, so the quantum statement and the classical one are on different pages.

The second group concerns the sprinkling of a given manifold with functionals of the order. A horizon count that follows the area law at every gradient order needs a kernel with a heavy tail and infinitely many sign changes under hypothesis (H) of the [kernel page](/quantum-gravity/nonlocal-weights/). The four-dimensional d'Alembertian has [no positive spectral representation](/quantum-gravity/causal-box-spectrum/). A single realisation needs a mesoscale ([per-realisation kinematics](/quantum-gravity/per-realisation-kinematics/)). No exactly wave-consistent completion rule exists on finitely many sprinkled elements ([local dynamics](/quantum-gravity/no-local-dynamics/)). The matter pages show that covariant order-only kinetic terms do not exist ([order-only fermions](/quantum-gravity/order-only-fermions/)), that three classes of frame-free overlap operators carry no chirality index ([chirality obstructions](/quantum-gravity/frame-free-chirality/)), and that frames, Wilson terms and reflection positivity each cost something ([frames and reflection positivity](/quantum-gravity/frames-and-reflection-positivity/)).

The positive results of the sprinkling have one form: the expected value of a functional of the sprinkling into a manifold $$M$$ equals a geometric quantity of $$M$$ plus corrections, and the functional is signed. The [area projector](/quantum-gravity/area-projector/) and the [boundary and corner terms](/quantum-gravity/boundary-terms/) are of this kind. The manifold is the input. The temperature, the first law and the non-linear Einstein equation depend on the matter state and on the assumptions of Jacobson's derivation, not on the sprinkling alone, and no page claims them.

## 5. Dropping exact Lorentz invariance

Lattice theories can keep locality and positivity by giving up exact Lorentz invariance. A lattice graviton theory in continuous time cannot have a conserved lattice momentum with an analytic symbol together with low-energy graviton scattering ([a conserved lattice momentum](/quantum-gravity/lattice-diffeomorphisms/)). Its soft coupling cannot be exactly universal when the symbol is a finite-dimensional real-analytic periodic one and the soft expansion has the stated form ([universal coupling](/quantum-gravity/spin-2-soft-locking/)). On that page the necessary algebraic condition fails at a stated relative order in the lattice spacing, and whether a lattice amplitude that is universal to finite order exists is undecided. A preferred frame leaks into the low-energy speeds: nothing forces the graviton, photon and matter speeds to coincide when the residual symmetry group is compact, a frame in a scalar matter regulator shifts the speed at one loop by a cutoff-independent amount, and stabiliser-code distance does not equalise speeds ([frame leakage](/quantum-gravity/lorentz-violation-leakage/)). Graviton models built from local degrees of freedom have their own dispersion constraints within the harmonic, analytic and stable class ([emergent gravitons](/quantum-gravity/emergent-spin-2/)).

## 6. Dropping positivity, and theories beyond the point set

Giving up positivity at the substrate does not help in the cases examined. A mixed-Poisson sprinkling normalises the four-dimensional d'Alembertian with a factor below one; a Born reading of a signed or complex process forces it to be classical in the cases treated; finite-order perturbations of a Poisson process are exactly Poisson above a fixed number of points under one form of the positivity axiom; and the quantum-poset kernels satisfying the stated conditions keep the obstruction ([signed substrates](/quantum-gravity/signed-substrates/)). Signed combinations of a Poisson process with an invariant lattice satisfy every form of the axiom that was tried in law, and their mean operator is open. In continuum theories, dimensional reduction within the stated class needs negative weight, and the entropy of the Lorentz-invariant, positive-energy Krein vacuum is unavailable ([entropy with indefinite metric](/quantum-gravity/indefinite-metric-entropy/)). Induced gravity gives a Newton constant that depends on the regulator shape ([induced Newton constant](/quantum-gravity/induced-gravity-number/)).

## 7. Routes to a number

A separate group of pages asks whether gravity can fix a measured number: the cosmological constant, the dark-matter density or mass, or the neutrino mass. Table 2 lists the routes examined and the page that closes each under its hypotheses.

| Target | Route | Page |
|---|---|---|
| Cosmological constant | linear global constraints, sequestering | [sign and drift of linear constraints](/quantum-gravity/linear-lambda-constraints/) |
| Cosmological constant | vacuum energy on an expanding background; horizon quantisations; log-coefficient matching | [structures that leave it undetermined](/quantum-gravity/dark-sector-moduli/) |
| Cosmological constant | volume degree of the causal-set horizon count | [counts and dimensionless constants](/quantum-gravity/no-number-from-kinematics/), conformally static backgrounds |
| Dark-matter to baryon ratio | any scheme with a free dark mass | [structures that leave it undetermined](/quantum-gravity/dark-sector-moduli/) |
| Sign of vacuum-matter exchange | equilibrium or passive vacuum-matter models | [sign of the exchange](/quantum-gravity/vacuum-matter-sign/) |
| Neutrino mass | an expansion history that mimics a negative mass | [negative neutrino mass in BAO data](/quantum-gravity/neutrino-mimic/) |
| Dark-matter coupling | dark-dimension radion and the minimal model | [dark-dimension coupling](/quantum-gravity/dark-dimension-coupling/) |
| Dark energy | Freezing Gravity with matter | [matter-induced instability](/quantum-gravity/matter-induced-instability/) |
| Primordial spectrum | holographic cosmology at large flavour number | [low multipoles](/quantum-gravity/holographic-cosmology/) |
| Non-Gaussianity | heavy-exchange inflation | [positivity wedge](/quantum-gravity/positivity-wedge/) |

Table 2. Routes to cosmological numbers and the pages that decide them.

None of the routes examined derives a measured number. Several leave an escape set that the page prices without choosing among its members, for example a frame-dependent weight, a spectator field during inflation, or off-branch initial data.

## 8. Statements this map does not make

Some natural statements are false or unproved, and each has a precise replacement on the pages.

- Not every Poincaré-invariant point process of finite density is Poisson: invariant random lattices are not. The statement holds for bosonic Fock states of finite density with sharp counts ([quantum sprinklings](/quantum-gravity/quantum-sprinklings/)).
- Lattice gravity is not excluded. What is excluded is an exactly universal soft coupling for the symbols treated, and an analytic conserved lattice momentum ([universal coupling](/quantum-gravity/spin-2-soft-locking/)).
- Poisson randomness does not by itself give kernel-scale kinematics for a single realisation: the fluctuation lower bound applies to the Poisson sprinkling too, and a mesoscale is needed ([per-realisation kinematics](/quantum-gravity/per-realisation-kinematics/)).
- The sign of the weighted horizon count in a single sprinkling is not fixed by the mean; it follows from the extrapolation weights ([area projector](/quantum-gravity/area-projector/)).
- The Krein vacuum has no state-based entropy only for the Lorentz-invariant, positive-energy vacuum; a positive Poincaré-invariant state on the global algebra exists at the price of negative energy for the ghost ([entropy with indefinite metric](/quantum-gravity/indefinite-metric-entropy/)).
- Invariant random sets of null lines or hyperplanes with a Radon mean measure are empty; for timelike and spacelike lines an invariant measure exists with infinite mass on sets meeting any open region ([invariant point sets](/quantum-gravity/invariant-point-sets/)).
- The temperature $$\hbar\kappa/2\pi$$ is not a theorem of the map; whether a causal-set state is thermal at that temperature was tested numerically and is undecided.
- The factor $$2\pi$$ of the area law is not derived from the action. The page on [boundary terms](/quantum-gravity/boundary-terms/) proves that the leading joint term of an interval functional is proportional to $$\coth\theta$$ at flat joints, with the half-angle function absent from that term.

## 9. What remains open

The following routes are not excluded by the theorems. Each is stated with its conditions.

- Chiral matter on a Poisson sprinkling. A Euclidean overlap operator carries the index with a Wick axis in two dimensions, averaged-law reflection positivity is undecided, and a Lorentzian operator smeared on a mesoscale has not been tested ([chirality obstructions](/quantum-gravity/frame-free-chirality/), [frames and reflection positivity](/quantum-gravity/frames-and-reflection-positivity/)).
- Lattice gravity whose algebraic universality condition holds to finite order, at the price of per-species tuning of the limiting speeds ([universal coupling](/quantum-gravity/spin-2-soft-locking/), [frame leakage](/quantum-gravity/lorentz-violation-leakage/)).
- Laws of change that read joint pair data, such as common pasts and interval contents; the separate-statistics laws are treated only on orders with a symmetry of the stated kind ([local dynamics](/quantum-gravity/no-local-dynamics/)).
- Counting observables that are neither commuting nor local along light rays, in particular four-volume counts ([count algebras](/quantum-gravity/count-algebras/)).
- Signed combinations of a Poisson process and an invariant lattice, whose kernel mean is not computed ([signed substrates](/quantum-gravity/signed-substrates/)).
- Continuum theories with a fakeon or an infinite tower of states, which the Krein and positive-weight theorems do not cover.
- Formulations without a Hilbert space; no theorem here applies.

## References

1. <a id="ref-blms"></a>Bombelli, Lee, Meyer, Sorkin, “Space-time as a causal set,” Phys. Rev. Lett. 59 (1987) 521.
2. <a id="ref-bd"></a>Benincasa, Dowker, “The scalar curvature of a causal set,” [arXiv:1001.2725](https://arxiv.org/abs/1001.2725).
3. <a id="ref-gw170817"></a>LIGO Scientific Collaboration, Virgo Collaboration, Fermi Gamma-Ray Burst Monitor, INTEGRAL, “Gravitational waves and gamma-rays from a binary neutron star merger: GW170817 and GRB 170817A,” [arXiv:1710.05834](https://arxiv.org/abs/1710.05834).
