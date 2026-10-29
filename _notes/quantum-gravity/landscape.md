---
title: "The landscape of quantum-gravity theories"
date: 2026-10-29
category: landscape
order: 1
description: A map of the approaches to quantum gravity that have been tested against the same list of requirements, with the theorems that exclude combinations of Lorentz invariance, locality, positivity, chiral matter, horizon thermodynamics and a law of change, and the routes that remain.
status: "A classification. Each exclusion it cites is a theorem with stated hypotheses, proved on the linked page; the choice of requirements and of axes is a modelling decision and not a theorem."
---

We ask which theories of quantum gravity can have, at the same time, the following properties: a low-energy limit with one limiting speed for gravitons, photons and matter; a massless graviton with universal coupling; a positive quantum state space; the chiral fermions of the Standard Model; the area law and the temperature of horizons; and a law of change that selects the geometry. The theorems collected in this wiki give a partial answer. For every family of theories examined, at least one of these properties is excluded under the hypotheses that define the family, or is available only at a stated cost, and for no family can all be shown to hold. The exclusions are narrow. Each is stated on its own page with its hypotheses and with the escape routes it leaves open, and the map below records which family loses which property and where the proof is.

The families are mostly built on one idea: spacetime is a random or regular set of points with some structure on it, and the continuum is recovered on average. The best-developed example is the causal set with its Poisson sprinkling [[Bombelli–Lee–Meyer–Sorkin]](#ref-blms), and a large part of the pages concerns it. The sprinkling carries the Benincasa–Dowker action [[Benincasa–Dowker]](#ref-bd) and a horizon count whose mean reproduces linearised Einstein gravity with the Bekenstein–Hawking coefficient. Alongside it we examine regular lattices, lattices with a preferred frame, finite-dimensional quantum models, continuum theories with indefinite propagators, and the routes by which gravity might fix a cosmological or dark-sector number. Strings and holography were not the object of any theorem here and are mentioned only where a theorem bears on them.

## 1. The requirements

Eight requirements define the columns of the map. They are chosen so that each can be tested by a calculation on a stated family.

1. **Lorentz invariance.** The limiting speeds of gravitational waves, light and matter agree, and photon birefringence is absent, to the accuracy of the observations. Gravitational waves and gamma rays from a neutron-star merger agree on the speed to about one part in $$10^{15}$$ [[GW170817]](#ref-gw170817).
2. **A graviton.** A massless spin-2 excitation that couples universally to energy and momentum, which at low energy gives general relativity coupled to the Standard Model.
3. **Locality.** The action is finite and local, or its non-locality is controlled.
4. **Positivity.** States and spectral weights are positive, so that probabilities are positive and the energy is bounded below.
5. **Chiral matter.** The chiral fermions and gauge fields of the Standard Model, with one species per field and no doublers.
6. **Horizon thermodynamics.** The entropy $$S=A/4G\hbar$$ with the correct coefficient, a horizon temperature $$\hbar\kappa/2\pi$$, and from them the Einstein equation through a first law.
7. **Dynamics.** A law of change, or a measure over histories, that selects a manifold-like geometry.
8. **A number.** Newton's constant in units of the discreteness scale, or a dark-sector quantity, computed instead of fitted.

## 2. The axes

The families differ in which of the following properties they keep. Each is defined where it first matters.

*Discreteness* means that spacetime is a locally finite set of events or a lattice. *Exact Lorentz invariance* means that the law of the point set, or the action, is invariant under the Lorentz group and not only approximately at long distance. *Positivity* means that the weights of the substrate are non-negative; we separate a positive *classical* point process, which is a probability measure on point sets, from a positive *quantum* state on an algebra of counts. A *fixed background* means that the manifold is an input and the point set is a random discretisation of it. The *place of locality* is where the dynamics is local: in spacetime, or only in some auxiliary space. The *finite-dimensional local space* axis asks whether each region carries a finite-dimensional Hilbert space.

The exact Lorentz group is non-compact, which is the source of most of the obstructions. A Lorentz-invariant measure of finite mass on a non-compact orbit does not exist, so an invariant random structure can carry no frame at a point, and a lattice that is invariant under a discrete subgroup mixes infrared and ultraviolet momenta. The pages on [invariant point sets](/quantum-gravity/invariant-point-sets/) and the [discrete Lorentz group](/quantum-gravity/discrete-lorentz-group/) make these statements precise.

## 3. The map

Table 1 lists the families in its rows and the requirements in its columns. A cell is empty when the combination was not examined. Otherwise it carries one of four words, each linked to the page that decides it:

- *excluded*: a theorem on the page rules the property out for the family, under the hypotheses stated there;
- *limited*: the property is available only under a stated condition or at a stated cost;
- *holds*: a construction or a theorem on the page achieves it in the regime stated there;
- *open*: examined and undecided.

A cell carries no more than the page proves. In particular *excluded* never means that the family is excluded as a whole.

| Family | Lorentz invariance | Graviton and locality | Positivity | Chiral matter | Horizon thermodynamics and dynamics | A number |
|---|---|---|---|---|---|---|
| Poisson sprinkling of a given manifold | holds in law; a single realisation is [limited](/quantum-gravity/per-realisation-kinematics/) | exact area law needs [non-local](/quantum-gravity/nonlocal-weights/) weights; the d'Alembertian is [excluded](/quantum-gravity/causal-box-spectrum/) from having positive spectral weight in four dimensions | area-law weights must be [signed](/quantum-gravity/area-projector/) | order-only kinetic terms [excluded](/quantum-gravity/order-only-fermions/); frame-free indices [excluded](/quantum-gravity/frame-free-chirality/); a Euclidean overlap with a Wick axis [holds](/quantum-gravity/frame-free-chirality/) | area law at linear order [holds](/quantum-gravity/area-projector/); the factor $$2\pi$$ is [not produced](/quantum-gravity/boundary-terms/) by the action; local rules [excluded](/quantum-gravity/no-local-dynamics/); path sums [open](/quantum-gravity/no-local-dynamics/) | [excluded](/quantum-gravity/no-number-from-kinematics/) |
| Quantised sprinkling (invariant Fock or CAR state of finite density) | [excluded](/quantum-gravity/quantum-sprinklings/) | | [excluded](/quantum-gravity/count-algebras/) | | | |
| Signed, complex or mixed-Poisson sprinkling | | normalisation of the 4D d'Alembertian [limited](/quantum-gravity/signed-substrates/) | [limited](/quantum-gravity/signed-substrates/) | | | |
| Invariant random lattices | holds in law; a single realisation [excluded](/quantum-gravity/per-realisation-kinematics/) for the cocompact lattice studied | the Poisson-weighted dictionary [fails](/quantum-gravity/invariant-point-sets/) | | a hand on the order [holds](/quantum-gravity/invariant-point-sets/) | | |
| Lattice with an exact discrete Lorentz group | [excluded](/quantum-gravity/discrete-lorentz-group/) | | | | | |
| Null-ray, twistor, complexified and frame-marked substrates | [excluded](/quantum-gravity/invariant-point-sets/) | | | | | |
| Lattice gravitons with lattice translations | a frame [leaks](/quantum-gravity/lorentz-violation-leakage/) | exact diffeomorphisms [excluded](/quantum-gravity/lattice-diffeomorphisms/); exact universal coupling [excluded](/quantum-gravity/spin-2-soft-locking/) | | | | |
| Gravity with a preferred frame (Hořava type) | [limited](/quantum-gravity/lorentz-violation-leakage/) | | | | | |
| Gravitons from local qubits, rotors or composite tetrads | | [excluded](/quantum-gravity/emergent-spin-2/) for the classes treated | | | | |
| Asymptotic safety with a positive propagator | | | dimensional reduction [excluded](/quantum-gravity/positivity-dimension-lemma/) | | | |
| Theories with a ghost or a fakeon | | | negative weight [needed](/quantum-gravity/positivity-dimension-lemma/) | | state-based entropy of the Krein vacuum [excluded](/quantum-gravity/indefinite-metric-entropy/) | |
| Induced gravity with a cutoff | | | | | | [excluded](/quantum-gravity/induced-gravity-number/) |

Table 1. The families examined against the requirements. The pages state the hypotheses of every cell.

The table has two features. The first is that the exclusions in the first three columns come from properties of the families that are shared by many of them: exact Lorentz invariance together with discreteness or positivity, or exact universality together with periodicity. The second is that the cells in the right-hand columns are mostly empty. Matter, horizon thermodynamics and dynamics have been examined for the sprinkling far more than for the other families, and for the sprinkling the question that remains open is a combination of the right-hand columns: the matter and dynamics requirements are simultaneously open.

## 4. How the theorems fit

The theorems fall into two groups by what they assume.

**Theorems about the Lorentz-invariant discrete substrate as such.** These hold for any discretisation that keeps the stated properties, and do not depend on the causal set. A Poincaré-invariant state of finite density on the bosonic Weyl algebra over spacetime is a mixture of coherent condensates, so counts are Poisson with a random intensity; no determinantal or number-squeezed state exists ([quantum sprinklings](/quantum-gravity/quantum-sprinklings/)). Integer-valued counts with positive-energy translations have zero density, whether or not they commute ([count algebras](/quantum-gravity/count-algebras/)). An exact, Zariski-dense arithmetic Lorentz group on a lattice allows no propagation ([discrete Lorentz group](/quantum-gravity/discrete-lorentz-group/)). Invariant random sets of null lines or hyperplanes with Radon mean measure do not exist ([invariant point sets](/quantum-gravity/invariant-point-sets/)). A Lorentz-invariant propagator with positive spectral weight has spectral dimension at least the dimension of space ([positivity and dimension](/quantum-gravity/positivity-dimension-lemma/)). Together they say that a substrate that is exactly Lorentz invariant, discrete and positive is Poisson-like, and that its observables are first moments. The exact statement of "Poisson-like" is narrower than it sounds: invariant random lattices exist, are not Poisson, and are classical, which is why the quantum statement and the classical statement are on different pages.

**Theorems about the sprinkling of a given manifold.** The remaining theorems concern the specific object, a Poisson sprinkling with order-only functionals. Kernels that reproduce the area law to all orders must be non-local ([non-local weights](/quantum-gravity/nonlocal-weights/)); the d'Alembertian built from them has negative spectral weight in four dimensions ([causal box](/quantum-gravity/causal-box-spectrum/)); a single realisation needs a mesoscale ([per-realisation kinematics](/quantum-gravity/per-realisation-kinematics/)); no local rule completes a field at sprinkled elements ([local dynamics](/quantum-gravity/no-local-dynamics/)); and the matter results exclude every frame-free chirality index that was tested ([order-only fermions](/quantum-gravity/order-only-fermions/), [frame-free chirality](/quantum-gravity/frame-free-chirality/), [frames and reflection positivity](/quantum-gravity/frames-and-reflection-positivity/)).

Every positive result in the sprinkling has the same form: the mean over the Poisson process of a functional of the sprinkling into a manifold $$M$$ equals a geometric quantity of $$M$$ plus corrections, and the functional is signed. The [area projector](/quantum-gravity/area-projector/) and the [boundary and corner terms](/quantum-gravity/boundary-terms/) are of this kind. The manifold is the input. The statements about the temperature, the first law and the non-linear Einstein equation depend on the choice of the matter state and on the assumptions of Jacobson's derivation, not on the sprinkling alone, and no page claims them.

## 5. Dropping exact Lorentz invariance

Lattice theories can keep locality and positivity by giving up exact Lorentz invariance. Two things then fail. Gravity on a lattice with lattice translations cannot have an exactly conserved lattice momentum with an analytic symbol together with low-energy graviton scattering ([lattice diffeomorphisms](/quantum-gravity/lattice-diffeomorphisms/)), and its soft graviton coupling cannot be exactly universal ([spin-2 soft locking](/quantum-gravity/spin-2-soft-locking/)). The word *exactly* carries both statements: universality to a finite order in the lattice spacing is possible, with violations of the equivalence principle suppressed by a power of the spacing. The second failure is the leakage of the preferred frame into the low-energy speeds: a frame in the matter regulator shifts the speed of light at one loop by an amount that does not depend on the cutoff, and no compact residual symmetry equalises the speeds ([Lorentz-violation leakage](/quantum-gravity/lorentz-violation-leakage/)). Graviton models built from local qubits, rotors or composite tetrads have their own obstructions ([emergent spin-2](/quantum-gravity/emergent-spin-2/)).

## 6. Dropping positivity, and theories beyond the point set

Giving up positivity at the substrate does not help in the cases examined: mixed-Poisson and signed processes normalise the four-dimensional d'Alembertian wrongly, finite-order perturbations of a Poisson process are exactly Poisson above a fixed number of points, and quantum-poset decorations keep the negative spectral weight ([signed substrates](/quantum-gravity/signed-substrates/)). The closure of the signed case depends on the form of the positivity axiom, and signed combinations of a Poisson process with an invariant lattice satisfy every form of it that was tried. Continuum theories pay for dimensional reduction with negative spectral weight, and the state-based entropy of a Krein vacuum is then unavailable ([positivity and dimension](/quantum-gravity/positivity-dimension-lemma/), [indefinite-metric entropy](/quantum-gravity/indefinite-metric-entropy/)). Induced gravity gives no pure number for Newton's constant ([induced gravity](/quantum-gravity/induced-gravity-number/)).

## 7. Routes to a number

A separate group of pages asks whether gravity can fix a measured number: the cosmological constant, the dark-matter density or mass, or the neutrino mass. Table 2 lists the routes examined and the page that closes each under its hypotheses.

| Target | Route | Page |
|---|---|---|
| Cosmological constant | linear global constraints, sequestering | [linear constraints](/quantum-gravity/linear-lambda-constraints/) |
| Cosmological constant | vacuum energy on an expanding background; horizon quantisations; log-coefficient matching | [dark-sector moduli](/quantum-gravity/dark-sector-moduli/) |
| Cosmological constant | volume degree of the causal-set horizon count | [no number from kinematics](/quantum-gravity/no-number-from-kinematics/) |
| Dark-matter to baryon ratio | any scheme with a free dark mass | [dark-sector moduli](/quantum-gravity/dark-sector-moduli/) |
| Sign of vacuum-matter exchange | equilibrium or passive vacuum-matter models | [vacuum-matter sign](/quantum-gravity/vacuum-matter-sign/) |
| Neutrino mass | an expansion history that mimics a negative mass | [neutrino mimic](/quantum-gravity/neutrino-mimic/) |
| Dark-matter coupling | dark-dimension radion | [dark-dimension coupling](/quantum-gravity/dark-dimension-coupling/) |
| Dark energy | Freezing Gravity with matter | [matter-induced instability](/quantum-gravity/matter-induced-instability/) |
| Primordial spectrum | holographic cosmology at large flavour number | [holographic cosmology](/quantum-gravity/holographic-cosmology/) |
| Non-Gaussianity | heavy-exchange inflation | [positivity wedge](/quantum-gravity/positivity-wedge/) |

Table 2. Routes to cosmological numbers and the pages that decide them.

None of the routes examined derives a measured number. Several leave a stated escape set, for example a frame-dependent weight, a spectator field during inflation, or off-branch initial data; the pages price each escape and do not choose among them.

## 8. Claims this map does not make

Some statements look natural and are not proved, or are false, and each has a precise replacement on the pages.

- It is not true that every Poincaré-invariant point process with finite density is Poisson. Invariant random lattices are not. What holds is the statement for bosonic Fock states ([quantum sprinklings](/quantum-gravity/quantum-sprinklings/)).
- It is not true that lattice gravity is excluded. What is excluded is exact universality of the soft coupling ([spin-2 soft locking](/quantum-gravity/spin-2-soft-locking/)).
- Poisson randomness does not by itself give kernel-scale kinematics for a single realisation. The lower bound on the fluctuation applies to the Poisson sprinkling too, and a mesoscale is needed ([per-realisation kinematics](/quantum-gravity/per-realisation-kinematics/)).
- The sign of the weighted horizon count in a single realisation carries no physical meaning; it follows from the extrapolation weights ([area projector](/quantum-gravity/area-projector/)).
- The Krein vacuum is not positive on any open region and has no state-based entropy, but a positive Poincaré-invariant state on the global algebra exists at the price of negative energy for the ghost ([indefinite-metric entropy](/quantum-gravity/indefinite-metric-entropy/)).
- The non-existence of invariant random sets is proved for null lines and null hyperplanes. For timelike and spacelike lines the obstruction is of a different kind and is not claimed ([invariant point sets](/quantum-gravity/invariant-point-sets/)).
- The temperature $$\hbar\kappa/2\pi$$ is not a theorem of the map. Whether a causal-set state is thermal at that temperature was tested numerically and is not decided.

## 9. What remains open

The following routes are not excluded by the theorems of this wiki. Each is stated with its conditions and without a ranking.

- **Chiral matter on a Poisson sprinkling.** A Euclidean overlap operator carries the index with a Wick axis in two dimensions, and averaged-law reflection positivity is undecided. A Lorentzian operator that is smeared on a mesoscale has not been tested. A compact internal index gives exact chirality bookkeeping but reduces to one-handed propagation in two dimensions ([frame-free chirality](/quantum-gravity/frame-free-chirality/), [frames and reflection positivity](/quantum-gravity/frames-and-reflection-positivity/)).
- **Lattice gravity with universality to finite order**, at the price of per-species tuning of the limiting speeds ([spin-2 soft locking](/quantum-gravity/spin-2-soft-locking/), [Lorentz-violation leakage](/quantum-gravity/lorentz-violation-leakage/)).
- **Laws of change that read joint pair data** (common pasts, interval contents, links), which the separate-statistics laws exclude ([local dynamics](/quantum-gravity/no-local-dynamics/)).
- **Counting observables that are neither commuting nor local along light rays**, in particular four-volume counts ([count algebras](/quantum-gravity/count-algebras/)).
- **Signed combinations of a Poisson process and an invariant lattice**, whose kernel mean has not been computed ([signed substrates](/quantum-gravity/signed-substrates/)).
- **Continuum theories** whose entropy is carried by a fakeon or by an infinite tower of states; the theorems on these pages exclude only the Krein reading and the positive-weight dimensional reduction.
- **Formulations that give up the Hilbert space.** No theorem here covers generalised probabilistic theories.

## References

1. <a id="ref-blms"></a>Bombelli, Lee, Meyer, Sorkin, “Space-time as a causal set,” Phys. Rev. Lett. 59 (1987) 521.
2. <a id="ref-bd"></a>Benincasa, Dowker, “The scalar curvature of a causal set,” [arXiv:1001.2725](https://arxiv.org/abs/1001.2725).
3. <a id="ref-gw170817"></a>LIGO Scientific Collaboration, Virgo Collaboration, Fermi Gamma-Ray Burst Monitor, INTEGRAL, “Gravitational waves and gamma-rays from a binary neutron star merger: GW170817 and GRB 170817A,” [arXiv:1710.05834](https://arxiv.org/abs/1710.05834).
