---
published: true
layout: post
title: "Mechanistic Interpretability: Part 1 - Foundations and the Circuits Paradigm"
categories: machine-learning
date: 2025-05-25
---

The quest to understand how neural networks arrive at their decisions is a central challenge in artificial intelligence. While many approaches treat these sophisticated models as "black boxes," focusing only on their input-output behavior, **mechanistic interpretability** embarks on a more ambitious journey: to reverse-engineer the internal algorithms learned by these networks. This series will delve into the theoretical underpinnings, mathematical formalisms, and core concepts that allow us to meticulously dissect and comprehend the computational mechanisms at play within modern AI.

## Beyond the Black Box: The Case for Mechanistic Understanding

Traditional interpretability techniques, such as saliency mapping or feature attribution, primarily offer *post-hoc explanations*. They attempt to rationalize a model's prediction without detailing the precise computational steps involved. While useful for providing high-level insights, these methods often fall short of revealing the underlying causal mechanisms. They can identify *what* parts of an input the model deems important, but not necessarily *how* those parts are processed and transformed into a decision.

This limitation becomes particularly salient when considering issues of safety, robustness, and reliability. If we don't understand the algorithm a model is executing, how can we confidently predict its behavior in novel situations, or guarantee it won't exhibit unintended or harmful behaviors?

Mechanistic interpretability proposes a paradigm shift. Instead of merely correlating inputs with outputs, it aims to identify and understand the specific, learned algorithms within the network. This approach treats neural networks not as inscrutable oracles, but as complex yet ultimately decipherable pieces of computational machinery. The core idea, inspired by fields like neuroscience and traditional reverse engineering, is that the functions of a network can be decomposed into smaller, understandable components and their interactions.

## The Circuits Hypothesis: Three Fundamental Claims

The "circuits" program, largely pioneered by researchers at OpenAI and Anthropic, provides a foundational framework for mechanistic interpretability. It rests on three fundamental claims about the internal workings of neural networks. If these claims hold true, they provide a strong basis for the systematic reverse engineering of these complex systems.

### Claim 1: Meaningful Features as Basic Units

The first claim posits that neural networks learn **meaningful features**. These features are not necessarily individual neurons but rather *directions in activation space*.

**Mathematical Formulation:**
Let $$\mathbf{a}^{(l)}(\mathbf{x})$$ be the activation vector of layer $$l$$ for an input $$\mathbf{x}$$. A feature $$f$$ is represented by a direction vector $$\mathbf{d}$$. The extent to which this feature is present for input $$\mathbf{x}$$ is given by its projection onto this direction:

$$f(\mathbf{x}) = \mathbf{d}^T \mathbf{a}^{(l)}(\mathbf{x})$$

In the simplest scenario, $$\mathbf{d}$$ could be a standard basis vector (e.g., $$\mathbf{e}_k$$ for the $$k$$-th neuron), meaning the feature corresponds directly to the activation of a single neuron. However, the claim is more general: features can be represented by linear combinations of neuron activations within a layer.

A feature is deemed "meaningful" or "interpretable" if it consistently responds to a semantically coherent property of the input that can be articulated in human-understandable terms. This could range from concrete visual elements like "vertical edges" or "curves of a certain radius" in vision models, to more abstract concepts like "negation" or "references to a specific entity" in language models.

The empirical validation for this claim involves observing consistent activation patterns for these feature directions across diverse inputs sharing the relevant semantic property and demonstrating a causal link between these activations and the network's outputs.

### Claim 2: Circuits as Meaningful Computational Subgraphs

The second claim builds upon the first: these meaningful features are connected to form **circuits**. A circuit is a computational subgraph—a specific arrangement of features and the weighted connections between them—that implements a particular algorithm or transformation.

**Mathematical Formulation:**
A circuit $$\mathcal{C}$$ can be conceptualized as a graph $$(V, E, W)$$, where:
-   $$V$$ is a set of features (as defined in Claim 1), potentially spanning multiple layers.
-   $$E \subseteq V \times V$$ is a set of directed edges representing connections between features, typically in adjacent or connected layers.
-   $$W: E \rightarrow \mathbb{R}$$ assigns weights to these edges, derived from the network's parameters.

If feature $$f_i$$ (with direction $$\mathbf{d}_i$$ in layer $$l$$) connects to feature $$f_j$$ (with direction $$\mathbf{d}_j$$ in layer $$l+1$$), the effective weight of this connection within the circuit can be expressed in terms of the network's weight matrix $$\mathbf{W}^{(l,l+1)}$$ connecting layer $$l$$ to $$l+1$$. Analyzing the computation pre-activation at layer $$l+1$$, the contribution of feature $$f_i$$ to feature $$f_j$$ is proportional to $$\mathbf{d}_j^T \mathbf{W}^{(l,l+1)} (\mathbf{d}_i f_i(\mathbf{x})) $$. Thus, the effective strength or weight $$W(f_i, f_j)$$ can be considered proportional to $$\mathbf{d}_j^T \mathbf{W}^{(l,l+1)} \mathbf{d}_i$$. More precise formulations depend on whether one considers the impact on the pre-activation of $$f_j$$ or its actual activation $$f_j(\mathbf{x})$$, and the specific activation functions involved.
(The exact formulation can vary depending on normalizations and specific circuit definitions).

These circuits can range in scale:
-   **Micro-circuits:** Small, localized subgraphs involving a few features, implementing basic computational primitives (e.g., a specific type of curve detector built from simpler edge detectors).
-   **Meso-circuits and Macro-circuits:** Larger, more complex assemblies of features and micro-circuits that implement more sophisticated functions or behaviors.

The idea is that complex network behaviors arise from the hierarchical composition of these simpler, identifiable circuits.

### Claim 3: Universality of Features and Circuits

The third claim is perhaps the most ambitious: that these meaningful features and circuits exhibit **universality**. This means that similar features and circuits tend to emerge across different model architectures, different training runs (with different random initializations), and even across different (but related) tasks.

If true, universality would imply that neural networks, when optimized for similar problems, tend to converge on similar computational solutions. This would be a powerful organizing principle, suggesting that there are common, perhaps even optimal, ways for neural networks to represent and process information derived from the structure of natural data or the nature of the tasks themselves.

Empirical evidence for universality involves identifying analogous features (e.g., curve detectors in various vision models) or circuits (e.g., specific attention patterns in different transformer models) and demonstrating their functional equivalence. This claim, if widely validated, would allow for the development of a taxonomy of neural network components, much like biologists classify cell types or organelles, significantly accelerating our understanding.

## The Scientific Methodology of Mechanistic Interpretability

Embracing mechanistic interpretability means adopting a rigorous scientific methodology. Interpretability claims are not merely subjective descriptions; they are testable hypotheses about the internal workings of a model.

**Falsifiability:** A core tenet, following Karl Popper, is that scientific claims must be falsifiable. An explanation for a circuit's function should lead to specific, testable predictions about how the network will behave under certain interventions or on novel inputs. If these predictions fail, the hypothesis is revised or rejected. This requires that claims are precise enough that contradictory evidence can be clearly identified.

**Operational Definitions:** Concepts like "feature" or "circuit" must be operationally defined in measurable terms. For example, claiming a feature "detects sadness" requires specifying quantifiable activation conditions related to inputs expressing sadness, and how "detection" is measured (e.g., activation strength, correlation with behavioral outputs). These definitions should be clear, consistent, and allow for empirical verification.

**Intervention-Based Validation:** The gold standard for establishing causality is intervention. If we hypothesize that a particular circuit implements a certain function, we should be able to demonstrate that directly manipulating that circuit (e.g., by ablating a feature, modifying a weight, or patching activations from another input) produces predictable changes in the network's output or internal state. This moves beyond mere correlation (observing that feature X activates when Y occurs) to causal understanding (demonstrating that activating X *causes* Y).

**Controlled Experiments:** Where possible, hypotheses should be tested using controlled experiments, such as presenting the model with synthetic stimuli designed to isolate the specific property a feature is thought to detect, while varying other factors. This helps disentangle confounds and confirm the feature's precise selectivity.

The journey of mechanistic interpretability is akin to the early days of biology when the microscope first allowed scientists to peer into the cell. Initial observations were qualitative and taxonomic, gradually leading to a deeper understanding of cellular machinery and eventually to fundamental theories. Similarly, by "zooming in" on the components of neural networks, we hope to uncover the principles of their learned algorithms.

## Looking Ahead

This introduction has laid out the conceptual terrain of mechanistic interpretability, contrasting it with black-box approaches and outlining the foundational "circuits" hypothesis. The core ambition is to move from opaque models to transparent, reverse-engineered computational systems.

The subsequent parts of this series will delve deeper into the challenges and breakthroughs in this field:
-   **[Part 2]({% post_url 2025-05-25-mechanistic-interpretability-part-2 %})**: The Superposition Hypothesis and Dictionary Learning – How networks represent more features than they have neurons and techniques for extracting meaningful features from superposition.
-   **[Part 3]({% post_url 2025-05-25-mechanistic-interpretability-part-3 %})**: Information Flow in Transformers – Mathematical framework for analyzing attention and composition in large language models.
-   **[Part 4]({% post_url 2025-05-25-mechanistic-interpretability-part-4 %})**: The Spectrum of Polysemanticity and Monosemanticity – Understanding the interpretability challenges posed by feature overlap and techniques to achieve cleaner representations.
-   **[Part 5]({% post_url 2025-05-25-mechanistic-interpretability-part-5 %})**: Validating Learned Features and Circuits – Rigorous methodologies for ensuring our interpretations are grounded in robust evidence.
-   **[Part 6]({% post_url 2025-05-25-mechanistic-interpretability-part-6 %})**: Neural Network Circuits: Taxonomy and Attention Patterns – A comprehensive exploration of circuit types and specialized attention mechanisms in Transformers.
-   **[Part 7]({% post_url 2025-05-25-mechanistic-interpretability-part-7 %})**: Induction Heads: The Mechanics of In-Context Learning – Detailed analysis of the remarkable circuits that enable Transformers to learn from context examples.

---

## References and Further Reading

This series draws heavily from the pioneering work published in Distill and the Transformer Circuits Thread, particularly:

-   Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S. (2020). [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/). *Distill*.
-   Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., ... & Olah, C. (2021). [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/). *Transformer Circuits Thread*.
-   The various articles on specific circuits (e.g., Curve Detectors, Induction Heads) available at [distill.pub/circuits](https://distill.pub/2020/circuits/) and [transformer-circuits.pub](https://transformer-circuits.pub/).
