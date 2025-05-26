---
published: true
layout: post
title: "Mechanistic Interpretability: Part 7 - Neural Network Circuits and their Taxonomy"
categories: machine-learning
date: 2025-05-25
---

Having explored how features might be represented ([Part 2]({% post_url 2025-05-25-mechanistic-interpretability-part-2 %}), [Part 4]({% post_url 2025-05-25-mechanistic-interpretability-part-4 %}), [Part 6]({% post_url 2025-05-25-mechanistic-interpretability-part-6 %})) and how to analyze information flow ([Part 3]({% post_url 2025-05-25-mechanistic-interpretability-part-3 %})), we now turn to how these features and information pathways combine to form **neural network circuits**. A circuit is a subgraph of the neural network—comprising features (neurons or dictionary elements) as nodes and weights (or effective connections) as edges—that implements a specific, identifiable algorithm or computation. This part introduces the concept of circuits, their hierarchical nature, and a taxonomy of common circuit motifs.

## Defining Neural Network Circuits

In mechanistic interpretability, a circuit is more than just an arbitrary collection of neurons. It is a specific computational mechanism hypothesized to perform a particular function within the larger network. Identifying and understanding these circuits is key to reverse-engineering how models perform tasks.

-   **Nodes:** Can be individual neurons, or more abstractly, monosemantic features derived via techniques like dictionary learning.
-   **Edges:** Represent the weighted connections between these nodes. In complex, multi-layer networks like Transformers, these are often *effective* or *virtual* weights (see [Part 3]({% post_url 2025-05-25-mechanistic-interpretability-part-3 %})) that summarize the influence of one node on another, potentially across several layers or components.
-   **Algorithm:** Each circuit is hypothesized to implement a specific algorithm (e.g., \"copy previous token,\" \"detect negated sentiment,\" \"identify repeated sequence\").

## The Hierarchy of Circuits

Circuits can exist at multiple scales, forming a hierarchy of computational abstraction:

1.  **Micro-circuits:** These are the smallest, most fundamental building blocks. They perform elementary operations.
    *   *Examples:* A single connection that copies a feature, a part of an attention head\'s OV circuit that transforms a feature in a specific way (e.g., $$\mathbf{W}_V \mathbf{W}_O$$ acting on a single input feature direction), a single MLP neuron acting as a detector for a simple pattern.

2.  **Meso-circuits (or Component Circuits):** These are formed by composing several micro-circuits to perform a more complex, but still relatively localized, sub-task. Often, an entire architectural component like an attention head or a small group of MLP neurons might form a meso-circuit.
    *   *Examples:* A full attention head implementing a specific attention pattern (e.g., a \"previous token head\" that attends to and copies the representation of the token at position $$t-1$$), a small ensemble of dictionary features and the weights connecting them that together implement a logical AND or OR operation on their inputs.

3.  **Macro-circuits:** These are larger-scale compositions, often spanning multiple layers and components, that implement significant, identifiable parts of the model\'s overall capability or a high-level algorithm.
    *   *Examples:* An **induction circuit** (detailed in [Part 9]({% post_url 2025-05-25-mechanistic-interpretability-part-9 %})) that enables in-context learning by composing multiple attention heads across layers, a circuit responsible for a specific type of multi-step reasoning.

Understanding often flows from identifying macro-level behaviors and then recursively decomposing them into their constituent meso- and micro-circuits.

## A Taxonomy of Common Circuit Motifs

Across different models and tasks, certain computational motifs or patterns appear repeatedly. Understanding these canonical circuit types provides a vocabulary for analyzing networks.

### 1. Copying/Memory Circuits
   - **Function:** Preserve or move information largely unchanged from one part of the network to another, or from one token position to another.
   - **Mathematical Signature:** In an attention head, if the OV circuit (defined by $$\mathbf{W}_V \mathbf{W}_O$$) is close to the identity matrix ($$\mathbf{W}_V \mathbf{W}_O \approx \mathbf{I}$$) for certain input directions, it effectively copies those input features. The QK circuit determines *what* gets copied *from where*.
   - **Example:** Previous token heads often use their OV circuit to copy the representation of the attended token to the current token\'s residual stream.

### 2. Inhibition Circuits
   - **Function:** Suppress or negate the influence of certain features or other circuits.
   - **Mathematical Signature:** Negative weights or effective weights. For example, if feature $$A$$ has a strong positive weight into a neuron, and feature $$B$$ has a strong negative weight into the same neuron, $$B$$ can inhibit the effect of $$A$$.
   - **Example:** A safety feature for toxicity might be inhibited by a feature indicating a fictional context.

### 3. Equivariant Circuits
   - **Function:** Respond consistently to transformations of the input. For example, permutation equivariance means that permuting the input tokens (in some way) leads to a corresponding permutation of the output features.
   - **Mathematical Signature:** The weight matrices of the circuit exhibit specific symmetries. Convolutional layers are a classic example of translation equivariant circuits.
   - **Example:** Some attention patterns might be (approximately) equivariant to shifting the positions of all tokens.

### 4. Union Circuits (OR-like behavior)
   - **Function:** Activate if *any* of a set of input features are present.
   - **Mathematical Signature:** Multiple input features $$f_1, f_2, \dots, f_k$$ have positive weights into a neuron or feature $$g$$, and a relatively low activation threshold for $$g$$.
   - **Example:** A feature for \"is a proper noun\" might be activated if a feature for \"is a person\" OR \"is a location\" OR \"is an organization\" is active.

### 5. Intersection Circuits (AND-like behavior)
    - **Function:** Activate only if *all* (or a specific combination) of a set of input features are present.
    - **Mathematical Signature:** This is often implemented by an MLP neuron. For example, an MLP neuron $$m = \text{ReLU}(w_1 f_1 + w_2 f_2 - b)$$ with positive $$w_1, w_2$$ and a bias $$b$$ greater than either $$w_1 f_1$$ or $$w_2 f_2$$ alone (but less than their sum) can act as an AND gate.
    - **Example:** A feature for \"is a capital city in Europe\" might require features for \"is a capital city\" AND \"is in Europe\" to be active.

### 6. Compositional Circuits (Sequential Processing)
   - **Function:** Combine the outputs of earlier circuits/features to compute more complex functions. This is the foundation of multi-step algorithms.
   - **Mathematical Signature:** As detailed in [Part 3]({% post_url 2025-05-25-mechanistic-interpretability-part-3 %}), Q-composition, K-composition, and V-composition in Transformers are prime examples. The output of head $$H_1$$ becomes part of the input to head $$H_2$$, allowing $$H_2$$'s computation to depend on $$H_1$$'s.
   - **Example:** Induction heads are a powerful example of compositional circuits.

## Mathematical Formalisms for Describing Circuits

-   **Graph Theory:** Circuits are naturally represented as graphs, with features/neurons as nodes and (effective) weights as directed, weighted edges.
-   **Path Expansion:** The overall effect of a circuit can be understood by summing the contributions of all paths through it. For a linear path $$\mathbf{x} \rightarrow \mathbf{W}_1 \rightarrow \mathbf{W}_2 \rightarrow \dots \rightarrow \mathbf{W}_k \rightarrow \mathbf{y}$$, the contribution is $$\mathbf{x} \mathbf{W}_1 \mathbf{W}_2 \dots \mathbf{W}_k$$.
-   **Virtual Weights:** Essential for analyzing circuits spanning multiple layers in residual networks like Transformers. The virtual weight $$\mathbf{W}_{I \rightarrow J}$$ from component $$I$$ to component $$J$$ summarizes the linear transformation along that path, accounting for all intermediate residual connections.

## Discovering and Validating Circuits

The process of identifying circuits is often iterative:
1.  Observe a model behavior or capability.
2.  Formulate a hypothesis about which components and pathways (i.e., a circuit) might be responsible.
3.  Use techniques like those in [Part 5]({% post_url 2025-05-25-mechanistic-interpretability-part-5 %}) (ablation, path patching, examining max activating examples) to test the hypothesis.
4.  Refine the hypothesized circuit based on validation results.

This iterative loop of hypothesis, prediction, and experimental validation is central to mechanistic interpretability.

## Conclusion

Viewing neural networks as collections of interacting computational circuits provides a powerful paradigm for understanding their internal mechanisms. By identifying these circuits, categorizing them based on common motifs, and understanding their hierarchical organization, we can begin to unravel the complex algorithms learned by these models. The mathematical tools of graph theory, path expansion, and virtual weights, combined with rigorous validation techniques, allow us to move from black-box observation to detailed mechanistic explanations.

Future parts will focus on specific, well-studied circuits in Transformers, such as those responsible for attention patterns and in-context learning.

In [Part 8 - Attention Head Circuits: Patterns and Functions]({% post_url 2025-05-25-mechanistic-interpretability-part-8 %})

---

## References and Further Reading

-   **Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S.** (2020). [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/). *Distill*.
-   **Cammarata, N., et al.** (2020). [Thread: Circuits](https://distill.pub/2020/circuits/). *Distill*.
-   **Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph,N., Mann, B., ... & Olah, C.** (2021). [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/). *Transformer Circuits Thread*.
-   **Wang, K., Varda, A., Nanda, N., ... & Steinhardt, J.** (2023). [Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593). 