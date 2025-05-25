---
published: true
layout: post
title: "Mechanistic Interpretability: Part 3 - Mathematical Framework for Transformer Circuits"
categories: machine-learning
date: 2025-05-25
---

In [Part 1]({% post_url 2025-05-25-mechanistic-interpretability-part-1 %}) and [Part 2]({% post_url 2025-05-25-mechanistic-interpretability-part-2 %}), we established the foundational principles of mechanistic interpretability and explored the superposition hypothesis. Now we turn to the mathematical framework that enables systematic analysis of transformer circuits—the computational subgraphs that implement specific algorithms within modern language models. This framework represents a fundamental shift from treating transformers as black boxes to understanding them as compositions of interpretable computational components.

## The Transformer Decomposition Challenge

Transformer models present unique challenges for mechanistic interpretability that distinguish them from the vision models analyzed in earlier circuits work:

**Architectural Complexity:** Transformers employ sophisticated attention mechanisms creating complex, context-dependent information routing patterns, making it difficult to trace computational pathways through static analysis alone.

**Residual Connections:** The pervasive use of residual connections creates a linear communication channel enabling complex interactions between distant layers, requiring new mathematical frameworks to analyze information flow.

**Multi-Head Attention:** The parallel operation of multiple attention heads within each layer creates a high-dimensional space of possible interactions that must be systematically decomposed and analyzed.

**Sequence Dependencies:** Unlike vision models processing fixed-size inputs, transformers operate on variable-length sequences with complex positional dependencies affecting circuit behavior.

### Methodological Principles

The mathematical framework developed here adheres to several key principles ensuring both theoretical rigor and practical applicability:

**Linearity Exploitation:** Transformers contain extensive linear structure that can be leveraged for exact mathematical analysis, particularly in the residual stream and attention computations.

**Compositional Decomposition:** Complex behaviors emerge from composition of simpler computational primitives, enabling hierarchical analysis from individual components to full circuits.

**Path-Based Analysis:** Model computations can be decomposed into interpretable paths tracing information flow from inputs to outputs through specific sequences of components.

**Virtual Weight Computation:** The linear structure of residual connections enables computation of effective weights between non-adjacent components, revealing implicit computational relationships.

## Transformer Architecture Decomposition

### Architectural Overview

A transformer language model consists of sequential components:

**Token Embedding:** $$\mathbf{E} \in \mathbb{R}^{d_{\text{vocab}} \times d_{\text{model}}}$$ maps discrete tokens to continuous vector representations.

**Positional Encoding:** $$\mathbf{P} \in \mathbb{R}^{n_{\text{ctx}} \times d_{\text{model}}}$$ provides position-dependent information.

**Transformer Blocks:** A sequence of $$L$$ identical blocks, each containing:
- Multi-head attention layer with $$H$$ heads
- Feed-forward MLP layer  
- Residual connections and layer normalization

**Output Unembedding:** $$\mathbf{U} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{vocab}}}$$ maps final representations to vocabulary logits.

### Simplified Model Assumptions

To develop the clearest possible mathematical framework, we initially focus on simplified transformer variants retaining essential computational structure while removing certain complexities:

**Attention-Only Models:** We primarily analyze transformers without MLP layers, focusing on novel challenges presented by attention mechanisms. This simplification allows more elegant mathematical treatment while capturing core transformer innovations.

**No Biases:** Bias terms are omitted for mathematical clarity. Any model with biases can be equivalently represented without them by augmenting input with a constant dimension.

**No Layer Normalization:** Layer normalization is initially ignored to focus on core linear structure. Up to scaling factors, layer normalization can often be absorbed into adjacent weight matrices.

These simplifications preserve essential computational structure while enabling exact mathematical analysis. Insights from simplified models provide the foundation for understanding more complex variants.

### Residual Stream as Central Communication Channel

The residual stream represents the most distinctive architectural feature of transformers, serving as a linear communication channel fundamentally shaping information flow.

**Mathematical Definition:** At layer $$\ell$$, the residual stream $$\mathbf{x}^{(\ell)} \in \mathbb{R}^{n_{\text{ctx}} \times d_{\text{model}}}$$ is defined recursively:

$$\mathbf{x}^{(\ell)} = \mathbf{x}^{(\ell-1)} + f^{(\ell)}(\mathbf{x}^{(\ell-1)})$$

where $$f^{(\ell)}$$ represents the transformation applied by layer $$\ell$$.

**Linear Structure:** The residual stream maintains purely linear, additive structure throughout the model. Each layer reads from the residual stream via linear transformation, processes information, and writes back via another linear transformation:

$$\text{Read:} \quad \mathbf{h}^{(\ell)} = \mathbf{x}^{(\ell-1)} \mathbf{W}_{\text{in}}^{(\ell)}$$

$$\text{Write:} \quad \mathbf{x}^{(\ell)} = \mathbf{x}^{(\ell-1)} + \mathbf{h}^{(\ell)} \mathbf{W}_{\text{out}}^{(\ell)}$$

**Non-Privileged Basis:** The residual stream does not have a privileged basis—any orthogonal transformation applied consistently to all interacting weight matrices leaves model behavior unchanged. This property has important implications for interpretability analysis.

**Communication Bandwidth:** The residual stream serves as a communication bottleneck, with dimensionality $$d_{\text{model}}$$ typically much smaller than the total number of computational units that must communicate through it. This creates a superposition-like scenario where multiple information streams must share representational space.

## Virtual Weights and Effective Connectivity

The linear structure of the residual stream enables computation of virtual weights capturing effective connectivity between non-adjacent components, providing a powerful tool for understanding long-range dependencies. Think of the residual stream as a highway where information from different components (cars) can be added. A component later in the network doesn't just see the output of the immediately preceding component; it sees the sum of outputs from *all* previous components that wrote to the residual stream. Virtual weights help us quantify the total influence of one component's output on another component's input, considering all the additions that happened in between.

### Virtual Weight Computation

**Definition:** Virtual weights $$\mathbf{W}_{\text{virtual}}^{(i,j)}$$ represent the effective linear transformation from component $$i$$ output to component $$j$$ input, computed by multiplying relevant weight matrices through the residual stream.

**Mathematical Formulation:** For components in layers $$i < j$$, the simplest virtual weight considers only the direct write from component $$i$$ and the direct read by component $$j$$ from the residual stream state *as it was after component i wrote to it*. If no other components wrote to the stream between $$i$$ and $$j$$, or if we are only interested in the direct path component $$i$$ adds to the stream that $$j$$ then reads, this is:

$$\mathbf{W}_{\text{virtual}}^{(i,j)} = \mathbf{W}_{\text{in}}^{(j)} \mathbf{W}_{\text{out}}^{(i)}$$

**Multi-Hop Virtual Weights:** However, the residual stream accumulates signals. For components separated by multiple layers, the input to component $$j$$ is influenced by component $$i$$ not just directly, but also through any intermediate layers $$k$$ (where $$i < k < j$$) that read from the stream (influenced by $$i$$) and then write back to it. The term $$\left( \mathbf{I} + \sum_{k=i+1}^{j-1} \mathbf{W}_{\text{out}}^{(k)} \mathbf{W}_{\text{in}}^{(k)} \right)$$ in the formula below accounts for these multi-hop paths. The identity matrix $$\mathbf{I}$$ represents the direct influence of component $$i$$'s output if it were to pass through to $$j$$ without intermediate modification, and the sum accumulates the effects of all intermediate components $$k$$ that read from the residual stream (which includes $$i$$'s contribution) and then write their output back to the stream before $$j$$ reads it. Thus, the total effective linear transformation from the output of component $$i$$ to the input of component $$j$$ is:

$$\mathbf{W}_{\text{virtual}}^{(i,j)} = \mathbf{W}_{\text{in}}^{(j)} \left( \mathbf{I} + \sum_{k=i+1}^{j-1} \mathbf{W}_{\text{out}}^{(k)} \mathbf{W}_{\text{in}}^{(k)} \right) \mathbf{W}_{\text{out}}^{(i)}$$

**Interpretation:** Virtual weights reveal which components can directly influence each other and quantify interaction strength. Large virtual weight magnitudes indicate strong potential for information transfer between components.

### Subspace Analysis

**Subspace Decomposition:** The residual stream can be decomposed into subspaces corresponding to different communication channels:

$$\mathbb{R}^{d_{\text{model}}} = \bigoplus_{i} \mathcal{S}_i$$

where each $$\mathcal{S}_i$$ represents a subspace used for specific types of information transfer.

**Singular Value Decomposition:** Virtual weights can be analyzed through SVD to identify primary directions of information transfer:

$$\mathbf{W}_{\text{virtual}}^{(i,j)} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$

where $$\mathbf{V}$$ describes the subspace read from component $$i$$ and $$\mathbf{U}$$ describes the subspace written to component $$j$$.

**Rank Analysis:** The rank of virtual weight matrices indicates effective dimensionality of communication between components, revealing potential bottlenecks or redundancies in information transfer.

## Attention Head Decomposition

Attention heads represent the core computational primitive of transformers, implementing context-dependent information routing through a sophisticated but analyzable mechanism.

### Attention as Information Movement

**Conceptual Framework:** Attention heads fundamentally perform information movement, reading information from the residual stream at source positions and writing it to destination positions. This perspective separates the "what" (information content) from the "where" (attention pattern) of computation.

**Mathematical Decomposition:** Standard attention computation can be decomposed into two independent operations:

$$\text{Attention Pattern:} \quad \mathbf{A} = \text{softmax}(\mathbf{x} \mathbf{W}_Q \mathbf{W}_K^T \mathbf{x}^T)$$

$$\text{Information Processing:} \quad \mathbf{h}(\mathbf{x}) = \mathbf{A} \mathbf{x} \mathbf{W}_V \mathbf{W}_O$$

**Tensor Product Formulation:** Using tensor notation, attention can be expressed as:

$$\mathbf{h}(\mathbf{x}) = (\mathbf{A} \otimes \mathbf{W}_{OV}) \cdot \mathbf{x}$$

where $$\mathbf{W}_{OV} = \mathbf{W}_O \mathbf{W}_V$$ and $$\otimes$$ denotes the tensor product.

### QK and OV Circuit Analysis

The decomposition of attention into query-key (QK) and output-value (OV) circuits provides a powerful framework for understanding attention head function.

**QK Circuit:** $$\mathbf{W}_{QK} = \mathbf{W}_Q^T \mathbf{W}_K$$ determines the attention pattern by computing similarity scores between query and key representations.

**Mathematical Properties:**
- $$\mathbf{W}_{QK}$$ is a $$d_{\text{model}} \times d_{\text{model}}$$ matrix operating on token representations
- The attention score between positions $$i$$ and $$j$$ is $$\mathbf{x}_i^T \mathbf{W}_{QK} \mathbf{x}_j$$
- $$\mathbf{W}_{QK}$$ can be analyzed through eigendecomposition to understand attention patterns

![Attention Head Decomposition](/assets/img/mech_interp_QK_OV.png)
*Decomposition of attention heads into QK and OV circuits, separating "where to attend" from "what information to move". Source: [Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html#splitting-attention-head-terms-into-circuits)*

**OV Circuit:** $$\mathbf{W}_{OV} = \mathbf{W}_O \mathbf{W}_V$$ determines what information is moved and how it is transformed.

**Mathematical Properties:**
- $$\mathbf{W}_{OV}$$ is a $$d_{\text{model}} \times d_{\text{model}}$$ matrix transforming information content
- The transformation applied to attended information is $$\mathbf{W}_{OV} \mathbf{x}_j$$
- $$\mathbf{W}_{OV}$$ can be analyzed through SVD to understand information processing

### Circuit Interpretation Techniques

**Eigenvalue Analysis:** The eigenvalues and eigenvectors of $$\mathbf{W}_{QK}$$ and $$\mathbf{W}_{OV}$$ provide insights into attention head function:
- Large positive eigenvalues in $$\mathbf{W}_{QK}$$ indicate strong attention to similar representations
- Large positive eigenvalues in $$\mathbf{W}_{OV}$$ suggest copying or amplification behavior
- Negative eigenvalues may indicate inhibition or deletion operations

**Copying Detection:** A "copying" attention head has $$\mathbf{W}_{OV} \approx \mathbf{I}$$, which can be detected through:

$$\text{Copying Score} = \frac{\text{tr}(\mathbf{W}_{OV})}{||\mathbf{W}_{OV}||_F}$$

**Attention Pattern Analysis:** The structure of $$\mathbf{W}_{QK}$$ determines characteristic attention patterns:
- Diagonal structure: attending to similar tokens
- Block structure: attending within semantic categories
- Low-rank structure: attending based on specific features

## Path Expansion and Circuit Composition

The linear structure of transformers enables exact decomposition of model computations into interpretable paths tracing information flow from inputs to outputs.

### Path Decomposition Framework

**Path Definition:** A path $$P$$ through a transformer is a sequence of components $$(c_1, c_2, \ldots, c_k)$$ that information can flow through from input to output.

**Path Contribution:** The contribution of path $$P$$ to final logits is:

$$\text{Logits}_P = \mathbf{U} \prod_{i=1}^{k} \mathbf{W}_{\text{out}}^{(c_i)} \mathbf{W}_{\text{in}}^{(c_i)} \mathbf{E}$$

**Complete Decomposition:** The total model output is the sum over all possible paths:

$$\text{Logits} = \sum_{P \in \mathcal{P}} \text{Logits}_P$$

where $$\mathcal{P}$$ is the set of all valid paths through the model.

![Path Expansion](/assets/img/mech_interp_logit_path.png)
*Path expansion through transformer layers: information flows through multiple possible routes from input to output. Source: [Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html#path-expansion-of-logits)*

### Attention-Only Path Analysis

For attention-only transformers, paths can be systematically enumerated and analyzed:

**Zero-Layer Paths:** Direct connection from embedding to unembedding:

$$\text{Logits}_{\text{direct}} = \mathbf{U} \mathbf{E}$$

This path implements bigram statistics, predicting the next token based solely on the current token.

**One-Layer Paths:** Paths through a single attention head:

$$\text{Logits}_{\text{1-layer}} = \mathbf{U} \mathbf{W}_{OV}^{(h)} \mathbf{A}^{(h)} \mathbf{E}$$

These paths implement skip-trigram statistics, enabling simple forms of in-context learning.

**Multi-Layer Paths:** Paths through multiple attention heads enable complex compositional behaviors:

$$\text{Logits}_{\text{multi}} = \mathbf{U} \prod_{i} \mathbf{W}_{OV}^{(h_i)} \mathbf{A}^{(h_i)} \mathbf{E}$$

### Virtual Attention Heads

**Definition:** Virtual attention heads represent the effective computation performed by compositions of multiple attention heads.

**Mathematical Formulation:** The composition of two attention heads creates a virtual attention head:

$$\mathbf{A}_{\text{virtual}} = \mathbf{A}^{(h_2)} \mathbf{A}^{(h_1)}$$

$$\mathbf{W}_{OV,\text{virtual}} = \mathbf{W}_{OV}^{(h_2)} \mathbf{W}_{OV}^{(h_1)}$$

**Interpretation:** Virtual attention heads reveal emergent behaviors arising from composition of simpler components, enabling analysis of complex multi-step algorithms.

## Composition Mechanisms in Transformers

Transformer expressivity emerges primarily through attention head composition across layers, which can occur through three distinct mechanisms operating on different aspects of attention computation.

### Types of Composition

**Q-Composition (Query Composition):** Earlier heads affect what later heads attend to by modifying query representations.

**Mathematical Description:** If head $$h_1$$ writes to the residual stream and head $$h_2$$ reads from it for query computation:

$$\mathbf{q}_i^{(h_2)} = (\mathbf{x}_i + \mathbf{A}^{(h_1)}_{i,:} \mathbf{x} \mathbf{W}_{OV}^{(h_1)}) \mathbf{W}_Q^{(h_2)}$$

**K-Composition (Key Composition):** Earlier heads affect what later heads attend from by modifying key representations.

**Mathematical Description:** Similarly for key computation:

$$\mathbf{k}_j^{(h_2)} = (\mathbf{x}_j + \mathbf{A}^{(h_1)}_{j,:} \mathbf{x} \mathbf{W}_{OV}^{(h_1)}) \mathbf{W}_K^{(h_2)}$$

**V-Composition (Value Composition):** Earlier heads affect what information is available to be moved by later heads.

**Mathematical Description:** For value computation:

$$\mathbf{v}_j^{(h_2)} = (\mathbf{x}_j + \mathbf{A}^{(h_1)}_{j,:} \mathbf{x} \mathbf{W}_{OV}^{(h_1)}) \mathbf{W}_V^{(h_2)}$$

### Composition Analysis Techniques

**Composition Strength Measurement:** The strength of composition between heads can be quantified through Frobenius norms of relevant virtual weight matrices:

$$\text{Q-Comp}(h_1, h_2) = ||\mathbf{W}_{OV}^{(h_1)} \mathbf{W}_Q^{(h_2)}||_F$$

$$\text{K-Comp}(h_1, h_2) = ||\mathbf{W}_{OV}^{(h_1)} \mathbf{W}_K^{(h_2)}||_F$$

$$\text{V-Comp}(h_1, h_2) = ||\mathbf{W}_{OV}^{(h_1)} \mathbf{W}_V^{(h_2)}||_F$$

**Composition Pattern Analysis:** Different composition patterns enable different types of algorithms:
- K-composition enables pattern matching and sequence detection
- V-composition enables information routing and transformation
- Q-composition enables conditional attention based on context

### Induction Heads: A Case Study

Induction heads represent a paradigmatic example of composition-enabled behavior demonstrating the power of the mathematical framework.

**Behavioral Definition:** Induction heads implement the pattern $$[A][B] \ldots [A] \rightarrow [B]$$, attending to tokens that follow previous instances of the current token.

**Mechanistic Implementation:** Induction heads require K-composition between two heads:
1. **Previous Token Head:** Attends to the previous token and copies its representation
2. **Induction Head:** Uses the copied representation to find tokens that previously followed the current token

**Mathematical Analysis:** The induction mechanism can be understood through the QK circuit of the induction head:

$$\mathbf{W}_{QK}^{\text{induction}} = \mathbf{W}_Q^T \mathbf{W}_K + \mathbf{W}_Q^T \mathbf{W}_{OV}^{\text{prev}} \mathbf{A}^{\text{prev}} \mathbf{W}_K$$

where the second term implements K-composition with the previous token head.

**Detection Criteria:** Induction heads can be identified through:
- Strong K-composition with previous token heads
- Positive eigenvalues in both QK and OV circuits (indicating copying behavior)
- Characteristic attention patterns on repeated sequences

## Zero and One-Layer Analysis

Analysis of simple transformer variants provides crucial insights into fundamental computational primitives and serves as a foundation for understanding more complex models.

### Zero-Layer Transformers

**Architecture:** A zero-layer transformer consists only of token embedding followed by unembedding:

$$\text{Logits} = \mathbf{U} \mathbf{E}$$

**Computational Interpretation:** This model can only implement bigram statistics, predicting the next token based solely on the current token. The optimal $$\mathbf{U} \mathbf{E}$$ approximates the bigram log-likelihood matrix.

**Implications for Larger Models:** The $$\mathbf{U} \mathbf{E}$$ term appears in every transformer as the "direct path" from embedding to unembedding, representing a baseline bigram component that other circuits build upon.

### One-Layer Attention-Only Transformers

**Path Decomposition:** One-layer models can be decomposed into interpretable paths:

$$\text{Logits} = \mathbf{U} \mathbf{E} + \sum_{h=1}^{H} \mathbf{U} \mathbf{W}_{OV}^{(h)} \mathbf{A}^{(h)} \mathbf{E}$$

**Skip-Trigram Implementation:** Each attention head implements skip-trigram statistics of the form $$[A] \ldots [B] \rightarrow [C]$$, where the head attends from position $$B$$ to position $$A$$ and predicts token $$C$$.

**Empirical Validation:** The skip-trigram interpretation can be validated by:
1. Extracting effective skip-trigram tables from model weights
2. Comparing these tables to empirical skip-trigram statistics from training data
3. Demonstrating that model predictions match extracted tables

**Limitations:** One-layer models are fundamentally limited to linear combinations of bigram and skip-trigram statistics, preventing more sophisticated forms of in-context learning.

## Two-Layer Models and Emergent Complexity

Two-layer attention-only transformers represent a critical transition point where composition enables qualitatively new computational capabilities.

### Compositional Algorithms

**Emergent Behaviors:** Two-layer models can implement algorithms impossible in one-layer models, primarily through attention head composition.

**Induction Head Formation:** The canonical example is induction head emergence, which requires K-composition between layers and enables sophisticated pattern matching.

**Algorithm Detection:** Compositional algorithms can be detected directly from weights through:
1. Computing composition strength matrices between all head pairs
2. Identifying heads with strong composition relationships
3. Analyzing QK and OV circuits of composed heads
4. Validating algorithmic hypotheses through behavioral testing

### Phase Transitions in Capability

**Qualitative Changes:** The transition from one to two layers represents a qualitative change in computational capability, not merely quantitative improvement.

**In-Context Learning:** Two-layer models exhibit genuine in-context learning through induction heads, while one-layer models only implement fixed statistical patterns.

**Algorithmic Sophistication:** Two-layer models can implement multi-step inference algorithms that adapt to context, representing a fundamental advance in computational sophistication.

## Experimental Validation Techniques

The mathematical framework must be validated through carefully designed experiments testing specific predictions about circuit behavior.

### Ablation Studies

**Component Ablation:** Individual attention heads or circuits can be ablated to test causal importance:

$$\text{Effect}(h) = \text{Performance}(\text{model}) - \text{Performance}(\text{model} \setminus h)$$

**Path Ablation:** Specific computational paths can be ablated by zeroing corresponding virtual weights:

$$\mathbf{W}_{\text{virtual}}^{(i,j)} \leftarrow \mathbf{0}$$

**Composition Ablation:** Composition relationships can be disrupted by modifying relevant weight matrices while preserving individual head function.

### Activation Patching

**Methodology:** Activation patching involves replacing activations from one input with activations from another input to test causal relationships.

**Circuit Validation:** Patching can validate circuit hypotheses by demonstrating that specific activations are necessary and sufficient for particular behaviors.

**Composition Testing:** Patching experiments can test composition hypotheses by selectively disrupting information flow between heads.

### Synthetic Data Experiments

**Controlled Environments:** Synthetic data enables precise control over statistical patterns that models must learn, facilitating clean tests of mechanistic hypotheses.

**Induction Head Testing:** Models can be trained on synthetic sequences with controlled repetition patterns to test induction head formation.

**Composition Analysis:** Synthetic tasks can be designed to require specific types of composition, enabling targeted analysis of compositional mechanisms.

## Scaling Considerations and Limitations

While the mathematical framework provides powerful tools for understanding transformer circuits, several challenges arise when scaling to larger, more realistic models.

### Computational Complexity

**Path Explosion:** The number of possible paths through a transformer grows exponentially with depth, making exhaustive analysis intractable for large models.

**Approximation Strategies:** Several approaches can make analysis tractable:
- Focus on the most important paths based on magnitude
- Use sampling techniques to estimate path contributions
- Develop hierarchical analysis methods grouping related paths

**Virtual Weight Computation:** Computing all pairwise virtual weights becomes prohibitive for large models, requiring selective analysis of the most promising component pairs.

### MLP Integration

**Current Limitations:** The framework developed here focuses primarily on attention-only models, with MLP layers representing a significant gap in current understanding.

**MLP Challenges:** MLP layers present several challenges for circuit analysis:
- Nonlinear activation functions complicate path analysis
- High dimensionality makes exhaustive analysis difficult
- Less clear computational primitives compared to attention heads

**Future Directions:** Extending the framework to include MLPs will require:
- Developing techniques for analyzing nonlinear transformations
- Identifying interpretable computational units within MLPs
- Understanding MLP-attention interactions

### Real-World Complexity

**Architectural Variations:** Real transformers include numerous architectural details (layer normalization, biases, different attention patterns) that complicate analysis.

**Training Dynamics:** The framework assumes converged models but doesn't address how circuits form during training.

**Scale-Dependent Phenomena:** Some circuit behaviors may only emerge at specific scales, requiring careful consideration of model size effects.

## Looking Forward

This comprehensive mathematical framework for transformer circuit analysis provides the tools necessary for systematic decomposition and understanding of transformer computations. The key insights include:

1. **Linear Structure Exploitation:** Transformers contain extensive linear structure enabling exact mathematical analysis of information flow and component interactions.

2. **Residual Stream Dynamics:** The residual stream serves as a linear communication channel that can be analyzed through virtual weights and subspace decomposition.

3. **Attention Decomposition:** Attention heads can be systematically analyzed through QK and OV circuit decomposition, revealing their computational functions.

4. **Path-Based Analysis:** Model computations can be decomposed into interpretable paths tracing information flow from inputs to outputs.

5. **Composition Mechanisms:** Three types of composition (Q, K, and V) enable complex behaviors through interaction of simpler components.

6. **Emergent Complexity:** Two-layer models exhibit qualitatively new capabilities through composition, representing a critical transition point in computational sophistication.

The mathematical framework developed here represents a foundational advance in our ability to understand and analyze transformer models. By providing precise tools for decomposing complex computations into interpretable components, this framework opens new possibilities for AI safety research, model debugging, and our fundamental understanding of how large language models process and generate text.

As we continue to scale these techniques to larger models and extend them to include MLP layers, we move closer to the ultimate goal of mechanistic interpretability: understanding neural networks well enough to predict their behavior, identify potential failure modes, and ensure their alignment with human values and intentions.

---

## References and Further Reading

This article is based on the mathematical framework developed in the Transformer Circuits Thread:

- **Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., ... & Olah, C.** (2021). [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/). *Transformer Circuits Thread*.
- **Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., ... & Olah, C.** (2022). [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/). *Transformer Circuits Thread*.
- **Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., ... & Olah, C.** (2022). [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/). *Transformer Circuits Thread*.
