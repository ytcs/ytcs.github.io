---
published: true
layout: post
title: "Mechanistic Interpretability: Part 3 - Mathematical Framework for Transformer Circuits"
categories: machine-learning
date: 2025-05-25
---

Following our exploration of the superposition hypothesis in [Part 2]({% post_url 2025-05-25-mechanistic-interpretability-part-2 %}), which addressed how features might be represented, we now turn to the specific architecture that dominates modern AI: the Transformer. To understand the learned algorithms within these powerful models, we need a robust **mathematical framework** that allows for systematic decomposition and analysis of their internal workings, particularly their attention mechanisms and information flow. This part lays out such a framework, building on the conceptual foundations from [Part 1]({% post_url 2025-05-25-mechanistic-interpretability-part-1 %}).

## Deconstructing the Transformer

Transformers, while complex, possess a highly structured architecture that lends itself to mathematical decomposition. Key components include token embeddings, positional encodings, multiple layers of attention and MLP (Multi-Layer Perceptron) blocks, and a final unembedding step. For the development of a clear mathematical framework, we often initially simplify by focusing on attention-only models or omitting biases and layer normalization, as these can be added back later without fundamentally altering the core computational pathways.

### The Residual Stream: A Central Communication Bus

The **residual stream** is arguably the most critical architectural feature for enabling mechanistic analysis. At each layer $$l$$, the output $$\mathbf{x}^{(l)}$$ is the sum of the input from the previous layer $$\mathbf{x}^{(l-1)}$$ and the computations performed by the current layer\'s components (e.g., attention head outputs, MLP outputs):

$$\mathbf{x}^{(l)} = \mathbf{x}^{(l-1)} + \text{ComponentOutput}^{(l)}(\mathbf{x}^{(l-1)})$$

This additive, linear structure means the residual stream acts like a shared communication bus or a "results highway." Any component can read from the accumulated state of the stream and write its own contribution back. This has profound implications:

1.  **Linearity for Analysis:** The primary information pathway is linear, allowing for techniques like path expansion and virtual weight computation.
2.  **Non-Privileged Basis:** The residual stream itself doesn't inherently have a privileged basis. Any global orthogonal transformation applied consistently to all interacting weight matrices would leave the model functionally unchanged. This reinforces the idea (from Part 2) that features are directions, not necessarily neuron alignments.
3.  **Superposition at Scale:** With many components writing to and reading from a residual stream of fixed dimensionality ($$d_{\text{model}}$$), it naturally becomes a place where multiple signals (feature activations) are superposed.

### Virtual Weights: Unveiling Effective Connectivity

Because of the residual stream's additive nature, a component in layer $$J$$ doesn't just see the output of layer $$J-1$$; it sees the sum of outputs from *all* preceding components $$I < J$$ that wrote to the stream. **Virtual weights** quantify the effective linear transformation from the output of an earlier component $$I$$ to the input of a later component $$J$$, considering all intermediate additions.

If component $$I$$ produces an output $$O_I$$ which is added to the stream, and component $$J$$ has an input weight matrix $$\mathbf{W}_{\text{in}}^{(J)}$$ that reads from the stream, the direct influence of $$O_I$$ on the input to $$J$$ (let's call this $$In_J$$) is part of $$In_J = \mathbf{W}_{\text{in}}^{(J)} (\dots + O_I + \dots)$$. If component $$I$$ has an output transformation $$\mathbf{W}_{\text{out}}^{(I)}$$ acting on its internal state $$h_I$$ to produce $$O_I = h_I \mathbf{W}_{\text{out}}^{(I)}$$, then the simplest virtual weight capturing the direct path is:

$$\mathbf{W}_{\text{virtual}}^{(I \rightarrow J)\text{, direct}} = \mathbf{W}_{\text{out}}^{(I)} \mathbf{W}_{\text{in}}^{(J)}$$
(Note: The order of multiplication depends on matrix dimension conventions; if $$h_I$$ is a row vector, this is correct. If column vectors, it would be $$\mathbf{W}_{\text{in}}^{(J)} \mathbf{W}_{\text{out}}^{(I)}$$. The key is the composition of the output transformation of $$I$$ and input transformation of $$J$$).

More generally, to account for the full residual stream accumulation between distant layers $$I$$ and $$J$$ (where layer $$I < \text{layer } k < \text{layer } J$$), the virtual weight from the output of component $$c_I$$ in layer $$L_I$$ to the input of component $$c_J$$ in layer $$L_J$$ can be thought of as the product of $$c_I$$'s output matrix, all intermediate layer transformations (identity if no component in a layer $$k$$ acts, or $$\mathbf{I} + \sum \mathbf{W}_{\text{out}}^{(k_m)} \mathbf{W}_{\text{in}}^{(k_m)}$$ for components $$k_m$$ in layer $$k$$), and $$c_J$$'s input matrix.

This concept is crucial for understanding how non-adjacent layers and components influence each other, forming long-range circuits.

## Decomposing the Attention Head

The attention mechanism is the heart of the Transformer. It dynamically routes information based on context. An attention head computes its output by attending to various positions in the input sequence and constructing a weighted sum of their value vectors.

Mathematically, for a single attention head, given input token representations $$\mathbf{x}_1, \dots, \mathbf{x}_N \in \mathbb{R}^{d_{\text{model}}}$$, the head first projects these into Query ($$\mathbf{q}_i$$), Key ($$\mathbf{k}_i$$), and Value ($$\mathbf{v}_i$$) vectors for each token $$i$$ using weight matrices $$\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{head}}}$$:

$$\mathbf{q}_i = \mathbf{x}_i \mathbf{W}_Q, \quad \mathbf{k}_j = \mathbf{x}_j \mathbf{W}_K, \quad \mathbf{v}_j = \mathbf{x}_j \mathbf{W}_V$$

Attention scores are computed as:
$$e_{ij} = \frac{\mathbf{q}_i \mathbf{k}_j^T}{\sqrt{d_{\text{head}}}}$$
These are normalized via Softmax to get attention weights $$\alpha_{ij} = \text{Softmax}_j(e_{ij})$$. The output for token $$i$$ from this head, before the final output projection $$\mathbf{W}_O \in \mathbb{R}^{d_{\text{head}} \times d_{\text{model}}}$$, is $$\mathbf{z}_i = \sum_j \alpha_{ij} \mathbf{v}_j$$. The head's final contribution to the residual stream for token $$i$$ is $$\mathbf{o}_i = \mathbf{z}_i \mathbf{W}_O$$.

This can be decomposed into two key conceptual circuits:

1.  **Query-Key (QK) Circuit:** Determines *where to attend*. The QK circuit computes the attention scores $$\alpha_{ij}$$. It is primarily governed by the effective QK matrix for comparing token $$i$$ with token $$j$$:
    $$\text{Score}_{ij} \propto (\mathbf{x}_i \mathbf{W}_Q) (\mathbf{x}_j \mathbf{W}_K)^T = \mathbf{x}_i (\mathbf{W}_Q \mathbf{W}_K^T) \mathbf{x}_j^T$$
    So, the matrix $$\mathbf{W}_{\text{eff-QK}} = \mathbf{W}_Q \mathbf{W}_K^T \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$ (acting on full token representations) defines how pairs of token representations in the residual stream are compared to produce attention scores. Its rank is at most $$d_{\text{head}}$$.

2.  **Output-Value (OV) Circuit:** Determines *what information to move* from the attended positions. Once attention weights $$\alpha_{ij}$$ are computed, the OV circuit processes the value vectors. The transformation applied to a value vector $$\mathbf{x}_j \mathbf{W}_V$$ and then projected out is described by:
    $$\mathbf{W}_{\text{OV}} = \mathbf{W}_V \mathbf{W}_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$\
    This matrix (also rank at most $$d_{\text{head}}$$) describes the transformation applied to a value vector (derived from a token representation in the residual stream) before it's written back to the residual stream. For example, if $$\mathbf{W}_{\text{OV}} \approx \mathbf{I}$$ (identity), the head primarily copies information.

Analyzing the properties (e.g., SVD, eigenvalues) of $$\mathbf{W}_Q \mathbf{W}_K^T$$ and $$\mathbf{W}_V \mathbf{W}_O$$ reveals the specific attention patterns and information processing strategies of individual heads.

## Path Expansion and Compositional Power

The overall computation of a Transformer can be viewed as a sum over all possible **paths** that information can take from the input embedding to the final output logits. Each path involves a sequence of components (attention heads, MLP layers) and their respective weight matrices.

For an attention-only transformer, the output logit for a token $$c$$ given a previous token (position $$pos$$) can be written as:

$$\text{Logits}(c | pos) = \mathbf{U}[c,:] \left( \mathbf{E}[pos,:] + \sum_{l,h} \text{Head}_{l,h}(\text{ResidualStreamInput}_{l,h}) \right)$$
where $$\mathbf{E}$$ is the token embedding matrix and $$\mathbf{U}$$ is the unembedding matrix. This can be expanded:

$$\text{Logits}(c | pos) = \underbrace{\mathbf{U}[c,:] \mathbf{E}[pos,:]}_{\text{Direct Path (0-layer)}} + \sum_{l,h} \underbrace{\mathbf{U}[c,:] \text{Head}_{l,h}(\mathbf{E}[pos,:])}_{\text{1-layer paths}} + \dots$$

-   **Zero-Layer Path:** The direct connection $$\mathbf{U} \mathbf{E}$$ effectively captures bigram statistics (predicting the next token based only on the current token).
-   **One-Layer Paths:** Paths passing through a single attention head can implement skip-trigram statistics (e.g., if $$A$$ attends to $$B$$, and $$B$$ was produced by $$C$$, this forms an $$A \dots C \rightarrow B$$ type pattern).

### Composition Mechanisms: The Source of Complex Algorithms

The true power of multi-layer Transformers comes from **composition**, where the output of earlier components influences the computation of later components. For attention heads, this occurs in three main ways:

1.  **Q-Composition (Query Composition):** The output of head $$h_1$$ (in layer $$L_1$$) modifies the residual stream. When head $$h_2$$ (in layer $$L_2 > L_1$$) computes its Query vector, it reads from this modified stream. Thus, $$h_1$$ influences what $$h_2$$ attends to.
    $$\mathbf{q}_i^{(h_2)} = (\mathbf{x}_i^{(L_2-1)}) \mathbf{W}_Q^{(h_2)} = (\dots + \text{Output}(h_1) + \dots) \mathbf{W}_Q^{(h_2)}$$

2.  **K-Composition (Key Composition):** Similarly, $$h_1$$ can influence the Key vectors that $$h_2$$ uses for comparison.
    $$\mathbf{k}_j^{(h_2)} = (\mathbf{x}_j^{(L_2-1)}) \mathbf{W}_K^{(h_2)} = (\dots + \text{Output}(h_1) + \dots) \mathbf{W}_K^{(h_2)}$$

3.  **V-Composition (Value Composition):** And $$h_1$$ can influence the Value vectors that $$h_2$$ aggregates and outputs.
    $$\mathbf{v}_j^{(h_2)} = (\mathbf{x}_j^{(L_2-1)}) \mathbf{W}_V^{(h_2)} = (\dots + \text{Output}(h_1) + \dots) \mathbf{W}_V^{(h_2)}$$

These composition mechanisms allow for the construction of **virtual attention heads**, where the combined effect of multiple heads implements a more complex attention pattern or information transformation than any single head could. For instance, K-composition is fundamental to **induction heads** (explored in Part 9), which enable transformers to perform in-context learning by recognizing repeated sequences.

### Emergent Complexity in Two-Layer Models

While a zero-layer Transformer is limited to bigrams and a one-layer attention-only Transformer to skip-trigrams, a **two-layer Transformer** can already exhibit qualitatively new capabilities due to composition. For example, an induction head typically requires at least two heads working in sequence: 
-   A "previous token" head (Head 1) copies the $$N-1$$-th token's representation.
-   An "induction" head (Head 2) uses this copied representation (via K-composition) to search for previous occurrences of token $$N-1$$ and attend to the token that followed it.
This simple two-head circuit allows the model to complete patterns like "A B ... A __" by predicting "B". This is a form of in-context learning that is impossible with single heads in isolation.

## Conclusion

This mathematical framework provides the tools to dissect Transformers into their constituent computational parts: the residual stream as a communication bus, attention heads decomposed into QK and OV circuits, and the powerful concept of composition that allows simple components to build complex algorithms. By analyzing virtual weights, path expansions, and composition strengths, we can start to reverse-engineer the specific computations learned by these models.

This foundation is crucial for understanding phenomena like superposition within these architectures and for developing techniques to extract and validate the features and circuits that implement their remarkable capabilities. It sets the stage for [Part 4]({% post_url 2025-05-25-mechanistic-interpretability-part-4 %}), where we will explore dictionary learning as a method to deal with superposition, and for later parts ([Parts 7-9]({% post_url 2025-05-25-mechanistic-interpretability-part-7 %})) where we will apply this framework to analyze specific circuits like induction heads in detail.

---

## References and Further Reading

This framework is primarily based on the work by Elhage et al. and Olsson et al. in the Transformer Circuits Thread:

-   **Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., ... & Olah, C.** (2021). [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/). *Transformer Circuits Thread*
-   **Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., ... & Olah, C.** (2022). [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/). *Transformer Circuits Thread*
-   Insights on the residual stream and attention also draw from the original Transformer paper: Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *NeurIPS*.
