---
published: true
layout: post
title: "Mechanistic Interpretability: Part 4 - Mathematical Framework for Transformer Circuits"
categories: machine-learning
date: 2025-05-25
---

Following our exploration of polysemanticity and monosemanticity in [Part 3]({% post_url 2025-05-25-mechanistic-interpretability-part-3 %}), we now turn to the specific architecture that dominates modern AI: the Transformer. To understand the learned algorithms within these powerful models, we need a robust **mathematical framework** that allows for systematic decomposition and analysis of their internal workings, particularly their attention mechanisms and information flow. This part lays out such a framework, building on the conceptual foundations from [Part 1]({% post_url 2025-05-25-mechanistic-interpretability-part-1 %}) and [Part 2]({% post_url 2025-05-25-mechanistic-interpretability-part-2 %}).

## Deconstructing the Transformer

Transformers, while complex, possess a highly structured architecture that lends itself to mathematical decomposition. Key components include token embeddings, positional encodings, multiple layers of attention and MLP (Multi-Layer Perceptron) blocks, and a final unembedding step. For the development of a clear mathematical framework, we often initially simplify by focusing on attention-only models or omitting biases and layer normalization, as these can be added back later without fundamentally altering the core computational pathways.

### The Residual Stream: A Central Communication Bus

The **residual stream** is arguably the most critical architectural feature for enabling mechanistic analysis. At each layer $$l$$, the output $$\mathbf{x}^{(l)}$$ is the sum of the input from the previous layer $$\mathbf{x}^{(l-1)}$$ and the computations performed by the current layer's components (e.g., attention head outputs, MLP outputs):

$$\mathbf{x}^{(l)} = \mathbf{x}^{(l-1)} + \sum_k \text{ComponentOutput}_k^{(l)}(\mathbf{x}^{(l-1)})$$

where $$\text{ComponentOutput}_k^{(l)}(\mathbf{x}^{(l-1)})$$ is the output of the $$k$$-th component in layer $$l$$ (e.g., an attention head or an MLP block), which itself is a function of the input to that layer, $$\mathbf{x}^{(l-1)}$$. This additive, linear structure means the residual stream acts like a shared communication bus or a "results highway." Any component can read from the accumulated state of the stream and write its own contribution back. This has profound implications:

1.  **Linearity for Analysis:** The primary information pathway is linear, allowing for techniques like path expansion and virtual weight computation.
2.  **Non-Privileged Basis:** The residual stream itself doesn't inherently have a privileged basis. Any global orthogonal transformation applied consistently to all interacting weight matrices would leave the model functionally unchanged. This reinforces the idea (from Part 2) that features are directions, not necessarily neuron alignments.
3.  **Superposition at Scale:** With many components writing to and reading from a residual stream of fixed dimensionality ($$d_{\text{model}}$$), it naturally becomes a place where multiple signals (feature activations) are superposed.

### Virtual Weights: Unveiling Effective Connectivity

Because of the residual stream's additive nature, a component in a later layer $$L_J$$ doesn't just see the direct output of layer $$L_J-1$$; it effectively sees the sum of outputs from *all* preceding components in layers $$L_I < L_J$$ that wrote to the stream. **Virtual weights** quantify the effective linear transformation from the input of an earlier component (or its output contribution) to an input processing stage of a later component, considering all intermediate additions and transformations in the residual stream.

Let's define some terms:
-   Let $$\mathbf{M}_{out}^{(C_I)}$$ be the effective output matrix of a component $$C_I$$ (e.g., an attention head $$H_I$$ or an MLP block) in layer $$L_I$$. If $$X$$ is the input to component $$C_I$$ from the residual stream, its output contribution to the stream is $$O_{C_I} = X \mathbf{M}_{out}^{(C_I)}$$. 
    For an attention head $$H_I$$, $$\mathbf{M}_{out}^{(H_I)}$$ would be its value-output transformation $$\mathbf{W}_V^{(H_I)}\mathbf{W}_O^{(H_I)}$$ (a $$d_{\text{model}} \times d_{\text{model}}$$ matrix), assuming the attention pattern itself is fixed or we are analyzing a specific path of information flow through a value vector. 
    For an MLP layer, if it's a simple linear transformation, $$\mathbf{M}_{out}^{(MLP)}$$ would be its weight matrix. If it's a non-linear MLP (e.g., with ReLU), $$\mathbf{M}_{out}^{(MLP)}$$ represents an effective linear matrix for a specific input or an average sense, or considers the path through specific active neurons. For the purpose of linear path analysis, we often approximate non-linear components by their local linear behavior or focus on paths where non-linearities are fixed (e.g. a specific ReLU activation pattern). A common formulation for a two-layer MLP is $$\mathbf{W}_{in}^{(MLP)}\mathbf{W}_{out}^{(MLP)}$$ (again, $$d_{\text{model}} \times d_{\text{model}}$$).
-   Let $$\mathbf{W}_{in-proj}^{(C_J)}$$ be an input projection matrix of a component $$C_J$$ in layer $$L_J$$. 
    For an attention head $$H_J$$, this could be its query matrix $$\mathbf{W}_Q^{(H_J)}$$ ($$d_{\text{model}} \times d_{\text{head}}$$), key matrix $$\mathbf{W}_K^{(H_J)}$$ ($$d_{\text{model}} \times d_{\text{head}}$$), or value matrix $$\mathbf{W}_V^{(H_J)}$$ ($$d_{\text{model}} \times d_{\text{head}}$$). 

**1. Direct Virtual Weight (No Intermediate Layers, i.e., $$L_J = L_I + 1$$ or within the same layer if analyzing parallel components):**

If component $$C_I$$ outputs $$O_{C_I} = X \mathbf{M}_{out}^{(C_I)}$$ to the stream, and component $$C_J$$ (in the next layer or a later component in the same layer reading from the updated stream) uses an input projection $$\mathbf{W}_{in-proj}^{(C_J)}$$, the part of $$C_J$$'s projected input that comes from $$X$$ via $$C_I$$ is $$ (X \mathbf{M}_{out}^{(C_I)}) \mathbf{W}_{in-proj}^{(C_J)} $$.
The **direct virtual weight matrix** $$\mathbf{W}_{\text{virtual}}^{(C_I \rightarrow C_J)}$$ mapping the input $$X$$ (that fed into $$C_I$$) to this specific contribution at $$C_J$$'s input projection is:

$$\mathbf{W}_{\text{virtual, direct}}^{(C_I \rightarrow C_J)} = \mathbf{M}_{out}^{(C_I)} \mathbf{W}_{in-proj}^{(C_J)}$$

For example, the virtual weight from the input of Head $$H_a$$'s OV circuit (matrix $$\mathbf{W}_V^{(Ha)}\mathbf{W}_O^{(Ha)}$$) to the Query input projection of Head $$H_b$$ (matrix $$\mathbf{W}_Q^{(Hb)}$$) in an immediately subsequent processing step is $$(\mathbf{W}_V^{(Ha)}\mathbf{W}_O^{(Ha)}) \mathbf{W}_Q^{(Hb)}$$. This resulting matrix has dimensions $$d_{\text{model}} \times d_{\text{head}}^{(Hb)}$$.

**2. Virtual Weight Across Intermediate Layers:**

Now, consider components $$C_I$$ in layer $$L_I$$ and $$C_J$$ in a later layer $$L_J$$ ($$L_J > L_I$$). The signal $$O_{C_I}$$ from $$C_I$$ passes through intermediate layers $$L_k$$ (for $$L_I < L_k < L_J$$).

Each intermediate layer $$L_k$$ applies a linear transformation to the signal passing through its residual stream. If layer $$L_k$$ contains components $$C_{k,m}$$ (heads or MLPs) with effective output matrices $$\mathbf{M}_{out}^{(k,m)}$$ (as defined above, noting the linear approximation for MLPs if non-linear), then a signal $$S$$ entering layer $$L_k$$ from the previous layer's residual stream is transformed to $$S + \sum_m S \mathbf{M}_{out}^{(k,m)} = S (\mathbf{I} + \sum_m \mathbf{M}_{out}^{(k,m)})$$ upon exiting layer $$L_k$$. 
Let $$T_k = (\mathbf{I} + \sum_m \mathbf{M}_{out}^{(k,m)})$$ be this full linear transformation for layer $$L_k$$, representing the cumulative effect of all parallel components in that layer on a signal passing through the residual stream.

The output contribution $$O_{C_I} = X \mathbf{M}_{out}^{(C_I)}$$ from component $$C_I$$ (where $$X$$ was its input from the stream) becomes $$ (X \mathbf{M}_{out}^{(C_I)}) \cdot T_{L_I+1} \cdot T_{L_I+2} \cdot \ldots \cdot T_{L_J-1} $$ by the time it reaches the input of layer $$L_J$$.
This transformed signal is then processed by $$C_J$$'s input projection $$\mathbf{W}_{in-proj}^{(C_J)}$$.
Thus, the **full virtual weight matrix** from the input $$X$$ of component $$C_I$$ to the specific projected input of component $$C_J$$ is:

$$\mathbf{W}_{\text{virtual}}^{(C_I \rightarrow C_J)} = \mathbf{M}_{out}^{(C_I)} \left( \prod_{k=L_I+1}^{L_J-1} T_k \right) \mathbf{W}_{in-proj}^{(C_J)}$$

If there are no intermediate layers ($$L_J = L_I+1$$), the product term is empty (or an identity matrix), reducing to the direct case.
This concept is crucial for understanding how non-adjacent layers and components influence each other, effectively forming long-range circuits by composing these linear transformations.

## Decomposing the Attention Head

The attention mechanism is the heart of the Transformer. It dynamically routes information based on context. An attention head computes its output by attending to various positions in the input sequence and constructing a weighted sum of their value vectors.

Mathematically, for a single attention head, given input token representations $$\mathbf{x}_1, \dots, \mathbf{x}_N \in \mathbb{R}^{d_{\text{model}}}$$, the head first projects these into Query ($$\mathbf{q}_i$$), Key ($$\mathbf{k}_j$$), and Value ($$\mathbf{v}_j$$) vectors for each token $$i$$ (query) and $$j$$ (key/value source) using weight matrices $$\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{head}}}$$:

$$\mathbf{q}_i = \mathbf{x}_i \mathbf{W}_Q, \quad \mathbf{k}_j = \mathbf{x}_j \mathbf{W}_K, \quad \mathbf{v}_j = \mathbf{x}_j \mathbf{W}_V$$

(Note: $$\mathbf{q}_i, \mathbf{k}_j, \mathbf{v}_j$$ are row vectors of dimension $$d_{\text{head}}$$.)

Attention scores are computed as the dot product of a query vector with a key vector, scaled by $$\sqrt{d_{\text{head}}}$$:

$$e_{ij} = \frac{\mathbf{q}_i \mathbf{k}_j^T}{\sqrt{d_{\text{head}}}}$$

These scores are then normalized via Softmax across all source positions $$j$$ to get attention weights $$\alpha_{ij} = \text{Softmax}_j(e_{ij})$$. The output for query token $$i$$ from this head, before the final output projection, is a weighted sum of value vectors: $$\mathbf{z}_i = \sum_j \alpha_{ij} \mathbf{v}_j$$. 

This output $$\mathbf{z}_i$$ (a $$d_{\text{head}}$$ dimensional row vector) is then projected back to the model dimension using the output weight matrix $$\mathbf{W}_O \in \mathbb{R}^{d_{\text{head}} \times d_{\text{model}}}$$. The head's final contribution to the residual stream for token $$i$$ is 

$$\mathbf{o}_i = \mathbf{z}_i \mathbf{W}_O$$

This mechanism can be decomposed into two key conceptual circuits:

1.  **Query-Key (QK) Circuit:** Determines *where to attend*. The QK circuit computes the attention scores $$e_{ij}$$ (before softmax). The core of this computation is the term $$\mathbf{q}_i \mathbf{k}_j^T$$. Let's derive its form in terms of the original residual stream vectors $$\mathbf{x}_i$$ and $$\mathbf{x}_j$$:

    $$\mathbf{q}_i \mathbf{k}_j^T = (\mathbf{x}_i \mathbf{W}_Q) (\mathbf{x}_j \mathbf{W}_K)^T$$

    Using the matrix transpose property $$(AB)^T = B^T A^T$$, we have $$( \mathbf{x}_j \mathbf{W}_K )^T = \mathbf{W}_K^T \mathbf{x}_j^T$$.
    Substituting this back, we get:

    $$\mathbf{q}_i \mathbf{k}_j^T = \mathbf{x}_i \mathbf{W}_Q \mathbf{W}_K^T \mathbf{x}_j^T$$

    This expression shows that the unnormalized attention score between token $$i$$ and token $$j$$ is a bilinear form $$\mathbf{x}_i (\mathbf{W}_Q \mathbf{W}_K^T) \mathbf{x}_j^T$$.

    The matrix $$\mathbf{W}_{\text{eff-QK}} = \mathbf{W}_Q \mathbf{W}_K^T$$ is an effective $$d_{\text{model}} \times d_{\text{model}}$$ matrix that defines how pairs of token representations in the residual stream are compared to produce attention scores. Since $$\mathbf{W}_Q$$ is $$d_{\text{model}} \times d_{\text{head}}$$ and $$\mathbf{W}_K^T$$ is $$d_{\text{head}} \times d_{\text{model}}$$, the rank of $$\mathbf{W}_{\text{eff-QK}}$$ is at most $$d_{\text{head}}$$, which is typically much smaller than $$d_{\text{model}}$$. This low-rank structure implies that the QK circuit is specialized in comparing specific types of information, effectively projecting the $$d_{\text{model}}$$-dimensional token representations into a shared $$d_{\text{head}}$$-dimensional space for comparison.

2.  **Output-Value (OV) Circuit:** Determines *what information to move* from the attended positions and how it's transformed. Once attention weights $$\alpha_{ij}$$ are computed, the OV circuit processes the value vectors. The full transformation from an original token representation $$\mathbf{x}_j$$ (at a source position $$j$$) to its potential contribution to the output (if fully attended, i.e., $$\alpha_{ij}=1$$) is $$\mathbf{x}_j \mathbf{W}_V \mathbf{W}_O$$.

    The matrix $$\mathbf{W}_{\text{OV}} = \mathbf{W}_V \mathbf{W}_O$$ is an effective $$d_{\text{model}} \times d_{\text{model}}$$ matrix. (Since $$\mathbf{W}_V$$ is $$d_{\text{model}} \times d_{\text{head}}$$ and $$\mathbf{W}_O$$ is $$d_{\text{head}} \times d_{\text{model}}$$, their product is $$d_{\text{model}} \times d_{\text{model}}$$).

    This matrix describes the transformation applied to a value vector (derived from a token representation $$\mathbf{x}_j$$ in the residual stream) before it's written back to the residual stream at position $$i$$. Its rank is also at most $$d_{\text{head}}$$. For example, if $$\mathbf{W}_{\text{OV}} \approx \mathbf{I}$$ (identity matrix), the head primarily copies information from attended positions. If it's different, it transforms the information.

Analyzing the properties (e.g., SVD, eigenvalues) of $$\mathbf{W}_Q \mathbf{W}_K^T$$ and $$\mathbf{W}_V \mathbf{W}_O$$ reveals the specific attention patterns and information processing strategies of individual heads.

## Path Expansion and Compositional Power

The overall computation of a Transformer can be viewed as a sum over all possible **paths** that information can take from the input embedding to the final output logits. Each path involves a sequence of components (attention heads, MLP layers) and their respective weight matrices. While attention introduces non-linearity via softmax, analyzing specific paths (e.g., by fixing attention patterns or looking at linear segments of the computation) is a key strategy.

For an attention-only transformer, the output logit for a token $$c$$ given a previous token (position $$pos$$) can be written as:

$$\text{Logits}(c | pos) = \mathbf{U}[c,:] \left( \mathbf{E}[pos,:] + \sum_{l,h} \text{OutputContribution}_{l,h} \right)$$

where $$\mathbf{E}$$ is the token embedding matrix (row per token, $$d_{vocab} \times d_{model}$$), $$\mathbf{U}$$ is the unembedding matrix (often $$\mathbf{E}^T$$, so $$d_{model} \times d_{vocab}$$), and $$\text{OutputContribution}_{l,h}$$ is the output vector added to the residual stream by head $$h$$ in layer $$l$$.

This can be expanded. For instance, the output contribution of head $$(l,h)$$ acting on the stream input $$\mathbf{S}^{(l-1)}$$ (output of layer $$l-1$$) is $$ ( \sum_j \alpha_{pos,j}^{(l,h)} ( \mathbf{S}_j^{(l-1)}\mathbf{W}_V^{(l,h)} ) ) \mathbf{W}_O^{(l,h)} $$.

The simplest paths are:
-   **Zero-Layer Path:** The direct connection from embedding to unembedding. If token at position $$pos$$ has embedding vector $$\mathbf{E}[pos,:]$$, then the direct contribution to logits is $$\mathbf{U}[c,:] \mathbf{E}[pos,:]$$. This path effectively captures token co-occurrence statistics similar to bigrams if $$\mathbf{U} \approx \mathbf{E}^T$$.

-   **One-Layer Paths:** Paths passing through a single attention head. The term $$\mathbf{U}[c,:] \text{Head}_{l,h}(\mathbf{E}[pos,:]) $$ describes the influence of head $$(l,h)$$ acting on the initial embedding $$\mathbf{E}[pos,:]$$ (if it's in the first layer) on the logit for token $$c$$. This can implement more complex statistics like skip-trigrams.

### Composition Mechanisms: The Source of Complex Algorithms

The true power of multi-layer Transformers comes from **composition**, where the output of earlier components influences the computation of later components. This is where virtual weights become essential for analysis. For attention heads, this occurs in three main ways:

1.  **Q-Composition (Query Composition):** The output of head $$H_1$$ (in layer $$L_1$$) modifies the residual stream. When head $$H_2$$ (in layer $$L_2 > L_1$$) computes its Query vector, it reads from this modified stream. Thus, $$H_1$$ influences what $$H_2$$ attends to.
    Let $$X$$ be the input to $$H_1$$'s OV circuit (matrix $$\mathbf{M}_{out}^{(H_1)} = \mathbf{W}_V^{(H_1)}\mathbf{W}_O^{(H_1)}$$). Its output contribution is $$X \mathbf{M}_{out}^{(H_1)}$$. If there are intermediate layers transforming this by a product of matrices $$T_{inter} = \prod_{k=L_1+1}^{L_2-1} ( \mathbf{I} + \sum_m \mathbf{M}_{out}^{(k,m)} )$$, this signal becomes $$X \mathbf{M}_{out}^{(H_1)} T_{inter}$$ as it enters the layer of $$H_2$$.

    The query vector for $$H_2$$ is formed from the stream $$S_{L_2-1}$$ as $$S_{L_2-1} \mathbf{W}_Q^{(H_2)}$$. The part of this query that comes from $$X$$ via $$H_1$$ is $$(X \mathbf{M}_{out}^{(H_1)} T_{inter}) \mathbf{W}_Q^{(H_2)}$$.
    
    The virtual weight matrix for this Q-composition path is $$\mathbf{W}_{\text{Q-comp}} = \mathbf{M}_{out}^{(H_1)} T_{inter} \mathbf{W}_Q^{(H_2)}$$.

2.  **K-Composition (Key Composition):** Similarly, $$H_1$$ can influence the Key vectors that $$H_2$$ uses for comparison. The output from $$H_1$$ ($$X \mathbf{M}_{out}^{(H_1)} T_{inter}$$) influences the stream from which $$H_2$$ forms its key vectors $$S_{L_2-1} \mathbf{W}_K^{(H_2)}$$.
    The virtual weight matrix for this K-composition path (from input of $$H_1$$'s OV to $$H_2$$'s K-projection) is $$\mathbf{W}_{\text{K-comp}} = \mathbf{M}_{out}^{(H_1)} T_{inter} \mathbf{W}_K^{(H_2)}$$.

3.  **V-Composition (Value Composition):** And $$H_1$$ can influence the Value vectors that $$H_2$$ aggregates. The output from $$H_1$$ ($$X \mathbf{M}_{out}^{(H_1)} T_{inter}$$) influences the stream from which $$H_2$$ forms its value vectors $$S_{L_2-1} \mathbf{W}_V^{(H_2)}$$. This then passes through $$H_2$$'s output projection $$\mathbf{W}_O^{(H_2)}$$.
    The virtual weight matrix for this V-composition path (from input of $$H_1$$'s OV to $$H_2$$'s OV output) is $$\mathbf{W}_{\text{V-comp}} = \mathbf{M}_{out}^{(H_1)} T_{inter} \mathbf{W}_V^{(H_2)} \mathbf{W}_O^{(H_2)}$$.

These composition mechanisms, understood via virtual weights, allow for the construction of **virtual attention heads**, where the combined effect of multiple heads implements a more complex attention pattern or information transformation than any single head could. For instance, K-composition is fundamental to **induction heads** (explored in Part 9).

### Emergent Complexity in Two-Layer Models

While a zero-layer Transformer is limited to bigrams and a one-layer attention-only Transformer to skip-trigrams, a **two-layer Transformer** can already exhibit qualitatively new capabilities due to composition. For example, an induction head typically requires at least two heads working in sequence:
-   A "previous token" head (Head 1) in an earlier layer $$L_1$$ copies (parts of) the $$N-1$$-th token's representation into the residual stream.
-   An "induction" head (Head 2) in a later layer $$L_2$$ uses this copied representation. Specifically, via K-composition, the Key vectors generated by $$H_2$$ for previous tokens in the sequence are modulated by the output of $$H_1$$. If $$H_2$$ is looking for the token that followed previous instances of token $$N-1$$, its Query vector (also potentially influenced by $$H_1$$'s output via Q-composition) will match strongly with Key vectors of tokens that *are* $$N-1$$, and the overall QK circuit of $$H_2$$ is further specialized to shift attention to the token *after* these matched $$N-1$$ tokens. The OV circuit of $$H_2$$ then copies this successfully identified token. This is a form of in-context learning that is impossible with single heads in isolation.

Let's derive the explicit mathematical formulations for zero-layer, one-layer, and two-layer transformers to better understand this emergent complexity:

#### Zero-Layer Transformer: Direct Token Mapping

In a zero-layer transformer, we have direct connections from token embeddings to output logits without any intermediate attention or MLP layers. The mathematical formulation for predicting the next token is simply:

$$\text{Logits}(c | \text{pos}) = \mathbf{U}[c,:] \cdot \mathbf{E}[\text{pos},:]$$

Where:
- $$\mathbf{E} \in \mathbb{R}^{|V| \times d_{\text{model}}}$$ is the embedding matrix
- $$\mathbf{U} \in \mathbb{R}^{d_{\text{model}} \times |V|}$$ is the unembedding matrix
- $$\text{pos}$$ is the position of the input token
- $$c$$ is the candidate output token (in the vocabulary)

When the unembedding matrix is (approximately) the transpose of the embedding matrix ($$\mathbf{U} \approx \mathbf{E}^T$$), this computation reduces to measuring token similarity:

$$\text{Logits}(c | \text{pos}) \approx \mathbf{E}[c,:] \cdot \mathbf{E}[\text{pos},:]^T = \text{similarity}(c, \text{pos})$$

This formulation can only capture simple bigram statistics based on embedding similarity. The zero-layer transformer effectively learns which tokens tend to follow other tokens directly, without any contextual understanding.

#### One-Layer Transformer: Attention-Based Contextual Processing

A one-layer transformer introduces attention mechanisms between the embedding and unembedding steps. For a model with $$H$$ attention heads, the logits are computed as:

$$\text{Logits}(c | \text{pos}) = \mathbf{U}[c,:] \left( \mathbf{E}[\text{pos},:] + \sum_{h=1}^{H} \text{Head}_h(\mathbf{E})[\text{pos},:] \right)$$

For each attention head $$h$$ processing position $$\text{pos}$$, the output contribution is:

$$\text{Head}_h(\mathbf{E})[\text{pos},:] = \sum_{j=1}^{\text{pos}} \alpha_{\text{pos},j}^{(h)} \cdot (\mathbf{E}[j,:] \cdot \mathbf{W}_V^{(h)}) \cdot \mathbf{W}_O^{(h)}$$

Where the attention weights $$\alpha_{\text{pos},j}^{(h)}$$ are calculated using softmax over attention scores:

$$\alpha_{\text{pos},j}^{(h)} = \frac{\exp(e_{\text{pos},j}^{(h)})}{\sum_{k=1}^{\text{pos}} \exp(e_{\text{pos},k}^{(h)})}$$

And the attention scores $$e_{\text{pos},j}^{(h)}$$ are:

$$e_{\text{pos},j}^{(h)} = \frac{(\mathbf{E}[\text{pos},:] \cdot \mathbf{W}_Q^{(h)}) \cdot (\mathbf{E}[j,:] \cdot \mathbf{W}_K^{(h)})^T}{\sqrt{d_{\text{head}}}}$$

This can be rewritten using the effective QK matrix as described earlier:

$$e_{\text{pos},j}^{(h)} = \frac{\mathbf{E}[\text{pos},:] \cdot \mathbf{W}_{\text{eff-QK}}^{(h)} \cdot \mathbf{E}[j,:]^T}{\sqrt{d_{\text{head}}}}$$

Where $$\mathbf{W}_{\text{eff-QK}}^{(h)} = \mathbf{W}_Q^{(h)} \cdot \mathbf{W}_K^{(h)T}$$.

The one-layer transformer can learn to selectively attend to previous tokens based on their relevance to the current position. This allows it to implement skip-trigram patterns by, for example, having position $$\text{pos}$$ attend strongly to positions $$\text{pos}-2$$ and $$\text{pos}-1$$ to predict the next token.

However, a one-layer transformer cannot implement the induction pattern (copying a token that followed a similar context elsewhere in the sequence) because each head operates independently on the original token embeddings.

#### Two-Layer Transformer: Composition and Emergent Capabilities

In a two-layer transformer, the output of the first layer's attention heads becomes the input for the second layer's heads, enabling composition. For a model with $$H_1$$ heads in layer 1 and $$H_2$$ heads in layer 2, the logits are:

$$\text{Logits}(c | \text{pos}) = \mathbf{U}[c,:] \cdot \mathbf{S}^{(2)}[\text{pos},:]$$

Where $$\mathbf{S}^{(2)}$$ is the residual stream after layer 2:

$$\mathbf{S}^{(2)}[\text{pos},:] = \mathbf{S}^{(1)}[\text{pos},:] + \sum_{h=1}^{H_2} \text{Head}_{2,h}(\mathbf{S}^{(1)})[\text{pos},:]$$

And $$\mathbf{S}^{(1)}$$ is the residual stream after layer 1:

$$\mathbf{S}^{(1)}[\text{pos},:] = \mathbf{E}[\text{pos},:] + \sum_{h=1}^{H_1} \text{Head}_{1,h}(\mathbf{E})[\text{pos},:]$$

Let's consider the induction head mechanism in detail. Suppose we're at position $$N$$ in the sequence, and we've previously seen the pattern "$$A$$ $$B$$" somewhere earlier in the sequence. Now at position $$N-1$$ we see token "$$A$$" again, and we want to predict "$$B$$" at position $$N$$. This requires:

1. A "previous token" head ($$H_1$$) in layer 1 that copies token $$N-1$$'s representation (the new occurrence of "$$A$$") to position $$N$$:

   $$\text{Head}_{1,h_1}(\mathbf{E})[N,:] \approx \mathbf{E}[N-1,:] \cdot \mathbf{W}_V^{(1,h_1)} \cdot \mathbf{W}_O^{(1,h_1)}$$

   This is achieved by having the OV circuit of $$H_1$$ approximate the identity function ($$\mathbf{W}_V^{(1,h_1)} \cdot \mathbf{W}_O^{(1,h_1)} \approx \mathbf{I}$$) and having the QK circuit attend to the previous token.

2. An "induction" head ($$H_2$$) in layer 2 that:
   
   a. Forms query vectors from the updated stream at position $$N$$, which now contains information about token $$N-1$$ (i.e., "$$A$$"):
   
   $$\mathbf{q}_N^{(2,h_2)} = \mathbf{S}^{(1)}[N,:] \cdot \mathbf{W}_Q^{(2,h_2)}$$
   
   This query includes contributions from $$H_1$$:
   
   $$\mathbf{q}_N^{(2,h_2)} = (\mathbf{E}[N,:] + \text{Head}_{1,h_1}(\mathbf{E})[N,:]) \cdot \mathbf{W}_Q^{(2,h_2)}$$
   
   $$\mathbf{q}_N^{(2,h_2)} \approx (\mathbf{E}[N,:] + \mathbf{E}[N-1,:] \cdot \mathbf{W}_V^{(1,h_1)} \cdot \mathbf{W}_O^{(1,h_1)}) \cdot \mathbf{W}_Q^{(2,h_2)}$$
   
   b. Forms key vectors for previous positions:
   
   $$\mathbf{k}_j^{(2,h_2)} = \mathbf{S}^{(1)}[j,:] \cdot \mathbf{W}_K^{(2,h_2)}$$
   
   $$\mathbf{k}_j^{(2,h_2)} = (\mathbf{E}[j,:] + \sum_{h=1}^{H_1} \text{Head}_{1,h}(\mathbf{E})[j,:]) \cdot \mathbf{W}_K^{(2,h_2)}$$
   
   c. Computes attention scores:
   
   $$e_{N,j}^{(2,h_2)} = \frac{\mathbf{q}_N^{(2,h_2)} \cdot \mathbf{k}_j^{(2,h_2)T}}{\sqrt{d_{\text{head}}}}$$
   
   The virtual weight matrix for this K-composition path is:
   
   $$\mathbf{W}_{\text{K-comp}} = \mathbf{W}_V^{(1,h_1)} \cdot \mathbf{W}_O^{(1,h_1)} \cdot \mathbf{W}_K^{(2,h_2)}$$
   
   If $$\mathbf{W}_{\text{K-comp}}$$ is structured appropriately, $$H_2$$ will attend strongly to positions where the token is the same as token $$N-1$$ (i.e., other occurrences of "$$A$$").
   
   d. Once $$H_2$$ attends to previous occurrences of "$$A$$", it then needs to shift attention to the tokens that followed them (i.e., "$$B$$"). This can be implemented through appropriate training of the QK circuit to focus on tokens at position $$j+1$$ when matching with token at position $$j$$.
   
   e. Finally, the OV circuit of $$H_2$$ copies the attended token ("$$B$$") to position $$N$$:
   
   $$\text{Head}_{2,h_2}(\mathbf{S}^{(1)})[N,:] \approx \mathbf{S}^{(1)}[j+1,:] \cdot \mathbf{W}_V^{(2,h_2)} \cdot \mathbf{W}_O^{(2,h_2)}$$
   
   Where $$j$$ is the position of a previous occurrence of "$$A$$".

This complex interaction between heads across layers enables the two-layer transformer to implement in-context learning - predicting "$$B$$" after seeing "$$A$$" based on previously observed "$$A$$ $$B$$" patterns. This capability emerges from the composition of simpler operations and cannot be achieved in models with fewer layers.

The key insight is that the output of the first layer's heads modifies the residual stream in a way that influences the attention patterns of the second layer's heads. This composition enables the emergence of algorithmic capabilities that transcend what each individual head can do in isolation.

## Conclusion

This mathematical framework provides the tools to dissect Transformers into their constituent computational parts: the residual stream as a communication bus, attention heads decomposed into QK and OV circuits, and the powerful concept of composition that allows simple components to build complex algorithms. By analyzing virtual weights, path expansions, and composition strengths, we can start to reverse-engineer the specific computations learned by these models.

This foundation is crucial for understanding phenomena like superposition within these architectures and for developing techniques to extract and validate the features and circuits that implement their remarkable capabilities. In [Part 5]({% post_url 2025-05-25-mechanistic-interpretability-part-5 %}), we will explore the validation of learned features and circuits, building on the mathematical framework established here.

---

## References and Further Reading

This framework is primarily based on the work by Elhage et al. and Olsson et al. in the Transformer Circuits Thread:

-   **Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., ... & Olah, C.** (2021). [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/). *Transformer Circuits Thread*
-   **Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., ... & Olah, C.** (2022). [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/). *Transformer Circuits Thread*
-   Insights on the residual stream and attention also draw from the original Transformer paper: Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *NeurIPS*.