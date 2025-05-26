---
published: true
layout: post
title: "Mechanistic Interpretability: Part 8 - Attention Head Circuits: Patterns and Functions"
categories: machine-learning
date: 2025-05-25
---

[Part 3]({% post_url 2025-05-25-mechanistic-interpretability-part-3 %}) of this series introduced the mathematical framework for deconstructing Transformer attention heads into Query-Key (QK) and Output-Value (OV) circuits. [Part 7]({% post_url 2025-05-25-mechanistic-interpretability-part-7 %}) discussed circuits more broadly. Now, we delve deeper into the specific **attention patterns and information processing functions** implemented by individual attention heads, treating them as crucial meso-circuits within the larger Transformer architecture. Understanding these patterns is key to deciphering how Transformers process sequential information.

Recall that for an input token representation $$\\mathbf{x}$$, an attention head computes:
- Queries: $$\\mathbf{q} = \\mathbf{x} \\mathbf{W}_Q$$\n- Keys: $$\\mathbf{k} = \\mathbf{x} \\mathbf{W}_K$$\n- Values: $$\\mathbf{v} = \\mathbf{x} \\mathbf{W}_V$$\n
Attention scores $$\\alpha_{ij} \\propto \\text{Softmax}(\\frac{\\mathbf{q}_i \\mathbf{k}_j^T}{\\sqrt{d_k}})$$ determine how much token $$i$$ attends to token $$j$$. The output is $$\\mathbf{o}_i = (\\sum_j \\alpha_{ij} \\mathbf{v}_j) \\mathbf{W}_O$$.\n

## The QK Circuit: Generating Attention Patterns\n

The QK circuit, defined by the effective weight matrix $$\\mathbf{W}_{QK} = \\mathbf{W}_Q \\mathbf{W}_K^T \\in \\mathbb{R}^{d_{\\text{model}} \\times d_{\\text{model}}}$$, determines *where* a head attends. The score $$\\mathbf{x}_i \\mathbf{W}_{QK} \\mathbf{x}_j^T$$ dictates the attention from query token $$i$$ to key token $$j$$. Different structures in $$\\mathbf{W}_{QK}$$ give rise to common attention patterns:\n

1.  **Previous Token Heads:**\n    -   **Pattern:** Attend strongly to the token immediately preceding the current token (i.e., token $$t-1$$ attends to token $$t$$ for queries, or token $$t$$ queries keys from token $$t-1$$).\n    -   **Mathematical Signature (Conceptual):** $$\\mathbf{W}_{QK}$$ is structured such that for a query $$\\mathbf{x}_t$$ (representing token at position $$t$$), the highest score $$\\mathbf{x}_t \\mathbf{W}_{QK} \\mathbf{x}_j^T$$ occurs when $$\\mathbf{x}_j$$ is the representation of token $$t-1$$. This often involves leveraging positional encoding differences.\n    -   **Function:** Crucial for tasks requiring local context, like predicting the next word based on the immediately previous one, or forming bigram-like statistics.\n

2.  **Positional/Fixed Offset Heads (Including BOS/CLS Heads):\n    -   **Pattern:** Attend to tokens at a fixed relative offset from the current token (e.g., $$t-k$$ for some $$k$$), or to a specific absolute position like the beginning-of-sequence (BOS/CLS) token.\n    -   **Mathematical Signature:** $$\\mathbf{W}_{QK}$$ interacts strongly with specific components of the positional encodings, or is biased to select specific absolute positions.\n    -   **Function:** BOS/CLS heads can aggregate global information. Fixed offset heads might capture n-gram like information or specific structural text properties (e.g., attending to a token two positions back).\n

3.  **Same Token Heads (Diagonal Attention):\n    -   **Pattern:** Attend primarily to the current token itself (position $$t$$ attends to position $$t$$).\n    -   **Mathematical Signature:** $$\\mathbf{W}_{QK}$$ might be structured to give high scores when query and key positions are identical, or it may be close to a scaled identity matrix for certain feature subspaces, $$\\mathbf{x}_t \\mathbf{W}_{QK} \\mathbf{x}_t^T \\gg \\mathbf{x}_t \\mathbf{W}_{QK} \\mathbf{x}_j^T$$ for $$j \\neq t$$.\n    -   **Function:** Can be used to gather information from the current token\\\'s own representation, potentially for feature amplification or transformation via the OV circuit.\n

4.  **Content-Based / Pattern-Matching Heads:\n    -   **Pattern:** Attend to tokens whose content (represented by their vector $$\\mathbf{x}_j$$) matches some pattern sought by the query $$\\mathbf{x}_i \\mathbf{W}_Q$$. This is more flexible than fixed positional attention.\n    -   **Mathematical Signature:** The matrix $$\\mathbf{W}_{QK}$$ is structured to compute high similarity scores (e.g., dot products) between query vectors representing a certain type of information and key vectors representing matching information. $$\\mathbf{W}_{QK}$$ might be low-rank, with its singular vectors/eigenvectors corresponding to specific semantic directions or features.\n    -   **Function:** Diverse. Can find related words, entities, or concepts anywhere in the context. For example, a head might learn to attend to all mentions of a specific entity discussed earlier in the text.\n

5.  **Broadcast/Diffuse Heads (Near No-Op for Attention Pattern):\n    -   **Pattern:** Attend very broadly, often with near-uniform attention weights across many tokens, or attend to nothing specific (e.g. only padding tokens).\n    -   **Mathematical Signature:** $$\\mathbf{W}_{QK}$$ might be close to a zero matrix, or structured such that $$\\mathbf{q}_i \\mathbf{k}_j^T / \\sqrt{d_k}$$ produces near-constant values before softmax, leading to diffuse attention. Alternatively, query vectors $$\\mathbf{x}_i \\mathbf{W}_Q$$ might be very small.\n    -   **Function:** Can act as a passthrough for the residual stream if their OV circuit also does minimal processing. Sometimes interpreted as effectively \\\"turned off\\\" or waiting for a very specific trigger. May also contribute to a default aggregation or smoothing.\n

## The OV Circuit: Processing Attended Information\n

Once the QK circuit determines *where* to attend (weights $$\\alpha_{ij}$$), the OV circuit, defined by $$\\mathbf{W}_{OV} = \\mathbf{W}_V \\mathbf{W}_O \\in \\mathbb{R}^{d_{\\text{model}} \\times d_{\\text{model}}}$$, determines *what information is moved and how it is processed* from the attended value vectors $$\\mathbf{v}_j = \\mathbf{x}_j \\mathbf{W}_V$$. The output written to the residual stream is $$\\sum_j \\alpha_{ij} (\\mathbf{x}_j \\mathbf{W}_V) \\mathbf{W}_O$$.\n

Common functions implemented by the OV circuit include:\n

1.  **Copying Information:**\n    -   **Mathematical Signature:** $$\\mathbf{W}_{OV} = \\mathbf{W}_V \\mathbf{W}_O \\approx c\\mathbf{I}$$ (a scaled identity matrix) for the subspace of features being attended to. This means the attended information is largely copied through to the residual stream.\n    -   **Combined with Pattern:** A previous token head with a copying OV circuit will copy the representation of token $$t-1$$ to position $$t$$.\n

2.  **Feature Transformation/Extraction:**\n    -   **Mathematical Signature:** $$\\mathbf{W}_{OV}$$ is not an identity matrix. It applies a specific linear transformation to the attended value vectors. $$\\mathbf{W}_{OV}$$ might be low-rank, indicating it projects information onto a smaller subspace, effectively extracting or emphasizing specific features from the attended tokens.\n    -   **Combined with Pattern:** A content-based head might attend to all mentions of \\\"dates,\\\" and its OV circuit could transform these date representations into a common format or extract a specific feature like \\\"is a past date.\\\"\n

3.  **Information Suppression/No-Op Output:**\n    -   **Mathematical Signature:** $$\\mathbf{W}_{OV} \\approx \mathbf{0}$$ (zero matrix), or the value vectors $$\\mathbf{x}_j \\mathbf{W}_V$$ are themselves near zero. Even if the head attends somewhere, it writes little or nothing back to the residual stream.\n    -   **Combined with Pattern:** A head might have a clear attention pattern but a near-zero OV circuit, making it effectively a no-op in terms of output, though its attention scores could still be read by other components in some theoretical probing scenarios.\n

## Analyzing Head Behavior: Connecting Weights to Function\n

Understanding an attention head requires analyzing both its QK and OV circuits:\n-   **Singular Value Decomposition (SVD) of $$\\mathbf{W}_{QK}$$ and $$\\mathbf{W}_{OV}$$**: Can reveal the principal directions (features) these matrices operate on and their effective rank. A low effective rank indicates specialization.\n-   **Max Activating Examples for $$\\mathbf{q}$$ and $$\\mathbf{k}$$ directions:** What kind of input features cause high query/key vector components in the important singular directions of $$\\mathbf{W}_Q$$ and $$\\mathbf{W}_K$$?\n-   **Probing $$\\mathbf{W}_{OV}$$:** How does $$\\mathbf{W}_{OV}$$ transform known input feature directions?\n

## Attention Heads as Composable Meso-Circuits\n

Each attention head, with its specific QK pattern and OV function, acts as a meso-circuit. These are the building blocks that, through composition (Q-composition, K-composition, V-composition, as discussed in [Part 3]({% post_url 2025-05-25-mechanistic-interpretability-part-3 %}) and [Part 7]({% post_url 2025-05-25-mechanistic-interpretability-part-7 %})), form more complex macro-circuits. For example, an induction head (which we will explore in [Part 9]({% post_url 2025-05-25-mechanistic-interpretability-part-9 %})) is typically formed by the composition of at least two simpler attention heads (e.g., a previous token head and a pattern-matching head).\n

## Conclusion\n

Individual attention heads in Transformers are not monolithic; they are decomposable circuits whose QK components determine attention patterns and whose OV components process the attended information. By analyzing the mathematical properties of their weight matrices ($$\\mathbf{W}_Q, \\mathbf{W}_K, \\mathbf{W}_V, \\mathbf{W}_O$$), we can identify recurring patterns like previous token attention, content-based attention, and functions like copying or feature transformation. These individual head circuits are the fundamental meso-circuits that compose to create the sophisticated algorithmic capabilities of Transformers.\n

Next, we will examine one of the most celebrated examples of such composed circuits: induction heads.\n

In [Part 9 - Induction Heads: The Mechanics of In-Context Learning]({% post_url 2025-05-25-mechanistic-interpretability-part-9 %})\n

---\n

## References and Further Reading\n

-   **Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S.** (2020). [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/). *Distill*.\n-   **Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., ... & Olah, C.** (2021). [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/). *Transformer Circuits Thread*.\n-   **Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., ... & Olah, C.** (2022). [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/). *Transformer Circuits Thread*. (Provides many examples of different head types).\n-   **Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I.** (2019). [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, Others Can Be Pruned](https://arxiv.org/abs/1905.09418). *ACL*. (Early work on identifying specialized head roles).\n 