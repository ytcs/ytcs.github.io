---
published: true
layout: post
title: "Mechanistic Interpretability: Part 4 - Dictionary Learning and Sparse Autoencoders for Superposition"
categories: machine-learning
date: 2025-05-25
---

In [Part 2]({% post_url 2025-05-25-mechanistic-interpretability-part-2 %}), we introduced the superposition hypothesis, which posits that neural networks represent more features than their dimensionality would classically allow by superposing them in non-privileged bases. While this is efficient, it leads to polysemantic neurons that are hard to interpret. [Part 3]({% post_url 2025-05-25-mechanistic-interpretability-part-3 %}) provided a mathematical framework for analyzing information flow in Transformers. Now, we explore a powerful technique to de-mix superposed signals and recover interpretable, monosemantic features: **dictionary learning** via **sparse autoencoders**.

## The Challenge of Superposition for Interpretability

If a neuron activates for multiple, seemingly unrelated concepts (polysemanticity), its individual activation tells us little about the specific feature being represented at that moment. To make progress in mechanistic interpretability, we need to find a representation where features are individuated and interpretable. Dictionary learning aims to find a new basis (the "dictionary") in which the network's activations can be represented sparsely, with each basis vector (dictionary element) corresponding to a single, meaningful concept.

## Sparse Autoencoders: Learning the Dictionary

A sparse autoencoder is an unsupervised neural network trained to reconstruct its input while keeping its internal hidden activations sparse. It consists of an encoder and a decoder:

-   **Encoder:** Maps the input activation vector $$\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$$ (e.g., from a Transformer's residual stream or MLP layer) to a higher-dimensional hidden representation $$\mathbf{f} \in \mathbb{R}^{d_{\text{dict}}}$$, where $$d_{\text{dict}} \gg d_{\text{model}}$$. This is often called an **overcomplete dictionary**.
    $$\mathbf{f} = \text{ReLU}(\mathbf{W}_e (\mathbf{x} - \mathbf{b}_p))$$
    where $$\mathbf{W}_e \in \mathbb{R}^{d_{\text{dict}} \times d_{\text{model}}}$$ are the encoder weights (the dictionary elements are often considered to be the rows of $$\mathbf{W}_e$$ or columns of $$\mathbf{W}_d$$) and $$\mathbf{b}_p$$ is a pre-encoder bias.

-   **Decoder:** Reconstructs the input $$\hat{\mathbf{x}}$$ from the sparse feature activations $$\mathbf{f}$$:
    $$\hat{\mathbf{x}} = \mathbf{W}_d \mathbf{f} + \mathbf{b}_d$$
    where $$\mathbf{W}_d \in \mathbb{R}^{d_{\text{model}} \times d_{\text{dict}}}$$ are the decoder weights (the dictionary) and $$\mathbf{b}_d$$ is a decoder bias.

### The Objective Function

The autoencoder is trained to minimize a loss function comprising two main terms:

1.  **Reconstruction Loss:** Ensures that the decoded representation $$\hat{\mathbf{x}}$$ is close to the original input $$\mathbf{x}$$. Typically Mean Squared Error (MSE):
    $$L_{\text{recon}} = ||\mathbf{x} - \hat{\mathbf{x}}||_2^2$$

2.  **Sparsity Penalty:** Encourages the hidden activations $$\mathbf{f}$$ to be sparse, meaning most elements of $$\mathbf{f}$$ should be zero for any given input $$\mathbf{x}$$. The L1 norm is a common choice:
    $$L_{\text{sparse}} = \lambda ||\mathbf{f}||_1 = \lambda \sum_i |f_i|$$
    where $$\lambda$$ is a hyperparameter controlling the strength of the sparsity regularization.

The total loss is: $$L = L_{\text{recon}} + L_{\text{sparse}}$$

### Theoretical Intuition: Recovering Ground Truth Features

Consider a simplified scenario where an activation vector $$\mathbf{x}$$ is a linear combination of a small number of "true" underlying monosemantic features $$\mathbf{s}_j$$ from a ground truth feature set $$\mathcal{S} = \{\mathbf{s}_1, \dots, \mathbf{s}_M\}}$$: $$\mathbf{x} = \sum_{j \in \text{ActiveSet}} c_j \mathbf{s}_j$$. If we can train a sparse autoencoder such that its dictionary elements (columns of $$\mathbf{W}_d$$) align with these true features $$\mathbf{s}_j$$, then the encoder will learn to identify which features are active and the sparse code $$\mathbf{f}$$ will ideally have non-zero entries only for those active features. The L1 penalty encourages solutions where few dictionary elements are used to reconstruct $$\mathbf{x}$$, pushing the autoencoder to find the most compact (and hopefully, most meaningful) representation.

Mathematically, if $$\mathbf{x} = \mathbf{D}^* \mathbf{a}^*$$ where $$\mathbf{D}^*$$ is the matrix of true features and $$\mathbf{a}^*$$ is a sparse vector of activations, the goal is for the learned dictionary $$\mathbf{W}_d$$ to approximate $$\mathbf{D}^*$$ (up to permutation and scaling) and for $$\mathbf{f}$$ to approximate $$\mathbf{a}^*$$. Theoretical results from sparse coding and dictionary learning suggest that this recovery is possible under certain conditions, such as when the true dictionary $$\mathbf{D}^*$$ has incoherent columns (low dot products between different features) and the activations $$\mathbf{a}^*$$ are sufficiently sparse.

## Interpreting Dictionary Elements

If training is successful, each learned dictionary element (a column of $$\mathbf{W}_d$$ or a row of $$\mathbf{W}_e$$ depending on formulation) ideally corresponds to a monosemantic feature. We can then interpret what a model is representing by looking at which dictionary features activate for given inputs. For example, in a language model, one dictionary feature might activate strongly for inputs related to "masculine pronouns," another for "programming concepts," and so on. This provides a much more granular and interpretable view than looking at raw polysemantic neuron activations.

## Insights from "Toy Models of Superposition"

The Anthropic paper "Toy Models of Superposition" provides crucial theoretical insights into how and why superposition occurs, and how dictionary learning might overcome it. Key takeaways include:

1.  **Privileged Basis:** Neural networks often learn features that are not aligned with the standard basis (e.g., individual neurons). Instead, features exist as directions in activation space.
2.  **Geometry of Features:** When the number of learnable features ($$N$$) exceeds the dimensionality of the representation space ($$d_{\text{model}}$$), the model is forced to superpose them. The geometry of these features (e.g., whether they are orthogonal, form a simplex, or have other structures) affects how they are superposed and how easily they can be recovered.
    *   If features are nearly orthogonal and sparse, superposition might involve assigning multiple features to activate a single neuron weakly, or distributing a single feature across multiple neurons.
    *   The "simplex" configuration, where features are as far apart as possible, is an efficient way to pack many features into a lower-dimensional space.
3.  **Sparsity is Key:** The ability to recover features from superposition critically depends on the sparsity of feature activations. If features are dense (many are active simultaneously for any given input), distinguishing them becomes much harder, even with an overcomplete dictionary.
4.  **Phase Changes:** The paper identifies phase transitions where, as the number of features or their sparsity changes, the model abruptly shifts its representation strategy (e.g., from representing features in a privileged basis to a superposed one).

Sparse autoencoders attempt to learn a basis (the dictionary $$\mathbf{W}_d$$) that aligns with these underlying feature directions. The overcompleteness ($$d_{\text{dict}} > d_{\text{model}}$$) provides enough representational capacity to assign individual dictionary elements to individual features, and the L1 penalty encourages the selection of the sparsest explanation for any given activation $$\mathbf{x}$$.

## Conclusion

Dictionary learning with sparse autoencoders offers a promising mathematical and practical approach to dissecting superposed representations within neural networks. By learning an overcomplete dictionary of monosemantic features, we can transform polysemantic activations into a sparse, interpretable code. This allows us to identify the specific concepts a model is using and how they combine to form more complex computations.

This technique is not a silver bullet; its success depends on factors like the true sparsity of underlying features, the geometry of the feature manifold, and careful training of the autoencoder. However, it represents a significant step towards decomposing the internal workings of neural networks into understandable components, a core goal of mechanistic interpretability.

In [Part 5]({% post_url 2025-05-25-mechanistic-interpretability-part-5 %}), we will delve into the crucial aspect of validating these learned features and the circuits they form.

---

## References and Further Reading

-   **Elhage, N., Hume, T., Olsson, C., Nanda, N., Joseph, N., Henighan, T., ... & Olah, C.** (2022). [Softmax Linear Units](https://transformer-circuits.pub/2022/solu/index.html). *Transformer Circuits Thread* (related to feature representation).
-   **Sharkey, L., Nanda, N., Pieler, M., Dosovitskiy, A., & Olah, C.** (2022). [Taking features out of superposition with sparse autoencoders](https://transformer-circuits.pub/2022/toy_models/index.html#appendix-autoencoders). *Transformer Circuits Thread*.
-   **Bricken, T., et al.** (2023). [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://arxiv.org/abs/2310.01889). *Anthropic*.
-   **Elhage, N., et al.** (2022). [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_models/index.html). *Transformer Circuits Thread*. (This is a key foundational paper for understanding superposition and the motivation for dictionary learning). 