---
published: true
layout: post
title: "Mechanistic Interpretability: Part 2 - The Superposition Hypothesis"
categories: machine-learning
date: 2025-05-25
---

In [Part 1]({% post_url 2025-05-25-mechanistic-interpretability-part-1 %}), we introduced mechanistic interpretability and the circuits paradigm, which posits that neural networks learn meaningful features and connect them into computational circuits. A key challenge to this vision arises when individual components of a network, like neurons, appear to respond to many unrelated concepts simultaneously. This phenomenon, known as **polysemanticity**, complicates the straightforward interpretation of individual neural units. The **superposition hypothesis** offers a compelling explanation for polysemanticity, suggesting that neural networks represent more features than their dimensionality would naively suggest by encoding them in a distributed, overlapping manner.

## Polysemanticity: The Challenge

Imagine a single neuron in a language model that activates strongly for inputs related to Shakespearean sonnets, but also for Python programming errors, and also for discussions about baking sourdough. Such a neuron is polysemantic. If this neuron is active, what specific concept is the model currently processing? Its activation alone is ambiguous.

If many neurons are polysemantic, reverse-engineering the specific algorithms the network uses becomes incredibly difficult. We need a way to understand how distinct concepts are represented, even if not by individual neurons.

## The Superposition Hypothesis Explained

The superposition hypothesis, prominently discussed in the context of Transformer models by researchers at Anthropic and elsewhere, proposes that neural networks learn to represent a large number of features ($$N$$) within a lower-dimensional activation space ($$\mathbb{R}^{d_{\text{model}}}$$, where $$d_{\text{model}} < N$$). This is achieved by representing features not as individual neurons firing, but as **directions in activation space**.

If a model has $$d_{\text{model}}$$ neurons in a layer, it has a $$d_{\text{model}}$$-dimensional space of possible activation vectors. The hypothesis states:

1.  **Features as Directions:** True, underlying conceptual features ($$\mathbf{f}_1, \mathbf{f}_2, \dots, \mathbf{f}_N$$) correspond to specific vector directions in this $$d_{\text{model}}$$-dimensional space.
2.  **Linear Encoding (Locally):** When a set of these features $$\mathcal{S} = \{\mathbf{f}_{i_1}, \mathbf{f}_{i_2}, \dots, \mathbf{f}_{i_k}\}$$ are simultaneously present or active in the input, the model's activation vector $$\mathbf{a} \in \mathbb{R}^{d_{\text{model}}}$$ in that layer is approximately a linear combination of these feature vectors:
    $$\mathbf{a} \approx c_1 \mathbf{f}_{i_1} + c_2 \mathbf{f}_{i_2} + \dots + c_k \mathbf{f}_{i_k}$$
    where $$c_j$$ are scalar coefficients representing the intensity or presence of feature $$\mathbf{f}_{i_j}$$.
3.  **Non-Privileged Basis (Neuron Basis vs. Feature Basis):** The crucial insight is that these feature directions $$\mathbf{f}_j$$ are generally *not* aligned with the standard basis vectors of the activation space (i.e., the directions corresponding to individual neurons firing). The neuron basis is an accident of architecture; the feature basis is what the network learns is useful.

### Geometric Interpretation of Superposition

Imagine the activation space $$\mathbb{R}^{d_{\text{model}}}$$: each neuron corresponds to one axis in this space. A feature $$\mathbf{f}_j$$ is a vector in this space. If the number of true features $$N$$ is greater than the dimension of the space $$d_{\text{model}}$$, these $$N$$ feature vectors cannot all be orthogonal. They must overlap.

When an activation vector $$\mathbf{a}$$ is formed by a sum of feature vectors, $$ \mathbf{a} = \sum c_j \mathbf{f}_j $$, this vector $$\mathbf{a}$$ will generally have non-zero projections onto many of the standard neuron axes, even if only a few features $$\mathbf{f}_j$$ are active (i.e., have $$c_j \neq 0$$).

A single neuron's activation value is its component of the vector $$\mathbf{a}$$ along its specific axis (e.g., for neuron $$k$$, its activation is $$a_k = \mathbf{a} \cdot \mathbf{e}_k$$, where $$\mathbf{e}_k$$ is the standard basis vector for that neuron). Because the feature vectors $$\mathbf{f}_j$$ are not aligned with these axes $$\mathbf{e}_k$$, a single feature $$\mathbf{f}_j$$ can contribute to the activation of many neurons. Conversely, a single neuron $$k$$ can be active because its axis has a non-zero projection from multiple active feature vectors $$\mathbf{f}_j$$.

This is polysemanticity from a geometric viewpoint: a neuron is active not because one specific feature it "owns" is active, but because the current linear combination of active *features* (which are directions) results in an overall activation vector that has a component along that neuron's axis.

## Why Superposition?

-   **Efficiency:** It allows the model to represent a vast number of potentially useful features (concepts, patterns, attributes) without requiring an equally vast number of neurons. This is a form of representational compression.
-   **Flexibility:** New features can potentially be learned and added to the mix without drastically reorganizing existing representations, by finding new "directions" in the existing space.

## Consequences of Superposition

-   **Polysemantic Neurons:** As explained, if features are directions not aligned with neuron axes, then a single neuron can be activated by many different combinations of underlying features that happen to have a projection along its axis.
-   **Difficulty of Direct Interpretation:** Looking at individual neuron activations becomes misleading. A highly active neuron doesn't necessarily mean one specific, interpretable concept is strongly present.
-   **Need for Advanced Techniques:** To find the true, monosemantic features, we need techniques that can look beyond individual neuron activations and identify these underlying feature directions. This motivates methods like dictionary learning using sparse autoencoders (covered in [Part 4]({% post_url 2025-05-25-mechanistic-interpretability-part-4 %})).

## Deeper Dive: A Toy Model of Superposition

To make the concept of superposition more concrete, let's explore a simplified toy model based on the work of Elhage et al. (2022). This model helps illustrate how non-linearities enable superposition, which is typically not optimal in purely linear systems.

**Assumptions:**
- We have $$N$$ distinct features we want the model to represent.
- These features are encoded in a hidden layer of dimensionality $$D$$.
- For simplicity, features are **sparse**: only one feature is active at any given time. (We'll discuss relaxing this later).
- Each feature $$k$$ is represented by a vector $$\mathbf{f}_k \in \mathbb{R}^D$$ in the hidden layer.
- The model tries to reconstruct which feature is active. If feature $$k$$ is active (input $$x_k=1$$, others $$x_j=0$$), the hidden state is $$\mathbf{h} = \mathbf{f}_k$$.
- A linear decoder with weights (matching the encoder vectors) $$\mathbf{f}_j$$ for each feature $$j$$ attempts to reconstruct the input. So, the reconstructed activity for feature $$j$$ is 

$$ \hat{x}_j = \sigma(\mathbf{f}_j \cdot \mathbf{h} + b_j) $$

, where $$b_j$$ is a bias term.

The goal of the model is to minimize reconstruction error. If feature $$k$$ is active, we want $$\hat{x}_k \approx 1$$ and $$\hat{x}_j \approx 0$$ for $$j \neq k$$. The loss function, assuming feature $$k$$ is active, would be:

$$ L_k = (1 - \hat{x}_k)^2 + \sum_{j \neq k} (0 - \hat{x}_j)^2 $$

The total loss is the average over all possible active features: 

$$ L = \mathbb{E}_k [L_k] $$


**1. The Linear Case (No Superposition)**

If the activation function is linear (i.e., identity, $$\sigma(z)=z$$) and we set biases $$b_j=0$$ for simplicity, the reconstruction is $$\hat{x}_j = \mathbf{f}_j \cdot \mathbf{h}$$.
When feature $$k$$ is active, $$\mathbf{h} = \mathbf{f}_k$$. So, $$\hat{x}_j = \mathbf{f}_j \cdot \mathbf{f}_k$$.
The loss for active feature $$k$$ becomes:

$$ L_k^{\text{linear}} = (1 - \mathbf{f}_k \cdot \mathbf{f}_k)^2 + \sum_{j \neq k} (\mathbf{f}_j \cdot \mathbf{f}_k)^2 $$

$$ L_k^{\text{linear}} = (1 - \|\mathbf{f}_k\|^2)^2 + \sum_{j \neq k} (\mathbf{f}_j \cdot \mathbf{f}_k)^2 $$

To minimize this loss, the model will try to:
1. Make $$\|\mathbf{f}_k\|^2 \approx 1$$ for all features $$k$$ it represents (so they have unit norm).
2. Make $$\mathbf{f}_j \cdot \mathbf{f}_k \approx 0$$ for all $$j \neq k$$ (so feature representations are orthogonal).

If the number of features $$N$$ is greater than the hidden dimensionality $$D$$ ($$N > D$$), it's impossible for all $$N$$ feature vectors $$\mathbf{f}_k \in \mathbb{R}^D$$ to be orthogonal and have unit norm. A linear model will optimally select $$D$$ features, make their representations $$\{\mathbf{f}_1, \dots, \mathbf{f}_D\}$$ orthogonal unit vectors (forming a basis for $$\mathbb{R}^D$$), and effectively ignore the remaining $$N-D$$ features by setting their representation norms $$\|\mathbf{f}_k\|$$ to zero. Thus, **linear models typically do not exhibit superposition**; they learn a subset of features that fit orthogonally into the available dimensions.

**2. The Non-Linear Case (with ReLU) Enables Superposition**

Now, let's introduce a non-linearity, specifically ReLU: $$\sigma(z) = \text{ReLU}(z) = \max(0, z)$$.
The reconstruction for feature $$j$$, when feature $$k$$ is active ($$\mathbf{h}=\mathbf{f}_k$$), is 

$$ \hat{x}_j = \text{ReLU}(\mathbf{f}_j \cdot \mathbf{f}_k + b_j) $$

The loss for active feature $$k$$ is:

$$ L_k^{\text{ReLU}} = (1 - \text{ReLU}(\mathbf{f}_k \cdot \mathbf{f}_k + b_k))^2 + \sum_{j \neq k} (0 - \text{ReLU}(\mathbf{f}_j \cdot \mathbf{f}_k + b_j))^2 $$

To minimize this loss, the network needs to satisfy two conditions for each active feature $$k$$:
1.  **Correctly identify active feature:** $$\text{ReLU}(\|\mathbf{f}_k\|^2 + b_k) \approx 1$$. This means $$\|\mathbf{f}_k\|^2 + b_k \approx 1$$.
2.  **Suppress inactive features:** $$\text{ReLU}(\mathbf{f}_j \cdot \mathbf{f}_k + b_j) \approx 0$$ for $$j \neq k$$. This means $$\mathbf{f}_j \cdot \mathbf{f}_k + b_j \le 0$$.

The crucial difference is the condition $$\mathbf{f}_j \cdot \mathbf{f}_k + b_j \le 0$$. Unlike the linear case requiring strict orthogonality ($$\mathbf{f}_j \cdot \mathbf{f}_k = 0$$), the ReLU allows for some non-zero dot product (interference) as long as it's pushed below zero by the bias $$b_j$$ before passing through the ReLU. This flexibility is key to superposition.

**Example: 2 Features in 1 Dimension ($$N=2, D=1$$)**

Let $$f_1$$ and $$f_2$$ be scalar representations since $$D=1$$.
-   **Linear Model:** The loss involves terms like $$(1-f_1^2)^2$$, $$(1-f_2^2)^2$$, and $$(f_1 f_2)^2$$. This is minimized if, say, $$f_1 = \pm 1$$ and $$f_2 = 0$$. Only one feature is learned.
-   **ReLU Model:** Can we represent both features? Let's try $$f_1 = 1$$ and $$f_2 = -1$$. This is an "antipodal pair".
    Let's choose a bias $$b_1 = b_2 = b = 0$$ (a common finding is that optimal biases are often near zero for sparse features).

    *   If feature 1 is active ($$\mathbf{h} = f_1 = 1$$):
        *   $$\hat{x}_1 = \text{ReLU}(f_1 \cdot f_1 + b) = \text{ReLU}(1 \cdot 1 + 0) = \text{ReLU}(1) = 1$$. (Correct)
        *   $$\hat{x}_2 = \text{ReLU}(f_2 \cdot f_1 + b) = \text{ReLU}(-1 \cdot 1 + 0) = \text{ReLU}(-1) = 0$$. (Correct)
        The loss terms are zero.

    *   If feature 2 is active ($$\mathbf{h} = f_2 = -1$$):
        *   $$\hat{x}_1 = \text{ReLU}(f_1 \cdot f_2 + b) = \text{ReLU}(1 \cdot (-1) + 0) = \text{ReLU}(-1) = 0$$. (Correct)
        *   $$\hat{x}_2 = \text{ReLU}(f_2 \cdot f_2 + b) = \text{ReLU}(-1 \cdot (-1) + 0) = \text{ReLU}(1) = 1$$. (Correct)
        The loss terms are zero.

With $$f_1=1, f_2=-1, b=0$$, the ReLU model successfully represents two features in a single dimension. The features are "superposed". The neuron in this 1D space would activate for feature 1 and (with opposite sign in its pre-activation) for feature 2. If we only looked at its activation magnitude, it might seem polysemantic.

**The Role of Sparsity**

The toy model above assumed only one feature is active at a time. What if multiple features are co-active? For example, if $$f_1=1, f_2=-1$$ and both are active, the combined hidden state (assuming linear aggregation) would be $$\mathbf{h} = f_1 + f_2 = 1 + (-1) = 0$$.
Then $$\hat{x}_1 = \text{ReLU}(f_1 \cdot 0 + 0) = 0$$ and $$\hat{x}_2 = \text{ReLU}(f_2 \cdot 0 + 0) = 0$$. Both are incorrectly reconstructed as inactive.
This is **interference**. Superposition relies heavily on the assumption that features are **sparse**—i.e., only a small number of features are active simultaneously.
- If features are sparse, simultaneous activations are rare.
- When they do co-occur, the ReLU can help filter out *small* amounts of interference. If many features are co-active, the interference can overwhelm the signal.

More generally, in higher dimensions ($$D > 1$$), features can arrange themselves in geometric configurations (like vertices of a polytope) to minimize interference while packing many features into the space. The ReLU (or other non-linearities) and biases then work to "carve up" the activation space so that different combinations of true features map to distinct, decodable regions.

This toy model demonstrates that the combination of non-linear activation functions and optimized biases allows neural networks to learn superposed representations, effectively packing more features into fewer dimensions than a linear system could, provided the features exhibit sparsity. This provides a mathematical basis for understanding the polysemanticity observed in neurons and motivates the search for these underlying, monosemantic feature directions.

## Conclusion

The superposition hypothesis provides a powerful theoretical lens for understanding why individual neurons in complex models like Transformers are often polysemantic. By representing features as directions in activation space rather than tying them to individual neurons, models can efficiently encode a large number of concepts. This, however, necessitates moving beyond naive neuron-level interpretations and developing methods to uncover these underlying, superposed feature directions.

Understanding superposition is crucial because it reshapes our approach to finding interpretable units within neural networks, guiding us towards techniques that can de-mix these overlapping signals and reveal the true semantic building blocks of the model's computations.

Next, in [Part 3]({% post_url 2025-05-25-mechanistic-interpretability-part-3 %}), we will introduce the mathematical framework for analyzing Transformer circuits, which will be essential for understanding how these features and components interact.

---

## References and Further Reading

-   **Elhage, N., et al.** (2022). [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_models/index.html). *Transformer Circuits Thread*. (This is a key paper for understanding the mechanics and theory of superposition).
-   **Olah, C.** (2022). [Mechanistic Interpretability, Variables, and the Importance of Interpretable Bases](https://distill.pub/2022/mechanistic-interpretability-scope/). *Distill*. (Discusses the concept of features as directions).
-   Original ideas about distributed representations also come from earlier work in connectionism, e.g., by Hinton, Rumelhart, McClelland.