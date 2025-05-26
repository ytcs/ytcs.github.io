---
published: true
layout: post
title: "Mechanistic Interpretability: Part 2 - The Superposition Hypothesis and Dictionary Learning"
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

1.  **Features as Directions:** True, underlying conceptual features ($$\mathbf{f}_1, \mathbf{f}_2, \dots, \mathbf{f}_N$$) correspond to specific vector directions in this $$d_{\text{model}}$$-dimensional space. These are the *feature vectors*.
2.  **Linear Encoding (Locally):** When a set of these features $$\mathcal{S} = \{\mathbf{f}_{i_1}, \mathbf{f}_{i_2}, \dots, \mathbf{f}_{i_k}\}$$ are simultaneously present or active in the input, the model's activation vector $$\mathbf{a} \in \mathbb{R}^{d_{\text{model}}}$$ in that layer is approximately a linear combination of these feature vectors:
    $$\mathbf{a} \approx \sum_{j=1}^{k} c_j \mathbf{f}_{i_j} = c_1 \mathbf{f}_{i_1} + c_2 \mathbf{f}_{i_2} + \dots + c_k \mathbf{f}_{i_k}$$
    where $$c_j$$ are scalar coefficients representing the intensity or presence of feature $$\mathbf{f}_{i_j}$$.
3.  **Non-Privileged Basis (Neuron Basis vs. Feature Basis):** The crucial insight is that these feature directions $$\mathbf{f}_j$$ are generally *not* aligned with the standard basis vectors of the activation space (i.e., the directions corresponding to individual neurons firing). The neuron basis is an accident of architecture; the feature basis is what the network learns is useful.

### Geometric Interpretation of Superposition

Imagine the activation space $$\mathbb{R}^{d_{\text{model}}}$$: each neuron corresponds to one axis in this space. A feature $$\mathbf{f}_j$$ is a vector in this space. If the number of true features $$N$$ is greater than the dimension of the space $$d_{\text{model}}$$, these $$N$$ feature vectors cannot all be orthogonal. They must overlap.

When an activation vector $$\mathbf{a}$$ is formed by a sum of feature vectors, $$ \mathbf{a} = \sum c_j \mathbf{f}_j $$, this vector $$\mathbf{a}$$ will generally have non-zero projections onto many of the standard neuron axes, even if only a few features $$\mathbf{f}_j$$ are active (i.e., have $$c_j \neq 0$$).

A single neuron's activation value is its component of the vector $$\mathbf{a}$$ along its specific axis (e.g., for neuron $$k$$, its activation is $$a_k = \mathbf{a} \cdot \mathbf{e}_k$$, where $$\mathbf{e}_k$$ is the standard basis vector for that neuron). Because the feature vectors $$\mathbf{f}_j$$ are not aligned with these axes $$\mathbf{e}_k$$, a single feature $$\mathbf{f}_j$$ can contribute to the activation of many neurons. Conversely, a single neuron $$k$$ can be active because its axis has a non-zero projection from multiple active feature vectors $$\mathbf{f}_j$$.

This is polysemanticity from a geometric viewpoint: a neuron is active not because one specific feature it "owns" is active, but because the current linear combination of active *features* (which are directions) results in an overall activation vector that has a component along that neuron's axis.

It's also worth noting that the picture of features being randomly superposed in a homogeneous space might be further nuanced by effects from the training process itself. Research such as **Elhage et al. (2023), "Privileged Bases in the Transformer Residual Stream,"** suggests that certain optimizers, like Adam, might introduce per-dimension normalization effects that cause some directions in the residual stream (the neuron basis, or close to it) to become "privileged" or to carry disproportionate representational importance. While this doesn't negate the superposition hypothesis (features can still be superposed within these or other directions), it adds another layer to how feature representations might organize within the model.

## Why Superposition?

-   **Efficiency:** It allows the model to represent a vast number of potentially useful features (concepts, patterns, attributes) without requiring an equally vast number of neurons. This is a form of representational compression.
-   **Flexibility:** New features can potentially be learned and added to the mix without drastically reorganizing existing representations, by finding new "directions" in the existing space.

## Consequences of Superposition

-   **Polysemantic Neurons:** As explained, if features are directions not aligned with neuron axes, then a single neuron can be activated by many different combinations of underlying features that happen to have a projection along its axis.
-   **Difficulty of Direct Interpretation:** Looking at individual neuron activations becomes misleading. A highly active neuron doesn't necessarily mean one specific, interpretable concept is strongly present.
-   **Need for Advanced Techniques:** To find the true, monosemantic features, we need techniques that can look beyond individual neuron activations and identify these underlying feature directions. This motivates the need for methods to uncover these true, monosemantic features. One such powerful technique is dictionary learning using sparse autoencoders, which we will explore after a deeper look at a toy model illustrating superposition.

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
2. Make $$\sum_{j \neq k} (\mathbf{f}_j \cdot \mathbf{f}_k)^2 \approx 0$$, which implies making individual dot products $$\mathbf{f}_j \cdot \mathbf{f}_k \approx 0$$ for all $$j \neq k$$ (so feature representations are orthogonal).

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

With $$f_1=1, f_2=-1, b=0$$, the ReLU model successfully represents two features in a single dimension, incurring zero loss for these single-feature activations. The features are "superposed". The neuron in this 1D space would activate for feature 1 and (with opposite sign in its pre-activation) for feature 2. If we only looked at its activation magnitude, it might seem polysemantic.

**The Role of Sparsity**

The toy model above assumed only one feature is active at a time. What if multiple features are co-active? For example, if $$f_1=1, f_2=-1$$ and both are active, the combined hidden state (assuming linear aggregation) would be $$\mathbf{h} = f_1 + f_2 = 1 + (-1) = 0$$.
Then $$\hat{x}_1 = \text{ReLU}(f_1 \cdot 0 + 0) = 0$$ and $$\hat{x}_2 = \text{ReLU}(f_2 \cdot 0 + 0) = 0$$. Both are incorrectly reconstructed as inactive.
This is **interference**. Superposition relies heavily on the assumption that features are **sparse**—i.e., only a small number of features are active simultaneously.
- If features are sparse, simultaneous activations are rare.
- When they do co-occur, the ReLU can help filter out *small* amounts of interference. If many features are co-active, the interference can overwhelm the signal.

More generally, in higher dimensions ($$D > 1$$), features can arrange themselves in geometric configurations (like vertices of a polytope) to minimize interference while packing many features into the space. The ReLU (or other non-linearities) and biases then work to "carve up" the activation space so that different combinations of true features map to distinct, decodable regions.

This toy model demonstrates that the combination of non-linear activation functions and optimized biases allows neural networks to learn superposed representations, effectively packing more features into fewer dimensions than a linear system could, provided the features exhibit sparsity. This provides a mathematical basis for understanding the polysemanticity observed in neurons and motivates the search for these underlying, monosemantic feature directions.

## Uncovering Features from Superposition with Dictionary Learning

The superposition hypothesis, as explored above, explains how neural networks can represent a vast number of features in a lower-dimensional space, leading to polysemantic neurons. To de-mix these superposed signals and recover interpretable, monosemantic features, we now turn to a powerful technique: **dictionary learning** via **sparse autoencoders**. This approach is crucial for advancing mechanistic interpretability, allowing us to probe deeper than individual neuron activations.

If a neuron activates for multiple, seemingly unrelated concepts (polysemanticity), its individual activation tells us little about the specific feature being represented at that moment. To make progress in mechanistic interpretability, we need to find a representation where features are individuated and interpretable. Dictionary learning aims to find a new basis (the "dictionary") in which the network's activations can be represented sparsely, with each basis vector (dictionary element) corresponding to a single, meaningful concept.

### Sparse Autoencoders: Learning the Dictionary

A sparse autoencoder is an unsupervised neural network trained to reconstruct its input while keeping its internal hidden activations sparse. It consists of an encoder and a decoder:

-   **Encoder:** Maps the input activation vector $$\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$$ (e.g., from a Transformer's residual stream or MLP layer) to a higher-dimensional hidden representation $$\mathbf{f} \in \mathbb{R}^{d_{\text{dict}}}$$, where $$d_{\text{dict}} \gg d_{\text{model}}$$. This is often called an **overcomplete dictionary**.

    $$\mathbf{f} = \text{ReLU}(\mathbf{W}_e (\mathbf{x} - \mathbf{b}_p))$$

    where $$\mathbf{W}_e \in \mathbb{R}^{d_{\text{dict}} \times d_{\text{model}}}$$ are the encoder weights (which learn to project the input onto the dictionary features) and $$\mathbf{b}_p$$ is an optional pre-encoder bias (sometimes called a "pre-shaping bias"). The ReLU (Rectified Linear Unit) ensures that the learned feature activations $$f_i$$ in $$\mathbf{f}$$ are non-negative, which is a common (though not strictly essential for all sparse coding) choice for promoting interpretability, as features are often conceptualized as being present or absent, or having an intensity.

-   **Decoder:** Reconstructs the input $$\hat{\mathbf{x}}$$ from the sparse feature activations $$\mathbf{f}$$:

    $$\hat{\mathbf{x}} = \mathbf{W}_d \mathbf{f} + \mathbf{b}_d$$

    where $$\mathbf{W}_d \in \mathbb{R}^{d_{\text{model}} \times d_{\text{dict}}}$$ are the decoder weights and $$\mathbf{b}_d$$ is an optional decoder bias. The columns of $$\mathbf{W}_d$$ are the **dictionary elements** or **basis vectors** for the learned features. Each column $$\mathbf{d}_j = (\mathbf{W}_d)_{:,j}$$ is a vector in the original $$d_{\text{model}}$$-dimensional space representing one learned feature. The reconstruction $$\hat{\mathbf{x}}$$ is thus a linear combination of these dictionary elements, weighted by the feature activations $$f_j$$. For better interpretability of the feature activations $$f_j$$ as direct measures of feature intensity, the dictionary elements (columns of $$\mathbf{W}_d$$) are often constrained or normalized (e.g., to have unit L2 norm).

#### The Objective Function

The autoencoder is trained to minimize a loss function comprising two main terms:

1.  **Reconstruction Loss:** Ensures that the decoded representation $$\hat{\mathbf{x}}$$ is close to the original input $$\mathbf{x}$$. Typically Mean Squared Error (MSE) is used:

    $$L_{\text{recon}} = ||\mathbf{x} - \hat{\mathbf{x}}||_2^2$$

    This is the squared Euclidean distance (or squared L2 norm of the difference vector) between the original input vector $$\mathbf{x}$$ and its reconstruction $$\hat{\mathbf{x}}$$. Minimizing this term forces the autoencoder to learn a dictionary $$\mathbf{W}_d$$ and an encoding process that captures as much information as possible about $$\mathbf{x}$$ in the feature activations $$\mathbf{f}$$.

2.  **Sparsity Penalty:** Encourages the hidden activations $$\mathbf{f}$$ to be sparse, meaning most elements of $$\mathbf{f}$$ should be zero (or very close to zero) for any given input $$\mathbf{x}$$. The L1 norm is a common choice for this penalty:

    $$L_{\text{sparse}} = \lambda ||\mathbf{f}||_1 = \lambda \sum_i |f_i|$$

    where $$\lambda$$ is a hyperparameter controlling the strength of the sparsity regularization. The L1 norm sums the absolute values of the feature activations $$f_i$$. This penalty is crucial for learning disentangled and interpretable features. Unlike the L2 norm (which penalizes large activations but tends to spread energy across many small activations), the L1 norm is known to promote true sparsity, meaning it encourages many $$f_i$$ to become exactly zero. Geometrically, in the context of minimizing loss subject to a constraint on the norm of $$\mathbf{f}$$, the L1 ball (e.g., a diamond in 2D, an octahedron in 3D) has sharp corners along the axes. Optimization procedures often find solutions at these corners, where some components of $$\mathbf{f}$$ are zero. This is in contrast to the L2 ball (a circle/sphere), which is smooth and tends to yield solutions where components are small but non-zero. The L1 penalty is effectively the closest convex relaxation to the non-convex L0 norm (which directly counts non-zero elements).

The total loss is: $$L = L_{\text{recon}} + L_{\text{sparse}}$$

#### Theoretical Intuition: Recovering Ground Truth Features

Consider a simplified scenario where an activation vector $$\mathbf{x}$$ is a linear combination of a small number of "true" underlying monosemantic features $$\mathbf{s}_j$$ from a ground truth feature set $$\mathcal{S} = \{\mathbf{s}_1, \dots, \mathbf{s}_M\}$$: 

$$\mathbf{x} = \sum_{j \in \text{ActiveSet}} c_j \mathbf{s}_j$$. 

If we can train a sparse autoencoder such that its dictionary elements (columns of $$\mathbf{W}_d$$) align with these true features $$\mathbf{s}_j$$, then the encoder will learn to identify which features are active and the sparse code $$\mathbf{f}$$ will ideally have non-zero entries only for those active features. The L1 penalty encourages solutions where few dictionary elements are used to reconstruct $$\mathbf{x}$$, pushing the autoencoder to find the most compact (and hopefully, most meaningful) representation.

Mathematically, if $$\mathbf{x} = \mathbf{D}^* \mathbf{a}^*$$ where $$\mathbf{D}^*$$ is the matrix of true features (each column is a feature vector) and $$\mathbf{a}^*$$ is a sparse vector of their activations, the goal is for the learned dictionary $$\mathbf{W}_d$$ to approximate $$\mathbf{D}^*$$ (up to permutation and scaling of columns) and for the learned feature activations $$\mathbf{f}$$ to approximate $$\mathbf{a}^*$$. Theoretical results from sparse coding and dictionary learning suggest that this recovery is possible under certain conditions, such as when the true dictionary $$\mathbf{D}^*$$ has incoherent columns (low dot products between different features) and the activations $$\mathbf{a}^*$$ are sufficiently sparse.

### Interpreting Dictionary Elements

If training is successful, each learned dictionary element (a column of $$\mathbf{W}_d$$ or a row of $$\mathbf{W}_e$$ depending on formulation) ideally corresponds to a monosemantic feature. We can then interpret what a model is representing by looking at which dictionary features activate for given inputs. For example, in a language model, one dictionary feature might activate strongly for inputs related to "masculine pronouns," another for "programming concepts," and so on. This provides a much more granular and interpretable view than looking at raw polysemantic neuron activations.

### Insights from "Toy Models of Superposition" (Relevance to Dictionary Learning)

The Anthropic paper "Toy Models of Superposition" provides crucial theoretical insights into how and why superposition occurs, and how dictionary learning might overcome it. Key takeaways include:

1.  **Privileged Basis:** Neural networks often learn features that are not aligned with the standard basis (e.g., individual neurons). Instead, features exist as directions in activation space.
2.  **Geometry of Features:** When the number of learnable features ($$N$$) exceeds the dimensionality of the representation space ($$d_{\text{model}}$$), the model is forced to superpose them. The geometry of these features (e.g., whether they are orthogonal, form a simplex, or have other structures) affects how they are superposed and how easily they can be recovered.
    *   If features are nearly orthogonal and sparse, superposition might involve assigning multiple features to activate a single neuron weakly, or distributing a single feature across multiple neurons.
    *   The "simplex" configuration, where features are as far apart as possible, is an efficient way to pack many features into a lower-dimensional space.
3.  **Sparsity is Key:** The ability to recover features from superposition critically depends on the sparsity of feature activations. If features are dense (many are active simultaneously for any given input), distinguishing them becomes much harder, even with an overcomplete dictionary.
4.  **Phase Changes:** The paper identifies phase transitions where, as the number of features or their sparsity changes, the model abruptly shifts its representation strategy (e.g., from representing features in a privileged basis to a superposed one).

Sparse autoencoders attempt to learn a basis (the dictionary $$\mathbf{W}_d$$) that aligns with these underlying feature directions. The overcompleteness ($$d_{\text{dict}} > d_{\text{model}}$$) provides enough representational capacity to assign individual dictionary elements to individual features, and the L1 penalty encourages the selection of the sparsest explanation for any given activation $$\mathbf{x}$$.

The practical success of this approach has been notably demonstrated in large-scale models. For instance, **Templeton et al. (2024) in "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet,"** successfully applied sparse autoencoders to Anthropic's Claude 3 Sonnet model. They were able to extract a vast number of interpretable features, some of which appeared to be safety-relevant (e.g., related to detecting harmful content). This work underscores that dictionary learning is not merely a theoretical construct but a technique that scales and can yield meaningful insights even in state-of-the-art LLMs.

## Conclusion

The superposition hypothesis offers a critical insight into the workings of neural networks, explaining how they efficiently represent numerous features by encoding them as directions in activation space, rather than assigning them to individual neurons. This leads to the phenomenon of polysemanticity, where single neurons respond to multiple unrelated concepts, complicating direct interpretation.

Understanding superposition is vital, as it pushes us beyond neuron-level analysis. To address the challenge of interpreting these mixed signals, dictionary learning via sparse autoencoders provides a powerful approach. By training an autoencoder to reconstruct activations using a sparse, overcomplete set of dictionary features, we can aim to recover the underlying monosemantic features that the network has learned to superpose. These learned dictionary elements can then offer a more granular and interpretable view of the model's internal representations.

While techniques like sparse autoencoders are not without their challenges and depend on factors such as feature sparsity and careful training, they represent a significant advancement in our ability to decompose complex neural network representations into more understandable components. This pursuit is central to the goals of mechanistic interpretability.

Next, in [Part 3]({% post_url 2025-05-25-mechanistic-interpretability-part-3 %}), we will explore the spectrum of polysemanticity and monosemanticity in more depth. Following that, in [Part 4]({% post_url 2025-05-25-mechanistic-interpretability-part-4 %}), we will introduce the mathematical framework for analyzing Transformer circuits, which builds upon understanding these features. In [Part 5]({% post_url 2025-05-25-mechanistic-interpretability-part-5 %}), we will delve into the crucial aspect of validating these learned features and the circuits they form.

---

## References and Further Reading

-   Original ideas about distributed representations also come from earlier work in connectionism, e.g., by Hinton, Rumelhart, McClelland.
-   **Elhage, N., et al.** (2022). [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_models/index.html). *Transformer Circuits Thread*. (This is a key paper for understanding the mechanics and theory of superposition, and motivates dictionary learning).
-   **Elhage, N., Hume, T., Olsson, C., Nanda, N., Joseph, N., Henighan, T., ... & Olah, C.** (2022). [Softmax Linear Units](https://transformer-circuits.pub/2022/solu/index.html). *Transformer Circuits Thread* (related to feature representation).
-   **Olah, C.** (2022). [Mechanistic Interpretability, Variables, and the Importance of Interpretable Bases](https://distill.pub/2022/mechanistic-interpretability-scope/). *Distill*. (Discusses the concept of features as directions).
-   **Sharkey, L., Nanda, N., Pieler, M., Dosovitskiy, A., & Olah, C.** (2022). [Taking features out of superposition with sparse autoencoders](https://transformer-circuits.pub/2022/toy_models/index.html#appendix-autoencoders). *Transformer Circuits Thread*. (Appendix to Toy Models, directly relevant to autoencoders).
-   **Bricken, T., et al.** (2023). [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html). *Transformer Circuits Thread*.
-   **Templeton, A., et al.** (2024). [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/). *Transformer Circuits Thread*.
-   **Elhage, N., et al.** (2023). [Privileged Bases in the Transformer Residual Stream](https://transformer-circuits.pub/2023/privileged-bases/). *Transformer Circuits Thread*.