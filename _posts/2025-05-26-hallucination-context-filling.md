---
published: true
layout: post
title: "Hallucination and Context Filling: A Toy Model"
categories: machine-learning
date: 2025-05-26
math: true
---

I have observed that hallucination likelihood might increase as more information fills the LLM's context window—as "context richness" grows. This post explores a plausible theoretical "toy model" to understand this empirical observation. We focus on how a specific mechanism—logit computation involving a $$1/N$$ scaling—can lead to a flatter distribution of pre-softmax logits, potentially increasing model uncertainty and thus hallucination.

## The Toy Model: Core Components

Let's establish the basic elements of our model.

### Context Richness and Logit Influence Vectors

We consider $$N$$ distinct "contextual features" or informational cues that the LLM processes. As the input context becomes richer, $$N$$ increases.
Each feature $$k$$ (for $$k=1, \dots, N$$) exerts an "effective logit influence vector" $$\mathbf{p}_k \in \mathbb{R}^V$$ on the next token prediction, where $$V$$ is the vocabulary size. The component $$p_{ki}$$ is the influence of feature $$k$$ on the logit for the $$i$$-th vocabulary token.

### The Key Mechanism: Logit Computation with $$1/N$$ Scaling

The cornerstone of our toy model is how these individual influences combine:

The final logit vector $$\mathbf{L}^{(N)} \in \mathbb{R}^V$$ (before the softmax function) is postulated to be the arithmetic mean of the effective logit influence vectors from the $$N$$ active contextual features:

$$
\mathbf{L}^{(N)} = \frac{1}{N} \sum_{k=1}^{N} \mathbf{p}_k 
$$

This explicit $$1/N$$ scaling is critical. If influences were merely summed (a $$1/1$$ scaling), logit variance would generally grow with $$N$$. If scaled by $$1/\sqrt{N}$$ (common in statistical averaging of i.i.d. variables), variance would remain constant (for uncorrelated $$\mathbf{p}_k$$). The $$1/N$$ scaling ensures that, under certain conditions, variance *decreases* as more features are added.

Why assume such scaling?
1.  **Mechanistic Aggregation in Circuits:** Neural architectures might implement such scaling. For example, if $$N$$ processed hidden states $$\mathbf{h}_k^{\text{proc}}$$ (derived from the contextual cues) are explicitly averaged before a final linear projection $$W_{out}$$ to the logit space:
    $$
    \mathbf{L}^{(N)} \approx W_{out} \left( \frac{1}{N} \sum_{k=1}^N \mathbf{h}_k^{\text{proc}} \right) = \frac{1}{N} \sum_{k=1}^N (W_{out} \mathbf{h}_k^{\text{proc}})
    $$
    Defining $$\mathbf{p}_k = W_{out} \mathbf{h}_k^{\text{proc}} - \mathbb{E}[W_{out} \mathbf{h}_k^{\text{proc}}]$$ (i.e., centered influences) directly yields the postulated form.

2.  **Neural Tangent Kernel (NTK) Implication:** In very wide networks (the NTK regime), if an input is a composition of $$N$$ elements $$\mathbf{c}_k$$, and the network's feature extractor $$\phi(\cdot)$$ effectively averages these (i.e., $$\phi(\text{input}) \approx \frac{1}{N} \sum_k \phi(\mathbf{c}_k)$$), then the logits, being a linear transformation of $$\phi(\text{input})$$, would also exhibit this scaling.

### Simplifying Assumptions for Logit Influences ($$\mathbf{p}_k$$)

For our toy model, we make the following simplifying statistical assumptions about the influence vectors $$\mathbf{p}_k$$:
1.  **Zero Mean Influence:** $$\mathbb{E}[\mathbf{p}_k] = \mathbf{0}$$. This means $$\mathbf{p}_k$$ represent deviations from a baseline influence, simplifying variance calculations.
2.  **Common Covariance:** The covariance matrix of any single feature's influence vector $$\mathbf{p}_k$$ is the same: $$\mathrm{Cov}(\mathbf{p}_k) = \mathbf{\Sigma}_p = \mathbb{E}[\mathbf{p}_k \mathbf{p}_k^T]$$.
3.  **Inter-Feature Covariance:** The covariance between influence vectors from different features $$k \neq l$$ is $$\mathrm{Cov}(\mathbf{p}_k, \mathbf{p}_l) = \mathbf{C}_{kl} = \mathbb{E}[\mathbf{p}_k \mathbf{p}_l^T]$$.

## Deriving the Expected Variance of Logits

Our goal is to see how the spread (sample variance) of the logit values $$L_i^{(N)}$$ across the vocabulary changes with $$N$$. The sample variance is $$\mathrm{Var}_i(L_i^{(N)}) = \frac{1}{V} \sum_{i=1}^{V} (L_i^{(N)} - \bar{L}^{(N)})^2$$, where $$\bar{L}^{(N)}$$ is the mean logit.

1.  **Covariance of the Scaled Logit Vector ($$\mathbf{L}^{(N)}$$):**
    Using $$\mathbf{L}^{(N)} = \frac{1}{N} \sum \mathbf{p}_k$$ and properties of covariance:
    $$
    \mathbf{\Sigma}_{\mathbf{L}^{(N)}} = \mathrm{Cov}\left(\frac{1}{N} \sum_{k=1}^{N} \mathbf{p}_k\right) = \frac{1}{N^2} \mathrm{Cov}\left(\sum_{k=1}^{N} \mathbf{p}_k\right)
    $$
    $$
    \mathbf{\Sigma}_{\mathbf{L}^{(N)}} = \frac{1}{N^2} \left( \sum_{k=1}^{N} \mathrm{Cov}(\mathbf{p}_k) + \sum_{k \neq l} \mathrm{Cov}(\mathbf{p}_k, \mathbf{p}_l) \right)
    $$
    Applying our assumptions:
    $$
    \mathbf{\Sigma}_{\mathbf{L}^{(N)}} = \frac{1}{N^2} \left( N \mathbf{\Sigma}_p + \sum_{k \neq l} \mathbf{C}_{kl} \right)
    $$

2.  **Expected Sample Variance of Logits:**
    Since $$\mathbb{E}[L_i^{(N)}] = 0$$ (due to $$\mathbb{E}[\mathbf{p}_k] = \mathbf{0}$$), the expected sample variance can be written using a function $$S(\mathbf{A}) = \frac{1}{V} \mathrm{Tr}(\mathbf{A}) - \frac{1}{V^2} \mathbf{1}^T \mathbf{A} \mathbf{1}$$, which measures the average variance of components of a zero-mean random vector with covariance $$\mathbf{A}$$:
    $$
    \mathbb{E}\left[\mathrm{Var}_i(L_i^{(N)})\right] = S(\mathbf{\Sigma}_{\mathbf{L}^{(N)}})
    $$
    Substituting the expression for $$\mathbf{\Sigma}_{\mathbf{L}^{(N)}}$$ and using the linearity of $$S(\cdot)$$:
    $$
    \mathbb{E}\left[\mathrm{Var}_i(L_i^{(N)})\right] = \frac{1}{N^2} \left( N S(\mathbf{\Sigma}_p) + \sum_{k \neq l} S(\mathbf{C}_{kl}) \right)
    $$

3.  **Introducing Simplified Correlation:**
    Let $$V_p = S(\mathbf{\Sigma}_p)$$ be the inherent expected sample variance from a single feature's influence. To simplify the correlation term, we assume a homogeneous structure: $$S(\mathbf{C}_{kl}) = \rho_{kl} V_p$$, where $$\rho_{kl}$$ is a scalar effective correlation. Let $$\bar{\rho} = \frac{1}{N(N-1)} \sum_{k \neq l} \rho_{kl}$$ be the average inter-feature correlation (for $$N \ge 2$$; if $$N=1$$, this term is zero).
    The sum becomes $$\sum_{k \neq l} S(\mathbf{C}_{kl}) = N(N-1)\bar{\rho} V_p$$.

4.  **Final Result for Expected Logit Variance:**
    Substituting this into the equation gives:
    $$
    \mathbb{E}\left[\mathrm{Var}_i(L_i^{(N)})\right] = \frac{1}{N^2} \left( N V_p + N(N-1)\bar{\rho} V_p \right)
    $$
    Simplifying this expression, we get:
    $$
    \mathbb{E}\left[\mathrm{Var}_i(L_i^{(N)})\right] = V_p \left( \frac{N + N(N-1)\bar{\rho}}{N^2} \right) = V_p \left( \frac{1 + (N-1)\bar{\rho}}{N} \right)
    $$
    This can be conveniently rewritten as:
    $$
    \mathbb{E}\left[\mathrm{Var}_i(L_i^{(N)})\right] = V_p \left( \frac{1 - \bar{\rho}}{N} + \bar{\rho} \right) \quad (\text{for } N \ge 1)
    $$

## What the Toy Model Shows

This final equation reveals how the expected variance of logits across the vocabulary changes with context richness ($$N$$) and average inter-feature correlation ($$\bar{\rho}$$):

1.  **Independent Features ($$\bar{\rho} = 0$$):**
    If contextual influences are uncorrelated, $$\mathbb{E}\left[\mathrm{Var}_i(L_i^{(N)})\right] = V_p/N$$. The logit variance decreases inversely with $$N$$. More diverse information leads to a tighter clustering of logit values.

2.  **Positively Correlated Features ($$0 < \bar{\rho} < 1$$):**
    The variance still decreases as $$N$$ increases due to the $$(1-\bar{\rho})/N$$ term. However, as $$N \to \infty$$, the variance approaches a non-zero floor: $$\lim_{N\to\infty} \mathbb{E}\left[\mathrm{Var}_i(L_i^{(N)})\right] = V_p \bar{\rho}$$. Shared information (correlation) limits how much this scaling can smooth out the logits.

3.  **Perfectly Correlated Features ($$\bar{\rho} = 1$$):**
    If all influences are perfectly correlated, $$\mathbb{E}\left[\mathrm{Var}_i(L_i^{(N)})\right] = V_p$$. Scaling identical (up to a constant factor) information provides no reduction in variance.

The core insight: as long as contextual features are not perfectly correlated ($$\bar{\rho} < 1$$), the $$1/N$$ scaling mechanism causes the expected variance of the logits to decrease as context richness $$N$$ increases.

## Link to Hallucination

A lower variance among logit values means the logits become more similar to each other. This has a direct consequence for the softmax probability distribution used to select the next token:
*   **Flatter Distribution:** Similar logit values lead to a more uniform probability distribution over the vocabulary.
*   **Increased Entropy:** A flatter distribution has higher Shannon entropy ($$H = -\sum p_i \log p_i$$), signifying greater model uncertainty.

This increased uncertainty—where the model is less "peaked" or confident in its next token choice—is hypothesized to increase the likelihood of sampling less coherent, factually ungrounded, or nonsensical tokens, characteristic of hallucinations. The model, by applying this scaling to many potentially disparate signals (especially if $$\bar{\rho}$$ is small), might lose strong individual signals in a sea of moderate ones, leading to this uncertain state.

## Conclusion

This toy model, built upon $$1/N$$ scaling of logits, demonstrates a potential mechanism for how increasing context richness ($$N$$) can decrease the expected variance of an LLM's pre-softmax logits. The derived relationship, $$\mathbb{E}[\text{Var}] = V_p ( (1 - \bar{\rho})/N + \bar{\rho} )$$, highlights that this reduction is most significant when new contextual cues are diverse (low $$\bar{\rho}$$). The resulting flatter softmax distribution (higher uncertainty) offers a plausible mathematical pathway to understanding why LLMs might become more prone to hallucination when processing very long and information-rich contexts.