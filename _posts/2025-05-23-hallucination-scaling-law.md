---
published: true
layout: post
title: "Scaling Law of LLM Hallucination"
categories: machine-learning
date: 2025-05-23
---

Understanding why Large Language Models (LLMs) sometimes 'hallucinate'—generating fluent but factually incorrect or nonsensical information—is an important challenge. While a complete picture remains elusive, we can construct theoretical models to gain insights. These models allow us to explore how the propensity for hallucination might scale with factors like the volume of training data, the inherent complexity of language, and the way we ask models to generate text. This post delves into one such mathematical exploration. We'll derive, from a set of foundational assumptions, a lower bound on hallucination probability and examine how this bound is governed by these critical parameters.

## Core Concepts and Notation

We begin by defining the essential components and notation used throughout this derivation.

An LLM with parameters $$\theta$$ models a conditional probability distribution $$P(Y \mid X, \theta)$$, where $$X = (x_1, \ldots, x_{L_X-1})$$ is the input context of length $$L_X$$, and $$Y$$ is the next token. The model outputs logits $$\mathbf{l}(X, \theta) = (l_1, \ldots, l_{N_V})$$, where $$l_i$$ is the logit for the $$i$$-th token in a vocabulary $$V$$ of size $$N_V$$.

The probability of generating a specific token $$y_i$$ given context $$X$$ is given by the softmax function applied to the logits:

$$
P(y_i \mid X, \theta) = \frac{\exp(l_i)}{\sum_{j=1}^{N_V} \exp(l_j)}
$$

When generating text with a temperature parameter $$T > 0$$, this probability is adjusted:

$$
P_T(y_i \mid X, \theta) = \frac{\exp(l_i/T)}{\sum_{j=1}^{N_V} \exp(l_j/T)}
$$

A lower temperature ($$T \to 0$$) makes the distribution sharper (more deterministic), while a higher temperature ($$T \to \infty$$) makes it flatter (more random).

For a given context $$X$$, let $$y^*(X)$$ denote the "correct" or desired next token (e.g., in factual recall tasks where the answer is unambiguous). The hallucination probability is then the probability of *not* generating this correct token:

$$
P_{\text{h}}(X, \theta, T) = 1 - P_T(y^*(X) \mid X, \theta)
$$

Our goal is to understand how $$\bar{P}_{\text{h}}$$, the average hallucination probability over all contexts, behaves.

## Modeling Logits from Corpus Evidence

Our first step is to model how an LLM might arrive at its logits based on its training data. Intuitively, the more 'evidence' a model has seen in its training data for a particular token to follow a given context, the higher its logit for that token should be.

**1. Evidence Density ($$E(y_k \mid X)$$):**
For a context $$X$$ and a potential next token $$y_k \in V$$, its evidence density from the training corpus $$\mathcal{D}$$ is defined as the sum of semantic similarities between $$X$$ and all contexts $$X_j$$ in $$\mathcal{D}$$ where $$y_k$$ was the target token:

$$
E(y_k \mid X) = \sum_{(X_j, y_{\text{target}}) \in \mathcal{D} \text{ s.t. } y_{\text{target}} = y_k} K(X, X_j)
$$

Here, $$K(X, X_j)$$ is a semantic similarity kernel, with $$0 \le K(X,X_j) \le 1$$. It measures how relevant a training example $$(X_j, y_k)$$ is to the current context $$X$$.

It's worth noting some basic properties of these evidence densities:
- $$0 \le E(y_k \mid X) \le N_{y_k}(\mathcal{D})$$, where $$N_{y_k}(\mathcal{D})$$ is the total count of token $$y_k$$ in the training corpus $$\mathcal{D}$$. This is because the kernel $$K(X,X_j)$$ is between 0 and 1.
- The total evidence density for a given context $$X$$ across all possible next tokens is $$\mathcal{E}_X = \sum_{k=1}^{N_V} E(y_k \mid X)$$. This sum can be rewritten as $$\mathcal{E}_X = \sum_{(X_j, y_{\text{target}}) \in \mathcal{D}} K(X, X_j)$$, so $$0 \le \mathcal{E}_X \le \vert\mathcal{D}\vert$$, the total size of the corpus.

**2. Simplified Logit Model:**
We adopt a simplified model where the logit $$l_i(X, \theta)$$ for token $$y_i$$ is approximated by a linear function of these evidence densities:

$$
l_i(X, \theta) \approx \beta_S E(y_i \mid X) - \beta_C \mathcal{E}_X + c
$$

This simplified model has a few key assumptions:
- Each token $$y_i$$ gains logit strength proportional to its own evidence $$E(y_i \mid X)$$ (scaled by $$\beta_S$$).
- It loses logit strength proportional to the *total* evidence $$\mathcal{E}_X$$ for *all* tokens in that context (scaled by $$\beta_C$$). This introduces a competitive dynamic: the more evidence there is for other tokens, the harder it is for $$y_i$$ to stand out.
- The parameters $$\beta_S > 0$$ (self-evidence boost) and $$\beta_C \ge 0$$ (cross-evidence competition factor) and the bias $$c$$ are constants for the model.
Essentially, a token's logit is determined by its own specific evidence, penalized by a measure of the 'background' evidence for all tokens in that context. This form is inspired by more general Neural Tangent Kernel (NTK) approaches but is simplified for tractability.

## Condition for Bounded Hallucination

Now, let's consider what it takes for the hallucination probability $$P_{\text{hall}}(X, \theta, T)$$ for a *specific* context $$X$$ to be below a chosen threshold $$\epsilon_0 \in (0,1)$$. If $$P_{\text{hall}}(X, \theta, T) \le \epsilon_0$$, then the probability of generating the correct token $$y^*(X)$$ must be $$P_T(y^*(X) \mid X, \theta) \ge 1 - \epsilon_0$$.

This implies that the logit of the correct token $$l_{y^*}$$ must be sufficiently larger than the logit of any other token $$l_{y_{\text{alt}}}$$. For the highest-logit alternative token $$y_{\text{alt}}$$, and assuming a large vocabulary $$N_V$$, this condition can be shown to require:

$$
l_{y^*}(X, \theta) - l_{y_{\text{alt}}}(X, \theta) \ge T \log\left(\frac{1 - \epsilon_0}{\epsilon_0}\right)
$$

Substituting our simplified logit model into this inequality, we get a condition on the difference in evidence densities:

$$
E(y^*(X) \mid X) - E(y_{\text{alt}} \mid X) \ge \frac{T \log\left(\frac{1 - \epsilon_0}{\epsilon_0}\right)}{\beta_S}
$$

Let's define the right-hand side as $$M'(\epsilon_0, T) = \frac{T \log\left(\frac{1 - \epsilon_0}{\epsilon_0}\right)}{\beta_S}$$. This term $$M'$$ represents the minimum evidence advantage the correct token must have over its strongest competitor.
If $$E(y^*(X) \mid X) - E(y_{\text{alt}} \mid X) < M'(\epsilon_0, T)$$, then $$P_{\text{h}}(X, \theta, T) > \epsilon_0$$.

In simpler terms, for the model to avoid hallucinating (with probability at least $$1-\epsilon_0$$) in a given instance, the 'evidence signal' for the correct token $$y^*$$ must exceed the evidence for the strongest competitor $$y_{\text{alt}}$$ by this margin $$M'$$. This margin depends on our desired confidence (related to $$\epsilon_0$$) and the generation temperature $$T$$. A higher temperature or a stricter (smaller) $$\epsilon_0$$ demands a larger evidence gap.

## Aggregate Hallucination and Data Sparsity

Deriving the exact average hallucination probability $$\bar{P}_{\text{h}}$$ is complex. Instead, we seek a lower bound. To make progress, we consider a simplified (and stronger) condition for when $$P_{\text{h}}(X, \theta, T) > \epsilon_0$$: we assume this happens if the *total* evidence density for the context, $$\mathcal{E}_X$$, is less than the required evidence margin $$M'(\epsilon_0, T)$$.

The intuition is that if the overall evidence available for *any* token in a given context $$X$$ is very low (i.e., the context is highly unfamiliar or ambiguous based on the training data, such that $$\mathcal{E}_X < M'$$), it becomes difficult for the correct token $$y^*$$ to gather enough evidence to significantly surpass alternatives by the necessary margin. This simplification allows us to use aggregate statistics of $$\mathcal{E}_X$$ to derive a bound.

**1. Average Total Evidence:**
The average total evidence density, $$\bar{\mathcal{E}}_X$$, taken over all possible contexts $$X$$ (assuming a uniform distribution over a context space of size $$N_V^{L_X-1}$$), is:

$$
\bar{\mathcal{E}}_X = \mathbb{E}_{X}[\mathcal{E}_X] = \frac{\vert\mathcal{D}\vert \bar{S}}{N_V^{L_X-1}}
$$

Where:
- $$\vert\mathcal{D}\vert$$ is the size of the training corpus.
- $$\bar{S}$$ is the average kernel influence (effectively, how many contexts $$X$$ are 'covered' by an average training example $$X_j$$).
- $$N_V^{L_X-1}$$ approximates the size of the unique context space.
This shows that average evidence density increases with corpus size and decreases with the vastness of the language space.

**2. Lower Bound on Fraction of Low-Evidence Contexts:**
Let $$f(\epsilon_0)$$ be the fraction of contexts for which $$\mathcal{E}_X < M'(\epsilon_0, T)$$. For these contexts, by our simplifying assumption, $$P_{\text{h}}(X, \theta, T) > \epsilon_0$$.
Using Markov's inequality, which states $$P(Z \ge a) \le \mathbb{E}[Z]/a$$ for a non-negative random variable $$Z$$, we have $$P(\mathcal{E}_X \ge M'(\epsilon_0, T)) \le \frac{\bar{\mathcal{E}}_X}{M'(\epsilon_0, T)}$$.
Therefore, the fraction of low-evidence contexts is bounded:

$$
f(\epsilon_0) = P(\mathcal{E}_X < M'(\epsilon_0, T)) \ge 1 - \frac{\bar{\mathcal{E}}_X}{M'(\epsilon_0, T)}
$$

This bound is informative (i.e., $$f(\epsilon_0) > 0$$) if $$\bar{\mathcal{E}}_X < M'(\epsilon_0, T)$$.

Let $$L(\epsilon_0) = \log\left(\frac{1 - \epsilon_0}{\epsilon_0}\right)$$. This term increases as $$\epsilon_0$$ gets closer to 0 or 1, reflecting a demand for a larger logit gap.
Let $$C = \bar{S} \beta_S$$ be a constant that groups model structure and kernel properties.
The ratio then becomes $$\frac{\bar{\mathcal{E}}_X}{M'(\epsilon_0, T)} = \frac{C \vert\mathcal{D}\vert}{N_V^{L_X-1} T L(\epsilon_0)}$$.
So, the lower bound on $$f(\epsilon_0)$$ is:

$$
f(\epsilon_0) \ge 1 - \frac{C \vert\mathcal{D}\vert}{N_V^{L_X-1} T L(\epsilon_0)}
$$

**3. Lower Bound on Average Hallucination:**
The average hallucination probability $$\bar{P}_{\text{h}} = \mathbb{E}_{X} [P_{\text{h}}(X, \theta, T)]$$ can be lower-bounded because, for the fraction $$f(\epsilon_0)$$ of contexts, $$P_{\text{h}} > \epsilon_0$$. Thus:
$$\bar{P}_{\text{h}} > \epsilon_0 \cdot f(\epsilon_0)$$
Substituting the bound for $$f(\epsilon_0)$$:

$$
\bar{P}_{\text{h}} > \epsilon_0 \left(1 - \frac{C \vert\mathcal{D}\vert}{N_V^{L_X-1} T L(\epsilon_0)}\right)
$$

This inequality gives us a lower bound on the average hallucination probability. Let's break down the term $$k_0(\epsilon_0) = \frac{C \vert\mathcal{D}\vert}{N_V^{L_X-1} T L(\epsilon_0)}$$:
- $$C \vert\mathcal{D}\vert$$ represents the 'total effective evidence' in the corpus, scaled by model and kernel parameters.
- $$N_V^{L_X-1}$$ is the vastness of the potential context space.
- $$T L(\epsilon_0)$$ represents the 'difficulty' of distinguishing the correct token, influenced by temperature and our chosen hallucination threshold $$\epsilon_0$$.
If this ratio $$k_0(\epsilon_0)$$ (representing data richness relative to task difficulty for a given $$\epsilon_0$$) is small (i.e., $$k_0(\epsilon_0) \ll 1$$), the term in the parenthesis approaches 1, and the bound approaches $$\epsilon_0$$. If $$k_0(\epsilon_0)$$ is large, the bound can become smaller. This expression already hints at the scaling relationships we're looking for, but it still depends on our arbitrary choice of $$\epsilon_0$$.

## Optimizing the Lower Bound

The derived lower bound $$g(\epsilon_0) = \epsilon_0 (1 - k_0(\epsilon_0))$$ depends on our choice of $$\epsilon_0$$. We can find the tightest possible bound by choosing $$\epsilon_0$$ to maximize $$g(\epsilon_0)$$.
Let $$k = \frac{C \vert\mathcal{D}\vert}{N_V^{L_X-1} T}$$ be the key scaling parameter, which encapsulates data density ($$\vert\mathcal{D}\vert/N_V^{L_X-1}$$), model characteristics ($$C$$), and inverse temperature ($$1/T$$). The bound is $$g(\epsilon_0) = \epsilon_0 (1 - \frac{k}{L(\epsilon_0)})$$.

The optimal $$L^* = L(\epsilon_0^*)$$ that maximizes $$g(\epsilon_0)$$ (where $$\epsilon_0^*$$ is the optimal threshold) is found by solving the following transcendental equation:

$$
(L^*)^2 = k \left(L^* + 1 + e^{-L^*}\right)
$$

While this equation doesn't have a simple closed-form solution for $$L^*$$, its properties can be analyzed, especially in asymptotic regimes (when $$k$$ is very small or very large). This $$L^*$$ represents an optimal 'logit difference scale' that balances making $$\epsilon_0^*$$ small (desirable for the $$\epsilon_0^*$$ factor in the bound) versus making the term $$(1 - k/L^*)$$ large.

The optimized value $$\epsilon_0^*$$ is then $$\frac{1}{1+e^{L^*}}$$.
Substituting $$\epsilon_0^*$$ and $$L^*$$ into $$g(\epsilon_0)$$ gives the optimized lower bound on average hallucination:

$$
\bar{P}_{\text{h}} > \frac{1}{1+e^{L^*}} \left(1 - \frac{k}{L^*}\right)
$$

This is our main result: a lower bound on average hallucination probability that depends on the consolidated parameter $$k$$.

![Hallucination Lower Bound](/assets/img/hallucination_bound.png)
*Figure 1: The optimized lower bound on average hallucination probability, $$\bar{P}_{\text{h}}$$, as a function of the data density, model, and temperature parameter $$k = \frac{C \vert\mathcal{D}\vert}{N_V^{L_X-1} T}$$. The bound transitions from $$\approx 0.5$$ for small $$k$$ (sparse data/high temperature) towards $$0$$ for large $$k$$ (dense data/low temperature).*

## Asymptotic Behavior of the Optimized Bound

The behavior of this optimized bound in limiting cases of the parameter $$k$$ is particularly insightful:

1.  **$$k \to 0$$ (Extreme Data Sparsity or High Temperature)**
    In this regime, data is very scarce relative to language complexity, or generation is very random. It can be shown that $$L^* \approx \sqrt{2k}$$, which approaches 0.
    As $$L^* \to 0$$, $$\epsilon_0^* = \frac{1}{1+e^{L^*}} \to \frac{1}{2}$$.
    Further analysis shows the entire bound $$\bar{P}_{\text{h}} \to \frac{1}{2}$$.
    This suggests that with very sparse data or at very high temperatures, the model's output is nearly random concerning correctness, leading to an average hallucination probability of at least 50%. This makes intuitive sense: if the model has very little reliable information from its training ($$k \to 0$$), its performance should degrade towards random guessing. For a binary decision of 'correct' vs 'any incorrect alternative', random guessing would yield a 50% error rate.

2.  **$$k \to \infty$$ (Data-Rich Regime or Low Temperature)**
    In this regime, data is abundant, or generation is nearly deterministic. It can be shown that $$L^* \approx k$$.
    Then $$\epsilon_0^* = \frac{1}{1+e^{L^*}} \approx e^{-L^*}$$ $$\approx e^{-k}$$.
    The term $$(1 - k/L^*)$$ approaches 0. More precisely, the bound $$\bar{P}_{\text{h}}$$ decays towards 0, approximately as $$\frac{e^{-k}}{k}$$.
    This indicates that with sufficiently dense data and low temperature, the derived lower bound on hallucination can be made arbitrarily small, decaying exponentially with $$k$$. This is also intuitive: if we have abundant data relative to complexity and low randomness ($$k \to \infty$$), the model should be able to learn the correct patterns effectively, and the floor for hallucination should drop significantly.

## Conclusion

This derivation provides a theoretical lower bound on the average hallucination probability in LLMs, starting from a model of how corpus evidence translates to logits. It formally demonstrates how this hallucination floor scales with fundamental parameters, encapsulated in the parameter $$k = \frac{C \vert\mathcal{D}\vert}{N_V^{L_X-1} T}$$:
-   **Corpus Size ($$\vert\mathcal{D}\vert$$):** Larger datasets tend to decrease the bound (increase $$k$$).
-   **Context Space Complexity ($$N_V^{L_X-1}$$):** Larger vocabularies or longer contexts increase complexity and tend to increase the bound (decrease $$k$$).
-   **Generation Temperature ($$T$$):** Higher temperatures tend to increase the bound (decrease $$k$$).
-   **Model/Kernel Properties ($$C$$):** Factors like effective semantic similarity spread also play a role.


It is important to remember that this model, while illustrative, is a significant simplification of the complex mechanisms underlying LLM behavior, and does not encompass more intricate forms of hallucination. Note that the exact form of the scaling we derived here is predicated on the softmax function. A natural follow-up question is whether there is an alternative to softmax that would result in qualitatively better scaling behavior.