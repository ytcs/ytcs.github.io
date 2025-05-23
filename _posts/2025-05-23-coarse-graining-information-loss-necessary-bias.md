---
layout: post
title: "Coarse Graining, Information Loss, and Necessary Bias"
date: 2025-05-23
categories: information-theory
---

The process of simplifying complex information, known as **coarse-graining**, is fundamental to both cognitive systems and the modeling of complex physical or computational systems. This involves mapping a large set of detailed **microstates** \(X\) to a smaller, more manageable set of **macrostates** \(K\). While essential for tractability, this simplification inevitably leads to information loss. This post explores the information-theoretic underpinnings of this loss and its direct consequences on the minimum achievable error, or **inherent bias**, when making judgments based on these simplified macrostate representations.

We will focus on the behavior of coarse-graining under general conditions, particularly when the mapping from microstates to macrostates lacks specific structure, akin to a random map. The discussion will adhere to the uniform microstate distribution assumption for key results concerning information loss and bias. The primary example used for illustration will be the coarse-graining of sequences of coin flips (microstates) to the count of heads (a macrostate).

## 1. Information-Theoretic Framework

We begin by establishing the core notation and assumptions from information theory.

**Assumption 1.1 (Finite State Spaces):**
The microstate space \(\Omega\) (with alphabet \(\mathcal{X}\)) and the macrostate space \(\Gamma\) (with alphabet \(\mathcal{K}\)) are finite sets. Let \(N = \lvert\mathcal{X}\rvert\) and \(M = \lvert\mathcal{K}\rvert\).

**Definition 1.1 (Microstate Description and Entropy):**
Let \(X\) be a random variable representing the microstate, taking values \(x \in \mathcal{X}\) with distribution \(p(x) = P(X=x)\). The **microstate entropy** is:
\[ H(X) := - \sum_{x \in \mathcal{X}} p(x) \log_2 p(x) \]
In the common case where microstates are assumed to be uniformly distributed (e.g., for lack of information suggesting otherwise, or to explore typical behavior), \(p(x) = 1/N\) for all \(x \in \mathcal{X}\). In this scenario, the microstate entropy simplifies to:
\[ H(X) = - \sum_{x \in \mathcal{X}} \frac{1}{N} \log_2 \left(\frac{1}{N}\right) = N \left( -\frac{1}{N} \log_2 \frac{1}{N} \right) = \log_2 N \]

*Example (Coin Flips): For \(n=100\) fair coin flips, there are \(N=2^{100}\) possible sequences (microstates). If each sequence is equally likely, \(H(X) = \log_2(2^{100}) = 100\) bits.*

**Definition 1.2 (Coarse-Graining Map):**
A **coarse-graining map** is a function \(f: \mathcal{X} \to \mathcal{K}\). We focus on deterministic maps.

**Assumption 1.2 (Surjective Map):**
For a deterministic map \(f\), we assume it is surjective, meaning each \(k \in \mathcal{K}\) is the image of at least one \(x \in \mathcal{X}\). This implies \(N \ge M\).

**Definition 1.3 (Macrostate Description - Deterministic Map):**
Given a microstate distribution \(p(x)\) and a deterministic map \(f\):
1.  **Partition Cell:** \(\Omega_k := f^{-1}(k) = \{x \in \mathcal{X} \mid f(x)=k\}\). The set of all partition cells \(\{\Omega_k\}_{k \in \mathcal{K}}\) forms a partition of \(\mathcal{X}\).
2.  **Multiplicity:** \(W_k := \lvert\Omega_k\rvert\). By surjectivity, \(W_k \ge 1\). Also, \(\sum_{k=1}^M W_k = N\).
3.  **Macrostate Variable:** \(K := f(X)\), taking values \(k \in \mathcal{K}\).
4.  **Macrostate Distribution:** \(p(k) := P(K=k) = \sum_{x \in \Omega_k} p(x)\).
5.  **Macrostate Entropy:** \(H(K) := - \sum_{k \in \mathcal{K}} p(k) \log_2 p(k)\).

*Example (Coin Flips): If \(X\) is a sequence of 100 flips and \(K\) is the number of heads, \(M=101\) (0 to 100 heads). \(W_k = \binom{100}{k}\) is the number of sequences with \(k\) heads. If \(p(x)=1/2^{100}\), then \(p(k) = \binom{100}{k}/2^{100}\).*

**Definition 1.4 (Conditional Entropy and Information Loss - Deterministic Map):**
1.  **Conditional Distribution:** For \(p(k) > 0\), the conditional probability of microstate \(x\) given macrostate \(k\) is \(p(x \mid k) := p(X=x \mid K=k)\). If \(x \notin \Omega_k\), then \(p(x \mid k)=0\). If \(x \in \Omega_k\), then \(p(x \mid k) = p(x)/p(k)\).
    *If \(p(x)=1/N\) (uniform microstates), then \(p(k) = W_k/N\). For \(x \in \Omega_k\), \(p(x \mid k) = (1/N) / (W_k/N) = 1/W_k\). This is the uniform distribution over the microstates within the partition cell \(\Omega_k\).*
2.  **Conditional Entropy (Specific):** \(H(X \mid K=k) := - \sum_{x \in \Omega_k} p(x \mid k) \log_2 p(x \mid k)\).
    *If \(p(x)=1/N\), then \(H(X \mid K=k) = - \sum_{x \in \Omega_k} (1/W_k) \log_2 (1/W_k) = \log_2 W_k\).*
3.  **Average Conditional Entropy (Information Loss):** \(H(X \mid K) := \sum_{k \in \mathcal{K}} p(k) H(X \mid K=k) = \mathbb{E}_{K} [H(X \mid K=k)]\). This quantity represents the average information lost about the microstate \(X\) when only the macrostate \(K\) is known.
    *If \(p(x)=1/N\), then \(H(X \mid K) = \sum_{k \in \mathcal{K}} (W_k/N) \log_2 W_k = \mathbb{E}_{K}[\log_2 W_K]\).*

**Proposition 1.1 (Chain Rule):**
The total microstate entropy can be decomposed as:
\[ H(X) = H(K) + H(X \mid K) \]
This means: Total Information = Retained Information (in macrostate) + Lost Information.

## 2. Information Loss for Random Maps (Uniform Microstates)

To understand the typical consequences of coarse-graining, especially when \(N \gg M\) and the map \(f\) is not specifically engineered to preserve certain information, we consider maps chosen randomly. The following proposition (Prop. 4.4 from the reference paper) details the behavior under a uniform microstate distribution \(p(x)=1/N\).

**Proposition 2.1 (Information Loss for Random Maps - Uniform Microstates):**
Let \(p(x)\) be the uniform distribution over \(\mathcal{X}\) (\(\lvert\mathcal{X}\rvert=N\)), so \(H(X) = \log_2 N\). Let \(\mathcal{F}_{N,M}\) be the set of all surjective deterministic maps \(f: \mathcal{X} \to \mathcal{K}\) (\(\lvert\mathcal{K}\rvert=M\), \(M \le N\)). If a map \(f\) is chosen uniformly at random from \(\mathcal{F}_{N,M}\), let \(K=f(X)\) and \(p_f(k)\) be the resulting macrostate distribution. Let \(W_k = \lvert f^{-1}(k) \rvert\). Then:
1.  The expected macrostate distribution is uniform: \(\mathbb{E}_{f}[p_f(k)] = 1/M\) for all \(k \in \mathcal{K}\).
2.  The expected retained information is bounded: \(\mathbb{E}_{f}[H(K)] \le \log_2 M\).
3.  The expected information loss is bounded below: \(\mathbb{E}_{f}[H(X \mid K)] \ge \log_2(N/M)\).
4.  In the asymptotic limit \(N \to \infty\) with \(M\) fixed, the retained information concentrates: \(H(K) \xrightarrow{P} \log_2 M\).
5.  In the asymptotic limit \(N \to \infty\) with \(M\) fixed, the information loss concentrates: \(H(X \mid K) \xrightarrow{P} \log_2(N/M)\).
(Convergence \(\xrightarrow{P}\) denotes convergence in probability).

**Proof of Proposition 2.1:**
Let \(\mathbb{E}_{f}[\cdot]\) denote the expectation over the uniform distribution on \(\mathcal{F}_{N,M}\).

**(a) Expected Macrostate Distribution:**
We compute \(\mathbb{E}_{f}[p_f(k)]\) for a fixed \(k \in \mathcal{K}\).
Since \(p(x)=1/N\), the macrostate probability is \(p_f(k) = \sum_{x \in \Omega_k} p(x) = W_k / N\).
Thus, the expected macrostate probability is:
\[ \mathbb{E}_{f}[p_f(k)] = \frac{1}{N} \mathbb{E}_{f}[W_k] \]

By symmetry, averaging over all surjective maps \(f\) (which are invariant to relabeling of the output \(k\)), the expected size \(\mathbb{E}_{f}[W_k]\) must be the same for all \(k\). Let this average size be \(W_{avg}\).

Since \(\sum_{j=1}^M W_j = N\) for any specific map \(f\), linearity of expectation gives:
\(\sum_{j=1}^M \mathbb{E}_{f}[W_j] = \mathbb{E}_{f}\left[\sum_{j=1}^M W_j\right] = \mathbb{E}_{f}[N] = N\).

This implies \(M \cdot W_{avg} = N\), so \(W_{avg} = N/M\).
Substituting this back, we get \(\mathbb{E}_{f}[p_f(k)] = \frac{1}{N} (N/M) = 1/M\). This proves part (1).

**(b) Expected Retained Information:**
The entropy function \(H(P)\) is concave in the distribution \(P\). By Jensen's inequality for concave functions:
\[ \mathbb{E}_{f}[H(K)] = \mathbb{E}_{f}[H(p_f)] \le H(\mathbb{E}_{f}[p_f]) \]

Using the result from part (a), \(\mathbb{E}_{f}[p_f]\) is the uniform distribution over \(M\) states, i.e., \((1/M, \dots, 1/M)\).
The entropy of this uniform distribution is \(H(1/M, \dots, 1/M) = -\sum_{j=1}^M (1/M) \log_2(1/M) = \log_2 M\).

Therefore, \(\mathbb{E}_{f}[H(K)] \le \log_2 M\). This proves part (2).

**(c) Expected Information Loss:**
From the chain rule for entropy, we have \(H(X \mid K) = H(X) - H(K)\).
Taking the expectation over \(f\):
\[ \mathbb{E}_{f}[H(X \mid K)] = \mathbb{E}_{f}[H(X) - H(K)] = H(X) - \mathbb{E}_{f}[H(K)] \]
(Note that \(H(X)\) is independent of \(f\) as \(p(x)\) is fixed).

Since \(H(X) = \log_2 N\) (because \(p(x)\) is uniform) and we know from part (b) that \(\mathbb{E}_{f}[H(K)] \le \log_2 M\):
\[ \mathbb{E}_{f}[H(X \mid K)] \ge \log_2 N - \log_2 M = \log_2(N/M) \]
This proves part (3).

**(d) Concentration of Retained Information:**
This result relies on analyzing the distribution of partition sizes \(\{W_k\}\) generated by random surjective maps when \(N \gg M\).
It is a known result from probabilistic combinatorics (related to allocation problems, often conceptualized as randomly placing \(N\) balls into \(M\) bins such that each bin has at least one ball) that as \(N \to \infty\) with \(M\) fixed, the proportion of microstates mapping to each macrostate concentrates around the mean. That is, \(W_k/N \xrightarrow{P} 1/M\) for each \(k \in \mathcal{K}\).

Since \(H(K) = -\sum_{j=1}^M (W_j/N) \log_2(W_j/N)\) is a continuous function of the empirical distribution \((W_1/N, \dots, W_M/N)\), the continuous mapping theorem implies that \(H(K)\) converges in probability to the entropy of the limiting uniform distribution \((1/M, \dots, 1/M)\):
\[ H(K) \xrightarrow[N\to\infty]{P} -\sum_{j=1}^M \frac{1}{M} \log_2 \frac{1}{M} = \log_2 M \]
This proves part (4).

**(e) Concentration of Information Loss:**
Using the chain rule again: \(H(X \mid K) = H(X) - H(K) = \log_2 N - H(K)\).
We want to show that \(H(X \mid K) - \log_2(N/M) \xrightarrow{P} 0\) as \(N \to \infty\) with \(M\) fixed.

Consider the difference:
\[ H(X \mid K) - \log_2(N/M) = (\log_2 N - H(K)) - (\log_2 N - \log_2 M) = \log_2 M - H(K) \]

From part (d), we established that \(H(K) \xrightarrow{P} \log_2 M\) as \(N \to \infty\) with \(M\) fixed.
Therefore, the difference \(\log_2 M - H(K)\) converges in probability to 0.
This proves the convergence: \(H(X \mid K) \xrightarrow{P} \log_2(N/M)\). This proves part (5).

**Implications of Proposition 2.1:**
For large microstate spaces (\(N\)) and a fixed, much smaller number of macrostates (\(M\)), a typical (randomly chosen) coarse-graining map inevitably discards an amount of information \(H(X \mid K)\) that approaches \(\log_2(N/M)\). This lost information grows logarithmically with \(N\). Since the total information is \(H(X) = \log_2 N\), the fraction of information lost, \(H(X \mid K)/H(X)\), approaches \((\log_2 N - \log_2 M) / (\log_2 N) = 1 - (\log_2 M) / (\log_2 N)\). As \(N \to \infty\) for fixed \(M\), this fraction approaches 1.
*Example (Coin Flips): With \(N=2^{100}\) microstates (sequences) and \(M=101\) macrostates (number of heads), the retained information \(H(K) \approx \log_2 101 \approx 6.66\) bits. The lost information \(H(X \mid K) \approx \log_2(2^{100}/101) \approx 100 - 6.66 = 93.34\) bits. Over \(93\%\) of the original information is lost.*

## 3. Inherent Bias from Information Loss

Now we connect this information loss to the limits of accuracy when making judgments based on simplified macrostates.

Let \(q_{\text{correct}}(x)\) represent the true value or normatively correct judgment associated with microstate \(x\). A cognitive system or model using a heuristic \(\mathcal{J}\) based on the observed macrostate \(K=f(X)\) produces a judgment \(q_{subj}(K) = \mathcal{J}(K)\).

**Definition 3.1 (Judgment Error and Bias):**
The error for a given microstate \(x\) is \(e(x) = q_{\text{correct}}(x) - q_{subj}(f(x))\). The overall magnitude of error is often measured using the Root Mean Square Error (RMSE):
\[ \operatorname{RMSE}(f, \mathcal{J}) := \sqrt{ \mathbb{E}_{X} [ (q_{\text{correct}}(X) - q_{subj}(f(X)))^2 ] } \]
where the expectation \(\mathbb{E}_{X}\) is over the microstate distribution \(p(x)\).

**Definition 3.2 (Optimal Heuristic and Inherent Bias):**
For a given map \(f\), the heuristic \(\mathcal{J}^*\) that minimizes the RMSE is the conditional expectation:
\[ q^*_{subj}(k) := \mathcal{J}^*(k) = \mathbb{E}[q_{\text{correct}}(X) \mid K=k] = \sum_{x \in \Omega_k} p(x \mid k) q_{\text{correct}}(x) \]
(Assuming \(p(x)\) is uniform, \(p(x \mid k) = 1/W_k\) for \(x \in \Omega_k\), so \(q^*_{subj}(k) = \frac{1}{W_k} \sum_{x \in \Omega_k} q_{\text{correct}}(x)\).)
The minimum achievable RMSE for map \(f\), obtained using \(\mathcal{J}^*\), represents the **inherent bias** (or inherent error) imposed by the coarse-graining itself:
\[ B^*(f) := \operatorname{RMSE}(f, \mathcal{J}^*) = \sqrt{ \mathbb{E}_{X} [ (q_{\text{correct}}(X) - \mathbb{E}[q_{\text{correct}}(X') \mid K'=f(X)])^2 ] } \]

**Lemma 3.1 (Inherent Bias and Conditional Variance):**
The square of the inherent bias for a map \(f\) is the expected conditional variance of the normative judgment given the macrostate:
\[ (B^*(f))^2 = \mathbb{E}_{K} [ \operatorname{Var}(q_{\text{correct}}(X) \mid K=k) ] \]

**Proof of Lemma 3.1:**
By definition of conditional variance:
\[
\begin{aligned}
\mathbb{E}_{K} [ \operatorname{Var}(q_{\text{correct}}(X) \mid K=k) ] &= \sum_{j \in \mathcal{K}} p(j) \operatorname{Var}(q_{\text{correct}}(X) \mid K=j) \\
&= \sum_{j \in \mathcal{K}} p(j) \mathbb{E} [ (q_{\text{correct}}(X) - \mathbb{E}[q_{\text{correct}}(X') \mid K'=j])^2 \mid K=j ] \\
&= \sum_{j \in \mathcal{K}} p(j) \sum_{x \in \Omega_j} p(x \mid j) (q_{\text{correct}}(X) - \mathbb{E}[q_{\text{correct}}(X') \mid K'=j])^2 \\
&= \sum_{x \in \mathcal{X}} p(x) (q_{\text{correct}}(X) - \mathbb{E}[q_{\text{correct}}(X') \mid K'=f(x)])^2 \quad (\text{since } p(x) = p(f(x))p(x \mid f(x)) ) \\
&= \mathbb{E}_{X} [ (q_{\text{correct}}(X) - \mathbb{E}[q_{\text{correct}}(X') \mid K'=f(X)])^2 ] \\
&= (B^*(f))^2
\end{aligned}
\]

The first line is the definition of expected conditional variance.
The second line expands the conditional variance.
The third line expands the conditional expectation definition.
The fourth line rewrites the sum over macrostates \(j\) and microstates \(x \in \Omega_j\) as a single sum over all microstates \(x \in \mathcal{X}\), using the fact that \(p(j)p(x \mid j) = p(x,j) = p(x)\) if \(x \in \Omega_j\) (i.e., \(f(x)=j\)), and \(p(x \mid j)=0\) otherwise.
The fifth line recognizes this sum as the definition of \(\mathbb{E}_{X} [ (q_{\text{correct}}(X) - \mathbb{E}[q_{\text{correct}}(X') \mid K'=f(X)])^2 ]\).
The final line is from the definition of \(B^*(f)\).

**Proposition 3.1 (Bias as Residual Variance):**
For any coarse-graining map \(f: \mathcal{X} \to \mathcal{K}\) and any function \(q_{\text{correct}}: \mathcal{X} \to \mathbb{R}\), the squared inherent bias \((B^*(f))^2\) is equal to the total variance of \(q_{\text{correct}}(X)\) minus the variance of the conditional expectation of \(q_{\text{correct}}(X)\) given the macrostate \(K=f(X)\).
\[ (B^*(f))^2 = \operatorname{Var}(q_{\text{correct}}(X)) - \operatorname{Var}_K ( \mathbb{E}[q_{\text{correct}}(X) \mid K=k] ) \]
where \(\operatorname{Var}_K(\cdot)\) denotes the variance taken over the distribution \(p(k)\) of the macrostates.

**Proof of Proposition 3.1:**
This follows directly from the Law of Total Variance, which states:
\(\operatorname{Var}(Y) = \mathbb{E}[\operatorname{Var}(Y \mid Z)] + \operatorname{Var}(\mathbb{E}[Y \mid Z])\).

Let \(Y = q_{\text{correct}}(X)\) and \(Z = K = f(X)\).
Let \(\sigma^2_{total} = \operatorname{Var}(q_{\text{correct}}(X))\) be the total variance of the normative judgment.

Substituting into the Law of Total Variance, we get:
\(\sigma^2_{total} = \mathbb{E}_{K}[\operatorname{Var}(q_{\text{correct}}(X) \mid K=k)] + \operatorname{Var}_K(\mathbb{E}[q_{\text{correct}}(X) \mid K=k])\).

By Lemma 3.1, we know that \(\mathbb{E}_{K}[\operatorname{Var}(q_{\text{correct}}(X) \mid K=k)] = (B^*(f))^2\).

Substituting this into the equation gives:
\(\sigma^2_{total} = (B^*(f))^2 + \operatorname{Var}_K(\mathbb{E}[q_{\text{correct}}(X) \mid K=k])\).

Rearranging this equation to solve for \((B^*(f))^2\) yields:
\((B^*(f))^2 = \sigma^2_{total} - \operatorname{Var}_K(\mathbb{E}[q_{\text{correct}}(X) \mid K=k])\).
This completes the proof.

**Proposition 3.2 (Necessary Bias under Random Coarse-Graining):**
Let \(p(x)\) be the uniform distribution over \(\mathcal{X}\) (\(\lvert\mathcal{X}\rvert=N\)). Let \(q_{\text{correct}}(x)\) be a function on \(\mathcal{X}\) representing the true value, with finite total variance \(\operatorname{Var}(q_{\text{correct}}(X)) = \sigma^2 > 0\). Consider the set \(\mathcal{F}_{N,M}\) of all surjective maps \(f: \mathcal{X} \to \mathcal{K}\) (\(\lvert\mathcal{K}\rvert=M\), \(M \le N\)). If a map \(f\) is chosen uniformly at random from \(\mathcal{F}_{N,M}\), then in the asymptotic limit \(N \to \infty\) with \(M\) fixed, the inherent bias converges in probability to the total standard deviation of \(q_{\text{correct}}\):
\[ B^*(f) \xrightarrow{P} \sigma = \sqrt{\operatorname{Var}(q_{\text{correct}}(X))} \]

**Proof of Proposition 3.2:**
From Proposition 3.1, we have the relationship:
\(\sigma^2 = (B^*(f))^2 + \sigma^2_{between}(f)\),
where \(\sigma^2 = \operatorname{Var}(q_{\text{correct}}(X))\) is the total variance of the normative judgment, and \(\sigma^2_{between}(f) = \operatorname{Var}_K ( \mathbb{E}[q_{\text{correct}}(X) \mid K=k] )\) is the variance of the conditional expectations (the "between-category" variance).

Our goal is to show that for a map \(f\) chosen uniformly at random from \(\mathcal{F}_{N,M}\), the term \(\sigma^2_{between}(f) \xrightarrow{P} 0\) as \(N \to \infty\) with \(M\) fixed.

Let \(\mu_k = \mathbb{E}[q_{\text{correct}}(X) \mid K=k]\). Since \(p(x)\) is uniform, \(p(x \mid k) = 1/W_k\) for \(x \in \Omega_k\).
Thus, \(\mu_k\) is the sample mean of \(q_{\text{correct}}(x)\) over the microstates in partition cell \(\Omega_k\):
\[ \mu_k = \frac{1}{W_k} \sum_{x \in \Omega_k} q_{\text{correct}}(x) \]

From the analysis of random maps (as used in Proposition 2.1), in the limit \(N \to \infty\) with \(M\) fixed, the partition sizes \(W_k\) concentrate around \(N/M\). Thus, \(W_k \to \infty\) almost surely for each \(k\).
Furthermore, for a randomly chosen map \(f\), the set of microstates \(\Omega_k\) effectively behaves like a large random sample drawn from the total set of microstates \(\mathcal{X}\). (More formally, this can be analyzed as sampling without replacement from a large finite population, or sampling with replacement if \(N\) is considered effectively infinite relative to the typical size of \(W_k\)).

Let \(\mu_{global} = \mathbb{E}[q_{\text{correct}}(X)]\) be the global mean of the true value over all microstates.
By the Law of Large Numbers (or related concentration inequalities for sampling from finite populations), the sample mean \(\mu_k\) calculated over a large random subset \(\Omega_k\) (of size \(W_k \to \infty\)) converges in probability to the global mean \(\mu_{global}\).
That is, for any fixed \(k \in \mathcal{K}\), \(\mu_k \xrightarrow{P} \mu_{global}\) as \(W_k \to \infty\) (which occurs as \(N \to \infty\)).

Since this convergence \(\mu_k \xrightarrow{P} \mu_{global}\) holds for all \(k\) in the finite set \(\mathcal{K}\) (as \(N \to \infty\), all \(W_k\) tend to infinity if \(M\) is fixed), the distribution of these conditional means \(\mu_k\) (each weighted by \(p(k)\), which for random maps tends towards \(1/M\) for each \(k\)) collapses towards a point mass at \(\mu_{global}\).

The overall mean of these conditional means is given by the law of total expectation:
\(\mathbb{E}_{K}[\mu_K] = \sum_j p(j) \mu_j = \sum_j p(j) \mathbb{E}[q_{\text{correct}}(X) \mid K=j] = \mathbb{E}[q_{\text{correct}}(X)] = \mu_{global}\).

Therefore, the variance of these conditional means, \(\sigma^2_{between}(f)\), converges in probability to zero:
\[ \sigma^2_{between}(f) = \operatorname{Var}_K(\mu_K) = \sum_{j=1}^M p(j) (\mu_j - \mathbb{E}_{K}[\mu_K])^2 = \sum_{j=1}^M p(j) (\mu_j - \mu_{global})^2 \xrightarrow{P} 0 \]
This is because each \(p(j)\) tends to \(1/M\) (or at least remains bounded and sums to 1), and each \((\mu_j - \mu_{global})^2\) tends to 0 in probability.

Since \(\sigma^2 = (B^*(f))^2 + \sigma^2_{between}(f)\) and we have shown \(\sigma^2_{between}(f) \xrightarrow{P} 0\), it follows directly that:
\[ (B^*(f))^2 \xrightarrow{P} \sigma^2 \]
Taking the square root (which is a continuous function for non-negative values), we get the final result:
\[ B^*(f) \xrightarrow{P} \sigma \]
This completes the proof.

**Interpretation of Necessary Bias:**
Proposition 3.2 delivers a crucial insight into the consequences of aggressive simplification. The term "map \(f\) chosen uniformly at random" is a mathematical way to describe the behavior of a *typical* or *generic* coarse-graining map, especially one constructed without specific knowledge of, or alignment with, the underlying quantity \(q_{\text{correct}}(x)\). It implies that if you don't meticulously design your simplification to preserve the relevant variations in \(q_{\text{correct}}(x)\), your map is overwhelmingly likely to behave like one of these information-destroying "random" maps.

The inherent bias, \(B^*(f)\), represents the *minimum possible* Root Mean Square Error (RMSE) if one tries to estimate the true value \(q_{\text{correct}}(x)\) using only the knowledge of the macrostate \(K=f(x)\). This minimum error is achieved by the optimal heuristic \(\mathcal{J}^*(k) = \mathbb{E}[q_{\text{correct}}(X) \mid K=k]\).

The Law of Total Variance helps clarify this: the total variance \(\sigma^2 = \operatorname{Var}(q_{\text{correct}}(X))\) can be decomposed as:
\[ \sigma^2 = \mathbb{E}_{K}[\operatorname{Var}(q_{\text{correct}}(X) \mid K=k)] + \operatorname{Var}_K(\mathbb{E}[q_{\text{correct}}(X) \mid K=k]) \]
In our terms, this is:
\[ \sigma^2 = (B^*(f))^2 + \sigma^2_{between}(f) \]
Here, \((B^*(f))^2\) is the average variance *within* each macrostate category (the squared inherent bias). The term \(\sigma^2_{between}(f) = \operatorname{Var}_K(\mathbb{E}[q_{\text{correct}}(X) \mid K=k])\) is the variance *between* the average values of \(q_{\text{correct}}\) across different macrostates. This \(\sigma^2_{between}(f)\) quantifies how much the macrostates actually help in distinguishing the underlying \(q_{\text{correct}}\) values; it's the "useful" variance captured by the coarse-graining.

Proposition 3.2 states that for such a typical (random-like) map \(f\) with \(N \gg M\), this "useful" variance term \(\sigma^2_{between}(f)\) converges in probability to zero. The macrostates lose their power to differentiate the average true values effectively. Consequently, the inherent bias \(B^*(f)\) converges in probability to \(\sigma\), the total standard deviation of \(q_{\text{correct}}(X)\). The simplification process itself, if generic and aggressive, effectively destroys the distinctions that would allow for more accurate predictions. The number of ways to simplify "badly" (destroying information) is vastly greater than the number of ways to simplify "well" (preserving relevant information).

To illustrate the scale: consider mapping a relatively modest \(N=20\) distinct microstates into just \(M=2\) macrostates. The number of ways to partition these 20 microstates into 2 non-empty groups (which forms the basis for the \(M!=2\) surjective maps) is given by the Stirling number of the second kind, \(S_2(20,2) = 2^{20-1} - 1 = 2^{19} - 1 = 524,287\). Out of these over half a million possible fundamental groupings, only a tiny fraction (perhaps only one or a few, depending on the specific structure of an assumed \(q_{\text{correct}}(x)\)) would be "good" or "optimal" at separating high \(q_{\text{correct}}\) values from low ones. The vast majority of these mappings would mix values indiscriminately, behaving like a random map.
Now, imagine scaling this to \(N=2^{100}\) microstates, as in our coin flip example, and mapping to, say, \(M=101\) macrostates. The number of possible ways to partition \(2^{100}\) items into 101 groups is hyper-astronomical, a number so large it defies physical intuition (far exceeding the number of atoms in the observable universe). The fraction of maps that would fortuitously preserve the relevant information about a specific \(q_{\text{correct}}(x)\) within this unimaginably vast space of possibilities is effectively zero. Achieving a low-bias coarse-graining is therefore an act of careful, informed design, not a matter of chance.

Any specific cognitive heuristic \(\mathcal{J}\) operating on \(K\) can only perform as well as, or worse than, the optimal heuristic \(\mathcal{J}^*\). Thus, the achievable RMSE is bounded below: \(\operatorname{RMSE}(f, \mathcal{J}) \ge B^*(f) \approx \sigma\).
Significant bias is therefore a necessary consequence for typical simplification schemes when operating on complex phenomena (\(\sigma > 0\)). The structure of the map \(f\) is paramount; arbitrary or uninformed simplification is fundamentally detrimental to accuracy.

*Example (Student Performance): Consider a very large university with \(N\) students (microstates). Let \(q_{\text{correct}}(x)\) be the precise final exam score (0-100) of student \(x\). These scores have an overall average \(\mu_{global}\) and a total standard deviation \(\sigma > 0\).
Now, suppose the university administration, for a particular report, maps all students to a very small number of \(M\) very broad categories (macrostates), e.g., \(M=2\) categories: "Student ID is Odd" vs "Student ID is Even". This is a surjective map \(f\).
The optimal heuristic \(\mathcal{J}^*(k)\) would be the average exam score of all students in category \(k\). For \(k=\text{"Odd ID"}\), \(\mathbb{E}[q_{\text{correct}}(X) \mid K=\text{"Odd ID"}]\) would be the average score of all students with odd IDs. Similarly for "Even ID".
If \(N\) is large, and there's no systematic relation between student ID parity and exam scores, both \(\mathbb{E}[q_{\text{correct}}(X) \mid K=\text{"Odd ID"}]\) and \(\mathbb{E}[q_{\text{correct}}(X) \mid K=\text{"Even ID"}]\) will be very close to the global average score \(\mu_{global}\).
The term \(\sigma^2_{between}(f)\) – the variance between these two average scores – will be virtually zero. The categories "Odd ID" and "Even ID" tell us almost nothing meaningful about a student's likely exam performance.
According to Proposition 3.2, the inherent bias \(B^*(f)\) (the minimum RMSE when predicting a student's score based only on their ID parity) will converge to \(\sigma\), the total standard deviation of all exam scores. Knowing the ID parity provides no real predictive power, and the error of our best possible estimate based on this coarse-graining is essentially the original uncertainty about any student's score.
If, instead, the macrostates were "Score \(\ge 90\)", "Score \(50-89\)", "Score \(<50\)", this map \(f\) would be highly non-random with respect to \(q_{\text{correct}}\), and \(\sigma^2_{between}(f)\) would be large, allowing for a lower \(B^*(f)\). The proposition highlights the danger of *arbitrary* or *random-like* coarse-graining.*

## 4. Conclusion

Coarse-graining is an essential process for dealing with complexity, but it carries an inherent cost in terms of information loss. When the microstate space is significantly larger than the macrostate space (\(N \gg M\)) and the mapping is typical (random-like), this information loss \(H(X \mid K)\) is substantial, concentrating around \(\log_2(N/M)\) under uniform microstate assumptions.

This lost information directly translates into a fundamental limit on the accuracy of judgments based on the simplified macrostates. The inherent bias \(B^*(f)\), representing the minimum achievable RMSE, converges to the total standard deviation \(\sigma\) of the quantity being judged for such typical maps. This means that a certain level of bias is not merely a product of suboptimal heuristics but a necessary consequence of the information destroyed by the simplification process itself when the underlying reality has inherent variability and the coarse-graining map does not specifically preserve the relevant distinctions.
