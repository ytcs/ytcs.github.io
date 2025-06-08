---
published: true
layout: post
title: "Why Some Puzzles Are Hard for LLMs: A Framework for Cognitive Load"
categories: machine-learning
date: 2025-06-08
math: true
---

As Large Language Models (LLMs) get better at complex reasoning, we need better ways to measure the difficulty of the tasks we give them. A fascinating paper from researchers at Apple, ["The Illusion of Thinking"](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf), systematically tested models like Claude 3.7 Sonnet on various puzzles. They measured complexity using straightforward metrics like the problem size ($$n$$) or the number of moves in the solution.

Their findings revealed an interesting inconsistency: models could solve a Tower of Hanoi puzzle requiring over 100 moves, yet fail on a River Crossing puzzle that needs only a dozen. This suggests that simple metrics like "number of moves" don't provide a universal measure of difficulty. Different puzzles stress a model's capabilities in different ways.

This post introduces a framework to normalize the concept of task difficulty into a single, unified metric: **Cognitive Load ($$\mathcal{L}$$)**. By defining difficulty in terms of the underlying computational work a model must perform, we can explain why some seemingly simple problems are incredibly hard for LLMs and predict when their performance will start to degrade.

## A Look at the Puzzles

To understand how cognitive load is calculated, it helps to first understand the mechanics of the puzzles used as benchmarks.

#### Tower of Hanoi
This puzzle features three pegs and $$n$$ disks of different sizes stacked on the first peg in size order (largest at bottom). The goal is to transfer all disks from the first peg to the third peg. Valid moves include moving only one disk at a time, taking only the top disk from a peg, and never placing a larger disk on top of a smaller one. The difficulty can be controlled by the number of initial disks; the minimum number of required moves with $$n$$ disks is $$2^n - 1$$.

#### Checker Jumping
This is a one-dimensional puzzle with red checkers, blue checkers, and a single empty space in a line. The objective is to swap the positions of all red and blue checkers, mirroring the initial configuration. Valid moves include sliding a checker into an adjacent empty space or jumping over exactly one checker of the opposite color to land in an empty space. Checkers cannot move backward. The complexity is controlled by the number of checkers: with $$2n$$ checkers (n of each color), the minimum number of moves is $$(n + 1)^2 - 1$$.

#### River Crossing
This is a constraint satisfaction puzzle involving $$n$$ "actors" and their corresponding $$n$$ "agents" who must cross a river in a boat. The goal is to transport all $$2n$$ individuals from one bank to the other. The boat has a limited capacity (*k*) and cannot travel empty. The core constraint is that an actor cannot be in the presence of another's agent unless their own agent is also present. For n=2 and n=3, the boat capacity is k=2, and for larger n, k=3.

#### Blocks World
This block-stacking puzzle requires rearranging blocks from an initial configuration to a specified goal state. The objective is to find the minimum number of moves. Valid moves are restricted to the topmost block of any stack, which can be placed either on an empty space to start a new stack or on top of another block. The complexity is controlled by the number of blocks. While part of the original study, this puzzle was excluded from our main analysis due to its stochastic nature, which introduces significant noise into the performance data.

## The Three Components of Cognitive Load

Our approach starts with a simple idea about how LLMs solve problems:

*The total computational work, $$W_{\text{total}}$$, an LLM expends to solve a problem is the product of the average work required per generative step, $$W_{\text{step}}$$, and the number of steps in the solution's reasoning chain, $$N_{\text{steps}}$$.*

This can be expressed with the simple equation:

$$
W_{\text{total}} = W_{\text{step}} \times N_{\text{steps}}
$$

We start with the hypothesis that an LLM is more likely to fail as this total work increases. To make this useful, we need to connect the terms in this equation to real, measurable properties of a given puzzle.

The total work can be broken down into three interacting components.

#### Component 1: Solution Path Complexity ($$D$$) - The Length of the Journey

The number of steps, $$N_{\text{steps}}$$, isn't just the final move count. It represents the length of the underlying reasoning chain the model must follow. In the Apple paper, this is referred to as "compositional depth." This corresponds directly to our first component, Solution Path Complexity ($$D$$).

**Definition:** The Solution Path Complexity, $$D(I)$$, for a problem instance $$I$$ is a measure of the effective number of sequential steps in the most efficient known algorithm for solving it.

$$
D(I) \approx N_{\text{steps}}
$$

#### Component 2: State Information Density ($$S$$) - The Weight of Memory

The work per step, $$W_{\text{step}}$$, is the effort needed to generate the next correct token. This effort is partly determined by how much information the model has to keep track of—its internal picture of the problem state. We can quantify this using information theory.

**Definition:** The State Information Density, $$S(I)$$, is the Shannon entropy of the state space for instance $$I$$, measured in bits. For a uniform distribution over possible states, it is the logarithm of the size of the state space, $$\vert \Sigma(I)\vert $$.

$$
S(I) = \log_2(\vert \Sigma(I)\vert )
$$

#### Component 3: Constraint Complexity ($$C$$) - The Intricacy of the Rules

The work per step also depends on how complex the puzzle's rules are. We approximate this by counting the number of fundamental logical predicates needed to express the rules.

**Definition:** The Constraint Complexity, $$C(P)$$, for a puzzle class $$P$$ is the number of fundamental logical or relational predicates required to express the algorithm that validates a single step or move.

$$
C(P) = \text{Number of atomic predicates in the logical definition of the ruleset}
$$

## Synthesis: The Cognitive Load Formula

The key idea here is that these components interact multiplicatively. The effort to apply the rules isn't separate from the effort of holding the state in memory; the rules must be applied *to* the state. So, the work per step is proportional to the product of state and constraint complexity.

$$
W_{\text{step}} \propto S(I) \times C(P)
$$

Putting this all together, we can substitute our definitions back into the original equation to get the final formula for Cognitive Load:

$$
\mathcal{L}(I) = S(I) \times C(P) \times D(I)
$$

This formula provides a concrete method for calculating a single, unified difficulty score for any discrete reasoning task.

Let's apply this formula to the three puzzles mentioned earlier to see how it works in practice.

#### Case Study 1: Tower of Hanoi
-   **State ($$S$$):** For $$n$$ disks and 3 pegs, there are $$3^n$$ configurations.

    $$
    S_{\text{hanoi}}(n) = \log_2(3^n) = n \log_2(3) \approx 1.58n
    $$

-   **Constraints ($$C$$):** A move is valid if: 1) the source disk is the top of its peg, AND 2) the destination peg is either empty OR its top disk is larger. This requires 3 atomic predicates.

    $$
    C_{\text{hanoi}} = 3
    $$

-   **Path ($$D$$):** The solution is recursive, and the number of cognitive steps corresponds to the recursion depth.

    $$
    D_{\text{hanoi}}(n) = n
    $$
    
-   **Cognitive Load ($$\mathcal{L}$$):**

    $$
    \mathcal{L}_{\text{hanoi}}(n) = S(n) \times C \times D(n) = (1.58n) \times 3 \times n = 4.74n^2
    $$

#### Case Study 2: Checker Jumping
-   **State ($$S$$):** The number of states for $$2n+1$$ squares grows such that the log-information is approximately $$2n$$.

    $$
    S_{\text{checkers}}(n) \approx 2n
    $$

-   **Constraints ($$C$$):** The rules for valid moves (slides, jumps, directionality) are complex, requiring approximately 8 atomic predicates to define.

    $$
    C_{\text{checkers}} = 8
    $$

-   **Path ($$D$$):** The puzzle requires planning to avoid dead ends, with a cognitive path length proportional to the minimum solution length.

    $$
    D_{\text{checkers}}(n) = n^2 + 2n
    $$

-   **Cognitive Load ($$\mathcal{L}$$):**

    $$
    \mathcal{L}_{\text{checkers}}(n) \approx (2n) \times 8 \times (n^2 + 2n) = 16n^3 + 32n^2
    $$

#### Case Study 3: River Crossing
-   **State ($$S$$):** With $$n$$ pairs of agents, the state is defined by the location of $$2n$$ individuals and the boat.

    $$
    S_{\text{river}}(n) = \log_2(2 \cdot 2^{2n}) = 2n + 1
    $$

-   **Constraints ($$C$$):** The safety condition must be checked for all actors on both banks, which involves checking pairs of actors. The complexity grows quadratically.

    $$
    C_{\text{river}}(n) \approx 2n(n-1) = 2n^2 - 2n
    $$

-   **Path ($$D$$):** Solving requires a state-space search, and the path length is the number of steps in the shortest solution, $$m(n)$$.

    $$
    D_{\text{river}}(n) = m(n)
    $$

-   **Cognitive Load ($$\mathcal{L}$$):**

    $$
    \mathcal{L}_{\text{river}}(n) \approx (2n+1) (2n^2 - 2n) m(n)
    $$

## Unified Performance Curve

The Apple paper provides accuracy data for Claude 3.7 Sonnet across these puzzles. When we re-analyze their data, not against problem size $$n$$ but against our calculated Cognitive Load, something remarkable happens. The performance data from all these different puzzles, which looked inconsistent before, now collapses onto a single, predictable curve.

By fitting a single sigmoid function (which describes a sharp transition) to the combined data from both puzzles, we can identify a unified critical threshold. The fit gives a consistent critical load of **$$\mathcal{L}_{\text{crit}} \approx 254$$**.

![Cognitive Load Plot](/assets/img/cognitive_load_plot.png)
*Figure 1: Side-by-side comparison of Claude 3.7 Sonnet's performance. **Left:** Plotting accuracy against problem size $$n$$ shows two completely different performance curves. **Right:** Plotting accuracy against the unified Cognitive Load ($$\mathcal{L}$$) metric aligns the data from both puzzles onto a single curve. The grey dashed line shows a single sigmoid curve fitted to all data points, revealing a consistent critical threshold of ~254.*

This curve reveals a unified **critical load threshold ($$\mathcal{L}_{\text{crit}}$$)** for the model. Based on our analysis of the published data, this threshold for Claude 3.7 Sonnet appears to be around **$$\mathcal{L}_{\text{crit}} \approx 254$$**.

This threshold acts as a fundamental constraint on the model's reasoning capacity.

## Predicting Performance: A River Crossing Case Study

The true power of this framework lies not just in explaining performance but in predicting it. We can test this by applying the Cognitive Load formula to the River Crossing problem. The Apple paper reports a sharp performance drop for Claude 3.7 Sonnet on this specific puzzle: from ~80% accuracy for the `n=2` case down to 0% for `n=3`. Let's see if our model predicts this cliff.

-   **Case 1: River Crossing with n=2** (2 pairs, optimal solution has 5 moves)
    -   **Cognitive Load:** $$\mathcal{L}_{\text{river}}(2) \approx (2 \cdot 2 + 1)(2 \cdot 2^2 - 2 \cdot 2) \times 5 = 5 \times 4 \times 5 = 100$$
    -   This load is well **below** the critical threshold of ~254, correctly predicting a high likelihood of success.

-   **Case 2: River Crossing with n=3** (3 pairs, optimal solution has 11 moves)
    -   **Cognitive Load:** $$\mathcal{L}_{\text{river}}(3) \approx (2 \cdot 3 + 1)(2 \cdot 3^2 - 2 \cdot 3) \times 11 = 7 \times 12 \times 11 = 924$$
    -   This load is nearly four times the critical threshold, correctly predicting catastrophic failure.

The framework's predictions align perfectly with the observed results. The model succeeds when the task is below its cognitive limit and fails when the task exceeds it. This demonstrates how Cognitive Load provides a more robust and predictive measure of task difficulty than simple metrics like the number of moves. This also helps explain the paper's finding that models' reasoning effort (measured in token usage) declines at high complexity; once the cognitive load is past the critical threshold, the model effectively gives up.

## A Practical Toolkit: The Standardized Cognitive Assessment Probe (SCAP)

The following is a step-by-step protocol to estimate a model's critical reasoning threshold ($$\mathcal{L}_{\text{crit}}$$). The procedure is designed to be fully self-contained and reproducible.

### Step 1: Measure Maximum State Capacity ($$S_{\text{max}}$$)
**Objective:** Find the maximum amount of state information the model can reliably hold in working memory.

- **Procedure:**
    1. Use the prompt template below. Start with a small number of registers (e.g., $$n_s=5$$).
    2. Incrementally increase $$n_s$$ and test the model repeatedly for each value.
    3. The largest value of $$n_s$$ for which the model consistently provides the correct answer is its maximum state channel capacity, $$n_{s, \text{max}}$$.
- **Prompt Template:**
    > Here is the current state with $$n_s$$ registers: R1=[val1], R2=[val2], ..., Rn=[valn]. The values are three-digit integers. What is the value in register Rk?
- **Calculation:** The first estimate of the critical load, $$\mathcal{L}_S$$, is the maximum state information the model can handle.

    $$
    \mathcal{L}_S = S_{\text{max}} = n_{s, \text{max}} \times \log_2(1000) \approx 9.97 \times n_{s, \text{max}}
    $$

### Step 2: Measure Maximum Constraint Capacity ($$C_{\text{max}}$$)
**Objective:** Find the maximum number of logical rules the model can apply simultaneously in a single step.

- **Procedure:**
    1. Use the prompt template below. Start with a small number of rules (e.g., $$n_c=3$$).
    2. Incrementally increase $$n_c$$ and test the model.
    3. The largest value of $$n_c$$ for which the model consistently calculates the correct final state is its maximum constraint capacity, $$n_{c, \text{max}}$$.
- **Prompt Template:**
    > Initial state: X=[val1], Y=[val2]. Apply all of the following $$n_c$$ rules simultaneously to the initial state and provide the final state of X and Y.
    > Rule 1: If X is [condition], then [action].
    > ...
    > Rule $$n_c$$: If X+Y is [condition], then [action].
- **Calculation:** The second estimate, $$\mathcal{L}_C$$, is the load produced by applying $$C_{\text{max}} = n_{c, \text{max}}$$ rules to the minimal state required for the task. This state holds two variables (values 0-99), so its information content is $$S_{\text{base}} = \log_2(100^2) \approx 13.3$$ bits.

    $$
    \mathcal{L}_C = S_{\text{base}} \times C_{\text{max}} \approx 13.3 \times n_{c, \text{max}}
    $$

### Step 3: Measure Maximum Path Capacity ($$D_{\text{max}}$$)
**Objective:** Find the maximum number of sequential operations the model can track.

- **Procedure:**
    1. Use the prompt template below. Start with a short sequence (e.g., $$n_d=5$$).
    2. Incrementally increase $$n_d$$ and test the model.
    3. The largest value of $$n_d$$ for which the model consistently provides the correct final value is its maximum path capacity, $$n_{d, \text{max}}$$.
- **Prompt Template:**
    > The initial value of register R1 is [val1]. Perform the following sequence of $$n_d$$ operations on R1: [Op1], [Op2], ..., [Op $$n_d$$]. What is the final value of R1?
- **Calculation:** The third estimate, $$\mathcal{L}_D$$, is the load from performing $$D_{\text{max}} = n_{d, \text{max}}$$ operations on a minimal state. The state holds one three-digit number, so $$S_{\text{base}} = \log_2(1000) \approx 10$$ bits.

    $$
    \mathcal{L}_D = S_{\text{base}} \times D_{\text{max}} \approx 10 \times n_{d, \text{max}}
    $$

### Step 4: Synthesize the Final Critical Load ($$\mathcal{L}_{\text{crit}}$$)
**Objective:** Combine the three independent measurements into a single, robust estimate of the model's reasoning limit.

- **Rationale:** The central hypothesis is that the three load values measured at the breaking point for each probe ($$\mathcal{L}_S, \mathcal{L}_C, \mathcal{L}_D$$) are all estimates of the same underlying cognitive limit.
- **Calculation:** Average the three values to obtain the final critical load threshold.

    $$
    \mathcal{L}_{\text{crit}} \approx \text{mean}(\mathcal{L}_S, \mathcal{L}_C, \mathcal{L}_D)
    $$

This protocol provides a fast and resource-efficient method to calibrate the Cognitive Load framework for any LLM, turning it into a powerful predictive tool.

## Conclusion

The Cognitive Load framework provides a structured way to think about and quantify the difficulty of reasoning tasks for LLMs. By breaking down the problem into State, Constraint, and Path complexity, we get a metric that is both theoretically sound and seems to work in practice when applied to existing research data.