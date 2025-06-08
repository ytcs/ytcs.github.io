---
published: true
layout: post
title: "Why Some Puzzles Are Hard for LLMs: A Framework for Cognitive Load"
categories: machine-learning
date: 2025-06-08
math: true
---

## Toward a Universal Measure of Reasoning Difficulty

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

## The Postulate of Computational Work

Our approach starts with a simple idea about how LLMs solve problems:

*The total computational work, $$W_{\text{total}}$$, an LLM expends to solve a problem is the product of the average work required per generative step, $$W_{\text{step}}$$, and the number of steps in the solution's reasoning chain, $$N_{\text{steps}}$$.*

This can be expressed with the simple equation:

$$
W_{\text{total}} = W_{\text{step}} \times N_{\text{steps}}
$$

We start with the hypothesis that an LLM is more likely to fail as this total work increases. To make this useful, we need to connect the terms in this equation to real, measurable properties of a given puzzle.

## The Three Components of Cognitive Load

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

## Case Studies: The Framework in Action

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

## The Payoff: A Unified Performance Curve

The Apple paper provides accuracy data for Claude 3.7 Sonnet across these puzzles. When we re-analyze their data, not against problem size $$n$$ but against our calculated Cognitive Load, something remarkable happens. The performance data from all these different puzzles, which looked inconsistent before, now collapses onto a single, predictable curve.

By fitting a single sigmoid function (which describes a sharp transition) to the combined data from both puzzles, we can identify a unified critical threshold. The fit gives a consistent critical load of **$$\mathcal{L}_{\text{crit}} \approx 254$$**.

![Cognitive Load Plot](/assets/img/cognitive_load_plot.png)
*Figure 1: Side-by-side comparison of Claude 3.7 Sonnet's performance. **Left:** Plotting accuracy against problem size $$n$$ shows two completely different performance curves. **Right:** Plotting accuracy against the unified Cognitive Load ($$\mathcal{L}$$) metric aligns the data from both puzzles onto a single curve. The grey dashed line shows a single sigmoid curve fitted to all data points, revealing a consistent critical threshold of ~254.*

This curve reveals a unified **critical load threshold ($$\mathcal{L}_{\text{crit}}$$)** for the model. Based on our analysis of the published data, this threshold for Claude 3.7 Sonnet appears to be around **$$\mathcal{L}_{\text{crit}} \approx 254$$**.

This threshold acts as a fundamental constraint on the model's reasoning capacity.

## Predicting Performance: A River Crossing Case Study

The true power of this framework lies not just in explaining performance but in predicting it. We can test this by applying the Cognitive Load formula to the River Crossing problem. The Apple paper reports a sharp performance drop for Claude 3.7 Sonnet on this specific puzzle: from ~80% accuracy for the `n=2` case down to 0% for `n=3`. Let's see if our model predicts this cliff.

-   **Case 1: River Crossing with n=2** (2 pairs, optimal solution has 5 moves)
    -   **Cognitive Load:** $$\\mathcal{L}_{\\text{river}}(2) \\approx (2 \\cdot 2 + 1)(2 \\cdot 2^2 - 2 \\cdot 2) \\times 5 = 5 \\times 4 \\times 5 = 100$$
    -   This load is well **below** the critical threshold of ~254, correctly predicting a high likelihood of success.

-   **Case 2: River Crossing with n=3** (3 pairs, optimal solution has 11 moves)
    -   **Cognitive Load:** $$\\mathcal{L}_{\\text{river}}(3) \\approx (2 \\cdot 3 + 1)(2 \\cdot 3^2 - 2 \\cdot 3) \\times 11 = 7 \\times 12 \\times 11 = 924$$
    -   This load is nearly four times the critical threshold, correctly predicting catastrophic failure.

The framework's predictions align perfectly with the observed results. The model succeeds when the task is below its cognitive limit and fails when the task exceeds it. This demonstrates how Cognitive Load provides a more robust and predictive measure of task difficulty than simple metrics like the number of moves. This also helps explain the paper's finding that models' reasoning effort (measured in token usage) declines at high complexity; once the cognitive load is past the critical threshold, the model effectively gives up.

## A Practical Toolkit: The Standardized Cognitive Assessment Probe (SCAP)

This framework is most useful if we can predict when a model will fail. But how can we find a model's $$\\mathcal{L}_{\\text{crit}}$$ without throwing hundreds of puzzles at it? For this, we developed a simple and efficient testing protocol: the **Standardized Cognitive Assessment Probe (SCAP)**.

SCAP uses simple, targeted prompts to measure the limits of a model's capacity for each of the three components—State, Constraints, and Path—in isolation. The protocol has three parts. In each one, we increase a specific variable ($$n_s, n_c, n_d$$) until the model's accuracy consistently drops, pinpointing the breaking point for that component.

1.  **Probe 1 (State Capacity $$S_{\text{max}}$$)**: Tests working memory by asking the model to recall a value from one of $$n_s$$ registers. $$S_{\text{max}}$$ is calculated from the maximum $$n_s$$ the model can handle.
2.  **Probe 2 (Constraint Capacity $$C_{\text{max}}$$)**: Tests rule application by giving the model an initial state and $$n_c$$ conditional rules to apply. $$C_{\text{max}}$$ is the maximum $$n_c$$ the model can apply correctly.
3.  **Probe 3 (Path Capacity $$D_{\text{max}}$$)**: Tests sequential reasoning by asking the model to perform a sequence of $$n_d$$ operations. $$D_{\text{max}}$$ is the maximum number of steps it can track.

By finding the load at each of these individual breaking points, we can estimate the model's general threshold, $$\mathcal{L}_{\text{crit}}$$, by averaging them. This gives us a fast, resource-efficient way to calibrate the framework for any LLM, turning it into a powerful predictive tool.

$$
\mathcal{L}_{\text{crit}} \approx \text{mean}(\mathcal{L}_S, \mathcal{L}_C, \mathcal{L}_D)
$$

## Conclusion

The Cognitive Load framework provides a structured way to think about and quantify the difficulty of reasoning tasks for LLMs. By breaking down the problem into State, Constraint, and Path complexity, we get a metric that is both theoretically sound and seems to work in practice when applied to existing research data.

It gives us a unified scale to predict model performance and diagnose why a model might be failing. With the SCAP protocol, it also becomes a practical tool for measuring the core reasoning capacity ($$\mathcal{L}_{\text{crit}}$$) of any model.

Ultimately, this work helps build a more rigorous science of AI reasoning, allowing us to better understand the limits of today's models and, hopefully, build more capable and robust systems in the future.

---
### References
1. Shojaee, P., et al. (2024). *The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity*. [https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf) 