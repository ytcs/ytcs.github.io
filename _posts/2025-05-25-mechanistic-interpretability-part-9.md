---
published: true
layout: post
title: "Mechanistic Interpretability: Part 9 - Induction Heads: The Mechanics of In-Context Learning"
categories: machine-learning
date: 2025-05-25
---

This far in our series, we have explored the foundational concepts of mechanistic interpretability, from superposition and dictionary learning to the general taxonomy of circuits and attention head patterns ([Part 1]({% post_url 2025-05-25-mechanistic-interpretability-part-1 %}) through [Part 8]({% post_url 2025-05-25-mechanistic-interpretability-part-8 %})). We now arrive at one of the most celebrated discoveries in Transformer circuitry: **induction heads**. These specialized circuits are believed to be a key mechanism enabling Transformers to perform a basic but powerful form of **in-context learning**—recognizing and completing repeated patterns within the current input sequence.

## What are Induction Heads and Why Are They Important?

An induction head is a type of macro-circuit, typically formed by at least two attention heads working in composition, that implements an algorithm for sequence completion based on repetition. The canonical example they solve is completing a pattern like: `A B C ... A B __` by predicting `C`.

Their significance lies in several areas:
1. **In-Context Learning:** They demonstrate a concrete mechanism by which Transformers can learn from the immediate context provided in their input, a hallmark of their impressive capabilities.
2. **Algorithmic Understanding:** They are one of the first complex, multi-head algorithmic circuits to be reverse-engineered in detail, showing that Transformer behavior isn't just opaque associations but can involve learnable, implementable algorithms.
3. **Emergent Capability:** Induction heads tend to emerge reliably in Transformers (even small ones) as they are trained on text data, suggesting this is a fundamental computation for sequence modeling.

## The Canonical Two-Head Induction Circuit

While variations exist, the most commonly described induction circuit involves the composition of two key attention heads (let's call them Head 1 and Head 2, typically in different layers, with Head 1 preceding Head 2).

Consider the task of completing the sequence `... TokenP TokenQ ... TokenP ___` (where `TokenP` is the token at position `t-1` in the prompt we want to complete, and we want to predict `TokenQ`).

1. **Head 1: The "Previous Token" Head (or Information Gathering Head)**
    - **Function:** At or before position `t-1` (where `TokenP` occurs), Head 1 attends to `TokenP`.
    - **QK Circuit:** Its QK circuit is specialized to identify `TokenP` (e.g., it could be a previous token head attending to `t-1` if the query is at `t`, or a content-based head that picks out `TokenP`).
    - **OV Circuit:** Its OV circuit then *copies* the representation of `TokenP` (or a feature vector strongly indicative of `TokenP`) into the residual stream. Let this copied representation be $$V_P$$.
    - **Mathematical Effect:** After Head 1, the residual stream at some position (say, `t-1` or `t`) now contains $$V_P$$.

2. **Head 2: The "Induction" Head (or Pattern Matching & Copying Head)**
    - **Function:** At the current position `t` (where we need to make a prediction), Head 2 needs to find a previous occurrence of `TokenP` in the context, and then copy the token that *followed that previous TokenP*.
    - **K-Composition is Key:** Head 2 uses the information $$V_P$$ (copied by Head 1 and now present in the residual stream) to form its Key vectors. When Head 2 processes a token at some earlier position `k` in the context to generate a key $$\\mathbf{k}_k^{(H2)}$$, this key is influenced by $$V_P$$. Specifically, if the token at position `k` *is* an instance of `TokenP`, then $$\\mathbf{k}_k^{(H2)}$$ becomes strongly associated with `TokenP` due to the presence of $$V_P$$ in its input from the residual stream.
        $$\\mathbf{k}_k^{(H2)} = (\\mathbf{x}_k^{\\text{orig}} + \\dots + V_P + \\dots) \\mathbf{W}_K^{(H2)}$$
        The term $$V_P \\mathbf{W}_K^{(H2)}$$ makes the key vector for `TokenP` distinctive.
    - **QK Circuit of Head 2:** The Query vector $$\\mathbf{q}_t^{(H2)}$$ at the current position `t` (which is also effectively \"about\" `TokenP` because the model is trying to complete a sequence starting with `TokenP`) will now match strongly with the Key vectors $$\\mathbf{k}_k^{(H2)}$$ from previous positions `k` where `TokenP` occurred.
        So, Head 2 attends to previous instances of `TokenP`.
    - **Shifting Attention for Prediction:** Crucially, an effective induction head doesn't just attend to the *previous TokenP itself*, but to the token *following* it (`TokenQ` at position `k+1`). This offset can be learned via modifications to the QK circuit (e.g., specific positional biases in $$\mathbf{W}_Q^{(H2)}$$ or $$\mathbf{W}_K^{(H2)}$$) or by an interaction with positional encodings that allow the query at position $$t$$ to target keys that are effectively one step ahead of the matched `TokenP`.
    - **OV Circuit of Head 2:** The OV circuit of Head 2 is typically a copying circuit ($$\\mathbf{W}_V^{(H2)} \\mathbf{W}_O^{(H2)} \\approx \\mathbf{I}$$). Once Head 2 attends to `TokenQ` (at position `k+1`), it copies `TokenQ`\'s representation to the current position `t`, leading to the prediction of `TokenQ`.

## Mathematical Mechanisms Underlying Induction

- **Composition (Primarily K-Composition):** The core idea is that Head 1 writes information about `TokenP` to the residual stream. Head 2 reads this information when forming its Key vectors. This allows Head 2\'s attention pattern to be conditional on `TokenP`.
    - The virtual weight from Head 1\'s output to Head 2\'s key input is $$\\mathbf{W}_{\\text{out}}^{(H1)} \\mathbf{W}_K^{(H2)}$$. This term, when applied to Head 1\'s internal state representing `TokenP`, shapes Head 2\'s keys.

- **Effective QK Circuit for Head 2:** Due to composition, Head 2\'s effective QK matrix (the one that determines its attention scores based on the *original* token embeddings, before Head 1\'s contribution) becomes more complex. It implicitly computes something like: \"Score how well the current query (related to `TokenP`) matches a previous key position `j` if token `j` is `TokenP`, and then look at `j+1`.\"

- **Role of Value-Vectors and OV Circuit:** The values $$\\mathbf{v}_j^{(H2)}$$ fetched by Head 2 are critical. If Head 2 attends to position `k+1` (where `TokenQ` is), then $$\\mathbf{v}_{k+1}^{(H2)}$$ must represent `TokenQ`. The OV circuit of Head 2 ($$\\mathbf{W}_V^{(H2)}\\mathbf{W}_O^{(H2)}$$) then ensures this representation of `TokenQ` is written to the residual stream at position `t` to inform the final prediction.

- **Positional Information:** While not always fully elucidated, positional encodings are thought to play a role, especially in helping Head 1 identify the *current* `TokenP` (e.g., at `t-1`) and potentially helping Head 2 implement the crucial \"attend to position `k+1` after matching `TokenP` at `k\" logic.

## Simplified Example: The Algorithm

1. **Prompt:** `... A B ... A ___` (current position is after the second `A`). `TokenP` is `A`.
2. **Head 1 (at/near second `A`):** Identifies the second `A`. Copies its representation $$V_A$$ to the residual stream.
3. **Head 2 (at `___` position):**
    * Its Query is effectively asking: \"What follows `A`?\"
    * Its Keys are formed using $$V_A$$. So, when it sees the *first* `A` in the context, its Key vector becomes strongly representative of `A`.
    * Its QK circuit (due to its internal weights and the $$V_A$$-influenced Keys) now makes it attend strongly to the position *after* the first `A` (i.e., where `B` is).
    * Its OV circuit copies the representation of `B` from that attended position.
4. **Prediction:** `B`.

## Significance and Limitations

- **Emergence:** Induction heads are not explicitly programmed; they emerge through standard training on sequence prediction tasks. Their discovery was a major step in showing that complex, learnable algorithms reside within Transformers.
- **Basis for More Complex ICL:** While simple, this two-token induction (`A B -> A B`) is believed to be a building block for more sophisticated forms of in-context learning (e.g., few-shot learning on novel patterns not explicitly seen in training but following a similar meta-pattern).
- **Limitations:** Basic induction heads are good at exact string matching and completion. They might struggle with more abstract pattern completion that requires semantic understanding beyond lexical repetition.

## Conclusion

Induction heads are a compelling example of a non-trivial, algorithmically specific circuit within Transformers. They demonstrate how composition of simpler attention head functionalities (like identifying previous tokens and content-based attention/copying) can lead to emergent capabilities like in-context learning of repetitions. The careful interplay of QK and OV circuits, mediated by modifications to the residual stream and primarily driven by K-composition, allows the model to look back, find relevant patterns, and copy information to make future predictions. This discovery has been pivotal in advancing the field of mechanistic interpretability, proving that we can, indeed, reverse-engineer parts of the algorithms learned by these complex models.

This concludes our initial nine-part exploration into the foundations and key findings of mechanistic interpretability. The field is rapidly evolving, with ongoing research into more complex circuits, the role of MLP layers, and scaling these techniques to even larger models.

---

## References and Further Reading

- **Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., ... & Olah, C.** (2022). [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/). *Transformer Circuits Thread*. (The primary source and most detailed explanation of induction heads).
- **Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., ... & Olah, C.** (2021). [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/). *Transformer Circuits Thread*. (Provides the compositional framework used to understand induction heads).
- **OpenAI.** (2023). [GPT-4 Technical Report](https://cdn.openai.com/papers/gpt-4.pdf) (and similar model cards often allude to in-context learning capabilities without detailing mechanisms). 