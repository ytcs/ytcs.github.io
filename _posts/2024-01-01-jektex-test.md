---
layout: post
title: "Jektex Rendering Test"
date: 2024-01-01
categories: test
---

This post tests various LaTeX syntaxes with Jektex.

## Kramdown Notation

Inline Kramdown: $$ \mathbb{E}_{K}[X] $$ should render.

Display Kramdown:

$$
\mathcal{K} = \sum_{i=1}^{N} W_i
$$

This should be on its own line.

## LaTeX-Style Notation

Inline LaTeX-style: \(\mathbb{R}^n\) should render.

Another inline: \( H(X \mid K) := \sum_{k \in \mathcal{K}} p(k) H(X \mid K=k) = \mathbb{E}_{K} [H(X \mid K=k)] \)

Display LaTeX-style:
\[
\int_a^b f(x) dx = F(b) - F(a)
\]

This should also be on its own line.

Let's try a specific problematic one: \( \mathbb{E}_{K} \)
And another: \( \mathcal{X} \) 