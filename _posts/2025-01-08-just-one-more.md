---
published: true
layout: post
title: Avoiding the Just-One-More Paradox
---

When managing our portfolio we need to be mindful about avoiding the Just-One-More paradox (explained [here](https://www.youtube.com/watch?v=_FuuYSM7yOo)) which could lead to ruin even when each trade we take has positve EV.

A simple example used in the video is as follows: for each trade there is a 50% chance to increase our whole portfolio value by 80%, and a 50% chance to decrease it by 50%. The expected value of this trade is a gain of 30%; thus barring any risk constraints this is a good trade. Note that in neither of the scenarios we will realize the mean gain of 30%, but rather we have two diverging outcomes **around the mean**. This is exacerbated when we repeat the same trade over and over again. The **mean** will continuously increase but there is no actual path that would produce anything close to the mean return. Rather we have diverging paths that increases much faster than the mean and paths that decreases quickly to zero, which average out to the mean return. The most probable (and also median) return is produced when there are equal numbers of wins and loses:

$$ R = (1.8\times 0.5)^{n} < 1 $$

The more times we repeat this trade, the higher the probability we drive our portfolio to zero compared to actually making any profit. The key figure of merit here is the most probable return, which needs to be larger than one in order to avoid the "just-one-more" paradox. Mathematically, we want to satisfy the condition

$$ (r_{win})^{p_{win}} \times (r_{lose})^{p_{lose}} > 1 $$

We can use this condition to set price levels when trading. As an example, let's assume that we think the price of a stock will either rise to \\(p_H\\) or fall to \\(p_L\\) with equal probability. We can rewrite our condition in terms of the entry price \\(p_0\\):

$$ \frac{p_H}{p_0} \times \frac{p_L}{p_0} > 1 \implies p_0 < \sqrt{p_H p_L}$$

In other words we want our entry to be lower than the the **geometric mean** of the high and low prices. This is a safer way than simply assuming some risk-reward ratio which most traders do.
