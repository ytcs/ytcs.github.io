---
published: true
layout: post
title: "How Index Funds Distort Markets"
categories: markets
date: 2025-09-27
math: true
---

Market-capitalization-weighted index funds are the bedrock of modern passive investing. The logic is simple and powerful: buy the entire market, hold it, and let the magic of compounding do the work. But what happens when trillions of dollars follow this same simple logic, persistently buying stocks merely because they are big? This post explores a simple theory that reveals the profound, market-distorting consequences of this massive, price-agnostic flow of capital. We'll show how a single, physically-motivated assumption about market liquidity can predict the emergence of a structural "alpha" and a warped "beta," creating a predictable class of winners and losers.

## The Core Assumption: Liquidity Scaling
The theory rests on a single, intuitive idea: a stock's ability to absorb capital without its price moving wildly—its market liquidity—does not grow as fast as its size. We can model this with a simple power law. The price impact ($$\eta_i$$), or the percentage price change per dollar of inflow for a stock $$i$$, is related to its market capitalization ($$C_i$$) by:

$$
\eta_i = k \cdot C_i^{-\gamma}
$$

Here, $$k$$ is a market friction constant, and $$\gamma$$ is the crucial **liquidity scaling exponent**. This single parameter governs everything. The key insight is that if market liquidity grows more slowly than market capitalization, then $$\gamma < 1$$. As we will see, this simple condition is all that's required to create significant market distortions.

## The Fundamental Dynamic Equation
From this one assumption, we can derive the equation of motion for any stock in the index. A stock's return ($$dR_i = dC_i / C_i$$) is simply the product of its price impact ($$\eta_i$$) and the capital it receives. In a market-cap index, the inflow to a stock is its weight ($$w_i = C_i / C$$) times the total fund inflow ($$F$$).

Putting it all together:

$$
\frac{1}{C_i}\frac{dC_i}{dt} = \eta_i F w_i = (k C_i^{-\gamma}) F \left(\frac{C_i}{C}\right)
$$

This simplifies to the fundamental equation for the growth of a component's market cap, the workhorse of our theory:

$$
\frac{dC_i}{dt} = \frac{kF}{C} C_i^{2-\gamma}
$$

## The Consequences: Warped Beta and Structural Alpha
This simple dynamic equation has two major consequences for investors, altering the traditional landscape of risk and return.

#### Flow-Driven Beta ($$\beta^F$$)
Beta measures a stock's sensitivity to the overall market's movement. In a world dominated by passive flows, a component of beta emerges that has nothing to do with fundamentals and everything to do with size. This "flow-driven beta" is defined as the ratio of the stock's return to the market's return. Using our dynamic equation, we can derive a closed-form solution:

$$
\beta_i^F = \frac{C \cdot C_i^{1-\gamma}}{\sum_{j=1}^{N} C_j^{2-\gamma}}
$$

The takeaway is clear. When $$\gamma < 1$$, the exponent $$(1-\gamma)$$ is positive. This means that as a company's capitalization $$C_i$$ gets larger, its flow-driven beta gets larger. The passive flow mechanism makes **large-cap stocks systematically more aggressive ($$\beta_i^F > 1$$)** and **small-cap stocks more defensive ($$\beta_i^F < 1$$)**, regardless of their underlying business.

#### Structural Alpha ($$\alpha$$)
Alpha is the excess return of an asset over its beta-adjusted required return. In our deterministic model, it's the excess return over the market index. A component's alpha ($$\alpha_i$$) is its instantaneous rate of excess return:

$$
\alpha_i = \frac{1}{C_i}\frac{dC_i}{dt} - \frac{1}{C}\frac{dC}{dt}
$$

Substituting our expressions for the rates of change yields the solution for structural alpha:

$$
\alpha_i = \frac{kF}{C} \left( C_i^{1-\gamma} - \frac{1}{C}\sum_{j=1}^{N} C_j^{2-\gamma} \right)
$$

This formula reveals a predictable and persistent alpha spread. The term in the parenthesis is the difference between the component's scaled size ($$C_i^{1-\gamma}$$) and the market-wide average of that same metric. When $$\gamma < 1$$, larger firms will have a value greater than the average, resulting in a **predictable positive alpha**. Conversely, smaller firms will have a **predictable negative alpha**. This isn't outperformance based on skill; it's a structural transfer of value from smaller components to larger ones, driven entirely by the mechanics of market-cap indexing.

## A Testable Prediction
This theory isn't just a thought experiment; it makes sharp, testable predictions. One of the most elegant tests is to look at the ratio of betas for any two stocks, $$i$$ and $$j$$. The complicated market-wide sum in the beta formula cancels out perfectly:

$$
\frac{\beta_i^F}{\beta_j^F} = \left(\frac{C_i}{C_j}\right)^{1-\gamma}
$$

Taking the logarithm of both sides gives a linear relationship:

$$
\ln\left(\frac{\beta_i}{\beta_j}\right) = (1-\gamma) \ln\left(\frac{C_i}{C_j}\right)
$$

This provides a clear hypothesis: if you plot the log-ratio of observed historical betas against the log-ratio of market caps for pairs of stocks, you should see a straight line. The slope of that line gives you a direct estimate of $$(1-\gamma)$$, telling you the strength of this market-distorting effect.

![Test of the Square-Root Law: Log-ratio of betas vs. log-ratio of market caps for all pairs of stocks in the S&P 500. We use data from Yahoo Finance. This is a noisy measurement as ETF flow is not the only market force that determines the beta. However the square-root law seems to capture the broad trend of the data.](assets\img\square-root-law.png)

## Implications for Investors
The theory has direct, actionable implications for all market participants.

#### For Active Managers
The existence of a predictable, structural alpha spread is an opportunity. An active manager can construct a market-neutral portfolio that goes long the large-cap stocks with positive predicted alpha and short the small-cap stocks with negative predicted alpha. By design, this strategy is uncorrelated with the market and systematically harvests the value being transferred from structurally disadvantaged to structurally advantaged companies.

#### For Passive Investors
The model serves as a profound warning: "passive" investing is not a neutral act. Over-reliance on market-cap-weighted indices creates systemic risks.
1.  **It is an implicit bet on momentum and concentration.** This exposes your portfolio to massive idiosyncratic risk from a few mega-cap stocks.
2.  **It erodes diversification.** As the index concentrates, the benefits of holding a "basket" of stocks diminish.

Prudent passive investors should consider augmenting their core holdings with strategies that break this feedback loop, such as **equal-weight**, **fundamental**, or **capped-weight** indices.

## Conclusion
Derived from a single assumption about market liquidity, this theory paints a concerning picture. Persistent, price-agnostic inflows into market-cap-weighted indices do not simply reflect the market; they actively shape it. This mechanism inevitably leads to market concentration, a size-based distortion of risk (beta), and the emergence of a predictable alpha spread. These are not market anomalies, but the direct, mathematical consequence of the indexing mechanism itself. It provides a new lens to understand market dynamics and offers clear strategies for both active managers seeking to generate alpha and passive investors seeking to preserve true, long-term diversification.
