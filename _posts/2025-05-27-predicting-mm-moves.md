---
published: true
layout: post
title: "Swimming with the Whale: A Model for Market Maker Price Manipulation"
categories: finance
date: 2025-05-27
---

This post explores the financial dynamics faced by an options market maker (MM), particularly within the volatile context of Zero Days to Expiration (0DTE) options. We will delve into the mechanics of delta hedging, the financial repercussions of imperfect hedging and gamma exposure, particularly as expiration approaches, the MM's Profit and Loss (P&L) characteristics given their current hedge, and the potential incentives for price manipulation as expiration approaches. While we employ the Black-Scholes framework for its analytical tractability, we acknowledge its inherent limitations in accurately modeling 0DTE scenarios. For parts of this analysis, we assume a single (monopolist) MM to simplify the connection between their options book and market observables. The insights from this analysis can be valuable for non-MM participants seeking to understand and potentially anticipate price dynamics influenced by MM activities, especially as 0DTE options approach expiry.

## 1. Options Pricing and Greeks in the Black-Scholes Framework

The Black-Scholes model provides benchmark prices for European call and put options.

A **call option** gives the holder the right, but not the obligation, to buy an underlying asset $$S$$ at a specified strike price $$K$$ on or before a specified expiration date. Its Black-Scholes price $$C(S, \tau)$$ is:

$$
C(S, \tau) = S N(d_1) - K e^{-r\tau} N(d_2)
$$

A **put option** gives the holder the right, but not the obligation, to sell an underlying asset $$S$$ at a specified strike price $$K$$ on or before a specified expiration date. Its Black-Scholes price $$P(S, \tau)$$ is:

$$
P(S, \tau) = K e^{-r\tau} N(-d_2) - S N(-d_1)
$$

Where:
- $$S$$ is the current price of the underlying asset.
- $$K$$ is the strike price of the option.
- $$\tau$$ is the time to expiration (in years).
- $$r$$ is the continuously compounded risk-free interest rate.
- $$\sigma$$ is the annualized volatility of the underlying asset's returns.
- $$N(\cdot)$$ is the cumulative distribution function (CDF) of the standard normal distribution.

The terms $$d_1$$ and $$d_2$$ are given by:

$$
d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)\tau}{\sigma\sqrt{\tau}}
$$

$$
d_2 = d_1 - \sigma\sqrt{\tau}
$$

### Key Greeks
The "Greeks" are sensitivities of the option price to changes in underlying parameters.
- **Delta ($$\Delta$$)**: Measures the rate of change of the option price with respect to a dollar change in the underlying asset's price.

  $$
  \Delta = \frac{\partial V}{\partial S}
  $$

  where $$V$$ is the option price. For calls, $$\Delta_C = N(d_1)$$, and for puts, $$\Delta_P = N(d_1) - 1 = -N(-d_1)$$.
- **Gamma ($$\Gamma$$)**: Measures the rate of change of Delta with respect to a dollar change in the underlying asset's price.

  $$
  \Gamma = \frac{\partial^2 V}{\partial S^2}
  $$

  For both calls and puts, $$\Gamma = \frac{N'(d_1)}{S\sigma\sqrt{\tau}}$$, where $$N'(x)$$ is the probability density function (PDF) of the standard normal distribution.

For 0DTE options, as $$\tau \to 0$$, Gamma for at-the-money (ATM) options (where $$S \approx K$$) becomes extremely large. This implies that Delta can change dramatically with even minor movements in the underlying price $$S$$.

## 2. Delta Hedging: The Ideal and The Intense Reality of 0DTE

Market makers typically aim to profit from the bid-ask spread on options they trade, rather than from directional bets on the underlying asset. They manage the price risk of their options positions primarily through delta hedging.

### 2.1. The Continuous Delta Hedging Ideal
The goal of delta hedging is to maintain a portfolio whose value is insensitive to small changes in the underlying asset's price. For an option position with value $$V$$, a delta-neutral portfolio $$\Pi$$ can be constructed by holding $$-\Delta_V$$ shares of the underlying asset for each option held (if $$V$$ represents being long the option), or $$\Delta_V$$ shares if $$V$$ represents the value of a short option position.
Let $$V$$ be the value of an MM's short option position. To hedge, the MM holds $$H = \Delta_V$$ shares of the underlying $$S$$. The portfolio value is $$\Pi = -V + HS$$.
The change in value of this portfolio, $$d\Pi$$, over an infinitesimal time interval $$dt$$, assuming continuous re-hedging, is:

$$
d\Pi = -dV + H dS
$$

By Ito's lemma, the change in option value $$dV$$ (for a single option, signs adjusted for a short position later) is:

$$
dV = \frac{\partial V}{\partial t} dt + \frac{\partial V}{\partial S} dS + \frac{1}{2} \frac{\partial^2 V}{\partial S^2} (dS)^2
$$

In Black-Scholes, $$(dS)^2 = S^2 \sigma^2 dt$$. Let $$\Theta = \frac{\partial V}{\partial t}$$, $$\Delta_V = \frac{\partial V}{\partial S}$$, and $$\Gamma_V = \frac{\partial^2 V}{\partial S^2}$$.
Then, $$dV$$ can be written as:

$$
dV = \Theta dt + \Delta_V dS + \frac{1}{2} \Gamma_V S^2 \sigma^2 dt
$$

If the MM is delta-neutral, $$H = \Delta_V$$. So, the change in the MM's portfolio value (where they are short the option $$V$$) is:

$$
d\Pi = -\left(\Theta + \frac{1}{2} \Gamma_V S^2 \sigma^2\right) dt
$$

Under Black-Scholes assumptions (and if $$V$$ represents a long option position), this $$d\Pi$$ would be the risk-free return. For an MM who is short options, this represents the decay they collect, offset by the cost of being short gamma.

### 2.2. 0DTE Hedging: Challenges and Inevitable Discrepancies
Perfect, continuous delta hedging is a theoretical ideal. For 0DTE options, practical challenges are significantly amplified:
-   **Extreme Gamma**: As noted, ATM Gamma skyrockets as $$\tau \to 0$$. Delta changes explosively with minor price moves in $$S$$, demanding extremely frequent, precise, and potentially large re-hedges.
-   **Transaction Costs & Market Impact**: Frequent re-hedging incurs substantial transaction costs and can exert market impact, affecting execution prices.
-   **System Latency & Execution Risk**: A finite time lag invariably exists between detecting a delta change requirement and executing the hedge. The underlying price $$S$$ can move significantly within this lag.
-   **Liquidity Fluctuations**: 0DTE liquidity can be volatile, especially for out-of-the-money strikes, making it difficult to hedge at desired prices or in required sizes.

These factors mean an MM's hedge can easily become mismatched from the true current delta of their options book. As expiration draws nearer, the rapidly increasing gamma and escalating transaction costs can make continuous, perfect re-hedging practically unattainable, leading to a situation where the MM's established hedge becomes effectively "stuck" or, at best, imperfectly managed relative to the fast-changing ideal.

## 3. P&L Consequences of a "Stuck" Hedge Near Expiration

As established, market makers (MMs) face escalating difficulties in maintaining a perfect delta hedge for 0DTE options as expiration nears. Extreme gamma, transaction costs, and liquidity constraints can lead to a situation where the MM's hedge, let's call it $$H_{\text{stuck}}$$, was established based on a previous market price $$S_{\text{ref}}$$ and market conditions, but is no longer perfectly aligned with the current underlying price $$S_{\text{current}}$$.

Consider the MM's portfolio just before expiration. They hold an options book (net short value $$V(S,t)$$) and their effectively "stuck" hedge $$H_{\text{stuck}}$$. Suppose the underlying price moves from $$S_{\text{ref}}$$ (where the hedge $$H_{\text{stuck}} = \Delta_V(S_{\text{ref}})$$, the delta of the short book at $$S_{\text{ref}}$$, was appropriate) to a new price $$S_{\text{current}}$$ over a short period where the hedge could not be perfectly adjusted. The change in the MM's portfolio value due to this unhedged component of price movement is:

$$
{\text{P\&L}}_{\text{gamma\_exposure}} = \left[-V(S_{\text{current}}, t_1) + H_{\text{stuck}} S_{\text{current}}\right] - \left[-V(S_{\text{ref}}, t_0) + H_{\text{stuck}} S_{\text{ref}}\right]
$$

Rearranging terms:

$$
{\text{P\&L}}_{\text{gamma\_exposure}} = -\left(V(S_{\text{current}}, t_1) - V(S_{\text{ref}}, t_0)\right) + H_{\text{stuck}} (S_{\text{current}} - S_{\text{ref}})
$$

Assuming the time interval $$t_1 - t_0$$ is very small, so $$t_1 \approx t_0$$, and the primary impact on option value comes from the change in $$S$$. We approximate $$V(S_{\text{current}}, t_1)$$ using a Taylor series expansion of $$V(S, t_0)$$ around $$S_{\text{ref}}$$, and recalling that $$H_{\text{stuck}} = \Delta_V(S_{\text{ref}})$$, the delta of the short options book at $$S_{\text{ref}}$$. Letting $$\Gamma_V(S_{\text{ref}})$$ be the gamma of the short options book at $$S_{\text{ref}}$$, the expansion gives:

$$
V(S_{\text{current}}, t_1) - V(S_{\text{ref}}, t_0) \approx H_{\text{stuck}} (S_{\text{current}} - S_{\text{ref}}) + \frac{1}{2} \Gamma_V(S_{\text{ref}}) (S_{\text{current}} - S_{\text{ref}})^2
$$

Substituting this back into the $$ {\text{P\&L}}_{\text{gamma\_exposure}} $$ equation:

$$
{\text{P\&L}}_{\text{gamma\_exposure}} \approx -\left( H_{\text{stuck}} (S_{\text{current}} - S_{\text{ref}}) + \frac{1}{2} \Gamma_V(S_{\text{ref}}) (S_{\text{current}} - S_{\text{ref}})^2 \right) + H_{\text{stuck}} (S_{\text{current}} - S_{\text{ref}})
$$

$$
{\text{P\&L}}_{\text{gamma\_exposure}} \approx -\frac{1}{2} \Gamma_V(S_{\text{ref}}) (S_{\text{current}} - S_{\text{ref}})^2
$$

If the MM is net short options, their book $$V$$ typically has positive gamma ($$\Gamma_V(S_{\text{ref}}) > 0$$). The MM's overall portfolio is therefore short gamma. Consequently, any significant unhedged price movement $$(S_{\text{current}} - S_{\text{ref}})^2 > 0$$ results in a negative $$ {\text{P\&L}}_{\text{gamma\_exposure}} $$. This represents the realized cost due to the combination of being short gamma and the inability to perfectly adjust the hedge $$H_{\text{stuck}}$$ during the price movement.

This P&L is largely a sunk cost by the time the MM evaluates their position at $$S_{\text{current}}$$. The key takeaway is that due to hedging frictions near expiration, such gamma-driven P&L events are likely. Following such an event, or simply acknowledging the current "stuck" nature of their hedge, the MM must manage their position into expiration.

## 4. MM P&L Profile with a "Stuck" Hedge into Expiration

Having arrived at a price $$S_{\text{current}}$$ (let's call this $$S_c$$ for current price) with a hedge $$H_{\text{stuck}}$$ that is no longer perfectly optimal (or perhaps was the result of a recent, best-effort re-hedge), the MM faces the final moments to expiration. The P&L incurred due to prior gamma exposure (as discussed in Section 3) is now a sunk cost. The critical question is how their P&L will behave from this point ($$S_c$$ with hedge $$H_{\text{stuck}}$$) until the final expiration price $$S_T$$. We will denote $$H_{\text{stuck}}$$ as $$H_c$$ for the current hedge quantity.

For the remainder of the 0DTE period (from $$S_c$$ with hedge $$H_c$$ until expiration at price $$S_T$$), the MM is effectively "stuck" with their existing options book (quantities $$Q_k$$) and its inherent gamma profile, alongside their current hedge $$H_c$$. Let $$O_k(S_c, K_k)$$ be the market value of option $$k$$ (per unit) at price $$S_c$$ when this evaluation is made. The P&L from this point until expiration is:

$$
\text{P\&L}_{S_c \to S_T}(S_T) = \sum_k Q_k \left[O_k(S_c, K_k) - \text{Payoff}_k(S_T, K_k)\right] + H_c (S_T - S_c)
$$

Here, $$Q_k$$ represents the quantity of option $$k$$ sold by the MM (so $$Q_k > 0$$ for short positions). $$\text{Payoff}_k(S_T, K_k)$$ is the final payoff of option $$k$$ at expiration if the underlying price is $$S_T$$. For example, for a call option, $$\text{Payoff}_k(S_T, K_k) = \max(0, S_T - K_k)$$.
This P&L can be rewritten to separate terms dependent on $$S_T$$:

$$
\text{P\&L}_{S_c \to S_T}(S_T) = \left( \sum_k Q_k O_k(S_c, K_k) - H_c S_c \right) + H_c S_T - \sum_k Q_k \text{Payoff}_k(S_T, K_k)
$$

The term in parentheses, $$\left( \sum_k Q_k O_k(S_c, K_k) - H_c S_c \right)$$, is constant with respect to the final expiration price $$S_T$$. The payoff functions $$\text{Payoff}_k(S_T, K_k)$$ are piecewise linear with kinks at their respective strikes $$K_k$$. Therefore, $$\text{P\&L}_{S_c \to S_T}(S_T)$$ is also piecewise linear in $$S_T$$.

### Optimal Expiration Price ($$S_{\text{MM}}^*$$) for the MM
The MM, having established the hedge $$H_c$$, seeks to maximize $$\text{P\&L}_{S_c \to S_T}(S_T)$$ with respect to the unknown expiration price $$S_T$$. The optimal expiration price for the MM, $$S_{\text{MM}}^*$$, can be found by examining the derivative of this P&L with respect to $$S_T$$:

$$
m(S_T) = \frac{d \text{P\&L}_{S_c \to S_T}(S_T)}{d S_T} = H_c - \sum_k Q_k \frac{d \text{Payoff}_k(S_T, K_k)}{d S_T}
$$

Let $$\Delta_k^{\text{exp}}(S_T, K_k) = \frac{d \text{Payoff}_k(S_T, K_k)}{d S_T}$$ be the option's delta at expiration. This delta is:
- For a call option: $$1$$ if $$S_T > K_k$$ (ITM), $$0$$ if $$S_T < K_k$$ (OTM), and can be considered $$0.5$$ or undefined if $$S_T = K_k$$.
- For a put option: $$-1$$ if $$S_T < K_k$$ (ITM), $$0$$ if $$S_T > K_k$$ (OTM).

The sum $$\sum_k Q_k \Delta_k^{\text{exp}}(S_T, K_k)$$ represents the Net Delta Obligation at Expiration, $$\text{NDO}(S_T)$$, if the price settles at $$S_T$$. This is the total number of shares the MM will have to deliver (for short calls) or receive (for short puts that are exercised against them).
So, the slope of the P&L function is:

$$
m(S_T) = H_c - \text{NDO}(S_T)
$$

Since $$H_c = \sum_k Q_k \Delta_k(S_c, K_k)$$ (the aggregate delta of the options book valued at $$S_c$$), we have:

$$
m(S_T) = \sum_k Q_k \Delta_k(S_c, K_k) - \text{NDO}(S_T)
$$

The function $$m(S_T)$$ is piecewise constant and changes values only at the strike prices $$K_k$$. The optimal expiration price $$S_{\text{MM}}^*$$ for the MM is typically a strike price $$K_k$$ where $$m(S_{\text{MM}}^*) \approx 0$$, or more precisely, where $$m(S_T)$$ changes sign from positive (for $$S_T < S_{\text{MM}}^*$$) to negative (for $$S_T > S_{\text{MM}}^*$$) for a local maximum. This condition $$H_c \approx \text{NDO}(S_{\text{MM}}^*)$$ means that the MM's hedge $$H_c$$ (taken at $$S_c$$) most closely matches the net shares needed for settlement at expiration if $$S_T = S_{\text{MM}}^*$$.

## 5. Potential for Price Manipulation by the MM

The MM, now holding a hedge of $$H_c$$ shares based on the price $$S_c$$, might consider influencing the final settlement price $$S_T$$ away from its "natural" level $$S_T^n$$ (which might be proxied by the current market price $$S_c$$) towards a manipulated price $$S_T^m$$.
Let $$a$$ be the marginal cost for the MM to move the price by 1 dollar (e.g., $$a$$ dollars per dollar price move per share involved in pushing). This is a simplification, as actual manipulation costs are complex and nonlinear.

The marginal P&L gain for the MM from moving $$S_T$$ is given by $$m(S_T) = H_c - \text{NDO}(S_T)$$. Manipulation becomes attractive if this marginal P&L exceeds the marginal cost $$a$$.
The condition for initiating manipulation is:

$$
\vert m(S_T^n) \vert > a
$$

Substituting the expression for $$m(S_T^n)$$:

$$
\vert H_c - \text{NDO}(S_T^n) \vert > a
$$

The term $$H_c - \text{NDO}(S_T^n)$$ represents the MM's "excess hedge" (if positive) or "hedge deficit" (if negative) relative to what would be needed if the price expires at $$S_T^n$$.

- If $$H_c - \text{NDO}(S_T^n) > a$$: The MM is over-hedged for $$S_T^n$$ (has more shares than needed if price stays at $$S_T^n$$) and benefits from $$S_T$$ rising. They have an incentive to push the price up towards $$S_{\text{MM}}^*$$.
- If $$H_c - \text{NDO}(S_T^n) < -a$$: The MM is under-hedged for $$S_T^n$$ (has fewer shares than needed) and benefits from $$S_T$$ falling. They have an incentive to push the price down towards $$S_{\text{MM}}^*$$.

The MM would theoretically attempt to move $$S_T$$ towards $$S_{\text{MM}}^*$$ as long as the marginal benefit $$\vert m(S_T) \vert$$ exceeds the marginal cost $$a$$.

## 6. Deriving a Market Observable for $$S_{\text{MM}}^*$$ and Manipulation Indicator

To make this operational, we assume a single (monopolist) MM whose sold quantities $$Q_k$$ correspond to the total market open interest (OI) for each option strike $$K_k$$.
- $$Q_k^{\text{call}} = \text{OI}_{\text{C}}(K_k)$$ (Open Interest for calls at strike $$K_k$$)
- $$Q_k^{\text{put}} = \text{OI}_{\text{P}}(K_k)$$ (Open Interest for puts at strike $$K_k$$)

The MM's optimal expiration price $$S_{\text{MM}}^*$$ is where their hedge $$H_c$$ approximately equals their Net Delta Obligation $$\text{NDO}(S_{\text{MM}}^*)$$.

### MM's Estimated Current Delta Hedge ($$H_{\text{MM}}(S_c, \sigma)$$)
We can proxy $$S_c$$ (the price at which the last major re-hedge $$H_c$$ occurred) with the current underlying price $$S_c$$. The time to expiration $$\tau_c$$ is very close to zero for 0DTE options.
The MM's current total delta hedge, $$H_{\text{MM}}$$, can be estimated using current option open interests and Black-Scholes deltas calculated at $$S_c$$ and an estimated 0DTE implied volatility $$\sigma$$:

$$
H_{\text{MM}}(S_c, \sigma) = \sum_{K_k} \text{OI}_{\text{C}}(K_k) \Delta_{\text{C}}(S_c, K_k, \sigma, \tau_c) + \sum_{K_k} \text{OI}_{\text{P}}(K_k) \Delta_{\text{P}}(S_c, K_k, \sigma, \tau_c)
$$

Where $$\Delta_{\text{C}}(S_c, K_k, \sigma, \tau_c)$$ and $$\Delta_{\text{P}}(S_c, K_k, \sigma, \tau_c)$$ are the deltas of call and put options, respectively, at strike $$K_k$$, given current price $$S_c$$, volatility $$\sigma$$, and near-zero time to expiry $$\tau_c$$.

### Net Delta Obligation at Expiration ($$\text{NDO}(S_T)$$)
The Net Delta Obligation at expiration $$S_T$$ depends on which options expire in-the-money:

$$
\text{NDO}(S_T) = \sum_{K_k < S_T} \text{OI}_{\text{C}}(K_k) \cdot (1) - \sum_{K_k > S_T} \text{OI}_{\text{P}}(K_k) \cdot (1)
$$

This formula assumes the MM is short calls (owes 1 share per ITM call) and short puts (receives 1 share per ITM put, which is a negative obligation for shares to *deliver*, or effectively reduces the number of shares they need to be long).

### Observable $$S_{\text{MM}}^*$$ Calculation
An observable estimate for $$S_{\text{MM}}^*$$ can be calculated as follows:
1.  Collect current open interest data: $$\text{OI}_{\text{C}}(K_k)$$ and $$\text{OI}_{\text{P}}(K_k)$$ for all relevant strikes $$K_k$$.
2.  Use the current underlying price $$S_c$$ as a proxy for $$S_c$$.
3.  Calculate the MM's estimated current total delta hedge $$H_{\text{MM}}(S_c, \sigma)$$.
4.  For each strike price $$K_j$$ in the option chain, calculate the Net Delta Obligation $$\text{NDO}(K_j)$$ that would occur if $$S_T = K_j$$.
5.  $$S_{\text{MM}}^*$$ is the strike $$K_j$$ for which $$\text{NDO}(K_j)$$ is closest to $$H_{\text{MM}}(S_c, \sigma)$$. That is, $$S_{\text{MM}}^*$$ is the $$K_j$$ that minimizes $$\vert H_{\text{MM}}(S_c, \sigma) - \text{NDO}(K_j) \vert$$. This is the strike where the MM's P&L slope $$m(K_j)$$ is closest to zero.

### 6.1. An Indicator for Predicting Price Manipulation
Let $$S_c$$ be the current market price, taken as a proxy for $$S_T^n$$. The MM's current delta hedge is approximated as $$H_c \approx H_{\text{MM}}(S_c, \sigma)$$.
The **Current Delta Imbalance (CDI)** at $$S_c$$ is defined as:

$$
\text{CDI}(S_c) = H_{\text{MM}}(S_c, \sigma) - \text{NDO}(S_c)
$$

This $$\text{CDI}(S_c)$$ represents the slope $$m(S_c)$$ of the MM's P&L function $$\text{P\&L}_{S_c \to S_T}(S_T)$$ evaluated at the current price $$S_c$$. It indicates the MM's exposure per dollar change in $$S_T$$ from $$S_c$$.

A **Manipulation Indicator** can be formulated: Price manipulation by the MM is predicted if $$\vert \text{CDI}(S_c) \vert > a$$, where $$a$$ is some threshold value we choose that represents the marginal cost of manipulation.
- If $$\text{CDI}(S_c) > a$$: The MM is over-hedged at $$S_c$$ and has an incentive to push the price upward, towards $$S_{\text{MM}}^*$$ (which would likely be greater than $$S_c$$).
- If $$\text{CDI}(S_c) < -a$$: The MM is under-hedged at $$S_c$$ and has an incentive to push the price downward, towards $$S_{\text{MM}}^*$$ (which would likely be less than $$S_c$$).

**Interpretation**:
This theoretical framework outlines how a market maker's hedging activities, particularly in the high-gamma environment of 0DTE options, can lead to specific P&L profiles and potential incentives to influence the market's expiration price. The increasing difficulty of perfect hedging as expiration approaches can lead to an effectively "stuck" hedge. This current hedge, when evaluated against the potential share obligations at different expiration prices, creates a P&L landscape where the MM may have a preferred expiration point ($$S_{\text{MM}}^*$$). The concept of $$S_{\text{MM}}^*$$, the MM's optimal expiration price given their current hedge, and the Current Delta Imbalance (CDI) provide a basis for understanding potential "pinning" pressures around certain strikes. While simplified, particularly in its assumptions about a monopolist MM and manipulation costs, this model offers insights into the complex dynamics market makers face and potentially induce in 0DTE markets when their ability to continuously and cost-effectively re-hedge is compromised by the extreme nature of near-expiry options.