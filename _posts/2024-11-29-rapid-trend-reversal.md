---
published: true
layout: post
title: Sharp reversals as an indicator of TR-dominated market
---

In the last post we discussed some high-level qualitative features of market behavior where we introduced the concept of a market dominated by trend-followers (TR-heavy). This note will go over another qualitative feature of a TR-heavy market.

We let \\(p'_n\\) be the current first derivative of the asset price (change in price per tick) . We assume that the market is sufficiently liquid such that the traded volume is given by some aggregate demand function \\(\nu(p')\\) (volume per tick, positive if buy and negative if sell) from the TR trades. We had also assumed that this function is stationary in time by limiting ourselves to a sufficiently short time-scale as to exclude any e.g. price shock or news event. The **price impact** resultant of this aggregate demand is again modelled by another function \\(\pi(\nu)\\), which suggests that the price change is governed (approximately) by some recurrance equation of the form

$$ p'_{n+1} \approx \pi(\nu(p'_n)) = F(p'_n) $$

There are more we can say about this equation without knowing its exact form. We expect this equation to be homogeneous, i.e. with the trivial fixed-point \\(p'=0\\) (price stablization). The stability of the trivial fixed-point is given by the product of the sentivity of the price impact to volume and the sentitivity of the volume to the derivative of price. In other words, if the price is not sensitive to the volume or there are not enough traders in the market then the market will continue to consolidate and mean-revert even if the market is TR-heavy.

Furthermore we expect \\(F\\) to be a non-decreasing function. Thus if the price is able to escape from the trivial fixed-point (breakout) to the upside we expect the slope to either increase indefinitely (hyperbolic rise), or reach some positive fixed point (linear rise). Similar if the price is breaking out to the downside.

The interesting collorary here is that if we are already seeing a breakout e.g. to upside, meaning the market is sufficiently sentitive and populated, then when the price reverts (e.g. due to a price shock) we do not expect stablization to be a possibility (at least on the short time scale) but rather a hyperbolic collapse or a linear decline, resulting in a U-shape or V-shape reversal. This is often attributed to fear and panic in the market but as seen in the discussion above this would happen even if the traders are being completely mathematical. The appearance of pronounced U or V shaped reversals indicate that the market is populated (perhaps over-crowded) with trend-following traders or algorithms that dominate the price action.
