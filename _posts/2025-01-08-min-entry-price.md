## Avoiding the "Just-One-More" Paradox

When managing our portfolio we need to be mindful about avoiding the Just-One-More paradox (explained [here](https://www.youtube.com/watch?v=_FuuYSM7yOo)) which could lead to ruin even when each trade we take has positve EV.

A simple example used in the video is as follows. For each trade there is a 50% chance to increase our whole portfolio value by 80%, and a 50% chance to decrease it by 50%. The expected value of this trade is a gain of 30% thus barring any risk constraints we should take this trade. Note that in neither of the scenario we will realize the mean gain of 30%, but rather we have two diverging outcomes **around the mean**. This is exacerbated when we repeat the same trade over and over again. The **mean** will continuously increase but there is no actual path that would produce anything close to the mean return. Rather we have diverging paths that increases much faster than the mean and paths that decreases quickly to zero which averages out to the mean return. The most probable (and also median) return is when there are equal numbers of wins and loses, given by

R_{probable} = (1.8\times 0.5)^{n} < 1

The more times we repeat this trade, the higher the probability we have to drive our portfolio to zero than to actually making any profit. We see that the key figure of merit here is the most probable return
