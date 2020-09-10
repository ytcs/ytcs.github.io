---
layout: post
title:  "Thermodynamics of Self-replicating Systems"
---
In 2013 Jeremy England published an interesting inequality which is essentially the equivalent of Landauer's bound for biology.

He derived a non-equilibrium generalization of the second law:

$$ \beta \Delta Q_{I\to II} + \ln \Big( \frac{\pi(II\to I)}{\pi(I\to II}\Big) + \Delta S_{int} \geq 0 $$

where $ I $ represents the macrostate of having a single organism and $ II $ represents the macrostate where the organism replicated into two. Using a simple exponential model with growth rate $ g $ and decay rate $ \delta $, we can express the tranistion probabilities in terms of these rates:

$$ \beta \Delta Q_{I\to II} + \Delta S_{int} \geq \ln(g/\delta) $$

From this we can see that the maximum growth rate of an organism is bounded by the heat dissipated and its internal entropy. In other words organisms that are simpler/less-organised (high $ S_{int} $) and produces more heat (high $ Q $) would have a reproduction advantage.


<iframe width="560" height="315" src="https://www.youtube.com/embed/10cVVHKCRWw" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

References
1. [Statistical physics of self-replication](http://www.englandlab.com/uploads/7/8/0/3/7803054/2013jcpsrep.pdf)
