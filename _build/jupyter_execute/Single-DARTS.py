#!/usr/bin/env python
# coding: utf-8

# # Single-DARTS

# The original DARTS is formulated as an approximation to a bilevel optimization problem. It updates the network weights on the training set and updates the architecture parameters on the validation set. Due to this, the learnable operations (convolution) will simply be learning noise in the early stages of the training due to the **independence of these different batches** {cite:p}`DBLP:journals/corr/abs-2108-08128`. The non-learnable operations will dominate and the performance collapse of the DARTS will thus be irreversable. 

# The author proposes to update the $w$ and $\alpha$ on the same batch of data and shows that this will lead to much better and also more stable performance.<br>
# $\alpha, w \  -= \xi \nabla_{\alpha,w} L_{train} (\alpha,w)$

# In[ ]:




