#!/usr/bin/env python
# coding: utf-8

# # SDARTS

# The Smooth-DARTS (SDARTS) {cite:p}`DBLP:journals/corr/abs-2002-05283` builds upon the empirical results of {cite:p}`DBLP:journals/corr/abs-1909-09656`. The authors show mathematically why the performance drop of the architecture parameter discretization step is correlated with the Hessian norm.
# 
# In particular, let $(w^*,\alpha^*)$ be the values that minimizes the validation loss in the continuous space and $\bar{\alpha}$ the architecture parameter after discretizing $\alpha^*$. Using the Taylor series:<br>
# 
# $L_{val}(w^*,\bar{\alpha}) = L_{val}(w^*,\alpha^*) + (\bar{\alpha} - \alpha^*).\nabla_{\alpha}L_{val}(w^*,\alpha^*) + (\bar{\alpha} - \alpha^*)^T.\nabla^2_{\alpha\alpha}L_{val}(w^*,\alpha^*).(\bar{\alpha} - \alpha^*)$

# Because of optimality condition: $\nabla_{\alpha}L_{val}(w^*,\alpha^*) = 0 $<br>
# 
# $L_{val}(w^*,\bar{\alpha}) = L_{val}(w^*,\alpha^*) +(\bar{\alpha} - \alpha^*)^T.\nabla^2_{\alpha\alpha}L_{val}(w^*,\alpha^*).(\bar{\alpha} - \alpha^*)$<br>
# 
# Thus the performance drop is bounded by:<br>
# $C = ||\nabla^2_{\alpha\alpha}L_{val}(w^*,\alpha^*)||.||\bar{\alpha} - \alpha^*||^2 $
# 
# After tuning the weights given $\bar{\alpha}$, we obtain $L_{val}(\bar{w},\bar{\alpha})$ which will be smaller than $L_{val}(w^*,\bar{\alpha})$, and thus the performance drop is still bounded by $C$.

# The authors proposed two regularization methods, one based on the idea of randomized smoothing and the other based on adversarial training, that implicitly control the Hessian norm and redirect the loss function to a smooth landscape $L_{val}(w(\alpha^*),\alpha^* + \Delta)$ w.r.t the pertubation $\Delta$. Experiment results show that both methods are better and require less computational resource than the R-DARTS (L2).

# In[ ]:




