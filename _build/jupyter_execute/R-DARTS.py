#!/usr/bin/env python
# coding: utf-8

# # R-DARTS

# Whereas the iDARTS {cite:p}`zhang2021idarts` includes an extra inverse Hessian (w.r.t the weights) term to obtain a more 'accurate' update on the architecture parameters (via $\nabla_\alpha L_{val}$), the R-DARTS {cite:p}`DBLP:journals/corr/abs-1909-09656`  attempts to analyze the Hessian of the architecture parameters ($\nabla^2_{\alpha \alpha} L_{val}$) to see how this correlates with the performance drop.
# 
# The authors argue that the DARTS do not generalize well because when training the architecture parameter $\alpha$ the model will settle on a sharp local minima; and a sharp minima doesn't generalize as well as a flat minima. The author found a strong correlation between the dominant eigenvalue of the Hessian $\nabla^2_{\alpha \alpha} L_{val}$ (an indicator of sharpness) and the test error.
# 
# Furthermore, the authors hypothesize that one reason for the performance drop is due the discretization of the continuous architecture parameter $\alpha^*$. Because $\alpha^*$ lies in a very sharp region, discretizing it could lead to a significantly worse loss function value. This is less likely in the case of flat minimas. The author investigated this and found that high curvature did in fact lead to large performance drops.
# 
# However, the author proposed an early stopping rule by keeping track of the last few Hessian eigenvalues. Computing the Hessian and obtaining the eigenvalues will incur additional computational resource, even more so than the second-order DARTS.
# 

# In[ ]:




