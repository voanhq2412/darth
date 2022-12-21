#!/usr/bin/env python
# coding: utf-8

# # iDARTS 

# Unlike some recent papers that have proposed regularization approaches for the DARTS, the iDARTS {cite:p}`zhang2021idarts` proposes a different way to compute the outer-loop gradient based on the implicit function theorem.<br> 
# 
# The Amended-DARTS {cite:p}`bi2020stabilizing` has actually arrived at the same outer-loop gradient (hypergradient) formula, stating that the gradient approximation in the original DARTS has been inaccurate. {cite:p}`bi2020stabilizing` have replaced the inverse Hessian with the Hessian itself in the formula to make computation more tractable however calculating the Hessian is still costly and it takes more time for the Amended-DARTS to reach optimality. Replacing the inverse Hessian within Hessian is also not a good approximation.
# 
# The iDARTS on the other hand, approximates the inverse Hessian and thus reduces the computational cost of the Amended-DARTS, as well as provides better results.
# 
# From the Original DARTS we had: <br>
# $\nabla_\alpha L_{val} (w^*(a),a) \approx \nabla_\alpha L_{val} (w - \xi \nabla_w L_{train}(w,\alpha),a)$ <br>
# 
# Ideally, we'd want the $w - \xi \nabla_w L_{train}(w,\alpha)$ to be as close to the optimal $w^*(\alpha)$ as possible however this is very unlikely because as a result of the approximation the weights of the inner loop get updated only once.
# 

# However, if indeed $w ' = w - \xi \nabla_w L_{train}(w,\alpha)$ was optimal then we would have the training loss gradient $\nabla_{w} L_{train}(w^*,\alpha) = 0$ {cite:p}`zhang2021idarts`

# Thus:<br>
# $ \nabla^2_{w,\alpha} L_{train}(w^*,\alpha) = 0$<br>
# $ \nabla_{w} (\nabla_{\alpha} L_{train}(w^*,\alpha)) = 0$<br>
# $ \nabla_{w} (\nabla_{\alpha} L_{train}(w^*,\alpha) + \nabla_{w} L_{train}(w^*,\alpha). \nabla_{\alpha} w) = 0$<br>
# $ \nabla^2_{w,\alpha} L_{train}(w^*,\alpha) + \nabla^2_{w,w} L_{train}(w^*,\alpha). \nabla_{\alpha} w = 0$<br>
# $\nabla_{\alpha} w = - \big[\nabla^2_{w,w} L_{train}(w^*,\alpha)   \big]^{-1} . \nabla^2_{w,\alpha} L_{train}(w^*,\alpha)$<br>

# Because the inverse Hessian of the training loss w.r.t the weights is expensive to compute the authors have proposed to use the Neumann approximation approach {cite:p} `DBLP:journals/corr/abs-1911-02590`
# computed based on mini-batch samples. 

# $\big[\nabla^2_{w,w} L_{train}(w^*,\alpha)   \big]^{-1} = \underset{i \to \infty}{lim} \overset{i}{\underset{j=0}{\sum}} \big[ I - \nabla^2_{w,w} L_{train}(w^*,\alpha) \big]^j    $

# By using only the first K terms of the Neumann series:<br>
# $\nabla_\alpha L_{val} (w',a) \approx \nabla_\alpha L_{val} (w' ,a) - \xi \nabla^2_{w\alpha} L_{train}(w,\alpha). \overset{K}{\underset{j=0}{\sum}} \big[ I - \xi \nabla^2_{w,w} L_{train}(w^*,\alpha) \big]^j  . \nabla_{w'} L_{val}(w',\alpha) \ \ \ \ $<br>

# In[ ]:




