#!/usr/bin/env python
# coding: utf-8

# ### Hessian in Neural Network Optimization

# The Hessian contains useful information about the curvature of the optimization landscape. However, second-order methods (such as Newton's method) that incoporates curvature information are not as popular as first-order methods in Deep Learning due to the fact that computation of the whole Hessian is intractable and that it may not always point in a descent direction. Even for convex problems, Quasi-Newton's methods are preferred over pure Newton's methods, much less high-dimensional non-convex problems.
# 
# Research effort has been expended in order to improve Neural Network optimization through the Hessian {cite:p}`https://doi.org/10.48550/arxiv.2012.03801` and {cite:p}`https://doi.org/10.48550/arxiv.2208.05924` provides
# empirical studies to asses the generalization performance of various Neural Network models on multiple datasets. They have both used the Hutchinson method to estimate the trace of the Hessian and then regularize it. Alternatively, {cite:p}`https://doi.org/10.48550/arxiv.1901.10159` and {cite:p}`https://doi.org/10.48550/arxiv.1912.07145` approximate the eigenvalue spectral density 
# of the whole optimization process and both show that batch normalization is key in controlling the eigenvalues. {cite:p}`https://doi.org/10.48550/arxiv.1901.10159` further posits that the existence of outlier eigenvalues impede optimization in the relatively flatter directions of the loss. All these papers have shown increased generalization performance by incoporating curvature information. 
# 
# Both SDARTS {cite:p}`DBLP:journals/corr/abs-2002-05283` and R-DARTS {cite:p}`DBLP:journals/corr/abs-1909-09656` analyze the performance drop of DARTS through the lens of the Hessian **(w.r.t $\alpha$)** and propose regularization methods. 
# 
# The iDARTS {cite:p}`zhang2021idarts` include an inverse Hessian **(w.r.t w)** term in the gradient update.

# In[ ]:




