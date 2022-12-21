#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('bash', '', 'jupyter nbconvert DARTS_paper_summary.ipynb --to latex --template citations.tplx\nlatex DARTS_paper_summary.tex\nbibtex DARTS_paper_summary.aux\npdflatex DARTS_paper_summary.tex\n# pdflatex DARTS_paper_summary.tex\n# pdflatex DARTS_paper_summary.tex\n')


# In[ ]:




