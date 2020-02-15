#!/usr/bin/env python
# coding: utf-8

# # Description
# 
# You can use script from notebook or from shell.
# 
# Convert this notebook to script (optionally)
# 
#     jupyter nbconvert --to script example_02.ipynb
# 
# 
# Show help message
#     
#     python example_02.py example_02.yaml --help
#     
# 
# Run with additional params:
#     
#     python example_02.py example_02.yaml --tracker.offline=True
#     
# 

# ## Imports

# In[1]:


from composed_trackers import Config, build_from_cfg, TRACKERS, is_notebook, load_config_with_shell_updates
from pathlib import Path


# ## Configuration

# In[2]:


notebook_config = Path('example_02.yaml')

notebook_shell_args = {
    'tracker.offline': True}


# In[3]:


cfg = load_config_with_shell_updates(notebook_config, notebook_shell_args, verbose=True, sep='.')


# ## Create tracker from config

# In[4]:


tracker = build_from_cfg(cfg.tracker, TRACKERS, {'params': cfg.to_flatten_dict(sep='.')})


# In[5]:


tracker.describe()


# ## Use trackers

# In[8]:


tracker.append_tag('introduction-minimal-example')


# In[9]:


n = 117
for i in range(1, n):
    tracker.log_metric('iteration', i)
    tracker.log_metric('loss', 1/i**0.5)
    tracker.log_text('magic values', 'magic value {}'.format(0.95*i**2))
tracker.set_property('n_iterations', n)


tracker.log_text_as_artifact('Hello', 'summary.txt')


# ## Stop

# In[10]:


tracker.stop()


# In[11]:


tracker.describe(ids_only=True)


# In[ ]:




