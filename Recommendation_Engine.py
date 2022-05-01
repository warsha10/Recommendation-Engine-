#!/usr/bin/env python
# coding: utf-8

# ## 1. Import Necessary libraries

# In[1]:


import pandas as pd


# ## 2. Import data

# In[3]:


movie_data = pd.read_csv('Movie.csv', header = TRUE)
movie_data


# ## 3. Data Understanding

# In[4]:


movie_data.shape


# In[6]:


movie_data.movie.unique()


# In[8]:


movie_data.movie.nunique()


# In[5]:


movie_data.head(40)


# In[9]:


movie_data.userId.nunique()


# ### NOTE:
# 
# #### UBCF - User Based Collaborative Filtering
# #### IBCF  - Item Based Collaborative Filtering

# ### 1. Correlation Matrix for Recommendation

# In[15]:


user_recomm = pd.pivot_table(data = movie_data, values='rating',index='movie',columns='userId').fillna(0)
user_recomm


# In[19]:


movie_data['userId'].unique()


# In[20]:


user_recomm.columns = movie_data['userId'].unique()


# In[21]:


user_recomm


# In[23]:


user_recomm.corr().round(2)


# ### 2. Euclideon Distance Metric for Recommedation

# In[24]:


from sklearn.metrics import pairwise_distances


# In[26]:


user_recomm_euclideon = user_recomm.T
user_recomm_euclideon


# In[27]:


user_recomm_euclideon.values


# In[35]:


user_to_user = pairwise_distances(X = user_recomm_euclideon.values, metric='euclidean')
user_to_user


# In[38]:


user_to_user_df = pd.DataFrame(data=user_to_user, columns = movie_data['userId'].unique(), index = movie_data['userId'].unique() )
user_to_user_df


# ### 3. Using Cosine Similarity for Recommedation Engine

# In[43]:


user_to_user_cosine = 1 - pairwise_distances(X = user_recomm_euclideon.values, metric='cosine')
user_to_user_cosine


# In[45]:


user_to_user_cosine_df = pd.DataFrame(data=user_to_user_cosine,columns = movie_data['userId'].unique(), index = movie_data['userId'].unique())
user_to_user_cosine_df


# ### Let's filter the data to understand which user goes well with which user

# In[47]:


first_50users = user_to_user_cosine_df.iloc[:50,:50]
first_50users


# In[ ]:


## USe HeatMap


# In[53]:


import numpy as np
np.fill_diagonal(first_50users.to_numpy(), val = 0)


# In[54]:


first_50users


# In[55]:


first_50users.idxmax()


# In[60]:


movie_data[(movie_data['userId'] == 3) | (movie_data['userId'] == 11)]


# #### Inference:
# 
# The movies watched by 11th user - **Golden Eye** will get recommended to user3 because they are similar.
