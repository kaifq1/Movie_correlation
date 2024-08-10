#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


plt.style.use('ggplot')
from matplotlib.pyplot import figure


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (12, 8)


# In[49]:


df = pd.read_csv(r"C:\Users\Kaif\Downloads\movies.csv.zip")


# In[51]:


df.head()


# In[52]:


for col in df.columns:
    percent_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col,percent_missing))


# In[55]:


df.dtypes


# null or infinite values to 0

# In[58]:


df = df.replace([np.inf, -np.inf], np.nan).fillna(0)


# In[60]:


df['budget'] =df['budget'].astype('int64')
df['gross'] =df['gross'].astype('int64')


# In[62]:


df['yearcorrect'] = df['released'].astype(str).str[:]
df.head(15)


# In[64]:


df = df.sort_values(by=['gross'],inplace=False, ascending=False)
pd.set_option('display.max_row',None)


# In[66]:


df.drop_duplicates()


# In[67]:


plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross')
plt.xlabel('Gross')
plt.ylabel('Budget')
plt.show()


# In[68]:


sns.regplot(x='budget' ,y='gross', data=df, scatter_kws={"color":"red"},line_kws={"color":"blue"})


# In[69]:


df.corr(method='pearson')


# In[74]:


df.corr(method='kendall')


# In[76]:


df.corr(method='spearman')


# In[92]:


print(correlation_matrix)


# In[100]:


correlation_matrix = df.corr(method='pearson')
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix , annot=True)
plt.title('Correlation for Numeric features')
plt.xlabel('Features')
plt.ylabel('Features')
plt.show


# In[88]:


correlation_matrix = df.corr(method='kendall')
sns.heatmap(correlation_matrix , annot=True)
plt.show


# In[84]:


df_numerized = df
for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
df_numerized


# In[110]:


df_numerized.corr()


# In[120]:


correlation_mat = df_numerized.corr()
corr_pairs = correlation_mat.unstack()

corr_pairs


# In[124]:


sorted_pairs = corr_pairs.sort_values()

sorted_pairs


# In[134]:


high_corr=sorted_pairs[(sorted_pairs) > 0.5]
high_corr


# HIGHEST CORRELATION : VOTES & BUDGET

# LOWEST CORRELATION  : COMPANY

# In[ ]:




