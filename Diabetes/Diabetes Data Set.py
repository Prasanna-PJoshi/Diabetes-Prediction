#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('diabetes.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


Y=df['Outcome']
X=df.drop(['Outcome'],axis=1)


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=.2, random_state=20)


# ### SVM(RBF)

# In[8]:


from sklearn.svm import SVC 
model=SVC(kernel='rbf')
model.fit(X_train, Y_train)


# In[9]:


model.score(X_test, Y_test)


# In[10]:


yp=model.predict(X_test)


# In[12]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,yp)


# In[14]:


from sklearn.metrics import confusion_matrix

cf=confusion_matrix(Y_test,yp)
print(cf)


# In[20]:


import seaborn as sns

mat = sns.heatmap(cf, annot=True, cmap='Reds')


# ### RandomForest

# In[30]:


from sklearn.ensemble import RandomForestClassifier
model_RF=RandomForestClassifier()
model_RF.fit(X_train,Y_train)


# In[31]:


model_RF.score(X_test,Y_test)


# ### SVM (POLY)

# In[34]:


model_poly=SVC(kernel='poly')
model_poly.fit(X_train,Y_train)


# In[35]:


model_poly.score(X_test,Y_test)


# In[49]:


yp2=model_poly.predict(X_test)


# In[51]:


mat2=confusion_matrix(yp2,Y_test)
mat2


# In[52]:


matrix = sns.heatmap(mat2, annot=True, cmap='Reds')


# ### Visual Presentation

# In[ ]:





# In[ ]:





# In[ ]:





# ## Logistic Regression

# In[41]:


from sklearn.linear_model import LogisticRegression
model_LR=LogisticRegression()
model_LR.fit(X_train, Y_train)


# In[42]:


model_LR.score(X_test,Y_test)


# In[ ]:




