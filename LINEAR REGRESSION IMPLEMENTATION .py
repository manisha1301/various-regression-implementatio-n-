#!/usr/bin/env python
# coding: utf-8

# # practical implementation on linear regression on boston housing price prediction 

# topic covered:-

# ##1 boston dataset analysis using EDA
# ##2 implementation of linear Regression
# ##3 implementation of Ridge Regression 
# ##4 implementation of Lasso Regression 
# ##5 implementation of Elastic Net Regression 

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[2]:


## loading Dataset from sklearn library 
from sklearn.datasets import load_boston


# In[3]:


boston = load_boston()


# In[4]:


print(boston.data)


# In[5]:


print(boston)


# In[6]:


boston.keys()


# In[7]:


print(boston.filename)


# In[8]:


print(boston.target)


# In[9]:


print(boston.DESCR)


# In[10]:


print(boston.feature_names)


# In[11]:


## let prepare the Dataframe


# In[12]:


dataset = pd.DataFrame(boston.data,columns = boston.feature_names)


# In[13]:


dataset


# In[14]:


dataset['price'] = boston.target


# In[15]:


dataset


# In[16]:


dataset.info()


# In[17]:


dataset.describe()


# In[18]:


## for checking missing value

dataset.isnull().sum()


# In[19]:


## eda

dataset.corr()   ## for analysis the dataset , we use correlation.


# In[20]:


import seaborn as sns
sns.pairplot(dataset)


# In[21]:


sns.set(rc = {'figure.figsize' : (10,8)})

sns.heatmap(dataset.corr(),annot = True)


# In[22]:


plt.scatter(dataset['CRIM'],dataset['price'])
plt.xlabel("Crime rate")
plt.ylabel("Price")


# In[23]:


sns.set(rc={'figure.figsize':(8,6)})
sns.regplot(x = "RM", y = "price", data = dataset)


# In[24]:


sns.regplot(x = "CRIM", y = "price", data = dataset)


# In[25]:


sns.regplot(x = "LSTAT", y = "price", data = dataset)


# In[26]:


sns.regplot(x = "CRIM", y = "price", data = dataset)


# In[27]:


sns.boxplot(dataset['price'])


# In[28]:


sns.boxplot(dataset['CRIM'])


# In[29]:


dataset.head(3)


# In[30]:


## Independent and Dependent Feature

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]


# In[31]:


x


# In[32]:


y


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


X_train,X_test, Y_train,Y_test = train_test_split(x,y,test_size = 0.33)


# In[35]:


X_train


# In[36]:


X_train.shape


# In[37]:


Y_train


# In[38]:


Y_train.shape


# In[39]:


X_test.shape


# In[40]:


Y_test.shape


# # standardize or feature scalling the dataset

# In[41]:


## standardize or feature scalling the dataset 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[42]:


scaler     ## Scaler is an object, an standard scaler objects


# In[43]:


X_train = scaler.fit_transform(X_train)     ## Fit_transport means 


# In[44]:


X_test 


# In[45]:


X_train


# In[46]:


X_test = scaler.transform(X_test)


# # model training

# ## linear Regression model

# In[47]:


from sklearn.linear_model import LinearRegression 


# In[48]:


regression = LinearRegression()


# In[49]:


regression 


# In[50]:


regression.fit(X_train, Y_train)


# ### print the coefficients and the intercept

# In[51]:


## print the coefficients
print(regression.coef_)


# In[52]:


## print the intercept
print(regression.intercept_)


# In[53]:


## predicition for the test data
reg_pred = regression.predict(X_test)


# In[54]:


reg_pred


# # assumption of linear regression

# In[55]:


import seaborn as sns 
import matplotlib.pyplot as plt


# In[56]:


## relationship between real data & predicated data

plt.scatter(Y_test,reg_pred)    
plt.xlabel("test Truth Data")
plt.ylabel("test Predicted Data")


# In[57]:


## calculating residual
residuals = Y_test - reg_pred


# In[58]:


residuals


# In[59]:


## distrubution of residual are approximately normal distribution 
sns.displot(residuals, kind = "kde")


# In[60]:


## scatter plot with predictions and resdiual 
### uniform distributions
plt.scatter(reg_pred, residuals) 


# # performance metrics 

# In[61]:


## performance metrics

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_squared_error(Y_test, reg_pred))
print(mean_absolute_error(Y_test,reg_pred))
print(np.sqrt(mean_squared_error(Y_test,reg_pred)))


# # r squared and adjusted r-squared

# In[62]:


from sklearn.metrics import r2_score 
score = r2_score(Y_test,reg_pred)
print(score)


# In[63]:


#adjusted R square 
#display adjusted R-squared
1 - (1-score)*(len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1)


# # Ridge Regression Model

# In[64]:


from sklearn.linear_model import Ridge


# In[65]:


ridge = Ridge()


# In[66]:


ridge


# In[67]:


ridge.fit(X_train,Y_train)


# # print the coefficient  and the intercept

# In[68]:


print(ridge.coef_)          ##coefficient 


# In[69]:


print(ridge.intercept_)         ##intercept


# In[70]:


##predication for test data
ridge_pred = ridge.predict(X_test)


# In[71]:


ridge_pred


# # assumptions of ridge regression 

# In[72]:


plt.scatter(Y_test,ridge_pred)
plt.xlabel("test truth data")
plt.ylabel("test Predicted data")


# In[73]:


###calculating residuals:-
residuals = Y_test-ridge_pred


# In[74]:


residuals 


# In[75]:


sns.displot(residuals,kind = "kde")


# In[76]:


## scatter plot with predictions and residual 
##uniform distribution
plt.scatter(ridge_pred,residuals)


# # performance metrics

# In[77]:


from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 
print(mean_squared_error(Y_test,ridge_pred))
print(mean_absolute_error(Y_test,ridge_pred))
print(np.sqrt(mean_squared_error(Y_test,ridge_pred)))


# # R squared and adjusted R squared

# In[78]:


from sklearn.metrics import r2_score
score= r2_score(Y_test,ridge_pred)
print(score )


# In[79]:


##adjusted r square
#display adjusted r-squared
1 - (1-score )*(len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1)


# # lasso Regression 

# In[80]:


from sklearn.linear_model import Lasso
lasso = Lasso()
lasso


# In[81]:


lasso.fit(X_train,Y_train)


# # print the cofficients and the intercept

# In[82]:


print(lasso.coef_)


# In[83]:


print(lasso.intercept_)


# In[84]:


##predication for the test data
lasso_pred = lasso.predict(X_test)


# In[85]:


lasso_pred


# # assumption of lasso regression 

# In[86]:


plt.scatter(Y_test,lasso_pred)
plt.xlabel("test truth data")
plt.ylabel("test Predicted  data")


# In[87]:


#calculating residuals 
residuals =Y_test- lasso_pred
residuals 


# In[88]:


## distributions of residual are approximately normal distribution
sns.displot(residuals,kind="kde")


# In[89]:


## scatter plot with predictions and residual 
## uniform distribution

plt.scatter(lasso_pred,residuals)


# # performance metrics

# In[90]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_squared_error(Y_test,lasso_pred))
print(mean_absolute_error(Y_test,lasso_pred))
print(np.sqrt(mean_squared_error(Y_test,lasso_pred)))


# # R squared and adjusted R squared

# In[91]:


from sklearn.metrics import r2_score 
score = r2_score(Y_test,lasso_pred)
print(score )


# In[92]:


##adjusted r square
#display adjusted r-squared
1 - (1-score )*(len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1)


# # elastic net regression model

# In[93]:


from sklearn.linear_model import ElasticNet


# In[94]:


elastic = ElasticNet()


# In[95]:


elastic


# In[96]:


elastic.fit(X_train,Y_train)


# # print the cofficients and the intercept

# In[97]:


print(elastic.coef_)


# In[98]:


print(elastic.intercept_)


# In[99]:


elastic_pred = elastic.predict(X_test)


# In[100]:


elastic_pred


# # assumptions of elasticNet regression 

# In[101]:


plt.scatter(Y_test,elastic_pred)
plt.xlabel("test truth data")
plt.ylabel("test Predicted  data")


# In[102]:


resuiduals =Y_test- elastic_pred


# In[103]:


residuals


# In[104]:


sns.displot(residuals,kind="kde")


# In[105]:


plt.scatter(elastic_pred,residuals)


# # performance metrics

# In[106]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_squared_error(Y_test,elastic_pred))
print(mean_absolute_error(Y_test,elastic_pred))
print(np.sqrt(mean_squared_error(Y_test,elastic_pred)))


# # R squared and adjusted R squared

# In[107]:


from sklearn.metrics import r2_score 
score = r2_score(Y_test,elastic_pred)
print(score )


# In[108]:


1 - (1-score )*(len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1)

