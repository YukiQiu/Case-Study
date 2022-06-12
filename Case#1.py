#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('loans_full_schema.csv')
data.head(5)


# In[3]:


data.info()


# In[4]:


data.describe()


# ## Data Cleaning
# There are 10 columns having missing values from upper result.

# In[5]:


total_length = 10000
null_data = []
null_col = []
null_rate = []
for column in list(data.columns):
    if data[column].isna().sum() > 0:
        null_col.append(column)
        null_data.append(data[column].isna().sum())
        null_rate.append(data[column].isna().sum()/10000*100)
miss_data = pd.DataFrame(columns = ['name','missing values','percentage'])
miss_data['name'] = null_col
miss_data['missing values'] = null_data
miss_data['percentage'] = null_rate
miss_data.sort_values('percentage',ascending = False)


# In[6]:


# Drop the columns that has high missing values ratios, suppose the ratio is 0.5
data.drop(miss_data[miss_data.percentage >= 50]['name'],axis = 1, inplace = True)


# In[7]:


data.head()


# In[8]:


# For other missing values, I tended to replace them by mode and for emp_title, I used other to replace
data['emp_title'] = data['emp_title'].fillna('other')
columns = ['months_since_last_credit_inquiry','emp_length','num_accounts_120d_past_due','debt_to_income']
for col in columns:
    data[col] = data[col].fillna(data[col].mode()[0])


# In[9]:


# check if there are missing values 
data.isna().sum().sum()


# In[10]:


corr = data.corr()['interest_rate']


# In[11]:


corr.sort_values()


# In[12]:


#drop columns that have very small correlations with interest_rate
data.drop(['num_accounts_30d_past_due','paid_principal','current_accounts_delinq','num_accounts_120d_past_due','state','emp_title','issue_month'],axis = 1,inplace=True)


# In[13]:


data.columns


# In[14]:


cat_cols = data.select_dtypes(include=("object"))
cat_cols


# In[15]:


# data['issue_month'] = data['issue_month'].astype('datetime64')


# In[ ]:





# ## Data Visulization

# In[16]:


# Correlation heatmap
corr1 = data.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr1)


# In[17]:


# Distribution of interest rate
# The distribution of interest rate is slightly right skewed, I will consider to take the log function on interest rate later if the result is not that good
from scipy.stats import norm
plt.figure(figsize=(10,7))
sns.distplot(data['interest_rate'],color = 'blue',fit = norm)
plt.title('Distribution of Interest Rate')


# In[18]:


plt.figure(figsize=(10,7))
sns.scatterplot(x='grade',y = 'interest_rate',data = data)
plt.title('annual income vs. interest rate')
# As the grade goes for alphabetic order, the intere rate goes higher


# In[19]:


plt.figure(figsize=(10,7))
sns.scatterplot(x='total_debit_limit',y = 'interest_rate',data = data)


# In[20]:


plt.figure(figsize=(10,7))
sns.scatterplot(x='paid_interest',y = 'interest_rate',hue = 'grade',data = data)
#lower paid interest, lower grade ,lower interest rate 


# In[21]:


plt.figure(figsize=(10,7))
sns.lineplot(x='emp_length',y = 'interest_rate',data = data)
# From employment year 0-4, the interest rate decreased by the increasing of the years
# But it fluctuated from year 6-10


# In[22]:


plt.figure(figsize = (10,7))
sns.barplot(x = 'verified_income',y='interest_rate',data = data)
# highly veryfied income get the highr interest rate


# ## Data Normalization
# Perform normalization to make it on the same scale

# In[23]:


data = pd.get_dummies(data,columns = ['homeownership','verified_income','loan_status','initial_listing_status','disbursement_method',
                        'grade','sub_grade','loan_purpose','application_type'],drop_first=True)
len(data.columns)


# In[24]:


# data['issue_month'] = data['issue_month'].astype('datetime64')


# In[25]:


X = data.drop(['interest_rate'],axis = 1)
Y = data['interest_rate']


# In[26]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size = 0.3)


# In[27]:


# forward feature selection
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
# nfeatures = len(X_train.columns)
# sfs = SFS(LinearRegression(),
#           k_features=30,
#           forward=True,
#           floating=False,
#           scoring = 'r2',
#           cv = 0)
# sfs.fit(X_train,y_train)
# len(sfs.k_feature_names_)


# In[28]:


# Use lasso feature selection to increase the accuracy and decrese the overfitting possibility 

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
pipeline = Pipeline([
                     ('scaler',StandardScaler()),
                     ('model',Lasso())
])
lasso = GridSearchCV(pipeline,
                      {'model__alpha':np.arange(0.1,10,0.1)},
                      cv = 3, scoring="r2",verbose=3
                      )
lasso.fit(X_train,y_train)
coefficients = lasso.best_estimator_.named_steps['model'].coef_
importance = np.abs(coefficients)
features = X_train.columns
new_features = np.array(features)[importance > 0]


# In[29]:


X_train = X_train.filter(new_features, axis=1)
X_test = X_test.filter(new_features, axis=1)
# X_train = X_train.filter(sfs.k_feature_names_, axis=1)
# X_test = X_test.filter(sfs.k_feature_names_, axis=1)


# In[30]:


X_train


# ## Models

# In[31]:


# Normalize data
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.fit_transform(X_test)


# In[32]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
rf = RandomForestRegressor(random_state = 2)
knn = KNeighborsRegressor()
lr = LinearRegression()


# In[33]:


models = [lr,rf,knn]
model_names = ['Linear Regression','Random Forest','KNeighbors Regression']


# In[34]:


validation_scores = {}
cv_scores = {}
finetuned_models = {}
solution_matrix = pd.DataFrame(index=np.arange(len(X_test)))


# In[35]:


model = lr.fit(X_train,y_train)
lr_score  = model.score(X_test,y_test)


# In[36]:


knn_par = {'n_neighbors':[2,4,6,8,10,12,14],"algorithm":['auto','ball_tree']}    
knn_grid_search = GridSearchCV(knn, knn_par, scoring="r2", cv=3)
knn_grid_search.fit(X_train, y_train)
knn_grid_search.best_estimator_ 


# In[37]:


rf_par = {"n_estimators":[15,20,25,30,40,50],"max_depth":[5,10,15,20,30]}
rf_grid_search = GridSearchCV(rf, rf_par, cv=3, scoring="r2")
rf_grid_search.fit(X_train,y_train)
rf_grid_search.best_estimator_


# In[38]:


rf = rf_grid_search.best_estimator_
rf_score = (rf.fit(X_train,y_train)).score(X_test,y_test)


# In[39]:


knn = knn_grid_search.best_estimator_ 
knn_score = (knn.fit(X_train,y_train)).score(X_test,y_test)
knn_score


# In[40]:


result = pd.DataFrame({'model':model_names,
                      'result':[lr_score,rf_score,knn_score]})
result


# In[41]:


real = y_test
predict = rf.predict(X_test)
error = y_test-predict
print(error)


# In[42]:


plt.figure(figsize = (10,7))
plt.scatter(x = real, y = predict, color = 'black')
sns.regplot(x = real,y = predict,color = 'salmon',scatter = False)
plt.xlabel('Real Interest Rate')
plt.ylabel('Predicted Interest Rate')
# The predicted value and true value is almost on the same line 


# ## What I did
# * imported data and replaced missing values by mode
# * deleted some columns that are not relatede to interest rate
# * Visulized data
# * Used lasso regression to do the feature selection which can make the result more accurate and discarded non-related variables(also tried forward feature selection, but the result was similar)
# * Built linear regression,random forest and knn regression models

# ## Improvements
# 
# I would try to handle categorical columns such as emp_title, using encoded rather than just delete the column. For some missing values, try other techniques such as kmeans to make it more accurate. And also try more models and tuning more times to get a better result. Moreover, I would do some research on each columns and find out their relationship in deeper level(maybe more data source in different years, do the time series thing if it has). 

# In[ ]:




