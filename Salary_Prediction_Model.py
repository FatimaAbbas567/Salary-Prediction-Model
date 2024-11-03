#!/usr/bin/env python
# coding: utf-8

# In[137]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk


# ##### Read Data File/ CSV File

# In[138]:


sal_data = pd.read_csv('Salary-Prediction-Model-Dataset.csv')
sal_data.head()


# ###### Number of Rows and Columns:

# In[139]:


sal_data.shape


# ##### List of columns:

# In[140]:


sal_data.columns


# ##### Rename Columns:

# In[141]:


sal_data.columns = ['Age', 'Gender', 'Qualification', 'Job Title', 'Experience', 'Salary']


# In[142]:


sal_data.head()


# In[143]:


sal_data.dtypes


# In[144]:


sal_data.info


# In[145]:


sal_data[sal_data.duplicated()]


# In[146]:


sal_data[sal_data.duplicated()].shape


# In[147]:


sal_data1 = sal_data.drop_duplicates(keep='first')


# In[148]:


sal_data1.shape


# In[149]:


sal_data1.isnull().sum()


# In[150]:


sal_data1.dropna(how = 'any', inplace = True)


# In[151]:


sal_data1.shape


# In[152]:


sal_data1.head()


# ### Data Exploration and Visualization:

# #### Statistics of Numerical Columns:

# In[153]:


sal_data1.describe()


# #### Correlation Matrix among Numerical Features:

# In[154]:


corr = sal_data1[['Age', 'Experience', 'Salary']].corr()
corr


# In[155]:


sns.heatmap(corr, annot = True)


# ##### Data Visualization - Bar Chart, Box Plot, Histogram:

# In[157]:


sal_data1['Qualification'].value_counts()


# In[158]:


sal_data1['Qualification'].value_counts().plot(kind = 'bar')


# In[159]:


sal_data1['Job Title'].value_counts()


# In[160]:


sal_data1['Job Title'].unique()


# In[161]:


sal_data1['Gender'].value_counts().plot(kind = 'bar')


# In[162]:


sal_data1.Age.plot(kind = 'hist')


# In[163]:


sal_data1.Age.plot(kind = 'box')


# In[164]:


sal_data1.Salary.plot(kind = 'box')


# In[165]:


sal_data1.Salary.plot(kind = 'hist')


# ### Feature Engineering

# ##### Label Encoding

# In[167]:


from sklearn.preprocessing import LabelEncoder
Label_Encoder = LabelEncoder()


# In[168]:


sal_data1['Gender_Encoder'] = Label_Encoder.fit_transform(sal_data1['Gender'])


# In[169]:


sal_data1['Qualification_Encoder'] = Label_Encoder.fit_transform(sal_data1['Qualification'])


# In[170]:


sal_data1['Job Title_Encoder'] = Label_Encoder.fit_transform(sal_data1['Job Title'])


# ##### Data after Label Encoder:

# In[171]:


sal_data1.head()


# #### Feature Scaling:

# In[172]:


from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()


# In[173]:


sal_data1['Age_scaled'] = std_scaler.fit_transform(sal_data1[['Age']])
sal_data1['Experience_scaled'] = std_scaler.fit_transform(sal_data1[['Experience']])


# #### Data After Scaling: 

# In[174]:


sal_data1.head()


# ### Dependent and Independent features:

# In[175]:


x = sal_data1[['Age_scaled', 'Gender_Encoder', 'Qualification_Encoder', 'Job Title_Encoder', 'Experience_scaled']]
y = sal_data1['Salary']


# In[176]:


x.head()


# #### Splitting the data into Training and Testing:

# In[177]:


from sklearn.model_selection import train_test_split


# In[178]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42 )


# In[179]:


x_train.head()


# In[180]:


x_train.shape


# In[181]:


x_test.shape


# #### Model Development:

# In[182]:


from sklearn.linear_model import LinearRegression


# In[183]:


Linear_regression_model = LinearRegression()


# #### Model Training:

# In[184]:


Linear_regression_model.fit(x_train, y_train)


# #### Model Prediction:

# In[185]:


y_pred_lr = Linear_regression_model.predict(x_test)
y_pred_lr


# In[186]:


df = pd.DataFrame({'y_Actual':y_test, 'y_Predicted':y_pred_lr })
df['Error'] = df['y_Actual'] - df['y_Predicted']
df['abs_error'] = abs(df['Error'])
df


# In[187]:


Mean_absolute_Error = df['abs_error'].mean()
Mean_absolute_Error


# ####  Model Evaluation

# In[188]:


from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error


# #### Model Accuracy

# In[189]:


r2_score(y_test, y_pred_lr)


# In[190]:


print(f'Accuracy of the model = {round( r2_score(y_test, y_pred_lr),4 )*100} %')


# #### Mean Absolute Error:

# In[192]:


round(mean_absolute_error(y_test, y_pred_lr),2)


# In[193]:


print(f"Mean Absolute Error = {round(mean_absolute_error(y_test, y_pred_lr),2)}")


# #### Mean Squared Error:

# In[194]:


mse = round(mean_squared_error(y_test, y_pred_lr),2)
mse


# In[195]:


print(f"Mean Squared Error = {round(mean_squared_error(y_test, y_pred_lr),2)}")


# #### Root Mean Squared Error:

# In[196]:


print('Root Mean Sqaured Error (RMSE) = ',mse**(0.5))


# #### Coefficient:

# In[197]:


Linear_regression_model.coef_


# #### Intercepts:

# In[198]:


Linear_regression_model.intercept_


# #### Customize Predictions:

# In[199]:


sal_data1.head()


# In[200]:


Age1 = std_scaler.transform([[49]])
Age = 5.86448677
Gender = 0
Qualification = 2
Job_Title = 22
Experience_year1 = std_scaler.transform([[15]])
Experience = 0.74415815


# In[201]:


std_scaler.transform([[15]])[0][0]


# In[202]:


Emp_Salary = Linear_regression_model.predict([[Age, Gender, Qualification, Job_Title, Experience]])
Emp_Salary


# In[203]:


print("Salary of that Employee with above Attributes = ", Emp_Salary[0])


# In[ ]:





# In[ ]:




