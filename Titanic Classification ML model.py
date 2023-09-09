#!/usr/bin/env python
# coding: utf-8

# ## TITANIC DATASET

# #### Importing Neccessary Libraries

# In[66]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


# ### Reading training and testing data

# In[3]:


df1 = pd.read_csv('Titanic_train.csv.csv')
df2 = pd.read_csv('Titanic_test.csv.csv')
dfs = [df1,df2]


# In[ ]:


## Merging both dataframes using concat() function of pandas


# In[4]:


df = pd.concat(dfs, ignore_index = True)
df = df.reset_index(drop = True)


# In[5]:


df


# In[6]:


len(df1)


# In[7]:


len(df2)


# ### Performing Visualization For Basic Analysis

# In[67]:


sns.countplot(df1['Survived'])


# In[68]:


sns.countplot(df1['Sex'])


# In[69]:


sns.countplot(df1['Pclass'])


# In[70]:


sns.countplot(df1['SibSp'])


# In[71]:


sns.countplot(df1['Parch'])


# In[72]:


sns.countplot(df1['Embarked'])


# In[73]:


sns.distplot(df1['Age'])


# In[74]:


sns.distplot(df1['Fare'])


# In[76]:


class_fare = df1.pivot_table(index = 'Pclass', values = 'Fare', aggfunc = np.sum)
class_fare.plot(kind = 'bar')
plt.xlabel('Pclass')
plt.ylabel('Fare')
plt.xticks(rotation = 0)
plt.show()


# In[17]:


df.tail()


# ### Data Preprocessing

# In[18]:


df.isnull().sum() ## Checking null values


# In[19]:


df = df.drop(columns = ['Cabin'], axis = 1)


# In[20]:


df


# In[21]:


df['Age'] = df['Age'].fillna(df['Age'].mean())


# In[22]:


df['Fare'] = df['Age'].fillna(df['Fare'].mean())


# In[23]:


df['Embarked'].mode()[0]


# In[24]:


df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


# In[25]:


df.isnull().sum()


# In[26]:


sns.barplot(data = df1, x = 'Pclass', y = 'Fare', hue = 'Survived')


# In[27]:


df.head()


# In[28]:


## Drop unneccessary columns

df = df.drop(columns=['Name','Ticket'], axis = 1)
df.head()


# ### Coverting non-numeric values to numeric values

# In[29]:


from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
cols = ['Sex', 'Embarked']

for col in cols:
    df[col] = le.fit_transform(df[col])
    
df.head()


# #### Splitting Data into training and testing data

# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


train = df.iloc[:len(df1), :]
test = df.iloc[len(df1):, :]


# In[32]:


train


# In[33]:


test


# In[34]:


df2.head()


# In[35]:


train.head()


# In[36]:


test.head()


# In[77]:


# df['column_name'] = df['column_name'].astype(int)
train['Survived'] = train['Survived'].astype(int)


# In[38]:


# Input splits 

X = train.drop(columns = ['PassengerId', 'Survived'], axis=1)
y = train['Survived']


# In[39]:


X.head()


# In[40]:


y


# ## Model Training

# In[41]:


# Classify Column
from sklearn.model_selection import train_test_split, cross_val_score
def classify(model):
    model.fit(X_train,y_train)
    print("Accuracy :", model.score(X_test, y_test))
    score = cross_val_score(model, X, y, cv = 10)
    
    print("CV score :", np.mean(score))


# In[42]:


threshold = 0.5
y_bin = (y >= threshold).astype(int)


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.25, random_state=42)


# In[44]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify(model)


# In[45]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model)


# In[46]:


from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
classify(model)


# In[47]:


from sklearn.ensemble import RandomForestClassifier
classify(model)


# In[48]:


from sklearn.ensemble import BaggingClassifier
classify(model)


# In[49]:


get_ipython().system('pip install catboost')


# In[50]:


import xgboost
from xgboost import XGBClassifier
model = XGBClassifier()
classify(model)


# In[51]:


import lightgbm
from lightgbm import LGBMClassifier
model = LGBMClassifier(num_boost_round =1000)
classify(model)


# In[52]:


import catboost
from catboost import CatBoostClassifier
model = CatBoostClassifier()
classify(model)


# ### Training Model With Entire Data

# In[53]:


# test data split
test.head()


# In[54]:


X_test = test.drop(columns = ['PassengerId', 'Survived'], axis = 1)
X_test


# In[55]:


model = CatBoostClassifier()
model.fit(X,y)


# In[56]:


pred = model.predict(X_test)


# In[57]:


pred ## Predicted Values


# In[59]:


new_dataframe = test[['PassengerId']].copy()
new_dataframe


# In[63]:


new_dataframe['Survived'] = pred
new_dataframe.head(20)


# ## Storing the predicted values in a csv file according to the passenger id

# In[65]:


new_dataframe.to_csv('Final_pred.csv', index=False)


# In[ ]:





# In[ ]:




