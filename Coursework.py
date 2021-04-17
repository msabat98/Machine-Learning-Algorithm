
# coding: utf-8

# In[2]:

#report 
import pandas as pd
df = pd.read_csv("student.csv")
print df.head(33)


# In[3]:

print df.dtypes


# In[ ]:




# In[6]:

duplicate_rows_df = df[df.duplicated()]
print "number of duplicate rows: ", duplicate_rows_df.shape


# In[7]:

df = df.drop_duplicates()

print df.count()


# In[8]:

print(df.isnull().sum())


# In[9]:

print df.describe()


# In[13]:




# In[26]:

import pandas as pd
from sklearn.tree import DecisionTreeClassifier 

from sklearn.model_selection import train_test_split 

from sklearn import metrics


# In[27]:

df2 = pd.read_csv("student.csv")
print df2.shape

df2.head()


# In[28]:

X = df2.drop(['G3'], axis=1)

X.head()


# In[29]:

y = df2.G3

print y[0:5]


# In[51]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print X_train.shape 


# In[57]:

clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)


# In[56]:

y_pred = clf.predict(X_test)

print("Accuracy", metrics.accuracy_score(y_test, y_pred))


# In[ ]:




# In[ ]:



