#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# In[3]:


raw_mail_data = pd.read_csv("spam.csv")


# In[4]:


raw_mail_data.shape


# In[5]:


raw_mail_data.head()


# In[6]:


#null values to empty strings


# In[7]:


mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
# mail_data = raw_mail_data.fillna('')


# In[8]:


mail_data = mail_data.rename(columns={"v1": "category", "v2": "text"})


# In[9]:


mail_data.head()


# In[10]:


# label spam mail as 0 and non-spam (ham) as 1 using loc

mail_data.loc[mail_data['category'] == 'spam' , 'category'] = 0
mail_data.loc[mail_data['category'] == 'ham' , 'category'] = 1


# In[11]:


mail_data.head()


# In[12]:


# separate category and text into X and Y

X = mail_data['text']
Y = mail_data['category']


# In[13]:


print(X)
print('........')
print(Y)


# In[14]:


# splitting data into training and testing data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size =0.2, random_state=3)


# In[20]:


# Converting the text data into feature vectors that can be used as input to SVM model using TfidfVectorizer
# Converting the text into lower case letters

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Converting the Y train and test values as integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[24]:


# Training the Support Vector Machine with the training data

model = LinearSVC()
model.fit(X_train_features, Y_train)


# In[25]:


# prediction on training data

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[28]:


print('Accuracy on training data: ',accuracy_on_training_data)


# In[29]:


# prediction on test data

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[30]:


print('Accuracy on test data: ',accuracy_on_test_data)


# In[43]:


# Predicting a mail to be spam or ham

input_mail = ["Hey, just checking in to see how you're doing, hope all is well!"]

# Convert text to feature vector
input_mail_features = feature_extraction.transform(input_mail)

# Prediction
prediction = model.predict(input_mail_features)
print(prediction) # 0 is for spam and 1 for ham

if prediction == 1:
    print('Ham mail')
    
else:
    print("Spam mail")


# In[ ]:




