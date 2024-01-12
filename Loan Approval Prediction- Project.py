#!/usr/bin/env python
# coding: utf-8

# # Loan Approval Prediction- Project

# In[2]:


import pandas as pd #use to Load Dataset.
import numpy as np# use to Perform mathematical and Logical Operations.
import matplotlib.pyplot as plt #To visualize the Data Features i.e. barplot.
import seaborn as sns #To see The correlation between features using heatmap.
from sklearn import svm


# In[3]:


#Load Dataset
df=pd.read_csv(r'Loan.csv')
df.head()


# In[4]:


#Use head() to print First 5 rows.
df.head()


# In[5]:


#print the Size of Shape
df.shape


# In[6]:


#info() method print the information on Dataframe.
df.info()


# # Data Cleaning

# In[7]:


#Using isnull() Function identifies the missing value  in the dataset.
df.isnull().sum()


# In[8]:


#The Histogram plots the no. of. observation for each range of values of teh numeric Feature.
#These ranges are called as Bins.


# In[9]:


df['loanAmount_log']=np.log(df['LoanAmount'])
df['loanAmount_log'].hist(bins=20)


# In[10]:


df.isnull().sum()


# In[11]:


df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']
df['TotalIncome_log']=np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)


# In[12]:


#using fillna() Function Handle Missing Values.
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)

df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
df.loanAmount_log = df.loanAmount_log.fillna(df.loanAmount_log.mean())


df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)

df.isnull().sum()


# In[13]:


#iloc() Function provides a straight forward way to access specific rows & columns in a pandas Dataframe using integer 
#based indexing.
#X variable is independant & y is dependent  variable.
X=df.iloc[:,np.r_[1:5,9:11,13:15]].values
y=df.iloc[:,12].values

X


# In[14]:


y


# In[15]:


print("per of missing gender is %2f%%" %((df['Gender'].isnull().sum()/df.shape[0]*100)))


# In[16]:


print("number of people who take loan as group by gender:")
print(df['Gender'].value_counts())
sns.countplot(x='Gender', data=df,palette= 'Set1')


# In[17]:


print("number of people who take loan as group by marital status:")
print(df['Married'].value_counts())
sns.countplot(x='Married', data=df,palette= 'Set1')


# In[18]:


print("number of people who take loan as group by dependents:")
print(df['Dependents'].value_counts())
sns.countplot(x='Dependents', data=df,palette= 'Set1')


# In[19]:


print("number of people who take loan as group by self employed:")
print(df['Self_Employed'].value_counts())
sns.countplot(x='Self_Employed', data=df,palette= 'Set1')


# In[20]:


print("number of people who take loan as group by self employed:")
print(df['LoanAmount'].value_counts())
sns.countplot(x='LoanAmount', data=df,palette= 'Set1')


# In[21]:


print("number of people who take loan as group by Credit History:")
print(df['Credit_History'].value_counts())
sns.countplot(x='Credit_History', data=df,palette= 'Set1')


# # Spliting the Dataset

# In[22]:


from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)


# In[23]:


print(X_train)


# # Label Encoding

# In[24]:


from sklearn.preprocessing import LabelEncoder
LabelEncoder_X = LabelEncoder()


# In[25]:


for i in range(0,5):
    X_train[:,i]=LabelEncoder_X.fit_transform(X_train[:,i])
    X_train[:,7]=LabelEncoder_X.fit_transform(X_train[:,7])
X_train


# In[26]:


LabelEncoder_y = LabelEncoder()
y_train = LabelEncoder_y.fit_transform(y_train)
y_train


# In[27]:


for i in range(0,5):
    X_test[:,i]=LabelEncoder_X.fit_transform(X_test[:,i])
    X_test[:,7]=LabelEncoder_X.fit_transform(X_test[:,7])
X_test


# In[28]:


Labelencoder_y= LabelEncoder()

y_test = LabelEncoder_y.fit_transform(y_test)

y_test


# In[29]:


df


# # Feature Scaling

# In[27]:


from sklearn.preprocessing import StandardScaler

sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# # Using Random Forest Algorithm

# In[28]:


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train,y_train)


# In[29]:


from sklearn import metrics
y_pred = rf_clf.predict(X_test)
print("acc of random forest clf is",metrics.accuracy_score(y_pred,y_test))

y_pred


# # Using SVM

# In[48]:


from sklearn.svm import SVC


# In[49]:


classification_rbf = SVC(kernel='rbf')
classification_rbf.fit(X_train, y_train)


# In[50]:


classification_rbf.score(X_test, y_test)


# In[51]:


y_pred


# # Using Naive Byes

# In[30]:


from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train,y_train)


# In[31]:


y_pred = nb_classifier.predict(X_test)
print("acc of guassianNB is % ",metrics.accuracy_score(y_pred,y_test))


# In[32]:


y_pred


# In[36]:


from sklearn.metrics import confusion_matrix,classification_report
conf_matrix = confusion_matrix(y_test,y_pred)


# In[37]:


target={'Y':0,'N':1}


# In[38]:


import seaborn as sns
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues",xticklabels=target,yticklabels=target)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[39]:


print("CLassification Report : ")
print(classification_report(y_test,y_pred,target_names=target))


# In[44]:


from sklearn.model_selection import cross_val_score, KFold

# Create a k-fold cross-validator
kf = KFold(n_splits=3, shuffle=True, random_state=51)

# Perform k-fold cross-validation for linear SVM
scores = cross_val_score(classification_rbf, X_train, y_train, cv=kf)

print("Cross-validation scores:", scores)


# In[45]:


score = np.mean(scores)
print("Mean Accuracy:",score)


# In[50]:


from sklearn.ensemble import BaggingClassifier
bagging_model = BaggingClassifier(base_estimator='deprecated', n_estimators=1000, random_state=42)
bagging_model.fit(X_train, y_train)
bagging_accuracy = bagging_model.score(X_test, y_test)
print("Bagging Model Accuracy:", bagging_accuracy)


# In[ ]:





# In[ ]:




