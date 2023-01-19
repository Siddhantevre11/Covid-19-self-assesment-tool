#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries 
import numpy as np 
import pandas as pd 
import math 
import random 
#import matplotlib.pyplot as plt


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Initilization of 
# Data set

# defining the columns using normal distribution 
# np.random.randint(0, 2, 5) --> To produces 5 numbers. Each no is either 0 or 1 
# column 1  Symptom-I
cold = np.zeros((119711,), dtype=int)
# column 2  Symptom2
cough = np.zeros((119711,), dtype=int)
# column 3 Symptom3
feaver = np.zeros((119711,), dtype=int)
# column 4 Symptom4
breating = np.zeros((119711,), dtype=int)
# Column 5 Symptom 5
cronic = np.zeros((119711,), dtype=int)
# Column 6 
age = np.zeros((119711,), dtype=int) #
#Column 7 The output either Covid-19 or NON-Covid
#disease = np.random.randint(1, 10, 500)
disease = np.zeros(119711, dtype='int')


# In[4]:


#General Symptoms(GS) are 321. It is multiplied with 7. Hence, 321*7=2247
b=0
for i in range(0,2247):  #This is to fill General Symptoms  (GS-->1 in disease column)
    b=b+1
    if(b==1):
        cold[i]=0
        cough[i]=0
        feaver[i]=1
        breating[i]=0
        cronic[i]=0
        age[i]=0  # age=0 indicates that age not considered it may be of any age
        disease[i]=1
    if(b==2):
        cold[i]=0
        cough[i]=1
        feaver[i]=0
        breating[i]=0
        cronic[i]=0
        age[i]=0
        disease[i]=1
    if(b==3):
        cold[i]=0
        cough[i]=1
        feaver[i]=1
        breating[i]=0
        cronic[i]=0
        age[i]=0
        disease[i]=1
    if(b==4):
        cold[i]=1
        cough[i]=0
        feaver[i]=0
        breating[i]=0
        cronic[i]=0
        age[i]=0
        disease[i]=1
    if(b==5):
        cold[i]=1
        cough[i]=0
        feaver[i]=1
        breating[i]=0
        cronic[i]=0
        age[i]=0
        disease[i]=1
    if(b==6):
        cold[i]=1
        cough[i]=1
        feaver[i]=0
        breating[i]=0
        cronic[i]=0
        age[i]=0
        disease[i]=1
    if(b==7):
        cold[i]=1
        cough[i]=1
        feaver[i]=1
        breating[i]=0
        cronic[i]=0
        age[i]=0
        disease[i]=1
    if(b==7):
        b=1
    


# In[5]:


# HR1 is 116. It is multiplied with 16. Hence, 116*16=1856. It is from 2247+1856=4103
b=0
for i in range(2247,4103):  #This is to fill   (HR1-->2 in disease column) or Low Risk
    b=b+1
    if(b==1):
        cold[i]=0
        cough[i]=0
        feaver[i]=0
        breating[i]=1
        cronic[i]=0
        age[i]=1        # age=1 indicates that age > 60
        disease[i]=2
    if(b==2):
        cold[i]=0
        cough[i]=0
        feaver[i]=1
        breating[i]=0
        cronic[i]=0
        age[i]=1
        disease[i]=2
    if(b==3):
        cold[i]=0
        cough[i]=1
        feaver[i]=0
        breating[i]=1
        cronic[i]=0
        age[i]=1
        disease[i]=2
    if(b==4):
        cold[i]=0
        cough[i]=1
        feaver[i]=0
        breating[i]=0
        cronic[i]=0
        age[i]=1
        disease[i]=2
    if(b==5):
        cold[i]=0
        cough[i]=1
        feaver[i]=0
        breating[i]=1
        cronic[i]=0
        age[i]=1
        disease[i]=2
    if(b==6):
        cold[i]=0
        cough[i]=1
        feaver[i]=1
        breating[i]=0
        cronic[i]=0
        age[i]=1
        disease[i]=2
    if(b==7):
        cold[i]=0
        cough[i]=1
        feaver[i]=1
        breating[i]=1
        cronic[i]=0
        age[i]=1
        disease[i]=2
    if(b==8):
        cold[i]=1
        cough[i]=0
        feaver[i]=0
        breating[i]=0
        cronic[i]=0
        age[i]=1
        disease[i]=2
    if(b==9):
        cold[i]=1
        cough[i]=0
        feaver[i]=0
        breating[i]=1
        cronic[i]=0
        age[i]=1
        disease[i]=2
    if(b==10):
        cold[i]=1
        cough[i]=0
        feaver[i]=1
        breating[i]=0
        cronic[i]=0
        age[i]=1
        disease[i]=2
    if(b==11):
        cold[i]=1
        cough[i]=0
        feaver[i]=1
        breating[i]=1
        cronic[i]=0
        age[i]=1
        disease[i]=2
    if(b==12):
        cold[i]=1
        cough[i]=1
        feaver[i]=0
        breating[i]=0
        cronic[i]=0
        age[i]=1
        disease[i]=2
    if(b==13):
        cold[i]=1
        cough[i]=1
        feaver[i]=0
        breating[i]=1
        cronic[i]=0
        age[i]=1
        disease[i]=2
    if(b==14):
        cold[i]=1
        cough[i]=1
        feaver[i]=1
        breating[i]=0
        cronic[i]=0
        age[i]=1
        disease[i]=2
    if(b==15):
        cold[i]=1
        cough[i]=1
        feaver[i]=1
        breating[i]=1
        cronic[i]=0
        age[i]=1
        disease[i]=2
    if(b==15):
        b=1
    


# In[6]:


# HR2 is 67. It is multiplied with 16. Hence, 67*16=1072. It is from 4103+1072=5175
b=0
for i in range(4103,5175):  #This is to fill   (HR2-->3 in disease column) or Medium Risk
    b=b+1
    if(b==1):
        cold[i]=0
        cough[i]=0
        feaver[i]=0
        breating[i]=1
        cronic[i]=1
        age[i]=2           #2 indicates that the age is <60
        disease[i]=3
    if(b==2):
        cold[i]=0
        cough[i]=0
        feaver[i]=1
        breating[i]=0
        cronic[i]=1
        age[i]=2
        disease[i]=3
    if(b==3):
        cold[i]=0
        cough[i]=1
        feaver[i]=0
        breating[i]=1
        cronic[i]=1
        age[i]=2
        disease[i]=3
    if(b==4):
        cold[i]=0
        cough[i]=1
        feaver[i]=0
        breating[i]=0
        cronic[i]=1
        age[i]=2
        disease[i]=3
    if(b==5):
        cold[i]=0
        cough[i]=1
        feaver[i]=0
        breating[i]=1
        cronic[i]=1
        age[i]=2
        disease[i]=3
    if(b==6):
        cold[i]=0
        cough[i]=1
        feaver[i]=1
        breating[i]=0
        cronic[i]=1
        age[i]=2
        disease[i]=3
    if(b==7):
        cold[i]=0
        cough[i]=1
        feaver[i]=1
        breating[i]=1
        cronic[i]=1
        age[i]=2
        disease[i]=3
    if(b==8):
        cold[i]=1
        cough[i]=0
        feaver[i]=0
        breating[i]=0
        cronic[i]=1
        age[i]=2
        disease[i]=3
    if(b==9):
        cold[i]=1
        cough[i]=0
        feaver[i]=0
        breating[i]=1
        cronic[i]=1
        age[i]=2
        disease[i]=3
    if(b==10):
        cold[i]=1
        cough[i]=0
        feaver[i]=1
        breating[i]=0
        cronic[i]=1
        age[i]=2
        disease[i]=3
    if(b==11):
        cold[i]=1
        cough[i]=0
        feaver[i]=1
        breating[i]=1
        cronic[i]=1
        age[i]=2
        disease[i]=3
    if(b==12):
        cold[i]=1
        cough[i]=1
        feaver[i]=0
        breating[i]=0
        cronic[i]=1
        age[i]=2
        disease[i]=3
    if(b==13):
        cold[i]=1
        cough[i]=1
        feaver[i]=0
        breating[i]=1
        cronic[i]=1
        age[i]=2
        disease[i]=3
    if(b==14):
        cold[i]=1
        cough[i]=1
        feaver[i]=1
        breating[i]=0
        cronic[i]=1
        age[i]=2
        disease[i]=3
    if(b==15):
        cold[i]=1
        cough[i]=1
        feaver[i]=1
        breating[i]=1
        cronic[i]=1
        age[i]=2
        disease[i]=3
    if(b==15):
        b=1


# In[7]:


# HR3 is 77. It is multiplied with 16. Hence, 77*16=1232. It is from 5175+1232=6407
b=0
for i in range(5175,6407):  #This is to fill   (HR3-->4 in disease column)
    b=b+1
    if(b==1):
        cold[i]=0
        cough[i]=0
        feaver[i]=0
        breating[i]=1
        cronic[i]=1
        age[i]=1
        disease[i]=4
    if(b==2):
        cold[i]=0
        cough[i]=0
        feaver[i]=1
        breating[i]=0
        cronic[i]=1
        age[i]=1
        disease[i]=4
    if(b==3):
        cold[i]=0
        cough[i]=0
        feaver[i]=1
        breating[i]=1
        cronic[i]=1
        age[i]=1
        disease[i]=4
    if(b==4):
        cold[i]=0
        cough[i]=1
        feaver[i]=0
        breating[i]=0
        cronic[i]=1
        age[i]=1
        disease[i]=4
    if(b==5):
        cold[i]=0
        cough[i]=1
        feaver[i]=0
        breating[i]=1
        cronic[i]=1
        age[i]=1
        disease[i]=4
    if(b==6):
        cold[i]=0
        cough[i]=1
        feaver[i]=1
        breating[i]=0
        cronic[i]=1
        age[i]=1
        disease[i]=4
    if(b==7):
        cold[i]=0
        cough[i]=1
        feaver[i]=1
        breating[i]=1
        cronic[i]=1
        age[i]=1
        disease[i]=4
    if(b==8):
        cold[i]=1
        cough[i]=0
        feaver[i]=0
        breating[i]=0
        cronic[i]=1
        age[i]=1
        disease[i]=4
    if(b==9):
        cold[i]=1
        cough[i]=0
        feaver[i]=0
        breating[i]=1
        cronic[i]=1
        age[i]=1
        disease[i]=4
    if(b==10):
        cold[i]=1
        cough[i]=0
        feaver[i]=1
        breating[i]=0
        cronic[i]=1
        age[i]=1
        disease[i]=4
    if(b==11):
        cold[i]=1
        cough[i]=0
        feaver[i]=1
        breating[i]=1
        cronic[i]=1
        age[i]=1
        disease[i]=4
    if(b==12):
        cold[i]=1
        cough[i]=1
        feaver[i]=0
        breating[i]=0
        cronic[i]=1
        age[i]=1
        disease[i]=4
    if(b==13):
        cold[i]=1
        cough[i]=1
        feaver[i]=0
        breating[i]=1
        cronic[i]=1
        age[i]=1
        disease[i]=4
    if(b==14):
        cold[i]=1
        cough[i]=1
        feaver[i]=1
        breating[i]=0
        cronic[i]=1
        age[i]=1
        disease[i]=4
    if(b==15):
        cold[i]=1
        cough[i]=1
        feaver[i]=1
        breating[i]=1
        cronic[i]=1
        age[i]=1
        disease[i]=4
    if(b==15):
        b=1
    


# In[8]:


#x.head()
data = pd.DataFrame()
data['cold']=cold
data['cough']=cough
data['feaver']=feaver
data['breating']=breating
data['cronic']=cronic
data['age']=age
data['disease']=disease
#data
#data.head()
    
#c=np.array(data.loc[3, ['cold', 'cough', 'feaver']])
#c.any()
#c.all()
data.to_csv('ds-covid-new34733-119711.csv', index=False, header=True)


# In[9]:


df = pd.read_csv("ds-covid-new34733-119711.csv")
df.sample(5)


# # Accuracy of algorithms before balancing a dataset
# 
# 
# Add corresponding code in below cells

# In[10]:


df.disease.value_counts() #disease is a field name and it is dependent or output variable 


# In[11]:


df.dtypes


# In[12]:


import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix , classification_report
from tensorflow_addons import losses
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score


# In[13]:


X = df.drop('disease',axis='columns')
y = df['disease']


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=0)


# In[15]:


y_train.value_counts()


# In[16]:


y_test.value_counts()


# In[17]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train, y_train)
train_acc=lr.predict(X_train)
train_score=metrics.accuracy_score(y_train, train_acc)
y_pred_lr=lr.predict(X_test)
test_scores=metrics.accuracy_score(y_test, y_pred_lr)
print("\n The Linear Regression: Accuracy of Training data:",train_score)
print("\n The Linear Regression: Accuracy of Testine data:",test_scores)
lr_cm1=confusion_matrix(y_test, y_pred_lr)
lr_cm1
lr_cr = classification_report(y_test, y_pred_lr)
print("Classification Report:\n",lr_cr)


# In[18]:


#from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import GaussianNB
naive=GaussianNB()
naive.fit(X_train, y_train)
train_naive=naive.predict(X_train)
naive_y_pred=naive.predict(X_test)
train_naive_score=metrics.accuracy_score(y_train, train_naive)
scores=metrics.accuracy_score(y_test, naive_y_pred)
print("\n The naive_bayes: Accuracy of Training data:",train_naive_score)
print("\n The naive_bayes: Accuracy of Testing data:",scores)
naive_cm1=confusion_matrix(y_test, naive_y_pred)
naive_cm1
naive_cr = classification_report(y_test, naive_y_pred)
print("Classification Report:\n",naive_cr)


# In[19]:


from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier(loss='modified_huber',shuffle=True,random_state=101)
sgd.fit(X_train, y_train)
y_pred_sgd=sgd.predict(X_test)
scores=metrics.accuracy_score(y_test, y_pred_sgd)
SGDTrain=sgd.predict(X_train)
print("\n The naive_bayes: Accuracy of Training data:",accuracy_score(y_train,SGDTrain))
print("\n The naive_bayes: Accuracy of Testing data:",scores)
sgd_cm1=confusion_matrix(y_test, y_pred_sgd)
sgd_cm1
sgd_cr = classification_report(y_test, y_pred_sgd)
print("Classification Report:\n",sgd_cr)


# In[20]:


#Code to find best K Value

#Importing KNN Classifier
from sklearn.neighbors import KNeighborsClassifier


# try K=1 through K=25 and record testing accuracy

#k_range = range(1, 26)

# We can create Python dictionary using [] or dict()
scores = []
k=5
# We use a loop through the range 1 to 26
# We append the scores in the dictionary

#for k in k_range:
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
knntrain=knn.predict(X_train)
y_pred_knn = knn.predict(X_test)
scores.append(metrics.accuracy_score(y_test, y_pred_knn))
print("\n KNN: Accuracy of Training data:",accuracy_score(y_train,knntrain))
print("\n KNN: Accuracy of Testing data:",scores)

knn_cr = classification_report(y_test, y_pred_knn)
print("Classification Report:\n",knn_cr)


# In[21]:


# cold	cough	feaver	breating	cronic	age
print(knn.predict([[1,1,0,0,0,0]]))# Prediction of a person with some diseases.
#The output shows it is "1" Means GS,2 means HR1, 3 means HR2, and 4 means HR3.
print(knn.predict([[1,1,0,1,0,0]]))
print(knn.predict([[1,1,0,0,0,1]]))
print(knn.predict([[1,1,0,1,1,0]]))
print(knn.predict([[1,1,0,1,1,1]]))
print(knn.predict([[1,1,0,0,1,1]]))
print(knn.predict([[0,0,1,0,1,0]]))
print(knn.predict([[0,0,0,0,0,0]]))
print(knn.predict([[1,1,1,1,1,1]]))


# In[22]:


# confusion matrix
knncm1=confusion_matrix(y_test, y_pred_knn)
knncm1


# In[23]:


from sklearn.tree import DecisionTreeClassifier as dtc
# gini model
# -------------------------------------
# Model 1) DT with gini index criteria
clf_gini = dtc(criterion = "gini", random_state = 100, max_depth=3,
               min_samples_leaf=5).fit(X_train, y_train)
print(clf_gini)


# In[24]:


from sklearn.metrics import accuracy_score, f1_score
pred_gini = clf_gini.predict(X_test)
DTTrain=clf_gini.predict(X_train)
print("Gini Testing Accuracy is ", accuracy_score(y_test,pred_gini)*100)
print("Gini Training Accuracy is ", accuracy_score(y_train,DTTrain)*100)


# In[25]:


# confusion matrix
cm1=confusion_matrix(y_test, pred_gini)
cm1


# In[26]:


from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(max_depth=10,random_state=101,max_features=None,min_samples_leaf=15)
dtree.fit(X_train, y_train)
y_pred_dtree=dtree.predict(X_test)
dtreetrain=dtree.predict(X_train)
scores=metrics.accuracy_score(y_test, y_pred_dtree)
print("\n DecisionTree: Accuracy of Testing data:",scores)
print("\n DecisionTree: Accuracy of Training data:",accuracy_score(y_train, dtreetrain))
dtree_cm1=confusion_matrix(y_test, y_pred_dtree)
dtree_cm1

dtree_cr = classification_report(y_test, y_pred_dtree)
print("Classification Report:\n",dtree_cr)


# In[27]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfm=RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=-1,random_state=101,max_features=None,min_samples_leaf=30)
rfm.fit(X_train, y_train)
y_pred_rfm=rfm.predict(X_test)
rfmtrain=rfm.predict(X_train)
rfmscores=metrics.accuracy_score(y_test, y_pred_rfm)
print("\n RandomForestClassifier: Accuracy of Testing data:",rfmscores)
print("\n RandomForestClassifier: Accuracy of Training data:",accuracy_score(y_train, rfmtrain))

rfm_cm1=confusion_matrix(y_test, y_pred_rfm)
rfm_cm1


rfm_cr = classification_report(y_test, y_pred_rfm)
print("Classification Report:\n",rfm_cr)


# In[28]:


from sklearn.svm import SVC
SVM=SVC(kernel="linear",C=0.025,random_state=101)
SVM.fit(X_train, y_train)
y_pred_SVM=SVM.predict(X_test)
SVMtrain=SVM.predict(X_train)
SVMscores=metrics.accuracy_score(y_test, y_pred_SVM)
print("\n DecisionTree: Accuracy of Testing data:",SVMscores)
print("\n DecisionTree: Accuracy of Training data:",accuracy_score(y_train, SVMtrain))

SVM_cm1=confusion_matrix(y_test, y_pred_SVM)
SVM_cm1

svm_cr = classification_report(y_test, y_pred_SVM)
print("Classification Report:\n",svm_cr)


# ### Different Methods for getting balanced dataset
# 
# **for getting balanced data set.**
# 
# 1. SMOTE -- other forms of SMOTE--such as SMOTE("minority") and SMOTE(sampling_strategy=strategy)
# 2. ADASYN
# 3. SMOTETomek
# 4. SMOTEENN
# 5. RandomUnderSampler
# 
# To install imbalanced-learn library use **pip install imbalanced-learn** command
# 
# The **stratify** option in train_test_split method gives 1.0 Accuracy. Which is uses as follows:
# 
# from sklearn.model_selection import train_test_split
# X_sm_train, X_sm_test, y_sm_train, y_sm_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=15, stratify=y_sm)
# 
# Where as **sometimes** without **stratify** option gives less accuracy for some balanced dataset methods. 
# 
# from sklearn.model_selection import train_test_split
# X_sm_train, X_sm_test, y_sm_train, y_sm_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=15)
# 
# **The below code executes The RandomUndersampling followed by SMOTE to obtain balanced data set.**

# In[29]:


#!pip install imbalanced-learn


# In[32]:


from imblearn.under_sampling import RandomUnderSampler
#under = RandomUnderSampler()
under = RandomUnderSampler(sampling_strategy='majority')
X_r, y_r=under.fit_resample(X,y)
y_r.value_counts()


# In[33]:


# RandomUnderSampler followed by either SMOTE or ADASYN
#Python code for SMOTE algorithm
from imblearn.over_sampling import SMOTE
smote=SMOTE()
X_rsm,y_rsm=smote.fit_resample(X_r,y_r)
y_rsm.value_counts()


# In[34]:


from sklearn.model_selection import train_test_split
X_rsm_train, X_rsm_test, y_rsm_train, y_rsm_test = train_test_split(X_r, y_r, test_size=0.3, random_state=15)
y_rsm_test.value_counts()


# In[35]:


y_rsm_train.value_counts()


# In[36]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_rsm_train, y_rsm_train)
lr_y_pred=lr.predict(X_rsm_test)
lrscores=metrics.accuracy_score(y_rsm_test, lr_y_pred)
lrTrain=lr.predict(X_rsm_train)
print("\n The naive_bayes: Accuracy of Training data:",accuracy_score(y_rsm_train,lrTrain))
print("\n The naive_bayes: Accuracy of Testing data:",lrscores)
lr_cm1=confusion_matrix(y_rsm_test, lr_y_pred)
lr_cm1

lr_cr = classification_report(y_rsm_test, lr_y_pred)
print("Classification Report:\n",lr_cr)


# In[37]:


#from sklearn.native_bayes import GaussianNB
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_rsm_train, y_rsm_train)
nb_y_pred=nb.predict(X_rsm_test)
nbscores=metrics.accuracy_score(y_rsm_test, nb_y_pred)
nbTrain=nb.predict(X_rsm_train)
print("\n The naive_bayes: Accuracy of Training data:",accuracy_score(y_rsm_train,nbTrain))
print("\n The naive_bayes: Accuracy of Testing data:",nbscores)
Gaussian_cm1=confusion_matrix(y_rsm_test, nb_y_pred)
Gaussian_cm1

nb_cr = classification_report(y_rsm_test, nb_y_pred)
print("Classification Report:\n",nb_cr)


# In[38]:


from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier(loss='modified_huber',shuffle=True,random_state=101)
sgd.fit(X_rsm_train, y_rsm_train)
y_pred_sgd=sgd.predict(X_rsm_test)
sgdscores=metrics.accuracy_score(y_rsm_test, y_pred_sgd)
sgdTrain=sgd.predict(X_rsm_train)
print("\n The naive_bayes: Accuracy of Training data:",accuracy_score(y_rsm_train,sgdTrain))
print("\n The naive_bayes: Accuracy of Testing data:",sgdscores)
sgd_cm1=confusion_matrix(y_rsm_test, y_pred_sgd)
sgd_cm1

sgd_cr = classification_report(y_rsm_test, y_pred_sgd)
print("Classification Report:\n",sgd_cr)


# In[51]:


knn_rsm = KNeighborsClassifier(n_neighbors=5)
knn_rsm.fit(X_rsm_train, y_rsm_train)
y_rsm_pred = knn_rsm.predict(X_rsm_test)
knnscores=metrics.accuracy_score(y_rsm_test, y_rsm_pred)
knnTrain=knn_rsm.predict(X_rsm_train)
print("\n The naive_bayes: Accuracy of Training data:",accuracy_score(y_rsm_train,knnTrain))
print("\n The naive_bayes: Accuracy of Testing data:",knnscores)

knn_cr = classification_report(y_rsm_test, y_rsm_pred)
print("Classification Report:\n",knn_cr)


# In[40]:


# cold	cough	feaver	breating	cronic	age
print(knn.predict([[1,1,0,0,0,0]]))# Prediction of a person with some diseases.
#The output shows it is "1" Means GS,2 means HR1, 3 means HR2, and 4 means HR3.
print(knn_rsm.predict([[1,1,0,1,0,0]]))
print(knn_rsm.predict([[1,1,0,0,0,1]]))
print(knn_rsm.predict([[1,1,0,1,1,0]]))
print(knn_rsm.predict([[1,1,0,1,1,1]]))
print(knn_rsm.predict([[1,1,0,0,1,1]]))
print(knn_rsm.predict([[0,0,1,0,1,0]]))
print(knn_rsm.predict([[0,0,0,0,0,0]]))
print(knn_rsm.predict([[1,1,1,1,1,1]]))


# In[41]:


knn_rsm_cm1=confusion_matrix(y_rsm_test, y_rsm_pred)
knn_rsm_cm1


# In[42]:


# gini model
# -------------------------------------
# Model 1) DT with gini index criteria
rsm_clf_gini = dtc(criterion = "gini", random_state = 100, max_depth=3,
               min_samples_leaf=5).fit(X_rsm_train, y_rsm_train)
print(rsm_clf_gini)


# In[43]:


#from sklearn.metrics import accuracy_score, f1_score
rsm_pred_gini = rsm_clf_gini.predict(X_rsm_test)
#smote_pred_gini.value_counts()
#print("Gini Accuracy is ", accuracy_score(y_rsm_test,rsm_pred_gini)*100)
giniTrain=rsm_clf_gini.predict(X_rsm_train)
print("\n The Gini: Accuracy of Training data:",accuracy_score(y_rsm_train,giniTrain))
print("\n The Gini: Accuracy of Testing data:",accuracy_score(y_rsm_test,rsm_pred_gini)*100)


# In[44]:


# confusion matrix
gini_rsm_cm1=confusion_matrix(y_rsm_test, rsm_pred_gini)
gini_rsm_cm1


# In[45]:


from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(max_depth=10,random_state=101,max_features=None,min_samples_leaf=15)
dtree.fit(X_rsm_train, y_rsm_train)
y_pred_dtree=dtree.predict(X_rsm_test)
dtreescores=metrics.accuracy_score(y_rsm_test, y_pred_dtree)
dtreeTrain=dtree.predict(X_rsm_train)
print("\n The naive_bayes: Accuracy of Training data:",accuracy_score(y_rsm_train,dtreeTrain))
print("\n The naive_bayes: Accuracy of Testing data:",dtreescores)
dtree_cm1=confusion_matrix(y_rsm_test, y_pred_dtree)
dtree_cm1

dtree_cr = classification_report(y_rsm_test, y_pred_dtree)
print("Classification Report:\n",dtree_cr)


# In[46]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfm=RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=-1,random_state=101,max_features=None,min_samples_leaf=30)
rfm.fit(X_rsm_train, y_rsm_train)
y_pred_rfm=rfm.predict(X_rsm_test)
rfmscores=metrics.accuracy_score(y_rsm_test, y_pred_rfm)
rfmTrain=rfm.predict(X_rsm_train)
print("\n The naive_bayes: Accuracy of Training data:",accuracy_score(y_rsm_train,rfmTrain))
print("\n The naive_bayes: Accuracy of Testing data:",rfmscores)
rfm_cm1=confusion_matrix(y_rsm_test, y_pred_rfm)
rfm_cm1

rfm_cr = classification_report(y_rsm_test, y_pred_rfm)
print("Classification Report:\n",rfm_cr)


# In[47]:


from sklearn.svm import SVC
SVM=SVC(kernel="linear",C=0.025,random_state=101)
SVM.fit(X_rsm_train, y_rsm_train)
y_pred_SVM=SVM.predict(X_rsm_test)
svmscores=metrics.accuracy_score(y_rsm_test, y_pred_SVM)
svmTrain=SVM.predict(X_rsm_train)
print("\n The naive_bayes: Accuracy of Training data:",accuracy_score(y_rsm_train,svmTrain))
print("\n The naive_bayes: Accuracy of Testing data:",svmscores)


SVM_cm1=confusion_matrix(y_rsm_test, y_pred_SVM)
print(SVM_cm1)

SVM_cr = classification_report(y_rsm_test, y_pred_SVM)
print("Classification Report:\n",SVM_cr)

