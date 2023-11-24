# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 11:43:59 2023

@author: RAJARSHI RAY
@Internship : CodSoft - Task - 3
"""

# Importing libraries and modules

import nltk
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset input

df = pd.read_csv('D:/CodSoft Internship/3. TASK_3 Email spam detection/spam.csv', encoding='ISO-8859-1')
df.columns

# Renaming v1 and v2 columns to type and message and dropping the other columns

df.rename(columns={'v1' : 'mail_type' , 'v2' : 'message'}, inplace = True)
df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],inplace=True)

df.head()

df.shape

# Dealing with null and duplicates

df.describe()  # dataset decription
df.info()      # dataset information

df.isnull().sum()   # null check
df.duplicated().sum()  # duplicate value check

df.drop_duplicates(keep='first', inplace=True)  # keeping only first unique and removing its duplicate
df.reset_index(drop=True, inplace=True)
df.shape

# Dataset operations

df.mail_type.value_counts()  # checking number of data for each type

### Inference : dataset is imbalanced due to un equal distribution of values in categories
### Solution : we will stratify the target column in train test split

## Dictionary of target column numerical assign

mail_type_labels = {
    'ham' : 0 ,
    'spam' : 1
    }


## Mapping the dictionary to target column

df['mail_type'] = df['mail_type'].map(mail_type_labels)
df.head()


# Text processing using NLP

stemmer = PorterStemmer()

## Custom function to process data

def nlp_processing(df):
    
    # Final list to store processed words
    
    text_list = []
    
    for i in range(0,len(df)):
        
        try:
            # Regular expression to remove everything except alphabets
            
            text_processing = re.sub('[^a-zA-Z]',' ',df['message'][i])
            
            # convert to lowercase for simplicity
            
            text_processing = text_processing.lower()
            
            # split into words
            
            text_processing = text_processing.split()
            
            # stem the words for NLP which are not stopwords
            
            text_processing = [stemmer.stem(words) for words in text_processing if not words in stopwords.words('english')]
        
            # assemble the words
            
            text_processing = ' '.join(text_processing)
            
            # combine into list
            
            text_list.append(text_processing)
        
        except KeyError:
            continue
        
    return text_list
    

# call function on dataset

text_data_train_file = nlp_processing(df)

# Vectorizer to create numerical vectors of processed text

def vectorizer(text_processed_list, training_vectorizer=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    if training_vectorizer is None:
        vectorizer = TfidfVectorizer()
        X_col = vectorizer.fit_transform(text_processed_list).toarray()
    else:
        X_col = training_vectorizer.transform(text_processed_list).toarray()
    
    return X_col, vectorizer if training_vectorizer is None else training_vectorizer


X_training_set, vectorizer_for_training = vectorizer(text_data_train_file)
X_training_set



# EDA on data
def count_plot(df):
    sns.countplot(x=df['mail_type'],data=df)
    plt.xticks(ticks=[0,1] , labels=['ham','spam'])
    plt.title('Email category counts')
    
    
count_plot(df)

# Separating the X and y

X = X_training_set
y = df['mail_type']

print(X.shape,y.shape)

# Train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 1,stratify=y)

# Multi model training
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

model_list = [MultinomialNB(),LogisticRegression(),SVC(kernel = 'linear')]

for model in model_list:
    model.fit(X_train,y_train)
    pred_score = model.predict(X_test)
    accuracy = accuracy_score(y_test,pred_score)
    print(f"Model used : {model} and accuracy score : {round(accuracy*100,2)} %")
    

# Getting a model

model_production = SVC(kernel='linear')
model_production.fit(X_train,y_train)
pred_score_svm = model_production.predict(X_test)
pred_score_svm

## Creating a prediction system

df_test = pd.read_csv('D:/CodSoft Internship/3. TASK_3 Email spam detection/test.csv', encoding='ISO-8859-1')
df_test.head()
df_test.columns
df_test.rename(columns={'ï»¿message' : 'message'}, inplace = True)


testing_data_file = nlp_processing(df_test)
X_testing_set,vectorizer_for_testing = vectorizer(testing_data_file, training_vectorizer=vectorizer_for_training)
X_testing_set.shape

prediction = model_production.predict(X_testing_set)
prediction


# ...

# Issue Faced:
# During testing, the original code was encountering a "ValueError: X has 18 features, but SVC is expecting 6221 features as input."
# This error indicated a mismatch in the number of features between the training and testing datasets when using the Support Vector Machine (SVM) model.

# Analysis:
# The root cause was identified as a discrepancy in the vectorization process. The TfidfVectorizer used during testing was creating a new instance,
# resulting in a different vocabulary and feature set compared to the training data.

# Resolution:
# To address this issue, the code was modified to store the vectorizer instance used during training (vectorizer_for_training).
# During testing, this stored vectorizer was then used to transform the testing data, ensuring consistency in the vocabulary and feature dimensions.

# Code Modification:
# During Training:
# X_training_set, vectorizer_for_training = vectorizer(text_data_train_file)

# During Testing:
# X_testing_set, _ = vectorizer(testing_data_file, training_vectorizer=vectorizer_for_training)

# This modification ensures that the same set of features is used for both training and testing, resolving the "ValueError" during predictions.

# ...
