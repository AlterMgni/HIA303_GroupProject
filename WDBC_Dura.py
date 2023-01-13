#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 11:02:48 2023

@author: duratulainbintimohamadnazri
"""

import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

###########################################################################
# Data Collection and preparation 

%pwd

#file path
'/Users/duratulainbintimohamadnazri/Desktop/HIA 303 Group Projects/wdbc.data'

# read text file
file = pd.read_csv('/Users/duratulainbintimohamadnazri/Desktop/HIA 303 Group Projects/wdbc.data')
print(file)

# convert text file into csv file format and save CSV format
wdbc_csv = file.to_csv ('/Users/duratulainbintimohamadnazri/Desktop/HIA 303 Group Projects/wdbc.csv')

#list of attribute

columns = ['ID','Diagnosis','radius1','texture2','perimeter3','area4','smoothness5','compactness6','concavity7','concave points8','symmetry9','fractal dimension10',
        'radius11','texture12','perimeter13','area14','smoothness15','compactness16','concavity17','concave points18','symmetry19','fractal dimension20',
        'radius21','texture22','perimeter23','area24','smoothness25','compactness26','concavity27','concave points28','symmetry29','fractal dimension30']

print(columns)

# Add column names while reading a CSV file
df = pd.read_csv('/Users/duratulainbintimohamadnazri/Desktop/HIA 303 Group Projects/wdbc.data', names=columns)

print(df.head())
print(df.tail())

# Data types for each attribute
df.dtypes
print("Data types for each attribute is :", df.dtypes)

# Shape of the df
print("The shape of the dataframe is : ",df.shape)

# Data dimension
print(df.shape)

###################################################################################
# Data cleaning operations
## 1. Identify the rows that contain the duplicate observations based on the ID

# calculate duplicates
duplicate = df.duplicated()

# report if there are any duplicates
print(duplicate.any())

# 2. Search for Inconsistent data entry for output variables (Diagnosis : B or M)
# get all the unique values in the 'Diagnosis' column
diagnosis = df['Diagnosis'].unique()
print(diagnosis)

# 3. Search for data outliers
# drop unnecessary column
df1 = df.drop(columns=(['ID','Diagnosis']))
print(df1.shape)

# search outlier using IQR score (based on df1)
q1 = df1.quantile(0.25)
q3 = df1.quantile(0.75)
IQR = q3 - q1
print(IQR)

ul = q3 + (1.5*IQR)
print(ul)
print(df1 > ul)
print((df1 > ul).sum())

ll = q1 - (1.5*IQR)
print(df1 < ll)
print((df1 < ll).sum())

# 4. Search for missing values
# using isnull() function 
missing_value = df.isnull().sum()
print(missing_value)

#############################################################################

# Descriptive analysis

# 1. describe df1 
#df1 = 30 columns (not include ID & diagnosis), contain column with numerical value only
pd.set_option('display.width', 100)
pd.set_option('precision', 3)
description = df1.describe().T
print(description)

# 2. class distribution 
# based on df
class_counts = df.groupby('Diagnosis').size()
print(class_counts)

# 3. Proportion of Benign and Malignant breast mass (using pie chart)
import numpy as np
import matplotlib.pyplot as plt

piechart_diagnosis = df.groupby(['Diagnosis']).size().plot(
    kind= 'pie',autopct= '%1.0f%%', colors = ['pink','blue'],
    title = 'Proportion of Benign & Malignant Breast mass')
piechart_diagnosis

# 4. correlation between numerical values in WDBC dataset
import seaborn as sns

#specify size of heatmap
fig, ax = plt.subplots(figsize=(100, 100))

#create heatmap
correlation1 = sns.heatmap(df1.corr(), annot=True, fmt= '.1f', linewidths=.3)
print(correlation1)


# 3. Check for skewness
skew = df1.skew()
print(skew)

# 4. Check for data distribution using histogram
hist = df1.hist(bins=3)
print(hist)

####################################################################################

#Performing Feature Selection for Numerical Input Data 
#with the Categorical Target Variable

# 1. retrieve numpy array for the dataset 
dataset = df.values
print(dataset)

# 2. split into input (X) and output (y) variables
X = dataset[:, :-1]
y = dataset[:,-1]

# 3. 
# load the dataset
file_name = '/Users/duratulainbintimohamadnazri/Desktop/HIA 303 Group Projects/wdbc.data'

# save csv file of wdbc
# read wdbc.data file
read_file = pd.read_csv(file_name)

# save to wdbc1.csv
data.to_csv('/Users/duratulainbintimohamadnazri/Desktop/HIA 303 Group Projects/wdbc1.csv')


################################################################

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from matplotlib import pyplot
 
# load the dataset
def load_dataset(file_name):
 # load the dataset as a pandas DataFrame
 data = read_csv(file_name, names= columns)
 # retrieve numpy array
 dataset = data.values
 # split into input (X) and output (y) variables
 X = dataset[:, :-1]
 y = dataset[:,-1]
 return X, y
 
# feature selection
def select_features(X_train, y_train, X_test):
 # configure to select all features
 fs = SelectKBest(score_func=f_classif, k='all')
 # learn relationship from training data
 fs.fit(X_train, y_train)
 # transform train input data
 X_train_fs = fs.transform(X_train)
 # transform test input data
 X_test_fs = fs.transform(X_test)
 return X_train_fs, X_test_fs, fs
 
# load the dataset
X, y = load_dataset('wdbc1.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
 print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show() 

############################################################


