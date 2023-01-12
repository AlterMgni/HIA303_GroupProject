#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Jan  12 12:48:01 2023

@author: nishakurup
"""

import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

%pwd

#file path
'/Users/nishakurup/Desktop/HIA 303/wdbc (1).data'

# read text file
file = pd.read_csv('/Users/nishakurup/Desktop/HIA 303/wdbc (1).data')
print(file)

# convert text file into csv file format and save CSV format
wdbc_csv = file.to_csv ('/Users/nishakurup/Desktop/HIA 303/wdbc (1).csv')

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

# Statistical Summary
# 1. describe df1 
#df1 = 30 columns (not include ID & diagnosis), contain column with numerical value only
pd.set_option('display.width', 100)
pd.set_option('precision', 3)
description = df1.describe()
print(description)

# 2. class distribution 
# based on df
class_counts = df.groupby('Diagnosis').size()
print(class_counts)


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

# 3. Check for skewness
skew = df1.skew()
print(skew)

