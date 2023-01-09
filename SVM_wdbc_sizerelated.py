#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 22:10:55 2023

@author: tajul
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset into a Pandas DataFrame
df = pd.read_csv('/Users/tajul/Documents/MHIA/HIA 303/Project/wdbc.data', header=None)

# Split the dataset into input features (X) and target label (y)
#Select radius,perimeter and area 
X = df.iloc[:, [2, 3, 4]]
y = df.iloc[:, 1]

# Encode the target label as a binary class
y = (y == 'M').astype(int)

from sklearn.model_selection import train_test_split

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

# Standardize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC

# Build the SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = model.score(X_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')

# Select first 6 columns; radius,perimeter,texture,area
X_plot = X.iloc[:, :6]

sns.scatterplot(x=X_plot.iloc[:, 0], y=X_plot.iloc[:, 1], hue=y)

plt.legend(labels=['Malignant','Benign'])

plt.xlabel('Size related features: radius,texture,perimeter and area')
plt.ylabel('Diagnosis')
plt.show()


