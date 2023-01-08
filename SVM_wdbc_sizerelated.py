#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 22:10:55 2023

@author: tajul
"""

import pandas as pd
import matplotlib.pyplot as plt

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

# Select the first two features
X_plot = X.iloc[:, :2]

# Create a scatter plot of the data
colors = ['blue', 'red']
for label, color in zip((0, 1), colors):
    mask = y == label
    plt.scatter(X_plot.iloc[:, 0][mask], X_plot.iloc[:, 1][mask], c=color, label=['Benign', 'Malignant'][label])

# Add a legend to the plot
plt.legend()

# Set the x-axis label
plt.xlabel('Size related features: radius,perimeter and area')

# Set the y-axis label
plt.ylabel('Diagnosis')

# Show the plot
plt.show()
