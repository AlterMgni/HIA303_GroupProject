#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 22:10:55 2023

@author: tajul
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset into a Pandas DataFrame
df = pd.read_csv('/Users/tajul/Documents/MHIA/HIA 303/Project/wdbc.data', header=None)

# Split the dataset into input features (X) and target label (y)
#Select compactness, concavity, concave points, symmetry, and fractal dimension 
X = df.iloc[:, 7:]
y = df.iloc[:, 1]

# Encode the target label as a binary class
y = (y == 'M').astype(int)


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Build the SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)


# Generate predictions for the test set
y_pred = model.predict(X_test)

# Create a confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print(confusion_mat)

# Plot the confusion matrix using seaborn
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Evaluate the model on the test set
accuracy = model.score(X_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')

# Select features from column 7 onwards
X_plot = X.iloc[:, 7:]

# Create a scatter plot of the data

sns.scatterplot(x=X_plot.iloc[:, 0], y=X_plot.iloc[:, 1], hue=y)
plt.legend(labels=['Malignant','Benign'])

plt.xlabel('Shape related features: compactness, concavity, concave points and fractal dimensions')
plt.ylabel('Diagnosis')
plt.show()

#find the sensitivity and specificity
tp = confusion_mat[0][0]
fp = confusion_mat[0][1]
fn = confusion_mat[1][0]
tn = confusion_mat[1][1]

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f'Sensitivity: {sensitivity:.2f}')
print(f'Specificity: {specificity:.2f}')
