#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:05:22 2023

@author: tajul
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data into a Pandas DataFrame
df = pd.read_csv('/Users/tajul/Documents/MHIA/HIA 303/Project/wdbc.data', header=None)

# Convert the diagnosis column to a binary label (0 for benign, 1 for malignant)
df[1] = df[1].apply(lambda x: 1 if x == 'M' else 0)

# Select the features and target columns
X = df.iloc[:, 2:11]
y = df.iloc[:, 1]

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a logistic regression model
model = LogisticRegression(penalty='l2', dual=False, max_iter=1000)

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')

# Create the confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

# Calculate sensitivity (true positive rate)
sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
print(f'Sensitivity : {sensitivity:.2f}')

# Calculate specificity (true negative rate)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 1] + conf_matrix[0, 0])
print(f'Specificity : {specificity:.2f}')

# Visualize the confusion matrix using seaborn
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')


#Add a title to the plot
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import roc_auc_score, roc_curve

# Get predicted probabilities of positive class
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f'AUC: {auc:.2f}')

# Get the false positive rate, true positive rate and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plot the ROC curve
plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.title('ROC Curve for Logistic Regression')
plt.show()






