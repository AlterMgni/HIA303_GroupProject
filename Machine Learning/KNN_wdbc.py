
import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# (suppress unnecessary warnign)
# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


#reading file as pandas data frame
df = pd.read_csv('wdbc.data', header=None)
df1=df.drop([0],axis=1)

#Convert the diagnosis column to a binary label (0 for benign, 1 for malignant)
df1[1] = df1[1].apply(lambda x: 1 if x == 'M' else 0)


#To split features vs outcome

df2=df1.iloc[:,1:31] #data only without label
df2_label=df1.iloc[:,0] #label only
y=np.ravel(df2_label) #label only

#impute data (2-nearest neighbor)
df2 = df2.replace('?', np.nan)
imputer = KNNImputer(n_neighbors=2)
X = imputer.fit_transform(df2)

# scale the data
X = preprocessing.scale(X)

# classification testing
predictions = np.zeros(X.shape[0])
probabilities = np.zeros([X.shape[0], 2])
loo = LeaveOneOut()

k_range = range(1, 40, 2)
accuracies = np.zeros(len(k_range))
count = 0

#loop from 1 to 40 with interval of 2 (off numbers only)
for k in k_range:
    for train_index, test_index in loo.split(X):
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]

        # Create KNN Classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        # Train the model using the training sets
        knn.fit(X_train, y_train)

        # get prediction and probabiliy
        prediction = knn.predict(X_test)
        probability = knn.predict_proba(X_test)

        # write to append predictions array
        predictions[test_index] = prediction
        probabilities[test_index] = probability


# report classification results
agreement = (predictions == y).sum()
accuracy = agreement / y.shape[0]
print("k={0},The leave-one-out accuracy is: {1:.4f}".format(k, accuracy))
accuracies[count] = accuracy
count = count + 1

print(classification_report(y, predictions))


#second method to calculate accuracy
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve
accuracy2=accuracy_score(y,predictions)


#plot the accuracies
plt.plot(k_range, accuracies)
plt.xlabel("Value of k")
plt.ylabel("Accuracy")

#calculate confusion matrix
conM=confusion_matrix(y,predictions)
print(conM)


#to calculate area under curve (AUC) and plot ROC
y=pd.Series(y)
fpr, tpr, thresholds = roc_curve(y,probabilities[:,1])
roc_auc=roc_auc_score(y,probabilities[:,1])

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('sensitivity')
plt.xlabel('1-specificity')
plt.show()
