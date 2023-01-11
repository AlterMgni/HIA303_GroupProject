import pandas as pd
import numpy as np


from sklearn.impute import KNNImputer
from sklearn.naive_bayes import GaussianNB #gaussian naive bayes
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report


#reading file as pandas data frame
df = pd.read_csv('wdbc.data', header=None)
df1=df.drop([0],axis=1)

#Convert the diagnosis column to a binary label (0 for benign, 1 for malignant)
df1[1] = df1[1].apply(lambda x: 1 if x == 'M' else 0)


#To split features vs outcome

df2=df1.iloc[:,1:31] #data only without label
df2_label=df1.iloc[:,0] #label only
y=np.ravel(df2_label) #label only

#impute data (using 2-nearest neighbor)
df2= df2.replace('?', np.nan)
imputer = KNNImputer(n_neighbors=2)
X = imputer.fit_transform(df2)

#Classification testing
predictions=np.zeros(X.shape[0])
probabilities=np.zeros([X.shape[0],2])

#using Random ForestCalssisier
from sklearn.ensemble import RandomForestClassifier

#split dataset using leave one out method
loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    X_train=X[train_index,:]
    y_train=y[train_index]
    X_test=X[test_index,:]
    y_test=y[test_index]

    #Create a RandomForrestClassifier  # to change algorithm  according to supervise learning method
    model =RandomForestClassifier()
    # Train the model using the training sets
    model.fit(X_train,y_train)

    #get prediction
    prediction = model.predict(X_test)

    #get probability
    probability = model.predict_proba(X_test)

    # write to append predictions and probabilities array
    predictions[test_index]= prediction
    probabilities[test_index]= probability

#report classification results
agreement=(predictions==y).sum()
accuracy=agreement/y.shape[0]

#second method to calculate accuracy
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve
accuracy2=accuracy_score(y,predictions)

print("The leave-one-out accuracy for the data set is: {0:.4f}".format(accuracy))
print(classification_report(y,predictions))

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
plt.ylabel('Sensitivity')
plt.xlabel('1-specificity')
plt.show()
