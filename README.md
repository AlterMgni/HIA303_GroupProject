# Can we predict benign or malignant Fine Needle Aspiration Cytology (FNAC) of breast tissue using supervised machine learning?
International Medical University HIA303 - Health Data Analytics [Group B] Official Group Project

# Contributor(s):
- Dr. Irfan – coding for decision tree, random forest, k-nearest neighbours (KNN), and Naïve Bayes
- Dr. Dura – coding for data preparation, descriptive analysis, and feature selection
- Tajul – coding for logistic regression and support vector model (SVM)
- Nisha – coding for descriptive analysis
- Ming – Github repository creation, organisation, and README writing and arrangement

The present repository concentrates on several supervised machine learning techniques conducted on a selected breast cancer dataset as an investigation into our established problem statement. Our additional aim is to determine the accuracy of each technique

# Prerequisite Data
This folder contains files retrieved from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

Of importance:
- _unformatted-data_ – original, unformatted database in a different arrangement than that of breast-cancer-wisconsin.data
- _breast-cancer-wisconsin.data_ – breast cancer database procured from the University of Wisconsin Hospitals, Madison
- _breast-cancer-wisconsin.names_ – supplementary names file for breast-cancer-wisconsin.data detailing its 11 attributes and accompanying domains
- _wdbc.data_ – the Wisconsin Diagnostic Breast Cancer (WPBC) database
- _wdbc.names_ – supplementary names file for wpbc.data detailing 32 attributes, including ID, diagnosis, and 30 real-valued input features with focus on diagnosis. Two types of diagnosis are set for its predicting field which are B = benign, and M = malignant
- _wpbc.data_ – the Wisconsin Prognostic Breast Cancer (WPBC) database
- _wpbc.names_ – supplementary names file for wpbc.data detailing 34 attributes, including ID, outcome, and 32 real-valued input features with focus on prognosis. Two types of outcomes are set for its predicting field which are R = recurrent, and N = nonrecurrent

# Machine Learning
This folder contains the codes written for several machine learning techniques, which are:
1. Logistic regression
2. Support vector model (SVM)
3. Decision tree
4. Random forest
5. K-nearest neighbour (KNN)
6. Naïve Bayes

# General Notes
1. Within the Prerequisite Data folder, wdbc.data in particular is chosen among all files to satisfy our project’s purpose because:
- In contrast with unformatted-data, this dataset is formatted
- As our problem statement involves exploration on benign and malignant breast tissues, this dataset has the advantage as it offers the aforementioned attributes unlike wpbc.data
-	Compared to breast-cancer-wisconsin.data and wpbc.data, this dataset does not carry missing attribute values
2. For descriptive_wdbc (Dura):
-	_wdbc.csv_ = wdbc.data in .csv format
-	df1 = pandas dataframe for wdbc.csv with ID and diagnosis attributes dropped
3. Within the Machine Learning folder, applicable to decision tree, random forest, KNN, and Naïve Bayes files:
-	df1 = pandas dataframe for wdbc.data after dropping of ID
-	df2 = cleaned dataframe from df1 to be used in training and testing
-	X = transformation of df2 using imputation method to replace null values   
-	y = labels used for testing
-	predictions = machine learning outcomes which are either benign or malignant
-	fpr = false positive rate (sensitivity)
-	tpr = true positive rate (1-sensitivity)
4. Programs involved in project:
-	Python 3.10
-	Jupyter Notebook
-	PyCharm
-	Spyder
