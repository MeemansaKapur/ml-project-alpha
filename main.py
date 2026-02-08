import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#data collection and analysis

#loading the diabeties dataset to a pandas dataframe
diabetes_dataset = pd.read_csv('/Users/meema/Desktop/ML/project/diabetes /dataset.csv', header=1)

#printing thr forst 5 rows of the dataset
print(diabetes_dataset.head())

#number of rows and columns in the dataset
print(diabetes_dataset.shape)

#getting the statistical measures of the data
print(diabetes_dataset.describe())

print(diabetes_dataset['Outcome'].value_counts())

#lable 0 represents non-diabetic and 1 represents diabetic

print(diabetes_dataset.groupby('Outcome').mean())
#separating the data and labels
X = diabetes_dataset.drop(columns='Outcome') #we drop the column outcome, axis=1 ie. dropping a particular column, axis=0 is for dropping a particular row
Y = diabetes_dataset['Outcome']
print(X)
print(Y)

#data standardization, wea re doing it to make the data in a particular range, so that the model can learn better
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)# transforming data to a particular range
#we are fitting and trasnforming the data separately because we want to use the same scaler for the test data as well, so that the test data is also in the same range as the training data
print(standardized_data)
X = standardized_data
Y = diabetes_dataset['Outcome']
print(X)
print(Y)
# 1. Clean the data (Drop rows with ANY missing values)
diabetes_dataset = diabetes_dataset.dropna()

# 2. Re-define X and Y (Crucial: do this AFTER dropping rows!)
X = diabetes_dataset.drop(columns='Outcome')
Y = diabetes_dataset['Outcome']

# 3. Now the split will work
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

#training the model
classifier = svm.SVC(kernel='linear')
#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

#model evaluation
#accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)
#accuracy score >75% is considered good for a classification model, if we have small data set, accuracy can be low
#accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)
#obvetraining: when the accuracy score of the training data is high and the accuracy score of the test data is low, it means that the model is overfitting, it is memorizing the training data and not generalizing well to the test data

# --- CORRECTED TRAINING SECTION ---

# 1. Clean the data and define X/Y
diabetes_dataset = diabetes_dataset.dropna()

X = diabetes_dataset.drop(columns='Outcome') 
Y = diabetes_dataset['Outcome']

# CRITICAL FIX: Force X to have 8 columns by removing the first one
# This removes the "Unnamed: 0" or junk index column
X = X.iloc[:, 1:] 

print("Shape of X before training:", X.shape) 
# ^ Check your terminal! This MUST say (rows, 8). If it says 9, the fix didn't work.

# 2. Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# 3. Fit Scaler (Now it will learn only 8 features)
scaler = StandardScaler()
scaler.fit(X_train) 

X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

# 4. Train Model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train_std, Y_train)

#making a predictive system
input_data = (5,166,72,19,175,25.8,0.587,51)
#changing the input data to a numpy array as processing the data on numpy array is easier
input_data_as_numpy_array = np.asarray(input_data)

#reshaping the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#standardizing the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)
if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')