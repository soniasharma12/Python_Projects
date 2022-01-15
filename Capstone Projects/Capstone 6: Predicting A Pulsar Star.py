# Create the DataFrames for both the train and test datasets and store them in the 'train_df' and 'test_df' variables respectively.
import pandas as pd, numpy as np
train_df = pd.read_csv("https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/project-5/pulsar-star-prediction-train.csv")
test_df = pd.read_csv("https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/project-5/pulsar-star-prediction-test.csv")

# Display the first 5 rows of the 'train_df' DataFrame.
train_df.head(5)
# Display the first 5 rows of the 'test_df' DataFrame.
test_df.head(5)
# Display the last 5 rows of the 'train_df' DataFrame.
train_df.tail()
# Display the last 5 rows of the 'test_df' DataFrame.
test_df.tail()

# Print the number of rows and columns in both the DataFrames.
print("Test data frame: ",test_df.shape)
print("Train data frame: ",train_df.shape)

# Check for the missing values in the 'train_df' DataFrame.
train_df.isnull
# Check for the missing values in the 'test_df' DataFrame.
test_df.isnull

# Print the count of the '0' and '1' classes in the 'train_df' DataFrame.
print(train_df['target_class'].value_counts(0))
print(train_df['target_class'].value_counts(1))
# Print the count of the '0' and '1' classes in the 'test_df' DataFrame.
print(test_df['target_class'].value_counts(0))
print(test_df['target_class'].value_counts(1))

# Get the feature variables from the 'train_df' DataFrame.
x_train = train_df.iloc[:,1:]
# Display the first 5 rows of the 'x_train' DataFrame.
x_train.head()
# Get the feature variables from the 'test_df' DataFrame.
x_test = test_df.iloc[:,1:]
# Display the first 5 rows of the 'x_test' DataFrame.
x_test.head()

# Get the target variable from the 'train_df' DataFrame.
y_train = train_df.iloc[:,0]
# Display the first 5 rows of the 'y_train' Pandas series.
y_train.head()
# Get the target variable from the 'test_df' DataFrame.
y_test = test_df.iloc[:,0]
# Display the first 5 rows of the 'y_test' Pandas series.
y_test.head()

# Build A XGBoost Classifier model
# Import the xgboost module.
import xgboost as xg
# Import the confusion_matrix and classification_report modules.
from sklearn.metrics import classification_report,confusion_matrix
# Predict the target variable based on the feature variables of the test dataframe.
pulstar = xg.XGBClassifier()
print(pulstar.fit(x_train,y_train))
acc = pulstar.score(x_train,y_train)
print(acc)
predict = pulstar.predict(x_test)
print(predict)

# Print the confusion matrix to see the number of TN, FN, TP and FP.
con_mat = confusion_matrix(y_test,predict)
print(con_mat)
# Print the precision, recall and f1-score values for both the '0' and '1' classes.
print(classification_report(y_test,predict))
