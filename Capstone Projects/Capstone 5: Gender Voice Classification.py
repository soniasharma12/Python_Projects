# Create the DataFrames for both the train and test datasets and store them in the 'train_df' and 'test_df' variables respectively.
import pandas as pd
train_df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/project-4/gender-voice-train.csv')
test_df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/project-4/gender-voice-test.csv')

# Display the first 5 rows of the 'train_df' DataFrame.
print(train_df.head(5))

# Display the first 5 rows of the 'test_df' DataFrame.
print(test_df.head(5))

# Display the last 5 rows of the 'train_df' DataFrame.
print(train_df.tail(5))

# Display the last 5 rows of the 'test_df' DataFrame.
print(test_df.tail(5))

# Print the number of rows and columns in both the DataFrames.
print(train_df.shape)
test_df.shape

# Check for the missing values in the 'train_df' DataFrame.
train_df.isnull()

# Check for the missing values in the 'test_df' DataFrame.
test_df.isnull()

# Print the count of the 'male' and 'female' classes in the 'train_df' DataFrame.
train_df['label'].value_counts()

# Print the count of the 'male' and 'female' classes in the 'test_df' DataFrame.
test_df['label'].value_counts()

# Get the feature variables from the 'train_df' DataFrame.
x_train = train_df.iloc[:, 1:]
# Display the first 5 rows of the 'x_train' DataFrame.
x_train.head(5)

# Get the feature variables from the 'test_df' DataFrame.
x_test = test_df.iloc[:, 1:]
# Display the first 5 rows of the 'x_test' DataFrame.
x_test.head(5)

# Get the target variable from the 'train_df' DataFrame.
y_train = train_df.iloc[:, 0]
# Display the first 5 rows of the 'y_train' Pandas series.
y_train.head(5)

# Get the target variable from the 'test_df' DataFrame.
y_test = test_df.iloc[:, 0]
# Display the first 5 rows of the 'y_test' Pandas series.
y_test.head(5)

# Build a Random Forest Classifier model.
# Import the 'RandomForestClassifier' module.
from sklearn.ensemble import RandomForestClassifier
# Import the confusion_matrix and classification_report modules.
from sklearn.metrics import confusion_matrix,classification_report

# Predict the target variable based on the feature variables of the test DataFrame.
rf_clf = RandomForestClassifier(n_jobs=-1, n_estimators=50)
rf_clf.fit(x_train, y_train)
y_predicted = rf_clf.predict(x_test)
y_predicted

# Print the confusion matrix to see the number of TN, FN, TP and FP values.
confusion_matrix(y_test,y_predicted)

# Print the precision, recall and f1-score values for both the male and female classes.
print(classification_report(y_test,y_predicted))
