# Import the required modules and load the dataset.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv('https://s3-student-datasets-bucket.whjr.online/whitehat-ds-datasets/house-prices.csv')
df.head()
# Get the information on DataFrame.
df.info()
# Check if there are any NULL values.
df.isnull().sum()

# Check categorical attributes
categ_df = df.select_dtypes(include = 'object')
categ_df.head()
# Boxplot for 'mainroad' vs 'price'
plt.figure(figsize = (16, 8))
sns.boxplot(x = df['mainroad'], y = df['price'])
plt.show()
# Boxplot for 'guestroom' vs 'price'
plt.figure(figsize =  (16, 8))
sns.boxplot(x = df['guestroom'], y = df['price'])
plt.show()
# Boxplot for 'basement' vs 'price'
plt.figure(figsize =  (16, 8))
sns.boxplot(x = df['basement'], y = df['price'])
plt.show()
# Boxplot for 'hotwaterheating' vs 'price'
plt.figure(figsize = (16, 8))
sns.boxplot(x = df['hotwaterheating'], y = df['price'])
plt.show()
# Boxplot for 'airconditioning' vs 'price'
plt.figure(figsize = (16, 8))
sns.boxplot(x = df['airconditioning'], y = df['price'])
plt.show()
# Boxplot for 'prefarea' vs 'price'
plt.figure(figsize = (16, 8))
sns.boxplot(x = df['prefarea'], y = df['price'])
plt.show()
# Boxplot for 'furnishingstatus' vs 'price'
plt.figure(figsize = (16, 8))
sns.boxplot(x = df['furnishingstatus'], y = df['price'])
plt.show()

# Create scatter plot with 'area' on X-axis and 'price' on Y-axis
plt.figure(figsize = (16, 8))
plt.scatter(x = df['area'], y = df['price'], color = 'purple')
plt.show()
# Create scatter plot with 'bedrooms' on X-axis and 'price' on Y-axis
plt.figure(figsize = (16, 8))
plt.scatter(x = df['bedrooms'], y = df['price'], color = 'purple')
plt.show()
# Create scatter plot with 'bathrooms' on X-axis and 'price' on Y-axis
plt.figure(figsize = (16, 8))
plt.scatter(x = df['bathrooms'], y = df['price'], color = 'purple')
plt.show()
# Create scatter plot with 'stories' on X-axis and 'price' on Y-axis
plt.figure(figsize = (16, 8))
plt.scatter(x = df['stories'], y = df['price'], color = 'purple')
plt.show()

# Create a normal distribution curve for the 'price'.
plt.figure(figsize = (16, 8))
sns.distplot(df['price'], bins = 'sturges', color = 'black', hist = False)
plt.axvline(df['price'].mean(), color = 'red', label = 'Mean of price')
plt.legend()
plt.show()
# Create a probablity density function for plotting the normal distribution
def prob_den_func(arr, mean, std):
  constant = 1/(std*np.sqrt(2*np.pi))
  power = np.exp(-(arr - mean)**2/(2*std**2))
  r = constant * power
  return r  
# Plot the normal distribution curve using plt.scatter() 
plt.figure(figsize = (16, 8))
plt.scatter(x = df['price'], y = prob_den_func(df['price'], df['price'].mean(), df['price'].std()), color = 'olive')
plt.axvline(df['price'].mean(), color = 'red', label = 'Mean of price')
plt.legend()
plt.show()

# Replace yes with 1 and no with 0 for all the values in features 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea' using map() function.
val = {'yes':1, 'no':0}
df['mainroad'] = df['mainroad'].map(val)
df['guestroom'] = df['guestroom'].map(val)
df['basement'] = df['basement'].map(val)
df['hotwaterheating'] = df['hotwaterheating'].map(val)
df['airconditioning'] = df['airconditioning'].map(val)
df['prefarea'] = df['prefarea'].map(val)
# Print dataframe
df.head()
# Perform one hot encoding for furnishingstatus feature.
dummy = pd.get_dummies(df['furnishingstatus'], dtype = int)
df = pd.concat([df, dummy], axis = 1)
dummy.head()
# Drop 'furnishingstatus' feature
df.drop(columns = ['furnishingstatus'], axis = 1, inplace = True)
df.head()
# Print dataframe 
print(df)

# Split the 'df' Dataframe into the train and test sets.
train_df, test_df = train_test_split(df, test_size = 0.33, random_state = 42)
# Create separate data-frames for the feature and target variables for both the train and test sets.
features = df.columns[:].to_list()
features.remove('price')
X_train = train_df[features]
X_test = test_df[features]
y_train = train_df['price']
y_test = test_df['price']
# Build a linear regression model using all the features to predict prices.
import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_sm).fit()
# Print the summary of the linear regression report.
print(lr.summary())
# Calculate N and p values
n = X_train.shape[0]
p = X_train.shape[1]
print(f"Row {n} \nColumns {p}")
# Calculate the adjusted R-square value.
print(lr.rsquared_adj)

# Build multiple linear regression model using all the features
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_train_pred = lin_reg.predict(X_train)
# Evaluate the linear regression model using the 'r2_score', 'mean_squared_error' & 'mean_absolute_error' functions of the 'sklearn' module.
print("Training Dataset")
print('-'*50)
print(f"R2 Score: {r2_score(y_train, y_train_pred):.3f}")
print(f"Mean Squared Error: {mean_squared_error(y_train, y_train_pred):.3f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_train, y_train_pred)}")

# Create a Python dictionary storing the moderately to highly correlated features with price and the corresponding correlation values.
# Keep correlation threshold to be 0.2
features = df.columns[:].to_list()
features.remove('price')
major_features = {}
for i in features:
  corr_cof = np.corrcoef(df['price'], df[i])[0,1]
  if corr_cof >= 0.2 or corr_cof <= -0.2:
    major_features[i] = corr_cof
major_features
# Perform RFE and select best 7 features  
rfe1 = RFE(lin_reg, 7)
rfe1.fit(x_train[major_features.keys()], y_train)
print(major_features.keys())
print(rfe1.support_)
print(rfe1.ranking_)
# Print the 7 features selected by RFE in the previous step.
rfe_features = x_train[major_features.keys()].columns[rfe1.support_]
rfe_features
# Build multiple linear regression model using all the features selected after RFE
X_train_rfe1 = X_train[rfe_features]
X_train_rfe1_sm = sm.add_constant(X_train_rfe1)
lr2= sm.OLS(y_train, X_train_rfe1_sm).fit()
print(lr2.summary())
# Split the DataFrame into the train and test sets such that test set has 33% of the values.
X = df[rfe_features]
y = df['price']
X_train_rfe2, X_test_rfe2, y_train_rfe2, y_test_rfe2 = train_test_split(X, y, test_size = 0.33, random_state = 42)
# Build linear regression model using the 'sklearn.linear_model' module.
lin_reg2 = LinearRegression()
lin_reg2.fit(X_train_rfe2, y_train_rfe2)
# Print the value of the intercept
print("\nConstant".ljust(15," "), f"{lin_reg2.intercept_:.3f}")
# Print the names of the features along with the values of their corresponding coefficients.
for i in list(zip(X.columns.values, lin_reg2.coef_)):
  print(f"{i[0]}".ljust(15, " "), f"{i[1]:.6f}")
# Evaluate the linear regression model using the 'r2_score', 'mean_squared_error' & 'mean_absolute_error' functions of the 'sklearn' module.
y_train_rfe_pred = lin_reg2.predict(X_train_rfe2)
y_test_rfe_pred = lin_reg2.predict(X_test_rfe2)
print(f"Training Dataset\n {'-' * 50}")
print(f"R2 Score: {r2_score(y_train_rfe2, y_train_rfe_pred):.3f}")
print(f"Mean Squared Error: {mean_squared_error(y_train_rfe2, y_train_rfe_pred):.3f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_train_rfe2, y_train_rfe_pred):.3f}")
print(f"\n\nTesting Dataset\n {'-' * 50}")
print(f"R2 Score: {r2_score(y_train_rfe2, y_train_rfe_pred):.3f}")
print(f"Mean Squared Error: {mean_squared_error(y_train_rfe2, y_train_rfe_pred):.3f}")
print(f"Mean Absolute Eror: {mean_absolute_error(y_train_rfe2, y_train_rfe_pred):.3f}")

# Create a histogram for the errors obtained in the predicted values for the train set.
errors_train = y_train_rfe2 - y_train_rfe_pred
plt.figure(figsize = (16, 8))
sns.histplot(errors_train, bins = 'sturges')
plt.show()
# Create a histogram for the errors obtained in the predicted values for the test set.
errors_test = y_test_rfe2 - y_test_rfe_pred
plt.figure(figsize = (16, 8))
sns.histplot(errors_test, bins = 'sturges')
plt.show()

# Create a scatter plot between the errors and the dependent variable for the train set.
plt.figure(figsize = (16, 8))
plt.scatter(errors_train, y_train_rfe2, color = 'olive')
plt.show()
