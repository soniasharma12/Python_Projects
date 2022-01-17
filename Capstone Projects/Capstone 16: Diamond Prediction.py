# Import the required modules and load the dataset.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/diamonds.csv')

# Get the information on DataFrame.
df.info()
# Check if there are any null values. If any column has null values, treat them accordingly
df.isnull().sum()
# Drop 'Unnamed: 0' column as it is of no use
df = df.drop(columns = 'Unnamed: 0', axis = 1)
df.head()

# Boxplot for 'cut' vs 'price'
plt.figure(figsize = (20,6))
sns.boxplot(df['cut'], df['price'])
plt.show()
# Boxplot for 'color' vs 'price'
plt.figure(figsize = (20,6))
sns.boxplot(df['color'], df['price'])
plt.show()
# Boxplot for 'clarity' vs 'price'
plt.figure(figsize = (20,6))
sns.boxplot(df['clarity'], df['price'])
plt.show()

# Create scatter plot with 'carat' on X-axis and 'price' on Y-axis
plt.figure(figsize = (20,6), dpi = 96)
plt.scatter(df['carat'], df['price'])
plt.title('Scatter plot of Carat vs. Price of Diamonds')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.show()
# Create scatter plot with 'depth' on X-axis and 'price' on Y-axis
plt.figure(figsize = (20,6),dpi = 96)
plt.scatter(df['depth'], df['price'])
plt.title('Scatter plot of Depth vs. Price of Diamonds')
plt.xlabel('Depth')
plt.ylabel('Price')
plt.show()
# Create scatter plot with 'table' on X-axis and 'price' on Y-axis
plt.figure(figsize = (20,6),dpi = 96)
plt.scatter(df['table'], df['price'])
plt.title('Scatter plot of Table vs. Price of Diamonds')
plt.xlabel('Table')
plt.ylabel('Price')
plt.show()
# Create scatter plot with attribute 'x' on X-axis and 'price' on Y-axis
plt.figure(figsize = (20,6),dpi = 96)
plt.scatter(df['x'], df['price'])
plt.title('Scatter plot of x vs. Price of Diamonds')
plt.xlabel('x')
plt.ylabel('Price')
plt.show()
# Create scatter plot with attribute 'y' on X-axis and 'price' on Y-axis
plt.figure(figsize = (20,6),dpi = 96)
plt.scatter(df['y'], df['price'])
plt.title('Scatter plot of y vs. Price of Diamonds')
plt.xlabel('y')
plt.ylabel('Price')
plt.show()
# Create scatter plot with 'z' on X-axis and 'price' on Y-axis
plt.figure(figsize = (20,6),dpi = 96)
plt.scatter(df['z'], df['price'])
plt.title('Scatter plot of z vs. Price of Diamonds')
plt.xlabel('z')
plt.ylabel('Price')
plt.show()

# Create a normal distribution curve for the `price`.
plt.figure(figsize = (20, 5))
sns.distplot(x = df['price'], bins = 'sturges', hist = False, color = 'green')
plt.axvline(df['price'].mean(), color = 'red', label = 'Mean')
plt.grid()
plt.legend()
plt.show()
# Create a probablity density function for plotting the normal distribution
def prob_dens(arr,mean,std):
  coeff=1/(std*np.sqrt(2*np.pi))
  power=np.exp(-((arr-mean)**2/(2*(std**2))))
  return coeff*power
# Plot the normal distribution curve using plt.scatter() 
rho = prob_dens(df["price"].sort_values(),df["price"].mean(),df["price"].std())
plt.figure(figsize=(20,5))
plt.scatter(df["price"].sort_values(),rho)
plt.axvline(x=df["price"].mean(),label="Mean of Price")
plt.title("Normal Distribution Curve of the Price of Diamonds")
plt.legend()
plt.grid()
plt.show()

# Replace values of 'cut' column
df["cut"].replace({"Fair":1,"Good":2,"Very Good":3,"Premium":4,"Ideal":5}, inplace = True)
# Replace values of 'color' column
df["color"].replace({"D":1,"E":2,"F":3,"G":4,"H":5,"I":6,"J":7}, inplace = True)
# Replace values of 'clarity' column
df["clarity"].replace({"I1":1,"SI2":2,"SI1":3,"VS2":4,"VS1":5,"VVS2":6,"VVS1":7,"IF":8}, inplace = True)
# Create a list of feature variables.
features = list(df.drop("price", axis = 1))
print(features)

# Build multiple linear regression model using all the features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Split the DataFrame into the train and test sets such that test set has 33% of the values.
X = df[features]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
# Build linear regression model using the 'sklearn.linear_model' module.
lr = LinearRegression()
lr.fit(X_train,y_train)
# Print the value of the intercept
print(f"Constant: {lr.intercept_}")
# Print the names of the features along with the values of their corresponding coefficients.
for i in list(zip(X.columns.values, lr.coef_)):
  print(f"{i[0]} : {i[1]}")

# Evaluate the linear regression model using the 'r2_score', 'mean_squared_error' & 'mean_absolute_error' functions of the 'sklearn' module.
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)
print(f"Train Set\n{'-' * 50}")
print(f"R Squared: {r2_score(y_train, y_train_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_train, y_train_pred):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_train, y_train_pred):.4f}")   
print(f"\n\nTest Set\n{'-' * 50}")
print(f"R Squared: {r2_score(y_test, y_test_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_test_pred):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_test_pred):.4f}")

# Heatmap to pinpoint the columns in the 'df' DataFrame exhibiting high correlation
plt.figure(figsize=(20,6),dpi=96)
sns.heatmap(df.corr(), annot=True)
plt.show()
# Drop features highly correlated with 'carat'
features_updated = list(df.columns.drop(["x","y","z", "price"]))
features_updated
# Again build a linear regression model using the remaining features
X = df[features_updated]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
# Build linear regression model using the 'sklearn.linear_model' module.
lr1 = LinearRegression()
lr1.fit(X_train, y_train)
# Print the value of the intercept
print(f"Constant: {lr1.intercept_}")
# Print the names of the features along with the values of their corresponding coefficients.
for i in list(zip(X.columns.values, lr1.coef_)):
  print(f"{i[0]} : {i[1]}")
# Evaluate the linear regression model using the 'r2_score', 'mean_squared_error' & 'mean_absolute_error' functions of the 'sklearn' module.
y_train_pred = lr1.predict(X_train)
y_test_pred = lr1.predict(X_test)
print(f"Train Set\n{'-' * 50}")
print(f"R Squared: {r2_score(y_train, y_train_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_train, y_train_pred):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_train, y_train_pred):.4f}")      
print(f"\n\nTest Set\n{'-' * 50}")
print(f"R Squared: {r2_score(y_test, y_test_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_test_pred):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_test_pred):.4f}")  

# Calculate the VIF values for the remaining features using the 'variance_inflation_factor' function.
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Add a constant to feature variables 
X_train_sm= sm.add_constant(X_train)
# Create a dataframe that will contain the names of the feature variables and their respective VIFs
vif = pd.DataFrame()
vif["Features"] = X_train_sm.columns
vif["VIF"] = [variance_inflation_factor(X_train_sm.values,i) for i in range(X_train_sm.values.shape[1])]
vif["VIF"] = round(vif["VIF"], 2)
vif = vif.sort_values(by="VIF",ascending=False)
vif

# Create a list of features having VIF values less than 10 
vif_list = []
for i in range(len(vif["VIF"])):
  if vif["VIF"][i]<10:
    vif_list.append(vif["Features"][i])
vif_list
# Again build a linear regression model using the features whose VIF values are less than 10 
X = df[vif_list]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
# Build linear regression model using the 'sklearn.linear_model' module.
lr2 = LinearRegression()
lr2.fit(X_train,y_train)
# Print the value of the intercept
print(f"Constant: {lr2.intercept_}")
# Print the names of the features along with the values of their corresponding coefficients.
for i in list(zip(X.columns.values, lr2.coef_)):
  print(f"{i[0]} : {i[1]}")
# Evaluate the linear regression model using the 'r2_score', 'mean_squared_error' & 'mean_absolute_error' functions of the 'sklearn' module.
y_train_pred = lr2.predict(X_train)
y_test_pred = lr2.predict(X_test)
print(f"Train Set\n{'-' * 50}")
print(f"R Squared: {r2_score(y_train, y_train_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_train, y_train_pred):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_train, y_train_pred):.4f}")    
print(f"\n\nTest Set\n{'-' * 50}")
print(f"R Squared: {r2_score(y_test, y_test_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_test_pred):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_test_pred):.4f}")  

# Again calculate the VIF values for the remaining features to find out if there is still multicollinearity
X_train_2 = df[["x", "y", "z"]]
X_test_2 = df[["x", "y", "z"]]
X_train_sm1 = sm.add_constant(X_train_2)
vif = pd.DataFrame()
vif["Features"] = X_train_sm1.columns
vif["VIF"] = [variance_inflation_factor(X_train_sm1.values,i) for i in range(X_train_sm1.values.shape[1])]
vif["VIF"] = round(vif["VIF"], 2)
vif = vif.sort_values(by="VIF", ascending = False)
vif

# Create a histogram for the errors obtained in the predicted values for the train set.
errors_train = y_train - y_train_pred
plt.figure(figsize=(20,5))
plt.hist(errors_train, bins="sturges", edgecolor="black")
plt.title("Histogram for the Errors Obtained in the Predicted Values for the Train Set")
plt.axvline(errors_train.mean(), label=f"Mean of Errors = {errors_train.mean():.3f}", color="red")
plt.xlabel("Errors")
plt.ylabel("Count")
plt.legend()
plt.show()
# Create a histogram for the errors obtained in the predicted values for the test set.
errors_test = y_test - y_test_pred
plt.figure(figsize=(20,5))
plt.hist(errors_test, bins="sturges", edgecolor="black")
plt.title("Histogram for the Errors Obtained in the Predicted Values for the Test Set")
plt.axvline(errors_test.mean(), label=f"Mean of Errors = {errors_test.mean():.3f}", color="red")
plt.xlabel("Errors")
plt.ylabel("Count")
plt.legend()
plt.show()

# Create a scatter plot between the errors and the dependent variable for the train set.
plt.figure(figsize=(20,5), dpi = 96)
plt.scatter(y_train, errors_train)
plt.title("Scatter Plot Between the Errors and the Dependent Variable for the Train Set")
plt.axhline(errors_train.mean(), label=f"Mean of Errors = {errors_train.mean():.3f}", color="red")
plt.legend()
plt.show()
