# Import require models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/project-3/sample_csv_file.csv')

# Display first/last five rows of your dataframe
print(df.head())
print(df.tail())

# Display number of rows and columns
df.shape

# Check whether the dataframe contains any null (missing) values
df.isnull()

# Create a pandas series called cum_freq_x1 which contains the cumulative frequency of the x1 values and print the first 10 values of the series. Also, verify whether it is a pandas series.
cum_freq_x1 = []
cumulative = 0
for i in range(df.shape[0]):
  cumulative += df["x1"][i]
  cum_freq_x1.append(cumulative)
cum_freq_x1 = pd.Series(cum_freq_x1)
print(cum_freq_x1.head(10))
print(cum_freq_x1)

# Print the minumum, maximum, mean and median values in the cum_freq_x1 column in the dataframe
print(cum_freq_x1.min())
print(cum_freq_x1.max())
print(cum_freq_x1.mean())
print(cum_freq_x1.median())

# Create Scatter & Line Plots

# Scatter plot between y and x1.
y = df.iloc[:, 0]
x1 = df.iloc[:,1]
plt.figure(figsize=(18, 4))
plt.scatter(y, x1)
plt.show()
# Line plot between y and x1.
plt.figure(figsize=(18,4))
plt.plot(y,x1)
plt.show()

# Scatter plot between y and x2.
y = df.iloc[:,0]
x2 = df.iloc[:, 2]
plt.figure(figsize=(18,4))
plt.scatter(y,x2)
plt.show()
# Line plot between y and x2.
plt.figure(figsize=(18,4))
plt.plot(y,x2)
plt.show()

# Scatter plot between y and x3.
x3 = df.iloc[:, 3]
plt.figure(figsize=(18,4))
plt.scatter(y,x3)
plt.show
# Line plot between y and x3.
plt.figure(figsize=(18,4))
plt.plot(y,x3)
plt.show()

# Scatter plot between y and x4.
x4 = df.iloc[:,4]
plt.figure(figsize=(18,4))
plt.scatter(y,x4)
plt.show()
# Line plot between y and x4.
plt.figure(figsize=(18,4))
plt.plot(y,x4)
plt.show()

# Scatter plot between y and 2 * x1 + 3 * x2 + 4 * x3 + 5 * x4.
plt.figure(figsize=(18,4))
plt.scatter(y, 2 * x1 + 3 * x2 + 4 * x3 + 5 * x4)
plt.show()
# Line plot between y and 2 * x1 + 3 * x2 + 4 * x3 + 5 * x4.
plt.figure(figsize=(18,4))
plt.plot(y, 2 * x1 + 3 * x2 + 4 * x3 + 5 * x4)
plt.show()
