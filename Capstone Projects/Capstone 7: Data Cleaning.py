# Import the necessary modules.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset.
bengluru_df = pd.read_csv("https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/bengaluru-house-prices/Bengaluru_House_Prices.csv")
# Display the number of rows and columns.
bengluru_df.shape

# Find the total number of missing values in each column.
bengluru_df.isnull().sum()
# Find the percentage of missing values
bengluru_df.isnull().sum() * 100 / bengluru_df.shape[0]

# Get all the rows having the missing values in the 'location'.
bengluru_df.loc[bengluru_df['location'].isnull() == True,:]
# Total number of rows having the missing values in the 'location' column.
bengluru_df.loc[bengluru_df['location'].isnull() == True,:].shape[0]
# Discard the rows containing the missing values in the 'location' column.
bengluru_df.loc[bengluru_df['location'].isnull() == False,:]

# Get the rows having the missing values in the 'size' column.
bengluru_df.loc[bengluru_df['size'].isnull() == True,:]
# Total number of rows having the missing values in the 'size' column.
bengluru_df.loc[bengluru_df['size'].isnull() == True,:].shape[0]
# Discard the rows containing the missing values in the 'size' column.
bengluru_df.loc[bengluru_df['size'].isnull() == False,:]

# Get the rows having the missing values in the 'total_sqft' column.
bengluru_df.loc[bengluru_df['total_sqft'].isnull() == True,:]
# Get the rows having more than 5 bathrooms in the 'bath' column.
bengluru_df[bengluru_df['bath'] > 5]
# Discard the rows having more than 5 bathrooms in the 'bath' column.
bengluru_df[bengluru_df['bath'] <= 5]

# Percentage of missing values.
bengluru_df.isnull().sum() * 100 / bengluru_df.shape[0]

# List of the columns to be retained.   
column_ret=[]
for ret in bengluru_df.columns:
  if ret!= 'society':
    column_ret.append(ret)
column_ret
# Retain the appropriate columns in the DataFrame.
bengluru_df.loc[:, column_ret].head()
# Percentage of the missing values in the DataFrame.
bengluru_df.isnull().sum() * 100 / bengluru_df.shape[0]

# Get the descriptive statistics of the identified column(s).
bengluru_df['balcony'].describe()
# Create boxplot before replacing the missing values in the identified column(s).
plt.figure(figsize=(20, 1.5))
sns.boxplot(x='balcony', data=bengluru_df)
plt.show()
# Computing the modal value in the 'balcony' column.
bengluru_df['balcony'].mode()
# Create a list of indices of the rows containing the missing values in the identified column(s).
row_indices_balcony = bengluru_df[bengluru_df['balcony'].isnull() == True].index
row_indices_balcony
# Replace the missing values in the identified column(s) with the appropriate value.
bengluru_df.loc[row_indices_balcony, 'balcony'] = 2.5
# Display the first 5 rows of the DataFrame, after replacing the missing values in the identified column(s).
bengluru_df.loc[row_indices_balcony,:].head()
# Check for missing values again.
bengluru_df.isnull().sum()

# Convert the values in the 'bath' and 'balcony' columns to integer values.
bengluru_df.astype({'bath': int, 'balcony': int})
# Print the data-types of the values in the 'bath' and 'balcony' columns.
print(bengluru_df['bath'].dtype)
print(bengluru_df['balcony'].dtype)
# Display the first 5 rows of the DataFrame.
bengluru_df.head()
