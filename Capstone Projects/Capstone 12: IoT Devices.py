# Import the required modules and load the dataset.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/iot-devices/IoT-device.csv')
# Get the information on DataFrame.
df.info()

# Check for the null values in the DataFrame.
df.isnull().sum()

# Convert the values contained in the 'noted_date' column into the 'datetime' objects.
dt_series = pd.to_datetime(df['noted_date'])
df = df.drop(columns = ['noted_date'], axis = 1)
df.insert(loc = 0, column = 'DateTime', value = dt_series)
# Verify whether the conversion is successful or not.
print(type(df['DateTime'][0]))

# Sort the DataFrame in the increasing order of dates and time.
df = df.sort_values(by = "DateTime")

# Create new columns for year, month, day, day name, hours and minutes values and add to the DataFrame.
year_series = df['DateTime'].dt.year
month_series = df['DateTime'].dt.month
day_series = df['DateTime'].dt.day
day_name_seires = df['DateTime'].dt.day_name()
hour_series = df['DateTime'].dt.hour
min_series = df['DateTime'].dt.minute
month_name_series = df['DateTime'].dt.month_name()
df['Year'] = year_series
df['Month'] = month_series
df['Day'] = day_series
df['Day_name'] = day_name_seires
df['Hour'] = hour_series
df['Minute'] = min_series
df['Month_name'] = month_name_series
# Display the first five rows of the DataFrame.
df.head()

# Create a DataFrame for the indoor temperature records.
df_indoor = df[df['out/in']=="In"]
df_indoor.head() 
# Create a time series line plot for the indoor temperature records.
plt.figure(figsize = (20,5))
plt.plot(df_indoor['DateTime'],df_indoor['temp'],'g-o')
plt.grid()
plt.show()
# Create a DataFrame for the outdoor temperature records.
df_outdoor=df[df['out/in']=='Out']
df_outdoor.head()
# Create a time series line plot for the outdoor temperature records.
plt.figure(figsize = (20,5))
plt.plot(df_outdoor['DateTime'],df_outdoor['temp'],'r-o')
plt.grid()
plt.show()
# Compare the time series line plots for both the indoor and outdoor temperature records.
plt.figure(figsize = (20,5))
plt.plot(df_indoor['DateTime'],df_indoor['temp'],'g-o',label = "Indoor")
plt.plot(df_outdoor['DateTime'],df_outdoor['temp'],'r-o',label = "Outdoor")
plt.grid()
plt.legend()
plt.show()
# Create a box plot to represent the distribution of indoor and outdoor temperatures for the whole year.
plt.figure(figsize=(20,5))
sns.boxplot(y=df_indoor["DateTime"],x=df_indoor['temp'],color="red")
plt.show()
plt.figure(figsize=(20,5))
sns.boxplot(y=df_outdoor['DateTime'],x=df_outdoor['temp'],color="green")
plt.show()

