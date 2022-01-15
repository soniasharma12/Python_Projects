# Import the modules required.
import pandas as pd
import matplotlib.pyplot as plt
# Load the dataset.
vg_sales_df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/video-games-sales/video-game-sales.csv')
# Get the dataset information.
vg_sales_df.info()

# Check for the null values in all the columns.
vg_sales_df.isnull().sum()
# Remove the rows/columns containing the null values.
vg_sales_df = vg_sales_df.dropna()
vg_sales_df.head()
# Convert the data-type of the year values into integer values.
vg_sales_df['Year'] = vg_sales_df['Year'].astype(int)
vg_sales_df.head()

# Find out the total number of units sold yearly across different regions and the world.
total_sold_df = vg_sales_df.groupby(by = 'Year', as_index = False)
total_sold_df.sum()
# Create the line plots for the total number of units sold yearly across different regions and the world.
plt.figure(figsize=[20,7])
x=total_sold_df.sum()["Year"]
y=total_sold_df.sum()["Global_Sales"]
plt.title("Number of sold units per year")
plt.xlabel("Year")
plt.ylabel("Number of sold units")
plt.plot(x,y)
plt.grid
plt.show
# In which year, the most number of games were sold globally and how many?
total_sold_df.sum()

# Find out the genre-wise total number of units sold across different regions and the world.
total_sold_genre_df = vg_sales_df.groupby(by = 'Genre', as_index = False)
total_sold_genre_df.head()
# Create line plots for genre-wise total number of units sold across different regions and the world.
plt.figure(figsize=[20,7])
x=total_sold_genre_df.sum()["Genre"]
y=total_sold_genre_df.sum()["Global_Sales"]
plt.title("Number of sold units per year")
plt.xlabel("Genre")
plt.ylabel("Number of sold units")
plt.plot(x,y)
plt.grid()
plt.show()
# Genre-wise total number of units sold across different regions and the world in descending order.
total_sold_genre_df=vg_sales_df.groupby(by="Genre",sort=True,as_index=False)
total_sold_genre_df.sum()

# Find out the publisher-wise total number of units sold across different regions and the world in the descending order.
total_sold_pub_df=vg_sales_df.groupby(by="Publisher",sort=False,as_index=False)
vg_sales_df.loc[:,['Publisher','Global_Sales']]

# Find out the platform-wise the total number of units sold across different regions and the world in the descending order.
total_sales_plat_df=vg_sales_df.groupby(by="Platform",sort=False,as_index=False)
total_sales_plat_df.sum()

