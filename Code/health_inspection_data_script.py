import numpy as np
import pandas as pd

raw_df = pd.read_csv("../Datasets/Food_Inspections_20250413.csv")
print(raw_df.head())
print(raw_df.info())

# Print the shape of the DataFrame
print("Shape of the DataFrame:", raw_df.shape)

# Print the count of missing values in each column
missing_data = raw_df.isnull().sum()
print("Missing data counts:\n", missing_data[missing_data > 0])

# Print the unique value counts for each column
unique_counts = raw_df.nunique()
print("Unique value counts:\n", unique_counts)

# Print descriptive statistics for numerical columns
print("Descriptive statistics for numerical columns:\n", raw_df.describe())

# Print the data types of each column
print("Data types of each column:\n", raw_df.dtypes)
