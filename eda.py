# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset (force encoding to avoid UnicodeDecodeError)
df = pd.read_csv("car_purchasing.csv", encoding='latin1')

# Show first few rows
print("First 5 rows:\n", df.head())

# Dataset info
print("\nDataset Info:\n")
print(df.info())

# Statistical summary
print("\nStatistical Summary:\n", df.describe())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Select only numeric columns for correlation matrix
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Plot correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
