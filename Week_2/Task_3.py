import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("insurance.csv")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())

# Identify numeric and categorical columns
numeric_columns = ["age", "bmi", "children", "charges"]
categorical_columns = ["sex", "smoker", "region"]

# (A) Histograms (distribution)
for col in numeric_columns:
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x=col, kde=True, bins=30)
    plt.title(f"Histogram of {col}")
    plt.tight_layout()
    plt.show() 
# (B) Boxplots (outliers)
for col in numeric_columns:
    plt.figure(figsize=(7, 3))
    sns.boxplot(data=df, x=col)
    plt.title(f"Boxplot of {col} (Outlier Check)")
    plt.tight_layout()
    plt.show()

# (A) Count plots (category frequency)
for col in categorical_columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col)
    plt.title(f"Count Plot of {col}")
    plt.tight_layout()
    plt.show() 


