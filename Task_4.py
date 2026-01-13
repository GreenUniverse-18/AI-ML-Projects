import pandas as pd
df = pd.read_csv("cars.csv")
print(df.head())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load data
df = pd.read_csv("cars.csv")

# Select SP and WT
sp = df["SP"]
wt = df["WT"]

# Basic statistics
mean_sp = sp.mean()
median_sp = sp.median()
mode_sp = sp.mode()[0]
std_sp = sp.std()
rank_sp = sp.rank()

print("SP Mean:", mean_sp)
print("SP Median:", median_sp)
print("SP Mode:", mode_sp)
print("SP Standard Deviation:", std_sp)
print("SP Rank (first 10):\n", rank_sp.head(10))

# Correlation
corr = df[["SP", "WT"]].corr()
print("\nCorrelation Matrix:\n", corr)

# Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap: SP vs WT")
plt.show()

# LDA requires class labels, so create a binary class from SP
df["SP_Class"] = np.where(df["SP"] >= df["SP"].median(), 1, 0)

X = df[["SP", "WT"]]
y = df["SP_Class"]

lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
lda_transformed = lda.transform(X)

print("\nLDA Coefficients:", lda.coef_)
print("LDA Intercept:", lda.intercept_)
print("LDA Transformed Data (first 10):\n", lda_transformed[:10])
