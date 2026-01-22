import pandas as pd

df = pd.read_csv("insurance.csv")

smokers = df[df["smoker"] == "yes"]
non_smokers = df[df["smoker"] == "no"]

# statistics
summary = pd.DataFrame({
    "Group": ["Smokers", "Non-Smokers"],
    "Mean Charges": [smokers["charges"].mean(), non_smokers["charges"].mean()],
    "Median Charges": [smokers["charges"].median(), non_smokers["charges"].median()]
})

print(summary)

