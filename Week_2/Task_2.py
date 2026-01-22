import pandas as pd

df = pd.read_csv("insurance.csv")

central_tendency = pd.DataFrame({
    "Variable": ["age", "bmi", "children", "charges", "sex", "smoker", "region"],
    "Mean": [
        df["age"].mean(),
        df["bmi"].mean(),
        df["children"].mean(),
        df["charges"].mean(),
        "Not applicable",
        "Not applicable",
        "Not applicable"
    ],
    "Median": [
        df["age"].median(),
        df["bmi"].median(),
        df["children"].median(),
        df["charges"].median(),
        "Not applicable",
        "Not applicable",
        "Not applicable"
    ],
    "Mode": [
        df["age"].mode()[0],
        df["bmi"].mode()[0],
        df["children"].mode()[0],
        df["charges"].mode()[0],
        df["sex"].mode()[0],
        df["smoker"].mode()[0],
        df["region"].mode()[0]
    ]
})

print(central_tendency) 

