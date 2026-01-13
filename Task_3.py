import pandas as pd
df = pd.read_csv("cars.csv")
print(df.head())
speed_values = df["SP"].tolist()
print(speed_values[:5])
MIN_SP = 100
MAX_SP = 135

total_invalid = 0
total_valid = 0

for value in speed_values:
    if value < MIN_SP or value > MAX_SP:
        print(value, "Invalid")
        print(value)
        total_invalid += 1        
    else:
        print(value, "Valid")
        print(value)
        total_valid += 1

print('-= FINAL SUMMARY =-')
print('Total valid: ',total_valid)
print('Total invalid: ',total_invalid)

