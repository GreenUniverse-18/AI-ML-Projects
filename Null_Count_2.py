Null_Counts = [0, 2, 15, 1, 0, 8]

count = 0
problematic_columns = 0

for n in Null_Counts:
    count = count + 1
    print("Total number of columns:", count, n)
    if n > 5:
        print("Null count > than 5 is", count, "and number is:", n)
        problematic_columns = problematic_columns + 1
print("Total problematic_columns:", problematic_columns) 


   

