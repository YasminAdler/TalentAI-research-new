import pandas as pd
import ast  # Importing ast module to safely evaluate literal expressions

# Replace 'your_file.csv' with the actual path to your CSV file
file_path = '../datasets/WineCopy.txt'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Assuming the 4th index column is named 'column_name', replace it with the actual column name
column_name = 'mamama'
print(df)
# Convert the string representation of the list to an actual list using ast.literal_eval
df[column_name] = df[column_name].apply(ast.literal_eval)

# Calculate the average list length
average_list_length = df[column_name].apply(len).mean()

print(f"The average list length in the '{column_name}' column is: {average_list_length}")
