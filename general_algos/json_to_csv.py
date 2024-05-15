import ast

import pandas as pd
from pandas import json_normalize

import pandas as pd
from pandas import json_normalize

# Assuming your CSV file is named 'your_file.csv'
df = pd.read_csv('output_all.csv')


# Extract keys from the 'experience' column
experience_keys = set()
for entry in df['experience']:
    if pd.notna(entry):
        experience_keys.update(ast.literal_eval(entry)[0].keys())

# Extract keys from the 'education' column
education_keys = set()
for entry in df['education']:
    if pd.notna(entry):
        education_keys.update(ast.literal_eval(entry)[0].keys())

# Combine all unique keys from both 'experience' and 'education'
all_keys = experience_keys.union(education_keys)

# Create new columns for each key
for key in all_keys:
    df[key] = None

# Extract values from the 'experience' column
for i, entry in enumerate(df['experience']):
    if pd.notna(entry):
        experience_data = ast.literal_eval(entry)[0]
        for key, value in experience_data.items():
            df.at[i, key] = value

# Extract values from the 'education' column
for i, entry in enumerate(df['education']):
    if pd.notna(entry):
        education_data = ast.literal_eval(entry)[0]
        for key, value in education_data.items():
            df.at[i, key] = value

# Drop the original 'experience' and 'education' columns
df = df.drop(['experience', 'education'], axis=1)

# Handle missing values if needed
df.fillna(value='N/A', inplace=True)

# Save the DataFrame to a new CSV file
df.to_csv('flattened_data.csv', index=False)

# Read the CSV file
# df = pd.read_csv('output_all.csv')
#
# # Define a function to safely load JSON
# def safe_load_json(s):
#     try:
#         return json.loads(s.replace("'", "\""))
#     except (json.JSONDecodeError, AttributeError):
#         return None
#
# # Flatten "experience" field
# df_exp = pd.json_normalize(df['experience'].apply(safe_load_json).explode(), sep='_', errors='ignore')
# df = pd.concat([df, df_exp], axis=1).drop(columns='experience')
#
# # Flatten "education" field
# df_edu = pd.json_normalize(df['education'].apply(safe_load_json).explode(), sep='_', errors='ignore')
# df = pd.concat([df, df_edu], axis=1).drop(columns='education')
#
# # Save the final DataFrame as a CSV file
# df.to_csv('output_all_final.csv', index=False)
#





#
# # Read JSON data from 'kk.json' file
# with open('amazon.json', 'r') as json_file:
#     data = json_file.read()
#
# # Parse JSON data
# json_data = json.loads(data)
#
# # Convert the JSON data to a pandas DataFrame
# df = pd.DataFrame(json_data)
#
# # Save the DataFrame as a CSV file
# df.to_csv('output.csv', index=False)
