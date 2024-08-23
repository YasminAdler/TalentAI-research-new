import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_dir, "datasets", "saved_sets/train_vectors.pkl.csv")

try:
  df = pd.read_csv(csv_file_path, header=None)
  print("Original dataset loaded")
  print(csv_file_path)

  # Define column indices (assuming zero-based indexing)
  columns_to_keep = list(range(len(df.columns)))  # All column indices initially

  # 1. Dataset with age and no gender (exclude columns 4 and 5)
  columns_to_drop = [4, 5]
  df_age_no_gender = df.drop(columns_to_drop, axis=1)
  df_age_no_gender.to_csv(os.path.join(script_dir, "employes_age_no_gender.csv"), index=False, header=False)
  print("Dataset with age and no gender saved (columns 4 & 5 excluded)")
  columns_to_keep = [col for col in columns_to_keep if col not in columns_to_drop]  # Update remaining columns

  # 2. Dataset with gender and no age (exclude column 3)
  columns_to_drop = [3]
  df_gender_no_age = df.drop(columns_to_drop, axis=1)
  df_gender_no_age.to_csv(os.path.join(script_dir, "employes_gender_no_age.csv"), index=False, header=False)
  print("Dataset with gender and no age saved (column 3 excluded)")
  columns_to_keep = [col for col in columns_to_keep if col not in columns_to_drop]  # Update remaining columns

  # 3. Dataset with no age and no gender (exclude columns 3, 4, and 5)
  columns_to_drop = [3, 4, 5]
  df_no_age_no_gender = df.drop(columns_to_drop, axis=1)
  df_no_age_no_gender.to_csv(os.path.join(script_dir, "employes_no_age_no_gender.csv"), index=False, header=False)
  print("Dataset with no age and no gender saved (columns 3, 4 & 5 excluded)")

  # 4. Dataset with gender and age (original dataset, all columns)
  df.to_csv(os.path.join(script_dir, "employes_with_gender_and_age.csv"), index=False, header=False)
  print("Original dataset with gender and age saved")

except FileNotFoundError:
  print("Error: CSV file not found. Please check the path.")
