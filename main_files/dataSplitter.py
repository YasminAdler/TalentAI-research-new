import pickle
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import csv

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_dataset(file_path):
    csv_data = []
    with open(file_path, "r", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            csv_data.append(row)
    vectors = [np.array(f, dtype=object) for f in csv_data]
    return vectors

def split_and_save_dataset(dataset_path, train_filename, test_filename):
    vectors = load_dataset(dataset_path)
    train_vectors, test_vectors = train_test_split(vectors, test_size=0.20, random_state=42)
    
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(train_filename), exist_ok=True)
    
    save_to_pickle(train_vectors, train_filename)
    save_to_pickle(test_vectors, test_filename)
    print(f"Train and test sets saved to {train_filename} and {test_filename}")
    return train_vectors, test_vectors

def save_dataset_variations(df, dataset_dir, prefix):
    # Define column indices (assuming zero-based indexing)
    columns_to_keep = list(range(len(df.columns)))  # All column indices initially

    # 1. Dataset with age and no gender (exclude columns 4 and 5)
    columns_to_drop = [4, 5]
    df_age_no_gender = df.drop(columns_to_drop, axis=1)
    df_age_no_gender.to_csv(os.path.join(dataset_dir, f"{prefix}_gender_no_age.csv"), index=False, header=False)
    print("Dataset with age and no gender saved (columns 4 & 5 excluded)")
    columns_to_keep = [col for col in columns_to_keep if col not in columns_to_drop]  # Update remaining columns

    # 2. Dataset with gender and no age (exclude column 3)
    columns_to_drop = [3]
    df_gender_no_age = df.drop(columns_to_drop, axis=1)
    df_gender_no_age.to_csv(os.path.join(dataset_dir, f"{prefix}_age_no_gender.csv"), index=False, header=False)
    print("Dataset with gender and no age saved (column 3 excluded)")
    columns_to_keep = [col for col in columns_to_keep if col not in columns_to_drop]  # Update remaining columns

    # 3. Dataset with no age and no gender (exclude columns 3, 4, and 5)
    columns_to_drop = [3, 4, 5]
    df_no_age_no_gender = df.drop(columns_to_drop, axis=1)
    df_no_age_no_gender.to_csv(os.path.join(dataset_dir, f"{prefix}_no_age_no_gender.csv"), index=False, header=False)
    print("Dataset with no age and no gender saved (columns 3, 4 & 5 excluded)")

    # 4. Dataset with gender and age (original dataset, all columns)
    df.to_csv(os.path.join(dataset_dir, f"{prefix}_with_gender_and_age.csv"), index=False, header=False)
    print("Original dataset with gender and age saved")

def main():
    dataset_path = "datasets/employes_flat_version.csv"
    train_vectors, test_vectors = split_and_save_dataset(dataset_path, "saved_sets/train_vectors.pkl", "saved_sets/test_vectors.pkl")

    # Convert train and test vectors to DataFrames for easier manipulation
    df_train = pd.DataFrame(train_vectors)
    df_test = pd.DataFrame(test_vectors)

    # Create directory for dataset variations if it doesn't exist
    dataset_dir = "datasets"
    os.makedirs(dataset_dir, exist_ok=True)

    # Save dataset variations for both train and test sets
    save_dataset_variations(df_train, dataset_dir, "train")
    save_dataset_variations(df_test, dataset_dir, "test")

if __name__ == "__main__":
    main()
