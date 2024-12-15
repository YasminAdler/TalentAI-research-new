import sys
import logging
import os
import csv
import pandas as pd
import ast  # To safely evaluate stringified arrays

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append("../TalentAI-research-new-last-update")
sys.path.append("C:/Users/adler/OneDrive/Talent.AI/TalentAI-research-new-last-update/.venv/Lib/site-packages")
sys.path.append(os.path.join(script_dir, "../clusters_info"))

# File paths
unbiasing_file_path = 'final_measurements/no_age_no_gender_intersection_unbiasing_per_cluster.xlsx'
clusters_file_path = 'final_measurements/clusters_info/clusters_no_age_no_gender_Statistic_intersection_train_model.csv'

# Ensure files exist before proceeding
if not os.path.exists(unbiasing_file_path):
    raise FileNotFoundError(f"File not found: {unbiasing_file_path}")
if not os.path.exists(clusters_file_path):
    raise FileNotFoundError(f"File not found: {clusters_file_path}")

# Load datasets
nearest_cluster_data = pd.read_excel(unbiasing_file_path)
clusters_data = pd.read_csv(clusters_file_path)

# Step 1: Get the nearest cluster ID column name
nearest_cluster_column = next(
    (col for col in nearest_cluster_data.columns if "Nearest Cluster" in col), None
)
if not nearest_cluster_column:
    raise KeyError("Column for 'Nearest Cluster' not found in nearest_cluster_data.")

# Step 2: Get all nearest cluster IDs in order
nearest_clusters = nearest_cluster_data[nearest_cluster_column].dropna().astype(int).tolist()

# Step 3: Define a function to parse and check "Row Data" for "Adobe"
def contains_adobe(row_data):
    try:
        # Safely evaluate the stringified array into a Python list
        parsed_data = ast.literal_eval(row_data)
        # Check columns 9, 10, and 11 (index 8, 9, 10 in 0-based indexing)
        return any("uber" in str(item).lower() for item in parsed_data[9:12]) #  adobe amazon apple facebook google ibm microsoft nvidia oracle salesforce tesla twitter uber-com
    except (ValueError, IndexError, SyntaxError):
        # Handle parsing errors or missing data gracefully
        return False

# Step 4: Process each cluster and store Adobe counts in a dictionary
cluster_adobe_counts = {}
for cluster_id in set(nearest_clusters):  # Process only unique cluster IDs
    filtered_clusters = clusters_data[clusters_data['Cluster ID'] == cluster_id]
    if filtered_clusters.empty:
        cluster_adobe_counts[cluster_id] = 0
        continue
    
    adobe_count = filtered_clusters['Row Data'].apply(contains_adobe).sum()
    cluster_adobe_counts[cluster_id] = adobe_count

# Step 5: Generate the output for each `Nearest Cluster` in order
output = [cluster_adobe_counts[cluster_id] for cluster_id in nearest_clusters]

# Step 6: Print the output in the required format
print("\n".join(map(str, output)))
