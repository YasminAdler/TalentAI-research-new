import sys
import logging
import os
import pickle
import csv
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append("../TalentAI-research-new-last-update")
sys.path.append("C:/Users/adler/OneDrive/Talent.AI/TalentAI-research-new-last-update/.venv/Lib/site-packages")
sys.path.append(os.path.join(script_dir, "../clusters_info"))


# File paths
clusters_info_path = 'statistic_regular_algo/clusters_info/clusters_with_gender_and_age_Statistic_list_frequency_train_model.csv'
recommendations_path = 'C:/Users/adler/Downloads/Statistic_intersection_multiclustering_recommendations.xlsx - in.csv'

# Load data into DataFrames
try:
    clusters_info = pd.read_csv(clusters_info_path)
    recommendations = pd.read_csv(recommendations_path)
except FileNotFoundError as e:
    print(f"File not found: {e}")
    sys.exit(1)  # Exit the script if a file is not found

# Function to find the nearest cluster for a query
import re

# Function to find the nearest cluster for a query
def find_nearest_cluster(query_name):
    # Escape special characters and ensure string columns
    query_name = re.escape(query_name)
    cluster_match = clusters_info[
        clusters_info.iloc[:, 1].astype(str).str.contains(query_name, case=False, na=False, regex=True)
    ]
    if not cluster_match.empty:
        return cluster_match.iloc[0, 0]  # Return the cluster ID
    return None

import ast  # To safely parse the string representation of lists

# Function to get the rank of a company for a query
def get_company_rank(query_name, company_name):
    query_data = recommendations[
        recommendations.iloc[:, 1].astype(str).str.contains(query_name, case=False, na=False, regex=False)
    ]
    if not query_data.empty:
        # Parse the query list from column 1
        query_list_str = query_data.iloc[0, 1]  # Assuming the list is in column 1
        try:
            query_list = ast.literal_eval(query_list_str)  # Convert string to Python list
            # Check if the company name exists at positions 11 or 12
            if len(query_list) > 11 and (query_list[11] == company_name or (len(query_list) > 12 and query_list[12] == company_name)):
                return query_data.iloc[0, 0]  # Return rank from column 0 (or adjust as needed)
        except (ValueError, SyntaxError):
            pass  # Handle parsing errors gracefully
    return None


import ast  # For safely evaluating the stringified lists

def count_test_records_in_cluster(cluster_id, company_name):
    # Filter the rows for the given cluster_id
    cluster_records = clusters_info[clusters_info.iloc[:, 0] == cluster_id]
    count = 0

    # Iterate through the filtered rows
    for _, row in cluster_records.iterrows():
        row_data = row['Row Data']  # Access the Row Data column
        try:
            # Parse the stringified list in 'Row Data'
            parsed_data = ast.literal_eval(row_data)
            # Count occurrences of the company name in the parsed list
            count += parsed_data.count(company_name)
        except (ValueError, SyntaxError):
            print(f"Error parsing row: {row_data}")
            continue  # Skip rows with invalid data

    return count


# Generate the desired output
output_data = []

for _, row in recommendations.iterrows():
    query_name = row.iloc[0]
    nearest_cluster = find_nearest_cluster(query_name)
    
    # Collect data for all companies
    companies = ['Amazon', 'Amazon', 'Apple', 'Facebook', 'Google', 'IBM', 
                 'Microsoft', 'Nvidia', 'Oracle', 'Salesforce', 'Tesla', 
                 'Twitter', 'Uber']
    record = {
        'Query': query_name,
        'Nearest Cluster': nearest_cluster
    }
    
    for company in companies:
        rank = get_company_rank(query_name, company)
        test_records = count_test_records_in_cluster(nearest_cluster, company)
        record[f'{company} Rank'] = rank
        record[f'Test Records in Cluster for {company}'] = test_records
    
    output_data.append(record)

# Convert to DataFrame and save to CSV
output_df = pd.DataFrame(output_data)
output_csv_path = 'output_file.csv'
output_df.to_csv(output_csv_path, index=False)

print(f"Output saved to {output_csv_path}")
