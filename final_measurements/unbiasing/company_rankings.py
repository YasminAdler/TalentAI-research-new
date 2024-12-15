import sys
import os
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


# Load the uploaded Excel file
file_path = 'results/with_gender_and_age/Statistic_intersection_standard_recommendations (2).xlsx'
data = pd.read_excel(file_path)

# Extract the relevant columns for ranking Amazon
columns = data.columns
company_columns = [col for col in columns if "Company" in col]

# Create a function to find Amazon's rank(s) in each row
def find_ranks(row):
    ranks = [str(idx) for idx, company in enumerate(company_columns, start=1) if row[company] == 'uber-com'] # adobe amazon apple facebook google ibm microsoft nvidia oracle salesforce tesla-motors twitter uber-com
    return ", ".join(ranks) if ranks else "None"  # Join ranks with commas, or return "None"

# Apply the function to the data
data['Ranks'] = data.apply(find_ranks, axis=1)

# Prepare the output in the desired format
formatted_output = "\n".join(data['Ranks'])

# Print the formatted output
print(formatted_output)
