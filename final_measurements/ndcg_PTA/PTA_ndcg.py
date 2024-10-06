import pandas as pd
import os
import sys
import logging
import math
import ast

# Set up logging and paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(filename='debug_log.txt', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

# Mapping of company indices and test sets
company_indices = {
    "with_gender_and_age": 11,
    "gender_no_age": 10,
    "age_no_gender": 10,
    "no_age_no_gender": 9
}

test_sets = {
    "with_gender_and_age": "datasets/test_with_gender_and_age.csv",
    "gender_no_age": "datasets/test_gender_no_age.csv",
    "age_no_gender": "datasets/test_age_no_gender.csv",
    "no_age_no_gender": "datasets/test_no_age_no_gender.csv"
}

# Recommendation files structure
recommendation_files = {
    "position_to_applicant": {
        "Statistic_intersection": {
            "with_gender_and_age": "results/position_to_applicant/with_gender_and_age_Statistic_intersection_recommendations.xlsx",
            "gender_no_age": "results/position_to_applicant/gender_no_age_Statistic_intersection_recommendations.xlsx",
            "age_no_gender": "results/position_to_applicant/age_no_gender_Statistic_intersection_recommendations.xlsx",
            "no_age_no_gender": "results/position_to_applicant/no_age_no_gender_Statistic_intersection_recommendations.xlsx"
        },
        "Statistic_list_frequency": {
            "with_gender_and_age": "results/position_to_applicant/with_gender_and_age_Statistic_list_frequency_recommendations.xlsx",
            "gender_no_age": "results/position_to_applicant/gender_no_age_Statistic_list_frequency_recommendations.xlsx",
            "age_no_gender": "results/position_to_applicant/age_no_gender_Statistic_list_frequency_recommendations.xlsx",
            "no_age_no_gender": "results/position_to_applicant/no_age_no_gender_Statistic_list_frequency_recommendations.xlsx"
        }
    }
}

# Function to read Excel file
def read_excel_file(file_path):
    try:
        df = pd.read_excel(file_path)
        df = df.iloc[1:]  # Skip the header row
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

# Function to calculate nDCG
def calculate_ndcg_for_company(recommendations, test_set, target_company, company_index):
    """
    Calculates the average nDCG for a given company across multiple queries (positions).

    - For each query (position) that belongs to the target company, 
      we calculate nDCG by comparing the target company's applicants.
    - We only look at odd-numbered columns (applicants), i.e., columns 1, 3, 5, ..., 21.
    - p = 11 (we are considering 11 applicants per query).
    - Relevance (rel) is 1 if the applicant's company matches the target company, else 0.
    
    """
    ndcg_scores = []
    
    for i in range(len(recommendations)):
        position_info = test_set.iloc[i, :]  # The position from the test set
        position_company = position_info.iloc[company_index].strip().lower()  
        
        if position_company == target_company:  # Only proceed for target company positions
            dcg = 0  
            relevances = []  # Store relevances to compute IDCG later
            for j, applicant_column in enumerate(range(1, 23, 2)):
                applicant_info = recommendations.iloc[i, applicant_column]
                
                if isinstance(applicant_info, str) and applicant_info.startswith("[") and applicant_info.endswith("]"):
                    applicant_info = ast.literal_eval(applicant_info)
                
                if isinstance(applicant_info, list):
                    applicant_company = str(applicant_info[company_index]).strip().lower()
                else:
                    applicant_company = str(applicant_info).strip().lower()

                # Relevance (rel) is 1 if the applicant's company matches the target company
                rel = 1 if applicant_company == target_company else 0
                relevances.append(rel) 
                if rel == 1:
                    dcg += rel / math.log2(j + 2)  # i starts at 0, so log2(j+2)
            
            # Ideal DCG is computed with the ideal order of relevances (all relevant items at the top)
            idcg = sum(1 / math.log2(k + 2) for k in range(min(len(relevances), relevances.count(1))))
            
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)

    average_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    return average_ndcg

def run_ndcg_for_all_combinations():
    dataset_variations = ["with_gender_and_age", "gender_no_age", "age_no_gender", "no_age_no_gender"]
    distance_functions = ["Statistic_intersection", "Statistic_list_frequency"]

    for dataset_variation in dataset_variations:
        for distance_function in distance_functions:
            recommendation_file = recommendation_files.get('position_to_applicant', {}).get(distance_function, {}).get(dataset_variation)
            test_set_file = test_sets.get(dataset_variation)
            company_index = company_indices.get(dataset_variation)

            if not recommendation_file or not os.path.exists(recommendation_file):
                print(f"Recommendation file not found for {distance_function}, {dataset_variation}.")
                continue

            if not test_set_file or not os.path.exists(test_set_file):
                print(f"Test set not found for {dataset_variation}.")
                continue

            recommendations = read_excel_file(recommendation_file)
            test_set = pd.read_csv(test_set_file)

            actual_companies = test_set.iloc[:, company_index].apply(str.lower).tolist()
            unique_companies = set(actual_companies)

            all_results = []

            for target_company in unique_companies:
                average_ndcg = calculate_ndcg_for_company(recommendations, test_set, target_company, company_index)
                all_results.append({
                    "Dataset Variation": dataset_variation,
                    "Distance Function": distance_function,
                    "Target Company": target_company,
                    "Average NDCG": average_ndcg
                })

            results_df = pd.DataFrame(all_results)
            print(results_df)

            # Save results for each combination to individual CSV files
            output_dir = "final_measurements"
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/{dataset_variation}_{distance_function}_ndcg_PTA.csv"
            results_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")

if __name__ == "__main__":
    try:
        run_ndcg_for_all_combinations()
    except Exception as e:
        logging.error(f"Error running NDCG calculations: {e}")
        print(f"Error running NDCG calculations: {e}")
