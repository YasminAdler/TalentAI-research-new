import pandas as pd
import glob
import os
import sys
import logging
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(filename='debug_log.txt', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

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

recommendation_files = {
    "multiclustering": {
        "Statistic_intersection": {
            "with_gender_and_age": "results/with_gender_and_age/Statistic_intersection_multiclustering_recommendations.xlsx",
            "gender_no_age": "results/gender_no_age/Statistic_intersection_multiclustering_recommendations.xlsx",
            "age_no_gender": "results/age_no_gender/Statistic_intersection_multiclustering_recommendations.xlsx",
            "no_age_no_gender": "results/no_age_no_gender/Statistic_intersection_multiclustering_recommendations.xlsx"
        },
        "Statistic_list_frequency": {
            "with_gender_and_age": "results/with_gender_and_age/Statistic_list_frequency_multiclustering_recommendations.xlsx",
            "gender_no_age": "results/gender_no_age/Statistic_list_frequency_multiclustering_recommendations.xlsx",
            "age_no_gender": "results/age_no_gender/Statistic_list_frequency_multiclustering_recommendations.xlsx",
            "no_age_no_gender": "results/no_age_no_gender/Statistic_list_frequency_multiclustering_recommendations.xlsx"
        }
    },
    "standard": {
        "Statistic_intersection": {
            "with_gender_and_age": "results/with_gender_and_age/Statistic_intersection_standard_recommendations.xlsx",
            "gender_no_age": "results/gender_no_age/Statistic_intersection_standard_recommendations.xlsx",
            "age_no_gender": "results/age_no_gender/Statistic_intersection_standard_recommendations.xlsx",
            "no_age_no_gender": "results/no_age_no_gender/Statistic_intersection_standard_recommendations.xlsx"
        },
        "Statistic_list_frequency": {
            "with_gender_and_age": "results/with_gender_and_age/Statistic_list_frequency_standard_recommendations.xlsx",
            "gender_no_age": "results/gender_no_age/Statistic_list_frequency_standard_recommendations.xlsx",
            "age_no_gender": "results/age_no_gender/Statistic_list_frequency_standard_recommendations.xlsx",
            "no_age_no_gender": "results/no_age_no_gender/Statistic_list_frequency_standard_recommendations.xlsx"
        }
    },
    "position_to_applicant": {
        "Statistic_intersection": {
            "with_gender_and_age": "results/position_to_applicant/with_gender_and_age_Statistic_intersection_recommendations.xlsx",
            "gender_no_age": "results/position_to_applicant/gender_no_age_Statistic_intersection_recommendations.xlsx",
            "age_no_gender": "results/position_to_applicant/age_no_gender_Statistic_intersection_recommendations.xlsx",
            "no_age_no_gender": "results/position_to_applicant/no_age_no_gender_Statistic_intersection_recommendations.xlsx"
        },
        "Statistic_list_frequency": {
            "with_gender_and_age": "results/position_to_applicant/with_gender_and_age_Statistic_list_frequency_recommendations.xlsx",
            "gender_no_age": "results/position_to_applicant/gender_no_age_Statistic_list_frequency_recommendations-Yasmin-PC.xlsx",
            "age_no_gender": "results/position_to_applicant/age_no_gender_Statistic_list_frequency_recommendations.xlsx",
            "no_age_no_gender": "results/position_to_applicant/no_age_no_gender_Statistic_list_frequency_recommendations.xlsx"
        }
    }
}

"""
Defenitions: 
i: Represents the position (1 to 13 | 11 ) where the recommended company matched the actual company.
𝑝 = 13 | 11 : The number of possible ranking positions
rel: Is binary (0 or 1), indicating whether there was a match

dcg = 1/log_2(i+1)
idcg = log_2(1+1) = 1

"""

def read_excel_file(file_path):
    try:
        df = pd.read_excel(file_path)
        df = df.iloc[1:]  # Drop the first row if it's an extra header
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

def calculate_ndcg_for_company(recommendations, actual_companies, target_company):
    """Calculate the average NDCG for the target company across all relevant rows."""
    y_pred_mid = {
        i: [str(company).strip().lower() if isinstance(company, str) else str(company)
            for company in recommendations.loc[i, :].fillna('').values[:13]]
        for i in range(len(recommendations))
    }

    y_true_mid = {i: company.strip().lower() for i, company in enumerate(actual_companies)}

    ndcg_scores = []
    
    

    # Calculate NDCG for each row where the target company is the actual company
    for idx, true_company in y_true_mid.items():
        if true_company == target_company:
            if target_company in y_pred_mid.get(idx, []):
                # Find the first match position for the target company
                match_position = y_pred_mid.get(idx, []).index(target_company) + 1
                dcg = 1 / math.log2(match_position + 1)  # DCG based on the first match position
                idcg = 1  # IDCG
                ndcg = dcg / idcg
            else:
                ndcg = 0.0  # No match found, NDCG is zero
            
            ndcg_scores.append(ndcg)

    # Calculate the average NDCG score for the target company across all relevant rows
    average_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    return average_ndcg

def run_ndcg():
    dataset_variation = input("Enter the dataset variation (e.g., with_gender_and_age, gender_no_age, age_no_gender, no_age_no_gender): ")
    algorithm = input("Enter the algorithm type (e.g., multiclustering, standard, position_to_applicant): ")
    distance_function = input("Enter the distance function (e.g., Statistic_intersection, Statistic_list_frequency): ")

    recommendation_file = recommendation_files.get(algorithm, {}).get(distance_function, {}).get(dataset_variation)
    test_set = test_sets.get(dataset_variation)
    company_index = company_indices.get(dataset_variation)

    if not recommendation_file or not os.path.exists(recommendation_file):
        print(f"Recommendation file not found for {algorithm}, {distance_function}, {dataset_variation}.")
        return

    if not test_set or not os.path.exists(test_set):
        print(f"Test set not found for {dataset_variation}.")
        return

    recommendations = read_excel_file(recommendation_file)
    if recommendations.empty:
        print("The recommendations file is empty or could not be loaded properly.")
        return

    test_data = pd.read_csv(test_set)
    actual_companies = test_data.iloc[:, company_index].apply(str.lower).tolist()

    unique_companies = set(actual_companies)

    all_results = []
    for company in unique_companies:
        print(f"Calculating NDCG for company: {company}")
        ndcg = calculate_ndcg_for_company(recommendations, actual_companies, company)
        all_results.append({
            "Algorithm": algorithm,
            "Distance Function": distance_function,
            "Dataset Variation": dataset_variation,
            "Company": company,
            "NDCG": ndcg
        })

    results_df = pd.DataFrame(all_results)
    print(results_df)

    output_dir = "final_measurements"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{dataset_variation}_{algorithm}_{distance_function}_ndcg_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    try:
        run_ndcg()
    except Exception as e:
        logging.error(f"Error running NDCG calculations: {e}")
        print(f"Error running NDCG calculations: {e}")