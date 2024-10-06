import pandas as pd
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
    }
}

"""
Defenitions: 
i: Represents the position (1 to 13) where the recommended company matched the actual company.
𝑝 = 13 : The number of possible ranking positions
rel: Is binary (0 or 1), indicating whether there was a match

dcg = 1/log_2(i+1)
idcg = log_2(1+1) = 1

"""
def read_excel_file(file_path):
    try:
        df = pd.read_excel(file_path)
        df = df.iloc[1:]  # Skip the header row in the recommendations set
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

def calculate_ndcg_for_company(recommendations, actual_companies, target_company):    
    ndcg_scores = []
    
    for row_idx, actual_company in enumerate(actual_companies):
        if actual_company == target_company:  # Only calculate for rows that match the target company
            # Fetch recommendation row for the current company
            adjusted_row_idx = row_idx-1 
            recommended_companies = recommendations.iloc[adjusted_row_idx, [i for i in range(1, 25, 2)]].values.tolist()
            
            # print(recommended_companies)
            
            dcg = 0
            for i, recommended_company in enumerate(recommended_companies):
                rel = 1 if recommended_company == actual_company else 0
                if rel == 1:
                    dcg += rel / math.log2(i + 2)  # i starts at 0, so log2(i+2)

            idcg = 1  # Ideal DCG (since relevance = 1 for the best possible position)
            
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
    
    average_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    return average_ndcg

def run_ndcg():
    dataset_variation = input("Enter the dataset variation (e.g., with_gender_and_age, gender_no_age, age_no_gender, no_age_no_gender): ")
    algorithm = input("Enter the algorithm type (e.g., multiclustering, standard): ")
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

    test_data = pd.read_csv(test_set, header=None)

    # Extract actual companies using the provided company index
    actual_companies = test_data.iloc[:, company_index].apply(str.lower).tolist()
    # print("actual_companies", actual_companies)
    unique_companies = set(actual_companies)

    all_results = []
    for company in unique_companies:
        print(f"Calculating NDCG for company: {company}")
        average_ndcg = calculate_ndcg_for_company(recommendations, actual_companies, company)
        all_results.append({
            "Algorithm": algorithm,
            "Distance Function": distance_function,
            "Dataset Variation": dataset_variation,
            "Company": company,
            "Average NDCG": average_ndcg
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