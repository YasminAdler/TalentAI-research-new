import pandas as pd
import os
import sys
import logging
import ast

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
            "gender_no_age": "results/position_to_applicant/gender_no_age_Statistic_list_frequency_recommendations.xlsx",
            "age_no_gender": "results/position_to_applicant/age_no_gender_Statistic_list_frequency_recommendations.xlsx",
            "no_age_no_gender": "results/position_to_applicant/no_age_no_gender_Statistic_list_frequency_recommendations.xlsx"
        }
    }
}

def read_excel_file(file_path):
    try:
        df = pd.read_excel(file_path)
        df = df.iloc[1:]  # Skip header row
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

def count_women_in_top_x(recommendations, gender_index, x_values):
    """
    Count the number of women in the top x places (1, 3, 5, 11/13) for each query/position.
    """
    women_counts = {x: 0 for x in x_values}

    for i in range(len(recommendations)):
        for x in x_values:
            women_count = 0
            for j in range(min(x, len(recommendations.columns[1::2]))):
                applicant_info = recommendations.iloc[i, 1 + 2 * j]
                
                if isinstance(applicant_info, str) and applicant_info.startswith("[") and applicant_info.endswith("]"):
                    try:
                        applicant_info = ast.literal_eval(applicant_info)
                    except Exception as e:
                        print(f"Error parsing applicant info: {e}")
                        continue
                
                if isinstance(applicant_info, list):
                    gender = applicant_info[gender_index].strip().lower()
                    if gender == 'female':
                        women_count += 1
            
            women_counts[x] += women_count
    
    return women_counts

def run_check_for_women():
    all_results = []
    
    for algorithm in recommendation_files.keys():
        for distance_function in recommendation_files[algorithm].keys():
            for dataset_variation in recommendation_files[algorithm][distance_function].keys():
                recommendation_file = recommendation_files[algorithm][distance_function][dataset_variation]
                test_set_file = test_sets.get(dataset_variation)
                company_index = company_indices.get(dataset_variation)
                gender_index = 3  # Assuming gender is in index 3

                if not recommendation_file or not os.path.exists(recommendation_file):
                    print(f"Recommendation file not found: {recommendation_file}")
                    continue

                if not test_set_file or not os.path.exists(test_set_file):
                    print(f"Test set not found: {test_set_file}")
                    continue

                recommendations = read_excel_file(recommendation_file)
                if recommendations.empty:
                    print(f"Recommendations file is empty: {recommendation_file}")
                    continue

                x_values = [1, 3, 5, 11] if algorithm == "position_to_applicant" else [1, 3, 5, 13]
                women_counts = count_women_in_top_x(recommendations, gender_index, x_values)

                for x, count in women_counts.items():
                    all_results.append({
                        "Algorithm": algorithm,
                        "Distance Function": distance_function,
                        "Dataset Variation": dataset_variation,
                        "x": x,
                        "Number of Women": count
                    })

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    print(results_df)

    # Save results to a CSV file
    output_dir = "results/women_counts"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"women_counts_all_algorithms.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    try:
        run_check_for_women()
    except Exception as e:
        logging.error(f"Error running women check: {e}")
        print(f"Error running women check: {e}")
