import pandas as pd
import os
import sys
import logging

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
# Logic Explanation for Accuracy:

This version of the algorithm calculates Accuracy by looking at multiple recommendation columns based on the value of `x`. 
For example, if `x=3`, the algorithm will check recommendation columns 1, 3, and 5 to see if the target company appears.

- True Positive (TP): The target company appears in one of the x-columns, and the actual company in the test set is also the target.
- False Positive (FP): The target company appears in one of the x-columns, but the actual company in the test set is not the target.
- True Negative (TN): The target company does not appear in any of the x-columns, and the actual company in the test set is also not the target.
- False Negative (FN): The target company does not appear in any of the x-columns, but the actual company in the test set is the target.
"""
def read_excel_file(file_path):
    try:
        df = pd.read_excel(file_path)
        df = df.iloc[1:]  
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

def get_x_columns(x):
    if x == 1:
        return [1]
    elif x == 3:
        return [1, 3, 5]
    elif x == 5:
        return [1, 3, 5, 7, 9]
    elif x == 13:
        return [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    else:
        raise ValueError("Invalid value for x. Must be one of [1, 3, 5, 13]")

def calculate_tp_fp_tn_fn_x(recommendations, test_set, target_company, company_index, x):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    x_columns = get_x_columns(x)

    for i in range(len(recommendations)):
        actual_company = test_set.iloc[i, company_index].strip().lower()

        predicted_companies = [recommendations.iloc[i, col].strip().lower() for col in x_columns]

        if target_company in predicted_companies and actual_company == target_company:
            TP += 1
        elif target_company in predicted_companies and actual_company != target_company:
            FP += 1
        elif target_company not in predicted_companies and actual_company != target_company:
            TN += 1
        elif target_company not in predicted_companies and actual_company == target_company:
            FN += 1

    return TP, FP, TN, FN

def calculate_accuracy(TP, FP, TN, FN):
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0.0
    return accuracy

def run_accuracy():
    dataset_variation = input("Enter the dataset variation (e.g., with_gender_and_age, gender_no_age, age_no_gender, no_age_no_gender): ")
    algorithm = input("Enter the algorithm type (e.g., multiclustering, standard, position_to_applicant): ")
    distance_function = input("Enter the distance function (e.g., Statistic_intersection, Statistic_list_frequency): ")

    recommendation_file = recommendation_files.get(algorithm, {}).get(distance_function, {}).get(dataset_variation)
    test_set_file = test_sets.get(dataset_variation)
    company_index = company_indices.get(dataset_variation)

    if not recommendation_file or not os.path.exists(recommendation_file):
        print(f"Recommendation file not found for {algorithm}, {distance_function}, {dataset_variation}.")
        return

    if not test_set_file or not os.path.exists(test_set_file):
        print(f"Test set not found for {dataset_variation}.")
        return

    recommendations = read_excel_file(recommendation_file)
    test_set = pd.read_csv(test_set_file)
    actual_companies = test_set.iloc[:, company_index].apply(str.lower).tolist()
    
    unique_companies = set(actual_companies)

    all_results = []

    for target_company in unique_companies:
        for x_value in [1, 3, 5, 13]:  # Loop over x = 1, 3, 5, 13
            print(f"Calculating accuracy for company: {target_company} with x = {x_value}")
            TP, FP, TN, FN = calculate_tp_fp_tn_fn_x(recommendations, test_set, target_company, company_index, x_value)
            accuracy = calculate_accuracy(TP, FP, TN, FN)
            
            all_results.append({
                "Algorithm": algorithm,
                "Distance Function": distance_function,
                "Dataset Variation": dataset_variation,
                "Target Company": target_company,
                "x": x_value,
                "TP": TP,
                "FP": FP,
                "TN": TN,
                "FN": FN,
                "Accuracy": accuracy
            })

    results_df = pd.DataFrame(all_results)
    print(results_df)

    output_dir = "final_measurements"
    os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist
    output_file = f"{output_dir}/{dataset_variation}_{algorithm}_{distance_function}_accuracy.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    try:
        run_accuracy()
    except Exception as e:
        logging.error(f"Error running accuracy calculations: {e}")
        print(f"Error running accuracy calculations: {e}")