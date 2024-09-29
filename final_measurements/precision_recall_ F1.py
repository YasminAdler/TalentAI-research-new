import pandas as pd
import glob
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
Logic Explanation:

This algorithm calculates Precision, Recall, and F1 Score for each target company using the following updated definitions:

- Positive: A prediction is considered positive if recommendation['Company_1'] == target company.
- Negative: A prediction is negative if recommendation['Company_1'] != target company.

- True Positive (TP): The recommendation correctly predicts the target company, and the actual company in the test set is also the target.
- False Positive (FP): The recommendation predicts the target company, but the actual company in the test set is not the target.
- True Negative (TN): The recommendation does not predict the target company, and the actual company in the test set is also not the target.
- False Negative (FN): The actual company in the test set is the target company, but the recommendation does not predict it.

Precision measures the ratio of correct positive predictions (TP) to all positive predictions (TP + FP). Recall measures the ratio of correct positive predictions (TP) to all actual positives (TP + FN). F1 Score balances Precision and Recall using their harmonic mean.

Example:

True Positive (TP): The actual label is Google, and the recommendation is also Google.
False Positive (FP): The actual label is not Google, but the recommendation is Google.
True Negative (TN): The actual label is not Google, and the recommendation is also not Google.
False Negative (FN): The actual label is Google, but the recommendation is not Google. ## to ask if this is definit?

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

def calculate_tp_fp_tn_fn(recommendations, test_set, target_company, company_index):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(recommendations)):
        predicted_company = recommendations['Company_1'].iloc[i].strip().lower()
        actual_company = test_set.iloc[i, company_index].strip().lower()

        if predicted_company == target_company and actual_company == target_company:
            TP += 1
        elif predicted_company == target_company and actual_company != target_company:
            FP += 1
        elif predicted_company != target_company and actual_company != target_company:
            TN += 1
        elif predicted_company != target_company and actual_company == target_company:
            FN += 1

    return TP, FP, TN, FN

def calculate_metrics(TP, FP, TN, FN):
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
    return precision, recall, f1

def run_metrics():
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
        print(f"Calculating for company: {target_company}")
        TP, FP, TN, FN = calculate_tp_fp_tn_fn(recommendations, test_set, target_company, company_index)
        precision, recall, f1 = calculate_metrics(TP, FP, TN, FN)
        
        all_results.append({
            "Algorithm": algorithm,
            "Distance Function": distance_function,
            "Dataset Variation": dataset_variation,
            "Target Company": target_company,
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        })

    results_df = pd.DataFrame(all_results)
    print(results_df)

    output_dir = "final_measurements"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{dataset_variation}_{algorithm}_{distance_function}_precision_recall_F1.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    try:
        run_metrics()
    except Exception as e:
        logging.error(f"Error running metrics calculations: {e}")
        print(f"Error running metrics calculations: {e}")