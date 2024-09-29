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

"""
# Logic Explanation for Accuracy:

For the position-to-applicant algorithm, we calculate Accuracy by looking at a varying number of applicants (1, 3, 5, or 11). 
Each time, the algorithm checks recommendation columns 1, 3, 5, ... corresponding to applicants.

- True Positive (TP): The target company appears in the recommendations (for applicants 1 to n), and the actual company in the test set is also the target.
- False Positive (FP): The target company appears in the recommendations (for applicants 1 to n), but the actual company in the test set is not the target.
- True Negative (TN): The target company does not appear in the recommendations, and the actual company in the test set is also not the target.
- False Negative (FN): The target company does not appear in the recommendations, but the actual company in the test set is the target.
"""

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

def get_applicant_columns(num_applicants):
    # Define the columns based on how many applicants you want to consider (1, 3, 5, or 11)
    if num_applicants == 1:
        return [1]
    elif num_applicants == 3:
        return [1, 3, 5]
    elif num_applicants == 5:
        return [1, 3, 5, 7, 9]
    elif num_applicants == 11:
        return [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    else:
        raise ValueError("Invalid number of applicants. Choose 1, 3, 5, or 11.")

def extract_company_from_applicant(applicant_info, company_index):
    # Check if applicant_info is a string representation of a list and safely evaluate it
    if isinstance(applicant_info, str) and applicant_info.startswith("[") and applicant_info.endswith("]"):
        try:
            applicant_info = ast.literal_eval(applicant_info)  # Convert the string into a list
        except (ValueError, SyntaxError):
            print(f"Error parsing applicant_info: {applicant_info}")
            return ''

    # Extract the company from Applicant_1's information based on company_index
    if isinstance(applicant_info, list) and len(applicant_info) > company_index:
        return str(applicant_info[company_index]).strip().lower()
    return ''  # Return empty string if there's an issue

def filter_test_vectors(test_vectors, company_index, exclude_companies=["nvidia", "tesla", "tesla-motors"]):
    # Filter out rows where the company is in the exclude_companies list
    filtered_test_vectors = test_vectors[~test_vectors.iloc[:, company_index].str.lower().isin(exclude_companies)]
    return filtered_test_vectors

def calculate_tp_fp_tn_fn(recommendations, test_set, target_company, company_index, num_applicants):
    TP, FP, TN, FN = 0, 0, 0, 0
    applicant_columns = get_applicant_columns(num_applicants)

    for i in range(len(recommendations)):
        actual_company = test_set.iloc[i, company_index].strip().lower()

        predicted_companies = []
        for col in applicant_columns:
            applicant_info = recommendations.iloc[i, col]
            predicted_company = extract_company_from_applicant(applicant_info, company_index)
            predicted_companies.append(predicted_company)

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

def run_accuracy_metrics():
    dataset_variation = input("Enter the dataset variation (e.g., with_gender_and_age, gender_no_age, age_no_gender, no_age_no_gender): ")
    distance_function = input("Enter the distance function (e.g., Statistic_intersection, Statistic_list_frequency): ")

    recommendation_file = recommendation_files.get('position_to_applicant', {}).get(distance_function, {}).get(dataset_variation)
    test_set_file = test_sets.get(dataset_variation)
    company_index = company_indices.get(dataset_variation)

    if not recommendation_file or not os.path.exists(recommendation_file):
        print(f"Recommendation file not found for {distance_function}, {dataset_variation}.")
        return

    if not test_set_file or not os.path.exists(test_set_file):
        print(f"Test set not found for {dataset_variation}.")
        return

    recommendations = read_excel_file(recommendation_file)
    test_set = pd.read_csv(test_set_file)
    
    # Filter out rows where the company is 'nvidia', 'tesla', or 'tesla-motors'
    filtered_test_set = filter_test_vectors(test_set, company_index)

    actual_companies = filtered_test_set.iloc[:, company_index].apply(str.lower).tolist()

    unique_companies = set(actual_companies)

    all_results = []

    for target_company in unique_companies:
        print(f"Calculating accuracy for company: {target_company}")

        # Calculate accuracy for all x-values (1, 3, 5, 11)
        for num_applicants in [1, 3, 5, 11]:
            TP, FP, TN, FN = calculate_tp_fp_tn_fn(recommendations, filtered_test_set, target_company, company_index, num_applicants)
            accuracy = calculate_accuracy(TP, FP, TN, FN)

            all_results.append({
                "Dataset Variation": dataset_variation,
                "Distance Function": distance_function,
                "Target Company": target_company,
                "Applicants Considered": num_applicants,
                "TP": TP,
                "FP": FP,
                "TN": TN,
                "FN": FN,
                "Accuracy": accuracy
            })

    results_df = pd.DataFrame(all_results)
    print(results_df)

    output_dir = "final_measurements"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{dataset_variation}_{distance_function}_accuracy_PTA.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    try:
        run_accuracy_metrics()
    except Exception as e:
        logging.error(f"Error running accuracy calculations: {e}")
        print(f"Error running accuracy calculations: {e}")