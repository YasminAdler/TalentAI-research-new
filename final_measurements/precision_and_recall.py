import pandas as pd
import glob
from sklearn.metrics import precision_score, recall_score
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
Using google for the example: 

True Positive (TP): The actual label is Google, and the recommendation is also Google.
False Negative (FN): The actual label is Google, but the recommendation is not Google. ## to ask if this is definit?
False Positive (FP): The actual label is not Google, but the recommendation is Google.
True Negative (TN): The actual label is not Google, and the recommendation is also not Google.

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
    
def calculate_precision_recall(recommendations, actual_companies, target_company):
    y_true_mid = {i: company for i, company in enumerate(actual_companies)}  ## a dictionary of all index actual, companies
    y_pred_mid = {i: company for i, company in enumerate(recommendations['Company_1'].str.lower())}  
    
    # print("y_true_mid",y_true_mid)
    # print("y_pred_mid", y_pred_mid)

    y_true = {} 
    y_pred = {} 
    
    for i in y_true_mid:
        if y_true_mid[i] == target_company:
            y_true[i] = y_true_mid[i]  # Store only if the actual company matches the target

    TP = 0
    FN = 0
    FP = 0

    for i in y_true:
        if y_pred_mid.get(i, None) == target_company:
            TP += 1  # True Positive: correct recommendation for the target company
        else:
            FN += 1  # False Negative: wrong recלommendation at this index

    for i in y_pred_mid:
        if y_pred_mid[i] == target_company and y_true_mid.get(i) != target_company:
            FP += 1  # False Positive: recommended the target company when the actual was different

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    # f1 = precision ## to complete 

    return precision, recall

def run_precision_recall():
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
        print(f"Calculating for company: {company}")
        precision, recall = calculate_precision_recall(recommendations, actual_companies, company)
        all_results.append({
            "Algorithm": algorithm,
            "Distance Function": distance_function,
            "Dataset Variation": dataset_variation,
            "Company": company,
            "Precision": precision,
            "Recall": recall
        })

    results_df = pd.DataFrame(all_results)
    print(results_df)

    average_precision = results_df["Precision"].mean()
    average_recall = results_df["Recall"].mean()
    print(f"\nAverage Precision: {average_precision}")
    print(f"Average Recall: {average_recall}")

    output_dir = "final_measurements"
    os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist
    output_file = f"{output_dir}/{dataset_variation}_{algorithm}_{distance_function}_precision&recall.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    try:
        run_precision_recall()
    except Exception as e:
        logging.error(f"Error running precision and recall calculations: {e}")
        print(f"Error running precision and recall calculations: {e}")