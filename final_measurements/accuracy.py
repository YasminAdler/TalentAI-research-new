


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
    
    
    
"""
Definitions: 

Accuracy = (TP + TN) / (TP + TN + FP + FN)

True Positive (TP): The actual label is Google, and the recommendation is also Google.
False Negative (FN): The actual label is Google, but the recommendation is not Google. ## to ask if this is definit?
False Positive (FP): The actual label is not Google, but the recommendation is Google.
True Negative (TN): The actual label is not Google, and the recommendation is also not Google.

"""

def calculate_metrics(recommendations, actual_companies, target_company, x_values):
    # y_pred_mid = {i: company.strip().lower() for i, company in recommendations}
    
    company_columns  = list(range(1, 26, 2))
    extracted_companies = {
        i: [recommendations.iloc[i,col] for col in company_columns]
        for i in range(len(recommendations))
    }

    # for i, companies in extracted_companies.items():
    #     print(f"Row {i}: {companies}")
            
    y_pred_mid = {i: extracted_company for i, extracted_company in extracted_companies.items()}
    # print("y_pred_mid", y_pred_mid)

    y_true_mid = {i: company for i, company in enumerate(actual_companies)}
    # print("y_true_mid", y_true_mid)
    results = {}

    for x in sorted(x_values):
        
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in y_true_mid:
            
            # True Positive and False Negative logic
            if y_true_mid[i] == target_company:
                print(x)
                # print("y_pred_mid.get(i, [])[:x]:", y_pred_mid.get(i, [])[:x])
                if target_company in y_pred_mid.get(i, [])[:x]:  
                    TP += 1  # True Positive: correct recommendation for the target company
                else:
                    FN += 1  # False Negative: wrong recommendation at this index ## useless
                    
            # False Positive and True Negative logic
            else: 
                if target_company in y_pred_mid.get(i, [])[:x]: 
                    FP += 1  # False Positive: recommended the target company when the actual was different ## useless
                else:
                    TN += 1  # True Negative: correctly did not recommend the target company

        # accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) != 0 else 0.0
        accuracy = (TP + TN) / 184
        results[x] = {'Accuracy': accuracy}

    return results

def run_metrics():
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

    x_values = [1, 3, 5, 13]

    all_results = []
    for company in unique_companies:
        print(f"Calculating metrics for company: {company}")
        results = calculate_metrics(recommendations, actual_companies, company, x_values)
        for x, metrics in results.items():
            all_results.append({
                "Algorithm": algorithm,
                "Distance Function": distance_function,
                "Dataset Variation": dataset_variation,
                "Company": company,
                "x": x,
                "Accuracy": metrics['Accuracy']
            })

    results_df = pd.DataFrame(all_results)
    print(results_df)

    output_dir = "final_measurements"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{dataset_variation}_{algorithm}_{distance_function}_accuracy.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    try:
        run_metrics()
    except Exception as e:
        logging.error(f"Error running metrics calculations: {e}")
        print(f"Error running metrics calculations: {e}")