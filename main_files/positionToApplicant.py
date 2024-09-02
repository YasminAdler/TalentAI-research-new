########################## INSTRUCTIONS ##########################
## this must be initiated only after the creation of the model using model generator
## the distance function and dataset option must be identical in both model generator and this recommendation algorithm
## Uncomment the corresponding normalization part according to the dataset you have chosen in the distance function file (Statistic_list_frequeny / Statistic_intersection)
## Change the company index according to the chosen dataset option : with_gender_and_age = 11 / gender_no_age = 10 / age_no_gender = 10 / no_age_no_gender = 9
## Uncomment the "columns to exclude" if statment in the chosen distance function file
##################################################################

import sys
import os
import csv
import pickle
import numpy as np
import pandas as pd
import logging
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(filename='debug_log.txt', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

from general_algos.Preprocess_for_hr import KMeansClusterer


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "main_files":
            module = "__main__"
        return super().find_class(module, name)

def load_model(model_path):
    """
    Load the model and hyperparameters from the saved pickle file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The file {model_path} does not exist.")
    
    with open(model_path, "rb") as f:
        model_data = CustomUnpickler(f).load()
    
    model = model_data["model"]
    hp = model_data["hp"]
    type_of_fields = model_data["type_of_fields"]
    distance_function = model._distance 
    return model, hp, type_of_fields, distance_function

def load_test_vectors(test_path):
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"The file {test_path} does not exist.")
    df = pd.read_csv(test_path, header=None)
    test_vectors = [row.tolist() for _, row in df.iterrows()]
    return test_vectors

def filter_test_vectors(test_vectors, company_index, exclude_companies=["nvidia", "tesla", "tesla-motors"]):
    filtered_test_vectors = [vector for vector in test_vectors if vector[company_index].lower() not in exclude_companies]
    return filtered_test_vectors

def select_positions_and_employees(filtered_vectors, company_index):
    positions, employees = [], []
    companies = list(set(vector[company_index] for vector in filtered_vectors))

    for company in companies:
        company_vectors = [vector for vector in filtered_vectors if vector[company_index] == company]
        
        if len(company_vectors) < 5:
            print(f"Not enough records for company: {company}")
            continue
        
        selected_positions = random.sample(company_vectors, 5)
        positions.append(selected_positions)
        remaining_vectors = [vector for vector in company_vectors if vector not in selected_positions]
        employees.extend(remaining_vectors)
    
    return positions, employees

def rank_employees_for_position(position, employees, model, distance_function):
    distances = []
    for employee in employees:
        distance, _ = distance_function(position, employee, model._type_of_fields, model._hyper_parameters)
        distances.append({
            'employee': employee,
            'distance': distance
        })
    
    distances.sort(key=lambda x: x['distance'])
    return {str(position): distances[:11]}



def calculate_distances_for_all_positions_and_employees(all_positions, employees, model, distance_function):
    all_ranked_employees = {}
    for positions_by_company in all_positions:
        for position in positions_by_company:
            position_and_ranked_employees = rank_employees_for_position(position, employees, model, distance_function)

            for pos, ranked_employees in position_and_ranked_employees.items():
                formatted_employees = []
                for ranking in ranked_employees:
                    formatted_employees.extend([ranking['employee'], ranking['distance']])
                all_ranked_employees[pos] = formatted_employees

    return all_ranked_employees


dataset_options = ["with_gender_and_age", "gender_no_age", "age_no_gender", "no_age_no_gender"]
distance_functions = {
    "Statistic_list_frequency": "Statistic_list_frequency",
    "Statistic_intersection": "Statistic_intersection"
}
company_indices = {
    "with_gender_and_age": 11,
    "gender_no_age": 10,
    "age_no_gender": 10,
    "no_age_no_gender": 9
}

print("Choose a dataset option for employees:")
for idx, option in enumerate(dataset_options):
    print(f"{idx + 1}. {option}")
dataset_choice = int(input("Enter the number of your choice: ")) - 1
if dataset_choice < 0 or dataset_choice >= len(dataset_options):
    print("Invalid choice. Please enter a valid number.")
    sys.exit(1)
dataset_option = dataset_options[dataset_choice]

print("Choose a distance function:")
for idx, option in enumerate(distance_functions.keys()):
    print(f"{idx + 1}. {option}")
distance_choice = int(input("Enter the number of your choice: ")) - 1
if distance_choice < 0 or distance_choice >= len(distance_functions):
    print("Invalid choice. Please enter a valid number.")
    sys.exit(1)
distance_function_name = list(distance_functions.keys())[distance_choice]

company_index = company_indices[dataset_option]

test_path = os.path.join("datasets", f"test_{dataset_option}.csv")
model_path = os.path.join("saved_models", f"{dataset_option}", f"{dataset_option}_{distance_function_name}_train_model.pkl")

try:
    test_vectors = load_test_vectors(test_path)
    model, hp, type_of_fields, distance_function = load_model(model_path)

    filtered_test_vectors = filter_test_vectors(test_vectors, company_index)
    all_positions, employees = select_positions_and_employees(filtered_test_vectors, company_index)

    num_positions = sum(len(position_by_company) for position_by_company in all_positions)
    num_employees = len(employees)

    print(f"Number of queries in positions: {num_positions}")
    print(f"Number of queries in employees: {num_employees}")

    # Calculate distances for all positions and employees
    ranked_employees_for_all_positions = calculate_distances_for_all_positions_and_employees(
        all_positions, employees, model, distance_function
    )

    output_file = os.path.join("results", "position_to_applicant", f"{dataset_option}_{distance_function_name}_recommendations.csv")
    
    # Open the file and write the results
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        header = ["Position", "Applicant_1", "Distance_1", "Applicant_2", "Distance_2", 
                "Applicant_3", "Distance_3", "Applicant_4", "Distance_4", 
                "Applicant_5", "Distance_5", "Applicant_6", "Distance_6", 
                "Applicant_7", "Distance_7", "Applicant_8", "Distance_8", 
                "Applicant_9", "Distance_9", "Applicant_10", "Distance_10", 
                "Applicant_11", "Distance_11"]
        writer.writerow(header)

        # Iterate over the dictionary of ranked employees for each position
        for position, ranked_applicants in ranked_employees_for_all_positions.items():
            row = [position]  # Start the row with the position query

            # Loop through the top 11 applicants and their distances
            for i in range(11):
                if i * 2 < len(ranked_applicants):  # Check to avoid index out-of-range
                    employee = ranked_applicants[i * 2]
                    distance = ranked_applicants[i * 2 + 1]
                    row.append(str(employee))  # Add the employee data as a string
                    row.append(distance)       # Add the distance value
                else:
                    # Append empty placeholders if fewer than 11 applicants
                    row.append("")  
                    row.append("")

            # Write the constructed row to the CSV file
            writer.writerow(row)

    print(f"Positions and applicant recommendations have been saved to {output_file}")

except FileNotFoundError as fnf_error:
    print(fnf_error)
except Exception as e:
    print(f"An error occurred: {e}")
