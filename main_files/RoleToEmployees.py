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

# Custom Unpickler class to handle model loading
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
    
    # Extract the model, hyperparameters, and type of fields
    model = model_data["model"]
    hp = model_data["hp"]
    type_of_fields = model_data["type_of_fields"]
    distance_function = model._distance  # Use the distance function saved in the model
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
    distances = [(employee, distance_function(position, employee, model._type_of_fields, model._hyper_parameters))
                 for employee in employees]
    distances.sort(key=lambda x: x[1])
    return distances

def calculate_distances_for_all_positions_and_employees(positions, employees, model, distance_function):
    all_ranked_employees = []

    for position_list in positions:
        for position in position_list:
            ranked_employees = rank_employees_for_position(position, employees, model, distance_function)
            all_ranked_employees.append((position, ranked_employees[:11]))
    return all_ranked_employees

# Define the available datasets and their company indices
dataset_options = ["with_gender_and_age", "gender_no_age", "age_no_gender", "no_age_no_gender"]
distance_functions = {
    "Statistic_list_frequency": "Statistic_list_frequency",
    "Statistic_dot_product": "Statistic_dot_product",
    "Statistic_intersection": "Statistic_intersection"
}
company_indices = {
    "with_gender_and_age": 11,
    "gender_no_age": 10,
    "age_no_gender": 10,
    "no_age_no_gender": 9
}

# User selects dataset option
print("Choose a dataset option for employees:")
for idx, option in enumerate(dataset_options):
    print(f"{idx + 1}. {option}")
dataset_choice = int(input("Enter the number of your choice: ")) - 1
if dataset_choice < 0 or dataset_choice >= len(dataset_options):
    print("Invalid choice. Please enter a valid number.")
    sys.exit(1)
dataset_option = dataset_options[dataset_choice]

# User selects distance function name for model path
print("Choose a distance function:")
for idx, option in enumerate(distance_functions.keys()):
    print(f"{idx + 1}. {option}")
distance_choice = int(input("Enter the number of your choice: ")) - 1
if distance_choice < 0 or distance_choice >= len(distance_functions):
    print("Invalid choice. Please enter a valid number.")
    sys.exit(1)
distance_function_name = list(distance_functions.keys())[distance_choice]

# Set company index based on the chosen dataset
company_index = company_indices[dataset_option]

# Load test vectors and model
test_path = os.path.join("datasets", f"test_{dataset_option}.csv")
model_path = os.path.join("saved_sets", f"{dataset_option}_{distance_function_name}_train_model.pkl")

try:
    test_vectors = load_test_vectors(test_path)
    model, hp, type_of_fields, distance_function = load_model(model_path)

    filtered_test_vectors = filter_test_vectors(test_vectors, company_index)
    positions, employees = select_positions_and_employees(filtered_test_vectors, company_index)

    num_positions = sum(len(position_list) for position_list in positions)
    num_employees = len(employees)
    print(f"Number of queries in positions: {num_positions}")
    print(f"Number of queries in employees: {num_employees}")

    ranked_employees_for_all_positions = calculate_distances_for_all_positions_and_employees(
        positions, employees, model, distance_function
    )
    
        # Save results
    # Save results
    output_file = os.path.join("results", f"second_direction_{distance_function_name}__to_applicant_recommendations.csv")
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Define the header with alternating Applicant and Distance columns
        header = ["Position"] + [f"Applicant_{i+1}" for i in range(11)] + [f"Distance_{i+1}" for i in range(11)]
        writer.writerow(header)

        for position, ranked_applicants in ranked_employees_for_all_positions:
            row = [str(position)]
            for i in range(11):
                if i < len(ranked_applicants):
                    # Use placeholders for applicants and their distances
                    applicant, distance = ranked_applicants[i]
                    row.append(f"Applicant_{i+1}")
                    row.append(distance)
                else:
                    row.append("")
                    row.append("")
            writer.writerow(row)

    print(f"Positions and applicant recommendations have been saved to {output_file}")


except FileNotFoundError as fnf_error:
    print(fnf_error)
except Exception as e:
    print(f"An error occurred: {e}")
