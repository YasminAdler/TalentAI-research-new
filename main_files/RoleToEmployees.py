import sys
import os
import csv
import pickle
import numpy as np
import pandas as pd
import logging
import random
import ast

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from general_algos.Preprocess_for_hr import KMeansClusterer
from statistic_regular_algo.Statistic_list_for_role_to_employee import Statistic_list_frequency
from statistic_regular_algo.Statistic_intersection_for_role_to_employee import Statistic_intersection

# Custom Unpickler class to handle model loading
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "main_files":
            module = "__main__"
        return super().find_class(module, name)

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The file {model_path} does not exist.")
    
    with open(model_path, "rb") as f:
        model_data = CustomUnpickler(f).load()
    
    return model_data["model"], model_data["model"]._wcss

def load_test_vectors(test_path):
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"The file {test_path} does not exist.")
    
    df = pd.read_csv(test_path, header=None)
    test_vectors = [row.tolist() for _, row in df.iterrows()]
        
    return test_vectors

def filter_test_vectors(test_vectors, exclude_companies=["nvidia", "tesla", "tesla-motors"]):
    filtered_test_vectors = []
    for vector in test_vectors:
        if vector[company_index].lower() not in exclude_companies:
            filtered_test_vectors.append(vector)    
    return filtered_test_vectors

def validate_and_clean_vectors(vectors):
    """
    Ensures that all vectors are in the correct format and handles missing values.
    """
    cleaned_vectors = []
    for vector in vectors:
        cleaned_vector = []
        for element in vector:
            if pd.isnull(element) or element == '':
                cleaned_vector.append(None)  # Use None for missing values
            else:
                cleaned_vector.append(element)
        cleaned_vectors.append(cleaned_vector)
    return cleaned_vectors

def select_positions_and_employees(filtered_vectors):
    positions = []
    employees = []
    
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
        print("EMPLOYEE QUERY: ", employee)
        print("POSITION QUERY: ", position)
        
        distance = distance_function(position, employee, model._type_of_fields, model._hyper_parameters)
        distances.append((employee, distance))

    distances.sort(key=lambda x: x[1])
    return distances

def safe_literal_eval(value):
    try:
        if isinstance(value, str):
            value = value.strip()

            # Handle cases for empty lists and malformed strings
            if value == "[]":
                return []

            if value.startswith("[") and value.endswith("]"):
                value = value.replace("\n", " ").replace("\r", " ")
                # Safely evaluate the string
                result = ast.literal_eval(value)
                # Convert nested lists to standard list if needed
                if isinstance(result, list):
                    return [convert_nested_lists(item) for item in result]
                return result
        return value
    except (ValueError, SyntaxError) as e:
        logging.error(f"Failed to parse: {value} | Error: {str(e)}")
        return value

def convert_nested_lists(value):
    if isinstance(value, str):
        # Handle common string cases like lists of strings
        if value.startswith("[") and value.endswith("]"):
            value = value.replace("\n", " ").replace("\r", " ")
            try:
                result = ast.literal_eval(value)
                if isinstance(result, list):
                    return [convert_nested_lists(item) for item in result]
                return result
            except (ValueError, SyntaxError):
                return value
    return value


def calculate_distances_for_all_positions_and_employees(positions, employees, model, distance_function):
    all_ranked_employees = []

    for position_list in positions:
        for position in position_list:
            ranked_employees = rank_employees_for_position(position, employees, model, distance_function)
            all_ranked_employees.append((position, ranked_employees[:11]))

    return all_ranked_employees

dataset_options = ["with_gender_and_age", "gender_no_age", "age_no_gender", "no_age_no_gender"]
distance_functions = {
    "Statistic_list_frequency": Statistic_list_frequency,
    "Statistic_intersection": Statistic_intersection
}

company_indices = {
    "with_gender_and_age": 11,
    "gender_no_age": 10,
    "age_no_gender": 10,
    "no_age_no_gender": 9
}

logging.basicConfig(filename='debug_log.txt', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

# User selects dataset option
print("Choose a dataset option for employees:")
for idx, option in enumerate(dataset_options):
    print(f"{idx + 1}. {option}")
dataset_choice = int(input("Enter the number of your choice: ")) - 1
if dataset_choice < 0 or dataset_choice >= len(dataset_options):
    print("Invalid choice. Please enter a valid number.")
    sys.exit(1)
dataset_option = dataset_options[dataset_choice]

# User selects distance function
print("Choose a distance function:")
for idx, option in enumerate(distance_functions.keys()):
    print(f"{idx + 1}. {option}")
distance_choice = int(input("Enter the number of your choice: ")) - 1

if distance_choice < 0 or distance_choice >= len(distance_functions):
    print("Invalid choice. Please enter a valid number.")
    sys.exit(1)
    
distance_function_name = list(distance_functions.keys())[distance_choice]
distance_function = distance_functions[distance_function_name]

# Set company index based on the chosen dataset
company_index = company_indices[dataset_option]

# Load test vectors from the selected CSV file
test_path = os.path.join("datasets", f"test_{dataset_option}.csv")
test_vectors = load_test_vectors(test_path)

# Validate and clean the vectors
cleaned_test_vectors = validate_and_clean_vectors(test_vectors)

# Filter test vectors
filtered_test_vectors = filter_test_vectors(cleaned_test_vectors)

# Select positions and employees
positions, employees = select_positions_and_employees(filtered_test_vectors)

num_positions = sum(len(position_list) for position_list in positions)
num_employees = len(employees)

print(f"Number of queries in positions: {num_positions}")
print(f"Number of queries in employees: {num_employees}")

# Define the model path
model_path = os.path.join("saved_sets", f"{dataset_option}_{distance_function_name}", f"{dataset_option}_{distance_function_name}_train_model.pkl")

TOP_RANKS = 11 
try:
    model = load_model(model_path)
    
    print(f"Loaded model from {model_path}")
    print(f"Number of test vectors: {len(test_vectors)}")
    
    output_file = os.path.join("results", f"{distance_function_name}_multicluster_position_to_employee.csv")
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        header = ["Position"] + [f"Employee_{i+1}" for i in range(TOP_RANKS)] + [f"Distance_{i+1}" for i in range(TOP_RANKS)]
        writer.writerow(header)

    ranked_employees_for_all_positions = calculate_distances_for_all_positions_and_employees(positions, employees, model, distance_function)

    for position, ranked_employees in ranked_employees_for_all_positions:
        row = [str(position)] 
        for i in range(TOP_RANKS):
            if i < len(ranked_employees):
                employee, distance = ranked_employees[i]
                row.append(str(employee))
                row.append(distance)
            else:
                row.append("")
                row.append("")
        writer.writerow(row)

    print(f"Positions and employee recommendations have been saved to {output_file}")

except FileNotFoundError as fnf_error:
    print(fnf_error)
except Exception as e:
    print(f"An error occurred: {e}")
