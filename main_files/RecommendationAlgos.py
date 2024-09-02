########################## INSTRUCTIONS ##########################
## this must be initiated only after the creation of the model using models_generator.py
## the distance function and dataset option must be identical in both model generator and this recommendation algorithm
## Uncomment the "columns to exclude" if statment in the choser distance function
## Change the company index according to the chosen dataset option 
## Uncomment normalization according to the dataset you have chosen in Statistic_list_frequeny / Statistic_intersection
## Uncomment in the section below the recommendation algorithm you want to initiate: Standard / Multiclustering
##################################################################

import sys
import os
import csv
import pickle
import numpy as np
import pandas as pd
import logging


script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(filename='debug_log.txt', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
from general_algos.Preprocess_for_hr import KMeansClusterer
from statistic_regular_algo.Statistic_list_frequency import Statistic_list_frequency
from statistic_regular_algo.Statistic_intersection import Statistic_intersection

 
################# CHANGE company_index TO: with_gender_and_age = 11 / gender_no_age = 10 / age_no_gender = 10 / no_age_no_gender = 9 #################
company_index = 11

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
    type_of_fields=model_data["type_of_fields"]
    # print("HP:", hp)

    if not model.all_clusters:
        raise ValueError("No clusters found in the loaded model.")
    for cluster in model.all_clusters:
        if cluster.subclusters is None or not cluster.subclusters:
            raise ValueError(f"Subclusters missing or not initialized for cluster {cluster.cluster_id}.")
    return model, hp, type_of_fields

def load_test_vectors(test_path):
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"The file {test_path} does not exist.")
    
    df = pd.read_csv(test_path, header=None)
    test_vectors = [row.tolist() for _, row in df.iterrows()]
        
    return test_vectors

def find_nearest_cluster(query, model, hp, type_of_fields, distance_function):
    min_distance = float('inf')
    closest_cluster = None
    closest_cluster_mean = None

    for cluster in model.all_clusters:
        if cluster is None:
            print("Encountered a None cluster during iteration.")
            continue  # Skip any None clusters
        cluster_mean = cluster.mean
        try:
            distance, result = distance_function(query, cluster_mean, type_of_fields, hp)

        except Exception as e:
            logging.error(f"Error calculating distance: {e}")
            continue  # Skip this cluster if there's an error

        if distance < min_distance:
            min_distance = distance
            closest_cluster = cluster
            closest_cluster_mean = cluster_mean

    return closest_cluster, closest_cluster_mean



def rank_nearest_subcluster(query, nearest_cluster, model, hp, type_of_fields, distance_function):
    distances = []

    for subcluster in nearest_cluster.subclusters.values():
        subcluster_centroid = subcluster.centroid 
        distance, result = distance_function(query, subcluster_centroid, type_of_fields, hp)
        distances.append((subcluster.company, distance))

    distances.sort(key=lambda x: x[1])
    return distances


def recommend_company(query, model, hp, type_of_fields, distance_function):
    nearest_cluster, nearest_cluster_mean = find_nearest_cluster(query, model, hp, type_of_fields, distance_function)
    if nearest_cluster_mean is not None and isinstance(nearest_cluster_mean, np.ndarray):
        nearest_cluster_mean = nearest_cluster_mean.tolist()
    
    ranked_subclusters = rank_nearest_subcluster(query, nearest_cluster, model, hp, type_of_fields, distance_function)

    return ranked_subclusters

dataset_options = ["with_gender_and_age", "gender_no_age", "age_no_gender", "no_age_no_gender"]
distance_functions = {
    "Statistic_list_frequency": Statistic_list_frequency,
    "Statistic_intersection": Statistic_intersection
}



def find_nearest_records_in_cluster(query, cluster, model, hp, R, distance_function):
    """
    Find the R nearest records in the specified cluster to the given query using the model and saved hyperparameters.
    """
    distances = []
    for record in cluster.data[cluster.cluster_id]:
        distance, _ = distance_function(query, record, model._type_of_fields, hp)
        company = record[company_index]
        distances.append((company, distance))

    distances.sort(key=lambda x: x[1])
    return distances[:R]


def recommend_company_standard(query, model, hp, R,type_of_fields, distance_function):
    """
    Recommend companies using the standard recommendation approach, which finds the nearest cluster and ranks records within it.
    """
    nearest_cluster, nearest_cluster_mean = find_nearest_cluster(query, model, hp, type_of_fields, distance_function)
    
    if nearest_cluster_mean is not None and isinstance(nearest_cluster_mean, np.ndarray):
        nearest_cluster_mean = nearest_cluster_mean.tolist()

    ranked_records = find_nearest_records_in_cluster(query, nearest_cluster, model, hp, R, distance_function)
    return ranked_records


# # code for inspecting the model pickel 

# file_path = 'saved_models\with_gender_and_age_Statistic_list_frequency\with_gender_and_age_Statistic_list_frequency_train_model.pkl'

# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# # Display the keys and basic structure of the loaded data
# data_summary = {
    
#         "train_results": data.get('train_results'),
#         "all_clusters": data.get('all_clusters'),
#         "hp": data.get('hp'),
#         "k_summery": data.get('k'),
#         "model": data.get('model'),
# }
# print(data_summary)
# print("_hyper_parameters")
# print(data_summary['model']._hyper_parameters)
# print("_type_of_fields")
# print(data_summary['model']._type_of_fields)


print("Choose a dataset option:")
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
distance_function = distance_functions[distance_function_name]

model_path = os.path.join("saved_models", f"{dataset_option}",f"{dataset_option}_{distance_function_name}_train_model.pkl")
test_path = os.path.join("datasets", f"test_{dataset_option}.csv")


R=13


# ######## Uncomment this to use the multiclustering recommendation algorithm
# try:
#     model, hp, type_of_fields = load_model(model_path)
#     test_vectors = load_test_vectors(test_path)

#     print(f"Loaded model from {model_path}")
#     print(f"Model: {model}")
#     print(f"Number of test vectors: {len(test_vectors)}")
    
#     output_dir = os.path.join("results", f"{dataset_option}")
#     os.makedirs(output_dir, exist_ok=True)
#     output_file = os.path.join(output_dir, f"{distance_function_name}_multiclustering_recommendations.csv")
    
#     with open(output_file, mode='w', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)
#         header = ["Query", "Company_1", "Distance_1", "Company_2", "Distance_2", "Company_3", "Distance_3", "Company_4", "Distance_4", "Company_5", "Distance_5"]
#         writer.writerow(header)

#         for query in test_vectors:
#             ranked_subclusters = recommend_company(query, model, hp, type_of_fields, distance_function)
#             row = [str(query)]  # Convert the query to a string representation
#             for company, distance in ranked_subclusters:  # Assuming you want the top 5 recommendations
#                 row.append(company)
#                 row.append(distance)
#             writer.writerow(row)
#     print(f"Queries and recommendations have been saved to {output_file}")

# except FileNotFoundError as fnf_error:
#     print(fnf_error)
# except Exception as e:
#     print(f"An error occurred: {e}")


######### Uncomment this to use the standard recommendation algorithm
try:
    model, hp, type_of_fields = load_model(model_path)
    test_vectors = load_test_vectors(test_path)
    
    print(f"Loaded model from {model_path}")
    print(f"Model: {model}")
    print(f"Number of test vectors: {len(test_vectors)}")

    output_dir = os.path.join("results", f"{dataset_option}")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{distance_function_name}_standard_recommendations.csv")
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        header = ["Query"] + [f"Company_{i+1}" for i in range(R)] + [f"Distance_{i+1}" for i in range(R)]
        writer.writerow(header)

        for query in test_vectors:
            ranked_records = recommend_company_standard(query, model, hp, R, type_of_fields, distance_function)
            row = [str(query)] 
            
            for i in range(R):
                if i < len(ranked_records):
                    company, distance = ranked_records[i]
                    row.append(company)
                    row.append(distance)
                else:
                    row.append("")
                    row.append("")
            
            writer.writerow(row)
    
    print(f"Queries and recommendations have been saved to {output_file}")

except FileNotFoundError as fnf_error:
    print(fnf_error)
except Exception as e:
    print(f"An error occurred: {e}")