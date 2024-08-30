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
from statistic_regular_algo.Statistic_dot_product import Statistic_dot_product
from statistic_regular_algo.Statistic_intersection import Statistic_intersection

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
    
    # Extract the model and hyperparameters
    model = model_data["model"]
    hp = model_data["hp"]  # Load the hyperparameters saved in the .pkl file
    type_of_fields=model_data["type_of_fields"]
    # print("HP:", hp)
    return model, hp, type_of_fields

def load_test_vectors(test_path):
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"The file {test_path} does not exist.")
    
    df = pd.read_csv(test_path, header=None)
    test_vectors = [row.tolist() for _, row in df.iterrows()]
        
    return test_vectors

def find_nearest_cluster(query, model, hp, type_of_fields):
    min_distance = float('inf')
    closest_cluster = None
    closest_cluster_mean = None
    # for cluster in model.all_clusters:
        # print(f"Cluster ID: {cluster.cluster_id}")
        # print(f"Mean: {cluster.mean}")
        # print(f"Average Distance: {cluster.average_distance}")
        # print(f"Max Distance: {cluster.max_distance}")
        # print(f"Standard Deviation: {cluster.std_dev}")
        # print(f"Attributes Average Distances: {cluster.attributes_average_distances}")
        # print(f"Attributes Standard Deviations: {cluster.attributes_std_devs}")
        # print(f"Data Points in Cluster: {len(cluster.data)}")  # Prints the number of data points in the cluster
        # print("Data Sample:", cluster.data[:5])  # Prints the first 5 data points as a sample, adjust as needed
        # print("-" * 50)  # Separator between clusters for readability

    for cluster in model.all_clusters:
        cluster_mean = cluster.mean
        distance, _ = model._distance(query, cluster_mean, type_of_fields, hp)
        if distance < min_distance:
            min_distance = distance
            closest_cluster = cluster
            closest_cluster_mean = cluster_mean
    return closest_cluster, closest_cluster_mean    


def rank_nearest_subcluster(query, cluster, model, hp, type_of_fields):
    distances = []

    for subcluster in cluster.subclusters.values():
        subcluster_centroid = subcluster.centroid 
        distance, result = model._distance(query, subcluster_centroid, type_of_fields, hp)
        distances.append((subcluster.company, distance))

    distances.sort(key=lambda x: x[1])
    return distances


def recommend_company(query, model, hp, type_of_fields):
    nearest_cluster, nearest_cluster_mean = find_nearest_cluster(query, model, hp, type_of_fields)
    
    # Ensure nearest_cluster_mean is in the correct format
    if nearest_cluster_mean is not None and isinstance(nearest_cluster_mean, np.ndarray):
        nearest_cluster_mean = nearest_cluster_mean.tolist()
    
    ranked_subclusters = rank_nearest_subcluster(query, nearest_cluster, model, hp, type_of_fields)

    return ranked_subclusters

dataset_options = ["with_gender_and_age", "gender_no_age", "age_no_gender", "no_age_no_gender"]
distance_functions = {
    "Statistic_list_frequency": Statistic_list_frequency,
    "Statistic_dot_product": Statistic_dot_product,
    "Statistic_intersection": Statistic_intersection
}

 
################# CHANGE company_index TO: with_gender_and_age = 11 / gender_no_age = 10 / age_no_gender = 10 / no_age_no_gender = 9 #################
company_index = 11

def find_nearest_records_in_cluster(query, cluster, model, hp, R):
    """
    Find the R nearest records in the specified cluster to the given query using the model and saved hyperparameters.
    """
    distances = []
    for record in cluster.data[cluster.cluster_id]:
        distance, _ = model._distance(query, record, model._type_of_fields, hp)
        company = record[company_index]
        distances.append((company, distance))

    distances.sort(key=lambda x: x[1])
    return distances[:R]


def recommend_company_standard(query, model, hp, R,type_of_fields):
    """
    Recommend companies using the standard recommendation approach, which finds the nearest cluster and ranks records within it.
    """
    nearest_cluster, nearest_cluster_mean = find_nearest_cluster(query, model, hp, type_of_fields)
    
    if nearest_cluster_mean is not None and isinstance(nearest_cluster_mean, np.ndarray):
        nearest_cluster_mean = nearest_cluster_mean.tolist()

    ranked_records = find_nearest_records_in_cluster(query, nearest_cluster, model, hp, R)
    return ranked_records


# # code for inspecting the model pickel 

# file_path = 'saved_sets\with_gender_and_age_Statistic_list_frequency\with_gender_and_age_Statistic_list_frequency_train_model.pkl'

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

# Define model and test set paths based on user selection
model_path = os.path.join("saved_sets", f"{dataset_option}_{distance_function_name}_train_model.pkl")
test_path = os.path.join("datasets", f"test_{dataset_option}.csv")



# ######### Uncomment this to use the multiclustering recommendation algorithm
# try:
#     # Load the model and test vectors
#     model, hp, type_of_fields = load_model(model_path)

#     test_vectors = load_test_vectors(test_path)
    
#     # Debugging information
#     print(f"Loaded model from {model_path}")
#     print(f"Model: {model}")
#     print(f"Number of test vectors: {len(test_vectors)}")

#     # Create directory for saving results if it doesn't exist
#     output_dir = os.path.join("results", f"{dataset_option}")
#     os.makedirs(output_dir, exist_ok=True)

#     # Prepare to write to CSV
#     output_file = os.path.join(output_dir, f"{distance_function_name}_multiclustering_recommendations.csv")
#     with open(output_file, mode='w', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)
#         header = ["Query", "Company_1", "Distance_1", "Company_2", "Distance_2", "Company_3", "Distance_3", "Company_4", "Distance_4", "Company_5", "Distance_5"]
#         writer.writerow(header)

#         # Get recommendations for each query in the test set
#         for query in test_vectors:

#             ranked_subclusters = recommend_company(query, model, hp, type_of_fields)
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

########## Uncomment this to use the standard recommendation algorithm
R = 13

try:
    # Load the model and test vectors
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
            ranked_records = recommend_company_standard(query, model, hp, R, type_of_fields)
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