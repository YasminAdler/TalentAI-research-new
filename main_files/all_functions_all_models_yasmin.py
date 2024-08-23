import sys
import logging
import csv
import numpy as np
import pandas as pd
import os
import openpyxl
import pickle
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))  # Use os.path.dirname(os.path.abspath(__file__)) to get the current script directory
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append("../TalentAI-research-new-last-update")
sys.path.append(
    "C:/Users/adler/OneDrive/Talent.AI/TalentAI-research-new-last-update/.venv/Lib/site-packages"
)

from general_algos.Preprocess_for_hr import preProcess, KMeansClusterer
from openpyxl import Workbook
from utilss import mean_generator
from collections import Counter
from statistic_regular_algo.Statistic_dot_product import Statistic_dot_product
from statistic_regular_algo.Statistic_list_frequency import Statistic_list_frequency
from statistic_regular_algo.Statistic_intersection import Statistic_intersection
from sklearn.model_selection import train_test_split
from customCentroids import *

logging.basicConfig(filename="yasmin_error_log.txt", level=logging.ERROR)
original_stdout = sys.stdout
output_file_path = "../output.txt"

output_file = open(output_file_path, "w")

# Types list
types_list = ['categoric', 'categoric', 'categoric', 'categoric', 'numeric',
              'categoric', 'categoric', 'categoric', 'categoric', 'categoric',
              'list', 'categoric', 'categoric', 'categoric', 'list', 'list',
              'categoric', 'categoric', 'categoric', 'numeric', 'categoric',
              'categoric', 'categoric', 'categoric', 'categoric', 'categoric',
              'categoric', 'categoric', 'categoric', 'list', 'categoric',
              'categoric', 'categoric', 'categoric', 'numeric', 'list', 'list', 'list']

types_list_gender_no_age = ['categoric', 'categoric', 'categoric', 'categoric',
                            'categoric', 'categoric', 'categoric', 'categoric',
                            'list', 'categoric', 'categoric', 'categoric', 'list', 'list',
                            'categoric', 'categoric', 'categoric', 'numeric', 'categoric',
                            'categoric', 'categoric', 'categoric', 'categoric', 'categoric',
                            'categoric', 'categoric', 'categoric', 'list', 'categoric',
                            'categoric', 'categoric', 'categoric', 'numeric', 'list', 'list', 'list']

types_list_age_no_gender = ['categoric', 'categoric', 'categoric', 'numeric',
              'categoric', 'categoric', 'categoric', 'categoric', 'categoric',
              'list', 'categoric', 'categoric', 'categoric', 'list', 'list',
              'categoric', 'categoric', 'categoric', 'numeric', 'categoric',
              'categoric', 'categoric', 'categoric', 'categoric', 'categoric',
              'categoric', 'categoric', 'categoric', 'list', 'categoric',
              'categoric', 'categoric', 'categoric', 'numeric', 'list', 'list', 'list']

types_list_no_age_no_gender = ['categoric', 'categoric', 'categoric',
              'categoric', 'categoric', 'categoric', 'categoric',
              'list', 'categoric', 'categoric', 'categoric', 'list', 'list',
              'categoric', 'categoric', 'categoric', 'numeric', 'categoric',
              'categoric', 'categoric', 'categoric', 'categoric', 'categoric',
              'categoric', 'categoric', 'categoric', 'list', 'categoric',
              'categoric', 'categoric', 'categoric', 'numeric', 'list', 'list', 'list']

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def create_save_directory(base_dir, dataset_option):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = os.path.join(base_dir, f"{dataset_option}_{timestamp}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def process_and_cluster(dataset_option, distance_function):
    file_path = {
        "gender_no_age": "datasets/train_gender_no_age.csv",
        "age_no_gender": "datasets/train_age_no_gender.csv",
        "no_age_no_gender": "datasets/train_no_age_no_gender.csv",
        "with_gender_and_age": "datasets/train_with_gender_and_age.csv",
    }.get(dataset_option, None)

    if dataset_option == "gender_no_age":
        types_list_modified = types_list_gender_no_age
    elif dataset_option == "age_no_gender":
        types_list_modified = types_list_age_no_gender
    elif dataset_option == "no_age_no_gender":
        types_list_modified = types_list_no_age_no_gender
    else:
        types_list_modified = types_list  # Default case with all indices

    if file_path is None:
        raise ValueError("Invalid dataset option")  # Handle invalid options

    # Load the dataset as list of lists
    csv_data = []
    with open(file_path, "r", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            csv_data.append(row)

    vectors = [np.array(f, dtype=object) for f in csv_data]
    print("done initializing the vector with 100% of the data")

    df = pd.DataFrame(vectors)
    df = df.fillna("")  # Replace nan with empty strings

    train_vectors, test_vectors = train_test_split(vectors, test_size=0.20, random_state=42)

    # Create directories to save the sets
    save_dir = create_save_directory("saved_sets", dataset_option)
    save_to_pickle(train_vectors, os.path.join(save_dir, "train_vectors.pkl"))
    save_to_pickle(test_vectors, os.path.join(save_dir, "test_vectors.pkl"))

    # Training
    hp_train, k = preProcess(train_vectors, types_list_modified, distance_function, 9, 9)
    model = KMeansClusterer(
        num_means=k,
        distance=distance_function,
        repeats=2,
        type_of_fields=types_list_modified,
        hyper_params=hp_train,
    )
    
    model.cluster_vectorspace(train_vectors)
    train_results = model.getModelData()

    # Testing
    hp_test, k = preProcess(test_vectors, types_list_modified, distance_function, 9, 9)
    model = KMeansClusterer(
        num_means=k,
        distance=distance_function,
        repeats=2,
        type_of_fields=types_list_modified,
        hyper_params=hp_test,
    )
    model.cluster_vectorspace(test_vectors)
    test_results = model.getModelData()
    return train_results, test_results, model.all_clusters, hp_train, k, model

def find_nearest_cluster(query, model):
    min_distance = float('inf')
    closest_cluster = None
    closest_cluster_mean = None

    for cluster in model.all_clusters:
        cluster_mean = cluster.mean
        distance, _ = model._distance(query, cluster_mean, model._type_of_fields, model._hyper_parameters)
        if distance < min_distance:
            min_distance = distance
            closest_cluster = cluster
            closest_cluster_mean = cluster_mean
    return closest_cluster, closest_cluster_mean    

def rank_nearest_subcluster(query, cluster, model):
    distances = []

    for subcluster in cluster.subclusters.values():
        subcluster_centroid = subcluster.centroid 
        distance, result = model._distance(query, subcluster_centroid, model._type_of_fields, model._hyper_parameters)
        distances.append((subcluster.company, distance))

    distances.sort(key=lambda x: x[1])  # Sort by distance (ascending order)
    print("distances:" , distances)
    return distances

def recommend_jobs(query, model):
    nearest_cluster, nearest_cluster_mean = find_nearest_cluster(query, model)
    
    # Ensure nearest_cluster_mean is in the correct format
    if nearest_cluster_mean is not None and isinstance(nearest_cluster_mean, np.ndarray):
        nearest_cluster_mean = nearest_cluster_mean.tolist()
    
    ranked_subclusters = rank_nearest_subcluster(query, nearest_cluster, model)

    print("Ranked subclusters (companies) within the nearest cluster:")
    for company, distance in ranked_subclusters:
        print(f"Company: {company}, Distance: {distance}")

    return ranked_subclusters


############################################################# Activation Area #################################################################

################# change dataset_paths to: with_gender_and_age / gender_no_age / age_no_gender / no_age_no_gender  #################
################# change function name to: Statistic_list_frequency / Statistic_dot_product / Statistic_intersection #################

train_results, test_results, all_clusters, hp_train, k, model = process_and_cluster("with_gender_and_age", Statistic_list_frequency)
# df = pd.read_csv('datasets/employes_flat_version.csv')


### TO DO: 
################# change types list according to dataset: types_list/ types_list_age_no_gender/types_list_gender_no_age/ types_list_no_age_no_gender #################

## Example queries for with age and gender

# query = ["John Doe", "John", "Doe", "male", 1985, "01/01/1985", "technology", "software developer",
#          "development", "backend", "['senior']", "", "ABC Corp", "2010-01", "['coding', 'music']",
#          "['C++', 'SQL']", "Internship Inc", "2009-06", "2009-12", 2009, "IT services", "2009-01", "2010-01",
#          "TRUE", "New York, NY", "USA", "North America", "Junior Developer", "R&D", "['software training']",
#          "Tech University", "College", "Computer Science", "2005-09", 2009, "['PMP']", "['English', 'Spanish']",
#          "['A Study on Software Development']"]

query = ["Jane Smith", "Jane", "Smith", "female", 1990, "02/02/1990", "healthcare", "data analyst",
         "analysis", "data science", "['junior']", "", "XYZ Corp", "2015-05", "['hiking', 'painting']",
         "['Python', 'R']", "Internship LLC", "2013-03", "2013-09", 2013, "Health services", "2013-04", "2014-03",
         "FALSE", "Los Angeles, CA", "USA", "North America", "Data Scientist", "Data Science", "['machine learning']",
         "Harvard University", "University", "Statistics", "2008-09", 2012, "['Data Science Certificate']", "['French', 'German']",
         "['A Study on Data Analysis']"]

print("RECOMMENDATION STARTED")
ranked_subclusters = recommend_jobs(query, model)
