########################## INSTRUCTIONS ##########################
## BEFORE CREATING A MODEL READ THIS: 
##
## Before choosing a dataset and a distance function: 
## Initiate dataSplitter.py
## uncomment the chosen distance fucntion part in statistic_regular_algo/KMeanClusterer.py
## Uncomment the part of distance function in  main_files/customCentroids
## Input the company index in the chosen distance function file: (statistic_regular_algo/(Statistic_list_frequeny OR Statistic_intersection))
## Uncomment normalization part for chosen dataset in distance function file: (Statistic_list_frequeny / Statistic_intersection)
###################################################################

import sys
import logging
import os
import pickle
import numpy as np
import csv
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append("../TalentAI-research-new-last-update")
sys.path.append("C:/Users/adler/OneDrive/Talent.AI/TalentAI-research-new-last-update/.venv/Lib/site-packages")

from general_algos.Preprocess_for_hr import preProcess, KMeansClusterer
from statistic_regular_algo.Statistic_list_frequency import Statistic_list_frequency
from statistic_regular_algo.Statistic_intersection import Statistic_intersection

logging.basicConfig(filename="yasmin_error_log.txt", level=logging.ERROR)

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

distance_functions = {
    "Statistic_list_frequency": Statistic_list_frequency,
    "Statistic_intersection": Statistic_intersection
}

dataset_options = {
    "with_gender_and_age": types_list,
    "gender_no_age": types_list_gender_no_age,
    "age_no_gender": types_list_age_no_gender,
    "no_age_no_gender": types_list_no_age_no_gender
}
company_indices = {
    "with_gender_and_age": 11,
    "gender_no_age": 10,
    "age_no_gender": 10,
    "no_age_no_gender": 9
}

def process_and_cluster(full_vectors, train_vectors, dataset_option, distance_function, types_list_modified):
    save_dir = f"saved_models/{dataset_option}"
    os.makedirs(save_dir, exist_ok=True)

    try:
        print("Preprocessing the full dataset...")
        hp, k = preProcess(full_vectors, types_list_modified, distance_function, 9, 9)

        print("Clustering the train set...")
        print("company_index", company_indices[dataset_option])
        model = KMeansClusterer(
            num_means=k,
            distance=distance_function,
            repeats=5,  # changed by yasmin from 5 to 2
            company_index=company_indices[dataset_option],
            type_of_fields=types_list_modified,
            hyper_params=hp,
        )

        # Error handling during clustering
        try:
            model.cluster_vectorspace(train_vectors)
        except Exception as e:
            logging.error(f"Error during clustering: {e}")
            print(f"An error occurred during clustering: {e}")
            return

        train_results = model.getModelData()

        # Validate clusters to ensure they are properly initialized
        if not model.all_clusters or any(c is None for c in model.all_clusters):
            logging.error("One or more clusters are not initialized correctly.")
            print("Error: One or more clusters are not initialized correctly.")
            return

        save_path_train = os.path.join(save_dir, f"{dataset_option}_{distance_function.__name__}_train_model.pkl")
        with open(save_path_train, "wb") as f:
            pickle.dump({
                "train_results": train_results,
                "all_clusters": model.all_clusters,
                "hp": hp,
                "k": k,
                "model": model,
                "type_of_fields": types_list_modified
            }, f)
            
        print(f"Training model and results saved to {save_path_train}")
        
    except Exception as e:
        logging.error(f"Unexpected error during processing and clustering: {e}")
        print(f"An unexpected error occurred: {e}")
        


def main():
    print("Choose a dataset option:")
    for idx, option in enumerate(dataset_options.keys()):
        print(f"{idx + 1}. {option}")
    dataset_choice = int(input("Enter the number of your choice: ")) - 1
    dataset_option = list(dataset_options.keys())[dataset_choice]
    types_list_modified = dataset_options[dataset_option]

    print("Choose a distance function:")
    for idx, option in enumerate(distance_functions.keys()):
        print(f"{idx + 1}. {option}")
    distance_choice = int(input("Enter the number of your choice: ")) - 1
    distance_function_name = list(distance_functions.keys())[distance_choice]
    distance_function = distance_functions[distance_function_name]

    train_file_path = os.path.join("datasets", f"train_{dataset_option}.csv")
    full_file_path= os.path.join("datasets", f"full_{dataset_option}.csv")

    if not os.path.exists(train_file_path):
        print(f"Train file for {dataset_option} not found.")
        sys.exit(1)
        
    if not os.path.exists(full_file_path):
        print(f"Full file for {dataset_option} not found.")
        sys.exit(1)
        
    train_vectors = []
    full_vectors = []
    
    with open(train_file_path, "r", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            train_vectors.append(np.array(row, dtype=object))
            
    with open(full_file_path, "r", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            full_vectors.append(np.array(row, dtype=object))
    process_and_cluster(full_vectors, train_vectors, dataset_option, distance_function, types_list_modified)

if __name__ == "__main__":
    main()
