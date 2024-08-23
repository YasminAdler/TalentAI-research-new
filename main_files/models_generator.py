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
from utilss import mean_generator
from collections import Counter
from statistic_regular_algo.Statistic_dot_product import Statistic_dot_product
from statistic_regular_algo.Statistic_list_frequency import Statistic_list_frequency
from statistic_regular_algo.Statistic_intersection import Statistic_intersection

logging.basicConfig(filename="yasmin_error_log.txt", level=logging.ERROR)

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

distance_functions = {
    "Statistic_list_frequency": Statistic_list_frequency,
    "Statistic_dot_product": Statistic_dot_product,
    "Statistic_intersection": Statistic_intersection
}

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def create_save_directory(base_dir, dataset_option, distance_function):
    dir_path = os.path.join(base_dir, f"{dataset_option}_{distance_function}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def load_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def filter_vectors(vectors, types_list):
    filtered_vectors = []
    for vector in vectors:
        filtered_vector = [value for value, type in zip(vector, types_list) if type != '']
        filtered_vectors.append(filtered_vector)
    return filtered_vectors

def process_and_cluster(train_vectors, dataset_option, distance_function, types_list_modified):
    save_dir = create_save_directory("saved_sets", dataset_option, distance_function.__name__)

    # Training
    hp_train, k = preProcess(train_vectors, types_list_modified, distance_function, 9, 9)
    model = KMeansClusterer(
        num_means=k,
        distance=distance_function,
        repeats=5,
        type_of_fields=types_list_modified,
        hyper_params=hp_train,
    )
    model.cluster_vectorspace(train_vectors)
    train_results = model.getModelData()

    # Save training model
    save_path_train = f"{save_dir}/{dataset_option}_{distance_function.__name__}_train_model.pkl"
    with open(save_path_train, "wb") as f:
        pickle.dump({
            "train_results": train_results,
            "all_clusters": model.all_clusters,
            "hp_train": hp_train,
            "k": k,
            "model": model
        }, f)
    print(f"Training model and results saved to {save_path_train}")

def main():
    dataset_options = {
        "with_gender_and_age": types_list,
        "gender_no_age": types_list_gender_no_age,
        "age_no_gender": types_list_age_no_gender,
        "no_age_no_gender": types_list_no_age_no_gender
    }

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

    # Load the corresponding dataset
    file_path = {
        "gender_no_age": "datasets/train_gender_no_age.csv",
        "age_no_gender": "datasets/train_age_no_gender.csv",
        "no_age_no_gender": "datasets/train_no_age_no_gender.csv",
        "with_gender_and_age": "datasets/train_with_gender_and_age.csv",
    }.get(dataset_option, None)

    if file_path is None:
        print("Invalid dataset option selected.")
        sys.exit(1)

    # Load the dataset
    csv_data = []
    with open(file_path, "r", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            csv_data.append(row)

    train_vectors = [np.array(f, dtype=object) for f in csv_data]

    process_and_cluster(train_vectors, dataset_option, distance_function, types_list_modified)

if __name__ == "__main__":
    main()
