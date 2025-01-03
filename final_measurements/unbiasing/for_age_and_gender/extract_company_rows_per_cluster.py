import sys
import logging
import os
import pickle
import csv

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append("../TalentAI-research-new-last-update")
sys.path.append("C:/Users/adler/OneDrive/Talent.AI/TalentAI-research-new-last-update/.venv/Lib/site-packages")
sys.path.append(os.path.join(script_dir, "../main_files"))

from general_algos.Preprocess_for_hr import preProcess, KMeansClusterer
from statistic_regular_algo.Statistic_list_frequency import Statistic_list_frequency
from statistic_regular_algo.Statistic_intersection import Statistic_intersection

logging.basicConfig(filename="yasmin_error_log.txt", level=logging.ERROR)

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

    return model, hp, type_of_fields


def save_cluster_rows_to_csv(model, output_csv_path):
    """
    Save the actual rows (data points) in each cluster to a CSV file.

    :param model: The trained KMeansClusterer model with clusters.
    :param output_csv_path: Path to the CSV file where the cluster rows will be saved.
    """
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write the header
        csvwriter.writerow(["Cluster ID", "Row Data"])

        # Iterate over each cluster in the model
        for cluster in model.all_clusters:
            if cluster is None:
                continue  # Skip any uninitialized clusters

            # Write each data point (row) in the cluster to the CSV
            for data_point in cluster.data[cluster.cluster_id]:
                csvwriter.writerow([cluster.cluster_id, data_point])


# Define the paths to saved models
model_paths = [
    'saved_models/with_gender_and_age/with_gender_and_age_Statistic_intersection_train_model.pkl',
    'saved_models/with_gender_and_age/with_gender_and_age_Statistic_list_frequency_train_model.pkl',
    'saved_models/no_age_no_gender/no_age_no_gender_Statistic_intersection_train_model.pkl',
    'saved_models/no_age_no_gender/no_age_no_gender_Statistic_list_frequency_train_model.pkl',
    'saved_models/gender_no_age/gender_no_age_Statistic_intersection_train_model.pkl',
    'saved_models/gender_no_age/gender_no_age_Statistic_list_frequency_train_model.pkl',
    'saved_models/age_no_gender/age_no_gender_Statistic_intersection_train_model.pkl',
    'saved_models/age_no_gender/age_no_gender_Statistic_list_frequency_train_model.pkl'
]

# Process all the models
for model_path in model_paths:
    try:
        # Load the model
        model, hp, type_of_fields = load_model(model_path)
        print(f"\nProcessing model: {model_path}")
        
        # Define output CSV path for each model
        output_csv_path = f"clusters_{os.path.basename(model_path).replace('.pkl', '.csv')}"
        
        # Save the actual rows per cluster into a CSV file
        save_cluster_rows_to_csv(model, output_csv_path)

        print(f"Saved cluster data to {output_csv_path}")

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except Exception as e:
        print(f"An error occurred while processing {model_path}: {e}")
