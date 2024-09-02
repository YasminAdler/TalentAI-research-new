import sys
import os
import csv
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_model(filename):
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def calculate_inter_cluster_distances(model):
    all_clusters = model['model'].all_clusters
    assert len(all_clusters) == 8, "Each model should have exactly 8 clusters."
    
    distances = []
    
    for i in range(8): 
        for j in range(i + 1, 8):  # Calculate only once for each pair
            dist = model['model']._distance(
                all_clusters[i].mean, 
                all_clusters[j].mean, 
                model['model']._type_of_fields, 
                model['model']._hyper_parameters
            )[0]
            distances.append({
                'Cluster_A': f'Cluster_{i}',
                'Cluster_B': f'Cluster_{j}',
                'Distance': dist
            })
                
    assert len(distances) == 28, "There should be exactly 28 distances calculated."
    return distances

def save_distances_to_csv(distances, filename):
    keys = distances[0].keys() if distances else ['Cluster_A', 'Cluster_B', 'Distance']
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(distances)

output_dir = "saved_measurments"  # Directory to save CSV files
os.makedirs(output_dir, exist_ok=True)

dataset_options = ["with_gender_and_age", "gender_no_age", "age_no_gender", "no_age_no_gender"]
distance_functions = ["Statistic_list_frequency", "Statistic_dot_product", "Statistic_intersection"]

for dataset_option in dataset_options:
    for distance_function in distance_functions:
        model_filename = os.path.join("saved_models", f"{dataset_option}",f"{dataset_option}_{distance_function}_train_model.pkl")
        
        if os.path.exists(model_filename):
            model_data = load_model(model_filename)
            distances = calculate_inter_cluster_distances(model_data)
            
            output_filename = os.path.join(output_dir, f"{dataset_option}_{distance_function}_inter_cluster_distances.csv")
            save_distances_to_csv(distances, output_filename)
            print(f"Distances saved to {output_filename}")
        else:
            print(f"Model file {model_filename} not found.")
