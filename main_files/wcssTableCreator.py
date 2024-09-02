import os
import sys
import pickle
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'main_files')))
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append("../TalentAI-research-new-last-update")
sys.path.append("C:/Users/adler/OneDrive/Talent.AI/TalentAI-research-new-last-update/.venv/Lib/site-packages")


base_directory = os.path.abspath('saved_models')
main_files_directory = os.path.abspath('main_files')

sys.path.append(main_files_directory)

def load_model_and_extract_wcss(model_path):
    with open(model_path, 'rb') as file:
        model_data = pickle.load(file)
        model = model_data['model']
        wcss = model._wcss  # Access the _wcss attribute from the KMeansClusterer instance
    return wcss

def create_wcss_table(model_paths):
    wcss_scores = []

    for model_name, model_path in model_paths.items():
        wcss = load_model_and_extract_wcss(model_path)
        wcss_scores.append({'Model Name': model_name, 'WCSS Score': wcss})

    df_wcss = pd.DataFrame(wcss_scores)
    return df_wcss


model_paths = {
    "age_no_gender_Statistic_list_frequency": os.path.join(base_directory,'age_no_gender','age_no_gender_Statistic_list_frequency_train_model.pkl'),
    "age_no_gender_Statistic_intersection": os.path.join(base_directory,'age_no_gender', 'age_no_gender_Statistic_intersection_train_model.pkl'),
    "gender_no_age_Statistic_intersection": os.path.join(base_directory, 'gender_no_age', 'gender_no_age_Statistic_intersection_train_model.pkl'),
    "gender_no_age_Statistic_list_frequency": os.path.join(base_directory, 'gender_no_age', 'gender_no_age_Statistic_list_frequency_train_model.pkl'),
    "no_age_no_gender_Statistic_intersection": os.path.join(base_directory, 'no_age_no_gender', 'no_age_no_gender_Statistic_intersection_train_model.pkl'),
    "no_age_no_gender_Statistic_list_frequency": os.path.join(base_directory, 'no_age_no_gender', 'no_age_no_gender_Statistic_list_frequency_train_model.pkl'),
    "with_gender_and_age_Statistic_intersection": os.path.join(base_directory, 'with_gender_and_age', 'with_gender_and_age_Statistic_intersection_train_model.pkl'),
    "with_gender_and_age_Statistic_list_frequency": os.path.join(base_directory, 'with_gender_and_age', 'with_gender_and_age_Statistic_list_frequency_train_model.pkl')
}
wcss_table = create_wcss_table(model_paths)

# Save the table to a CSV file
wcss_table.to_csv('wcss_scores.csv', index=False)

# Display the table
print(wcss_table)
