################################## Yasmin's testing part:##################################
                                        
#                                         plan: 

#                   splitting the dataset into 80% training, 20% test
#                   creating models repteatedly with the training set
#                   running Dana's code with the dot product and intersection algorithm (Kmeans + "shallowing" data types + counting frequencies)
#                   doing the same with the 20% test data
#                   compare the results of the two and analyze the results




import sys
sys.path.append("../TalentAI-dana's code-new")
sys.path.append("C:/Users/adler/AppData/Local/Programs/Python/Python311/Lib/site-packages")

from general_algos.Preprocess_for_hr import preProcess, KMeansClusterer
import csv
import numpy as np
import pandas as pd
import os as os 
import openpyxl as openpyxl
from openpyxl import Workbook
from statistic_regular_algo.Statistic_dot_product import Statistic_dot_product
from statistic_regular_algo.Statistic_list_frequency import Statistic_list_frequency
from statistic_regular_algo.Statistic_intersection import Statistic_intersection
from statistic_regular_algo.MixedDistance import MixedDistance
from sklearn.model_selection import train_test_split 

# Save the reference to the original sys.stdout
original_stdout = sys.stdout

# Specify the file path where you want to redirect the prints
output_file_path = '../output.txt'

# Open the file in write mode, this will create the file if it doesn't exist
output_file = open(output_file_path, 'w')

# Redirect sys.stdout to the file, comment this out if no need for print
# sys.stdout = output_file


import numpy as np
import re


types_list = ['categoric', 'categoric', 'categoric', 'categoric', 'numeric',
              'categoric', 'categoric', 'categoric', 'categoric', 'categoric',
              'list', 'categoric', 'categoric', 'categoric', 'list', 'list',
              'categoric', 'categoric', 'categoric', 'numeric', 'categoric',
              'categoric', 'categoric', 'categoric', 'categoric', 'categoric',
              'categoric', 'categoric', 'categoric', 'list', 'categoric',
              'categoric', 'categoric', 'categoric', 'numeric', 'list', 'list', 'list']

i = 0
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get directory where the script is located
csv_file_path = os.path.join(script_dir, '..', 'datasets', 'employes_flat_version.csv')

with open(csv_file_path, 'r', encoding='utf-8') as csvfile:  # employes.csv
    # Create a CSV reader object
    csv_data = []
    csvreader = csv.reader(csvfile)
    i += 1
    # Iterate through each row in the CSV file

    for row in csvreader:
        # Append each row as a list to the csv_data list
        csv_data.append(row)

vectors = [np.array(f, dtype=object) for f in csv_data]
print("done initializing the vector with 100\% of the data")

df = pd.DataFrame(vectors)

# splitting the dataset into 80% training, 20% test:
# train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
train_vectors, test_vectors = train_test_split(vectors, test_size=0.20, random_state=42)

# ######################## training ########################

# # train_vectors = train_df.values 

# hp, k = preProcess(train_vectors, types_list, Statistic_intersection, 9, 9)
# # in order to run this you need to comment out the part that refers to one hot vector in kmeansclusterer

# print("making model of dot for train data")
# model = KMeansClusterer(num_means=k,
#                         distance=Statistic_intersection,
#                         repeats=1,
#                         type_of_fields=types_list,
#                         hyper_params=hp)

# print("done initializing model with KMeansClusterer class")

# # print(hp["list_freq_dict"])

# model.cluster_vectorspace(train_vectors)

# print("done making model")

# # model.calc_min_max_dist(vectors)
# model.get_wcss()
# model.calc_distance_between_clusters()

# train_results = model.getModelData()

# exit()

############################# test #################################


# test_vectors = test_df.values

hp, k = preProcess(test_vectors, types_list, Statistic_intersection, 9, 9)
# in order to run this you need to comment out the part that refers to one hot vector in kmeansclusterer

print("making model of dot for test data")
model = KMeansClusterer(num_means=k,
                        distance=Statistic_intersection,
                        repeats=1, #was 8 
                        type_of_fields=types_list,
                        hyper_params=hp)

# print(hp["list_freq_dict"])

model.cluster_vectorspace(test_vectors)

print("done making model")

# model.calc_min_max_dist(vectors)
model.get_wcss()
model.calc_distance_between_clusters()

test_results = model.getModelData()


################# Saving Results and Analysis #################

# writer = pd.ExcelWriter('clustering_results.xlsx', engine='openpyxl')

# # Convert results to DataFrame if they're not already, and write to Excel
# # pd.DataFrame(train_results).to_excel(writer, sheet_name='Training Results')
# # pd.DataFrame(test_results).to_excel(writer, sheet_name='Testing Results')

print(test_results)
# writer.save()

# def write_results_to_excel(filename, train_results, test_results):
#     workbook = Workbook()
    
#     # Write training results
#     train_sheet = workbook.create_sheet("Training Results")
#     for i, row in enumerate(train_results):
#         for j, val in enumerate(row):
#             cell = train_sheet.cell(row=i+1, column=j+1)
#             cell.value = val
    
#     # Write testing results
#     test_sheet = workbook.create_sheet("Testing Results")
#     for i, row in enumerate(test_results):
#         for j, val in enumerate(row):
#             cell = test_sheet.cell(row=i+1, column=j+1)
#             cell.value = val
    
#     # Save the workbook
#     workbook.save(filename)
#     print(f"Results saved successfully to {filename}")

# # Call the function to save results
# write_results_to_excel('clustering_results.xlsx', train_results, test_results)

# ################ Comparative Analysis #################

# try:
#     train_silhouette = model.get_Silhouette(train_vectors)
#     test_silhouette = model.get_Silhouette(test_vectors)
#     print(f"Training Silhouette Score: {train_silhouette}")
#     print(f"Testing Silhouette Score: {test_silhouette}")
# except Exception as e:
#     print("An error occurred while calculating silhouette scores:", str(e))

# # Closing the output file
# sys.stdout = original_stdout
# output_file.close()

exit()

##################################################################
print("###################### making model of Statistic_intersection ")
hp, k = preProcess(vectors, types_list, Statistic_intersection, 9, 9)
model = KMeansClusterer(num_means=k,
                        distance=Statistic_intersection,
                        repeats=5,
                        type_of_fields=types_list,
                        hyper_params=hp)
print(hp["frequencies"])
model.cluster_vectorspace(vectors)

print("done making model")

model.get_wcss()
model.calc_distance_between_clusters()
exit()
##########################################################3

