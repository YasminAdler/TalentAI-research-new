import sys

from avivit_res_talentai.general_algos.Preprocess_for_hr import *
import csv
import numpy as np
from avivit_res_talentai.talent_ai_algo.talentai_dot_product import Statistic_dot_product
from avivit_res_talentai.talent_ai_algo.Statistic_intersection_talentai import Statistic_intersection
from avivit_res_talentai.talent_ai_algo.Statistic_list_frequency_talentai import Statistic_list_frequency
from avivit_res_talentai.talent_ai_algo.KMeanClusterer_talentai_version import KMeansClusterer_talentai
from avivit_res_talentai.talent_ai_algo.talentai_dot_product import Statistic_dot_product
# Save the reference to the original sys.stdout
original_stdout = sys.stdout

# Specify the file path where you want to redirect the prints
output_file_path = '../logger_of_dana.txt'

# Open the file in write mode, this will create the file if it doesn't exist
output_file = open(output_file_path, 'w')

# Redirect sys.stdout to the file
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

with open('../datasets/employes_flat_version.csv', 'r', encoding='utf-8') as csvfile:  # employes.csv
    # Create a CSV reader object
    csv_data = []
    csvreader = csv.reader(csvfile)

    for row in csvreader:
        csv_data.append(row)

vectors = [np.array(f, dtype=object) for f in csv_data]
hp, k = preProcess(vectors, types_list, Statistic_intersection, 9, 9)

# make categoric vals as numeric freqs
for vec in vectors:
    for i in range(len(types_list)):
        if (types_list[i]=="categoric"):
            if (vec[i])!="":
                vec[i] = hp["frequencies"][str(i)][vec[i]]
            else:
                vec[i]=1
        # # #TODO: these lines are only for list frequency
        # if (types_list[i]=="list"):
        #     new_lst=[]
        #     old_lst=ast.literal_eval(vec[i])
        #     new_lst=[]
        #     for j in old_lst:
        #         new_lst.append(hp["list_freq_dict"][i][j])
        #     vec[i]=new_lst

vectors = [array.tolist() for array in vectors]


print("####################3making model of intersection")
model = KMeansClusterer_talentai(num_means=k,
                        distance=Statistic_intersection,
                        repeats=8,
                        type_of_fields=types_list,
                        hyper_params=hp)

model.cluster_vectorspace(vectors)

print("done making model")

# model.calc_min_max_dist(vectors)
model.get_wcss()
model.calc_distance_between_clusters()
exit()

##################################################################3

with open('../datasets/employes_flat_version.csv', 'r', encoding='utf-8') as csvfile:  # employes.csv
    # Create a CSV reader object
    csv_data = []
    csvreader = csv.reader(csvfile)
    i += 1
    # Iterate through each row in the CSV file

    for row in csvreader:
        # print(i)
        # print(row)
        # Append each row as a list to the csv_data list
        csv_data.append(row)

vectors = [np.array(f, dtype=object) for f in csv_data]
# model.calc_min_max_dist(vectors)
#vectors=convert_month_year_to_year(vectors)


hp, k = preProcess(vectors, types_list, Statistic_dot_product, 9, 9)

# make categoric vals as numeric freqs
for vec in vectors:
    for i in range(len(types_list)):
        if (types_list[i]=="categoric"):
            vec[i]=hp["frequencies"][str(i)][vec[i]]

vectors = [array.tolist() for array in vectors]


print("######################making model of dot product")
model = KMeansClusterer_talentai(num_means=k,
                        distance=Statistic_dot_product,
                        repeats=5,
                        type_of_fields=types_list,
                        hyper_params=hp)

model.cluster_vectorspace(vectors)

print("done making model")

# model.calc_min_max_dist(vectors)
model.get_wcss()
model.calc_distance_between_clusters()

##########################################################3



