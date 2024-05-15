import ast
import sys

from avivit_res_talentai.general_algos.Preprocess_for_wine import *
import csv
import numpy as np
from avivit_res_talentai.talent_ai_algo.talentai_dot_product import Statistic_dot_product
from avivit_res_talentai.talent_ai_algo.Statistic_intersection_talentai import Statistic_intersection
from avivit_res_talentai.talent_ai_algo.Statistic_list_frequency_talentai import Statistic_list_frequency
from avivit_res_talentai.talent_ai_algo.KMeanClusterer_talentai_version import KMeansClusterer_talentai

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

types_list = ['categoric', 'categoric', 'categoric', 'categoric', 'list',
              'list', 'numeric', 'categoric', 'categoric', 'categoric',
              'categoric', 'categoric', 'categoric',
              'categoric', 'categoric', 'categoric', 'list']

i = 0

with open('../datasets/wine.txt', 'r', encoding='utf-8') as csvfile:
    # Create a CSV reader object
    csv_data = []
    csvreader = csv.reader(csvfile)
    i += 1
    # Iterate through each row in the CSV file

    for row in csvreader:
        csv_data.append(row)

vectors = [np.array(f, dtype=object) for f in csv_data]

hp, k = preProcess(vectors, types_list, Statistic_intersection, 9, 9)

# make list vals as numeric freqs
for vec in vectors:
    for i in range(len(types_list)):
        if (types_list[i] == "categoric"):
            if (vec[i])!="":
                vec[i] = hp["frequencies"][str(i)][vec[i]]
            else:
                print("fff")
                vec[i]=1
        # #TODO: this if should be comment for one hot vector methods
        # if (types_list[i]=="list"):
        #     old_lst=ast.literal_eval(vec[i])
        #     new_lst=[]
        #     for j in old_lst:
        #         #print(hp["list_freq_dict"][i][j])
        #         new_lst.append(hp["list_freq_dict"][i][j])
        #     vec[i]=new_lst

vectors = [array.tolist() for array in vectors]
print(vectors)

print("making model of intersection")
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

#############################################################3

print("######################3making model of dot product")

hp, k = preProcess(vectors, types_list, Statistic_dot_product, 9, 9)
model = KMeansClusterer(num_means=k,
                        distance=Statistic_dot_product,
                        repeats=15,
                        type_of_fields=types_list,
                        hyper_params=hp)

model.cluster_vectorspace(vectors)

print("done making model")

model.calc_min_max_dist(vectors)
print("min distance is", model.min_dist)
print("max distance is", model.max_dist)
model.get_wcss()
model.calc_distance_between_clusters()

exit()
##########################################################3
