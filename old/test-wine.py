import sys

from avivit_res_talentai.general_algos.Preprocess_for_wine import *
import csv
import numpy as np
from avivit_res_talentai.statistic_regular_algo.Statistic_dot_product import Statistic_dot_product
from avivit_res_talentai.statistic_regular_algo.Statistic_intersection import Statistic_intersection
from avivit_res_talentai.statistic_regular_algo.Statistic_list_frequency import Statistic_list_frequency

from avivit_res_talentai.statistic_regular_algo.MixedDistance import MixedDistance

# Save the reference to the original sys.stdout
original_stdout = sys.stdout

# Specify the file path where you want to redirect the prints
output_file_path = 'logger_of_dana_wine_iranians.txt'

# Open the file in write mode, this will create the file if it doesn't exist
output_file = open(output_file_path, 'w')

# # Redirect sys.stdout to the file
# sys.stdout = output_file

types_list = ['categoric', 'categoric', 'categoric', 'categoric', 'list',
              'list', 'numeric', 'categoric', 'categoric', 'categoric',
              'categoric', 'categoric', 'categoric',
              'categoric', 'categoric', 'categoric', 'list']

i = 0
with open('../datasets/wine.txt', 'r', encoding='utf-8') as csvfile:  # employes.csv
    # Create a CSV reader object
    csv_data = []
    csvreader = csv.reader(csvfile)
    i += 1
    # Iterate through each row in the CSV file

    for row in csvreader:
        csv_data.append(row)

print("done rows")
vectors = [np.array(f, dtype=object) for f in csv_data]

hp, k = preProcess(vectors, types_list,Statistic_dot_product , 9, 9)
for i in range(len(types_list)):
    if types_list[i] =="categoric":
        print(i, hp['domain si100zes'][i])

exit()

print(hp['domain sizes'])
print(len(hp["one_hot_vector_prep"][4]))
print(len(hp["one_hot_vector_prep"][5]))
print(len(hp["one_hot_vector_prep"][16]))
exit()

 # need to comment out the part that refers to one hot vec in kmeansclusterer
print("making model of Statistic_dotproduct")
model = KMeansClusterer(num_means=k,
                        distance=Statistic_dot_product,
                        repeats=20,
                        type_of_fields=types_list,
                        hyper_params=hp)



model.cluster_vectorspace(vectors)

print("done making model")

model.calc_min_max_dist(vectors)
model.get_wcss()
model.calc_distance_between_clusters()
print("min distance is", model.min_dist)
print("max distance is", model.max_dist)

#######################################3


hp, k = preProcess(vectors, types_list,Statistic_intersection, 9, 9)


print("making model of Statistic_intersection")
model = KMeansClusterer(num_means=k,
                        distance=Statistic_intersection,
                        repeats=20,
                        type_of_fields=types_list,
                        hyper_params=hp)



model.cluster_vectorspace(vectors)

print("done making model")

model.calc_min_max_dist(vectors)
model.get_wcss()
model.calc_distance_between_clusters()
print("min distance is", model.min_dist)
print("max distance is", model.max_dist)


exit()
##################################################################3
