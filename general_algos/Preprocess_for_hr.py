import ast
import math
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from statistic_regular_algo.KMeanClusterer import KMeansClusterer
from datetime import datetime
import numpy as np
from collections import Counter

MAX_CLUSTERS_IN_ELBOW = 10
MIN_CLUSTERS_IN_ELBOW = 1


# Define a custom function to count unique elements
def count_unique_elements(column):
    unique_elements = set()

    for item in column:
        if isinstance(item, list):
            unique_elements.add(frozenset(item))  # Convert lists to frozensets for hashing
        else:
            unique_elements.add(item)

    return len(unique_elements)


def find_dict_of_freqs(index_to_count, vectors):
    # Extract the specified index column from each NumPy array
    attribute_column = [vector[index_to_count] for vector in vectors]

    # Use Counter to count the frequencies of attributes
    attribute_counts = Counter(attribute_column)

    # Convert the Counter object to a dictionary
    attribute_counts_dict = dict(attribute_counts)

    # Now, attribute_counts_dict contains the frequencies of attributes from the specified index
    return attribute_counts_dict  # Return the dictionary instead of printing it


def apply_elbow_method(fields_data, vectors, distance_function, triesNumber, _repeats, params):
    # hr dataset- 8
    wcss=[]
    print("doing elbow")
    for i in range(1,10):
        print("iteration ", i ,"of elbow")
        model = KMeansClusterer(hyper_params=params, distance=distance_function, num_means=int(i),
                                type_of_fields=fields_data, repeats=5)

        model.cluster_vectorspace(vectors)
        print("yay!",i, model.get_wcss())
        wcss.append(model.get_wcss())

    print(wcss)
    plt.plot(range(1, 10), wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Cost')
    plt.show()
    exit()



def preProcess(vectors, fieldsData, distance_function, triesNumber, repeats):
    type_of_fields = fieldsData
    # preparation for the one hot vector - extract all of the possible values of each list

    # list_of_lists=[]
    dict_of_lists = dict()
    for typee in range(len(type_of_fields)):
        if type_of_fields[typee] == "list":
            dict_of_lists[typee] = set()
            for elem in vectors:

                output_list = ast.literal_eval(elem[typee])

                for i in output_list:
                    dict_of_lists[typee].add(i)

    for i in dict_of_lists.keys():
        dict_of_lists[i] = list(dict_of_lists[i])



    # end of one hot vector preparation

    params_dict = dict()
    params_dict["one_hot_vector_prep"] = dict_of_lists

    df = pd.DataFrame(vectors)
    domain_sizes = df.nunique()


    params_dict["domain sizes"] = domain_sizes.tolist()

    frequencies_dict = dict()
    minimal_frequencies_dict = dict()
    max_frequencies_dict = dict()
    z = 0
    for i in range(len(fieldsData)):
        if type_of_fields[i] == 'categoric':

            frequencies_dict[str(i)] = find_dict_of_freqs(i, vectors)

            minimal_frequencies_dict[str(i)] = min(frequencies_dict[str(i)].values())
            max_frequencies_dict[str(i)] = max(frequencies_dict[str(i)].values())
            if max(frequencies_dict[str(i)].values()) > z:
                z = max(frequencies_dict[str(i)].values())
        else:
            frequencies_dict[str(i)] = dict()
            minimal_frequencies_dict[str(i)] = dict()
            max_frequencies_dict[str(i)] = dict()

    params_dict["frequencies"] = frequencies_dict
    params_dict["minimum_freq_of_each_attribute"] = minimal_frequencies_dict
    params_dict["theta"] = 0.1

    time = datetime.now()

    # calculate the average list length for list frequency method
    list_freq_dict = dict()

    for i in range(len(type_of_fields)):
        if (type_of_fields[i] == "list"):
            list_lengths = [len(ast.literal_eval(arr[i])) for arr in vectors]
            list_freq_dict[i] = math.ceil(np.mean(list_lengths))


    # calculate average list length
    params_dict["avg_list_len"] = list_freq_dict
    freq_dict = dict()

    for i in range(len(type_of_fields)):
        if (type_of_fields[i] == "list"):
            freq_dict[i] = defaultdict(int)
            for vec in range(len(vectors)):
                for elem in ast.literal_eval(vectors[vec][i]):
                    # todo: freq is 1 is "".
                    freq_dict[i][elem] += 1
    params_dict["list_freq_dict"] = freq_dict

    k = 8 #apply_elbow_method(type_of_fields, vectors, distance_function, triesNumber, repeats, params_dict)


    print("started genetic algorithm")
    # hr dataset-  ( 4, 13, 0.07, 0.06)
    theta1, theta2, betha, gamma =  ( 4, 13, 0.07, 0.06)#genetic_algorithm(params_dict, distance_function, k, vectors, type_of_fields, z)
    print(theta1, theta2, betha, gamma)
    #exit()
    print("GENETIC COMPLETED AND TOOK:", (datetime.now() - time).seconds, "seconds")

    params_dict["theta1"] = theta1  # 3
    params_dict["theta2"] = theta2  # 10
    params_dict["betha"] = betha  # 0.05
    params_dict["gamma"] = gamma  # 0.01
    # print('done genetics: ', params_dict)
    return params_dict, k
