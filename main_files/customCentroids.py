
import numpy as np
import ast
from collections import Counter
import sys
import os as os

script_dir = os.path.dirname(os.path.abspath(__file__))  
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append("../TalentAI-research-new-last-update")
sys.path.append(
    "C:/Users/adler/OneDrive/Talent.AI/TalentAI-research-new-last-update/.venv/Lib/site-packages"
)


class CustomCentroidCalculator:
    def __init__(self, types_list, hyper_params):
        self._type_of_fields = types_list
        self._hyper_parameters = hyper_params

    def calculate_centroid(self, cluster):
        if len(cluster):

            frequent_value_list = []
            for ind in range(len(cluster[0])):

                if self._type_of_fields[ind] == "categoric":
                    # todo: if the most frequent value is "", then choose the second next. if all are missing val, chose missing val. done
                    counter = Counter(arr[ind] for arr in cluster if len(arr) > ind)
                    most_common = counter.most_common(2)
                    frequent_value_list.append(
                        most_common[1][0] if most_common[0][0] == "" and len(most_common) > 1 else most_common[0][0] if
                        most_common[0][0] != "" else "")
                    # counter = Counter(arr[ind] for arr in cluster if len(arr) > ind)
                    # frequent_value_list.append(counter.most_common(1)[0][0])

                if self._type_of_fields[ind] == "numeric":
                    # ignore missing values
                    values = [float(arr[ind]) for arr in cluster if arr[ind] != '']
                    result = np.mean(values) if len(values) > 0 else 0
                    frequent_value_list.append(np.mean(result))

                if self._type_of_fields[ind] == "list":

                    # ############ this version is for hamming talentai method (סדר מקורי)

                    # lists_at_ind_index = [ast.literal_eval(vector[ind]) for vector in cluster]
                    
                    # # this line is only for hamming sorted
                    # lists_at_ind_index = [sorted(sublist, key=lambda x: self._hyper_parameters["list_freq_dict"][ind].get(x, 0), reverse=True) for sublist
                    #                      in lists_at_ind_index]
                    
                    # padded_data = [lst + ['missing_val'] * (self._hyper_parameters["avg_list_len"][ind] - len(lst)) for
                    #                lst in lists_at_ind_index]
                    
                    # transposed_data = zip(*padded_data)
                    # # Initialize the voting list
                    # voting_list = []
                    # # Iterate over each column
                    # for column in transposed_data:
                    #     # Count occurrences of each value in the column
                    #     counts = Counter(column)
                    #     # Find the most common values
                    #     most_common_values = counts.most_common(2)
                    
                    #     # Check if the most common value is "missing_val"
                    #     if most_common_values[0][0] == "missing_val":
                    #         if len(most_common_values) >= 2:
                    #             most_common_value = most_common_values[1][0]
                    #         else:
                    #             most_common_value = "missing_val"
                    #     else:
                    #         most_common_value = most_common_values[0][0]
                    
                    #     # todo: most common value needs to be a real value, exclude missing vals - done!
                    #     # Append the most common value to the voting list
                    #     voting_list.append(most_common_value)
                    # data = voting_list


                #     ########### this version is for dot product and intersection
                # #    Extract lists from the indth index of each vector

                #     lists_at_ind_index = [ast.literal_eval(vector[ind]) for vector in cluster]

                #     # Flatten the lists and count occurrences of each value
                #     flattened_list = [item for sublist in lists_at_ind_index for item in sublist]
                #     value_counts = Counter(flattened_list)

                #     # Find values that appear in at least 50% of the lists
                #     threshold = len(cluster) / 2
                #     most_common_values = [value for value, count in value_counts.items() if count >= threshold]
                #     data = most_common_values

                    ############## this version is for list frequency (איראנים רשימה) #####
                    avg_length = self._hyper_parameters["avg_list_len"][ind]
                    # extract lists
                    lists_at_index = [ast.literal_eval(vector[ind]) for vector in cluster]
                    
                    # sort all lists
                    lists_at_index = [sorted(sublist, key=lambda x: self._hyper_parameters["list_freq_dict"][ind].get(x, 0), reverse=True) for sublist
                                       in lists_at_index]
                    # pad data
                    
                    padded_data = [lst + ['missing_val'] * (self._hyper_parameters["avg_list_len"][ind] - len(lst)) for lst in lists_at_index]
                    # voting
                    
                    transposed_data = zip(*padded_data)
                    # Initialize the voting list
                    voting_list = []
                    # Iterate over each column
                    for column in transposed_data:
                        # Count occurrences of each value in the column
                        counts = Counter(column)
                        # Find the most common values
                        most_common_values = counts.most_common(2)
                    
                        # Check if the most common value is "missing_val"
                        if most_common_values[0][0] == "missing_val":
                            if len(most_common_values) >= 2:
                                most_common_value = most_common_values[1][0]
                            else:
                                most_common_value = "missing_val"
                        else:
                            most_common_value = most_common_values[0][0]
                    
                        # todo: most common value needs to be a real value, exclude missing vals - done!
                        # Append the most common value to the voting list
                        voting_list.append(most_common_value)
                    data = voting_list


                    ## for every method keep these lines
                    str_list = '[' + ', '.join(repr(item) for item in data) + ']'
                    frequent_value_list.append(str_list)

            centroid = np.array(frequent_value_list)

            return centroid
        else:
            raise Exception("bad seed")
