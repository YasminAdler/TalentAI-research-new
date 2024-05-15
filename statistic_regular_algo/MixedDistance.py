import ast
import math
import re


def custom_sort(obj, frequencies):
    # If the object is "missing_val," return a tuple with a large frequency value
    if obj == "missing_val":
        return (float('inf'),)
    # Otherwise, return a tuple with the negative frequency value
    return (-frequencies.get(obj, 0),)


def MixedDistance(u, v, type_values, parameters):

    distance = 0
    results = []
    for i in range(len(u)):
        # if type is categorical
        if type_values[i] == "categoric":
            if v[i] != u[i]:  # and v[i] != "" and u[i] != "":
                distance += 1
            # results.append(1)
        # if type is numeric
        if type_values[i] == "numeric":
            if u[i] != '' and v[i] != '':
                # normalization for wine
                # u_val = (float(u[i]) - 4) / (48 - 4)
                # v_val = (float(v[i]) - 4) / (48 - 4)

                #normalization for hr
                if (i == 4):
                    u_val = (float(u[i]) - 1913) / (1997 - 1913)
                    v_val = (float(v[i]) - 1913) / (1997 - 1913)
                if (i == 19):
                    u_val = (float(u[i]) - 1666) / (2020 - 1666)
                    v_val=(float(v[i]) - 1666) / (2020 - 1666)

                if (i == 34):
                    v_val=(float(v[i]) - 3.11) / (5 - 3.11)
                    u_val=(float(u[i]) - 3.11) / (5 - 3.11)

                val = (u_val - v_val) ** 2
            # val = 0#abs(float(u[i]) - float(v[i])) **2
            else:
                val = 0

            distance += val

        if type_values[i] == "list":
            #print("errors")
            #print(u[i])
            #print(v[i])
            u_list = ast.literal_eval(u[i])
            v_list = ast.literal_eval(v[i])
            #print("done error")
            # second hamming is sorted by freq, comment the two following lines if using talentai method
            u_list = sorted(u_list, key=lambda x: parameters["list_freq_dict"][i][x], reverse=True)
            v_list = sorted(v_list, key=lambda x: parameters["list_freq_dict"][i][x], reverse=True)


            #  adapt according to average list length

            if (len(u_list) < parameters["avg_list_len"][i]):
                u_list.extend(["missing_val"] * (parameters["avg_list_len"][i] - len(u_list)))
            if len(u_list) > parameters["avg_list_len"][i]:
                u_list = u_list[:parameters["avg_list_len"][i]]
            if len(v_list) < parameters["avg_list_len"][i]:
                v_list.extend(["missing_val"] * (parameters["avg_list_len"][i] - len(v_list)))
            if len(v_list) > parameters["avg_list_len"][i]:
                v_list = v_list[:parameters["avg_list_len"][i]]

            # print("unlist", u_list)
            # print("vnlist", v_list)
            # set1 = set(v_list)
            # set2 = set(u_list)
            #
            # intersection_size = len(set1.intersection(set2))
            # distance += len(u_list) - intersection_size

            for j in range(len(v_list)):
                if v_list[j] != u_list[j]:
                    distance += 1


    #     print(distance)
    return math.sqrt(distance), results
