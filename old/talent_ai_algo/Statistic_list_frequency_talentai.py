import ast

import numpy as np
import math


def Statistic_list_frequency(u, v, type_values, parameters):
    # print("started statistic")
    # print(parameters)
    distance = 0
    results = []

    def custom_sort(item):
        if item == 'missing_val':
            return float('inf')  # Place 'missing_val' at the end
        else:
            return item


    def f_freq(z, theta1, betha, theta2, gamma):
        if z <= theta1:
            return 1
        if theta1 < z <= theta2:
            return 1 - betha * (z - theta1)
        if z > theta2:
            return 1 - betha * (theta2 - theta1) - gamma * (z - theta2)
    betha = parameters["betha"]
    theta1 = parameters["theta1"]
    theta2 = parameters["theta2"]
    theta = parameters["theta"]
    gamma = parameters["gamma"]
    # print("u is: ", u)
    # print("v is: ", v)

    for i in range(len(v)):

        # catrgorical handle
        try:
            if type_values[i] == "categoric":
                if (u[i] != "" and v[i]!=""):
                    # if attributes are same
                    if float(u[i]) == float(v[i]):
                        results.append(0)
                    # attributes are not the same - calculate max{f(|vak|), dfr(vi, ui), theta)
                    else:
                        specific_domain_size = parameters["domain sizes"][i]
                        f_v_ak = f_freq(specific_domain_size, theta1, betha, theta2, gamma)
                        fr_u = float(u[i])#parameters["frequencies"][str(i)][str((u[i]))] if u[i] != "" else 1
                        fr_v = float(v[i])#parameters["frequencies"][str(i)][str((v[i]))] if v[i] != "" else 1
                        m_fk = parameters["minimum_freq_of_each_attribute"][str(i)]
                        d_fr = (abs(fr_u - fr_v) + m_fk) / max(fr_u, fr_v)
                        results.append(abs(max(d_fr, theta, f_v_ak)))
                        distance += pow(max(d_fr, theta, f_v_ak), 2)
        except Exception as e:
            print("error!!!!!", e)
            print("v is", v)
            print("i is", i)
            print("type values is", type_values, len(type_values))

        # numberic handle
        if type_values[i] == "numeric":

            try:
                if u[i] != '' and v[i] != '':
                    # lines for wine
                    # u_val = (float(u[i]) - 4) / (48 - 4)
                    # v_val = (float(v[i]) - 4) / (48 - 4)

                    # lines for hr
                    if i == 4:
                        u_val = (float(u[i]) - 1913) / (1997 - 1913)
                        v_val = (float(v[i]) - 1913) / (1997 - 1913)

                    if i == 19:
                        u_val = (float(u[i]) - 1666) / (2020 - 1666)
                        v_val = (float(v[i]) - 1666) / (2020 - 1666)

                    if i == 34:
                        v_val = (float(v[i]) - 3.11) / (5 - 3.11)
                        u_val = (float(u[i]) - 3.11) / (5 - 3.11)

                    val = (u_val - v_val) ** 2
                    distance += val

            except Exception as e:
                print(e)
                print(u[i])
                print(i)
                print(v[i])
                exit()

        if type_values[i] == "list":
            # sort according to frequency descending order
            u_list = sorted(u[i], key=custom_sort, reverse=True) #sorted(u_list, key=lambda x: parameters["list_freq_dict"][i][x], reverse=True)
            v_list = sorted(v[i], key=custom_sort, reverse=True) #sorted(v_list, key=lambda x: parameters["list_freq_dict"][i][x], reverse=True)


            #  adapt according to average list length
            if len(u_list) < parameters["avg_list_len"][i]:
                u_list.extend(["missing_val"] * (parameters["avg_list_len"][i] - len(u_list)))
            if len(u_list) > parameters["avg_list_len"][i]:
                u_list = u_list[:parameters["avg_list_len"][i]]
            if len(v_list) < parameters["avg_list_len"][i]:
                v_list.extend(["missing_val"] * (parameters["avg_list_len"][i] - len(v_list)))
            if len(v_list) > parameters["avg_list_len"][i]:
                v_list = v_list[:parameters["avg_list_len"][i]]

            specific_domain_size = len(parameters["one_hot_vector_prep"][i])
            try:
                for j in range(len(u_list)):
                    if u_list[j]!="missing_val" and v_list[j]!="missing_val":
                        f_v_ak = f_freq(specific_domain_size, theta1, betha, theta2, gamma)
                        fr_u = float(u_list[j]) #parameters["list_freq_dict"][i][u_list[j]] if u_list[j] != "missing_val" else 1
                        fr_v = float(v_list[j])#parameters["list_freq_dict"][i][v_list[j]] if v_list[j] != "missing_val" else 1
                        m_fk = min(parameters["list_freq_dict"][i].values())
                        d_fr = (abs(fr_u - fr_v) + m_fk) / max(fr_u, fr_v)
                        results.append(abs(max(d_fr, theta, f_v_ak)))
                        distance += pow(max(d_fr, theta, f_v_ak), 2)
            except Exception as e:
                print(e)
                print("bad!!!", j, i)

                print(fr_u)
                print(fr_v)
                print(type(fr_u))
                print(type(fr_v))
                print(d_fr)
                print(f_v_ak)
                print(parameters["list_freq_dict"][i])
                exit()

    distance = math.sqrt(distance)
    return distance, results
