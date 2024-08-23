import ast

import numpy as np
import math


def Statistic_dot_product(u, v, type_values, parameters):
    distance = 0
    results = []

    def f_freq(z, theta1, betha, theta2, gamma):
        if z <= theta1:
            return 1
        if theta1 < z <= theta2:
            return 1 - betha * (z - theta1)
        if z > theta2:
            return 1 - betha * (theta2 - theta1) - gamma * (z - theta2)

    def calculate_union(one_hot_vector1, one_hot_vector2):
        if len(one_hot_vector1) != len(one_hot_vector2):
            raise ValueError("Input vectors must have the same length")

        union_result = [0] * len(one_hot_vector1)
        for i in range(len(one_hot_vector1)):
            union_result[i] = one_hot_vector1[i] or one_hot_vector2[i]

        return union_result

    betha = parameters["betha"]
    theta1 = parameters["theta1"]
    theta2 = parameters["theta2"]
    theta = parameters["theta"]
    gamma = parameters["gamma"]
    
    
    ## uncomment for with age with gender
    
def Statistic_dot_product(u, v, type_values, parameters):
    distance = 0
    results = []

    def f_freq(z, theta1, betha, theta2, gamma):
        if z <= theta1:
            return 1
        if theta1 < z <= theta2:
            return 1 - betha * (z - theta1)
        if z > theta2:
            return 1 - betha * (theta2 - theta1) - gamma * (z - theta2)

    def calculate_union(one_hot_vector1, one_hot_vector2):
        if len(one_hot_vector1) != len(one_hot_vector2):
            raise ValueError("Input vectors must have the same length")

        union_result = [0] * len(one_hot_vector1)
        for i in range(len(one_hot_vector1)):
            union_result[i] = one_hot_vector1[i] or one_hot_vector2[i]

        return union_result

    betha = parameters["betha"]
    theta1 = parameters["theta1"]
    theta2 = parameters["theta2"]
    theta = parameters["theta"]
    gamma = parameters["gamma"]

    for i in range(len(v)):
    # catrgorical handle
        try:
            if type_values[i] == "categoric":
                if u[i] is not None and v[i] is not None and u[i] != "" and v[i] != "":
                    # if attributes are same
                    if u[i] == v[i]:
                        results.append(0)
                    # attributes are not the same - calculate max{f(|vak|), dfr(vi, ui), theta)
                    else:
                        specific_domain_size = parameters["domain sizes"][i]
                        f_v_ak = f_freq(specific_domain_size, theta1, betha, theta2, gamma)
                        fr_u = parameters["frequencies"][str(i)][str((u[i]))] if u[i] != "" else 1
                        fr_v = parameters["frequencies"][str(i)][str((v[i]))] if v[i] != "" else 1
                        m_fk = parameters["minimum_freq_of_each_attribute"][str(i)]
                        d_fr = (abs(fr_u - fr_v) + m_fk) / max(fr_u, fr_v)
                        results.append(abs(max(d_fr, theta, f_v_ak)))
                        distance += pow(max(d_fr, theta, f_v_ak), 2)
        except Exception as e:
            print("error!!!!!", e)
            print("fr_u",fr_u)
            print("fr_v",fr_v)
            print("dfr" ,d_fr)
            print("vi is", v[i])
            print("ui is", u[i])
            print("i is", i)
            print("type values is", type_values, len(type_values))
            exit()
            

    # # Numeric Handling - With Gender With Age
    #     if type_values[i] == "numeric":
    #         try:
    #             if u[i] != '' and v[i] != '':
    #                 # normalization for wine
    #                 # u_val = (float(u[i]) - 4) / (48 - 4)
    #                 # v_val = (float(v[i]) - 4) / (48 - 4)

    #                 # normalization for hr
    #                 if i == 4:
    #                     u_val = (float(u[i]) - 1913) / (1997 - 1913)
    #                     v_val = (float(v[i]) - 1913) / (1997 - 1913)
                    
    #                 if i == 19:
    #                     u_val = (float(u[i]) - 1666) / (2020 - 1666)
    #                     v_val = (float(v[i]) - 1666) / (2020 - 1666)
                    
    #                 if i == 34:
    #                     v_val = (float(v[i]) - 3.11) / (5 - 3.11)
    #                     u_val = (float(u[i]) - 3.11) / (5 - 3.11)

    #                 val = (u_val - v_val) ** 2
    #                 distance += val

    #                 # results.append(abs(np.float64(u[i]) - np.float64(v[i])))
    #                 # distance += pow(np.float64(u[i]) - np.float64(v[i]), 2)

    #         except Exception as e:
    #             print(e)
    #             print(u[i])
    #             print(i)
    #             print(v[i])
    #             exit()
            
    # Numeric Handling - No Gender with Age
    if type_values[i] == "numeric":
        try:
            
            if u[i] != '' and v[i] != '':
                
                if i == 3:
                    u_val = (float(u[i]) - 1913) / (1997 - 1913)
                    v_val = (float(v[i]) - 1913) / (1997 - 1913)

                if i == 18:
                    u_val = (float(u[i]) - 1666) / (2020 - 1666)
                    v_val = (float(v[i]) - 1666) / (2020 - 1666)

                if i == 33:
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

    # # Numeric Handling - No Age with Gender
    # if type_values[i] == "numeric":
    #     try:
    #         if u[i] != '' and v[i] != '':

    #             if i == 4:
    #                 u_val = (float(u[i]) - 1913) / (1997 - 1913)
    #                 v_val = (float(v[i]) - 1913) / (1997 - 1913)

    #             if i == 17:
    #                 u_val = (float(u[i]) - 1666) / (2020 - 1666)
    #                 v_val = (float(v[i]) - 1666) / (2020 - 1666)

    #             if i == 32:
    #                 v_val = (float(v[i]) - 3.11) / (5 - 3.11)
    #                 u_val = (float(u[i]) - 3.11) / (5 - 3.11)
                    
    #                 val = (u_val - v_val) ** 2
    #                 distance += val 

                
    #     except Exception as e:
    #         print(e)
    #         print(u[i])
    #         print(i)
    #         print(v[i])
    #         exit()

    # # Numeric Handling - No Age No Gender
    #     try:
    #         if u[i] != '' and v[i] != '':

    #             if i == 3:
    #                 u_val = (float(u[i]) - 1913) / (1997 - 1913)
    #                 v_val = (float(v[i]) - 1913) / (1997 - 1913)

    #             if i == 16:
    #                 u_val = (float(u[i]) - 1666) / (2020 - 1666)
    #                 v_val = (float(v[i]) - 1666) / (2020 - 1666)

    #             if i == 31:
    #                 v_val = (float(v[i]) - 3.11) / (5 - 3.11)
    #                 u_val = (float(u[i]) - 3.11) / (5 - 3.11)    
                    
    #             val = (u_val - v_val) ** 2
    #             distance += val 

                
    #     except Exception as e:
    #         print(e)
    #         print(u[i])
    #         print(i)
    #         print(v[i])
    #         exit()
    
        
    if type_values[i] == "list":
        # create one hot vector

        u_list = ast.literal_eval(u[i])
        v_list = ast.literal_eval(v[i])

        ##### dot product

        one_hot_vec_u = [1 if word in u_list else 0 for word in parameters["one_hot_vector_prep"][i]]
        one_hot_vec_v = [1 if word in v_list else 0 for word in parameters["one_hot_vector_prep"][i]]

        # Calculate the dot product
        dot_product = sum(a * b for a, b in zip(one_hot_vec_u, one_hot_vec_v))
        # Scale the dot product to always be less than 1
        distance += 1 - (dot_product / len(one_hot_vec_v))  # scaled dot product

        distance = math.sqrt(distance)

    return distance, results
