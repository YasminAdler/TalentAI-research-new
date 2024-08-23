import ast

import numpy as np
import math

################# change company_index to: with_gender_and_age = 11 / gender_no_age = 10 / age_no_gender = 10 / no_age_no_gender = 9 #################

def Statistic_intersection(u, v, type_values, parameters, company_index = 11):
    # print("started statistic")

    # print(parameters)
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
    # print("u is: ", u)
    # print("v is: ", v)

    # with_gender_and_age 
    columns_to_ignore = [0,1,2,3,4,5]
    # gender_no_age
    columns_to_ignore = [0,1,2,3]
    # age_no_gender 
    columns_to_ignore = [0,1,2,3,4]
    # no_age_no_gender
    columns_to_ignore = [0,1,2]
    
    
    for i in range(len(v)):
        if(i == company_index):
            continue
        if(i in columns_to_ignore):
            continue
                    
        # catrgorical handle
        try:
            if type_values[i] == "categoric":
                if (u[i] != "" and v[i]!=""):
                    # if attributes are same
                    if u[i] == v[i]:
                        results.append(0)
                    else:
                        specific_domain_size = parameters["domain sizes"][i]
                        f_v_ak = f_freq(specific_domain_size, theta1, betha, theta2, gamma)
                        
                        ## This part was changed by yasmin from:
                        # fr_u = parameters["frequencies"][str(i)][str((u[i]))] if u[i]!="" else 1
                        # fr_v = parameters["frequencies"][str(i)][str((v[i]))] if v[i]!="" else 1
                        
                        ## to this: 
                        fr_u = parameters["frequencies"][str(i)].get(str(u[i]), 1)
                        fr_v = parameters["frequencies"][str(i)].get(str(v[i]), 1)
                        
                        m_fk = parameters["minimum_freq_of_each_attribute"][str(i)]
                        d_fr = (abs(fr_u - fr_v) + m_fk) / max(fr_u, fr_v)
                        results.append(abs(max(d_fr, theta, f_v_ak)))
                        distance += pow(max(d_fr, theta, f_v_ak), 2)
        except Exception as e:
            print("error!!!!!", e)
            print("v is", v)
            print("i is", i)
            print("type values is", type_values, len(type_values))

            # # Numeric Handling - With Gender with Age
            # if type_values[i] == "numeric":
            #     try:
            #         if u[i] != '' and v[i] != '':
            #             if i == 4:
            #                 u_val = (float(u[i]) - 1913) / (1997 - 1913)
            #                 v_val = (float(v[i]) - 1913) / (1997 - 1913)
            #             if i == 19:
            #                 u_val = (float(u[i]) - 1666) / (2020 - 1666)
            #                 v_val = (float(v[i]) - 1666) / (2020 - 1666)
            #             if i == 34:
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
        
            # # Numeric Handling - with Gender No Age 
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
                            
                
            # # Numeric Handling - with Age No Gender 
            # if type_values[i] == "numeric":
            #     try:
            #         if u[i] != '' and v[i] != '':
            #             if i == 3:
            #                 u_val = (float(u[i]) - 1913) / (1997 - 1913)
            #                 v_val = (float(v[i]) - 1913) / (1997 - 1913)

            #             if i == 18:
            #                 u_val = (float(u[i]) - 1666) / (2020 - 1666)
            #                 v_val = (float(v[i]) - 1666) / (2020 - 1666)

            #             if i == 33:
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
                           

            # Numeric Handling - No Age No Gender
            if u[i] != '' and v[i] != '':
                try:
                    if i == 3:
                        u_val = (float(u[i]) - 1913) / (1997 - 1913)
                        v_val = (float(v[i]) - 1913) / (1997 - 1913)

                    if i == 16:
                        u_val = (float(u[i]) - 1666) / (2020 - 1666)
                        v_val = (float(v[i]) - 1666) / (2020 - 1666)

                    if i == 31:
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
            # create one hot vector
            u_list = ast.literal_eval(u[i])
            v_list = ast.literal_eval(v[i])

            ##### intersection
            one_hot_vec_u = [1 if word in u_list else 0 for word in parameters["one_hot_vector_prep"][i]]
            one_hot_vec_v = [1 if word in v_list else 0 for word in parameters["one_hot_vector_prep"][i]]


            # Calculate the intersection using element-wise AND
            intersection = [a & b for a, b in zip(one_hot_vec_u, one_hot_vec_v)]
            union = calculate_union(one_hot_vec_u, one_hot_vec_v)

            distance += 1 if sum(union) == 0 else 1 - sum(intersection) / sum(union)



    distance = math.sqrt(distance)
    # print("ended statistic")

    return distance, results
