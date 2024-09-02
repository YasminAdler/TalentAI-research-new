import ast
import numpy as np
import math
import pandas as pd

################# change company_index to: with_gender_and_age = 11 / gender_no_age = 10 / age_no_gender = 10 / no_age_no_gender = 9 #################
def Statistic_intersection(u, v, type_values, parameters):
    company_index=10
    columns_to_exclude=[0,1,2]
    
    
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



    for i in range(len(v)):
        if(i == company_index):
            continue
        if(i in columns_to_exclude): ## if in RoleToEmployee uncomment this
            continue
        
        # catrgorical handle
        if type_values[i] == "categoric":
            try:
                ################################# this was changed by yasmin #################################
                # if (u[i] and v[i]) or (u[i] != "" and v[i] != ""):
                if pd.notna(u[i]) and pd.notna(v[i]):
                    if u[i] == v[i]:
                        results.append(0)
                        ########### up here
                        
                ################################# Original handling: #################################
                # if (u[i] != "" and v[i]!=""):
                #     # if attributes are same
                #     if u[i] == v[i]:
                #         results.append(0)
                
                ########### up here
                    else:
                        specific_domain_size = parameters["domain sizes"][i]
                        f_v_ak = f_freq(specific_domain_size, theta1, betha, theta2, gamma)
                        
                        fr_u = parameters["frequencies"][str(i)][str((u[i]))] if u[i] == 'nan'  else 1
                        fr_v = parameters["frequencies"][str(i)][str((v[i]))] if v[i] == 'nan' else 1
                                                
                        m_fk = parameters["minimum_freq_of_each_attribute"][str(i)]
                        d_fr = (abs(fr_u - fr_v) + m_fk) / max(fr_u, fr_v)
                        results.append(abs(max(d_fr, theta, f_v_ak)))
                        distance += pow(max(d_fr, theta, f_v_ak), 2)
                    
            except Exception as e:
                print("error!!!!!", e)
                print("v is", v)
                print("i is", i)
                print("type values is", type_values, len(type_values))

        #  # Numeric Handling - with_gender_and_age
        # if type_values[i] == "numeric":
        #     try:
        #         if pd.notna(u[i]) and pd.notna(v[i]): ###
        #             # ### was changed to u[i] and v[i] and u[i] != "" and v[i] != "" 
        #             ## to do: change in all distance functions
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
        #             if math.isnan(val):
        #                 print(f"NaN found in numeric calculation at index {i}. u_val: {u_val}, v_val: {v_val}")
        #             distance += val
        #     except Exception as e:
        #         distance += 0
        #         # print("u[i]", u[i]) 
        #         # print("v[i]", v[i]) // This showed that a lot of values are actually nan, for v and u     
    
        # ## Numeric Handling - gender_no_age 
        # if type_values[i] == "numeric":
        #     try:
        #         if pd.notna(u[i]) and pd.notna(v[i]): ## to do delete or (u[i] != "" and v[i] != "") for alllll I THINK THIS IS BECAUSE THEYRE VALUE IS 'nan'

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
        #         distance +=0    
        #         # print("u[i]", u[i]) 
        #         # print("v[i]", v[i]) // This showed that a lot of values are actually nan, for v and u         
                
        # Numeric Handling - age_no_gender 
        if type_values[i] == "numeric":
            try:
                if pd.notna(u[i]) and pd.notna(v[i]):
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
                distance +=0    
                # print("u[i]", u[i]) 
                # print("v[i]", v[i]) // This showed that a lot of values are actually nan, for v and u     
                

        # # Numeric Handling - no_age_no_gender
        # if type_values[i] == "numeric":
        #     try:
        #         if pd.notna(u[i]) and pd.notna(v[i]):
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
        #         distance +=0    
        #         # print("u[i]", u[i]) 
        #         # print("v[i]", v[i]) // This showed that a lot of values are actually nan, for v and u     


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


    if math.isnan(distance):
        print("Final distance is NaN. Exiting function.")
        return float('inf'), results

    distance = math.sqrt(distance)
    if math.isnan(distance):
        print("Distance is NaN after sqrt, exiting function.")
        return float('inf'), results

    return distance, results