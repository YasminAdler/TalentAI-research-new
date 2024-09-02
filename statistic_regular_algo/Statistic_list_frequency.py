import ast
import numpy as np
import math
import pandas as pd

################# change company_index to: with_gender_and_age = 11 / gender_no_age = 10 / age_no_gender = 10 / no_age_no_gender = 9 #################
def Statistic_list_frequency(u, v, type_values, parameters):
    company_index = 9

    columns_to_exclude = [0, 1, 2]
    
    distance = 0
    results = []

    attributes_shared_by_both = []

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

    for i in range(len(v)):
        if(i == company_index):
            continue
        if(i in columns_to_exclude): ## if in RoleToEmployee uncomment this
            continue
        
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
                        
                        fr_u = parameters["frequencies"][str(i)].get(str(u[i]), 1)
                        fr_v = parameters["frequencies"][str(i)].get(str(v[i]), 1)
                        
                        if fr_u == 0 or fr_v == 0:
                            print(f"Warning: Zero frequency found at index {i}. fr_u: {fr_u}, fr_v: {fr_v}")

                        m_fk = parameters["minimum_freq_of_each_attribute"][str(i)]
                        d_fr = (abs(fr_u - fr_v) + m_fk) / max(fr_u, fr_v, 1e-9)  # Avoid division by zero
                        term = max(d_fr, theta, f_v_ak)
                        if math.isnan(term):
                            print(f"NaN found in categoric calculation at index {i}. d_fr: {d_fr}, theta: {theta}, f_v_ak: {f_v_ak}")
                        results.append(abs(term))
                        distance += pow(term, 2)
                    
            except Exception as e:
                print("error!!!!!", e)
                print("u[i]", u[i],  "v[i]", v[i])

                print("v is", v)
                print("u is", u)
                print("i is", i)
                print("type values are", type_values, len(type_values))
                        
        # # Numeric Handling - with_gender_and_age
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
                
        # # Numeric Handling - age_no_gender 
        # if type_values[i] == "numeric":
        #     try:
        #         if pd.notna(u[i]) and pd.notna(v[i]):
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
        #         distance +=0    
        #         # print("u[i]", u[i]) 
        #         # print("v[i]", v[i]) // This showed that a lot of values are actually nan, for v and u     
                

        # Numeric Handling - no_age_no_gender
        if type_values[i] == "numeric":
            try:
                if pd.notna(u[i]) and pd.notna(v[i]):
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
                distance +=0    
                # print("u[i]", u[i]) 
                # print("v[i]", v[i]) // This showed that a lot of values are actually nan, for v and u     

        if type_values[i] == "list":
            ####list frequency
            ## normalize? ask the team
            u_list = ast.literal_eval(u[i])
            v_list = ast.literal_eval(v[i])

            u_list = sorted(u_list, key=lambda x: parameters["list_freq_dict"][i].get(x, 0), reverse=True)
            v_list = sorted(v_list, key=lambda x: parameters["list_freq_dict"][i].get(x, 0), reverse=True)


            #  adapt according to average list length
            if (len(u_list) < parameters["avg_list_len"][i]):
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
                    if u_list[j] != "missing_val" and v_list[j] != "missing_val":
                        
                        f_v_ak = f_freq(specific_domain_size, theta1, betha, theta2, gamma)
                        fr_u = parameters["list_freq_dict"][i].get(u_list[j], 1)
                        fr_v = parameters["list_freq_dict"][i].get(v_list[j], 1)
                        
                        if fr_u == 0 or fr_v == 0:
                            print(f"Warning: Zero frequency in list at index {i}, position {j}. fr_u: {fr_u}, fr_v: {fr_v}")

                        m_fk = min(parameters["list_freq_dict"][i].values())
                        d_fr = (abs(fr_u - fr_v) + m_fk) / max(fr_u, fr_v, 1e-9)  # Avoid division by zero
                        term = max(d_fr, theta, f_v_ak)
                        if math.isnan(term):
                            print(f"NaN found in list calculation at index {i}, position {j}. d_fr: {d_fr}, theta: {theta}, f_v_ak: {f_v_ak}")
                        results.append(abs(term))
                        distance += pow(term, 2)
                        
            except Exception as e:
                print(e)
                print("error!!!")
                print(u_list)
                print(v_list)
                print(fr_u)
                print(fr_v)
                print(j)
                print(parameters["list_freq_dict"][i])
                print(i)
                exit()

    if math.isnan(distance):
        print("Final distance is NaN. Exiting function.")
        return float('inf'), results

    distance = math.sqrt(distance)
    if math.isnan(distance):
        print("Distance is NaN after sqrt, exiting function.")
        return float('inf'), results

    return distance, results
