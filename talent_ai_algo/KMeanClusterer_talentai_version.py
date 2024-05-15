import ast
import sys

REPEATS_NUM = 5

import json
# import traceback
import numpy as np
import math
#sys.path.append("..")  # Add parent directory to path
from .. import utilss as ut
from collections import Counter
from sklearn.metrics import silhouette_score

MAX_ITERATION = 30

def average_of_lists(list_of_lists):
    return [sum(val for val in col if val != "missing_val") / sum(1 for val in col if val != "missing_val")
            if any(val != "missing_val" for val in col) else "missing_val"
            for col in zip(*list_of_lists)]


class KMeansClusterer_talentai:

    def __init__(
            self,
            num_means,  # k value
            distance,  # distance function
            repeats=REPEATS_NUM,
            mean_values=None,
            conv_test=1e-6,  # threshold for converging
            type_of_fields=None,
            repeats_method="best_wcss",
            hyper_params=dict()):
        self.repeats_method = repeats_method
        self._num_means = num_means
        self._distance = distance
        self._repeats = repeats
        self._mean_values = mean_values
        self._type_of_fields = type_of_fields
        self._means = None
        ##NOAM LINES FOR ANOMALIES
        self.clustersAverageDistance = None
        self.clustersStdDev = None
        self.clustersMaxDistances = None
        self.attributesAverageDistances = None
        self.attributesStdDevs = None

        self.silhouette = None
        self._max_difference = conv_test
        self._wcss = None
        self._normalized_wcss = None
        self._clusters_info = []
        self._model_json_info = 0
        self._hyper_parameters = hyper_params
        self._overall_mean = None
        self._overall_std = None
        self.min_dist = 0
        self.max_dist = 0
        self.average_dist_between_clusters = 0

    def createClusterJson(self):
        jsonData = {
            "wcss": self._wcss,
            "silhouette": self.silhouette,
        }
        listObj = []
        for i in range(len(self._means)):
            listObj.append(
                {
                    "cluster": i,
                    "mean": self._means[i].tolist(),
                    "averageDistance": self.clustersAverageDistance[i],
                    "maxDistance": self.clustersMaxDistances[i],
                    "stdDev": self.clustersStdDev[i],
                    "attributesAverageDistances": self.attributesAverageDistances[i],
                    "attributesStdDevs": self.attributesStdDevs[i],
                }
            )
        jsonData['clusters_info'] = listObj
        jsonData['cluster_values'] = self._clusters_info
        jsonData['hyperParams'] = self._hyper_parameters
        self._model_json_info = jsonData

    def getModelData(self):
        return self._model_json_info

    def store_model(self, filename):
        jsonData = self.getModelData()

        with open(filename, 'w') as json_file:
            json.dump(jsonData, json_file,
                      indent=4,
                      separators=(',', ': '))

    # calculates the mean of a cluster
    def _centroid(self, cluster, mean):
        # initialize an empty list, with size of number of features
        if len(cluster):

            frequent_value_list = []
            for ind in range(len(cluster[0])):

                if self._type_of_fields[ind] == "categoric":

                    values = [float(arr[ind]) for arr in cluster if arr[ind] != '']
                    result = np.mean(values) if len(values) > 0 else 0
                    frequent_value_list.append(np.mean(result))

                if self._type_of_fields[ind] == "numeric":
                    values = [float(arr[ind]) for arr in cluster if arr[ind] != '']

                    result = np.mean(values) if len(values)>0 else 0
                    frequent_value_list.append(np.mean(result))

                if self._type_of_fields[ind] == "list":
                    ## version for list frequency!!! comment this is using one hot vector
                    # avg_length = self._hyper_parameters["avg_list_len"][ind]
                    # lists_at_ind_index = [vector[ind] for vector in cluster]
                    # sorted_lists = [sorted(sublist, reverse=True) for sublist in lists_at_ind_index]
                    # # extend or shorten
                    # for lst in sorted_lists:
                    #     list_len = len(lst)
                    #     if list_len > avg_length:
                    #         lst[:] = lst[:avg_length]  # Shorten the list
                    #     elif list_len < avg_length:
                    #         lst.extend(["missing_val"] * (avg_length - list_len))  # Extend the list
                    #

                    # todo: missing values dont participate in determing the mean- done!
                   # most_common_values=average_of_lists(sorted_lists)
                    #version for one hot representation, comment out if using intersection/dot
                    #Extract lists from the indth index of each vector


                    lists_at_ind_index = [ast.literal_eval(vector[ind]) for vector in cluster]
                    # Flatten the lists and count occurrences of each value
                    flattened_list = [item for sublist in lists_at_ind_index for item in sublist]
                    value_counts = Counter(flattened_list)
                    # Find values that appear in at least 50% of the lists
                    threshold = len(cluster) / 2
                    most_common_values = [value for value, count in value_counts.items() if count >= threshold]
                    most_common_values = '[' + ', '.join(repr(item) for item in most_common_values) + ']'


                    frequent_value_list.append(most_common_values)

            centroid = frequent_value_list
            # exit()
            #            print("new centroid is:", centroid)
            return centroid
        else:
            # print(cluster)
            raise Exception("bad seed")

    def get_means(self):
        return self._means

    def get_wcss(self):
        self.wcssCalculate()
        print("wcss is:", self._wcss)
        # print("wcss unnormalized is:", self._wcss)
        #
        # normalize_wcss = (self._wcss - self.min_dist) / (self.max_dist - self.min_dist)
        # print("wcss normalized is:", normalize_wcss)

        return self._wcss

    def calc_distance_between_clusters(self):
        distance = 0
        num_pairs = 0
        # print(self._means)
        max_val=0
        min_val=999999999999
        dists=[]
        for i in range(len(self._means)):
            for j in range(i + 1, len(self._means)):
                d = self._distance(self._means[i], self._means[j], self._type_of_fields,
                                   self._hyper_parameters)
                max_val=max(d[0], max_val)
                min_val=min(min_val,d[0])
                #distance += d[0]
                dists.append(d[0])
                num_pairs += 1

        normalized_dists = [(x - min_val) / (max_val - min_val) for x in dists]
        average_normalized = sum(normalized_dists) / len(normalized_dists)

        self.average_dist_between_clusters = average_normalized
        print("average distance between clusters is: ", average_normalized)

    def calc_min_max_dist(self, vecs):
        print("calc min and max distances for normalization")
        self.min_dist = self._distance(vecs[0], vecs[1], self._type_of_fields, self._hyper_parameters)[0]
        print(self.min_dist)
        self.max_dist = self.min_dist
        print(self.max_dist)
        for u in range(len(vecs)):
            for v in range(u + 1, len(vecs)):
                dist = self._distance(vecs[v], vecs[u], self._type_of_fields, self._hyper_parameters)[0]
                self.min_dist = min(dist, self.min_dist)
                self.max_dist = max(dist, self.max_dist)

        print("min distance is", self.min_dist)
        print("max distance is", self.max_dist)
    def wcssCalculate(self):
        # for i in
        wcss = 0

        # find all distances:
        min_val=9999999999999999999999999
        max_val=0
        for i in range(len(self._clusters_info)):
            for vec in self._clusters_info[i]:
                dist, res= self._distance(list(vec), self._means[i], self._type_of_fields,
                                                   self._hyper_parameters)
                max_val=max(dist,max_val)
                min_val=min(dist,min_val)


        for i in range(len(self._clusters_info)):
            for vec in self._clusters_info[i]:
                distance, results = self._distance(list(vec), self._means[i], self._type_of_fields,
                                                   self._hyper_parameters)
                wcss += ((distance-min_val)/(max_val-min_val)) ** 2
        self._wcss = wcss

    def get_Silhouette(self):
        if self.silhouette is None:
            self.SilhouetteCalculate()
        return self.silhouette

    def SilhouetteCalculate(self):
        # list to hold all vectors
        concatenated_list = []
        cluster_labels = []
        # build the cluster labels
        for index in range(len(self._clusters_info)):
            concatenated_list.extend(self._clusters_info[index])
            cluster_labels.extend([index] * len(self._clusters_info[index]))

        score = silhouette_score(concatenated_list, cluster_labels,
                                 metric=lambda x, y: self._distance(x, y, self._type_of_fields, self._hyper_parameters)[
                                     0])
        self.silhouette = score

    def metaDataCalculation(self):
        numberOfFeatures = len(self._means[0])
        numberOfClusters = len(self._means)

        self.clustersAverageDistance = []
        self.clustersStdDev = []
        self.attributesStdDevs = [[] for _ in range(numberOfClusters)]
        self.attributesAverageDistances = [[] for _ in range(numberOfClusters)]
        self.clustersMaxDistances = []

        # calculate average
        for index in range(numberOfClusters):
            maxDistance = 0
            sumOfTotalDistance = 0
            sumOfAttributesDistances = [0 for _ in range(numberOfFeatures)]
            self.attributesAverageDistances[index] = [0 for _ in range(numberOfFeatures)]
            for vec in self._clusters_info[index]:
                ##check distance between vec in cluster with the cluster mean
                distance, results = self._distance(vec, self._means[index], self._type_of_fields,
                                                   self._hyper_parameters)
                ##check for max distance
                if distance > maxDistance:
                    maxDistance = distance
                ##sum total distance for average calculate
                sumOfTotalDistance += distance
                ##sum each distances for average calculate
                for i in range(numberOfFeatures):
                    sumOfAttributesDistances[i] += abs(results[i])
            self.clustersMaxDistances.append(maxDistance)
            self.clustersAverageDistance.append(sumOfTotalDistance / len(self._clusters_info[index]))
            for i in range(numberOfFeatures):
                self.attributesAverageDistances[index][i] = sumOfAttributesDistances[i] / len(
                    self._clusters_info[index])

        # calculate standard deviation
        for index in range(numberOfClusters):
            sumOfSquareDistances = 0
            squareDeltaDistances = [0 for _ in range(numberOfFeatures)]
            for vec in self._clusters_info[index]:
                distance, results = self._distance(vec, self._means[index], self._type_of_fields,
                                                   self._hyper_parameters)
                for i in range(numberOfFeatures):
                    squareDeltaDistances[i] += (results[i] - self.attributesAverageDistances[index][i]) ** 2
                sumOfSquareDistances += (distance - self.clustersAverageDistance[index]) ** 2
            ##deal with clusters with only one data sample
            if len(self._clusters_info[index]) < 2:
                self.clustersStdDev.append(0)
                for i in range(numberOfFeatures):
                    self.attributesStdDevs[index].append(0)
            else:
                self.clustersStdDev.append(math.sqrt(sumOfSquareDistances / (len(self._clusters_info[index]) - 1)))
                for i in range(numberOfFeatures):
                    self.attributesStdDevs[index].append(
                        math.sqrt(squareDeltaDistances[i] / (len(self._clusters_info[index]) - 1)))

    def _sum_distances(self, vectors1, vectors2):
        difference = 0.0

        for u, v in zip(vectors1, vectors2):
            distance, results = self._distance(u, v, self._type_of_fields, self._hyper_parameters)

            difference += distance
        return difference

    # cluster the data given to kmeans
    def cluster_vectorspace(self, vectors):
        meanss = []
        wcsss = []
        best_clusters = []
        # make _repeats repeats to get the best means
        i = 0
        while i < self._repeats:
            # for trial in range(self._repeats):
            #   print("kmeans cluster_vectorspace, doing repeats", trial)
            # generate new means
            try:
                self._means = ut.mean_generator(self._num_means, vectors)
                # cluster the vectors to the given means
                try:
                    self._cluster_vectorspace(vectors)
                    i += 1
                    print("succeed once", i, "out of", self._repeats)
                except Exception as e:
                    print("error occured",sys.exc_info()[-1].tb_lineno, ":", e)
                    # print("hello")
                    print("problem generating, trying again")
                    #  exit() #nooo
                    self._means = ut.mean_generator(self._num_means, vectors)
                    continue
                # add the new means each time
                meanss.append(self._means)
                self.wcssCalculate()
                # if (min())
                wcsss.append(self._wcss)
                if min(wcsss) == self._wcss:
                    best_clusters = self._clusters_info
                # if ()
            except Exception as e:
                # print(e, ": ", trial)
                raise e
        # at this point meanss holds an array of arrays, each array has k means in it.
        if len(meanss) > 1:
            if self.repeats_method == "best_wcss":
                lowest_wcss = wcsss.index(min(wcsss, key=lambda x: x))
                self._wcss = wcsss[lowest_wcss]
                self._means = meanss[lowest_wcss]
                self._clusters_info = best_clusters

            elif self.repeats_method == "minimal_difference":
                # find the set of means that's minimally different from the others
                min_difference = min_means = None
                for i in range(len(meanss)):
                    d = 0
                    for j in range(len(meanss)):
                        if i != j:
                            d += self._sum_distances(meanss[i], meanss[j])
                    if min_difference is None or d < min_difference:
                        min_difference, min_means = d, meanss[i]

                    # use the best means
                    self._means = min_means

    # cluster for specific mean values
    def _cluster_vectorspace(self, vectors):
        # print("in cluster vectorspace")
        if self._num_means < len(vectors):
            # max iteration if there is no conversion
            current_iteration = 0
            # perform k-means clustering
            converged = False
            while not converged:
                current_iteration += 1
                # assign the tokens to clusters based on minimum distance to
                # the cluster means

                clusters = [[] for m in range(self._num_means)]

                for vector in vectors:
                    index, distances = self.classify_vectorspace(vector)
                    clusters[index].append(vector) #clusters[index].append(vector.tolist())

                try:

                    new_means = list(map(self._centroid, clusters, self._means))

                # print("new means:", new_means)
                except Exception as e:
                    # Propagate the exception from function c to function a
                    print("An error occurred on line", sys.exc_info()[-1].tb_lineno, ":", e)
                    raise e
                # print("new means are:", new_means)
                # recalculate cluster means by computing the centroid of each cluster
                ###### new_means = list(map(self._centroid, clusters, self._means))

                # measure the degree of change from the previous step for convergence
                difference = self._sum_distances(self._means, new_means)
                # remember the new means
                self._means = new_means
                if difference < self._max_difference or current_iteration == MAX_ITERATION:
                    converged = True

            self._clusters_info = clusters
            # self.createClusterJson()
            # print ('cluster means: ', self._means)
        else:
            print("erorr!!!!")
            pass  # todo: return error here

    def classify_vectorspace(self, vector):
        # finds the closest cluster centroid
        # returns that cluster's index
        best_distance = best_index = None
        distances = []
        for index in range(len(self._means)):
            mean = self._means[index]
            distance, results = self._distance(vector, mean, self._type_of_fields, self._hyper_parameters)
            cluster_info = {
                "cluster": index,
                "distance": distance
            }
            distances.append(cluster_info)
            if best_distance is None or distance < best_distance:
                best_index, best_distance = index, distance
        return best_index, distances
