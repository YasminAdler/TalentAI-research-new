
import sys
import os as os
import openpyxl as openpyxl

script_dir = os.path.dirname(os.path.abspath(__file__))  # Use os.path.dirname(os.path.abspath(__file__)) to get the current script directory
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append("../TalentAI-research-new-last-update")
sys.path.append(
    "C:/Users/adler/OneDrive/Talent.AI/TalentAI-research-new-last-update/.venv/Lib/site-packages"
)
from customCentroids import *
# from recommendationAlgo import *



class Subcluster:
    def __init__(self, company, data, calculator):
        self.company = company
        self.data = data
        self.centroid = calculator.calculate_centroid(data)


class Cluster:
    def __init__(self, cluster_id, mean, average_distance, max_distance, std_dev, attributes_average_distances, attributes_std_devs, data, calculator, _distance):
        self.cluster_id = cluster_id
        self.mean = mean
        self.average_distance = average_distance
        self.max_distance = max_distance
        self.std_dev = std_dev
        self.attributes_average_distances = attributes_average_distances
        self.attributes_std_devs = attributes_std_devs
        self.calculator = calculator
        self.data = data
        self.subclusters = self._create_subclusters()
        self._distance = _distance

        
    def to_dict(self):
        return {
            "cluster_id": self.cluster_id,
            "mean": self.mean,
            "average_distance": self.average_distance,
            "max_distance": self.max_distance,
            "std_dev": self.std_dev,
            "attributes_average_distances": self.attributes_average_distances,
            "attributes_std_devs": self.attributes_std_devs
        }
        
################# BEFORE THE MODEL GENERATION change company_index to: with_gender_and_age = 11 / gender_no_age = 10 / age_no_gender = 10 / no_age_no_gender = 9 #################
    
    def _create_subclusters(self, company_index = 11):
        subclusters = {}
        company_subclusters = {}
        # company_subclusters_by_cluster = []
        for cluster_index, cluster in enumerate(self.data):
            for row in cluster:
                if len(row) > company_index:  
                    company = row[company_index]  
                    if isinstance(company, list):
                        company = company[0] 
                    if company not in company_subclusters:
                        company_subclusters[company] = []
                    company_subclusters[company].append(row)
                    
            for company, subcluster_data in company_subclusters.items():
                subclusters[company] = Subcluster(company, subcluster_data, self.calculator)
                # print("subcluster_data: ",subcluster_data)
        
        return subclusters

