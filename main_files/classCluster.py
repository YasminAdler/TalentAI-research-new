
import sys
import os as os
import openpyxl as openpyxl
import logging

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
    def __init__(self, cluster_id, mean, average_distance, max_distance, std_dev, attributes_average_distances, attributes_std_devs, data, calculator, _distance, company_index):
        self.cluster_id = cluster_id
        self.mean = mean
        self.average_distance = average_distance
        self.max_distance = max_distance
        self.std_dev = std_dev
        self.attributes_average_distances = attributes_average_distances
        self.attributes_std_devs = attributes_std_devs
        self.calculator = calculator
        self.data = data
        self._distance = _distance
        self.subclusters = self._create_subclusters(company_index)

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

    def _create_subclusters(self, company_index):
        subclusters = {}
        try:
            for cluster_index, cluster in enumerate(self.data):
                # Reset company_subclusters for each cluster to avoid mixing data between clusters
                company_subclusters = {}
                
                for row in cluster:
                    # Check if the row has enough elements to include the company index
                    if len(row) > company_index:
                        company = row[company_index]

                        # If company is a list, extract the first element; otherwise, use as is
                        if isinstance(company, list):
                            company = company[0]
                            
                        if company:  # Ensure the company value is not empty
                            if company not in company_subclusters:
                                company_subclusters[company] = []
                            company_subclusters[company].append(row)
                        else:
                            print(f"Empty company value found in row: {row}")

                # Create subclusters for each company in the current cluster
                for company, subcluster_data in company_subclusters.items():
                    if subcluster_data:
                        subclusters[company] = Subcluster(company, subcluster_data, self.calculator)
                    else:
                        print(f"No data found for company: {company} in cluster {cluster_index}")

            # Check if subclusters were created correctly
            if not subclusters:
                print("Warning: No subclusters created. Ensure the data is correctly formatted and company index is correct.")
        
        except Exception as e:
            logging.error(f"Error creating subclusters: {e}")
            print(f"Error creating subclusters: {e}")
        return subclusters
