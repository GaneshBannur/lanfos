import torch
import numpy as np

class QueryModule:
    def __init__(self, image_text_model, clustering_result_filepaths, data_device="cuda"):
        self.image_text_model = image_text_model
        cluster_centers = {}
        for feature_level in clustering_result_filepaths.keys():
            cluster_centers[feature_level] = torch.from_numpy(np.load(clustering_result_filepaths[feature_level])).to(data_device)
            cluster_centers[feature_level] = cluster_centers[feature_level].to(torch.float)

        self.cluster_centers = cluster_centers

    @torch.no_grad()
    def get_cluster_relevancies(self, text_query):
        cluster_relevancies_per_level = {}
        for feature_level, level_cluster_centers in self.cluster_centers.items():
            with torch.amp.autocast("cuda"):
                # text_feature has shape [1, D]
                text_feature = self.image_text_model.encode_text([text_query])
                cluster_relevancies = self.image_text_model.get_relevancy(level_cluster_centers, text_feature)
                # text_feature needed to be 2D for get_relevancy but now remove unused dim
                cluster_relevancies = cluster_relevancies.squeeze()
                cluster_relevancies_per_level[feature_level] = cluster_relevancies

        return cluster_relevancies_per_level
    
    @torch.no_grad()
    def get_relevancy(self, text_query, feature_distribution):
        cluster_relevancies_per_level = self.get_cluster_relevancies(text_query)
        relevancies = []
        for feature_level, dist_for_level in feature_distribution.items():
            level_relevancies = cluster_relevancies_per_level[feature_level]
            relevancy_for_level = torch.matmul(dist_for_level, level_relevancies)
            relevancies.append(relevancy_for_level)

        relevancies = torch.stack(relevancies, dim=0)
        relevancies = relevancies.max(dim=0).values
        return relevancies
