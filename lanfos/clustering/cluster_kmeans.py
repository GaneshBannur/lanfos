import os
import json
import numpy as np
import sklearn.cluster
from pathlib import Path

def load_features(feature_dir, feature_level):
    with open(os.path.join(feature_dir, "sorted_train_image_names.json"), "r") as json_fp:
        sorted_image_names = json.load(json_fp)
    features = []
    for image_name in sorted_image_names:
        image_feature_path = os.path.join(feature_dir, image_name+f"_f_{feature_level}.npy")
        image_feature = np.load(image_feature_path)
        features.append(image_feature)
    features = np.concatenate(features, axis=0)
    return features

def replace_with_feature(cluster_centers, features):
    similarities = np.matmul(cluster_centers, features.T)
    closest_feature_idx = np.argmax(similarities, axis=-1)
    replaced_centers = features[closest_feature_idx]
    return replaced_centers

def kmeans(feature_dir, output_dir, num_clusters, feature_level):
    features = load_features(feature_dir, feature_level)
    cluster_centers, labels, inertia = sklearn.cluster.k_means(features, num_clusters)
    cluster_centers = replace_with_feature(cluster_centers, features)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    np.save(output_dir/"cluster_centers.npy", cluster_centers)
    np.save(output_dir/"labels.npy", labels)

if __name__=="__main__":
    import importlib, argparse, sys, shutil
    from typing import Any

    class runtime_input:
        num_clusters: int
        data_dir: str
        output_dir: str
        feature_level: Any

    def import_from_filepath(module_name, filepath):
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config_filepath", type=str, required=True)
    args = arg_parser.parse_args()
    runtime_input = import_from_filepath("runtime_input", args.config_filepath)
    output_dir = Path(runtime_input.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config_filepath, output_dir/"config.py")

    kmeans(runtime_input.data_dir, runtime_input.output_dir, runtime_input.num_clusters, runtime_input.feature_level)
