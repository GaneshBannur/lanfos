import os
import shutil
import json
import numpy as np
from pathlib import Path

def get_feature_filename_num_features_map(feature_dir, feature_level):
    with open(os.path.join(feature_dir, "sorted_train_image_names.json"), 'r') as json_fp:
        sorted_image_names = json.load(json_fp)
    feature_filename_num_features_map = []
    for image_name in sorted_image_names:
        feature_filename = image_name+f"_f_{feature_level}.npy"
        image_feature_path = os.path.join(feature_dir, feature_filename)
        features = np.load(image_feature_path)
        feature_filename_num_features_map.append((feature_filename, features.shape[0]))
    return feature_filename_num_features_map

def copy_seg_map(data_dir, output_dir, feature_level):
    for filename in os.listdir(data_dir):
        if filename.endswith(f"_s_{feature_level}.npy"):
            source_path = os.path.join(data_dir, filename)
            target_path = os.path.join(output_dir, filename)
            shutil.copy(source_path, target_path)

def save_low_dim(feature_filename_num_features_map, output_dir, all_features):
    start_idx = 0
    for feature_filename, num_features in feature_filename_num_features_map:
        feature_path = os.path.join(output_dir, feature_filename)
        np.save(feature_path, all_features[start_idx: start_idx+num_features])
        start_idx += num_features

def replace_test(feature_dir, cluster_centers, test_cam_names_path, feature_level):
    with open(test_cam_names_path, 'r') as json_fp:
        test_cam_name_list = json.load(json_fp)

    test_features = []
    num_features_per_image = []
    feature_filename_list = []
    for test_cam_name in test_cam_name_list:
        feature_filename = test_cam_name+f"_f_{feature_level}.npy"
        image_feature_path = os.path.join(feature_dir, feature_filename)
        image_features = np.load(image_feature_path)
        test_features.append(image_features)
        num_features_per_image.append(image_features.shape[0])
        feature_filename_list.append(feature_filename)
    
    test_features = np.concatenate(test_features, axis=0)
    similarities = np.matmul(test_features, cluster_centers.T)
    selected_clusters = np.argmax(similarities, axis=-1)
    replaced_test_features = cluster_centers[selected_clusters]
    return replaced_test_features, feature_filename_list, num_features_per_image

def replace_with_custers(cluster_result_dir, output_dir, feature_dir, feature_level, replace_test_images,
                         test_cam_names_path=''):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    copy_seg_map(feature_dir, output_dir, feature_level)
    cluster_centers = np.load(os.path.join(cluster_result_dir, "cluster_centers.npy"))
    cluster_labels = np.load(os.path.join(cluster_result_dir, "labels.npy"))
    replaced_features = cluster_centers[cluster_labels]
    feature_filename_num_features_map = get_feature_filename_num_features_map(feature_dir, feature_level)
    save_low_dim(feature_filename_num_features_map, output_dir, replaced_features)
    if replace_test_images==True:
        assert Path(test_cam_names_path).exists(), "test_cam_names_path must point to a valid file if replace_test_images=True"
        replaced_test_feats, test_filenames, test_num_feats_per_img = replace_test(feature_dir, cluster_centers,
                                                                                   test_cam_names_path, feature_level)
        save_low_dim(zip(test_filenames, test_num_feats_per_img), output_dir, replaced_test_feats)

if __name__ == '__main__':
    import importlib.util
    import sys
    import argparse
    from typing import Any

    class runtime_input:
        feature_dir: str
        output_dir: str
        cluster_result_dir: str
        feature_level: Any
        replace_test_images: bool
        test_cam_names_path: str

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
    replace_with_custers(runtime_input.cluster_result_dir, runtime_input.output_dir, runtime_input.feature_dir,
                         runtime_input.feature_level, runtime_input.replace_test_images,
                         test_cam_names_path=runtime_input.test_cam_names_path)
