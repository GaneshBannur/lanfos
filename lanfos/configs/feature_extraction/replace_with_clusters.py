dataset_path: str   # Path to dataset's top level folder
feature_level: str # One of "small", "medium" or "large"
feature_dir = f"{dataset_path}/siglip_features"
output_dir = f"{dataset_path}/clustered_siglip_features"
cluster_result_dir = f"{dataset_path}/clusters_{feature_level}"
replace_test_images = True
test_cam_names_path = f"{dataset_path}/test_cam_names.json"
