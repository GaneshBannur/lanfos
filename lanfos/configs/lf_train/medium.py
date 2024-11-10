save_path: str  # Path where trained model will be saved
source_path: str    # Path to dataset's top level folder
gs_ckpt_path: str   # Path to .pth file of RGB Gaussian Splatting
dataset_name: str
test_frequency = 4_200
feature_level = "medium"
lf_dir = f"{source_path}/siglip_features"
model_path = f"{save_path}/{dataset_name}/{feature_level}"
lf_clusters_path = f"{source_path}/clusters_{feature_level}/cluster_centers.npy"
iterations = 52_500
test_iterations = [1]+[test_frequency*i for i in range(1, iterations//test_frequency)] + [iterations]
save_iterations = [iterations]
checkpoint_iterations = [iterations]
debug_from = -1
detect_anomaly = False
quiet = False
lf_decoder_dims = [3, 48, 192, 192]
lf_decoder_lr = 1e-3
lf_lr = 1e-3
lf_decoder_batch_size = 128
data_device = "cpu"
eval = True
test_cam_names_path = f"{source_path}/test_cam_names.json"
lf_dim = 3
num_lf_iter = 300
num_lf_decoder_iter = 750
