# Convenience definitions
dataset_name = "figurines"
run_name = "lanfos_test"
gs_ckpt_iter = 30_000
test_frequency = 4_200

# Actual runtime inputs
lf_feature_level = "small"
lf_dir = f"/scratch/ganesh/3d_grounding/data/lerf_ovs/{dataset_name}/language_features_siglip_replaced"
source_path = f"/scratch/ganesh/3d_grounding/data/lerf_ovs/{dataset_name}"
model_path = f"/scratch/ganesh/3d_grounding/runs/{run_name}/{dataset_name}/{dataset_name}_{lf_feature_level}"
lf_clusters_path = f"/scratch/ganesh/3d_grounding/lanfos/clustering/siglip/{dataset_name}/clustering_{lf_feature_level}/cluster_centers.npy"
gs_ckpt_path = f"/scratch/ganesh/3d_grounding/runs/3dgs_no_test/{dataset_name}/chkpnt{gs_ckpt_iter}.pth"
iterations = 52_500
test_iterations = [1]+[test_frequency*i for i in range(1, iterations//test_frequency)] + [iterations]
save_iterations = [iterations]
checkpoint_iterations = [iterations]
debug_from = -1
detect_anomaly = False
quiet = False
lf_decoder_dims = [3, 144, 144, 576, 576]
lf_decoder_lr = 1e-3
lf_lr = 1e-3
lf_decoder_batch_size = 128
data_device = "cpu"
eval = True
test_cam_names_path = f"/scratch/ganesh/3d_grounding/data/lerf_ovs/label/{dataset_name}/test_cam_names.json"
lf_dim = 3
num_lf_iter = 300
num_lf_decoder_iter = 750
