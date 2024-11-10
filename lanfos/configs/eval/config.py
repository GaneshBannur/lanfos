save_path: str  # Path where trained model is saved
gt_json_folder: str     # Directory which has ground truth json files with same format as LangSplat dataset
dataset_path: str    # Path to dataset's top level folder
dataset_name: str
model_dir = f"{save_path}/{dataset_name}"
output_dir = f"{model_dir}/eval_result"
mask_thresh = 0.1
torch_device = "cuda"
siglip_model_name = "ViT-B-16-SigLIP-512"
siglip_pretraining_dataset = "webli"
siglip_weight_precision = "fp16"
feature_levels = ["small", "medium", "large"]
clustering_result_filepaths = {
    level: f"{dataset_path}/clusters_{level}/cluster_centers.npy"
    for level in feature_levels
}
data_device = "cpu"
test_cam_names_path = f"{dataset_path}/test_cam_names.json"
