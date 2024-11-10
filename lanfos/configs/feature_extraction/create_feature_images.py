dataset_path: str   # Path to dataset's top level folder
sam_ckpt_path = "sam_vit_h_4b8939.pth"
image_dir_path = f"{dataset_path}/images"
save_dir_path = f"{dataset_path}/siglip_features"
image_text_model_name = "ViT-B-16-SigLIP-512"
image_text_pretraining_dataset = "webli"
image_text_weight_precision = "fp16"
torch_device = "cuda"
test_cam_names_path = f"{dataset_path}/test_cam_names.json"
test_images_present = True
image_file_extension = "jpg"
