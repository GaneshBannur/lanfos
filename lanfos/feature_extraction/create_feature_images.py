import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import json
from pathlib import Path

from lanfos.feature_extraction.segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from lanfos.image_text_models.siglip import SigLIPInference

def get_images(image_dir_path, test_images_present, save_dir_path, image_file_extension,
               test_cam_names_path=None):
    data_list = os.listdir(image_dir_path)
    image_name_list = [n.split(".")[0] for n in data_list]
    if test_images_present==True:
        assert Path(test_cam_names_path).exists() , "test_cam_names_path must point to a valid file when running with eval=True"
        with open(test_cam_names_path, 'r') as f:
            test_cam_names = json.load(f)
        train_image_names = [n for n in image_name_list if n not in test_cam_names]
    else:
        train_image_names = image_name_list

    sorted_train_image_names = sorted(train_image_names)
    with open(os.path.join(save_dir_path, "sorted_train_image_names.json"), "w") as json_fp:
        json.dump(sorted_train_image_names, json_fp)

    image_list = []
    for image_name in image_name_list:
        image_path = os.path.join(image_dir_path, f"{image_name}.{image_file_extension}")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_list.append(image)

    return image_list, image_name_list

def create(image_dir_path, save_dir_path, image_text_model, mask_generator, test_images_present, image_file_extension,
           test_cam_names_path=None, torch_device="cuda"):
    image_list, image_name_list = get_images(image_dir_path, test_images_present, save_dir_path, image_file_extension,
                                             test_cam_names_path=test_cam_names_path)
    mask_generator.predictor.model.to(torch_device)
    for i, img in tqdm(enumerate(image_list), desc="Embedding images", total=len(image_list)):
        image_features, image_seg_maps = segment_and_embed(img, image_text_model, mask_generator)
        for feature_level in image_features.keys():
            feature_save_path = os.path.join(save_dir_path, image_name_list[i]+f"_f_{feature_level}.npy")
            np.save(feature_save_path, image_features[feature_level])
            seg_map_save_path = os.path.join(save_dir_path, image_name_list[i]+f"_s_{feature_level}.npy")
            np.save(seg_map_save_path, image_seg_maps[feature_level])

def segment_and_embed(image, image_text_model, mask_generator):
    seg_images, seg_maps = sam_segmentation(image, mask_generator)
    image_text_embeds = {}
    all_tiles = []
    num_segments_in_mode = []
    for mode in seg_images.keys():
        all_tiles.extend(seg_images[mode])
        num_segments_in_mode.append(len(seg_images[mode]))
    
    cumulative_num_segments_in_mode = [0]
    for num_segments in num_segments_in_mode:
        cumulative_num_segments_in_mode.append(cumulative_num_segments_in_mode[-1]+num_segments)

    all_tiles = [Image.fromarray(i) for i in all_tiles]
    batched_image_tensor = image_text_model.preprocess_image_list(all_tiles)
    image_text_embedding = image_text_model.encode_image(batched_image_tensor)
    image_text_embedding = image_text_embedding.detach().cpu().numpy()
    for idx, mode in enumerate(seg_images.keys()):
        image_text_embeds[mode] = image_text_embedding[cumulative_num_segments_in_mode[idx]: cumulative_num_segments_in_mode[idx+1]]
    
    return image_text_embeds, seg_maps

def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0, 0], dtype=image.dtype)
    x, y, w, h = np.int32(mask['bbox'])
    seg_img = image[y: y+h, x: x+w, ...]
    return seg_img

def sam_segmentation(image, mask_generator):
    # The first output is the mask for the "default" level which is not used
    _, masks_s, masks_m, masks_l = mask_generator.generate(image)
    
    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)
            seg_img_list.append(seg_img)
            seg_map[masks[i]['segmentation']] = i

        return seg_img_list, seg_map

    seg_images, seg_maps = {}, {}
    if len(masks_s)!=0:
        seg_images["small"], seg_maps["small"] = mask2segmap(masks_s, image)
    if len(masks_m)!=0:
        seg_images["medium"], seg_maps["medium"] = mask2segmap(masks_m, image)
    if len(masks_l)!=0:
        seg_images["large"], seg_maps["large"] = mask2segmap(masks_l, image)

    return seg_images, seg_maps

if __name__ == '__main__':
    import importlib
    import argparse
    import sys
    import shutil

    class runtime_input:
        dataset_path: str
        sam_ckpt_path: str
        image_dir_path: str
        save_dir_path: str
        image_text_model_name: str
        image_text_pretraining_dataset: str
        image_text_weight_precision: str
        torch_device: str
        test_images_present: bool
        test_cam_names_path: str
        image_file_extension: str

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
    save_dir_path = Path(runtime_input.save_dir_path)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config_filepath, save_dir_path/"config.py")

    image_text_model = SigLIPInference(runtime_input.image_text_model_name,
                                       runtime_input.image_text_pretraining_dataset,
                                       runtime_input.image_text_weight_precision,
                                       torch_device=runtime_input.torch_device)
    sam = sam_model_registry["vit_h"](checkpoint=runtime_input.sam_ckpt_path).to(runtime_input.torch_device)

    # Default SAM settings
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side = 32,
        points_per_batch = 64,
        pred_iou_thresh = 0.88,
        stability_score_thresh = 0.95,
        stability_score_offset = 1.0,
        box_nms_thresh = 0.7,
        crop_n_layers = 0,
        crop_nms_thresh = 0.7,
        crop_overlap_ratio = 512 / 1500,
        crop_n_points_downscale_factor = 1,
        point_grids = None,
        min_mask_region_area = 0,
        output_mode = "binary_mask"
    )

    create(runtime_input.image_dir_path, save_dir_path, image_text_model, mask_generator,
           runtime_input.test_images_present, runtime_input.image_file_extension,
           test_cam_names_path=runtime_input.test_cam_names_path, torch_device=runtime_input.torch_device)
