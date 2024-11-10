import json
import os
import glob
import random
import sys
import torch
import cv2
from collections import defaultdict
from pathlib import Path
from typing import Union
import numpy as np
from tqdm import tqdm
from lanfos.eval.colormaps import apply_colormap, ColormapOptions
from lanfos.eval.utils import colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result
from lanfos.language_field.scene import Scene, GaussianModel
from lanfos.language_field.gaussian_renderer import render


def eval_gt_lerfdata(json_folder: Union[str, Path] = None, ouput_path: Path = None):
    """
    organise lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: image_name
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.json')))
    image_paths  ={}
    gt_ann = {}
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)
        
        h, w = gt_data['info']['height'], gt_data['info']['width']
        image_filename = gt_data['info']['name']
        image_name = image_filename.split('.')[0]
        image_paths[image_name] = os.path.join(json_folder, image_filename)
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']
            box = np.asarray(prompt_data['bbox']).reshape(-1)           # x1y1x2y2
            mask = polygon_to_mask((h, w), prompt_data['segmentation'])
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask
            
            save_path = ouput_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)

        gt_ann[image_name] = img_ann

    return gt_ann, image_paths


def evaluate_segmentation(image, image_path, image_annotation, text_query_list, text_query_relevancies,
                          threshold = 0.5, colormap_options = None):
    iou_list = []
    
    for text_query in tqdm(text_query_list, desc="Segmentation evaluation progress", leave=False):
        relevancies = text_query_relevancies[text_query]

        relevancy_output_path = Path(image_path)/"heatmap"/f"{text_query}.png"
        relevancy_output_path.parent.mkdir(exist_ok=True, parents=True)
        colormap_saving(relevancies.unsqueeze(-1), colormap_options, relevancy_output_path)

        clipped_relevancies = torch.clip(relevancies - 0.5, 0, 1).unsqueeze(-1)
        composited_heatmap = apply_colormap(
            clipped_relevancies/(clipped_relevancies.max() + 1e-6),
            colormap_options = ColormapOptions("turbo")
        )
        mask = (relevancies < threshold).squeeze()
        composited_heatmap[mask, :] = image[mask, :] * 0.3
        composited_output_path = Path(image_path)/"composited"/f"{text_query}.png"
        composited_output_path.parent.mkdir(exist_ok=True, parents=True)
        colormap_saving(composited_heatmap, colormap_options, composited_output_path)

        mask_pred = (relevancies.cpu().numpy() > threshold).astype(np.uint8)
        mask_gt = image_annotation[text_query]['mask'].astype(np.uint8)
        
        intersection = np.sum(np.logical_and(mask_gt, mask_pred))
        union = np.sum(np.logical_or(mask_gt, mask_pred))
        iou = intersection/union
        iou_list.append(iou)

        mask_save_path = Path(image_path)/f"chosen_{text_query}.png"
        vis_mask_save(mask_pred, mask_save_path)

    return iou_list


def evaluate_localization(image, image_path, image_annotation, text_query_list, text_query_relevancies):
    localization_output_path = Path(image_path)/"localization"
    localization_output_path.mkdir(exist_ok=True, parents=True)

    accuracy_num = 0
    for text_query in tqdm(text_query_list, desc="Localization evaluation progress", leave=False):
        relevancies = text_query_relevancies[text_query]

        # Switch x and y to convert from array indexing (first dim is up/down, second dim is left/right) to 
        # image indexing (first dim is left/right like x-axis and second dim is up/down like y-axis (but flipped))
        max_relevancy_points_y, max_relevancy_points_x = torch.nonzero(relevancies==relevancies.max(), as_tuple=True)
        max_relevancy_points = torch.stack((max_relevancy_points_x, max_relevancy_points_y), dim=1)
        
        for box in image_annotation[text_query]['bboxes'].reshape(-1, 4):
            flag = 0
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            for point in max_relevancy_points:
                if (point[0] >= x_min and point[0] <= x_max and 
                    point[1] >= y_min and point[1] <= y_max):
                    accuracy_num += 1
                    flag = 1
                    break
            if flag != 0:
                break

        clipped_relevancies = torch.clip(relevancies - 0.5, 0, 1).unsqueeze(-1)
        composited_heatmap = apply_colormap(
            clipped_relevancies/(clipped_relevancies.max() + 1e-6),
            ColormapOptions("turbo")
        )
        mask = (relevancies < 0.5).squeeze()
        composited_heatmap[mask, :] = image[mask, :] * 0.3
        save_path = localization_output_path/f"{text_query}.png"
        show_result(
            composited_heatmap.cpu().numpy(),
            max_relevancy_points.cpu().numpy(),
            image_annotation[text_query]['bboxes'],
            save_path
        )

    return accuracy_num

@torch.no_grad()
def setup_model(dataset_args, all_args, torch_device="cuda"):
    gaussians = GaussianModel(dataset_args.sh_degree)
    scene = Scene(dataset_args, gaussians, shuffle=False, include_feature=all_args.include_feature)
    checkpoint = os.path.join(all_args.model_path, f"chkpnt{all_args.iterations}.pth")
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, all_args, all_args.include_feature, mode='test',
                      decoder_dims=all_args.decoder_dims, cluster_centers_path=all_args.cluster_centers_path,
                      data_device=all_args.data_device)
    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=torch_device)
    return scene, background

@torch.no_grad()
def render_dist(image_name, scene, pipeline, background, all_gs_args):
    '''
    Renders distribution over cluster centers for each pixel
    '''
    view_to_render = None
    for view in scene.getTestCameras():
        cur_view_name = view.image_name.split(".")[0]
        if cur_view_name==image_name:
            view_to_render = view
            break
    
    assert view_to_render is not None, f"Cannot find {image_name}. Specify the correct path for test_cam_names_path in the model parameters"
    rendering = render(view_to_render, scene.gaussians, pipeline, background, all_gs_args, get_distribution=True)
    # Downstream code expects distribution dimension at the end, i.e., shape [..., N]
    rendering = rendering.movedim(0, -1)
    return rendering

@torch.no_grad()
def evaluate(feature_dirs, output_path, query_module, gt_json_folder, mask_thresh, gs_dataset_args,
             gs_pipeline_args, gs_train_args, test_cam_names_path, torch_device = "cuda"):
    scenes = {}
    backgrounds = {}
    for feature_level, feature_dir_path in feature_dirs.items():
        scenes[feature_level], backgrounds[feature_level] = setup_model(gs_dataset_args[feature_level],
                                                                        gs_train_args[feature_level],
                                                                        torch_device=torch_device)    
    colormap_options = ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    gt_annotation, image_paths = eval_gt_lerfdata(Path(gt_json_folder), Path(output_path))
    output_path = Path(output_path)
    with open(test_cam_names_path, 'r') as f:
        test_cam_names = json.load(f)
    iou_list = []
    accuracy_num = 0
    for image_name in tqdm(test_cam_names, desc="Image progress"):
        image_gt = gt_annotation[image_name]
        cluster_dists = {}
        for feature_level, feature_dir_path in feature_dirs.items():
            cluster_dists[feature_level] = render_dist(image_name, scenes[feature_level],
                                                       gs_pipeline_args[feature_level], backgrounds[feature_level],
                                                       gs_train_args[feature_level])
            
        image_path = output_path/image_name
        image_path.mkdir(exist_ok=True, parents=True)

        bgr_img = cv2.imread(image_paths[image_name])
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb_img = (rgb_img/255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(torch_device)

        text_query_list = list(image_gt.keys())
        relevancies = {}
        for text_query in tqdm(text_query_list, desc="Query progress", leave=False):
            relevancies[text_query] = query_module.get_relevancy(text_query, cluster_dists)

        iou_list_for_image = evaluate_segmentation(rgb_img, image_path, image_gt, text_query_list, relevancies,
                                                   threshold=mask_thresh, colormap_options=colormap_options)
        iou_list.extend(iou_list_for_image)

        accuracy_num_img = evaluate_localization(
            rgb_img,
            image_path,
            image_gt,
            text_query_list,
            relevancies
        )
        accuracy_num += accuracy_num_img

    mean_iou = sum(iou_list)/len(iou_list)
    print(f'Mask threshold: {mask_thresh}')
    print(f"Mean IoU: {mean_iou:.4f}")

    total_bboxes = 0
    for img_ann in gt_annotation.values():
        total_bboxes += len(list(img_ann.keys()))
    acc = accuracy_num/total_bboxes
    print("Localization accuracy: " + f'{acc:.4f}')


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

if __name__ == "__main__":
    import importlib
    import argparse
    import shutil
    from typing import Dict, Any
    from argparse import Namespace, ArgumentParser
    from lanfos.image_text_models.siglip import SigLIPInference
    from lanfos.inference.query_module import QueryModule
    from lanfos.language_field.arguments import ModelParams, PipelineParams, get_combined_args

    class EvalConfig:
        '''
        Runtime input format
        '''
        model_dir: str
        output_dir: str
        gt_json_folder: str
        mask_thresh: float
        torch_device: str
        siglip_model_name: str
        siglip_pretraining_dataset: str
        siglip_weight_precision: str
        feature_levels: list
        clustering_result_filepaths: Dict[Any, str]
        data_device: str
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
    EvalConfig = import_from_filepath("EvalConfig", args.config_filepath)

    output_dir = Path(EvalConfig.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config_filepath, output_dir/"config.py")
    
    seed_num = 42
    seed_everything(seed_num)

    ckpt_dirs = {
        level: os.path.join(EvalConfig.model_dir, level)
        for level in EvalConfig.feature_levels
    }

    siglip_model = SigLIPInference(EvalConfig.siglip_model_name, EvalConfig.siglip_pretraining_dataset,
                                   EvalConfig.siglip_weight_precision, torch_device=EvalConfig.torch_device)
    query_module = QueryModule(siglip_model, EvalConfig.clustering_result_filepaths,
                               data_device=EvalConfig.data_device)
    dataset_args = {}
    pipeline_args = {}
    gs_train_args = {}
    new_args = Namespace(data_device=EvalConfig.data_device, eval=True)
    for feature_level, feature_dir in ckpt_dirs.items():
        parser = ArgumentParser(description="Evaluation script parameters")
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        train_args = get_combined_args(parser, feature_dir)
        vars(train_args).update(vars(new_args))
        dataset_args[feature_level] = model.extract(train_args)
        pipeline_args[feature_level] = pipeline.extract(train_args)
        gs_train_args[feature_level] = train_args

    evaluate(ckpt_dirs, output_dir, query_module, EvalConfig.gt_json_folder, EvalConfig.mask_thresh, dataset_args,
             pipeline_args, gs_train_args, EvalConfig.test_cam_names_path, torch_device=EvalConfig.torch_device)
