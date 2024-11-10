#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from lanfos.language_field.gaussian_renderer import render_raw
import sys
from lanfos.language_field.scene import Scene, GaussianModel
from lanfos.language_field.utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from lanfos.language_field.arguments import ModelParams, PipelineParams, OptimizationParams
from lanfos.language_field.utils.loss_utils import mask_l2_loss, mask_l2_unnormalized_loss, l2_loss
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# @torch.compile(fullgraph=True, dynamic=False)
def compiled_for_lf_train(lf_image, pc, viewpoint_camera):
    # lf_image has shape [D, H, W] but the decoder needs [..., D]
    lf_image = lf_image.movedim(0, -1)
    lf_image = pc.lf_decoder(lf_image)
    loss = mask_l2_loss(lf_image, viewpoint_camera.gt_lf.to("cuda"), viewpoint_camera.lf_bg_mask.to("cuda"),
                        viewpoint_camera.lf_num_considered)
    return loss

def loss_for_lf_train(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0,
                      override_color = None):
    render_pkg = render_raw(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=scaling_modifier,
                            override_color=override_color)
    loss = compiled_for_lf_train(render_pkg["lf_image"], pc, viewpoint_camera)
    return loss

@torch.compile(fullgraph=True, dynamic=False)
def compiled_for_unnormalized_lf_loss(lf_image, pc, viewpoint_camera):
    # lf_image has shape [D, H, W] but the decoder needs [..., D]
    lf_image = lf_image.movedim(0, -1)
    lf_image = pc.lf_decoder(lf_image)
    unnormalized_loss = mask_l2_unnormalized_loss(lf_image, viewpoint_camera.gt_lf.to("cuda"),
                                                  viewpoint_camera.lf_bg_mask.to("cuda"),
                                                  viewpoint_camera.lf_num_considered)
    return unnormalized_loss

def unnormalized_lf_loss(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0,
                    override_color = None):
    render_pkg = render_raw(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=scaling_modifier,
                            override_color=override_color)
    unnormalized_loss = compiled_for_unnormalized_lf_loss(render_pkg["lf_image"], pc, viewpoint_camera)
    return unnormalized_loss

@torch.compile(fullgraph=True, dynamic=False)
def compiled_for_decoder_train(lf_image, pc, batch_indices, gt_batch_feature):
    # language_feature_image has shape [D, H, W] but the decoder needs [..., D]
    lf_image = lf_image.movedim(0, -1)
    # Flatten it to allow indexing. Now has shape [H*W, D]
    lf_image = lf_image.flatten(0, 1)
    lf_batch = torch.index_select(input=lf_image, dim=0, index=batch_indices)
    lf_batch = pc.lf_decoder(lf_batch)
    loss = l2_loss(lf_batch, gt_batch_feature)
    return loss

def loss_for_lf_decoder_train(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor,
                              lf_decoder_batch_size, scaling_modifier = 1.0, override_color = None):
    render_pkg = render_raw(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=scaling_modifier,
                            override_color=override_color)
    gt_batch_feature, batch_indices = viewpoint_camera.get_lf_decoder_batch(lf_decoder_batch_size)
    loss = compiled_for_decoder_train(render_pkg["lf_image"], pc, batch_indices, gt_batch_feature)
    return loss

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, gs_ckpt_path,
             debug_from, test_cam_names_path, lf_dim, lf_lr, lf_decoder_lr, lf_decoder_dims, lf_clusters_path,
             lf_decoder_batch_size, num_lf_iter, num_lf_decoder_iter, lf_dir, lf_feature_level):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt, pipe)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, test_cam_names_path, lf_dir, lf_feature_level)
    gaussians.training_setup(opt, lf_dim, lf_lr, lf_decoder_lr, lf_decoder_dims, lf_clusters_path)
    (gs_model_params, gs_first_iter) = torch.load(gs_ckpt_path,)
    gaussians.restore_gs(gs_model_params)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    train_lf_decoder = False
    gaussians.freeze_lf_decoder()
    gaussians.unfreeze_lf()
    cur_num_lf_iter = 0
    cur_num_lf_decoder_iter = 0

    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Render and calculate loss
        if train_lf_decoder==False:
            loss = loss_for_lf_train(viewpoint_cam, gaussians, pipe, bg)
        elif train_lf_decoder==True:
            loss = loss_for_lf_decoder_train(viewpoint_cam, gaussians, pipe, bg, lf_decoder_batch_size)
        
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, loss, iter_start.elapsed_time(iter_end), testing_iterations, scene,
                            pipe, bg)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Optimizer step
            if iteration < opt.iterations:
                if train_lf_decoder==False:
                    gaussians.lf_optimizer.step()
                    gaussians.lf_optimizer.zero_grad(set_to_none = True)
                    cur_num_lf_iter += 1
                    if cur_num_lf_iter==num_lf_iter:
                        cur_num_lf_iter = 0
                        train_lf_decoder = True
                        gaussians.freeze_lf()
                        gaussians.unfreeze_lf_decoder()
                        viewpoint_stack = scene.getTrainCameras().copy()

                elif train_lf_decoder==True:
                    gaussians.lf_decoder_optimizer.step()
                    gaussians.lf_decoder_optimizer.zero_grad(set_to_none = True)
                    cur_num_lf_decoder_iter += 1
                    if cur_num_lf_decoder_iter==num_lf_decoder_iter:
                        cur_num_lf_decoder_iter = 0
                        train_lf_decoder = False
                        gaussians.freeze_lf_decoder()
                        gaussians.unfreeze_lf()
                        viewpoint_stack = scene.getTrainCameras().copy()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(*args):  
    combined_args = {}
    for a in args:
        combined_args.update(vars(a))
    args = Namespace(**combined_args)  
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, loss, elapsed, testing_iterations, scene : Scene, pipe, bg_color):
    if tb_writer:
        tb_writer.add_scalar("train/loss", loss.item(), iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'Test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'Train', 'cameras' : scene.getTrainCameras()})

        lf_decoder_was_training = scene.gaussians.lf_decoder.training
        scene.gaussians.lf_decoder.eval()

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                val_unnormalized_loss = 0
                val_num_elements = 0
                for viewpoint in tqdm(config['cameras'], desc=f"Evaluating {config['name']}", leave=False):
                    val_unnormalized_loss += unnormalized_lf_loss(viewpoint, scene.gaussians, pipe, bg_color)
                    val_num_elements += viewpoint.num_considered
                val_loss = val_unnormalized_loss/val_num_elements         
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: Loss {val_loss}")
                if tb_writer:
                    tb_writer.add_scalar(f"val/{config['name']}_set/loss", val_loss, iteration)

        if lf_decoder_was_training==True:
            scene.gaussians.lf_decoder.train()

        torch.cuda.empty_cache()

if __name__ == "__main__":

    # TEMPORARY
    import torch
    free_gpu_mem_gb = torch.cuda.mem_get_info()[0]/(1024**3)
    print("Waiting for free GPU memory")
    while free_gpu_mem_gb<15:
        free_gpu_mem_gb = torch.cuda.mem_get_info()[0]/(1024**3)
    print("Found free GPU memory")

    from typing import Any, List
    import importlib.util
    from pathlib import Path
    import shutil

    class TrainConfig:
        '''
        Runtime input format
        '''
        lf_feature_level: Any
        lf_dir: str
        source_path: str
        model_path: str
        lf_clusters_path: str
        gs_ckpt_path: str
        test_iterations: List[int]
        save_iterations: List[int]
        checkpoint_iterations: List[int]
        debug_from: int
        detect_anomaly: bool
        quiet: bool
        lf_decoder_dims: List[int]
        lf_decoder_lr: float
        lf_lr: float
        lf_decoder_batch_size: int
        data_device: str
        iterations: int
        eval: bool
        test_cam_names_path: str
        lf_dim: int
        num_lf_iter: int
        num_lf_decoder_iter: int

    def import_from_filepath(module_name, filepath):
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    
    cmd_line_parser = ArgumentParser()
    cmd_line_parser.add_argument("--config_filepath", type=str, required=True)
    cmd_line_args = cmd_line_parser.parse_args()
    TrainConfig = import_from_filepath("TrainConfig", cmd_line_args.config_filepath)
    model_path = Path(TrainConfig.model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cmd_line_args.config_filepath, model_path/"config.py")

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args([])
    args_from_config = Namespace(source_path=TrainConfig.source_path, model_path=TrainConfig.model_path,
                                 data_device=TrainConfig.data_device, iterations=TrainConfig.iterations,
                                 eval=TrainConfig.eval)
    vars(args).update(vars(args_from_config))
    if TrainConfig.iterations not in TrainConfig.save_iterations:
        TrainConfig.save_iterations.append(TrainConfig.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(TrainConfig.quiet)

    torch.autograd.set_detect_anomaly(TrainConfig.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), TrainConfig.test_iterations,
             TrainConfig.save_iterations, TrainConfig.checkpoint_iterations, TrainConfig.gs_ckpt_path,
             TrainConfig.debug_from, TrainConfig.test_cam_names_path, TrainConfig.lf_dim, TrainConfig.lf_lr,
             TrainConfig.lf_decoder_lr, TrainConfig.lf_decoder_dims, TrainConfig.lf_clusters_path,
             TrainConfig.lf_decoder_batch_size, TrainConfig.num_lf_iter, TrainConfig.num_lf_decoder_iter,
             TrainConfig.lf_dir, TrainConfig.lf_feature_level)

    # All done
    print("\nTraining complete.")
