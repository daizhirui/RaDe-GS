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
import time
from argparse import ArgumentParser
from os import makedirs

import cv2
import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from scene import Scene
from utils.general_utils import safe_state


class CpuTimer:

    def __init__(self, message, repeats: int = 1, warmup: int = 0):
        self.message = message
        self.repeats = repeats
        self.warmup = warmup
        self.cnt = 0
        self.t = 0
        self.average_t = 0
        self._total_t = 0
        self.total_t = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        if self.cnt < self.warmup:
            self.cnt += 1
            return
        self.cnt += 1
        assert self.cnt <= self.repeats
        self.t = self.end - self.start
        self._total_t += self.t
        self.average_t = self._total_t / (self.cnt - self.warmup)
        self.total_t = self.average_t * self.cnt
        tqdm.write(f"{self.message}: {self.t:.6f}(cur)/{self.average_t:.6f}(avg)/{self.total_t:.6f}(total) seconds")


class GpuTimer:

    def __init__(self, message, repeats: int = 1, warmup: int = 0):
        self.message = message
        self.repeats = repeats
        self.warmup = warmup
        self.cnt = 0
        self.t = 0
        self.average_t = 0
        self._total_t = 0
        self.total_t = 0

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()
        self.t = self.start.elapsed_time(self.end) / 1e3
        if self.cnt < self.warmup:
            self.cnt += 1
            return
        self.cnt += 1
        assert self.cnt <= self.repeats
        self._total_t += self.t
        self.average_t = self._total_t / (self.cnt - self.warmup)
        self.total_t = self.average_t * self.cnt
        tqdm.write(f"{self.message}: {self.t:.6f}(cur)/{self.average_t:.6f}(avg)/{self.total_t:.6f}(total) seconds")


def compute_mae(pred, gt):
    mask = gt > 1e-3
    error = np.mean(np.abs(pred[mask] - gt[mask]))  # mean absolute error
    return error


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, dataset_path=None):
    output_dir = os.path.join(model_path, name, "ours_{}".format(iteration))
    render_path = os.path.join(output_dir, "renders")
    gts_path = os.path.join(output_dir, "gt")
    render_depth_path = os.path.join(output_dir, "depth")
    render_range_path = os.path.join(output_dir, "range")
    render_normal_path = os.path.join(output_dir, "normal")

    makedirs(render_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_range_path, exist_ok=True)
    makedirs(render_normal_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    timer = GpuTimer("Rendering", repeats=len(views), warmup=1)
    errors = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress", ncols=80)):
        tqdm.write(f"{idx}")
        with timer:
            render_pkg = render(view, gaussians, pipeline, background, kernel_size=kernel_size)
            range_img = torch.linalg.norm(render_pkg["median_coord"], dim=0).detach().cpu().numpy()

        rgb_img = render_pkg["render"]
        depth_img = render_pkg["median_depth"].detach().cpu().numpy()[0, ...]
        depth_img_jet = cv2.applyColorMap(
            cv2.normalize(
                depth_img,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8UC1,
            ),
            cv2.COLORMAP_JET,
        )

        range_img_jet = cv2.applyColorMap(
            cv2.normalize(
                range_img,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8UC1,
            ),
            cv2.COLORMAP_JET,
        )

        normal_img = render_pkg["normal"].detach().cpu().numpy()
        normal_img = normal_img.transpose(1, 2, 0)
        normal_img = normal_img / np.linalg.norm(normal_img, axis=-1, keepdims=True)

        gt = view.original_image[0:3, :, :]
        gt = gt.cpu().numpy().transpose(1, 2, 0)
        rgb_img = rgb_img.cpu().numpy().transpose(1, 2, 0)

        if dataset_path is not None:
            gt_range_file = os.path.join(dataset_path, f"scans/test/range/{idx:06d}.tiff")
            gt_range = cv2.imread(gt_range_file, cv2.IMREAD_UNCHANGED)
            error = compute_mae(range_img, gt_range) * 100
            tqdm.write(f"Range error: {error:.3f} cm")
            errors.append(error)

        # save range
        cv2.imwrite(os.path.join(render_range_path, "{0:06d}".format(idx) + ".tiff"), range_img.astype(np.float32))
        cv2.imwrite(os.path.join(render_range_path, "colored_{0:06d}".format(idx) + ".png"), range_img_jet)

        # save depth
        cv2.imwrite(os.path.join(render_depth_path, "{0:06d}".format(idx) + ".tiff"), depth_img.astype(np.float32))
        cv2.imwrite(os.path.join(render_depth_path, "colored_{0:06d}".format(idx) + ".png"), depth_img_jet)

        # save normal
        cv2.imwrite(
            os.path.join(render_normal_path, "{0:06d}".format(idx) + ".tiff"),  # tiff supports float32
            normal_img[:, :, ::-1].astype(np.float32),  # RGB to BGR
        )
        cv2.imwrite(
            os.path.join(render_normal_path, "colored_{0:06d}".format(idx) + ".png"),
            ((normal_img + 1) * 0.5 * 255).astype(np.uint8)[..., ::-1],  # RGB to BGR
        )

        cv2.imwrite(
            os.path.join(render_path, "{0:06d}".format(idx) + ".png"),
            np.clip(rgb_img[:, :, ::-1] * 255, 0, 255).astype(np.uint8),  # RGB to BGR
        )
        cv2.imwrite(
            os.path.join(gts_path, "{0:06d}".format(idx) + ".png"),  # png supports uint8 or uint16
            np.clip(gt[:, :, ::-1] * 255, 0, 255).astype(np.uint8),  # RGB to BGR
        )

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    min_error = np.min(errors)
    max_error = np.max(errors)
    num_params = sum(
        p.numel()
        for p in (
            gaussians._xyz,
            gaussians._features_dc,
            gaussians._features_rest,
            gaussians._scaling,
            gaussians._rotation,
            gaussians._opacity,
            gaussians.max_radii2D,
        )
    )
    print(
        f"Test complete:\n"
        f"{len(views)} images rendered in total.\n"
        f"model size: {num_params / 1e6:.3f}M.\n"
        f"timing(second): (itr){timer.average_t:.6f}/(total){timer.total_t:.6f}.\n"
        f"error(cm): (mean){mean_error:.3f}/(std){std_error:.3f}/(min){min_error:.3f}/(max){max_error:.3f}."
    )
    for i, error in enumerate(errors):
        print(f"error {i}: {error:.3f} cm.")
    np.save(os.path.join(output_dir, "errors.npy"), errors)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=iteration,
            shuffle=False,
            skip_train=skip_train,
            skip_test=skip_test,
        )

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                dataset.kernel_size,
                dataset.source_path,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                dataset.kernel_size,
                dataset.source_path,
            )


def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)


if __name__ == "__main__":
    main()
