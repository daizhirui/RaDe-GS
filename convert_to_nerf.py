#!/usr/bin/env python
import os
import pickle
import yaml
import numpy as np
import json
import fire

from utils.graphics_utils import focal2fov

def sddf_header_to_nerf(sensor_setting):
    transforms_header = {
        "camera_model": "OPENCV",
        "fl_x": sensor_setting["camera_fx"], 
        "fl_y": sensor_setting["camera_fy"], 
        "cx": sensor_setting["camera_cx"], 
        "cy": sensor_setting["camera_cy"], 
        "w": sensor_setting["image_width"], 
        "h": sensor_setting["image_height"],
        "camera_angle_x": focal2fov(sensor_setting["camera_fx"], sensor_setting["image_width"]),
        "camera_angle_y": focal2fov(sensor_setting["camera_fy"], sensor_setting["image_height"]),
        "frames": []
    }
    for field in ["k1", "k2", "k3", "k4", "p1", "p2"]:
        transforms_header[field] = 0.0 # no distortion
    return transforms_header

def sddf_pose_to_nerf_pose(R_wTc, T_wTc):
    oTc = np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ])


    R_oTw = oTc @ R_wTc.T # try R_oTw
    T_oTw = oTc @ (-R_wTc.T @ T_wTc) # try T_oTw

    R_wTo = R_oTw.T
    T_wTo = -R_oTw.T @ T_oTw

    R_wTo_nerf = R_wTo
    R_wTo_nerf[:, 1:3] *= -1
    return R_wTo_nerf, T_wTo

def get_matrix(R, T):
    matrix = np.eye(4)
    matrix[:3, :3] = R
    matrix[:3, 3] = T
    return matrix

def convert_to_nerf(base_path, output_file=None):

    sensor_setting_file_path = os.path.join(base_path, 'sensor_setting.yaml')
    with open(sensor_setting_file_path, 'r') as f:
        sensor_setting = yaml.safe_load(f)

    transforms_json = sddf_header_to_nerf(sensor_setting)

    poses_pickle_file_path = os.path.join(base_path, 'poses.pkl') # path/train/poses.pkl
    with open(poses_pickle_file_path, 'rb') as f:
        poses = pickle.load(f)
    

    rgb_image_path_base = os.path.join(base_path, 'rgb')
    depth_image_path_base = os.path.join(base_path, 'depth')

    idx_to_image_path = lambda idx: f"{idx:06}.png"

    frames_list = [
        {
            'file_path': os.path.join(rgb_image_path_base, idx_to_image_path(idx)),
            'depth_file_path': os.path.join(depth_image_path_base, idx_to_image_path(idx)),
            'transform_matrix': get_matrix(*sddf_pose_to_nerf_pose(*pose)).tolist()
        } for idx, pose in enumerate(poses)
    ]

    transforms_json['frames'] = frames_list

    if output_file is None:
        json_string = json.dumps(transforms_json, indent=2)
        print(json_string)
    else:
        with open(output_file, 'wb') as f:
            json.dump(transforms_json, f, indent=2)


if __name__ == "__main__":
    fire.Fire(convert_to_nerf)
    
