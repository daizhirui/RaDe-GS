import os
import pickle
import yaml
import numpy as np

def sddf_header_to_nerf(sensor_setting):
    transforms_header = {
        "camera_model": "OPENCV",
        "fl_x": sensor_setting["camera_fx"], 
        "fl_y": sensor_setting["camera_fy"], 
        "cx": sensor_setting["cx"], 
        "cy": sensor_setting["cy"], 
        "w": sensor_setting["image_width"], 
        "h": sensor_setting["image_height"],
        "frames": []
    }
    for field in ["k1", "k2", "k3", "k4", "p1", "p2"]:
        transforms_header[field] = 0.0 # no distortion
    return transforms_header

def convert_to_nerf(path, suffix):
    base_path = os.path.join(path, suffix)
    sensor_setting_file_path = os.path.join(base_path, 'sensor_setting.yaml')
    with open(sensor_setting_file_path, 'r') as f:
        sensor_setting = yaml.safe_load(f)

    transforms_json = sddf_header_to_nerf(sensor_setting)

    poses_pickle_file_path = os.path.join(base_path, 'poses.pkl')
    with open(poses_pickle_file_path, 'rb') as f:
        poses = pickle.load(f)
    

    oTc = np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ])

    rgb_image_path_base = os.path.join(base_path, 'rgb')
    depth_image_path_base = os.path.join(base_path, 'depth')


    for idx, (R_wTc, T_wTc) in enumerate(poses):
        image_name = f"{idx:06}.png"
        rgb_image_file_path = os.path.join(
            rgb_image_path_base,
            image_name
        )

        depth_image_file_path = os.path.join(
            depth_image_path_base,
            image_name
        )

        R_oTw = oTc @ R_wTc.T # try R_oTw
        T_oTw = oTc @ (-R_wTc.T @ T_wTc) # try T_oTw

        R_wTo = R_oTw.T
        T_wTo = -R_oTw.T @ T_oTw

        R_wTo_nerf = 


    
