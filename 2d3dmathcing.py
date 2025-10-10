from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import random
import cv2
import time
import open3d as o3d
import os

from tqdm import tqdm

np.random.seed(1428) # do not change this seed
random.seed(1428) # do not change this seed

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def pnpsolver(query, model, cameraMatrix=0, distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    # TODO: solve PnP problem using OpenCV
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = matcher.knnMatch(desc_query, desc_model, k=2)
    good_matches = []
    LOWE_RATIO = 0.7
    for m, n in matches:
        if m.distance < LOWE_RATIO * n.distance:
            good_matches.append(m)
    # print(f"Number of matches: {len(matches)} -> {len(good_matches)} ")
    pts2d = np.float32([kp_query[m.queryIdx] for m in good_matches])
    pts3d = np.float32([kp_model[m.trainIdx] for m in good_matches])
    kp_query = pts2d
    # print("Start solving PnP...")
    # start_time = time.time()
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(pts3d, pts2d, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_P3P)
    # end_time = time.time()

    # print(f"PnP solved in {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))} with {len(inliers)} inliers")
    # Hint: you may use "Descriptors Matching and ratio test" first
    return retval, rvec, tvec, inliers

def rotation_error(rotq_gt, rotq_est):
    #TODO: calculate rotation error
    # input: quaternion in shape (1,4) [x,y,z,w]
    # R_gt = R_err * R_est
    R_gt = R.from_quat(rotq_gt)
    R_est = R.from_quat(rotq_est)
    R_err = R_gt * R_est.inv()
    angle_rad = np.linalg.norm(R_err.as_rotvec())
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def translation_error(t1, t2):
    #TODO: calculate translation error
    err = np.linalg.norm(np.abs(t1 - t2), ord=2)
    return err

def visualization(o3d_parameters, points3D_df):
    # === 1. 讀取點雲 ===
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector()
    pcd.colors = o3d.utility.Vector3dVector()
    for row in points3D_df.itertuples():
        assert len(row.XYZ) == 3
        assert len(row.RGB) == 3
        pcd.points.append(row.XYZ)
        pcd.colors.append(row.RGB / 255.0)

    # pcd.points = o3d.utility.Vector3dVector(points3D_df[["XYZ"]].values.reshape(-1).astype(np.float64))
    # pcd.colors = o3d.utility.Vector3dVector(points3D_df[["RGB"]].values.reshape(-1).astype(np.float64) / 255.0)
    # pcd.paint_uniform_color([0.7, 0.7, 0.7])

    # === 2. 建立所有相機的視覺化幾何 ===
    cameras = []
    cam_centers = []

    for i, param in enumerate(o3d_parameters):
        # 使用 Open3D 提供的相機可視化工具
        cam = o3d.geometry.LineSet.create_camera_visualization(
            intrinsic=param.intrinsic,
            extrinsic=param.extrinsic,
            scale=0.2
        )
        cam.paint_uniform_color([1, 0, 0])  # 紅色
        cameras.append(cam)

        # 取相機中心位置 (C = -R^T t)
        extr = np.asarray(param.extrinsic)
        R = extr[:3, :3]
        t = extr[:3, 3]
        center = -R.T @ t
        cam_centers.append(center)

    # === 3. 繪製相機軌跡 ===
    cam_centers = np.array(cam_centers)
    lines = [[i, i + 1] for i in range(len(cam_centers) - 1)]
    traj = o3d.geometry.LineSet()
    traj.points = o3d.utility.Vector3dVector(cam_centers)
    traj.lines = o3d.utility.Vector2iVector(lines)
    traj.paint_uniform_color([0, 1, 0])  # 綠色軌跡

    # === 4. 一起繪出 ===
    o3d.visualization.draw_geometries([pcd, traj, *cameras])
    # o3d.visualization.draw_geometries([traj, *cameras])

if __name__ == "__main__":
    # Load data
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")

    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)


    IMAGE_ID_LIST = list(range(1,15))
    # IMAGE_ID_LIST = list(range(1,294))
    r_list = []
    t_list = []
    rotation_error_list = []
    translation_error_list = []
    for idx in tqdm(IMAGE_ID_LIST):
        # Load query image
        fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
        rimg = cv2.imread("data/frames/" + fname, cv2.IMREAD_GRAYSCALE)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp
        retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model))
        rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat() # Convert rotation vector to quaternion in shape (1,4) [x,y,z,w]
        tvec = tvec.reshape(1,3) # Reshape translation vector
        r_list.append(rvec)
        t_list.append(tvec)

        # Get camera pose groudtruth
        ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
        tvec_gt = ground_truth[["TX","TY","TZ"]].values

        # Calculate error
        r_error = rotation_error(rotq_gt, rotq)
        t_error = translation_error(tvec_gt, tvec)
        rotation_error_list.append(r_error)
        translation_error_list.append(t_error)

    # TODO: calculate median of relative rotation angle differences and translation differences and print them
    # r_error_median = np.median(rotation_error_list)
    # t_error_median = np.median(translation_error_list)
    # print(f"Rotation error median (degrees): {r_error_median:.4f}")
    # print(f"Translation error median: {t_error_median:.4f}")

    # TODO: result visualization
    # Camera2World_Transform_Matrixs = []
    o3d_parameters = []
    intrinsic = o3d.camera.PinholeCameraIntrinsic(1080, 1920, 1868.27, 1869.18, 540, 960)
    for r, t in zip(r_list, t_list):
        # TODO: calculate camera pose in world coordinate system
        rotation_matrix = R.from_rotvec(r.reshape(1,3)).as_matrix() # Convert rotation vector to rotation matrix
        translation_vector = t.reshape(3,)
        c2w = np.eye(4)
        c2w[:3, :3] = rotation_matrix
        c2w[:3, 3] = translation_vector
        # Camera2World_Transform_Matrixs.append(c2w)
        param = o3d.camera.PinholeCameraParameters()
        param.extrinsic = c2w
        param.intrinsic = intrinsic
        o3d_parameters.append(param)
    visualization(o3d_parameters, points3D_df)