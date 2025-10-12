from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import random
import cv2
import time
import open3d as o3d
import os

from tqdm import tqdm, trange

np.random.seed(1428) # do not change this seed
random.seed(1428) # do not change this seed

def my_RANSAC_pnp(pts3d, pts2d, cameraMatrix, distCoeffs, max_iter=2000, err_threshold=3):
    """
    使用 RANSAC 從點中挑出 4 個點，以使用 PnP 得到 camera pose
    
    params:
        pts3d: list of (3,) ndarray, 3D points in world coordinate system
        pts2d: list of (2,) ndarray, 2D points in image coordinate system
        max_iter: 最大迭代次數
        err_threshold: reprojection error threshold to determine inliers (in pixels)
    
    returns:
        rtval: bool, PnP 是否成功
        rvec: (3,) ndarray, rotation vector
        tvec: (3,) ndarray, translation vector
        inliers: (M,) ndarray, 內點的索引
    """
    best_inliers = []
    for _ in trange(max_iter, desc="RANSAC PnP", leave=False):
        sample_idx = np.random.choice(len(pts3d), 4, replace=False)
        sample_pts3d = pts3d[sample_idx]
        sample_pts2d = pts2d[sample_idx]
        _, rvecs, tvecs = cv2.solveP3P(sample_pts3d, sample_pts2d, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_AP3P)
        
        # reprojection
        for rvec, tvec in zip(rvecs, tvecs):
            r_mat = R.from_rotvec(rvec.reshape(1,3)).as_matrix()
            projected_pts = (cameraMatrix @ (r_mat @ pts3d.T + tvec)).T
            projected_pts = projected_pts[:, :2] / projected_pts[:, 2:3]
            projected_pts = projected_pts.reshape(-1, 2)
            errs = np.linalg.norm(projected_pts - pts2d, axis=1)

            # count inliers
            inliers = np.where(errs < err_threshold)[0]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                # best_pose = (rvec, tvec)

    _, rvec, tvec = cv2.solvePnP(pts3d[best_inliers], pts2d[best_inliers], cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    inliers_ratio = len(best_inliers) / len(pts3d) if len(pts3d) > 0 else 0

    return len(best_inliers) >= 4, np.array(rvec), np.array(tvec), np.array(best_inliers), inliers_ratio

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def pnpsolver(query, model):
    kp_query, desc_query = query
    kp_model, desc_model = model
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    # TODO: solve PnP problem using OpenCV
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = matcher.knnMatch(desc_query, desc_model, k=2)
    good_matches = []
    LOWE_RATIO = 0.7
    # Vectorized Lowe's ratio test for speed
    matches = np.array(matches)
    if len(matches) > 0:
        distances = np.array([[m[0].distance, m[1].distance] for m in matches])
        mask = distances[:, 0] < LOWE_RATIO * distances[:, 1]
        good_matches = [m[0] for m, keep in zip(matches, mask) if keep]
    else:
        good_matches = []
    pts2d = np.float32([kp_query[m.queryIdx] for m in good_matches])
    pts3d = np.float32([kp_model[m.trainIdx] for m in good_matches])
    kp_query = pts2d
    # print("Start solving PnP...")
    # start_time = time.time()
    # retval, rvec, tvec, inliers = cv2.solvePnPRansac(pts3d, pts2d, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_P3P)
    retval, rvec, tvec, inliers, inliers_ratio = my_RANSAC_pnp(pts3d, pts2d, cameraMatrix, distCoeffs, err_threshold=2)
    # end_time = time.time()

    # print(f"PnP solved in {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))} with {len(inliers)} inliers")
    # Hint: you may use "Descriptors Matching and ratio test" first
    return retval, rvec, tvec, inliers, inliers_ratio

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
    if t1.dtype == np.dtype('O'):
        t1 = t1.astype(np.float64)
    if t2.dtype == np.dtype('O'):
        t2 = t2.astype(np.float64)
    err = np.linalg.norm(np.abs(t1 - t2), ord=2)
    return err

def visualization(o3d_parameters, points3D_df):
    # === 1. 讀取點雲 ===
    # Vectorized version for speed
    pcd = o3d.geometry.PointCloud()
    xyz = np.vstack(points3D_df["XYZ"].to_numpy())
    rgb = np.vstack(points3D_df["RGB"].to_numpy()) / 255.0
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

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
    o3d.visualization.draw_geometries([pcd, traj, *cameras], up=[0, -1, 0], front=[0,0,-1], zoom=0.4)

def sort_by_image_id(images_df):
    tmp = lambda x: (0 if x.startswith("train") else 1,  # train 放前面，valid 放後面
            int(x.split(".")[0].split("_img")[-1]))  # 按編號排序
    images_df = images_df.sort_values(
        by="NAME",
        key=lambda col: col.apply(tmp)
    )
    return images_df

if __name__ == "__main__":
    # Load data
    PREFIX = "" if os.getcwd().endswith("homework2-xh-cham") else "homework2-xh-cham/"
    images_df = pd.read_pickle(f"{PREFIX}data/images.pkl")
    # sort images_df by the number at the end of NAME column
    # NAME example: train_img1.jpg, train_img100.jpg, valid_img5.jpg
    train_df = pd.read_pickle(f"{PREFIX}data/train.pkl")
    points3D_df = pd.read_pickle(f"{PREFIX}data/points3D.pkl")
    point_desc_df = pd.read_pickle(f"{PREFIX}data/point_desc.pkl")
    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

    images_df = sort_by_image_id(images_df)

    IMAGE_ID_LIST = list(range(163, 293, 15))  # 0-163: train, 163:293 valid
    # IMAGE_ID_LIST = list(range(1,294))
    r_list = []
    t_list = []
    rotation_error_list = []
    translation_error_list = []
    inliers_ratio = 1
    for idx in tqdm(IMAGE_ID_LIST):
        # Load query image
        current_id = (images_df.iloc[idx])["IMAGE_ID"]
        # fname = (images_df.iloc[idx])["NAME"]
        fname = (images_df.loc[images_df["IMAGE_ID"] == current_id])["NAME"].values[0]
        rimg = cv2.imread(f"{PREFIX}data/frames/" + fname, cv2.IMREAD_GRAYSCALE)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == current_id]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp
        retval, rvec, tvec, inliers, inliers_ratio = pnpsolver((kp_query, desc_query), (kp_model, desc_model))
        # tqdm.write(f"Current inliers ratio: {inliers_ratio:.4f}")
        rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat() # Convert rotation vector to quaternion in shape (1,4) [x,y,z,w]
        # tvec = tvec.reshape(1,3) # Reshape translation vector
        r_list.append(rvec)
        t_list.append(tvec)

        # Get camera pose groudtruth
        ground_truth = images_df.iloc[idx]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
        tvec_gt = ground_truth[["TX","TY","TZ"]].values

        # Calculate error
        r_error = rotation_error(rotq_gt, rotq)
        t_error = translation_error(tvec_gt, tvec)
        rotation_error_list.append(r_error)
        translation_error_list.append(t_error)

    # TODO: calculate median of relative rotation angle differences and translation differences and print them
    r_error_median = np.median(rotation_error_list)
    t_error_median = np.median(translation_error_list)
    print(f"Rotation error median (degrees): {r_error_median:.4f}")
    print(f"Translation error median: {t_error_median:.4f}")

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
    # np.save('camera_params.npy', o3d_parameters)
    o3d_parameters_list = [[param.extrinsic, param.intrinsic.intrinsic_matrix] for param in o3d_parameters]
    o3d_parameters_df = pd.DataFrame(o3d_parameters_list, columns=["extrinsic", "intrinsic"])
    pd.to_pickle(o3d_parameters_df, f'{PREFIX}camera_params.pkl')
    visualization(o3d_parameters, points3D_df)