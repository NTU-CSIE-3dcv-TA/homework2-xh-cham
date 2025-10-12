from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import os

class VoxelGridCube:
    def __init__(self, vertices, voxel_size=0.02):
        self.vertices = vertices
        self.voxel_size = voxel_size
        self.pts3d, self.clrs = self.create_cube(vertices)
        self.pts3d = self.get_homogeneous_points()  # (N,4)
        self.orig_pts3d = self.pts3d.copy()
        
    def reset_points_from_df(self):
        self.pts3d = self.orig_pts3d.copy()
        self.pts3d = self.get_homogeneous_points()

    @staticmethod
    def surface_points(v0, v1, v3, v2, n=13):
        """由四個頂點產生面上均勻分布的 nxn 點"""
        us = np.linspace(0, 1, n)
        vs = np.linspace(0, 1, n)
        pts = []
        for u in us:
            for v in vs:
                p = (1-u)*(1-v)*v0 + u*(1-v)*v1 + (1-u)*v*v2 + u*v*v3
                pts.append(p)
        return np.array(pts)

    def create_cube(self, vertices):
        v0, v1, v2, v3, v4, v5, v6, v7 = vertices
        faces = [
            (v0, v1, v3, v2),  # bottom
            (v4, v5, v7, v6),  # top
            (v0, v1, v5, v4),  # front
            (v2, v3, v7, v6),  # back
            (v1, v3, v7, v5),  # right
            (v0, v2, v6, v4)   # left
        ]
        colors = [
            [0.7, 0, 0],
            [0, 0.7, 0],
            [0, 0, 0.7],
            [0.7, 0.7, 0],
            [0, 0.7, 0.7],
            [0.7, 0, 0.7]
        ]
        points, point_colors = [], []
        for (v0, v1, v2, v3), c in zip(faces, colors):
            face_pts = VoxelGridCube.surface_points(v0, v1, v2, v3, n=19)
            points.append(face_pts)
            point_colors.append(np.tile(c, (face_pts.shape[0], 1)))

        points = np.vstack(points)  # (N,3)
        point_colors = np.vstack(point_colors)  # (N,3)
        return points, point_colors
    
    def get_homogeneous_points(self):
        return np.hstack([self.pts3d, np.ones((self.pts3d.shape[0], 1))])  # (N, x) -> (N, x+1)
    
    def dehomogenize(self):
        return self.pts3d[:, :-1] / self.pts3d[:, -1:]
    
    def project(self, K, Rt):
        X_cam = (Rt @ self.pts3d.T).T
        x = (K @ X_cam[:, :3].T).T
        uv = x[:, :2] / x[:, 2:3]
        depth = X_cam[:, 2]
        return uv, depth


    def painter_render(self, uvs, depths, rimg):
        # sort by depth
        indices = np.argsort(depths)[::-1]  # far -> near
        for i in indices:
            u, v = np.round(uvs[i]).astype(int)
            if 0 <= u < rimg.shape[1] and 0 <= v < rimg.shape[0]:
                color = (255 * self.clrs[i]).astype(np.uint8)
                cv2.circle(rimg, (u, v), radius=5, color=tuple(int(c) for c in color.tolist()), thickness=-1)
        return rimg


def sort_by_image_id(images_df):
    tmp = lambda x: (0 if x.startswith("train") else 1,  # train 放前面，valid 放後面
            int(x.split(".")[0].split("_img")[-1]))  # 按編號排序
    images_df = images_df.sort_values(
        by="NAME",
        key=lambda col: col.apply(tmp)
    )
    return images_df

def main():
    # print(VoxelGridCube.surface_points(np.array([0,0,0]), np.array([1,0,0]), np.array([0,1,0]), np.array([1,1,0]), n=5))
    # return
    # Load data
    PREFIX = "" if os.getcwd().endswith("homework2-xh-cham") else "homework2-xh-cham/"
    images_df = pd.read_pickle(f"{PREFIX}data/images.pkl")
    # sort images_df by the number at the end of NAME column
    # NAME example: train_img1.jpg, train_img100.jpg, valid_img5.jpg
    images_df = sort_by_image_id(images_df)
    
    camera_params_df = pd.read_pickle(f"{PREFIX}camera_params.pkl")
    cube_vertices = np.load(f'{PREFIX}cube_vertices.npy')
    # cube_transform_mat = np.load(f'{PREFIX}cube_transform_mat.npy')  # (3,4)

    voxel_grid_cube = VoxelGridCube(cube_vertices)
    
    # set VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fname = (images_df.loc[images_df["IMAGE_ID"] == 1])["NAME"].values[0]
    HEIGHT, WIDTH = cv2.imread(f"{PREFIX}data/frames/" + fname).shape[:2]
    out = cv2.VideoWriter("result.mp4", fourcc, 15, (WIDTH, HEIGHT))

    IMAGE_ID_LIST = list(range(163, 293))
    for i, idx in tqdm(enumerate(IMAGE_ID_LIST), total=len(IMAGE_ID_LIST), desc="Rendering frames", unit="frame"):
        current_id = (images_df.iloc[idx])["IMAGE_ID"]
        fname = (images_df.loc[images_df["IMAGE_ID"] == current_id])["NAME"].values[0]
        raw_img = cv2.imread(f"{PREFIX}data/frames/" + fname)
        # raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        extrinsic_mat = camera_params_df.iloc[i]["extrinsic"]  # (4,4)
        intrinsic_mat = camera_params_df.iloc[i]["intrinsic"]  # (3,3)

        uv, depth = voxel_grid_cube.project(intrinsic_mat, extrinsic_mat)

        # Render the voxel grid
        assert len(voxel_grid_cube.clrs) == len(voxel_grid_cube.pts3d) == len(uv) == len(depth)
        rendered_img = voxel_grid_cube.painter_render(uv, depth, raw_img)

        # cv2.waitKey(70)  # wait 70 ms before next frame
        out.write(rendered_img)
        
    out.release()
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()