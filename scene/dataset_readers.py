# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from scene.gaussian_model import BasicPointCloud
import glob
import torch

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    R_gt: np.array
    T_gt: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = torch.cat(cam_centers, dim=1)
        avg_cam_center = torch.mean(cam_centers, dim=1, keepdim=True)
        center = avg_cam_center
        dist = torch.norm(cam_centers - center, dim=0, keepdim=True)
        diagonal = torch.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View(cam.R, cam.T)
        C2W = torch.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    translate = translate.cpu().numpy()
    radius = radius.cpu().numpy()

    return {"translate": translate, "radius": radius}

## TODO: Add support for reading images from a json file ##
def readJsonCameras(cameras, cameras_gt, images_folder):
    cam_infos = []
    for idx, cam in enumerate(cameras):
        sys.stdout.write('\rReading camera {}/{}'.format(idx+1, len(cameras)))
        sys.stdout.flush()

        R_gt = None
        T_gt = None
        
        # Only try to get GT poses if cameras_gt is available
        if cameras_gt is not None:
            for key in cameras_gt:
                cam_gt = cameras_gt[key]
                if cam["image_name"] == cam_gt.name.split(".")[0]:
                    R_gt = np.transpose(qvec2rotmat(cam_gt.qvec))
                    T_gt = np.array(cam_gt.tvec)
                    break
        
        R = np.array(cam["R"])
        T = np.array(cam["T"])
        FovY = focal2fov(cam["Focaly"], cam["height"])
        FovX = focal2fov(cam["Focalx"], cam["width"])

        pattern = os.path.join(images_folder, cam["image_name"] + ".*")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"Image file not found for {cam['image_name']}")
        image_path = files[0]

        image = Image.open(image_path)

        cam_info = CameraInfo(uid=idx, R=R, T=T, R_gt=R_gt, T_gt=T_gt,
                               FovY=FovY, FovX=FovX, image=image,
                               image_path=image_path, image_name=cam["image_name"],
                               width=image.size[0], height=image.size[1])
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readUnposedCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            print(f"WARNING: Colmap camera model not handled: {intr.model}")
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=idx, R=None, T=None, R_gt=R, T_gt=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readUnposedCameras2(images_folder):
    cam_infos = []
    image_names = os.listdir(images_folder)
    image_names = sorted(image_names)
    for image_name in image_names:
        image_path = os.path.join(images_folder, image_name)
        image = Image.open(image_path)
        width, height = image.size
        image_name = image_name.split(".")[0]
        cam_info = CameraInfo(uid=len(cam_infos), R=None, T=None, R_gt=None, T_gt=None, FovY=None, FovX=None, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            # if 'university1' in image_path:
                # image = image.rotate(90, expand=True)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readEvalSceneInfo(path, model_path, images, eval, llffhold=8):
    if "co3d" not in path:
        cam_extrinsics = None
        cam_intrinsics = None
        
        # Try to read GT poses, but don't fail if they're not available
        try:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
            if os.path.exists(cameras_extrinsic_file) and os.path.exists(cameras_intrinsic_file):
                cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
                cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
                print(f"Successfully loaded GT poses from {cameras_extrinsic_file}")
            else:
                print(f"GT pose files not found at {cameras_extrinsic_file}, running in custom mode without GT poses")
        except Exception as e:
            try:
                cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
                cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
                if os.path.exists(cameras_extrinsic_file) and os.path.exists(cameras_intrinsic_file):
                    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
                    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
                    print(f"Successfully loaded GT poses from {cameras_extrinsic_file}")
                else:
                    print(f"GT pose files not found at {cameras_extrinsic_file}, running in custom mode without GT poses")
            except Exception as e2:
                print(f"Failed to load GT poses: {e2}. Running in custom mode without GT poses.")

        reading_dir = "images" if images == None else images
        if eval:
            camera_train_path = os.path.join(model_path, "cameras_all_train.json")

            cam_infos_train_unsorted = readJsonCameras(json.load(open(camera_train_path)), cam_extrinsics, os.path.join(path, reading_dir))
            cam_infos_train = sorted(cam_infos_train_unsorted.copy(), key = lambda x : x.image_name)

            camera_test_path = os.path.join(model_path, "cameras_all_test.json")
            cam_infos_test_unsorted = readJsonCameras(json.load(open(camera_test_path)), cam_extrinsics, os.path.join(path, reading_dir))
            cam_infos_test = sorted(cam_infos_test_unsorted.copy(), key = lambda x : x.image_name)
        else:
            camera_train_path = os.path.join(model_path, "cameras_all.json")
            cam_infos_train_unsorted = readJsonCameras(json.load(open(camera_train_path)), cam_extrinsics, os.path.join(path, reading_dir))
            cam_infos_train = sorted(cam_infos_train_unsorted.copy(), key = lambda x : x.image_name)
            cam_infos_test = []

        train_cam_infos = cam_infos_train
        test_cam_infos = cam_infos_test

        nerf_normalization = None

        # Only try to load point cloud if GT poses are available
        pcd = None
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        if cam_extrinsics is not None:
            bin_path = os.path.join(path, "sparse/0/points3D.bin")
            txt_path = os.path.join(path, "sparse/0/points3D.txt")
            if not os.path.exists(ply_path):
                print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
                try:
                    xyz, rgb, _ = read_points3D_binary(bin_path)
                except:
                    try:
                        xyz, rgb, _ = read_points3D_text(txt_path)
                    except:
                        print("Failed to load point cloud data")
                        xyz, rgb = None, None
                if xyz is not None:
                    storePly(ply_path, xyz, rgb)
            try:
                pcd = fetchPly(ply_path)
            except:
                pcd = None
        else:
            print("Skipping point cloud loading as GT poses are not available")

        scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=train_cam_infos,
                            test_cameras=test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path)
    else:
        sequences = defaultdict(list)
        dataset = json.loads(gzip.GzipFile(os.path.join(path, "frame_annotations.jgz"), "rb").read().decode("utf8"))

        for data in dataset:
            sequences[data["sequence_name"]].append(data)
        sequence_name = os.path.basename(path)
        root_path = '/project2/localGS/co3d/'
        cam_infos_unsorted = readCo3dCameras(sequences[sequence_name], root_path)
        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)


        if eval:
            sample_rate = 8
            ids = np.arange(len(cam_infos))
            i_test = ids[int(sample_rate/2)::sample_rate]
            i_train = np.array([i for i in ids if i not in i_test])
            train_cam_infos = [cam_infos[i] for i in i_train]
            test_cam_infos = [cam_infos[i] for i in i_test]
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []
        
        camera_train_path = os.path.join(model_path, "cameras_all_train.json")
        pred_cameras = json.load(open(camera_train_path))
        new_train_cam_infos = []
        for cam_info in train_cam_infos:
            for pred_cam in pred_cameras:
                if cam_info.image_name == pred_cam["image_name"]:
                    # Create a new CameraInfo with updated R and T
                    new_cam_info = CameraInfo(
                        uid=cam_info.uid,
                        R=pred_cam["R"],
                        T=pred_cam["T"],
                        R_gt=cam_info.R_gt,
                        T_gt=cam_info.T_gt,
                        FovY=cam_info.FovY,
                        FovX=cam_info.FovX,
                        image=cam_info.image,
                        image_path=cam_info.image_path,
                        image_name=cam_info.image_name,
                        width=cam_info.width,
                        height=cam_info.height
                    )
                    new_train_cam_infos.append(new_cam_info)
                    break
            else:
                new_train_cam_infos.append(cam_info)

        nerf_normalization = None
        scene_info = SceneInfo(point_cloud=None,
                            train_cameras=new_train_cam_infos,
                            test_cameras=test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=None)
    return scene_info


def readFreeSceneInfo(path, images, eval, llffhold=8):
    if os.path.exists(os.path.join(path, "sparse")):
        try:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        reading_dir = "images" if images == None else images
        cam_infos_unsorted = readUnposedCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    else:
        print("No sparse folder found in the path, please check the path")
        return None
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = None

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)

    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readTanksSceneInfo(path, images, eval, llffhold=8):
    if os.path.exists(os.path.join(path, "sparse")):
        try:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        reading_dir = "images" if images == None else images
        cam_infos_unsorted = readUnposedCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    else:
        print("No sparse folder found in the path, please check the path")
        return None
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        llffhold = 2 if "Family" in path else 8
        ids = np.arange(len(cam_infos))
        i_test = ids[int(llffhold/2)::llffhold]
        i_train = np.array([i for i in ids if i not in i_test])
        train_cam_infos = [cam_infos[i] for i in i_train]
        test_cam_infos = [cam_infos[i] for i in i_test]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = None

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)

    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readHikeSceneInfo(path, images, eval, llffhold=10):
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readUnposedCameras2(images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = None

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)

    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCustomSceneInfo(path, images, eval, llffhold=8):
    # Check if sparse folder exists and has camera data
    has_camera_data = False
    cam_extrinsics = None
    cam_intrinsics = None
    
    if os.path.exists(os.path.join(path, "sparse")):
        try:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
            has_camera_data = True
        except:
            try:
                cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
                cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
                cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
                cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
                has_camera_data = True
            except:
                print("No valid camera data found in sparse folder, proceeding without camera poses")
                has_camera_data = False
    
    reading_dir = "images" if images == None else images
    
    # Load cameras based on available data
    if has_camera_data:
        cam_infos_unsorted = readUnposedCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    else:
        cam_infos_unsorted = readUnposedCameras2(images_folder=os.path.join(path, reading_dir))
    
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = None

    # Handle point cloud data - only try to load if sparse folder exists
    pcd = None
    ply_path = None
    
    if os.path.exists(os.path.join(path, "sparse")):
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
                storePly(ply_path, xyz, rgb)
            except:
                try:
                    xyz, rgb, _ = read_points3D_text(txt_path)
                    storePly(ply_path, xyz, rgb)
                except:
                    print("No valid point cloud data found, proceeding without point cloud")
                    ply_path = None
        
        if ply_path and os.path.exists(ply_path):
            try:
                pcd = fetchPly(ply_path)
            except:
                print("Failed to load point cloud from PLY file")
                pcd = None
                ply_path = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

import gzip
from collections import defaultdict, OrderedDict
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.utils import opencv_from_cameras_projection
import torch

def load_camera(data, scale=1.0):
    """
    Load a camera from a CO3D annotation.
    """

    principal_point = torch.tensor(
        data["viewpoint"]["principal_point"], dtype=torch.float)
    focal_length = torch.tensor(
        data["viewpoint"]["focal_length"], dtype=torch.float)
    half_image_size_wh_orig = (
        torch.tensor(
            list(reversed(data["image"]["size"])), dtype=torch.float) / 2.0
    )
    format_ = data["viewpoint"]["intrinsics_format"]
    if format_.lower() == "ndc_norm_image_bounds":
        # this is e.g. currently used in CO3D for storing intrinsics
        rescale = half_image_size_wh_orig
    elif format_.lower() == "ndc_isotropic":
        rescale = half_image_size_wh_orig.min()
    else:
        raise ValueError(f"Unknown intrinsics format: {format}")

    principal_point_px = half_image_size_wh_orig - principal_point * rescale
    focal_length_px = focal_length * rescale

    # now, convert from pixels to PyTorch3D v0.5+ NDC convention
    out_size = list(reversed(data["image"]["size"]))

    half_image_size_output = torch.tensor(
        out_size, dtype=torch.float) / 2.0
    half_min_image_size_output = half_image_size_output.min()

    # rescaled principal point and focal length in ndc
    principal_point = (
        half_image_size_output - principal_point_px * scale
    ) / half_min_image_size_output
    focal_length = focal_length_px * scale / half_min_image_size_output

    camera = PerspectiveCameras(
        focal_length=focal_length[None],
        principal_point=principal_point[None],
        R=torch.tensor(data["viewpoint"]["R"], dtype=torch.float)[None],
        T=torch.tensor(data["viewpoint"]["T"], dtype=torch.float)[None],
    )

    img_size = torch.tensor(data["image"]["size"], dtype=torch.float)[None]
    R, t, intr_mat = opencv_from_cameras_projection(camera, img_size)
    FoVy = focal2fov(intr_mat[0, 1, 1], img_size[0, 0])
    FoVx = focal2fov(intr_mat[0, 0, 0], img_size[0, 1])

    return R[0].numpy().T, t[0].numpy(), FoVx, FoVy, intr_mat[0].numpy()

def readCo3dCameras(data, path):
    cam_infos = []
    for idx, d in enumerate(data):
        
        cam_info = dict()
        image_name = data[idx]["image"]["path"]
        image_path = os.path.join(path, image_name)
        image_name = os.path.basename(image_path).split(".")[0]

        R, t, FoVx, FoVy, intr_mat = load_camera(data[idx])
        
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        cam_info = CameraInfo(uid=idx, R=None, T=None, R_gt=R, T_gt=t, FovY=FoVy, FovX=FoVx, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    return cam_infos

def readCo3dSceneInfo(path, images, eval, llffhold=8):
    sequences = defaultdict(list)
    dataset = json.loads(gzip.GzipFile(os.path.join(path, "frame_annotations.jgz"), "rb").read().decode("utf8"))

    for data in dataset:
        sequences[data["sequence_name"]].append(data)
    sequence_name = os.path.basename(path)
    root_path = '/project2/localGS/co3d/'
    cam_infos_unsorted = readCo3dCameras(sequences[sequence_name], root_path)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        sample_rate = 8
        ids = np.arange(len(cam_infos))
        i_test = ids[int(sample_rate/2)::sample_rate]
        i_train = np.array([i for i in ids if i not in i_test])
        train_cam_infos = [cam_infos[i] for i in i_train]
        test_cam_infos = [cam_infos[i] for i in i_test]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = None
    scene_info = SceneInfo(point_cloud=None,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=None)
    return scene_info

sceneLoadTypeCallbacks = {
    "Eval": readEvalSceneInfo,
    "Custom" : readCustomSceneInfo,
    "Free" : readFreeSceneInfo,
    "Tanks" : readTanksSceneInfo,
    "Hike" : readHikeSceneInfo,
    "Co3d" : readCo3dSceneInfo,
}