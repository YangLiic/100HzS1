import os
from torchvision.utils import save_image
from tqdm import tqdm
import cv2
import torch
import numpy as np

def save_depth(depth, path):
    # depth: [B,H,W,C]
    depth_norm = ((depth-depth.min())/(depth.max()-depth.min())).permute(0,3,1,2)
    save_image(depth_norm, path)

def save_images(images, path):
    # images: [B,H,W,C]
    save_image(images, path)

def process_and_save(point_cloud, rgb_image, filename, original_height):
    """
    Process point cloud and RGB images, concatenate and save as a point cloud file.
    Args:
        point_cloud: [V, H, W, 3] point cloud tensor
        rgb_image: [V, 3, H, W] RGB image tensor
        filename: output file name
        original_height: original height H
    """
    import open3d as o3d
    # Get tensor shapes
    V, H, W, C = point_cloud.shape
    _, C_img, H_img, W_img = rgb_image.shape

    # Validate input shape consistency
    assert H == H_img and W == W_img, "point cloud and image spatial dims do not match"
    assert C == 3 and C_img == 3, "channel count must be 3"

    # 1. Point cloud concat: [V, H, W, 3] -> [H, V*W, 3]
    point_cloud_reshaped = point_cloud.permute(1, 0, 2, 3).reshape(H, V*W, 3)

    # 2. Image concat: [V, 3, H, W] -> [3, H, V*W]
    image_reshaped = rgb_image.permute(0, 2, 3, 1)  # [V, H, W, 3]
    image_reshaped = image_reshaped.permute(1, 0, 2, 3).reshape(H, V*W, 3)  # [H, V*W, 3]
    image_reshaped = image_reshaped.permute(2, 0, 1)  # [3, H, V*W]

    # 3. Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        point_cloud_reshaped.reshape(-1, 3).cpu().float().numpy()
    )
    pcd.colors = o3d.utility.Vector3dVector(
        image_reshaped.permute(1, 2, 0).reshape(-1, 3).cpu().float().numpy()
    )

    # 4. Save point cloud file and store the original height
    o3d.io.write_point_cloud(filename, pcd)
    with open(filename + ".meta", "w") as f:
        f.write(f"{original_height}")

    return point_cloud_reshaped, image_reshaped

def load_and_restore(filename):
    """
    Load a point cloud file and restore the original tensors.
    Args:
        filename: point cloud file name
    Returns:
        point_cloud_tensor: [H, V*W, 3] point cloud tensor
        rgb_tensor: [3, H, V*W] RGB tensor
    """
    import open3d as o3d
    # 1. Load point cloud and metadata
    pcd = o3d.io.read_point_cloud(filename)
    with open(filename + ".meta", "r") as f:
        original_height = int(f.readline().strip())

    # 2. Convert to numpy arrays
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # 3. Compute the second-dim size
    total_points = points.shape[0]
    VW = total_points // original_height

    # 4. Restore point cloud tensor: [H, V*W, 3]
    point_cloud_tensor = torch.tensor(points).reshape(original_height, VW, 3)

    # 5. Restore RGB tensor: [3, H, V*W]
    colors_reshaped = torch.tensor(colors).reshape(original_height, VW, 3)
    rgb_tensor = colors_reshaped.permute(2, 0, 1)

    return point_cloud_tensor, rgb_tensor

def render_point_cloud_video(
    point_cloud,       # [S, H, W, 3] torch.Tensor
    rgb_images,        # [S, 3, H, W] torch.Tensor
    output_path='output.mp4',
    fps=10,
    camera_params=(1.0, 0.0, 0.0),  # (r, theta, phi)
):
    """
    Render a point cloud sequence as a video.

    Args:
        point_cloud: point cloud sequence [S, H, W, 3]
        rgb_images: matching RGB images [S, 3, H, W] (values in 0-1)
        output_path: output video path
        fps: video frame rate
        camera_params: camera parameter tuple (r, theta, phi)
    """
    from pytorch3d.structures import Pointclouds
    from pytorch3d.renderer import (
        look_at_rotation,
        FoVPerspectiveCameras,
        PointsRasterizationSettings,
        PointsRasterizer,
        PointsRenderer,
        AlphaCompositor
    )
    # Check device
    device = point_cloud.device
    S, H, W, _ = point_cloud.shape

    # Use the first frame's point cloud center as the scene center
    center = point_cloud[0].reshape(-1, 3).mean(dim=0, keepdim=True)  # [1, 3]

    # Translate the point cloud center to the origin
    point_cloud_centered = point_cloud - center

    # Extract camera parameters
    r, theta, phi = camera_params


    def get_RT(r, theta, phi):
        # Convert spherical coordinates to Cartesian
        theta = torch.deg2rad(torch.tensor(theta))
        phi = torch.deg2rad(torch.tensor(phi))
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
        camera_position = torch.tensor([[x, y, z]], device=device)  # [1, 3]

        # Camera rotation matrix (looking at the origin)
        R = look_at_rotation(camera_position, device=device)  # [1, 3, 3]
        T = -torch.bmm(R.transpose(1, 2), camera_position[:, :, None])[:, :, 0]  # [1, 3]
        return R, T

    R, T = get_RT(r, theta, phi)
    # Create camera object
    cameras = FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device,
        znear=0.01 * r,
        zfar=10 * r,
        fov=60  # field of view (adjustable)
    )

    # Point cloud renderer settings
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=0.008,  # point radius (tune to the point cloud distribution)
        points_per_pixel=10,  # points considered per pixel
        bin_size=None if H <= 256 else 0  # optimization for large sizes
    )

    # Create renderer
    rasterizer = PointsRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (W, H)
    )

    # Render point cloud frame by frame
    for i in tqdm(range(S)):
        # R, T = get_RT(r, i*(36/S), phi)
        # cameras.R = R.to(device)
        # cameras.T = T.to(device)

        # Prepare current frame's points and colors
        points = point_cloud_centered[i].reshape(1, -1, 3)  # [1, H*W, 3]
        colors = rgb_images[i].permute(1, 2, 0).reshape(1, -1, 3)  # [1, H*W, 3]

        # Create point cloud object
        pcl = Pointclouds(points=points, features=colors)

        # Render point cloud (PyTorch3D output: [1, H, W, 4] RGBA)
        rendered_image = renderer(pcl)[0]  # [H, W, 4]

        # Extract RGB channels and convert to OpenCV format
        rgb = rendered_image[..., :3].cpu().numpy()  # [H, W, 3]
        bgr = (rgb * 255).astype(np.uint8)[..., ::-1]  # convert to BGR

        # Write video frame
        video_writer.write(bgr)

    # Release video resources
    video_writer.release()
    print(f"Video saved to: {output_path}")

def reproject_point_cloud(
    point_cloud,       # [V, H, W, 3] point cloud in camera-1 frame (x, y, z)
    cam1_ext,          # [V, 3, 4] camera-1 extrinsics (R|t)
    cam1_int,          # [V, 3, 3] camera-1 intrinsics
    cam2_ext,          # [V, 3, 4] camera-2 extrinsics
    cam2_int           # [V, 3, 3] camera-2 intrinsics
):
    V, H, W, _ = point_cloud.shape
    device = point_cloud.device

    # 1. Transform point cloud from camera-1 world frame -> camera-1 camera frame
    # Extend to homogeneous coordinates [V, H*W, 4]
    points_hom = torch.cat([point_cloud, torch.ones(V, H, W, 1, device=device)], dim=-1).reshape(V, H * W, 4)
    # Extend extrinsics to homogeneous [V, 4, 4]: pad a zero column, last row [0,0,0,1]
    extr_hom = torch.cat([cam1_ext, torch.tensor([0,0,0,1], device=cam1_ext.device).repeat(V,1,1)], dim=1)
    # Inverse transform: world frame -> camera frame
    extr_inv = torch.inverse(extr_hom)  # shape [V, 4, 4]
    # Apply transform: cam_coords = T_world_to_cam * world_coords
    cam_points_hom = torch.einsum("vij,vkj->vki", extr_inv, points_hom)  # shape [V, H*W, 4]
    # Back to non-homogeneous coordinates, reshaped to the original shape
    cam_points = cam_points_hom[..., :3] / cam_points_hom[..., 3:4]  # shape [V, H*W, 3]

    point_cloud2 = []
    # Extract rotation R and translation t from the extrinsics
    for v in range(len(cam2_ext)):
        points = cam_points[v]
        points = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=1)
        point_cloud2.append(torch.mm(points, cam2_ext[v].T))
    point_cloud2 = torch.stack(point_cloud2).view(V, H, W, 3)

    # Restore original shape [V, H, W, 3]
    return point_cloud2
