"""
[DEPRECATED] correct_perspective_transform.py

Remove deprecated correct_perspective_transform.py

Purpose (removed):
  - Mathematical reference for perspective transform matrix derivation
  - Verification script for homography transformation accuracy
  - Educational material for coordinate system conversion

Reasons for removal:
  1. Unused code - no active imports or dependencies
  2. Standalone script only - cannot be integrated into pipeline
  3. Duplicate functionality - homography_transform_utils.py already implements the logic
  4. Maintenance burden - reduces project complexity

The actual transformation logic remains in:
  - homography_transform_utils.py (compute_transformation_matrix, transform_frame_manual)
  - collision_detection_pipeline_yolo_first_method_a.py (Method A)

Correct perspective transformation logic
"""

import cv2
import numpy as np
import json
import os

def load_homography(json_path):
    """Load Homography matrix"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    H = np.array(data['homography_matrix'], dtype=np.float32)
    return H, data['pixel_points'], data['world_points']

def compute_output_to_pixel_matrix(H, world_bounds, output_size):
    """
    Compute mapping matrix from output coordinates to pixel coordinates
    
    Parameters:
    - H: Matrix for pixel coordinates → world coordinates
    - world_bounds: (min_x, max_x, min_y, max_y) World coordinate range
    - output_size: (width, height) Output image size
    
    Returns:
    - M: Matrix from output coordinates to pixel coordinates, used for warpPerspective
    """
    
    min_x, max_x, min_y, max_y = world_bounds
    out_w, out_h = output_size
    
    # Step 1: Construct mapping matrix A from output coordinates → world coordinates
    # Definition: output image (0, 0) corresponds to world coordinates (min_x, max_y)
    #            output image (out_w, 0) corresponds to world coordinates (max_x, max_y)
    #            output image (0, out_h) corresponds to world coordinates (min_x, min_y)
    #            output image (out_w, out_h) corresponds to world coordinates (max_x, min_y)
    
    # Linear mapping:
    # world_x = min_x + (out_x / out_w) * (max_x - min_x)
    # world_y = max_y - (out_y / out_h) * (max_y - min_y)
    
    # Matrix form: [world_x, world_y, 1]^T = A @ [out_x, out_y, 1]^T
    
    world_width = max_x - min_x
    world_height = max_y - min_y
    
    A = np.array([
        [world_width / out_w, 0, min_x],           # world_x = (world_width/out_w)*out_x + min_x
        [0, -world_height / out_h, max_y],         # world_y = -(world_height/out_h)*out_y + max_y
        [0, 0, 1]
    ], dtype=np.float32)
    
    print("\n[Output Coordinates → World Coordinates Mapping Matrix A]")
    print(f"world_x = ({world_width/out_w:.6f}) * out_x + {min_x}")
    print(f"world_y = ({-world_height/out_h:.6f}) * out_y + {max_y}")
    print(A)
    
    # Step 2: Calculate inverse of H matrix (world coordinates → pixel coordinates)
    H_inv = np.linalg.inv(H)
    
    print("\n[Inverse of H Matrix H_inv (World Coordinates → Pixel Coordinates)]")
    print(H_inv)
    
    # Step 3: Combine matrices M = H_inv @ A
    # This way: pixel coordinates = M @ output coordinates
    M = H_inv @ A
    
    print("\n[Final Matrix M = H_inv @ A (Output Coordinates → Pixel Coordinates)]")
    print("This is the matrix needed by warpPerspective")
    print(M)
    
    return M

def test_matrix(M, world_bounds, output_size, H):
    """验证矩阵是否正确"""
    print("\n" + "=" * 60)
    print("【矩阵验证】")
    print("=" * 60)
    
    min_x, max_x, min_y, max_y = world_bounds
    out_w, out_h = output_size
    
    # 测试四个角点
    test_points = [
        (0, 0, "左上"),
        (out_w, 0, "右上"),
        (0, out_h, "左下"),
        (out_w, out_h, "右下")
    ]
    
    expected_world = [
        (min_x, max_y, "左上"),
        (max_x, max_y, "右上"),
        (min_x, min_y, "左下"),
        (max_x, min_y, "右下")
    ]
    
    for (out_x, out_y, label), (exp_x, exp_y, _) in zip(test_points, expected_world):
        # 输出坐标 → 像素坐标
        out_homo = np.array([out_x, out_y, 1], dtype=np.float32)
        pixel_homo = M @ out_homo
        pixel_x = pixel_homo[0] / pixel_homo[2]
        pixel_y = pixel_homo[1] / pixel_homo[2]
        
        # 验证：像素坐标 → 世界坐标
        pixel_homo2 = np.array([pixel_x, pixel_y, 1], dtype=np.float32)
        world_homo = H @ pixel_homo2
        world_x = world_homo[0] / world_homo[2]
        world_y = world_homo[1] / world_homo[2]
        
        print(f"\n{label} (输出坐标 {out_x}, {out_y}):")
        print(f"  → 像素坐标 ({pixel_x:.2f}, {pixel_y:.2f})")
        print(f"  → 世界坐标 ({world_x:.2f}, {world_y:.2f})")
        print(f"  期望世界坐标 ({exp_x:.2f}, {exp_y:.2f})")
        error = np.sqrt((world_x - exp_x)**2 + (world_y - exp_y)**2)
        print(f"  误差: {error:.6f} {'✓' if error < 0.1 else '❌'}")

# Main program
video_path = '../../videos/Homograph_Teset_FullScreen.mp4'
homography_path = '../../calibration/Homograph_Teset_FullScreen_homography.json'

print("=" * 60)
print("Correct Perspective Transform Logic Derivation")
print("=" * 60)

H, pixel_points, world_points = load_homography(homography_path)

print("\n[Input Data]")
print("H Matrix (pixel → world):")
print(H)
print("\nReference points:")
for i, (p, w) in enumerate(zip(pixel_points, world_points)):
    print(f"  Point {i+1}: pixel {p} → world {w}")

# World coordinate range
min_x = min(w[0] for w in world_points)
max_x = max(w[0] for w in world_points)
min_y = min(w[1] for w in world_points)
max_y = max(w[1] for w in world_points)

world_bounds = (min_x, max_x, min_y, max_y)
output_size = (180, 1200)

print(f"\n[World Coordinate Range]")
print(f"X: {min_x} ~ {max_x}")
print(f"Y: {min_y} ~ {max_y}")

# Compute mapping matrix
M = compute_output_to_pixel_matrix(H, world_bounds, output_size)

# Verify matrix
test_matrix(M, world_bounds, output_size, H)
