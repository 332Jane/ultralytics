"""
[DEPRECATED]direct_verify_mapping.py
Purpose:
  - Debug script to verify output→pixel coordinate mapping
  - Test matrix M = H_inv @ A calculation
  - Check if output pixels stay within image bounds
  - Diagnose black region issues

Status:
  - Standalone debug tool only
  - No active code uses it
  - Manual verification only when needed

Reason for removal:
  - Not essential to pipeline
  - Debug/diagnostic tool
  - Can be recreated if needed for future debugging

Directly verify if the mapping from output coordinates to pixel coordinates is correct
"""

import cv2
import numpy as np
import json

def load_homography(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    H = np.array(data['homography_matrix'], dtype=np.float32)
    return H, data['pixel_points'], data['world_points']

# Load data
H, pixel_points, world_points = load_homography('../../calibration/Homograph_Teset_FullScreen_homography.json')

print("=" * 70)
print("Directly Verify Mapping Relationship")
print("=" * 70)

# World coordinate range
min_x = -3.75
max_x = 3.75
min_y = 0
max_y = 50

out_w = 180
out_h = 1200

print(f"\n[Settings]")
print(f"Output image: {out_w}x{out_h}")
print(f"World coordinate range: X[{min_x}, {max_x}], Y[{min_y}, {max_y}]")

# Construct A matrix (output → world)
world_width = max_x - min_x  # 7.5
world_height = max_y - min_y  # 50

A = np.array([
    [world_width / out_w, 0, min_x],
    [0, -world_height / out_h, max_y],
    [0, 0, 1]
], dtype=np.float32)

print(f"\n[Matrix A: Output → World]")
print(A)

# H_inv (world → pixel)
H_inv = np.linalg.inv(H)

print(f"\n[Matrix H_inv: World → Pixel]")
print(H_inv)

# M (output → pixel)
M = H_inv @ A

print(f"\n[Matrix M = H_inv @ A: Output → Pixel]")
print(M)

# Now directly test some output points
print(f"\n" + "=" * 70)
print("[Direct Mapping Test]")
print("=" * 70)

test_points = [
    (0, 0, "top-left"),
    (180, 0, "top-right"),
    (0, 1200, "bottom-left"),
    (180, 1200, "bottom-right"),
    (90, 600, "center"),
]

for out_x, out_y, label in test_points:
    print(f"\nOutput coordinates ({out_x}, {out_y}) - {label}")
    
    # Method 1: Use matrix M to map directly
    out_homo = np.array([out_x, out_y, 1], dtype=np.float32)
    pixel_homo = M @ out_homo
    pixel_x = pixel_homo[0] / pixel_homo[2]
    pixel_y = pixel_homo[1] / pixel_homo[2]
    
    print(f"  → Pixel coordinates: ({pixel_x:.2f}, {pixel_y:.2f})")
    
    # Verify: Is this pixel coordinate within original image bounds
    if 0 <= pixel_x < 1080 and 0 <= pixel_y < 1920:
        print(f"  ✓ Within original image bounds")
    else:
        print(f"  ❌ Outside original image bounds! (original image 1080x1920)")
    
    # Verify: What world coordinates does this pixel coordinate correspond to
    pixel_homo2 = np.array([pixel_x, pixel_y, 1], dtype=np.float32)
    world_homo = H @ pixel_homo2
    world_x = world_homo[0] / world_homo[2]
    world_y = world_homo[1] / world_homo[2]
    
    print(f"  → World coordinates: ({world_x:.4f}, {world_y:.4f})")
    
    # Compute expected world coordinates for this output point
    expected_world_x = min_x + (out_x / out_w) * world_width
    expected_world_y = max_y - (out_y / out_h) * world_height
    
    print(f"  Expected world coordinates: ({expected_world_x:.4f}, {expected_world_y:.4f})")
    
    error = np.sqrt((world_x - expected_world_x)**2 + (world_y - expected_world_y)**2)
    print(f"  Error: {error:.6f} {'✓' if error < 0.1 else '❌'}")

# Key issue: Check if all output pixels map to valid range
print(f"\n" + "=" * 70)
print("[Key Diagnostic: Output Pixel Range Analysis]")
print("=" * 70)

# Check pixel coordinates of output image four corners
corners = [(0, 0), (out_w, 0), (0, out_h), (out_w, out_h)]
pixel_coords = []

print(f"\nOutput image's 4 corner points mapped to pixel coordinates:")
for out_x, out_y in corners:
    out_homo = np.array([out_x, out_y, 1], dtype=np.float32)
    pixel_homo = M @ out_homo
    px = pixel_homo[0] / pixel_homo[2]
    py = pixel_homo[1] / pixel_homo[2]
    pixel_coords.append((px, py))
    print(f"  ({out_x:3d}, {out_y:4d}) → Pixel({px:7.2f}, {py:7.2f})")

# Analyze pixel coordinate range
px_coords = [p[0] for p in pixel_coords]
py_coords = [p[1] for p in pixel_coords]

print(f"\nPixel X range: {min(px_coords):.2f} ~ {max(px_coords):.2f}")
print(f"Pixel Y range: {min(py_coords):.2f} ~ {max(py_coords):.2f}")
print(f"Original image range: 0 ~ 1080 (X), 0 ~ 1920 (Y)")

if min(px_coords) >= 0 and max(px_coords) <= 1080 and min(py_coords) >= 0 and max(py_coords) <= 1920:
    print(f"✓ All output pixels are within original image bounds")
else:
    print(f"❌ Some output pixels exceed original image bounds!")
    print(f"   This may cause black regions!")
