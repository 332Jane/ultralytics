"""
verify_homography.py

================================================================================
HOMOGRAPHY TRANSFORMATION VERIFICATION TOOL
================================================================================

PURPOSE:
  This script generates a visual verification image to validate that the
  Homography calibration is correct by comparing:
  1. Original video frame with reference points marked (green circles)
  2. Transformed bird's-eye view with reference points marked (blue circles)

FUNCTIONALITY:
  - Loads video first frame and Homography matrix from calibration JSON
  - Marks 4 reference points in original frame (green circles with numbers)
  - Applies perspective transformation to generate bird's-eye view
  - Marks corresponding reference points in warped frame (blue circles)
  - Generates side-by-side comparison image for visual inspection
  - Prints detailed statistics and verification checklist

USAGE:
  Basic usage:
    python verify_homography.py \
      --video ../../videos/Homograph_Teset_FullScreen.mp4 \
      --homography ../../calibration/Homograph_Teset_FullScreen_homography.json

  Custom output path:
    python verify_homography.py \
      --video video.mp4 \
      --homography calibration.json \
      --output ./my_verification.jpg

PARAMETERS:
  --video <path>           : Input video file path (REQUIRED)
  --homography <path>      : Homography calibration JSON file (REQUIRED)
  --output <path>          : Output verification image path
                             (Default: verify_homography.jpg)

OUTPUT:
  - Verification image showing:
    * Left side: Original frame with green reference points (pixel coords)
    * Right side: Warped bird's-eye view with blue reference points (world coords)
  - Console output with:
    * Reference point coordinates (pixel → world)
    * World coordinate ranges (X, Y in meters)
    * Transformation verification checklist

VERIFICATION CHECKLIST:
  ✓ Load video and calibration successfully
  ✓ Reference points marked clearly (green in original, blue in warped)
  ✓ Blue points form a regular rectangle (indicates correct calibration)
  ✓ No extreme distortion in warped view
  ✓ Coordinate ranges match expected scene (road width, viewing distance)

If any issues detected:
  - Check pixel point accuracy in calibration JSON
  - Verify world point coordinates are correct
  - Recalibrate using more precise reference points
  - Check if video first frame is clear and undistorted

TECHNICAL DETAILS:
  - Uses OpenCV's warpPerspective for transformation
  - Calculates H_inv (inverse Homography matrix) to verify transformation
  - Generates output at 100 pixels/meter scale for visualization

================================================================================
"""

import cv2
import numpy as np
import json
import argparse
import os


def load_homography(json_path):
    """Load Homography matrix and reference points from calibration JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    H = np.array(data['homography_matrix'], dtype=np.float32)
    pixel_points = data['pixel_points']
    world_points = data['world_points']
    return H, pixel_points, world_points


def verify_homography(video_path, homography_path, output_path):
    """
    Generate a verification image showing the Homography transformation
    
    Parameters:
      video_path: Path to input video file
      homography_path: Path to Homography calibration JSON
      output_path: Path where verification image will be saved
    """
    
    # Load Homography matrix and reference points
    H, pixel_points, world_points = load_homography(homography_path)
    
    # Read first frame from video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"ERROR: Cannot read video: {video_path}")
        return
    
    h, w = frame.shape[:2]
    print(f"✓ Video first frame loaded: {w}x{h} pixels")
    
    # Create marked version of original frame
    frame_marked = frame.copy()
    
    # Mark reference points on original frame
    print("\n【Original Frame - Reference Points】")
    print("-" * 50)
    for i, (px, py) in enumerate(pixel_points):
        # Draw green circle and number
        cv2.circle(frame_marked, (int(px), int(py)), 15, (0, 255, 0), 3)  # Green circle
        cv2.putText(frame_marked, str(i+1), (int(px)+20, int(py)-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        print(f"Point {i+1}: Pixel({px:.0f}, {py:.0f}) → World({world_points[i][0]:.2f}, {world_points[i][1]:.2f})")
    
    # Add title to original frame
    cv2.putText(frame_marked, "Original Frame with Reference Points",
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    # Calculate world coordinate ranges
    # Find bounding box of all world reference points
    world_pts = np.array(world_points, dtype=np.float32)
    min_x = min(p[0] for p in world_points)
    max_x = max(p[0] for p in world_points)
    min_y = min(p[1] for p in world_points)
    max_y = max(p[1] for p in world_points)
    
    world_width = max_x - min_x
    world_height = max_y - min_y
    
    print(f"\n【World Coordinate Range】")
    print("-" * 50)
    print(f"X range: {min_x:.2f} to {max_x:.2f} (width {world_width:.2f}m)")
    print(f"Y range: {min_y:.2f} to {max_y:.2f} (height {world_height:.2f}m)")
    
    # Create warped image using inverse Homography matrix
    H_inv = np.linalg.inv(H)
    
    # Use larger output canvas to display transformation result
    output_scale = 100  # 100 pixels per meter
    output_width = int(world_width * output_scale) + 100
    output_height = int(world_height * output_scale) + 100
    
    warped = cv2.warpPerspective(frame, H_inv, (output_width, output_height))
    
    # Mark reference points on warped frame
    warped_marked = warped.copy()
    
    print(f"\n【Warped Frame - Bird's Eye View】")
    print("-" * 50)
    print(f"Output size: {output_width}x{output_height} pixels (1m = 100 pixels)")
    
    # Mark transformed reference points
    for i, (wx, wy) in enumerate(world_points):
        # Convert to output image coordinates (relative to minimum values)
        out_x = int((wx - min_x) * output_scale) + 50
        out_y = int((max_y - wy) * output_scale) + 50  # Flip Y-axis
        
        cv2.circle(warped_marked, (out_x, out_y), 15, (255, 0, 0), 3)  # Blue circle
        cv2.putText(warped_marked, f"{i+1}", (out_x+20, out_y-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        print(f"Point {i+1}: World({wx:.2f}, {wy:.2f}) → Output({out_x}, {out_y})")
    
    cv2.putText(warped_marked, "Warped View (Bird's Eye View)",
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
    
    # Combine two images for side-by-side comparison
    # Resize to match heights
    h1, w1 = frame_marked.shape[:2]
    h2, w2 = warped_marked.shape[:2]
    
    # Scale down warped image for better comparison
    scale = min(h1 / h2, 800 / w2)
    warped_resized = cv2.resize(warped_marked, (int(w2*scale), int(h2*scale)))
    
    # Create white background canvas
    canvas_w = w1 + warped_resized.shape[1] + 30
    canvas_h = max(h1, warped_resized.shape[0]) + 60
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # Place both images on canvas
    canvas[10:10+h1, 10:10+w1] = frame_marked
    canvas[10:10+warped_resized.shape[0], w1+20:w1+20+warped_resized.shape[1]] = warped_resized
    
    # Add title
    cv2.putText(canvas, "Homography Verification",
               (10, canvas_h-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Save result
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    cv2.imwrite(output_path, canvas)
    
    print(f"\n✓ Verification image saved: {output_path}")
    print("\n【Verification Guidelines】")
    print("-" * 50)
    print("Green circles: Reference point positions in original frame (pixel coords)")
    print("Blue circles:  Reference point positions in warped frame (world coords)")
    print("\nSuccessful calibration indicators:")
    print("  ✓ Blue circles form a regular rectangle")
    print("  ✓ No extreme distortion in warped view")
    print("  ✓ Coordinate ranges match expected scene layout")
    print("\nIf verification fails:")
    print("  - Check reference point accuracy in calibration JSON")
    print("  - Verify world coordinates are correct")
    print("  - Recalibrate using more precise reference points")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Homography Transformation Verification Tool')
    parser.add_argument('--video', type=str, required=True, help='Input video file path')
    parser.add_argument('--homography', type=str, required=True, help='Homography calibration JSON file path')
    parser.add_argument('--output', type=str, default='verify_homography.jpg', help='Output verification image path')
    
    args = parser.parse_args()
    
    verify_homography(args.video, args.homography, args.output)
