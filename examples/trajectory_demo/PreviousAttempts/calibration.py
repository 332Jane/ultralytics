"""
[DEPRECATED] calibration.py
Homography Matrix Computation Tool
Command-line annotation tool, now replaced by YAML configuration



Usage flow:
1. Provide pixel coordinates and corresponding world coordinates for 4 reference points
2. Compute the Homography matrix
3. Save the matrix to a JSON file for subsequent use

Usage:
python calibration.py --pixel-points "100,50 1800,80 1850,1000 120,1050" --world-points "0,0 12,0 12,8 0,8" --output calibration/
"""

import cv2
import numpy as np
import json
import os
import argparse
from pathlib import Path


class HomographyCalibrator:
    def __init__(self, pixel_points, world_points, output_dir='calibration'):
        """Initialize the calibration tool
        
        Parameters:
        - pixel_points: [(px1,py1), (px2,py2), (px3,py3), (px4,py4)] List of pixel coordinates
        - world_points: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] List of world coordinates
        - output_dir: Output directory (to save the matrix)
        """
        if len(pixel_points) != 4 or len(world_points) != 4:
            raise ValueError("Must provide exactly 4 pixel coordinates and 4 world coordinates")
        
        self.pixel_points = pixel_points
        self.world_points = world_points
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def compute_homography(self):
        """Compute the Homography matrix"""
        # 转换为numpy数组
        src_pts = np.float32(self.pixel_points)
        dst_pts = np.float32(self.world_points)
        
        # 计算homography矩阵
        H, _ = cv2.findHomography(src_pts, dst_pts)
        
        if H is None:
            print("Failed to compute Homography matrix")
            return None
        
        print("\n✓ Homography matrix computed successfully!")
        print("\nMatrix content:")
        print(H)
        
        return H
    
    def save_homography(self, H, video_name='calibration'):
        """Save the Homography matrix to a JSON file"""
        if H is None:
            return None
        
        # Convert matrix to list (for JSON serialization)
        H_list = H.tolist()
        
        output_file = os.path.join(self.output_dir, f"{video_name}_homography.json")
        
        data = {
            "video_name": video_name,
            "homography_matrix": H_list,
            "pixel_points": self.pixel_points,
            "world_points": self.world_points,
            "notes": "Use this matrix to convert pixel coordinates to world coordinates"
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Homography matrix saved to: {output_file}")
        return output_file
    
    def test_transform(self, H):
        """Test the transformation effect"""
        if H is None:
            return
        
        print("\n[Transformation Test Results]")
        print("-" * 50)
        
        for i, (px_pt, w_pt) in enumerate(zip(self.pixel_points, self.world_points)):
            # Perform perspective transformation manually
            pixel_coord = np.array([[[px_pt[0], px_pt[1]]]], dtype=np.float32)
            world_coord = cv2.perspectiveTransform(pixel_coord, H)
            transformed_x, transformed_y = world_coord[0][0]
            
            error_x = abs(transformed_x - w_pt[0])
            error_y = abs(transformed_y - w_pt[1])
            
            print(f"Point {i+1}:")
            print(f"  Pixel coordinates: {px_pt}")
            print(f"  Input world coordinates: ({w_pt[0]:.2f}, {w_pt[1]:.2f})")
            print(f"  Transformed coordinates: ({transformed_x:.2f}, {transformed_y:.2f})")
            print(f"  Error: ({error_x:.4f}, {error_y:.4f}) meters")
            print()


def parse_coordinates(coord_str):
    """Parse coordinate string
    
    Format: "x1,y1 x2,y2 x3,y3 x4,y4"
    Example: "100,50 1800,80 1850,1000 120,1050"
    """
    points = []
    try:
        for point_str in coord_str.split():
            x, y = point_str.split(',')
            points.append((float(x), float(y)))
    except Exception as e:
        raise ValueError(f"Incorrect coordinate format: {e}. Correct format should be: 'x1,y1 x2,y2 x3,y3 x4,y4'")
    
    if len(points) != 4:
        raise ValueError(f"Need 4 coordinate points, but {len(points)} were provided")
    
    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Homography Calibration Tool - Compute transformation matrix from coordinates')
    parser.add_argument('--pixel-points', type=str, required=True,
                       help='Pixel coordinates (format: "x1,y1 x2,y2 x3,y3 x4,y4")')
    parser.add_argument('--world-points', type=str, required=True,
                       help='World coordinates (format: "x1,y1 x2,y2 x3,y3 x4,y4", unit: meters)')
    parser.add_argument('--video-name', type=str, default='calibration',
                       help='Video name (used for output filename)')
    parser.add_argument('--output', type=str, default='calibration',
                       help='Output directory')
    
    args = parser.parse_args()
    
    try:
        # Parse coordinates
        pixel_points = parse_coordinates(args.pixel_points)
        world_points = parse_coordinates(args.world_points)
        
        print(f"Pixel coordinates: {pixel_points}")
        print(f"World coordinates: {world_points}")
        
        # Perform calibration
        calibrator = HomographyCalibrator(pixel_points, world_points, args.output)
        H = calibrator.compute_homography()
        
        if H is not None:
            calibrator.save_homography(H, args.video_name)
            calibrator.test_transform(H)
            print("\n✓ Calibration completed!")
        else:
            print("❌ Calibration failed")
            exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
