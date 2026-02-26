"""
[DEPRECATED] coord_transform.py
# DEPRECATED: coord_transform.py
# Reason: Coordinate transformation logic has been integrated directly into 
# collision_detection_pipeline_yolo_first_method_a.py (Method A).
# The standalone module is no longer needed as current pipeline handles 
# all coordinate transformations internally.


Coordinate transformation: Use Homography matrix to transform pixel coordinates to world coordinates
Principle:
  Homography matrix is a 3x3 matrix that maps points from one plane to another
  In our application:
    Pixel coordinates (px, py) --[H]--> World coordinates (x_world, y_world)
  
  This converts calculated distances from "pixels" to "meters", more accurate for real-world use

Usage:
  1. First use calibration.py to calibrate and obtain the homography matrix
  2. Load the matrix in yolo_runner.py
  3. Use transform_point() or transform_batch() for transformation
"""

import numpy as np
import json
import os
from typing import Tuple, List, Optional


def load_homography(json_path: str) -> Optional[np.ndarray]:
    """Load Homography matrix from JSON file
    
    Parameters:
        json_path: Path to homography matrix JSON file
    
    Returns:
        H: 3x3 homography matrix, or None if loading fails
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        H = np.array(data['homography_matrix'], dtype=np.float32)
        print(f"✓ Homography matrix loaded: {json_path}")
        return H
    except Exception as e:
        print(f"❌ Failed to load Homography matrix: {e}")
        return None


def transform_point(pixel_point: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    """Perspective transform: Convert single point from pixel coordinates to world coordinates
    
    Parameters:
        pixel_point: (px, py) pixel coordinates
        H: 3x3 homography matrix
    
    Returns:
        (x_world, y_world) world coordinates (unit: meters)
    """
    # Construct homogeneous coordinates
    px, py = pixel_point
    pixel_homo = np.array([[[px, py, 1]]], dtype=np.float32)
    
    # Use OpenCV perspectiveTransform for transformation
    import cv2
    world_homo = cv2.perspectiveTransform(pixel_homo, H)
    
    x_world = float(world_homo[0][0][0])
    y_world = float(world_homo[0][0][1])
    
    return (x_world, y_world)


def transform_batch(pixel_points: List[Tuple[float, float]], H: np.ndarray) -> List[Tuple[float, float]]:
    """Perspective transform: Batch convert points from pixel coordinates to world coordinates
    
    Parameters:
        pixel_points: [(px1, py1), (px2, py2), ...] list of pixel coordinates
        H: 3x3 homography matrix
    
    Returns:
        [(x1, y1), (x2, y2), ...] list of world coordinates
    """
    if not pixel_points:
        return []
    
    # Convert to numpy array (format: N x 1 x 2)
    pixel_array = np.array(pixel_points, dtype=np.float32)
    pixel_array = pixel_array.reshape(-1, 1, 2)
    
    # Use OpenCV perspectiveTransform for batch transformation
    import cv2
    world_array = cv2.perspectiveTransform(pixel_array, H)
    
    # Convert back to list format
    world_points = [tuple(pt[0]) for pt in world_array]
    return world_points


def compute_world_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two world coordinates
    
    Parameters:
        point1: (x1, y1) world coordinates
        point2: (x2, y2) world coordinates
    
    Returns:
        Distance (unit: meters)
    """
    x1, y1 = point1
    x2, y2 = point2
    
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return float(distance)
