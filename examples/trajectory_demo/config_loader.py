"""
config_loader.py
******MAIN FILE********
This module loads YAML configuration files to manage various pipeline parameters—including YOLO, 
Homography, collision detection, and TTC/PET thresholds. It enables users to flexibly adjust these 
parameters through the configuration file without modifying the source code.

配置管理模块 - 用于加载和管理YAML配置文件
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class VideoConfig:
    """Video configuration"""
    path: str
    description: str = ""
    width: int = None
    height: int = None
    fps: float = 30


@dataclass
class YOLOConfig:
    """YOLO detection configuration"""
    model: str = "yolo11m"
    confidence_threshold: float = 0.45
    skip_frames: int = 3


@dataclass
class HomographyConfig:
    """Homography configuration"""
    enabled: bool = True
    matrix_file: str = None
    calibration_points: list = None


@dataclass
class CollisionConfig:
    """Collision detection configuration"""
    proximity_threshold_m: float = 4.5
    anchor_distance_threshold_m: float = 1.0
    vertex_shrink: float = 0.75


@dataclass
class TTCPETConfig:
    """TTC and PET configuration"""
    rear_end_serious: float = 2.8
    rear_end_general: float = 4.7
    sideswipe_serious: float = 2.3
    sideswipe_general: float = 4.2
    safety_margin_m: float = 1.5


@dataclass
class OutputConfig:
    """Output configuration"""
    save_yolo_detection: bool = True
    save_keyframes: bool = True
    generate_pdf: bool = True
    save_json: bool = True
    output_dir: str = None


@dataclass
class ReportConfig:
    """Report configuration"""
    max_events_in_report: int = 10
    include_filtered_events: bool = True
    events_per_page_pdf: int = 3


@dataclass
class AdvancedConfig:
    """Advanced configuration"""
    min_track_length: int = 3
    track_gap_threshold: int = 4
    merge_same_frame_objects: bool = True
    merge_distance_threshold_px: int = 100


class ConfigLoader:
    """Configuration loader"""
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration loader
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.raw_config = yaml.safe_load(f)
        
        self._parse_config()
    
    def _parse_config(self):
        """Parse configuration file"""
        # Video configuration
        video_dict = self.raw_config.get('video', {})
        self.video = VideoConfig(
            path=video_dict.get('path'),
            description=video_dict.get('description', ''),
            width=video_dict.get('resolution', {}).get('width'),
            height=video_dict.get('resolution', {}).get('height'),
            fps=video_dict.get('fps', 30)
        )
        
        # YOLO configuration
        yolo_dict = self.raw_config.get('yolo', {})
        self.yolo = YOLOConfig(
            model=yolo_dict.get('model', 'yolo11m'),
            confidence_threshold=yolo_dict.get('confidence_threshold', 0.45),
            skip_frames=yolo_dict.get('skip_frames', 3)
        )
        
        # Homography configuration
        homo_dict = self.raw_config.get('homography', {})
        self.homography = HomographyConfig(
            enabled=homo_dict.get('enabled', True),
            matrix_file=homo_dict.get('matrix_file'),
            calibration_points=homo_dict.get('calibration_points', [])
        )
        
        # Collision detection configuration
        collision_dict = self.raw_config.get('collision', {})
        self.collision = CollisionConfig(
            proximity_threshold_m=collision_dict.get('proximity_threshold_m', 4.5),
            anchor_distance_threshold_m=collision_dict.get('anchor_distance_threshold_m', 1.0),
            vertex_shrink=collision_dict.get('vertex_shrink', 0.75)
        )
        
        # TTC and PET configuration
        ttc_pet_dict = self.raw_config.get('ttc_pet', {})
        thresholds = ttc_pet_dict.get('ttc_thresholds', {})
        self.ttc_pet = TTCPETConfig(
            rear_end_serious=thresholds.get('rear_end_serious', 2.8),
            rear_end_general=thresholds.get('rear_end_general', 4.7),
            sideswipe_serious=thresholds.get('sideswipe_serious', 2.3),
            sideswipe_general=thresholds.get('sideswipe_general', 4.2),
            safety_margin_m=ttc_pet_dict.get('safety_margin_m', 1.5)
        )
        
        # Output configuration
        output_dict = self.raw_config.get('output', {})
        self.output = OutputConfig(
            save_yolo_detection=output_dict.get('save_yolo_detection', True),
            save_keyframes=output_dict.get('save_keyframes', True),
            generate_pdf=output_dict.get('generate_pdf', True),
            save_json=output_dict.get('save_json', True),
            output_dir=output_dict.get('output_dir')
        )
        
        # Report configuration
        report_dict = self.raw_config.get('report', {})
        self.report = ReportConfig(
            max_events_in_report=report_dict.get('max_events_in_report', 10),
            include_filtered_events=report_dict.get('include_filtered_events', True),
            events_per_page_pdf=report_dict.get('events_per_page_pdf', 3)
        )
        
        # Advanced configuration
        advanced_dict = self.raw_config.get('advanced', {})
        self.advanced = AdvancedConfig(
            min_track_length=advanced_dict.get('min_track_length', 3),
            track_gap_threshold=advanced_dict.get('track_gap_threshold', 4),
            merge_same_frame_objects=advanced_dict.get('merge_same_frame_objects', True),
            merge_distance_threshold_px=advanced_dict.get('merge_distance_threshold_px', 100)
        )
    
    def get_homography_matrix(self) -> Tuple[np.ndarray, float]:
        """
        Get or compute the homography matrix
        
        Returns:
            (H matrix, scale factor)
        
        Raises:
            ValueError: If calibration points are insufficient
        """
        # Load if matrix file is provided
        if self.homography.matrix_file:
            import json
            with open(self.homography.matrix_file, 'r') as f:
                data = json.load(f)
                H = np.array(data['H'])
                scale = data.get('scale', 1.0)
                return H, scale
        
        # Otherwise compute using calibration points
        if not self.homography.calibration_points or len(self.homography.calibration_points) < 4:
            raise ValueError(f"Insufficient homography calibration points: need at least 4, have {len(self.homography.calibration_points or [])}")
        
        # Extract pixel and world coordinates
        src_points = []
        dst_points = []
        
        for point in self.homography.calibration_points:
            pixel = point.get('pixel')
            world = point.get('world')
            if pixel and world:
                src_points.append(pixel)
                dst_points.append(world)
        
        if len(src_points) < 4:
            raise ValueError(f"Insufficient valid calibration points: need at least 4, have {len(src_points)}")
        
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        # Compute homography matrix using OpenCV
        import cv2
        H, _ = cv2.findHomography(src_points, dst_points)
        
        if H is None:
            raise ValueError("Failed to compute homography matrix, calibration points may be unsuitable")
        
        # Calculate scale factor (pixels/meter)
        # Use distance ratio between first and second points
        pixel_dist_1_2 = np.linalg.norm(src_points[1] - src_points[0])
        world_dist_1_2 = np.linalg.norm(dst_points[1] - dst_points[0])
        scale = pixel_dist_1_2 / world_dist_1_2 if world_dist_1_2 > 0 else 1.0
        
        return H, scale
    
    def print_config(self):
        """Print configuration summary"""
        print("\n" + "="*70)
        print("Configuration loaded successfully")
        print("="*70)
        print(f"\n[Video]")
        print(f"  Path: {self.video.path}")
        print(f"  Description: {self.video.description}")
        if self.video.width and self.video.height:
            print(f"  Resolution: {self.video.width}x{self.video.height}")
        print(f"  Frame rate: {self.video.fps} fps")
        
        print(f"\n[YOLO Detection]")
        print(f"  Model: {self.yolo.model}")
        print(f"  Confidence: {self.yolo.confidence_threshold}")
        print(f"  Skip frames: process 1 of {self.yolo.skip_frames} frames")
        
        print(f"\n[Collision Detection]")
        print(f"  Proximity threshold: {self.collision.proximity_threshold_m}m")
        print(f"  Anchor distance threshold: {self.collision.anchor_distance_threshold_m}m")
        print(f"  Vertex shrink factor: {self.collision.vertex_shrink}")
        
        if self.homography.enabled:
            print(f"\n[Homography]")
            if self.homography.matrix_file:
                print(f"  Matrix file: {self.homography.matrix_file}")
            else:
                print(f"  Calibration points: {len(self.homography.calibration_points or [])}")
        
        print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    # Test
    import sys
    if len(sys.argv) > 1:
        loader = ConfigLoader(sys.argv[1])
        loader.print_config()
    else:
        print("Usage: python config_loader.py <config_file.yaml>")
