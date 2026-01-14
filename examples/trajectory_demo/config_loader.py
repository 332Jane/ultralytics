"""
配置管理模块 - 用于加载和管理YAML配置文件
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class VideoConfig:
    """视频配置"""
    path: str
    description: str = ""
    width: int = None
    height: int = None
    fps: float = 30


@dataclass
class YOLOConfig:
    """YOLO检测配置"""
    model: str = "yolo11m"
    confidence_threshold: float = 0.45
    skip_frames: int = 3


@dataclass
class HomographyConfig:
    """Homography配置"""
    enabled: bool = True
    matrix_file: str = None
    calibration_points: list = None


@dataclass
class CollisionConfig:
    """碰撞检测配置"""
    proximity_threshold_m: float = 4.5
    anchor_distance_threshold_m: float = 1.0
    vertex_shrink: float = 0.75


@dataclass
class TTCPETConfig:
    """TTC和PET配置"""
    rear_end_serious: float = 2.8
    rear_end_general: float = 4.7
    sideswipe_serious: float = 2.3
    sideswipe_general: float = 4.2
    safety_margin_m: float = 1.5


@dataclass
class OutputConfig:
    """输出配置"""
    save_yolo_detection: bool = True
    save_keyframes: bool = True
    generate_pdf: bool = True
    save_json: bool = True
    output_dir: str = None


@dataclass
class ReportConfig:
    """报告配置"""
    max_events_in_report: int = 10
    include_filtered_events: bool = True
    events_per_page_pdf: int = 3


@dataclass
class AdvancedConfig:
    """高级配置"""
    min_track_length: int = 3
    track_gap_threshold: int = 4
    merge_same_frame_objects: bool = True
    merge_distance_threshold_px: int = 100


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: str):
        """
        初始化配置加载器
        
        Args:
            config_path: YAML配置文件路径
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.raw_config = yaml.safe_load(f)
        
        self._parse_config()
    
    def _parse_config(self):
        """解析配置文件"""
        # 视频配置
        video_dict = self.raw_config.get('video', {})
        self.video = VideoConfig(
            path=video_dict.get('path'),
            description=video_dict.get('description', ''),
            width=video_dict.get('resolution', {}).get('width'),
            height=video_dict.get('resolution', {}).get('height'),
            fps=video_dict.get('fps', 30)
        )
        
        # YOLO配置
        yolo_dict = self.raw_config.get('yolo', {})
        self.yolo = YOLOConfig(
            model=yolo_dict.get('model', 'yolo11m'),
            confidence_threshold=yolo_dict.get('confidence_threshold', 0.45),
            skip_frames=yolo_dict.get('skip_frames', 3)
        )
        
        # Homography配置
        homo_dict = self.raw_config.get('homography', {})
        self.homography = HomographyConfig(
            enabled=homo_dict.get('enabled', True),
            matrix_file=homo_dict.get('matrix_file'),
            calibration_points=homo_dict.get('calibration_points', [])
        )
        
        # 碰撞检测配置
        collision_dict = self.raw_config.get('collision', {})
        self.collision = CollisionConfig(
            proximity_threshold_m=collision_dict.get('proximity_threshold_m', 4.5),
            anchor_distance_threshold_m=collision_dict.get('anchor_distance_threshold_m', 1.0),
            vertex_shrink=collision_dict.get('vertex_shrink', 0.75)
        )
        
        # TTC和PET配置
        ttc_pet_dict = self.raw_config.get('ttc_pet', {})
        thresholds = ttc_pet_dict.get('ttc_thresholds', {})
        self.ttc_pet = TTCPETConfig(
            rear_end_serious=thresholds.get('rear_end_serious', 2.8),
            rear_end_general=thresholds.get('rear_end_general', 4.7),
            sideswipe_serious=thresholds.get('sideswipe_serious', 2.3),
            sideswipe_general=thresholds.get('sideswipe_general', 4.2),
            safety_margin_m=ttc_pet_dict.get('safety_margin_m', 1.5)
        )
        
        # 输出配置
        output_dict = self.raw_config.get('output', {})
        self.output = OutputConfig(
            save_yolo_detection=output_dict.get('save_yolo_detection', True),
            save_keyframes=output_dict.get('save_keyframes', True),
            generate_pdf=output_dict.get('generate_pdf', True),
            save_json=output_dict.get('save_json', True),
            output_dir=output_dict.get('output_dir')
        )
        
        # 报告配置
        report_dict = self.raw_config.get('report', {})
        self.report = ReportConfig(
            max_events_in_report=report_dict.get('max_events_in_report', 10),
            include_filtered_events=report_dict.get('include_filtered_events', True),
            events_per_page_pdf=report_dict.get('events_per_page_pdf', 3)
        )
        
        # 高级配置
        advanced_dict = self.raw_config.get('advanced', {})
        self.advanced = AdvancedConfig(
            min_track_length=advanced_dict.get('min_track_length', 3),
            track_gap_threshold=advanced_dict.get('track_gap_threshold', 4),
            merge_same_frame_objects=advanced_dict.get('merge_same_frame_objects', True),
            merge_distance_threshold_px=advanced_dict.get('merge_distance_threshold_px', 100)
        )
    
    def get_homography_matrix(self) -> Tuple[np.ndarray, float]:
        """
        获取或计算homography矩阵
        
        Returns:
            (H矩阵, 缩放因子)
        
        Raises:
            ValueError: 如果标定点不足
        """
        # 如果提供了矩阵文件，则加载
        if self.homography.matrix_file:
            import json
            with open(self.homography.matrix_file, 'r') as f:
                data = json.load(f)
                H = np.array(data['H'])
                scale = data.get('scale', 1.0)
                return H, scale
        
        # 否则使用标定点计算
        if not self.homography.calibration_points or len(self.homography.calibration_points) < 4:
            raise ValueError(f"Homography标定点不足：需要至少4个点，现有{len(self.homography.calibration_points or [])}个")
        
        # 提取像素坐标和世界坐标
        src_points = []
        dst_points = []
        
        for point in self.homography.calibration_points:
            pixel = point.get('pixel')
            world = point.get('world')
            if pixel and world:
                src_points.append(pixel)
                dst_points.append(world)
        
        if len(src_points) < 4:
            raise ValueError(f"有效标定点不足：需要至少4个点，现有{len(src_points)}个")
        
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        # 使用OpenCV计算homography矩阵
        import cv2
        H, _ = cv2.findHomography(src_points, dst_points)
        
        if H is None:
            raise ValueError("无法计算homography矩阵，标定点可能不合适")
        
        # 计算缩放因子（像素/米）
        # 使用第一和第二个点之间的距离比
        pixel_dist_1_2 = np.linalg.norm(src_points[1] - src_points[0])
        world_dist_1_2 = np.linalg.norm(dst_points[1] - dst_points[0])
        scale = pixel_dist_1_2 / world_dist_1_2 if world_dist_1_2 > 0 else 1.0
        
        return H, scale
    
    def print_config(self):
        """打印配置摘要"""
        print("\n" + "="*70)
        print("配置加载成功")
        print("="*70)
        print(f"\n【视频】")
        print(f"  路径: {self.video.path}")
        print(f"  描述: {self.video.description}")
        if self.video.width and self.video.height:
            print(f"  分辨率: {self.video.width}x{self.video.height}")
        print(f"  帧率: {self.video.fps} fps")
        
        print(f"\n【YOLO检测】")
        print(f"  模型: {self.yolo.model}")
        print(f"  置信度: {self.yolo.confidence_threshold}")
        print(f"  跳帧: 每{self.yolo.skip_frames}帧处理一帧")
        
        print(f"\n【碰撞检测】")
        print(f"  接近距离阈值: {self.collision.proximity_threshold_m}m")
        print(f"  锚点距离阈值: {self.collision.anchor_distance_threshold_m}m")
        print(f"  顶点收缩因子: {self.collision.vertex_shrink}")
        
        if self.homography.enabled:
            print(f"\n【Homography】")
            if self.homography.matrix_file:
                print(f"  矩阵文件: {self.homography.matrix_file}")
            else:
                print(f"  标定点数: {len(self.homography.calibration_points or [])}")
        
        print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    # 测试
    import sys
    if len(sys.argv) > 1:
        loader = ConfigLoader(sys.argv[1])
        loader.print_config()
    else:
        print("使用方法: python config_loader.py <config_file.yaml>")
