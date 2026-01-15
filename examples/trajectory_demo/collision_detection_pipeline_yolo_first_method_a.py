"""
collision_detection_pipeline_yolo_first_method_a.py

YOLO-First 碰撞检测管道 (Method A )
执行顺序: YOLO检测 → 轨迹构建(px) → 关键帧检测 → Homography变换(仅关键帧) → TTC分析


流程:
1. YOLO 检测 (原始视频或跳帧视频)
   - 在原始分辨率上检测所有物体
   - 保存检测框和 Track ID (像素空间)

2. 轨迹构建 (像素空间)
   - 关联 Track ID，建立轨迹
   - 估计速度 (px/s)
   - 所有计算在像素坐标系

3. 关键帧检测 (接近事件识别)
   - 识别"距离 < 150px"的物体对
   - 标记为关键帧

4. Homography 变换 (仅关键帧) 
   - 只对关键帧中的物体点做变换
   - 转换距离单位 (px → m)
   - 转换速度单位 (px/s → m/s)

5. TTC 和 Event 分级
   - 计算 TTC (在世界坐标)
   - 分级事件 (L1/L2/L3)
   - 生成报告

优势: 
- 
-  性能最优 (仅变换5-10%的数据)
-  逻辑清晰 (先找接近的，再精确分析)
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 导入YOLO
sys.path.append(os.path.dirname(__file__))
from ultralytics import YOLO

# 导入多锚点碰撞检测模块
from anchor_points import VehicleAnchors, PedestrianAnchors, BicycleAnchors, MotorcycleAnchors, get_vehicle_heading
from collision_analyzer import CollisionAnalyzer


class YOLOFirstPipelineA:
    def __init__(self, video_path, homography_path=None, output_base=None, skip_frames=3, model='yolo11n', min_track_length=3):
        """初始化 YOLO-First pipeline 
        
        Args:
            video_path: 原始视频路径
            homography_path: Homography JSON路径 (用于Step 4)
            output_base: 结果基础目录
            skip_frames: 抽帧参数，每隔 skip_frames 帧处理一帧 (最小值=3，用于性能优化和速度准确性)
            model: YOLO 模型选择 (yolo11n/yolo11m/yolo11l)
            min_track_length: 最小轨迹长度，短于此的被认为是误检
        """
        self.video_path = video_path
        self.homography_path = homography_path
        # 强制skip_frames至少为3，避免完全不跳帧的低效处理
        self.skip_frames = max(3, skip_frames)  # 强制至少跳帧3
        self.model = model  # YOLO 模型
        self.min_track_length = min_track_length  # 最小轨迹长度
        # 使用 /workspace/ultralytics/results 作为输出目录（确保路径正确）
        if output_base is None:
            output_base = "/workspace/ultralytics/results"
        self.output_base = Path(output_base)
        self.H = None
        self.pixel_per_meter = 1.0
        
        # 初始化配置对象（用于YAML配置）
        self.config = None
        
        # 创建带时间戳的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = (self.output_base / f"{timestamp}_yolo_first_method_a").resolve()
        
        # 创建子目录结构 (Method A)
        self.detection_dir = self.run_dir / "1_yolo_detection"
        self.trajectory_dir = self.run_dir / "2_trajectories"
        self.keyframe_dir = self.run_dir / "3_key_frames"
        self.homography_dir = self.run_dir / "4_homography_transform"
        self.analysis_dir = self.run_dir / "5_collision_analysis"
        
        for d in [self.detection_dir, self.trajectory_dir, self.keyframe_dir, 
                  self.homography_dir, self.analysis_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"YOLO-First 碰撞检测Pipeline (Method A - 导师推荐)")
        print(f"{'='*70}")
        print(f"时间戳: {timestamp}")
        print(f"结果目录: {self.run_dir}")
        print(f"执行顺序: YOLO → 轨迹(px) → 关键帧 → Homography(关键帧) → TTC")
    
    @classmethod
    def from_config(cls, config_path):
        """从YAML配置文件创建pipeline实例
        
        Args:
            config_path: YAML配置文件路径
        
        Returns:
            YOLOFirstMethodA实例
        """
        from config_loader import ConfigLoader
        
        config = ConfigLoader(config_path)
        
        # 获取homography矩阵或计算
        H = None
        if config.homography.enabled:
            try:
                H, scale = config.get_homography_matrix()
            except Exception as e:
                print(f"  ⚠️  无法加载/计算homography: {e}")
        
        # 创建pipeline实例
        pipeline = cls(
            video_path=config.video.path,
            homography_path=None,  # 我们直接使用H矩阵，不需要路径
            output_base=config.output.output_dir,
            skip_frames=config.yolo.skip_frames,
            model=config.yolo.model,
            min_track_length=config.advanced.min_track_length
        )
        
        # 保存配置对象和homography矩阵
        pipeline.config = config
        if H is not None:
            pipeline.H = H
            pipeline.pixel_per_meter = scale
        
        # 打印配置摘要
        config.print_config()
        
        return pipeline
    
    def load_homography(self):
        """加载 Homography 矩阵 (Step 4 需要)"""
        if not self.homography_path:
            print(f"\n⚠️  未提供 Homography，将仅在像素空间处理")
            return False
        
        try:
            with open(self.homography_path) as f:
                H_data = json.load(f)
            
            self.H = np.array(H_data['homography_matrix'], dtype=np.float32)
            pixel_points = H_data['pixel_points']
            world_points = H_data['world_points']
            
            # 保存到输出目录
            with open(self.homography_dir / 'homography.json', 'w') as f:
                json.dump(H_data, f, indent=2)
            
            # 计算像素到米的缩放因子
            if len(world_points) >= 2 and len(pixel_points) >= 2:
                px_dist = np.sqrt((pixel_points[0][0] - pixel_points[1][0])**2 + 
                                 (pixel_points[0][1] - pixel_points[1][1])**2)
                world_dist = np.sqrt((world_points[0][0] - world_points[1][0])**2 + 
                                    (world_points[0][1] - world_points[1][1])**2)
                
                self.pixel_per_meter = px_dist / world_dist if world_dist > 0 else 1.0
            
            print(f"  ✓ Homography已加载")
            print(f"    缩放因子: {self.pixel_per_meter:.2f} px/m")
            
            return True
        
        except Exception as e:
            print(f"  ❌ 加载Homography失败: {e}")
            return False
    
    def visualize_homography_calibration(self):
        """可视化Homography标定：分别保存原始帧和变换后的帧
        
        从YAML配置中读取标定点信息，在视频第一帧上标记这些点，
        然后使用Homography矩阵进行透视变换，展示变换效果
        """
        if self.H is None:
            return  # 如果没有Homography矩阵，不需要可视化
        
        if self.config is None:
            return  # 如果没有config对象，无法获取标定点信息
        
        print(f"\n【Homography 标定可视化】")
        
        try:
            # 获取第一帧
            cap = cv2.VideoCapture(self.video_path)
            ret, frame_original = cap.read()
            cap.release()
            
            if not ret:
                print(f"  ⚠️  无法读取视频第一帧")
                return
            
            frame_height, frame_width = frame_original.shape[:2]
            
            # ===== 图1：原始帧 + 标定点标记 =====
            frame_with_points = frame_original.copy()
            
            # 从config中获取标定点
            calibration_points = self.config.homography.calibration_points if self.config else []
            
            if not calibration_points:
                print(f"  ⚠️  未找到标定点信息")
                return
            
            # 在原始帧上绘制标定点
            pixel_points_list = []
            world_points_list = []
            
            for i, point in enumerate(calibration_points):
                # point是字典，从字典中提取坐标
                if isinstance(point, dict):
                    pixel_coords = point.get('pixel_coordinates', point.get('pixel', [0, 0]))
                    world_coords = point.get('world_coordinates', point.get('world', [0, 0]))
                    name = point.get('name', f'Point{i+1}')
                else:
                    # 如果是对象，尝试访问属性
                    pixel_coords = point.pixel_coordinates
                    world_coords = point.world_coordinates
                    name = point.name
                
                pixel_x, pixel_y = pixel_coords[0], pixel_coords[1]
                world_x, world_y = world_coords[0], world_coords[1]
                
                pixel_points_list.append([pixel_x, pixel_y])
                world_points_list.append([world_x, world_y])
                
                # 绘制圆点（原始像素坐标）
                center = (int(pixel_x), int(pixel_y))
                cv2.circle(frame_with_points, center, 8, (0, 255, 0), -1)      # 实心绿色圆
                cv2.circle(frame_with_points, center, 8, (0, 0, 255), 2)       # 红色边框
                
                # 绘制文字标签（点名称和世界坐标）
                label = f"{name} ({world_x:.2f}, {world_y:.2f}m)"
                cv2.putText(frame_with_points, label, 
                           (int(pixel_x)+15, int(pixel_y)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 编号
                cv2.putText(frame_with_points, f"#{i+1}", 
                           (int(pixel_x)-8, int(pixel_y)+8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # 在原始帧上添加标题和说明
            cv2.putText(frame_with_points, "Original Frame (Pixel Coordinates)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame_with_points, f"Calibration Points: {len(calibration_points)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame_with_points, f"Scale: {self.pixel_per_meter:.2f} px/m", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 保存原始帧
            original_path = self.homography_dir / '01_original_frame_with_calibration_points.jpg'
            cv2.imwrite(str(original_path), frame_with_points)
            
            # ===== 图2：使用原始homography矩阵手工逐像素变换 =====
            if len(pixel_points_list) >= 4:
                # 计算世界坐标范围
                world_xs = [p[0] for p in world_points_list]
                world_ys = [p[1] for p in world_points_list]
                
                min_x, max_x = min(world_xs), max(world_xs)
                min_y, max_y = min(world_ys), max(world_ys)
                
                # 添加边距（米）
                margin = 1.0
                min_x -= margin
                max_x += margin
                min_y -= margin
                max_y += margin
                
                # 输出图像尺寸：使用实际的像素/米比例
                output_width = int((max_x - min_x) * self.pixel_per_meter)
                output_height = int((max_y - min_y) * self.pixel_per_meter)
                
                # 确保尺寸合理（最小100x100，最大1200x1200）
                output_width = max(100, min(output_width, 1200))
                output_height = max(100, min(output_height, 1200))
                
                # 手工逐像素变换：对于输出图像中的每个像素，反向变换到原始帧采样
                frame_warped = np.ones((output_height, output_width, 3), dtype=np.uint8) * 200  # 灰色背景
                
                # 计算逆矩阵用于反向变换（从世界坐标回到像素坐标）
                try:
                    _, H_inv = cv2.invert(self.H)
                except:
                    H_inv = np.linalg.inv(self.H)
                
                # 逐像素填充（优化：每行处理一次而不是逐像素）
                for out_y in range(output_height):
                    for out_x in range(output_width):
                        # 将输出像素坐标映射回世界坐标
                        world_x = min_x + (out_x / output_width) * (max_x - min_x)
                        world_y = min_y + (out_y / output_height) * (max_y - min_y)
                        
                        # 使用逆矩阵反向变换到原始帧的像素坐标
                        world_pt = np.array([[[world_x, world_y]]], dtype=np.float32)
                        try:
                            pixel_pt = cv2.perspectiveTransform(world_pt, H_inv)[0, 0]
                            px, py = pixel_pt[0], pixel_pt[1]
                            
                            # 双线性插值采样
                            if 0 <= px < frame_width-1 and 0 <= py < frame_height-1:
                                x0, y0 = int(px), int(py)
                                x1, y1 = min(x0 + 1, frame_width-1), min(y0 + 1, frame_height-1)
                                
                                # 权重
                                wx = px - x0
                                wy = py - y0
                                
                                # 采样四个相邻像素进行双线性插值
                                val = (frame_original[y0, x0] * (1-wx) * (1-wy) +
                                      frame_original[y0, x1] * wx * (1-wy) +
                                      frame_original[y1, x0] * (1-wx) * wy +
                                      frame_original[y1, x1] * wx * wy)
                                frame_warped[out_y, out_x] = val.astype(np.uint8)
                        except:
                            pass
                
                # 在变换后的帧上标记标定点
                for i, (world_x, world_y) in enumerate(world_points_list):
                    # 映射世界坐标到输出图像像素坐标
                    out_x = int((world_x - min_x) / (max_x - min_x) * output_width)
                    out_y = int((world_y - min_y) / (max_y - min_y) * output_height)
                    
                    # 只在有效范围内绘制
                    if 0 <= out_x < output_width and 0 <= out_y < output_height:
                        center = (out_x, out_y)
                        # 绘制圆点
                        cv2.circle(frame_warped, center, 8, (0, 255, 0), -1)      # 实心绿色圆
                        cv2.circle(frame_warped, center, 8, (0, 0, 255), 2)       # 红色边框
                        
                        # 绘制编号
                        cv2.putText(frame_warped, f"#{i+1}", 
                                   (out_x-8, out_y+8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # 添加标题和说明
                cv2.putText(frame_warped, "Homography Transformed Frame (Bird's Eye View)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame_warped, f"World Range: X=[{min_x:.1f}, {max_x:.1f}]m, Y=[{min_y:.1f}, {max_y:.1f}]m", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(frame_warped, f"Output: {output_width}x{output_height}px ({self.pixel_per_meter:.1f} px/m)", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # 旋转180度（因为homography变换会导致图像上下颠倒）
                frame_warped = cv2.rotate(frame_warped, cv2.ROTATE_180)
                
                # 保存变换后的帧
                transformed_path = self.homography_dir / '02_homography_transformed_frame.jpg'
                cv2.imwrite(str(transformed_path), frame_warped)
                
                print(f"  ✓ Homography标定可视化已保存")
                print(f"    原始帧: {original_path.name}")
                print(f"    变换帧: {transformed_path.name}")
                print(f"    标定点数: {len(calibration_points)}")
                print(f"    像素坐标范围: X=[0, {frame_width}], Y=[0, {frame_height}]")
                print(f"    世界坐标范围: X=[{min_x:.2f}, {max_x:.2f}]m, Y=[{min_y:.2f}, {max_y:.2f}]m")
                print(f"    输出尺寸: {output_width}x{output_height}px ({self.pixel_per_meter:.2f} px/m)")
            else:
                print(f"  ⚠️  标定点少于4个，无法进行透视变换")
                transformed_path = None
            
        except Exception as e:
            print(f"  ⚠️  Homography标定可视化失败: {e}")
            import traceback
            traceback.print_exc()
    

    # =========================================================================
    # 可视化和图像保存
    # =========================================================================
    
    def save_detection_frame(self, video_path, frame_num, output_path, detections=None):
        """保存指定帧的图像（带 YOLO 检测框）"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False
        
        # 如果有检测数据，绘制检测框
        if detections:
            for obj in detections.get('objects', []):
                x, y, w, h = obj['bbox_xywh']
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                
                # 绘制检测框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制 Track ID
                track_id = obj['track_id']
                conf = obj['conf']
                text = f"ID:{track_id} ({conf:.2f})"
                cv2.putText(frame, text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存图像
        cv2.imwrite(str(output_path), frame)
        return True
    
    def save_detection_frame_from_array(self, frame_img, frame_num, output_path, boxes, ids, classes):
        """从frame数组和检测结果绘制并保存检测框"""
        frame = frame_img.copy()
        
        # 绘制检测框
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            
            # 绘制检测框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制 Track ID
            track_id = int(ids[i]) if ids[i] is not None else -1
            text = f"ID:{track_id}"
            cv2.putText(frame, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存图像
        cv2.imwrite(str(output_path), frame)
        return True
    
    
    def save_keyframe_with_distance(self, video_path, frame_num, output_path, proximity_event):
        """保存关键帧图像（绘制两个接近的物体、距离、多锚点碰撞点）"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False
        
        # 获取两个物体的信息
        center_1 = proximity_event.get('center_1_px', proximity_event.get('center_1', [0, 0]))
        center_2 = proximity_event.get('center_2_px', proximity_event.get('center_2', [0, 0]))
        track_id_1 = proximity_event.get('track_id_1', -1)
        track_id_2 = proximity_event.get('track_id_2', -1)
        distance_pixel = proximity_event.get('distance_pixel', 0)
        distance_meters = proximity_event.get('distance_meters', 0)
        class_1 = proximity_event.get('class_1', 'Unknown')
        class_2 = proximity_event.get('class_2', 'Unknown')
        
        # 绘制两个物体的中心点
        pt1 = tuple(map(int, center_1))
        pt2 = tuple(map(int, center_2))
        
        # 绘制圆点和ID
        cv2.circle(frame, pt1, 5, (0, 255, 0), -1)  # 绿色圆点
        cv2.putText(frame, f"ID:{track_id_1}", (pt1[0]+10, pt1[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.circle(frame, pt2, 5, (0, 0, 255), -1)  # 红色圆点
        cv2.putText(frame, f"ID:{track_id_2}", (pt2[0]+10, pt2[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 绘制连接线
        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)  # 蓝色线
        
        # ========== 多锚点碰撞可视化 ==========
        # 如果有多锚点分析结果，绘制最近碰撞点
        if 'multi_anchor_detailed' in proximity_event:
            try:
                multi_anchor = proximity_event['multi_anchor_detailed']
                closest_parts = multi_anchor.get('closest_parts', {})
                
                point1_px = closest_parts.get('point1_px')
                point2_px = closest_parts.get('point2_px')
                
                if point1_px and point2_px:
                    # 确保坐标是整数
                    anchor_pt1 = tuple(map(int, point1_px))
                    anchor_pt2 = tuple(map(int, point2_px))
                    
                    # 绘制最近碰撞点：大圆圈（绿/红）
                    cv2.circle(frame, anchor_pt1, 12, (0, 255, 0), 2)  # 绿色大圆圈 (Object 1)
                    cv2.circle(frame, anchor_pt2, 12, (0, 0, 255), 2)  # 红色大圆圈 (Object 2)
                    
                    # 绘制最近碰撞点之间的连线（紫色）
                    cv2.line(frame, anchor_pt1, anchor_pt2, (255, 0, 255), 2)
                    
                    # 显示锚点名称和距离
                    obj1_part = closest_parts.get('object1_part', '?')
                    obj2_part = closest_parts.get('object2_part', '?')
                    min_dist_m = multi_anchor.get('min_distance_meters', 0)
                    risk_level = multi_anchor.get('risk_level', 'UNKNOWN')
                    ttc = multi_anchor.get('ttc_seconds')
                    
                    # 在连接线中点显示距离和风险等级
                    mid_x = (anchor_pt1[0] + anchor_pt2[0]) // 2
                    mid_y = (anchor_pt1[1] + anchor_pt2[1]) // 2
                    
                    # 距离信息
                    dist_text = f"Anchor: {min_dist_m:.2f}m"
                    cv2.putText(frame, dist_text, (mid_x-80, mid_y-30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    
                    # 风险等级和颜色
                    risk_color_map = {
                        'CRITICAL': (0, 0, 255),    # Red
                        'HIGH': (0, 165, 255),       # Orange
                        'MEDIUM': (0, 255, 255),     # Yellow
                        'LOW': (0, 255, 0),          # Green
                    }
                    risk_color = risk_color_map.get(risk_level, (255, 255, 255))
                    cv2.putText(frame, f"Risk: {risk_level}", (mid_x-80, mid_y+10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2)
                    
                    # TTC信息
                    if ttc is not None:
                        ttc_text = f"TTC: {ttc:.2f}s" if ttc > 0 else "TTC: CRITICAL"
                        cv2.putText(frame, ttc_text, (mid_x-80, mid_y+50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2)
                    
                    # 碰撞部分标注
                    cv2.putText(frame, obj1_part, (anchor_pt1[0]-50, anchor_pt1[1]-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, obj2_part, (anchor_pt2[0]+20, anchor_pt2[1]-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            except Exception as e:
                # 如果多锚点可视化失败，继续使用简单的中心点可视化
                pass
        
        # 显示距离信息 (像素和世界坐标)
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        distance_text = f"Center Distance: {distance_meters:.2f}m ({distance_pixel:.0f}px)"
        cv2.putText(frame, distance_text, (mid_x-130, mid_y+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 显示物体类别信息
        class_text = f"{class_1} vs {class_2}"
        cv2.putText(frame, class_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 保存图像
        cv2.imwrite(str(output_path), frame)
        return True

    
    # =========================================================================
    # STEP 1: YOLO 检测 (像素空间)
    # =========================================================================
    
    def run_yolo_detection(self, conf_threshold=0.45):
        """Step 1: YOLO 检测 (原始视频或跳帧视频)
        
        输出:
        - 保存所有检测框和 Track ID (像素空间)
        - 生成检测统计
        """
        print(f"\n【Step 1: YOLO 检测】")
        
        # 加载指定的 YOLO 模型
        model = YOLO(f'{self.model}.pt')
        print(f"  加载模型: {self.model}.pt")
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        all_detections = []
        frame_count = 0
        detection_frames_count = 0
        
        # 计算跳帧后的处理帧数
        expected_processing_frames = (total_frames + self.skip_frames - 1) // self.skip_frames
        
        # 抽帧处理
        if self.skip_frames > 1:
            print(f"  处理中: 将处理 ~{expected_processing_frames} 帧 (从总共 {total_frames}帧中，每隔{self.skip_frames}帧处理一帧)...")
        else:
            print(f"  处理中: {total_frames}帧 @ {fps:.2f}FPS...")
        
        # 如果需要跳帧，先收集要处理的帧
        frames_to_process = []
        if self.skip_frames > 1:
            # 只读取需要处理的帧
            for frame_idx in range(0, total_frames, self.skip_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames_to_process.append((frame_idx + 1, frame))  # frame_idx+1 because frames are 1-indexed
            cap.release()
            
            print(f"    ✓ 已加载{len(frames_to_process)}帧进行处理")
            
            # 用YOLO处理这些帧
            for frame_num, frame_img in frames_to_process:
                results = model.track(source=frame_img, persist=True, conf=conf_threshold)
                
                for result in results:
                    frame_count = frame_num
                    
                    if result.boxes is None or len(result.boxes) == 0:
                        if frame_num % 30 == 0:
                            print(f"    Frame {frame_num}/{total_frames} - 无物体")
                        continue
                    
                    detection_frames_count += 1
                    boxes = result.boxes.xywh.cpu().numpy()
                    ids = result.boxes.id
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    confs = result.boxes.conf.cpu().numpy()
                    
                    frame_detections = {
                        'frame': frame_count,
                        'time': frame_count / fps,
                        'objects': []
                    }
                    
                    # 只有在有检测到对象时才处理
                    if len(boxes) > 0 and ids is not None:
                        for i in range(len(boxes)):
                            obj_data = {
                                'track_id': int(ids[i]) if ids[i] is not None else -1,
                                'class': int(classes[i]),
                                'conf': float(confs[i]),
                                'bbox_xywh': boxes[i].tolist(),  # [x_center, y_center, w, h] 像素
                            }
                            frame_detections['objects'].append(obj_data)
                        
                        # 保存检测框图像（每个有检测的帧）
                        frame_img_path = self.detection_dir / f"frame_{frame_count:04d}.jpg"
                        self.save_detection_frame_from_array(frame_img, frame_count, frame_img_path, boxes, ids, classes)
                    
                    all_detections.append(frame_detections)
                    
                    if frame_num % 30 == 0:
                        print(f"    Frame {frame_num}/{total_frames} - {len(boxes)}个物体")
        else:
            # 不跳帧：处理所有帧
            for result in model.track(source=self.video_path, stream=True, 
                                     persist=True, conf=conf_threshold):
                frame_count += 1
                
                if result.boxes is None or len(result.boxes) == 0:
                    if frame_count % 30 == 0:
                        print(f"    Frame {frame_count}/{total_frames} - 无物体")
                    continue
                
                detection_frames_count += 1
                boxes = result.boxes.xywh.cpu().numpy()
                ids = result.boxes.id
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                
                frame_detections = {
                    'frame': frame_count,
                    'time': frame_count / fps,
                    'objects': []
                }
                
                # 只有在有检测到对象时才处理
                if len(boxes) > 0 and ids is not None:
                    for i in range(len(boxes)):
                        obj_data = {
                            'track_id': int(ids[i]) if ids[i] is not None else -1,
                            'class': int(classes[i]),
                            'conf': float(confs[i]),
                            'bbox_xywh': boxes[i].tolist(),  # [x_center, y_center, w, h] 像素
                        }
                        frame_detections['objects'].append(obj_data)
                    
                    # 保存检测框图像（每个有检测的帧）
                    frame_img_path = self.detection_dir / f"frame_{frame_count:04d}.jpg"
                    self.save_detection_frame(self.video_path, frame_count, frame_img_path, frame_detections)
                
                all_detections.append(frame_detections)
                
                if frame_count % 30 == 0:
                    print(f"    Frame {frame_count}/{total_frames} - {len(boxes)}个物体")
            
            cap.release()

        
        # 保存原始检测结果 (像素空间)
        detections_path = self.detection_dir / 'detections_pixel.json'
        with open(detections_path, 'w') as f:
            json.dump(all_detections, f, indent=2)
        
        # 生成检测统计
        stats = {
            'total_frames': total_frames,
            'fps': fps,
            'detection_frames': detection_frames_count,
            'confidence_threshold': conf_threshold,
        }
        stats_path = self.detection_dir / 'detection_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"  ✓ Step 1完成: {detection_frames_count}帧检测到物体")
        print(f"    输出: {detections_path.name}")
        
        return all_detections
    
    # =========================================================================
    # STEP 1.5: 同帧内物体分割合并 (处理YOLO把一个物体分成两个的问题)
    # =========================================================================
    
    def merge_fragmented_objects_in_frame(self, all_detections, same_class_distance_threshold=200):
        """在同一帧内，合并被分割的同类物体
        
        原理：
        - YOLO有时会把一个物体检测成多个（比如摩托车的前后部分）
        - 在同一帧内，如果两个物体：
          1. 类别相同（都是motorcycle）
          2. 中心距离 < same_class_distance_threshold (像素)
          3. 则认为是同一物体被分割，合并它们
        - 保留置信度更高的那个，删除置信度低的
        """
        print(f"\n【Step 1.5: 同帧内物体分割合并】")
        print(f"  ℹ️  在每一帧内检测和合并被分割的同类物体")
        
        merged_count = 0
        
        for frame_data in all_detections:
            frame = frame_data['frame']
            objects = frame_data['objects']
            
            # 记录哪些物体应该被删除（因为被合并了）
            to_remove = set()
            
            # 检查所有物体对
            for i, obj1 in enumerate(objects):
                if i in to_remove:
                    continue
                
                for j, obj2 in enumerate(objects):
                    if j <= i or j in to_remove:
                        continue
                    
                    # 检查是否是同类别且距离近
                    if obj1['class'] == obj2['class']:
                        x1, y1 = obj1['bbox_xywh'][0], obj1['bbox_xywh'][1]
                        x2, y2 = obj2['bbox_xywh'][0], obj2['bbox_xywh'][1]
                        
                        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        
                        if distance < same_class_distance_threshold:
                            # 合并：保留置信度高的，删除置信度低的
                            if obj1['conf'] >= obj2['conf']:
                                to_remove.add(j)
                                merged_count += 1
                                class_name = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
                                             4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'}.get(obj1['class'], f"class_{obj1['class']}")
                                print(f"  🔀 合并: Frame {frame:03d} - 两个 {class_name} (ID {obj1['track_id']}, ID {obj2['track_id']}) 距离 {distance:.1f}px < {same_class_distance_threshold}px")
                            else:
                                to_remove.add(i)
                                merged_count += 1
                                class_name = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
                                             4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'}.get(obj2['class'], f"class_{obj2['class']}")
                                print(f"  🔀 合并: Frame {frame:03d} - 两个 {class_name} (ID {obj2['track_id']}, ID {obj1['track_id']}) 距离 {distance:.1f}px < {same_class_distance_threshold}px")
            
            # 删除被合并的物体
            frame_data['objects'] = [obj for i, obj in enumerate(objects) if i not in to_remove]
        
        print(f"  ✓ Step 1.5 完成: 合并了 {merged_count} 个同帧内的分割物体")
        
        return all_detections
    
    # =========================================================================
    # =========================================================================
    
    def build_trajectories(self, all_detections):
        """Step 2: 构建轨迹 (像素空间 + 世界坐标，px/s + m/s)
        
        输入: 原始检测结果
        输出: 完整轨迹 (按 track_id 组织，同时包含像素和世界坐标)
        
        Option B: 在轨迹构建时就进行Homography转换，后续直接使用世界坐标
        """
        print(f"\n【Step 2: 轨迹构建 (像素空间 + 世界坐标)】")
        
        # 按 track_id 组织轨迹
        tracks = {}
        
        for frame_data in all_detections:
            for obj in frame_data['objects']:
                track_id = obj['track_id']
                
                if track_id not in tracks:
                    tracks[track_id] = []
                
                # 获取像素坐标
                center_x_px = obj['bbox_xywh'][0]
                center_y_px = obj['bbox_xywh'][1]
                
                # 转换到世界坐标 (如果有Homography矩阵)
                center_x_world = center_x_px
                center_y_world = center_y_px
                if self.H is not None:
                    pts_px = np.array([[center_x_px, center_y_px]], dtype=np.float32)
                    pts_world = cv2.perspectiveTransform(pts_px.reshape(1, 1, 2), self.H)
                    center_x_world = pts_world[0, 0, 0]
                    center_y_world = pts_world[0, 0, 1]
                
                # 轨迹点 (像素空间 + 世界坐标)
                track_point = {
                    'frame': frame_data['frame'],
                    'time': frame_data['time'],
                    'class': obj['class'],
                    'conf': obj['conf'],
                    # 像素坐标
                    'center_x': float(center_x_px),
                    'center_y': float(center_y_px),
                    # 世界坐标 (Option B新增) - 转换为Python float以便JSON序列化
                    'center_x_world': float(center_x_world),
                    'center_y_world': float(center_y_world),
                }
                
                tracks[track_id].append(track_point)
        
        # 计算每个轨迹的速度信息 (px/s 和 m/s)
        for track_id, track_points in tracks.items():
            track_points.sort(key=lambda p: p['frame'])
            
            if len(track_points) >= 2:
                for i in range(1, len(track_points)):
                    prev = track_points[i-1]
                    curr = track_points[i]
                    
                    dt = curr['time'] - prev['time']
                    if dt > 0:
                        # 像素空间速度
                        dx = curr['center_x'] - prev['center_x']
                        dy = curr['center_y'] - prev['center_y']
                        curr['vx'] = dx / dt  # px/s
                        curr['vy'] = dy / dt  # px/s
                        curr['speed'] = np.sqrt(dx**2 + dy**2) / dt  # px/s
                        
                        # 世界坐标速度 (Option B新增)
                        dx_world = curr['center_x_world'] - prev['center_x_world']
                        dy_world = curr['center_y_world'] - prev['center_y_world']
                        curr['vx_world'] = dx_world / dt  # m/s
                        curr['vy_world'] = dy_world / dt  # m/s
                        curr['speed_world'] = np.sqrt(dx_world**2 + dy_world**2) / dt  # m/s
                    else:
                        curr['vx'] = 0.0
                        curr['vy'] = 0.0
                        curr['speed'] = 0.0
                        curr['vx_world'] = 0.0
                        curr['vy_world'] = 0.0
                        curr['speed_world'] = 0.0
                
                track_points[0]['vx'] = 0.0
                track_points[0]['vy'] = 0.0
                track_points[0]['speed'] = 0.0
                track_points[0]['vx_world'] = 0.0
                track_points[0]['vy_world'] = 0.0
                track_points[0]['speed_world'] = 0.0
        
        # 保存轨迹
        tracks_path = self.trajectory_dir / 'tracks.json'
        with open(tracks_path, 'w') as f:
            json.dump(tracks, f, indent=2)
        
        # 生成轨迹统计
        stats = {
            'total_tracks': len(tracks),
            'track_lengths': {str(tid): len(points) for tid, points in tracks.items()},
            'coordinate_system': 'pixel + world',
            'velocity_unit': 'px/s (pixel) + m/s (world)',
        }
        stats_path = self.trajectory_dir / 'track_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"  ✓ Step 2完成: {len(tracks)}条轨迹 (像素空间 + 世界坐标)")
        print(f"    坐标系统: 像素 + 世界坐标 (已在Step 2转换)")
        print(f"    速度单位: px/s (像素) + m/s (世界)")
        print(f"    输出: {tracks_path.name}")
        
        return tracks
    
    # =========================================================================
    # STEP 2.4: 轨迹间断检测 (识别出现→消失→重新出现的可疑轨迹)
    # =========================================================================
    
    def detect_discontinuous_tracks(self, all_detections, max_gap_frames=None):
        """检测轨迹间断 (出现→消失→重新出现)
        
        原理：
        - 同一个 Track ID 在时间序列中出现了间断
        - 比如 Track ID 13 在 Frame 82-91 出现，Frame 92-93 消失，Frame 94 重新出现
        - 这很不合理，除非物体真的离开了视野（极少见）
        - 更可能是追踪失败或误检导致的幽灵轨迹
        
        注意：考虑 frame skipping 的影响
        - 如果用 --skip-frames 3，检测的是每隔3帧的情况
        - 所以"间断"的定义应该相对宽松
        
        返回：
        - suspicious_track_ids: 包含真正的间断的 Track ID 集合
        """
        print(f"\n【Step 2.4: 轨迹间断检测】")
        print(f"  ℹ️  检测轨迹间断（出现→消失→重新出现，考虑frame skipping）")
        
        # 如果没有指定，根据 skip_frames 设置默认值
        if max_gap_frames is None:
            max_gap_frames = self.skip_frames * 3  # 允许最多 skip_frames*3 的间断
        
        # 为每个 Track ID 记录它出现的所有帧
        track_frames = {}  # {track_id: [frame_nums]}
        
        for frame_data in all_detections:
            frame = frame_data['frame']
            for obj in frame_data['objects']:
                track_id = obj['track_id']
                if track_id not in track_frames:
                    track_frames[track_id] = []
                track_frames[track_id].append(frame)
        
        # 检查每个轨迹的连续性
        suspicious_tracks = {}  # {track_id: 'gap_info'}
        
        for track_id, frames in track_frames.items():
            frames = sorted(set(frames))  # 去重并排序
            
            if len(frames) < 2:
                continue
            
            # 检查是否有真正的间断（不是因为 frame skipping 导致的）
            gaps = []
            for i in range(1, len(frames)):
                gap = frames[i] - frames[i-1]
                # 只有间断 > skip_frames 时才算真正的间断
                if gap > self.skip_frames + 1:  # 允许1帧的偏差
                    gaps.append((frames[i-1], frames[i], gap))
            
            if gaps:
                # 有真正间断的轨迹
                suspicious_tracks[track_id] = {
                    'total_frames': len(frames),
                    'first_frame': frames[0],
                    'last_frame': frames[-1],
                    'gaps': gaps
                }
        
        # 打印可疑轨迹信息
        if suspicious_tracks:
            print(f"  ⚠️  检测到 {len(suspicious_tracks)} 条有真正间断的轨迹（可能是误检或追踪失败）:")
            for track_id, info in sorted(suspicious_tracks.items()):
                print(f"     - Track ID {track_id}: {info['total_frames']}帧 ({info['first_frame']}-{info['last_frame']})")
                for gap_start, gap_end, gap_size in info['gaps']:
                    print(f"       └─ 间断: Frame {gap_start} → Frame {gap_end} (间隔 {gap_size} 帧，超过允许的 {self.skip_frames + 1} 帧)")
        else:
            print(f"  ✓ 没有检测到真正的间断轨迹（考虑了 skip_frames={self.skip_frames} 的影响）")
        
        print(f"  ℹ️  注意: 间断阈值 = {self.skip_frames + 1} 帧（基于 skip_frames 参数）")
        
        return suspicious_tracks
    
    # =========================================================================
    # STEP 2.5: 轨迹连续性过滤 (排除短轨迹误检)
    # =========================================================================
    
    def filter_short_tracks(self, all_detections, min_track_length=3):
        """过滤短轨迹 (可能是 YOLO 误检)
        
        如果一个 Track ID 只出现在少于 min_track_length 帧中，
        则认为是误检，将其从检测结果中移除。
        
        原理：
        - 真实物体应该在连续的多个帧中被检测到
        - 如果物体突然出现又消失，通常是 YOLO 的误检
        """
        print(f"\n【Step 2.5: 轨迹连续性过滤】")
        
        # 统计每个 Track ID 的出现帧数
        track_lengths = {}
        for frame_data in all_detections:
            for obj in frame_data['objects']:
                track_id = obj['track_id']
                track_lengths[track_id] = track_lengths.get(track_id, 0) + 1
        
        # 找出短轨迹 (可能是误检)
        short_tracks = {tid: length for tid, length in track_lengths.items() if length < min_track_length}
        valid_tracks = {tid: length for tid, length in track_lengths.items() if length >= min_track_length}
        
        print(f"  轨迹长度统计:")
        print(f"    - 总轨迹: {len(track_lengths)}")
        print(f"    - 有效轨迹 (>= {min_track_length}帧): {len(valid_tracks)}")
        print(f"    - 短轨迹/误检 (< {min_track_length}帧): {len(short_tracks)}")
        
        if short_tracks:
            print(f"  🗑️  移除的短轨迹:")
            for tid, length in sorted(short_tracks.items(), key=lambda x: x[1]):
                print(f"     - Track ID {tid}: {length} 帧")
        
        # 过滤检测结果，移除短轨迹中的物体
        filtered_detections = []
        for frame_data in all_detections:
            new_frame_data = frame_data.copy()
            new_frame_data['objects'] = [
                obj for obj in frame_data['objects']
                if obj['track_id'] in valid_tracks
            ]
            filtered_detections.append(new_frame_data)
        
        print(f"  ✓ Step 2.5 完成: 移除了 {len(short_tracks)} 条短轨迹")
        
        return filtered_detections
    
    # =========================================================================
    # STEP 2.6: Track ID 重连检测 (同一物体多个ID合并)
    # =========================================================================
    
    def merge_fragmented_tracks(self, all_detections, max_gap_frames=2, max_distance_pixels=100):
        """合并被断开的Track ID (同一物体追踪失败导致的重复ID)
        
        原理：
        - 如果 Track ID A 在某帧消失
        - 然后在 max_gap_frames 帧内，一个新的 Track ID B 出现
        - 且两个ID的物体中心距离 < max_distance_pixels
        - 则认为是同一物体，应该合并ID
        
        这特别适合解决摩托车被分成ID13和ID15的问题。
        """
        print(f"\n【Step 2.6: Track ID 重连检测】")
        print(f"  ℹ️  检测并合并被断开的轨迹 (消失<{max_gap_frames}帧后重新出现)")
        
        # 第一步：为每个Track ID统计最后出现的帧和位置
        track_last_appearance = {}  # {track_id: {'frame': f, 'x': x, 'y': y, 'class': c}}
        
        for frame_data in all_detections:
            frame = frame_data['frame']
            for obj in frame_data['objects']:
                track_id = obj['track_id']
                track_last_appearance[track_id] = {
                    'frame': frame,
                    'x': obj['bbox_xywh'][0],
                    'y': obj['bbox_xywh'][1],
                    'class': obj['class']
                }
        
        # 第二步：寻找可能断开的Track ID对
        merge_map = {}  # {old_id: new_id} 映射
        merged_count = 0
        
        track_ids = sorted(track_last_appearance.keys())
        
        for i, track_a in enumerate(track_ids):
            if track_a in merge_map:
                continue  # 已经被合并过了
            
            last_a = track_last_appearance[track_a]
            frame_a = last_a['frame']
            x_a, y_a = last_a['x'], last_a['y']
            class_a = last_a['class']
            
            # 查找后续出现的Track ID
            for track_b in track_ids[i+1:]:
                if track_b in merge_map:
                    continue
                
                # 找track_b的第一次出现
                first_b_frame = None
                first_b_pos = None
                first_b_class = None
                
                for frame_data in all_detections:
                    for obj in frame_data['objects']:
                        if obj['track_id'] == track_b:
                            if first_b_frame is None:
                                first_b_frame = frame_data['frame']
                                first_b_pos = (obj['bbox_xywh'][0], obj['bbox_xywh'][1])
                                first_b_class = obj['class']
                            break
                
                if first_b_frame is None:
                    continue
                
                # 检查是否满足重连条件
                frame_gap = first_b_frame - frame_a
                if 1 <= frame_gap <= max_gap_frames:  # 中间有间隔但不超过max_gap
                    distance = np.sqrt((first_b_pos[0] - x_a)**2 + (first_b_pos[1] - y_a)**2)
                    
                    if distance < max_distance_pixels and class_a == first_b_class:
                        # 认为是同一物体，应该合并
                        merge_map[track_b] = track_a
                        merged_count += 1
                        print(f"  🔗 合并: Track ID {track_b} (首次Frame {first_b_frame}) → ID {track_a} (末次Frame {frame_a})")
                        print(f"     间隔: {frame_gap}帧, 距离: {distance:.1f}px, 类别: {class_a}")
        
        # 第三步：应用合并到所有检测结果
        if merge_map:
            for frame_data in all_detections:
                for obj in frame_data['objects']:
                    if obj['track_id'] in merge_map:
                        old_id = obj['track_id']
                        new_id = merge_map[old_id]
                        obj['track_id'] = new_id
        
        print(f"  ✓ Step 2.6 完成: 合并了 {merged_count} 个断开的Track ID")
        
        return all_detections
    
    # =========================================================================
    # STEP 3: 关键帧检测 (接近事件)
    # =========================================================================

    def extract_key_frames(self, all_detections, tracks, world_distance_threshold=2.0, debug_threshold=5.0):
        """Step 3: 关键帧检测 (接近事件) - 基于Homography世界坐标
        
        流程说明:
        1. Step 2已在轨迹中使用Homography转换得到世界坐标 (center_x_world, center_y_world)
        2. Step 3使用这些世界坐标计算物体间距离，检测接近事件
        3. 通过空间验证过滤：确保物体在Homography标定区域内 (X[-1.75,1.75]m, Y[0,25]m)
           - 若物体世界坐标超出范围，说明Homography变换可能不可靠，应过滤
        4. 保存通过阈值的接近事件作为关键帧
        
        参数:
        - all_detections: 原始检测结果 (用于保存关键帧图像)
        - tracks: Step 2返回的轨迹信息 (已包含Homography变换的world坐标)
        - world_distance_threshold: 关键帧检测阈值（默认 4.5 米）
        """
        print(f"\n【Step 3: 关键帧检测 (基于Homography世界坐标)】")
        print(f"  ℹ️  使用Step 2中Homography变换的世界坐标进行距离计算和空间验证")
        
        proximity_events = []
        all_proximity_pairs = []
        
        # 物体类别映射
        class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
                      4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'}
        
        # 建立track_id -> 轨迹数据的映射，方便查找
        track_map = {}
        for track_id, track_points in tracks.items():
            for point in track_points:
                frame = point['frame']
                if frame not in track_map:
                    track_map[frame] = {}
                track_map[frame][int(track_id)] = point
        
        # 遍历每一帧，检测物体对之间的世界坐标距离
        for frame_data in all_detections:
            frame = frame_data['frame']
            if frame not in track_map or len(track_map[frame]) < 2:
                continue
            
            objects = frame_data['objects']
            frame_tracks = track_map[frame]
            
            # 检查所有物体对
            for i in range(len(objects)):
                for j in range(i+1, len(objects)):
                    obj1 = objects[i]
                    obj2 = objects[j]
                    
                    tid1 = obj1['track_id']
                    tid2 = obj2['track_id']
                    
                    # 获取轨迹中保存的世界坐标
                    if tid1 not in frame_tracks or tid2 not in frame_tracks:
                        continue
                    
                    track1 = frame_tracks[tid1]
                    track2 = frame_tracks[tid2]
                    
                    # 获取世界坐标 (Option B: 直接使用保存的world坐标)
                    x1_world = track1['center_x_world']
                    y1_world = track1['center_y_world']
                    x2_world = track2['center_x_world']
                    y2_world = track2['center_y_world']
                    
                    # ✨ 新增: 验证两个对象都在标定区域内
                    # 标定区域范围: X [-1.75, 1.75] m, Y [0, 25] m
                    world_x_min, world_x_max = -1.75, 1.75
                    world_y_min, world_y_max = 0.0, 25.0
                    world_margin = 0.3  # 允许轻微超出范围
                    
                    # 检查两个物体是否都在有效范围内
                    obj1_valid = (world_x_min - world_margin <= x1_world <= world_x_max + world_margin and
                                  world_y_min - world_margin <= y1_world <= world_y_max + world_margin)
                    obj2_valid = (world_x_min - world_margin <= x2_world <= world_x_max + world_margin and
                                  world_y_min - world_margin <= y2_world <= world_y_max + world_margin)
                    
                    if not (obj1_valid and obj2_valid):
                        # 跳过超出标定区域的对象对
                        continue
                    
                    # 获取像素坐标用于图像保存
                    x1_px = track1['center_x']
                    y1_px = track1['center_y']
                    x2_px = track2['center_x']
                    y2_px = track2['center_y']
                    distance_pixel = np.sqrt((x2_px-x1_px)**2 + (y2_px-y1_px)**2)
                    
                    # 使用多锚点碰撞检测分析器
                    # ⚠️ 注意: 多锚点分析虽然已集成，但为了性能考虑，暂时禁用
                    # 如需启用，设置 USE_MULTI_ANCHOR=True
                    USE_MULTI_ANCHOR = False
                    
                    if USE_MULTI_ANCHOR:
                        try:
                            # 获取锚点
                            anchors1 = self._get_object_anchors(obj1['class'], obj1['bbox_xywh'])
                            anchors2 = self._get_object_anchors(obj2['class'], obj2['bbox_xywh'])
                            
                            # 获取速度信息（使用世界坐标速度 m/s，而不是像素速度）
                            # 世界坐标速度已经考虑了跳帧，单位为 m/s
                            vx1 = track1.get('vx_world', 0.0)
                            vy1 = track1.get('vy_world', 0.0)
                            vx2 = track2.get('vx_world', 0.0)
                            vy2 = track2.get('vy_world', 0.0)
                            
                            # 创建碰撞分析器
                            analyzer = CollisionAnalyzer(pixel_per_meter=self.pixel_per_meter)
                            
                            # 执行碰撞分析
                            collision_result = analyzer.analyze(
                                obj1=obj1,
                                obj2=obj2,
                                obj1_anchors=anchors1,
                                obj2_anchors=anchors2,
                                obj1_velocity=(vx1, vy1),
                                obj2_velocity=(vx2, vy2),
                                obj1_track=track1,
                                obj2_track=track2,
                                H=self.H
                            )
                            
                            # 使用多锚点距离
                            distance_meters = collision_result.min_distance
                            closest_parts = (collision_result.object1_part, collision_result.object2_part)
                            ttc = collision_result.ttc
                            risk_level = collision_result.risk_level
                            
                        except Exception as e:
                            # 如果多锚点分析失败，回退到中心点距离
                            print(f"  ⚠️  多锚点分析异常: {e}，使用中心点距离")
                            distance_meters = np.sqrt((x2_world-x1_world)**2 + (y2_world-y1_world)**2)
                            closest_parts = ('center', 'center')
                            ttc = None
                            risk_level = 'UNKNOWN'
                    else:
                        # 使用中心点距离（与之前兼容）
                        distance_meters = np.sqrt((x2_world-x1_world)**2 + (y2_world-y1_world)**2)
                        closest_parts = ('center', 'center')
                        ttc = None
                        risk_level = 'UNKNOWN'
                    
                    class1_name = class_names.get(obj1['class'], f"class_{obj1['class']}")
                    class2_name = class_names.get(obj2['class'], f"class_{obj2['class']}")
                    
                    # 记录所有 < debug_threshold 的检测，用于调试
                    if distance_meters < debug_threshold:
                        all_proximity_pairs.append({
                            'frame': frame,
                            'class_1': class1_name,
                            'class_2': class2_name,
                            'distance_meters': distance_meters,
                            'track_ids': [tid1, tid2]
                        })
                    
                    # 检查是否为接近事件 (使用世界距离阈值)
                    if distance_meters < world_distance_threshold:
                        event = {
                            'frame': frame,
                            'time': frame_data['time'],
                            'track_id_1': tid1,
                            'track_id_2': tid2,
                            'class_1': class1_name,
                            'class_2': class2_name,
                            'distance_pixel': float(distance_pixel),
                            'distance_meters': float(distance_meters),
                            'object_classes': (obj1['class'], obj2['class']),
                            'center_1_px': [float(x1_px), float(y1_px)],
                            'center_2_px': [float(x2_px), float(y2_px)],
                            'center_1_world': [float(x1_world), float(y1_world)],
                            'center_2_world': [float(x2_world), float(y2_world)],
                            # 添加bbox大小信息用于后续的bbox相似度过滤
                            'bbox_1': obj1['bbox_xywh'],  # [x, y, w, h]
                            'bbox_2': obj2['bbox_xywh'],  # [x, y, w, h]
                            'positions': {
                                'obj1': {'x': x1_px, 'y': y1_px},
                                'obj2': {'x': x2_px, 'y': y2_px}
                            },
                            'positions_world': {
                                'obj1': {'x': x1_world, 'y': y1_world},
                                'obj2': {'x': x2_world, 'y': y2_world}
                            },
                            # 多锚点碰撞分析信息
                            'multi_anchor': {
                                'closest_parts': closest_parts,
                                'risk_level': risk_level,
                                'ttc': ttc
                            }
                        }
                        proximity_events.append(event)
                        
                        # 保存关键帧图像
                        frame_img_path = self.keyframe_dir / f"keyframe_{frame:04d}_ID{tid1}_ID{tid2}.jpg"
                        self.save_keyframe_with_distance(self.video_path, frame, frame_img_path, event)
        
        # 保存接近事件
        events_path = self.keyframe_dir / 'proximity_events.json'
        with open(events_path, 'w') as f:
            json.dump(proximity_events, f, indent=2)
        
        print(f"  ✓ Step 3完成: {len(proximity_events)}个关键帧 (< {world_distance_threshold}m)")
        print(f"    总检测到的近距离对: {len(all_proximity_pairs)}个 (< {debug_threshold}m)")
        print(f"    距离阈值: {world_distance_threshold}米 (世界坐标)")
        print(f"    坐标来源: Step 2保存的轨迹数据 (已转换)")
        print(f"    输出: {events_path.name}")
        
        # 打印被排除的事件
        excluded_count = len(all_proximity_pairs) - len(proximity_events)
        if excluded_count > 0:
            print(f"\n  ℹ️  被排除的接近事件 ({world_distance_threshold}-{debug_threshold}m):")
            for pair in all_proximity_pairs:
                if pair['distance_meters'] >= world_distance_threshold:
                    print(f"     - Frame {pair['frame']:03d}: Track {pair['track_ids'][0]}({pair['class_1']}) + Track {pair['track_ids'][1]}({pair['class_2']}) = {pair['distance_meters']:.2f}m")
        
        # 调试输出：检查输入数据
        total_object_pairs = 0
        for frame_data in all_detections:
            if len(frame_data['objects']) >= 2:
                total_object_pairs += len(frame_data['objects']) * (len(frame_data['objects']) - 1) // 2
        print(f"\n  ℹ️  调试信息: 检查了 {total_object_pairs} 个物体对，其中 {len(all_proximity_pairs)} 个距离 < {debug_threshold}m")
        
        return proximity_events
    
    # =========================================================================
    # STEP 3.1: 获取物体的锚点
    # =========================================================================
    
    def _shrink_bbox(self, bbox_xywh, shrink_ratio=0.8):
        """缩小bounding box - 从中心往外缩小到原来的比例
        
        Args:
            bbox_xywh: [x_center, y_center, width, height]
            shrink_ratio: 缩小比例 (0.8 = 保留原来的80%)
        
        Returns:
            缩小后的 bbox [x_center, y_center, width*shrink_ratio, height*shrink_ratio]
        """
        x, y, w, h = bbox_xywh
        new_w = w * shrink_ratio
        new_h = h * shrink_ratio
        return [x, y, new_w, new_h]
    
    def _get_object_anchors(self, class_id, bbox_xywh):
        """根据物体类别获取相应的锚点
        
        Args:
            class_id: YOLO物体类别ID (0=person, 1=bicycle, 2=car, 3=motorcycle, etc.)
            bbox_xywh: 边界框 [x_center, y_center, width, height]
        
        Returns:
            dict: {anchor_name: (x, y), ...}
        """
        # 缩小bounding box到原来的80%，确保锚点在物体内
        bbox_xywh = self._shrink_bbox(bbox_xywh, shrink_ratio=0.8)
        
        try:
            if class_id == 0:  # person
                return PedestrianAnchors.get_anchors(bbox_xywh)
            elif class_id == 2:  # car
                return VehicleAnchors.get_anchors(bbox_xywh, class_id)
            elif class_id == 1:  # bicycle
                return BicycleAnchors.get_anchors(bbox_xywh)
            elif class_id == 3:  # motorcycle
                return MotorcycleAnchors.get_anchors(bbox_xywh)
            elif class_id == 5:  # bus
                return VehicleAnchors.get_anchors(bbox_xywh, class_id)
            elif class_id == 7:  # truck
                return VehicleAnchors.get_anchors(bbox_xywh, class_id)
            else:
                # 其他类别使用通用锚点
                return VehicleAnchors.get_anchors(bbox_xywh, class_id)
        except Exception as e:
            print(f"  ⚠️  获取锚点失败 (class_id={class_id}): {e}")
            # 降级方案：返回简单的中心锚点
            return {'center': (bbox_xywh[0], bbox_xywh[1])}
    
    # =========================================================================
    # STEP 3.5: 同类别物体误检过滤
    # =========================================================================
    
    def filter_same_class_false_positives(self, proximity_events, same_class_distance_threshold=0.3):
        """过滤同类别物体的误检 + 跨帧同一物体的误分割
        
        过滤条件：
        1. 极近距离 (< 0.1m) → 同一物体两部分
        2. 不合理的类别组合 + 距离稳定 (std < 0.5) → 同速不可能
        3. 断断续续出现的Track ID对 + 距离近 (< 2.0m) → 同一物体被误分割
        4. 都是汽车类型 + 距离 < 0.5m → 同一车辆的不同部分（如卡车头和身体）
        """
        print(f"\n【Step 3.5: 物体误检过滤 (智能策略)】")
        
        # 不合理的类别组合（不可能同时出现且同速运动）
        illogical_class_combinations = [
            ('person', 'motorcycle'),
            ('person', 'car'),
            ('person', 'truck'),
            ('person', 'bus'),
            ('bicycle', 'motorcycle'),
            ('bicycle', 'car'),
        ]
        
        # 定义汽车类型
        vehicle_types = {'car', 'truck', 'bus', 'motorcycle'}
        
        # 首先分析Track ID对的出现情况
        track_pair_analysis = {}
        for event in proximity_events:
            tid1, tid2 = event['track_id_1'], event['track_id_2']
            pair_key = tuple(sorted([tid1, tid2]))
            
            if pair_key not in track_pair_analysis:
                track_pair_analysis[pair_key] = {
                    'events': [],
                    'frames': [],
                    'distances': [],
                    'classes': None
                }
            
            track_pair_analysis[pair_key]['events'].append(event)
            track_pair_analysis[pair_key]['frames'].append(event['frame'])
            track_pair_analysis[pair_key]['distances'].append(event['distance_meters'])
            track_pair_analysis[pair_key]['classes'] = (event['class_1'], event['class_2'])
        
        # 识别"断断续续出现"的Track ID对
        suspicious_discontinuous_pairs = set()
        for pair, info in track_pair_analysis.items():
            frames = sorted(info['frames'])
            distances = info['distances']
            avg_distance = sum(distances) / len(distances)
            
            # 检查是否有明显的间隔（出现-消失-再出现）
            has_gap = False
            for i in range(len(frames) - 1):
                if frames[i+1] - frames[i] > 3:  # 间隔 > 3帧
                    has_gap = True
                    break
            
            # 如果断断续续出现且距离近，标记为可疑
            if has_gap and avg_distance < 2.0:
                suspicious_discontinuous_pairs.add(pair)
        
        # 现在进行逐个事件的过滤
        filtered_events = []
        filtered_count = 0
        filter_reasons = []
        
        for event in proximity_events:
            class_1 = event['class_1']
            class_2 = event['class_2']
            distance = event['distance_meters']
            frame = event['frame']
            tid1, tid2 = event['track_id_1'], event['track_id_2']
            pair_key = tuple(sorted([tid1, tid2]))
            
            reason = None
            
            # 条件1: 极近距离 (< 0.1m) → 同一物体的两部分
            if distance < 0.1:
                reason = f"极近 ({distance:.3f}m < 0.1m)"
                filtered_count += 1
                filter_reasons.append((frame, tid1, tid2, class_1, class_2, distance, reason))
                continue
            
            # 条件1.5: bbox大小相似度过滤 → 同一物体被分割成多个Track ID
            # 如果两个物体的bbox大小非常相似（比如都是 500x300），且距离近
            # 这通常表示YOLO把同一物体分割成了两个Track ID
            bbox_1 = event.get('bbox_1')
            bbox_2 = event.get('bbox_2')
            if bbox_1 and bbox_2:
                w1, h1 = bbox_1[2], bbox_1[3]
                w2, h2 = bbox_2[2], bbox_2[3]
                
                # 计算bbox大小的相似度 (面积相对差异)
                area_1 = w1 * h1
                area_2 = w2 * h2
                area_diff_ratio = abs(area_1 - area_2) / max(area_1, area_2)
                
                # 如果面积相差 < 20% 且都是汽车类型，则认为可能是同一物体
                if (area_diff_ratio < 0.2 and 
                    class_1 in vehicle_types and class_2 in vehicle_types and
                    distance < 2.0):  # 距离 < 2.0m
                    reason = f"bbox相似 (面积相差{area_diff_ratio*100:.1f}% < 20%) + 距离{distance:.3f}m"
                    filtered_count += 1
                    filter_reasons.append((frame, tid1, tid2, class_1, class_2, distance, reason))
                    continue
            
            # 条件4: 都是汽车类型 + 距离 < 0.5m → 同一车辆的不同部分
            if (class_1 in vehicle_types and class_2 in vehicle_types) and distance < 0.5:
                reason = f"都是汽车类型 ({class_1}+{class_2}, 距离{distance:.3f}m < 0.5m)"
                filtered_count += 1
                filter_reasons.append((frame, tid1, tid2, class_1, class_2, distance, reason))
                continue
            
            # 条件2: 不合理的类别组合 + 距离稳定 → 同速不可能
            class_pair = tuple(sorted([class_1, class_2]))
            if class_pair in [tuple(sorted(p)) for p in illogical_class_combinations]:
                pair_info = track_pair_analysis[pair_key]
                std_distance = np.std(pair_info['distances'])
                
                if std_distance < 0.5:  # 距离非常稳定 = 同速 = 不可能
                    reason = f"不合理类别组合+同速 ({class_1}+{class_2}, std={std_distance:.2f}m)"
                    filtered_count += 1
                    filter_reasons.append((frame, tid1, tid2, class_1, class_2, distance, reason))
                    continue
            
            # 条件3: 断断续续出现的Track ID对 + 距离近 → 同一物体被误分割
            if pair_key in suspicious_discontinuous_pairs:
                pair_info = track_pair_analysis[pair_key]
                reason = f"断断续续出现({len(pair_info['frames'])}帧) + 距离近"
                filtered_count += 1
                filter_reasons.append((frame, tid1, tid2, class_1, class_2, distance, reason))
                continue
            
            # 保留这个事件
            filtered_events.append(event)
        
        # 打印过滤详情
        if filter_reasons:
            print(f"  🗑️  过滤的事件:")
            for frame, tid1, tid2, class1, class2, dist, reason in filter_reasons[:20]:  # 只打印前20个
                print(f"      Frame {frame}: {class1}({tid1}) + {class2}({tid2}) = {dist:.3f}m ({reason})")
            if len(filter_reasons) > 20:
                print(f"      ... 还有 {len(filter_reasons)-20} 个")
        
        print(f"  ✓ 过滤完成: 排除了 {filtered_count} 个误检, 保留 {len(filtered_events)} 个事件")
        print(f"    条件1: 距离 < 0.1m")
        print(f"    条件1.5: bbox面积相似 (< 20% 差异) + 汽车类型 + 距离 < 2.0m")
        print(f"    条件4: 都是汽车类型 (car/truck/bus/motorcycle等) + 距离 < 0.5m")
        print(f"    条件2: 不合理类别组合 (person/motorcycle等) + 距离稳定 (std < 0.5m)")
        print(f"    条件3: Track ID对断断续续出现 + 平均距离 < 2.0m")
        
        # 保存过滤后的事件
        events_path = self.keyframe_dir / 'proximity_events_filtered.json'
        with open(events_path, 'w') as f:
            json.dump({
                'total_detected': len(proximity_events),
                'false_positives_filtered': filtered_count,
                'valid_events': len(filtered_events),
                'events': filtered_events
            }, f, indent=2)
        
        return filtered_events
    
    # =========================================================================
    # STEP 3.6: 清理被过滤的关键帧图片
    # =========================================================================
    
    def cleanup_filtered_keyframes(self, original_events, filtered_events):
        """删除被过滤掉的关键帧图片文件"""
        # 收集被保留的 keyframe 文件名
        kept_frames = set()
        for event in filtered_events:
            frame_id = event['frame']
            tid1 = event['track_id_1']
            tid2 = event['track_id_2']
            # 生成应该被保留的文件名
            filename = f"keyframe_{frame_id:04d}_ID{tid1}_ID{tid2}.jpg"
            kept_frames.add(filename)
        
        # 删除不在 kept_frames 中的 keyframe 文件
        for img_file in self.keyframe_dir.glob('keyframe_*.jpg'):
            if img_file.name not in kept_frames:
                try:
                    img_file.unlink()
                    print(f"  🗑️  删除关键帧图片: {img_file.name}")
                except Exception as e:
                    print(f"  ⚠️  删除图片失败 {img_file.name}: {e}")
    
    # =========================================================================
    # STEP 3.6: 多锚点碰撞分析 (仅关键帧) ✨ 新增功能
    # =========================================================================
    
    def analyze_keyframes_with_multi_anchor(self, proximity_events, all_detections, tracks):
        """Step 3.6: 对关键帧执行多锚点碰撞分析（仅在已确定为接近事件的帧上执行）
        
        这样可以大幅降低计算量：
        - Step 3: 用简单的中心点距离快速筛选接近事件
        - Step 3.6: 只对这些关键帧执行详细的多锚点分析
        
        Args:
            proximity_events: 从Step 3筛选出的接近事件
            all_detections: 所有检测结果
            tracks: 轨迹数据
        
        Returns:
            proximity_events: 增强后的事件（包含多锚点分析信息）
        """
        print(f"\n【Step 3.6: 多锚点碰撞分析 (仅关键帧)】")
        
        if not proximity_events:
            print(f"  ℹ️  无关键帧，跳过多锚点分析")
            return proximity_events
        
        # 建立track_id -> 轨迹数据的映射
        track_map = {}
        for track_id, track_points in tracks.items():
            for point in track_points:
                frame = point['frame']
                if frame not in track_map:
                    track_map[frame] = {}
                track_map[frame][int(track_id)] = point
        
        # 建立frame -> objects的映射
        detection_map = {}
        for frame_data in all_detections:
            detection_map[frame_data['frame']] = frame_data
        
        # 对每个关键帧事件执行多锚点分析
        analyzed_count = 0
        failed_frames = []
        
        for event in proximity_events:
            frame = event['frame']
            tid1 = event['track_id_1']
            tid2 = event['track_id_2']
            
            # 跳过已经有多锚点信息的
            if 'multi_anchor_detailed' in event:
                continue
            
            try:
                # 获取该帧的检测和轨迹数据
                if frame not in detection_map or frame not in track_map:
                    failed_frames.append((frame, tid1, tid2, "Frame/Track data not found"))
                    continue
                
                frame_data = detection_map[frame]
                frame_tracks = track_map[frame]
                
                # 查找两个物体
                obj1, obj2 = None, None
                track1_point, track2_point = None, None
                track1_history, track2_history = None, None
                
                for obj in frame_data['objects']:
                    if obj['track_id'] == tid1:
                        obj1 = obj
                        track1_point = frame_tracks.get(tid1)
                        # 获取完整的轨迹历史（用于计算速度和方向）
                        if tid1 in tracks:
                            track1_history = tracks[tid1]
                    elif obj['track_id'] == tid2:
                        obj2 = obj
                        track2_point = frame_tracks.get(tid2)
                        # 获取完整的轨迹历史
                        if tid2 in tracks:
                            track2_history = tracks[tid2]
                
                if obj1 is None or obj2 is None or track1_point is None or track2_point is None:
                    reason = []
                    if obj1 is None: reason.append(f"obj1 not found")
                    if obj2 is None: reason.append(f"obj2 not found")
                    if track1_point is None: reason.append(f"track1_point not found")
                    if track2_point is None: reason.append(f"track2_point not found")
                    failed_frames.append((frame, tid1, tid2, ", ".join(reason)))
                    continue
                
                # 获取锚点
                anchors1 = self._get_object_anchors(obj1['class'], obj1['bbox_xywh'])
                anchors2 = self._get_object_anchors(obj2['class'], obj2['bbox_xywh'])
                
                # 获取速度信息（从该帧的轨迹点）
                vx1 = track1_point.get('vx', 0.0)
                vy1 = track1_point.get('vy', 0.0)
                vx2 = track2_point.get('vx', 0.0)
                vy2 = track2_point.get('vy', 0.0)
                
                # 执行多锚点碰撞分析
                analyzer = CollisionAnalyzer(pixel_per_meter=self.pixel_per_meter)
                collision_result = analyzer.analyze(
                    obj1=obj1,
                    obj2=obj2,
                    obj1_anchors=anchors1,
                    obj2_anchors=anchors2,
                    obj1_velocity=(vx1, vy1),
                    obj2_velocity=(vx2, vy2),
                    obj1_track=track1_history,  # 传入完整的轨迹历史
                    obj2_track=track2_history,  # 传入完整的轨迹历史
                    H=self.H
                )
                
                # 添加详细的多锚点分析结果
                event['multi_anchor_detailed'] = collision_result.to_dict()
                analyzed_count += 1
                
            except Exception as e:
                # 记录失败的帧
                import traceback
                error_msg = f"{type(e).__name__}: {str(e)}"
                failed_frames.append((frame, tid1, tid2, error_msg))
        
        # 报告分析结果
        if analyzed_count > 0:
            print(f"  ✓ 多锚点分析完成: {analyzed_count}/{len(proximity_events)}个关键帧")
        
        if failed_frames:
            print(f"  ⚠️  {len(failed_frames)}个关键帧分析失败:")
            for frame, tid1, tid2, reason in failed_frames:
                print(f"     - Frame {frame}: ID{tid1}+ID{tid2} ({reason})")
        else:
            print(f"  ⚠️  多锚点分析完成: 0/{len(proximity_events)}个关键帧 (无法获取锚点数据或发生错误)")
        
        # =================================================================
        # STEP 3.7: 多锚点距离过滤（仅保留距离 ≤ 1.0m 的高风险事件）
        # =================================================================
        print(f"\n【Step 3.7: 多锚点距离过滤 (≤1.0m)】")
        
        anchor_filtered_events = []
        for event in proximity_events:
            multi = event.get('multi_anchor_detailed', {})
            min_distance = multi.get('min_distance_meters', float('inf'))
            
            # 保留距离 ≤ 1.0m 的事件（高风险）
            if min_distance <= 1.0:
                anchor_filtered_events.append(event)
            else:
                frame = event['frame']
                tid1, tid2 = event['track_id_1'], event['track_id_2']
                print(f"  ⊗ 过滤 Frame {frame}: Track {tid1}+{tid2} (锚点距离={min_distance:.2f}m > 1.0m)")
        
        filtered_count = len(proximity_events) - len(anchor_filtered_events)
        print(f"  🔍 多锚点距离过滤: 排除 {filtered_count} 个事件")
        print(f"  ✓ Step 3.7完成: 保留 {len(anchor_filtered_events)} 个关键帧 (≤ 1.0m)")
        
        return anchor_filtered_events
    
    # =========================================================================
    # STEP 4: Homography 信息保存 (仅作元数据保存) ✨ Homography已在Step 2使用
    # =========================================================================
    
    def transform_key_frames_to_world(self, proximity_events):
        """Step 4: Homography 信息保存
        
        Warning: Homography transformation already completed in Step 2!
        - Step 2: Trajectory construction + Homography transform -> world coordinates
        - Step 3: Use world coordinates to detect keyframes
        - Step 4: Only save Homography metadata, no duplicate transform
        """
        print(f"\n【Step 4: Homography 信息保存】")
        print(f"  ℹ️  注意: 坐标变换已在Step 2中完成（使用Homography）")
        
        if self.H is None:
            print(f"  ⚠️  未加载Homography")
            return proximity_events
        
        # 直接返回事件（已包含世界坐标）
        # 保存 Homography 矩阵信息供参考
        trans_path = self.homography_dir / 'transformed_key_frames.json'
        with open(trans_path, 'w') as f:
            json.dump(proximity_events, f, indent=2)
        
        print(f"  ✓ Step 4完成: {len(proximity_events)}个关键帧信息已保存")
        print(f"    缩放因子: {self.pixel_per_meter:.2f} px/m")
        print(f"    坐标系统: 世界坐标 (已在Step 3变换)")
        print(f"    输出: {trans_path.name}")
        
        return proximity_events
    
    # =========================================================================
    # STEP 5: TTC 和 Event 分级
    # =========================================================================
    
    def analyze_collision_risk(self, transformed_events):
        """Step 5: TTC 计算和 Event 分级
        
        计算 TTC，分级事件 (L1/L2/L3)
        
         改进: 过滤同类别物体的极近接近事件
        - 如果两个物体都是同一类别（如两个car，两个motorcycle）
        - 且距离 < 0.5m，则可能是同一物体的误检
        - 标记为 'Filtered_SameClass' 并排除
        """
        print(f"\n【Step 5: 碰撞风险分析】")
        
        if not transformed_events:
            print(f"  ⚠️  没有关键帧，无法分析")
            return [], {0: 0, 1: 0, 2: 0, 3: 0}
        
        analyzed_events = []
        filtered_count = 0
        
        for event in transformed_events:
            analyzed = event.copy()
            
            # 在世界坐标中进行分级
            distance = event['distance_meters']
            class_1 = event.get('class_1', '')
            class_2 = event.get('class_2', '')
            
            # 🔍 检查是否为同类别物体的极近接近事件
            if class_1 == class_2 and distance < 0.5:
                # 同一类别 + 距离很近 = 可能是同一物体的不同部分
                analyzed['level'] = 0
                analyzed['level_name'] = 'Filtered_SameClass'
                analyzed['reason'] = f"Same class ({class_1}) with distance {distance:.3f}m < 0.5m - likely same object"
                filtered_count += 1
            # 分级标准 (米)
            elif distance < 0.5:
                analyzed['level'] = 1
                analyzed['level_name'] = 'Collision'
            elif distance < 1.5:
                analyzed['level'] = 2
                analyzed['level_name'] = 'Near Miss'
            else:
                analyzed['level'] = 3
                analyzed['level_name'] = 'Avoidance'
            
            analyzed_events.append(analyzed)
        
        # 统计
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for event in analyzed_events:
            level_counts[event['level']] += 1
        
        # 保存分析结果
        analysis_path = self.analysis_dir / 'collision_events.json'
        with open(analysis_path, 'w') as f:
            json.dump(analyzed_events, f, indent=2)
        
        print(f"  ✓ Step 5完成")
        print(f"    - Filtered (Same class, <0.5m): {level_counts[0]} 🚫")
        print(f"    - Level 1 (Collision, <0.5m): {level_counts[1]}")
        print(f"    - Level 2 (Near Miss, 0.5-1.5m): {level_counts[2]}")
        print(f"    - Level 3 (Avoidance, >1.5m): {level_counts[3]}")
        print(f"    输出: {analysis_path.name}")
        
        return analyzed_events, level_counts
    
    # =========================================================================
    # 报告生成
    # =========================================================================
    
    def generate_report(self, proximity_events, analyzed_events, level_counts):
        """生成最终分析报告 (改进版：根据TTC动态分类，包含PDF输出)"""
        report_path = self.analysis_dir / 'analysis_report.txt'
        pdf_path = self.analysis_dir / 'analysis_report_with_images.pdf'
        
        # 辅助函数：格式化TTC值（支持毫秒显示）
        def format_ttc(ttc_seconds):
            if ttc_seconds is None or ttc_seconds <= 0:
                return "N/A"
            elif ttc_seconds < 0.01:  # 小于10ms，用毫秒显示
                return f"{ttc_seconds*1000:.2f}ms"
            elif ttc_seconds < 0.1:   # 小于100ms，用4位小数
                return f"{ttc_seconds:.4f}s"
            else:  # 大于等于100ms
                return f"{ttc_seconds:.2f}s"
        
        # 辅助函数：查找keyframe图片
        def find_keyframe(frame_num, track_id_1, track_id_2):
            """根据frame和两个track_id查找对应的keyframe图片"""
            keyframes_dir = self.run_dir / '3_key_frames'
            # 尝试两种ID顺序
            candidates = [
                keyframes_dir / f"keyframe_{frame_num:04d}_ID{track_id_1}_ID{track_id_2}.jpg",
                keyframes_dir / f"keyframe_{frame_num:04d}_ID{track_id_2}_ID{track_id_1}.jpg"
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return None
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("YOLO-First 碰撞检测分析报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入视频: {self.video_path}\n")
            f.write(f"Homography: {self.homography_path if self.H is not None else '未提供'}\n")
            f.write(f"结果目录: {self.run_dir}\n\n")
            
            f.write(f"处理方式: YOLO-First\n")
            f.write(f"流程: YOLO检测 → 轨迹(px) → 关键帧 → Homography(关键帧) → 分析\n\n")
            
            f.write(f"关键帧统计:\n\n")
            f.write(f"总接近事件: {len(analyzed_events)}\n")
            if analyzed_events:
                f.write(f"Level 1 (Collision): {level_counts[1]}\n")
                f.write(f"Level 2 (Near Miss): {level_counts[2]}\n")
                f.write(f"Level 3 (Avoidance): {level_counts[3]}\n\n")
            
            # 根据TTC分类事件
            ttc_classified = self._classify_events_by_ttc(analyzed_events)
            
            # 输出分类结果
            f.write("根据TTC值的碰撞风险分类:\n\n")
            
            # Rear-end 碰撞
            if ttc_classified['rear_end_serious']:
                f.write(f"【Rear-end - Serious Conflict (TTC 0-2.8s)】: {len(ttc_classified['rear_end_serious'])} 个\n")
                for event in ttc_classified['rear_end_serious'][:5]:
                    ttc = event['multi_anchor_detailed'].get('ttc_seconds', 0)
                    ttc_str = format_ttc(ttc)
                    f.write(f"  Frame {event['frame']}: TTC={ttc_str}, 距离={event['multi_anchor_detailed'].get('min_distance_meters', 0):.3f}m\n")
                if len(ttc_classified['rear_end_serious']) > 5:
                    f.write(f"  ... 还有 {len(ttc_classified['rear_end_serious']) - 5} 个\n")
                f.write("\n")
            
            if ttc_classified['rear_end_general']:
                f.write(f"【Rear-end - General Conflict (TTC 2.8-4.7s)】: {len(ttc_classified['rear_end_general'])} 个\n")
                for event in ttc_classified['rear_end_general'][:5]:
                    ttc = event['multi_anchor_detailed'].get('ttc_seconds', 0)
                    ttc_str = format_ttc(ttc)
                    f.write(f"  Frame {event['frame']}: TTC={ttc_str}, 距离={event['multi_anchor_detailed'].get('min_distance_meters', 0):.3f}m\n")
                if len(ttc_classified['rear_end_general']) > 5:
                    f.write(f"  ... 还有 {len(ttc_classified['rear_end_general']) - 5} 个\n")
                f.write("\n")
            
            # Sideswipe 碰撞
            if ttc_classified['sideswipe_serious']:
                f.write(f"【Sideswipe - Serious Conflict (TTC 0-2.3s)】: {len(ttc_classified['sideswipe_serious'])} 个\n")
                for event in ttc_classified['sideswipe_serious'][:5]:
                    ttc = event['multi_anchor_detailed'].get('ttc_seconds', 0)
                    ttc_str = format_ttc(ttc)
                    f.write(f"  Frame {event['frame']}: TTC={ttc_str}, 距离={event['multi_anchor_detailed'].get('min_distance_meters', 0):.3f}m\n")
                if len(ttc_classified['sideswipe_serious']) > 5:
                    f.write(f"  ... 还有 {len(ttc_classified['sideswipe_serious']) - 5} 个\n")
                f.write("\n")
            
            if ttc_classified['sideswipe_general']:
                f.write(f"【Sideswipe - General Conflict (TTC 2.3-4.2s)】: {len(ttc_classified['sideswipe_general'])} 个\n")
                for event in ttc_classified['sideswipe_general'][:5]:
                    ttc = event['multi_anchor_detailed'].get('ttc_seconds', 0)
                    ttc_str = format_ttc(ttc)
                    f.write(f"  Frame {event['frame']}: TTC={ttc_str}, 距离={event['multi_anchor_detailed'].get('min_distance_meters', 0):.3f}m\n")
                if len(ttc_classified['sideswipe_general']) > 5:
                    f.write(f"  ... 还有 {len(ttc_classified['sideswipe_general']) - 5} 个\n")
                f.write("\n")
            
            if not any([ttc_classified['rear_end_serious'], ttc_classified['rear_end_general'],
                       ttc_classified['sideswipe_serious'], ttc_classified['sideswipe_general']]):
                f.write("未检测到具有有效TTC值的碰撞事件\n\n")
            
            f.write("\n前10个高风险事件（详细信息）:\n\n")
            
            if analyzed_events:
                sorted_events = sorted(analyzed_events, key=lambda e: e.get('level', 3))
                
                for event in sorted_events[:10]:
                    f.write(f"Frame {event['frame']} ({event['time']:.2f}s)\n")
                    obj_ids = event.get('object_ids') or [event.get('track_id_1', -1), event.get('track_id_2', -1)]
                    f.write(f"物体ID: {obj_ids}\n")
                    f.write(f"风险等级: Level {event['level']} ({event.get('level_name', '?')})\n")
                    f.write(f"距离(像素): {event['distance_pixel']:.1f}px\n")
                    
                    if 'distance_meters' in event:
                        f.write(f"距离(米): {event['distance_meters']:.2f}m\n")
                    
                    # 从multi_anchor_detailed中提取TTC和碰撞类型信息
                    if 'multi_anchor_detailed' in event:
                        multi_anchor = event['multi_anchor_detailed']
                        ttc = multi_anchor.get('ttc_seconds')
                        pet = multi_anchor.get('pet_seconds')
                        approaching = multi_anchor.get('heading_analysis', {}).get('approaching', False)
                        
                        if ttc is not None and ttc > 0:
                            ttc_str = format_ttc(ttc)
                            f.write(f"TTC (时间碰撞): {ttc_str}\n")
                        else:
                            # 根据approaching标志判断原因
                            if approaching:
                                f.write(f"TTC (时间碰撞): 无法计算 / Insufficient Speed\n")
                            else:
                                f.write(f"TTC (时间碰撞): 远离 / Separating\n")
                        
                        if pet is not None and pet > 0:
                            f.write(f"PET (近距离通过时间): {pet:.4f}s\n")
                        
                        closest_parts = multi_anchor.get('closest_parts', {})
                        if 'description' in closest_parts:
                            f.write(f"碰撞部位: {closest_parts['description']}\n")
                        
                        min_dist = multi_anchor.get('min_distance_meters')
                        if min_dist is not None:
                            f.write(f"最小距离: {min_dist:.3f}m\n")
                    
                    f.write("\n")
            else:
                f.write("未检测到接近事件\n\n")
            
            f.write("="*70 + "\n\n")
            
            # 显示被过滤的事件（平行、远离等）
            if ttc_classified['filtered']:
                f.write(f"被过滤事件 (物体平行/远离/TTC过长): {len(ttc_classified['filtered'])} 个\n")
                f.write("（这些事件未纳入高风险列表，原因如下）\n\n")
                
                for event in ttc_classified['filtered'][:20]:
                    f.write(f"Frame {event['frame']} ({event['time']:.2f}s)\n")
                    obj_ids = event.get('object_ids') or [event.get('track_id_1', -1), event.get('track_id_2', -1)]
                    f.write(f"  物体ID: {obj_ids}\n")
                    
                    # 获取过滤原因
                    reason = self._get_filter_reason(event)
                    f.write(f"  过滤原因: {reason}\n")
                    
                    if 'multi_anchor_detailed' in event:
                        multi = event['multi_anchor_detailed']
                        dist = multi.get('min_distance_meters', 0)
                        ttc = multi.get('ttc_seconds')
                        approaching = multi.get('heading_analysis', {}).get('approaching', False)
                        
                        f.write(f"  距离: {dist:.3f}m | ")
                        f.write(f"接近: {'是' if approaching else '否'} | ")
                        if ttc and ttc > 0:
                            f.write(f"TTC: {format_ttc(ttc)}")
                        f.write("\n")
                    
                    f.write("\n")
                
                if len(ttc_classified['filtered']) > 20:
                    f.write(f"  ... 还有 {len(ttc_classified['filtered']) - 20} 个被过滤事件\n\n")
            
            f.write("\n" + "="*70 + "\n\n")
            
            # TTC 分级标准表
            f.write("TTC (时间碰撞) 分级标准参考:\n\n")
            f.write("┌─────────────────┬──────────────────┬──────────────────┐\n")
            f.write("│ 碰撞类型         │ 严重程度         │ TTC阈值 (秒)     │\n")
            f.write("├─────────────────┼──────────────────┼──────────────────┤\n")
            f.write("│ Rear-end        │ Serious conflict │ 0 – 2.8 s        │\n")
            f.write("│ (追尾)          │ General conflict │ 2.8 – 4.7 s      │\n")
            f.write("├─────────────────┼──────────────────┼──────────────────┤\n")
            f.write("│ Sideswipe       │ Serious conflict │ 0 – 2.3 s        │\n")
            f.write("│ (侧面碰撞)      │ General conflict │ 2.3 – 4.2 s      │\n")
            f.write("└─────────────────┴──────────────────┴──────────────────┘\n\n")
            
            f.write("过滤策略:\n")
            f.write("  1. 物体远离（距离增加）: 不计入高风险事件\n")
            f.write("  2. 物体速度平行或同速: 无有效TTC，不计入高风险事件\n")
            f.write("  3. TTC过长（Rear-end>4.7s / Sideswipe>4.2s）: 风险较低，不计入高风险事件\n")
            f.write("  4. 高风险事件: 仅包括TTC在分级标准范围内且物体真正接近的事件\n\n")
            
            f.write("="*70 + "\n")
            f.write("报告结束\n")
        
        print(f"\n  ✓ 报告已保存: {report_path.name}")
        
        # 生成包含图片的PDF报告
        self._generate_pdf_report(analyzed_events, ttc_classified, pdf_path, find_keyframe, format_ttc)
    
    def _generate_pdf_report(self, analyzed_events, ttc_classified, pdf_path, find_keyframe_func, format_ttc_func):
        """生成包含图片的PDF报告"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            from reportlab.lib import colors
            
            # 创建PDF文档
            doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
            story = []
            styles = getSampleStyleSheet()
            
            # 自定义样式
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                textColor=colors.HexColor('#000000'),
                spaceAfter=10,
                alignment=TA_CENTER
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=12,
                textColor=colors.HexColor('#1a1a1a'),
                spaceAfter=8
            )
            
            # 添加标题
            from datetime import datetime
            story.append(Paragraph("碰撞检测分析报告（含图片）", title_style))
            story.append(Paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # 添加高风险事件和对应的keyframe图片
            all_high_risk = (ttc_classified.get('rear_end_serious', []) + 
                            ttc_classified.get('rear_end_general', []) +
                            ttc_classified.get('sideswipe_serious', []) +
                            ttc_classified.get('sideswipe_general', []))
            
            if all_high_risk:
                story.append(Paragraph("高风险碰撞事件", heading_style))
                story.append(Spacer(1, 0.2*inch))
                
                for idx, event in enumerate(all_high_risk[:10], 1):  # 最多显示10个
                    frame = event['frame']
                    ttc = event['multi_anchor_detailed'].get('ttc_seconds')
                    dist = event['multi_anchor_detailed'].get('min_distance_meters', 0)
                    track_id_1 = event.get('track_id_1', -1)
                    track_id_2 = event.get('track_id_2', -1)
                    
                    # 添加事件信息
                    ttc_str = format_ttc_func(ttc) if ttc and ttc > 0 else "远离"
                    event_text = f"Frame {frame}: ID{track_id_1}+ID{track_id_2} | TTC={ttc_str} | 距离={dist:.2f}m"
                    story.append(Paragraph(event_text, styles['Normal']))
                    
                    # 查找并添加keyframe图片
                    keyframe_path = find_keyframe_func(frame, track_id_1, track_id_2)
                    if keyframe_path and keyframe_path.exists():
                        try:
                            img = Image(str(keyframe_path), width=6*inch, height=4.5*inch)
                            story.append(img)
                        except Exception as e:
                            story.append(Paragraph(f"[无法加载图片: {keyframe_path.name}]", styles['Normal']))
                    else:
                        story.append(Paragraph(f"[未找到keyframe图片]", styles['Normal']))
                    
                    story.append(Spacer(1, 0.2*inch))
                    
                    # 每3个事件后添加分页
                    if (idx) % 3 == 0:
                        story.append(PageBreak())
            
            # 构建PDF
            doc.build(story)
            print(f"  ✓ PDF报告已保存: {pdf_path.name}")
            
        except ImportError:
            print(f"  ⚠️  缺少reportlab库，跳过PDF生成。可运行: pip install reportlab")
        except Exception as e:
            print(f"  ⚠️  PDF生成失败: {e}")
    
    def _get_filter_reason(self, event):
        """获取事件被过滤的原因"""
        if 'multi_anchor_detailed' not in event:
            return "无多锚点数据"
        
        multi = event['multi_anchor_detailed']
        ttc = multi.get('ttc_seconds')
        heading_analysis = multi.get('heading_analysis', {})
        
        # 检查是否接近
        if not heading_analysis.get('approaching', False):
            return "物体远离（距离增加）"
        
        # 检查TTC是否有效
        if ttc is None or ttc <= 0:
            return "两物体速度平行或同速（无有效TTC）"
        
        # 检查TTC是否过长
        relative_heading = heading_analysis.get('relative_heading_rad', 0)
        import math
        is_sideswipe = abs(relative_heading) > math.pi / 4
        
        if is_sideswipe and ttc >= 4.2:
            return f"侧面碰撞但TTC过长({ttc:.2f}s > 4.2s)"
        elif not is_sideswipe and ttc >= 4.7:
            return f"追尾但TTC过长({ttc:.2f}s > 4.7s)"
        
        return "其他"
    
    def _classify_events_by_ttc(self, analyzed_events):
        """根据TTC值和相对方向判断碰撞类型和严重程度"""
        classified = {
            'rear_end_serious': [],      # TTC 0-2.8s
            'rear_end_general': [],      # TTC 2.8-4.7s
            'sideswipe_serious': [],     # TTC 0-2.3s
            'sideswipe_general': [],     # TTC 2.3-4.2s
            'filtered': []               # 被过滤的事件（平行、远离等）
        }
        
        for event in analyzed_events:
            if 'multi_anchor_detailed' not in event:
                classified['filtered'].append(event)
                continue
            
            heading_analysis = event['multi_anchor_detailed'].get('heading_analysis', {})
            approaching = heading_analysis.get('approaching', False)
            ttc = event['multi_anchor_detailed'].get('ttc_seconds')
            
            # 判断是否真正接近且有有效TTC
            if not approaching or ttc is None or ttc <= 0:
                classified['filtered'].append(event)
                continue
            
            # 根据相对heading判断是rear-end还是sideswipe
            relative_heading = heading_analysis.get('relative_heading_rad', 0)
            
            # 将heading标准化到[-π, π]
            import math
            heading_abs = abs(relative_heading)
            is_sideswipe = heading_abs > math.pi / 4  # 大于45度则判定为侧向
            
            if is_sideswipe:
                # Sideswipe 碰撞
                if ttc < 2.3:
                    classified['sideswipe_serious'].append(event)
                elif ttc < 4.2:
                    classified['sideswipe_general'].append(event)
                else:
                    classified['filtered'].append(event)
            else:
                # Rear-end 碰撞
                if ttc < 2.8:
                    classified['rear_end_serious'].append(event)
                elif ttc < 4.7:
                    classified['rear_end_general'].append(event)
                else:
                    classified['filtered'].append(event)
        
        return classified
    
    def _copy_results_to_workspace(self):
        """自动复制结果到 /workspace/ultralytics/results（使其在 VS Code 中可见）"""
        import shutil
        
        workspace_results = Path("/workspace/ultralytics/results")
        if workspace_results.exists() and self.run_dir.parent != workspace_results:
            try:
                # 检查结果是否已在 workspace_results 中
                result_in_workspace = workspace_results / self.run_dir.name
                if not result_in_workspace.exists():
                    shutil.copytree(self.run_dir, result_in_workspace)
                    print(f"\n  ✓ 结果已复制到: {result_in_workspace}")
                    print(f"    现在可以在 VS Code 中直接查看！")
            except Exception as e:
                print(f"\n  ⚠️  复制失败 ({e})，但结果已保存在: {self.run_dir}")
    
    # =========================================================================
    # 管道编排
    # =========================================================================
    
    def run(self, conf_threshold=0.45):
        """运行完整 YOLO-First 管道 (Method A)"""
        try:
            # Step 0: 加载Homography (如果提供)
            if self.homography_path:
                print(f"\n【Step 0: 加载资源】")
                self.load_homography()
            
            # Step 0.5: Homography 标定可视化（如果使用YAML配置）
            if self.H is not None and self.config is not None:
                self.visualize_homography_calibration()
            
            # Step 1: YOLO 检测
            all_detections = self.run_yolo_detection(conf_threshold)
            
            if not all_detections:
                print(f"\n❌ 未检测到任何物体，停止处理")
                return
            
            # Step 1.5: 同帧内物体分割合并  - 合并YOLO分割的同类物体
            all_detections = self.merge_fragmented_objects_in_frame(all_detections, same_class_distance_threshold=100)
            
            # 调试：检查Step 1.5后的数据
            print(f"\n【调试: Step 1.5后的数据】")
            for frame_data in all_detections:
                if frame_data['frame'] == 115:
                    frame_115_objects_after_15 = [(obj['track_id'], obj['class']) for obj in frame_data['objects']]
                    print(f"  Frame 115 Step 1.5后: {frame_115_objects_after_15}")
            
            # Step 2: 轨迹构建
            tracks = self.build_trajectories(all_detections)
            
            # Step 2.4: 轨迹间断检测  - 检测出现→消失→重新出现的可疑轨迹
            suspicious_tracks = self.detect_discontinuous_tracks(all_detections, max_gap_frames=3)
            
            # Step 2.5: 轨迹连续性过滤  - 移除短轨迹误检
            all_detections = self.filter_short_tracks(all_detections, min_track_length=self.min_track_length)
            
            # 调试：检查Step 2.5后的数据
            print(f"\n【调试: Step 2.5后的数据】")
            frame_115_objects = []
            frame_148_objects = []
            for frame_data in all_detections:
                if frame_data['frame'] == 115:
                    frame_115_objects = [(obj['track_id'], obj['class']) for obj in frame_data['objects']]
                if frame_data['frame'] == 148:
                    frame_148_objects = [(obj['track_id'], obj['class']) for obj in frame_data['objects']]
            print(f"  Frame 115: {frame_115_objects}")
            print(f"  Frame 148: {frame_148_objects}")
            
            # ⚠️ 关键：Step 2.5过滤后需要重新构建轨迹，否则Step 3.6会找不到对象
            tracks = self.build_trajectories(all_detections)
            
            # Step 3: 关键帧检测 (Option B: 使用Step 2保存的轨迹world坐标)
            proximity_events = self.extract_key_frames(all_detections, tracks, world_distance_threshold=4.5)
            
            if not proximity_events:
                print(f"\n⚠️  未检测到接近事件")
                analyzed_events = []
                level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            else:
                # Step 3.5: 同类别物体误检过滤 ✨ 新增
                filtered_events = self.filter_same_class_false_positives(proximity_events, same_class_distance_threshold=0.3)
                
                # 清理Step 3.5后被过滤的keyframe图片
                if len(filtered_events) < len(proximity_events):
                    self.cleanup_filtered_keyframes(proximity_events, filtered_events)
                
                # Step 3.6: 多锚点碰撞分析 (仅关键帧)
                try:
                    filtered_events = self.analyze_keyframes_with_multi_anchor(filtered_events, all_detections, tracks)
                except Exception as e:
                    print(f"\n  ⚠️  Step 3.6 多锚点分析失败: {e}")
                    print(f"     继续使用简单的中心点距离分析结果")
                
                # 保存最终的proximity_events（包含多锚点分析结果）
                events_path = self.keyframe_dir / 'proximity_events.json'
                with open(events_path, 'w') as f:
                    json.dump(filtered_events, f, indent=2)
                
                # 重新绘制关键帧（现在包含多锚点可视化）
                for event in filtered_events:
                    frame_num = event['frame']
                    tid1 = event['track_id_1']
                    tid2 = event['track_id_2']
                    frame_img_path = self.keyframe_dir / f"keyframe_{frame_num:04d}_ID{tid1}_ID{tid2}.jpg"
                    self.save_keyframe_with_distance(self.video_path, frame_num, frame_img_path, event)
                
                # STEP 3.7: 多锚点距离过滤（在这里执行，不是在Step 5）
                print(f"\n【Step 3.7: 多锚点距离过滤 (≤1.0m)】")
                anchor_filtered_events = []
                removed_reasons = {'no_anchor_data': [], 'distance_too_far': []}
                
                for event in filtered_events:
                    frame = event['frame']
                    tid1 = event['track_id_1']
                    tid2 = event['track_id_2']
                    
                    # 检查是否有多锚点分析数据
                    if 'multi_anchor_detailed' not in event:
                        removed_reasons['no_anchor_data'].append((frame, tid1, tid2))
                        continue
                    
                    multi = event['multi_anchor_detailed']
                    min_distance = multi.get('min_distance_meters', float('inf'))
                    
                    # 保留距离 ≤ 1.0m 的事件（高风险）
                    if min_distance <= 1.0:
                        anchor_filtered_events.append(event)
                    else:
                        removed_reasons['distance_too_far'].append((frame, tid1, tid2, min_distance))
                
                # 报告被过滤的事件
                if removed_reasons['no_anchor_data']:
                    print(f"  ⊗ 移除 {len(removed_reasons['no_anchor_data'])} 个无多锚点数据的事件")
                
                if removed_reasons['distance_too_far']:
                    print(f"  ⊗ 移除 {len(removed_reasons['distance_too_far'])} 个距离>1.0m的事件:")
                    for frame, tid1, tid2, dist in removed_reasons['distance_too_far']:
                        print(f"     - Frame {frame}: ID{tid1}+ID{tid2} (锚点距离={dist:.2f}m)")
                
                filtered_count = len(filtered_events) - len(anchor_filtered_events)
                print(f"  🔍 多锚点距离过滤: 排除 {filtered_count} 个事件")
                print(f"  ✓ Step 3.7完成: 保留 {len(anchor_filtered_events)} 个关键帧 (≤ 1.0m)")
                
                # 清理：只保留Step 3.7后被保留的keyframe图片
                self.cleanup_filtered_keyframes(filtered_events, anchor_filtered_events)
                
                # 保存Step 3.7后的最终关键帧JSON（仅包含被保留的事件）
                events_path = self.keyframe_dir / 'proximity_events.json'
                with open(events_path, 'w') as f:
                    json.dump(anchor_filtered_events, f, indent=2)
                
                # 用Step 3.7过滤后的事件继续后续步骤
                filtered_events = anchor_filtered_events
                # Step 4: Homography 变换 (仅关键帧)
                if self.H is not None:
                    transformed_events = self.transform_key_frames_to_world(filtered_events)
                else:
                    print(f"\n【Step 4: Homography 变换】")
                    print(f"  ⚠️  跳过 (未加载Homography)")
                    transformed_events = filtered_events
                
                # Step 5: 风险分析
                analyzed_events, level_counts = self.analyze_collision_risk(transformed_events if transformed_events else filtered_events)
            
            # 生成报告
            self.generate_report(proximity_events, analyzed_events, level_counts)
            
            print(f"\n{'='*70}")
            print(f"✓ YOLO-First Pipeline (Method A) 完成！")
            print(f"{'='*70}")
            print(f"结果保存在: {self.run_dir}")
            
            # 自动复制结果到 /workspace/ultralytics/results（如果不同的话）
            self._copy_results_to_workspace()
            
            print(f"\n文件夹结构:")
            print(f"  1_yolo_detection/")
            print(f"    ├── detections_pixel.json")
            print(f"    ├── detection_stats.json")
            print(f"    └── *.jpg (所有有检测的帧)")
            print(f"  2_trajectories/")
            print(f"    ├── tracks.json")
            print(f"    └── track_stats.json")
            print(f"  3_key_frames/")
            print(f"    ├── proximity_events.json")
            print(f"    └── *.jpg (接近事件的关键帧)")
            print(f"  4_homography_transform/")
            print(f"    ├── homography.json")
            print(f"    └── transformed_key_frames.json")
            print(f"  5_collision_analysis/")
            print(f"    ├── collision_events.json")
            print(f"    └── analysis_report.txt")
            
        except Exception as e:
            print(f"\n❌ Pipeline 错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO-First 碰撞检测Pipeline (Method A - 导师推荐)')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--homography', type=str, default=None, 
                       help='Homography JSON路径 (可选)')
    parser.add_argument('--output', type=str, default='../../results', 
                       help='结果基础目录')
    parser.add_argument('--conf', type=float, default=0.45, 
                       help='YOLO置信度阈值 (越高=越严格，减少误检) (默认: 0.45)')
    parser.add_argument('--skip-frames', type=int, default=3,
                       help='抽帧参数: 3=每隔3帧处理1帧, 5=每隔5帧处理1帧 (最小值为3，用于提高速度计算准确性) (默认: 3)')
    parser.add_argument('--model', type=str, default='yolo11m',
                       help='YOLO 模型: yolo11n(快速), yolo11m(中等,更精确), yolo11l(最精确) (默认: yolo11m)')
    parser.add_argument('--min-track-length', type=int, default=3,
                       help='最小轨迹长度(帧数)，短于此的轨迹被认为是误检并排除 (默认: 3)')
    
    args = parser.parse_args()
    
    pipeline = YOLOFirstPipelineA(args.video, args.homography, args.output, 
                                  skip_frames=args.skip_frames, 
                                  model=args.model,
                                  min_track_length=args.min_track_length)
    pipeline.run(args.conf)
