"""
[DEPRECATED] collision_detection_pipeline_yolo_first_method_c.py
Relatively slow, not used

YOLO-First Collision Detection Pipeline (Approach 2, Method C: Homography First)
**Relatively slow, not used
Execution order: YOLO Detection → Homography Transform → Trajectory Building → Key Frame Extraction → Analysis

Workflow:
1. YOLO Detection (original video, all frames)
   - Detect all objects at original resolution
   - Save detection boxes and Track IDs (pixel space)

2. Homography Transform (all detection boxes)
   - Convert all detection boxes from pixel coordinates to world coordinates
   - Calculate scale factor (px → meters)

3. Trajectory Building (world coordinates, metric)
   - Associate Track IDs and build trajectories
   - Estimate velocity (m/s)
   - All calculations in the same coordinate system

4. Key Frame Extraction (proximity event detection)
   - Detect object pairs with distance < 1.5m
   - Mark as key frames

5. TTC and Event Classification
   - Calculate TTC (world coordinates)
   - Classify events (L1/L2/L3)
   - Generate report

Advantages:
- Trajectories in metric space, clear and intuitive
- All calculations in the same coordinate system, good consistency
- Distance threshold in meters (1.5m), clearer
- Still 1.5-2 times faster than Homography-First
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# 导入YOLO和相关模块
sys.path.append(os.path.dirname(__file__))
from ultralytics import YOLO


class YOLOFirstPipelineC:
    def __init__(self, video_path, homography_path=None, output_base='../../results'):
        """Initialize YOLO-First pipeline (Method C: Homography First)
        
        Args:
            video_path: Path to the original video
            homography_path: Path to Homography JSON (required for coordinate transformation)
            output_base: Base directory for results
        """
        self.video_path = video_path
        self.homography_path = homography_path
        self.output_base = Path(output_base)
        self.H = None
        self.pixel_per_meter = 1.0
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_base / f"{timestamp}_yolo_first_c"
        
        # Create subdirectory structure (improved version, conforming to Method C)
        self.detection_dir = self.run_dir / "1_raw_detections"
        self.homography_dir = self.run_dir / "2_homography_transform"
        self.trajectory_dir = self.run_dir / "3_trajectories"
        self.keyframe_dir = self.run_dir / "4_key_frames"
        self.analysis_dir = self.run_dir / "5_collision_analysis"
        
        for d in [self.detection_dir, self.homography_dir, self.trajectory_dir, 
                  self.keyframe_dir, self.analysis_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"YOLO-First Collision Detection Pipeline (Method C: Homography First)")
        print(f"{'='*70}")
        print(f"Timestamp: {timestamp}")
        print(f"Results directory: {self.run_dir}")
        print(f"Execution order: YOLO → Homography → Trajectories(metric) → Key Frames → Analysis")
        
        if not homography_path:
            print(f"⚠️  Warning: No Homography provided, will process in pixel space only")
    
    def load_homography(self):
        """Step 0.5: Load Homography matrix"""
        if not self.homography_path:
            print(f"\n⚠️  No Homography provided, will process in pixel space")
            return False
        
        print(f"\n[Step 0.5: Load Homography Matrix]")
        
        try:
            with open(self.homography_path) as f:
                H_data = json.load(f)
            
            self.H = np.array(H_data['homography_matrix'], dtype=np.float32)
            pixel_points = H_data['pixel_points']
            world_points = H_data['world_points']
            
            # Save to output directory
            with open(self.homography_dir / 'homography.json', 'w') as f:
                json.dump(H_data, f, indent=2)
            
            # Calculate pixel-to-meter scale factor
            if len(world_points) >= 2 and len(pixel_points) >= 2:
                px_dist = np.sqrt((pixel_points[0][0] - pixel_points[1][0])**2 + 
                                 (pixel_points[0][1] - pixel_points[1][1])**2)
                world_dist = np.sqrt((world_points[0][0] - world_points[1][0])**2 + 
                                    (world_points[0][1] - world_points[1][1])**2)
                
                self.pixel_per_meter = px_dist / world_dist if world_dist > 0 else 1.0
            
            print(f"✓ Homography matrix loaded")
            print(f"  Scale factor: {self.pixel_per_meter:.2f} px/m")
            print(f"  Reference points: {len(pixel_points)}")
            
            return True
        
        except Exception as e:
            print(f"❌ Failed to load Homography: {e}")
            return False
    
    def run_yolo_detection(self, conf_threshold=0.45):
        """Step 1: YOLO Detection (original video, all frames)"""
        print(f"\n[Step 1: YOLO Detection]")
        
        model = YOLO('yolo11n.pt')
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        all_detections = []
        frame_count = 0
        detection_frames_count = 0
        
        print(f"Processing: {total_frames}frames @ {fps:.2f}FPS...")
        
        for result in model.track(source=self.video_path, stream=True, 
                                 persist=True, conf=conf_threshold):
            frame_count += 1
            
            if result.boxes is None or len(result.boxes) == 0:
                if frame_count % 30 == 0:
                    print(f"  Frame {frame_count}/{total_frames} - No objects")
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
            
            for i in range(len(boxes)):
                obj_data = {
                    'track_id': int(ids[i]) if ids[i] is not None else -1,
                    'class': int(classes[i]),
                    'conf': float(confs[i]),
                    'bbox_xywh': boxes[i].tolist(),
                }
                frame_detections['objects'].append(obj_data)
            
            all_detections.append(frame_detections)
            
            if frame_count % 30 == 0:
                print(f"  Frame {frame_count}/{total_frames} - {len(boxes)}objects")
        
        cap.release()
        
        detections_path = self.detection_dir / 'detections_pixel.json'
        with open(detections_path, 'w') as f:
            json.dump(all_detections, f, indent=2)
        
        stats = {
            'total_frames': total_frames,
            'fps': fps,
            'detection_frames': detection_frames_count,
            'confidence_threshold': conf_threshold,
        }
        stats_path = self.detection_dir / 'detection_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ YOLO detection completed: {detection_frames_count}frames with detected objects")
        print(f"  Detection results saved: {detections_path.name}")
        
        return all_detections
    
    def transform_detections_to_world(self, all_detections):
        """Step 2: Homography Transform (all detection boxes)"""
        print(f"\n[Step 2: Homography Coordinate Transform]")
        
        if self.H is None:
            print(f"⚠️  Homography not loaded, remaining in pixel space")
            return all_detections
        
        transformed_detections = []
        
        for frame_data in all_detections:
            trans_frame = {
                'frame': frame_data['frame'],
                'time': frame_data['time'],
                'objects': []
            }
            
            for obj in frame_data['objects']:
                trans_obj = obj.copy()
                
                x_px, y_px = obj['bbox_xywh'][0], obj['bbox_xywh'][1]
                x_world = x_px / self.pixel_per_meter
                y_world = y_px / self.pixel_per_meter
                
                trans_obj['center_x_world'] = x_world
                trans_obj['center_y_world'] = y_world
                trans_obj['center_x_pixel'] = x_px
                trans_obj['center_y_pixel'] = y_px
                
                trans_frame['objects'].append(trans_obj)
            
            transformed_detections.append(trans_frame)
        
        trans_path = self.homography_dir / 'detections_world.json'
        with open(trans_path, 'w') as f:
            json.dump(transformed_detections, f, indent=2)
        
        print(f"✓ Homography transform completed: {len(all_detections)}frames of detection boxes transformed to world coordinates")
        print(f"  Transform results saved: {trans_path.name}")
        
        return transformed_detections
    
    def build_trajectories_world(self, transformed_detections):
        """Step 3: Trajectory Building (world coordinates, metric)"""
        print(f"\n[Step 3: Trajectory Building (World Coordinates)]")
        
        tracks = {}
        
        for frame_data in transformed_detections:
            for obj in frame_data['objects']:
                track_id = obj['track_id']
                
                if track_id not in tracks:
                    tracks[track_id] = []
                
                track_point = {
                    'frame': frame_data['frame'],
                    'time': frame_data['time'],
                    'class': obj['class'],
                    'conf': obj['conf'],
                    'center_x': obj['center_x_world'],
                    'center_y': obj['center_y_world'],
                }
                
                tracks[track_id].append(track_point)
        
        # Calculate velocity (m/s)
        for track_id, track_points in tracks.items():
            track_points.sort(key=lambda p: p['frame'])
            
            if len(track_points) >= 2:
                for i in range(1, len(track_points)):
                    prev = track_points[i-1]
                    curr = track_points[i]
                    
                    dt = curr['time'] - prev['time']
                    if dt > 0:
                        dx = curr['center_x'] - prev['center_x']
                        dy = curr['center_y'] - prev['center_y']
                        
                        curr['vx'] = dx / dt
                        curr['vy'] = dy / dt
                        curr['speed'] = np.sqrt(dx**2 + dy**2) / dt
                    else:
                        curr['vx'] = 0.0
                        curr['vy'] = 0.0
                        curr['speed'] = 0.0
                
                track_points[0]['vx'] = 0.0
                track_points[0]['vy'] = 0.0
                track_points[0]['speed'] = 0.0
        
        tracks_path = self.trajectory_dir / 'tracks_world.json'
        with open(tracks_path, 'w') as f:
            json.dump(tracks, f, indent=2)
        
        stats = {
            'total_tracks': len(tracks),
            'track_lengths': {str(tid): len(points) for tid, points in tracks.items()},
            'coordinate_system': 'world (meters)',
            'velocity_unit': 'm/s',
        }
        stats_path = self.trajectory_dir / 'track_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Trajectory building completed: {len(tracks)}trajectories (world coordinates)")
        print(f"  Coordinate system: World coordinates (meters)")
        print(f"  Velocity unit: m/s")
        print(f"  Trajectory results saved: {tracks_path.name}")
        
        return tracks
    
    def extract_key_frames_world(self, transformed_detections, distance_threshold=1.5):
        """Step 4: Key Frame Extraction (proximity events, world coordinates)"""
        print(f"\n[Step 4: Key Frame Extraction]")
        
        proximity_events = []
        
        for frame_data in transformed_detections:
            if len(frame_data['objects']) < 2:
                continue
            
            frame = frame_data['frame']
            objects = frame_data['objects']
            
            for i in range(len(objects)):
                for j in range(i+1, len(objects)):
                    obj1 = objects[i]
                    obj2 = objects[j]
                    
                    x1, y1 = obj1['center_x_world'], obj1['center_y_world']
                    x2, y2 = obj2['center_x_world'], obj2['center_y_world']
                    
                    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    if distance < distance_threshold:
                        event = {
                            'frame': frame,
                            'time': frame_data['time'],
                            'object_ids': (obj1['track_id'], obj2['track_id']),
                            'distance_meters': float(distance),
                            'object_classes': (obj1['class'], obj2['class']),
                        }
                        proximity_events.append(event)
        
        events_path = self.keyframe_dir / 'proximity_events.json'
        with open(events_path, 'w') as f:
            json.dump(proximity_events, f, indent=2)
        
        print(f"✓ Key frame extraction completed: {len(proximity_events)}proximity events")
        print(f"  Distance threshold: {distance_threshold}m")
        print(f"  Proximity events saved: {events_path.name}")
        
        return proximity_events
    
    def analyze_collision_risk(self, proximity_events):
        """Step 5: Collision Risk Analysis and Event Classification"""
        print(f"\n[Step 5: Collision Risk Analysis]")
        
        analyzed_events = []
        
        for event in proximity_events:
            analyzed = event.copy()
            
            distance = event['distance_meters']
            
            # Classification standard (meters)
            if distance < 0.5:
                analyzed['level'] = 1
                analyzed['level_name'] = 'Collision'
            elif distance < 1.5:
                analyzed['level'] = 2
                analyzed['level_name'] = 'Near Miss'
            else:
                analyzed['level'] = 3
                analyzed['level_name'] = 'Avoidance'
            
            analyzed_events.append(analyzed)
        
        level_counts = {1: 0, 2: 0, 3: 0}
        for event in analyzed_events:
            level_counts[event['level']] += 1
        
        analysis_path = self.analysis_dir / 'collision_events.json'
        with open(analysis_path, 'w') as f:
            json.dump(analyzed_events, f, indent=2)
        
        print(f"✓ Collision risk analysis completed")
        print(f"  - Level 1 (Collision, <0.5m): {level_counts[1]} events")
        print(f"  - Level 2 (Near Miss, 0.5-1.5m): {level_counts[2]} events")
        print(f"  - Level 3 (Avoidance, >1.5m): {level_counts[3]} events")
        print(f"  Analysis results saved: {analysis_path.name}")
        
        return analyzed_events, level_counts
    
    def generate_report(self, proximity_events, analyzed_events, level_counts):
        """Generate final report"""
        report_path = self.analysis_dir / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("YOLO-First Collision Detection Analysis Report (Method C: Homography First)\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input video: {self.video_path}\n")
            f.write(f"Homography: {self.homography_path if self.H is not None else 'Not provided'}\n")
            f.write(f"Results directory: {self.run_dir}\n\n")
            
            f.write(f"Processing method: YOLO-First (Method C)\n")
            f.write(f"Workflow: YOLO Detection → Homography Transform → Trajectories(metric) → Key Frames → Analysis\n")
            f.write(f"Coordinate system: World coordinates (meters)\n")
            f.write(f"Velocity unit: m/s\n\n")
            
            f.write(f"Proximity event statistics:\n")
            f.write(f"  - Total proximity events: {len(proximity_events)}\n")
            f.write(f"  - Level 1 (Collision, <0.5m): {level_counts[1]}\n")
            f.write(f"  - Level 2 (Near Miss, 0.5-1.5m): {level_counts[2]}\n")
            f.write(f"  - Level 3 (Avoidance, >1.5m): {level_counts[3]}\n\n")
            
            if analyzed_events:
                f.write("Top 10 high-risk events:\n")
                f.write("-"*70 + "\n")
                
                sorted_events = sorted(analyzed_events, key=lambda e: e.get('level', 3))
                
                for i, event in enumerate(sorted_events[:10], 1):
                    f.write(f"\n{i}. Frame {event['frame']} ({event['time']:.2f}s)\n")
                    f.write(f"   Object IDs: {event['object_ids']}\n")
                    f.write(f"   Risk level: Level {event['level']} ({event.get('level_name', 'Unknown')})\n")
                    f.write(f"   Distance: {event['distance_meters']:.2f}m\n")
            else:
                f.write("No proximity events detected\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("End of report\n")
        
        print(f"✓ Report saved: {report_path.name}")
    
    def run(self, conf_threshold=0.45):
        """Run complete YOLO-First pipeline (Method C)"""
        try:
            # Step 0.5: Load Homography
            self.load_homography()
            
            # Step 1: YOLO Detection
            all_detections = self.run_yolo_detection(conf_threshold)
            
            if not all_detections:
                print(f"❌ No objects detected, stop processing")
                return
            
            # Step 2: Homography Transform (all detection boxes)
            transformed_detections = self.transform_detections_to_world(all_detections)
            
            # Step 3: Trajectory Building (world coordinates)
            tracks = self.build_trajectories_world(transformed_detections)
            
            # Step 4: Key Frame Extraction
            proximity_events = self.extract_key_frames_world(
                transformed_detections, 
                distance_threshold=1.5
            )
            
            if not proximity_events:
                print(f"⚠️  No proximity events detected")
                analyzed_events = []
                level_counts = {1: 0, 2: 0, 3: 0}
            else:
                # Step 5: Risk Analysis
                analyzed_events, level_counts = self.analyze_collision_risk(proximity_events)
            
            # Generate Report
            self.generate_report(proximity_events, analyzed_events, level_counts)
            
            print(f"\n{'='*70}")
            print(f"✓ YOLO-First Pipeline (Method C) completed!")
            print(f"{'='*70}")
            print(f"Results saved in: {self.run_dir}")
            print(f"\nFolder structure:")
            print(f"  1_raw_detections/")
            print(f"    ├── detections_pixel.json (pixel space)")
            print(f"    └── detection_stats.json")
            print(f"  2_homography_transform/")
            print(f"    ├── homography.json")
            print(f"    └── detections_world.json (world coordinates)")
            print(f"  3_trajectories/")
            print(f"    ├── tracks_world.json (metric trajectories)")
            print(f"    └── track_stats.json")
            print(f"  4_key_frames/")
            print(f"    └── proximity_events.json")
            print(f"  5_collision_analysis/")
            print(f"    ├── collision_events.json (classified events)")
            print(f"    └── analysis_report.txt")
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO-First Collision Detection Pipeline (Method C)')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--homography', type=str, required=True, 
                       help='Homography JSON path (required)')
    parser.add_argument('--output', type=str, default='../../results', 
                       help='Results base directory')
    parser.add_argument('--conf', type=float, default=0.45, 
                       help='YOLO confidence threshold')
    
    args = parser.parse_args()
    
    pipeline = YOLOFirstPipelineC(args.video, args.homography, args.output)
    pipeline.run(args.conf)
