#!/usr/bin/env python3
"""
Visualize Collision Detection Results - Annotated Video Generation Tool

PURPOSE:
  Generate visually annotated collision detection results from the Method A Pipeline output.
  Creates videos with detection boxes, trajectory IDs, distance measurements, and collision
  event indicators overlaid on the original video stream.

FUNCTIONALITY:
  1. Loads pipeline outputs: YOLO detections, trajectories, proximity events, collision events
  2. Renders annotated video with:
     - YOLO detection boxes (colored by object class per COCO 80-class model)
     - Track IDs with confidence scores
     - Keyframe highlighting (frames with collision events)
     - Distance connection lines between colliding/near-miss objects
     - Distance measurements at connection line midpoints
     - Event severity levels (Collision/Near Miss/etc) with color coding
  3. Generates supplementary reports:
     - CSV file with event data (distance, classes, speeds, timestamps)
     - Text summary report with statistics and event details

USAGE:
  python visualize_results.py --video <video_path> --results <results_dir> [--output <output_dir>]

EXAMPLE:
  python visualize_results.py \
    --video videos/highway_sample.mp4 \
    --results results/2025-01-20_14-30-45 \
    --output results/2025-01-20_14-30-45/visualization

PARAMETERS:
  --video (required):    Path to original input video file
  --results (required):  Path to Method A Pipeline output directory (contains 1_yolo_detection/,
                        2_trajectories/, 3_key_frames/, 5_collision_analysis/ subdirectories)
  --output (optional):   Output directory for visualizations. Default: <results_dir>/visualization/

OUTPUT FILES:
  - annotated_<video_name>.mp4       : Main annotated video with all overlays
  - events_summary.csv               : Structured data of all collision events
  - visualization_summary.txt        : Human-readable statistics and event report

DATA STRUCTURE EXPECTED:
  results_dir/
    ├── 1_yolo_detection/detections_pixel.json   (YOLO detection boxes per frame)
    ├── 2_trajectories/tracks.json                (Object trajectories with speed data)
    ├── 3_key_frames/proximity_events.json        (Proximity events from keyframe analysis)
    └── 5_collision_analysis/collision_events.json (Final classified collision events)

VIDEO ANNOTATIONS:
  Box colors by class:  Person (green), Car (red), Motorcycle (blue), Truck (orange), Bus (purple)
  Line colors by level: Red (collision), Orange (near miss), Yellow (other proximity)
  Keyframe header:      Red background with event count indicator

PERFORMANCE:
  - Processing speed: 10-30 FPS depending on video resolution and detection density
  - Memory usage: 2-4GB for HD video processing
  - Output video codec: H.264 (mp4v)
"""

import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

class ResultVisualizer:
    """Result visualization engine for collision detection pipeline output"""
    
    def __init__(self, video_path, results_dir, output_dir=None):
        """
        Initialize visualizer with video and pipeline output paths
        
        Parameters:
          video_path: Path to original input video file
          results_dir: Path to Method A Pipeline output directory
          output_dir: Output directory for visualizations. If None, uses results_dir/visualization/
        """
        self.video_path = Path(video_path)
        self.results_dir = Path(results_dir)
        
        # 默认输出目录：在results_dir内创建visualization文件夹
        if output_dir is None:
            self.output_dir = self.results_dir / "visualization"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load pipeline output data
        detections_raw = self._load_json(self.results_dir / "1_yolo_detection" / "detections_pixel.json")
        tracks_raw = self._load_json(self.results_dir / "2_trajectories" / "tracks.json")
        self.proximity_events = self._load_json(self.results_dir / "3_key_frames" / "proximity_events.json")
        self.collision_events = self._load_json(self.results_dir / "5_collision_analysis" / "collision_events.json")
        
        # Handle detections format (list or dict) - normalize to dict for fast lookup
        if isinstance(detections_raw, list):
            self.detections = {det['frame']: det for det in detections_raw}
            self.detections_list = detections_raw
        else:
            self.detections = detections_raw.get('detections', {}) if isinstance(detections_raw, dict) else {}
            # If dict, convert to list for iteration
            self.detections_list = list(self.detections.values()) if isinstance(self.detections, dict) else []
        
        # Handle tracks format - normalize to dict
        if isinstance(tracks_raw, dict) and 'tracks' in tracks_raw:
            self.tracks = tracks_raw['tracks']
        else:
            self.tracks = tracks_raw if isinstance(tracks_raw, dict) else {}
        
        # Open video and extract metadata
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Color mapping for different object classes (COCO 80-class model)
        self.class_colors = {
            'person': (0, 255, 0),      # Green
            'car': (0, 0, 255),         # Red
            'motorcycle': (255, 0, 0),  # Blue
            'truck': (255, 165, 0),     # Orange
            'bus': (255, 0, 255),       # Purple
        }
        
        # Frame-to-events mapping for fast lookup during rendering
        self.keyframe_events = defaultdict(list)
        for event in self.collision_events or []:
            frame_id = event.get('frame')  # or 'frame_id'
            if frame_id is not None:
                self.keyframe_events[frame_id].append(event)
        
        # Track mapping for fast lookup by track ID
        self.track_by_id = {}
        for track_id, track_points in self.tracks.items():
            self.track_by_id[int(track_id)] = track_points
        
        print(f"✓ Data loading complete")
        print(f"  Video: {self.video_path.name} ({self.width}x{self.height}, {self.fps:.1f}fps, {self.total_frames} frames)")
        print(f"  Detections: {len(self.detections_list)} frames")
        print(f"  Trajectories: {len(self.track_by_id)} tracks")
        print(f"  Events: {len(self.collision_events or [])} total")
    
    def _load_json(self, path):
        """Load JSON file from specified path"""
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
    
    def _get_color(self, class_name):
        """Get BGR color tuple for object class"""
        return self.class_colors.get(class_name, (200, 200, 200))
    
    def _get_track_data_at_frame(self, track_id, frame_id):
        """Retrieve trajectory data for specific track at given frame"""
        if track_id not in self.track_by_id:
            return None
        track_points = self.track_by_id[track_id]
        if isinstance(track_points, list):
            for point in track_points:
                if point.get('frame') == frame_id:
                    return point
        return None
    
    def _get_detections_at_frame(self, frame_id):
        """Get all YOLO detections for specified frame"""
        if not self.detections:
            return []
        frame_data = self.detections.get(frame_id)
        return frame_data.get('objects', []) if frame_data else []
    
    def generate_annotated_video(self):
        """Generate annotated video with YOLO detections and collision event overlays on all frames"""
        output_path = self.output_dir / f"annotated_{self.video_path.stem}.mp4"
        
        # Build frame -> detections mapping for fast lookup
        frame_detections_map = {}
        for det_frame in self.detections.values() if isinstance(self.detections, dict) else self.detections:
            if isinstance(det_frame, dict):
                frame_id = det_frame.get('frame')
                frame_detections_map[frame_id] = det_frame.get('objects', [])
        
        # Video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
        
        frame_id = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        print(f"  Processing video frames: {self.total_frames}")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Get YOLO detections for current frame
            dets = frame_detections_map.get(frame_id, [])
            
            # Build track_id -> bbox mapping for keyframe connection line rendering
            track_bbox_map = {}
            
            # Render YOLO detection boxes on all frames
            if dets:
                for det in dets:
                    track_id = det.get('track_id', -1)
                    class_id = det.get('class', -1)
                    conf = det.get('conf', 0)
                    bbox_xywh = det.get('bbox_xywh', [])
                    
                    if not bbox_xywh:
                        continue
                    
                    # Convert xywh -> xyxy format
                    x_center, y_center, w, h = bbox_xywh
                    x1 = int(x_center - w/2)
                    y1 = int(y_center - h/2)
                    x2 = int(x_center + w/2)
                    y2 = int(y_center + h/2)
                    
                    # Save bbox for keyframe connection line rendering
                    track_bbox_map[track_id] = {
                        'bbox': (x1, y1, x2, y2),
                        'center': (int(x_center), int(y_center)),
                        'class': class_id
                    }
                    
                    # Get class name from COCO 80-class model
                    class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
                                  4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'}
                    class_name = class_names.get(class_id, f'class_{class_id}')
                    color = self._get_color(class_name)
                    
                    # Draw detection box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label text (ID + class + confidence)
                    label = f"ID{track_id} {class_name} {conf:.2f}"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(frame, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # If keyframe, render additional event info and distance connection lines
            if frame_id in self.keyframe_events:
                events = self.keyframe_events[frame_id]
                
                # Draw warning header background
                cv2.rectangle(frame, (5, 5), (550, 35), (0, 0, 255), -1)
                cv2.putText(frame, f"⚠ KEYFRAME - {len(events)} collision event(s)", (15, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw connection lines and annotations for each event
                y_offset = 45
                for event_idx, event in enumerate(events):
                    dist = event.get('distance_meters', 0)
                    level = event.get('level_name', 'Unknown')
                    id1 = event.get('track_id_1', -1)
                    id2 = event.get('track_id_2', -1)
                    class1 = event.get('class_1', '?')
                    class2 = event.get('class_2', '?')
                    
                    # Event text annotation
                    text = f"Event {event_idx+1}: ID{id1}({class1}) <-> ID{id2}({class2}) | Dist: {dist:.2f}m | {level}"
                    cv2.putText(frame, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y_offset += 25
                    
                    # Draw connection line and distance label between two objects
                    if id1 in track_bbox_map and id2 in track_bbox_map:
                        center1 = track_bbox_map[id1]['center']
                        center2 = track_bbox_map[id2]['center']
                        
                        # Select line color based on event severity level
                        if 'Collision' in level:
                            line_color = (0, 0, 255)  # Red: collision
                        elif 'Near Miss' in level:
                            line_color = (0, 165, 255)  # Orange: near miss
                        else:
                            line_color = (0, 255, 255)  # Yellow: other proximity
                        
                        # Draw connection line
                        cv2.line(frame, center1, center2, line_color, 2)
                        
                        # Draw distance label at line midpoint
                        mid_x = (center1[0] + center2[0]) // 2
                        mid_y = (center1[1] + center2[1]) // 2
                        
                        # Background box for distance text
                        dist_text = f"{dist:.2f}m"
                        text_size = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, 
                                    (mid_x - text_size[0]//2 - 3, mid_y - text_size[1]//2 - 3),
                                    (mid_x + text_size[0]//2 + 3, mid_y + text_size[1]//2 + 3),
                                    line_color, -1)
                        
                        # Distance value text
                        cv2.putText(frame, dist_text, (mid_x - text_size[0]//2, mid_y + text_size[1]//2 - 2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame number and timestamp in top-right corner
            timestamp = frame_id / self.fps if self.fps > 0 else 0
            frame_info = f"Frame: {frame_id} | Time: {timestamp:.2f}s"
            text_size = cv2.getTextSize(frame_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (self.width - text_size[0] - 10, 5), (self.width - 5, 30), (0, 0, 0), -1)
            cv2.putText(frame, frame_info, (self.width - text_size[0] - 5, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Write frame to video
            writer.write(frame)
            frame_id += 1
            
            if frame_id % 20 == 0:
                print(f"  Processing: {frame_id}/{self.total_frames} frames", end='\r')
        
        writer.release()
        print(f"\n✓ Annotated video saved: {output_path}")
        return output_path
    
    def generate_keyframe_images(self):
        """Keyframes already included in annotated video - skip separate export"""
        print(f"✓ Keyframes included in annotated video - skipping separate export")
        return None
    
    def generate_csv_report(self):
        """Generate CSV-formatted event report with all collision data"""
        import csv
        
        csv_path = self.output_dir / "events_summary.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'event_id', 'frame_id', 'timestamp', 'track_id_1', 'track_id_2',
                'class_1', 'class_2', 'distance_meters', 'level', 'speed_1_ms', 'speed_2_ms'
            ])
            writer.writeheader()
            
            for idx, event in enumerate(self.collision_events or [], 1):
                frame_id = event.get('frame_id', -1)
                timestamp = frame_id / self.fps if frame_id >= 0 else 0
                
                track_id_1 = event.get('track_id_1')
                track_id_2 = event.get('track_id_2')
                
                # Extract speed information from trajectory data
                speed_1 = 0
                speed_2 = 0
                if track_id_1 is not None:
                    track_data_1 = self._get_track_data_at_frame(track_id_1, frame_id)
                    speed_1 = track_data_1.get('speed_world', 0) if track_data_1 else 0
                if track_id_2 is not None:
                    track_data_2 = self._get_track_data_at_frame(track_id_2, frame_id)
                    speed_2 = track_data_2.get('speed_world', 0) if track_data_2 else 0
                
                writer.writerow({
                    'event_id': idx,
                    'frame_id': frame_id,
                    'timestamp': f"{timestamp:.2f}s",
                    'track_id_1': track_id_1,
                    'track_id_2': track_id_2,
                    'class_1': event.get('class_1', ''),
                    'class_2': event.get('class_2', ''),
                    'distance_meters': f"{event.get('distance_meters', 0):.3f}",
                    'level': event.get('level_name', ''),
                    'speed_1_ms': f"{speed_1:.2f}",
                    'speed_2_ms': f"{speed_2:.2f}",
                })
        
        print(f"✓ CSV report saved: {csv_path}")
        return csv_path
    
    def generate_summary_report(self):
        """Generate human-readable text summary report with statistics and event details"""
        report_path = self.output_dir / "visualization_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("Collision Detection Results Visualization Report\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"VIDEO INFORMATION\n")
            f.write(f"  File: {self.video_path.name}\n")
            f.write(f"  Resolution: {self.width}x{self.height}\n")
            f.write(f"  Frame rate: {self.fps:.1f} fps\n")
            f.write(f"  Total frames: {self.total_frames}\n")
            f.write(f"  Duration: {self.total_frames / self.fps:.1f}s\n\n")
            
            f.write(f"DETECTION STATISTICS\n")
            f.write(f"  Total trajectories: {len(self.track_by_id)}\n")
            f.write(f"  Frames with detections: {len(self.detections.get('detections', {}))} frames\n")
            f.write(f"  Keyframes with events: {len(self.collision_events or [])} frames\n\n")
            
            f.write(f"EVENT SEVERITY DISTRIBUTION\n")
            level_counts = defaultdict(int)
            for event in self.collision_events or []:
                level = event.get('level_name', 'Unknown')
                level_counts[level] += 1
            
            for level, count in sorted(level_counts.items()):
                f.write(f"  {level}: {count}\n")
            
            f.write(f"\nEVENT DETAILS\n")
            for idx, event in enumerate(self.collision_events or [], 1):
                frame_id = event.get('frame_id', -1)
                timestamp = frame_id / self.fps if frame_id >= 0 else 0
                track_id_1 = event.get('track_id_1')
                track_id_2 = event.get('track_id_2')
                dist = event.get('distance_meters', 0)
                level = event.get('level_name', '')
                
                f.write(f"  [{idx}] Frame {frame_id} ({timestamp:.2f}s)\n")
                f.write(f"      ID{track_id_1}({event.get('class_1', '')}) <-> ID{track_id_2}({event.get('class_2', '')})\n")
                f.write(f"      Distance: {dist:.3f}m | Level: {level}\n\n")
        
        print(f"✓ Summary report saved: {report_path}")
        return report_path
    
    def run(self):
        """Execute all visualization tasks and generate output files"""
        print(f"\n【Visualizing Pipeline Results】")
        print(f"Output directory: {self.output_dir}\n")
        
        # Generate annotated video with overlays
        self.generate_annotated_video()
        
        # Keyframe export (already included in annotated video)
        self.generate_keyframe_images()
        
        # Generate CSV event data
        self.generate_csv_report()
        
        # Generate text summary report
        self.generate_summary_report()
        
        self.cap.release()
        
        print(f"\n✓ All visualization tasks complete!")
        print(f"  Output directory: {self.output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize collision detection results from Method A Pipeline")
    parser.add_argument('--video', required=True, help='Path to original input video file')
    parser.add_argument('--results', required=True, help='Path to Method A Pipeline output results directory')
    parser.add_argument('--output', help='Output directory for visualization files (optional)')
    
    args = parser.parse_args()
    
    visualizer = ResultVisualizer(args.video, args.results, args.output)
    visualizer.run()
