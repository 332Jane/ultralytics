"""
visualize_collision_events.py

================================================================================
COLLISION EVENTS VISUALIZATION TOOL
================================================================================

PURPOSE:
  This script extracts and visualizes the most critical collision/near-miss
  events from collision detection results. It creates annotated images showing:
  1. YOLO detection boxes with object IDs
  2. Contact points (3 points per object)
  3. Distance measurements between contact points
  4. TTC (Time-To-Collision) values and risk levels

FUNCTIONALITY:
  - Loads collision events from JSON file
  - Extracts object trajectories from tracks JSON
  - Sorts events by risk level (TTC priority, then distance)
  - Visualizes top-K most critical events on video frames
  - Annotates with:
    * Detection boxes (blue for object 1, green for object 2)
    * Contact points (3 points: front, center, back)
    * Distance and TTC values
    * Risk level classification (CRITICAL/HIGH/MEDIUM)
  - Saves annotated frames as JPEG images
  - Generates summary report

USAGE:
  Basic usage:
    python examples/trajectory_demo/visualize_collision_events.py \\
      --near-misses results/xxx/5_collision_analysis/collision_events.json \\
      --tracks results/xxx/5_collision_analysis/tracks.json \\
      --video videos/input_video.mp4 \\
      --output collision_frames/ \\
      --top-k 10

  Custom filtering:
    python visualize_collision_events.py \\
      --near-misses near_misses.json \\
      --tracks tracks.json \\
      --video video.mp4 \\
      --output output/ \\
      --top-k 5

PARAMETERS:
  --near-misses <path>     : Path to collision events JSON file (REQUIRED)
  --tracks <path>          : Path to object trajectories JSON file (REQUIRED)
  --video <path>           : Path to input video file (REQUIRED)
  --output <path>          : Output directory for annotated frames
                             (Default: collision_frames/)
  --top-k <int>            : Number of most critical events to visualize
                             (Default: 10)

OUTPUT:
  Directory structure:
    output/
    ├── collision_event_001_frame_45_obj1_vs_obj2_CRITICAL.jpg
    ├── collision_event_002_frame_78_obj3_vs_obj5_HIGH.jpg
    ├── collision_event_003_frame_112_obj2_vs_obj4_MEDIUM.jpg
    └── collision_summary.txt

RISK LEVELS:
  CRITICAL: TTC < 0.5s   (imminent collision)
  HIGH:     TTC < 2.0s   (high risk)
  MEDIUM:   TTC >= 2.0s  (moderate risk)

COLOR CODING:
  Blue:   Object 1 (box + contact points)
  Green:  Object 2 (box + contact points)
  Yellow: Distance and TTC labels

================================================================================
"""

import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
import math


def load_data(near_misses_path, tracks_path):
    """Load collision events and object trajectory data from JSON files"""
    with open(near_misses_path, 'r') as f:
        near_misses = json.load(f)
    with open(tracks_path, 'r') as f:
        tracks = json.load(f)
    return near_misses, tracks


def get_object_info_at_frame(tracks, obj_id, frame_num):
    """Retrieve object information at a specific frame number"""
    if str(obj_id) not in tracks:
        return None
    
    trajectory = tracks[str(obj_id)]
    
    # Find the sample with closest timestamp to frame_num
    best_sample = None
    min_diff = float('inf')
    
    for sample in trajectory:
        diff = abs(sample['t'] - frame_num)
        if diff < min_diff:
            min_diff = diff
            best_sample = sample
    
    return best_sample


def draw_detection_box(frame, sample, obj_id, color=(0, 255, 0)):
    """Draw YOLO detection box with object ID label on frame"""
    if sample is None or sample.get('bbox') is None:
        return frame
    
    x1, y1, x2, y2 = sample['bbox']
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Draw detection box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw ID label
    label = f"ID:{obj_id}"
    cv2.putText(frame, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return frame


def draw_contact_points(frame, sample, obj_id, color=(0, 255, 0), alpha=0.7):
    """Draw contact points (front, center, back) on frame"""
    if sample is None or sample.get('contact_points_pixel') is None:
        return frame
    
    points = sample['contact_points_pixel']
    point_names = ['front', 'center', 'back']
    
    # Create transparent layer for drawing
    overlay = frame.copy()
    
    for i, (x, y) in enumerate(points):
        x, y = int(x), int(y)
        
        # Draw circle (contact point)
        cv2.circle(overlay, (x, y), 8, color, -1)
        cv2.circle(frame, (x, y), 8, color, 2)  # Outer outline
        
        # Label point name
        label = point_names[i]
        cv2.putText(frame, label, (x+15, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw lines connecting the three points
    for i in range(len(points) - 1):
        x1, y1 = int(points[i][0]), int(points[i][1])
        x2, y2 = int(points[i+1][0]), int(points[i+1][1])
        cv2.line(frame, (x1, y1), (x2, y2), color, 1)
    
    return frame


def draw_collision_info(frame, near_miss_event, obj1_sample, obj2_sample, fps=30.0):
    """Draw collision information (distance, TTC, contact type) on frame"""
    
    # Information lines to display
    info_lines = []
    
    obj1_id = near_miss_event['id1']
    obj2_id = near_miss_event['id2']
    distance = near_miss_event['distance']
    ttc = near_miss_event['ttc']
    
    info_lines.append(f"Collision: Object {obj1_id} <-> Object {obj2_id}")
    info_lines.append(f"Distance: {distance:.2f}")
    
    if 'closest_point_pair' in near_miss_event:
        cpp = near_miss_event['closest_point_pair']
        pt1_type = cpp['obj1_point_type']
        pt2_type = cpp['obj2_point_type']
        info_lines.append(f"Contact: {pt1_type} <-> {pt2_type}")
    
    if ttc is not None:
        info_lines.append(f"TTC: {ttc:.2f}s")
        if ttc < 0.5:
            risk_level = "CRITICAL"
        elif ttc < 2.0:
            risk_level = "HIGH RISK"
        else:
            risk_level = "MEDIUM"
        info_lines.append(f"Risk: {risk_level}")
    
    # Draw information text in top-right corner
    h, w = frame.shape[:2]
    x_offset = w - 350
    y_offset = 30
    line_height = 30
    
    # Background rectangle
    box_height = len(info_lines) * line_height + 20
    cv2.rectangle(frame, (x_offset - 10, y_offset - 20),
                  (w - 10, y_offset + box_height),
                  (0, 0, 0), -1)
    
    # Draw text
    for i, line in enumerate(info_lines):
        color = (0, 0, 255) if "CRITICAL" in line or "HIGH" in line else (0, 255, 255)
        cv2.putText(frame, line, (x_offset, y_offset + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame


def visualize_collision_events(video_path, near_misses, tracks, output_dir, top_k=10):
    """
    Extract and visualize the most critical collision events
    
    Parameters:
      video_path: Path to input video file
      near_misses: List of collision events
      tracks: Object trajectory data
      output_dir: Output directory for annotated frames
      top_k: Number of top events to visualize (sorted by risk level)
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Sort events by risk level (TTC first, then distance)
    sorted_events = sorted(
        near_misses,
        key=lambda x: (
            x.get('ttc') if x.get('ttc') is not None else float('inf'),
            -x['distance']  # Same TTC: prioritize smaller distances
        )
    )
    
    print(f"\n{'='*60}")
    print(f"Visualizing Top {min(top_k, len(sorted_events))} Collision Events")
    print(f"{'='*60}\n")
    
    saved_frames = []
    
    for event_idx, event in enumerate(sorted_events[:top_k]):
        frame_num = int(event['timestamp'])
        obj1_id = event['id1']
        obj2_id = event['id2']
        distance = event['distance']
        ttc = event['ttc']
        
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"⚠ Cannot read frame {frame_num}")
            continue
        
        # Get object information
        obj1_sample = get_object_info_at_frame(tracks, obj1_id, frame_num)
        obj2_sample = get_object_info_at_frame(tracks, obj2_id, frame_num)
        
        # Draw object 1 (blue)
        if obj1_sample:
            frame = draw_detection_box(frame, obj1_sample, obj1_id, color=(255, 0, 0))
            frame = draw_contact_points(frame, obj1_sample, obj1_id, color=(255, 0, 0))
        
        # Draw object 2 (green)
        if obj2_sample:
            frame = draw_detection_box(frame, obj2_sample, obj2_id, color=(0, 255, 0))
            frame = draw_contact_points(frame, obj2_sample, obj2_id, color=(0, 255, 0))
        
        # Draw collision information
        frame = draw_collision_info(frame, event, obj1_sample, obj2_sample, fps)
        
        # Save frame
        risk_level = "CRITICAL" if ttc and ttc < 0.5 else ("HIGH" if ttc and ttc < 2.0 else "MED")
        output_filename = (
            f"collision_event_{event_idx+1:03d}_"
            f"frame_{frame_num:05d}_"
            f"obj{obj1_id}_vs_obj{obj2_id}_"
            f"{risk_level}.jpg"
        )
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), frame)
        saved_frames.append(output_path)
        
        # Print information
        ttc_str = f"{ttc:.2f}s" if ttc is not None else "N/A"
        print(f"✓ Event {event_idx+1}:")
        print(f"    Frame {frame_num} | Object {obj1_id} <-> {obj2_id}")
        print(f"    Distance: {distance:.2f} | TTC: {ttc_str} | {risk_level} RISK")
        
        if 'closest_point_pair' in event:
            cpp = event['closest_point_pair']
            print(f"    Contact: {cpp['obj1_point_type']} <-> {cpp['obj2_point_type']}")
        print()
    
    cap.release()
    
    # Generate summary report
    summary_path = output_dir / "collision_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("COLLISION EVENTS VISUALIZATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total events visualized: {len(saved_frames)}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        f.write("Saved files:\n")
        for path in saved_frames:
            f.write(f"  - {path.name}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Color Coding:\n")
        f.write("  Object 1 (Blue):  Detection box + contact points\n")
        f.write("  Object 2 (Green): Detection box + contact points\n")
        f.write("  Info Box:         Distance, TTC, Contact type, Risk level\n")
        f.write("=" * 60 + "\n")
    
    print(f"\n✓ Summary saved: {summary_path}")
    print(f"✓ All frames saved to: {output_dir}\n")
    
    return saved_frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize collision events on video frames')
    parser.add_argument('--near-misses', required=True, help='Path to collision events JSON file')
    parser.add_argument('--tracks', required=True, help='Path to object trajectories JSON file')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--output', default='collision_frames', help='Output directory for annotated frames')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top critical events to visualize')
    
    args = parser.parse_args()
    
    # Load data
    near_misses, tracks = load_data(args.near_misses, args.tracks)
    
    # Visualize collision events
    visualize_collision_events(args.video, near_misses, tracks, args.output, args.top_k)
