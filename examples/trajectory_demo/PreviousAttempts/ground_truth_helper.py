#!/usr/bin/env python3
"""
Ground Truth Annotation Helper Tool

Purpose: Help manually annotate collision events in videos
Provides: Video analysis, keyframe viewing, easy-to-use annotation interface
"""

import json
from pathlib import Path
from typing import List, Dict
import subprocess

class GroundTruthHelper:
    """Ground Truth annotation helper tool"""
    
    def __init__(self, video_path: str, output_dir: str = None):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir) if output_dir else self.video_path.parent
        
    def get_video_info(self):
        """Get video information"""
        try:
            import subprocess
            cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of csv=p=0 "{self.video_path}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                width, height, fps_str, frames = parts[0], parts[1], parts[2], parts[3]
                
                # Parse fps (may be in format "30/1" or "30")
                if '/' in fps_str:
                    fps = eval(fps_str)
                else:
                    fps = float(fps_str)
                
                total_frames = int(frames) if frames else "unknown"
                total_seconds = total_frames / fps if total_frames != "unknown" else "unknown"
                
                print("\n" + "="*60)
                print("📹 Video Information")
                print("="*60)
                print(f"  File: {self.video_path.name}")
                print(f"  Resolution: {width}x{height}")
                print(f"  Frame Rate: {fps:.2f} fps")
                print(f"  Total Frames: {total_frames}")
                print(f"  Duration: {total_seconds:.1f}s" if total_seconds != "unknown" else "  Duration: unknown")
                print("="*60 + "\n")
                
                return {
                    "width": int(width),
                    "height": int(height),
                    "fps": fps,
                    "total_frames": total_frames,
                    "duration": total_seconds
                }
            else:
                print(f"⚠️ Failed to get video information: {result.stderr}")
                return None
        except Exception as e:
            print(f"⚠️ Failed to get video information: {e}")
            return None
    
    def create_annotation_template(self, video_info: Dict = None):
        """Create annotation template"""
        
        template = {
            "metadata": {
                "video_file": self.video_path.name,
                "video_path": str(self.video_path),
                "annotation_date": "YYYY-MM-DD",
                "annotator": "Your Name",
                "notes": "Add any additional notes here"
            },
            "video_info": video_info if video_info else {},
            "instructions": {
                "event_types": {
                    "collision": "Two objects in contact or distance < 0.5m",
                    "near_miss": "Objects passing closely, distance 0.5-1.5m",
                    "close_approach": "Distance 1.5-3m, objects approaching each other"
                },
                "severity_levels": {
                    "critical": "Imminent collision risk (distance<0.5m or TTC<0.5s)",
                    "high": "High collision risk (distance 0.5-1.5m or TTC<2s)",
                    "medium": "Medium risk (distance 1.5-3m or TTC<5s)",
                    "low": "Low risk (distance>3m)"
                }
            },
            "ground_truth_events": [
                {
                    "frame": "Video frame number (starting from 0)",
                    "time_seconds": "Time in seconds",
                    "object_1": {
                        "class": "vehicle|person|motorcycle|bicycle|truck|bus|car",
                        "description": "Description of object 1"
                    },
                    "object_2": {
                        "class": "vehicle|person|motorcycle|bicycle|truck|bus|car",
                        "description": "Description of object 2"
                    },
                    "event_type": "collision|near_miss|close_approach",
                    "severity": "critical|high|medium|low",
                    "distance_estimate": "Estimated distance (meters)",
                    "visible_interaction": "Describe how objects interact",
                    "notes": "Additional notes"
                }
            ],
            "summary": {
                "total_events": 0,
                "collision_count": 0,
                "near_miss_count": 0,
                "close_approach_count": 0,
                "reviewed_minutes": "Annotation time spent (minutes)"
            }
        }
        
        return template
    
    def save_template(self, template: Dict, output_file: str = None):
        """Save annotation template"""
        if output_file is None:
            output_file = self.output_dir / "ground_truth_template.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Template saved: {output_file}")
        return output_file
    
    def print_quick_reference(self):
        """Print quick reference guide"""
        print("\n" + "="*70)
        print("🎯 Quick Reference Guide")
        print("="*70)
        print("""
Collision Classification Standards:

1️⃣  COLLISION
   - Two objects in contact or extremely close (< 0.5m)
   - Severity: CRITICAL
   - Example: Vehicle collision, serious scraping

2️⃣  NEAR MISS
   - Two objects passing closely (0.5m - 1.5m)
   - Severity: HIGH
   - Example: Vehicle overtaking within 1.5m distance, pedestrian brushing past

3️⃣  CLOSE APPROACH
   - Two objects approaching each other (1.5m - 3m)
   - Severity: MEDIUM / LOW
   - Example: Vehicle gradually approaching another vehicle (but will not collide)

Tips:
  • Prioritize marking COLLISION and NEAR MISS events
  • Only mark real interactions (not coincidental proximity)
  • Use pipeline output keyframe images as reference
  • Watch video, not just keyframes - understand complete object trajectories
  • Record accurate frame numbers (corresponding to video position)

""")
        print("="*70 + "\n")

if __name__ == "__main__":
    # Usage example
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ground_truth_helper.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    helper = GroundTruthHelper(video_path)
    
    # 1. Get video information
    video_info = helper.get_video_info()
    
    # 2. Print quick reference
    helper.print_quick_reference()
    
    # 3. Create and save annotation template
    template = helper.create_annotation_template(video_info)
    output_path = helper.save_template(template)
    
    print(f"""
✅ Preparation complete!

Next steps:
  1. Open the annotation template in your preferred editor:
     {output_path}
  
  2. Watch the video and annotate all collision/near-distance events
     (Use the example format in ground_truth_template.json)
  
  3. When finished, save as:
     {helper.output_dir / 'ground_truth_homograph_fullscreen.json'}
  
  4. Run pipeline for accuracy evaluation:
     python collision_detection_pipeline_yolo_first_method_a.py \\
       --video <video_path> --homography <homography_path>
""")
