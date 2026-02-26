#!/usr/bin/env python3
"""
test_yolo_first_method_c.py

Test script for YOLO-First Method C (Homography-First) approach
Demonstrates complete five-step pipeline execution
"""

import os
import sys
import json
from pathlib import Path

# 添加路径
sys.path.append(os.path.dirname(__file__))
from collision_detection_pipeline_yolo_first_method_c import YOLOFirstPipelineC


def test_pipeline():
    """Test Method C pipeline"""
    
    # Configure paths
    video_path = "../../videos/Homograph_Teset_FullScreen.mp4"
    homography_path = "../../calibration/Homograph_Teset_FullScreen_homography.json"
    output_base = "../../results"
    
    # Check input files
    print("\n" + "="*70)
    print("YOLO-First Method C Test Script")
    print("="*70)
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return False
    
    if not Path(homography_path).exists():
        print(f"❌ Homography file not found: {homography_path}")
        return False
    
    print(f"\n✓ Input file validation passed")
    print(f"  Video: {Path(video_path).name}")
    print(f"  Homography: {Path(homography_path).name}")
    
    # Create pipeline instance
    print(f"\n[Initialize Pipeline]")
    pipeline = YOLOFirstPipelineC(
        video_path=video_path,
        homography_path=homography_path,
        output_base=output_base
    )
    
    # Run pipeline
    print(f"\n[Run Complete Pipeline]")
    pipeline.run(conf_threshold=0.45)
    
    # Verify output
    print(f"\n[Verify Output Files]")
    output_dir = pipeline.run_dir
    
    expected_files = [
        "1_raw_detections/detections_pixel.json",
        "1_raw_detections/detection_stats.json",
        "2_homography_transform/homography.json",
        "2_homography_transform/detections_world.json",
        "3_trajectories/tracks_world.json",
        "3_trajectories/track_stats.json",
        "4_key_frames/proximity_events.json",
        "5_collision_analysis/collision_events.json",
        "5_collision_analysis/analysis_report.txt",
    ]
    
    all_exist = True
    for file_path in expected_files:
        full_path = output_dir / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"  ✓ {file_path} ({size} bytes)")
        else:
            print(f"  ❌ {file_path} (missing)")
            all_exist = False
    
    if all_exist:
        print(f"\n✓ All output files generated")
    else:
        print(f"\n❌ Some output files missing")
        return False
    
    # Analyze results
    print(f"\n[Analysis Results Summary]")
    
    try:
        # Read analysis report
        report_path = output_dir / "5_collision_analysis/analysis_report.txt"
        with open(report_path, 'r') as f:
            report_content = f.read()
        
        # Extract key information
        print(f"\nReport content preview:")
        print("-" * 70)
        print(report_content[:800] + "\n...")
        
        # Read event statistics
        events_path = output_dir / "5_collision_analysis/collision_events.json"
        with open(events_path, 'r') as f:
            events = json.load(f)
        
        print(f"\nEvent statistics:")
        print(f"  Total events: {len(events)}")
        
        if events:
            levels = {1: 0, 2: 0, 3: 0}
            for event in events:
                if 'level' in event:
                    levels[event['level']] += 1
            
            print(f"  - L1 (Collision): {levels[1]}")
            print(f"  - L2 (Near Miss): {levels[2]}")
            print(f"  - L3 (Avoidance): {levels[3]}")
            
            # Display highest risk events
            if levels[1] > 0 or levels[2] > 0:
                high_risk = [e for e in events if e.get('level') in [1, 2]]
                print(f"\nHighest risk events (first 3):")
                for i, event in enumerate(high_risk[:3], 1):
                    dist = event.get('distance_meters', 0)
                    print(f"  {i}. Frame {event['frame']} - " +
                          f"distance {dist:.2f}m - Level {event['level']} ({event.get('level_name', '?')})")
        
        # Read trajectory statistics
        track_stats_path = output_dir / "3_trajectories/track_stats.json"
        with open(track_stats_path, 'r') as f:
            track_stats = json.load(f)
        
        print(f"\nTrajectory statistics:")
        print(f"  Total tracks: {track_stats['total_tracks']}")
        print(f"  Coordinate system: {track_stats['coordinate_system']}")
        print(f"  Velocity unit: {track_stats['velocity_unit']}")
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return False
    
    print(f"\n" + "="*70)
    print(f"✓ Test completed! Results saved to: {output_dir}")
    print(f"="*70)
    
    return True


if __name__ == '__main__':
    success = test_pipeline()
    sys.exit(0 if success else 1)
