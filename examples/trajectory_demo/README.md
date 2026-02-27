# YOLO-First Collision Detection Pipeline

A real-time collision risk analysis system based on YOLO11 object detection and Homography perspective transformation.

## 🚀 Quick Start

### Install Dependencies

```bash
pip install ultralytics opencv-python numpy matplotlib reportlab
```

### Run Example

```bash
python test_with_short_video.py
```

Or call the Pipeline directly:

```python
from collision_detection_pipeline_yolo_first_method_a import YOLOFirstPipelineA

pipeline = YOLOFirstPipelineA(
    video_path='../../videos/Homograph_Teset_FullScreen.mp4',
    homography_path='../../calibration/Homograph_Teset_FullScreen_homography.json',
    skip_frames=3
)
pipeline.run()
```

## 📋 Project Structure

```
├── collision_detection_pipeline_yolo_first_method_a.py  # Main Pipeline class
├── collision_analyzer.py                                 # Collision analysis algorithm
├── anchor_points.py                                      # Anchor point configuration
├── config_loader.py                                      # Configuration management
├── visualize_collision_events.py                         # Event visualization
├── visualize_contact_points.py                           # Contact point visualization
├── visualize_results.py                                  # Result visualization
├── run_with_visualization.sh                             # Run script
├── configs/                                              # Configuration files
├── calibration/                                          # Calibration data
├── ground_truth_annotations/                             # Annotation data
├── PreviousAttempts/                                     # Historical implementations
└── test/                                                 # Test scripts
```

## 🔧 Core Features

### Step 1: YOLO Detection
- Real-time object detection using YOLO11
- Support for multiple YOLO models (yolo11n, yolo11s, yolo11m, etc.)
- Configurable confidence threshold

### Step 2: Trajectory Building
- Track object motion in both pixel and world coordinates
- Trajectory continuity checking
- Short trajectory filtering

### Step 3: Key Frame Extraction
- Identify key frames of proximity events (distance < 3.0m)
- Generate proximity_events.json

### Step 4: Homography Transformation
- Map pixel coordinates to world coordinates
- Eliminate perspective distortion
- Automatic transformation quality verification

### Step 5: Collision Analysis
- Multi-anchor point collision detection
- Calculate TTC (Time-To-Collision)
- Risk level classification (Level 0-3)
- Generate PDF visualization report

## 📊 Output Results

After running, results are saved in the `results/` directory with 5 subdirectories:

```
results/Homograph_Teset_FullScreen_YYYYMMDD_HHMMSS/
├── 1_yolo_detection/        # YOLO detection results + detection images
├── 2_trajectories/          # Trajectory data
├── 3_key_frames/            # Key frame visualization
├── 4_homography_transform/  # World coordinate data
└── 5_collision_analysis/    # Collision analysis + PDF report
```

## ⚙️ Parameter Configuration

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `video_path` | Input video file | Required |
| `homography_path` | Homography calibration file | Required |
| `skip_frames` | Frame skipping | 1 |
| `model` | YOLO model | yolo11n |
| `conf_threshold` | Confidence threshold | 0.45 |
| `world_distance_threshold` | Proximity distance threshold | 3.0m |

### Custom Usage

```python
pipeline = YOLOFirstPipelineA(
    video_path='video.mp4',
    homography_path='homography.json',
    skip_frames=2,              # Process every 2 frames
    model='yolo11m',            # More accurate model
    output_base='./results'
)

# Run YOLO detection
all_detections = pipeline.run_yolo_detection(conf_threshold=0.5)

# Build trajectories
tracks = pipeline.build_trajectories(all_detections)

# Extract key frames
proximity_events = pipeline.extract_key_frames(all_detections, tracks)

# Complete run
pipeline.run()
```

## 🎯 Technical Highlights

- **Accurate**: Multi-anchor point collision detection, not just center-to-center distance
- **Efficient**: skip_frames mechanism increases processing speed by 3x
- **Practical**: Automatic Homography perspective transformation eliminates viewing angle distortion
- **Complete**: End-to-end pipeline from detection to analysis with visualization and PDF reports
- **Maintainable**: Clear code structure with well-organized modules

## ❓ FAQ

**Q: No objects detected**  
A: Adjust confidence threshold: `pipeline.run_yolo_detection(conf_threshold=0.3)`

**Q: What is the Homography file format?**  
A: JSON format containing homography_matrix and scale_factor_px_per_meter

**Q: How to process new videos?**  
A: Generate corresponding Homography calibration file for new videos, see calibration directory

**Q: Processing is too slow?**  
A: Use skip_frames parameter for frame skipping, or use a smaller YOLO model

## 📞 Related Files

| File | Description |
|------|-------------|
| `test/test_method_a.py` | Test script |
| `test/verify_homography.py` | Calibration verification script |
| `run_with_visualization.sh` | Visualization run script |
| `configs/` | Configuration file templates |
| `PreviousAttempts/` | Historical implementations and alternatives |

## 📝 Last Updated

- **Date**: 2026-02-26
- **Version**: 1.0 (Method A)
- **Status**: ✅ Production Ready
