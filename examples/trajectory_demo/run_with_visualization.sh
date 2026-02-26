#!/bin/bash

################################################################################
# COLLISION DETECTION PIPELINE WITH VISUALIZATION
################################################################################
#
# PURPOSE:
#   This script is a convenient wrapper for the Method A collision detection
#   pipeline (YOLOFirstPipelineA). It simplifies parameter passing and provides
#   automated result visualization.
#
# FUNCTIONALITY:
#   - Runs YOLO-based collision detection on video input
#   - Builds object trajectories in pixel coordinates
#   - Identifies key frames with proximity events
#   - Applies Homography transformation for world-coordinate analysis
#   - Generates collision events and visualizations
#
# USAGE:
#   Basic run with default homography:
#     ./run_with_visualization.sh --video path/to/video.mp4
#
#   Custom parameters:
#     ./run_with_visualization.sh \
#       --video path/to/video.mp4 \
#       --homography path/to/homography.json \
#       --skip-frames 5 \
#       --conf 0.5 \
#       --model yolo11m
#
# PARAMETERS:
#   --video <path>            : Input video file path (REQUIRED)
#   --homography <path>       : Homography matrix JSON file
#                               (Default: calibration/Homograph_Teset_FullScreen_homography.json)
#   --skip-frames <int>       : Frame skip factor for YOLO inference (Default: 3)
#   --conf <float>            : YOLO confidence threshold (Default: 0.45)
#   --model <name>            : YOLO model size (yolo11n/yolo11m/yolo11l, Default: yolo11m)
#
# OUTPUT:
#   Results are saved in: /workspace/ultralytics/results/<video_name>_<timestamp>/
#   Subdirectories:
#     - 1_yolo_detection/          : Raw YOLO detections
#     - 2_trajectories/            : Object trajectories (pixel coords)
#     - 3_key_frames/              : Proximity events detected
#     - 4_homography_transform/    : World coordinate transformations
#     - 5_collision_analysis/      : Final collision events and reports
#
# DEPENDENCIES:
#   - collision_detection_pipeline_yolo_first_method_a.py (Main pipeline)
#   - visualize_results.py (Optional visualization tool)
#   - YOLO model weights (auto-downloaded if missing)
#   - Homography calibration JSON (optional)
#
# EXAMPLE OUTPUT:
#   Near-miss events with distance, TTC (Time-To-Collision), and contact points
#
################################################################################

# Default parameters
VIDEO_PATH=""
HOMOGRAPHY_PATH="calibration/Homograph_Teset_FullScreen_homography.json"
SKIP_FRAMES=3
CONF=0.45
MODEL="yolo11m"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --video)
            VIDEO_PATH="$2"
            shift 2
            ;;
        --homography)
            HOMOGRAPHY_PATH="$2"
            shift 2
            ;;
        --skip-frames)
            SKIP_FRAMES="$2"
            shift 2
            ;;
        --conf)
            CONF="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$VIDEO_PATH" ]; then
    echo "ERROR: Video path is required"
    echo "Usage: ./run_with_visualization.sh --video path/to/video.mp4"
    exit 1
fi

echo "=========================================="
echo "COLLISION DETECTION PIPELINE WITH VISUALIZATION"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Video:              $VIDEO_PATH"
echo "  Homography:         $HOMOGRAPHY_PATH"
echo "  Frame skip factor:  $SKIP_FRAMES"
echo "  Confidence thresh:  $CONF"
echo "  YOLO model:         $MODEL"
echo ""

# Step 1: Run collision detection pipeline
echo "Step 1: Running collision detection pipeline..."
python collision_detection_pipeline_yolo_first_method_a.py \
    --video "$VIDEO_PATH" \
    --homography "$HOMOGRAPHY_PATH" \
    --skip-frames $SKIP_FRAMES \
    --conf $CONF \
    --model $MODEL

# Find the latest results directory
LATEST_RESULT=$(ls -td /workspace/ultralytics/results/*/20* 2>/dev/null | head -1)

if [ -z "$LATEST_RESULT" ]; then
    echo "ERROR: Could not find results directory"
    exit 1
fi

echo ""
echo "Step 2: Generating visualizations..."
python visualize_results.py \
    --video "$VIDEO_PATH" \
    --results "$LATEST_RESULT"

echo ""
echo "=========================================="
echo "✓ PIPELINE COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Output locations:"
echo "  Analysis results:   $LATEST_RESULT"
echo "  Visualizations:     $(dirname $LATEST_RESULT)/visualization"
echo ""
echo "Next steps:"
echo "  1. Review collision events: cat $LATEST_RESULT/5_collision_analysis/collision_events.json"
echo "  2. Check visualizations in the output directory"
echo "  3. Examine HTML reports if generated"
echo ""
