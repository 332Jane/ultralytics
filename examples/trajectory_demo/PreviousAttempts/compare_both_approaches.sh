#!/bin/bash

# [DEPRECATED] compare_both_approaches.sh
# Run two pipelines on the same video to generate comparison results

cd /workspace/ultralytics/examples/trajectory_demo

echo "=========================================="
echo "Running two approach comparison test"
echo "=========================================="
echo ""

VIDEO="../../videos/Homograph_Teset_FullScreen.mp4"
HOMOGRAPHY="../../calibration/Homograph_Teset_FullScreen_homography.json"

# Check files
if [ ! -f "$VIDEO" ]; then
    echo "❌ Video file not found: $VIDEO"
    exit 1
fi

if [ ! -f "$HOMOGRAPHY" ]; then
    echo "❌ Homography file not found: $HOMOGRAPHY"
    exit 1
fi

echo "✓ File check completed"
echo "  Video: $VIDEO"
echo "  Homography: $HOMOGRAPHY"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

echo "=========================================="
echo "运行方案1: Homography-First"
echo "=========================================="
python collision_detection_pipeline.py \
    --video "$VIDEO" \
    --homography "$HOMOGRAPHY" \
    --conf 0.45

HOMOGRAPHY_FIRST_TIME=$(date +%s)
HOMOGRAPHY_FIRST_DURATION=$((HOMOGRAPHY_FIRST_TIME - START_TIME))

echo ""
echo "=========================================="
echo "Running Method 2: YOLO-First Method C"
echo "=========================================="
python collision_detection_pipeline_yolo_first_method_c.py \
    --video "$VIDEO" \
    --homography "$HOMOGRAPHY" \
    --conf 0.45

YOLO_FIRST_TIME=$(date +%s)
YOLO_FIRST_DURATION=$((YOLO_FIRST_TIME - HOMOGRAPHY_FIRST_TIME))

echo ""
echo "=========================================="
echo "Performance Comparison Results"
echo "=========================================="
echo "Method 1 (Homography-First): ${HOMOGRAPHY_FIRST_DURATION}s"
echo "Method 2 (YOLO-First):       ${YOLO_FIRST_DURATION}s"

if [ $YOLO_FIRST_DURATION -gt 0 ]; then
    SPEEDUP=$(echo "scale=2; $HOMOGRAPHY_FIRST_DURATION / $YOLO_FIRST_DURATION" | bc)
    echo "Speedup: ${SPEEDUP}x"
fi

echo ""
echo "✓ Comparison completed, please check results directory"
echo "  Results: ../../results/"
