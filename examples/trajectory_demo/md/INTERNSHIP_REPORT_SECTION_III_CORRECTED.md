# Section III: System Implementation

## Overview

The near-miss detection system employs a YOLO-first processing pipeline designed to achieve real-time performance while maintaining high accuracy. The pipeline is structured in five sequential stages that progressively refine detections into actionable collision risk assessments. This section details each stage's design and implementation.

## III.A: YOLO Object Detection (Stage 1)

The system begins with YOLOv11-nano, a lightweight real-time object detector. Running at full video resolution without frame skipping initially proved computationally expensive. To optimize performance while preserving tracking continuity, we implemented a frame-skipping mechanism that processes every Nth frame (default: N=3) for YOLO inference. Despite the frame skipping, motion estimation remains accurate because trajectories bridge the unprocessed frames using interpolation based on velocity estimates calculated from consecutive detections.

The detector operates at a confidence threshold of 0.45, configured to favor recall over precision—a deliberate choice since false positives are filtered in downstream stages, while false negatives at this stage cannot be recovered. The detector outputs bounding boxes in xywh format (center coordinates and width/height) along with YOLO's native Track IDs, which provide multi-object tracking capabilities without requiring external tracking frameworks.

## III.B: Trajectory Construction (Stage 2)

After detection, the system constructs object trajectories by associating Track IDs across frames. For each tracked object, we maintain a trajectory consisting of:
- **Pixel-space coordinates** (original detection centers)
- **World-space coordinates** (after homography transformation)
- **Velocity vectors** in both coordinate systems
- **Temporal metadata** (frame number, timestamp)

A critical design decision here is the timing of coordinate transformation. Rather than transforming all detections immediately, we transform coordinates while building trajectories, storing both pixel and world coordinates for each trajectory point. This dual-space representation enables both pixel-based key-frame identification (efficient) and world-based risk analysis (accurate).

Velocity estimation uses first-order finite differences: if an object appears at position P₁ at time T₁ and position P₂ at time T₂, we compute velocity as V = (P₂ - P₁)/(T₂ - T₁). For skipped frames, the denominator accounts for the actual time elapsed (accounting for frame skipping). We apply a minimum trajectory length filter (3 frames minimum) to eliminate brief, potentially spurious detections.

## III.C: Key-Frame Detection (Stage 3)

The system next identifies "key frames"—temporal snapshots where two objects enter a proximity event. Rather than analyzing every frame, this stage filters to frames where objects are sufficiently close to warrant detailed examination. The proximity threshold is set at **3.0 meters in world coordinates**. 

This threshold reflects a trade-off: 3.0m represents the approximate distance at which a driver might begin reacting to a nearby vehicle; closer distances (< 3.0m) indicate higher collision risk. All pairs of objects with minimum distance ≤ 3.0m at any frame are marked as proximity events.

For each proximity event, we record:
- **Object identities** (Track IDs and class labels)
- **Temporal information** (frame number, timestamp)
- **Spatial data** (positions in pixel and world coordinates)
- **Multi-anchor distances** (see Section III.D below)

The system also applies class-based filtering: if two objects belong to the same class (two cars, two bicycles, etc.) and are extremely close (< 0.5m apart), this suggests detection fragmentation rather than genuine collision risk, and such pairs are flagged for filtering in later stages.

## III.D: Multi-Anchor Distance Measurement

A novel aspect of our implementation is **multi-anchor distance measurement**. Rather than measuring distance between bounding box centers alone, we define multiple reference points ("anchors") on each object that represent structurally or collision-relevantly distinct locations:

**For vehicles** (cars, buses, trucks): Eight anchors are defined—
- Front points: front_center, front_left, front_right
- Rear points: rear_center, rear_left, rear_right  
- Side points: left_center, right_center

**For pedestrians**: Four anchors—head (most vulnerable), torso, lower body, and feet

**For motorcycles and bicycles**: Analogous sets of structural points

Each anchor point's position is computed as a function of the bounding box:
$$\text{Anchor}_{\text{name}} = \text{BBox}_{\text{center}} + \text{offset}_{\text{name}} \times \text{shrink\_factor}$$

where the shrink_factor (default: 0.75) brings vertex-based anchors slightly toward the bounding box center to avoid overshooting for loose bounding boxes.

The minimum distance between two objects is then computed as:
$$d_{\min} = \min_{i,j} ||\text{Anchor}_i(\text{Obj}_1) - \text{Anchor}_j(\text{Obj}_2)||$$

This approach better captures collision proximity than simple center-to-center distance, particularly for large vehicles and across different object classes.

## III.E: Homography Transformation (Stage 4)

Homography calibration maps pixel coordinates to real-world (world-space) coordinates using a perspective transformation matrix H computed from calibration points. At initialization, the system loads or computes H based on manually-annotated correspondences between image pixels and known world positions (e.g., lane markings, road infrastructure).

The transformation is applied to trajectory coordinates using:
$$\begin{bmatrix} x_w \\ y_w \\ 1 \end{bmatrix} = H \begin{bmatrix} x_p \\ y_p \\ 1 \end{bmatrix}$$

where (xₚ, yₚ) are pixel coordinates and (xw, yw) are world coordinates. Importantly, this transformation is applied during trajectory construction (Stage 2), not retroactively. The resulting scale factor—typically ~30.5 pixels/meter for standard camera views—enables accurate distance measurements in meters.

## III.F: Risk Classification (Stage 5)

The final stage classifies proximity events into risk levels based on distance in world coordinates:

- **Level 1 (Collision)**: distance < 0.5m
- **Level 2 (Near-Miss)**: 0.5m ≤ distance < 1.5m  
- **Level 3 (Avoidance)**: distance ≥ 1.5m

These thresholds were informed by road safety literature and AT's operational guidelines. Level 1 represents objects in actual contact or extreme proximity. Level 2 captures "near misses"—events where collision was avoided but the gap was critically small. Level 3 represents avoidable encounters.

Additionally, the system computes optional Time-to-Collision (TTC) metrics based on relative velocity:
$$\text{TTC} = \frac{d_{\min}}{||\vec{v}_{\text{rel}}||_2}$$

where d_min is the minimum distance and v_rel is the relative velocity vector. TTC provides temporal context for risk assessment and can identify accelerating conflicts even when current distances appear safe.

## III.G: Development Methodology

The implementation followed an iterative refinement cycle:

1. **Baseline pipeline** (Method A): YOLO detection → pixel-space trajectories → proximity events → output
2. **Coordinate transformation integration**: Added homography transformation, revealing the need for calibration procedures
3. **Multi-anchor measurement**: Discovered that center-to-center distance underestimated collision risk across object classes; implemented anchor points to capture the closest approach
4. **Ground truth validation**: Compared system outputs against manually-annotated collision events, identifying systematic biases (e.g., vehicle front corners often closer than centers)
5. **Parameter tuning**: Refined distance thresholds, frame-skip rates, and confidence levels based on test results

Throughout development, we maintained a modular architecture where each stage operates independently and can be tested or modified without affecting others. The output of each stage is persisted as JSON, facilitating debugging and enabling re-runs of downstream stages without re-executing computationally expensive early stages.

## III.H: Technical Challenges and Solutions

**Challenge 1: Coordinate system inconsistency**  
Early versions mixed pixel-space and world-space coordinates, causing unit mismatches in distance calculations. Solution: Maintain explicit separation of coordinate systems in data structures, with clear labeling (e.g., `distance_pixels` vs. `distance_meters`, `track_point_px` vs. `track_point_world`).

**Challenge 2: Frame skipping and trajectory discontinuity**  
Processing every 3rd frame created large gaps where moving objects' positions couldn't be tracked. Solution: Implement velocity-based interpolation to estimate positions in skipped frames, smoothing the gap and enabling accurate speed estimation across frame-skipped data.

**Challenge 3: Detection fragmentation**  
YOLO occasionally detected a single vehicle as multiple overlapping bounding boxes, creating spurious proximity events. Solution: Implement same-class filtering (Section III.C) and multi-anchor distance filtering to identify and suppress these false positives.

**Challenge 4: Homography calibration accuracy**  
Camera calibration from manually-annotated points introduced errors that cascaded through distance calculations. Solution: Implement calibration verification visualizations and compute calibration error on reference points, accepting only matrices with < 0.1% error.

