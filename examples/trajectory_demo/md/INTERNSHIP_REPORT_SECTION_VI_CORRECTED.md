# Section VI: Limitations and Future Work

## A. Current System Limitations

While the prototype successfully demonstrates the feasibility of near-miss detection on CCTV footage, several limitations constrain its current applicability and inform directions for future refinement.

### 1) Tracking Consistency Under Challenging Conditions

The current tracking-by-detection approach using YOLO's native tracking is effective under normal traffic conditions but degrades when occlusions are frequent or extended. For example, in dense traffic scenarios (not present in our test video), vehicles frequently overlap, and brief ID switches can disrupt trajectory continuity. This affects TTC/PET computation because the kinematic history is broken when IDs are mismatched.

Future work should explore re-identification (re-ID) features, either through fine-tuning a CNN on AT's intersection footage or adopting a more sophisticated tracker such as DeepSORT. However, this must be weighed against computational cost and the empirical rarity of such scenarios at the pilot intersection.

### 2) Homography Calibration and Far-Field Accuracy

The homography approach assumes a planar road surface (Z = 0), which is valid for flat intersections but breaks down on curved or hilly terrain. Additionally, while calibration error on reference points was < 0.1%, errors in far-field regions (distant background) can exceed 5–10%, leading to inaccurate distance estimates for vehicles far from the camera. This is partially mitigated by world-space distance thresholding (far vehicles are automatically excluded), but a vehicle at 30 meters might be misclassified as 25 meters.

For future deployment across AT's network—which includes hilly terrain in suburban areas—we should consider multi-region homography (dividing the image into piecewise planar regions) or full camera calibration using Tsai's method to recover intrinsic and extrinsic parameters. These approaches are more computationally expensive but provide sub-pixel accuracy.

### 3) Weather and Lighting Robustness

The test video was recorded under clear daylight conditions. CCTV footage under rain, night, or glare presents significantly different challenges: reduced contrast, motion blur from rain streaks, and color shifts under sodium-vapor lighting. YOLO's accuracy may degrade substantially, and homography calibration may become unreliable if reference landmarks are obscured or poorly lit.

We have not yet evaluated the system under these conditions. This is a critical gap because Auckland's weather is variable and many intersections lack sufficient lighting. Future validation must include diverse weather and temporal conditions.

### 4) Limited Ground-Truth Validation Data

The system incorporates automated ground-truth annotation (`quick_annotation.py`) that labels events based on TTC/PET presence, and a metrics analysis framework (`metric_comparison_analysis.py`) that computes Precision, Recall, and F1 scores. However, this validation has been performed only on a single test video and relies on automatic annotation rules rather than independent human expert judgment.

Ground-truth annotation on a broader dataset is labor-intensive and requires domain expertise from traffic safety engineers—what constitutes a genuine collision risk vs. a false alarm can be ambiguous in edge cases. Before claiming production readiness, we must:
- Conduct human expert review of 50–100 events to establish inter-annotator agreement
- Validate automatic annotation logic against this expert consensus
- Test on representative samples from multiple intersections and traffic conditions

## B. Future Development Roadmap

### 1) Phase 2: Expanded Validation (Weeks 11–16)

Immediate priorities:
- Conduct human expert annotation of 50–100 frames from the pilot intersection
- Compute Precision, Recall, F1, and Confusion Matrix against expert labels
- Test on 3–5 additional intersections to assess generalization robustness
- Evaluate system performance under diverse weather (rain, overcast, night) and lighting conditions
- Compare automated metrics (Precision/Recall) across different distance thresholds to optimize for AT's operational needs

**Deliverable**: Validation report with metrics broken down by intersection and weather condition.

### 2) Phase 3: Advanced Tracking and Optimization (Weeks 17–24)

- Integrate re-identification features via DeepSORT or fine-tuned CNN backbone
- Validate PET computation against ground-truth near-miss events and optimize rear-end/sideswipe detection thresholds
- Implement multi-region homography to improve accuracy for non-planar terrain
- Develop acceleration estimation for kinematic analysis (currently assumes constant velocity)
- Establish confidence intervals for TTC/PET to quantify uncertainty in safety-critical estimates

**Deliverable**: Optimized tracker with documented accuracy improvements on multi-intersection data.

### 3) Phase 4: Integration with Theia Platform (Weeks 25–32)

- Adapt the pipeline to ingest video streams from Theia's CCTV infrastructure
- Integrate with CVT's Triton inference service for scalable GPU-accelerated deployment
- Develop web dashboard for visualization and filtering of near-miss events
- Establish feedback loops with AT operations teams to validate thresholds and refine alert policies
- Implement real-time alert mechanisms for safety planners (e.g., email/SMS for high-risk events)

**Deliverable**: Pilot deployment on 1–2 intersections with operator feedback collection.

### 4) Phase 5: Network-Wide Deployment (Weeks 33+)

- Deploy across AT's 100+ instrumented intersections
- Build multi-region homography models calibrated for AT's diverse intersection geometries
- Develop adaptive thresholding based on empirical collision risk profiles per intersection
- Integrate with AT's injury collision database to correlate near-miss detection with actual crashes
- Establish continuous retraining pipeline to maintain detector accuracy as lighting and infrastructure change

**Deliverable**: Production system monitoring AT's network with periodic safety impact reports.

## C. Broader Implications and Lessons Learned

This internship has demonstrated that computer vision can provide actionable safety insights from existing CCTV infrastructure. However, the path from prototype to production involves not just technical refinement but also stakeholder engagement, validation rigor, and organizational integration. 

**Key insight on interpretability**: The homography transformation proved critical for stakeholder buy-in and understanding. When we presented results in "150 pixels," planners were skeptical. When we showed "1.5 meters—the width of a car"—a world-grounded metric—the system's outputs became immediately interpretable and credible. This insight carries profound implications for future CV work at AT: interpretability and world-grounded metrics should be design principles from the outset, not afterthoughts. Pixels are implementation details; meters and seconds are operational language.

**On agile methodology in research**: Working within AT's Scrum-of-Scrums framework exposed the tension between research iteration and delivery commitments. Unlike traditional software sprints, CV research benefits from "exploration sprints" where the goal is learning (whether something works), not shipping features. Negotiating this with product and operations teams was a key professional lesson—establishing clear acceptance criteria and knowing when validation is "sufficient" rather than "perfect" is crucial.

**On team dynamics**: The AT Computer Vision team's psychological safety and rapid feedback culture accelerated learning dramatically. The ability to ask "why would this fail?" in daily stand-ups without defensive reactions enabled hypothesis-driven debugging rather than random parameter tuning. This is a replicable competitive advantage for AT's future CV initiatives.

