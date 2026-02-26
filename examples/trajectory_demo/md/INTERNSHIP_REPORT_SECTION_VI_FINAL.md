# Section VI: Limitations and Future Work

## A. Current System Limitations

While the prototype successfully demonstrates the feasibility of near-miss detection on CCTV footage, several limitations constrain its current applicability and inform directions for future refinement.

### 1) Tracking Consistency Under Challenging Conditions

The current tracking-by-detection approach using YOLO's native tracking (persist=True flag) is effective under normal traffic conditions but degrades when occlusions are frequent or extended. For example, in dense traffic scenarios (not present in our test video), vehicles frequently overlap, and brief ID switches can disrupt trajectory continuity. This affects TTC/PET computation because the kinematic history is broken when IDs are mismatched.

Future work should explore more sophisticated tracking approaches to improve robustness under occlusion, either through appearance-based re-identification features or enhanced motion prediction models. However, this must be weighed against computational cost and the empirical rarity of such scenarios at the pilot intersection.

### 2) Homography Calibration and Far-Field Accuracy

The homography approach assumes a planar road surface (Z = 0), which is valid for flat intersections but breaks down on curved or hilly terrain. Additionally, while calibration on reference points demonstrated high fidelity, errors in far-field regions (distant background) can exceed 5–10%, leading to inaccurate distance estimates for vehicles far from the camera. This is partially mitigated by world-space distance thresholding (far vehicles are automatically excluded), but a vehicle at 30 meters might be misclassified as 25 meters.

For future deployment across AT's network—which includes hilly terrain in suburban areas—we should consider multi-region homography (dividing the image into piecewise planar regions) or full camera calibration using Tsai's method to recover intrinsic and extrinsic parameters. These approaches are more computationally expensive but provide sub-pixel accuracy.

### 3) Weather and Lighting Robustness

The test video was recorded under clear daylight conditions. CCTV footage under rain, night, or glare presents significantly different challenges: reduced contrast, motion blur from rain streaks, and color shifts under sodium-vapor lighting. YOLO's accuracy may degrade substantially, and homography calibration may become unreliable if reference landmarks are obscured or poorly lit.

We have not yet evaluated the system under these conditions. This is a critical gap because Auckland's weather is variable and many intersections lack sufficient lighting. Future validation must include diverse weather and temporal conditions.

### 4) Ground-Truth Validation and Operational Metrics

The system currently lacks quantitative validation metrics (precision, recall, F1 score for risk classification) computed against manually-labeled ground truth. While we have identified proximity events from the test video, we have not conducted expert annotation or validation against independently-judged collision risk labels. This represents a significant limitation when claiming the system is "feasible" for AT's operational deployment.

Ground-truth validation requires domain expertise from traffic safety engineers—defining what constitutes a genuine collision risk vs. a false alarm can be ambiguous in edge cases (e.g., slow-speed intentional close passes vs. near-misses). Before production deployment, we must establish inter-annotator agreement protocols with AT's safety and operations teams and validate system outputs on a representative sample of multiple intersections and traffic conditions.

## B. Future Development Roadmap

### 1) Phase 2: Expanded Validation (Weeks 11–16)

Immediate priorities:
- Conduct expert review and annotation of 50–100 proximity events from multiple videos
- Compute Precision, Recall, F1, and Confusion Matrix against expert labels
- Test on 3–5 additional intersections to assess generalization robustness
- Evaluate system performance under diverse weather (rain, overcast, night) and lighting conditions
- Establish sensitivity analysis: how do detection accuracy, distance thresholds, and confidence levels affect risk classification?

**Deliverable**: Validation report with metrics broken down by intersection, weather condition, and traffic density.

### 2) Phase 3: Advanced Kinematics and Tracking Optimization (Weeks 17–24)

- Validate TTC and PET computation against annotated ground-truth near-miss events
- Implement acceleration estimation for kinematic analysis (currently assumes constant velocity)
- Optimize rear-end and sideswipe collision detection thresholds using empirical multi-intersection data
- Explore enhanced tracking robustness through motion-based filtering or simple appearance consistency checks
- Develop confidence intervals for TTC/PET to quantify uncertainty in safety-critical estimates

**Deliverable**: Optimized system with documented accuracy improvements on multi-intersection validation data.

### 3) Phase 4: Integration with Theia Platform (Weeks 25–32)

- Adapt the pipeline to ingest video streams from Theia's CCTV infrastructure
- Integrate with CVT's Triton inference service for scalable GPU-accelerated deployment
- Develop web dashboard for visualization and filtering of proximity events
- Establish feedback loops with AT operations teams to validate thresholds and refine alert policies
- Implement real-time alert mechanisms for safety planners (e.g., email/SMS for high-risk events)

**Deliverable**: Pilot deployment on 1–2 intersections with operator feedback collection and system refinement based on operational use.

### 4) Phase 5: Network-Wide Deployment (Weeks 33+)

- Deploy across AT's 100+ instrumented intersections
- Build multi-region homography models calibrated for AT's diverse intersection geometries and terrain
- Develop adaptive thresholding based on empirical collision risk profiles per intersection
- Integrate with AT's injury collision database to correlate near-miss detection with actual crashes
- Establish continuous retraining pipeline to maintain detector accuracy as lighting and infrastructure change

**Deliverable**: Production system monitoring AT's network with periodic safety impact assessment reports.

## C. Broader Implications and Lessons Learned

This internship has demonstrated that computer vision can provide actionable safety insights from existing CCTV infrastructure. However, the path from prototype to production involves not just technical refinement but also stakeholder engagement, validation rigor, and organizational integration.

**Key insight on interpretability**: The homography transformation proved critical for stakeholder buy-in and understanding. When results were presented in "150 pixels," planners were skeptical and questioned the relevance. When we showed "1.5 meters—the width of a car"—a world-grounded metric—the system's outputs became immediately interpretable and credible. This insight carries profound implications for future CV work at AT: interpretability and world-grounded metrics should be design principles from the outset, not afterthoughts. Pixels are implementation details; meters and seconds are operational language.

**On pragmatic engineering within research constraints**: Working within a 10-week internship timeline required deliberate prioritization. Rather than pursuing perfect solutions (DeepSORT for all occlusion scenarios, full camera calibration, multi-modal fusion), we implemented core functionality that could be demonstrated and validated quickly. This pragmatism—choosing "good enough for prototype" over "perfect for research"—enabled forward progress and created a foundation for iterative refinement in future phases. The distinction between research-grade solutions and production-ready systems proved valuable.

**On agile methodology in research**: Working within AT's Scrum-of-Scrums framework exposed the tension between research iteration and delivery commitments. Unlike traditional software sprints, CV research benefits from "exploration sprints" where the goal is learning (whether something works), not shipping features. Negotiating this with product and operations teams—establishing clear acceptance criteria and knowing when validation is "sufficient" rather than "perfect"—proved crucial for professional communication and stakeholder management.

**On team dynamics and psychological safety**: The AT Computer Vision team's culture of psychological safety and rapid, non-defensive feedback accelerated learning dramatically. The ability to ask "why would this fail?" or "what's wrong with this approach?" in daily stand-ups without defensive reactions enabled hypothesis-driven debugging rather than random parameter tuning. Being part of a team that celebrated learning from failures—not just successes—is a replicable competitive advantage for AT's future CV initiatives and a powerful professional lesson.

