# Harris / SIFT controlled repeatability experiment

Four real 960x540 frames: EXO and EGO F50/F200 from paired clip04. Rotations, scale changes and Gaussian blur are **controlled synthetic image changes**, not new viewpoints or panorama results.

Fixed detector settings are recorded in metrics.json. Harris uses blockSize2, ksize3, k0.04, 1% of maximum positive response and 3x3 local NMS. SIFT uses nfeatures500, nOctaveLayers3, contrastThreshold0.04, edgeThreshold10 and sigma1.6. Centers rounded to 0.001px are deduplicated before response-ranked 5px minimum spacing and the 500-location budget. SIFT orientation copies are not independent locations. Harris raw_count is the number of thresholded pixels before local NMS; SIFT raw_count is its detector keypoint count. These raw counts have different definitions and are not a detector ranking.

Each controlled warp has a saved source-to-target 3x3 H. A warped valid mask is eroded by a 33x33 square, removing 16px around image/warp boundaries. Source and target centers are filtered by forward/inverse H in the common visible region. Positions are matched using maximum-cardinality one-to-one bipartite matching at 3 **target-image pixels**. The denominator is min(common source count, common target count); zero produces null. Four image scores are macro-averaged with null cases reported separately. This does not use SIFT descriptors or a fitted transform.

metrics.csv has 48 image-condition-method rows; coordinates/ records retained position coordinates, H, common-visible indices and one-to-one correspondences. audit.json verifies all 8 identity controls equal1, homography round trips, visibility, spacing/uniqueness, maximum matching with minimum vertex-cover certificates and independent CSV macro recalculation. Timing is one descriptive CPU measurement including selection, not a repeated benchmark.

Figures: overview_exo.png / overview_ego.png show original, log-positive Harris response, Harris centers and SIFT keypoints; controlled_changes_exo.png / controlled_changes_ego.png show all five changes of F50; repeatability_examples.png shows the numerically highest and lowest nonidentity cases without changing parameters; repeatability_summary.png shows all condition means. A 16px center margin does not guarantee that large SIFT support patches avoid all invalid pixels. Four selected frames do not establish generalization.

Numeric evaluation uses the in-memory uint8 transformed images, before any JPEG writing. transformed_images/*.jpg are lossy display assets only and must not be reloaded to reproduce the saved metrics. Plot fonts try Monaco, Menlo, DejaVu Sans Mono and Consolas before Pillow default; font differences do not affect detection or matching.

Reproduce from the project root with Python3.12, OpenCV4.11, NumPy and Pillow; use an empty output directory:

```sh
python scripts/course_feature_detection.py \
  --frames-root output/week08/frames/scene04 \
  --output output/week08_harris_sift/results_reproduced
```
