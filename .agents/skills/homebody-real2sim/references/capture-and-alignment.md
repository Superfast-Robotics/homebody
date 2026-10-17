# Capture and alignment

Use this for a new capture audit, camera fitting or disagreement between LiDAR
and images. The maintained collection contract is the repository's
[schema](../../../../configs/capture/capture_manifest.schema.json) and
[template](../../../../configs/capture/capture_manifest.template.json).
Use the [capture audit commands](../../../../guides/real2sim.md#audit-a-closed-run-first)
on a working copy. Do not create a second manifest schema or rewrite originals.

For implementation details, inspect the current
[manifest writer](../../../../src/homebody/real2sim/exploration/record/capture_contract.py),
[bag checks](../../../../src/homebody/real2sim/exploration/record/postcheck.py) and
[review-video exporter](../../../../src/homebody/real2sim/exploration/record/export_mp4.py).
The [phone setup guide](../../../../guides/phone-camera.md) describes the
supported streaming paths; it is not a measured camera calibration.

## Assign evidence a defensible role

| Input | Useful evidence | Important limit |
| --- | --- | --- |
| Raw LiDAR and IMU | Metric support, motion constraints and reprocessing | Missing returns are not free space; self returns, glass and motion can mislead |
| Registered scans and SLAM poses | Room layout, common frame and alignment initialization | A scene scored against the same SLAM inputs has internal consistency, not independent accuracy |
| Native phone video | Inventory, silhouettes, appearance and camera-motion estimation | Sharp RGB alone does not provide metric scale or a calibrated mount |
| Selected gallery frames and route | Observation history and useful search locations | Receipt-time robot positions and intended knots are not exposure poses |
| Independent dimensions or repeat observations | Checks on scale, placement and generalization | A few anchors do not certify every object or surface |

Read the actual stream/frame metadata, not just filenames or device labels.
Inventory bound files, sizes/hashes, duration, gaps, epochs, units and encoding.
Inspect native moving views as well as stationary gallery images. Keep preview
video, fused PLY and live occupancy products identified as derivatives.

## Establish one metric frame

Select a SLAM epoch; register separate epochs only when correspondences support
the transform. Record direction explicitly as `T_target_from_source`, alongside
units, handedness, axes, quaternion order and any estimated scale. Apply a known
point and ray through the transform chain before fitting a scene.

Keep exploration-start coordinates, SLAM-world poses and camera optical axes
distinct. Do not independently recenter the geometry, map, camera or robot spawn.
The HomeBody scene contract uses matching named frames in `semantics.json` and
`task_scene.json`; see the [package contract](../../../../guides/real2sim.md#homebody-scene-package).

## Use motion to fit time and projection

Decode video timestamps as PTS times the stream time base, retaining segment
identity and any timestamp rebasing. Host receipt times and the approximately
three-second online RGB delay are clues, not a measured exposure clock. Do not
apply one assumed delay to every native segment.

Match turns, starts/stops or visual motion against SLAM/IMU to estimate a clock
offset; add clock drift only when evidence supports it. Verify early, middle and
late moving segments and more than one direction. Stationary alignment barely
constrains timing. Preserve source-video hashes and exact frame timestamps for
each camera fit.

Validate intrinsics, distortion, crop/rotation and stabilization at the decoded
resolution. Distinguish a per-view camera fit for appearance from a calibrated
rigid mount suitable for sensor simulation. Rolling shutter, changing crop,
timing, interpolation and extrinsics are alternative explanations for motion
misalignment; test them separately before moving scene objects to compensate.

Check scale and alignment on spatially separated surfaces at different depths.
Nearly planar motion can leave translation or mount height weakly constrained.
Low reprojection error alone does not resolve that ambiguity. Test on another
segment or independently measured feature; if it informs fitting, label it as
development evidence rather than untouched evaluation.

## Diagnose before filtering or filling

Separate clock/projection errors, pose drift, real occlusion, dynamics and weak
surface support. Keep rejected points/frames with reasons. Dense floors should
not overwhelm sparse walls or door edges in a fit; inspect errors by region and
surface family as well as pooled residuals.

Do not erase the traveled corridor to remove robot returns or fill an apparent
hole before checking masking and registration. If a source video was
cropped, retimed or intentionally masked, verify timestamp and image
correspondence before reusing old camera fits. Ordinary blur is weak evidence;
only an explicit exclusion changes the requested scene content.

When an important quantity is unobservable, keep it unknown or request a targeted
measurement. Reconstruction work does not itself authorize another robot run.
