---
name: homebody-real2sim
description: Reconstruct a measured HomeBody/MuJoCo scene with geometry, semantics, appearance and requested physics. Use for capture handoff, reconstruction, alignment or scene validation; not to operate the robot or deploy a scene to hardware.
---

# HomeBody real-to-sim

Turn a closed exploration capture into a HomeBody scene, autonomously: audit the capture,
estimate the calibration, fit the geometry, bind the semantics, validate with the builders and
deliver the package: the MuJoCo `scene.xml` with its meshes, the semantic `.glb`, the navigation map
and the semantics, with a `.usd` export for Isaac Sim as an option. Use any tool available to you for
the work, including MuJoCo, Isaac Lab and Blender; the deliverable is still the HomeBody scene
package described below, and it must align
with the captured room both visually (appearance, materials, views) and geometrically
(metric scale, floor, walls, openings, object placement) before it is delivered. For objects and
furniture, use assets from open-source online databases when they match what was observed, or make
your own; either way they must align with the capture and keep their licence notice. Use the
diversity of the collected data: the LiDAR, the odometry and the camera views taken across the
exploration from different positions and headings; surfaces seen from several viewpoints are the
ones you can place and texture with confidence.

Resolve the repository root three levels above this skill. Read the
[workflow guide](../../../guides/real2sim.md) for commands and the scene
format, then inspect the actual capture and current scene interfaces. For new
collection, use the [exploration guide](../../../guides/exploration.md).

## Supporting references

Read the reference for the current bottleneck; these supplement the workflow
without requiring every reconstruction to repeat every check.

| When needed | Reference |
| --- | --- |
| Audit a capture, fit cameras, or diagnose source disagreement | [Capture and alignment](references/capture-and-alignment.md) |
| Fit objects, register appearance, or correct exclusions | [Geometry, RGB and semantics](references/geometry-rgb-and-semantics.md) |
| Compare candidates or package a scene for delivery | [Evaluation and delivery](references/evaluation-and-delivery.md) |

## Select the next useful deliverable

Honor the requested area, important objects, exclusions, simulator, tolerances
and allowed compute. For a new scene, audit the capture and build a small
representative region end to end before scaling up. For a resumed reconstruction
or a narrow material/object edit, retain the working scene and existing verified
alignment. Ask only for decisions that materially affect the deliverable; use
explicit assumptions where a bounded result can proceed.

Use the scene's existing notes for scope, source identities, unresolved gaps and
acceptance evidence. Separate capture integrity, metric geometry, visual
registration, localization and interaction claims. The model's DONE and a valid
manifest establish neither complete reconstruction nor calibrated accuracy.

## Preserve and interpret the capture

- Keep the original closed run immutable. Work in a separate output directory;
  make a copy before running recorder `check`, `export` or `capture`, which write
  reports, derived products or metadata. Retain the original manifest and file
  hashes rather than regenerating them to conceal a discrepancy.
- Use [configs/capture](../../../configs/capture/) as the sole maintained schema
  and template source. Map native streams and file bindings into that contract;
  do not duplicate its schema in this skill or create a new capture format.
  A capture-only manifest describes collection provenance, not permission to
  rewrite it when reconstruction is requested later.
- Check that bound files, bag splits and stream selectors exist and decode;
  verify sizes/hashes, time spans, gaps, SLAM epochs and frame conventions.
  Inspect the full route's native RGB as well as selected gallery images.
  Preview videos and fused point clouds are derived products, not replacements
  for native video, raw LiDAR/IMU and registered scans/poses.
- Select one SLAM epoch or establish an evidenced registration between epochs.
  Preserve metric scale, units, handedness and the explicit transform direction
  `T_target_from_source`. Exploration-start coordinates, gallery poses and
  SLAM-world coordinates are different representations; do not independently
  recenter them or relabel a saved live occupancy map as a scene map.
- Keep decoded video PTS/time base, device timestamps and host receipt times
  distinct. Walking-gallery positions are approximate receipt-time associations,
  not exposure poses. Estimate and validate camera intrinsics, distortion,
  crop/stabilization, timing and optical-frame extrinsics across early, middle
  and late motion before using RGB as metric evidence. A stated lens zoom or
  mount location is not calibration. Missing inputs remain explicitly unknown.

## Build the scene in evidence order

1. **Layout and metric support.** Fit the floor, walls, openings and main supports
   from LiDAR/SLAM. Inspect free-space contradictions, weakly observed surfaces
   and dynamic/self returns. Preserve rejected data and explain filters. Use
   measured support and several views to place major objects in the same frame.
2. **Meaningful objects.** Reconcile the RGB inventory with the scene. Prioritize
   gross shape, accessible fronts/work surfaces, handles, controls, openings and
   task-relevant movable objects. Do not spend repeated passes inventing unseen
   undersides or featureless backs. Record inferred closures and unresolved
   features; important missing geometry must not disappear from the inventory.
3. **Appearance.** Project or fit RGB only after checking registration and
   occlusion. Use multiple views, preserve fine visible features and keep people,
   changing displays and robot reflections out of static textures. Ordinary blur
   is weak evidence. Asset-library shapes
   are priors: verify dimensions, support contacts, axes and license.
4. **Requested physics.** Keep visual meshes, colliders and the navigation map
   distinct but in one frame. Add joints, masses and friction only to the level
   needed by the task, labeling assumptions. A static capture cannot establish
   a drawer's travel or a graspable object's dynamics. The supplied scene
   builder creates static parts and selected free bodies; articulation requires
   additional implementation and validation.

For each change, evaluate the next failing layer rather than replacing the whole
scene. Geometry edits require affected map/collision checks; material-only edits
should leave geometry unchanged. Keep unobserved regions unknown or labeled as
priors. Seek another measurement only when it resolves a task-relevant limitation.

## Integrate and validate the exact package

Follow the [scene package contract](../../../guides/real2sim.md#homebody-scene-package).
[build_kitchen.py](../../../tools/build_kitchen.py) consumes an **already
reconstructed semantic GLB**; it does not reconstruct bags or calibrate cameras.
Use [build_scene_map.py](../../../tools/build_scene_map.py) for a map derived from
the accepted scene collision geometry and finite floor. Select the resolution and
robot-relevant vertical band explicitly. Do not copy an unrelated `map.npz`,
expand its free area to satisfy a route, or transform one asset alone.

Run `homebody check --scene …` for prerequisite/file checks, then compile the
actual MuJoCo scene and check its relative dependencies, robot spawn, floor,
colliders and map agreement. That command alone does not validate geometry,
physics, camera fidelity or successful task execution.

Compare representative fixed-camera MuJoCo renders against registered source
views, including failures. Report meaningful per-region geometry errors and
coverage using declared tolerances. Hold out independent views/routes when
available; scores against scans used to create SLAM are internal consistency,
not independent accuracy. Evaluate localization against its named target, and
exercise joints/contacts only when those capabilities are claimed.
The default simulated ego camera is not the recorded iPhone's calibrated
projection; use the fitted phone camera for source-image comparisons.

Deliver one named scene directory, runtime dependencies, source/calibration
identities, commands, measured checks and limitations. Keep bulky raw captures
and intermediate fits separate with resolvable provenance. Reopen a copied or
extracted package before delivery. An optional USD must preserve the same frame
and have its own native-load/material/physics checks; export success alone does
not establish parity.
