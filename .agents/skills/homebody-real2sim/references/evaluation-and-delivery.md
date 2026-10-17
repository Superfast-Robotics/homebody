# Evaluation and delivery

Use this when choosing a candidate, validating a substantial edit or handing off
a reconstructed scene. Choose tolerances for the intended use; scene
results and unit-test counts are not accuracy thresholds.

## Keep claims separate

| Claim | Evidence to inspect |
| --- | --- |
| Capture integrity | Bound files, hashes, decoding, gaps, epochs and source lineage |
| Camera/frame alignment | Projection fixtures and agreement across moving segments, depths and regions |
| Geometry and coverage | Dimensions, support contacts, openings, free-space contradictions and observed scan-to-surface errors |
| Appearance | Registered fixed-camera native renders, source crops and full-frame failures |
| Navigation/localization | Exact delivered map/collision/target representation, spawn and representative routes |
| Requested interaction | Joint axes/limits, collision, contact and dynamics tests; clearly labeled assumptions |
| Package correctness | Native loading and relative dependencies after copying or extracting the delivery |

Report denominators and unsupported regions, not only averages. For meaningful
geometry comparisons include typical and worst-region errors, coverage within
the chosen tolerance, and balanced surface/region results. Missing or failed
items remain in the accounting. Hidden surfaces cannot be certified by a lack
of LiDAR returns.

Separate construction, development and evaluation views or route blocks. Nearby
video frames are correlated. If fitting and scoring share SLAM inputs, report
internal consistency; do not call it independent absolute accuracy. Once a
held-out result informs changes, it becomes development evidence.

Keep source identities, camera fits, masks, geometry version and scoring fixed
when comparing candidates. Preserve the baseline and failures. Do not improve
a score by deleting a difficult surface or silently changing its evaluated area.

## Validate the actual HomeBody package

The [scene package contract](../../../../guides/real2sim.md#homebody-scene-package)
is the maintained layout and command reference. Its executable counterparts are
the [scene loader](../../../../src/homebody/backends/scene.py),
[scene-builder fixtures](../../../../tests/test_build_scene.py) and
[map-builder fixtures](../../../../tests/test_build_scene_map.py).

The already shipped [SRC kitchen](../../../../assets/real2sim/src_kitchen/)
is a worked runtime example. Compare its
[entity bindings](../../../../assets/real2sim/src_kitchen/semantics.json),
[task configuration](../../../../assets/real2sim/src_kitchen/task_scene.json) and
[physics report](../../../../assets/real2sim/src_kitchen/physics.json) when authoring
a new package. Its [source note](../../../../assets/real2sim/src_kitchen/source.json)
does not include a full raw-capture/calibration audit. Do not reuse its geometry,
positions or assumptions as measurements of a different room. Check asset rights
against [third-party notices](../../../../THIRD_PARTY_NOTICES.md) and the actual
new source material before distribution.

- `homebody check --scene …` is a prerequisite/file check. Compile `scene.xml`
  with MuJoCo as well, resolve its assets and check the assembled robot spawn.
- Inspect floor support and colliders around doors, table supports and concave
  objects. The [map builder](../../../../tools/build_scene_map.py) uses finite
  floor and initial collision geometry; the planner applies footprint inflation.
  Check both a genuinely traversable doorway and a genuinely blocked gap using
  the actual planner. Never expand free cells simply to make a test pass.
- Recheck affected maps/colliders after geometry edits. An initial static map
  does not track future movement of free objects automatically.
- Render through the fitted phone camera for source comparisons. Keep the
  runtime's default ego view separate from a calibrated source projection.
- Mechanism and manipulation claims require their own native tests. Default
  masses/friction are assumptions; a static capture does not measure dynamics.

If localization is requested, test the named delivered target. A separate point
map does not establish localization against the scene mesh. If USD is requested,
preserve the same frame and verify its composed stage, materials and any physics
independently; conversion success alone does not establish MuJoCo/USD parity.

## Deliver a compact, reproducible scene

Use one named directory with relative runtime dependencies, source/calibration
identities, object/task metadata, licenses, launch commands, measured results and
limitations. Keep the raw bags, video and large intermediate fits in the capture
directory, outside the scene package, and reference them from the package notes.

Before delivering, reopen the package from a clean location and check its checksums
and dependency closure. Remove duplicate and unused artifacts before reducing geometry
or texture quality. Check third-party asset rights, and keep unresolved calibration or
geometry claims explicit even when every package check passes.
