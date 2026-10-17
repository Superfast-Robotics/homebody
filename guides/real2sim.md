# Real-to-sim: capture to HomeBody

A coding agent reconstructs a closed G1 exploration run (LiDAR and iPhone) into a **HomeBody** scene package (the
MuJoCo `scene.xml` with its meshes, the semantic `.glb`, the navigation map and the semantics): it audits the
capture, fits the geometry, binds the semantics, checks the package with the builders and delivers it; a `.usd`
export for Isaac Sim is optional. The workflow is the [homebody-real2sim skill](../.agents/skills/homebody-real2sim/SKILL.md)
(`$homebody-real2sim`) and its [capture and alignment](../.agents/skills/homebody-real2sim/references/capture-and-alignment.md),
[geometry, RGB and semantics](../.agents/skills/homebody-real2sim/references/geometry-rgb-and-semantics.md),
[evaluation and delivery](../.agents/skills/homebody-real2sim/references/evaluation-and-delivery.md).

## Record, export or reconstruct

```bash
homebody real2sim explore --enable-ask-operator --iphone-input "$PHONE_STREAM_URL" --budget-s 1200 \
  --task "Explore the accessible parts of this room. Capture useful views of the room layout, work surfaces and important equipment."
homebody real2sim export --exploration-data /path/to/closed-run --output /path/to/new-sensor-package
homebody real2sim reconstruct --exploration-data /path/to/closed-run --output /path/to/new-scene \
  --work /path/to/work --task "Reconstruct the observed room and its work surfaces"
```

- Recording is on by default ([Where the run is](exploration.md#where-the-run-is)); **STOP** or Ctrl-C finalizes it
  and leaves the body host balancing, while Ctrl-\\ is the separate emergency halt ([Stopping](exploration.md#stopping)).
- Ask for the actual areas and features; collect mechanism motion or measurements only when the task needs them.
- `reconstruct` runs one coding-agent command (`--agent-command`, the Codex CLI by default) on a closed run or an
  `export` package and builds the scene only after the agent's audit reports it ready; request, agent output,
  intermediate geometry and audit stay in `--work`, and only the runtime scene goes to `--output`.
- `export` (needs the `mcap==1.5.0` exploration dependency) copies sensor channels into new MCAP files and native
  phone media unchanged, verified before the package appears. It requires `sensor_integrity=accepted`, not just exit
  status zero or `collection_state=finalized`, and a hash-verified `session/finalization.json` with status `closed`
  and no recording failure.

## Audit a closed run first

Keep the complete original (`capture_manifest.json`, native bag chunks, phone media, frame/clock metadata,
gallery/model evidence, source identity); [configs/capture](../configs/capture/) is the contract (its schema checks
structure only). The recorder's closed-run utilities **write** reports and products and `check` can reindex an
interrupted bag, so run them and any repair on a separate working copy; they use the recorder image from the [SuperOdometry build](../hardware/g1/superodom/README.md) and do not operate the robot:

```bash
# CLOSED_RUN is the original; WORK_RUN is a new, non-existing directory; a reflink saves space where supported.
cp -a --reflink=auto -- "$CLOSED_RUN" "$WORK_RUN"
src/homebody/real2sim/exploration/record/record.sh check "$WORK_RUN"
src/homebody/real2sim/exploration/record/record.sh export "$WORK_RUN"
```

- `record.sh capture "$WORK_RUN"` regenerates bookkeeping after derived-output changes without re-verifying old
  checksums: keep the original manifest and record the differences. Check hashes, decoding, stream gaps, epochs,
  frame directions, video timestamps and full-route native phone images (a review video can repeat frames; a fused
  PLY lacks raw scan timing, rays and poses).
- LiDAR/SLAM is metric support, not an accuracy survey: keep SLAM epochs, their transforms and the exploration-start
  frame (a navigation frame, not a calibration) distinct, and fuse epochs only with a registration. DONE judges the
  exploration task, not coverage, metric accuracy or manipulation.
- The iPhone feed is uncalibrated and about three seconds late; gallery poses are receipt-time robot poses, not
  exposure poses, and the mount is not a measured extrinsic: check intrinsics, distortion, crop/stabilization,
  optical-frame extrinsics and clock alignment (video PTS, receipt and LiDAR time) or leave them explicitly uncertain.

## Reconstruct

The task names the area, key objects and exclusions. The agent leaves the capture manifest unchanged, notes
choices, calibration and source bindings with the scene, and builds metric layout from LiDAR/SLAM, then objects and
semantics fitted to measured supports and several views (modeled, unresolved or excluded), then registered native RGB
without people, changing displays or robot reflections, then requested physics with recorded mass and friction
assumptions. It proves source → geometry → MuJoCo render → map and collision agreement on one area first (fitting and
scoring on the same SLAM data shows internal consistency only), then reports each requested capability with
denominators, tolerances, failures and unknown areas, plus the dependency closure, source versions, reproducible
commands and limitations, after checking that a copied package resolves its assets and loads; an offline package
check is not a claim of robot readiness. Example prompt:

> Use $homebody-real2sim on the closed capture at `/path/to/capture`. Preserve the originals. Audit integrity, clocks,
> calibration and coverage, then build a static HomeBody/MuJoCo scene in `/path/to/my_room`, prioritizing layout,
> openings and work surfaces. Label inferred hidden geometry, check one area end to end first, and do not operate
> hardware or start paid remote jobs.

## HomeBody scene package

A new scene uses the layout of the released [example](../assets/real2sim/src_kitchen/):

```text
my_room/
  <name>.glb        # exactly one semantic GLB (the example's is kitchen_semantic.glb), in the declared metric frame
  semantics.json    # coordinate_frame and entities with GLB node bindings
  task_scene.json   # scene name, robot_start, movable entities, collision declarations
  scene.xml         # generated MuJoCo scene
  meshes/           # generated visual/collision meshes and textures
  physics.json      # generated physics/assumptions report
  map.npz           # generated navigation map in that same frame
```

- `semantics.json` names the coordinate frame (metres, right-handed Z-up) and each entity's stable ID and
  `glb_node_indices`; movable entities need `bounds_ws_map_m` as `[[xmin, ymin, zmin], [xmax, ymax, zmax]]`, which
  scoring also reads for destination entities.
- `task_scene.json` has `robot_start: [x, y, yaw]` (metres, radians), `frame` equal to the semantic frame name, and
  `movable_entities` (an empty list is valid); optional `collision_boxes` and `object_contact` give explicit
  collision/contact choices. Name collision boxes `<entity id>__<part>` so their contacts count for that entity.
- The finite reconstructed ground is entity `floor`, bound to exactly one GLB node with an exactly planar mesh: the
  scene's only ground plane. The loader casts down from z = 2 m at the spawn and must hit that floor first.
- The [builder](../tools/build_kitchen.py), [loader](../src/homebody/backends/scene.py) and
  [small-room fixture](../tests/test_build_scene.py) are the executable contract; do not copy the kitchen's positions
  or dynamics into another room.
- The builder needs an **already reconstructed** semantic GLB whose meshes load in Trimesh with a PBR material
  (`TextureVisuals` with `PBRMaterial`); textured and single-triangle meshes need UVs, and unknown appearance gets a
  neutral PBR base color noted as a placeholder. Edit the derivative scene GLB, never the source capture. The builder
  keeps the GLB's frame and does not fit LiDAR, calibrate the iPhone, infer identity or create joints; its default
  masses and contacts are assumptions (inspect coarse convex colliders around openings, supports and concave objects).

```bash
SCENE_DIR="$PWD/assets/real2sim/my_room"
python tools/build_kitchen.py --scene "$SCENE_DIR"
# Example geometry choices: choose the band for the intended robot/task.
python tools/build_scene_map.py --scene "$SCENE_DIR" --resolution 0.05 --height-band 0.05 1.6
homebody check --scene "$SCENE_DIR"
homebody run --scene "$SCENE_DIR"   # after compiling scene.xml in MuJoCo and checking the G1 spawn
```

- The [map builder](../tools/build_scene_map.py) projects collisions through the height band (metres above the floor)
  at the initial object configuration, without robot-radius inflation; other shapes, nonzero margins, stairs and
  several floor levels are refused or unsupported, unknown floor stays occupied, and an existing map needs `--replace`.
- `homebody check` checks prerequisites and files, **not** the scene, contacts, camera fidelity, paths or task success;
  `homebody run` starts simulation, not the robot ([SIM guide](../README.md#-sim)). Compare with the fitted iPhone
  camera, not the default ego camera; an optional USD keeps scale and frame and needs its own load and physics checks.
