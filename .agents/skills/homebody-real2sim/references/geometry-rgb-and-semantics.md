# Geometry, RGB and semantics

Use this after establishing the [capture and alignment](capture-and-alignment.md)
contract, or when correcting a particular object or material in a working scene.
Keep the existing verified frame and unaffected geometry.

## Inventory before polishing

Review sampled views across the full native recording, then inspect clear source
frames at entrances, turns and occlusion changes. A bounded model gallery is not
the full capture inventory. Record each important in-scope object's stable ID,
source segment/PTS, observed surfaces, support, estimated dimensions and uncertainty.
Distinguish modeled, unresolved and explicitly excluded content.

Prioritize layout, gross shape, accessible work surfaces and task-relevant
features such as openings, handles and controls. Hidden backs and undersides may
remain documented priors when the task does not depend on them. A recognizable
brand or shape does not prove contents, mechanism, reachability or dynamics.

## Fit measured supports and objects

Anchor floors, walls, counters and large fixtures to LiDAR support before fitting
small items. For weakly sampled objects, combine registered translated views
with the actual local support plane. Nearby points may belong to the background
or counter; an image bounding box is not a segmented metric object.

Check contact and connectivity: handles meet mounts, seats meet frames, objects
rest on the local surface, and openings remain open. Check library assets for
scale, axes, variant, support and license; their hidden geometry is a prior.
Use a custom primitive or mesh when that matches the evidence more directly.

The existing [scene builder](../../../../tools/build_kitchen.py) consumes an
already reconstructed semantic GLB. Its material and node-binding requirements
are in the [package contract](../../../../guides/real2sim.md#homebody-scene-package).
It does not infer room geometry or create articulated mechanisms automatically.

## Project appearance onto accepted geometry

1. Bind each source to its video/frame identity, intrinsics and
   `T_map_from_camera(t)`. Register to the common frame rather than independently
   fitting every object to a chosen photograph.
2. Check positive depth, image bounds, resolution, incidence and occlusion before
   transferring color. Test visibility logic on both a sloped surface and a real
   foreground occluder. Masked people still occluded the background at capture.
3. Prefer clear, sufficiently resolved views with credible alignment, including
   useful walking views. Avoid baking people, robot reflections or changing
   displays into static texture. Preserve visible fine features and seams.
4. Track observed texture, reused patches, neutral material priors, unresolved
   appearance and intentional exclusions separately. Do not invent hidden text
   or count blur as reconstructed detail.
5. Compare fixed-camera native MuJoCo renders against several source views,
   including whole frames and failures. Material-only edits should preserve
   geometry, transforms, colliders and map bytes; beveling is a geometry change.

HomeBody's default simulated ego camera is not the fitted iPhone camera. Use the
source camera model for these comparisons; a free camera adjusted for every
render can conceal a geometry defect.

## Keep semantics and exclusions consistent

Bind entity IDs to the final GLB nodes and use the same frame in task metadata.
Object bounds are references, not grasp contacts, opening centers or free walking
goals. Derive such features separately when the task needs them.

When the request excludes an object, update its visuals, collisions, semantic and task
entries and dependent source references together in the derivative scene, and leave
the original capture unchanged.

Maintain visual meshes, collision geometry and the navigation map as distinct
representations in one frame. Convex collision proxies can close a real concavity
or doorway; inspect them and run the actual planner checks before delivery.
Use [evaluation and delivery](evaluation-and-delivery.md) for the final package.
