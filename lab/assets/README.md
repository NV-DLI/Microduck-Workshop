# Shared teaching scene

All dimensions are metres. `scene_style.py` applies the same official robot asset,
materials and daylight setup to the walking, terrain, keyboard and indoor football
labs. RTX camera capture is used for the new scene media. External community videos
retain their original rendering and attribution.

## Robot

`microduck_allcollisions_backlash.upstream.usd` is the full-contact conversion of
Pollen Robotics' official Microduck backlash model. Its SHA-256 is
`c1acc45caabc9c6b5845574c79e2c9cd3ae94cdcf625aeacce2e8bddd79cf516`.
`microduck_contact.usda` overlays course white/orange materials. It changes no
mass, inertia, joint limits or actuator calibration. The full-contact asset adds
head, hip, shin and battery collision geometry to the two foot colliders used in
the old walking asset, preserving the official shin/ankle collision exclusions.

Source: [Pollen Robotics microduck_rl](https://github.com/pollen-robotics/microduck_rl),
CAD baseline `2b581c641406a48346e696212930ea881c222c52`.
Comparison with `b02fdea4902c2b93b7d34a83a6f05da4ececb7d3` on 2026-10-23 found
no changes to the robot STL geometry or base XML parameters. This is a versioned
simulation configuration, not a claim that every firmware component is latest.
Robot geometry retains CC BY-NC-SA 4.0; robot XML/source retains Apache-2.0.
See `THIRD_PARTY_NOTICES.md` and upstream `LICENSE-MODELS.md`.

## Props

`artwork/build_scene_assets.py` creates the original parametric goal and four
original mat drawings. Product reference:
[mat](https://e.tb.cn/h.jVgqwsimco7KxBM),
[miniature steel goal](https://e.tb.cn/h.jVgIdgPCtwTum1F).
Dimensions are based on supplied product screenshots, not physical measurements.
No retailer photo is redistributed. No community mesh was reused.

- Mat: 1.6 × 1.0 m, centre at (0.1, 0). Four printed styles A–D.
- Goal: outer width 0.40, depth 0.20, height 0.25; front plane x=0.68.
- Assumed frame diameter 0.006; front post centres y=±0.197.
- Clear opening width 0.388, height 0.244.
- Assumed wire diameter 0.0008 and spacing 0.012. Frame and wires have static
  collision geometry. The goal represents a secured installation.
- Mat print is 0.00005 above the rigid Newton contact plane. Its 0.004 visual
  backing extends below that plane. Raised edges and compression are not modeled.
- Ball remains diameter 0.07, mass 0.045 kg. A different real ball requires both
  physical and visual-scale calibration.

Mat thickness, friction, compression, frame/wire diameters and goal mass have not
been physically measured. Rendering quality does not establish a calibrated
physical twin. The new scene has its own perception, map and evaluation contract;
historical task success rates must not be transferred to it.
