---
myst:
  html_meta:
    description: "The onboard policy finds the ball and goal with Microduck's own sensors: the RGB camera recognizes targets, ToF gives the floor and nearby distances, and unreliable information is rejected."
---

# 4.1 Find the ball and goal: RGB detection and ToF ranging

:::{lesson-intro}
:meta: Section 4.1 · Estimated learning time: 20 minutes

The onboard policy finds the ball and goal using only Microduck's own sensors.
:::

{#sensors}
## What the robot sees

:::{course-video} media/sensors-onboard-recorded
:alt: One simulated shot with the onboard policy: the third-person view on the left, the head RGB image with detector boxes at the top right, the 8×8 ToF distances at the bottom right and the task stages over time at the bottom.

One simulated shot with the onboard policy: the third-person view, the head RGB image with detector boxes and the 8×8 ToF distances are synchronized to the same capture time; NR marks a zone without a valid return. This recorded attempt scores and returns to standing.
:::

| Sensor | Each capture gives | Answers |
|---|---|---|
| RGB camera | A 960 × 540 color image, 15 per second | In which direction the ball and goal are |
| ToF ranging | Distances of 8 × 8 zones, in meters | Where the floor is, and whether something is too close to the feet |
| Joint encoders and IMU | Joint angles, gravity direction and angular velocity | Where the camera and ToF point right now |

ToF only gives distances and cannot tell whether a reading belongs to the ball or the floor; recognizing what something is relies on the camera.

{#steps}
## From sensors to positions

```{concept-diagram}
kind: onboard-sensors
label: The RGB image and 8×8 ToF distances from one capture, and how the onboard policy uses each
description: The first row is a line drawing of the camera view; green boxes are the ball (confidence 0.95) and goal (0.95) found by the detector, used to recognize targets. The second row is the 64 ToF zone distances from the same instant, in meters; dashed cells have no valid return; they are used to find the floor and check near range. The third row is the joint encoders and IMU, used to compute the camera and ToF orientation. The three lanes combine into ball and goal positions relative to the robot.
rgb:
  title: RGB camera · 960 × 540 pixels
  size: [960, 540]
  boxes:
  - [ball, 0.954, [443, 312, 576, 448]]
  - [goal, 0.946, [362, 119, 607, 273]]
tof:
  title: ToF ranging · 8 × 8 zones · meters
  grid:
  - [null, null, null, null, null, null, null, null]
  - [null, null, 1.15, 1.15, 1.15, null, null, null]
  - [2.34, 2.32, 1.30, 1.30, 1.23, 1.23, 2.51, 2.61]
  - [1.27, 1.26, 1.25, 1.25, 1.26, 1.20, 1.31, 1.35]
  - [0.80, 0.79, 0.78, 0.86, 0.50, 0.50, 0.81, 0.83]
  - [0.63, 0.62, 0.61, 0.61, 0.48, 0.48, 0.63, 0.65]
  - [0.50, 0.49, 0.48, 0.51, 0.49, 0.49, 0.50, 0.51]
  - [0.42, 0.41, 0.43, 0.42, 0.42, 0.43, 0.42, 0.43]
proprio: [Joint encoders + IMU, "Joint angles, gravity direction and angular velocity"]
steps:
- [Recognize targets, "Detector boxes for ball and goal\nUnits: pixels"]
- [Floor and near range, "Fit a floor plane\nStop if something is too close to the feet"]
- [Sensor orientation, From same-time joint angles]
result: [Ball and goal positions, "Ball position on the floor\nRobot position and heading relative to the goal"]
caption: The sensor panels come from the same capture as the start of the recording above. The camera panel is a line drawing placed by the detector boxes; the ToF numbers are recorded ranges, darker means nearer, and dashed cells have no valid return. The cells of about 0.5 m below the middle face the ball.
```

After each capture, the onboard policy does three things:

1. **Recognize targets.** The detector boxes the ball and goal in the RGB image. A box is only a pixel position, not yet a place the robot can walk to.
2. **Work out the floor and distances.** ToF fits the floor plane and joint angles give the camera orientation. The known ball size helps estimate the ball's position; the bases of the two goalposts and their known spacing give the robot's position and heading relative to the goal.
3. **Reject unreliable information.** If an observation is too old, a target is only partly visible, successive results disagree or the floor cannot be found, the policy does not use that result and stops to look again; it also stops moving when something is too close to its feet. A zone without a valid return only means unknown; it does not mean the way ahead is clear.

{#truth}
## Sensors and scoring

The onboard policy reads only these sensors, never the true ball or goal coordinates from the simulator. Whether a shot scored is judged separately by the evaluator using simulator truth. Everything runs in simulation and has not been tested on a real robot.

:::{enrichment} Key code parameters
- **Detector.** `models/scene-detector/deployment.json` names the weights `detector.pt` and the classes `ball` and `goal`; the scene loads it through its `--detector-deployment` parameter.
- **Perception and geometry.** `lab/runtime/simulation/core/football/perception.py` handles detector boxes and sensor orientation; `football/ground_plane.py` fits the floor from ToF.
- **Sensor settings.** `lab/runtime/simulation/core/sensors/reports/sensor-profile.json` records the simulated camera and ToF settings.
:::

::::{quiz}
The detector boxes the ball, and a few ToF zones in the middle read nearer. Why does the onboard policy combine the two?

:::{answer}
The box only says which pixels contain the ball, and ToF only gives distances without knowing what the object is. Combining them with the camera orientation at that instant and the floor fitted from ToF gives how far the ball is on the floor.
:::
::::

{#references}
## References

{.references}
- [Ultralytics YOLO documentation ↗](https://docs.ultralytics.com/)
- [VL53L8CX 8×8 multizone ToF sensor ↗](https://www.st.com/en/imaging-and-photonics-solutions/vl53l8cx.html)
- [Microduck open-source project ↗](https://github.com/pollen-robotics/microduck)
