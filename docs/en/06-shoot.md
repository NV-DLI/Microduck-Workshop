---
myst:
  html_meta:
    description: Run the onboard policy and observe how sensing, stage decisions and low-level skills work together over each control cycle.
---

# 4.3 Shoot!

:::{lesson-intro}
:meta: Section 4.3 · Estimated time 20 minutes

Run the onboard policy and observe how sensing, stage decisions and low-level skills work together over each control cycle.
:::

Review the shooting stages in [Section 4.2](06-onboard.md), then run the experiment and observe how the stages progress.

{#run}
## Shoot!

Prepare the course container as described in [Section 1.2](setup.md), then return to its terminal. If it is already running, enter it from the host with `docker exec -it microduck-workshop bash`. Run:

```{code-block} bash
:caption: Terminal · Inside the container
:class: terminal

python3 scripts/run.py football --output outputs/onboard-football
```

:::{enrichment} Parameter details

- **`--output`**: A new folder; rounds are kept in separate `episode-001`, `episode-002`, etc. folders
:::

The robot automatically observes, approaches the ball and attempts a shot. The first round starts from the nominal position. Click the Isaac Sim window and press **R** to restart: the window stays open and each round starts from the standing initialization, with the start randomly offset by up to 5 cm forward/backward and up to 5 cm sideways, and the heading turned by up to 15° clockwise or counterclockwise. The ball and goal stay in place. Allow a moment for the scene to initialize; the task then starts automatically.

The student experiment has no overall time limit. Scoring, missing or being unable to kick does not automatically pause the simulation; sensors keep updating. Press **R** to reset the attempt. Close Isaac or press **Ctrl+C** in its terminal to stop the program. Perception, alignment or handover failures pause the current maneuver and trigger another observation attempt.

The simulation renders several views at once, the third-person viewport, the head RGB camera and ToF, so its speed depends on your hardware and the motion in the window may look jerky; the task advances on simulation time.

:::{course-image} media/isaac-ui-onboard-sensors.png
:alt: The complete Isaac Sim window of the onboard shot, showing the green pitch, the head RGB image and the 8×8 ToF ranges.

Watch the robot on the left and compare what it sees on the right: do the detection boxes stay on the ball and goal, and do the ToF readings ahead of the feet shrink as it approaches? NR means no return and does not mean that nothing is ahead.
:::

`summary.json` summarizes saved rounds. Each round's `manifest.json` includes `navigation_retries`, which records why the robot reacquired observations. A visible detection box does not mean the geometry has passed its checks; if the close-range position is not fresh enough, the policy pauses to observe again. Also check lateral error and heading during the approach, whether the kick conditions were met and whether the robot recovered its balance after contact.

When you press R, the program saves the current record before starting the next round.

{#timing}
## The onboard control cycle

```{control-timing}
kind: onboard-theory
```

In <a href="08-groot-interface.html#timing">Section 5.2</a>, compare how the sensing and action cycles change when GR00T generates the high-level requests.
