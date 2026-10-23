"""Read-only Newton foot-to-ground contact evidence, never a controller input."""

import hashlib
import json
from pathlib import Path


def configure_ground_filter(cfg):
    sensor = cfg.scene.contact_forces
    if sensor.filter_prim_paths_expr or sensor.filter_shape_prim_expr:
        raise ValueError("Do not overwrite an existing contact counterpart contract")
    sensor.filter_shape_prim_expr = [cfg.scene.terrain.prim_path + "/.*"]


class GroundContactEvidence:
    def __init__(self, env, foot_ids, output):
        from environment_hook import T
        from newton import GeoType

        self.T = T
        self.sensor = env.scene["contact_forces"]
        self.foot_ids = foot_ids
        model = env.sim.backend_manager.get_model()
        view = self.sensor.contact_view
        if view.sensing_type != "body" or view.counterpart_type != "shape":
            raise ValueError(
                "Require actual body sensing with shape-level counterparts"
            )
        matrix = self.sensor.data.force_matrix_w
        if matrix is None or tuple(T(matrix).shape) != (
            env.num_envs,
            self.sensor.num_sensors,
            1,
            3,
        ):
            raise ValueError("Each foot must have exactly one ground-plane counterpart")
        body = model.shape_body.numpy()
        kind = model.shape_type.numpy()
        flags = model.shape_flags.numpy()
        transform = model.shape_transform.numpy()
        records = []
        for world in range(env.num_envs):
            for side, foot in zip(("left", "right"), foot_ids):
                row = world * self.sensor.num_sensors + foot
                counterparts = view.counterpart_indices[row]
                if len(counterparts) != 1:
                    raise ValueError("Ground counterpart missing or ambiguous")
                ground = counterparts[0]
                label = model.shape_label[ground]
                if not label.startswith(env.cfg.scene.terrain.prim_path + "/"):
                    raise ValueError("Matched an unrelated shape")
                if (
                    int(body[ground]) != -1
                    or int(kind[ground]) != int(GeoType.PLANE)
                    or not int(flags[ground]) & 2
                ):
                    raise ValueError(
                        "Ground must be an actual global static collidable plane"
                    )
                sensing_body = view.sensing_indices[row]
                sensing_label = model.body_label[sensing_body]
                if not sensing_label.startswith(
                    f"/World/envs/env_{world}/Robot/"
                ) or not sensing_label.endswith("/ankle_" + side):
                    raise ValueError("Foot force row/body/world mapping is incorrect")
                records.append(
                    dict(
                        world=world,
                        side=side,
                        sensor_row=row,
                        sensor_body_index=sensing_body,
                        sensor_body_label=sensing_label,
                        counterpart_shape_index=ground,
                        counterpart_shape_label=label,
                        counterpart_shape_body_index=int(body[ground]),
                        counterpart_shape_type=int(kind[ground]),
                        counterpart_shape_flags=int(flags[ground]),
                        counterpart_shape_transform_xyzw=transform[ground].tolist(),
                    )
                )
        self.path = Path(output) / "actual-ground-contact-binding.json"
        self.path.write_text(
            json.dumps(
                dict(
                    schema="microduck.actual-ground-contact-binding.v1",
                    source_sha256=hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    force_semantics="Newton CONTACT_F actual foot-body to static ground-shape force in world coordinates, N; excludes ball, goal and self contacts",
                    counterpart_type=view.counterpart_type,
                    all_worlds_present=True,
                    rows=records,
                ),
                indent=2,
            )
            + "\n"
        )

    def read(self):
        return self.T(self.sensor.data.force_matrix_w)[:, self.foot_ids, 0].clone()

    def report(self):
        return dict(
            path=self.path.name,
            sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            sampling_hz=200,
            actor_input=False,
            controls_unchanged=True,
        )
