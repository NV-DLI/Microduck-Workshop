#!/usr/bin/env python3
"""Execute: Microduck simulation and control."""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
from paths import M, K10_DEPLOYMENT

A = M / "evaluation"
F = M / "football"
S = M / "sensors"
sys.path[:0] = [str(D), str(A), str(F), str(S), str(M / "gait")]
import isaac_execute as entry
import controller
import nav2_bridge
import isaac_beta_body_imu as body_module
from execute_long_episode import BoundedLifetimeSession
from angular_navigation_native import AngularBallNavigator

CANDIDATE = S / "processing/body50_scalar_finite_range"
sys.path.insert(0, str(CANDIDATE))
spec = importlib.util.spec_from_file_location(
    "football_v121_scalar_finite_adapter", CANDIDATE / "adapter.py"
)
finite = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = finite
spec.loader.exec_module(finite)
NativeBody = body_module.IsaacBodyImu
base_build = entry.build_cfg
base_run = entry.run
contract = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FiniteBody(NativeBody):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        finite.install_before_first_update(self)

    def metadata(self):
        return {**super().metadata(), **finite.parameters_metadata()}


def build(a):
    if a.layout != "empty":
        raise ValueError(
            "This preparation-waypoint development is limited to the empty layout"
        )
    if a.trained_recipe_dir is not None or a.bounded_actions:
        raise ValueError("Native v121 cannot use a projected/trained action adapter")
    if a.kick_model is not None:
        raise ValueError(
            "v121 binds the reviewed fixed K10 runtime; legacy --kick-model is not supported"
        )
    if (
        a.policy_imu != "body50"
        or a.body_fusion_tau_s != 2.0
        or a.body_added_delay_s != 0.0
    ):
        raise ValueError("Select explicitly BODY50, tau2 and zero extra delivery delay")
    expected = M / "models/reference/velstand.onnx"
    if a.model.resolve() != expected.resolve():
        raise ValueError(
            "This independently named reference uses the pinned community actor, never a student actor"
        )
    from recovery_config import verify_approval

    approval = verify_approval(
        a.recovery_approval, parameters_dir=D, allow_test_only=False
    )
    if approval["test_only"] or not approval["parameters"].calibrated:
        raise ValueError("v121 production requires calibrated root approval")
    cfg, fixture = base_build(a)
    from isaaclab_tasks.contrib.microduck.mdp.actions import BiasedJointPositionAction

    if (
        cfg.actions.joint_pos.class_type is not BiasedJointPositionAction
        or cfg.actions.joint_pos.clip is not None
    ):
        raise ValueError(
            "Original BiasedJointPositionAction without target/action clipping is required"
        )
    if float(cfg.actions.joint_pos.scale) != 1.0:
        raise ValueError("Official scale1 is mandatory")
    files = {
        Path(__file__),
        D / "design.json",
        D / "design-recovery.json",
        D / "release.json",
        Path(approval["manifest_path"]),
        expected,
        expected.with_name("manifest.json"),
    }
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path:
            path = Path(path).resolve()
            if (
                path.is_file()
                and path.suffix == ".py"
                and path.is_relative_to(M)
                and not path.is_relative_to(M / "runtime")
            ):
                files.add(path)
    files.update(D.glob("*.py"))
    files.update(CANDIDATE.glob("*.py"))
    files.update([D / "sole-hull-vertices.json", D / "odometry-selection-source.json"])
    files.update(
        [
            A / "ball_read_adapter.py",
            A / "ball_status_adapter.py",
            F / "ball_memory/memory.py",
            F / "kick_policy/runtime.py",
            A / "ground_contact_evidence.py",
        ]
    )
    files.update((S / "processing/ground_geometry").glob("*.py"))
    files.update((S / "processing/ground_estimator").glob("*.py"))
    deployment = K10_DEPLOYMENT
    kd = json.loads(deployment.read_text())
    files.update(
        [
            deployment,
            deployment.parent / "training-ledger.snapshot.json",
            Path(kd["onnx"]),
            Path(kd["checkpoint"]),
            *[Path(x) for x in kd["teacher_sha256"]],
        ]
    )
    contract.update(
        schema="microduck.native-football-reference.v121",
        qualification_claimed=False,
        model_identity="published community velstand reference; not student-trained model",
        model_sha256=sha(expected),
        native_action="scale1 affine raw+home-bias to native BAM; no EMA, projection or target slew; previous raw history",
        sensor={
            **finite.parameters_metadata(),
            "orientation_correction_tau_s": 2.0,
            "extra_delivery_delay_s": 0.0,
        },
        head_control="Official61D head-command input plus bounded public same-read outer head integral in permitted zero-body observation only; no raw action or physical state override",
        controller="Explicit point-estimate development: grounded current sphere, bounded final approach, same-read memory and physical neutral-head return, one native K10 kick; v121 bounded pre-kick retreat recovery with exact fixed-duration 1.6 s public-heading-feedback helper (max3, original clock, no kick authority)",
        retreat_recovery_approval={
            "manifest": approval["manifest_path"],
            "manifest_sha256": approval["manifest_sha256"],
            "parameters": approval["parameters"].as_dict(),
            "stage_a": approval["stage_a_binding"],
            "stage_b": approval["stage_b_binding"],
        },
        mechanical_interpretation="Original active/passive simulator stops and motor torque remain. Historical combined-output gate is recorded as a conservative course diagnostic, not proven hardware stop.",
        autonomous_kick_enabled=True,
        task_reset_or_assistance=False,
        shared_gpu_not_performance=True,
        source_sha256={str(p): sha(p) for p in sorted(files)},
        source_bytes={str(p): p.read_bytes() for p in sorted(files)},
    )
    return cfg, fixture


def run(a, cfg, fixture):
    public = {k: v for k, v in contract.items() if k != "source_bytes"}
    wrapper = a.output.with_name(a.output.name + "-native-v121-launcher.json")
    if wrapper.exists():
        raise FileExistsError("Retain original v121 launcher evidence")
    public.update(started_unix_s=time.time(), argv=sys.argv, status="starting")
    wrapper.write_text(json.dumps(public, indent=2) + "\n")
    try:
        base_run(a, cfg, fixture)
        public["status"] = "completed"
    except BaseException as e:
        public.update(status="failed", error=repr(e))
        raise
    finally:
        changed = [p for p, h in contract["source_sha256"].items() if sha(p) != h]
        public.update(source_changed_during_run=changed, ended_unix_s=time.time())
        if a.output.exists():
            archive = a.output / "native-v121-source-snapshot"
            archive.mkdir(exist_ok=True)
            for i, (p, b) in enumerate(contract["source_bytes"].items()):
                (archive / f"{i:03}-{Path(p).name}").write_bytes(b)
            manifest = a.output / "manifest.json"
            if manifest.exists():
                value = json.loads(manifest.read_text())
                value["native_reference_contract"] = public
                value["velocity_mapping"] = (
                    "See native_command_adapter on each control row; actual command is not an achieved velocity claim"
                )
                value["formal_qualification_claimed"] = False
                manifest.write_text(json.dumps(value, indent=2) + "\n")
        wrapper.write_text(json.dumps(public, indent=2) + "\n")
        if changed:
            raise RuntimeError("v121 dependencies changed during the run")


def main():
    from ball_reacquisition import BallReacquisitionNavigator
    from visibility_gate import install as install_visibility_gate

    install_visibility_gate()
    import head_adapter as head_module
    from native_head_domain import HeadCommandDomainAdapter

    head_module.HeadAdapter = HeadCommandDomainAdapter
    controller.BetaNavigator = BallReacquisitionNavigator
    body_module.IsaacBodyImu = FiniteBody
    nav2_bridge.Nav2Session = BoundedLifetimeSession
    entry.build_cfg = build
    entry.run = run
    entry.main()


if __name__ == "__main__":
    main()
