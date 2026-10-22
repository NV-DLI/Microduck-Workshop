"""CPU equivalence against the exact production body/head wrapper operations."""
import ast
import math
from pathlib import Path
import types
import unittest

import numpy as np
import torch

from beta_head_imu import BetaHeadImu
from beta_mounts import get_mount
from isaac_beta_body_imu import IsaacBodyImu
from shared_imu_snapshot import SharedIsaacImuProducer


def legacy_head_class():
    # Import the two actual rig methods and their pure NumPy helpers, without
    # importing/launching Kit. This detects drift from the production algorithm.
    source = Path(__file__).resolve().parents[1] / 'isaac_beta_sensor_rig.py'
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('array', 'rotate', 'qmul')]
    rig = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'IsaacBetaSensors')
    methods = [node for node in rig.body if isinstance(node, ast.FunctionDef) and node.name in ('_head_pose', 'update_imu')]
    namespace = {'np': np}
    exec(compile(ast.Module(body=functions + methods, type_ignores=[]), str(source), 'exec'), namespace)
    return type('LegacyHead', (), {name: namespace[name] for name in ('_head_pose', 'update_imu')})


class Robot:
    body_names = ['trunk_base', 'jaw_soft']

    def __init__(self):
        self.data = types.SimpleNamespace(body_pos_w=torch.zeros(1, 2, 3),
            body_quat_w=torch.zeros(1, 2, 4), body_ang_vel_w=torch.zeros(1, 2, 3))

    def find_bodies(self, names, preserve_order):
        return [self.body_names.index(name) for name in names], names

    def move(self, t, order):
        elapsed = max(0., t - 1.)
        for body_id, scale in enumerate((.4, 1.)):
            angle = .1 * scale * math.sin(2 * elapsed)
            self.data.body_pos_w[0, body_id] = torch.tensor([
                .01 * math.sin(elapsed) ** 2, .003 * math.sin(3 * elapsed) ** 2, .14 + .12 * body_id])
            q = [math.sin(angle / 2), 0., 0., math.cos(angle / 2)]
            if order == 'wxyz':
                q = [q[3], *q[:3]]
            self.data.body_quat_w[0, body_id] = torch.tensor(q)
            self.data.body_ang_vel_w[0, body_id] = torch.tensor([
                .2 * scale * math.cos(2 * elapsed) if elapsed > 0 else 0., 0., 0.])


def make_head(robot, order):
    head = legacy_head_class()()
    head.robot, head.head_id, head.body_quaternion_order = robot, None, order
    head.mount = get_mount('rl-jaw-soft-20261012')
    head.imu = BetaHeadImu(seed=83)
    reset_head(head)
    return head


def reset_head(head):
    head.imu.reset()
    head._imu_last_time = head._imu_last_position = head._imu_last_velocity = None
    head._imu_batch = []


class SharedSensorTests(unittest.TestCase):
    def test_exact_delivered_packets_for_motion_and_resets(self):
        for order in ('xyzw', 'wxyz'):
            robot = Robot()
            legacy_body = IsaacBodyImu(robot, seed=84, body_quaternion_order=order)
            batched_body = IsaacBodyImu(robot, seed=84, body_quaternion_order=order)
            legacy_head, batched_head = make_head(robot, order), make_head(robot, order)
            producer = SharedIsaacImuProducer(batched_body, batched_head)
            valid_body, valid_head = 0, 0
            for episode in range(2):
                if episode:
                    legacy_body.reset(); batched_body.reset()
                    reset_head(legacy_head); reset_head(batched_head)
                    producer.reset_after_sensor_resets()
                for step in range(1, 601):
                    t = step * .005
                    robot.move(t, order)
                    expected_head = legacy_head.update_imu(timestamp_s=t)
                    expected_body = legacy_body.update(timestamp_s=t)
                    result = producer.update(timestamp_s=t)
                    self.assertEqual(result['body'], expected_body)
                    self.assertEqual(result['head'], expected_head)
                    self.assertEqual(set(result), {'body', 'head'})
                    valid_body += sum(packet['orientation_valid'] for packet in result['body'])
                    valid_head += sum(packet['orientation_valid'] for packet in result['head'])
            self.assertGreater(valid_body, 100)
            self.assertGreater(valid_head, 100)
            self.assertEqual(producer.cpu_snapshot_transfers, 1200)
            self.assertFalse(producer.metadata()['gpu_performance_verified'])

    def test_duplicate_tick_and_mixed_update_paths_rejected(self):
        robot = Robot(); robot.move(.005, 'xyzw')
        body = IsaacBodyImu(robot)
        head = make_head(robot, 'xyzw')
        producer = SharedIsaacImuProducer(body, head)
        producer.update(timestamp_s=.005)
        with self.assertRaises(ValueError):
            producer.update(timestamp_s=.005)
        body.update(timestamp_s=.01)
        with self.assertRaisesRegex(ValueError, 'both adapters'):
            producer.update(timestamp_s=.01)


if __name__ == '__main__':
    unittest.main()
