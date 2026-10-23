"""CPU TEST FIXTURE ONLY: synthetic leg odometry for recovery fixtures.

Encodes a body velocity in encoder_qd channels 0/1 so the unchanged public
memory integrates a known relative pose. Never used by the executor; the
production auditor uses the controller's SoleContact odometry instead.
"""


class FixtureLeg:
    def estimate(self, q, qd, gyro, gravity, joint_names):
        return {
            "velocity_body_m_s": [float(qd[0]), float(qd[1]), 0.0],
            "diagnostic_low_confidence": False,
        }


def make_leg():
    return FixtureLeg()
