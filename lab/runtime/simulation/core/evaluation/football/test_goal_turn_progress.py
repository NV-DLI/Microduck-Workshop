import unittest
from test_track_prealignment import TrackAlignment
from test_neutral_fine_step import packet


class NetTurn(unittest.TestCase):
    def test_rate_spike_without_observed_progress_does_not_stop(self):
        f, m, r = TrackAlignment().start(0.10)
        for k in range(78, 90):
            t = k / 50
            f.capture_s = t
            f.goal_s = t
            f.goal_heading = 0.09
            q = packet(t)
            q["gyro_body_rad_s"] = [0.0, 0.0, 0.8]
            r = f.policy(t=t, packet=q, memory=m, actual_head=[0] * 4)
        self.assertEqual(f.state, "TURN")
        self.assertEqual(r["velocity"], [0.0, 0.0, 0.8])
        t = 1.80
        f.capture_s = t
        f.goal_s = t
        f.goal_heading = 0.06
        q = packet(t)
        q["gyro_body_rad_s"] = [0.0, 0.0, 0.8]
        r = f.policy(t=t, packet=q, memory=m, actual_head=[0] * 4)
        self.assertEqual(f.state, "SETTLE")
        self.assertEqual(r["velocity"], [0.0, 0.0, 0.0])

    def test_goal_crossing_still_stops_at_original_minimum(self):
        f, m, r = TrackAlignment().start(0.10)
        for k in range(78, 84):
            t = k / 50
            f.capture_s = t
            f.goal_s = t
            f.goal_heading = 0.0
            r = f.policy(t=t, packet=packet(t), memory=m, actual_head=[0] * 4)
        self.assertEqual(f.state, "SETTLE")


if __name__ == "__main__":
    unittest.main()
