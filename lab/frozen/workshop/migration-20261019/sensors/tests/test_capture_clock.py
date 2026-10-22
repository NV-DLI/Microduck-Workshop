import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture_clock import CaptureClock


class CaptureClockTests(unittest.TestCase):
    def test_15hz_captures_on_200hz_ticks_preserve_elapsed_time(self):
        clock = CaptureClock()
        intervals = []
        for timestamp in (0.070, 0.135, 0.200, 0.270):
            intervals.append(clock.elapsed(timestamp))
            clock.commit(timestamp)
        for actual, expected in zip(intervals, (0.070, 0.065, 0.065, 0.070)):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(sum(intervals), 0.270)

    def test_failed_capture_does_not_advance_and_reset_can_change_origin(self):
        clock = CaptureClock()
        self.assertAlmostEqual(clock.elapsed(0.07), 0.07)
        self.assertAlmostEqual(clock.elapsed(0.135), 0.135)
        clock.commit(0.135)
        self.assertEqual(clock.elapsed(0.135), 0.0)
        clock.reset(2.0)
        self.assertAlmostEqual(clock.elapsed(2.07), 0.07)
        clock.reset()
        self.assertAlmostEqual(clock.elapsed(0.07), 0.07)

    def test_invalid_or_backwards_time_is_rejected(self):
        clock = CaptureClock(1.0)
        for timestamp in (0.9, float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(ValueError):
                clock.elapsed(timestamp)
        self.assertEqual(clock.last_timestamp_s, 1.0)


if __name__ == '__main__':
    unittest.main()
