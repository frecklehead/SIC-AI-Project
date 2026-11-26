"""Pure, deterministic tests for BacklogPolicy. Not run in this sandbox task."""
import unittest

from backlog_policy import BacklogPolicy


class FakeClock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


class BacklogPolicyTests(unittest.TestCase):
    def test_normal_flow_does_not_drop(self):
        clock = FakeClock(0.2)
        policy = BacklogPolicy(clock=clock)
        self.assertEqual(policy.samples_to_drop(0.0, 4800, 48000), 0)

    def test_ten_second_client_pause_does_not_age_new_input(self):
        clock = FakeClock(10.0)
        policy = BacklogPolicy(clock=clock)
        # The reader stamps the first post-pause message at receipt time.
        self.assertEqual(policy.samples_to_drop(10.0, 480, 48000), 0)

    def test_one_second_queue_stall_skips_to_recovery_window(self):
        clock = FakeClock(1.0)
        policy = BacklogPolicy(clock=clock)
        self.assertEqual(policy.samples_to_drop(0.0, 48000, 48000), 40800)


if __name__ == "__main__":
    unittest.main()
