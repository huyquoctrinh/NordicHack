import unittest

from survival_agent.episode import starts_new_episode


class EpisodeTests(unittest.TestCase):
    def test_initial_frame_after_probe_resets(self):
        self.assertTrue(starts_new_episode(.1, [{"age": .1}], 0.0))

    def test_normal_frame_does_not_reset(self):
        self.assertFalse(starts_new_episode(.2, [{"age": .2}], .1))
        self.assertFalse(starts_new_episode(80.0, [{"age": 1.0}], 79.9))

    def test_clock_rollback_resets(self):
        self.assertTrue(starts_new_episode(0.0, [], 10.0))
