import unittest
from types import SimpleNamespace
import numpy as np
from collection.sim1.robot import GripperCollisionFilter, gripper_filter


class Array:
    def __init__(self, value): self.value = np.asarray(value)
    def numpy(self): return self.value.copy()
    def assign(self, value): self.value = value.copy()


class GripperFilterTests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(shape_flags=Array([7, 3, 7, 7, 3, 7, 7]),
            joint_child=Array([0, 1, 3, 4]),
            joint_key=['left_joint17', 'left_joint18', 'right_joint27', 'right_joint28'],
            body_shapes={i: [i] for i in range(7)})
        self.env = SimpleNamespace(lee_index=2, ree_index=5)

    def test_open_close_cycles_restore_only_original_contact_bits(self):
        filt = GripperCollisionFilter(self.env, self.model)
        for _ in range(3):
            filt.apply([1., 1.])
            np.testing.assert_array_equal(self.model.shape_flags.numpy(), [3, 3, 3, 3, 3, 3, 7])
            filt.apply([0., 0.])
            np.testing.assert_array_equal(self.model.shape_flags.numpy(), [7, 3, 7, 7, 3, 7, 7])

    def test_independent_sides_and_unchanged_native_threshold(self):
        filt = GripperCollisionFilter(self.env, self.model)
        filt.apply([.6, .60001])
        np.testing.assert_array_equal(self.model.shape_flags.numpy(), [7, 3, 7, 3, 3, 3, 7])

    def test_one_shot_home_preserves_legacy_output(self):
        gripper_filter(self.env, self.model, [1., 0.])
        np.testing.assert_array_equal(self.model.shape_flags.numpy(), [3, 3, 3, 7, 3, 7, 7])

    def test_rejects_invalid_openness_without_mutation(self):
        before = self.model.shape_flags.numpy()
        for openness in ([np.nan, 0], [0, 2], [0]):
            with self.assertRaises(ValueError): gripper_filter(self.env, self.model, openness)
        np.testing.assert_array_equal(before, self.model.shape_flags.numpy())
