import unittest
from types import SimpleNamespace

import numpy as np

from collection.sim1.actual_export import export_actual, make_templates, surface_samples, training_sample


class Array:
    def __init__(self, value): self.value = np.asarray(value)
    def numpy(self): return self.value.copy()


class ActualExportTests(unittest.TestCase):
    def setUp(self):
        S = 7
        bodies = np.zeros((S, 6, 7), np.float32); bodies[:, :, 6] = 1
        bodies[:, :, 0] = np.arange(6)[None, :] + np.arange(S)[:, None] * .01
        self.raw = dict(particle_q=np.zeros((S, 4, 3), np.float32), body_q=bodies,
            actual_joint_q=np.arange(S * 19, dtype=np.float32).reshape(S, 19) * .001,
            command_joint_q=np.ones((S - 1, 19), np.float32) * .9,
            command_openness=np.arange((S - 1) * 2).reshape(S - 1, 2) / 12,
            state_index=np.arange(S), transition_index=np.arange(S - 1),
            state_time_s=np.arange(S) / 60, fps=np.array(60.))
        self.template = dict(link_ids=np.repeat(np.arange(3), 2),
            left_local_points=np.zeros((6, 3)), right_local_points=np.zeros((6, 3)),
            left_body_ids=np.arange(3), right_body_ids=np.arange(3, 6))

    def test_actual_states_not_commands_and_distinct_sides(self):
        out = export_actual(self.raw, self.template)
        np.testing.assert_array_equal(out['joint_q'], self.raw['actual_joint_q'])
        np.testing.assert_allclose(out['right_ee_points'] - out['left_ee_points'],
                                   np.broadcast_to([3, 0, 0], (7, 6, 3)), atol=2e-7)
        self.assertFalse(out['transition_valid'][-1])

    def test_integer_downsample_preserves_all_commands_and_terminal(self):
        out = export_actual(self.raw, self.template, stride=3)
        np.testing.assert_array_equal(out['source_state_index'], [0, 3, 6])
        np.testing.assert_array_equal(out['command_interval_offsets'], [0, 3, 6])
        np.testing.assert_array_equal(out['native_command_joint_q'], self.raw['command_joint_q'])
        np.testing.assert_array_equal(out['cloth_points'][-1], self.raw['particle_q'][-1])
        np.testing.assert_allclose(out['delta_joint_q'][:-1], np.diff(self.raw['actual_joint_q'][::3], axis=0))
        self.assertEqual(float(out['fps']), 20.)
        self.assertNotIn('command_joint_q', out)  # no fabricated coarse held commands

    def test_rejects_incomplete_terminal_and_bad_clock(self):
        with self.assertRaises(ValueError): export_actual(self.raw, self.template, stride=4)
        self.raw['state_time_s'][2] += 1e-6
        with self.assertRaises(ValueError): export_actual(self.raw, self.template)

    def test_rejects_command_state_confusion_and_aliasing(self):
        self.template['right_body_ids'] = np.arange(3)
        with self.assertRaises(ValueError): export_actual(self.raw, self.template)
        self.template['right_body_ids'] = np.arange(3, 6)
        self.raw['command_joint_q'] = self.raw['actual_joint_q']
        with self.assertRaises(ValueError): export_actual(self.raw, self.template)

    def test_sampling_is_reconstructible_and_degenerate_mesh_rejected(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.]])
        faces = np.array([[0, 1, 2]])
        points, ids, bary = surface_samples(vertices, faces, 50, np.random.default_rng(42))
        np.testing.assert_allclose(points, np.sum(vertices[faces[ids]] * bary[:, :, None], axis=1))
        np.testing.assert_allclose(bary.sum(axis=1), 1)
        with self.assertRaises(ValueError): surface_samples(vertices * 0, faces, 5, np.random.default_rng(1))

    def test_templates_apply_each_shapes_scale_and_transform(self):
        poses = np.zeros((6, 7)); poses[:, 6] = 1; poses[:, 2] = np.arange(6)
        mesh = SimpleNamespace(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.]]),
                               indices=np.array([0, 1, 2]))
        model = SimpleNamespace(body_key=['left_link16', 'left_link17', 'left_link18',
                                         'right_link26', 'right_link27', 'right_link28'],
            shape_body=Array(np.arange(6)), shape_flags=Array(np.ones(6, dtype=int)),
            shape_scale=Array(np.ones((6, 3)) * 2), shape_transform=Array(poses), shape_source=[mesh] * 6)
        template, evidence = make_templates(model, count=10)
        for side, offset in [('left', 0), ('right', 3)]:
            np.testing.assert_array_equal(template[side + '_local_points'][:, 2], template['link_ids'] + offset)
        self.assertEqual(evidence['points_per_link'], [2, 4, 4])
        self.assertTrue(np.any(template['left_local_points'][:, :2] > 1))

    def test_training_view_rejects_wrong_clock_and_never_double_strides(self):
        self.template['link_ids'] = np.repeat(np.arange(3), [200, 400, 400])
        self.template['left_local_points'] = np.zeros((1000, 3))
        self.template['right_local_points'] = np.zeros((1000, 3))
        for stride in (1, 3):
            with self.assertRaises(ValueError):
                training_sample(export_actual(self.raw, self.template, stride=stride))
        exported = export_actual(self.raw, self.template, stride=2)
        sample = training_sample(exported)
        self.assertEqual(sample['particle_pos'].shape, (4, 2004, 3))
        np.testing.assert_array_equal(sample['source_state_index'], [0, 2, 4, 6])
        np.testing.assert_array_equal(sample['particle_pos'][:, :4],
            exported['cloth_points'] + np.array([-.65, 0., -.8], dtype=np.float32))


if __name__ == '__main__': unittest.main()
