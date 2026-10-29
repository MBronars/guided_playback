import unittest
import numpy as np
from collection.sim1.inputs import validate_input
from collection.sim1.compare import compare_arrays


def material():
    return dict(vertices_world_m=np.array([[0.,0.,.81],[1.,0.,.81],[0.,1.,.81]]),
        triangles=np.array([[0,1,2]]), edge_indices=np.array([[-1,-1,0,1],[-1,-1,1,2],[-1,-1,2,0]]),
        panel_names=np.array(['panel']), tri_poses=np.eye(2)[None], tri_areas_m2=np.array([.5]),
        particle_mass_kg=np.full(3,1/3), edge_rest_angle_rad=np.zeros(3),
        edge_rest_length_m=np.array([1.,np.sqrt(2),1.]), edge_bending_properties=np.zeros((3,2)),
        face_panel=np.zeros(1,dtype=int), material_triangle_uv_m=np.array([[[0.,0.],[1.,0.],[0.,1.]]]))


class InputTests(unittest.TestCase):
    def test_explicit_metric_and_source_area_mass(self):
        a=material(); self.assertIs(validate_input(a),a)

    def test_bad_mass_is_not_normalized_away(self):
        a=material(); a['particle_mass_kg']*=2
        with self.assertRaisesRegex(ValueError,'mass mismatch'): validate_input(a)

    def test_area_and_metric_disagreement_rejected(self):
        a=material(); a['tri_poses']*=2
        with self.assertRaisesRegex(ValueError,'metric/area'): validate_input(a)

    def test_area_preserving_metric_corruption_rejected(self):
        a=material(); a['material_triangle_uv_m'][0,1,0]=2; a['material_triangle_uv_m'][0,2,1]=.5
        with self.assertRaisesRegex(ValueError,'reference metric mismatch'): validate_input(a)

    def test_bad_indices_and_nonfinite_inputs(self):
        a=material(); a['triangles'][0,0]=9
        with self.assertRaises(ValueError): validate_input(a)
        a=material(); a['vertices_world_m'][0,0]=np.nan
        with self.assertRaises(ValueError): validate_input(a)

    def test_numerical_comparison_reports_not_hides_difference(self):
        a={'particle_q':np.zeros((2,3,3),np.float32)}
        b={'particle_q':a['particle_q'].copy()}; b['particle_q'][1,0,0]=.01
        result=compare_arrays(a,b)['particle_q']
        self.assertFalse(result['exact']); self.assertAlmostEqual(result['max_vector_difference'],.01)

    def test_unmatched_array_dtype_is_not_silently_cast_for_parity(self):
        with self.assertRaises(ValueError): compare_arrays({'x':np.zeros(2,np.float32)},{'x':np.zeros(2,np.float64)})
