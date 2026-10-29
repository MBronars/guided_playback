from types import SimpleNamespace
import unittest
import numpy as np
from collection.sim1.qualify import validate_options
from collection.sim1.compare import MATCH_FIELDS, match_protocols
from collection.sim1.geometry import strict_crossing


class DiagnosticOptionsTests(unittest.TestCase):
    def options(self,**kwargs):
        return SimpleNamespace(**dict(dict(frames=30,substeps=24,wall_seconds=1200,
            self_contact_distance=.002,probe_vertex=[]),**kwargs))

    def test_native_default_and_long_budget(self):
        for frames in (30,720):validate_options(self.options(frames=frames))

    def test_undeclared_physics_or_unbounded_recording_rejected(self):
        for change in (dict(frames=721),dict(substeps=12),dict(self_contact_distance=0),
                       dict(self_contact_distance=float('nan')),dict(wall_seconds=1201)):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_options(self.options(**change))

    def test_probe_ids_are_bounded_distinct_material_ids(self):
        validate_options(self.options(probe_vertex=[2563,6976]))
        for ids in ([1,1],[-1],list(range(9))):
            with self.assertRaises(ValueError):validate_options(self.options(probe_vertex=ids))

    def test_crossings_separation_and_uncertainty_are_distinct(self):
        a=np.array([[0.,0.,0.],[1,0,0],[0,1,0]])
        b=np.array([[.25,.25,-1.],[.25,.25,1.],[.8,.25,.5]])
        self.assertEqual(strict_crossing(a,b),'proper_noncoplanar_crossing')
        self.assertEqual(strict_crossing(a,b+3),'separated_by_plane')
        self.assertEqual(strict_crossing(a,a+.01*np.array([1,0,0])),'coplanar_uncertain')


class MatchedProtocolTests(unittest.TestCase):
    def protocol(self, version='v2'):
        result = dict.fromkeys(MATCH_FIELDS, 'same')
        result.update(schema='sim1-matched-home-execution-' + version, mode='eager')
        if version == 'v1':
            del result['self_contact_distance_m'], result['probe_vertex_ids']
        return result

    def test_legacy_and_current_required_fields(self):
        for version in ('v1', 'v2'):
            a = self.protocol(version)
            match_protocols(a, dict(a, mode='captured'))

    def test_missing_mandatory_field_on_both_sides_is_not_a_match(self):
        for version in ('v1', 'v2'):
            a = self.protocol(version)
            del a['source_hashes']
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                match_protocols(a, a)

    def test_current_schema_requires_contact_metadata(self):
        a = self.protocol()
        del a['self_contact_distance_m']
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            match_protocols(a, a)

    def test_changed_contact_setting_and_unknown_schema_rejected(self):
        a = self.protocol()
        with self.assertRaisesRegex(ValueError, 'Unmatched'):
            match_protocols(a, dict(a, self_contact_distance_m=.001))
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            match_protocols(dict(a, schema='unknown'), dict(a, schema='unknown'))
if __name__=='__main__':unittest.main()
