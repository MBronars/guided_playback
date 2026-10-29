"""Optional convex-component distances with first-order pose derivatives.

FCL supplies signed distances and witness points. Freezing the witnesses gives
the local distance derivative away from contact/feature switches. This does not
differentiate a simulator or establish contact forces. Components are explicit
convex proxies; overlapping parts do not define an exact union penetration SDF.
"""
from pathlib import Path
import sys

import numpy as np
import torch


class ConvexCollisionDistance:
    def __init__(self, geometry, *, object_body=0, other_bodies=(1, 2, 3), vendor_paths=()):
        # Optional isolated dependencies are appended after the model's numpy
        # and torch have loaded; never replace the model environment's packages.
        for path in vendor_paths:
            sys.path.append(str(Path(path).resolve()))
        try:
            import fcl
            from scipy.spatial import ConvexHull
        except ImportError as error:
            raise ImportError('Convex collision distances require python-fcl and scipy') from error
        self.fcl = fcl
        self.object_body, self.other_bodies = object_body, tuple(other_bodies)
        self.keys = tuple(sorted(geometry['bodies']))
        if object_body not in self.keys or any(k not in self.keys for k in self.other_bodies):
            raise ValueError('Missing requested collision bodies')
        self.parts = {}
        volumes, moments = [], []
        for key in self.keys:
            body = geometry['bodies'][key]
            vertices = np.asarray(body['vertices'], dtype=np.float64)
            objects = []
            for planes in body['planes']:
                planes = np.asarray(planes, dtype=np.float64)
                v = np.unique(vertices[(vertices @ planes[:, :3].T + planes[:, 3]).max(-1) <= 2e-5], axis=0)
                hull = ConvexHull(v)
                triangles = hull.simplices.copy()
                normals = np.cross(v[triangles[:, 1]]-v[triangles[:, 0]], v[triangles[:, 2]]-v[triangles[:, 0]])
                flip = (normals*hull.equations[:, :3]).sum(-1) < 0
                triangles[flip] = triangles[flip][:, [0, 2, 1]]
                remap = np.full(len(v), -1)
                remap[hull.vertices] = np.arange(len(hull.vertices))
                triangles, v = remap[triangles], v[hull.vertices]
                if key == object_body:
                    tetra = np.einsum('ij,ij->i', v[triangles[:, 0]],
                                     np.cross(v[triangles[:, 1]], v[triangles[:, 2]]))/6
                    if tetra.sum() <= 0:
                        raise ValueError('Collision component has nonpositive signed volume')
                    volumes.append(tetra.sum())
                    moments.append((tetra[:, None]*(v[triangles].sum(1)/4)).sum(0))
                faces = np.column_stack([np.full(len(triangles), 3), triangles]).ravel()
                objects.append(fcl.CollisionObject(fcl.Convex(v, len(triangles), faces)))
            self.parts[key] = objects
        self.com_local = np.sum(moments, axis=0)/sum(volumes)
        self.request = fcl.DistanceRequest(enable_nearest_points=True, enable_signed_distance=True,
                                          gjk_solver_type=fcl.GJKSolverType.GST_LIBCCD)

    def query(self, R, t):
        """One scene in self.keys order; world-space witnesses for each body pair."""
        if R.shape != (len(self.keys), 3, 3) or t.shape != (len(self.keys), 3):
            raise ValueError('Collision poses do not match geometry bodies')
        for key, rr, tt in zip(self.keys, R, t):
            transform = self.fcl.Transform(np.asarray(rr, dtype=np.float64), np.asarray(tt, dtype=np.float64))
            for obj in self.parts[key]:
                obj.setTransform(transform)
        distances, witnesses = [], []
        for key in self.other_bodies:
            best, witness = float('inf'), None
            for a in self.parts[self.object_body]:
                for b in self.parts[key]:
                    result = self.fcl.DistanceResult()
                    distance = self.fcl.distance(a, b, self.request, result)
                    if distance < best:
                        best, witness = distance, np.asarray(result.nearest_points)
            if witness is None or not np.isfinite(best) or not np.isfinite(witness).all():
                raise FloatingPointError('Nonfinite collision query')
            if abs(np.linalg.norm(witness[1]-witness[0])-abs(best)) > 5e-6:
                raise FloatingPointError('Signed distance and witness separation disagree')
            distances.append(best)
            witnesses.append(witness)
        return np.array(distances), np.array(witnesses)

    def __call__(self, R, t, rigid_idx):
        """Distances [..., other_bodies], with gradients to rotations/translations.

        Only first derivatives are supported. At exact contact, the normal is
        undefined and a zero subgradient is used; hinge tolerances avoid that
        point for penetration penalties. Query selection stays nondifferentiable.
        """
        columns = {int(k): i for i, k in enumerate(rigid_idx.tolist())}
        order = [columns[k] for k in self.keys]
        R, t = R[..., order, :, :], t[..., order, :]
        shape = t.shape[:-2]
        rr = R.detach().double().cpu().reshape(-1, len(order), 3, 3).numpy()
        tt = t.detach().double().cpu().reshape(-1, len(order), 3).numpy()
        results = [self.query(a, b) for a, b in zip(rr, tt)]
        distance = torch.as_tensor(np.stack([r[0] for r in results]), dtype=t.dtype, device=t.device).reshape(*shape, -1)
        witnesses = torch.as_tensor(np.stack([r[1] for r in results]), dtype=t.dtype, device=t.device).reshape(*shape, -1, 2, 3)
        a = self.keys.index(self.object_body)
        others = [self.keys.index(k) for k in self.other_bodies]
        wa, wb = witnesses[..., 0, :], witnesses[..., 1, :]
        with torch.no_grad():
            local_a = torch.einsum('...ji,...nj->...ni', R[..., a, :, :], wa-t[..., a, None, :])
            local_b = torch.einsum('...nji,...nj->...ni', R[..., others, :, :], wb-t[..., others, :])
            safe = distance.abs() > 1e-8
            denominator = torch.where(safe, distance, torch.ones_like(distance))
            normal = (wb-wa)/denominator[..., None]*safe[..., None]
        moved_a = torch.einsum('...ij,...nj->...ni', R[..., a, :, :], local_a)+t[..., a, None, :]
        moved_b = torch.einsum('...nij,...nj->...ni', R[..., others, :, :], local_b)+t[..., others, :]
        linear = ((moved_b-moved_a)*normal).sum(-1)
        return distance + (linear-linear.detach())
