"""Independent saved-frame self-intersection checks, not continuous collision proof.

Open3D supplies candidates; float64 segment/triangle checks classify strict
crossings separately from coplanar/boundary uncertainty. Shared-vertex pairs are
excluded. No uncertain contact is silently treated as a certified clear pair.
"""
import numpy as np


def strict_crossing(a,b,distance_tol=1e-10,barycentric_tol=1e-8):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    def normal(tri):
        n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);length=np.linalg.norm(n)
        return n/length if length>1e-15 else None
    na,nb=normal(a),normal(b)
    if na is None or nb is None:return 'degenerate_uncertain'
    da=(a-b[0])@nb;db=(b-a[0])@na
    if np.max(np.abs(da))<=distance_tol and np.max(np.abs(db))<=distance_tol:
        return 'coplanar_uncertain'
    for edges,tri,n in ((a,b,nb),(b,a,na)):
        e1,e2=tri[1]-tri[0],tri[2]-tri[0]
        g00,g01,g11=np.dot(e1,e1),np.dot(e1,e2),np.dot(e2,e2)
        determinant=g00*g11-g01*g01
        if determinant<=0:return 'degenerate_uncertain'
        for i,j in ((0,1),(1,2),(2,0)):
            d0,d1=np.dot(edges[i]-tri[0],n),np.dot(edges[j]-tri[0],n)
            if not ((d0>distance_tol and d1 < -distance_tol) or (d1>distance_tol and d0 < -distance_tol)):continue
            t=d0/(d0-d1);point=edges[i]+t*(edges[j]-edges[i]);v=point-tri[0]
            p0,p1=np.dot(v,e1),np.dot(v,e2)
            u=(g11*p0-g01*p1)/determinant;w=(g00*p1-g01*p0)/determinant
            if u>barycentric_tol and w>barycentric_tol and u+w<1-barycentric_tol:
                return 'proper_noncoplanar_crossing'
    if (np.min(da)>distance_tol or np.max(da)<-distance_tol or np.min(db)>distance_tol or np.max(db)<-distance_tol):
        return 'separated_by_plane'
    return 'boundary_or_near_tolerance_uncertain'


def self_intersections(points,faces):
    import open3d as o3d
    from .contracts import finite_array
    q=finite_array(points,'points',(None,3)).astype(np.float64)
    f=finite_array(faces,'faces',(None,3))
    if not len(f) or f.dtype.kind not in 'iu' or f.min()<0 or f.max()>=len(q):
        raise ValueError('Valid nonempty triangle indices required')
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(q),o3d.utility.Vector3iVector(f))
    candidates=np.asarray(mesh.get_self_intersecting_triangles(),dtype=np.int64).reshape(-1,2)
    records=[];excluded=0
    for i,j in candidates:
        if np.intersect1d(f[i],f[j]).size:excluded+=1;continue
        records.append(dict(face_ids=[int(i),int(j)],classification=strict_crossing(q[f[i]],q[f[j]])))
    return dict(candidate_pairs=len(candidates),shared_vertex_pairs_excluded=excluded,pairs=records,
        proper_crossings=sum(r['classification']=='proper_noncoplanar_crossing' for r in records),
        uncertain_pairs=sum(r['classification'].endswith('uncertain') for r in records),
        continuous_time_checked=False,open3d_version=o3d.__version__)


def broadphase(points, vertices, radius=.008, tolerance=1e-6):
    """Keep triangle-surface overlap independent of vertex-sphere proximity."""
    from .contracts import finite_array
    q = finite_array(points, 'cloth points', (None, 3))
    v = finite_array(vertices, 'robot mesh', (None, 3))
    if not len(q) or not len(v) or not np.isfinite(radius) or radius <= 0 or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Nonempty geometry and explicit positive radius required')
    low, high = v.min(0), v.max(0)
    distances = np.linalg.norm(np.maximum(np.maximum(low - q, q - high), 0.), axis=1)
    nearby = np.flatnonzero(distances <= radius + tolerance)
    surface_possible = bool(np.all(high >= q.min(0)) and np.all(low <= q.max(0)))
    return nearby, surface_possible


def robot_meshes(model):
    """Actual visible AND collision meshes in body coordinates, not placeholders.

    Primitive shapes are not converted to meshes by this function. The native
    table is handled separately by support/footprint checks. The receipt lists
    each checked mesh and its actual collision flags rather than claiming full
    collision-model coverage.
    """
    from .actual_export import transform_points
    bodies, flags = model.shape_body.numpy(), model.shape_flags.numpy()
    scales, poses = model.shape_scale.numpy(), model.shape_transform.numpy()
    names = list(model.body_key)
    meshes = []
    for shape, body in enumerate(bodies):
        source = model.shape_source[shape]
        if body < 0 or not hasattr(source, 'vertices') or not hasattr(source, 'indices'):
            continue
        vertices = transform_points(np.asarray(source.vertices) * scales[shape], poses[shape])
        triangles = np.asarray(source.indices, dtype=np.int64).reshape(-1, 3)
        if not len(triangles) or triangles.min() < 0 or triangles.max() >= len(vertices):
            raise ValueError('Invalid actual robot mesh topology')
        meshes.append(dict(shape_id=int(shape), body_id=int(body), body_name=names[body],
            visible=bool(flags[shape] & 1), active_particle_collider=bool(flags[shape] & 4),
            vertices=vertices, triangles=triangles))
    if not meshes:
        raise ValueError('No actual robot meshes available')
    return meshes


def cross_mesh_pairs(a, fa, b, fb):
    """Conservatively prune triangle AABBs, preserving original face identities.

    Testing whole combined meshes redundantly searches all within-cloth and
    within-robot pairs. Only triangles whose bounds overlap the other mesh can
    participate in a cross-mesh hit. No surface-distance threshold is involved.
    """
    import open3d as o3d
    from .contracts import finite_array
    a = finite_array(a, 'mesh A', (None, 3)).astype(np.float64)
    b = finite_array(b, 'mesh B', (None, 3)).astype(np.float64)
    fa = finite_array(fa, 'faces A', (None, 3))
    fb = finite_array(fb, 'faces B', (None, 3))
    for v, f in ((a, fa), (b, fb)):
        if not len(f) or f.dtype.kind not in 'iu' or f.min() < 0 or f.max() >= len(v):
            raise ValueError('Nonempty in-range triangular mesh required')
    if np.any(a.max(0) < b.min(0)) or np.any(b.max(0) < a.min(0)):
        return []
    ta, tb = a[fa], b[fb]
    ma = np.all(ta.max(1) >= b.min(0), axis=1) & np.all(ta.min(1) <= b.max(0), axis=1)
    mb = np.all(tb.max(1) >= a.min(0), axis=1) & np.all(tb.min(1) <= a.max(0), axis=1)
    if not ma.any() or not mb.any():
        return []
    def compact(v, f, mask):
        ids, inverse = np.unique(f[mask], return_inverse=True)
        return v[ids], inverse.reshape(-1, 3), np.flatnonzero(mask)
    av, af, ai = compact(a, fa, ma); bv, bf, bi = compact(b, fb, mb)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.concatenate((av, bv))),
        o3d.utility.Vector3iVector(np.concatenate((af, bf + len(av)))))
    pairs = np.asarray(mesh.get_self_intersecting_triangles(), dtype=np.int64).reshape(-1, 2)
    rows = []
    for i, j in pairs:
        if (i < len(af)) == (j < len(af)):
            continue
        ia, ib = (ai[i], bi[j - len(af)]) if i < len(af) else (ai[j], bi[i - len(af)])
        rows.append(dict(robot_face=int(ia), cloth_face=int(ib), classification=strict_crossing(a[fa[ia]], b[fb[ib]])))
    return sorted(rows, key=lambda r: (r['robot_face'], r['cloth_face']))


def robot_clearance(points, faces, body_poses, meshes, *, radius=.008, tolerance=1e-6):
    """Discrete surface and sphere checks using actual recorded body poses.

    A tiny body can cross a large cloth face while staying far from every cloth
    vertex, so a clear vertex-distance query never skips an overlapping surface
    query. Signed whole-solid containment and continuous collision remain out of
    scope. Radius is the cloth particle radius, not self-contact distance.
    """
    import open3d as o3d
    from .actual_export import transform_points
    from .contracts import finite_array, unit_quaternions
    q = finite_array(points, 'cloth points', (None, 3)).astype(np.float64)
    f = finite_array(faces, 'cloth faces', (None, 3))
    b = finite_array(body_poses, 'actual body poses', (None, 7))
    unit_quaternions(b[:, 3:])
    if not meshes or not len(b):
        raise ValueError('Explicit nonempty mesh inventory and actual body states required')
    if not len(f) or f.dtype.kind not in 'iu' or f.min() < 0 or f.max() >= len(q):
        raise ValueError('Valid cloth triangle indices required')
    rows = []
    for item in meshes:
        bid = item['body_id']
        if type(bid) is not int or not 0 <= bid < len(b):
            raise ValueError('Robot mesh body outside the actual-state array')
        vertices = transform_points(item['vertices'], b[bid])
        triangles = finite_array(item['triangles'], 'robot triangles', (None, 3))
        if not len(triangles) or triangles.dtype.kind not in 'iu' or triangles.min() < 0 or triangles.max() >= len(vertices):
            raise ValueError('Valid robot mesh triangle indices required')
        candidates, surface_possible = broadphase(q, vertices, radius, tolerance)
        if not len(candidates) and not surface_possible:
            continue
        row = {key: item[key] for key in ('shape_id', 'body_id', 'body_name', 'visible', 'active_particle_collider')}
        row.update(crossing_pairs=[], near_particle_ids=[], minimum_candidate_surface_distance_m=None)
        if surface_possible:
            row['crossing_pairs'] = cross_mesh_pairs(vertices, triangles, q, f)
        if len(candidates):
            scene = o3d.t.geometry.RaycastingScene(nthreads=1)
            scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)),
                                o3d.core.Tensor(triangles.astype(np.uint32)))
            distances = scene.compute_distance(o3d.core.Tensor(q[candidates].astype(np.float32)), nthreads=1).numpy()
            row['minimum_candidate_surface_distance_m'] = float(distances.min())
            row['near_particle_ids'] = candidates[distances <= radius + tolerance].tolist()
        rows.append(row)
    return dict(shapes_tested=len(rows), shapes=rows,
        proper_crossings=sum(p['classification'] == 'proper_noncoplanar_crossing' for r in rows for p in r['crossing_pairs']),
        uncertain_pairs=sum(p['classification'].endswith('uncertain') for r in rows for p in r['crossing_pairs']),
        near_particle_shape_pairs=sum(len(r['near_particle_ids']) for r in rows),
        radius_m=radius, tolerance_m=tolerance, actual_body_poses_used=True,
        continuous_time_checked=False, whole_solid_containment_checked=False,
        geometry_scope='actual mesh shapes, including visible meshes with disabled or placeholder colliders',
        open3d_version=o3d.__version__)
