"""Particles -> smoothed density -> marching cubes -> path-traced render (Figure-1 style).

    python scripts/render_water3d/render_water.py --npz trajectory.npz --frames 0,100,200,400,600,800 --out renders/ \
        [--grid 160] [--kernel 1.0] [--iso 0.5] [--spp 256] [--res 640] [--variant cuda_ad_rgb]

Input npz: positions [T, N, 3] in the raw unit-box coordinates (DeepMind Water-3D: bounds
[0.125, 0.875]^3, y up), e.g. np.load of one materialized trajectory. Needs the render environment
(env/render_water3d.requirements.txt; Mitsuba's cuda variant needs a GPU, scalar_rgb does not). The scene: a glass tank matching the bounds, a dark floor slab, a
warm backdrop, one large area light, water as a dielectric with a faint blue tint.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage import measure

BOUNDS = np.array([[0.125, 0.875]] * 3)      # metadata.json of Water-3D; overridden by --bounds


def density_grid(pts: np.ndarray, bounds: np.ndarray, grid: int, kernel_cells: float, margin_cells: int = 4):
    """Trilinear splat of the particles into a grid over the bounds (+margin), then a
    Gaussian blur of ``kernel_cells`` cells. Returns (field, origin, cell)."""
    lo, hi = bounds[:, 0], bounds[:, 1]
    cell = float((hi - lo).max()) / grid
    origin = lo - margin_cells * cell
    shape = np.ceil((hi - lo) / cell).astype(int) + 2 * margin_cells + 1
    field = np.zeros(shape, dtype=np.float32)
    g = (pts - origin) / cell
    i0 = np.floor(g).astype(int)
    f = g - i0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                idx = i0 + np.array([dx, dy, dz])
                ok = np.all((idx >= 0) & (idx < shape), axis=1)
                np.add.at(field, (idx[ok, 0], idx[ok, 1], idx[ok, 2]), w[ok])
    field = gaussian_filter(field, sigma=kernel_cells)
    return field, origin, cell


def particle_spacing(pts: np.ndarray) -> float:
    """Mean nearest-neighbour distance (the SPH particle spacing)."""
    d, _ = cKDTree(pts).query(pts, k=2)
    return float(d[:, 1].mean())


def surface(pts: np.ndarray, bounds: np.ndarray, grid: int, kernel_spacing: float, iso_frac: float):
    """Marching cubes on the smoothed density. The Gaussian sigma is ``kernel_spacing``
    times the measured particle spacing (so the surface is one particle layer smooth
    regardless of resolution); the iso level is ``iso_frac`` of the 60th-percentile
    interior density (robust to particle count). Returns verts, faces, normals, level, spacing."""
    spacing = particle_spacing(pts)
    cell = float((bounds[:, 1] - bounds[:, 0]).max()) / grid
    kernel_cells = kernel_spacing * spacing / cell
    field, origin, cell = density_grid(pts, bounds, grid, kernel_cells)
    interior = field[field > 0]
    level = float(np.percentile(interior, 60)) * iso_frac
    verts, faces, normals, _ = measure.marching_cubes(field, level=level)
    verts = verts * cell + origin
    return verts, faces, normals, level, spacing


def write_ply(path: str, verts: np.ndarray, faces: np.ndarray, normals: np.ndarray | None = None):
    with open(path, "wb") as fh:
        head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(verts)}",
                "property float x", "property float y", "property float z"]
        if normals is not None:
            head += ["property float nx", "property float ny", "property float nz"]
        head += [f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header"]
        fh.write(("\n".join(head) + "\n").encode())
        v = verts.astype("<f4") if normals is None else np.hstack([verts, normals]).astype("<f4")
        fh.write(v.tobytes())
        f = np.hstack([np.full((len(faces), 1), 3, dtype=np.uint8).view("u1"), faces.astype("<i4").view("u1").reshape(len(faces), -1)])
        fh.write(f.tobytes())


def scene_dict(ply_path: str, bounds: np.ndarray, res: int, spp: int, up: str = "y"):
    """Figure-1-style tank: glass panes on the five closed sides, dark floor slab, warm
    backdrop, big soft area light, 3/4 elevated camera. ``up`` names the gravity axis of
    the data; the scene is built in a y-up frame and the water is rotated if needed."""
    import mitsuba as mi
    lo, hi = bounds[:, 0], bounds[:, 1]
    c = (lo + hi) / 2
    size = hi - lo
    if up == "z":                                   # rotate data (z up) into the y-up render frame
        to_world_water = mi.ScalarTransform4f().rotate([1, 0, 0], -90)
        c = np.array([c[0], c[2], -c[1]]); size = np.array([size[0], size[2], size[1]])
        lo = c - size / 2; hi = c + size / 2
    else:
        to_world_water = mi.ScalarTransform4f()
    T = mi.ScalarTransform4f
    pane = 0.004
    def slab(center, scale, bsdf):
        return {"type": "cube", "to_world": T().translate(list(center)).scale(list(scale)), "bsdf": bsdf}
    glass = {"type": "thindielectric", "int_ior": 1.5}
    dark = {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.16, 0.16, 0.17]}}
    backdrop = {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.93, 0.86, 0.80]}}
    water = {"type": "dielectric", "int_ior": "water", "ext_ior": "air",
             "specular_transmittance": {"type": "rgb", "value": [0.80, 0.92, 1.0]}}
    cam_target = [c[0], lo[1] + 0.42 * size[1], c[2]]
    cam_origin = [c[0] + 1.45 * size[0], hi[1] + 1.05 * size[1], c[2] + 1.95 * size[2]]
    sc = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 24, "hide_emitters": True},
        "sensor": {"type": "perspective", "fov": 34, "fov_axis": "smaller",
                   "to_world": T().look_at(origin=cam_origin, target=cam_target, up=[0, 1, 0]),
                   "film": {"type": "hdrfilm", "width": res, "height": res, "rfilter": {"type": "gaussian"}, "pixel_format": "rgb"},
                   "sampler": {"type": "independent", "sample_count": spp}},
        "env": {"type": "constant", "radiance": {"type": "rgb", "value": [0.55, 0.52, 0.50]}},
        "key": {"type": "rectangle", "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [9.0, 8.6, 8.0]}},
                "to_world": T().look_at(origin=[c[0] - 0.6 * size[0], hi[1] + 1.6 * size[1], c[2] + 0.4 * size[2]],
                                        target=list(c), up=[0, 0, 1]).scale([0.9 * size[0], 0.9 * size[2], 1])},
        "floor": slab([c[0], lo[1] - 0.012 - 0.5 * pane, c[2]], [0.62 * size[0], 0.012, 0.62 * size[2]], dark),
        "ground": {"type": "rectangle", "to_world": T().translate([c[0], lo[1] - 0.025, c[2]]).rotate([1, 0, 0], -90).scale(8.0), "bsdf": backdrop},
        "back": {"type": "rectangle", "to_world": T().translate([c[0], c[1], lo[2] - 2.5 * size[2]]).scale(8.0), "bsdf": backdrop},
        "water": {"type": "ply", "filename": ply_path, "to_world": to_world_water, "bsdf": water},
    }
    hx, hy, hz = size / 2
    sc["pane_bottom"] = slab([c[0], lo[1] - pane / 2, c[2]], [hx + pane, pane / 2, hz + pane], glass)
    sc["pane_back"] = slab([c[0], c[1], lo[2] - pane / 2], [hx + pane, hy, pane / 2], glass)
    sc["pane_front"] = slab([c[0], c[1], hi[2] + pane / 2], [hx + pane, hy, pane / 2], glass)
    sc["pane_left"] = slab([lo[0] - pane / 2, c[1], c[2]], [pane / 2, hy, hz], glass)
    sc["pane_right"] = slab([hi[0] + pane / 2, c[1], c[2]], [pane / 2, hy, hz], glass)
    # thin dark frame along the 12 tank edges so the glass reads as a tank
    edge = {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.25, 0.27, 0.24]}}
    e = 0.0045
    xs, ys, zs = (lo[0] - pane, hi[0] + pane), (lo[1] - pane, hi[1]), (lo[2] - pane, hi[2] + pane)
    k = 0
    for y in ys:
        for z in zs:
            sc[f"edge{k}"] = slab([c[0], y, z], [hx + pane, e, e], edge); k += 1
        for x in xs:
            sc[f"edge{k}"] = slab([x, y, c[2]], [e, e, hz + pane], edge); k += 1
    for x in xs:
        for z in zs:
            sc[f"edge{k}"] = slab([x, c[1], z], [e, hy, e], edge); k += 1
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="one trajectory: a materialized .npy [T, N, 3], or an npz with positions")
    ap.add_argument("--frames", default="0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=160)
    ap.add_argument("--kernel", type=float, default=1.0, help="Gaussian sigma in multiples of the particle spacing")
    ap.add_argument("--iso", type=float, default=0.5, help="iso level as a fraction of the 60th-percentile interior density")
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--res", type=int, default=640)
    ap.add_argument("--variant", default="cuda_ad_rgb")
    ap.add_argument("--up", default="y", choices=["y", "z"])
    ap.add_argument("--bounds", default=None, help="json [[lo,hi],[lo,hi],[lo,hi]]; default Water-3D metadata")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import mitsuba as mi
    mi.set_variant(a.variant)
    d = np.load(a.npz)
    P = d if isinstance(d, np.ndarray) else d["positions"]     # a materialized .npy, or an npz with positions
    bounds = np.array(json.loads(a.bounds)) if a.bounds else BOUNDS
    frames = [int(f) for f in a.frames.split(",")]
    receipts = []
    for fi in frames:
        t0 = time.time()
        pts = P[fi].astype(np.float64)
        verts, faces, normals, level, spacing = surface(pts, bounds, a.grid, a.kernel, a.iso)
        ply = os.path.join(a.out, f"water_{fi:04d}.ply")
        write_ply(ply, verts, faces, normals)
        t1 = time.time()
        img = mi.render(mi.load_dict(scene_dict(ply, bounds, a.res, a.spp, up=a.up)), spp=a.spp)
        png = os.path.join(a.out, f"water_{fi:04d}.png")
        mi.util.write_bitmap(png, img)
        t2 = time.time()
        receipts.append({"frame": fi, "n_particles": int(len(pts)), "verts": int(len(verts)), "faces": int(len(faces)),
                         "iso_level": level, "particle_spacing": round(spacing, 5), "mesh_s": round(t1 - t0, 2), "render_s": round(t2 - t1, 2)})
        print(receipts[-1], flush=True)
    json.dump({"args": vars(a), "frames": receipts}, open(os.path.join(a.out, "receipt.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
