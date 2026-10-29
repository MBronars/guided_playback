"""Letter moulding: the previous paper's `letter_template` objective on this package's guidance path.

The target is a frozen letter footprint fitted once to the frame-0 blob (pwm_efficient/letter_template.py; needs scipy, so
the fit runs in the pctrl env: sweeps/letters/fit_fields.py -> sweeps/letters/fields/letter_<L>_seed<S>.npz, the old
letter_field.npz layout: coverage [M, 2] interior points, contour [240, 3], chosen_rot_deg, fit_*). The objective and the
paper metric are the same chamfer: cloud->template + coverage_weight * template->cloud, each the mean squared nearest-
neighbour distance in the plane between the dough particles and the coverage points; reported as its square root (RMS, m).
The paper number is that RMS on the LAST frame of the Genesis replay, in mm."""
import os
import numpy as np, torch

FIELDS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sweeps", "letters", "fields")


def rotation_for(letter, master_seed=42):
    """The old evaluation's per-letter rotation (Genesis convention): default_rng(SeedSequence([seed, ord(L)])).uniform(-40, 40)."""
    rng = np.random.default_rng(np.random.SeedSequence([master_seed] + [ord(c) for c in letter]))
    return float(rng.uniform(-40.0, 40.0))


def field_path(letter, seed=42, root=FIELDS):
    return os.path.join(root, f"letter_{letter}_seed{seed}.npz")


def chamfer_sq(pxy, cov, coverage_weight=1.0):
    """pxy [..., N, 2] dough points, cov [M, 2] -> [...] cloud->template + w * template->cloud (mean squared NN, m^2)."""
    d2 = ((pxy[..., :, None, :] - cov[..., None, :, :]) ** 2).sum(-1)          # [..., N, M]
    return d2.min(-1).values.mean(-1) + coverage_weight * d2.min(-2).values.mean(-1)


def chamfer_rms_np(xy, cov, coverage_weight=1.0):
    """numpy, one frame: RMS chamfer in metres (sqrt of the paper's sim_chamfer_m2)."""
    d2 = ((xy[:, None, :] - cov[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.min(1).mean() + coverage_weight * d2.min(0).mean()))


class LetterObjective:
    """objective = mean over the scored frames of the RMS chamfer (m) of the dough rows to the letter's coverage points
    + floor_weight * mean depth below z = 0. frames: 'all' (every generated frame, the old planner's choice) or an int k
    (the last k frames; k = 1 is the paper metric alone)."""
    def __init__(self, npz, obj_mask, valid, device, frames="all", coverage_weight=1.0, floor_weight=0.0, phantom_weight=0.0, contact_mm=20.0, contact_soft_mm=5.0):
        z = np.load(npz, allow_pickle=True)
        B = obj_mask.shape[0]
        self.letter = str(z["letter"]); self.chosen_rot_deg = float(z["chosen_rot_deg"]); self.npz = npz
        self.coverage = torch.tensor(np.asarray(z["coverage"], np.float32), device=device)          # [M, 2]
        self.contour = np.asarray(z["contour"], np.float32)
        self.obj_mask, self.valid = obj_mask, valid
        self.frames, self.coverage_weight, self.floor_weight = frames, float(coverage_weight), float(floor_weight)
        self.phantom_weight, self.contact_m, self.contact_soft_m = float(phantom_weight), float(contact_mm) / 1e3, float(contact_soft_mm) / 1e3
        self.target = torch.full((B, 3), float("nan"), device=device)                                # no centroid goal
        self.boxes = []
        self.extra = dict(letter_contour=np.repeat(self.contour[None], B, 0), letter_coverage=np.repeat(np.asarray(z["coverage"], np.float32)[None], B, 0),
                          letter=np.array(self.letter), chosen_rot_deg=np.array(self.chosen_rot_deg))
        same = (obj_mask == obj_mask[:1]).all()
        if not bool(same): raise SystemExit("LetterObjective: members must share the dough mask (one episode per population)")

    def frame_idx(self, T):
        if self.frames == "all": return list(range(1, T))
        k = int(self.frames); return list(range(max(1, T - k), T))

    def per_frame(self, pos):
        """[B, T] RMS chamfer (m) at every frame (no grad use: diagnostics, series)."""
        m = self.obj_mask[0]; pxy = pos[:, :, m, :2]
        return torch.sqrt(chamfer_sq(pxy, self.coverage, self.coverage_weight).clamp_min(1e-12))

    def phantom(self, pos):
        """Phantom motion (m): per frame, the dough-centroid displacement weighted by how far the nearest paddle point is from the
        nearest dough point beyond `contact_mm` (soft gate of width contact_soft_mm); summed over frames. Dough that moves while
        nothing touches it is what the model hallucinates under long optimization and what the simulator does not reproduce."""
        m = self.obj_mask[0]; pad = (~self.obj_mask[0]) & self.valid[0]
        c = pos[:, :, m].mean(2)                                                                     # [B, T, 3]
        disp = (c[:, 1:] - c[:, :-1]).norm(dim=-1)                                                   # [B, T-1]
        d = torch.cdist(pos[:, 1:, pad], pos[:, 1:, m]).amin(-1).amin(-1)                            # [B, T-1] nearest paddle-dough distance
        gate = torch.sigmoid((d - self.contact_m) / self.contact_soft_m)
        return (disp * gate).sum(1)

    def terms(self, pos):
        B, T = pos.shape[:2]; m = self.obj_mask[0]
        idx = self.frame_idx(T)
        pxy = pos[:, idx][:, :, m, :2]                                                               # [B, F, Nd, 2]
        goal = torch.sqrt(chamfer_sq(pxy, self.coverage, self.coverage_weight).clamp_min(1e-12)).mean(1)
        floor = pos.new_zeros(B)
        if self.floor_weight > 0:
            P = pos[:, 1:]; sel = self.valid[:, None].expand(P.shape[:3])
            floor = ((-P[..., 2]).clamp(min=0) * sel).sum((1, 2)) / sel.sum((1, 2))
        return goal, pos.new_zeros(B), floor

    def __call__(self, pos):
        goal, box, floor = self.terms(pos)
        total = goal + self.floor_weight * floor; d = dict(goal=goal, box=box, floor=floor)
        if self.phantom_weight > 0:
            ph = self.phantom(pos); total = total + self.phantom_weight * ph; d["phantom"] = ph
        return total, d
