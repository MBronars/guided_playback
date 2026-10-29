"""GT vs model rollout of one Water-3D window, same tank scene for both.

    python scripts/render_water3d/render_rollout.py --npz dump_w0.npz --out render_w0 [--spp 160] [--res 448] [--label model]
Dump keys: pred/gt [F, N, 3], context [C, N, 3] (raw box units, y up), e.g. from dump_overfit_windows.py. The density grid covers the
union of the tank and the predicted extent so escaping particles stay visible; the tank is drawn
on the metadata bounds. Output: per-frame PNG pairs, a side-by-side MP4 and a small GIF."""
import argparse, json, os, sys, time
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_water import BOUNDS, scene_dict, surface, write_ply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--spp", type=int, default=128); ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--grid", type=int, default=160); ap.add_argument("--kernel", type=float, default=1.0); ap.add_argument("--iso", type=float, default=0.5)
    ap.add_argument("--variant", default="cuda_ad_rgb")
    ap.add_argument("--label", default="model", help="right-panel caption")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import mitsuba as mi; mi.set_variant(a.variant)
    d = np.load(a.npz)
    ctx, gt, pred = d["context"], d["gt"], d["pred"]
    seq = {"gt": np.concatenate([ctx, gt]), "pred": np.concatenate([ctx, pred])}
    lo = np.minimum(BOUNDS[:, 0], pred.min(axis=(0, 1))) - 0.02; hi = np.maximum(BOUNDS[:, 1], pred.max(axis=(0, 1))) + 0.02
    grid_bounds = np.stack([lo, hi], 1)
    C = int(ctx.shape[0]); T = int(seq["gt"].shape[0]); rec = []
    for t in range(T):
        row = {"frame": t, "context": t < C}
        for k in ("gt", "pred"):
            pts = seq[k][t].astype(np.float64)
            inside = float(np.mean(np.all((pts > BOUNDS[:, 0]) & (pts < BOUNDS[:, 1]), axis=1)))
            verts, faces, normals, level, spacing = surface(pts, grid_bounds, a.grid, a.kernel, a.iso)
            ply = os.path.join(a.out, f"{k}_{t:03d}.ply"); write_ply(ply, verts, faces, normals)
            img = mi.render(mi.load_dict(scene_dict(ply, BOUNDS, a.res, a.spp)), spp=a.spp)
            mi.util.write_bitmap(os.path.join(a.out, f"{k}_{t:03d}.png"), img)
            row[k] = {"verts": int(len(verts)), "inside_frac": inside, "spacing": spacing}
        rec.append(row); print(row, flush=True)
    json.dump({"args": vars(a), "frames": rec}, open(os.path.join(a.out, "receipt.json"), "w"), indent=1)
    # side-by-side frames with labels
    frames = []
    for t in range(T):
        L = Image.open(os.path.join(a.out, f"gt_{t:03d}.png")).convert("RGB"); R = Image.open(os.path.join(a.out, f"pred_{t:03d}.png")).convert("RGB")
        w, h = L.size; im = Image.new("RGB", (2 * w + 8, h + 28), (250, 248, 244)); im.paste(L, (0, 28)); im.paste(R, (w + 8, 28))
        dr = ImageDraw.Draw(im); tag = "context" if t < C else f"predicted frame {t - C + 1}/{T - C}"
        dr.text((8, 7), f"ground truth   |   {tag}", fill=(40, 40, 40)); dr.text((w + 16, 7), a.label, fill=(40, 40, 40))
        frames.append(im); im.save(os.path.join(a.out, f"pair_{t:03d}.png"))
    import imageio
    wr = imageio.get_writer(os.path.join(a.out, "gt_vs_pred.mp4"), fps=8, codec="libx264", quality=8, macro_block_size=None)
    for f in frames: wr.append_data(np.asarray(f))
    wr.close()
    small = [f.resize((f.width * 300 // f.height, 300), Image.LANCZOS).quantize(colors=96, method=Image.Quantize.MEDIANCUT) for f in frames]
    small[0].save(os.path.join(a.out, "gt_vs_pred_small.gif"), save_all=True, append_images=small[1:], duration=125, loop=0, optimize=True)
    print("gif %.2f MB" % (os.path.getsize(os.path.join(a.out, "gt_vs_pred_small.gif")) / 1e6))


if __name__ == "__main__":
    main()
