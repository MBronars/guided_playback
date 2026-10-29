"""Asset sweep driver: load the clean_pickup adapter ONCE (eager bf16), scan the val set ONCE for every entry of
sweeps/assetsweep_select.json, then run the fixed lift recipe per entry -> runs/pickup/assetsweep_<label>/.
Skips entries whose run dir already has rollout.npz (never overwrites)."""
import dataclasses, json, os, sys, time, traceback
import torch
ROOT = "/data/user_data/mbronars/packages/guided_playback"
sys.path.insert(0, ROOT)
from gp.session import Args, DEFAULTS, load_adapter, guided_run
from gp.adapters import clean_pickup as CP

SELECT = os.path.join(ROOT, "sweeps/assetsweep_select.json")
CKPT = "/data/user_data/mbronars/packages/pwm_final/runs/pickup_uncond_x0_regpass_cont200k/checkpoints/final.pt"
only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
torch.backends.cuda.matmul.allow_tf32 = True
base = dict(DEFAULTS, adapter="clean_pickup", ckpt=CKPT, domain="pickup", task="lift", transitions=39, steps=10, offset="0.15,0,0.15",
            iterations=10, lr=0.1, lr_final=None, optimizer="adam", shell=True, seed=0, eager=True, runs=os.path.join(ROOT, "runs"), select=SELECT)
t0 = time.time(); ad = load_adapter(Args(base)); ad.select_path = SELECT
print(f"loaded in {time.time() - t0:.0f} s ({ad.mode})", flush=True)
sel = json.load(open(SELECT))

# one scan of the val set for all entries
sys.path.insert(0, os.path.join(CP.REPO, "guidance"))
from scripts.unified.eval_unified import evaluation_dataset
from streaming_guidance import select_and_group
C, F = int(ad.cfg.data.context_len), int(ad.cfg.data.future_len)
cwd = os.getcwd(); os.chdir(CP.REPO)
t1 = time.time()
try:
    ds = evaluation_dataset(ad.cfg, "pickup", C, F, roots_override={"pickup": CP.ROOTS})
    ds.emit_sidecar = True; ds.articulated = bool(getattr(ad.model.cfg, "articulated_bodies", False)); ds._pose_stats(); ds.future_len = 10_000
    groups = select_and_group(ds, sel, 1, torch.device("cpu"))
finally:
    os.chdir(cwd)
cache = {lab[0]: b for lab, b in groups}
print(f"val scan {time.time() - t1:.0f} s, found {len(cache)}/{len(sel)}", flush=True)


def episodes(indices, transitions, domain=None):
    ad.domain = "pickup"
    parts = [cache[sel[i]["label"]] for i in indices]
    batch = dataclasses.replace(parts[0], **{f.name: torch.cat([getattr(b, f.name) for b in parts], 0) for f in dataclasses.fields(parts[0])
                                              if torch.is_tensor(getattr(parts[0], f.name))})
    batch = batch.to(ad.dev)
    T = batch.positions.shape[1]
    if int(transitions) + 1 > T:
        from gp.adapters.paper_core import pad_frames
        batch, _ = pad_frames(batch, int(transitions))
    batch.gt_frames = [T] * len(indices); batch.episodes = [sel[i]["label"] for i in indices]
    ad.n_tr = int(transitions)
    return batch


ad.episodes = episodes
timing = {}
for i, e in enumerate(sel):
    lab = e["label"]
    if only and lab not in only: continue
    name = f"assetsweep_{lab}"; out = os.path.join(ROOT, "runs/pickup", name)
    if os.path.exists(os.path.join(out, "rollout.npz")):
        print(f"skip {name} (exists)", flush=True); continue
    T = cache[lab].positions.shape[1]
    A = Args(dict(base, episodes=",".join([str(i)] * 4), transitions=T - 1, name=name,
                  desc=f"asset sweep: {lab} (asset_id {e['asset_id']}), S=10, lift +15 cm x +15 cm z, Adam lr 0.1, 10 it, shell, eager bf16, pop 4"))
    t2 = time.time()
    try:
        guided_run(ad, A)
        timing[lab] = round(time.time() - t2, 1)
    except Exception:
        timing[lab] = "FAILED: " + traceback.format_exc().splitlines()[-1]
        traceback.print_exc()
    print(f"== {lab}: {timing[lab]}", flush=True)
    torch.cuda.empty_cache()
json.dump(timing, open(os.path.join(ROOT, "sweeps/assetsweep_tools/guidance_timing.json"), "a"))
print("done", timing, flush=True)
