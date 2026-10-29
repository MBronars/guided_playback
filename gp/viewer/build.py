"""Style-1 page: one 3D viewer, domain tabs -> run list (newest first) -> member tabs; layers = generated (guided),
unguided, simulated (when a replay exists), goal, wall boxes, table. Per-run table of the replay analysis.

Data: one js file per (run, member) with int16-quantized clouds, loaded on demand. Camera framed on the guided rollout."""
import base64, glob, json, os, shutil, time
import numpy as np

DOMAIN_LABEL = {"pickup": "pick & place", "rl_push": "pushing (rl_push)", "mould": "moulding", "pushing": "block pushing", "yikai_small": "yikai"}
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def q16(x, lo, span):
    u = (x - lo) / np.where(span > 0, span, 1.0)
    return np.clip(np.rint(u * 65534 - 32767), -32767, 32767).astype(np.int16)


def shrink_gif(src, dst, scale=0.62, colors=96):
    """Web copy of a render gif: scaled, palette-reduced (the run dir keeps the original)."""
    from PIL import Image, ImageSequence
    im = Image.open(src); frames = []
    for fr in ImageSequence.Iterator(im):
        f = fr.convert("RGB"); f = f.resize((int(f.width * scale), int(f.height * scale)), Image.LANCZOS)
        frames.append(f.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    frames[0].save(dst, save_all=True, append_images=frames[1:], duration=im.info.get("duration", 100), loop=0, optimize=True)


def scan(runs_root):
    out = {}
    for rj_path in sorted(glob.glob(os.path.join(runs_root, "*", "*", "result.json"))):
        run = os.path.dirname(rj_path); rj = json.load(open(rj_path)); dom = rj.get("domain") or os.path.basename(os.path.dirname(run))
        replays = {os.path.basename(d)[len("replay_"):]: d for d in glob.glob(os.path.join(run, "replay_*")) if os.path.exists(os.path.join(d, "replay.npz"))}
        out.setdefault(dom, []).append(dict(dir=run, name=os.path.basename(run), rj=rj, replays=replays, mtime=os.path.getmtime(rj_path)))
    for dom in out:
        out[dom].sort(key=lambda r: r["rj"].get("created", "") or time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["mtime"])), reverse=True)
    return out


def build(runs_root, out, title="Guided Playback", subtitle="", max_runs=6):
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    doms = scan(runs_root); index = []
    for dom, runs in doms.items():
        runs = runs[:max_runs]                                   # newest first; the page stays short
        dentry = dict(id=dom, label=DOMAIN_LABEL.get(dom, dom), runs=[])
        for r in runs:
            z = np.load(os.path.join(r["dir"], "rollout.npz")); rj = r["rj"]
            B, T, N = z["predicted"].shape[:3]
            sims = {}
            for sname, sdir in r["replays"].items():
                sims[sname] = (np.load(os.path.join(sdir, "replay.npz")), json.load(open(os.path.join(sdir, "analysis.json"))))
                for row in sims[sname][1]["rows"]:                                                   # letter task: the page's "to goal" columns show the chamfer (mm)
                    if "sim_chamfer_mm" in row: row["sim_to_target_mm"] = row["sim_chamfer_mm"]; row["gen_to_target_mm"] = row["gen_chamfer_mm"]
            members = []
            only = int(np.argmin(rj["best_mm"])) if os.environ.get("GP_PAGE_ONLY_BEST") else None   # planner view: the member with the lowest generated objective
            for m in range(B):
                if only is not None and m != only:
                    continue
                key = f"{dom}__{r['name']}__m{m}".replace("-", "_").replace(".", "_")
                G_, U = z["predicted"][m], z["unguided"][m]; tgt = z["target"][m]; boxes = z["boxes"]
                cov = z["letter_coverage"][m] if "letter_coverage" in z.files else None          # letter task: no centroid goal, a footprint instead
                pts = [G_.reshape(-1, 3)] + ([] if np.isnan(tgt).any() else [tgt[None]]) + ([z["letter_contour"][m]] if cov is not None else []) + ([boxes[:, :3] - boxes[:, 3:] / 2, boxes[:, :3] + boxes[:, 3:] / 2] if len(boxes) else [])
                allp = np.concatenate(pts); lo, hi = allp.min(0), allp.max(0); lo[2] = min(lo[2], 0); span = hi - lo
                stack = [G_.reshape(-1, 3), U.reshape(-1, 3)] + [S[0]["sim"][:, m].reshape(-1, 3) for S in sims.values()]
                box = np.concatenate(stack); qlo = np.minimum(lo, np.maximum(box.min(0), lo - 0.6)); qhi = np.maximum(hi, np.minimum(box.max(0), hi + 0.6)); qspan = qhi - qlo
                rec = dict(id=key, member=m, episode=str(z["episodes"][m]), T=int(T), N=int(N), qlo=qlo.round(4).tolist(), qspan=qspan.round(4).tolist(), lo=lo.round(4).tolist(), span=span.round(4).tolist(),
                           roles=b64(z["roles"][m].astype(np.int8)), gen=b64(q16(G_, qlo, qspan)), ung=b64(q16(U, qlo, qspan)), target=tgt.round(4).tolist(), boxes=boxes.round(4).tolist(),
                           goal_mm=dict(guided=rj["best_mm"][m], unguided=rj["unguided_mm"][m]), sims={},
                           contour=(z["letter_contour"][m].round(4).tolist() if "letter_contour" in z.files else None))
                obj = z["object_mask"][m]
                if cov is not None:                                                                   # goal series = per-frame RMS chamfer to the letter (mm)
                    from ..letter import chamfer_rms_np
                    goal_of = lambda X: np.array([chamfer_rms_np(X[t, obj, :2], cov) for t in range(X.shape[0])]) * 1e3
                else:
                    goal_of = lambda X: np.linalg.norm(X[:, obj].mean(1) - tgt, axis=-1) * 1e3
                series = dict(gen_goal=goal_of(G_).round(1).tolist(), ung_goal=goal_of(U).round(1).tolist())
                for sname, (S, A) in sims.items():
                    variants = json.loads(str(S["variants"]))
                    layer = {}
                    for v, var in enumerate(variants):
                        if var != "guided" and os.environ.get("GP_PAGE_GUIDED_SIM_ONLY"):
                            continue                                              # page size: only the guided replay layer
                        sim = np.clip(S["sim"][v, m], qlo, qhi)
                        layer[var] = b64(q16(sim, qlo, qspan))
                        so = sim[:, obj].mean(1); series[f"sim_{var}_goal"] = goal_of(sim).round(1).tolist()
                        go = (G_ if var == "guided" else U)[:, obj].mean(1); series[f"sim_{var}_vs_gen"] = (np.linalg.norm(so - go, axis=-1) * 1e3).round(1).tolist()
                    rec["sims"][sname] = dict(layers=layer, rows=[row for row in A["rows"] if row["member"] == m], label=A.get("sim", sname))
                rec["series"] = series
                rec["renders"] = {}
                for rd in glob.glob(os.path.join(r["dir"], "render_*", "renders.json")):
                    RJ = json.load(open(rd)); style = "arm" if RJ.get("arm") else "hand"
                    for c in RJ["cases"]:
                        if c["member"] == m and os.path.exists(c["gif"]):
                            dst = f"render/{key}_{c['variant']}_{style}.gif"; os.makedirs(os.path.join(out, "render"), exist_ok=True)
                            shrink_gif(c["gif"], os.path.join(out, dst))
                            rec["renders"][f"{c['variant']}_{style}"] = dict(src=dst, sim=RJ["sim"], frames=c["frames"], size=c["size"], style=style, variant=c["variant"],
                                                                            arm_track_mm_max=c.get("arm_track_mm_max"), arm_track_deg_max=c.get("arm_track_deg_max"), scale=c.get("scale", 1.0))
                with open(os.path.join(out, "data", f"case_{key}.js"), "w") as f: f.write(f"window.CASE_{key} = {json.dumps(rec)};")
                simrow = next((row for S_, A_ in sims.values() for row in A_["rows"] if row["member"] == m and row["variant"] == "guided"), None)
                members.append(dict(id=key, label=f"m{m}", gen_goal=round(float(np.linalg.norm(G_[-1, obj].mean(0) - tgt) * 1e3), 1),
                                    sim_goal=(round(float(simrow["sim_to_target_mm"]), 1) if simrow and simrow.get("sim_to_target_mm") is not None else None)))
            knobs = {k: rj.get(k) for k in ("task", "transitions", "steps", "keep", "churn", "t_freeze", "fix_fps", "iterations", "lr", "lr_final", "optimizer", "shell", "joint_limits_mm", "close_fingers", "hand_start", "offset", "random_offset", "own_goal", "seed", "mode", "model_calls") if rj.get(k) not in (None, False, "")}
            wall = f"Wall {round(float(rj.get('wall_height', 0)) * 100)} cm" if rj.get("task") == "wall" else "No wall"
            short = f"shell {'on' if rj.get('shell') else 'off'} · lr {rj.get('lr')} · {rj.get('iterations')} it · {rj.get('transitions')} frames"
            if os.environ.get("GP_PAGE_LABEL") == "cyl":                         # random-target picks: group = asset, label = target id + offset
                off = [float(v) for v in str(rj.get("offset", "0,0,0")).split(",")]
                wall = str((rj.get("episodes") or ["?"])[0]).replace("_asset_", " #")
                tid = r["name"].split("_")[1]
                short = f"{tid}: dx {off[0]*100:+.1f} cm, dy {off[1]*100:+.1f} cm, dz +{off[2]*100:.0f} cm"
            if os.environ.get("GP_PAGE_LABEL") == "target":                      # group by target / wall, label = asset
                off = [float(v) for v in str(rj.get("offset", "0,0,0")).split(",")]
                wall = f"Goal +{off[0]*100:.0f} cm x" + (f", +{off[2]*100:.0f} cm z" if off[2] else " (on the table)") + (f" · {round(float(rj.get('wall_height', 0))*100)} cm wall" if rj.get("task") == "wall" else " · no wall")
                short = str((rj.get("episodes") or [r["name"]])[0]).replace("_asset_", " #")
            if os.environ.get("GP_PAGE_LABEL") == "asset":                       # asset pages: one run per asset, label = the asset
                wall = os.environ.get("GP_PAGE_GROUP", wall); short = str((rj.get("episodes") or [r["name"]])[0]).replace("_asset_", " #")
            if os.environ.get("GP_PAGE_LABEL") == "ckpt":                        # variant pages: group = checkpoint, label = the run's settings
                wall = "two-domain 120k" if "two120k" in r["name"] else "pickup-only"
                short = f"{rj.get('transitions')} fr · {rj.get('iterations')} it · lr {rj.get('lr')} · shell {'on' if rj.get('shell') else 'off'} · wall w {rj.get('box_weight')} / {rj.get('clearance_mm'):.0f} mm"
            if os.environ.get("GP_PAGE_LABEL") == "method":                      # planner comparison: group = asset + target, label = planner + plan time
                tid = [t for t in r["name"].split("_") if t[:1] == "t" and t[1:].isdigit()]
                wall = str((rj.get("episodes") or ["?"])[0]).replace("_asset_", " #") + (f" · {tid[0]}" if tid else "")
                secs = sum(rj.get("iteration_seconds") or [0])
                short = (f"MPPI argmin · K {rj.get('K')} · commit {rj.get('commit')} · {secs:.0f} s" if rj.get("method") == "mppi_argmin"
                         else f"guided · {rj.get('iterations')} it · pop {len(rj.get('episodes') or [])} · {secs:.0f} s")
            if rj.get("method") == "mppi_argmin":                                 # no-guidance planner: its own knobs, not the guidance defaults
                knobs = {k: rj.get(k) for k in ("method", "K", "commit", "task", "transitions", "steps", "offset", "seed", "mode")}
            dentry["runs"].append(dict(group=wall, short=short, name=r["name"], desc=rj.get("desc", ""), created=rj.get("created", ""), step=rj.get("step"), ckpt=os.path.basename(rj.get("ckpt", "")), run_dir=rj.get("ckpt", "").split("/")[-3] if rj.get("ckpt") else "",
                                       sampler=rj.get("sampler", ""), knobs=knobs, iteration_seconds=rj.get("iteration_seconds"), peak_gib=rj.get("peak_gib"), members=members,
                                       best_mm=rj["best_mm"], unguided_mm=rj["unguided_mm"], sims={s: A["rows"] for s, (S, A) in sims.items()}))
        dentry["runs"].sort(key=lambda x: (x["group"] != "No wall", x["group"], x["short"]))
        index.append(dentry)
    html = TEMPLATE.replace("__TITLE__", title).replace("__SUBTITLE__", subtitle).replace("__INDEX__", json.dumps(index))
    with open(os.path.join(out, "index.html"), "w") as f: f.write(html)
    n_cases = sum(len(r["members"]) for d in index for r in d["runs"])
    print("wrote", os.path.join(out, "index.html"), "domains", [(d["id"], len(d["runs"])) for d in index], "cases", n_cases)
    return index


TEMPLATE = r"""<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>
  :root { --plane:#f9f9f7; --surface:#fcfcfb; --stage:#f2f1ed; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --rule:#e4e3de; --rule2:#efeeea;
    --s1:#2a78d6; --s2:#eb6834; --fingerc:#7a4fd1; --u1:#9db7d6; --u2:#f0b39a; --simg:#1b7a3a; --simo:#b8860b; --goal:#1baf7a; --wall:#8a8a86;
    --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace; --sans:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif; --cond:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --plane:#0d0d0d; --surface:#1a1a19; --stage:#151514; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781; --rule:#2c2c29; --rule2:#212120; --s1:#3987e5; --s2:#d95926; --fingerc:#9b7be6; --u1:#3a5570; --u2:#7a4a35; --simg:#3ec26a; --simo:#e0a52a; --goal:#22c184; --wall:#6b6b67; } }
  :root[data-theme="dark"] { --plane:#0d0d0d; --surface:#1a1a19; --stage:#151514; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781; --rule:#2c2c29; --rule2:#212120; --s1:#3987e5; --s2:#d95926; --fingerc:#9b7be6; --u1:#3a5570; --u2:#7a4a35; --simg:#3ec26a; --simo:#e0a52a; --goal:#22c184; --wall:#6b6b67; }
  body { background:var(--plane); color:var(--ink); font-family:var(--sans); line-height:1.55; margin:0; }
  .wrap { width:min(1080px,100%); margin-inline:auto; padding-inline:16px; padding-block:28px 56px; }
  h1 { font-family:var(--cond); font-weight:600; font-size:clamp(26px,5vw,38px); line-height:1.1; margin:0 0 6px; letter-spacing:-0.01em; }
  h2 { font-family:var(--cond); font-weight:600; font-size:20px; margin:0 0 4px; }
  .sub { color:var(--ink2); margin:0 0 18px; max-width:70ch; } .eyebrow { font-family:var(--mono); font-size:11px; letter-spacing:.09em; text-transform:uppercase; color:var(--muted); margin:0 0 8px; }
  .method { border:1px solid var(--rule); border-radius:5px; padding:12px 16px; font-size:14px; color:var(--ink2); margin-bottom:22px; } .method code { font-family:var(--mono); font-size:12.5px; color:var(--ink); }
  .dtabs { display:flex; flex-wrap:wrap; gap:0; border-bottom:2px solid var(--rule); margin-top:10px; }
  .dtabs button { font-family:var(--cond); font-size:16px; font-weight:600; color:var(--ink2); background:none; border:0; padding:10px 18px; cursor:pointer; margin-bottom:-2px; border-bottom:2px solid transparent; }
  .dtabs button[aria-selected="true"] { color:var(--ink); border-bottom-color:var(--s1); }
  .layout { display:grid; grid-template-columns: 260px 1fr; gap:18px; margin-top:16px; } @media (max-width: 760px) { .layout { grid-template-columns: 1fr; } }
  .runs { display:flex; flex-direction:column; gap:6px; max-height:820px; overflow:auto; } .rgroup { font-family:var(--cond); font-weight:600; font-size:15px; color:var(--ink); margin:10px 0 2px; }
  .runs button { text-align:left; font-family:var(--mono); font-size:11.5px; color:var(--ink2); background:var(--surface); border:1px solid var(--rule); border-radius:4px; padding:8px 10px; cursor:pointer; line-height:1.4; }
  .runs button[aria-selected="true"] { color:var(--ink); border-color:var(--s1); box-shadow:inset 3px 0 0 var(--s1); } .runs b { color:var(--ink); font-weight:500; display:block; } .runs small { color:var(--muted); display:block; }
  .viewer { border:1px solid var(--rule); border-radius:5px; overflow:hidden; background:var(--surface); }
  .tabs { display:flex; flex-wrap:wrap; border-bottom:1px solid var(--rule); } .tabs button { font-family:var(--mono); font-size:12px; color:var(--ink2); background:none; border:0; border-right:1px solid var(--rule); padding:9px 14px; cursor:pointer; }
  .tabs button[aria-selected="true"] { color:var(--ink); background:var(--rule2); box-shadow:inset 0 -2px 0 var(--s1); }
  .tabs button.best-both { background:color-mix(in srgb, var(--goal) 28%, transparent); color:var(--ink); } .tabs button.best-gen { box-shadow:inset 0 0 0 2px var(--s1); } .tabs button.best-sim { box-shadow:inset 0 0 0 2px var(--simo); } .tabs button small { color:var(--muted); } .tabkey { font-family:var(--mono); font-size:11px; color:var(--muted); padding:9px 12px; }
  button:focus-visible, input:focus-visible { outline:2px solid var(--s1); outline-offset:1px; }
  .stage { position:relative; background:var(--stage); aspect-ratio:16/10; } .stage canvas { position:absolute; inset:0; width:100%; height:100%; display:block; touch-action:none; cursor:grab; }
  .hud { position:absolute; left:12px; top:10px; font-family:var(--mono); font-size:11.5px; color:var(--ink2); pointer-events:none; max-width:90%; }
  .transport { display:flex; align-items:center; gap:10px; padding:9px 12px; border-top:1px solid var(--rule); } .transport .play { font-size:12px; width:30px; height:26px; border:1px solid var(--rule); background:var(--plane); color:var(--ink); border-radius:3px; cursor:pointer; }
  .transport .seek { flex:1; min-width:80px; } .fnum { font-family:var(--mono); font-size:11.5px; color:var(--ink2); white-space:nowrap; font-variant-numeric:tabular-nums; }
  .opts { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; padding:0 12px 11px; font-family:var(--mono); font-size:11.5px; color:var(--ink2); } .opts label { display:flex; align-items:center; gap:5px; }
  .opts .sz input { width:76px; } .opts .reset { font-family:var(--mono); font-size:11px; border:1px solid var(--rule); background:var(--plane); color:var(--ink2); border-radius:3px; padding:3px 9px; cursor:pointer; }
  .mini { display:block; width:100%; height:auto; border-top:1px solid var(--rule); } .meta { font-family:var(--mono); font-size:11.5px; color:var(--muted); padding:8px 12px 11px; border-top:1px solid var(--rule2); word-break:break-word; }
  .key { display:flex; flex-wrap:wrap; gap:6px 18px; font-family:var(--mono); font-size:12px; color:var(--ink2); margin-top:12px; } .swatch { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:6px; }
  .rundesc { font-family:var(--mono); font-size:12px; color:var(--ink2); margin:0 0 10px; word-break:break-word; } .rundesc b { color:var(--ink); font-weight:500; }
  .scroller { overflow-x:auto; margin-top:14px; } table { border-collapse:collapse; width:100%; font-size:13px; min-width:640px; } th, td { text-align:left; padding:7px 9px; border-bottom:1px solid var(--rule2); }
  thead th { font-family:var(--mono); font-size:10.5px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); font-weight:400; border-bottom:1px solid var(--rule); } tbody th { font-weight:400; } .num { text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; } .strong { font-weight:500; }
  .empty { color:var(--muted); font-family:var(--mono); font-size:12px; padding:20px; }
  .render { margin-top:14px; } .render img { display:block; width:100%; height:auto; border:1px solid var(--rule); border-radius:5px; background:#fff; margin-bottom:10px; } .rlabel { font-family:var(--mono); font-size:11.5px; color:var(--ink2); margin:0 0 4px; }
</style><div class="wrap">
  <p class="eyebrow">guided rollouts &middot; simulator playback</p>
  <h1>__TITLE__</h1>
  <p class="sub">__SUBTITLE__ Drag to orbit, scroll to zoom, shift-drag to pan. Pick a domain, a run (newest first), then a member; layers toggle below the viewer.</p>
  <div class="method"><b>Method.</b> Streaming rollout of N transitions from frame 0, every action hidden. Full churn (η = 1): each active row is re-formed from the model's x<sub>0</sub> with a pre-decided draw <code>x = x0 + l (E[row, i] − x0)</code>; the rollout is a deterministic function of the draw bank E. Guidance: <code>J = ‖c_obj(x_N) − target‖ + w_box · clearance shortfall + w_floor · depth</code>, gradient through every model call (activation-checkpointed), update of the free draws (level ≥ t_freeze) by Adam or a unit-gradient step, best (member, iteration) kept; iteration 0 = unguided. Simulated layer = the generated actuator trajectory played back in the domain's simulator (Isaac floating hand / Genesis MPM / SAPIEN PushBoundary), the object left to physics.</div>
  <div class="key"><span><i class="swatch" style="background:var(--s1)"></i>generated actuator</span><span><i class="swatch" style="background:var(--fingerc)"></i>generated fingers</span><span><i class="swatch" style="background:var(--s2)"></i>generated object</span><span><i class="swatch" style="background:var(--u1)"></i>unguided actuator</span><span><i class="swatch" style="background:var(--u2)"></i>unguided object</span>
    <span><i class="swatch" style="background:var(--simg)"></i>simulated actuator</span><span><i class="swatch" style="background:var(--simo)"></i>simulated object</span><span><i class="swatch" style="background:var(--goal)"></i>goal</span><span><i class="swatch" style="background:var(--wall)"></i>wall</span></div>
  <div class="dtabs" role="tablist" aria-label="domains" id="dtabs"></div>
  <div class="layout"><div class="runs" id="runs" role="tablist" aria-label="runs"></div>
    <div><p class="rundesc" id="rundesc"></p>
      <div class="viewer" id="viewer"><div class="tabs" role="tablist" aria-label="members" id="mtabs"></div>
        <div class="stage"><canvas></canvas><div class="hud"><span class="now"></span></div></div>
        <div class="transport"><button class="play" type="button" aria-label="Play or pause">&#9654;</button><input class="seek" type="range" min="0" max="60" value="0" step="1" aria-label="Frame"><span class="fnum"></span></div>
        <div class="opts"><label><input type="checkbox" class="t-gen" checked> guided</label><label><input type="checkbox" class="t-ung"> unguided</label><label><input type="checkbox" class="t-sim" checked> simulated: <select class="t-simvar"><option value="guided">of guided</option><option value="unguided">of unguided</option></select></label>
          <label><input type="checkbox" class="t-floor" checked> table</label><label class="sz">points <input type="range" class="t-size" min="4" max="26" value="12" aria-label="Point size"></label><button class="reset" type="button">reset view</button></div>
        <svg class="mini" viewBox="0 0 1000 150" role="img" aria-label="object centre to goal per frame"></svg><div class="meta"></div></div>
      <div class="render" id="render"><p class="eyebrow">rendered frames &middot; left generated, right simulated, same camera</p>
        <p class="rlabel" id="renderlabel1"></p><img id="renderimg" alt="" hidden><p class="rlabel" id="renderlabel2"></p><img id="renderimg2" alt="" hidden><p class="empty" id="rendernote">no rendered frames for this case yet</p></div>
      <div id="table"></div></div></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const INDEX = __INDEX__;
const CACHE = {}; const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const unb64 = s => { const b = atob(s); const u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u; };
const I16 = s => new Int16Array(unb64(s).buffer);
function loadCase(key) { if (CACHE[key]) return Promise.resolve(CACHE[key]);
  return new Promise((res, rej) => { const s = document.createElement('script'); s.src = 'data/case_' + key + '.js';
    s.onload = () => { const raw = window['CASE_' + key]; const c = { ...raw, gQ: I16(raw.gen), uQ: I16(raw.ung), roleA: new Int8Array(unb64(raw.roles).buffer), simQ: {} };
      for (const [sn, sd] of Object.entries(raw.sims)) { c.simQ[sn] = {}; for (const [v, b] of Object.entries(sd.layers)) c.simQ[sn][v] = I16(b); } CACHE[key] = c; res(c); };
    s.onerror = () => rej(new Error('case ' + key)); document.body.appendChild(s); }); }
const root = document.getElementById('viewer'); const canvas = root.querySelector('canvas'); const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true }); renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene(); const camera = new THREE.PerspectiveCamera(42, 1.6, 0.005, 200);
let gPts = null, uPts = null, sPts = null, floor = null, goalM = null, contourM = null, walls = []; const center = new THREE.Vector3(), panOff = new THREE.Vector3();
let camTheta = -0.95, camPhi = 1.12, camRad = 1.2, home = 1.2; let ep = null, frame = 0, playing = false, lastT = 0, simName = null;
function mkPoints(n, opacity) { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3)); g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  return new THREE.Points(g, new THREE.PointsMaterial({ size: 0.006, vertexColors: true, sizeAttenuation: true, transparent: true, opacity, depthWrite: opacity > 0.6 })); }
const dq = (q, a) => ep.qlo[a] + (q + 32767) / 65534 * ep.qspan[a];
function clear() { for (const o of [gPts, uPts, sPts, floor, goalM, contourM, ...walls]) if (o) scene.remove(o); gPts = uPts = sPts = floor = goalM = contourM = null; walls = []; }
function build(next) { ep = next; frame = 0; clear(); simName = Object.keys(ep.sims)[0] || null;
  gPts = mkPoints(ep.N, 0.9); uPts = mkPoints(ep.N, 0.55); sPts = mkPoints(ep.N, 0.6); scene.add(gPts); scene.add(uPts); scene.add(sPts);
  const mid = [0, 1, 2].map(a => ep.lo[a] + ep.span[a] / 2); center.set(mid[0], mid[1], mid[2]); const fs = Math.max(ep.span[0], ep.span[1]) * 2.2;
  floor = new THREE.Mesh(new THREE.PlaneGeometry(fs, fs), new THREE.MeshBasicMaterial({ color: new THREE.Color(cssv('--muted')), transparent: true, opacity: 0.13, side: THREE.DoubleSide, depthWrite: false })); floor.position.set(center.x, center.y, 0); scene.add(floor);
  goalM = new THREE.Mesh(new THREE.SphereGeometry(0.012, 16, 12), new THREE.MeshBasicMaterial({ color: new THREE.Color(cssv('--goal')), transparent: true, opacity: 0.85 })); goalM.position.set(ep.target[0], ep.target[1], ep.target[2]); if (!Number.isNaN(ep.target[0])) scene.add(goalM);
  if (ep.contour) { const cg = new THREE.BufferGeometry(); cg.setAttribute('position', new THREE.Float32BufferAttribute(ep.contour.flat(), 3)); contourM = new THREE.Points(cg, new THREE.PointsMaterial({ color: new THREE.Color(cssv('--goal')), size: 0.006, sizeAttenuation: true })); scene.add(contourM); }
  for (const b of ep.boxes) { const w = new THREE.Mesh(new THREE.BoxGeometry(b[3], b[4], b[5]), new THREE.MeshBasicMaterial({ color: new THREE.Color(cssv('--wall')), transparent: true, opacity: 0.35 })); w.position.set(b[0], b[1], b[2]); scene.add(w); walls.push(w); }
  home = Math.hypot(ep.span[0], ep.span[1], ep.span[2]) * 1.35; camRad = home; panOff.set(0, 0, 0); camTheta = -0.95; camPhi = 1.12;
  root.querySelector('.seek').max = String(ep.T - 1); root.querySelector('.seek').value = '0';
  const simSel = root.querySelector('.t-sim'); simSel.disabled = !simName; if (!simName) simSel.checked = false;
  root.querySelector('.meta').textContent = 'ep' + ep.episode + ' m' + ep.member + ' · goal: guided ' + ep.goal_mm.guided.toFixed(0) + ' mm, unguided ' + ep.goal_mm.unguided.toFixed(0) + ' mm' + (simName ? ' · simulated (' + ep.sims[simName].label + '): ' + ep.sims[simName].rows.map(r => r.variant + ' ' + (r.sim_to_target_mm !== undefined ? r.sim_to_target_mm.toFixed(0) + ' mm to goal, ' : '') + 'sim vs gen ' + r.sim_vs_gen_final_mm.toFixed(0) + ' mm').join('; ') : ' · no replay yet');
  const R = ep.renders || {}; const slots = [['guided_hand', 'renderimg', 'renderlabel1', 'gripper only'], ['guided_arm', 'renderimg2', 'renderlabel2', 'gripper on the Franka arm (IK-solved)']]; let any = false;
  for (const [k, iid, lid, label] of slots) { const img = document.getElementById(iid), lab = document.getElementById(lid); const rr = R[k];
    if (rr) { img.src = rr.src; img.alt = label + ': generated (left) and ' + rr.sim + ' replay (right)'; img.hidden = false; lab.textContent = label + (rr.scale && rr.scale !== 1 ? ' · scene drawn at ' + rr.scale + 'x (render only; the robot is life-size)' : '') + (rr.arm_track_mm_max != null ? ' · arm tracks the hand to ' + rr.arm_track_mm_max.toFixed(1) + ' mm / ' + rr.arm_track_deg_max.toFixed(1) + ' deg' : ''); any = true; }
    else { img.hidden = true; img.removeAttribute('src'); lab.textContent = ''; } }
  document.getElementById('rendernote').hidden = any;
  drawMini(); update(); }
function fill(P, Q, roleA, n, off, cols) { const pp = P.geometry.attributes.position.array, pc = P.geometry.attributes.color.array;
  for (let i = 0; i < n; i++) { pp[i*3] = dq(Q[off+i*3], 0); pp[i*3+1] = dq(Q[off+i*3+1], 1); pp[i*3+2] = dq(Q[off+i*3+2], 2); const r = roleA[i]; const c = cols[r < 0 ? 2 : r]; pc[i*3] = c.r; pc[i*3+1] = c.g; pc[i*3+2] = c.b; }
  P.geometry.attributes.position.needsUpdate = true; P.geometry.attributes.color.needsUpdate = true; }
function update() { if (!ep) return; const off = frame * ep.N * 3; const C = n => new THREE.Color(cssv(n));
  fill(gPts, ep.gQ, ep.roleA, ep.N, off, [C('--s1'), C('--fingerc'), C('--s2')]); fill(uPts, ep.uQ, ep.roleA, ep.N, off, [C('--u1'), C('--u1'), C('--u2')]);
  const simVar = root.querySelector('.t-simvar').value; const showSim = root.querySelector('.t-sim').checked && simName && ep.simQ[simName][simVar];
  if (showSim) fill(sPts, ep.simQ[simName][simVar], ep.roleA, ep.N, off, [C('--simg'), C('--simg'), C('--simo')]);
  gPts.visible = root.querySelector('.t-gen').checked; uPts.visible = root.querySelector('.t-ung').checked; sPts.visible = !!showSim; floor.visible = root.querySelector('.t-floor').checked;
  const sz = +root.querySelector('.t-size').value; gPts.material.size = sz * 0.0004; uPts.material.size = sz * 0.0004; sPts.material.size = sz * 0.00045;
  const s = ep.series; let h = 'frame ' + frame + ' · object to goal: guided ' + s.gen_goal[frame].toFixed(0) + ' mm, unguided ' + s.ung_goal[frame].toFixed(0) + ' mm';
  if (showSim && s['sim_' + simVar + '_goal']) h += ' · simulated ' + s['sim_' + simVar + '_goal'][frame].toFixed(0) + ' mm (sim vs gen ' + s['sim_' + simVar + '_vs_gen'][frame].toFixed(0) + ' mm)';
  root.querySelector('.hud').innerHTML = '<span class="now">' + h + '</span>'; root.querySelector('.fnum').textContent = 'frame ' + frame + ' / ' + (ep.T - 1); root.querySelector('.seek').value = String(frame); drawCursor(); }
const mini = root.querySelector('.mini'); const CW = 1000, CH = 150, PL = 52, PR = 12, PT = 12, PB = 24;
function drawMini() { if (!ep) return; const s = ep.series; const simVar = root.querySelector('.t-simvar').value; const lines = [['gen_goal', '--s2'], ['ung_goal', '--u2']]; if (simName && s['sim_' + simVar + '_goal']) lines.push(['sim_' + simVar + '_goal', '--simo']);
  const ymax = Math.max(20, ...lines.flatMap(([k]) => s[k].map(v => Math.min(v, 800)))) * 1.15; const X = i => PL + i / (ep.T - 1) * (CW - PL - PR); const Y = v => CH - PB - Math.min(v, ymax) / ymax * (CH - PT - PB); let o = '';
  for (let k = 0; k <= 2; k++) { const v = ymax * k / 2, yy = Y(v); o += '<line x1="' + PL + '" y1="' + yy + '" x2="' + (CW - PR) + '" y2="' + yy + '" stroke="' + cssv('--rule2') + '" stroke-width="1"/><text x="' + (PL - 7) + '" y="' + (yy + 3.5) + '" text-anchor="end" font-family="' + cssv('--mono') + '" font-size="11" fill="' + cssv('--muted') + '">' + v.toFixed(0) + '</text>'; }
  for (const [k, col] of lines) { const d = s[k]; let p = ''; for (let i = 0; i < d.length; i++) p += (i === 0 ? 'M' : 'L') + X(i).toFixed(1) + ' ' + Y(d[i]).toFixed(1); o += '<path d="' + p + '" fill="none" stroke="' + cssv(col) + '" stroke-width="2.2" stroke-linejoin="round"/>'; }
  o += '<text x="' + (CW - PR) + '" y="' + (CH - 7) + '" text-anchor="end" font-family="' + cssv('--mono') + '" font-size="10.5" fill="' + cssv('--muted') + '">object centre to goal, mm (guided / unguided / simulated)</text><line id="cur" x1="0" y1="' + PT + '" x2="0" y2="' + (CH - PB) + '" stroke="' + cssv('--ink') + '" stroke-width="1.2"/>'; mini.innerHTML = o; }
function drawCursor() { const c = mini.querySelector('#cur'); if (!c || !ep) return; const x = PL + frame / (ep.T - 1) * (CW - PL - PR); c.setAttribute('x1', x); c.setAttribute('x2', x); }
function resize() { const w = canvas.clientWidth, h = canvas.clientHeight, r = renderer.getPixelRatio(); if (!w || !h) return; if (canvas.width !== Math.round(w * r) || canvas.height !== Math.round(h * r)) { renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); } }
function loop(t) { resize(); if (playing && ep && t - lastT > 110) { lastT = t; frame = (frame + 1) % ep.T; update(); }
  if (ep) { const p = Math.max(0.08, Math.min(Math.PI - 0.08, camPhi)); camera.position.set(center.x + panOff.x + camRad * Math.sin(p) * Math.cos(camTheta), center.y + panOff.y + camRad * Math.sin(p) * Math.sin(camTheta), center.z + panOff.z + camRad * Math.cos(p)); camera.up.set(0, 0, 1); camera.lookAt(center.x + panOff.x, center.y + panOff.y, center.z + panOff.z); renderer.render(scene, camera); } requestAnimationFrame(loop); }
requestAnimationFrame(loop);
let drag = null; canvas.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, shift: e.shiftKey }; canvas.setPointerCapture(e.pointerId); canvas.style.cursor = 'grabbing'; });
canvas.addEventListener('pointermove', e => { if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY; if (drag.shift) { const s = camRad * 0.0016; panOff.x += -dx * s * Math.sin(camTheta); panOff.y += dx * s * Math.cos(camTheta); panOff.z += dy * s; } else { camTheta -= dx * 0.008; camPhi -= dy * 0.008; } });
addEventListener('pointerup', () => { drag = null; canvas.style.cursor = 'grab'; }); canvas.addEventListener('wheel', e => { e.preventDefault(); camRad = Math.max(home * 0.15, Math.min(home * 4, camRad * (1 + Math.sign(e.deltaY) * 0.09))); }, { passive: false });
root.querySelector('.play').addEventListener('click', ev => { playing = !playing; ev.currentTarget.innerHTML = playing ? '&#10073;&#10073;' : '&#9654;'; });
root.querySelector('.seek').addEventListener('input', e => { playing = false; root.querySelector('.play').innerHTML = '&#9654;'; frame = +e.currentTarget.value; update(); });
for (const cls of ['.t-gen', '.t-ung', '.t-sim', '.t-floor', '.t-size']) root.querySelector(cls).addEventListener('input', update);
root.querySelector('.t-simvar').addEventListener('input', () => { drawMini(); update(); });
root.querySelector('.reset').addEventListener('click', () => { camRad = home; panOff.set(0, 0, 0); camTheta = -0.95; camPhi = 1.12; });
addEventListener('keydown', e => { if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return; if (e.key === ' ') { e.preventDefault(); root.querySelector('.play').click(); } if (e.key === 'ArrowLeft' && ep) { frame = (frame - 1 + ep.T) % ep.T; update(); } if (e.key === 'ArrowRight' && ep) { frame = (frame + 1) % ep.T; update(); } });
// ---- navigation: domain -> run -> member
const dtabs = document.getElementById('dtabs'), runsEl = document.getElementById('runs'), mtabs = document.getElementById('mtabs'), tableEl = document.getElementById('table'), rundesc = document.getElementById('rundesc');
let cur = { d: null, r: null, m: null };
function fmt(v) { return v === undefined || v === null ? '' : (typeof v === 'number' ? v.toFixed(0) : String(v)); }
function renderTable(run) { const sims = Object.keys(run.sims); let h = '<div class="scroller"><table><thead><tr><th scope="col">member</th><th scope="col" class="num">goal: unguided</th><th scope="col" class="num">goal: guided</th>';
  for (const s of sims) h += '<th scope="col" class="num">' + s + ': sim to goal (guided)</th><th scope="col" class="num">sim vs gen final</th><th scope="col" class="num">sim moved</th><th scope="col" class="num">gen moved</th><th scope="col" class="num">track mm</th>';
  h += '</tr></thead><tbody>';
  run.members.forEach((m, i) => { h += '<tr><th scope="row">' + m.label + '</th><td class="num">' + fmt(run.unguided_mm[i]) + '</td><td class="num strong">' + fmt(run.best_mm[i]) + '</td>';
    for (const s of sims) { const r = run.sims[s].find(x => x.member === i && x.variant === 'guided') || {}; h += '<td class="num strong">' + fmt(r.sim_to_target_mm) + '</td><td class="num">' + fmt(r.sim_vs_gen_final_mm) + '</td><td class="num">' + fmt(r.sim_moved_mm) + '</td><td class="num">' + fmt(r.gen_moved_mm) + '</td><td class="num">' + (r.track_mm_mean !== undefined ? r.track_mm_mean.toFixed(1) + ' / ' + r.track_mm_max.toFixed(1) : '') + '</td>'; }
    h += '</tr>'; });
  h += '</tbody></table></div>'; if (!sims.length) h += '<p class="empty">no simulator replay for this run yet</p>'; tableEl.innerHTML = h; }
function selectMember(run, mi, push) { cur.m = mi; [...mtabs.querySelectorAll('button')].forEach((b, i) => { b.setAttribute('aria-selected', String(i === mi)); b.tabIndex = i === mi ? 0 : -1; });
  loadCase(run.members[mi].id).then(build).catch(err => { root.querySelector('.meta').textContent = 'could not load ' + err.message; }); if (push !== false) location.hash = cur.d + '/' + run.name + '/' + mi; }
function selectRun(dom, ri, mi, push) { cur.r = ri; const run = dom.runs[ri]; [...runsEl.querySelectorAll('button')].forEach((b, i) => b.setAttribute('aria-selected', String(i === ri)));
  const k = run.knobs; rundesc.innerHTML = '<b>' + run.name + '</b>' + (run.desc ? ' — ' + run.desc : '') + '<br>' + run.run_dir + ' · ' + run.ckpt + ' · ' + run.sampler + '<br>' + Object.entries(k).map(([a, b]) => a + '=' + (Array.isArray(b) ? b.join(',') : b)).join(' · ') + (run.iteration_seconds ? ' · ' + run.iteration_seconds.slice(1, -1).reduce((a, b) => a + b, 0) / Math.max(1, run.iteration_seconds.length - 2) + ' s/it, peak ' + (run.peak_gib ? Math.max(...run.peak_gib) : '–') + ' GiB' : '') + ' · ' + run.created;
  const argmin = k => { let bi = -1, bv = Infinity; run.members.forEach((m, i) => { if (m[k] != null && m[k] < bv) { bv = m[k]; bi = i; } }); return bi; };
  const bg = argmin('gen_goal'), bs = argmin('sim_goal');
  mtabs.innerHTML = run.members.map((m, i) => { const g = i === bg, sm = i === bs; const cls = g && sm ? 'best-both' : g ? 'best-gen' : sm ? 'best-sim' : '';
    const tag = g && sm ? ' ✓ closest gen + sim' : g ? ' ◆ closest gen' : sm ? ' ● closest sim' : '';
    return '<button role="tab" type="button" class="' + cls + '">' + m.label + ' <small>gen ' + (m.gen_goal != null ? m.gen_goal.toFixed(0) : '–') + ' / sim ' + (m.sim_goal != null ? m.sim_goal.toFixed(0) : '–') + ' mm</small>' + tag + '</button>'; }).join('')
    + '<span class="tabkey">distance to goal at the last frame · ◆ closest in the generated rollout · ● closest in Isaac · green = closest in both</span>';
  [...mtabs.querySelectorAll('button')].forEach((b, i) => { b.addEventListener('click', () => selectMember(run, i)); b.addEventListener('keydown', e => { const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0; if (!d) return; e.preventDefault(); const n = (i + d + run.members.length) % run.members.length; mtabs.querySelectorAll('button')[n].focus(); selectMember(run, n); }); });
  renderTable(run); selectMember(run, Math.min(mi || 0, run.members.length - 1), push); }
function selectDomain(di, ri, mi, push) { cur.d = INDEX[di].id; [...dtabs.children].forEach((b, i) => b.setAttribute('aria-selected', String(i === di))); const dom = INDEX[di];
  let lastG = null; runsEl.innerHTML = dom.runs.map((r, i) => { const head = (r.group && r.group !== lastG) ? '<p class="rgroup">' + r.group + '</p>' : ''; lastG = r.group;
    return head + '<button role="tab" type="button"><b>' + (r.short || r.name) + '</b></button>'; }).join('');
  [...runsEl.querySelectorAll('button')].forEach((b, i) => b.addEventListener('click', () => selectRun(dom, i, 0)));
  if (dom.runs.length) selectRun(dom, Math.min(ri || 0, dom.runs.length - 1), mi, push); else { rundesc.textContent = ''; mtabs.innerHTML = ''; tableEl.innerHTML = '<p class="empty">no runs</p>'; } }
dtabs.innerHTML = INDEX.map((d, i) => '<button role="tab" type="button">' + d.label + ' <small>(' + d.runs.length + ')</small></button>').join('');
[...dtabs.children].forEach((b, i) => b.addEventListener('click', () => selectDomain(i, 0, 0)));
(function start() { const h = location.hash.slice(1).split('/'); let di = INDEX.findIndex(d => d.id === h[0]); if (di < 0) di = 0; const ri = di >= 0 ? Math.max(0, INDEX[di].runs.findIndex(r => r.name === h[1])) : 0; selectDomain(di, ri, +(h[2] || 0), false); })();
</script>"""
