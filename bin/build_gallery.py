"""Build the gallery page (site_gallery/index.html) from gallery_chain.sh outputs.

usage: python bin/build_gallery.py [--cases sweeps/gallery_cases.json] [--site site_gallery]
cases json: [{"base": "gal_jar_air_fr24", "object": "peanut butter jar", "title": "Air lift, 30 cm forward and 15 cm up",
              "twin": "<base of the no-obstacle twin>" (optional), "note": "" (optional)}, ...]
Per case it reads logs/<base>_chain.log (the "gallery pick" + LIFT_TABLE lines), the picked run's result.json / rollout.npz,
its commanded-arm replay (replay_isaac_arm/isaac/replay.{json,npz}) and copies the asset render mp4 + keyframes into the site dir.
"""
import argparse, glob, html, json, os, re, shutil, time
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--cases", default="sweeps/gallery_cases.json"); ap.add_argument("--site", default="site_gallery")
ap.add_argument("--title", default="Pick, carry and place gallery"); A = ap.parse_args()
ROOT = "/data/user_data/mbronars/packages/guided_playback"; os.chdir(ROOT)
SHIFT = np.array([0.45, 0.0, 0.10])
PICK = re.compile(r"gallery pick (\S+) m(\d+) \(lifting members (\d+)/(\d+); floating lift (-?\d+) mm, gen (-?\d+) mm, sim (-?\d+) mm\)")


def case_data(c):
    base = c["base"]; log = f"logs/{base}_chain.log"
    d = dict(c, ok=False)
    pj = f"runs/pickup/{base}_pick.json"; env = 0
    if os.path.exists(pj):                                    # arm-based pick (gallery_repick.sh)
        pk = json.load(open(pj)); run, mem, env = pk["run"], int(pk["member"]), int(pk["env"])
        d.update(run=run, member=mem, lifting=pk["lifting"], members=pk["members"], arm_hold=pk["arm_hold"], float_lift=pk["float_lift"], gen_mm=pk["gen_mm"],
                 arm_table="; ".join(f"s{r['seed']}m{r['member']}: peak {r['peak']:.0f} end {r['end']:.0f} final {r['final']:.0f} mm" for r in pk["arm_table"]))
        txt = open(log).read() if os.path.exists(log) else ""
    else:
        if not os.path.exists(log): d["status"] = "not started"; return d
        txt = open(log).read(); m = PICK.search(txt)
        if not m: d["status"] = "planning / replaying"; return d
        run, mem, nl, nt, fl, gen, sim = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), float(m.group(5)), float(m.group(6)), float(m.group(7))
        R = f"runs/pickup/{run}"; d.update(run=run, member=mem, lifting=nl, members=nt, float_lift=fl, gen_mm=gen, float_sim_mm=sim)
    lt = re.search(r"LIFT_TABLE (.*)", txt); d["lift_table"] = lt.group(1) if lt else ""
    R = f"runs/pickup/{run}"
    rj = json.load(open(f"{R}/result.json")); d["desc"] = rj.get("desc", ""); d["iterations"] = rj.get("iterations"); d["offset"] = rj.get("offset")
    d["freeze"] = rj.get("freeze_frames"); d["still"] = rj.get("still_weight"); d["obstacle_asset"] = rj.get("obstacle_asset"); d["upright"] = rj.get("upright_weight")
    z = np.load(f"{R}/rollout.npz"); tgt = z["target"][mem] + SHIFT
    arm = f"{R}/replay_isaac_arm"
    rp = f"{arm}/isaac/replay.json"
    if not os.path.exists(rp): d["status"] = "arm replay pending"; return d
    res = [r for r in json.load(open(rp))["results"] if int(r["episode"]) == mem and r["variant"] == "guided"]
    Z = np.load(f"{arm}/isaac/replay.npz", allow_pickle=True)
    k = [env] if int(Z["env_episode"][env]) == mem else [i for i in range(len(Z["env_episode"])) if int(Z["env_episode"][i]) == mem and int(Z["env_variant"][i]) == 0]
    if res and k:
        r = res[0]; op = Z["obj_pos"][k[0]]
        d.update(arm_hand_mean=r["hand_pos_err_mm_mean"], arm_hand_max=r["hand_pos_err_mm_max"], arm_rot_max=r["hand_rot_err_deg_max"],
                 arm_peak_lift=float((op[:, 2].max() - op[0, 2]) * 1e3), arm_final_mm=float(np.linalg.norm(op[-1] - tgt) * 1e3), arm_end_lift=float((op[-1, 2] - op[0, 2]) * 1e3))
    mp4 = sorted(glob.glob(f"{arm}/render_asset/env{env}_ep{mem}.mp4")); kf = f"{arm}/render_asset/keyframes_trace_env{env}.png"; hull = sorted(glob.glob(f"{arm}/render_x/env{env}_ep{mem}.mp4"))
    if not mp4 or not os.path.exists(kf): d["status"] = "rendering"; return d
    out = os.path.join(A.site, base); os.makedirs(out, exist_ok=True)
    shutil.copy(mp4[0], f"{out}/arm.mp4"); shutil.copy(kf, f"{out}/keyframes.png")
    if hull: shutil.copy(hull[0], f"{out}/hull.mp4")
    d.update(ok=True, status="done", hull=bool(hull)); return d


def fmt(v, nd=0):
    return "–" if v is None else (f"{v:.{nd}f}")


def cell(d):
    if not d.get("ok"):
        return f'<div class="case pending"><h3>{html.escape(d.get("title", d["base"]))}</h3><p class="facts">{html.escape(d.get("note") or d.get("status", ""))}</p></div>'
    facts = (f'{d["lifting"]}/{d["members"]} members lift in the floating hand' + (f', the commanded arm holds {d["arm_hold"]} of them' if d.get("arm_hold") is not None else '') + f'; shown: {html.escape(d["run"])} m{d["member"]}. '
             f'Generation {fmt(d["gen_mm"])} mm from the target. Commanded arm: ends {fmt(d.get("arm_final_mm"))} mm from the target, '
             f'peak lift {fmt(d.get("arm_peak_lift"))} mm, hand tracking {fmt(d.get("arm_hand_mean"), 1)} / {fmt(d.get("arm_hand_max"), 1)} mm, rotation max {fmt(d.get("arm_rot_max"), 1)}&deg;. '
             f'{d["iterations"]} iterations' + (f', freeze {d["freeze"]}' if d.get("freeze") else "") + (f', still {d["still"]}' if d.get("still") else "") + (f', upright {d["upright"]}' if d.get("upright") else "") + ".")
    note = f'<p class="facts">{html.escape(d["note"])}</p>' if d.get("note") else ""
    hull = f' <a href="{d["base"]}/hull.mp4">hull render</a>' if d.get("hull") else ""
    return (f'<div class="case"><h3>{html.escape(d.get("title", d["base"]))}</h3>'
            f'<figure><video src="{d["base"]}/arm.mp4" controls muted loop playsinline preload="metadata"></video></figure>'
            f'<figure><img src="{d["base"]}/keyframes.png" alt="{html.escape(d.get("title", ""))}: start and end"><figcaption>{facts}{hull}</figcaption></figure>{note}'
            f'<details><summary class="facts">all 8 members (floating hand)</summary><p class="meta">floating: {html.escape(d.get("lift_table", ""))}</p>' + (f'<p class="meta">arm: {html.escape(d["arm_table"])}</p>' if d.get("arm_table") else '') + '</details></div>')


cases = json.load(open(A.cases)); data = {c["base"]: case_data(c) for c in cases}
groups = {}
for c in cases:
    if not c.get("failed"): groups.setdefault(c["object"], []).append(data[c["base"]])
failed = [data[c["base"]] for c in cases if c.get("failed")]
rows = "".join(f'<tr><td>{html.escape(d["object"])}</td><td>{html.escape(d.get("title", d["base"]))}</td><td>{d.get("lifting", "–")}/{d.get("members", 8)}</td>'
               f'<td>{fmt(d.get("gen_mm"))}</td><td>{fmt(d.get("arm_final_mm"))}</td><td>{fmt(d.get("arm_peak_lift"))}</td><td>{html.escape(d.get("status", ""))}</td></tr>' for d in data.values())
body = []
for obj, ds in groups.items():
    body.append(f'<section><h2>{html.escape(obj)}</h2><div class="grid">')
    done = set()
    for d in ds:
        if d["base"] in done: continue
        tw = data.get(d.get("twin")) if d.get("twin") else None
        if tw:
            body.append(f'<div class="pair">{cell(tw)}{cell(d)}</div>'); done.add(tw["base"])
        else:
            body.append(cell(d))
        done.add(d["base"])
    body.append('</div></section>')
if failed:
    body.append('<section><h2>Cases that did not work (kept for the record)</h2><div class="grid">' + ''.join(cell(d) for d in failed) + '</div></section>')
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(A.title)}</title>
<style>
:root{{--bg:#f4f5f3;--panel:#fff;--ink:#1c2320;--muted:#5e6964;--rule:#d8ddd9;--serif:"Times New Roman",Times,"Nimbus Roman",serif}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{color-scheme:dark;--bg:#141816;--panel:#1c211f;--ink:#e4e9e6;--muted:#9aa6a0;--rule:#2e3633}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#141816;--panel:#1c211f;--ink:#e4e9e6;--muted:#9aa6a0;--rule:#2e3633}}
body{{background:var(--bg);color:var(--ink);font-family:var(--serif);font-size:17px;line-height:1.5;margin:0}}
.wrap{{max-width:1400px;margin:0 auto;padding:26px 16px 44px;display:grid;gap:26px}}
h1{{font-size:30px;margin:0;line-height:1.15}}h2{{font-size:22px;margin:0 0 10px}}h3{{font-size:16px;margin:0 0 6px}}
.lede{{color:var(--muted);max-width:90ch;margin:6px 0 0}}.facts{{font-size:14.5px;color:var(--muted);margin:4px 0}}.meta{{font-size:12.5px;color:var(--muted);word-break:break-word}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}}.pair{{grid-column:1/-1;display:grid;grid-template-columns:1fr 1fr;gap:14px}}
@media (max-width:760px){{.pair{{grid-template-columns:1fr}}}}
.case{{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:14px;display:grid;gap:8px;align-content:start}}.pending{{opacity:.7}}
video,img{{width:100%;max-width:100%;border-radius:6px;display:block;background:#fff}}figure{{margin:0;display:grid;gap:6px}}figcaption{{font-size:14px;color:var(--muted)}}
table{{border-collapse:collapse;font-size:14px}}th,td{{text-align:left;padding:3px 10px 3px 0;border-bottom:1px solid var(--rule)}}
</style></head><body><div class="wrap">
<header><h1>{html.escape(A.title)}</h1>
<p class="lede">Pickup-only checkpoint (cont200k final.pt). Each case: 8 guided members (two seeds x population 4), every member replayed with the floating Franka hand in Isaac;
every member that lifts is then replayed with the commanded Franka arm, and the member the arm holds with the smallest final distance to the target is rendered (asset meshes; dark-green X on the table = ground target, dark-green ball = target in the air).
Guidance: Adam on the initial pose noise, shell reprojection, grasp-phase freeze (noise rows of the first frames are not optimized) and a pre-grasp stillness term.
Numbers are measurements in the simulator; obstacle cases sit next to their no-obstacle twin to the same target.</p></header>
<p class="meta">Last built {time.strftime("%Y-%m-%d %H:%M")} · {sum(d.get("ok", False) for d in data.values())} of {len(data)} cases rendered.</p>
<section><h2>Summary</h2><table><thead><tr><th>object</th><th>case</th><th>lift in floating hand</th><th>generation to target (mm)</th><th>arm: final distance (mm)</th><th>arm: peak lift (mm)</th><th>status</th></tr></thead><tbody>{rows}</tbody></table></section>
{''.join(body)}
</div></body></html>"""
os.makedirs(A.site, exist_ok=True); open(os.path.join(A.site, "index.html"), "w").write(page)
print("wrote", os.path.abspath(os.path.join(A.site, "index.html")), "| cases done:", sum(d.get("ok", False) for d in data.values()), "/", len(data))
