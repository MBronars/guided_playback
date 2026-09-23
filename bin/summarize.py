"""Summary of a sweep (or any list of runs): per run, generated + Isaac numbers of the guided rollouts (all members + the best
member by generated objective). Writes <arms>.summary.json / .md next to the arms file.
  python bin/summarize.py sweeps/<file>.json"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = json.load(open(sys.argv[1])); base = spec.get("base", {}); arms = [dict(base, **a) for a in spec["arms"]]
rows = []
for a in arms:
    run = os.path.join(ROOT, "runs", a.get("domain", "pickup"), a["name"])
    if not os.path.exists(os.path.join(run, "result.json")):
        continue
    rj = json.load(open(os.path.join(run, "result.json"))); z = np.load(os.path.join(run, "rollout.npz"))
    an = json.load(open(os.path.join(run, "replay_isaac", "analysis.json"))) if os.path.exists(os.path.join(run, "replay_isaac", "analysis.json")) else None
    B = len(rj["best_mm"]); mem = []
    for m in range(B):
        o = z["object_mask"][m]; G = z["predicted"][m][:, o].mean(1); U = z["unguided"][m][:, o].mean(1); tgt = z["target"][m]
        d = dict(member=m, unguided_goal=float(np.linalg.norm(U[-1] - tgt) * 1e3), gen_goal=float(np.linalg.norm(G[-1] - tgt) * 1e3),
                 gen_end_dz=float((G[-1, 2] - G[0, 2]) * 1e3), gen_peak_dz=float((G[:, 2].max() - G[0, 2]) * 1e3), objective=rj["best_mm"][m])
        if len(z["boxes"]):
            c, h = z["boxes"][0][:3], z["boxes"][0][3:] / 2; P = z["predicted"][m]; q = np.abs(P - c) - h
            sd = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0); d["gen_wall_min_mm"] = float(sd[1:].min() * 1e3)
            d["gen_crossed"] = bool((G[-1, 0] - c[0]) * np.sign(tgt[0] - G[0, 0]) > h[0])
        if an:
            r = [x for x in an["rows"] if x["member"] == m and x["variant"] == "guided"][0]
            S = np.load(os.path.join(run, "replay_isaac", "replay.npz"))["sim"][0, m][:, o].mean(1)
            d.update(sim_goal=r.get("sim_to_target_mm"), sim_end_dz=float((S[-1, 2] - S[0, 2]) * 1e3), sim_peak_dz=r["sim_lift_mm"], sim_vs_gen=r["sim_vs_gen_final_mm"],
                     held=bool((S[-1, 2] - S[0, 2]) > 0.05), track=r["track_mm_mean"])
            if len(z["boxes"]):
                Pz = np.load(os.path.join(run, "replay_isaac", "replay.npz"))["sim"][0, m]; q = np.abs(Pz - c) - h
                sd = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0); d["sim_wall_min_mm"] = float(sd[1:].min() * 1e3)
                d["sim_crossed"] = bool((S[-1, 0] - c[0]) * np.sign(tgt[0] - S[0, 0]) > h[0])
        mem.append(d)
    best = min(mem, key=lambda d: d["objective"])
    knobs = {k: a.get(k) for k in ("sampler", "transitions", "steps", "churn", "optimizer", "shell", "lr", "lr_final", "iterations", "t_freeze")}
    rows.append(dict(name=a["name"], knobs=knobs, seconds=sum(rj["iteration_seconds"]), members=mem, best=best,
                     sim_goal_mean=float(np.mean([d["sim_goal"] for d in mem])) if an else None, gen_goal_mean=float(np.mean([d["gen_goal"] for d in mem])),
                     held=sum(d.get("held", False) for d in mem) if an else None))
out = os.path.splitext(sys.argv[1])[0]
json.dump(rows, open(out + ".summary.json", "w"), indent=1)
L = ["| arm | knobs | s | gen goal mean | sim goal mean | held in sim | best m: gen goal / sim goal / sim end dz |", "|---|---|---|---|---|---|---|"]
for r in rows:
    k = r["knobs"]; b = r["best"]
    ks = f"{k['sampler']} N{k['transitions']} S{k['steps']} {'churn' if k['churn'] else 'det'} {k['optimizer']} {'shell' if k['shell'] else 'noshell'} lr {k['lr']}->{k['lr_final']} it{k['iterations']} tf{k['t_freeze']}"
    L.append(f"| {r['name']} | {ks} | {r['seconds']:.0f} | {r['gen_goal_mean']:.0f} | {r['sim_goal_mean'] if r['sim_goal_mean'] is None else round(r['sim_goal_mean'])} | {r['held']}/{len(r['members'])} | m{b['member']}: {b['gen_goal']:.0f} / {b.get('sim_goal', float('nan')):.0f} / {b.get('sim_end_dz', float('nan')):.0f} |")
open(out + ".summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
