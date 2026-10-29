"""Post-hoc: would another pick rule (from imagination only) have chosen better members?  python pickrule.py <tag>"""
import sys, os, glob, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from tilt import report
HERE = "/data/user_data/mbronars/packages/guided_playback"; tag = sys.argv[1]
cases = []
for run in sorted(glob.glob(f"{HERE}/runs/rl_push/{tag}_ep*")):
    if not os.path.exists(run + "/replay_isaac/analysis.json"): continue
    z = np.load(run + "/rollout.npz"); rj = json.load(open(run + "/result.json")); tr = report(run)
    an = {r["member"]: r for r in json.load(open(run + "/replay_isaac/analysis.json"))["rows"] if r["variant"] == "guided"}
    roles = z["roles"][0]; hand = roles == 0; obj = z["object_mask"]
    feats = []
    for m in range(len(tr)):
        g = z["predicted"][m]; v = np.linalg.norm(np.diff(g[:, hand].mean(1), axis=0), axis=1) * 1e3
        oc = g[:, obj[m]].mean(1); reach = int(np.argmax(np.linalg.norm(oc - z["target"][m], axis=1) < 0.05)) if (np.linalg.norm(oc - z["target"][m], axis=1) < 0.05).any() else 60
        post = np.linalg.norm(oc[-1] - oc[reach]) * 1e3 if reach < 60 else 0.0
        feats.append(dict(imag=rj["best_mm"][m], tilt=tr[m]["gen_tilt_max"], lift=an[m]["gen_lift_mm"], vmax=float(v.max()), vmean=float(v.mean()), reach=reach, post=post,
                          sim=an[m]["sim_to_target_mm"], simtilt=tr[m]["sim_tilt_max"], gap=an[m]["sim_vs_gen_final_mm"]))
    cases.append(feats)
def evaluate(rule, name):
    sims = []; ok = 0
    for feats in cases:
        m = rule(feats); sims.append(feats[m]["sim"]); ok += feats[m]["sim"] <= 50 and feats[m]["simtilt"] < 20
    print(f"{name:55s} median {np.median(sims):6.1f} mm   ok {ok}/{len(cases)}")
evaluate(lambda f: int(np.argmin([x["imag"] for x in f])), "argmin imagined (planner)")
evaluate(lambda f: int(np.argmin([x["sim"] for x in f])), "oracle (best in sim)")
for k in (1, 3, 10): evaluate(lambda f, k=k: int(np.argmin([x["imag"] + k * x["tilt"] for x in f])), f"argmin imagined + {k} mm/deg * gen tilt")
for k in (1, 3): evaluate(lambda f, k=k: int(np.argmin([x["imag"] + k * x["lift"] for x in f])), f"argmin imagined + {k} * gen lift")
for k in (0.5, 1, 2): evaluate(lambda f, k=k: int(np.argmin([x["imag"] + k * x["vmax"] for x in f])), f"argmin imagined + {k} * max hand speed (mm/frame)")
for k in (0.5, 1): evaluate(lambda f, k=k: int(np.argmin([x["imag"] + k * x["post"] for x in f])), f"argmin imagined + {k} * post-reach object drift")
evaluate(lambda f: int(np.argmin([max(x["imag"], 20) for x in f])), "argmin max(imagined, 20)")
evaluate(lambda f: int(np.argmin([x["imag"] + 3 * x["tilt"] + 1 * x["vmax"] for x in f])), "argmin imagined + 3*tilt + 1*vmax")
# correlations
allf = [x for f in cases for x in f]
for key in ("imag", "tilt", "lift", "vmax", "vmean", "post", "reach"):
    a = np.array([x[key] for x in allf]); s = np.array([x["sim"] for x in allf]); print(f"corr({key}, sim) = {np.corrcoef(a, s)[0,1]:+.2f}")
