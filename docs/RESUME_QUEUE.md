# Queue state when babel-o9-24's GPU driver wedged (2026-09-25 ~05:33 EDT)

Symptoms: new CUDA contexts never initialise on either GPU, nvidia-smi hangs, an Isaac floating-replay process (pid 1988397) ignores
SIGKILL, dmesg shows a kernel stack inside nvidia rm_ioctl. Both pickup servers (:8768 GPU 0, :8769 GPU 1) still answer HTTP but any
new request would hang. Other interactive allocations (l9-20 job 10560764, l9-16 job 10560734) had both GPUs at 85-100 % (other agents).

## Done (on the page V79DtuKonyoLdmTmWer5Cm)
lifts: jar (side), carton, bottle, can; obstacle: carton over cereal box (40 + 80 it), carton fwd35 twin, carton on cabinet (upright).
failures: jar fwd (reach), pasta box (too close), tea box, juice carton, tomato, tuna can, mustard bottle, bowl 15 cm.

## Interrupted / not started (relaunch on a working node; every chain skips existing result.json / analysis.json)
- gal_carton_air_cab: s0 planned; s1 plan may or may not have been written; floating replay of s0 HUNG (delete runs/pickup/gal_carton_air_cab_s0/replay_isaac if analysis.json is missing).
- gal_bowl1_air_fr24: s0 + s1 planned (s1 result.json may be missing); floating s0 done; continue with gallery_chain2 (arm pick).
- stage2c (diverse objects, GPU 1 server): wineglass s0+s1 and lemon s0+s1 planned (no replays yet); glass, winebottle, oilbottle, ketchup, saltshaker, orange, pear, mokapot not started.
- stage3c: carton over cracker row (80 it); stage3d: carton with hand start 30 cm straight above (needs server restart: gp/session.py changed
  for object_mesh / hand_start=above; the :8769 server already has object_mesh but not hand_start=above).
- stage3b_jar (jar over cereal box sideways, jar on cabinet, twins) not started.

## To resume on node X
1. start servers: `CUDA_VISIBLE_DEVICES=<g> python -u bin/serve.py --adapter clean_pickup --ckpt <cont200k final.pt> --port 8768 --warmup --warmup_transitions 60 --warmup_episodes 1,1,1,1 --warmup_iterations 1`
2. edit SRV host in sweeps/gallery_chain2.sh / gallery_repick.sh (babel-o9-24 -> X) or export GP_SRV; Isaac + EGL renders use --gpu 1 (replay_arm.py, bin/replay.py).
3. rerun the stage scripts (they wait on marker lines: remove the `until grep -q ...` lines or touch the marker logs).

## 2026-09-25 16:00 EDT: babel-p9-28 (job 10563761) wedged the same way
Three Isaac floating replays (jar_side30 s0, pear s1, oilbottle s0) started within the same minute (15:56) and all stalled at
startup, plus the EGL render of gal_jar_above_air started then; new srun steps on the node hang, HTTP servers still answer.
On o9-24 the wedge also coincided with two Isaac launches in the same second. **Mitigation now in the scripts:** every Isaac /
EGL launch (bin/replay.py in gallery_chain2.sh, replay_arm.py in gallery_repick.sh) runs under
`flock /tmp/gp_isaac_$(hostname).lock` so only one starts at a time per node.

Done on p9-28 before the wedge (picks exist): carton_air_cab, carton_over_crackers, carton_above_air, wineglass, winebottle;
failures: glass, orange, jar_over_cereal. Planned but not replayed: oilbottle s0 (+s1 missing), pear s0+s1, jar_side30 s0,
jar_above (picked, render stuck), bottle_above s0 (+s1?), lemon s0+s1, bowl1 s0+s1, saltshaker s0. Not started: ketchup,
mokapot, jar_on_cabinet, jar_air_cab, can_above, wineglass_above.

## 2026-09-25 16:40: babel-p9-28 hung the same way (~3 h after start)
srun steps stop returning, CUDA probes hang, Isaac floating replays stall after the carb plugin warnings (pear s1 stuck since 15:56).
Both hangs (o9-24 05:33, p9-28 ~15:56) happened while SEVERAL Isaac Sim instances ran concurrently on one node (o9-24: two floating
replays started the same second; p9-28: 3-4 queues each launching Isaac). Mitigation now in the scripts: every Isaac / render launch
goes through `flock /tmp/gp_isaac.lock` (node-local) so only ONE Isaac process runs per node. Servers (planning) stay parallel.
State: done on p9-28: carton air-cab twin, carton over crackers, carton above start, wine glass, wine bottle (all on the page);
failed/recorded: drinking glass, orange, jar over cereal. Planned but not replayed: oil bottle s0, pear s0+s1 (s0 floating done),
jar side30 s0, bowl1 s0+s1 (s1 planned server-side), saltshaker s0 (server-side), lemon s0+s1. Not started: ketchup, moka pot,
jar on cabinet + air twin, overhead-start cases (other session added them).
