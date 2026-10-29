#!/bin/bash
# replay every finished pick-only S10 run (x15z15, and the x20z10 10-it runs) in Isaac, no rendering; rebuild the small page
cd /data/user_data/mbronars/packages/guided_playback
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
for r in runs/pickup/pickonlyS10_x15z15_* runs/pickup/pickonlyS10_x20z10_*it10*; do
  [ -f $r/result.json ] || continue
  [ -f $r/replay_isaac/analysis.json ] && continue
  CUDA_VISIBLE_DEVICES=0 $P -u bin/replay.py --run $r > logs/$(basename $r)_replay.log 2>&1; echo "replay $(basename $r) exit $?"
done
rm -rf site_po_runs site_po; mkdir -p site_po_runs/pickup
for r in runs/pickup/pickonlyS10_x15z15_*it10* runs/pickup/pickonlyS10_x20z10_*it10*; do [ -f $r/result.json ] && ln -s $PWD/$r site_po_runs/pickup/; done
$P bin/build_page.py --runs site_po_runs --out site_po --title "Pickup-Only Guidance" --max_runs 12 --subtitle "Pickup-only checkpoint (cont200k final.pt), bf16 compiled, streaming S=10, Adam lr 0.1, 10 iterations, shell re-projection on vs off. Targets = object start + (15 cm x, 15 cm z) or (20 cm x, 10 cm z). Population 4 per episode. Isaac floating-hand replay overlaid (simulated layer)." | tail -1
