#!/bin/bash
# Download DeepMind Learning-to-Simulate Water-3D (arXiv 2002.09405): metadata.json and the
# train / valid / test TFRecords (75.1 / 7.5 / 7.5 GB), resumable, each checked against the
# server's Content-Length, group-readable. Then materialize with scripts/unified/materialize_water3d.py.
#   bash scripts/unified/download_water3d.sh [target_dir] [files...]
# default target: the registered location; default files: all four.
set -u
T=${1:-/data/group_data/katefgroup/datasets/deepmind_lts/Water-3D}
shift || true
FILES=${*:-metadata.json train.tfrecord valid.tfrecord test.tfrecord}
B=https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/Water-3D
mkdir -p "$T" && cd "$T" || exit 1
status=0
for f in $FILES; do
  echo "[dl] $f start $(date -u +%FT%TZ)"
  curl -sS -L -C - -o "$f" "$B/$f" || echo "[dl] $f curl exit=$?"
  want=$(curl -sI "$B/$f" | grep -i "^content-length" | awk '{print $2}' | tr -d '\r')
  have=$(stat -c %s "$f")
  if [ "$have" = "$want" ]; then echo "[dl] $f OK $have bytes"; else echo "[dl] $f MISMATCH have=$have want=$want"; status=1; fi
  chmod g+r "$f"
done
exit $status
