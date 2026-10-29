#!/bin/bash
# Check out tactile-genesis at the commit tactile_genesis_teachers_v1 was generated with, pull its LFS teacher
# checkpoints, and build the dexterous-hands environment (dexterous-hands/.venv, Python 3.12) with uv from
# tactile_genesis_uv.lock, the resolution v1 ran with (tactile-genesis does not track a lockfile, so an unpinned sync
# picks up newer releases).
#   bash collection/dex/launch/setup_tactile_genesis.sh <parent dir>        # creates <parent dir>/tactile-genesis
# Needs git-lfs and uv (UV=/path/to/uv if uv is not on PATH). The vendored Eden and Genesis 1.1.2 are used as shipped.
# pymeshlab is skipped: its wheel needs glibc 2.35 (the cluster has 2.34), and v1 was generated without it.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMMIT=3023bcd2d43fe947327713ddce3c2123d475735d
URL=https://github.com/neuroagents-lab/tactile-genesis.git
PARENT=$(realpath -m "${1:?parent directory}")
DEST=$PARENT/tactile-genesis
UV=${UV:-uv}
if [ -e "$DEST" ]; then echo "refusing to reuse existing $DEST" >&2; exit 1; fi

mkdir -p "$PARENT"
git init -q "$DEST"
git -C "$DEST" lfs install --local
git -C "$DEST" remote add origin "$URL"
git -C "$DEST" fetch --depth 1 origin "$COMMIT"
git -C "$DEST" checkout -q FETCH_HEAD
git -C "$DEST" lfs pull
for pt in "$DEST"/dexterous-hands/checkpoints/teachers/*.pt; do          # an LFS pointer file is ~130 bytes
    if [ "$(stat -c %s "$pt")" -lt 100000 ]; then echo "teacher checkpoint not downloaded: $pt" >&2; exit 1; fi
done

cd "$DEST/dexterous-hands"
cp "$HERE/tactile_genesis_uv.lock" uv.lock
"$UV" sync --frozen --no-install-package pymeshlab
.venv/bin/python -c "import eden, genesis, rsl_rl; print('tactile-genesis environment ready:', '$DEST')"
