"""Check queue entries the way they will actually be parsed: qworker str.format -> bash -c (word splitting, quote removal) ->
with_server*.sh `eval "$@"` (second parse) -> the driver's json.loads(--extra). Exits non-zero on the first bad entry.
  python sweeps/queue/check_cmds.py <queue>/pending/a00*.json"""
import json, shlex, sys
bad = 0
for f in sys.argv[1:]:
    cmd = json.load(open(f))["cmd"].format(GPU="0", HOST="host", URL="u", CORES="0", JOB="0")
    try:
        lvl1 = shlex.split(cmd.replace("\\$URL", "URLVAR"))            # bash -c
        args = shlex.split(" ".join(lvl1[4:])) if lvl1[1].startswith("sweeps/queue/with_server") else lvl1   # eval "$@"
        if "--extra" in args:
            json.loads(args[args.index("--extra") + 1])
        print("ok ", f.split("/")[-1], args[args.index("--extra") + 1] if "--extra" in args else "")
    except Exception as e:
        bad += 1; print("BAD", f.split("/")[-1], repr(e))
sys.exit(1 if bad else 0)
