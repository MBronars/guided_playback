"""Inference server: holds one compiled checkpoint (adapter) on one GPU; POST /guide runs a guided rollout with the same
knobs as bin/guide.py (JSON body = knob overrides), POST /generate = /guide with iterations 0 (unguided rollouts only).
GET /status. One request at a time. Writes runs/<domain>/<name>/ like the CLI.

  python bin/serve.py --adapter paper --ckpt <ckpt> --port 8766 [--joint_limits 13.5,53.5] [--eager] [--warmup]
  curl -X POST localhost:8766/guide -d '{"domain":"pickup","episodes":"10,82","transitions":60,"steps":10,"churn":true,"fix_fps":true,"offset":"0.15,0,0","iterations":10,"lr":0.2,"lr_final":0.02,"name":"x"}'
"""
import argparse, json, os, socket, sys, threading, time, traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from gp.session import DEFAULTS, Args, load_adapter, guided_run

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="paper"); ap.add_argument("--ckpt", required=True); ap.add_argument("--port", type=int, default=8766)
ap.add_argument("--joint_limits", default=None); ap.add_argument("--eager", action="store_true"); ap.add_argument("--runs", default=os.path.join(os.path.dirname(HERE), "runs"))
ap.add_argument("--warmup", action="store_true", help="run a 2-member, 19-transition, 1-iteration guided rollout at boot (compiles the graphs)"); ap.add_argument("--warmup_domain", default=None)
S = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True
base = Args(DEFAULTS, adapter=S.adapter, ckpt=S.ckpt, joint_limits=S.joint_limits, eager=S.eager, runs=S.runs)
t0 = time.time(); AD = load_adapter(base); print(f"[serve] loaded {S.adapter} step {AD.step} in {time.time() - t0:.0f} s", flush=True)
LOCK = threading.Lock(); LOG = []


def handle(req, iterations=None):
    A = Args(base, **req)
    if iterations is not None: A["iterations"] = iterations
    if not A.name: A["name"] = time.strftime("req_%Y%m%d_%H%M%S")
    if A.churn is None: A["churn"] = True
    with LOCK:
        t = time.time(); out, res = guided_run(AD, A)
        rec = dict(name=A.name, out=out, best_mm=res["best_mm"], unguided_mm=res["trace_mm"][0], seconds=round(time.time() - t, 1), iteration_seconds=res["iteration_seconds"], peak_gib=res["peak_gib"])
        LOG.append(rec); return rec


if S.warmup:
    print("[serve] warmup", flush=True)
    handle(dict(domain=S.warmup_domain, episodes="0,0", transitions=19, steps=10, churn=True, fix_fps=True, iterations=1, name="_warmup", runs=os.path.join(S.runs, "_warmup")))


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        self._send(200, dict(adapter=S.adapter, ckpt=S.ckpt, step=AD.step, mode=AD.mode, busy=LOCK.locked(), requests=LOG[-20:]))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/guide": self._send(200, handle(req))
            elif self.path == "/generate": self._send(200, handle(req, iterations=0))
            else: self._send(404, dict(error="unknown path"))
        except Exception as e:
            traceback.print_exc(); self._send(500, dict(error=str(e)))

    def log_message(self, *a):
        pass


print(f"[serve] http://{socket.gethostname()}:{S.port}  (POST /guide, POST /generate, GET /)", flush=True)
HTTPServer(("0.0.0.0", S.port), H).serve_forever()
