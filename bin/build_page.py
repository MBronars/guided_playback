"""Build the style-1 page from every run under runs/ (rollouts + any replay_* results).
  python bin/build_page.py [--runs runs] [--out site] [--title "Guided Playback"]"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from gp.viewer.build import build
ap = argparse.ArgumentParser(); ap.add_argument("--runs", default=os.path.join(os.path.dirname(HERE), "runs")); ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "site"))
ap.add_argument("--title", default="Guided Playback"); ap.add_argument("--max_runs", type=int, default=6); ap.add_argument("--subtitle", default="Guided streaming rollouts of the pick/place, pushing, moulding and block-pushing checkpoints, each replayed in its simulator.")
A = ap.parse_args(); build(A.runs, A.out, A.title, A.subtitle, A.max_runs)
