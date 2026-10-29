"""Re-rank a saved candidate bank without generating new trajectories."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from guidance.grasp import StagedPickObjective
from guidance.pick_validation import selection_key
from guidance.unconditional import prediction
from utils.reproducibility import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); bank=Path(a.bank); out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    proto=json.loads((bank/'protocol.json').read_text()); args=proto['arguments']
    b=torch.load(args['input'],map_location='cpu',weights_only=False)
    g=torch.load(args['geometry'],map_location='cpu',weights_only=False)
    grasp=torch.load(args['grasp'],map_location='cpu',weights_only=False)
    proto['offline_reranking']=dict(source=str(bank.resolve()),rule='interaction feasibility before task performance',
        source_protocol_sha256=file_sha256(bank/'protocol.json'),new_sampling=False)
    (out/'protocol.json').write_text(json.dumps(proto,indent=2)+'\n')
    results=[]
    for original in json.loads((bank/'results.json').read_text()):
        started=time.perf_counter(); seed=original['seed']; directory=bank/f'seed_{seed}_candidates'
        candidates=[json.loads(f.read_text()) for f in sorted(directory.glob('draw_*.json'))]
        best=min(candidates,key=lambda r:selection_key(r['loss'],r['validation']))
        rollout=torch.load(directory/f'draw_{best["draw"]:03d}.pt',map_location='cpu',weights_only=False)
        source=np.load(bank/f'seed_{seed}.npz'); data={k:source[k] for k in source.files}
        for key in ('positions','R','T'): data[key]=rollout[key].numpy()
        diag=StagedPickObjective(b,torch.tensor(data['target']),g,grasp,
            proposal=args.get('proposal',0),approach_frames=args.get('approach_frames',12),
            close_frame=args.get('close_frame',18),hold_frames=args.get('hold_frames',6))
        record=dict(original,selected=diag.measure(prediction(rollout,b.context_len)),selected_draw=best['draw'],
            validation=best['validation'],selection_status='accepted_by_geometry_proxy' if best['validation']['accepted'] else 'no_acceptable_candidate; diagnostic fallback',
            offline_reranking_seconds=time.perf_counter()-started)
        np.savez_compressed(out/f'seed_{seed}.npz',**data)
        torch.save(rollout,out/f'seed_{seed}_selected.pt')
        (out/f'seed_{seed}.json').write_text(json.dumps(record,indent=2)+'\n'); results.append(record)
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__': main()
