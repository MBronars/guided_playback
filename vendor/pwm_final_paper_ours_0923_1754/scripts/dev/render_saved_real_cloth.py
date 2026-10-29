"""Render previously saved real-cloth predictions without rerunning the model."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from scripts.dev.real_cloth_probe import measure, render_real, report
from scripts.dev.observation_probe_common import write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--source',default='validation/public_real_cloth_2026-09-14');a=ap.parse_args()
    out=Path(a.out);root=Path(a.source);cases=[]
    for path in sorted(out.glob('*__seed*.npz')):
        key=path.stem; case=json.loads((out/(key+'.json')).read_text()); receipt=case['input']
        capture=receipt['capture'];folder=root/'green_tshirt'/capture
        batch=torch.load(out/(capture+'_input.pt'),map_location='cpu',weights_only=False)
        files=sorted((folder/'segment_pcds').glob('*.pcd'))
        metadata={'files':files,'stamps':np.array([float(p.name.split('_')[1]) for p in files]),
                  'world_from_camera':np.linalg.inv(np.array(json.loads((folder/'calibration/world_to_camera_transform.json').read_text()))),
                  'origin':np.array(receipt['origin_world']),'n_cloth':receipt['cloth_particles']}
        pred=np.load(path)['prediction']
        measurements,clouds,colors,times=measure(pred,batch,metadata,receipt)
        render_real(out,key,pred,batch,metadata,receipt,measurements,clouds,colors,times)
        cases.append({'key':key,'status':'ok'});report(out,cases)
        print('RENDERED',key,flush=True)
    write_json(out/'render_status.json',{'state':'finished','cases':cases,'time_unix':time.time(),
                                       'note':'Re-rendered saved predictions; no model calls'})
    write_json(out/'status.json',{'state':'finished','cases':cases,'render_repair':'H.264 even frame dimensions'})


if __name__=='__main__':main()
