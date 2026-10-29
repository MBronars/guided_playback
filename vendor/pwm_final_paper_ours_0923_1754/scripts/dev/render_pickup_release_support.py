"""Show the gap between sparse-point release proxies and collision-mesh clearance."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np,torch


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path);parser.add_argument('--dependency-path',action='append',default=[])
    args=parser.parse_args();root=args.root.resolve();out=root/'place_release_report_v1';out.mkdir(exist_ok=True)
    import sys
    for path in args.dependency_path:sys.path.append(str(Path(path).resolve()))
    from scipy.spatial import ConvexHull
    sample=json.loads((root/'place_release_support_v1/report_v2.json').read_text());candidates=json.loads((root/'place_release_candidates_v1/summary.json').read_text())['results'];tasks={v['candidate']:v for v in json.loads((root/'place_release_candidates_v1/task_audit_v2.json').read_text())['results']}
    point=np.array([min(v['original']['final6_min_finger_gap_mm']) for v in candidates]);mesh=np.array([min(v['last6_mesh_gap_min_mm']) for v in candidates]);labels=[str(v['original']['asset_id']) for v in candidates];x=np.arange(len(candidates))
    fig,ax=plt.subplots(figsize=(11,4.5));ax.bar(x-.18,point,.36,label='Sparse sampled points',color='#0072b2');ax.bar(x+.18,mesh,.36,label='Collision component surfaces',color='#d55e00');ax.axhline(5,color='#cc0077',ls=':',label='Required clearance: 5 mm')
    ax.set(xticks=x,xticklabels=labels,xlabel='Recorded asset ID',ylabel='Minimum final-interval finger clearance (mm)',title='Eight apparent releases from point distances; only one passes the mesh clearance test')
    ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'point_vs_mesh_release.png',dpi=140);plt.close(fig)
    table='<table><tr><th>Asset</th><th>Point / mesh clearance</th><th>Final goal error</th><th>Complete primary task</th><th>Failing criteria</th></tr>'
    for item,p,m in zip(candidates,point,mesh):
        task=tasks[item['name']];failures=[k for k,v in task['validation']['residuals_m'].items() if v>0]
        table+=f'<tr><td>{item["original"]["asset_id"]}</td><td>{p:.2f} / {m:.2f} mm</td><td>{task["final_point_error_mm"]:.2f} mm</td><td>{"PASS" if task["validation"]["accepted"] else "FAIL"}</td><td>{", ".join(failures)}</td></tr>'
    # Show the only mesh-clear candidate; this is explicitly a recorded segment.
    clear=[v for v in candidates if v['collision_mesh_final6_clear']];assert len(clear)==1
    item=clear[0];directory=root/'place_release_candidates_v1'/item['name'];movie=out/'recorded_mesh_clear_candidate.gif'
    if not movie.exists():
        b=torch.load(directory/'input.pt',weights_only=False,map_location='cpu');g=torch.load(directory/'collision_geometry.pt',weights_only=False,map_location='cpu');poses=b.body_poses[0].numpy();task=tasks[item['name']];goal=np.array(task['target']);support=task['support_z'];body=b.particle_body[0].numpy();points=b.positions[0].numpy();colors=np.array(['#e69f00','#888888','#0072b2','#009e73']);parts={}
        for k in range(4):
            vertices=g['bodies'][k]['vertices'].numpy();triangles=[]
            for planes in g['bodies'][k]['planes']:
                p=planes.numpy();v=np.unique(vertices[(vertices@p[:,:3].T+p[:,3]).max(-1)<=2e-5],axis=0);triangles.append(v[ConvexHull(v).simplices])
            parts[k]=np.concatenate(triangles)
        cloud=np.concatenate([points.reshape(-1,3),goal[None]]);center=(cloud.min(0)+cloud.max(0))/2;span=max(np.ptp(cloud,axis=0).max()*1.15,.2)
        fig=plt.figure(figsize=(11,5.8));axes=[fig.add_subplot(1,2,i+1,projection='3d') for i in range(2)];select=np.concatenate([np.where(body==k)[0][::max(1,int((body==k).sum())//200)] for k in range(4)]);scatter=axes[0].scatter(*points[0,select].T,c=colors[body[select]],s=8);polys=[]
        for k in range(4):
            poly=Poly3DCollection(parts[k],facecolors=colors[k],edgecolors='none',alpha=.75 if k==0 else .55);axes[1].add_collection3d(poly);polys.append(poly)
        for ax,title in zip(axes,['Recorded sampled particles','Matched collision components']):
            ax.set(xlim=(center[0]-span/2,center[0]+span/2),ylim=(center[1]-span/2,center[1]+span/2),zlim=(center[2]-span/2,center[2]+span/2),title=title,xticks=[],yticks=[],zlabel='height (m)');ax.view_init(23,-65);ax.set_box_aspect((1,1,1));ax.scatter(*goal,c='#cc0077',marker='x',s=50)
            xx,yy=np.meshgrid(center[0]+np.array([-.5,.5])*span,center[1]+np.array([-.5,.5])*span);ax.plot_surface(xx,yy,np.full_like(xx,support),color='#999999',alpha=.1)
        title=fig.suptitle('');fig.subplots_adjust(top=.77,bottom=.04,left=.01,right=.99,wspace=.03)
        def update(t):
            scatter._offsets3d=tuple(points[t,select].T)
            for k,poly in enumerate(polys):poly.set_verts(parts[k]@poses[t,k,:3,:3].T+poses[t,k,:3,3])
            title.set_text(f'Recorded training segment · asset 4888 · {(t-3)*.05:.2f}s relative to context anchor\nOrange: object; gray/blue/green: gripper. Pink cross: requested object position.\nFinal mesh clearance passes; point-goal error {task["final_point_error_mm"]:.1f} mm exceeds the 20 mm criterion.')
            return [scatter,title]+polys
        FuncAnimation(fig,update,frames=len(points),interval=85).save(movie,writer=PillowWriter(fps=12));update(len(points)-1);fig.savefig(out/'recorded_mesh_clear_candidate_final.png',dpi=140);plt.close(fig)
    page=f'''<!doctype html><meta charset="utf-8"><title>Placing release support</title><style>body{{font:16px system-ui;line-height:1.5;margin:30px;max-width:1400px}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}table{{border-collapse:collapse}}</style><h1>How much evidence of released placing is in these training segments?</h1><p>A fixed seeded sample of eight training shards and four batch keys per shard yields {sample['segments']} stored segments across {sample['distinct_assets']} assets after the registered training exclusions. These are not independent full episodes. No outcome was used to choose the initial sample.</p><p>Eight segments pass a sparse-point clearance proxy. All eight were then checked with collision components whose asset/template correspondence was verified. Seven still have finger contact; one has at least5mm clearance at both fingers throughout the final six frames. The other98 segments were not mesh audited, so this is not a full-corpus release rate.</p><p>The one mesh-clear segment also passes the primary rest, settling, collision, joint and orientation checks. Its held point-goal criterion fails (final error25.90mm). This demonstrates some release behavior in the data; it does not establish abundant precise released-placing supervision. The video below is recorded data, not a generated prediction.</p><p><a href="../index.html">Main guidance report</a> · <a href="../place_release_support_v1/report_v2.json">Seeded segment audit and exclusions</a> · <a href="../place_release_candidates_v1/summary.json">Every candidate mesh check</a> · <a href="../place_release_candidates_v1/task_audit_v2.json">Complete task checks</a></p><img src="point_vs_mesh_release.png">'''
    (out/'index.html').write_text(page+table+'</table><img src="recorded_mesh_clear_candidate.gif">');print('Rendered release support audit')


if __name__=='__main__':main()
