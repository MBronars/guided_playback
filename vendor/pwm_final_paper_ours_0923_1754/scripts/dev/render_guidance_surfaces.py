"""Show identical collision surfaces for a model plan and independent replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from scripts.dev.render_guidance_push import draw_obstacles, obstacle_vertices
from utils.reproducibility import file_sha256


def obstacle_surface_points(box, spacing=.008):
    """Point-only display of the supplied box surfaces, including its rotation."""
    size=np.asarray(box['size']);faces=[]
    axes=[np.linspace(-s/2,s/2,max(2,int(np.ceil(s/spacing))+1)) for s in size]
    for fixed in range(3):
        other=[i for i in range(3) if i!=fixed]
        for sign in (-1,1):
            grid=np.array(np.meshgrid(axes[other[0]],axes[other[1]],indexing='ij')).reshape(2,-1).T
            points=np.empty((len(grid),3));points[:,other]=grid;points[:,fixed]=sign*size[fixed]/2
            faces.append(points)
    points=np.unique(np.concatenate(faces),axis=0)
    return points@np.asarray(box.get('rotation',np.eye(3))).T+np.asarray(box['centre'])


def local_triangles(body):
    from scipy.spatial import ConvexHull
    vertices=np.asarray(body['vertices'],dtype=float);parts=[]
    for planes in body['planes']:
        planes=np.asarray(planes);v=np.unique(vertices[(vertices@planes[:,:3].T+planes[:,3]).max(-1)<=2e-5],axis=0)
        hull=ConvexHull(v);faces=hull.simplices.copy()
        normal=np.cross(v[faces[:,1]]-v[faces[:,0]],v[faces[:,2]]-v[faces[:,0]])
        flip=(normal*hull.equations[:,:3]).sum(-1)<0;faces[flip]=faces[flip][:,[0,2,1]]
        parts.append(v[faces])
    return np.concatenate(parts)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--keys',required=True,help='Comma-separated completed gallery export keys')
    parser.add_argument('--criterion',choices=['task','fidelity'],default='task')
    args=parser.parse_args();root=args.root.resolve();folder=root/'surface_figures';folder.mkdir(exist_ok=True)
    lines=['# Surface views: prediction and execution','',
        'These movies use the actual convex components used by the collision audit. '+
        'They are geometric proxies, not textured visual meshes. Orange is the object; grey is the palm; '+
        'blue and green are the fingers. The red cross and wire sphere mark the requested body-origin position and 20 mm tolerance.',
        '', 'Both panels use the same initial observation, geometry, camera, playback rate and gripper plan. '+
        'The right panel executes only the gripper commands; physics determines object motion. The measured timestep is 0.25 ms.',
        '', ('[All model comparisons](GALLERY.md) · [Paired-timestep execution checks](EXECUTION_GALLERY.md)' if args.criterion=='task'
             else '[All ordinary draws and fidelity criteria](README.md) · [Guidance comparisons](../GALLERY.md)'), '']
    if args.criterion=='fidelity':
        lines[2]='Unguided positive examples selected for illustration after auditing the entire bank. No external destination is requested here; the panels report object lift. These are collision surfaces, not textured visual meshes.'
    results=[]
    for key in args.keys.split(','):
        export=root/'exports'/key;replay=root/'replays'/key/'dt00025'
        if not (export/'trajectory.npz').exists() or not (replay/'result.json').exists():
            lines.extend(['## '+key,'','Pending.','']);continue
        protocol=json.loads((export/'protocol.json').read_text());audit=json.loads((root/'replays'/key/'audit.json').read_text())
        for path in protocol['arguments']['collision_vendor']:sys.path.append(path)
        g=torch.load(protocol['arguments']['geometry'],map_location='cpu',weights_only=False)
        if file_sha256(protocol['arguments']['geometry'])!=protocol['geometry_sha256']:raise ValueError('Changed geometry')
        planned=np.load(export/'trajectory.npz');executed=np.load(replay/'execution.npz');C=int(planned['context_len'])
        np.testing.assert_array_equal(executed['desired_R'],planned['R'][0,C-1:])
        np.testing.assert_array_equal(executed['desired_T'],planned['T'][0,C-1:])
        tracks=[(planned['R'][0,C-1:],planned['T'][0,C-1:]),(executed['model_R'],executed['model_T'])]
        signature=hashlib.sha256((file_sha256(export/'trajectory.npz')+file_sha256(replay/'execution.npz')+file_sha256(__file__)+args.criterion).encode()).hexdigest()[:12]
        movie=folder/f'{key}_{signature}.gif';still=movie.with_suffix('.png')
        if not movie.exists():render(key,protocol,g,tracks,movie,still,show_goal=args.criterion=='task')
        errors=[float(np.linalg.norm(t[-1,0]-np.asarray(protocol['target'][0]))*1000) for r,t in tracks]
        outcome=(f'Model final error: **{errors[0]:.1f} mm**. Physics final error: **{errors[1]:.1f} mm**. '+
            f'Complete paired-timestep execution pass: **{audit["qualified_task_pass"]}**.') if args.criterion=='task' else (
            f'Passed the ordinary-sample replay-fidelity diagnostic: **{audit["replay_fidelity_pass"]}**. '+
            'This is an unguided lift, not success at a new requested destination.')
        lines.extend(['## '+key,'',f'![{key}]({movie.relative_to(root)})','',outcome,
            '',f'[Final frame]({still.relative_to(root)}) · [Execution audit](replays/{key}/audit.json)',''])
        results.append(dict(key=key,movie=str(movie.relative_to(root)),still=str(still.relative_to(root)),
            plan_sha256=file_sha256(export/'trajectory.npz'),execution_sha256=file_sha256(replay/'execution.npz')))
    lines+=['The independent engine uses a floating hand and approximate inertia/contact parameters. '+
        'These movies do not establish source-simulator or real-robot performance.','']
    (root/'SURFACE_GALLERY.md').write_text('\n'.join(lines))
    (root/'surface_gallery_summary.json').write_text(json.dumps(dict(results=results),indent=2)+'\n')


def render(key,protocol,geometry,tracks,movie,still,*,show_goal=True,panel_titles=None,heading=None,subtitle=None,
           point_templates=None,point_tracks=None,frame_notes=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    keys=sorted(geometry['bodies'])
    point_mode=point_templates is not None
    triangles=([np.asarray(point_templates[k]) for k in keys] if point_mode else
               [local_triangles(geometry['bodies'][k]) for k in keys])
    goal=np.asarray(protocol['target'][0]);obstacles=protocol.get('obstacle_spec')
    clouds=[]
    for panel,(R,T) in enumerate(tracks):
        if point_tracks is not None:
            clouds.extend(points.reshape(-1,3) for points in point_tracks[panel].values())
            continue
        for k,tri in zip(keys,triangles):
            lo=tri.reshape(-1,3).min(0);hi=tri.reshape(-1,3).max(0)
            corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
            clouds.append(np.einsum('fij,nj->fni',R[:,k],corners).reshape(-1,3)+np.repeat(T[:,k],8,axis=0))
    for box in (obstacles or {}).get('boxes',[]):clouds.append(obstacle_vertices(box))
    pts=np.concatenate(clouds+[goal[None]]);lo,hi=pts.min(0),pts.max(0);centre=(lo+hi)/2;span=max((hi-lo).max()*1.15,.3)
    colours=['#dfa52c','#747b84','#298cc8','#26a183'];light=np.array([-.4,-.5,1.]);light/=np.linalg.norm(light)
    cols=3 if point_mode and len(tracks)==3 else 2;rows=(len(tracks)+cols-1)//cols
    fig=plt.figure(figsize=(6*cols,6.2*rows),facecolor='#fbfaf7');collections=[];labels=[]
    theta=np.linspace(0,2*np.pi,65)
    for j,(R,T) in enumerate(tracks):
        ax=fig.add_subplot(rows,cols,j+1,projection='3d',proj_type='ortho');ax.set_facecolor('#fbfaf7')
        bodies=[]
        for tri,color in zip(triangles,colours):
            if point_mode:
                artist=ax.scatter(*tri.T,c=color,s=1.5,depthshade=False,linewidths=0)
            else:
                artist=Poly3DCollection(tri,facecolor=color,linewidth=0,zsort='average');ax.add_collection3d(artist)
            bodies.append(artist)
        collections.append(bodies)
        floor=geometry['metadata']['support_z'];x,y=np.meshgrid(centre[0]+span*np.array([-.7,.7]),centre[1]+span*np.array([-.7,.7]))
        if point_mode:
            edge=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5],[-.5,-.5]])*span+centre[:2]
            ax.plot(edge[:,0],edge[:,1],np.full(5,floor),color='#c7c3bc',lw=.6)
        else:ax.plot_surface(x,y,np.full_like(x,floor),color='#d7d3cb',alpha=.23,shade=False)
        if show_goal:
            ax.scatter(*goal,marker='x',color='#bf174d',s=65,linewidth=2)
            for axis in range(3):
                circle=np.zeros((len(theta),3));other=[i for i in range(3) if i!=axis]
                circle[:,other[0]]=.02*np.cos(theta);circle[:,other[1]]=.02*np.sin(theta)
                ax.plot(*(circle+goal).T,color='#bf174d',alpha=.45,lw=.7)
        if not point_mode:ax.plot(*T[:,0].T,color='#ba8420',alpha=.28,lw=1)
        if point_mode:
            for box in (obstacles or {}).get('boxes',[]):
                ax.scatter(*obstacle_surface_points(box).T,c='#8c4db5',s=2,depthshade=False,linewidths=0)
        else:draw_obstacles(ax,obstacles)
        ax.set(xlim=(centre[0]-span/2,centre[0]+span/2),ylim=(centre[1]-span/2,centre[1]+span/2),
            zlim=(centre[2]-span/2,centre[2]+span/2));ax.set_box_aspect((1,1,1));ax.view_init(24,-64);ax.set_axis_off()
        ax.set_title(panel_titles[j] if panel_titles else 'Model prediction' if j==0 else 'Independent physics replay',fontsize=15,pad=-5)
        labels.append(ax.text2D(.14,.025,'',transform=ax.transAxes,fontsize=11))
    names={'8270':'Bottle pick','7803':'Teacup placement','8026':'Spice jar pick','4943':'Bowl placement'}
    name=next((label for token,label in names.items() if token in key),'Generated task')
    a=protocol['arguments']
    setting='unguided sample' if 'ordinary' in key or a['method'] in ('sample','best_of_k') else f'{a["iterations"]} updates, noise cap {a["max_noise_rms"]:g}'
    if obstacles:setting+=' with obstacle avoidance'
    title=fig.suptitle('',fontsize=13);fig.subplots_adjust(left=0,right=1,top=.87,bottom=.03,wspace=-.02)
    def update(frame):
        for j,((R,T),bodies) in enumerate(zip(tracks,collections)):
            f=min(frame,len(T)-1)
            for k,tri,artist,color in zip(keys,triangles,bodies,colours):
                world=point_tracks[j][k][f] if point_tracks is not None else tri@R[f,k].T+T[f,k]
                if point_mode:artist._offsets3d=tuple(world.T)
                else:
                    artist.set_verts(world)
                    normal=np.cross(world[:,1]-world[:,0],world[:,2]-world[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-12)
                    shade=.48+.52*np.maximum(normal@light,0);artist.set_facecolor(np.array(to_rgb(color))[None]*shade[:,None])
            if show_goal:
                error=np.linalg.norm(T[f,0]-goal)*1000;labels[j].set_text(f'Target distance: {error:.1f} mm')
            else:labels[j].set_text(f'Object lift: {(T[f,0,2]-T[0,0,2])*1000:.1f} mm')
            if frame_notes and frame_notes[j]:labels[j].set_text(labels[j].get_text()+'\n'+frame_notes[j][f])
        title.set_text((heading if heading else name+' · '+setting)+f'  ·  {frame*.05:.2f} s\n'+
            (subtitle if subtitle is not None else
             'Top: model trajectories. Bottom: physics with the same barrier.' if panel_titles else 'Same requested gripper plan; compare the object motion'))
    count=max(len(t) for r,t in tracks)
    temp=movie.with_name(movie.stem+'.pending.gif')
    FuncAnimation(fig,update,frames=count,interval=50).save(temp,writer=PillowWriter(fps=20));temp.replace(movie)
    update(count-1);fig.savefig(still,dpi=150,facecolor=fig.get_facecolor());plt.close(fig)


if __name__=='__main__':main()
