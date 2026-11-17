"""E001 real-simulator sweep. No decoder, training, compiled path or science scoring.

Restore complete MuJoCo integration state plus Python controller/cache state;
validate a nonzero action twice and neutral after that excursion before use.
Commanded predictions are fsynced and hashed before each physical execution.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
from sweep_contract import candidates

ROOT=Path(os.environ['SESSION_ROOT'])
SHAPE=(2,15,20,1664)
RJ='f93106f5f4a451fd526d51778aa8ab7c852eb824'
SOURCES={'robo_jepa':RJ,'robosuite':'5ce6643f3092639d08f7b0f90ed1c6a84f50552c',
 'robocasa':'456174f62b89b8fca99eaaf33949c29fec9cfc2a','vjepa2':'204698b45b3712590f06245fbfba32d3be539812'}
WEIGHTS={'predictor':'bdd77b212d81e11438fd6bd85525bed72ff1f47a6b3211ac925ddfbc7c8351a6',
 'encoder':'7aae1a3c7a31d258af9c985388b5d2f20587469380f2e11b54d7876ac8cfe58a'}
CONTROLLER={'mode':'blocking-until-position-tolerance', 'position_tolerance_m':.001,'max_substeps':200,
 'evidence':['scripts/deployment/examples/reproduce_reach.sh:45-51 (--blocking_control --blocking_pos_tol 0.001 --blocking_max_steps 200)',
 'scripts/deployment/robots/robocasa/generate_reach_invariants_golden.py:114-144 (_build_env returns ReachBlockingRoboCasaEnv)',
 'scripts/deployment/robots/robocasa/blocking_robot_env.py:285-401 (target offset; residual test; unchanged native step)'],
 'comment_caveat':'Executable residual termination tests position only at lines 353-355, despite orientation-convergence prose.',
 'command_mapping':'base-frame metric XYZ passed directly, not divided by four',
 'settling':'nine fixed-four-substep inner neutral ticks as pilot; last two real observations provide context'}

class CollectionCap(Exception): pass

def digest(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def json_write(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with open(tmp,'w') as f:
        json.dump(obj,f,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)

def np_write(path, **arrays):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with open(tmp,'wb') as f:
        np.savez(f,**arrays); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)

def command(x):
    return {'eef_translation':np.asarray(x,dtype=np.float64),'eef_rotation':np.zeros(3),'gripper':np.array([-1.])}

def images(obs): return np.stack([obs['img_0'],obs['img_1']])

# Preserve only state, not simulator/renderer handles or sensor callables.
def is_plain(x):
    if x is None or isinstance(x,(str,bytes,bool,int,float,np.number,np.ndarray)): return True
    if isinstance(x,(list,tuple)): return all(is_plain(v) for v in x)
    if isinstance(x,dict): return all(is_plain(v) for v in x.values())
    return False

class Snapshot:
    def __init__(self,inner):
        import mujoco
        self.inner=inner; self.sim=inner._env.sim
        self.model=self.sim.model._model; self.data=self.sim.data._data
        self.spec=mujoco.mjtState.mjSTATE_INTEGRATION
        self.vector=np.empty(mujoco.mj_stateSize(self.model,self.spec),dtype=np.float64)
        mujoco.mj_getState(self.model,self.data,self.vector,self.spec)
        self.python=[]; seen=set()
        def visit(obj):
            if id(obj) in seen: return
            seen.add(id(obj))
            if isinstance(obj,dict):
                for v in obj.values(): visit(v)
            elif isinstance(obj,(list,tuple)):
                for v in obj: visit(v)
            elif hasattr(obj,'__dict__'):
                module=type(obj).__module__
                if not (obj is inner or obj is inner._env or module.startswith(('robosuite.robots','robosuite.controllers','robosuite.utils.observables','robosuite.utils.buffers','robosuite.models.grippers'))): return
                fields={k:copy.deepcopy(v) for k,v in vars(obj).items() if is_plain(v)}
                self.python.append((obj,fields))
                for v in vars(obj).values():
                    if not is_plain(v): visit(v)
        visit(inner); visit(inner._env)
        self.rng=copy.deepcopy(np.random.get_state())
    def restore(self):
        import mujoco
        for obj,fields in self.python:
            for k,v in fields.items(): setattr(obj,k,copy.deepcopy(v))
        np.random.set_state(copy.deepcopy(self.rng))
        mujoco.mj_setState(self.model,self.data,self.vector,self.spec)
        self.sim.forward()
    def physics(self):
        # Includes controls, warmstart, applied forces, mocap and all time/DOF state.
        import mujoco
        vec=np.empty_like(self.vector)
        mujoco.mj_getState(self.model,self.data,vec,self.spec)
        return vec

def build_context(seed):
    from scripts.deployment.robots.robocasa.generate_reach_invariants_golden import _build_env
    wrapper=_build_env(0,'pretrain'); inner=wrapper._env
    obs,_=wrapper.reset(mode='init',seed=0,reach_task_seed=seed,reach_episode_idx=0)
    for _ in range(8): obs,_,_,_=inner.step(command(np.zeros(3)))
    h0=copy.deepcopy(obs)
    h1,_,_,info=inner.step(command(np.zeros(3)))
    assert info['sim_steps']==4
    return wrapper, [h0,copy.deepcopy(h1)]

class Instrument:
    def __init__(self,wrapper):
        self.wrapper=wrapper; self.inner=wrapper._env; self.records=[]
        self.original=self.inner._reach_apply_orientation_hold
        def observed(sub):
            before=sub.copy(); result=self.original(sub)
            self.records.append({'input':before.tolist(),'output':result.copy().tolist(),
               'translation_rewritten':bool(np.any(before[5:8]!=result[5:8])),
               'rotation_rewritten':bool(np.any(before[8:11]!=result[8:11]))})
            return result
        self.inner._reach_apply_orientation_hold=observed
    def step(self,xyz):
        from robots.robocasa.robot_env import _REACH_WORKSPACE_R
        self.records=[]
        inner=self.inner; sim=inner._env.sim
        target=inner._get_eef_pos()+inner._get_base_world_R()@xyz
        projected=bool(np.linalg.norm(target-inner._reach_home_eef_pos)>_REACH_WORKSPACE_R)
        before=float(sim.data.time)
        obs,_,_,info=self.wrapper.step(command(xyz))
        seconds=float(sim.data.time)-before
        receipt={'info':info,'target_projected':projected,'substeps':copy.deepcopy(self.records)}
        rewritten=projected or any(r['translation_rewritten'] or r['rotation_rewritten'] for r in self.records)
        return obs,seconds,rewritten,receipt

def validate_restore(wrapper,snapshot,probe_xyz):
    """Nonzero excursion/replay guards against a vacuous zero-step neutral test."""
    endpoints=[]; states=[]
    for _ in range(2):
        snapshot.restore()
        obs,_,_,_=wrapper.step(command(probe_xyz))
        endpoints.append(obs['proprio_robocasa'].copy()); states.append(snapshot.physics())
    error=float(np.max(np.abs(states[0]-states[1])))
    proprio_error=float(np.max(np.abs(endpoints[0]-endpoints[1])))
    assert error<1e-10 and proprio_error<1e-10, ('nonzero restoration replay failed',error,proprio_error)
    snapshot.restore(); a,_,_,_=wrapper.step(command(np.zeros(3))); pa=snapshot.physics()
    wrapper.step(command(probe_xyz))
    snapshot.restore(); b,_,_,_=wrapper.step(command(np.zeros(3))); pb=snapshot.physics()
    neutral_error=float(np.max(np.abs(pa-pb)))
    assert neutral_error==0, ('neutral physics after excursion differs',neutral_error)
    assert np.array_equal(a['proprio_robocasa'],b['proprio_robocasa'])
    return {'method':'saved MuJoCo mjSTATE_INTEGRATION plus controller/robot/observable/cache Python state',
      'python_objects_restored':len(snapshot.python),'integration_state_entries':len(snapshot.vector),
      'nonzero_probe_xyz':probe_xyz.tolist(),'nonzero_repeat_physics_max_abs':error,
      'nonzero_repeat_proprio_max_abs':proprio_error,'neutral_repeat_physics_max_abs':neutral_error}

def load_model():
    import torch
    from app.robo_jepa.world_model import WorldModel
    torch.manual_seed(29)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    wm=WorldModel.load_pretrained(str(ROOT/'weights'),weights_filename='predictor.pth.tar',
        load_decoder=False,load_compiled=False,load_encoder=False,device='cuda')
    ckpt=torch.load(ROOT/'weights/predictor.pth.tar',map_location='cpu',weights_only=True)
    wm.transition_model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['predictor' if 'predictor' in ckpt else 'model'].items()},strict=True)
    del ckpt
    wm.transition_model.eval(); wm.transition_model.requires_grad_(False)
    wm.load_encoder(str(ROOT/'weights/encoder.pt'),use_proj=False,compile=False)
    wm.encoder.eval(); wm.encoder.requires_grad_(False)
    assert wm.model.embodiment_list[3]=='franka_omron_state_delta'
    assert wm.model.state_dim==wm.model.action_dim==64
    assert all(p.dtype==torch.float32 for p in wm.model.parameters())
    assert all(p.dtype==torch.float32 for p in wm.encoder.parameters())
    return wm

def encode(wm,pixels):
    import torch
    with torch.no_grad(): z=wm.encode(pixels,size=(240,320),dtype=torch.float32)
    out=z.cpu().numpy().astype(np.float32,copy=False)
    assert np.isfinite(out).all()
    return out

def model_state(wm,h):
    import torch
    import torch.nn.functional as F
    from app.robo_jepa.world_model import WorldModelState
    context_pixels=np.stack([images(o) for o in h])
    context=encode(wm,context_pixels.reshape(4,240,320,3)).reshape(1,2,2,15,20,1664)
    poses=F.pad(torch.from_numpy(np.stack([o['proprio_robocasa'] for o in h])).float().cuda()[None],(0,50))
    state=WorldModelState.from_features(torch.from_numpy(context).cuda(),actions=poses[:,1:]-poses[:,:-1],states=poses,
       view_map=torch.tensor([[0,2]],device='cuda'),embodiment_ids=torch.tensor([[3]],device='cuda'))
    return state,context,context_pixels

def predict(wm,state,xyz):
    import torch
    from robojepa_probe import differentiable_step
    with torch.no_grad():
        y,_=differentiable_step(wm,torch.from_numpy(np.asarray(xyz,dtype=np.float32)).cuda().reshape(1,1,3),state)
    out=y.cpu().numpy().reshape(SHAPE).astype(np.float32,copy=False)
    assert np.isfinite(out).all()
    return out

def parity(wm,state,xyz):
    import torch
    from robojepa_probe import translation_action,differentiable_step
    x=torch.from_numpy(np.asarray(xyz,dtype=np.float32)).cuda().reshape(1,1,3)
    with torch.no_grad():
        native,_=wm.step(translation_action(x,64),state,use_cache=False,no_grad=True)
        adapted,_=differentiable_step(wm,x,state)
        torch.testing.assert_close(native.to('cuda'),adapted,rtol=1e-5,atol=1e-5)
        error=float((native.to('cuda')-adapted).abs().max())
    return error

def collect_instance(wm,index,smoke,deadline):
    from validate_raw import validate
    started=time.time(); seed=1000+index
    names,kinds,xyz,v,axes=candidates(seed)
    if smoke:
        sel=[0,18,1]; names,kinds,xyz=names[sel],kinds[sel],xyz[sel]
    label='smoke' if smoke else f'instance-{index:02d}'
    final=ROOT/'results'/label
    if final.exists():
        validate(final,len(names)); return
    partial=ROOT/'results'/f'.{label}-{time.time_ns()}.partial'; partial.mkdir()
    wrapper=None
    try:
        wrapper,h=build_context(seed)
        snapshot=Snapshot(wrapper._env)
        reset_receipt=validate_restore(wrapper,snapshot,.01*v)
        instrument=Instrument(wrapper)
        state,context,context_pixels=model_state(wm,h)
        forward_error=parity(wm,state,.01*v)
        np_write(partial/'context.npz',images=context_pixels,features=context,
          proprio=np.stack([o['proprio_robocasa'] for o in h]),integration_state=snapshot.vector,
          start_ee_world=h[-1]['pose'][:3])
        reset_receipt['context_build_and_restore_check_seconds']=time.time()-started
        values={k:[] for k in ['z_real','z_pred_cmd','z_pred_exec','images','executed_delta14','ee_world','sim_seconds','rewritten']}
        trials=[]; neutral_physics=[]
        for j,(name,kind,cmd) in enumerate(zip(names,kinds,xyz)):
            if time.time()>=deadline: raise CollectionCap('collect wall-clock cap')
            trial_start=time.time()
            snapshot.restore()
            before_obs=wrapper._env._build_obs()
            start_error=float(np.max(np.abs(before_obs['proprio_robocasa']-h[-1]['proprio_robocasa'])))
            assert start_error<1e-10, ('restored start state mismatch',start_error)
            pred=predict(wm,state,cmd)
            seal=partial/f'sealed-{j:02d}.npy'
            with open(seal,'wb') as f:
                np.save(f,pred,allow_pickle=False); f.flush(); os.fsync(f.fileno())
            seal_sha=digest(seal); seal_time=time.time_ns()
            # A durable receipt exists before action execution, as does the prediction.
            json_write(partial/f'sealed-{j:02d}.json',{'name':str(name),'commanded_xyz':cmd.tolist(),
                'sha256':seal_sha,'sealed_time_ns':seal_time})
            execution_begin=time.time_ns()
            obs,sim_seconds,rewritten,guard=instrument.step(cmd)
            execution_end=time.time_ns()
            delta=obs['proprio_robocasa']-h[-1]['proprio_robocasa']
            realized=encode(wm,images(obs)); assert realized.shape==SHAPE
            pred_exec=predict(wm,state,delta[7:10])
            assert digest(seal)==seal_sha
            if kind=='neutral': neutral_physics.append(snapshot.physics())
            for key,val in [('z_real',realized),('z_pred_cmd',np.load(seal,allow_pickle=False)),('z_pred_exec',pred_exec),
                ('images',images(obs)),('executed_delta14',delta),('ee_world',obs['pose'][:3]),
                ('sim_seconds',sim_seconds),('rewritten',rewritten)]: values[key].append(val)
            row={'name':str(name),'commanded_xyz':cmd.tolist(),'executed_xyz':delta[7:10].tolist(),
                'seal_time_ns':seal_time,'seal_sha256':seal_sha,'sealed_readback_sha256':digest(seal),
                'execution_begin_ns':execution_begin,'execution_end_ns':execution_end,
                'sim_seconds':sim_seconds,'start_proprio_max_abs':start_error,
                'rewritten':rewritten,'guard':guard,'wall_seconds':time.time()-trial_start}
            trials.append(row); json_write(partial/'progress.json',trials)
            print(json.dumps({'instance':index,'smoke':smoke,'trial':str(name),'wall_seconds':row['wall_seconds'],
               'sim_seconds':sim_seconds,'commanded_xyz':cmd.tolist(),'executed_xyz':delta[7:10].tolist()}),flush=True)
        arrays={k:np.asarray(vv) for k,vv in values.items()}
        arrays.update(names=names,kind=kinds,commanded_xyz=xyz)
        goals=np.flatnonzero(kinds=='goal'); neutrals=np.flatnonzero(kinds=='neutral')
        goal=arrays['z_real'][goals[0]].astype(np.float64)
        costs=[float(np.abs(arrays['z_real'][n].astype(np.float64)-goal).mean()) for n in neutrals]
        sanity={'feature_shape':list(goal.shape),'goal_self_l1':float(np.abs(goal-goal).mean()),
            'neutral_goal_costs':costs,'neutral_goal_cost_difference':abs(costs[0]-costs[1])}
        assert sanity['neutral_goal_cost_difference']<1e-4,sanity
        reset_receipt['dataset_neutral_repeat_physics_max_abs']=float(np.max(np.abs(neutral_physics[0]-neutral_physics[1])))
        assert reset_receipt['dataset_neutral_repeat_physics_max_abs']==0,reset_receipt
        archive=f'{label}.npz'; np_write(partial/archive,**arrays)
        meta={'instance':index,'archive':archive,'smoke_only':smoke,'scene_seed':0,'reach_task_seed':seed,
            'reach_episode_idx':0,'direction_rng_seed':seed,'rng':'numpy.default_rng; native seeded Reach sampler for start',
            'v_g':v.tolist(),'orthogonal_directions':[a.tolist() for a in axes],
            'start_sampler':{'kitchen_split':'pretrain','reach_min_distance':.25,'reach_max_distance':.35,'reach_clear_radius':1.0},
            'weight_download_origin':'https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/ (same official bucket; original hostname TLS expired; SHA256 unchanged)',
            'controller':CONTROLLER,'source_commits':SOURCES,'weight_sha256':WEIGHTS,
            'embodiment_id':3,'future_slots':[7,8,9],'other_future_action_slots':'zero',
            'context_history':'two real observations; full measured 14-D history difference, padded to64',
            'view_map':[0,2],'fp32':True,'decoder':False,'compiled':False,'model_ticks':1,
            'preprocessing':'native WorldModel.encode: bilinear 240x320, uint8, ImageNet normalization, token layer norm',
            'native_adapter_forward_max_abs':forward_error,'reset':reset_receipt,'sanity':sanity,
            'trials':trials,'elapsed_seconds':time.time()-started,'started_unix_seconds':started,
            'completed_unix_seconds':time.time(),'deviations':[]}
        json_write(partial/'meta.json',meta)
        validate(partial,len(names))
        partial.replace(final)  # Archive, metadata, context and seals become visible together.
        print(json.dumps({'finished':str(final),'elapsed_seconds':meta['elapsed_seconds'],'sanity':sanity}),flush=True)
    finally:
        if wrapper is not None: wrapper.close()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--collect',action='store_true'); parser.add_argument('--cap-seconds',type=int,default=6000)
    args=parser.parse_args(); assert args.smoke!=args.collect
    start=time.time(); deadline=float('inf')
    if args.collect:
        clock=ROOT/'artifacts/collect-clock.json'
        if not clock.exists(): json_write(clock,{'start_unix_seconds':start,'cap_seconds':args.cap_seconds})
        config=json.loads(clock.read_text()); deadline=config['start_unix_seconds']+config['cap_seconds']
        if time.time()>=deadline:
            print('Collection deadline already passed',flush=True); return
        def expire(*_): raise CollectionCap('100-minute collection deadline reached')
        signal.signal(signal.SIGALRM,expire); signal.alarm(max(1,int(deadline-time.time())))
    try:
        wm=load_model()
        print('MODEL_READY '+str(time.time()-start),flush=True)
        for i in range(1 if args.smoke else 8): collect_instance(wm,i,args.smoke,deadline)
    except CollectionCap as e:
        json_write(ROOT/'artifacts/collection-cap.json',{'reason':str(e),'at':time.time()})
        print(str(e),flush=True)
    finally:
        signal.alarm(0)
        json_write(ROOT/'artifacts'/('smoke-summary.json' if args.smoke else 'collection-summary.json'),
            {'elapsed_seconds':time.time()-start,'completed_instances':[p.name for p in sorted((ROOT/'results').glob('instance-*')) if (p/'meta.json').exists()]})

if __name__=='__main__': main()
