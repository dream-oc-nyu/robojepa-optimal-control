"""Validate saved E001 raw contract and temporal seal ledger without scoring it."""
import json
from pathlib import Path
import numpy as np

def validate(folder, n=19):
    folder=Path(folder)
    meta=json.loads((folder/'meta.json').read_text())
    path=folder/meta['archive']
    with np.load(path,allow_pickle=False) as d:
        expected={'names':((n,), 'U'), 'kind':((n,), 'U'),
          'commanded_xyz':((n,3),'float64'), 'executed_delta14':((n,14),'float64'),
          'ee_world':((n,3),'float64'), 'sim_seconds':((n,),'float64'),
          'rewritten':((n,),'bool'), 'images':((n,2,240,320,3),'uint8')}
        for key in ['z_real','z_pred_cmd','z_pred_exec']:
            expected[key]=((n,2,15,20,1664),'float32')
        assert set(d.files)==set(expected),d.files
        for key,(shape,dtype) in expected.items():
            a=d[key]; assert a.shape==shape,(key,a.shape)
            assert a.dtype.kind=='U' if dtype=='U' else a.dtype==np.dtype(dtype),(key,a.dtype)
            if a.dtype.kind not in 'US': assert np.isfinite(a).all(),key
        assert len(set(d['names']))==n
        assert meta['embodiment_id']==3 and meta['future_slots']==[7,8,9]
        assert all(r['seal_time_ns']<r['execution_begin_ns'] for r in meta['trials'])
        assert all(r['seal_sha256']==r['sealed_readback_sha256'] for r in meta['trials'])
        assert meta['sanity']['goal_self_l1']==0
        assert meta['sanity']['neutral_goal_cost_difference']<1e-4
        assert meta['reset']['neutral_repeat_physics_max_abs']==0
        if n==19:
            assert list(d['kind']).count('neutral')==2
            assert list(d['kind']).count('onray')==8
            assert list(d['kind']).count('offray')==8
            assert list(d['kind']).count('goal')==1
    return {'archive':str(path),'candidates':n,'valid':True}

if __name__=='__main__':
    import sys
    print(json.dumps(validate(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 19)))
