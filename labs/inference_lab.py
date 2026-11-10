"""Pure Python/NumPy teaching calculations, not imports of inference engines.
Run: python labs/inference_lab.py all
"""
import argparse
import numpy as np

def packed():
    sizes=[2,1,3];computed=[4,10,0];tables=[[7,2],[1,3,5],[8]];bs=4
    req=[];positions=[];slots=[];starts=[0]
    for i,n in enumerate(sizes):
        for offset in range(n):
            p=computed[i]+offset;req.append(i);positions.append(p);slots.append(tables[i][p//bs]*bs+p%bs)
        starts.append(starts[-1]+n)
    assert slots==[8,9,22,32,33,34]
    result=dict(req_indices=req,positions=positions,query_start_loc=starts,seq_lens=[c+n for c,n in zip(computed,sizes)],logits_indices=[x-1 for x in starts[1:]],slots=slots)
    print('Packed metadata:',result);return result

def eplb():
    # Single step/layer worked example. Full policy supports history and layers.
    load=np.array([100.,20.,20.]);copies=np.ones(3,dtype=int)
    scores=[]
    for expert in range(3):
        candidate=copies.copy();candidate[expert]+=1;scores.append(max(load/candidate))
    extra=int(np.argmin(scores));copies[extra]+=1
    physical=[0,1,2,extra];per_copy=[load[e]/copies[e] for e in physical]
    order=sorted(range(4),key=lambda i:(-per_copy[i],i))
    ranks=[[],[]];rank_load=[0.,0.]
    for slot in order:
        choices=[(max(rank_load[j]+per_copy[slot] if j==r else rank_load[j] for j in range(2)),r) for r in range(2) if len(ranks[r])<2]
        _,r=min(choices);ranks[r].append(slot);rank_load[r]+=per_copy[slot]
    final=[physical[i] for rank in ranks for i in rank]
    mapping=[[i for i,e in enumerate(final) if e==expert] for expert in range(3)]
    assert final==[0,1,0,2] and rank_load==[70.,70.]
    print('Single-step EPLB example:',dict(physical_to_logical=final,logical_to_physical=mapping,rank_load=rank_load));return rank_load

def spec():
    raw=[[.9,.5],[.8,.25]];rates=[np.cumprod(a).tolist() for a in raw]
    candidates=sorted([(p,i) for i,row in enumerate(rates) for p in row],key=lambda x:(-x[0],x[1]))
    curve=[(2,1.),(4,1.2),(6,2.)];rows=[];expected=2.
    for total in range(2,7):
        if total>2:expected+=candidates[total-3][0]
        bucket,time=next((b,t) for b,t in curve if b>=total)
        rows.append((total,round(expected,6),bucket,time,expected/time))
    best=max(rows,key=lambda row:row[-1]);lengths=[1,1]
    for _,i in candidates[:best[0]-2]:lengths[i]+=1
    assert best[0]==4 and lengths==[2,2]
    computed=10+4-(3-1);submitted=11+1+1
    assert (computed,submitted)==(12,13)
    print('Hypothetical DSpark cost curve (NOT GPU measurements):',rows)
    print('Best total query:',best[0],'per-request query:',lengths,'vLLM ledger:',computed,submitted)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('lab',choices=['all','packed','eplb','spec'],nargs='?',default='all');a=p.parse_args()
    for name,fn in {'packed':packed,'eplb':eplb,'spec':spec}.items():
        if a.lab in ('all',name):fn()
    print('PASS: selected toy examples; no engine or GPU execution was performed.')
