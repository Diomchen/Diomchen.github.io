"""Deterministic CPU simulation: token scheduling + paged KV metadata.
This is an explicit teaching model, NOT nano-vllm/vLLM execution or real latency.
No model, CUDA, torch or numpy required. Output JSON + an offline step viewer.
"""
import argparse
from dataclasses import dataclass,field
from pathlib import Path
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[1]

@dataclass
class Request:
    name:str
    arrival:int
    tokens:list
    target:int
    computed:int=0
    generated:int=0
    table:list=field(default_factory=list)
    state:str='not-arrived'
    start:int=-1
    preemptions:int=0

class Pool:
    def __init__(self,count,size,prefix):
        self.size=size;self.prefix=prefix
        self.blocks=[dict(id=i,ref=0,key=None) for i in range(count)]
        self.free=list(range(count));self.hits=0
    def release(self,r):
        for bid in reversed(r.table):
            b=self.blocks[bid];b['ref']-=1
            assert b['ref']>=0
            if b['ref']==0:self.free.append(bid)
        r.table=[];r.computed=0
    def reuse(self,r):
        if not self.prefix or r.table:return
        for end in range(self.size,len(r.tokens),self.size):
            key=tuple(r.tokens[:end])
            found=next((b for b in self.blocks if b['key']==key),None)
            if found is None:break
            if found['ref']==0:self.free.remove(found['id'])
            found['ref']+=1;r.table.append(found['id']);r.computed=end;self.hits+=1
    def ensure(self,r,end):
        need=math.ceil(end/self.size)-len(r.table)
        if need>len(self.free):return False
        for _ in range(max(0,need)):
            bid=self.free.pop(0);b=self.blocks[bid];assert b['ref']==0
            b['ref']=1;b['key']=None;r.table.append(bid)
        return True
    def complete(self,r,old,end):
        for boundary in range((old//self.size+1)*self.size,end+1,self.size):
            b=self.blocks[r.table[boundary//self.size-1]]
            # Only publish complete blocks AFTER simulated forward completion.
            b['key']=tuple(r.tokens[:boundary])
    def validate(self,requests):
        expected=[0]*len(self.blocks)
        for r in requests:
            assert 0<=r.computed<=len(r.tokens)
            assert len(r.table)*self.size>=r.computed
            for i,bid in enumerate(r.table):
                expected[bid]+=1
                if (i+1)*self.size<=r.computed:
                    assert self.blocks[bid]['key']==tuple(r.tokens[:(i+1)*self.size])
        assert len(set(self.free))==len(self.free)
        for b in self.blocks:
            assert b['ref']==expected[b['id']]
            assert (b['id'] in self.free)==(b['ref']==0)

def workload():
    common=list(range(10,18))
    return [Request('A',0,common+[31],4),Request('B',1,common[:4]+[41,42,43],5),Request('C',3,common+[51,52,53,54,55],3),Request('D',6,[61,62,63,64,65],4)]

def simulate(blocks=6,budget=4,policy='mixed',prefix=True):
    reqs=workload();pool=Pool(blocks,4,prefix);trace=[];events=[]
    if blocks<max(math.ceil((len(r.tokens)+r.target-1)/4) for r in reqs):raise ValueError('Too few blocks for one request to finish; choose --blocks >= 4.')
    for step in range(1000):
        events=[];scheduled=[];spent=0
        for r in reqs:
            if r.state=='not-arrived' and r.arrival<=step:r.state='waiting';events.append(f'{r.name} arrived')
        running=[r for r in reqs if r.state=='running'];waiting=[r for r in reqs if r.state=='waiting']
        # Both are simplified policies. No multimodal/CP/spec/async/input-budget logic.
        if policy=='prefill-first':
            prefill=[r for r in running if r.generated==0 or len(r.tokens)-r.computed>1]+waiting
            order=prefill+[r for r in running if r not in prefill]
        else:order=running+waiting
        seen=[]
        for r in order:
            if r.name in seen or spent>=budget:continue
            seen.append(r.name)
            if r.state not in ('waiting','running'):continue
            phase='prefill' if r.generated==0 or r.computed<len(r.tokens)-1 else 'decode'
            if policy=='prefill-first' and scheduled and phase!=scheduled[0]['phase']:continue
            if not r.table:
                before=r.computed;pool.reuse(r)
                if r.computed>before:events.append(f'{r.name} prefix reused {r.computed} tokens')
            n=min(len(r.tokens)-r.computed,budget-spent)
            if n<=0:continue
            if not pool.ensure(r,r.computed+n):
                # Protect this step's selected work and earlier running requests.
                victims=[x for x in reqs if x is not r and x.table and x.name not in [x['request'] for x in scheduled] and x.start>r.start and r.state=='running']
                for victim in sorted(victims,key=lambda x:x.start,reverse=True):
                    pool.release(victim);victim.state='waiting';victim.preemptions+=1
                    events.append(f'preempt {victim.name}; keep known token IDs; release KV')
                    if pool.ensure(r,r.computed+n):break
                else:
                    if r.state=='waiting' and r.table:pool.release(r)
                    continue
            if r.state=='waiting':r.state='running';r.start=step if r.start<0 else r.start
            old=r.computed;positions=list(range(old,old+n));slots=[r.table[t//4]*4+t%4 for t in positions]
            pool.complete(r,old,old+n);r.computed+=n;spent+=n
            scheduled.append(dict(request=r.name,phase=phase,input_tokens=n,positions=positions,slots=slots))
            if r.computed==len(r.tokens):
                r.tokens.append(1000+ord(r.name)*10+r.generated);r.generated+=1
                events.append(f'{r.name} emitted toy token {r.generated}/{r.target}')
                if r.generated==r.target:r.state='finished';pool.release(r);events.append(f'{r.name} finished; KV refs released')
        pool.validate(reqs)
        trace.append(dict(step=step,spent=spent,budget=budget,scheduled=scheduled,events=events,free=list(pool.free),blocks=[dict(id=b['id'],ref=b['ref'],cached_prefix_length=len(b['key']) if b['key'] else 0) for b in pool.blocks],requests=[dict(name=r.name,state=r.state,known=len(r.tokens),computed=r.computed,generated=r.generated,target=r.target,table=list(r.table),preemptions=r.preemptions) for r in reqs]))
        if all(r.state=='finished' for r in reqs):break
        if spent==0 and all(r.arrival<=step for r in reqs):raise RuntimeError('Simulation made no progress; choose more blocks or inspect policy.')
    else:raise RuntimeError('Simulation hit iteration limit; no successful completion claimed.')
    summary=dict(iterations=len(trace),input_token_work=sum(x['spent'] for x in trace),output_tokens=sum(r.generated for r in reqs),prefix_block_hits=pool.hits,preemptions=sum(r.preemptions for r in reqs),all_finished=all(r.state=='finished' for r in reqs))
    return dict(kind='conceptual simulation; iterations are not milliseconds',config=dict(blocks=blocks,block_size=4,budget=budget,policy=policy,prefix=prefix),summary=summary,trace=trace)

def save_report(data,directory):
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'scheduler-trace.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    # Output folder can be anywhere: embed the small viewer assets to make it portable.
    # The reusable asset implementations remain in assets/, not copied into lesson sources.
    css=(ROOT/'assets/course.css').read_text(encoding='utf-8')+'\n'+(ROOT/'assets/deep-dive.css').read_text(encoding='utf-8')
    js=(ROOT/'assets/simulator-viewer.js').read_text(encoding='utf-8')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>本地调度模拟报告</title><style>'+css+'</style><main class="page"><p class="kicker">LOCAL LAB / CPU / SYNTHETIC WORKLOAD</p><h1>一次调度，哪些状态改变了</h1><p>这是有明确简化的教学模拟；迭代编号不是毫秒，toy token不代表语言模型输出。</p><pre id="sim-summary"></pre><section class="lab"><label for="step">选择迭代</label><div class="controls"><button id="prev">上一步</button><input id="step" type="range" min="0" value="0"><button id="next">下一步</button></div><div id="sim-state" aria-live="polite"></div></section></main><script id="sim-data" type="application/json">'+json.dumps(data,ensure_ascii=False).replace('</',r'<\/')+'</script><script>'+js+'</script></html>'
    (directory/'scheduler-trace.html').write_text(page,encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--blocks',type=int,default=6,choices=range(4,33));p.add_argument('--budget',type=int,default=4,choices=range(1,33));p.add_argument('--policy',choices=['mixed','prefill-first'],default='mixed');p.add_argument('--no-prefix',action='store_true');p.add_argument('--out',type=Path,default=ROOT/'labs/results/scheduler');a=p.parse_args()
    result=simulate(a.blocks,a.budget,a.policy,not a.no_prefix);save_report(result,a.out)
    print(json.dumps(result['summary'],indent=2));print('Saved:',a.out/'scheduler-trace.html')
if __name__=='__main__':main()
