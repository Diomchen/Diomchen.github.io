"""Small synthetic CUDA labs for a 12 GiB RTX 4070 Ti; no model download.
Run in your WSL PyTorch environment: python labs/gpu_lab.py --suite all
Diagnose without importing a model: python labs/gpu_lab.py --diagnose
SDPA auto is not guaranteed to select FlashAttention. Timings are operator-only.
"""
import argparse
import csv
from datetime import datetime,timezone
import html
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def diagnose():
    info=dict(python=sys.executable,python_version=platform.python_version(),platform=platform.platform(),torch_installed=importlib.util.find_spec('torch') is not None)
    if info['torch_installed']:
        import torch
        info.update(torch=torch.__version__,torch_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available())
        if info['cuda_available']:
            prop=torch.cuda.get_device_properties(0)
            info.update(gpu=prop.name,capability=list(torch.cuda.get_device_capability()),vram_bytes=prop.total_memory,bf16_supported=torch.cuda.is_bf16_supported())
    return info

def measure(torch,fn,repeats):
    for _ in range(5):fn()
    torch.cuda.synchronize();samples=[]
    for _ in range(3):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize();wall=time.perf_counter();start.record()
        for _ in range(repeats):fn()
        end.record();end.synchronize()
        samples.append((start.elapsed_time(end)/repeats,(time.perf_counter()-wall)*1000/repeats))
    return dict(cuda_ms=statistics.median(x[0] for x in samples),wall_ms=statistics.median(x[1] for x in samples),repeats=repeats,groups=3)

def correctness(torch,F,dtype):
    torch.manual_seed(20261108);t=32;hq=4;hkv=2;d=32
    q=torch.randn(1,hq,t,d,device='cuda',dtype=dtype)
    k_small=torch.randn(1,hkv,t,d,device='cuda',dtype=dtype);v_small=torch.randn_like(k_small)
    # Explicit repeat is only a teaching convenience, not memory-efficient GQA.
    k=k_small.repeat_interleave(hq//hkv,dim=1);v=v_small.repeat_interleave(hq//hkv,dim=1)
    def run(start,end,wrong=False):
        positions=torch.arange(end-start,device='cuda') if wrong else torch.arange(start,end,device='cuda')
        mask=torch.arange(end,device='cuda')[None,:]<=positions[:,None]
        # True = visible for SDPA. Explicit offsets avoid non-square upper-left causal bias.
        return F.scaled_dot_product_attention(q[:,:,start:end],k[:,:,:end],v[:,:,:end],attn_mask=mask,dropout_p=0.,is_causal=False)
    full=run(0,t);incremental=torch.cat([run(i,i+1) for i in range(t)],dim=2)
    chunked=torch.cat([run(0,5),run(5,12),run(12,t)],dim=2);bad=run(12,t,True)
    atol=rtol=1e-4 if dtype==torch.float32 else 2e-2
    torch.testing.assert_close(full,incremental,atol=atol,rtol=rtol);torch.testing.assert_close(full,chunked,atol=atol,rtol=rtol)
    bad_error=(full[:,:,12:]-bad).abs().max().item();assert bad_error>.01,'Wrong-mask negative control failed'
    return dict(lab='correctness',status='passed',full_vs_token_max_error=(full-incremental).abs().max().item(),full_vs_chunk_max_error=(full-chunked).abs().max().item(),wrong_mask_error=bad_error,atol=atol,rtol=rtol,scope='attention only; fixed QKV; no full language model')

def sdpa_benchmark(torch,F,dtype,sizes,repeats):
    rows=[]
    for n in sizes:
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        q=torch.randn(1,8,n,64,device='cuda',dtype=dtype);k=torch.randn_like(q);v=torch.randn_like(q)
        visible=torch.ones(n,n,device='cuda',dtype=torch.bool).tril()
        def naive():
            scores=(q.float()@k.float().transpose(-1,-2))/(64**.5)
            probs=torch.softmax(scores.masked_fill(~visible,-float('inf')),dim=-1)
            return (probs@v.float()).to(dtype)
        def sdpa():return F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=True)
        a=naive();b=sdpa();atol=rtol=2e-4 if dtype==torch.float32 else 2e-2
        torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
        for name,fn in [('naive-fp32-intermediates',naive),('sdpa-auto',sdpa)]:
            torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();timing=measure(torch,fn,repeats)
            rows.append(dict(lab='attention-timing',implementation=name,tokens=n,dtype=str(dtype),**timing,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_extra_bytes=torch.cuda.max_memory_allocated()-base))
        del q,k,v,a,b,visible
    return rows

def kv_scaling(torch,sizes):
    rows=[]
    for n in sizes:
        torch.cuda.empty_cache();base=torch.cuda.memory_allocated()
        # Standard KV geometry of Qwen3-0.6B, no weights or model allocated.
        cache=torch.empty(2,28,1,n,8,128,device='cuda',dtype=torch.bfloat16);cache.zero_();torch.cuda.synchronize()
        expected=2*28*n*8*128*2;actual=cache.numel()*cache.element_size()
        assert actual==expected
        rows.append(dict(lab='kv-scaling',tokens=n,logical_bytes=actual,allocator_delta_bytes=torch.cuda.memory_allocated()-base,logical_mib=actual/2**20,scope='KV tensor only; excludes model/workspace/graphs'))
        del cache
    return rows

def graph_lab(torch,repeats):
    x=torch.randn(1,128,device='cuda');weights=[torch.randn(128,128,device='cuda')*.01 for _ in range(8)]
    def chain():
        y=x
        for w in weights:y=torch.relu(y@w)+.01
        return y
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):chain()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):static_output=chain()
    # Keep x/weights/output alive; change contents in place, never replace addresses.
    x.copy_(torch.randn_like(x));graph.replay();torch.cuda.synchronize();torch.testing.assert_close(static_output,chain())
    eager=measure(torch,chain,repeats);replay=measure(torch,graph.replay,repeats)
    return [dict(lab='graph-timing',implementation='eager',**eager),dict(lab='graph-timing',implementation='graph-replay',**replay)]

def save(report,directory):
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'gpu-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    fields=sorted(set(k for row in report['results'] for k in row))
    with (directory/'gpu-results.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(report['results'])
    css=(ROOT/'assets/course.css').read_text(encoding='utf-8')+(ROOT/'assets/deep-dive.css').read_text(encoding='utf-8')
    body='<h1>本机GPU合成实验报告</h1><p>以下是本次运行的实测算子/张量结果。不是vLLM、nano-vllm或DeepInfer端到端吞吐。SDPA auto不保证选择FlashAttention；naive使用FP32中间结果，两条实现精度路径不同。</p><pre>'+html.escape(json.dumps(report['environment'],indent=2,ensure_ascii=False))+'</pre>'
    for row in report['results']:body+='<h2>'+html.escape(row['lab'])+'</h2><pre>'+html.escape(json.dumps(row,indent=2,ensure_ascii=False))+'</pre>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPU实验报告</title><style>'+css+'</style><main class="page">'+body+'</main></html>'
    (directory/'gpu-report.html').write_text(page,encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnose',action='store_true');p.add_argument('--suite',choices=['all','correctness','attention','memory','graph'],default='all');p.add_argument('--dtype',choices=['fp32','bf16'],default='fp32');p.add_argument('--sizes',type=int,nargs='+',default=[256,512,1024]);p.add_argument('--repeats',type=int,default=20);p.add_argument('--out',type=Path,default=ROOT/'labs/results/gpu');a=p.parse_args()
    info=diagnose();print(json.dumps(info,ensure_ascii=False,indent=2))
    if a.diagnose:return
    if not info.get('torch_installed'):p.exit(2,'PyTorch is missing in this interpreter. Activate your existing WSL PyTorch environment, or use the official install selector linked in labs/LOCAL-LABS.md. No package was installed.\n')
    if not info.get('cuda_available'):p.exit(2,'This interpreter cannot access CUDA. Check nvidia-smi and the installed PyTorch CUDA build.\n')
    if any(n<16 or n>2048 for n in a.sizes) or not 1<=a.repeats<=200:p.error('For this teaching preset use sizes 16..2048 and repeats 1..200.')
    import torch
    import torch.nn.functional as F
    if a.dtype=='bf16' and not torch.cuda.is_bf16_supported():p.exit(2,'BF16 is not supported in this environment; use --dtype fp32.\n')
    free,total=torch.cuda.mem_get_info()
    if free<1024**3:p.exit(2,'Less than 1 GiB CUDA memory is currently free. Close a competing GPU workload before this preset.\n')
    torch.backends.cuda.matmul.allow_tf32=False;dtype=torch.float32 if a.dtype=='fp32' else torch.bfloat16;rows=[]
    with torch.inference_mode():
        if a.suite in ('all','correctness'):rows.append(correctness(torch,F,dtype))
        if a.suite in ('all','attention'):rows.extend(sdpa_benchmark(torch,F,dtype,a.sizes,a.repeats))
        if a.suite in ('all','memory'):rows.extend(kv_scaling(torch,a.sizes))
        if a.suite in ('all','graph'):rows.extend(graph_lab(torch,a.repeats))
    report=dict(time_utc=datetime.now(timezone.utc).isoformat(),environment=info,configuration=vars(a)|{'out':str(a.out)},results=rows,scope='synthetic local CUDA experiments, not an inference framework benchmark')
    save(report,a.out);print('Saved:',a.out/'gpu-report.html')
if __name__=='__main__':main()
