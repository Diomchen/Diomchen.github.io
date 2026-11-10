"""CPU-only numerical labs. Requires NumPy; no model download or GPU.
These small random models verify mathematical contracts, not model quality/performance.
Run: python labs/transformer_lab.py all
"""
import argparse
import math
import numpy as np

def softmax(x):
    x=x-np.max(x,axis=-1,keepdims=True)
    e=np.exp(x)
    return e/e.sum(axis=-1,keepdims=True)

def attention_lab():
    q=np.array([[1.,0.]])
    k=np.array([[1.,0.],[0.,1.]])
    v=np.array([[1.,0.],[0.,2.]])
    scores=q@k.T/math.sqrt(2)
    p=softmax(scores);out=p@v
    expected=1/(1+math.exp(-1/math.sqrt(2)))
    np.testing.assert_allclose(out,[[expected,2*(1-expected)]])
    print('scores:',scores,'weights:',p,'output:',out)
    return {'output':out.tolist(),'row_sum':p.sum(axis=-1).tolist()}

def rope(x,positions,base=10000.):
    """[T,heads,d], half-split layout matching the teaching vendor convention."""
    d=x.shape[-1]
    assert d%2==0
    theta=np.asarray(positions)[:,None,None]*(base**(-np.arange(0,d,2)/d))[None,None,:]
    a,b=np.split(x,2,axis=-1)
    return np.concatenate([a*np.cos(theta)-b*np.sin(theta),b*np.cos(theta)+a*np.sin(theta)],axis=-1)

def rope_lab():
    rng=np.random.default_rng(42)
    q=rng.normal(size=(1,2,4));k=rng.normal(size=(1,2,4))
    dot=lambda a,b:np.sum(a*b,axis=-1)
    before=dot(rope(q,[2]),rope(k,[5]))
    after=dot(rope(q,[9]),rope(k,[12]))
    np.testing.assert_allclose(before,after,atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(q,axis=-1),np.linalg.norm(rope(q,[7]),axis=-1),atol=1e-12)
    error=float(np.max(np.abs(before-after)))
    print('RoPE common-shift max error:',error,'half-split pairs: (0,2), (1,3)')
    return {'common_shift_max_error':error}

def rms(x):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)

class ToyDecoder:
    """Two random pre-norm decoder layers with GQA, QK norm, RoPE and SwiGLU.
    No tokenizer/training/dropout/quantization. Dense weights use row-vector convention.
    """
    def __init__(self,seed=19):
        rng=np.random.default_rng(seed);self.layers=[]
        self.h=8;self.hq=4;self.hkv=2;self.d=4;self.f=12
        def w(a,b):return rng.normal(0,.18,(a,b))
        for _ in range(2):self.layers.append(dict(q=w(8,16),k=w(8,8),v=w(8,8),o=w(16,8),g=w(8,12),u=w(8,12),down=w(12,8)))
        self.head=w(8,11)

    def forward(self,x,cache=None,wrong_mask=False):
        cache=[None]*len(self.layers) if cache is None else cache
        next_cache=[]
        for index,w in enumerate(self.layers):
            z=rms(x);t=len(x);past=cache[index];p=0 if past is None else len(past[0])
            q=rope(rms((z@w['q']).reshape(t,self.hq,self.d)),np.arange(p,p+t))
            k=rope(rms((z@w['k']).reshape(t,self.hkv,self.d)),np.arange(p,p+t))
            v=(z@w['v']).reshape(t,self.hkv,self.d)
            if past is not None:k=np.concatenate([past[0],k]);v=np.concatenate([past[1],v])
            next_cache.append((k.copy(),v.copy()))
            # Repeat is for clarity; production GQA kernels need not duplicate storage.
            kr=np.repeat(k,self.hq//self.hkv,axis=1);vr=np.repeat(v,self.hq//self.hkv,axis=1)
            scores=np.einsum('thd,shd->hts',q,kr)/math.sqrt(self.d)
            qpos=np.arange(t) if wrong_mask else np.arange(p,p+t)
            allowed=np.arange(len(k))[None,:]<=qpos[:,None]
            weights=softmax(np.where(allowed[None,:,:],scores,-np.inf))
            out=np.einsum('hts,shd->thd',weights,vr).reshape(t,-1)
            x=x+out@w['o'];z=rms(x);gate=z@w['g'];up=z@w['u']
            x=x+(gate/(1+np.exp(-gate))*up)@w['down']
        return rms(x)@self.head,next_cache

def cache_lab():
    rng=np.random.default_rng(7);x=rng.normal(size=(7,8));model=ToyDecoder()
    full,_=model.forward(x)
    cache=None;pieces=[]
    for row in x:
        y,cache=model.forward(row[None,:],cache);pieces.append(y)
    incremental=np.concatenate(pieces)
    a,cache=model.forward(x[:3]);b,_=model.forward(x[3:],cache);chunked=np.concatenate([a,b])
    wrong,_=model.forward(x[3:],cache,wrong_mask=True)
    np.testing.assert_allclose(full,incremental,atol=1e-11,rtol=1e-11)
    np.testing.assert_allclose(full,chunked,atol=1e-11,rtol=1e-11)
    bad_error=float(np.max(np.abs(full[3:]-wrong)))
    assert bad_error>1e-4,'Negative control failed: wrong mask must differ.'
    result={'full_vs_incremental':float(np.max(np.abs(full-incremental))),'full_vs_chunked':float(np.max(np.abs(full-chunked))),'wrong_mask_error':bad_error,'layers':2,'dtype':'float64'}
    print('KV equivalence:',result)
    return result

def online_lab():
    rng=np.random.default_rng(13);s=rng.normal(0,5,11);v=rng.normal(size=(11,4))
    m=-np.inf;l=0.;o=np.zeros(4)
    for start in range(0,len(s),3):
        sb=s[start:start+3];vb=v[start:start+3];mb=max(sb);eb=np.exp(sb-mb)
        m_new=max(m,mb);alpha=np.exp(m-m_new);beta=np.exp(mb-m_new)
        l=alpha*l+beta*sum(eb);o=alpha*o+beta*(eb@vb);m=m_new
    dense=softmax(s)@v;stream=o/l
    np.testing.assert_allclose(dense,stream,atol=1e-12)
    result={'online_vs_dense':float(np.max(np.abs(dense-stream))),'toy_output':7/3}
    print('Online softmax:',result)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('lab',choices=['all','attention','rope','cache','online'],default='all',nargs='?');args=parser.parse_args()
    labs={'attention':attention_lab,'rope':rope_lab,'cache':cache_lab,'online':online_lab}
    for name,fn in labs.items():
        if args.lab in ('all',name):fn()
    print('PASS: selected numerical contracts; this is not a GPU benchmark.')
