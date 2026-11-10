/* Small, deterministic tensor examples. Values are computed; labels explain each axis. */
(() => {
  'use strict';
  const range=n=>Array.from({length:n},(_,i)=>i);
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt=x=>Number.isInteger(x)?String(x):x.toFixed(3);
  const panel=(title,body)=>`<div class="dimension-panel"><h4>${title}</h4>${body}</div>`;
  const select=(name,title,options,value)=>`<label>${title}<select name="${name}">${options.map(([v,t])=>`<option value="${v}" ${String(v)===String(value)?'selected':''}>${t}</option>`).join('')}</select></label>`;
  function matrix(values,{caption='',row='t',col='d',cell=()=>({})}={}) {
    return `<div class="dimension-matrix"><table><caption>${caption}</caption><thead><tr><th>${row} ↓ / ${col} →</th>${values[0].map((_,j)=>`<th scope="col">${col}${j}</th>`).join('')}</tr></thead><tbody>${values.map((r,i)=>`<tr><th scope="row">${row}${i}</th>${r.map((v,j)=>{const c=cell(i,j,v),label=c.label||`${row}${i}, ${col}${j}, 值 ${fmt(v)}`;return `<td class="${c.cls||''}" data-value="${v}">${c.action?`<button type="button" data-action="${c.action}" data-i="${i}" data-j="${j}" ${c.extra||''} aria-label="${esc(label)}" aria-pressed="${!!c.selected}">${fmt(v)}</button>`:`<span>${fmt(v)}</span>`}</td>`;}).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  function mount(el,controls,render,action) {
    el.innerHTML=`${controls}<div class="dimension-view"></div><div class="dimension-proof" role="status" aria-live="polite" aria-atomic="true"></div>`;
    const refresh=()=>{const result=render();el.querySelector('.dimension-view').innerHTML=result.body;el.querySelector('.dimension-proof').innerHTML=result.proof;};
    el.addEventListener('change',e=>{if(e.target.name){action(e.target.name,e.target.value);refresh();}});
    el.addEventListener('click',e=>{const b=e.target.closest('button[data-action]');if(!b)return;const actionName=b.dataset.action,i=b.dataset.i,j=b.dataset.j;action(actionName,b.dataset);refresh();const same=[...el.querySelectorAll('button[data-action]')].find(x=>x.dataset.action===actionName&&x.dataset.i===i&&x.dataset.j===j);if(same)same.focus({preventScroll:true});});
    refresh();
  }
  function layout(el) {
    const s={stage:'transpose',batch:0,chosen:2};
    const stages={split:['拆末维','q.reshape(2,3,2,2)','[B,T,Hq,d] = [2,3,2,2]'],transpose:['交换 token/head 两轴','q_split.transpose(1,2)','[B,Hq,T,d] = [2,2,3,2]'],wrong:['错误：直接 reshape 成目标 shape','q.reshape(2,2,3,2)','shape 正确，元素归属错误'],merge:['合头：先换回 token 在前','o.transpose(1,2).contiguous().view(2,3,4)','[B,T,Hq·d] = [2,3,4]']};
    mount(el,`<h3>点击一个元素，追踪它的坐标</h3><p>这里用 0–23 作为元素身份编号，展示布局变化；这些编号不是模型的 Q 数值。为了看到 batch 轴，额外设置 B=2。所有视图共24个元素，当前展示一个 batch 的12个。</p><div class="dimension-controls">${select('stage','要观察的操作',Object.entries(stages).map(([k,v])=>[k,v[0]]),s.stage)}${select('batch','选择 batch 切片',[[0,'b=0'],[1,'b=1']],0)}</div>`,()=>{
      const base=s.batch*12,head=Math.floor((s.chosen%4)/2),token=Math.floor((s.chosen%12)/4),dim=s.chosen%2;
      const cls=v=>`group-${Math.floor((v%4)/2)} ${v===s.chosen?'is-active':''}`;
      const flat=range(3).map(t=>range(4).map(c=>base+t*4+c));
      const source=s.stage==='merge'?'<div class="dimension-slices">'+range(2).map(h=>matrix(range(3).map(t=>range(2).map(d=>base+t*4+h*2+d)),{caption:`O[b,${h},:,:] · [T,d]=[3,2]`,cell:(i,j,v)=>({cls:cls(v),action:'choose-head-'+h,selected:v===s.chosen})})).join('')+'</div>':matrix(flat,{caption:'q_flat[b,:,:] · [T,Hq·d] = [3,4]',col:'c',cell:(i,j,v)=>({cls:cls(v),action:'choose',selected:v===s.chosen})});
      let dest='',code=stages[s.stage][1],shape=stages[s.stage][2];
      if(s.stage==='split')dest=range(3).map(t=>matrix(range(2).map(h=>range(2).map(d=>base+t*4+h*2+d)),{caption:`固定 b=${s.batch}, t=${t}；[Hq,d]=[2,2]`,row:'h',cell:(i,j,v)=>({cls:cls(v)})})).join('');
      else if(s.stage==='merge')dest=matrix(flat,{caption:'先恢复 [B,T,Hq,d]，再合并相邻 Hq,d 轴',col:'c',cell:(i,j,v)=>({cls:cls(v)})});
      else dest=range(2).map(h=>matrix(range(3).map(t=>range(2).map(d=>base+(s.stage==='wrong'?h*6+t*2+d:t*4+h*2+d))),{caption:`固定 b=${s.batch}, h=${h}；[T,d]=[3,2]`,cell:(i,j,v)=>({cls:cls(v)+(s.stage==='wrong'&&v!==base+i*4+h*2+j?' is-error':'')})})).join('');
      let proof=`选中的元素是 <strong>${s.chosen}</strong>：<code>q_flat[${s.batch},${token},${head*2+dim}] = q_split[${s.batch},${token},${head},${dim}] = q_heads[${s.batch},${head},${token},${dim}]</code>。末维索引 c=h×d_size+d_index=${head}×2+${dim}。`;
      if(s.stage==='wrong')proof+=`<br>这个元素在错误布局中变成 <code>q_wrong[${s.batch},${Math.floor((s.chosen%12)/6)},${Math.floor((s.chosen%6)/2)},${dim}] = ${s.chosen}</code>。<br><strong>反例：</strong>正确 q_heads[b,0,1,0] 应取 token1/head0/d0，即 ${base+4}；直接 reshape 得到 ${base+2}，取成了 token0/head1/d0。元素总数相等，只保证 reshape 可构造，不保证语义正确。`;
      else if(s.stage==='merge')proof=`选中的编号 <strong>${s.chosen}</strong> 从 <code>O[${s.batch},${head},${token},${dim}]</code> 合入 <code>concat[${s.batch},${token},${head*2+dim}]</code>，因为列号 c=head×d_size+d_index=${head}×2+${dim}。<br>先把T轴移到head前，再合并相邻的head与d轴。此例O沿用身份编号便于追踪；真实attention会算出新的O数值。`;
      else proof+='<br>transpose 改变逻辑坐标与 stride；它不必搬动物理内存。黄框跟踪同一个值，蓝/紫底色标记它原属的 head。';
      return {body:`<div class="dimension-stage">${code}<br><strong>${shape}</strong></div><div class="dimension-panels">${panel(s.stage==='merge'?'起点：head在前，分别展开两个head':'起点：每行一个 token；每2列属于一个 head',source)}${panel(s.stage==='merge'?'终点：每行收集同一token的全部head':'终点：固定外层轴，展开剩下的二维切片',`<div class="dimension-slices">${dest}</div>`)}</div>`,proof};
    },(name,value)=>{if(name==='stage')s.stage=value;else if(name==='batch'){s.batch=Number(value);s.chosen=s.batch*12+s.chosen%12;}else if(name==='choose')s.chosen=s.batch*12+Number(value.i)*4+Number(value.j);else if(name.startsWith('choose-head-'))s.chosen=s.batch*12+Number(value.i)*4+Number(name.slice(-1))*2+Number(value.j);});
  }
  function projection(el) {
    const X=[[1,2,0,1],[0,1,1,0],[2,0,1,1]];
    const W=[[1,0,0,1,1,0,2,0],[0,1,1,0,0,1,0,2],[1,1,0,0,1,0,1,0],[0,0,1,1,0,1,0,1]];
    const Y=X.map(row=>range(8).map(c=>row.reduce((sum,v,k)=>sum+v*W[k][c],0)));
    const s={t:0,c:2};
    mount(el,'<h3>点输出中的一个格子，展开那次乘加</h3><p>固定 b=0；省略 bias。输入特征轴 H=4 被求和，保留 token 轴 T=3，并生成新的投影轴 C=8。</p>',()=>{
      const group=c=>c<4?0:c<6?1:2;
      const inputs=matrix(X,{caption:'X[b] = [T,H] = [3,4]',col:'k',cell:(i,j)=>({cls:i===s.t?'is-related':''})});
      const weights=matrix(W,{caption:'数学记法 W = [H,C] = [4,8]；PyTorch weight 是 [8,4]',row:'k',col:'c',cell:(i,j)=>({cls:`group-${group(j)} ${j===s.c?'is-related':''}`})});
      const outputs=matrix(Y,{caption:'Y = XW = [T,C] = [3,8]；点击任意格子',col:'c',cell:(i,j)=>({cls:`group-${group(j)} ${i===s.t&&j===s.c?'is-active':''}`,action:'output',selected:i===s.t&&j===s.c})});
      return {body:`<div class="dimension-panels">${panel('① 输入的一整行 × 权重的一整列',inputs+weights)}${panel('② 一个输出标量；最后再按列切片',outputs+'<div class="dimension-badges"><span>Q = Y[:,0:4] → [3,4]</span><span>K = Y[:,4:6] → [3,2]</span><span>V = Y[:,6:8] → [3,2]</span></div><p>split([4,2,2], dim=-1) 是取三段切片，不是平均分三份。投影做了计算；split 本身不再做乘加。</p>')}</div>`,proof:`<code>Y[${s.t},${s.c}] = Σₖ X[${s.t},k] × W[k,${s.c}]</code><br>${X[s.t].map((v,k)=>`${v}×(${W[k][s.c]})`).join(' + ')} = <strong>${Y[s.t][s.c]}</strong>。收缩的是 k 轴；t=${s.t} 与 c=${s.c} 确定输出格子。输入12个数、输出24个数，线性投影不要求元素数守恒。`};
    },(name,d)=>{if(name==='output'){s.t=Number(d.i);s.c=Number(d.j);}});
  }
  function attention(el) {
    const Q=[[[1,0],[0,1],[1,1]],[[0,1],[1,0],[1,-1]]],K=[[1,0],[0,1],[1,1]],V=[[1,2],[3,4],[5,6]];
    const s={h:0,t:2,key:1,mask:'on'};
    mount(el,`<h3>沿一行 query 看两次矩阵乘法</h3><div class="dimension-controls">${select('h','Q head',[[0,'h=0 → KV head 0'],[1,'h=1 → KV head 0']],0)}${select('t','查询位置',range(3).map(i=>[i,`tq=${i}`]),2)}${select('key','匹配哪个 key',range(3).map(i=>[i,`tk=${i}`]),1)}${select('mask','因果遮罩',[['on','只看当前及过去'],['off','关闭（观察未来泄漏）']],'on')}</div>`,()=>{
      const scores=Q[s.h].map(q=>K.map(k=>q.reduce((sum,x,d)=>sum+x*k[d],0)/Math.sqrt(2)));
      const A=scores.map((row,t)=>{const visible=row.map((v,k)=>s.mask==='on'&&k>t?-Infinity:v),m=Math.max(...visible),exps=visible.map(v=>Math.exp(v-m)),z=exps.reduce((x,y)=>x+y,0);return exps.map(v=>v/z);});
      const O=A.map(row=>range(2).map(d=>row.reduce((sum,w,k)=>sum+w*V[k][d],0)));
      const q=matrix(Q[s.h],{caption:`Q[b,${s.h},:,:] = [Tq,d] = [3,2]`,cell:(i)=>({cls:i===s.t?'is-related':''})});
      const kt=matrix(range(2).map(d=>K.map(k=>k[d])),{caption:'Kᵀ：只交换 K 的 Tk,d 两轴 → [d,Tk] = [2,3]',row:'d',col:'k',cell:(i,j)=>({cls:j===s.key?'is-related':''})});
      const sm=matrix(scores,{caption:'S = QKᵀ / √2 → [Tq,Tk] = [3,3]',row:'q',col:'k',cell:(i,j)=>({cls:i===s.t&&j===s.key?'is-active':''})});
      const am=matrix(A,{caption:'A = softmax(S+mask)，沿 key 轴；每行和=1',row:'q',col:'k',cell:(i,j,v)=>({cls:`${i===s.t?'is-related':''} ${v===0?'is-muted':''}`})});
      const vm=matrix(V,{caption:'共享的 V[b,0,:,:] = [Tk,d] = [3,2]',row:'k'});
      const om=matrix(O,{caption:'O = AV → [Tq,d] = [3,2]',row:'q',cell:(i)=>({cls:i===s.t?'is-active':''})});
      const qv=Q[s.h][s.t],kv=K[s.key];
      return {body:`<div class="dimension-stage">[B,Hq,Tq,d] × [B,Hq,d,Tk] → [B,Hq,Tq,Tk]<br>[B,Hq,Tq,Tk] × [B,Hq,Tk,d] → [B,Hq,Tq,d]</div><p>GQA：Hq=2、Hkv=1，两组 Q 都读同一组 K/V；上式按逻辑配对写 Hq，内核不一定复制 K/V。batch/head 是独立切片轴，不参与本次乘加。</p><div class="dimension-panels">${panel('① 收缩 head 内的特征轴 d',q+kt+sm)}${panel('② 收缩 key 位置轴 Tk',am+vm+om)}</div>`,proof:`匹配分数：<code>S[${s.t},${s.key}] = (${qv[0]}×${kv[0]} + ${qv[1]}×${kv[1]}) / √2 = ${fmt(scores[s.t][s.key])}</code>。${s.mask==='on'&&s.key>s.t?'该格子属于未来，mask 后权重为0。':''}<br>输出第0维：<code>O[${s.t},0] = ${A[s.t].map((w,k)=>`${fmt(w)}×${V[k][0]}`).join(' + ')} ≈ ${fmt(O[s.t][0])}</code>。显示值保留3位；计算用未舍入数值。先对 d 求和得到“位置×位置”，再对 Tk 求和带回“位置×特征”。`};
    },(name,value)=>{s[name]=name==='mask'?value:Number(value);});
  }
  function packed(el) {
    const rows=[[10,11,12],[20,21]],s={r:1,t:1};
    mount(el,`<h3>一个 token 的三种坐标</h3><p>A 请求长度3，B请求长度2；H=4。分页块长2，A块表=[5,2]，B块表=[7]，用于演示映射。</p><div class="dimension-controls">${select('r','请求',[[0,'A（长度3）'],[1,'B（长度2）']],1)}${select('t','请求内位置',[[0,'t=0'],[1,'t=1']],1)}</div>`,()=>{
      const offsets=[0,3,5],blockTables=[[5,2],[7]],n=offsets[s.r]+s.t,physical=blockTables[s.r][Math.floor(s.t/2)],slot=physical*2+s.t%2;
      const flat=rows.flat(),x=flat.map((_,n)=>range(4).map(d=>n*4+d));
      const padded=matrix([[10,11,12],[20,21,-1]],{caption:'便于观察的 padded IDs [B,Tmax]=[2,3]；−1代表PAD（不是合法token）',row:'r',col:'t',cell:(i,j,v)=>({cls:v===-1?'is-muted':i===s.r&&j===s.t?'is-active':''})});
      const packedX=matrix(x,{caption:'packed hidden [N,H]=[5,4]，每行4个身份编号',row:'n',cell:(i)=>({cls:i===n?'is-active':''})});
      return {body:`<div class="dimension-panels">${panel('① 按请求取有效 token，再串接',padded+'<div class="dimension-stage">lengths = [3,2]<br>cu_seqlens = [0,3,5]<br>packed IDs = [10,11,12,20,21]<br>positions = [0,1,2,0,1]</div>')}${panel('② packed 行号和 KV 写入槽号各算各的',packedX+`<div class="dimension-stage">逻辑块 = t // 2 = ${Math.floor(s.t/2)}<br>物理块 = block_table[逻辑块] = ${physical}<br>块内偏移 = t % 2 = ${s.t%2}<br>KV slot = ${physical}×2+${s.t%2} = ${slot}</div>`)}</div>`,proof:`请求 ${s.r===0?'A':'B'} 的 token位置 ${s.t}：<code>packed行 n = cu_seqlens[${s.r}] + ${s.t} = ${n}</code>，<code>position = ${s.t}</code>，<code>KV slot = ${slot}</code>。这三者回答“这次输入在哪行”“语义上处于哪个位置”“缓存存在哪格”，不能互换。<br>这里是没有prefix命中的完整prefill小例；chunk、decode和prefix复用还要读取已缓存长度。把 padded [2,3,4] 直接reshape，仍有6行；不会自动去掉PAD变成5行。`};
    },(name,value)=>{s[name]=Number(value);if(name==='r'){s.t=Math.min(s.t,s.r===0?2:1);const control=el.querySelector('select[name=t]');control.innerHTML=range(s.r===0?3:2).map(i=>`<option value="${i}" ${i===s.t?'selected':''}>t=${i}</option>`).join('');}});
  }
  const labs={layout,projection,attention,packed};
  document.querySelectorAll('[data-dimension-lab]').forEach(el=>{const fn=labs[el.dataset.dimensionLab];if(fn)fn(el);});
})();
