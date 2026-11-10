/* Conceptual teaching models, NOT framework emulators or performance predictors. */
(function () {
  'use strict';
  const esc = v => String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const softmax = a => { const m=Math.max(...a),e=a.map(x=>Math.exp(x-m)),z=e.reduce((s,x)=>s+x,0); return e.map(x=>x/z); };
  const table = (heads,rows) => '<table><thead><tr>'+heads.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(c=>'<td>'+esc(c)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
  const bar = (name,n,max) => '<div class="bar-row"><span>'+esc(name)+'</span><div class="bar-track"><div class="bar-fill" style="width:'+Math.max(0,Math.min(100,n/max*100))+'%"></div></div><span>'+n.toFixed(3)+'</span></div>';
  const controls = {
    attention:'<label>Query 位置<select name="row"><option>0</option><option>1</option><option selected>2</option></select></label><label>温度（教学旋钮）<input name="temp" type="range" min="0.25" max="2" step="0.25" value="1"></label><label>因果遮罩<select name="mask"><option value="on">开启</option><option value="off">关闭</option></select></label>',
    rope:'<label>Query 位置 m<input name="m" type="range" min="0" max="12" value="2"></label><label>Key 位置 n<input name="n" type="range" min="0" max="12" value="5"></label><label>共同平移 Δ<input name="shift" type="range" min="0" max="12" value="0"></label>',
    kv:'<label>层数 L<input name="layers" type="number" min="1" max="128" value="28"></label><label>KV heads<select name="heads"><option>1</option><option>4</option><option selected>8</option><option>16</option></select></label><label>序列长度 T<input name="tokens" type="range" min="256" max="8192" step="256" value="4096"></label><label>并发 B<input name="batch" type="number" min="1" max="64" value="1"></label>',
    pages:'<label>逻辑 token 位置 t<input name="token" type="range" min="0" max="11" value="6"></label>',
    schedule:'<label>本轮 token 预算<input name="budget" type="range" min="1" max="12" value="6"></label><label>长 prompt 剩余<input name="remaining" type="range" min="1" max="12" value="9"></label>',
    moe:'<label>热点专家 E0 的 token 路由数<input name="load" type="range" min="4" max="32" value="16"></label><label>E0 副本数<select name="copies"><option>1</option><option selected>2</option><option>4</option></select></label>',
    spec:'<label>草稿长度 K<input name="k" type="range" min="1" max="5" value="4"></label><label>首个拒绝位置（1 起；6=全接受）<input name="reject" type="range" min="1" max="6" value="3"></label>',
    sampling:'<label>Temperature<input name="temp" type="range" min="0.25" max="2" step="0.25" value="1"></label><label>Top-p<input name="p" type="range" min="0.1" max="1" step="0.1" value="0.9"></label>'
  };
  function render(kind,v,out) {
    if(kind==='attention') {
      const Q=[[1,0],[0,1],[1,1]],K=[[1,0],[0,1],[1,1]],V=[[1,0],[0,2],[3,1]],i=+v.row;
      const scores=K.map(k=>(Q[i][0]*k[0]+Q[i][1]*k[1])/Math.sqrt(2)/v.temp);
      const masked=scores.map((x,j)=>v.mask==='on'&&j>i?-Infinity:x),w=softmax(masked);
      const result=[0,1].map(d=>w.reduce((s,p,j)=>s+p*V[j][d],0));
      out.innerHTML=table(['Key 位置','缩放分数','遮罩后','权重'],scores.map((s,j)=>[j,s.toFixed(4),Number.isFinite(masked[j])?masked[j].toFixed(4):'−∞',w[j].toFixed(4)]))+'<p>Q['+i+'] = ['+Q[i]+']；温度 = '+v.temp+'；Σ 权重 = '+w.reduce((s,x)=>s+x,0).toFixed(4)+'</p><p>输出 = Σⱼ weightⱼ Vⱼ = ['+result.map(x=>x.toFixed(4)).join(', ')+']</p>';
    } else if(kind==='rope') {
      const a=(+v.m + +v.shift)*.35,b=(+v.n + +v.shift)*.35,point=x=>[180+100*Math.cos(x),140-100*Math.sin(x)]; const p=point(a),q=point(b);
      out.innerHTML='<svg viewBox="0 0 400 290" role="img" aria-label="二维 RoPE 旋转，蓝色 Q，红色 K"><circle cx="180" cy="140" r="100" fill="none" stroke="#bdcfd7"/><path d="M60 140H300 M180 20V260" stroke="#d2d5d1"/><path d="M180 140L'+p.join(' ')+'" stroke="#245c78" stroke-width="4"/><path d="M180 140L'+q.join(' ')+'" stroke="#a63a2a" stroke-width="3"/><text x="15" y="280" fill="#245c78">Q 蓝线</text><text x="110" y="280" fill="#a63a2a">K 红线</text></svg><p>m = '+v.m+'，n = '+v.n+'，共同平移 = '+v.shift+'。θ = 0.35 rad。</p><p>旋转后点积 = cos((n−m)θ) = '+Math.cos(b-a).toFixed(6)+'。原始 Q=K=[1,0]；只演示一个二维对子。</p>';
    } else if(kind==='kv') {
      const bytes=2*v.layers*v.heads*128*2, total=bytes*v.tokens*v.batch;
      out.innerHTML=table(['项','数值'],[['每 token 全层 KV',bytes/1024+' KiB'],['T / B',v.tokens+' / '+v.batch],['逻辑 KV 总量',(total/2**20).toFixed(1)+' MiB'],['换成 GiB',(total/2**30).toFixed(3)+' GiB']])+'<p>假设标准 K/V、head_dim=128、BF16、单卡；不含权重、激活、页尾空槽和 CUDA Graph。不是整模型显存估计，也不适用于 MLA 的缓存布局。</p>';
    } else if(kind==='pages') {
      const t=+v.token,B=4,blocks=[5,2,7],logical=Math.floor(t/B),slot=blocks[logical]*B+t%B;
      out.innerHTML=table(['逻辑块','物理块','逻辑位置','物理槽'],blocks.map((b,i)=>[i,b,`${i*4}…${i*4+3}`,`${b*4}…${b*4+3}`]))+'<p><strong>t='+t+' → 逻辑块 '+logical+' → 物理块 '+blocks[logical]+' → slot='+slot+'</strong></p><p>slot = block_table[floor(t/4)] × 4 + (t mod 4)。教学块大小为 4；本地 nano-vllm 默认值为 256。</p>';
    } else if(kind==='schedule') {
      let budget=+v.budget; const rows=[];[['decode A',1],['decode B',1],['prefill C',+v.remaining]].forEach(([name,need])=>{const n=Math.min(need,budget);rows.push([name,need,n,need-n]);budget-=n;});
      out.innerHTML=table(['请求','待计算','本轮分配','尚欠'],rows)+'<p>剩余预算 = '+budget+'。抽象演示：先两个 running decode，再 chunked prefill；忽略 KV 不足、优先级、多模态、投机和异步占位，不是完整 vLLM 调度器。</p>';
    } else if(kind==='moe') {
      const hot=+v.load,copies=+v.copies,loads=[4,4,4,4];for(let i=0;i<copies;i++) loads[i]+=hot/copies;
      out.innerHTML=loads.map((n,i)=>bar('Rank '+i,n,36)).join('')+'<p>总路由数 '+(hot+16)+'；最大 rank 负载 '+Math.max(...loads)+'；理想均衡下界 '+((hot+16)/4)+'。</p><p>假设每卡背景负载 4，E0 流量在副本间均分；忽略每卡槽位上限、通信、权重迁移。这是解释副本价值的玩具模型，不是 EPLB policy 的复现。</p>';
    } else if(kind==='spec') {
      const k=+v.k,r=+v.reject,a=Math.min(k,r-1);const rows=Array.from({length:k},(_,i)=>[i+1,i<a?'接受':i===a?'拒绝':'丢弃后缀']);
      out.innerHTML=table(['草稿位置','处理'],rows)+'<p>接受草稿 '+a+' 个；'+(a===k?'再从 target 的末位置分布采样 bonus token':'在首拒绝位置按算法要求采样替代 token')+'；本轮输出 '+(a+1)+' 个。</p><p>通用投机协议示意。这里人为设置拒绝位置，不模拟置信度头，也不声称这是 DSpark 的采样公式。KV 回退到有效前缀，后缀槽位数据可仍存在，但不可再读作有效上下文。</p>';
    } else if(kind==='sampling') {
      const p=softmax([2,1,0,-1].map(x=>x/v.temp));let sum=0; const kept=p.map(x=>{const keep=sum < +v.p;sum+=x;return keep;});const z=p.reduce((s,x,i)=>s+(kept[i]?x:0),0);
      out.innerHTML=p.map((x,i)=>bar('token '+i,x,1)).join('')+table(['token','Top-p 保留','重归一化概率'],p.map((x,i)=>[i,kept[i]?'是':'否',(kept[i]?x/z:0).toFixed(4)]))+'<p>logits=[2,1,0,−1]，temperature='+v.temp+'，top-p='+v.p+'。包含使累计概率首次达到阈值的那个 token；top-p 是概念扩展，本地 nano-vllm sampler 没有实现。</p>';
    }
  }
  document.querySelectorAll('[data-explorer]').forEach(root=>{
    const kind=root.dataset.explorer;if(!controls[kind])return;
    root.innerHTML='<h3>'+esc(root.dataset.title||'动手改变一个变量')+'</h3><div class="controls">'+controls[kind]+'</div><div class="lab-output" aria-live="polite"></div><p class="lab-note">先预测再拖动。所有数字在本地计算；图例采用明确简化的教学模型。</p>';
    const out=root.querySelector('.lab-output');function update(){let v={};root.querySelectorAll('input,select').forEach(x=>{v[x.name]=x.type==='number'?Math.max(+x.min,Math.min(+x.max,Number(x.value)||+x.min)):x.value;});render(kind,v,out);}root.addEventListener('input',update);update();
  });
  document.querySelectorAll('script[data-quiz-config]').forEach(node=>{const root=node.previousElementSibling;if(window.CourseQuiz)window.CourseQuiz.mount(root,JSON.parse(node.textContent));});
  document.querySelectorAll('[data-number-check]').forEach(root=>{
    const input=root.querySelector('input'),feedback=root.querySelector('[role=status]');root.querySelector('button').addEventListener('click',()=>{
      if(input.value.trim()===''||!Number.isFinite(Number(input.value))){feedback.textContent='先输入你的计算结果。';return;}
      const correct=Math.abs(Number(input.value)-Number(root.dataset.answer))<=Number(root.dataset.tolerance||0);
      feedback.textContent=(correct?'正确。':'再检查一次。')+root.dataset.explain;
    });
  });
  document.querySelectorAll('[data-study]').forEach(root=>{
    const key='vllm-learn:review:'+root.dataset.study, status=root.querySelector('[role=status]');
    function show(){try{const x=JSON.parse(localStorage.getItem(key)||'null');status.textContent=x?'上次自评：'+x.state+' · '+new Date(x.time).toLocaleString()+'。建议隔 1、3、7 天闭卷重做。':'尚未自评。阅读完成不等于掌握；用闭卷复述和数值练习检验。';}catch(e){status.textContent='当前浏览器无法保存进度；可以将自评记在本地笔记。';}}
    root.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{try{localStorage.setItem(key,JSON.stringify({state:b.textContent,time:Date.now()}));show();}catch(e){status.textContent='无法保存，请手动记录此次自评。';}}));show();
  });
  window.CourseMath = {softmax};
})();
