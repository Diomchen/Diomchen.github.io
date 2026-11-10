> 公开课程说明：以下4070 Ti / 64GB / WSL记录来自作者的验证环境，不表示检测访问者设备。请先按[下载与运行指南](../reference/start-here.html)下载完整课程。GPU实测尚未完成。

# 4070 Ti / 64GB 本地实验

2026-11-08 实测宿主GPU：NVIDIA GeForce RTX 4070 Ti，12282 MiB，compute capability 8.9；系统物理内存约64GB。WSL发行版为Ubuntu。宿主内存和GPU显存分开计算，64GB RAM不能当作64GB CUDA显存。

本次检查时，Ubuntu默认`/usr/bin/python3`没有PyTorch；可能需要先激活你日常的虚拟环境。以下脚本不会自动安装包、下载模型或改动vendor。GPU结果只有脚本成功运行才生成，不预填性能数字。

## 入口

```bash
cd ~/vllm-learn
bash labs/run-wsl.sh diagnose
# 若用已有虚拟环境：先激活，或显式指定解释器
LAB_PYTHON=/你的环境/bin/python bash labs/run-wsl.sh gpu
```

如果还没有PyTorch环境，可在项目里单独创建环境，再使用[PyTorch官方安装选择器](https://pytorch.org/get-started/locally/)选择Linux/Pip/Python与支持当前驱动的CUDA构建。以当前选择器给出的命令为准，不照搬不匹配的旧wheel。

```bash
python3 -m venv .venv-labs
source .venv-labs/bin/activate
python -m pip install numpy
# 按官方选择器安装PyTorch；安装后先跑diagnose
```

若Ubuntu缺少venv/ensurepip，先配置发行版的Python venv支持。这里不给未经核验的PyTorch版本兼容保证，也不改系统Python。

## 实验1：纯CPU调度与分页KV（不需要任何第三方包）

```bash
python3 labs/scheduler_sim.py
python3 labs/scheduler_sim.py --blocks 4 --budget 4 --out labs/results/scheduler-tight
python3 labs/scheduler_sim.py --blocks 8 --budget 4 --no-prefix --policy prefill-first --out labs/results/scheduler-prefill-first
```

打开每个输出目录里的`scheduler-trace.html`，拖动迭代滑块：观察waiting/running/finished、已知ID数与computed的一位差、positions→slot_mapping、共享引用数、free队列、前缀命中与抢占。JSON保存同一份完整轨迹。

模拟每轮都验证：引用计数等于活跃块表引用；free与ref=0一致；已计算的完整块内容对应正确前缀；computed不越过已知token。它在CPU上模拟元数据，不执行模型或真实KV张量，不含多模态/CP/异步/投机。`mixed`和`prefill-first`是教学策略，不能把迭代数当作框架延迟。finish后KV释放，所以报告中的computed归零，生成token仍保留。

做对照时**一次只改一个参数**，不要拿同时更改blocks、policy、prefix的两次结果归因。建议固定blocks=8，分别开/关prefix；再固定prefix与policy，把blocks降到4；最后比较两种policy。

## 实验2：Transformer数值实验（CPU，需NumPy）

```bash
python3 labs/transformer_lab.py all
python3 labs/inference_lab.py all
```

共7个小实验：attention、RoPE、两层KV等价、在线softmax、packed元数据、简化EPLB、DSpark人为成本曲线。详见同目录README。数值验证用float64，不代表GPU BF16误差或性能。

## 实验3：GPU attention与KV（需CUDA PyTorch，无需flash-attn包）

```bash
python3 labs/gpu_lab.py --diagnose
python3 labs/gpu_lab.py --suite correctness --dtype fp32
python3 labs/gpu_lab.py --suite attention --sizes 256 512 1024 --dtype bf16
python3 labs/gpu_lab.py --suite memory
```

- correctness：固定QKV，比较full/token/chunk attention；错误的非方阵mask必须出现偏差。这一GPU测试只覆盖attention，不宣称完整模型等价。
- attention：朴素物化FP32 scores/softmax与SDPA auto比较。auto可能选flash、memory-efficient或math后端，不保证FlashAttention；精度路径不同，不能据此宣传某框架加速比。
- memory：实际分配标准Qwen3-0.6B几何形状的BF16 KV张量，检查112KiB/token和allocator统计。默认最大T=1024，逻辑KV为112MiB；不含权重/激活/graph。

脚本默认小形状，限制T≤2048，检查至少1GiB空闲CUDA显存。算子计时做warmup、CUDA同步、3组中位数，并分开保存CUDA event与墙钟每调用耗时。不生成假TTFT或端到端tok/s。

## 实验4：CUDA Graph的地址合约

```bash
python3 labs/gpu_lab.py --suite graph
# 全部合成GPU实验一次运行
bash labs/run-wsl.sh gpu
bash labs/run-wsl.sh bf16
```

捕获小型8层矩阵/激活链，改变同一输入buffer的值，验证replay结果，再比较eager与replay。图中没有语言模型、token调度或采样，所以结果只说明这条合成算子链。输出JSON/CSV/HTML在`labs/results/gpu/`或指定目录；重复运行同一目录会覆盖上次该目录的报告，保留实验时使用不同`--out`。

## 主源

- [PyTorch SDPA文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)：boolean mask的True表示可见；非方阵is_causal默认左上对齐。因此带cache的chunk显式构造绝对位置mask。
- [PyTorch CUDA语义与CUDA Graph](https://docs.pytorch.org/docs/2.15/notes/cuda.html)：side-stream warmup、静态buffer生命周期和同步。
- 课程15–32及`reference/source-map.html`：将CPU模拟的字段映射回固定框架源码；GPU实验不导入三个vendor框架。
