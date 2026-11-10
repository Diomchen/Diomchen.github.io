# 推理框架 Resources

> 本课程依据论文、模型配置与固定版本源码编写。在线文档可能更新；源码证据以课程标明的commit为准。

## Knowledge

- [源码：nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（本地 `vendor/nano-vllm` @ `bb823b3`）
  1450 行从零实现的 vLLM。**本课程的主语料**。用于：每一个概念的「最小可读实现」。
- [源码：`nanovllm/engine/llm_engine.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3/nanovllm/engine/llm_engine.py)
  90 行引擎骨架，第 1 课的全部内容。用于：循环结构、prefill/decode 分支、吞吐统计。
- [论文：PagedAttention / vLLM（SOSP'23）](https://arxiv.org/abs/2309.06180)
  分页 KV 管理的原始论文，附 2–4× 吞吐数据。用于：为什么 block 化能省显存、为什么能共享前缀。只读 §1–3。
- [vLLM 官方文档](https://docs.vllm.ai/en/latest/)
  生产实现的权威说明。用于：对照 nano-vllm 的简化，查真实参数与行为。
- [vLLM 发布博客（2023-06-20）](https://blog.vllm.ai/2023/06/20/vllm.html)
  第一版设计动机，比论文短。用于：10 分钟建立全局直觉。
- [论文：Orca（OSDI'22）](https://www.usenix.org/conference/osdi22/presentation/yu)
  提出 continuous batching / iteration-level scheduling 的系统。用于：调度器设计的源头，面试常问「continuous batching 和 static batching 差在哪」。
- [论文：FlashAttention](https://arxiv.org/abs/2205.14135)
  IO-aware 注意力，nano-vllm 直接调它的 kernel。用于：理解 prefill 的 `varlen` 与 decode 的 `with_kvcache` 两种调用形态。
- [论文：SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)
  用基数树做前缀共享的另一种解法。用于：对比 prefix caching 的两种实现路线。
- [论文：DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)
  MLA、MoE、MTP 的出处，DeepInfer 服务的模型家族。用于：进入 DeepInfer 前的模型侧背景。
- [源码：DeepInfer](https://github.com/deepseek-ai/DeepInfer)
  DeepSeek 官方推理框架，README 自述「originally built primarily on vLLM」。用于：课程第三阶段；注意需 SM100 / Ascend 950，本机只能读。
- [模型：Qwen3-0.6B（hf-mirror 镜像）](https://hf-mirror.com/Qwen/Qwen3-0.6B)
  课程实验用模型：28 层、8 KV head、head_dim 128、bf16。用于：所有实跑与显存计算。

## Wisdom (Communities)

- [vLLM GitHub Discussions](https://github.com/vllm-project/vllm/discussions)
  维护者与用户在同一个地方讨论设计取舍。用于：提 PR 前确认「这是不是想要的改法」；也是找 good-first-issue 的地方。
- [DeepInfer GitHub Discussions](https://github.com/deepseek-ai/DeepInfer/discussions)
  新项目，讨论少但作者在场。用于：DeepInfer 阶段的提问与 PR 方向。
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
  大量真实部署者的吞吐/显存实测。用于：验证自己的 benchmark 数字是否离谱。
- 中文：知乎「推理加速 / vLLM」话题下的高赞长文
  用于：中文语境下的面试题复盘。**注意**：质量参差，只作补充，事实以源码和论文为准。

## Gaps

- **缺少**：系统的中文「从零实现推理引擎」教程。目前只能靠源码 + 英文论文，课程本身在补这个缺口。
- **缺少**：DeepInfer 的公开设计文档（项目 2026-10 才开源），只能读代码与 README。
- **缺少**：4070 Ti 级别显卡上的权威 benchmark 数据，需要自己跑才有可信数字（这正是课程后期的作品）。


## 2026-11-08 主源补充与核验边界

以下论文与模型config已通过官方网页获取并阅读相关章节/摘要。web搜索工具未返回结果，使用requests获取主源；三个框架按本地固定commit直接核对，未更新vendor，也不将其称为最新。完整实现证据与行号见`docs/research/framework-deep-dive-2026-11-08.md`、`docs/research/source-manifest.json`。

### Knowledge

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，§3.1–3.4。用于15–17课：结构、Q/K/V、缩放、mask；与现代Qwen3的norm/FFN/RoPE差别明确区分。
- [RoFormer](https://arxiv.org/abs/2104.09864)，§3.2.1–3.2.2。用于19课：二维旋转、相对位置；半切布局以本地源码为准。
- [GQA](https://arxiv.org/abs/2305.13245)。用于18课：MHA/MQA/GQA的KV头共享与uptraining前提。
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)。用于20课：RMSNorm不减均值；具体FP32累积由本地源码核验。
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)。用于20课：门控FFN和SwiGLU，与nano的SiluAndMul对照。
- [FlashAttention](https://arxiv.org/abs/2205.14135)。用于22课：IO-aware精确attention、tile与在线softmax；不将IO下降误写为dense算术复杂度变线性。
- [Qwen3-0.6B官方config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)。用于15/18/21课具体数字；2026-11-08快照保存在`docs/research/qwen3-0.6b-config.json`，H=1024、Hq=16、Hkv=8、d=128、L=28。
- [vLLM固定提交CUDA Graph设计](https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/docs/design/cuda_graphs.md)。用于29课和旧课10勘误：FULL的backend能力与执行边界。
- [DeepInfer固定提交部署QA](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md)。用于32课SWA近似重建；其中内部服务设计不直接等同开源scheduler实现。

### Gaps

- 课程参考硬件4070 Ti的真实GPU数据仍待运行；CPU数学等价实验不能填补性能证据。
- 部分第三方内核子模块未展开，本课程不根据依赖名称猜测全部职责。
- 发布日期、最新版API和社区是否接收某PR未在本次研究中作完整核验，课程不依赖这些断言。
