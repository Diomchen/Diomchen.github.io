# Transformer 与推理框架学习课程

网站：https://diomchen.github.io/

32节中文课，46张SVG，维度交互图解、nano-vllm/vLLM/DeepInfer固定源码摘录，以及CPU与CUDA合成实验脚本。

- [学习入口](index.html)
- [使用、下载和固定源码准备](reference/start-here.html)
- [维度图解](reference/dimension-walkthrough.html)
- [本地实验](labs/README.md)
- [第三方源码归属](reference/attribution.html)

## 本地运行

```bash
python3 -m http.server 8000
python3 labs/dimension_lab.py
python3 labs/scheduler_sim.py
```

后两项只使用Python标准库。NumPy数值实验和GPU实验依赖见labs文档；GPU性能尚未实际测量。

本仓库是可直接发布的静态课程，不包含上游完整vendor、个人学习记录、虚拟环境、模型或本机缓存。
GitHub Pages从main分支根目录发布，.nojekyll保留原始HTML。旧博客保留于Git历史，网站内容已替换。
