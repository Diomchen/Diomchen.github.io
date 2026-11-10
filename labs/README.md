> 公开课程说明：以下4070 Ti / 64GB / WSL记录来自作者的验证环境，不表示检测访问者设备。请先按[下载与运行指南](../reference/start-here.html)下载完整课程。GPU实测尚未完成。

# 数值实验

## 维度转换：逐元素打印与核对

配合 [逐格维度图解](../reference/dimension-walkthrough.html) 运行：

```bash
cd ~/vllm-learn
python3 labs/dimension_lab.py
# 也可：bash labs/run-wsl.sh dimensions
```

只需 Python 标准库。六组实验覆盖 reshape / transpose / 正确与错误合头、QKV 不等宽切片、attention 收缩轴、packed 与 paged 索引、SwiGLU 和普通 Linear 的 TP。先预测某个元素的新坐标，再对照打印；列表实验不冒充 PyTorch 的物理 stride 或 GPU 性能验证。

## Transformer 与框架数值实验

4070 Ti / 64GB / WSL的完整实验入口见 [LOCAL-LABS.md](LOCAL-LABS.md)：包含无依赖调度模拟、逐轮HTML报告以及4组GPU合成实验脚本。GPU脚本需要已有CUDA PyTorch环境；默认WSL解释器本次未发现PyTorch，未宣称GPU结果已验证。

在项目根目录运行，需要 Python 3 与 NumPy；不下载模型、不修改 vendor、不要求 CUDA。

```bash
python labs/transformer_lab.py all
python labs/inference_lab.py all
```

- `attention`：查看 scores / weights / output。
- `rope`：半切布局、范数、共同位置平移的不变量。
- `cache`：两层随机 toy decoder（GQA、Q/K RMSNorm、RoPE、SwiGLU），验证完整前向与逐 token / chunked KV 前向的数值等价；错误 mask 是负对照，必须产生偏差。
- `online`：在线 softmax 的 tile 合并与完整 softmax 对照。
- `packed`：多请求的 positions、query_start_loc、seq_lens、slot_mapping。
- `eplb`：单历史步、单层的副本放置手算例，不是完整 EPLB 实现。
- `spec`：人为 graph 耗时曲线下的 DSpark 预算直觉与 vLLM 回退账本，不是实测。

数学实验以 float64 小随机矩阵隔离算法问题，不能代表 BF16 GPU kernel 的误差范围、模型质量或吞吐。单卡GPU实跑沿原课程07，在既有WSL环境中完成；用 `benchmark-template.csv` 记真实数据，未知指标留空。只有总generate耗时，不能编造TTFT/ITL。

每次练习先预测结果，再运行；隔1、3、7天不看答案重做。自评只保存在页面的浏览器localStorage，学习掌握记录需后续对话中有实际证据后再更新。
