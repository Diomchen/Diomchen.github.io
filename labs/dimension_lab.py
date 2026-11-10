#!/usr/bin/env python3
"""用标准库追踪维度、元素和索引；不需要 NumPy、PyTorch 或 GPU。

默认小例：B=1, T=3, H=4, Hq=2, Hkv=1, d=2, F=6。
Python 列表只验证逻辑元素顺序；并不测量 PyTorch 的 view/copy/stride 行为。
运行：python3 labs/dimension_lab.py [all|layout|qkv|attention|packed|mlp|tp]
"""

import argparse
import itertools
import math


B, T, H, HQ, HKV, D, F = 1, 3, 4, 2, 1, 2, 6
X = [[1, 2, 0, 1], [0, 1, 1, 0], [2, 0, 1, 1]]
# 数学约定 W[in,out]；torch Linear.weight 会存这个矩阵的转置。
W_QKV = [
    [1, 0, 0, 1, 1, 0, 2, 0],
    [0, 1, 1, 0, 0, 1, 0, 2],
    [1, 1, 0, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 0, 1, 0, 1],
]


def flat(values):
    if isinstance(values, list):
        return [value for item in values for value in flat(item)]
    return [values]


def reshape(values, shape):
    """按现有逻辑遍历顺序重新分组；这里总会建立 Python 列表。"""
    values = flat(values)
    if math.prod(shape) != len(values):
        raise ValueError(f"元素数不同：{len(values)} -> {shape}")
    iterator = iter(values)

    def build(dims):
        if not dims:
            return next(iterator)
        return [build(dims[1:]) for _ in range(dims[0])]

    return build(shape)


def at(values, index):
    for coordinate in index:
        values = values[coordinate]
    return values


def permute(values, shape, order):
    """建立轴交换后的逻辑列表；不模拟 torch.permute 的共享存储。"""
    assert sorted(order) == list(range(len(shape)))
    new_shape = [shape[axis] for axis in order]
    reordered = []
    for new_index in itertools.product(*(range(size) for size in new_shape)):
        old_index = [0] * len(shape)
        for new_axis, old_axis in enumerate(order):
            old_index[old_axis] = new_index[new_axis]
        reordered.append(at(values, old_index))
    return reshape(reordered, new_shape)


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def matmul(a, b):
    assert a and b and len(a[0]) == len(b)
    assert all(len(row) == len(a[0]) for row in a)
    assert all(len(row) == len(b[0]) for row in b)
    return [[sum(left * right for left, right in zip(row, column))
             for column in zip(*b)] for row in a]


def show(label, values):
    print(f"\n{label}")
    for row in values:
        if isinstance(row, list):
            print("  " + str([round(v, 6) if isinstance(v, float) else v for v in row]))
        else:
            print(f"  {row}")


def projection():
    fused = matmul(X, W_QKV)
    q = [row[:HQ * D] for row in fused]
    k = [row[HQ * D:(HQ + HKV) * D] for row in fused]
    v = [row[(HQ + HKV) * D:] for row in fused]
    return fused, q, k, v


def layout_lab():
    print("\n=== 分头 / 转置 / 合头：每个数字是唯一元素编号 ===")
    source = reshape(list(range(B * T * HQ * D)), [B, T, HQ * D])
    split = reshape(source, [B, T, HQ, D])
    heads = permute(split, [B, T, HQ, D], [0, 2, 1, 3])
    wrong_heads = reshape(source, [B, HQ, T, D])
    show("投影后 Q[0,:,:]，形状 [T,Hq*d]=[3,4]", source[0])
    for head in range(HQ):
        show(f"先 reshape 后 transpose：Q[0,{head},:,:] [T,d]=[3,2]", heads[0][head])
    show("错误：直接 reshape 成 [B,Hq,T,d] 时的 head 0", wrong_heads[0][0])
    assert heads != wrong_heads
    for batch, token, head, component in itertools.product(range(B), range(T), range(HQ), range(D)):
        assert heads[batch][head][token][component] == source[batch][token][head * D + component]
    restored = permute(heads, [B, HQ, T, D], [0, 2, 1, 3])
    correct_merge = reshape(restored, [B, T, HQ * D])
    wrong_merge = reshape(heads, [B, T, HQ * D])
    assert correct_merge == source
    assert wrong_merge != source
    show("合头正确：先回到 [B,T,Hq,d]，再合并末两轴", correct_merge[0])
    show("合头错误：把 head-major 的逻辑顺序直接 reshape", wrong_merge[0])
    print("索引等式：heads[b,h,t,j] = source[b,t,h*d+j]。")
    print("编号 6：source[0,1,2] -> split[0,1,1,0] -> heads[0,1,1,0]。")
    print("连续 Q 的示意 stride：[12,4,1] -> [12,4,2,1] -> [12,2,4,1]。")
    print("最后一步仅按轴重解释；torch.transpose 通常不搬数据，contiguous 必要时才复制。")
    # B=1 容易掩盖 batch 轴错误，独立用 B=2 检查所有坐标。
    two_batches = reshape(list(range(2 * T * HQ * D)), [2, T, HQ * D])
    two_heads = permute(reshape(two_batches, [2, T, HQ, D]), [2, T, HQ, D], [0, 2, 1, 3])
    for b, t, h, j in itertools.product(range(2), range(T), range(HQ), range(D)):
        assert two_heads[b][h][t][j] == two_batches[b][t][h * D + j]
    print("PASS：全部坐标映射、正确往返与错误反例成立；B=2 也通过。")


def qkv_lab():
    print("\n=== QKV：矩阵乘法改变数值，split / 分头只改变组织方式 ===")
    fused, q, k, v = projection()
    show("X [T,H]=[3,4]（省略 B=1）", X)
    show("数学 Wqkv [H,8]=[4,8]", W_QKV)
    show("X @ Wqkv -> fused [3,8]", fused)
    for name, tensor in (("Q [3,4]", q), ("K [3,2]", k), ("V [3,2]", v)):
        show(name, tensor)
    assert [q[i] + k[i] + v[i] for i in range(T)] == fused
    assert len(flat(q)) + len(flat(k)) + len(flat(v)) == 24
    assert HQ * D == 4 and HKV * D == 2
    torch_weight = transpose(W_QKV)
    assert len(torch_weight) == 8 and len(torch_weight[0]) == 4
    for t in range(T):
        for out in range(8):
            assert fused[t][out] == sum(X[t][i] * torch_weight[out][i] for i in range(H))
    print("split([4,2,2], dim=-1)，不能平均切三份。Q 是 [1,3,2,2]，K/V 是 [1,3,1,2]。")
    print("例 fused[0,2] = 1*0 + 2*1 + 0*0 + 1*1 =", fused[0][2])
    print("PyTorch weight 为 [8,4]，因此运算是 X @ weight.T。")
    print("本例 H=Hq*d=4 只是取小数字；Qwen3-0.6B 是 H=1024、Hq*d=2048。")
    print("PASS：不等宽切分可复原，投影的每个输出都与显式点积相等。")


def attention_lab():
    print("\n=== Attention：QK^T 消去 d，AV 消去 key 位置 Tk ===")
    print("这里沿用本脚本 QKV 投影的数值，省略 Q/K norm 与 RoPE；网页 attention 交互是独立的小矩阵例子。")
    _, q, k, v = projection()
    q_heads = permute(reshape(q, [T, HQ, D]), [T, HQ, D], [1, 0, 2])
    outputs = []
    for head in range(HQ):
        scores = matmul(q_heads[head], transpose(k))
        weights = []
        for t, row in enumerate(scores):
            visible_scores = [value / math.sqrt(D) for value in row[:t + 1]]
            maximum = max(visible_scores)
            exponentials = [math.exp(value - maximum) for value in visible_scores]
            denominator = sum(exponentials)
            weights.append([value / denominator for value in exponentials] + [0.0] * (T - t - 1))
        output = matmul(weights, v)
        outputs.append(output)
        show(f"Q head {head} [Tq,d]=[3,2]，两个 Q 头共享 KV head 0", q_heads[head])
        show(f"head {head} 未缩放 QK^T [Tq,Tk]=[3,3]", scores)
        show(f"head {head} 缩放 + 因果 mask + softmax 后 A [3,3]", weights)
        show(f"head {head} O=A@V [3,2]", output)
        for t in range(T):
            assert math.isclose(sum(weights[t]), 1.0, abs_tol=1e-12)
            assert all(weights[t][j] == 0 for j in range(t + 1, T))
            for key in range(T):
                assert scores[t][key] == sum(q_heads[head][t][j] * k[key][j] for j in range(D))
            for j in range(D):
                expected = sum(weights[t][key] * v[key][j] for key in range(t + 1))
                assert math.isclose(output[t][j], expected, abs_tol=1e-12)
    assert outputs[0] != outputs[1]
    merged = reshape(permute(outputs, [HQ, T, D], [1, 0, 2]), [T, HQ * D])
    show("每个 token 拼接各 Q 头输出 [T,Hq*d]=[3,4]", merged)
    print("S[h,t,k] = sum_j Q[h,t,j]*K[0,k,j]；O[h,t,j] = sum_k A[h,t,k]*V[0,k,j]。")
    print("PASS：收缩轴、归一化、因果遮罩、GQA 的独立 Q 头输出均验证。")


def packed_lab():
    print("\n=== Packed 与分页：不要把去 padding 当成 reshape ===")
    requests = [["A0", "A1"], ["B0", "B1", "B2"]]
    padded = [requests[0] + ["PAD"], requests[1]]
    packed = [token for request in requests for token in request]
    cu_q = [0]
    for request in requests:
        cu_q.append(cu_q[-1] + len(request))
    print("padded [B,Tmax]=[2,3]：", padded)
    print("直接 flatten 得6项：", flat(padded))
    print("筛选有效 token 后 packed [N]=[5]：", packed)
    print("cu_seqlens_q [B+1]=[3]：", cu_q)
    assert cu_q == [0, 2, 5]
    assert "PAD" not in packed and len(flat(padded)) != len(packed)
    for request in range(len(requests)):
        assert packed[cu_q[request]:cu_q[request + 1]] == requests[request]
    # 加入已有缓存后，本轮 query packed 行号、绝对位置、物理 slot 三者不同。
    cached = [4, 0]
    positions = [cached[r] + local for r, request in enumerate(requests) for local in range(len(request))]
    cu_k = [0]
    for prefix, request in zip(cached, requests):
        cu_k.append(cu_k[-1] + prefix + len(request))
    block_size = 2
    block_tables = [[3, 1, 4], [0, 2]]
    slots = []
    rows = []
    for r, request in enumerate(requests):
        for local, token in enumerate(request):
            position = cached[r] + local
            logical_block, offset = divmod(position, block_size)
            physical_block = block_tables[r][logical_block]
            slot = physical_block * block_size + offset
            slots.append(slot)
            rows.append([cu_q[r] + local, token, position, logical_block, physical_block, offset, slot])
    show("列：packed行 / token / 绝对位置 / 逻辑块 / 物理块 / 块内偏移 / slot", rows)
    assert positions == [4, 5, 0, 1, 2]
    assert cu_k == [0, 6, 9]
    assert slots == [8, 9, 0, 1, 4]
    assert len(set(slots)) == len(slots)
    payloads = [[10 * i + j for j in range(HKV * D)] for i in range(len(packed))]
    physical_cache = {slot: payload for slot, payload in zip(slots, payloads)}
    assert [physical_cache[slot] for slot in slots] == payloads
    print("cu_seqlens_k =", cu_k, "；query总数5，key总数9，不能用同一组边界代替。")
    print("每个 slot 保存 Hkv*d=2 个数；这是单层 K 的示意，V 独立同形。")
    print("PASS：变长切片还原、cache位置偏移和分页写读映射一致。")


def mlp_lab():
    print("\n=== SwiGLU：两条 F 宽分支逐元素相乘，再投回 H ===")
    w_gate_up = [[((i + 1) * (j + 2)) % 5 - 2 for j in range(2 * F)] for i in range(H)]
    gate_up = matmul(X, w_gate_up)
    gate = [row[:F] for row in gate_up]
    up = [row[F:] for row in gate_up]
    activated = [[g / (1 + math.exp(-g)) * u for g, u in zip(gs, us)] for gs, us in zip(gate, up)]
    down_weight = [[(i + j) % 3 - 1 for j in range(H)] for i in range(F)]
    output = matmul(activated, down_weight)
    show("gate_up [3,12]", gate_up)
    show("gate [3,6]", gate)
    show("up [3,6]", up)
    show("SiLU(gate) * up [3,6]，* 为逐元素乘", activated)
    show("乘 down 的数学 W[6,4] 后 [3,4]", output)
    assert [g + u for g, u in zip(gate, up)] == gate_up
    assert len(flat(activated)) == T * F
    assert len(flat(output)) == T * H
    for t in range(T):
        for j in range(F):
            expected = gate[t][j] / (1 + math.exp(-gate[t][j])) * up[t][j]
            assert math.isclose(activated[t][j], expected, abs_tol=1e-12)
    print("PASS：split 的元素守恒、逐元素门控及输出残差宽度均正确。")


def tp_lab():
    print("\n=== 两路 Tensor Parallel：只演示普通 Linear 的切分 ===")
    print("注意本课 Hkv=1 不满足 nano 的 TP=2 整除限制；以下不是两卡 Qwen 配置。")
    weight = [[1, 0, 2, -1], [0, 1, -1, 2], [1, 1, 0, 1], [2, 0, 1, 1]]
    full = matmul(X, weight)
    column_parts = [matmul(X, [row[start:start + 2] for row in weight]) for start in (0, 2)]
    cat = [left + right for left, right in zip(*column_parts)]
    row_parts = [matmul([row[start:start + 2] for row in X], weight[start:start + 2]) for start in (0, 2)]
    summed = [[left + right for left, right in zip(a, b)] for a, b in zip(*row_parts)]
    assert full == cat == summed
    show("完整 X[3,4] @ W[4,4] -> Y[3,4]", full)
    show("列并行 rank0 输出 [3,2]（数学W按输出列切）", column_parts[0])
    show("列并行 rank1 输出 [3,2]（沿最后一维cat可还原）", column_parts[1])
    show("行并行 rank0 partial [3,4]（数学W按输入行切）", row_parts[0])
    show("行并行 rank1 partial [3,4]（逐元素all-reduce sum还原）", row_parts[1])
    assert [left + right for left, right in zip(*row_parts)] != full
    print("PyTorch weight=[out,in]，所以数学W的列分片对应weight的行分片。")
    print("PASS：列并行cat、行并行sum均还原未切分结果；行并行cat反例不等价。")


LABS = {"layout": layout_lab, "qkv": qkv_lab, "attention": attention_lab,
        "packed": packed_lab, "mlp": mlp_lab, "tp": tp_lab}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lab", choices=["all", *LABS], nargs="?", default="all")
    args = parser.parse_args()
    print("维度实验：纯 Python 标准库；数值和索引教学，不是 GPU 性能测试。")
    if args.lab == "all":
        for run in LABS.values():
            run()
    else:
        LABS[args.lab]()
    print("\n所选实验完成；所有等式与预期失败反例的断言均通过。")


if __name__ == "__main__":
    main()
