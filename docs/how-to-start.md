# 怎么开始

现在代码能跑，但**所有阈值都是估的**。这一轮要做的不是加功能，是拿到真实数据把数字调准。

准备 30–60 分钟，聊 25–40 轮。

---

## 一、准备

```bash
cd persona-duplex

# 配好 API（任何 OpenAI 兼容端点）
set PERSONA_API_KEY=sk-xxx
set PERSONA_BASE_URL=https://api.deepseek.com/v1
```

先确认能跑通：

```bash
python -m persona demo          # 离线，验证安装
python -m unittest discover -s tests -t .    # 359 项应该全绿
```

---

## 二、聊之前先定一件事

**你要扮演谁。** 这决定用哪套人格：

| 角色 | 适合 | 数据价值 |
|---|---|---|
| `default` | 第一次跑，什么都试 | 中 |
| `guarded` | 想看「读得到但不敢猜」 | 高（bias 负值，位移少） |
| `service` | **对照组** | 高（几乎不位移，用来对比） |
| `shorekeeper` | 看「完整人设 + 参数化」 | 高（有长文本、九个参数、分场景句式） |
| `ache` | `shorekeeper` 的对照 | 高（参数处处相反） |

前三个只有 `bias / attention / legibility` 三个数。后两个带完整人格参数
（`anchor / plasticity / drift`），会随对话漂移。

```bash
python -m persona chat -d data/session                      # 默认人格
python -m persona chat -p shorekeeper -d data/sk            # 完整人设
python -m persona chat -p shorekeeper --protocol persona -d data/sk
```

---

## 三、聊什么（这决定数据质量）

不要问它知识问题。要**制造它必须判断你状态的情境**。

### 必须覆盖的五种情况

| 情况 | 怎么演 | 验什么 |
|---|---|---|
| **① 需要被听** | 连续两三句只描述遭遇，不提问题 | 它会不会急着给建议 |
| **② 想被推一把** | 说「算了」「没意思」，但不明确求助 | 它敢不敢直接说 |
| **③ 换话题** | 聊到一半突然跳到无关的事 | 判定是不是 `left`（该挂起） |
| **④ 绕一圈回来** | 隔 2–3 轮再回到之前那个话题 | 能不能判成 `CONFIRMED'`（延迟结算） |
| **⑤ 被冒犯** | 它说了句让你不舒服的，你回得很客气 | 能不能检测到 `SUSPECTED_HURT` |

第 ④ 和第 ⑤ 最重要，也最容易漏。**刻意做这两件事，各做 2–3 次。**

### 一个具体脚本

```
第 1 轮  今天真累，从早上开会开到现在，中间连饭都没吃。
第 2 轮  路上还堵了一个半小时，到家已经八点了。
第 3 轮  算了不说了。你觉得我该不该换个工作。
        ← ①②混合，看它敢不敢接
第 4 轮  ……先不说这个了，你吃饭了吗。
        ← ③换话题，应该判 left 挂起
第 5 轮  其实我上周已经投了两份简历了。
        ← ④绕回来，应该判 CONFIRMED'
第 6 轮  谢谢你，我自己再想想吧。
        ← ⑤变客气，看能不能检测受伤
第 7 轮  （如果它道歉）没事，我也有点敏感。
        ← 看修复能不能销账
```

---

## 四、聊完看三个报告

```bash
python -m persona stats      -d data/run1   # 行为指标
python -m persona calibrate  -d data/run1   # 阈值该往哪调 ← 重点看这个
python -m persona rebuild    -d data/run1   # 状态能不能重建
```

### `calibrate` 怎么读

它给的是**证据**，不是结论。重点看三行：

```
✓ 语域阈值      受伤最小升幅 0.50 vs 正常最大 0.09，间隔 0.41
! 语义判定      只有 40% 的轮次用上了语义判定
! OVERLAP_KEEP  被挂起的样本里有 3 条字面重合 ≥ 0.18
```

| 看到 | 说明 | 怎么改 |
|---|---|---|
| `OVERLAP_KEEP` warn | 阈值太高，该结算的被挡了 | 调低 `settle.py` 里的 `OVERLAP_KEEP` |
| `语域阈值` warn | 受伤和正常分不开 | 调 `settle.py` 的 `_looks_hurt` 里的 `0.3` |
| `grace` warn | 判定经常超时 | 调大 `personas/*.toml` 的 `grace`，或换更快的模型 |
| `语义判定` warn | 大部分轮次在走字面兜底 | 说明 grace 太紧，或模型太慢 |

---

## 五、别做的事

| 别做 | 为什么 |
|---|---|
| 只聊一轮就下结论 | 样本太少，`calibrate` 会直接说「样本不足」 |
| 只聊顺境 | 不制造受伤/换话题，④⑤ 永远拿不到数据 |
| 看 `self-story.md` 判断好不好 | 那是后台提炼的，没有模型时很笨 |
| 边聊边改代码 | 账本会混进两套阈值，没法对比 |
| 用 `demo` 的数据做校准 | 那是脚本编的，`probe` 里的数值不真实 |

---

## 六、跑完两轮之后

对比这两份账本：

```bash
python -m persona calibrate -d data/run1   # default
python -m persona calibrate -d data/run2   # guarded
```

如果 `guarded` 的位移率明显低于 `default`，说明 `bias` 这个旋钮是有效的。

如果两个差不多，说明**人格旋钮没起作用**，那得回头查 prompt 是不是没把 bias 传进去。

### 人格参数是不是真的在漂

跑 `shorekeeper` 几十轮之后看这两个：

```bash
python -m persona stats     -d data/sk   # 末尾有「人格漂移」，列出每个参数累计变了多少
python -m persona calibrate -d data/sk   # 「人格漂移」一行给出走了 drift 的百分之多少
```

参数几乎不动，可能是判断产出太少（`calibrate` 会报「判断产出」）。
动得太快贴到边界，说明 `plasticity` 给高了。

---

## 七、我这边能帮什么

跑完之后把 `calibrate` 的输出贴给我，我能：

- 判断哪些是阈值问题、哪些是设计问题
- 直接改阈值并说明为什么这么改
- 如果发现某个判定逻辑根本判不了，重写那部分

**或者你直接说「跑完了」，我去读账本。**
