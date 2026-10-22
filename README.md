# persona-duplex

轻量对话人格框架。**不微调、不做 steering、不重建 RAG**，只靠单次生成、参数更新和账本重放。

核心机制叫**位移（displacement）**：模型先写出此刻真正想说的原话，再写出实际说出口的版本。两者之间的差就是人格感的载体。原话永不发出，也永不点破。

- 纯标准库，Python ≥ 3.11，零依赖
- 单次生成产出「说出去的话」+「状态变更」
- 观测在线写，结算由用户的下一句来填

---

## 快速开始

```bash
# 不需要 API key，跑一遍全流程
python -m persona demo

# 真实对话（任何 OpenAI 兼容端点），默认流式
set PERSONA_API_KEY=sk-xxx
set PERSONA_BASE_URL=https://api.deepseek.com/v1
python -m persona chat -m deepseek-chat

# 带 trace：能看到 direct / displace / 结算 / 修复
python -m persona trace -p guarded -m deepseek-chat

# 指标
python -m persona stats
```

跑测试：

```bash
python -m unittest discover -s tests -t .
```

---

## 作为库用

```python
from persona.config import PersonaConfig
from persona.store import Store
from persona.session import Session, SessionConfig
from persona.llm import openai_compatible, openai_stream

cfg = PersonaConfig.load("personas/default.toml")
store = Store("data")

with Session(cfg, store, openai_compatible("deepseek-chat"),
             stream_llm=openai_stream("deepseek-chat"),
             options=SessionConfig(consolidate_interval=300)) as s:
    res = s.say("今天好累啊")
    print(res.say)                  # 用户看到的那一句
    print(res.parsed.head.direct)   # 真正想说的那句（永不发出）
```

`Session` 退出时会跑最后一次后台固化，否则积压会一直留在 buffer 里。

---

## 目录结构

```
persona-duplex/
├── persona/
│   ├── config.py       # 站位表、限额、人格参数
│   ├── prompt.py       # HEAD/SAY/TAIL 协议 + 上下文组装
│   ├── parse.py        # best-effort 解析 + 声明检测
│   ├── stream.py       # 增量解析，标记前缀扣留
│   ├── settle.py       # 判定：WATCH / 延迟 / 受伤
│   ├── repair.py       # 受伤检测、力度选择、修复生命周期
│   ├── similarity.py   # 字面相似度，可插拔
│   ├── topicjudge.py   # 语义话题判定，异步不阻塞
│   ├── engine.py       # 单轮编排
│   ├── session.py      # 会话 + 后台线程 + 优雅关闭
│   ├── store.py        # 账本 + 视图 + 重放
│   ├── consolidate.py  # 后台固化
│   ├── metrics.py      # 从账本算指标
│   ├── llm.py          # 零依赖 HTTP（含流式）+ mock
│   └── cli.py
├── personas/*.toml     # 三套人格配置
└── tests/              # 140 项测试
```

---

## 三套人格

人格是几个数，不是一万字人设。起作用的主要是 `bias`（把握度偏移）：

| 角色 | bias | attention | legibility | 感觉 |
|---|---|---|---|---|
| `default` | +0.00 | subtle | 2 | 完整人格 |
| `service` | +0.15 | surface | 0 | 客服型，永远对，没人味 |
| `guarded` | −0.28 | subtle | 1 | 很懂你，但隔着玻璃 |

```bash
python -m persona trace -p service    # 对照
python -m persona trace -p guarded    # 疏离
```

---

## 设计要点

### 一次生成，三段

```
<<<HEAD>>>     need / confidence / direct / displace   （必须先于 SAY）
<<<SAY>>>      唯一可见内容
<<<TAIL>>>     settle / delta / pending                （不阻塞输出）
```

`DIRECT` 必须在 `SAY` 之前——自回归的 token 顺序是唯一能保证「真的被条件化」的东西。反过来先写回复，那个 direct 就是事后编的理由。

### 位移靠留痕，不靠声明

禁止「我本来想说 X 但没说」——那是表演，而且索取信用。

正确做法是：存在一个明显的直接版本，但实际说出口的版本偏离了它，且偏离本身从不被点破。

| 类型 | 直接版 | 位移版 |
|---|---|---|
| 降级 | 「你不该辞职」 | 「你都想过哪些了？」 |
| 转移 | 「我担心你」 | 「这个点一个人回去没事吗」 |
| 截断 | 「其实我……」 | 「……算了。」 |
| 悬置 | 「他就是在骗你」 | 「他可能只是……嗯。」 |

### 观测在线写，结算下一轮填

| | 何时可得 | 由谁产生 |
|---|---|---|
| **观测** | 本轮 | 模型自己 |
| **结算** | 下一轮 | **用户的下一句** |

不需要独立裁判模型。`TAIL.pending` 挂一个预言，下一轮看用户实际怎么反应。

### 话题判定：语义优先，异步兜底

字面 bigram 只认原话。用户说「投了两家简历」而预言写的是「会说出某个具体方案」，
字面重合接近零，语义上却完全对上了。所以接一个轻模型做判断：

```toml
[judge]
enabled = true
grace = 0.35    # 收集时最多等这么久
```

三个设计约束：

1. **异步** —— 判定在主生成的同时并行跑，不在关键路径上
2. **不开思考模式** —— 这是个分类任务，`temperature=0`、`max_tokens=120` 就够
3. **grace 而非 timeout** —— 收的时候最多等 0.35 秒，拿不到就立刻用字面兜底

第 3 条是关键。最初实现用「超时 3 秒」，实测发现**判定慢于主生成时它会变成瓶颈**：

```
主生成 1.0s 时
  无判定器        1.000s
  判定  2.0s      1.000s   (+0.000)   ← 生成期间跑完了，零等待
  判定 10.0s      1.016s   (+0.016)   ← 拿不到，立刻兜底

主生成 0.1s 时（生成比判定快）
  判定  2.0s      0.359s   (+0.266)   ← 封顶在 grace，不会无限等
```

没拿到的判定不浪费：预言还挂着，下一轮会被重新判定。

```bash
python -m persona chat --no-judge     # 关掉，退回纯字面
python -m persona chat -j gpt-4o-mini # 判定单独用轻模型
```

### 「不接话题」有三种含义，当轮不可区分

① 只是不关注　② 回避　③ 被冒犯但忍耐

所以不能塞进 confirmed / refuted——那测的不是读得准不准，是这段关系有没有被碰伤。

| 用户下一轮 | 判定 | 记账 |
|---|---|---|
| 接住·一致 / 冲突 | `CONFIRMED` / `REFUTED` | 校准账本 |
| 不接话题 | `WATCH` | 暂不记账 |
| ↳ 话题后来回来 | `CONFIRMED'` | 校准 + 关系 |
| ↳ 语域变正式/变短 | `SUSPECTED_HURT` | 关系账本 |
| ↳ 超时（默认 72h） | `EXPIRED` | 丢弃 |

`watch.ttl_hours` 可配。

### 受伤与修复

用户被冒犯之后不会说「你冒犯我了」，他会把句子变短、不再用语气词。所以检测靠语域，不靠内容。

修复是「位移不许声明」的唯一例外：

```
声明位移 = 索取信用（关于我）
承认损伤 = 请求修复（关于你）
```

力度要匹配线索强度，越弱越要轻。而且修复不在受伤的当轮发——当场道歉往往是在处理自己的尴尬。

### 三层状态 + 在线白名单

| 层 | 内容 | 谁能写 |
|---|---|---|
| 快层 | read / direct / displace | 用完即弃 |
| 中层 | relation / pending | 模型，受限 |
| 慢层 | judgments / self-story | 只有后台线程 |

`relation.closeness` 每轮最多 ±0.05，`pending` 每轮最多 1 条，慢层字段在线不可写。

### 账本优先

```
data/
├── ledger.jsonl       # 追加式，唯一真相源
├── state/             # 物化视图，可随时重建
├── judgments.jsonl    # 判断记忆，可推翻
├── self-story.md      # 「原来 X → 现在 Y → 因为 Z」
└── buffer/            # 后台未处理积压
```

`state/` 坏了不要紧：`python -m persona rebuild`。也可以拿同一份账本、不同版本的代码重放，对比参数策略。

### 一致性是尺子，不是目标

```
可读性 = 位移量 ÷ 该角色的基线
```

同一个含糊，出现在一向直的人嘴里是响的，出现在一向软的人嘴里是无声的。没有稳定基线，位移就不可测。所以 `bias` 和 `legibility_max` 必须稳，它们是分母。

---

## 指标

`python -m persona stats`，全部从账本重放计算，只读不改：

```
指标
──────────────────────────────────────────────────────
  ! 位移率     62% —— 每句都留白，是谜语人
  ✓ 声明违规    无
  · 校准      样本只有 2 条，还算不准
  · 命中率     100%（2/2）
  ✓ 受伤修复    1 次受伤，1 次修复，成功率 100%
```

样本少于 4 条时不报校准偏差；只有「高把握 + 低命中」才算在读空气，反过来是保守，不该报警。

---

## 扩展

换 embedding 只需要一行：

```python
from persona.similarity import EmbeddingSimilarity, HybridSimilarity

sim = HybridSimilarity(EmbeddingSimilarity(embed=my_embed_fn))
sess = Session(cfg, store, llm, similarity=sim)
```

默认是汉字 bigram，零依赖但只认字面。`HybridSimilarity` 取字面与语义的最大值，embedding 挂了会自动退回字面。

---

## 已知限制

| 限制 | 现状 | 影响 |
|---|---|---|
| 话题判定要一次额外调用 | 已接轻模型，异步 + grace 兜底 | 模型慢时退化为字面，判定延到下一轮 |
| 受伤检测是启发式 | 看语域、语气词、句长 | 沉默寡言的人基线低，容易误判 |
| 修复只有一次机会 | 两次没回应就转雷区 | 故意的：反复道歉是二次伤害 |
| 后台提炼无模型时很笨 | 只认重复出现的行为倾向 | 接 LLM 才会提炼有洞察的判断 |
| 没有多用户 | 单 store 单用户 | 多用户需给 store 加命名空间 |
| bias 校准幅度极小 | 每次 ±0.005 | 需要很多轮才见效，但这是对的 |

---

## 硬约束

1. `DIRECT` 必须先于 `SAY` 写
2. `DIRECT` 永不进入输出（**修复除外**——承认损伤是请求修复，不是索取信用）
3. 一轮最多一次位移
4. 位移只在情绪承重时发生
5. 无位移是常态

---

## 参考

- [The Assistant Axis](https://www.anthropic.com/research/assistant-axis) — 人格漂移与稳定
- [Persona Vectors](https://www.anthropic.com/research/persona-vectors) — 人格作为可测方向
- [The Personality Illusion](https://arxiv.org/abs/2509.03730) — 自述 ≠ 行为
- [A-MEM](https://arxiv.org/abs/2502.12110) — memory evolution
- [MIRIX](https://arxiv.org/abs/2507.07957) — Core 层与事件层分离
- [AutoPersonas](https://arxiv.org/abs/2607.08252) — 人格锁死
- [Feeling Right vs. Being Right](https://aclanthology.org/2026.acl-long.2046/) — 谄媚与校准的摩擦

---

## 许可

MIT
