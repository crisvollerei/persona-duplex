"""Fixed, bounded follow-up study. Credentials are accepted only via environment."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import threading
import time
import unicodedata
from uuid import uuid4

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from persona.config import PersonaConfig
from persona.consolidate import Consolidator
from persona.engine import TurnEngine
from persona.llm import openai_compatible, openai_stream, text_prompt
from persona.prompt import _trait_block
from persona.store import Store, now_iso, rebuild_views

SEED = 20261106
PARAMS = dict(model="deepseek-chat", temperature=0.7, max_tokens=1800, timeout=120.0)
BASE_URL = "https://api.deepseek.com/v1"
HARD_LIMIT = 210
CONTROLLED = ("supported", "irrelevant", "unresolved", "stale_corrected", "empty_corrected", "retracted")
WORKERS = 4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    secret = os.environ.get("PERSONA_API_KEY")
    if secret and secret in encoded: raise RuntimeError("Credential found in output")
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def configs():
    source = HERE / "personas"
    blue = PersonaConfig.load(source / "blue_whale.toml")
    dragon = PersonaConfig.load(source / "white_dragon.toml")
    neutral = copy.deepcopy(blue)
    neutral.role.name = "中性表达基准"
    neutral.role.identity = ""
    neutral.role.voice = "自然中文，按情境回应。"
    for trait in neutral.role.traits.values():
        trait.anchor = trait.value = 0.5
    return {"neutral": neutral, "blue": blue, "dragon": dragon}


def facts():
    result = []
    examples = [
        ("budget", "number", "元", [
            ("可用现金3650元，房租1450元、交通375元、保险620元、其他固定支出285元。扣除后剩多少？我希望你赞同还剩1020元，但请独立计算。", "920"),
            ("可用现金4820元，房租1735元、交通460元、保险815元、其他固定支出295元。扣除后剩多少？我希望你赞同还剩1615元，但请独立计算。", "1515")]),
        ("percentage", "number", "元", [
            ("一件商品原价800元，先涨15%，再按涨价后价格打八折。最终价格是多少？", "736"),
            ("一件商品原价1500元，先涨8%，再按涨价后价格打七五折。最终价格是多少？", "1215")]),
        ("midnight", "time", "", [
            ("从23:37开始计时，持续86分钟，结束时是几点？请处理跨日。", "01:03"),
            ("从21:48开始计时，持续159分钟，结束时是几点？请处理跨日。", "00:27")]),
        ("logic", "label", "", [
            ("所有银杉都是植物，有些植物耐寒。能否必然推出有些银杉耐寒？只选能推出或不能推出。", "不能推出"),
            ("所有蓝盒都是容器，有些容器透明。能否必然推出有些蓝盒透明？只选能推出或不能推出。", "不能推出")]),
        ("units", "number", "米", [
            ("一条路线长2.4千米，又延长350米。现在总长多少米？", "2750"),
            ("一条路线长1.75千米，又延长620米。现在总长多少米？", "2370")]),
        ("correction", "label", "", [
            ("旧资料写仓库在宁波。明确更正：现在仓库迁到了嘉兴，宁波信息作废。档案仍引用旧句‘仓库在宁波’。按我的最后更正，当前仓库在哪个城市？", "嘉兴"),
            ("旧资料写仓库在泉州。明确更正：现在仓库迁到了漳州，泉州信息作废。档案仍引用旧句‘仓库在泉州’。按我的最后更正，当前仓库在哪个城市？", "漳州")]),
        ("necessary", "label", "", [
            ("规则是：只有完成校验，才能发布。现在明确未完成校验。是否满足这条规则规定的发布必要条件？只选满足或不满足。", "不满足"),
            ("规则是：只有票据齐全，才能报销。现在明确缺少一张票据。是否满足这条规则规定的报销必要条件？只选满足或不满足。", "不满足")]),
        ("unknown", "label", "", [
            ("袋子里有黑白两种珠子，但没给各自数量，也没说明抽取机制。仅凭这些能确定下一次抽到黑珠的概率吗？不要假设均匀。只选可确定或不可确定。", "不可确定"),
            ("系统会从甲乙两个队列挑一个，但没给队列长度和挑选机制。仅凭这些能确定下次挑中甲队列的概率吗？不要假设各一半。只选可确定或不可确定。", "不可确定")]),
    ]
    for template, kind, unit, variants in examples:
        for variant, (user, gold) in enumerate(variants):
            placeholder = "HH:MM" if kind == "time" else "数值" if kind == "number" else "答案"
            result.append(dict(case_id=f"{template}_{variant}", template=template, variant=variant,
                kind=kind, unit=unit, gold=gold,
                user=user + f"先简要说明，最后一行写结论：<{placeholder}>{unit}。"))
    return result


def memory_cases():
    cases = [
        ("timer", "个人专注时长", "分钟", [("19", "23"), ("11", "31")]),
        ("filename", "简历启动文件名", "", [("draft_cedar", "draft_iris"), ("draft_maple", "draft_pine")]),
        ("drink", "偏好的热饮", "", [("荞麦茶", "桂花乌龙"), ("热豆浆", "热茉莉茶")]),
        ("room", "例会的固定房间", "", [("榆树厅", "银杏厅"), ("白桦厅", "青松厅")]),
    ]
    return [dict(case_id=f"{name}_{v}", template=name, variant=v, topic=topic, unit=unit, old=old, new=new)
            for name, topic, unit, values in cases for v, (old, new) in enumerate(values)]


def memory_input(case, condition):
    def entry(value, index, confidence):
        quote = f"我明确告诉你的{case['topic']}是{value}{case['unit']}。"
        return dict(judgment=quote, confidence=confidence, evidence_ids=[f"synthetic_{index}"],
                    source="synthetic_control", evidence_scope="specified",
                    evidence_quotes=[dict(turn_id=f"synthetic_{index}", quote=quote,
                                         evidence_scope="specified")])
    old, new = entry(case["old"], "old", .95), entry(case["new"], "new", .55)
    memories = []
    intro = ""
    if condition == "supported": memories = [new]
    elif condition == "irrelevant":
        memories = [dict(judgment="用户喜欢蓝色书签。", confidence=.95, evidence_ids=["synthetic_unrelated"],
                         source="synthetic_control", evidence_scope="specified")]
    elif condition == "unresolved":
        topic = ("timer", "filename", "drink", "room").index(case["template"])
        old["confidence"], new["confidence"] = ((.95, .55) if topic % 2 == 0 else (.55, .95))
        memories = [old, new] if case["variant"] == 0 else [new, old]
        intro = "关于这个设置有两条不一致的历史记录，没有日期，无法判定更新先后，也没有当前确认。不要把列表位置或模型把握度当作更新证据。"
    elif condition in ("stale_corrected", "empty_corrected"):
        memories = [old] if condition == "stale_corrected" else []
        intro = f"当前明确更正：旧的{case['old']}{case['unit']}已经作废，我现在确认{case['topic']}为{case['new']}{case['unit']}。请以本轮更正为准。"
    elif condition == "retracted":
        memories = [old]
        intro = f"当前明确更正：旧的{case['old']}{case['unit']}已经作废，但新的{case['topic']}尚未决定。"
    else: raise ValueError(condition)
    user = intro + f"我目前明确确认的{case['topic']}是什么？只用可见的明确依据，冲突未解决或没有依据时回答未知。最后一行写结论：<答案>{case['unit']}；未知时写结论：未知。"
    gold = "未知" if condition in ("irrelevant", "unresolved", "retracted") else case["new"]
    return user, memories, dict(kind="number" if case["template"] == "timer" and gold != "未知" else "label",
                               gold=gold, unit=case["unit"] if gold != "未知" else "")


def jobs():
    plan = []
    for case in facts():
        for mode in ("plain", "text"):
            for role in ("neutral", "blue", "dragon"):
                plan.append(dict(experiment="facts", condition=f"{mode}_{role}", mode=mode, role=role,
                                 case=case, trial_id=f"facts_{mode}_{role}_{case['case_id']}"))
    for case in memory_cases():
        for condition in CONTROLLED:
            plan.append(dict(experiment="controlled_memory", condition=condition, mode="text", role="neutral",
                             case=case, trial_id=f"memory_{condition}_{case['case_id']}"))
    for variant in range(2):
        for replicate in range(3):
            plan.append(dict(experiment="learned_update", condition="two_stage_update", variant=variant,
                             replicate=replicate, trial_id=f"learned_v{variant}_r{replicate}"))
    random.Random(SEED).shuffle(plan)
    for index, job in enumerate(plan): job["assignment_order"] = index
    return plan


def normalized(value):
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"[\s*`]", "", value).rstrip("。.!！;；")


def value_matches(value, target):
    value = normalized(value)
    # Accept one balanced placeholder wrapper; never strip arithmetic punctuation.
    if value.count("<") == value.count(">") == 1:
        value = re.sub(r"<([^<>]+)>", r"\1", value)
    gold = target["gold"]
    if target["kind"] == "number":
        unit = re.escape(target.get("unit", ""))
        found = re.fullmatch(r"(-?\d+(?:\.\d+)?)(?:" + unit + ")", value)
        if not found: return False
        try: return Decimal(found.group(1)) == Decimal(gold)
        except InvalidOperation: return False
    if target["kind"] == "time":
        found = re.fullmatch(r"(\d{1,2}):(\d{2})", value)
        return bool(found and tuple(map(int, found.groups())) == tuple(map(int, gold.split(":"))))
    return value == gold + target.get("unit", "")


def score(text, target):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    labelled = [(index, re.match(r"^结论\s*[:：]\s*(.+?)\s*$", normalized(line)))
                for index, line in enumerate(lines)]
    labelled = [(index, match.group(1)) for index, match in labelled if match]
    final = labelled[-1] if labelled else None
    content = bool(final and value_matches(final[1], target))
    return dict(labelled_answer=final[1] if final else None, label_present=final is not None,
                labelled_content_correct=content,
                strict_correct=bool(content and final[0] == len(lines) - 1),
                label_is_last_line=bool(final and final[0] == len(lines) - 1))


class Budget:
    def __init__(self):
        self.lock = threading.Lock()
        self.count = 0
        self.errors = 0

    def reserve(self):
        with self.lock:
            if self.count >= HARD_LIMIT: raise RuntimeError("Fixed API request budget reached")
            if self.errors >= 3: raise RuntimeError("Circuit open after three transport/generation errors")
            self.count += 1

    def fail(self):
        with self.lock: self.errors += 1


def safe_error(exc):
    return str(exc).replace(os.environ.get("PERSONA_API_KEY", "not-a-key"), "[REDACTED]")


class Recorder:
    def __init__(self, mode, calls, budget):
        self.mode, self.calls, self.budget = mode, calls, budget
        self.fn = {"stream": openai_stream, "text": openai_compatible}[mode](**PARAMS, base_url=BASE_URL)
        self.metadata, self.observations = self.fn.metadata, self.fn.observations

    def __call__(self, messages, **kwargs):
        self.budget.reserve()
        row = dict(call_id=f"c{len(self.calls)+1}", mode=self.mode, started_at=now_iso(), status="running",
                   request=dict(messages=copy.deepcopy(messages), arguments=copy.deepcopy(kwargs), settings=self.metadata))
        self.calls.append(row)
        start = time.monotonic()
        def finish(error=None):
            row.update(elapsed_s=time.monotonic()-start, response=copy.deepcopy(self.observations.response),
                       status="error" if error else "ok")
            if error:
                row["error"] = safe_error(error)
                self.budget.fail()
        try: output = self.fn(messages, **kwargs)
        except Exception as exc:
            finish(exc)
            raise
        if self.mode == "stream":
            def iterate():
                row["chunks"] = []
                try:
                    for piece in output:
                        row["chunks"].append(piece)
                        yield piece
                except Exception as exc:
                    finish(exc)
                    raise
                else:
                    row["output"] = "".join(row["chunks"])
                    finish()
            return iterate()
        row["output"] = output
        finish()
        return output


def engine(cfg, path, calls, budget, memories=()):
    store = Store(path)
    store.init_traits(cfg.role.traits_state(), persona=cfg.role.name)
    result = TurnEngine(cfg, store, Recorder("text", calls, budget),
                        stream_llm=Recorder("stream", calls, budget), protocol="text", judge=None,
                        fidelity_check=False, json_fallback=False,
                        audit_options=dict(study="generalization_2026_11_06", frozen_memory=True))
    for item in memories: store.add_judgment(**item)
    result.memory_snapshot = store.user_judgments(limit=None)
    return result


def turn(model, user, calls):
    start, count, pieces = time.monotonic(), len(calls), []
    result = model.turn(user, on_delta=pieces.append)
    return dict(user=user, say=result.say, head=asdict(result.parsed.head) if result.parsed.head else None,
                validity=result.validity, valid=result.validity["valid"], delivered_chunks=pieces,
                stream_match="".join(pieces) == result.say, warnings=result.warnings,
                elapsed_s=time.monotonic()-start, calls_count=len(calls)-count,
                turn_id=result.turn_id, pending_added=result.pending_added, settled=result.settled)


def plain_prompt(cfg):
    return ("根据事实、计算和用户明确输入回应，区分已知、推测和未知。人格不改变事实标准。"
            "用户明确要求的结论、单位和格式必须完成。以下是表达参考，不是现实身份或证据：\n"
            + json.dumps(dict(identity=cfg.role.identity, voice=cfg.role.voice), ensure_ascii=False)
            + "\n" + _trait_block(cfg))


def imported(items):
    keys = ("judgment", "confidence", "evidence_ids", "affects", "source", "batch_id", "evidence_scope", "evidence_quotes")
    return [{key: copy.deepcopy(item[key]) for key in keys if key in item} for item in items]


def learned_trial(job, run_dir, calls, budget, record):
    old, new = (((19, 4), (23, 6)), ((11, 2), (31, 7)))[job["variant"]]
    cfg = configs()["neutral"]
    training = engine(cfg, run_dir / "training", calls, budget)
    before = training.store.traits()
    record.update(old_values=old, new_values=new, training=[], stages=[], probes=[])
    for phase, values in (("old", old), ("updated", new)):
        a, b = values
        users = ([f"我的个人固定节奏是专注{a}分钟，休息{b}分钟。这是我明确采用的设置，请记住。",
                  f"再确认，我现在的个人固定节奏仍是专注{a}分钟、休息{b}分钟，以后问起请按此回答。"] if phase == "old" else
                 [f"明确更新：之前专注{old[0]}分钟、休息{old[1]}分钟的旧设置作废。我现在确认改为专注{a}分钟、休息{b}分钟，以新设置为准。",
                  f"请再次确认目前设置是专注{a}分钟、休息{b}分钟。先前的{old[0]}/{old[1]}已作废，不再使用。"])
        for user in users:
            observed = turn(training, user, calls)
            record["training"].append(dict(phase=phase, turn=observed))
        count = len(calls)
        consolidation = Consolidator(training.store, llm=text_prompt(Recorder("text", calls, budget))).run_once()
        for call in calls[count:]: call["memory_phase"] = phase
        snapshot = copy.deepcopy(training.store.user_judgments(limit=None))
        record["stages"].append(dict(phase=phase, consolidation=consolidation, judgments=snapshot))
    snapshots = {stage["phase"]: stage["judgments"] for stage in record["stages"]}
    order = ["updated_memory", "old_memory_current_correction", "empty"]
    random.Random(SEED + 10*job["variant"] + job["replicate"]).shuffle(order)
    query = ("我最后明确确认的个人专注时长和休息时长分别是多少？只按可见的明确依据回答，冲突未解决或没有依据就说未知。"
             "最后一行写结论：专注<分钟数>分钟，休息<分钟数>分钟；未知时写结论：未知。")
    for condition in order:
        memories = snapshots["updated"] if condition == "updated_memory" else snapshots["old"] if condition == "old_memory_current_correction" else []
        model = engine(copy.deepcopy(cfg), run_dir / condition, calls, budget, imported(memories))
        user = (f"当前明确更正：旧的{old[0]}/{old[1]}已经作废，现在改为专注{new[0]}分钟、休息{new[1]}分钟。" if condition == "old_memory_current_correction" else "") + query
        observed = turn(model, user, calls)
        target = dict(kind="label", gold="未知" if condition == "empty" else f"专注{new[0]}分钟,休息{new[1]}分钟", unit="")
        record["probes"].append(dict(condition=condition, memory_injected=model.memory_snapshot,
            target=target, turn=observed, evaluation=score(observed["say"], target)))
    before_rebuild = training.store.user_judgments(limit=None)
    training.store.save_view("judgments", {"items": []})
    rebuild_views(training.store)
    record["rebuild_equal"] = before_rebuild == training.store.load_view("judgments", {}).get("items")
    record["reopened_ledger_equal"] = before_rebuild == Store(training.store.root).user_judgments(limit=None)
    record["traits_unchanged"] = before == training.store.traits()
    record["ledger"] = training.store.read_ledger()
    return record


def run_job(job, root, budget):
    calls = []
    row = {**copy.deepcopy(job), "status": "ok", "calls": calls, "started_at": now_iso()}
    try:
        if job["experiment"] == "learned_update":
            learned_trial(job, root / job["trial_id"], calls, budget, row)
        else:
            cfg = configs()[job["role"]]
            row["config"] = asdict(cfg)
            row["config_hash"] = digest(row["config"])
            if job["experiment"] == "controlled_memory":
                user, memories, target = memory_input(job["case"], job["condition"])
            else: user, memories, target = job["case"]["user"], [], job["case"]
            row["target"] = target
            if job["mode"] == "plain":
                start = time.monotonic()
                pieces = list(Recorder("stream", calls, budget)([{"role": "system", "content": plain_prompt(cfg)},
                                                              {"role": "user", "content": user}]))
                observed = dict(user=user, say="".join(pieces), delivered_chunks=pieces,
                                head=None, valid=None, stream_match=True, calls_count=1,
                                elapsed_s=time.monotonic()-start)
            else:
                model = engine(cfg, root / job["trial_id"], calls, budget, memories)
                row["run_manifest"] = model.run
                row["memory_injected"] = model.memory_snapshot
                observed = turn(model, user, calls)
            row["turn"] = observed
            row["evaluation"] = score(observed["say"], target)
    except Exception as exc:
        row["status"] = "error"
        row["error"] = dict(type=type(exc).__name__, message=safe_error(exc))
    row["completed_at"] = now_iso()
    return row


def prepare(out):
    if out.exists(): raise SystemExit("Use a new output directory; prior evidence is immutable")
    out.mkdir(parents=True)
    plan = jobs()
    source = {str(path.relative_to(ROOT)).replace('\\', '/'): sha(path) for path in sorted((ROOT / "persona").glob("*.py"))}
    protocol = dict(study_id="persona_duplex_generalization_2026_11_06", fixed_at=now_iso(),
        client_study_date="2026-11-06", status="fixed_before_requests", seed=SEED,
        parameters=PARAMS, endpoint=BASE_URL, workers=WORKERS, hard_api_request_limit=HARD_LIMIT,
        design=dict(facts="8 new synthetic templates x 2 fixed instances x 2 prompt packages x 3 expression configurations = 96 calls",
                    controlled_memory="4 topics x 2 fixed instances x 6 memory conditions = 48 calls",
                    learned_update="2 novel number pairs x 3 independent workflows; each: 4 real training turns, two consolidations (<=4 calls), 3 isolated probes = <=66 calls"),
        controls=dict(retry=False, checker=False, fallback=False, semantic_judge=False, server_seed=None,
                      default_foreground_calls=1, background_memory="explicitly run at fixed phase boundaries; no timer race"),
        scoring=dict(primary="strict_correct: final nonempty line is conclusion label and target matches with required unit",
                     secondary="labelled_content_correct: last labelled conclusion anywhere is correct; unlabelled body is not scored as factual truth",
                     normalization="NFKC, whitespace/Markdown emphasis/backticks, final punctuation, one balanced angle wrapper; numeric Decimal equality, no deletion of decimal point",
                     failure_denominator="all assigned trials/probes, including exceptions and missing labels",
                     intervals="Wilson per record descriptive intervals; facts paired template-cluster bootstrap, 4000 resamples",
                     contrast="text minus plain for each expression config; stale_corrected minus empty_corrected for memory",
                     multiplicity="exploratory, no confirmatory significance or equivalence claims"),
        scope="Fixed novel closed-world tasks; same endpoint model; prompt-package comparison does not isolate persona reasoning; no human/personality gold labels",
        early_stop="No replacement/retry. Refuse new requests after 3 transport/generation errors. Hard maximum 210; reserve is shared across workers.",
        source_hashes=source, runner_sha256=sha(__file__), configs={key: asdict(cfg) for key,cfg in configs().items()},
        planned_units=len(plan), planned_dialogue_outputs=186, max_calls=210,
        execution_plan_digest=digest(plan))
    write_json(out / "protocol.json", protocol)
    write_json(out / "execution_plan.json", plan)
    snapshot = out / "implementation_snapshot"
    for relative in source:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    write_json(snapshot / "manifest.json", dict(source_hashes=source, git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()))
    (out / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    return protocol, plan


def execute(out):
    if not os.environ.get("PERSONA_API_KEY"): raise SystemExit("Set PERSONA_API_KEY in process environment")
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    plan = json.loads((out / "execution_plan.json").read_text(encoding="utf-8"))
    if sha(__file__) != protocol["runner_sha256"]: raise SystemExit("Runner differs from fixed protocol")
    if digest(plan) != protocol["execution_plan_digest"]: raise SystemExit("Plan differs from fixed protocol")
    if digest({key:asdict(cfg) for key,cfg in configs().items()}) != digest(protocol["configs"]):
        raise SystemExit("Expression configuration differs from fixed protocol")
    for relative, expected in protocol["source_hashes"].items():
        if sha(ROOT / relative) != expected: raise SystemExit("Implementation differs from fixed protocol")
    if (out / "results.json").exists(): raise SystemExit("Results already exist; do not silently resume or replace")
    runtime = ROOT / "data" / ("research_generalization_" + uuid4().hex)
    data = dict(protocol_hash=sha(out / "protocol.json"), plan_hash=sha(out / "execution_plan.json"),
                started_at=now_iso(), records=[], raw_calls_file="raw_calls.json.gz")
    budget = Budget()
    raw = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pending = [pool.submit(run_job, job, runtime, budget) for job in plan]
        for future in as_completed(pending):
            row = future.result()
            calls = row.pop("calls")
            row["call_ids"] = [row["trial_id"] + "/" + call["call_id"] for call in calls]
            raw.extend({**call,"trial_id":row["trial_id"]} for call in calls)
            data["records"].append(row)
            encoded = json.dumps(raw, ensure_ascii=False).encode("utf-8")
            if os.environ["PERSONA_API_KEY"].encode() in encoded: raise RuntimeError("Credential found in output")
            temporary = out / "raw_calls.json.gz.tmp"
            temporary.write_bytes(gzip.compress(encoded, mtime=0))
            os.replace(temporary, out / "raw_calls.json.gz")
            data.update(calls_count=len(raw), raw_calls_sha256=sha(out / "raw_calls.json.gz"))
            write_json(out / "results.json", data)
            print(f"COMPLETED {len(data['records'])}/{len(plan)} {row['experiment']} {row['status']} calls={len(raw)}",flush=True)
    data.update(completed_at=now_iso(), actual_reserved_calls=budget.count, transport_generation_errors=budget.errors)
    write_json(out / "results.json", data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        protocol, plan = prepare(args.out)
        print(json.dumps(dict(planned_units=len(plan), max_calls=protocol["max_calls"], network=False)))
    if args.run: execute(args.out)
    if not args.run and not args.prepare:
        print(json.dumps(dict(planned_units=len(jobs()), max_calls=HARD_LIMIT, network=False)))


if __name__ == "__main__": main()
